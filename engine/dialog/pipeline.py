"""E2E对话 · 级联链路编排

一轮对话的主流程：ASR → RAG → LLM → TTS，以及错误兜底（D3）与告别识别（D4）。
只做编排，具体推理都委托给注入的引擎。
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from .constants import (
    ASR_ACCUM_MAX_SEGMENTS,
    CHUNK_SAMPLES,
    _ASR_TAG_RE,
)
from .state import State

log = logging.getLogger("Voice Agent.cm")


class Pipeline:
    """E2E对话 · 级联链路编排。由 ConversationManager 组合，`self` 上共享整机会话状态。"""

    async def _run_pipeline(self, audio: np.ndarray, turn: int,
                            spec_result: Optional[dict], spec_audio_len: int,
                            spec_task: Optional[asyncio.Task]):
        """一轮完整链路的崩溃安全外壳：ASR →（可选中段合并）→ RAG → LLM → TTS。

        所有异常都在这里兜住，保证任何组件挂掉时用户听到的是一句道歉而不是静默；CancelledError
        （断连/被取消）只记日志。`spec_task` 会在结果还没好时最多再等 200ms。finally 里若仍停在
        THINKING 则回收为 IDLE 并重置空闲计时。
        """
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        try:
            await self._run_pipeline_inner(
                audio, turn, spec_result, spec_audio_len, spec_task,
                loop, t_start,
            )
        except asyncio.CancelledError:
            log.info("[Turn %d] Pipeline cancelled (disconnect?)", turn)
        except Exception as e:
            log.error("[Turn %d] Pipeline CRASHED: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            try:
                await self._send({"type": "error", "message": str(e), "turn": turn})
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._play_system_tts(self._error_apology, turn)
            except Exception:
                pass
        finally:
            # 只有还停在 THINKING（即 inner 里没走到 SPEAKING）才需要回收状态：
            # 关键约束——**不能**无条件把 SPEAKING 改回 IDLE，否则 _stream_llm_tts
            # 还在播报时状态就被提前清掉，导致打断判定失效/状态竞争
            if self.state == State.THINKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                try:
                    await self._send({"type": "state", "state": "idle"})
                except Exception:
                    pass


    async def _run_pipeline_from_text(self, user_text: str, turn: int):
        """文本路径的轻量 pipeline（累积器兜底用）：没有音频，跳过 ASR，只做 RAG → LLM → TTS。

        命中告别词则置 `_session_ending`；finally 里若仍停在 THINKING 则回收 IDLE；不向外抛异常。
        """
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()
        try:
            self._consecutive_empty_asr = 0

            if any(kw in user_text for kw in self._farewell_keywords):
                self._session_ending = True

            rag_context = ""
            try:
                rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
                rag_context = rag_result["context"]
            except Exception:
                pass

            self._cancel_speaking.clear()
            self._barge_confirm_count = 0
            self._speak_log_count = 0
            self.state = State.SPEAKING
            await self._send({"type": "state", "state": "speaking"})
            await self._stream_llm_tts(user_text, rag_context, turn, t_start, 0, 0)
        except asyncio.CancelledError:
            log.info("[Turn %d] Accum pipeline cancelled", turn)
        except Exception as e:
            log.error("[Turn %d] Accum pipeline CRASHED: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            try:
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._play_system_tts(self._error_apology, turn)
            except Exception:
                pass
        finally:
            if self.state == State.THINKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                try:
                    await self._send({"type": "state", "state": "idle"})
                except Exception:
                    pass


    async def _run_pipeline_inner(self, audio, turn, spec_result, spec_audio_len,
                                  spec_task, loop, t_start):
        """真正的链路实现，按编号 0~6 顺序执行（保留原编号便于对照日志）：

        1 ASR（可复用投机结果）→ 1c 合并 THINKING 后的后到语音 → ASR 累积器 → D3 空结果 →
        D4 告别 → 2 打断文本聚合 → 3 RAG → 4 LLM 流式 → TTS。
        **不吞异常**：任何失败都向上抛给 `_run_pipeline` 的兜底逻辑（让崩溃处理只有一处）。
        """
        # 音频是否"显著变长"（>15%）：投机结果是在静音刚满 160ms 时跑的，用户之后又说了不少内容就必须重跑；spec_audio_len=0 时强制重跑
        audio_grew = len(audio) > spec_audio_len * 1.15 if spec_audio_len > 0 else True

        # 投机任务可能还在跑：最多再等 200ms（用户已等了一整段静音，不能再赌太久），超时/被取消就走正常 ASR
        if spec_task and not spec_task.done():
            try:
                await asyncio.wait_for(spec_task, timeout=0.2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                spec_result = None

        if spec_result and not audio_grew:
            user_text = _ASR_TAG_RE.sub("", spec_result["text"]).strip()
            asr_ms = spec_result["latency_ms"]
            log.info("[Turn %d] Used speculative ASR (saved ~%dms)", turn, round(asr_ms))
            # 计入首响的 ASR 时间记为 0：这段耗时发生在判停之前，不在用户感知的"说完→出声"区间内
            asr_ms = 0.0
        else:
            try:
                asr_result = await loop.run_in_executor(None, self.asr.transcribe, audio)
                user_text = _ASR_TAG_RE.sub("", asr_result["text"]).strip()
                asr_ms = asr_result["latency_ms"]
            except Exception as e:
                # ASR 引擎整个挂了（而非只返回空文本）：本轮无法继续，用"没太听清"话术让用户重说
                log.error("[Turn %d] ASR failed: %s", turn, e)
                self.metrics.record_error()
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._send(
                    {"type": "system_message", "text": self._asr_empty_text, "turn": turn}
                )
                await self._play_system_tts(self._asr_empty_text, turn)
                return

        # THINKING 期间接续说的话单独转写后拼进同一句，这样"我想问…那个…价格"不会被拆成两轮
        if self._thinking_has_speech and self._thinking_extra_buffer:
            extra_audio = np.concatenate(self._thinking_extra_buffer)
            self._thinking_extra_buffer = []
            self._thinking_has_speech = False
            if len(extra_audio) > CHUNK_SAMPLES * 3:
                try:
                    extra_result = await loop.run_in_executor(
                        None, self.asr.transcribe, extra_audio
                    )
                    extra_text = _ASR_TAG_RE.sub("", extra_result["text"]).strip()
                    if extra_text and len(extra_text) >= 2:
                        user_text = f"{user_text}，{extra_text}" if user_text else extra_text
                        asr_ms += extra_result["latency_ms"]
                        log.info("[Turn %d] Merged late speech: '%s'", turn, extra_text)
                except Exception as e:
                    # 补充转写失败只告警：主文本已拿到，不影响本轮对话
                    log.warning("[Turn %d] Late ASR merge failed: %s", turn, e)

        await self._send(
            {"type": "asr", "text": user_text, "latency_ms": round(asr_ms, 1), "turn": turn}
        )

        too_short = user_text and self._is_short_utterance(user_text)
        if too_short and len(self._asr_accum_texts) < ASR_ACCUM_MAX_SEGMENTS:
            self._asr_accum_texts.append(user_text)
            log.info("[Turn %d] ASR buffered ('%s', total=%d), returning to LISTENING",
                     turn, user_text, len(self._asr_accum_texts))
            self.state = State.LISTENING
            self._audio_buffer = []
            self._silence_count = 0
            await self._send({"type": "state", "state": "listening"})
            self._cancel_accum_flush()
            self._asr_accum_flush_task = self._track_task(
                self._accum_flush_timer(), name=f"accum-flush-{turn}")
            return

        if self._asr_accum_texts:
            prefix = "，".join(self._asr_accum_texts)
            user_text = f"{prefix}，{user_text}" if user_text else prefix
            self._asr_accum_texts = []
            # 已合并下发，定时器必须取消，否则 3s 后同一段文本会被再发一轮
            self._cancel_accum_flush()
            log.info("[Turn %d] Merged accumulated ASR: '%s'", turn, user_text)

        # 空结果恢复：前几次静默回 LISTENING（用户可能只是清嗓子），连续达 _max_empty_asr(3) 次才出声提示，再空则建议打字
        if not user_text or len(user_text) < 2:
            self._consecutive_empty_asr += 1
            if self._consecutive_empty_asr >= self._max_empty_asr:
                recovery = self._asr_noisy_text
                log.info(
                    "[Turn %d] ASR empty (#%d), recovery: '%s'",
                    turn, self._consecutive_empty_asr, recovery,
                )
                self._cancel_speaking.clear()
                self._barge_confirm_count = 0
                self._speak_log_count = 0
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._send({"type": "system_message", "text": recovery, "turn": turn})
                await self._play_system_tts(recovery, turn)
            else:
                log.info("[Turn %d] ASR too short ('%s', #%d), silent return to LISTENING",
                         turn, user_text or "", self._consecutive_empty_asr)
                self.state = State.LISTENING
                self._audio_buffer = []
                self._silence_count = 0
                await self._send({"type": "state", "state": "listening"})
            # 两条分支都到此结束本轮（finally 里 state 是 LISTENING 或 SPEAKING，不回收）
            return

        self._consecutive_empty_asr = 0

        # 命中告别词：标记本轮播完后结束会话（session_end 在 _stream_llm_tts 的 finally 里发）
        if any(kw in user_text for kw in self._farewell_keywords):
            self._session_ending = True
            log.info("[Turn %d] Farewell detected in: '%s'", turn, user_text)

        # 把被打断时存下的 AI 半句与用户新话一起交给 LLM，并明确告知"用户依次说了…"，让模型综合回应而非重复被自己打断的话
        if self._interrupted_texts:
            combined = "；".join(self._interrupted_texts + [user_text])
            user_text = f"用户依次说了：{combined}。请自然地综合回应。"
            self._interrupted_texts = []
        else:
            # asr_ms > 0 表示跑了真实 ASR；此处清空是幂等的防御性写法，保证没有残留的"被打断文本"进入下一轮
            if asr_ms > 0:
                self._interrupted_texts = []

        rag_context = ""
        rag_ms = 0.0
        try:
            rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]
        except Exception as e:
            log.warning("[Turn %d] RAG failed (degraded to no-context): %s", turn, e)
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": turn,
        })

        llm_user_text = user_text

        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        # 关键约束：播报任务挂到 _speaking_task（供 reset 取消），并且**必须 await 它**——
        # 否则 _run_pipeline 会提前走完 finally，在 TTS 还在播时就把状态机复位
        self._speaking_task = self._track_task(
            self._stream_llm_tts(llm_user_text, rag_context, turn, t_start, asr_ms, rag_ms),
            name=f"llm-tts-{turn}",
        )
        await self._speaking_task
