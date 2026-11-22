"""E2E对话 · 流式播报（LLM 句子 → TTS 音频）

把 LLM 逐句产出实时转成音频流：一边收句子一边合成、一边下发，
是 600ms 首响预算的关键路径，也是打断最频繁的地方。
"""

import asyncio
import logging
import time

import numpy as np

from .state import State

log = logging.getLogger("Voice Agent.cm")


class StreamingMixin:
    """E2E对话 · 流式播报（LLM 句子 → TTS 音频）。由 ConversationManager 组合，`self` 上共享整机会话状态。"""

    async def _stream_llm_tts(self, user_text, rag_context, turn, t_start, asr_ms, rag_ms):
        """整条链路的"流式"核心：LLM 每凑齐一句就立刻送 TTS，TTS 每生成一块就立刻下发（首句优先出声）。

        副作用：在线程池跑 `llm.stream_sentences`（生产者线程）与 `tts.synthesize_stream`；下发
        llm_sentence/metrics/audio_start/裸音频/llm 消息；被打断时把已生成的部分回复写进 `_interrupted_texts`
        供下轮引用；finally 里回收状态为 IDLE、必要时发 session_end。
        不向外抛异常：LLM 线程的异常通过队列里的 `{"_error": ...}` 标记回传后在此抛出并处理。
        """
        loop = asyncio.get_event_loop()
        # 每个回复轮都重置打断诊断计数器：manager 的 SPEAKING 分支只打印前 5 帧
        # 以及"高 VAD 且每 10 帧"的帧，计数器不归零的话第二轮之后就再也不会输出，
        # 恰好把"为什么这次没触发打断"最需要看的信息丢掉。
        # （系统话术 _speak_system 与文本路径 _run_pipeline_from_text 都已各自重置）
        self._speak_log_count = 0
        # 有界句子队列（maxsize=8，v3.0 G7 修复）：无界队列会让 LLM 生产远超消费速度、句子在内存里堆积；
        # 有界队列让生产者线程在 put 时阻塞，形成"线程侧背压"，TTS 消费多快就生成多快
        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        full_response = ""
        sentence_count = 0
        first_ttfa = None
        llm_first_ms = 0.0
        was_interrupted = False

        def _produce_sentences():
            """生产者：在线程池线程里跑阻塞的 LLM 流式生成，把句子投进队列。

            这个函数**不在事件循环线程**里：不能直接 await，只能用 run_coroutine_threadsafe 投递协程；
            判断打断用 `self._cancel_speaking.is_set()`（Event 读操作线程安全）。副作用：写 LLM 对话历史、
            投递句子 dict 与异常标记 `{"_error": ...}`。finally 保证投一个 None 哨兵，否则消费端要等到 35s 超时。
            """
            try:
                for s in self.llm.stream_sentences(user_text, rag_context=rag_context):
                    # 用户已打断：不再让 LLM 继续产出（尽早省 GPU，也避免污染历史）
                    if self._cancel_speaking.is_set():
                        break
                    fut = asyncio.run_coroutine_threadsafe(sentence_queue.put(s), loop)
                    # 最多等 30s：队列被下游卡死时不能让线程永远阻塞
                    fut.result(timeout=30)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    sentence_queue.put({"_error": str(e)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(sentence_queue.put(None), loop)

        def _tts_stream_sentence(text):
            """把单句文本合成 TTS 音频块，返回列表（给 run_in_executor 的同步目标）。

            返回 `[{"audio": np.ndarray(float32, 24kHz), "chunk_idx", "ttfa_ms"}, ...]`；
            `ttfa_ms` 挂在首段上。list(...) 一次性抽干生成器，同步遍历最省切换。
            """
            return list(self.tts.synthesize_stream(text))

        producer_future = loop.run_in_executor(None, _produce_sentences)

        try:
            while True:
                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    log.info("[Turn %d] Cancelled before next sentence", turn)
                    break

                # 等下一句：35s 是"LLM 彻底卡死"的兜底（正常 SSE 间隔远小于此）
                try:
                    s = await asyncio.wait_for(sentence_queue.get(), timeout=35.0)
                except asyncio.TimeoutError:
                    log.error("[Turn %d] LLM sentence queue timeout (35s)", turn)
                    self.metrics.record_error()
                    break

                if s is None:
                    break
                if "_error" in s:
                    raise RuntimeError(f"LLM stream error: {s['_error']}")

                # 取到句子后再查一次打断：等待期间刚被打断就不要发这一句（避免"打断后又冒出一句"）
                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    break

                sentence = s["sentence"]
                full_response += sentence
                sentence_count += 1
                # 只有第一句带真实 ttfs_ms（后续句子的 ttfs_ms 在 llm.py 里就是 0）
                if sentence_count == 1:
                    llm_first_ms = s["ttfs_ms"]

                await self._send({
                    "type": "llm_sentence", "text": sentence,
                    "latency_ms": round(s["ttfs_ms"], 1) if sentence_count == 1 else 0,
                    "sentence_idx": sentence_count - 1, "turn": turn,
                })

                # 合成前的第三次打断检查：把一次没必要的 TTS 合成省掉（GPU 最贵的一步）
                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    break

                tts_chunks = await loop.run_in_executor(None, _tts_stream_sentence, sentence)

                for i, chunk_data in enumerate(tts_chunks):
                    # 一次合成可能产出几十块，逐块检查才能"按毫秒级闭嘴"
                    if self._cancel_speaking.is_set():
                        was_interrupted = True
                        log.info(
                            "[Turn %d] Cancelled mid-TTS chunk %d/%d", turn, i, len(tts_chunks)
                        )
                        break

                    # 第一句的第一块：这是"AI 真正开口"的瞬间，首响指标在这里落账
                    if sentence_count == 1 and i == 0:
                        first_ttfa = chunk_data.get("ttfa_ms", 0)
                        # audio_start 让前端把 playableTurn 指向本 turn；在此之前收到的音频帧会被忽略
                        await self._send({"type": "audio_start", "turn": turn})
                        # 端到端首响 = ASR + RAG + LLM 首句 + TTS 首帧（RAG 与 LLM 实为串行，这里只是相加用于观测）
                        first_response_ms = asr_ms + rag_ms + llm_first_ms + first_ttfa
                        self.metrics.record_turn(
                            asr_ms=asr_ms, llm_ms=llm_first_ms,
                            tts_ms=first_ttfa, fr_ms=first_response_ms,
                        )
                        await self._send({
                            "type": "metrics",
                            "asr_ms": round(asr_ms, 1),
                            "rag_ms": round(rag_ms, 1),
                            "llm_ms": round(llm_first_ms, 1),
                            "tts_ttfa_ms": round(first_ttfa, 1),
                            "first_response_ms": round(first_response_ms, 1),
                            "turn": turn,
                        })

                    audio = chunk_data["audio"]
                    if audio.dtype != np.int16:
                        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                    raw = audio.tobytes()
                    # 单块硬上限保护：TTS 生成退化时一块可能巨大、会撑爆前端缓冲；截断比丢弃安全（丢块会造成爆破音）
                    if len(raw) > self.MAX_TTS_AUDIO_BYTES:
                        log.warning("[Turn %d] TTS chunk oversized (%d bytes) — truncating",
                                    turn, len(raw))
                        raw = raw[:self.MAX_TTS_AUDIO_BYTES]
                    await self._send({"_audio": raw})
                    await asyncio.sleep(0.01)

                if sentence_count >= 3:
                    log.info("[Turn %d] Sentence cap (3) reached, stopping", turn)
                    break

            await self._send({
                "type": "llm", "text": full_response,
                "turn": turn, "sentences": sentence_count,
            })

        except asyncio.CancelledError:
            was_interrupted = True
            log.info("[Turn %d] Pipeline cancelled", turn)
        except Exception as e:
            # 注意这里**不**播道歉：外层 _run_pipeline 负责，避免重复播
            log.error("[Turn %d] LLM/TTS error: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            await self._send({"type": "error", "message": str(e), "turn": turn})
        finally:
            # 收尾生产者线程：最多等 2s，超时/异常都忽略（线程池的残留工作不影响状态机正确性）
            try:
                await asyncio.wait_for(asyncio.wrap_future(producer_future), timeout=2.0)
            except Exception:
                pass

            # 被打断且已说出内容：存起来给下一轮当上下文，下一句才是"接着上句说"而不是重头再来
            if was_interrupted and full_response:
                self._interrupted_texts.append(full_response)
                log.info("[Turn %d] Saved partial AI response (%d chars) for context",
                         turn, len(full_response))

            # 只在仍处于 SPEAKING 时回收：已被打断转入 INTERRUPTED/LISTENING 时必须保留那个状态
            if self.state == State.SPEAKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                await self._send({"type": "state", "state": "idle"})
            if self._session_ending:
                self._session_ending = False
                await self._send({"type": "session_end", "reason": "farewell"})
