"""E2E对话 · ASR 文本累积器

极短 ASR 结果（≤4 字，如“嗯”“那个”）先缓存，等后续语音拼成整句再送 LLM，
避免把一句话切成两轮对话；超时或攒够段数强制下发。
"""

import asyncio
import logging

from .constants import (
    ASR_ACCUM_FLUSH_S,
    ASR_ACCUM_MAX_CHARS,
)

from .state import State

log = logging.getLogger("Voice Agent.cm")


class AsrAccumulator:
    """E2E对话 · ASR 文本累积器。由 ConversationManager 组合，`self` 上共享整机会话状态。"""

    def _cancel_accum_flush(self):
        """取消并清空累积器超时定时器（幂等）。"""
        # 只取消还在跑的任务；已完成的不能再 cancel（会抛 InvalidStateError）
        if self._asr_accum_flush_task and not self._asr_accum_flush_task.done():
            self._asr_accum_flush_task.cancel()
        self._asr_accum_flush_task = None


    def _is_short_utterance(self, text: str) -> bool:
        """文本是否短到需要先缓存、等后续语音拼成整句（≤ ASR_ACCUM_MAX_CHARS）。"""
        # 纯按字符数判断（中文一字≈一词），不做分词：便宜且不需要额外依赖
        return len(text) <= ASR_ACCUM_MAX_CHARS


    def _drain_accum(self, new_text: str = "") -> str:
        """取出并清空累积器，把待合并文本拼成一句可直接送 LLM 的话（中文逗号连接）。"""
        parts = self._asr_accum_texts + ([new_text] if new_text else [])
        self._asr_accum_texts = []
        # 内容已经取出，兜底定时器必须取消，否则 3s 后会把同一段文本再发一遍
        self._cancel_accum_flush()
        return "，".join(parts)


    async def _accum_flush_timer(self):
        """累积器超时兜底：等 ASR_ACCUM_FLUSH_S(3s) 后缓存非空就把它当一轮输入，置 THINKING 并起后台 pipeline。

        只走文本路径（无音频，跳过 ASR）；被取消（用户又说话了）时静默退出。
        """
        try:
            await asyncio.sleep(ASR_ACCUM_FLUSH_S)
            # 醒来后二次确认：缓存非空（确实没人接着说）且仍在听（没被别的路径抢先处理）
            if self._asr_accum_texts and self.state == State.LISTENING:
                flushed = self._drain_accum()
                log.info("[Accum] Flush timer fired, sending buffered: '%s'", flushed)
                self.state = State.THINKING
                self._turn += 1
                await self._send({"type": "state", "state": "thinking"})
                # latency_ms=0（没跑 ASR）；accumulated=True 让前端/日志区分"兜底合并"与正常转写
                await self._send({"type": "asr", "text": flushed,
                                  "latency_ms": 0, "turn": self._turn, "accumulated": True})
                self._track_task(
                    self._run_pipeline_from_text(flushed, self._turn),
                    name=f"accum-pipeline-{self._turn}",
                )
        except asyncio.CancelledError:
            # 被 _cancel_accum_flush() 取消是设计内的正常路径（用户接着说或已完成合并）
            pass
