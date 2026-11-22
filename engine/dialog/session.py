"""E2E对话 · 会话生命周期

会话级的入口与计时：主动问候（D1）、空闲超时追赶（D2）、
以及前端打字输入。
"""

import asyncio
import logging
import time

from .state import State

log = logging.getLogger("Voice Agent.cm")


class Session:
    """E2E对话 · 会话生命周期。由 ConversationManager 组合，`self` 上共享整机会话状态。"""

    async def handle_text_input(self, text: str):
        """处理前端打字输入（协议 `text_input`）：跳过 VAD/ASR，直接走 RAG → LLM → TTS。

        副作用：自增 `_turn`、状态置 THINKING→SPEAKING；下发 state/asr/rag；线程池跑 RAG；起
        `text-tts-{turn}` 后台任务播报且**不 await**（ws_server 的消息循环要立刻回来继续收消息，用户打字时
        也可能按打断），代价是返回时播报还在进行，状态由任务自己回收。G2 修复：RAG 失败降级为无上下文。
        """
        self._turn += 1
        self.state = State.THINKING
        await self._send({"type": "state", "state": "thinking"})
        await self._send(
            {"type": "asr", "text": text, "latency_ms": 0, "turn": self._turn, "source": "text"}
        )

        self._consecutive_empty_asr = 0

        # D4：打字说"再见"也要能正常告别（否则用户只能靠说话结束通话）
        if any(kw in text for kw in self._farewell_keywords):
            self._session_ending = True
            log.info("[Turn %d] Farewell detected (text): '%s'", self._turn, text)

        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        rag_context = ""
        rag_ms = 0.0
        try:
            rag_result = await loop.run_in_executor(None, self.rag.get_context, text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]
        except Exception as e:
            log.warning("[Turn %d] RAG failed in text_input (degraded): %s", self._turn, e)
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": self._turn,
        })

        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = self._track_task(
            self._stream_llm_tts(text, rag_context, self._turn, t_start, 0, rag_ms),
            name=f"text-tts-{self._turn}",
        )


    async def start_greeting(self):
        """D1 开场白：连接建立后由 ws_server 以任务方式启动，主动播报一句问候（期间可被打断）。

        未配置开场白（空字符串）时只重置空闲计时基准并返回；否则自增 turn、置 SPEAKING、下发
        state/system_message，并同步 await `_play_system_tts` 播完（本方法通常作为后台任务运行）。
        """
        if not self._greeting_text:
            self._idle_since = time.monotonic()
            return
        log.info("Playing greeting: '%s'", self._greeting_text)
        # 开播前置：清打断闸门 + 清连击 + 清 SPEAKING 日志计数（让前 5 帧的打断判定日志重新打印）
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self._speak_log_count = 0
        self.state = State.SPEAKING
        self._turn += 1
        await self._send({"type": "state", "state": "speaking"})
        await self._send(
            {"type": "system_message", "text": self._greeting_text, "turn": self._turn}
        )
        await self._play_system_tts(self._greeting_text, self._turn)


    async def _check_idle_timeout(self):
        """D2 空闲超时检查：先提示"您还在吗"，再超时则告别并结束会话。

        由 `_process_chunk` 在 IDLE 且无语音时每约 1 秒调用一次（搭在音频帧流上的轮询，不是定时器）。
        第一次到 `_idle_timeout_s`(15s) 播提示；已提示过且到 `_idle_goodbye_s`(30s) 则播告别并带
        session_end_reason="idle_timeout"。两个阈值都以 `_idle_since` 为基准的绝对时间。
        """
        elapsed = time.monotonic() - self._idle_since
        if not self._idle_prompted and elapsed >= self._idle_timeout_s:
            self._idle_prompted = True
            log.info("Idle timeout (%.0fs), prompting user", elapsed)
            await self._speak_system(self._idle_prompt_text)
        elif (
            self._idle_prompted
            and not self._idle_goodbye_sent
            and elapsed >= self._idle_goodbye_s
        ):
            self._idle_goodbye_sent = True
            self._session_ending = True
            log.info("Idle goodbye (%.0fs), ending session", elapsed)
            await self._speak_system(self._idle_goodbye_text, session_end_reason="idle_timeout")
