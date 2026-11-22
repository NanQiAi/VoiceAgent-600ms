"""E2E对话 · 会话生命周期

会话级的入口与计时：主动问候（D1）、空闲超时追赶（D2）。
（文本输入通道已移除——只保留全双工语音。）
"""

import logging
import time

from .state import State

log = logging.getLogger("Voice Agent.cm")


class Session:
    """E2E对话 · 会话生命周期。由 ConversationManager 组合，`self` 上共享整机会话状态。"""

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
