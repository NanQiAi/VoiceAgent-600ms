"""E2E对话 · 下行发送

所有“把消息/音频发到前端”的动作集中在这里：系统话术、TTS 音频。
每条下行都带 turn 序列号，前端靠它丢弃旧轮次的迟到音频。
"""

import asyncio
import logging
import time

import numpy as np

from .constants import (
    TTS_SEND_CHUNK_BYTES,
)
from .state import State

log = logging.getLogger("Voice Agent.cm")


class Transport:
    """E2E对话 · 下行发送。由 ConversationManager 组合，`self` 上共享整机会话状态。"""

    async def _send_audio(self, audio_bytes: bytes):
        """把一段完整音频分片下发（每片 TTS_SEND_CHUNK_BYTES≈200ms），分片间检查打断。

        超长先截断到 MAX_TTS_AUDIO_BYTES 并告警；每片 `await asyncio.sleep(0)` 让出事件循环。
        注意：本方法在当前代码里**没有调用点**（`_stream_llm_tts` 与 `_play_system_tts` 都自己逐块发）。
        """
        if len(audio_bytes) > self.MAX_TTS_AUDIO_BYTES:
            log.warning("Audio too large (%d bytes, %.1fs) — truncating to 30s",
                        len(audio_bytes), len(audio_bytes) / 24000 / 2)
            audio_bytes = audio_bytes[:self.MAX_TTS_AUDIO_BYTES]
        for i in range(0, len(audio_bytes), TTS_SEND_CHUNK_BYTES):
            # 每片前查打断：置位就立刻停止后续下发（前端 barge_in 消息会清它的播放队列）
            if self._cancel_speaking.is_set():
                return
            await self._send({"_audio": audio_bytes[i:i + TTS_SEND_CHUNK_BYTES]})
            # sleep(0) 是"最小让出"：把控制权交给事件循环处理其它就绪回调（如收音频帧）
            await asyncio.sleep(0)


    async def _play_system_tts(self, text: str, turn: int, session_end_reason: str = None):
        """系统话术（D1 开场白 / D2 超时 / D3 错误恢复 / 崩溃道歉）的统一播报实现。

        与 `_stream_llm_tts` 的区别：文本既定、不需要 LLM，整个合成是一次阻塞调用，拿到全部块后再逐块下发。
        副作用：线程池跑完整 TTS 合成；下发 audio_start 与裸音频；finally 里从 SPEAKING 回收为 IDLE、
        重置空闲计时、可能发 session_end。CancelledError 静默吞掉，其它异常只记日志。
        注意 finally 不区分是否被打断：被打断时 `_cancel_speaking` 已置位，因此**不**发 session_end。
        """
        loop = asyncio.get_event_loop()
        try:
            def _gen():
                """在线程池线程里把整段文本合成完并返回音频块列表。"""
                return list(self.tts.synthesize_stream(text))

            tts_chunks = await loop.run_in_executor(None, _gen)
            await self._send({"type": "audio_start", "turn": turn})

            for i, chunk_data in enumerate(tts_chunks):
                if self._cancel_speaking.is_set():
                    log.info("[Turn %d] System message interrupted at chunk %d", turn, i)
                    break
                audio = chunk_data["audio"]
                if audio.dtype != np.int16:
                    audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                await self._send({"_audio": audio.tobytes()})
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("[Turn %d] System TTS error: %s", turn, e)
        finally:
            if self.state == State.SPEAKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                await self._send({"type": "state", "state": "idle"})
            # 有结束原因且**没有**被打断时才发 session_end：用户打断一句告别说明他还想继续聊
            if session_end_reason and not self._cancel_speaking.is_set():
                await self._send({"type": "session_end", "reason": session_end_reason})


    async def _speak_system(self, text: str, session_end_reason: str = None):
        """下发一条系统话术并把播报挂成后台任务（不 await，调用方立即返回）。

        `session_end_reason` 非 None 时，播完后由 `_play_system_tts` 发
        `{"type":"session_end","reason":...}`。
        与 `_play_system_tts` 的区别：后者是"播"（同步等待），本方法是"安排播"，用于不能阻塞音频帧处理的路径。
        """
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self._speak_log_count = 0
        self.state = State.SPEAKING
        self._turn += 1
        await self._send({"type": "state", "state": "speaking"})
        await self._send({"type": "system_message", "text": text, "turn": self._turn})
        # 挂后台任务：本方法在音频帧处理路径上被调用，绝不能在这里 await 播报
        self._speaking_task = self._track_task(
            self._play_system_tts(text, self._turn, session_end_reason),
            name=f"system-tts-{self._turn}",
        )
