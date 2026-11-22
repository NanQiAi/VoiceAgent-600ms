"""E2E对话 · 判停与打断判定

决定“用户说完了没有”：按本次说话时长自适应选择静音阈值。
本项目只有静音帧计数这一条判停依据，没有语义判停器。
"""

import logging

import numpy as np

from .constants import (
    CHUNK_MS,
    ENDPOINT_DEFAULT_CHUNKS,
    ENDPOINT_SLOW_CHUNKS,
)
from .state import State

log = logging.getLogger("Voice Agent.cm")


class Endpointing:
    """E2E对话 · 判停与打断判定。由 ConversationManager 组合，`self` 上共享整机会话状态。"""

    def _adaptive_endpoint_threshold(self) -> int:
        """按本次说话时长选判停所需的静音帧数。

        只有两档：
          · 中段（500~3000ms）→ DEFAULT（13 帧 ≈ 416ms），最常用；
          · 两端（很短 / 很长）→ SLOW（20 帧 ≈ 640ms）。

        两端都给足等待是有意的：短句多半是用户还在换气、马上要接着说，
        早判停会把一句话切两半；长句则怕用户中途长停顿。

        历史上这里还有一档 ENDPOINT_FAST_CHUNKS（8 帧），从未被返回过，
        属于会误导读者的死常量，已删除。
        """
        audio_duration_ms = len(self._audio_buffer) * CHUNK_MS
        if 500 <= audio_duration_ms <= 3000:
            return ENDPOINT_DEFAULT_CHUNKS
        return ENDPOINT_SLOW_CHUNKS


    async def _on_endpointing(self):
        """判停：把本轮音频冻结成一段、起 pipeline 后台任务，并立即切到 THINKING。

        副作用：取消累积器定时器；复位 `_silence_count`/`_barge_confirm_count`/THINKING 后到语音缓冲；
        自增 `_turn`；下发 state=thinking；把投机 ASR 的结果/长度/句柄摘走转移给 pipeline。
        这里**不 await pipeline**：否则 feed_audio 会被整条 LLM+TTS 链路阻塞，用户说话没人收集。
        """
        # 本轮音频即将定稿，缓冲器的兜底定时器必须取消（否则 3s 后会把旧文本再发一轮）
        self._cancel_accum_flush()
        self.state = State.THINKING
        self._silence_count = 0
        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False
        self._turn += 1
        await self._send({"type": "state", "state": "thinking"})

        if self._audio_buffer:
            audio = np.concatenate(self._audio_buffer)
        else:
            audio = np.array([], dtype=np.float32)
        self._audio_buffer = []

        # "摘走"投机 ASR 的结果/长度/句柄（局部变量持有）并复位实例字段，避免与新一轮冲突
        spec_result = self._spec_asr_result
        spec_audio_len = self._spec_asr_audio_len
        spec_task = self._spec_asr_task
        self._spec_asr_result = None
        self._spec_asr_task = None
        self._spec_asr_audio_len = 0

        self._track_task(
            self._run_pipeline(audio, self._turn, spec_result, spec_audio_len, spec_task),
            name=f"pipeline-{self._turn}",
        )
