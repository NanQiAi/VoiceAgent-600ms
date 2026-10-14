"""FireRedVAD 流式包装 —— Silero VAD 的 drop-in 替代（F1 97.57% vs 95.95%，误报 2.69% vs 9.41%）。

接口与 `engine/vad.py::SileroVAD` 完全一致（load / reset / process_chunk / is_speech_active），
ConversationManager 无需改动即可换用；启用在 ws_server（USE_FIRERED_VAD=1，每个连接单独 new
一个实例以隔离多会话状态）。FireRedVAD 按 25ms 帧 / 10ms 帧移判决，本包装器把 pipeline 的
512 点（32ms）chunk 切成它的帧节奏。两点差异：判停门限折算约 150ms（Silero 侧约 480ms）更灵敏；
speech_prob 是为兼容 Silero 由布尔事件映射出的"等效概率"，非模型原始概率。
"""
import os
import logging
import numpy as np

log = logging.getLogger("voxlabs.firered_vad")

FIRERED_FRAME_LEN = 400    # 25ms at 16kHz
FIRERED_FRAME_SHIFT = 160  # 10ms at 16kHz


class FireRedVADWrapper:
    """把 FireRedStreamVad 包装成 ConversationManager 使用的 SileroVAD 接口；输入 512 点 chunk，输出同构事件字典。"""

    def __init__(self, model_dir: str, threshold: float = 0.5):
        """只记配置并初始化流式状态；model_dir 下若有 Stream-VAD 子目录会优先使用，threshold 越小越容易判成语音。"""
        self.threshold = threshold
        self._model_dir = model_dir
        self._stream_vad = None
        self._speech_active = False
        self._silence_count = 0
        # 跨 chunk 余数缓冲：不足一帧的尾部样本留到下次拼接，保证 25ms 帧不被 chunk 边界切断
        self._leftover = np.array([], dtype=np.int16)

    def load(self):
        """加载 FireRedVAD 流式模型并返回 self；未装 fireredvad 会 ImportError，且在**事件循环内同步执行**（建连时调用）。"""
        # 延迟导入：只有真的启用 FireRedVAD 时才要求安装该包
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

        stream_model_dir = os.path.join(self._model_dir, "Stream-VAD")
        # 没有该子目录就退回整个 model_dir（兼容不同的解包布局）
        if not os.path.isdir(stream_model_dir):
            stream_model_dir = self._model_dir

        # 帧移 10ms / 帧长 25ms。use_gpu=False：模型很小，且不能和 ASR/TTS 抢显存；
        # smooth_window_size=5 ≈ 50ms 滑窗压单帧抖动；pad_start_frame=5 ≈ 50ms 防吃首字；
        # min_speech_frame=8 ≈ 80ms 才确认起点（滤咔嗒/爆破声）；min_silence_frame=15 ≈ 150ms 判终点
        config = FireRedStreamVadConfig(
            use_gpu=False,
            speech_threshold=self.threshold,
            smooth_window_size=5,
            pad_start_frame=5,
            min_speech_frame=8,
            min_silence_frame=15,
        )
        self._stream_vad = FireRedStreamVad.from_pretrained(stream_model_dir, config)
        log.info("FireRedVAD loaded (model_dir=%s, threshold=%.2f)", stream_model_dir, self.threshold)
        return self  # 支持 FireRedVADWrapper(...).load() 链式写法

    def reset(self):
        """清空流式状态（含底层帧缓存与平滑窗口，否则上一轮的语音尾巴会污染新一轮），准备开始新一轮会话/识别。"""
        if self._stream_vad is not None:
            self._stream_vad.reset()
        self._speech_active = False
        self._silence_count = 0
        self._leftover = np.array([], dtype=np.int16)

    def process_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        """处理一个 chunk（512 点 = 32ms@16kHz），返回与 SileroVAD 同构的事件字典；speech_end=True 即判停。

        把 chunk 拼上上次余下的尾巴，按 25ms 窗 / 10ms 移切帧（相邻帧重叠 15ms），逐帧 detect_frame 后聚合。sr 不参与重采样。
        """
        # float32 → int16（detect_frame 吃 int16）；本身已是 int16 则零转换
        int16 = (np.clip(audio_chunk, -1, 1) * 32767).astype(np.int16) if audio_chunk.dtype != np.int16 else audio_chunk

        samples = np.concatenate([self._leftover, int16])

        max_prob = 0.0
        any_speech_start = False  # 本 chunk 内是否出现语音起点事件
        any_speech_end = False    # 本 chunk 内是否出现语音终点事件（判停）
        any_is_speech = False

        # 逐帧消费：一帧 400 点（25ms），每次前进 160 点（10ms），相邻帧重叠 240 点；
        # 不足一整帧的尾部会在循环结束后留在 _leftover 里等下一个 chunk
        pos = 0
        while pos + FIRERED_FRAME_LEN <= len(samples):
            frame = samples[pos:pos + FIRERED_FRAME_LEN]
            result = self._stream_vad.detect_frame(frame)
            max_prob = max(max_prob, result.smoothed_prob)
            # 起点/终点事件做"或"聚合：32ms 内少一帧都会漏掉，必须逐帧记录
            if result.is_speech:
                any_is_speech = True
            if result.is_speech_start:
                any_speech_start = True
            if result.is_speech_end:
                any_speech_end = True
            pos += FIRERED_FRAME_SHIFT

        self._leftover = samples[pos:]

        # 布尔事件 → Silero 风格的"等效概率"（上层阈值判断在 0.5 附近）：在说话/刚起说抬到
        # >= 0.8 确保判为语音；刚判停压到 0.1 防止被当成说话重启一轮；纯静音夹到 0.3 以下
        if any_is_speech or any_speech_start:
            effective_prob = max(max_prob, 0.8)
        elif any_speech_end:
            effective_prob = 0.1
        else:
            effective_prob = min(max_prob, 0.3)

        # 字段名/类型都不能变：与 SileroVAD.process_chunk 完全同构
        event = {
            "speech_prob": effective_prob,
            "speech_start": any_speech_start,
            "speech_end": any_speech_end,
        }

        # 状态机：仅在"未在说话"时接受起点、仅在"正在说话"时接受终点，避免重复触发 start/end
        if any_speech_start and not self._speech_active:
            self._speech_active = True
            self._silence_count = 0
        if any_speech_end and self._speech_active:
            self._speech_active = False
            self._silence_count = 0

        # 注意 _silence_count 目前只写不读：判停已由 FireRedVAD 的 min_silence_frame 内部完成，
        # 真正决定 speech_end 的是 any_speech_end，这里属于保留结构（与 SileroVAD 对齐），仅标注不修改
        if not any_speech_start and not any_speech_end:
            if not any_is_speech and self._speech_active:
                self._silence_count += 1
            elif any_is_speech:
                self._silence_count = 0

        return event  # 与 SileroVAD.process_chunk 返回值同构

    @property
    def is_speech_active(self):
        """当前是否处于"说话中"（True 表示 ConversationManager 正在积累用户语音）。"""
        return self._speech_active
