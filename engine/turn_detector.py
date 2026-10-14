"""语义判停（Endpointing）—— Pipecat Smart Turn v3（8M ONNX）判断"用户这句话说完了吗"。

吃原始波形，用升/降调等韵律线索判句末（固定 160ms 静默阈值会抢话/迟钝）；由 ws_server 在
USE_SMART_TURN=1 时加载并经 conversation_manager 注入状态机：should_endpoint() == True → 进
THINKING，一直 False 且静默达到 ENDPOINT_SLOW_CHUNKS（20 chunk ≈ 640ms）则兜底判停。
"现在有没有声音"由 VAD 负责，不是本模块职责。依赖 pipecat（未启用则不必安装）。
"""
import logging
import time
import numpy as np

log = logging.getLogger("voxlabs.turn_detector")

# Smart Turn v3 在 16kHz 上训练，重采样会破坏韵律特征，故写死（全链路统一 16k）
SAMPLE_RATE = 16000


class SmartTurnDetector:
    def __init__(self):
        # 模型懒加载：真正的 ONNX 会话在 load() 里创建，未 load 时为 None
        self._analyzer = None

    def load(self):
        """加载 Smart Turn v3 的 ONNX 模型并初始化分析器，返回 self；读 pipecat 包内约 8MB 权重，未装 pipecat 会 ImportError。"""
        # 延迟到方法内 import：没启用 Smart Turn 的环境完全不需要装 pipecat
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        self._analyzer = LocalSmartTurnAnalyzerV3()
        self._analyzer.set_sample_rate(SAMPLE_RATE)
        log.info("SmartTurnDetector loaded (Pipecat Smart Turn v3, 8M ONNX)")
        return self

    def reset(self):
        """清空分析器内部状态（含上一轮残留音频），开启新一轮用户语音；未 load() 时是空操作。"""
        # conversation_manager 检测到用户重新开口（is_speech=True）时调用
        if self._analyzer:
            self._analyzer.clear()

    def should_endpoint(self, audio_buffer: list, is_speech_active: bool) -> bool:
        """把缓冲音频喂给 Smart Turn，判断用户是否说完了（True → 状态机立即进 THINKING；一直 False 兜底到约 640ms）。

        audio_buffer 每块 512 点（32ms）float32；is_speech_active 形参当前未使用（见函数内 # 注意），仅为兼容调用签名保留。
        """
        # 只在方法内 import 枚举，避免模块级依赖 pipecat
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

        # 先清空：保证判停只依赖本次传入的完整 buffer，不受上次残留影响
        self._analyzer.clear()

        for chunk in audio_buffer:
            # float32 [-1,1] → int16 满量程（32767）；先 clip 防止越界回绕成反相噪声
            int16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            # 注意：is_speech_active 形参在此处并未使用，实际恒定传 True（已知瑕疵，保留说明）
            state = self._analyzer.append_audio(int16.tobytes(), is_speech=True)

        # 追加一帧 32ms 静音并把 is_speech 置 False：模型需要"语音结束"边界信号才能判句末
        silence = np.zeros(512, dtype=np.int16)
        # 这一帧的返回值才是最终判停结果（循环里的 state 每次都被覆盖）
        state = self._analyzer.append_audio(silence.tobytes(), is_speech=False)

        # COMPLETE 表示话轮结束；其余状态（INCOMPLETE 等）都视为"还没说完"
        return state == EndOfTurnState.COMPLETE

    def get_probability(self, audio_buffer: list) -> float:
        """返回话轮完成的粗粒度置信度（0.0/0.2/…/1.0，不是模型的 softmax 概率，而是 5 帧静音判停比例）；主链路未调用。"""
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

        self._analyzer.clear()

        for chunk in audio_buffer:
            int16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            state = self._analyzer.append_audio(int16.tobytes(), is_speech=True)

        # 连喂 5 帧静音（5 × 32ms = 160ms）"投票"，帧数越多结果越稳但越慢
        silence_chunks = 5
        complete_count = 0
        for _ in range(silence_chunks):
            silence = np.zeros(512, dtype=np.int16)
            state = self._analyzer.append_audio(silence.tobytes(), is_speech=False)
            if state == EndOfTurnState.COMPLETE:
                complete_count += 1

        # 用"说完帧占比"近似概率，返回步长 0.2 的离散值
        return complete_count / silence_chunks
