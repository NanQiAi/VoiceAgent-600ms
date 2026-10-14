"""Silero VAD 薄封装：逐帧语音概率聚合成 speech_start / speech_end 端点事件（链路第一级）。"""
import os
import torch
import numpy as np

class SileroVAD:
    """逐帧概率 → 端点事件；ws_server.py 在它 / SpeakerAwareVAD / FireRedVADWrapper 间三选一装配。"""

    def __init__(self, model_dir: str, threshold: float = 0.5):
        """只保存配置，不加载模型。

        threshold 是"这一帧是人声"的判定概率：调高更难触发 speech_start、更容易判停（抗噪但可能
        切碎句子），调低更灵敏但噪声/TTS 回声易误触发 Barge-in。
        """
        self.threshold = threshold
        self._model = None
        self._model_dir = model_dir

    def load(self):
        """加载模型并置为 eval，返回 self 供链式写法；内部会 reset() 清空状态。

        本地没有 `silero_vad.jit` 时会退化为 torch.hub 联网下载（首次较慢）。
        """
        jit_path = os.path.join(self._model_dir, "silero_vad.jit")
        if os.path.exists(jit_path):
            # map_location="cpu"：VAD 常驻 CPU，避免与 ASR/LLM/TTS 抢 GPU 显存
            self._model = torch.jit.load(jit_path, map_location="cpu")
        else:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=False,
            )
        self._model.eval()
        self.reset()
        return self

    def reset(self):
        """重置状态：新会话 / Barge-in 打断后 / 一轮对话结束都要调用（清 RNN 隐状态与端点状态机）。"""
        if self._model is not None:
            # Silero 是状态模型，隐状态跨帧延续；不重置会让上一轮的"人声"惯性污染新会话
            self._model.reset_states()
        self._speech_active = False
        self._silence_count = 0

    def process_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        """处理一帧音频，返回 speech_prob / speech_start / speech_end（有状态，须按帧顺序调用）。

        Silero v5 在 16kHz 下只接受精确 512 采样点（=32ms），与前端 AudioWorklet 分帧和
        config.CHUNK_SAMPLES 严格对齐。
        """
        tensor = torch.from_numpy(audio_chunk).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        prob = self._model(tensor, sr).item()

        event = {"speech_prob": prob, "speech_start": False, "speech_end": False}

        # 上升沿：上游据此清空音频缓冲、开始新一轮 Turn 累积，并在 TTS 播放期触发 Barge-in
        if prob >= self.threshold and not self._speech_active:
            self._speech_active = True
            self._silence_count = 0
            event["speech_start"] = True
        elif prob < self.threshold and self._speech_active:
            self._silence_count += 1
            # 15 帧 × 32ms ≈ 480ms 静音才判停：经验值，短于此会把句中换气、停顿切碎
            if self._silence_count >= 15:
                self._speech_active = False
                self._silence_count = 0
                event["speech_end"] = True
        elif prob >= self.threshold and self._speech_active:
            self._silence_count = 0

        return event

    @property
    def is_speech_active(self):
        """当前是否处于人声段（只读）。"""
        return self._speech_active
