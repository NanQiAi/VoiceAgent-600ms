"""Silero VAD 薄封装（链路第一级）：逐帧给出"这一帧是人声"的概率。

⚠️ 状态机目前**只消费 `speech_prob`**，自己按 `>= 0.6` 判定并计静音帧数（见
`engine/dialog/manager.py` 的 `_process_chunk`）。本类里的 `speech_start` /
`speech_end` 事件与 `is_speech_active` 属性目前**没有任何调用方**——它们是更早一版
"由 VAD 自己判端点"的设计残留，阈值（构造参数 `threshold`，默认 0.5）也只影响这两个
未被使用的事件，不影响实际判停。判停的真实阈值与帧数在 `engine/dialog/constants.py`。
"""
import os

import numpy as np
import torch

class SileroVAD:
    """逐帧语音概率；**本项目唯一的 VAD 实现**（Silero，模型 snakers4/silero-vad）。"""

    def __init__(self, model_dir: str, threshold: float = 0.5):
        """只保存配置，不加载模型。

        threshold 只用于下面那套 `speech_start` / `speech_end` 事件的状态机，而那些事件目前
        无人消费（见模块说明），所以这个参数**不影响实际判停与打断**。真正生效的门限：
        判停/打断各自在 `engine/dialog/manager.py` 与 `constants.py` 里。
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

        # 上升沿（目前无消费方，见模块说明）
        if prob >= self.threshold and not self._speech_active:
            self._speech_active = True
            self._silence_count = 0
            event["speech_start"] = True
        elif prob < self.threshold and self._speech_active:
            self._silence_count += 1
            # 15 帧 × 32ms ≈ 480ms —— 这套端点事件目前无人消费；实际判停用
            # constants.py 的 ENDPOINT_*_CHUNKS（13/20 帧），不要混为一谈
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
