"""Moonshine Tiny（27M / ONNX Runtime / **纯 CPU**）—— 投机式预推理用的小 ASR。

不是主 ASR 的替代品：ConversationManager 在静音满 5 帧（约 160ms）且还没抢跑过时先跑一次本类
transcribe，VAD 真判停且音频没明显变长（> 快照长度 × 1.15 即作废）就直接复用，省下约 117ms；
定稿文本仍由 engine/asr_firered.py 或 engine/asr.py 产出，失败记 None 退回主 ASR。放 CPU 是因为
显存要留给 TTS/LLM，27M ONNX 在 CPU 上约 20ms 且完全不抢显存。接口与主 ASR 一致，差别：额外
接受二维 (batch, N)、返回值会 strip；中文能力弱，只作投机候选抢时间。
"""
# os：历史遗留导入，本文件未使用（保留原样不删）
import os
import time
import logging
import numpy as np

log = logging.getLogger("voxlabs.asr_moonshine")


class MoonshineASR:
    def __init__(self, model_name: str = "moonshine/tiny"):
        """只记模型标识；传 HF 仓库名时首次 load() 会联网下载（离线请预先下载），tiny 与 base 的 tokenizer 不同。"""
        self._model_name = model_name
        self._model = None
        self._tokenizer = None

    def load(self):
        """加载 ONNX 模型与分词器并预热，返回 self；只走 CPU 不占显存，模型标识/网络问题由下游抛出（不捕获）。"""
        # 延迟导入：onnxruntime 体积大、初始化慢，只在真要加载时才付这份开销
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        self._model = MoonshineOnnxModel(model_name=self._model_name)
        # 不传参数：moonshine_onnx 默认取 tiny 的分词器，与默认 model_name 配套
        self._tokenizer = load_tokenizer()
        self._warmup()
        log.info("MoonshineASR loaded (model=%s)", self._model_name)
        return self

    def _warmup(self):
        """用 1 秒白噪声跑一次完整 transcribe，触发 onnxruntime 的惰性初始化。"""
        dummy = np.random.randn(16000).astype(np.float32)
        self.transcribe(dummy)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """转写音频；**同步阻塞**占一个 CPU 线程（须放进线程池），sr 不用（ONNX 固定 16k），text 已 strip、无语音时为空串。"""
        t0 = time.perf_counter()
        # moonshine_onnx 的 generate 期望 (batch, samples)，直接喂 (N,) 会维度不匹配报错
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        # astype 默认 copy=True，dtype 已正确时仍多拷贝一份（毫秒级，相比抢回的 117ms 可忽略）
        tokens = self._model.generate(audio.astype(np.float32))
        texts = self._tokenizer.decode_batch(tokens)
        text = texts[0] if texts else ""
        latency = (time.perf_counter() - t0) * 1000
        # 与 engine/asr.py 的 SenseVoice 实现不同：这里 strip 首尾空白，小模型容易吐多余空格
        return {"text": text.strip(), "latency_ms": latency}
