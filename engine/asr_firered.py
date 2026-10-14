"""FireRedASR2-AED 适配层 —— 把 vendored 的 engine/fireredasr2/ 包成与 SenseVoiceASR 一致的接口。

开关在 ws_server.load_engines()：USE_FIRERED_ASR=1 → 本实现（AED，中文 CER 2.89%，覆盖
20+ 方言，单句延迟约占 117ms）；否则用 engine/asr.py。上游只接受音频文件路径（内部 kaldiio
读盘 + fbank），故每次转写要先落一个临时 wav 再删 —— 这份磁盘 I/O 是复用上游代码、不魔改
vendored 源码的代价。
"""
import os
# sys：把 engine/ 插入 sys.path，才能 from fireredasr2.asr import ...
import sys
import time
import tempfile
import logging
# threading：给调用计数器加锁；transcribe 可能被多个会话的线程池并发调用
import threading
import numpy as np
import soundfile as sf

log = logging.getLogger("voxlabs.asr_firered")


class FireRedASR:
    """FireRedASR2-AED 的统一接口封装（与 SenseVoiceASR 兼容）；transcribe() 线程安全，load() 须在启动阶段单次调用。"""

    def __init__(self, model_dir: str, device: str = "cuda:0", use_half: bool = True):
        """只记配置并在系统临时目录下建一个专属目录放临时 wav；device 含 "cuda" 时启用 GPU，use_half 默认 FP16。"""
        self._model_dir = model_dir
        self._device = device
        self._use_half = use_half
        self._model = None
        self._tmp_dir = tempfile.mkdtemp(prefix="firered_asr_")
        self._call_counter = 0
        self._counter_lock = threading.Lock()

    def load(self):
        """加载模型并预热，返回 self；**阻塞**，会改进程级 sys.path 与当前 CUDA 设备，torch/权重问题直接抛出。"""
        # engine/ 绝对路径：fireredasr2 包就在它下面
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        # 幂等插入，避免重复 append 让路径列表无限增长
        if engine_dir not in sys.path:
            # 插到最前，确保优先加载仓库内 vendored 的 fireredasr2 而不是同名 pip 包
            sys.path.insert(0, engine_dir)

        # 延迟 import torch：让不需要 GPU 的环境也能 import 本模块
        import torch
        # 上游 asr.py 用的是 torch.cuda.current_device()，只有 CUDA 设备才需要切换
        if "cuda" in self._device:
            # 形如 "cuda:2" 取冒号后的序号；裸 "cuda" 默认 0 号卡
            gpu_idx = int(self._device.split(":")[-1]) if ":" in self._device else 0
            torch.cuda.set_device(gpu_idx)

        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
        use_gpu = "cuda" in self._device
        config = FireRedAsr2Config(use_gpu=use_gpu, use_half=self._use_half)
        # "aed" 选择 AED 架构分支（另一分支是 "llm"）
        self._model = FireRedAsr2.from_pretrained("aed", self._model_dir, config)
        # 预热：提前建好 CUDA 上下文、算子 autotune 与内存池
        self._warmup()
        log.info("FireRedASR loaded (model_dir=%s, device=%s)", self._model_dir, self._device)
        return self

    def _warmup(self):
        """用一段静音级随机噪声（×0.01 避免削波）跑一次完整转写，消除首句额外延迟（会读写一个临时 wav）。"""
        dummy = np.random.randn(16000).astype(np.float32) * 0.01
        self.transcribe(dummy)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """转写音频，返回 {'text': str, 'latency_ms': float}；线程安全，每次用唯一临时 wav。"""
        t0 = time.perf_counter()

        # 加锁自增，避免多线程同时进来拿到重复的 call_id
        with self._counter_lock:
            self._call_counter += 1
            call_id = self._call_counter

        tmp_path = os.path.join(self._tmp_dir, f"chunk_{call_id}.wav")
        # float32 [-1,1] → int16 满量程；clip 防止越界回绕
        int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        sf.write(tmp_path, int16, sr)

        # 上游只接受文件路径（kaldiio 读盘 + fbank），故落盘后必须删除；latency_ms 含这份 I/O
        try:
            # 上游批量接口：第一参数是 uttid 列表（这里固定占位 "utt"），第二参数是 wav 路径列表
            results = self._model.transcribe(["utt"], [tmp_path])
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        latency = (time.perf_counter() - t0) * 1000

        text = ""
        if results and len(results) > 0:
            text = results[0].get("text", "")

        # sr 必须为 16000；去掉上游 detokenize 后可能残留的首尾空格
        return {"text": text.strip(), "latency_ms": latency}
