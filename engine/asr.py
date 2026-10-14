"""SenseVoiceSmall ASR（FunASR）—— ASR 环节的默认实现，**不是** ASR 工厂。

开关在 ws_server.load_engines()（USE_FIRERED_ASR=1 → engine/asr_firered.py）。契约：load() 预热
返回 self；transcribe(audio, sr=16000) → {"text", "latency_ms"}，audio 为 float32 [-1,1] 单声道
(N,)，sr 固定 16k 且本实现不使用，text 可为空串且不 strip；**同步阻塞**，调用方须放进线程池。
"""
import time
# os：历史遗留导入，本文件已不使用（保留原样不删）
import os
import numpy as np


class SenseVoiceASR:
    def __init__(self, model_dir: str, device: str = "cuda:2"):
        """只记配置；model_dir 缺 config.yaml/model.pt 会在 load() 报错，device 默认 "cuda:2" 只是历史值（ws_server 实际传 ASR_DEVICE）。"""
        self._model_dir = model_dir
        self._device = device
        self._model = None

    def load(self):
        """加载并预热，返回 self；**阻塞**且占显存，权重缺失/CUDA 不可用/显存不足直接抛出（不捕获）。"""
        # 延迟导入：funasr 会连带拉入 torch，放文件顶部会让 import engine.asr 也变重
        from funasr import AutoModel
        self._model = AutoModel(
            # 本地权重目录：离线部署，不走 ModelScope/HuggingFace 下载
            model=self._model_dir,
            # SenseVoice 的 modeling 不在 pip 包内，必须允许执行权重目录里的自定义代码
            trust_remote_code=True,
            device=self._device,
            # 关掉联网版本检查：内网/断网下会拖慢甚至卡住冷启动
            disable_update=True,
        )
        # 把一次性开销（cuDNN 算法选择、显存分配）挪到启动阶段，别让用户第一句话买单
        self._warmup()
        return self

    def _warmup(self):
        """用 1 秒白噪声（幅值未缩放，超出 [-1,1]；只预热不看结果）跑一次完整 transcribe，触发全部惰性初始化。"""
        dummy = np.random.randn(16000).astype(np.float32)
        self.transcribe(dummy)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """转写一整句音频（**同步阻塞**跑 GPU 推理，调用方须自行放进线程池）；返回字段名是三个 ASR 实现共用的契约。"""
        t0 = time.perf_counter()
        result = self._model.generate(
            input=audio,
            # 必须传空 dict：否则 FunASR 会把上一句的解码状态带进来，造成串音/丢字
            cache={},
            language="zh",
            # use_itn=True：逆文本正则化，把"二零二六年"归一成"2026年"，便于下游 LLM 读数字
            use_itn=True,
            # batch_size_s 按"总时长不超过 60 秒"动态组 batch；单句远小于 60s，等价 batch=1
            batch_size_s=60,
        )
        latency = (time.perf_counter() - t0) * 1000

        # 兜底空串：静音段/纯噪声时下游要靠空串去走"没听清"话术
        text = ""
        if result and len(result) > 0:
            text = result[0].get("text", "")

        return {"text": text, "latency_ms": latency}
