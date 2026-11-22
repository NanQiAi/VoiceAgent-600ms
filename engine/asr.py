"""Parakeet-TDT-0.6B ASR（NVIDIA）—— **本项目唯一的 ASR 实现**（没有工厂、没有多后端分支）。

用的是 `transformers` 对 Parakeet 的原生实现（`ParakeetForTDT` + `ParakeetProcessor`），
**不需要安装 NeMo**：`nvidia/parakeet-tdt-0.6b-v3` 仓库里同时提供了 `model.safetensors`
和 `tokenizer.json`，transformers 能直接加载。

契约：`load()` 预热并返回 self；`transcribe(audio, sr=16000)` → `{"text", "latency_ms"}`，
audio 为 float32 [-1,1] 单声道 (N,)，text 可为空串且不 strip（下游 pipeline 会 strip 并
清 ASR 标签）；**同步阻塞**，调用方须放进线程池。
"""
import time

import numpy as np


class ParakeetASR:
    def __init__(self, model_dir: str, device: str = "cuda:0"):
        """只记配置；model_dir 缺 config.json/model.safetensors 会在 load() 报错。

        device 由 loader 传 ASR_DEVICE（本项目 run.sh 固定给 cuda:0，物理卡由
        CUDA_VISIBLE_DEVICES 选）。
        """
        self._model_dir = model_dir
        self._device = device
        self._model = None
        self._processor = None

    def load(self):
        """加载并预热，返回 self；**阻塞**且占显存，权重缺失/显存不足直接抛出（不捕获）。"""
        # 延迟导入：transformers 会连带拉入 torch，放文件顶部会让 import engine.asr 也变重
        import torch
        from pathlib import Path

        from transformers import ParakeetForTDT, ParakeetProcessor

        # 先自己做完整性检查：权重下了一半时 safetensors 只会抛
        # "Error while deserializing header: incomplete metadata"，看不出该怎么办。
        model_dir = Path(self._model_dir)
        required = ["config.json", "tokenizer.json"]
        missing = [n for n in required if not (model_dir / n).is_file()]
        weights = sorted(model_dir.glob("*.safetensors"))
        if not weights:
            missing.append("*.safetensors")
        elif weights[0].stat().st_size < 500 * 1024 * 1024:
            missing.append(
                f"{weights[0].name} 只有 {weights[0].stat().st_size // 1048576}MB"
                f"（Parakeet-TDT-0.6B 约 2.5GB，疑似下载不完整）"
            )
        if missing:
            raise FileNotFoundError(
                f"ASR 权重目录不完整: {model_dir}\n"
                f"  缺少/异常: {', '.join(missing)}\n"
                f"  获取方式: huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 "
                f"--local-dir {model_dir}"
            )

        # 显式指定类而不是 AutoModel：Parakeet 家族还有 CTC/RNNT 两种头，
        # 自动分派虽然能靠 config 判断，但写死 TDT 能让"用错模型"在加载时就报错。
        self._processor = ParakeetProcessor.from_pretrained(self._model_dir)
        self._model = ParakeetForTDT.from_pretrained(self._model_dir)
        self._model.to(self._device).eval()
        self._torch = torch

        # 把一次性开销（显存分配、kernel 选择、首次解码循环）挪到启动阶段，
        # 别让用户第一句话买单。
        self._warmup()
        return self

    def _warmup(self):
        """用 1 秒低幅噪声跑一次完整 transcribe，触发全部惰性初始化（只预热、不看结果）。"""
        dummy = (np.random.randn(16000).astype(np.float32) * 0.01)
        self.transcribe(dummy)

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> dict:
        """转写整句音频（**同步阻塞**跑推理，调用方须自行放进线程池）。"""
        t0 = time.perf_counter()

        # processor 负责重采样校验 + 提取 log-mel 特征；传 sampling_rate 让它校验采样率
        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}

        # inference_mode 关闭 autograd，比 no_grad 再省一点显存与开销
        with self._torch.inference_mode():
            output = self._model.generate(**inputs)

        # generate 可能返回 Tensor，也可能返回带 .sequences 的输出对象，两种都兼容
        sequences = getattr(output, "sequences", output)
        text = self._processor.batch_decode(sequences, skip_special_tokens=True)[0]

        latency = (time.perf_counter() - t0) * 1000
        # 兜底空串：静音段/纯噪声时下游要靠空串去走"没听清"话术
        return {"text": text or "", "latency_ms": latency}
