"""Kokoro-82M 流式 TTS（美式英语）+ 雅思陪练音色，级联 Pipeline 的最后一环。

输出固定**单声道 24000Hz float32 裸 PCM**（无 WAV 头，量化成 int16 是调用方的事）；
`SAMPLE_RATE=24000` 是下行链路，与上行 ASR 的 16000Hz（config.SAMPLE_RATE）不是一回事。
⚠️ 24000 是 Kokoro 的**硬约束**（模型内部声码器采样率），不要"为了统一"改成 44100——
那等于让浏览器按错误速率播放，声音会变调。

权重从**本地目录**（`models/Kokoro-82M`）加载，不走 HuggingFace 缓存。

音色**固定在 `VOICE`**（`af_heart`）：标准美式英语女声，社区评级 A、韵律最自然、
长段落不易听疲劳，正适合雅思听力陪练。

本项目**不做音色克隆，也不开放换音色**：
  · Kokoro 是用预置 voicepack 的模型，本来就不支持零样本克隆（旧版 VoxCPM 那套已整体移除）；
  · 因此 `synthesize()` 系列没有 `voice` 参数，也没有 `set_default_voice()` —— 想换只能改
    这里的 `VOICE` 常量（11 个 `af_*` 备选见 README 10.2）。
  · 与旧版 VoxCPM 的另一处差异：**没有"丢弃首块"**。Kokoro 不是自回归声码器，首块就是
    正常音频，照抄旧行为会每句都丢音。

引擎**不是线程安全**的：所有合成调用必须串行（由调用方的有界队列间接保证）。
"""
import logging
import time

import numpy as np

from config import KOKORO_MODEL_DIR, TTS_SAMPLE_RATE


# ws_server 里 logging.basicConfig 的格式串含 %(name)s，命名空间统一成 "Voice Agent.*"，
# 可一条命令过滤整条链路：grep "Voice Agent" logs/voice_agent.log
log = logging.getLogger("Voice Agent.tts")


class KokoroTTS:
    """Kokoro-82M TTS 引擎（单实例 = 一份 82M 权重 + 一个**固定**音色）。

    生命周期：`__init__`（只存配置）→ `load()`（拉权重/建 pipeline）→ `warmup()` → `synthesize*()`。
    """

    # ── 固定的音频与音色参数（不做克隆、不开放切换）──────────────────────────
    # 输出采样率：Kokoro 声码器的硬约束。状态机用它算发送分片与 30s 安全上限，
    # /api/info 靠它告诉前端按什么速率播放。
    SAMPLE_RATE = TTS_SAMPLE_RATE

    # 音色：标准美式英语女声。选它的理由 —— 社区评级 A、韵律最自然、
    # 长段落不易听疲劳，贴合雅思听力陪练场景。
    # 备选 af_bella / af_nicole / af_sarah 等（见 README 10.2），但本项目不提供切换入口。
    VOICE = "af_heart"

    # 上游仓库名：仅用于日志/消警告，权重实际从 KOKORO_MODEL_DIR 本地读取
    UPSTREAM_REPO = "hexgrad/Kokoro-82M"

    # 语言代码由音色名前缀决定（af_* → 'a' 美式英语）。两者必须匹配，写死在这里
    # 就不存在"配错口音"的可能。
    LANG_CODE = VOICE.split("_")[0][0]

    def __init__(self, model_dir: str = KOKORO_MODEL_DIR, speed: float = 1.0,
                 device: str | None = None):
        """保存配置，**不加载模型、不碰 CUDA**（延迟到 `load()`）。

        `model_dir` 是**本地权重目录**（默认 `models/Kokoro-82M`），里面要有
        `config.json` / `*.pth` / `voices/<音色>.pt`。**不走 HuggingFace 缓存**。
        `speed` 是整体语速倍率（雅思听力按自然速 1.0，想让学员听清可用 0.9）。
        音色与语言代码不可配置，见类常量 `VOICE` / `LANG_CODE`。
        """
        self._model_dir = model_dir
        self._speed = speed
        self._device = device
        self._pipeline = None          # load() 之前调用 synthesize* 会在 None 上取属性报错
        self._warmed_up = False        # warmup() 幂等靠这个标志

    def load(self):
        """从**本地目录**建 KModel + KPipeline（阻塞：要读 327MB 权重并搬上设备）。

        为什么不直接把本地路径当 `repo_id` 传：Kokoro 内部用
        `hf_hub_download(repo_id=...)` 取 `config.json` 和 `voices/*.pt`，
        `repo_id` 只能是 Hub 上的仓库名。所以这里自己 new 出 `KModel`（它接受本地
        `config` / `model` 路径）再交给 `KPipeline(model=...)`；音色也直接传本地
        `.pt` 路径（`load_single_voice` 见到 `.pt` 就当文件路径用）。**全程不联网**。

        延迟 import `kokoro`：这个包会拉起 torch，只在真要跑 TTS 的进程里才 import。
        Kokoro 只认 'cuda' / 'cpu' 这种粗粒度设备名，不认 'cuda:1'；
        选哪张物理卡请用 CUDA_VISIBLE_DEVICES（启动脚本已经这么做了）。
        """
        from pathlib import Path

        import torch
        # ── 先把本地 spaCy 模型挂上 sys.path，再构造 KPipeline ──
        # misaki 的英文 G2P 会 `spacy.load("en_core_web_sm")`；找不到包时它会
        # `spacy.cli.download()` 现场联网（GitHub raw 通常不可达，报错还跟 TTS 毫无关系）。
        # 必须有 dist-info 才能让 `spacy.util.is_package()` 命中，所以目录要解压到
        # models/en_core_web_sm/（内含 en_core_web_sm/ 与 *.dist-info/）。
        # 放在这里而不是 server/bootstrap.py：这样任何直接用 engine.tts 的脚本
        # 也能跑，不依赖"必须经由服务进程启动"。
        import sys as _sys
        _spacy_dir = Path(KOKORO_MODEL_DIR).parent / "en_core_web_sm"
        if _spacy_dir.is_dir() and str(_spacy_dir) not in _sys.path:
            _sys.path.insert(0, str(_spacy_dir))

        from kokoro import KModel, KPipeline

        model_dir = Path(self._model_dir)
        config_path = model_dir / "config.json"
        # 权重文件名随版本变化（kokoro-v1_0.pth 等），按扩展名找更稳
        weights = sorted(model_dir.glob("*.pth"))
        voice_path = model_dir / "voices" / f"{self.VOICE}.pt"
        missing = [str(p) for p in (config_path, voice_path) if not p.is_file()]
        if not weights:
            missing.append(str(model_dir / "*.pth"))
        if missing:
            raise FileNotFoundError(
                f"Kokoro 权重目录不完整: {model_dir}\n"
                f"  缺少: {', '.join(missing)}\n"
                f"  获取方式: huggingface-cli download hexgrad/Kokoro-82M "
                f"--local-dir {model_dir}"
            )

        # 设备：cuda → 'cuda'；cuda:N → 'cuda'（物理卡由 CUDA_VISIBLE_DEVICES 选）；其它 → cpu
        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"TTS_DEVICE={device} 但 CUDA 不可用。"
                    "检查驱动，或显式用 TTS_DEVICE=cpu"
                )
            device = "cuda"
        else:
            device = "cpu"

        # 传 repo_id 只为消掉 Kokoro 内部的 "Defaulting repo_id" 提示：
        # 权重和音色都走本地路径，这个值不会被用来下载任何东西。
        kmodel = KModel(repo_id=self.UPSTREAM_REPO,
                        config=str(config_path), model=str(weights[0]))
        self._pipeline = KPipeline(
            lang_code=self.LANG_CODE,
            repo_id=self.UPSTREAM_REPO,
            model=kmodel.to(device).eval(),
        )
        # 音色用本地 .pt 的**绝对路径**：KPipeline 见到 .pt 就直接 torch.load，不查 Hub
        self._voice_path = str(voice_path)
        log.info("Kokoro TTS loaded (dir=%s, lang=%s, voice=%s, speed=%.2f, device=%s)",
                 model_dir, self.LANG_CODE, self.VOICE, self._speed, device)
        return self

    def warmup(self):
        """预热：跑一次最短合成（幂等），把首次推理的一次性开销提前付掉。

        第一次合成还要付 torch 算子/推理图初始化、权重搬到显存的成本（几百 ms）；
        不预热它们会叠加在第一轮真实对话的 ttfa 上、直接击穿 600ms 首响预算。
        """
        if not self._warmed_up:
            # 用一句极短英文做预热文本，长度接近真实首句
            self.synthesize("Hello, how can I help you?")
            self._warmed_up = True   # 成功后才置位：失败时下次仍会重试

    def synthesize(self, text: str) -> dict:
        """非流式整句合成，返回 dict（audio / sr / ttfa_ms / total_ms / chunks）。

        给"不适合流式"的场景用（如 `warmup()` 预热）。主链路请用 `synthesize_stream()`。
        Kokoro 按内部标点把文本切成若干段、每段一次性出音频，所以 `chunks` = 段数。
        `audio` 是未量化的 float32 裸 PCM；文本为空时是空数组，调用方需自行判断。
        """
        t0 = time.perf_counter()
        chunks = []
        ttfa = None
        for seg_audio in self._generate(text):
            if ttfa is None:
                ttfa = (time.perf_counter() - t0) * 1000   # 首个音频段到达时刻
            chunks.append(seg_audio)

        total = (time.perf_counter() - t0) * 1000
        audio_out = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        return {
            "audio": audio_out,
            "sr": self.SAMPLE_RATE,
            "ttfa_ms": ttfa or total,   # 无产出时退化成 total，保证下游永远拿到有限数
            "total_ms": total,
            "chunks": len(chunks),
        }

    def synthesize_stream(self, text: str):
        """流式合成：边生成边 yield，每个音频段一个 dict（audio / chunk_idx / ttfa_ms）。

        "160ms 精确打断"落在调用方：engine/dialog/streaming.py 每拿到一段就立即下发裸 PCM，
        并在**每段之前**检查取消标志，用户一开口就 break。所以打断粒度 = 一个音频段的
        播放时长（Kokoro 按句切段，通常 1~3s，比旧版 VoxCPM 的块更粗）。
        本函数是**生成器**：不迭代就不会跑推理。
        """
        idx = 0
        t0 = time.perf_counter()
        for seg_audio in self._generate(text):
            idx += 1
            yield {
                "audio": seg_audio,
                "chunk_idx": idx,
                # 只有首段带 ttfa；其余用 0 表示无意义，避免下游误读成每段都这么慢
                "ttfa_ms": (time.perf_counter() - t0) * 1000 if idx == 1 else 0,
            }

    def _generate(self, text: str):
        """内部：调 Kokoro 逐段产出 float32 一维数组；空文本直接不产出。

        KPipeline 的签名是 `pipeline(text, voice=..., speed=...)`，yield `(gs, ps, audio)`：
        gs=字形串、ps=音素串（都只用于调试），audio 是 torch.Tensor（可能在 CPU 上）。
        首段做一个 5ms 淡入，避免硬切起音的"啪"声——**只淡入、绝不丢弃**。
        """
        text = (text or "").strip()
        if not text or self._pipeline is None:
            return

        first = True
        for _gs, _ps, audio in self._pipeline(
            text, voice=self._voice_path, speed=self._speed
        ):
            a = self._to_np(audio)
            if a.size == 0:
                continue
            # 本段是本次合成的首段 → 淡入。5ms = 24000 × 0.005 = 120 点，
            # 短到听不出音量变化，又能压掉起音爆点。每次调用都会重新淡入（first 是局部变量）。
            if first:
                fade = int(self.SAMPLE_RATE * 0.005)
                if a.size > fade:
                    a[:fade] *= np.linspace(0, 1, fade, dtype=a.dtype)
                first = False
            yield a

    @staticmethod
    def _to_np(audio) -> np.ndarray:
        """把 torch.Tensor / list / ndarray 统一成 float32 一维 ndarray。"""
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32, copy=False).reshape(-1)
        if hasattr(audio, "detach"):        # torch.Tensor
            audio = audio.detach().cpu().numpy()
            return audio.astype(np.float32, copy=False).reshape(-1)
        return np.asarray(audio, dtype=np.float32).reshape(-1)
