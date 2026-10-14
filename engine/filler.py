"""语气词（filler）引擎：启动时把预置应答词合成为裸 PCM 缓存，运行时零延迟取用，填补 THINKING 阶段的空档。

下发由 engine/conversation_manager.py 的 `_send_filler()` 负责（`type=filler` + 裸音频帧）；
本模块只在启动阶段和 asyncio 事件循环线程里被访问，没有加锁。
"""
import random
import logging
import numpy as np

log = logging.getLogger("voxlabs.filler")

# 预置词表全是 1~2 音节的极短应答词；末尾逗号是刻意的——让 VoxCPM 用"话没说完"的升调读出来，
# 听起来像"我在听"而不是"我说完了"，避免用户误以为轮到自己说话而抢话（Barge-in）。
FILLER_TEXTS = ["嗯，", "好的，", "是的，", "嗯嗯，", "好，"]

# 单条 filler 的硬上限（秒），来自 v2.9.1 的一次线上问题：VoxCPM 偶发生成退化/停不下来
# （generation loop），一条语气词被合成十几秒，与后面正式回答叠着播（串音）。
# conversation_manager._send_filler() 里还有一个 44100*2*2 的等价上限做二次防御。
MAX_FILLER_SEC = 2.0


class FillerEngine:
    """语气词引擎：启动时预合成、运行时轮询取用（纯内存操作，不阻塞事件循环）。"""

    def __init__(self, tts_engine):
        """只保存 TTS 引用：需实现 `synthesize(text) -> dict`（含 "audio"/"ttfa_ms"）与 `SAMPLE_RATE`。"""
        self._tts = tts_engine
        # 实际存的是 int16 裸 PCM bytes，标注写 np.ndarray 属历史遗留，保留不改。
        self._cache: list[tuple[str, np.ndarray]] = []
        self._idx = 0

    def pregenerate(self):
        """把 FILLER_TEXTS 逐条串行预合成为缓存（启动时调用一次）。

        会触发 TTS/GPU 推理，是阻塞调用（启动期成本，运行时零成本）；单条失败只告警跳过，
        不向上抛异常——启动流程不能被个别语气词拖垮。
        """
        sr = getattr(self._tts, "SAMPLE_RATE", 44100)
        # 单声道 int16 = 2 字节/采样点：2.0s × 44100Hz × 2B = 176400 字节（必为偶数，
        # 按字节截断不会切在半个采样点中间）。
        max_bytes = int(MAX_FILLER_SEC * sr * 2)
        log.info("Pre-generating %d filler words (max %.1fs / %d bytes each)...",
                 len(FILLER_TEXTS), MAX_FILLER_SEC, max_bytes)
        for text in FILLER_TEXTS:
            try:
                # 非流式整句合成（不能换成 synthesize_stream：缓存需要完整波形）。
                result = self._tts.synthesize(text)
            except Exception as e:
                log.warning("  Filler '%s' TTS failed: %s — skipped", text, e)
                continue
            audio = result["audio"]
            if audio.dtype != np.int16:
                # 先 clip 再乘 32767（而非 32768）：不裁剪直接转 int16 会溢出成反相爆音。
                audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            raw = audio.tobytes()
            if len(raw) > max_bytes:
                # 正常语气词远低于 2s，命中即说明这次生成退化成循环了；砍尾留头（onset 决定自然度）。
                log.warning("  Filler '%s' too long: %d bytes (%.1fs) — truncating to %.1fs",
                            text, len(raw), len(raw) / sr / 2, MAX_FILLER_SEC)
                raw = raw[:max_bytes]
            self._cache.append((text, raw))
            log.info("  Filler '%s': %.1fms, %d bytes", text, result["ttfa_ms"], len(raw))
        if not self._cache:
            log.error("No filler words cached! Filler disabled.")
        else:
            log.info("Filler cache ready: %d items", len(self._cache))

    def get_filler(self) -> tuple[str, bytes]:
        """轮询取用下一条 filler，返回 `(文本, 裸 PCM int16 字节)`；一轮发完原地打乱顺序。

        缓存为空时返回 `("", b"")`；非线程安全，只允许在 asyncio 事件循环单线程里调用。
        """
        if not self._cache:
            return "", b""
        # 取模而非下标：天然实现轮询，且 _idx 与列表长度不一致时也不会越界。
        text, audio_bytes = self._cache[self._idx % len(self._cache)]
        self._idx += 1
        if self._idx >= len(self._cache):
            random.shuffle(self._cache)
            self._idx = 0
        return text, audio_bytes

    @property
    def sample_rate(self):
        """TTS 输出采样率（Hz，VoxCPM 为 44100）；刻意不做默认值兜底，缺属性应尽早报错。"""
        return self._tts.SAMPLE_RATE
