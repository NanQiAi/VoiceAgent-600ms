"""对话状态机的调优常量与音频契约。

数值全部有实测依据，注释里写的是"为什么是这个值"。改任何一项都会影响
600ms 首响预算 / 打断灵敏度 / 判停快慢，改前先看 README 第 14 节。
"""

import re

import numpy as np

# 音频契约（采样率 / 帧长 / 帧点数）的唯一来源是 config.py，这里只做重导出，
# 让状态机内部统一从 `.constants` 取，不必各自记得去 import config。
from config import CHUNK_MS, CHUNK_SAMPLES, SAMPLE_RATE  # noqa: F401

# 剥掉 ASR 输出里的富文本标签（<|zh|>/<|NEUTRAL|>/<|Speech|> 等）：
# 对 LLM 无意义且会污染 RAG 检索
_ASR_TAG_RE = re.compile(r"<\|[^>]*\|>")

# 打断后连续静音 10 帧（≈320ms）就认为用户说完：比正常判停短，因为打断场景用户更急且已过一次 VAD 确认
INTERRUPT_SILENCE_CHUNKS = 10     # ~320ms before processing interrupted speech

# 打断判定的 VAD 语音概率门限（0~1）：比普通判停的 0.6 更严，专门对抗 TTS 外放回声
BARGE_IN_VAD_THRESHOLD = 0.6
# 打断判定能量门限（float32 RMS）：语音概率会被回声/键盘声骗过，要求真实能量，双门限同时满足
BARGE_IN_RMS_THRESHOLD = 0.012
# 连续 2 帧（≈64ms）都满足才真正打断：单帧毛刺（咳嗽/鼠标/回声尖峰）不足以打断 AI
BARGE_IN_CONFIRM_CHUNKS = 2       # ~64ms consecutive confirmation

# 判停阈值（单位：静音帧数）。只有两档，选择规则见
# `endpointing._adaptive_endpoint_threshold()`：
#   中段（500~3000ms）用 DEFAULT，其余用 SLOW。
ENDPOINT_DEFAULT_CHUNKS = 13      # ~416ms，中段语音用
ENDPOINT_SLOW_CHUNKS = 20         # ~640ms，短句与长句用（见函数的说明）

# 静音满 5 帧（≈160ms）且未启动过投机 ASR，就先用当前音频快照跑一次 ASR，判停时大概率已算好
SPECULATIVE_ASR_CHUNKS = 5        # ~160ms silence → start speculative ASR

# 每个 WebSocket 音频帧的字节数：4800 采样点 × 2 字节 ≈ 200ms@24kHz；比"一帧一块"大是为了减少帧数、提高弱网稳定性
TTS_SEND_CHUNK_BYTES = 4800 * 2   # ~200ms at 24kHz 16-bit (larger chunks for network stability)

# IDLE 下每 31 帧（≈1s）才查一次闲聊超时：避免每帧调 time.monotonic()，也避免超时话术被高频触发
IDLE_CHECK_INTERVAL_CHUNKS = 31   # ~1s — poll idle timeout at this cadence

# ASR 文本累积器：VAD 误切或用户只说"嗯/那个"时立刻送 LLM 既浪费一轮 turn 又容易答非所问，故先缓存极短文本等后续拼整句
ASR_ACCUM_MAX_CHARS = 4           # ASR result ≤ this → buffer, don't send to LLM yet
#   ↑ 经验值：≤4 个字符基本只可能是语气词或半截话（如"嗯""那个""就是"）
ASR_ACCUM_FLUSH_S = 3.0           # if no new speech within this, flush buffer to LLM
#   ↑ 超时兜底：用户说完半句就沉默走神时，3s 后把缓存内容照样发给 LLM，不能一直攒着
ASR_ACCUM_MAX_SEGMENTS = 3        # max buffered segments before forced flush
#   ↑ 防止无限累积：攒够 3 段就必须下发一轮，否则用户永远等不到回应


def compute_rms(chunk: np.ndarray) -> float:
    """一帧 float32 PCM 的均方根能量（范围约 [0,1]：静音≈0.0005，正常说话≈0.02~0.2）。"""
    return float(np.sqrt(np.mean(chunk ** 2)))
