"""对话状态机：吃 32ms 音频帧/文本输入，驱动 VAD→ASR→RAG→LLM→TTS 级联链路。

每个 WebSocket 连接一份实例；只决定"什么时候做什么"，重活全部委托给注入的引擎（见 ConversationManager）。
"""

# 对外提供 ConversationManager（唯一对外类）、State（5 状态枚举）、_SessionMetrics（被 /api/metrics 读取）。
# 状态流转：IDLE --VAD检出--> LISTENING --静音达阈值--> THINKING --首句就绪--> SPEAKING --播完--> IDLE；
# SPEAKING/THINKING --打断--> INTERRUPTED --静音≥320ms--> THINKING。
# 三条主线：① turn 序列号防串音（所有下行消息带 turn，前端靠 audio_start / barge_in 对齐 playableTurn）；
# ② 双层打断：状态机置位 _cancel_speaking，各发送/合成循环自己检查后退出，没人去 task.cancel() 别人；
# ③ 到处 sleep/yield 是为了交还事件循环，否则同步 for 会卡住 feed_audio、打断迟钝。
# 线程边界：全部方法在同一 asyncio 事件循环里（uvicorn 单 worker），状态字段无需加锁；唯一的例外是
# run_in_executor 把 ASR/降噪/RAG/Captioner/TTS 等阻塞调用丢进默认线程池。_tasks 持强引用防 GC，shutdown 统一取消。

import re
import time
import asyncio
import logging
import numpy as np
from enum import Enum
from typing import Optional, Callable, Awaitable

# 日志命名空间与 ws_server/tts/filler 等统一成 voxlabs.*，便于按前缀过滤整条链路
log = logging.getLogger("voxlabs.cm")

# 剥掉 ASR 输出里的富文本标签（<|zh|>/<|NEUTRAL|>/<|Speech|> 等）：对 LLM 无意义且会污染 RAG 检索
_ASR_TAG_RE = re.compile(r"<\|[^>]*\|>")


class State(Enum):
    """对话状态机的 5 个状态。

    取值与下行协议 `{"type":"state","state":...}` 的字符串**一一对应**（前端 static/voice_agent.html 用它切指示灯）。
    """

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


# 前端 AudioWorklet 的采集采样率：必须与 static/voice_agent.html 里的 input SR 一致，否则 VAD/ASR 的时频特征都会偏
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512               # 32ms
# 16000Hz × 32ms = 512 点，与前端 AudioWorklet 帧长严格对齐；下面以"帧"为单位的阈值乘 32 即毫秒
CHUNK_MS = 32

# 打断后连续静音 10 帧（≈320ms）就认为用户说完：比正常判停短，因为打断场景用户更急且已过一次 VAD 确认
INTERRUPT_SILENCE_CHUNKS = 10     # ~320ms before processing interrupted speech

# 打断判定的 VAD 语音概率门限（0~1）：比普通判停的 0.6 更严，专门对抗 TTS 外放回声
BARGE_IN_VAD_THRESHOLD = 0.6
# 打断判定能量门限（float32 RMS）：语音概率会被回声/键盘声骗过，要求真实能量，双门限同时满足
BARGE_IN_RMS_THRESHOLD = 0.012
# 连续 2 帧（≈64ms）都满足才真正打断：单帧毛刺（咳嗽/鼠标/回声尖峰）不足以打断 AI
BARGE_IN_CONFIRM_CHUNKS = 2       # ~64ms consecutive confirmation

# 判停阈值按"本次话有多长"分档（单位：静音帧数）。
# 注意：名字里的 FAST/SLOW 指的是**响应快慢**，不是语音快慢——帧数越少=越早判停=响应越快。
ENDPOINT_FAST_CHUNKS = 8          # ~256ms for short utterances (<0.5s)
#   ↑ 预留档位：当前 `_adaptive_endpoint_threshold()` 分支里用的是 SLOW，见那里的注释
ENDPOINT_DEFAULT_CHUNKS = 13      # ~416ms for normal utterances
ENDPOINT_SLOW_CHUNKS = 20         # ~640ms for long utterances (>3s)

# 静音满 5 帧（≈160ms）且未启动过投机 ASR，就先用当前音频快照跑一次 ASR，判停时大概率已算好
SPECULATIVE_ASR_CHUNKS = 5        # ~160ms silence → start speculative ASR

# 每个 WebSocket 音频帧的字节数：8820 采样点 × 2 字节 ≈ 200ms@44.1kHz；比"一帧一块"大是为了减少帧数、提高弱网稳定性
TTS_SEND_CHUNK_BYTES = 8820 * 2   # ~200ms at 44.1kHz 16-bit (larger chunks for network stability)

# IDLE 下每 31 帧（≈1s）才查一次闲聊超时：避免每帧调 time.monotonic()，也避免超时话术被高频触发
IDLE_CHECK_INTERVAL_CHUNKS = 31   # ~1s — poll idle timeout at this cadence

# ASR 文本累积器：VAD 误切或用户只说"嗯/那个"时立刻送 LLM 既浪费一轮 turn 又容易答非所问，故先缓存极短文本等后续拼整句
ASR_ACCUM_MAX_CHARS = 4           # ASR result ≤ this → buffer, don't send to LLM yet
#   ↑ 经验值：≤4 个字符基本只可能是语气词或半截话（如"嗯""那个""就是"）
ASR_ACCUM_FLUSH_S = 3.0           # if no new speech within this, flush buffer to LLM
#   ↑ 超时兜底：用户说完半句就沉默走神时，3s 后把缓存内容照样发给 LLM，不能一直攒着
ASR_ACCUM_MAX_SEGMENTS = 3        # max buffered segments before forced flush
#   ↑ 防止无限累积：攒够 3 段就必须下发一轮，否则用户永远等不到回应

# 说话人门控：前端"声纹登记"后，IDLE 与 SPEAKING 都要求"本帧属于已登记声纹"，挡旁人插话/电视声/回声
SPEAKER_GATE_CONFIRM_CHUNKS = 5   # consecutive target-speaker chunks to open gate
#   ↑ 连续 5 帧（≈160ms）判定本人声纹才进 LISTENING：ECAPA-TDNN 每 3 帧才刷新相似度，连续计数抹平抖动
SPEAKER_GATE_BUFFER_CHUNKS = 10   # ~320ms ring buffer for verification in IDLE


def _rms(chunk: np.ndarray) -> float:
    """一帧 float32 PCM 的均方根能量（范围约 [0,1]：静音≈0.0005，正常说话≈0.02~0.2）。"""
    return float(np.sqrt(np.mean(chunk ** 2)))


class _SessionMetrics:
    """单个 WebSocket 会话的内存态延迟统计（不上报、不落盘），由 ws_server 的 /api/metrics 读取。"""

    # __slots__ 限定实例属性：省内存（每连接一份）并防止手滑写错属性名
    __slots__ = ("turns", "errors", "asr_ms_list", "llm_ms_list", "tts_ms_list", "fr_ms_list")

    def __init__(self):
        """初始化为空统计（不读配置、无副作用）。"""
        self.turns = 0
        self.errors = 0
        self.asr_ms_list: list[float] = []
        self.llm_ms_list: list[float] = []
        self.tts_ms_list: list[float] = []
        # 端到端首响样本（ms）= asr + rag + llm首句 + tts首帧，对应 600ms 预算线
        self.fr_ms_list: list[float] = []

    def record_turn(self, asr_ms=0.0, llm_ms=0.0, tts_ms=0.0, fr_ms=0.0):
        """记录一轮对话的各段耗时，并自增 self.turns（只有非 0 参数才入样本）。"""
        self.turns += 1
        # 逐个 if 判真：0 表示这一段没测到（如播报被立刻打断），不计入平均
        if asr_ms: self.asr_ms_list.append(asr_ms)
        if llm_ms: self.llm_ms_list.append(llm_ms)
        if tts_ms: self.tts_ms_list.append(tts_ms)
        if fr_ms: self.fr_ms_list.append(fr_ms)

    def record_error(self):
        """累加一次错误计数（供 /api/metrics 汇总）。"""
        self.errors += 1

    def summary(self) -> dict:
        """汇总本会话统计：turns/errors 与各段均值（无样本时 avg 为整数 0）。"""
        def _avg(lst):
            """均值，保留 1 位小数；空列表返回整数 0（避免除零）。"""
            return round(sum(lst) / len(lst), 1) if lst else 0
        # 键名与 ws_server 的 /api/metrics 输出、前端 updateLatency() 保持一致
        return {
            "turns": self.turns,
            "errors": self.errors,
            "avg_asr_ms": _avg(self.asr_ms_list),
            "avg_llm_ms": _avg(self.llm_ms_list),
            "avg_tts_ms": _avg(self.tts_ms_list),
            "avg_first_response_ms": _avg(self.fr_ms_list),
        }


class ConversationManager:
    """单个语音会话的对话状态机：吃 32ms 音频帧，吐协议消息与 TTS 音频。

    生命周期由 ws_server.ws_voice() 控制（建连时构造，断开时 shutdown()）。除 _produce_sentences
    跑在线程池的同步函数外，所有方法都必须在构造它的事件循环里调用，因此内部字段无需加锁。
    """

    def __init__(self, asr, llm, tts, rag, vad, filler, send_fn: Callable[..., Awaitable],
                 captioner=None, spec_asr=None, turn_detector=None, denoiser=None,
                 experience_config: dict = None):
        """装配依赖、建立初始状态与 D1~D4 体验层参数（不加载模型、不发网络请求）。

        各依赖都是阻塞式同步接口（本类统一丢线程池）：asr.transcribe -> {"text","latency_ms"}；
        llm.stream_sentences(...) 逐句 yield {"sentence","ttfs_ms",...}；tts.synthesize_stream 逐块 yield
        {"audio","chunk_idx","ttfa_ms"}（44.1kHz）；rag.get_context -> {"context","total_ms"}；
        vad.process_chunk(chunk, sr) -> 含 "speech_prob"（说话人版另有 is_target_speaker/speaker_similarity）；
        filler.get_filler -> (文本, 裸 PCM int16)；send_fn(dict) 中 {"_audio": bytes} 走二进制帧，其余 JSON。
        captioner / spec_asr / turn_detector / denoiser 均可选，None 时能力降级；experience_config 来自 config.py。
        """
        self.asr = asr
        self.llm = llm
        # 语音合成引擎（VoxCPM，输出 44.1kHz 单声道 int16）
        self.tts = tts
        self.rag = rag
        self.vad = vad
        self.filler = filler
        self.captioner = captioner
        self.spec_asr = spec_asr
        self.turn_detector = turn_detector
        self.denoiser = denoiser
        self._send = send_fn

        self.state = State.IDLE
        # 本轮已累积的语音帧列表（每元素 512 点 float32），判停后 concat 成整段送 ASR
        self._audio_buffer = []
        # 被打断的 AI 半截回复文本：下一轮拼进用户话里，让 LLM 知道"我上一句说到哪被打断了"
        self._interrupted_texts = []
        self._silence_count = 0
        # Turn 序列号：判停/文本输入/系统话术都自增，随所有下行消息下发，与前端 playableTurn 配对防串音
        self._turn = 0
        self._speaking_task: Optional[asyncio.Task] = None
        # 打断总闸：置位即"停止一切正在播放/合成的音频"，它是协作式取消信号而非"已取消"标志——
        # 没人去 task.cancel() 别人，全靠各循环自己检查它；每开始新一轮播报前必须 clear()，否则新音频会被自己掐掉
        self._cancel_speaking = asyncio.Event()
        self._dead = False

        self._barge_confirm_count = 0
        # THINKING 期间用户继续说话时暂存的音频：等本 turn 首段 ASR 完成后单独转写并拼进同一句
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False

        # 投机 ASR：静音刚满 160ms 就用当前音频先跑一次，判停时若音频没怎么变长就直接复用结果
        self._spec_asr_task: Optional[asyncio.Task] = None
        self._spec_asr_result: Optional[dict] = None
        self._spec_asr_audio_len = 0

        # 全部后台任务的强引用：asyncio 只持弱引用，不自己存一份会被 GC 提前回收；shutdown() 靠它统一取消
        self._tasks: set[asyncio.Task] = set()

        self.metrics = _SessionMetrics()

        # 系统级错误话术：pipeline 崩溃时替换成"一句道歉"播给用户，避免电话里突然静音
        self._error_apology = "抱歉系统开了个小差，您能再说一次吗？"

        # D1~D4 的话术与阈值由 ws_server 从 config.py 组装传入；None 或键缺失时用下面的兜底默认值
        cfg = experience_config or {}
        self._greeting_text = cfg.get("greeting_text", "")
        self._idle_timeout_s = cfg.get("idle_timeout_s", 15.0)
        # D2：静音多久（秒）后直接告别结束会话，必须 > idle_timeout_s
        self._idle_goodbye_s = cfg.get("idle_goodbye_s", 30.0)
        self._idle_prompt_text = cfg.get("idle_prompt_text", "您还在吗？有什么需要帮助的吗？")
        self._idle_goodbye_text = cfg.get("idle_goodbye_text", "好的，如果后续有问题随时找我，再见！")
        self._asr_empty_text = cfg.get("asr_empty_retry_text", "抱歉没太听清，您能再说一次吗？")
        self._asr_noisy_text = cfg.get("asr_noisy_suggest_text", "您那边环境好像有点嘈杂，要不您试试打字发给我？")
        self._max_empty_asr = cfg.get("max_consecutive_empty_asr", 3)
        self._farewell_keywords = cfg.get("farewell_keywords", ["再见", "拜拜", "挂了"])

        # D2 计时基准：用 monotonic 而非 time.time()，避免系统改时间/NTP 校正把超时算错
        self._idle_since: float = time.monotonic()
        self._idle_chunk_count = 0
        self._idle_prompted = False
        self._idle_goodbye_sent = False
        self._consecutive_empty_asr = 0
        self._session_ending = False

        self._speaker_enrolled = False
        self._idle_gate_confirm = 0
        self._idle_speech_ring: list[np.ndarray] = []
        self._idle_speech_samples = 0

        self._asr_accum_texts: list[str] = []
        self._asr_accum_flush_task: Optional[asyncio.Task] = None

    def enroll_speaker(self, audio: np.ndarray) -> dict:
        """用一段本人录音登记声纹（ECAPA-TDNN 嵌入），登记后 IDLE/SPEAKING 都额外要求"帧属于本人"。

        返回 {"success", "samples", "duration_s"}；失败时带 "reason"（"speaker_vad_not_available"
        或 "audio_too_short"，后者要求长度 ≥8000 点）。在事件循环线程上做一次 ECAPA-TDNN 前向（**阻塞**
        几十毫秒）；设计为建连初期调用，此时没人在说话，短暂阻塞可接受。
        """
        # VAD 不一定支持声纹（纯 Silero 版没有该方法）：用鸭子类型探测，避免 AttributeError
        if not hasattr(self.vad, 'enroll_speaker'):
            return {"success": False, "reason": "speaker_vad_not_available"}
        # 太短的音频算不出稳定嵌入，且 speaker_vad 内部也会拒（ENROLL_MIN_SAMPLES=8000）
        if len(audio) < 8000:
            return {"success": False, "reason": "audio_too_short",
                    "samples": len(audio), "duration_s": round(len(audio) / SAMPLE_RATE, 2)}
        ok = self.vad.enroll_speaker(audio)
        self._speaker_enrolled = ok
        self._idle_gate_confirm = 0
        self._idle_speech_ring = []
        self._idle_speech_samples = 0
        log.info("Speaker enrollment: %s (%d samples, %.1fs)",
                 "OK" if ok else "FAIL", len(audio), len(audio) / SAMPLE_RATE)
        return {"success": ok, "samples": len(audio),
                "duration_s": round(len(audio) / SAMPLE_RATE, 2)}

    def unenroll_speaker(self):
        """取消声纹登记，回退到只靠 VAD 概率 + RMS 能量的门控（任何人都能打断）。"""
        self._speaker_enrolled = False
        self._idle_gate_confirm = 0
        self._idle_speech_ring = []
        self._idle_speech_samples = 0
        log.info("Speaker unenrolled")

    @property
    def speaker_enrolled(self) -> bool:
        """当前是否已登记声纹（供 ws_server 回 `shield_status` 消息）。"""
        return self._speaker_enrolled

    def _cancel_accum_flush(self):
        """取消并清空累积器超时定时器（幂等）。"""
        # 只取消还在跑的任务；已完成的不能再 cancel（会抛 InvalidStateError）
        if self._asr_accum_flush_task and not self._asr_accum_flush_task.done():
            self._asr_accum_flush_task.cancel()
        self._asr_accum_flush_task = None

    def _is_short_utterance(self, text: str) -> bool:
        """文本是否短到需要先缓存、等后续语音拼成整句（≤ ASR_ACCUM_MAX_CHARS）。"""
        # 纯按字符数判断（中文一字≈一词），不做分词：便宜且不需要额外依赖
        return len(text) <= ASR_ACCUM_MAX_CHARS

    def _drain_accum(self, new_text: str = "") -> str:
        """取出并清空累积器，把待合并文本拼成一句可直接送 LLM 的话（中文逗号连接）。"""
        parts = self._asr_accum_texts + ([new_text] if new_text else [])
        self._asr_accum_texts = []
        # 内容已经取出，兜底定时器必须取消，否则 3s 后会把同一段文本再发一遍
        self._cancel_accum_flush()
        return "，".join(parts)

    async def _accum_flush_timer(self):
        """累积器超时兜底：等 ASR_ACCUM_FLUSH_S(3s) 后缓存非空就把它当一轮输入，置 THINKING 并起后台 pipeline。

        只走文本路径（无音频，跳过 ASR/降噪/Captioner）；被取消（用户又说话了）时静默退出。
        """
        try:
            await asyncio.sleep(ASR_ACCUM_FLUSH_S)
            # 醒来后二次确认：缓存非空（确实没人接着说）且仍在听（没被别的路径抢先处理）
            if self._asr_accum_texts and self.state == State.LISTENING:
                flushed = self._drain_accum()
                log.info("[Accum] Flush timer fired, sending buffered: '%s'", flushed)
                self.state = State.THINKING
                self._turn += 1
                await self._send({"type": "state", "state": "thinking"})
                # latency_ms=0（没跑 ASR）；accumulated=True 让前端/日志区分"兜底合并"与正常转写
                await self._send({"type": "asr", "text": flushed,
                                  "latency_ms": 0, "turn": self._turn, "accumulated": True})
                self._track_task(
                    self._run_pipeline_from_text(flushed, self._turn),
                    name=f"accum-pipeline-{self._turn}",
                )
        except asyncio.CancelledError:
            # 被 _cancel_accum_flush() 取消是设计内的正常路径（用户接着说或已完成合并）
            pass

    def _track_task(self, coro, *, name: str = "") -> asyncio.Task:
        """创建后台任务并纳入统一生命周期管理（强引用防 GC，完成时自动移出集合）。"""
        task = asyncio.create_task(coro, name=name)
        # asyncio 自身只持弱引用，不存一份会被 GC 提前回收
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def reset(self):
        """回到刚建连的干净状态（前端点"重置" / 复用时调用）。

        清空全部缓冲与计数、state 置回 IDLE、重置 VAD 与 LLM 内部状态、取消发言任务与投机 ASR；
        **不动 `_tasks`**（正在跑的 pipeline 不在这里取消）。`self._turn` 也清零，前端收到 reset_ack
        后会把 turn 显示重置为 0，两边对齐才能继续防串音。
        """
        self.state = State.IDLE
        self._audio_buffer = []
        self._interrupted_texts = []
        self._silence_count = 0
        self._turn = 0
        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False
        self._cancel_speculative_asr()
        # VAD 是流式模型（LSTM 隐状态），必须 reset，否则上一句的上下文会污染下一句
        self.vad.reset()
        self.llm.reset()
        if self._speaking_task and not self._speaking_task.done():
            self._speaking_task.cancel()
        # 注意：reset() 会连带取消声纹门控（vad 内部嵌入仍保留，但本类不再启用）；
        # 前端若想恢复门控需重新 enroll_speaker()
        self._speaker_enrolled = False
        self._idle_gate_confirm = 0
        self._idle_speech_ring = []
        self._idle_speech_samples = 0
        self._asr_accum_texts = []
        self._cancel_accum_flush()
        self._idle_since = time.monotonic()
        self._idle_chunk_count = 0
        self._idle_prompted = False
        self._idle_goodbye_sent = False
        self._consecutive_empty_asr = 0
        self._session_ending = False

    async def process_ptt(self, audio: np.ndarray):
        """按住说话（Push-to-Talk）演示模式：整段录音一次性送入，同步跑完 ASR→RAG→LLM→TTS。

        没有 VAD/语气词/打断路径（音频整段到达），保留降噪与错误兜底，且**不等待** LLM/TTS 结束就返回；
        内部捕获所有异常：任何失败都退化成"播一句道歉 + 回 IDLE"。
        """
        self._turn += 1
        # 局部快照：后续 await 期间 self._turn 可能被别的路径改动，用局部变量保证本轮消息 turn 一致
        turn = self._turn
        t_start = time.perf_counter()
        loop = asyncio.get_event_loop()

        self.state = State.THINKING
        await self._send({"type": "state", "state": "thinking"})

        try:
            # 太短的音频（<3 帧≈96ms）走降噪没有意义，纯浪费 CPU
            if self.denoiser and len(audio) > CHUNK_SAMPLES * 3:
                # DTLN 降噪是同步阻塞的，必须 run_in_executor，否则事件循环被占满、连 WebSocket 心跳都发不出去
                audio = await loop.run_in_executor(None, self.denoiser.process, audio)

            asr_result = await loop.run_in_executor(None, self.asr.transcribe, audio)
            user_text = _ASR_TAG_RE.sub("", asr_result["text"]).strip()
            asr_ms = asr_result["latency_ms"]
            await self._send({"type": "asr", "text": user_text,
                              "latency_ms": round(asr_ms, 1), "turn": turn})

            if not user_text or len(user_text) < 2:
                log.info("[PTT Turn %d] ASR empty, returning to IDLE", turn)
                self.state = State.IDLE
                await self._send({"type": "state", "state": "idle"})
                return

            rag_context = ""
            rag_ms = 0.0
            try:
                rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
                rag_context = rag_result["context"]
                rag_ms = rag_result["total_ms"]
            except Exception:
                pass
            await self._send({"type": "rag", "context": rag_context,
                              "latency_ms": round(rag_ms, 1), "turn": turn})

            # 清掉打断闸门：这是"开始播报"的固定前置动作，否则新音频会被立刻掐断
            self._cancel_speaking.clear()
            self.state = State.SPEAKING
            await self._send({"type": "state", "state": "speaking"})
            await self._stream_llm_tts(user_text, rag_context, turn, t_start, asr_ms, rag_ms)

        except Exception as e:
            log.error("[PTT Turn %d] Pipeline error: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            try:
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._play_system_tts(self._error_apology, turn)
            except Exception:
                pass
        finally:
            # 只要还停在 THINKING/SPEAKING 就回收成 IDLE；若已被打断转入 LISTENING，则保留用户状态
            if self.state in (State.THINKING, State.SPEAKING):
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                try:
                    await self._send({"type": "state", "state": "idle"})
                except Exception:
                    pass

    async def handle_barge_in(self):
        """前端手动打断（空格键/按钮，协议 `manual_barge_in`）：不需要 VAD 确认，一按就停。

        只在 SPEAKING / THINKING 生效（IDLE/LISTENING 下按键无效，误触发会把状态机搞乱）。
        副作用：置位 `_cancel_speaking`、切状态、下发 barge_in + state。
        """
        if self.state == State.SPEAKING and not self._cancel_speaking.is_set():
            self._cancel_speaking.set()
            self.state = State.LISTENING
            self._audio_buffer = []
            self._barge_confirm_count = 0
            self._silence_count = 0
            log.info("[Turn %d] Manual barge-in from client", self._turn)
            # source=manual 让前端/日志能区分自动与手动打断
            await self._send({"type": "barge_in", "turn": self._turn, "source": "manual"})
            await self._send({"type": "state", "state": "listening"})
        elif self.state == State.THINKING:
            self._cancel_speaking.set()
            self.state = State.LISTENING
            log.info("[Turn %d] Manual barge-in (was THINKING)", self._turn)
            await self._send({"type": "barge_in", "turn": self._turn, "source": "manual"})
            await self._send({"type": "state", "state": "listening"})
            # 注意：这里**不**清 _audio_buffer/_silence_count（与上面 SPEAKING 分支不同）：
            # THINKING 期间后到的语音可能已攒进缓冲，保留才不会丢掉用户刚说的话

    def shutdown(self):
        """连接断开时的收尾：置死标记、置位打断闸门、取消所有后台任务，最后 reset。

        本方法是**同步**的（ws_server 在 finally 里直接调用）：cancel() 只投递取消请求，
        任务真正退出由事件循环调度，所以此处不需要 await。
        """
        self._dead = True
        self._cancel_speaking.set()
        for t in list(self._tasks):
            if not t.done():
                t.cancel()
        self._tasks.clear()
        self.reset()

    def _cancel_speculative_asr(self):
        """取消投机 ASR 任务并清空结果缓存（幂等）：投机结果一旦与新音频不匹配就必须作废。"""
        # 只在任务还在跑时取消；已完成的再 cancel 会抛 InvalidStateError
        if self._spec_asr_task and not self._spec_asr_task.done():
            self._spec_asr_task.cancel()
        self._spec_asr_task = None
        # 结果缓存作废：音频已变，旧转写绝不能再被 _run_pipeline_inner 复用
        self._spec_asr_result = None
        self._spec_asr_audio_len = 0

    async def feed_audio(self, samples: np.ndarray):
        """把上游 PCM 切成固定 512 点（32ms）帧后逐帧喂给状态机。

        **必须切帧**：VAD 是流式模型，帧长不对齐会显著掉准确率，而网络分包不保证到达时正好是整帧；
        尾部不足一帧的样本补零后照样处理（丢尾样本在连续流里会累积成可听的断续）。
        """
        for i in range(0, len(samples), CHUNK_SAMPLES):
            chunk = samples[i:i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            # 串行处理：每帧都可能改变状态机，绝不能并行
            await self._process_chunk(chunk)

    async def _process_chunk(self, chunk: np.ndarray):
        """单帧处理的核心状态机：按当前 state 分派到五个分支。

        会改变 state 与各计数/缓冲、下发 state/barge_in 消息，并在 LISTENING 判停时启动 pipeline 后台任务。
        跑在事件循环线程；vad.process_chunk 是 CPU 同步调用（Silero 很小，单帧约 1ms），没有丢线程池。
        """
        vad_result = self.vad.process_chunk(chunk, SAMPLE_RATE)
        speech_prob = vad_result["speech_prob"]
        # 普通判停用的语音判定门限固定 0.6（打断用另一个更严的常量）
        is_speech = speech_prob >= 0.6
        rms = _rms(chunk)

        if self.state == State.IDLE:
            # 双门限：VAD 0.6 且 RMS > 0.015（经验值，约等于安静房间里正常说话的音量，滤掉风扇/空调底噪）
            if is_speech and rms > 0.015:
                if self._speaker_enrolled:
                    # speaker_vad 在未满验证窗口时返回 None 表示"还没算"，此时不该据此开门
                    has_speaker = vad_result.get("is_target_speaker") is not None
                    if has_speaker:
                        is_target = vad_result.get("is_target_speaker", False)
                        if is_target:
                            self._idle_gate_confirm += 1
                        self._idle_speech_ring.append(chunk.copy())
                        self._idle_speech_samples += len(chunk)
                        while self._idle_speech_samples > SPEAKER_GATE_BUFFER_CHUNKS * CHUNK_SAMPLES:
                            removed = self._idle_speech_ring.pop(0)
                            self._idle_speech_samples -= len(removed)
                        if self._idle_gate_confirm >= SPEAKER_GATE_CONFIRM_CHUNKS:
                            sim = vad_result.get("speaker_similarity", 0)
                            log.info("Speaker gate opened (confirm=%d, sim=%.2f)", self._idle_gate_confirm, sim)
                            self._idle_gate_confirm = 0
                            self._idle_speech_ring = []
                            self._idle_speech_samples = 0
                            self.state = State.LISTENING
                            # 注意只放当前帧：不把环形缓冲里的历史音频一起塞进去，避免把门外的杂音也转写
                            self._audio_buffer = [chunk]
                            self._silence_count = 0
                            self._idle_prompted = False
                            self._idle_goodbye_sent = False
                            self._consecutive_empty_asr = 0
                            self._cancel_speculative_asr()
                            await self._send({"type": "state", "state": "listening"})
                    else:
                        # 本帧还没有声纹结论（验证窗口未满）：什么都不做，继续收帧等 speaker_vad 每 3 帧给出的判定
                        pass
                else:
                    self.state = State.LISTENING
                    self._audio_buffer = [chunk]
                    self._silence_count = 0
                    self._idle_prompted = False
                    self._idle_goodbye_sent = False
                    self._consecutive_empty_asr = 0
                    self._cancel_speculative_asr()
                    await self._send({"type": "state", "state": "listening"})
            else:
                self._idle_gate_confirm = 0
                self._idle_chunk_count += 1
                if (self._idle_chunk_count >= IDLE_CHECK_INTERVAL_CHUNKS
                        and not self._idle_goodbye_sent
                        and not self._session_ending):
                    self._idle_chunk_count = 0
                    await self._check_idle_timeout()

        elif self.state == State.LISTENING:
            if is_speech:
                self._audio_buffer.append(chunk)
                self._silence_count = 0
                # 用户还在说，之前基于旧音频快照启动的投机 ASR 已经过期 → 取消
                if self._spec_asr_task:
                    self._cancel_speculative_asr()
                # 判停器内部也是流式状态：音频继续推进时清掉它的历史，避免把上一句的声学上下文带进本次判停
                if self.turn_detector:
                    self.turn_detector.reset()
            else:
                # 静音帧也进缓冲（ASR 需要句尾静音断句），并累计静音帧数用于判停
                self._audio_buffer.append(chunk)
                self._silence_count += 1

                # 静音刚满 160ms，且没启动过、音频确实有内容（>1 帧）→ 抢跑一次 ASR，结果先缓存着
                if (self._silence_count == SPECULATIVE_ASR_CHUNKS
                        and self._spec_asr_task is None
                        and len(self._audio_buffer) > CHUNK_SAMPLES):
                    self._launch_speculative_asr()

                # 有判停器时先问它是否说完；否则最多等到 ENDPOINT_SLOW_CHUNKS(640ms) 强制判停，防模型卡住不收尾
                if self.turn_detector and self._silence_count >= SPECULATIVE_ASR_CHUNKS:
                    if self._check_smart_turn():
                        await self._on_endpointing()
                    elif self._silence_count >= ENDPOINT_SLOW_CHUNKS:
                        await self._on_endpointing()
                else:
                    threshold = self._adaptive_endpoint_threshold()
                    if self._silence_count >= threshold:
                        await self._on_endpointing()

        elif self.state == State.SPEAKING:
            if not hasattr(self, '_speak_log_count'):
                self._speak_log_count = 0
            self._speak_log_count += 1
            if self._speak_log_count <= 5 or (speech_prob > 0.3 and self._speak_log_count % 10 == 0):
                log.info("[SPEAKING] chunk %d: vad=%.2f rms=%.4f threshold=%.2f/%.4f",
                         self._speak_log_count, speech_prob, rms,
                         BARGE_IN_VAD_THRESHOLD, BARGE_IN_RMS_THRESHOLD)

            if self._speaker_enrolled:
                is_real_speech = (
                    vad_result.get("is_target_speaker", False)
                    and rms > BARGE_IN_RMS_THRESHOLD
                )
            else:
                is_real_speech = (
                    speech_prob >= BARGE_IN_VAD_THRESHOLD
                    and rms > BARGE_IN_RMS_THRESHOLD
                )
            if is_real_speech:
                self._barge_confirm_count += 1
                if self._barge_confirm_count >= BARGE_IN_CONFIRM_CHUNKS:
                    if not self._cancel_speaking.is_set():
                        self._cancel_speaking.set()
                        self.state = State.INTERRUPTED
                        # 把触发打断的这帧作为新输入的第一帧，尽量不丢用户开头的话
                        self._audio_buffer = [chunk]
                        self._barge_confirm_count = 0
                        # 通知前端清播放队列（stopPlayback + playableTurn=0），这是"听感上立刻闭嘴"的关键；无 source 字段=自动打断
                        await self._send({"type": "barge_in", "turn": self._turn})
                        await self._send({"type": "state", "state": "interrupted"})
                    else:
                        self._audio_buffer.append(chunk)
            else:
                self._barge_confirm_count = 0

        elif self.state == State.INTERRUPTED:
            if is_speech:
                self._audio_buffer.append(chunk)
                self._silence_count = 0
            else:
                self._audio_buffer.append(chunk)
                self._silence_count += 1
                if self._silence_count >= INTERRUPT_SILENCE_CHUNKS:
                    await self._on_endpointing()

        elif self.state == State.THINKING:
            is_accepted = is_speech and rms > 0.015
            if self._speaker_enrolled and is_accepted:
                is_accepted = vad_result.get("is_target_speaker", False)
            if is_accepted:
                self._thinking_extra_buffer.append(chunk)
                self._thinking_has_speech = True
            elif self._thinking_has_speech:
                self._thinking_extra_buffer.append(chunk)

    def _launch_speculative_asr(self):
        """在判停前用当前音频快照跑一次 ASR（fire-and-forget），结果缓存供 pipeline 复用。

        优先用 spec_asr（Moonshine 27M CPU ~20ms），没有就退回主 ASR。副作用：concat 一份音频快照
        （额外一次内存拷贝），起后台任务 spec-asr-{turn}（线程池跑阻塞转写），句柄写进 `_spec_asr_task`、
        快照长度写进 `_spec_asr_audio_len`。不 await，不阻塞音频帧处理。
        """
        # 必须在后台任务启动前复制快照，否则后续 _audio_buffer.clear() 会让后台任务读到空列表
        audio_snapshot = np.concatenate(self._audio_buffer) if self._audio_buffer else None
        if audio_snapshot is None or len(audio_snapshot) < CHUNK_SAMPLES * 3:
            return
        self._spec_asr_audio_len = len(audio_snapshot)
        asr_engine = self.spec_asr if self.spec_asr else self.asr

        async def _run():
            """在线程池里跑一次阻塞转写，结果（或失败时的 None）写回 `_spec_asr_result`。"""
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, asr_engine.transcribe, audio_snapshot)
                self._spec_asr_result = result
            except Exception:
                self._spec_asr_result = None

        self._spec_asr_task = self._track_task(_run(), name=f"spec-asr-{self._turn}")
        engine_name = "Moonshine" if self.spec_asr else "SenseVoice"
        log.debug("Speculative ASR (%s) launched on %d samples", engine_name, len(audio_snapshot))

    def _check_smart_turn(self) -> bool:
        """问语义判停器（Pipecat Smart Turn v3）用户是否说完了：True=可以判停。

        它看韵律线索（语调下降=说完），比数静音帧更准，能减少"用户只是想一下就被抢话"。
        副作用：整段缓冲音频喂给 ONNX 模型（同步推理，几十毫秒 CPU，会短暂占用事件循环）。
        """
        if not self.turn_detector or len(self._audio_buffer) < 3:
            return False
        try:
            # is_speech_active 固定传 False：此刻我们已在静音分支里，语义上就是"当前无语音"
            return self.turn_detector.should_endpoint(self._audio_buffer, is_speech_active=False)
        except Exception:
            return False

    def _adaptive_endpoint_threshold(self) -> int:
        """按本次话的长度自适应选择判停所需的静音帧数（返回 ENDPOINT_* 常量之一）：短句更快判停、
        长句更慢，避免一句话被从中间切断。

        注意：第一个分支返回的是 ENDPOINT_SLOW_CHUNKS 而不是 ENDPOINT_FAST_CHUNKS——`ENDPOINT_FAST_CHUNKS`
        目前**未被使用**，推测是按下限保护而有意为之，也可能是笔误；按作业规范不改代码，仅在注释中标出风险。
        """
        audio_duration_ms = len(self._audio_buffer) * CHUNK_MS
        # 短话：返回 SLOW 档（见 docstring 的注意项，疑似应为 FAST）
        if audio_duration_ms < 500:
            return ENDPOINT_SLOW_CHUNKS
        elif audio_duration_ms > 3000:
            return ENDPOINT_SLOW_CHUNKS
        else:
            return ENDPOINT_DEFAULT_CHUNKS

    async def _on_endpointing(self):
        """判停：把本轮音频冻结成一段、起 pipeline 后台任务，并立即切到 THINKING。

        副作用：取消累积器定时器；复位 `_silence_count`/`_barge_confirm_count`/THINKING 后到语音缓冲；
        自增 `_turn`；下发 state=thinking（长句时先发一句 filler）；把投机 ASR 的结果/长度/句柄摘走转移给 pipeline。
        这里**不 await pipeline**：否则 feed_audio 会被整条 LLM+TTS 链路阻塞，用户说话没人收集。
        """
        # 本轮音频即将定稿，缓冲器的兜底定时器必须取消（否则 3s 后会把旧文本再发一轮）
        self._cancel_accum_flush()
        self.state = State.THINKING
        self._silence_count = 0
        self._barge_confirm_count = 0
        self._thinking_extra_buffer = []
        self._thinking_has_speech = False
        self._turn += 1
        await self._send({"type": "state", "state": "thinking"})

        audio_duration_ms = len(self._audio_buffer) * CHUNK_MS
        # 累积器里已有待合并短文本时马上还会有 pipeline 在跑，再插 filler 会显得语无伦次，故跳过
        has_pending_accum = len(self._asr_accum_texts) > 0
        if self.filler and audio_duration_ms > 1000 and not has_pending_accum:
            await self._send_filler()

        audio = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.float32)
        self._audio_buffer = []

        # "摘走"投机 ASR 的结果/长度/句柄（局部变量持有）并复位实例字段，避免与新一轮冲突
        spec_result = self._spec_asr_result
        spec_audio_len = self._spec_asr_audio_len
        spec_task = self._spec_asr_task
        self._spec_asr_result = None
        self._spec_asr_task = None
        self._spec_asr_audio_len = 0

        self._track_task(
            self._run_pipeline(audio, self._turn, spec_result, spec_audio_len, spec_task),
            name=f"pipeline-{self._turn}",
        )

    async def _run_pipeline(self, audio: np.ndarray, turn: int,
                            spec_result: Optional[dict], spec_audio_len: int,
                            spec_task: Optional[asyncio.Task]):
        """一轮完整链路的崩溃安全外壳：ASR →（可选中段合并）→ RAG → LLM → TTS。

        所有异常都在这里兜住，保证任何组件挂掉时用户听到的是一句道歉而不是静默；CancelledError
        （断连/被取消）只记日志。`spec_task` 会在结果还没好时最多再等 200ms。finally 里若仍停在
        THINKING 则回收为 IDLE 并重置空闲计时。
        """
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        try:
            await self._run_pipeline_inner(
                audio, turn, spec_result, spec_audio_len, spec_task,
                loop, t_start,
            )
        except asyncio.CancelledError:
            log.info("[Turn %d] Pipeline cancelled (disconnect?)", turn)
        except Exception as e:
            log.error("[Turn %d] Pipeline CRASHED: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            try:
                await self._send({"type": "error", "message": str(e), "turn": turn})
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._play_system_tts(self._error_apology, turn)
            except Exception:
                pass
        finally:
            # 只有还停在 THINKING（即 inner 里没走到 SPEAKING）才需要回收状态：
            # v2.9 的 G1 修复点——**不能**无条件把 SPEAKING 改回 IDLE，否则 _stream_llm_tts
            # 还在播报时状态就被提前清掉，导致打断判定失效/状态竞争
            if self.state == State.THINKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                try:
                    await self._send({"type": "state", "state": "idle"})
                except Exception:
                    pass

    async def _run_pipeline_from_text(self, user_text: str, turn: int):
        """文本路径的轻量 pipeline（累积器兜底用）：没有音频，跳过降噪/ASR/Captioner，只做 RAG → LLM → TTS。

        命中告别词则置 `_session_ending`；finally 里若仍停在 THINKING 则回收 IDLE；不向外抛异常。
        """
        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()
        try:
            self._consecutive_empty_asr = 0

            if any(kw in user_text for kw in self._farewell_keywords):
                self._session_ending = True

            rag_context = ""
            try:
                rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
                rag_context = rag_result["context"]
            except Exception:
                pass

            self._cancel_speaking.clear()
            self._barge_confirm_count = 0
            self._speak_log_count = 0
            self.state = State.SPEAKING
            await self._send({"type": "state", "state": "speaking"})
            await self._stream_llm_tts(user_text, rag_context, turn, t_start, 0, 0)
        except asyncio.CancelledError:
            log.info("[Turn %d] Accum pipeline cancelled", turn)
        except Exception as e:
            log.error("[Turn %d] Accum pipeline CRASHED: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            try:
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._play_system_tts(self._error_apology, turn)
            except Exception:
                pass
        finally:
            if self.state == State.THINKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                try:
                    await self._send({"type": "state", "state": "idle"})
                except Exception:
                    pass

    async def _run_pipeline_inner(self, audio, turn, spec_result, spec_audio_len,
                                  spec_task, loop, t_start):
        """真正的链路实现，按编号 0~6 顺序执行（保留原编号便于对照日志）：

        0 降噪 → 1 ASR（可复用投机结果）→ 1c 合并 THINKING 后的后到语音 → ASR 累积器 → D3 空结果 →
        D4 告别 → 2 打断文本聚合 → 3 Captioner → 4 RAG → 5 注入观察 → 6 LLM 流式 → TTS。
        **不吞异常**：任何失败都向上抛给 `_run_pipeline` 的兜底逻辑（让崩溃处理只有一处）。
        """
        if self.denoiser and len(audio) > CHUNK_SAMPLES * 3:
            audio = await loop.run_in_executor(None, self.denoiser.process, audio)
            log.info("[Turn %d] Denoised %d samples", turn, len(audio))

        # 音频是否"显著变长"（>15%）：投机结果是在静音刚满 160ms 时跑的，用户之后又说了不少内容就必须重跑；spec_audio_len=0 时强制重跑
        audio_grew = len(audio) > spec_audio_len * 1.15 if spec_audio_len > 0 else True

        # 投机任务可能还在跑：最多再等 200ms（用户已等了一整段静音，不能再赌太久），超时/被取消就走正常 ASR
        if spec_task and not spec_task.done():
            try:
                await asyncio.wait_for(spec_task, timeout=0.2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                spec_result = None

        if spec_result and not audio_grew:
            user_text = _ASR_TAG_RE.sub("", spec_result["text"]).strip()
            asr_ms = spec_result["latency_ms"]
            log.info("[Turn %d] Used speculative ASR (saved ~%dms)", turn, round(asr_ms))
            # 计入首响的 ASR 时间记为 0：这段耗时发生在判停之前，不在用户感知的"说完→出声"区间内
            asr_ms = 0.0
        else:
            try:
                asr_result = await loop.run_in_executor(None, self.asr.transcribe, audio)
                user_text = _ASR_TAG_RE.sub("", asr_result["text"]).strip()
                asr_ms = asr_result["latency_ms"]
            except Exception as e:
                # ASR 引擎整个挂了（而非只返回空文本）：本轮无法继续，用"没太听清"话术让用户重说
                log.error("[Turn %d] ASR failed: %s", turn, e)
                self.metrics.record_error()
                self._cancel_speaking.clear()
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._send({"type": "system_message", "text": self._asr_empty_text, "turn": turn})
                await self._play_system_tts(self._asr_empty_text, turn)
                return

        # THINKING 期间接续说的话单独转写后拼进同一句，这样"我想问…那个…价格"不会被拆成两轮
        if self._thinking_has_speech and self._thinking_extra_buffer:
            extra_audio = np.concatenate(self._thinking_extra_buffer)
            self._thinking_extra_buffer = []
            self._thinking_has_speech = False
            if len(extra_audio) > CHUNK_SAMPLES * 3:
                try:
                    extra_result = await loop.run_in_executor(None, self.asr.transcribe, extra_audio)
                    extra_text = _ASR_TAG_RE.sub("", extra_result["text"]).strip()
                    if extra_text and len(extra_text) >= 2:
                        user_text = f"{user_text}，{extra_text}" if user_text else extra_text
                        asr_ms += extra_result["latency_ms"]
                        log.info("[Turn %d] Merged late speech: '%s'", turn, extra_text)
                except Exception as e:
                    # 补充转写失败只告警：主文本已拿到，不影响本轮对话
                    log.warning("[Turn %d] Late ASR merge failed: %s", turn, e)

        await self._send({"type": "asr", "text": user_text, "latency_ms": round(asr_ms, 1), "turn": turn})

        if user_text and self._is_short_utterance(user_text) and len(self._asr_accum_texts) < ASR_ACCUM_MAX_SEGMENTS:
            self._asr_accum_texts.append(user_text)
            log.info("[Turn %d] ASR buffered ('%s', total=%d), returning to LISTENING",
                     turn, user_text, len(self._asr_accum_texts))
            self.state = State.LISTENING
            self._audio_buffer = []
            self._silence_count = 0
            await self._send({"type": "state", "state": "listening"})
            self._cancel_accum_flush()
            self._asr_accum_flush_task = self._track_task(
                self._accum_flush_timer(), name=f"accum-flush-{turn}")
            return

        if self._asr_accum_texts:
            prefix = "，".join(self._asr_accum_texts)
            user_text = f"{prefix}，{user_text}" if user_text else prefix
            self._asr_accum_texts = []
            # 已合并下发，定时器必须取消，否则 3s 后同一段文本会被再发一轮
            self._cancel_accum_flush()
            log.info("[Turn %d] Merged accumulated ASR: '%s'", turn, user_text)

        # 空结果恢复：前几次静默回 LISTENING（用户可能只是清嗓子），连续达 _max_empty_asr(3) 次才出声提示，再空则建议打字
        if not user_text or len(user_text) < 2:
            self._consecutive_empty_asr += 1
            if self._consecutive_empty_asr >= self._max_empty_asr:
                recovery = self._asr_noisy_text
                log.info("[Turn %d] ASR empty (#%d), recovery: '%s'", turn, self._consecutive_empty_asr, recovery)
                self._cancel_speaking.clear()
                self._barge_confirm_count = 0
                self._speak_log_count = 0
                self.state = State.SPEAKING
                await self._send({"type": "state", "state": "speaking"})
                await self._send({"type": "system_message", "text": recovery, "turn": turn})
                await self._play_system_tts(recovery, turn)
            else:
                log.info("[Turn %d] ASR too short ('%s', #%d), silent return to LISTENING",
                         turn, user_text or "", self._consecutive_empty_asr)
                self.state = State.LISTENING
                self._audio_buffer = []
                self._silence_count = 0
                await self._send({"type": "state", "state": "listening"})
            # 两条分支都到此结束本轮（finally 里 state 是 LISTENING 或 SPEAKING，不回收）
            return

        self._consecutive_empty_asr = 0

        # 命中告别词：标记本轮播完后结束会话（session_end 在 _stream_llm_tts 的 finally 里发）
        if any(kw in user_text for kw in self._farewell_keywords):
            self._session_ending = True
            log.info("[Turn %d] Farewell detected in: '%s'", turn, user_text)

        # 把被打断时存下的 AI 半句与用户新话一起交给 LLM，并明确告知"用户依次说了…"，让模型综合回应而非重复被自己打断的话
        if self._interrupted_texts:
            combined = "；".join(self._interrupted_texts + [user_text])
            user_text = f"用户依次说了：{combined}。请自然地综合回应。"
            self._interrupted_texts = []
        else:
            # asr_ms > 0 表示跑了真实 ASR；此处清空是幂等的防御性写法，保证没有残留的"被打断文本"进入下一轮
            if asr_ms > 0:
                self._interrupted_texts = []

        # 副语言观察（笑声/叹气/背景人声）注入 LLM 让回复更贴合语气；属"锦上添花"，失败必须静默降级
        caption = ""
        caption_ms = 0.0
        if self.captioner and len(audio) >= SAMPLE_RATE:
            try:
                t_cap = time.perf_counter()
                # 0.5s 硬超时：Captioner 可能是远端 Omni 模型，绝不能让"可选增强"拖垮 600ms 首响预算
                caption = await asyncio.wait_for(
                    loop.run_in_executor(None, self.captioner.describe, audio),
                    timeout=0.5
                )
                caption_ms = (time.perf_counter() - t_cap) * 1000
                log.info("[Turn %d] Captioner: '%s' (%.0fms)", turn, caption, caption_ms)
                await self._send({
                    "type": "caption", "text": caption,
                    "latency_ms": round(caption_ms, 1), "turn": turn,
                })
            except (asyncio.TimeoutError, Exception) as e:
                log.warning("[Turn %d] Captioner failed/timeout: %s", turn, e)
                caption = ""

        rag_context = ""
        rag_ms = 0.0
        try:
            rag_result = await loop.run_in_executor(None, self.rag.get_context, user_text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]
        except Exception as e:
            log.warning("[Turn %d] RAG failed (degraded to no-context): %s", turn, e)
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": turn,
        })

        # LLM 只看文本，只能用自然语言提示把副语言信息传进去
        llm_user_text = user_text
        if caption:
            llm_user_text = f"[语音观察：{caption}]\n用户说：{user_text}"

        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        # v2.9 G1 修复点：播报任务挂到 _speaking_task（供 reset 取消），并且**必须 await 它**——
        # 否则 _run_pipeline 会提前走完 finally，在 TTS 还在播时就把状态机复位
        self._speaking_task = self._track_task(
            self._stream_llm_tts(llm_user_text, rag_context, turn, t_start, asr_ms, rag_ms),
            name=f"llm-tts-{turn}",
        )
        await self._speaking_task

    async def _stream_llm_tts(self, user_text, rag_context, turn, t_start, asr_ms, rag_ms):
        """整条链路的"流式"核心：LLM 每凑齐一句就立刻送 TTS，TTS 每生成一块就立刻下发（首句优先出声）。

        副作用：在线程池跑 `llm.stream_sentences`（生产者线程）与 `tts.synthesize_stream`；下发
        llm_sentence/metrics/audio_start/裸音频/llm 消息；被打断时把已生成的部分回复写进 `_interrupted_texts`
        供下轮引用；finally 里回收状态为 IDLE、必要时发 session_end。
        不向外抛异常：LLM 线程的异常通过队列里的 `{"_error": ...}` 标记回传后在此抛出并处理。
        """
        loop = asyncio.get_event_loop()
        # 有界句子队列（maxsize=8，v2.9 G7 修复）：无界队列会让 LLM 生产远超消费速度、句子在内存里堆积；
        # 有界队列让生产者线程在 put 时阻塞，形成"线程侧背压"，TTS 消费多快就生成多快
        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        full_response = ""
        sentence_count = 0
        first_ttfa = None
        llm_first_ms = 0.0
        was_interrupted = False

        def _produce_sentences():
            """生产者：在线程池线程里跑阻塞的 LLM 流式生成，把句子投进队列。

            这个函数**不在事件循环线程**里：不能直接 await，只能用 run_coroutine_threadsafe 投递协程；
            判断打断用 `self._cancel_speaking.is_set()`（Event 读操作线程安全）。副作用：写 LLM 对话历史、
            投递句子 dict 与异常标记 `{"_error": ...}`。finally 保证投一个 None 哨兵，否则消费端要等到 35s 超时。
            """
            try:
                for s in self.llm.stream_sentences(user_text, rag_context=rag_context):
                    # 用户已打断：不再让 LLM 继续产出（尽早省 GPU，也避免污染历史）
                    if self._cancel_speaking.is_set():
                        break
                    fut = asyncio.run_coroutine_threadsafe(sentence_queue.put(s), loop)
                    # 最多等 30s：队列被下游卡死时不能让线程永远阻塞
                    fut.result(timeout=30)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    sentence_queue.put({"_error": str(e)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(sentence_queue.put(None), loop)

        def _tts_stream_sentence(text):
            """把单句文本合成 TTS 音频块，返回列表（给 run_in_executor 的同步目标）。

            返回 `[{"audio": np.ndarray(float32, 44.1kHz), "chunk_idx", "ttfa_ms"}, ...]`；首块已由 TTS 内部
            丢弃（VoxCPM VAE 启动瞬态），`ttfa_ms` 挂在第 2 块上。list(...) 一次性抽干生成器，同步遍历最省切换。
            """
            return list(self.tts.synthesize_stream(text))

        producer_future = loop.run_in_executor(None, _produce_sentences)

        try:
            while True:
                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    log.info("[Turn %d] Cancelled before next sentence", turn)
                    break

                # 等下一句：35s 是"LLM 彻底卡死"的兜底（正常 SSE 间隔远小于此）
                try:
                    s = await asyncio.wait_for(sentence_queue.get(), timeout=35.0)
                except asyncio.TimeoutError:
                    log.error("[Turn %d] LLM sentence queue timeout (35s)", turn)
                    self.metrics.record_error()
                    break

                if s is None:
                    break
                if "_error" in s:
                    raise RuntimeError(f"LLM stream error: {s['_error']}")

                # 取到句子后再查一次打断：等待期间刚被打断就不要发这一句（避免"打断后又冒出一句"）
                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    break

                sentence = s["sentence"]
                full_response += sentence
                sentence_count += 1
                # 只有第一句带真实 ttfs_ms（后续句子的 ttfs_ms 在 llm.py 里就是 0）
                if sentence_count == 1:
                    llm_first_ms = s["ttfs_ms"]

                await self._send({
                    "type": "llm_sentence", "text": sentence,
                    "latency_ms": round(s["ttfs_ms"], 1) if sentence_count == 1 else 0,
                    "sentence_idx": sentence_count - 1, "turn": turn,
                })

                # 合成前的第三次打断检查：把一次没必要的 TTS 合成省掉（GPU 最贵的一步）
                if self._cancel_speaking.is_set():
                    was_interrupted = True
                    break

                tts_chunks = await loop.run_in_executor(None, _tts_stream_sentence, sentence)

                for i, chunk_data in enumerate(tts_chunks):
                    # 一次合成可能产出几十块，逐块检查才能"按毫秒级闭嘴"
                    if self._cancel_speaking.is_set():
                        was_interrupted = True
                        log.info("[Turn %d] Cancelled mid-TTS chunk %d/%d", turn, i, len(tts_chunks))
                        break

                    # 第一句的第一块：这是"AI 真正开口"的瞬间，首响指标在这里落账
                    if sentence_count == 1 and i == 0:
                        first_ttfa = chunk_data.get("ttfa_ms", 0)
                        # audio_start 让前端把 playableTurn 指向本 turn；在此之前收到的音频帧会被忽略
                        await self._send({"type": "audio_start", "turn": turn})
                        # 端到端首响 = ASR + RAG + LLM 首句 + TTS 首帧（RAG 与 LLM 实为串行，这里只是相加用于观测）
                        first_response_ms = asr_ms + rag_ms + llm_first_ms + first_ttfa
                        self.metrics.record_turn(
                            asr_ms=asr_ms, llm_ms=llm_first_ms,
                            tts_ms=first_ttfa, fr_ms=first_response_ms,
                        )
                        await self._send({
                            "type": "metrics",
                            "asr_ms": round(asr_ms, 1),
                            "rag_ms": round(rag_ms, 1),
                            "llm_ms": round(llm_first_ms, 1),
                            "tts_ttfa_ms": round(first_ttfa, 1),
                            "first_response_ms": round(first_response_ms, 1),
                            "turn": turn,
                        })

                    audio = chunk_data["audio"]
                    if audio.dtype != np.int16:
                        audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                    raw = audio.tobytes()
                    # 单块硬上限保护：VoxCPM 生成退化时一块可能巨大、会撑爆前端缓冲；截断比丢弃安全（丢块会造成爆破音）
                    if len(raw) > self.MAX_TTS_AUDIO_BYTES:
                        log.warning("[Turn %d] TTS chunk oversized (%d bytes) — truncating",
                                    turn, len(raw))
                        raw = raw[:self.MAX_TTS_AUDIO_BYTES]
                    await self._send({"_audio": raw})
                    await asyncio.sleep(0.01)

                if sentence_count >= 3:
                    log.info("[Turn %d] Sentence cap (3) reached, stopping", turn)
                    break

            await self._send({
                "type": "llm", "text": full_response,
                "turn": turn, "sentences": sentence_count,
            })

        except asyncio.CancelledError:
            was_interrupted = True
            log.info("[Turn %d] Pipeline cancelled", turn)
        except Exception as e:
            # 注意这里**不**播道歉：外层 _run_pipeline 负责，避免重复播
            log.error("[Turn %d] LLM/TTS error: %s", turn, e, exc_info=True)
            self.metrics.record_error()
            await self._send({"type": "error", "message": str(e), "turn": turn})
        finally:
            # 收尾生产者线程：最多等 2s，超时/异常都忽略（线程池的残留工作不影响状态机正确性）
            try:
                await asyncio.wait_for(asyncio.wrap_future(producer_future), timeout=2.0)
            except Exception:
                pass

            # 被打断且已说出内容：存起来给下一轮当上下文，下一句才是"接着上句说"而不是重头再来
            if was_interrupted and full_response:
                self._interrupted_texts.append(full_response)
                log.info("[Turn %d] Saved partial AI response (%d chars) for context",
                         turn, len(full_response))

            # 只在仍处于 SPEAKING 时回收：已被打断转入 INTERRUPTED/LISTENING 时必须保留那个状态
            if self.state == State.SPEAKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                await self._send({"type": "state", "state": "idle"})
            if self._session_ending:
                self._session_ending = False
                await self._send({"type": "session_end", "reason": "farewell"})

    MAX_TTS_AUDIO_BYTES = 44100 * 2 * 30  # 30s at 44.1kHz 16-bit — hard safety cap

    async def _send_audio(self, audio_bytes: bytes):
        """把一段完整音频分片下发（每片 TTS_SEND_CHUNK_BYTES≈200ms），分片间检查打断。

        超长先截断到 MAX_TTS_AUDIO_BYTES 并告警；每片 `await asyncio.sleep(0)` 让出事件循环。
        注意：本方法在当前代码里**没有调用点**（`_stream_llm_tts` 与 `_play_system_tts` 都自己逐块发）。
        """
        if len(audio_bytes) > self.MAX_TTS_AUDIO_BYTES:
            log.warning("Audio too large (%d bytes, %.1fs) — truncating to 30s",
                        len(audio_bytes), len(audio_bytes) / 44100 / 2)
            audio_bytes = audio_bytes[:self.MAX_TTS_AUDIO_BYTES]
        for i in range(0, len(audio_bytes), TTS_SEND_CHUNK_BYTES):
            # 每片前查打断：置位就立刻停止后续下发（前端 barge_in 消息会清它的播放队列）
            if self._cancel_speaking.is_set():
                return
            await self._send({"_audio": audio_bytes[i:i + TTS_SEND_CHUNK_BYTES]})
            # sleep(0) 是"最小让出"：把控制权交给事件循环处理其它就绪回调（如收音频帧）
            await asyncio.sleep(0)

    async def _send_filler(self):
        """在 THINKING 期间下发一条预缓存的语气词（"嗯/好的"），盖住 RAG+LLM 首响的静音空档。

        音频在服务启动时就合成好了（见 engine/filler.py），这里是纯内存+发送。副作用：调
        `filler.get_filler()`（会自增轮询游标）；下发 filler/audio_start/裸音频。全捕获，失败只告警。
        """
        try:
            filler_text, filler_bytes = self.filler.get_filler()
            if filler_bytes:
                # 2s 硬上限：超长 filler 会压到正式回答上变成串音（与 filler.py 的 MAX_FILLER_SEC 呼应）
                max_filler = 44100 * 2 * 2  # 2s hard cap for fillers
                if len(filler_bytes) > max_filler:
                    log.warning("[Turn %d] Filler '%s' oversized (%d bytes) — truncating",
                                self._turn, filler_text, len(filler_bytes))
                    filler_bytes = filler_bytes[:max_filler]
                await self._send({"type": "filler", "text": filler_text, "turn": self._turn})
                # audio_start 让前端把 playableTurn 指向当前 turn，随后的音频才会被播放
                await self._send({"type": "audio_start", "turn": self._turn})
                await self._send({"_audio": filler_bytes})
                log.info("[Turn %d] Filler sent: '%s' (%d bytes)", self._turn, filler_text, len(filler_bytes))
        except Exception as e:
            log.warning("Filler send failed: %s", e)

    async def handle_text_input(self, text: str):
        """处理前端打字输入（协议 `text_input`）：跳过 VAD/ASR/降噪，直接走 RAG → LLM → TTS。

        副作用：自增 `_turn`、状态置 THINKING→SPEAKING；下发 state/asr/rag；线程池跑 RAG；起
        `text-tts-{turn}` 后台任务播报且**不 await**（ws_server 的消息循环要立刻回来继续收消息，用户打字时
        也可能按打断），代价是返回时播报还在进行，状态由任务自己回收。G2 修复：RAG 失败降级为无上下文。
        """
        self._turn += 1
        self.state = State.THINKING
        await self._send({"type": "state", "state": "thinking"})
        await self._send({"type": "asr", "text": text, "latency_ms": 0, "turn": self._turn, "source": "text"})

        self._consecutive_empty_asr = 0

        # D4：打字说"再见"也要能正常告别（否则用户只能靠说话结束通话）
        if any(kw in text for kw in self._farewell_keywords):
            self._session_ending = True
            log.info("[Turn %d] Farewell detected (text): '%s'", self._turn, text)

        loop = asyncio.get_event_loop()
        t_start = time.perf_counter()

        rag_context = ""
        rag_ms = 0.0
        try:
            rag_result = await loop.run_in_executor(None, self.rag.get_context, text)
            rag_context = rag_result["context"]
            rag_ms = rag_result["total_ms"]
        except Exception as e:
            log.warning("[Turn %d] RAG failed in text_input (degraded): %s", self._turn, e)
        await self._send({
            "type": "rag", "context": rag_context,
            "latency_ms": round(rag_ms, 1), "turn": self._turn,
        })

        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self.state = State.SPEAKING
        await self._send({"type": "state", "state": "speaking"})

        self._speaking_task = self._track_task(
            self._stream_llm_tts(text, rag_context, self._turn, t_start, 0, rag_ms),
            name=f"text-tts-{self._turn}",
        )

    async def start_greeting(self):
        """D1 开场白：连接建立后由 ws_server 以任务方式启动，主动播报一句问候（期间可被打断）。

        未配置开场白（空字符串）时只重置空闲计时基准并返回；否则自增 turn、置 SPEAKING、下发
        state/system_message，并同步 await `_play_system_tts` 播完（本方法通常作为后台任务运行）。
        """
        if not self._greeting_text:
            self._idle_since = time.monotonic()
            return
        log.info("Playing greeting: '%s'", self._greeting_text)
        # 开播前置：清打断闸门 + 清连击 + 清 SPEAKING 日志计数（让前 5 帧的打断判定日志重新打印）
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self._speak_log_count = 0
        self.state = State.SPEAKING
        self._turn += 1
        await self._send({"type": "state", "state": "speaking"})
        await self._send({"type": "system_message", "text": self._greeting_text, "turn": self._turn})
        await self._play_system_tts(self._greeting_text, self._turn)

    async def _check_idle_timeout(self):
        """D2 空闲超时检查：先提示"您还在吗"，再超时则告别并结束会话。

        由 `_process_chunk` 在 IDLE 且无语音时每约 1 秒调用一次（搭在音频帧流上的轮询，不是定时器）。
        第一次到 `_idle_timeout_s`(15s) 播提示；已提示过且到 `_idle_goodbye_s`(30s) 则播告别并带
        session_end_reason="idle_timeout"。两个阈值都以 `_idle_since` 为基准的绝对时间。
        """
        elapsed = time.monotonic() - self._idle_since
        if not self._idle_prompted and elapsed >= self._idle_timeout_s:
            self._idle_prompted = True
            log.info("Idle timeout (%.0fs), prompting user", elapsed)
            await self._speak_system(self._idle_prompt_text)
        elif self._idle_prompted and not self._idle_goodbye_sent and elapsed >= self._idle_goodbye_s:
            self._idle_goodbye_sent = True
            self._session_ending = True
            log.info("Idle goodbye (%.0fs), ending session", elapsed)
            await self._speak_system(self._idle_goodbye_text, session_end_reason="idle_timeout")

    async def _speak_system(self, text: str, session_end_reason: str = None):
        """下发一条系统话术并把播报挂成后台任务（不 await，调用方立即返回）。

        `session_end_reason` 非 None 时，播完后由 `_play_system_tts` 发 `{"type":"session_end","reason":...}`。
        与 `_play_system_tts` 的区别：后者是"播"（同步等待），本方法是"安排播"，用于不能阻塞音频帧处理的路径。
        """
        self._cancel_speaking.clear()
        self._barge_confirm_count = 0
        self._speak_log_count = 0
        self.state = State.SPEAKING
        self._turn += 1
        await self._send({"type": "state", "state": "speaking"})
        await self._send({"type": "system_message", "text": text, "turn": self._turn})
        # 挂后台任务：本方法在音频帧处理路径上被调用，绝不能在这里 await 播报
        self._speaking_task = self._track_task(
            self._play_system_tts(text, self._turn, session_end_reason),
            name=f"system-tts-{self._turn}",
        )

    async def _play_system_tts(self, text: str, turn: int, session_end_reason: str = None):
        """系统话术（D1 开场白 / D2 超时 / D3 错误恢复 / 崩溃道歉）的统一播报实现。

        与 `_stream_llm_tts` 的区别：文本既定、不需要 LLM，整个合成是一次阻塞调用，拿到全部块后再逐块下发。
        副作用：线程池跑完整 TTS 合成；下发 audio_start 与裸音频；finally 里从 SPEAKING 回收为 IDLE、
        重置空闲计时、可能发 session_end。CancelledError 静默吞掉，其它异常只记日志。
        注意 finally 不区分是否被打断：被打断时 `_cancel_speaking` 已置位，因此**不**发 session_end。
        """
        loop = asyncio.get_event_loop()
        try:
            def _gen():
                """在线程池线程里把整段文本合成完并返回音频块列表。"""
                return list(self.tts.synthesize_stream(text))

            tts_chunks = await loop.run_in_executor(None, _gen)
            await self._send({"type": "audio_start", "turn": turn})

            for i, chunk_data in enumerate(tts_chunks):
                if self._cancel_speaking.is_set():
                    log.info("[Turn %d] System message interrupted at chunk %d", turn, i)
                    break
                audio = chunk_data["audio"]
                if audio.dtype != np.int16:
                    audio = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                await self._send({"_audio": audio.tobytes()})
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("[Turn %d] System TTS error: %s", turn, e)
        finally:
            if self.state == State.SPEAKING:
                self.state = State.IDLE
                self._idle_since = time.monotonic()
                await self._send({"type": "state", "state": "idle"})
            # 有结束原因且**没有**被打断时才发 session_end：用户打断一句告别说明他还想继续聊
            if session_end_reason and not self._cancel_speaking.is_set():
                await self._send({"type": "session_end", "reason": session_end_reason})
