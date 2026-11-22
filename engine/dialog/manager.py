"""对话状态机门面 —— 吃 32ms 音频帧/文本输入，驱动 VAD→ASR→RAG→LLM→TTS 级联链路。

每个 WebSocket 连接一份实例；只决定"什么时候做什么"，重活全部委托给注入的引擎。

## 状态流转
```
IDLE --VAD检出--> LISTENING --静音达阈值--> THINKING --首句就绪--> SPEAKING --播完--> IDLE
SPEAKING --VAD 自动打断--> INTERRUPTED --静音≥320ms--> THINKING
SPEAKING / THINKING --手动打断(manual_barge_in)--> LISTENING
THINKING --用户又开口--> 追加进同一轮(_thinking_extra_buffer)，不打断
```
注意两条打断路径去向不同：**VAD 自动打断**进 INTERRUPTED（还要再等一段静音才继续处理），
**手动打断**（前端空格键/按钮）直接回 LISTENING——按键是明确的"我要说话"，不必再等。

## 三条主线
1. **turn 序列号防串音**：所有下行消息带 turn，前端靠 `audio_start` / `barge_in` 对齐 playableTurn。
2. **双层打断**：状态机置位 `_cancel_speaking`，各发送/合成循环自己检查后退出，
   **没人去 task.cancel() 别人**。
3. **到处 sleep/yield 是为了交还事件循环**，否则同步 for 会卡住 feed_audio、打断迟钝。

## 线程边界
全部方法在同一 asyncio 事件循环里（uvicorn 单 worker），状态字段无需加锁；
唯一例外是 `run_in_executor` 把 ASR/RAG/TTS 等阻塞调用丢进默认线程池。
`_tasks` 持强引用防 GC，`shutdown()` 统一取消。

## 代码组织
本文件只保留**状态机主干**（音频入口、状态分支、打断、生命周期）。按职责拆出的
功能模块与它**平级**放在 `engine/` 下，`ConversationManager` 组合它们 ——
它们共享 `self` 上的整机会话状态，这样拆分是为了"一个文件读一件事"，不改变行为：

| 模块 | 职责 |
|---|---|
| `dialog/constants.py`   | 调优常量与音频契约（采样率/帧长/各类阈值） |
| `dialog/state.py`       | 5 状态枚举 `State` |
| `dialog/metrics.py`     | 单会话延迟统计 `SessionMetrics` |
| `dialog/accumulator.py` | ASR 文本累积器（短句先缓存、拼整句再送 LLM） |
| `dialog/endpointing.py` | 判停（按说话时长自适应选静音阈值） |
| `dialog/transport.py`   | 下行发送（系统话术/语气词/TTS 音频） |
| `dialog/streaming.py`   | LLM 句子流 → TTS 音频流（首响关键路径） |
| `dialog/pipeline.py`    | 一轮对话的编排：ASR→RAG→LLM→TTS + 错误兜底 |
| `dialog/session.py`     | 会话生命周期：问候/空闲超时/PTT/文本输入 |
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

import numpy as np

from .accumulator import AsrAccumulator
from .constants import (
    BARGE_IN_CONFIRM_CHUNKS,
    BARGE_IN_RMS_THRESHOLD,
    BARGE_IN_VAD_THRESHOLD,
    CHUNK_SAMPLES,
    IDLE_CHECK_INTERVAL_CHUNKS,
    INTERRUPT_SILENCE_CHUNKS,
    SAMPLE_RATE,
    SPECULATIVE_ASR_CHUNKS,
    compute_rms,
)
from .endpointing import Endpointing
from .metrics import SessionMetrics
from .pipeline import Pipeline
from .session import Session
from .state import State
from .streaming import Streaming
from .transport import Transport

# 日志命名空间与 ws_server/tts/filler 等统一成 "Voice Agent.*"，
# 便于一条命令过滤整条链路：grep "Voice Agent" logs/voice_agent.log
log = logging.getLogger("Voice Agent.cm")


class ConversationManager(
    Session,        # 问候 / 空闲超时 / PTT / 文本输入
    Pipeline,       # ASR → RAG → LLM → TTS 编排
    Streaming,      # LLM 句子 → TTS 音频流
    Transport,      # 下行发送
    Endpointing,    # 判停
    AsrAccumulator, # ASR 文本累积
):
    """单个语音会话的对话状态机：吃 32ms 音频帧，吐协议消息与 TTS 音频。

    生命周期由 ws_server.ws_voice() 控制（建连时构造，断开时 shutdown()）。除 _produce_sentences
    跑在线程池的同步函数外，所有方法都必须在构造它的事件循环里调用，因此内部字段无需加锁。

    上面列出的 6 个基类各管一类职责（见模块文档的表格），都操作同一份 `self` 状态。
    """

    # 单条 TTS 音频的硬上限：30s @24kHz 16-bit。生成退化时一块可能巨大、会撑爆前端缓冲；
    # 截断比丢弃安全（丢块会造成爆破音）。transport 与 streaming 两个基类共用这个值。
    MAX_TTS_AUDIO_BYTES = 24000 * 2 * 30

    def __init__(self, asr, llm, tts, rag, vad, filler, send_fn: Callable[..., Awaitable],
                 spec_asr=None,
                 experience_config: dict = None):
        """装配依赖、建立初始状态与 D1~D4 体验层参数（不加载模型、不发网络请求）。

        各依赖都是阻塞式同步接口（本类统一丢线程池）：asr.transcribe -> {"text","latency_ms"}；
        llm.stream_sentences(...) 逐句 yield {"sentence","ttfs_ms",...}；
        tts.synthesize_stream 逐块 yield
        {"audio","chunk_idx","ttfa_ms"}（24kHz）；rag.get_context -> {"context","total_ms"}；
        vad.process_chunk(chunk, sr) -> 含 "speech_prob"（本项目 VAD 固定为 Silero，无说话人判定）；
        filler.get_filler -> (文本, 裸 PCM int16)；send_fn(dict) 中 {"_audio": bytes} 走二进制帧，其余 JSON。
        spec_asr 可选，None 时能力降级；experience_config 来自 config.py。
        """
        self.asr = asr
        self.llm = llm
        # 语音合成引擎（Kokoro，输出 24kHz 单声道 int16）
        self.tts = tts
        self.rag = rag
        self.vad = vad
        self.filler = filler
        self.spec_asr = spec_asr
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

        self.metrics = SessionMetrics()

        # 系统级错误话术：pipeline 崩溃时替换成"一句道歉"播给用户，避免电话里突然静音
        # 兜底话术与 config.py 保持一致（英文产品）；生产值总由 build_experience_config 传入
        self._error_apology = "Sorry, something went wrong on my side. Could you say that again?"

        # D1~D4 的话术与阈值由 ws_server 从 config.py 组装传入；None 或键缺失时用下面的兜底默认值
        cfg = experience_config or {}
        self._greeting_text = cfg.get("greeting_text", "")
        self._idle_timeout_s = cfg.get("idle_timeout_s", 15.0)
        # D2：静音多久（秒）后直接告别结束会话，必须 > idle_timeout_s
        self._idle_goodbye_s = cfg.get("idle_goodbye_s", 30.0)
        self._error_apology = cfg.get("error_apology_text", self._error_apology)
        self._idle_prompt_text = cfg.get("idle_prompt_text", "Are you still there? Shall we keep going?")
        self._idle_goodbye_text = cfg.get("idle_goodbye_text",
                                          "Okay, talk to you next time. Good luck with your IELTS!")
        self._asr_empty_text = cfg.get("asr_empty_retry_text",
                                       "Sorry, I didn't quite catch that. Could you say it again?")
        self._asr_noisy_text = cfg.get("asr_noisy_suggest_text",
                                       "It sounds a bit noisy on your side. "
                                       "Could you move somewhere quieter?")
        self._max_empty_asr = cfg.get("max_consecutive_empty_asr", 3)
        self._farewell_keywords = cfg.get("farewell_keywords",
                                          ["goodbye", "good bye", "bye", "see you"])

        # D2 计时基准：用 monotonic 而非 time.time()，避免系统改时间/NTP 校正把超时算错
        self._idle_since: float = time.monotonic()
        self._idle_chunk_count = 0
        self._idle_prompted = False
        self._idle_goodbye_sent = False
        self._consecutive_empty_asr = 0
        self._session_ending = False

        self._idle_gate_confirm = 0
        self._idle_speech_ring: list[np.ndarray] = []
        self._idle_speech_samples = 0

        self._asr_accum_texts: list[str] = []
        self._asr_accum_flush_task: Optional[asyncio.Task] = None

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
        # 注意：reset() 只清状态机与 VAD 流式隐状态；本项目的 VAD 只有 Silero，没有声纹门控
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
        # 普通判停的语音门限：**硬编码 0.6**，不是从 SileroVAD 的 threshold 来的
        # （那个参数只喂给 VAD 内部无人消费的端点事件，见 engine/vad.py 模块说明）。
        # 打断另有 BARGE_IN_VAD_THRESHOLD（同为 0.6）+ RMS 双门限 + 2 帧确认。
        is_speech = speech_prob >= 0.6
        rms = compute_rms(chunk)

        if self.state == State.IDLE:
            # 双门限：VAD 0.6 且 RMS > 0.015（经验值，约等于安静房间里正常说话的音量，滤掉风扇/空调底噪）
            if is_speech and rms > 0.015:
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
            else:
                # 静音帧也进缓冲（ASR 需要句尾静音断句），并累计静音帧数用于判停
                self._audio_buffer.append(chunk)
                self._silence_count += 1

                # 静音刚满 160ms，且没启动过、音频确实有内容（>1 帧）→ 抢跑一次 ASR，结果先缓存着
                if (self._silence_count == SPECULATIVE_ASR_CHUNKS
                        and self._spec_asr_task is None
                        and len(self._audio_buffer) > CHUNK_SAMPLES):
                    self._launch_speculative_asr()

                # 纯静音帧判停：按话长自适应选阈值（见 _adaptive_endpoint_threshold）
                threshold = self._adaptive_endpoint_threshold()
                if self._silence_count >= threshold:
                    await self._on_endpointing()

        elif self.state == State.SPEAKING:
            if not hasattr(self, '_speak_log_count'):
                self._speak_log_count = 0
            self._speak_log_count += 1
            should_log = (
                self._speak_log_count <= 5
                or (speech_prob > 0.3 and self._speak_log_count % 10 == 0)
            )
            if should_log:
                log.info("[SPEAKING] chunk %d: vad=%.2f rms=%.4f threshold=%.2f/%.4f",
                         self._speak_log_count, speech_prob, rms,
                         BARGE_IN_VAD_THRESHOLD, BARGE_IN_RMS_THRESHOLD)

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
            if is_accepted:
                self._thinking_extra_buffer.append(chunk)
                self._thinking_has_speech = True
            elif self._thinking_has_speech:
                self._thinking_extra_buffer.append(chunk)

    def _launch_speculative_asr(self):
        """在判停前用当前音频快照跑一次 ASR（fire-and-forget），结果缓存供 pipeline 复用。

        用主 ASR（Parakeet-TDT）在判停前先跑一遍。副作用：concat 一份音频快照
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
        engine_name = "Parakeet-TDT"
        log.debug("Speculative ASR (%s) launched on %d samples", engine_name, len(audio_snapshot))

    def _cancel_speculative_asr(self):
        """取消投机 ASR 任务并清空结果缓存（幂等）：投机结果一旦与新音频不匹配就必须作废。"""
        # 只在任务还在跑时取消；已完成的再 cancel 会抛 InvalidStateError
        if self._spec_asr_task and not self._spec_asr_task.done():
            self._spec_asr_task.cancel()
        self._spec_asr_task = None
        # 结果缓存作废：音频已变，旧转写绝不能再被 _run_pipeline_inner 复用
        self._spec_asr_result = None
        self._spec_asr_audio_len = 0

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
