"""对话状态机的 5 个状态。

取值即下行协议 `{"type":"state","state":...}` 里的字符串，前端用它切指示灯。
状态流转（谁能跳到谁）见 `engine/dialog/manager.py` 模块头部；每个状态
"每收到一帧做什么"见该文件的 `_process_chunk()` 五个分支。
"""

from enum import Enum


class State(Enum):
    """对话状态机的 5 个状态。

    取值与下行协议 `{"type":"state","state":...}` 的字符串**一一对应**（前端
    static/voice_agent.html 用它切指示灯）。下面每个状态按"何时处于它 /
    这期间在做什么 / 什么让它离开"三段说明，并配一个典型场景。
    """

    # ── 空闲：没有进行中的一轮对话 ──
    # 何时处于：刚建立连接；以及每一轮 AI 播报正常结束之后（都会回到这里）。
    # 在做什么：等用户开口。开口门限是 **VAD 概率 ≥0.6 且 RMS >0.015**（双门限，
    #           滤掉风扇/空调底噪）；不满足时每 31 帧（≈1s）查一次空闲超时。
    # 何时离开：检出语音 → LISTENING；播系统话术（问候）→ SPEAKING。
    # 例子：用户点了"开始通话"但还在犹豫没开口。前端指示灯是灰的，麦克风照常送帧，
    #       系统在 15s 时提醒一句（IDLE_TIMEOUT_S），到 30s 仍无语音就告别并收线
    #       （IDLE_GOODBYE_S）。
    IDLE = "idle"

    # ── 收集中：正在攒用户这一轮说的话 ──
    # 何时处于：IDLE 检出语音后进入；THINKING 发现上一句太短也会回落到这里。
    # 在做什么：每帧都进 `_audio_buffer`（**包括静音帧**——ASR 需要句尾静音来断句），
    #           同时给静音帧计数；静音满 5 帧（160ms）抢跑一次投机 ASR，
    #           计数达阈值（13 或 20 帧，按本次说话时长自适应）就判停。
    # 何时离开：判停 → THINKING；累积器 3s 兜底 → THINKING；系统话术 → SPEAKING。
    # 例子：用户说 "I'd like to practise IELTS speaking today"——这 2 秒多里一直是
    #       LISTENING，缓冲从 0 帧涨到约 70 帧，最后一句尾静音触发判停。
    LISTENING = "listening"

    # ── 思考中：已判停，后端链路在跑，等第一句 ──
    # 何时处于：判停后立刻进入。600ms 首响预算就是从这里开始计时的。
    # 在做什么：**不再判停**。ASR → RAG → LLM 在后台跑，期间可能先播一条语气词
    #           盖住空档（filler）；若这期间用户又开口，帧会追加进
    #           `_thinking_extra_buffer`，等下并进本轮一起处理。
    # 何时离开：首句就绪 → SPEAKING；ASR 过短/空结果 → LISTENING；出错 → IDLE。
    # 例子：判停后要等 ASR（约 180ms）+ RAG（数十 ms）+ LLM 首 token（约 163ms），
    #       这四百来毫秒都在 THINKING。前端指示灯是黄的，用户只听到一声 "Well,"。
    THINKING = "thinking"

    # ── 播报中：正在放 AI 的回复，同时监听插话 ──
    # 何时处于：首句的 TTS 音频合成好就进入；系统话术（问候/道歉/提醒）也走这个状态。
    # 在做什么：播音频的同时每帧做打断判定——**VAD ≥0.6 且 RMS >0.012，连续 2 帧**
    #           （≈64ms）才认账；单帧毛刺（咳嗽、鼠标、回声尖峰）不足以打断。
    # 何时离开：检出真实语音 → INTERRUPTED；播完 → IDLE；用户按键打断 → LISTENING。
    # 例子：AI 正读 "Sure, let's start with Part 1..."，用户突然插一句 "Wait"——
    #       连续 2 帧确认后立刻置位 `_cancel_speaking`，前端清空播放队列，AI 当场闭嘴。
    SPEAKING = "speaking"

    # ── 被打断：用户在 AI 说话时插话，正在收他这句话 ──
    # 何时处于：**只从 SPEAKING 的自动打断进入**（用户按键打断直接回 LISTENING，
    #           不经过这里——按键是明确的"我要说话"，不必再等）。
    # 在做什么：和 LISTENING 一样攒缓冲、数静音，但阈值更短：
    #           静音 10 帧（320ms，INTERRUPT_SILENCE_CHUNKS）就收尾。
    # 何时离开：静音达阈值 → THINKING（把这半句交给 pipeline 正常处理）。
    # 例子：用户打断后只说 "can we slow down a bit"，说完停顿 320ms 就进入 THINKING；
    #       这半句会作为新一轮提问，被截断的 AI 回复则作为上下文一并送给 LLM。
    INTERRUPTED = "interrupted"
