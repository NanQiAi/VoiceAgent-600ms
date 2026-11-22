<div align="center">

# 🎙️ VoiceAgent-600ms

### Voice · ASR · TTS · LLM · Infra

**级联 Pipeline：VAD → ASR → RAG → LLM → TTS —— 首响实测 458ms，低于 600ms 工程预算**

<br>

[![Pipeline](https://img.shields.io/badge/Cascade_Pipeline-458ms-00c853?style=for-the-badge&logo=speedtest&logoColor=white)](/)
[![Barge-in](https://img.shields.io/badge/Barge--in-~160ms-ff6f00?style=for-the-badge&logo=bolt&logoColor=white)](/)
[![LLM](https://img.shields.io/badge/LLM-DeepSeek-4a90d9?style=for-the-badge&logo=openai&logoColor=white)](/)
[![License](https://img.shields.io/badge/Apache-2.0-7c4dff?style=for-the-badge)](/)

<br>

> **800ms 是语音对话的图灵测试门槛**——超过它，人类会感知到"在和机器说话"。
> 本项目以 **600ms 为工程预算线**，实测 **458ms** 首响。

</div>

---

## 📖 这份文档怎么读

| 你的目的 | 从这里开始 | 预计耗时 |
|---|---|---|
| 🚀 **我要把它跑起来** | [第二部分 · 部署](#第二部分--部署)（先看 [快速开始](#1-快速开始)） | 1~2 小时（含下载） |
| 📚 **我要读懂代码** | [第一部分](#第一部分--认识项目) → [第三部分 · 读代码](#第三部分--读代码) | 半天 ~ 1 周 |
| 🔧 **我在改它 / 排查问题** | [通信协议](#5-通信协议) · [配置速查](#6-配置速查) · [排障](#9-排障) · [附录 A 陷阱](#附录-a已知的坑) | 按需查阅 |

**全文结构**

- **第一部分 · 认识项目**（1–6 节）：定位、数据流、仓库地图、核心概念、协议、配置
- **第二部分 · 部署**（1–10 节）：Debian 13 上的完整部署（**DeepSeek 云端 + 一块 24GB 卡**）
- **第三部分 · 读代码**：代码地图、三条关键链路、学习路线
- **附录**：已知的坑、常见任务、术语表

> ⚠️ **关于准确性**：本文档按**当前代码**逐文件核实写成。若与代码不符，**以代码为准**。

---

# 第一部分 · 认识项目

## 1. 项目是什么

一个**全双工中文/英文语音对话 Agent**，用**级联 Pipeline** 而不是端到端大模型：

```
VAD → ASR → RAG → LLM → TTS
```

**为什么是级联而不是端到端？** 可控性。每个组件能单独替换、单独观测、单独调优：
换 ASR 只动 `engine/asr.py`，换 LLM 只动一个环境变量，换 TTS 只动 `engine/tts.py`。
端到端方案要改行为得重训整个多模态模型。而当两条路线的延迟都已低于人类感知门槛时，
**可控性 > 更低的延迟**。

### 可替换组件

| 组件 | 当前实现 | 可替换为 |
|---|---|---|
| VAD | **Silero**（CPU） | FireRedVAD · WebRTC VAD |
| ASR | **Parakeet-TDT-0.6B**（NVIDIA，经 transformers） | Whisper · SenseVoice · Paraformer |
| RAG | **bge-m3（1024 维）+ FAISS** | bge-small-zh-v1.5 · Milvus · Elasticsearch |
| LLM | **DeepSeek 云端**（`deepseek-flash`） | 任何 OpenAI 兼容服务（改 `LLM_BASE_URL` 即可） |
| TTS | **Kokoro-82M**（24000Hz） | VoxCPM · CosyVoice · IndexTTS · Fish-Speech |

> LLM 默认走 DeepSeek 云端：**不需要本地显卡跑大模型、不需要下载 14B 权重、不需要起 vLLM**。
> 只有 ASR + TTS 需要 GPU。**本项目不支持本地大模型**（vLLM / Qwen3 那套已整体移除）。

### 核心指标（实测，RTX 4090）

| 指标 | 实测 |
|:---:|:---:|
| 首响延迟 | **458ms** |
| 打断延迟（听感） | **~160ms** |
| 换 ASR / LLM / TTS | 替换单个模块或一个环境变量 |

> 📌 这些是**实测记录**，不是代码里的常量。要复现需按第二部分真正跑起来后测。

---

## 2. 一次完整对话发生了什么

```
用户说话
   │  16kHz / 单声道 / int16 / 512 采样/帧（32ms）
   ▼
┌─────────────────────────────────────────────────────────────┐
│ ConversationManager（engine/dialog/manager.py，状态机核心）    │
│                                                             │
│  feed_audio() → _process_chunk()                            │
│     ├─ VAD：逐帧判"是不是人声"（Silero，CPU）                  │
│     ├─ 判停：连续静音够久 → _on_endpointing()                 │
│     └─ 打断：SPEAKING 时检测到人声 → handle_barge_in()        │
└─────────────────────────────────────────────────────────────┘
   │ 判停后
   ▼
_run_pipeline()
   ├─ ASR.transcribe()        Parakeet-TDT，~180ms（CPU 实测 3.7s 音频）
   ├─ RAG.get_context()       bge + FAISS，~4ms（命中才注入）
   └─ _stream_llm_tts()       LLM 逐句 → TTS 逐段
        ├─ LLMChat.stream_sentences()   DeepSeek 流式，首 token ~163ms
        └─ KokoroTTS.synthesize_stream() 按句合成，24000Hz
             └─ _send({"_audio": ...})  → 前端 JitterBufferPlayer
```

**首响分解**（`first_response_ms`）：

| 环节 | 耗时 | 代码位置 |
|---|---|---|
| ASR（Parakeet-TDT-0.6B） | **~180ms**（CPU 实测，3.7s 音频；GPU 应更快） | `engine/asr.py` |
| RAG（bge-m3 + FAISS） | ~15-50ms | `engine/rag.py` |
| LLM 首 token（DeepSeek 云端） | ~163ms | `engine/llm.py` |
| TTS 首段（Kokoro） | 未单独实测 | `engine/tts.py` |
| **合计** | **~458ms** | |

> 指标**不含** VAD 判停等待（**416~640ms**）——那是用户说完话到系统开始处理之间的静默。

**五个让它变快的工程手段**：

| # | 手段 | 效果 |
|---|---|---|
| 1 | **流式 TTS** — 逐句合成、逐段下发，不等整段回答 | 用户听到第一句时 LLM 还在生成后半段 |
| 2 | **投机式 ASR** — 静音 160ms 就先跑一遍 ASR | 判停后直接复用，省一次 ASR |
| 3 | **自适应判停** — 按说话时长选静音阈值 | 少误切断 |
| 4 | **Turn 序列号** — 每轮 `audio_start(turn=N)`，前端丢弃旧轮次音频 | 消除打断后的串音 |
| 5 | **双层打断** — 服务端停发 + 前端 flush 播放队列 | 听感上立刻闭嘴 |

---

## 3. 仓库地图

项目只有 **3 个顶层 Python 文件**，其余在 `engine/` 与 `server/` 两个包里：

| 文件 / 目录 | 职责 | 优先级 |
|---|---|---|
| `config.py` | **配置中枢**：路径、设备、音频契约、阈值、话术、prompt（8 个分区） | ⭐⭐⭐ 第一个读 |
| `ws_server.py` | 进程入口（55 行）：sys.path 引导 + 装配引擎 + 起 uvicorn | ⭐⭐⭐ 第二个读 |
| `engine/__init__.py` | **结构总览**：`engine/` 的分层与依赖方向 | ⭐⭐⭐ |
| `engine/dialog/manager.py` | **对话状态机门面**：音频入口、状态分支、打断、生命周期 | ⭐⭐⭐ 最难，最后啃 |
| `server/` | 服务层（独立包）：bootstrap / runtime / loader / app + 三组路由 | ⭐⭐⭐ |
| `engine/llm.py` | DeepSeek 客户端 + 流式切句 + 历史/摘要 + think 清洗 | ⭐⭐ |
| `engine/tts.py` | Kokoro 流式合成（音色固定 `af_heart`）+ 打断检查 | ⭐⭐ |
| `engine/rag.py` | bge 向量化 + FAISS 检索 + 置信度阈值 | ⭐⭐ |
| `engine/asr.py` | **唯一的 ASR 实现**（Parakeet-TDT / transformers） | ⭐⭐ |
| `engine/vad.py` | Silero VAD（CPU） | ⭐⭐ |
| `static/` | 前端：`voice_agent.html`（86 行骨架，只有 Start Call / Reset 两个按钮）+ `css/` + `js/`（9 个 ES module） | ⭐⭐ |
| `run.sh` | **唯一的启停入口**：配置 + 预检 + 启动 + 守护 + 排障 | ⭐⭐ |
| `test/test_rag.py` | RAG 检索基准（bge-m3 × 有无 reranker）。10 条英文问句含 8 正例 + 2 负例，输出延迟与**正负例分数间隔**，用于判断阈值是否合理 | ⭐ |
| `check_deps.py` | 依赖自检：17 必需 + 1 可选（共 18 项），缺必需包时退出码 1 | ⭐ |
| `models/download_model.py` | 批量下载 ASR / RAG / TTS 权重（路径按 `__file__` 推仓库根，任意目录可跑） | ⭐ |
| `.env` / `.env.example` | 本地环境变量；`.env` 已 gitignore，`.env.example` 可提交 | ⭐ |

### `engine/` 分两层（依赖单向，下层不 import 上层）

```
engine/
├── asr.py  llm.py  rag.py  tts.py  vad.py          ① 可替换引擎（互不 import）
└── dialog/                                             ② 对话状态机
    manager.py       门面（ConversationManager）
    constants.py     调优常量与音频契约
    state.py         5 状态枚举
    metrics.py       单会话延迟统计
    accumulator.py   ASR 文本累积
    endpointing.py   判停
    transport.py     下行发送
    streaming.py     LLM 句子流 → TTS 音频流
    pipeline.py      一轮对话编排
    session.py       问候 / 空闲超时
```

### `server/` 是独立顶层包（不在 `engine/` 里）

```
server/
├── bootstrap.py    nest_asyncio 补丁 + 日志 + 仓库路径（参数都在 config.py）
├── runtime.py      全局引擎表 + 活动会话表 + 计数
├── loader.py       load_engines()：启动期一次性装配
├── routes_meta.py  /            /api/info      /api/metrics
├── routes_rag.py   /api/rag/docs  /query       /reload
├── routes_voice.py /ws/voice
└── app.py          FastAPI 实例 + 静态挂载 + 挂路由
```

**传输方式**：WebSocket（`ws_server.py` + `static/voice_agent.html`）。走 TCP，
所以打断需要前端显式 flush 播放队列（见 [4.3](#43-barge-in打断-本项目的重点)）。

**`data/`**：只有 `data/sample_kb.json`（RAG 知识库，59 条问答）被代码读取。

---

## 4. 核心概念速通

### 4.1 VAD（Voice Activity Detection）

`engine/vad.py`，Silero，**跑 CPU**（`torch.jit.load(..., map_location="cpu")`）。
每帧 32ms 输出一次"是人声的概率"。

> ⚠️ **真正生效的门限是 0.6**（硬编码在 `engine/dialog/manager.py` 的 `_process_chunk`）。
> `SileroVAD(threshold=0.5)` 这个构造参数**不影响判停/打断**——它只驱动 VAD 内部那套
> `speech_start` / `speech_end` 事件，而那些事件目前没有调用方。详见 `engine/vad.py` 模块说明。

### 4.2 判停（Endpointing）

**只有一个依据：连续静音帧数**（没有语义判停器）。档位按本次说话时长自适应：

| 本次说话时长 | 档位 | 需要静音 |
|---|---|---|
| < 0.5s | `ENDPOINT_SLOW_CHUNKS` | 20 帧 ≈ **640ms** |
| 0.5 ~ 3s | `ENDPOINT_DEFAULT_CHUNKS` | 13 帧 ≈ **416ms** |
| > 3s | `ENDPOINT_SLOW_CHUNKS` | 20 帧 ≈ **640ms** |

> **为什么短句也等 640ms？** 短句多半是用户还在换气、马上要接着说，早判停会把一句话切两半；
> 长句则怕用户中途长停顿。只有中段才用较快的 416ms。

### 4.3 Barge-in（打断）—— 本项目的重点

**触发条件**（`engine/dialog/constants.py`，双门限 + 连续确认）：

```python
BARGE_IN_VAD_THRESHOLD = 0.6    # VAD 语音概率（与判停同为 0.6；更严的是下面的 RMS + 2 帧确认）
BARGE_IN_RMS_THRESHOLD = 0.012  # 真实能量（语音概率会被回声/键盘声骗过）
BARGE_IN_CONFIRM_CHUNKS = 2     # 连续 2 帧（≈64ms）才真打断，挡单帧毛刺
```

**两层机制**：

| 层面 | 做什么 |
|---|---|
| **服务端** | 置 `_cancel_speaking`，三条发送/合成循环各自检查后 `break` |
| **前端** | 收到 `barge_in` → `flushPlayback()` 清空播放队列 + `playableTurn = 0` 关闸 |

| 实际粒度 | 值 | 说明 |
|---|---|---|
| **用户听感** | ~160ms | VAD 2 帧（≈64ms）确认后下发 `barge_in`，前端立即清队列 → 立刻静音 |
| **服务端停止下发** | 1 个 TTS 句段（1~3s） | 只影响"省不省 GPU"，不影响听感 |

### 4.4 Turn 序列号（防串音）

每轮 `audio_start` 带 `turn=N`；前端只播当前授权轮次的音频。打断后 `playableTurn=0`，
在途的旧轮次帧到达时被**直接丢弃**。这是"打断后不会冒出半句旧语音"的关键。

### 4.5 投机式 ASR（Speculative ASR）

静音满 5 帧（≈160ms）且还没判停，就先用当前音频快照跑一次 ASR。
若判停时音频没变长，直接复用结果，省掉一次 ASR 等待。阈值 `SPECULATIVE_ASR_CHUNKS`。

### 4.6 ASR 文本累积

VAD 误切或用户只说"嗯/那个"时，立刻送 LLM 既浪费一轮 turn 又容易答非所问，
所以极短文本（≤4 字符）先缓存，等后续拼成整句再送（`ASR_ACCUM_*`）。

### 4.7 5 个状态

```
IDLE ──VAD检出──> LISTENING ──静音达阈值──> THINKING ──首句就绪──> SPEAKING ──播完──> IDLE
                                            ↑                      │
                                            └──静音≥320ms── INTERRUPTED ←──VAD 自动打断──┘
SPEAKING / THINKING ──手动打断(协议 manual_barge_in，前端未接)──> LISTENING
THINKING ──用户又开口──> 追加进同一轮，不打断
```

两条打断路径去向不同：**VAD 自动打断**进 `INTERRUPTED`（需再静音 320ms 才继续处理后面的半句），
**手动打断**直接回 `LISTENING`——用户是明确要说话，不必再等。

> ⚠️ `manual_barge_in` 目前**只有协议与服务端实现**：页面上没有按钮、也没有快捷键
> （只有 `Start Call` / `Reset` 两个按钮）。要触发需客户端自行发送，例如控制台执行
> `app.socket.send(JSON.stringify({type:"manual_barge_in"}))`。

状态通过 `{"type":"state","state":"..."}` 下发，前端显示状态灯。定义在 `engine/dialog/state.py`。

---

## 5. 通信协议

一条 WebSocket `/ws/voice`，**二进制帧 = 音频，文本帧 = JSON 控制消息**。

### 5.1 上行（客户端 → 服务端）

| 帧 | 含义 |
|---|---|
| **二进制** | 16kHz / 单声道 / int16 裸 PCM，**512 采样（32ms）一帧** |
| `{"type":"manual_barge_in"}` | 手动打断，不需要 VAD 确认。**前端未提供入口**，需客户端自行发送 |
| `{"type":"reset"}` | 复位会话（清对话历史与状态机） |

### 5.2 下行（服务端 → 客户端）

| 帧 | 含义 |
|---|---|
| **二进制** | 24kHz / 单声道 / int16 裸 PCM |
| `ready` | 握手完成（含 `session_id` / `version` / `mode`；`mode` 恒为 `duplex`，全双工是唯一模式） |
| `state` | 状态机变化（idle / listening / thinking / speaking） |
| `asr` | 识别结果（`text` / `latency_ms` / `turn`；`source="text"` 表示文本回显） |
| `rag` | RAG 命中的上下文 |
| `llm_sentence` | LLM 生成的一句（`sentence_idx=0` 时开新气泡） |
| `llm` | 整轮 LLM 输出结束（前端据此收尾播放） |
| `audio_start` | **播放授权**：随后到达的音频属于第 `turn` 轮 |
| `metrics` | 本轮延迟分解 |
| `barge_in` | 用户打断（`source` = `vad` / `manual`） |
| `system_message` | 服务端主动文案（问候 / 空闲 / 错误 / 告别） |
| `session_end` | 会话结束（`reason`） |
| `error` | 错误 |
| `reset_ack` | 复位的回执 |

### 5.3 音频格式约定（**最容易写错的地方**）

| 方向 | 采样率 | 帧长 | 格式 |
|---|---|---|---|
| 上行（麦克风 → 服务端） | **16000 Hz** | **512 采样 = 32ms** | int16 单声道裸 PCM |
| 下行（服务端 → 播放器） | **24000 Hz** | 按句段 | int16 单声道裸 PCM |

> ⚠️ 采样率错配**不会报错**，只会"声音变调/变速"。上行 16k 是 VAD/ASR 的时频特征基准，
> 下行 24k 是 Kokoro 声码器的硬约束。改任何一个都要同步改前端 `static/js/config.js`。
>
> 前端用 AudioWorklet：采集侧攒够 512 点发一帧，播放侧用抖动缓冲（起播水位 300ms）。

### 5.4 容错约定（服务端不因单帧异常中断会话）

| 情况 | 服务端行为 |
|---|---|
| 控制帧不是合法 JSON | 记一条 warning，**丢弃该帧**，会话继续 |
| 二进制帧字节数为奇数（int16 不成立） | 记 warning，丢弃该帧，会话继续 |
| 未知 `type` | 静默忽略（便于前端先升级） |
| 其它未预期异常 | 记完整堆栈，`finally` 里注销会话、取消后台任务并**显式 `close()`** |

> 最后一条很关键：不能依赖"ASGI app 返回即关闭连接"的隐式行为——不同服务器/测试客户端
> 表现不一致，对端可能一直阻塞在 `receive()`。

---

## 6. 配置速查

所有参数集中在 `config.py`，分 8 个分区：

| # | 分区 | 主要内容 |
|---|---|---|
| 1 | 路径基准 | `BASE_DIR` / `MODEL_DIR` |
| 2 | 权重目录与 TTS 参数 | `ASR_MODEL_DIR` / `VAD_MODEL_DIR` / `EMBED_MODEL_DIR` / `KOKORO_MODEL_DIR` / `KOKORO_SPEED`（**音色固定**，见 8.2） |
| 3 | 设备与服务开关 | `ASR_DEVICE` / `TTS_DEVICE` / `RAG_DEVICE` / `PORT` / `USE_SSL` / **`VERSION`** |
| 4 | LLM | `LLM_BASE_URL` / `LLM_MODEL_NAME` / `LLM_API_KEY`（env 名 `DEEPSEEK_*`）/ `LLM_MAX_HISTORY` |
| 5 | 音频契约 | `SAMPLE_RATE`(16k) / `CHUNK_MS`(32) / `CHUNK_SAMPLES`(512) / `TTS_SAMPLE_RATE`(24k) |
| 6 | RAG | `KB_DATA_PATH` / `RAG_TOP_K` / `RAG_SCORE_THRESHOLD` |
| 7 | 体验层话术 | `GREETING_TEXT` / `IDLE_*` / `ASR_*` / `ERROR_APOLOGY_TEXT` / `MAX_CONSECUTIVE_EMPTY_ASR` / `FAREWELL_KEYWORDS` |
| 8 | System prompt | `SYSTEM_PROMPT_RAG` |

> `VERSION`（`"3.0"`）是**版本号唯一来源**：启动日志、`/api/info`、WS `ready` 消息、
> FastAPI 标题都读它，改版本只改这一处。

**优先级：命令行 `export` > `.env` > `config.py` 默认值。**

```bash
# 例：换端口、把 RAG 放 CPU、语速降到 0.9
PORT=8080 RAG_DEVICE=cpu KOKORO_SPEED=0.9 bash run.sh
```

> 常量的注释写的是"**为什么是这个值**"。改任何一项前先读注释——很多值影响
> 600ms 首响预算 / 打断灵敏度 / 判停快慢。

---

# 第二部分 · 部署

> 目标环境：**Debian 13 + 一块 24GB 显卡**（RTX 3090 Ti / 4090 均可）。
> **LLM 走 DeepSeek 云端**，所以本地只需要跑 ASR + TTS，单卡绰绰有余。

## 1. 快速开始

```bash
cd <仓库根目录>
conda create -n voiceagent python=3.11 && conda activate voiceagent

# ① 配置 DeepSeek API Key（唯一必填的凭据）
cp .env.example .env          # 然后编辑 .env，填入 DEEPSEEK_API_KEY
# 也可以临时 export（命令行优先于 .env）：
#   export DEEPSEEK_API_KEY=sk-...

# ② 预检 + 启动
bash run.sh

# ③ 浏览器打开 http://<本机IP>:3000/ ；Ctrl+C 停止
```

**`.env` 说明**

| 文件 | 是否入库 | 用途 |
|---|---|---|
| `.env` | ❌ 已忽略（`.gitignore`） | 你的真实配置：`DEEPSEEK_API_KEY` 等 |
| `.env.example` | ✅ 可提交 | 模板，`cp` 一份改名 `.env` 即可 |

加载是**自动**的：`config.py` 与 `run.sh` 都会读仓库根目录的 `.env`。
规则是**已 `export` 的同名变量优先**，`.env` 不会覆盖命令行；格式只认 `KEY=value`、
`#` 注释与空行，写错的行会被静默跳过（不会因此启动失败）。

> 🔑 `.env` 已加入 `.gitignore`，但**别把它贴进 issue / 截图 / 聊天记录**。
> 若你曾把 key 写进代码或提交过 `.env`，去 [platform.deepseek.com](https://platform.deepseek.com)
> 吊销重发，并 `git rm --cached .env` 把文件移出版本控制。

**LLM 固定走 DeepSeek 云端**（`config.py` §4），只需要一个 `DEEPSEEK_API_KEY`：

```python
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
```

底层是 OpenAI 兼容协议，换任何兼容服务只改这三个值。**本项目不支持本地大模型**
—— 历史上的 `LLM_BACKEND` / vLLM / Qwen3-14B-AWQ 支持已整体移除。

> ⚠️ DeepSeek 的思考模式**默认是开的**，项目已显式关掉
> （`extra_body={"thinking": {"type": "disabled"}}`）。不关的话模型会先生成一大段思维链，
> "首句可播"晚几百 ms，600ms 预算直接崩。

---

## 2. 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| OS | Debian 13（其它 Linux 同理） | 脚本用 `ss` 探端口 |
| GPU | **一块 24GB 显卡** | 只有 ASR + TTS 上卡，峰值 ~14GB |
| 驱动 | 支持 CUDA 13 | `nvidia-smi` 能正常输出 |
| Python | **3.11**（3.10~3.12 可） | `kokoro` 要求 `>=3.10,<3.13` |
| 磁盘 | ~2GB | 权重 1.1GB + 依赖；Kokoro 权重首次自动下载 330MB |
| 网络 | 能访问 DeepSeek API | 可选：`HF_ENDPOINT` 镜像 |

---

## 3. 安装依赖

**只用 [`requirements.txt`](requirements.txt)**，不要逐包手装——它用 `==` 锁定了 **16 个**
直接依赖的精确版本；其余传递依赖（transformers / librosa / scipy / spacy / pydantic …）
由 pip 按上游声明自动解析。

```bash
conda create -y -n voiceagent python=3.11 && conda activate voiceagent

# ① 先装 torch 三件套 + CUDA 索引（必须第一步）
#    其它包会重新解析 torch 依赖，不先装好会被换成别的构建版本。
pip install torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ② 再装其余（含 kokoro；torch 已满足，不会被重装）
pip install -r requirements.txt

# ③ 系统依赖（非 pip）：英文 OOV 词的兜底发音
sudo apt install -y espeak-ng

# ④ spaCy 英文模型 —— misaki 的英文 G2P **必需**，放仓库里而不是 pip 装
#    （下载方式见下一节「下载模型」）
#    models/en_core_web_sm/
#      ├─ en_core_web_sm/                 ← 模型包
#      └─ en_core_web_sm-3.8.0.dist-info/ ← 让 spacy.util.is_package() 认得

# ⑤ 自检
python check_deps.py
```

> ⚠️ **`en_core_web_sm` 是容易被漏掉的硬依赖**：缺了它 `KPipeline` 会在构造时调用
> `spacy.cli.download()` 现场联网下载，GitHub 不通就直接抛
> `requests.exceptions.ConnectionError` —— 报错信息看起来和 TTS 毫无关系。
>
> 本项目**不从 pip 安装它**：模型解压到 `models/en_core_web_sm/`，
> `engine/tts.py` 在 `load()` 里把该目录插到 `sys.path` 最前面，
> `spacy.util.is_package('en_core_web_sm')` 命中后就不会再联网。这样整个仓库可整体迁移。

验证 GPU 可见：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 期望：2.13.0+cu130 True 1（或 2）
```

> **关于 `--index-url`**：它让 pip **只用** pytorch 索引。该索引完整托管 torch 2.13.0 的
> 全部依赖（含 `cuda-toolkit`、`nvidia-cudnn-cu13` 等），不会缺依赖。若网络不可达，
> 改成 `--extra-index-url`（保留 PyPI 作主索引）。
>
> **关于 Kokoro 音色**：音色**固定**为 `af_heart`（标准美式英语女声，评级 A、韵律最自然）。
> 本项目不做音色克隆、也不开放换音色，详见 [10.2](#82-tts-音色固定不可切换)。

---

## 4. 下载模型

**五个**必需权重，全部放 `models/` 下（TTS 与 spaCy 模型都不再依赖 pip 装包 / HF 缓存）：

| 目录 | 用途 | 大小 | 关键文件 |
|---|---|---|---|
| `models/parakeet-tdt-0.6b-v3` | ASR（英语，WER ≈ 6%） | 2.5G | `model.safetensors` + `tokenizer.json` |
| `models/bge-m3` | RAG 向量化（**1024 维**） | 4.3G | `config.json` + 权重 |
| `models/snakers4_silero-vad` | VAD | 14M | **`silero_vad.jit`** |
| `models/Kokoro-82M` | TTS | 347M | `config.json` + `kokoro-v1_0.pth` + `voices/af_heart.pt` |
| `models/en_core_web_sm` | spaCy 英文模型（misaki 英文 G2P） | 15M | `en_core_web_sm/` + `*.dist-info/` |

> **Kokoro 也从本地目录加载**（`config.KOKORO_MODEL_DIR`），不再走 `~/.cache/huggingface`。
> `engine/tts.py` 自己构造 `KModel(config=..., model=...)` 并把音色 `.pt` 的绝对路径
> 传给 `KPipeline`，所以加载 TTS 全程不联网。
>
> `models/bge-m3` 比旧的 `bge-small-zh-v1.5`（184M / 512 维，**已删除**）大 20 倍、检索慢一个量级，
> 但多语言召回明显更好。索引每次启动在内存重建，换模型不需要手动清任何缓存。

**下载方式**：

```bash
# Parakeet-TDT-0.6B v3 —— 只托管在 HuggingFace（ModelScope 没有），约 2.5GB
huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 \
    --local-dir models/parakeet-tdt-0.6b-v3

# 国内访问 HuggingFace 慢/断流时，先设镜像，再用 wget -c 断点续传单文件
# （实测 `snapshot_download` 在这种网络下容易 TLS 断连，wget 更稳）
export HF_ENDPOINT=https://hf-mirror.com
wget -c --tries=20 --timeout=60 \
  -O models/parakeet-tdt-0.6b-v3/model.safetensors \
  "$HF_ENDPOINT/nvidia/parakeet-tdt-0.6b-v3/resolve/main/model.safetensors"

# bge-m3 走 ModelScope 更快
python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='models/bge-m3')
"
```

> ⚠️ Parakeet 需要 `model.safetensors`（约 2.5GB）**完整下载**才可用。
> `config.json` / `tokenizer.json` 只有几十 KB，会先下完——此时目录看着"有东西"，
> 但体积只有几 MB。`bash run.sh preflight` 会靠**体积**识别出这种半成品状态。

| 模型 | 来源 |
|---|---|
| Parakeet-TDT-0.6B v3 | HuggingFace `nvidia/parakeet-tdt-0.6b-v3`（`--local-dir models/parakeet-tdt-0.6b-v3`） |
| bge-m3 | BAAI 官方仓库（ModelScope `BAAI/bge-m3`） |
| Kokoro-82M | HuggingFace `hexgrad/Kokoro-82M`（`--local-dir models/Kokoro-82M`） |
| en_core_web_sm | GitHub `explosion/spacy-models` release 的官方 wheel（**不在** ModelScope / PyPI） |
| silero-vad | `snakers4/silero-vad` 的 `silero_vad.jit`（⚠️ 埋在 `src/silero_vad/data/` 下，需拷到 `models/snakers4_silero-vad/`） |

**spaCy 英文模型（`en_core_web_sm`）单独说明**：它不在 ModelScope，也不在 PyPI，
只有 GitHub release 的官方 wheel。装法是**解压进仓库**（不是 `pip install`）：

```bash
W=en_core_web_sm-3.8.0-py3-none-any.whl
URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/$W

# ① 下载（GitHub 慢/不通时给 URL 加代理前缀，例如 https://gh-proxy.com/）
curl -L -o /tmp/$W "$URL"

# ② 解压到 models/en_core_web_sm/，解压后应有两项：模型包 + dist-info
mkdir -p models/en_core_web_sm
python -m zipfile -e /tmp/$W models/en_core_web_sm/
ls models/en_core_web_sm/     # 期望: en_core_web_sm/  en_core_web_sm-3.8.0.dist-info/
```

> 版本要 ≥ 3.8（与 `spacy==3.8.x` 配套）。`misaki` 只认 `en_core_web_sm` 这个名字，
> 所以**目录名和里面的包名都不能改**。

**自检**：

```bash
python - <<'EOF'
import os
need = {"models/parakeet-tdt-0.6b-v3": "ASR", "models/snakers4_silero-vad": "VAD",
        "models/bge-m3": "RAG", "models/Kokoro-82M": "TTS",
        "models/en_core_web_sm": "spaCy 英文模型"}
for p, why in need.items():
    print(f"  {'✅' if os.path.isdir(p) else '❌'} {p:<34} {why}")
jit = "models/snakers4_silero-vad/silero_vad.jit"
print(f"  {'✅' if os.path.isfile(jit) else '❌'} {jit}   <- VAD 真正加载这个文件")
EOF
```

> `models/download_model.py` 已同步：只下 Parakeet-TDT（HuggingFace）/ bge-m3 / Kokoro（ModelScope），
> 路径按 `__file__` 推仓库根，**任意目录都能跑**。VAD 与 spaCy 模型来源特殊，需按本节单独获取。

---

## 5. 显存与 GPU 分配

**默认（DeepSeek 云端）：一块 24GB 卡绰绰有余。**

```
┌──────────────────────────────────────────────┐
│  GPU 0  (24GB)   ← ws_server 单进程独占       │
│  ──────────────────────────────────────────  │
│  ASR (Parakeet-TDT)       ≈ 2~3GB            │
│  TTS (Kokoro-82M)         ≈ 0.3GB（不预分配） │
│  实际峰值                 ≈ 13~14GB           │
│  余量                     ≈ 10GB              │
└──────────────────────────────────────────────┘

不占显存（跑 CPU）：Silero VAD / RAG（建议 cpu）/ LLM（在 DeepSeek 云端）
```

| 组件 | 跑在哪 | 依据 |
|---|---|---|
| **LLM** | **云端 API** | 固定 DeepSeek，本机不占显存 |
| ASR（Parakeet-TDT） | GPU（可设 CPU） | `ASR_DEVICE` |
| TTS（Kokoro） | GPU（很轻，可放 CPU） | `TTS_DEVICE` |
| RAG（bge） | CPU | `RAG_DEVICE=cpu`，检索只占几毫秒 |
| Silero VAD | **CPU**（固定） | `engine/vad.py` 用 `map_location="cpu"` |

**有两张卡时**：可以把 `TTS_DEVICE` 挪到另一张（`CUDA_VISIBLE_DEVICES=0,1` + `TTS_DEVICE=cuda:1`），
或者干脆只用一张、另一张留给别的任务——默认路径没有必须双卡的组件。

> ⚠️ **`CUDA_VISIBLE_DEVICES` 与 `*_DEVICE` 是两回事**：
> 前者是**物理卡号**（暴露哪些卡给进程），后者是**进程内可见序号**。
> `CUDA_VISIBLE_DEVICES=1` + `ASR_DEVICE=cuda:0` 表示"用物理 1 号卡"。
> 两种写法选一种，别混（也可以不用 `CUDA_VISIBLE_DEVICES`，直接写 `ASR_DEVICE=cuda:1`）。

---

## 6. 启动与运维：`run.sh`

**唯一起停入口**（历史上拆成 5 个脚本，现已合并）。

**不需要先 `conda activate voiceagent`**：`run.sh` 启动时会自动找一个名为
`voiceagent` 的 conda 环境并用它的解释器，探测顺序是

1. `export VOICEAGENT_PYTHON=<绝对路径>`（显式指定，最高优先级）
2. `conda env list` 里名为 `voiceagent` 的环境
3. 当前已激活环境的同级 `envs/voiceagent`（如 `base` 在 `/opt/miniforge3` → `/opt/miniforge3/envs/voiceagent`）
4. 常见安装根目录：`~/miniforge3`、`~/miniconda3`、`~/anaconda3`、`/opt/conda`
5. 都没有 → 退回 PATH 里的 `python`，并告警「正在用 base / 系统 Python」

所以从 `base` 环境里直接 `bash run.sh` 也能跑对。想换环境名改 `VA_CONDA_ENV_NAME`。

| 命令 | 作用 |
|---|---|
| `bash run.sh` / `bash run.sh dev` | 前台启动，Ctrl+C 连子进程一起停 |
| `bash run.sh start` | 常驻（脱离终端）+ 自动拉起 watchdog |
| `bash run.sh stop` / `restart` | 停止 / 重启（按**进程组**清理，含 TTS 子进程） |
| `bash run.sh status` | 进程 / 显卡 / 版本 / **当前用的 ASR+EMBED 模型** / 配置 / 日志位置 |
| `bash run.sh clean-gpu` | 列出占显存的进程；`FORCE=1` 才真杀 |
| `bash run.sh watchdog [--once]` | 看护循环；`--once` 单轮巡检 |
| `bash run.sh preflight` | 只跑预检，不启动服务 |
| `bash run.sh --help` | 用法 |

**为什么用进程组清理**：TTS/推理后端可能 fork 出持有 CUDA 的子进程，它**不监听任何端口**，
按端口找不到；只杀父进程会留下孤儿占显存，导致下次启动 OOM。所以用 `setsid` +
`kill -TERM -PGID` 整组清理。

**`run.sh` 的参数都从 `config.py` 读**（`va_cfg`）：模型名、端口、三个权重目录，
所以脚本的预检不可能和代码的期望对不上。

### 常用环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VOICEAGENT_PYTHON` | 自动发现的 `voiceagent` 环境 | 强制指定解释器绝对路径（最高优先级） |
| `VA_CONDA_ENV_NAME` | `voiceagent` | 自动发现时找哪个 conda 环境 |
| `ASR_DEVICE` / `TTS_DEVICE` | `cuda:0` | 显式设 `cpu` 可强制 CPU（覆盖默认） |
| `CUDA_VISIBLE_DEVICES` | `$GPU_VOICE` | 显式设置时不再被 `GPU_VOICE` 覆盖 |
| `VOICE_PORT` | `3000` | 写进 `config.PORT` |
| `GPU_VOICE` | `1` | 语音栈用的物理卡号 |
| `DEEPSEEK_API_KEY` | 空 | LLM 凭据；缺失会导致预检失败 |
| `RAG_DEVICE` | `cpu` | 同 `config.py` |
| `VA_SKIP_PREFLIGHT` | `0` | `1` = 跳过预检（应急） |
| `VA_LOG_DIR` | `<仓库>/logs` | 日志目录 |
| `FORCE` | `0` | `clean-gpu` 时 `1` = 真 kill |
| `CHECK_INTERVAL` / `CHECK_TIMEOUT` | `30` / `300` | 守护巡检节奏（秒） |
| `ENABLE_TUNNEL` / `VPS_IP` | 关 / 空 | 公网反向隧道，没有 VPS 保持关闭 |
| `CLEAN_ORPHANS` | `0` | 守护重启前强杀 GPU 孤儿；**多人共用显卡时勿开** |

**成功启动的日志**（依次加载各引擎，真实格式带时间戳与命名空间）：

```
2026-11-22 11:03:47,984 [INFO] Voice Agent Server === Loading engines (v3.0) ===
2026-11-22 11:03:47,984 [INFO] Voice Agent Server Loading VAD (Silero)...
2026-11-22 11:03:48,009 [INFO] Voice Agent Server Loading ASR (Parakeet-TDT)...
2026-11-22 11:03:51,685 [INFO] Voice Agent Server Loading RAG (bge-m3, threshold=0.35)...
2026-11-22 11:03:56,276 [INFO] Voice Agent Server RAG index: 59 docs, dim=1024
2026-11-22 11:03:56,276 [INFO] Voice Agent Server LLM client ready (deepseek @ ...)
2026-11-22 11:03:56,276 [INFO] Voice Agent Server Loading TTS (Kokoro-82M, voice=af_heart)...
2026-11-22 11:03:57,290 [INFO] Voice Agent.tts Kokoro TTS loaded (dir=.../models/Kokoro-82M, ...)
2026-11-22 11:03:57,290 [INFO] Voice Agent Server Warming up TTS...
2026-11-22 11:03:58,566 [INFO] Voice Agent Server === All engines loaded (v3.0) ===
```

**日志命名空间**统一成 `Voice Agent.*`，便于按模块过滤：

| 命名空间 | 覆盖 |
|---|---|
| `Voice Agent Server` | 装配、连接生命周期、路由 |
| `Voice Agent.cm` | 对话状态机（`engine/dialog/` 全部 mixin） |
| `Voice Agent.tts` | TTS 加载与合成 |

```bash
grep "Voice Agent" logs/voice_agent.log            # 整条 pipeline
grep "Voice Agent.cm" logs/voice_agent.log         # 只看状态机
grep -v "Voice Agent" logs/voice_agent.log         # 滤掉本项目，只看三方库（如 faiss）的噪音
```

版本号来自 `config.VERSION`（当前 `3.0`），日志、`/api/info`、WS `ready` 都用它。

---

## 7. 验证

```bash
curl -s http://localhost:3000/api/info    | python -m json.tool   # 模型名 / 采样率
curl -s http://localhost:3000/api/metrics | python -m json.tool   # 延迟统计
curl -s "http://localhost:3000/api/rag/query?q=你们公司叫什么名字" | python -m json.tool
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000/   # 前端页面，应 200
```

**浏览器对话**：打开 `http://<IP>:3000/` → 允许麦克风 → 收到问候语 → 说一句，
应看到状态灯 `idle → listening → thinking → speaking`，以及 ASR 文本、RAG 命中、AI 回复与音频。

> 浏览器要求**用户手势**后才能播放音频（自动播放策略）。没声音先点一下页面。

**观察延迟**：

```bash
watch -n1 'curl -s http://localhost:3000/api/metrics | python -m json.tool'
```

关注 `first_response_ms`：ASR + RAG + LLM 首句 + TTS 首段，**不含** VAD 判停等待。

---

## 8. 数据与音色

### 8.1 知识库（RAG）

`data/sample_kb.json` 已在仓库里（59 条，键为 `id` / `question` / `answer`）。
启动时自动建 FAISS 索引（**只在内存里**，所以每次启动重新编码一遍——59 条很快）。
热更新用 `POST /api/rag/reload`。

**检索质量怎么验**：`python test/test_rag.py`（可加 `RAG_DEVICE=cpu`）会跑 10 条英文问句，
其中 2 条是知识库里没有的负例，并给出"正例最低分 / 负例最高分"。当前实测：

```
正例 8 条：0.486 ~ 0.800（全部命中正确条目）
负例 2 条：0.454 / 0.477（硬凑到了不相关的条目）
→ 可分离窗口仅 (0.477, 0.486)，窄到无法靠单一阈值可靠区分
```

所以 `RAG_SCORE_THRESHOLD=0.35` 主要起"挡掉明显不相关的"作用，**不能指望它判断"知识库有没有答案"**——
兜底靠 `SYSTEM_PROMPT_RAG` 里的"没覆盖就说 Let me check that"指令。

### 8.2 TTS 音色（固定，不可切换）

TTS 是 **Kokoro-82M**：用 11 个预置 voicepack，**没有"零样本音色克隆"这回事**——
旧版 VoxCPM 那套（`VOICE_PROMPT_WAV` / `register_voice()`）已整体移除。

本项目**只用默认音色**，且不提供切换入口：

```python
# engine/tts.py
class KokoroTTS:
    VOICE = "af_heart"                    # 标准美式英语女声
    LANG_CODE = VOICE.split("_")[0][0]    # 由前缀推导 → 'a'（美式），配不错
```

选 `af_heart` 的理由：社区评级 **A**，韵律最自然、长段落不易听疲劳，正适合雅思听力陪练。

`synthesize()` / `synthesize_stream()` **没有 `voice` 参数**，也没有 `set_default_voice()`。
真要换音色只能改上面那个常量（改完重启），备选清单：

| 音色 | 特点 |
|---|---|
| **`af_heart`** ← 当前 | 评级 A，韵律最自然，最适合雅思听力陪练 |
| `af_bella` | 评级 A-，更"清晰专业"，长文本有轻微机械感 |
| `af_sarah` | 清晰中性，接近播报风格 |
| `af_nicole` | 气声偏多，适合旁白、不适合考试语境 |
| `af_sky` / `af_nova` / `af_river` / `af_kore` / `af_jessica` / `af_aoede` / `af_alloy` | 其余备选 |

唯一能调的 TTS 参数是**语速**：

```bash
KOKORO_SPEED=0.9 bash run.sh     # 1.0 = 自然速；降到 0.9 让学员听得更清
```

---

## 9. 排障

| 现象 | 原因 / 处理 |
|---|---|
| 预检报 `缺少 DEEPSEEK_API_KEY` | 填 `.env` 或 `export`；从 platform.deepseek.com 获取 |
| 预检报 `缺少 kokoro` / `misaki` | `pip install kokoro==0.9.4`（会带上 `misaki[en]`） |
| 预检报 `权重缺失` | 见 [4. 下载模型](#4-下载模型) |
| 卡在 `Loading VAD` 很久 | Silero 找不到本地 `silero_vad.jit` 会**联网**去 torch.hub 下载 |
| `Loading ASR` 报 `safetensors` / `incomplete metadata` | Parakeet 权重没下完。`du -sh models/parakeet-tdt-0.6b-v3` 应约 2.5G；用 `wget -c` 续传 |
| 预检报 `ASR 权重目录不完整（权重体积不足…）` | 同上：只有 `config.json`/`tokenizer.json` 下完了，`model.safetensors` 还缺 |
| ASR 一直识别为空串 | 音频太轻或纯静音。`bash run.sh status` 看 ASR 模型对不对，再看麦克风增益 |
| `Loading TTS` 阶段 CUDA OOM | 最常见。用 `nvidia-smi` 看是谁占着 → `bash run.sh clean-gpu` 或 `bash run.sh stop` |
| 声音变调/变速 | 采样率错配。上行必须 **16k**、下行必须 **24k**；错配不报错只变调 |
| 能识别但没声音 | TTS 没装好（`kokoro`/`misaki`）；看日志 `Loading TTS` 之后有没有报错 |
| 打断后还能听到半句 | WebSocket 走 TCP 的已知特性；前端靠 `playableTurn` 过滤。要更彻底只能换 UDP |
| 一直提示"没太听清" | 连续 3 次 ASR 空结果会升级提示。检查麦克风格式、环境噪声 |
| 前端连不上 | 检查 `PORT`、防火墙（`sudo ufw allow 3000`）；HTTPS 页面连 ws 需 `USE_SSL` |
| `401 Authentication Fails`（DeepSeek） | key 没设或填错；`echo $DEEPSEEK_API_KEY` 确认 |

---

## 10. 生产化（常驻与开机自启）

| 方式 | 命令 | 说明 |
|---|---|---|
| 前台托管 | `bash run.sh`（或 `dev`） | 开发调试；Ctrl+C 全停 |
| **常驻 + 内置守护**（推荐） | `bash run.sh start` | 服务脱离终端；`watchdog` 每 30s 巡检 Voice Agent，掉线自动拉起 |
| systemd | 自己写 unit | 能**开机自启**、崩溃按策略重启、日志进 journald |

**为什么仍然推荐 systemd**：`run.sh` 的守护是用户态轮询 —— 机器重启后就没了。

```ini
# /etc/systemd/system/voiceagent.service
[Unit]
Description=VoiceAgent-600ms
After=network.target

[Service]
Type=simple
User=<你的用户名>
WorkingDirectory=<仓库绝对路径>
Environment=VOICEAGENT_PYTHON=<解释器绝对路径>
Environment=DEEPSEEK_API_KEY=<你的 key>
Environment=CUDA_VISIBLE_DEVICES=0
Environment=ASR_DEVICE=cuda:0
Environment=TTS_DEVICE=cuda:0
Environment=RAG_DEVICE=cpu
Environment=PORT=3000
# 直接跑入口；它会在 __main__ 里调 load_engines() 后起 uvicorn
ExecStart=<解释器绝对路径> <仓库绝对路径>/ws_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voiceagent
journalctl -u voiceagent -f          # 看日志（日志进 journald，不用 logs/）
```

> 📌 **预检不要放进 `ExecStart`**：systemd 的 `ExecStart` 只能有一条主命令（多条要写成
> `ExecStartPre`），而且预检失败会让 unit 进入 failed 状态、被 `Restart=always` 反复重启。
> 部署时手动跑一次 `bash run.sh preflight all` 即可。
>
> ⚠️ `WorkingDirectory` 必须设成仓库根目录 —— `config.py` 用 `__file__` 定位 `models/`，
> 但 `run.sh` 与相对路径的日志位置依赖 cwd。


# 第三部分 · 读代码

## 建议的阅读顺序

**不要顺着文件从头到尾读**。状态机在 `engine/dialog/`：`manager.py` 是门面，另有 9 个
按职责拆分的模块。**按调用链读**效率高得多：

```
浏览器 static/js/audio.js（AudioWorklet 采集 512 点/32ms）
   ↓ WebSocket 二进制帧
server/routes_voice.py  ws_voice()      ← 连接生命周期、消息分发
   └─ ConversationManager.feed_audio()  ← 音频入口
        └─ _process_chunk()             ← ★ 状态机核心
             ├─ VAD.process_chunk()     ← 逐帧判语音
             ├─ _adaptive_endpoint_threshold()  ← 判停档位
             └─ _on_endpointing()       ← 判停触发
                  └─ _run_pipeline()    ← ★ 一轮编排
                       ├─ ASR.transcribe()
                       ├─ RAG.get_context()
                       └─ _stream_llm_tts()      ← ★ 流式串联
                            ├─ LLMChat.stream_sentences()    逐句
                            └─ KokoroTTS.synthesize_stream() 逐段
                                 └─ _send({"_audio": ...}) → 前端播放
```

### 三条关键链路，读懂了项目就懂了大半

| 链路 | 入口 | 要点 |
|---|---|---|
| **首响** | `server/loader.py: load_engines()` → `engine/dialog/streaming.py: _stream_llm_tts()` | LLM 逐句 yield、TTS 逐段下发；首段一出声就发 `audio_start` |
| **判停** | `engine/dialog/manager.py: _process_chunk()` → `endpointing.py` | 只数静音帧；档位按说话时长自适应 |
| **打断** | `engine/dialog/manager.py: handle_barge_in()` + `_process_chunk` 里的 VAD 分支 | 双门限 + 2 帧确认；服务端停发、前端 flush |

### 学习节奏建议

| 阶段 | 读什么 |
|---|---|
| 第 1 天 | `config.py`（8 个分区）→ 本文档第 2 节的时序图 → `engine/__init__.py` 的结构图 |
| 第 2~3 天 | `server/loader.py` 的 `load_engines()`（装配顺序）→ `ws_voice()`（消息分发） |
| 第 4~5 天 | `engine/dialog/manager.py`：先读模块顶部总览，再按 5 个状态分支读 `_process_chunk()` |
| 第 2 周 | 按需深入：`engine/llm.py`（切句策略）、`engine/tts.py`（流式）、`pipeline.py`（编排） |

> 💡 本仓库的注释**只写"为什么"，不复述代码**——所以看到没有注释的行是正常的，
> 那说明它一眼就懂。遇到看不懂的常量，去 `config.py` 或模块顶部的注释找来历；
> 踩过的坑统一搜 `⚠️` 和 `注意`。

---

## 附录 A：已知的坑

**信代码，不要信名字。** 以下是逐文件核实确认的偏差：

| # | 位置 | 现象 |
|---|---|---|
| 1 | `engine/asr.py` | 名字像"ASR 抽象层/工厂"，**实际就是唯一的 Parakeet-TDT 实现**。仓库已精简为单实现，没有多后端分支 |
| 2 | `engine/rag.py` | `RAG_SCORE_THRESHOLD=0.35` 是按**余弦相似度**标定的；一旦启用 reranker，比较的是 **logit 量纲**的分数，**两者不等价**，需重新标定 |
| 3 | `engine/dialog/transport.py` | `_send_audio()` 是**死代码**（当前零调用点），`TTS_SEND_CHUNK_BYTES` 只被它使用。真实下发在 `_stream_llm_tts` / `_play_system_tts` 里逐段进行 |
| 4 | `engine/dialog/endpointing.py` | 短句（<0.5s）与长句（>3s）**都返回 SLOW 档**（640ms），只有中段用 DEFAULT（416ms）。名字叫"自适应"，实际是两档 |

> **已修复的历史问题**（最近一轮重构中处理，留档备查）：
> `config.py` 的 `LLM_GPU`/`ASR_GPU`/`TTS_GPU`/`RAG_GPU` 四个死变量（已删）；
> `ENDPOINT_FAST_CHUNKS` 这个从未被返回的死常量（已删）；
> `SYSTEM_PROMPT` / `RAG_INDEX_PATH` 两个零引用配置（已删）；
> `KOKORO_VOICE` / `KOKORO_LANG_CODE` 两个配置项（音色改为固定常量）；
> `run.sh` 与 `config.py` 的模型名/端口漂移（已用 `va_cfg` 统一读取）。

---

## 附录 B：常见任务

| 任务 | 怎么做 |
|---|---|
| **换话术 / 人设** | 改 `config.py` §8 的 `SYSTEM_PROMPT_RAG`，§7 的 `GREETING_TEXT` / `IDLE_*` / `ASR_*`；改完重启 |
| **调语速** | `KOKORO_SPEED=0.9 bash run.sh` |
| **换音色** | 只能改 `engine/tts.py` 的 `KokoroTTS.VOICE` 后重启（默认不提供切换） |
| **调判停灵敏度** | `engine/dialog/constants.py` 的 `ENDPOINT_*_CHUNKS`。调小更灵敏，但更容易切断用户 |
| **换 RAG 知识库** | 改 `data/sample_kb.json`（数组，元素含 `id`/`question`/`answer`），然后 `POST /api/rag/reload` 热重建 |
| **换 ASR** | 新类实现 `transcribe(audio, sr) -> {"text","latency_ms"}`，改 `server/loader.py` 的装配行 |
| **换 LLM** | 任何 OpenAI 兼容服务：改 `LLM_BASE_URL` / `LLM_MODEL_NAME` / `LLM_API_KEY` |
| **加一个系统话术** | 在 `config.py` 加常量，`server/routes_voice.py` 的 `build_experience_config()` 里放进 exp_cfg，`engine/dialog/` 里调用 |

---

## 附录 C：术语表

| 术语 | 含义 |
|---|---|
| **首响（first response）** | 用户说完到听到 AI 第一段音频。目标 < 600ms，实测 458ms |
| **判停（endpointing）** | 判断"用户说完了没有"。本项目只数连续静音帧 |
| **打断（barge-in）** | AI 说话时用户插话，AI 立即停止 |
| **Turn** | 一轮对话的序列号，用于丢弃旧轮次的在途音频 |
| **投机 ASR** | 判停前提前跑 ASR，判停后复用结果 |
| **D1~D4** | 体验层需求编号：D1 问候 / D2 空闲超时 / D3 错误恢复 / D4 告别识别 |
| **多继承组合** | `ConversationManager` 由 6 个职责基类（`Session` / `Pipeline` / `Streaming` / `Transport` / `Endpointing` / `AsrAccumulator`）组合而成，共享同一份 `self` 状态 |

---

## 附录 D：本文档的依据

**已核实（直接读代码 / 跑命令确认）**：
- 依赖清单：从全部 `.py` 的 import 反推，与 `requirements.txt` 的 16 个 pin 逐版本比对；
- GPU 占用者只有 ASR / TTS：`engine/vad.py`（`map_location="cpu"`）、`engine/rag.py`（默认 cpu）；
- 配置集中度：8 个分区全在 `config.py`，`server/` 只管进程启动；
- 端口与协议：`server/routes_*.py` 的路由定义（无鉴权）；
- `data/sample_kb.json` 实为 59 条、键为 `id`/`question`/`answer`。

**未核实（受环境限制，需按官方文档确认）**：
- 各模型在 ModelScope/HuggingFace 上的**当期**模型 ID 与下载命令；
- Kokoro 的**当期**音色清单与 `kokoro` 包版本；
- 458ms / 160ms 等实测数字来自历史记录，需按第二部分真正跑起来后复测。

遇到与本文不符之处，**以代码和官方文档为准**。
