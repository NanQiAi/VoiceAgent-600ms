<div align="center">

# 🎙️ VoiceAgent-600ms

### Voice · ASR · TTS · LLM · Infra

**级联 Pipeline:VAD → ASR → RAG → LLM → TTS — 全部低于 600ms 感知预算**

<br>

[![Pipeline](https://img.shields.io/badge/Cascade_Pipeline-458ms-00c853?style=for-the-badge&logo=speedtest&logoColor=white)](/)
[![Barge-in](https://img.shields.io/badge/Barge--in-160ms-ff6f00?style=for-the-badge&logo=bolt&logoColor=white)](/)
[![ASR CER](https://img.shields.io/badge/ASR_CER-2.89%25-9c27b0?style=for-the-badge&logo=microphone&logoColor=white)](/)
[![License](https://img.shields.io/badge/Apache-2.0-7c4dff?style=for-the-badge)](/)

<br>

> **800ms 是语音对话的图灵测试门槛**——超过它,人类会感知到"在和机器说话"。
> 本项目以 **600ms 为工程预算线**,实测 **458ms** 首响,远低于预算。

</div>

---

## 📖 这份文档怎么读

本文档把"认识项目 → 部署上线 → 精读代码"三件事**合成了一份**。按你的目的选路径：

| 你的目的 | 从这里开始 | 预计耗时 |
|---|---|---|
| 🚀 **我要把它跑起来** | 直接跳到 [第二部分 · 部署](#第二部分--部署debian-13--双卡-rtx-3090-ti)，但**务必先读** [第 0 节 开始之前：四个坑](#4-开始之前先看这四个坑) | 1~4 小时（含下载） |
| 📚 **我要读懂代码** | 先读 [第一部分](#第一部分--认识项目)（建立全局观），再进 [第三部分 · 精读代码](#第三部分--精读代码12-课) | 1~2 周 |
| 🔧 **我在改它 / 排查问题** | [第 5 节 通信协议](#5-通信协议websocket-消息全集)、[第 6 节 配置速查](#6-配置速查configpy)、[常见错误排查](#15-常见错误排查)、[附录 A 不一致清单](#附录-a文档与代码不一致清单实测重要) | 按需查阅 |
| ⏱️ **我只有 30 分钟** | [第三部分末尾的 30 分钟路径](#如果只有-30-分钟) | 30 分钟 |

**全文结构**

- **第一部分 · 认识项目**（第 1–6 节）：项目定位、一次对话的完整数据流、仓库地图、核心概念、通信协议、配置速查
- **第二部分 · 部署**（第 1–18 节）：Debian 13 + 双卡 3090 Ti 的完整 step-by-step 部署
- **第三部分 · 精读代码**（12 课）：按调用链带你读一遍代码，每课含精确行号锚点与自测题
- **附录**：文档与代码不一致清单（实测）、常见任务、术语表、事实来源

> ⚠️ **关于准确性**：本文档**基于当前代码逐文件核实**写成，不是照抄设计稿。
> 凡是我实测发现"代码与名称/注释/设计稿不一致"的地方，都汇总在
> [附录 A](#附录-a文档与代码不一致清单实测重要)——**代码才是唯一事实来源**。

---

# 第一部分 · 认识项目

## 1. 项目是什么

**一句话**：一个**低延迟中文语音对话 Agent**（级联 Pipeline 架构）。工程预算线是 **600ms 首响**——因为超过 800ms，人类会明显感觉"在跟机器说话"。

```
级联 Pipeline（仓库根目录）
  VAD → ASR(FireRedASR2) → RAG(bge+FAISS) → LLM(Qwen3-14B/vLLM) → TTS(VoxCPM 流式)
  458ms 首响 · 每个组件可独立替换
```

**为什么是级联,而不是端到端大模型?** 可控性。级联的每个组件能单独换（换音色是语音克隆、小时级;换 ASR/LLM/TTS 都是替换单个模块）;端到端方案要改行为得重训整个多模态模型。而两条路线的延迟都已低于人类感知门槛,**此时可控性 > 更低的延迟**。

### 核心指标 (RTX 4090)

| 指标 | 实测 |
|:---:|:---:|
| 首响延迟 | **458ms** |
| 打断精度 | **160ms** |
| 换音色 | 语音克隆 · 小时级 |
| 换 ASR / LLM / TTS | 替换单个模块 |

### 可替换组件

| 组件 | 当前 | 可替换为 |
|---|---|---|
| ASR | FireRedASR2-AED | Whisper · Paraformer · SenseVoice |
| LLM | Qwen3-14B-AWQ / vLLM | 任何 OpenAI 兼容服务(DeepSeek 等) |
| TTS | VoxCPM 1.5 / nanovllm | CosyVoice · IndexTTS · Fish-Speech |
| VAD | Silero / ECAPA-TDNN | FireRedVAD · WebRTC VAD |
| RAG | bge-small + FAISS | bge-m3 · Milvus · Elasticsearch |
| 传输 | WebSocket | LiveKit WebRTC · SIP |

### 版本演进

| 版本 | 核心改动 |
|---|---|
| v1.0 – v2.9 | 全双工 pipeline → 5 状态机 → 双层打断 → FireRedASR2 → 崩溃恢复 |
| v3.0 | Qwen3-14B-AWQ、声纹门控、ASR 文本累积 |
| v3.1 | PTT Demo Mode、句子 Cap、Watchdog 守护 |
| 2026-10 | 移除端到端 Omni 路线（原 Hybrid-VoiceAgent），仓库聚焦级联 Pipeline |

## 2. 一次完整对话发生了什么（核心）

这是**最该先建立的心智模型**。下面是一次"用户说完一句话、Agent 回答"的完整链路。

```
时间轴 ──────────────────────────────────────────────────────────────────►

① 用户说话
   浏览器麦克风 ──AudioWorklet──► 16kHz / 单声道 / int16 ──WebSocket 二进制帧──► ws_server
   每帧 512 采样点 = 32ms（CHUNK_SAMPLES=512, CHUNK_MS=32）

② VAD 逐帧判断"这帧有没有人声"
   ├─ 可选插一层 DTLN 降噪（USE_DENOISE=1）
   └─ 可选插一层声纹门控（USE_SPEAKER_VAD=1）：不是"本人"说话就丢弃，防电视/旁人误触发

③ Endpointing 判停：连续静音多少帧算"说完了"
   ├─ 说话 <0.5s  → 等 20 帧 ≈ 640ms
   ├─ 0.5~3s      → 等 13 帧 ≈ 416ms   ← 最常用
   └─ >3s         → 等 20 帧 ≈ 640ms
   状态机：IDLE → LISTENING →（判停）→ THINKING

④ 进入 THINKING，四件事并发/接续发生
   ├─ ASR：把累积音频转文字（FireRedASR2，约 200ms）
   ├─ RAG：拿问题去知识库检索（bge + FAISS，约 4ms）
   ├─ Filler：若 LLM 首 token 来得慢，先播一句"嗯，""好的，"
   └─ LLM：拼 system prompt(+RAG 上下文) + 历史，流式生成（首 token 约 163ms）

⑤ LLM 流式输出 → 就近按标点切句 → 逐句丢给 TTS
   TTS 流式合成（VoxCPM，首 chunk 约 174ms），每个 chunk 发之前检查"是否被打断"

⑥ 音频回传：服务端先发 {"type":"audio_start","turn":N} 授权本轮，再发二进制音频帧
   前端 playableTurn 机制丢弃被打断轮次的在途帧

⑦ 服务端进入 SPEAKING，期间用户若开口 → Barge-in（打断）
```

**首响延迟 = ASR + RAG + LLM 首句 + TTS 首 chunk**，注意**不含** VAD 判停那 416~640ms 的等待。

实测值（RTX 4090）：

| 环节 | 耗时 | 代码位置 |
|---|---|---|
| ASR（FireRedASR2-AED） | ~200ms | `engine/asr_firered.py` → `engine/fireredasr2/asr.py` |
| RAG（bge-small + FAISS） | ~4ms | `engine/rag.py` |
| LLM 首 token（Qwen3-14B-AWQ / vLLM） | ~163ms | `engine/llm.py` |
| TTS 首 chunk（VoxCPM） | ~174ms | `engine/tts.py` |
| **合计首响** | **~458ms** | |

> ⚠️ 这些数字来自实测记录，**不是**代码里硬编码的常量。要复现需要按第二部分把服务真正跑起来后实测。

**为什么能这么快**（6 个工程优化，详细原理见[第三部分](#第三部分--精读代码12-课)）：

| # | 优化 | 效果 |
|---|---|---|
| 1 | **流式 TTS 打断** — `synthesize_stream()` 逐 chunk yield，每个 chunk 发送前检查 `_cancel_speaking` | 打断精度 160ms |
| 2 | **事件循环让出** — `asyncio.sleep(0.05)` 强制间隔，让循环有时间处理麦克风 chunk | 修好 SPEAKING 态打断 |
| 3 | **Turn 序列号过滤** — 每轮发 `audio_start(turn=N)`，前端 `playableTurn` 丢弃不匹配帧 | 消除在途帧串音 |
| 4 | **投机式预推理** — 用户停顿 160ms 时就后台跑 Moonshine Tiny(27M, CPU) | 省 ~117ms |
| 5 | **自适应 Endpointing + ASR 文本累积** — 按说话时长选判停档位；短句不立即送 LLM | 减少误切断 |
| 6 | **PTT Demo Mode** — 按住说话，跳过 VAD/判停/filler | 排障隔离手段 |

## 3. 仓库地图

| 文件 | 职责 | 新人优先级 |
|---|---|---|
| `config.py` | **全项目配置中枢**：路径、设备、采样率、阈值、话术、system prompt | ⭐⭐⭐ 第一个读 |
| `ws_server.py` | **主力服务入口**：FastAPI + WebSocket，装配所有引擎，处理所有消息 | ⭐⭐⭐ 第二个读 |
| `engine/conversation_manager.py` | **对话状态机核心**（全项目最大文件）：状态流转、判停、打断、filler、空闲超时 | ⭐⭐⭐ 最难，最后啃 |
| `engine/llm.py` | vLLM 调用 + 流式切句 + 历史/摘要管理 + think 清洗 | ⭐⭐ |
| `engine/tts.py` | VoxCPM 流式合成 + 音色克隆 + 打断检查 | ⭐⭐ |
| `engine/rag.py` | bge 向量化 + FAISS 检索 + 置信度阈值 | ⭐⭐ |
| `engine/asr.py` | ASR 默认实现（SenseVoiceSmall / FunASR） | ⭐⭐ |
| `engine/asr_firered.py` | FireRedASR2 适配层（生产默认走的这条） | ⭐⭐ |
| `engine/asr_moonshine.py` | Moonshine Tiny 27M（CPU/ONNX），用于**投机式 ASR 预推理** | ⭐ |
| `engine/vad.py` | Silero VAD | ⭐⭐ |
| `engine/speaker_vad.py` | 声纹感知 VAD（Silero + ECAPA-TDNN） | ⭐ |
| `engine/firered_vad.py` | FireRedVAD（另一套 VAD 实现，可替换） | ⭐ |
| `engine/turn_detector.py` | Smart Turn v3 语义判停 | ⭐ |
| `engine/denoiser.py` | DTLN 降噪 | ⭐ |
| `engine/filler.py` | 语气词预合成与轮询取用 | ⭐ |
| `engine/captioner.py` | 旁路组件：给音频生成一句"语音观察"注入 LLM 上下文 | ⭐ |
| `engine/fireredasr2/` | **FireRedASR2 上游开源代码（vendored）**，见下方警告 | ⭐ |
| `static/voice_agent.html` | 前端：AudioWorklet 播放器、麦克风采集、打断、Turn 过滤 | ⭐⭐ |
| `voiceagent_env.sh` | **共享配置层**（不是启动脚本）：模型路径、GPU 分配、端口、依赖预检都在这 | ⭐⭐ |
| `start_all.sh` | 一键起 vLLM + Voice Agent（前台托管，Ctrl+C 停） | ⭐⭐ |
| `start_prod.sh` | `start`/`stop`/`status`/`restart`，服务常驻 + 自动守护 | ⭐⭐ |
| `keep_alive.sh` | 看护循环：vLLM / Voice Agent / 隧道掉线自动拉起 | ⭐ |
| `start_vllm.sh` | 只启动 LLM 服务（调试用） | ⭐⭐ |
| `test_rag.py` | RAG 检索方案的 A/B 基准脚本（bge-small/bge-m3 × 有无 reranker） | ⭐ |
| `check_deps.py` | 依赖自检：逐个 import 并打印可用/缺失，用来确认环境装全了 | ⭐ |
| `models/download_model.py` | 用 modelscope 批量下载 7 个权重（⚠️ `local_dir` 写死为本机绝对路径，用前先改，见 9.2） | ⭐ |

**传输方式**：WebSocket（`ws_server.py` + `static/voice_agent.html`）。
走 TCP，所以打断需要前端显式 `flush` 播放队列（见第四部分 Barge-in 一节）。

> 历史上还有一条 WebRTC(LiveKit) 接入分支（`livekit_agent/` + `start_livekit.sh` +
> `tts_server.py` + `local_server.py`），**当前代码里已全部移除**，本仓库只保留
> WebSocket 这一条。

**`data/` 目录里有什么**：

| 路径 | 用途 |
|---|---|
| `data/sample_kb.json` | **RAG 知识库**（59 条问答）——只有它被代码读取（`config.KB_DATA_PATH`） |
| `data/tts_chinese_dataset/` | 语音数据集（3.8 万文件 / 15 GB），**当前代码不读它**；已在 `.gitignore` 里 |
| 其余 `*.json` | 历史评测结果（final_comparison / outbound_* / sft_comparison），**代码不读**，可自行清理 |

> ⚠️ **`engine/fireredasr2/` 是 vendored 上游代码**：整份小红书 FireRedASR2 源码原样放进仓库。**读它之前先明确**：主链路只用到 `asr.py`（`FireRedAsr2` 类）和 `data/asr_feat.py`（特征提取）；`speech2text.py` 是上游自带的离线批量 CLI、`utils/wer.py` 是离线 CER/WER 评测（这几个都在 `engine/fireredasr2/` 下），**在线链路都不走它们**。改动这个目录要格外小心（升级上游时会冲突）。


---

## 4. 核心概念速通

### 4.1 VAD（Voice Activity Detection）
逐帧判断"这 32ms 里有没有人声"。项目里有**三套可换**实现，用环境变量选：

| 开关 | 实现 | 说明 |
|---|---|---|
| `USE_FIRERED_VAD=1` | FireRedVAD | 优先级最高 |
| `USE_SPEAKER_VAD=1` | Silero + ECAPA-TDNN | 声纹门控版 |
| 默认 | Silero VAD | 兜底 |

### 4.2 Endpointing（判停）
"用户说完了没有"。**三档自适应阈值**（`engine/conversation_manager.py`）：

| 说话时长 | 静音阈值 | 帧数 | 场景 |
|---|---|---|---|
| < 0.5s | ~256ms | 8 帧 | "嗯……"思考中，别急着接话 |
| 0.5~3s | ~416ms | 13 帧 | 正常对话 |
| > 3s | ~640ms | 20 帧 | 长句 |

另可选 `USE_SMART_TURN=1` 启用 Smart Turn v3 做**语义**判停（不只靠静音）。

### 4.3 Barge-in（打断）—— 本项目的重点
**双层结构，且"没有人 cancel 别人"**：

- **第一层**：状态机把 `_cancel_speaking` 这个 Event 置位；
- **第二层**：三条发送/合成循环（LLM 句子循环、TTS 逐 chunk 循环、系统话术循环）**各自**检查该 Event 并主动退出。

这样打断粒度 = **1 个音频 chunk（约 160ms）**，而不是"等整句合成完"。

触发条件（`engine/conversation_manager.py` 常量）：
```
BARGE_IN_VAD_THRESHOLD  = 0.6     # VAD 语音概率
BARGE_IN_RMS_THRESHOLD  = 0.012   # 音量门限（防噪声误触发）
BARGE_IN_CONFIRM_CHUNKS = 2       # 连续 2 帧 ≈ 64ms 才确认，防抖
```

### 4.4 Turn 序列号（防串音）
服务端每轮回复前发 `{"type":"audio_start","turn":N}`。前端维护 `playableTurn`：

- `-1` = 尚未授权
- `N` = 只有第 N 轮的音频帧可播
- `0` = 打断后的"闸门关闭"哨兵

不匹配的帧**全部丢弃**——这解决了"打断后，之前已发出去但还没播的音频又冒出来"的问题。

### 4.5 Filler（语气词）
VAD 判停后到 LLM 首 token 之间有几秒静默。`engine/filler.py` 在**服务启动时**把 5 条语气词（"嗯，""好的，"…）预合成成裸 PCM 缓存，运行时取到即发，延迟≈0。

> 为什么语气词末尾都带逗号？为了让它用"话没说完"的升调读出来，听起来像"我在听"而不是"我说完了"——避免用户误以为轮到自己而抢话。

### 4.6 投机式 ASR（Speculative ASR）
用户停顿约 160ms（5 帧）时，后台先用 Moonshine Tiny（27M，CPU）**抢跑**一次 ASR。如果随后 Endpointing 确认说完了，就直接复用这次结果，**省掉约 117ms**。用 `USE_MOONSHINE_ASR=1` 开启。

### 4.7 声纹门控（Speaker Gate）
`engine/speaker_vad.py`：Silero 判断"有没有人声"，ECAPA-TDNN 判断"是不是本人"（1 秒音频编码成 192 维向量，算余弦相似度，阈值 0.25）。

用途：只有本人说话才开门，防电视/旁人声音误触发。**未注册时一律放行**。

### 4.8 ASR 文本累积
短句（≤4 字，如"嗯""什么"）**不立即送 LLM**，先缓冲，等后续语音拼起来再一起送——避免把一句话切成两轮对话。

```
ASR_ACCUM_MAX_CHARS    = 4     # ≤4 字就缓冲
ASR_ACCUM_FLUSH_S      = 3.0   # 3 秒内没有新语音就直接送出去
ASR_ACCUM_MAX_SEGMENTS = 3     # 最多攒 3 段，防无限累积
```

### 4.9 PTT / Demo 模式
`DEMO_MODE=1`：按住说话（`ptt_start`）→ 音频先缓存在本地 → 松手（`ptt_end`）→ 整段一次性走 **ASR → RAG → LLM → TTS**。

与全双工的确切差异（`engine/conversation_manager.py: process_ptt()`）：
- ✅ **没有** VAD 判停、**没有**语气词 filler、**没有**打断路径（音频是整段到达的）
- ✅ 仍然**保留**降噪与错误兜底；也**不主动播问候语**
- 握手时 `ready.mode` 会告诉前端是 `"demo"` 还是 `"duplex"`

用途：演示，以及**隔离问题**——如果 PTT 正常而全双工不正常，问题就在 VAD/判停这一层。

---

## 5. 通信协议：WebSocket 消息全集

连接地址：`ws://<host>:3000/ws/voice`（可选 `?token=<API_KEY>`）。

### 5.1 上行（客户端 → 服务端）

| 形式 | 说明 |
|---|---|
| **二进制帧** | 16kHz / 单声道 / int16 裸 PCM，每帧 512 点 = 32ms。服务端按 `/32768` 归一化 |
| `{"type":"ptt_start"}` | 按住说话开始（仅 DEMO_MODE） |
| `{"type":"ptt_end"}` | 松手，触发处理 |
| `{"type":"text_input","text":"..."}` | 直接发文本（跳过 ASR，用于调试） |
| `{"type":"manual_barge_in"}` | 前端手动打断 |
| `{"type":"enroll"}` / `{"type":"unenroll"}` | 声纹登记 / 注销 |
| `{"type":"reset"}` | 重置会话 |

### 5.2 下行（服务端 → 客户端）

| type | 关键字段 | 含义 |
|---|---|---|
| `ready` | `session_id` / `version` / `mode` | 握手完成 |
| `state` | `state` | 状态机：`idle`/`listening`/`thinking`/`speaking`/`interrupted` |
| `asr` | `text` / `asr_ms` | 识别结果 |
| `rag` | `context` / `score` / `rag_ms` | 检索到的知识库内容 |
| `filler` | （text + 二进制音频） | 语气词 |
| `llm_sentence` | `text` | LLM 切出的单句（逐句下发给 TTS） |
| `llm` | `text` | LLM 完整回复 |
| `audio_start` | `turn` / `sample_rate` | **授权本轮音频**，前端据此设 playableTurn |
| `metrics` | `asr_ms`/`rag_ms`/`llm_ms`/`tts_ttfa_ms`/`first_response_ms` | 延迟指标 |
| `barge_in` | — | 通知前端清播放队列 |
| `caption` | `text` | 旁路组件的"语音观察" |
| `shield_status` | 声纹盾状态 | enroll 结果 |
| `system_message` / `error` / `session_end` | `text` | 系统话术/错误/结束 |
| `ptt_ack` / `enroll_ack` / `unenroll_ack` / `reset_ack` | — | 各类确认 |

### 5.3 音频格式约定（**最容易写错的地方**）

| 方向 | 采样率 | 格式 | 备注 |
|---|---|---|---|
| 上行（麦克风） | **16000 Hz** | 单声道 int16 裸 PCM | 无 WAV 头；必须与 `config.SAMPLE_RATE`、`CHUNK_SAMPLES=512` 严格一致 |
| 下行（扬声器） | **44100 Hz** | 单声道 int16 裸 PCM | VoxCPM 输出；靠 `audio_start` 里的 `sample_rate` 告知前端 |

> 服务端如果按 16k 解释 44.1k 音频（或反之），表现是"声音变调/变速"，而不是报错。

---

## 6. 配置速查（`config.py`）

`config.py` 是新人**第一个该读**的文件。逐项要点：

| 常量 | 默认值 | 怎么理解 |
|---|---|---|
| `ASR_MODEL_DIR` | `models/SenseVoiceSmall` | **注意**：这是"默认 ASR"的路径。生产实际用 FireRedASR2，走 `USE_FIRERED_ASR=1` |
| `TTS_MODEL_DIR` | `models/VoxCPM1.5` | VoxCPM 权重 |
| `VLLM_BASE_URL` | `http://localhost:8100/v1` | LLM 服务地址（见第 9 节：启动脚本端口可能对不上） |
| `VLLM_MODEL_NAME` | `Qwen3-14B-AWQ` | 必须与 vLLM 的 `--served-model-name` **完全一致**，否则报 model not found |
| `SAMPLE_RATE` | `16000` | 上行采样率。**不是** TTS 的 44100 |
| `CHUNK_MS` / `CHUNK_SAMPLES` | `32` / `512` | 帧长，与前端 AudioWorklet 严格对齐 |
| `LLM_MAX_HISTORY` | `30` | 保留多少条历史（超出后滑动窗口 + 粗摘要） |
| `RAG_SCORE_THRESHOLD` | `0.35` | **低于此相似度不注入知识库**，让 LLM 走"这个我问下同事"兜底，防幻觉 |
| `RAG_TOP_K` | `3` | 检索条数 |
| `VOICE_PROMPT_WAV` / `_TEXT` | 参考音频 + 对应转写 | 语音克隆用；**两者必须对应**，否则音色会飘 |
| `GREETING_TEXT` | 开场白 | 连接后主动播报 |
| `IDLE_TIMEOUT_S` | `15` | 多久没说话就追问"您还在吗" |
| `IDLE_GOODBYE_S` | `30` | 多久没说话就告别挂断 |
| `MAX_CONSECUTIVE_EMPTY_ASR` | `3` | 连续几次听不清后升级提示 |
| `FAREWELL_KEYWORDS` | 再见/拜拜/挂了… | **子串匹配**，注意误判（如"再见"出现在"再见一面"里） |
| `SYSTEM_PROMPT` | 客服 lucy 人设 | 无 RAG 时用 |
| `SYSTEM_PROMPT_RAG` | 同上 + `{context}` | 有 RAG 时用，`{context}` 会被替换成检索结果 |

---


# 第二部分 · 部署（Debian 13 + 双卡 RTX 3090 Ti）

## 1. 一键启动（最快路径）

> **先看这一节。** 第 4 节起的"手动部署"是拆开讲原理的，第一次上手用脚本更快。

### 1.1 最短三步

```bash
cd <仓库根目录>
conda create -n voiceagent python=3.11
conda activate voiceagent          # 或 export VOICEAGENT_PYTHON=<python 绝对路径>

# ① 先跑一次预检：它会告诉你缺什么包、缺什么权重（不会启动任何服务）
bash start_all.sh                  # 预检不通过会直接停下并打印安装命令

# ② 预检通过后，同一条命令会拉起 vLLM + Voice Agent 并等待就绪
#    ③ 浏览器打开 http://<本机IP>:3000/ 开始对话；Ctrl+C 停止
```

### 1.2 五个脚本各管什么

| 脚本 | 用途 | 典型场景 |
|---|---|---|
| `voiceagent_env.sh` | **不是启动脚本**，是被下面 4 个 source 的共享配置层：模型路径、GPU 分配、端口、预检、启动函数都在这里 | 改配置只改这一个文件 |
| `start_all.sh` | 一键起 vLLM + Voice Agent，**前台托管**（Ctrl+C 全停） | 开发调试、现场演示 |
| `start_prod.sh` | `start`/`stop`/`status`/`restart` 四合一，服务**脱离终端常驻**并自动拉起守护 | 长期运行、无人值守 |
| `keep_alive.sh` | 纯守护循环：巡检 vLLM 与 Voice Agent，掉线自动拉起；`--once` 可单轮巡检 | 配合 cron / 单独跑 |
| `start_vllm.sh` | 只起 LLM 服务（前台，Ctrl+C 停；`--bg` 转后台） | 单独调试 LLM、排查模型加载 |

### 1.3 常用命令

```bash
# 开发：前台启动，能直接看到日志，Ctrl+C 停
bash start_all.sh

# 生产：后台常驻 + 守护
bash start_prod.sh            # 启动
bash start_prod.sh status     # 看端口/PID/显存
bash start_prod.sh stop       # 停止（按进程组，连 TTS 子进程一起清）
bash start_prod.sh clean-gpu  # 只列出占着显存的进程（排查 OOM 用）
bash start_prod.sh restart

# 只调试 LLM（前台，日志打屏）
bash start_vllm.sh

# PTT 演示模式（按住说话，跳过 VAD/判停，用于隔离问题）
DEMO_MODE=1 bash start_all.sh

# 守护：只巡检一轮（适合挂 cron）
bash keep_alive.sh --once

# 切换 LLM 权重（两块权重仓库里都有）
LLM_MODEL_DIR=models/MiniCPM4.1-8B-GPTQ LLM_SERVED_NAME=MiniCPM4.1-8B-GPTQ bash start_all.sh
```

### 1.4 设计上的三个关键点

1. **模型名与端口不再硬编码**——共享层直接从 `config.py` 读取 `VLLM_MODEL_NAME`
   与 `VLLM_BASE_URL`，所以脚本起的服务和 `ws_server` 的期望**不可能**再对不上
   （这正是旧脚本"起了 vLLM 但 ws_server 报 model not found"的根因）。
2. **端口探测不依赖 `lsof`**——改用 Debian 自带的 `ss`，缺工具不会导致误判"服务已死"而反复重启。
3. **就绪判断用轮询而非固定 sleep**——旧脚本写死 `sleep 70/110`，慢机器上会把同一服务重复拉起；
   现在改成轮询 `/v1/models` 与 `/api/info`。

## 2. 脚本配置项速查

所有可调项都在 **`voiceagent_env.sh`** 里，用环境变量覆盖（不必改文件）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `VOICEAGENT_PYTHON` | 取 PATH 里的 `python` | 解释器绝对路径。建议 `conda activate voiceagent` 或显式指定 |
| `GPU_LLM` | `0` | vLLM 用的物理卡号 |
| `GPU_VOICE` | `1` | ASR + TTS 用的物理卡号 |
| `LLM_MODEL_DIR` | `config.LLM_MODEL_DIR` | LLM 权重目录（**默认自动与 config.py 一致**） |
| `LLM_SERVED_NAME` | `config.VLLM_MODEL_NAME` | 对外模型名（自动对齐，改模型时要一起改） |
| `LLM_PORT` | `config.VLLM_BASE_URL` 的端口 | vLLM 端口（默认 8100） |
| `LLM_GPU_UTIL` | `0.85` | vLLM 显存占比；独占一张 24G 卡时合适 |
| `LLM_QUANT` | 按目录名自动推断 | `awq` / `gptq_marlin`；`*AWQ*` 目录自动用 awq |
| `VOICE_PORT` | `3000` | ws_server 端口 |
| `TTS_GPU_UTIL` | `0.45` | TTS 显存占比（与 ASR 同卡，24G 上约 10.8G） |
| `RAG_DEVICE` | `cpu` | RAG 放 CPU，把显存留给 ASR/TTS |
| `DEMO_MODE` | `0` | `1` = PTT 按住说话模式 |
| `CHECK_INTERVAL` | `30` | 守护巡检间隔（秒） |
| `ENABLE_TUNNEL` / `VPS_IP` | 关 / 空 | 公网反向隧道，**没有 VPS 就保持关闭** |
| `CLEAN_ORPHANS` | `0` | 守护重启前强杀 GPU 孤儿进程。**多人共用显卡时勿开** |
| `VA_SKIP_PREFLIGHT` | `0` | `1` = 跳过预检（应急） |
| `VA_LOG_DIR` | `<仓库>/logs` | 日志目录（已加进 `.gitignore`）；想放别处就覆盖它 |

## 3. 当前环境实况（部署前必读）

> 这一节是**实测**结果，不是模板。别人的机器上可能不同，请以自己 `bash start_all.sh` 的预检输出为准。

### 3.1 Python 环境

```
conda 环境   : voiceagent
Python       : 3.11.17
torch        : 2.13.0+cu130     （CUDA 13.0 轮子）
vLLM         : 0.31.0
```

**依赖清单已落盘**：仓库根目录的 [`requirements.txt`](requirements.txt) 用 `==` 锁定了
**37 个包**的精确版本（与上述环境逐版本一致，已用脚本比对验证）。安装：

```bash
conda create -y -n voiceagent python=3.11 && conda activate voiceagent

# ① 先装 torch 三件套 + CUDA 索引 —— 必须第一步
#    （vllm 硬依赖 torchvision==0.28.0，所以一起装，保证三者都是 +cu130 构建）
pip install torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ② 再装其余（torch 已满足，不会被重装）
pip install -r requirements.txt

# ③ 装 TTS 引擎（两步；flash_attn 要编译，见 3.2 节）
export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH
pip install flash-attn==2.8.3.post1 --no-build-isolation
pip install nano-vllm-voxcpm==2.0.4

# ④ 自检
bash start_all.sh
```

> **索引说明**：已核验 pytorch 的 cu130 索引**完整托管** torch 2.13.0 的全部依赖
> （含 `cuda-toolkit`、`cuda-bindings`、`nvidia-cudnn-cu13` 等 CUDA 包），因此
> `--index-url` 不会出现"依赖找不到"。若该索引在你网络下不可达，把 ① 的
> `--index-url` 换成 `--extra-index-url`（保留 PyPI 作主索引）即可。
>
> **想要完全可复现环境**：第一次装好并验证能跑通后，执行
> `pip freeze > requirements.lock.txt`——它会记下全部 300+ 个包（含 vllm 的
> `torchcodec`、`protobuf` 等本清单未锁定的传递依赖），以后用
> `pip install -r requirements.lock.txt` 重建即逐版本一致。

### 3.2 依赖缺口（**只差 TTS 引擎**）

实测结果：**主链路只缺 `nanovllm_voxcpm` 一个包**，其余全部就位。

| 缺失包 | 影响 | 处理 |
|---|---|---|
| **`nanovllm_voxcpm`** | `engine/tts.py` 的 TTS `load()` 用它——**没有它就没有声音**；且 `ws_server.py:204` 是硬调用（无 try/except、无跳过开关），服务会直接起不来 | PyPI 包名是 **`nano-vllm-voxcpm`**，见下 |

> ⚠️ **别搞混三个东西**：
> - `nanovllm_voxcpm`（import 名）/ `nano-vllm-voxcpm`（PyPI 名）→ **推理引擎**，来自 [a710128/nanovllm-voxcpm](https://github.com/a710128/nanovllm-voxcpm)，这才是 `engine/tts.py` 要的
> - `OpenBMB/VoxCPM` → 模型/训练代码（包名 `voxcpm`），**不是**本项目要的
> - `models/VoxCPM1.5/` → **模型权重**，仓库里已有

**安装分两步**（因为它的依赖 `flash_attn` 必须本地编译）：

```bash
# ① 编译安装 flash-attn（--no-build-isolation 是关键，否则看不到已装的 torch）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH          # 让 nvcc 进 PATH
pip install flash-attn==2.8.3.post1 --no-build-isolation

# ② 装 TTS 引擎
pip install nano-vllm-voxcpm==2.0.4
python -c "import nanovllm_voxcpm; print('TTS OK')"
```

> 🔴 **这一步有已知风险，请留足时间（约 30~60 分钟）**：
> flash-attn 官方预编译轮子**最高只覆盖到 torch 2.10**——我全量核验了它的
> **4528 个 Linux wheel**，其中**不存在**任何 `cu13 + torch2.13 + cp311` 组合
> （cu13 线只有 cp312/cp313 的 torch 2.9/2.10）。而本项目 `vllm==0.31.0`
> 又硬性要求 `torch==2.13.0`，无法降级绕开。**所以只能源码编译**。
> 若报编译器版本错误（本机 gcc 14 偏新），可试：
> `export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"`
> 编译吃内存，可用 `MAX_JOBS=8` 控制并行度。

> `nest_asyncio`（`ws_server.py:18` 的模块级导入）与 `sentence_transformers`
> （RAG 加载 bge 用）**都已安装**，无需再装。它们是对照下面这份实测结果确认的。

**已就位（实测 import 通过，共 25 个）**：
`torch`、`torchaudio`、`numpy`、`vllm`、`funasr`、`modelscope`、`sentence_transformers`、
`faiss`、`FlagEmbedding`、`onnxruntime`、`speechbrain`、`fastapi`、`uvicorn`、`websockets`、
`nest_asyncio`、`starlette`、`pydantic`、`openai`、`httpx`、`soundfile`、`scipy`、`librosa`、
`cn2an`、`sentencepiece`、`kaldiio`

**可选组件**（缺失只少功能，不影响主链路；装了才能开对应开关）：

| 缺失包 | 影响的功能 | 安装 |
|---|---|---|
| `pipecat` | Smart Turn 语义判停（`USE_SMART_TURN=1`） | `pip install pipecat-ai` |
| `moonshine_onnx` | 投机式 ASR（`USE_MOONSHINE_ASR=1`） | `pip install moonshine-onnx` |
| `fireredvad` | 替代 VAD（`USE_FIRERED_VAD=1`） | 按上游说明安装 |
| `kaldi_native_fbank` / `textgrid` | FireRedASR2 离线批量工具 | 见第 16 节（在线链路用不到） |

### 3.3 权重实况

| 目录 | 大小 | 状态 | 谁在用 |
|---|---|---|---|
| `models/Qwen3-14B-AWQ` | 9.4G | ✅ | **LLM 默认**（与 `config.py` 一致） |
| `models/MiniCPM4.1-8B-GPTQ` | 4.9G | ✅ | LLM 备选（`LLM_MODEL_DIR` 切换） |
| `models/SenseVoiceSmall` | 897M | ✅ | ASR（默认） |
| `models/VoxCPM1.5` | 1.9G | ✅ | TTS |
| `models/bge-small-zh-v1.5` | 184M | ✅ | RAG 向量化 |
| `models/bge-m3` / `bge-reranker-v2-m3` | 4.3G / 2.2G | ✅ | RAG 可选增强 |
| `models/snakers4_silero-vad` | 36M | ✅ | VAD（**布局已就位**，见 3.4） |
| `models/FireRedASR2-AED` | — | ❌ 缺失 | 生产 ASR（且代码也不完整，见第 4 节坑 1） |
| `models/dtln` | — | ❌ 缺失 | DTLN 降噪（`USE_DENOISE` 自动跳过） |
| `models/spkrec-ecapa-voxceleb` | — | ❌ 缺失 | 声纹门控（`USE_SPEAKER_VAD` 自动跳过） |

### 3.4 VAD 权重布局（**已就位，实测可加载**）

`engine/vad.py` 直接读 `config.VAD_MODEL_DIR` 下的固定文件名：

```
models/snakers4_silero-vad/silero_vad.jit      ← engine/vad.py 只认这个路径
```

**当前状态**：该文件已就位。已按 `engine/vad.py` 的实际加载方式实测通过：

```python
torch.jit.load("models/snakers4_silero-vad/silero_vad.jit", map_location="cpu")
# ✅ 加载成功；前向推理成功
```

目录内容（silero 官方发布的全部推理格式，另外几个是备选，主链路只用 `.jit`）：

| 文件 | 用途 |
|---|---|
| **`silero_vad.jit`** | **主链路使用**（TorchScript，`engine/vad.py` 读它） |
| `silero_vad.onnx` / `_half` / `_16k_op15` / `_16k_sequence` / `_op18_ifless` / `_openvino_16k` | ONNX 各版本，当前代码未用（备选） |
| `silero_vad_16k.safetensors` | safetensors 格式，当前代码未用 |

> ⚠️ **重新下载权重时的坑**（供以后换机器参考）：
> 从 GitHub 直接 clone `snakers4/silero-vad` 得到的是**源码仓库**，
> `.jit` 埋在 `src/silero_vad/data/` 下、目录名还带 `-master` 后缀。
> 必须**拷平**到 `models/snakers4_silero-vad/silero_vad.jit`，否则
> `engine/vad.py` 会退化成**联网**走 `torch.hub` 下载——离线环境直接失败，
> 现场表现是"卡在 `Loading VAD` 很久"（见第 15 节排障表）。
>
> 启动脚本 `va_check_vad_path()` 会在启动前检查这个路径：缺失时**打印出期望路径
> 和拷贝命令**，但只告警不中断（因为联网时 `torch.hub` 仍能兜底）。


> 目标：在本机跑通 `VAD → ASR → RAG → LLM → TTS` 全链路，浏览器能对话。
> 预计：环境 1~2 小时，模型下载视网速 1~4 小时。

## 4. 开始之前：先看这四个坑

坑 1、3、4 是我**逐文件核实代码后实测确认**的；坑 2 是历史问题，现已由启动脚本修掉。

### 坑 1：仓库里的 FireRedASR2 代码**不完整**，直接启用会崩

`engine/asr_firered.py` 会做 `from fireredasr2.asr import FireRedAsr2`，解析到仓库内的
`engine/fireredasr2/asr.py`。而该文件**无条件**导入了：

```python
from .models.fireredasr_aed import FireRedAsrAed
from .models.fireredasr_llm import FireRedAsrLlm
from .models.lstm_lm     import LstmLm
from .models.param       import count_model_parameters
```

**但仓库里根本没有 `engine/fireredasr2/models/` 这个目录。**

**后果**：只要设了 `USE_FIRERED_ASR=1`，ASR 加载就会 `ModuleNotFoundError: No module named 'fireredasr2.models'`。

**处理方式**：**先用默认的 SenseVoice 跑通全链路**（一键脚本默认就是这样），
把 FireRedASR2 作为可选升级放在[第 16 节](#16-可选把-asr-升级为-fireredasr2)，那里给了补齐 `models/` 的方法。

> 这不是你的环境问题，是仓库本身的缺口。README 宣称的 CER 2.89% 是 FireRedASR2 的成绩。

### 坑 2：模型名/端口曾经对不上（**已由启动脚本修复**）

历史问题：`config.py` 写 `Qwen3-14B-AWQ` / `:8100`，而脚本实际加载 `MiniCPM4.1-8B-GPTQ`，
且 `start_vllm.sh` 用的是 **8000** 端口。这就是"vLLM 明明起来了，ws_server 却报
`model not found` / 连不上"的根因。

**现在已修复**：`voiceagent_env.sh` 通过 `va_cfg` 直接从 `config.py` 读取
`VLLM_MODEL_NAME` 与 `VLLM_BASE_URL`，脚本与代码**不可能**再漂移，你无需手动对齐。

> 但如果你**手工**起服务（不走脚本），这条铁律仍然适用：
> `config.VLLM_MODEL_NAME` 必须等于 vLLM 的 `--served-model-name`，
> `config.VLLM_BASE_URL` 的端口必须等于 vLLM 的 `--port`。

### 坑 3：`config.py` 里的 `*_GPU` 是**死变量**，设了完全没用

`config.py` 里有这几个看起来很像"指定显卡"的配置：

```python
LLM_GPU = int(os.environ.get("LLM_GPU", "1"))
ASR_GPU = int(os.environ.get("ASR_GPU", "2"))
TTS_GPU = int(os.environ.get("TTS_GPU", "2"))
RAG_GPU = int(os.environ.get("RAG_GPU", "2"))
```

我全仓搜索过：**它们只出现在注释和 docstring 里，没有任何一行代码读取它们**。
真正生效的是**环境变量** `ASR_DEVICE` / `TTS_DEVICE` / `RAG_DEVICE`（以及 vLLM 自己的 `CUDA_VISIBLE_DEVICES`）。

**所以**：别去改 `config.py` 的 `ASR_GPU`，要设置 `ASR_DEVICE=cuda:N`。教程里给你的命令已经是正确的。

### 坑 4：显存占用者其实只有三个，别给"不占显存"的组件留卡

我核实了各引擎的设备使用：

| 组件 | 跑在哪 | 依据 |
|---|---|---|
| **vLLM（LLM）** | GPU | 独立进程 |
| **ASR** | GPU（可设 CPU） | `ASR_DEVICE` |
| **TTS（VoxCPM）** | GPU | `TTS_DEVICE` + `TTS_GPU_UTIL` |
| RAG（bge） | GPU 或 CPU | `RAG_DEVICE`（**建议 CPU**，你有 128GB 内存） |
| Silero VAD | **CPU**（固定） | `engine/vad.py` 用 `torch.jit.load(..., map_location="cpu")` |
| 声纹 VAD（ECAPA） | **CPU**（默认） | `engine/speaker_vad.py` 的 `load(device="cpu")` |
| DTLN 降噪 | **CPU**（强制） | `engine/denoiser.py` 显式 `providers=["CPUExecutionProvider"]` |
| 投机 ASR（Moonshine） | **CPU**（固定） | ONNX Runtime CPU |

**结论**：两块卡只需要装 **LLM** 和 **ASR+TTS** 两件事，分配非常干净，而且
CPU 侧那堆组件还能顺便吃掉一部分负载，帮 GPU 减压。

---

## 5. 双卡分配方案（推荐直接照抄）

这块项目**生产脚本用的就是"LLM 独占一张、语音栈共用另一张"**：

```
┌─────────────────────────────┐   ┌─────────────────────────────┐
│  GPU 0  (24GB)              │   │  GPU 1  (24GB)              │
│  ─────────────────────────  │   │  ─────────────────────────  │
│  vLLM  (Qwen3-14B-AWQ)      │   │  ASR  (SenseVoice/FireRed)  │
│  --gpu-memory-utilization   │   │  TTS  (VoxCPM)              │
│          0.85  ≈ 20.4GB     │   │  TTS_GPU_UTIL=0.45 ≈10.8GB  │
│                             │   │  ASR ≈ 2~3GB                │
│  余量 ≈ 3.6GB               │   │  余量 ≈ 10GB                │
└─────────────────────────────┘   └─────────────────────────────┘
        ↑                                      ↑
   CUDA_VISIBLE_DEVICES=0            CUDA_VISIBLE_DEVICES=1
   （vLLM 进程独占）                  （ws_server 进程独占）

跑在 CPU 上、不占显存：Silero VAD / 声纹 ECAPA / DTLN 降噪 / 投机 ASR / RAG（建议设 cpu）
```

### 为什么这样分

- **vLLM 要 0.85**：它需要大 KV cache 才能扛住并发。独占一张卡时可以放心给高
  （README 的官方 Quick Start 用的就是 0.85）。
- **语音栈卡要留足余量**：TTS 的 `gpu_memory_utilization` 是"**按整卡显存比例**预分配"的，
  0.45 在 24GB 卡上就是 10.8GB；再加 ASR 的 2~3GB，总共约 14GB，还剩 10GB 安全余量。
- **不要把 vLLM 和 TTS 塞进同一张卡**：那是单卡玩家被迫做的取舍（得把 `TTS_GPU_UTIL`
  压到 0.18），你有两块卡，没必要受这个罪。

### 关于"两块卡能不能一起给 vLLM"

可以（`--tensor-parallel-size 2`），但**本教程不推荐**：

- 14B AWQ 单卡 24GB 完全放得下，TP=2 收益有限；
- 语音栈就没卡可用了，只能去和 vLLM 抢显存，又把单卡的老问题请回来；
- TP 会引入额外的进程间通信开销，对语音这种短请求不一定划算。

除非你主要用途是**离线批量评测**（`test_*.py` 那批），那时再单独起一个 TP=2 的实例。

---

## 6. 系统准备（NVIDIA 驱动 + CUDA）

### 6.1 确认两块卡都在

```bash
lspci | grep -i nvidia          # 应看到两条 VGA/3D controller
nvidia-smi -L                   # 应列出 GPU 0 与 GPU 1
cat /etc/debian_version         # 应显示 13.x
free -g                         # 确认内存 ~128GB
```

**建议：固定设备序号**。CUDA 默认按"最快优先"排序，重启或多卡插槽变动时序号可能漂移，
让你 `CUDA_VISIBLE_DEVICES=1` 指到另一张卡上。统一加上环境变量可以让序号按 PCI 总线固定：

```bash
# 写进 ~/.bashrc，对交互式会话生效（systemd 服务里要单独设）
echo 'export CUDA_DEVICE_ORDER=PCI_BUS_ID' >> ~/.bashrc
source ~/.bashrc
nvidia-smi -L                   # 再确认一次顺序
```

顺便看一眼两块卡的互联与插槽（本方案不依赖 P2P，但插槽带宽异常会拖慢 ASR/TTS）：

```bash
nvidia-smi topo -m
# 关注 CPU Affinity 与 NV# 列；两张卡分别在不同 NUMA 节点时，
# 可以用 numactl 把进程绑到对应节点（可选优化，非必需）
```

### 6.2 安装 NVIDIA 驱动

Debian 13 需要启用 `non-free` 与 `non-free-firmware` 源。推荐**用 Debian 官方仓库**装驱动（比 NVIDIA 官方 runfile 更少踩坑）：

```bash
# 确认 contrib/non-free 已启用
grep -rE "non-free" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null

# 若未启用，追加（Debian 13 用 deb822 格式，文件通常在 /etc/apt/sources.list.d/debian.sources）
sudo sed -i 's/^Components: main.*/Components: main contrib non-free non-free-firmware/' \
  /etc/apt/sources.list.d/debian.sources 2>/dev/null || true

sudo apt update
sudo apt install -y linux-headers-$(uname -r) build-essential dkms
# 安装驱动（会自动选适配当前内核的版本；同一版本驱动同时管两块卡）
sudo apt install -y nvidia-driver firmware-misc-nonfree
sudo reboot
```

重启后验证**两块卡都正常**：

```bash
nvidia-smi
# 期望看到两行 GPU：RTX 3090 Ti，各约 24564MiB，
# Driver Version 5xx+，CUDA Version 12.x

# 开启持久模式，避免驱动模块被卸载导致首个 CUDA 任务卡顿（两块卡都要开）
sudo nvidia-smi -pm 1
nvidia-smi --query-gpu=index,name,persistence_mode,memory.total --format=csv
```

> ⚠️ 若只认到一块卡：检查 BIOS 里两个 PCIe 插槽是否都启用、是否都插到底；
> `lspci | grep -i nvidia` 看不到硬件就是物理/BIOS 问题，不是驱动问题。

### 6.3 关于 CUDA Toolkit

**大多数情况下你不需要单独装 CUDA Toolkit**——PyTorch/vLLM 的 pip wheel 自带 CUDA 运行时。
只有当你需要自己编译算子时才装。**先跳过，装完 PyTorch 用 `torch.cuda.device_count()` 验证即可。**

编译工具链建议装上（某些包需要现场编译）：

```bash
sudo apt install -y git curl wget ffmpeg sox libsndfile1 libsndfile1-dev \
                    python3-dev cmake ninja-build pkg-config
```

---

## 7. Python 环境

### 7.1 为什么不用系统 Python

Debian 13 自带 **Python 3.13**。深度学习生态对它支持滞后（尤其 vLLM、funasr 等），
**强烈建议用 Miniconda 建 Python 3.11 环境**。

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 建环境（Python 3.11 是当前最稳的选择）
conda create -y -n voiceagent python=3.11
conda activate voiceagent

python -V          # 应为 Python 3.11.x
```

> 📌 本项目的启动脚本用的是**绝对路径 python**（如
> `/cache/zhangjing/miniconda3/envs/voiceagent/bin/python`），不是 `conda activate`。
> 你要么改脚本里的路径，要么就用 `conda activate` + 手动命令（本教程走后者）。

### 7.2 设置 pip 国内镜像（可选，但强烈建议）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 8. 安装依赖

> **首选方式**：直接用仓库根目录的 [`requirements.txt`](requirements.txt)——它用 `==` 锁定了
> 37 个包的精确版本，与第 3 节实测的环境逐版本一致。
>
> ```bash
> conda create -y -n voiceagent python=3.11 && conda activate voiceagent
> # ① 先装 torch 三件套 + CUDA 索引（必须第一步；vllm 硬依赖 torchvision==0.28.0）
> pip install torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
>     --index-url https://download.pytorch.org/whl/cu130
> # ② 再装其余
> pip install -r requirements.txt
> # ③ TTS 引擎不在 PyPI，需从源码装（见 3.2 节）
> ```
>
> 下面按组件列出**每个包为什么需要**，便于你替换或裁剪。

### 8.1 核心：PyTorch

```bash
# 必须用带 CUDA 的轮子索引，且必须最先装：
# vllm 会重新解析 torch 依赖，不先装好会被它换成另一个构建版本。
# torchvision 也必须在这一步一起装 —— vllm==0.31.0 硬依赖 torchvision==0.28.0。
pip install torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 验证：应看到 2（两块卡都可见）
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
# 期望输出：2.13.0+cu130 True 2
#          ['NVIDIA GeForce RTX 3090 Ti', 'NVIDIA GeForce RTX 3090 Ti']
```

> **关于 `--index-url`**：它会让 pip **只用** pytorch 索引。已逐包核验该索引完整托管
> torch 2.13.0 的全部依赖（`filelock`/`sympy`/`networkx`/`jinja2`/`fsspec`/`setuptools`/
> `triton`，以及 `cuda-toolkit`、`cuda-bindings`、`nvidia-cudnn-cu13`、`nvidia-nccl-cu13`
> 等 CUDA 包），所以不会缺依赖。若该索引在你网络下不可达，改成
> `--extra-index-url https://download.pytorch.org/whl/cu130`（保留 PyPI 作主索引）。
>
> ⚠️ 如果 `device_count()` 是 1：回到第 6 节查物理连接与 BIOS 设置。
> 如果 `cuda.is_available()` 是 `False`：装成 CPU 版了或驱动不匹配。

### 8.2 LLM 推理服务：vLLM

```bash
pip install vllm
python -c "import vllm; print(vllm.__version__)"
```

> ⚠️ vLLM 会**重新解析 torch 依赖**，可能升级/降级你的 torch。装完 vLLM 后**务必重新验证**
> `torch.cuda.device_count()` 仍是 2。如果被改坏，按 4.1 重装 torch 后再确认 vLLM 仍能 import。

### 8.3 语音链路（ASR / VAD / TTS / 降噪）

```bash
# --- ASR：默认实现是 SenseVoice（FunASR）---
pip install funasr modelscope

# --- VAD / 降噪 / 投机 ASR：都靠 ONNX Runtime（且都跑 CPU）---
pip install onnxruntime

# --- RAG：向量化 + 检索（本教程建议跑 CPU）---
pip install FlagEmbedding faiss-cpu sentence-transformers

# --- 声纹 VAD（可选，USE_SPEAKER_VAD=1 才需要；跑 CPU）---
pip install speechbrain torchaudio

# --- 音频 IO ---
pip install soundfile numpy scipy librosa

# --- 文本归一化（评测脚本 wer.py 用）---
pip install cn2an
```

### 8.4 Web 服务

```bash
pip install fastapi "uvicorn[standard]" websockets nest_asyncio pydantic httpx openai
```

### 8.5 TTS：Nano-vLLM-VoxCPM（需要单独装，且要编译）

代码里 `engine/tts.py` 依赖 **`nanovllm_voxcpm`**，它对应 PyPI 包 **`nano-vllm-voxcpm`**，
来自 [a710128/nanovllm-voxcpm](https://github.com/a710128/nanovllm-voxcpm)。

> ⚠️ **最容易踩的坑**：不要 clone `OpenBMB/VoxCPM` —— 那是模型与训练代码（包名 `voxcpm`），
> 装出来**不会**提供 `nanovllm_voxcpm`。这两个仓库名字像，但用途完全不同。

它的硬依赖 `flash_attn` 在 PyPI 上**只有源码包**，所以分两步：

```bash
# ① 编译 flash-attn（必须加 --no-build-isolation，否则编译环境看不到已装的 torch）
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
# 本机 nvcc 为 13.3，支持 sm_86（3090 Ti）；gcc 14 偏新，必要时加下面这行
# export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
pip install flash-attn==2.8.3.post1 --no-build-isolation

# ② 装 TTS 引擎
pip install nano-vllm-voxcpm==2.0.4
python -c "import nanovllm_voxcpm; print('nanovllm_voxcpm OK')"
```

> 🔴 **已知风险（重要）**：flash-attn 的官方预编译轮子**最高只到 torch 2.10**。
> 我全量核验了它的 4528 个 Linux wheel，**没有** `cu13 + torch2.13 + cp311` 组合。
> 而 `vllm==0.31.0` 硬性要求 `torch==2.13.0`，无法降级绕开 —— 因此只能源码编译。
> 编译耗时约 30~60 分钟，可用 `MAX_JOBS=8` 控制并行度（吃内存）。
> 装不上 TTS 的话，链路会卡在"能识别但不出声"，见[第 15 节](#15-常见错误排查)。

### 8.6 依赖自检（**强烈建议跑一下**）

```bash
cat > /tmp/check_deps.py <<'EOF'
mods = [
    ("torch", "核心"), ("torchaudio", "核心"),
    ("vllm", "LLM服务"), ("funasr", "ASR(SenseVoice)"),
    ("onnxruntime", "降噪/投机ASR"), ("FlagEmbedding", "RAG"), ("faiss", "RAG"),
    ("sentence_transformers", "RAG"), ("soundfile", "音频IO"), ("numpy", "核心"),
    ("scipy", "音频"), ("cn2an", "评测"), ("fastapi", "Web"), ("uvicorn", "Web"),
    ("websockets", "Web"), ("nest_asyncio", "Web"), ("httpx", "LLM客户端"),
    ("openai", "LLM客户端"),
    # 可选组件
    ("nanovllm_voxcpm", "TTS(必须)"), ("speechbrain", "声纹VAD(可选)"),
    ("pipecat", "SmartTurn(可选)"), ("moonshine_onnx", "投机ASR(可选)"),
    ("fireredvad", "FireRedVAD(可选)"),
]
ok = miss = 0
for name, why in mods:
    try:
        __import__(name); print(f"  ✅ {name:<24} {why}"); ok += 1
    except Exception as e:
        print(f"  ❌ {name:<24} {why}  -> {type(e).__name__}"); miss += 1
print(f"\n可用 {ok} / 缺失 {miss}")
EOF
python /tmp/check_deps.py
```

**判读方法**：
- 标"核心/必须"的缺失 → 必须解决，否则跑不起来；
- 标"可选"的缺失 → 只要**不开对应的环境变量**（`USE_SPEAKER_VAD` / `USE_SMART_TURN` /
  `USE_MOONSHINE_ASR` / `USE_FIRERED_VAD`），不影响主链路。

---

## 9. 下载模型

### 9.1 目录约定（**必须严格按这个结构放**）

所有模型都放在仓库根目录的 `models/` 下。仓库里的 `models/` 目录**只含一个下载脚本**
（`models/download_model.py`，用法见 9.2），**权重本身不入库**，需要你自行下载。
若目录不存在（比如你只取了部分文件），先建一下：

```bash
cd /path/to/600ms_voice_agent
mkdir -p models
```

| 目录 | 用途 | 是否必需 | 关键文件（用于自检） |
|---|---|---|---|
| `models/SenseVoiceSmall` | ASR（默认） | ✅ **必需** | FunASR 权重目录 |
| `models/VoxCPM1.5` | TTS | ✅ **必需** | VoxCPM 权重 |
| `models/Qwen3-14B-AWQ` | LLM | ✅ **必需** | AWQ 量化权重 + tokenizer |
| `models/snakers4_silero-vad` | VAD | ✅ 必需 | **`silero_vad.jit`** |
| `models/bge-small-zh-v1.5` | RAG 向量化 | ✅ 必需 | 模型权重 |
| `models/FireRedASR2-AED` | ASR（生产版） | ⭕ 可选（坑 1） | `dict.txt`、`train_bpe1000.model`、`cmvn.ark` |
| `models/dtln` | 降噪 | ⭕ 可选 | **`model_1.onnx`**、**`model_2.onnx`** |
| `models/spkrec-ecapa-voxceleb` | 声纹 VAD | ⭕ 可选 | ECAPA-TDNN 权重 |
| `models/FireRedVAD-stream` | FireRedVAD | ⭕ 可选 | — |
| `models/bge-m3` / `bge-reranker-v2-m3` | RAG 增强 | ⭕ 可选 | 体积大，先不装 |

> 这些路径来自 `config.py:14-29` 与 `ws_server.py:58-61`，是**代码里写死的**。
> 想放到别处就改 `config.py` / `ws_server.py`，不要指望脚本参数。

### 9.2 下载方式

用 `modelscope` 或 `huggingface-cli`（**国内建议 modelscope**，速度快很多）：

```bash
pip install modelscope huggingface_hub

# 示例：下载 SenseVoiceSmall（模型 ID 请以官方当期为准）
python -c "
from modelscope import snapshot_download
p = snapshot_download('iic/SenseVoiceSmall', local_dir='models/SenseVoiceSmall')
print('下载到:', p)
"
```

其它模型的获取渠道：

| 模型 | 建议来源 |
|---|---|
| Qwen3-14B-AWQ | ModelScope / HuggingFace 搜 `Qwen3-14B-AWQ` |
| VoxCPM1.5 | OpenBMB/VoxCPM 的官方发布页 |
| bge-small-zh-v1.5 | BAAI 官方仓库 |
| silero-vad | `snakers4/silero-vad` 仓库里的 `silero_vad.jit`（⚠️ 它埋在 `src/silero_vad/data/` 下，需拷到 `models/snakers4_silero-vad/`，见 3.4 节） |
| FireRedASR2-AED | HuggingFace `FireRedTeam/FireRedASR2-AED`（需配合第 12 步） |
| DTLN | `breizhn/DTLN` 官方仓库的 ONNX 模型 |

> 我无法在当前环境联网核实每个模型 ID 的当期名称，**请以各项目官方页面为准**。

**省事的做法**：仓库里的 `models/download_model.py` 会把 7 个模型一次性拉下来
（SenseVoiceSmall / MiniCPM4.1-8B-GPTQ / VoxCPM1.5 / Qwen3-14B-AWQ / bge-small-zh-v1.5 /
bge-m3 / bge-reranker-v2-m3）：

```bash
pip install modelscope
python models/download_model.py
```

> ⚠️ **用之前先改路径**：该脚本里 7 处 `local_dir` 都是**写死的本机绝对路径**
> （`/home/victor/code/600ms_voice_agent/models/...`），换机器或换目录会把权重下到错误位置。
> 另外它**不包含 `snakers4_silero-vad`**（那个要按上表单独取 `silero_vad.jit`），
> 末尾还有一段 silero-vad 的实验性代码（会联网 `torch.hub.load`），跑完下载可以直接忽略。

### 9.3 模型自检

```bash
cd /path/to/600ms_voice_agent
cat > /tmp/check_models.py <<'EOF'
import os
need = {
    "models/SenseVoiceSmall": "ASR(SenseVoice) 必需",
    "models/VoxCPM1.5": "TTS 必需",
    "models/Qwen3-14B-AWQ": "LLM 必需",
    "models/snakers4_silero-vad": "VAD 必需",
    "models/bge-small-zh-v1.5": "RAG 必需",
    "models/FireRedASR2-AED": "ASR(FireRedASR2) 可选",
    "models/dtln": "降噪 可选",
    "models/spkrec-ecapa-voxceleb": "声纹VAD 可选",
}
for p, why in need.items():
    ok = os.path.isdir(p)
    print(f"  {'✅' if ok else '❌'} {p:<38} {why}")
jit = "models/snakers4_silero-vad/silero_vad.jit"
print(f"\n  {'✅' if os.path.isfile(jit) else '❌'} {jit}   <- VAD 真正加载这个文件")
EOF
python /tmp/check_models.py
```

---

## 10. 准备数据与参考音频

### 10.1 知识库（RAG）

`data/sample_kb.json` **仓库里已经有**（实测：**59 条**，键为 `id` / `question` / `answer`）。
`ws_server` 启动时会读它并**自动建 FAISS 索引**（不需要你手动建；索引只在内存里，
所以**每次启动都要重新编码一遍**——59 条很快，文档上千条时启动会明显变慢）。

格式（实测的第一条）：

```json
[
  {"id": "1",
   "question": "你们公司叫什么名字？",
   "answer": "面壁智能（ModelBest）是一家专注于人工智能大模型技术创新与应用落地的科技公司。"}
]
```

`text` 字段可选：不提供时，代码用 `question + " " + answer` 作为被编码的文本。

> 关键参数：`config.RAG_SCORE_THRESHOLD = 0.35`——**相似度低于这个值就不注入知识库**，
> 让 LLM 走"这个我问下同事"的兜底话术，防止编造。调高更保守，调低更容易命中。

### 10.2 语音克隆参考音频（TTS）

代码默认找 `data/lipsync_real_clone_sample_5s.wav`，但**仓库里没有任何 .wav 文件**：

```bash
ls data/*.wav 2>/dev/null || echo "没有参考音频（不影响启动，会跳过音色克隆）"
```

**两种选择**：

**A. 暂不克隆音色**（推荐先这样）——`ws_server` 发现文件不存在会自动跳过 `register_voice`，
用 VoxCPM 的默认音色。链路完全正常。

**B. 用自己的音色**：准备一段 **5~10 秒、单人、干净、无背景音乐**的 wav，然后：

```bash
# 把音频放到 data/ 下，并修改 config.py：
#   VOICE_PROMPT_WAV  = "data/你的音频.wav"
#   VOICE_PROMPT_TEXT = "音频里逐字说的内容"
```

> ⚠️ **`VOICE_PROMPT_TEXT` 必须与音频内容逐字一致**（包括标点习惯）。两者不对应，
> 合成出来的音色会"飘"、不稳定。
>
> 也可以不改 `config.py`，用环境变量覆盖：`VOICE_PROMPT_WAV=... VOICE_PROMPT_TEXT=...`。

---

## 11. 在 **GPU 0** 启动 vLLM（LLM 服务）

**单独开一个终端**（它会常驻、独占 GPU 0）：

```bash
conda activate voiceagent
cd /path/to/600ms_voice_agent

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-14B-AWQ \
  --served-model-name Qwen3-14B-AWQ \
  --trust-remote-code \
  --dtype auto \
  --quantization awq \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --port 8100
```

> ⚠️ **不要加 `--enforce-eager`**：它会连带关掉 JIT kernel warmup，让 FlashInfer 的采样
> 内核拖到**第一个真实请求**才现场编译（实测 24 秒），直接撞穿 `engine/llm.py` 的
> `read=30s` 超时。踩坑记录见 [15.2 节](#152-第一次对话卡二十几秒然后-llm-stream-error-request-timed-out)。

**每个参数的用意（双卡视角）**：

| 参数 | 值 | 为什么 |
|---|---|---|
| `--served-model-name` | `Qwen3-14B-AWQ` | **必须等于 `config.VLLM_MODEL_NAME`**，否则报 model not found |
| `--port` | `8100` | **必须等于 `config.VLLM_BASE_URL` 的端口** |
| `--quantization` | `awq` | 4bit 权重省显存（14B → ~9.5GB） |
| `--gpu-memory-utilization` | **`0.85`** | **独占一张卡**，可以给高（≈20.4GB），KV cache 充足、并发稳；README 官方 Quick Start 也是 0.85 |
| `--max-model-len` | `4096` | 语音对话轮次短，4096 足够；这是 KV cache 的主要开关 |
| `--max-num-seqs` | `8` | 双卡有富余，可以比单卡方案放宽 |
| ~~`--enforce-eager`~~ | **不要加** | 它会连带关掉 JIT kernel warmup，导致 FlashInfer 采样内核拖到**第一个真实请求**才现场编译（实测 **24 秒**），撞穿 `engine/llm.py` 的 `read=30s` 超时 → 报 `LLM stream error: Request timed out.`。详见 [15.2 节](#152-第一次对话卡二十几秒然后-llm-stream-error-request-timed-out) |

**等到日志出现** `Application startup complete` 或 `Uvicorn running on http://0.0.0.0:8100` 才算好。
14B 模型首次加载通常 1~3 分钟。**不带 `--enforce-eager` 时启动会再久一些**——因为 JIT 内核
预热与 CUDA Graph 捕获被挪到了启动阶段（这正是我们想要的：把 20 多秒的一次性代价从
"第一次对话"搬到"启动"，见 15.2 节）。启动脚本的等待上限是 300 秒，够用。

**验证**（另开终端）：

```bash
curl -s http://localhost:8100/v1/models | python -m json.tool
# 期望看到 "id": "Qwen3-14B-AWQ"

curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-14B-AWQ","messages":[{"role":"user","content":"你好"}],"max_tokens":16}'
# 期望返回一个正常的 JSON 回复
```

顺手确认它没越界跑到另一张卡上：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
nvidia-smi    # GPU 0 应被 python 占住约 20GB，GPU 1 应基本空闲
```

> 💡 如果你的用途偏向**离线批量评测**（`test_*.py` 那批脚本），可以另起一个
> `--tensor-parallel-size 2` 的实例把两块卡都用上；但**在线语音服务不要这么干**（见第 1 节）。

---

## 12. 在 **GPU 1** 启动 Voice Agent（ws_server）

**再开一个终端**：

```bash
conda activate voiceagent
cd /path/to/600ms_voice_agent

CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
ASR_DEVICE=cuda:0 \
TTS_DEVICE=cuda:0 \
RAG_DEVICE=cpu \
TTS_GPU_UTIL=0.45 \
PORT=3000 \
python ws_server.py
```

**关键点（双卡适配）**：

| 环境变量 | 值 | 说明 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | **`1`** | 把**物理 1 号卡**暴露给本进程；此时它对进程内而言就是 `cuda:0` |
| `ASR_DEVICE` / `TTS_DEVICE` | **`cuda:0`** | ⚠️ 这里的 `0` 是**重映射后的卡内序号**，指向物理 1 号卡。**不是**让你写 `cuda:1` |
| `RAG_DEVICE` | `cpu` | 走 128GB 内存，**把语音卡显存全留给 ASR+TTS** |
| `TTS_GPU_UTIL` | `0.45` | 约 10.8GB，是**生产脚本同款值**；因为这张卡只有 ASR 共享，放得下 |
| `PORT` | `3000` | HTTP/WS 端口 |

> ⚠️ **`CUDA_VISIBLE_DEVICES=1` + `ASR_DEVICE=cuda:0` 这个组合最容易看错**：
> 前者是"物理卡号"，后者是"进程内可见卡号"。项目自己的 `keep_alive.sh` 就是这么写的
> （`CUDA_VISIBLE_DEVICES=2` + `ASR_DEVICE=cuda:0`）。
>
> 如果你更喜欢**不用 CUDA_VISIBLE_DEVICES** 的写法，那就直接写物理号：
> `ASR_DEVICE=cuda:1 TTS_DEVICE=cuda:1`。**两种写法选一种，千万别混**。

**注意：故意没有设 `USE_FIRERED_ASR=1`**——先用默认的 SenseVoice 跑通（坑 1）。

**成功日志长这样**（依次加载各引擎）：

```
=== Loading engines (v2.0) ===
Loading VAD (Silero)...
Loading ASR (SenseVoice)...
Loading RAG (bge-small-zh-v1.5, threshold=0.35)...
RAG index: 59 docs, dim=512
LLM client ready (vLLM @ http://localhost:8100/v1, max_history=30)
Loading TTS (VoxCPM nanovllm)...
Warming up TTS...
Pre-generating filler words...
=== All engines loaded (v2.9) ===
```

> ⚠️ 看到 `Loading ASR (SenseVoice)` 而不是 `FireRedASR2` 是**预期的**（坑 1）。
>
> ⚠️ 如果卡在 `Loading VAD` 很久：Silero 在找不到本地 `silero_vad.jit` 时会**联网**去
> torch.hub 下载。把文件放到 `models/snakers4_silero-vad/silero_vad.jit`（第 3.4 节）
> 就能离线快速加载。

**此时再看一眼两张卡的分工是否如预期**：

```bash
nvidia-smi
# GPU 0：vLLM 约 20GB
# GPU 1：ASR + TTS 约 13~14GB
```

---

## 13. 验证

### 13.1 HTTP 接口

```bash
curl -s http://localhost:3000/api/info | python -m json.tool
# 期望：模型名、采样率、设备等信息

curl -s http://localhost:3000/api/metrics | python -m json.tool
# 期望：延迟统计

curl -s "http://localhost:3000/api/rag/query?q=服务时间" | python -m json.tool
# 期望：检索到的知识库条目（验证 RAG 通了）
```

> 若设了 `VOICEAGENT_API_KEY`，上述请求需带 `?token=<key>`，否则 HTTP 返回 **403**；
> WebSocket 则会被 `close(code=4003)`。两者错误码不同。

### 13.2 浏览器对话

1. 打开 `http://<服务器IP>:3000/`（根路径直接返回前端页面 `static/voice_agent.html`）；
2. 允许麦克风权限；
3. 页面连上后会收到问候语（`GREETING_TEXT`）；
4. 对着麦克风说一句，例如"你们服务时间是什么"；
5. 应能看到：状态灯 `idle → listening → thinking → speaking`，以及 ASR 文本、RAG 命中、AI 回复与音频。

> 浏览器要求**用户手势**后才能播放音频（自动播放策略）。如果没声音，先点一下页面。

### 13.3 用 HTTP 接口验证

> 说明：仓库原有的 `test_*.py` 测试脚本已按精简要求移除，因此这里改用接口自查。
> 想做端到端回归测试，可以自己写一个 WebSocket 客户端脚本（协议见
> 第一部分第 5 节的消息表）。

```bash
# 1) 服务信息与健康检查
curl -s http://localhost:3000/api/info    | python -m json.tool
curl -s http://localhost:3000/api/metrics | python -m json.tool

# 2) 验证 RAG 检索通了（把 q 换成你知识库里的问题）
curl -s "http://localhost:3000/api/rag/query?q=你们公司叫什么名字" | python -m json.tool

# 3) 验证前端页面能取到
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000/
```

判读：
- `/api/info` 应返回模型名、采样率、设备；
- `/api/rag/query` 应返回 `data/sample_kb.json` 里的相关条目；
- 第 3 条返回 `HTTP 200` 说明静态页面服务正常。

> 若设了 `VOICEAGENT_API_KEY`，这些请求需要带 `?token=<key>`，否则返回 **403**。

### 13.4 观察延迟

```bash
# 一边对话一边看指标
watch -n1 'curl -s http://localhost:3000/api/metrics | python -m json.tool'
```

关注 `first_response_ms`（首响）。它的构成是 **ASR + RAG + LLM 首句 + TTS 首 chunk**，
**不含** VAD 判停那 256~640ms 的等待。

---

## 14. 显存与延迟调优

### 14.1 先看两张卡的实况

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
watch -n2 nvidia-smi      # 对话时观察哪张卡在动
```

**健康状态**：GPU 0 稳定在 ~20GB（vLLM 预分配），GPU 1 在 ~13~14GB；
GPU 利用率上，GPU 0 在 LLM 生成时冲高、GPU 1 在 ASR/TTS 阶段冲高。

### 14.2 如果 OOM（双卡方案下很少见）

按这个顺序排查/降级：

1. **确认是不是真的分卡了**：`nvidia-smi` 看是不是两个进程挤在同一张卡上
   （最常见原因：`ws_server` 忘了设 `CUDA_VISIBLE_DEVICES`，或 vLLM 没设）
2. **降 TTS**：`TTS_GPU_UTIL=0.45 → 0.35`（这张卡只有 ASR 共享，通常不用动）
3. **确认 RAG 在 CPU**：`RAG_DEVICE=cpu`（别漏）
4. **降 vLLM**：`--gpu-memory-utilization 0.85 → 0.80`，同时 `--max-model-len 4096 → 2048`
5. **关掉可选组件**：不设 `USE_SPEAKER_VAD` / `USE_DENOISE` / `USE_SMART_TURN` / `USE_MOONSHINE_ASR`
   （这些本来就跑 CPU，主要影响算力不救显存，但能减少排队）
6. **换小模型**：8B AWQ（记得同步改 `config.py` 的 `LLM_MODEL_DIR` 与 `VLLM_MODEL_NAME`）
7. **把 ASR 挪到 GPU 0**：`ASR_DEVICE=cuda:1`（在 ws_server 进程里，如果只暴露了物理 1 号卡就写 `cuda:0`），
   让 ASR 与 vLLM 共享 GPU 0——不推荐，只是应急

### 14.3 延迟调优

| 想优化 | 改哪里 | 代价 |
|---|---|---|
| LLM 首 token | `--max-model-len` 调小（KV cache 压力小） | **别用 `--enforce-eager` 换速度**，它会触发首次请求 JIT 编译卡 20+ 秒，见 15.2 节 |
| 首响体感 | `engine/filler.py` 的语气词（等待时先出声） | 无 |
| ASR | 换 FireRedASR2（第 12 步），或开投机 ASR `USE_MOONSHINE_ASR=1` | 投机 ASR 中文能力弱，只适合做预推理 |
| 判停等待 | `engine/conversation_manager.py` 的 `ENDPOINT_*_CHUNKS`（8/13/20 帧） | **调小会更灵敏，但也更容易打断用户说话**，慎调 |
| TTS 首 chunk | `engine/tts.py` 的切句粒度 | 切太碎会不自然 |

> ⚠️ 改 `engine/*.py` 的常量属于**改代码**。建议先用环境变量（`config.py` 里大部分项都支持
> 环境变量覆盖），实在没有再改代码。

### 14.4 双卡才值得做的进阶优化

- **NUMA 绑定**：`nvidia-smi topo -m` 看两块卡的 CPU Affinity，用 `numactl --cpunodebind=N --membind=N`
  把 vLLM / ws_server 分别绑到对应 NUMA 节点，减少跨节点内存访问（对 128GB 双路机器收益明显）。
- **PCIe 带宽**：确认两块卡都跑在 x8 以上（`nvidia-smi -q | grep -A2 "Link Width"`）。
  如果语音卡被插在 x4 槽，ASR/TTS 的模型加载和数据搬运会变慢。
- **CPU 侧组件分配**：Silero VAD / 降噪 / 投机 ASR / RAG 都在 CPU，可以把它们的线程数
  与 NUMA 节点错开（`OMP_NUM_THREADS`、`torch.set_num_threads`），避免和 GPU 喂数据抢 CPU。

---

## 15. 常见错误排查

| 症状 | 原因与解法 |
|---|---|
| `ModuleNotFoundError: No module named 'fireredasr2.models'` | **坑 1**。要么去掉 `USE_FIRERED_ASR=1`，要么做第 12 步 |
| `model not found` / 404 from vLLM | `config.VLLM_MODEL_NAME` ≠ vLLM `--served-model-name` |
| 连不上 LLM / Connection refused | 端口不一致：`config.VLLM_BASE_URL` 是 8100，而 `start_vllm.sh` 起在 **8000** |
| 两个进程挤在同一张卡 | 忘了设 `CUDA_VISIBLE_DEVICES`；用 `nvidia-smi` 确认分工 |
| `torch.cuda.device_count()` 只有 1 | 物理/BIOS 问题或驱动没认全（第 2.1 节） |
| `torch.cuda.is_available()` 为 False | PyTorch 装成 CPU 版，或与驱动不匹配（第 8.1 节） |
| **`Loading TTS` 阶段 `torch.OutOfMemoryError: CUDA out of memory`** | **最常见、也最容易误判**。见下方专节 |
| CUDA OOM | 第 14.2 节；双卡下先怀疑"没分卡" |
| 卡在 `Loading VAD` 很久 | Silero 在联网下载，把 `silero_vad.jit` 放到 `models/snakers4_silero-vad/`（3.4 节） |
| 声音变调/变速 | 采样率错配。上行必须 16k、下行 44.1k；**错配不报错，只变调** |
| 能识别但没声音 | TTS 没装好（`nanovllm_voxcpm`），看启动日志有没有 `Loading TTS` 之后报错 |
| 打断后还能听到半句 | WebSocket 走 TCP 的已知特性；前端靠 `playableTurn` 过滤。要更彻底只能换 UDP 传输（当前代码未实现） |
| 一直提示"没太听清" | 连续 3 次 ASR 空结果会升级提示。检查麦克风格式、是否开了降噪 |
| `/api/info` 403 | 设了 `VOICEAGENT_API_KEY` 但请求没带 token |
| 前端连不上 | 检查 `PORT`、防火墙（`sudo ufw allow 3000`）；HTTPS 页面连 ws 需要 `USE_SSL` 配合 |

### 15.1 `Loading TTS` 阶段 CUDA OOM（实测踩过，重点看）

**典型日志**：

```
21:19:57 [INFO] Loading TTS (VoxCPM nanovllm)...
[W] memory allocation failed with OOM on device 0 while trying to allocate 620756992 bytes
    (free: 98435072, total: 25282805760)          ← 只剩 94MB 空闲！
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 882.00 MiB.
GPU 0 has a total capacity of 23.55 GiB of which 315.88 MiB is free.
  Process 86214 has 10.33 GiB memory in use.          ← 谁？
  Process 89291 has 1.27 GiB memory in use.
```

**根因（最重要的一条）**：报错里"`Process xxx has N GiB`"的那些进程**不是你这次启动的**——
TTS 还没加载起来，卡上却已经有 10GB+ 被占了。**几乎总是上一次运行没退干净**。

为什么退不干净？**`nanovllm`（TTS 后端）在加载时会 fork 出持有 CUDA 上下文的子进程**
（`engine/tts.py` 里也提到过这个 fork 行为）。
这个子进程**不监听任何端口**，所以：

- 只 `kill` 父进程 → 子进程变孤儿，继续占 10GB+
- 按端口找 PID 去杀 → **找不到它**（它不监听端口）

**三重解法（本仓库已内置）**：

```bash
# ① 先看是谁占着，并清掉
bash start_prod.sh clean-gpu              # 只列出占用者（安全）
FORCE=1 bash start_prod.sh clean-gpu      # 确认后真的杀（会杀该卡所有计算进程）

# ② 用 stop（它现在按【进程组】杀，能带走 TTS 子进程）
bash start_prod.sh stop

# ③ 再启动；脚本现在会在启动前检查显存并提示
bash start_all.sh
```

**脚本侧的修复**（如果你自己写启动脚本，照这个做）：

- 服务一律用 `setsid` 启动 → 成为独立进程组组长，PID == PGID
- 停止时用 `kill -TERM -PGID`（**负号 = 进程组**），而不是 `kill PID`
- 已实测：只杀父 PID 会留下 2 个孤儿；整组杀后为 0

**如果清干净后仍然 OOM**，才考虑调小显存占用：

```bash
# TTS 默认按卡的 45% 预占显存，与 ASR 同卡时可能不够
TTS_GPU_UTIL=0.35 bash start_all.sh
```

**通用手段：开 PTT 演示模式隔离问题**

```bash
DEMO_MODE=1 CUDA_VISIBLE_DEVICES=1 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 \
RAG_DEVICE=cpu TTS_GPU_UTIL=0.45 python ws_server.py
```

Demo 模式**跳过 VAD/判停/filler/打断**，按住说话→松手→直接跑 ASR→RAG→LLM→TTS。
如果 Demo 正常、全双工不正常，问题就锁定在 VAD/判停这一层。

### 15.2 第一次对话卡二十几秒，然后 `LLM stream error: Request timed out`

**这是本项目最隐蔽的一个坑**：报错指向"超时"，但真因在 vLLM 的**启动参数**里。

**实测日志**（vLLM 日志与对话日志对照着看）：

```
21:57:55,084  POST /v1/chat/completions "HTTP/1.1 200 OK"      ← 请求已被接受
21:57:55,807  flashinfer.jit: Building JIT module sampling;
              this can take several minutes on first use.        ← 真因在这里
21:58:25,112  [Turn 3] LLM/TTS error: LLM stream error: Request timed out.
              RuntimeError: LLM stream error: Request timed out.
```

**为什么**：如果启动 vLLM 时带了 `--enforce-eager`，它会**连带关掉 JIT kernel warmup**，
vLLM 自己的日志就写着：

```
WARNING Enforce eager set, disabling torch.compile, CUDAGraphs, and JIT kernel warmup.
        This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
        and --kernel_config.enable_jit_warmup=False
```

于是 FlashInfer 的采样内核**没在启动期编译好**，被推迟到**第一个真实请求**时现场编译——
实测这一次编译耗了 **24 秒**。而 `engine/llm.py` 用的是
`httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=10.0)`，其中 `read=30s` 的含义是
"**30 秒内没收到任何数据**就超时"，于是正好被撞穿。

两个容易看漏的点：

- 日志里的 `HTTP/1.1 200 OK` 是**响应头**到达（httpx 只记这个），**不代表已经开始吐 token**
- `logs/vllm.log` 里同时能看到 `Avg generation throughput: 3.3 tokens/s`（正常应为 **30~50**）

**解法**：启动参数里**去掉 `--enforce-eager`**（本仓库的启动脚本已经不带它了）。

```bash
# ❌ 会让首次请求现场 JIT 编译，卡 20+ 秒
--enforce-eager

# ✅ 不加：启动期就把 JIT / CUDA Graph 准备好
#    代价只是启动多花 1~3 分钟，换来运行期正常速度
```

**改完怎么确认好了**：JIT 预热会挪到**启动阶段**（vLLM 日志里能看到它编译），
之后第一句对话应当 **2~4 秒**内出完整回答，不再有 timeout。

**顺手压一下 vLLM 的真实速度**（绕开语音链路，单独测 LLM）：

```bash
time curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-14B-AWQ","messages":[{"role":"user","content":"用一句话介绍你自己"}],
       "max_tokens":80,"temperature":0.85}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['usage'])"
```

看 `completion_tokens` 与耗时：**正常是 30~50 tokens/s**。
如果只有个位数，先按 15.1 节确认有没有别的进程在抢同一张卡。


---

## 16. 可选：把 ASR 升级为 FireRedASR2

这一步是为了拿到 README 宣称的 **CER 2.89%** 和 ~200ms 的 ASR 延迟。

### 16.1 补齐缺失的 `models/` 子包（**必须**）

前面说过，`engine/fireredasr2/models/` 在仓库里不存在，而 `asr.py` 硬依赖它。补法：

```bash
cd ~
# 上游仓库（ASR/VAD/LID/Punc 全家桶）
git clone https://github.com/FireRedTeam/FireRedASR2S.git
cd FireRedASR2S

# 看一下上游的目录结构，确认 models/ 的位置
find . -maxdepth 3 -type d -name models

# 典型结构是 fireredasr2s/fireredasr2/models/...
# 把它整体复制进本仓库：
cp -r fireredasr2s/fireredasr2/models \
      /path/to/600ms_voice_agent/engine/fireredasr2/models

cd /path/to/600ms_voice_agent
ls engine/fireredasr2/models/     # 应看到 fireredasr_aed.py / fireredasr_llm.py / lstm_lm.py / param.py
```

> ⚠️ **我无法在当前环境联网核实上游当前的目录层级**（网络抓取超时），
> 上面的路径是**基于本仓库 `asr.py` 的相对导入推断**的：它要求
> `engine/fireredasr2/models/{fireredasr_aed,fireredasr_llm,lstm_lm,param}.py` 存在。
> 请用 `find` 的实际结果为准调整 `cp` 的源路径。

### 16.2 下载 FireRedASR2-AED 权重

放到 `models/FireRedASR2-AED/`，并确认关键文件齐全：

```bash
ls models/FireRedASR2-AED/
# 代码会去找这些（见 engine/fireredasr2/tokenizer/aed_tokenizer.py 与 data/asr_feat.py）：
#   dict.txt              <- 词表
#   train_bpe1000.model   <- 英文 sentencepiece
#   cmvn.ark              <- Kaldi CMVN 统计量
#   + 模型权重文件
```

> ⚠️ `dict.txt` 与 `train_bpe1000.model` **必须成套**（来自同一次发布），
> 否则会出现"识别结果看着像乱码"的问题。
> 另外上游的 `models/` 子包里还有一些**参数字典**需要与权重匹配。

### 16.3 启用

```bash
CUDA_VISIBLE_DEVICES=1 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 RAG_DEVICE=cpu \
TTS_GPU_UTIL=0.45 USE_FIRERED_ASR=1 python ws_server.py
```

日志应变成 `Loading ASR (FireRedASR2-AED, CER 2.89%)...`。

> 另需装 `sentencepiece`、`kaldi_native_fbank`、`kaldiio`、`textgrid`（若尚未安装）：
> ```bash
> pip install sentencepiece kaldi_native_fbank kaldiio textgrid
> ```
>
> 💡 升级后 ASR 显存占用会上升（AED 比 SenseVoice 大），如果 GPU 1 紧张，
> 可以把 `TTS_GPU_UTIL` 从 0.45 降到 0.40 腾地方。

---

## 17. 生产化（常驻与开机自启）

跑通之后想长期运行，三种方式可靠性递增：

| 方式 | 命令 | 说明 |
|---|---|---|
| 前台托管 | `bash start_all.sh` | 开发调试用；Ctrl+C 全停 |
| **常驻 + 内置守护**（推荐） | `bash start_prod.sh` | 服务脱离终端常驻；`keep_alive.sh` 每 30s 巡检 vLLM 与 Voice Agent，掉线自动拉起 |
| systemd | 自己写 unit | 能**开机自启**、崩溃按策略重启、日志进 journald |

`start_prod.sh` 的配套命令：

```bash
bash start_prod.sh status    # 端口 / PID / 显存 / 日志位置
bash start_prod.sh stop      # 停止全部（按进程组杀，含守护与 TTS 子进程）
bash start_prod.sh clean-gpu # 列出占用显存的进程；FORCE=1 时强制清理
bash start_prod.sh restart
```

**为什么仍然推荐 systemd**：`start_prod.sh` 的守护是用户态轮询——机器重启后就没了。
systemd 能开机拉起、崩溃自动重启；双卡下还能给两个 unit 分别设 `CUDA_VISIBLE_DEVICES`，
避免"谁先启动谁抢到卡"的竞态。

vLLM 的 unit 示例（`/etc/systemd/system/voiceagent-vllm.service`）：

```ini
[Unit]
Description=VoiceAgent vLLM (GPU 0)
After=network.target

[Service]
User=<你的用户名>
WorkingDirectory=<仓库绝对路径>
Environment=CUDA_VISIBLE_DEVICES=0
Environment=VOICEAGENT_PYTHON=<解释器绝对路径>
# start_vllm.sh 是前台运行（exec），正好符合 systemd 的 Type=simple
ExecStart=/bin/bash <仓库绝对路径>/start_vllm.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Voice Agent 同理，`ExecStart` 换成直接跑 `ws_server.py`，并在 `Environment=` 里设好
`CUDA_VISIBLE_DEVICES=1 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 RAG_DEVICE=cpu TTS_GPU_UTIL=0.45`。

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voiceagent-vllm
```

> 新脚本已彻底去掉 `sleep 70/110` 那种死等，改为轮询就绪信号，不会与 systemd 的重启策略互相打架。

## 18. 一页速查

```bash
cd <仓库根目录>
conda activate voiceagent                 # 或 export VOICEAGENT_PYTHON=<python 绝对路径>

# ── 开发/演示：前台启动，Ctrl+C 停 ──
bash start_all.sh
#   浏览器打开 http://<本机IP>:3000/

# ── 生产：常驻 + 守护 ──
bash start_prod.sh            # 启动
bash start_prod.sh status     # 状态
bash start_prod.sh stop       # 停止
bash start_prod.sh clean-gpu  # 显存被占满时先看是谁（FORCE=1 强制清理）

# ── PTT 演示模式（隔离 VAD/判停问题）──
DEMO_MODE=1 bash start_all.sh

# ── 只调试 LLM ──
bash start_vllm.sh

# ── 换 LLM 权重（仓库里有两份）──
LLM_MODEL_DIR=models/MiniCPM4.1-8B-GPTQ LLM_SERVED_NAME=MiniCPM4.1-8B-GPTQ bash start_all.sh

# ── 验证 ──
curl -s http://localhost:8100/v1/models | python -m json.tool
curl -s http://localhost:3000/api/info  | python -m json.tool
nvidia-smi                                # GPU0≈vLLM / GPU1≈ASR+TTS
```

**必须记住的五条**：
1. 依赖只差 TTS 引擎 `nano-vllm-voxcpm`（要编译 flash-attn，见第 3.2 节）；其余用 `pip install -r requirements.txt` 即可；
2. 模型名与端口由 `voiceagent_env.sh` 从 `config.py` 自动读取，**不需要你手动对齐**；
3. `CUDA_VISIBLE_DEVICES=1`（物理卡）与 `ASR_DEVICE=cuda:0`（进程内序号）含义不同，别混；
4. `config.py` 的 `LLM_GPU`/`ASR_GPU`/`TTS_GPU`/`RAG_GPU` 是**死变量**，设了没用；
5. `USE_FIRERED_ASR=1` 在补齐 `engine/fireredasr2/models/` 之前**不要开**。

---

# 第三部分 · 精读代码（12 课）

> 建议先读完第一部分再进来。每课含：精确行号锚点 → 机制解释 → 自测题。
> 行号基于当前代码；若你改过代码导致漂移，用函数名搜索即可。

## 怎么用这份材料

- **不要顺着文件从头到尾读**。主文件 `conversation_manager.py` 有 1300+ 行（注释精简后），
  按调用链读效率高得多。
- 每课的节奏：先看「目标」→ 打开「读哪些」列出的位置 → 对照「核心机制」理解 →
  合上代码回答「自测题」。**答不出来就回去再读那一段**。
- 遇到常量，直接去 `config.py` 或该模块顶部的注释查来历——注释写的是"为什么"，没写"是什么"。
- 建议配合一次真实运行（第二部分），一边看日志一边读代码。

**总体路线图**（这就是你要建立的调用链心智模型）：

```
浏览器 static/voice_agent.html
   └─ PCMSender(AudioWorklet)  攒 512 点/32ms  → WebSocket 二进制帧
        ↓
ws_server.py  ws_voice()                      ← 连接生命周期、消息分发
   └─ ConversationManager.feed_audio()        ← 音频入口
        └─ _process_chunk()                   ← ★ 状态机核心（第 4 课）
             ├─ VAD.process_chunk()           ← 逐帧判语音（第 3 课）
             ├─ _adaptive_endpoint_threshold()← 判停档位（第 5 课）
             └─ _on_endpointing()             ← 判停触发
                  └─ _run_pipeline()          ← ★ 一轮编排（第 11 课）
                       ├─ ASR.transcribe()    ← 第 6 课
                       ├─ RAG.get_context()   ← 第 7 课
                       └─ _stream_llm_tts()   ← ★ 流式串联（第 8/9 课）
                            ├─ VLLMChat.stream_sentences()  逐句
                            └─ VoxCPMTTS.synthesize_stream() 逐 chunk
                                 └─ _send_audio() → 前端 JitterBufferPlayer
```

---

## 学习节奏建议

### 第 1 天：建立全局观
1. `config.py`（195 行，每个常量都标了单位/取值/消费者）—— 认识所有可调项
2. 本文档第 2 节时序图 —— 建立数据流心智模型
3. `README.md` —— 看设计取舍（注意第 9 节的偏差清单）
4. 跑 `python -m py_compile` 或直接读 `static/voice_agent.html` 的协议部分

### 第 2~3 天：走通主链路
5. `ws_server.py`：先看 `load_engines()`（装配顺序），再看 `ws_voice()`（消息分发）
6. 按消息类型顺藤摸瓜：二进制帧 → `conversation_manager.feed_audio()`

### 第 4~5 天：啃状态机
7. `engine/conversation_manager.py`：**先读模块顶部的中文总览**（有状态流转图），再按 5 个状态分支读 `_process_chunk()`
8. 重点理解三件事：判停逻辑、双层打断、Turn 序列号

### 第 2 周：按需深入
9. 想调延迟 → `engine/llm.py`（切句策略）、`engine/tts.py`（流式 chunk）、`engine/filler.py`

> 💡 **阅读技巧**：本仓库的注释**只写"为什么"，不复述代码**——所以看到没有注释的行是正常的，那说明它一眼就懂。
> 遇到看不懂的常量，去 `config.py` 或模块顶部的注释找它的来历；踩过的坑统一搜 `⚠️` 和 `注意`。

---


## 第 1 课：入口与装配 —— `ws_server.py: load_engines()`

**🎯 目标**：搞清"服务启动时到底按什么顺序把哪些引擎装起来"，以及**哪些是共享单例、哪些必须每连接新建**。

**📖 读哪些**
- `ws_server.py:101` `load_engines()` —— 装配主函数
- `ws_server.py:63` `_model_dir_ready()` —— "auto" 模式的目录探测
- `ws_server.py:168` 附近 —— FireRed / SenseVoice 的 ASR 二选一

**🔑 核心机制**

装配顺序是**刻意**的（注释里有）：
```
VAD(三选一) → 投机ASR → 降噪 → SmartTurn → 主ASR → RAG(建索引) → LLM(只存工厂)
→ TTS(load+音色克隆+warmup) → filler(预合成) → captioner
```

三条最关键的边界，务必记住：

| 组件 | 生命周期 | 为什么 |
|---|---|---|
| **VAD** | **每连接新建**（启动时那份只当"模板") | VAD 带流式状态（滑窗、静音计数），共享会串话 |
| **LLM** | **每会话新建**（启动时只存 lambda 工厂） | `VLLMChat` 持有 `_history`，共享会让两个用户看到彼此的对话 |
| **ASR / TTS / RAG / filler** | **跨会话共享单例** | 无会话状态（或状态已按调用隔离），加载一次省显存和时间 |

这就是为什么代码里是 `engine["llm_factory"] = lambda: VLLMChat(...)` 而不是直接实例化——
**这是一个必须看懂的设计**。

**❓ 自测题**
1. 为什么 TTS 要在启动时 `warmup()`，而 ASR 不用？
2. 如果把 VAD 改成全局单例，会出现什么现象？
3. `auto` 模式下 `_model_dir_ready()` 判断"目录就绪"的依据是什么？

---

## 第 2 课：连接生命周期与消息分发 —— `ws_server.py: ws_voice()`

**🎯 目标**：理解一条 WebSocket 连接从建立到关闭发生了什么，以及消息是怎么分流的。

**📖 读哪些**
- `ws_server.py:379` `ws_voice()` —— 主处理函数（docstring 里有完整的上下行协议表）
- `ws_server.py:251` `_check_api_key()`、`ws_server.py:265` `verify_api_key()`
- `ws_server.py:279` `index()` —— 根路径返回前端页面

**🔑 核心机制**

1. **鉴权在 `accept()` 之前**：不合法直接 `close(code=4003)`，不建立会话、不耗时加载 VAD。
   HTTP 接口的鉴权失败则是 **403**——两者错误码不同，别混。
2. 下发 `ready` 帧（含 `session_id` / `version` / `mode`），然后**非 DEMO_MODE 时**播问候语。
3. 进入收发循环，按帧类型分流：
   - **二进制帧** → 音频（16k/单声道/int16/512 点）→ `cm.feed_audio()`
   - **文本帧** → JSON，按 `type` 分发：`ptt_start` / `ptt_end` / `text_input` /
     `manual_barge_in` / `enroll` / `unenroll` / `reset`
4. `send_fn` 是服务端唯一的下行出口，用 `"_audio"` 键区分走二进制还是 JSON。
   死连接只记一次失败就退出循环。

**❓ 自测题**
1. 为什么鉴权要放在 `accept()` 之前？（提示：想想"不合法连接"的资源开销）
2. 客户端发了一段 1024 采样点的音频，服务端会怎么处理？（提示：`CHUNK_SAMPLES`）
3. `ready` 帧里的 `mode` 字段是给谁用的？

---

## 第 3 课：音频帧的旅程（从麦克风到状态机）

**🎯 目标**：把"声音怎么变成服务端能处理的数据"这条链条走通，并记住**两端必须严格对齐的参数**。

**📖 读哪些**
- `static/voice_agent.html:442` `PCMSender`（内嵌 AudioWorklet）
- `static/voice_agent.html:391` 附近 —— `getUserMedia` 的采样率设置
- `engine/conversation_manager.py:477` `feed_audio()`
- `engine/vad.py:46` `process_chunk()`

**🔑 核心机制**

```
浏览器采集 16kHz 单声道 → float32 [-1,1]
  → PCMSender 每攒够 512 点（=32ms）发一帧
  → 转 int16 小端（×32767 后取整）
  → WebSocket 二进制帧
  → 服务端 int16 / 32768 还原成 float32
  → feed_audio() → 按 CHUNK_SAMPLES 重新切帧 → _process_chunk()
```

**必须对齐的三个数**（任一处不一致都会出问题）：

| 参数 | 值 | 在哪定义 |
|---|---|---|
| 上行采样率 | 16000 Hz | `config.SAMPLE_RATE` |
| 帧长 | 512 采样点 / 32ms | `config.CHUNK_SAMPLES` / `CHUNK_MS` |
| 下行采样率 | **44100 Hz**（TTS 输出） | `engine/tts.py` 的 `SAMPLE_RATE` |

> ⚠️ 上下行采样率**不同**是设计（ASR 用 16k，VoxCPM 输出 44.1k），前端靠 `audio_start`
> 消息里的 `sample_rate` 得知。**错配不会报错，只会变调**——这是最难查的一类 bug。

**❓ 自测题**
1. 为什么用 AudioWorklet 而不是普通的 `setInterval` 采集？
2. `feed_audio` 收到的数据为什么还要"重新切帧"？
3. 如果前端发 48kHz 音频会发生什么？

---

## 第 4 课：状态机核心 —— `conversation_manager.py: _process_chunk()`

**🎯 目标**：这是**全项目最重要的一课**。读懂 `_process_chunk()`，你就读懂了整个对话编排。

**📖 读哪些**
- `engine/conversation_manager.py:30` `class State` —— 5 个状态的定义与语义
- `engine/conversation_manager.py:490` `_process_chunk()` —— **主战场**
- `engine/conversation_manager.py:7` 附近 —— 模块顶部的状态流转图（先读它！）

**🔑 核心机制**

5 个状态（取值与下行 `{"type":"state"}` 一一对应，前端据此亮灯）：

| 状态 | 含义 | 关键行为 |
|---|---|---|
| `IDLE` | 待机 | 唯一的"闲聊超时检查点"（D2）；声纹门控在此校验 |
| `LISTENING` | 用户在说 | 累积音频到 `_audio_buffer`，累计静音帧数判停 |
| `THINKING` | 已判停，正在跑 ASR/RAG/LLM | 用户若继续说话 → 进 `_thinking_extra_buffer`，等首段 ASR 出来**合并**，避免一句话被切成两轮 |
| `SPEAKING` | 正在播报 | 检测到用户语音 → 触发 Barge-in |
| `INTERRUPTED` | 播报被打断 | 收敛旧音频、复位，然后回 IDLE/LISTENING |

`_process_chunk()` 就是**一个大分支**：每个状态一个 `elif` 分支，分支内用 `is_speech`
再分"有语音/无语音"两条路。

**三个容易看漏的细节**：
1. **升/降沿驱动**：VAD 返回的是 `speech_start` / `speech_end` 事件，状态迁移由边沿触发，
   不是每帧都判断。
2. **`await asyncio.sleep(0.01)` / `sleep(0)`** 这类让出不是凑数——它是为了让事件循环
   能处理下一批麦克风帧（否则长循环会饿死接收协程，表现为"打断不灵"）。
3. **THINKING 的后到语音合并**是"自然对话"的关键，去掉它会明显感觉"抢话"。

**❓ 自测题**
1. 用户在 THINKING 阶段又说了半句，会发生什么？
2. 为什么超时检查（D2）只在 IDLE 做？
3. 状态从 SPEAKING 到 INTERRUPTED 是被什么触发的？

---

## 第 5 课：判停（Endpointing）—— 名字会骗你的地方

**🎯 目标**：搞清"用户说完了"是怎么判定的，并**记住一个代码与注释不一致的陷阱**。

**📖 读哪些**
- `engine/conversation_manager.py:682` `_adaptive_endpoint_threshold()`
- `engine/conversation_manager.py:668` `_check_smart_turn()`
- `engine/conversation_manager.py:698` `_on_endpointing()`
- 常量：`engine/conversation_manager.py:61-64`

**🔑 核心机制**

判停 = "连续静音了多少帧"。档位由**本次说话的时长**自适应：

| 本次说话时长 | 实际使用的档位 | 需要静音 |
|---|---|---|
| < 0.5s | `ENDPOINT_SLOW_CHUNKS` | 20 帧 ≈ **640ms** |
| 0.5 ~ 3s | `ENDPOINT_DEFAULT_CHUNKS` | 13 帧 ≈ **416ms** |
| > 3s | `ENDPOINT_SLOW_CHUNKS` | 20 帧 ≈ **640ms** |

> ⚠️ **陷阱（一定要记住）**：`_adaptive_endpoint_threshold()` 的 docstring 和常量名都暗示
> "短句走 FAST 档（8 帧≈256ms）"，但**代码里两个分支返回的都是 SLOW 档**，
> `ENDPOINT_FAST_CHUNKS` 是**从未被使用的死常量**。
> 所以"短句会更快判停"这个说法**与代码行为不符**——实际短句等得最久。
> 推测是有意的下限保护，也可能是笔误。**读代码时要信代码，不要信名字。**

另可选 `USE_SMART_TURN=1` 启用 Smart Turn v3 做**语义**判停（不只靠静音），
对应 `engine/turn_detector.py`；它判 False 时会兜底回到 640ms 静音阈值。

**❓ 自测题**
1. 用户说了一句 1 秒钟的话，需要多久静音才算说完？
2. 为什么"长句 >3s"要用更长的静音阈值？
3. 如果用户说话总是被"从中间切断"，你会先调哪里？（注意调小的副作用）

---

## 第 6 课：ASR 链路（含投机式预推理）

**🎯 目标**：理解 ASR 的两个实现如何替换，以及"投机 ASR"这个省 117ms 的技巧。

**📖 读哪些**
- `engine/asr.py:13` `SenseVoiceASR`（默认实现，FunASR）
- `engine/asr_firered.py:22` `FireRedASR`（生产主力，FireRedASR2-AED）
- `engine/asr_moonshine.py` —— 投机 ASR（Moonshine Tiny 27M，**纯 CPU**）
- `engine/conversation_manager.py:641` `_launch_speculative_asr()`
- `engine/conversation_manager.py:276` `_is_short_utterance()` / `:281` `_drain_accum()`

**🔑 核心机制**

**统一接口契约**（换 ASR 只需实现它）：
```python
transcribe(audio: np.ndarray, sr: int = 16000) -> dict   # 至少含 text / asr_ms
```

**投机式 ASR**（`USE_MOONSHINE_ASR=1`）：
用户停顿约 160ms（5 帧）时，后台**抢跑**一次轻量 ASR；如果随后 Endpointing 确认说完了，
就直接复用这次结果，**省掉一次 ASR 时间（约 117ms）**。它靠"音频长度新鲜度"
（`spec_audio_len * 1.15`）判断抢跑结果是否还有效。命中时 `asr_ms` 记为 0——因为它没有
落在首响关键路径上。

**ASR 短文本累积**：识别结果 ≤4 字（如"嗯""什么"）时**不立即送 LLM**，先缓冲，
等后续语音拼起来再一起送，避免把一句话切成两轮。超时 3s、最多 3 段。

**❓ 自测题**
1. 想接入 Whisper，你需要改哪些文件？（提示：分支逻辑在哪）
2. 投机 ASR 为什么用 27M 的小模型而不是直接复用主 ASR？
3. 短文本累积器的存在，会让哪些测试用例"看起来失败"？

---

## 第 7 课：RAG 检索与置信度门控

**🎯 目标**：理解检索链路，重点理解**0.35 这个阈值为什么重要**。

**📖 读哪些**
- `engine/rag.py:25` `__init__` / `:48` `load()` / `:76` `build_index()`
- `engine/rag.py:115` `query()` / `:186` `get_context()`
- `config.py` 的 `RAG_SCORE_THRESHOLD` / `RAG_TOP_K`

**🔑 核心机制**

```
documents(JSON) → bge 向量化（L2 归一化 → 内积 == 余弦）→ FAISS IndexFlatIP
查询 → 编码 → 检索 top_k → 打分 → 与阈值比较 → 拼成 context 注入 system prompt
```

**置信度门控（防幻觉的核心设计）**：
相似度 `< RAG_SCORE_THRESHOLD (0.35)` 就**不注入知识库内容**，让 LLM 走
`SYSTEM_PROMPT_RAG` 里约定的"这个我问下同事，再给您回复哈"兜底话术——
**宁可说不知道，也不编造**。

启动时 `ws_server` 会读 `data/sample_kb.json` 并**在内存里建索引**（不落盘），
所以每次启动都要重新编码一遍。

> ⚠️ 陷阱：一旦启用 reranker，阈值比较的是 **logit 量纲**的 rerank 分数，
> 和余弦相似度 0.35 **不等价**，需要重新标定。

**❓ 自测题**
1. 为什么用 `IndexFlatIP` 而不是 IVF/HNSW？
2. 把阈值从 0.35 调到 0.6 会发生什么？
3. 知识库有 1000 条时，启动会慢在哪一步？

---

## 第 8 课：LLM 流式与切句 —— 首响的关键

**🎯 目标**：理解"为什么 LLM 的第一句能这么快出来"，以及切句策略的取舍。

**📖 读哪些**
- `engine/llm.py:30` `class VLLMChat`
- `engine/llm.py:91` `_build_messages()` —— system + RAG + 历史怎么拼
- `engine/llm.py:150` `stream_sentences()` —— **流式切句主逻辑**
- `engine/llm.py:68` `_trim_history()` / `:237` `_clean()`

**🔑 核心机制**

**关键设计：不等整段回复，而是"就近按标点切句"**，每切出一句就立刻丢给 TTS。
这是首响能到 458ms 的核心——LLM 还在生成后半段时，用户已经听到第一句了。

切句用 `_SENTENCE_END_RE`，**标点集合里刻意包含逗号**：宁可早说半句，也不要让用户干等。
代价是可能切在语义不完整处，但语音场景下"早出声"比"句子完整"更重要。

其他要点：
- `enable_thinking=False`（关 Qwen3 思考模式）+ `repetition_penalty=1.15`（压复读）
- 历史用**滑动窗口 + 60 字粗摘要**，避免长对话把 prompt 撑爆
- `_clean()` 用三道正则清洗 `<think>` / XML 残留（二次防御）
- `max_tokens=80` 是刻意的——语音回复本来就不该长

**❓ 自测题**
1. 如果把逗号从切句标点里去掉，首响会变快还是变慢？
2. 为什么 LLM 的延迟指标叫 `ttfs`（time to first sentence）而不是 `ttft`？
3. 历史摘要为什么拼在 system 消息末尾，而不是单独一条消息？

---

## 第 9 课：TTS 流式合成与语气词

**🎯 目标**：理解流式合成的细节，以及"语气词"这个体验优化。

**📖 读哪些**
- `engine/tts.py:176` `synthesize_stream()` —— 流式合成（**打断的粒度就在这里**）
- `engine/tts.py:66` `register_voice()` / `:96` `warmup()`
- `engine/filler.py:32` `pregenerate()` / `:68` `get_filler()`

**🔑 核心机制**

**流式合成的三个细节**：
1. **丢弃第一个 chunk**：VoxCPM 的 VAE 有启动瞬态，首块音频是"噗"的杂音。
2. **第二个 chunk 做 10ms 线性淡入**：防止拼接处爆点。
3. `ttfa` 记在首块上，但对外挂在**首个可听块**上——否则指标会虚高。

**"160ms 精确打断"的真相**：打断**不是**在 `synthesize_stream` 里实现的，
而是**调用方**在每次下发 chunk 之前检查 `_cancel_speaking`。
所以打断粒度 = **1 个 chunk（约 160ms）**，且取消只停"下发"、不掐"推理"。

**语气词（filler）**：VAD 判停到 LLM 首 token 之间有几秒空档。启动时把 5 条中文语气词
预合成成裸 PCM 缓存，运行时**取到即发、延迟≈0**。

> 细节：语气词末尾都带逗号（"嗯，"），是为了让模型用**升调**读出来，
> 听起来像"我在听"而不是"我说完了"——避免用户误以为轮到自己而抢话。

**❓ 自测题**
1. 为什么语气词要在**启动时**预生成，而不是用到时再合成？
2. `MAX_FILLER_SEC = 2.0` 这个上限在防什么？
3. 打断时"只停下发不掐推理"有什么好处和坏处？

---

## 第 10 课：打断（Barge-in）—— 本项目的招牌能力

**🎯 目标**：彻底搞懂"双层打断"和"Turn 序列号"，这是全项目最有工程含量的部分。

**📖 读哪些**
- `engine/conversation_manager.py:428` `handle_barge_in()`
- 常量：`BARGE_IN_VAD_THRESHOLD` / `BARGE_IN_RMS_THRESHOLD` / `BARGE_IN_CONFIRM_CHUNKS`
- `static/voice_agent.html:273` `stopPlayback()` / `:220` `playAudio()`
- `static/voice_agent.html` 里 `playableTurn` 的全部出现位置（搜一下）

**🔑 核心机制**

**第一层：状态机置位**。`handle_barge_in()` 把 `_cancel_speaking` 这个 `asyncio.Event` 置位。

**第二层：各循环自己检查**。三条发送/合成循环（LLM 句子循环、TTS 逐 chunk 循环、
系统话术循环）**各自**在关键点检查这个 Event 并主动退出。

> **关键设计："没有人 cancel 别人"。** 不是由一个"管理者"去 `task.cancel()`，
> 而是每个循环自己看标志位优雅退出。这样避免了 `CancelledError` 在任意点抛出导致的
> 资源泄漏和状态错乱。

**触发条件（防误触发）**：
```
VAD 语音概率 > 0.6  且  RMS > 0.012  且  连续 2 帧（≈64ms）确认
```
三道条件缺一不可——否则环境噪声、TTS 回声都会误触发。

**Turn 序列号（防串音）**：服务端每轮回复前发 `{"type":"audio_start","turn":N}`，
前端维护 `playableTurn`：

| 值 | 含义 |
|---|---|
| `-1` | 尚未授权 |
| `N` | 只有第 N 轮的音频可播 |
| `0` | 打断后的"闸门关闭"哨兵 |

**不匹配的帧全部丢弃**——这解决了"打断后，之前已发出去但还没播的音频又冒出来"。
这是 WebSocket（TCP）方案必须做的补偿；WebRTC 方案靠 UDP 语义天然规避。

**❓ 自测题**
1. 为什么不用 `task.cancel()` 实现打断？
2. 如果把 `BARGE_IN_CONFIRM_CHUNKS` 改成 1，会有什么副作用？
3. `playableTurn=0` 和 `playableTurn=-1` 的区别是什么？

---

## 第 11 课：一轮对话的编排 —— `_run_pipeline()`

**🎯 目标**：把前面各课串起来，看清"一轮对话"的完整编排与异常兜底。

**📖 读哪些**
- `engine/conversation_manager.py:737` `_run_pipeline()`（含投机结果的入口）
- `engine/conversation_manager.py:779` `_run_pipeline_from_text()`（跳过 ASR 的入口）
- `engine/conversation_manager.py:826` `_run_pipeline_inner()` —— **真正的主干**
- `engine/conversation_manager.py:1002` `_stream_llm_tts()` —— LLM↔TTS 流式串联
- `engine/conversation_manager.py:1178` `_send_audio()` / `:1196` `_send_filler()`

**🔑 核心机制**

一轮对话的编排顺序（`_run_pipeline_inner`）：
```
置状态 THINKING
  → 降噪（若开启）
  → ASR（或复用投机结果）
      ├─ 空结果 → D3 兜底话术，连击计数 +1
      └─ 短文本 → 进累积器，本轮不继续
  → RAG 检索 → 拿到 context（或走兜底）
  → _stream_llm_tts()：逐句 LLM → 逐 chunk TTS → 逐块下发
       ├─ 每个 chunk 前检查 _cancel_speaking（打断点）
       └─ 首块音频前发 audio_start(turn=N) 授权
  → 汇总 metrics 下发 → 回 IDLE、重置空闲计时
异常 → 道歉话术兜底 → 回 IDLE
```

**三个入口值得注意**：
- `_run_pipeline` = 语音输入（可能带投机结果）
- `_run_pipeline_from_text` = 文本输入（`text_input` 消息，跳过 ASR）
- `process_ptt` = PTT 整段音频（**不走 VAD/判停**）

**❓ 自测题**
1. `audio_start` 为什么必须在**首块音频之前**发？
2. 如果 LLM 在生成到一半时报错，用户会听到什么？
3. 三个入口中，哪个完全没有 VAD 参与？

---

## 第 12 课：体验层（D1–D4）与延迟指标

**🎯 目标**：理解"像一个真人"的那些细节设计，以及延迟指标怎么算、怎么读。

**📖 读哪些**
- `engine/conversation_manager.py:1264` `start_greeting()` —— D1 开场白
- `engine/conversation_manager.py:1284` `_check_idle_timeout()` —— D2 空闲追问/告别
- D3 空 ASR 重试：搜 `_consecutive_empty_asr`
- D4 告别关键词：`config.FAREWELL_KEYWORDS`
- `engine/conversation_manager.py:94` `class _SessionMetrics` / `:123` `summary()`
- `engine/conversation_manager.py:1302` `_speak_system()` / `:1321` `_play_system_tts()`

**🔑 核心机制**

四类"体验层"话术（产品需求编号，代码注释里保留了编号）：

| 编号 | 触发 | 话术来源 |
|---|---|---|
| **D1** | 连接建立 | `config.GREETING_TEXT` |
| **D2** | IDLE 超时（15s 追问 / 30s 告别） | `IDLE_PROMPT_TEXT` / `IDLE_GOODBYE_TEXT` |
| **D3** | ASR 连续空结果（最多 3 次） | `ASR_EMPTY_RETRY_TEXT` / `ASR_NOISY_SUGGEST_TEXT` |
| **D4** | 命中告别关键词（**子串匹配**） | 直接结束会话 |

> ⚠️ D4 用的是**子串匹配**（`FAREWELL_KEYWORDS` 里含"再见""没了"等），
> 所以"我还**没**有别的**没了**解"这种句子可能被误判为告别——实际部署要留意。

**延迟指标怎么算**：
```
first_response_ms = asr_ms + rag_ms + llm_ms + tts_ttfa_ms
```
四项分别来自各组件自己的计时。**注意它不含 VAD 判停那 256~640ms 的等待**，
也不含网络往返——所以 `first_response_ms` 会比用户真实体感略乐观。

`/api/metrics` 会打印累计统计（均值/条数），可以一边对话一边 `watch` 观察。

**❓ 自测题**
1. D2 的 15s/30s 计时基准是什么？（提示：`_idle_since`）
2. 为什么 `first_response_ms` 会低估用户的真实等待感？
3. 想加一个"用户骂人时安抚"的话术，你会照哪个模式加？

---


## 动手练习（建议按顺序做）

1. **加一条日志**：在 `_on_endpointing()` 里打印本次判停用了多少帧、累积音频多少 ms。
   跑一次对话，验证第 5 课的"短句实际等 640ms"。
2. **改判停阈值**：把 `ENDPOINT_DEFAULT_CHUNKS` 从 13 改成 8，感受"反应变快但更容易被切断"。
3. **关掉 RAG**：把 `RAG_SCORE_THRESHOLD` 调到 0.99，观察 LLM 是否开始走"问我同事"兜底话术。
4. **观察切句**：在 `_stream_llm_tts()` 里打印每次切出的句子，看逗号切分的效果。
5. **实现一个开关**：给 `SERVER` 加一个环境变量 `USE_FILLER=0` 来关闭语气词，
   走一遍"配置 → 装配 → 运行时生效"的完整链路（这是最接近真实需求的练习）。

## 如果只有 30 分钟

按这个顺序读，能抓住 80% 的骨架：

1. `engine/conversation_manager.py:30` 的 `State` 类（5 个状态）
2. `engine/conversation_manager.py:490` `_process_chunk()` 的分支结构（不用读细节）
3. `engine/conversation_manager.py:1002` `_stream_llm_tts()`（首响的秘密）
4. `engine/conversation_manager.py:428` `handle_barge_in()`（打断）
5. `ws_server.py:101` `load_engines()`（装配与生命周期）

# 附录

## 附录 A：文档与代码不一致清单（实测，重要）

**这是本文档最有价值的一节。** 以下都是我在逐文件核实代码时**实测确认**的偏差——
不是 bug 清单（我没有修改任何代码），而是"**你以为是这样、实际是那样**"的陷阱。
读代码时请记住一条原则：**信代码，不要信名字。**

### A.1 部署必读（不解决就跑不起来）

| # | 位置 | 现象 | 处理 |
|---|---|---|---|
| 1 | `engine/fireredasr2/` | **vendored 代码不完整**：`asr.py` 顶层无条件导入 `.models.*`，但仓库里**没有 `models/` 目录**。启用 `USE_FIRERED_ASR=1` 必然 `ModuleNotFoundError` | 见[第二部分第 12 节](#16-可选把-asr-升级为-fireredasr2)：先用默认 SenseVoice 跑通，再补齐上游 `models/` |
| 2 | `config.py` | LLM 写 `Qwen3-14B-AWQ` / 端口 `8100`，但启动脚本实际加载 `MiniCPM4.1-8B-GPTQ`，且 **`start_vllm.sh` 用的是 8000 端口** | 手动起服务，保证 `--served-model-name` 与 `VLLM_MODEL_NAME`、`--port` 与 `VLLM_BASE_URL` 逐字一致 |
| 3 | `config.py` | `LLM_GPU` / `ASR_GPU` / `TTS_GPU` / `RAG_GPU` 四个常量**没有任何代码读取**（全仓搜索：只出现在注释与 docstring 里） | 别去改它们；真正生效的是环境变量 `ASR_DEVICE` / `TTS_DEVICE` / `RAG_DEVICE` |
| 4 | `engine/tts.py:30` | `gpu_memory_utilization` 默认 **0.55**（`TTS_GPU_UTIL` 默认 0.45），单卡上会挤爆显存 | 单卡必须显式设小（如 0.18）；双卡分卡方案下 0.45 可用 |

### A.2 名字会骗你（读代码时的认知陷阱）

| # | 位置 | 现象 |
|---|---|---|
| 5 | `conversation_manager.py: _adaptive_endpoint_threshold()` | 函数名与 docstring 都说"短句走 FAST 档（8 帧≈256ms）"，但**代码里三个分支中有两个返回 SLOW 档（20 帧≈640ms）**，`ENDPOINT_FAST_CHUNKS` 是**从未被使用的死常量**。实际行为：短句等得**最久**。疑似有意的下限保护，也可能是笔误 |
| 6 | `engine/asr.py` | 名字像"ASR 抽象层/工厂"，**实际只是 SenseVoice 默认实现**（FunASR）。`USE_FIRERED_ASR` / `USE_MOONSHINE_ASR` 的分支逻辑在 **`ws_server.py` 的 `load_engines()`** 里 |
| 7 | `engine/rag.py` | `RAG_SCORE_THRESHOLD=0.35` 是按**余弦相似度**标定的；一旦启用 reranker，比较的是 **logit 量纲**的 rerank 分数，**两者不等价**，需要重新标定 |

### A.3 已知但不影响主链路的小问题

| # | 位置 | 现象 |
|---|---|---|
| 8 | `engine/fireredasr2/utils/wer.py` | 有若干 `SyntaxWarning: invalid escape sequence`。这是**上游原有**写法（我用编译对照验证过，非注释作业引入），不影响运行 |
| 9 | `test_rag.py` | 文件头第 2 行仍写着 `Usage: python ab_test_rag.py` —— 这个脚本是从 `ab_test_rag.py` 改名而来，docstring 没跟着改。实际用法是 `python test_rag.py` |
| 10 | `engine/captioner.py` | `HeuristicCaptioner` 的语速分支是**死代码**（`words_est/duration` 恒为 3.5，判断永不命中） |
| 11 | `engine/firered_vad.py` | `_silence_count` **只写不读**，判停实际由 FireRedVAD 内部的 `min_silence_frame` 决定 |
| 12 | `engine/fireredasr2/asr.py` | 代码里 `SHIFT = 0.06`（=60ms）与同处注释"shift 40ms"**不一致** |
| 13 | `engine/turn_detector.py` | `is_speech_active` 形参**恒传 `True`**，实际未生效 |
| 14 | `config.py` / `engine/tts.py` | 注释里仍把 **`tts_server.py`、`livekit_agent/run.py`、`ab_test_rag.py`** 列为消费者/调用方，但这三者已从仓库删除（见 `config.py:4`、`config.py:61`，以及 `engine/tts.py` 里 5 处 `tts_server`）。纯注释瑕疵，不影响运行 |

> 💡 **这份清单的用法**：改代码前先来这里查一眼，避免"修好一个假 bug"或"按错误的前提重构"。
> 每一条在源码里都有对应的 `# 注意：` 注释，搜这几个字就能找到现场。

---

## 附录 B · 常见任务怎么做

### B.1 换音色（语音克隆）
改 `config.py` 的两个常量（**必须成对**）：
```python
VOICE_PROMPT_WAV  = "data/你的参考音频.wav"     # 5~10 秒、干净、单人
VOICE_PROMPT_TEXT = "参考音频的逐字转写"          # 必须与音频内容完全一致，否则音色会飘
```
TTS 引擎在启动时用这两个做 `register_voice`。

### B.2 换话术（提示词工程）
改 `config.py` 的 `SYSTEM_PROMPT` / `SYSTEM_PROMPT_RAG` 与 `GREETING_TEXT` 等话术常量——
**不用改代码，改完重启即可**，这是最快的方式。
（本仓库原来的 `sft/` LoRA 微调工具链已移除；如需稳定定制话术，需要另行搭建微调流程。）

### B.3 换 ASR
1. 实现 `transcribe(audio) -> (text, ms)` 统一接口（参考 `engine/asr.py`）；
2. 在 `ws_server.py` 的 `load_engines()` 里加分支（**分支逻辑在这里，不在 engine/asr.py**）；
3. 加环境变量开关。

### B.4 调延迟
| 想优化 | 去哪里 |
|---|---|
| LLM 首 token | `engine/llm.py`（`max_tokens`、是否带历史/RAG、vLLM 参数） |
| TTS 首 chunk | `engine/tts.py`（切句粒度、chunk 大小） |
| ASR | `engine/asr_firered.py`；或开投机 ASR（`USE_MOONSHINE_ASR=1`）省约 117ms |
| 判停等待 | `engine/conversation_manager.py` 的三档阈值（**注意：调小会更灵敏，但也更容易打断用户**） |
| 首响体感 | `engine/filler.py`（语气词占位） |

### B.5 换 RAG 知识库
改 `config.KB_DATA_PATH`（默认 `data/sample_kb.json`），格式见该文件；重建索引后生效。 `RAG_SCORE_THRESHOLD` 控制"多像才算命中"。

### B.6 加一个新意图/系统话术
`engine/conversation_manager.py` 里已有 D1（问候）/D2（空闲）/D3（听不清）/D4（告别）四类体验层话术，照它们的模式加。

---


---

## 附录 C · 术语表

| 术语 | 含义 |
|---|---|
| **首响（First Response）** | 用户说完 → 听到第一个音节的时间。本项目预算 600ms |
| **VAD** | Voice Activity Detection，逐帧判断有没有人声 |
| **Endpointing / 判停** | 判断"用户说完了"，触发后续处理 |
| **Barge-in / 打断** | 用户中途插话，Agent 立即停止播报 |
| **Turn** | 一个对话轮次；用序列号区分，防止旧音频串进来 |
| **Filler / 语气词** | "嗯""好的"这类占位词，填补等待空档 |
| **投机 ASR** | 用户停顿时就抢先跑 ASR，省后续等待 |
| **声纹门控** | 用 ECAPA-TDNN 判断是不是本人说话 |
| **PTT** | Push-To-Talk，按住说话 |
| **TTFT** | Time To First Token，LLM 首个 token 的延迟 |
| **TTFA / ttfa_ms** | Time To First Audio，TTS 首个音频块的延迟 |
| **CER / WER** | 字错误率 / 词错误率，ASR 评测指标（本项目 2.89% CER） |
| **LoRA / GPTQ / AWQ** | 微调 / 量化方案 |
| **vendored** | 把上游开源代码整份放进自己仓库（`engine/fireredasr2/`） |
| **AudioWorklet** | 浏览器音频线程 API，能做到"已发未播帧立即停" |

---


---

## 附录 D · 本教程的依据与不确定性

**已核实（直接读代码确认）**：
- 依赖清单：从根目录与 `engine/` 下所有 `.py` 的 import 反推；
- **GPU 占用者只有 vLLM / ASR / TTS**：`engine/vad.py`（`map_location="cpu"`）、
  `engine/speaker_vad.py`（`load(device="cpu")`）、`engine/denoiser.py`
  （`providers=["CPUExecutionProvider"]`）、`engine/asr_moonshine.py`（CPU）逐个确认；
- **`LLM_GPU`/`ASR_GPU`/`TTS_GPU`/`RAG_GPU` 未被任何代码读取**：全仓搜索确认只出现在注释/docstring；
- 模型路径与文件名：`config.py:14-29`、`ws_server.py:58-61`、
  `engine/vad.py`（`silero_vad.jit`）、`engine/denoiser.py`（`model_1/2.onnx`）、
  `engine/fireredasr2/tokenizer/aed_tokenizer.py`（`dict.txt`/`train_bpe1000.model`）、
  `engine/fireredasr2/data/asr_feat.py`（`cmvn.ark`）；
- 端口与协议：`ws_server.py` 的路由定义（HTTP 403 / WS close 4003）；
- 显存默认值：`engine/tts.py:30`（0.55）、`ws_server.py`（`TTS_GPU_UTIL` 默认 0.45）；
- 坑 1（缺 `models/`）：用 AST 确认 `asr.py` 顶层无条件导入 `.models.*`，且该目录不存在；
- RAG 自动建索引、参考音频缺失时的降级：`ws_server.py` 的启动段；
- `data/sample_kb.json` 实为 59 条、键为 `id`/`question`/`answer`。

**未核实（受网络限制，需你按官方文档确认）**：
- 各模型在 ModelScope/HuggingFace 上的**当期**模型 ID 与下载命令；
- VoxCPM / `nanovllm_voxcpm` 的**当期**安装方式；
- vLLM 与 PyTorch 的**当期**推荐版本组合；
- FireRedASR2S 上游仓库的**实际目录层级**（第 12.1 节的 `cp` 源路径需按 `find` 结果调整）。

遇到与本文不符之处，**以代码和官方文档为准**。
