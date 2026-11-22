"""全局配置中枢：全项目唯一的"参数 + 话术"来源，不含任何逻辑。

约定
----
* 环境变量优先于代码默认值（`os.environ.get(NAME, default)`）。
* 仓库根目录的 `.env`（若存在）会被自动加载；**已 export 的同名变量仍然优先**。
* 常量注释按【单位 / 取值 / 消费者 / 改动后果】写。
* 进程启动时求值一次，改完必须重启才生效。
* **音频契约只在本文件定义一份**：`engine/dialog/constants.py` 与
  `engine/server/*` 都从这里导入，不要再各写一遍。

分层位置：`config.py` 是最底层，不 import 任何项目内模块。

D1~D4 是产品需求编号，与 `engine/dialog/` 里的同名分节对应：
    D1 问候 / D2 空闲超时 / D3 错误恢复 / D4 告别识别
"""

import os


def _load_dotenv(filename: str = ".env") -> None:
    """把仓库根目录 `.env` 里的 KEY=value 注入 `os.environ`。

    为什么自己写而不引入 python-dotenv：本项目 .env 格式极简单（KEY=value /
    # 注释 / 空行 / 可选引号），二十行够用，少一个依赖就少一处版本冲突。

    规则：**已存在的环境变量不覆盖** —— 命令行 `export` 永远优先于 .env 文件。
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return  # 没有 .env 是正常情况，不是错误

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        # 剥掉成对的首尾引号（允许值里有空格）
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
# 1. 路径基准
# ═══════════════════════════════════════════════════════════════════════
# 用 abspath(__file__) 而不是 CWD：保证 systemd / nohup / 任意目录启动
# 都能定位到同级的 models/ 与 data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 所有离线权重放 <repo>/models/<模型名>/；配合 HF_HUB_OFFLINE=1 强制离线
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ═══════════════════════════════════════════════════════════════════════
# 2. 各引擎权重目录与 TTS 参数
# ═══════════════════════════════════════════════════════════════════════
# SenseVoiceSmall（FunASR）—— 本项目唯一的 ASR 实现
ASR_MODEL_DIR = os.path.join(MODEL_DIR, "SenseVoiceSmall")
# Silero VAD —— 唯一的 VAD 模型（snakers4/silero-vad），固定 threshold=0.5
VAD_MODEL_DIR = os.path.join(MODEL_DIR, "snakers4_silero-vad")
# bge-m3：约 568M 参数 / **1024 维** / 15-20ms per query，多语言、召回更强。
# engine/rag.py 用 normalize_embeddings=True + FAISS IndexFlatIP，
# 向量归一化后内积即余弦，所以 RAG_SCORE_THRESHOLD 可按"余弦 0-1"理解。
# ⚠️ 比 bge-small-zh-v1.5（512 维 / 3-5ms）慢一个量级，但召回明显更好；
#    索引每次启动在内存重建，换模型无需手清缓存，但阈值可能需要重新标定。
EMBED_MODEL_DIR = os.path.join(MODEL_DIR, "bge-m3")

# Kokoro-82M TTS（Apache-2.0，82M 参数）。权重**放本地目录**，不走 HuggingFace 缓存：
#   huggingface-cli download hexgrad/Kokoro-82M --local-dir models/Kokoro-82M
# 目录内需要 config.json + kokoro-v1_0.pth + voices/<音色>.pt（加载逻辑见 engine/tts.py）。
KOKORO_MODEL_DIR = os.path.join(MODEL_DIR, "Kokoro-82M")
# 音色**不在这里配**：KokoroTTS.VOICE 已固定为 af_heart（标准美式英语女声），
# 本项目不做音色克隆、也不开放换音色，见 engine/tts.py 的类常量。
# 语速倍率。1.0 = 自然速（贴近雅思听力原速）；想让学员听清可降到 0.9
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))

# ═══════════════════════════════════════════════════════════════════════
# 3. 推理设备与服务开关
# ═══════════════════════════════════════════════════════════════════════
# 各引擎推理设备（"cuda:N" 用可见列表内的序号，或 "cpu"）。
# 生产可把 ASR/TTS/RAG 分散到不同卡；RAG 常设 cpu —— 检索只占几毫秒，
# 不值得占显存，也不会卡住事件循环。
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "cuda:0")

# DEMO_MODE=1 → PTT（按住说话）演示：上行音频先本地缓存，收到 ptt_end 才整段送
# pipeline；且不主动播问候语。默认 0 = 全双工。
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"

# 服务监听端口与是否由 uvicorn 自己终结 TLS。
# USE_SSL=0 时明文 HTTP，TLS 交给前置 Nginx（生产默认）。
PORT = int(os.environ.get("PORT", "3000"))
USE_SSL = os.environ.get("USE_SSL", "0") == "1"

# 项目版本号（**唯一来源**）：启动日志、/api/info、WS ready 消息、FastAPI 标题都读它。
# 以前 /api/info 与 WS ready 消息各写各的版本号、彼此对不上，别再抄字面量。
VERSION = "3.0"

# ═══════════════════════════════════════════════════════════════════════
# 4. LLM：DeepSeek 官方 API
# ═══════════════════════════════════════════════════════════════════════
# 本项目**只用 DeepSeek 云端**，不再兼容本地大模型（vLLM / Qwen3 等已整体移除）。
# 底层仍是 OpenAI 兼容协议 —— 换任何兼容服务只需改下面三个值。

# 官方文档给的 OpenAI 格式地址就是 https://api.deepseek.com（不带 /v1 也可以）
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# deepseek-flash 即 DeepSeek-V4.1-Flash-1017；旧的 deepseek-v4-flash 已下线
LLM_MODEL_NAME = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
# ⚠️ API Key 只能从环境变量 / .env 来，**不要写进代码**（本仓库是公开的）。
# 未设置时构造 LLM 客户端会立刻抛错（早失败，不等第一次对话才 401）。
# 获取地址 https://platform.deepseek.com
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 保留的最近消息条数（2 条 = 1 轮）。超限时 engine/llm.py 从最旧处丢弃，并把
# 被丢的成对轮次压成一行摘要塞进 system prompt（最多最近 5 轮、每段 60 字），
# 所以是"粗粒度记忆"而非硬遗忘。
# 调大 → 更完整但首 token 更慢；调小 → 更容易忘事。
LLM_MAX_HISTORY = int(os.environ.get("LLM_MAX_HISTORY", "30"))

# ═══════════════════════════════════════════════════════════════════════
# 5. 音频契约
# ═══════════════════════════════════════════════════════════════════════
# 上行采样率 Hz。前端采集与 WS 二进制帧按 16k 单声道 int16，
# 收帧后 /32768 归一化成 float32。
# ⚠️ 改这里必须同步改 static/js/config.js 的 SAMPLE_RATE_IN，
#    否则音调/语速全错（不会报错，只会听起来不对）。
SAMPLE_RATE = 16000
# 帧长 ms：VAD 每帧判一次，所有静音超时都用"帧数"表达
# （如 ENDPOINT_DEFAULT_CHUNKS=13 帧 ≈ 416ms）。
CHUNK_MS = 32
# 16000Hz × 32ms = 512 点，与前端 AudioWorklet 帧长严格对齐
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
# 下行采样率 Hz：Kokoro TTS 的原生输出，前端 AudioContext 必须按这个值建。
# ⚠️ 与上面的 SAMPLE_RATE（上行 16k）是两回事，别混用；
#    改这里要同步改 static/js/config.js 的 SAMPLE_RATE_OUT。
TTS_SAMPLE_RATE = 24000

# ═══════════════════════════════════════════════════════════════════════
# 6. RAG：知识库与检索参数
# ═══════════════════════════════════════════════════════════════════════
# 知识库 JSON 数组，元素 {"id","question","answer","text"(可选)}。
# 启动时 build_index，也支持 POST /api/rag/reload 热重建。
# 文件缺失不会崩，但检索恒为空、退化成纯人设闲聊。
KB_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_kb.json")
# 最终注入 prompt 的知识条目数。取值 1-5：
# 调大 → 上下文变长、挤占 --max-model-len、首 token 变慢；调小 → 召回不足。
RAG_TOP_K = 3
# 余弦相似度低于它就当没召回（<=0 表示不过滤）。0.35 是经验值：低于此基本是
# 噪声，宁可让 LLM 走"我问下同事"的兜底，也别拿错答案去答。
# ⚠️ 一旦启用 reranker，比较的是未归一化的 rerank logits，
#    这个 0.35 就失去余弦含义、需重新标定。
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.35"))

# ═══════════════════════════════════════════════════════════════════════
# 7. 体验层话术与阈值
# ═══════════════════════════════════════════════════════════════════════
# --- D1 问候 ---
# 接通后 Agent 主动说的第一句（TTS 朗读 + 文本下发）。以 asyncio 任务发出，
# 故问候期间可被 Barge-in 打断。置空 = 关闭主动问候。
GREETING_TEXT = os.environ.get(
    "GREETING_TEXT",
    "Hi, I'm your IELTS speaking coach. What would you like to practise today?",
)

# --- D2 空闲超时（秒）---
# 第一级：到点先播 IDLE_PROMPT_TEXT 询问。检测按约 1 秒节拍轮询，
# 故有 ~1s 量化误差；计时基准 _idle_since 在每次进入 IDLE 时刷新。
IDLE_TIMEOUT_S = float(os.environ.get("IDLE_TIMEOUT_S", "15"))
# 第二级：比较的是同一个 _idle_since，即"距最后一次交互的总静默时长"，
# 所以 30/15 意味着提示后 15 秒告别。
# ⚠️ 必须 > IDLE_TIMEOUT_S，否则第二个分支永不触发。
IDLE_GOODBYE_S = float(os.environ.get("IDLE_GOODBYE_S", "30"))
# 第一级：轻问一句、给台阶，不能听起来像催促
IDLE_PROMPT_TEXT = os.environ.get(
    "IDLE_PROMPT_TEXT", "Are you still there? Shall we keep going?"
)
# 第二级：播完即挂断，语气要收尾，不要留需要用户回答的问题
IDLE_GOODBYE_TEXT = os.environ.get(
    "IDLE_GOODBYE_TEXT",
    "Okay, talk to you next time. Good luck with your IELTS!",
)

# --- D3 错误恢复 ---
# ASR 整段失败（transcribe 抛异常）时立即播放的道歉话术。
# 注意与"识别结果为空"是两条不同路径：空结果前 N-1 次是静默回到 LISTENING。
ASR_EMPTY_RETRY_TEXT = "Sorry, I didn't quite catch that. Could you say it again?"
# pipeline / 播报链路整个崩掉时的兜底道歉（原先硬编码在 ConversationManager 里，
# 不可配置且是中文，与当前英文产品不一致）。
ERROR_APOLOGY_TEXT = "Sorry, something went wrong on my side. Could you say that again?"

# 连续空结果达到 MAX_CONSECUTIVE_EMPTY_ASR 后的兜底：
# 假设"环境太吵"，所以给换环境的建议。
ASR_NOISY_SUGGEST_TEXT = (
    "It sounds a bit noisy on your side. Could you move somewhere quieter?"
)
# 连续空/过短（<2 字）的容忍次数，达到即播上面的话术并清零；
# 任何一次有效文本也清零。
MAX_CONSECUTIVE_EMPTY_ASR = 3

# --- D4 告别识别 ---
# 用**子串包含**匹配（kw in user_text），不是分词或意图模型：
# 零额外延迟但有误判（如"我没了别的需求"会命中"没了"）。
# 命中后先正常答完这一轮，TTS 播完再下发 session_end。
# 三个消费点：语音 pipeline / 累积文本 pipeline / 纯文本输入。
FAREWELL_KEYWORDS = [
    "goodbye", "good bye", "bye", "bye bye", "see you", "that's all",
    "nothing else", "i'm done", "no more questions", "have to go",
]

# ═══════════════════════════════════════════════════════════════════════
# 8. System prompt
# ═══════════════════════════════════════════════════════════════════════
# 带知识库版（server/loader.py 实际使用）：多了防幻觉指令，
# 与 RAG_SCORE_THRESHOLD 一起构成"宁可少答、不可答错"。
# ⚠️ 末尾 {context} 是占位符，由 engine/llm.py 用
#    str.replace("{context}", rag_context) 替换（**不是** .format，
#    改成 .format 会抛 KeyError）；rag_context 最多 800 字符，
#    检索为空时传空串。
SYSTEM_PROMPT_RAG = (
    "You are Emma, a warm and experienced IELTS speaking coach.\n"
    "Speaking style:\n"
    "- Sound like a real person on a phone call, using natural fillers "
    "('well', 'sure', 'right')\n"
    "- Friendly but not over the top; professional but not stiff\n"
    "- Say only 1-2 short sentences per turn, like real conversation rhythm\n"
    "- Ask a short follow-up question instead of waiting passively\n"
    "- Never use bullet points, numbering, lists or markdown\n"
    "- Never say 'as an AI' or 'I am a language model'\n"
    "- Always speak clear, natural American English\n"
    "- Prefer the knowledge base when it applies; if it does not cover the "
    "question, say 'Let me check that and get back to you' instead of making "
    "things up\n\n"
    "Knowledge base:\n{context}"
)
