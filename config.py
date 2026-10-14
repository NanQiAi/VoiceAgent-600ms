"""全局配置中枢 —— 全项目唯一的“参数 + 话术”字典，不含任何逻辑。

消费者：ws_server.py（组装 exp_cfg）、engine/llm.py、engine/rag.py、engine/tts.py、
tts_server.py、livekit_agent/run.py。D1-D4 是产品需求编号（问候 / 空闲超时 /
错误恢复 / 告别识别），对应 engine/conversation_manager.py 的同名分节。

约定：环境变量优先于代码默认值；常量注释按【单位 / 取值 / 消费者 / 改动后果】写。
注意：进程启动时求值一次，改完必须重启才生效。
"""
import os

# ── 路径基准 ────────────────────────────────────────────────────────────────
# 用 abspath(__file__) 而不是 CWD：保证 systemd / nohup / 任意目录启动都能定位到同级的 models/ 与 data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 所有离线权重放 <repo>/models/<模型名>/；配合 HF_HUB_OFFLINE=1 强制离线，目录缺失会直接加载失败
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── 各引擎的本地权重目录 ────────────────────────────────────────────────────
# SenseVoiceSmall（FunASR）。仅未开 USE_FIRERED_ASR=1 时使用；FireRedASR2 的路径在 ws_server.py 单独定义
ASR_MODEL_DIR = os.path.join(MODEL_DIR, "SenseVoiceSmall")
# VoxCPM 1.5（nanovllm 后端），输出 44.1kHz 单声道，显存占比由 TTS_GPU_UTIL 控制。
# 它是全链路最慢的加载项，启动等待时间以它为准
TTS_MODEL_DIR = os.path.join(MODEL_DIR, "VoxCPM1.5")
# Qwen3-14B-AWQ（AWQ int4）。Python 侧不 import 本常量，它是“vLLM 该加载哪份权重”的部署约定：
# 必须与 VLLM_MODEL_NAME、vLLM 的 --served-model-name 一致，否则报 model not found。
# 启动脚本直接读它拼 --model，所以换模型改这里即可（同时要改 VLLM_MODEL_NAME）
LLM_MODEL_DIR = os.path.join(MODEL_DIR, "Qwen3-14B-AWQ")
# Silero VAD；也是 SpeakerAwareVAD（Silero + ECAPA-TDNN 声纹）的底座，统一 threshold=0.5
VAD_MODEL_DIR = os.path.join(MODEL_DIR, "snakers4_silero-vad")

# ── vLLM（OpenAI 兼容 HTTP 服务）地址与模型名 ───────────────────────────────
# 必须带 /v1 后缀（openai SDK 自己拼 /chat/completions）；端口要与 vLLM 的 --port 一致
VLLM_BASE_URL = "http://localhost:8100/v1"
# 必须等于 vLLM 的 --served-model-name。对不上时 LLM 调用 404，表现为“Agent 完全不开口”
VLLM_MODEL_NAME = "Qwen3-14B-AWQ"

# ── GPU 分配（0 起的物理卡号）────────────────────────────────────────────────
# ⚠️ Python 侧没有任何代码读取这三个常量（全仓搜索确认过），它们只是“部署约定 + 环境变量读取口”。
# 真正生效的是 ASR_DEVICE / TTS_DEVICE / RAG_DEVICE（cuda:N，N 为可见列表内序号）。
# 默认 1/2/2 表示 ASR 与 TTS 同卡；启动脚本用 GPU_LLM / GPU_VOICE 控制实际分卡
LLM_GPU = int(os.environ.get("LLM_GPU", "1"))
ASR_GPU = int(os.environ.get("ASR_GPU", "2"))
TTS_GPU = int(os.environ.get("TTS_GPU", "2"))

# ── 音频协议：输入端采样率与分帧（前端 AudioWorklet ↔ WebSocket）────────────
# 上行采样率 Hz。前端采集与 WS 二进制帧按 16k 单声道 int16，收帧后 /32768 归一化成 float32。
# ⚠️ TTS 输出是 44100Hz（engine/tts.py 的 SAMPLE_RATE），别与本值混用；
# 改这里必须同步改 static/ 前端 AudioContext 的采样率，否则音调/语速全错
SAMPLE_RATE = 16000
# 帧长 ms，是整条链路的时序基本单位：VAD 每帧判一次，所有静音超时都用“帧数”表达
# （如 ENDPOINT_DEFAULT_CHUNKS=13 帧 ≈ 416ms）。32ms = 16k 下的 512 点，与前端 AudioWorklet 对齐
CHUNK_MS = 32
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 512

# ── RAG：向量模型、知识库与检索参数 ─────────────────────────────────────────
# bge-small-zh-v1.5：约 24M 参数 / 512 维 / 3-5ms per query。engine/rag.py 用
# normalize_embeddings=True + FAISS IndexFlatIP —— 向量归一化后内积即余弦，
# 所以 RAG_SCORE_THRESHOLD 可按“余弦 0-1”理解。换模型必须重建索引（维度变了）
EMBED_MODEL_DIR = os.path.join(MODEL_DIR, "bge-small-zh-v1.5")
# 备选 bge-m3：约 568M 参数 / 1024 维 / 15-20ms，召回更好但慢一个量级。主流程未使用，
# 启用需同时改 EMBED_MODEL_DIR 并重建索引（ab_test_rag.py 里有对照实验）
EMBED_MODEL_LARGE_DIR = os.path.join(MODEL_DIR, "bge-m3")
# 可选精排 bge-reranker-v2-m3。engine/rag.py 只在 reranker_model_dir 非空且目录存在时加载；
# 主流程 ws_server 没传该参数 → 线上默认不精排，只有“向量召回 + 阈值过滤”
RERANKER_MODEL_DIR = os.path.join(MODEL_DIR, "bge-reranker-v2-m3")
# 知识库 JSON 数组，元素 {"id","question","answer","text"(可选)}。启动时 build_index，
# 也支持 POST /api/rag/reload 热重建。文件缺失不会崩，但检索恒为空、退化成纯人设闲聊
KB_DATA_PATH = os.path.join(BASE_DIR, "data", "sample_kb.json")
# FAISS 索引落盘前缀（产出 <path>.faiss / <path>.json）。主流程每次启动都在内存重建，本路径属预留
RAG_INDEX_PATH = os.path.join(BASE_DIR, "data", "rag_index")
# 最终注入 prompt 的知识条目数。取值 1-5：调大 → 上下文变长、挤占 --max-model-len、首 token 变慢；
# 调小 → 召回不足、答非所问
RAG_TOP_K = 3
# ⚠️ 同样未被 Python 读取（见上面的 *_GPU）。真正生效的是环境变量 RAG_DEVICE；
# 生产脚本设 RAG_DEVICE=cpu —— embedding 很小，放 CPU 省显存且不卡事件循环
RAG_GPU = int(os.environ.get("RAG_GPU", "2"))

# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------
# 保留的最近消息条数（2 条 = 1 轮）。超限时 engine/llm.py 从最旧处丢弃，并把被丢的成对轮次
# 压成一行摘要塞进 system prompt（最多最近 5 轮、每段 60 字），所以是“粗粒度记忆”而非硬遗忘。
# 调大 → 更完整但首 token 更慢（vLLM 的 --max-model-len 通常只有 2k-4k）；调小 → 更容易忘事
LLM_MAX_HISTORY = int(os.environ.get("LLM_MAX_HISTORY", "30"))

# ---------------------------------------------------------------------------
# API key (empty = auth disabled)
# ---------------------------------------------------------------------------
# 留空 = 关闭鉴权（默认，方便内网演示）。注意环境变量名是 VOICEAGENT_API_KEY（与常量名不同）。
# HTTP 接口接受 Authorization: Bearer 或 ?token=；WebSocket /ws/voice 只认 ?token=，失败以 close 4003 拒绝
API_KEY = os.environ.get("VOICEAGENT_API_KEY", "")

# ---------------------------------------------------------------------------
# RAG confidence threshold (cosine similarity, 0-1)
# ---------------------------------------------------------------------------
# 余弦相似度低于它就当没召回（<=0 表示不过滤）。0.35 是经验值：低于此基本是噪声，
# 宁可让 LLM 走“我问下同事”的兜底，也别拿错答案去答。
# ⚠️ 一旦启用 reranker，比较的是未归一化的 rerank logits，这个 0.35 就失去余弦含义、需重新标定
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.35"))

# ── 声音克隆（zero-shot voice cloning）参考素材 ────────────────────────────
# 约 5 秒的 wav（也支持 .mp3/.flac，按后缀判格式）。启动时文件存在就注册为默认音色，
# 缺失则静默跳过、用模型自带音色。⚠️ 换素材需重启（只在启动时注册一次）
VOICE_PROMPT_WAV = os.environ.get(
    "VOICE_PROMPT_WAV",
    # os.path.join(BASE_DIR, "data", "lipsync_real_clone_sample_5s.wav"),
    os.path.join(BASE_DIR, "data", "zh_vo_Huanglong_main_1_2EX_51_6.wav"),
)
# 上面那段音频的**逐字转写**：VoxCPM 用它做文本-音频对齐，写错会明显拉低相似度与韵律。
# 它不是提示词、不参与 LLM。留空则退化成“只有音频、无文本对齐”
VOICE_PROMPT_TEXT = os.environ.get(
    "VOICE_PROMPT_TEXT",
    "大家好，我是Voice Agent",
)

# ---------------------------------------------------------------------------
# D1: Greeting
# ---------------------------------------------------------------------------
# 接通后 Agent 主动说的第一句（TTS 朗读 + 文本下发）。以 asyncio 任务发出，故问候期间可被 Barge-in 打断。
# 置空 = 关闭主动问候。改文案会改变首句 TTS 时长，属于首响体验的一部分
GREETING_TEXT = os.environ.get(
    "GREETING_TEXT",
    "您好，我是Voice Agent。有什么可以帮您的呢？",
)

# ---------------------------------------------------------------------------
# D2: Idle timeout (seconds)
# ---------------------------------------------------------------------------
# 第一级空闲阈值（秒）。到点先播 IDLE_PROMPT_TEXT 询问。检测按约 1 秒节拍轮询，故有 ~1s 量化误差；
# 计时基准 _idle_since 在每次进入 IDLE 时刷新
IDLE_TIMEOUT_S = float(os.environ.get("IDLE_TIMEOUT_S", "15"))
# 第二级空闲阈值（秒）。比较的是同一个 _idle_since，即“距最后一次交互的总静默时长”，
# 所以 30/15 意味着提示后 15 秒告别。⚠️ 必须 > IDLE_TIMEOUT_S，否则第二个分支永不触发
IDLE_GOODBYE_S = float(os.environ.get("IDLE_GOODBYE_S", "30"))
# 第一级：轻问一句、给台阶，不能听起来像催促
IDLE_PROMPT_TEXT = os.environ.get("IDLE_PROMPT_TEXT", "您还在吗？有什么需要帮助的吗？")
# 第二级：播完即挂断，语气要收尾，不要留需要用户回答的问题
IDLE_GOODBYE_TEXT = os.environ.get("IDLE_GOODBYE_TEXT", "好的，如果后续有问题随时找我")

# ---------------------------------------------------------------------------
# D3: Error recovery
# ---------------------------------------------------------------------------
# ASR 整段失败（transcribe 抛异常）时立即播放的道歉话术。注意与“识别结果为空”是两条不同路径：
# 空结果前 N-1 次是静默回到 LISTENING，不打扰用户
ASR_EMPTY_RETRY_TEXT = "抱歉没太听清，您能再说一次吗？"
# 连续空结果达到 MAX_CONSECUTIVE_EMPTY_ASR 后的兜底：假设“环境太吵”，所以给换环境的建议
ASR_NOISY_SUGGEST_TEXT = "您那边好像有点吵，要不，您换个安静点环境？"
# 连续空/过短（<2 字）的容忍次数，达到即播上面的话术并清零；任何一次有效文本也清零
MAX_CONSECUTIVE_EMPTY_ASR = 3

# ---------------------------------------------------------------------------
# D4: Farewell detection
# ---------------------------------------------------------------------------
# 用**子串包含**匹配（kw in user_text），不是分词或意图模型：零额外延迟但有误判
# （如“我没了别的需求”会命中“没了”）。命中后先正常答完这一轮，TTS 播完再下发 session_end。
# 三个消费点：语音 pipeline / 累积文本 pipeline / 纯文本输入
FAREWELL_KEYWORDS = [
    "再见", "拜拜", "拜了", "没了", "就这些", "没有了",
    "没其他", "没别的", "谢谢再见", "好的再见", "挂了",
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
# 无知识库版人设，供无知识库场景或对比实验（主流程用的是下面的 SYSTEM_PROMPT_RAG）。
# ⚠️ 文案里的约束同时是性能优化：engine/llm.py 按 。！？；，,\n 切句后逐句送 TTS，
# “每次只说1-2句”直接决定首句出声时间（600ms 首响预算）；“不用编号/markdown”是避免 TTS 念出符号
SYSTEM_PROMPT = (
    "你是面壁智能的资深客服lucy。你在这家公司工作了三年，对产品非常熟悉。\n"
    "说话风格：\n"
    "- 像真人打电话一样自然，会用口语化表达（'嗯'、'对的'、'是这样的'）\n"
    "- 热情但不夸张，专业但不生硬\n"
    "- 每次只说1-2句话，像真人对话节奏，每句话的内容要尽可能简短精炼\n"
    "- 会主动追问客户需求，而不是被动等提问\n"
    "- 不使用编号、列举、markdown格式\n"
    "- 不说'作为AI'、'我是语言模型'之类的话\n"
    "- 只使用中文普通话\n"
)

# 带知识库版（ws_server 实际使用）：多了防幻觉指令，与 RAG_SCORE_THRESHOLD 一起构成
# “宁可少答、不可答错”。⚠️ 末尾 {context} 是占位符，由 engine/llm.py 用
# str.replace("{context}", rag_context) 替换（**不是** .format，改成 .format 会抛 KeyError）；
# rag_context 最多 800 字符，检索为空时传空串
SYSTEM_PROMPT_RAG = (
    "你是面壁智能的资深客服lucy。你在这家公司工作了三年，对产品非常熟悉。\n"
    "说话风格：\n"
    "- 像真人打电话一样自然，会用口语化表达（'嗯'、'对的'、'是这样的'）\n"
    "- 热情但不夸张，专业但不生硬\n"
    "- 每次只说1-2句话，像真人对话节奏，每句话的内容要尽可能简短精炼\n"
    "- 会主动追问客户需求，而不是被动等提问\n"
    "- 不使用编号、列举、markdown格式\n"
    "- 不说'作为AI'、'我是语言模型'之类的话\n"
    "- 只使用中文普通话\n"
    "- 优先根据知识库回答，知识库没有的坦诚说'这个我问下同事，再给您回复哈'而不是编造\n\n"
    "知识库：\n{context}"
)
