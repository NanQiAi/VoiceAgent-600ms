"""VoxLabs Voice Agent v2.9 — 全双工 WebSocket 服务入口（FastAPI + uvicorn）。

启动期用 load_engines() 把引擎装配进全局 `engine` 单例；每条连接新建有状态的 VAD/LLM 并装配
ConversationManager（状态机本体在 engine/conversation_manager.py），本文件只做协议转换。
音频约定：上行 16kHz、下行 44.1kHz，均单声道 int16 裸 PCM；消息协议见 `ws_voice` 的说明。
注意：只有 `python ws_server.py` 会调 load_engines()；用 `uvicorn ws_server:app` 直接加载模块时
`engine` 恒为空字典，各接口只能降级返回空数据。
"""
import os
import sys
import json
import time
import hmac
import asyncio
import logging
import numpy as np
import uvicorn
import nest_asyncio
# 模块级（import 时立即执行）给全局事件循环策略打补丁：兼容内部自己调 asyncio.run()/
# run_until_complete() 的引擎封装，否则它们会在 uvicorn 事件循环里抛"loop already running"。
nest_asyncio.apply()
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException, Request
# JSONResponse 当前未被使用（保留的历史导入，勿删）。
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.status import HTTP_403_FORBIDDEN

# 把仓库根目录插到 sys.path 最前：让 `from config import ...` 与 engine/* 里对 fireredasr2 等
# vendored 包的导入不依赖进程启动时的工作目录。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 模型目录、服务地址、话术常量与阈值统一取自 config.py；API_KEY 为空表示关闭鉴权。
from config import (
    ASR_MODEL_DIR, TTS_MODEL_DIR, VAD_MODEL_DIR,
    VLLM_BASE_URL, VLLM_MODEL_NAME,
    EMBED_MODEL_DIR, KB_DATA_PATH, RAG_TOP_K, RAG_SCORE_THRESHOLD,
    SYSTEM_PROMPT_RAG, VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT,
    GREETING_TEXT, IDLE_TIMEOUT_S, IDLE_GOODBYE_S,
    IDLE_PROMPT_TEXT, IDLE_GOODBYE_TEXT,
    ASR_EMPTY_RETRY_TEXT, ASR_NOISY_SUGGEST_TEXT, MAX_CONSECUTIVE_EMPTY_ASR,
    FAREWELL_KEYWORDS,
    LLM_MAX_HISTORY, API_KEY,
)
# 可选的 Omni captioner（语音副语言描述：情绪/语速/环境）服务地址；为空则退化为本地启发式实现。
CAPTIONER_URL = os.environ.get("CAPTIONER_URL", "")
CAPTIONER_MODEL = os.environ.get("CAPTIONER_MODEL", "MiniCPM-o-4.5-awq")
# DEMO_MODE=1 → PTT（按住说话）演示模式：上行音频先本地缓存，收到 ptt_end 才整段送 pipeline。
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"
# 选型开关：主 ASR 可切 FireRedASR2-AED（CER 2.89%），Moonshine 是 CPU/ONNX 投机式 ASR，Smart Turn v3 用韵律判停。
USE_FIRERED_VAD = os.environ.get("USE_FIRERED_VAD", "0") == "1"
USE_MOONSHINE_ASR = os.environ.get("USE_MOONSHINE_ASR", "0") == "1"
USE_FIRERED_ASR = os.environ.get("USE_FIRERED_ASR", "0") == "1"
USE_SMART_TURN = os.environ.get("USE_SMART_TURN", "0") == "1"

# 声纹 VAD / 降噪三态："1" 强开、"0" 强关、"auto"（默认）按本地模型目录是否就绪自动探测。
_speaker_vad_env = os.environ.get("USE_SPEAKER_VAD", "auto")
_denoise_env = os.environ.get("USE_DENOISE", "auto")
# 以下模型目录按"本文件所在目录/models/..."拼接，与当前工作目录无关（FIRERED_VAD_DIR 内含 Stream-VAD 子目录）。
DTLN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dtln")
SPEAKER_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "spkrec-ecapa-voxceleb")
FIRERED_VAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "FireRedVAD-stream")
FIRERED_ASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "FireRedASR2-AED")

def _model_dir_ready(d: str) -> bool:
    """粗判模型目录是否"下载完整"，供 auto 模式决定是否启用对应引擎。

    判定：目录存在、且非隐藏条目数 >= 2（HF/FunASR 目录通常至少含 config.json 与权重两个文件；
    空目录或只有一个占位文件说明下载中断，此时走降级分支比启动报错更稳）。
    """
    if not os.path.isdir(d):
        return False
    # 过滤 .cache/.lock 之类隐藏条目，避免只下了缓存就被误判为就绪。
    entries = [e for e in os.listdir(d) if not e.startswith(".")]
    return len(entries) >= 2

# 声纹 VAD / DTLN 降噪的最终生效值：显式 "1" 强制开；"auto" 时以模型目录是否就绪为准。
USE_SPEAKER_VAD = (
    _speaker_vad_env == "1" or
    (_speaker_vad_env == "auto" and _model_dir_ready(SPEAKER_MODEL_DIR))
)
USE_DENOISE = (
    _denoise_env == "1" or
    (_denoise_env == "auto" and _model_dir_ready(DTLN_MODEL_DIR))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger("voxlabs")

# 各引擎推理设备；生产脚本显式指定以把 ASR/TTS/RAG 分散到不同 GPU（RAG 常设 cpu：检索不值得占显存）。
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
RAG_DEVICE = os.environ.get("RAG_DEVICE", "cuda:0")

# 全局引擎注册表（跨会话共享的单例）：load_engines() 启动期写入，之后所有请求只读；键含
# vad_type / spec_asr / denoiser / turn_detector / asr / rag / llm_factory / tts / filler / captioner。
engine = {}


def load_engines():
    """启动期一次性装配全部引擎，把结果写进全局 `engine` 字典（阻塞数十秒、占 GPU 显存）。

    顺序（先轻后重）：VAD → 投机 ASR → 降噪 → Smart Turn → 主 ASR → RAG(建索引) → LLM 工厂 →
    TTS(加载+注册音色+warmup) → filler(预合成) → captioner。不捕获异常，模型缺失/显存不足会直接
    抛出终止进程（启动期快速失败优于带残缺引擎服务）；函数内的 import 都是延迟导入。
    """
    from engine.vad import SileroVAD
    from engine.asr import SenseVoiceASR
    from engine.llm import VLLMChat
    from engine.tts import VoxCPMTTS
    from engine.rag import RAGEngine
    from engine.filler import FillerEngine

    log.info("=== Loading engines (v2.0) ===")

    if USE_FIRERED_VAD:
        # FireRedVAD：F1 97.57%、误报率 2.69%，优于 Silero。
        from engine.firered_vad import FireRedVADWrapper
        log.info("Loading FireRedVAD (F1=97.57%%, false alarm 2.69%%)...")
        engine["vad_type"] = "firered"
        # 模板/预热实例，不会被任何会话直接使用：VAD 内部有流式状态，不能跨会话共享。
        engine["vad_template"] = FireRedVADWrapper(FIRERED_VAD_DIR, threshold=0.5)
        engine["vad_template"].load()
    elif USE_SPEAKER_VAD:
        from engine.speaker_vad import SpeakerAwareVAD
        log.info("Loading SpeakerAwareVAD (Silero + ECAPA-TDNN)...")
        engine["vad_type"] = "speaker"
        svad = SpeakerAwareVAD(VAD_MODEL_DIR, SPEAKER_MODEL_DIR, threshold=0.5)
        # 强制 CPU：声纹编码模型很小，放 CPU 既省显存也避免与 ASR/TTS 抢 GPU。
        svad.load(device="cpu")
        engine["vad_template"] = svad
    else:
        # 兜底 Silero；阈值 0.5 须与 conversation_manager 里 0.6 的判语音阈值配套。
        log.info("Loading VAD (Silero)...")
        engine["vad_type"] = "silero"
        engine["vad_template"] = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
        engine["vad_template"].load()

    # 投机 ASR：判停前静音到 ~160ms 就先跑一遍，判停后音频没变长就直接复用结果。
    if USE_MOONSHINE_ASR:
        from engine.asr_moonshine import MoonshineASR
        log.info("Loading Moonshine Tiny (speculative ASR, 27M, ONNX CPU)...")
        engine["spec_asr"] = MoonshineASR(model_name="moonshine/tiny")
        engine["spec_asr"].load()
    else:
        engine["spec_asr"] = None

    # DTLN 降噪：双阶段 ONNX，512 采样/块，CPU 上约 8ms/块，只在嘈杂环境有收益。
    if USE_DENOISE:
        from engine.denoiser import DTLNDenoiser
        log.info("Loading DTLN Denoiser (4MB ONNX, ~8ms/block)...")
        engine["denoiser"] = DTLNDenoiser(DTLN_MODEL_DIR)
        engine["denoiser"].load()
    else:
        engine["denoiser"] = None

    # Smart Turn v3 语义判停：按韵律判断"用户是否说完了"，减少抢话/等太久两种坏体验。
    if USE_SMART_TURN:
        from engine.turn_detector import SmartTurnDetector
        log.info("Loading Smart Turn Detector (Pipecat v3, 8M ONNX)...")
        engine["turn_detector"] = SmartTurnDetector()
        engine["turn_detector"].load()
    else:
        # None：状态机改用 _adaptive_endpoint_threshold() 的 256~640ms 经验判停。
        engine["turn_detector"] = None

    if USE_FIRERED_ASR:
        from engine.asr_firered import FireRedASR
        log.info("Loading ASR (FireRedASR2-AED, CER 2.89%%)...")
        # ASR 是共享单例（线程安全由内部保证），并发会话在这里排队。
        engine["asr"] = FireRedASR(FIRERED_ASR_DIR, device=ASR_DEVICE)
        engine["asr"].load()
    else:
        log.info("Loading ASR (SenseVoice)...")
        engine["asr"] = SenseVoiceASR(ASR_MODEL_DIR, device=ASR_DEVICE)
        engine["asr"].load()

    log.info("Loading RAG (bge-small-zh-v1.5, threshold=%.2f)...", RAG_SCORE_THRESHOLD)
    rag = RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K,
                    score_threshold=RAG_SCORE_THRESHOLD)
    rag.load()
    # KB 文件存在才建索引；否则 rag 能用但检索恒为空（对应降级日志与 /api/rag/* 的空结果）。
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        # build_index 做一次批量向量化（启动期成本，换运行时 4ms 级检索）。
        info = rag.build_index(docs)
        log.info("RAG index: %d docs, dim=%d", info["num_docs"], info["dim"])
    engine["rag"] = rag

    # LLM 只注册"工厂函数"而非实例：VLLMChat 持有对话历史 _history，共享会让多会话上下文串台。
    log.info("LLM client ready (vLLM @ %s, max_history=%d)", VLLM_BASE_URL, LLM_MAX_HISTORY)
    engine["llm_factory"] = lambda: VLLMChat(
        base_url=VLLM_BASE_URL, model=VLLM_MODEL_NAME,
        # 复用带 RAG 占位符 {context} 的 system prompt（检索结果由 VLLMChat 注入）。
        system_prompt=SYSTEM_PROMPT_RAG, max_history=LLM_MAX_HISTORY,
    )

    log.info("Loading TTS (VoxCPM nanovllm)...")
    # TTS 独占 GPU 的显存占比，默认 0.45；生产脚本会显式调大（如 0.55）。
    tts_util = float(os.environ.get("TTS_GPU_UTIL", "0.45"))
    tts = VoxCPMTTS(TTS_MODEL_DIR, device=TTS_DEVICE, gpu_memory_utilization=tts_util)
    tts.load()

    if os.path.exists(VOICE_PROMPT_WAV):
        log.info("Registering voice clone...")
        pid = tts.register_voice(VOICE_PROMPT_WAV, VOICE_PROMPT_TEXT)
        tts.set_default_voice(pid)
        log.info("Voice clone: %s", pid)

    # 预热一次极短合成，把模型/显存/算子的首次开销在启动期付掉，
    # 这样运行时第一句回答的 TTS TTFA 才是稳态值（~174ms 量级）。
    log.info("Warming up TTS...")
    tts.warmup()
    engine["tts"] = tts

    # filler：把"嗯，/好的，"等 5 条预合成为裸 PCM 缓存，填补 THINKING 阶段的听感空档。
    log.info("Pre-generating filler words...")
    filler = FillerEngine(tts)
    filler.pregenerate()
    engine["filler"] = filler

    # captioner：把副语言特征注入 LLM；有远程 Omni 就用它，否则用零依赖的启发式实现。
    if CAPTIONER_URL:
        from engine.captioner import OmniCaptioner
        log.info("Loading Captioner (OmniCaptioner @ %s)...", CAPTIONER_URL)
        engine["captioner"] = OmniCaptioner(
            base_url=CAPTIONER_URL, model=CAPTIONER_MODEL,
        ).load()
    else:
        from engine.captioner import HeuristicCaptioner
        log.info("Using HeuristicCaptioner (no CAPTIONER_URL set)")
        engine["captioner"] = HeuristicCaptioner()

    log.info("=== All engines loaded (v2.9) ===")
    log.info("  Speaker-aware VAD: %s (env=%s)", USE_SPEAKER_VAD, _speaker_vad_env)
    log.info("  Denoise (DTLN): %s (env=%s)", USE_DENOISE, _denoise_env)
    log.info("  API key auth: %s", "enabled" if API_KEY else "disabled")


app = FastAPI(title="VoxLabs Voice Agent v2.9")
BASE_DIR = Path(__file__).parent

# 进程级会话表 {session_id: {"started", "cm"}} 与累计计数；只在事件循环单线程里读写，故不加锁。
_active_sessions: dict[str, dict] = {}
_total_sessions = 0
_total_errors = 0


def _check_api_key(token: str | None) -> bool:
    """校验 token 是否是合法 API key：API_KEY 未配置时恒为 True，否则要求完全相等。

    纯函数，不发网络请求、不抛异常；用 `hmac.compare_digest` 做**常量时间**比较，避免通过响应
    耗时逐字节猜测密钥（时序侧信道）。
    """
    if not API_KEY:
        return True
    # 没带凭证一律拒绝（注意与上面"空 API_KEY"分支的顺序不能反）。
    if not token:
        return False
    return hmac.compare_digest(token, API_KEY)


async def verify_api_key(request: Request):
    """HTTP 接口鉴权依赖：接受 `Authorization: Bearer <key>` 或 `?token=<key>` 查询串，失败抛 403。"""
    if not API_KEY:
        return
    # 用 removeprefix 而不是 split，避免多空格解析歧义；无 Bearer 头时退化到查询串。
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    if not token:
        token = request.query_params.get("token", "")
    if not _check_api_key(token):
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端单页 `static/voice_agent.html`（每次请求都从磁盘读，改完刷新即生效）。"""
    return HTMLResponse((BASE_DIR / "static" / "voice_agent.html").read_text(encoding="utf-8"))


@app.get("/api/info", dependencies=[Depends(verify_api_key)])
async def api_info():
    """能力清单接口：返回版本、各组件模型名、特性开关与 RAG 规模（只读，不触发推理）。

    `models.tts.sample_rate`（44100）是前端解释下行裸 PCM 的依据。这里直接访问了
    `RAGEngine._documents` 私有属性，改 RAG 内部结构时要同步检查；start_prod.sh 用它探活。
    """
    rag = engine.get("rag")
    # ASR 的实际名称/框架按开关决定，避免前端把 FireRed 显示成 SenseVoice。
    asr_name = "FireRedASR2-AED" if USE_FIRERED_ASR else "SenseVoiceSmall"
    asr_fw = "FireRedASR" if USE_FIRERED_ASR else "FunASR"
    return {
        # 这里的 "2.9" 是引擎版本；ready 消息里另有一个协议版本号 "3.1"，含义不同。
        "version": "2.9",
        "models": {
            "asr": {"name": asr_name, "framework": asr_fw},
            "llm": {"name": VLLM_MODEL_NAME, "endpoint": VLLM_BASE_URL, "streaming": True},
            # 硬编码 44100：与 VoxCPMTTS.SAMPLE_RATE 一致，供客户端配置播放器。
            "tts": {"name": "VoxCPM 1.5", "framework": "nanovllm", "sample_rate": 44100},
            "embedding": {"name": "bge-small-zh-v1.5"},
        },
        "features": [
            "filler_words", "sentence_streaming", "barge_in",
            "utterance_aggregation", "experience_layer_d1_d4", "crash_safe",
            "api_key_auth", "rag_confidence", "context_summary",
        ],
        # num_docs 为 0 有两种可能：没加载 RAG，或 KB 文件不存在（见 load_engines）。
        "rag": {"num_docs": len(rag._documents) if rag else 0, "top_k": RAG_TOP_K},
    }


@app.get("/api/metrics", dependencies=[Depends(verify_api_key)])
async def api_metrics():
    """运维监控接口：汇总进程级会话计数与每个活动会话的延迟/错误指标。

    每项是 `ConversationManager.metrics.summary()` 再补上 session_id、uptime_s、state；
    与 ws_voice 同处一个事件循环线程，遍历字典不会遇到并发修改。
    """
    sessions = []
    for sid, info in _active_sessions.items():
        cm = info.get("cm")
        # cm 缺失（异常收尾中）时用空字典兜底，避免接口 500。
        m = cm.metrics.summary() if cm else {}
        m["session_id"] = sid
        m["uptime_s"] = round(time.time() - info["started"], 1)
        m["state"] = cm.state.value if cm else "unknown"
        sessions.append(m)
    return {
        "total_sessions": _total_sessions,
        "active_sessions": len(_active_sessions),
        "total_errors": _total_errors,
        "sessions": sessions,
    }


@app.get("/api/rag/docs", dependencies=[Depends(verify_api_key)])
async def rag_docs():
    """列出知识库文档摘要（id / question / answer 前 100 字，仅为展示），RAG 未加载时返回空列表。"""
    rag = engine.get("rag")
    if not rag:
        return {"docs": []}
    return {"docs": [{"id": d.get("id"), "question": d.get("question"),
                       "answer": d.get("answer", "")[:100]} for d in rag._documents]}


@app.get("/api/rag/query", dependencies=[Depends(verify_api_key)])
async def rag_query(q: str):
    """调试接口：拿一句话查 RAG，返回命中文档与各段耗时（results + embed_ms / search_ms /
    rerank_ms / total_ms），不经过 LLM/TTS；RAG 未加载时返回 `{"error": "RAG not loaded"}`。

    本函数是 async 却在事件循环里**同步**执行检索，会短暂阻塞其他会话的收发，只用于运维调试。
    """
    rag = engine.get("rag")
    if not rag:
        return {"error": "RAG not loaded"}
    return rag.query(q)


@app.post("/api/rag/reload", dependencies=[Depends(verify_api_key)])
async def rag_reload():
    """运维接口：重读 KB_DATA_PATH 的 JSON 并原地重建 FAISS 索引（热更新知识库）。

    会覆盖 `RAGEngine` 的 `_documents` / `_index` / `_doc_embeddings`，正在进行的检索可能读到旧
    引用（Python 赋值原子，不会读到半个索引）。RAG 未加载或 KB 不存在返回 `{"error": "No KB file"}`；
    KB 不是合法 JSON 时 `json.load` 抛 JSONDecodeError → 500。
    """
    rag = engine.get("rag")
    if not rag or not os.path.exists(KB_DATA_PATH):
        return {"error": "No KB file"}
    with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {"status": "ok", **rag.build_index(docs)}


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket, token: str = Query(default="")):
    """全双工语音主通道：一条 WebSocket 连接 = 一个会话（含独立的 VAD/LLM/状态机）。

    鉴权(?token=) → accept → 建 VAD/LLM → 装配 ConversationManager（send_fn 为唯一下行出口）
    → 发 ready → 后台问候语 → 收发循环；finally 汇总指标、注销会话、cm.shutdown() 取消其后台任务。
    上行：二进制帧 = 16kHz 单声道 int16 裸 PCM，512 采样/帧；文本 ptt_start / ptt_end /
    text_input / manual_barge_in / enroll(base64 int16 PCM) / unenroll / reset。
    下行：二进制帧 = 44.1kHz 单声道 int16 裸 PCM；文本 ready / state / asr / rag / filler /
    llm_sentence / llm / audio_start / metrics / caption / system_message / barge_in /
    session_end / error / ptt_ack / enroll_ack / unenroll_ack / reset_ack / shield_status。
    契约：turn 是轮次序列号（audio_start 让前端丢弃旧轮次的迟到音频）；asr 的 latency_ms=0
    表示复用了投机 ASR 或聚合文本；文本帧一律 ensure_ascii=False。
    """
    # 鉴权在 accept 之前完成：不合法直接 close(4003)（自定义关闭码，供客户端区分密钥错误与断网），
    # 不建会话、不耗时加载 VAD。
    if not _check_api_key(token):
        await ws.close(code=4003, reason="Invalid API key")
        return
    await ws.accept()
    session_id = f"s-{int(time.time() * 1000)}"
    slog = logging.LoggerAdapter(log, {"session": session_id})
    slog.info("Connected")

    from engine.conversation_manager import ConversationManager

    # 每条连接新建 VAD，不复用 engine["vad_template"]：VAD 内部保存流式状态（_speech_active /
    # _silence_count / 语音环形缓冲），共享会互相污染；重新 load() 只付很轻的成本。
    vad_type = engine.get("vad_type", "silero")
    if vad_type == "firered":
        from engine.firered_vad import FireRedVADWrapper
        vad = FireRedVADWrapper(FIRERED_VAD_DIR, threshold=0.5)
        vad.load()
    elif vad_type == "speaker":
        from engine.speaker_vad import SpeakerAwareVAD
        vad = SpeakerAwareVAD(VAD_MODEL_DIR, SPEAKER_MODEL_DIR, threshold=0.5)
        vad.load(device="cpu")
    else:
        from engine.vad import SileroVAD
        vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
        vad.load()

    # 每条连接独立一个 LLM 客户端：VLLMChat 持有 _history，共享会让两个用户的上下文串台。
    llm = engine["llm_factory"]()

    # 死连接标记：send_fn 发送失败就置位，避免对端已断开时反复抛异常刷日志。
    _ws_dead = False

    async def send_fn(data):
        """ConversationManager 的唯一下行出口：把一条消息翻译成 WebSocket 帧发出。

        含 "_audio" 键 → 取 `data["_audio"]`（bytes，裸 PCM int16）作为二进制帧发送（该键本身不会
        出现在网络上）；否则整条 dict 序列化成 JSON 文本帧（ensure_ascii=False，中文不被转义）。
        发送失败只记一次 warning 并置外层 `_ws_dead`：异常被有意吞掉，下行失败不中断状态机。
        """
        nonlocal _ws_dead
        if _ws_dead:
            return
        try:
            # 二进制音频帧优先：裸 PCM 不走 JSON（避免 base64 膨胀与编解码开销）。
            if "_audio" in data:
                await ws.send_bytes(data["_audio"])
            else:
                await ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            if not _ws_dead:
                _ws_dead = True
                slog.warning("WebSocket send failed (marking dead): %s", e)

    # D1-D4 体验层配置：把 config.py 的常量打包传入，使状态机不直接依赖 config（便于单测注入）。
    exp_cfg = {
        "greeting_text": GREETING_TEXT,
        # D2 空闲：15s 未说话提醒一次，30s 仍未说话则告别并结束会话（单位：秒）。
        "idle_timeout_s": IDLE_TIMEOUT_S,
        "idle_goodbye_s": IDLE_GOODBYE_S,
        "idle_prompt_text": IDLE_PROMPT_TEXT,
        "idle_goodbye_text": IDLE_GOODBYE_TEXT,
        "asr_empty_retry_text": ASR_EMPTY_RETRY_TEXT,
        "asr_noisy_suggest_text": ASR_NOISY_SUGGEST_TEXT,
        "max_consecutive_empty_asr": MAX_CONSECUTIVE_EMPTY_ASR,
        "farewell_keywords": FAREWELL_KEYWORDS,
    }

    # 装配状态机：asr/rag/tts/filler 用全局共享实例，vad/llm 用本会话独有实例。
    cm = ConversationManager(
        asr=engine["asr"], llm=llm, tts=engine["tts"],
        rag=engine["rag"], vad=vad, filler=engine["filler"],
        send_fn=send_fn,
        captioner=engine.get("captioner"),
        spec_asr=engine.get("spec_asr"),
        turn_detector=engine.get("turn_detector"),
        denoiser=engine.get("denoiser"),
        experience_config=exp_cfg,
    )

    global _total_sessions, _total_errors
    _total_sessions += 1
    _active_sessions[session_id] = {"started": time.time(), "cm": cm}

    mode = "demo" if DEMO_MODE else "duplex"
    await send_fn({"type": "ready", "session_id": session_id, "version": "3.1", "mode": mode})

    # D1 问候语用 _track_task 起后台任务而不是 await：收发循环立刻开始，问候播放期间用户就能
    # 说话并触发打断；DEMO_MODE 下不主动问候。
    if GREETING_TEXT and not DEMO_MODE:
        cm._track_task(cm.start_greeting(), name="greeting")

    ptt_active = False
    ptt_buffer: list[np.ndarray] = []

    try:
        while True:
            # 对端已死（TCP 未及时报错）→ 去走 finally，不再空转收包。
            if _ws_dead:
                slog.info("send_fn dead, exiting loop")
                break

            # 阻塞等下一条 ASGI 消息；客户端断开时 Starlette 抛 WebSocketDisconnect（下方捕获）。
            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                raw = data["bytes"]
                # int16 → float32 归一化：除以 32768（2^15）而非 32767，保证最小值 -32768 映射到 -1.0。
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if DEMO_MODE:
                    if ptt_active:
                        ptt_buffer.append(samples)
                else:
                    await cm.feed_audio(samples)

            # 文本帧 = JSON 控制消息（未知 type 静默忽略，向下兼容前端先行升级）。
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                msg_type = msg.get("type")

                if msg_type == "ptt_start":
                    ptt_active = True
                    ptt_buffer = []
                    slog.info("PTT start")
                    await send_fn({"type": "ptt_ack", "recording": True})

                elif msg_type == "ptt_end":
                    ptt_active = False
                    if ptt_buffer:
                        # 多帧拼成整段并立即清空缓存（避免下一轮重复送 ASR）；时长按 16000Hz 折算。
                        audio = np.concatenate(ptt_buffer)
                        ptt_buffer = []
                        dur = len(audio) / 16000
                        slog.info("PTT end (%.1fs, %d samples)", dur, len(audio))
                        await send_fn({"type": "ptt_ack", "recording": False,
                                       "duration_s": round(dur, 2)})
                        await cm.process_ptt(audio)
                    else:
                        slog.info("PTT end (empty)")
                        await send_fn({"type": "ptt_ack", "recording": False,
                                       "duration_s": 0})

                elif msg_type == "text_input":
                    await cm.handle_text_input(msg["text"])

                elif msg_type == "manual_barge_in":
                    await cm.handle_barge_in()

                elif msg_type == "enroll":
                    # 声纹登记：前端录一段话以 base64(int16) 传来；base64 只在本分支用到。
                    import base64
                    audio_b64 = msg.get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        # 状态机侧校验最小长度后登记，返回 success/samples/duration_s，或 success=False。
                        result = cm.enroll_speaker(audio_np)
                        await send_fn({"type": "enroll_ack", **result})
                        # speaker_vad=本会话 VAD 是否具备声纹能力（否则登记必失败）。
                        await send_fn({"type": "shield_status",
                                       "speaker_gate": cm.speaker_enrolled,
                                       "speaker_vad": vad_type == "speaker"})
                    else:
                        await send_fn({"type": "enroll_ack", "success": False,
                                       "reason": "no_audio"})

                elif msg_type == "unenroll":
                    cm.unenroll_speaker()
                    await send_fn({"type": "unenroll_ack", "success": True})
                    await send_fn({"type": "shield_status",
                                   "speaker_gate": False,
                                   "speaker_vad": vad_type == "speaker"})

                elif msg_type == "reset":
                    cm.reset()
                    ptt_active = False
                    ptt_buffer = []
                    await send_fn({"type": "reset_ack"})

    # 客户端主动断开属正常收尾：只记录已进行到第几轮，不报错。
    except WebSocketDisconnect:
        slog.info("Disconnected (turn %d)", cm._turn)
    except Exception as e:
        # 其它异常（如 msg["text"] 缺字段的 KeyError、JSON 解析失败）记完整堆栈后由 finally 收尾。
        slog.error("Error: %s", e, exc_info=True)
    finally:
        # 必须走这里释放资源，否则会泄漏 asyncio 任务（pipeline / TTS 播放 / 投机 ASR / 空闲计时器）。
        metrics = cm.metrics.summary()
        _total_errors += metrics.get("errors", 0)
        slog.info("Session metrics: %s", metrics)
        _active_sessions.pop(session_id, None)
        # 取消该会话所有后台任务并复位状态机（shutdown 内部会再调 reset()）。
        cm.shutdown()


if __name__ == "__main__":
    # 只有直接运行本文件才装配引擎并起服务；被 import 时 engine 为空，各接口降级返回空数据。
    load_engines()

    # 端口默认 3000（与 start_prod.sh / 前端同源访问一致），可用 PORT 覆盖。
    port = int(os.environ.get("PORT", "3000"))
    # USE_SSL=1 → uvicorn 自己终结 TLS；否则默认由前置 Nginx 终结。
    use_ssl = os.environ.get("USE_SSL", "0") == "1"

    if use_ssl:
        # 证书固定为仓库内 certs/ 下的自签 cert.pem / key.pem。
        cert_dir = BASE_DIR / "certs"
        log.info("Starting on https://0.0.0.0:%d (self-signed SSL)", port)
        uvicorn.run(app, host="0.0.0.0", port=port,
                    ssl_certfile=str(cert_dir / "cert.pem"),
                    ssl_keyfile=str(cert_dir / "key.pem"),
                    log_level="info")
    else:
        # 生产路径：明文 HTTP，TLS 由前置 Nginx 负责。
        log.info("Starting on http://0.0.0.0:%d (SSL via Nginx reverse proxy)", port)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
