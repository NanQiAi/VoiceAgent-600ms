"""启动期引擎装配：把全部引擎一次性加载进全局注册表。

只由进程入口（`python ws_server.py`）调用一次。装配完所有请求共享同一批实例。
"""

import json
import os
from pathlib import Path

from config import (
    VERSION,
    ASR_DEVICE,
    ASR_MODEL_DIR,
    EMBED_MODEL_DIR,
    KB_DATA_PATH,
    KOKORO_MODEL_DIR,
    KOKORO_SPEED,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_HISTORY,
    LLM_MODEL_NAME,
    RAG_DEVICE,
    RAG_SCORE_THRESHOLD,
    RAG_TOP_K,
    SYSTEM_PROMPT_RAG,
    TTS_DEVICE,
    VAD_MODEL_DIR,
)

from .bootstrap import log
from .runtime import runtime


def load_engines():
    """一次性装配全部引擎，写进 `runtime.engine`（阻塞数十秒、占 GPU 显存）。

    顺序（先轻后重）：VAD → 投机 ASR → 主 ASR → RAG(建索引) → LLM 工厂 →
    TTS(加载 + warmup)。不捕获异常：模型缺失/显存不足会直接抛出终止进程
    （启动期快速失败优于"带残缺引擎服务"）。函数内的 import 都是延迟导入。
    """
    from engine.vad import SileroVAD
    from engine.asr import ParakeetASR
    from engine.llm import LLMChat
    from engine.tts import KokoroTTS
    from engine.rag import RAGEngine

    reg = runtime.engine

    log.info("=== Loading engines (v%s) ===", VERSION)

    # ── VAD：只有 Silero 一种实现。这里的 threshold=0.5 只影响 VAD 内部那套
    #    无人消费的端点事件；真正生效的判停门限 0.6 硬编码在 engine/dialog/manager.py ──
    log.info("Loading VAD (Silero)...")
    reg["vad_type"] = "silero"
    # 模板/预热实例，不会被任何会话直接使用：VAD 内部有流式状态，不能跨会话共享。
    reg["vad_template"] = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    reg["vad_template"].load()

    # ── 投机 ASR：判停前静音到 ~160ms 就先跑一遍，判停后音频没变长就直接复用结果 ──
    # 只保留主 ASR 一条路径（无可选投机引擎），由 engine/dialog/manager.py 内部直接复用 asr。
    reg["spec_asr"] = None

    # ── ASR：只有 Parakeet-TDT 一种实现 ──
    log.info("Loading ASR (Parakeet-TDT)...")
    reg["asr"] = ParakeetASR(ASR_MODEL_DIR, device=ASR_DEVICE)
    reg["asr"].load()

    # ── RAG：向量化 + FAISS 索引 ──
    # 模型名从权重目录名推出，避免换模型后日志说谎
    log.info("Loading RAG (%s, threshold=%.2f)...",
             Path(EMBED_MODEL_DIR).name, RAG_SCORE_THRESHOLD)
    rag = RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K,
                    score_threshold=RAG_SCORE_THRESHOLD)
    rag.load()
    # KB 文件存在才建索引；否则 rag 能用但检索恒为空（对应降级日志与 /api/rag/* 的空结果）。
    if os.path.exists(KB_DATA_PATH):
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        # build_index 做一次批量向量化（59 条约 2s 启动成本），换运行时单次检索约 40ms
        # （其中 embedding 占绝大部分，FAISS search 仅 0.1ms）。
        info = rag.build_index(docs)
        log.info("RAG index: %d docs, dim=%d", info["num_docs"], info["dim"])
    reg["rag"] = rag

    # ── LLM：只注册"工厂函数"而非实例 ──
    # LLMChat 持有对话历史 _history，共享会让多会话上下文串台。
    # 接入点固定为 DeepSeek 云端（config.py §4）；只注册工厂，实例由每个连接新建。
    log.info("LLM client ready (%s @ %s, model=%s, max_history=%d)",
             "deepseek", LLM_BASE_URL, LLM_MODEL_NAME, LLM_MAX_HISTORY)
    reg["llm_factory"] = lambda: LLMChat(
        base_url=LLM_BASE_URL, model=LLM_MODEL_NAME,
        # 复用带 RAG 占位符 {context} 的 system prompt（检索结果由 LLMChat 注入）。
        system_prompt=SYSTEM_PROMPT_RAG, max_history=LLM_MAX_HISTORY,
        # DeepSeek 要真实 key；本地 vLLM 用占位（config 里已按后端选好）
        api_key=LLM_API_KEY,
    )

    # ── TTS：Kokoro-82M。音色固定在 KokoroTTS.VOICE（af_heart，标准美式英语女声），
    #     不需要显存预分配，也不支持/不提供音色克隆 ──
    log.info("Loading TTS (Kokoro-82M, voice=%s)...", KokoroTTS.VOICE)
    tts = KokoroTTS(
        model_dir=KOKORO_MODEL_DIR, speed=KOKORO_SPEED,
        device=TTS_DEVICE,
    )
    tts.load()

    # 预热一次极短合成，把模型/算子的首次开销在启动期付掉，
    # 这样运行时第一句回答的 TTS TTFA 才是稳态值。
    log.info("Warming up TTS...")
    tts.warmup()
    reg["tts"] = tts

    log.info("=== All engines loaded (v%s) ===", VERSION)
    log.info("  VAD: Silero（固定）")
