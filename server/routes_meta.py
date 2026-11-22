"""页面与元信息接口。

  GET /            前端页面（每次从磁盘读，改完刷新即生效）
  GET /api/info    能力清单（模型名 / 特性开关 / RAG 规模），run.sh status 用它探活
  GET /api/metrics 进程级会话计数 + 每个活动会话的延迟指标
"""

import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from config import (
    VERSION,
    EMBED_MODEL_DIR,
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    RAG_TOP_K,
    TTS_SAMPLE_RATE,
)
from engine.tts import KokoroTTS

from .bootstrap import REPO_ROOT
from .runtime import runtime

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面 `static/voice_agent.html`（每次请求都从磁盘读，改完刷新即生效）。

    页面本身只有 HTML 骨架，样式与脚本通过 `/static/...` 外链（见 app.py 的 StaticFiles 挂载）。
    """
    return HTMLResponse((REPO_ROOT / "static" / "voice_agent.html").read_text(encoding="utf-8"))


@router.get("/api/info")
async def api_info():
    """能力清单：版本、各组件模型名、特性开关与 RAG 规模（只读，不触发推理）。

    `models.tts.sample_rate`（24000）是前端解释下行裸 PCM 的依据。这里直接访问了
    `RAGEngine._documents` 私有属性，改 RAG 内部结构时要同步检查；run.sh status 用它探活。
    """
    rag = runtime.engine.get("rag")
    return {
        # 版本号唯一来源是 config.VERSION（日志 / ready 消息 / 这里都用它）
        "version": VERSION,
        "models": {
            "asr": {"name": "Parakeet-TDT-0.6B", "framework": "transformers"},
            "llm": {"name": LLM_MODEL_NAME, "endpoint": LLM_BASE_URL,
                    "streaming": True},
            # 采样率与音色都直接取 TTS 引擎的常量，避免两处各写一份。
            "tts": {"name": "Kokoro-82M", "framework": "kokoro",
                    "voice": KokoroTTS.VOICE, "sample_rate": TTS_SAMPLE_RATE},
            "embedding": {"name": Path(EMBED_MODEL_DIR).name},
        },
        "features": [ "sentence_streaming", "barge_in",
            "utterance_aggregation", "experience_layer_d1_d4", "crash_safe",
            "rag_confidence", "context_summary",
        ],
        # num_docs 为 0 有两种可能：没加载 RAG，或 KB 文件不存在（见 engines.load_engines）。
        "rag": {"num_docs": len(rag._documents) if rag else 0, "top_k": RAG_TOP_K},
    }


@router.get("/api/metrics")
async def api_metrics():
    """运维监控：进程级计数 + 每个活动会话的延迟/错误指标。

    每项是 `ConversationManager.metrics.summary()` 再补上 session_id、uptime_s、state；
    与 WebSocket 循环同处一个事件循环线程，遍历字典不会遇到并发修改。
    """
    sessions = []
    for sid, info in runtime.active_sessions.items():
        cm = info.get("cm")
        # cm 缺失（异常收尾中）时用空字典兜底，避免接口 500。
        m = cm.metrics.summary() if cm else {}
        m["session_id"] = sid
        m["uptime_s"] = round(time.time() - info["started"], 1)
        m["state"] = cm.state.value if cm else "unknown"
        sessions.append(m)
    return {
        "total_sessions": runtime.total_sessions,
        "active_sessions": len(runtime.active_sessions),
        "total_errors": runtime.total_errors,
        "sessions": sessions,
    }
