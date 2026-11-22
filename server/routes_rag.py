"""RAG 调试接口（运维用，不走 LLM/TTS）。

  GET  /api/rag/docs    列出知识库文档摘要
  GET  /api/rag/query   用一句话查检索，返回命中与各段耗时
  POST /api/rag/reload  重读 KB 文件并原地重建索引（热更新知识库）
"""

import asyncio
import json
import os

from fastapi import APIRouter

from config import KB_DATA_PATH
from .runtime import runtime

router = APIRouter()


@router.get("/api/rag/docs")
async def rag_docs():
    """列出知识库文档摘要（id / question / answer 前 100 字，仅为展示）；RAG 未加载时返回空列表。"""
    rag = runtime.engine.get("rag")
    if not rag:
        return {"docs": []}
    return {"docs": [{"id": d.get("id"), "question": d.get("question"),
                      "answer": d.get("answer", "")[:100]} for d in rag._documents]}


@router.get("/api/rag/query")
async def rag_query(q: str):
    """用一句话查 RAG，返回命中文档与各段耗时（results + embed_ms / search_ms /
    rerank_ms / total_ms）；RAG 未加载时返回 `{"error": "RAG not loaded"}`。

    检索要跑一次 embedding（十几毫秒到几十毫秒），必须丢线程池：
    在事件循环里同步执行会卡住所有 WebSocket 会话的音频收发。
    """
    rag = runtime.engine.get("rag")
    if not rag:
        return {"error": "RAG not loaded"}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, rag.query, q)


@router.post("/api/rag/reload")
async def rag_reload():
    """重读 KB_DATA_PATH 的 JSON 并原地重建 FAISS 索引（热更新知识库）。

    会覆盖 `RAGEngine` 的 `_documents` / `_index` / `_doc_embeddings`，正在进行的检索可能读到旧
    引用（Python 赋值原子，不会读到半个索引）。RAG 未加载或 KB 不存在返回 `{"error": "No KB file"}`；
    KB 不是合法 JSON 时 `json.load` 抛 JSONDecodeError → 500。
    """
    rag = runtime.engine.get("rag")
    if not rag or not os.path.exists(KB_DATA_PATH):
        return {"error": "No KB file"}

    # 读文件 + 重新编码全库（59 条约 1~2s）都在线程里做：留在事件循环会
    # 把正在进行的语音会话整段卡住，用户会听到明显停顿。
    def _reload() -> dict:
        with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return rag.build_index(docs)

    loop = asyncio.get_running_loop()
    return {"status": "ok", **await loop.run_in_executor(None, _reload)}
