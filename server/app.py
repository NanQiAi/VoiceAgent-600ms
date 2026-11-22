"""FastAPI 应用装配：建实例、挂静态资源、挂三组路由。

分层意图：本文件只做"接线"，不含任何业务逻辑 —— 想加接口就去对应的 routes_*.py。
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import VERSION

from .bootstrap import REPO_ROOT
from .routes_meta import router as meta_router
from .routes_rag import router as rag_router
from .routes_voice import router as voice_router

app = FastAPI(title=f"Voice Agent v{VERSION}")

# 前端静态资源（CSS + 一组 ES module）。页面本身由 routes_meta.index() 从磁盘读取，
# 改完刷新即生效；这里挂载子目录是为了让 <link>/<script src> 能拿到文件。
app.mount("/static", StaticFiles(directory=str(REPO_ROOT / "static")), name="static")

app.include_router(meta_router)    # /            /api/info   /api/metrics
app.include_router(rag_router)     # /api/rag/docs  /query  /reload
app.include_router(voice_router)   # /ws/voice
