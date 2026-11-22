"""服务层：FastAPI 应用、路由、引擎装配与进程级共享状态。

分层位置：本包是**最上层**，可以 import `engine/` 里的引擎与状态机，反之不允许。
放在 `engine/` 之外的顶层，是为了让「引擎库」与「HTTP/WebSocket 服务」边界一眼可见。

模块地图
--------
settings.py     路径 / 演示模式 / 推理设备 / 日志（nest_asyncio 补丁也在这里）
runtime.py      全局引擎表 + 活动会话表 + 累计计数（进程级单例）
loader.py       `load_engines()`：启动期一次性装配全部引擎
routes_meta.py  `/`、`/api/info`、`/api/metrics`
routes_rag.py   `/api/rag/docs`、`/query`、`/reload`
routes_voice.py `/ws/voice`：连接生命周期 + 消息分发
app.py          FastAPI 实例 + 静态挂载 + 挂路由（只接线，无业务逻辑）

无鉴权：本项目定位为 demo，连上即会话。
"""

from . import bootstrap  # noqa: F401  必须最先导入（nest_asyncio 补丁）
from .app import app

__all__ = ["app"]
