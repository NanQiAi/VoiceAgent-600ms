""" Voice Agent v3.0 — 进程入口。

本文件只做三件事：把仓库根目录放进 `sys.path`、装配引擎、起 uvicorn。

后端实现全部在 `engine/` 包里，按三层组织（详见 `engine/__init__.py`）：

    engine/asr.py llm.py rag.py tts.py vad.py filler.py   ① 可替换引擎
    engine/dialog/                                        ② 对话状态机
    server/                                               ③ 服务层（独立顶层包）

音频约定：上行 16kHz、下行 24kHz（Kokoro），均单声道 int16 裸 PCM。
参数与话术全部来自 `config.py`；无鉴权（连上即会话）。

⚠️ 只有 `python ws_server.py` 会调 `load_engines()`。用 `uvicorn ws_server:app`
直接加载模块时引擎注册表恒为空，各接口只能降级返回空数据。
"""

import os
import sys

# 把仓库根目录插到 sys.path 最前：让 `from config import ...` 与 engine/* 的导入
# 不依赖进程启动时的工作目录。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# nest_asyncio 的补丁在 server/bootstrap.py 的模块级执行（必须早于任何引擎导入），
# 这里只是通过导入链间接触发它。
import uvicorn

import config
from server.app import app  # noqa: F401  供 `uvicorn ws_server:app` 使用
from server.loader import load_engines
from server.bootstrap import REPO_ROOT, log


if __name__ == "__main__":
    # 只有直接运行本文件才装配引擎并起服务；被 import 时注册表为空，各接口降级返回空数据。
    load_engines()

    # 端口默认 3000（与 run.sh / 前端同源访问一致），可用 PORT 覆盖。
    port = config.PORT
    # USE_SSL=1 → uvicorn 自己终结 TLS；否则默认由前置 Nginx 终结。
    use_ssl = config.USE_SSL

    if use_ssl:
        # 证书固定为仓库内 certs/ 下的自签 cert.pem / key.pem。
        cert_dir = REPO_ROOT / "certs"
        log.info("Starting on https://0.0.0.0:%d (self-signed SSL)", port)
        uvicorn.run(app, host="0.0.0.0", port=port,
                    ssl_certfile=str(cert_dir / "cert.pem"),
                    ssl_keyfile=str(cert_dir / "key.pem"),
                    log_level="info")
    else:
        # 生产路径：明文 HTTP，TLS 由前置 Nginx 负责。
        log.info("Starting on http://0.0.0.0:%d (SSL via Nginx reverse proxy)", port)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
