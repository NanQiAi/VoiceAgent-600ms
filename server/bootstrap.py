"""服务进程的一次性初始化：nest_asyncio 补丁、日志、仓库路径。

**参数不在这里** —— 所有可调项（路径 / 设备 / 端口 / 阈值 / 话术）都在 `config.py`。
本模块只处理"启动时确定、之后不该被业务代码改"的运行时环境，
与 `config.py` 的分工是：`config` 回答"用什么值"，`bootstrap` 回答"进程怎么起来"。
"""

import logging
from pathlib import Path

import nest_asyncio

from config import BASE_DIR as _CONFIG_BASE_DIR

# ⚠️ 必须在任何引擎被导入之前执行（本模块是 server 包里最先被导入的）：
# 给全局事件循环策略打补丁，兼容内部自己调 asyncio.run() / run_until_complete()
# 的引擎封装，否则它们会在 uvicorn 的事件循环里抛 "loop already running"。
nest_asyncio.apply()

# 仓库根目录。唯一来源是 config.BASE_DIR，这里只做 str → Path 的转换。
REPO_ROOT = Path(_CONFIG_BASE_DIR)

# 注：本地 spaCy 模型（models/en_core_web_sm/）的 sys.path 注入在
# engine/tts.py 的 load() 里做——那里才是真正用到它的地方，也保证
# 不经服务进程、直接用 engine.tts 的脚本同样能跑。

# 全局日志格式：含 %(name)s，命名空间统一成 "Voice Agent.*"，
# 便于用 grep "Voice Agent" logs/voice_agent.log 过滤整条链路。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
log = logging.getLogger("Voice Agent Server")
