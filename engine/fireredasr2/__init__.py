# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""FireRedASR2 包入口：只做导入与环境变量，不含任何算法逻辑（vendored 上游代码）。

本项目走 ``from fireredasr2.asr import ...`` 子模块导入，不经过本文件；``__all__`` 只影响
``from fireredasr2 import ...`` 的写法。
"""

import os
import sys
import warnings

# 进程级副作用：忽略所有 warning（上游为了让推理日志干净）；注意是全局生效，会吞掉其它库的告警
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=ALL, 1=INFO, 2=WARNING, 3=ERROR
# 环境变量必须在任何 TensorFlow import 之前设置：
# TF_CPP_MIN_LOG_LEVEL=3 只输出 TF 的 ERROR 级日志；TF_ENABLE_ONEDNN_OPTS=0 关闭 oneDNN 优化
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

__version__ = "0.0.1"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 兼容两种安装形态：① 上游包作为 site-packages 安装（包名带尾部 s：fireredasr2s）；
# ② 失败（ImportError）则退化为本仓库内的相对导入
try:
    from fireredasr2s.fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
except ImportError:
    # 把本包目录插到 sys.path 最前：包内 speech2text.py 用的是绝对导入
    # ``from fireredasr2.asr import ...``，只有 fireredasr2 能作为顶层包被找到才成立。
    # 副作用：永久修改进程 sys.path，且本地 vendored 副本会遮蔽 site-packages 里的同名包。
    if _CURRENT_DIR not in sys.path:
        sys.path.insert(0, _CURRENT_DIR)
    from .asr import FireRedAsr2, FireRedAsr2Config


# API
__all__ = [
    "__version__",
    "FireRedAsr2",
    "FireRedAsr2Config",
]
