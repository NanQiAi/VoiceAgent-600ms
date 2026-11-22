"""对话状态机：一条 WebSocket 连接 = 一个 `ConversationManager` 实例。

`manager.py` 是门面（音频入口 / 状态分支 / 打断 / 生命周期），其余模块按职责拆开，
由它以多继承组合起来 —— 其中 6 个是基类，`constants` / `state` / `metrics` 只是被引用的
普通模块。它们共享同一份 `self` 上的会话状态。

模块地图
--------
constants.py    调优常量与音频契约（采样率 / 帧长 / 判停 / 打断 / 累积阈值）
state.py        5 状态枚举 `State`
metrics.py      单会话延迟统计 `SessionMetrics`
accumulator.py  ASR 文本累积：短句先缓存、拼整句再送 LLM
endpointing.py  判停：按本次说话时长自适应选静音阈值
transport.py    下行发送：系统话术 / 语气词 / TTS 音频
streaming.py    LLM 句子流 → TTS 音频流（首响关键路径）
pipeline.py     一轮对话的编排：ASR → RAG → LLM → TTS + 错误兜底
session.py      会话生命周期：问候 / 空闲超时 / 文本输入
manager.py      门面 `ConversationManager`

只在这里导出最常用的三个名字，其余请从各自模块导入。
"""

from .manager import ConversationManager
from .metrics import SessionMetrics
from .state import State

__all__ = ["ConversationManager", "SessionMetrics", "State"]
