"""对话状态机的 5 个状态。

取值即下行协议 `{"type":"state","state":...}` 里的字符串，前端用它切指示灯。
状态流转见 `conversation_manager` 模块头部说明。
"""

from enum import Enum


class State(Enum):
    """对话状态机的 5 个状态。

    取值与下行协议 `{"type":"state","state":...}` 的字符串**一一对应**（前端 static/voice_agent.html 用它切指示灯）。
    """

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
