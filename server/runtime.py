"""进程级共享状态：全局引擎表 + 活动会话表 + 累计计数。

为什么集中成一个对象：
  · 引擎表是**跨会话共享**的单例（VAD 模板、ASR、RAG、TTS 只加载一份）；
  · 会话表与计数是**进程级**统计，`/api/metrics` 和每个连接都要读写。

全部只在 asyncio 事件循环单线程里读写（uvicorn 单 worker），所以**不加锁**。
"""

import time


class Runtime:
    """一个进程一份。字段都是"活得比任何单个会话久"的东西。"""

    def __init__(self):
        # 引擎表：load_engines() 启动期写入，之后所有请求只读。
        # 键：vad_type / vad_template / spec_asr / asr / rag / llm_factory / tts
        self.engine: dict = {}

        # 活动会话：{session_id: {"started": epoch秒, "cm": ConversationManager}}
        self.active_sessions: dict[str, dict] = {}

        # 累计计数：total_sessions 建连数，total_errors 会话级错误数之和
        self.total_sessions = 0
        self.total_errors = 0

    # ── 会话登记/注销：成对调用，避免漏掉计数或残留条目 ──

    def register_session(self, session_id: str, cm) -> None:
        """建连时登记：写会话表并自增建连计数。"""
        self.total_sessions += 1
        self.active_sessions[session_id] = {"started": time.time(), "cm": cm}

    def unregister_session(self, session_id: str, cm) -> dict:
        """断连时注销：累加该会话的错误数、移出会话表，返回它的指标快照。"""
        metrics = cm.metrics.summary()
        self.total_errors += metrics.get("errors", 0)
        self.active_sessions.pop(session_id, None)
        return metrics


# 全进程唯一实例（其它模块一律 `from .runtime import runtime` 后按属性访问）
runtime = Runtime()
