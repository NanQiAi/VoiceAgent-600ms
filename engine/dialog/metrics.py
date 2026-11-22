"""单会话的延迟统计（内存态，不上报、不落盘）。

由 ConversationManager 持有一份，ws_server 的 `/api/metrics` 读它。
键名与前端 updateLatency() 保持一致。
"""


class SessionMetrics:
    """单个 WebSocket 会话的内存态延迟统计（不上报、不落盘），由 ws_server 的 /api/metrics 读取。"""

    # __slots__ 限定实例属性：省内存（每连接一份）并防止手滑写错属性名
    __slots__ = ("turns", "errors", "asr_ms_list", "llm_ms_list", "tts_ms_list", "fr_ms_list")

    def __init__(self):
        """初始化为空统计（不读配置、无副作用）。"""
        self.turns = 0
        self.errors = 0
        self.asr_ms_list: list[float] = []
        self.llm_ms_list: list[float] = []
        self.tts_ms_list: list[float] = []
        # 端到端首响样本（ms）= asr + rag + llm首句 + tts首帧，对应 600ms 预算线
        self.fr_ms_list: list[float] = []

    def record_turn(self, asr_ms=0.0, llm_ms=0.0, tts_ms=0.0, fr_ms=0.0):
        """记录一轮对话的各段耗时，并自增 self.turns（只有非 0 参数才入样本）。"""
        self.turns += 1
        # 逐个 if 判真：0 表示这一段没测到（如播报被立刻打断），不计入平均
        if asr_ms: self.asr_ms_list.append(asr_ms)
        if llm_ms: self.llm_ms_list.append(llm_ms)
        if tts_ms: self.tts_ms_list.append(tts_ms)
        if fr_ms: self.fr_ms_list.append(fr_ms)

    def record_error(self):
        """累加一次错误计数（供 /api/metrics 汇总）。"""
        self.errors += 1

    def summary(self) -> dict:
        """汇总本会话统计：turns/errors 与各段均值（无样本时 avg 为整数 0）。"""
        def _avg(lst):
            """均值，保留 1 位小数；空列表返回整数 0（避免除零）。"""
            return round(sum(lst) / len(lst), 1) if lst else 0
        # 键名与 ws_server 的 /api/metrics 输出、前端 updateLatency() 保持一致
        return {
            "turns": self.turns,
            "errors": self.errors,
            "avg_asr_ms": _avg(self.asr_ms_list),
            "avg_llm_ms": _avg(self.llm_ms_list),
            "avg_tts_ms": _avg(self.tts_ms_list),
            "avg_first_response_ms": _avg(self.fr_ms_list),
        }
