"""LLM 推理：通过 vLLM 的 OpenAI 兼容 API 做流式/非流式对话（只是 HTTP 客户端，不加载权重）。

`chat()` 等整段回答（评测用）；`stream_sentences()` 把 delta 拼成完整句子逐句 yield，
主链路用后者——首句一凑齐就送 TTS。多轮历史由本类维护，超出上限的旧轮次压成中文摘要。
ws_server 为**每个 WebSocket 会话**新建一个实例；同一实例不要并发调用 `stream_sentences()`。
"""
# re：清洗模型输出（去 think 块 / XML 标记）与断句都靠预编译正则，模块级编译一次、全程复用
import re
# time：只用 perf_counter——单调时钟，适合测延迟差，不受系统改时间影响
import time
# httpx：不是用来直接发请求的，而是为了构造分项超时对象 httpx.Timeout 再交给 OpenAI 客户端
import httpx
# OpenAI：vLLM 暴露 OpenAI 兼容接口，所以直接复用官方 SDK；换任何兼容后端都不用改本文件
from openai import OpenAI

# <think>...</think>：Qwen3 思考模式下的思维链片段。服务端已用 extra_body 的 enable_thinking=False
# 关了思考，这里再删一遍是**二次防御**——绝不能让"内心独白"被念给用户听。
# re.DOTALL 让 `.` 匹配换行，否则跨行思维链删不干净。
_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# 落单的 think 块（被截断/还在生成中）：贪婪吃到末尾，抹掉已吐出来的半截思维链，避免污染尾句
_INCOMPLETE_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
# 其它 XML / 特殊 token（<xxx>、</xxx>、<|xxx|>，如 <|im_start|>、模型自造的 <emotion>）：念出来就是噪音
_ANY_XML_TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")
# 断句符集合：中英文句末标点 + 分号 + **逗号** + 换行。逗号也算断句点是有意的工程取舍——
# 语音场景宁可早说半句也不愿让用户多等（代价是语气可能被切断，靠 config.SYSTEM_PROMPT 里
# "每句话的内容要尽可能简短精炼"的约束缓解）。换行同理，是天然停顿点。
_SENTENCE_END_RE = re.compile(r"[。！？!?；;，,\n]")


class VLLMChat:
    """vLLM Chat Completions 的会话级客户端（一个实例 = 一次通话）。

    只负责拼 messages（system + 历史 + 当前输入 + RAG 上下文）→ 发请求 → 解析 → 维护历史；
    不负责 RAG 检索、不负责 TTS，也不持有任何 GPU 资源。
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "MiniCPM4.1-8B-GPTQ",
                 system_prompt: str = "", max_history: int = 20):
        """构造客户端（只建连接池，不发请求），并初始化空的会话上下文。

        `base_url` **必须带 `/v1` 后缀**（生产传 config.VLLM_BASE_URL，默认 8100 端口）；
        `model` 必须与服务端 `--served-model-name` 一致，否则返回 404（生产值
        config.VLLM_MODEL_NAME = "Qwen3-14B-AWQ"）；`max_history` 是消息**条**数上限
        （一问一答算 2 条，生产值 config.LLM_MAX_HISTORY = 30）。构造不发网络请求，失败推迟到第一次调用。
        """
        # api_key 固定 "dummy" 占位：vLLM 默认不校验鉴权，但 openai SDK 在 key 为空时会直接抛错。
        # timeout 用 httpx.Timeout 分段设置而非一个总超时，因为各阶段合理等待量级差很多：
        #   connect=5.0  建连失败要快速失败，别让用户干等（vLLM 在本机，5s 足够）；
        #   read=30.0    流式读单个 chunk 的等待上限；prefill 慢或 vLLM 排队时给足余量；
        #   write=5.0    请求体只有几 KB 文本；pool=10.0 从连接池借连接的等待上限。
        self._client = OpenAI(
            base_url=base_url, api_key="dummy",
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=10.0),
        )
        self._model = model
        self._system_prompt = system_prompt
        self._history = []  # 元素 {"role": "user"/"assistant", "content": str}，成对追加
        self._max_history = max_history  # 注意是"条"不是"轮"
        self._summary = ""  # 被裁掉的历史压成的摘要，追加到 system prompt 末尾；空串 = 还没裁过

    def reset(self):
        """清空多轮上下文（历史 + 摘要），回到"刚接起电话"的状态（连接池不受影响）。"""
        # 直接换新列表而不是 clear()：杜绝外部还持有旧列表引用时的串音风险
        self._history = []
        self._summary = ""

    def _trim_history(self):
        """Keep last max_history messages; summarize evicted turns into _summary.

        为什么需要它：prompt 越长 prefill 越慢，会直接吃掉 600ms 预算，而电话场景里很早
        以前的寒暄对当前回答几乎没有价值。摘要**逐代覆盖**（不累积拼接），每句只取前 60 个
        字符——刻意做得粗糙而便宜，定位是"别彻底失忆"而不是精确记忆；只在超限时才写入。
        """
        if len(self._history) <= self._max_history:
            return  # 绝大多数轮次走这条快路径，零开销
        # 切片产生新列表，不改动 self._history
        evicted = self._history[:len(self._history) - self._max_history]
        self._history = self._history[-self._max_history:]
        turns = []
        # step=2：历史按 (user, assistant) 成对追加，隔一条取一条正好凑出问答对；
        # 上界 len(evicted)-1 保证 i+1 不越界（凑不出完整问答对时自然跳过落单那条）
        for i in range(0, len(evicted) - 1, 2):
            u = evicted[i].get("content", "")[:60]
            a = evicted[i + 1].get("content", "")[:60] if i + 1 < len(evicted) else ""
            turns.append(f"用户:{u} → AI:{a}")
        if turns:  # 只有真凑出过问答对才覆盖摘要；否则保留上一代摘要（宁可旧、不可空）
            # 只保留最近 5 组避免摘要无限增长；固定前缀便于模型把它识别为"历史摘要"而非当前指令
            self._summary = "之前的对话摘要：" + "；".join(turns[-5:])

    def _build_messages(self, user_text: str, rag_context: str = None):
        """拼装本次请求的 messages（system? + 历史 + 当前输入）；纯拼装，不改 self 状态。"""
        messages = []
        system = self._system_prompt
        # 有 RAG 结果时才注入：优先原地替换 prompt 里的 {context} 占位符（config.SYSTEM_PROMPT_RAG
        # 用的就是这种模板），没有占位符就追加"知识库信息"小节——这样纯指令版与带占位符版都能兼容
        if rag_context:
            system = system.replace("{context}", rag_context) if "{context}" in system else (
                system + f"\n\n知识库信息：\n{rag_context}"
            )
        # 摘要拼在 system 末尾（同一条消息内），不额外占一条消息：prompt 短一点，prefill 就快一点
        if self._summary:
            system = system + f"\n\n{self._summary}"
        # system_prompt 为空（例如离线测试）时不发空 system 消息，某些服务端对空 content 会报错
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self._history)  # 引用扩展，不复制也不改写历史
        # 当前用户输入必须放最后一条，否则模型会把历史最后一条当成要回应的对象
        messages.append({"role": "user", "content": user_text})
        return messages

    def chat(self, user_text: str, rag_context: str = None) -> dict:
        """非流式一次性拿完整回答，返回 dict（text / latency_ms / tokens）；低延迟主链路请用 `stream_sentences()`。

        `text` 已清洗（去 think 块与 XML 标记）；`latency_ms` 含排队 + prefill + 全部 decode + 回传；
        `tokens` 取服务端 usage 的 completion_tokens，缺失时记 0。副作用：同步 HTTP 请求会
        **阻塞当前线程**，成功后把这一问一答追加进 `_history` 并裁剪；异常向上抛给调用方兜底。
        """
        messages = self._build_messages(user_text, rag_context)  # 每轮现拼，历史与 RAG 都是动态的
        t0 = time.perf_counter()
        # extra_body 是 openai SDK 的逃生口，用来传 vLLM 私有字段：
        #   chat_template_kwargs.enable_thinking=False
        #       —— Qwen3 默认开思考模式，会先生成一大段思维链："首 token"其实不慢，但"首句可播"
        #          会晚几百 ms，且内容不适合念给用户，所以显式关闭；
        #   repetition_penalty=1.15
        #       —— 经验值：电话话术句子短、容易复读同一句，1.15 的轻惩罚能明显减少复读机现象；
        #          再往上调会伤自然度，故不用更高值。
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15}

        resp = self._client.chat.completions.create(  # 同步阻塞：等整段 decode 完才返回
            model=self._model, messages=messages,
            # max_tokens=80：刻意压短输出——电话场景只要 1~2 句话，既控制单轮通话时长，
            #   也把 decode 耗时钉在可预测范围内。
            # temperature=0.85 + top_p=0.9：偏高的温度让口语更自然多变、避免客服腔，
            #   再用 top_p 截掉长尾防止乱说，是"像真人"与"不出格"之间的取舍。
            max_tokens=80, temperature=0.85, top_p=0.9, extra_body=extra_body,
        )
        latency = (time.perf_counter() - t0) * 1000  # 下游协议字段统一用毫秒
        # content 可能为 None（例如只吐了 stop token），用 `or ""` 兜底后再清洗
        text = self._clean(resp.choices[0].message.content or "")
        tokens = resp.usage.completion_tokens if resp.usage else 0

        # 只有成功拿到回答才写历史，保证 _history 里永远是完整的 (user, assistant) 对
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": text})
        self._trim_history()

        return {"text": text, "latency_ms": latency, "tokens": tokens}

    def stream_sentences(self, user_text: str, rag_context: str = None):
        """流式把 delta 拼成完整句子逐句 yield，每句一个 dict（sentence / ttfs_ms / is_first / is_last）。

        低延迟主链路入口：conversation_manager 在独立线程里跑生产者（有界队列 maxsize=8 做背压），
        主协程拿到一句就立刻调 TTS。断句用 `_SENTENCE_END_RE` 就近贪心切分（一次 delta 可能带多个
        标点，故内层 while 反复切）；长度 < 2 的碎片丢弃，避免白抖一次 TTS 推理。`ttfs_ms` 只在
        第一句上给真实值。真正的请求发生在被**迭代**时，不迭代等于没调用；只有迭代完整跑完才把整轮
        回答写入 `_history`，中途 break（用户打断）则历史不落库——刻意为之，半句不值得进上下文。
        """
        messages = self._build_messages(user_text, rag_context)
        t0 = time.perf_counter()  # ttfs_ms 的计时起点，覆盖建连 + prefill + 攒够首句标点
        # 与非流式完全相同的采样与关思考参数，保证两种形态输出风格一致
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.15}

        stream = self._client.chat.completions.create(  # 返回可迭代的 SSE 流对象，此刻尚无 token
            model=self._model, messages=messages,
            max_tokens=80, temperature=0.85, top_p=0.9,
            # stream=True：让 vLLM 用 SSE 逐 token 推送，每行形如 data: {"choices":[{"delta":{...}}]}。
            # 这是 163ms 首 token 能被"利用"的前提——不流式就只能等 80 个 token 全部 decode 完
            # （数百 ms）才拿到第一个字，首响预算直接崩。
            stream=True, extra_body=extra_body,
        )

        buffer = ""  # 尚未成句的增量文本
        full_text = ""  # 已 yield 出去的句子累积，仅用于流结束后写历史
        first_sentence = True
        # 首个非空 delta 的到达时刻；当前未参与返回值，保留是为了方便后续接入"纯首 token 延迟"指标
        first_token_time = None

        for chunk in stream:  # 每帧是 vLLM 新 decode 出的若干字符（1 个 delta）
            # "心跳/结束"帧（choices 为空，或 delta.content 为 None）必须跳过，否则会拼进 None 或下标越界
            if not chunk.choices or not chunk.choices[0].delta.content:
                continue
            token = chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = time.perf_counter()  # 只记第一次；perf_counter 单调，差值即真实等待

            buffer += token

            # 一次 delta 可能带进多个标点（例如"好的，我明白了。"），所以循环切到切不出新句为止
            while True:
                match = _SENTENCE_END_RE.search(buffer)
                if not match:
                    break  # 缓冲里已无断句符：本帧切不出新句，等下一个 delta 再试
                # 断句符本身要留在句子里（TTS 靠标点拿语气与停顿），所以用 match.end()
                idx = match.end()
                sentence = self._clean(buffer[:idx])
                buffer = buffer[idx:].lstrip()  # 下一句开头；lstrip 去掉标点后常跟的空格/换行

                # 过滤空串和单字符碎片：单标点合成没意义，且下游 conversation_manager 也要求 ≥2 字
                if sentence and len(sentence) >= 2:
                    elapsed = (time.perf_counter() - t0) * 1000  # 从请求发出到"可播"的累计耗时
                    yield {
                        "sentence": sentence,
                        # 只有第一句带 ttfs：它才是"LLM 首句延迟"这个指标的含义
                        "ttfs_ms": elapsed if first_sentence else 0,
                        "is_first": first_sentence,  # 下游据此发 audio_start、记首响指标
                        "is_last": False,  # 中间句一律 False；真正的尾句在循环外单独发
                    }
                    full_text += sentence
                    first_sentence = False

        # 流结束后缓冲区可能还剩没有断句符的尾巴（模型最后一句没打标点，或 max_tokens=80 恰好截断）——
        # 这段也必须播出去，否则回答会缺结尾
        remainder = self._clean(buffer)
        if remainder and len(remainder) >= 2:
            elapsed = (time.perf_counter() - t0) * 1000
            yield {
                "sentence": remainder,
                "ttfs_ms": elapsed if first_sentence else 0,  # 整轮只有这一句尾巴时它同时也是首句
                "is_first": first_sentence,
                "is_last": True,
            }
            full_text += remainder
        elif full_text:
            pass  # last sentence already yielded
        # 注意：若 remainder 太短（<2 字）且 full_text 为空，本轮会一句都不 yield，上层表现为
        # "AI 没说话"。这是已知边界情况，此处不改行为。

        # 生成器主体跑完（调用方正常迭代到底）才写历史，保证 (user, assistant) 成对；
        # 中途 break 会跳过这几行，见 docstring 的"副作用"说明
        final_text = self._clean(full_text)
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": final_text})  # 与用户实际听到的一致
        self._trim_history()

    @staticmethod
    def _clean(text: str) -> str:
        """清洗模型输出并 strip 首尾空白。三道正则替换的**顺序有意义**：先删成对/半截 think 块，
        再用通用 XML 规则扫剩下的标记。纯函数、无状态，因此声明为 staticmethod。"""
        # 1) 完整的思维链块：带结尾标签的，整块连尾部空白一起删
        text = _TAG_RE.sub("", text)
        # 2) 半截思维链：没有结尾标签（流被截断），贪婪吃到结尾
        text = _INCOMPLETE_THINK_RE.sub("", text)
        # 3) 其它 XML / 特殊 token：如 <|im_end|>、<emotion> 等
        text = _ANY_XML_TAG_RE.sub("", text)
        return text.strip()
