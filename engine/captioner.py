"""副语言音频描述（captioner）—— 把语音的情绪/语速/环境写成一句中文，注入 LLM 系统提示。

本模块是级联 Pipeline 的**旁路**组件，不在 VAD → ASR → RAG → LLM → TTS 主链上：
ConversationManager 用 asyncio.wait_for 限时 0.5s 调 describe(audio)，结果经 WS 下发并拼进 LLM
用户消息（"[语音观察：{caption}]\\n用户说：{user_text}"）。两实现由 ws_server.load_engines() 二选一：
设了 CAPTIONER_URL → OmniCaptioner（vLLM 上的 MiniCPM-o，质量高但多一次网络推理），否则 →
HeuristicCaptioner（RMS/过零率，零依赖零延迟）。约定：describe() 返回一句中文，失败返回 ""，
**不允许抛异常打断对话主链**。
"""
import time
import logging
import struct
import base64
import io
import numpy as np

log = logging.getLogger("voxlabs.captioner")

# 限定 Omni 模型输出：一句、≤30 字、四个维度（性别/语速/情绪/背景），避免注入 LLM 的文本过长
CAPTION_SYSTEM_PROMPT = (
    "你是一个语音分析助手。请用一句简短的中文描述这段语音的特征，包括：说话人性别、语速(快/正常/慢)、"
    "情绪(平静/高兴/着急/生气/犹豫/悲伤)、以及背景环境(安静/嘈杂/有音乐等)。"
    "只输出一句描述，不超过30字。"
)


def _audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """把 float32 波形手写成完整的 16bit PCM WAV 字节流（纯内存拼接，不落盘），用于套 data URL 传给音频接口。"""
    # float32 → int16：先裁剪到 [-1,1] 再乘 32767（int16 满量程），避免溢出回绕
    int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    # 采样点数，用于推算 data 块大小；16bit 单声道 → 每采样点 2 字节
    num_samples = len(int16)
    data_size = num_samples * 2
    buf.write(b'RIFF')
    # RIFF 块长度：36 是除 "RIFF"+本字段外的固定头长度，再加音频数据长度（小端 uint32）
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    # fmt 子块标识（注意末尾有一个空格）
    buf.write(b'fmt ')
    # "<IHHIIHH" 依次为：16=子块字节数, 1=PCM, 1=单声道, sr=采样率, sr*2=字节率, 2=块对齐, 16=位深
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(int16.tobytes())
    return buf.getvalue()  # 完整 WAV 字节流，可直接 base64 后走 HTTP


class HeuristicCaptioner:
    """无模型的降级实现：只看音量（RMS）与过零率，给出"语气/背景"标签；零依赖、微秒级。"""

    def describe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """用音量与过零率粗略描述这一段用户语音，任何情况下都返回非空字符串（无 IO、无网络）。"""
        # RMS（均方根）音量，量纲与波形一致，约 [0,1]，作为"激动程度"的代理指标
        rms = float(np.sqrt(np.mean(audio ** 2)))
        duration = len(audio) / sr

        # 过零率 ZCR：相邻采样点符号变化的平均比例；清音/噪声的过零率明显高于浊音
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio))) > 0))

        parts = []

        # 经验阈值：>0.15 视为喊话/激动，0.05~0.15 正常，0.01~0.05 平缓，<0.01 几乎听不见
        if rms > 0.15:
            parts.append("语气较激动")
        elif rms > 0.05:
            parts.append("语气正常")
        elif rms > 0.01:
            parts.append("语气平缓")
        else:
            parts.append("语气很轻")

        # 按中文约 3.5 字/秒的经验语速把时长折算成估算字数（仅用于下面的语速判断）
        words_est = duration * 3.5
        if duration > 0:
            # 死代码：words_est 恒等于 duration*3.5，故 words_est/duration 恒为 3.5，下面两个分支
            # 永远不会命中（作者的语速判断实际未生效，返回值里不会出现"语速偏快/偏慢"），仅标注不修改
            if words_est / duration > 4.5:
                parts.append("语速偏快")
            elif words_est / duration < 2.0:
                parts.append("语速偏慢")

        # 音量极低时（rms<0.005）基本只剩底噪，直接报"背景安静"
        if rms < 0.005:
            parts.append("背景安静")
        # 弱能量 + 高过零率 → 稳态背景噪声（经验组合判断，非严格声学分类）
        elif zcr > 0.3 and rms < 0.03:
            parts.append("可能有背景噪音")

        return "，".join(parts) if parts else "语气正常，背景安静"


class OmniCaptioner:
    """基于 vLLM 托管的音频多模态模型（MiniCPM-o、Qwen-Audio 等）的 captioner，走 OpenAI 兼容 HTTP 接口。"""

    def __init__(self, base_url: str = "http://localhost:8200/v1",
                 model: str = "MiniCPM-o-4.5-awq",
                 api_key: str = "dummy"):
        """只记地址与模型名，不建连接、不依赖 openai 包；base_url 默认指向本机 8200，model 须与 vLLM --served-model-name 一致。"""
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._client = None

    def load(self):
        """按需导入 openai 并创建客户端，返回 self；不发起网络请求，服务没起也不会在这里报错。"""
        # 延迟导入：只有真正使用 Omni 路线时才要求安装 openai
        from openai import OpenAI
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self  # 支持 OmniCaptioner(...).load() 链式写法

    def describe(self, audio: np.ndarray, sr: int = 16000) -> str:
        """调多模态模型生成一句副语言描述；客户端未加载或请求失败时返回 ""（**不向上抛异常**）。

        会发起一次同步 HTTP 请求，故调用方丢进 run_in_executor；外层 asyncio.wait_for 限时 0.5s，
        但线程池里的请求无法真正取消，超时后仍会在后台跑完（结果被丢弃）。
        """
        # 未 load（或 openai 不可用）：静默降级，返回空串让上层不注入 caption
        if self._client is None:
            return ""  # 未加载 → 本轮不加 caption，主链不受影响

        # 最多取 10 秒：限制 base64 后的请求体大小与推理耗时；取末尾而非开头，句尾语调最能反映情绪
        trim_samples = min(len(audio), sr * 10)
        audio_trimmed = audio[-trim_samples:]

        wav_bytes = _audio_to_wav_bytes(audio_trimmed, sr)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        # data URL：OpenAI 音频输入约定格式，前端不需要额外可访问的文件服务
        audio_url = f"data:audio/wav;base64,{audio_b64}"

        t0 = time.perf_counter()
        try:
            # 一次同步 chat completions 调用：system 限定输出格式，user 里同时给音频与文字指令
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    # user：多模态 content 数组，先音频后文字（MiniCPM-o 支持 input_audio）
                    {"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {"data": audio_url, "format": "wav"}},
                        {"type": "text", "text": "请描述这段语音的特征。"},
                    ]},
                ],
                # 只要一句话，60 token 足够，同时限制最坏情况下的生成耗时
                max_tokens=60,
                # 低温度：描述性任务不需要多样性，保证同段语音输出稳定
                temperature=0.3,
            )
            # 部分实现会返回 content=None，统一兜成空串
            text = resp.choices[0].message.content or ""
            latency = (time.perf_counter() - t0) * 1000
            log.info("OmniCaptioner: '%s' (%.0fms)", text.strip(), latency)
            # 去掉首尾空白后再返回，避免注入 prompt 时出现多余换行
            return text.strip()
        # 任何异常（连不上、超时、模型输出异常、配额等）都降级为空串，绝不影响正常对话
        except Exception as e:
            log.warning("OmniCaptioner failed: %s", e)
            return ""
