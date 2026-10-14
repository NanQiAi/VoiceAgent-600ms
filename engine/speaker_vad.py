"""Speaker-aware VAD — Silero VAD 逐帧判有无语音，ECAPA-TDNN 验是不是本人。

用途：外呼/坐席场景里只让本人说话开门，挡掉旁人、外放串音与自己的 TTS 回声
（纯能量 VAD 分不出这些，误触发会白跑一整条 ASR→RAG→LLM→TTS 流水线）。
与 `engine/vad.py` 的 `SileroVAD` 是同一 `process_chunk` 契约，可无痛替换。
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

# 兼容补丁：torchaudio ≥2.1 删除了 list_audio_backends()，而 speechbrain（ECAPA
# 编码器）在导入/构造时会调用它做后端探测，缺失就抛 AttributeError。
# 这里在**模块导入时**就补好占位实现（放在模块顶部而非 load() 内部，
# 目的是保证 speechbrain 被导入之前补丁一定已生效）。
# 注意这是 monkey patch（改的是第三方模块），不是本项目的业务逻辑。
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    # 用 lambda 伪造一个只返回 soundfile 的后端列表，够 speechbrain 通过探测即可。
    torchaudio.list_audio_backends = lambda: ['soundfile']

log = logging.getLogger("voxlabs.svad")  # 沿用 "voxlabs" 前缀，便于统一过滤引擎日志

# ECAPA-TDNN（speechbrain/spkrec-ecapa-voxceleb）输出嵌入维度；本常量只是文档性声明，代码未引用。
SPEAKER_EMBED_DIM = 192
# 注册最少要 16000Hz × 0.5s = 8000 点；更短的音频 ECAPA 估计方差很大，宁可拒绝注册。
ENROLL_MIN_SAMPLES = 8000
# 验证滑窗 16000 点 = 1s，是 ECAPA 官方推荐的最短可靠量级；再短嵌入不稳，再长则增加 CPU 耗时。
VERIFY_WINDOW_SAMPLES = 16000
# 余弦相似度阈值（可由构造函数覆盖）：同一人同一声学通路通常 >0.5，不同人常 <0.3；
# 取 0.25 偏宽松，是为了容忍麦克风增益差与 AEC 残留，宁可漏挡也不要误挡真人。
SIMILARITY_THRESHOLD = 0.25


class SpeakerAwareVAD:
    """Silero VAD + ECAPA-TDNN 说话人确认；非线程安全，每个连接必须独立 deepcopy 一份实例。

    对外入口：`load()` / `enroll_speaker()` / `process_chunk()` / `reset()`。
    """

    def __init__(self, vad_model_dir: str, speaker_model_dir: str,
                 threshold: float = 0.5, speaker_threshold: float = SIMILARITY_THRESHOLD):
        """只把配置写进实例属性，不加载模型（加载在 `load()`）：Silero 走 JIT 或 torch.hub，
        声纹交给 speechbrain 的 `savedir=` 缓存目录；首次运行可能联网下载权重。

        `threshold` 是 Silero 语音概率阈值，`speaker_threshold` 是余弦相似度阈值（默认 0.25）。
        """
        # 两个阈值存为公开属性（无下划线），上层可在运行时按现场噪声调整。
        self.threshold = threshold
        self.speaker_threshold = speaker_threshold
        self._vad_model_dir = vad_model_dir
        self._speaker_model_dir = speaker_model_dir

        # 模型句柄：`load()` 之前保持 None，便于判断"是否已加载"。
        self._vad_model = None
        self._speaker_encoder = None
        # 本人声纹嵌入（L2 归一化后的 192 维向量）与"是否已注册"标志位。
        self._target_embedding = None
        self._enrolled = False

        # 状态机 / 滑动窗口的运行时状态，由 `reset()` 统一初始化
        self._speech_active = False
        self._silence_count = 0
        self._recent_speech_buffer = []  # 最近语音帧的环形缓冲（list[np.ndarray]，每项 512 点）
        self._recent_speech_samples = 0  # 缓冲内总采样点数，避免每次求和；上限 VERIFY_WINDOW_SAMPLES

    def load(self, device: str = "cpu"):
        """加载 Silero 与 ECAPA 编码器、复位状态；返回 self 以便链式调用。

        `device` **只作用于 ECAPA**（Silero 恒在 CPU）。副作用：可能经 torch.hub /
        speechbrain 联网下载权重，并在未设置时把 `HF_ENDPOINT` 指向 hf-mirror 镜像。
        模型缺失且无法下载时异常直接向上抛——启动期就失败，好过运行中才发现声纹过滤失效。
        """
        # 记住推理设备，`enroll_speaker`/`verify_speaker` 都要把张量搬到这里。
        self._device = device

        # 优先用工程内预置的 TorchScript 模型：离线可跑、启动快，不依赖 Hub。
        jit_path = os.path.join(self._vad_model_dir, "silero_vad.jit")
        if os.path.exists(jit_path):
            # map_location="cpu" 保证不因权重里记录的 cuda 设备号导致加载失败。
            self._vad_model = torch.jit.load(jit_path, map_location="cpu")
        else:
            # 兜底：本地没有 .jit 就现场从 GitHub 拉；onnx=False 表示要 PyTorch 版。
            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False,
            )
        # Silero 内部有 LSTM，必须 eval()，否则概率输出会漂。
        self._vad_model.eval()

        # 国内网络通常连不上 huggingface.co，speechbrain 却要从那下载 ECAPA 权重，
        # 所以默认切到 hf-mirror.com；只在用户没有显式设置时才写入，尊重已有配置。
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 延迟导入 speechbrain：导入链很重，放在这里可缩短冷启动时间。
        from speechbrain.inference.speaker import EncoderClassifier
        self._speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self._speaker_model_dir,
            run_opts={"device": device},
        )

        self.reset()
        log.info("SpeakerAwareVAD loaded (Silero + ECAPA-TDNN, device=%s)", device)
        return self

    def reset(self):
        """复位 VAD 递归状态、三态机与说话人验证缓存。

        副作用：调用 Silero 的 `reset_states()`，复位后前几帧 `speech_prob` 可能短暂不稳。
        **不**清除 `_target_embedding` / `_enrolled`：声纹是跨会话长期有效的注册信息。
        """
        # 只有模型已加载时才复位：load() 之前调用 reset() 不应因 None 而崩。
        if self._vad_model is not None:
            # 清空 Silero 的 LSTM 隐状态，否则上一轮的语音上下文会被带进这一轮。
            self._vad_model.reset_states()
        self._speech_active = False
        self._silence_count = 0
        # 列表与累计采样点数必须同时归零，否则两者会不一致。
        self._recent_speech_buffer = []
        self._recent_speech_samples = 0
        self._verify_counter = 0  # 距离下一次真正跑 ECAPA 还差几帧（每 3 个语音帧重算一次）
        # 最近一次相似度与"是否本人"缓存，用于在两次 ECAPA 之间保持 is_target_speaker 连续。
        self._last_similarity = 0.0
        self._last_is_target = False

    def enroll_speaker(self, audio: np.ndarray, sr: int = 16000):
        """用一整段**纯本人语音**注册声纹，写入 `_target_embedding` 并把 `_enrolled` 置位。

        音频短于 0.5s 时返回 False（不写入半成品）；只用前 10s 做 ECAPA 编码。
        副作用：跑一次 ECAPA 前向（CPU 上数十毫秒，阻塞调用线程）；置位后
        `process_chunk` 开始按声纹过滤，注册音频质量直接决定整场对话可用性。
        """
        # 长度门限：不足 0.5s 的音频估不出稳定嵌入，直接拒绝，避免污染声纹。
        if len(audio) < ENROLL_MIN_SAMPLES:
            log.warning("Enrollment audio too short (%d samples, need %d)", len(audio), ENROLL_MIN_SAMPLES)
            return False

        # 最多只用前 10s：更长的音频 ECAPA 收益很小，而编码耗时随长度线性增长。
        trim = min(len(audio), sr * 10)
        audio_trimmed = audio[:trim]

        # speechbrain 要求 (1, T) 的 2D float32 输入。
        tensor = torch.from_numpy(audio_trimmed).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = self._speaker_encoder.encode_batch(tensor)
        # L2 归一化后，余弦相似度退化成点积，`verify_speaker` 因此可以直接比阈值。
        self._target_embedding = F.normalize(embedding.squeeze(), dim=0)
        self._enrolled = True
        # 日志里打印归一化后向量的模长（正常接近 1.0，明显偏离说明嵌入异常，可用于排查）。
        log.info("Speaker enrolled (%.1fs audio, embedding norm=%.3f)",
                 len(audio_trimmed) / sr, self._target_embedding.norm().item())
        return True

    def verify_speaker(self, audio: np.ndarray) -> float:
        """对一段音频整体做 ECAPA 编码，返回与本人嵌入的余弦相似度 [-1, 1]。

        纯测量、不维护状态、不负责阈值判定（由调用方拿返回值比 `speaker_threshold`）。
        两个短路返回值：未注册时返回 1.0（无从验证就全放行，退化成纯 Silero VAD）；
        音频短于 3200 点（0.2s）时返回中性值 0.5（高于默认阈值 0.25，即按本人处理）。
        副作用：跑一次 ECAPA 前向（CPU 上数十毫秒），阻塞调用线程。
        """
        # 未注册：没有可比对的基准，返回最大相似度让上层保持放行。
        if not self._enrolled or self._target_embedding is None:
            return 1.0

        # 太短的片段嵌入不可靠，返回 0.5 这一中性且"偏放行"的值（详见 docstring）。
        if len(audio) < 3200:
            return 0.5

        tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = self._speaker_encoder.encode_batch(tensor)
        # 查询嵌入同样 L2 归一化，保证与 `_target_embedding` 的度量一致。
        query_emb = F.normalize(embedding.squeeze(), dim=0)
        similarity = F.cosine_similarity(self._target_embedding, query_emb, dim=0).item()
        return similarity

    def process_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        """主入口：逐 32ms 帧调用，返回标准 VAD dict 加说话人验证字段。

        返回始终含 5 个键：`speech_prob` / `speech_start` / `speech_end` /
        `is_target_speaker` / `speaker_similarity`（未检测到语音时后两个为 False / 0.0）。
        未注册时只要检测到语音就返回 `is_target_speaker=True`、相似度 1.0（不验证、全放行）。
        已注册时进入验证分支的前 2 个语音帧拿到的仍是上一次缓存（会话刚开始是 0.0/False），
        约 96ms 后才变真实判定，上层必须靠多帧确认来抵消这段暖机期。
        副作用：更新三态机与 1s 滑窗，周期性触发 ECAPA 前向（阻塞当前线程数十毫秒）。
        """
        # from_numpy 是零拷贝视图，dtype 转换会产生新张量。
        tensor = torch.from_numpy(audio_chunk).float()
        # Silero 接受 (T,) 或 (1, T)；统一补成带 batch 维的 2D 形状更保险。
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        # Silero 是**有状态**模型（内部 LSTM），必须严格按时间顺序调用，跳帧/乱序会让概率失真。
        prob = self._vad_model(tensor, sr).item()

        event = {
            "speech_prob": prob,
            "speech_start": False,
            "speech_end": False,
            "is_target_speaker": False,
            "speaker_similarity": 0.0,
        }

        # ---- 1s 滑动窗口：只保留最近 1s 的语音帧供 ECAPA 验证 ----
        if prob >= self.threshold:
            # 必须 copy：调用方的 chunk 往往来自复用的接收缓冲，不拷贝会被后续帧覆盖。
            self._recent_speech_buffer.append(audio_chunk.copy())
            self._recent_speech_samples += len(audio_chunk)
            # 超出 1s 从队首丢弃；`pop(0)` 对 32 帧级别的短列表足够快。
            while self._recent_speech_samples > VERIFY_WINDOW_SAMPLES:
                removed = self._recent_speech_buffer.pop(0)
                self._recent_speech_samples -= len(removed)
        # 静音帧且尚未进入说话态：清掉上一段说话的残留，以免污染下一段的说话人嵌入。
        else:
            if not self._speech_active:
                self._recent_speech_buffer = []
                self._recent_speech_samples = 0

        # ---- 三态机：静音 → 说话 → 确认（边沿产生）----
        # 静音→说话：置位、清零静音计数，抛 speech_start（上层据此开始积累音频）。
        if prob >= self.threshold and not self._speech_active:
            self._speech_active = True
            self._silence_count = 0
            event["speech_start"] = True
        # 说话→静音：连续 15 帧 × 32ms ≈ 480ms 才判结束，以抗句中短停顿。
        elif prob < self.threshold and self._speech_active:
            self._silence_count += 1
            if self._silence_count >= 15:
                self._speech_active = False
                self._silence_count = 0
                event["speech_end"] = True
        # 说话中又回到语音帧：清零静音计数，避免句中换气被误判成句尾。
        elif prob >= self.threshold and self._speech_active:
            self._silence_count = 0

        # ---- 说话人验证（与三态机正交，仅已注册时运行）----
        # 门槛：本帧是语音 + 已注册 + 窗口至少 0.2s（3200 点）才值得跑 ECAPA。
        if prob >= self.threshold and self._enrolled and self._recent_speech_samples >= 3200:
            self._verify_counter += 1
            # 节流：每 3 个语音帧（≈96ms）才算一次——ECAPA 在 CPU 上单次要几十毫秒，
            # 逐帧计算会压垮 600ms 首响预算；3 帧一次是响应延迟与判定新鲜度的折中。
            if self._verify_counter >= 3:
                recent_audio = np.concatenate(self._recent_speech_buffer)
                similarity = self.verify_speaker(recent_audio)
                # 缓存原始相似度（而非布尔判定），供后两帧与上层日志复用。
                self._last_similarity = similarity
                self._last_is_target = similarity >= self.speaker_threshold
                self._verify_counter = 0
            # 无论本帧是否重算都回填缓存，保证 is_target_speaker 在两次 ECAPA 之间连续。
            event["speaker_similarity"] = self._last_similarity
            event["is_target_speaker"] = self._last_is_target

        # 未注册时：只要检测到语音就当作本人，返回最大相似度，行为等价于纯能量 Silero VAD。
        elif prob >= self.threshold and not self._enrolled:
            event["is_target_speaker"] = True
            event["speaker_similarity"] = 1.0  # 约定值，便于上层统一格式化日志（并非真比对过）

        return event

    @property
    def is_speech_active(self):
        """内部三态机是否处于"说话中"（True 表示已抛 speech_start 且还没抛 speech_end）。"""
        return self._speech_active

    @property
    def is_enrolled(self):
        """本人声纹是否已注册（True 时 `process_chunk` 启用声纹过滤）；reset() 不会清除。"""
        return self._enrolled
