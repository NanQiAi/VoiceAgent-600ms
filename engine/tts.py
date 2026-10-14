"""VoxCPM 1.5 流式 TTS（nanovllm-voxcpm 后端）+ 零样本音色克隆，级联 Pipeline 的最后一环。

输出固定**单声道 44100Hz float32 裸 PCM**（无 WAV 头，量化成 int16 是调用方的事）；
`SAMPLE_RATE=44100` 是下行链路，与上行 ASR 的 16000Hz（config.SAMPLE_RATE）不是一回事。
引擎**不是线程安全**的：所有 generate() 调用必须串行（由调用方的有界队列间接保证）。
"""
# os 只用于 basename 打日志与后缀判断；time.perf_counter 测 ttfa/total；
# logging 统一用 voxlabs.* 命名空间；numpy 承载音频块与淡入斜坡
import os
import time
import logging
import numpy as np


# ws_server 里 logging.basicConfig 的格式串含 %(name)s，用 "voxlabs.*" 前缀可一条命令过滤整条 pipeline
log = logging.getLogger("voxlabs.tts")


class VoxCPMTTS:
    """VoxCPM 1.5 流式 TTS 引擎（单实例 = 单卡一份模型 + 若干克隆音色 prompt）。

    生命周期：`__init__`（只存配置）→ `load()`（吃显存）→ `register_voice()` → `warmup()` → `synthesize*()`。
    """

    # VoxCPM 固定输出采样率（Hz），与上行 ASR 的 16000Hz（config.SAMPLE_RATE）不是一回事：
    # conversation_manager 用它算 30s 安全上限，filler 用它算截断字节数，tts_server 靠它填 X-Sample-Rate
    SAMPLE_RATE = 44100

    def __init__(self, model_dir: str, device: str = "cuda:2",
                 gpu_memory_utilization: float = 0.55):
        """保存加载参数，**不加载模型、不碰 CUDA**（延迟到 `load()`）。

        `gpu_memory_utilization` 是 nanovllm 的显存预分配比例：既决定 KV cache 大小，
        也决定留给同卡 ASR/RAG 的余量——TTS 与 ASR 默认同卡，必须留白，经验值 0.45~0.55。
        """
        self._model_dir = model_dir
        self._device = device
        self._gpu_memory_utilization = gpu_memory_utilization
        # 后端句柄；load() 之前调用 synthesize*/register_voice 会在 None 上取属性报错，
        # 调用顺序由 ws_server / tts_server 的启动流程保证
        self._engine = None
        self._warmed_up = False  # warmup() 幂等靠这个标志，避免重复付预热代价
        self._default_prompt_id = None  # None 时 synthesize 走模型自带音色

    def load(self):
        """把 VoxCPM 1.5 加载到目标 GPU（阻塞，数秒到数十秒），返回 self 以便链式调用。

        延迟 import `nanovllm_voxcpm`：这个包依赖 CUDA，只在真要跑 TTS 的进程里才 import，
        避免单纯 import engine.tts 就触发 GPU 初始化（tts_server.py 正是靠这一点把它放进
        独立进程，规避 fork-in-fork 的 CUDA 问题）。副作用：初始化 CUDA context 并按
        gpu_memory_utilization 预分配显存；权重缺失/显存不足/CUDA 不可用时异常向上抛，直接启动失败。
        """
        # 延迟导入：只有真正加载模型时才拉起 CUDA 相关依赖
        from nanovllm_voxcpm import VoxCPM
        # 从 "cuda:2" 里取出 2 作为**进程内**可见设备序号；不带冒号回退为 0。
        # 若外部用 CUDA_VISIBLE_DEVICES 限制可见卡，这里就是"可见卡内序号"：
        # CUDA_VISIBLE_DEVICES=2 + device="cuda:0" 指向物理 2 号卡（tts_server 用法）。
        gpu_idx = int(self._device.split(":")[-1]) if ":" in self._device else 0
        self._engine = VoxCPM.from_pretrained(
            model=self._model_dir,
            gpu_memory_utilization=self._gpu_memory_utilization,
            devices=[gpu_idx],  # 后端要求设备序号列表，而不是字符串
        )
        return self

    def register_voice(self, wav_path: str, prompt_text: str) -> str:
        """用"参考音频 + 该音频的转写文本"注册克隆音色，返回 prompt_id。

        VoxCPM 是零样本 TTS：参考音频提供音色/口音/语速，参考文本让模型把音频对齐到文字，
        两者必须严格对应，否则克隆出的音色会飘。prompt 注册进后端后可反复复用，
        所以只在启动期注册一次；**想换音色素材必须重启进程**（后端没有替换/反注册接口）。
        注意：格式只按后缀判断且**只识别 .mp3 / .flac，其余一律当 wav**，
        传 .m4a/.ogg 会被按 wav 解析而失败；文件不存在时 FileNotFoundError。
        """
        # 以二进制读整份音频：后端要的是原始字节，自己负责解码
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        fmt = "wav"  # 默认按 wav 解析（与 config.VOICE_PROMPT_WAV 的后缀一致）
        # 后端按字符串选解码器，填错即失败，所以不做"猜格式"
        if wav_path.endswith(".mp3"):
            fmt = "mp3"
        elif wav_path.endswith(".flac"):
            fmt = "flac"

        prompt_id = self._engine.add_prompt(wav_bytes, fmt, prompt_text)
        # 只打文件名不打全路径：日志更短，也避免泄漏本机目录结构
        log.info("Voice prompt registered: id=%s, file=%s", prompt_id, os.path.basename(wav_path))
        return prompt_id

    def set_default_voice(self, prompt_id: str):
        """把 prompt_id 设为后续合成的默认音色；**不校验 id 是否存在**，传错要等合成时才由后端报错。"""
        self._default_prompt_id = prompt_id  # 只存引用，prompt 数据仍在后端引擎里
        log.info("Default voice set to: %s", prompt_id)

    def warmup(self):
        """预热：跑一次最短合成（幂等），把首次推理的一次性开销提前付掉。

        第一次 `generate()` 还要额外付 CUDA context 初始化、cuBLAS/cuDNN autotune、
        KV cache 首次分配、VAE 解码器首次图构建等成本（几百 ms~秒级）；不预热它们会叠加在
        第一轮真实对话的 ttfa 上、直接击穿 600ms 首响预算。启动流程在开始服务前调用。
        副作用：触发一次真实 GPU 推理（阻塞数百 ms~秒级），成功后置 `_warmed_up`。
        """
        if not self._warmed_up:  # 幂等：即使被多次调用也只付一次成本
            # 用一句极短中文问候做预热文本，长度接近真实首句，让预热代价贴近线上第一句
            self.synthesize("你好，欢迎致电。")
            # 合成成功后才置位：失败时下次调用仍会重试，不会被误标为已预热
            self._warmed_up = True

    def synthesize(self, text: str, prompt_id: str = None) -> dict:
        """非流式整句合成，返回 dict（audio / sr / ttfa_ms / total_ms / chunks）。

        给"不适合流式"的场景用：启动期预合成 filler、standalone tts_server 的 /synthesize。
        主链路请用 `synthesize_stream()`，否则会把首帧延迟藏到整句合成完之后。
        首 chunk 被丢弃：VoxCPM 的第一个 chunk 是 **VAE 启动瞬态**（上电阶段的咔哒/爆音），
        整块跳过；第二个 chunk 才是首个可听块，对它做 10ms 线性淡入避免硬切爆点。
        因此 `ttfa_ms` 记的是"第一个 chunk（含被丢弃的那个）"的到达时刻，与 README 174ms 同口径。
        `audio` 是未量化的裸 PCM；后端一个 chunk 都没产出时是空数组，调用方需自行判断。
        副作用：触发一次完整 GPU 自回归推理（阻塞当前线程）；**原地修改**第二个 chunk 的前 10ms。
        """
        t0 = time.perf_counter()
        chunks = []
        ttfa = None  # 首 chunk 到达时刻（ms）；None 表示还没出现过任何 chunk

        pid = prompt_id or self._default_prompt_id
        # temperature=0.7 比 LLM 低得多：TTS 要的是稳定复现同一音色，温度高会让音色/韵律每句都漂。
        # cfg_value=3.0 是 CFG 强度：越大越"贴"参考音色但可能发闷，3.0 是自然度与相似度的折中。
        kwargs = {"target_text": text, "temperature": 0.7, "cfg_value": 3.0}
        # 只有拿到有效 prompt 才传 prompt_id：传 None 会让后端按"无参考"处理，与省略该参数未必等价
        if pid:
            kwargs["prompt_id"] = pid

        chunk_idx = 0  # 先自增，于是第一个 chunk 的 chunk_idx == 1
        for audio_chunk in self._engine.generate(**kwargs):
            # 后端一般直接给 CPU 上的 np.ndarray（零拷贝）；torch.Tensor 需先搬 CPU 才能转 numpy
            if isinstance(audio_chunk, np.ndarray):
                c = audio_chunk
            elif hasattr(audio_chunk, 'numpy'):
                c = audio_chunk.numpy()
            else:
                c = np.array(audio_chunk, dtype=np.float32)

            chunk_idx += 1
            if chunk_idx == 1:
                ttfa = (time.perf_counter() - t0) * 1000  # README 里 TTS ≈174ms 的口径
                # Skip first chunk entirely — VoxCPM VAE has startup transient
                # （保留原注释）VAE 上电瞬态，整块丢弃，不参与拼接也不送播放
                continue

            if chunk_idx == 2:  # 首个可听块：只对它做起音淡入，并让它携带 ttfa
                # Fade-in on the second chunk (now the first audible one)
                # （保留原注释）丢弃首块后这块成了真正起音，硬切会有爆点。
                # 10ms = 44100 × 0.01 = 441 点，短到听不出音量变化，又足够压掉跳变的"啪"声。
                fade = int(self.SAMPLE_RATE * 0.01)
                if len(c) > fade:
                    # dtype=c.dtype 避免 float32 被 linspace 默认的 float64 提升。
                    # 注意这是**原地修改**：后端 yield 的数组若被它自己复用会污染后端状态；
                    # 当前每块都是独立数组所以安全，换后端时要留意。
                    c[:fade] *= np.linspace(0, 1, fade, dtype=c.dtype)

            chunks.append(c)  # 只有非首块才入列，顺序即播放顺序

        total = (time.perf_counter() - t0) * 1000
        # 一个 chunk 都没有时给空数组，避免 np.concatenate 对空列表抛 ValueError
        audio_out = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

        return {  # 波形 / 采样率 / 首帧延迟 / 总耗时 / 有效块数
            "audio": audio_out,
            "sr": self.SAMPLE_RATE,
            # ttfa 为 None（后端没产出任何 chunk）时退化成 total，保证下游拿到的永远是有限数
            "ttfa_ms": ttfa or total,
            "total_ms": total,
            "chunks": len(chunks),  # 已排除被丢弃的首块
        }

    def synthesize_stream(self, text: str, prompt_id: str = None):
        """流式合成：边生成边 yield，每个 chunk 一个 dict（audio / chunk_idx / ttfa_ms）。

        "160ms 精确打断"落在调用方：conversation_manager 每拿到一个 chunk 就立即下发裸 PCM，
        并在**每个 chunk 之前**检查取消标志，用户一开口就 break。所以打断粒度 = 一个 chunk 的
        播放时长（README 标称 ~160ms），而且取消只停"下发"、不会掐断已跑起来的 TTS 推理。
        块处理与 `synthesize()` 相同（丢弃首块瞬态 + 第二块 10ms 淡入），区别是不收集、不拼接、
        不统计 total；`chunk_idx` 从 2 开始（1 是被丢弃的瞬态块），`ttfa_ms` 只在首个可听块上给真实值。
        本函数是**生成器**：不迭代就不会跑推理；首次迭代触发流式 GPU 推理并阻塞当前线程。
        """
        t0 = time.perf_counter()  # ttfa_ms 从"进入本函数"算起，含建 prompt + 首帧推理
        ttfa = None

        pid = prompt_id or self._default_prompt_id
        # 采样参数与 synthesize() 保持一致，保证流式/非流式音色相同
        kwargs = {"target_text": text, "temperature": 0.7, "cfg_value": 3.0}
        if pid:
            kwargs["prompt_id"] = pid

        chunk_idx = 0
        for audio_chunk in self._engine.generate(**kwargs):
            if isinstance(audio_chunk, np.ndarray):
                c = audio_chunk
            elif hasattr(audio_chunk, 'numpy'):  # torch.Tensor：需搬到 CPU 才能转 numpy
                c = audio_chunk.numpy()
            else:
                c = np.array(audio_chunk, dtype=np.float32)

            chunk_idx += 1
            if chunk_idx == 1:
                # 记录真正的首帧时间；这块随后被丢弃（不 yield），所以 ttfa_ms 会挂到下一块上
                ttfa = (time.perf_counter() - t0) * 1000
                # 丢弃首块（VAE 启动瞬态）。这里是 continue 而不是 break：继续向下产出真正的音频
                continue

            if chunk_idx == 2:  # 首个可听块：10ms 淡入消爆点，且只有它带 ttfa_ms（44100 × 0.01 = 441 点）
                fade = int(self.SAMPLE_RATE * 0.01)
                if len(c) > fade:
                    c[:fade] *= np.linspace(0, 1, fade, dtype=c.dtype)  # 原地乘斜坡，dtype 跟随音频块保精度

            # 逐个 yield 会挂起生成器、把控制权交回 conversation_manager，
            # 它此时就发这块并检查取消标志 → 这正是打断能及时生效的原因
            yield {
                "audio": c,
                "chunk_idx": chunk_idx,  # 已含"第 1 块被丢弃"的信息，便于日志里对照
                # 只有首个可听块带 ttfa；其余用 0 表示无意义，避免下游误读成每块都这么慢
                "ttfa_ms": ttfa if chunk_idx == 2 else 0,
            }
