"""DTLN 两级 ONNX 降噪（stage1 频域掩码 + stage2 时域增强），CPU 上约 8ms / 512 点帧。

级联 Pipeline 中 ASR 之前的可选环节，整句收齐后由 conversation_manager 丢线程池跑。
非线程安全：`_state1/_state2` 是实例级 LSTM 状态，同一实例同一时刻只能服务一个调用方。
（上游实现：https://github.com/breizhn/DTLN）
"""
import os
import logging
import numpy as np
import onnxruntime as ort

log = logging.getLogger("voxlabs.denoiser")  # 沿用 "voxlabs" 前缀，便于统一过滤引擎日志

BLOCK_LEN = 512  # 512 点 = 32ms@16kHz，必须等于 config.CHUNK_SAMPLES（ONNX 输入维度写死，改值即报错）
BLOCK_SHIFT = 128  # 上游 STFT 帧移 128 点 = 8ms@16kHz，即官方标称的 ~8ms 延迟；本实现整块切分、未引用
FFT_LEN = 512  # rfft 后取 257 个单边频点（512/2+1），即 stage1 的输入维度


class DTLNDenoiser:
    """DTLN 两级降噪封装：LSTM 状态跨帧保留，不可逐帧重建实例（降噪量骤降、帧边界出咔哒声）。"""

    def __init__(self, model_dir: str):
        """只记录模型目录，不加载模型；ONNX 会话在 `load()` 里创建。"""
        self._model_dir = model_dir
        self._model1 = None  # stage1 频域掩码会话，load() 后才有值
        self._model2 = None  # stage2 时域增强会话
        self._state1 = None  # LSTM 隐状态，形状 (1,2,128,2)
        self._state2 = None
        self._in_buffer = None
        self._out_buffer = None  # 上游 overlap-add 遗留字段，只在 reset() 里分配，读写路径都没用到

    def load(self):
        """创建两个 CPU ONNX 会话并复位状态（返回 self 以便链式调用）。

        缺 `model_1.onnx` / `model_2.onnx` 时由 ONNX Runtime 抛异常（权重需另行下载），
        启动流程不捕获、直接启动失败。
        """
        # 降噪在 CPU 且处在首响关键路径之前，重点是别抢 CPU、别引入抖动
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # 单个算子内部只用 1 线程：避免与 vLLM/ASR 抢 CPU 核
        opts.inter_op_num_threads = 1  # 算子之间也只用 1 线程：并发度交给上层线程池控制
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        m1_path = os.path.join(self._model_dir, "model_1.onnx")
        m2_path = os.path.join(self._model_dir, "model_2.onnx")
        # 强制 CPUExecutionProvider：模型仅 4MB、单帧约 8ms，放 GPU 要多占显存与 CUDA stream，还会和 ASR/TTS 抢卡
        self._model1 = ort.InferenceSession(m1_path, opts, providers=["CPUExecutionProvider"])
        self._model2 = ort.InferenceSession(m2_path, opts, providers=["CPUExecutionProvider"])
        self.reset()
        log.info("DTLNDenoiser loaded (model_dir=%s)", self._model_dir)
        return self

    def reset(self):
        """把 LSTM 隐状态与帧缓冲清零；开始时、换一段互不相关的音频时调用，不可在处理中途调用。"""
        # 形状由 ONNX 模型 input_3/input_5 签名固定为 (1, 2, 128, 2)：batch=1，128 为 LSTM 单元数，
        # 其余两轴是层/方向与 h、c 的打包；改维度会直接触发 ORT 输入校验失败
        self._state1 = np.zeros((1, 2, 128, 2), dtype=np.float32)
        self._state2 = np.zeros((1, 2, 128, 2), dtype=np.float32)
        self._in_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)
        self._out_buffer = np.zeros(BLOCK_LEN, dtype=np.float32)

    def process_chunk(self, audio: np.ndarray) -> np.ndarray:
        """处理一帧 512 点（32ms@16kHz）并推进 LSTM 状态，返回等长已降噪波形。

        流式入口，必须按时间顺序连续调用；不足 512 点右侧补零，超过则只取前 512 点。
        会改写 `_state1/_state2/_in_buffer`，同一实例不可多线程并发调用。
        """
        # ONNX 输入维度写死为 512，多退少补以免报错
        if len(audio) != BLOCK_LEN:
            if len(audio) < BLOCK_LEN:
                audio = np.pad(audio, (0, BLOCK_LEN - len(audio)))
            else:
                audio = audio[:BLOCK_LEN]

        self._in_buffer = audio.astype(np.float32)  # 必须 float32，类型不匹配 ONNX 会报错

        # ---- Stage 1（频域）：用幅度谱过 LSTM 预测干净语音的幅度掩码 ----
        in_mag = np.abs(np.fft.rfft(self._in_buffer)).reshape(1, 1, -1).astype(np.float32)  # (1,1,257)，对齐输入 input_2
        in_phase = np.angle(np.fft.rfft(self._in_buffer))  # 原始相位 (257,)；为取幅度和相位各算了一次 rfft，本可缓存

        # run 返回 (输出列表, 新状态)：上一帧 state 传进去、新 state 存回来，跨帧保持 RNN 状态就靠这一步
        out1, self._state1 = self._model1.run(
            None,
            {"input_2": in_mag, "input_3": self._state1},
        )

        # 估计幅度 × 原始相位重组复数谱：DTLN 只改幅度不改相位，因为相位估计极易引入 artifacts
        estimated_mag = out1[0]
        estimated_complex = estimated_mag * np.exp(1j * in_phase)
        estimated_block = np.real(np.fft.irfft(estimated_complex.squeeze())).astype(np.float32)

        # ---- Stage 2（时域）：再跑一级 LSTM，抑制 stage1 残留的 musical noise ----
        in_time = estimated_block.reshape(1, 1, -1).astype(np.float32)  # (1,1,512) 匹配 input_4
        out2, self._state2 = self._model2.run(
            None,
            {"input_4": in_time, "input_5": self._state2},
        )

        return out2[0].squeeze()

    def process(self, audio: np.ndarray) -> np.ndarray:
        """整句离线入口：补零到 512 整数倍后逐帧降噪，返回裁剪回原长的结果（会先 `reset()`）。"""
        original_len = len(audio)
        audio = audio.astype(np.float32)

        # 取模两次保证 len % 512 == 0 时 pad_len 为 0，不会多补一整帧
        pad_len = (BLOCK_LEN - len(audio) % BLOCK_LEN) % BLOCK_LEN
        if pad_len > 0:
            audio = np.pad(audio, (0, pad_len))  # 只在末尾补零：不改变原采样点时间轴，裁掉后逐点对齐

        self.reset()
        output = np.zeros_like(audio)
        # 整块切分、无重叠，与上游 128 点 hop + overlap-add 不同：帧边界不连续性由 stage2 时域网络补偿
        for i in range(0, len(audio), BLOCK_LEN):
            block = audio[i:i + BLOCK_LEN]
            output[i:i + BLOCK_LEN] = self.process_chunk(block)

        return output[:original_len]
