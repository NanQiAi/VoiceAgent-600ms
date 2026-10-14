# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""FireRedASR2 声学特征前端（fbank + CMVN），由 `FireRedAsr2.transcribe()` 调用。
特征配置必须与训练一致（16kHz、25ms 帧长、10ms 帧移、80 维 mel、snip_edges、推理
dither=0、同一份 cmvn.ark），否则输入分布漂移、CER 明显劣化。
"""
import math
import os

import kaldiio
import kaldi_native_fbank as knf
import numpy as np
import torch


class ASRFeatExtractor:
    """批处理特征提取器：fbank → CMVN → padding 成定长 batch。
    80 维 mel、25ms/10ms 写死在内部，外部只能通过 `kaldi_cmvn_file` 决定是否做 CMVN。
    """

    def __init__(self, kaldi_cmvn_file):
        """构造时读盘解析 CMVN；传空串跳过 CMVN，文件缺失/格式错会 assert 失败。"""
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        # dither 是训练期数据增强，推理必须关闭以保证特征可复现
        self.fbank = KaldifeatFbank(num_mel_bins=80, frame_length=25,
            frame_shift=10, dither=0.0)

    def __call__(self, wav_paths, wav_uttids):
        """批量提取 fbank + CMVN，返回 (feats_pad, lengths, durs, wav_paths, uttids)。
        wav_paths 可为路径 str（kaldiio 读盘）或 (sample_rate, wav_np) 已解码波形；提取失败
        的样本会被剔除（返回顺序可能与入参不同），整批无有效样本时前两项为 None。
        """
        feats = []
        durs = []
        return_wav_paths = []
        return_wav_uttids = []

        # 用第 0 个元素的类型区分"文件路径"与"已解码波形"（调用方保证列表非空）
        wav_datas = []
        if isinstance(wav_paths[0], str):
            for wav_path in wav_paths:
                sample_rate, wav_np = kaldiio.load_mat(wav_path)
                wav_datas.append([sample_rate, wav_np])
        else:
            wav_datas = wav_paths

        for (sample_rate, wav_np), path, uttid in zip(wav_datas, wav_paths, wav_uttids):
            # 时长（秒）= 采样点数 / 采样率，用于统计 RTF
            dur = wav_np.shape[0] / sample_rate
            fbank = self.fbank((sample_rate, wav_np))
            # 音频过短（snip_edges 后不足一帧）会产生空特征，送进模型会 shape 报错，直接跳过
            if fbank.shape[0] < 1:
                continue
            # 逐维减训练集均值、乘逆标准差，把分布对齐到训练统计量
            if self.cmvn is not None:
                fbank = self.cmvn(fbank)
            fbank = torch.from_numpy(fbank).float()
            feats.append(fbank)
            durs.append(dur)
            return_wav_paths.append(path)
            return_wav_uttids.append(uttid)
        if len(feats) > 0:
            lengths = torch.tensor([feat.size(0) for feat in feats]).long()
            feats_pad = self.pad_feat(feats, 0.0)
        else:
            lengths, feats_pad = None, None
        # 返回顺序固定为 (特征, 长度, 时长, 路径, uttid)，asr.py 按此顺序解包
        return feats_pad, lengths, durs, return_wav_paths, return_wav_uttids

    def pad_feat(self, xs, pad_value):
        """把 list[Tensor(T_i, D)] 右侧填充成 (N, T_max, D)；padding 区配合 lengths 做 mask。"""
        # type: (List[Tensor], int) -> Tensor
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = torch.ones(n_batch, max_len, *xs[0].size()[1:]).to(xs[0].device).to(xs[0].dtype).fill_(pad_value)
        for i in range(n_batch):
            pad[i, :xs[i].size(0)] = xs[i]
        return pad


class CMVN:
    """Kaldi 风格 CMVN：把 fbank 逐维标准化成均值 0、方差 1。
    统计量是训练集离线算好的（`cmvn.ark`），不是对当前音频在线统计，推理必须用同一份文件。
    """

    def __init__(self, kaldi_cmvn_file):
        """读取 cmvn.ark，缓存均值与逆标准差；只读盘一次，之后都是向量运算。"""
        # 原文变量名拼写如此，语义即 1/σ
        self.dim, self.means, self.inverse_std_variences = \
            self.read_kaldi_cmvn(kaldi_cmvn_file)

    def __call__(self, x, is_train=False):
        """逐维归一化 y = (x - mean) * (1 / std)；维度不符必须 assert 失败，广播会静默算错。"""
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        """解析 Kaldi CMVN 归档（(2, dim+1)），返回 (dim, means, inverse_std_variences)。
        第 0 行前 dim 列是 Σx，第 1 行前 dim 列是 Σx²，最后一列是帧数 count。
        """
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        # 方差下限：常数维方差为 0 时 1/sqrt(0) 会得到 inf，必须夹住
        floor = 1e-20
        means = []
        inverse_std_variences = []
        # 逐维手算，保持与 Kaldi apply-cmvn 一致的数值路径（不要改成 numpy 向量化）
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            # σ² = Σx²/count - μ²（总体方差，除以 count 而非 count-1）
            varience = (stats[1, d] / count) - mean*mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)


class KaldifeatFbank:
    """基于 kaldi_native_fbank（Kaldi C++ 实现）的 80 维 log-mel fbank 提取器，参数与
    Kaldi `compute-fbank-feats` 一致（25ms/10ms、snip_edges=True、不含能量项、不做 VTLN）。
    选它而非 torchaudio 是为了与训练端 Kaldi 特征逐比特对齐，避免窗函数/梅尔刻度实现差异。
    """

    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10,
                 dither=1.0):
        """构造并缓存 FbankOptions；frame_length/frame_shift 没写进 opts，
        实际生效的是 knf.FbankOptions 的默认值 25ms/10ms（与本实现恰好一致）。
        """
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        # snip_edges=True 与 Kaldi 默认一致，代价是首尾最多损失约一帧信息
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        """从一段波形提取 (T, num_mel_bins) 的 float32 fbank；wav 可为路径 str 或
        (sample_rate, wav_np) 二元组，is_train 为 True 才启用 dither（推理必须为 False）。
        """
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        # 只接受单声道一维波形，多声道需要调用方先降混
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        # 先覆盖 opts 再建 OnlineFbank，dither 的修改才真正生效
        self.opts.frame_opts.dither = dither
        # 每次调用都新建 OnlineFbank，避免在线对象残留上次的帧状态导致跨调用串帧
        fbank = knf.OnlineFbank(self.opts)

        # .tolist() 是因为该 C++ 绑定只接受 Python list
        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            # kaldi_native_fbank 不报错，只是取不到帧；打印告警便于定位坏数据
            print("Check data, len(feat) == 0", wav, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat
