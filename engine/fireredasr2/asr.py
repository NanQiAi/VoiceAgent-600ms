# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""FireRedASR2 语音识别主类（vendored 上游代码；项目内只补注释，绝不改行为）。

对外只有 ``FireRedAsr2``（from_pretrained / transcribe）与 ``FireRedAsr2Config``（推理超参）。
在线链路（VAD → ASR → RAG → LLM）走的就是这里；项目适配层 engine/asr_firered.py 负责选卡并调用。
"""

import logging
import os
import re
import time
import traceback
from dataclasses import dataclass

import torch

from .data.asr_feat import ASRFeatExtractor
# 注意：上游 models/ 子包未随本仓库 vendored 进来，实际 import 需按上游目录结构补齐
from .models.fireredasr_aed import FireRedAsrAed
from .models.fireredasr_llm import FireRedAsrLlm
from .models.lstm_lm import LstmLm
from .models.param import count_model_parameters
from .tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from .tokenizer.llm_tokenizer import LlmTokenizerWrapper

logger = logging.getLogger(__name__)


# 字段顺序与 speech2text.py 里 FireRedAsr2Config(...) 的位置参数一一对应，增删/调序字段必须同步
@dataclass
class FireRedAsr2Config:
    """FireRedASR2 推理超参容器（dataclass）。

    所有字段都有默认值，AED 与 LLM 两种模式共用这一类；只对单模式生效的字段在字段注释里标注。
    """
    use_gpu: bool = True
    # 只对主干调用 .half()，外部 LSTM LM（elm）不加 half
    use_half: bool = False
    beam_size: int = 3
    nbest: int = 1
    # 0 = 自动：AED 用编码器输出帧数，LLM 用语音特征长度作上限
    decode_max_len: int = 0
    # 仅 AED：log_softmax 温度除子，>1 让分布更平缓
    softmax_smoothing: float = 1.25
    # 仅 AED：GNMT 式长度归一化指数，抵消"长句累计 log-prob 更低"的偏置
    aed_length_penalty: float = 0.6
    # 仅 AED：<eos> 分数乘子，<1 抑制过早结束；上游有 assert 要求 >0
    eos_penalty: float = 1.0
    return_timestamp: bool = False
    # 仅 LLM：类型标注写成 bool 但语义是 int，默认 0（不限制）；上游原样，不改
    decode_min_len: bool = 0
    repetition_penalty: float = 1.0
    llm_length_penalty: float = 0.0
    # 仅 LLM：上游固定 do_sample=False，温度缩放只在采样路径生效，此用法下通常不起作用
    temperature: float = 1.0
    # 外部 LSTM LM 目录（需含 model.pth.tar）；空串 = 不启用
    elm_dir: str = ""
    # 外部 LM 的融合权重（t_scores + elm_weight * elm_logprob）；0.0 = 不融合
    elm_weight: float = 0.0

    def __post_init__(self):
        """dataclass 初始化后钩子，上游当前留空（预留的校验/联动修正位置）。"""
        pass


class FireRedAsr2:
    """FireRedASR2 推理入口主类（本项目唯一使用的类）。

    用法：``from_pretrained(asr_type, model_dir, config)`` 加载权重，
    ``transcribe(batch_uttid, batch_wav_path)`` 推理。

    asr_type 两种模式："aed" = Conformer + Transformer + CTC，beam search 可挂外部 LSTM LM /
    可选字级时间戳；"llm" = 同一编码器 + Adapter 接 Qwen2-7B-Instruct 走 HF generate()。
    模型是有状态 nn.Module，通常多会话共享；transcribe 只读（@torch.no_grad）但不保证线程安全，
    并发保护在调用方 engine/asr_firered.py（唯一临时文件名 + 锁）。
    """

    @classmethod
    def from_pretrained(cls, asr_type, model_dir, config=FireRedAsr2Config()):
        """按 asr_type 加载对应架构的权重与 tokenizer，返回已 eval() 的 FireRedAsr2 实例。

        model_dir 需含 cmvn.ark；AED 另需 model.pth.tar / dict.txt / train_bpe1000.model，
        LLM 另需 model.pth.tar / asr_encoder.pth.tar / Qwen2-7B-Instruct/。
        asr_type 非法或 config.elm_dir 路径不存在时会 assert 失败；会读数百 MB ~ 数 GB 的
        checkpoint 并常驻内存/显存。config 的默认值在 import 时只构造一次，会被多次调用共享。
        """
        assert asr_type in ["aed", "llm"]

        # CMVN = 倒谱均值方差归一化（Kaldi 导出的统计量，对 fbank 逐维归一）。
        # 特征契约：80 维 fbank、帧长 25ms / 帧移 10ms（见 data/asr_feat.py）；两种模式共用
        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        # ---- AED 分支：Conformer 编码器 + Transformer 解码器 + CTC 头 ----
        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            # dict.txt：汉字单字与 <sos>/<eos>/<blank>/<sil> 等特殊标记的 id 映射
            dict_path =os.path.join(model_dir, "dict.txt")
            # 英文部分用的 SentencePiece BPE 模型（1000 词表规模）
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        # ---- LLM 分支：Conformer 编码器 + Adapter + Qwen2-7B 因果语言模型 ----
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            # 编码器权重单独存放，不在 LLM 的 model.pth.tar 里
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir)
        elm = None
        if config.elm_dir:
            # 路径不存在会让后续 torch.load 以更难懂的方式失败，这里先 assert
            assert os.path.exists(config.elm_dir), f"{config.elm_dir}"
            model_path = os.path.join(config.elm_dir, "model.pth.tar")
            elm = load_lstm_lm(model_path)
            # 外部 LM 也要 eval()，否则 dropout 让打分抖动
            elm.eval()
            logger.info(elm)
        # 打印参数量是"权重是否真的加载成功"的第一道肉眼检查
        count_model_parameters(model)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer, elm, config)

    def __init__(self, asr_type, feat_extractor, model, tokenizer, elm, config):
        """保存各组件引用，并按 config 把模型搬到目标设备/精度（可能就地对 model 调 half()/cuda）。

        会读当前 CUDA 设备并缓存到 self._gpu_device，之后改 torch 默认设备也不影响它；
        因此调用方必须在 from_pretrained 之前 torch.cuda.set_device。
        """
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.elm = elm
        self.config = config
        # 不用 GPU 时置 None，后续 .cuda() 分支不会被走到
        self._gpu_device = torch.cuda.current_device() if self.config.use_gpu else None
        logger.info(self.config)
        if self.config.use_gpu:
            if self.config.use_half:
                self.model.half()
            # 搬到 self._gpu_device 指定的那张卡（多卡时避免默认卡错位）
            self.model.cuda(self._gpu_device)
            # 外部 LM 必须搬同一张卡，否则打分时设备不一致
            if self.elm:
                self.elm.cuda(self._gpu_device)
        else:
            self.model.cpu()

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path):
        """对一批音频做识别，返回逐条结果 dict（本类唯一推理入口，同步阻塞）。

        batch_wav_path 元素是 wav 路径（str，Kaldi 风格读取，16k 单声道为宜）或
        ``(sample_rate, np.ndarray)`` 波形；同批长度不一致会 pad 到最长（短音频尾部补 0.0）。
        返回 ``List[dict]``：uttid、text（已 lower()，AED 另去 <blank>/<sil>）、rtf（4 位小数字符串）；
        confidence/dur_s 仅 AED，路径输入回填 wav，AED 且开 return_timestamp 时才有 timestamp。
        异常策略（上游原样）：AED 解码抛异常返回 ``[]``；特征提取失败/全为静音返回等长
        ``{"text": ""}`` 占位；异常栈打到 stderr 后吞掉，保证上层服务不因一条坏音频整体崩掉。
        """
        # 备份原始 uttid：特征提取可能跳过无效样本，出错时要用它生成等长占位结果
        batch_uttid_origin = batch_uttid
        try:
            feats, lengths, durs, batch_wav_path, batch_uttid = \
                self.feat_extractor(batch_wav_path, batch_uttid)
            # feats 为 None：本批样本都太短/无有效帧（提取器里被 continue 掉）
            if feats is None:
                return [{"uttid": uttid, "text":""} for uttid in batch_uttid_origin]
        # 裸 except 会连 KeyboardInterrupt 之类的 BaseException 一起吞掉，属上游原样逻辑
        except:
            traceback.print_exc()
            return [{"uttid": uttid, "text":""} for uttid in batch_uttid_origin]
        # RTF 的分母：本批音频总时长（秒），两种分支共用
        total_dur = sum(durs)
        if self.config.use_gpu:
            # lengths 也要在 GPU 上（编码器用它构造 mask），两者必须同一设备
            feats, lengths = feats.cuda(self._gpu_device), lengths.cuda(self._gpu_device)
            # 半精度时特征同步降为 fp16，与 half() 后的权重 dtype 对齐
            if self.config.use_half:
                feats = feats.half()

        # ---- 分支一：AED（Conformer + Transformer 解码器，beam search） ----
        if self.asr_type == "aed":
            # 只统计解码耗时（不含 fbank 提取），RTF 反映的是模型本身的开销
            start_time = time.time()

            # 解码异常（显存不足、数值 NaN 等）不应让服务挂掉
            try:
                hyps = self.model.transcribe(
                    feats, lengths,
                    self.config.beam_size,
                    self.config.nbest,
                    self.config.decode_max_len,
                    self.config.softmax_smoothing,
                    self.config.aed_length_penalty,
                    self.config.eos_penalty,
                    self.config.return_timestamp,
                    self.elm,
                    self.config.elm_weight
                )
            except Exception as e:
                # 注意：变量 e 实际未被使用，仅作占位（上游原样）
                traceback.print_exc()
                # 空列表 → 下方 zip 不产出任何元素 → 返回 []
                hyps = []

            elapsed = time.time() - start_time
            # total_dur 为 0（空 batch）时按 0 处理避免除零
            rtf= elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp, dur in zip(batch_uttid, batch_wav_path, hyps, durs):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                # detokenize：汉字直接出字，英文 SPM 子词的 ▁ 会被还原成空格
                text = self.tokenizer.detokenize(hyp_ids)
                # 清掉解码序列里可能残留的 CTC blank 与静音标记 <sil>
                text = re.sub(r"(<blank>)|(<sil>)", "", text)
                results.append({"uttid": uttid, "text": text.lower(),
                    # confidence 是 1-best 各 token 概率的均值；dur_s 是音频时长（秒）
                    "confidence": round(hyp["confidence"].cpu().item(), 3),
                    "dur_s": round(dur, 3), "rtf": f"{rtf:.4f}"})
                # 只有入参确实是文件路径时才回填 wav（数组输入没有可写出的路径）；
                # 此处用 type(wav) == str 而非 isinstance，是上游原样写法
                if type(wav) == str:
                    results[-1]["wav"] = wav
                if self.config.return_timestamp:
                    results[-1]["timestamp"] = self._get_and_fix_timestamp(hyp, hyp_ids, dur)
            return results

        # ---- 分支二：LLM（Conformer + Adapter + Qwen2-7B，HF generate） ----
        elif self.asr_type == "llm":
            # origin_texts 传 batch 个空串（转写任务的答案位置为空）；decode=True 时上游 chat
            # template 只保留 user 段 "<speech>请转写音频为文字"，max_len=128 足够容纳该固定 prompt
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if self.config.use_gpu:
                # 这里用默认当前卡而非 self._gpu_device，靠调用方先 set_device 对齐
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            # generate 失败时不打栈（与 AED 分支不同），静默降级为空文本
            try:
                generated_ids = self.model.transcribe(
                    feats, lengths, input_ids, attention_mask,
                    self.config.beam_size,
                    # 0 = 用语音特征长度作 max_new_tokens
                    self.config.decode_max_len,
                    self.config.decode_min_len,
                    self.config.repetition_penalty,
                    self.config.llm_length_penalty,
                    self.config.temperature
                )
                # 跳过 <|im_start|>/<|im_end|>/<speech> 等特殊 token
                texts = self.tokenizer.batch_decode(generated_ids,
                                                    skip_special_tokens=True)
            except Exception as e:
                # 此处刻意不打印异常栈，属上游原样行为，线上排查时需留意
                texts = []

            elapsed = time.time() - start_time
            rtf= elapsed / total_dur if total_dur > 0 else 0
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({"uttid": uttid, "text": text.lower(),
                                "rtf": f"{rtf:.4f}"})
                if type(wav) == str:
                    results[-1]["wav"] = wav
            return results

        # 两个分支都没命中时函数隐式返回 None；
        # 正常情况下 from_pretrained 的 assert 已保证 asr_type ∈ {"aed","llm"}，走不到这里

    def _get_and_fix_timestamp(self, hyp, hyp_ids, dur):
        """把 1-best 的 token 序列对齐成字级时间戳，并修掉越界/贴末尾的时间点。

        返回 ``List[[token, start, end]]``（token 已 lower()，时间保留 3 位小数），英文 SPM 子词
        会被合并成整词。hyp 缺少 "timestamp"（或为 None）时退化为按 token 数均匀切分。
        会通过 logger.info 打印被修正的 start/end 前后值。
        """
        # 时间统一保留 3 位小数（毫秒级），减小 JSON 体积
        r3 = lambda x: round(x, 3)
        if "timestamp" not in hyp or hyp["timestamp"] is None:
            timestamp = []
            # 每个 token 平均占用的时长；token 数为 0 时按 0 处理避免除零
            avg_dur = dur / len(hyp_ids) if len(hyp_ids) > 0 else 0
            last_end = dur
            for i, hyp_id in enumerate(hyp_ids):
                # join_symbol="" 且不做 ▁ 替换，好让 merge_spm_timestamp 靠 ▁ 前缀识别英文词边界
                token = self.tokenizer.detokenize([hyp_id], "", False)
                start = min(max(0, i*avg_dur), last_end)
                end = min((i+1)*avg_dur, dur)
                last_end = end
                timestamp.append([token.lower(), r3(start), r3(end)])
        else:
            starts, ends = hyp["timestamp"]
            timestamp = []
            last_end = dur
            # 经验偏移量，补偿帧中心/下采样带来的系统性提前量。
            # 注意：原注释写 "shift 40ms"，但 0.06 秒实际是 60ms，注释与数值不一致（上游原样，不改）
            SHIFT = 0.06  # shift 40ms
            for hyp_id, start, end in zip(hyp_ids, starts, ends):
                token = self.tokenizer.detokenize([hyp_id], "", False)
                start = min(max(0, start - SHIFT), last_end)
                end = min(max(0, end - SHIFT), dur)
                last_end = end
                timestamp.append([token.lower(), r3(start), r3(end)])
        # Fix case: start == dur and end == dur
        # 从后往前扫描，避免修改影响尚未处理的元素
        for i in range(len(timestamp)):
            idx = -(i+1)
            _, start, end = timestamp[idx]
            # 起点几乎贴着音频末尾（容差 1ms）说明对齐退化，往左挪一点
            if abs(dur - start) < 0.001:
                logger.info(f"start before {timestamp[idx]}")
                timestamp[idx][1] = dur - (i+1)*0.001
                logger.info(f"start after {timestamp[idx]}")
            # 终点贴末尾同理；i != 0 是因为最后一项的 end 本来就该等于 dur
            if i != 0 and abs(dur - end) < 0.001:
                logger.info(f"end before {timestamp[idx]}")
                timestamp[idx][2] = dur - i*0.001
                logger.info(f"end after {timestamp[idx]}")
        # 把拆开的英文 SPM 子词合并成整词，并顺带去掉 ▁ 前缀
        timestamp = self.tokenizer.merge_spm_timestamp(timestamp)
        return timestamp


def load_fireredasr_aed_model(model_path):
    """读取 AED 打包权重（model.pth.tar 内含 args 与 model_state_dict）并复原模型。

    权重先落到 CPU 内存，不直接占显存；返回尚未 eval() 的裸模型；文件缺失或反序列化失败时
    由 torch.load 抛出。
    """
    # weights_only=False：checkpoint 里打包了 argparse.Namespace 等非张量对象，
    # 新版 torch 默认的 weights_only=True 会拒绝反序列化
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    model = FireRedAsrAed.from_args(package["args"])
    # strict=False：容忍权重与结构间少量键缺失/多余（兼容上游历史版本差异）
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    """读取 LLM 打包权重，注入编码器与底座 LLM 路径，复原模型与配套 Qwen tokenizer。"""
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    # 训练时保存的是旧路径，必须注入当前环境的路径，
    # 否则 FireRedAsrLlm.from_args 会按训练机的旧路径去找编码器与底座 LLM
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    # 会在底座词表里追加 <speech> 特殊 token（见 llm_tokenizer.py）
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer


def load_lstm_lm(model_path):
    """读取外部 LSTM LM 权重并复原 LstmLm（用于 AED beam 浅融合）；返回尚未 eval() 的裸模型。"""
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    model = LstmLm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model
