#!/usr/bin/env python3

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""FireRedASR2 离线批量推理 CLI：读 wav（单文件/多文件/目录/scp）→ 批量识别 → 落盘 TSV/JSONL。

是**离线评测 / 批量转写**用的脚本，不属于在线服务链路：在线低延迟链路走 engine/asr_firered.py，
它直接调用同目录 asr.py 的 transcribe()，不会执行本脚本的 argparse。
"""

import argparse
import json
import logging
import os

from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
from fireredasr2.utils.io import get_wav_info, write_textgrid

# 全局日志配置：INFO 级，格式带时间戳 + 模块名 + 行号，方便批量跑时定位输出位置
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
# logger 名字与上游 fireredasr2.bin.speech2text 保持一致，便于按名字过滤日志
logger = logging.getLogger("fireredasr2.bin.speech2text")


parser = argparse.ArgumentParser()
# aed = 纯 ASR 自回归解码；llm = 语音编码器接 LLM 解码
parser.add_argument('--asr_type', type=str, required=True, choices=["aed", "llm"])
# 模型目录：需包含 cmvn.ark 以及对应架构的权重/词表文件
parser.add_argument('--model_dir', type=str, required=True)

# Input / Output
# 以下 5 个输入来源互斥，实际由 get_wav_info 按优先级挑选，最终统一成 [(uttid, wav_path)]
parser.add_argument("--wav_path", type=str)  # 单个 wav 文件
parser.add_argument("--wav_paths", type=str, nargs="*")  # 多个 wav（内部按路径排序保证可复现）
parser.add_argument("--wav_dir", type=str)  # 递归扫描目录下所有 .wav
parser.add_argument("--wav_scp", type=str)  # Kaldi 风格清单：每行 "uttid wav_path"
parser.add_argument("--sort_wav_by_dur", type=int, default=0)  # 仅 wav_scp 生效：1=按 utt2dur 降序
parser.add_argument("--output", type=str)  # 输出前缀；不填则只打日志不落盘

# Decode Options
parser.add_argument('--use_gpu', type=int, default=1)  # 1=GPU（默认），0=CPU
parser.add_argument('--use_half', type=int, default=0)  # 1=fp16，省显存、通常更快，精度略降
parser.add_argument("--batch_size", type=int, default=1)  # 在线链路固定 1 条，离线可调大提吞吐
parser.add_argument("--beam_size", type=int, default=1)  #  beam 宽度，1 即贪心解码（最低延迟）
parser.add_argument("--decode_max_len", type=int, default=0)  # 0=不限制（由模型内部定）
# FireRedASR-AED
parser.add_argument("--nbest", type=int, default=1)
parser.add_argument("--softmax_smoothing", type=float, default=1.0)
parser.add_argument("--aed_length_penalty", type=float, default=0.0)
parser.add_argument("--eos_penalty", type=float, default=1.0)
parser.add_argument("--return_timestamp", type=int, default=0)  # 1=返回每个 token 的起止时间（秒）
parser.add_argument("--write_textgrid", type=int, default=0)  # 1=写 Praat TextGrid（需 return_timestamp）
# AED External LM
parser.add_argument("--elm_dir", type=str, default="")  # 空=不做外部 LSTM LM 浅融合
parser.add_argument("--elm_weight", type=float, default=0.0)
# FireRedASR-LLM
parser.add_argument("--decode_min_len", type=int, default=0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--llm_length_penalty", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=1.0)


def main(args):
    """离线批量转写主流程：收集 wav 清单 → 构建配置与模型 → 按 batch_size 攒批转写 → 落盘。

    args.output 非空时写两个文件：``{output}``（uttid\\ttext 的 TSV）与 ``{output}.jsonl``；
    同时开 write_textgrid 与 return_timestamp 时还会在每条 wav 旁写 .TextGrid。
    本函数不捕获异常（模型加载失败、wav 不存在、输出路径不可写都会直接向上抛）；
    asr.py::transcribe 内部对特征提取和解码异常做了兜底，返回空文本而不是抛错。
    """
    wavs = get_wav_info(args)
    # 无 --output 时为 None，后续所有写操作都要判空
    fout = open(args.output, "w") if args.output else None
    foutl = open(args.output + ".jsonl", "w") if args.output else None

    # 位置参数顺序必须与 asr.py 中 FireRedAsr2Config 的 dataclass 字段顺序严格一致，写错会静默错配；
    # int 传入的 use_gpu/use_half 不做类型检查，后续按真值使用
    asr_config = FireRedAsr2Config(
            args.use_gpu,
            args.use_half,
            args.beam_size,
            args.nbest,
            args.decode_max_len,
            args.softmax_smoothing,
            args.aed_length_penalty,
            args.eos_penalty,
            args.return_timestamp,
            args.decode_min_len,
            args.repetition_penalty,
            args.llm_length_penalty,
            args.temperature,
            args.elm_dir,
            args.elm_weight
    )
    # 整个脚本最重的一步（读 cmvn.ark + torch.load 权重），只在开头执行一次
    model = FireRedAsr2.from_pretrained(args.asr_type, args.model_dir, asr_config)

    batch_uttid = []
    batch_wav_path = []
    for i, wav in enumerate(wavs):
        # get_wav_info 保证每个元素是二元组 (uttid, wav_path)
        uttid, wav_path = wav
        batch_uttid.append(uttid)
        batch_wav_path.append(wav_path)
        # 还没攒够一批、且不是最后一条 → 继续攒；反之落下去做推理
        if len(batch_wav_path) < args.batch_size and i != len(wavs) - 1:
            continue

        # 单条失败时 asr.py 会返回 {"uttid": ..., "text": ""} 占位，不会中断整批
        results = model.transcribe(batch_uttid, batch_wav_path)

        for result in results:
            logger.info(result)
            if fout is not None:
                # JSONL 保留完整 dict（confidence / dur_s / rtf 等），ensure_ascii=False 保留中文原文
                foutl.write(f"{json.dumps(result, ensure_ascii=False)}\n")
                fout.write(f"{result['uttid']}\t{result['text']}\n")
            # 需同时开 --return_timestamp 与 --write_textgrid，且 result 里确有 "timestamp" 键
            # （LLM 分支不产生时间戳）
            if args.write_textgrid and "timestamp" in result:
                write_textgrid(result["wav"], result["dur_s"], result["timestamp"])

        # 每批结束立刻 flush，避免长跑中途崩溃丢掉已识别好的结果
        if fout: fout.flush()
        if foutl: foutl.flush()
        batch_uttid = []
        batch_wav_path = []
    # 异常路径不会执行到这里，两个句柄不会被关闭，属已知的上游行为
    if fout: fout.close()
    if foutl: foutl.close()


if __name__ == "__main__":
    args = parser.parse_args()
    # 先把全部参数打进日志，方便复现某次评测的具体配置
    logger.info(args)
    main(args)
