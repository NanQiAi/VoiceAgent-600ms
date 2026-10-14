"""FireRedASR2 的音频清单解析与 TextGrid 时间戳导出工具（上游 vendored 代码，不做推理）。

项目主链路（engine/asr_firered.py → fireredasr2/asr.py）不经过本文件；它主要服务于上游
命令行入口 speech2text.py（批量离线转写/评测）与 sft/、data/ 下的数据集制作脚本。
"""
# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import glob
import os
import logging
logger = logging.getLogger(__name__)

from textgrid import TextGrid, IntervalTier


def get_wav_info(args):
    """
    Returns:
        wavs: list of (uttid, wav_path)
    """
    # uttid = basename 去掉 ".wav"（只替换第一处），评测时用它对齐参考文本
    base = lambda p: os.path.basename(p).replace(".wav", "")
    if args.wav_path:
        wavs = [(base(args.wav_path), args.wav_path)]
    # sorted 保证同样入参每次跑的顺序一致，结果可复现
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = [(base(p), p) for p in sorted(args.wav_paths)]
    elif args.wav_scp:
        wavs = [line.strip().split() for line in open(args.wav_scp)]
        # 按时长降序排：batch 内长度相近可以减少 padding 浪费
        if args.sort_wav_by_dur:
            logger.info("Sort wav by duration...")
            # 约定 utt2dur 与 wav.scp 同目录，每行 "uttid 秒数"
            utt2dur = os.path.join(os.path.dirname(args.wav_scp), "utt2dur")
            if os.path.exists(utt2dur):
                utt2dur = [l.strip().split() for l in open(utt2dur)]
                # 只保留字段数为 2 的合法行，防脏数据
                utt2dur = {l[0]: float(l[1]) for l in utt2dur if len(l) == 2}
                wavs = sorted(wavs, key=lambda x: -utt2dur[x[0]])
                logger.info("Sort Done")
            else:
                # 没有 utt2dur 就退化为不排序（保持 scp 原顺序），只记日志不报错
                logger.info(f"Not find {utt2dur}, un-sort")
    elif args.wav_dir:
        # **/*.wav + recursive=True 会连子目录一起收
        wavs = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        # glob 本身不保证顺序，sorted 保证跨机器/跨文件系统稳定
        wavs = [(base(p), p) for p in sorted(wavs)]
    else:
        # 四个入口都没给：直接报错，避免"空清单静默成功"这种难查的问题
        raise ValueError("Please provide valid wav info")
    logger.info(f"#wavs={len(wavs)}")
    return wavs


def write_textgrid(wav_path, wav_dur, event):
    """把逐 token 时间戳写成 Praat TextGrid（与 wav 同目录同名的 .TextGrid）。

    wav_dur 与 event 里的 start/end 单位都是秒，wav_dur 同时是时间轴上界。start == end 的
    零长度区间会被跳过（Praat 不允许空区间），区间裁剪到 [0, wav_dur] 以防时间戳越界。
    """
    textgrid_file = wav_path.replace(".wav", ".TextGrid")
    logger.info(f"Write {textgrid_file}")
    textgrid = TextGrid(maxTime=wav_dur)
    # 单层标注：层名 "token"，与 AED 模型的输出粒度对应
    tier = IntervalTier(name="token", maxTime=wav_dur)
    for token, start_s, end_s in event:
        # 零长度区间（常见于对齐失败的 token）跳过，否则 TextGrid 非法
        if start_s == end_s:
            logger.info(f"Write TG, skip start=end {start_s}")
            continue
        # 夹紧到时间轴范围：起止都可能因对齐误差出现负值或超过总时长
        start_s = max(start_s, 0)
        end_s = min(end_s, wav_dur)
        tier.add(minTime=start_s, maxTime=end_s, mark=token)
    textgrid.append(tier)
    textgrid.write(textgrid_file)
