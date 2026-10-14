#!/usr/bin/env python3

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""FireRedASR2 评测口径核心：WER / CER / SER 计算与文本归一化（独立离线 CLI）。

**评测口径的唯一定义处**：改一处归一化规则 CER/WER 就会整体变动，故只补注释、不改行为；
在线链路不 import 本文件。中文按字切 = CER，英文/数字按词切 = WER，公式同为 (S+D+I)/N。
"""

import argparse
import re
from collections import OrderedDict


parser = argparse.ArgumentParser()
# --ref/--hyp 为 Kaldi text 格式 ``<uttid> <token>...``，两份必须使用同一批 uttid
parser.add_argument("--ref", type=str, required=True)
parser.add_argument("--hyp", type=str, required=True)
# --print_sentence_wer 逐句打印明细；--do_tn 用 cn2an 把阿拉伯数字转成中文；--rm_special 先删 <|...|>
parser.add_argument("--print_sentence_wer", type=int, default=0)
parser.add_argument("--do_tn", type=int, default=0, help="simple tn by cn2an")
parser.add_argument("--rm_special", type=int, default=0, help="remove <\|.*?\|>")


def main(args):
    """读 ref/hyp 两份文本 → 切 token → 逐句算编辑距离 → 打印全局统计（结果只走 stdout）。"""
    uttid2refs = read_uttid2tokens(args.ref, args.do_tn, args.rm_special)
    # hyp 侧归一化参数必须与 ref 完全一致，否则对比不公平
    uttid2hyps = read_uttid2tokens(args.hyp, args.do_tn, args.rm_special)
    uttid2wer_info, wer_stat, en_dig_stat = compute_uttid2wer_info(
        uttid2refs, uttid2hyps, args.print_sentence_wer)
    # 全局 WER/CER、S/D/I 分项与 SER，即评测报告里的那几行数字
    wer_stat.print()
    en_dig_stat.print()


def read_uttid2tokens(filename, do_tn=False, rm_special=False):
    """读 Kaldi text 文件，返回 ``OrderedDict[uttid → token 列表]``（键序 = 文件行序）。

    会打印进度；同一文件内 uttid 重复会 assert 失败。token 粒度见 ``text2tokens``。
    """
    print(f">>> Read uttid to tokens: {filename}", flush=True)
    # OrderedDict 保证输出顺序与文件行序一致，便于多次运行 diff
    uttid2tokens = OrderedDict()
    uttid2text = read_uttid2text(filename, do_tn, rm_special)
    for uttid, text in uttid2text.items():
        # ref 与 hyp 共用 text2tokens，确保归一化口径一致
        tokens = text2tokens(text)
        uttid2tokens[uttid] = tokens
    return uttid2tokens


def read_uttid2text(filename, do_tn=False, rm_special=False):
    """按行读 Kaldi text 文件，返回 ``OrderedDict[uttid → 文本串]``。

    只有 uttid 的行记为空串 ""（合法的"空参考/空识别"）；空行告警后跳过；uttid 重复直接 assert 失败。
    """
    uttid2text = OrderedDict()
    # 显式 utf8：语料含中文，避免平台默认编码差异
    with open(filename, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            cols = line.split()
            if len(cols) == 0:
                print("[WARN] empty line, continue", i, flush=True)
                continue
            assert cols[0] not in uttid2text, f"repeated uttid: {line}"
            if len(cols) == 1:
                uttid2text[cols[0]] = ""
                continue
            txt = " ".join(cols[1:])
            # 丢掉 <|...|> 片段（如 <|nospeech|>），剩余非空片段用单空格接回
            if rm_special:
                txt = " ".join([t for t in re.split("<\|.*?\|>", txt) if t.strip() != ""])
            # 可选数字文本归一化：把 "2026" 念成"二零二六"，否则数字会被逐个 token 判错
            if do_tn:
                import cn2an
                txt = cn2an.transform(txt, "an2cn")
            uttid2text[cols[0]] = txt
    return uttid2text


def text2tokens(text):
    """把一句文本归一化成 token 序列：**中文按字（= CER）、英文/数字按词（= WER）**。

    顺序：删 ``<unk>`` → 标点替换成空格（丢弃标点）→ 删 ``<.*>`` 标记 →
    ``fix_abbr_simple`` 合并逐字母拼读的缩写 → 按 CJK 边界切分、非 CJK 片段再按空白分词 →
    ``upper()``。因此大小写、标点、空格数量的差异都不计入错误。
    """
    # 标点表 = 中文全角 + 英文半角。末尾的 "\":" 也是被丢弃的字符，属上游历史写法，保持原样
    PUNCTUATIONS = "，。？！,\.?!＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·｡\":" + "()\[\]{}/;`|=+"
    if text == "":
        return []
    tokens = []

    # 只按精确字面量删 <unk>；此处早于标点替换，残留的 "unk" 会被当成英文词——上游既有行为
    text = re.sub("<unk>", "", text)
    text = re.sub(r"[%s]+" % PUNCTUATIONS, " ", text)
    # 删 <.*> 标记（<blank>/<sil> 等），放在标点处理之后避免 "<" ">" 先被清掉
    text = re.sub("<.*>", "", text)
    # 合并逐字母拼读的英文缩写 "u s b" → "usb"，避免一个词被算成多个词
    text = fix_abbr_simple(text)
    #pattern = re.compile(r'([\u4e00-\u9fff])')
    # CJK = 基本区汉字 + 平假名 + 片假名（含扩展小写假名），日文假名同样按"字"切
    pattern = re.compile(r'([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff])')
    parts = pattern.split(text.strip().upper())
    parts = [p for p in parts if len(p.strip()) > 0]
    for part in parts:
        if pattern.fullmatch(part) is not None:
            tokens.append(part)
        else:
            for word in part.strip().split():
                tokens.append(word)
    return tokens


def fix_abbr_simple(text):
    """把被逐字母拼读的英文缩写合并回一个词（"U S B" → "USB"）。

    文本被改动时会向 stdout 打印 ``WER FIX: '原' --> '新'``，便于人工核查归一化是否误伤。
    """
    ori_text = text
    # 5 个以上"单字母 + 空格"的连续拼读更像正常逐读/枚举，强行合并可能误伤，直接原样返回
    if re.search(r"(?<![a-zA-Z'])([a-zA-Z] ){5,}[a-zA-Z](?![a-zA-Z'])", text):
        return text
    
    prev = None
    # 每轮只吃掉"单字母 + 其后的空格"；零宽断言保证只合并"独立的单字母"，不动普通单词之间
    while prev != text:
        prev = text
        text = re.sub(r"(?<![a-zA-Z'])([a-zA-Z]) (?=[a-zA-Z](?![a-zA-Z']))", r'\1', text)
    
    if ori_text != text:
        print(f"WER FIX: '{ori_text}' --> '{text}'")
    return text


def compute_uttid2wer_info(refs, hyps, print_sentence_wer=False):
    """逐句算 WER 信息（含对齐路径）并汇总全局统计。

    只遍历 refs，**hyps 里缺失的 uttid 会被跳过**，故统计范围是 ref ∩ hyp（漏跑 ≠ 全错）。
    """
    print(f">>> Compute uttid to wer info", flush=True)

    uttid2wer_info = OrderedDict()
    wer_stat = WerStats()
    en_dig_stat = EnDigStats()

    # 以 ref 为主序遍历，顺序稳定，故 --print_sentence_wer 的输出可复现
    for uttid, ref in refs.items():
        # hyp 缺失（漏跑 / uttid 拼写不一致）→ 告警跳过，不计入分母
        if uttid not in hyps:
            print(f"[WARN] No hyp for {uttid}", flush=True)
            continue
        hyp = hyps[uttid]

        # hyp 比 ref 多 8 个 token 以上（疑似严重幻觉/重复解码）时打印上下文供排查；
        # 下方 "#continue" 被注释掉了，所以只告警不剔除；该条件不覆盖 hyp 偏短的情况
        if len(hyp) - len(ref) >= 8:
            print(f"[BidLengthDiff]: {uttid} {len(ref)} {len(hyp)}#{' '.join(ref)}#{' '.join(hyp)}")
            #continue

        # 单句最小编辑距离 DP + 回溯，得到 S/D/I/正确数与对齐路径 ali
        wer_info = compute_one_wer_info(ref, hyp)
        uttid2wer_info[uttid] = wer_info
        # count_english_ditgit 依赖对齐路径 ali，统计英文词/数字的细分正确率
        ns = count_english_ditgit(ref, hyp, wer_info)
        wer_stat.add(wer_info)
        en_dig_stat.add(*ns)
        if print_sentence_wer:
            print(f"{uttid} {wer_info}")

    return uttid2wer_info, wer_stat, en_dig_stat


# 三类编辑操作代价都取 3（sclite 等传统 ASR 评测工具的等权惯例）。指标用的是回溯数出的
# S/D/I 个数、不是 cost 本身，所以三者相等时取 3 与取 1 的 WER/CER 完全相同。
COST_SUB = 3
COST_DEL = 3
COST_INS = 3

# 回溯方向标记（每格记录"从哪个方向走来"）：0 正确 / 1 替换 / 2 删除（漏字）/ 3 插入（多字）
ALIGN_CRT = 0
ALIGN_SUB = 1
ALIGN_DEL = 2
ALIGN_INS = 3
# 4 结束标记；本实现未使用（回溯靠 i>0 or j>0 判停），保留以对齐上游常量表
ALIGN_END = 4


def compute_one_wer_info(ref, hyp):
    """最小编辑距离 DP + 回溯，返回单句 WerInfo。

    dp[i][j] = ref 前 i 个 token 与 hyp 前 j 个 token 的最小对齐代价；为回溯保留了整张表
    （时间/空间均 O(ref_len × hyp_len)，离线评测不追内存）。
    """
    ref_len = len(ref)
    # hyp 长度只影响 DP 表宽度，不影响 WER 分母
    hyp_len = len(hyp)

    # 单格结构体：cost = 到该格的最小对齐代价，align = 到达该格的方向标记 ALIGN_*
    class _DpPoint:
        def __init__(self, cost, align):
            self.cost = cost
            self.align = align

    dp = []
    for i in range(0, ref_len + 1):
        dp.append([])
        for j in range(0, hyp_len + 1):
            # i*j 只是占位初值：0 行/0 列恰好为 0，其余稍后被覆盖
            dp[-1].append(_DpPoint(i * j, ALIGN_CRT))

    # Initialize
    # 第 0 行：ref 为空 → 全部是插入；第 0 列：hyp 为空 → 全部是删除
    for i in range(1, hyp_len + 1):
        dp[0][i].cost = dp[0][i - 1].cost + COST_INS;
        dp[0][i].align = ALIGN_INS
    for i in range(1, ref_len + 1):
        dp[i][0].cost = dp[i - 1][0].cost + COST_DEL
        dp[i][0].align = ALIGN_DEL

    # DP
    # 三操作取最小：对角（相同=正确/不同=替换）、向上=删除、向左=插入；用严格小于比较，
    # 代价相同时优先级固定为 替换 > 删除 > 插入，回溯路径确定、结果可复现。
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            min_cost = 0
            min_align = ALIGN_CRT
            if hyp[j - 1] == ref[i - 1]:
                min_cost = dp[i - 1][j - 1].cost
                min_align = ALIGN_CRT
            else:
                min_cost = dp[i - 1][j - 1].cost + COST_SUB
                min_align = ALIGN_SUB

            del_cost = dp[i - 1][j].cost + COST_DEL
            if del_cost < min_cost:
                min_cost = del_cost
                min_align = ALIGN_DEL

            ins_cost = dp[i][j - 1].cost + COST_INS
            if ins_cost < min_cost:
                min_cost = ins_cost
                min_align = ALIGN_INS

            dp[i][j].cost = min_cost
            dp[i][j].align = min_align

    # Backtrace：从 dp[ref_len][hyp_len] 沿 align 倒推到 (0,0)。
    # crt=正确，sub=替换，ins=插入，det=删除（源码里 deletion 的计数名）。
    crt = sub = ins = det = 0
    i = ref_len
    j = hyp_len
    # 对齐路径，倒序累积：元素 (i, j, 操作)，i/j 是 **1-based** 下标（i 对应 ref 第 i 个 token），
    # count_english_ditgit 靠它按下标查询某参考 token 是否被正确识别
    align = []
    while i > 0 or j > 0:
        if dp[i][j].align == ALIGN_CRT:
            align.append((i, j, ALIGN_CRT))
            i -= 1
            j -= 1
            crt += 1
        elif dp[i][j].align == ALIGN_SUB:
            align.append((i, j, ALIGN_SUB))
            i -= 1
            j -= 1
            sub += 1
        elif dp[i][j].align == ALIGN_DEL:
            align.append((i, j, ALIGN_DEL))
            i -= 1
            det += 1
        elif dp[i][j].align == ALIGN_INS:
            align.append((i, j, ALIGN_INS))
            j -= 1
            ins += 1

    err = sub + det + ins
    # 回溯是倒序的，reverse 成正序（按 ref 从前往后）
    align.reverse()
    wer_info = WerInfo(ref_len, err, crt, sub, det, ins, align)
    return wer_info


class WerInfo:
    """单句 WER/CER 结果（含 S/D/I 分项、对齐路径与句级 WER）。

    属性名极短：r=参考 token 数（分母），e=错误数 s+d+i，c=正确数，s/d/i=替换/删除/插入数，
    ali=对齐路径 list[(i, j, op)]（1-based），wer=句级错误率（百分比）。
    """

    def __init__(self, ref, err, crt, sub, dele, ins, ali):
        self.r = ref
        self.e = err
        self.c = crt
        self.s = sub
        self.d = dele
        self.i = ins
        self.ali = ali
        # 除零保护：空参考句用 1 当分母，于是"参考为空、识别有内容"的句子 WER = 100 × ins，
        # 既避免 ZeroDivisionError/inf，也不会被静默算成 0%（属上游既有口径，不改）
        r = max(self.r, 1)
        # WER/CER 定义式：(S+D+I)/N × 100；直接加总 S/D/I，不用 DP 里的加权 cost
        self.wer = 100.0 * (self.s + self.d + self.i) / r

    def __repr__(self):
        s = f"wer {self.wer:.2f} ref {self.r:2d} sub {self.s:2d} del {self.d:2d} ins {self.i:2d}"
        return s


class WerStats:
    """多句 WER 的全局累加器，输出全局 WER/CER、S/D/I 分项与 SER。

    全局 WER = ``Σ(s+d+i) / Σr × 100``，即按 ref token 数加权的微平均，而不是"各句 WER 的
    算术平均"——长句影响更大，与 Kaldi/sclite 系工具口径一致。
    """

    def __init__(self):
        # 统一在 print() 时求和（不在 add 时累加），保证同一批数据可复算
        self.infos = []

    def add(self, wer_info):
        """追加一句的结果。"""
        self.infos.append(wer_info)

    def print(self):
        """汇总所有已 add 的句子，向 stdout 打印全局指标表（S/D/I 分项与 SER）。"""
        r = sum(info.r for info in self.infos)
        # 兜底：ref 全为空（r<=0）时把分母置 1，避免除零
        if r <= 0:
            print(f"REF len is {r}, check")
            r = 1
        s = sum(info.s for info in self.infos)
        d = sum(info.d for info in self.infos)
        i = sum(info.i for info in self.infos)
        se = 100.0 * s / r
        de = 100.0 * d / r
        ie = 100.0 * i / r
        wer = 100.0 * (s + d + i) / r
        # SER 分母 = 句数（至少 1 防除零），分子 = 至少有一个错误（e>0）的句子数
        sen = max(len(self.infos), 1)
        errsen = sum(info.e > 0 for info in self.infos)
        ser = 100.0 * errsen / sen
        print("-"*80)
        print(f"ref{r:6d} sub{s:6d} del{d:6d} ins{i:6d}")
        print(f"WER{wer:6.2f} sub{se:6.2f} del{de:6.2f} ins{ie:6.2f}")
        print(f"SER{ser:6.2f} = {errsen} / {sen}")
        print("-"*80)


class EnDigStats:
    """英文单词与数字的细分正确率累加器（中英混排 / 数字场景的观察项）。

    只统计参考里的英文词与数字串，**不能**与全局 WER/CER 直接相加减。
    """

    def __init__(self):
        self.n_en_word = 0
        self.n_en_correct = 0
        self.n_dig_word = 0
        self.n_dig_correct = 0

    def add(self, n_en_word, n_en_correct, n_dig_word, n_dig_correct):
        """累加一句的英文词 / 数字串计数。"""
        self.n_en_word += n_en_word
        self.n_en_correct += n_en_correct
        self.n_dig_word += n_dig_word
        self.n_dig_correct += n_dig_correct

    def print(self):
        """打印英文与数字的累计词数与正确数（比率留给读者用 correct/word 自行计算）。"""
        print(f"English #word={self.n_en_word}, #correct={self.n_en_correct}\n"
              f"Digit #word={self.n_dig_word}, #correct={self.n_dig_correct}")
        print("-"*80)


def count_english_ditgit(ref, hyp, wer_info):
    """统计参考句中英文词与数字串的数量，以及其中被完全正确识别的数量。

    命中英文/数字模式计入分母，再在 ``wer_info.ali`` 里找该参考位置的对齐项，只有 ALIGN_CRT
    （正确对齐）才计入分子。``hyp`` 参数保留但未使用；非 CJK 且非英文/数字的 token 会打印
    ``[WiredChar]`` 告警。返回 ``(n_en_word, n_en_correct, n_dig_word, n_dig_correct)``。
    """
    # 英文词模式：字母、点、连字符、撇号（如 "u.s."、"e-mail"、"don't"）；
    # 用 re.match（前缀匹配而非 fullmatch）是上游既有语义，保持不改
    patt_en = "[a-zA-Z\.\-\']+"
    patt_dig = "[0-9]+"
    # CJK 模式（仅基本区汉字）：用于排除中文，只统计英文与数字
    patt_cjk = re.compile(r'([\u4e00-\u9fff])')
    n_en_word = 0
    n_en_correct = 0
    n_dig_word = 0
    n_dig_correct = 0
    # 对齐路径元素 (ref下标, hyp下标, 操作)，下标为 1-based
    ali = wer_info.ali
    # enumerate 的 i 是 0-based，与 ali 的 1-based 下标比较时用 i+1
    for i, token in enumerate(ref):
        if re.match(patt_en, token):
            n_en_word += 1
            # 线性扫描找"ref 第 i+1 个 token"的对齐项；只认 ALIGN_CRT，即被替换/漏掉/多识别
            # 都不算正确；每个 ref 位置最多一项，找到即可 break
            for y in ali:
                if y[0] == i+1 and y[2] == ALIGN_CRT:
                    # j 只是把对应 hyp 下标取出（1-based 转 0-based），当前未再使用
                    j = y[1] - 1
                    n_en_correct += 1
                    break
        if re.match(patt_dig, token):
            n_dig_word += 1
            # 逻辑与英文词一致：数字串必须整串被正确对齐才算正确
            for y in ali:
                if y[0] == i+1 and y[2] == ALIGN_CRT:
                    j = y[1] - 1
                    n_dig_correct += 1
                    break
        if not re.match(patt_cjk, token) and not re.match(patt_en, token) \
           and not re.match(patt_dig, token):
            # 只告警不中断：这类字符不计入英文/数字统计，留给人工确认归一化是否漏网
            print("[WiredChar]:", token)
    return n_en_word, n_en_correct, n_dig_word, n_dig_correct


if __name__ == "__main__":
    args = parser.parse_args()
    # 先把参数打进日志，保证指标可复现（便于事后追溯口径）
    print(args, flush=True)
    main(args)
