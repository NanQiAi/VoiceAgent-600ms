# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""FireRedASR2 **AED 分支**专用 tokenizer：中文按字、英文按 SPM 子词。
`ChineseCharEnglishSpmTokenizer` 提供"文本 ↔ token ↔ id"双向转换（asr.py 用 detokenize
出转写、用 merge_spm_timestamp 合并子词时间戳）；词表由 data/token_dict.py 读取。
本文件是 vendored 上游代码，行为不得改动。
"""

import logging
import re

import sentencepiece as spm

from ..data.token_dict import TokenDict


class ChineseCharEnglishSpmTokenizer:
    """中文：一个汉字一个 token；英文：一个 SPM 子词一个 token（子词必须已在 dict 词表中，
    否则落到 <unk>）。中文之间的空格编码时丢弃，英文之间的空格由 "▁" 前缀表达。
    """

    SPM_SPACE = "▁"  # sentencepiece 词首标记（U+2581），detokenize 时还原成普通空格

    def __init__(self, dict_path, spm_model, unk="<unk>", space="<space>"):
        """初始化词表与 sentencepiece 编码器；spm_model 传空串/None 时退化为英文字符级。"""
        self.dict = TokenDict(dict_path, unk=unk)
        self.space = space
        if spm_model:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model)  # 老式 API 名（新版为 LoadFromFile），与上游保持一致
        else:
            self.sp = None
            print("[WRAN] Not set spm_model, will use English char")  # 上游拼写错误（WRAN），原样保留
            print("[WARN] Please check how to deal with ' '(space)")
            if self.space not in self.dict:
                # 词表缺 <space> 时英文词间空格会映射成 <unk>，英文可读性直接变差
                print("Please add <space> to your dict, or it will be <unk>")

    def tokenize(self, text, replace_punc=True):
        """把整段文本编码成 (tokens, tokens_id)：先 text.upper()，中文单字直接成 token，
        其余片段交给 SPM 编子词，未登录词落到 unk 的 id。
        """
        #if text == "":
        #    logging.info(f"empty text")
        # ↑ 上游遗留的调试代码，原样保留（含缩进与注释符）
        text = text.upper()
        tokens = []
        if replace_punc:
            # 字符组里的 "." 写了两遍（上游历史写法）
            text = re.sub("[，。？！,\.?!]", " ", text)
        # 捕获组 split：汉字作为独立元素保留，于是 parts 里"连续非汉字片段"与"单个汉字"交替出现
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')
        parts = pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        for part in parts:
            if pattern.fullmatch(part) is not None:
                tokens.append(part)
            else:
                if self.sp:
                    for piece in self.sp.EncodeAsPieces(part.strip()):
                        tokens.append(piece)
                else:
                    for char in part.strip():
                        tokens.append(char if char != " " else self.space)
        tokens_id = []
        for token in tokens:
            # 查不到时回退到 unk 的 id（TokenDict.get 允许默认值传字符串）
            tokens_id.append(self.dict.get(token, self.dict.unk))
        return tokens, tokens_id

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        """inputs is ids or tokens, do not need self.sp

        把 id 列表或 token 列表拼回可读文本；replace_spm_space=False 时保留 "▁" 前缀，
        供 asr.py::_get_and_fix_timestamp 判断。
        """
        if len(inputs) > 0 and type(inputs[0]) == int:
            tokens = [self.dict[id] for id in inputs]
        else:
            tokens = inputs
        s = f"{join_symbol}".join(tokens)  # f-string 包一层，兼容 join_symbol 传 None
        if replace_spm_space:
            s = s.replace(self.SPM_SPACE, ' ').strip()
        return s

    def merge_spm_timestamp(self, timestamp):
        """把被 SPM 拆散的英文子词时间戳合并成整词（仅 `return_timestamp=True` 时用到）。
        以 "▁" 开头者为词首，向右吸收连续的纯字母子词（正则 `^[a-zA-Z']+$`），end 取最后
        一个子词的结束时间；中文一字一 token 天然对齐，原样透传。
        """
        merged_timestamp = []
        i = 0
        while i < len(timestamp):
            token, start, end = timestamp[i]
            if token.startswith(self.SPM_SPACE):
                # 以 "▁" 开头：英文单词首个子词，开始向后吸收
                token = token.replace(self.SPM_SPACE, "")
                current_end = end
                next_i = i + 1
                while next_i < len(timestamp):
                    next_token, next_start, next_end = timestamp[next_i]
                    if re.match("^[a-zA-Z']+$", next_token):
                        token += next_token
                        current_end = next_end
                        next_i += 1
                    else:
                        # 遇到中文/标点/下一个 "▁" 词首：单词结束
                        break
                end = current_end
                i = next_i
            else:
                i += 1
            merged_timestamp.append((token, start, end))
        return merged_timestamp
