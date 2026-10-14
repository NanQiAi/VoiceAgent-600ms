# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""LLM 式解码分支专用的 tokenizer 封装（FireRedASR2 上游代码，只加注释不改行为）。

LLM 分支复用 `model_dir/Qwen2-7B-Instruct` 自带的 HuggingFace tokenizer，音频不进词表，
只用一个 `<speech>` 占位符代表"这里放了一段语音特征"；本文件负责构造 Qwen chat 格式的
输入、对齐 padding、生成 loss mask。AED 分支的 tokenizer 见同目录 aed_tokenizer.py。
"""

import re

import torch
from transformers import AutoTokenizer
# 只借用 LabelSmoother 的 ignore_index 常量（=-100），不做标签平滑
from transformers.trainer_pt_utils import LabelSmoother

# 语音占位符：文本侧只出现这一个 token 来代表整段音频，音频特征由 ASR encoder 单独注入
DEFAULT_SPEECH_TOKEN = "<speech>"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class LlmTokenizerWrapper:
    """LLM 解码分支的 tokenizer 工具类：方法都是 @classmethod，不需要实例化。"""

    @classmethod
    def build_llm_tokenizer(cls, llm_path, use_flash_attn=False):
        """加载 LLM 自带的 HF tokenizer，并追加 `<speech>` 特殊 token。

        会从磁盘加载（首次可能触发联网下载），并**就地修改** tokenizer 的词表大小与
        padding_side。词表加了新 token 之后，配套的 LLM 词嵌入必须同步 resize，否则
        `<speech>` 会越界（由模型加载侧负责，本文件不做校验）。
        """
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        # padding 侧必须与 attention 实现匹配
        if use_flash_attn:
            # FlashAttention 要求左侧 padding，否则右侧 pad 会污染注意力
            tokenizer.padding_side = "left"
        else:
            tokenizer.padding_side = "right"
        # 注册成"额外特殊 token"，确保它不会被 BPE 切成多个子词、也不会被 clean 掉
        special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer

    @classmethod
    def clean_text(cls, origin_text):
        """remove punc, remove space between Chinese and keep space between English.

        这是训练标签的归一化口径（推理出的文本也应与之同分布）；不做 ITN，数字/符号保持原样。
        """
        text = re.sub("[，。？！,\.!?《》（）\·“”、\\/]", "", origin_text)
        text = re.sub("\s+", " ", text)

        # 以 CJK 字符为捕获组切分，中文单字会成为独立片段
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')  # Chinese
        parts = pattern.split(text.strip())
        # 丢掉去空白后为空的片段——正是这些片段消灭了中文之间的空格，英文词间空格不受影响
        parts = [p for p in parts if len(p.strip()) > 0]
        text = "".join(parts)
        text = text.strip()

        text = text.lower()
        return text

    @classmethod
    def preprocess_texts(cls, origin_texts, tokenizer, max_len, decode=False):
        """把一批文本按 Qwen chat 模板编码成模型输入，并生成 loss 用的 target 与 mask。

        返回 (input_ids, attention_mask, target_ids, clean_texts)。三者都是 [B, L] 张量：
        input_ids 与 attention_mask 供前向使用，target_ids 与 input_ids 同值，但 padding 位
        与 prompt 位被置为 -100（CrossEntropy 自动跳过）。decode=True 是推理——assistant
        内容置空且模板末尾不闭合，作为 generation prompt；tokenizer 必须是同一个，`<speech>`
        才对得上。只创建 CPU 张量，不写文件、不发网络请求。
        """
        messages = []
        clean_texts = []
        for i, origin_text in enumerate(origin_texts):
            text = cls.clean_text(origin_text)
            clean_texts.append(text)
            # 推理时清空 assistant 内容，只留 "<|im_start|>assistant\n" 作为生成起点
            text = text if not decode else ""
            message = [
                {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"},
                {"role": "assistant", "content": text},
            ]
            messages.append(message)

        texts = []
        # 两套模板的唯一区别：最后一条消息之后要不要补 <|im_end|>
        if not decode:
            TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        else:
            TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
        for i, msg in enumerate(messages):
            texts.append(
                tokenizer.apply_chat_template(
                    msg,
                    tokenize=True,
                    chat_template=TEMPLATE,
                    add_generation_prompt=False,
                    padding="longest",
                    max_length=max_len,
                    truncation=True,
                )
            )

        max_len_texts = max([len(text) for text in texts])
        # padding 侧必须与 build_llm_tokenizer 中设置的策略一致
        if tokenizer.padding_side == "right":
            texts = [
                text + [tokenizer.pad_token_id] * (max_len_texts - len(text))
                for text in texts
            ]
        else:
            texts = [
                [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
                for text in texts
            ]
        input_ids = torch.tensor(texts, dtype=torch.int)

        target_ids = input_ids.clone()
        # padding 位不参与 loss，统一置为 ignore_index(-100)
        target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

        # 屏蔽 prompt 部分，只对 assistant 的回答算 loss
        mask_prompt = True
        if mask_prompt:
            mask_indices = torch.where(
                input_ids == tokenizer.convert_tokens_to_ids("assistant")
                )
            for i in range(mask_indices[0].size(0)):
                row = mask_indices[0][i]
                col = mask_indices[1][i]
                # 把该行 [0, col+2) 置为忽略：<|im_start|>assistant 及其后的换行 token，
                # +2 正是为了吃掉这个换行，剩余未被屏蔽的部分才是助手回答
                target_ids[row, : col + 2] = IGNORE_TOKEN_ID

        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        target_ids = target_ids.type(torch.LongTensor)
        input_ids = input_ids.type(torch.LongTensor)
        return input_ids, attention_mask, target_ids, clean_texts
