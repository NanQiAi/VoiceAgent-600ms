# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

"""FireRedASR2 的词表加载工具（上游 vendored 代码，行为保持不变）。
TokenDict 维护 id2word（列表，下标即 id）与 word2id 两张表：每行形如 "词 id"（缺 id 时
用行号），空行占位成一个空格词，字面量 "<space>" 还原成空格；id 必须从 0 连续递增。
"""
import logging

logger = logging.getLogger(__name__)


class TokenDict:
    """词表对象：id2word（列表）与 word2id（字典）双向映射。
    支持 get / [] / len() / in；配置了 unk 时未登录词会回退到 unk 的 id。
    """

    def __init__(self, dict_path, unk=""):
        """读取词表并建立双向映射；unk 传空串表示不回退（查不到直接 KeyError），
        非空时该词必须已在词表中，否则 assert 失败。只读盘一次。
        """
        assert dict_path != ""
        # 列表下标 == id，这是 __getitem__ 能 O(1) 查词的前提
        self.id2word, self.word2id = self.read_dict(dict_path)
        self.unk = unk
        assert unk == "" or unk in self.word2id
        self.unkid = self.word2id[unk] if unk else -1   # -1 表示"无回退目标"

    def get(self, key, default):
        """按词查 id，查不到时返回 default；default 允许是 str（先翻译成该词的 id）。"""
        if type(default) == str:
            # 默认词不存在属于配置错误，应当立刻暴露
            default = self.word2id[default]
        return self.word2id.get(key, default)

    def __getitem__(self, key):
        """支持 d[词] 取 id、d[id] 取词；未配置 unk 且查不到抛 KeyError，int 越界抛 IndexError。"""
        if type(key) == str:
            # 配置了 unk 就启用回退，保证解码阶段不会因为生僻词崩掉
            if self.unk:
                return self.word2id.get(key, self.word2id[self.unk])
            else:
                return self.word2id[key]
        elif type(key) == int:
            return self.id2word[key]
        else:
            raise TypeError("Key should be str or int")

    def __len__(self):
        """返回词表大小（id2word 的长度，等于最大 id + 1）。"""
        return len(self.id2word)

    def __contains__(self, query):
        """支持 `词 in dictobj` / `id in dictobj`；与 __getitem__ 不同，这里不做 unk 回退。"""
        if type(query) == str:
            return query in self.word2id
        elif type(query) == int:
            return query in self.id2word
        else:
            raise TypeError("query should be str or int")

    def read_dict(self, dict_path):
        """解析词表文件，构建并返回 (id2word, word2id)。
        每行以空白切分：两列时第二列是显式 id，只有一列时用行号；空行/纯空格行视为一个空格
        词；字面量 "<space>" 替换成空格。断言强制 id 从 0 连续递增。
        """
        id2word, word2id = [], {}
        with open(dict_path, encoding='utf8') as f:
            for i, line in enumerate(f):
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    word, index = tokens[0], int(tokens[1])
                elif len(tokens) == 1:
                    word, index = tokens[0], i
                else:  # empty line or space
                    logger.info(f"Find empty line or space '{line.strip()}' in {dict_path}:L{i}, set to ' '")
                    word, index = " ", i
                # 断言 id 等于当前词条数 → 保证 id 从 0 连续且不重复
                assert len(id2word) == index
                assert len(word2id) == index
                if word == "<space>":
                    logger.info(f"NOTE: Find <space> in {dict_path}:L{i} and convert it to ' '")
                    word = " "
                word2id[word] = index
                id2word.append(word)
        assert len(id2word) == len(word2id)
        return id2word, word2id
