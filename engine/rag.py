"""RAG 检索：bge 向量化 + FAISS 内积检索（向量已归一化，内积即余弦），可选 bge-reranker 重排。

级联 Pipeline 中的一环：ASR → 【RAG】→ LLM → TTS；`get_context()` 的输出填进
`config.SYSTEM_PROMPT_RAG` 的 `{context}`，命中为空则 LLM 走兜底话术。
"""
import os
import json
import time
import numpy as np
import faiss


class RAGEngine:
    """RAG 检索引擎：bge 向量化 + FAISS 检索 + 可选 rerank + 阈值过滤。

    典型用法（ws_server.py 启动时）：
        rag = RAGEngine(EMBED_MODEL_DIR, device=RAG_DEVICE, top_k=RAG_TOP_K,
                        score_threshold=RAG_SCORE_THRESHOLD).load()
        rag.build_index(docs)          # 或 rag.load_index(RAG_INDEX_PATH)
        ctx = rag.get_context(question)["context"]   # 拼进 system prompt

    模型推理在调用方线程同步执行（本类不自建线程池、不异步），耗时会计入首响延迟。
    """

    def __init__(self, embed_model_dir: str, device: str = "cuda:0",
                 reranker_model_dir: str = None, top_k: int = 3,
                 rerank_top_k: int = 5, score_threshold: float = 0.0):
        """只记配置，不加载模型、不占显存（真正加载在 `load()`）。

        `score_threshold` 是余弦相似度置信度阈值，0 表示不过滤；生产由
        `config.RAG_SCORE_THRESHOLD` 传 0.35。`reranker_model_dir` 为 None 时不 rerank。
        """
        # 模型与索引相关配置，只在 load()/build_index() 时使用
        self._embed_model_dir = embed_model_dir
        self._reranker_model_dir = reranker_model_dir
        self._device = device
        self._top_k = top_k
        self._rerank_top_k = rerank_top_k  # 送进 reranker 的候选数（须 ≥ top_k）
        self._score_threshold = score_threshold

        # 运行时状态：先置空，等 load() / build_index() / load_index() 逐步填充
        self._embedder = None          # SentenceTransformer 实例（bge 模型）
        self._reranker = None          # FlagReranker 实例；未配置或目录不存在时保持 None
        self._index = None             # faiss.IndexFlatIP，向量已归一化，内积即余弦
        self._documents = []           # 与索引行号一一对应的原始文档列表
        self._doc_embeddings = None    # 文档向量矩阵 (num_docs, dim)，供调试/重建

    def load(self):
        """加载 bge 向量模型与可选 reranker，返回 self 以便链式调用。

        副作用：从磁盘读权重并占用 device 指定的显存（reranker 用 fp16 省显存、提速）。
        方法内 import 是为了没装这两个可选重依赖时本模块仍可被 import。
        目录不存在、权重损坏或 device 不可用时由三方库抛异常。
        """
        # 延迟导入：避免其它模块仅 import engine.rag 时就被迫依赖 sentence-transformers
        from sentence_transformers import SentenceTransformer
        # 不做 warmup，首次 encode 会多含一次 CUDA 初始化开销
        self._embedder = SentenceTransformer(
            self._embed_model_dir,
            device=self._device,
        )

        # 只有显式配置了 reranker 且目录真实存在才加载，避免配置笔误导致启动失败
        if self._reranker_model_dir and os.path.isdir(self._reranker_model_dir):
            # FlagEmbedding 是 reranker 的可选依赖，只在需要时导入
            from FlagEmbedding import FlagReranker
            # use_fp16=True：568M 参数的 reranker 用半精度能把 top-5 重排压到 ~10-15ms
            self._reranker = FlagReranker(
                self._reranker_model_dir,
                use_fp16=True,
                device=self._device,
            )

        return self

    def build_index(self, documents: list[dict]):
        """把知识库文档编码成归一化向量并全量重建 FAISS 索引（不是增量追加）。

        每条 doc 形如 {'id','question','answer','text'(可选)}；无 'text' 时用
        `question + " " + answer` 作为被编码的文本，`texts` 与 documents 同序同长。
        返回统计 dict：num_docs / dim / encode_ms / encode_per_doc_ms。
        副作用：覆盖 `_documents/_doc_embeddings/_index`，不写磁盘（持久化另调 `save_index()`）。
        未先 `load()` 会 AttributeError；documents 为空列表时末尾除法会 ZeroDivisionError。
        """
        self._documents = documents
        texts = []
        for doc in documents:
            # 优先用显式 text 字段；否则退化为 question + answer 拼接（客服知识库常用口径）
            text = doc.get("text", f"{doc.get('question', '')} {doc.get('answer', '')}")
            texts.append(text.strip())

        t0 = time.perf_counter()
        # batch_size=64 是吞吐与显存的经验折中；normalize_embeddings=True 让内积恒等于余弦，
        # 这样 FAISS 用 IndexFlatIP 就能算余弦
        embeddings = self._embedder.encode(
            texts, batch_size=64, show_progress_bar=False,
            normalize_embeddings=True,
        )
        encode_ms = (time.perf_counter() - t0) * 1000

        self._doc_embeddings = embeddings.astype(np.float32)  # FAISS 只接受 float32
        dim = embeddings.shape[1]  # bge-small-zh-v1.5 为 512，bge-m3 为 1024
        # IndexFlatIP = 精确暴力内积检索，无近似误差；知识库仅几十条，暴力搜索耗时在微秒级，
        # 不需要 IVF/HNSW 等近似索引（它们还会引入召回损失与额外参数）
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._doc_embeddings)

        return {
            "num_docs": len(documents),
            "dim": dim,
            "encode_ms": encode_ms,
            "encode_per_doc_ms": encode_ms / len(documents),
        }

    def query(self, question: str, top_k: int = None) -> dict:
        """检索主流程：编码问句 → FAISS top-k →（可选）reranker 重排 → 截断到 top_k → 阈值过滤。

        返回 dict：results（每条形如原文档 + 'score'/'rank'，有 reranker 时还有
        'rerank_score'，已按最终得分降序且已过滤）、embed_ms / search_ms / rerank_ms / total_ms。
        未 `load()` 或未建索引时 AttributeError。
        """
        k = top_k or self._top_k
        # 有 reranker 时先多召回候选再精排；max 保证候选数不少于最终要返回的 k 条
        rerank_k = max(k, self._rerank_top_k) if self._reranker else k

        t0 = time.perf_counter()
        # encode 接受列表以复用批处理接口；同样必须归一化，才能和索引里的内积口径一致
        q_emb = self._embedder.encode(
            [question], normalize_embeddings=True,
        ).astype(np.float32)
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        # search 返回 (scores, indices)，形状 (1, rerank_k)；indices 中 -1 表示候选不足
        scores, indices = self._index.search(q_emb, rerank_k)
        search_ms = (time.perf_counter() - t1) * 1000

        candidates = []
        # i 同时充当 FAISS 原始排名（从 0 开始）
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS 在"库内文档少于 rerank_k"时填充的无效槽位
                continue
            # copy() 防止把 score/rank 写回 self._documents，污染原始知识库
            doc = self._documents[idx].copy()
            doc["score"] = float(score)  # 余弦相似度（已归一化内积），越高越相关
            doc["rank"] = i  # 记录召回排名，方便对比 rerank 前后的顺序变化
            candidates.append(doc)

        rerank_ms = 0.0  # 无 reranker 时保持 0.0，保证返回结构字段稳定
        if self._reranker and candidates:
            t2 = time.perf_counter()
            # 重排用 [问句, 答案] 对；没答案时退化用 text，避免把 None 传给 reranker
            pairs = [[question, c.get("answer", c.get("text", ""))] for c in candidates]
            rerank_scores = self._reranker.compute_score(pairs)
            # compute_score 在只有 1 个 pair 时返回标量而非列表，统一成可迭代防 TypeError
            if isinstance(rerank_scores, (int, float)):
                rerank_scores = [rerank_scores]
            rerank_ms = (time.perf_counter() - t2) * 1000

            # rerank_score 是模型 logit，与余弦 score 不同量纲
            for c, rs in zip(candidates, rerank_scores):
                c["rerank_score"] = float(rs)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        results = candidates[:k]
        # score_threshold > 0 才启用过滤；等于 0 表示"不过滤"（本文件默认值）
        if self._score_threshold > 0:
            # 有 reranker 时按 rerank_score 过滤，否则按余弦相似度过滤
            score_key = "rerank_score" if self._reranker else "score"
            # 低于阈值直接丢弃：宁可不注入知识库，也不给 LLM 不相干的材料（防幻觉），
            # 知识库为空时 LLM 会走"这个我问下同事，再给您回复哈"的兜底话术。
            # 注意：配了 reranker 时比较的是 reranker 打分（logit 量纲，可为负、可大于 1），
            # 此时 config.RAG_SCORE_THRESHOLD=0.35 是按余弦标定的值，量纲不同、必须重新标定。
            results = [r for r in results if r.get(score_key, 0) >= self._score_threshold]
        # 总耗时口径包含 embed + search + rerank，供 ws_server 打到延迟日志里
        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "results": results,
            "embed_ms": embed_ms,
            "search_ms": search_ms,
            "rerank_ms": rerank_ms,
            "total_ms": total_ms,
        }

    def get_context(self, question: str, top_k: int = None, max_chars: int = 800) -> dict:
        """把 query() 结果渲染成可直接塞进 system prompt 的纯文本，并透传各阶段耗时。

        返回 dict：context（无命中或全被阈值过滤时为 ""）、num_results 与四个耗时字段。
        `max_chars` 是 context 字符上限（默认 800）：客服答案较长，全量注入会撑大 prompt、
        抬高首 token 延迟，因此硬截断。
        """
        result = self.query(question, top_k)
        context_parts = []
        for r in result["results"]:
            # 与 query() 保持一致的口径：优先 answer，退化用 text
            answer = r.get("answer", r.get("text", ""))
            q = r.get("question", "")
            if q:
                # 带问句一起注入，让 LLM 知道这条知识的适用问题，减少答非所问
                context_parts.append(f"Q: {q}\nA: {answer}")
            else:
                context_parts.append(answer)

        # 多条之间用空行分隔，避免 LLM 把两条答案误读成一条
        context = "\n\n".join(context_parts)
        # 结果已按相关度排序，截断优先保住前面的高相关条目
        if len(context) > max_chars:
            context = context[:max_chars] + "..."

        return {
            "context": context,
            "num_results": len(result["results"]),
            "embed_ms": result["embed_ms"],
            "search_ms": result["search_ms"],
            "rerank_ms": result["rerank_ms"],
            "total_ms": result["total_ms"],
        }

    def save_index(self, path: str):
        """持久化两个文件：`<path>.faiss`（含全部文档向量的二进制索引）与 `<path>.json`（文档原文）。

        `path` 是不带扩展名的前缀（`config.RAG_INDEX_PATH` 即 data/rag_index），扩展名由本函数拼。
        副作用：创建父目录并写磁盘两个文件（覆盖同名旧文件）。
        """
        # path 无目录部分（如 "rag_index"）时退回当前目录 "."
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # 存 FAISS 原生格式：加载后无需重新 encode，可显著缩短冷启动时间
        faiss.write_index(self._index, path + ".faiss")
        # 文档原文单独存 JSON：FAISS 只存向量，回查原文必须依赖这份同序列表
        with open(path + ".json", "w", encoding="utf-8") as f:
            # ensure_ascii=False 让中文原样以 UTF-8 写入；indent=2 便于人工排查知识库
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

    def load_index(self, path: str):
        """从磁盘读回索引与文档（`save_index()` 的配对操作），返回 self 以便链式调用。

        用于跳过耗时的全库 encode。本函数不校验向量模型，因此 embed 模型必须与建索引时
        一致，否则维度不匹配会直接报错、维度恰好相同时则会静默检索出错。
        副作用：覆盖 `_index` 与 `_documents`；文件缺失或损坏时由 faiss / json 抛异常。
        """
        # 行号顺序与 save 时一致，因此能继续和下面的 documents 对齐
        self._index = faiss.read_index(path + ".faiss")
        with open(path + ".json", "r", encoding="utf-8") as f:
            self._documents = json.load(f)
        return self
