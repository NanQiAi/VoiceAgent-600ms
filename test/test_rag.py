"""RAG 检索离线基准：bge-m3（可选再加 reranker）。

用法：python test/test_rag.py
      RAG_DEVICE=cpu python test/test_rag.py     # 无卡机器也能跑

对应 engine/rag.py 的检索环节：在同一份知识库、同一批问句上比较
「只用 bge-m3」与「bge-m3 + bge-reranker-v2-m3」两种组合，输出
加载 / 编码 / 检索 / 重排延迟与 top-1 命中，用来回答
"多花十几毫秒加 reranker 值不值、会不会超出 600ms 首响预算"。

⚠️ 本脚本用 RAGEngine 默认 score_threshold=0.0（不过滤），看的是"检索上限"；
线上 server/loader.py 传 config.RAG_SCORE_THRESHOLD=0.35，会丢掉低分结果。
若想看阈值后的效果，把 benchmark_config 里的 score_threshold 传进来即可。

指标：延迟取 rag.query() 返回的 embed/search/rerank/total 四项，对全部问句求
avg/p50/min/max；跑前先 warmup 以免首调 CUDA 初始化污染统计；模型加载耗时单独统计。
质量只打印 top-1 的 question 与 score 供人工核对——没有标注答案，故不算 Recall/MRR/nDCG。

历史：本脚本原先还对比 bge-small-zh-v1.5（512 维），但该权重已从仓库移除，
那一组会自动跳过；问句原先也是中文，知识库译成英文后一并改成英文。
"""
import json
import os
import sys
import time

import numpy as np

# 本脚本在 test/ 下，仓库根目录是它的上一层：插到 sys.path[0] 才能 `from engine.rag import ...`
# （任意 cwd 下运行都不会找不到包）
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# 线上实际使用的阈值，仅用于对比（本脚本自身用 0.0 = 不过滤）
from config import RAG_SCORE_THRESHOLD as ONLINE_THRESHOLD  # noqa: E402

MODEL_DIR = os.path.join(REPO_ROOT, "models")
# 与线上 config.KB_DATA_PATH 指向同一份数据
DATA_PATH = os.path.join(REPO_ROOT, "data", "sample_kb.json")

# 可用 RAG_DEVICE=cpu 做无卡冒烟测试
DEVICE = os.environ.get("RAG_DEVICE", "cuda:0")

BGE_M3_DIR = os.path.join(MODEL_DIR, "bge-m3")
# bge-m3 的另一种落盘布局（ModelScope / HF 缓存风格），与 BGE_M3_DIR 二选一
BGE_M3_MS_DIR = os.path.join(MODEL_DIR, "BAAI", "bge-m3")
# 只有它存在才会额外跑"带 reranker"的一组（权重不在仓库，默认跳过）
RERANKER_DIR = os.path.join(MODEL_DIR, "bge-reranker-v2-m3")

# 候选 embedding 模型：显示名 → 可能的落盘路径（按顺序取第一个存在的）。
# 想把别的模型拉进来对比，往这里加一项即可，main() 会自动跑。
EMBED_CANDIDATES = [
    ("bge-m3", [BGE_M3_DIR, BGE_M3_MS_DIR]),
]

# 固定问句：**英文**，与 data/sample_kb.json（英文面壁智能 FAQ）同语种。
# 分两组是为了能自动算"正例最低分 vs 负例最高分"的间隔，从而判断
# config.RAG_SCORE_THRESHOLD 定得合不合适（理想情况：正例最低分 > 阈值 > 负例最高分）。
POSITIVE_QUERIES = [
    "What is MiniCPM?",
    "When was the company founded?",
    "Do you support private deployment?",
    "How can I get your open-source models?",
    "How do you protect customer data?",
    "Can your models run on a phone?",
    "How is API usage priced?",
    "Do you offer internships for university students?",
]

# 负例：知识库里**没有**对应条目，检索不该硬凑答案。
NEGATIVE_QUERIES = [
    "Can I get a refund for my subscription?",
    "Do you support Malay?",
]

TEST_QUERIES = POSITIVE_QUERIES + NEGATIVE_QUERIES


def load_docs():
    """读取知识库 JSON（list[dict]，每条含 id/question/answer）。顺序即 FAISS 索引行号顺序。"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def benchmark_config(name, embed_dir, reranker_dir, docs, queries, device):
    """跑一组 (embedding, reranker) 组合，打印延迟与 top-1 质量并返回汇总指标。

    副作用: 打印进度到 stdout；加载模型会占 device 显存（返回后由 GC 释放，但不保证立刻归还，
    连跑多个组合时留意显存峰值）。函数内 import engine.rag，让纯文本解析场景不必依赖重库。
    """
    from engine.rag import RAGEngine

    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    # 模型加载（读盘 + 上显存）是一次性冷启动成本，不计入单次检索延迟
    t0 = time.perf_counter()
    # top_k / rerank_top_k 与线上 config.RAG_TOP_K 及 server/loader.py 用法一致；
    # 不传 score_threshold → 用默认 0.0 = 不过滤，以便看到完整检索结果
    rag = RAGEngine(
        embed_model_dir=embed_dir,
        device=device,
        reranker_model_dir=reranker_dir,
        top_k=3,
        rerank_top_k=5,
    ).load()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model load: {load_ms:.0f}ms")

    build = rag.build_index(docs)
    print(
        f"  Index build: {build['encode_ms']:.0f}ms "
        f"({build['num_docs']} docs, dim={build['dim']})"
    )
    print(f"  Per-doc encode: {build['encode_per_doc_ms']:.1f}ms")

    # 首次 encode 会触发 CUDA kernel 编译/显存分配，不预热会把第一条问句的延迟拉高
    rag.query("warmup query")

    latencies = {"embed": [], "search": [], "rerank": [], "total": []}
    results_log = []

    for q in queries:
        result = rag.query(q)
        latencies["embed"].append(result["embed_ms"])
        latencies["search"].append(result["search_ms"])
        # 无 reranker 的组合该字段恒为 0.0
        latencies["rerank"].append(result["rerank_ms"])
        # total 才是计入 600ms 首响预算的口径
        latencies["total"].append(result["total_ms"])
        # 阈值过滤可能把结果全丢掉，此时 top1 用空 dict，下面 get 都取默认值
        top1 = result["results"][0] if result["results"] else {}
        results_log.append({
            "query": q,
            "top1_q": top1.get("question", ""),
            # 未启用 reranker 时即余弦相似度；启用后仍固定打印 score 字段，便于跨组合横向比较
            "score": top1.get("score", 0),
        })

    print(f"\n  Latency ({len(queries)} queries):")
    for k, v in latencies.items():
        # 样本量小，p50 即中位数
        arr = np.array(v)
        print(f"    {k:>8s}: avg={arr.mean():.1f}ms  p50={np.median(arr):.1f}ms  "
              f"min={arr.min():.1f}ms  max={arr.max():.1f}ms")

    # 无 ground truth，只做人工核对；打印全部问句，负例用 [neg] 标出
    neg_set = set(NEGATIVE_QUERIES)
    print(f"\n  Retrieval quality (top-1, {len(results_log)} queries):")
    for r in results_log:
        tag = "[neg]" if r["query"] in neg_set else "     "
        print(f"    {tag} [{r['score']:.3f}] {r['query']}")
        print(f"              → {r['top1_q'] or '(无结果)'}")

    # 正负例分数间隔 —— 直接决定阈值该怎么定
    pos_scores = [r["score"] for r in results_log if r["query"] not in neg_set]
    neg_scores = [r["score"] for r in results_log if r["query"] in neg_set]
    if pos_scores and neg_scores:
        print(f"\n  Threshold analysis (线上 config.RAG_SCORE_THRESHOLD = {ONLINE_THRESHOLD}):")
        print(f"    正例最低分 = {min(pos_scores):.3f}")
        print(f"    负例最高分 = {max(neg_scores):.3f}")
        if min(pos_scores) > max(neg_scores):
            print(f"    → 可用阈值区间 ({max(neg_scores):.3f}, {min(pos_scores):.3f})；"
                  f"线上 {ONLINE_THRESHOLD} "
                  f"{'落在区间内 ✅' if max(neg_scores) < ONLINE_THRESHOLD < min(pos_scores) else '⚠️ 不在区间内'}")
        else:
            print(f"    → 正负例分数**重叠**（{max(neg_scores):.3f} ≥ {min(pos_scores):.3f}），"
                  f"单靠阈值无法完全分开；负例仍会被放行")

    # avg_total_ms 是选型最关键的指标（直接叠加进首响预算）
    return {
        "name": name,
        "load_ms": load_ms,
        "avg_total_ms": np.mean(latencies["total"]),
        "avg_embed_ms": np.mean(latencies["embed"]),
        "avg_rerank_ms": np.mean(latencies["rerank"]),
    }


def main():
    """入口：对每个存在的 embedding 模型跑"无/有 reranker"两组，最后打印 SUMMARY。

    目录缺失的组合被 SKIP 而非报错，因此只有 bge-m3 的机器（也就是当前仓库）
    会跑 1~2 组，不会因为缺权重而失败。不写任何文件。
    """
    docs = load_docs()
    print(f"Loaded {len(docs)} documents")
    print(f"Device: {DEVICE}")

    results = []

    for name, candidates in EMBED_CANDIDATES:
        embed_dir = next((d for d in candidates if os.path.isdir(d)), None)
        if not embed_dir:
            print(f"SKIP: {name} not found at {candidates}")
            continue

        # 先跑不带 reranker 的基线，便于单一变量对比 reranker 的收益
        results.append(benchmark_config(
            f"{name} (no reranker)", embed_dir, None, docs, TEST_QUERIES, DEVICE
        ))

        # 有 reranker 权重才跑第二组：理论上质量最高、延迟也最高，最可能突破 600ms 预算
        if os.path.isdir(RERANKER_DIR):
            results.append(benchmark_config(
                f"{name} + reranker-v2-m3", embed_dir, RERANKER_DIR, docs, TEST_QUERIES, DEVICE
            ))
        else:
            print(f"SKIP: {RERANKER_DIR} not found（跳过带 reranker 的一组）")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<40s} {'Load':>8s} {'Embed':>8s} {'Rerank':>8s} {'Total':>8s}")
    print("-" * 72)
    for r in results:
        # Load 为一次性冷启动；Embed/Rerank/Total 为全部问句的平均值（ms）
        print(f"{r['name']:<40s} {r['load_ms']:>7.0f}ms {r['avg_embed_ms']:>6.1f}ms "
              f"{r['avg_rerank_ms']:>6.1f}ms {r['avg_total_ms']:>6.1f}ms")


# 仅直接运行时才跑基准（被 import 时不会触发模型加载与显存占用）
if __name__ == "__main__":
    main()
