"""RAG AB Test: bge-small-zh-v1.5 vs bge-m3, with/without reranker.
Usage: python ab_test_rag.py

RAG 检索环节的离线 A/B 脚本（对应 engine/rag.py）：在同一份知识库与同一批 10 个问句上，
横向比较 4 种组合，输出加载/编码/检索/重排延迟与 top-1 命中，用来回答
“多花 15-20ms 换 bge-m3 值不值、再加 reranker 会不会超出 600ms 预算”。

对比的组合（缺模型目录则自动跳过，见 main()）：
  1. bge-small-zh-v1.5（24M / 512 维 / 3-5ms）——最快，当前线上选择
  2. bge-small-zh-v1.5 + bge-reranker-v2-m3（fp16 重排 top-5，10-15ms）
  3. bge-m3（568M / 1024 维 / 15-20ms）——质量最好但更慢
  4. bge-m3 + bge-reranker-v2-m3
⚠️ 本脚本用 RAGEngine 默认 score_threshold=0.0（不过滤），看的是“检索上限”；
线上 ws_server 传 config.RAG_SCORE_THRESHOLD=0.35，会丢掉低分结果。

指标：延迟取 rag.query() 返回的 embed/search/rerank/total 四项，对 10 条问句求
avg/p50/min/max；跑前先 warmup 以免首调 CUDA 初始化污染统计；模型加载耗时单独统计。
质量只打印 top-1 的 question 与 score 供人工核对——没有标注答案，故不算 Recall/MRR/nDCG。
"""
import os
import sys
import json
import time
import numpy as np

# 仓库根目录插到 sys.path[0]，保证任意 cwd 下都能 `from engine.rag import RAGEngine`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# 与线上 config.KB_DATA_PATH 指向同一份数据
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_kb.json")

# 可用 RAG_DEVICE=cpu 做无卡冒烟测试
DEVICE = os.environ.get("RAG_DEVICE", "cuda:0")

BGE_SMALL_DIR = os.path.join(MODEL_DIR, "bge-small-zh-v1.5")
BGE_M3_DIR = os.path.join(MODEL_DIR, "bge-m3")
# bge-m3 的另一种落盘布局（ModelScope / HF 缓存风格），main() 里与 BGE_M3_DIR 二选一
BGE_M3_MS_DIR = os.path.join(MODEL_DIR, "BAAI", "bge-m3")
# 只有它存在才会跑“带 reranker”的两组
RERANKER_DIR = os.path.join(MODEL_DIR, "bge-reranker-v2-m3")

# 固定问句：覆盖价格/客服/语种/数据安全/并发/声音定制/试用/退款/地域等意图，
# 也刻意包含“支持马来语吗”“你们在哪个城市”这类知识库可能没有的问题
TEST_QUERIES = [
    "你们的价格是多少",
    "怎么联系客服",
    "支持马来语吗",
    "AI可以处理什么问题",
    "数据安全怎么保障",
    "能同时接多少电话",
    "声音可以自定义吗",
    "怎么开始试用",
    "退款怎么办",
    "你们在哪个城市",
]


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
    # top_k / rerank_top_k 与线上 config.RAG_TOP_K 及 ws_server 用法一致；
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
    print(f"  Index build: {build['encode_ms']:.0f}ms ({build['num_docs']} docs, dim={build['dim']})")
    print(f"  Per-doc encode: {build['encode_per_doc_ms']:.1f}ms")

    # 首次 encode 会触发 CUDA kernel 编译/显存分配，不预热会把第一条问句的延迟拉高
    rag.query("测试查询")

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
        # 样本只有 10 条，p50 即中位数
        arr = np.array(v)
        print(f"    {k:>8s}: avg={arr.mean():.1f}ms  p50={np.median(arr):.1f}ms  "
              f"min={arr.min():.1f}ms  max={arr.max():.1f}ms")

    # 无 ground truth，只做人工核对；只打前 5 条避免刷屏
    print(f"\n  Retrieval quality (top-1):")
    for r in results_log[:5]:
        print(f"    Q: {r['query']}")
        print(f"    → {r['top1_q']} (score={r['score']:.3f})")

    # avg_total_ms 是选型最关键的指标（直接叠加进首响预算）
    return {
        "name": name,
        "load_ms": load_ms,
        "avg_total_ms": np.mean(latencies["total"]),
        "avg_embed_ms": np.mean(latencies["embed"]),
        "avg_rerank_ms": np.mean(latencies["rerank"]),
    }


def main():
    """入口：按“模型目录是否存在”逐个跑 4 种组合，最后打印 SUMMARY 横向对比表。

    执行顺序 bge-small（无/有 reranker）→ bge-m3（无/有 reranker）；目录缺失的组合被 SKIP
    而非报错，因此只有小模型的机器也能单独跑基线。不写任何文件。
    """
    docs = load_docs()
    print(f"Loaded {len(docs)} documents")
    print(f"Device: {DEVICE}")

    results = []

    if os.path.isdir(BGE_SMALL_DIR):
        # 先跑不带 reranker 的基线，便于单一变量对比 reranker 的收益
        r = benchmark_config(
            "bge-small-zh-v1.5 (no reranker)",
            BGE_SMALL_DIR, None, docs, TEST_QUERIES, DEVICE
        )
        results.append(r)

        if os.path.isdir(RERANKER_DIR):
            r = benchmark_config(
                "bge-small-zh-v1.5 + reranker-v2-m3",
                BGE_SMALL_DIR, RERANKER_DIR, docs, TEST_QUERIES, DEVICE
            )
            results.append(r)
    else:
        print(f"SKIP: {BGE_SMALL_DIR} not found")

    # bge-m3 两种落盘布局二选一；都不存在则跳过全部 m3 组合
    m3_dir = BGE_M3_DIR if os.path.isdir(BGE_M3_DIR) else BGE_M3_MS_DIR
    if os.path.isdir(m3_dir):
        r = benchmark_config(
            "bge-m3 (no reranker)",
            m3_dir, None, docs, TEST_QUERIES, DEVICE
        )
        results.append(r)

        # 理论上质量最高、延迟也最高，最可能突破 600ms 预算
        if os.path.isdir(RERANKER_DIR):
            r = benchmark_config(
                "bge-m3 + reranker-v2-m3",
                m3_dir, RERANKER_DIR, docs, TEST_QUERIES, DEVICE
            )
            results.append(r)
    else:
        print(f"SKIP: bge-m3 not found at {BGE_M3_DIR} or {BGE_M3_MS_DIR}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<40s} {'Load':>8s} {'Embed':>8s} {'Rerank':>8s} {'Total':>8s}")
    print("-" * 72)
    for r in results:
        # Load 为一次性冷启动；Embed/Rerank/Total 为 10 条问句的平均值（ms）
        print(f"{r['name']:<40s} {r['load_ms']:>7.0f}ms {r['avg_embed_ms']:>6.1f}ms "
              f"{r['avg_rerank_ms']:>6.1f}ms {r['avg_total_ms']:>6.1f}ms")


# 仅直接运行时才跑基准（被 import 时不会触发模型加载与显存占用）
if __name__ == "__main__":
    main()
