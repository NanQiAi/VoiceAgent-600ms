"""依赖自检：逐个 import 并打印版本，用于快速确认当前 Python 环境是否装全。

用法：
    python check_deps.py            # 打印清单（缺必需项时退出码 1）
    python check_deps.py --quiet    # 只打印缺失项

与 `run.sh preflight` 的分工：
    本文件  —— 只查 Python 包（更细、带版本号），不依赖 bash
    run.sh  —— 除包之外还查模型权重、GPU、API Key，是启动前的完整预检

包的分类与 `requirements.txt` 的 [直接] / [传递] 标记一致。
"""

import sys

# 必需：缺任何一个，主链路都跑不起来
REQUIRED = [
    # 推理底座
    ("torch", "推理底座"),
    ("numpy", "数值计算"),
    # ASR（Parakeet-TDT / transformers）
    ("transformers", "ASR 引擎（ParakeetForTDT）"),
    ("modelscope", "权重下载（models/download_model.py）"),
    # RAG
    ("sentence_transformers", "RAG 向量化"),
    ("faiss", "RAG 检索"),
    # TTS
    ("kokoro", "TTS 引擎"),
    ("misaki", "TTS 英文 G2P"),
    ("spacy", "misaki 的分词/标注"),
    # Web 服务
    ("fastapi", "Web 服务"),
    ("uvicorn", "ASGI 服务器"),
    ("starlette", "WebSocket 异常类"),
    ("pydantic", "数据校验（fastapi 传递）"),
    ("websockets", "WS 实现（uvicorn 传递）"),
    ("nest_asyncio", "事件循环补丁"),
    # LLM 客户端
    ("openai", "LLM 客户端"),
    ("httpx", "HTTP 超时控制"),
]

# 可选：缺失只影响对应功能开关，不影响主链路启动
OPTIONAL = [
    ("FlagEmbedding", "RAG 精排（需 bge-reranker 权重）"),
]


# spaCy 的英文模型**不从 pip 装**：解压到 models/en_core_web_sm/，
# 由 server/bootstrap.py 在启动时塞进 sys.path。
SPACY_MODEL_DIR = "models/en_core_web_sm"


def check(entries, quiet):
    """逐个 import，返回缺失数量。"""
    missing = 0
    for name, why in entries:
        try:
            module = __import__(name)
            if not quiet:
                version = getattr(module, "__version__", "")
                print(f"  ✓ {name:<24} {version:<14} {why}")
        except Exception as exc:
            print(f"  ✗ {name:<24} {'':<14} {why}  ({type(exc).__name__})")
            missing += 1
    return missing


def main():
    quiet = "--quiet" in sys.argv
    print("必需依赖")
    missing_required = check(REQUIRED, quiet)

    print("\nspaCy 模型（仓库内，不经 pip）")
    import glob
    import os
    spacy_ok = os.path.isdir(os.path.join(SPACY_MODEL_DIR, "en_core_web_sm")) \
        and bool(glob.glob(os.path.join(SPACY_MODEL_DIR, "*.dist-info")))
    if spacy_ok:
        if not quiet: print(f"  ✓ {SPACY_MODEL_DIR:<24} {'':<14} misaki 英文 G2P 必需")
    else:
        print(f"  ✗ {SPACY_MODEL_DIR:<24} {'':<14} misaki 英文 G2P 必需（见 README「下载模型」）")
        missing_required += 1

    print("\n可选依赖")
    missing_optional = check(OPTIONAL, quiet)

    total = len(REQUIRED) + len(OPTIONAL)
    print(f"\n已装 {total - missing_required - missing_optional} / {total}")

    if missing_required:
        print(f"❌ 缺 {missing_required} 个必需包 —— 安装：pip install -r requirements.txt")
        print("   若只缺 kokoro / misaki：pip install kokoro==0.9.4")
        return 1
    if missing_optional and not quiet:
        print(f"·  缺 {missing_optional} 个可选包（不影响主链路）")
    print("✅ 必需依赖齐全")
    return 0


if __name__ == "__main__":
    sys.exit(main())
