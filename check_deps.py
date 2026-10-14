"""逐个 import 依赖清单并打印可用/缺失统计，用于快速确认当前 python 环境是否装全。"""
mods = [
    ("torch", "核心"), ("torchaudio", "核心"),
    ("vllm", "LLM服务"), ("funasr", "ASR(SenseVoice)"),
    ("onnxruntime", "降噪/投机ASR"), ("FlagEmbedding", "RAG"), ("faiss", "RAG"),
    ("sentence_transformers", "RAG"), ("soundfile", "音频IO"), ("numpy", "核心"),
    ("scipy", "音频"), ("cn2an", "评测"), ("fastapi", "Web"), ("uvicorn", "Web"),
    ("websockets", "Web"), ("nest_asyncio", "Web"), ("httpx", "LLM客户端"),
    ("openai", "LLM客户端"),
    # 可选组件
    ("nanovllm_voxcpm", "TTS(必须)"), ("speechbrain", "声纹VAD(可选)"),
    ("pipecat", "SmartTurn(可选)"), ("moonshine_onnx", "投机ASR(可选)"),
    ("fireredvad", "FireRedVAD(可选)"),
]
ok = miss = 0
for name, why in mods:
    try:
        __import__(name); print(f"  ✅ {name:<24} {why}"); ok += 1
    except Exception as e:
        print(f"  ❌ {name:<24} {why}  -> {type(e).__name__}"); miss += 1
print(f"\n可用 {ok} / 缺失 {miss}")
