"""批量下载本项目需要的权重。

用法（任意目录都能跑，路径按 `__file__` 推仓库根，不再写死绝对路径）：

    python models/download_model.py

分工：
  · ModelScope 能拿到的走 modelscope（国内更快）—— Kokoro / bge-m3
  · Parakeet-TDT 只托管在 HuggingFace，走 huggingface_hub
  · 国内访问 HuggingFace 慢时先 `export HF_ENDPOINT=https://hf-mirror.com`
    （若 snapshot_download 反复 TLS 断连，改用 README「下载模型」里的 wget -c 单文件续传）
"""
from pathlib import Path

# 仓库根 = 本文件所在目录的上一级；models/ 就是它的子目录
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"


def from_modelscope(repo_id: str, dirname: str) -> None:
    from modelscope import snapshot_download

    print(f"[modelscope] {repo_id} → models/{dirname}")
    print("  ", snapshot_download(repo_id, local_dir=str(MODELS / dirname)))


def from_huggingface(repo_id: str, dirname: str, patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    print(f"[huggingface] {repo_id} → models/{dirname}")
    print("  ", snapshot_download(repo_id, local_dir=str(MODELS / dirname),
                                 allow_patterns=patterns))


# ── ASR：Parakeet-TDT-0.6B（只有 HuggingFace 有，约 2.5GB）──
from_huggingface(
    "nvidia/parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v3",
    patterns=[
        "model.safetensors",      # 权重主体（约 2.5GB）
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ],
)

# ── RAG 向量化 ──
from_modelscope("BAAI/bge-m3", "bge-m3")

# ── TTS ──
from_modelscope("AI-ModelScope/Kokoro-82M", "Kokoro-82M")

# VAD（silero_vad.jit）与 spaCy 模型（en_core_web_sm）来源特殊，
# 见 README「下载模型」，本脚本不处理。
print("\n完成。VAD 与 spaCy 模型请按 README「下载模型」单独获取。")
