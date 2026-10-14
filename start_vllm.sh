#!/bin/bash
# start_vllm.sh — 只启动 vLLM（OpenAI 兼容 API），不动 ASR/TTS/Web，用于单独调试 LLM 或给别人当后端。
# 用法：bash start_vllm.sh（前台打屏）｜bash start_vllm.sh --bg（后台，日志写 logs/vllm.log）。
# 模型名与端口从 config.py 读取（见 voiceagent_env.sh），不会再与 ws_server 期望值漂移。
# 可覆盖：GPU_LLM / LLM_MODEL_DIR / LLM_SERVED_NAME / LLM_PORT / LLM_GPU_UTIL / LLM_MAX_LEN / LLM_QUANT 等。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/voiceagent_env.sh"

BACKGROUND=0
[ "${1:-}" = "--bg" ] && BACKGROUND=1

va_preflight llm

QUANT="$(va_llm_quant)"
va_log "模型      : $LLM_MODEL_DIR"
va_log "服务名    : $LLM_SERVED_NAME   （已与 config.VLLM_MODEL_NAME 自动对齐）"
va_log "物理 GPU  : $GPU_LLM（进程内可见为 cuda:0）"
va_log "监听      : http://0.0.0.0:$LLM_PORT/v1    显存占比 $LLM_GPU_UTIL"
[ -n "$QUANT" ] && va_log "量化      : $QUANT"

if [ "$BACKGROUND" = "1" ]; then
    va_start_vllm_bg
    va_wait_http "http://localhost:$LLM_PORT/v1/models" 300 "vLLM" \
        && va_log "验证: curl -s http://localhost:$LLM_PORT/v1/models"
    exit 0
fi

# --served-model-name 必须等于客户端请求里的 model 字段，否则 404 model not found；
# --max-model-len 4096 是 "system prompt + 历史 + RAG 上下文 + 回复" 的总预算。
# ⚠️ 不要加 --enforce-eager：会连带关掉 JIT kernel warmup，FlashInfer 采样内核被拖到第一个真实
#    请求才现场编译（实测 24 秒），撞穿 engine/llm.py 的 read=30s 超时；详见 voiceagent_env.sh。
ARGS=(
    --model "$LLM_MODEL_DIR"
    --served-model-name "$LLM_SERVED_NAME"
    --trust-remote-code --dtype auto
    --gpu-memory-utilization "$LLM_GPU_UTIL"
    --max-model-len "$LLM_MAX_LEN"
    --max-num-seqs "$LLM_MAX_SEQS"
    --host 0.0.0.0 --port "$LLM_PORT"
)
[ -n "$QUANT" ] && ARGS+=(--quantization "$QUANT")

va_log "前台启动（Ctrl+C 停止）…"
exec env CUDA_VISIBLE_DEVICES="$GPU_LLM" \
    "$PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
