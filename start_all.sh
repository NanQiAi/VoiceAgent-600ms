#!/bin/bash
# start_all.sh — 一键启动 vLLM + Voice Agent（前台托管，Ctrl+C 连子进程一起停）。
# 与 start_prod.sh 的区别：本脚本进程随终端存活，适合开发调试/现场演示；长期无人值守请用 start_prod.sh。
# 用法：bash start_all.sh；可覆盖 GPU_LLM / GPU_VOICE / VOICE_PORT / LLM_PORT / DEMO_MODE 等（见 voiceagent_env.sh）。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/voiceagent_env.sh"

VA_VLLM_PID=""
VA_VOICE_PID=""

CLEANED=0
cleanup() {
    local code=$?                 # 先接住触发本次清理的退出码，最后原样返回
    [ "$CLEANED" = "1" ] && return 0
    CLEANED=1
    trap - INT TERM EXIT          # 先摘掉 trap，避免 exit 递归触发
    # 只有真的启动过服务才打印停止过程；预检就失败时不必输出这段
    if [ -n "${VA_VOICE_PID:-}" ] || [ -n "${VA_VLLM_PID:-}" ]; then
        echo
        va_log "正在停止本次启动的服务…"
        # 按【进程组】杀：只杀父进程会留下持有 CUDA 的 TTS 孤儿，占 10GB+ 显存（原因见 voiceagent_env.sh）
        va_kill_group voice TERM || true
        va_kill_group vllm  TERM || true
        sleep 3
        # 温柔信号没走掉的，用 KILL 兜底整组
        va_kill_group voice KILL || true
        va_kill_group vllm  KILL || true
        # 最后按端口兜底（覆盖"不是本脚本起的"残留监听）
        local p
        for p in "$VOICE_PORT" "$LLM_PORT"; do
            if va_port_busy "$p"; then
                va_warn "端口 $p 仍被占用，可用 'bash start_prod.sh stop' 或手工 kill 清理"
            fi
        done
        va_log "已停止"
    fi
    exit "$code"                  # 保留原始退出码：预检失败就该以非 0 退出
}
trap cleanup INT TERM EXIT

echo "=========================================="
echo "  VoiceAgent-600ms — 级联 Pipeline 启动"
echo "=========================================="

va_check_vad_path
va_preflight all

# 显存体检：把"上次没退干净 → 这次 OOM"在启动前就说清楚
LLM_TOTAL="$(va_gpu_total "$GPU_LLM" || true)"
VOICE_TOTAL="$(va_gpu_total "$GPU_VOICE" || true)"
if [ -n "$LLM_TOTAL" ]; then
    va_check_gpu_free "$GPU_LLM" $(( LLM_TOTAL * 85 / 100 )) \
        "vLLM（LLM_GPU_UTIL=$LLM_GPU_UTIL 按比例预占显存）" || true
fi
if [ -n "$VOICE_TOTAL" ]; then
    va_check_gpu_free "$GPU_VOICE" $(( VOICE_TOTAL * 60 / 100 )) \
        "Voice Agent（ASR + TTS，TTS_GPU_UTIL=$TTS_GPU_UTIL）" || true
fi

if va_port_busy "$LLM_PORT"; then
    va_log "[1/2] vLLM 已在 :$LLM_PORT 运行，复用"
    VA_VLLM_PID="$(va_read_pid vllm)"
else
    va_log "[1/2] 启动 vLLM…"
    va_start_vllm_bg
    # 轮询 /v1/models 而不是死等固定秒数：模型真正加载完才返回
    va_wait_http "http://localhost:$LLM_PORT/v1/models" 300 "vLLM" \
        || va_warn "vLLM 未就绪；首轮对话可能失败，请看 $VLLM_LOG"
fi

if va_port_busy "$VOICE_PORT"; then
    va_log "[2/2] Voice Agent 已在 :$VOICE_PORT 运行，跳过"
    VA_VOICE_PID="$(va_read_pid voice)"
else
    va_log "[2/2] 启动 Voice Agent（依次加载 VAD/ASR/RAG/TTS，通常 60~120s）…"
    va_start_voice_bg
    va_wait_http "http://localhost:$VOICE_PORT/api/info" 300 "Voice Agent" \
        || va_warn "Voice Agent 未就绪，请看 $VOICE_LOG"
fi

va_start_tunnel

# 就绪后仍可能立刻崩（典型是 TTS 预热阶段 CUDA OOM），主动确认一次，
# 避免出现"报了就绪、其实已经死了"这种误导。
if ! va_port_busy "$VOICE_PORT" && ! pgrep -f "$VA_REPO_DIR/ws_server.py" >/dev/null 2>&1; then
    echo
    va_warn "Voice Agent 启动后立刻退出了，日志末尾 25 行："
    tail -n 25 "$VOICE_LOG" 2>/dev/null | sed 's/^/  /'
    echo
    va_warn "完整日志: $VOICE_LOG"
    cleanup
fi

va_summary
va_log "日志实时查看： tail -f $VOICE_LOG"
va_log "保持本终端开启；Ctrl+C 停止服务"

# 前台守候：Voice Agent 退出即视为本次启动结束
while va_port_busy "$VOICE_PORT"; do
    sleep 5
done

# 退出码是判断"怎么死的"最快的线索：137=SIGKILL（多半被 OOM killer 杀）、139=SIGSEGV 段错误
# （多为 CUDA/原生扩展崩）、143=SIGTERM 正常终止、1=Python 未捕获异常（看日志末尾 Traceback）。
if [ -n "${VA_VOICE_PID:-}" ]; then
    wait "$VA_VOICE_PID" 2>/dev/null
    code=$?
    case "$code" in
        137) hint="（被 SIGKILL —— 常见于内存不足被 OOM killer 杀掉，可用 dmesg | tail 确认）" ;;
        139) hint="（SIGSEGV 段错误 —— 多为 CUDA / 原生扩展崩溃）" ;;
        143) hint="（收到 SIGTERM，正常终止）" ;;
        0)   hint="（正常退出，退出码 0 —— 说明是代码主动结束的）" ;;
        *)   hint="（看日志末尾的 Traceback）" ;;
    esac
    va_warn "Voice Agent 已退出，退出码 $code $hint"
else
    va_warn "Voice Agent 已退出（未记录到 PID）"
fi

echo
va_log "──── 日志末尾 20 行（$VOICE_LOG）────"
tail -n 20 "$VOICE_LOG" 2>/dev/null | sed 's/^/  /' || va_warn "  日志文件不存在"
echo
va_log "完整日志: $VOICE_LOG"
cleanup
