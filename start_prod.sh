#!/bin/bash
# start_prod.sh — 常驻启动（setsid 脱离终端，关掉 SSH 也继续跑）+ stop/status/clean-gpu/restart 子命令。
# 与 start_all.sh 的区别：start_all.sh 前台托管、Ctrl+C 全停；本脚本会顺带拉起 keep_alive.sh 守护。
# 启动顺序 vLLM → 等就绪 → Voice Agent → 等就绪 → 守护（先起 vLLM，避免首轮对话因 LLM 未就绪而 500）。
# 用法：bash start_prod.sh [start|stop|status|clean-gpu|restart]；日志在 logs/ 下。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/voiceagent_env.sh"


# 按端口找出监听进程的 PID（用 ss，不依赖 lsof）
va_pids_on_port() {
    local port="$1"
    ss -H -ltnp "sport = :$port" 2>/dev/null \
        | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
}

# 用**绝对路径**匹配并排除自身 PID：pgrep -f 是按整条命令行做正则，只写 "keep_alive.sh"
# 会把编辑器、grep、乃至调用它的 shell 自己都误匹配进来，进而误杀。
VA_KEEPALIVE_PATTERN="$VA_REPO_DIR/keep_alive.sh"
va_keepalive_pids() {
    pgrep -f "$VA_KEEPALIVE_PATTERN" 2>/dev/null | grep -vw "$$" | grep -vw "$PPID" || true
}

do_stop() {
    va_log "停止服务…"
    # 先停守护，避免它把我们正在停的服务又拉起来（同样用绝对路径 + 排除自身）
    local kpids; kpids="$(va_keepalive_pids)"
    if [ -n "$kpids" ]; then
        va_log "  停止守护进程: $(echo "$kpids" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill $kpids 2>/dev/null || true
    fi
    sleep 1

    # 按【进程组】停：nanovllm 的 TTS 子进程**不监听任何端口**，按端口找 PID 杀不到它 ——
    # 它会变成孤儿继续占 10GB+ 显存，导致下次启动在 TTS 加载阶段 OOM（实际踩过的坑）。
    va_log "  按进程组停止 vLLM / Voice Agent（含 nanovllm 的 TTS 子进程）…"
    va_kill_group voice TERM || true
    va_kill_group vllm  TERM || true
    sleep 3
    va_kill_group voice KILL || true
    va_kill_group vllm  KILL || true

    # 再按端口兜底（覆盖不是本脚本启动的残留监听进程）
    local port
    for port in "$VOICE_PORT" "$LLM_PORT"; do
        local pids; pids="$(va_pids_on_port "$port")"
        if [ -n "$pids" ]; then
            va_log "  停止端口 $port 上的进程: $(echo "$pids" | tr '\n' ' ')"
            # shellcheck disable=SC2086
            kill $pids 2>/dev/null || true
        fi
    done
    sleep 2

    # 仍在监听就用 SIGKILL（uvicorn/vLLM 有时忽略 SIGTERM 的优雅退出）
    for port in "$VOICE_PORT" "$LLM_PORT"; do
        local pids; pids="$(va_pids_on_port "$port")"
        if [ -n "$pids" ]; then
            va_warn "  端口 $port 仍未释放，强制结束"
            # shellcheck disable=SC2086
            kill -9 $pids 2>/dev/null || true
        fi
    done
    va_clear_pid voice; va_clear_pid vllm
    va_log "已停止"
}

do_status() {
    echo "──────────────────────────────────────────────"
    printf "  %-22s %s\n" "进程" "状态"
    printf "  %-22s %s\n" "vLLM (:$LLM_PORT)" \
        "$(va_port_busy "$LLM_PORT" && echo "运行中 (PID $(va_pids_on_port "$LLM_PORT" | tr '\n' ' '))" || echo "未运行")"
    printf "  %-22s %s\n" "Voice Agent (:$VOICE_PORT)" \
        "$(va_port_busy "$VOICE_PORT" && echo "运行中 (PID $(va_pids_on_port "$VOICE_PORT" | tr '\n' ' '))" || echo "未运行")"
    printf "  %-22s %s\n" "守护 keep_alive" \
        "$(kp="$(va_keepalive_pids)"; [ -n "$kp" ] && echo "运行中 (PID $(echo "$kp" | tr '\n' ' '))" || echo "未运行")"
    if [ "$ENABLE_TUNNEL" = "1" ] && [ -n "$VPS_IP" ]; then
        printf "  %-22s %s\n" "SSH 隧道" \
            "$(pgrep -f "ssh .*-R ${VOICE_PORT}.*${VPS_IP}" >/dev/null && echo 运行中 || echo 未运行)"
    fi
    echo "──────────────────────────────────────────────"
    printf "  %-22s %s\n" "配置" "GPU_LLM=$GPU_LLM  GPU_VOICE=$GPU_VOICE"
    printf "  %-22s %s\n" ""     "模型 $(basename "$LLM_MODEL_DIR")  服务名 $LLM_SERVED_NAME"
    echo "──────────────────────────────────────────────"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  GPU:"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                   --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    fi
    echo "──────────────────────────────────────────────"
    echo "  日志: $VLLM_LOG"
    echo "        $VOICE_LOG"
    echo "        $KEEPALIVE_LOG"
    echo "──────────────────────────────────────────────"
}

do_start() {
    va_check_vad_path
    va_preflight all

    va_log "=== 启动（常驻模式）==="

    if va_port_busy "$LLM_PORT"; then
        va_log "vLLM 已在 :$LLM_PORT 运行，复用"
    else
        va_start_vllm_bg
        va_wait_http "http://localhost:$LLM_PORT/v1/models" 300 "vLLM" \
            || va_warn "vLLM 未就绪，请查看 $VLLM_LOG"
    fi

    if va_port_busy "$VOICE_PORT"; then
        va_log "Voice Agent 已在 :$VOICE_PORT 运行，复用"
    else
        va_start_voice_bg
        va_wait_http "http://localhost:$VOICE_PORT/api/info" 300 "Voice Agent" \
            || va_warn "Voice Agent 未就绪，请查看 $VOICE_LOG"
    fi

    va_start_tunnel

    # 守护进程（脱离终端常驻）
    if [ -n "$(va_keepalive_pids)" ]; then
        va_log "守护进程已在运行，跳过"
    else
        setsid nohup bash "$VA_KEEPALIVE_PATTERN" > "$KEEPALIVE_LOG" 2>&1 &
        va_log "守护进程已启动（日志 $KEEPALIVE_LOG）"
    fi

    va_summary
    va_log "查看状态： bash start_prod.sh status"
    va_log "停止服务： bash start_prod.sh stop"
}

# clean-gpu：nanovllm 的 TTS 子进程**不监听任何端口**，`stop` 按端口抓不到它；一旦服务异常退出，
# 它就变成孤儿占 10GB+ 显存，导致下次启动 TTS 加载阶段 CUDA OOM。默认只列出占用者，FORCE=1 才真杀。
do_clean_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || { va_warn "没有 nvidia-smi，无法检查显存"; return 0; }
    local gpu found=0 apps
    for gpu in "$GPU_LLM" "$GPU_VOICE"; do
        echo "──── GPU $gpu ────"
        # 两处防御：`|| true` 兜住 nvidia-smi 驱动异常的非 0 退出（本脚本开了 set -e）；
        # grep 过滤是因为它出错时会把错误文本打到 **stdout**，不过滤会被当进程列表打印。
        apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
                    --format=csv,noheader 2>/dev/null \
                | grep -E '^[[:space:]]*[0-9]' || true)"
        if [ -z "$apps" ]; then echo "  （无进程占用）"; continue; fi
        echo "$apps" | sed 's/^/  /'
        found=1
    done
    if [ "$found" = "0" ]; then va_log "两张卡都很干净，无需清理"; return 0; fi

    if [ "${FORCE:-0}" != "1" ]; then
        echo
        va_warn "以上是当前占用两张卡的进程。若确认都是本项目的残留，执行："
        va_warn "  FORCE=1 bash $0 clean-gpu"
        va_warn "（注意：它会杀掉这两张卡上的**所有**计算进程，包括别人的任务）"
        return 0
    fi

    echo
    va_warn "FORCE=1：开始清理两张卡上的计算进程…"
    local pids pid
    for gpu in "$GPU_LLM" "$GPU_VOICE"; do
        pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
                | grep -E '^[[:space:]]*[0-9]' || true)"
        for pid in $pids; do
            pid="$(printf '%s' "$pid" | tr -dc '0-9')"
            [ -n "$pid" ] || continue
            va_log "  GPU $gpu: kill -9 $pid"
            kill -9 "$pid" 2>/dev/null || true
        done
    done
    sleep 3
    va_log "清理完成，当前显存："
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
}

case "${1:-start}" in
    start)     do_start ;;
    stop)      do_stop ;;
    status)    do_status ;;
    clean-gpu) do_clean_gpu ;;
    restart)   do_stop; sleep 2; do_start ;;
    *)         echo "用法: bash $0 {start|stop|status|clean-gpu|restart}"; exit 1 ;;
esac
