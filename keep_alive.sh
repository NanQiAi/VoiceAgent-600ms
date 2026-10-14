#!/bin/bash
# keep_alive.sh — 看护循环：谁掉线就拉起谁（vLLM :8100 / Voice Agent :3000 / 可选 SSH 隧道）。
# 与 start_prod.sh 的分工：start_prod.sh 负责"启动一次并常驻"（会顺带把本脚本拉起），本脚本只管长期看护。
# 用法：bash keep_alive.sh（前台守护）｜bash keep_alive.sh --once（只巡检一轮，可挂 cron）。
# 可覆盖：CHECK_INTERVAL（默认 30s）/ CHECK_TIMEOUT（单次就绪等待，默认 300s）。
# 相比旧版：补上了 ws_server 看护、删掉硬编码 PID 白名单、把固定 sleep 改成轮询端口就绪。
set -uo pipefail      # 注意：这里**不用 -e** —— 守护进程必须能容忍单次巡检失败
source "$(cd "$(dirname "$0")" && pwd)/voiceagent_env.sh"

CHECK_INTERVAL="${CHECK_INTERVAL:-30}"   # 巡检间隔（秒）
CHECK_TIMEOUT="${CHECK_TIMEOUT:-300}"    # 单次启动后就绪等待上限（秒）
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

va_check_vad_path

# 可选：清理占用 GPU 但不属于本服务的孤儿进程。默认关闭 —— 它会杀掉同机其它 CUDA 进程，
# 多人共用显卡时请勿开启。
va_clean_orphans() {
    [ "${CLEAN_ORPHANS:-0}" = "1" ] || return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    va_log "清理 GPU 上的孤儿进程（CLEAN_ORPHANS=1）…"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read -r pid; do
        [ -z "$pid" ] && continue
        # 跳过我们自己的 vLLM / Voice Agent
        if va_port_busy "$LLM_PORT" || va_port_busy "$VOICE_PORT"; then continue; fi
        va_log "  kill -9 $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3   # 等驱动回收显存
}

check_once() {
    if ! va_port_busy "$LLM_PORT"; then
        va_warn "vLLM 已下线，正在重启…"
        va_clean_orphans
        va_start_vllm_bg
        va_wait_http "http://localhost:$LLM_PORT/v1/models" "$CHECK_TIMEOUT" "vLLM（重启）"
    fi

    if ! va_port_busy "$VOICE_PORT"; then
        va_warn "Voice Agent 已下线，正在重启…"
        va_start_voice_bg
        va_wait_http "http://localhost:$VOICE_PORT/api/info" "$CHECK_TIMEOUT" "Voice Agent（重启）"
    fi

    if [ "$ENABLE_TUNNEL" = "1" ] && [ -n "$VPS_IP" ]; then
        # 隧道在远端监听，本地端口探测不到；只能靠进程表模糊匹配
        if ! pgrep -f "ssh .*-R ${VOICE_PORT}:localhost:${VOICE_PORT}.*${VPS_IP}" >/dev/null 2>&1; then
            va_warn "SSH 隧道已断，重连…"
            va_start_tunnel
        fi
    fi
}

va_log "=== 守护启动（巡检间隔 ${CHECK_INTERVAL}s）==="
va_log "看护：vLLM :$LLM_PORT ｜ Voice Agent :$VOICE_PORT$( [ "$ENABLE_TUNNEL" = 1 ] && echo " ｜ 隧道 $VPS_IP" )"

if [ "$ONCE" = "1" ]; then
    check_once
    va_log "单轮巡检完成"
    exit 0
fi

while true; do
    check_once
    sleep "$CHECK_INTERVAL"
done
