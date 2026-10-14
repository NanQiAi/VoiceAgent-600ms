#!/bin/bash
# voiceagent_env.sh — 4 个启动脚本共享的配置 + 预检 + 启动函数（不单独运行）：
#   start_vllm.sh(只起 LLM) / start_all.sh(前台一键) / start_prod.sh(常驻 + stop/status/clean-gpu)
#   keep_alive.sh(看护循环)
# 模型名与端口一律从 config.py 读取，脚本不再可能与代码漂移（历史上因此报过 model not found）。
# 可调项都在本文件：VOICEAGENT_PYTHON / GPU_LLM / GPU_VOICE / LLM_PORT / LLM_GPU_UTIL /
#   TTS_GPU_UTIL / VOICE_PORT / VA_LOG_DIR / VA_SKIP_PREFLIGHT=1。

# 幂等保护：被 source 两次时不重复初始化
[ -n "${VA_ENV_LOADED:-}" ] && return 0
VA_ENV_LOADED=1

# VA_REPO_DIR：本文件所在目录即仓库根，调用方无论怎么启动都以此为准
VA_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VA_REPO_DIR

# 解释器解析顺序：显式 VOICEAGENT_PYTHON → PATH 里的 python3/python；不再硬编码 conda 路径
va_resolve_python() {
    if [ -n "${VOICEAGENT_PYTHON:-}" ]; then
        [ -x "$VOICEAGENT_PYTHON" ] || va_die "VOICEAGENT_PYTHON 指向的不是可执行文件: $VOICEAGENT_PYTHON"
        echo "$VOICEAGENT_PYTHON"; return 0
    fi
    local c
    for c in python python3; do
        if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
    done
    va_die "找不到 python。请先 'conda activate voiceagent'，或设置 VOICEAGENT_PYTHON=<解释器绝对路径>"
}
PYTHON="$(va_resolve_python)"
export PYTHON

# 解释器体检：环境选错会表现为后面一堆莫名的"缺少 xxx"，提前说清楚更省时间
va_check_interpreter() {
    case "$PYTHON" in
        /usr/bin/python*|/bin/python*)
            va_warn "当前用的是系统 Python（$PYTHON），不是 conda 环境里的。"
            va_warn "  请先： conda activate voiceagent"
            va_warn "  或：   export VOICEAGENT_PYTHON=/path/to/envs/voiceagent/bin/python"
            ;;
    esac

    # conda base 能用，但把项目依赖装进 base 很容易和别的项目打架
    local prefix
    prefix="$("$PYTHON" -c 'import sys;print(sys.prefix)' 2>/dev/null)"
    case "$prefix" in
        */miniforge3|*/miniconda3|*/anaconda3|*/miniforge)
            va_warn "当前是 conda 的 base 环境（$prefix），不是项目专用环境。"
            va_warn "  建议： conda create -y -n voiceagent python=3.11 && conda activate voiceagent"
            ;;
    esac

    # 本依赖清单按 Python 3.11 验证；其它版本各包都有轮子，但未逐一验证
    local ver
    ver="$("$PYTHON" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    if [ -n "$ver" ] && [ "$ver" != "3.11" ]; then
        va_warn "Python 版本为 $ver，而本依赖清单按 3.11 验证（其余版本可装但未验证）。"
    fi
}
# 把解释器所在目录前置到 PATH：让脚本里裸调的 pip/ninja 等也命中同一环境
export PATH="$(dirname "$PYTHON"):$PATH"

# 运行期环境变量：离线优先，避免现场联网卡住
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # HF 走国内镜像
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$VA_REPO_DIR/models}"  # 权重统一放仓库内
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"                 # 只用本地权重，不联网
export FUNASR_DISABLE_UPDATE="${FUNASR_DISABLE_UPDATE:-1}"   # 关掉 FunASR 更新检查

# config.py 只 import os，可安全单独加载取值；脚本与代码因此不可能出现模型名/端口漂移
va_cfg() {
    "$PYTHON" -c "
import importlib.util, sys, os
p = os.path.join(os.environ['VA_REPO_DIR'], 'config.py')
spec = importlib.util.spec_from_file_location('va_cfg', p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(getattr(m, sys.argv[1]))
" "$1" 2>/dev/null
}

VA_CFG_MODEL_DIR="$(va_cfg LLM_MODEL_DIR)"
VA_CFG_MODEL_NAME="$(va_cfg VLLM_MODEL_NAME)"
VA_CFG_BASE_URL="$(va_cfg VLLM_BASE_URL)"
# 从 http://localhost:8100/v1 里抠出端口号
VA_CFG_PORT="$(printf '%s' "$VA_CFG_BASE_URL" | sed -n 's|.*:\([0-9]\{2,\}\)/.*|\1|p')"

# GPU 分配：LLM 独占一张，ASR+TTS 共用另一张
GPU_LLM="${GPU_LLM:-0}"       # vLLM 所在物理卡
GPU_VOICE="${GPU_VOICE:-1}"   # ASR/TTS 所在物理卡

# LLM 服务：默认值全部来自 config.py，保证与 ws_server 一致
LLM_MODEL_DIR="${LLM_MODEL_DIR:-$VA_CFG_MODEL_DIR}"
LLM_SERVED_NAME="${LLM_SERVED_NAME:-$VA_CFG_MODEL_NAME}"
LLM_PORT="${LLM_PORT:-${VA_CFG_PORT:-8100}}"
LLM_GPU_UTIL="${LLM_GPU_UTIL:-0.85}"   # 独占一张卡时可以给高，KV cache 更充裕
LLM_MAX_LEN="${LLM_MAX_LEN:-4096}"
LLM_MAX_SEQS="${LLM_MAX_SEQS:-8}"

# Voice Agent：ASR + RAG + TTS 都在这一个进程里
VOICE_PORT="${VOICE_PORT:-3000}"
RAG_DEVICE="${RAG_DEVICE:-cpu}"        # 检索只需几毫秒，放 CPU 把显存留给 ASR/TTS
TTS_GPU_UTIL="${TTS_GPU_UTIL:-0.45}"   # 与 ASR 同卡，按 24G 卡算约 10.8G
DEMO_MODE="${DEMO_MODE:-0}"
# FireRedASR2 在本仓库不可用（缺 models/ 子包、权重也未下载），固定用默认的 SenseVoice
USE_FIRERED_ASR="${USE_FIRERED_ASR:-0}"

# 日志默认放仓库 logs/（跟项目在一起、不会被系统清理），可用 VA_LOG_DIR 覆盖
VA_LOG_DIR="${VA_LOG_DIR:-$VA_REPO_DIR/logs}"
mkdir -p "$VA_LOG_DIR" 2>/dev/null || VA_LOG_DIR="/tmp"
VLLM_LOG="${VA_LOG_DIR}/vllm.log"
VOICE_LOG="${VA_LOG_DIR}/voice_agent.log"
KEEPALIVE_LOG="${VA_LOG_DIR}/keepalive.log"

# 可选：公网反向隧道（默认关闭）
VPS_IP="${VPS_IP:-}"                   # 例如 1.2.3.4；留空 = 不建隧道
ENABLE_TUNNEL="${ENABLE_TUNNEL:-0}"    # 置 1 且 VPS_IP 非空时才建

va_log()  { echo "$(date '+%H:%M:%S') $*"; }
va_warn() { echo "$(date '+%H:%M:%S') [警告] $*" >&2; }
va_die()  { echo "$(date '+%H:%M:%S') [错误] $*" >&2; exit 1; }

# 端口是否已被监听：优先 ss（Debian 自带），退化到 /dev/tcp；不再依赖 lsof
# （原脚本把 lsof 当硬依赖，缺了会误判"服务已死"并反复重启）
va_port_busy() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        [ -n "$(ss -H -ltn "sport = :$port" 2>/dev/null)" ]
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && { exec 3<&- 3>&-; return 0; } || return 1
    fi
}

# 轮询一个 HTTP 地址直到可用。用法：va_wait_http <url> <超时秒> <说明>
# 必须看 HTTP 状态码：curl 对 404/500 同样返回 0，只看退出码会把"服务起了但接口报错"误判成就绪
va_wait_http() {
    local url="$1" timeout="${2:-120}" label="${3:-服务}" i=0 code=""
    va_log "等待 $label 就绪（最多 ${timeout}s）…"
    while [ "$i" -lt "$timeout" ]; do
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$url" 2>/dev/null)"
        case "$code" in
            2*) va_log "$label 已就绪（HTTP $code）"; return 0 ;;
            "") : ;;
            *)  va_warn "$label 返回 HTTP $code，继续等待…" ;;
        esac
        sleep 2; i=$((i + 2))
    done
    va_warn "$label 在 ${timeout}s 内未就绪（最后 HTTP 状态: ${code:-无响应}），请查看日志"
    return 1
}

va_has_py() { "$PYTHON" -c "import $1" >/dev/null 2>&1; }

# 进程组管理（关键）：nanovllm 的 TTS 后端加载时会 fork 出持有 CUDA 上下文的子进程，
# 只 kill 父进程的话它变成孤儿继续占 10GB+ 显存，下次启动就在 TTS 加载阶段 CUDA OOM。
# 所以服务一律用 setsid 起（PGID == PID）并落盘 PID，停止时用 `kill -TERM -PGID` 一次带走整组。
VA_PID_DIR="$VA_LOG_DIR"
va_write_pid() { printf '%s\n' "$2" > "$VA_PID_DIR/$1.pid" 2>/dev/null || true; }
va_read_pid()  { [ -r "$VA_PID_DIR/$1.pid" ] && cat "$VA_PID_DIR/$1.pid" 2>/dev/null || true; }
va_clear_pid() { rm -f "$VA_PID_DIR/$1.pid" 2>/dev/null || true; }

# 杀某个服务及其全部子孙进程。用法：va_kill_group <vllm|voice> [TERM|KILL]
va_kill_group() {
    local name="$1" sig="${2:-TERM}" pgid
    pgid="$(va_read_pid "$name")"
    [ -n "$pgid" ] || return 1
    # 负号表示"进程组"；即使组长已死，组内其它进程仍会被送达
    kill "-$sig" "-$pgid" 2>/dev/null || kill "-$sig" "$pgid" 2>/dev/null || true
    [ "$sig" = "KILL" ] && va_clear_pid "$name"
    return 0
}

# 启动前的显存体检：最容易踩的坑是"上次的 TTS 子进程没退干净 → 这次加载 TTS 时 CUDA OOM"，
# 直接把占用者列出来比事后翻日志快。没有 nvidia-smi 时静默跳过。
va_gpu_total() {  # 返回该卡总显存 MiB；查不到就返回非 0 且无输出
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local out
    out="$(nvidia-smi -i "$1" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)" || return 1
    # 只保留数字：驱动异常时 nvidia-smi 会打错误文本，混进下面的整数比较会报表达式错误
    out="$(printf '%s' "$out" | tr -dc '0-9')"
    [ -n "$out" ] || return 1
    printf '%s' "$out"
}
va_check_gpu_free() {  # va_check_gpu_free <物理卡号> <至少需要MiB> <用途>
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local gpu="$1" need="$2" what="${3:-服务}" free
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)" || return 0
    free="$(printf '%s' "$free" | tr -dc '0-9')"
    # 取不到数值（驱动异常/无卡）就静默跳过，不做误判
    [ -n "$free" ] || return 0
    if [ "$free" -lt "$need" ]; then
        va_warn "GPU $gpu 空闲显存只有 ${free} MiB，而 $what 约需 ${need} MiB —— 很可能 OOM"
        va_warn "  当前占用 GPU $gpu 的进程："
        nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
                   --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        va_warn "  若这些是上次没退干净的残留（尤其 nanovllm 的 TTS 子进程）："
        va_warn "    bash start_prod.sh stop        # 按端口+进程组清理"
        va_warn "    kill -9 <上面的PID>            # 实在清不掉时手工杀"
        return 1
    fi
    va_log "GPU $gpu 空闲显存 ${free} MiB（$what 约需 ${need} MiB）"
    return 0
}

# engine/vad.py 直接读 config.VAD_MODEL_DIR 下的 silero_vad.jit（models/snakers4_silero-vad/），
# 找不到时会退化成 torch.hub **联网**下载（离线环境直接失败，表现是"卡在 Loading VAD 很久"）。
# 常见坑：GitHub 上的 snakers4/silero-vad 是源码仓库，.jit 埋在 src/silero_vad/data/ 下，需拷过来。
VA_VAD_DIR="$VA_REPO_DIR/models/snakers4_silero-vad"
va_check_vad_path() {
    local jit="$VA_VAD_DIR/silero_vad.jit"
    [ -f "$jit" ] && return 0

    va_warn "找不到 VAD 权重: $jit"
    va_warn "  engine/vad.py 会因此退化为联网 torch.hub 下载（离线环境会失败）"

    # 可能只是放错位置了（例如 silero 源码仓库的嵌套结构）
    local found
    found="$(find "$VA_REPO_DIR/models" -maxdepth 6 -name silero_vad.jit 2>/dev/null | head -1)"
    if [ -n "$found" ]; then
        va_warn "  在别处找到了同名文件: $found"
        va_warn "  移到期望位置即可:  mkdir -p '$VA_VAD_DIR' && cp '$found' '$jit'"
    else
        va_warn "  获取方式见 README「下载模型」一节；注意 GitHub 上它是源码仓库，"
        va_warn "  .jit 位于 src/silero_vad/data/ 下，需拷进 models/snakers4_silero-vad/"
    fi
    # 刻意返回 0（只告警、不中断）：若机器联网，torch.hub 仍能兜底下载；
    # 且调用方脚本都开了 set -e，返回非 0 会把整个启动流程打断。
}

# 预检。用法：va_preflight <llm|voice|all>
# 把"缺什么、装什么"一次说清楚，避免服务起一半才报 ImportError
va_preflight() {
    local mode="${1:-all}"
    [ "${VA_SKIP_PREFLIGHT:-0}" = "1" ] && { va_warn "已跳过预检（VA_SKIP_PREFLIGHT=1）"; return 0; }

    va_check_interpreter

    local fatal=0
    va_log "预检（模式: $mode）"

    if [ "$mode" = "llm" ] || [ "$mode" = "all" ]; then
        if va_has_py vllm; then
            va_log "  ✓ vllm"
        else
            va_warn "  ✗ 缺少 vllm —— 安装： pip install vllm"; fatal=1
        fi
        if [ -d "$LLM_MODEL_DIR" ]; then
            va_log "  ✓ LLM 权重 $LLM_MODEL_DIR"
        else
            va_warn "  ✗ LLM 权重目录不存在: $LLM_MODEL_DIR"; fatal=1
        fi
    fi

    if [ "$mode" = "voice" ] || [ "$mode" = "all" ]; then
        local m
        # nest_asyncio 是 ws_server.py 的**模块级**导入，缺了连启动都不行
        for pair in "nest_asyncio:nest-asyncio" "funasr:funasr" "onnxruntime:onnxruntime" \
                    "sentence_transformers:sentence-transformers" "nanovllm_voxcpm:nano-vllm-voxcpm" \
                    "soundfile:soundfile" "faiss:faiss-cpu"; do
            m="${pair%%:*}"
            if va_has_py "$m"; then va_log "  ✓ $m"
            else va_warn "  ✗ 缺少 $m —— 安装： pip install ${pair##*:}"; fatal=1; fi
        done
        [ -d "$VA_REPO_DIR/models/SenseVoiceSmall" ] \
            && va_log "  ✓ ASR 权重 SenseVoiceSmall" \
            || { va_warn "  ✗ ASR 权重缺失: models/SenseVoiceSmall"; fatal=1; }
        [ -d "$VA_REPO_DIR/models/VoxCPM1.5" ] \
            && va_log "  ✓ TTS 权重 VoxCPM1.5" \
            || { va_warn "  ✗ TTS 权重缺失: models/VoxCPM1.5"; fatal=1; }
    fi

    # 可选组件：缺失只是少个功能，不阻塞
    va_has_py moonshine_onnx || va_warn "  · 可选组件缺失: moonshine_onnx（投机式 ASR，不影响主链路）"
    va_has_py pipecat        || va_warn "  · 可选组件缺失: pipecat（Smart Turn 语义判停，不影响主链路）"
    va_has_py fireredvad     || va_warn "  · 可选组件缺失: fireredvad（替代 VAD，不影响主链路）"
    [ -d "$VA_REPO_DIR/models/dtln" ]                  || va_warn "  · 可选权重缺失: models/dtln（DTLN 降噪）"
    [ -d "$VA_REPO_DIR/models/spkrec-ecapa-voxceleb" ] || va_warn "  · 可选权重缺失: models/spkrec-ecpa（声纹门控）"

    if [ "$fatal" = "1" ]; then
        va_die "预检未通过。按上面提示逐个安装，或批量执行： pip install -r requirements.txt
        （TTS 引擎缺失时请装： pip install nano-vllm-voxcpm==2.0.4
          注意它依赖 flash-attn，PyPI 上只有源码包需本地编译，详见 requirements.txt 文末）
        应急跳过预检： VA_SKIP_PREFLIGHT=1  （不推荐：缺包会在服务启动时才报错）"
    fi
    va_log "预检通过"
}

# 推断 vLLM 的量化参数：按权重目录名猜，可用 LLM_QUANT 覆盖
va_llm_quant() {
    local q="${LLM_QUANT:-}"
    if [ -z "$q" ]; then
        case "$LLM_MODEL_DIR" in
            *AWQ*|*awq*)   q="awq" ;;
            *GPTQ*|*gptq*) q="gptq_marlin" ;;
        esac
    fi
    printf '%s' "$q"
}

# 后台启动 vLLM（写日志）；调用方负责后续 va_wait_http 等就绪
va_start_vllm_bg() {
    if va_port_busy "$LLM_PORT"; then
        va_log "vLLM 已在 :$LLM_PORT 运行，跳过启动"
        return 0
    fi
    local quant; quant="$(va_llm_quant)"
    # ⚠️ 刻意不加 --enforce-eager：它会连带关掉 JIT kernel warmup，FlashInfer 的采样内核被拖到
    # 第一个真实请求才现场编译（实测耗 24 秒），撞穿 engine/llm.py 的 read=30s 超时并报
    # "LLM stream error: Request timed out."。代价只是启动多花 1~3 分钟预热，别再加回来。
    local args=(
        --model "$LLM_MODEL_DIR"
        --served-model-name "$LLM_SERVED_NAME"
        --trust-remote-code --dtype auto
        --gpu-memory-utilization "$LLM_GPU_UTIL"
        --max-model-len "$LLM_MAX_LEN"
        --max-num-seqs "$LLM_MAX_SEQS"
        --host 0.0.0.0 --port "$LLM_PORT"
    )
    [ -n "$quant" ] && args+=(--quantization "$quant")
    va_log "启动 vLLM（GPU $GPU_LLM，:$LLM_PORT，模型 $(basename "$LLM_MODEL_DIR")）"
    # 用 setsid 起：服务成为独立进程组组长（PGID == PID），停止时能一次带走它和全部子进程
    setsid env CUDA_VISIBLE_DEVICES="$GPU_LLM" "$PYTHON" \
        -m vllm.entrypoints.openai.api_server "${args[@]}" \
        > "$VLLM_LOG" 2>&1 &
    VA_VLLM_PID=$!
    va_write_pid vllm "$VA_VLLM_PID"
    va_log "vLLM 已启动（PGID $VA_VLLM_PID），日志: $VLLM_LOG"
}

# 后台启动 ws_server（Voice Agent）
va_start_voice_bg() {
    if va_port_busy "$VOICE_PORT"; then
        va_log "Voice Agent 已在 :$VOICE_PORT 运行，跳过启动"
        return 0
    fi
    va_log "启动 Voice Agent（GPU $GPU_VOICE，:$VOICE_PORT）"
    cd "$VA_REPO_DIR" || va_die "无法进入仓库根目录"
    # CUDA_VISIBLE_DEVICES 只暴露一张卡 → 进程内 cuda:0 即该物理卡
    local run=(env
        CUDA_VISIBLE_DEVICES="$GPU_VOICE"
        ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 RAG_DEVICE="$RAG_DEVICE"
        TTS_GPU_UTIL="$TTS_GPU_UTIL"
        PORT="$VOICE_PORT" DEMO_MODE="$DEMO_MODE"
        USE_FIRERED_ASR="$USE_FIRERED_ASR"
        "$PYTHON" "$VA_REPO_DIR/ws_server.py")
    # 同样用 setsid：TTS 后端 nanovllm 会再 fork 出持有 CUDA 的子进程，必须整组清理
    setsid "${run[@]}" > "$VOICE_LOG" 2>&1 &
    VA_VOICE_PID=$!
    va_write_pid voice "$VA_VOICE_PID"
    va_log "Voice Agent 已启动（PGID $VA_VOICE_PID），日志: $VOICE_LOG"
}

# 可选：反向 SSH 隧道（把本机 VOICE_PORT 暴露到 VPS）
va_start_tunnel() {
    [ "$ENABLE_TUNNEL" = "1" ] || return 0
    [ -n "$VPS_IP" ] || { va_warn "ENABLE_TUNNEL=1 但未设置 VPS_IP，跳过隧道"; return 0; }
    va_log "建立反向隧道 VPS:$VOICE_PORT -> 本机:$VOICE_PORT ($VPS_IP)"
    ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -R "$VOICE_PORT:localhost:$VOICE_PORT" "root@$VPS_IP" -N -f 2>/dev/null \
        && va_log "隧道已建立" || va_warn "隧道建立失败（后续可由 keep_alive.sh 重试）"
}

va_summary() {
    va_log "──────────────────────────────────────────────"
    va_log " 页面     : http://localhost:$VOICE_PORT/"
    va_log " 接口     : /api/info  /api/metrics  /api/rag/docs  /api/rag/query?q=..."
    va_log " 语音通道 : ws://localhost:$VOICE_PORT/ws/voice"
    va_log " LLM      : http://localhost:$LLM_PORT/v1   (模型 $LLM_SERVED_NAME)"
    va_log " 日志     : $VLLM_LOG"
    va_log "            $VOICE_LOG"
    va_log "──────────────────────────────────────────────"
}
