#!/bin/bash
# ============================================================================
# run.sh —— 唯一的启停入口：配置 + 预检 + 启动 + 守护 + 排障
# ============================================================================
# 本项目只用 **DeepSeek 云端** LLM，本地只跑 Voice Agent（ASR + TTS）一个进程，
# 所以脚本里没有任何 vLLM / 本地大模型的分支。
#
# 用法：
#   bash run.sh               前台启动，Ctrl+C 连子进程一起停（开发/演示）
#   bash run.sh dev           同上（显式写法）
#   bash run.sh start         常驻启动（脱离终端 + 自动拉起 watchdog）
#   bash run.sh stop          停止全部（含守护、按进程组清理子进程）
#   bash run.sh restart       重启
#   bash run.sh status        查看进程 / 显卡 / 配置 / 日志位置
#   bash run.sh clean-gpu     查看显卡占用者；FORCE=1 才真杀
#   bash run.sh watchdog [--once]
#                             看护循环；一般由 start 自动拉起
#   bash run.sh preflight     只跑预检，不启动任何服务
#
# 常用可覆盖的环境变量（都有合理默认值）：
#   VOICEAGENT_PYTHON=<python 绝对路径>   DEEPSEEK_API_KEY=sk-...
#   GPU_VOICE=0   VOICE_PORT=3000   RAG_DEVICE=cpu   DEMO_MODE=0
#   VA_LOG_DIR=<日志目录>   VA_SKIP_PREFLIGHT=1
#   VPS_IP=<ip>  ENABLE_TUNNEL=1（反向 SSH 隧道，默认关闭）
#   CHECK_INTERVAL=30  CHECK_TIMEOUT=300（守护节奏）
# ============================================================================
set -euo pipefail


# ============================================================================
# 1. 仓库根 / .env / 解释器 / 运行期环境
# ============================================================================

# 本脚本所在目录即仓库根：无论从哪个 cwd 调用都以它为准
VA_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VA_REPO_DIR

# 脚本自身的绝对路径与"守护进程"的命令行特征。
# 守护用它们的绝对路径做 pgrep，避免把编辑器 / grep / 调用它的 shell 自己误匹配进来。
VA_SCRIPT="$VA_REPO_DIR/run.sh"
VA_WATCHDOG_MARKER="$VA_SCRIPT watchdog"

# ---------------------------------------------------------------------------
# 自动加载仓库根目录的 .env（DEEPSEEK_API_KEY 等）。
# 必须在这里做：下面的预检要在 **shell 里** 读 DEEPSEEK_API_KEY，
# 而 config.py 的加载只覆盖 Python 侧。
# 规则：**已 export 的同名变量优先**，.env 不覆盖它们。
# 只认 `KEY=value` / `#` 注释 / 空行；非法行静默跳过，绝不因 .env 写法错误中断启动。
# ---------------------------------------------------------------------------
va_load_dotenv() {
    local f="$VA_REPO_DIR/.env" line key val
    [ -r "$f" ] || return 0
    local loaded=0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"          # 去掉行首空白
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"
        val="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"             # 去掉 key 尾部空白
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        # 剥掉成对的首尾引号（允许值里有空格）
        if [ "${#val}" -ge 2 ]; then
            case "$val" in
                \"*\") val="${val#\"}"; val="${val%\"}" ;;
                \'*\') val="${val#\'}"; val="${val%\'}" ;;
            esac
        fi
        # 已存在（哪怕是空串）就跳过：命令行优先
        [ -n "${!key+x}" ] && continue
        export "$key=$val"
        # 只有真有值才计入提示：.env 里 KEY= 这种空占位不该每次刷一行输出
        [ -n "$val" ] && loaded=$((loaded + 1))
    done < "$f"
    [ "$loaded" -gt 0 ] && echo "[配置] 已从 .env 加载 $loaded 项" >&2
    return 0
}
va_load_dotenv

# 解释器解析顺序：显式 VOICEAGENT_PYTHON → PATH 里的 python3/python；不硬编码 conda 路径
# 目标 conda 环境名。想用别的环境改这里，或直接设 VOICEAGENT_PYTHON。
VA_CONDA_ENV_NAME="${VA_CONDA_ENV_NAME:-voiceagent}"

# 找一个 conda 环境里的 python 解释器。成功时打印绝对路径并返回 0。
# 探测顺序（越靠前越可靠）：
#   ① conda env list  的文本输出（最后一列是环境路径）—— 最权威，跟着用户的 conda 配置走
#   ② 当前已激活环境的同级 envs/ —— CONDA_PREFIX=/root，则看 /root/envs/<name>
#   ③ 常见安装根目录 —— 覆盖 conda 不在 PATH、也没激活过的情况
# 不直接 source conda.sh 再 activate：那会改整个 shell 的环境，
# 而本脚本只需要一个解释器路径（下面已把它的 bin 前置到 PATH）。
va_find_conda_env_python() {
    local name="$1" cand root
    # ① conda env list
    if command -v conda >/dev/null 2>&1; then
        cand="$(conda env list 2>/dev/null \
                | awk '{print $NF}' \
                | grep -E "/envs/${name}$" | head -1)"
        if [ -n "$cand" ] && [ -x "$cand/bin/python" ]; then
            printf '%s' "$cand/bin/python"; return 0
        fi
    fi
    # ② 已激活环境的同级
    if [ -n "${CONDA_PREFIX:-}" ]; then
        cand="$(dirname "$CONDA_PREFIX")/envs/$name/bin/python"
        if [ -x "$cand" ]; then printf '%s' "$cand"; return 0; fi
    fi
    # ③ 常见根目录
    for root in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
                /opt/conda /opt/miniforge3 /opt/miniconda3; do
        cand="$root/envs/$name/bin/python"
        if [ -x "$cand" ]; then printf '%s' "$cand"; return 0; fi
    done
    return 1
}

va_resolve_python() {
    # ① 显式指定最高优先
    if [ -n "${VOICEAGENT_PYTHON:-}" ]; then
        [ -x "$VOICEAGENT_PYTHON" ] || { echo "[错误] VOICEAGENT_PYTHON 指向的不是可执行文件: $VOICEAGENT_PYTHON" >&2; exit 1; }
        echo "$VOICEAGENT_PYTHON"; return 0
    fi
    # ② 自动挑 conda 里的 voiceagent 环境（本项目依赖都装在这里）
    local envpy
    if envpy="$(va_find_conda_env_python "$VA_CONDA_ENV_NAME")"; then
        echo "[配置] 自动使用 conda 环境 $VA_CONDA_ENV_NAME: $envpy" >&2
        echo "$envpy"; return 0
    fi
    # ③ 兜底：PATH 里的 python（在 base 或系统环境里时 va_check_interpreter 会告警）
    local candidate
    for candidate in python python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "[警告] 没找到 conda 环境 $VA_CONDA_ENV_NAME，退回 PATH 里的 $(command -v "$candidate")" >&2
            command -v "$candidate"; return 0
        fi
    done
    echo "[错误] 找不到 python。请先 'conda create -n voiceagent python=3.11'，或设置 VOICEAGENT_PYTHON=<解释器绝对路径>" >&2
    exit 1
}

PYTHON="$(va_resolve_python)"
export PYTHON
# 把解释器所在目录前置到 PATH：让脚本里裸调的 pip / ninja 等也命中同一环境
export PATH="$(dirname "$PYTHON"):$PATH"

# 若选中的确实是 conda 环境，补齐 CONDA_* ，让子进程（pip / 引擎里的 conda 探测）行为一致。
# 判据是 conda 环境特有的 conda-meta/history，避免把 venv 误判成 conda。
_PY_PREFIX="$(dirname "$(dirname "$PYTHON")")"
if [ -f "$_PY_PREFIX/conda-meta/history" ]; then
    export CONDA_PREFIX="$_PY_PREFIX"
    export CONDA_DEFAULT_ENV="$(basename "$_PY_PREFIX")"
fi
unset _PY_PREFIX

# 运行期环境变量：离线优先，避免现场联网卡住
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"          # HF 走国内镜像
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$VA_REPO_DIR/models}"  # 权重统一放仓库内
# HF 缓存也放仓库内（models/.hf）：跟项目在一起、可整体迁移，
# 且 models/* 已在 .gitignore 里，不会被误提交。
# Kokoro 的权重就落在这儿（默认 ~/.cache/huggingface 可能只读或不可控）。
export HF_HOME="${HF_HOME:-$VA_REPO_DIR/models/.hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"                        # 只用本地权重，不联网


# ============================================================================
# 2. 配置：模型路径与端口一律从 config.py 读取（脚本因此不可能与代码漂移）
# ============================================================================

# config.py 只 import os，可安全单独加载取值
va_cfg() {
    "$PYTHON" -c "
import importlib.util, sys, os
p = os.path.join(os.environ['VA_REPO_DIR'], 'config.py')
spec = importlib.util.spec_from_file_location('va_cfg', p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(getattr(m, sys.argv[1]))
" "$1" 2>/dev/null
}

# 三个必需权重目录：从 config.py 读，避免脚本与代码各写一份路径
VA_CFG_VAD_DIR="$(va_cfg VAD_MODEL_DIR)"
VA_CFG_ASR_DIR="$(va_cfg ASR_MODEL_DIR)"
VA_CFG_EMBED_DIR="$(va_cfg EMBED_MODEL_DIR)"
VA_VAD_DIR="${VA_CFG_VAD_DIR:-$VA_REPO_DIR/models/snakers4_silero-vad}"

# GPU 分配：Voice Agent（ASR + TTS）用哪张物理卡
GPU_VOICE="${GPU_VOICE:-0}"

# Voice Agent：ASR + RAG + TTS 都在这一个进程里
VOICE_PORT="${VOICE_PORT:-3000}"
RAG_DEVICE="${RAG_DEVICE:-cpu}"        # 检索只需几毫秒，放 CPU 把显存留给 ASR/TTS
DEMO_MODE="${DEMO_MODE:-0}"

# 可选：反向 SSH 隧道（把本机 VOICE_PORT 暴露到 VPS）
ENABLE_TUNNEL="${ENABLE_TUNNEL:-0}"
VPS_IP="${VPS_IP:-}"

# 守护节奏
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"   # 巡检间隔（秒）
CHECK_TIMEOUT="${CHECK_TIMEOUT:-300}"    # 单次重启后就绪等待上限（秒）


# ============================================================================
# 3. 日志与目录
# ============================================================================

# 日志默认放仓库 logs/（跟项目在一起、不会被系统清理），可用 VA_LOG_DIR 覆盖
VA_LOG_DIR="${VA_LOG_DIR:-$VA_REPO_DIR/logs}"
VA_PID_DIR="$VA_LOG_DIR"
VOICE_LOG="$VA_LOG_DIR/voice_agent.log"
KEEPALIVE_LOG="$VA_LOG_DIR/keepalive.log"
mkdir -p "$VA_LOG_DIR"

VA_VOICE_PID=""
CLEANED=0

va_log()  { echo "$(date '+%H:%M:%S') $*"; }
va_warn() { echo "$(date '+%H:%M:%S') [警告] $*" >&2; }
va_die()  { echo "$(date '+%H:%M:%S') [错误] $*" >&2; exit 1; }

va_has_py() { "$PYTHON" -c "import $1" >/dev/null 2>&1; }


# ============================================================================
# 4. 解释器体检
# ============================================================================

va_check_interpreter() {
    case "$PYTHON" in
        /usr/bin/python*|/bin/python*)
            va_warn "当前用的是系统 Python（$PYTHON）—— 没找到 conda 环境 $VA_CONDA_ENV_NAME。"
            va_warn "  建环境： conda create -y -n $VA_CONDA_ENV_NAME python=3.11"
            va_warn "  或指定： export VOICEAGENT_PYTHON=/path/to/envs/$VA_CONDA_ENV_NAME/bin/python"
            ;;
    esac

    # conda base 能用，但把项目依赖装进 base 很容易和别的项目打架
    local prefix
    prefix="$("$PYTHON" -c 'import sys;print(sys.prefix)' 2>/dev/null)"
    case "$prefix" in
        */miniforge3|*/miniconda3|*/anaconda3|*/miniforge)
            va_warn "当前是 conda 的 base 环境（$prefix），不是项目专用环境 $VA_CONDA_ENV_NAME。"
            va_warn "  建环境： conda create -y -n $VA_CONDA_ENV_NAME python=3.11"
            va_warn "  或指定： export VOICEAGENT_PYTHON=/path/to/envs/$VA_CONDA_ENV_NAME/bin/python"
            ;;
    esac

    # 本依赖清单按 Python 3.11 验证；其它版本各包都有轮子，但未逐一验证
    local ver
    ver="$("$PYTHON" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
    if [ -n "$ver" ] && [ "$ver" != "3.11" ]; then
        va_warn "Python 版本为 $ver，而本依赖清单按 3.11 验证（其余版本可装但未验证）。"
    fi
}


# ============================================================================
# 5. 端口 / HTTP / 进程组
# ============================================================================

# 端口是否已被监听：优先 ss（Debian 自带），退化到 /dev/tcp；不依赖 lsof
va_port_busy() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        [ -n "$(ss -H -ltn "sport = :$port" 2>/dev/null)" ]
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && { exec 3<&- 3>&-; return 0; } || return 1
    fi
}

# 按端口找出监听进程的 PID（用 ss，不依赖 lsof）
va_pids_on_port() {
    local port="$1"
    ss -H -ltnp "sport = :$port" 2>/dev/null \
        | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
}

# 轮询一个 HTTP 地址直到可用。用法：va_wait_http <url> <超时秒> <说明> [被监视的PID]
# 必须看 HTTP 状态码：curl 对 404/500 同样返回 0，只看退出码会把"服务起了但接口报错"误判成就绪。
#
# 两个关键行为：
#   1. 传入 PID 后**进程一死就立刻返回**——否则引擎加载崩了还要白等满超时（踩过）。
#   2. HTTP 000 = 端口还没监听。本项目 load_engines() 在 uvicorn.run() **之前**跑，
#      所以加载的 60~120s 里一直是 000，属正常；只在每 20s 报一次进度，不刷屏。
va_wait_http() {
    local url="$1" timeout="${2:-120}" label="${3:-服务}" watch_pid="${4:-}"
    local i=0 code=""
    va_log "等待 $label 就绪（最多 ${timeout}s）…"
    while [ "$i" -lt "$timeout" ]; do
        # 被监视的进程已经没了 → 没必要再等，直接失败让调用方去 tail 日志
        if [ -n "$watch_pid" ] && ! kill -0 "$watch_pid" 2>/dev/null; then
            va_warn "$label 进程已退出（等待了 ${i}s），停止等待"
            return 1
        fi
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$url" 2>/dev/null)"
        case "$code" in
            2*) va_log "$label 已就绪（HTTP $code，用时 ${i}s）"; return 0 ;;
            000|"") : ;;                    # 还没开始监听：加载期正常，静默
            *)  va_warn "$label 返回 HTTP $code，继续等待…" ;;
        esac
        if [ "$i" -gt 0 ] && [ $((i % 20)) -eq 0 ]; then
            va_log "  仍在加载…已等 ${i}s（端口尚未监听，属正常）"
        fi
        sleep 2; i=$((i + 2))
    done
    va_warn "$label 在 ${timeout}s 内未就绪（最后 HTTP 状态: ${code:-无响应}），请查看日志"
    return 1
}

# 进程组管理（关键）：TTS 后端可能 fork 出持有 CUDA 上下文的子进程，
# 只 kill 父进程的话它变成孤儿继续占显存，下次启动就直接 CUDA OOM。
# 所以服务一律用 setsid 起（PGID == PID）并落盘 PID，停止时用 `kill -TERM -PGID` 一次带走整组。
va_write_pid() { printf '%s\n' "$2" > "$VA_PID_DIR/$1.pid" 2>/dev/null || true; }
va_read_pid()  { [ -r "$VA_PID_DIR/$1.pid" ] && cat "$VA_PID_DIR/$1.pid" 2>/dev/null || true; }
va_clear_pid() { rm -f "$VA_PID_DIR/$1.pid" 2>/dev/null || true; }

# 杀某个服务及其全部子孙进程。用法：va_kill_group <voice> [TERM|KILL]
va_kill_group() {
    local name="$1" sig="${2:-TERM}" pgid
    pgid="$(va_read_pid "$name")"
    [ -n "$pgid" ] || return 1
    # 负号表示"进程组"；即使组长已死，组内其它进程仍会被送达
    kill "-$sig" "-$pgid" 2>/dev/null || kill "-$sig" "$pgid" 2>/dev/null || true
    [ "$sig" = "KILL" ] && va_clear_pid "$name"
    return 0
}

# 守护进程的 PID 列表。用**绝对路径**匹配并排除自身 PID：
# pgrep -f 是按整条命令行做正则，只写 "run.sh watchdog" 会把编辑器、grep、乃至调用它的 shell 自己都误匹配。
va_watchdog_pids() {
    pgrep -f "$VA_WATCHDOG_MARKER" 2>/dev/null | grep -vw "$$" | grep -vw "$PPID" || true
}

va_watchdog_running() { [ -n "$(va_watchdog_pids)" ]; }


# ============================================================================
# 6. GPU 体检
# ============================================================================

# 驱动是否真的可用。注意：驱动异常时 nvidia-smi 会把错误文本写到 **stdout** 且退出码为 0，
# 所以只能按"输出形状"判断（有效行以数字开头），不能只看退出码或字符串非空。
va_gpu_ready() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local out
    out="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)"
    printf '%s\n' "$out" | grep -qE '^[0-9]'
}

# 返回该卡总显存 MiB；查不到就返回非 0 且无输出
va_gpu_total() {
    va_gpu_ready || return 1
    local out
    out="$(nvidia-smi -i "$1" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)" || return 1
    # 只保留数字：驱动异常时 nvidia-smi 会打错误文本，混进整数比较会报表达式错误
    out="$(printf '%s' "$out" | tr -dc '0-9')"
    [ -n "$out" ] || return 1
    printf '%s' "$out"
}

# 启动前的显存体检：最容易踩的坑是"上次的子进程没退干净 → 这次加载时 CUDA OOM"，
# 直接把占用者列出来比事后翻日志快。没有可用 GPU 时静默跳过。
# 用法：va_check_gpu_free <物理卡号> <至少需要MiB> <用途>
va_check_gpu_free() {
    va_gpu_ready || return 0
    local gpu="$1" need="$2" what="${3:-服务}" free
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)" || return 0
    free="$(printf '%s' "$free" | tr -dc '0-9')"
    [ -n "$free" ] || return 0
    if [ "$free" -lt "$need" ]; then
        va_warn "GPU $gpu 空闲显存只有 ${free} MiB，而 $what 约需 ${need} MiB —— 很可能 OOM"
        va_warn "  当前占用 GPU $gpu 的进程："
        nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory \
                   --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        va_warn "  若这些是上次没退干净的残留（尤其 TTS 子进程）："
        va_warn "    bash run.sh stop              # 按进程组 + 端口清理"
        va_warn "    bash run.sh clean-gpu         # 看是谁占着；FORCE=1 真杀"
        return 1
    fi
    va_log "GPU $gpu 空闲显存 ${free} MiB（$what 约需 ${need} MiB）"
    return 0
}


# ============================================================================
# 7. VAD 权重检查
# ============================================================================

# engine/vad.py 直接读 config.VAD_MODEL_DIR 下的 silero_vad.jit，
# 找不到时会退化成 torch.hub **联网**下载（离线环境直接失败，表现是"卡在 Loading VAD 很久"）。
# 常见坑：GitHub 上的 snakers4/silero-vad 是源码仓库，.jit 埋在 src/silero_vad/data/ 下。
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
    # 且调用方都在 set -e 下，返回非 0 会把整个启动流程打断。
}


# ============================================================================
# 8. 预检：把"缺什么、装什么"一次说清楚，避免服务起一半才报 ImportError
# ============================================================================

va_preflight() {
    [ "${VA_SKIP_PREFLIGHT:-0}" = "1" ] && { va_warn "已跳过预检（VA_SKIP_PREFLIGHT=1）"; return 0; }

    va_check_interpreter

    local fatal=0
    va_log "预检"

    # --- LLM 凭据：本项目只用 DeepSeek 云端 API ---
    if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
        va_log "  ✓ DEEPSEEK_API_KEY 已设置"
    else
        va_warn "  ✗ 缺少 DEEPSEEK_API_KEY —— 写进 .env 或 export DEEPSEEK_API_KEY=sk-..."; fatal=1
    fi

    # --- Python 包：按"缺了就直接起不来"排序 ---
    local pair module
    for pair in "torch:torch" "numpy:numpy" \
                "transformers:transformers" \
                "sentence_transformers:sentence-transformers" "faiss:faiss-cpu" \
                "kokoro:kokoro" "misaki:misaki[en]" \
                "fastapi:fastapi" "uvicorn:uvicorn" "starlette:starlette" \
                "openai:openai" "httpx:httpx" \
                "nest_asyncio:nest-asyncio"; do
        module="${pair%%:*}"
        if va_has_py "$module"; then va_log "  ✓ $module"
        else va_warn "  ✗ 缺少 $module —— 安装： pip install ${pair##*:}"; fatal=1; fi
    done

    # --- 模型权重 ---
    # ASR 目录必须**有实际权重**才算就位：只检查目录会在"下了一半"时误判通过，
    # 服务起来后才在加载阶段炸（Parakeet 需要 model.safetensors + config.json + tokenizer.json）。
    local asr_missing=""
    [ -f "$VA_CFG_ASR_DIR/config.json" ] || asr_missing="$asr_missing config.json"
    [ -f "$VA_CFG_ASR_DIR/tokenizer.json" ] || asr_missing="$asr_missing tokenizer.json"
    compgen -G "$VA_CFG_ASR_DIR/*.safetensors" >/dev/null || asr_missing="$asr_missing model.safetensors"
    # 权重被截断时文件存在但内容不对，用大小兜底（Parakeet-TDT-0.6B 约 2.5GB）
    local asr_bytes
    asr_bytes=$(du -sb "$VA_CFG_ASR_DIR" 2>/dev/null | cut -f1)
    if [ -n "$asr_bytes" ] && [ "$asr_bytes" -lt 1000000000 ]; then
        asr_missing="$asr_missing 权重体积不足($((asr_bytes/1048576))MB<1GB)"
    fi
    if [ -z "$asr_missing" ]; then
        va_log "  ✓ ASR 权重 ${VA_CFG_ASR_DIR#$VA_REPO_DIR/}"
    else
        va_warn "  ✗ ASR 权重目录不完整: $VA_CFG_ASR_DIR（缺$asr_missing）"
        va_warn "      获取： huggingface-cli download nvidia/parakeet-tdt-0.6b-v3 \\"
        va_warn "              --local-dir ${VA_CFG_ASR_DIR#$VA_REPO_DIR/}"
        fatal=1
    fi
    [ -d "$VA_CFG_EMBED_DIR" ] \
        && va_log "  ✓ RAG 权重 ${VA_CFG_EMBED_DIR#$VA_REPO_DIR/}" \
        || { va_warn "  ✗ RAG 权重缺失: $VA_CFG_EMBED_DIR"; fatal=1; }
    # Kokoro 权重**放本地目录**（models/Kokoro-82M），不走 HuggingFace 缓存。
    # 目录里必须齐：config.json + 一个 *.pth + voices/<音色>.pt。
    local kokoro_dir kokoro_voice
    kokoro_dir="$(va_cfg KOKORO_MODEL_DIR)"
    kokoro_voice="$("$PYTHON" -c "
import sys; sys.path.insert(0,'$VA_REPO_DIR')
from engine.tts import KokoroTTS; print(KokoroTTS.VOICE)" 2>/dev/null || echo af_heart)"
    local kokoro_missing=""
    [ -f "$kokoro_dir/config.json" ] || kokoro_missing="$kokoro_missing config.json"
    compgen -G "$kokoro_dir/*.pth" >/dev/null || kokoro_missing="$kokoro_missing *.pth"
    [ -f "$kokoro_dir/voices/$kokoro_voice.pt" ] || kokoro_missing="$kokoro_missing voices/$kokoro_voice.pt"
    if [ -z "$kokoro_missing" ]; then
        va_log "  ✓ TTS 权重 ${kokoro_dir#$VA_REPO_DIR/}（音色 $kokoro_voice）"
    else
        va_warn "  ✗ TTS 权重目录不完整: $kokoro_dir（缺$kokoro_missing）"
        va_warn "      获取： huggingface-cli download hexgrad/Kokoro-82M --local-dir ${kokoro_dir#$VA_REPO_DIR/}"
        fatal=1
    fi
    # misaki 的英文 G2P 依赖 spaCy 的 en_core_web_sm。模型**放仓库里**
    # （models/en_core_web_sm/，由 server/bootstrap.py 加进 sys.path），
    # 不依赖 pip 装包；缺了会在建 KPipeline 时联网下载并失败。
    local spacy_dir="$VA_REPO_DIR/models/en_core_web_sm"
    if [ -d "$spacy_dir/en_core_web_sm" ] && compgen -G "$spacy_dir/*.dist-info" >/dev/null; then
        va_log "  ✓ spaCy 模型 models/en_core_web_sm"
    else
        va_warn "  ✗ 缺少 spaCy 模型目录: $spacy_dir"
        va_warn "      说明见 README「下载模型」；获取方式（GitHub 不通时加代理前缀）："
        va_warn "        pip download --no-deps -d /tmp en_core_web_sm==3.8.0 失败时改用官方 wheel："
        va_warn "        https://github.com/explosion/spacy-models/releases/download/\\"
        va_warn "          en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        fatal=1
    fi

    if [ "$fatal" = "1" ]; then
        va_die "预检未通过。按上面 ✗ 的提示逐条解决，常见四类：
        ① 缺 Python 包 → pip install -r requirements.txt
           仅缺 TTS： pip install kokoro==0.9.4   （英文发音还需要 sudo apt install espeak-ng）
        ② 缺模型权重 → 见 README「下载模型」（含 models/Kokoro-82M）
        ③ 缺 spaCy 模型 → 下载解压到 models/en_core_web_sm/（见 README「下载模型」）
        ④ 缺 DEEPSEEK_API_KEY → export DEEPSEEK_API_KEY=sk-...（从 platform.deepseek.com 获取）
        应急跳过预检： VA_SKIP_PREFLIGHT=1  （不推荐：问题会推迟到服务启动时才暴露）"
    fi
    va_log "预检通过"
}


# ============================================================================
# 9. 启动 / 停止
# ============================================================================

# 后台启动 ws_server（Voice Agent）
va_start_voice_bg() {
    if va_port_busy "$VOICE_PORT"; then
        va_log "Voice Agent 已在 :$VOICE_PORT 运行，跳过启动"
        return 0
    fi
    va_log "启动 Voice Agent（GPU $GPU_VOICE，:$VOICE_PORT）"
    cd "$VA_REPO_DIR" || va_die "无法进入仓库根目录"
    # 设备选择的三层优先级：
    #   ① 用户显式 export 的 CUDA_VISIBLE_DEVICES / ASR_DEVICE / TTS_DEVICE —— 原样尊重
    #      （想强制 CPU 就 `ASR_DEVICE=cpu TTS_DEVICE=cpu bash run.sh`）
    #   ② 否则只暴露 GPU_VOICE 一张卡，进程内 cuda:0 即该物理卡
    # 注意 CUDA_VISIBLE_DEVICES 与 *_DEVICE 是两层：前者选"暴露哪些物理卡"，
    # 后者是"进程内序号"。设了 CUDA_VISIBLE_DEVICES 后语音栈就写 cuda:0。
    local vis_dev asr_dev tts_dev
    vis_dev="${CUDA_VISIBLE_DEVICES:-$GPU_VOICE}"
    asr_dev="${ASR_DEVICE:-cuda:0}"
    tts_dev="${TTS_DEVICE:-cuda:0}"
    va_log "  设备: CUDA_VISIBLE_DEVICES=$vis_dev ASR=$asr_dev TTS=$tts_dev RAG=$RAG_DEVICE"
    local run=(env
        CUDA_VISIBLE_DEVICES="$vis_dev"
        ASR_DEVICE="$asr_dev" TTS_DEVICE="$tts_dev" RAG_DEVICE="$RAG_DEVICE"
        PORT="$VOICE_PORT" DEMO_MODE="$DEMO_MODE"
        "$PYTHON" "$VA_REPO_DIR/ws_server.py")
    # 用 setsid：TTS 后端可能 fork 出持有 CUDA 的子进程，必须整组清理
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
        && va_log "隧道已建立" || va_warn "隧道建立失败（后续可由 watchdog 重试）"
}

# 拉起 Voice Agent 并等就绪；已在运行的就复用。dev 与 start 共用这一段。
va_bring_up_service() {
    if va_port_busy "$VOICE_PORT"; then
        va_log "Voice Agent 已在 :$VOICE_PORT 运行，复用"
        VA_VOICE_PID="$(va_read_pid voice)"
        va_start_tunnel
        return 0
    fi
    va_log "启动 Voice Agent（依次加载 VAD/ASR/RAG/TTS，通常 60~120s）…"
    va_start_voice_bg
    if ! va_wait_http "http://localhost:$VOICE_PORT/api/info" 300 "Voice Agent" "$VA_VOICE_PID"; then
        va_warn "Voice Agent 未就绪，日志末尾 30 行："
        tail -n 30 "$VOICE_LOG" 2>/dev/null | sed 's/^/  /'
        va_warn "完整日志: $VOICE_LOG"
        return 1
    fi
    va_start_tunnel
    return 0
}

# 启动失败时统一收尾：停掉半死的进程组，保留非 0 退出码
va_fail_start() {
    va_kill_group voice TERM || true
    sleep 2
    va_kill_group voice KILL || true
    exit 1
}

# 把最常用的地址与日志位置打出来
va_summary() {
    va_log "──────────────────────────────────────────────"
    va_log " 页面     : http://localhost:$VOICE_PORT/"
    va_log " 接口     : /api/info  /api/metrics  /api/rag/docs  /api/rag/query?q=..."
    va_log " 语音通道 : ws://localhost:$VOICE_PORT/ws/voice"
    va_log " LLM      : $("$PYTHON" -c "import sys;sys.path.insert(0,'$VA_REPO_DIR');import config;print(config.LLM_MODEL_NAME+' @ '+config.LLM_BASE_URL)" 2>/dev/null || echo 'DeepSeek 云端')"
    va_log " 日志     : $VOICE_LOG"
    va_log "──────────────────────────────────────────────"
}


# ============================================================================
# 10. 子命令 dev：前台托管（Ctrl+C 连子进程一起停）
#     适合开发调试 / 现场演示；长期无人值守请用 start。
# ============================================================================

va_cleanup_foreground() {
    local code=$?                 # 先接住触发本次清理的退出码，最后原样返回
    [ "$CLEANED" = "1" ] && return 0
    CLEANED=1
    trap - INT TERM EXIT          # 先摘掉 trap，避免 exit 递归触发
    # 只有真的启动过服务才打印停止过程；预检就失败时不必输出这段
    if [ -n "${VA_VOICE_PID:-}" ]; then
        echo
        va_log "正在停止本次启动的服务…"
        # 按【进程组】杀：只杀父进程会留下持有 CUDA 的 TTS 孤儿，占显存
        va_kill_group voice TERM || true
        sleep 3
        va_kill_group voice KILL || true
        # 最后按端口兜底（覆盖"不是本脚本起的"残留监听）
        if va_port_busy "$VOICE_PORT"; then
            va_warn "端口 $VOICE_PORT 仍被占用，可用 'bash run.sh stop' 或手工 kill 清理"
        fi
        va_log "已停止"
    fi
    exit "$code"                  # 保留原始退出码：预检失败就该以非 0 退出
}

va_do_dev() {
    trap va_cleanup_foreground INT TERM EXIT

    echo "=========================================="
    echo "  VoiceAgent-600ms — 级联 Pipeline 启动"
    echo "=========================================="

    va_check_vad_path
    va_preflight

    # 显存体检：LLM 在云端，本地只有 ASR + TTS
    local voice_total
    voice_total="$(va_gpu_total "$GPU_VOICE" || true)"
    if [ -n "$voice_total" ]; then
        va_check_gpu_free "$GPU_VOICE" $(( voice_total * 60 / 100 )) \
            "Voice Agent（ASR + TTS）" || true
    fi

    va_bring_up_service || va_fail_start

    # 就绪后仍可能立刻崩（典型是加载阶段 CUDA OOM），主动确认一次，
    # 避免出现"报了就绪、其实已经死了"这种误导。
    if ! va_port_busy "$VOICE_PORT" && ! pgrep -f "$VA_REPO_DIR/ws_server.py" >/dev/null 2>&1; then
        echo
        va_warn "Voice Agent 启动后立刻退出了，日志末尾 25 行："
        tail -n 25 "$VOICE_LOG" 2>/dev/null | sed 's/^/  /'
        echo
        va_warn "完整日志: $VOICE_LOG"
        va_cleanup_foreground
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
        local code hint
        wait "$VA_VOICE_PID" 2>/dev/null || true
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
    va_cleanup_foreground
}


# ============================================================================
# 11. 子命令 start / stop / status / restart / clean-gpu（常驻模式）
# ============================================================================

# 常驻启动：setsid 脱离终端，关掉 SSH 也继续跑，并顺带拉起 watchdog。
va_do_start() {
    va_check_vad_path
    va_preflight

    va_log "=== 启动（常驻模式）==="

    va_bring_up_service || va_fail_start

    # 守护进程（脱离终端常驻）
    if va_watchdog_running; then
        va_log "守护进程已在运行，跳过"
    else
        setsid nohup bash "$VA_SCRIPT" watchdog > "$KEEPALIVE_LOG" 2>&1 &
        va_log "守护进程已启动（日志 $KEEPALIVE_LOG）"
    fi

    va_summary
    va_log "查看状态： bash run.sh status"
    va_log "停止服务： bash run.sh stop"
}

va_do_stop() {
    va_log "停止服务…"

    # 先停守护，避免它把我们正在停的服务又拉起来（同样用绝对路径 + 排除自身）
    local watchdog_pids
    watchdog_pids="$(va_watchdog_pids)"
    if [ -n "$watchdog_pids" ]; then
        va_log "  停止守护进程: $(echo "$watchdog_pids" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill $watchdog_pids 2>/dev/null || true
    fi
    sleep 1

    # 按【进程组】停：TTS 子进程**不监听任何端口**，按端口找 PID 杀不到它 ——
    # 它会变成孤儿继续占显存，导致下次启动 OOM（实际踩过的坑）。
    va_log "  按进程组停止 Voice Agent（含 TTS 子进程）…"
    va_kill_group voice TERM || true
    sleep 3
    va_kill_group voice KILL || true

    # 再按端口兜底（覆盖不是本脚本启动的残留监听进程）
    local pids
    pids="$(va_pids_on_port "$VOICE_PORT")"
    if [ -n "$pids" ]; then
        va_log "  停止端口 $VOICE_PORT 上的进程: $(echo "$pids" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
    fi
    sleep 2

    # 仍在监听就用 SIGKILL（uvicorn 有时忽略 SIGTERM 的优雅退出）
    pids="$(va_pids_on_port "$VOICE_PORT")"
    if [ -n "$pids" ]; then
        va_warn "  端口 $VOICE_PORT 仍未释放，强制结束"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
    va_clear_pid voice
    va_log "已停止"
}

va_do_status() {
    echo "──────────────────────────────────────────────"
    printf "  %-22s %s\n" "进程" "状态"
    printf "  %-22s %s\n" "Voice Agent (:$VOICE_PORT)" \
        "$(va_port_busy "$VOICE_PORT" && echo "运行中 (PID $(va_pids_on_port "$VOICE_PORT" | tr '\n' ' '))" || echo "未运行")"
    printf "  %-22s %s\n" "守护 watchdog" \
        "$(pids="$(va_watchdog_pids)"; [ -n "$pids" ] && echo "运行中 (PID $(echo "$pids" | tr '\n' ' '))" || echo "未运行")"
    if [ "$ENABLE_TUNNEL" = "1" ] && [ -n "$VPS_IP" ]; then
        printf "  %-22s %s\n" "SSH 隧道" \
            "$(pgrep -f "ssh .*-R ${VOICE_PORT}.*${VPS_IP}" >/dev/null && echo 运行中 || echo 未运行)"
    fi
    echo "──────────────────────────────────────────────"
    # 一次性读出所有要展示的配置（只起一个 python 进程，避免 status 卡顿）
    local _info
    _info="$("$PYTHON" -c "
import os, sys
sys.path.insert(0, '$VA_REPO_DIR')
import config
print(config.VERSION)
print(os.path.basename(config.ASR_MODEL_DIR))
print(os.path.basename(config.EMBED_MODEL_DIR))
print(config.LLM_MODEL_NAME)
" 2>/dev/null || true)"
    local ver asr_m embed_m llm_m
    ver="$(printf '%s\n' "$_info" | sed -n 1p)"
    asr_m="$(printf '%s\n' "$_info" | sed -n 2p)"
    embed_m="$(printf '%s\n' "$_info" | sed -n 3p)"
    llm_m="$(printf '%s\n' "$_info" | sed -n 4p)"
    printf "  %-22s %s\n" "版本" "v${ver:-未知}"
    printf "  %-22s %s\n" "模型" "ASR=${asr_m:-?}  EMBED=${embed_m:-?}"
    printf "  %-22s %s\n" "配置" "GPU_VOICE=$GPU_VOICE  RAG_DEVICE=$RAG_DEVICE"
    printf "  %-22s %s\n" "" "LLM ${llm_m:-deepseek-flash}（DeepSeek 云端）"
    echo "──────────────────────────────────────────────"
    # 只有真能取到数据才打印 GPU 段：驱动异常时 nvidia-smi 会把错误文本写到 **stdout**，
    # 直接透传会显示成一堆乱入的 "NVIDIA-SMI has failed..."，比不显示更误导。
    local gpu_out=""
    if va_gpu_ready; then
        gpu_out="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                    --format=csv,noheader 2>/dev/null | grep -E '^[0-9]' || true)"
    fi
    if [ -n "$gpu_out" ]; then
        echo "  GPU:"
        printf '%s\n' "$gpu_out" | sed 's/^/    /'
    else
        echo "  GPU: 不可用（无 nvidia-smi，或驱动/设备未就绪）"
    fi
    echo "──────────────────────────────────────────────"
    echo "  日志: $VOICE_LOG"
    echo "        $KEEPALIVE_LOG"
    echo "──────────────────────────────────────────────"
}

# clean-gpu：TTS 子进程**不监听任何端口**，stop 按端口抓不到它；一旦服务异常退出，
# 它就变成孤儿占显存，导致下次启动 CUDA OOM。默认只列出占用者，FORCE=1 才真杀。
va_do_clean_gpu() {
    if ! va_gpu_ready; then
        va_warn "GPU 不可用（没有 nvidia-smi，或驱动/设备未就绪），无法检查显存占用"
        return 0
    fi
    local found=0 apps
    echo "──── GPU $GPU_VOICE ────"
    apps="$(nvidia-smi -i "$GPU_VOICE" --query-compute-apps=pid,process_name,used_memory \
                --format=csv,noheader 2>/dev/null | grep -E '^[[:space:]]*[0-9]' || true)"
    if [ -z "$apps" ]; then
        echo "  （无进程占用）"
    else
        echo "$apps" | sed 's/^/  /'
        found=1
    fi
    if [ "$found" = "0" ]; then va_log "该卡很干净，无需清理"; return 0; fi

    if [ "${FORCE:-0}" != "1" ]; then
        echo
        va_warn "以上是当前占用 GPU $GPU_VOICE 的进程。若确认都是本项目的残留，执行："
        va_warn "  FORCE=1 bash $VA_SCRIPT clean-gpu"
        va_warn "（注意：它会杀掉这张卡上的**所有**计算进程，包括别人的任务）"
        return 0
    fi

    echo
    va_warn "FORCE=1：开始清理 GPU $GPU_VOICE 上的计算进程…"
    local pids pid
    pids="$(nvidia-smi -i "$GPU_VOICE" --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | grep -E '^[[:space:]]*[0-9]' || true)"
    for pid in $pids; do
        pid="$(printf '%s' "$pid" | tr -dc '0-9')"
        [ -n "$pid" ] || continue
        va_log "  kill -9 $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
    va_log "清理完成，当前显存："
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null \
        | grep -E '^[0-9]' | sed 's/^/  /' || true
}


# ============================================================================
# 12. 守护
# ============================================================================

# 可选：清理占用 GPU 但不属于本服务的孤儿进程。默认关闭 —— 它会杀掉同机其它 CUDA 进程，
# 多人共用显卡时请勿开启。
va_clean_orphans() {
    [ "${CLEAN_ORPHANS:-0}" = "1" ] || return 0
    va_gpu_ready || return 0
    va_log "清理 GPU 上的孤儿进程（CLEAN_ORPHANS=1）…"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read -r pid; do
        [ -z "$pid" ] && continue
        # 跳过我们自己的 Voice Agent
        if va_port_busy "$VOICE_PORT"; then continue; fi
        va_log "  kill -9 $pid"
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3   # 等驱动回收显存
}

va_check_once() {
    if ! va_port_busy "$VOICE_PORT"; then
        va_warn "Voice Agent 已下线，正在重启…"
        va_clean_orphans
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

va_do_watchdog() {
    # ⚠️ 关掉 set -e：守护进程必须能容忍单次巡检失败，否则一次短暂异常就整体退出。
    #    （这是本脚本里唯一一处刻意偏离全局 set -e 的地方）
    set +e

    local once=0
    [ "${1:-}" = "--once" ] && once=1

    va_check_vad_path

    va_log "=== 守护启动（巡检间隔 ${CHECK_INTERVAL}s）==="
    va_log "看护：Voice Agent :$VOICE_PORT$( [ "$ENABLE_TUNNEL" = 1 ] && echo " ｜ 隧道 $VPS_IP" )（LLM 走 DeepSeek 云端 API，不本地看护）"

    if [ "$once" = "1" ]; then
        va_check_once
        va_log "单轮巡检完成"
        exit 0
    fi

    while true; do
        va_check_once
        sleep "$CHECK_INTERVAL"
    done
}


# ============================================================================
# 13. 入口
# ============================================================================

va_usage() {
    cat <<EOF
用法: bash run.sh <command>

  （无参数）/ dev   前台启动，Ctrl+C 连子进程一起停（开发/演示）
  start             常驻启动（脱离终端 + 自动拉起 watchdog）
  stop              停止全部（含 watchdog、按进程组清理）
  restart           重启（= stop + start）
  status            查看进程 / 显卡 / 配置 / 日志位置
  clean-gpu         查看显卡占用者；FORCE=1 才真杀
  watchdog [--once] 看护循环；一般由 start 自动拉起
  preflight         只跑预检，不启动服务

示例:
  bash run.sh                                  # 前台跑起来
  bash run.sh start && bash run.sh status      # 常驻 + 查状态
  FORCE=1 bash run.sh clean-gpu                # 清理占卡的残留进程
  bash run.sh preflight                        # 装完依赖后自查

解释器：自动使用 conda 环境 $VA_CONDA_ENV_NAME（找不到才退回 PATH 里的 python），
        无需先 conda activate；也可用 VOICEAGENT_PYTHON=<绝对路径> 直接指定。

LLM 固定走 DeepSeek 云端（config.py §4），不需要本地大模型。
EOF
}

case "${1:-dev}" in
    dev|"")     va_do_dev ;;
    start)      va_do_start ;;
    stop)       va_do_stop ;;
    restart)    va_do_stop; sleep 2; va_do_start ;;
    status)     va_do_status ;;
    clean-gpu)  va_do_clean_gpu ;;
    watchdog)   shift; va_do_watchdog "${1:-}" ;;
    preflight)  va_preflight ;;
    -h|--help|help) va_usage ;;
    *)          va_usage; exit 1 ;;
esac
