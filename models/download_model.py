from modelscope import snapshot_download

model_id = "OpenBMB/MiniCPM-o-4_5-awq"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/MiniCPM-o-4.5-awq",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "speechbrain/spkrec-ecapa-voxceleb"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/spkrec-ecapa-voxceleb",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "jiangjiangaha/FireRedVAD-Stream-VAD"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/FireRedVAD-stream",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "FireRedTeam/FireRedASR2-AED"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/FireRedASR2-AED",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "iic/SenseVoiceSmall"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/SenseVoiceSmall",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "OpenBMB/MiniCPM4.1-8B-GPTQ"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/MiniCPM4.1-8B-GPTQ",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "OpenBMB/VoxCPM1.5"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/VoxCPM1.5",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "Qwen/Qwen3-14B-AWQ"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/Qwen3-14B-AWQ",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "BAAI/bge-small-zh-v1.5"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/bge-small-zh-v1.5",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "BAAI/bge-m3"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/bge-m3",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "BAAI/bge-reranker-v2-m3"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="./models/bge-reranker-v2-m3",
    # revision="master" #可选，指定版本
)
print(model_dir)
