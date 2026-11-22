from modelscope import snapshot_download

model_id = "AI-ModelScope/Kokoro-82M"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/Kokoro-82M",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "iic/SenseVoiceSmall"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/SenseVoiceSmall",
    # revision="master" #可选，指定版本
)
print(model_dir)


model_id = "BAAI/bge-small-zh-v1.5"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/bge-small-zh-v1.5",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "BAAI/bge-m3"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/bge-m3",
    # revision="master" #可选，指定版本
)
print(model_dir)
