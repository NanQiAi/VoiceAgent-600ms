from modelscope import snapshot_download

model_id = "iic/SenseVoiceSmall"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/SenseVoiceSmall",
    # revision="master" #可选，指定版本
)
print(model_dir)

# model = AutoModel(model=model_dir, vad_model="fsmn-vad", device="cuda:0")
# res = model.generate(
#     input=f"{model_dir}/example/en.mp3",
#     language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
#     use_itn=True,
#     batch_size_s=60,
# )
# print(rich_transcription_postprocess(res[0]["text"]))


model_id = "OpenBMB/MiniCPM4.1-8B-GPTQ"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/MiniCPM4.1-8B-GPTQ",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "OpenBMB/VoxCPM1.5"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/VoxCPM1.5",
    # revision="master" #可选，指定版本
)
print(model_dir)

model_id = "Qwen/Qwen3-14B-AWQ"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/Qwen3-14B-AWQ",
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

model_id = "BAAI/bge-reranker-v2-m3"
# 下载到目标目录，文件直接落在 local_dir
model_dir = snapshot_download(
    model_id,
    local_dir="/home/victor/code/600ms_voice_agent/models/bge-reranker-v2-m3",
    # revision="master" #可选，指定版本
)
print(model_dir)


from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
model = load_silero_vad()
wav = read_audio('/home/victor/code/600ms_voice_agent/models/SenseVoiceSmall/example/en.mp3')
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)

import torch
torch.set_num_threads(1)

# model, utils = torch.hub.load(repo_or_dir='/home/victor/code/600ms_voice_agent/models/snakers4/silero-vad', model='silero_vad')
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, _) = utils

wav = read_audio('/home/victor/code/600ms_voice_agent/models/SenseVoiceSmall/example/en.mp3')
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)