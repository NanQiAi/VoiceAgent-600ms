"""engine 包：语音链路的两层后端 —— 引擎库 + 对话状态机。

配套的顶层文件与包：

    config.py          所有参数与话术（唯一配置来源）
    ws_server.py       进程入口
    server/            服务层（HTTP / WebSocket）——在 engine 之外，边界一眼可见
    engine/            本包：引擎库 + 状态机
    check_deps.py      依赖自检

包内分两层，依赖方向 **自上而下**，下层绝不 import 上层：

    engine/
    ├── asr.py  llm.py  rag.py  tts.py  vad.py  filler.py   ① 可替换引擎
    └── dialog/                                             ② 对话状态机

① 可替换引擎
    asr.py      SenseVoice ASR（FunASR）
    llm.py      DeepSeek / 本地 vLLM 客户端 + 流式切句 + 历史管理
    rag.py      bge 向量化 + FAISS 检索 + 置信度阈值
    tts.py      Kokoro TTS（24kHz）
    vad.py      Silero VAD
    filler.py   语气词预合成
    六个模块**互不 import**，只通过统一的方法签名通信；换实现只需提供同名方法。

② dialog/ —— 对话状态机
    manager.py 门面 `ConversationManager`（音频入口 / 状态分支 / 打断 / 生命周期），
    用 Mixin 组合 9 个职责模块，共享同一份会话状态。详见 `engine.dialog` 的包说明。

本文件刻意不写代码：`import engine` 不会连带拉起 torch / fastapi 等重依赖。
"""
