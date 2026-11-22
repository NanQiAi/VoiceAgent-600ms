"""全双工语音通道 `/ws/voice`：一条 WebSocket 连接 = 一个会话。

上行：二进制帧 = 16kHz 单声道 int16 裸 PCM（512 采样/帧）；
      文本帧 = JSON 控制消息 `text_input` / `manual_barge_in` / `reset`。
下行：二进制帧 = 24kHz 单声道 int16 裸 PCM；
      文本帧 = `ready` / `state` / `asr` / `rag` / `filler` / `llm_sentence` / `llm` / `audio_start` /
              `metrics` / `system_message` / `barge_in` / `session_end` / `error` /
              `reset_ack`。
契约：`turn` 是轮次序列号（`audio_start` 让前端丢弃旧轮次的迟到音频）；
      `asr` 的 `latency_ms=0` 表示复用了投机 ASR 或聚合文本；文本帧一律 `ensure_ascii=False`。
"""

import json
import uuid
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketDisconnected

import numpy as np

from config import (
    ERROR_APOLOGY_TEXT,
    VERSION,
    ASR_EMPTY_RETRY_TEXT,
    ASR_NOISY_SUGGEST_TEXT,
    FAREWELL_KEYWORDS,
    GREETING_TEXT,
    IDLE_GOODBYE_S,
    IDLE_GOODBYE_TEXT,
    IDLE_PROMPT_TEXT,
    IDLE_TIMEOUT_S,
    MAX_CONSECUTIVE_EMPTY_ASR,
    VAD_MODEL_DIR,
)

from engine.dialog.manager import ConversationManager
from engine.vad import SileroVAD

from .bootstrap import log
from .runtime import runtime

router = APIRouter()



class DownlinkSender:
    """ConversationManager 的唯一下行出口：把一条消息翻译成 WebSocket 帧发出。

    含 `_audio` 键 → 取 `data["_audio"]`（裸 PCM int16 bytes）作为二进制帧发送（该键不会出现在
    网络上）；否则整条 dict 序列化成 JSON 文本帧（`ensure_ascii=False`，中文不被转义）。

    发送失败只记一次 warning 并置 `dead`：异常被有意吞掉，下行失败不该中断状态机，
    但要把连接标记为死，让主循环尽快退出而不是反复抛异常刷日志。
    """

    def __init__(self, ws: WebSocket, session_log: logging.LoggerAdapter):
        self._ws = ws
        self._log = session_log
        self.dead = False

    async def __call__(self, data: dict):
        if self.dead:
            return
        try:
            # 二进制音频帧优先：裸 PCM 不走 JSON（避免 base64 膨胀与编解码开销）。
            if "_audio" in data:
                await self._ws.send_bytes(data["_audio"])
            else:
                await self._ws.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            if not self.dead:
                self.dead = True
                self._log.warning("WebSocket send failed (marking dead): %s", e)


def build_experience_config() -> dict:
    """把 config.py 的 D1~D4 话术与阈值打包，让状态机不直接依赖 config（便于单测注入）。"""
    return {
        "greeting_text": GREETING_TEXT,
        # D2 空闲：IDLE_TIMEOUT_S 未说话提醒一次，IDLE_GOODBYE_S 仍未说话则告别并结束会话（秒）。
        "idle_timeout_s": IDLE_TIMEOUT_S,
        "idle_goodbye_s": IDLE_GOODBYE_S,
        "idle_prompt_text": IDLE_PROMPT_TEXT,
        "idle_goodbye_text": IDLE_GOODBYE_TEXT,
        # D3 错误恢复：pipeline 崩掉时的兜底道歉
        "error_apology_text": ERROR_APOLOGY_TEXT,
        "asr_empty_retry_text": ASR_EMPTY_RETRY_TEXT,
        "asr_noisy_suggest_text": ASR_NOISY_SUGGEST_TEXT,
        "max_consecutive_empty_asr": MAX_CONSECUTIVE_EMPTY_ASR,
        "farewell_keywords": FAREWELL_KEYWORDS,
    }


async def handle_control_message(msg: dict, cm: ConversationManager,
                                 send: DownlinkSender, session_log: logging.LoggerAdapter):
    """处理一条 JSON 控制帧；未知 type 静默忽略（向下兼容前端先行升级）。"""
    msg_type = msg.get("type")

    if msg_type == "text_input":
        text = msg.get("text")
        # 缺字段 / 空串：直接忽略（前端半成品帧不该拆掉会话）
        if not isinstance(text, str) or not text.strip():
            session_log.warning("text_input without usable text, ignored: %r", msg)
            return
        await cm.handle_text_input(text)

    elif msg_type == "manual_barge_in":
        await cm.handle_barge_in()

    elif msg_type == "reset":
        cm.reset()
        await send({"type": "reset_ack"})


@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    """全双工语音主通道：accept → 建 VAD/LLM → 装配状态机 → ready → 收发循环。

    无鉴权（本项目定位为 demo，直接连上就是会话）；收尾统一走 `finally`：
    汇总指标、注销会话、`cm.shutdown()` 取消其全部后台任务。
    """
    await ws.accept()
    # ⚠️ 不能用毫秒时间戳：同一毫秒内建两个连接会撞 ID，后一个覆盖前一个的会话表项，
    # 先断开的那个还会把对方从表里 pop 掉（/api/metrics 就少了一个活着的会话）。
    # 用 uuid4 前 12 位：足够唯一，又不至于把日志撑得太长。
    session_id = f"s-{uuid.uuid4().hex[:12]}"
    session_log = logging.LoggerAdapter(log, {"session": session_id})
    session_log.info("Connected")

    # 每条连接新建 VAD，不复用 runtime.engine["vad_template"]：VAD 内部保存流式状态
    # （_speech_active / _silence_count / 语音环形缓冲），共享会互相污染；重新 load() 成本很低。
    vad = SileroVAD(VAD_MODEL_DIR, threshold=0.5)
    vad.load()

    # 每条连接独立一个 LLM 客户端：LLMChat 持有 _history，共享会让两个用户的上下文串台。
    llm = runtime.engine["llm_factory"]()

    send = DownlinkSender(ws, session_log)

    # 装配状态机：asr/rag/tts/filler 用全局共享实例，vad/llm 用本会话独有实例。
    cm = ConversationManager(
        asr=runtime.engine["asr"], llm=llm, tts=runtime.engine["tts"],
        rag=runtime.engine["rag"], vad=vad, filler=runtime.engine["filler"],
        send_fn=send,
        spec_asr=runtime.engine.get("spec_asr"),
        experience_config=build_experience_config(),
    )

    runtime.register_session(session_id, cm)

    # mode 字段保留在协议里（前端目前不读），全双工是唯一模式。
    await send({"type": "ready", "session_id": session_id, "version": VERSION,
                "mode": "duplex"})

    # D1 问候语用 _track_task 起后台任务而不是 await：收发循环立刻开始，
    # 问候播放期间用户就能说话并触发打断。
    if GREETING_TEXT:
        cm._track_task(cm.start_greeting(), name="greeting")

    try:
        while True:
            # 对端已死（TCP 未及时报错）→ 去走 finally，不再空转收包。
            if send.dead:
                session_log.info("send_fn dead, exiting loop")
                break

            # 阻塞等下一条 ASGI 消息；客户端断开时 Starlette 抛 WebSocketDisconnect（下方捕获）。
            data = await ws.receive()

            if "bytes" in data and data["bytes"]:
                raw = data["bytes"]
                # int16 帧必须是偶数长度；奇数长度 np.frombuffer 会抛 ValueError。
                # 这类畸形帧（例如前端/代理发了个半帧）只丢这一帧，**不能**因此拆掉整条会话。
                if len(raw) % 2 != 0:
                    session_log.warning("Dropped audio frame with odd byte length: %d", len(raw))
                    continue
                # int16 → float32 归一化：除以 32768（2^15）而非 32767，保证最小值 -32768 映射到 -1.0。
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                await cm.feed_audio(samples)

            elif "text" in data and data["text"]:
                # 控制帧解析失败只跳过这一帧：一条畸形 JSON 不该让整个通话中断。
                try:
                    msg = json.loads(data["text"])
                except json.JSONDecodeError as e:
                    session_log.warning("Dropped malformed control frame: %s", e)
                    continue
                await handle_control_message(msg, cm, send, session_log)

    # 客户端主动断开属正常收尾：只记录已进行到第几轮，不报错。
    except WebSocketDisconnect:
        session_log.info("Disconnected (turn %d)", cm._turn)
    except WebSocketDisconnected:
        # 已经收到过 disconnect 后再 receive() 会抛这个（RuntimeError 子类，**不是** WebSocketDisconnect）。
        # 同样是正常收尾，不能落进下面的 except Exception 打 ERROR 堆栈。
        session_log.info("Disconnected (already closed, turn %d)", cm._turn)
    except Exception as e:
        # 兜底：仍有意料之外的异常时记完整堆栈，再由 finally 收尾。
        session_log.error("Error: %s", e, exc_info=True)
    finally:
        # 必须走这里释放资源，否则会泄漏 asyncio 任务（pipeline / TTS 播放 / 投机 ASR / 空闲计时器）。
        metrics = runtime.unregister_session(session_id, cm)
        session_log.info("Session metrics: %s", metrics)
        # 取消该会话所有后台任务并复位状态机（shutdown 内部会再调 reset()）。
        cm.shutdown()
        # **显式关闭**：不能依赖"ASGI app 返回即关闭"的隐式行为 —— uvicorn 与 Starlette
        # 的 TestClient 表现不一致，后者会让对端一直阻塞在 receive()。已断开时会抛
        # RuntimeError（"Cannot call send"），忽略即可。
        try:
            await ws.close()
        except Exception:
            pass
