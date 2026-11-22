/**
 * 下行消息分派表。
 *
 * 服务端每条 JSON 控制帧都带 `type`，这里按 type 查表处理 —— 新增消息类型只需加一个键，
 * 不必改动连接层的 if/switch 主体。
 *
 * 与 session 的关系：只有 `session_end` 需要主动挂断通话，而挂断动作在 session.js。
 * 为避免「protocol ↔ session 循环 import」，这里留一个**回调注册口**，
 * 由 session.js 在加载时把真正的挂断函数注册进来。
 */

import { TURN_GATE_CLOSED } from './config.js';
import { app } from './state.js';
import {
  appendMessage,
  appendToAssistantBubble,
  el,
  renderLatency,
  setAgentStateBadge,
} from './ui.js';
import { drainPlayback, enqueuePcmFrame, flushPlayback } from './audio.js';

/** 会话结束时要做的事（由 session.js 注入，默认什么都不做）。 */
let onSessionEnd = () => {};

/** 注册"会话结束"回调。 */
export function setSessionEndHandler(fn) {
  onSessionEnd = fn;
}

/**
 * 一条 WebSocket 消息 = 二进制音频 或 JSON 控制帧。
 * 音频先过播放闸门；控制帧按 type 查表分派。
 */
export function handleSocketMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    // 闸门：被打断（playableTurn=0）后，在途的旧轮次音频直接丢弃，不送入播放队列
    if (app.playableTurn !== TURN_GATE_CLOSED) enqueuePcmFrame(event.data);
    return;
  }

  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch (err) {
    console.warn('无法解析的服务端消息:', event.data, err);
    return;
  }

  const handler = MESSAGE_HANDLERS[msg.type];
  if (handler) {
    handler(msg);
  } else {
    console.warn('未知消息类型:', msg.type);
  }
}

export const MESSAGE_HANDLERS = {
  // 握手完成
  ready(msg) {
    appendMessage('system', 'Connected — v' + (msg.version || '3.0'));
  },

  // 会话状态机（idle / listening / thinking / speaking / interrupted）
  state(msg) {
    setAgentStateBadge(msg.state);
  },

  // 识别结果（文本输入通道已移除，source 只可能是缺省）
  asr(msg) {
    if (!msg.text) return;
    appendMessage('user', msg.text, `ASR ${msg.latency_ms}ms`);
    el('turnCount').textContent = msg.turn || '';
  },


  // RAG 命中的上下文
  rag(msg) {
    el('ragBox').textContent = msg.context || '(empty)';
  },

  // 分句流式：第 0 句开新气泡，后续句追加到同一气泡
  llm_sentence(msg) {
    if (msg.sentence_idx === 0) {
      const meta = msg.latency_ms ? `LLM 1st sentence ${msg.latency_ms}ms` : '';
      app.assistantBubble = appendMessage('assistant', msg.text, meta);
    } else {
      appendToAssistantBubble(msg.text);
    }
  },

  // 整轮结束：收尾播放（尾音要放完，但不能永远停在 playing）
  llm() {
    app.assistantBubble = null;
    drainPlayback();
  },

  metrics(msg) {
    renderLatency(msg);
  },

  // 播放授权：随后到达的二进制帧属于第 N 轮
  audio_start(msg) {
    app.playableTurn = msg.turn;
  },

  // 用户打断
  barge_in() {
    appendMessage('system', '💬 barge-in');
    // 关键一步：flush 播放队列，丢掉"已下发到浏览器但还没播出去"的音频（服务端那半场是停止下发）
    flushPlayback();
    // 关闸：被打断轮次的在途帧到达时会被 handleSocketMessage 顶部拦掉，直到下一轮 audio_start(N)
    app.playableTurn = TURN_GATE_CLOSED;
  },

  // 服务端主动文案
  system_message(msg) {
    appendMessage('assistant', msg.text, 'system');
  },

  session_end(msg) {
    appendMessage('system', '📞 Session ended — ' + (msg.reason || 'unknown'));
    onSessionEnd();
  },

  error(msg) {
    appendMessage('system', '⚠ ' + msg.message);
  },

  reset_ack() {
    appendMessage('system', 'Reset');
    el('turnCount').textContent = '0';
    el('ragBox').textContent = 'Waiting...';
  },
};
