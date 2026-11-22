/**
 * 用户操作：开始/结束通话、文本输入、复位。
 *
 * 这里是唯一把「播放链路 + 连接 + 采集」串起来的地方，所以顺序很关键：
 * 先建播放链路（必须在用户手势里），再等连接就绪，最后开麦。
 */

import { app } from './state.js';
import { appendMessage, el, setConnectionStatus } from './ui.js';
import { ensurePlayerReady, startCapture, stopCapture } from './audio.js';
import { connectSocket, sendJson, waitForSocketOpen } from './network.js';
import { setSessionEndHandler } from './protocol.js';

/**
 * Start / End Call。
 * 开始 = 建播放链路 + 连 WS + 开麦；结束 = 停轨道、拆采集 worklet、关采集 AudioContext。
 */
export async function toggleCall() {
  const button = el('btnCall');

  if (app.isCalling) {
    stopCapture();
    app.isCalling = false;
    button.textContent = 'Start Call';
    button.classList.remove('active');
    setConnectionStatus('connected', 'Connected');
    el('micStatus').textContent = 'Off';
    return;
  }

  await ensurePlayerReady();
  if (!app.socket || app.socket.readyState !== WebSocket.OPEN) connectSocket();
  try {
    await waitForSocketOpen(app.socket);
  } catch (err) {
    appendMessage('system', 'Connection failed: ' + err.message);
    return;
  }

  try {
    await startCapture();
  } catch (err) {
    appendMessage('system', 'Mic denied: ' + err.message);
    return;
  }

  app.isCalling = true;
  button.textContent = 'End Call';
  button.classList.add('active');
  setConnectionStatus('active', 'In Call');
  el('micStatus').textContent = 'On';
}

/** 文本输入跳过 ASR 直接进 Pipeline（服务端 _turn++）；顺便借这次用户手势解锁播放链路。 */
export async function sendTextMessage() {
  const input = el('textInput');
  const text = input.value.trim();
  if (!text) return;

  // 用户可能没点 Start Call 就直接打字：这次点击同样是手势，
  // 先解锁播放链路再建连（服务端会在建连后播开场白），否则消息发不出去。
  await ensurePlayerReady();
  if (!app.socket || app.socket.readyState !== WebSocket.OPEN) connectSocket();
  try {
    await waitForSocketOpen(app.socket);
  } catch (err) {
    appendMessage('system', 'Connection failed: ' + err.message);
    return;
  }

  if (!sendJson({ type: 'text_input', text })) return;
  input.value = '';
}

/** 复位会话（清对话历史与状态机，不断连接）。 */
export function resetSession() {
  sendJson({ type: 'reset' });
}

// 服务端说"会话结束"时自动挂断。放在这里而不是 protocol.js，是为了避免两模块循环 import。
setSessionEndHandler(() => {
  if (app.isCalling) toggleCall();
});
