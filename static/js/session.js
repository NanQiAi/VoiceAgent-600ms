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

/** 复位会话（清对话历史与状态机，不断连接）。 */
export function resetSession() {
  sendJson({ type: 'reset' });
}

// 服务端说"会话结束"时自动挂断。放在这里而不是 protocol.js，是为了避免两模块循环 import。
setSessionEndHandler(() => {
  if (app.isCalling) toggleCall();
});
