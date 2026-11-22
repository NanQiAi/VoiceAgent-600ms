/**
 * WebSocket 连接层：建连、发消息、等就绪。
 *
 * 消息**处理**在 protocol.js（分派表），这里只负责链路本身。
 * 页面只用一条连接（上行音频 + 下行音频/控制帧共用），所以不需要连接池。
 */

import { app } from './state.js';
import { setConnectionStatus } from './ui.js';
import { handleSocketMessage } from './protocol.js';

/** ws:// 还是 wss:// 跟着页面协议走（本项目 demo 版无鉴权，直接连）。 */
function buildVoiceSocketUrl() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${scheme}://${location.host}/ws/voice`;
}

/** 建连（幂等：已连或正在连就复用，避免出现两条音频流）。 */
export function connectSocket() {
  if (app.socket && app.socket.readyState <= WebSocket.OPEN) return;

  const socket = new WebSocket(buildVoiceSocketUrl());
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => setConnectionStatus('connected', 'Connected');
  socket.onclose = () => {
    setConnectionStatus('', 'Disconnected');
    app.socket = null;
  };
  socket.onerror = () => setConnectionStatus('', 'Error');
  socket.onmessage = handleSocketMessage;
  app.socket = socket;
}

/** 等到 socket 进入 OPEN；连接失败/被关则 reject，避免调用方无限等待。 */
export function waitForSocketOpen(socket) {
  return new Promise((resolve, reject) => {
    if (!socket) return reject(new Error('没有可用的 WebSocket'));
    if (socket.readyState === WebSocket.OPEN) return resolve();
    if (socket.readyState >= WebSocket.CLOSING) return reject(new Error('WebSocket 已关闭'));

    const cleanup = () => {
      socket.removeEventListener('open', onOpen);
      socket.removeEventListener('close', onClose);
    };
    const onOpen = () => { cleanup(); resolve(); };
    const onClose = () => { cleanup(); reject(new Error('WebSocket 连接失败')); };
    socket.addEventListener('open', onOpen);
    socket.addEventListener('close', onClose);
  });
}

/** 发一条 JSON 控制帧；未连接时返回 false（调用方据此决定是否提示）。 */
export function sendJson(payload) {
  if (app.socket && app.socket.readyState === WebSocket.OPEN) {
    app.socket.send(JSON.stringify(payload));
    return true;
  }
  return false;
}
