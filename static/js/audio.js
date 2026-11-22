/**
 * 音频链路：下行播放（24kHz）与上行采集（16kHz）。
 *
 * 两条链路都基于 AudioWorklet，处理器源码在 worklets.js。这里只负责：
 * 建 AudioContext、加载 worklet、把 PCM 在 int16 / float32 之间转换。
 */

import { CAPTURE_CHUNK_SAMPLES, SAMPLE_RATE_IN, SAMPLE_RATE_OUT } from './config.js';
import { app } from './state.js';
import { el } from './ui.js';
import { CAPTURE_WORKLET_SOURCE, PLAYER_WORKLET_SOURCE } from './worklets.js';

/** 把 worklet 源码包成 Blob URL（AudioWorklet 只能按 URL 加载）。 */
function workletModuleUrl(source) {
  return URL.createObjectURL(new Blob([source], { type: 'application/javascript' }));
}

// ───────────────────────────── 下行：播放 ─────────────────────────────

/**
 * 惰性初始化播放链路。必须在用户手势的调用栈里 await ——
 * 否则 AudioContext 停在 suspended，表现为"能连上但没声音"。
 */
export async function ensurePlayerReady() {
  if (app.playerReady) return;
  app.playerContext = new AudioContext({ sampleRate: SAMPLE_RATE_OUT });
  await app.playerContext.audioWorklet.addModule(workletModuleUrl(PLAYER_WORKLET_SOURCE));
  app.playerNode = new AudioWorkletNode(app.playerContext, 'queue-player');
  app.playerNode.connect(app.playerContext.destination);
  await app.playerContext.resume();
  app.playerReady = true;
}

/** int16 → float：除以 32768（不是 32767）让 -32768 精确映射到 -1.0。两端都是小端，无需换字节序。 */
export function enqueuePcmFrame(arrayBuffer) {
  // 播放链路还没建好就收到音频（典型原因：没在用户手势里 ensurePlayerReady，
  // 或服务端在建连后立刻播开场白）。这种丢帧是静默的，只发生一次提示便于排查。
  if (!app.playerReady) {
    if (!app.warnedPlayerNotReady) {
      app.warnedPlayerNotReady = true;
      console.warn('[audio] dropped PCM frame: player not ready (需要用户手势解锁播放)');
    }
    return;
  }
  const int16 = new Int16Array(arrayBuffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  app.playerNode.port.postMessage(float32.buffer, [float32.buffer]);
}

/** 打断：已在音响里播出的收不回，但"已下发未播"的音频会瞬时消失。 */
export function flushPlayback() {
  if (app.playerNode) app.playerNode.port.postMessage('flush');
}

/** 正常收尾：只置 draining 标记、不清队列，因此不会截断尾音。 */
export function drainPlayback() {
  if (app.playerNode) app.playerNode.port.postMessage('drain');
}

// ───────────────────────────── 上行：采集 ─────────────────────────────

/** 开麦 + 建采集 worklet，把每帧通过 WebSocket 发出去。 */
export async function startCapture() {
  // 回声消除是双工打断的前提：否则 TTS 会被麦克风收回去，服务端把 AI 自己的声音判成用户插话。
  // 显式 16000Hz：硬件不支持时浏览器会在此重采样，保证 worklet 拿到的就是 16kHz 单声道。
  app.mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: SAMPLE_RATE_IN,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  app.captureContext = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
  await app.captureContext.audioWorklet.addModule(workletModuleUrl(CAPTURE_WORKLET_SOURCE));

  const source = app.captureContext.createMediaStreamSource(app.mediaStream);
  app.captureNode = new AudioWorkletNode(app.captureContext, 'pcm-sender');
  app.captureNode.port.onmessage = (event) => {
    // 音频有实时性：只在连接打开时发，宁可丢帧也不缓存补发（补发会累积延迟）
    if (app.socket && app.socket.readyState === WebSocket.OPEN) {
      app.socket.send(event.data);
    }
    updateMicLevel(event.data);
  };

  // 刻意不接 destination，避免把自己的声音放出来形成啸叫
  source.connect(app.captureNode);
}

/** 用本帧 RMS 刷新音量条（300 是经验放大系数，让正常说话就接近满格）。 */
export function updateMicLevel(pcmBuffer) {
  const samples = new Int16Array(pcmBuffer);
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += Math.abs(samples[i]);
  const rms = sum / samples.length / 32768;
  el('vuBar').style.width = Math.min(100, rms * 300) + '%';
}

/** 停麦并拆掉采集链路（播放链路保留，文本输入仍可用）。 */
export function stopCapture() {
  if (app.mediaStream) {
    app.mediaStream.getTracks().forEach((track) => track.stop());
    app.mediaStream = null;
  }
  if (app.captureNode) {
    app.captureNode.disconnect();
    app.captureNode = null;
  }
  if (app.captureContext) {
    app.captureContext.close();
    app.captureContext = null;
  }
  el('vuBar').style.width = '0%';
}
