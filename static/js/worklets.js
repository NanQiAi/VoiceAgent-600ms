/**
 * 两个 AudioWorklet 处理器的源码（跑在**音频线程**，不是主线程）。
 *
 * 为什么存成字符串而不是独立 .js 文件：worklet 在独立的 AudioWorkletGlobalScope 里运行，
 * 不能 import 主线程的模块；把源码放这里可以用模板字符串把采样率等常量**插值进去**，
 * 保证与 config.js 只有一个真相来源（独立文件就得把这些数字抄一遍）。
 * 加载时用 Blob URL 交给 `audioWorklet.addModule()`。
 */

import { CAPTURE_CHUNK_SAMPLES, PLAYER_PREBUFFER_SAMPLES } from './config.js';

/**
 * 抖动缓冲播放器：攒够水位才开播；'flush' 立即清空（打断），'drain' 放完尾音再停（正常收尾）。
 * 跑在音频线程，onmessage 同步执行，所以 flush 在下一个 128 点量子就生效。
 */
export const PLAYER_WORKLET_SOURCE = `
class JitterBufferPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];              // 待播的 Float32 块
    this.bufferedSamples = 0;     // 已入队但还没播的总采样数
    this.current = null;          // 正在播的块
    this.offset = 0;              // current 内的读取位置
    this.phase = 'idle';          // idle | buffering | playing
    this.draining = false;        // 收到 drain 后，队列排空即回 idle
    this.preBufferSamples = ${PLAYER_PREBUFFER_SAMPLES};
    this.port.onmessage = (event) => {
      if (event.data === 'flush') {
        this.queue = []; this.bufferedSamples = 0;
        this.current = null; this.offset = 0;
        this.phase = 'idle'; this.draining = false;
        return;
      }
      if (event.data === 'drain') { this.draining = true; return; }
      const samples = new Float32Array(event.data);
      this.queue.push(samples);
      this.bufferedSamples += samples.length;
      if (this.phase === 'idle') this.phase = 'buffering';
      if (this.phase === 'buffering' && this.bufferedSamples >= this.preBufferSamples) {
        this.phase = 'playing';
      }
    };
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    if (this.phase !== 'playing') { out.fill(0); return true; }
    for (let i = 0; i < out.length; i++) {
      if (!this.current || this.offset >= this.current.length) {
        if (this.queue.length > 0) {
          this.current = this.queue.shift();
          this.bufferedSamples -= this.current.length;
          this.offset = 0;
        } else {
          this.current = null; this.offset = 0;
          if (this.draining) { this.phase = 'idle'; this.draining = false; }
          out.fill(0, i);
          return true;
        }
      }
      out[i] = this.current[this.offset++];
    }
    return true;
  }
}
registerProcessor('queue-player', JitterBufferPlayer);
`;

/**
 * 采集 worklet：把麦克风样本攒成 512 点一帧、转 int16 再发。
 * 512 是硬约定（16000Hz × 32ms），等于服务端 CHUNK_SAMPLES，服务端收到即可直接送 VAD。
 */
export const CAPTURE_WORKLET_SOURCE = `
class PcmSender extends AudioWorkletProcessor {
  constructor() { super(); this.pending = []; }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.pending.push(Math.max(-1, Math.min(1, channel[i])));
    }
    while (this.pending.length >= ${CAPTURE_CHUNK_SAMPLES}) {
      const chunk = this.pending.splice(0, ${CAPTURE_CHUNK_SAMPLES});
      const int16 = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) int16[i] = chunk[i] * 32767;
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-sender', PcmSender);
`;
