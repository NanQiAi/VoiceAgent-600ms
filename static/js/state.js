/**
 * 跨模块共享的运行时状态。
 *
 * 整个页面只有一个会话，所以集中成一个对象而不是散落的全局变量：
 * 便于一眼看清"页面到底持有哪些句柄"，也便于排查泄漏（断开时哪些没清）。
 * 各字段的归属用注释分组，谁负责清理一目了然。
 */

import { TURN_GATE_OPEN_BEFORE_FIRST } from './config.js';

export const app = {
  // ── 连接 ──
  socket: null,              // WebSocket
  isCalling: false,          // 是否在通话中（决定按钮语义）
  playableTurn: TURN_GATE_OPEN_BEFORE_FIRST,  // 播放闸门，见 config.js 注释

  // ── 播放侧（下行 24kHz）──
  playerContext: null,
  playerNode: null,
  playerReady: false,
  warnedPlayerNotReady: false,  // 只在控制台提示一次"未解锁播放就收到音频"

  // ── 采集侧（上行 16kHz）──
  mediaStream: null,
  captureContext: null,
  captureNode: null,

  // ── 聊天区 ──
  assistantBubble: null,     // 当前这轮的助手气泡（流式追加句子用）
};
