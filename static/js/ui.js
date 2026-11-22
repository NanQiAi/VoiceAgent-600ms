/**
 * 界面渲染：DOM 工具与三个展示区域（消息流 / 连接状态 / 延迟面板）。
 *
 * 只做"把数据变成 DOM"，不发起任何网络或音频操作 —— 这样协议层和音频层改动时
 * 不会牵连界面代码。
 */

import { LATENCY_GOOD_MS, LATENCY_OK_MS } from './config.js';
import { app } from './state.js';

/** 按 id 取元素；页面里 id 都是唯一的，省掉 querySelector 的开销与拼字符串。 */
export const el = (id) => document.getElementById(id);

/**
 * 追加一条消息气泡，返回该元素（供流式追加使用）。
 *
 * 用 createTextNode 而不是 innerHTML —— 服务端文本可能含 < > &，直接当 HTML 会破版甚至被注入。
 * 文本节点**始终是 childNodes[0]**（meta 在其后），appendToAssistantBubble 依赖这一点。
 */
export function appendMessage(role, text, meta) {
  const bubble = document.createElement('div');
  bubble.className = 'msg ' + role;
  bubble.appendChild(document.createTextNode(text));
  if (meta) {
    const metaLine = document.createElement('div');
    metaLine.className = 'meta';
    metaLine.textContent = meta;
    bubble.appendChild(metaLine);
  }
  const chat = el('chat');
  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

/** 把下一个 assistant 句子追加到当前气泡（首句之后才有气泡，见 protocol.js 的 llm_sentence）。 */
export function appendToAssistantBubble(text) {
  if (!app.assistantBubble) return;
  app.assistantBubble.childNodes[0].textContent += text;
}

/** 顶栏的圆点 + 文案：''=未连接 / connected / active。 */
export function setConnectionStatus(dotClass, text) {
  el('statusDot').className = 'status-dot ' + dotClass;
  el('statusText').textContent = text;
}

/** 顶栏的状态徽章：idle / listening / thinking / speaking / interrupted。 */
export function setAgentStateBadge(stateName) {
  const badge = el('stateBadge');
  badge.className = 'state-badge state-' + stateName;
  badge.textContent = stateName.toUpperCase();
}

/** 侧栏首响延迟面板：四段占比条 + 四个数值 + 总延迟徽章。 */
export function renderLatency(metrics) {
  const asrMs = metrics.asr_ms || 0;
  const ragMs = metrics.rag_ms || 0;
  const llmMs = metrics.llm_ms || 0;
  const ttsMs = metrics.tts_ttfa_ms || 0;
  const totalMs = asrMs + ragMs + llmMs + ttsMs;
  if (totalMs === 0) return;

  // 各段按占比展示，但最小 5% 保证数字还看得见
  const toPercent = (value) => Math.max(5, (value / totalMs) * 100).toFixed(0) + '%';
  const segments = [
    ['segASR', 'valASR', asrMs, Math.round(asrMs) + 'ms'],
    ['segRAG', 'valRAG', ragMs, ragMs.toFixed(1) + 'ms'],
    ['segLLM', 'valLLM', llmMs, Math.round(llmMs) + 'ms'],
    ['segTTS', 'valTTS', ttsMs, Math.round(ttsMs) + 'ms'],
  ];
  for (const [segId, valId, rawMs, label] of segments) {
    el(segId).style.width = toPercent(rawMs);
    el(segId).textContent = Math.round(rawMs);
    el(valId).textContent = label;
  }

  const firstResponseMs = metrics.first_response_ms || totalMs;
  const badgeClass = firstResponseMs < LATENCY_GOOD_MS ? 'badge-green'
    : firstResponseMs < LATENCY_OK_MS ? 'badge-yellow'
    : 'badge-red';
  // 这里用 innerHTML 是因为要造一个带 class 的 span；数字由本函数计算，不含外部输入
  el('valFR').innerHTML = `<span class="metric-badge ${badgeClass}">${Math.round(firstResponseMs)}ms</span>`;
}

/** 侧栏模型清单（来自 /api/info）。 */
export function renderModelInfo(models) {
  const container = el('modelInfo');
  container.textContent = '';
  for (const [key, value] of Object.entries(models)) {
    const row = document.createElement('div');
    row.className = 'info-row';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = key.toUpperCase();
    const val = document.createElement('span');
    val.className = 'value';
    val.textContent = value.name;
    row.append(label, val);
    container.appendChild(row);
  }
}
