/**
 * 页面入口：拉取模型清单、绑定按钮、初始化连接。
 *
 * 这是唯一被 HTML 直接加载的脚本（`<script type="module" src="/static/js/main.js">`），
 * 其余模块由它（或其依赖）间接引入。
 */

import { el, renderModelInfo } from './ui.js';
import { resetSession, sendTextMessage, toggleCall } from './session.js';

/**
 * 侧栏模型清单。
 * 拉取失败（服务未就绪 / 网络错误）不影响通话，静默忽略。
 */
async function loadModelInfo() {
  try {
    const response = await fetch('/api/info');
    const info = await response.json();
    if (info.models) renderModelInfo(info.models);
  } catch (err) {
    // 网络错误不影响通话，静默即可
  }
}

function bindControls() {
  el('btnCall').addEventListener('click', toggleCall);
  el('btnSend').addEventListener('click', sendTextMessage);
  el('btnReset').addEventListener('click', resetSession);
  el('textInput').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') sendTextMessage();
  });
}

function init() {
  bindControls();
  loadModelInfo();
  // ⚠️ **不要在这里 connectSocket()**：浏览器的 AudioContext 必须在用户手势里
  // 才能 resume，而服务端一建连就播开场白。页面加载即建连的话，问候的音频会在
  // playerReady=false 时到达并被丢弃 —— 表现就是"只看到问候文字、听不到声音"。
  // 建连统一放到 toggleCall()（用户点 Start Call）：那里先 ensurePlayerReady()
  // 再连 WS，顺序见 session.js 顶部注释。
}

// 脚本在 </body> 前，DOM 已就绪；用 load 保证样式/字体也稳定后再初始化
window.addEventListener('load', init);
