"""Local webpage that shows live transcripts. Listens on 127.0.0.1 only."""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from sysaudio_stt.cli import (
    ENGINE_CHOICES,
    LANGUAGE_CHOICES,
    MODEL_CHOICES,
    build_parser,
    configure_runtime,
    require_binary,
    resolve_language,
    run_transcribe,
    scan_outputs,
)

PAGE = r'''<!DOCTYPE html>
<html lang='zh-CN'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>VoiceCat</title>
<style>
html, body { height: 100%; }
:root {
  --base-100: #ffffff;
  --base-200: #fbfafd;
  --base-300: #dcdcde;
  --content: #171321;
  --muted: #737278;
  --faint: #89888d;
  --primary: #1f75cb;
  --primary-content: #ffffff;
  --active: #e9f3fc;
  --error: #dd2b0e;
  --error-content: #ffffff;
  --field: 4px;
  --box: 4px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --base-100: #1f1e24;
    --base-200: #18171d;
    --base-300: #45424d;
    --content: #ececef;
    --muted: #bfbfc3;
    --faint: #a4a3a8;
    --primary: #428fdc;
    --primary-content: #ffffff;
    --active: #2e2d33;
    --error: #f26147;
    --error-content: #1f1e24;
  }
}
html, body { height: 100%; }
body { margin: 0; font-family: "PingFang SC", "HarmonyOS Sans SC", "Microsoft YaHei", sans-serif; font-size: 14px; background: #f5f6f8; color: #1f2329; display: flex; flex-direction: row; user-select: none; }
aside { width: 216px; flex: none; display: flex; flex-direction: column; background: #ffffff; border-right: 1px solid #e5e6eb; transition: width .16s ease; }
body.sider-collapsed aside { width: 64px; }
.shell { flex: 1; min-width: 0; min-height: 0; display: flex; flex-direction: column; }
.sider-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; padding: 16px 12px 8px; }
.brand { min-width: 0; }
nav, .sider-foot { display: flex; flex-direction: column; gap: 2px; padding: 8px; }
.sider-foot { margin-top: auto; border-top: 1px solid #eef0f3; }
nav button, .sider-foot button, #sider-toggle { width: 100%; display: flex; align-items: center; gap: 8px; text-align: left; border-color: transparent; background: transparent; color: #1f2329; border-radius: 8px; }
nav button.active, #nav-settings.active { background: #e8f3ff; color: #1456d9; }
.mark { width: 22px; flex: none; text-align: center; color: #646a73; font-size: 12px; }
body.sider-collapsed #sider-toggle { justify-content: center; }
#sider-toggle { width: 28px; height: 28px; padding: 0; justify-content: center; color: #646a73; }
header { min-height: 48px; padding: 0 16px; display: flex; align-items: center; background: #ffffff; border-bottom: 1px solid #e5e6eb; }
#sub-sider { display: none; width: 196px; flex: none; flex-direction: column; background: #f7f8fa; border-right: 1px solid #e5e6eb; }
body.settings-open #sub-sider { display: flex; }
.sub-title { padding: 16px 16px 8px; color: #646a73; font-size: 12px; }
#sub-sider nav { padding: 4px 8px; }
nav svg, .sider-foot svg, #sider-toggle svg { width: 18px; height: 18px; flex: none; }
body.sider-collapsed .label, body.sider-collapsed .tagline, body.sider-collapsed h1, body.sider-collapsed .sub-title { display: none; }
body.sider-collapsed #sub-sider { width: 64px; }
body.sider-collapsed #sider-toggle svg { transform: rotate(180deg); }
body.sider-collapsed nav button, body.sider-collapsed .sider-foot button { justify-content: center; }
h1 { margin: 0; font-size: 16px; font-weight: 650; }
.tagline { color: var(--faint); font-size: 12px; }
#status { margin-left: auto; max-width: 360px; color: #9aa3b2; font-size: 12px; line-height: 1.45; text-align: right; }
#live-note { min-height: 18px; margin: 0; color: #9aa3b2; font-size: 12px; }
main { flex: 1; min-height: 0; display: flex; flex-direction: column; padding: 16px; background: #f5f6f8; }
.page { display: none; flex: 1; min-height: 0; flex-direction: column; }
.page.active { display: flex; }
.page:not(#page-live) { overflow: auto; }
.settings { display: flex; flex-direction: column; gap: 16px; width: min(860px, 100%); margin: 0; padding: 0; border: 0; background: transparent; box-sizing: border-box; }
.settings h2 { margin: 0; font-size: 18px; font-weight: 650; }
.settings label { display: flex; flex-direction: column; gap: 4px; color: var(--muted); font-size: 12px; }
#scan-log { min-height: 180px; overflow: auto; padding: 8px; border: 1px solid var(--base-300); border-radius: var(--box); background: var(--base-100); white-space: pre-wrap; color: var(--content); user-select: text; }
#download-list { display: flex; flex-direction: column; gap: 8px; }
#download-list h2 { margin: 12px 0 0; font-size: 13px; color: var(--muted); }
.download-row { display: grid; grid-template-columns: 88px 1fr; gap: 12px; align-items: center; padding: 10px 12px; border: 1px solid var(--base-300); border-radius: 12px; background: var(--base-200); }
.download-row span { color: var(--content); font-size: 13px; line-height: 1.45; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 8px; }
select, button, textarea, input[type='text'], input[type='password'] { font: inherit; font-size: 14px; border-radius: 8px; border: 1px solid #dee0e3; background: #ffffff; color: #1f2329; padding: 8px 12px; }
.field { display: block; box-sizing: border-box; width: 100%; border: 1px solid #dee0e3; border-radius: 8px; background: #ffffff; padding: 10px 14px; }
.field:focus-within { border-color: #1456d9; }
.field select, .field textarea, .field input { -webkit-appearance: none; appearance: none; box-sizing: border-box; width: 100%; height: 22px; margin: 0; padding: 0; border: 0; border-radius: 0; outline: none; background: transparent; box-shadow: none; text-indent: 0; line-height: 22px; color: #1f2329; user-select: text; }
.field textarea { display: block; height: auto; min-height: 168px; resize: vertical; line-height: 1.6; overflow: auto; }
.field select { height: 22px; padding-right: 22px; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1.5 6 6.5 11 1.5' fill='none' stroke='%23646a73' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right center; }
button { cursor: pointer; padding: 6px 12px; background: var(--base-100); }
button:hover { background: var(--base-300); }
button:active, button.pressed { background: var(--base-300); }
button.primary { background: var(--primary); color: var(--primary-content); border-color: var(--primary); }
button.primary:hover, button.primary:active, button.primary.pressed { filter: brightness(0.96); }
button.primary.listening { background: var(--primary); color: var(--primary-content); border-color: var(--primary); }
button#stop.listening { background: var(--error); color: var(--error-content); border-color: var(--error); }
button:disabled { opacity: 0.55; cursor: default; }
.tabs button.active { background: var(--primary); color: var(--primary-content); border-color: var(--primary); }
#wave { width: 100%; height: 36px; display: block; background: #ffffff; border: 1px solid #e5e6eb; border-radius: 8px; margin: 0; }
.tabs { display: flex; gap: 6px; margin: 0 0 8px; }
.tabs button { background: transparent; border-color: transparent; color: var(--muted); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

.split { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; margin: 0 0 8px; }
.split input { width: 160px; }
#panes { flex: 1; min-height: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.pane { border: 1px solid #e5e6eb; border-radius: 10px; background: #ffffff; padding: 0; min-height: 0; height: 100%; overflow: auto; }
.pane h2 { position: sticky; top: 0; margin: 0; padding: 12px 16px; font-size: 12px; font-weight: 650; color: #646a73; background: #ffffff; border-bottom: 1px solid #eef0f3; }
#raw, #polished { padding: 8px 16px 20px; white-space: pre-wrap; font-size: 16px; line-height: 1.7; user-select: text; color: #1f2329; }
.line { display: grid; grid-template-columns: 72px 1fr; gap: 12px; margin: 0 0 14px; }
#correct-box { display: none; position: fixed; z-index: 5; width: 280px; padding: 10px; border: 1px solid var(--base-300); border-radius: var(--box); background: var(--base-100); box-shadow: 0 8px 24px rgba(0,0,0,.12); }
#correct-box.show { display: flex; flex-direction: column; gap: 6px; }
#correct-box input { width: 100%; box-sizing: border-box; }
.draft, .polished { color: #1f2329; }
.time { color: #7f8b9c; font-size: 12px; }
#debug-box { display: none; height: 150px; overflow: auto; margin-top: 8px; padding: 8px; border: 1px solid var(--base-300); border-radius: var(--box); background: var(--base-100); color: var(--muted); font: 12px/1.45 ui-monospace, Menlo, Consolas, monospace; white-space: pre-wrap; user-select: text; }
body.debug #debug-box { display: block; }
@media (max-width: 860px) {
  header { flex-wrap: wrap; height: auto; padding: 8px 12px; }
  #status { margin-left: 0; text-align: left; max-width: none; }
  #panes { grid-template-columns: 1fr; }
}
::-webkit-scrollbar { width: 8px; height: 8px; background: transparent; }
::-webkit-scrollbar-thumb { background: color-mix(in srgb, var(--content) 22%, transparent); border-radius: 6px; border: 2px solid transparent; background-clip: content-box; }
</style>
</head>
<body>
<aside id='sider'>
<div class='sider-top'>
<div class='brand'><h1>VoiceCat</h1><span class='tagline'>本机实时转写</span></div>
<button id='sider-toggle' type='button' title='收起侧栏'><svg viewBox='0 0 20 20' aria-hidden='true'><path d='M12.5 4.5 7 10l5.5 5.5' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg></button>
</div>
<nav>
<button id='nav-live' class='active' type='button' data-page='live'><svg viewBox='0 0 20 20' aria-hidden='true'><path d='M4 8.5v3M8 5.5v9M12 3.5v13M16 7.5v5' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg><span class='label'>识别</span></button>
</nav>
<div class='sider-foot'>
<button id='nav-settings' type='button' title='设置'><svg viewBox='0 0 20 20' aria-hidden='true'><path d='M8.1 2.6h3.8l.45 1.8c.5.2 1 .48 1.42.82l1.72-.72 1.9 3.28-1.27 1.27c.08.47.08.96 0 1.42l1.27 1.27-1.9 3.28-1.72-.72c-.42.34-.92.62-1.42.82l-.45 1.8H8.1l-.45-1.8a5.2 5.2 0 0 1-1.42-.82l-1.72.72-1.9-3.28 1.27-1.27a5.4 5.4 0 0 1 0-1.42L2.61 7.78l1.9-3.28 1.72.72c.42-.34.92-.62 1.42-.82l.45-1.8z' fill='none' stroke='currentColor' stroke-width='1.4' stroke-linejoin='round'/><circle cx='10' cy='10' r='2.05' fill='none' stroke='currentColor' stroke-width='1.4'/></svg><span class='label'>设置</span></button>
</div>
</aside>
<aside id='sub-sider'>
<div class='sub-title'>设置</div>
<nav>
<button id='nav-download' type='button' data-page='download'><svg viewBox='0 0 20 20' aria-hidden='true'><path d='M10 3.5v8M6.5 8.5 10 12l3.5-3.5M4.5 15.5h11' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/></svg><span class='label'>下载</span></button>
<button id='nav-audio' type='button' data-page='audio'><svg viewBox='0 0 20 20' aria-hidden='true'><path d='M4 8.5h2.2L10 5v10l-3.8-3.5H4zM12.2 8.2a3 3 0 0 1 0 3.6M14 6.2a5.5 5.5 0 0 1 0 7.6' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linejoin='round'/></svg><span class='label'>音频输入</span></button>
<button id='nav-model' type='button' data-page='model'><svg viewBox='0 0 20 20' aria-hidden='true'><path d='M10 3.5 16 7v6l-6 3.5L4 13V7z' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linejoin='round'/></svg><span class='label'>模型</span></button>
<button id='nav-prompt' type='button' data-page='prompt'><svg viewBox='0 0 20 20' aria-hidden='true'><path d='M5 5.5h10M5 10h10M5 14.5h6' fill='none' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg><span class='label'>提示词</span></button>
</nav>
</aside>
<div class='shell'>
<header><div id='status'>还没开始。识别结果会显示在下面。</div></header>
<main>
<section id='page-live' class='page active'>
<div class='controls'>
<button id='start' class='primary' type='button'>开始</button>
<button id='stop' type='button'>停止</button>
<button id='clear' type='button'>清空画面</button>
<label><input id='debug' type='checkbox'> debug</label>
</div>
<div id='live-note'></div>
<canvas id='wave' width='900' height='42'></canvas>
<label class='split'>左右比例 <input id='split' type='range' min='25' max='75' value='50'><span id='meter'></span></label>
<div id='panes'>
<section class='pane'><h2>原文</h2><div id='raw'></div></section>
<section class='pane'><h2>润色</h2><div id='polished'></div></section>
</div>
<pre id='debug-box'></pre>
<div id='correct-box'>
<div>把选中的文字改成正确写法，会记到专有名词里。</div>
<div id='correct-from'></div>
<input id='correct-to' type='text' placeholder='正确写法'>
<div class='controls'><button id='correct-save' type='button'>保存</button><button id='correct-cancel' type='button'>取消</button></div>
</div>
</section>
<section id='page-audio' class='page'>
<div class='settings'>
<h2>音频输入</h2>
<div class='controls'>
<button id='scan' type='button'>测量输出</button>
<span class='field'><select id='device'></select></span>
</div>
<pre id='scan-log'></pre>
</div>
</section>
<section id='page-model' class='page'>
<div class='settings'>
<h2>模型</h2>
<label>识别引擎<span class='field'><select id='engine'></select></span></label>
<label>识别模型<span class='field'><select id='model'></select></span></label>
<label>语言<span class='field'><select id='language'></select></span></label>
<label><span><input id='polish' type='checkbox' checked> 大模型润色</span></label>
<label>润色模型<span class='field'><select id='polish-model'></select></span></label>
<h2>大模型配置</h2>
<label>接口地址<span class='field'><input id='llm-base-url' type='text'></span></label>
<label>接口类型<span class='field'><input id='llm-wire-api' type='text' value='responses'></span></label>
<label>默认模型<span class='field'><input id='llm-model' type='text'></span></label>
<label>API Key<span class='field'><input id='llm-api-key' type='password' autocomplete='off' placeholder='已保存则留空，不会回显'></span></label>
<button id='save-llm' type='button'>保存大模型配置</button>
<div id='llm-note'></div>
</div>
</section>
<section id='page-download' class='page'>
<div class='settings'>
<h2>模型下载</h2>
<div id='download-note'>模型保存在 ~/Library/Application Support/VoiceCat/models/。下完后回到模型页就能选，也可以在这里删除。</div>
<div class='controls'><button id='open-models' type='button'>在目录中打开</button></div>
<div id='download-list'></div>
</div>
</section>
<section id='page-prompt' class='page'>
<div class='settings'>
<h2>提示词</h2>
<div class='tabs'>
<button id='tab-prompt' class='active' type='button'>润色提示词</button>
<button id='tab-glossary' type='button'>专有名词</button>
</div>
<div id='panel-prompt' class='tab-panel active'><div class='field'><textarea id='polish-prompt' rows='8'></textarea></div></div>
<div id='panel-glossary' class='tab-panel'><div class='field'><textarea id='glossary' rows='8' placeholder='专有名词，用空格或顿号分开，例如 换峰'></textarea></div></div>
</div>
</section>
</main>
</div>
<script>
var since = 0;
var wave = [0];
var bars = [];
var latest = 0;
var promptReady = false;
var glossaryReady = false;
var raw = document.getElementById('raw');
var polished = document.getElementById('polished');
var status = document.getElementById('status');
var canvas = document.getElementById('wave');
function setStatus(text) { status.textContent = text; }
function makeLine(item, kind) {
  var row = document.createElement('div');
  row.className = 'line ' + kind;
  if (item.batch) row.dataset.batch = String(item.batch);
  var time = document.createElement('span');
  time.className = 'time';
  time.textContent = item.ts || '';
  row.appendChild(time);
  row.appendChild(document.createTextNode(item.text || ''));
  return row;
}
var correctFrom = '';
function hideCorrect() {
  document.getElementById('correct-box').classList.remove('show');
}
function replaceVisible(wrong, right) {
  [raw, polished].forEach(function (root) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function (node) {
      if (node.parentNode && node.parentNode.className === 'time') return;
      if (node.nodeValue && node.nodeValue.indexOf(wrong) >= 0) node.nodeValue = node.nodeValue.split(wrong).join(right);
    });
  });
}
document.addEventListener('mouseup', function (event) {
  var box = document.getElementById('correct-box');
  if (box.contains(event.target)) return;
  var selection = window.getSelection();
  var text = selection ? selection.toString().trim() : '';
  if (!text || text.length > 40 || !selection.rangeCount) {
    hideCorrect();
    return;
  }
  var node = selection.anchorNode;
  var inside = node && (raw.contains(node) || polished.contains(node));
  if (!inside) {
    hideCorrect();
    return;
  }
  correctFrom = text;
  document.getElementById('correct-from').textContent = '识别成：' + text;
  document.getElementById('correct-to').value = '';
  var rect = selection.getRangeAt(0).getBoundingClientRect();
  box.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - 300)) + 'px';
  box.style.top = Math.max(8, rect.bottom + 8) + 'px';
  box.classList.add('show');
  document.getElementById('correct-to').focus();
});
document.getElementById('correct-cancel').onclick = hideCorrect;
document.getElementById('correct-save').onclick = function () {
  var right = document.getElementById('correct-to').value.trim();
  if (!correctFrom || !right) {
    setStatus('请填写正确写法');
    return;
  }
  fetch('/api/correct', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ wrong: correctFrom, right: right })
  }).then(function (res) { return res.json(); }).then(function (data) {
    if (data.error) {
      setStatus(data.error);
      return;
    }
    replaceVisible(correctFrom, right);
    var glossary = document.getElementById('glossary');
    if (glossary && data.glossary) glossary.value = data.glossary;
    hideCorrect();
    setStatus('已记住「' + right + '」，这次和下次都会替换。');
  }).catch(function (err) { setStatus(String(err)); });
};
function addText(item) {
  var id = item.draft && item.draft !== true ? String(item.draft) : '';
  if (!id) {
    raw.appendChild(makeLine(item, 'draft'));
  } else {
    var nodes = raw.querySelectorAll('[data-line="' + id + '"]');
    if (nodes.length && nodes[0].dataset.frozen && item.partial) return;
    var row = makeLine(item, 'draft');
    row.dataset.line = id;
    if (!item.partial) row.dataset.frozen = '1';
    if (!nodes.length) raw.appendChild(row);
    else nodes[0].replaceWith(row);
  }
  raw.parentNode.scrollTop = raw.parentNode.scrollHeight;
}
function replaceBatch(item) {
  var nodes = polished.querySelectorAll('[data-batch="' + item.batch + '"]');
  var row = makeLine(item, 'polished');
  if (!nodes.length) polished.appendChild(row);
  else nodes[0].replaceWith(row);
  polished.parentNode.scrollTop = polished.parentNode.scrollHeight;
}
function applySplit() {
  var value = Number(document.getElementById('split').value);
  document.getElementById('panes').style.gridTemplateColumns = value + 'fr ' + (100 - value) + 'fr';
}
function drawWave() {
  var width = canvas.clientWidth || 900;
  var height = 42;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  var count = Math.max(24, Math.floor(width / 7));
  if (!bars.length) {
    for (var n = 0; n < count; n++) bars.push(0);
  }
  bars.push(latest);
  if (bars.length > count) bars = bars.slice(bars.length - count);
  var loud = 200;
  for (var i = 0; i < bars.length; i++) if (bars[i] > loud) loud = bars[i];
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--primary').trim() || '#16b364';
  var gap = width / bars.length;
  for (var j = 0; j < bars.length; j++) {
    var amp = Math.max(0.04, (bars[j] || 0) / loud);
    var bar = Math.max(2, amp * (height - 4));
    var x = j * gap + 1;
    ctx.fillRect(x, (height - bar) / 2, Math.max(2, gap - 3), bar);
  }
}
function fillSelect(select, rows, label) {
  select.innerHTML = '';
  rows.forEach(function (row, index) {
    var option = document.createElement('option');
    option.value = String(index);
    option.textContent = label(row);
    option.dataset.raw = JSON.stringify(row);
    select.appendChild(option);
  });
}
function selected(select) {
  var option = select.options[select.selectedIndex];
  return option ? JSON.parse(option.dataset.raw) : null;
}
var savedSettings = {};
function remember(partial) {
  Object.keys(partial || {}).forEach(function (key) {
    savedSettings[key] = partial[key];
  });
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(partial)
  });
}
function selectSaved(select, id) {
  if (!select || !id) return;
  for (var i = 0; i < select.options.length; i++) {
    var raw = {};
    try { raw = JSON.parse(select.options[i].dataset.raw || '{}'); } catch (err) { raw = {}; }
    if (raw.id === id || raw.uid === id || raw.name === id) {
      select.selectedIndex = i;
      return;
    }
  }
}
var modelCatalog = [];
function engineModels(engine) {
  var id = engine && engine.id;
  for (var i = 0; i < modelCatalog.length; i++) {
    if (modelCatalog[i].id === id) return modelCatalog[i].models || [];
  }
  return [];
}
function fillEngineModels() {
  var engine = selected(document.getElementById('engine'));
  var select = document.getElementById('model');
  var rows = engineModels(engine);
  fillSelect(select, rows, function (row) {
    return row.label + '  ' + row.note;
  });
  var saved = savedSettings.model;
  var belongs = rows.some(function (row) { return row.id === saved; });
  if (belongs) selectSaved(select, saved);
}
function loadModels() {
  fetch('/api/models').then(function (res) { return res.json(); }).then(function (data) {
    modelCatalog = data.engines || [];
    var select = document.getElementById('engine');
    fillSelect(select, modelCatalog, function (row) {
      return row.label;
    });
    selectSaved(select, savedSettings.engine || 'transformers');
    fillEngineModels();
  });
  fetch('/api/languages').then(function (res) { return res.json(); }).then(function (data) {
    var select = document.getElementById('language');
    fillSelect(select, data.languages, function (row) {
      return row.label;
    });
    selectSaved(select, savedSettings.language);
  });
  promptReady = false;
  fetch('/api/polish-prompt').then(function (res) { return res.json(); }).then(function (data) {
    var box = document.getElementById('polish-prompt');
    if (box && !box.value) box.value = data.prompt || '';
    promptReady = true;
  });
  var promptTimer = 0;
  document.getElementById('polish-prompt').addEventListener('input', function () {
    var value = this.value;
    clearTimeout(promptTimer);
    promptTimer = setTimeout(function () {
      if (!promptReady) return;
      fetch('/api/polish-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: value })
      });
    }, 400);
  });
  glossaryReady = false;
  fetch('/api/glossary').then(function (res) { return res.json(); }).then(function (data) {
    var box = document.getElementById('glossary');
    if (box) box.value = data.glossary || '';
    glossaryReady = true;
  });
  var glossaryTimer = 0;
  document.getElementById('glossary').addEventListener('input', function () {
    var value = this.value;
    clearTimeout(glossaryTimer);
    glossaryTimer = setTimeout(function () {
      if (!glossaryReady) return;
      fetch('/api/glossary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ glossary: value })
      });
    }, 400);
  });
  fetch('/api/polish-models').then(function (res) { return res.json(); }).then(function (data) {
    var select = document.getElementById('polish-model');
    fillSelect(select, data.models || [], function (row) {
      return row.label || row.id;
    });
    selectSaved(select, savedSettings.polish_model);
    if (data.error) setStatus(data.error);
  }).catch(function (err) { setStatus(String(err)); });
}
function addScan(text) {
  var box = document.getElementById('scan-log');
  box.textContent += text + '\n';
  box.scrollTop = box.scrollHeight;
}
function scan() {
  setStatus('正在测量各路输出，请保持声音播放。');
  document.getElementById('scan-log').textContent = '';
  addScan('正在测量各路输出…');
  return fetch('/api/devices').then(function (res) { return res.json(); }).then(function (data) {
    var devices = data.devices || [];
    if (data.error) {
      setStatus(data.error);
      addScan(data.error);
      return;
    }
    fillSelect(document.getElementById('device'), devices, function (row) {
      return row.name + '  rms=' + row.rms + ' peak=' + row.peak;
    });
    if (!devices.length) {
      var empty = data.hint || '没有测到输出设备。';
      setStatus(empty);
      addScan(empty);
      return;
    }
    var best = 0;
    devices.forEach(function (row, index) {
      if ((row.rms || 0) > (devices[best].rms || 0)) best = index;
    });
    document.getElementById('device').selectedIndex = best;
    selectSaved(document.getElementById('device'), savedSettings.device_uid || savedSettings.device_name);
    addScan('测量结果');
    devices.forEach(function (row) {
      var marks = [];
      if (row.is_default_system_output) marks.push('当前系统输出');
      if (row.error) marks.push('测量失败');
      addScan(row.name + '  rms=' + row.rms + '  peak=' + row.peak + (marks.length ? '  [' + marks.join(', ') + ']' : ''));
    });
    setStatus(data.hint || '测量完成。回到识别页后点开始。');
    if (data.hint) addScan(data.hint);
  }).catch(function (err) {
    setStatus(String(err));
    addScan('测量失败：' + err);
  });
}
function poll() {
  fetch('/api/events?since=' + since).then(function (res) { return res.json(); }).then(function (data) {
    data.events.forEach(function (item) {
      since = item.id;
      if (item.type === 'text' && item.text) addText(item);
      if (item.type === 'polish' && item.text) replaceBatch(item);
      if (item.type === 'status' && item.message) {
        setStatus(item.message);
        var note = document.getElementById('live-note');
        if (note) note.textContent = item.message;
        addLog({id: item.id, ts: item.ts, message: item.message});
      }
      if (item.type === 'log') addLog(item);
    });
    if (data.wave) wave = data.wave;
    if (data.level) {
      latest = data.level.peak || data.level.rms || 0;
      document.getElementById('meter').textContent = 'rms=' + data.level.rms + ' peak=' + data.level.peak;
    }
    if (data.error) setStatus(data.error);
    if (data.running) setListening(true);
    else if (data.running === false && !document.getElementById('start').classList.contains('pending')) setListening(false);
  }).catch(function () {});
}
function setListening(on) {
  var start = document.getElementById('start');
  var stop = document.getElementById('stop');
  start.classList.toggle('listening', on);
  stop.classList.toggle('listening', on);
  start.textContent = on ? '正在听' : '开始';
  start.disabled = on;
  stop.disabled = false;
}
function press(button, action) {
  if (!button || button.disabled) return;
  var label = button.textContent;
  button.classList.add('pressed');
  button.disabled = true;
  var done = function () {
    button.textContent = label;
    button.classList.remove('pressed');
    button.disabled = false;
  };
  try {
    var result = action();
    if (result && result.then) result.then(done, done);
    else setTimeout(done, 280);
  } catch (err) {
    done();
    setStatus(String(err));
  }
}
function showPage(name) {
  var settings = name === 'audio' || name === 'model' || name === 'prompt' || name === 'download' || name === 'settings';
  if (name === 'settings') name = 'download';
  ['live', 'audio', 'model', 'download', 'prompt'].forEach(function (id) {
    document.getElementById('page-' + id).classList.toggle('active', id === name);
    var nav = document.getElementById('nav-' + id);
    if (nav) nav.classList.toggle('active', id === name);
  });
  document.body.classList.toggle('settings-open', settings);
  document.getElementById('nav-settings').classList.toggle('active', settings);
  if (name === 'model') loadModels();
  if (name === 'download') loadDownloads();
}
function renderDownloads(rows) {
  var box = document.getElementById('download-list');
  box.innerHTML = '';
  var family = '';
  (rows || []).forEach(function (row) {
    if (row.family !== family) {
      family = row.family;
      var title = document.createElement('h2');
      title.textContent = row.family_label;
      box.appendChild(title);
    }
    var line = document.createElement('div');
    line.className = 'download-row';
    var button = document.createElement('button');
    var progress = '';
    if (row.status === 'downloading' && row.total) progress = ' ' + Math.min(99, Math.round(row.received * 100 / row.total)) + '%';
    button.textContent = row.installed ? '删除' : (row.status === 'downloading' ? '下载中' + progress : '下载');
    button.disabled = row.status === 'downloading';
    button.onclick = function () {
      var path = row.installed ? '/api/model-delete' : '/api/model-download';
      fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: row.id })
      }).then(function (res) { return res.json(); }).then(function (data) {
        if (data.error) setStatus(data.error);
        else if (data.message) setStatus(data.message);
        loadDownloads();
      }).catch(function (err) { setStatus(String(err)); });
    };
    var text = document.createElement('span');
    text.textContent = row.label + '  ' + row.size + '  ' + row.note + (row.error ? '  ' + row.error : '');
    line.appendChild(button);
    line.appendChild(text);
    box.appendChild(line);
  });
}
document.getElementById('open-models').onclick = function () {
  fetch('/api/model-dir-open', { method: 'POST' }).then(function (res) { return res.json(); }).then(function (data) {
    if (data.error) setStatus(data.error);
    else setStatus('已打开模型目录');
  }).catch(function (err) { setStatus(String(err)); });
};
function loadDownloads() {
  fetch('/api/model-catalog').then(function (res) { return res.json(); }).then(function (data) {
    if (data.dir) document.getElementById('download-note').textContent = '模型保存在 ' + data.dir + '。下完后回到模型页就能选，也可以在这里删除。';
    renderDownloads(data.models || []);
  }).catch(function (err) { setStatus(String(err)); });
}
document.getElementById('nav-live').onclick = function () { showPage('live'); };
document.getElementById('nav-audio').onclick = function () { showPage('audio'); };
document.getElementById('nav-model').onclick = function () { showPage('model'); };
document.getElementById('nav-download').onclick = function () { showPage('download'); };
document.getElementById('nav-prompt').onclick = function () { showPage('prompt'); };
document.getElementById('nav-settings').onclick = function () { showPage('settings'); };
document.getElementById('sider-toggle').onclick = function () {
  document.body.classList.toggle('sider-collapsed');
  try { localStorage.setItem('voicecat-sider', document.body.classList.contains('sider-collapsed') ? '1' : '0'); } catch (err) {}
};
if (localStorage.getItem('voicecat-sider') === '1') document.body.classList.add('sider-collapsed');
function showTab(name) {
  document.getElementById('tab-prompt').classList.toggle('active', name === 'prompt');
  document.getElementById('tab-glossary').classList.toggle('active', name === 'glossary');
  document.getElementById('panel-prompt').classList.toggle('active', name === 'prompt');
  document.getElementById('panel-glossary').classList.toggle('active', name === 'glossary');
}
document.getElementById('tab-prompt').onclick = function () { showTab('prompt'); };
document.getElementById('tab-glossary').onclick = function () { showTab('glossary'); };
document.getElementById('scan').onclick = function () {
  press(this, scan);
};
function clientLog(message) {
  setStatus(message);
  var note = document.getElementById('live-note');
  if (note) note.textContent = message;
  if (debugOn && debugBox) {
    debugBox.textContent += message + '\n';
    debugBox.scrollTop = debugBox.scrollHeight;
  }
  fetch('/api/log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: message })
  }).catch(function () {});
}
document.getElementById('start').onclick = function () {
  var button = this;
  button.classList.add('pending');
  button.textContent = '加载中';
  button.disabled = true;
  clientLog('正在准备。第一次使用这个模型要先加载，页面可能停几秒，不是死机。加载完会自动开始听。');
  try {
    var device = selected(document.getElementById('device'));
    if (!device && (savedSettings.device_name || savedSettings.device_uid)) {
      device = { name: savedSettings.device_name || '', uid: savedSettings.device_uid || '' };
    }
    var engineEl = document.getElementById('engine');
    var engine = engineEl ? (selected(engineEl) || { id: savedSettings.engine || 'transformers' }) : { id: savedSettings.engine || 'transformers' };
    var model = selected(document.getElementById('model'));
    var rows = engineModels(engine);
    if ((!model || !model.id) && savedSettings.model && rows.some(function (row) { return row.id === savedSettings.model; })) {
      model = { id: savedSettings.model };
    }
    if ((!model || !model.id) && rows.length) model = rows[0];
    var language = selected(document.getElementById('language'));
    if (!language && savedSettings.language) language = { id: savedSettings.language };
    if (!language) language = { id: 'zh-Hans' };
    var polishModel = selected(document.getElementById('polish-model'));
    if (document.getElementById('polish').checked && !polishModel) polishModel = { id: savedSettings.polish_model || 'local' };
    if (!model || !model.id) {
      button.classList.remove('pending');
      button.disabled = false;
      button.textContent = '开始';
      clientLog('没有开始：模型页还没有已下载的模型。请先打开下载页下载。');
      return;
    }
    if (!device || !(device.name || device.uid)) {
      button.classList.remove('pending');
      button.disabled = false;
      button.textContent = '开始';
      clientLog('没有开始：还没有音频输入。请先打开音频输入页，点测量输出。');
      return;
    }
    clientLog('正在开始，引擎 ' + engine.id + '，模型 ' + model.id + '，设备 ' + (device.name || device.uid));
    var payload = { name: device.name, uid: device.uid, backend: engine.id, model: model.id, language: language.id, polish: document.getElementById('polish').checked, polish_model: polishModel ? polishModel.id : '', polish_prompt: document.getElementById('polish-prompt').value };
    if (glossaryReady) payload.glossary = document.getElementById('glossary').value;
    remember({ device_name: device.name || '', device_uid: device.uid || '', engine: engine.id, model: model.id, language: language.id, polish: payload.polish, polish_model: payload.polish_model, debug: document.getElementById('debug').checked });
    var timer = setTimeout(function () {
      button.classList.remove('pending');
      button.disabled = false;
      button.textContent = '开始';
      clientLog('还在加载模型，不是卡死。加载完会自动开始听。');
    }, 8000);
    fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (res) { return res.json(); }).then(function (data) {
      clearTimeout(timer);
      button.classList.remove('pending');
      if (data.ok === false) {
        setListening(false);
        button.disabled = false;
        clientLog(data.message || '没能开始');
        return;
      }
      setListening(true);
      clientLog(data.message || '已开始');
    }).catch(function (err) {
      clearTimeout(timer);
      button.classList.remove('pending');
      setListening(false);
      button.disabled = false;
      clientLog('开始失败：' + err);
    });
  } catch (err) {
    button.classList.remove('pending');
    button.disabled = false;
    clientLog('开始失败：' + err);
  }
};
document.getElementById('stop').onclick = function () {
  var button = this;
  button.textContent = '停止中';
  fetch('/api/stop', { method: 'POST' }).then(function () {
    button.textContent = '停止';
    setListening(false);
    setStatus('已停止');
    clientLog('已停止');
  }).catch(function (err) {
    button.textContent = '停止';
    setStatus(String(err));
    clientLog('停止失败：' + err);
  });
};
var debugOn = false;
var lastLogId = 0;
var debugBox = document.getElementById('debug-box');
function addLog(item) {
  if (!debugOn || !item || !item.id || item.id <= lastLogId) return;
  lastLogId = item.id;
  debugBox.textContent += (item.ts ? item.ts + ' ' : '') + (item.message || '') + '\n';
  debugBox.scrollTop = debugBox.scrollHeight;
}
document.getElementById('debug').onchange = function () {
  debugOn = this.checked;
  document.body.classList.toggle('debug', debugOn);
  fetch('/api/debug', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ on: debugOn })
  }).then(function () {
    if (!debugOn) return;
    return fetch('/api/logs').then(function (res) { return res.json(); }).then(function (data) {
      debugBox.textContent = '';
      lastLogId = 0;
      (data.logs || []).forEach(addLog);
      if (!debugBox.textContent) debugBox.textContent = 'debug 已打开。开始识别后，采集和转写日志会显示在这里。\n';
    });
  });
};
document.getElementById('clear').onclick = function () {
  press(this, function () { raw.innerHTML = ''; polished.innerHTML = ''; });
};
setListening(false);
document.getElementById('split').oninput = applySplit;
applySplit();
fetch('/api/settings').then(function (res) { return res.json(); }).then(function (data) {
  savedSettings = data.settings || {};
  if (savedSettings.polish === false) document.getElementById('polish').checked = false;
  if (savedSettings.debug) {
    document.getElementById('debug').checked = true;
    document.getElementById('debug').onchange();
  }
  loadModels();
  scan();
});
fetch('/api/llm-config').then(function (res) { return res.json(); }).then(function (data) {
  document.getElementById('llm-base-url').value = data.base_url || '';
  document.getElementById('llm-wire-api').value = data.wire_api || 'responses';
  document.getElementById('llm-model').value = data.model || '';
  document.getElementById('llm-note').textContent = data.has_api_key ? '密钥已保存在本机应用数据里，不在项目目录，留空不会覆盖。' : '还没有保存密钥。配置会写到本机应用数据，不写进项目目录。';
});
document.getElementById('engine').onchange = function () {
  var row = selected(this);
  savedSettings.model = '';
  fillEngineModels();
  var model = selected(document.getElementById('model'));
  remember({ engine: row ? row.id : '', model: model ? model.id : '' });
};
document.getElementById('model').onchange = function () {
  var row = selected(this);
  if (row) {
    remember({ model: row.id });
    fetch('/api/model-preload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: row.id, language: (selected(document.getElementById('language')) || {}).id || savedSettings.language || 'zh-Hans', backend: (selected(document.getElementById('engine')) || {}).id || savedSettings.engine || '' })
    });
  }
};
document.getElementById('language').onchange = function () {
  var row = selected(this);
  if (row) remember({ language: row.id });
};
document.getElementById('polish-model').onchange = function () {
  var row = selected(this);
  if (row) remember({ polish_model: row.id });
};
document.getElementById('polish').onchange = function () {
  remember({ polish: this.checked });
};
document.getElementById('device').onchange = function () {
  var row = selected(this);
  if (!row) return;
  remember({ device_name: row.name || '', device_uid: row.uid || '' });
};
document.getElementById('save-llm').onclick = function () {
  var button = this;
  press(button, function () {
    return fetch('/api/llm-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        base_url: document.getElementById('llm-base-url').value,
        wire_api: document.getElementById('llm-wire-api').value,
        model: document.getElementById('llm-model').value,
        api_key: document.getElementById('llm-api-key').value
      })
    }).then(function (res) { return res.json(); }).then(function (data) {
      document.getElementById('llm-api-key').value = '';
      document.getElementById('llm-note').textContent = data.has_api_key ? '已保存。密钥不会回显。' : '已保存地址和模型，密钥仍为空。';
    });
  });
};
setInterval(drawWave, 80);
setInterval(poll, 200);
setInterval(function () {
  if (document.getElementById('page-download').classList.contains('active')) loadDownloads();
}, 1000);
</script>
</body>
</html>'''


class LiveBoard:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.seq = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.running = False
        self.error = ''
        self.binary = None
        self.level = {'rms': 0, 'peak': 0}
        self.wave = [0]
        self.debug = False
        self.logs = []

    def push(self, event):
        if event.get('type') == 'level':
            with self.lock:
                rms = int(event.get('rms') or 0)
                self.level = {'rms': rms, 'peak': int(event.get('peak') or 0)}
                self.wave.append(rms)
                del self.wave[:-180]
            return event
        with self.lock:
            self.seq += 1
            item = dict(event)
            item['id'] = self.seq
            self.events.append(item)
            del self.events[:-300]
            return item

    def add_log(self, message):
        from sysaudio_stt.stt import redact
        line = redact(str(message or '')).strip()
        if not line:
            return None
        with self.lock:
            self.seq += 1
            item = {'id': self.seq, 'type': 'log', 'ts': time.strftime('%H:%M:%S'), 'message': line}
            self.logs.append(item)
            del self.logs[:-400]
            if self.debug:
                self.events.append(item)
                del self.events[:-300]
            return item

    def since(self, last_id):
        with self.lock:
            items = [item for item in self.events if item['id'] > last_id]
            return items, self.running, self.error, dict(self.level), list(self.wave)

    def stop(self):
        self.stop_event.set()
        thread = self.thread
        if thread and thread.is_alive():
            thread.join(timeout=2)
        with self.lock:
            self.running = False


BOARD = LiveBoard()


def make_args(device, uid, model, language, polish=True, glossary=None, polish_model='', polish_prompt='', backend='transformers'):
    args = build_parser().parse_args([])
    configure_runtime(args)
    chosen = resolve_language(language or 'zh-Hans')
    args.source = 'system'
    args.device = device or ''
    args.uid = uid or ''
    from sysaudio_stt.models_catalog import get_model
    item = get_model(model)
    args.catalog_id = item['id'] if item else ''
    args.model_label = ((item or {}).get('family_label') or '') + ' / ' + ((item or {}).get('label') or model or '')
    kind = (item or {}).get('kind') or ''
    if kind == 'faster':
        args.backend = 'local'
        args.model = item.get('runtime') or model
        args.local_device = 'cpu'
        args.compute_type = 'int8'
        args.max_segment = min(float(args.max_segment or 8), 4.0)
    elif kind == 'transformers':
        args.backend = 'transformers'
        args.model = item.get('runtime') or model
        args.max_segment = min(float(args.max_segment or 8), 4.0)
    elif kind in ('sensevoice', 'offline-paraformer', 'online-paraformer', 'online-transducer'):
        args.backend = kind
        args.model = item['id']
        args.max_segment = min(float(args.max_segment or 8), 3.0)
    elif backend == 'faster-whisper':
        args.backend = 'local'
        args.model = model or args.model
        args.local_device = 'cpu'
        args.compute_type = 'int8'
    else:
        args.backend = 'transformers'
        args.model = model or args.model
    args.language = chosen['whisper']
    args.script = chosen['script']
    args.style_prompt = chosen['prompt']
    args.polish = bool(polish)
    args.polish_model = polish_model or 'local'
    args.polish_prompt = polish_prompt or ''
    if glossary is not None:
        args.glossary = glossary
    return args


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, content_type):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), 'application/json; charset=utf-8')

    def _read_json(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > 100000:
            return {}
        return json.loads(self.rfile.read(length).decode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self._send(200, PAGE, 'text/html; charset=utf-8')
            return
        if parsed.path == '/api/models':
            from sysaudio_stt.models_catalog import installed_engines
            engines = installed_engines()
            models = []
            for group in engines:
                models.extend(group['models'])
            self._json(200, {'models': models, 'engines': engines})
            return
        if parsed.path == '/api/model-catalog':
            from sysaudio_stt.models_catalog import models_root, public_rows
            self._json(200, {'models': public_rows(), 'dir': str(models_root())})
            return
        if parsed.path == '/api/languages':
            self._json(200, {'languages': list(LANGUAGE_CHOICES)})
            return
        if parsed.path == '/api/polish-prompt':
            from sysaudio_stt.polish import load_polish_prompt
            self._json(200, {'prompt': load_polish_prompt()})
            return
        if parsed.path == '/api/settings':
            from sysaudio_stt.polish import load_settings
            self._json(200, {'settings': load_settings()})
            return
        if parsed.path == '/api/llm-config':
            from sysaudio_stt.polish import public_llm_config
            self._json(200, public_llm_config())
            return
        if parsed.path == '/api/glossary':
            from sysaudio_stt.polish import load_glossary
            self._json(200, {'glossary': load_glossary()})
            return
        if parsed.path == '/api/polish-models':
            from sysaudio_stt.polish import list_llm_models
            from sysaudio_stt.stt import STTError, redact
            local = [{'id': 'local', 'label': '本地纠错（快）'}]
            try:
                ids = list_llm_models()
            except (STTError, OSError, ValueError) as exc:
                self._json(200, {'models': local, 'error': redact(str(exc))})
                return
            self._json(200, {'models': local + [{'id': item, 'label': item} for item in ids], 'error': ''})
            return
        if parsed.path == '/api/devices':
            import sys
            from sysaudio_stt.capture import CaptureError
            from sysaudio_stt.stt import redact
            try:
                if sys.platform != 'win32':
                    require_binary(BOARD.binary)
                scanned = scan_outputs(BOARD.binary)
            except (CaptureError, OSError, ValueError) as exc:
                self._json(200, {'devices': [], 'hint': '', 'error': redact(str(exc))})
                return
            devices = [item for item in scanned if item.get('type') != 'hint']
            hint = next((str(item.get('message') or '') for item in scanned if item.get('type') == 'hint'), '')
            self._json(200, {'devices': devices, 'hint': hint, 'error': ''})
            return
        if parsed.path == '/api/logs':
            with BOARD.lock:
                logs = list(BOARD.logs)
            self._json(200, {'logs': logs})
            return
        if parsed.path == '/api/events':
            query = parse_qs(parsed.query)
            last = int((query.get('since') or ['0'])[0] or 0)
            items, running, error, level, wave = BOARD.since(last)
            self._json(200, {'events': items, 'running': running, 'error': error, 'level': level, 'wave': wave})
            return
        self._json(404, {'error': 'not found'})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/polish-prompt':
            from sysaudio_stt.polish import save_polish_prompt
            body = self._read_json()
            save_polish_prompt(body.get('prompt') or '')
            self._json(200, {'ok': True})
            return
        if parsed.path == '/api/glossary':
            from sysaudio_stt.polish import save_glossary
            body = self._read_json()
            save_glossary(body.get('glossary') or '')
            self._json(200, {'ok': True})
            return
        if parsed.path == '/api/correct':
            from sysaudio_stt.polish import add_correction
            body = self._read_json()
            try:
                saved = add_correction(body.get('wrong') or '', body.get('right') or '')
            except ValueError as exc:
                self._json(200, {'ok': False, 'error': str(exc)})
                return
            self._json(200, {'ok': True, **saved})
            return
        if parsed.path == '/api/settings':
            from sysaudio_stt.polish import save_settings
            body = self._read_json()
            body.pop('api_key', None)
            self._json(200, {'settings': save_settings(body)})
            return
        if parsed.path == '/api/llm-config':
            from sysaudio_stt.polish import save_llm_config
            body = self._read_json()
            key = body.get('api_key')
            saved = save_llm_config(body.get('base_url') or '', body.get('wire_api') or 'responses', body.get('model') or '', None if key in (None, '') else key)
            self._json(200, saved)
            return
        if parsed.path == '/api/model-download':
            from sysaudio_stt.models_catalog import start_download
            body = self._read_json()
            try:
                message = start_download(body.get('id') or '')
            except (ValueError, RuntimeError) as exc:
                self._json(200, {'ok': False, 'error': str(exc)})
                return
            self._json(200, {'ok': True, 'message': message})
            return
        if parsed.path == '/api/model-preload':
            body = self._read_json()
            threading.Thread(target=preload_model, args=(body.get('id') or '', body.get('language') or 'zh-Hans', body.get('backend') or ''), daemon=True).start()
            self._json(200, {'ok': True})
            return
        if parsed.path == '/api/model-dir-open':
            from sysaudio_stt.models_catalog import open_models_dir
            self._json(200, {'ok': True, 'dir': open_models_dir()})
            return
        if parsed.path == '/api/model-delete':
            from sysaudio_stt.models_catalog import delete_model
            body = self._read_json()
            try:
                message = delete_model(body.get('id') or '')
            except (ValueError, RuntimeError) as exc:
                self._json(200, {'ok': False, 'error': str(exc)})
                return
            self._json(200, {'ok': True, 'message': message})
            return
        if parsed.path == '/api/log':
            body = self._read_json()
            BOARD.add_log(body.get('message') or '')
            self._json(200, {'ok': True})
            return
        if parsed.path == '/api/debug':
            body = self._read_json()
            with BOARD.lock:
                BOARD.debug = bool(body.get('on'))
            self._json(200, {'ok': True, 'on': BOARD.debug})
            return
        if parsed.path == '/api/stop':
            BOARD.stop()
            BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '已停止'})
            self._json(200, {'ok': True})
            return
        if parsed.path != '/api/start':
            self._json(404, {'error': 'not found'})
            return
        if BOARD.thread and BOARD.thread.is_alive():
            BOARD.add_log('已经在听。先停止再开始。')
            self._json(200, {'ok': False, 'message': '已经在听。先停止再开始。'})
            return
        try:
            body = self._read_json()
            args = make_args(
                body.get('name') or '',
                body.get('uid') or '',
                body.get('model') or '',
                body.get('language') or '',
                polish=body.get('polish', True),
                glossary=body.get('glossary') if 'glossary' in body else None,
                polish_model=body.get('polish_model') or '',
                polish_prompt=body.get('polish_prompt') or '',
                backend=body.get('backend') or 'transformers',
            )
            BOARD.stop_event = threading.Event()
            BOARD.error = ''
            BOARD.running = True

            def runner():
                try:
                    BOARD.add_log('正在加载识别模型。第一次会占住页面几秒，不是死机。')
                    BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '正在加载识别模型。第一次会占住页面几秒，不是死机。'})
                    code = run_transcribe(args, on_event=BOARD.push, stop_event=BOARD.stop_event, install_signals=False)
                    if code:
                        BOARD.add_log('识别没有开始，退出码 ' + str(code))
                        BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '识别没有开始。请看调试信息。'})
                except Exception as exc:
                    logging.exception('网页转写失败')
                    BOARD.error = str(exc)
                    BOARD.add_log('转写失败：' + str(exc))
                    BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': str(exc)})
                finally:
                    BOARD.running = False

            def warm():
                try:
                    from sysaudio_stt.local_polish import warmup
                    warmup()
                except Exception:
                    logging.exception('本地纠错模型预热失败')

            if not args.polish:
                polish_note = '不润色'
            elif args.polish_model == 'local':
                polish_note = '用本地纠错，不走远程大模型'
            else:
                from sysaudio_stt.polish import llm_credentials
                _config, key_name, ready = llm_credentials()
                polish_note = '会润色' if ready else ('未配置 ' + key_name + '，只显示原文')
            message = '开始听「%s」，%s，润色模型 %s，语言 %s。%s。' % (
                args.device or '当前输出',
                args.model_label or args.model,
                args.polish_model or '未选',
                args.language,
                polish_note,
            )
            BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': message})
            BOARD.add_log(message)
            self._json(200, {'ok': True, 'message': message})
            BOARD.thread = threading.Thread(target=runner, daemon=True)
            BOARD.thread.start()
            if args.polish and args.polish_model == 'local':
                threading.Thread(target=warm, daemon=True).start()
        except Exception as exc:
            from sysaudio_stt.stt import redact
            text = redact(str(exc))
            BOARD.add_log('开始失败：' + text)
            self._json(200, {'ok': False, 'message': '开始失败：' + text})

def attach_debug_logging():
    import logging
    root = logging.getLogger()
    if any(getattr(item, '_sysaudio_debug', False) for item in root.handlers):
        return
    handler = logging.Handler()
    handler._sysaudio_debug = True
    handler.setFormatter(logging.Formatter('%(levelname)s %(name)s %(message)s'))

    def emit(record):
        try:
            BOARD.add_log(handler.format(record))
        except Exception:
            return

    handler.emit = emit
    root.addHandler(handler)


def preload_model(model_id='', language='zh-Hans', backend=''):
    from sysaudio_stt.polish import load_settings
    from sysaudio_stt.stt import preload_transcriber
    settings = load_settings()
    model_id = model_id or settings.get('model') or ''
    if not model_id:
        BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '还没有选择模型。先到模型页选一个，再点开始。'})
        return
    BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '正在预加载「' + model_id + '」。第一次会占住页面几秒，不是死机。'})
    time.sleep(0.4)
    try:
        args = make_args(
            settings.get('device_name') or '',
            settings.get('device_uid') or '',
            model_id,
            language or settings.get('language') or 'zh-Hans',
            backend=backend or settings.get('engine') or 'transformers',
        )
        preload_transcriber(args)
        from sysaudio_stt.local_polish import warmup
        BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '识别模型已就绪，正在准备标点。'})
        warmup()
        BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '模型已就绪。点开始就能听，不会再卡在加载。'})
    except Exception as exc:
        BOARD.push({'type': 'status', 'ts': time.strftime('%H:%M:%S'), 'message': '预加载失败：' + str(exc)})


def serve(binary, host='127.0.0.1', port=8765, headless=False):
    import sys
    attach_debug_logging()
    if sys.platform != 'win32':
        require_binary(binary)
    BOARD.binary = binary
    httpd = None
    last_error = None
    for offset in range(8):
        try:
            httpd = ThreadingHTTPServer((host, port + offset), Handler)
            port = port + offset
            break
        except OSError as exc:
            last_error = exc
    if httpd is None:
        raise SystemExit('无法打开本机网页: %s' % last_error)
    url = 'http://%s:%d/' % (host, port)
    print('READY ' + url, flush=True)
    print('只监听本机。关闭窗口会停止识别。', flush=True)
    class _LogStream:
        def write(self, data):
            text = str(data or '')
            try:
                sys.__stdout__.write(text)
            except Exception:
                pass
            line = text.strip()
            if line and not line.startswith('READY '):
                BOARD.add_log(line)
        def flush(self):
            try:
                sys.__stdout__.flush()
            except Exception:
                pass
    sys.stdout = _LogStream()
    BOARD.add_log('识别引擎已启动 ' + url)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=preload_model, daemon=True).start()
    if headless:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            BOARD.stop()
        finally:
            httpd.server_close()
        return 0
    try:
        import webview
    except ImportError as exc:
        httpd.shutdown()
        httpd.server_close()
        raise SystemExit('未安装桌面窗口组件 pywebview') from exc

    def on_closed():
        BOARD.stop()
        httpd.shutdown()

    window = webview.create_window('VoiceCat', url, width=1100, height=760, min_size=(860, 560))
    window.events.closed += on_closed
    try:
        webview.start()
    except KeyboardInterrupt:
        BOARD.stop()
    finally:
        BOARD.stop()
        httpd.shutdown()
        httpd.server_close()
    return 0

