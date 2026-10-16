'use strict';
/**
 * Nova AI — chat interface, model selector, training flow.
 * Every visible button here is functional. There is no backend: the model
 * and all data live in the browser.
 */

/* ---------------- model registry ----------------
   Designed so more Nova models can be added later. Only Nova 1.0 exists
   today — nothing unavailable is ever shown as selectable. */
const MODELS = [
  {
    id: 'nova-1.0',
    name: 'Nova 1.0',
    project: 'nova',
    type: 'Custom, independently developed language model',
    status: 'Initial model',
    selected: true,
  },
];

/* ---------------- state ---------------- */

let chats = [];            // [{id, title, created, messages:[{role, content, kind}]}]
let activeChatId = null;
let model = null;          // Nova1Model instance
let generating = false;

/* ---------------- tiny DOM helpers ---------------- */

const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

/* ---------------- markdown-lite renderer (safe) ---------------- */

function renderMarkdown(src) {
  if (src === null || src === undefined) return '';
  const blocks = String(src).split('```');

  const renderInline = (s) => s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`\n]+)`/g, '<code class="inline">$1</code>');

  const renderText = (text) => {
    const lines = text.split('\n');
    let out = '', i = 0;
    let listType = null, listItems = [];
    const flushList = () => {
      if (!listType) return;
      out += `<${listType}>` + listItems.map((li) => `<li>${li}</li>`).join('') + `</${listType}>`;
      listType = null; listItems = [];
    };
    while (i < lines.length) {
      const raw = lines[i];
      const esc = escapeHtml(raw);
      const ul = esc.match(/^\s*[-*]\s+(.*)$/);
      const ol = esc.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ul) {
        if (listType !== 'ul') { flushList(); listType = 'ul'; }
        listItems.push(renderInline(ul[1]));
      } else if (ol) {
        if (listType !== 'ol') { flushList(); listType = 'ol'; }
        listItems.push(renderInline(ol[1]));
      } else {
        flushList();
        if (esc.trim() === '') {
          // paragraph break
        } else {
          out += `<p>${renderInline(esc).replace(/\n/g, '<br>')}</p>`;
        }
      }
      i++;
    }
    flushList();
    return out;
  };

  let html = '';
  for (let b = 0; b < blocks.length; b++) {
    if (b % 2 === 1) {
      // code block: strip optional language line
      let code = blocks[b];
      code = code.replace(/^[a-zA-Z0-9+#-]*\n?/, '');
      html += `<pre class="codeblock"><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`;
    } else if (blocks[b].trim() !== '' || b === blocks.length - 1) {
      html += renderText(blocks[b]);
    }
  }
  return html;
}

/* ---------------- persistence helpers ---------------- */

function saveChats() {
  const r = NovaStorage.set(NovaStorage.KEYS.chats, chats);
  if (!r.ok) handleSaveFailure(r, 'chat history');
}

function saveActiveChat() {
  NovaStorage.set(NovaStorage.KEYS.activeChat, activeChatId);
}

function saveModel() {
  if (!model) return;
  const r = NovaStorage.set(NovaStorage.KEYS.model, model.toJSON());
  if (!r.ok) handleSaveFailure(r, 'model weights');
}

function handleSaveFailure(r, what) {
  if (r.reason === 'quota') {
    toast(`Browser storage is full — ${what} could not be saved for this session.`, 'err');
  } else if (r.reason === 'unavailable') {
    // banner already shown at startup
  } else {
    toast(`Could not save ${what} to storage.`, 'err');
  }
}

/* ---------------- toasts ---------------- */

function toast(msg, kind) {
  const t = el('div', 'toast' + (kind ? ' ' + kind : ''), escapeHtml(msg));
  $('toasts').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity 0.4s'; }, 3400);
  setTimeout(() => t.remove(), 3900);
}

/* ---------------- model selector ---------------- */

function renderModelMenu() {
  const menu = $('modelMenu');
  menu.innerHTML = '';
  for (const m of MODELS) {
    const card = el('button', 'model-card' + (m.selected ? ' selected' : ''));
    card.type = 'button';
    card.setAttribute('role', 'option');
    card.setAttribute('aria-selected', m.selected ? 'true' : 'false');
    const status = model && model.trained
      ? `<div class="mc-status trained">${model._base ? 'Base training' : 'Trained'} · ${model.trainSteps.toLocaleString()} steps · loss ${Number(model.lastLoss).toFixed(3)}</div>`
      : `<div class="mc-status untrained">Not trained yet — needs training data</div>`;
    card.innerHTML = `
      <div class="mc-head">
        <svg width="16" height="16" viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="mg${m.id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#4dd6ff"/></linearGradient></defs><path fill="url(#mg${m.id})" d="M16 1c1.4 8.2 5.4 12.2 14 14-8.6 1.8-12.6 5.8-14 14-1.4-8.2-5.4-12.2-14-14 8.6-1.8 12.6-5.8 14-14z"/></svg>
        ${escapeHtml(m.name)}
        ${m.selected ? '<span class="badge-check">Selected</span>' : ''}
      </div>
      <div class="mc-rows">
        <div>Model ID: <b>${escapeHtml(m.id)}</b></div>
        <div>Project: <b>${escapeHtml(m.project)}</b></div>
        <div>Type: ${escapeHtml(m.type)}</div>
        <div>Status: ${escapeHtml(m.status)}</div>
      </div>
      ${status}`;
    card.addEventListener('click', () => {
      m.selected = true; // only one model, stays selected
      closeModelMenu();
    });
    menu.appendChild(card);
  }
  const foot = el('div', 'mm-foot', 'More Nova models are planned. Nothing unavailable is selectable.');
  menu.appendChild(foot);
}

function closeModelMenu() {
  $('modelMenu').classList.remove('open');
  $('modelBtn').setAttribute('aria-expanded', 'false');
}

/* ---------------- chat rendering ---------------- */

function activeChat() {
  return chats.find((c) => c.id === activeChatId) || null;
}

function renderChatList() {
  const list = $('chatList');
  list.innerHTML = '';
  if (chats.length === 0) {
    list.appendChild(el('div', 'empty', 'No conversations yet'));
    return;
  }
  const sorted = [...chats].sort((a, b) => b.created - a.created);
  for (const c of sorted) {
    const item = el('div', 'chat-item' + (c.id === activeChatId ? ' active' : ''));
    item.innerHTML = `<div class="chat-title">${escapeHtml(c.title || 'New chat')}</div><button class="chat-del" title="Delete conversation" aria-label="Delete conversation">✕</button>`;
    item.addEventListener('click', (e) => {
      if (e.target.closest('.chat-del')) return;
      activeChatId = c.id;
      saveActiveChat();
      renderChatList();
      renderMessages();
      closeSidebar();
    });
    item.querySelector('.chat-del').addEventListener('click', (e) => {
      e.stopPropagation();
      chats = chats.filter((x) => x.id !== c.id);
      if (activeChatId === c.id) {
        activeChatId = chats.length ? chats[chats.length - 1].id : null;
      }
      saveChats(); saveActiveChat();
      renderChatList(); renderMessages();
    });
    list.appendChild(item);
  }
}

function welcomeHTML() {
  return `
    <div class="welcome">
      <div class="w-logo">
        <svg width="52" height="52" viewBox="0 0 32 32" aria-hidden="true">
          <defs><linearGradient id="wg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#4dd6ff"/></linearGradient></defs>
          <path fill="url(#wg)" d="M16 1c1.4 8.2 5.4 12.2 14 14-8.6 1.8-12.6 5.8-14 14-1.4-8.2-5.4-12.2-14-14 8.6-1.8 12.6-5.8 14-14z"/>
        </svg>
      </div>
      <h2>Nova AI</h2>
      <p>Chat with <b>Nova 1.0</b>, a small language model trained from scratch — no external APIs, everything local. ${
        model && model.trained
          ? (model._base
              ? 'The model comes with base conversational training and answers from its own predictions. Use <b>Train</b> to teach it more with your own text.'
              : 'Your trained model is loaded and answering from its own predictions.')
          : 'The model has not been trained yet. Open <b>Train</b> in the top bar to give it learning data — until then, replies are genuine but meaningless model noise.'
      }</p>
    </div>`;
}

function messageHTML(m) {
  const role = m.role === 'user' ? 'You' : 'Nova 1.0';
  return `
    <div class="avatar">${m.role === 'user' ? 'You'[0] : ''}</div>
    <div class="bubble">
      <div class="role">${role}</div>
      ${renderMarkdown(m.content)}
    </div>`;
}

function avatarHTML(isUser) {
  if (isUser) return `<div class="avatar">${escapeHtml('Y')}</div>`;
  return `<div class="avatar"><svg width="17" height="17" viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="ag" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#4dd6ff"/></linearGradient></defs><path fill="url(#ag)" d="M16 1c1.4 8.2 5.4 12.2 14 14-8.6 1.8-12.6 5.8-14 14-1.4-8.2-5.4-12.2-14-14 8.6-1.8 12.6-5.8 14-14z"/></svg></div>`;
}

function renderMessages() {
  const inner = $('chatInner');
  inner.innerHTML = '';
  const chat = activeChat();
  if (!chat || chat.messages.length === 0) {
    inner.innerHTML = welcomeHTML();
    return;
  }
  for (const m of chat.messages) {
    const row = el('div', 'msg ' + m.role + (m.error ? ' error' : ''));
    row.innerHTML = avatarHTML(m.role === 'user') + `
      <div class="bubble"><div class="role">${m.role === 'user' ? 'You' : 'Nova 1.0'}</div>${renderMarkdown(m.content)}</div>`;
    inner.appendChild(row);
  }
  scrollChat();
}

function scrollChat() {
  const area = $('chatArea');
  requestAnimationFrame(() => { area.scrollTop = area.scrollHeight; });
}

/* ---------------- conversation flow ---------------- */

function newChat() {
  const c = { id: 'c' + Date.now() + Math.floor(Math.random() * 1000), title: 'New chat', created: Date.now(), messages: [] };
  chats.push(c);
  activeChatId = c.id;
  saveChats(); saveActiveChat();
  renderChatList(); renderMessages();
  closeSidebar();
  $('input').focus();
}

async function handleSend() {
  const input = $('input');
  const text = input.value.trim();
  if (!text || generating) return;

  let chat = activeChat();
  if (!chat) {
    newChat();
    chat = activeChat();
  }

  chat.messages.push({ role: 'user', content: text, ts: Date.now() });
  if (chat.title === 'New chat') {
    chat.title = text.length > 42 ? text.slice(0, 42) + '…' : text;
  }
  input.value = '';
  autosize(input);
  updateSendState();
  saveChats();
  renderChatList();
  renderMessages();

  generating = true;
  const loading = el('div', 'loading-row', `
    <div class="avatar"><svg width="17" height="17" viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="lg2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c5cff"/><stop offset="1" stop-color="#4dd6ff"/></linearGradient></defs><path fill="url(#lg2)" d="M16 1c1.4 8.2 5.4 12.2 14 14-8.6 1.8-12.6 5.8-14 14-1.4-8.2-5.4-12.2-14-14 8.6-1.8 12.6-5.8 14-14z"/></svg></div>
    <div class="dots"><i></i><i></i><i></i></div>`);
  $('chatInner').appendChild(loading);
  scrollChat();

  try {
    // Nova 1.0 generates by continuing your message from its own next-token
    // predictions (seeded with your text). This is real sampling from the
    // trained weights — never scripted replies.
    const seedIds = model.tokenizer.encode(text).slice(0, -1); // drop <eos>: we continue the text
    const maxTokens = Math.min(320, 60 + Math.floor(text.length / 2));
    const out = trimReply(await sampleAsync(seedIds, maxTokens, 0.62));

    let content;
    const clean = out.trim();
    if (!model.trained) {
      content =
        `> **Nova 1.0 has not been trained yet.** Its weights are randomly initialized, so what you see below is the model's genuine raw prediction — real neural network output, but it carries no meaning yet.\n\n` +
        `**Why this happens:** a language model must be trained on text to produce sensible language. Right now Nova 1.0 has never seen any.\n\n` +
        `**What to do:** tap **Train** in the top bar and paste some text (stories, dialogue, notes). Nova will train on it, in your browser, and replies will start reflecting that data.\n\n` +
        'Raw model output for this input:\n\n```\n' + (clean === '' ? '(the untrained model produced an empty completion)' : clean) + '\n```';
    } else if (clean === '') {
      content = 'The model produced an empty continuation for this input. Try rephrasing, or train it on more (or more varied) text.';
    } else {
      content = clean;
    }

    chat.messages.push({ role: 'assistant', content, ts: Date.now() });
  } catch (err) {
    console.error(err);
    chat.messages.push({
      role: 'assistant',
      error: true,
      content: `**Something went wrong while generating.**\n\n\`${escapeHtml(err.message || String(err))}\``,
      ts: Date.now(),
    });
  } finally {
    loading.remove();
    generating = false;
    saveChats();
    renderChatList();
    renderMessages();
  }
}

/**
 * Trims a raw continuation to a tidy reply-sized chunk: cut at the first
 * sentence end (or a line break) within a readable window. The text itself
 * is untouched model output — only where we stop reading is chosen.
 */
function trimReply(text) {
  let t = String(text).replace(/\s+$/, '');
  if (!t) return t;
  const window = t.slice(0, 300);
  let cut = -1;
  for (let i = 0; i < window.length; i++) {
    if ((window[i] === '.' || window[i] === '!' || window[i] === '?') &&
        (i + 1 >= window.length || window[i + 1] === ' ' || window[i + 1] === '\n')) {
      cut = i + 1;
      break;
    }
  }
  if (cut === -1) {
    const nl = window.indexOf('\n', 20);
    cut = nl !== -1 ? nl : 220;
  }
  return t.slice(0, cut).trim();
}

/** Runs generation in a microtask so the loading row paints first. */
function sampleAsync(seedIds, maxTokens, temp) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      try {
        resolve(model.generate(maxTokens, temp, seedIds));
      } catch (e) { reject(e); }
    }, 30);
  });
}

/* ---------------- composer ---------------- */

function autosize(t) {
  t.style.height = 'auto';
  t.style.height = Math.min(t.scrollHeight, 180) + 'px';
}

function updateSendState() {
  $('btnSend').disabled = generating || $('input').value.trim() === '';
}

/* ---------------- training ---------------- */

let training = false;
let cancelTraining = false;

function openTrainModal() {
  $('trainBackdrop').classList.add('open');
  $('trainText').value = '';
  $('trainProgress').classList.remove('show');
  $('btnDoTrain').disabled = false;
  $('btnDoTrain').style.display = '';
  $('btnCancelTrain').style.display = 'none';
  updateTrainButtonLabel();
  setTimeout(() => $('trainText').focus(), 60);
}

function closeTrainModal() {
  if (training) return; // must cancel first
  $('trainBackdrop').classList.remove('open');
}

function updateTrainButtonLabel() {
  $('trainBtnLabel').textContent = model && model.trained ? 'Train more' : 'Train';
}

async function startTraining() {
  if (training) return;
  const text = $('trainText').value;
  if (!text || text.trim().length < 50) {
    toast('Paste at least ~50 characters of text for Nova to learn from.', 'err');
    return;
  }
  const epochs = Math.max(1, Math.min(parseInt($('trainEpochs').value, 10) || 8, 50));
  $('trainEpochs').value = epochs;

  training = true; cancelTraining = false;
  $('btnDoTrain').disabled = true;
  $('btnDoTrain').style.display = 'none';
  $('btnCancelTrain').style.display = '';
  $('trainProgress').classList.add('show');
  $('trainBar').style.width = '0%';
  $('trainSample').textContent = 'Training will show a live sample after each epoch…';
  $('trainStatusText').textContent = 'Starting…';

  try {
    const res = await model.train(text, {
      epochs,
      shouldCancel: () => cancelTraining,
      onProgress: (p) => {
        const pct = Math.round(((p.epoch - 1 + (p.step % 1e9 ? 0 : 0)) / p.epochs) * 100);
        if (p.sample === null) {
          $('trainStatusText').textContent =
            `Epoch ${p.epoch}/${p.epochs} · step ${p.step.toLocaleString()} · loss ${p.loss.toFixed(3)}`;
          $('trainBar').style.width = `${Math.round((p.epoch - 1) / p.epochs * 100 + 100 / p.epochs / 2)}%`;
        } else {
          $('trainStatusText').textContent =
            `Epoch ${p.epoch}/${p.epochs} done · loss ${p.loss.toFixed(3)}`;
          $('trainBar').style.width = `${Math.round(p.epoch / p.epochs * 100)}%`;
          $('trainSample').textContent = p.sample.trim() || '(empty sample)';
          saveModel(); // checkpoint after each epoch
        }
      },
    });
    saveModel();
    if (res.cancelled) {
      toast('Training stopped. Progress so far was saved.', 'ok');
    } else {
      toast(`Training complete — loss ${res.loss.toFixed(3)} after ${res.steps.toLocaleString()} steps.`, 'ok');
    }
  } catch (err) {
    console.error(err);
    toast('Training failed: ' + (err.message || err), 'err');
  } finally {
    training = false;
    $('btnDoTrain').disabled = false;
    $('btnDoTrain').style.display = '';
    $('btnCancelTrain').style.display = 'none';
    updateTrainButtonLabel();
    renderModelMenu();
    renderMessages(); // refresh welcome/trained state
  }
}

function resetModel() {
  if (training) return;
  if (!confirm('Reset Nova 1.0? All learned weights and vocabulary will be deleted from this browser. Chats are kept.')) return;
  model.reset();
  NovaStorage.remove(NovaStorage.KEYS.model);
  saveModel();
  updateTrainButtonLabel();
  renderModelMenu();
  renderMessages();
  toast('Nova 1.0 was reset to its untrained initial state.', 'ok');
}

/* ---------------- sidebar (mobile) ---------------- */

function openSidebar() { $('sidebar').classList.add('open'); $('sidebarBackdrop').classList.add('show'); }
function closeSidebar() { $('sidebar').classList.remove('open'); $('sidebarBackdrop').classList.remove('show'); }

/* ---------------- boot ---------------- */

function boot() {
  // storage support
  if (!NovaStorage.available) {
    $('storageBanner').textContent =
      'Browser storage is unavailable or disabled — Nova will run in memory for this session only.';
    $('storageBanner').classList.add('show');
  }

  // load model
  let loadedFromStorage = false;
  const saved = NovaStorage.get(NovaStorage.KEYS.model, null);
  if (saved) {
    try {
      model = Nova1Model.fromJSON(saved);
      loadedFromStorage = true;
    } catch (e) {
      console.warn('Nova checkpoint unreadable, starting fresh.', e);
      model = new Nova1Model();
    }
  } else if (window.NOVA_BASE_CHECKPOINT) {
    try {
      model = Nova1Model.fromJSON(window.NOVA_BASE_CHECKPOINT);
      model._base = true; // base conversational training (train/corpus.txt)
    } catch (e) {
      console.warn('Base checkpoint unreadable, starting fresh.', e);
      model = new Nova1Model();
    }
  } else {
    model = new Nova1Model();
  }
  if (loadedFromStorage && model.trained) {
    setTimeout(() => toast('Loaded your trained Nova 1.0 from browser storage.', 'ok'), 400);
  } else if (!loadedFromStorage && model._base && model.trained) {
    setTimeout(() => toast('Nova 1.0 loaded with base conversational training.', 'ok'), 400);
  }

  // load chats
  chats = NovaStorage.get(NovaStorage.KEYS.chats, []);
  if (!Array.isArray(chats)) chats = [];
  const savedActive = NovaStorage.get(NovaStorage.KEYS.activeChat, null);
  if (savedActive && chats.some((c) => c.id === savedActive)) {
    activeChatId = savedActive;
  } else if (chats.length) {
    activeChatId = chats[chats.length - 1].id;
  }

  // --- wire up events ---
  $('composer').addEventListener('submit', (e) => { e.preventDefault(); handleSend(); });
  const input = $('input');
  input.addEventListener('input', () => { autosize(input); updateSendState(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  $('btnNewChatSide').addEventListener('click', newChat);
  $('btnNewChatTop').addEventListener('click', newChat);
  $('btnMenu').addEventListener('click', openSidebar);
  $('sidebarBackdrop').addEventListener('click', closeSidebar);

  $('modelBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    const menu = $('modelMenu');
    const open = !menu.classList.contains('open');
    menu.classList.toggle('open', open);
    $('modelBtn').setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) renderModelMenu();
  });
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.model-selector')) closeModelMenu();
  });

  $('btnTrain').addEventListener('click', openTrainModal);
  $('btnDoTrain').addEventListener('click', startTraining);
  $('btnCancelTrain').addEventListener('click', () => { cancelTraining = true; $('trainStatusText').textContent = 'Stopping…'; });
  $('btnResetModel').addEventListener('click', resetModel);
  $('trainBackdrop').addEventListener('click', (e) => { if (e.target === $('trainBackdrop')) closeTrainModal(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeTrainModal(); closeModelMenu(); }
  });

  renderModelMenu();
  renderChatList();
  renderMessages();
  updateTrainButtonLabel();
  updateSendState();
}

document.addEventListener('DOMContentLoaded', boot);
