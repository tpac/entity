// ===========================================================================
// tabs/thalamus.js — the brain's standing-intent queue.
// ---------------------------------------------------------------------------
// Streams is the streams speaking to each other; this is the brain speaking to
// its streams. An item is an ASK (needs an answer), a REMINDER (carries a
// clock) or a NOTICE (an undated FYI) — the kind is DERIVED, never stored, and
// the derivation is mirrored from servers/channels/thalamus/thalamus_contract
// (kind_of). The dashboard may not import servers.*, so it's replicated here;
// keep the two in step if the contract's partition ever changes.
//
// The card carries what the operator can't get from the queue count: the
// delivery ledger — which sessions actually saw this, and at which moment
// (boot / stop). An ask delivered to four streams and still unanswered looks
// nothing like one that never reached anybody, and only the ledger tells them
// apart.
//
// Read-only. Answering is a write, and it belongs to Anchor (thalamus_resolve)
// — the dashboard's one write path stays the self-send composer.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { escapeHtml, relativeTime, untilTime, localTime } from '/static/lib/dom.js';
import { sessionLabel, sessionTooltip } from '/static/lib/sessions.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

let _items = [];
const _open = new Set();     // item ids whose ledger/detail is expanded

// ── Kind (mirrors thalamus_contract.kind_of) ───────────────────────────
const KINDS = {
  ask:      { label: 'asks',   glyph: '❓', cls: 'ask' },
  reminder: { label: 'reminds', glyph: '⏰', cls: 'reminder' },
  notice:   { label: 'notes',  glyph: '📌', cls: 'notice' },
};

function _kind(it) {
  if (it.needs_answer) return 'ask';
  if (it.deliver_at) return 'reminder';
  return 'notice';
}

// ── Audience, in plain words ───────────────────────────────────────────
function _audienceText(it) {
  if (it.target_session) {
    return 'for ' + (sessionLabel(it.target_session) || it.target_short);
  }
  return it.audience === 'every_session'
    ? 'every session, once each'
    : 'the first session after it is due';
}

function _stateText(it) {
  if (it.state !== 'open') return it.state;
  // An open item that is past its expiry hasn't been swept yet — say so
  // rather than showing it as healthy.
  if (it.expires_at && it.expires_at < new Date().toISOString()) return 'overdue';
  return 'open';
}

// ── One item ───────────────────────────────────────────────────────────
function _renderItem(it) {
  const kind = _kind(it);
  const k = KINDS[kind];
  const state = _stateText(it);
  const open = _open.has(it.id);
  const dels = it.deliveries || [];

  // Deadlines are the point of a queue, so they read forward ('in 13d'),
  // not through relativeTime's past-only clamp.
  const meta = [];
  meta.push(_audienceText(it));
  if (it.deliver_at) meta.push('due ' + untilTime(it.deliver_at));
  if (it.expires_at) meta.push('expires ' + untilTime(it.expires_at));
  if (it.armed_epoch) meta.push('deferred ×' + it.armed_epoch);

  const refsHtml = (it.refs && it.refs.length)
    ? '<div class="thal-refs">'
      + it.refs.map(r => '<span class="thal-ref" data-ref="' + escapeHtml(String(r)) + '">'
          + escapeHtml(String(r)) + '</span>').join('')
      + '</div>'
    : '';

  const answerHtml = it.answer
    ? '<div class="thal-answer"><span class="thal-answer-label">answered</span>'
      + escapeHtml(it.answer) + '</div>'
    : '';

  // Ledger — who saw it. Collapsed to a count until the card is opened.
  let ledger;
  if (!dels.length) {
    ledger = '<div class="thal-ledger thal-ledger--none">not delivered to any session yet</div>';
  } else if (!open) {
    ledger = '<div class="thal-ledger">👁 seen by ' + dels.length
      + ' session' + (dels.length === 1 ? '' : 's') + ' · click to see who</div>';
  } else {
    ledger = '<div class="thal-ledger"><div class="thal-ledger-head">👁 seen by</div>'
      + dels.map(d =>
          '<div class="thal-ledger-row" title="' + escapeHtml(sessionTooltip(d.session_id)) + '">'
          + '<span class="thal-ledger-who">' + escapeHtml(sessionLabel(d.session_id) || d.session_short) + '</span>'
          + '<span class="thal-ledger-via">' + escapeHtml(d.via || '') + '</span>'
          + '<span class="thal-ledger-when" title="' + escapeHtml(localTime(d.delivered_at) || '') + '">'
          +   escapeHtml(relativeTime(d.delivered_at)) + '</span>'
          + '</div>').join('')
      + '</div>';
  }

  return '<div class="thal-card thal-card--' + k.cls + ' state-' + escapeHtml(state) + '"'
    + ' data-tid="' + escapeHtml(it.id) + '">'
    + '<div class="thal-head">'
    +   '<span class="thal-kind thal-kind--' + k.cls + '">' + k.glyph + ' ' + kind + '</span>'
    +   '<span class="thal-source">' + escapeHtml(it.source || '?') + ' ' + k.label + '</span>'
    +   '<span class="thal-state thal-state--' + escapeHtml(state) + '">' + escapeHtml(state) + '</span>'
    +   '<span class="thal-when" title="' + escapeHtml(localTime(it.created_at) || '') + '">'
    +     escapeHtml(relativeTime(it.created_at)) + '</span>'
    + '</div>'
    + '<div class="thal-body">' + escapeHtml(it.body || '') + '</div>'
    + refsHtml
    + answerHtml
    + '<div class="thal-meta">' + escapeHtml(meta.join(' · ')) + '</div>'
    + ledger
  + '</div>';
}

// ── Summary strip ──────────────────────────────────────────────────────
function _renderSummary(items) {
  const openItems = items.filter(i => i.state === 'open');
  const by = { ask: 0, reminder: 0, notice: 0 };
  for (const i of openItems) by[_kind(i)]++;
  const unseen = openItems.filter(i => !(i.deliveries || []).length).length;

  const pill = (n, label, cls) =>
    '<span class="thal-pill thal-pill--' + cls + '"><b>' + n + '</b> ' + label + '</span>';

  const parts = [
    pill(by.ask, by.ask === 1 ? 'open ask' : 'open asks', 'ask'),
    pill(by.reminder, by.reminder === 1 ? 'reminder' : 'reminders', 'reminder'),
    pill(by.notice, by.notice === 1 ? 'notice' : 'notices', 'notice'),
  ];
  if (unseen) parts.push(pill(unseen, 'not yet seen', 'unseen'));
  const closed = items.length - openItems.length;
  if (closed) parts.push(pill(closed, 'settled', 'closed'));
  return parts.join('');
}

// `.notif` is display:none until it also carries `.show` — toggling
// inline style alone leaves the badge invisible.
function _setBadge(n) {
  const b = document.getElementById('thalamus-badge');
  if (!b) return;
  b.textContent = n > 0 ? String(n) : '';
  b.classList.toggle('show', n > 0);
}

// ── Load + paint ───────────────────────────────────────────────────────
export async function loadThalamus() {
  const scopeEl = document.getElementById('thalamus-scope');
  const hoursEl = document.getElementById('thalamus-hours');
  const scope = scopeEl ? scopeEl.value : 'all';
  const hours = hoursEl ? hoursEl.value : 168;
  try {
    const body = await api.thalamus({ hours, closed: scope === 'open' ? 0 : 1 });
    _items = (body && body.items) || [];
    _paint();
  } catch (e) { console.error('[thalamus] load', e); }
}

function _paint() {
  _setBadge(_items.filter(i => i.state === 'open').length);

  const summary = document.getElementById('thal-summary');
  if (summary) summary.innerHTML = _renderSummary(_items);
  const count = document.getElementById('thalamus-count');
  if (count) count.textContent = _items.length + ' item' + (_items.length === 1 ? '' : 's');

  const feed = document.getElementById('feed-thalamus');
  if (!feed) return;
  if (!_items.length) {
    feed.innerHTML = '<div class="feed-empty">'
      + 'The queue is empty — the brain has nothing standing for its streams right now.'
      + '</div>';
    return;
  }
  feed.innerHTML = _items.map(_renderItem).join('');
}

// Click a card → expand its delivery ledger. Click a ref chip → open that node.
function _onFeedClick(e) {
  const ref = e.target.closest('.thal-ref');
  if (ref) {
    e.stopPropagation();
    loadNodeDetail(ref.getAttribute('data-ref'));
    return;
  }
  const card = e.target.closest('.thal-card');
  if (!card) return;
  const tid = card.getAttribute('data-tid');
  if (!tid) return;
  if (_open.has(tid)) _open.delete(tid);
  else _open.add(tid);
  _paint();
}

// ── Lifecycle ──────────────────────────────────────────────────────────
export function init() {
  const feed = document.getElementById('feed-thalamus');
  if (feed && !feed._thalClickBound) {
    feed.addEventListener('click', _onFeedClick);
    feed._thalClickBound = true;
  }

  poll.register({
    key: 'thalamus-feed',
    interval: 10000,
    activeWhen: () => {
      const t = document.getElementById('tab-thalamus');
      return t && t.classList.contains('active');
    },
    fetcher: loadThalamus,
  });

  // Queue depth on the tab, whether or not the tab is open — an unanswered ask
  // the operator never navigates to is exactly the one worth a badge. Always-on
  // (poll.js still pauses on a hidden window).
  poll.register({
    key: 'thalamus-badge',
    interval: 30000,
    fetcher: async () => {
      try {
        const body = await api.thalamus({ closed: 0, limit: 100 });
        _setBadge(((body && body.items) || []).length);
      } catch (_) { /* badge is advisory — a failed poll leaves the last count */ }
    },
  });
}

export function activate() { loadThalamus(); }
export function deactivate() {}
