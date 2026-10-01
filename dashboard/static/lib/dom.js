// ===========================================================================
// lib/dom.js — DOM construction helpers (escaped by default).
// ---------------------------------------------------------------------------
// Why: string-built innerHTML needs an escapeHtml() per interpolation (one
// miss = XSS); el() escapes by default.
//
// The core helper is `el(tag, attrs, ...children)`. Strings in `children`
// are auto-escaped. To insert raw HTML, use the `html()` marker.
//
//   el('div', { class: 'card card--s1' },
//     el('span', { class: 'badge badge--blue' }, 'HOOK'),
//     ' ',
//     evt.title,           // auto-escaped
//   )
//
// Returns an HTMLElement. Mount with parent.replaceChildren(el) or
// parent.appendChild(el) — no innerHTML assignment, no string concatenation.
//
// Why not just `template` elements + cloneNode? That's a fine pattern for
// fixed shapes, but the dashboard renders open-ended dynamic content (the
// activity feed varies per row type). `el()` is more compositional.
// ===========================================================================

const _HTML_MARKER = Symbol('html');

/** Mark a string as raw HTML so el() inserts it unescaped. Use sparingly —
 * always prefer composing with el() so escaping is automatic. */
export function html(str) {
  return { [_HTML_MARKER]: true, value: String(str) };
}

export function escapeHtml(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Create an element. Strings in children are escaped; pass `html(...)` for
 * raw. Attrs:
 *   { class: 'foo bar' }                    → class attribute
 *   { onclick: () => ... }                  → addEventListener('click', ...)
 *   { dataset: { foo: 'bar' } }             → data-foo="bar"
 *   { style: { color: 'red', padding:'4px'}} → inline style (avoid; prefer class)
 *   { 'aria-label': '...' }                 → arbitrary attribute */
export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') {
        node.className = Array.isArray(v) ? v.filter(Boolean).join(' ') : String(v);
      } else if (k === 'dataset') {
        for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
      } else if (k === 'style' && typeof v === 'object') {
        for (const [sk, sv] of Object.entries(v)) node.style[sk] = sv;
      } else if (k.startsWith('on') && typeof v === 'function') {
        node.addEventListener(k.slice(2).toLowerCase(), v);
      } else {
        node.setAttribute(k, v);
      }
    }
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    if (child instanceof Node) {
      node.appendChild(child);
    } else if (child && child[_HTML_MARKER]) {
      // Raw HTML — must trust the caller. Use sparingly.
      const tmp = document.createElement('div');
      tmp.innerHTML = child.value;
      while (tmp.firstChild) node.appendChild(tmp.firstChild);
    } else {
      node.appendChild(document.createTextNode(String(child)));
    }
  }
  return node;
}

// Seconds elapsed between now and a server ISO timestamp — positive in the
// past, negative in the future. NaN for anything unparseable.
function _secsSince(utcStr) {
  let s = utcStr;
  if (s.length >= 19 && !s.endsWith('Z') && !s.includes('+')) s += 'Z';
  const d = new Date(s);
  return isNaN(d) ? NaN : (Date.now() - d.getTime()) / 1000;
}

// A span's magnitude → '3s' / '2m' / '5h' / '9d' / '4mo' / '2y'. Shared by
// relativeTime and untilTime so past and future never drift into different
// tiers.
function _span(secs) {
  if (secs < 60)    return Math.round(secs) + 's';
  const mins = secs / 60;
  if (mins < 60)    return Math.round(mins) + 'm';
  const hours = mins / 60;
  if (hours < 24)   return Math.round(hours) + 'h';
  const days = hours / 24;
  if (days < 30)    return Math.round(days) + 'd';
  const months = days / 30;
  if (months < 12)  return Math.round(months) + 'mo';
  return Math.round(days / 365) + 'y';
}

/** Short relative-time helper — '3m ago', '2h ago', '5d ago'. Pairs with
 * localTime() in tooltip-style UI: show 'created 3h ago' inline with the
 * absolute timestamp in title= or as a sibling. A future timestamp clamps to
 * '0s ago' — use untilTime() for deadlines. */
export function relativeTime(utcStr) {
  if (!utcStr) return '';
  const secs = _secsSince(utcStr);
  if (isNaN(secs)) return '';
  return _span(Math.max(0, secs)) + ' ago';
}

/** The FUTURE counterpart — 'in 2h', 'in 14d'; a timestamp already past reads
 * '3h overdue'. For deadlines (a Thalamus item's due date or expiry), where
 * relativeTime's past-only clamp would print a live deadline as '0s ago'. */
export function untilTime(utcStr) {
  if (!utcStr) return '';
  const secs = _secsSince(utcStr);
  if (isNaN(secs)) return '';
  return secs > 0 ? _span(secs) + ' overdue' : 'in ' + _span(-secs);
}

/** Elapsed seconds → 'Nh Nm' ('Nm' under an hour, 'Ns' under a minute). For
 * spans the server reports as a NUMBER (daemon uptime) rather than as the
 * timestamp relativeTime() takes. No day tier on purpose — uptime past a day
 * reads fine as hours, and one unit less to scan. */
export function formatDuration(secs) {
  const s = Math.max(0, Math.floor(Number(secs) || 0));
  if (s < 60) return s + 's';
  const mins = Math.floor(s / 60);
  if (mins < 60) return mins + 'm';
  return Math.floor(mins / 60) + 'h ' + (mins % 60) + 'm';
}

/** Convert a server ISO timestamp to a human-readable local time. */
export function localTime(utcStr, mode) {
  if (!utcStr) return '';
  let s = utcStr;
  if (s.length >= 19 && !s.endsWith('Z') && !s.includes('+')) s += 'Z';
  const d = new Date(s);
  if (isNaN(d)) return utcStr;
  if (mode === 'time') {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  return d.toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

/** Speaker→responder identity chip as an HTML string, for the traces.js chain
 * header that still builds innerHTML. Returns '' when neither side is set, so
 * it drops into a template-literal slot without a guard. */
export function identityChipHTML(human, agent) {
  if (!human && !agent) return '';
  const h = human ? escapeHtml(human) : '?';
  const a = agent ? escapeHtml(agent) : '?';
  return '<span class="identity-chip" title="speaker → responder">' +
    h + '<span style="color:#555;margin:0 3px">→</span>' + a + '</span>';
}

/** The S0 session stamp as a chip: which model produced the turn, with the
 *  host runtime in the hover. Empty string when the row predates the stamp. */
export function modelChipHTML(model, host) {
  if (!model) return '';
  const title = 'model' + (host ? ' · host: ' + escapeHtml(host) : '');
  return '<span class="chip chip--model" title="' + title + '">' + escapeHtml(model) + '</span>';
}

/** Convenience: clear a node and append a list of children atomically.
 * Avoids the half-rendered flash that `node.innerHTML = ''` + repeated
 * appends causes. */
export function mount(parent, ...children) {
  parent.replaceChildren(...children.flat().filter(c => c != null));
}
