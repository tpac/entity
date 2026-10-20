// ===========================================================================
// tabs/config.js — the brain's run conditions, and the cross-tab pause banner.
// ---------------------------------------------------------------------------
// One payload (/api/config) feeds both: the daemon's ping (which brain it
// writes), the gate's status (pause, ignored projects, a verdict for each live
// session's project) and the dashboard's own view of the same paths. The one
// rule rendered here — the include-beats-exclude matcher — runs in the daemon,
// so the screen can only say what the brain itself would decide.
//
// The poll is always-on (like the Thalamus badge): a paused brain must show on
// every tab, not only on this one. The tab renders from the same payload when
// it is open.
//
// Two writes, both the daemon's gate command through the server: pause and
// resume. The ignore list is shown, not edited — that stays with the `gate`
// MCP tool, so a page cannot rewrite the operator's project filter.
// ===========================================================================

import { api } from '/static/lib/api.js';
import { poll } from '/static/lib/poll.js';
import { el, formatDuration } from '/static/lib/dom.js';
import { liveSessions, sessionChip } from '/static/lib/sessions.js';

let _cfg = null;      // the last /api/config payload
let _busy = false;    // a pause/resume round trip in flight
let _notice = '';     // the last write's outcome, shown under the switch

function _active() {
  const t = document.getElementById('tab-config');
  return t && t.classList.contains('active');
}

// ── Fetch ───────────────────────────────────────────────────────────────
async function refresh() {
  // The gate answers a verdict per project; ask for the live sessions' ones.
  const projects = [...new Set(liveSessions().map(s => s.project || '').filter(Boolean))];
  try {
    _cfg = await api.config({ projects: projects.join(',') });
  } catch (e) {
    console.error('[config] fetch failed:', e);
    return;
  }
  _renderBanner();
  if (_active()) render();
}

async function _setPaused(on) {
  if (_busy) return;
  if (on && !window.confirm(
      'Pause the brain?\n\nEvery hook in every session on this machine goes silent '
      + 'from its next call — no boot context, no recall, no traces, no encoding — '
      + 'until you resume. Explicit brain tools keep working.')) return;
  _busy = true;
  _notice = '';
  try {
    await api.gate({ action: on ? 'pause' : 'resume' });
    _notice = on ? 'Paused.' : 'Resumed.';
  } catch (e) {
    _notice = 'Failed: ' + (e && e.message ? e.message : e);
  }
  _busy = false;
  await poll.fireNow('config');
}

// ── Pieces ──────────────────────────────────────────────────────────────
function card(title, wide, ...body) {
  return el('div', { class: wide ? 'cfg-card cfg-card--wide' : 'cfg-card' },
    el('div', { class: 'cfg-card-title' }, title), ...body);
}
function row(key, val, mono) {
  return el('div', { class: 'cfg-row' },
    el('span', { class: 'cfg-key' }, key),
    el('span', { class: mono ? 'cfg-val cfg-val--mono' : 'cfg-val' }, ...(Array.isArray(val) ? val : [val])));
}
function pill(text, kind) {
  return el('span', { class: 'cfg-pill cfg-pill--' + kind }, text);
}
function note(text) {
  return el('div', { class: 'cfg-note' }, text);
}

const PAUSED_TEXT = 'Paused: every hook in every session on this machine does nothing '
  + 'from its next call — no boot context, no recall or surface, no traces, no '
  + 'encoding. Explicit brain tools keep working, so a session can resume from '
  + 'inside. Nothing already recorded changes.';

function _locationCard() {
  const d = _cfg.daemon, me = _cfg.dashboard;
  const body = [];
  if (d.alive) {
    body.push(row('Daemon writes', d.db_dir || '(unknown)', true));
    body.push(row('Dashboard reads', me.db_dir, true));
    if (_cfg.db_dir_matches) {
      body.push(row('', pill('same brain', 'ok')));
    } else {
      body.push(el('div', { class: 'cfg-warn' },
        'The daemon and this dashboard point at different brains — what you see '
        + 'here is not what the brain is writing. Check BRAIN_DB_DIR in the config '
        + 'file and the dashboard service (ensure-dashboard.sh re-renders its plist).'));
    }
    body.push(row('Daemon code', (d.source_dir || '?') + ' · ' + (d.code_fingerprint || '?')
      + ' · pid ' + d.pid + ' · up ' + formatDuration(d.uptime), true));
  } else {
    body.push(row('Dashboard reads', me.db_dir, true));
    body.push(note('Daemon unreachable — which brain it writes is unknown until it is back.'));
  }
  body.push(row('Config file', me.env_file, true));
  if (_cfg.gate && _cfg.gate.env_file && _cfg.gate.env_file !== me.env_file) {
    body.push(el('div', { class: 'cfg-warn' },
      'The daemon reads a different config file: ' + _cfg.gate.env_file));
  }
  return card('Brain location', false, ...body);
}

function _pauseCard() {
  const g = _cfg.gate;
  if (!g) {
    return card('Brain activity', false,
      note('Daemon unreachable — the pause state is unknown here. Hooks read the '
         + 'switch from the config file directly, so a pause still holds.'));
  }
  const paused = !!g.paused;
  return card('Brain activity', false,
    row('State', pill(paused ? 'PAUSED' : 'RUNNING', paused ? 'warn' : 'ok')),
    el('div', { class: 'cfg-row' },
      el('button', {
        class: paused ? 'cfg-btn' : 'cfg-btn cfg-btn--danger',
        disabled: _busy,
        onclick: () => _setPaused(!paused),
      }, paused ? 'Resume brain activity' : 'Pause all brain activity'),
      _notice ? el('span', { class: 'cfg-note' }, _notice) : null),
    note(PAUSED_TEXT),
    note('Switch: BRAIN_PAUSED in the config file, written by the daemon\'s gate '
       + 'command — the same one the `gate` MCP tool uses.'));
}

// A session on the live roster has reached the daemon through a hook the gate
// let run, so it is recording. The gate's verdict for its project only adds
// the one case presence cannot see: a pattern (or a pause) that now matches
// it, which silences it from its NEXT hook.
function _verdictPill(project) {
  const g = _cfg.gate;
  const v = g && g.verdicts ? g.verdicts[project || ''] : null;
  if (!v || v.run) return pill('recording', 'ok');
  return v.reason === 'paused'
    ? pill('paused from next hook', 'warn')
    : pill('ignored from next hook', 'bad');
}

function _filtersCard() {
  const g = _cfg.gate;
  const body = [];
  if (!g) {
    body.push(note('Daemon unreachable — the project filter is unknown here.'));
  } else {
    const pats = g.ignore_projects || [];
    const excludes = pats.filter(p => !p.startsWith('!'));
    const includes = pats.filter(p => p.startsWith('!')).map(p => p.slice(1));
    if (!pats.length) {
      body.push(row('Projects', pill('every project records', 'ok')));
    } else {
      body.push(row('Ignored', excludes.length ? excludes.map(p => pill(p, 'bad')) : pill('none', 'muted')));
      body.push(row('Always recorded', includes.length ? includes.map(p => pill(p, 'ok')) : pill('none', 'muted')));
    }
    body.push(note('BRAIN_IGNORE_PROJECTS: fnmatch patterns on a session\'s derived project '
      + '(its repo folder name, shared by every worktree). An include (`!pattern`) beats '
      + 'any match — `*,!brain` means only brain records. Edit with the `gate` MCP tool '
      + '(ignore / unignore).'));
  }

  const live = liveSessions();
  const table = el('table', { class: 'cfg-table' },
    el('thead', null, el('tr', null,
      el('th', null, 'Live session'), el('th', null, 'Project'), el('th', null, 'Verdict'))),
    el('tbody', null, ...live.map(s => el('tr', null,
      el('td', null, sessionChip(s.id)),
      el('td', null, s.project || '(none derived)'),
      el('td', null, _verdictPill(s.project))))));
  body.push(live.length ? table : note('No live sessions right now.'));
  body.push(note('A session the gate refused at its first hook never reaches the daemon, so it '
    + 'cannot appear here — its boot prints one line saying the brain is not recording it. '
    + 'A pattern added while a session is live shows here as ignored from its next hook.'));
  return card('What the brain does not record', true, ...body);
}

function _keyCard() {
  const present = !!_cfg.api_key_present;
  return card('Anthropic API key', false,
    row('ANTHROPIC_API_KEY', [
      pill(present ? 'present' : 'missing', present ? 'ok' : 'bad'),
      el('a', { href: '/setup', class: 'cfg-link' }, present ? 'replace it' : 'add it'),
    ]),
    note('Learning and surfacing use the Anthropic API; storage, traces and direct recall '
       + 'work without it. The /setup page hands the key to the daemon, which writes the '
       + 'config file.'));
}

function render() {
  const root = document.getElementById('config-root');
  if (!root) return;
  if (!_cfg) {
    root.replaceChildren(el('div', { class: 'feed-empty' }, 'Loading…'));
    return;
  }
  root.replaceChildren(_locationCard(), _pauseCard(), _filtersCard(), _keyCard());
}

// ── Banner + badge — every tab ──────────────────────────────────────────
function _renderBanner() {
  const banner = document.getElementById('gate-banner');
  const badge = document.getElementById('config-badge');
  const g = _cfg && _cfg.gate;
  const paused = !!(g && g.paused);
  const ignoring = (g && g.ignore_projects || []).filter(p => !p.startsWith('!')).length;
  if (badge) {
    // ⏸ while paused; otherwise the number of ignore patterns, so a filter
    // set months ago stays visible in the tab bar instead of being forgotten.
    badge.textContent = paused ? '⏸' : (ignoring ? String(ignoring) : '');
    badge.classList.toggle('show', paused || ignoring > 0);
  }
  if (!banner) return;
  if (!paused) { banner.replaceChildren(); return; }
  banner.replaceChildren(el('div', { class: 'gate-banner' },
    el('span', null, '⏸ Brain paused — every hook in every session is silent from its next call; explicit brain tools keep working.'),
    el('button', { class: 'cfg-btn gate-banner-btn', disabled: _busy, onclick: () => _setPaused(false) }, 'Resume'),
    el('a', { href: '#', class: 'cfg-link', onclick: (e) => { e.preventDefault(); window.switchTab('config'); } }, 'Config')));
}

// ── Lifecycle ───────────────────────────────────────────────────────────
export function init() {
  // Always-on: the pause banner belongs to every tab (poll.js still pauses
  // on a hidden window).
  poll.register({ key: 'config', interval: 10000, fetcher: refresh });
}

export function activate() {
  render();
  poll.fireNow('config');
}

export function deactivate() {}
