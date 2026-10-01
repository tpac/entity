// ===========================================================================
// lib/trace_friendly.js — the "Friendly" view: a jargon-free activity digest.
// ---------------------------------------------------------------------------
// Renders one trace CHAIN as a single plain-language "what the brain did"
// card — for someone who's never heard of Anchor, traces, scales, or O/K/Δ.
// No ids, no tokens, no event-type letters. The Technical view (trace_detail
// + traces.js) keeps all of that; this is its friendly sibling.
//
// Pure presentation over the SAME trace data the Technical view reads — no
// separate pipe, no fetch. (Brain principles: "Dashboard inspects existing
// data, never changes core behavior to serve display" / "Dashboard must show
// the same data Claude sees — no disconnected internal pipes.")
//
// Clickable memory titles reuse the `.trace-nodeid` chip + the delegated
// handler in traces.js, so a newcomer can click a remembered memory to read it.
// ===========================================================================

import { escapeHtml, relativeTime, localTime } from '/static/lib/dom.js';
import { parseTraceMeta } from '/static/lib/trace_detail.js';

// ── chain → kind ────────────────────────────────────────────────────────────
// Chain ids stay prefixed (s0-/s1r-/s1e-/s2-<op>) even though event ids are
// hex — the prefix is what tells us which kind of work the chain represents.
function _chainKind(chainId) {
  // Revise chains are `{scale}-{YYYYMMDD}-revise` for ANY scale (s0 for direct
  // operator revises, s1/s2 for encoder/unit revises — dispatch_write.py).
  // Detect by the op SUFFIX before the scale-prefix routing, else an
  // `s0-…-revise` is caught by the `s0-` branch and mislabeled 'conversation'.
  if (chainId.endsWith('-revise')) return 'revise';
  if (chainId.startsWith('archive-') || chainId.endsWith('-archive')) return 'archive';
  if (chainId.startsWith('s1r-')) return 'remember';
  if (chainId.startsWith('s1e-')) return 'learn';
  if (chainId.startsWith('s2-')) {
    const op = chainId.split('-').slice(2).join('-');
    if (op.includes('consolid')) return 's2_merge';
    if (op.includes('community')) return 's2_group';
    if (op.includes('healer'))    return 's2_heal';
    if (op.includes('aspect'))    return 's2_aspect';
    return 's2_other';
  }
  if (chainId.startsWith('s0-')) return 'conversation';
  return 'other';
}

const _STYLE = {
  remember:     { icon: '🧠', accent: '#45B7D1' },
  learn:        { icon: '💡', accent: '#33d17a' },
  s2_merge:     { icon: '🔗', accent: '#c08cff' },
  s2_group:     { icon: '🗂️', accent: '#c08cff' },
  s2_heal:      { icon: '🩹', accent: '#ffaa55' },
  s2_aspect:    { icon: '🏷️', accent: '#c08cff' },
  s2_other:     { icon: '✨', accent: '#c08cff' },
  revise:       { icon: '✏️', accent: '#33d17a' },
  archive:      { icon: '🗑️', accent: '#7a7a86' },
  conversation: { icon: '💬', accent: '#33384a' },
  other:        { icon: '•',  accent: '#555' },
};

// ── metadata extractors over a chain's events ───────────────────────────────
function _deltaMeta(events) {
  for (const ev of events) {
    const m = parseTraceMeta(ev.metadata);
    if ('write_actions' in m || 'created' in m || 'journal_entry' in m) return m;
  }
  return null;
}

// Recall: prefer the event whose `selected` carries "id|title" (the
// surface_selected pick) over an id-only array (additionalContext), no matter
// which is timestamped first — the titled one is what lets us show AND link
// the actual memories that surfaced.
function _selection(events) {
  let titled = null, anyArr = null, count = 0;
  for (const ev of events) {
    const m = parseTraceMeta(ev.metadata);
    if (Array.isArray(m.selected) && m.selected.length) {
      count = Math.max(count, m.selected.length);
      if (!anyArr) anyArr = m.selected;
      if (!titled && m.selected.some(s => typeof s === 'string' && s.includes('|'))) titled = m.selected;
    }
  }
  const best = titled || anyArr || [];
  return { selected: best, count: best.length || count };
}

function _plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }

// ── per-kind copy ────────────────────────────────────────────────────────────
function _remember(events) {
  const { selected, count } = _selection(events);
  const mems = selected.map(s => {
    const str = String(s);
    const bar = str.indexOf('|');
    return bar >= 0 ? { id: str.slice(0, bar), title: str.slice(bar + 1) } : { id: str, title: '' };
  }).filter(m => m.title);
  return {
    title: count ? 'Remembered ' + _plural(count, 'memory', 'memories') : 'Looked for relevant memories',
    line: count ? 'Brought up what’s relevant to the latest message.'
                : 'Nothing relevant surfaced this time.',
    memories: mems,
  };
}

function _learn(events) {
  const m = _deltaMeta(events) || {};
  const c = (m.created || []).length, r = (m.revised || []).length,
        a = (m.archived || []).length;
  let title;
  if (c && r)      title = 'Learned ' + _plural(c, 'new thing') + ' and refined ' + r;
  else if (c)      title = 'Learned ' + _plural(c, 'new thing');
  else if (r)      title = 'Refined ' + _plural(r, 'memory', 'memories');
  else if (a)      title = 'Cleaned up ' + _plural(a, 'memory', 'memories');   // archive-only run
  else             title = 'Reviewed the conversation';
  const bits = [];
  if (c) bits.push(_plural(c, 'new memory', 'new memories'));
  if (r) bits.push(r + ' updated');
  if (a) bits.push(a + ' archived');
  return {
    title,
    line: bits.length ? 'Saved what mattered — ' + bits.join(', ') + '.'
                      : 'Looked back at the conversation; nothing new to save.',
  };
}

function _s2(kind, events) {
  const m = _deltaMeta(events) || {};
  const c = (m.created || []).length, r = (m.revised || []).length, a = (m.archived || []).length;
  if (kind === 's2_merge') return {
    title: a || r ? 'Tidied up similar memories' : 'Looked for memories to tidy',
    line: a ? 'Merged ' + _plural(a, 'duplicate') + (r ? ' and linked related ideas' : '') + '.'
            : (r ? 'Linked ' + _plural(r, 'related memory', 'related memories') + '.'
                 : 'Reviewed memories for overlap while you were away.'),
  };
  if (kind === 's2_group') return {
    title: c ? 'Grouped memories into ' + _plural(c, 'theme') : 'Looked for themes among memories',
    line: 'Found clusters of related memories and named the themes.',
  };
  if (kind === 's2_heal') {
    const healed = m.nodes_healed != null ? m.nodes_healed : r;
    return {
      title: healed ? 'Filled in missing details' : 'Checked memories for gaps',
      line: healed ? 'Added missing context to ' + _plural(healed, 'memory', 'memories') + '.'
                   : 'Looked for memories missing context.',
    };
  }
  if (kind === 's2_aspect') return {
    title: 'Organized memory categories',
    line: 'Sorted new kinds of memory into the right buckets.',
  };
  return { title: 'Reflected on its memories', line: 'Reorganized and connected memories in the background.' };
}

function _revise(events) {
  // node_revised and edge_relation_revised share one `{scale}-{date}-revise`
  // chain. Count node revises as memories; edge-relation revises are links,
  // not memories — tally them separately so the headline isn't inflated.
  const memIds = new Set();
  let edges = 0;
  for (const ev of events) {
    if (ev.ref_type === 'edge_relation_revised') { edges++; continue; }
    const id = ev.ref_id || parseTraceMeta(ev.metadata).node_id;
    if (id) memIds.add(id);
  }
  const n = memIds.size;
  if (n) return {
    title: 'Refined ' + _plural(n, 'memory', 'memories'),
    line: 'Updated details on memories it already had.',
  };
  if (edges) return {
    title: 'Updated ' + _plural(edges, 'connection'),
    line: 'Adjusted links between memories.',
  };
  return { title: 'Refined memories', line: 'Updated details on memories it already had.' };
}

function _archive(events) {
  const n = events.length;
  return {
    title: 'Cleaned up ' + _plural(n, 'memory', 'memories'),
    line: 'Let go of memories that were no longer needed.',
  };
}

function _conversation(events) {
  let msgs = 0, tools = 0;
  for (const ev of events) {
    if (ev.ref_type === 'assistant_message' || ev.ref_type === 'user_message') msgs++;
    if (ev.ref_type === 'tool_result') tools++;
  }
  const parts = [];
  if (msgs)  parts.push(_plural(msgs, 'message'));
  if (tools) parts.push(_plural(tools, 'action'));
  return { line: parts.length ? parts.join(' · ') : 'A turn in the conversation.' };
}

// ── render ────────────────────────────────────────────────────────────────
function _memChip(m) {
  const id = escapeHtml(m.id), title = escapeHtml(m.title);
  return '<span class="trace-nodeid" data-nodeid="' + id + '" title="Open this memory" '
    + 'style="display:inline-block;cursor:pointer;color:#bfe3ff;background:#10202e;'
    + 'border:1px solid #1d3346;border-radius:10px;padding:2px 9px;margin:3px 5px 0 0;'
    + 'font-size:11px;max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
    + 'vertical-align:bottom">' + title + '</span>';
}

/** Render one chain as a friendly activity card (HTML string). `events` is
 *  pre-sorted oldest→newest by the caller; the card timestamps off the
 *  latest event. */
export function renderFriendlyChain(chainId, events) {
  const kind = _chainKind(chainId);
  const st = _STYLE[kind] || _STYLE.other;
  const last = events[events.length - 1] || {};
  const when = last.created_at || (events[0] || {}).created_at || '';

  // S0 conversation: a slim, dimmed line — present for rhythm, but the
  // brain's *actions* (remember/learn/organize) are the story.
  if (kind === 'conversation') {
    const c = _conversation(events);
    return '<div class="brain-card brain-card--quiet" style="display:flex;gap:10px;align-items:center;'
      + 'padding:5px 14px;margin:3px 0;opacity:.62">'
      + '<span style="font-size:13px;width:22px;text-align:center">' + st.icon + '</span>'
      + '<span style="color:#8a8a9a;font-size:12px;flex:1">Conversation · ' + escapeHtml(c.line) + '</span>'
      + '<span style="color:#556;font-size:10px" title="' + escapeHtml(localTime(when)) + '">' + escapeHtml(relativeTime(when)) + '</span>'
      + '</div>';
  }

  let body;
  if (kind === 'remember')      body = _remember(events);
  else if (kind === 'learn')    body = _learn(events);
  else if (kind === 'revise')   body = _revise(events);
  else if (kind === 'archive')  body = _archive(events);
  else if (kind.startsWith('s2_')) body = _s2(kind, events);
  else                          body = { title: 'Worked on its memories', line: 'Did some background work on its memories.' };

  let detail = '';
  if (kind === 'remember' && body.memories && body.memories.length) {
    const shown = body.memories.slice(0, 6).map(_memChip).join('');
    const more = body.memories.length > 6
      ? '<span style="color:#667;font-size:11px;margin-left:4px">+ ' + (body.memories.length - 6) + ' more</span>' : '';
    detail = '<div style="margin-top:6px">' + shown + more + '</div>';
  }

  return '<div class="brain-card" style="display:flex;gap:12px;padding:12px 14px;margin:8px 0;'
    + 'background:#0d0d16;border-radius:10px;border-left:3px solid ' + st.accent + '">'
    + '<div style="font-size:21px;line-height:1.1;width:26px;text-align:center;flex-shrink:0">' + st.icon + '</div>'
    + '<div style="flex:1;min-width:0">'
    +   '<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">'
    +     '<span style="color:#e8e8f2;font-size:14px;font-weight:600">' + escapeHtml(body.title) + '</span>'
    +     '<span style="color:#667;font-size:11px;flex-shrink:0;white-space:nowrap" title="' + escapeHtml(localTime(when)) + '">' + escapeHtml(relativeTime(when)) + '</span>'
    +   '</div>'
    +   (body.line ? '<div style="color:#9a9aab;font-size:12px;margin-top:3px">' + escapeHtml(body.line) + '</div>' : '')
    +   detail
    + '</div>'
    + '</div>';
}
