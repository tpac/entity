// ===========================================================================
// lib/trace_detail.js — technical-density rendering of one trace event's
// metadata payload.
// ---------------------------------------------------------------------------
// The Traces tab shows a one-line summary per event. This expands a single
// event into the full O/K/Δ record the trace actually carries: token cost,
// action breakdown, op-attributed node-id lists, field-level diffs, the
// encoder's journal, raw agent text, and errors — i.e. everything
// build_delta_metadata / build_revise_metadata / build_selection_metadata
// (servers/trace_contract.py) stamps.
//
// Dispatch is by PAYLOAD SHAPE, detected by key presence rather than
// ref_type — the ref_type vocabulary is open (50+ values, encoder-extensible)
// but the metadata builders are a closed set of four shapes. Key-detection
// stays correct when a new ref_type appears.
//
// Returns an HTML string (matches tabs/traces.js's string-concat idiom).
// Node-ids render as `.trace-nodeid[data-nodeid]` chips; tabs/traces.js owns
// the delegated click handler that routes them to loadNodeDetail.
// ===========================================================================

import { escapeHtml } from '/static/lib/dom.js';

/** Parse the raw metadata blob. Returns {} on null/garbage — a trace that
 *  predates a builder, or a bare early-out marker, legitimately has none. */
export function parseTraceMeta(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try { return JSON.parse(raw) || {}; } catch (_) { return {}; }
}

// Shape detection. Order matters: delta is checked first because a delta
// payload can also carry list-typed keys that look generic.
export function traceShape(ev, meta) {
  if ('write_actions' in meta || 'final_text' in meta ||
      'action_details' in meta || 'journal_entry' in meta) return 'delta';
  if (Array.isArray(meta.deltas) && ('node_id' in meta || 'edge_id' in meta)) return 'revise';
  // A selection payload carries `selected` even when `content` is empty/absent
  // (a turn that surfaced nothing). Detect on `selected` alone so it doesn't
  // fall through to the raw-JSON generic renderer. (delta/revise are ruled out
  // above and carry no `selected`, so this can't steal them.)
  if ('candidates_considered' in meta || 'outcomes_per_candidate' in meta ||
      Array.isArray(meta.selected)) return 'selection';
  return 'generic';
}

// ── presentation atoms ─────────────────────────────────────────────────────

// Plain-English definitions, shown on hover. Keyed by the chip/section label
// so _stat / _section attach them automatically. "What all the params mean."
const TIPS = {
  // delta cost & provenance
  'rounds':     'LLM conversation rounds this run took',
  'actions':    'Total tool calls the agent made this run',
  'writes':     'Successful graph writes — create / revise / connect / archive',
  'inputs':     'Items processed this run (clusters / proposals / nodes seen)',
  'rej.skipped':'Proposals skipped via rejection-fingerprint — already decided, not re-proposed',
  'elapsed':    'Wall-clock time to produce this change',
  'tok in':     'Prompt (input) tokens sent to the model',
  'tok out':    'Completion (output) tokens the model generated',
  'cache rd':   'Prompt-cache tokens read — a cache hit (cheap)',
  'cache wr':   'Prompt-cache tokens written — cache creation (one-time cost)',
  'K v':        'Interaction VERSION — which versioned prompt/config produced this Δ (the learnable boundary)',
  'K':          'Run on the code default — no DB override active for this interaction',
  'K#':         'Interaction FINGERPRINT — content-address (12-hex sha256) of the exact prompt+config this run used; stable across installs and across override collapses',
  'K id':       'Interaction row id in this install\'s interactions table — install-local and unenforced (no FK); historic ids can dangle',
  // selection / recall
  'candidates': 'Memories scored as recall candidates before selection',
  'selected':   'Memories chosen to surface this turn — the actual recall',
  'dropped':    'Candidates considered but not surfaced',
  'outcome':    'Did this recall serve context (served) or surface nothing (empty)',
  'truncated':  'Rounds whose response was cut at max_tokens',
  // section headers
  'cost & provenance': 'What this run cost and which prompt version produced it',
  'phase timing': 'Per-phase wall-clock breakdown of the recall (the queryable form of hook_phase_timing)',
  'graph Δ':    'Nodes this run created / revised / archived — the change to memory (edges are their own edge events)',
  'outcomes':   'Unit-specific tally of what the run decided (e.g. merged, kept, consolidated)',
  'output (what was seen)': 'The exact additionalContext block this recall injected',
  'field deltas': 'Per-field old → new values changed by this revise',
};

function _stat(label, val, color, title) {
  const tip = title || TIPS[label] || '';
  return '<span title="' + escapeHtml(tip) + '" style="display:inline-block;background:#15151f;border-radius:3px;'
    + 'padding:1px 6px;margin:0 4px 4px 0;color:#778;font-size:10px;'
    + (tip ? 'cursor:help;' : '')
    + '">' + escapeHtml(label) + ' <b style="color:' + (color || '#bcd') + '">'
    + escapeHtml(String(val)) + '</b></span>';
}

function _section(title, body) {
  if (!body) return '';
  // Strip a trailing "(N)" count so the tip lookup matches the base label.
  const tip = TIPS[title] || TIPS[title.replace(/\s*\(\d+\)\s*$/, '')] || '';
  return '<div title="' + escapeHtml(tip) + '" style="color:#667;font-size:9px;text-transform:uppercase;'
    + 'letter-spacing:.6px;margin:8px 0 3px;' + (tip ? 'cursor:help;' : '')
    + '">' + escapeHtml(title) + '</div>' + body;
}

function _code(text, maxH) {
  return '<pre style="margin:3px 0;padding:6px 8px;background:#0c0c16;'
    + 'border:1px solid #181826;border-radius:4px;color:#bcd;white-space:pre-wrap;'
    + 'word-break:break-word;max-height:' + (maxH || 300) + 'px;overflow:auto;'
    + 'font-family:ui-monospace,SFMono-Regular,monospace;font-size:11px">'
    + escapeHtml(text) + '</pre>';
}

/** Clickable node-id chip. tabs/traces.js's delegated handler reads
 *  data-nodeid and calls loadNodeDetail — no inline onclick, so node-ids
 *  (which can contain arbitrary chars) never get interpolated into a JS
 *  string. */
function _nodeChip(id) {
  const s = escapeHtml(String(id));
  return '<span class="trace-nodeid" data-nodeid="' + s + '" title="Open node ' + s + '" '
    + 'style="display:inline-block;cursor:pointer;color:#7eb8ff;background:#10182a;'
    + 'border:1px solid #1d2c44;border-radius:3px;padding:0 5px;margin:0 4px 4px 0;'
    + 'font-family:ui-monospace,monospace;font-size:10px">' + s + '</span>';
}

function _nodeList(ids) {
  if (!ids || !ids.length) return '';
  return ids.map(_nodeChip).join('');
}

/** Render any value compactly — scalars inline, objects/arrays as JSON. */
function _val(v) {
  if (v == null) return '<span style="color:#556">∅</span>';
  if (typeof v === 'object') {
    try { return _code(JSON.stringify(v, null, 2)); } catch (_) { return escapeHtml(String(v)); }
  }
  return '<span style="color:#cfe;font-family:ui-monospace,monospace;font-size:11px">'
    + escapeHtml(String(v)) + '</span>';
}

function _fmtMs(ms) {
  ms = Number(ms) || 0;
  return ms >= 1000 ? (ms / 1000).toFixed(1) + 's' : ms + 'ms';
}

// Keys the delta renderer surfaces in dedicated lanes — everything else in a
// delta payload is a unit-specific extra (build_delta_metadata's **extras,
// e.g. nodes_healed, fields_written, clusters_processed) and gets its own
// section so the technical view stays complete.
const _DELTA_KNOWN = new Set([
  'actions', 'write_actions', 'rounds', 'inputs_processed', 'outcomes',
  'rejection_skipped', 'journal_entry', 'action_details', 'read_calls',
  'final_text', 'errors', 'created', 'revised', 'archived', 'classifications',
  'elapsed_ms', 'input_tokens', 'output_tokens', 'cache_read_tokens',
  'cache_creation_tokens', 'truncated', 'model', 'interaction_version',
  'interaction_fingerprint', 'interaction_source',
  'human_identity', 'agent_identity',
]);

// ── per-shape renderers ─────────────────────────────────────────────────────

// Shared agent-run cost lane (the build_run_telemetry field-set + K provenance).
// Used by BOTH _renderDelta (encoders) and _renderSelection (Surface), so the
// cost block renders identically wherever an agent spent tokens — the render
// mirror of the one trace_contract.build_run_telemetry that produces the data.
// Returns the chip HTML; the per-shape `truncated` banner stays with each caller
// (a delta truncation is a corrupted write; a selection truncation is a
// cut-short pick — different wording).
function _runCostLane(ev, m) {
  let cost = '';
  if (m.rounds != null)            cost += _stat('rounds', m.rounds);
  if (m.elapsed_ms != null)        cost += _stat('elapsed', _fmtMs(m.elapsed_ms));
  if (m.input_tokens != null)      cost += _stat('tok in', m.input_tokens);
  if (m.output_tokens != null)     cost += _stat('tok out', m.output_tokens);
  if (m.cache_read_tokens)         cost += _stat('cache rd', m.cache_read_tokens);
  if (m.cache_creation_tokens)     cost += _stat('cache wr', m.cache_creation_tokens);
  // Which LLM the run called — what makes the token counts priceable.
  if (m.model)                     cost += _stat('model', m.model);
  // K provenance: source decides the label — an override shows its version,
  // a default-run shows 'default' (version 0 would render as nothing).
  // Otherwise (override, or no recorded source) show the version chip.
  if (m.interaction_source === 'default') cost += _stat('K', 'default', '#ffaa33');
  else if (m.interaction_version)  cost += _stat('K v', m.interaction_version, '#ffaa33');
  if (m.interaction_fingerprint)   cost += _stat('K#', m.interaction_fingerprint);
  if (ev.interaction_id)           cost += _stat('K id', ev.interaction_id);   // truthy: hide 0/null, matching K v
  return cost;
}

function _renderDelta(ev, m) {
  let h = '';

  // Cost & provenance — the lane that turns Traces into a hardening cockpit.
  // Delta-specific loop stats first, then the shared run-cost lane.
  let cost = '';
  if (m.actions != null)           cost += _stat('actions', m.actions);
  if (m.write_actions != null)     cost += _stat('writes', m.write_actions, '#33ff88');
  if (m.inputs_processed != null)  cost += _stat('inputs', m.inputs_processed);
  if (m.rejection_skipped)         cost += _stat('rej.skipped', m.rejection_skipped, '#ffaa33');
  cost += _runCostLane(ev, m);
  if (m.truncated) {
    cost += '<span style="display:inline-block;background:#3a0e0e;border:1px solid #c33;'
      + 'border-radius:3px;padding:1px 6px;margin:0 4px 4px 0;color:#ff7777;font-size:10px;'
      + 'font-weight:bold">⚠ TRUNCATED (write cut mid-tool-call — data loss)</span>';
  }
  h += _section('cost & provenance', cost);

  // Outcomes — unit-specific {action: count} vocab.
  if (m.outcomes && Object.keys(m.outcomes).length) {
    const chips = Object.entries(m.outcomes)
      .map(([k, v]) => _stat(k, v, '#9fd')).join('');
    h += _section('outcomes', chips);
  }

  // Op-attributed node lists — each id navigable to the graph.
  const lanes = [
    ['created', '#33ff88'], ['revised', '#ffaa33'],
    ['archived', '#888'],
  ];
  let nodeBody = '';
  for (const [k, c] of lanes) {
    const ids = m[k];
    if (ids && ids.length) {
      nodeBody += '<div style="margin-bottom:4px"><span style="color:' + c
        + ';font-size:10px;margin-right:6px">' + k + ' (' + ids.length + ')</span>'
        + _nodeList(ids) + '</div>';
    }
  }
  h += _section('graph Δ', nodeBody);

  if (m.journal_entry) h += _section('journal entry', _code(m.journal_entry));
  if (m.final_text)    h += _section('agent text (raw, ≤2KB)', _code(m.final_text));

  if (m.read_calls && m.read_calls.length) {
    const body = m.read_calls.map(c =>
      typeof c === 'object' ? JSON.stringify(c) : String(c)).join('\n');
    h += _section('read calls (' + m.read_calls.length + ')', _code(body, 160));
  }

  if (m.errors && m.errors.length) {
    const body = m.errors.map(e =>
      typeof e === 'object' ? JSON.stringify(e, null, 2) : String(e)).join('\n');
    h += _section('errors (' + m.errors.length + ')',
      '<div style="border-left:2px solid #c33;padding-left:6px">' + _code(body, 200) + '</div>');
  }

  if (m.action_details && m.action_details.length) {
    h += _section('action details (' + m.action_details.length + ')',
      _code(JSON.stringify(m.action_details, null, 2), 240));
  }

  // Unit-specific extras (healer's nodes_healed/fields_written, etc.).
  const extras = Object.keys(m).filter(k => !_DELTA_KNOWN.has(k));
  if (extras.length) {
    let body = '';
    for (const k of extras) {
      const v = m[k];
      if (v != null && typeof v !== 'object') body += _stat(k, v, '#9fd');
    }
    const objExtras = extras.filter(k => m[k] && typeof m[k] === 'object');
    for (const k of objExtras) body += _section(k, _code(JSON.stringify(m[k], null, 2), 160));
    h += _section('unit extras', body);
  }
  return h;
}

function _renderRevise(ev, m) {
  let h = '';
  let head = '';
  if (m.node_id) head += '<span style="color:#778;font-size:10px;margin-right:4px">node</span>' + _nodeChip(m.node_id);
  if (m.edge_id) head += _stat('edge', m.edge_id) + (m.relation ? _stat('relation', m.relation, '#9fd') : '');
  if (m.encoding_source) head += _stat('by', m.encoding_source);
  h += _section('target', head);
  if (m.reason) h += _section('reason',
    '<div style="color:#cdd;font-size:11px">' + escapeHtml(m.reason) + '</div>');

  // Field-level diffs: the whole point of a revise trace.
  if (m.deltas && m.deltas.length) {
    let rows = '<table style="width:100%;border-collapse:collapse;font-size:11px">'
      + '<tr style="color:#667;font-size:9px;text-transform:uppercase">'
      + '<td style="padding:2px 6px">field</td><td style="padding:2px 6px">old</td>'
      + '<td style="padding:2px 6px">new</td></tr>';
    for (const d of m.deltas) {
      rows += '<tr style="border-top:1px solid #15151f;vertical-align:top">'
        + '<td style="padding:3px 6px;color:#9fd;font-family:ui-monospace,monospace">'
        + escapeHtml(String(d.field)) + '</td>'
        + '<td style="padding:3px 6px;color:#a88">' + _val(d.old) + '</td>'
        + '<td style="padding:3px 6px;color:#8c8">' + _val(d.new) + '</td></tr>';
    }
    rows += '</table>';
    h += _section('field deltas (' + m.deltas.length + ')', rows);
  }
  if (m.warnings && m.warnings.length) {
    const body = m.warnings.map(w => '• ' + (typeof w === 'object' ? JSON.stringify(w) : w)).join('\n');
    h += _section('warnings', '<div style="color:#ffaa33">' + _code(body, 160) + '</div>');
  }
  return h;
}

function _renderSelection(ev, m) {
  let h = '';
  let stats = '';
  if (m.candidates_considered != null) stats += _stat('candidates', m.candidates_considered);
  if (Array.isArray(m.selected)) stats += _stat('selected', m.selected.length, '#33ff88');
  if (Array.isArray(m.dropped))  stats += _stat('dropped', m.dropped.length, '#888');
  if (m.outcome) stats += _stat('outcome', m.outcome,
                                m.outcome === 'served' ? '#33ff88' : '#ffaa33');
  h += _section('selection', stats);

  // Cost & provenance — Surface is an LLM agent too; its K trace carries
  // the shared run-cost block. Same lane as the encoders (only present on the
  // K event, which carries the telemetry; the Δ additionalContext event won't).
  let cost = _runCostLane(ev, m);
  if (m.truncated) cost += _stat('truncated', m.truncated, '#ff7777');
  if (cost) h += _section('cost & provenance', cost);

  // Per-phase latency — the structured form of the hook_phase_timing string.
  if (Array.isArray(m.phase_timing) && m.phase_timing.length) {
    const chips = m.phase_timing
      .map(p => _stat(p.phase, _fmtMs(p.ms))).join('');
    h += _section('phase timing', chips);
  }

  if (m.selected && m.selected.length) {
    h += _section('selected', m.selected.map(s =>
      _stat('', s, '#9fd')).join(''));
  }
  if (m.outcomes_per_candidate && Object.keys(m.outcomes_per_candidate).length) {
    h += _section('outcomes per candidate',
      _code(JSON.stringify(m.outcomes_per_candidate, null, 2), 180));
  }
  if (m.content) h += _section('output (what was seen)', _code(m.content, 360));
  return h;
}

function _renderGeneric(ev, m) {
  const keys = Object.keys(m).filter(k => k !== 'human_identity' && k !== 'agent_identity');
  if (!keys.length) return '<div style="color:#556;font-size:11px">No metadata payload.</div>';
  const clean = {};
  for (const k of keys) clean[k] = m[k];
  return _section('metadata', _code(JSON.stringify(clean, null, 2)));
}

// ── public entry ────────────────────────────────────────────────────────────

/** Render the expanded technical detail for one trace event as an HTML
 *  string. `ev` is a row from /api/traces (carries raw `metadata`). */
export function renderTraceDetail(ev) {
  const m = parseTraceMeta(ev.metadata);
  let h = '';

  // Full (untruncated) summary — the collapsed row caps at 300 chars.
  if (ev.summary && ev.summary.length > 300) {
    h += _section('summary (full)',
      '<div style="color:#cdd;font-size:11px;white-space:pre-wrap;word-break:break-word">'
      + escapeHtml(ev.summary) + '</div>');
  }
  if (ev.ref_id) {
    h += '<div style="color:#667;font-size:10px;margin:2px 0 4px">ref: '
      + '<span style="color:#9ab;font-family:ui-monospace,monospace">'
      + escapeHtml(ev.ref_id) + '</span></div>';
  }

  const shape = traceShape(ev, m);
  if (shape === 'delta')          h += _renderDelta(ev, m);
  else if (shape === 'revise')    h += _renderRevise(ev, m);
  else if (shape === 'selection') h += _renderSelection(ev, m);
  else                            h += _renderGeneric(ev, m);

  return h || '<div style="color:#556;font-size:11px">No detail.</div>';
}

/** Lightweight badges for the COLLAPSED row — data-loss / errors shouldn't
 *  require a click to notice while hardening. Returns '' for clean events.
 *  Takes one event or a RUN of them (a merged group row badges the whole run,
 *  so a single bad write inside a collapsed run of 40 still shows). */
export function collapsedBadges(evOrRun) {
  const evs = Array.isArray(evOrRun) ? evOrRun : [evOrRun];
  let trunc = 0, errors = 0;
  for (const ev of evs) {
    const m = parseTraceMeta(ev.metadata);
    if (m.truncated) trunc++;
    if (m.errors && m.errors.length) errors += m.errors.length;
  }
  let b = '';
  if (trunc) {
    b += '<span title="write cut mid-tool-call — data loss" style="background:#3a0e0e;'
      + 'border:1px solid #c33;border-radius:2px;color:#ff7777;font-size:9px;font-weight:bold;'
      + 'padding:0 4px;margin-left:4px">⚠ TRUNC' + (trunc > 1 ? ' ×' + trunc : '') + '</span>';
  }
  if (errors) {
    b += '<span title="' + escapeHtml(String(errors)) + ' error(s)" style="background:#2a1a00;'
      + 'border:1px solid #c83;border-radius:2px;color:#ffaa33;font-size:9px;'
      + 'padding:0 4px;margin-left:4px">' + errors + ' err</span>';
  }
  return b;
}

// ── merged runs ─────────────────────────────────────────────────────────────
// A write-heavy chain emits ONE event per node / edge / note: an S2 community
// run lands 40+ identical `edge_relation_revised` rows, an encode 6 identical
// `node_created` rows. Collapsed into one row per run, the chain reads as what
// it did instead of a scroll of repeats; the run expands to the same rows,
// each still individually expandable.
//
// A ref_type is mergeable IFF it has a summary here — the map is the registry.
// Everything else (encoding_run, recall, community_enriched, …) is a
// once-per-run event whose summary already carries the story: never merged.
//
// Keyed by ref_type, unlike this module's payload renderers (see the header's
// shape-detection rule) — deliberately. "How many of these does one run emit?"
// is a property of the ref_type, not of the payload: node_created and
// tool_result carry unrelated shapes and both repeat per item. Copy-only, so an
// unlisted ref_type just renders its rows individually, never wrongly.

/** "a ×3, b, c" — descending by count, capped. */
function _tally(values, cap = 4) {
  const counts = new Map();
  for (const v of values) {
    const k = String(v || '?');
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const shown = sorted.slice(0, cap)
    .map(([k, n]) => k + (n > 1 ? ' ×' + n : '')).join(', ');
  const rest = sorted.length - cap;
  return shown + (rest > 0 ? ', +' + rest + ' more' : '');
}

function _metas(evs) { return evs.map(ev => parseTraceMeta(ev.metadata)); }

const _GROUP_SUMMARY = {
  node_created: (evs) => {
    const types = _metas(evs).map(m => m.type).filter(Boolean);
    return evs.length + ' created' + (types.length ? ' — ' + _tally(types) : '');
  },
  node_revised: (evs) => {
    const ms = _metas(evs);
    const fields = ms.flatMap(m => (m.deltas || []).map(d => d.field));
    const nodes = new Set(ms.map(m => m.node_id).filter(Boolean)).size;
    return (nodes || evs.length) + ' revised'
      + (fields.length ? ' — ' + fields.length + ' field changes: ' + _tally(fields) : '');
  },
  edge_relation_revised: (evs) => {
    const rels = _metas(evs).map(m => m.relation).filter(Boolean);
    return evs.length + ' connections' + (rels.length ? ' — ' + _tally(rels) : '');
  },
  node_archived: (evs) => evs.length + ' archived',
  journal_note: (evs) => evs.length + ' journal notes',
  // The heaviest run of all: a working turn lands dozens between two messages.
  // Tallied by tool, so the collapsed row still says what the turn DID.
  tool_result: (evs) => {
    const tools = _metas(evs).map(m => String(m.tool || '')
      .replace(/^mcp__.*__/, '').replace(/^mcp__/, '')).filter(Boolean);
    return evs.length + ' tool calls' + (tools.length ? ' — ' + _tally(tools, 6) : '');
  },
};

/** Can a run of this ref_type collapse into one row? */
export function isMergeableRefType(refType) {
  return Object.prototype.hasOwnProperty.call(_GROUP_SUMMARY, refType);
}

/** One-line headline for a collapsed run. Only reached for a ref_type
 *  isMergeableRefType accepted, so the lookup always hits. */
export function groupSummary(run) {
  return _GROUP_SUMMARY[run[0].ref_type](run);
}
