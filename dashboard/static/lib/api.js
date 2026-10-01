// ===========================================================================
// lib/api.js — single fetch wrapper.
// ---------------------------------------------------------------------------
// GET with de-duplication (two panels asking for /api/X at the same tick
// coalesce into one network hit) plus POST. Both return the raw response
// body; each caller reads its endpoint's own shape (`.events`, `.runs`, …).
// ===========================================================================

const _inflight = new Map();   // url → in-flight Promise

/** Low-level GET. Returns the parsed JSON body, or throws on network/HTTP/
 * parse failure. Concurrent identical URLs coalesce. */
export async function get(url) {
  if (_inflight.has(url)) return _inflight.get(url);
  const p = (async () => {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error('HTTP ' + r.status + ' on ' + url);
      return await r.json();
    } finally {
      _inflight.delete(url);
    }
  })();
  _inflight.set(url, p);
  return p;
}

/** Low-level POST with a JSON body. Returns the parsed JSON body; throws on
 * network/HTTP/parse failure. NOT coalesced (writes aren't idempotent). This
 * is the dashboard's only write path — it reaches the daemon via the server,
 * never the DB directly. */
export async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  let parsed = null;
  try { parsed = await r.json(); } catch (_) { /* leave null */ }
  if (!r.ok) {
    const msg = (parsed && parsed.error) ? parsed.error : ('HTTP ' + r.status);
    throw new Error(msg);
  }
  return parsed;
}

// Query-string builder — skip empty values so URLs stay clean.
function _qs(params) {
  if (!params) return '';
  const parts = Object.entries(params)
    .filter(([_, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(v));
  return parts.length ? '?' + parts.join('&') : '';
}

/** Named endpoint shortcuts. Each returns the raw body. */
export const api = {
  stats:               ()       => get('/api/stats'),
  systemStatus:        ()       => get('/api/system-status'),
  insights:            ()       => get('/api/insights'),
  insightsLive:        ()       => get('/api/insights/live'),
  aspects:             ()       => get('/api/aspects'),
  sessions:            ()       => get('/api/sessions'),
  recalls:             (p = {}) => get('/api/recalls' + _qs(p)),
  recallPrompt:        (p = {}) => get('/api/recall-prompt' + _qs(p)),
  encodingRuns:        (p = {}) => get('/api/encoding-runs' + _qs(p)),
  encodingPrompt:      (p = {}) => get('/api/encoding-prompt' + _qs(p)),
  consolidationRuns:   (p = {}) => get('/api/consolidation-runs' + _qs(p)),
  communityRuns:       (p = {}) => get('/api/community-runs' + _qs(p)),
  healerRuns:          (p = {}) => get('/api/healer-runs' + _qs(p)),
  aspectRuns:          (p = {}) => get('/api/aspect-runs' + _qs(p)),
  // Encoder journals — every encoder's residue, one endpoint.
  journals:            (p = {}) => get('/api/journals' + _qs(p)),
  journalSummary:      (p = {}) => get('/api/journals/summary' + _qs(p)),
  consolidationPrompt: (p = {}) => get('/api/consolidation-prompt' + _qs(p)),
  traces:              (p = {}) => get('/api/traces' + _qs(p)),
  errors:              (p = {}) => get('/api/errors' + _qs(p)),
  dashboardErrors:     (p = {}) => get('/api/dashboard-errors' + _qs(p)),
  nodes:               (p = {}) => get('/api/nodes' + _qs(p)),
  graph3d:             ()       => get('/api/graph3d'),
  node:                (id)     => get('/api/node/' + encodeURIComponent(id)),
  nodeCorrections:     (id)     => get('/api/node/' + encodeURIComponent(id) + '/corrections'),
  nodeSourceRefs:      (id)     => get('/api/node/' + encodeURIComponent(id) + '/source-refs'),
  // Self-channel (Streams tab)
  selfMessages:        (p = {}) => get('/api/self-messages' + _qs(p)),
  bootRenders:         (p = {}) => get('/api/boot-renders' + _qs(p)),
  // One session's OWN operator conversation (identity cue, not stream traffic)
  sessionMessages:     (p = {}) => get('/api/session-messages' + _qs(p)),
  selfPresence:        (p = {}) => get('/api/self-presence' + _qs(p)),
  selfSend:            (body)   => post('/api/self-send', body),
  // Thalamus (the brain's standing-intent queue) — items + delivery ledger
  thalamus:            (p = {}) => get('/api/thalamus' + _qs(p)),
  // The recall probe — POST because it carries a query body, not because it
  // writes. The server pins mark_accessed=False, so it stays a pure read.
  recall:              (body)   => post('/api/recall', body),
};

export default api;
