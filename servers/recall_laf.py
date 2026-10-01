"""LAF v1 — the recall scorer (§19 P1).

The only scorer: brain_recall._recall_impl calls scores() on every recall, and
a failure here returns empty + `laf_unavailable` rather than degrading to
anything else.

The composition (measured on the 24-cue lens-independent gold, eval/laf/composition_probe.py
+ maxsim_decomp.py — 18% need@5 / 28% need@25 vs production ~10% need@5):

    score(n) = sigmoid( Σ_field gain_field · z(field(n)) )

The FIELD REGISTRY (LafV1Engine._fields) is the extension seam: adding a lane = one
registry entry + a 'gain_<name>' config key. v1 fields:

    maxsim — best raw cosine(query, node) across the 6 embedding views (nanmax; the
             decomposed sum(z)/max(z) variants measured worse IN-STACK at static gains —
             they become fitted columns at P3, see eval/laf/maxsim_decomp.md)
    pick   — nodes Haiku surfaced AND selected at past moments similar to the query
             (uncapped full-history trace scan — the newest-500 recall_episodes cap is
             deliberately NOT inherited — structurally wrong for a field)
    enc    — nodes created/revised at similar past moments
    idf    — production's idf2 rare-token title boost re-expressed as a lane
    sit    — cosine(query, node _situation vector) as its own lane (NOT folded into
             maxsim's nanmax: separate gain so P4 can re-cue it from situation state)

Gains resolve through ONE seam (LafV1Engine.config: module defaults ⊕ the interactions
K-store entry 'recall_laf', TTL-cached) — P3 fits the values, P4 swaps the resolver to
g(query); the runtime shape here never changes for that.

RANGE CONTRACT: scores() returns values in the OPEN interval (0,1) — the machinery
brain_recall wraps around the score (fatigue multiply, noise/relevance floor) assumes
cosine-like magnitudes. The sigmoid guarantees it mathematically;
scores() validates finiteness and raises rather than ship an out-of-contract vector
(the caller reports the failure and returns empty).

PER-QUERY TELEMETRY: scores() also returns a compact per-field z-score record for the
top-scoring nodes ({node_id: {field: z}}); brain_recall attaches it to the result as
`_laf_fields` so the S1R trace substrate accretes (query, per-field scores, outcome)
rows in production — the P2 dataset walker's training feed.

Known accepted gap (code-review 2026-07-02, finding 10): a node with NO embedding rows
is invisible to recall until its vectors land (the embed worker writes them after
remember; daemon boot re-queues any gap). Lexical retrieval returns only via the P4 gate.

AS_OF (§20.11 read-side time travel): scores(..., as_of=<ISO ts>) scores the
query against corpus state at that instant — masks, not copies. Caches stay
current-state supersets; as_of builds per-call boolean row masks from two
creation-date arrays (node rows / trace rows). The node mask applies ONCE, in
the z-score registry loop, so every node-indexed lane — including future ones —
inherits time travel for free; the trace mask applies at the shared episodic
door (roles_for_moments) plus the moment-similarity line. as_of=None builds no
masks and is the identical code path (inert by construction, pinned by tests).

Caches (daemon-resident, staleness-checked per call, lock-guarded; the daemon
builds them at boot via warm(), any other Brain on its first scores() call):
  field matrices  growable [N×768] per view — INCREMENTAL rowid-watermark upserts via
                  VectorDAL.vectors_since; full rebuild only on deletion (count shrink)
  node created    row-aligned created_at array (the as_of node-mask ingredient)
  title idf       full rebuild, but only when nodes' change_key moved AND a TTL expired
                  (+ per-token sorted creation timestamps — the walker's as-of df
                  invention moved into the cache)
  trace matrix    block list (no per-append copy), incremental by created_at
                  (+ parallel created_at list — the as_of trace-mask ingredient)
  config          TTL-cached K-store overlay
"""
import bisect
import math
import re
import threading
import time
from collections import defaultdict

import numpy as np

try:
    from .brain_constants import _TITLE_BOOST_STOPWORDS
    from .pipeline_contract import EMBEDDING_GROUPS
    from .trace_contract import OPERATOR_DIALOGUE_REF_TYPES, is_machine_turn
    from .scales.s1.trace_links import gather, nodes_for_traces, _stop_of
except ImportError:                                    # direct-script import shape
    from brain_constants import _TITLE_BOOST_STOPWORDS
    from pipeline_contract import EMBEDDING_GROUPS
    from trace_contract import OPERATOR_DIALOGUE_REF_TYPES, is_machine_turn
    from scales.s1.trace_links import gather, nodes_for_traces, _stop_of

# v1 gains + knobs — the measured static composition. Overridable via the interactions
# K-store (name='recall_laf', config JSON with these keys); P3 replaces the values,
# not the shape. One 'gain_<name>' per registry field.
DEFAULT_CONFIG = {
    'gain_maxsim': 1.0,
    'gain_pick': 0.5,
    'gain_enc': 0.3,
    'gain_idf': 0.5,
    'gain_sit': 0.5,
    # proj: session-project provenance match. SHIPPED AT ZERO — the lane is
    # wired (telemetry live, per-candidate _laf_fields carry its z) but
    # contributes nothing until a measured gain is registered via the
    # recall_laf interaction. The gate corpus is single-project (all cues are
    # brain-repo) so a nonzero gain here is untestable until cross-project
    # cues are minted — see docs/BACKLOG.md.
    'gain_proj': 0.0,
    'top_moments': 15,        # similar past moments to seed episodic roles from
    'window_turns': 1,        # ±N-turn moment window (the measured winner)
    'sigmoid_scale': 3.0,     # z-sum → (0,1) squash temperature
    'session_trace_pull': 2000,
    'telemetry_top_n': 50,    # per-field z-scores recorded for this many top nodes
    # Lane normalizer (P3.0): 'current' | 'support' | 'rank' — see Z_NORMS.
    # Sparse lanes (pick/enc/idf are mostly-zero) explode under plain z
    # (enc z≈11 beside cosine z≈2 — q1_reverse eyeball); the P3.0 winner
    # ships as a K-store flip of this key, rollback = flip back.
    'z_norm': 'current',
    # Moment stack (§20.17/§20.18, DORMANT — both defaults falsy, so the
    # moment code path is unreachable until a K-store flip):
    #   moment_K     — conversational turns of history to pull (0 = off).
    #   moment_gains — the composition TABLE, per-(lane,slot,side) z-gains:
    #       slot keys   '{lane}_{side}{j}' (maxsim_a1, sit_o2, idf_a3, ...)
    #                   — only slots present in the table are computed;
    #       o0 keys     '{lane}_o0' override the current-message content-lane
    #                   gains (the full-fitted-table arms; absent → gain_*);
    #       bare keys   'pick'/'enc'/'proj' override those lanes' gains
    #                   (the fitted tables zero pick/enc; absent → gain_*).
    #     A nonempty table activates the overrides even at moment_K=0 — that
    #     IS the fitted-K0 arm (A0f, §20.18); an arm is a gain table, not code.
    #     Values are frozen from eval/laf/walker/definitive_fit.json — never
    #     refit on an eval corpus.
    'moment_K': 0,
    'moment_gains': {},
}

MOMENT_TEXT_CAP = 500   # history-slot idf cap, matching the walker's fitted TEXT_CAP;
                        # independent of the current prompt's recall-query cap

CONFIG_TTL_S = 60.0           # K-store overlay refresh cadence
TITLES_TTL_S = 60.0           # min seconds between title-idf rebuilds
PROJECTS_TTL_S = 60.0         # min seconds between project-map rebuilds
TRACE_BLOCK_CONSOLIDATE = 32  # merge trace blocks when the list grows past this
_GROW = 1.3                   # matrix capacity growth factor

# The 6 maxsim views: every live embedding group (weight>0) by vector_type.
MAXSIM_VIEWS = []
for _g in EMBEDDING_GROUPS.values():
    _vt = _g.get('vector_type')
    if _g.get('weight', 0) > 0 and _vt and _vt not in MAXSIM_VIEWS:
        MAXSIM_VIEWS.append(_vt)
_ALL_VIEWS = MAXSIM_VIEWS + ['_situation']

_IDF_TOK = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")   # idf2's tokenizer, verbatim


def _unit(blob):
    """Embedding blob → unit float32 vector (None if absent/zero).

    THE single normalizer for LAF vectors — eval/laf/operators.py aliases this
    so probes measure the exact function production runs."""
    if blob is None:
        return None
    v = (np.frombuffer(blob, dtype=np.float32)
         if isinstance(blob, (bytes, bytearray)) else np.asarray(blob, dtype=np.float32))
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else None


def _zscore(x, n, mask=None):
    """Standardize over finite (optionally masked) entries → unit variance; 0
    elsewhere. THE LAF fusion normalizer — gains stay pure influence dials only
    through this exact form; eval/laf/laf_metrics.py delegates here."""
    m = np.isfinite(x)
    if mask is not None:
        m = m & mask
    o = np.zeros(n)
    if int(m.sum()) > 2 and np.std(x[m]) > 1e-9:
        o[m] = (x[m] - x[m].mean()) / x[m].std()
    return o


def _zscore_support(x, n, mask=None):
    """support-z (P3.0 variant): statistics over the NONZERO finite support
    only; zeros — 'no activation' in the sparse lanes (pick/enc/idf), which
    plain z counts as real values, shrinking std until matches explode —
    stay exactly 0 (neutral). Dense lanes (cosines are never exactly 0.0)
    are bit-identical to _zscore. Same small-support guard shape.

    LANE CONSTRAINT: valid ONLY for lanes whose 0.0 means absence (the
    SUPPORT_ZERO_SEA_LANES). Lanes where 0.0 is a REAL activation per the
    _fields LANE CONTRACT (proj: cross-project inhibition) must NOT route
    here — support-z would drop the zeros from the stats and zero the whole
    lane (all-1.0 support, std<1e-9). The scores() z-loop enforces this."""
    m = np.isfinite(x)
    if mask is not None:
        m = m & mask
    sup = m & (x != 0.0)
    o = np.zeros(n)
    if int(sup.sum()) > 2 and np.std(x[sup]) > 1e-9:
        o[sup] = (x[sup] - x[sup].mean()) / x[sup].std()
    return o


def _zscore_rank(x, n, mask=None):
    """rank-norm (P3.0 variant): average-tie fractional percentiles over the
    finite (masked) universe, mapped by the ANALYTIC uniform z — centered
    percentile × √12 — so output is z-space-commensurate (equals z-of-ranks
    for a tie-free dense lane, where rank std is k/√12) and BOUNDED by ±√3
    unconditionally. Deliberately NOT _zscore of the rank vector: a sparse
    lane's zero-tie block shrinks the rank std until the matches explode
    again (the p3_norm sanity fixture caught z=7.1) — the fixed analytic
    scale is the whole point. The zero sea sits at its own percentile; the
    matches rank above it, capped at +√3."""
    m = np.isfinite(x)
    if mask is not None:
        m = m & mask
    o = np.zeros(n)
    k = int(m.sum())
    if k <= 2:
        return o
    xv = x[m]
    order = np.argsort(xv, kind='stable')
    ranks = np.empty(k)
    ranks[order] = np.arange(k, dtype=float)
    _, inv, counts = np.unique(xv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    rv = (sums / counts)[inv]                 # average rank across ties
    p = (rv + 0.5) / k                        # fractional percentile (0,1)
    o[m] = (p - p.mean()) * np.sqrt(12.0)     # analytic uniform z, mean-0
    return o


Z_NORMS = {'current': _zscore, 'support': _zscore_support, 'rank': _zscore_rank}

# Lanes whose 0.0 genuinely means "no activation" (the sparse zero seas).
# support-z applies ONLY to these; every other lane keeps plain z under
# z_norm='support' because its zeros are real values (proj's 0.0 is
# cross-project INHIBITION — the _fields LANE CONTRACT). rank-norm is
# lane-safe (ranks preserve the 0<1 ordering) and needs no gating.
SUPPORT_ZERO_SEA_LANES = frozenset(('pick', 'enc', 'idf'))


def zscore_variant(x, n, mask=None, kind='current'):
    """The ONE normalizer dispatch — engine z-loop and the walker's compose()
    both route here so eval measures the exact function production runs."""
    try:
        fn = Z_NORMS[kind]
    except KeyError:
        raise ValueError('unknown z_norm %r (valid: %s)'
                         % (kind, '/'.join(Z_NORMS)))
    return fn(x, n, mask=mask)


def idf_scores(query, title_tok, title_df, n, n_titles=None):
    """Production idf2 title boost as a lane: per node-row, Σ log-idf of the
    query's rare tokens found in its title, normalized by the query's total idf
    mass. Pure function so eval probes score the identical formula.

    title_tok: {row: frozenset(tokens)}; title_df: {token: doc frequency}.
    n_titles: corpus size for the idf denominator — defaults to len(title_tok)
    (the production shape, where title_tok spans the full node matrix). Callers
    scoring a RESTRICTED row set against a larger corpus (the §20 walker's
    as-of replay) pass the true corpus size explicitly.
    """
    vec = np.zeros(n)
    q_tokens = {t for t in _IDF_TOK.findall(query.lower())
                if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS}
    if not q_tokens or not title_tok:
        return vec
    if n_titles is None:
        n_titles = len(title_tok)
    n_titles = max(n_titles, 1)
    idf = {t: math.log((n_titles + 1) / (title_df.get(t, 0) + 1))
           for t in q_tokens}
    total = sum(idf.values()) or 1.0
    for i, ts in title_tok.items():
        m = sum(idf[t] for t in q_tokens if t in ts)
        if m > 0:
            vec[i] = m / total
    return vec


def roles_for_moments(brain, moments, window_turns, pull_limit, as_of=None):
    """(session, short, stop)-keyed similar-moment scores → per-moment role records.

    THE shared episodic role-join (production engine AND eval/laf/episodic_ops.py
    consume this, so the measured join semantics are the shipped ones): per session,
    gather the surface/encode/recall streams once, join by stop via nodes_for_traces,
    union roles across the ±window stops, picked-wins-over-dropped within a moment.

    as_of (§20.11 #5, the shared episodic-door chokepoint): ISO timestamp —
    the pull window is positioned AT it (SQL `created_at <` via gather's
    older_than, strict; the walker's `<=` can't diverge — as_of anchors to an
    s0 row, these streams are s1, and stamps are microsecond). Pushed into
    SQL, never Python-filtered after the fetch: the pull is newest-first
    LIMIT pull_limit, so a post-filter kept only rows NEWER than as_of and a
    deep-history replay got empty role sets — no ground truth, no error.
    None = live, no bound (the identical path). A pull that still fills the
    window at as_of is flagged loud (limit+1 probe → laf_roles_pull_truncated)
    — under replay a clipped coverage read means wrong numbers, not slower ones.

    moments: {(session_id, short, stop): score}
    Returns [{'score', 'picked', 'encoded', 'dropped'}] — sets of node ids.
    """
    w = int(window_turns)
    by_sess = defaultdict(list)
    for (sess, short, stop), s in moments.items():
        by_sess[sess].append((short, stop, s))
    records = []
    for sess, moms in by_sess.items():
        if as_of is None:
            st = gather(brain, sess, streams=('surface', 'encode', 'recall'),
                        limit=pull_limit)
        else:
            # limit+1 probe (the truncation contract): the extra row is proof
            # the session holds more stream rows before as_of than the window
            # carries — moments at early stops would join against nothing.
            st = gather(brain, sess, streams=('surface', 'encode', 'recall'),
                        limit=pull_limit + 1, older_than=as_of)
            clipped = [k for k, v in st.items() if len(v) > pull_limit]
            if clipped:
                st = {k: v[:pull_limit] for k, v in st.items()}
                brain._log_error(
                    'laf_roles_pull_truncated', None,
                    'session=%s streams=%s hold >%d rows before as_of=%s — '
                    'role join is measuring a clipped window'
                    % (sess[:8], ','.join(clipped), pull_limit, as_of))
        targets = [{'id': '%d@%d' % (mi, ws), 'chain_id': 's0-%s-%d' % (short, ws)}
                   for mi, (short, stop, _s) in enumerate(moms)
                   for ws in range(max(stop - w, 0), stop + w + 1)]
        links = nodes_for_traces(st['surface'], st['encode'], targets,
                                 recall_traces=st['recall'])
        for mi, (short, stop, s) in enumerate(moms):
            picked, encoded, dropped = set(), set(), set()
            for ws in range(max(stop - w, 0), stop + w + 1):
                link = links.get('%d@%d' % (mi, ws)) or {}
                picked.update(link.get('surfaced', []))
                encoded.update(link.get('encoded', []))
                dropped.update(link.get('dropped', []))
            dropped -= picked          # picked-wins within the whole moment
            records.append({'score': s, 'picked': picked,
                            'encoded': encoded, 'dropped': dropped})
    return records


def role_rows(brain, ids, resolve):
    """Episodic role node ids → ({id: master row}, unresolvable ids).

    Role ids come from TRACES, so they are HISTORY, not live node ids (the
    docs/TRACE-NODE-RESOLUTION.md contract). S2 consolidation absorbs A into
    live survivor B and archives A; the LAF matrix is live-only, so `resolve(A)`
    returns None and that moment's evidence is dropped on the floor — B never
    inherits the activation history its own content earned, and the loss
    compounds with every consolidation cycle.

    One batched `resolve_live` walk over the UNRESOLVED ids only maps each dead
    id to its survivor's row instead. Live ids never enter the walk, so an
    all-live id set costs zero queries; otherwise the walk is level-batched
    (2 queries per chain level, independent of id count). Retired nodes (archived,
    no survivor) and true orphans stay dropped — there is nothing live to credit.

    The second return is those still-unresolvable ids. Dropping them is routine
    (§6: the redirect is normal retrieval, the drop is at most low-severity), so
    this reports rather than logs — a per-recall warning would fire constantly
    and drown the signal it exists to carry. The engine owns the cadence.
    """
    rows, missing = {}, []
    for nid in set(ids):
        i = resolve(nid)
        if i is not None:
            rows[nid] = i
        else:
            missing.append(nid)
    if not missing:
        return rows, set()
    walked = brain.resolve_live(missing, on_orphan='mark')
    for dead, surv in walked['redirected'].items():
        i = resolve(surv)
        if i is not None:
            rows[dead] = i
    # A survivor outside the matrix (retired terminal, or no embedding row)
    # leaves the input unresolved just as an orphan does — same outcome, so
    # the report is keyed on what LANDED, not on how the walk classified it.
    return rows, {n for n in missing if n not in rows}


class LafV1Engine:
    """Daemon-resident scorer: caches the matrices, computes per-query field scores.

    One instance per Brain (lazily attached as brain._laf_engine). All cache
    refreshes are guarded by one lock; concurrent recalls of different queries
    share the refreshed caches.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # field matrices — growable, row-aligned across views
        self._mats = None            # {view: [cap×768] float32, NaN where absent}
        self._master = []            # ordered node ids (rows 0.._n)
        self._idx = {}               # node_id → row
        self._n = 0                  # rows in use
        self._cap = 0                # allocated rows
        self._dim = 768
        self._vec_key = None         # VectorDAL.change_key() at last sync
        self._vec_watermark = 0      # max node_enrichments rowid ingested
        # node creation timestamps — row-aligned ISO strings ('' = unknown,
        # compares ≤ everything → always visible). The as_of node-mask source.
        self._created = np.full(0, '', dtype='<U40')
        # title idf
        self._title_tok = {}         # row → frozenset(title tokens)
        self._title_df = {}          # token → doc frequency over titles
        self._token_created = {}     # token → SORTED [created_at] (as-of df)
        self._title_created = []     # SORTED [created_at] of titled rows (as-of n_titles)
        self._titles_key = None
        self._titles_ts = 0.0
        # project provenance (proj lane) — row-keyed parallel arrays for the
        # vectorized per-query lane build (rows of project-carrying nodes only)
        self._proj_rows = np.empty(0, dtype=np.int64)
        self._proj_vals = np.empty(0, dtype=object)
        self._proj_key = None        # (nodes change_key, kv change_key)
        self._proj_ts = 0.0
        # trace matrix — block list, no per-append copy
        self._tr_blocks = []         # [ndarray [k×768]]
        self._tr_meta = []           # [(chain_id, session_id)] aligned to blocks' rows
        self._tr_ids = {}            # trace_events.id → global row (moment-stack join)
        self._tr_created = []        # [created_at] aligned to _tr_meta (as_of trace mask)
        self._tr_created_arr = np.full(0, '', dtype='<U40')   # lazy np mirror
        self._tr_last = ''           # max created_at ingested
        # config overlay
        self._cfg = None
        self._cfg_ts = 0.0
        # last moment-stack coverage ledger (V2 / W1 instrument; None when
        # the moment path didn't run on the most recent scores() call)
        self._last_moment_ledger = None

    # ── config: the ONE gain-resolution seam (P4 swaps this body to g(φ(q))) ──
    def config(self, brain):
        now = time.monotonic()
        if self._cfg is not None and now - self._cfg_ts < CONFIG_TTL_S:
            return self._cfg
        try:
            # The resolver overlays any DB override onto DEFAULT_CONFIG
            # (the registry's default for 'recall_laf') — total by
            # construction, parse failures already logged there.
            cfg = brain.get_interaction_config('recall_laf')
        except Exception as e:
            # Loud: recall must survive a broken K-store read (DAL/DB-level
            # failure). Falls back to the SAME default home the resolver
            # uses. TTL-bounded, so this can't spam (≤1 log per CONFIG_TTL_S).
            cfg = dict(DEFAULT_CONFIG)
            try:
                brain._log_error('recall_laf_config', e,
                                 'get_interaction_config failed — running on '
                                 'module defaults')
            except Exception:
                pass
        # z_norm validates at the MERGE, not in the hot loop: an unrecognized
        # value (typo'd K-store flip, JSON non-string) must degrade to
        # 'current' with one loud log — NOT raise per-query inside scores(),
        # which would take recall down fleet-wide (there is no second scorer).
        # Code-review catch 2026-07-16.
        if cfg.get('z_norm') not in Z_NORMS:
            bad = cfg.get('z_norm')
            cfg['z_norm'] = 'current'
            try:
                brain._log_error('recall_laf_config',
                                 ValueError('invalid z_norm %r' % (bad,)),
                                 'unknown z_norm in K-store config — falling '
                                 'back to current (valid: %s)'
                                 % '/'.join(Z_NORMS))
            except Exception:
                pass
        self._cfg, self._cfg_ts = cfg, now
        return cfg

    def invalidate_config(self):
        """Drop the cached overlay so the next config() re-resolves now.

        Called by Brain.invalidate_interaction_caches when the recall_laf
        pointer flips or clears — without it a deploy/clear runs up to
        CONFIG_TTL_S late.
        """
        self._cfg, self._cfg_ts = None, 0.0

    # ── field matrices: growable buffers + rowid-watermark incremental upserts ──
    def _ensure_capacity(self, need):
        if need <= self._cap:
            return
        new_cap = max(int(need * _GROW), 64)
        for vt, m in self._mats.items():
            grown = np.full((new_cap, self._dim), np.nan, dtype=np.float32)
            grown[:self._n] = m[:self._n]
            self._mats[vt] = grown
        grown_c = np.full(new_cap, '', dtype='<U40')
        grown_c[:self._n] = self._created[:self._n]
        self._created = grown_c
        self._cap = new_cap

    def _row_for(self, node_id):
        """Row for node_id, appending a fresh NaN row if new."""
        i = self._idx.get(node_id)
        if i is not None:
            return i
        i = self._n
        self._ensure_capacity(self._n + 1)
        self._n += 1
        self._master.append(node_id)
        self._idx[node_id] = i
        return i

    def _full_matrix_build(self, brain, model):
        gv = {vt: {} for vt in _ALL_VIEWS}
        for r in brain._vec_dal.get_all_vectors(vector_types=_ALL_VIEWS,
                                                model=model or None):
            uv = _unit(r.get('embedding'))
            if uv is not None:
                gv[r['vector_type']][r['node_id']] = uv
        master = sorted(set().union(*[set(d) for d in gv.values()]) if gv else set())
        self._dim = next((len(v) for d in gv.values() for v in d.values()), 768)
        self._n = len(master)
        self._cap = max(int(self._n * _GROW), 64)
        self._master = list(master)
        self._idx = {nid: i for i, nid in enumerate(master)}
        self._mats = {}
        for vt in _ALL_VIEWS:
            m = np.full((self._cap, self._dim), np.nan, dtype=np.float32)
            for nid, v in gv[vt].items():
                m[self._idx[nid]] = v
            self._mats[vt] = m
        # node creation timestamps, row-aligned (the as_of node-mask source)
        created = dict(brain._nodes.created_rows())
        self._created = np.full(self._cap, '', dtype='<U40')
        for nid, i in self._idx.items():
            self._created[i] = created.get(nid) or ''
        # titles are row-keyed — force a rebuild against the new row space
        # (the as-of title structures rebuild with them)
        self._title_tok, self._titles_key, self._titles_ts = {}, None, 0.0
        self._token_created, self._title_created = {}, []
        # proj arrays are row-keyed too — same forced rebuild, or a reindexed
        # row space serves WRONG-node project labels (rows shift when a node
        # drops out of master, and the nodes/kv change keys can't see that)
        self._proj_rows = np.empty(0, dtype=np.int64)
        self._proj_vals = np.empty(0, dtype=object)
        self._proj_key, self._proj_ts = None, 0.0

    def _refresh_matrices(self, brain, model):
        key = brain._vec_dal.change_key()
        if self._mats is not None and key == self._vec_key:
            return
        if self._mats is None or key[0] < (self._vec_key or (0, 0))[0]:
            # first build, or rows were deleted — the watermark can't see
            # deletions, so rebuild from scratch (rare: revise/archive cleanup)
            self._full_matrix_build(brain, model)
        else:
            n_before = self._n
            for rowid, nid, vt, blob in brain._vec_dal.vectors_since(
                    self._vec_watermark, vector_types=_ALL_VIEWS,
                    model=model or None):
                uv = _unit(blob)
                if uv is None or vt not in self._mats:
                    continue
                # _row_for FIRST: it may grow + REBIND self._mats[vt]; the
                # single-expression form resolved the OLD array before the
                # growth side effect and blew up (IndexError) on exactly the
                # append that crossed the capacity boundary (found 2026-07-17,
                # 20-item pooled build, gate-2 error ledger).
                row = self._row_for(nid)
                self._mats[vt][row] = uv
            if self._n > n_before:
                # backfill created_at for the appended rows (one bulk read;
                # only fires when the refresh actually added nodes)
                created = dict(brain._nodes.created_rows())
                for i in range(n_before, self._n):
                    self._created[i] = created.get(self._master[i]) or ''
        self._vec_key = key
        self._vec_watermark = key[1]

    # ── title idf: full rebuild, TTL-throttled (cheap but not free) ──
    def _refresh_titles(self, brain):
        """Returns the nodes change_key it computed, or None on a TTL skip —
        _refresh_projects reuses it so one expiry window pays one probe."""
        now = time.monotonic()
        if self._title_tok and now - self._titles_ts < TITLES_TTL_S:
            return None
        key = brain._nodes.change_key()
        if key == self._titles_key and self._title_tok:
            self._titles_ts = now
            return key
        tok, df = {}, defaultdict(int)
        tok_created, all_created = defaultdict(list), []
        for nid, title in brain._nodes.title_rows():
            i = self._idx.get(nid)
            if i is None:
                continue
            ts = frozenset(_IDF_TOK.findall(title.lower()))
            tok[i] = ts
            for t in ts:
                df[t] += 1
            # as-of df substrate: per-token creation timestamps (walker's
            # invention, §20.11 #2). Unknown created_at ('') sorts before
            # every ISO string, so it counts at ANY as_of — same
            # always-visible convention as the node mask, and it keeps
            # as_of=now ≡ live df exactly.
            c = str(self._created[i])
            all_created.append(c)
            for t in ts:
                tok_created[t].append(c)
        all_created.sort()
        for lst in tok_created.values():
            lst.sort()
        self._title_tok, self._title_df = tok, dict(df)
        self._token_created, self._title_created = dict(tok_created), all_created
        self._titles_key, self._titles_ts = key, now
        return key

    # ── project provenance map: full rebuild, TTL-throttled (same shape) ──
    def _refresh_projects(self, brain, node_key=None):
        now = time.monotonic()
        if self._proj_key is not None and now - self._proj_ts < PROJECTS_TTL_S:
            return
        # Two-component staleness key: nodes change_key alone is blind to
        # node_metadata_kv writes, and project_rows reads COALESCE(kv, column)
        # — a kv backfill (the column→kv migration, a set_many) must
        # invalidate this cache or the lane serves pre-migration labels
        # until an unrelated node insert.
        nk = node_key if node_key is not None else brain._nodes.change_key()
        key = (nk, brain._meta_kv.change_key())
        if key == self._proj_key:
            self._proj_ts = now
            return
        rows, vals = [], []
        for nid, p in brain._nodes.project_rows():
            i = self._idx.get(nid)
            if i is not None:
                rows.append(i)
                vals.append(p)
        self._proj_rows = np.asarray(rows, dtype=np.int64)
        self._proj_vals = np.asarray(vals, dtype=object)
        self._proj_key, self._proj_ts = key, now

    # ── trace matrix: block list — appends never copy the resident rows ──
    def _refresh_traces(self, brain):
        rows = brain._trace_dal.event_vector_rows(
            scale='s0', ref_types=list(OPERATOR_DIALOGUE_REF_TYPES),
            since=self._tr_last or None)
        if not rows:
            return
        base = len(self._tr_meta)
        meta, vecs, created, last = [], [], [], self._tr_last
        for chain_id, session_id, created_at, blob, trace_id, _ref_type in rows:
            uv = _unit(blob)
            if uv is None:
                continue
            if trace_id:
                self._tr_ids[trace_id] = base + len(vecs)
            meta.append((chain_id, session_id))
            vecs.append(uv)
            created.append(created_at or '')
            if created_at > last:
                last = created_at
        self._tr_last = last
        if not vecs:
            return
        self._tr_blocks.append(np.stack(vecs))
        self._tr_meta.extend(meta)
        self._tr_created.extend(created)
        if len(self._tr_blocks) > TRACE_BLOCK_CONSOLIDATE:
            self._tr_blocks = [np.vstack(self._tr_blocks)]   # one copy, occasional

    def _tr_vec(self, idx):
        """Global trace-row index → unit vector (block-list addressing;
        consolidation vstacks in order, so indices stay valid)."""
        for b in self._tr_blocks:
            if idx < len(b):
                return b[idx]
            idx -= len(b)
        return None

    # ── as_of masks: the §20.11 chokepoints' shared builder ──
    def _asof_masks(self, as_of, n):
        """(node_mask[n], trace_mask[T]) — True = existed at as_of
        (created_at ≤ as_of, string-lexicographic over the brain's uniform
        ISO shape). Masks over current-state caches, never copies. Unknown
        created_at ('') compares ≤ everything → always visible (benign:
        every row carries one)."""
        node_mask = self._created[:n] <= as_of
        if len(self._tr_created_arr) != len(self._tr_created):
            self._tr_created_arr = np.asarray(self._tr_created, dtype='<U40')
        return node_mask, self._tr_created_arr <= as_of

    # ── as-of idf: df + n_titles at corpus-state as_of ──
    def _idf_asof(self, query, n, as_of):
        """idf lane with the df denominator time-travelled: per-token df and
        n_titles via bisect over sorted creation timestamps — the walker's
        exact semantics (eval/laf/walker/scores.py asof_tok_df: bisect_left,
        strictly-before), so engine-as_of and walker rows cross-check
        row-level. Same pure idf_scores formula; only the corpus counts move."""
        q_tokens = {t for t in _IDF_TOK.findall(query.lower())
                    if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS}
        df = {t: bisect.bisect_left(self._token_created[t], as_of)
              for t in q_tokens if t in self._token_created}
        n_titles = bisect.bisect_left(self._title_created, as_of)
        return idf_scores(query, self._title_tok, df, n,
                          n_titles=max(n_titles, 1))

    # ── episodic field (shared join semantics via roles_for_moments) ──
    def _episodic_vectors(self, brain, qv, cfg, n, as_of=None, trace_mask=None):
        """(pick, enc) [n] activation vectors from similar past surface-moments.

        Full-history block scan (no newest-500 cap), top `top_moments`
        (session, stop) moments deduped at max similarity, roles joined via the
        canonical shared roles_for_moments, ±window unioned, picked-wins.
        """
        pick = np.zeros(n)
        enc = np.zeros(n)
        if not self._tr_blocks:
            return pick, enc
        sims = np.concatenate([b @ qv for b in self._tr_blocks])
        if trace_mask is not None:
            # the moment-similarity chokepoint (§20.11 #4): traces created
            # after as_of can never seed a moment
            sims = np.where(trace_mask, sims, -np.inf)
        want = int(cfg['top_moments'])
        moments = {}                          # (session, short, stop) → score
        for i in np.argsort(-sims)[:want * 3]:
            chain = self._tr_meta[i][0] or ''
            stop = _stop_of(chain)
            parts = chain.split('-')          # s0-{short}-{stop}
            sess = self._tr_meta[i][1]
            if stop is None or not sess or len(parts) < 3:
                continue
            key = (sess, parts[1], stop)
            s = float(sims[i])
            if s > moments.get(key, 0.0):
                moments[key] = s
            if len(moments) >= want:
                break
        records = roles_for_moments(brain, moments, cfg['window_turns'],
                                    cfg['session_trace_pull'], as_of=as_of)
        harvested = set()
        for r in records:
            harvested |= r['picked'] | r['encoded']
        rows, unresolvable = role_rows(brain, harvested, self._resolve)
        self._report_resolution(brain, harvested, rows, unresolvable)
        for r in records:
            s = r['score']
            for node in r['picked']:
                i = rows.get(node)
                if i is not None and s > pick[i]:
                    pick[i] = s
            for node in r['encoded']:
                i = rows.get(node)
                if i is not None and s > enc[i]:
                    enc[i] = s
        return pick, enc

    def _report_resolution(self, brain, harvested, rows, unresolvable):
        """Per-call episodic resolution health — a GAUGE, not an incident.

        Dropped evidence is the normal background of a brain with history:
        traces outlive nodes, and a retired id carries no survivor to credit.
        Warning per unresolvable id therefore reports a standing population as
        if each sighting were news, and cannot see the only thing worth an
        alarm — the drop share moving.

        So every call writes its reading to debug, and TOTAL loss is the one
        thing that warns: nothing resolved is a cold matrix or a resolver that
        stopped resolving, and it needs no threshold to recognise. A partial
        rise would need a band, a band needs a measured distribution to sit
        in, and survivor credit shipped on contract rather than a corpus win
        — not worth a number someone has to maintain. The share is recorded
        either way; if it ever moves, the rows are there to start from.

        An empty harvest reports too: "the lane ran and found no roles" and
        "the lane never ran" must not look the same to a reader.
        """
        n = len(harvested)
        dropped = len(unresolvable)
        sample = sorted(unresolvable)[:8]
        brain.log_debug('laf_resolution', 'recall_laf',
                        harvested=n, credited=len(rows), dropped=dropped,
                        share=round(dropped / n, 3) if n else 0.0,
                        sample=sample)
        if n and not rows:
            brain._log_warning(
                'laf_resolution_loss',
                'all %d episodic role ids reached no live row' % n,
                'sample=%s' % sample)

    def _resolve(self, node_id):
        """Role node id → master row, or None. Exact 8-char ids everywhere —
        surface traces and encode deltas both store the full id."""
        return self._idx.get(node_id)

    def _project_field(self, session_project, n):
        """proj lane raw activations: same-project 1.0, cross-project 0.0,
        no-project NaN (neutral — _zscore's isfinite mask excludes it, the
        sit-lane lesson as a design rule). No session project → all-NaN, the
        whole lane inert. z-scoring makes the lane self-calibrating: in a
        near-single-project brain the finite entries have ~no variance and
        the lane contributes ~nothing; its power is INHIBITION when the
        operator works outside the dominant project."""
        vec = np.full(n, np.nan)
        if not session_project or not len(self._proj_rows):
            return vec
        m = self._proj_rows < n
        vec[self._proj_rows[m]] = (
            self._proj_vals[m] == session_project).astype(float)
        return vec

    # ── the per-message content-lane triple: ONE code path for j0 + slots ──
    def _content_lanes(self, text, qv, n, as_of=None):
        """{maxsim, sit, idf} raw activations for ONE message — the j0 query
        and every moment slot route here (§20.18: a slot is a lane set, not a
        mechanism), so the walker's lane×slot cells and the engine compose the
        same math by construction (G1). qv=None → cosine lanes skipped; empty
        text → idf skipped (the walker's j_missing semantics). Raw activations
        per the LANE CONTRACT (NaN = missing, never 0)."""
        lanes = {}
        if qv is not None:
            with np.errstate(all='ignore'):
                lanes['maxsim'] = np.nanmax(
                    np.stack([self._mats[vt][:n] @ qv for vt in MAXSIM_VIEWS]),
                    axis=0)
            # NaN (no _situation vector) stays NaN: _zscore's isfinite mask
            # excludes it from the stats and scores it 0 — absence is neutral,
            # same semantics as maxsim's nanmax. Zero-filling here scored a
            # missing vector as a real cosine of 0.0 — ~10σ below the corpus
            # mean (0.475±0.045), burying just-encoded nodes and any node in
            # the revise→re-embed window (sit z −10.6 ≈ −5.3 zsum at gain 0.5).
            lanes['sit'] = self._mats['_situation'][:n] @ qv
        if text:
            lanes['idf'] = (idf_scores(text, self._title_tok, self._title_df, n)
                            if as_of is None else self._idf_asof(text, n, as_of))
        return lanes

    # ── the field registry: name → per-node raw activation [n] ──
    def _fields(self, brain, query, qv, cfg, n, session_project=None,
                as_of=None, trace_mask=None):
        """Adding a lane = one entry here + a 'gain_<name>' DEFAULT_CONFIG key.
        Every vector is z-scored by the caller — return RAW activations.

        LANE CONTRACT: missing data is NaN, NEVER 0.0 — _zscore's isfinite
        mask excludes NaN from the stats so absence is neutral. A 0.0 is a
        REAL activation that z-scores far below the corpus mean (the sit-lane
        zero-fill buried fresh nodes at −10σ; tests pin the NaN passthrough
        for sit and proj)."""
        content = self._content_lanes(query, qv, n, as_of=as_of)
        pick, enc = self._episodic_vectors(brain, qv, cfg, n,
                                           as_of=as_of, trace_mask=trace_mask)
        return {
            'maxsim': content['maxsim'],
            'pick': pick,
            'enc': enc,
            # empty query text → no idf tokens → zero lane (idf_scores'
            # own empty-query shape, preserved through the extraction)
            'idf': content.get('idf', np.zeros(n)),
            'sit': content['sit'],
            'proj': self._project_field(session_project, n),
        }

    # ── the moment stack: last-K turns as slot lanes (§20.17/§20.18) ──
    def _moment_stack(self, brain, session_id, K, as_of=None):
        """The session's last-K conversational turns, walker slot semantics:
        slot j = TURN DISTANCE — turn t−j contributes its operator message as
        oj and its assistant message as aj; a0 never joins (W3 — at live time
        the response doesn't exist; under as_of the strict < cut excludes it).

        Turns come through the traces-layer door (get_conversation — the same
        object the S1 encoder reads); machine turns drop their operator side
        but keep slot occupancy and their assistant side (the shared
        is_machine_turn — W2, the v6 mislabel lesson); vectors join from the
        resident trace matrix by trace_id. A turn whose embedding hasn't
        landed yet joins as vec=None (cosine slots skipped, idf still fires) —
        ledger-counted, so W1 cache freshness is a measured number, never an
        assumption.

        THE LIVE-EDGE RULE: the stack is COMPLETED turns. The user_message
        trace is written at prompt-arrival (not Stop — see get_session_turns'
        exclude_trace_id note), so at recall time the CURRENT prompt is
        already in the conversation; without this rule it would enter as j=1
        and double-count the j0 query. A trailing answer-less user turn is
        exactly that in-flight prompt → dropped (ledger 'live_edge_dropped');
        under replay the as_of strict < cut (get_conversation's older_than —
        in SQL, so the window is the last turns AT as_of, not the last turns
        NOW minus the future) removes the cue row before this rule sees it.

        Returns ([(side, j, unit_vec_or_None, text)], ledger)."""
        ledger = {'rows': 0, 'turns': 0, 'machine_dropped': 0,
                  'missing_vec': 0, 'live_edge_dropped': 0}
        try:
            rows = brain.get_conversation(session_id, limit=4 * K + 8,
                                          with_judge_output=False,
                                          older_than=as_of)
        except Exception as e:
            # Degrading to bare-query recall beats killing the whole field
            # (a scores() failure means no recall at all) — but NEVER
            # silently: logged, and the ledger says the stack is gone.
            brain._log_error('laf_moment_stack', e,
                             'get_conversation failed — recalling without '
                             'moment context')
            ledger['error'] = str(e)
            return [], ledger
        ledger['rows'] = len(rows)
        turns = []                       # [[op_row_or_None, anchor_row_or_None]]
        for r in rows:
            # The moment stack is operator↔Anchor pairs — pinned like the
            # vector matrix above (_refresh_traces), or a dial-on delivery
            # row would occupy a vec-less j-slot and displace real turns.
            # Own ledger key: a correspondent drop removes the ROW, while a
            # machine drop keeps its j-slot ([None, None]) — different shapes.
            if r.get('ref_type') and r['ref_type'] not in OPERATOR_DIALOGUE_REF_TYPES:
                ledger['correspondent_dropped'] = (
                    ledger.get('correspondent_dropped', 0) + 1)
                continue
            role = r.get('role')
            if role == 'user':
                if is_machine_turn(r.get('content')):
                    ledger['machine_dropped'] += 1
                    turns.append([None, None])   # slot occupied, op side dropped
                else:
                    turns.append([r, None])
            elif role == 'assistant':
                if turns and turns[-1][1] is None:
                    turns[-1][1] = r
                else:
                    turns.append([None, r])      # orphan assistant (session tails)
        if turns and turns[-1][1] is None:
            turns.pop()                          # the live-edge rule (docstring)
            ledger['live_edge_dropped'] = 1
        stack = []
        for j, (op, anchor) in enumerate(reversed(turns[-K:]), start=1):
            for side, r in (('o', op), ('a', anchor)):
                if r is None:
                    continue
                idx = self._tr_ids.get(r.get('trace_id'))
                vec = self._tr_vec(idx) if idx is not None else None
                if vec is None:
                    ledger['missing_vec'] += 1
                stack.append((side, j,
                              vec, (r.get('content') or '')[:MOMENT_TEXT_CAP]))
        ledger['turns'] = min(K, len(turns))
        return stack, ledger

    # ── the scorer ──
    def _refresh(self, brain, model):
        """Bring every cache current — the per-call staleness sync. Caller
        holds self._lock."""
        self._refresh_matrices(brain, model)
        # titles returns the change_key it probed (None on TTL skip) so
        # both node-derived caches pay one probe per expiry window
        node_key = self._refresh_titles(brain)
        self._refresh_projects(brain, node_key=node_key)
        self._refresh_traces(brain)

    def warm(self, brain, model=None):
        """Build the caches before the first query — Brain.warm_up calls
        this at daemon boot, so the full matrix build (~0.8 s at 12.5k
        nodes) is not paid inside the first recall. `model` must be the
        value recall passes: the matrices are not keyed by model. Returns
        the resident node-row count."""
        with self._lock:
            self._refresh(brain, model)
            return self._n

    def scores(self, brain, query, query_vec, model=None, session_project=None,
               as_of=None, session_id=None):
        """({node_id: score01}, telemetry) over every node with ≥1 embedding view.

        score01 is a monotonic sigmoid of the z-scored gain-weighted field sum,
        guaranteed ∈ (0,1) (the range contract brain_recall's floors/boosts
        assume); telemetry is {node_id: {field: z}} for the top nodes — the
        production feed for the P2 dataset walker.

        session_project: the calling session's derived project (ctx.project) —
        the query-side source for the proj lane. None/'' → lane inert.

        as_of: ISO-8601 UTC timestamp (the brain's iso_now shape) — score
        against corpus state at that instant (§20.11 read-side time travel):
        nodes/traces created after it contribute nothing, z-stats run over the
        masked universe, and masked nodes are absent from the result. None
        (default) builds no masks — the identical live path. scores() is
        read-only by construction, so replay needs no side-effect suppression.

        session_id: the moment stack's turn source (§20.17/§20.18) — with
        moment_K>0 in config, the session's last-K turns contribute per-slot
        content lanes gained by the moment_gains table. None, moment_K=0, or
        an empty table → no stack is pulled (the dormant default). The last
        stack's coverage ledger is kept on self._last_moment_ledger (the V2
        moment-mass / W1 freshness instrument).
        """
        qv = _unit(query_vec)
        if qv is None or not MAXSIM_VIEWS:
            return {}, None
        if as_of is not None:
            as_of = str(as_of)
            if 'T' not in as_of:
                # lexicographic masks silently corrupt on a non-ISO shape —
                # refuse loudly instead
                raise ValueError('as_of must be an ISO-8601 timestamp, got %r'
                                 % as_of)
        cfg = self.config(brain)
        with self._lock:
            self._refresh(brain, model)
            n = self._n
            if n == 0:
                return {}, None
            node_mask = trace_mask = None
            if as_of is not None:
                node_mask, trace_mask = self._asof_masks(as_of, n)
                if not node_mask.any():
                    return {}, None
            fields = self._fields(brain, query, qv, cfg, n,
                                  session_project=session_project,
                                  as_of=as_of, trace_mask=trace_mask)
            # Moment slot lanes (§20.17/§20.18) — DORMANT until the K-store
            # flips moment_K + moment_gains. Each stack entry contributes the
            # content-lane triple through the SAME _content_lanes the j0 query
            # uses, named '{lane}_{side}{j}'; only slots present in the gain
            # table are added (an arm is a gain table, not code).
            mg = cfg.get('moment_gains') or {}
            m_k = int(cfg.get('moment_K') or 0)
            self._last_moment_ledger = None
            if mg and m_k > 0 and session_id:
                stack, m_ledger = self._moment_stack(brain, session_id, m_k,
                                                     as_of=as_of)
                self._last_moment_ledger = m_ledger
                for side, j, mv, text in stack:
                    if not any('%s_%s%d' % (l, side, j) in mg
                               for l in ('maxsim', 'sit', 'idf')):
                        continue
                    for lane, vec in self._content_lanes(
                            text, mv, n, as_of=as_of).items():
                        name = '%s_%s%d' % (lane, side, j)
                        if name in mg:
                            fields[name] = vec
            # THE node-mask chokepoint (§20.11 #1/#6): one masked z-score
            # loop covers every node-indexed lane, including future ones.
            # support-z is lane-gated: only the zero-sea lanes take it —
            # matched on the lane PREFIX so slot instances (idf_a3) inherit
            # their base lane's normalizer; contract-zero lanes (proj) keep
            # plain z — see Z_NORMS block.
            zn = str(cfg.get('z_norm', 'current'))
            zf = {name: zscore_variant(
                      vec, n, mask=node_mask,
                      kind=(zn if zn != 'support'
                            or name.split('_', 1)[0] in SUPPORT_ZERO_SEA_LANES
                            else 'current'))
                  for name, vec in fields.items()}
            # Gain resolution: the moment table (when active) is the
            # composition — slot keys verbatim; '{lane}_o0' overrides the
            # current-message content lanes; bare 'pick'/'enc'/'proj' override
            # those lanes. Anything absent falls back to the production
            # gain_* — so an o0-only table IS the fitted-K0 arm (A0f) and an
            # empty table is bit-identical production (the dormancy invariant).
            zsum = np.zeros(n)
            for name, z in zf.items():
                if mg and name in mg:
                    gain = float(mg[name])
                elif mg and (name + '_o0') in mg:
                    gain = float(mg[name + '_o0'])
                else:
                    gain = float(cfg['gain_' + name])
                zsum = zsum + gain * z
            s01 = 1.0 / (1.0 + np.exp(-zsum / float(cfg['sigmoid_scale'])))
            if not np.all(np.isfinite(s01)):
                # out-of-contract vector — refuse to ship it; the caller
                # reports the failure and returns an empty recall
                raise ValueError('laf_v1 produced non-finite scores')
            if node_mask is None:
                score_map = {nid: float(s01[i])
                             for i, nid in enumerate(self._master[:n])}
                top = np.argsort(-s01)[:int(cfg['telemetry_top_n'])]
            else:
                score_map = {self._master[i]: float(s01[i])
                             for i in np.flatnonzero(node_mask)}
                top = [i for i in np.argsort(-np.where(node_mask, s01, -np.inf))
                       [:int(cfg['telemetry_top_n'])] if node_mask[i]]
            telemetry = {self._master[i]: {name: round(float(z[i]), 3)
                                           for name, z in zf.items()}
                         for i in top}
            return score_map, telemetry


_ENGINE_LOCK = threading.Lock()


def get_engine(brain):
    """The per-Brain engine singleton (lazily attached). Creation is locked:
    the daemon's boot warm and a first recall can arrive together, and two
    engines would each pay the full matrix build."""
    eng = getattr(brain, '_laf_engine', None)
    if eng is None:
        with _ENGINE_LOCK:
            eng = getattr(brain, '_laf_engine', None)
            if eng is None:
                eng = LafV1Engine()
                brain._laf_engine = eng
    return eng
