"""trace↔node links — what S1 surfaced and encoded, joined to traces.

Scale: S1 (turn). This composes S1's OWN behavioral output — S1R surface
selections and S1E encode runs — into a per-trace view of which nodes the brain
touched around each turn. It is the S1 sibling of brain_traces.py's
conversation reads: those compose s0 traces into the conversation; this
composes s1 traces into the node-links over that conversation. Recall lives at
the brain level (it spans every scale); surface+encode are about the turn, so
they live here.

It is a consumer-NEUTRAL link, not "provenance" (which is one reader's lens).
The capability returns, per target trace, the nodes linked to it by relation:

    trace_id -> {
        'surfaced':   [node_id, ...],   # recall surfaced AND Haiku picked (S1R)
        'encoded':    [node_id, ...],   # what the owning run wrote    (S1E)
        'encoded_by': trace_id | None,  # the encoding_run trace; None = unencoded
        'encoded_by_stop': int | None,  # that run's stop — the turn it ran on
        'authored':   [node_id, ...],   # what Anchor's own tools wrote (S0, anchor_touched)
        'created':    [node_id, ...],   # authored, split: nodes Anchor created (S0)
        'revised':    [node_id, ...],   # authored, split: nodes Anchor revised (S0)
        'archived':   [node_id, ...],   # nodes Anchor archived this turn (S0)
        'recalled':   [node_id, ...],   # Anchor's by-id reads — per _LOOKUP_KEY (S0)
        'looked_up':  [node_id, ...],   # Anchor's search results — per _LOOKUP_KEY (S0)
        'endo':       [node_id, ...],   # endo-surfaced this turn (S0, empty until wired)
        'dropped':    [node_id, ...],   # offered to Haiku, NOT picked (S1R pool − surfaced)
        'fetched_by': {node_id: tool},  # tool-fetched, ADMITTED to the pool (S1R tool_trace)
        'floored_by': {node_id: tool},  # tool-fetched, floor-rejected — never pooled
    }

    `created`/`revised`/`archived` are the per-verb split of the same
    anchor_touched delta `authored` merges (`authored` = created ∪ revised, kept
    for the readers that only need "Anchor wrote this here") — the encoder's
    <provenance> renders the verbs, so the split rides the same join.

    `fetched_by`/`floored_by` are the tool-provenance tier (K rows carry
    tool_trace with per-call result_ids/dropped_ids from 2026-07-02; older
    traces simply yield empty dicts). Three label tiers for one node at one
    turn: picked (`surfaced`), pooled-but-not-picked (`dropped`, which
    INCLUDES admitted tool fetches — fetched_by says which tool), and
    fetched-but-floored (`floored_by` only — these never reached Haiku, a
    harder negative than dropped).

Two readers, ONE join (do not fork a sibling join per reader — that is how the
2-vs-3-tuple drift happened):
  • The S1 encoder reads a link as "already handled here → revise, don't dupe,"
    and reads `encoded_by is None` as the emergent unencoded boundary.
  • The LAF episodic layers read the SAME link at similar past moments:
    `surfaced` (+act picked), `encoded` (+act learned), `dropped` (−inhibit —
    offered in a situation like this and consistently not chosen).

THE JOIN IS STRUCTURAL, by stop_counter — never by timestamp proximity. Every
chain for turn N ends in `-N` (`s0-{short}-N`, `s1r-{short}-N`, `s1e-{short}-N`),
so `_stop_of` extracts the join key from any of them. surfaced is 1:1 with a turn
(same stop); encode is per-RUN (one run at stop S closes the turn-range ending
at S), so a turn's owning run is the first run whose stop is STRICTLY greater
than the turn's stop (turn chains stamp pre-increment, run chains post-), and
`encoded_by` disambiguates which run / marks the boundary.

TWO LAYERS so it is robust to a raw trace dataset run sequentially:
  • nodes_for_traces(...) — PURE. Plain trace records in, link map out. No brain,
    no DB. This is what eval replay (Frozen Corpus), unit tests, and any
    historical/sequential dataset drive directly; it assumes nothing about storage.
  • gather(brain, session_id, streams=...) — thin live adapter. Pulls the named
    trace streams via the public `query_traces` door (no bespoke DAL) and returns
    a DICT keyed by stream name — never a bare tuple (tuple arity is unauditable
    drift; a named dict extends without breaking any consumer). Eval/tests skip it.

Anchor's own actions (`authored`/`recalled`) come from the S0 `anchor_touched`
delta — a per-turn aggregate the daemon flushes at the Stop boundary (the S0
mirror of the S1 encode delta). It reuses the encode delta's `created`/`revised`
keys, so `_delta_ids` parses both with no second path. NOT sourced from
`encoding_source` (in-flux, encoder-invisible) — the touched delta is captured
structurally at dispatch, where only Anchor's TCP calls flow. `endo` rides the
same delta when endo recall is wired.
"""
import json


def _stop_of(chain_id):
    """The stop_counter that names a chain — the trace↔trace join key.

    Chain ids are `{scale}-{session_short}-{N}`; the trailing segment is the
    stop. Returns the int, or None for a missing/odd-shaped chain (those targets
    simply get an empty link rather than crashing the join)."""
    if not chain_id:
        return None
    tail = str(chain_id).rsplit('-', 1)[-1]
    return int(tail) if tail.isdigit() else None


def display_turn(chain_id):
    """The 1-BASED turn number a turn chain names — the coordinate system the
    encoder-facing renders speak. Turn chains stamp pre-increment (first turn
    = chain-stop 0) while run chains stamp post-increment (a run firing after
    the 5th turn = stop 5, already 1-based) — so chain-stop + 1 puts turns on
    the runs' axis, and 'encode every 5 turns' reads as happening ON turn 5.
    None for odd-shaped chains (callers keep their fallback)."""
    stop = _stop_of(chain_id)
    return None if stop is None else stop + 1


def _surface_ids(ref_id):
    """Node ids a surface_selected trace recorded — its ref_id is a JSON list.
    Malformed → [] (one bad row never sinks the join; the live writer validates
    on its own side, and this layer stays pure with nothing to log to)."""
    try:
        v = json.loads(ref_id or '[]')
        return list(v) if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


def _dedup(seq):
    """Order-preserving dedup — node ids stay in first-seen order (the codebase
    idiom; O(n))."""
    return list(dict.fromkeys(seq))


def _delta_ids(meta, *keys):
    """Node ids out of a delta's id-list fields — the ONE parser shared by the S1
    encode delta and the S0 anchor_touched delta. Both carry the same
    `created`/`revised`/`archived` keys (the symmetry), so neither needs its own
    parsing path; the S0 delta just has extra keys (`recalled`/`endo`) this reads
    on demand. Concatenates the requested keys, order-preserving dedup."""
    out = []
    for k in keys:
        out.extend((meta or {}).get(k) or [])
    return _dedup(out)


def _tool_provenance(meta):
    """(fetched_by, floored_by) from a surface K trace's tool_trace. PURE.

    Walks metadata.tool_trace's per-round tool_calls: `result_ids` are the
    fetched candidates ADMITTED to the pool, `dropped_ids` the floor-rejected
    ones that never reached Haiku. Returns two {short_id: tool_name} dicts.
    Traces older than 2026-07-02 lack the id fields → empty dicts. First
    tool wins on the (rare) same-id-two-tools collision — dedupe order
    matches the call order Haiku issued."""
    if not isinstance(meta, dict):
        return {}, {}
    fetched, floored = {}, {}
    for rnd in (meta.get('tool_trace') or []):
        for call in (rnd.get('tool_calls') or []) if isinstance(rnd, dict) else []:
            tool = call.get('tool') or '?'
            for nid in (call.get('result_ids') or []):
                fetched.setdefault(nid, tool)
            for nid in (call.get('dropped_ids') or []):
                floored.setdefault(nid, tool)
    return fetched, floored


def _candidate_outcomes(meta):
    """(candidate_ids, dropped_ids) from a candidate-pool trace's metadata. PURE.

    Reads, in priority order, the two trace shapes that carry the candidate pool:
      1. Δ `additionalContext` — `outcomes_per_candidate {short_id: verdict}` is
         the authoritative Haiku-resolved pool; candidates = its keys, dropped =
         keys whose verdict == 'dropped'. (Falls back to the `dropped`/`selected`
         mirror lists if outcomes is absent but those are present.)
      2. O `recall` — `candidates ["short_id|title|score|type", ...]` is the raw
         pool with no verdict; returns (candidate_ids, None) — the caller derives
         dropped as candidates − surfaced picks.
    Returns (candidate_ids, dropped_ids_or_None). Empty/odd metadata → ([], None),
    so a malformed row contributes nothing rather than crashing the join."""
    if not isinstance(meta, dict):
        return [], None
    outcomes = meta.get('outcomes_per_candidate')
    if isinstance(outcomes, dict) and outcomes:
        cands = list(outcomes.keys())
        dropped = [cid for cid, v in outcomes.items() if v == 'dropped']
        return cands, dropped
    # Δ trace mirror lists (outcomes absent but selected/dropped present)
    sel = meta.get('selected')
    drp = meta.get('dropped')
    if isinstance(sel, list) or isinstance(drp, list):
        sel = list(sel or [])
        drp = list(drp or [])
        return _dedup(sel + drp), drp
    # O recall trace: "short_id|title|score|type" rows, no verdict
    raw = meta.get('candidates')
    if isinstance(raw, list) and raw:
        cands = [str(r).split('|', 1)[0] for r in raw if str(r).split('|', 1)[0]]
        return _dedup(cands), None
    return [], None


def nodes_for_traces(surface_traces, encode_traces, target_traces,
                     touched_traces=None, recall_traces=None):
    """Join S1 surface/encode + S0 anchor_touched + S1R candidate-pool traces to
    target traces by stop. PURE.

    Args:
        surface_traces: S1R `surface_selected` trace records (chain_id + ref_id).
        encode_traces:  S1E `encoding_run` trace records (chain_id + metadata
                        with `created`/`revised` node-id lists).
        target_traces:  the traces to attach links to — each a record with `id`
                        (the map key) and `chain_id` (the stop to join on).
                        Usually the turns' user_message traces; a recall layer
                        passes embedding-matched traces instead.
        touched_traces: S0 `anchor_touched` trace records (chain_id + metadata) —
                        what Anchor's OWN tools touched. Optional (the feed may not
                        be wired); when omitted, authored/recalled/endo are empty.
        recall_traces:  S1R candidate-POOL trace records — either Δ
                        `additionalContext` rows (metadata.outcomes_per_candidate,
                        the authoritative Haiku-resolved pool + verdict) or O
                        `recall` rows (metadata.candidates, the raw pool). Mixed is
                        fine; per stop the verdict shape wins. Optional; when
                        omitted, `dropped` is empty.

    Returns:
        {target_trace_id: {'surfaced': [ids], 'encoded': [ids],
                           'encoded_by': run_trace_id | None,
                           'authored': [ids], 'recalled': [ids], 'endo': [ids],
                           'dropped': [ids]}}

    surfaced / authored / recalled / endo / dropped are 1:1 with the turn (same
    stop); encoded is per-RUN (the owning run). `dropped` = the turn's candidate
    pool MINUS surfaced (explicit verdict wins over the derived subtraction; a
    node never appears in both — surfaced wins). surfaced/dropped ids are what
    the surface traces record (8-char short ids); encoded/authored are full —
    verbatim either way, width resolution is the consumer's job.
    """
    # surfaced + tool provenance, indexed by stop (1:1 with a turn; merge if a
    # stop ever repeats). The SAME K rows carry both: ref_id is Haiku's picks,
    # metadata.tool_trace is which tool fetched what (and what the floor cut).
    surf_by_stop = {}
    fetched_by_stop = {}
    floored_by_stop = {}
    for t in surface_traces:
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        bucket = surf_by_stop.setdefault(stop, [])
        bucket.extend(_surface_ids(t.get('ref_id')))
        fetched, floored = _tool_provenance(t.get('metadata'))
        if fetched:
            fb = fetched_by_stop.setdefault(stop, {})
            for nid, tool in fetched.items():
                fb.setdefault(nid, tool)
        if floored:
            fl = floored_by_stop.setdefault(stop, {})
            for nid, tool in floored.items():
                fl.setdefault(nid, tool)

    # anchor_touched, indexed by stop (1:1 with a turn, like surfaced). authored =
    # created∪revised (live nodes Anchor wrote), also carried split per verb —
    # created/revised/archived — for readers that render the verbs (the encoder's
    # <provenance>). recalled = deliberate lookups; endo = endo-surface (empty
    # until wired). Shared _delta_ids parser.
    touched_by_stop = {}
    for t in (touched_traces or []):
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        meta = t.get('metadata')
        e = touched_by_stop.setdefault(stop, {'authored': [], 'created': [],
                                              'revised': [], 'archived': [],
                                              'recalled': [], 'looked_up': [],
                                              'endo': []})
        e['authored'].extend(_delta_ids(meta, 'created', 'revised'))
        e['created'].extend(_delta_ids(meta, 'created'))
        e['revised'].extend(_delta_ids(meta, 'revised'))
        e['archived'].extend(_delta_ids(meta, 'archived'))
        e['recalled'].extend(_delta_ids(meta, 'recalled'))
        e['looked_up'].extend(_delta_ids(meta, 'looked_up'))
        e['endo'].extend(_delta_ids(meta, 'endo'))

    # candidate pool + explicit-dropped (when the trace carries a verdict),
    # indexed by stop. A stop may have both an O row (raw pool) and a Δ row
    # (verdict) — keep the union of candidates and any explicit dropped set.
    cands_by_stop = {}
    explicit_dropped_by_stop = {}
    for t in (recall_traces or []):
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        cands, dropped = _candidate_outcomes(t.get('metadata') or {})
        if cands:
            cands_by_stop.setdefault(stop, []).extend(cands)
        if dropped is not None:
            explicit_dropped_by_stop.setdefault(stop, []).extend(dropped)

    # encode runs as (stop, trace_id, [created+revised]), ascending by stop.
    runs = []
    for t in encode_traces:
        stop = _stop_of(t.get('chain_id'))
        if stop is None:
            continue
        ids = _delta_ids(t.get('metadata'), 'created', 'revised')  # shared parser
        runs.append((stop, t.get('id'), ids))
    runs.sort(key=lambda r: r[0])

    out = {}
    for tt in target_traces:
        tid = tt.get('id')
        if tid is None:
            continue
        stop = _stop_of(tt.get('chain_id'))
        surfaced = _dedup(surf_by_stop.get(stop, [])) if stop is not None else []
        encoded, encoded_by, encoded_by_stop = [], None, None
        if stop is not None:
            # owning run = first run that closes a range STRICTLY AFTER this
            # turn. STRICT, not >=: a turn's S0 chain is stamped at the counter
            # value BEFORE the Stop-hook increment ("before increment, so the
            # stop matches" — daemon_hooks), while a run's s1e chain carries the
            # POST-increment counter — so a run at stop S saw exactly the turns
            # with chain-stop < S, never == S. The >= form marked the first turn
            # AFTER each encode as covered (found live: 9-true/1-false where the
            # replay's own count said 2 unencoded).
            # LIMIT (proximity is heuristic): the encoder reads a sliding message
            # window, so a single failed run's turns are normally re-covered by
            # the next run's lookback (correct). But if 2+ consecutive runs fail,
            # a turn can be attributed to a later run that never saw it — a false
            # "already encoded." Traces carry no per-run input-range to fix this
            # precisely; the encoder's "revise if shifted, don't blind-skip"
            # how-to-read is the mitigation. Rare (needs consecutive failures).
            for rstop, rtid, rids in runs:
                if rstop > stop:
                    encoded, encoded_by, encoded_by_stop = list(rids), rtid, rstop
                    break
        tch = touched_by_stop.get(stop) or {} if stop is not None else {}
        # dropped: explicit verdict wins over the derived pool−surfaced; either way
        # a node can never be both surfaced and dropped at one stop — surfaced wins.
        surfaced_set = set(surfaced)
        if stop is None:
            pool = []
        elif stop in explicit_dropped_by_stop:
            pool = explicit_dropped_by_stop[stop]
        else:
            pool = cands_by_stop.get(stop, [])
        dropped = [d for d in _dedup(pool) if d not in surfaced_set]
        out[tid] = {
            'surfaced': surfaced, 'encoded': encoded, 'encoded_by': encoded_by,
            'encoded_by_stop': encoded_by_stop,
            'authored': _dedup(tch.get('authored', [])),
            'created': _dedup(tch.get('created', [])),
            'revised': _dedup(tch.get('revised', [])),
            'archived': _dedup(tch.get('archived', [])),
            'recalled': _dedup(tch.get('recalled', [])),
            'looked_up': _dedup(tch.get('looked_up', [])),
            'endo': _dedup(tch.get('endo', [])),
            'dropped': dropped,
            'fetched_by': dict(fetched_by_stop.get(stop, {})) if stop is not None else {},
            'floored_by': dict(floored_by_stop.get(stop, {})) if stop is not None else {},
        }
    return out


def session_node_ids(encode_traces, touched_traces, surface_traces=None):
    """Session-level UNION of the nodes the brain wrote/Anchor touched — the
    catalog's view (vs nodes_for_traces' per-turn view). PURE. Membership stays
    un-keyed (the widened catalog holds each body once regardless of how often
    an id recurs), but each id's RECENCY is preserved: catalog aging renders the
    newest encode rounds full and trims the older ones, which needs to know
    which stop each id was last written/touched at. Same `_delta_ids` parser as
    everywhere else.

    Returns {'encoded': set, 'authored': set, 'recalled': set, 'surfaced': set,
             'stops': {node_id: last_stop}, 'run_stops': [ascending stops]} —
    `encoded` from the S1 encode runs (created∪revised), `authored`/`recalled`
    from the S0 anchor_touched deltas, `surfaced` from the S1R
    `surface_selected` picks. `stops` maps each id to the NEWEST stop
    that produced it (an id revised across runs ages by its latest touch);
    `run_stops` is every encode run's stop, ascending — the aging cutoff's
    axis.

    `surfaced` used to be omitted here on the reasoning that "the catalog
    already has it from the Haiku judge outputs". That was false for 37% of
    window-surfaced ids (measured 2026-09-16 over 26 captured payloads, 180 of
    483): the catalog's own source regex-scrapes ids out of the RENDERED inject
    text (`delta`/`additionalContext` → metadata['content']), so anything the
    inject render cut, shrank or never wrote is simply absent — while the
    timeline names the same ids in its `surfaced:` provenance, because it reads
    the structured picks. Two sources for one fact, and the lossy one fed the
    bodies. This is that fact's single source now.

    `looked_up` is DELIBERATELY not here either — search-tool
    result pages render on the per-turn <provenance> line but folding them
    into the catalog would flood it (a single recall returns up to a page of
    ids). `endo` is omitted until the stream is wired.

    ONE COORDINATE SYSTEM: `stops` values are 1-BASED turn numbers (the
    `display_turn` axis) — run stops are post-increment (already 1-based);
    touched turn chains normalize through `display_turn`, matching what the
    view-policy timeline displays.
    """
    encoded, authored, recalled, surfaced = set(), set(), set(), set()
    stops, run_stops = {}, set()

    def _mark(ids, stop):
        if stop is None:
            return
        for nid in ids:
            if stop > stops.get(nid, -1):
                stops[nid] = stop

    for t in (encode_traces or []):
        ids = _delta_ids(t.get('metadata'), 'created', 'revised')
        encoded.update(ids)
        stop = _stop_of(t.get('chain_id'))          # post-increment: 1-based
        if stop is not None:
            run_stops.add(stop)
        _mark(ids, stop)
    for t in (touched_traces or []):
        a = _delta_ids(t.get('metadata'), 'created', 'revised')
        r = _delta_ids(t.get('metadata'), 'recalled')
        authored.update(a)
        recalled.update(r)
        turn = display_turn(t.get('chain_id'))      # → the 1-based axis
        _mark(a, turn)
        _mark(r, turn)
    # The S1R picks, from the same structured trace the timeline's `surfaced:`
    # provenance reads — so a body exists for every id the timeline names.
    # Stops are marked too: a node surfaced at turn 40 must age by turn 40, not
    # sort as unknown-and-oldest.
    for t in (surface_traces or []):
        ids = _surface_ids(t.get('ref_id'))
        surfaced.update(ids)
        # display_turn, not the raw stop: an s1r recall chain stamps on the
        # TURN axis (pre-increment), like the s0 touched chains and unlike the
        # run chains. Marking the raw stop would age every surfaced node one
        # turn early — the ONE COORDINATE SYSTEM note above.
        _mark(ids, display_turn(t.get('chain_id')))

    return {'encoded': encoded, 'authored': authored, 'recalled': recalled,
            'surfaced': surfaced,
            'stops': stops, 'run_stops': sorted(run_stops)}


# The streams gather() can pull: name → (ref_type, scale). One registry, so a
# new stream is one line here and zero signature changes anywhere.
GATHER_STREAMS = {
    'surface': ('surface_selected', 's1'),    # S1R picks (the `surfaced` role)
    'encode':  ('encoding_run', 's1'),        # S1E deltas (the `encoded` role)
    'touched': ('anchor_touched', 's0'),      # Anchor's own tool actions
    'recall':  ('additionalContext', 's1'),   # candidate pool + verdict (`dropped`)
}


def gather(brain, session_id, streams=('surface', 'encode', 'touched'),
           limit=500, older_than=None):
    """Live adapter: fetch named trace streams for a session. Thin.

    Composes the public `query_traces` door (no bespoke DAL, per the §10.2
    trace-query law). Returns {stream_name: [trace records]} for the requested
    `streams` — a NAMED dict, never a bare tuple: tuple arity drifts silently
    when two consumers evolve the stream list apart (the 2-vs-3-tuple lesson);
    a keyed dict extends without breaking anyone. Eval/tests bypass this and
    feed records to nodes_for_traces directly.

    Defaults to the encoder's three streams. The episodic layers pass
    ('surface', 'encode', 'recall') — `recall` is the Δ `additionalContext`
    candidate-pool rows carrying `metadata.outcomes_per_candidate` (the
    authoritative Haiku-resolved pool + verdict; `_candidate_outcomes` also
    reads the O `recall` raw-pool shape for datasets that only logged that).

    `limit` is the session trace-pull bound (created_at DESC, so it keeps the
    most-recent traces — what the recent-turn window needs). Distinct from the
    timeline's LIVED_SEQUENCE_PULL; a reusable adapter default, not a per-piece
    constant. A consumer with a different need passes its own.

    `older_than` (ISO, strict `created_at <`) is the replay as-of bound —
    pushed through query_traces into SQL so the DESC LIMIT window sits at
    that instant. A Python post-filter here would clip exactly the rows a
    deep-history replay needs (the fetch-then-filter class).
    """
    out = {}
    for name in streams:
        ref_type, scale = GATHER_STREAMS[name]   # unknown name = caller bug, loud
        out[name] = brain.query_traces(
            ref_type=ref_type, scale=scale,
            session_id=session_id, hours=None, limit=limit,
            older_than=older_than or '').get('events', [])
    return out
