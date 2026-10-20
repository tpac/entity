"""Encoding Contract — S1 turn encoder (Sonnet) config and catalog building.

The encoding agent reads conversation turns and creates/revises brain nodes.
This contract defines:
- What the encoder sees (ENCODING_AGENT config)
- How the node catalog is built (build_node_catalog)

Node formatting uses render_rich_node() from servers.contract.
Interactions: `s1e` (the system prompt, encoding_prompt.py) and `s1e_gist`
(the pre-timeline rules, encoding_gist_prompt.py) — both learnable, each read
once per run through get_interaction_effective; their config defaults live
here.
"""

import os
import sys

from servers.contract import render_rich_node

ENCODE_EVERY_DEFAULT = 5


def _resolve_encode_every() -> int:
    """The cadence knob's value. Env-first (brain-env.sh sources
    ~/.config/brain/env into the daemon's environment), so an operator sets it
    without touching the code. A non-positive or malformed value warns and
    falls back — 0 would fire the Scribe on every turn and a negative one
    on none, and a typo in a hand-edited config must not silently retune
    encoding."""
    raw = os.environ.get('BRAIN_ENCODE_EVERY', '').strip()
    if not raw:
        return ENCODE_EVERY_DEFAULT
    try:
        turns = int(raw)
    except ValueError:
        turns = 0
    if turns < 1:
        sys.stderr.write(
            "[encode-contract] WARN: BRAIN_ENCODE_EVERY=%r is not a positive "
            "integer — using %d\n" % (raw, ENCODE_EVERY_DEFAULT))
        return ENCODE_EVERY_DEFAULT
    return turns


# How many conversational turns accumulate before the S1 Scribe fires. The gate
# is a LEVEL trigger on turns-since-last-encode (read live from traces), not a
# modular counter — a skipped run (lock busy) isn't lost; the next turn re-checks.
# Operator knob: BRAIN_ENCODE_EVERY in ~/.config/brain/env, read once at import
# (daemon restart to apply). The encoder's window holds every turn since the
# last encode (encode._window), so a larger cadence costs window size, not
# coverage, up to ENCODING_AGENT['window_max_turns'].
ENCODE_EVERY = _resolve_encode_every()

# turns-since-last-encode at/above this means the Scribe is WEDGED: it should have
# fired at ENCODE_EVERY but the backlog kept growing (lock jammed, or runs erroring
# before they write their encoding_prompt trace). The gate logs a loud error here —
# this is the monitor that would have caught the 20h encode-drought on hour one.
# 4× the cadence: well past normal, so a rapid burst mid-run can't trip it.
SCRIBE_STARVATION_TURNS = 4 * ENCODE_EVERY


def scribe_is_starved(turns_since: int) -> bool:
    """True when the Scribe is wedged and the gate should emit a loud signal.
    Level condition, rate-limited to one alert per cadence of continued starvation
    (fires at 20, 25, 30… not every turn) so the error log isn't spammed. The
    floor of 2 matters because the cadence is operator-settable: at
    BRAIN_ENCODE_EVERY=1 the modulo is true for every turn, and the rate limit
    would flood the error log instead of bounding it."""
    return (turns_since >= SCRIBE_STARVATION_TURNS
            and turns_since % max(ENCODE_EVERY, 2) == 0)


# ── Idle-tail trigger (the second clause of the Scribe gate) ──
# A session that goes quiet below the ENCODE_EVERY threshold would otherwise
# never have its last turns encoded (the 5+ trigger never fires, and the Stop
# hook is gone). The poll-driven reactor catches that tail: once a session has
# been idle this long AND has at least SCRIBE_TAIL_MIN_TURNS unencoded turns,
# fire one final encode. An hour is long enough that it only ever fires on a
# genuinely abandoned session; the turn floor skips a lone leftover turn, not
# worth a Sonnet call.
SCRIBE_TAIL_IDLE_SECONDS = 3600
SCRIBE_TAIL_MIN_TURNS = 2

# The 5+ clause fires only for a session whose last turn is within this window —
# i.e. one ACTIVELY conversing. Without it, the poll would `5+`-encode every
# recently-present session (90-min window) regardless of how long ago it went
# quiet — a backlog sweep, worst on restart. A session that goes quiet below the
# active window waits for the 1h idle tail instead (and re-qualifies for `5+`
# the moment it takes another turn). This is the "is this a live conversation?"
# bound the old Stop-hook trigger had implicitly (it only fired on a turn).
SCRIBE_ACTIVE_WINDOW_SECONDS = 600

# The reactor only sees sessions present within this wall-clock window, so it
# must outrun the tail threshold (1h) — otherwise a session that crossed 1h idle
# would age out before the tail could fire. Set to 5h so a tail survives a normal
# same-day work pause (lunch, a meeting, a short sleep) + a daemon restart, not
# just a bare margin. NOT longer: we stamp writes at now() (transaction-time, not
# the conversation's time), so resurrecting a genuinely-stale conversation would
# date its nodes "today" — the unbounded catch-up is bi-temporal work (deferred).
SCRIBE_CANDIDATE_WINDOW_MIN = 300   # 5 hours

# The reactor re-evaluates every few seconds. A session whose encode FAILS or
# SKIPS never advances the cadence, so it stays "due" — without a guard the poll
# would re-fire it every tick (a tight machine-paced retry the old human-paced
# Stop gate never had). The reactor records each attempt and won't re-fire the
# same session within this cooldown — bounding a failing session to one retry per
# interval, and giving the poll the cheap per-session gate it otherwise lacks. A
# SUCCESSFUL encode clears the entry immediately (the cadence reset already
# prevents a re-fire), so healthy sessions are never throttled.
SCRIBE_RETRY_COOLDOWN_SECONDS = 120

# Consecutive re-fires of the same session WITHOUT the cadence advancing means
# the encode is wedged (crashing before its encoding_prompt trace, or skipping).
# Escalate loudly at this count — the starvation alarm can't see this case (turns
# is frozen at a fixed value, so its `% ENCODE_EVERY` rate-limit never trips).
SCRIBE_MAX_FAILED_RETRIES = 3

# Wall-clock ceiling for one whole S1E run_llm_loop (all rounds + SDK retries +
# the stream fallback). Healthy runs finish in 2-4 min; the multipliers turned
# one stuck stream into a 5.5h hang holding the single-flight lock (2026-07-28).
# Past the deadline the loop raises → the loud failure path records
# encoding_run_failed and the cooldown paces the retry.
SCRIBE_RUN_DEADLINE_SECONDS = 1200

# If an encode permit has been held longer than this, that encode thread is
# presumed hung (blocked read the deadline can't preempt, or a leaked permit).
# The poll can't kill a thread — it logs `scribe_hung` once per incident so the
# outage is visible instead of silent.
SCRIBE_HUNG_ALARM_SECONDS = 1800

# Hard cap on concurrent S1 Scribe encodes across all sessions. Each encode is a
# multi-round Sonnet stream; this bounds (a) concurrent Sonnet load against the
# org's per-model rate bucket (encoders + S2 share the Sonnet bucket) and (b)
# daemon-process contention — GIL, CPU, the ONNX query-embedder recall shares,
# network — that would otherwise inflate the latency-critical recall hook. It
# caps DISTINCT sessions encoding at once, not re-encodes of one: per-session
# single-flight rides on top in the daemon poll (_encode_inflight). Recall (Haiku)
# is a separate rate bucket, so this is about process headroom, not Haiku's API
# limit. 4 = headroom for parallel worktree sessions without threatening recall
# latency (2026-07-28); revisit against the tier's Sonnet ITPM/OTPM if raised.
MAX_CONCURRENT_ENCODES = 4

# ═══════════════════════════════════════════════════════════════
# ENCODING AGENT CONFIG
# ═══════════════════════════════════════════════════════════════

# Encoding agent v3.2 (Sonnet) — split node catalog + timeline with references
ENCODING_AGENT = {
    'message_display_limit': None,    # no additional per-message timeline cut
    # The conversation window (encode._window): every turn since the last
    # encode, plus up to window_context_turns already-encoded turns before
    # them, at most window_max_turns in all. E3 (eval/s1e_window_sizing.py,
    # 233 production encodes 2026-10-12..19): the backlog since the last
    # encode ran p50 5 / p90 17 / p99 38 / max 81 turns; 40 holds 99%, and a
    # bigger backlog keeps the newest turns and logs s1e_window_capped.
    'window_max_turns': 40,
    'window_context_turns': 2,
    'max_rounds': 5,                  # Sonnet API round limit (target: 2-3)
    'max_tokens': 20000,              # output cap per round; thinking counts against it on models that think by default
    'session_context_limit': 800,     # session context chars (additive within session, editable by S2)
}

# Interaction config default for the `s1e` K (the learnable prompt+config
# boundary). Live reads: `effort` maps to the API's output_config.effort,
# `model` picks the encoder model (encode.py). Encoder loop knobs stay in
# ENCODING_AGENT above — code config, not part of the per-version K.
S1E_INTERACTION_DEFAULT = {
    'effort': 'medium',
    'model': 'claude-sonnet-4-6',
}

# Interaction config default for `s1e_gist` — the operating rules restated
# just before <timeline> (template: encoding_gist_prompt.py). `enabled` is the
# one knob: an eval arm or an operator turns the gist off without touching the
# words (override parameters={'enabled': False}); the assembler subscripts it,
# so the config is total by construction.
S1E_GIST_INTERACTION_DEFAULT = {
    'enabled': True,
}


def validate_s1e_gist_config(config):
    """Violations for an `s1e_gist` override config: `enabled` must be a
    bool. A typo'd `"enabled": "false"` is truthy — it would keep the gist on
    while the operator believes it off, which is exactly the silent shape the
    resolver's validator door exists to refuse."""
    enabled = config.get('enabled', True)
    if not isinstance(enabled, bool):
        return ["enabled must be true or false, got %r" % (enabled,)]
    return []

# The encoder's toolset — the brain_mcp tools handed to the S1 encoding
# agent (encode._get_tool_schemas). Contract-owned so the vocabulary guardrail
# (tests/test_teaching_vocabulary_sync.py) and the prompt's tool names bind
# to the same set. Standalone tools outside it (revise_edge, connect, ...)
# are Anchor's and S2's, not the encoder's.
ENCODING_TOOLS = frozenset({
    'remember_batch', 'revise_batch', 'brain_batch', 'connect_batch',
    'recall_batch', 'get_nodes',
})


# ═══════════════════════════════════════════════════════════════
# NODE CATALOG — assembly. What each tier SHOWS is encoder_view's
# (S1_NODE_CONFIG / AGED_NODE_CONFIG); this file builds the block.
# ═══════════════════════════════════════════════════════════════

# Provenance tags for the widened catalog (Piece 3), in PRIORITY order — a node in
# several categories gets the FIRST/highest tag (Anchor's deliberate commit >
# deliberate lookup > a prior run's encode; surfaced is lowest and stays untagged).
# Single source: the encoder prompt's how-to-read names these same labels, so they
# can't drift between the catalog and the prompt. Keys match session_node_ids.
PROVENANCE_TAGS = (
    ('authored', '[anchor-authored]'),
    ('recalled', '[anchor-recalled]'),
    ('encoded',  '[encoded]'),
)

# The subconscious tag: nodes production recall
# ranked near this window that did NOT make the surface cut render as the
# catalog's LAST entries under this tag — same reading protocol as the other
# provenance tags, origin carried by the tag grammar. Not in PROVENANCE_TAGS:
# associated ids are disjoint from the catalog categories by construction
# (recall excludes them), so they never compete for priority.
ASSOCIATED_TAG = '[associated]'


# The one pattern for ids referenced in surface judge_output text. Exact
# 8-char ids only (the id-resolution unification) — shorter fragments are
# corrupted emissions and are not treated as ids. Owned here; encode's
# lived-timeline SURFACED parse imports it so the two can never disagree.
SURFACED_ID_PATTERN = r'id:([a-z0-9_]{8})'


def surfaced_ids_of(judge_outputs):
    """Node ids referenced by the surface outputs (SURFACED_ID_PATTERN) —
    the single owner of that parse. build_node_catalog derives its surfaced
    category from this; encode's associated-stub retrieval derives its
    exclusion set from the same parse so the two can never disagree about
    what the catalog will show."""
    import re
    ids = set()
    for jo in (judge_outputs or ()):
        if not jo or jo == '(no selection)':
            continue
        for match in re.finditer(SURFACED_ID_PATTERN, jo):
            ids.add(match.group(1))
    return ids


def build_node_catalog(judge_outputs, brain, extra_ids=None,
                       scope=None, view_policy=False, now=None,
                       window_first_turn=None, aged_content_chars=-1,
                       associated_ids=None):
    """Build the deduplicated rich-node catalog the encoder dereferences by id.

    Uses system render_rich_node() with S1 config (full rich, corrections heavy).

    Args:
        judge_outputs: list of surface_output strings (one per turn, may be None)
        brain: Brain instance
        extra_ids: optional {'encoded': set, 'authored': set, 'recalled': set} of
            node ids to fold in alongside the Haiku-surfaced ones (Piece 3 — the
            widened catalog, sourced from trace_links.session_node_ids). Each
            category is tagged by provenance so the encoder reads what it already
            wrote/looked-up vs what recall surfaced. None → surfaced-only (the
            flag-off control arm; byte-behavior unchanged). When
            session_node_ids also carried `stops`/`run_stops`, those feed the
            aging below.
        view_policy: catalog aging (encoder_view; resolved once per run in
            run_encoding). ON: entries sort oldest→newest by last-touched stop
            and entries older than the cutoff render body-only — complete
            content, no edges (each announces its count in place — no tag, no
            header), lean corrections — the savings live in the dropped
            surround, reversible via get_nodes. Headers
            render relative fine-grained time ('3h ago') + a `this session`
            ownership mark on ids this session wrote. OFF (default): unsorted
            full-depth render, byte-identical to the long-standing path.
        now: the as-of instant for relative time (conversation time — replays
            must pass it; None → wall-clock). Only read when view_policy is on.
        window_first_turn: the 1-based turn the rendered timeline opens on
            (encode.window_first_turn). Supplied → aging ages by the CHAT
            WINDOW instead of the newest N encode rounds, so widening the
            window widens the full-depth catalog with it. None → round-based.
            Only read when view_policy is on.
        associated_ids: ordered list (rank order) of subconscious stub ids —
            nodes production recall ranked near the window that didn't make
            the surface cut (encode._associated_stub_ids). Rendered as the
            catalog's LAST entries, tagged ASSOCIATED_TAG, always FULL body
            (never aged — a swap needs its verbatim `old`
            copied from what the encoder sees). Deliberate: they fold into
            the returned id set, so the K trace and the S2 blindness check
            see them — they ARE shown to the encoder. None → no stubs
            (byte-identical to the pre-stub render).

    Returns:
        (catalog_text, node_id_set) — formatted catalog + set of IDs rendered.
    """
    conn = getattr(brain, 'conn', brain)  # tests may pass raw conn
    # Node ids are 8-char hex (v29), so these match the full ids the trace
    # streams carry.
    surfaced_ids = surfaced_ids_of(judge_outputs)

    # Provenance tag per id. A node in several categories gets the HIGHEST-signal
    # tag (first assignment wins via setdefault): Anchor's deliberate commit >
    # deliberate lookup > a prior run's encode > Haiku's surface bet (untagged).
    extra_ids = extra_ids or {}
    # The picks the S1R `surface_selected` trace recorded, unioned with what the
    # rendered inject text still shows. The trace is the honest source — the
    # inject render cuts and shrinks, and 37% of window-surfaced ids reached the
    # timeline's provenance with no catalog body because only the render was
    # read (measured 2026-09-16, 180 of 483 across 26 payloads). Untagged and
    # lowest priority, exactly as the parsed set has always been.
    trace_surfaced = set(extra_ids.get('surfaced') or ())
    surfaced_ids = surfaced_ids | trace_surfaced
    tag_for = {}

    def _assign(ids, tag):
        for nid in (ids or ()):
            tag_for.setdefault(nid, tag)   # first (highest-priority) wins

    if view_policy:
        # same keys/priority; first-person tags in TURN coordinates — one axis
        # with the timeline's real turn numbers ('[encoded(me, turn 36)]')
        from servers.scales.s1.encoder_view import provenance_tag_view
        stops_map = extra_ids.get('stops') or {}
        for key, _tag in PROVENANCE_TAGS:  # PRIORITY order (highest first)
            for nid in (extra_ids.get(key) or ()):
                tag_for.setdefault(nid, provenance_tag_view(key, stops_map.get(nid)))
    else:
        for key, tag in PROVENANCE_TAGS:   # PRIORITY order (highest first)
            _assign(extra_ids.get(key), tag)
    _assign(surfaced_ids, '')              # surfaced: lowest priority, untagged (legacy)

    all_ids = set(tag_for)
    if not all_ids:
        return '', set()
    # Widened iff some provenance category landed a (non-empty) tag — derived from
    # the assignment, not a second scan of extra_ids (one source for the keyset).
    widened = any(tag_for.values())

    # Skip community nodes — S2CE manages communities, S1E encodes from conversation.
    # S1E still sees the community node referenced in the timeline but doesn't get
    # its full content here, so it can't revise/correct/connect to a community node
    # instead of its members. Associated ids join the scan: this render is the
    # chokepoint for the invariant — no community body leaves this function
    # regardless of which caller supplied the ids.
    community_ids = set()
    scan_ids = all_ids | set(associated_ids or ())
    placeholders = ','.join('?' * len(scan_ids))
    for row in conn.execute(
            "SELECT id FROM nodes WHERE id IN (%s) AND type = 'community'" % placeholders,
            list(scan_ids)):
        community_ids.add(row[0])

    catalog_ids = all_ids - community_ids

    # Associated stubs join the batched fetch below but stay OUT of catalog_ids
    # — aging is a catalog concern; the stubs only render last and fold into
    # the returned id set. Deduped at construction so the header count matches
    # what renders. Their tag is provenance, so their presence widens the
    # header (the surfaced-only phrasing would lie above a tagged entry).
    assoc_order = [a for a in dict.fromkeys(associated_ids or ())
                   if a not in catalog_ids and a not in community_ids]
    # Counted from what RENDERS, not from the ids we hoped to render — the
    # header is written after the loop (see `_header` below). Two payloads in
    # the 2026-09-16 census claimed 32 and 88 nodes while rendering 5 and 61:
    # the count came from the pre-fetch id set and the loop dropped the rest in
    # silence, so the cause was unrecoverable after the fact.
    def _header(n):
        return ('Node Catalog (%d nodes)' % n if widened or assoc_order else
                'Node Catalog (%d nodes surfaced this session)' % n)
    if view_policy:
        # The arrow legend. Direction used to ride the word `this` inside each
        # edge line, once per edge; it is stated once here instead.
        from servers.scales.s1.encoder_view import EDGE_ARROW_LEGEND
        legend = EDGE_ARROW_LEGEND
    else:
        legend = ''

    # Catalog aging (view policy): order oldest→newest and tier. Surfaced ids
    # for the CURRENT window are protected — the likeliest dedup/revise targets
    # keep full bodies. OFF: `order` stays the raw set (byte-identical render).
    order, aged = catalog_ids, set()
    if view_policy:
        from servers.scales.s1.encoder_view import aging_cutoff, catalog_view
        cutoff = (window_first_turn if window_first_turn is not None
                  else aging_cutoff(extra_ids.get('run_stops')))
        order, aged = catalog_view(
            catalog_ids, stops=extra_ids.get('stops'),
            run_stops=extra_ids.get('run_stops'),
            # Protected = surfaced IN THIS WINDOW. The judge-derived half is
            # window-scoped by construction (it is scraped from the window's
            # own message rows), so it protects whole, as it always has. The
            # trace-derived half is SESSION-scoped — protecting it whole would
            # exempt every node surfaced all session from aging and quietly
            # undo the tier — so it protects only where its stop is in-window.
            protected=(surfaced_ids - trace_surfaced) | {
                i for i in trace_surfaced
                if cutoff is None
                or (extra_ids.get('stops') or {}).get(i, cutoff) >= cutoff},
            cutoff=cutoff)

    lines = [None, '']          # header written last, from what rendered
    formatted_ids = set()
    # One batched fetch (returns {id: node}) — the widened union can be hundreds of
    # ids, and per-id get_node would run correction_enrich + a resolve LIKE-scan
    # each. brain.get_node(list) is the batch form.
    fetch_ids = list(catalog_ids) + assoc_order
    rich_map = brain.get_node(fetch_ids) if fetch_ids else {}
    # Loop-invariant: one scoped cfg per tier for the whole catalog (can be
    # hundreds of nodes), never a per-node clone.
    from servers.scales.s1.encoder_view import S1_NODE_CONFIG
    base = S1_NODE_CONFIG
    aged_cfg = None
    if view_policy:
        from servers.scales.s1.encoder_view import (
            CATALOG_TIME_CONFIG, CATALOG_FULL_CONFIG)
        # Each tier arrives already resolved from encoder_view; only runtime
        # values are merged here — fine relative time and the ownership mark
        # (ids this session WROTE). Flag-gated, so OFF stays the A/B control.
        base = CATALOG_FULL_CONFIG
        own = (extra_ids.get('encoded') or set()) | (extra_ids.get('authored') or set())
        view_cfg = dict(CATALOG_TIME_CONFIG, time_now=now, this_session_ids=own,
                        show_edge_total=True)
    catalog_cfg = dict(base, scope=scope) if scope else base
    if view_policy:
        catalog_cfg = dict(catalog_cfg, **view_cfg)
        if aged:
            from servers.scales.s1.encoder_view import AGED_NODE_CONFIG
            aged_cfg = dict(AGED_NODE_CONFIG, **view_cfg)
            # -1 = use the policy's own cap. None = keep content whole and let
            # the OTHER cuts (edges, corrections, metadata) do the saving —
            # the variant that makes a partial-view revise impossible because
            # there is no truncated body to rewrite from.
            if aged_content_chars != -1:
                aged_cfg['content_limit'] = aged_content_chars
            if scope:
                aged_cfg['scope'] = scope
    dropped = []
    for nid in order:
        node = rich_map.get(nid)
        if not node:
            dropped.append(nid)      # the batch fetch returned no body
            continue
        is_aged = nid in aged
        formatted = render_rich_node(node, aged_cfg if is_aged else catalog_cfg)
        if not formatted:
            dropped.append(nid)      # a body that rendered to nothing
            continue
        tag = tag_for.get(nid)
        lines.append('%s %s' % (tag, formatted) if tag else formatted)
        lines.append('')
        formatted_ids.add(nid)

    # Subconscious stubs LAST — adjacent to the catalog→timeline attention
    # boundary, full body always (never aged: a patch-revise needs the
    # verbatim text the encoder sees). Rank order preserved from retrieval.
    for nid in assoc_order:
        node = rich_map.get(nid)
        if not node:
            dropped.append(nid)
            continue
        formatted = render_rich_node(node, catalog_cfg)
        if not formatted:
            dropped.append(nid)
            continue
        lines.append('%s %s' % (ASSOCIATED_TAG, formatted))
        lines.append('')
        formatted_ids.add(nid)

    # The header names what the encoder can actually dereference.
    lines[0] = _header(len(formatted_ids)) + legend
    # A requested id with no body is a reference the timeline may still name —
    # the dangling-reference class. Never silent: the errors table is the
    # monitored surface, and without this the cause is unrecoverable.
    if dropped:
        try:
            brain._log_error(
                's1e_catalog_dropped',
                RuntimeError('%d of %d catalog ids rendered no body'
                             % (len(dropped), len(order))),
                'ids: %s%s' % (', '.join(sorted(dropped)[:20]),
                               '  +%d more' % (len(dropped) - 20)
                               if len(dropped) > 20 else ''))
        except Exception:
            pass

    return '\n'.join(lines), formatted_ids
