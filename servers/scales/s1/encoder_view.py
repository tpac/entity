"""Encoder view policy — what the S1 Scribe is SHOWN, not what is captured.

POLICY, not mechanism: pure predicates + constants over plain values — no I/O,
no brain, no env reads beyond the flag readers. encode.py and
encode_contract.py render; this file decides what the render feeds. The
boundary rule: numbers shared with other consumers stay in encode_contract
(brain_remember reads ENCODING_AGENT too); single-consumer feeding decisions
live here.

This module shapes the encoder view. Explicit capture exclusions are owned
by the shared action_policy, applied at the daemon write door; other traces
remain available to the dashboard and episodic reads. Every filter marks itself in place (a stubbed
<actions> says "trimmed", an aged catalog entry says how to expand), so
omissions remain distinguishable from absence of activity. Git exclusion and
the total action ceiling also apply when the view-policy flag is off.

Flag: BRAIN_S1E_VIEW_POLICY, ON by default (arm D activated 2026-08-18 on an
operator gate). Set to 0 for the emergency off-switch and the A/B control arm
(eval/encoder_prompt_ab.py renders both arms per capture). Off is the
pre-policy render minus the retired encoded-turn trim.

Measured basis (re-measured 2026-08-16 at 80-89% catalog share):
a catalog node renders at ~4-5K chars of which content is only ~22% — edges
and heavy corrections dominate. So aging drops edges + heavy corrections and
keeps the content WHOLE; cutting content saves almost nothing and a truncated
body is what let the encoder rewrite nodes from fragments — 6/36 runs revised
an aged entry from its visible head, destroying the tail, even under a prompt
that said to expand first. Body-whole makes that failure
structurally impossible.
"""
import os

from servers.contract import META_DEEP, META_NONE, NODE_REF_SEP


def view_policy_enabled():
    """The view-policy flag (see module docstring). run_encoding resolves it
    once per run and threads it down — no torn state if the env flips
    mid-run. Default ON — unset means the policy."""
    return os.environ.get('BRAIN_S1E_VIEW_POLICY', '1') in ('1', 'true', 'True')


# ── Associated stubs (the encoder's subconscious) ──

def associated_stubs_enabled():
    """BRAIN_S1E_ASSOCIATED_STUBS: render the subconscious — nodes production
    recall ranked near this window that didn't make the surface cut — as the
    catalog's last entries, tagged [associated] (encode._associated_stub_ids →
    build_node_catalog). Default OFF (hard law: input changes ship flag-gated
    so the eval can A/B them — the view-policy flag is the precedent); the
    daemon reads the env at start via brain-env.sh."""
    return os.environ.get('BRAIN_S1E_ASSOCIATED_STUBS', '0') in ('1', 'true', 'True')


# K: stubs rendered per run. Small by design — full format_node bodies
# (load-bearing for patch-mode revise), so K is the token knob.
ASSOCIATED_STUBS_K = 5

# Per-seed recall fetch BEFORE catalog exclusion — must exceed the
# catalog-hot head or novel candidates starve (measured in
# eval/encode_moment_recall_probe.py; needle at rank 2 with this limit).
ASSOCIATED_RECALL_LIMIT = 40

# Chars of a window message used as the recall seed — the seed is the
# message, not the essay (embedding sanity; probe's QUERY_CAP).
ASSOCIATED_QUERY_CAP = 1500

# Most seeds paid for per run, keeping the NEWEST (the tail turns are the
# encoder's actual working material). Bounds the sequential recall loop when
# every turn reads unencoded (first encode of a long session, or a degraded
# trace join) — the one dimension the other three constants don't cap.
ASSOCIATED_SEED_CAP = 20


# ── Catalog render: the two tiers ──

# Corrector content cap for the ⚠ blocks (render_rich_node's default is
# max(metadata_limit, 400)). 13% of `⚠ Updated by` bodies were hitting 400 —
# that block says THIS node has been superseded, so a cut body is cut exactly
# where the encoder needs to know what replaced it.
CORRECTION_CONTENT_LIMIT = 800

# Full depth, no truncation. Correction render: 'heavy' — full corrector
# content + reasoning + their_raw_quote. The 2026-05-17 three-way A/B (lean vs
# balanced vs heavy) on 3 items showed heavy was actually CHEAPER than balanced
# on both encoder time (+8.1% vs lean, vs balanced's +11.4%) AND answerer tokens
# (+1.0% vs balanced's +17.5%) at equal pass rate. Correction context is
# targeted signal, not noise — heavy lets the encoder converge faster, not
# flood it. The locked principle "more signal ≠ better encoding" still holds
# for ARBITRARY signal flooding; correction-aspect context is the structured
# exception that demonstrably helps.
# Healer also uses 'heavy' (set in HealerEncoder._format_batch).
# Surface stays lighter — latency-critical: 'lean' for the Haiku menu and the
# inject's seeds, 'none' for background seeds (only the opt-in full
# HAIKU_FORMAT renders 'balanced').
S1_NODE_CONFIG = {
    'content_limit': None,         # full content — no truncation for encoder
    'edge_limit': 5,               # structural edges per node (with descriptions)
    'correction_render': 'heavy',
    # encoding_source is a technical attribution field (encoder:sonnet / anchor /
    # s2:*) — noise to the encoder, which shouldn't reason about who wrote a node.
    # render_rich_node defaults show_encoding_source=True, so hide it explicitly.
    'show_encoding_source': False,
    # Placement as context, titles only: the encoder reads which communities
    # a node sits in but is handed no id to link to — community nodes stay
    # out of the catalog and S2CE owns placement.
    'communities': 'title',
    # Declared, not inherited: this is the view the encoder copies `old` out
    # of to author a swap, so it must not disagree with the get_nodes drill-in
    # it is told to expand with (GET_NODES_DETAIL_FORMAT, same 600). A value
    # cut by the render can never match the stored one — the encoder copies
    # the ellipsis with it and the whole revise op is refused (0682f808).
    'metadata_limit': META_DEEP,
    # The OFF arm keeps a corrector's relation in Edges as well as in its ⚠
    # line — the pre-policy render the A/B control reproduces. The policy's
    # full tier drops the repeat (CATALOG_TRIM_CONFIG).
    'correction_dedup': False,
}

# The 2026-09-16 trim and shape, applied over S1_NODE_CONFIG for the FULL tier
# — flag-gated like every other input change, so the OFF arm stays the A/B
# control (module docstring's first invariant). Measured per field on 26
# captured payloads: a full entry is 4,716 chars, of which content is 1,586 and
# edges 1,240 — both keep their policies. What changes:
CATALOG_TRIM_CONFIG = {
    # Three fields out, 333 chars/entry. `situation` and `question` are
    # retrieval surfaces — they earn their keep in the embedding, not in the
    # encoder's view: it WRITES them on every new node and revises them only
    # when the meaning changed, so rendering them bought a 55% edit rate that
    # measurably degraded them (state tails displacing trigger register).
    # `communities` is placement the encoder is handed no id to act on.
    # BOTH raw quotes stay. their_raw_quote carries words the encoder did not
    # write; my_raw_quote can go stale like any other claim and no S2 unit
    # repairs it — if the encoder cannot see it, nothing ever fixes it.
    'extra_skip_keys': ('situation', 'question'),
    'communities': None,
    # One rendered line per field of the write that repairs the edge —
    # relation + target, then `description:` carrying it whole on its own line,
    # so the encoder reads {target, relation, description} off the lines instead
    # of extracting them from prose.
    'edge_style': 'recall',
    # The canonical id token everywhere in this payload: "title" (id:x · …).
    # Other readers keep ', ' until they move — see NODE_REF_SEP_LEGACY.
    'node_ref_sep': NODE_REF_SEP,
    # A ⚠ line names a node that replaced, corrected, resolved or reframed
    # THIS one — a constraint on the write about to happen, so it renders
    # whole. A verb with no orientation yet names no side and renders as the
    # pointer alone. (The nodes this one corrected are history and ride in
    # Edges.)
    'correction_render': {'corrector': 'heavy'},
    'correction_content_limit': CORRECTION_CONTENT_LIMIT,
    # A corrector's relation shows once, in its ⚠ line, and spends no edge slot.
    'correction_dedup': True,
}


# The full tier under the flag, resolved HERE rather than merged at the call
# site: an overlay could let CATALOG_TRIM_CONFIG reach the aged tier and hand
# its ⚠ blocks the heavy corrector body back. One resolved config per tier
# makes that unrepresentable; build_node_catalog merges only runtime values.
CATALOG_FULL_CONFIG = {**S1_NODE_CONFIG, **CATALOG_TRIM_CONFIG}


# Stated once on the catalog header, because edge_style='recall' carries
# direction on the arrow instead of repeating the word `this` per edge line.
EDGE_ARROW_LEGEND = '  —  edges: → reads "this <relation> …" · ← reads "… <relation> this"'


# ── Catalog aging (the ~80-89% lever) ──

# Newest N encode rounds render full depth: the encoder keeps seeing complete
# bodies for what it wrote most recently (the quality-bar feedback loop).
# Everything older trims to a stub.
CATALOG_FULL_ROUNDS = 2

# The aged entry: id + type + title (recognition + dedup key), the WHOLE
# content, one ⚠ correction line. NO edges (the render announces the count),
# NO situation/reasoning/quotes/KV — the whole body is the "does this already
# exist" anchor, and it is never cut. content_limit None is
# the policy: a complete body means a revise can never rewrite a node from a
# fragment it mistook for the whole. Trimming is reversible: get_nodes is in
# ENCODING_TOOLS, so the encoder can expand any id on demand. The eval harness
# overrides content_limit per arm to reproduce the retired truncating arms.
AGED_NODE_CONFIG = {
    'content_limit': None,         # body stays whole — the arm-D invariant
    'edge_limit': 0,               # the heaviest render weight — dropped
    'correction_render': 'lean',   # one ⚠ header line, not the corrector body
    'metadata_limit': META_NONE,    # no reasoning / quotes / generic KV
    'show_encoding_source': False,
    # Same id token as the full tier — two shapes inside one payload is the
    # defect the canonical token exists to remove.
    'node_ref_sep': NODE_REF_SEP,
    # situation renders top-level, so metadata_limit=0 does not reach it —
    # skipped explicitly, same ruling as the full tier. An aged entry is now
    # title + whole content + one ⚠ line; the whole body is the dedup anchor.
    'extra_skip_keys': ('situation',),
}

# No tag on aged entries: with the body whole, everything an
# aged entry withholds announces itself in place — render_rich_node's
# "Edges (N, not shown — get_nodes for them):" line — so a marker would add
# nothing the encoder can't see.

# Catalog header time render: relative with sub-day steps ('25m ago', '3h ago')
# — the encoder's questions are recency-shaped and mid-session the hour is the
# signal ("revised 20m ago" = my own recent write; an absolute date can't say
# that). Matches surface's time vocabulary (one system, one clock). Side
# effect by design: relative mode suppresses render_rich_node's duplicate
# absolute `Created:` line. The caller supplies `time_now` (conversation time
# — replay-safe) and `this_session_ids` (the ids this session WROTE: encoded ∪
# authored; reads deliberately don't qualify — reading isn't ownership).
CATALOG_TIME_CONFIG = {'time_format': 'relative', 'time_fine': True}


def timeline_now_attr(now):
    """The <timeline now="…"> stamp — the absolute anchor that makes every
    relative label in the prompt invertible, and the current-time declaration
    the encoder's date resolution never had before the view policy. `now` is
    conversation time (replay-safe); renders UTC to
    match the Frame's 'Now:' vocabulary. Returns '' when unstampable."""
    try:
        from datetime import timezone
        return ' now="%s"' % now.astimezone(timezone.utc).strftime(
            '%Y-%m-%d %H:%M UTC')
    except Exception:
        return ''


def aging_cutoff(run_stops, full_rounds=CATALOG_FULL_ROUNDS):
    """The stop at/after which catalog ids stay full: the Nth-newest encode
    run's stop. None = no aging yet (fewer than N runs this session — nothing
    is "before the last N rounds")."""
    rs = sorted(set(run_stops or ()))
    if len(rs) < full_rounds:
        return None
    return rs[-full_rounds]


def catalog_view(ids, stops, run_stops, protected=(), cutoff=None):
    """Order + tier the catalog: returns (ordered_ids, aged_id_set).

    ordered_ids — oldest→newest by each id's last-touched stop, so the most
    recent encodes sit last, adjacent to the timeline. Ids with no known stop
    sort last. Ties break on id for a deterministic render. Surfaced ids
    carry their turn's stop from the surface trace and sort by real recency;
    the caller protects the in-window ones from aging explicitly.

    aged — ids last touched strictly before the cutoff, minus `protected`
    (ids surfaced for the current window: the encoder's likeliest dedup/revise
    targets keep full bodies even when their last write is old).

    `cutoff` — an explicit turn number replacing the round-based one. Passing
    the first turn the timeline renders aligns the full-depth catalog with the
    conversation window: ONE knob widens both, where the round-based cutoff
    lets them drift apart (a 10-turn window against a 42-round catalog).
    None → derived from run_stops."""
    stops = stops or {}
    if cutoff is None:
        cutoff = aging_cutoff(run_stops)
    aged = set()
    if cutoff is not None:
        protected = set(protected or ())
        aged = {i for i in ids
                if i not in protected and stops.get(i, cutoff) < cutoff}
    ordered = sorted(ids, key=lambda i: (stops.get(i, float('inf')), i))
    return ordered, aged


# ── Actions (the <actions> block inside timeline turns) ──

# Brain node-op tools whose lines leave <actions>,
# split by what the drop would lose:
#   DROPPED — the turn's <provenance> already carries everything actionable
#     (verb + ids + titles for writes; for get_node[s] the arguments ARE
#     the ids provenance shows). brain_batch is dropped WITH the known cost
#     that its batched connect sub-ops vanish (accepted: the timeline is not
#     the edge ledger).
#   STUBBED — search tools render a trimmed line keeping the QUERY head:
#     provenance shows result ids only, so a full drop loses the search intent
#     — and a search that found nothing would vanish entirely, though "Anchor
#     looked for X and the brain had nothing" is encodeable signal.
# Edge tools (connect / disconnect / revise_edge) stay fully VISIBLE:
# anchor_touched is node-ids only, deliberately — never edges. Unknown tools
# default to visible.
DROPPED_ACTION_TOOLS = frozenset({
    'remember', 'remember_batch', 'revise', 'revise_batch', 'brain_batch',
    'get_node', 'get_nodes',
})
STUBBED_ACTION_TOOLS = frozenset({
    'recall', 'recall_batch', 'find_node_by_title', 'filter_nodes',
})

# Kept head of a stubbed search line — enough for the query, not the args blob.
ACTION_STUB_HEAD = 60

# Action presentation tuning; shared user settings live in action_policy.
ACTION_BUDGETS = {'thin': (5, 10), 'balanced': (15, 30)}
ACTIONS_KEEP_LAST = 2
ACTIONS_BUDGET_SOFT_EDGE = 2
ACTION_LABEL_CAP = 180
COMMENT_SCAN_DEPTH = 8
PATH_KEEP_SEGMENTS = 3
ROLLUP_TARGET_CAP = 8
ROLLUP_TOOLS_CAP = 6
ROLLUP_SUBS_CAP = 4

# Only supported recorded vocabulary versions may drive normalized behavior.
SUPPORTED_ACTION_VOCAB_VERSIONS = frozenset({1})


def action_mode(tool_name):
    """'full' | 'stub' | 'drop' for a tool_result line. Keys on the raw tool
    name the trace metadata carries (`mcp__<server>__<tool>` — post_tool_trace
    records the host's name verbatim); which names are the brain's own is
    dispatch_common.is_brain_tool's call, on either host. Non-brain and unknown
    tools render full."""
    from ...dispatch_common import is_brain_tool
    name = str(tool_name or '')
    if not is_brain_tool(name):
        return 'full'
    tool = name.split('__')[-1]
    if tool in DROPPED_ACTION_TOOLS:
        return 'drop'
    if tool in STUBBED_ACTION_TOOLS:
        return 'stub'
    return 'full'


def action_stub(summary):
    """The trimmed line for a stubbed search action: bare tool name + the
    query head + where the results went. Whitespace-collapsed; the caller
    XML-escapes (the pointer says 'provenance' in words so escaping can't
    mangle it)."""
    s = ' '.join(str(summary or '').split())
    tool, _, args = s.partition(': ')
    base = tool.split('__')[-1]
    head = args[:ACTION_STUB_HEAD] + ('…' if len(args) > ACTION_STUB_HEAD else '')
    return '%s: %s → results in provenance' % (base, head)


def actions_stub_line(n_actions):
    """The <actions> body for an already-encoded turn. The element renders with
    this stub rather than disappearing — absence would read as "nothing
    happened this turn"; the stub states the filter. First
    person — the prompt speaks as the encoder, and the reader of this line IS
    the one who read them last run."""
    return ('trimmed — %d action(s) recorded on this turn; I already read '
            'them in a previous run' % n_actions)


# ── Provenance verb split ──

# Node-op categories rendered per turn when the policy is on, replacing the
# merged `encoded(me)` (= created∪revised) with the verbs, plus the
# categories the merged form dropped. Labels use the timeline's identity
# vocabulary — (me), matching <me>/<other>. Each entry
# is (label, link-keys): recalled(me) merges the by-id reads (`recalled`,
# catalog-folded) with the search-tool results (`looked_up`, provenance-only)
# into ONE line — the reader's question is "what did I look at", not which
# tool answered it. Keys match nodes_for_traces' link dict; the substrate
# (anchor_touched → ANCHOR_TOUCHED_KEYS) has kept them separate all along.
# Third element: render the node's TITLE beside its id, or the id alone.
# Measured 2026-09-16 over 26 captured payloads — how often a provenance id
# also has a full catalog entry above, which is the only reason to repeat its
# title here: revised 100%, encoded 94%, created 94% — but recalled only 43%
# and surfaced 64%. So the write verbs point at bodies the reader already has
# and cost nothing but the id; the read verbs point at nodes that are often
# nowhere else in the prompt, and a bare id there is the dangling reference
# that makes a reader mint a twin instead of cross-referencing.
PROVENANCE_SPLIT = (
    ('created(me)',  ('created',),                 False),
    ('revised(me)',  ('revised',),                 False),
    ('recalled(me)', ('recalled', 'looked_up'),    True),
    ('archived(me)', ('archived',),                True),
)

# Attribution speaks TURN COORDINATES — the same axis the timeline's real turn
# numbers use (a turn's chain stop; view policy renders <turn n="37"> with the
# session-global number, not a window ordinal). 'encoded(me, turn 36)' against
# a window of turns 37-46 lets a stateless reader compute where it is, how far
# apart its runs land, and which entries are old — no internal names (scribe,
# Anchor, S1S) and no gloss needed. Turn numbers are structural (chain stops),
# so unlike relative ages they stay honest in eval replays.


def encoded_run_label(run_stop=None):
    """Run attribution for the provenance frontier line. With the stop: turn
    coordinates. Without (odd-shaped chain): the plain first-person allusion —
    'my earlier run' is the stub sentence's vocabulary, already grounded."""
    return ('encoded(me, turn %d)' % run_stop) if run_stop is not None \
        else 'encoded(my earlier run)'


def provenance_tag_view(key, stop=None):
    """Catalog provenance tag, view arm — same verbs as the control arm's tags
    (authored / recalled / encoded), first person, with the turn coordinate
    when known: '[encoded(me, turn 36)]'. Keys + priority order stay
    encode_contract.PROVENANCE_TAGS's; surfaced stays untagged on both arms."""
    return ('[%s(me, turn %d)]' % (key, stop)) if stop is not None \
        else '[%s(me)]' % key
