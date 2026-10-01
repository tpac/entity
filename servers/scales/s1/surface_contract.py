"""Surface Contract — S1 surface (Haiku) prompt building, candidate formatting, output formatting.

S1 Surface pushes relevant memories into awareness. This contract defines:
- What the surfacer sees (SURFACE config, CANDIDATE_POOL)
- How candidates are formatted (format_candidate_for_surface)
- How the prompt is assembled (build_surface_prompt)
- How output is formatted for Anchor (render_surface_inject)

Correction enrichment lives on Brain (brain.correction_enrich) — it's a
graph join, not a surface concern. Every canonical node pull (brain.get_node)
attaches `_corrections` automatically.

Interaction: 'surface' in interactions table. Prompt is learnable.
"""

import hashlib
import random
import threading
from datetime import datetime, timezone
from collections import OrderedDict

from servers.contract import META_BRIEF, META_NONE

# Hoisted: any embedder API drift fails at daemon boot, not 16s into hook_recall.
from servers.embedder import embed_batch
from servers import embedder as _embedder

# Bounded cache for edge-description embeddings. Same descriptions recur
# across turns and across sessions — caching eliminates redundant embed calls.
#
# Key: (model_name, description_text). Model-scoped so a model swap doesn't
# hand out vectors from the wrong geometry. Values are L2-normalized blobs.
# LRU eviction via OrderedDict; capped at 5000 entries (~15MB at 3KB/vector).
# All reads/writes guarded by _DESC_CACHE_LOCK so concurrent embed callers
# (future multi-threaded recall/backfill) can't corrupt LRU order.
_DESC_VEC_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_DESC_VEC_CACHE_MAX = 5000
_DESC_CACHE_LOCK = threading.Lock()
# No hit/miss counters: the cache only serves edges whose stored embedding
# is still NULL, so its hit rate says nothing about production recall.


def _desc_vecs_batched(descs):
    """Resolve descriptions → normalized blobs, using cache + one batched embed.

    Returns list of blobs aligned with descs input. Empty strings return None.
    Keys the cache by active model name so model swaps don't hand out stale
    geometry. Dedupes within a single call so a batch with duplicates only
    pays one embed per unique text.

    This function is the FALLBACK for edges whose stored
    `edge_relations.embedding` is still NULL — write paths clear the column
    and enqueue the edge, and the embed_queue worker fills it
    asynchronously. Edges written since the last drain land here; sustained
    heavy traffic means the worker is falling behind or failing.
    """
    if not descs:
        return []

    model = _embedder.stats.get('model_name') or ''
    out = [None] * len(descs)

    # Partition under lock: cached → out[i]; uncached → unique_texts + index map
    # (unique_texts is deduped so duplicate descriptions in one call only embed once)
    unique_texts: list = []
    text_to_unique_idx: dict = {}
    indices_for_unique: list = []  # list of lists of original indices per unique text
    with _DESC_CACHE_LOCK:
        for i, d in enumerate(descs):
            if not d:
                continue
            key = (model, d)
            blob = _DESC_VEC_CACHE.get(key)
            if blob is not None:
                out[i] = blob
                _DESC_VEC_CACHE.move_to_end(key)
                continue
            u_idx = text_to_unique_idx.get(d)
            if u_idx is None:
                u_idx = len(unique_texts)
                text_to_unique_idx[d] = u_idx
                unique_texts.append(d)
                indices_for_unique.append([i])
            else:
                indices_for_unique[u_idx].append(i)

    # Embed outside the lock — embedder call may be slow; we hold no state.
    if unique_texts:
        fresh = embed_batch(unique_texts, kind='document')
        # Populate out[] + cache under lock
        with _DESC_CACHE_LOCK:
            for u_idx, blob in enumerate(fresh):
                if blob is None:
                    continue
                for i in indices_for_unique[u_idx]:
                    out[i] = blob
                _DESC_VEC_CACHE[(model, unique_texts[u_idx])] = blob
            while len(_DESC_VEC_CACHE) > _DESC_VEC_CACHE_MAX:
                _DESC_VEC_CACHE.popitem(last=False)

    return out


def _relative_time(iso_str, now=None, fine=False):
    """Convert UTC ISO timestamp to relative time label.

    Returns human-readable age: 'just now', 'today', 'yesterday', '3d ago', '2w ago', '1mo ago'.
    Both surface and Anchor see this instead of raw UTC timestamps.

    `fine` (encoder catalog): sub-day steps — '25m ago' / '3h ago' instead of
    'just now' / 'today'. Mid-session the hour is the signal ("revised 20m ago
    → my own recent write" vs "today" which swallows the whole working day).
    Default off — surface keeps its coarse vocabulary.

    `now` — the as-of instant (tz-aware). Replays must pass conversation time
    explicitly (the same rule as iso_cutoff — see tests/test_time_window_contract);
    None → operator wall-clock, correct for live renders only.
    """
    if not iso_str:
        return None
    try:
        ts_str = str(iso_str).replace('Z', '+00:00')
        if '+' not in ts_str and ts_str.count('-') <= 2:
            ts_str += '+00:00'
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now is None:
            # brain_now() in operator's TZ — relative-time labels ("today",
            # "yesterday") must reflect the operator's day boundaries, not the
            # daemon host's UTC day boundaries. See servers/clock.py.
            from servers.clock import brain_now
            now = brain_now()
        delta = now - ts
        hours = delta.total_seconds() / 3600
        days = delta.days

        if fine and hours < 24:
            if delta.total_seconds() < 120:
                return "just now"
            if hours < 1:
                return "%dm ago" % (delta.total_seconds() // 60)
            return "%dh ago" % hours
        if hours < 1:
            return "just now"
        elif hours < 24:
            return "today"
        elif hours < 48:
            return "yesterday"
        elif days < 7:
            return "%dd ago" % days
        elif days < 30:
            return "%dw ago" % (days // 7)
        else:
            months = days // 30
            return "%dmo ago" % months
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════
# SURFACE CONFIG
# ═══════════════════════════════════════════════════════════════

# Haiku's fetch tools — the discovery values that mean "came from a tool
# call" (vs recall MODES like laf_v1/embedding, which are engine provenance).
# Must stay in sync with fetch_tools.TOOL_DEFINITIONS; pinned by
# test_surface_transitions.test_fetch_tool_names_in_sync.
SURFACE_FETCH_TOOLS = ('recall_topical', 'recall_by_time')

# Candidate pool — how many nodes recall pulls for the surface funnel.
CANDIDATE_POOL = {
    'max_candidates': 25,
    # Over-fetch so the seen-dedup filter (already-surfaced nodes dropped
    # from the pool) has ranked replacements to backfill from. Without it
    # the filter only SHRINKS the pool — recall truncates at its limit
    # before the filter runs (a98143f review, finding 1). Sized to the
    # seen-set ceiling: the window holds ~4 selections × ~4 turns.
    'seen_dedup_headroom': 15,
}


# The model used by the S1 surface step. Single source of truth — read by:
#   - surface.py:_call_surface (the actual selection call)
#   - brain.py:warm_anthropic_connection (the boot-time ping that pre-pays
#     SDK + TLS + Haiku route cold-start so the user's first prompt doesn't
#     carry it)
# Kept as a flat constant rather than buried in SURFACE so the warmup path
# doesn't have to reach into surface-call config to know which model to ping.
#
# Deliberately CODE, not interaction config (decided 2026-08-17): the value is
# coupled to CACHE_MIN_PREFIX_TOKENS below (model-specific cache floor) and to
# the warmup ping, so a config-driven model would silently desync both. Moves
# into the `surface` interaction's parameters when a provider adapter owns
# cache strategy — at that point the coupling this protects dissolves.
SURFACE_MODEL = 'claude-haiku-4-5'

# The surface path's name in traces (surface_selected's `surface_variant`, the
# capture stamp). The agentic loop is the only path; the value keeps its
# historical name so trace readers (eval/longmem, ab_tool_use_audit) match
# old and new rows alike.
SURFACE_VARIANT = 'v5_agentic'

# Anthropic prompt-cache minimum cacheable prefix for SURFACE_MODEL.
# Floors are model-specific: Haiku 4.5 + Opus 4.x = 4096; Sonnet 4.6 +
# Fable 5 = 2048; Sonnet 4.5 and older = 1024. This constant is scoped to
# SURFACE_MODEL (Haiku 4.5 → 4096); the Sonnet encoders live above that
# floor at 2048. Prompts under the floor are silently not cached by the API.
# The agentic loop's cache-miss tripwire gates on it so sub-minimum prompts
# (tests, tiny brains) don't warn.
CACHE_MIN_PREFIX_TOKENS = 4096

# Interaction config default for the `surface` K. `layout` is the ONLY key
# the runtime reads from this config (surface.py stamps it on the replay
# capture; it selects no renderer); prompt-size limits live in SURFACE below,
# and the MODEL is deliberately code, not config — see SURFACE_MODEL above
# (cache-floor coupling).
SURFACE_INTERACTION_DEFAULT = {
    'layout': 'xml_v13',
}

# Judge (Haiku) — selects relevant nodes with reasoning
SURFACE = {
    'max_candidates': 30,          # menu slice cap (prepare_presented_candidates); the
                                    # hook's pool is capped lower (CANDIDATE_POOL), so in
                                    # production this is a backstop.
    # Haiku picks at most this many. Sized with SURFACE_INJECT_BUDGET so each
    # pick's share holds a typical node whole: the picker, which has read the
    # message, decides what to leave out — not the renderer's arithmetic.
    # The system prompt states the same number in prose (pinned by test).
    'max_selected': 4,
    # Conversation caps for the selector's <conversation> block. The current
    # message is the moment and arrives whole up to 8,000 (the S0 store keeps
    # 8,000). The assistant's message right before it is the single richest
    # prior turn — most of the vocabulary Haiku borrows from history comes
    # from assistant turns, sniper-style from the most recent — so it gets
    # the same room. Older turns are context, capped tighter. Every cap
    # must stay under PIPELINE['recent_message_content'] (the trace pull).
    'current_message_limit': 8000,
    'last_anchor_message_limit': 8000,
    'user_message_limit': 1000,     # older user turns
    'anchor_message_limit': 1200,   # older assistant turns
    'recent_messages': 8,           # previous MESSAGES (~4 exchanges when turns alternate;
                                    # interrupted turns leave lone user messages, so pairing
                                    # is not guaranteed). Single source of truth: daemon_hooks
                                    # pulls exactly this many from traces (current chain
                                    # excluded via exclude_trace_id, wake envelopes filtered);
                                    # build_surface_prompt slices the same number. The current
                                    # prompt is NOT in this window — it renders as the final
                                    # <turn current_msg="true">.
    'max_tokens': 600,              # Haiku output cap
}


def recall_score(recall_result):
    """The ONE score semantic for surface candidates.

    brain.recall() results carry no 'score' key — the surface pipeline's
    score is `effective_activation` from the scoring pipeline. Every reader
    — the hook's cosine pool (daemon_hooks), fetch_tools.recall_topical,
    recall_by_time's query tiers, and the encoder's associated-stub ranking
    (encode._associated_stub_ids) — MUST read it through this function: the
    agentic admission floor compares tool-fetched scores against the pool
    median, so readers forking
    on field name silently zero one of them (2026-07-02: recall_topical
    read 'score' → 0.0 for every fetch → floor dropped 100% for 3 weeks).
    Type-guarded: a non-numeric (or bool) value scores 0.0 rather than
    leaking into comparisons.
    """
    v = recall_result.get('effective_activation')
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return 0.0


# ═══════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════

def format_candidate_for_surface(c, scope=None):
    """Format a single candidate as the surface prompt's <candidate> element.

    Thin wrapper around render_rich_node(HAIKU_FORMAT*) — the node body
    wrapped in the element Haiku picks from: id as attribute (the pick key),
    locked / source_tool as flags. No score header: exposing match floats
    invites anchoring on our numbers instead of semantic judgment (v13
    decision), and since 2026-07-14 (§20.12 A2) the menu order is shuffled
    per turn, so the menu carries no rank signal at all — Haiku judges on
    meaning alone.

    Candidates must be in get_rich_node() shape (with _metadata,
    _corrections, connections).

    Render DEPTH: BRAIN_HAIKU_RENDER env ∈ {'lean' (default), 'full'} picks
    HAIKU_FORMAT_LEAN vs HAIKU_FORMAT. It selects the BODY only — the
    element and its attribute set are identical either way, so a depth
    change can never move the pick key. Default flipped full→lean
    2026-06-12 after the ablation (ab_render_ablation.py): gold-neutral at
    −41% tokens, pick-divergence (J=0.64) at the same-prompt noise floor
    (full-vs-full J=0.72).

    One render for the whole menu: the initial cosine candidates AND
    tool-fetched results come through here (fetch_tools calls this
    function), so the two halves of one prompt cannot speak different
    grammars or carry different data richness.
    """
    import os as _os_hr
    from servers.contract import render_rich_node

    fmt = HAIKU_FORMAT \
        if _os_hr.environ.get('BRAIN_HAIKU_RENDER', 'lean').strip().lower() == 'full' \
        else HAIKU_FORMAT_LEAN
    if scope is not None:
        # Differential project exposure. dict(fmt, ...) COPIES — mutating
        # the module constant here would leak this session's scope into
        # every later render in the process.
        fmt = dict(fmt, scope=scope)

    attrs = ['id="%s"' % str(c.get('id', ''))[:8]]
    if c.get('locked'):
        attrs.append('locked="true"')
    # Strict tool-name match: `discovery` also carries recall MODES
    # (laf_v1, direct_lookup) which are not tool provenance — an
    # "anything non-embedding" predicate would mislabel them.
    if c.get('discovery', '') in SURFACE_FETCH_TOOLS:
        attrs.append('source_tool="true"')
    return '<candidate %s>\n%s\n</candidate>' % (
        ' '.join(attrs), render_rich_node(c, fmt))


def _dedup_candidates(candidates):
    """Remove near-duplicate candidates by title similarity.
    Keeps the higher-scored candidate when two titles share >80% words."""
    if len(candidates) <= 1:
        return candidates
    seen_titles = {}  # normalized title words → candidate
    result = []
    for c in candidates:
        title_words = set(c.get("title", "").lower().split())
        duplicate = False
        for seen_key, seen_c in list(seen_titles.items()):
            seen_words = set(seen_key.split())
            if not title_words or not seen_words:
                continue
            overlap = len(title_words & seen_words) / max(len(title_words), len(seen_words))
            if overlap > 0.8:
                # Keep the higher scored one
                if (c.get("score", 0) or 0) > (seen_c.get("score", 0) or 0):
                    result.remove(seen_c)
                    seen_titles[" ".join(sorted(title_words))] = c
                    result.append(c)
                duplicate = True
                break
        if not duplicate:
            key = " ".join(sorted(title_words))
            seen_titles[key] = c
            result.append(c)
    return result


def _group_turns(msgs):
    """Group a flat oldest-first message list into user+assistant turns.

    A user message always opens a new turn; an assistant message closes the
    open turn or opens its own (lone assistant — e.g. the user half was a
    filtered wake envelope). No strict-alternation assumption.
    """
    groups = []
    for m in msgs:
        if m.get('role') == 'user':
            groups.append({'user': m})
        elif groups and 'assistant' not in groups[-1]:
            groups[-1]['assistant'] = m
        else:
            groups.append({'assistant': m})
    return groups


def _build_user_content_xml(candidates, user_message, recent_messages,
                            retrieval_stats, frame, cfg,
                            scope=None):
    """v13 XML user content — one grammar, oldest first, current message last.

    Sections: <partnership_context> (Frame), <conversation> (turns with
    per-turn <shown> dedup elements; final turn is current_msg="true", user
    only), <candidates> (id-attribute grammar, no score floats), and a
    conditional <note> when retrieval is weak (the threshold logic stays in
    code; Haiku sees the conclusion, never our numbers).

    Already-shown memories sit inline on the turn they surfaced for, so
    out-of-scope is structural (<shown> vs <candidate>), not a rule about a
    distant block.
    """
    parts = []
    parts.append('<partnership_context>\n%s\n</partnership_context>' % (
        frame.strip() if frame
        else '(none — fresh session or Frame unavailable)'))

    lines = ['<conversation>']
    n = 0
    groups = _group_turns((recent_messages or [])[-(cfg['recent_messages']):])
    last_assistant_gi = max((i for i, g in enumerate(groups) if g.get('assistant')),
                            default=-1)
    for gi, g in enumerate(groups):
        n += 1
        lines.append('<turn n="%d">' % n)
        u = g.get('user')
        if u:
            lines.append('<user>%s</user>'
                         % (u.get('content') or '')[:cfg['user_message_limit']])
            # Id only: the one rule that reads this element is "its id
            # appears in any <shown>" (surface_prompt). The title was never
            # matched, never mentioned — it cost ~93 chars an entry, and it
            # made the picker compare a cut title against the whole one the
            # candidate menu renders. Nothing reads it, so nothing pays.
            for s in (u.get('surfaced') or []):
                lines.append('<shown id="%s"/>' % str(s.get('id', ''))[:8])
        a = g.get('assistant')
        if a:
            # The most recent assistant message gets its own, larger cap
            # (SURFACE['last_anchor_message_limit']); older assistant turns
            # are context at 'anchor_message_limit'. Keyed on the last group
            # that HAS an assistant — an interrupted turn leaves a trailing
            # lone user message, and that must not demote the real last reply.
            a_limit = (cfg.get('last_anchor_message_limit', cfg['anchor_message_limit'])
                       if gi == last_assistant_gi else cfg['anchor_message_limit'])
            lines.append('<assistant>%s</assistant>'
                         % (a.get('content') or '')[:a_limit])
        lines.append('</turn>')
    n += 1
    lines.append('<turn n="%d" current_msg="true">' % n)
    lines.append('<user>%s</user>' % (
        user_message[:cfg.get('current_message_limit', cfg['user_message_limit'])]
        if user_message else '(no message)'))
    lines.append('</turn>')
    lines.append('</conversation>')
    parts.append('\n'.join(lines))

    cand_lines = ['<candidates n="%d">' % len(candidates)]
    for c in candidates:
        cand_lines.append('')
        cand_lines.append(format_candidate_for_surface(c, scope=scope))
    cand_lines.append('</candidates>')
    parts.append('\n'.join(cand_lines))

    if retrieval_stats:
        from servers.brain_constants import RETRIEVAL_LOW_CONFIDENCE
        if retrieval_stats.get('top_score', 0) < RETRIEVAL_LOW_CONFIDENCE:
            parts.append('<note>Retrieval is weak for this message — the '
                         'brain likely has nothing relevant. Prefer '
                         'selecting 0.</note>')

    return '\n\n'.join(parts), cfg['max_tokens']


def presentation_shuffle_seed(session_id, user_message):
    """Deterministic per-turn seed for the candidate presentation shuffle.

    Derived from (session_id, user_message) — the two per-turn inputs
    _call_surface already holds — so a replay of the same turn reproduces
    production's presentation order exactly. sha256, not hash(): Python
    salts hash() per process, and the seed must be stable across daemon
    restarts and replay processes.
    """
    key = ('%s|%s' % (session_id or '', user_message or '')).encode(
        'utf-8', 'replace')
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def prepare_presented_candidates(candidates, shuffle_seed=None, cfg=None):
    """Slice + dedup + presentation shuffle — the exact candidate list
    build_surface_prompt renders, in the order Haiku sees it.

    The shuffle (2026-07-14, RECALL-SR-REDESIGN.md §20.12 A2): Haiku shows
    a mild pure position bias (1.19× top-vs-bottom pick rate within
    identical-score tie groups, 92k candidates), so the PRESENTATION order
    is randomized per turn. The scorer's ranking is untouched everywhere
    else — the O-trace cand_detail and the walker's rank_in_pool keep
    recording it; only what Haiku sees shuffles. Side effect: every
    picked/dropped log row becomes randomization-grade training data for
    the LAF P3 fit (exact uniform propensities, no estimation).

    Slice + dedup run BEFORE the shuffle so top-N membership stays
    scorer-determined. Seeded RNG so the capture/replay byte-contract
    holds; shuffle_seed=None skips the shuffle (evals and tests that don't
    pass a seed render in scorer order).
    """
    cfg = cfg or SURFACE
    cands = _dedup_candidates(candidates[:cfg['max_candidates']])
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(cands)
    return cands


# The one layout this renderer builds; the surface interaction config's
# `layout` must name it (test_interaction_defaults pins the pair).
SURFACE_LAYOUT = 'xml_v13'


def build_surface_prompt(candidates, user_message,
                       recent_messages=None,
                       retrieval_stats=None, frame="", shuffle_seed=None,
                       scope=None):
    """Build the S1 recall surface USER message — per-turn delta only.

    Builds ONLY the per-turn user content: <partnership_context> (Frame),
    <conversation> (with per-turn <shown>), <candidates>, and a
    weak-retrieval <note> when it applies. The registered `surface`
    interaction template is the cached system block (sent by
    `_call_surface`); this is the user block. Two parts → two-block API
    call → caching becomes possible.

    The operator renders as <user>, never by name — the brain plugin ships
    to different operators, prompts must not hardcode personal names.

    Args:
        candidates: List of candidate node dicts (enriched with metadata)
        user_message: The operator's latest message — rendered as the final
            <turn current_msg="true"> (user only, no reply yet)
        recent_messages: List of {"role": str, "content": str} — PREVIOUS
            turns only; the caller excludes the current turn's chain
            (daemon_hooks passes exclude_trace_id to get_session_turns)
        retrieval_stats: Dict with brain_size, top_score, median_score, source_breakdown
        frame: Markdown Frame (Anchor's prior). When non-empty becomes the
            <partnership_context> block. When empty, explicit degraded marker.
        shuffle_seed: When not None, presentation order is a seeded shuffle
            of the deduped pool (see prepare_presented_candidates). None →
            scorer order.

    Returns: (user_content_string, max_tokens)
    """
    cfg = SURFACE

    # Slice + dedup + presentation shuffle (§20.12 A2) — one prep path,
    # shared with the presented-order record _call_surface writes to traces.
    candidates = prepare_presented_candidates(candidates, shuffle_seed, cfg)

    return _build_user_content_xml(
        candidates, user_message, recent_messages, retrieval_stats,
        frame, cfg, scope=scope)


# ═══════════════════════════════════════════════════════════════
# INJECT RENDER POLICY — what the stream reads after Haiku's selection
# ═══════════════════════════════════════════════════════════════

# Seeds first: Haiku's picks are the memories the moment needs. They split
# the budget between them and every one of them renders; spread neighbors
# are peripheral vision, one title line each, after the seeds. A field is
# shown whole or not at all — a cut names how much is behind it and where
# to get it, and nothing renders as a stub. The 12-moment audit behind the
# ruling: content carried the stream's need 21 times and was cut or absent
# in 27 of its renders; situation, reasoning and edge descriptions carried
# it 0 times in 36 blocks.

# Soft target for the whole inject; the hard exit cap is _MAX_INJECT_CHARS.
# Split across SURFACE['max_selected'] picks it gives each ~2,000 — about
# 1,300-1,600 of content after the block's skeleton — so nodes of typical
# length render whole and only the long ones meet a sentence-boundary cut.
SURFACE_INJECT_BUDGET = 8000

# Hard byte cap on the inject. Claude Code spills additionalContext to a
# file above ~10k chars, and Anchor doesn't read that file path back. Cap
# below the ceiling with headroom for any wrapper bytes Claude Code adds.
_MAX_INJECT_CHARS = 9500

# When the picks' content was cut under the soft target, the second pass may
# grow it into the room below the hard cap, keeping this much back.
SURFACE_HARD_CAP_MARGIN = 300

# Below this a content cut is a stub, not a memory: the content is left out
# and a pointer line says how long it is. A sentence cut's trailing
# ' … (+N chars: get_nodes)' marker is budgeted at SURFACE_CUT_MARKER_CHARS.
SURFACE_CONTENT_MIN_CHARS = 400
SURFACE_CUT_MARKER_CHARS = 32

# Peripheral vision: how many lit-but-unpicked neighbors get a title line.
SURFACE_ALSO_LIT_LIMIT = 10

# Corrections on a seed: the first few as one-liners, then a count.
SURFACE_CORRECTION_LIMIT = 2

# Valid render modes Haiku may emit in selection JSON. Default 'arc'
# when a pick has no `mode` field.
SURFACE_MODES = ('arc', 'fact', 'background')
SURFACE_MODE_DEFAULT = 'arc'

# Fields that never render in the inject. situation and reasoning sit on
# every node and are selection / encoding scaffolding once the pick is
# made; question and keywords are recall scaffolding; my own quote earned
# its place once in 17 renders where the counterpart's did seven times in
# 16; source_context is session bookkeeping; event_time renders once, as
# the Event date line under the header, never again through the KV loop;
# evolution_status renders as the header's ⚠ DISPROVEN / ⚠ DISMISSED mark
# (mark_status), not as a KV line — the other statuses are bookkeeping.
SURFACE_SKIP_KEYS = ('question', 'situation', 'reasoning', 'my_raw_quote',
                     'source_context', 'event_time', 'evolution_status',
                     'keywords')
# A community seed is its title + narrative; its community_* bookkeeping
# rendered as 40-char stubs.
SURFACE_SKIP_KEY_PREFIXES = ('community_',)

# Seed render for arc and fact modes. Content depth is decided per seed by
# the budget split (the renderer sets content_limit); the rest is fixed
# policy: edges as titles with the neighbor's id, corrections capped, the
# counterpart's quote labeled with their name, sparse fields (event date,
# correction pattern, thought, evolution status) when a node has them, and
# the Conversation pointer when it carries source_refs.
SURFACE_SEED_FORMAT = {
    'edge_limit': 3,
    'edge_style': 'oneline',
    'edge_ids': True,
    'time_format': 'relative',
    'show_confidence': False,
    'show_encoding_source': False,
    'correction_render': 'lean',
    'correction_limit': SURFACE_CORRECTION_LIMIT,
    'extra_skip_keys': SURFACE_SKIP_KEYS,
    'skip_key_prefixes': SURFACE_SKIP_KEY_PREFIXES,
    'content_cut': 'sentence',
    'mark_status': True,
    'show_source_refs': True,
    'metadata_limit': META_BRIEF,
}

# Background mode — the header line (plus the event date) and nothing else:
# framing that is not load-bearing, at the cost of a title.
SURFACE_BACKGROUND_FORMAT = {
    'content_limit': 0,
    'metadata_limit': META_NONE,
    'edge_limit': 0,
    'correction_render': 'none',
    'time_format': 'relative',
    'show_confidence': False,
    'show_encoding_source': False,
    'mark_status': True,
    # metadata_limit 0 silences the KV loop; situation is a top-level field
    # and needs the skip named to stay out of the header-only block.
    'extra_skip_keys': SURFACE_SKIP_KEYS,
}


def surface_voice_labels(scope):
    """The quote label for the inject: the counterpart's name from the
    session scope ('<counterpart> said'), or 'They said' when the session declares
    none. Only the counterpart's quote renders (my_raw_quote is a skip
    key); the label says whose words they are, in a form the reader reads
    through rather than parses."""
    who = str((scope or {}).get('counterpart') or '').strip()
    return {'their_raw_quote': ('%s said' % who) if who else 'They said'}


def seed_render_cfg(mode, scope=None, content_limit=None):
    """A fresh render_rich_node cfg for one seed: the mode's format plus
    the session scope and this seed's content cap. Always a new dict — the
    format constants are shared, and a scope written into one would leak
    into every other session's renders."""
    fmt = (SURFACE_BACKGROUND_FORMAT if mode == 'background'
           else SURFACE_SEED_FORMAT)
    cfg = dict(fmt)
    if mode != 'background':
        cfg['content_limit'] = content_limit
        cfg['voice_labels'] = surface_voice_labels(scope)
    if scope:
        cfg['scope'] = scope
    return cfg

# ═══════════════════════════════════════════════════════════════
# PICKER RENDER — what Haiku reads when selecting a few from 25
# ═══════════════════════════════════════════════════════════════

# Distinct from the SURFACE_*_FORMAT constants above: HAIKU_FORMAT is
# the IN (rendered candidate menu Haiku reads), the SURFACE_*_FORMATs
# are the OUT (what Anchor receives after Haiku's selection).
HAIKU_FORMAT = {
    'content_limit': 600, 'edge_limit': 5, 'metadata_limit': META_BRIEF,
    'time_format': 'relative',
    # Haiku surface receives correction context at balanced fidelity:
    # relation + edge_description + content excerpt (~150 chars). Enough
    # for picks to factor in superseded knowledge without bloating the
    # 25-candidate prompt.
    'correction_render': 'balanced',
    'communities': 'title',
}

# Selection-grade lean render (Area 2, 2026-06-12). The selector's job is
# "pick the few relevant ids", not "read the node" — injection still renders
# the SELECTED nodes at full richness, so no information leaves the
# pipeline; it's relocated to the stage that uses it. Recon numbers
# (measured 2026-06-12 over 150 candidates, pre-v13 candidate grammar): full render
# 385 tok/cand — edges-with-descriptions 29%, content 23%, situation 10%,
# reasoning 10%, question 6%, corrections 6%, quotes 5%. Lean keeps every
# SELECTION signal in cheapest sufficient form (~60% cut):
#   • situation kept whole — it IS the selection question ("when relevant")
#   • content 600 and 5 edges (operator call: the 300/3 lean cut was too
#     cheap for a selector that must recognize the node it is picking)
#   • encoding_source kept in header (future: guide Haiku to prefer
#     src:anchor manual encodings)
#   • edges → oneline (direction + relation + title; descriptions are
#     injection payload). Edge CHOICE stays query-aware via select_edges.
#   • corrections → lean flag (the "superseded" signal, not the payload)
#   • dropped: reasoning, question, quotes — encoder/recall
#     scaffolding and voice; zero hypothesized selection value (the
#     inject path already drops question for the same reason).
# DEFAULT since 2026-06-12 (ablation-cleared: gold-neutral, −41% tokens,
# divergence at the same-prompt noise floor — ab_render_ablation.py).
# BRAIN_HAIKU_RENDER=full reverts to the heavy render.
# Open follow-on: lean_noedge scored equal-or-better at −53% — edge lines
# carry no measurable selection signal TODAY, but the cut is deferred until
# the aspect-aligned edge-choice experiment decides whether edges can earn
# their place in selection (operator's aspect-traversal thread).
HAIKU_FORMAT_LEAN = {
    'content_limit': 600, 'edge_limit': 5, 'metadata_limit': META_BRIEF,
    'time_format': 'relative',
    'correction_render': 'lean',
    'edge_style': 'oneline',
    'extra_skip_keys': ('question', 'reasoning', 'their_raw_quote',
                        'my_raw_quote'),
    # `communities` stays off: this render is ablation-measured
    # (ab_render_ablation.py) and a line per candidate is a cost the
    # selector has not been shown to repay — turn it on with a number.
}


# ═══════════════════════════════════════════════════════════════
# OUTPUT SCHEMA — what Haiku must produce (Anthropic Structured Outputs)
# ═══════════════════════════════════════════════════════════════

# Enforced during generation via output_config={'format':{'type':'json_schema',
# 'schema': SURFACE_SELECTION_SCHEMA}} on the final agentic-round API call.
# See surface.py:_call_surface_agentic.
# id pattern: hex-only kills the observed corruption class — Haiku copying a
# candidate id as its BPE chunks with spaces leaked in ('9 9a 2e ' for
# 99a2e…) — by masking the space token during constrained decoding. Length
# {4,8}, not {8}: Haiku is known to emit 7-char leading-zero-dropped ids, and
# a hard 8 would force it to guess digits it doesn't know; short-but-honest
# fragments are recovered downstream (surface.py unique-prefix match).
# No per-pick rationale field by design: generation time tracks output
# tokens (~20ms/token), a why per pick cost ~150-200 tokens/recall, and
# nothing rendered it. The recall-level `reason` (kept, cheap) is the
# journal signal — persisted as selection_reason in the K trace.
SURFACE_SELECTED_ID_PATTERN = "^[0-9a-f]{4,8}$"

SURFACE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string",
                             "pattern": SURFACE_SELECTED_ID_PATTERN},
                    "mode": {"type": "string", "enum": list(SURFACE_MODES)},
                },
                "required": ["id", "mode"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["selected"],
    "additionalProperties": False,
}


# ═══════════════════════════════════════════════════════════════
# SPREAD_ACTIVATION — the kernel that decides what surfaces
#
# One mechanism, three granularities:
#   • Nodes receive activation  (max over per-field cosines with query)
#   • Edges transmit activation (cosine of query to enriched edge text)
#   • Fields within a node hold per-field activation (for render budget)
#
# Depth is emergent — high-coefficient edges keep propagating, low-
# coefficient edges stop on the first hop. Saturation (tanh) bounds
# accumulation. The per-hop transmission gate is self-calibrating:
# when no edges in the batch exceed the brain's noise floor, propagation
# halts — the brain is allowed to be quiet.
#
# No static blend weights. No multiplicative family boost (family meaning
# rides inside the enriched edge text). No count-based fatigue. The
# activation map IS the ranking, the tiebreaker, and the render-budget
# allocator, all in one.
# ═══════════════════════════════════════════════════════════════

import numpy as np

# Safety cap — if activation hasn't settled by 3 hops, force stop.
# Real termination comes from the transmission gate + saturation, not this.
_SPREAD_MAX_STEPS = 3

# Per-source neighbor cap during spread expansion. 50 is the benched value:
# a cap of 30 starved recall during ingestion (the encoder missed specific
# facts; 80%→60% on one sample, 5 items hurt, 0 helped). That bench ran with
# the scout muster in the ingestion loop, so re-bench before moving this.
SPREAD_NEIGHBOR_LIMIT = 50

# Outer propagation gate: if the max transmission coefficient in the batch
# falls below the brain's noise floor, nothing meaningful to spread —
# halt rather than forcing below-noise edges through. Imported from
# brain_constants to stay in sync with recall()'s own noise definition.
from servers.brain_constants import NOISE_FLOOR_THRESHOLD as _SPREAD_NOISE_FLOOR


def _batch_load_field_vectors(brain_conn, node_ids):
    """One SQL round-trip: all field-cohort + legacy vectors for given nodes.

    Returns: {full_node_id: {vector_type: numpy.ndarray}}
    Short-prefix matching handled by caller if needed; here we read full IDs.
    """
    if not brain_conn or not node_ids:
        return {}
    ids = list(node_ids)
    placeholders = ','.join('?' * len(ids))
    rows = brain_conn.execute(
        "SELECT node_id, vector_type, embedding FROM node_enrichments "
        "WHERE node_id IN (%s) AND embedding IS NOT NULL" % placeholders,
        ids).fetchall()
    out = {}
    for nid, vtype, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        out.setdefault(nid, {})[vtype] = vec
    return out


def _field_cosines_for_node(query_vec, node_vectors, norm_q=None):
    """Cosines against query, per semantic field.

    `node_vectors` = {vector_type: vec} from _batch_load_field_vectors.
    For each field type in the field-cohort stable order, try the per-field
    vector first; fall back to the blended legacy vector via
    FIELD_VECTOR_FALLBACK. Missing field → not in the result (no cosine
    fabricated).

    Returns: {field_name: cosine} — order matches field_vector_types().
    """
    from servers.pipeline_contract import field_vector_types, FIELD_VECTOR_FALLBACK
    out = {}
    for field_type in field_vector_types():
        # Field-cohort vector available?
        vec = node_vectors.get(field_type)
        # Fall back through legacy chain
        if vec is None:
            for fb in FIELD_VECTOR_FALLBACK.get(field_type, []):
                vec = node_vectors.get(fb)
                if vec is not None:
                    break
        if vec is None:
            continue
        out[field_type] = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
    return out


def _bulk_load_stored_edge_embeddings(brain_conn, edge_keys,
                                       brain=None, error_label=None):
    """Bulk-load (edge_id, relation) → embedding blob, filtered by active model.

    Used by both `_build_edge_coeffs` (spread) and `select_edges`
    (per-candidate render). Single SQL round-trip per chunk via
    `(edge_id, relation) IN (VALUES (?, ?), ...)`. Filters out:
      - Rows with NULL embedding (not yet filled by the embed_queue worker)
      - Rows where embedding_model ≠ active model (stale after model swap;
        same staleness pattern node_enrichments uses)

    Stale and NULL rows fall through to the on-demand embed path in the
    caller — correctness preserved at the cost of a fastembed call.

    A key with a relation but no edge_id is a producer bug, not staleness:
    every GraphDAL edge read carries edge_id (EDGE_ROW_SHAPE). It also falls
    through to on-demand embed, and is logged as `edge_embedding_key_missing`
    — silently, it once cost every hook turn ~150 live embeds for months.

    Args:
        brain_conn: sqlite3 connection.
        edge_keys: iterable of (edge_id, relation) tuples — duplicates fine.
        brain: optional Brain — used only for error logging via _log_error.
            Function works without it (errors silently swallowed) but
            production callers should pass it.
        error_label: brain._log_error tag for failures (e.g.
            'edge_embedding_read_spread'). None disables the log.

    Returns:
        Dict[(edge_id, relation), bytes]. Empty if no input or all stale.
    """
    if not edge_keys:
        return {}
    edge_keys = list(edge_keys)
    valid = [(eid, rel) for eid, rel in edge_keys if eid and rel]
    missing = sum(1 for eid, rel in edge_keys if rel and not eid)
    if missing and brain is not None:
        brain._log_error(
            'edge_embedding_key_missing',
            ValueError('%d of %d edge keys carry no edge_id' % (
                missing, len(edge_keys))),
            '%s: the producer dropped edge_id; these edges re-embed live'
            % (error_label or 'stored edge embedding read'))
    if not valid:
        return {}

    from servers import embedder as _embedder_mod
    active_model = _embedder_mod.stats.get('model_name') or ''

    out: dict = {}  # (edge_id, relation) -> embedding blob
    # SQLite parameter limit is 999 (default; some builds 32766). With
    # 2 params per pair + 1 for active_model, chunk_size=400 → 801 params,
    # safely under 999.
    chunk_size = 400
    keys = list(valid)
    try:
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i:i + chunk_size]
            ph = ','.join(['(?, ?)'] * len(chunk))
            params = [v for pair in chunk for v in pair] + [active_model]
            rows = brain_conn.execute(
                'SELECT edge_id, relation, embedding FROM edge_relations '
                'WHERE (edge_id, relation) IN (VALUES %s) '
                'AND embedding IS NOT NULL '
                'AND embedding_model = ?' % ph, params).fetchall()
            for eid, rel, blob in rows:
                if blob:
                    out[(eid, rel)] = blob
    except Exception as e:
        # Stored-embedding lookup failure: caller's loop falls through to
        # on-demand embed for everything it didn't get back here. Log if
        # we have a brain to log against; otherwise swallow (defensive).
        if brain is not None and error_label:
            try:
                brain._log_error(
                    error_label, e,
                    'bulk fetch of stored edge embeddings — '
                    'falling back to on-demand embed')
            except Exception:
                pass
    return out


def _build_edge_coeffs(brain, activated_nodes, query_vec,
                      cached_edge_coeffs):
    """Collect outgoing edges from all currently activated nodes; compute
    their transmission coefficients.

    Returns list of (source_id, target_id, coeff, edge_dict).
    Uses cached_edge_coeffs as an in-kernel memo so repeat edges in later
    hops don't recompute cosine.

    Implementation note (schema v26+): edges have stored embeddings on
    `edge_relations.embedding`, computed ASYNC by the embed_queue worker
    (`Brain.backfill_edge_embeddings`) — write paths only invalidate (NULL the
    blob) + enqueue_edge. For each edge we look up the stored blob by `edge_id`
    and skip fastembed entirely. Edges still NULL (pre-v26, or written since the
    last worker drain) fall through to the on-demand embed fallback; recall
    stays correct, just pays the per-edge fastembed cost. The worker repopulates
    NULLs on its next drain.

    Enriched-text composition delegates to
    `brain.aspects.compose_edge_text(relation, description)` — single
    source of truth for the embed format.

    Bulk SQL: get_neighbors_bulk for all sources in one round-trip (3 SQL
    queries per recall instead of 200+). Per-owner neighbor cap enforced
    in Python.
    """
    gdal = brain._graph
    excluded = set(brain.aspects.traversal_exclusions)


    # ── ONE SQL round-trip for all sources ──
    bulk = gdal.get_neighbors_bulk(
        activated_nodes, exclude_relations=excluded)

    # Collect every edge first; we'll fetch stored embeddings in bulk.
    pending = []  # list of (source_id, target_id, edge_dict, enriched_text)
    edge_keys = set()  # set of (edge_id, relation) for the bulk lookup
    for source_id in activated_nodes:
        rows = bulk.get(source_id, [])
        if len(rows) > SPREAD_NEIGHBOR_LIMIT:
            rows = sorted(rows, key=lambda r: r.get('weight') or 0,
                          reverse=True)[:SPREAD_NEIGHBOR_LIMIT]
        for r in rows:
            target_id = r.get('id', '')
            # Compose enriched text via AspectRegistry (single source of
            # truth for the format — see aspects.py:compose_edge_text).
            # Used as the in-call memo key + as the input to the on-demand
            # embed fallback if the stored blob is NULL.
            enriched = brain.aspects.compose_edge_text(
                r.get('relation', ''),
                r.get('edge_description') or '')
            pending.append((source_id, target_id, r, enriched))
            edge_keys.add((r.get('edge_id'), r.get('relation') or ''))

    # Bulk-load stored embeddings (schema v26+) via the shared helper.
    # See _bulk_load_stored_edge_embeddings docstring for SQL/staleness
    # semantics. Empty result on any failure — caller falls through to
    # on-demand embed for all edges (correctness preserved).
    stored_embeddings = _bulk_load_stored_edge_embeddings(
        brain.conn, edge_keys, brain=brain,
        error_label='edge_embedding_read_spread')

    edges_out = []
    enriched_to_embed = []  # texts still needing embedding (NULL in DB)
    enriched_keys = []      # parallel: (source, target, edge, text)
    norm_q = float(np.linalg.norm(query_vec))

    for source_id, target_id, r, enriched in pending:
        # Per-call memo (in-kernel cache across hops in this recall) —
        # avoids cosine recomputation when the same edge appears as a
        # source-target reverse in a later hop.
        cached = cached_edge_coeffs.get(enriched)
        if cached is not None:
            edges_out.append((source_id, target_id, cached, r))
            continue

        # Stored embedding (schema v26+): skip fastembed entirely. Populated
        # ASYNC by the embed_queue worker (Brain.backfill_edge_embeddings) after
        # a write invalidates+enqueues — see brain_connections.py.
        eid = r.get('edge_id')
        rel = r.get('relation') or ''
        blob = stored_embeddings.get((eid, rel)) if eid and rel else None
        if blob is not None:
            vec = np.frombuffer(blob, dtype=np.float32)
            coeff = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
            cached_edge_coeffs[enriched] = coeff
            edges_out.append((source_id, target_id, coeff, r))
            continue

        # Fallback: stored blob still NULL (awaiting the embed_queue worker)
        # or model-stale — queue for the on-demand embed.
        enriched_to_embed.append(enriched)
        enriched_keys.append((source_id, target_id, r, enriched))

    # Batch-embed queued enriched texts (hits _desc_vecs_batched cache).
    # On a fully-backfilled brain this list is empty and the fastembed
    # call is skipped entirely.
    if enriched_to_embed:
        blobs = _desc_vecs_batched(enriched_to_embed)
        for (src, tgt, edge, text), blob in zip(enriched_keys, blobs):
            if blob is None:
                coeff = 0.0
            else:
                vec = np.frombuffer(blob, dtype=np.float32)
                coeff = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
            cached_edge_coeffs[text] = coeff
            edges_out.append((src, tgt, coeff, edge))

    return edges_out


def spread_activation(seed_ids, query_vec, brain, prior_vecs=None):
    """Spreading activation from seed nodes through the graph.

    One mechanism:
      • Activation originates at each seed from its max field-cosine with query
      • It flows through edges, weighted by the edge's own enriched-text cosine
      • Nodes receiving activation from multiple paths accumulate (tanh-saturated)
      • Depth is emergent — a strong-matching edge propagates far, a weak one
        stops on hop 1. Mutual traversal is automatic: two seeds whose paths
        meet at a neighbor boost that neighbor above singleton reach.

    Args:
        seed_ids: List of full node IDs to start activated (typically the set
            Haiku selected from the 25 candidates).
        query_vec: numpy array (768d) — current query embedding.
        brain: Brain instance — conn access, the aspect registry's traversal
            exclusions, and error logging.
        prior_vecs: Optional list of prior-turn query embeddings for
            multi-turn blending (dampens one-word query ambiguity).

    Returns:
        dict with:
          'node_activation':  {node_id: float in [0,1]}
          'field_activation': {node_id: {field_name: float in [0,1]}}
          'trace': list of per-step {step, new_nodes, edges_considered,
                                     edges_transmitted, max_act}
    """
    if not seed_ids or query_vec is None:
        return {'node_activation': {}, 'field_activation': {}, 'trace': []}

    # ── Multi-turn query blend (existing pattern, kept) ──
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # ── Step 0: seeds' own activations (per-field cosines vs query) ──
    all_touched_ids = list(seed_ids)
    node_vectors = _batch_load_field_vectors(brain.conn, all_touched_ids)
    # Silent-failure guard: if we got NO vectors for a seed, the kernel can't
    # produce activation for it. This happened at N=15 once (seed had no
    # vectors at all), which is the failure mode worth surfacing.
    seeds_missing_vectors = [sid for sid in seed_ids if sid not in node_vectors]
    if seeds_missing_vectors:
        brain._log_error('spread_seed_no_vectors',
                         RuntimeError('seeds without any vectors'),
                         'seeds=%s (kernel will produce zero activation for these)' %
                         ','.join(s[:8] for s in seeds_missing_vectors[:5]))
    node_activation = {}
    field_activation = {}

    for nid in seed_ids:
        vecs = node_vectors.get(nid, {})
        field_cos = _field_cosines_for_node(blended, vecs, norm_q=norm_q)
        field_activation[nid] = field_cos
        # MAX over signals, per TRIZ — best discriminating field wins
        node_activation[nid] = max(field_cos.values()) if field_cos else 0.0

    # ── Spread loop ──
    trace_steps = []
    cached_edge_coeffs = {}  # memo across hops: enriched_text → coeff

    for step in range(_SPREAD_MAX_STEPS):
        # Source nodes at this hop: whichever are currently activated above zero
        active_sources = [n for n, a in node_activation.items() if a > 0]
        if not active_sources:
            break

        edges = _build_edge_coeffs(
            brain, active_sources, blended,
            cached_edge_coeffs)

        if not edges:
            break

        # Outer gate: any meaningfully-matching edges at all?
        max_coeff = max(e[2] for e in edges)
        if max_coeff < _SPREAD_NOISE_FLOOR:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'max_act': max(node_activation.values())
                                   if node_activation else 0.0,
                               'halted': 'below_noise_floor'})
            break

        # Per-hop self-calibrating threshold: median of positive coeffs.
        # Ensures only better-than-typical edges in THIS batch transmit.
        threshold = float(np.percentile([e[2] for e in edges], 50))

        # Accumulate contributions for each target
        contributions = {}
        transmitted_count = 0
        for src, tgt, coeff, _ in edges:
            if coeff < threshold:
                continue
            transmitted_count += 1
            source_act = node_activation.get(src, 0)
            # Target hasn't been computed yet? Skip source-act lookup failures.
            transferred = source_act * coeff
            contributions[tgt] = contributions.get(tgt, 0.0) + transferred

        if not contributions:
            trace_steps.append({'step': step, 'new_nodes': 0,
                               'edges_considered': len(edges),
                               'edges_transmitted': 0,
                               'max_act': max(node_activation.values()),
                               'halted': 'all_below_threshold'})
            break

        # Load field vectors for any newly-touched targets (one round-trip)
        new_target_ids = [t for t in contributions.keys() if t not in node_activation]
        if new_target_ids:
            fresh_vecs = _batch_load_field_vectors(brain.conn, new_target_ids)
            node_vectors.update(fresh_vecs)
            all_touched_ids.extend(new_target_ids)

        # Apply saturation — tanh blends old activation + incoming contribution.
        # Also seed target's own field-cosines (against query) so rendering has
        # field-level signal for newly-activated targets.
        pre_activation = dict(node_activation)
        new_nodes_added = 0
        for target, incoming in contributions.items():
            prior = node_activation.get(target, 0.0)
            saturated = float(np.tanh(prior + incoming))
            node_activation[target] = saturated
            if target not in field_activation:
                new_nodes_added += 1
                vecs = node_vectors.get(target, {})
                field_activation[target] = _field_cosines_for_node(
                    blended, vecs, norm_q=norm_q)

        trace_steps.append({'step': step, 'new_nodes': new_nodes_added,
                           'edges_considered': len(edges),
                           'edges_transmitted': transmitted_count,
                           'max_act': max(node_activation.values()),
                           'threshold': threshold})

        # No-change check: if nothing moved meaningfully, stop early
        delta = max((node_activation[n] - pre_activation.get(n, 0.0)
                    for n in node_activation), default=0.0)
        if delta < 0.01:
            break

    return {
        'node_activation': node_activation,
        'field_activation': field_activation,
        'trace': trace_steps,
    }


# ═══════════════════════════════════════════════════════════════
# EDGE SELECTION — per-node edge ranking for DISPLAY
#
# Used when a single activated node is being rendered and we need to pick
# which of its edges to show (typically 3). Scoring is the same MAX formula
# as the spreading-activation kernel uses for transmission — one rule at
# two granularities (node-level and edge-level).
#
#   edge_score = max(
#       cos(query, target_node_embedding),
#       cos(query, enriched_edge_text)
#   )
#
# No blend weights. No family_boost term (family meaning is inlined into
# the enriched text, so it rides inside the second signal). No fatigue
# discount (novelty is handled at the activation-set level, not per edge).
# No weight tiebreaker (edge weights are static ~0.6 across the graph
# today — they're not signal yet).
#
# Multi-turn query blending is still applied: when a prior turn's query
# vector is available, the effective query is a weighted average that
# smooths single-word follow-ups.
# ═══════════════════════════════════════════════════════════════

# Multi-turn query blending (current turn, previous, two-back)
TURN_WEIGHTS = [0.6, 0.3, 0.1]


def _cosine_nonneg(a, b, norm_a=None):
    """Non-negative cosine. Computes norm_b on the fly; caller may pre-pass
    norm_a since it's invariant across edges in a single select_edges call."""
    import numpy as np
    if a is None or b is None or len(a) != len(b):
        return 0.0
    if norm_a is None:
        norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, float(np.dot(a, b)) / (norm_a * norm_b))


def select_edges(connections, query_vec, limit=3, prior_vecs=None,
                 brain_conn=None, brain=None):
    """Pick the top-N edges of a single node for display rendering.

    Used by the Haiku candidate menu (daemon_hooks) and the inject's seed
    render (_prepare_seed_edges) — a node shows only a few edges, and this
    function ranks them by fit to the query.

    Scoring: max(cos(query, target_node_embedding), cos(query, enriched_edge_text)).
    Family meaning is composed into the enriched text via
    AspectRegistry.compose_edge_text (see aspects.py), so it contributes
    automatically to the second term without a separate signal. After
    schema v26, the edge text vector is read from edge_relations.embedding;
    only rows with NULL embedding fall through to live compose + fastembed.

    Args:
        connections: list of edge dicts (from get_rich_node().connections OR
            from GraphDAL.get_neighbors()). Required keys: id, relation;
            useful keys: title/target_title, description.
        query_vec: numpy array (768d) — current query embedding.
        limit: max edges to return.
        prior_vecs: prior-turn query embeddings for multi-turn blending.
        brain_conn: sqlite3 connection — used to load stored target node
            embeddings. Without it, node_rel falls back to 0 and scoring
            leans on enriched_edge only.
        brain: Brain instance (required) — its aspect registry composes the
            enriched edge text for edges with no stored embedding, and it
            logs read failures.

    Returns:
        list of edge dicts (top N by score), same shape as input.
        On empty input or missing query_vec, returns the input unchanged
        (truncated to limit) — no graceful reordering.
    """
    import numpy as np

    if not connections or query_vec is None:
        return connections[:limit]

    # `brain` is required: fallback path calls brain.aspects.compose_edge_text
    # when a stored edge embedding is missing. Callers (daemon_hooks.py,
    # _prepare_seed_edges, eval/frame_replay.py) always pass brain. No brain
    # → raise: a composer other than the AspectRegistry's would produce
    # different text, so the embedding geometry would silently drift.
    if brain is None:
        raise ValueError(
            "select_edges requires a Brain instance — the fallback edge-text "
            "composer used to silently drop family meaning, producing a "
            "different vector than the AspectRegistry composer. If you're "
            "calling this from a test or eval script, pass the same brain you "
            "loaded the connections from.")

    # Multi-turn blend — fold prior queries into effective query vector
    if prior_vecs:
        ws = TURN_WEIGHTS[:len([query_vec] + prior_vecs)]
        total = sum(ws)
        ws = [w / total for w in ws]
        blended = sum(w * v for w, v in zip(ws, [query_vec] + prior_vecs))
        nm = np.linalg.norm(blended)
        if nm > 0:
            blended = blended / nm
    else:
        blended = query_vec
    norm_q = float(np.linalg.norm(blended))

    # Family meaning composition is owned by AspectRegistry — see
    # aspects.py:compose_edge_text. The fallback path below calls
    # brain.aspects.compose_edge_text(rel, desc) directly when a stored
    # edge embedding is missing; no per-call dict-building needed.

    # Batch-load target node embeddings (one SQL round-trip).
    # Connections carry full UUIDs, so use IN with full IDs — indexed
    # lookup. Dedupe so two connections to the same target only consume
    # one SQL parameter.
    full_target_ids = list({c.get('id', '') for c in connections
                            if c.get('id')})
    stored_embeddings = {}
    if brain_conn is not None and full_target_ids:
        try:
            ph = ','.join(['?'] * len(full_target_ids))
            rows = brain_conn.execute(
                "SELECT node_id, embedding FROM node_enrichments "
                "WHERE vector_type = '_primary' AND node_id IN (%s)" % ph,
                full_target_ids).fetchall()
            for full_id, blob in rows:
                stored_embeddings[full_id] = np.frombuffer(blob, dtype=np.float32)
        except Exception as _e:
            # node_enrichments query failure: log and proceed without
            # node-cosine signal. Scoring falls back to enriched-edge
            # cosine alone. Symmetric with the edge_relations fetch
            # below — both are best-effort, neither breaks recall.
            try:
                brain._log_error(
                    'select_edges_node_embed', _e,
                    'bulk fetch of target node embeddings — '
                    'falling back to enriched-edge-only scoring')
            except Exception:
                pass

    # Bulk-load stored edge embeddings (schema v26+) via the shared
    # helper. See _bulk_load_stored_edge_embeddings docstring.
    edge_keys = [(c.get('edge_id'), c.get('relation') or '')
                 for c in connections]
    stored_edge_embeddings = _bulk_load_stored_edge_embeddings(
        brain_conn, edge_keys, brain=brain,
        error_label='edge_embedding_read_select_edges')

    # Edges WITHOUT a stored embedding fall through to live compose +
    # fastembed. Build the enriched text only for those.
    enriched_blobs = [None] * len(connections)
    needs_embed_idx = []
    needs_embed_text = []
    for i, c in enumerate(connections):
        key = (c.get('edge_id'), c.get('relation') or '')
        stored = stored_edge_embeddings.get(key)
        if stored is not None:
            enriched_blobs[i] = stored
        else:
            # Live compose for a missing blob (not yet filled by the
            # embed_queue worker, or model-stale). Connections
            # come from get_neighbors (key 'edge_description') OR
            # get_rich_node (key 'description'); accept either.
            rel = c.get('relation', '') or ''
            desc = (c.get('description') or
                    c.get('edge_description') or '')
            text = brain.aspects.compose_edge_text(rel, desc)
            needs_embed_idx.append(i)
            needs_embed_text.append(text)
    if needs_embed_text:
        live_blobs = _desc_vecs_batched(needs_embed_text)
        for idx, blob in zip(needs_embed_idx, live_blobs):
            enriched_blobs[idx] = blob

    # Score each edge: MAX of node-target and enriched-edge cosines
    scored = []
    for c, enriched_blob in zip(connections, enriched_blobs):
        target_vec = stored_embeddings.get(c.get('id', ''))
        node_rel = (_cosine_nonneg(blended, target_vec, norm_a=norm_q)
                    if target_vec is not None else 0.0)

        if enriched_blob is not None:
            enriched_vec = np.frombuffer(enriched_blob, dtype=np.float32)
            enriched_rel = _cosine_nonneg(blended, enriched_vec, norm_a=norm_q)
        else:
            enriched_rel = 0.0

        scored.append((max(node_rel, enriched_rel), c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]


# ═══════════════════════════════════════════════════════════════
# INJECT RENDER — seeds first, whole or nothing, neighbors as one line
# ═══════════════════════════════════════════════════════════════


def _event_time_line(node):
    """If node has event_time in kv metadata, return a structured render line
    exposing it (absolute date). The encoder writes event_time as kv, and
    without this line the reader only meets the date buried in content prose.

    Returns '' when no event_time is present, so callers can unconditionally
    prepend without guarding.
    """
    kv = (node.get('metadata_kv') or node.get('kv') or {})
    if not isinstance(kv, dict):
        return ''
    et = kv.get('event_time') or kv.get('event_date')
    if not et:
        # Also peek into _metadata where some render paths copy kv
        m = node.get('_metadata') or {}
        if isinstance(m, dict):
            et = m.get('event_time') or m.get('event_date')
    if not et:
        return ''
    et_str = str(et).strip()
    if not et_str:
        return ''
    # Absolute date only — no relative-to-wall-clock gloss.
    # The relative would anchor to brain_now() (real today) which is the
    # WRONG reference frame for the answerer: in eval the conversation's
    # question_date is its "now"; in production it's wall-clock and the
    # answerer can derive the delta from created_at/updated_at if needed.
    # Adding "X mo ago" relative-to-wall-clock is noise, not signal.
    return '  Event date: %s' % et_str


def _prepare_seed_edges(node, query_vec, brain, seen_root_ids):
    """A seed's edges for its block: drop edge-lines to nodes already
    rendered as roots in this inject (their target is shown in full above,
    so the line is a redundant restatement), then re-rank the rest by query
    relevance so the pertinent bridges come first. Conservative —
    seen_root_ids only ever holds nodes rendered BEFORE this one, so a
    dropped edge always has its target shown above. Returns a copy."""
    seed = dict(node)
    connections = seed.get('connections') or []
    if seen_root_ids and connections:
        connections = [c for c in connections
                       if c.get('id') not in seen_root_ids]
    if query_vec is not None and connections:
        connections = select_edges(
            connections, query_vec, limit=10,
            brain_conn=brain.conn if brain is not None else None,
            brain=brain)
    seed['connections'] = connections
    return seed


def _render_seed(node, mode, content_cap, query_vec=None, brain=None,
                 seen_root_ids=None, scope=None, edges=True,
                 prepare_edges=True, force_cut=False):
    """Render one Haiku pick with at most `content_cap` chars of content.

    Returns (text, meta); meta = {'content_chars': N, 'content': disposition}
    with disposition 'whole' | 'cut' | 'omitted' | 'none' (no content).

      - 'background': the header line (+ event date) only.
      - 'fact': content whole — verbatim is the point (`force_cut` overrides
        it, for the hard-cap ladder).
      - 'arc': whole when it fits `content_cap`; a sentence-boundary cut
        naming the remainder when at least SURFACE_CONTENT_MIN_CHARS fit;
        otherwise no content and a pointer line saying how long it is.
        `content_cap=None` means no cap.

    `edges=False` drops the Edges block (the ladder's second rung);
    `prepare_edges=False` says the caller already ran _prepare_seed_edges.
    """
    from servers.contract import render_rich_node
    event_line = _event_time_line(node)

    def _after_header(body, line):
        head, _, tail = body.partition('\n')
        return head + '\n' + line + ('\n' + tail if tail else '')

    content = node.get('content') or ''
    meta = {'content_chars': len(content), 'content': 'none'}
    if mode == 'background':
        text = render_rich_node(node, seed_render_cfg('background', scope=scope))
        return (_after_header(text, event_line) if event_line else text), meta

    seed = (_prepare_seed_edges(node, query_vec, brain, seen_root_ids)
            if prepare_edges else dict(node))
    if not edges:
        seed['connections'] = []

    if not content:
        limit = 0
    elif content_cap is None or (mode == 'fact' and not force_cut) \
            or len(content) <= content_cap:
        limit, meta['content'] = None, 'whole'
    elif content_cap >= SURFACE_CONTENT_MIN_CHARS:
        limit = max(SURFACE_CONTENT_MIN_CHARS,
                    content_cap - SURFACE_CUT_MARKER_CHARS)
        meta['content'] = 'cut'
    else:
        limit, meta['content'] = 0, 'omitted'

    text = render_rich_node(seed, seed_render_cfg(mode, scope=scope,
                                                  content_limit=limit))
    if meta['content'] == 'omitted':
        text = _after_header(
            text, '  (content: %d chars — get_nodes)' % len(content))
    if event_line:
        text = _after_header(text, event_line)
    return text, meta


def _also_lit_block(neighbors, rich_nodes):
    """Peripheral vision — one title line per lit neighbor; the reader
    fetches what it wants with get_nodes."""
    if not neighbors:
        return ''
    from servers.contract import render_skinny_node
    lines = ['Also lit (%d):' % len(neighbors)]
    for nid in neighbors:
        n = rich_nodes.get(nid) or {}
        # A lure — the line exists so a neighbour worth fetching gets
        # fetched. render_skinny_node owns the title cut for both its
        # callers; this one just hands it the whole title.
        lines.append('  ' + render_skinny_node(
            {'id': nid, 'type': n.get('type', '?'),
             'title': n.get('title') or '?'}))
    return '\n'.join(lines)


def _inject_log(brain, level, source, message, context=''):
    """Log through the brain's own loggers, which never raise; a brain-less
    render (tests, offline eval) has nowhere to log."""
    if brain is None:
        return
    if level == 'error':
        brain._log_error(source, RuntimeError(message), context)
    else:
        brain._log_warning(source, message, context)


def render_surface_inject(node_activation, field_activation, rich_nodes,
                          selected_mode=None, query_vec=None, brain=None,
                          total_budget=SURFACE_INJECT_BUDGET, scope=None,
                          seen_ids=None):
    """Render the inject: Haiku's picks in full, then one line per lit neighbor.

    Returns {'text', 'shown', 'not_shown', 'also_lit'}:
      shown     — full ids of the seeds that rendered, in render order
      not_shown — [{'id': short id, 'reason'}] for seeds the render could
                  not deliver: 'no_node' (not in rich_nodes) or 'budget'
                  (the hard-cap ladder's last rung — logged as an error)
      also_lit  — full ids of the neighbors rendered as title lines

    Budget: the seeds split `total_budget` (less the neighbor-list
    reservation) in activation order, each rendered once and cut only when
    its whole block outgrows its share; what one leaves unused flows to the
    next, and a second pass hands cut seeds the soft leftover plus, when a
    pick overran its share, the room below the hard cap up to that overrun
    — one long pick must not starve the others while the inject sits under
    the ceiling. Nothing scales with how many nodes spread lit. Over the
    hard cap the block shrinks from its
    least valuable end (the neighbor list, then edges, then content to the
    minimum, then the lowest seeds) and never mid-line.

    `seen_ids`: nodes already shown to this stream in the window — never a
    neighbor line (they are in context already).
    """
    selected_mode = selected_mode or {}
    node_activation = node_activation or {}
    field_activation = field_activation or {}
    rich_nodes = rich_nodes or {}
    seen8 = {str(s)[:8] for s in (seen_ids or ()) if s}
    out = {'text': '', 'shown': [], 'not_shown': [], 'also_lit': []}
    if not selected_mode:
        return out

    def _mean_field(nid):
        fa = field_activation.get(nid) or {}
        return (sum(fa.values()) / len(fa)) if fa else 0.0

    def _mode(nid):
        return selected_mode.get(nid) or SURFACE_MODE_DEFAULT

    # Seeds in activation order (Haiku's pick order breaks ties): the order
    # they render in, and the order the hard-cap ladder shrinks from the end.
    pick_order = {nid: i for i, nid in enumerate(selected_mode)}
    seeds = sorted(selected_mode, key=lambda nid: (
        -node_activation.get(nid, 0.0), -_mean_field(nid), pick_order[nid]))
    for nid in [s for s in seeds if s not in rich_nodes]:
        out['not_shown'].append({'id': nid[:8], 'reason': 'no_node'})
    seeds = [s for s in seeds if s in rich_nodes]
    if not seeds:
        return out

    # Edges once per seed (a background seed renders none — no ranking
    # work for it). Then the neighbors: lit, not a seed, not in the window's
    # shown set, and not already named by a seed's edge line — the edge line
    # carries the relation, the richer mention.
    prepared, roots = {}, set()
    for nid in seeds:
        prepared[nid] = (dict(rich_nodes[nid]) if _mode(nid) == 'background'
                         else _prepare_seed_edges(rich_nodes[nid], query_vec,
                                                  brain, roots))
        roots.add(nid)
    edge_limit = SURFACE_SEED_FORMAT['edge_limit']
    named = {c.get('id') for nid in seeds if _mode(nid) != 'background'
             for c in (prepared[nid].get('connections') or [])[:edge_limit]}
    neighbors = [nid for nid, _act in sorted(node_activation.items(),
                                             key=lambda kv: -kv[1])
                 if nid not in selected_mode and nid in rich_nodes
                 and nid[:8] not in seen8 and nid not in named
                 ][:SURFACE_ALSO_LIT_LIMIT]
    also_lit = _also_lit_block(neighbors, rich_nodes)
    reserve = len(also_lit) + 2 if also_lit else 0

    def _render(nid, cap, **kw):
        return _render_seed(prepared[nid], _mode(nid), cap, scope=scope,
                            prepare_edges=False, **kw)

    # Pass 1 — fair share with carry-forward. Render whole; cut only when
    # the whole block outgrows the share, to what the share leaves after
    # the block's skeleton (header, quote, corrections, edges).
    blocks, metas, caps = {}, {}, {}
    remaining = total_budget - reserve
    overrun = 0   # chars picks took beyond their share (whole fact content)
    for i, nid in enumerate(seeds):
        share = max(0, remaining) // (len(seeds) - i)
        text, meta = _render(nid, None)
        cap = None
        if (meta['content'] == 'whole' and _mode(nid) != 'fact'
                and len(text) > share):
            skeleton = len(text) - len(prepared[nid].get('content') or '')
            cap = max(0, share - skeleton)
            text, meta = _render(nid, cap)
        blocks[nid], metas[nid], caps[nid] = text, meta, cap
        overrun += max(0, len(text) + 2 - share)
        remaining -= len(text) + 2

    def _assemble(seed_list, with_neighbors):
        parts = [blocks[nid] for nid in seed_list]
        if with_neighbors and also_lit:
            parts.append(also_lit)
        return '\n\n'.join(parts)

    # Pass 2 — leftover flows back to seeds whose content was cut or left
    # out, split evenly. What the soft target has left, plus — when a pick
    # overran its share — the room below the hard cap, up to that overrun:
    # one long pick must not starve the others while the inject sits under
    # the ceiling, and the soft target still governs when nothing overran.
    cut = [nid for nid in seeds if metas[nid]['content'] in ('cut', 'omitted')]
    if cut:
        headroom = ((_MAX_INJECT_CHARS - SURFACE_HARD_CAP_MARGIN)
                    - len(_assemble(seeds, True)))
        leftover = max(remaining, min(overrun, headroom))
        floor = SURFACE_CONTENT_MIN_CHARS + SURFACE_CUT_MARKER_CHARS
        for i, nid in enumerate(cut):
            if leftover <= SURFACE_CUT_MARKER_CHARS:
                break
            # An even split below the content minimum would leave every
            # starved pick a pointer; when the room is short, the top-ranked
            # picks reach the minimum first and the tail stays honest.
            grow = leftover // (len(cut) - i)
            if grow < floor:
                grow = min(leftover, floor)
            # A cut seed's block already holds its pass-1 cap of content; an
            # omitted seed's holds none, so its cap starts from zero or the
            # block would grow by cap + grow and outrun the grant.
            base = caps[nid] if metas[nid]['content'] == 'cut' else 0
            new_cap = (base or 0) + grow
            text, meta = _render(nid, new_cap)
            leftover -= len(text) - len(blocks[nid])
            blocks[nid], metas[nid], caps[nid] = text, meta, new_cap

    shown = list(seeds)
    with_neighbors = bool(also_lit)
    text = _assemble(shown, with_neighbors)
    if len(text) > _MAX_INJECT_CHARS:
        _inject_log(brain, 'warning', 'surface_inject_shrink',
                    'inject %d > cap %d — shrinking: neighbors, edges, '
                    'content, seeds' % (len(text), _MAX_INJECT_CHARS),
                    'seeds=%d neighbors=%d' % (len(shown), len(neighbors)))
        with_neighbors = False
        text = _assemble(shown, False)
        # Edges off, lowest seed first.
        for nid in reversed(shown):
            if len(text) <= _MAX_INJECT_CHARS:
                break
            blocks[nid], metas[nid] = _render(nid, caps[nid], edges=False)
            text = _assemble(shown, False)
        # Content down to the minimum, lowest seed first — only where more
        # than the minimum is showing (an omitted or short block would GROW).
        for nid in reversed(shown):
            if len(text) <= _MAX_INJECT_CHARS:
                break
            if (metas[nid]['content'] not in ('whole', 'cut')
                    or metas[nid]['content_chars']
                    <= SURFACE_CONTENT_MIN_CHARS + SURFACE_CUT_MARKER_CHARS):
                continue
            blocks[nid], metas[nid] = _render(
                nid, SURFACE_CONTENT_MIN_CHARS, edges=False, force_cut=True)
            text = _assemble(shown, False)
        # Last rung: keep the seeds that fit, in rank order — the shared
        # fit-under-cap composer (an oversize first seed still shows).
        if len(text) > _MAX_INJECT_CHARS:
            from servers.loud_truncation import compose_block_loud
            text, kept, dropped = compose_block_loud(
                shown, lambda nid: blocks[nid], _MAX_INJECT_CHARS)
            for nid in shown[kept:]:
                out['not_shown'].append({'id': nid[:8], 'reason': 'budget'})
            shown = shown[:kept]
            if dropped:
                _inject_log(brain, 'error', 'surface_inject_overflow',
                            'inject over cap %d after shrinking — %d pick(s) '
                            'dropped' % (_MAX_INJECT_CHARS, dropped),
                            'seeds_left=%d' % len(shown))
            if len(text) > _MAX_INJECT_CHARS:
                # One seed, still over: cut at a line boundary rather than
                # lose the inject to the spill file.
                _inject_log(brain, 'error', 'surface_inject_overflow',
                            'single seed %d > cap %d — cut at a line boundary'
                            % (len(text), _MAX_INJECT_CHARS), 'seeds=1')
                text = text[:_MAX_INJECT_CHARS].rsplit('\n', 1)[0]

    out['text'] = text
    out['shown'] = shown
    out['also_lit'] = neighbors if with_neighbors else []
    return out
