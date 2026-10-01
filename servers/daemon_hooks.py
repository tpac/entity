"""Centralized hook logic — all hook brain interactions live here.

Previously each hook .py file had a _run_daemon() and _run_direct() path,
duplicating logic and diverging over time. Now hooks are thin clients that
send a single command to the daemon, and this module contains all the logic.

Each function signature: hook_*(brain, args, graph_changes) -> dict
  - brain: Brain instance (already loaded by daemon)
  - args: dict from the hook client
  - graph_changes: list[str] — in-memory mutation log, drained by hook_recall

Returns: {"output": str} for text hooks, or {"output": str, "json": dict}
for hooks that need structured JSON output (recall, pre-edit, pre-bash).
"""

import json
import os
import time

from servers.brain_traces import _s0_trace
from servers.brain_voice import BrainVoice
from servers.contract import TITLE_WARN_LIMIT
from servers.loud_truncation import cap_text_loud
# Hoisted — if this import ever breaks, the daemon fails at boot instead of
# silently degrading recall 16s in.
from servers.embedder import embed_query
from servers.daemon_config import brain_tmp_dir
from servers.trace_contract import REGISTER_ONLY_FIELD


DESTRUCTIVE_REGEXES = [
    r"rm\s+(-[rf]+\s+|.*--force)",
    r"git\s+worktree\s+remove",
    r"git\s+reset\s+--hard",
    r"git\s+clean\s+-[fd]",
    r"git\s+checkout\s+--\s",
    r"git\s+push\s+.*--force",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
    r"TRUNCATE",
    r"\brmdir\b",
    r"xargs\s+rm",
]

ENV_CHANGE_PATTERNS = [
    r"\bpip\b.*\binstall\b", r"\bpip\b.*\buninstall\b",
    r"\bbrew\b.*\binstall\b", r"\bbrew\b.*\buninstall\b",
    r"\bapt\b.*\binstall\b", r"\bnpm\b.*\binstall\b",
    r"\bcargo\b.*\binstall\b", r"\bgem\b.*\binstall\b",
    r"\bpyenv\b", r"\bnvm\b.*\buse\b", r"\bnvm\b.*\binstall\b",
    r"\bconda\b.*\binstall\b", r"\bconda\b.*\bactivate\b",
]


# ── Helpers ──


from .scales.s1.surface import run_surface as _run_surface


class PhaseTimer:
    """Lightweight per-call phase timer — emits one log line at the end.

    Usage:
        pt = PhaseTimer(enabled=brain.get_config('phase_timing.enabled', True))
        ... work ...
        pt.mark('recall')
        ... work ...
        pt.mark('candidates')
        ...
        pt.log(brain, 'hook_recall')      # emits a single debug line

    Each mark records elapsed-since-last-mark in milliseconds. Order is
    preserved. Total wall time is also reported.

    Why it exists: hook_recall has ~13 phases (recall, candidates,
    enrichment, edge selection, traces, signal producers,
    Frame build, surface call, etc.). When the end-to-end takes 12 s we
    need to know which phase ate the budget. Single log line keeps the
    daemon log readable without a per-phase line storm.

    Cost when enabled: ~1µs per mark, ~50µs for the final log line. 14
    phases ≈ 14µs of CPU and one log line per recall. Default ON because
    the diagnostic value swamps the overhead — you should know where
    recall's seconds went, always.

    Disable via `brain.get_config('phase_timing.enabled', False)` or via
    `phase_timing.enabled = False` in the config table. When disabled,
    `mark()` and `log()` are no-ops and the timer holds no state — the
    cost goes to zero.
    """

    __slots__ = ('_enabled', '_t0', '_last', '_phases')

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        if not enabled:
            self._t0 = 0.0
            self._last = 0.0
            self._phases = []
            return
        now = time.monotonic()
        self._t0 = now
        self._last = now
        self._phases: list[tuple[str, int]] = []

    def mark(self, label: str) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        self._phases.append((label, int((now - self._last) * 1000)))
        self._last = now

    def snapshot(self) -> list:
        """Per-phase breakdown captured so far, as a JSON-friendly list of
        {phase, ms} dicts (marks recorded after this call aren't included).

        The structured, queryable form of the same data the final log() line
        renders into the unqueryable 'hook_phase_timing' string — surface
        writes it into its K trace so 'which phase ate the budget' is a query,
        not a log grep. Empty list when the timer is disabled.
        """
        return [{'phase': label, 'ms': ms} for label, ms in self._phases]

    def log(self, brain, hook_name: str, **extras) -> None:
        """Emit one debug line. extras land as key=value pairs at the end."""
        if not self._enabled:
            return
        total_ms = int((time.monotonic() - self._t0) * 1000)
        body = ' '.join('%s:%dms' % (label, ms) for label, ms in self._phases)
        if extras:
            body += ' ' + ' '.join('%s=%s' % (k, v) for k, v in extras.items())
        msg = '[%s] total:%dms %s' % (hook_name, total_ms, body)
        try:
            brain.log_debug('hook_phase_timing', msg)
        except Exception:
            # Logging must never affect the hook path. If brain.log_debug
            # is unavailable (test harness, unusual init) fall back to print.
            print(msg, flush=True)







# ══════════════════════════════════════════════════════════════════════════════
# HOOK FUNCTIONS — one per hook event
# ══════════════════════════════════════════════════════════════════════════════


def hook_recall(brain, args, graph_changes):
    """Pre-response recall — surfaces brain context before Claude responds.

    Fires on UserPromptSubmit. Returns JSON with additionalContext.

    Flow:
    1. Session setup + store user message
    2. Recall candidates from brain
    3. Enrich candidates (rich-node pull + edge selection + recent turns)
    4. Priming check, signal producers
    5. S1 Surface (Haiku) → select → expand → format → trace
    6. Return additionalContext or approve

    Phase timing: every recall emits a single 'hook_phase_timing' debug
    line splitting the total wall time across phases. See PhaseTimer.
    Use it to find which phase ate the budget when latencies spike;
    don't waste time guessing where seconds went.

    Gated by `phase_timing.enabled` in brain config (default True).
    Flip to False to silence the per-recall log line entirely; mark/log
    become no-ops and the cost goes to zero.
    """
    pt = PhaseTimer(
        enabled=bool(brain.get_config('phase_timing.enabled', True)))
    user_message = args.get("prompt", "")
    ctx = brain.get_or_create_session(args.get('session_id', ''))
    session_id = ctx.session_id
    _current_stop = str(ctx.stop_counter)
    # Mark that a real UserPromptSubmit ran recall for THIS stop. The Stop hook
    # reads this to classify the turn: recall ran → conversational. A /watch
    # wakeup skips recall client-side (pre_response_recall.py: slash/bang/short),
    # so its stop never reaches here → it reads as a heartbeat. The client has
    # already filtered, so reaching hook_recall means a real prompt.
    # See trace_contract S0 TURN CLASSIFICATION.
    ctx.last_recall_stop = ctx.stop_counter
    _set_hook_env(ctx, args.get('model', ''), args.get('tells'))

    # Write the user_message S0 trace NOW, at prompt-arrival — not at Stop. This
    # is what lets presence/peek surface a stream's current prompt mid-turn
    # (rendezvous identity), instead of only its last completed turn. The
    # assistant half is still written at Stop (post_response_common); both use
    # ctx.s0_chain() with the same stop_counter, so they stay paired. Reaching
    # hook_recall means a real prompt (client filters watch-wakes), so this is
    # exactly the conversational set — heartbeat turns never get here.
    _user_meta = None
    if user_message:
        _user_meta = {'recall_chain': ctx.s1r_chain()}
        if args.get("register_only"):
            _user_meta[REGISTER_ONLY_FIELD] = True   # no recall/surface below
    _user_msg_trace_id = _s0_trace(
        brain, ctx, event_type='K', ref_type='user_message',
        summary=user_message[:200] if user_message else '',
        metadata=_user_meta,
        content=user_message or None)

    # Write current stop counter to tmp file — PostToolUse reads this (cross-process)
    try:
        with open(os.path.join(brain_tmp_dir(), 'brain-%s-current-stop.txt' % session_id), 'w') as _f:
            _f.write(_current_stop)
    except Exception as e:
        brain._log_error('write_current_stop', e, 'hook_recall')

    # ── Register-only fast path ──
    # Short answers ("yes", "ok", "no") that the client routes here with
    # register_only=True. The turn IS conversational and is now fully registered:
    # the user_message S0 trace + last_recall_stop are written ABOVE, and the
    # daemon reset last_user_activity (the clock is gated on cmd=='hook_recall',
    # which this still is). We skip ONLY the expensive recall + Haiku surface,
    # which carry no signal on a 3-char reply. WITHOUT this branch the client
    # dropped the whole turn before reaching the daemon: no user_message trace,
    # the turn misclassified as a /watch heartbeat at Stop, and the operator's
    # words lost — often the highest-signal turns (approvals/decisions). brain.save()
    # persists the session/ctx mutations made above. See trace_contract S0 TURN
    # CLASSIFICATION and pre_response_recall.py.
    if args.get("register_only"):
        brain.save()
        return {"json": {"decision": "approve"}, "session_id": session_id}

    from .pipeline_contract import PIPELINE as _PL
    enriched = user_message[:_PL['user_message_query']]
    pt.mark('setup')

    # Recall — logging happens inside brain.recall() (single source of truth)
    from .scales.s1.surface_contract import CANDIDATE_POOL as _CF
    try:
        # Pass ctx directly — already loaded above, skip the redundant DB
        # lookup recall would otherwise do. ctx mutations (fatigue) are
        # saved at turn end in post_response_common.
        # Over-fetch by seen_dedup_headroom: the seen-dedup filter below
        # drops already-surfaced nodes, and without headroom recall's own
        # truncation leaves nothing to backfill with — the pool would only
        # shrink (a98143f review, finding 1).
        result = brain.recall(query=enriched,
                              limit=_CF['max_candidates']
                              + _CF.get('seen_dedup_headroom', 0),
                              ctx=ctx, source='hook')
    except Exception as e:
        brain._log_error('recall_first_attempt', e, 'hook_recall')
        result = brain.recall(query=enriched,
                              limit=_CF['max_candidates']
                              + _CF.get('seen_dedup_headroom', 0),
                              ctx=ctx, source='hook')

    results = result.get("results", [])
    pt.mark('recall')

    recall_ref = '%s-%s' % (session_id[:8], _current_stop)

    # Enrich candidates for the surface call: batch rich-node pull, then
    # query-aware edge selection.
    candidates_data = []
    recent_messages = []
    _query_vec = None
    _prior_vecs = []
    try:
        capped = results[:_CF['max_candidates']]

        # Previous turns — ONE pull serves both consumers below (the surface
        # conversation window and the prior-query embedding blend), keeping
        # their notion of "previous" aligned by construction. Exclude the
        # current prompt's own trace row: it was written at prompt-arrival
        # (above) and the current message reaches build_surface_prompt
        # separately as `user_message`, rendered as its own block. Keyed on
        # the trace id, not the chain — after an interrupt the current chain
        # also holds the previous real prompt, which belongs in the window.
        # Drop wake envelopes — task-notification ignites are
        # recorded as user_message traces but are machine payloads, not
        # operator speech; Haiku must not read them as conversation.
        # Limit comes from SURFACE['recent_messages'] so the upstream pull and
        # the downstream slice in build_surface_prompt share one source of truth.
        from .scales.s1.surface_contract import SURFACE as _SURFACE
        from .trace_contract import WAKE_ENVELOPE_MARKER
        turns = []
        try:
            # Over-fetch so filtered wake envelopes don't cost window slots,
            # then trim back to the configured window after filtering.
            turns = brain.get_conversation(
                session_id, limit=_SURFACE['recent_messages'] + 4,
                with_judge_output=False, with_surfaced=True,
                exclude_trace_id=_user_msg_trace_id)
            turns = [t for t in turns
                     if not (t.get('content') or '').startswith(WAKE_ENVELOPE_MARKER)]
            turns = turns[-_SURFACE['recent_messages']:]
            recent_messages = [
                {"role": t['role'],
                 "content": (t['content'] or '')[:_PL['recent_message_content']],
                 "surfaced": t.get('surfaced') or []}
                for t in turns]
        except Exception as _e:
            brain._log_error('surface_recent_messages', _e, 'fetching recent messages from traces')
        pt.mark('traces')

        # Already-shown dedup, BEFORE the final cap — replacements backfill
        # from the seen_dedup_headroom over-fetch at the recall call above
        # (without it recall's own truncation leaves nothing to backfill:
        # a98143f review, finding 1). Window-bounded, not session-bounded:
        # a node re-enters once it slides out of the recent-messages window
        # (deliberate — distance earns a refresher). Within the window it can't re-enter
        # <candidates>. The v13 <shown> prompt rule alone doesn't hold —
        # Haiku re-picked shown nodes with the element in-prompt (2026-07-27
        # capture) — so out-of-scope is now structural, in code. The seen
        # set reuses the turns pulled above (no extra query). The same set
        # gates Haiku's tool results and the final selection in run_surface:
        # a node the stream already has in context this window does not
        # render again through any door (the stream can get_nodes it). Falls
        # back to the plain cap when the turns pull failed (recent_messages
        # empty → seen empty).
        from .scales.s1.surface import seen_node_ids
        _seen = seen_node_ids(recent_messages)
        if _seen:
            kept = [r for r in results
                    if str(r.get('id') or '')[:8] not in _seen]
            n_dropped = len(results) - len(kept)
            if n_dropped:
                capped = kept[:_CF['max_candidates']]
                stats = result.get('_retrieval_stats') or {}
                stats['seen_dropped'] = n_dropped
                result['_retrieval_stats'] = stats

        # Batch enrichment: one call, 5 queries for all 25 nodes
        node_ids = [r.get("id", "") for r in capped if r.get("id")]
        rich_nodes = brain.get_node(node_ids)

        # Scope veil for edge-attachment scrubbing below — cache hit (the
        # recall above already built this session's veil, or raised).
        _hook_veil = brain.scope_veil(session_id)

        # Edge selection: query-aware scoring (strategy D)
        # Get query embedding + prior turn embeddings for multi-turn blend
        import numpy as np
        _query_emb = result.get("_query_embedding")
        if _query_emb is not None:
            _query_vec = np.frombuffer(_query_emb, dtype=np.float32) if isinstance(_query_emb, bytes) else np.array(_query_emb, dtype=np.float32)
            # Prior user messages for the multi-turn blend — sliced from the
            # single turns pull above (last 4 messages, first 2 user turns).
            try:
                _user_turns = [t for t in turns[-4:] if t.get('role') == 'user'][:2]
                for _t in _user_turns:
                    _text = (_t.get('content') or '')[:500]
                    if _text and len(_text) > 5:
                        _blob = embed_query(_text)
                        if _blob:
                            _prior_vecs.append(np.frombuffer(_blob, dtype=np.float32))
            except Exception as _e:
                brain._log_error('edge_select_prior_vecs', _e, 'embedding prior turns')

        # recall_score is the shared score semantic — fetch_tools reads the
        # same function, keeping the agentic admission floor comparable.
        from .scales.s1.surface_contract import recall_score
        for r in capped:
            nid = r.get("id", "")
            node_data = rich_nodes.get(nid)
            if not node_data:
                # Fallback if node disappeared between recall and enrichment
                node_data = {
                    "id": nid, "type": r.get("type", ""),
                    "title": r.get("title", ""), "content": r.get("content", ""),
                    "confidence": r.get("confidence", 0), "locked": r.get("locked", False),
                    "created_at": r.get("created_at"), "revised_at": r.get("revised_at"),
                }
            # Scope veil on edge attachments: candidates were gated by
            # recall, but their connections/_corrections still carry walled
            # neighbor titles (and the corrector's full text) into the
            # Haiku menu — scrub before edge selection.
            from .scopes import scrub_node
            scrub_node(node_data, _hook_veil)
            # Query-aware edge selection (S1 intelligence)
            # Render gets the selected subset via edge_limit in config.
            if _query_vec is not None and node_data.get('connections'):
                from .scales.s1.surface_contract import select_edges
                node_data['connections'] = select_edges(
                    node_data['connections'], _query_vec,
                    limit=10,  # keep 10, render truncates to 3
                    prior_vecs=_prior_vecs, brain_conn=brain.conn, brain=brain)
            # Attach recall-specific fields (not in DB — from scoring pipeline)
            node_data["score"] = recall_score(r)
            node_data["discovery"] = r.get("_discovery", "embedding")
            candidates_data.append(node_data)

        pt.mark('candidates')
    except Exception as e:
        brain._log_error('surface_candidates_build', e, 'Failed to enrich candidates for surface')

    if not results:
        # Empty is TWO different facts: "brain knows nothing" (healthy) vs
        # "recall infrastructure failed" (_empty_recall's embedder_unavailable
        # / embed_failed). recall() itself already logs the failure to the
        # errors table; what the hook path swallowed was the MODE in the
        # return value — carry it as a sibling key (the hook client reads
        # only "json"; eval's query_brain reads this to mark the rep suspect).
        brain.save()
        pt.log(brain, 'hook_recall', n_results=0)
        return {"json": {"decision": "approve"},
                "recall_mode": result.get('_recall_mode', '')}

    if not capped:
        # Every candidate was already surfaced this window (seen-dedup ate
        # the whole pool, headroom included) — the context already holds
        # them all. Skip the Haiku call instead of rendering an empty
        # <candidates n="0"> menu (a98143f review, finding 4).
        brain.save()
        pt.log(brain, 'hook_recall', n_results=len(results), all_seen=1)
        return {"json": {"decision": "approve"}}

    brain.save()

    # ── S1 Surface: push relevant memories into awareness ──
    # 2026-05-02 (Frame Phase 2): Frame is the canonical session prior.
    # ctx.get_frame(brain) builds it from this session's state (header +
    # arc). Surface receives it as the "Partnership context:"
    # block. If Frame Constructor raises, the error is logged loudly and
    # surface runs without partnership context (explicit degraded mode,
    # no silent fallback to a different layout). See docs/FRAME-DESIGN.md.
    additional_context = None
    try:
        try:
            _frame = ctx.get_frame(brain)
        except Exception as _frame_err:
            brain._log_error('frame_build_failed', _frame_err,
                             'Frame Constructor failed — surface runs without partnership context')
            _frame = ''
        pt.mark('frame')
        # Thread the same PhaseTimer into surface so it splits the surface
        # phase into haiku / id_resolve / spread / render / trace marks
        # on the same final log line. Without this we only know "surface
        # took N ms" without knowing which sub-phase ate the budget.
        # LLM layer paused (no key, or a key the provider refused): surface
        # is a Haiku call that can only fail — skip it (noted once, not one
        # error per user turn). Recall candidates were still computed
        # locally; MCP recall stays live.
        if brain.llm_available:
            # `results=capped` — traces must count the pool the surface
            # actually saw (post seen-dedup, post cap), not recall's raw
            # over-fetched list (a98143f review, finding 3).
            additional_context = _run_surface(
                brain, ctx, candidates_data, user_message,
                recent_messages=recent_messages,
                result=result, enriched=enriched, results=capped,
                recall_ref=recall_ref, session_id=session_id,
                graph_changes=graph_changes,
                query_vec=_query_vec, prior_vecs=_prior_vecs,
                frame=_frame, pt=pt)
        else:
            brain.note_llm_unavailable('S1 surface')
            # Loud to Claude — the operator must hear this from the
            # assistant, not discover it in the dashboard (first laptop
            # install: boot looked fine, nothing told the session, the
            # operator found the error themselves). Capped at 3 notices per
            # session (operator-set): enough that it can't be missed, not a
            # nag on every turn of a deliberately-keyless session. Counter
            # is per-session (parallel sessions don't share it), in-memory
            # (a daemon restart re-arms it — fine, boots are rare). The text
            # is BrainVoice's, shared with the boot banner, so the two
            # channels cannot disagree about why the layer is paused.
            _counts = getattr(brain, '_llm_paused_notice_counts', None)
            if _counts is None:
                _counts = brain._llm_paused_notice_counts = {}
            if _counts.get(session_id, 0) < 3:
                _counts[session_id] = _counts.get(session_id, 0) + 1
                additional_context = (
                    "[BRAIN]\n%s\n"
                    "If you have not told the operator this session, tell "
                    "them now.\n"
                    "[/BRAIN]" % BrainVoice.llm_paused_notice(brain))
    except Exception as _surface_err:
        brain._log_error('daemon_surface', _surface_err,
                         'S1 Surface failed in daemon (query=%s)' % user_message[:100])

    pt.log(brain, 'hook_recall', n_results=len(results))

    # NOTE: self-message delivery deliberately does NOT happen here. on_prompt
    # (additionalContext) is the weakest channel — it competes with recall/Frame
    # and would lose the consume-once race against higher-salience hooks. A
    # block prepended here also overflowed the inject spill cap, losing context.
    # Delivery lives on Stop alone (block — don't end with an unread tap), via
    # signal.drain_and_render. See docs/SELF-CHANNEL-DESIGN.md.
    if additional_context:
        return {"json": {"additionalContext": additional_context}, "session_id": session_id}
    else:
        return {"json": {"decision": "approve"}, "session_id": session_id}






def _set_hook_env(ctx, model, tells):
    """Refresh session identity only from a resolved prompt/Stop observation.

    An old client's pre-resolved host is no longer an input. Unknown and
    ambiguous observations leave the session's last known host alone.
    """
    from .host_contract import resolve_host
    present = {t: True for t in tells if isinstance(t, str)} \
        if isinstance(tells, (list, tuple)) else {}
    host, status, _ = resolve_host(present)
    ctx.set_env(model=model)
    if status in ('strong', 'family'):
        ctx.set_env(host=host)


def post_response_common(brain, session_id, assistant_response, *,
                         model='', tells=None):
    """Shared post-response path: S0 traces, heartbeat, stop counter
    increment. Used by prod Stop hook and by the eval harness —
    same code, same ordering, one source of truth.

    `model` / `tells`: raw Stop observations. The daemon resolves the host;
    only strong/family resolution updates the session's latest identity.
    Empty (the eval harness) leaves the known value.

    Returns the SessionContext after increment.
    """
    ctx = brain.get_or_create_session(session_id)
    _set_hook_env(ctx, model, tells)
    # No pre-cap here: _s0_trace owns the one (loud) stored-content cap; a
    # second slice against the same constant is how the sides drift apart.
    assistant_response = assistant_response or ""

    # Turn classification (trace_contract S0 TURN CLASSIFICATION): a turn is
    # conversational iff a real UserPromptSubmit ran hook_recall THIS stop (which
    # sets last_recall_stop). A /watch wakeup skips recall client-side, so its
    # stop never matches → heartbeat. Heartbeats are recorded for observability
    # but never enter the conversation stream and never tick the Scribe cadence.
    # last_turn_conversational records the classification on the ctx (diagnostic
    # / potential consumers). The Scribe cadence itself no longer reads it — the
    # poll-driven reactor derives the count from traces (turns_since_last_encode
    # counts s0 user_message turns), which heartbeats never write.
    is_conversational = (ctx.last_recall_stop == ctx.stop_counter)
    # Delivery-continuation: this stop is the reaction to a delivery that
    # blocked the previous one — the Stop hook arms the stamp only for a
    # dial-on, traced correspondent (trace_contract.arms_continuation), so
    # dial-off leaves this structurally dead and the stop stays a heartbeat.
    # One-shot read-and-clear; the freshness window is the real guard against
    # a stale stamp — an ESC'd continuation fires no Stop, so counter and
    # stamp freeze, and a /watch wakeup hours later would otherwise match.
    # Cadence is untouched either way (it counts user_message rows only).
    from .clock import iso_cutoff as _iso_cutoff
    from .trace_contract import DELIVERY_REACTION_WINDOW_MIN as _RW
    armed_stop, armed_at = ctx.last_delivery_stop, ctx.last_delivery_armed_at
    ctx.last_delivery_stop, ctx.last_delivery_armed_at = -1, ''
    is_delivery_reaction = (
        not is_conversational
        and armed_stop == ctx.stop_counter
        and armed_at >= _iso_cutoff(minutes=_RW))
    ctx.last_turn_conversational = is_conversational or is_delivery_reaction

    # S0 traces (using SessionContext for chain IDs)
    try:
        if is_conversational or is_delivery_reaction:
            # For an operator turn: user_message was written at
            # UserPromptSubmit (hook_recall), when the prompt ARRIVED — so
            # presence/peek can surface a stream's current prompt mid-turn.
            # Only the assistant half is written here, at Stop; stop_counter
            # is unchanged between hook_recall and this Stop (incremented
            # below), so the pair shares the chain. For a delivery REACTION:
            # the incoming K was written by deliver() at the PREVIOUS stop's
            # hook, after that stop's increment — which put it on THIS chain,
            # so the pair shares the chain by the same invariant from the
            # other side.
            _s0_trace(
                brain, ctx, event_type='delta', ref_type='assistant_message',
                summary=assistant_response[:200],
                content=assistant_response or None)
        else:
            # Heartbeat: wakeup re-arm, no real prompt. One observability marker
            # (off CONVERSATIONAL_REF_TYPES → never encoded). The peer message,
            # if any, is recorded separately as a self_message on drain.
            _s0_trace(
                brain, ctx, event_type='K', ref_type='heartbeat',
                summary=(assistant_response[:200] or 'wakeup re-arm'))
    except Exception as e:
        brain._log_error('trace_s0', e, 'post_response_common')

    # stop_counter is the per-stop SEQUENCE number — it advances on EVERY stop
    # (incl. heartbeats) so S0/S1 chain IDs stay unique. The integration CADENCE
    # the Scribe gates on is NOT a counter here — it's derived live from traces
    # (turns_since_last_encode counts s0 user_message turns), so there's nothing
    # to tick on this path. Heartbeats are excluded structurally: they write no
    # user_message trace, so they can't advance that count. See trace_contract
    # S0 TURN CLASSIFICATION.
    # Flush the per-turn "Anchor touched" accumulator as one anchor_touched S0
    # delta on THIS turn's chain (before increment, so the stop matches). The S0
    # mirror of the encoder's ops-delta — feeds the encoder's widened catalog via
    # trace_links. Snapshot-and-swap FIRST (one rebind, atomic under the GIL): a
    # concurrent lock-free read that already holds the old dict extends a
    # detached list, so the next turn starts clean either way — at worst a touch
    # landing in this exact window is dropped (advisory feed; Stop fires post-turn
    # so same-session reads aren't normally in flight). Only written when non-empty.
    try:
        pending, ctx.touched = ctx.touched, {k: [] for k in ctx.touched}
        if any(pending.values()):
            from .trace_contract import build_anchor_touched_metadata
            meta = build_anchor_touched_metadata(**pending)
            n = sum(len(v) for v in meta.values())
            _s0_trace(brain, ctx, event_type='delta', ref_type='anchor_touched',
                      summary='%d nodes touched' % n, metadata=meta)
    except Exception as e:
        brain._log_error('anchor_touched_flush', e, 'post_response_common')

    ctx.increment_stop()
    return ctx


def hook_post_response_track(brain, args, graph_changes):
    """Stop event — record the turn's assistant half (S0 traces), deliver self-messages.

    The S1 Scribe is NO LONGER triggered here. Encoding cadence moved to the
    daemon's poll loop (brain.scribe_due → daemon._run_scribe_poll): the Stop
    hook just records the turn — keeping the trace log, the cadence's source of
    truth, current — and delivers pending self-messages. One trigger owner (the
    poll) means no hook/poll double-fire race.
    """
    # Loud at the write boundary: every Stop payload must say which model
    # produced the turn (Codex puts it on the hook payload; Claude Code's
    # transcript carries it on every assistant entry — hook_common.turn_model
    # reads it). A gap here means the S0 record goes down without a model, and
    # a boot-time check would fire once and be easy to bypass — this fires at
    # the write, per session (the error text carries the session, so the
    # dedup fingerprint is per stream: one row per session per dedup window,
    # and a second stream's gap is never masked by the first's). Logged, never
    # blocking: the turn is still recorded.
    if args.get("hook_event_name") == "Stop" and not args.get("model"):
        _sid_short = (args.get('session_id', '') or '')[:8]
        brain._log_error(
            's0_model_unset',
            ValueError('no model on Stop for session %s' % _sid_short),
            'the turn\'s S0 rows carry no model stamp')
    ctx = post_response_common(
        brain,
        args.get('session_id', ''),
        args.get("last_assistant_message", "") or "",
        model=args.get("model", "") or "",
        tells=args.get("tells"),
    )
    session_id = ctx.session_id

    # Delivery — the Stop moment, the forcing leg: a `decision:block` reason
    # compels a read, where a passive mid-thread additionalContext competes
    # with recall and can be missed. channels/delivery.py owns the leg:
    # eligible sources (courier drain + Thalamus pull — asks excluded at
    # Stop, they deliver at boot), each failure-isolated and traced. Blocks
    # at most once per batch per source (the next Stop finds nothing and
    # allows it). Gated on the Stop event: hooks.json wires this handler to
    # Stop, and the guard keeps any future second wiring from blocking where
    # blocking would be wrong.
    _reason = ''
    if args.get("hook_event_name") == "Stop":
        try:
            from .channels.delivery import deliver, STOP
            from .trace_contract import arms_continuation
            from .clock import iso_now as _iso_now
            # post_response_common already advanced the counter, so the
            # delivery traces land on the NEXT chain — deliberately: a
            # delivery that blocks this stop OPENS the next turn as its
            # incoming side, and the continuation's response completes that
            # turn on the same chain (incoming K + assistant delta, the same
            # shape as an operator turn). The stamp arms that continuation's
            # classification; it stays inside this try because everything
            # after deliver() runs post-consume — a raise here must never
            # cost the block.
            _reason, _traced = deliver(brain, ctx, STOP)
            if _reason and arms_continuation(_traced):
                ctx.last_delivery_stop = ctx.stop_counter
                ctx.last_delivery_armed_at = _iso_now()
        except Exception as _dlv_err:
            brain._log_error('delivery_stop', _dlv_err,
                             'Stop delivery leg raised (session=%s)' % session_id)
    brain.save()
    if _reason:
        return {"output": "(stored)",
                "decision": "block", "reason": _reason}
    return {"output": "(stored)"}








def hook_pre_edit(brain, args, graph_changes):
    """PreToolUse(Edit|Write) — surface brain rules before file edits.
    (Self-message delivery lives on Stop alone — hook_post_response_track.)

    Returns JSON {"decision":"approve","reason":"..."}.
    """
    filename = args.get("filename", "")
    tool_name = args.get("tool_name") or ""

    if not filename:
        return {"json": {"decision": "approve"}}

    sid = args.get("session_id", "")
    ctx = brain.get_or_create_session(sid) if sid else None
    try:
        data = brain.pre_edit(file=filename, tool_name=tool_name, ctx=ctx)
    except Exception as e:
        brain._log_error('pre_edit', e, 'hook_pre_edit')
        return {"json": {"decision": "approve"}}

    suggestions = data.get("suggestions", [])
    procedures = data.get("procedures", [])
    context_files = data.get("context_files", [])
    debug_enabled = data.get("debug_enabled", False)

    # Change impact maps
    change_impacts = []
    try:
        change_impacts = brain.get_change_impact(filename)
    except Exception as e:
        brain._log_error('get_change_impact', e, 'hook_pre_edit')

    if not suggestions and not procedures and not context_files and not change_impacts:
        return {"json": {"decision": "approve"}}

    context = BrainVoice.format_suggestions(filename, suggestions, procedures,
                                            context_files, change_impacts)

    # Debug logging
    if debug_enabled:
        try:
            node_ids = [s.get("id", "") for s in suggestions if s.get("type") != "procedure"]
            node_ids += [p.get("id", "") for p in procedures]
            brain.log_debug(
                event_type="pre_edit",
                source="hook",
                file_target=filename,
                suggestions_served=len([s for s in suggestions if s.get("type") != "procedure"]),
                procedures_served=len(procedures),
                node_ids_served=json.dumps(node_ids),
                metadata=json.dumps({"tool": tool_name}),
            )
        except Exception as e:
            brain._log_error('debug_log_pre_edit', e, 'hook_pre_edit')

    brain.save()
    return {"json": {"decision": "approve", "reason": context}}


def hook_pre_bash_safety(brain, args, graph_changes):
    """PreToolUse(Bash) — safety context for destructive commands.

    Returns JSON {"decision":"approve","reason":"..."} — the brain informs the
    model about matching warnings; it never blocks the command (the host's own
    permission rules decide).
    """
    command = args.get("command", "")

    try:
        result = brain.safety_check(command)
    except Exception as e:
        brain._log_error('safety_check', e, 'hook_pre_bash_safety')
        return {"json": {
            "decision": "approve",
            "reason": "[BRAIN] \u26a0\ufe0f Safety check failed — proceed carefully. [/BRAIN]",
        }}

    warnings = result.get("warnings", [])

    if warnings:
        lines = ["[BRAIN] \u26a0\ufe0f WARNING: Destructive command detected. Relevant brain context:"]
        lines.append("")
        for w in warnings[:5]:
            title = cap_text_loud(w.get("title", ""),
                                    TITLE_WARN_LIMIT)
            content = cap_text_loud(w.get("content", ""), 200)
            lines.append("  [%s] %s" % (w.get("type", "?"), title))
            lines.append("    %s" % content)
            lines.append("")
        lines.append("Proceed carefully — verify this command is intentional.")
        lines.append("[/BRAIN]")
        return {"json": {"decision": "approve", "reason": "\n".join(lines)}}

    else:
        return {"json": {
            "decision": "approve",
            "reason": "[BRAIN] \u26a0\ufe0f Destructive command detected. No safety rules match, but proceed carefully. [/BRAIN]",
        }}


def hook_session_end(brain, args, graph_changes):
    """SessionEnd — save, and drop this session's cached SessionContext."""
    # Final save + drop the cached SessionContext for this session so
    # the in-memory cache doesn't accumulate ended-session entries.
    sid = args.get("session_id", "")
    if sid:
        try:
            brain.discard_session_context(sid)
        except Exception as e:
            brain._log_error('session_context_discard', e, 'hook_session_end')

    brain.save()
    return {"output": ""}


def hook_stop_failure_log(brain, args, graph_changes):
    """StopFailure — logs API failures to debug_log."""
    error_type = args.get("error", "unknown")
    error_details = args.get("error_details", "")
    try:
        brain.log_debug("stop_failure", "API error: %s — %s" % (error_type, str(error_details)[:200]))
    except Exception as _le:
        # Logging-path failure: route to _log_error as a fallback, then
        # stderr as the last-resort. Swallowing this silently would hide
        # API failures from production telemetry — the very thing this
        # hook exists to capture.
        try:
            brain._log_error('stop_failure_log', _le,
                             'log_debug raised on stop_failure capture; '
                             'original error_type=%s' % error_type)
        except Exception:
            import sys as _sys
            print('[hook_stop_failure_log] log_debug failed AND _log_error '
                  'failed: %s' % _le, file=_sys.stderr)
    return {"output": ""}


def hook_worktree_context(brain, args, graph_changes):
    """WorktreeCreate — stamps the per-session worktree/branch identity.

    MUST NOT write to stdout: Claude Code consumes a WorktreeCreate hook's STDOUT
    as the new worktree path. A previous version printed a `[BRAIN] GIT CONTEXT`
    block here, so CC tried to chdir into the trailing `[/BRAIN]` marker → the
    `ENOENT chdir '<repo>' -> '[/BRAIN]'` failure. Output stays empty; the context
    is recorded on the session object only.
    """
    session_id = args.get("session_id", "")
    worktree_name = args.get("name", "")
    cwd = args.get("cwd", "")

    # Branch + project from cwd in ONE git call (same combined probe as boot).
    # Project matters here too: a session that booted outside a repo and then
    # entered a worktree must not keep project='' — the provenance stamp would
    # strip every subsequent write from a demonstrably-repo session. The hook's
    # `name` arg stays authoritative for worktree (git's derived name ignored).
    branch, _, project = brain.detect_git_env(cwd)

    # Per-session identity — replaces the global current_worktree config, which was
    # last-writer-wins across parallel streams. session_id is backfilled from
    # CLAUDE_CODE_SESSION_ID by get_hook_input, so it's reliably present; the guard
    # stays defensive — never fall back to the singleton. Persist immediately under
    # write_lock (reentrant): set_env only mutates memory, and logs_conn writes
    # must be serialized (see get_or_create_session).
    if session_id:
        with brain.write_lock:
            ctx = brain.get_or_create_session(session_id)
            ctx.set_env(cwd=cwd, branch=branch, worktree=worktree_name,
                        project=project)
            ctx.save(brain._session_state)

    graph_changes.append("WORKTREE: created %s (branch: %s)" % (worktree_name or '(unnamed)', branch))
    brain.save()
    return {"output": ""}


def hook_worktree_cleanup(brain, args, graph_changes):
    """WorktreeRemove — clears the per-session worktree identity."""
    session_id = args.get("session_id", "")
    old_worktree = ""
    if session_id:
        with brain.write_lock:
            ctx = brain.get_or_create_session(session_id)
            old_worktree = ctx.worktree
            ctx.set_env(worktree="")
            ctx.save(brain._session_state)
    if old_worktree:
        graph_changes.append("WORKTREE: removed %s" % old_worktree)
    return {"output": ""}
