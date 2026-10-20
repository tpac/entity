"""Centralized hook logic — all hook brain interactions live here.

Previously each hook .py file had a _run_daemon() and _run_direct() path,
duplicating logic and diverging over time. Now hooks are thin clients that
send a single command to the daemon, and this module contains all the logic.

Each function signature: hook_*(brain, args, graph_changes) -> dict
  - brain: Brain instance (already loaded by daemon)
  - args: dict from the hook client
  - graph_changes: list[str] — in-memory mutation log, drained by hook_recall

Returns: {"output": str} for text hooks, or {"output": str, "json": dict}
for hooks that need structured JSON output (recall, pre-edit).
"""

import time

from servers.brain_traces import _s0_trace
from servers.brain_voice import BrainVoice
# Hoisted — if this import ever breaks, the daemon fails at boot instead of
# silently degrading recall 16s in.
from servers.embedder import embed_query
from servers.host_contract import envelope_policy
from servers.trace_contract import (ENVELOPE_EXTRACT_PREFIX, HOST_TURN_FIELD,
                                    READER_STAMP_FIELDS, REGISTER_ONLY_FIELD,
                                    SPEAKER_REF_TYPES, session_short)


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


def _prompt_delivery(brain, ctx):
    """The Thalamus block for this session's first operator prompt, '' after
    that or when nothing is due. The flag is set FIRST: a raise inside the
    leg must not re-fire the moment on every later prompt."""
    if ctx.prompt_delivered:
        return ''
    ctx.prompt_delivered = True
    try:
        from .channels.delivery import deliver, PROMPT
        return deliver(brain, ctx, PROMPT)[0]
    except Exception as e:
        brain._log_error('prompt_delivery', e,
                         'first-prompt delivery raised (session=%s)' % ctx.session_id)
        return ''


def _reader_stamp(args):
    """The contract identity a hook sent beside what its reader decided
    (trace_contract.READER_STAMP_FIELDS); {} from a client that predates it."""
    return {k: args[k] for k in READER_STAMP_FIELDS if args.get(k) is not None}


def _pieces(args):
    """The prompt hook's reading of the prompt (host_contract.read_prompt):
    the (speaker, text, tag) pieces, or None from a client that predates the
    reader or sent pieces the write door cannot place — those record the
    whole prompt as one operator row, as before."""
    pieces = args.get("pieces")
    if pieces is None:
        return None
    if pieces and all(isinstance(p, (list, tuple)) and len(p) == 3
                      and p[0] in SPEAKER_REF_TYPES
                      and isinstance(p[1], str) and isinstance(p[2], str)
                      for p in pieces):
        return pieces
    return None


def _operator_spoke(args):
    """True when the operator's own text was recalled on this prompt — the
    first-prompt moment's condition. A harness notice or a subagent's report
    is not the operator arriving."""
    if args.get("register_only"):
        return False
    pieces = _pieces(args)
    return pieces is None or pieces[0][0] == 'operator'


def _write_incoming(brain, ctx, args, host):
    """Write the prompt's incoming S0 rows — one per piece the prompt hook
    read (host_contract.read_prompt), each as its speaker's ref_type, the
    piece that opens the turn first so the pieces cut from the prompt ride in
    that turn. Each row carries the contract identity that read it. The first
    row is the turn's: it holds the recall chain and, when nothing is
    recalled, the register-only mark. A client that predates the reader (or
    sent pieces this door cannot place, logged) records the whole prompt as
    one operator row, as before. The reader declares and this door checks:
    a piece whose tag the contract does not declare, and an extract envelope
    that read nothing, are each logged as a warning — against `host`, the
    host this prompt's own tells resolved, the table the hook read. Every row
    carries the host's name for the turn when the hook read one
    (host_contract.read_turn) — the turn rule joins a message absorbed
    mid-turn to its turn by it. Returns
    (first row's trace id, the text recall reads)."""
    prompt = args.get("prompt", "")
    pieces = _pieces(args)
    if pieces is None and args.get("pieces") is not None:
        brain._log_error('hook_recall_pieces', None,
                         'prompt hook sent pieces the write door cannot place '
                         '— recorded the whole prompt as one operator row '
                         '(session=%s)' % ctx.session_id, ctx)
    stamp = {}
    if pieces is None:
        pieces = [['operator', prompt, '']]
    else:
        stamp = _reader_stamp(args)
    turn = args.get(HOST_TURN_FIELD)
    if isinstance(turn, str) and turn:
        stamp[HOST_TURN_FIELD] = turn
    elif turn not in (None, ''):
        brain._log_warning(
            'host_turn_malformed', 'the prompt hook sent a host turn that is '
            'not a string — this prompt\'s rows carry none, so a message '
            'absorbed mid-turn opens a turn of its own',
            'session=%s value=%.80r' % (ctx.session_id, turn), ctx)
    first = None
    for speaker, text, tag in pieces:
        meta = dict(stamp)
        if first is None:
            meta['recall_chain'] = ctx.s1r_chain()
            if args.get("register_only"):
                meta[REGISTER_ONLY_FIELD] = True   # no recall/surface below
        trace_id = _s0_trace(
            brain, ctx, event_type='K', ref_type=SPEAKER_REF_TYPES[speaker],
            summary=text[:200], metadata=meta, content=text or None, ref_id=tag)
        first = first or trace_id
        if tag and envelope_policy(host, tag) is None:
            brain._log_warning(
                'unknown_envelope', 'a prompt arrived wrapped in %s, which no '
                'envelope declares — recorded as the harness\'s, without recall; '
                'declare it in host_contract' % tag,
                'session=%s host=%s' % (ctx.session_id, host or 'unknown'), ctx)
    policy = (envelope_policy(host, pieces[0][2]) or ('', ''))[1]
    if policy.startswith(ENVELOPE_EXTRACT_PREFIX) and not args.get("recall_text"):
        # An error, not a warning: a declared envelope the parser could not
        # read is the host's format moving under the contract — surfaced at boot.
        brain._log_error(
            'hook_recall_extract', None, 'a %s envelope matched but its extractor '
            'read nothing — recorded the raw envelope without recall (session=%s)'
            % (pieces[0][2], ctx.session_id), ctx)
    return first, args.get("recall_text") or prompt


def hook_recall(brain, args, graph_changes):
    """UserPromptSubmit: the recall turn (_recall_turn), then — once per
    session, on its first operator prompt — the Thalamus PROMPT delivery
    (ruling id:15e46fa9) composed ahead of whatever the turn surfaced. ONE
    composition site, so no early return inside the turn can drop a
    session's due items. A prompt the operator did not speak in (a short
    answer, a harness notice, a subagent's report) never reaches the moment:
    boot cannot tell an operator from a scheduler, the first prompt can. The
    Thalamus rides this passive moment because it survives a miss; the
    courier does not — self-message delivery stays on Stop alone (a block the
    host cannot skip; on_prompt additionalContext would lose the consume-once
    race and once overflowed the inject spill cap —
    docs/SELF-CHANNEL-DESIGN.md)."""
    result = _recall_turn(brain, args, graph_changes)
    if not _operator_spoke(args):
        return result
    block = _prompt_delivery(brain, brain.get_or_create_session(args.get('session_id', '')))
    if block:
        prior = (result.get("json") or {}).get("additionalContext", "")
        result["json"] = {"additionalContext": block + ('\n\n' + prior if prior else '')}
    return result


def _recall_turn(brain, args, graph_changes):
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
    ctx = brain.get_or_create_session(args.get('session_id', ''))
    session_id = ctx.session_id
    _current_stop = str(ctx.stop_counter)
    # Mark that a real UserPromptSubmit ran recall for THIS stop. The Stop hook
    # reads this to classify the turn: recall ran → conversational. A /watch
    # wakeup skips recall client-side (pre_response_recall.py: a repeating
    # command or a shell escape has no pieces), so its stop never reaches
    # here → it reads as a heartbeat. The client has
    # already filtered, so reaching hook_recall means a real prompt.
    # See trace_contract S0 TURN CLASSIFICATION.
    ctx.last_recall_stop = ctx.stop_counter
    host = _set_hook_env(ctx, args.get('model', ''), args.get('tells'))

    # Write the incoming S0 rows NOW, at prompt-arrival — not at Stop. This
    # is what lets presence/peek surface a session's current prompt mid-turn
    # (rendezvous identity), instead of only its last completed turn. The
    # assistant half is still written at Stop (post_response_common); both use
    # ctx.s0_chain() with the same stop_counter, so they stay paired. Reaching
    # hook_recall means a real prompt (client filters watch-wakes), so this is
    # exactly the conversational set — heartbeat turns never get here.
    _user_msg_trace_id, recall_text = _write_incoming(brain, ctx, args, host)

    # ── Register-only fast path ──
    # A prompt with nothing to recall on — a short answer ("yes", "ok", "no"),
    # a harness notice — that the client routes here with register_only=True.
    # The turn IS conversational and is now fully registered: its S0 rows +
    # last_recall_stop are written ABOVE, and the
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
    enriched = recall_text[:_PL['user_message_query']]
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
        # shrink (a98143f review, finding 1). The cue is this prompt's first
        # row: the LAF moment stack's window ends just before it, as the
        # Surface's window below does.
        result = brain.recall(query=enriched,
                              limit=_CF['max_candidates']
                              + _CF.get('seen_dedup_headroom', 0),
                              ctx=ctx, source='hook',
                              cue_trace_id=_user_msg_trace_id)
    except Exception as e:
        brain._log_error('recall_first_attempt', e, 'hook_recall')
        result = brain.recall(query=enriched,
                              limit=_CF['max_candidates']
                              + _CF.get('seen_dedup_headroom', 0),
                              ctx=ctx, source='hook',
                              cue_trace_id=_user_msg_trace_id)

    results = result.get("results", [])
    pt.mark('recall')

    recall_ref = '%s-%s' % (ctx.session_short, _current_stop)

    # Enrich candidates for the surface call: batch rich-node pull, then
    # query-aware edge selection.
    candidates_data = []
    recent_turns = []
    _query_vec = None
    _prior_vecs = []
    try:
        capped = results[:_CF['max_candidates']]

        # Previous turns — ONE read serves the surface conversation window,
        # the seen-dedup set and the prior-query embedding blend below,
        # keeping their notion of "previous" aligned by construction: whole
        # turns (brain.conversation_turns, the reader the encoder shares),
        # windowed by surface_contract.window_turns. The read ends before this
        # prompt's first row (before_trace_id) — written at prompt-arrival
        # above, while the current message reaches build_surface_prompt
        # separately as `user_message` — so the prompt's own pieces never
        # form a turn, and a message absorbed mid-turn still sees the turn it
        # joined.
        from .scales.s1.surface_contract import SURFACE as _SURFACE, window_turns
        try:
            recent_turns = window_turns(brain.conversation_turns(
                session_id, _SURFACE['window_read_turns'], with_surfaced=True,
                before_trace_id=_user_msg_trace_id))
        except Exception as _e:
            brain._log_error('surface_recent_turns', _e, 'reading the previous turns from traces')
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
        # set reuses the turns read above (no extra query). The same set
        # gates Haiku's tool results and the final selection in run_surface:
        # a node the agent already has in context this window does not
        # render again through any door (the agent can get_nodes it). Falls
        # back to the plain cap when the turns read failed (recent_turns
        # empty → seen empty).
        # Seen is read over every incoming row of the window's turns, not
        # only the operator's: a subagent's report runs recall too, and what
        # it surfaced is in context just the same. A node re-enters once its
        # turn slides out of what Haiku sees.
        from .scales.s1.surface import seen_node_ids
        _seen = seen_node_ids(recent_turns)
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
            # The operator's last two messages for the multi-turn blend —
            # from the single turns read above, wherever in the window they
            # sit (turns others opened carry none).
            try:
                from .trace_contract import row_speaker, row_text, turn_incoming
                _prior = [r for t in recent_turns for r in turn_incoming(t)
                          if row_speaker(r) == 'operator'][-2:]
                for _t in _prior:
                    _text = row_text(_t)[:500]
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
                brain, ctx, candidates_data, recall_text,
                recent_turns=recent_turns,
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
                         'S1 Surface failed in daemon (query=%s)' % recall_text[:100])

    pt.log(brain, 'hook_recall', n_results=len(results))

    if additional_context:
        return {"json": {"additionalContext": additional_context}, "session_id": session_id}
    return {"json": {"decision": "approve"}, "session_id": session_id}






def _set_hook_env(ctx, model, tells):
    """Refresh session identity only from a resolved prompt/Stop observation.

    An old client's pre-resolved host is no longer an input. Unknown and
    ambiguous observations leave the session's last known host alone.
    Returns the host this observation resolved — '' when unknown or
    ambiguous, as the prompt hook's own resolution reads it.
    """
    from .host_contract import resolve_host
    present = {t: True for t in tells if isinstance(t, str)} \
        if isinstance(tells, (list, tuple)) else {}
    host, status, _ = resolve_host(present)
    ctx.set_env(model=model)
    if status in ('strong', 'family'):
        ctx.set_env(host=host)
    return host


def post_response_common(brain, session_id, assistant_response, *,
                         model='', tells=None, stamp=None):
    """Shared post-response path: S0 traces, heartbeat, stop counter
    increment. Used by prod Stop hook and by the eval harness —
    same code, same ordering, one source of truth.

    `assistant_response`: the reply the Stop hook's contract reader decided
    (host_contract.read_reply); `stamp` the reader's identity
    (_reader_stamp), carried on the assistant_message row.

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
            # presence/peek can surface a session's current prompt mid-turn.
            # Only the assistant half is written here, at Stop; stop_counter
            # is unchanged between hook_recall and this Stop (incremented
            # below), so the pair shares the chain. For a delivery REACTION:
            # the incoming K was written by deliver() at the PREVIOUS stop's
            # hook, after that stop's increment — which put it on THIS chain,
            # so the pair shares the chain by the same invariant from the
            # other side.
            _s0_trace(
                brain, ctx, event_type='delta', ref_type='assistant_message',
                summary=assistant_response[:200], metadata=stamp or None,
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
    # dedup fingerprint is per session: one row per session per dedup window,
    # and a second session's gap is never masked by the first's). Logged, never
    # blocking: the turn is still recorded.
    if args.get("hook_event_name") == "Stop" and not args.get("model"):
        brain._log_error(
            's0_model_unset',
            ValueError('no model on Stop for session %s'
                       % session_short(args.get('session_id', ''))),
            'the turn\'s S0 rows carry no model stamp')
    ctx = post_response_common(
        brain,
        args.get('session_id', ''),
        args.get("last_assistant_message", "") or "",
        model=args.get("model", "") or "",
        tells=args.get("tells"),
        stamp=_reader_stamp(args),
    )
    session_id = ctx.session_id

    # Delivery — the Stop moment, the forcing leg: a `decision:block` reason
    # compels a read, where a passive mid-thread additionalContext competes
    # with recall and can be missed. channels/delivery.py owns the leg:
    # eligible sources (courier drain + Thalamus pull — broadcast asks
    # excluded at Stop, they deliver at the first prompt; directed asks
    # deliver here), each failure-isolated and traced. Blocks
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
    """PreToolUse(Edit|Write) — the tool moment: memories about a file at its
    agent's first edit of it, the attention moment beside the prompt.

    One recall on the cue "editing <path>": nothing promoted, at most
    TOOL_MOMENT_MAX_NODES, never a node the agent has already been shown.
    An agent recalls a cue once per session — the moment's own `tool_shown`
    rows (brain.tool_moment_rows) say which cues ran and what they showed;
    every run writes one, shown or not. The session's agent skips what the
    prompt surface showed in its window and what the moment itself showed this
    session. The moment's shows are not Haiku's picks: the prompt surface may
    still pick them. A subagent (`agent_id`) skips only its own shows — it never
    saw the session agent's context. A path TOOL_MOMENT_SKIP matches recalls
    nothing and writes no row.

    Returns {"json": {"additionalContext": text}}, or {"json": {}} when there
    is nothing to say.
    """
    from .scales.s1.surface_contract import (
        CANDIDATE_POOL, SURFACE, TOOL_MOMENT_MAX_NODES, TOOL_MOMENT_SKIP,
        render_tool_inject, window_turns)
    path = args.get("path", "")
    sid = args.get("session_id", "")
    agent_id = args.get("agent_id") or ""
    if not path or not sid or TOOL_MOMENT_SKIP.search(path):
        return {"json": {}}
    from .scales.s1.surface import seen_node_ids
    cue = "editing %s" % path
    try:
        ctx = brain.get_or_create_session(sid)
        rows = brain.tool_moment_rows(sid, agent_id)
        if any(r['cue'] == cue for r in rows):
            return {"json": {}}
        seen = {str(i)[:8] for r in rows for i in r['ids']}
        if not agent_id:
            seen |= seen_node_ids(window_turns(brain.conversation_turns(
                sid, SURFACE['window_read_turns'], with_surfaced=True)))
        # Over-fetch by the surface's seen-dedup headroom, so dropping
        # already-shown nodes leaves replacements to fill the cap.
        result = brain.recall(
            query=cue, ctx=ctx, source='tool_moment',
            limit=TOOL_MOMENT_MAX_NODES + CANDIDATE_POOL['seen_dedup_headroom'])
        ids = [r['id'] for r in result.get('results', [])
               if r.get('id') and r['id'][:8] not in seen
               ][:TOOL_MOMENT_MAX_NODES]
        fetched = (brain.get_node(ids) or {}) if ids else {}
        # The pull resolves an archived id to its survivor, which can be a
        # node the agent was already shown, or the same node twice.
        nodes = list({n['id']: n for n in (fetched.get(i) for i in ids)
                      if n and n['id'][:8] not in seen}.values())
        text = render_tool_inject(nodes, path, scope=brain.session_scope(sid))
    except Exception as e:
        brain._log_error('tool_moment', e, 'hook_pre_edit cue=%s' % cue[:80])
        return {"json": {}}
    brain.write_tool_moment(
        chain_id=ctx.s1r_chain(), session_id=sid, cue=cue, agent_id=agent_id,
        shown=[(n['id'], n.get('title', '')) for n in nodes])
    return {"json": {"additionalContext": text}} if text else {"json": {}}


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
    # last-writer-wins across parallel sessions. session_id is backfilled from
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
