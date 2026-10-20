"""S1 Turn Encoder — LLM-powered brain encoding via Sonnet API.

Scale: S1 (Turn integration — fires once ENCODE_EVERY turns await encoding, or on the idle tail)
Chain: s1e (encode)
Interaction: 's1e' in interactions table (learnable boundary)

Triggered by: the daemon's Scribe poll (brain.scribe_due → S1Scribe.run in scribe.py)
Reads: traces (conversation turns), brain nodes (catalog), interactions table
Writes: nodes/edges via dispatch, traces (O/K), journal notes + the session arc
"""

import os
import time

from servers.scales.dispatch import load_env
from servers.scales.runner import run_llm_loop
from servers.trace_contract import (build_delta_metadata, interaction_fields,
                                    session_short)


def scribe_journal_kwargs(session_id):
    """The Scribe's binding shape beyond scale/source: session-walled residue
    plus the `## Arc` opt-in. One spelling, used by S1Scribe.journal (the
    production binding, which adds the source) and by _journal below."""
    return {'session_id': session_id, 'arc': True}


def _journal(brain, session_id=''):
    """A SOURCE-LESS Scribe-shaped binding for the standalone callers (evals,
    tests) that carry no unit — tell/ask lines stay plain notes there. The
    production binding is `S1Scribe.journal`, handed in as `journal=`.
    Decorate-only sites may pass brain=None."""
    from servers.scales.journal import JournalBinding
    return JournalBinding(brain, scale='s1', **scribe_journal_kwargs(session_id))


def run_encoding(brain, dispatch_fn, counter, session_id, log_fn=None,
                 journal=None):
    """S1 turn encoder: gather → prompt → trace O/K → LLM loop → post-process.

    Args:
        brain: Brain instance (reads, plus the payload and journal-note writes;
            graph writes go through dispatch_fn)
        dispatch_fn: function(cmd, args) for writes (the in-process encoder
            dispatch in production)
        counter: Stop counter value
        session_id: Session ID (required)
        log_fn: Optional logging function

    Returns:
        dict with encoding results summary.
    """
    def _log(msg):
        print("[s1e] %s" % msg, flush=True)
        if log_fn:
            log_fn("S1 encode: %s" % msg)

    from servers.scales.s1.encode_contract import (
        ENCODING_AGENT, SCRIBE_RUN_DEADLINE_SECONDS)

    t0 = time.time()
    profile = []

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    # ONE binding per run — decoration, continuity and harvest all read it.
    # The production caller (S1Scribe) hands in its sourced binding; the
    # standalone callers get the source-less Scribe shape.
    journal = journal or _journal(brain, session_id)

    load_env()
    _step("env_loaded")

    from ..runner import make_client
    try:
        client = make_client()
    except Exception as e:
        print('[s1e] ERROR: Cannot create Anthropic client: %s' % e, flush=True)
        return {"error": str(e)}
    _step("api_client")

    if not session_id:
        raise ValueError("session_id is required")

    # 1. The conversation window (S1-specific: reads from traces)
    turns = _window(brain, session_id)
    _step("turns(%d)" % len(turns))
    if not turns:
        _log("no turns, skipping")
        return {"skipped": True, "reason": "no turns"}

    # View policy and the associated-stubs flag (encoder_view.py): resolved
    # ONCE per run and threaded down (env read is non-atomic across call
    # sites) so the catalog aging and the timeline render agree, and the
    # untaught-tag check below and the stub render agree, even if a flag is
    # toggled mid-run.
    from servers.scales.s1.encoder_view import (
        associated_stubs_enabled, view_policy_enabled)
    view = view_policy_enabled()
    stubs = associated_stubs_enabled()
    # Conversation time, resolved ONCE per run (like the view flag above) —
    # the catalog's relative headers and the timeline's now/age stamps must
    # agree on the instant.
    view_now = _conversation_now_safe(brain, session_id, turns) if view else None

    # 2. Build prompt. Each learnable boundary resolves ONCE per run —
    # template, config and stamp from one read — so the delta records the K
    # this run actually used. `s1e` is the system prompt; its config carries
    # `effort` (the API's output_config.effort) and `model`, so either ships
    # as a version, not a code edit. The gist is its own K (`s1e_gist`,
    # rendered into the body); its stamp rides the same delta so a run's
    # provenance names both texts. Resolved config is total — the defaults
    # live in encode_contract, nowhere else.
    enc = brain.get_interaction_effective('s1e')
    gist = brain.get_interaction_effective('s1e_gist')
    enc_stamp, gist_stamp = enc['stamp'], gist['stamp']
    enc_effort = enc['config']['effort']
    enc_model = enc['config']['model']
    system_prompt = _build_system_prompt(enc['template'], journal=journal)

    # The stubs flag can run ahead of the prompt: the encoder reads the
    # associated tag correctly only once the active s1e prompt teaches it — a
    # comment can't enforce that ordering, this check can. Advisory: logs,
    # never blocks (the A/B arm decides).
    from servers.scales.s1.encode_contract import ASSOCIATED_TAG
    if stubs and ASSOCIATED_TAG not in enc['template']:
        brain._log_error(
            's1e_associated_stubs_untaught',
            'BRAIN_S1E_ASSOCIATED_STUBS is on but the active s1e '
            'prompt never mentions %s' % ASSOCIATED_TAG,
            'stubs render untaught — activate the prompt version '
            'that teaches the tag')

    # 2a. Catalog first, then the body — the two halves profile separately
    # and the body reuses the catalog's trace streams.
    catalog_text, catalog_ids, streams = _build_catalog(
        brain, turns, session_id, view_policy=view, now=view_now,
        associated_stubs=stubs)
    _step("catalog(%d ids)" % len(catalog_ids))

    # 2b. Body assembly: the XML lived-sequence timeline.
    user_preamble, user_content, _cat_text2, _cat_ids2 = _build_user_content(
        brain, turns, counter, session_id, gist=gist,
        precomputed=(catalog_text, catalog_ids, streams), view_policy=view,
        view_now=view_now, journal=journal)
    _step("prompt(preamble=%d chars, body=%d chars)" % (
        len(user_preamble), len(user_content)))

    # 3. Get tools (S1-specific subset)
    tools = _get_tool_schemas()
    _step("tools(%d)" % len(tools))

    # 4. Record the assembled prompt via the payload recorder (dashboard +
    # post-hoc eval read it back by pointer/chain; per-arm eval brains land
    # it inside their own db_dir by construction). The O trace below carries
    # the returned db_dir-relative pointer as its ref_id.
    from servers.session_context import SessionContext
    enc_chain = SessionContext(session_id, stop_counter=counter).s1e_chain()
    prompt_pointer = brain.record_payload(
        enc_chain, 'prompt', user_preamble + "\n\n" + user_content)

    # 5. Write S1 traces: O (observation) and K (knowledge)
    _write_pre_traces(brain, dispatch_fn, turns, user_content, enc_chain,
                      session_id, enc_stamp, catalog_ids=catalog_ids,
                      prompt_pointer=prompt_pointer)

    # 6. Run generic LLM loop (shared with S2+)
    _log("calling %s with %d tools, %d chars context, effort=%s..." % (
        enc_model, len(tools), len(user_content), enc_effort or 'model default'))
    _log("PROFILE so far: %s" % " → ".join("%s:%dms" % (n, t) for n, t in profile))

    try:
        result = run_llm_loop(
            client=client,
            model=enc_model,
            effort=enc_effort,
            max_tokens=ENCODING_AGENT['max_tokens'],
            max_rounds=ENCODING_AGENT.get('max_rounds', 5),
            system_prompt=system_prompt,
            user_content=user_content,
            user_preamble=user_preamble,
            tools=tools,
            dispatch_fn=dispatch_fn,
            log_fn=_log,
            record_round_fn=brain.round_recorder(enc_chain),
            deadline_seconds=SCRIBE_RUN_DEADLINE_SECONDS)
    except Exception as e:
        # The LLM loop itself died — no writes happened this run. Loud by
        # default: errors table + a `encoding_run_failed` delta so the run is
        # visible (dashboard, forensics) WITHOUT claiming turn coverage —
        # trace_links joins on `encoding_run` only, so a failed run never
        # marks turns encoded. The print stays for daemon.log grep-ability.
        _step("FAILED")
        profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
        print('[s1e] ERROR: Sonnet API call failed: %s PROFILE: %s' % (e, profile_str), flush=True)
        brain._log_error('s1e_run_failed', e,
                         'session=%s stop=%d — LLM loop failed; writes from '
                         'completed rounds (if any) are kept, turns stay '
                         'unencoded and will retry' % (session_short(session_id), counter))
        # Salvage the actions that ran before the failure (RunLoopError
        # carries them). Bounding + record shape live in the traces layer
        # (build_failed_run_metadata) — this consumer just hands them over.
        # The full conversation at failure time (RunLoopError.msgs, tool
        # results already capped) is recorded as the failed_run payload —
        # the 2AM story: the next 6M-char-class incident is diagnosable
        # from the trace row + one file, in normal mode. The traces layer
        # owns the payload shape and caps; round-0 failures arrive unwrapped
        # (no msgs) and record nothing — the prompt kind is that half.
        from servers.trace_contract import build_failed_run_metadata
        failed_ptr = brain.record_failed_run(enc_chain, e)
        with brain.loud('s1e_failed_trace_write', 'recording encoding_run_failed delta'):
            dispatch_fn('trace_append', {
                'chain_id': enc_chain, 'scale': 's1', 'event_type': 'delta',
                'ref_type': 'encoding_run_failed',
                'summary': 'FAILED: %s' % str(e)[:200],
                'metadata': build_failed_run_metadata(
                    error=e, stop_counter=counter,
                    inputs_processed=len(turns), interaction=enc_stamp,
                    partial_actions=getattr(e, 'partial_actions', None),
                    payload_pointer=failed_ptr),
                'session_id': session_id,
            })
        # Reflective residue (docs/TRACE-MODES-DESIGN.md §Failed-run residue):
        # a dead run must leave a journal note so the retry's continuity
        # window shows the failure instead of amnesia. Written through the
        # journal row door, like every other note.
        with brain.loud('s1e_failed_journal_note',
                        'recording failure journal note'):
            brain.write_journal_note_rows([dict(
                subject='encoding-run-failure', tag='failure',
                note='stop %d run FAILED: %s — turns stay unencoded and '
                     'retry; writes from completed rounds (if any) were '
                     'kept' % (counter, str(e)[:200]))],
                chain_id=enc_chain, scale='s1', session_id=session_id)
        return {"error": str(e), "profile": profile}

    _step("done")
    profile_str = " → ".join("%s:%dms" % (n, t) for n, t in profile)
    _log("done. %d rounds, %d actions. PROFILE: %s" % (
        result['rounds'], result['actions'], profile_str))

    # Log truncation errors to brain errors table
    for trunc in result.get('truncations', []):
        brain._log_error(
            's1e_truncation',
            'max_tokens truncation: round %d used %s/%s output tokens' % (
                trunc['round'], trunc['output_tokens'], trunc['max_tokens']),
            'S1E tool call likely corrupted, encoding data may be lost')
    if result.get('undispatched_tool_calls'):
        brain._log_error(
            'undispatched_tool_calls',
            'max_rounds exhausted — last reply\'s tool call(s) not run: %s'
            % ', '.join(result['undispatched_tool_calls']),
            'unit=s1e session=%s stop=%d' % (session_short(session_id), counter))
    for failure in result.get('annotation_errors') or ():
        brain._log_error(
            's1e_reply_annotation_failed',
            RuntimeError('%s reply annotation failed: %s'
                         % (failure['tool'], failure['error'])),
            'the reply reached the encoder without its notes; '
            'session=%s stop=%d' % (session_short(session_id), counter))

    # Oversized-tool-result scan — the runner truncates and marks the action
    # (it has no brain reference); the loud errors-table entry lands here.
    for a in (result.get('action_details', []) + result.get('read_calls', [])):
        if a.get('result_truncated'):
            brain._log_error(
                's1e_oversized_tool_result',
                RuntimeError('%s returned %d chars — truncated to cap' % (
                    a.get('tool'), a.get('result_chars', -1))),
                'session=%s stop=%d; head: %s' % (
                    session_short(session_id), counter,
                    (a.get('result_head') or '')[:200]))

    # 7. Post-process (S1-specific: the residue). Guarded as a whole: the
    # loop's writes already landed — a post-process failure (e.g. `database
    # is locked`) must not swallow step 8's delta trace, which coverage
    # attribution and the dashboard both read.
    final_text = result.get('final_text', '')
    with brain.loud('s1e_postprocess',
                    'journal/arc post-process failed — writes intact, '
                    'continuing to delta trace; session=%s stop=%d'
                    % (session_short(session_id), counter)):
        # The `## Review` note contract, SESSION-BOUND: write_journal_notes
        # extracts the fence + writes one journal_note trace per note, all
        # sharing enc_chain (this run); session_id walls continuity to this
        # conversation. The arc is a SEPARATE object with its own journal
        # component: the encoder emits a `## Arc` fenced one-liner
        # (render_journal_arc_block) and write_session_arc accumulates it into
        # session_context_{sid}. Both write doors are failure-isolated
        # internally.
        journal.harvest(
            final_text, enc_chain,
            arc_limit=ENCODING_AGENT.get('session_context_limit', 800))

    # 8. Delta trace — unified shape across S1E + S2 encoders. Own guard:
    # losing this trace makes the run invisible (dashboard "0 actions") and
    # pushes coverage attribution onto the next run — so a failure here is
    # loud, and post-process failures above can't reach it.
    # Outcomes: count write actions by tool (remember / revise / connect / …).
    with brain.loud('s1e_delta_write',
                    'session=%s stop=%d: encoding_run delta lost — run shows '
                    '0 actions on dashboard, coverage attribution falls to '
                    'next run' % (session_short(session_id), counter)):
        action_details = result.get('action_details', [])
        outcomes = {}
        for a in action_details:
            tool = a.get('tool', 'unknown')
            outcomes[tool] = outcomes.get(tool, 0) + 1

        # enc_chain computed at step 4 — reused here for the delta.
        # Which K produced this Δ — the stamp (fingerprint + source + version,
        # resolved with the template at run start) rides the metadata; the
        # install-local rowid rides the trace row for display. Lets higher
        # scales A/B prompt versions from production traces (the whole point
        # of interactions-as-K-store).
        enc_metadata = build_delta_metadata(
            actions=result.get('actions', 0),
            write_actions=result.get('write_actions', 0),
            rounds=result.get('rounds', 0),
            inputs_processed=len(turns),
            outcomes=outcomes,
            action_details=action_details,
            read_calls=result.get('read_calls', []),
            final_text=final_text,
            elapsed_ms=result.get('elapsed_ms', 0),
            input_tokens=result.get('input_tokens', 0),
            output_tokens=result.get('output_tokens', 0),
            cache_read_tokens=result.get('cache_read_tokens', 0),
            cache_creation_tokens=result.get('cache_creation_tokens', 0),
            truncated=len(result.get('truncations', []) or []),
            model=result.get('model', ''),
            interaction=enc_stamp,
            **interaction_fields(gist_stamp, 'gist_'),
            stop_counter=counter,
        )
        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'delta',
            'ref_type': 'encoding_run',
            'interaction_id': enc_stamp['id'],
            'summary': '%d actions (%d writes) in %d rounds, %dms, %d→%d tok' % (
                result.get('actions', 0),
                result.get('write_actions', 0),
                result.get('rounds', 0),
                result.get('elapsed_ms', 0),
                result.get('input_tokens', 0),
                result.get('output_tokens', 0)),
            'metadata': enc_metadata,
            'session_id': session_id,
        })
        # Loud-at-the-write-boundary: same telemetry guard IntegrationUnit.trace
        # applies to the S2 encoders — but encoding_run is S1E's delta, written
        # via dispatch (not IntegrationUnit), so the check lives here too. brain
        # is in scope; logs to the errors view. S1E already threads telemetry, so
        # this is a regression guard (it stays silent on a healthy run).
        from servers.trace_contract import check_delta_telemetry
        _tel_warn = check_delta_telemetry('encoding_run', enc_metadata)
        if _tel_warn:
            brain._log_error('s1e_telemetry_gap', ValueError(_tel_warn),
                             'delta trace missing LLM telemetry')

    result['profile'] = profile
    return result


# ── S1-Specific Helpers ──


def _window(brain, session_id, older_than=None):
    """The encoder's conversation window: every turn new since this session's
    last successful encode started (brain.last_encode_anchor, the instant the
    Scribe cadence counts from), plus up to ENCODING_AGENT
    ['window_context_turns'] already-encoded turns before them, so the first
    new turn's referent is in view. A turn is new when any of its incoming
    rows arrived after the anchor: a message typed into a turn that opened
    before it is still the operator's new words. One read
    (brain.conversation_turns) serves the timeline, the catalog cutoff, the
    associated-stub seeds and the catalog's surfaced ids. [] when nothing is
    new — the run has nothing to encode.

    At most ENCODING_AGENT['window_max_turns'] turns. A backlog past it keeps
    the newest and logs s1e_window_capped with the count left out — never
    silent. `older_than`: the window as of that instant (an eval replaying a
    past run). Raises on a read failure: the timeline IS the input, so the run
    fails and the next due cycle retries it."""
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    from servers.trace_contract import turn_incoming, turn_opened_at
    cap = ENCODING_AGENT['window_max_turns']
    since = brain.last_encode_anchor(session_id, older_than=older_than)

    def new(turn):
        stamps = [r.get('created_at') or '' for r in turn_incoming(turn)]
        return max(stamps or [turn_opened_at(turn)]) > since

    turns = brain.conversation_turns(session_id, cap, with_actions=True,
                                     with_judge_output=True, older_than=older_than)
    first = next((i for i, t in enumerate(turns) if new(t)), len(turns))
    if first == len(turns):
        return []
    if first == 0:
        # Every turn read is new, so the backlog may run past the read.
        left_out = sum(1 for t in brain.conversation_turns(
            session_id, cap, older_than=turn_opened_at(turns[0])) if new(t))
        if left_out:
            brain._log_error(
                's1e_window_capped',
                ValueError('session %s has more unencoded turns than the '
                           'window holds' % session_short(session_id)),
                '%s%d unencoded turns left out (window_max_turns=%d); the '
                'newest are encoded' % ('at least ' if left_out >= cap else '',
                                        left_out, cap))
    return turns[max(0, first - ENCODING_AGENT['window_context_turns']):]


def _build_system_prompt(prompt_instructions, journal=None):
    """Build encoding agent system prompt: the resolved s1e template plus
    the contract field summary."""
    if not prompt_instructions:
        raise ValueError('s1e system prompt needs its template, got %r'
                         % (prompt_instructions,))
    prompt = prompt_instructions
    try:
        from servers.contract import generate_field_summary
        prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception as e:
        print('[s1e] WARNING: could not load field summary: %s' % e, flush=True)

    # Residue contract: the WRITE-side instructions — closing acts in §7.2
    # order (Encode → Arc → Review), then the closure (terminal-turn + DONE)
    # as the LAST block. THREE SEPARATE injects (the closure must not be
    # entangled with either block), placed at the end by design: these are
    # the encoder's final acts, so their instructions sit where the action
    # lands (recency). The arc block is the journal mechanism's per-encoder
    # opt-in second component — S1E opts in because it owns the session arc
    # (`session_context_{sid}` → Frame 'Current focus'). Contract-owned text
    # (single source across all encoders) — never hardcoded in the s1e
    # template, so it can't drift.
    try:
        prompt = (journal or _journal(None)).decorate_system(prompt)
    except Exception as e:
        print('[s1e] WARNING: could not inject arc/review block/closure: %s' % e, flush=True)
    return prompt


def _build_catalog(brain, turns, session_id, view_policy=False, now=None,
                   associated_stubs=None):
    """The catalog half of the encoding input, built before the body so
    run_encoding can profile the two halves separately. Gathers the trace
    streams once; the same tuple threads into the timeline's <provenance>.
    `view_policy` (resolved once in run_encoding) turns on catalog aging in
    build_node_catalog. `associated_stubs` — the stubs flag, resolved once in
    run_encoding; None → read the env (tests / standalone callers).

    Returns (node_catalog, cataloged_ids, streams)."""
    from servers.scales.s1.encode_contract import build_node_catalog

    from servers.trace_contract import turn_incoming
    judge_outputs = [r.get("judge_output") for t in turns for r in turn_incoming(t)]

    streams, extra_ids = None, None
    try:
        from servers.scales.s1.trace_links import gather, session_node_ids
        streams = gather(brain, session_id)           # {'surface','encode','touched'}
        extra_ids = session_node_ids(streams['encode'], streams['touched'],
                                    streams['surface'])
    except AttributeError:
        # Stub brain (tests) without query_traces — expected, quiet. Degrade
        # to surfaced-only catalog + self-gathering timeline.
        streams, extra_ids = None, None
    except Exception as e:
        # A real failure (trace-contract drift, malformed record) — LOUD via
        # brain.errors (the monitored surface), then degrade. Mirrors the
        # _turn_links guard; a bare print in the daemon log is not the surface.
        try:
            brain._log_error('s1e_catalog_streams', e,
                             'trace_links gather/union failed; catalog stays surfaced-only')
        except Exception:
            pass
        streams, extra_ids = None, None

    if view_policy and now is None:
        now = _conversation_now_safe(brain, session_id, turns)

    if associated_stubs is None:
        from servers.scales.s1.encoder_view import associated_stubs_enabled
        associated_stubs = associated_stubs_enabled()
    stubs_on = associated_stubs

    # Window-aligned cutoff (arm D): the catalog ages against the same turn
    # window the timeline renders — widen the chat window and the full-depth
    # catalog widens with it. None (no datable head, or flag off) falls back
    # to round-based aging inside build_node_catalog.
    head = window_first_turn(turns) if view_policy else None

    # Associated stubs (the subconscious): production
    # recall over the window's unencoded messages, catalog excluded, rendered
    # as the catalog's LAST entries tagged [associated]. Flag-gated for the
    # A/B. Guarded loud: retrieval is advisory — never block the catalog.
    associated = None
    if stubs_on:
        try:
            from servers.scales.s1.encode_contract import (
                PROVENANCE_TAGS, surfaced_ids_of)
            # Exactly the id set build_node_catalog will catalog (surfaced ∪
            # the provenance categories) — derived from the same parse + the
            # same PROVENANCE_TAGS keys, so exclusion can't drift from the
            # render. Empty ⇔ the catalog will be empty: no home for stubs.
            exclude = surfaced_ids_of(judge_outputs)
            exclude |= set((extra_ids or {}).get('surfaced') or ())
            for key, _tag in PROVENANCE_TAGS:
                exclude |= set((extra_ids or {}).get(key) or ())
            if exclude:
                associated = _associated_stub_ids(
                    brain, turns, session_id, exclude, streams=streams)
        except Exception as e:
            try:
                brain._log_error('s1e_associated_stubs', e,
                                 'associated-stub recall failed; catalog renders without stubs')
            except Exception:
                pass
            associated = None

    try:
        node_catalog, cataloged_ids = build_node_catalog(
            judge_outputs, brain, extra_ids=extra_ids,
            scope=brain.session_scope(session_id), view_policy=view_policy,
            now=now, window_first_turn=head, associated_ids=associated)
    except Exception as e:
        print('[s1e] ERROR building node catalog: %s' % e, flush=True)
        node_catalog, cataloged_ids = '', set()
    return node_catalog, cataloged_ids, streams


def _conversation_now_safe(brain, session_id, turns):
    """Conversation time for the view policy's relative-time renders — replays
    inject historical [Current date:] prefixes; wall-clock would corrupt them
    (the S1 rule — tests/test_time_window_contract). A live session's now is
    the wall clock, so no session start is passed. Guarded: a failure is
    logged and degrades to None (wall-clock render)."""
    try:
        from servers.clock import conversation_now
        from servers.trace_contract import row_text, turn_incoming
        return conversation_now(
            messages=[{'role': 'user', 'content': row_text(r)}
                      for t in turns for r in turn_incoming(t)], brain=brain)
    except Exception as e:
        log_error = getattr(brain, '_log_error', None)
        if log_error is not None:
            log_error('s1e_conversation_now', e,
                      'session=%s — relative times render against wall-clock'
                      % session_short(session_id))
        return None


def _associated_stub_ids(brain, turns, session_id, exclude_ids, streams=None):
    """The subconscious: ids of the K nodes production recall ranks nearest
    this window that the catalog does NOT already show: the encoder receives
    not just the conscious catalog but the associative penumbra recall ranked
    below the surface cut.

    Measured shape (eval/encode_moment_recall_probe.py, [thread:encoder-
    staleness]): one recall per UNENCODED window message, both voices — never
    a whole-window blob (needle absent from its top-200); turnmax aggregation
    (a node's score is its best single-message match — turnsum buried the
    needle at rank 18-20). Production recall is deliberate: LAF, edge walks,
    correction chains and the scope veil all apply — no parallel ranker.

    Time: NO `as_of`, deliberately. Every candidate value breaks something
    (code-review verified, 2026-08-21): a live conversation_now is
    operator-TZ and LAF's masks compare lexicographically against uniform
    UTC, skewing the field by the host offset; a replay's [Current date:]
    prefix predates every corpus node's wall-clock created_at, emptying the
    masked universe (empty field → `laf_unavailable` each round). Live,
    as_of≈now is a no-op mask anyway. A harness replaying against a brain
    that holds FUTURE nodes relative to the replayed moment must pin the
    field itself (the probe passes the run's stored UTC trace timestamp).

    Reads are pure: mark_accessed=False — assembly is machinery, and marking
    would inflate access counts + fatigue with the encoder's own looking
    (the dashboard-observer precedent). Community nodes never render:
    recall's default NODE_TYPE_EXCLUSIONS drops them here, and
    build_node_catalog's community scan is the render chokepoint.

    `turns`: the run's window (_window). Coverage joins through _turn_links
    on the pre-gathered `streams`; a turn with no link is real (degraded
    join / orphan) and counts as unencoded — the probe's convention.

    Returns rank-ordered id list (may be empty). Raises to the caller's
    guard on read failure — advisory, never blocks the catalog.
    """
    from servers.scales.s1.encoder_view import (
        ASSOCIATED_QUERY_CAP, ASSOCIATED_RECALL_LIMIT, ASSOCIATED_SEED_CAP,
        ASSOCIATED_STUBS_K)
    from servers.scales.s1.surface_contract import recall_score
    from servers.trace_contract import row_text

    links, _ends = _turn_links(brain, session_id, turns, streams=streams)

    def _body(ep):
        return row_text(ep)[:ASSOCIATED_QUERY_CAP].strip()

    seeds = []
    for t in turns:
        uid = (t['user'] or {}).get('id') if t['user'] else None
        link = links.get(uid) if uid else None
        if link is not None and link.get('encoded_by'):
            continue    # covered turn — a prior run already encoded it
        for ep in (t['user'], t['assistant']):
            text = _body(ep) if ep else ''
            if text:
                seeds.append(text)
    # Dedup (repeated 'ok'/'continue' turns), then keep the NEWEST seeds —
    # the tail turns are the encoder's working material. The cap bounds the
    # sequential recall loop when every turn reads unencoded (first encode
    # of a long session, or a degraded trace join).
    seeds = list(dict.fromkeys(seeds))[-ASSOCIATED_SEED_CAP:]
    if not seeds:
        return []

    exclude = {(i or '')[:8] for i in exclude_ids if i}
    best = {}
    for text in seeds:
        r = brain.recall(query=text, limit=ASSOCIATED_RECALL_LIMIT,
                         session_id=session_id, source='s1e_associated',
                         mark_accessed=False)
        for h in (r.get('results') or ()):
            nid = str(h.get('id') or '')
            if not nid or nid[:8] in exclude:
                continue
            score = recall_score(h)
            if score > best.get(nid, -1.0):
                best[nid] = score    # turnmax, never sum

    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return [nid for nid, _s in ranked[:ASSOCIATED_STUBS_K]]


def _build_user_content(brain, turns, counter, session_id, precomputed=None,
                        view_policy=None, view_now=None, journal=None, gist=None):
    """Assemble S1 encoding prompt: stable preamble + dynamic body.

    The split is deliberate for caching. The stable preamble (one operational
    sentence — the section legend lives in the system prompt) is
    byte-identical across every encoding cycle and gets a 1h cache breakpoint
    via run_llm_loop's `user_preamble` arg. The dynamic body (continuity,
    failed encodes, catalog, gist, timeline) gets the 5m breakpoint.

    Returns:
        (user_preamble, user_body, catalog_text, catalog_ids)
        - user_preamble: the stable operational line; safe to cache 1h.
        - user_body: dynamic content for this cycle (5m cache).
        - catalog_text: rendered catalog block.
        - catalog_ids: set of node ids in the catalog.

    `precomputed` — the (node_catalog, cataloged_ids, streams) tuple from
    _build_catalog, when run_encoding already built it. None → build here
    (tests / standalone callers).
    `view_policy` — the encoder_view flag, resolved once in run_encoding and
    threaded in; None → read the env (tests / standalone callers). Shapes the
    timeline render (aging rides the precomputed catalog).
    `view_now` — the run's conversation-time instant (resolved once alongside
    the flag); None → resolve here (standalone callers).
    `gist` — the effective `s1e_gist` (brain.get_interaction_effective),
    resolved once in run_encoding so the delta's gist stamp names the gist
    this run resolved (rendered unless its config disables it); None →
    resolve here (standalone callers).
    """
    # The input is ONE unit: the XML lived-sequence timeline (messages + tool
    # actions + per-turn <provenance>) AND the widened catalog (surfaced ∪
    # encoded ∪ authored ∪ recalled, tagged). Both consume the same trace
    # streams, gathered ONCE and threaded into both.
    if view_policy is None:
        from servers.scales.s1.encoder_view import view_policy_enabled
        view_policy = view_policy_enabled()
    if precomputed is not None:
        node_catalog, cataloged_ids, streams = precomputed
    else:
        node_catalog, cataloged_ids, streams = _build_catalog(
            brain, turns, session_id, view_policy=view_policy)

    conv_now = view_now
    if view_policy and conv_now is None:
        conv_now = _conversation_now_safe(brain, session_id, turns)
    timeline = _render_lived_sequence_timeline(
        brain, session_id, turns, streams=streams,
        view_policy=view_policy, now=conv_now, cataloged_ids=cataloged_ids)

    # Previous session context (per-session — no global leak across parallel sessions)
    prev_context = brain.session_context_for(session_id)

    # Continuity — the encoder's prior residue this session (the READ side; the
    # encoder's own prompt framing, NOT the identity Frame): the `## Review`
    # note contract, SESSION-BOUND (scale='s1', session_id → last K=5 runs in
    # THIS conversation; never carries across sessions). Self-labeled block.
    from servers.session_context import SessionContext
    journal_block = (journal or _journal(brain, session_id)).continuity(
        chain_id=SessionContext(session_id, stop_counter=counter).s1e_chain())

    # ── Stable preamble — byte-identical across encoding cycles (1h cache via
    # run_llm_loop's user_preamble). The section legend + reading order live in
    # the registered system prompt (it names <continuity>/<node_catalog>/
    # <timeline>), so the preamble keeps only the operational anchor — two
    # voices describing the layout would drift.
    if os.environ.get('BRAIN_S1E_LISTS_PREAMBLE', '0') in ('1', 'true', 'True'):
        # The operating-guide arm (flag-gated input change, like the view policy
        # and associated stubs): the encoder's first reply carries its four lists
        # as text, then the tool call they call for, in the same reply.
        preamble = (
            "I'm encoding what I've just observed. I read everything below; my "
            "first reply opens with my four lists — changes, targets, fetch, new — "
            "and ends in the tool call they call for.\n"
        )
    else:
        # First person — matches the system prompt's register (the encoder
        # speaks as itself: "This is me encoding my own memory").
        preamble = (
            "I'm encoding what I've just observed. I read everything below before "
            "calling any tools, then put all operations (remember + revise + "
            "connect) in one tool call.\n"
        )

    # ── Dynamic body — varies per cycle, 5m cache. Each section is wrapped in
    # the XML label the system prompt names (<continuity> folds residue + arc;
    # <node_catalog>; <timeline>).
    # Failed-run residue (docs/TRACE-MODES-DESIGN.md §Failed-run residue):
    # encoding_run_failed traces newer than the last successful run mean the
    # turns below were ALREADY attempted and died — the retry must encode
    # with knowledge of the failure, not amnesia.
    failed_block = _render_failed_encodes(brain, session_id)

    continuity = ""
    if journal_block:   # self-labeled ("RECENT REVIEW NOTES …"); empty on a fresh session
        continuity += journal_block
    if prev_context:
        continuity += "Session arc: %s\n" % prev_context
    body = ""
    if continuity:
        # Escaped like every other free-form string in the lived prompt:
        # the producer view carries operator-authored answer text, which
        # must not be able to close this block or forge a sibling tag.
        body += "<continuity>\n%s</continuity>\n\n" % _xml_escape(continuity)
    if failed_block:
        body += "<failed_encodes>\n%s</failed_encodes>\n\n" % failed_block
    if node_catalog:
        body += "<node_catalog>\n%s\n</node_catalog>\n\n" % node_catalog
    # The gist — the last instruction before the timeline, its own K
    # (`s1e_gist`): an override can edit it or turn it off (`enabled`), and
    # its fingerprint says what the encoder actually read. Config is total
    # by construction — subscript.
    if gist is None:
        gist = brain.get_interaction_effective('s1e_gist')
    if gist['config']['enabled']:
        body += gist['template'].rstrip('\n') + "\n\n"
    # `now=` stamp (view policy): the absolute anchor that makes every
    # relative label below invertible — and the current-time declaration
    # the encoder's date resolution never had before the view policy.
    now_attr = ''
    if view_policy and conv_now is not None:
        from servers.scales.s1.encoder_view import timeline_now_attr
        now_attr = timeline_now_attr(conv_now)
    body += "<timeline%s>\n%s</timeline>\n" % (now_attr, timeline)
    return preamble, body, node_catalog, cataloged_ids


def _render_failed_encodes(brain, session_id, cap=3):
    """Render this session's encoding_run_failed traces newer than the last
    successful encoding_run — a short block (≤`cap` newest) so the retry sees
    what its predecessor attempted and how it died. Reads go through the
    query_traces door (session-scoped pulls ignore wall-clock windows —
    eval-replay safe); created_at comparison is transaction-time on both
    sides. Returns '' when there's nothing to show; never raises."""
    try:
        # query_traces returns newest first — limit bounds the pull:
        # newest failures only, and one row is enough for the success mark.
        failed = (brain.query_traces(ref_type='encoding_run_failed',
                                     session_id=session_id, hours=None,
                                     limit=cap * 2)
                  or {}).get('events') or []
        if not failed:
            return ''
        ok = (brain.query_traces(ref_type='encoding_run',
                                 session_id=session_id, hours=None, limit=1)
              or {}).get('events') or []
        last_ok = max((e.get('created_at') or '' for e in ok), default='')
        fresh = sorted((e for e in failed
                        if (e.get('created_at') or '') > last_ok),
                       key=lambda e: e.get('created_at') or '')
        if not fresh:
            return ''
        lines = []
        for e in fresh[-cap:]:
            md = e.get('metadata') or {}
            # Error text lands inside the XML-structured lived prompt —
            # escape it so an API error echoing request content can't close
            # the block or forge timeline tags (same guard as message text).
            parts = ['stop %s FAILED: %s' % (
                md.get('stop_counter', '?'),
                _xml_escape(str(md.get('error') or
                                e.get('summary') or '?')))]
            n_partial = len(md.get('partial_actions') or [])
            if n_partial:
                parts.append('%d action(s) completed before death (their '
                             'writes are already in the graph — do not '
                             're-create them)' % n_partial)
            if md.get('payload_pointer'):
                parts.append('full conversation at failure: %s'
                             % md['payload_pointer'])
            lines.append('- ' + '; '.join(parts))
        return '\n'.join(lines) + '\n'
    except Exception as e:
        brain._log_error('s1e_failed_encodes_block', e,
                         'session=%s — omitting block' % session_short(session_id))
        return ''


def _xml_escape(s):
    """Escape the three XML-significant chars so message/tool text can't malform
    the lived-sequence timeline or forge tags (e.g. a '</other>' or '<turn>'
    substring in a user prompt, or 'Bash: a > b' / 'Grep: <svg>' in a tool cue)."""
    return (s or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def window_first_turn(turns):
    """The 1-based turn number the rendered timeline opens on — the
    window-aligned catalog cutoff (encode_contract.build_node_catalog's
    `window_first_turn`) for the run's window (_window). None when the window
    has no datable head, which the caller reads as "no window cutoff
    available", falling back to round-based aging rather than ageing
    everything."""
    from servers.scales.s1.trace_links import display_turn
    for t in turns:
        if t.get('user'):
            return display_turn(t['user'].get('chain_id'))
    return None


def _render_lived_sequence_timeline(brain, session_id, turns, streams=None,
                                    view_policy=False, now=None,
                                    cataloged_ids=None):
    """The XML lived sequence — messages + tool actions interleaved (piece 1).

    `view_policy` (encoder_view, resolved once in run_encoding): ON, the
    render policies apply — already-encoded turns get a stubbed <actions>
    element; node-op/lookup tool lines provenance already shows are dropped
    from every <actions>; <provenance> renders the verbs split
    (created/revised/recalled/searched/archived (me)) instead of the merged
    encoded(me). OFF: the pre-policy render (message text is full in both
    arms — the encoded-turn trim is retired).

    `cataloged_ids` — the ids the node catalog rendered this run
    (build_node_catalog). Under the policy a provenance ref carries its title
    only when its node is NOT among them: a node in the catalog is named by
    its id, whose full body the reader already has above; any other node by
    its title, the only body it has in the prompt. None → nothing counts as
    cataloged, so every ref keeps its title.

    `turns` is the run's window (_window, read through
    brain.conversation_turns). Tool actions arrive as `tool_result` rows whose
    `summary` already carries the per-tool cue ("Edit: foo.py", "Bash: …").

    `streams` (optional) is the pre-gathered {'surface','encode','touched'} dict from
    one trace_links.gather call, threaded in so the catalog and the <provenance>
    block share a single pull. None → _turn_links gathers its own.

    surfaced/judge_output is deliberately NOT rendered here — it belongs to the
    <provenance> block (piece 2). Piece 1 is a pure timeline read.
    """
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    from servers.scales.s1.encoder_view import SPEAKER_ELEMENTS
    from servers.trace_contract import (SPEAKER_REF_TYPES, row_speaker, row_text,
                                        turn_incoming)
    lim = ENCODING_AGENT['message_display_limit']

    def _text(ep):
        # The row's full text (the store-capped body, any store-limit marker
        # kept) when the timeline limit is None, escaped for XML.
        body = row_text(ep)
        cut = body[:lim] + ('…' if lim is not None and len(body) > lim else '')
        return _xml_escape(cut)

    # Piece 2: <provenance> — the trace↔node links (what recall surfaced / what
    # prior runs encoded, joined by stop; one line per stop). Guarded: any
    # failure degrades to the piece-1 timeline (no provenance), never breaks
    # the lived arm.
    links, ends = _turn_links(brain, session_id, turns, streams=streams)

    # Provenance ref titles. Legacy arm: every ref carries its 1-line «tag».
    # Policy: only a node with no catalog entry does (cataloged_ids) —
    # the catalog's ids are what the reader dereferences. Fetched NAKED in one
    # batched get_bulk (a title-only lookup). Guarded: a stub brain (tests, no
    # _nodes) or any failure → bare refs, never a broken render.
    titles = {}
    ref_ids = set()
    for lk in links.values():
        ref_ids.update(lk.get('surfaced') or ())
        ref_ids.update(lk.get('encoded') or ())
        ref_ids.update(lk.get('authored') or ())
        if view_policy:
            # the verb split also renders recalled/looked_up/archived refs —
            # fetch their titles too (authored covers created ∪ revised)
            ref_ids.update(lk.get('recalled') or ())
            ref_ids.update(lk.get('looked_up') or ())
            ref_ids.update(lk.get('archived') or ())
    if view_policy:
        ref_ids -= set(cataloged_ids or ())
    if ref_ids:
        try:
            titles = {nid: (row.get('title') or '')
                      for nid, row in brain._nodes.get_bulk(list(ref_ids)).items()}
        except Exception:
            titles = {}

    # Covered turns keep their FULL message text — trimming an encoded turn
    # would defeat the purpose; what a covered turn drops is its <actions>,
    # stubbed below. The `encoded` attr states coverage on the turn itself.
    # No attr when the trace-link join is unavailable (degraded piece-1 path —
    # coverage unknown).
    from servers.action_policy import load_action_policy
    from servers.scales.s1.encoder_actions import prepare_action_block, render_action_blocks
    action_policy = load_action_policy()
    action_blocks = [prepare_action_block(
        t['actions'], is_tail=(i == len(turns) - 1),
        encoded=bool(links.get((t['user'] or {}).get('id'), {}).get('encoded_by')),
        view_policy=view_policy, policy=action_policy)
        for i, t in enumerate(turns)]
    action_xml, action_notice = render_action_blocks(action_blocks, action_policy)

    if view_policy:
        from servers.scales.s1.trace_links import display_turn

    out = ""
    names = {}   # speaker → name, asked once per render
    for n, t in enumerate(turns, 1):
        uid = (t['user'] or {}).get('id') if t['user'] else None
        link = links.get(uid) if uid else None
        n_disp = n   # n stays the window ordinal — the `ends` index below
        if view_policy:
            # the real turn number — session-global, 1-based (trace_links
            # owns the axis; run attribution 'encoded(me, turn 5)' shares it
            # with no shift). Orphan turns keep the ordinal.
            real = display_turn((t['user'] or {}).get('chain_id')) if t['user'] else None
            n_disp = real if real is not None else n
        enc_attr, is_enc = '', False
        if link is not None:
            is_enc = bool(link.get('encoded_by'))
            enc_attr = ' encoded="%s"' % ('true' if is_enc else 'false')
        age_attr = ''
        if view_policy and now is not None:
            # per-turn recency against conversation time — with the <timeline
            # now=…> stamp this gives full orientation (relative labels are
            # invertible). Caveat: replayed corpora stamp turns at replay
            # wall-clock, so ages there degrade to 'just now' — both A/B arms
            # share the artifact.
            from servers.scales.s1.surface_contract import _relative_time
            first = t['user'] or t['assistant'] or (t['actions'][0] if t['actions'] else None)
            age = _relative_time((first or {}).get('created_at'), now=now, fine=True)
            if age:
                age_attr = ' age="%s"' % age
        out += '<turn n="%d"%s%s>\n' % (n_disp, age_attr, enc_attr)
        # Tag vocabulary is identity-native, not role-native:
        # <me> = my side of the exchange; the incoming side renders as its
        # speaker's element (encoder_view.SPEAKER_ELEMENTS) — <other> for the
        # operator, my partner this session, named when the session declares
        # who that is. Presentation-layer only: the substrate keeps its
        # ref_types; the join is by trace id.
        for row in turn_incoming(t):
            speaker = row_speaker(row)
            el = SPEAKER_ELEMENTS[speaker]
            if speaker not in names:
                names[speaker] = brain.speaker_for(
                    session_id, SPEAKER_REF_TYPES.get(speaker, row.get('ref_type')))
            name = (' name="%s"' % _xml_escape(names[speaker]).replace('"', '&quot;')
                    if names[speaker] else '')
            out += '  <%s%s trace="%s">%s</%s>\n' % (
                el, name, row.get('id', ''), _text(row), el)
        # <provenance> sits right after the incoming side: chronologically,
        # recall/surface fires on the prompt — before my reply exists.
        prov = _render_provenance(links, ends, t, n - 1, titles,
                                  view_policy=view_policy)
        if prov:
            out += '  <provenance>%s</provenance>\n' % prov
        if t['assistant']:
            out += '  <me trace="%s">%s</me>\n' % (
                t['assistant'].get('id', ''), _text(t['assistant']))
        out += action_xml[n - 1]
        out += '</turn>\n\n'
    return out + action_notice


def _short_refs(ids, titles=None, title_first=False):
    """Node ids as 8-char `id:` refs, each carrying its 1-line «tag» (title) when
    known — the locality affordance (fork #2): WHERE a node came up in the
    timeline, complementary to the catalog's full WHAT-it-is. The 8-char short
    still matches the catalog so a ref dereferences there; the «tag» saves the
    scan. Renders a bare `id:` when no title is mapped (a cataloged node under
    the view policy, a stub brain, or an id the title fetch missed).

    `title_first` (view policy): the canonical node_ref token — title-leading,
    double-quoted, id in parens, identical to every other mention of a node in
    the payload (catalog headers, edges, corrections); the id-first «tag» form
    is the legacy arm's."""
    from servers.contract import node_ref
    titles = titles or {}
    out = []
    for i in ids:
        short = str(i)[:8]
        # Escape the title + collapse whitespace: it lands inside the strict
        # per-turn timeline XML, next to message/action text that _xml_escape()
        # already guards — a title with '<'/'>'/'&' (or a forged '</provenance>')
        # must not malform it, and a stray newline must not break the one-line
        # provenance.
        t = _xml_escape(' '.join((titles.get(i) or '').split()))
        if title_first:
            out.append(node_ref(short, t) if t else node_ref(short))
        elif not t:
            out.append('id:%s' % short)
        else:
            out.append('id:%s «%s»' % (short, t))
    return ' '.join(out)


_NO_ENDS = {'stop': frozenset(), 'run': frozenset()}


def _turn_links(brain, session_id, turns, streams=None):
    """Compute the trace↔node link map + where each grouped ref list shows.

    Returns (links, ends):
      links — {user_trace_id: {surfaced, encoded, encoded_by, …}} from trace_links.
      ends  — window indices of the LAST turn of each group, where that group's
              refs render once instead of on every turn in it:
              'stop' — the last turn on each chain stop. The link join is by
                       stop, so every turn on one stop (several prompts before
                       one reply) holds the same refs; the line renders on the
                       last, after every prompt that fed it, beside the reply.
              'run'  — the last turn each run covers, where its full encoded
                       id-list renders (adjacent to the unencoded boundary);
                       earlier covered turns carry only their `encoded` attr —
                       no 5× repetition (which the design warns would nudge
                       dense source_refs).

    `streams` (optional): the pre-gathered {'surface','encode','touched'} dict, so
    the catalog and this share one gather. None → gather here.

    Guarded: any failure returns ({}, {}) → the timeline renders exactly as piece
    1 did. Also the path the piece-1 stub brains take (no query_traces).
    """
    try:
        from servers.scales.s1.trace_links import gather, nodes_for_traces
        targets = [t['user'] for t in turns if t['user']]
        st = streams if streams is not None else gather(brain, session_id)
        links = nodes_for_traces(st['surface'], st['encode'], targets,
                                 touched_traces=st['touched'])
    except AttributeError:
        # Expected, quiet: a stub brain (tests) without query_traces. Degrade to
        # the piece-1 timeline (no provenance) — production brains always have it.
        return {}, _NO_ENDS
    except Exception as e:
        # A real failure (trace-contract drift, malformed record) — LOUD, then
        # degrade (brain.errors is the surface, a bare print is not).
        # Provenance is advisory; never block the timeline.
        try:
            brain._log_error('s1e_provenance', e,
                             'trace-link provenance failed; timeline renders without it')
        except Exception:
            pass
        return {}, _NO_ENDS

    last = {'stop': {}, 'run': {}}   # group key → index of its last turn
    for idx, t in enumerate(turns):
        user = t['user'] or {}
        last['stop'][user.get('chain_id') or idx] = idx   # ascending walk → last wins
        eb = (links.get(user.get('id')) or {}).get('encoded_by')
        if eb:
            last['run'][eb] = idx
    return links, {group: set(by_key.values()) for group, by_key in last.items()}


def _render_provenance(links, ends, turn, idx, titles=None,
                       view_policy=False):
    """One <provenance> line for a turn — REAL refs only, no coverage markers.

    Coverage lives on the turn itself (the `encoded="true|false"` attribute the
    timeline render stamps), so provenance never says "covered, nothing to
    show". The line is the stop's: it renders only on the stop's last turn
    (`ends['stop']`, from _turn_links), so turns sharing a stop show it once.
    What renders: `surfaced:` refs (the stop's prompts' picks), the owning
    run's full id-list ONCE at the run's last turn (`ends['run']`; an edge-only
    run has an empty id set → nothing), and the session's own node ops (joined by
    stop; empty in replayed eval corpora) — legacy: the merged
    `encoded(me):` (link['authored'] = created ∪ revised); view policy ON:
    the verb split (encoder_view.PROVENANCE_SPLIT — created / revised /
    recalled / searched / archived), which is what lets the node-op action
    lines drop without losing the information. `titles` maps id→title; a ref
    whose id it lacks renders as the id alone (under the policy the map holds
    only nodes with no catalog entry). Returns '' when there's nothing real to
    show.
    """
    uid = (turn['user'] or {}).get('id') if turn['user'] else None
    link = links.get(uid) if uid else None
    if not link or idx not in ends['stop']:
        return ''
    parts = []
    tf = bool(view_policy)   # title-first refs under the policy (system-wide shape)
    if link['surfaced']:
        parts.append('surfaced: %s' % _short_refs(link['surfaced'], titles, tf))
    eb = link['encoded_by']
    if eb and idx in ends['run'] and link['encoded']:
        # policy: turn-coordinate attribution ('encoded(me, turn 36)'); the
        # legacy arm keeps its 'encoded(S1S)' string byte-for-byte
        run_label = 'encoded(S1S)'
        if view_policy:
            from servers.scales.s1.encoder_view import encoded_run_label
            run_label = encoded_run_label(link.get('encoded_by_stop'))
        parts.append('%s: %s' % (run_label, _short_refs(
            link['encoded'], titles, tf)))
    if view_policy:
        from servers.scales.s1.encoder_view import PROVENANCE_SPLIT
        for label, keys, minus in PROVENANCE_SPLIT:
            skip = {i for k in minus for i in (link.get(k) or ())}
            ids = list(dict.fromkeys(
                i for k in keys for i in (link.get(k) or ()) if i not in skip))
            if ids:
                parts.append('%s: %s' % (label, _short_refs(ids, titles, tf)))
    elif link.get('authored'):
        parts.append('encoded(me): %s' % _short_refs(link['authored'], titles))
    return ' | '.join(parts)


def _write_pre_traces(brain, dispatch_fn, turns, user_content, enc_chain,
                      session_id, interaction, catalog_ids=(),
                      prompt_pointer=None):
    """Write S1 encode traces: O (encoding prompt) and K (node catalog).

    `interaction` — the run's s1e stamp; the O summary names its fingerprint.

    `prompt_pointer` — the db_dir-relative payload pointer for this run's
    recorded prompt (brain.record_payload). It becomes the O trace's ref_id;
    '' when recording was gated off or failed (readers render "no payload").

    `catalog_ids` — the ids build_node_catalog actually rendered for this
    run. The K trace's ref_id carries them (8-char, comma-separated); the S2
    consolidation blindness check (_load_catalog_data) reads that list to
    decide whether node B was created blind to A.
    [associated] stub ids are IN this set by design — they render in the
    catalog, so the encoder was not blind to them and the K trace must say so.
    """
    try:
        turn_count = len(turns) if turns else 0
        node_ids = {(cid or '')[:8] for cid in (catalog_ids or ()) if cid}

        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'O',
            'ref_type': 'encoding_prompt',
            'ref_id': prompt_pointer or '',
            'summary': '%d turns, %d chars context, interaction: s1e K#%s' % (
                turn_count, len(user_content), interaction['fingerprint']),
            'session_id': session_id})
        dispatch_fn('trace_append', {
            'chain_id': enc_chain, 'scale': 's1', 'event_type': 'K',
            'ref_type': 'node_catalog',
            'ref_id': ','.join(sorted(node_ids)),
            'summary': '%d unique nodes in catalog from %d turns' % (
                len(node_ids), turn_count),
            'session_id': session_id})
    except Exception as e:
        print('[s1e] TRACE ERROR: %s' % e, flush=True)


def _get_tool_schemas():
    """Get S1 encoding tool schemas from brain_mcp (single source of truth)."""
    from servers import brain_mcp
    from servers.scales.s1.encode_contract import ENCODING_TOOLS
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["inputSchema"]}
            for t in brain_mcp.TOOLS if t["name"] in ENCODING_TOOLS]
