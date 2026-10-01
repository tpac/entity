"""S1 Surface Chain — surface relevant memories, graph expansion, correction enrichment, trace writing.

Scale: S1 (Turn integration)
Chain: s1r (surface)
Interaction: 'surface' in interactions table (learnable boundary)

Triggered by: hook_recall (UserPromptSubmit) in daemon_hooks.py
Reads: recall candidates from brain.recall(), interactions table
Writes: S1 traces (O/K/Δ)
"""

import json
import re
import time

from servers.scales.dispatch import load_env
from servers.scales.runner import assistant_block, read_usage, sum_usage
from servers.scales.s1 import surface_capture
from servers.trace_contract import (
    build_selection_metadata, build_run_telemetry, check_surface_telemetry,
    interaction_fields)


# SURFACE_SELECTION_SCHEMA lives in surface_contract.py alongside the other
# surface I/O contracts (the render formats for each mode). Imported below.
def seen_node_ids(recent_messages):
    """Short (8-char) ids of nodes already injected in this session's window.

    Reads the per-turn `surfaced` lists the hook already fetched
    (get_session_turns with_surfaced=True) — no extra query. Used to drop
    already-shown nodes from the candidate pool BEFORE the final cap: the
    v13 design said out-of-scope is structural (<shown> vs <candidate>),
    but the same id kept re-entering <candidates> and Haiku re-picked it
    over the never-select-again rule (verified in the 2026-07-27 20:58
    capture: 3a507484 re-selected with its <shown> element in-prompt).
    Selected-only by construction — spread/expanded nodes are not in the
    `surfaced` lists and are NOT deduped here (known gap, separate fix).
    """
    seen = set()
    for t in recent_messages or []:
        for s in (t.get('surfaced') or []):
            sid = str(s.get('id') or '')[:8]
            if sid:
                seen.add(sid)
    return seen


def _call_surface(brain, candidates_data, user_message,
                  recent_messages, session_id, result, frame='',
                  scope=None):
    """Call Haiku to surface relevant nodes from candidates.

    Returns: (surfaced_dict, surface_prompt, max_tokens, stamp, telemetry)
        surfaced_dict has 'selected' list. Empty on failure.
        stamp is the K-provenance dict resolved with the instructions
        ('fingerprint'/'source'/'version'/'id') for the trace writers.
        telemetry is the shared run-cost dict (build_run_telemetry kwargs:
        token counts + elapsed_ms + rounds + truncated) for the K trace.

    The call is the agentic loop: fetch tools (fetch_tools.TOOL_DEFINITIONS)
    offered every round, final round forced to select.
    """
    from servers.scales.s1.surface_contract import (
        build_surface_prompt, prepare_presented_candidates,
        presentation_shuffle_seed, SURFACE_MODEL, SURFACE_VARIANT)

    # Retrieval stats from recall result.
    retrieval_stats = result.get('_retrieval_stats') if isinstance(result, dict) else None

    # One resolution of the override model (DB override, if any, overlaid on
    # the code default — total by construction): template, config and the
    # K-provenance stamp ({'fingerprint','source','version','id'}) come from
    # the same read, so the stamp threaded to the trace writers and the
    # replay capture names the instructions this call sends.
    effective = brain.get_interaction_effective('surface')
    surface_instructions = effective['template']
    stamp = effective['stamp']

    # Layout rides in the interaction CONFIG so template and renderer can
    # only flip together (pinned by test_interaction_defaults). One layout
    # ships, so this selects nothing today — it is the value the capture
    # stamps, so a replayed run records which renderer produced it.
    layout = effective['config']['layout']

    # Presentation shuffle (2026-07-14, RECALL-SR-REDESIGN.md §20.12 A2):
    # the menu Haiku sees is a deterministic per-turn shuffle of the
    # scorer's ranking — position bias dies at the source, and picked/
    # dropped trace rows become uniform-propensity P3 training data.
    # candidates_data itself is NEVER reordered: traces (cand_detail),
    # short_to_full, and the admission floor all keep scorer order.
    # presented_order (the exact round-1 menu, pre-tool-fetches) is stashed
    # per-session for the K trace — same pattern as _surface_tool_traces.
    shuffle_seed = presentation_shuffle_seed(session_id, user_message)
    presented_order = [
        str(c.get('id', ''))[:8]
        for c in prepare_presented_candidates(candidates_data, shuffle_seed)]
    if not hasattr(brain, '_surface_presented'):
        brain._surface_presented = {}
    brain._surface_presented[session_id] = {
        'shuffle_seed': shuffle_seed, 'presented_order': presented_order}

    user_content, max_tokens = build_surface_prompt(
        candidates_data, user_message,
        recent_messages=recent_messages,
        retrieval_stats=retrieval_stats,
        frame=frame,
        shuffle_seed=shuffle_seed,
        scope=scope)

    surface_prompt = (surface_instructions + "\n\n---\n\n" + user_content) \
        if surface_instructions else user_content

    # Production capture at the Haiku boundary (replay-bench corpus). Must
    # begin BEFORE the agentic loop: it deep-copies candidates_data, which
    # the loop mutates in place with tool-fetched entries — replay needs
    # the round-1 pool. Returns None when disabled; every consumer of
    # `capture` below is None-safe.
    capture = surface_capture.begin(
        brain, candidates_data=candidates_data, user_message=user_message,
        recent_messages=recent_messages,
        retrieval_stats=retrieval_stats, frame=frame, layout=layout,
        shuffle_seed=shuffle_seed, scope=scope,
        surface_instructions=surface_instructions,
        interaction_stamp=stamp, user_content=user_content,
        max_tokens=max_tokens, variant=SURFACE_VARIANT, model=SURFACE_MODEL,
        session_id=session_id)

    # Shared client, built-and-cached once on the brain. No per-call throwaway
    # fallback: Brain._ensure_anthropic_client is the single construction site
    # and self-heals if a boot-warmup failure left the client unset.
    client = brain._ensure_anthropic_client()

    # Haiku has tools, can extend the candidate pool before final selection.
    # Tool-fetched candidates are appended to `candidates_data` in place so
    # the downstream short_to_full mapping resolves them. The window's shown
    # set (the same one that deduped the cosine pool in the hook) gates tool
    # results too.
    raw, tool_trace, telemetry = _call_surface_agentic(
        client, brain, candidates_data, surface_instructions,
        user_content, max_tokens, session_id, SURFACE_MODEL,
        capture=capture, scope=scope,
        seen_ids=seen_node_ids(recent_messages))
    # Attach tool trace to brain for the caller to write into K trace.
    # Stashed on the brain instance per-session-id so parallel sessions
    # don't clobber each other.
    try:
        if not hasattr(brain, '_surface_tool_traces'):
            brain._surface_tool_traces = {}
        brain._surface_tool_traces[session_id] = tool_trace
    except Exception:
        pass

    # Parse JSON — robust to the three shapes Haiku sometimes returns:
    #   (a) bare JSON: {"selected": [...]}
    #   (b) fenced: ```json\n{...}\n```
    #   (c) JSON + trailing prose: {...}\n\nHere's why I picked...
    # `raw_decode` consumes the first valid JSON object and reports the
    # tail, which we discard — no "Extra data" crash on (c).
    # Stash the capture (raw attached) for run_surface to finish() with the
    # resolved selection — same per-session-id stash pattern as
    # _surface_tool_traces, so parallel sessions can't clobber each other.
    if capture is not None:
        capture['output'] = {'raw': raw}
        if not hasattr(brain, '_surface_captures'):
            brain._surface_captures = {}
        brain._surface_captures[session_id] = capture

    surfaced = _parse_surfacer_json(raw)
    if surfaced is None and raw:
        # We had a non-empty Haiku response but couldn't parse anything
        # dict-shaped from it. Surface this — it's the silent-failure mode
        # that produced empty additionalContext at N=15 (fishing query).
        brain._log_error(
            'surface_haiku_unparseable',
            ValueError('Haiku response did not yield a parseable JSON dict'),
            'first 300 chars: %r' % raw[:300])
        surfaced = {"selected": []}
    elif surfaced is None:
        surfaced = {"selected": []}

    return surfaced, surface_prompt, max_tokens, stamp, telemetry


def _call_surface_agentic(client, brain, candidates_data, surface_instructions,
                           user_content, max_tokens, session_id, model,
                           max_rounds=2, capture=None,
                           scope=None, seen_ids=None, run_tool=None):
    """Agentic surface call: Haiku may use fetch tools to extend the candidate
    pool before final JSON selection.

    The final round is sent with tool_choice='none', so Haiku must finalize
    with the selection JSON — max_rounds is the hard cap on API calls.

    Returns: (raw_final_text, tool_trace, telemetry) where tool_trace is a list
    of per-round dicts {round, stop_reason, total_ms, <USAGE_FIELDS>,
    tool_calls: [...]} for trace observability (total_ms + usage per API call,
    mirroring run_llm_loop's per_round_stats), and telemetry is the shared
    run-cost dict (build_run_telemetry kwargs) summed across the loop's
    Haiku rounds.

    Mutates `candidates_data` IN PLACE — tool-fetched candidates are appended
    so the downstream short_to_full ID mapping (in run_surface) can resolve them.

    `seen_ids` (8-char): nodes already shown to this stream in the window.
    Tool results carrying them are dropped before Haiku sees them — the
    stream has them in context, and a re-fetch through a tool was how one
    audited turn re-rendered 8,431 chars it already had.

    `run_tool` executes each tool call (default fetch_tools.execute_tool,
    which reads `brain`). An eval whose whole candidate menu is a fixture
    passes its own, so the loop runs unchanged while the tools answer from
    the fixture instead of a brain.
    """
    from servers.scales.s1.fetch_tools import (
        TOOL_DEFINITIONS, execute_tool, format_tool_result_for_haiku,
    )
    run_tool = run_tool or execute_tool
    from servers.scales.s1.surface_contract import (
        SURFACE_SELECTION_SCHEMA, CACHE_MIN_PREFIX_TOKENS)

    # Track existing IDs to dedupe tool-fetched candidates against cosine pool
    existing_ids = {c.get('id') for c in candidates_data if isinstance(c, dict)}

    # Admission-floor reference: median score of the ORIGINAL cosine pool,
    # snapshotted before any tool results join (recall_topical results are
    # score-comparable and must beat this to be admitted — see the floor
    # at the execute_tool site below).
    import statistics as _stats
    _pool_scores = [c.get('score') or 0 for c in candidates_data
                    if isinstance(c, dict) and (c.get('score') or 0) > 0]
    _pool_median = _stats.median(_pool_scores) if _pool_scores else 0.0

    # Cache breakpoints (runner convention: BP1 system 1h, BP2 user 5m):
    #   BP1 (in api_kwargs below) — last system block. Caches tools+system,
    #   which is byte-identical ACROSS recalls (cross-recall reuse) and
    #   survives the final round's tool_choice flip — tool_choice changes
    #   invalidate only the messages-tier cache (Anthropic invalidation
    #   hierarchy), never the tools/system tiers.
    #   BP2 (here) — the round-1 user content (~20K tokens incl. the
    #   candidate pool): read back by round 2 at 0.1× price and much
    #   faster prefill. Requires tools to be byte-identical across rounds.
    # Prompts under the model's 4096-token cacheable minimum are silently
    # not cached (no error).
    messages = [{"role": "user", "content": [
        {"type": "text", "text": user_content,
         "cache_control": {"type": "ephemeral"}}]}]
    tool_trace = []
    raw_final = ''

    # Cost telemetry — summed across rounds (the agentic loop calls Haiku up to
    # max_rounds times) via the shared sum_usage, then built through the shared
    # builder (one construction point, same as the encoders).
    _t0 = time.time()
    usage_total = read_usage(None)   # all-zero baseline of the right keys
    rounds_used = 0
    truncated = 0

    def _telemetry():
        return build_run_telemetry(
            **usage_total,
            elapsed_ms=int((time.time() - _t0) * 1000),
            rounds=rounds_used, truncated=truncated, model=model)

    def _absorb_response(resp):
        """Fold one Haiku response into the running cost totals and capture
        any text content as the candidate final answer. The ONE place a
        response's usage/truncation/text is absorbed, so telemetry can't
        drift. Returns the response's usage dict for per-round checks."""
        nonlocal rounds_used, truncated, raw_final
        rounds_used += 1
        ru = read_usage(resp)
        sum_usage(usage_total, ru)
        if getattr(resp, 'stop_reason', None) == 'max_tokens':
            truncated += 1
        for block in resp.content:
            if getattr(block, 'type', None) == 'text':
                raw_final = block.text.strip()
        return ru

    # Anthropic Structured Outputs runs on EVERY round, alongside tools.
    # When Haiku tool-uses, the schema doesn't apply to tool_use blocks;
    # when Haiku finalizes with a text response, the schema enforces
    # SURFACE_SELECTION_SCHEMA. Round 1 needs it too: when Haiku skips tools
    # and answers in prose on round 1, the loop exits at
    # `stop_reason != 'tool_use'`, and unconstrained prose would be
    # unparseable (surface_haiku_unparseable).
    output_config = {
        'format': {
            'type': 'json_schema',
            'schema': SURFACE_SELECTION_SCHEMA,
        },
    }

    for round_idx in range(max_rounds):
        is_final = (round_idx == max_rounds - 1)

        # Tools ride on EVERY round byte-identical (2026-07-02) so the BP1
        # prefix never shifts. The FINAL round adds tool_choice='none'
        # (2026-07-11): Haiku must answer with the selection JSON, so
        # max_rounds IS the hard API-call cap — no forced-finalize third
        # call (that path cost an extra ~5.7s and breached the 20s hook
        # budget on 2-tool-round recalls).
        api_kwargs = {
            'model': model,
            'max_tokens': max_tokens,
            'system': [{"type": "text", "text": surface_instructions,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            'messages': messages,
            'output_config': output_config,
            'tools': TOOL_DEFINITIONS,
        }
        if is_final:
            api_kwargs['tool_choice'] = {'type': 'none'}

        _t_call = time.time()
        try:
            api_resp = client.messages.create(**api_kwargs)
        except Exception as e:
            brain._log_error('surface_agentic_api', e,
                              'agentic Haiku call round=%d' % round_idx)
            return raw_final, tool_trace, _telemetry()
        _call_ms = int((time.time() - _t_call) * 1000)

        # Accumulate cost + capture any text (the candidate final answer).
        round_usage = _absorb_response(api_resp)
        # Cache visibility (operator ask): round 2+ shares round 1's prefix
        # byte-for-byte, so zero cache reads on a cacheable-sized prompt
        # means the cache broke (tools drift, block-shape change). Gate on
        # prompt size so sub-minimum prompts (tests, tiny brains) don't warn.
        if round_idx > 0 and (round_usage.get('cache_read_tokens') or 0) == 0 \
                and (round_usage.get('input_tokens') or 0) \
                    + (round_usage.get('cache_creation_tokens') or 0) \
                    > CACHE_MIN_PREFIX_TOKENS:
            brain._log_warning(
                'surface_cache_miss',
                'agentic round %d read 0 cache tokens (input=%d created=%d) '
                '— round-1 prefix should have hit' % (
                    round_idx, round_usage.get('input_tokens') or 0,
                    round_usage.get('cache_creation_tokens') or 0),
                'session=%s' % session_id)

        stop_reason = api_resp.stop_reason
        # Per-API-call cost, same field names as run_llm_loop's
        # per_round_stats (total_ms + USAGE_FIELDS; no ttft_ms — this path
        # is non-streaming). Distinguishes a slow/retried call (high
        # total_ms, normal output) from a verbose one (total_ms tracks
        # output_tokens at ~20ms/token).
        round_record = {'round': round_idx, 'stop_reason': stop_reason,
                         'total_ms': _call_ms, **round_usage,
                         'tool_calls': []}

        if stop_reason != 'tool_use':
            tool_trace.append(round_record)
            break

        if is_final:
            # Unreachable via the API: this round was sent with
            # tool_choice='none', so stop_reason can't be tool_use. Reaching
            # it means the constraint stopped constraining — log loudly,
            # keep whatever text was absorbed, and spend no extra call.
            brain._log_warning(
                'surface_final_round_tool_use',
                "stop_reason=tool_use despite tool_choice='none' on the "
                'final round — constraint not honored',
                'round=%d session=%s' % (round_idx, session_id))
            tool_trace.append(round_record)
            break

        # Append Haiku's full assistant message (with tool_use blocks) to history.
        assistant_blocks = []
        tool_results = []
        for block in api_resp.content:
            assistant_blocks.append(assistant_block(block))
            if getattr(block, 'type', None) == 'tool_use':
                tool_use_id = block.id
                tool_name = block.name
                tool_input = block.input or {}
                # Execute the tool
                exec_result = run_tool(brain, tool_name, tool_input,
                                       session_id=session_id)
                # Admission floor (2026-06-12): recall_topical scores come
                # from the same recall pipeline as the cosine pool, so they
                # are directly comparable — a fetched node scoring below the
                # original pool's median doesn't beat what's already here.
                # Score field parity is pinned by surface_contract.recall_score
                # (both the pool and the tool read it). Filter exec_result IN
                # PLACE so the rendered tool output and the candidate pool
                # agree (Haiku must never see an id it can't select). Other
                # tools get no floor: verbatim, aspect, expand_node and
                # recall_by_time's time/edge tiers carry synthetic scores, and
                # recall_by_time's query tiers, though recall-scored, rank by
                # tier first — a floor would cut its query+time hits.
                _raw_results = [c for c in (exec_result.get('results') or [])
                                if isinstance(c, dict)]
                _dropped_below_floor = 0
                _dropped_ids = []
                if tool_name == 'recall_topical' and _pool_median > 0:
                    _kept = []
                    _dropped = []
                    for c in _raw_results:
                        if (c.get('score') or 0) >= _pool_median:
                            _kept.append(c)
                        else:
                            _dropped.append(c)
                    _dropped_below_floor = len(_dropped)
                    _dropped_ids = [str(c.get('id') or '')[:8] for c in _dropped]
                    exec_result['results'] = _kept
                    if _raw_results and not _kept:
                        # Tripwire: the tool fetched candidates and the floor
                        # dropped every one. Occasional all-drops are fine;
                        # EVERY call all-dropping means the score contract
                        # forked again (the 3-week silent death of 2026-07).
                        _top_fetched = max((c.get('score') or 0)
                                           for c in _raw_results)
                        brain._log_warning(
                            'surface_floor_dropped_all',
                            'admission floor dropped ALL %d recall_topical '
                            'results (pool_median=%.3f top_fetched=%.3f) — '
                            'score-contract drift if this repeats'
                            % (len(_raw_results), _pool_median, _top_fetched),
                            'args=%r' % str(tool_input)[:200])
                # Same-session dedup on tool results — the shown set that
                # already gates the cosine pool. Filtered IN PLACE so the
                # rendered tool output and the pool agree (Haiku never sees
                # an id it can't select).
                _dropped_seen_ids = []
                if seen_ids:
                    _kept_seen = []
                    for c in (exec_result.get('results') or []):
                        _cid8 = (str(c.get('id') or '')[:8]
                                 if isinstance(c, dict) else '')
                        if _cid8 and _cid8 in seen_ids:
                            _dropped_seen_ids.append(_cid8)
                        else:
                            _kept_seen.append(c)
                    exec_result['results'] = _kept_seen
                # Append fetched results to candidates_data (dedupe)
                for cand in exec_result.get('results') or []:
                    cid = cand.get('id') if isinstance(cand, dict) else None
                    if cid and cid not in existing_ids:
                        candidates_data.append(cand)
                        existing_ids.add(cid)
                # Record for trace. result_ids/dropped_ids make per-tool
                # selection attribution computable downstream (LAF training
                # reads these traces) — counts alone can't say which tool
                # sourced a picked node.
                round_record['tool_calls'].append({
                    'tool': tool_name,
                    'args': tool_input,
                    'result_count': len(exec_result.get('results') or []),
                    'result_ids': [str(c.get('id') or '')[:8]
                                   for c in (exec_result.get('results') or [])
                                   if isinstance(c, dict)],
                    'dropped_below_floor': _dropped_below_floor,
                    'dropped_ids': _dropped_ids,
                    'dropped_seen': len(_dropped_seen_ids),
                    'dropped_seen_ids': _dropped_seen_ids,
                    'latency_ms': exec_result.get('latency_ms', 0),
                    'error': exec_result.get('error'),
                    'dropped_args': exec_result.get('dropped_args'),
                })
                # Compose tool_result message block. format_tool_result_for_haiku
                # renders tool results with the SAME formatter that produced
                # the initial 25 cosine candidates (content + situation +
                # edges + _corrections) — no data-richness asymmetry.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": format_tool_result_for_haiku(
                        exec_result, scope=scope),
                })
        if not tool_results:
            # stop_reason said tool_use but no tool_use block arrived (the
            # May-2026 empty-tool_use Haiku mode). Appending an assistant
            # message with no tool_use + an empty tool_results user message
            # would 400 the next round. Leave `messages` untouched and go
            # around again — the identical request re-reads the cached
            # prefix, and the final round's tool_choice='none' guarantees
            # a JSON finalize.
            brain._log_warning(
                'surface_empty_tool_use',
                'stop_reason=tool_use with no tool_use blocks — retrying '
                'without history append',
                'round=%d session=%s' % (round_idx, session_id))
            tool_trace.append(round_record)
            continue
        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": tool_results})
        tool_trace.append(round_record)

    # Replay-bench capture: the literal loop history (round-1 user content,
    # assistant tool_use blocks, full rendered tool results) — the round-2
    # story that can't be eyeballed from a prompt diff. Normal exits only;
    # an in-loop API-error return leaves rounds empty (visible as such).
    surface_capture.record_rounds(capture, messages=messages,
                                  raw_final=raw_final, tool_trace=tool_trace)
    return raw_final, tool_trace, _telemetry()


def _parse_surfacer_json(raw):
    """Extract the {"selected": [...]} object from Haiku's response.

    Returns the parsed dict, or None if no valid JSON object is found.
    Strips ```-fenced blocks first, then uses raw_decode so trailing prose
    after a valid JSON object doesn't trigger "Extra data" errors.
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip ```…``` fences if present (any language tag)
    if text.startswith("```"):
        # After the first newline, up to the last ``` fence
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()

    decoder = json.JSONDecoder()

    # First: try decoding from the first '{' we see. raw_decode parses
    # the first valid object and returns how far it got — any trailing
    # text is ignored. This is the common case when Haiku adds prose.
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _end = decoder.raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: bracket-slice (handles malformed JSON inside but a clean
    # outer pair of braces). Preserves the prior behavior for edge cases
    # the raw_decode path doesn't cover.
    end = text.rfind("}") + 1
    if end > start:
        candidate = text[start:end]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        # Last resort: Haiku sometimes emits a trailing comma before a
        # closing brace/bracket ('{"selected":[...], }') — observed twice
        # in the 2026-07-18 pool60 build, both payloads otherwise complete
        # (one carried a real selection that was being thrown away).
        # Strip ',\s*}' / ',\s*]' and retry once.
        try:
            obj = json.loads(re.sub(r',\s*([}\]])', r'\1', candidate))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _sanitize_selected_id(raw):
    """Strip whitespace Haiku occasionally injects inside emitted ids.

    Observed (eval run v12_1_full, item d7c942c3-r1): picks emitted as
    '9 9a 2e ' / 'd 6d3 f8' — real candidate ids with spaces inserted.
    Must run BEFORE any [:8] truncation, or the cut keeps the spaces and
    drops real hex chars.
    """
    return ''.join(str(raw or '').split())


def _unique_prefix_match(short_id, candidate_short_ids):
    """Recover a corrupted pick by unique-prefix match against the menu.

    Returns the single candidate short id that starts with `short_id`, or
    None when zero or several match. Requires >= 4 chars so a near-empty
    fragment can't land on a candidate by coincidence. Candidates only —
    never the whole brain — so recovery can't resurrect an archived node
    the menu already excluded.
    """
    if len(short_id) < 4:
        return None
    hits = [cid for cid in candidate_short_ids if cid.startswith(short_id)]
    return hits[0] if len(hits) == 1 else None


def _graph_expand(brain, selected_ids, query_vec=None, prior_vecs=None):
    """Expand the graph from selected seeds via spreading activation
    (surface_contract.spread_activation — depth-emergent breadth-first
    traversal with a median-threshold gate).

    Args:
        brain: Brain instance (spread_activation's conn/aspects, and the
            get_node batch)
        selected_ids: list of seed node IDs (Haiku's picks, at most
            SURFACE['max_selected'])
        query_vec: numpy array (768d) — query embedding. Required for
            meaningful activation; without it, returns empty.
        prior_vecs: prior-turn query embeddings for multi-turn blend.

    Returns dict:
        'node_activation':  {full_node_id: float in [0,1]}
        'field_activation': {full_node_id: {field_name: float}}
        'rich_nodes':       {full_node_id: rich_node_dict} — from get_node batch
        'trace':            per-hop diagnostics
    """
    from servers.scales.s1.surface_contract import spread_activation

    resolved = [sid for sid in (selected_ids or []) if sid]
    if query_vec is None or not resolved:
        return {'node_activation': {}, 'field_activation': {},
                'rich_nodes': {}, 'trace': []}

    result = spread_activation(resolved, query_vec, brain,
                               prior_vecs=prior_vecs)
    node_activation = result['node_activation']
    field_activation = result['field_activation']

    # Batch-load rich node data for everything activated
    all_ids = list(node_activation.keys())
    rich_nodes = brain.get_node(all_ids) if all_ids else {}

    return {
        'node_activation':  node_activation,
        'field_activation': field_activation,
        'rich_nodes':       rich_nodes,
        'trace':            result['trace'],
    }


def _write_traces(brain, ctx, candidates_data, selected_ids,
                  graph_neighbors, additional_context, enriched, results,
                  recall_ref, stamp, session_id, expansion=None,
                  frame='', telemetry=None, pt=None, selection_reason='',
                  seen_dropped=0, judge_pointer=None, picked=None,
                  not_shown=None, redirected=None, also_lit=None):
    """Write S1 surface traces: O (candidates), K (surfaced), Δ (additionalContext).

    `selected_ids` is what the stream SAW — the seeds the render delivered.
    Every reader of the K trace's `selected` / ref_id (the per-turn <shown>
    list and pool dedup, co-access, LAF's surfaced role, the time tool's
    'discussed' anchor, the S2 decoders) means exactly that, so it is the
    fact recorded there. Haiku's decision stays auditable alongside:
    `picked` (its resolved picks before any gate), `not_shown` (every pick
    that did not render, with its reason: unresolvable, archived_no_survivor,
    veil_unavailable, walled, already_shown, no_node, budget), `redirected`
    (archived pick → survivor that rendered instead) and `also_lit`
    (neighbors shown as one title line — in context by title only, so NOT
    part of `selected`). The Δ trace's selection fields stay the picker's
    verdict over its menu (built from `picked`).

    `expansion` carries activation data from spread_activation when present —
    we attach per-node activation values and the kernel's per-hop trace to
    the K-event metadata so dashboards / S3 can see which nodes lit up and
    by how much, not just which were surfaced.

    `telemetry` is the run-cost dict from _call_surface (build_run_telemetry
    kwargs). It's emitted FLAT into the K-event metadata via build_run_telemetry
    — the same shared cost block the encoder delta carries — so Surface's
    input/output tokens + cache + elapsed_ms are queryable from traces, closing
    the long-standing surface cost-telemetry gap. Guarded by
    check_surface_telemetry so it can't silently regress to zeros.

    `pt` (optional PhaseTimer): when supplied, its per-phase breakdown is
    snapshotted into the K trace as `phase_timing` — the structured, queryable
    form of the hook_phase_timing debug string. The K trace also records an
    `outcome` flag (served / empty) so a turn that surfaced nothing is
    distinguishable from one that did, without cross-referencing other logs.

    `selection_reason` (Haiku's recall-level rationale, in practice the
    why-nothing-was-picked note) is the S1Surface journal on the K event —
    the trace is its only durable home; it is never rendered to Anchor.
    """
    recall_chain = ctx.s1r_chain()

    # O: candidates detail
    cand_detail = ['%s|%s|%.2f|%s' % (
        c.get('id', '')[:8], c.get('title', '')[:80],
        c.get('score', 0), c.get('type', ''))
        for c in candidates_data[:25]]

    # K: surfaced detail
    sel_detail = ['%s|%s' % (c.get('id', '')[:8], c.get('title', ''))
                  for c in candidates_data if c.get('id', '')[:8] in selected_ids]
    # Haiku picks resolved OUTSIDE the candidate menu (conversation /
    # <shown> text) are surfaced too — record them, or they never enter
    # the seen-dedup set and ambient re-injection survives through
    # exactly this path (a98143f review, finding 6). Rare (loud-logged
    # as surface_selected_from_context), so the per-id title lookup is fine.
    _in_menu = {c.get('id', '')[:8] for c in candidates_data}
    for _sid in selected_ids:
        if _sid and _sid not in _in_menu:
            try:
                _title = brain._nodes.get_title(_sid) or ''
            except Exception:
                _title = ''
            sel_detail.append('%s|%s' % (_sid, _title))

    # Expanded detail
    exp_detail = ['%s|%s|%s' % (
        nb.get('id', '')[:8], nb.get('title', '')[:60], nb.get('relation', ''))
        for nb in graph_neighbors[:10]]

    # Selection delta metadata — the PICKER's verdict over the menu it saw
    # (trace_links / recall_laf read `dropped` as "offered to Haiku, not
    # picked" — a supervision negative). Built from `picked`, never from the
    # shown set: a pick a gate or the budget removed is not a negative.
    candidate_ids = [c.get('id', '')[:8] for c in candidates_data]
    _picked = set(picked) if picked is not None else set(selected_ids)
    selected_short = sorted(_picked)
    dropped_short = [cid for cid in candidate_ids if cid not in _picked]
    outcomes_per_candidate = {
        cid: ('selected' if cid in _picked else 'dropped')
        for cid in candidate_ids
    }

    # Frame metadata — how much partnership context was injected this turn.
    # Tracking the size + section count gives the dashboard observability into
    # what Anchor's prior actually looked like, without bloating traces with
    # the full Frame text. Empty dict when frame was unavailable (degraded).
    frame_meta = {}
    if frame:
        frame_meta = {
            'frame_chars': len(frame),
            'frame_tokens_est': len(frame) // 4,
            'frame_sections': frame.count('\n## ') + (1 if frame.startswith('## ') else 0),
        }
    else:
        frame_meta = {'frame_chars': 0, 'frame_unavailable': True}

    # Activation metadata — per-node activation values + kernel trace.
    # run_surface always passes _graph_expand's dict (keys present, empty
    # when nothing lit: no picks, no query_vec), so a no-selection turn
    # records activation_count 0 rather than omitting the keys. None only
    # from callers that never expanded.
    activation_meta = {}
    if expansion:
        node_act = expansion.get('node_activation') or {}
        field_act = expansion.get('field_activation') or {}
        kernel_trace = expansion.get('trace') or []

        # Compact: short-id → activation value. Sorted descending for readability.
        activation_meta['activations'] = [
            {'id': nid[:8], 'act': round(act, 3)}
            for nid, act in sorted(node_act.items(), key=lambda x: x[1], reverse=True)
        ][:30]  # cap to 30 for trace size
        activation_meta['activation_count'] = len(node_act)

        # Top-3 per-node field activations (which fields lit up, for render debug)
        top_fields = {}
        for nid, fa in field_act.items():
            if not fa:
                continue
            top3 = sorted(fa.items(), key=lambda x: x[1], reverse=True)[:3]
            top_fields[nid[:8]] = [(f, round(a, 3)) for f, a in top3]
        # Cap at 10 nodes to bound trace size
        activation_meta['top_fields'] = dict(list(top_fields.items())[:10])

        # Kernel trace — hops, new nodes, threshold applied, edges transmitted
        activation_meta['kernel_trace'] = kernel_trace

    # K-event metadata — built as a local so the telemetry guard can inspect it
    # before the write (loud-at-the-write-boundary). The shared run-cost block
    # (build_run_telemetry) sits flat alongside the rich tool_trace/kernel_trace,
    # so Surface now carries BOTH cost and loop detail — the gap this closes.
    from servers.scales.s1.surface_contract import SURFACE_VARIANT
    k_metadata = {
        'selected': sel_detail, 'expanded': exp_detail,
        # The picker's decision and what became of it — logging beside the
        # shown set (see the docstring). `picked` ⊇ shown ∪ not_shown ids,
        # modulo redirects.
        'picked': sorted(picked or []),
        'not_shown': list(not_shown or []),
        'redirected': dict(redirected or {}),
        'also_lit': list(also_lit or []),
        # Seen-dedup observability: how many pool candidates the hook
        # dropped as already-surfaced this window. Without this the filter
        # is write-only telemetry — unverifiable in production, the exact
        # silent-no-op failure the dedup replaced (a98143f review, finding 2).
        'seen_dropped': int(seen_dropped or 0),
        # S1Surface journal (2026-07-11): Haiku's recall-level `reason` —
        # in practice the why-nothing-was-picked note (the prompt asks for
        # it only on empty selections). Not rendered to Anchor; the K trace
        # is its only durable home. Bounded so a runaway can't bloat it.
        'selection_reason': (selection_reason or '')[:500],
        **frame_meta, **activation_meta,
        # Agentic surface tool trace. Stashed by _call_surface on
        # the brain instance so we don't change the run_surface signature;
        # popped here so one recall's record can't
        # ride on the next and the stash never grows.
        'tool_trace': (getattr(brain, '_surface_tool_traces', {}) or {}).pop(session_id, None) or [],
        # Presentation shuffle record (§20.12 A2): shuffle_seed + the exact
        # round-1 menu order Haiku saw (presented_order, 8-char ids).
        # cand_detail in the O trace stays scorer-ordered — propensity
        # analysis joins picked/dropped against presented_order.
        **((getattr(brain, '_surface_presented', {}) or {}).pop(session_id, None) or {}),
        'surface_variant': SURFACE_VARIANT,
        **interaction_fields(stamp),
        # telemetry is already a complete build_run_telemetry dict from the
        # surface loop; spread it flat (fallback to the all-zero block on None).
        **(telemetry or build_run_telemetry()),
        # Per-phase latency (structured, queryable — the hook_phase_timing debug
        # string's data) + the run outcome. 'served' when context reached Anchor,
        # 'empty' when nothing surfaced. ('timeout' is deferred — the daemon
        # can't observe a client abandoning the recall.)
        'phase_timing': pt.snapshot() if pt is not None else [],
        'outcome': 'served' if additional_context else 'empty',
    }
    if judge_pointer:
        # db_dir-relative pointer to this recall's `judge` payload
        # (_record_judge_payload). The dashboard's polled feed reads it
        # O(1) via read_payload_pointer — without it every poll row would
        # glob the payloads tree (and a same-chain collision ordinal would
        # sort wrong). Enrichment only: absent when the kind is gated off.
        k_metadata['payload_pointer'] = judge_pointer
    # Loud guard — Haiku ran but recorded 0 output tokens means the cost
    # telemetry wasn't threaded. Log, don't block (write the full payload
    # regardless), same contract as the encoder check_delta_telemetry.
    _tel_warn = check_surface_telemetry(k_metadata)
    if _tel_warn:
        try:
            brain._log_error('surface_telemetry_missing',
                             RuntimeError(_tel_warn), 'K trace write boundary')
        except Exception:
            pass

    # Batch all three trace writes in one transaction
    brain._trace_dal.append_batch([
        dict(chain_id=recall_chain, scale='s1', event_type='O',
             ref_type='recall', ref_id=str(recall_ref or ''),
             summary='%d candidates for: %s' % (len(results), enriched[:100]),
             metadata={'source': 'hook', 'query': enriched[:500], 'candidates': cand_detail},
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='K',
             ref_type='surface_selected',
             ref_id=json.dumps(list(selected_ids)),
             summary='%d surfaced, %d expanded, %d activated' % (
                 len(selected_ids), len(graph_neighbors),
                 activation_meta.get('activation_count', 0)),
             metadata=k_metadata,
             session_id=session_id),
        dict(chain_id=recall_chain, scale='s1', event_type='delta',
             ref_type='additionalContext',
             summary='%d nodes surfaced' % len(selected_ids) if selected_ids else '(no selection)',
             metadata=build_selection_metadata(
                 candidates_considered=len(results),
                 selected=selected_short,
                 dropped=dropped_short,
                 outcomes_per_candidate=outcomes_per_candidate,
                 content=additional_context or '',
                 interaction=stamp,
                 expanded=exp_detail,
                 query=enriched[:500],
             ),
             interaction_id=(stamp or {}).get('id'),
             session_id=session_id),
    ])


def _record_judge_payload(ctx, recall_ref, surface_prompt, output, brain):
    """Record the surface/judge result as a `judge` payload on the S1R chain
    (docs/TRACE-MODES-DESIGN.md). Returns the pointer (or None when gated off) — the
    caller threads it into the K trace so the dashboard's polled feed reads
    the payload O(1), falling back to a chain-layout scan only on card
    expand. Direct file reads keep it daemon-down safe. Post-selection tail
    position — never on the recall scoring path. record_payload never raises."""
    return brain.record_payload(ctx.s1r_chain(), 'judge', {
        "recall_ref": recall_ref,
        "surface_prompt": surface_prompt,
        "surface_output": output,
    })


def _drop_archived_selected(brain, selected_mode, redirected=None):
    """Resolve Haiku's picks to live nodes, in place.

    An archived pick WITH a survivor becomes the survivor — the memory lives
    on there, and the render marks the redirect (the canonical pull's own
    behavior, applied before seeding so the survivor has vectors to seed
    with). An archived pick with NO survivor is dropped, with an ERROR per
    event — operator mandate: an archived node being picked anywhere must
    be loud, never stat-only. Runs BEFORE seeding and the surface_selected
    trace, so a dead id never seeds spread or re-enters the shown set.

    Mutates selected_mode (full-id keyed). `redirected`, when a dict is
    passed, is filled {picked_full_id: survivor_full_id}. Returns the list
    of dropped full ids.
    """
    if not selected_mode:
        return []
    try:
        walk = brain.resolve_live(list(selected_mode), on_orphan='mark')
    except Exception as e:
        brain._log_error(
            'surface_liveness_gate', e,
            'survivor walk failed — selection passes unfiltered')
        return []
    swaps = walk.get('redirected') or {}
    for old, new in swaps.items():
        mode = selected_mode.pop(old, None)
        if mode is not None and new not in selected_mode:
            selected_mode[new] = mode
        if redirected is not None:
            redirected[old] = new
    if swaps:
        brain._log_warning(
            'surface_selected_redirected',
            'archived pick(s) followed their survivor pointer: %s' % ', '.join(
                '%s→%s' % (o[:8], n[:8]) for o, n in swaps.items()),
            'the id came from session history, not the candidate menu; '
            'the survivor renders with the redirect marker')
    dead = sorted(walk.get('orphans') or [])
    if not dead:
        return []
    for nid in dead:
        selected_mode.pop(nid, None)
    brain._log_error(
        'surface_selected_archived',
        RuntimeError('Haiku selected archived node(s) %s with no survivor — '
                     'dropped before seeding' % ','.join(nid[:8] for nid in dead)),
        'liveness gate in run_surface; the id came from session history '
        '(conversation / recently-surfaced block), not the candidate menu')
    return dead


def _resolve_picks(brain, selected, candidates_data, session_id):
    """Haiku's emitted picks → ({full_id: render_mode}, not_shown entries for
    ids that resolved to nothing).

    An emitted id resolves against the candidate menu (whitespace-sanitized,
    then by unique prefix when corruption left fewer than 8 chars), else
    against the brain by exact id — a real node Haiku saw only in session
    history, admitted loud — with the leading-zero recovery for 7-char
    emissions. Modes come from the contract; anything else renders as the
    default mode.
    """
    from servers.scales.s1.surface_contract import (
        SURFACE_MODES, SURFACE_MODE_DEFAULT)
    # Map short-id → full-id over the WHOLE candidate pool (≤25 entries) —
    # sanitized / prefix-recovered ids below must be able to land on any
    # candidate, not just ones whose raw emitted form matched.
    short_to_full = {}
    for c in candidates_data:
        cid = c.get('id', '')
        if cid:
            short_to_full[cid[:8]] = cid
    selected_mode = {}
    not_shown = []
    for s in selected:
        raw_id = s.get('id', '')
        short_id = _sanitize_selected_id(raw_id)[:8]
        mode = (s.get('mode') or SURFACE_MODE_DEFAULT).strip().lower()
        if mode not in SURFACE_MODES:
            mode = SURFACE_MODE_DEFAULT
        full_id = short_to_full.get(short_id)
        if not full_id and short_id:
            # Whitespace corruption often leaves fewer than 8 real chars
            # ('d 6d3 f8' → 'd6d3f8') — a unique prefix of exactly one
            # candidate is still an unambiguous pick. Recover it rather
            # than dropping the selection.
            recovered = _unique_prefix_match(short_id, short_to_full)
            if recovered:
                full_id = short_to_full[recovered]
                brain._log_warning(
                    'surface_id_fuzzy_recovered',
                    'emitted id %r recovered to candidate %s by unique prefix'
                    % (raw_id, recovered),
                    'session=%s' % session_id)
        if full_id:
            selected_mode[full_id] = mode
            continue
        # Haiku returned an ID not in its candidate menu — either a
        # hallucination or a typo. Check it against the brain by exact
        # id (the ID might be a real node from session context). If
        # found, use it; if not, log loudly so the failure isn't silent.
        try:
            # get_title as a 1-column existence probe (title is NOT NULL;
            # `is not None` so an empty-string title still counts as found)
            ndal = brain._nodes
            resolved = (short_id
                        if ndal.get_title(short_id) is not None else None)
            # Haiku occasionally drops a leading '0' from 8-char IDs,
            # producing a 7-char output ('95c2b96' for '095c2b96') — 2 of 4
            # 'unresolvable' cases in the error log were this. When the
            # short_id is 7 chars and doesn't resolve, retry with '0'
            # prepended — that reconstructs a full 8-char id, still an
            # exact match. If THAT resolves, recover the selection and log
            # it as a leading-zero recovery (distinct from hallucinations).
            if not resolved and len(short_id) == 7:
                _padded = '0' + short_id
                if ndal.get_title(_padded) is not None:
                    resolved = _padded
                    brain._log_error(
                        'haiku_id_leading_zero_recovered',
                        RuntimeError('Haiku dropped leading 0 — recovered'),
                        'short_id=%s recovered_as=%s' % (short_id, resolved[:8]))
        except Exception as _re:
            # A bare except here used to mask real DB errors as "ID is
            # hallucinated" — a SQL/index issue would become
            # indistinguishable from a Haiku confabulation, breaking the
            # diagnostic value of the surface_selected_from_context vs
            # surface_unknown_selected_id distinction below.
            resolved = None
            brain._log_error(
                'haiku_id_resolve_failed', _re,
                'exact-id lookup raised for short_id=%s — treating as '
                'unresolvable but real cause logged' % short_id)
        if resolved:
            selected_mode[resolved] = mode
            # NOT a failure — the pick is kept. Haiku never queries; a real
            # id outside <candidates> came from the prompt's other visible
            # sources: a <shown> element, or a node id named in an earlier
            # conversation turn. `seen_node_ids` strips already-shown ids
            # from the candidate pool on purpose, so a <shown> re-pick is
            # outside-the-menu BY CONSTRUCTION. The two outcomes that do
            # cost something each have their own channel: an id that
            # resolves nowhere (surface_unknown_selected_id) and a re-pick
            # this stream already holds (surface_selected_already_shown,
            # dropped below). This one stays visible as a rate, not an
            # alarm.
            brain._log_warning(
                'surface_selected_from_context',
                'Haiku selected %s from prompt context (<shown> or an '
                'earlier turn) rather than the candidate menu — it '
                'resolves, selection kept' % short_id,
                'short_id=%s resolved=%s' % (short_id, resolved[:12]))
        else:
            # Single loud channel for an id that exists nowhere — the
            # scoreboard's drift section counts this stream, and the
            # dashboard error feed shows warnings alongside errors. A
            # silent drop here is exactly how the v12_1_full
            # empty-context miss went unnoticed.
            brain._log_warning(
                'surface_unknown_selected_id',
                'emitted id %r matches no candidate and resolves to '
                'no node — pick dropped' % raw_id,
                'sanitized=%s session=%s' % (short_id, session_id))
            not_shown.append({'id': (short_id or str(raw_id))[:8],
                              'reason': 'unresolvable'})
    return selected_mode, not_shown


def run_surface(brain, ctx, candidates_data, user_message,
                recent_messages, result, enriched, results, recall_ref,
                session_id, graph_changes, query_vec=None, prior_vecs=None,
                frame='', pt=None):
    """S1 Surface: Haiku-select → gates → spread_activation → seeds-first
    render → trace.

    The complete S1 Surface chain. Called from hook_recall in daemon_hooks.py.
    query_vec + prior_vecs feed the spreading-activation kernel; without them
    no neighbors light and the picks render alone.

    `pt` (optional PhaseTimer): when supplied by hook_recall, surface marks
    its internal phases on the same timer so the daemon log line splits
    `surface_haiku`, `surface_spread`, `surface_render`, `surface_trace`.
    No-op if None — surface still runs, just without the breakdown.

    Returns: additional_context string, or None when nothing rendered.
    """
    load_env()

    def _mark(label):
        if pt is not None:
            pt.mark(label)

    def _seen_dropped():
        stats = result.get('_retrieval_stats') if isinstance(result, dict) else None
        return int((stats or {}).get('seen_dropped', 0) or 0)

    # Differential scope exposure: the session's declared side of every
    # scope dimension, computed ONCE here (run_surface is the pipeline's
    # session boundary) and threaded as plain data everywhere below — never
    # re-derived at depth (a deep brain lookup is a hidden dependency test
    # doubles can't see). None when nothing is declared.
    scope = brain.session_scope(session_id)

    # Call Haiku selector — picks up to SURFACE['max_selected'] from the menu.
    surfaced, surface_prompt, max_tokens, stamp, telemetry = _call_surface(
        brain, candidates_data, user_message, recent_messages,
        session_id, result, frame=frame, scope=scope)
    _mark('surface_haiku')

    selected = surfaced.get("selected", [])
    # Haiku's per-recall rationale. Rendered nowhere (Anchor never sees
    # it) — its one consumer is the S1Surface journal in the K trace.
    selection_reason = surfaced.get("reason") or ''

    # Replay-bench capture stashed by _call_surface — popped (not read) so
    # a failed finish can't leak a stale capture into the next recall.
    capture = (getattr(brain, '_surface_captures', {}) or {}).pop(
        session_id, None)

    # Haiku's picks as {full_id: render_mode}. Every pick that does not
    # reach the stream is recorded in not_shown with its reason, so the K
    # trace's `selected` can mean SHOWN while the picker's decision stays
    # auditable (`picked`, `redirected`).
    selected_mode, not_shown = _resolve_picks(
        brain, selected, candidates_data, session_id)
    picked_short = sorted(fid[:8] for fid in selected_mode)
    redirected = {}

    def _drop(fid, reason):
        selected_mode.pop(fid, None)
        not_shown.append({'id': fid[:8], 'reason': reason})

    # Liveness gate — Haiku's prompt carries node ids in historical text
    # (conversation, <shown> elements) that read-time archived filters can't
    # reach, so a node archived mid-session (S2 absorb) can come back as a
    # selection through the outside-candidates path. Its vectors are gone
    # (deleted at archive), so seeding it yields zero activation, and every
    # acceptance would re-write the id into the shown set — a
    # self-perpetuating loop. Enforce liveness structurally — code beats
    # prompt compliance. A pick with a survivor is redirected to it rather
    # than dropped.
    for _dead in _drop_archived_selected(brain, selected_mode,
                                         redirected=redirected):
        not_shown.append({'id': _dead[:8], 'reason': 'archived_no_survivor'})

    # Scope veil on the SELECTION — same structural stance as the archived
    # gate above (code beats prompt compliance): Haiku can emit an id it saw
    # only as a leaked reference (the out-of-candidate admission path), and
    # a walled id must not become a spread seed or render. Veil failure
    # fails CLOSED for the surfacing (selection purged, turn survives) —
    # scope_veil deliberately raises on a first-build failure, and an
    # unguarded raise here would take the whole turn's traces with it.
    try:
        _veil = brain.scope_veil(session_id)
    except Exception as _veil_err:
        brain._log_error(
            'surface_scope_veil', _veil_err,
            'CRITICAL: veil unavailable — failing CLOSED (selection purged)')
        _veil = None
    if _veil is None:
        for _pid in list(selected_mode):
            _drop(_pid, 'veil_unavailable')
        _veil = frozenset()
    for _wid in [i for i in selected_mode if i in _veil]:
        _drop(_wid, 'walled')
        brain._log_error(
            'surface_selected_walled',
            RuntimeError('walled node %s reached selection — dropped '
                         '(isolation veil)' % _wid[:8]),
            'session=%s' % session_id)

    # Same-session gate — a pick the stream already has in context this
    # window (the shown set the traces record) does not render again. The
    # prompt tells Haiku never to re-pick a <shown> id; this is the code
    # that holds it, loud so the re-pick rate stays visible.
    seen8 = seen_node_ids(recent_messages)
    for _sid in [i for i in selected_mode if i[:8] in seen8]:
        _drop(_sid, 'already_shown')
        brain._log_warning(
            'surface_selected_already_shown',
            'Haiku re-picked %s, shown to this stream earlier in the window '
            '— not rendered again' % _sid[:8],
            'session=%s' % session_id)
    _mark('surface_id_resolve')

    # Graph expansion via spreading activation — what lights around the
    # picks; the render shows it as one title line per neighbor.
    expansion = _graph_expand(
        brain, list(selected_mode), query_vec=query_vec, prior_vecs=prior_vecs)
    rich_nodes = expansion['rich_nodes']

    # Every seed renders, so every seed needs its rich node — spread only
    # fetches what it lit (nothing when query_vec is absent). The rest come
    # through the canonical pull. A redirected pick is fetched by the id
    # Haiku picked: the pull resolves it forward and marks the survivor it
    # returns (REDIRECTED_FROM_KEY) — the one owner of that mark. A seed
    # that still has no node vanished between resolution and here (an
    # idle-time archive racing the recall) and is dropped loud.
    _keys_for = {}   # seed → the ids to fetch it by (its picks, else itself)
    for _old, _new in redirected.items():
        _keys_for.setdefault(_new, []).append(_old)
    for _s in selected_mode:
        if _s not in rich_nodes and _s not in _keys_for:
            _keys_for[_s] = [_s]
    if _keys_for:
        _all_keys = [k for ks in _keys_for.values() for k in ks]
        try:
            _fetched = brain.get_node(_all_keys) or {}
        except Exception as _fe:
            brain._log_error('surface_seed_fetch', _fe,
                             'canonical pull for %d seed(s) failed' % len(_all_keys))
            _fetched = {}
        from servers.contract import REDIRECTED_FROM_KEY
        for _s, _ks in _keys_for.items():
            _hits = [_fetched[k] for k in _ks if _fetched.get(k)]
            if _hits:
                # Two picks absorbed into one survivor: the pull stamps each
                # requested id on its own copy; the seed carries them all.
                _node = _hits[0]
                _marks = [m for h in _hits for m in (h.get(REDIRECTED_FROM_KEY) or [])]
                if len(_marks) > 1:
                    _node[REDIRECTED_FROM_KEY] = list(dict.fromkeys(_marks))
                rich_nodes[_s] = _node
            elif _s not in rich_nodes:
                _drop(_s, 'no_node')
                brain._log_error(
                    'surface_seed_missing',
                    RuntimeError('pick %s resolved but has no node to render'
                                 % _s[:8]),
                    'session=%s' % session_id)

    # The veil gates the EXPANSION OUTPUT in one pass, seeds included: spread
    # walks edges freely, but a walled node's activation, fields and rich
    # payload never reach the render, and every surviving node's edge list
    # is scrubbed — a walled neighbor's title + edge description is a
    # paraphrase of the walled claim.
    if _veil:
        from servers.scopes import scrub_node
        for _k in ('node_activation', 'field_activation', 'rich_nodes'):
            _d = expansion.get(_k)
            if isinstance(_d, dict):
                expansion[_k] = {i: v for i, v in _d.items()
                                 if i not in _veil}
        rich_nodes = expansion['rich_nodes']
        for _n in rich_nodes.values():
            scrub_node(_n, _veil)
    _mark('surface_spread')

    # The conversation behind each pick — the inject names it as the call
    # that opens it (Conversation: get_traces([...])).
    try:
        _refs = brain.get_source_refs_bulk(list(selected_mode))
    except Exception as _re:
        brain._log_error('surface_source_refs', _re,
                         'source_refs pull for the picks failed — line omitted')
        _refs = {}
    for _s in selected_mode:
        if _s in rich_nodes:
            rich_nodes[_s]['source_refs'] = _refs.get(_s, [])

    # Seeds-first render: Haiku's picks in full, lit neighbors as one line
    # each, the window's shown set kept out of the neighbor list.
    from servers.scales.s1.surface_contract import render_surface_inject
    inject = render_surface_inject(
        node_activation=expansion['node_activation'],
        field_activation=expansion['field_activation'],
        rich_nodes=rich_nodes,
        selected_mode=selected_mode,
        query_vec=query_vec,
        brain=brain,
        scope=scope,
        seen_ids=seen8,
    )
    additional_context = inject['text']
    not_shown.extend(inject['not_shown'])
    # What the stream SAW — the K trace's `selected`. The picks are kept in
    # `picked`; the difference is spelled out in `not_shown`/`redirected`.
    shown_short_ids = {fid[:8] for fid in inject['shown']}
    _mark('surface_render')

    # One tail for every turn — picks or none, rendered or all gated — so
    # the K trace has one shape. Judge payload first: its pointer rides the
    # K trace, so the dashboard's polled feed reads it O(1).
    judge_ptr = _record_judge_payload(
        ctx, recall_ref, surface_prompt,
        additional_context or '(no selection)', brain)
    try:
        _write_traces(brain, ctx, candidates_data, shown_short_ids,
                      _activation_to_trace_list(expansion, selected_mode),
                      additional_context, enriched, results,
                      recall_ref, stamp, session_id,
                      expansion=expansion, frame=frame, telemetry=telemetry, pt=pt,
                      selection_reason=selection_reason,
                      seen_dropped=_seen_dropped(),
                      judge_pointer=judge_ptr,
                      picked=picked_short, not_shown=not_shown,
                      redirected={o[:8]: n[:8] for o, n in redirected.items()},
                      also_lit=[i[:8] for i in inject['also_lit']])
    except Exception as e:
        brain._log_error('trace_s1_surface', e, 'S1 surface trace capture')

    # Replay-bench capture — written last, with the post-gate resolution
    # (production's actual picks are the concordance baseline for replay).
    # Empty selections are corpus-worthy too: a prompt candidate that
    # changes WHEN Haiku picks nothing needs them to be judged.
    surface_capture.finish(
        brain, capture, recall_ref=recall_ref, surfaced=surfaced,
        resolved_mode=selected_mode, selection_reason=selection_reason,
        telemetry=telemetry)
    _mark('surface_trace')

    return additional_context or None


def _activation_to_trace_list(expansion, selected_mode):
    """The lit neighbors (seeds excluded), strongest first, in the shape the
    trace writer's `expanded` detail reads: id, title, and the activation
    in the relation slot."""
    rich = expansion.get('rich_nodes') or {}
    return [
        {"id": nid,
         "title": (rich.get(nid) or {}).get('title', ''),
         "relation": "activation=%.2f" % act}
        for nid, act in sorted((expansion.get('node_activation') or {}).items(),
                               key=lambda kv: -kv[1])
        if nid not in selected_mode]
