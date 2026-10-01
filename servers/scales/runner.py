"""Scale runner infrastructure — background-thread lifecycle + the LLM tool loop.

Scale agents run IN-PROCESS on the daemon's brain. The S1 Scribe runs on a
background worker thread so a slow encode never blocks the daemon poll that
launched it — `run_unit_in_background` owns that thread lifecycle (run the unit,
fire on_complete, release the lock); the S2 units run through Brain.run_s2. Each
unit writes through its own `_make_encoder_dispatch` under brain.write_lock. The generic
`run_llm_loop` (the call → tool_use → dispatch loop with prompt caching) lives
here too. Scale-specific logic lives in each scale's module (scales/s1/scribe.py
+ encode.py, scales/s2/*).

Models: the request shape is the same for every current Claude model, so a
unit changes model through its interaction config alone — `model`, plus
`effort` (passed as output_config; absent = the model's API default, which is
`high` on Sonnet 4.6 / Sonnet 5 but `medium` on Opus 5.5, so set it explicitly
when moving a unit to Opus 5.5). Models that think by default return thinking
blocks: assistant_block echoes them back unchanged and response_text skips
them. A safety-classifier refusal raises ModelRefusal. Thinking counts against
max_tokens, so size it for thinking plus the reply; a round that spends the
whole budget before any write fails the run rather than ending it empty.

Opus 5.5, unverified before a switch: it returns prose written ahead of a tool
call as a progress-update thinking block, empty unless the request sets
thinking.display 'updates' (beta), so that prose is absent from round_texts /
final_text; and its thinking blocks are bound to the conversation prefix,
which _roll_cache_breakpoint edits each round (cache markers only) — check
both against the live API before moving a unit to it.
"""

import time
import threading
import json

# Transport policy is daemon-level, not encoder-lane: brain.py builds the
# daemon's shared client with the same bound, so the constant lives in
# brain_constants and both layers import it from there. make_client() below is
# this module's consumer.
from ..brain_constants import (ANTHROPIC_CLIENT_TIMEOUT,
                               ANTHROPIC_CONNECT_TIMEOUT)


class RunLoopError(Exception):
    """run_llm_loop failed mid-run. Carries the actions accumulated before the
    failure (`partial_actions`) so the caller's failure path can persist them —
    without this, the runs that most need forensics (a tool result so large the
    next API call 400s) are exactly the ones that lose their action log.
    `__cause__` holds the original exception."""

    def __init__(self, message, partial_actions=None, msgs=None):
        super().__init__(message)
        self.partial_actions = partial_actions or []
        # The conversation exactly as the model saw it (tool results already
        # capped) at failure time — the encoder-side handler records it as
        # the `failed_run` payload (docs/TRACE-MODES-DESIGN.md).
        self.msgs = msgs or []


class ModelRefusal(RuntimeError):
    """The model declined the request (stop_reason 'refusal' — the safety
    classifiers on Opus 5.x / Fable 5.x). A refused reply carries no tool
    call, so the loop would otherwise end as a clean zero-write run and the
    caller would record the window as handled. Raising sends it down the
    caller's failure path instead: logged, and the work stays pending."""


def raise_on_refusal(response):
    """Raise ModelRefusal when `response` stopped on a refusal."""
    if getattr(response, 'stop_reason', None) == 'refusal':
        details = getattr(response, 'stop_details', None)
        raise ModelRefusal('model refused the request (category=%s)'
                           % getattr(details, 'category', None))


def _effort_kwargs(effort):
    """The request kwargs for an effort level: output_config when set, nothing
    when None (the model's API default, which differs by model)."""
    return {"output_config": {"effort": effort}} if effort else {}


def response_text(response):
    """The reply's prose: every text block, joined in order. Models that think
    by default put a thinking block first, so content[0] is not the text."""
    return "".join(b.text for b in (getattr(response, 'content', None) or [])
                   if getattr(b, 'type', None) == "text")


def assistant_block(block):
    """One response block as it goes back into the conversation. Text and
    tool_use are rebuilt from their fields; every other block (thinking,
    redacted_thinking) is echoed unchanged — models that think by default
    reject or drop an edited block on the next round."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name,
                "input": block.input or {}}
    return block.to_dict()


# The created/revised/archived node-lifecycle split comes from each dispatch
# write handler's authoritative `affected` dict (it knows its op + has the
# brain result, incl. connect_to edges a tool-name heuristic can't see). The
# runner just reads it off the dispatch return — see _dispatch_tool_uses. Edges are not in `affected`;
# they're directional edge_relation_revised traces emitted by the handlers.


# Canonical token-usage telemetry field names — the keys run_llm_loop returns,
# base._call_llm returns, and build_delta_metadata accepts. Defined once so the
# SDK-attribute mapping (read_usage) and the accumulator (sum_usage, reused by
# run_llm_loop, IntegrationUnit._accumulate_run, and the surface loop) read the
# same field set and can't drift.
USAGE_FIELDS = ('input_tokens', 'output_tokens',
                'cache_read_tokens', 'cache_creation_tokens')


def read_usage(response):
    """Map an Anthropic response's `.usage` onto USAGE_FIELDS (all int, 0 when
    the attribute or the whole usage object is absent).

    Single source for the SDK attribute names so run_llm_loop's per-round
    tracking and base._call_llm's single-shot path read them identically — an
    SDK field rename is fixed in one place. `read_usage(None)` returns the
    all-zero dict (the no-response / pre-call telemetry baseline)."""
    usage = getattr(response, 'usage', None)
    return {
        'input_tokens':          getattr(usage, 'input_tokens', 0) or 0,
        'output_tokens':         getattr(usage, 'output_tokens', 0) or 0,
        'cache_read_tokens':     getattr(usage, 'cache_read_input_tokens', 0) or 0,
        'cache_creation_tokens': getattr(usage, 'cache_creation_input_tokens', 0) or 0,
    }


def sum_usage(total, usage):
    """Accumulate one `usage` dict's USAGE_FIELDS into `total`, in place
    (defensive on missing/None), and return `total`.

    The single token-accumulator primitive — one definition reused by
    run_llm_loop's per-round tracking, IntegrationUnit's per-batch run
    accumulation, and the surface agentic loop, so the "sum the four token
    fields" loop exists exactly once. Start `total` from read_usage(None) (the
    all-zero baseline) for a clean dict of the right keys."""
    for f in USAGE_FIELDS:
        total[f] = total.get(f, 0) + (usage.get(f) or 0)
    return total


def make_client():
    """The ONE construction point for the encoder lane's Anthropic client.

    Every encoder execution path (the run_llm_loop callers and run_llm_once)
    builds its client here, so the provider SDK stays behind this module's
    seam — swapping providers means reimplementing this module's internals,
    nothing above it. (Recall-lane sites — surface, query expansion, the
    daemon's shared warm client — have their own lifecycles and sit outside
    this seam.)
    """
    import anthropic
    import httpx
    # max_retries=1, not the SDK default 2: retry policy above this client is
    # the callers' concern — the Scribe's cooldown paces S1E re-runs and the
    # S2 loop encoders wrap rounds in retry_on_transient_api_error — so SDK
    # retries only multiply a blocked read (600s × attempts before the caller
    # sees the exception). One retry keeps the cheap Retry-After path for
    # rate-limit blips, which S1E has no wrapper to recover from.
    return anthropic.Anthropic(
        timeout=httpx.Timeout(ANTHROPIC_CLIENT_TIMEOUT,
                              connect=ANTHROPIC_CONNECT_TIMEOUT),
        max_retries=1)


def run_llm_once(client, model, max_tokens, system_prompt, user_content,
                 effort=None):
    """Single-shot LLM call — the degenerate (no tools, one round) runner entry.

    Five-minute breakpoints cover the system prompt and the full user prompt.
    `user_content` is one string, or a list of strings sent as consecutive
    text blocks: the FIRST block then carries its own breakpoint, so a
    caller that keeps it byte-identical across calls (a menu, a catalog)
    pays for it once per five minutes while the blocks after it change.
    Returns the RAW response text — envelope parsing (extract_json, journal
    harvest) is the caller's/contract's concern, so the text surface stays
    available to both. `effort` is run_llm_loop's: output_config when set,
    omitted (the model's API default) when None.

    Exceptions propagate — failure policy (log-and-skip, retry) belongs to
    the caller, mirroring run_llm_loop. A refusal raises ModelRefusal.

    Returns:
        (raw_text, telemetry): telemetry is {'elapsed_ms', 'stop_reason',
        **USAGE_FIELDS}. stop_reason rides along so single-shot callers can
        surface max_tokens truncation loudly (the loop path checks it in
        _track_usage; without this the single-shot path reported a truncated
        response as a generic parse failure). raw_text is '' when the reply
        has no text block, never an IndexError.
    """
    blocks = [user_content] if isinstance(user_content, str) else list(user_content)
    content = [{"type": "text", "text": b} for b in blocks]
    if len(content) > 1:
        content[0]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
    content[-1]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral", "ttl": "5m"}}],
        messages=[{"role": "user", "content": content}],
        **_effort_kwargs(effort))
    raise_on_refusal(response)
    telemetry = {'elapsed_ms': int((time.time() - t0) * 1000),
                 'stop_reason': getattr(response, 'stop_reason', None),
                 'model': model,   # rides with the usage → build_delta_metadata
                 **read_usage(response)}
    return response_text(response).strip(), telemetry


def _roll_cache_breakpoint(api_messages):
    """Move the rolling cache breakpoint to the newest block of the conversation.

    The three static breakpoints (system 1h, user preamble 1h, user body 5m) all
    sit on the ASSEMBLED prompt. Everything the loop appends after it — assistant
    turns and tool results — sat outside every cached prefix, so each later round
    re-sent it at full input price. Measured on an external install (2026-08-14):
    a 657K-token tool result billed four times while cache_read stayed frozen at
    the round-0 total for the whole run.

    A breakpoint caches the entire prefix up to it, so one marker on the newest
    block turns those re-sends into 0.1× reads. MIGRATED, never accumulated: the
    API allows 4 breakpoints and the static three are permanent. A tail below the
    model's cacheable floor is processed uncached with no error, so the marker is
    always safe to place.
    """
    for msg in api_messages[1:]:          # [0] carries the static user breakpoints
        for block in msg.get('content') or ():
            if isinstance(block, dict):
                block.pop('cache_control', None)
    tail = [b for b in (api_messages[-1].get('content') or ())
            if isinstance(b, dict)]
    if tail:
        tail[-1]['cache_control'] = {"type": "ephemeral", "ttl": "5m"}


def _try_parse_json(text):
    """First-array-then-object bracket scan over one candidate text.
    Returns parsed JSON, or None."""
    # Try array first (most common for batched proposals)
    start = text.find('[')
    if start >= 0:
        end = text.rfind(']') + 1
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    start = text.find('{')
    if start >= 0:
        end = text.rfind('}') + 1
        if end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def extract_json(text):
    """Extract JSON array or object from LLM response text.

    Handles markdown code fences, leading/trailing text.
    Returns parsed JSON (list or dict), or None on failure.

    Every fence is a candidate, then the fence-free remainder, then the raw
    text — so a non-payload fence ahead of or after the payload (a `## Review`
    fence that lost its heading and survived the journal strip) cannot poison
    the response and discard a paid, correct payload as a parse failure.
    """
    candidates = []
    if '```' in text:
        parts = text.split('```')
        for seg in parts[1::2]:          # fence interiors, in order
            seg = seg.strip()
            if seg.startswith('json'):
                seg = seg[4:].strip()
            candidates.append(seg)
        # The fence-free remainder — a bare payload alongside a notes fence.
        candidates.append('\n'.join(parts[0::2]))
    candidates.append(text)
    for candidate in candidates:
        result = _try_parse_json(candidate)
        if result is not None:
            return result
    return None


# Per-batch retry policy for transient API errors. The SDK's built-in
# max_retries handles pre-stream failures (connect refused, 5xx before
# body, rate-limit). It can't retry once a stream has started and stalls
# mid-body (httpx ReadTimeout) — that's what this wrapper covers.
#
# attempts=2 means one retry. First failure => 8s backoff then retry.
# Second failure => give up, batch fails, work moves on. This caps the
# wall-clock cost of a stuck batch at ~2*timeout + backoff and recovers
# the happy-path on transient blips.
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_BASE_S = 8.0


def retry_on_transient_api_error(fn, *, attempts=RETRY_ATTEMPTS,
                                 base_backoff_s=RETRY_BACKOFF_BASE_S,
                                 log_fn=None):
    """Call fn() with retry on transient Anthropic SDK exceptions.

    Retries on: APITimeoutError, APIConnectionError, InternalServerError
    (5xx from Anthropic). Also catches httpx TimeoutException as a safety
    net in case a raw httpx error leaks through streaming.

    Does NOT retry on: BadRequestError, AuthenticationError, PermissionDenied,
    NotFoundError, UnprocessableEntityError, RateLimitError. Those either
    indicate a client bug (retry won't help) or are already handled by the
    SDK's built-in max_retries (rate limit respects Retry-After header).

    Args:
        fn: zero-arg callable that makes the API call
        attempts: total attempts including the first call; 2 = one retry
        base_backoff_s: seconds to wait before first retry; doubles each attempt
        log_fn: optional logger invoked on retry with a one-line message

    Returns: whatever fn() returns

    Raises: the last exception if all attempts fail (transient), or the
        original exception immediately if non-transient
    """
    import anthropic
    try:
        import httpx
        httpx_timeout = (httpx.TimeoutException,)
    except Exception:
        httpx_timeout = ()

    transient = (
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    ) + httpx_timeout

    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            # RunLoopError wraps mid-run exceptions to carry partial_actions —
            # match transience on the CAUSE too, or the wrapper would
            # silently disable retry for every round-≥1 failure (node
            # c98efe35). Non-transient → re-raise.
            # Retrying re-runs the whole loop, including writes that already
            # landed — deliberately: encoder writes are idempotent-enough and
            # a lost batch costs more than a doubled edge.
            if not (isinstance(e, transient)
                    or isinstance(getattr(e, '__cause__', None), transient)):
                raise
            last_err = e
            if i < attempts - 1:
                sleep_s = base_backoff_s * (2 ** i)
                if log_fn:
                    log_fn('transient API error (%s): %s — retrying in %.0fs '
                           '(attempt %d/%d)' % (
                               type(e).__name__, e, sleep_s, i + 2, attempts))
                time.sleep(sleep_s)
    # Exhausted retries — re-raise the last error so callers can log + skip
    raise last_err


def run_unit_in_background(unit, name, lock, on_complete=None, on_release=None):
    """Run an in-process integration unit on a daemon worker thread.

    The unit runs on the daemon's OWN brain and writes through its
    `_make_encoder_dispatch` (direct, under brain.write_lock; vectors via the
    async embed_queue). The background thread keeps a slow encode from blocking
    the daemon poll that launched it. Used by S1 Scribe and any future
    in-process scale unit.

    Owns only thread lifecycle: it runs unit.run(), invokes on_complete with the
    run's write_actions (so callers can gate on "actually wrote material" — e.g.
    the S2 activity counter), releases `lock` in finally, then fires the optional
    on_release() cleanup. lock.release() and on_release() both run on success AND
    crash, so a caller can free per-run state (e.g. a per-session single-flight
    slot) without leaking it when the encode dies. The caller acquires the lock
    and transfers ownership to this thread.
    """
    def _thread_fn():
        t0 = time.time()
        try:
            print("[%s] STARTING" % name, flush=True)
            result = unit.run()
            elapsed_ms = int((time.time() - t0) * 1000)
            if isinstance(result, dict) and result.get('error'):
                # A caught-and-returned failure is NOT completion: skip
                # on_complete so the caller's cooldown/failure state stands
                # (for the Scribe, that's what paces the retry — clearing it
                # here would re-fire a failing session every poll tick).
                print("[%s] FAILED (returned error) after %dms: %s" % (
                    name, elapsed_ms, result['error']), flush=True)
                return
            actions = result.get('actions', 0) if isinstance(result, dict) else 0
            print("[%s] DONE: %d actions in %dms" % (name, actions, elapsed_ms), flush=True)

            if on_complete is not None:
                try:
                    write_actions = (result.get('write_actions', 0)
                                     if isinstance(result, dict) else 0)
                    on_complete(write_actions)
                except Exception as _oce:
                    try:
                        unit.brain._log_error(
                            'scale_runner_on_complete', _oce,
                            '%s on_complete callback failed' % name)
                    except Exception:
                        pass
        except Exception as e:
            elapsed_ms = int((time.time() - t0) * 1000)
            print("[%s] FAILED after %dms: %s" % (name, elapsed_ms, e), flush=True)
            # Background thread crash — the encoder silently stopped producing.
            # Surface it so operators see recurring failures (same as the S2
            # coordinator's crash logging).
            try:
                unit.brain._log_error(
                    'scale_runner_thread_crash', e,
                    '%s thread died after %dms' % (name, elapsed_ms))
            except Exception:
                pass
        finally:
            lock.release()
            if on_release is not None:
                try:
                    on_release()
                except Exception as _re:
                    try:
                        unit.brain._log_error(
                            'scale_runner_on_release', _re,
                            '%s on_release callback failed' % name)
                    except Exception:
                        pass

    threading.Thread(target=_thread_fn, daemon=True, name=name).start()


def run_llm_loop(client, model, max_tokens, max_rounds, system_prompt,
                 user_content, tools, dispatch_fn, log_fn=None,
                 user_preamble=None, get_nodes_config=None,
                 record_round_fn=None, effort=None, deadline_seconds=None,
                 annotate_fn=None):
    """Generic LLM tool loop — call model, process tool_use, dispatch, repeat.

    Used by the loop encoders (S1 Scribe, consolidation, community).
    Scale-specific logic is in what system_prompt, user_content, and tools
    contain — the loop is identical.

    Prompt caching: up to 3 static `cache_control` markers placed by
    stability, plus one rolling marker on the newest block
    (_roll_cache_breakpoint):
      - System (1h TTL): byte-identical across every call within a prompt
        version. Whole eval / long session keeps system warm.
      - User preamble (1h TTL, optional): stable instructions/format
        reminders moved to the START of the user content. When provided,
        creates a cross-call cached block that joins system in the cache.
      - User content body (5m TTL): the dynamic per-cycle content
        (catalog, journal, timeline). Cached within a single run so
        round-2 hits cache reliably.

    Args:
        client: anthropic.Anthropic() instance
        model: Model ID (e.g. 'claude-sonnet-4-6')
        max_tokens: Max output tokens
        max_rounds: Max tool-use rounds before stopping
        system_prompt: System prompt string
        user_content: User content string (the dynamic body)
        tools: List of tool schema dicts
        dispatch_fn: function(cmd, args) -> dict for tool execution
        log_fn: Optional function(msg) for logging
        user_preamble: Optional stable string to prefix the user content with
            its own 1h cache breakpoint. Use for instructions/format that
            don't change per call. None disables (single-block user content).
        get_nodes_config: Optional render_rich_node config dict. When set, the
            caller's `get_nodes` tool results render with THIS config at every
            batch size, over both the count-picked view and the call's own
            rich=true (contract.node_format_for). None = the default view,
            picked by count and lifted by a call's rich=true. Consumers with
            tight token budgets (the community encoder) pass their own lean
            config.

        record_round_fn: Optional callback(round_idx, parts) invoked with the
            LITERAL per-round request pieces (model/effort/system/messages/
            tools) right before each API call. The runner has no brain by
            design — callers pass brain.round_recorder(chain_id), which owns
            the payload shape, the `round_payload` gate, and never raises.
            None = no per-round capture (production normal config anyway
            gates the kind off).

        deadline_seconds: Optional wall-clock ceiling for the whole loop
            (all rounds + SDK retries + the stream fallback). Checked before
            each round and before the fallback re-issue; past it the loop
            raises RuntimeError — the caller's failure path owns the loud
            handling. None (default) = unbounded.
        effort: Optional API effort level ('low'|'medium'|'high'|'xhigh'|'max').
            Passed as output_config={'effort': ...} on every request. None
            (default) omits output_config entirely — the model's API default
            (high on Sonnet 4.6 / Sonnet 5, medium on Opus 5.5).
            Comes from the encoder's interaction `parameters` JSON (the
            K-store), so it's A/B-able per prompt version without code change.
        annotate_fn: Optional callback(tool_name, tool_input, reply) invoked
            with each tool call's dispatch reply before it is rendered, so the
            caller can add information and actions for its own run
            (contract.add_reply_note). tool_input is the dict dispatch
            received, which handlers may normalize in place. Every reply
            renders through brain_mcp.render_reply, the render the MCP proxy
            uses, and the notes it carried ride its action record. A raising
            callback costs that reply its notes, never the run; the failure
            returns in `annotation_errors` for the caller to log. None = the
            reply renders as dispatched.

    Returns:
        dict with: rounds, actions, write_actions, action_details,
                   read_calls, final_text, profile, cache_*_tokens
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)

    WRITE_TOOLS = {
        'remember', 'remember_batch', 'revise', 'revise_batch',
        'connect', 'connect_batch', 'brain_batch',
    }

    t0 = time.time()
    profile = []
    # Token totals as a dict over USAGE_FIELDS, accumulated via the shared
    # sum_usage primitive. read_usage(None) is the all-zero baseline of the
    # right keys; the per-round breakdown derives from per_round_stats.
    usage_total = read_usage(None)

    # Per-round diagnostic data so we can answer "where did r1's 100s go".
    # Each entry: {round, ttft_ms, total_ms} + USAGE_FIELDS (read_usage's keys).
    # ttft_ms isolates server prefill (largely invariant in generation cost)
    # from the actual token-by-token output.
    per_round_stats = []

    def _step(name):
        profile.append((name, int((time.time() - t0) * 1000)))

    truncations = []
    annotation_errors = []   # annotate_fn failures, for the caller to log

    def _track_usage(resp, round_num, ttft_ms=None, total_ms=None):
        u = read_usage(resp)
        sum_usage(usage_total, u)
        per_round_stats.append({'round': round_num, 'ttft_ms': ttft_ms,
                                'total_ms': total_ms, **u})
        if getattr(resp, 'stop_reason', None) == 'max_tokens':
            _log("WARNING: max_tokens hit (round %d, %s/%d output tokens) — response truncated" % (
                round_num, u['output_tokens'], max_tokens))
            truncations.append({
                'round': round_num,
                'output_tokens': u['output_tokens'],
                'max_tokens': max_tokens,
            })

    # BP1 — tools + system cached at 1h TTL. System prompt is byte-identical
    # across every call within a prompt version; 1h TTL keeps a whole eval or
    # a long chat session warm. Writes cost 2× on the first call of the hour;
    # amortizes after ~2 reads.
    system_param = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }]

    # Per-round capture: hands the literal request pieces to the caller's
    # record_round_fn (brain.round_recorder closure) before each API call.
    # The closure owns gating + shape + file writes and never raises, but the
    # runner still guards — capture failure must not break a cycle. The round
    # index comes from the call site (same value _track_usage receives), so
    # captured round labels can never desync from the telemetry rounds.
    def _capture_payload(msgs, round_idx):
        if record_round_fn is None:
            return
        try:
            record_round_fn(round_idx, {
                "model": model,
                "effort": effort,              # None = the model's API default
                "system": system_prompt,       # full text, not a length
                "messages": msgs,              # full, every content block
                "tools": [t.get("name") for t in (tools or [])],
            })
        except Exception as _e:
            (log_fn or (lambda *_a: None))("[capture] round record failed: %s" % _e)

    # Wall-clock deadline for the WHOLE loop (all rounds + retries). The
    # per-request client timeout bounds a single read, but SDK retries × the
    # stream fallback × multi-round accumulation multiplied one stuck encode
    # into a 5.5-hour hang holding the Scribe's single-flight lock (fb78aab9,
    # 2026-07-28). None = unbounded (S2 units keep their own pacing).
    loop_t0 = time.time()

    def _check_deadline(where):
        if deadline_seconds and time.time() - loop_t0 > deadline_seconds:
            raise RuntimeError(
                'run_llm_loop deadline exceeded (%ds) at %s' % (
                    deadline_seconds, where))

    def _create_message(msgs, round_idx):
        """Create message with streaming. Returns (final_message, ttft_ms,
        total_ms). ttft_ms is the time from request issue to the first
        server event — isolates "server prefill / cache lookup / wait"
        from token-by-token generation time. None if iteration fails
        (we fall back to the unmeasured path so a diagnostic glitch
        can't kill an encoder cycle)."""
        _capture_payload(msgs, round_idx)   # literal per-round payload; no-op unless capturing
        request_t0 = time.time()
        ttft_ms = None
        extra = _effort_kwargs(effort)
        try:
            with client.messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=system_param, messages=msgs, tools=tools,
                    **extra) as stream:
                # Iterate raw events so we can timestamp the first server
                # signal. SDK's stream object exposes a synchronous iter
                # that yields RawMessageStreamEvent objects — message_start,
                # content_block_start, content_block_delta, content_block_stop,
                # message_delta, message_stop. We only need the first event
                # to know when output began.
                for _event in stream:
                    if ttft_ms is None:
                        ttft_ms = int((time.time() - request_t0) * 1000)
                final_msg = stream.get_final_message()
        except Exception:
            # Defensive fallback — if event iteration breaks for any reason,
            # re-issue without timing. Costs an extra API call only on bug,
            # never on happy path. Better than nuking the encoder cycle.
            # Past the deadline, re-raise instead of doubling down — the
            # fallback re-issue is one of the hang multipliers.
            _check_deadline('stream fallback')
            with client.messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=system_param, messages=msgs, tools=tools,
                    **extra) as stream:
                final_msg = stream.get_final_message()
            ttft_ms = None
        total_ms = int((time.time() - request_t0) * 1000)
        return final_msg, ttft_ms, total_ms

    # User content. When `user_preamble` is provided, it forms a stable 1h
    # block that joins system in the cache (cross-call reuse). The dynamic
    # `user_content` body keeps a 5m breakpoint so within a single
    # run_llm_loop, round-2 always reads from cache.
    user_blocks = []
    if user_preamble:
        # BP-stable-user — 1h cached, byte-identical across calls
        user_blocks.append({
            "type": "text",
            "text": user_preamble,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        })
    # BP-dynamic-user — 5m cached, the per-cycle content
    user_blocks.append({
        "type": "text",
        "text": user_content,
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    })
    api_messages = [{"role": "user", "content": user_blocks}]
    response, ttft_ms, total_ms = _create_message(api_messages, 0)
    _track_usage(response, 0, ttft_ms=ttft_ms, total_ms=total_ms)
    raise_on_refusal(response)
    _step("llm_r0")

    actions = []
    rounds = 0
    # Every reply's prose, in order — the encoder's plan (its lists) rides as
    # text beside the tool_use blocks, and the eval scores it against the ops.
    round_texts = []
    last_seen = None

    def _dispatch_tool_uses(response_obj):
        """Dispatch every tool_use block in response. Append to actions.
        Returns a tool_results list usable as a user-role message block.
        """
        tool_uses = [b for b in response_obj.content if b.type == "tool_use"]
        tool_results = []
        for tu in tool_uses:
            _tu_t0 = time.time()
            result = dispatch_fn(tu.name, tu.input)
            latency_ms = int((time.time() - _tu_t0) * 1000)
            error = None if result.get("ok") else result.get("error", "Unknown")
            if annotate_fn is not None:
                # A failed annotation costs its notes, never the cycle (the
                # writes already landed). The runner has no brain, so the
                # failure returns to the caller to log (like truncations).
                try:
                    annotate_fn(tu.name, tu.input, result)
                except Exception as _e:
                    _log("[annotate] %s reply annotation failed: %s" % (tu.name, _e))
                    annotation_errors.append({"tool": tu.name, "error": repr(_e)})
            # render_reply's two halves, composed here so the size cap below
            # can bound each: the body, then the reply's notes.
            from servers import brain_mcp
            from servers.contract import reply_notes
            notes = reply_notes(result)
            # `rich` is the call's own render opt-in, honored as the MCP proxy
            # honors it; for get_nodes a declared get_nodes_config still wins
            # (contract.node_format_for).
            body = brain_mcp.render_reply_body(
                tu.name, result, rich=bool(tu.input.get("rich", False)),
                get_nodes_config=get_nodes_config, error=error)
            notes_text = brain_mcp.render_reply_notes(notes)
            result_text = body + notes_text
            result_chars = len(result_text)
            # Oversized-result guard (trace-mode caps): one brain_batch
            # result of ~6M chars pushed the next request past the API's 1M
            # token cap and 400-killed the run (2026-07-31). Truncate before
            # the conversation sees it, dump the full payload for forensics,
            # and log loud — a result this size is always a bug upstream.
            from servers.trace_contract import trace_detail
            _td = trace_detail()
            cap = _td['tool_result_cap']
            result_truncated = result_chars > cap
            if result_truncated:
                # The bounded head + result_chars on the action record
                # (below) are the forensics that ride the trace substrate.
                # The truncated form also appears verbatim in the next
                # round_payload capture (record_round_fn); a separate
                # tool_result kind for the UNtruncated content has no call
                # site yet — the runner has no brain, so if it's ever wired
                # it rides a caller closure, never a file dump here.
                _log("OVERSIZED tool result: %s returned %d chars (cap %d) — "
                     "truncated" % (tu.name, result_chars, cap))
                # The notes are addressed to this reader, so they keep their
                # room inside the cap and the body takes the rest; notes
                # longer than the whole cap are cut too.
                notes_room = min(len(notes_text), cap)
                if notes_room < len(notes_text):
                    notes_text = (notes_text[:notes_room]
                                  + "\n\n[TRUNCATED by runner: the notes were "
                                    "%d chars.]" % len(notes_text))
                result_text = (
                    body[:cap - notes_room]
                    + "\n\n[TRUNCATED by runner: result was %d chars, cap is %d. "
                      "Do NOT retry this call — proceed with what is shown.]"
                    % (result_chars, cap)
                    + notes_text)
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id,
                "content": result_text,
            })
            # `or` chain + str(), not .get defaults: a model can emit an
            # explicit {"title": null} or a non-string ({"title": 123}) —
            # either would TypeError at [:60] and kill the whole run for a
            # log nicety.
            action_summary = str(tu.input.get("title")
                                 or tu.input.get("query")
                                 or tu.input.get("node_id") or "")[:60]
            result_ids = []
            # Authoritative node-lifecycle split — the dispatch handler returned
            # it as a TOP-LEVEL `affected` (sibling of `result`), computed where
            # the op + brain result are both known. build_delta_metadata
            # aggregates created/revised/archived over these; S2 reads them.
            affected = result.get("affected") if result.get("ok") else None
            affected = affected if isinstance(affected, dict) else {}
            if result.get("ok"):
                r = result.get("result", {})
                if isinstance(r, dict):
                    if r.get("id"):
                        result_ids.append(r["id"])
                    for item in r.get("results", []):
                        if isinstance(item, dict):
                            if item.get("id"):
                                result_ids.append(item["id"])
                            elif isinstance(item.get("result"), dict) and item["result"].get("id"):
                                result_ids.append(item["result"]["id"])
                elif isinstance(r, list):
                    for item in r:
                        if isinstance(item, dict) and item.get("id"):
                            result_ids.append(item["id"])
            # Per-tool result_count — how many result items the call returned
            # (read hits; write op-results). Self-contained count off the result
            # shape; doesn't touch the result_ids parse above. recall_batch
            # returns a list of {query, results} groups, so sum the nested hits
            # rather than count queries; a flat list (get_nodes) counts itself.
            result_count = 0
            if result.get("ok"):
                _r = result.get("result")
                if isinstance(_r, dict):
                    _rl = _r.get("results")
                    result_count = len(_rl) if isinstance(_rl, list) else (1 if _r.get("id") else 0)
                elif isinstance(_r, list):
                    if _r and all(isinstance(x, dict) and isinstance(x.get("results"), list)
                                  for x in _r):
                        result_count = sum(len(x["results"]) for x in _r)
                    else:
                        result_count = len(_r)
            # Per-tool detail (latency_ms/result_count/error) rides on every
            # action record, so action_details (writes) + read_calls (reads) —
            # already threaded into build_delta_metadata by the run_llm_loop
            # encoders — gain the same per-call observability Surface's
            # tool_trace has. `input` is the args.
            action_rec = {"tool": tu.name, "summary": action_summary,
                          "node_ids": result_ids,
                          "created": affected.get("created") or [],
                          "revised": affected.get("revised") or [],
                          "archived": affected.get("archived") or [],
                          "input": tu.input,
                          "latency_ms": latency_ms,
                          "result_count": result_count,
                          "result_chars": result_chars,
                          "error": error}
            # Refused write ops — reported by dispatch beside `affected`
            # (dispatch_common.failed_write_ops); a batch answers ok=True with
            # ok=False op rows, so `error` stays None for it.
            # build_delta_metadata reads them to tell a refusal the run
            # repaired from one it left lost.
            if result.get("failed_ops"):
                action_rec["failed_ops"] = result["failed_ops"]
            # What the reader was told beside the result, each entry bounded
            # like result_head, so the run trace shows the guidance it got.
            if notes:
                action_rec["notes"] = {k: [e[:_td['result_head_cap']] for e in v]
                                       for k, v in notes.items()}
            if result_truncated:
                # Oversized result — bounded head rides the action record so
                # the delta trace shows WHAT the content was, and the
                # encoder-side loud scan can log it to the errors table
                # (the runner has no brain reference).
                action_rec["result_truncated"] = True
                action_rec["result_head"] = \
                    result_text[:_td['result_head_cap']]
            actions.append(action_rec)
            _log("  [%s] %s" % (tu.name, action_summary))
        return tool_results, tool_uses

    try:
        for rounds in range(max_rounds):
            round_texts.append(response_text(response))
            last_seen = response
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # A reply cut off by max_tokens before anything was written is
                # an empty run, not a finished one — a thinking model can
                # spend the whole budget before its tool call. Fail it so the
                # caller retries instead of recording the work as done. Once
                # writes have landed, a truncated wrap-up stays a logged
                # truncation (re-running would duplicate those writes).
                if (getattr(response, 'stop_reason', None) == 'max_tokens'
                        and not any(a['tool'] in WRITE_TOOLS for a in actions)):
                    raise RuntimeError(
                        'max_tokens (%d) reached before any write — the run '
                        'produced nothing' % max_tokens)
                break

            _check_deadline('round %d' % (rounds + 1))
            # Append the assistant turn BEFORE dispatching its tool calls:
            # if dispatch raises, RunLoopError.msgs (→ the failed_run
            # payload) must show the tool_use that died, not a conversation
            # ending cleanly one round earlier.
            api_messages.append({"role": "assistant", "content": [
                assistant_block(b) for b in response.content]})
            tool_results, _ = _dispatch_tool_uses(response)
            api_messages.append({"role": "user", "content": tool_results})
            _roll_cache_breakpoint(api_messages)
            response, ttft_ms, total_ms = _create_message(api_messages, rounds + 1)
            _track_usage(response, rounds + 1, ttft_ms=ttft_ms, total_ms=total_ms)
            raise_on_refusal(response)
            _step("llm_r%d" % (rounds + 1))
    except Exception as e:
        # Mid-run failure loses the actions accumulated so far — exactly the
        # forensics the failure path needs (the 1M-token 400s left no record
        # of which ops ran). Re-raise wrapped, actions attached; __cause__
        # keeps the original for callers matching on exception type/message.
        # The original type name rides the message — error logs and trace
        # summaries grep by anthropic class names, not by the wrapper.
        raise RunLoopError('%s: %s' % (type(e).__name__, e),
                           partial_actions=actions,
                           msgs=api_messages) from e

    final_text = response_text(response)
    undispatched = []
    if response is not last_seen:   # max_rounds exhausted: the last reply never reached the loop head
        round_texts.append(final_text)
        # Its tool calls never run. The runner has no brain, so it returns
        # them and the caller logs the loss (same path as truncations).
        undispatched = [b.name for b in response.content if b.type == "tool_use"]
        if undispatched:
            _log("WARNING: max_rounds (%d) exhausted — tool call(s) in the "
                 "last reply were not run: %s" % (max_rounds, ', '.join(undispatched)))
    write_actions = [a for a in actions if a['tool'] in WRITE_TOOLS]
    read_calls = [a for a in actions if a['tool'] not in WRITE_TOOLS]

    # Total "billed as fresh input" = uncached input. cache_read_input_tokens
    # is read from cache at 0.1× cost; cache_creation_input_tokens is written
    # at 1.25× (5min TTL) or 2× (1h TTL) cost. Report all three so the
    # operator can see hit ratio over many runs.
    total_cache_pool = usage_total['cache_creation_tokens'] + usage_total['cache_read_tokens']
    hit_rate = (usage_total['cache_read_tokens'] / total_cache_pool * 100) if total_cache_pool else 0.0
    _log("Rounds: %d | Actions: %d (writes: %d, reads: %d) | Tokens: %d fresh / %d cached-read / %d cached-write / %d out | hit=%.0f%% | Profile: %s" % (
        rounds + 1, len(actions), len(write_actions), len(read_calls),
        usage_total['input_tokens'], usage_total['cache_read_tokens'],
        usage_total['cache_creation_tokens'], usage_total['output_tokens'],
        hit_rate,
        ', '.join('%s=%dms' % (n, t) for n, t in profile)))

    # Per-round breakdown — answers "where did r1's wall-clock go?"
    # Format: r0 ttft=12000ms gen=75798ms out=4521tok in=6432 cr=28135 cw=28135
    # ttft separates server prefill from generation. If gen >> output/50tok-per-sec,
    # there's slow generation; if ttft >> gen, prefill dominates (large context).
    for rd in per_round_stats:
        ttft = rd.get('ttft_ms')
        tot = rd.get('total_ms') or 0
        gen = (tot - ttft) if (ttft is not None and tot) else None
        out_tok = rd.get('output_tokens') or 0
        gen_rate = (out_tok * 1000.0 / gen) if (gen and gen > 0) else None
        _log("  r%d ttft=%s gen=%s out=%dtok rate=%s in=%d cr=%d cw=%d" % (
            rd['round'],
            '%dms' % ttft if ttft is not None else '?',
            '%dms' % gen if gen is not None else '?',
            out_tok,
            ('%.0ftok/s' % gen_rate) if gen_rate is not None else '?',
            rd.get('input_tokens') or 0,
            rd.get('cache_read_tokens') or 0,
            rd.get('cache_creation_tokens') or 0))

    return {
        "rounds": rounds + 1,
        "actions": len(actions),
        "write_actions": len(write_actions),
        "action_details": write_actions,
        "read_calls": read_calls,
        # Full text — build_delta_metadata caps the TRACE loudly; a cut
        # here would silence that marker.
        "final_text": final_text or '',
        "round_texts": round_texts,
        "profile": profile,
        "elapsed_ms": int((time.time() - t0) * 1000),
        # USAGE_FIELDS keys match the four token return keys exactly.
        **usage_total,
        "model": model,   # which LLM the loop called — threaded into the delta
        "truncations": truncations,
        "undispatched_tool_calls": undispatched,
        "annotation_errors": annotation_errors,
    }
