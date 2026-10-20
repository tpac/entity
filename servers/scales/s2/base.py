"""Integration Unit — universal contract for O/K/Δ units at any scale.

Every integration unit declares what it reads (O_SOURCES, K_SOURCES),
implements run() which produces Δ, and writes its own traces.

The contract is what it reads and what it produces, not how it's
structured internally. Units are free to organize their run() however
makes sense for the operation.

## Shared S2 methods

S2 units share common infrastructure for:
- Reading S1 traces (windowed lookback)
- Deciding whether to run (_last_run_timestamp / _has_new_traces)
- Calling LLM with learnable prompts from interactions table
  (transport, retry, and JSON envelope parsing live behind the
  runner seam — scales/runner.py)

These live here so every S2 unit (aspects, community, healer,
consolidation) uses the same patterns. An interaction override tunes any
unit's LLM behavior without a code change.
"""

import os
import time
from datetime import datetime, timezone, timedelta


# read_usage / sum_usage: the canonical token-telemetry contract shared with
# run_llm_loop, so _call_llm and _accumulate_run don't re-derive it. read_usage
# + sum_usage are also re-exported here for the encoder subclasses (e.g. healer).
# Provider mechanics (client construction, single-shot call, retry, JSON
# envelope parsing) live behind the runner seam — see scales/runner.py.
from ..runner import (read_usage, sum_usage,  # noqa: F401 — re-export for callers
                      make_client, run_llm_once, extract_json)
from ..dispatch import (ATTRIBUTED_WRITE_COMMANDS, OPERATOR_ONLY_COMMANDS,
                        stamp_scope_provenance)


# Writes whose handlers emit chain-bearing traces (node_revised /
# edge_relation_revised). They must carry the unit's run chain_id so those
# traces join the run's chain instead of dispatch_write's date-based
# '{scale}-{date}-revise' fallback — the phantom "S2 revise" / "S1 revise" unit.
# Derived from the attributed writes + revise_edge (one source, so a new
# attributed write becomes chain-aware for free — the house pattern).
CHAIN_AWARE_WRITES = ATTRIBUTED_WRITE_COMMANDS | {'revise_edge'}


def apply_encoder_attribution(cmd, cmd_args, *, encoding_source, run_chain_id,
                              scope=None):
    """Stamp scale-agent attribution onto an outgoing write's args — the single
    chokepoint these facts flow through:

    - ``encoding_source`` on attributed writes (who minted the node/edge),
    - ``run_chain_id`` on chain-bearing writes (which run's chain its trace
      joins),
    - ``scope`` provenance policy (stamp_scope_provenance semantics —
      {field: policy} over SCOPE_PROVENANCE_FIELDS): the Scribe passes its
      session's values ('' per field when the session has none) so
      node-creating ops carry deterministic provenance; S2 units pass ''
      (graph-scope work never invents provenance); None leaves args untouched.

    ``encoding_source`` is force-stamped like the scope fields — attribution
    is run-derived, never agent-authored (an LLM-authored value in a tool
    call's args must not win over the unit's identity; a differing supplied
    value is overridden with a warning). ``chain_id`` keeps ``setdefault``
    so an explicitly-supplied chain wins. Mutates ``cmd_args`` in place; a
    no-op on non-dict args and on reads (get_nodes / recall_batch are in
    neither set — they carry no attribution, and a chain_id on a read is
    meaningless). The in-process encoder dispatch is the one factory S1
    Scribe and the S2 units share, so the attribution rules live here, once.

    Returns: warning strings from the attribution + scope stamps
    (dropped/overridden agent-supplied values) for the caller to log.
    """
    if not isinstance(cmd_args, dict):
        return []
    warnings = []
    if cmd in ATTRIBUTED_WRITE_COMMANDS:
        supplied = cmd_args.get('encoding_source')
        if supplied is not None and supplied != encoding_source:
            warnings.append(
                "encoding_source %r overridden to %r on %s (system-stamped, "
                "never agent-authored)" % (supplied, encoding_source, cmd))
        cmd_args['encoding_source'] = encoding_source
    if run_chain_id and cmd in CHAIN_AWARE_WRITES:
        cmd_args.setdefault('chain_id', run_chain_id)
    return warnings + stamp_scope_provenance(cmd, cmd_args, scope)


class IntegrationUnit:
    """Base contract for all integration units at any scale."""

    # Subclass must define:
    NAME = ''                    # e.g. 'community_detection'
    SCALE = ''                   # e.g. 's2'
    ENCODING_SOURCE = ''         # e.g. 's2:community_detection'

    # O/K contract — what this unit reads and what shapes its decisions
    O_SOURCES = []               # e.g. ['graph_nodes', 'graph_edges']
    K_SOURCES = []               # e.g. ['leidenalg', 'resolution_param']

    # True when the S2 cycle was requested (s2_force, an eval's final flush)
    # rather than scheduled by the maintenance poll; the coordinator sets it.
    # A unit that batches work to save calls runs whatever is waiting then.
    forced = False

    def __init__(self, brain, dispatch_fn=None):
        """Initialize with brain instance and optional dispatch.

        Args:
            brain: Brain instance — the daemon's brain. Units run in-process
                   with direct DB access (writes serialize under
                   brain.write_lock via _make_encoder_dispatch).
            dispatch_fn: Optional executor for the unit's tool calls (evals,
                         tests). _make_encoder_dispatch wraps it with the
                         unit's rules in place of dispatch_command; None (the
                         norm) executes through dispatch_command under
                         brain.write_lock.
        """
        self.brain = brain
        self.dispatch = dispatch_fn

    def run(self):
        """Execute the unit. Returns a result dict describing the delta.

        Must be implemented by subclass. The result dict should include
        at minimum {actions: int}, plus `error` or `skipped` when they apply —
        the coordinator reads those two for its failure policy.
        """
        raise NotImplementedError('%s.run() not implemented' % type(self).__name__)

    def chain_id(self):
        """Generate trace chain ID for this run.

        Format: {scale}-{YYYYMMDDHHMMSS}-{name}
        S2 chains are time-based (not session-based) and stamped ONCE per
        run, then cached: every trace() in this run shares one chain_id,
        while two runs of the same unit get distinct ids. Seconds (not just
        date) make notes groupable per-run — date-only collapsed every
        same-day run of a unit onto one id, breaking "last K runs"
        continuity. A unit can't run twice within one second (min-interval
        gating is in minutes), so seconds is collision-free.

        Two parts are LOAD-BEARING — do not split the timestamp into its own
        dash-segment: `_last_run_timestamp` suffix-matches `LIKE '%-{name}'`
        and the dashboard's `_unit_slug_from_chain` reads `split('-', 2)[2]`
        as the unit slug. One combined timestamp segment + trailing `-{name}`
        keeps both intact.
        """
        if not getattr(self, '_chain_id', None):
            self._chain_id = '%s-%s-%s' % (
                self.SCALE,
                datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'),  # clock-ok — S2 idle-cycle chain id; UTC to align with created_at + avoid DST-rollback second repeats
                self.NAME)
        return self._chain_id

    def scope_policy(self):
        """Scope-provenance policy for this unit's writes (see
        stamp_scope_provenance — {field: policy}). S2 units work at graph
        scope — they never invent provenance, so the default is '' per field
        (authoritative strip: any agent-supplied value is dropped).
        Session-scoped units (S1 Scribe) override with their session's
        derived values."""
        from ..dispatch import SCOPE_PROVENANCE_FIELDS
        return {f: '' for f in SCOPE_PROVENANCE_FIELDS}

    def trace(self, event_type, ref_type, summary, ref_id='', metadata=None):
        """Write a trace event for this unit's current run.

        Uses direct TraceDAL when no dispatch was injected (the norm).
        Uses dispatch('trace_append', ...) when a dispatch_fn was injected.
        """
        # Loud-at-the-write-boundary: an LLM-encoder delta that ran + did work
        # but recorded no output tokens means telemetry wasn't threaded into the
        # delta. Every S2 unit writes its delta through here, so a NEW unit that
        # forgets telemetry is caught for free — the recurrence guard for the
        # 2026-06-24 S2 telemetry gap. The detector is the pure contract function
        # (no logger); logging lives here, where the brain is reachable (the
        # other chokepoint, TraceDAL.append, can't write the errors table
        # mid-append without risking the brain_batch commit).
        if event_type == 'delta' and metadata is not None:
            from servers.trace_contract import check_delta_telemetry
            warn = check_delta_telemetry(ref_type, metadata)
            if warn:
                try:
                    self.brain._log_error(
                        's2_%s_telemetry_gap' % self.NAME, ValueError(warn),
                        'delta trace missing LLM telemetry')
                except Exception:
                    pass

        trace_data = {
            'chain_id': self.chain_id(),
            'scale': self.SCALE,
            'event_type': event_type,
            'ref_type': ref_type,
            'ref_id': ref_id,
            'summary': summary[:200] if summary else '',
            'metadata': metadata,
            'session_id': '',
        }

        if self.dispatch:
            self.dispatch('trace_append', trace_data)
        else:
            self.brain._trace_dal.append(**trace_data)

    # ── Shared S2 infrastructure ──

    @property
    def journal(self):
        """This unit's binding to the journal component — continuity read,
        instruction decoration, residue harvest, in one object. Scoped by
        (scale, unit NAME), cached per unit instance. See scales/journal.py.
        """
        if getattr(self, '_journal_binding', None) is None:
            from ..journal import JournalBinding
            self._journal_binding = JournalBinding(
                self.brain, scale=self.SCALE, source=self.ENCODING_SOURCE,
                **self._journal_kwargs())
        return self._journal_binding

    def _journal_kwargs(self):
        """The binding's identity beyond scale and source — an S2 unit is
        scoped by its NAME; the Scribe overrides with its session (+ arc)."""
        return {'unit': self.NAME}

    def _inject_edge_aspects(self, system_prompt):
        """Append the edge-relation aspect vocabulary so the encoder picks
        specific relations over generic ones. Shared single source across
        S2/S1 encoders — same tail-append pattern as the journal blocks. The block
        (skip set + heading + render) lives in servers.aspects; here we just
        feed it brain.aspects and this brain's relation counts (each aspect
        shows its most-used relations) and append. No-op append when there's
        nothing to show.
        """
        from servers.aspects import render_edge_aspects_block
        block = render_edge_aspects_block(self.brain.aspects.all(),
                                          self.brain._graph.relation_counts())
        return system_prompt.rstrip() + ('\n\n' + block if block else '')

    def _make_encoder_dispatch(self, archive_guard=None):
        """Build the dispatch function S2 encoders use for brain_batch calls.

        Shared by S1 Scribe and the S2 encoders. Sets
        `encoding_source` at the top level (handler cascades to each op).
        Vectors are filled asynchronously by the embed_queue worker, so the
        encoder never embeds inline.

        Args:
            archive_guard: Optional set of node IDs that archive ops are
                allowed to target this batch. When set, archives of nodes
                outside the set are dropped + logged. Used by consolidation
                to prevent the encoder archiving nodes outside its cluster.
                None → no guard.

        An injected dispatch_fn (evals, tests) is WRAPPED, not returned: the
        rules below are the unit's, whatever executes its tool calls, so an
        eval arm writes with the identity and attribution production stamps.
        Only the final call goes to it, and it owns its own locking. (The
        unit's own trace() rows go to it directly — they carry no
        attribution.)

        Returns: dispatch(cmd, cmd_args) → handler result.
        """
        injected = self.dispatch

        from servers.daemon_dispatch import dispatch_command

        brain = self.brain
        encoding_source = self.ENCODING_SOURCE
        unit_name = self.NAME
        # The run chain — s1e-{session}-{stop} for the Scribe, s2-{ts}-{unit}
        # for S2 units — stamped on writes so their revise/edge traces join THIS
        # run instead of the date-fallback chain. Computed once; chain_id() is
        # cached on the unit.
        run_chain_id = self.chain_id()
        # Scope provenance policy — the Scribe stamps its session's values,
        # S2 units strip ('' — graph-scope work never invents provenance).
        # Computed once per dispatch build.
        scope_policy = self.scope_policy()
        # Session identity for READS: a session-scoped unit's recalls must
        # be veiled as ITS session (an isolated session's Scribe needs the
        # inward veil to see its own walled nodes; without this the ''
        # fallback gives the outward-only veil and the encoder goes blind
        # to the very project it's encoding). S2 units have no session —
        # their recall-based reads get the sessionless outward veil.
        unit_session = getattr(self, 'session_id', '') or ''

        def dispatch(cmd, cmd_args):
            # Channel refusal for operator-only commands (the declared set in
            # scales/dispatch.py). Defense-in-depth: the owner method also
            # refuses non-anchor encoding_source at the write boundary — this
            # closure is one channel, not the whole gate.
            if cmd in OPERATOR_ONLY_COMMANDS:
                brain._log_error(
                    's2_%s_lock_refused' % unit_name,
                    ValueError('%s is operator-channel only' % cmd),
                    'encoder dispatch refused %s (unit=%s)' % (cmd, unit_name))
                return {'ok': False,
                        'error': '%s is operator-channel only' % cmd}
            if isinstance(cmd_args, dict):
                from ...dispatch_common import CALLER_SESSION_KEY
                # Assigned, never defaulted: identity is the unit's, so a stamp
                # or session_id the model wrote into its tool call claims
                # nothing — and an S2 unit's '' stays sessionless.
                cmd_args[CALLER_SESSION_KEY] = unit_session
            _scope_warnings = apply_encoder_attribution(
                cmd, cmd_args,
                encoding_source=encoding_source, run_chain_id=run_chain_id,
                scope=scope_policy)
            for _w in _scope_warnings:
                # encoder drift, not a failure — warning severity so the
                # error feed stays real errors only
                brain._log_warning('scope_provenance_stamp', _w,
                                   'unit=%s' % unit_name)

            if cmd == 'brain_batch' and isinstance(cmd_args, dict):
                if archive_guard is not None:
                    surviving_ops = []
                    for op in cmd_args.get('operations', []):
                        if isinstance(op, dict) and op.get('op') == 'archive':
                            nid = op.get('node_id') or op.get('id')
                            if nid and nid not in archive_guard:
                                brain._log_error(
                                    's2_%s_out_of_scope_archive' % unit_name,
                                    ValueError(
                                        'archive op for %s not in allowed set'
                                        % nid),
                                    'allowed=%d ids; rejected archive (encoder drift)'
                                    % len(archive_guard))
                                continue
                        surviving_ops.append(op)
                    cmd_args['operations'] = surviving_ops

            if injected is not None:
                return injected(cmd, cmd_args)
            # Serialize against daemon dispatch + autosave + embed_queue. S2
            # runs in-process but on a pool worker thread; without this, encoder
            # writes can interleave with concurrent client writes (and with each
            # other across S2 units). dispatch_command runs INSIDE the lock —
            # its brain.db writes need the graph serializer; logs writes it
            # triggers serialize themselves inside the DAL write boundary
            # (logs_write_lock nests under write_lock, leaf ordering).
            with brain.write_lock:
                return dispatch_command(brain, cmd, cmd_args, [])

        return dispatch

    def _last_run_timestamp(self):
        """Find this unit's most recent completed run timestamp.

        Returns ISO timestamp string, or '' if never run. Counts only REAL
        integration deltas. Two exclusions, same reason — both share this run's
        chain_id + event_type='delta' without being the unit's per-RUN delta:
        residue (journal_note), so a run that only journaled doesn't read as a
        completed integration; and per-mutation rows (EMITTER_REF_TYPES), so a
        single node revise doesn't. Safe because every unit stamps its own delta
        ref_type on every exit path, early-outs included.

        Goes through the trace API (query_traces), not raw SQL; chain_suffix
        scopes to this unit's chains and escapes the '_' in names like
        community_detection.
        """
        from ...trace_contract import RESIDUE_REF_TYPES, EMITTER_REF_TYPES
        try:
            res = self.brain.query_traces(
                scale=self.SCALE, event_type='delta', chain_suffix=self.NAME,
                exclude_ref_types=list(RESIDUE_REF_TYPES) + list(EMITTER_REF_TYPES),
                hours=None, limit=1)
            events = res.get('events', [])
            return events[0]['created_at'] if events else ''
        except Exception as e:
            # Log to brain errors (not just stderr) so repeated failures
            # show in the errors view. A broken _last_run_timestamp makes
            # every cold-start decision wrong — keep it visible.
            try:
                self.brain._log_error(
                    's2_%s_last_run_timestamp' % self.NAME, e,
                    'scale=%s name=%s' % (self.SCALE, self.NAME))
            except Exception:
                import sys
                print('[%s] _last_run_timestamp error: %s' % (self.NAME, e),
                      file=sys.stderr)
            return ''

    def _has_new_traces(self, scale, ref_type=None):
        """Check if there are traces newer than this unit's last run.

        Generic: any S2 unit can check any scale's traces.
        Returns True on first run (no prior traces).

        Args:
            scale: Scale to check (e.g. 's1', 's2')
            ref_type: Optional filter (e.g. 'encoding_run')
        """
        last_ts = self._last_run_timestamp()
        if not last_ts:
            return True  # Never run — cold start

        try:
            if ref_type:
                traces = self.brain.query_traces(
                    ref_type=ref_type, scale=scale, hours=168, limit=5)['events']
            else:
                traces = self.brain.query_traces(
                    scale=scale, event_type='delta', limit=5)['events']
            for t in traces:
                if t.get('created_at', '') > last_ts:
                    return True
        except Exception as e:
            # Fail-open to "run" is safer than stuck-idle, but log the
            # underlying cause so the operator sees the trace-read outage
            # instead of a silently-retriggering unit.
            try:
                self.brain._log_error(
                    's2_%s_has_new_traces' % self.NAME, e,
                    'trace read failed; falling back to run=True')
            except Exception:
                pass
            return True

        return False

    def _read_traces_window(self, scale, hours=168, ref_types=None):
        """Read every trace event from a scale within a lookback window.

        Generic windowed reader — each unit interprets the results
        according to what it needs.

        Args:
            scale: Scale to read ('s1', 's2', etc.)
            hours: Lookback window (the config lookbacks bind directly).
            ref_types: Optional list of ref_types. None → all ref_types at
                non-s0 scales (s0 inherits the conversational default from
                recall_episodes).

        Returns:
            List of trace event dicts, newest-first.
        """
        # Uncapped (limit=None through the honest-limit door): the window
        # itself is the bound — s1 trace volume is conversation-rate
        # (~100 rows/day/ref_type), and consumers aggregate over the result
        # (co-recall counts), so a row cap here would silently bias the
        # aggregates toward the newest slice. The window bound runs in SQL
        # (`younger_than`), never as a Python post-filter.
        res = self.brain.recall_episodes(
            scale=scale, ref_type=ref_types,
            younger_than='%dh' % hours, sort_order='desc', limit=None)
        return res.get('episodes', [])

    def _accumulate_run(self, total, result):
        """Fold one run_llm_loop result batch into `total` (in place): the loop
        counts (rounds / actions / write_actions), the per-tool records
        (action_details + read_calls — both carry per-tool latency/result_count),
        and token telemetry (via the shared sum_usage). The single multi-batch
        accumulator, shared by the consolidation and community encoders so the
        batch-fold loop lives in one place.

        Per-unit fields (final_text formatting, unit-specific counters) stay in
        each encoder. elapsed_ms is NOT summed — each encoder wall-clocks its
        whole batch loop once. `total` must pre-init action_details/read_calls
        to []."""
        total['rounds'] += result.get('rounds', 0)
        total['actions'] += result.get('actions', 0)
        total['write_actions'] += result.get('write_actions', 0)
        total['action_details'].extend(result.get('action_details') or [])
        total['read_calls'].extend(result.get('read_calls') or [])
        sum_usage(total, result)

    def _fold_batch_result(self, total, result, batch_num, trunc_source,
                           trunc_detail='tool call likely corrupted'):
        """Process one batch's run_llm_loop result for a multi-batch encoder:
        accumulate it into `total` (via _accumulate_run), append this batch's
        final_text, persist its review notes as journal_note rows PER BATCH, and
        log any max_tokens truncation. The shared per-batch body for the
        consolidation + community encoders, so the loop logic lives in one place;
        per-unit steps (state refresh, progress traces) stay in each loop.

        A batch outcome is also a completion boundary: an explicit error or
        zero completed rounds leaves the aggregate failed, even if other
        batches succeeded. Consumers may retain writes and telemetry, but
        must not settle proposals or advance a scan past a failed run.
        Returns whether this batch completed.

        Per-batch journal write (not post-loop): extract_review_block keys on the
        FIRST `## Review` fence, so a single post-loop write over the accumulated
        final_text would drop every batch's notes but the first. Writing per
        batch, all sharing this run's chain_id, groups them as one run's notes.
        The journal write is failure-isolated inside harvest (a journal hiccup
        never aborts the run)."""
        self._accumulate_run(total, result)
        error = result.get('error')
        if not error and not result.get('rounds'):
            error = 'encoder returned no completed rounds'
        if error:
            message = 'batch %d: %s' % (batch_num, error)
            total['error'] = '\n'.join(filter(None, (total.get('error'), message)))
        batch_text = result.get('final_text', '')
        if batch_text:
            total['final_text'] += '\n--- batch %d ---\n%s' % (batch_num, batch_text)
            self.journal.harvest(batch_text, self.chain_id())
        for trunc in result.get('truncations', []):
            self.brain._log_error(
                trunc_source,
                'max_tokens truncation: round %d used %s/%s output tokens' % (
                    trunc['round'], trunc['output_tokens'], trunc['max_tokens']),
                'batch %d — %s' % (batch_num, trunc_detail))
        if result.get('undispatched_tool_calls'):
            self.brain._log_error(
                'undispatched_tool_calls',
                'max_rounds exhausted — last reply\'s tool call(s) not run: %s'
                % ', '.join(result['undispatched_tool_calls']),
                'unit=%s batch %d' % (self.NAME, batch_num))
        for failure in result.get('annotation_errors') or ():
            self.brain._log_error(
                'reply_annotation_failed',
                RuntimeError('%s reply annotation failed: %s'
                             % (failure['tool'], failure['error'])),
                'the reply reached the encoder without its notes; '
                'unit=%s batch %d' % (self.NAME, batch_num))
        return not error

    def _llm_client(self):
        """This unit's Anthropic client, built once per run.

        Units are constructed per run, so caching on the instance gives the
        single-shot path the same per-run client lifetime the loop encoders
        get by hoisting make_client() above their batch loop
        (consolidation_encoder, community_encoder). Healer calls _call_llm
        once per BATCH; building inside the call would give every batch a
        fresh client and a cold TLS pool. Aspect calls it once per run.
        """
        client = getattr(self, '_client', None)
        if client is None:
            client = self._client = make_client()
        return client

    def _call_llm(self, interaction_name, user_content, journal=False,
                  effective=None):
        """Call LLM with a learnable prompt from interactions table.

        Loads system prompt from interaction template.
        Loads config (model, max_tokens) from interaction parameters.
        Handles JSON extraction from response.

        Args:
            interaction_name: Key in interactions table (e.g. 's2_community_enrichment')
            user_content: String content for the user message
            journal: When True, this call carries the unit's journal binding —
                the review block decorates the system tail (single-shot: no
                closure, no arc) and the response is harvested (residue notes
                written on this run's chain, journal sections stripped BEFORE
                extract_json — a `]`/`}` inside a fence after the payload
                would corrupt its rfind-based scan). The single wiring point
                for single-shot units (healer, aspect); continuity is the
                caller's to prepare before each request (see
                scales/journal.py placement rules). Decoration is
                deterministic, so the system-prompt cache stays byte-stable.
            effective: `interaction_name` already resolved
                (brain.get_interaction_effective) — a unit calling once per
                batch resolves once per run and passes it, so every batch
                runs on the read its run records. None → resolve here.

        Returns:
            (parsed_json, telemetry): parsed_json is the JSON parsed from the
            LLM response (None on failure); telemetry is a dict
            {elapsed_ms, input_tokens, output_tokens, cache_read_tokens,
            cache_creation_tokens, model, interaction} for the call — token
            zeros when the call never ran or failed before a usage report;
            `interaction` is the K stamp resolved with the prompt, on every
            path; `failure` ('LLM call failed' / 'truncated response' /
            'unparseable response') is set exactly when parsed_json is None,
            and every cause is already logged here — callers read it, never
            log again. Single-shot encoders (healer,
            aspect) thread this into their delta trace via build_delta_metadata
            (this is run_llm_loop's per-call telemetry, hand-built here because
            this path uses the runner's single-shot entry, not the loop).
        """
        from ..dispatch import load_env

        # Load learnable prompt and config — resolved through the override
        # model (DB override overlaid on the code default; total by
        # construction, unknown names raise in the resolver).
        effective = effective or self.brain.get_interaction_effective(
            interaction_name)
        system_prompt, config = effective['template'], effective['config']
        stamp = effective['stamp']
        model = config['model']
        max_tokens = config['max_tokens']

        if journal:
            system_prompt = self.journal.decorate_system(
                system_prompt, multi_round=False)

        # read_usage(None) is the all-zero token baseline — reused on the
        # pre-usage failure path.
        telemetry = {'elapsed_ms': 0, 'model': model, 'interaction': stamp,
                     **read_usage(None)}

        # Ensure API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        t0 = time.time()
        try:
            # Transport + caching live behind the runner seam (run_llm_once:
            # 5m cache breakpoints, read_usage telemetry). This method keeps
            # only the unit concerns: the interaction-table prompt/config load
            # above, the payload envelope, and the log-and-return-None failure
            # policy below.
            raw, telemetry = run_llm_once(
                self._llm_client(), model, max_tokens, system_prompt, user_content,
                effort=config.get('effort'))
            telemetry['interaction'] = stamp
        except Exception as e:
            print('[%s] LLM call failed: %s' % (self.NAME, e), flush=True)
            self.brain._log_error(self.NAME, e, 'LLM call for %s' % interaction_name)
            telemetry['elapsed_ms'] = int((time.time() - t0) * 1000)
            telemetry['failure'] = 'LLM call failed'
            return None, telemetry

        # Truncation is loud on the single-shot path too — the loop path
        # checks stop_reason in _track_usage; without this, a response that
        # hit the output ceiling read as a generic parse failure.
        if telemetry.get('stop_reason') == 'max_tokens':
            self.brain._log_error(
                's2_%s_truncation' % self.NAME,
                'max_tokens truncation: %s/%s output tokens' % (
                    telemetry.get('output_tokens', 0), max_tokens),
                '%s response truncated — payload likely unparseable'
                % interaction_name)

        if journal:
            # Residue notes out, journal sections off the payload — the
            # strip-before-extract ordering is enforced here by construction.
            # Outside the transport try, in its own guard: a journal-layer
            # fault must never discard a successful, paid response (degrade
            # to the unstripped raw — extract_json is fence-robust).
            try:
                raw = self.journal.harvest(raw, self.chain_id())
            except Exception as e:
                self.brain._log_error(
                    's2_%s_journal_harvest' % self.NAME, e,
                    'harvest failed — parsing the unstripped response')
        parsed = extract_json(raw)
        if parsed is None and telemetry.get('stop_reason') == 'max_tokens':
            telemetry['failure'] = 'truncated response'   # logged above
        elif parsed is None:
            # The call succeeded and was paid for; the reply is the only
            # evidence of why it can't be used, so it goes in the log whole
            # rather than being dropped.
            self.brain._log_error(
                's2_%s_unparseable' % self.NAME,
                'no JSON payload in the %s response (stop_reason=%s, %s '
                'output tokens)' % (interaction_name,
                                    telemetry.get('stop_reason'),
                                    telemetry.get('output_tokens', 0)),
                raw)
            telemetry['failure'] = 'unparseable response'
        return parsed, telemetry
