"""S2CE — Community Encoder.

Takes decoder proposals → runs the S2 Community Encoder agent → writes
community nodes. Uses brain_batch tool through run_llm_loop (same pattern
as S1E). The encoder agent is configured in the s2_community_enrichment
interaction — model and prompt are both learnable.

The agent sees each batch as text: decision evidence for the communities and
nodes the proposals name, nearby communities, and the proposals (counts, edge
signatures, member lines). It creates community nodes with narratives,
situations, and metadata via tool calls.
"""

import os
import time

from .base import IntegrationUnit
from .community_contract import S2CE_NODE_FORMAT
from servers.trace_contract import build_delta_metadata, interaction_fields

from .rejection_table import (
    match_proposals_to_actions,
    record_rejections,
    sort_proposals_by_priority,
    node_ids_touched_by_invalid_ops,
    had_rejected_batch_call,
    get_proposed_ids,
    thwart_shield,
    settle_thwart_streak,
    loud_warning,
)


class CommunityEncoder(IntegrationUnit):
    NAME = 'community_detection'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:community_detection'

    O_SOURCES = ['community_proposals']
    K_SOURCES = ['llm_enrichment', 'journal_notes']

    # Consecutive runs whose thwarted brain_batch attempts (invalid ops or a
    # rejected call) shielded proposals from fingerprinting — the give-up
    # bound reads/resets it across cycles.
    THWARTED_STREAK_KEY = 's2_community_thwarted_streak'

    # Residue flows to journal_note trace rows via brain.write_journal_notes,
    # read back via the journal binding's continuity() (the note contract).

    def __init__(self, brain, dispatch_fn=None, config=None, config_effective=None):
        super().__init__(brain, dispatch_fn)
        # The s2_community resolution the decoder made (`config_effective`),
        # or one made here with a caller's `config=` laid over it (evals).
        # The run records its stamp beside its prompt's — the config shapes
        # every batch: rounds, quotas, sizes.
        self.config_effective = (config_effective or
                                 brain.get_interaction_effective(
                                     's2_community', overlay=config))
        self.config = self.config_effective['config']

    def run(self, proposals, community_state):
        """Encode proposals into community nodes.

        Args:
            proposals: List of proposal dicts from decoder.
                       Only actionable types are processed:
                       new_community, add_to_existing, drift, health_update,
                       merge_communities.
            community_state: Current community state from decoder.

        Returns:
            dict with keys: actions, write_actions, rounds,
                            action_details, final_text
            None on failure.
        """
        # Filter to actionable proposals
        encoder_proposals = [p for p in proposals
                             if p['type'] in ('new_community', 'add_to_existing',
                                              'drift', 'health_update',
                                              'merge_communities')]
        if not encoder_proposals:
            self.trace('delta', 'community_enriched', 'No actionable proposals')
            return {'actions': 0, 'write_actions': 0, 'rounds': 0,
                    'action_details': [], 'final_text': ''}

        # Filter corridors — track in traces, don't send to encoder.
        # Corridors (int_frac < 20%) rarely form coherent communities.
        # If they mature, health_update will catch them.
        corridors = [p for p in encoder_proposals
                     if p['type'] == 'new_community' and p.get('is_corridor')]
        if corridors:
            encoder_proposals = [p for p in encoder_proposals
                                 if not (p['type'] == 'new_community' and p.get('is_corridor'))]
            print('[s2ce] Filtered %d corridor proposals (tracked in traces)' % len(corridors),
                  flush=True)
            self.trace('K', 'community_proposals',
                       'Filtered %d corridors: %s' % (
                           len(corridors),
                           ', '.join('%d members' % c.get('member_count', 0)
                                     for c in corridors[:5])))

        # Honest skip: if corridor filtering emptied the actionable set, there
        # is genuinely nothing to encode. Report it as a skip — NOT as a
        # "COMPLETE: 0 actions in 0 rounds", which reads like a successful run
        # on the dashboard and masks that the encoder never had work to do.
        if not encoder_proposals:
            self.trace('delta', 'community_enriched',
                       'SKIPPED: %d corridor(s) filtered, no other actionable proposals'
                       % len(corridors))
            return {'actions': 0, 'write_actions': 0, 'rounds': 0,
                    'action_details': [], 'final_text': '',
                    'skipped': 'corridors_only'}

        # Cap with per-type quotas (process backlog over multiple idle cycles).
        # Sort within each type by confidence (highest first) so strong
        # proposals get encoder time before borderline ones.
        max_per_run = self.config.get('max_actionable_per_run', 60)
        quotas = self.config.get('type_quotas', {})
        if quotas and len(encoder_proposals) > max_per_run:
            sorted_proposals = sort_proposals_by_priority(encoder_proposals)
            capped = []
            by_type = {}
            for p in sorted_proposals:
                by_type.setdefault(p['type'], []).append(p)
            for ptype, quota in quotas.items():
                available = by_type.get(ptype, [])
                capped.extend(available[:quota])
            skipped = len(encoder_proposals) - len(capped)
            # Re-sort capped set by priority so encoder sees structural-impact
            # proposals first (merge → new_community → add_to_existing → ...)
            encoder_proposals = sort_proposals_by_priority(capped[:max_per_run])
            print('[s2ce] Capped to %d proposals (%d deferred): %s' % (
                len(encoder_proposals), skipped,
                ', '.join('%s=%d' % (t, sum(1 for p in encoder_proposals if p['type'] == t))
                          for t in quotas if any(p['type'] == t for p in encoder_proposals))),
                flush=True)
        else:
            # Even without quota cap, sort by priority for encoder batching
            encoder_proposals = sort_proposals_by_priority(encoder_proposals)

        result = self._encode(encoder_proposals, community_state)
        if not result:
            self.trace('delta', 'community_enriched', 'Enrichment failed')
            return None

        actions = result.get('actions', 0)
        write_actions = result.get('write_actions', 0)
        rounds = result.get('rounds', 0)

        # Precise rejection stamping: walk brain_batch operations to identify
        # which proposals the encoder actually acted on. Only stamp skipped
        # ones. Accepted proposals auto-invalidate on the next decode because
        # the graph state changed (new community_member edges, etc.).
        # Completion is carried by the shared batch result. Model prose is
        # evidence, never a control signal for suppression.
        action_details = result.get('action_details', [])
        final_text = result.get('final_text', '') or ''
        if result.get('error'):
            print('[s2ce] Encoder failed or incomplete - NOT stamping rejections',
                  flush=True)
            result['rejection_skipped_count'] = 0
            acted_on = []
        else:
            acted_on, skipped_proposals = match_proposals_to_actions(
                encoder_proposals, action_details)
            # Thwarted proposals: the encoder TRIED to act and dispatch
            # dropped the attempt — an invalid op naming the proposal's nodes
            # (e.g. a concept-verb `op: reject` rather than a real op),
            # or a rejected brain_batch call, which names no node ids and so
            # taints every un-acted-on proposal. Retry, not stamp; the bound
            # and its rationale live on THWARTED_RETRY_LIMIT (fingerprints
            # are community's only suppression).
            invalid_touched = node_ids_touched_by_invalid_ops(action_details)
            rejected_call = had_rejected_batch_call(action_details)
            retry, kept = [], []
            if rejected_call or invalid_touched:
                for p in skipped_proposals:
                    if rejected_call or set(get_proposed_ids(p)) & invalid_touched:
                        retry.append(p)
                    else:
                        kept.append(p)
            streak, shield_active = thwart_shield(
                self.brain, self.THWARTED_STREAK_KEY)
            shielded = bool(retry) and shield_active
            # Name predates the rejected-call cause — the key is a public
            # seam (delta metadata, eval consumers), so it stays.
            invalid_op_failures = len(retry) if shielded else 0
            if shielded:
                skipped_proposals = kept
                cause = ('a rejected brain_batch call' if rejected_call
                         else 'invalid brain_batch ops')
                loud_warning(
                    self.brain, 's2_community_thwarted_retry', 's2ce',
                    '%d proposal(s) thwarted by %s — retrying next cycle, '
                    'NOT suppressed' % (invalid_op_failures, cause))
            elif retry:
                loud_warning(
                    self.brain, 's2_community_thwarted_giveup', 's2ce',
                    'thwarted brain_batch attempts in %d shielded runs — '
                    'stamping %d proposal(s) anyway to unpin the unit'
                    % (streak, len(retry)))
            settle_thwart_streak(
                self.brain, self.THWARTED_STREAK_KEY, streak, shielded=shielded)
            if skipped_proposals:
                record_rejections(self.brain, skipped_proposals)
                print('[s2ce] Stamped %d rejected proposals (fingerprints)' % len(skipped_proposals),
                      flush=True)
                # Algorithmic Δ on the same signal as the fingerprints above:
                # the agent's judgment is "reject", the resulting NUMBER is
                # arithmetic, so code writes it. Ordered AFTER the fingerprints
                # deliberately — the fingerprint is the durable record and the
                # one that lifts when evidence changes, so a failure here may
                # cost a recomputable raise but never that. Failure-isolated
                # for the same reason.
                try:
                    self._raise_drift_thresholds(skipped_proposals)
                except Exception as e:
                    self.brain._log_error('s2_community_drift_threshold', e,
                                          'drift threshold stamp failed')
            result['rejection_skipped_count'] = len(skipped_proposals)
            result['invalid_op_failures'] = invalid_op_failures

        # Outcome vocab — count acted-on proposals by type.
        outcomes = {}
        for p in acted_on:
            ptype = p.get('type', 'unknown')
            outcomes[ptype] = outcomes.get(ptype, 0) + 1

        self.trace('delta', 'community_enriched',
                   '%s: %d actions (%d writes) in %d rounds, %d stamped, '
                   '%dms, %d→%d tok' % (
                       'FAILED' if result.get('error') else 'COMPLETE',
                       actions, write_actions, rounds,
                       result.get('rejection_skipped_count', 0),
                       result.get('elapsed_ms', 0),
                       result.get('input_tokens', 0),
                       result.get('output_tokens', 0)),
                   metadata=build_delta_metadata(
                       actions=actions,
                       write_actions=write_actions,
                       rounds=rounds,
                       inputs_processed=len(encoder_proposals),
                       outcomes=outcomes,
                       rejection_skipped=result.get('rejection_skipped_count', 0),
                       invalid_op_failures=result.get('invalid_op_failures', 0),
                       action_details=action_details,
                       read_calls=result.get('read_calls', []),
                       final_text=final_text,
                       corridors_filtered=len(corridors),
                       elapsed_ms=result.get('elapsed_ms', 0),
                       input_tokens=result.get('input_tokens', 0),
                       output_tokens=result.get('output_tokens', 0),
                       cache_read_tokens=result.get('cache_read_tokens', 0),
                       cache_creation_tokens=result.get('cache_creation_tokens', 0),
                       model=result.get('model', ''),
                       interaction=result['interaction'],
                       **interaction_fields(self.config_effective['stamp'], 'config_'),
                       errors=[result['error']] if result.get('error') else [],
                       context_chars=result.get('context_chars', []),
                       context_parts=result.get('context_parts', []),
                   ))

        return result

    def _raise_drift_thresholds(self, skipped_proposals):
        """Per-encode Δ: raise `_sys_drift_threshold` on every drift proposal
        the encoder saw and declined.

        Authorship split, same as the structural fields (id:e0b7f842):
        rejecting is the agent's judgment, `current + step` is arithmetic, so
        the algorithm owns the number. The agent used to write it from a
        prompt template and produced unparseable strings ("1.6x") that died
        inside the decoder's float() — a rejection that suppressed nothing.

        Rides the same signal as record_rejections: the encoder completed,
        saw the proposal, and did not act on it. Thwarted proposals (invalid
        ops, rejected batch call) are already filtered out upstream and are
        retried rather than stamped.
        """
        step = self.config.get('drift_ratio_step', 0.1)
        default = self.config.get('drift_ratio', 1.5)
        ops = []
        for p in skipped_proposals:
            if p.get('type') != 'drift' or not p.get('node_id'):
                continue
            current = p.get('current_drift_threshold', default)
            ops.append({
                'op': 'revise',
                'node_id': p['node_id'],
                'reason': 'drift proposal declined — threshold raised',
                '_sys_drift_threshold': '%.1f' % (current + step),
            })
        if not ops:
            return 0

        dispatch_fn = self._make_dispatch()
        dispatch_fn('brain_batch', {
            'operations': ops,
            'encoding_source': self.ENCODING_SOURCE,
        })
        self.trace('delta', 'community_enriched',
                   'DRIFT THRESHOLD STAMP: %d node(s) raised by %.1f' % (
                       len(ops), step))
        print('[s2ce] drift threshold stamp: %d node(s)' % len(ops), flush=True)
        return len(ops)

    def _stamp_structural_fields(self, encoder_proposals, previous_ids, state):
        """Per-batch Δ: stamp the structural fields of every community a batch
        touched, derived from the member edges its writes left.

        `state` is the community state read after the batch — this unit's live
        communities with their stored values — and `previous_ids` the ids it
        held before. Touched = created (live now, absent before) + added-to /
        merged / health-updated / drifted-into (from the proposals). Every
        other community was written at the decode seam this cycle
        (CommunityDetection._stamp_structural). Gate-safe: community-node
        metadata revises are ignored by the idle gate (type='community') and
        create no edges. Returns the count written.
        """
        from .community_structural import (
            compute_community_structural, write_structural_fields)

        touched = set()
        for p in encoder_proposals:
            t = p.get('type')
            if t == 'health_update' and p.get('community_id'):
                touched.add(p['community_id'])
            elif t == 'merge_communities' and p.get('larger_id'):
                touched.add(p['larger_id'])
            elif t == 'add_to_existing':
                touched.update(c['id'] for c in p.get('communities', [])
                               if c.get('id'))
            elif t == 'drift':
                touched.update(f['id'] for f in p.get('foreign', [])
                               if f.get('id'))

        live = {c['id']: c for c in state}
        touched |= set(live) - previous_ids
        # A merge's smaller or a health archive may be gone; never stamp one.
        ids = sorted(touched & set(live))
        if not ids:
            return 0
        derived = compute_community_structural(self.brain, ids)
        n = write_structural_fields(
            self.brain, self._make_dispatch(), derived,
            {cid: live[cid]['meta'] for cid in ids}, self.ENCODING_SOURCE)
        # The state the next batch reads, current without a re-read.
        for cid in ids:
            live[cid]['meta'].update(derived[cid])
        if n:
            # Bare marker (no build_delta_metadata payload) — the contract
            # lets delta ref_types double as markers; a partial dict would
            # trip the required-keys guard. The count lives in the summary.
            self.trace('delta', 'community_enriched',
                       'STRUCTURAL STAMP: %d communities derived from edges' % n)
            print('[s2ce] structural stamp: %d communities' % n, flush=True)
        return n

    # ══════════════════════════════════════════════════════════
    # _encode — S2 Community Encoder agent via run_llm_loop
    # ══════════════════════════════════════════════════════════

    def _encode(self, proposals, community_state):
        """Run S2CE in batches — processes ALL proposals.

        Proposals are split into chunks of max_proposals_per_call.
        Each chunk gets one encoder-agent call. Between chunks, community_state
        is refreshed so later chunks see what earlier ones created.
        """
        from ..dispatch import load_env
        from ..runner import run_llm_loop, make_client, retry_on_transient_api_error
        from .community_decoder import CommunityDecoder

        effective = self.brain.get_interaction_effective('s2_community_enrichment')
        system_prompt, config = effective['template'], effective['config']
        model = config['model']
        max_tokens = config['max_tokens']

        # Prompt closers — the edge-aspect vocabulary, then the journal
        # component's system-tail decoration (review block + closure, DONE
        # genuinely last). Single-sourced: aspects in servers/aspects.py, the
        # journal blocks in trace_contract via scales/journal.py.
        system_prompt = self._inject_edge_aspects(system_prompt)
        system_prompt = self.journal.decorate_system(system_prompt)

        if not os.environ.get('ANTHROPIC_API_KEY'):
            load_env()

        tools = self._get_tool_schemas()
        dispatch_fn = self._make_dispatch()
        client = make_client()

        # Batch proposals
        batch_size = self.config.get('max_proposals_per_call', 15)
        context_limit = self.config['max_batch_context_chars']
        total_result = {
            'rounds': 0, 'actions': 0, 'write_actions': 0,
            'action_details': [], 'read_calls': [], 'final_text': '',
            # The K stamp resolved with the prompt above — run()'s delta
            # records which prompt+config produced it.
            'interaction': effective['stamp'],
            'context_chars': [],
            'context_parts': [],
        }
        # Cost/latency telemetry — loop counts, per-tool records, and tokens are
        # folded per batch by the shared _accumulate_run; elapsed by a wall-clock
        # timer around the whole batch loop. Mirrors S1 Scribe / consolidation.
        _t0 = time.time()
        current_state = list(community_state)

        # Need a decoder instance to refresh community state between batches
        decoder = CommunityDecoder(self.brain, self.dispatch,
                                   config_effective=self.config_effective)

        proposal_offset = 0
        batch_idx = 0
        while proposal_offset < len(proposals):
            continuity = self.journal.continuity(chain_id=self.chain_id())
            batch = proposals[proposal_offset:proposal_offset + batch_size]
            # Largest contiguous prefix that fits. Shared targets are rendered
            # once, so count-based halving can needlessly repeat their evidence.
            # Search actual renders: optional recall can make sizes non-monotonic.
            while True:
                user_content, context_parts = self._build_batch_context(
                    batch, current_state, continuity, context_limit)
                if len(user_content) <= context_limit or len(batch) == 1:
                    break
                batch = batch[:-1]
            if len(user_content) > context_limit:
                self.brain._log_warning(
                    's2ce_oversized_context',
                    'single proposal exceeds community context budget',
                    '%d chars > %d; retaining complete decision evidence' % (
                        len(user_content), context_limit))

            batch_num = batch_idx + 1
            print('[s2ce] Batch %d (%d proposals, %d context chars)' % (
                batch_num, len(batch), len(user_content)), flush=True)
            total_result['context_chars'].append(len(user_content))
            total_result['context_parts'].append(context_parts)

            try:
                result = retry_on_transient_api_error(
                    lambda: run_llm_loop(
                        client=client,
                        model=model,
                        effort=config.get('effort'),
                        max_tokens=max_tokens,
                        max_rounds=self.config.get('max_rounds', 4),
                        system_prompt=system_prompt,
                        user_content=user_content,
                        tools=tools,
                        dispatch_fn=dispatch_fn,
                        get_nodes_config=S2CE_NODE_FORMAT,
                        log_fn=lambda msg: print('[s2ce] %s' % msg, flush=True),
                        record_round_fn=self.brain.round_recorder(
                            self.chain_id(), seq_base=batch_num * 100)),
                    log_fn=lambda msg: print('[s2ce] %s' % msg, flush=True))

            except Exception as e:
                print('[s2ce] BATCH %d FAILED: %s' % (batch_num, e), flush=True)
                self.brain._log_error(self.NAME, e,
                                      'encode batch %d' % batch_num)
                result = {'error': str(e) or type(e).__name__}

            completed = self._fold_batch_result(
                total_result, result, batch_num, 's2ce_truncation',
                trunc_detail='tool call likely corrupted, community data may be lost')
            # Reads must see writes from any completed rounds, including a
            # batch that failed after dispatching some of its operations.
            previous_ids = {c['id'] for c in current_state}
            current_state = decoder._read_community_state()
            # Second, ALGORITHMIC Δ: the structural fields of what this batch
            # touched, from the edges its writes left — arithmetic the agent
            # no longer authors (it guessed blind and they drifted). Stamped
            # before the next batch renders those communities. Failure-
            # isolated — a stamp failure never breaks the run.
            try:
                self._stamp_structural_fields(batch, previous_ids, current_state)
            except Exception as e:
                self.brain._log_error('community_structural_stamp', e,
                                      'structural stamp failed')
            if completed:
                self.trace('delta', 'community_enriched',
                           'batch %d: %d actions (%d writes), %d total' % (
                               batch_num,
                               result.get('actions', 0), result.get('write_actions', 0),
                               total_result['actions']))
            else:
                self.trace('delta', 'community_enriched',
                           'batch %d FAILED: %s' % (
                               batch_num, total_result['error'][-200:]))
            proposal_offset += len(batch)
            batch_idx += 1

        total_result['elapsed_ms'] = int((time.time() - _t0) * 1000)
        return total_result

    # ══════════════════════════════════════════════════════════
    # Tool infrastructure
    # ══════════════════════════════════════════════════════════

    def _get_tool_schemas(self):
        """S2CE tools: brain_batch (primary) + get_nodes (one read call)."""
        from servers import brain_mcp
        S2CE_TOOLS = {
            'brain_batch',
            'get_nodes',
        }
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["inputSchema"]}
                for t in brain_mcp.TOOLS if t["name"] in S2CE_TOOLS]

    def _make_dispatch(self):
        """S2CE dispatch — inherits base `_make_encoder_dispatch` (no archive
        guard; community encoder archives whole communities, not individual
        members from a tracked batch).
        """
        return self._make_encoder_dispatch()

    # ══════════════════════════════════════════════════════════
    # Relevant community lookup
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _decision_community_ids(proposals):
        """Every community named by a decision, independently of recall."""
        ids = []
        for p in proposals:
            ids.extend(p.get(k) for k in (
                'community_id', 'home_id', 'larger_id', 'smaller_id'))
            ids.extend(c.get('id') for k in ('communities', 'foreign')
                       for c in p.get(k, []))
            ids.append((p.get('overlaps_existing') or {}).get('id'))
        return list(dict.fromkeys(i for i in ids if i))

    def _decision_context(self, proposals):
        """Render targets and proposed members through the shared node view.

        The member preview is evidence, not a second stored membership list.
        A merge shows every member absorb moves over, past the preview, so
        the merged account can carry them.
        """
        from servers.contract import render_rich_node
        from .community_contract import S2CE_DECISION_FORMAT, S2CE_MEMBER_PREVIEW

        ids = self._decision_community_ids(proposals)
        node_ids = list(dict.fromkeys(p['node_id'] for p in proposals
                                     if p.get('node_id')))
        if not ids and not node_ids:
            return '', set()
        nodes = self.brain.get_node(list(dict.fromkeys(ids + node_ids))) or {}
        communities = {n['id']: n for n in nodes.values()
                       if n.get('type') == 'community' and not n.get('archived')}
        members = self.brain._graph.get_members_bulk(list(communities))
        lines = ['DECISION EVIDENCE (targets fetched by ID; full community narrative, '
                 'metadata values capped at %d characters):' %
                 S2CE_DECISION_FORMAT['metadata_limit']]
        for requested_id, node in nodes.items():
            if requested_id != node['id']:
                lines.append('  ID MOVED: %s → %s; use the current ID for writes.' % (
                    requested_id, node['id']))
        for nid, node in communities.items():
            # Pure membership has its own read below. Keep mixed-relation
            # edges: a member can also correct or extend the community.
            member_rows = sorted(members.get(nid, []),
                                 key=lambda m: (m.get('created_at') or '', m['id']),
                                 reverse=True)
            view = dict(node, connections=[e for e in node.get('connections', [])
                                          if any(r.get('relation') != 'community_member'
                                                 for r in e.get('relations') or [e])])
            lines.append(render_rich_node(view, S2CE_DECISION_FORMAT))
            shown = member_rows[:S2CE_MEMBER_PREVIEW]
            lines.append('  Live members: %d; %d newest shown, %d not shown. '
                         'Use get_nodes on member IDs for more detail.' % (
                             len(member_rows), len(shown), len(member_rows) - len(shown)))
            for m in shown:
                lines.append('    [%s] "%s" (id:%s, %s)' % (
                    m.get('type', '?'), m.get('title', '?'), m['id'],
                    (m.get('created_at') or '?')[:10]))
            # All IDs stay available, including members outside the preview.
            lines.append('  All live member IDs: %s' % (
                ', '.join(m['id'] for m in member_rows) or '(none)'))
            lines.append('')

        for nid in ids + node_ids:
            if nid not in nodes:
                lines.append('  UNAVAILABLE: %s — inspect before deciding on it.' % nid)
        rendered = set(communities)
        for nid in ids + node_ids:
            node = nodes.get(nid)
            if node and node['id'] not in rendered:
                lines.append('  Node detail (content capped at %d characters):' %
                             S2CE_NODE_FORMAT['content_limit'])
                lines.append(render_rich_node(node, S2CE_NODE_FORMAT))
                rendered.add(node['id'])
        for p in proposals:
            if p['type'] == 'merge_communities':
                larger = nodes.get(p['larger_id'], {}).get('id')
                smaller = nodes.get(p['smaller_id'], {}).get('id')
                if larger == smaller and larger in communities:
                    lines.append('  Merge already resolved to %s; do not merge it '
                                 'with itself.' % larger)
                elif larger in communities and smaller in communities:
                    existing = {m['id'] for m in members.get(larger, [])}
                    transfer = sorted(m['id'] for m in members.get(smaller, [])
                                      if m['id'] not in existing)
                    lines.append('  Merge %s → %s: absorb moves ALL %d members over: %s' % (
                        smaller, larger, len(transfer), ', '.join(transfer) or '(none)'))
        return '\n'.join(lines), set(communities)

    def _build_batch_context(self, proposals, community_state, continuity, limit):
        """Assemble the actual model input and account for each contribution."""
        proposal_text = self._format_proposals(proposals)
        decision = self._decision_context(proposals)
        relevant = self._find_relevant_communities(
            proposals, community_state,
            max_chars=limit - len(continuity) - len(proposal_text) - 2,
            decision_context=decision) or 'EXISTING COMMUNITIES: None.'
        text = continuity + relevant + '\n\n' + proposal_text
        return text, {
            'decision_chars': len(decision[0]),
            'nearby_chars': len(relevant) - len(decision[0]),
            'proposal_chars': len(proposal_text),
            'continuity_chars': len(continuity),
            'separator_chars': 2,
        }

    def _find_relevant_communities(self, batch_proposals, community_state,
                                   max_chars=None, decision_context=None):
        """Exact decision targets plus compact semantically recalled neighbours."""
        from servers.contract import render_rich_node
        from .community_contract import S2CE_COMMUNITY_FORMAT

        decision_text, decision_ids = (decision_context if decision_context is not None
                                       else self._decision_context(batch_proposals))
        if max_chars is not None and len(decision_text) >= max_chars:
            return decision_text or None

        # Build a query from member titles in this batch
        titles = []
        for p in batch_proposals:
            if p['type'] == 'new_community':
                for m in p.get('all_members', [])[:5]:
                    titles.append(m.get('title', ''))
            elif p.get('node_title'):
                titles.append(p['node_title'])
        query = ' '.join(titles)[:500]
        if not query.strip():
            return decision_text or None

        try:
            recall_result = self.brain.recall(
                query, limit=15,
                filter={'type': {'in': ['community']}})
            results = recall_result.get('results', []) if isinstance(recall_result, dict) else []
            if not results:
                return decision_text or None

            # Batch-fetch full nodes for rendering
            result_ids = [r['id'] for r in results
                          if r.get('id') and r['id'] not in decision_ids]
            rich_nodes = self.brain.get_node(result_ids) if result_ids else {}
            if not rich_nodes:
                return decision_text or None

            lines = [decision_text, 'NEARBY COMMUNITIES (compact context; %d total):' %
                     len(community_state)]
            omitted = 0
            omission_notice = ('  %d nearby communities omitted to preserve '
                               'the decision evidence within the batch budget.')
            for nid, node in rich_nodes.items():
                rendered = render_rich_node(node, S2CE_COMMUNITY_FORMAT)
                # Optional context yields to decision evidence. Reserve room
                # for the omission notice so the budget remains honest.
                if max_chars is not None and (
                        len('\n'.join(lines)) + len(rendered) + 3 +
                        len(omission_notice % len(rich_nodes)) > max_chars):
                    omitted += 1
                    continue
                lines.append(rendered)
                lines.append('')
            if omitted:
                lines.append(omission_notice % omitted)
            text = '\n'.join(lines)
            # Even an omission notice is optional; it cannot force a split.
            return decision_text if max_chars is not None and len(text) > max_chars else text
        except Exception as e:
            # Fallback: compact one-liner listing
            print('[s2ce] Community recall failed: %s — using compact listing' % e,
                  flush=True)
            lines = [decision_text,
                     'EXISTING COMMUNITIES (%d total):' % len(community_state)]
            shown = 0
            notice = '  ... %d communities not shown'
            for comm in community_state[:20]:
                line = '  "%s" (%d members)' % (
                    comm['title'][:60], len(comm['members']))
                if max_chars is not None and (
                        len('\n'.join(lines)) + len(line) + 2 +
                        len(notice % len(community_state)) > max_chars):
                    continue
                lines.append(line)
                shown += 1
            if len(community_state) > shown:
                lines.append(notice % (len(community_state) - shown))
            text = '\n'.join(lines)
            return decision_text if max_chars is not None and len(text) > max_chars else text

    # ══════════════════════════════════════════════════════════
    # Proposal formatting (text rendering for the encoder agent)
    # ══════════════════════════════════════════════════════════

    def _format_proposals(self, proposals):
        from servers.scales.s1.surface_contract import _relative_time

        lines = ['PROPOSALS:\n']

        for i, prop in enumerate(proposals):
            ptype = prop['type'].upper().replace('_', ' ')
            lines.append('[%d] %s' % (i + 1, ptype))

            if prop['type'] == 'new_community':
                lines.append('    %d members, int_frac=%.0f%%' % (
                    prop['member_count'],
                    prop['internal_fraction'] * 100))

                ov = prop.get('overlaps_existing')
                if ov:
                    lines.append(
                        '    ⚠ %.0f%% of members connect into existing '
                        'community "%s" (community_id: %s) — judge: '
                        'genuinely new story, or an extension of that one?' % (
                            ov.get('connected_frac', 0) * 100,
                            ov.get('title', '?'),
                            (ov.get('id') or '?')[:8]))

                # Edge signature — what kind of story
                sig = prop.get('edge_signature', {})
                if sig:
                    parts = ['%s(%.0f%%)' % (f, p * 100)
                             for f, p in sorted(sig.items(),
                                                key=lambda x: -x[1])[:4]]
                    lines.append('    Signature: %s' % ', '.join(parts))

                # Members — one line each with relative time.
                # Encoder can call get_nodes() to inspect specific ones.
                all_members = prop.get('all_members', [])
                if all_members:
                    lines.append('    Members:')
                    for m in all_members:
                        age = _relative_time(m.get('date', '')) or m.get('date', '?')
                        lines.append('      [%s] "%s" (id:%s, %s)' % (
                            m.get('type', '?'),
                            m.get('title', '?'),
                            m.get('id', '?')[:8],
                            age))

            elif prop['type'] == 'add_to_existing':
                node_id = prop.get('node_id', '?')
                lines.append('    Node: [%s] "%s" (node_id: %s)' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?'),
                    node_id[:8] if node_id else '?'))
                for comm in prop.get('communities', []):
                    lines.append('    → connect to "%s" (community_id: %s, affinity: %.0f%%)' % (
                        comm.get('title', '?'),
                        comm.get('id', '?')[:8],
                        comm.get('affinity', 0) * 100))

            elif prop['type'] == 'drift':
                node_id = prop.get('node_id', '?')
                lines.append('    Node: [%s] "%s" (node_id: %s)' % (
                    prop.get('node_type', '?'),
                    prop.get('node_title', '?'),
                    node_id[:8] if node_id else '?'))
                lines.append('    Home: "%s" (community_id: %s, affinity: %.0f%%)' % (
                    prop.get('home_community', '?'),
                    prop.get('home_id', '?'),
                    prop.get('home_affinity', 0) * 100))
                lines.append('    Current threshold: %.1f' % (
                    prop.get('current_drift_threshold', 1.5)))
                for f in prop.get('foreign', []):
                    lines.append('    Drifting toward: "%s" (community_id: %s, affinity: %.0f%%)' % (
                        f.get('title', '?'),
                        f.get('id', '?')[:8],
                        f.get('affinity', 0) * 100))

            elif prop['type'] == 'health_update':
                lines.append('    Community: "%s" (community_id: %s)' % (
                    prop.get('community_title', '?'),
                    prop.get('community_id', '?')[:8]))
                lines.append('    Signal: %s (int_frac %.2f)' % (
                    prop.get('signal', '?'), prop.get('fraction', 0)))

            elif prop['type'] == 'merge_communities':
                lines.append('    Larger: "%s" (%d members, id:%s)' % (
                    prop.get('larger_title', '?'),
                    prop.get('larger_size', 0),
                    prop.get('larger_id', '?')[:8]))
                lines.append('    Smaller: "%s" (%d members, id:%s)' % (
                    prop.get('smaller_title', '?'),
                    prop.get('smaller_size', 0),
                    prop.get('smaller_id', '?')[:8]))
                lines.append('    Overlap: %d shared (%.0f%% of smaller), %d unique in smaller' % (
                    prop.get('shared_count', 0),
                    prop.get('overlap_pct', 0) * 100,
                    prop.get('unique_in_smaller', 0)))

            lines.append('')

        return '\n'.join(lines)
