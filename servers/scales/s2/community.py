"""S2 Community Detection — orchestrator.

Wires CommunityDecoder (algorithmic) → rejection filter → CommunityEncoder.
This file is the public API — callers import CommunityDetection from here.

Decoder: servers/scales/s2/community_decoder.py
Encoder: servers/scales/s2/community_encoder.py
Rejection suppression: servers/scales/s2/rejection_table.py
Contract: servers/scales/s2/community_contract.py
Prompt: s2_community_enrichment interaction (v12+)

Phase 1 idle gate (2026-05-29): the decode is a pure function of graph state,
so re-running it on an unchanged graph re-derives identical, already-rejected
proposals. The gate skips the whole unit unless the graph changed since the
last decode AND a minimum interval elapsed — turning a ~15s full-graph scan
that fired every idle cycle (87% doing zero work) into a cheap no-op when
nothing has changed.
"""

import time as _time
from datetime import datetime, timezone

from .community_decoder import CommunityDecoder
from .community_encoder import CommunityEncoder
from .community_structural import write_structural_fields
from .rejection_table import (
    filter_rejected, record_rejections, clear_unplaceable_rejections)


class CommunityDetection(CommunityDecoder):
    """Full S2CD+S2CE pipeline — gate → decode → filter rejections → encode.

    Inherits from CommunityDecoder so all decoder methods are available.
    Callers that only need the decoder can use CommunityDecoder directly.
    """

    # brain_meta key: epoch of the last completed pipeline.
    LAST_RUN_KEY = 's2_community_last_run_ts'

    def run(self):
        """Idle gate → decode → filter → encode, with timing.

        Returns:
            dict with: actions, proposals, communities, details, skipped,
            error, elapsed_ms
        """
        skip_reason = self._should_skip()
        if skip_reason:
            # Silent — no trace. With the gate in place, skipped cycles are
            # the common case; tracing each one would re-bloat the very trace
            # table this gate exists to shrink. The visible proof is simply
            # far fewer real runs (each carrying its elapsed_ms).
            return {'actions': 0, 'skipped': skip_reason}

        result = None
        t0 = _time.time()  # clock-ok — idle-cycle wall-clock, not conversation time
        try:
            result = self._run_pipeline()
            return result
        finally:
            # A failed pipeline has not settled its work. Hold the cutoff
            # so the next cycle can retry; successful writes remain in the
            # graph and the decoder observes them on its next scan.
            if isinstance(result, dict) and not result.get('error'):
                self.brain.set_config(self.LAST_RUN_KEY, str(_time.time()))  # clock-ok
            elapsed_ms = int((_time.time() - t0) * 1000)
            if isinstance(result, dict):
                result['elapsed_ms'] = elapsed_ms
            print('[s2cd] community decode+encode: %dms' % elapsed_ms, flush=True)

    # ══════════════════════════════════════════════════════════
    # Phase 1 idle gate
    # ══════════════════════════════════════════════════════════

    def _should_skip(self):
        """Return a skip-reason string, or None to proceed.

        System bookkeeping — wall-clock is correct here (real time since the
        last decode), the same basis as Brain._maintenance_last_run_ts. This
        is NOT conversation time, so it takes no `at=` anchor.
        """
        raw = self.brain.get_config(self.LAST_RUN_KEY) or '0'
        try:
            last_run_ts = float(raw)
        except (TypeError, ValueError):
            last_run_ts = 0.0
        if last_run_ts <= 0:
            return None  # cold start / never run — always proceed

        now = _time.time()  # clock-ok
        since = now - last_run_ts
        min_interval = self.config.get('min_run_interval_seconds', 30 * 60)
        if since < min_interval:
            return 'throttled (%.0fm < %.0fm min interval)' % (
                since / 60.0, min_interval / 60.0)

        cutoff_iso = datetime.fromtimestamp(
            last_run_ts, tz=timezone.utc).isoformat()
        if not self._graph_changed_since(cutoff_iso):
            return 'no graph change since last run'
        return None

    def _graph_changed_since(self, cutoff_iso):
        """True if the graph changed since cutoff_iso in a way that could
        produce a NEW community proposal.

        Counts: any non-community node created or revised; any non-noise,
        non-self typed edge_relation added. Excludes this unit's own writes
        (community nodes by type; community edges by encoding_source) so a
        productive run does not immediately re-trigger itself. Noise-aspect
        edges (e.g. co_anchored) are excluded — they must not wake
        community detection.
        """
        c = self.brain.conn
        if c.execute(
                "SELECT 1 FROM nodes "
                "WHERE (created_at > ? OR updated_at > ?) "
                "AND type != 'community' LIMIT 1",
                (cutoff_iso, cutoff_iso)).fetchone():
            return True

        noise = list(self.brain.aspects.relations_in(['noise', 'generic_relation']))
        if noise:
            placeholders = ','.join('?' * len(noise))
            query = (
                "SELECT 1 FROM edge_relations "
                "WHERE created_at > ? AND archived = 0 "
                "AND encoding_source != ? "
                "AND relation NOT IN (%s) LIMIT 1" % placeholders)
            params = (cutoff_iso, self.ENCODING_SOURCE, *noise)
        else:
            query = (
                "SELECT 1 FROM edge_relations "
                "WHERE created_at > ? AND archived = 0 "
                "AND encoding_source != ? LIMIT 1")
            params = (cutoff_iso, self.ENCODING_SOURCE)
        return c.execute(query, params).fetchone() is not None

    # ══════════════════════════════════════════════════════════
    # Pipeline — decode → filter rejections → encode (behavior unchanged)
    # ══════════════════════════════════════════════════════════

    def _run_pipeline(self):
        """Run full decode → filter → encode pipeline.

        Rejection filter runs between decoder and encoder. Proposals whose
        fingerprints match previous rejections are suppressed before the
        encoder sees them — prevents the re-proposal loop where the encoder
        wastes tokens judging identical proposals every idle cycle.

        Returns:
            dict with: actions, proposals, communities, details, skipped, error
        """
        # s2_rejections table is created by ensure_schema() at Brain startup
        # (see servers/schema.py). No per-run creation needed.

        # Decode
        decode_result = super().run()

        if decode_result.get('skipped'):
            return {'actions': 0, 'skipped': decode_result['skipped']}

        # Deterministic seam (2026-06-23): archive communities whose internal
        # cohesion fell below the hard floor BEFORE the encoder sees anything —
        # no encoder round, no rejection fingerprint, so they can't get stuck
        # suppressed-but-undead. Mirrors _heal_graph's 0-member sweep.
        archived_ids = self._auto_archive_dead(
            decode_result.get('auto_archive_dead', []))

        raw_proposals = decode_result['proposals']
        community_state = decode_result['community_state']
        if archived_ids:
            community_state = [c for c in community_state
                               if c['id'] not in archived_ids]
            raw_proposals = self._drop_proposals_for_archived(
                raw_proposals, archived_ids)

        self._stamp_structural(decode_result.get('structural', {}),
                               community_state)

        if not raw_proposals:
            # Decoder examined the pending nodes and proposed nothing → none
            # placed → mark them all unplaceable so they rest next cycle.
            self._mark_unplaced_pending(decode_result.get('pending_probes', []))
            return {'actions': 0, 'proposals': 0,
                    'auto_archived': len(archived_ids),
                    'communities': len(community_state)}

        # Filter through rejection table — suppress re-proposals
        proposals, suppressed_count = filter_rejected(self.brain, raw_proposals)

        if suppressed_count:
            # Use community_proposals ref_type (valid per trace contract).
            # Summary prefix 'SUPPRESSED:' lets the dashboard distinguish this
            # from a regular decoder-output proposals trace.
            self.trace('K', 'community_proposals',
                       'SUPPRESSED: %d proposals matched prior rejections (of %d raw), %d surviving' % (
                           suppressed_count, len(raw_proposals), len(proposals)),
                       metadata={'suppressed': suppressed_count,
                                 'raw_total': len(raw_proposals),
                                 'surviving': len(proposals)})

        # Encode (only if anything survived filtering).
        encode_result = None
        if proposals:
            encoder = CommunityEncoder(
                self.brain, self.dispatch, config_effective=self.config_effective)
            encode_result = encoder.run(proposals, community_state)

        # Failure must propagate BEFORE either suppression or scan completion.
        # No placement during a failed call is not an unplaceable judgment.
        if proposals and (not encode_result or encode_result.get('error')):
            return {'actions': (encode_result or {}).get('write_actions', 0),
                    'proposals': len(proposals),
                    'raw_proposals': len(raw_proposals),
                    'suppressed': suppressed_count,
                    'auto_archived': len(archived_ids),
                    'error': (encode_result or {}).get('error') or 'enrichment failed'}

        # Phase 2: mark unplaceable the pending nodes NOT actually placed into a
        # community this cycle. Read AFTER encode via get_communities_for, so it
        # observes real placement — correct under corridor-drop, quota deferral,
        # and encoder skips (none of which place a node). A marked node sleeps
        # until its 1-hop neighborhood fingerprint moves.
        self._mark_unplaced_pending(decode_result.get('pending_probes', []))

        if not proposals:
            return {'actions': 0, 'proposals': 0,
                    'raw_proposals': len(raw_proposals),
                    'suppressed': suppressed_count,
                    'auto_archived': len(archived_ids),
                    'communities': len(community_state),
                    'skipped': 'all proposals suppressed'}

        return {
            'actions': encode_result.get('write_actions', 0),
            'proposals': len(proposals),
            'raw_proposals': len(raw_proposals),
            'suppressed': suppressed_count,
            'auto_archived': len(archived_ids),
            'communities': len(community_state),
            'details': {
                'rounds': encode_result.get('rounds', 0),
                'actions': encode_result.get('actions', 0),
                'write_actions': encode_result.get('write_actions', 0),
            },
        }

    def _auto_archive_dead(self, dead_list):
        """Deterministically archive communities below the hard cohesion floor.

        These have essentially no internal edges left — not clusters anymore.
        Archived in code (no encoder round, no rejection fingerprint) so a
        community can never get stuck suppressed-but-undead: this is the seam
        below the encoder's low-cohesion judgment band. Mirrors _heal_graph's
        deterministic 0-member sweep, but keyed on int_frac (computed in the
        community decoder, not available in consolidation's _heal_graph).
        Returns the set of archived community ids.
        """
        archived = []
        if dead_list:
            # Through dispatch, not brain.archive_node directly — the direct
            # call left no trace (plan step 10). UNGUARDED encoder dispatch:
            # dead communities are by definition outside any cluster set an
            # archive_guard would allow, and the guard drops out-of-scope
            # archives while reporting success. The batch stamps
            # encoding_source + this run's chain via apply_encoder_attribution.
            # locked pre-filtered (same pattern as the supersession
            # heal): a guarded target would otherwise log archive_guarded
            # PLUS batch_op_failed every cycle, forever (review 2026-08-07).
            ph = ','.join('?' * len(dead_list))
            guarded = {r[0] for r in self.brain.conn.execute(
                'SELECT id FROM nodes WHERE id IN (%s) '
                'AND locked = 1' % ph,
                [d['id'] for d in dead_list]).fetchall()}
            dead_list = [d for d in dead_list if d['id'] not in guarded]
        if dead_list:
            dispatch = self._make_encoder_dispatch()
            ops = [{'op': 'archive', 'node_id': d['id'],
                    'reason': 'dead — internal cohesion %.2f below floor'
                              % d['int_frac']}
                   for d in dead_list]
            batch = dispatch('brain_batch', {'operations': ops}) or {}
            results = (batch.get('result') or {}).get('results', [])
            for d, r in zip(dead_list, results):
                if r.get('ok'):
                    archived.append(d)
                    print('[s2cd] auto-archived dead community "%s" (%s, int_frac %.2f)'
                          % (d['title'][:50], d['id'][:8], d['int_frac']), flush=True)
        if archived:
            self.trace('O', 'heal_archive',
                       'Auto-archived %d dead communities (int_frac < floor): %s' % (
                           len(archived),
                           ', '.join('%s (%.2f)' % (a['title'][:30], a['int_frac'])
                                     for a in archived[:5])),
                       metadata={'auto_archived': archived})
        return {a['id'] for a in archived}

    def _stamp_structural(self, structural, community_state):
        """Write the structural fields the decoder computed for every live
        community, where they moved since the last write.

        Runs on every cycle the decoder decodes, before the encoder, so its
        views read this cycle's values off the nodes, and whether or not any
        proposal survives — a community no proposal names stays current too.
        A cycle the decoder rests (nothing new to place) writes nothing. The
        encoder re-stamps what its batches change. Failure-isolated: a stamp
        failure never costs the cycle its encode.
        """
        stored = {c['id']: c['meta'] for c in community_state}
        live = {cid: f for cid, f in structural.items() if cid in stored}
        try:
            n = write_structural_fields(self.brain, self._make_encoder_dispatch(),
                                        live, stored, self.ENCODING_SOURCE)
        except Exception as e:
            self.brain._log_error('community_structural_stamp', e,
                                  'decode-seam structural stamp failed')
            return 0
        # Each write is its own node_revised trace; no run-level delta here —
        # the decoder's chain would read as a completed integration run.
        if n:
            print('[s2cd] structural stamp: %d of %d communities moved'
                  % (n, len(live)), flush=True)
        return n

    def _drop_proposals_for_archived(self, proposals, archived_ids):
        """Drop/trim proposals that target a community archived this cycle.

        The decoder builds add/drift/merge/health proposals before the
        dead-floor sweep runs, so some can reference a community we just
        archived. Re-pointing a node into an archived community would create a
        dangling edge; this removes those refs before the encoder sees them.
        """
        if not archived_ids:
            return proposals
        kept = []
        for p in proposals:
            t = p.get('type')
            if t == 'health_update' and p.get('community_id') in archived_ids:
                continue
            if t == 'merge_communities' and (
                    p.get('larger_id') in archived_ids
                    or p.get('smaller_id') in archived_ids):
                continue
            if t == 'add_to_existing':
                comms = [c for c in p.get('communities', [])
                         if c.get('id') not in archived_ids]
                if not comms:
                    continue
                p = {**p, 'communities': comms}
            elif t == 'drift':
                foreign = [f for f in p.get('foreign', [])
                           if f.get('id') not in archived_ids]
                if not foreign:
                    continue
                p = {**p, 'foreign': foreign}
            kept.append(p)
        return kept

    def _mark_unplaced_pending(self, probes):
        """Mark unplaceable the pending nodes that did NOT land in any community
        this cycle. Placement is read AFTER the encoder ran (get_communities_for),
        so corridor-dropped, quota-deferred, and encoder-skipped nodes — none of
        which were actually placed — are correctly marked rather than shielded
        (which would leave them pending every cycle, defeating the rest gate).
        """
        if not probes:
            return
        ids = [pr['node_id'] for pr in probes]
        placed = self.brain._graph.get_communities_for(ids)  # {id: [communities]}
        to_mark = [pr for pr in probes if not placed.get(pr['node_id'])]
        if to_mark:
            self._mark_unplaceable(to_mark)

    def _mark_unplaceable(self, probes):
        """Record `unplaceable` rejections, one current row per node.

        Holds write_lock — this is a foreground write on the shared brain.conn,
        same as every other S2 writer; without it the DELETE+INSERT can
        interleave with a concurrent client brain_batch on the connection. Drops
        each node's prior unplaceable fingerprint in a single batched DELETE
        (one scan, not one per node) before recording the fresh one, so
        s2_rejections keeps one row per node, not one per historical
        neighborhood-state. Lock + single record_rejections commit make the
        DELETE+INSERT atomic.
        """
        if not probes:
            return
        with self.brain.write_lock:
            clear_unplaceable_rejections(self.brain, probes)
            record_rejections(self.brain, probes)
