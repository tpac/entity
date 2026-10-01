"""Tests for the S2 consolidation decoder's scan gate + stamp timing.

Consolidation's expensive step is the embedding scan. The gate (2026-05-29)
turns a full O(graph) scan every idle cycle into: one cold-start that covers
the existing backlog, then incremental scans of only changed nodes, skipping
entirely when nothing changed. A similarity-threshold change forces a fresh
cold-start.

Stamp timing: the decoder captures the baseline (`_stamp`) but does NOT write
it. The orchestrator advances the last-run cutoff only AFTER a run fully
completes — never on a skip or a mid-run encoder failure — so failed work is
retried, not skipped past.

The incremental scan's no-miss property is guaranteed by construction — it
compares every changed node against ALL nodes (changed @ all.T). The
fingerprint suppression itself is covered by test_consolidation_fingerprint.py.
"""

import time
import unittest
import sys
import os
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestConsolidationScanGate(BrainTestBase):
    """Decoder-level: mode selection, change detection, skip. No encoder."""

    needs_embedder = False

    def _decoder(self):
        from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
        return ConsolidationDecoder(self.brain)

    def _mark_scanned_now(self, d):
        """Simulate 'we just scanned' — stamp last-run ts + current threshold."""
        time.sleep(0.02)
        self.brain.set_config(d.LAST_RUN_TS_KEY, str(time.time()))
        self.brain.set_config(d.LAST_THRESHOLD_KEY, str(d.config['similarity_threshold']))
        time.sleep(0.02)

    def test_cold_start_scans_and_defers_stamp(self):
        # No recorded timestamp → cold start → it scans (not skipped) and hands
        # a baseline up for the orchestrator to stamp. The decoder itself must
        # NOT write the cutoff. (No embeddings present → empty scan; we assert
        # control flow, not pair discovery.)
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        result = d.run()
        self.assertIsNone(result.get('skipped'))
        self.assertIsNotNone(result.get('_stamp'))
        self.assertIsNone(self.brain.get_config(d.LAST_RUN_TS_KEY))  # decoder didn't stamp

    def test_skips_when_nothing_changed(self):
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        self._mark_scanned_now(d)
        result = d.run()
        self.assertEqual(result.get('skipped'), 'no graph change')
        self.assertIsNone(result.get('_stamp'))  # nothing to stamp on a skip

    def test_new_node_triggers_scan(self):
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        self._mark_scanned_now(d)
        self.brain.remember(type='fact', title='b', content='c')  # after cutoff
        result = d.run()
        self.assertNotEqual(result.get('skipped'), 'no graph change')

    def test_threshold_change_forces_cold_start(self):
        d = self._decoder()
        self.brain.remember(type='fact', title='a', content='c')
        time.sleep(0.02)
        self.brain.set_config(d.LAST_RUN_TS_KEY, str(time.time()))
        self.brain.set_config(d.LAST_THRESHOLD_KEY, '0.99')  # differs from config default
        time.sleep(0.02)
        # No new nodes, but the threshold changed → must re-scan, not skip.
        result = d.run()
        self.assertNotEqual(result.get('skipped'), 'no graph change')

    def test_changed_ids_use_timestamps_not_just_s1e_traces(self):
        # MCP-created nodes carry no encoding_run trace; timestamp-based
        # detection must still catch them (the bug in the old trace-based impl).
        d = self._decoder()
        self.brain.remember(type='fact', title='old', content='c')
        time.sleep(0.02)
        cutoff = datetime.now(timezone.utc).isoformat()
        time.sleep(0.02)
        new_id = self.brain.remember(type='fact', title='new', content='c')['id']
        changed = d._get_changed_node_ids(cutoff)
        self.assertIn(new_id, changed)

    def test_changed_ids_exclude_community_nodes(self):
        d = self._decoder()
        time.sleep(0.02)
        cutoff = datetime.now(timezone.utc).isoformat()
        time.sleep(0.02)
        comm_id = self.brain.remember(type='community', title='comm', content='c')['id']
        fact_id = self.brain.remember(type='fact', title='f', content='c')['id']
        changed = d._get_changed_node_ids(cutoff)
        self.assertIn(fact_id, changed)
        self.assertNotIn(comm_id, changed)  # consolidation scans non-community only

    def test_access_mark_excluded_only_revise_enters_change_set(self):
        # Fix (2026-06-26): _get_changed_node_ids keys on created_at/revised_at,
        # NOT updated_at. A recalled node gets updated_at bumped by the
        # recall_write_queue access drain — that must NOT pull it into the
        # change set, or the incremental scan re-checks the whole graph every
        # cycle AND drops the node's suppression edge out of the reviewed set
        # (settled pairs re-surface). Only a real revise (revised_at) re-enters.
        d = self._decoder()
        access_id = self.brain.remember(type='fact', title='accessed', content='c')['id']
        revise_id = self.brain.remember(type='fact', title='revised', content='c')['id']
        untouched_id = self.brain.remember(type='fact', title='untouched', content='c')['id']
        time.sleep(0.02)
        cutoff = datetime.now(timezone.utc).isoformat()
        time.sleep(0.02)
        after = datetime.now(timezone.utc).isoformat()
        # Mirror the recall access drain (recall_write_queue): updated_at only.
        self.brain.conn.execute(
            "UPDATE nodes SET updated_at = ? WHERE id = ?", (after, access_id))
        # Mirror a real revise(): revised_at moves.
        self.brain.conn.execute(
            "UPDATE nodes SET revised_at = ? WHERE id = ?", (after, revise_id))
        self.brain.conn.commit()
        changed = d._get_changed_node_ids(cutoff)
        self.assertIn(revise_id, changed)        # revise re-enters
        self.assertNotIn(access_id, changed)     # access-only must NOT (the fix)
        self.assertNotIn(untouched_id, changed)  # untouched stays out


class TestConsolidationSuppressionSource(BrainTestBase):
    """The suppression set derives from the settlement aspect, not a literal.

    The drift this pins against: a verb added to the taxonomy's settlement
    role (e.g. `resolves`, 2026-08-11) that a hand-typed Python set never
    learns about — pairs settled by that verb re-propose forever.
    """

    needs_embedder = False

    def _decoder(self):
        from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
        return ConsolidationDecoder(self.brain)

    def test_suppression_derives_from_settlement_aspect(self):
        d = self._decoder()
        derived = d._suppression_relations()
        self.assertTrue(derived)
        self.assertEqual(
            derived, set(self.brain.aspects.settlement.edge_relations))
        self.assertIn('resolves', derived)  # the original gap

    def test_encoder_payload_renders_decoder_reviewed_set(self):
        # The payload's 'Settlement relations' line and the decoder's scan
        # filter come from ONE derivation — the encoder must describe to
        # Sonnet exactly the set the decoder suppressed with.
        from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
        from servers.scales.s2.consolidation_contract import (
            CONSOLIDATION, suppression_relations)
        enc = ConsolidationEncoder(self.brain, config=CONSOLIDATION)
        cluster = {
            'nodes': [], 'size': 0, 'pre_class': 'needs_judgment',
            'content_cosine_max': 0.9, 'title_cosine_max': 0.9,
            'node_details': {}, 'co_recall_count': 0, 'judge_preference': {},
            'catalog_blind': {}, 'shared_edge_count': 0,
            'same_community': False, 'has_correction_edge': False,
        }
        payload = enc._format_clusters([cluster])
        expected = ', '.join(sorted(suppression_relations(self.brain)))
        first_line = payload.splitlines()[0]
        self.assertIn('Settlement relations', first_line)
        self.assertIn(expected, first_line)
        self.assertEqual(
            suppression_relations(self.brain),
            self._decoder()._suppression_relations())

    def test_settlement_does_not_steal_primary_homes(self):
        # settlement multi-homes verbs owned by earlier aspects. The primary
        # (first-claimant) reverse lookup must be unaffected — similar_to
        # stays generic_relation, so community typed adjacency keeps
        # skipping it (the regression test_community_health_seam caught).
        pm = self.brain.aspects.primary_edge_map()
        self.assertEqual(pm['similar_to'], 'generic_relation')
        self.assertEqual(pm['corrects'], 'correction_improvement')
        self.assertNotIn('settlement', pm.values())

    def test_contract_fallback_mirrors_seed_settlement(self):
        # The degraded-registry fallback must not drift from the seed —
        # a fresh install and a broken registry should suppress identically.
        import json
        from servers.aspect_store import SEED_ASPECTS_JSON_PATH
        from servers.scales.s2.consolidation_contract import CONSOLIDATION
        from servers.aspect_store import member_name
        seed = json.load(open(SEED_ASPECTS_JSON_PATH))
        self.assertEqual(
            set(CONSOLIDATION['suppression_relations']),
            {member_name(m, 'edge_relations') for m in seed['settlement']['edge_relations']})

    def test_empty_settlement_falls_back_to_contract(self):
        # Present-but-empty aspect → contract fallback. A MISSING aspect
        # raises AspectContractError instead (loud by design, same as
        # _has_correction_edge) — the coordinator contains it per-unit.
        d = self._decoder()
        # Instance attr shadows AspectRegistry.__getattr__ resolution.
        with mock.patch.object(self.brain.aspects, 'settlement',
                               mock.Mock(edge_relations=()), create=True):
            self.assertEqual(
                d._suppression_relations(),
                set(d.config['suppression_relations']))


class TestConsolidationStampTiming(BrainTestBase):
    """Orchestrator-level: the cutoff advances ONLY after a run completes."""

    needs_embedder = False

    def _unit(self):
        from servers.scales.s2.consolidation import Consolidation
        return Consolidation(self.brain)

    def test_completed_empty_scan_advances_cutoff(self):
        # Empty brain (no embeddings) → decode cold-start → no candidates →
        # the run completed, so the orchestrator stamps. No encoder involved.
        u = self._unit()
        result = u.run()
        self.assertIsNone(result.get('skipped'))
        self.assertIsNotNone(self.brain.get_config(u.LAST_RUN_TS_KEY))

    def test_skip_does_not_advance_cutoff(self):
        u = self._unit()
        self.brain.remember(type='fact', title='a', content='c')
        time.sleep(0.02)
        ts = str(time.time())
        self.brain.set_config(u.LAST_RUN_TS_KEY, ts)
        self.brain.set_config(u.LAST_THRESHOLD_KEY, str(u.config['similarity_threshold']))
        time.sleep(0.02)
        result = u.run()
        self.assertEqual(result.get('skipped'), 'no graph change')
        self.assertEqual(self.brain.get_config(u.LAST_RUN_TS_KEY), ts)  # unchanged

    def test_encoder_failure_does_not_advance_cutoff(self):
        # The crux of the fix: if the encoder fails mid-run, the cutoff must NOT
        # advance — the work is retried next cycle, not skipped past.
        from servers.scales.s2 import consolidation as consol_mod
        u = self._unit()
        fake_decode = {
            'clusters': [{'nodes': ['n1', 'n2'],
                          'node_details': {'n1': {'updated_at': ''},
                                           'n2': {'updated_at': ''}},
                          'pre_class': 'needs_judgment'}],
            'stats': {},
            '_stamp': {'ts': 12345.0, 'threshold': '0.89'},
        }
        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=fake_decode), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               return_value=None):
            result = u.run()
        self.assertEqual(result.get('error'), 'encoding failed')
        self.assertIsNone(self.brain.get_config(u.LAST_RUN_TS_KEY))  # NOT stamped


class TestConsolidationAbsorbDetection(BrainTestBase):
    """Orchestrator-level: a successful ABSORB archives the absorbed peer but
    writes NO similar_to/consolidated_into edge. Edge-only detection would read
    that as a SKIP and stamp a false rejection fingerprint. The archived-member
    snapshot is the fix — an archived member ⇒ the cluster was merged ⇒ handled.
    """

    needs_embedder = False

    def _unit(self):
        from servers.scales.s2.consolidation import Consolidation
        return Consolidation(self.brain)

    def _rejection_count(self):
        return self.brain.conn.execute(
            "SELECT COUNT(*) FROM s2_rejections "
            "WHERE integration_unit = 's2:consolidation'").fetchone()[0]

    def _fake_decode(self, n1, n2, pre_class):
        return {
            'clusters': [{'nodes': [n1, n2],
                          'node_details': {n1: {'updated_at': ''},
                                           n2: {'updated_at': ''}},
                          'pre_class': pre_class}],
            'stats': {},
            '_stamp': {'ts': 12345.0, 'threshold': '0.89'},
        }

    def test_archived_member_is_handled_not_rejected(self):
        from servers.scales.s2 import consolidation as consol_mod
        n1 = self.brain.remember(type='fact', title='survivor', content='c')['id']
        n2 = self.brain.remember(type='fact', title='absorbed', content='c')['id']
        u = self._unit()

        def _encoder_absorbs(clusters):
            # The absorb op archives the absorbed peer — simulate that effect.
            self.brain.conn.execute(
                "UPDATE nodes SET archived = 1 WHERE id = ?", (n2,))
            self.brain.conn.commit()
            return {'write_actions': 1, 'rounds': 2, 'actions': 1,
                    'action_details': [{'tool': 'brain_batch', 'input': {
                        'operations': [{'op': 'absorb', 'survivor_id': n1,
                                        'absorbed_id': n2}]}}]}

        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=self._fake_decode(n1, n2, 'likely_consolidate')), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               side_effect=_encoder_absorbs):
            result = u.run()

        self.assertEqual(result.get('skipped_recorded'), 0)  # merge, not a SKIP
        self.assertEqual(self._rejection_count(), 0)          # no false fingerprint
        self.assertIsNotNone(self.brain.get_config(u.LAST_RUN_TS_KEY))  # cutoff advanced

    def test_genuine_skip_still_recorded(self):
        # Negative control: encoder archives nothing and writes no edge → a real
        # SKIP → fingerprint stamped. Proves the detector didn't go blind.
        from servers.scales.s2 import consolidation as consol_mod
        n1 = self.brain.remember(type='fact', title='a', content='c')['id']
        n2 = self.brain.remember(type='fact', title='b', content='c')['id']
        u = self._unit()
        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=self._fake_decode(n1, n2, 'needs_judgment')), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               return_value={'write_actions': 0, 'rounds': 1,
                                             'actions': 0, 'action_details': []}):
            result = u.run()
        self.assertEqual(result.get('skipped_recorded'), 1)   # real skip stamped
        self.assertEqual(self._rejection_count(), 1)

    def test_rejected_batch_call_is_retried_not_stamped(self):
        # The encoder called brain_batch with NO operations: dispatch rejects
        # the call, nothing is written, and the call names no node ids — the
        # invalid-op shield can't attribute it to a cluster. The orchestrator
        # must treat the un-acted-on cluster as thwarted: no fingerprint,
        # baseline NOT advanced, so the cluster returns next cycle.
        from servers.scales.s2 import consolidation as consol_mod
        n1 = self.brain.remember(type='fact', title='left', content='c')['id']
        n2 = self.brain.remember(type='fact', title='right', content='c')['id']
        u = self._unit()
        rejected_call = {'write_actions': 1, 'rounds': 2, 'actions': 1,
                         'action_details': [{'tool': 'brain_batch',
                                             'input': {'operations': []},
                                             'error': 'operations array is required'}]}
        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=self._fake_decode(n1, n2, 'needs_judgment')), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               return_value=rejected_call):
            result = u.run()
        self.assertEqual(result.get('rejected_call_clusters'), 1)
        self.assertEqual(self._rejection_count(), 0)          # no fingerprint
        self.assertIsNone(self.brain.get_config(u.LAST_RUN_TS_KEY))  # retry
        self.assertEqual(self.brain.get_config(u.THWARTED_STREAK_KEY), '1')

    def test_rejected_call_shield_exempts_acted_on_clusters(self):
        # A stray rejected call must not unstamp a cluster a real op touched
        # in the same run — the shield is scoped by valid-op attribution.
        from servers.scales.s2 import consolidation as consol_mod
        n1 = self.brain.remember(type='fact', title='kept a', content='c')['id']
        n2 = self.brain.remember(type='fact', title='kept b', content='c')['id']
        u = self._unit()
        details = [
            {'tool': 'brain_batch',
             'input': {'operations': [{'op': 'connect', 'source_id': n1,
                                       'target_id': n2,
                                       'relation': 'addresses'}]}},
            {'tool': 'brain_batch', 'input': {'operations': []},
             'error': 'operations array is required'},
        ]
        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=self._fake_decode(n1, n2, 'needs_judgment')), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               return_value={'write_actions': 2, 'rounds': 2,
                                             'actions': 2,
                                             'action_details': details}):
            result = u.run()
        self.assertEqual(result.get('rejected_call_clusters'), 0)
        self.assertEqual(self._rejection_count(), 1)   # acted-on → stamped
        self.assertIsNotNone(self.brain.get_config(u.LAST_RUN_TS_KEY))

    def test_rejected_call_giveup_stamps_after_retry_limit(self):
        # The shield is bounded: after THWARTED_RETRY_LIMIT consecutive
        # shielded runs, the run stamps anyway and the baseline advances —
        # a persistently rejecting encoder must not pin the unit forever.
        from servers.scales.s2 import consolidation as consol_mod
        from servers.scales.s2.rejection_table import THWARTED_RETRY_LIMIT
        n1 = self.brain.remember(type='fact', title='p', content='c')['id']
        n2 = self.brain.remember(type='fact', title='q', content='c')['id']
        u = self._unit()
        rejected_call = {'write_actions': 1, 'rounds': 2, 'actions': 1,
                         'action_details': [{'tool': 'brain_batch',
                                             'input': {'operations': []},
                                             'error': 'operations array is required'}]}
        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=self._fake_decode(n1, n2, 'needs_judgment')), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               return_value=rejected_call):
            for _ in range(THWARTED_RETRY_LIMIT):
                u.run()
            self.assertEqual(self._rejection_count(), 0)   # still shielded
            result = u.run()                               # limit exceeded
        self.assertEqual(result.get('rejected_call_clusters'), 0)
        self.assertEqual(self._rejection_count(), 1)       # stamped anyway
        self.assertIsNotNone(self.brain.get_config(u.LAST_RUN_TS_KEY))
        self.assertEqual(self.brain.get_config(u.THWARTED_STREAK_KEY), '0')

    def test_invalid_op_giveup_stamps_after_retry_limit(self):
        # The bound covers BOTH thwarted causes: a persistently invalid-op
        # encoder (concept verbs every run) must also stamp after
        # THWARTED_RETRY_LIMIT shielded runs instead of pinning the unit
        # forever (Fix #1's retry, now bounded).
        from servers.scales.s2 import consolidation as consol_mod
        from servers.scales.s2.rejection_table import THWARTED_RETRY_LIMIT
        n1 = self.brain.remember(type='fact', title='x', content='c')['id']
        n2 = self.brain.remember(type='fact', title='y', content='c')['id']
        u = self._unit()
        invalid_run = {'write_actions': 1, 'rounds': 2, 'actions': 1,
                       'action_details': [{'tool': 'brain_batch', 'input': {
                           'operations': [{'op': 'consolidate',
                                           'node_id': n1}]}}]}
        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=self._fake_decode(n1, n2, 'needs_judgment')), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               return_value=invalid_run):
            for _ in range(THWARTED_RETRY_LIMIT):
                result = u.run()
                self.assertEqual(result.get('invalid_op_clusters'), 1)
            self.assertEqual(self._rejection_count(), 0)   # still shielded
            self.assertIsNone(self.brain.get_config(u.LAST_RUN_TS_KEY))
            result = u.run()                               # limit exceeded
        self.assertEqual(result.get('invalid_op_clusters'), 0)
        self.assertEqual(self._rejection_count(), 1)       # stamped anyway
        self.assertIsNotNone(self.brain.get_config(u.LAST_RUN_TS_KEY))
        self.assertEqual(self.brain.get_config(u.THWARTED_STREAK_KEY), '0')

    def test_invalid_op_cluster_with_archived_member_still_retried(self):
        # A cluster can carry BOTH a landed absorb and a thwarted invalid op.
        # The invalid-op check must run before the archived exclusion, or the
        # cluster is treated as resolved and the baseline advances past the
        # thwarted half.
        from servers.scales.s2 import consolidation as consol_mod
        n1 = self.brain.remember(type='fact', title='surv', content='c')['id']
        n2 = self.brain.remember(type='fact', title='abs', content='c')['id']
        n3 = self.brain.remember(type='fact', title='thwart', content='c')['id']
        u = self._unit()
        fake_decode = {
            'clusters': [{'nodes': [n1, n2, n3],
                          'node_details': {nid: {'updated_at': ''}
                                           for nid in (n1, n2, n3)},
                          'pre_class': 'likely_consolidate'}],
            'stats': {},
            '_stamp': {'ts': 12345.0, 'threshold': '0.89'},
        }

        def _encoder_mixed(clusters):
            self.brain.conn.execute(
                "UPDATE nodes SET archived = 1 WHERE id = ?", (n2,))
            self.brain.conn.commit()
            return {'write_actions': 1, 'rounds': 2, 'actions': 1,
                    'action_details': [{'tool': 'brain_batch', 'input': {
                        'operations': [
                            {'op': 'absorb', 'survivor_id': n1, 'absorbed_id': n2},
                            {'op': 'consolidate', 'node_id': n3},
                        ]}}]}

        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=fake_decode), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               side_effect=_encoder_mixed):
            result = u.run()
        self.assertEqual(result.get('invalid_op_clusters'), 1)
        self.assertEqual(self._rejection_count(), 0)
        self.assertIsNone(self.brain.get_config(u.LAST_RUN_TS_KEY))  # retry

    def test_edge_resolved_cluster_also_fingerprinted(self):
        # Policy change 2026-07-27 (journal finding #4): suppression follows
        # the encoder's DECISION, not its edge vocabulary. A KEEP that draws
        # an edge — whatever the verb (`resolves`, `depends_on`, ...) — must
        # still stamp a fingerprint, or verb-mismatched resolutions re-propose
        # every cycle. Under the old policy this cluster was exempt.
        from servers.scales.s2 import consolidation as consol_mod
        n1 = self.brain.remember(type='fact', title='kept a', content='c')['id']
        n2 = self.brain.remember(type='fact', title='kept b', content='c')['id']
        u = self._unit()

        def _encoder_keeps(clusters):
            # KEEP resolution with a verb the decoder's skip-list doesn't know.
            self.brain.connect(n1, n2, relation='resolves')
            return {'write_actions': 1, 'rounds': 2, 'actions': 1,
                    'action_details': [{'tool': 'brain_batch', 'input': {
                        'operations': [{'op': 'connect', 'source_id': n1,
                                        'target_id': n2,
                                        'relation': 'resolves'}]}}]}

        with mock.patch.object(consol_mod.ConsolidationDecoder, 'run',
                               return_value=self._fake_decode(n1, n2, 'needs_judgment')), \
             mock.patch.object(consol_mod.ConsolidationEncoder, 'run',
                               side_effect=_encoder_keeps):
            u.run()

        self.assertEqual(self._rejection_count(), 1)

        # The fingerprint must be re-derivable NEXT cycle: it was recorded
        # with post-encode updated_at, so a proposal rebuilt from the current
        # DB state must be filtered out.
        from servers.scales.s2.rejection_table import filter_rejected
        fresh = {nid: self.brain.conn.execute(
            'SELECT updated_at FROM nodes WHERE id = ?', (nid,)).fetchone()[0]
            for nid in (n1, n2)}
        proposal = {'type': 'consolidation_cluster',
                    'members': sorted([n1, n2]),
                    'member_updated_at': fresh}
        surviving, suppressed = filter_rejected(self.brain, [proposal])
        self.assertEqual(surviving, [],
                         'edge-resolved cluster re-proposed despite fingerprint')
        self.assertEqual(suppressed, 1)


if __name__ == '__main__':
    unittest.main()
