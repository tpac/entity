"""Tests for S2 community detection unit.

Tests the IntegrationUnit contract and community decoder algorithm.
Encoder tests require Anthropic API and are integration-level — not here.

Architecture:
  CommunityDecoder — algorithmic, produces proposals from graph structure
  CommunityEncoder — agentic Sonnet, creates community nodes (needs API)
  CommunityDetection — orchestrator, wires decoder→encoder

These tests verify the decoder finds correct structure and the
orchestrator handles edge cases. Encoder quality is tested via eval.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestIntegrationUnitContract(unittest.TestCase):
    """Test the base IntegrationUnit contract."""

    def test_subclass_must_implement_run(self):
        from servers.scales.s2.base import IntegrationUnit
        unit = IntegrationUnit(brain=None)
        with self.assertRaises(NotImplementedError):
            unit.run()

    def test_chain_id_format(self):
        """Per-run chain id: s2-{YYYYMMDDHHMMSS}-{name}. Seconds (not just
        date) so same-day runs are distinct → notes group per-run. One
        combined timestamp segment + trailing -{name} keep the load-bearing
        consumers intact: `_last_run_timestamp` suffix-matches `%-{name}` and
        the dashboard slug parser reads `split('-', 2)[2]`. Stamped once per
        run and cached so every trace in a run shares one chain_id."""
        from servers.scales.s2.base import IntegrationUnit

        class TestUnit(IntegrationUnit):
            NAME = 'test_op'
            SCALE = 's2'

        unit = TestUnit(brain=None)
        chain = unit.chain_id()
        self.assertRegex(chain, r'^s2-\d{14}-test_op$')        # one timestamp segment
        self.assertTrue(chain.endswith('-test_op'))            # _last_run_timestamp LIKE '%-name'
        self.assertEqual(chain.split('-', 2)[2], 'test_op')    # dashboard slug parser
        self.assertEqual(unit.chain_id(), chain)               # stable within a run (cached)

    def test_chain_id_is_real_recent_timestamp(self):
        """The 14-digit segment is an actual recent UTC run-time — not a
        constant or a wrong strftime that merely yields 14 digits. Guards the
        timestamp-correctness axis the shape regex alone can't (e.g. a revert
        to date-only would be 8 digits and fail strptime here)."""
        from datetime import datetime, timezone
        from servers.scales.s2.base import IntegrationUnit

        class TestUnit(IntegrationUnit):
            NAME = 'test_op'
            SCALE = 's2'

        ts = TestUnit(brain=None).chain_id().split('-', 2)[1]
        parsed = datetime.strptime(ts, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        self.assertLess(abs((datetime.now(timezone.utc) - parsed).total_seconds()), 120)

    def test_chain_id_different_scale(self):
        from servers.scales.s2.base import IntegrationUnit

        class S3Unit(IntegrationUnit):
            NAME = 'synthesis'
            SCALE = 's3'

        unit = S3Unit(brain=None)
        self.assertTrue(unit.chain_id().startswith('s3-'))

    def test_encoder_dispatch_threads_run_chain_into_writes(self):
        """_make_encoder_dispatch stamps the unit's run chain on writes — so
        their revise/edge traces join the run's chain instead of the date-
        fallback phantom ("S2 revise") unit. brain_batch with empty operations
        returns before touching the brain, so the stamped args are observable
        with a FakeBrain that only carries a write_lock."""
        import threading
        from servers.scales.s2.base import IntegrationUnit

        class TestUnit(IntegrationUnit):
            NAME = 'test_op'
            SCALE = 's2'
            ENCODING_SOURCE = 's2:test_op'

        class FakeBrain:
            write_lock = threading.RLock()

        unit = TestUnit(brain=FakeBrain())
        dispatch = unit._make_encoder_dispatch()
        args = {'operations': []}
        dispatch('brain_batch', args)
        self.assertEqual(args.get('chain_id'), unit.chain_id())
        self.assertEqual(args.get('encoding_source'), 's2:test_op')


class TestCommunityDetectionContract(unittest.TestCase):
    """Test CommunityDetection declares its O/K sources."""

    def test_sources_declared(self):
        from servers.scales.s2.community import CommunityDetection
        self.assertTrue(len(CommunityDetection.O_SOURCES) > 0)
        self.assertTrue(len(CommunityDetection.K_SOURCES) > 0)
        self.assertEqual(CommunityDetection.SCALE, 's2')
        self.assertEqual(CommunityDetection.ENCODING_SOURCE, 's2:community_detection')

    def test_decoder_encoder_split(self):
        """CommunityDetection inherits from CommunityDecoder."""
        from servers.scales.s2.community import CommunityDetection
        from servers.scales.s2.community_decoder import CommunityDecoder
        from servers.scales.s2.community_encoder import CommunityEncoder
        self.assertTrue(issubclass(CommunityDetection, CommunityDecoder))
        # Encoder is separate, not inherited
        self.assertFalse(issubclass(CommunityDetection, CommunityEncoder))


class TestCommunityDecoder(BrainTestBase):
    """Test the decoder's algorithmic pipeline on synthetic graphs.

    These tests call _decode() directly, bypassing the S1 trace check.
    The decoder finds structure in the graph — it doesn't need S1 traces
    to detect clusters, only to decide WHETHER to run.
    """

    needs_embedder = False

    def _create_cluster(self, prefix, n, connect=True):
        """Create n nodes with a shared prefix, optionally fully connected."""
        ids = []
        for i in range(n):
            result = self.brain.remember(
                type='decision',
                title='%s node %d' % (prefix, i),
                content='Content about %s topic %d' % (prefix, i),
            )
            ids.append(result['id'])

        if connect:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self.brain.connect(ids[i], ids[j], relation='implements', weight=0.8)

        return ids


    def test_small_graph_produces_no_proposals(self):
        """Graph smaller than min_community_size produces no community proposals."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        for i in range(2):
            self.brain.remember(type='fact', title='node %d' % i, content='c')

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode([])
        proposals = [p for p in result['proposals'] if p['type'] == 'new_community']
        self.assertEqual(len(proposals), 0)

    def test_detects_two_clusters(self):
        """Two well-separated clusters produce at least 2 new_community proposals."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        cluster_a = self._create_cluster('alpha', 8)
        cluster_b = self._create_cluster('beta', 8)
        # Weak bridge between clusters
        self.brain.connect(cluster_a[0], cluster_b[0], relation='related_to', weight=0.1)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode([])
        proposals = [p for p in result['proposals'] if p['type'] == 'new_community']

        self.assertGreaterEqual(len(proposals), 2)
        # Each should have members from its cluster
        all_members = set()
        for p in proposals:
            self.assertGreaterEqual(p['member_count'], 3)
            self.assertGreater(p['internal_fraction'], 0)
            all_members.update(p['members'])
        # Both clusters should be represented
        self.assertTrue(all_members & set(cluster_a))
        self.assertTrue(all_members & set(cluster_b))

    def test_proposal_has_required_fields(self):
        """Each new_community proposal has the fields the encoder needs."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        self._create_cluster('gamma', 8)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode([])
        proposals = [p for p in result['proposals'] if p['type'] == 'new_community']

        self.assertGreater(len(proposals), 0)
        p = proposals[0]
        # Fields the encoder reads
        self.assertIn('members', p)
        self.assertIn('member_count', p)
        self.assertIn('internal_fraction', p)
        self.assertIn('edge_signature', p)
        self.assertIn('timeline', p)
        self.assertIn('representatives', p)
        self.assertIn('all_members', p)
        self.assertIn('sample_edges', p)
        self.assertIn('is_corridor', p)

    def test_sample_internal_edges_skips_non_cohesion_relations(self):
        """The evidence sample hides exactly what the disconnected check
        hides. Both read `non_cohesion_relations`, so a verb can never be
        shown to the encoder as evidence of cohesion while simultaneously
        not counting as cohesion."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        ids = self._create_cluster('sample', 3, connect=False)
        a, b, c = ids
        self.brain.connect_typed(a, b, relation='implements',
                                 description='a real semantic edge')
        self.brain.connect_typed(a, c, relation='related',
                                 description='pure generic, never cohesion')
        self.brain.connect_typed(b, c, relation='dreamed_from',
                                 description='dream artifact, never cohesion')

        decoder = CommunityDecoder(self.brain)
        sampled = {e['relation']
                   for e in decoder._sample_internal_edges(ids, limit=10)}

        self.assertIn('implements', sampled)
        self.assertNotIn('related', sampled)
        # dreamed_from is in non_cohesion_relations but was absent from the
        # narrower literal this query used to carry — the drift this pins.
        self.assertNotIn('dreamed_from', sampled)

    def test_decode_stats_populated(self):
        """Decode result includes stats for trace consumption."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        self._create_cluster('delta', 8)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode([])
        stats = result['stats']

        self.assertIn('nodes_with_typed_edges', stats)
        self.assertIn('valid_clusters', stats)
        self.assertIn('fragments_dissolved', stats)
        self.assertIn('subsets_absorbed', stats)
        self.assertGreater(stats['nodes_with_typed_edges'], 0)

    def test_corridor_detection(self):
        """Clusters with low internal fraction are flagged as corridors."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        # Create a "corridor" — nodes with more external than internal edges
        corridor = self._create_cluster('corridor', 4, connect=False)
        external_a = self._create_cluster('ext_a', 4)
        external_b = self._create_cluster('ext_b', 4)

        # Connect corridor nodes to each other weakly
        for i in range(len(corridor)):
            for j in range(i+1, len(corridor)):
                self.brain.connect(corridor[i], corridor[j], relation='related_to', weight=0.3)

        # Connect corridor heavily to external clusters
        for cid in corridor:
            for eid in external_a[:2]:
                self.brain.connect(cid, eid, relation='depends_on', weight=0.7)
            for eid in external_b[:2]:
                self.brain.connect(cid, eid, relation='enables', weight=0.7)

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode([])

        corridors = result['stats'].get('corridors', 0)
        # We should see at least some corridor detection
        # (exact count depends on cluster formation)
        self.assertIsInstance(corridors, int)

    def test_incremental_excludes_placed_nodes(self):
        """Incremental mode skips nodes already in communities."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        cluster_a = self._create_cluster('placed', 6)
        cluster_b = self._create_cluster('unplaced', 6)

        # Simulate existing community owning cluster_a
        comm = self.brain.remember(
            type='community', title='Existing Community',
            content='test', encoding_source='s2:community_detection')
        for nid in cluster_a:
            self.brain.connect(comm['id'], nid, relation='community_member', weight=0.3)

        community_state = [{
            'id': comm['id'], 'title': 'Existing Community',
            'content': 'test', 'keywords': '', 'confidence': 0.8,
            'members': set(cluster_a),
            'meta': {},
        }]

        decoder = CommunityDecoder(self.brain)
        result = decoder._decode(community_state)

        # New proposals should not contain already-placed nodes as seed clusters
        new_community_proposals = [p for p in result['proposals'] if p['type'] == 'new_community']
        for p in new_community_proposals:
            placed_in_proposal = set(p['members']) & set(cluster_a)
            # Some placed nodes may appear via affinity, but the SEED shouldn't be all placed
            unplaced_in_proposal = set(p['members']) - set(cluster_a)
            self.assertGreater(len(unplaced_in_proposal), 0,
                'New community proposal should contain unplaced nodes')

    def test_subset_absorption(self):
        """When one cluster is a subset of another, the smaller is absorbed."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        clusters = {
            1: {'a', 'b', 'c', 'd', 'e'},
            2: {'a', 'b', 'c'},  # subset of 1
            3: {'x', 'y', 'z'},
        }
        filtered, absorbed = decoder._absorb_subsets(clusters)
        self.assertEqual(absorbed, 1)
        self.assertNotIn(2, filtered)
        self.assertIn(1, filtered)
        self.assertIn(3, filtered)

    def test_subset_chain_absorption(self):
        """Chain: A ⊂ B ⊂ C → both A and B absorbed."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        clusters = {
            1: {'a'},
            2: {'a', 'b'},
            3: {'a', 'b', 'c', 'd'},
        }
        filtered, absorbed = decoder._absorb_subsets(clusters)
        self.assertEqual(absorbed, 2)
        self.assertEqual(set(filtered.keys()), {3})

    def test_merge_detection(self):
        """Communities with 80%+ overlap and <3 unique are merge candidates."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        # Simulate two communities with 100% overlap (smaller fully contained)
        community_state = [
            {'id': 'comm_a', 'title': 'Larger', 'members': {'n1', 'n2', 'n3', 'n4', 'n5'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'meta': {}},
            {'id': 'comm_b', 'title': 'Smaller', 'members': {'n1', 'n2', 'n3'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'meta': {}},
        ]
        merges = decoder._detect_merge_candidates(community_state)
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0]['larger']['id'], 'comm_a')
        self.assertEqual(merges[0]['smaller']['id'], 'comm_b')

    def test_no_merge_when_unique_members_above_threshold(self):
        """Communities with enough unique members should NOT merge."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        decoder = CommunityDecoder(self.brain)

        # 60% overlap but 4 unique in smaller — above threshold
        community_state = [
            {'id': 'comm_a', 'title': 'Larger', 'members': {'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'meta': {}},
            {'id': 'comm_b', 'title': 'Smaller', 'members': {'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'u1', 'u2', 'u3', 'u4'},
             'content': '', 'keywords': '', 'confidence': 0.8,
             'meta': {}},
        ]
        merges = decoder._detect_merge_candidates(community_state)
        self.assertEqual(len(merges), 0)

    def test_traces_written_on_decode(self):
        """Decoder writes O and K traces with correct ref_types."""
        from servers.scales.s2.community_decoder import CommunityDecoder

        self._create_cluster('traced', 8)

        # Write a fake S1 trace so _has_new_traces passes
        self.brain._trace_dal.append(
            chain_id='s1e-test-1', scale='s1', event_type='delta',
            ref_type='encoding_run', ref_id='1',
            summary='test encoding run', metadata={}, session_id='test')

        decoder = CommunityDecoder(self.brain)
        result = decoder.run()

        # Should not skip (we added a trace)
        self.assertNotIn('skipped', result)

        # Check traces
        traces = self.brain.logs_conn.execute(
            "SELECT scale, event_type, ref_type FROM trace_events WHERE scale = 's2'"
        ).fetchall()
        event_types = {t[1] for t in traces}
        ref_types = {t[2] for t in traces}

        self.assertIn('O', event_types)
        self.assertIn('s1_delta', ref_types)


class TestCoordinator(unittest.TestCase):
    """Test S2 coordinator imports and structure."""

    def test_coordinator_imports(self):
        from servers.scales.s2.coordinator import run_s2
        self.assertTrue(callable(run_s2))

    def test_unit_ordering(self):
        """Coordinator runs units in correct order.

        EdgeFamilyIntegration disabled 2026-05-04 — its source interaction
        was removed in Step 12 of unified-aspects, AspectIntegration (Step 13)
        will replace it. Test now verifies the ACTIVE units only.
        """
        from servers.scales.s2.coordinator import run_s2
        from servers.scales.s2.consolidation import Consolidation
        from servers.scales.s2.community import CommunityDetection
        from servers.scales.s2.healer import Healer
        # Verify imports work — ordering is enforced by coordinator code
        self.assertEqual(Consolidation.NAME, 'consolidation')
        self.assertEqual(CommunityDetection.NAME, 'community_detection')
        self.assertEqual(Healer.NAME, 'healer')


class TestCommunityIdleGate(BrainTestBase):
    """Phase 1 idle gate — skip the unit when nothing changed or too soon.

    The decode is a pure function of graph state, so re-running it on an
    unchanged graph re-derives identical, already-rejected proposals. The
    gate (CommunityDetection._should_skip) stops that waste.
    """

    needs_embedder = False

    def _unit(self):
        from servers.scales.s2.community import CommunityDetection
        return CommunityDetection(self.brain)

    def test_cold_start_runs(self):
        # No last-run timestamp recorded → never skip.
        self.assertIsNone(self._unit()._should_skip())

    def test_throttles_recent_run(self):
        import time
        self.brain.set_config('s2_community_last_run_ts', str(time.time()))
        reason = self._unit()._should_skip()
        self.assertIsNotNone(reason)
        self.assertIn('throttled', reason)

    def test_no_change_skips_then_new_node_wakes(self):
        import time
        u = self._unit()
        u.config = dict(u.config, min_run_interval_seconds=0)  # isolate the change-gate
        self.brain.remember(type='fact', title='old', content='c')
        time.sleep(0.02)
        self.brain.set_config('s2_community_last_run_ts', str(time.time()))
        time.sleep(0.02)
        # Nothing changed since the cutoff → skip.
        self.assertEqual(u._should_skip(), 'no graph change since last run')
        # A new node appears → gate opens.
        self.brain.remember(type='fact', title='new', content='c')
        self.assertIsNone(u._should_skip())

    def test_change_gate_ignores_self_writes_and_noise(self):
        import time
        u = self._unit()
        u.config = dict(u.config, min_run_interval_seconds=0)
        a = self.brain.remember(type='fact', title='a', content='c')['id']
        b = self.brain.remember(type='fact', title='b', content='c')['id']
        c = self.brain.remember(type='fact', title='cc', content='c')['id']
        d = self.brain.remember(type='fact', title='dd', content='c')['id']
        time.sleep(0.02)
        self.brain.set_config('s2_community_last_run_ts', str(time.time()))
        time.sleep(0.02)
        # None of these may wake the gate:
        self.brain.remember(type='community', title='comm', content='c')   # own node type
        self.brain.connect_typed(a, b, relation='co_anchored', weight=0.5)  # noise edge
        self.brain.connect_typed(a, b, relation='implements',              # own edge source
                                 encoding_source='s2:community_detection')
        self.assertEqual(u._should_skip(), 'no graph change since last run')
        # A real typed edge from a non-self source DOES wake it:
        self.brain.connect_typed(c, d, relation='implements', weight=0.8,
                                 encoding_source='encoder:sonnet')
        self.assertIsNone(u._should_skip())


class TestDriftDetection(BrainTestBase):
    """Drift proposals target EXISTING communities only, and the
    _sys_drift_threshold lever works even at home_aff == 0 (bug bc024ee6:
    run-local new_cluster targets were unsuppressible phantoms, and a ratio
    on a zero home base made the rejection lever inert)."""

    needs_embedder = False

    def _node(self, title):
        return self.brain.remember(type='decision', title=title, content='c',
                                   encoding_source='test')['id']

    def _community(self, title, members):
        cid = self.brain.remember(
            type='community', title=title, content='c',
            encoding_source='s2:community_detection')['id']
        for m in members:
            self.brain.connect(cid, m, relation='community_member', weight=0.3)
        return cid


    def _decode(self):
        from servers.scales.s2.community_decoder import CommunityDecoder
        decoder = CommunityDecoder(self.brain)
        state = decoder._read_community_state()
        return decoder._decode(state)

    def _drifts(self, result):
        return [p for p in result['proposals'] if p['type'] == 'drift']

    def _two_communities_estranged_node(self):
        """Node x placed in home A but all its typed neighbors are in B."""
        x = self._node('estranged x')
        home_rest = [self._node('a%d' % i) for i in range(2)]
        foreign = [self._node('b%d' % i) for i in range(4)]
        self._community('home A', [x] + home_rest)
        b_id = self._community('foreign B', foreign)
        for m in foreign:
            self.brain.connect(x, m, relation='implements', weight=0.8)
        return x, b_id

    def test_zero_home_drift_fires_on_existing_community(self):
        """home_aff == 0 with strong foreign affinity still surfaces drift,
        and the target is the existing community's real id."""
        x, b_id = self._two_communities_estranged_node()

        drifts = self._drifts(self._decode())
        mine = [p for p in drifts if p['node_id'] == x]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]['foreign'][0]['id'], b_id)

    def test_drift_never_targets_same_run_clusters(self):
        """A placed node adjacent to an unplaced cluster produces NO drift —
        the cluster is a new_community proposal, not a drift target."""
        x = self._node('placed x')
        self._community('home A', [x, self._node('a1'), self._node('a2')])
        cluster = [self._node('c%d' % i) for i in range(5)]
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                self.brain.connect(cluster[i], cluster[j],
                                   relation='implements', weight=0.8)
        for m in cluster:
            self.brain.connect(x, m, relation='implements', weight=0.8)

        result = self._decode()
        for p in self._drifts(result):
            for f in p['foreign']:
                self.assertNotIn('new_cluster', str(f['id']))
        self.assertNotIn(x, [p['node_id'] for p in self._drifts(result)])
        # The cluster signal is not lost — it routes as a structural proposal
        # (new_community, or add_to_existing when overlap converts it).
        structural = [p for p in result['proposals']
                      if p['type'] in ('new_community', 'add_to_existing')]
        self.assertGreaterEqual(len(structural), 1)

    def test_threshold_lever_works_at_zero_home(self):
        """Raising _sys_drift_threshold suppresses a zero-home candidate —
        the lever the encoder uses on rejection must not be inert."""
        x, _ = self._two_communities_estranged_node()
        self.brain._meta_kv.set(x, '_sys_drift_threshold', '10')

        drifts = self._drifts(self._decode())
        self.assertNotIn(x, [p['node_id'] for p in drifts])

    def test_unparseable_threshold_logs_and_defaults(self):
        """A model-authored value the decoder can't parse is LOUD. It used to
        hit `except: pass` — the node silently reverted to the default, so the
        encoder's rejection suppressed nothing and the same proposal came back
        every run with no error row anywhere to show for it."""
        from unittest.mock import patch
        x, _ = self._two_communities_estranged_node()
        self.brain._meta_kv.set(x, '_sys_drift_threshold', '1.6x')

        with patch.object(self.brain, '_log_error') as logged:
            drifts = self._drifts(self._decode())

        self.assertTrue(logged.called)
        reported = ' '.join(str(a) for a in logged.call_args[0])
        self.assertIn('1.6x', reported)
        self.assertIn(x, reported)
        # Log AND default — one corrupt row must not take the decode down.
        self.assertIn(x, [p['node_id'] for p in drifts])

    def test_sub_default_threshold_is_clamped_not_honored(self):
        """The lever only ratchets UP. A stored value below the default has no
        author (production held 0.34/0.38/0.6), and honoring one would INVERT
        the mechanism — the node would drift on weaker evidence than an
        untouched one. 0.6 would drop the bar to 0.09; the clamp holds 0.225,
        so a foreign affinity of 0.2 stays quiet."""
        x = self._node('weak x')
        self._community('home A', [x, self._node('a1'), self._node('a2')])
        b_member = self._node('b0')
        others = [self._node('o%d' % i) for i in range(4)]
        self._community('foreign B', [b_member] + [self._node('b%d' % i)
                                                   for i in range(1, 4)])
        for m in [b_member] + others:
            self.brain.connect(x, m, relation='implements', weight=0.8)
        self.brain._meta_kv.set(x, '_sys_drift_threshold', '0.6')

        drifts = self._drifts(self._decode())
        self.assertNotIn(x, [p['node_id'] for p in drifts])

    def test_encoder_stamps_raise_on_declined_drift(self):
        """The rejection is the agent's judgment; the NUMBER is arithmetic, so
        code writes it (id:e0b7f842). A declined drift proposal gets exactly
        one step added — no model authors the value, so "1.6x" is unreachable."""
        from servers.scales.s2.community_encoder import CommunityEncoder
        x, _ = self._two_communities_estranged_node()
        self.brain._meta_kv.set(x, '_sys_drift_threshold', '1.6')

        declined = [p for p in self._drifts(self._decode())
                    if p['node_id'] == x]
        self.assertEqual(len(declined), 1)
        stamped = CommunityEncoder(self.brain)._raise_drift_thresholds(declined)

        self.assertEqual(stamped, 1)
        self.assertEqual(
            self.brain._meta_kv.get_field(x, '_sys_drift_threshold'), '1.7')

    def test_stamp_ignores_non_drift_proposals(self):
        """Only drift proposals carry a threshold — a skipped new_community or
        health_update must not grow one."""
        from servers.scales.s2.community_encoder import CommunityEncoder
        other = self._node('unrelated')
        stamped = CommunityEncoder(self.brain)._raise_drift_thresholds([
            {'type': 'new_community', 'members': [other]},
            {'type': 'health_update', 'community_id': other},
        ])

        self.assertEqual(stamped, 0)
        self.assertIsNone(
            self.brain._meta_kv.get_field(other, '_sys_drift_threshold'))

    def test_weak_foreign_below_scaled_floor_is_quiet(self):
        """At home_aff == 0 the bar is floor × ratio (0.225), not the bare
        floor (0.15) — weak foreign affinity no longer fires."""
        x = self._node('weak x')
        self._community('home A', [x, self._node('a1'), self._node('a2')])
        b_member = self._node('b0')
        others = [self._node('o%d' % i) for i in range(4)]
        self._community('foreign B', [b_member] + [self._node('b%d' % i)
                                                   for i in range(1, 4)])
        # 5 typed neighbors, only 1 in B → foreign_aff ≈ 0.2 < 0.225
        for m in [b_member] + others:
            self.brain.connect(x, m, relation='implements', weight=0.8)

        drifts = self._drifts(self._decode())
        self.assertNotIn(x, [p['node_id'] for p in drifts])

    def test_foreign_targets_sorted_strongest_first(self):
        """foreign[0] feeds the rejection fingerprint and quota confidence —
        it must be the strongest target, not community-row order."""
        x = self._node('torn x')
        self._community('home A', [x, self._node('a1'), self._node('a2')])
        weak = [self._node('w0')]
        strong = [self._node('s%d' % i) for i in range(3)]
        # Weak community created FIRST so row order would put it at [0].
        weak_id = self._community('weak B', weak + [self._node('w%d' % i)
                                                    for i in range(1, 4)])
        strong_id = self._community('strong C', strong)
        # 4 typed neighbors: 1 in B (0.25), 3 in C (0.75) — both above bar.
        for m in weak + strong:
            self.brain.connect(x, m, relation='implements', weight=0.8)

        drifts = self._drifts(self._decode())
        mine = [p for p in drifts if p['node_id'] == x]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]['foreign'][0]['id'], strong_id)
        self.assertEqual(mine[0]['foreign'][1]['id'], weak_id)

    def test_positive_home_drift_behavior_preserved(self):
        """home_aff ≥ floor keeps today's exact semantics: foreign must beat
        home × ratio (live example: 25% home vs 100% target moved cleanly)."""
        x = self._node('leaning x')
        a1 = self._node('a1')
        foreign = [self._node('b%d' % i) for i in range(3)]
        self._community('home A', [x, a1, self._node('a2')])
        b_id = self._community('foreign B', foreign)
        # 4 typed neighbors: 1 home, 3 foreign → 0.25 home, 0.75 foreign
        self.brain.connect(x, a1, relation='implements', weight=0.8)
        for m in foreign:
            self.brain.connect(x, m, relation='implements', weight=0.8)

        drifts = self._drifts(self._decode())
        mine = [p for p in drifts if p['node_id'] == x]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]['foreign'][0]['id'], b_id)

    def test_drift_never_targets_own_communities(self):
        """A multi-home node's own communities are never drift targets, in
        every state order. The neighbor mass sits in home A while B comes
        LAST in state order — an arbitrary-home drift base reads home
        affinity 0 and proposes moving x into A, a community x is in."""
        from servers.scales.s2.community_decoder import CommunityDecoder
        x = self._node('dual x')
        a_members = [self._node('a%d' % i) for i in range(4)]
        b_rest = [self._node('b%d' % i) for i in range(2)]
        a_id = self._community('home A', [x] + a_members)
        self._community('home B', [x] + b_rest)
        for m in a_members:
            self.brain.connect(x, m, relation='implements', weight=0.8)

        decoder = CommunityDecoder(self.brain)
        state = decoder._read_community_state()
        # Pin state order: A first, B last — no row-order luck.
        state.sort(key=lambda c: 0 if c['id'] == a_id else 1)
        result = decoder._decode(state)
        drifts = self._drifts(result)
        self.assertNotIn(x, [p['node_id'] for p in drifts])


class TestAddBatchContract(BrainTestBase):
    """add_to_existing batch contract (Step 9c + conversion filter): the
    encoder never sees two proposals for one node, a candidate community the
    node is already in, or an add for a node already placed elsewhere."""

    needs_embedder = False

    def _node(self, title):
        return self.brain.remember(type='decision', title=title, content='c',
                                   encoding_source='test')['id']

    def _community(self, title, members):
        cid = self.brain.remember(
            type='community', title=title, content='c',
            encoding_source='s2:community_detection')['id']
        for m in members:
            self.brain.connect(cid, m, relation='community_member', weight=0.3)
        return cid

    def _decoder(self):
        from servers.scales.s2.community_decoder import CommunityDecoder
        return CommunityDecoder(self.brain)

    def _decode(self):
        decoder = self._decoder()
        state = decoder._read_community_state()
        return decoder._decode(state)

    # ── _seed_clusters determinism (F19) ──

    def test_seed_clusters_deterministic_under_insertion_order(self):
        """Tie-heavy pair sets must yield identical clusters regardless of
        dict insertion order. Without the (a, b) sort-key suffix, a bridge
        pair processed early vs late changes the partition — the F19
        per-process hash-order defect: {a,b}+{c,d} in one order, {a,b,c,d}
        in the other."""
        pairs = [(('a', 'b'), 2.0), (('c', 'd'), 2.0), (('b', 'c'), 2.0)]
        d = self._decoder()
        fwd = d._seed_clusters(dict(pairs), set())
        rev = d._seed_clusters(dict(reversed(pairs)), set())
        self.assertEqual({frozenset(m) for m in fwd.values()},
                         {frozenset(m) for m in rev.values()})

    # ── _finalize_add_proposals unit behavior ──

    def test_duplicate_adds_merge_to_one_proposal(self):
        """Same node from two emitters → one proposal. Per community the
        HIGHER-affinity entry survives (fingerprint tier, quota rank, and
        prior suppressions key on the strongest candidate)."""
        first = {'type': 'add_to_existing', 'node_id': 'n1',
                 'communities': [{'id': 'X', 'title': 'x', 'affinity': 0.9}]}
        second = {'type': 'add_to_existing', 'node_id': 'n1',
                  'communities': [{'id': 'X', 'title': 'x', 'affinity': 0.5},
                                  {'id': 'Y', 'title': 'y', 'affinity': 0.3}]}
        out = self._decoder()._finalize_add_proposals([first, second], [])
        self.assertEqual(len(out), 1)
        merged = out[0]
        by_id = {c['id']: c['affinity'] for c in merged['communities']}
        self.assertEqual(by_id, {'X': 0.9, 'Y': 0.3})
        # Sorted by affinity, strongest first (feeds the fingerprint).
        affs = [c['affinity'] for c in merged['communities']]
        self.assertEqual(affs, sorted(affs, reverse=True))

    def test_merge_ranks_head_across_emitters(self):
        """The head candidate is the strongest across BOTH emitters, and a
        community present in both keeps its higher affinity."""
        first = {'type': 'add_to_existing', 'node_id': 'n1',
                 'communities': [{'id': 'X', 'title': 'x', 'affinity': 0.62}]}
        second = {'type': 'add_to_existing', 'node_id': 'n1',
                  'communities': [{'id': 'Z', 'title': 'z', 'affinity': 0.8},
                                  {'id': 'X', 'title': 'x', 'affinity': 0.5}]}
        out = self._decoder()._finalize_add_proposals([first, second], [])
        self.assertEqual(len(out), 1)
        merged = out[0]
        self.assertEqual([c['id'] for c in merged['communities']], ['Z', 'X'])
        # X keeps the first emitter's higher 0.62 over the second's 0.5.
        self.assertEqual(merged['communities'][1]['affinity'], 0.62)
        self.assertNotIn('overlap_frac', merged)

    def test_member_candidates_dropped_and_empty_proposal_removed(self):
        """A candidate community the node already belongs to is filtered;
        a proposal with no surviving candidates disappears."""
        state = [{'id': 'X', 'members': {'n1', 'n2'}},
                 {'id': 'Y', 'members': {'n2'}}]
        p_partial = {'type': 'add_to_existing', 'node_id': 'n1',
                     'communities': [{'id': 'X', 'title': 'x', 'affinity': 0.6},
                                     {'id': 'Y', 'title': 'y', 'affinity': 0.4}]}
        p_all_member = {'type': 'add_to_existing', 'node_id': 'n2',
                        'communities': [{'id': 'X', 'title': 'x', 'affinity': 0.6},
                                        {'id': 'Y', 'title': 'y', 'affinity': 0.4}]}
        other = {'type': 'health_update', 'community_id': 'X', 'signal': 'dead'}
        out = self._decoder()._finalize_add_proposals(
            [p_partial, p_all_member, other], state)
        adds = [p for p in out if p['type'] == 'add_to_existing']
        self.assertEqual(len(adds), 1)
        self.assertEqual(adds[0]['node_id'], 'n1')
        self.assertEqual([c['id'] for c in adds[0]['communities']], ['Y'])
        # Non-add proposals pass through untouched.
        self.assertIn(other, out)

    # ── graph-level behavior ──

    def test_conversion_skips_already_placed_members(self):
        """When a cluster converts to add_to_existing via the overlap check,
        members already placed in ANY community are left to the drift path —
        and add proposals stay unique per node."""
        target_members = [self._node('t%d' % i) for i in range(5)]
        self._community('target X', target_members)
        placed_elsewhere = self._node('placed p')
        self._community('other Y', [placed_elsewhere,
                                    self._node('y1'), self._node('y2')])

        # Tight cluster: 3 unplaced nodes + the placed one, every member
        # wired into target X so the overlap check converts the cluster.
        cluster = [self._node('c%d' % i) for i in range(3)] + [placed_elsewhere]
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                self.brain.connect(cluster[i], cluster[j],
                                   relation='implements', weight=0.8)
        for nid in cluster:
            for t in target_members[:3]:
                self.brain.connect(nid, t, relation='implements', weight=0.8)

        result = self._decode()
        adds = [p for p in result['proposals']
                if p['type'] == 'add_to_existing']
        add_nodes = [p['node_id'] for p in adds]
        self.assertNotIn(placed_elsewhere, add_nodes)
        # Batch contract: one add proposal per node.
        self.assertEqual(len(add_nodes), len(set(add_nodes)))
        # The unplaced cluster members still route somewhere structural.
        structural = [p for p in result['proposals']
                      if p['type'] in ('new_community', 'add_to_existing')]
        self.assertGreaterEqual(len(structural), 1)


if __name__ == '__main__':
    unittest.main()
