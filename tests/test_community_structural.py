"""Community structural-field derivation — servers/scales/s2/community_structural.

The stored structural fields (COMMUNITY_STRUCTURAL_KEYS) are arithmetic over
community_member edges, computed here instead of hand-written by the encoder
LLM. Three things must hold:
  1. the derivation writes exactly the contract's structural keys;
  2. correctness — derived values match the edges by hand-computation;
  3. PARITY — the decoder and the stamp read one adjacency builder, so the
     value the decode seam writes equals the one the stamp would.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.brain_test_base import BrainTestBase
from servers.scales.s2.community_structural import (
    build_member_adjacency, compute_community_structural, structural_fields,
    structural_metrics, write_structural_fields)
from servers.scales.s2.community_decoder import CommunityDecoder
from servers.scales.s2.community_encoder import CommunityEncoder
from servers.scales.s2.community_contract import COMMUNITY_STRUCTURAL_KEYS


class TestCommunityStructural(BrainTestBase):
    needs_embedder = False

    def _node(self, title, ntype='finding'):
        return self.brain.remember(type=ntype, title=title, content='c',
                                   encoding_source='test')['id']

    def _community(self, title='C'):
        return self.brain.remember(
            type='community', title=title, content='c',
            encoding_source='s2:community_detection')['id']

    def _connect(self, a, b, rel, w=0.5):
        self.brain.connect(a, b, relation=rel, weight=w)

    # ── correctness ──

    def test_derives_exactly_the_contract_structural_keys(self):
        cid = self._community()
        for m in (self._node('a'), self._node('b')):
            self._connect(cid, m, 'community_member', 0.3)

        res = compute_community_structural(self.brain, [cid])[cid]

        self.assertEqual(set(res), set(COMMUNITY_STRUCTURAL_KEYS))
        self.assertTrue(all(isinstance(v, str) for v in res.values()))

    def test_clique_is_fully_internal_not_corridor(self):
        cid = self._community()
        ms = [self._node('m%d' % i) for i in range(4)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        for i in range(4):
            for j in range(i + 1, 4):
                self._connect(ms[i], ms[j], 'extends')

        res = compute_community_structural(self.brain, [cid])[cid]

        self.assertEqual(res['community_internal_fraction'], '1.0')

    def test_low_internal_fraction(self):
        cid = self._community()
        ms = [self._node('c%d' % i) for i in range(5)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        self._connect(ms[0], ms[1], 'extends')          # 1 internal
        for i in range(5):                                # 5 external
            self._connect(ms[0], self._node('x%d' % i), 'extends')

        res = compute_community_structural(self.brain, [cid])[cid]

        # int_frac = 1 / (1 + 5) ≈ 0.167, stored at the two decimals shown
        self.assertEqual(res['community_internal_fraction'], '0.17')

    def test_non_cohesion_relations_excluded(self):
        cid = self._community()
        a, b, c = self._node('a'), self._node('b'), self._node('c')
        for m in (a, b, c):
            self._connect(cid, m, 'community_member', 0.3)
        # 'related' (pure-generic, non_cohesion_relations) must NOT count
        # as internal cohesion.
        self._connect(a, b, 'related')

        res = compute_community_structural(self.brain, [cid])[cid]

        self.assertEqual(res['community_internal_fraction'], '0.0')

    def test_empty_community_zero_fraction(self):
        cid = self._community()  # declared, no member edges
        res = compute_community_structural(self.brain, [cid])[cid]
        self.assertEqual(res['community_internal_fraction'], '0.0')

    def test_multiple_communities_one_call(self):
        c1, c2 = self._community('C1'), self._community('C2')
        a, b = self._node('a'), self._node('b')
        x, y, z = self._node('x'), self._node('y'), self._node('z')
        for m in (a, b):
            self._connect(c1, m, 'community_member', 0.3)
        for m in (x, y, z):
            self._connect(c2, m, 'community_member', 0.3)
        self._connect(x, y, 'extends')  # c2's only edge is internal

        res = compute_community_structural(self.brain, [c1, c2])

        self.assertEqual(res[c1]['community_internal_fraction'], '0.0')
        self.assertEqual(res[c2]['community_internal_fraction'], '1.0')

    # ── parity: the decoder and the stamp share one builder ──

    def test_decoder_and_stamp_derive_the_same_value(self):
        # The fixture carries a MULTI-HOMED verb on purpose: similar_to is
        # claimed by generic_relation (first, skipped) and settlement (last,
        # counted). A stamper that builds its relation→family map as a
        # last-claimant comprehension counts these edges while the decoder
        # skips them, and this test must fail on that regression. Guard the
        # premise so a taxonomy change can't silently defuse the test.
        registry = self.brain.aspects
        self.assertEqual(registry.primary_edge_map()['similar_to'],
                         'generic_relation')
        self.assertIn('similar_to', registry.settlement.edge_relations)

        cid = self._community()
        ms = [self._node('p%d' % i) for i in range(5)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        self._connect(ms[0], ms[1], 'extends')
        self._connect(ms[1], ms[2], 'extends')
        for i in range(3):
            self._connect(ms[0], self._node('q%d' % i), 'extends')
        # Internal similar_to edges: counted by a last-claimant stamper
        # (int_frac 2/5 → 5/8), skipped by the decoder.
        self._connect(ms[2], ms[3], 'similar_to')
        self._connect(ms[3], ms[4], 'similar_to')
        self._connect(ms[1], ms[4], 'similar_to')

        helper = compute_community_structural(self.brain, [cid])[cid]   # scoped

        # The whole-graph view the decoder runs on, and what it hands the seam.
        whole = structural_metrics(set(ms), build_member_adjacency(self.brain))
        dec = CommunityDecoder(self.brain)
        decoded = dec._decode(dec._read_community_state())['structural']

        self.assertEqual(whole['internal'], 2)   # similar_to skipped
        self.assertEqual(helper, structural_fields(whole))
        self.assertEqual(decoded[cid], helper)
        self.assertEqual(helper['community_internal_fraction'], '0.4')


class TestStructuralStamp(BrainTestBase):
    """The second algorithmic Δ — CommunityEncoder._stamp_structural_fields."""
    needs_embedder = False

    def _node(self, title, ntype='finding'):
        return self.brain.remember(type=ntype, title=title, content='c',
                                   encoding_source='test')['id']

    def _community(self, title='C'):
        return self.brain.remember(
            type='community', title=title, content='c',
            encoding_source='s2:community_detection')['id']

    def _connect(self, a, b, rel, w=0.5):
        self.brain.connect(a, b, relation=rel, weight=w)

    def _stamped(self, cid):
        return self.brain._meta_kv.get_fields_bulk(
            [cid], list(COMMUNITY_STRUCTURAL_KEYS)).get(cid, {})

    def _state(self):
        return CommunityDecoder(self.brain)._read_community_state()

    def _member_count(self, cid):
        return len(self.brain._graph.get_members_bulk([cid]).get(cid, []))

    def test_dispatch_connection_shapes_complete_encoder_bookkeeping(self):
        import json
        from unittest.mock import patch
        from servers.daemon_dispatch import COMMAND_TABLE

        members = [self._node('Member A'), self._node('Member B')]
        proposal = {'type': 'new_community', 'members': members,
                    'member_count': len(members), 'internal_fraction': 0.0}
        entries = [{'title': mid, 'relation': 'community_member'} for mid in members]
        for connect_to, expected_size in [(None, 0), (json.dumps(entries), 2),
                                          ('[' * 1100 + ']' * 1100, 0)]:
            with self.subTest(connect_to=connect_to):
                op = {'op': 'remember', 'type': 'community',
                      'title': 'New story %s' % type(connect_to).__name__, 'content': 'A story',
                      'connect_to': connect_to}
                created = []

                def encode(**kw):
                    response = COMMAND_TABLE['brain_batch'].handler(self.brain, {
                        'operations': [op], 'encoding_source': 's2:community_detection'}, [])
                    write = response['result']['results'][0]
                    # A shape that names no members is refused at the write.
                    self.assertEqual(write['ok'], bool(expected_size), response)
                    if write['ok']:
                        created.append(write['result']['id'])
                    return {'actions': 1, 'write_actions': 1, 'rounds': 1,
                            'interaction': self.brain.get_interaction_stamp(
                                's2_community_enrichment'),
                            'action_details': [{'tool': 'brain_batch',
                                                'input': {'operations': [op]}}]}

                encoder = CommunityEncoder(self.brain)
                state = CommunityDecoder(self.brain)._read_community_state()
                # The real batch loop runs — the stamp lives in it — with only
                # the model call replaced.
                with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                        patch('servers.scales.runner.make_client', return_value=object()), \
                        patch('servers.scales.runner.run_llm_loop', side_effect=encode):
                    result = encoder.run([proposal], state)
                self.assertFalse(result.get('error'), result)
                self.assertEqual(result['rejection_skipped_count'],
                                 0 if expected_size else 1)
                if expected_size:
                    self.assertEqual(self._member_count(created[0]), expected_size)
                    self.assertEqual(
                        self._stamped(created[0])['community_internal_fraction'], '0.0')
                else:
                    self.assertEqual(created, [])

    def test_stamps_newly_created_community(self):
        # Live now, absent from pre_community_ids → treated as created.
        cid = self._community()
        ms = [self._node('m%d' % i) for i in range(4)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        for i in range(4):
            for j in range(i + 1, 4):
                self._connect(ms[i], ms[j], 'extends')

        enc = CommunityEncoder(self.brain, None, {})
        n = enc._stamp_structural_fields([], set(), self._state())

        self.assertEqual(n, 1)
        self.assertEqual(self._stamped(cid), {'community_internal_fraction': '1.0'})

    def test_stamps_add_to_existing_target(self):
        cid = self._community()
        a, b = self._node('a'), self._node('b')
        for m in (a, b):
            self._connect(cid, m, 'community_member', 0.3)
        self._connect(a, b, 'extends')

        enc = CommunityEncoder(self.brain, None, {})
        proposals = [{'type': 'add_to_existing', 'communities': [{'id': cid}]}]
        # pre_community_ids includes cid → NOT new; it's touched via the proposal.
        n = enc._stamp_structural_fields(proposals, {cid}, self._state())

        self.assertEqual(n, 1)
        self.assertEqual(self._stamped(cid)['community_internal_fraction'], '1.0')

    def test_untouched_existing_community_not_stamped(self):
        cid = self._community()
        m = self._node('m')
        self._connect(cid, m, 'community_member', 0.3)

        enc = CommunityEncoder(self.brain, None, {})
        # Pre-existing, not in proposals, not new → skipped.
        n = enc._stamp_structural_fields([], {cid}, self._state())

        self.assertEqual(n, 0)
        self.assertEqual(self._stamped(cid), {})  # nothing written

    def test_writes_only_what_moved_in_bounded_chunks(self):
        from unittest.mock import patch
        calls = []

        def dispatch(cmd, args):
            calls.append(len(args['operations']))
            return {'ok': True, 'result': {'results': [
                {'ok': True} for _ in args['operations']]}}

        fields = {'c%d' % i: {'community_internal_fraction': '0.5'} for i in range(5)}
        stored = {'c0': {'community_internal_fraction': '0.5'},   # unchanged
                  'c1': {'community_internal_fraction': '0.4'}}
        with patch('servers.scales.s2.community_structural.STRUCTURAL_STAMP_CHUNK', 2):
            n = write_structural_fields(self.brain, dispatch, fields, stored,
                                        's2:community_detection')

        self.assertEqual(n, 4)            # c0 already current: nothing written
        self.assertEqual(calls, [2, 2])   # never more than a chunk per call

    def test_a_refused_chunk_is_logged_and_not_counted(self):
        from unittest.mock import patch

        def dispatch(cmd, args):
            return {'ok': False, 'error': 'batch refused'}

        fields = {'c1': {'community_internal_fraction': '0.5'}}
        with patch.object(self.brain, '_log_error') as logged:
            n = write_structural_fields(self.brain, dispatch, fields, {},
                                        's2:community_detection')
        self.assertEqual(n, 0)
        self.assertEqual(logged.call_args[0][0], 'community_structural_stamp')

    def test_corrects_a_drifted_stored_value(self):
        # The drift bug: a stale wrong value sitting in metadata. The stamp
        # must overwrite it with the value the edges give.
        cid = self._community()
        ms = [self._node('d%d' % i) for i in range(3)]
        for m in ms:
            self._connect(cid, m, 'community_member', 0.3)
        self.brain._meta_kv.set(cid, 'community_internal_fraction', '0.9')  # stale

        enc = CommunityEncoder(self.brain, None, {})
        enc._stamp_structural_fields(
            [{'type': 'health_update', 'community_id': cid}], {cid}, self._state())

        self.assertEqual(self._stamped(cid)['community_internal_fraction'], '0.0')


class TestDecodeSeamStamp(BrainTestBase):
    """The decode seam — CommunityDetection._stamp_structural writes every
    community's fields where they moved, before the encoder runs."""
    needs_embedder = False

    def test_rewrites_a_stale_value_once(self):
        from servers.scales.s2.community import CommunityDetection
        cid = self.brain.remember(type='community', title='C', content='c',
                                  encoding_source='s2:community_detection')['id']
        ms = [self.brain.remember(type='finding', title='m%d' % i, content='c',
                                  encoding_source='test')['id'] for i in range(3)]
        for m in ms:
            self.brain.connect(cid, m, relation='community_member', weight=0.3)
        self.brain.connect(ms[0], ms[1], relation='extends', weight=0.5)
        self.brain.connect(ms[1], ms[2], relation='extends', weight=0.5)
        self.brain._meta_kv.set(cid, 'community_internal_fraction', '0.9')

        unit = CommunityDetection(self.brain)
        state = unit._read_community_state()
        structural = unit._decode(state)['structural']

        self.assertEqual(unit._stamp_structural(structural, state), 1)
        self.assertEqual(self.brain._meta_kv.get_fields_bulk(
            [cid], ['community_internal_fraction'])[cid],
            {'community_internal_fraction': '1.0'})
        # Current now: the next cycle's identical values write nothing.
        self.assertEqual(unit._stamp_structural(
            structural, unit._read_community_state()), 0)

    def test_pipeline_stamps_even_when_no_proposal_survives(self):
        from unittest.mock import patch
        from servers.scales.s2.community import CommunityDetection
        cid = self.brain.remember(type='community', title='C', content='c',
                                  encoding_source='s2:community_detection')['id']
        self.brain._meta_kv.set(cid, 'community_internal_fraction', '0.9')
        unit = CommunityDetection(self.brain)
        decoded = {'proposals': [], 'pending_probes': [],
                   'community_state': unit._read_community_state(),
                   'structural': {cid: {'community_internal_fraction': '0.5'}}}
        with patch.object(CommunityDecoder, 'run', return_value=decoded):
            unit._run_pipeline()
        self.assertEqual(self.brain._meta_kv.get_fields_bulk(
            [cid], ['community_internal_fraction'])[cid],
            {'community_internal_fraction': '0.5'})


if __name__ == '__main__':
    unittest.main()
