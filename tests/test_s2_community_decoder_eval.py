"""Decoder simulation applies overlapping decisions to current survivors."""
from tests.brain_test_base import BrainTestBase
from servers.scales.s2.community_decoder import CommunityDecoder
from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree
from eval.s2_community_decoder_eval import simulate_acceptance


class TestSimulatedCommunityMerges(BrainTestBase):
    needs_embedder = False

    def node(self, title, type='community'):
        return self.brain.remember(type=type, title=title, content=title,
                                   encoding_source='s2:community_detection')['id']

    def test_overlapping_decoder_pairs_converge_to_one_live_union(self):
        communities = [self.node(title) for title in ('A', 'B', 'C')]
        members = {self.node(title, 'finding') for title in ('One', 'Two', 'Three')}
        for cid in communities:
            for mid in members:
                self.brain._graph.add_relation(cid, mid, 'community_member')
        decoder = CommunityDecoder(self.brain)
        candidates = decoder._detect_merge_candidates(decoder._read_community_state())
        proposals = [{'type': 'merge_communities',
                      'larger_id': c['larger']['id'], 'smaller_id': c['smaller']['id']}
                     for c in candidates]
        self.assertEqual(len(proposals), 3)

        result = simulate_acceptance(self.brain, proposals, accept_rate=1)

        self.assertEqual(result['accepted_by_type']['merge_communities'], 3)
        state = decoder._read_community_state()
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0]['members'], members)
        self.assertEqual(decoder._detect_merge_candidates(state), [])
        self.assertEqual(self.brain._nodes.resolve_live(communities)['live'],
                         [state[0]['id']])

    def test_absorbed_survivor_target_resolves_before_next_merge(self):
        a, b, c = [self.node(title) for title in ('A', 'B', 'C')]
        proposals = [{'type': 'merge_communities', 'larger_id': b, 'smaller_id': a},
                     {'type': 'merge_communities', 'larger_id': a, 'smaller_id': c}]
        simulate_acceptance(self.brain, proposals, accept_rate=1)
        self.assertEqual(self.brain._nodes.resolve_live([a, b, c])['live'], [b])

    def test_unrelated_archive_refusal_still_fails_loudly(self):
        survivor, retired = self.node('Survivor'), self.node('Retired')
        self.brain.archive_node(retired, archived_by='test',
                                reason='Retired without a survivor')
        with self.assertRaisesRegex(RuntimeError, 'simulated merge failed'):
            simulate_acceptance(self.brain, [{
                'type': 'merge_communities', 'larger_id': survivor,
                'smaller_id': retired}], accept_rate=1)
