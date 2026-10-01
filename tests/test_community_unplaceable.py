"""Phase 2 — community unplaceable-marking.

The community decoder's only rest condition used to be `unplaced_count == 0`,
unreachable because ~28% of nodes never cluster. Phase 2 marks an examined-but-
unplaceable node with a fingerprint of its 1-hop neighborhood (each neighbor +
that neighbor's community); filter_rejected suppresses it until that fingerprint
moves — a node gains/loses an edge, or a neighbor joins a community. Then the
backlog drains and the decoder rests. These tests stay LLM- and embedder-free:
they exercise the fingerprint, the suppress/re-eligible cycle, the one-row-per-
node marking, and the decoder's rest path (which returns before _decode).
"""

import json
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.brain_test_base import BrainTestBase
from servers.scales.s2.community import CommunityDetection
from servers.scales.s2.community_decoder import CommunityDecoder
from servers.scales.s2.rejection_table import compute_fingerprint, filter_rejected


class TestCommunityUnplaceable(BrainTestBase):
    needs_embedder = False

    def _node(self, title, ntype='finding'):
        return self.brain.remember(type=ntype, title=title, content='c',
                                   encoding_source='test')['id']

    # ── fingerprint basis ────────────────────────────────────────────
    def test_neighborhood_str_order_independent_and_change_sensitive(self):
        d = CommunityDecoder(self.brain)
        base = d._neighborhood_str('X', {'X': {'a', 'b'}}, {'a': '', 'b': ''})
        # order-independent
        self.assertEqual(base, d._neighborhood_str('X', {'X': {'b', 'a'}},
                                                    {'a': '', 'b': ''}))
        # a neighbor joining a community changes it
        self.assertNotEqual(base, d._neighborhood_str('X', {'X': {'a', 'b'}},
                                                       {'a': 'commA', 'b': ''}))
        # a new edge changes it
        self.assertNotEqual(base, d._neighborhood_str('X', {'X': {'a', 'b', 'c'}},
                                                      {'a': '', 'b': '', 'c': ''}))

    def test_unplaceable_fingerprint_tracks_neighborhood(self):
        same_a = compute_fingerprint({'type': 'unplaceable', 'node_id': 'X',
                                      'neighborhood': 'a>;b>'})
        same_b = compute_fingerprint({'type': 'unplaceable', 'node_id': 'X',
                                      'neighborhood': 'a>;b>'})
        moved = compute_fingerprint({'type': 'unplaceable', 'node_id': 'X',
                                     'neighborhood': 'a>commA;b>'})
        self.assertEqual(same_a, same_b)        # stable
        self.assertNotEqual(same_a, moved)      # neighborhood change → new fp

    # ── suppress / re-eligible cycle ─────────────────────────────────
    def test_marked_node_suppressed_until_neighborhood_moves(self):
        unit = CommunityDetection(self.brain)
        x = self._node('X')
        neighbors = {x: {'neighborA'}}
        probe = {'type': 'unplaceable', 'node_id': x,
                 'neighborhood': unit._neighborhood_str(x, neighbors, {})}

        unit._mark_unplaceable([probe])
        pending, suppressed = filter_rejected(self.brain, [probe])
        self.assertEqual(pending, [])           # asleep
        self.assertEqual(suppressed, 1)

        # neighborA joins a community → fingerprint moves → re-eligible
        probe2 = {'type': 'unplaceable', 'node_id': x,
                  'neighborhood': unit._neighborhood_str(x, neighbors,
                                                         {'neighborA': 'commZ'})}
        pending2, suppressed2 = filter_rejected(self.brain, [probe2])
        self.assertEqual(len(pending2), 1)      # awake again
        self.assertEqual(suppressed2, 0)

    def test_mark_keeps_one_row_per_node(self):
        unit = CommunityDetection(self.brain)
        x = self._node('X')
        unit._mark_unplaceable([{'type': 'unplaceable', 'node_id': x, 'neighborhood': 'v1'}])
        unit._mark_unplaceable([{'type': 'unplaceable', 'node_id': x, 'neighborhood': 'v2'}])
        rows = self.brain.conn.execute(
            "SELECT COUNT(*) FROM s2_rejections "
            "WHERE proposal_type='unplaceable' AND proposed_ids=?",
            (json.dumps([x]),)).fetchone()[0]
        self.assertEqual(rows, 1)               # replaced, not accumulated

    # ── _load_neighbors ──────────────────────────────────────────────
    def test_load_neighbors_both_directions_excludes_noise(self):
        unit = CommunityDetection(self.brain)
        x, a, b = self._node('X'), self._node('A'), self._node('B')
        self.brain.connect(x, a, relation='depends_on', weight=0.6)   # x -> a
        self.brain.connect(b, x, relation='depends_on', weight=0.6)   # b -> x
        nbrs = unit._load_neighbors({x})
        self.assertEqual(nbrs[x], {a, b})       # both directions

    # ── marking is by ACTUAL placement, not predicted ────────────────
    def test_mark_unplaced_pending_skips_actually_placed(self):
        # A pending node the encoder actually placed must NOT be marked; one
        # left unplaced (corridor-dropped / quota-deferred / skipped) must be.
        unit = CommunityDetection(self.brain)
        placed, orphan = self._node('placed'), self._node('orphan')
        comm = self.brain.remember(type='community', title='C', content='c',
                                   encoding_source='s2:community_detection')['id']
        self.brain.connect(comm, placed, relation='community_member', weight=0.6)

        probes = [{'type': 'unplaceable', 'node_id': placed, 'neighborhood': ''},
                  {'type': 'unplaceable', 'node_id': orphan, 'neighborhood': ''}]
        unit._mark_unplaced_pending(probes)

        def marked(nid):
            return self.brain.conn.execute(
                "SELECT COUNT(*) FROM s2_rejections "
                "WHERE proposal_type='unplaceable' AND proposed_ids=?",
                (json.dumps([nid]),)).fetchone()[0]
        self.assertEqual(marked(placed), 0)    # in a community → not marked
        self.assertEqual(marked(orphan), 1)    # not placed → marked

    # ── node_to_comm is order-independent (multi-membership) ──────────
    def test_node_to_comm_deterministic_multi_membership(self):
        d = CommunityDecoder(self.brain)
        cs1 = [{'id': 'commA', 'members': ['x', 'y']},
               {'id': 'commB', 'members': ['x']}]
        cs2 = list(reversed(cs1))
        m1, m2 = d._node_to_comm(cs1), d._node_to_comm(cs2)
        self.assertEqual(m1, m2)                       # community_state order irrelevant
        self.assertEqual(m1['x'], 'commA,commB')       # all communities, sorted
        self.assertEqual(m1['y'], 'commA')

    # ── decoder rests when the backlog is fully marked ───────────────
    def test_decoder_rests_when_all_unplaced_marked(self):
        unit = CommunityDetection(self.brain)
        d = CommunityDecoder(self.brain)
        x, y = self._node('X'), self._node('Y')

        # mark both at their current (empty) neighborhood — the prior-cycle state
        probes = [{'type': 'unplaceable', 'node_id': nid,
                   'neighborhood': unit._neighborhood_str(nid, {}, {})}
                  for nid in (x, y)]
        unit._mark_unplaceable(probes)

        # decoder.run() now finds pending empty → rests BEFORE _decode
        result = d.run()
        self.assertIn('skipped', result)
        self.assertIn('unplaceable', result['skipped'])
        self.assertEqual(result['proposals'], [])


if __name__ == '__main__':
    unittest.main()
