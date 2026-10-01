"""S2 community health — two-threshold seam (2026-06-23).

Pins the redesign of Step 5d health signals:
  - `degrading` is REMOVED entirely (it fired every cycle on diffused-but-alive
    communities and the encoder's rote maturity='forming' never cleared the
    creation-frozen baseline → perpetual zero-value churn).
  - within typed int_frac < low_cohesion (0.10): DISCONNECTED (no internal
    edge of any real-cohesion relation) → DETERMINISTIC auto-archive, never an
    encoder proposal (can't get stuck suppressed-but-undead); else → 'dead'
    encoder proposal (judge). Counting ALL relations means a similar_to-
    cohesive community is NOT auto-archived — closes the typed-int_frac blind
    spot.
  - corridor_maturing unchanged.

Tests call _decode() directly with a constructed community_state, same as
TestCommunityDecoder. needs_embedder=False (no auto-connect; edges are exactly
what we add).

Run: ./dev python3 -m pytest tests/test_community_health_seam.py -v
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.scales.s2.community_decoder import CommunityDecoder


class TestCommunityHealthSeam(BrainTestBase):
    needs_embedder = False


    def _members(self, prefix, n):
        ids = []
        for i in range(n):
            r = self.brain.remember(
                type='decision', title='%s node %d' % (prefix, i),
                content='content %s %d' % (prefix, i))
            ids.append(r['id'])
        return ids

    def _make_community(self, member_ids):
        comm = self.brain.remember(
            type='community', title='Test Community', content='t',
            encoding_source='s2:community_detection')
        for m in member_ids:
            self.brain.connect(comm['id'], m, relation='community_member', weight=0.3)
        state = [{
            'id': comm['id'], 'title': 'Test Community', 'content': 't',
            'keywords': '', 'confidence': 0.8, 'members': set(member_ids),
            'meta': {},
        }]
        return comm['id'], state

    def _decode(self, community_state):
        return CommunityDecoder(self.brain)._decode(community_state)

    # ── degrading is gone ───────────────────────────────────────────

    def test_no_degrading_signal_for_diffused_but_alive(self):
        """A community whose int_frac (~0.5) dropped below 70% of its stored
        creation fraction (0.9) is exactly what OLD code flagged 'degrading'.
        New code emits NOTHING for it — not degrading, not dead, not archive."""
        members = self._members('half', 4)
        # 2 internal edges, 2 external → int_frac = 2/(2+2) = 0.5
        self.brain.connect(members[0], members[1], relation='implements', weight=0.8)
        self.brain.connect(members[2], members[3], relation='implements', weight=0.8)
        ext = self._members('half_ext', 2)
        self.brain.connect(members[0], ext[0], relation='depends_on', weight=0.5)
        self.brain.connect(members[2], ext[1], relation='depends_on', weight=0.5)
        comm_id, state = self._make_community(members)
        # Stored creation fraction high → OLD 'degrading' trigger (0.5 < 0.9*0.7).
        self.brain._meta_kv.set(comm_id, 'community_internal_fraction', '0.9')

        result = self._decode(state)
        signals = [p.get('signal') for p in result['proposals']
                   if p['type'] == 'health_update' and p.get('community_id') == comm_id]
        self.assertNotIn('degrading', signals)   # removed entirely
        self.assertEqual(signals, [])            # healthy-but-loose → no signal
        self.assertNotIn(comm_id, {d['id'] for d in result['auto_archive_dead']})

    # ── deterministic disconnected sweep ────────────────────────────

    def test_disconnected_auto_archives_not_proposed(self):
        """A community with NO internal edge of any kind (members only point
        outward) is structurally disconnected → deterministic auto-archive,
        NOT sent to the encoder."""
        members = self._members('dead', 3)            # no internal edges
        ext = self._members('dead_ext', 4)
        for m in members:
            for e in ext:
                self.brain.connect(m, e, relation='depends_on', weight=0.5)
        comm_id, state = self._make_community(members)

        result = self._decode(state)
        self.assertIn(comm_id, {d['id'] for d in result['auto_archive_dead']})
        proposed = [p for p in result['proposals']
                    if p['type'] == 'health_update' and p.get('community_id') == comm_id]
        self.assertEqual(proposed, [])

    def test_similar_to_cohesive_routes_to_encoder_not_auto_archived(self):
        """Blind-spot closure: a community whose ONLY internal links are
        similar_to (excluded from typed int_frac, so typed int_frac=0) is NOT
        disconnected → routed to the encoder for judgment, never auto-archived.
        Consolidation writes similar_to between kin, so this is the realistic
        path to typed int_frac=0 with real cohesion."""
        members = self._members('kin', 3)
        # internal cohesion ONLY via similar_to (generic_relation, dropped from
        # typed adjacency → typed internal = 0)
        self.brain.connect(members[0], members[1], relation='similar_to', weight=0.5)
        self.brain.connect(members[1], members[2], relation='similar_to', weight=0.5)
        ext = self._members('kin_ext', 2)
        for m in members:                              # typed external edges
            self.brain.connect(m, ext[0], relation='depends_on', weight=0.5)
        comm_id, state = self._make_community(members)

        result = self._decode(state)
        self.assertNotIn(comm_id, {d['id'] for d in result['auto_archive_dead']})
        proposed = [p for p in result['proposals']
                    if p['type'] == 'health_update' and p.get('community_id') == comm_id]
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0]['signal'], 'dead')

    # ── encoder judgment band ───────────────────────────────────────

    def test_low_cohesion_band_goes_to_encoder_as_dead(self):
        """int_frac in [floor, low_cohesion) → 'dead' encoder proposal (judge),
        NOT a deterministic archive."""
        members = self._members('low', 3)
        self.brain.connect(members[0], members[1], relation='implements', weight=0.8)  # 1 internal
        ext = self._members('low_ext', 4)
        for m in members:                              # 12 external → 1/(1+12)=0.077
            for e in ext:
                self.brain.connect(m, e, relation='depends_on', weight=0.5)
        comm_id, state = self._make_community(members)

        result = self._decode(state)
        self.assertNotIn(comm_id, {d['id'] for d in result['auto_archive_dead']})
        proposed = [p for p in result['proposals']
                    if p['type'] == 'health_update' and p.get('community_id') == comm_id]
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0]['signal'], 'dead')


if __name__ == '__main__':
    unittest.main()
