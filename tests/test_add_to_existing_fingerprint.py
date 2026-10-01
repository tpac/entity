"""Fingerprint tests for add_to_existing proposals.

The decoder builds add_to_existing proposals with candidate communities
under a `communities: [{id, title, affinity}]` list (sorted by affinity
desc), NOT top-level community_id/affinity keys — see the two builders in
community_decoder.py (Step 9b incremental + Step 9 overlap-check convert).

compute_fingerprint used to read the absent top-level keys, collapsing
every add_to_existing fingerprint to md5('add:<node>::borderline'): one
rejection of (N → commA, weak) suppressed N into EVERY community at EVERY
affinity tier forever. These tests pin the discrimination back on.

Run: ./dev python3 -m pytest tests/test_add_to_existing_fingerprint.py -v
"""
import pytest

from servers.scales.s2.rejection_table import (
    compute_fingerprint,
    get_proposed_ids,
    match_proposals_to_actions,
    sort_proposals_by_priority,
)


def _proposal(node_id, communities):
    """communities: list of (id, affinity) in decoder-emitted order (desc)."""
    return {
        'type': 'add_to_existing',
        'node_id': node_id,
        'communities': [
            {'id': cid, 'title': cid, 'affinity': aff}
            for cid, aff in communities],
    }


class TestReportedBug:
    """The bug this file exists for: a weak rejection must not suppress a
    strong proposal to a different community."""

    def test_weak_rejection_does_not_suppress_strong_other_community(self):
        weak_to_a = _proposal('N', [('commA', 0.30)])     # borderline
        strong_to_b = _proposal('N', [('commB', 0.80)])   # strong, different comm
        assert compute_fingerprint(weak_to_a) != compute_fingerprint(strong_to_b)

    def test_different_community_same_tier_still_distinct(self):
        """Even at the same affinity tier, a different community is a
        different proposal — must not collapse."""
        to_a = _proposal('N', [('commA', 0.50)])  # moderate
        to_b = _proposal('N', [('commB', 0.50)])  # moderate, different comm
        assert compute_fingerprint(to_a) != compute_fingerprint(to_b)

    def test_same_community_different_tier_distinct(self):
        """Same community, affinity crossing a tier boundary → re-propose."""
        borderline = _proposal('N', [('commA', 0.30)])
        moderate = _proposal('N', [('commA', 0.50)])
        strong = _proposal('N', [('commA', 0.80)])
        fps = {compute_fingerprint(borderline),
               compute_fingerprint(moderate),
               compute_fingerprint(strong)}
        assert len(fps) == 3


class TestStableWithinTier:
    """Tier captures the regime where judgment changes — affinity jitter
    inside a tier must NOT invalidate a rejection (else suppression never
    holds)."""

    def test_same_tier_same_fingerprint(self):
        p1 = _proposal('N', [('commA', 0.66)])  # strong
        p2 = _proposal('N', [('commA', 0.95)])  # strong
        assert compute_fingerprint(p1) == compute_fingerprint(p2)

    def test_tier_boundaries(self):
        """0.40 and 0.65 are the moderate/strong cut points (>=)."""
        assert compute_fingerprint(_proposal('N', [('c', 0.39)])) \
            == compute_fingerprint(_proposal('N', [('c', 0.00)]))   # borderline
        assert compute_fingerprint(_proposal('N', [('c', 0.40)])) \
            == compute_fingerprint(_proposal('N', [('c', 0.64)]))   # moderate
        assert compute_fingerprint(_proposal('N', [('c', 0.65)])) \
            == compute_fingerprint(_proposal('N', [('c', 1.00)]))   # strong


class TestTopCandidateDrives:
    """Only communities[0] (the strongest candidate) is fingerprinted —
    it's the dominant input to the encoder's place/skip judgment."""

    def test_only_top_candidate_matters(self):
        """Same top candidate, different secondary candidates → same fp."""
        p1 = _proposal('N', [('commA', 0.70), ('commB', 0.30)])
        p2 = _proposal('N', [('commA', 0.70), ('commC', 0.25)])
        assert compute_fingerprint(p1) == compute_fingerprint(p2)

    def test_different_node_distinct(self):
        p_n = _proposal('N', [('commA', 0.50)])
        p_m = _proposal('M', [('commA', 0.50)])
        assert compute_fingerprint(p_n) != compute_fingerprint(p_m)


class TestDegenerateShapes:
    """Defensive: a missing/empty communities list must not raise."""

    def test_empty_communities_no_raise(self):
        fp = compute_fingerprint({'type': 'add_to_existing', 'node_id': 'N',
                                  'communities': []})
        assert isinstance(fp, str) and len(fp) == 16

    def test_absent_communities_no_raise(self):
        fp = compute_fingerprint({'type': 'add_to_existing', 'node_id': 'N'})
        assert isinstance(fp, str) and len(fp) == 16


def _connect_community_member(community_id, node_id):
    """A brain_batch action the encoder emits to place a node."""
    return {'tool': 'brain_batch', 'input': {'operations': [
        {'op': 'connect', 'relation': 'community_member',
         'source_id': community_id, 'target_id': node_id}]}}


class TestMatcherRecognizesPlacement:
    """match_proposals_to_actions read the phantom top-level community_id,
    so a placed add_to_existing node never matched → it was mis-stamped as
    rejected and its acted-on count was lost. Match on candidate community."""

    def test_placement_to_top_candidate_is_acted_on(self):
        prop = _proposal('N', [('commA', 0.70), ('commB', 0.30)])
        acted, skipped = match_proposals_to_actions(
            [prop], [_connect_community_member('commA', 'N')])
        assert acted == [prop]
        assert skipped == []

    def test_placement_to_non_top_candidate_is_acted_on(self):
        """Encoder may pick a weaker candidate — still acted on, not rejected."""
        prop = _proposal('N', [('commA', 0.70), ('commB', 0.30)])
        acted, skipped = match_proposals_to_actions(
            [prop], [_connect_community_member('commB', 'N')])
        assert acted == [prop]

    def test_no_placement_is_skipped(self):
        """Encoder reviewed and connected nothing → genuine skip."""
        prop = _proposal('N', [('commA', 0.70)])
        acted, skipped = match_proposals_to_actions([prop], [])
        assert acted == []
        assert skipped == [prop]

    def test_placement_of_other_node_does_not_match(self):
        """A community_member connect for a different node must not match."""
        prop = _proposal('N', [('commA', 0.70)])
        acted, skipped = match_proposals_to_actions(
            [prop], [_connect_community_member('commA', 'OTHER')])
        assert acted == []
        assert skipped == [prop]


class TestMatcherRecognizesCommunityWrites:
    @pytest.mark.parametrize('connect_to', [None, 42, {'title': 'a'}, 'not JSON',
        '[' * 1100 + ']' * 1100,
        [{'title': ['a'], 'relation': 'community_member'}]])
    def test_absent_or_malformed_connections_do_not_crash_outcomes(self, connect_to):
        prop = {'type': 'new_community', 'members': ['a', 'b']}
        actions = [{'tool': 'brain_batch', 'input': {'operations': [{
            'op': 'remember', 'type': 'community', 'connect_to': connect_to}]}}]
        assert match_proposals_to_actions([prop], actions) == ([], [prop])

    @pytest.mark.parametrize('key', ['target', 'title'])
    def test_creation_matches_connect_to_members_only(self, key):
        prop = {'type': 'new_community', 'members': ['a', 'b', 'c', 'd']}
        unrelated = {'type': 'new_community', 'members': ['x', 'y']}
        actions = [{'tool': 'brain_batch', 'input': {'operations': [{
            'op': 'remember', 'type': 'community', 'connect_to': [
                {key: 'a', 'relation': 'community_member'},
                {key: 'b', 'relation': 'community_member'},
                {key: 'x', 'relation': 'extends'},
                {key: 'y', 'relation': 'extends'},
            ]}]}}]
        acted, skipped = match_proposals_to_actions([prop, unrelated], actions)
        assert acted == [prop]
        assert skipped == [unrelated]

    def test_absorb_matches_only_the_merged_pair(self):
        prop = {'type': 'merge_communities', 'larger_id': 'a', 'smaller_id': 'b'}
        unrelated = {'type': 'merge_communities', 'larger_id': 'a', 'smaller_id': 'c'}
        actions = [{'tool': 'brain_batch', 'input': {'operations': [{
            'op': 'absorb', 'survivor_id': 'a', 'absorbed_id': 'b'}]}}]
        acted, skipped = match_proposals_to_actions([prop, unrelated], actions)
        assert acted == [prop]
        assert skipped == [unrelated]


class TestPrioritySortByAffinity:
    """sort_proposals_by_priority read phantom top-level affinity → every
    add_to_existing sorted as confidence 0, losing strong-first ordering."""

    def test_strong_affinity_sorts_before_weak(self):
        weak = _proposal('Nweak', [('commA', 0.30)])
        strong = _proposal('Nstrong', [('commB', 0.90)])
        ordered = sort_proposals_by_priority([weak, strong])
        assert ordered[0] is strong
        assert ordered[1] is weak


class TestProposedIdsCaptureCommunities:
    """get_proposed_ids must include the candidate community ids for S3."""

    def test_community_ids_present(self):
        prop = _proposal('N', [('commA', 0.70), ('commB', 0.30)])
        ids = get_proposed_ids(prop)
        assert 'N' in ids
        assert 'commA' in ids
        assert 'commB' in ids
