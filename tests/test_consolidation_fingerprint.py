"""Fingerprint tests for consolidation_cluster proposals.

Verifies the content-signature behavior: a rejected cluster stays
rejected only while its member content is unchanged. If any member's
updated_at changes, the fingerprint changes and the encoder re-evaluates
on the next run.

Run: ./dev python3 -m pytest tests/test_consolidation_fingerprint.py -v
"""
import pytest

from servers.scales.s2.rejection_table import compute_fingerprint


def _proposal(members, updated_at=None):
    p = {'type': 'consolidation_cluster', 'members': list(members)}
    if updated_at is not None:
        p['member_updated_at'] = updated_at
    return p


class TestSameInputsSameFingerprint:
    def test_same_members_same_ts_stable(self):
        """Same members + same timestamps = same fingerprint (deterministic)."""
        p1 = _proposal(['n1', 'n2', 'n3'],
                       {'n1': '2026-04-19T12:00', 'n2': '2026-04-19T12:05',
                        'n3': '2026-04-19T12:10'})
        p2 = _proposal(['n1', 'n2', 'n3'],
                       {'n1': '2026-04-19T12:00', 'n2': '2026-04-19T12:05',
                        'n3': '2026-04-19T12:10'})
        assert compute_fingerprint(p1) == compute_fingerprint(p2)

    def test_member_order_doesnt_matter(self):
        """Input order varies, fingerprint stable (sorted internally)."""
        ts = {'n1': 'a', 'n2': 'b', 'n3': 'c'}
        p1 = _proposal(['n1', 'n2', 'n3'], ts)
        p2 = _proposal(['n3', 'n1', 'n2'], ts)
        assert compute_fingerprint(p1) == compute_fingerprint(p2)


class TestContentChangeInvalidates:
    def test_updated_at_change_on_one_member_invalidates(self):
        """Revise one member → fingerprint changes → re-proposal passes."""
        base = _proposal(['n1', 'n2'],
                         {'n1': '2026-04-19T12:00', 'n2': '2026-04-19T12:00'})
        revised = _proposal(['n1', 'n2'],
                            {'n1': '2026-04-19T12:00',
                             'n2': '2026-04-19T13:00'})  # n2 edited
        assert compute_fingerprint(base) != compute_fingerprint(revised)

    def test_all_members_revised_invalidates(self):
        base = _proposal(['n1', 'n2'], {'n1': 'old', 'n2': 'old'})
        revised = _proposal(['n1', 'n2'], {'n1': 'new', 'n2': 'new'})
        assert compute_fingerprint(base) != compute_fingerprint(revised)


class TestMembershipChangeInvalidates:
    def test_adding_a_member_invalidates(self):
        p1 = _proposal(['n1', 'n2'],
                       {'n1': 'a', 'n2': 'b'})
        p2 = _proposal(['n1', 'n2', 'n3'],
                       {'n1': 'a', 'n2': 'b', 'n3': 'c'})
        assert compute_fingerprint(p1) != compute_fingerprint(p2)

    def test_removing_a_member_invalidates(self):
        p1 = _proposal(['n1', 'n2', 'n3'],
                       {'n1': 'a', 'n2': 'b', 'n3': 'c'})
        p2 = _proposal(['n1', 'n2'], {'n1': 'a', 'n2': 'b'})
        assert compute_fingerprint(p1) != compute_fingerprint(p2)


class TestTimestampsRequired:
    def test_missing_updated_at_raises(self):
        """A proposal without member timestamps would fingerprint blind to
        edits; the producer's omission fails loudly instead."""
        with pytest.raises(KeyError):
            compute_fingerprint(_proposal(['n1', 'n2']))
