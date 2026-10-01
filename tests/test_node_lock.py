"""set_node_lock — the one door for lock flips on existing nodes.

Contract under test:
  - Two-phase confirm: phase 1 executes nothing, returns a one-shot token +
    summary; phase 2 with the token flips the flag.
  - Guards: unknown node, locking an archived node, no-op (already in state),
    invalid/expired/mismatched token, token is one-shot.
  - revise() immutability is UNCHANGED — locked stays skipped there.
  - Dispatch roundtrip emits a node_revised trace with the locked delta.
  - Guardrail: the shared S1/S2 encoder dispatch closure refuses the command.
"""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.brain_test_base import BrainTestBase


def _locked_in_db(brain, node_id):
    row = brain.conn.execute(
        'SELECT locked FROM nodes WHERE id = ?', (node_id,)).fetchone()
    return bool(row[0])


def _make_node(brain, **kwargs):
    defaults = {'type': 'concept',
                'title': 'Lock test node %d' % int(time.time() * 1000),
                'content': 'content'}
    defaults.update(kwargs)
    return brain.remember(**defaults)['id']


class TestSetNodeLock(BrainTestBase):
    needs_embedder = False

    def _confirm_flip(self, node_id, locked, reason='test'):
        """Run both phases; returns the phase-2 result."""
        p1 = self.brain.set_node_lock(node_id, locked, reason=reason)
        self.assertTrue(p1.get('confirmation_required'))
        return self.brain.set_node_lock(node_id, locked, reason=reason,
                                        confirm_token=p1['confirm_token'])

    def test_phase1_returns_token_and_flips_nothing(self):
        nid = _make_node(self.brain)
        res = self.brain.set_node_lock(nid, True, reason='canonical principle')
        self.assertTrue(res['ok'])
        self.assertTrue(res['confirmation_required'])
        self.assertTrue(res['confirm_token'].startswith('lock-'))
        self.assertIn('needs an explicit yes', res['summary'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_phase2_flips_and_reports_delta(self):
        nid = _make_node(self.brain)
        res = self._confirm_flip(nid, True)
        self.assertTrue(res['ok'])
        self.assertTrue(res['changed'])
        self.assertTrue(_locked_in_db(self.brain, nid))
        self.assertEqual(res['deltas'],
                         [{'field': 'locked', 'old': False, 'new': True}])
        self.assertIn('confirm_latency_s', res)

    def test_unlock_roundtrip(self):
        nid = _make_node(self.brain, locked=True, encoding_source='anchor')
        self.assertTrue(_locked_in_db(self.brain, nid))
        res = self._confirm_flip(nid, False)
        self.assertTrue(res['changed'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_noop_needs_no_confirmation(self):
        nid = _make_node(self.brain)
        res = self.brain.set_node_lock(nid, False, reason='already unlocked')
        self.assertTrue(res['ok'])
        self.assertFalse(res.get('confirmation_required', False))
        self.assertFalse(res['changed'])
        self.assertIn('already unlocked', res['note'])

    def test_unknown_node(self):
        res = self.brain.set_node_lock('ffffffff', True, reason='x')
        self.assertFalse(res['ok'])
        self.assertIn('not found', res['error'].lower())

    def test_cannot_lock_archived_node(self):
        nid = _make_node(self.brain)
        self.brain.archive_node(nid, archived_by='anchor', reason='test')
        res = self.brain.set_node_lock(nid, True, reason='x')
        self.assertFalse(res['ok'])
        self.assertIn('archived', res['error'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_token_is_one_shot(self):
        nid = _make_node(self.brain)
        p1 = self.brain.set_node_lock(nid, True, reason='x')
        tok = p1['confirm_token']
        first = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertTrue(first['changed'])
        # unlock so the state allows a second attempt, then replay the token
        self._confirm_flip(nid, False)
        replay = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertFalse(replay['ok'])
        self.assertIn('invalid or expired', replay['error'])

    def test_token_bound_to_node_and_direction(self):
        nid_a = _make_node(self.brain)
        nid_b = _make_node(self.brain)
        tok = self.brain.set_node_lock(nid_a, True, reason='x')['confirm_token']
        res = self.brain.set_node_lock(nid_b, True, reason='x', confirm_token=tok)
        self.assertFalse(res['ok'])
        self.assertFalse(_locked_in_db(self.brain, nid_b))
        # A mismatched confirm must NOT burn the pending token — the operator's
        # yes for node A still works after the wrong-node slip.
        good = self.brain.set_node_lock(nid_a, True, reason='x', confirm_token=tok)
        self.assertTrue(good['changed'])
        self.assertTrue(_locked_in_db(self.brain, nid_a))

    def test_token_bound_to_session(self):
        nid = _make_node(self.brain)
        tok = self.brain.set_node_lock(nid, True, reason='x',
                                       session_id='stream-a')['confirm_token']
        other = self.brain.set_node_lock(nid, True, reason='x',
                                         confirm_token=tok, session_id='stream-b')
        self.assertFalse(other['ok'])
        self.assertFalse(_locked_in_db(self.brain, nid))
        same = self.brain.set_node_lock(nid, True, reason='x',
                                        confirm_token=tok, session_id='stream-a')
        self.assertTrue(same['changed'])

    def test_noop_phase2_consumes_token(self):
        """A phase-2 match that lands as a no-op must still burn the token —
        otherwise it lingers and can replay after a later state change."""
        nid = _make_node(self.brain)
        tok = self.brain.set_node_lock(nid, True, reason='x')['confirm_token']
        self.brain._nodes.set_locked(nid, True)  # raced by another door
        noop = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertTrue(noop['ok'])
        self.assertFalse(noop['changed'])
        self.brain._nodes.set_locked(nid, False)
        replay = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertFalse(replay['ok'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_locked_must_be_boolean(self):
        """bool('false') is True — a string must be rejected, not coerced."""
        nid = _make_node(self.brain)
        res = self.brain.set_node_lock(nid, 'false', reason='unlock please')
        self.assertFalse(res['ok'])
        self.assertIn('boolean', res['error'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_non_anchor_encoding_source_refused(self):
        """Write-boundary mirror of remember()'s anchor-only lock rule."""
        nid = _make_node(self.brain)
        res = self.brain.set_node_lock(nid, True, reason='x',
                                       encoding_source='s2:consolidation')
        self.assertFalse(res['ok'])
        self.assertIn('anchor-only', res['error'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_expired_token_rejected(self):
        nid = _make_node(self.brain)
        tok = self.brain.set_node_lock(nid, True, reason='x')['confirm_token']
        self.brain._pending_lock_confirms[tok]['requested_at'] -= (
            self.brain.LOCK_CONFIRM_TTL_S + 1)
        res = self.brain.set_node_lock(nid, True, reason='x', confirm_token=tok)
        self.assertFalse(res['ok'])
        self.assertFalse(_locked_in_db(self.brain, nid))

    def test_revise_still_refuses_locked_field(self):
        """The new door must not relax revise() immutability."""
        nid = _make_node(self.brain)
        res = self.brain.revise(node_id=nid, reason='attempt lock via revise',
                                updates={'locked': True})
        self.assertFalse(_locked_in_db(self.brain, nid))
        self.assertTrue(any('immutable' in w for w in res.get('warnings', [])))


class TestSetNodeLockDispatch(BrainTestBase):
    """The dispatch door: arg contract + the node_revised trace on a flip."""
    needs_embedder = False

    def _dispatch(self, args):
        from servers.daemon_dispatch import dispatch_command
        return dispatch_command(self.brain, 'set_node_lock', args, [])

    def test_requires_node_id_locked_and_reason(self):
        nid = _make_node(self.brain)
        self.assertFalse(self._dispatch({'locked': True, 'reason': 'x'})['ok'])
        self.assertFalse(self._dispatch({'node_id': nid, 'reason': 'x'})['ok'])
        self.assertFalse(self._dispatch({'node_id': nid, 'locked': True})['ok'])
        # Type guard: a string 'false' must be rejected at the handler.
        bad = self._dispatch({'node_id': nid, 'locked': 'false', 'reason': 'x'})
        self.assertFalse(bad['ok'])
        self.assertIn('boolean', bad['error'])

    def test_flip_emits_node_lock_changed_trace(self):
        nid = _make_node(self.brain)
        p1 = self._dispatch({'node_id': nid, 'locked': True, 'reason': 'canon'})
        self.assertTrue(p1['ok'])
        tok = p1['result']['confirm_token']
        p2 = self._dispatch({'node_id': nid, 'locked': True, 'reason': 'canon',
                             'confirm_token': tok})
        self.assertTrue(p2['ok'])
        self.assertTrue(_locked_in_db(self.brain, nid))
        rows = self.brain._trace_dal.conn.execute(
            "SELECT metadata FROM trace_events "
            "WHERE ref_type = 'node_lock_changed' AND ref_id = ?", (nid,)).fetchall()
        self.assertEqual(len(rows), 1)
        meta = json.loads(rows[0][0])
        self.assertTrue(meta['locked'])
        self.assertEqual(meta['reason'], 'canon')
        self.assertIn('confirm_latency_s', meta)
        # And it must NOT masquerade as a content revision.
        revised = self.brain._trace_dal.conn.execute(
            "SELECT 1 FROM trace_events "
            "WHERE ref_type = 'node_revised' AND ref_id = ?", (nid,)).fetchall()
        self.assertEqual(revised, [])

    def test_phase1_emits_no_trace(self):
        nid = _make_node(self.brain)
        self._dispatch({'node_id': nid, 'locked': True, 'reason': 'canon'})
        rows = self.brain._trace_dal.conn.execute(
            "SELECT 1 FROM trace_events "
            "WHERE ref_type IN ('node_lock_changed', 'node_revised') "
            "AND ref_id = ?", (nid,)).fetchall()
        self.assertEqual(rows, [])


class TestEncoderClosureRefusesLock(unittest.TestCase):
    """Guardrail: the shared S1/S2 encoder dispatch refuses set_node_lock."""

    def test_encoder_dispatch_refuses(self):
        import threading
        from servers.scales.s2.base import IntegrationUnit

        errors = []

        class FakeBrain:
            write_lock = threading.RLock()

            def _log_error(self, kind, exc, ctx=''):
                errors.append(kind)

        class TestUnit(IntegrationUnit):
            NAME = 'test_lock_refusal'
            SCALE = 's2'
            ENCODING_SOURCE = 's2:test_lock_refusal'

        dispatch = TestUnit(brain=FakeBrain())._make_encoder_dispatch()
        res = dispatch('set_node_lock',
                       {'node_id': 'deadbeef', 'locked': True, 'reason': 'x'})
        self.assertFalse(res['ok'])
        self.assertIn('operator-channel only', res['error'])
        self.assertEqual(errors, ['s2_test_lock_refusal_lock_refused'])


if __name__ == '__main__':
    unittest.main()
