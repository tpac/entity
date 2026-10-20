"""Self-message delivery — the Stop hook is the SOLE path (Stop-only, 2026-06-04).

The self-channel envelope bug (c4f6386) escaped because inner functions were
unit-tested but the dispatch/handler wiring was not. This layer closes that for
DELIVERY: it drives the real daemon handlers — hook_pre_edit (PreToolUse) and
hook_post_response_track (Stop) — and asserts that self-messages land on the
prominent Stop block and that PreToolUse stays silent. PreToolUse delivery into
additionalContext was removed (the model missed it, and consuming there starved
the reliable Stop block); UserPromptSubmit was never a delivery point. So
PreToolUse must NOT surface or consume the tap — it leaves it for Stop.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.daemon_hooks import hook_pre_edit, hook_post_response_track
from servers.channels.self_channel import signal, self_contract


def _seed(brain, to_session, body):
    """Queue a directed self-message from a different session (so drain_inbox's
    'not your own' filter delivers it)."""
    signal.send(brain, from_session='other-session',
                address=self_contract.address_for_session(to_session), body=body)


def _edit(brain, session_id, path='auth.py'):
    return hook_pre_edit(brain, {'path': path, 'session_id': session_id}, [])


def _stop(brain, session_id):
    return hook_post_response_track(brain, {
        'hook_event_name': 'Stop', 'session_id': session_id,
        'last_assistant_message': 'done'}, [])


def _pretool_ac(out):
    """The additionalContext the daemon hands the PreToolUse hook — the text
    that reaches the model (the hook wraps it in hookSpecificOutput)."""
    return out.get('json', {}).get('additionalContext', '')


class TestPreToolDoesNotDeliver(BrainTestBase):
    """Stop-only (2026-06-04): PreToolUse must NOT surface a self-message and
    must NOT consume it — the tap survives untouched so the Stop block delivers."""
    needs_embedder = False

    def test_pretool_does_not_surface_self_message(self):
        _seed(self.brain, 'S', 'heads up: I am in auth.py')
        out = _edit(self.brain, 'S')
        self.assertNotIn('heads up: I am in auth.py', _pretool_ac(out))
        self.assertNotIn('decision', out.get('json', {}))

    def test_pretool_does_not_consume_tap(self):
        # The tap must survive PreToolUse so Stop can still deliver it.
        _seed(self.brain, 'S', 'one tap')
        _edit(self.brain, 'S', 'x.py')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')

    def test_pretool_no_message_is_clean(self):
        """Regression: nothing pending → no decision, no spurious self-block."""
        out = _edit(self.brain, 'S', 'y.py')
        self.assertNotIn('decision', out.get('json', {}))
        self.assertNotIn('🧵', _pretool_ac(out))


class TestStopDelivery(BrainTestBase):
    needs_embedder = False

    def test_stop_blocks_to_deliver(self):
        _seed(self.brain, 'S', 'before you finish: note X')
        out = _stop(self.brain, 'S')
        self.assertEqual(out.get('decision'), 'block')
        self.assertIn('before you finish: note X', out.get('reason', ''))

    def test_stop_consume_once(self):
        _seed(self.brain, 'S', 'one tap')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')
        # already consumed → next stop is allowed (no block, no loop)
        self.assertNotEqual(_stop(self.brain, 'S').get('decision'), 'block')

    def test_stop_no_message_allows_stop(self):
        """Regression: nothing pending → no block, normal output."""
        self.assertNotEqual(_stop(self.brain, 'FRESH').get('decision'), 'block')

    def test_stop_trace_attributes_to_recipient_session(self):
        """Regression (2026-06-05): the self_message delivery trace must carry
        the recipient session_id. It was written empty while its three sibling
        S0 appends passed it, so a session-scoped (dashboard) query couldn't see
        the delivery. All four S0 turn-traces now bind session_id via _s0_trace —
        a session-scoped query must return the delivery."""
        _seed(self.brain, 'S', 'attributed tap')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')
        events = self.brain.query_traces(session_id='S').get('events', [])
        self_msgs = [e for e in events if e.get('ref_type') == 'self_message']
        self.assertTrue(
            self_msgs,
            "self_message delivery trace not attributed to recipient session 'S'")


class TestStopDeliveryChain(BrainTestBase):
    needs_embedder = False

    def test_delivery_opens_the_next_turns_chain(self):
        """A delivery that blocks a stop OPENS the next turn: its K trace
        lands on the successor chain — where the continuation's response
        will complete the turn — not on the chain of the turn it blocked
        (same shape as an operator turn: incoming K + assistant delta share
        one chain)."""
        _seed(self.brain, 'S', 'chained tap')
        self.assertEqual(_stop(self.brain, 'S').get('decision'), 'block')
        events = self.brain.query_traces(session_id='S').get('events', [])
        chains = {e.get('ref_type'): e.get('chain_id') for e in events}
        self.assertIn('self_message', chains)
        # The blocked turn's own trace (heartbeat here — _stop without a
        # prior recall is non-conversational) sits on the PRIOR chain.
        self.assertNotEqual(chains['self_message'], chains.get('heartbeat'))


class TestDeliveryContinuationClassification(BrainTestBase):
    """The stop AFTER a delivery-block is the turn's reaction. The gate is
    trace_contract.arms_continuation over the dial dict (read live so a test
    can simulate a flip via patch.dict — in production the dict only changes
    with a source edit + restart). Dial-off (today) → heartbeat, matching
    pre-substrate behavior; dial-on → a real assistant_message. Cadence
    never moves either way."""
    needs_embedder = False

    def _flipped(self):
        from unittest import mock
        from servers import trace_contract as tc
        return mock.patch.dict(
            tc.S0_CONVERSATIONAL_INCOMING, {'self_message': True})

    def _continuation(self, sid, dial_on):
        from contextlib import nullcontext
        _seed(self.brain, sid, 'react to this')
        with self._flipped() if dial_on else nullcontext():
            self.assertEqual(_stop(self.brain, sid).get('decision'), 'block')
            _stop(self.brain, sid)   # the continuation's stop
        return self.brain.query_traces(session_id=sid).get('events', [])

    def test_dial_off_continuation_stays_heartbeat(self):
        events = self._continuation('SOFF', dial_on=False)
        refs = [e.get('ref_type') for e in events]
        self.assertNotIn('assistant_message', refs)
        self.assertIn('heartbeat', refs)

    def test_dial_on_continuation_is_the_turns_reaction(self):
        events = self._continuation('SON', dial_on=True)
        by_ref = {}
        for e in events:
            by_ref.setdefault(e.get('ref_type'), []).append(e)
        self.assertIn('assistant_message', by_ref)
        # The reaction shares the delivery's chain — one turn, incoming + said.
        self.assertEqual(by_ref['assistant_message'][0].get('chain_id'),
                         by_ref['self_message'][0].get('chain_id'))
        # And the Scribe cadence never moves: no user_message rows exist.
        self.assertEqual(self.brain.turns_since_last_encode('SON'), 0)

    def test_reaction_only_follows_a_blocking_delivery(self):
        # No delivery → flip state is irrelevant; a bare wakeup stays a
        # heartbeat even flipped on.
        with self._flipped():
            _stop(self.brain, 'SBARE')
        refs = [e.get('ref_type') for e in
                self.brain.query_traces(session_id='SBARE').get('events', [])]
        self.assertNotIn('assistant_message', refs)

    def test_stamp_is_consumed_by_an_operator_turn(self):
        # A real prompt lands on the continuation's stop: conversational wins
        # AND the stamp is read-and-cleared — asserted on the session state
        # directly, so removing the clear fails this test.
        from servers.daemon_hooks import hook_recall
        _seed(self.brain, 'SESC', 'tap then interrupt')
        with self._flipped():
            self.assertEqual(_stop(self.brain, 'SESC').get('decision'), 'block')
            self.assertNotEqual(
                self.brain.get_or_create_session('SESC').last_delivery_stop, -1)
            # Operator interrupts the continuation and types a real prompt.
            hook_recall(self.brain, {'session_id': 'SESC',
                                     'prompt': 'a real interrupting prompt'}, [])
            _stop(self.brain, 'SESC')          # conversational turn consumes
        self.assertEqual(
            self.brain.get_or_create_session('SESC').last_delivery_stop, -1)

    def test_stale_stamp_is_disarmed_by_boot_and_window(self):
        # An ESC'd continuation fires no Stop — counter and stamp freeze. The
        # two guards: a boot/resume disarms (reset_session_activity), and a
        # stamp older than the freshness window never matches.
        from servers.trace_contract import DELIVERY_REACTION_WINDOW_MIN
        from servers.clock import iso_cutoff
        _seed(self.brain, 'SFRZ', 'tap then vanish')
        with self._flipped():
            self.assertEqual(_stop(self.brain, 'SFRZ').get('decision'), 'block')
            ctx = self.brain.get_or_create_session('SFRZ')
            self.assertNotEqual(ctx.last_delivery_stop, -1)
            # Hours pass (age the stamp past the window), then a bare wakeup.
            ctx.last_delivery_armed_at = iso_cutoff(
                minutes=DELIVERY_REACTION_WINDOW_MIN + 5)
            _stop(self.brain, 'SFRZ')
        events = self.brain.query_traces(session_id='SFRZ').get('events', [])
        refs = [e.get('ref_type') for e in events]
        self.assertNotIn('assistant_message', refs)   # wakeup stayed heartbeat
        # And a boot disarms outright.
        _seed(self.brain, 'SFRZ', 'tap again')
        with self._flipped():
            self.assertEqual(_stop(self.brain, 'SFRZ').get('decision'), 'block')
        self.brain.reset_session_activity('SFRZ')
        self.assertEqual(
            self.brain.get_or_create_session('SFRZ').last_delivery_stop, -1)


class TestCrossHookConsumeOnce(BrainTestBase):
    needs_embedder = False

    def test_pretool_leaves_tap_then_stop_delivers_once(self):
        # Stop-only: PreToolUse stays silent and does not consume, so the Stop
        # block delivers — exactly once (the next stop finds nothing, no loop).
        _seed(self.brain, 'S', 'single tap')
        pre = _edit(self.brain, 'S', 'a.py')
        self.assertNotIn('single tap', _pretool_ac(pre))     # PreToolUse silent
        first = _stop(self.brain, 'S')
        self.assertEqual(first.get('decision'), 'block')
        self.assertIn('single tap', first.get('reason', ''))
        self.assertNotEqual(_stop(self.brain, 'S').get('decision'), 'block')


if __name__ == '__main__':
    unittest.main()
