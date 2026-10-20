"""Dispatch-layer contract for the self-channel handlers.

The Phase 1/2a tests (test_self_presence, test_self_signal) exercise the inner
presence.*/signal.* functions, which return raw payloads and pass. But the
daemon table-dispatch (daemon_server._dispatch) sends a handler's return
VERBATIM to the MCP client, so each handler MUST wrap its payload in the
{"ok": True, "result": ...} envelope. A raw return reaches brain_mcp as a
falsy `ok` with no `error` and surfaces as "Unknown daemon error" — a
successful call misread as a failure, silently (no exception, nothing logged).

This is the coverage that was missing when the self-channel first ran live:
nothing had ever invoked _handle_self_* — only the inner functions.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.dispatch_self import (
    _handle_self_presence, _handle_self_peek, _handle_self_send, _handle_self_inbox)
from servers.channels.self_channel import signal, self_contract
from servers.trace_contract import session_short


def _assert_envelope(case, result):
    case.assertIsInstance(result, dict)
    case.assertIs(result.get("ok"), True,
                  "handler must return the {'ok': True, ...} envelope, got: %r" % (result,))
    case.assertIn("result", result, "enveloped return must carry a 'result' payload")


class TestSelfDispatchEnvelope(BrainTestBase):
    needs_embedder = False

    def test_presence_handler_enveloped(self):
        r = _handle_self_presence(self.brain, {'session_id': 'A', 'limit': 3}, [])
        _assert_envelope(self, r)
        self.assertIn('sessions', r['result'])
        self.assertIn('line', r['result'])

    def test_peek_handler_enveloped(self):
        r = _handle_self_peek(self.brain, {'session_id': 'A'}, [])
        _assert_envelope(self, r)
        self.assertIn('found', r['result'])

    def test_send_handler_enveloped(self):
        # Graceful resolution: a target must be a full session id, a live session short,
        # or 'broadcast' — a bare toy id no longer resolves (loud-on-dead-target is the
        # point). A full session UUID is canonical and resolves directly.
        r = _handle_self_send(
            self.brain,
            {'to': 'aaaaaaaa-1111-2222-3333-444444444444', 'body': 'hi B', 'from_session': 'A'}, [])
        _assert_envelope(self, r)
        self.assertIn('id', r['result'])
        self.assertNotIn('note_to_self', r['result'])   # only a send to your own id

    def test_send_with_an_empty_body_is_an_error_envelope(self):
        r = _handle_self_send(
            self.brain,
            {'to': 'aaaaaaaa-1111-2222-3333-444444444444', 'from_session': 'A',
             'body': '   '}, [])
        self.assertIs(r.get('ok'), False)
        self.assertIn('empty body', r.get('error', ''))

    def test_send_over_the_limit_is_an_error_envelope(self):
        r = _handle_self_send(
            self.brain,
            {'to': 'aaaaaaaa-1111-2222-3333-444444444444', 'from_session': 'A',
             'body': 'x' * (self_contract.SIGNAL_BODY_MAX + 1)}, [])
        self.assertIs(r.get('ok'), False)
        self.assertIn('the limit is {:,}'.format(self_contract.SIGNAL_BODY_MAX),
                      r.get('error', ''))

    def test_inbox_handler_enveloped(self):
        r = _handle_self_inbox(self.brain, {'session_id': 'B'}, [])
        _assert_envelope(self, r)
        self.assertIn('messages', r['result'])

    def test_send_then_inbox_roundtrip_through_dispatch(self):
        """End-to-end through the enveloped dispatch layer (not the inner fns)."""
        target = 'aaaaaaaa-1111-2222-3333-444444444444'   # full session id → resolves canonically
        sent = _handle_self_send(
            self.brain, {'to': target, 'body': 'ping B', 'from_session': 'A'}, [])
        _assert_envelope(self, sent)
        drained = _handle_self_inbox(self.brain, {'session_id': target}, [])
        _assert_envelope(self, drained)
        bodies = [m['body'] for m in drained['result']['messages']]
        self.assertIn('ping B', bodies)

    def test_peek_resolves_short_id_to_full_session(self):
        """self_peek takes the short you see in a message (session_short), not
        only the full id — it routes through the SAME resolver self_send's target
        uses, so a short peeks the full session."""
        full = 'bbbbbbbb-1111-2222-3333-444444444444'
        # Put the session in the resolver's reach (a recent courier sender).
        signal.send(self.brain, from_session=full,
                    address=self_contract.address_for_session('rcpt'), body='x')
        r = _handle_self_peek(self.brain, {'session_id': session_short(full)}, [])
        _assert_envelope(self, r)
        self.assertEqual(r['result']['session_id'], full)   # short → full session

    def test_send_canonical_caller_id_beats_short_from_session(self):
        """Write-boundary fix: an explicit SHORT from_session must NOT override the
        proxy-stamped full caller id — else the courier stores one session under two
        id formats and the resolver false-positives ambiguity (db79e0c1 / 41c6ebed)."""
        full = 'cccccccc-1111-2222-3333-444444444444'
        target = 'eeeeeeee-1111-2222-3333-444444444444'
        _handle_self_send(self.brain, {
            'to': target, 'body': 'coordinate',
            'from_session': full[:8],          # caller mistakenly passes its SHORT
            '_caller_session': full}, [])        # proxy stamped the full id
        stored = self.brain.logs_conn.execute(
            "SELECT from_session FROM self_inflight WHERE address = ?",
            (self_contract.address_for_session(target),)).fetchone()[0]
        self.assertEqual(stored, full,
                         "canonical full caller id must win over a short from_session")

    def test_inbox_drains_the_stamped_caller_not_a_written_session_id(self):
        """Identity is the stamp: a session_id the model writes beside it (here
        its own short) must not redirect the drain to a phantom inbox."""
        me = 'f1f1f1f1-1111-2222-3333-444444444444'
        _handle_self_send(self.brain, {
            'to': me, 'body': 'for the stamped caller', 'from_session': 'A'}, [])
        drained = _handle_self_inbox(self.brain, {
            '_caller_session': me, 'session_id': session_short(me)}, [])
        self.assertEqual([m['body'] for m in drained['result']['messages']],
                         ['for the stamped caller'])

    def test_presence_excludes_the_stamped_caller_not_a_written_session_id(self):
        me = 'f2f2f2f2-1111-2222-3333-444444444444'
        other = 'f3f3f3f3-1111-2222-3333-444444444444'
        for sess in (me, other):
            self.brain._trace_dal.append(
                chain_id='s0-presence-%s' % sess, scale='s0', event_type='K',
                ref_type='user_message', summary='presence probe',
                session_id=sess)
        r = _handle_self_presence(self.brain, {
            '_caller_session': me, 'session_id': session_short(me)}, [])
        ids = {s['session_id'] for s in r['result']['sessions']}
        self.assertIn(other, ids)
        self.assertNotIn(me, ids, "the stamped caller must be left out of its own roster")

    def test_calls_that_need_a_caller_refuse_an_empty_identity(self):
        """An unattributed proxy call arrives with an empty stamp. The four
        tools whose answer is about the caller refuse it by name rather than
        answer for nobody — an empty inbox reads as "no messages" — and a
        written session_id / from_session beside the stamp claims nothing."""
        from servers.dispatch_self import _handle_self_inbox_peek, _handle_self_outbox
        anon = {'_caller_session': '', 'session_id': 'A', 'from_session': 'A'}
        for name, handler, extra in (
                ('self_inbox', _handle_self_inbox, {}),
                ('self_inbox_peek', _handle_self_inbox_peek, {}),
                ('self_outbox', _handle_self_outbox, {}),
                ('self_send', _handle_self_send,
                 {'to': 'aaaaaaaa-1111-2222-3333-444444444444', 'body': 'from nobody'})):
            r = handler(self.brain, dict(anon, **extra), [])
            self.assertIs(r.get('ok'), False, '%s answered an unidentified caller: %r' % (name, r))
            self.assertIn('no session identity', r.get('error', ''), name)
        stored = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM self_inflight WHERE body = 'from nobody'").fetchone()[0]
        self.assertEqual(stored, 0, 'a message with no sender was stored')

    def test_send_to_own_short_id_delivers_to_self(self):
        """A session may address its own short id: the sender is not dropped from
        resolution, and the inbox delivers the note back to it."""
        me = 'dddddddd-1111-2222-3333-444444444444'
        self.brain.stamp_boot_liveness(me)                 # in the live roster
        sent = _handle_self_send(self.brain, {
            'to': session_short(me), 'body': 'note to me', '_caller_session': me}, [])
        _assert_envelope(self, sent)
        self.assertIs(sent['result'].get('note_to_self'), True)   # visible at send time
        drained = _handle_self_inbox(self.brain, {'session_id': me}, [])
        self.assertEqual([m['body'] for m in drained['result']['messages']], ['note to me'])


if __name__ == '__main__':
    unittest.main()
