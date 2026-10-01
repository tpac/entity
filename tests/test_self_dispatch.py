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
        self.assertIn('streams', r['result'])
        self.assertIn('line', r['result'])

    def test_peek_handler_enveloped(self):
        r = _handle_self_peek(self.brain, {'stream_id': 'A'}, [])
        _assert_envelope(self, r)
        self.assertIn('found', r['result'])

    def test_send_handler_enveloped(self):
        # Graceful resolution: a target must be a full session id, a live label/prefix,
        # or 'broadcast' — a bare toy id no longer resolves (loud-on-dead-target is the
        # point). A full session UUID is canonical and resolves directly.
        r = _handle_self_send(
            self.brain,
            {'to': 'aaaaaaaa-1111-2222-3333-444444444444', 'body': 'hi B', 'from_session': 'A'}, [])
        _assert_envelope(self, r)
        self.assertIn('id', r['result'])
        self.assertNotIn('note_to_self', r['result'])   # only a send to your own id

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

    def test_peek_resolves_short_id_to_full_stream(self):
        """self_peek takes an id-prefix (the 8-char short you see in a message), not
        only the full id — it routes through the SAME resolver self_send's target
        uses, so a short peeks the full stream."""
        full = 'bbbbbbbb-1111-2222-3333-444444444444'
        # Put the stream in the resolver's reach (a recent courier sender).
        signal.send(self.brain, from_session=full,
                    address=self_contract.address_for_stream('rcpt'), body='x')
        r = _handle_self_peek(self.brain, {'stream_id': full[:8]}, [])
        _assert_envelope(self, r)
        self.assertEqual(r['result']['session_id'], full)   # short → full stream

    def test_send_canonical_caller_id_beats_short_from_session(self):
        """Write-boundary fix: an explicit SHORT from_session must NOT override the
        proxy-stamped full caller id — else the courier stores one stream under two
        id formats and the resolver false-positives ambiguity (db79e0c1 / 41c6ebed)."""
        full = 'cccccccc-1111-2222-3333-444444444444'
        target = 'eeeeeeee-1111-2222-3333-444444444444'
        _handle_self_send(self.brain, {
            'to': target, 'body': 'coordinate',
            'from_session': full[:8],          # caller mistakenly passes its SHORT
            '_caller_session': full}, [])        # proxy stamped the full id
        stored = self.brain.logs_conn.execute(
            "SELECT from_session FROM self_inflight WHERE address = ?",
            (self_contract.address_for_stream(target),)).fetchone()[0]
        self.assertEqual(stored, full,
                         "canonical full caller id must win over a short from_session")

    def test_send_to_own_short_id_delivers_to_self(self):
        """A stream may address its own short id: the sender is not dropped from
        resolution, and the inbox delivers the note back to it."""
        me = 'dddddddd-1111-2222-3333-444444444444'
        self.brain.stamp_boot_liveness(me)                 # in the live roster
        sent = _handle_self_send(self.brain, {
            'to': me[:8], 'body': 'note to me', '_caller_session': me}, [])
        _assert_envelope(self, sent)
        self.assertIs(sent['result'].get('note_to_self'), True)   # visible at send time
        drained = _handle_self_inbox(self.brain, {'session_id': me}, [])
        self.assertEqual([m['body'] for m in drained['result']['messages']], ['note to me'])


if __name__ == '__main__':
    unittest.main()
