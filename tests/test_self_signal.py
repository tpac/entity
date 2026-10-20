"""Phase 2a self-channel signal — directed/broadcast courier (send + drain + reap)."""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.clock import iso_cutoff
from servers.channels.self_channel import signal, self_contract
from servers.trace_contract import session_short


class TestSelfSignal(BrainTestBase):
    needs_embedder = False

    def test_send_and_directed_consume_once(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='hi B')
        first = signal.drain_inbox(self.brain, 'B')
        self.assertEqual([m['body'] for m in first], ['hi B'])
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])  # consumed once

    def test_directed_not_delivered_to_others(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='for B only')
        self.assertEqual(signal.drain_inbox(self.brain, 'C'), [])  # not for C
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')],
                         ['for B only'])

    def test_broadcast_fans_out(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='everyone')
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')],
                         ['everyone'])
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'C')],
                         ['everyone'])  # each live session gets it once
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])  # B already consumed

    def test_rendered_field_and_attribution(self):
        signal.send(self.brain, from_session='abcd1234ef',
                    address=self_contract.address_for_session('B'), body='stop editing X')
        msg = signal.drain_inbox(self.brain, 'B')[0]
        self.assertIn('stop editing X', msg['rendered'])
        self.assertEqual(msg['from'], 'abcd1234ef')  # its short: a 10-char id shows whole

    def test_empty_body_rejected(self):
        with self.assertRaises(ValueError):
            signal.send(self.brain, from_session='A',
                        address=self_contract.address_for_session('B'), body='   ')

    def test_expired_not_delivered_then_reaped(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='stale')
        past = iso_cutoff(hours=1)   # expires_at in the past → already expired
        with self.brain.write_lock:
            self.brain.logs_conn.execute(
                "UPDATE self_inflight SET expires_at = ? WHERE address = ?",
                (past, self_contract.address_for_session('B')))
            self.brain.logs_conn.commit()
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])   # expired → not delivered
        self.assertEqual(signal.reap_expired(self.brain), 1)         # dead-letter swept

    def test_sender_does_not_hear_own_broadcast(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='heads up all')
        self.assertEqual(signal.drain_inbox(self.brain, 'A'), [])   # sender excluded from own broadcast
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')],
                         ['heads up all'])

    def test_directed_note_to_self_delivers_as_own(self):
        """Only broadcast drops the sender: a directed message to your own id is a
        note to yourself — delivered, marked own, never a first-contact intro."""
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('A'), body='note to me')
        drained = signal.drain_inbox(self.brain, 'A')
        self.assertEqual([m['body'] for m in drained], ['note to me'])
        self.assertTrue(drained[0]['own'])
        self.assertFalse(drained[0]['first_contact'])
        self.assertIn('your own note (this session) says:', drained[0]['rendered'])
        self.assertEqual(signal.drain_inbox(self.brain, 'A'), [])   # consumed once

    def test_note_to_self_block_is_not_framed_as_another_session(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('A'), body='remember X')
        block, n = signal.drain_and_render(self.brain, 'A')
        self.assertEqual(n, 1)
        self.assertIn('a note you sent to this session', block)
        self.assertIn('your own note (this session) says:', block)
        self.assertNotIn('other sessions', block)
        self.assertNotIn('first contact', block)

    def test_mixed_block_keeps_other_session_frame_and_labels_own_note(self):
        signal.send(self.brain, from_session='B',
                    address=self_contract.address_for_session('A'), body='from B')
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('A'), body='from me')
        block, n = signal.drain_and_render(self.brain, 'A')
        self.assertEqual(n, 2)
        self.assertIn('other sessions', block)
        self.assertIn('other session (id:B) says:', block)
        self.assertIn('your own note (this session) says:', block)

    def test_reap_cleans_delivered_orphans(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='ephemeral')
        signal.drain_inbox(self.brain, 'B')  # B consumes → a self_delivered row now exists
        past = iso_cutoff(hours=1)
        with self.brain.write_lock:
            self.brain.logs_conn.execute("UPDATE self_inflight SET expires_at = ?", (past,))
            self.brain.logs_conn.commit()
        signal.reap_expired(self.brain)
        inflight = self.brain.logs_conn.execute("SELECT COUNT(*) FROM self_inflight").fetchone()[0]
        delivered = self.brain.logs_conn.execute("SELECT COUNT(*) FROM self_delivered").fetchone()[0]
        self.assertEqual(inflight, 0)
        self.assertEqual(delivered, 0, "orphan delivery rows must be swept with the message")

    def test_refs_persisted(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'),
                    body='see node', refs=['node-abc', 'file.py'])
        stored = self.brain.logs_conn.execute("SELECT refs FROM self_inflight").fetchone()[0]
        self.assertIn('node-abc', stored)

    # ── Truncation contract: SEND stores in full, no silent slice ──

    def test_send_stores_full_body_no_silent_cap(self):
        """The body is stored verbatim — no SIGNAL_BODY_MAX slice. Truncation,
        if any, happens only at delivery render, loudly."""
        long_body = "z" * 1500
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body=long_body)
        stored = self.brain.logs_conn.execute("SELECT body FROM self_inflight").fetchone()[0]
        self.assertEqual(len(stored), 1500)

    def test_send_stores_all_refs_no_silent_cap(self):
        """All refs persist — no REFS_MAX slice silently dropping the tether."""
        refs = ['n%d' % i for i in range(30)]
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='many refs', refs=refs)
        stored = self.brain.logs_conn.execute("SELECT refs FROM self_inflight").fetchone()[0]
        self.assertIn('n29', stored)   # the 30th ref survived (would be gone at a cap of 12)

    def test_outbox_shows_delivery_status(self):
        """A sender can see who drained its message, and whether a directed target
        is still pending — silence read correctly (delivered-vs-never-delivered)."""
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='ping B')
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='hey all')
        # Before anyone drains: directed message is pending, nobody delivered.
        ob = signal.outbox(self.brain, from_session='A')['messages']
        self.assertEqual(len(ob), 2)
        directed = [m for m in ob if m.get('target')][0]
        self.assertTrue(directed['pending'])
        self.assertEqual(directed['delivered_to'], [])
        # B drains → directed no longer pending; delivered_to names B.
        signal.drain_inbox(self.brain, to_session='B')
        directed2 = [m for m in signal.outbox(self.brain, from_session='A')['messages']
                     if m.get('target')][0]
        self.assertFalse(directed2['pending'])
        self.assertTrue(any(d['to'] == 'B' for d in directed2['delivered_to']))

    def test_outbox_empty_for_silent_session(self):
        """A session that never sent anything has an empty outbox (no crash)."""
        self.assertEqual(signal.outbox(self.brain, from_session='Z')['messages'], [])

    def test_drain_attributes_by_short_id(self):
        """No self-labeling — a delivered message is attributed by the sender's
        short id, in both 'from' and the render."""
        signal.send(self.brain, from_session='AAAAAAAA',
                    address=self_contract.address_for_session('streamB'), body='hi B')
        drained = signal.drain_inbox(self.brain, to_session='streamB')
        self.assertEqual(drained[0]['from'], 'AAAAAAAA')
        self.assertIn('other session (id:AAAAAAAA) says:', drained[0]['rendered'])

    def test_first_contact_flag_set_once_per_sender(self):
        """drain flags the FIRST message from a sender as first_contact and
        carries the full from_session; later ones from the same sender are lean."""
        addr = self_contract.address_for_session('rcpt1')
        signal.send(self.brain, from_session='senderXX', address=addr, body='one')
        d1 = signal.drain_inbox(self.brain, to_session='rcpt1')
        self.assertTrue(d1[0]['first_contact'])
        self.assertEqual(d1[0]['from_full'], 'senderXX')
        signal.send(self.brain, from_session='senderXX', address=addr, body='two')
        d2 = signal.drain_inbox(self.brain, to_session='rcpt1')
        self.assertFalse(d2[0]['first_contact'])

    def test_first_contact_intro_rendered_then_dropped(self):
        """drain_and_render attaches the sender's PEEK on the first message from a
        session — intro carries the reply hint AND what they're working on — and
        nothing on the next. (Regression: the peek enrichment must actually wire.)"""
        self.brain.set_config('session_context_senderYY', 'refactoring the courier')
        addr = self_contract.address_for_session('rcpt2')
        signal.send(self.brain, from_session='senderYY', address=addr, body='hello')
        block1, _ = signal.drain_and_render(self.brain, 'rcpt2')
        self.assertIn('first contact', block1)
        self.assertIn('self_send to="senderYY"', block1)
        self.assertIn('working on: refactoring the courier', block1)  # peek wired in
        signal.send(self.brain, from_session='senderYY', address=addr, body='again')
        block2, _ = signal.drain_and_render(self.brain, 'rcpt2')
        self.assertNotIn('first contact', block2)

    def test_resolve_short_matches_recent_sender(self):
        """A session that messaged me but isn't in the live roster is still
        reply-able by its short id — its (full-UUID) from_session is in the courier,
        so its session short resolves to it."""
        full = 'dabc1234-5678-4abc-8def-000000000001'
        signal.send(self.brain, from_session=full,
                    address=self_contract.address_for_session('me'), body='hi')
        addr, err = signal.resolve_to(self.brain, session_short(full))
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_session(full))

    def test_resolve_short_and_full_of_one_session_not_ambiguous(self):
        """Regression (live 2026-06-24): a short naming ONE logical session must
        resolve, even when the courier holds BOTH the full session id AND a leaked
        short form of it as a sender. The leaked short is not a full session id, so
        the candidate filter excludes it — only the full matches, so it resolves
        rather than reading as two 'live sessions'. The original bug printed the
        same id twice: \"'37a32ee9' matches 2 live sessions (37a32ee9, 37a32ee9)\"."""
        full = '37a32ee9-9770-46ac-ab03-0a87b6762647'
        leaked = full[:8]                                  # the 2026-06 leak's form
        # The leaked short + the canonical full, both in the courier for one session.
        signal.send(self.brain, from_session=full,
                    address=self_contract.address_for_session('me'), body='from full')
        signal.send(self.brain, from_session=leaked,
                    address=self_contract.address_for_session('me'), body='from short')
        addr, err = signal.resolve_to(self.brain, session_short(full))
        self.assertIsNone(err, "a unique session under its short must resolve, not error")
        # Resolves to the canonical FULL form (the deliverable address), not the short.
        self.assertEqual(addr, self_contract.address_for_session(full))

    def test_resolve_genuine_ambiguity_names_full_ids(self):
        """Two DISTINCT full sessions sharing a short ARE ambiguous — and the
        error must name the FULL session ids, not the short, or the 'use the full
        session id to disambiguate' instruction is impossible to act on (you'd be
        handed the very short that's ambiguous)."""
        a = '11111111-1111-4aaa-8bbb-dddddddddddd'
        b = '22222222-2222-4aaa-8bbb-dddddddddddd'
        signal.send(self.brain, from_session=a,
                    address=self_contract.address_for_session('me'), body='from a')
        signal.send(self.brain, from_session=b,
                    address=self_contract.address_for_session('me'), body='from b')
        addr, err = signal.resolve_to(self.brain, 'dddddddddddd')
        self.assertIsNone(addr)
        self.assertIn(a, err)        # FULL ids named — actionable
        self.assertIn(b, err)
        self.assertIn('matches 2 live sessions', err)

    # ── Phase 2b: render + delivery-into-Observation ──

    def test_render_received_block_composes(self):
        msgs = [
            {"body": "first tap", "from": "aaaa1111", "rendered": '⚡ aaaa1111 says:\n   "first tap"'},
            {"body": "second tap", "from": "bbbb2222", "rendered": '⚡ bbbb2222 says:\n   "second tap"'},
        ]
        block, shown = self_contract.render_received_block(msgs)
        self.assertEqual(shown, 2)
        self.assertIn("first tap", block)
        self.assertIn("second tap", block)
        self.assertIn("session", block.lower())  # framed with a header

    def test_render_received_block_empty(self):
        self.assertEqual(self_contract.render_received_block([]), ("", 0))

    def test_render_received_block_overflow_is_loud(self):
        """Over-budget input is bounded, counts what it shows, and names the
        call that reads the rest — no silent cut."""
        msgs = [{"body": "x" * 300, "from": "s%02d" % i,
                 "rendered": ('⚡ s%02d says:\n   "' % i) + "x" * 300 + '"'} for i in range(20)]
        block, shown = self_contract.render_received_block(msgs, cap=800)
        self.assertLess(len(block), 800 + 200)   # bounded near the cap
        self.assertTrue(0 < shown < 20)
        self.assertEqual(block.count(' says:'), shown)
        self.assertIn("(+%d more waiting in your inbox — call self_inbox" % (20 - shown),
                      block)

    def test_render_frame_describes_shown_messages_only(self):
        """Own notes shown with another session's message waiting behind them
        get the own-note frame — the frame describes what the block shows."""
        msgs = [{"body": "note one", "from": "me", "own": True},
                {"body": "note two", "from": "me", "own": True},
                {"body": "w" * 900, "from": "s1", "own": False}]
        block, shown = self_contract.render_received_block(msgs, cap=500)
        self.assertEqual(shown, 2)
        self.assertIn("a note you sent to this session", block)
        self.assertNotIn("from your other sessions", block)
        self.assertIn("(+1 more waiting in your inbox", block)

    def test_render_shows_a_message_at_the_limit_whole(self):
        """Delivery never cuts a message: one at SIGNAL_BODY_MAX, over the block
        budget on its own, still shows whole as the block's first message."""
        body = "y" * self_contract.SIGNAL_BODY_MAX
        block, shown = self_contract.render_received_block([{"body": body, "from": "s1"}])
        self.assertEqual(shown, 1)
        self.assertIn(body, block)

    def test_send_refuses_a_body_over_the_limit(self):
        """The limit sits at send: the sender learns the length and the limit,
        and nothing reaches the inbox."""
        addr = self_contract.address_for_session('B')
        signal.send(self.brain, from_session='A', address=addr,
                    body='z' * self_contract.SIGNAL_BODY_MAX)          # at the limit: kept
        with self.assertRaises(ValueError) as cm:
            signal.send(self.brain, from_session='A', address=addr,
                        body='z' * (self_contract.SIGNAL_BODY_MAX + 1))
        self.assertIn('{:,} characters and the limit is {:,}'.format(
            self_contract.SIGNAL_BODY_MAX + 1, self_contract.SIGNAL_BODY_MAX),
            str(cm.exception))
        self.assertEqual(len(signal.peek_inbox(self.brain, 'B')), 1)

    def test_drain_and_render_and_consumes(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='ping B')
        block, n = signal.drain_and_render(self.brain, 'B')
        self.assertEqual(n, 1)
        self.assertIn('ping B', block)
        # consume-once: the second drain finds nothing
        block2, n2 = signal.drain_and_render(self.brain, 'B')
        self.assertEqual(n2, 0)
        self.assertEqual(block2, "")

    def test_drain_and_render_empty_inbox(self):
        block, n = signal.drain_and_render(self.brain, 'NOBODY-HOME')
        self.assertEqual(n, 0)
        self.assertEqual(block, "")

    def test_drain_and_render_overflow_stays_in_the_inbox(self):
        """A message past the block budget is not consumed: it stays pending,
        self_inbox returns it whole, and nothing is lost to the budget."""
        addr = self_contract.address_for_session('B')
        bodies = ['msg %02d ' % i + 'z' * 900 for i in range(8)]
        for b in bodies:
            signal.send(self.brain, from_session='A', address=addr, body=b)
        block, n = signal.drain_and_render(self.brain, 'B')
        self.assertTrue(0 < n < len(bodies))
        self.assertIn('(+%d more waiting in your inbox' % (len(bodies) - n), block)
        self.assertEqual([m['body'] for m in signal.peek_inbox(self.brain, 'B')],
                         bodies[n:])                      # the rest still pending
        rest = signal.drain_inbox(self.brain, 'B')
        self.assertEqual([m['body'] for m in rest], bodies[n:])  # whole, in order
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])

    def test_drain_and_render_overflow_arrives_at_the_next_stop(self):
        """Each Stop shows at least one message, so a backlog drains to empty
        across turn ends with every message shown once."""
        addr = self_contract.address_for_session('B')
        bodies = ['tap %02d ' % i + 'q' * 1500 for i in range(6)]
        for b in bodies:
            signal.send(self.brain, from_session='A', address=addr, body=b)
        seen = []
        for _ in range(len(bodies)):
            block, n = signal.drain_and_render(self.brain, 'B')
            if not n:
                break
            seen += [b for b in bodies if b[:7] in block]
        self.assertEqual(seen, bodies)
        self.assertEqual(signal.drain_and_render(self.brain, 'B'), ("", 0))

    def test_first_contact_once_per_sender_in_a_batch(self):
        """The first message from a session carries the intro; a second one in
        the same delivery does not."""
        addr = self_contract.address_for_session('rcpt3')
        signal.send(self.brain, from_session='senderZZ', address=addr, body='one')
        signal.send(self.brain, from_session='senderZZ', address=addr, body='two')
        block, n = signal.drain_and_render(self.brain, 'rcpt3')
        self.assertEqual(n, 2)
        self.assertEqual(block.count('first contact'), 1)


class TestResolveStream(BrainTestBase):
    """The shared session-reference resolver — full id / session short → canonical
    full id. The single 'name a session' path behind self_send's target AND self_peek;
    returns (full_id | None, error), tool-agnostic so every caller can wrap it."""
    needs_embedder = False

    def test_full_id_passes_through_even_when_not_live(self):
        """A full session UUID is canonical — honored without a roster hit (it
        drains within TTL even if the target isn't awake this instant)."""
        full = 'abcdef01-2345-6789-abcd-ef0123456789'
        fid, err = signal.resolve_session(self.brain, full)
        self.assertIsNone(err)
        self.assertEqual(fid, full)

    def test_short_unique_recent_sender_resolves_to_full(self):
        full = 'dabc1234-5678-4abc-8def-000000000002'
        signal.send(self.brain, from_session=full,
                    address=self_contract.address_for_session('me'), body='hi')
        fid, err = signal.resolve_session(self.brain, session_short(full))
        self.assertIsNone(err)
        self.assertEqual(fid, full)

    def test_short_and_full_of_one_session_resolves_to_full(self):
        """The courier holding BOTH the canonical full id and a leaked short sender
        (one session, two strings) resolves to the full: the leaked short is not a
        full session id, so the candidate filter drops it — no phantom ambiguity."""
        full = '37a32ee9-9770-46ac-ab03-0a87b6762647'
        signal.send(self.brain, from_session=full,
                    address=self_contract.address_for_session('me'), body='full')
        signal.send(self.brain, from_session=full[:8],
                    address=self_contract.address_for_session('me'), body='short')
        fid, err = signal.resolve_session(self.brain, session_short(full))
        self.assertIsNone(err)
        self.assertEqual(fid, full)

    def test_same_full_in_roster_and_courier_resolves_once(self):
        """The SAME session surfacing from BOTH the roster and the courier must dedup
        to one match, not read as ambiguous — dict.fromkeys carries this (the
        is_session_id filter alone wouldn't collapse the identical id twice)."""
        full = 'aaaaaaaa-1111-4222-8333-444444444444'
        self.brain.stamp_boot_liveness(full)              # → present_sessions (roster)
        signal.send(self.brain, from_session=full,         # → recent courier sender
                    address=self_contract.address_for_session('me'), body='x')
        fid, err = signal.resolve_session(self.brain, session_short(full))
        self.assertIsNone(err)
        self.assertEqual(fid, full)

    def test_caller_excluded_from_own_resolution(self):
        """exclude_session removes that session from the pool — the form self_peek
        uses, so a peek by short never resolves to the session looking."""
        me = 'cafe1234-5678-4abc-8def-000000000003'
        signal.send(self.brain, from_session=me,
                    address=self_contract.address_for_session('someone'), body='x')
        fid, err = signal.resolve_session(self.brain, session_short(me), exclude_session=me)
        self.assertIsNone(fid)
        self.assertIn('no live session matches', err)

    def test_genuine_ambiguity_names_full_ids_tool_agnostic(self):
        """Distinct full ids sharing a short ARE ambiguous — the error names the
        FULL ids and carries NO tool prefix (self_peek wraps the same resolver)."""
        a = '11111111-1111-4aaa-8bbb-dddddddddddd'
        b = '22222222-2222-4aaa-8bbb-dddddddddddd'
        signal.send(self.brain, from_session=a,
                    address=self_contract.address_for_session('me'), body='a')
        signal.send(self.brain, from_session=b,
                    address=self_contract.address_for_session('me'), body='b')
        fid, err = signal.resolve_session(self.brain, 'dddddddddddd')
        self.assertIsNone(fid)
        self.assertIn(a, err)
        self.assertIn(b, err)
        self.assertNotIn('self_send', err)

    def test_only_the_full_id_or_the_exact_short_resolves(self):
        """A session is named one way: its session short, exactly (or its full
        id). A start, a partial or hyphenated tail, or a hyphen-only ref is
        refused loudly — never matched to a session. Codex ids open with a
        timestamp, so a stale start would land on a same-minute chat."""
        full = '01a131e3-a8a5-7bb0-8415-781037851afe'
        signal.send(self.brain, from_session=full,
                    address=self_contract.address_for_session('me'), body='hi')
        self.assertEqual(signal.resolve_session(self.brain, '781037851afe'), (full, None))
        for ref in ('01a131e3', '1afe', '7bb0-8415-781037851afe', '-', '--'):
            fid, err = signal.resolve_session(self.brain, ref)
            self.assertIsNone(fid, ref)
            self.assertIn('no live session matches', err)

    def test_operator_sender_named_whole(self):
        """The dashboard sends as 'operator-dashboard' — a label, not a session
        id, so a delivered message names it whole, never cut like a session."""
        signal.send(self.brain, from_session='operator-dashboard',
                    address=self_contract.address_for_session('B'), body='from the operator')
        self.assertEqual(signal.peek_inbox(self.brain, 'B')[0]['from'], 'operator-dashboard')
        msg = signal.drain_inbox(self.brain, 'B')[0]
        self.assertEqual(msg['from'], 'operator-dashboard')
        self.assertIn('id:operator-dashboard', msg['rendered'])

    def test_no_match_is_loud(self):
        fid, err = signal.resolve_session(self.brain, 'nobodyhome')
        self.assertIsNone(fid)
        self.assertIn('no live session matches', err)

    def test_empty_ref_rejected(self):
        fid, err = signal.resolve_session(self.brain, '')
        self.assertIsNone(fid)
        self.assertIn('empty', err)

    def _trace(self, session_id):
        """Put a session in trace history without making it live: an s1 row is
        not a turn, so presence does not count it."""
        self.brain._trace_dal.append(
            chain_id='hist-%s' % session_id, scale='s1', event_type='K',
            ref_type='surface_selected', summary='a surface', session_id=session_id)

    def test_history_pool_resolves_a_traced_session_that_is_not_live(self):
        """A trace filter may name any session ever recorded: the history pool
        is every session with traces, where the live pool refuses it."""
        full = '01a131e3-a8a5-7bb0-8415-781037851afe'
        self._trace(full)
        self.assertIn('no live session matches',
                      signal.resolve_session(self.brain, session_short(full))[1])
        self.assertEqual(
            signal.resolve_session(self.brain, session_short(full), pool='history'),
            (full, None))

    def test_history_pool_is_exact(self):
        """History matches the short exactly, like the live pool: a start, a
        partial or a hyphenated tail is refused, naming where it looked."""
        full = '01a131e3-a8a5-7bb0-8415-781037851afe'
        self._trace(full)
        for ref in ('01a131e3', '1afe', '7bb0-8415-781037851afe'):
            fid, err = signal.resolve_session(self.brain, ref, pool='history')
            self.assertIsNone(fid, ref)
            self.assertIn('no session in the trace history matches', err)

    def test_history_pool_names_a_non_uuid_session_by_id_or_short(self):
        """Trace history keeps ids that are not UUIDs (eval and test sessions),
        and trace headers show a short for them too: a filter reaches one by
        its exact id or by that short. The live pool never does."""
        sid = 'roundtrip-tr-sess'
        self._trace(sid)
        for ref in (sid, session_short(sid)):
            self.assertEqual(
                signal.resolve_session(self.brain, ref, pool='history'), (sid, None), ref)
            self.assertIsNone(signal.resolve_session(self.brain, ref)[0], ref)

    def test_history_exact_id_and_another_sessions_short_is_ambiguous(self):
        """A reference that is one stored id exactly and another session's
        short names two sessions — refused, never settled silently."""
        stored = 'abcdefabcdef'
        full = '33333333-3333-4aaa-8bbb-abcdefabcdef'
        self._trace(stored)
        self._trace(full)
        fid, err = signal.resolve_session(self.brain, stored, pool='history')
        self.assertIsNone(fid)
        self.assertIn(stored, err)
        self.assertIn(full, err)

    def test_history_ambiguity_names_full_ids(self):
        a = '11111111-1111-4aaa-8bbb-eeeeeeeeeeee'
        b = '22222222-2222-4aaa-8bbb-eeeeeeeeeeee'
        self._trace(a)
        self._trace(b)
        fid, err = signal.resolve_session(self.brain, 'eeeeeeeeeeee', pool='history')
        self.assertIsNone(fid)
        self.assertIn(a, err)
        self.assertIn(b, err)
        self.assertIn('sessions in the trace history', err)

    def test_unknown_pool_raises_whatever_the_ref(self):
        for ref in ('abcdefabcdef', 'abcdef01-2345-6789-abcd-ef0123456789', ''):
            with self.assertRaises(ValueError, msg=ref):
                signal.resolve_session(self.brain, ref, pool='everywhere')


class TestPeekInbox(BrainTestBase):
    """Read-only peek — the /watch-live arrival detector. Must NOT consume."""
    needs_embedder = False

    def test_peek_returns_pending_without_consuming(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='hi B')
        # Peek twice — same message both times, never consumed.
        self.assertEqual([m['body'] for m in signal.peek_inbox(self.brain, 'B')], ['hi B'])
        self.assertEqual([m['body'] for m in signal.peek_inbox(self.brain, 'B')], ['hi B'])
        # The real drain still delivers it — peek didn't mark it delivered.
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'B')], ['hi B'])
        # And after the drain consumes it, peek sees nothing.
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])

    def test_peek_excludes_already_drained(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='once')
        signal.drain_inbox(self.brain, 'B')                 # consumed
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])  # peek agrees it's gone

    def test_peek_excludes_own_broadcast(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.ADDR_BROADCAST, body='to all')
        self.assertEqual(signal.peek_inbox(self.brain, 'A'), [])   # sender doesn't see own
        self.assertEqual([m['body'] for m in signal.peek_inbox(self.brain, 'B')], ['to all'])

    def test_peek_sees_directed_note_to_self(self):
        """The /watch poller must see a note to self — it's how a session checks
        that its own listener fires."""
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('A'), body='probe')
        signal.send(self.brain, from_session='B',
                    address=self_contract.address_for_session('A'), body='from B')
        peeked = signal.peek_inbox(self.brain, 'A')
        self.assertEqual([(m['body'], m['own']) for m in peeked],
                         [('probe', True), ('from B', False)])

    def test_peek_excludes_expired(self):
        signal.send(self.brain, from_session='A',
                    address=self_contract.address_for_session('B'), body='stale')
        past = iso_cutoff(hours=1)
        with self.brain.write_lock:
            self.brain.logs_conn.execute(
                "UPDATE self_inflight SET expires_at = ? WHERE address = ?",
                (past, self_contract.address_for_session('B')))
            self.brain.logs_conn.commit()
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])

    def test_peek_empty_for_silent_session(self):
        self.assertEqual(signal.peek_inbox(self.brain, 'NOBODY'), [])

    def test_peek_attributes_by_short_id(self):
        signal.send(self.brain, from_session='AAAAAAAA',
                    address=self_contract.address_for_session('B'), body='hi')
        self.assertEqual(signal.peek_inbox(self.brain, 'B')[0]['from'], 'AAAAAAAA')


class TestTTLByCategory(BrainTestBase):
    """Per-message TTL resolved by address: broadcast ephemeral, directed waits
    a day; config-tunable per category."""
    needs_embedder = False

    def test_broadcast_gets_shorter_ttl_than_directed(self):
        b = signal.send(self.brain, from_session='X',
                        address=self_contract.ADDR_BROADCAST, body='who is live')
        d = signal.send(self.brain, from_session='X',
                        address=self_contract.address_for_session('S'), body='for S')
        self.assertIn('expires_at', b)
        self.assertLess(b['expires_at'], d['expires_at'])  # broadcast expires sooner

    def test_unexpired_directed_still_delivers(self):
        signal.send(self.brain, from_session='X',
                    address=self_contract.address_for_session('S'), body='fresh')
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'S')],
                         ['fresh'])

    def test_config_override_can_kill_broadcast_immediately(self):
        # 0h broadcast TTL → expires_at == send-time → any later drain excludes it.
        self.brain.set_config('self_channel.broadcast_ttl_hours', 0)
        signal.send(self.brain, from_session='X',
                    address=self_contract.ADDR_BROADCAST, body='instant-dead')
        self.assertEqual(signal.drain_inbox(self.brain, 'anyone'), [])

    def test_config_override_broadcast_does_not_touch_directed(self):
        # Shrinking broadcast TTL must not affect the directed default.
        self.brain.set_config('self_channel.broadcast_ttl_hours', 0)
        signal.send(self.brain, from_session='X',
                    address=self_contract.address_for_session('S'), body='still here')
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'S')],
                         ['still here'])

    def test_nonnumeric_ttl_config_falls_back_not_crash(self):
        # A typo'd / non-numeric TTL config must NOT crash send() (get_config
        # returns the raw string when its numeric auto-parse fails) — _resolve_ttl_hours
        # falls back to the documented default and the message is still delivered.
        self.brain.set_config('self_channel.broadcast_ttl_hours', 'not-a-number')
        b = signal.send(self.brain, from_session='X',
                        address=self_contract.ADDR_BROADCAST, body='resilient')
        self.assertTrue(b['expires_at'])   # stamped via the default TTL, no crash
        self.assertEqual([m['body'] for m in signal.drain_inbox(self.brain, 'anyone')],
                         ['resilient'])


class TestLegacyNullExpires(BrainTestBase):
    """A pre-expires_at courier row (the column added to the one existing DB →
    expires_at NULL) is treated as dead: never delivered, and reaped on the next
    sweep. No backfill — the brain was never released, so there's no in-flight
    state worth preserving across the column add, and send() stamps expires_at
    on every new message."""
    needs_embedder = False

    def _insert_legacy(self, mid='m1', to='B', body='legacy'):
        from servers.clock import iso_now
        with self.brain.write_lock:
            self.brain.logs_conn.execute(
                "INSERT INTO self_inflight "
                "(id, from_session, address, body, refs, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (mid, 'A', self_contract.address_for_session(to), body, '', iso_now()))
            self.brain.logs_conn.commit()

    def test_null_expires_not_delivered(self):
        self._insert_legacy()
        self.assertEqual(signal.drain_inbox(self.brain, 'B'), [])
        self.assertEqual(signal.peek_inbox(self.brain, 'B'), [])

    def test_null_expires_reaped_as_dead(self):
        self._insert_legacy()
        self.assertEqual(signal.reap_expired(self.brain), 1)
        remaining = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM self_inflight").fetchone()[0]
        self.assertEqual(remaining, 0)


if __name__ == '__main__':
    unittest.main()
