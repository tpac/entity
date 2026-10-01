"""Phase 1 self-channel presence — the PULL primitives.

Covers brain.present_streams (wall-clock roster), presence.build_presence
(roster + rendered line), and presence.peek (drill into one stream). Pure
reads; no embedder needed.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.clock import iso_cutoff
from servers.channels.self_channel import presence, self_contract, signal


class TestSelfPresence(BrainTestBase):
    needs_embedder = False

    def _save_stream(self, sid, focus='', updated_at=None):
        """Seed a real-turn S0 trace (user_message) for `sid` — the presence
        liveness + focus source (present_streams reads traces, not
        session_state, so autosave can't fake liveness). `focus` becomes the
        trace summary; `updated_at` backdates created_at to place the turn
        earlier in the window."""
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % sid[:8], scale='s0', event_type='K',
            ref_type='user_message', summary=focus, session_id=sid)
        if updated_at is not None:
            self.brain.logs_conn.execute(
                "UPDATE trace_events SET created_at = ? "
                "WHERE session_id = ? AND ref_type = 'user_message'",
                (updated_at, sid))
            self.brain.logs_conn.commit()

    def test_present_streams_excludes_self_and_stale(self):
        self._save_stream('streamAAAA', focus='dashboard fix')
        self._save_stream('streamBBBB', focus='self-channel design')
        # Outside the wall-clock window — awake days ago, not now.
        self._save_stream('streamOLD0', focus='ancient',
                          updated_at=iso_cutoff(days=3))

        rows = self.brain.present_streams(
            exclude_session='streamAAAA', window_min=30, limit=5)
        ids = {r['session_id'] for r in rows}

        self.assertIn('streamBBBB', ids)
        self.assertNotIn('streamAAAA', ids, "caller must be excluded from its own roster")
        self.assertNotIn('streamOLD0', ids, "stale stream outside window must drop")

    def test_build_presence_renders_line_and_one_line_focus(self):
        self._save_stream('streamAAAA', focus='dashboard fix\nSECOND LINE IGNORED')
        self._save_stream('streamBBBB', focus='self-channel presence build')

        # limit=10 so both of mine surface (the test brain has its own
        # session_state row too — present_streams correctly returns all live
        # streams except the caller, so we assert behavior, not an exact count).
        out = presence.build_presence(self.brain, my_session_id='streamZZZZ', limit=10)

        ids = {s['session_id'] for s in out['streams']}
        self.assertIn('streamAAAA', ids)
        self.assertIn('streamBBBB', ids)
        focuses = {s['focus'] for s in out['streams']}
        self.assertIn('dashboard fix', focuses)  # first line only
        self.assertNotIn('dashboard fix\nSECOND LINE IGNORED', focuses)
        # the rendered line reflects exactly the streams returned
        self.assertIn('streams of thought live: %d' % len(out['streams']), out['line'])

    def test_build_presence_rich_carries_worktree(self):
        # rich=True roster must forward the per-session worktree — the "where is
        # each stream working" surface. peek exposes it; the rich projection (which
        # re-copies peek's fields) must too, or the field ships invisible.
        from servers.session_context import SessionContext
        self._save_stream('streamWT00', focus='worktree work')
        ctx = SessionContext(session_id='streamWT00')
        ctx.cwd = '/Users/t/brain/.claude/worktrees/emb-bench'
        ctx.branch = 'emb-bench-eval'
        ctx.worktree = 'emb-bench'
        ctx.save(self.brain._session_state)
        out = presence.build_presence(self.brain, my_session_id='other', limit=10, rich=True)
        entry = next(s for s in out['streams'] if s['session_id'] == 'streamWT00')
        self.assertEqual(entry['worktree'], 'emb-bench')
        self.assertEqual(entry['cwd'], '/Users/t/brain/.claude/worktrees/emb-bench')

    def test_peek_returns_full_focus(self):
        # peek drills into the full session ARC (session_context_for), a
        # separate source from the roster's trace-derived one-line focus.
        self.brain.set_config('session_context_streamAAAA', 'line one\nline two')

        hit = presence.peek(self.brain, 'streamAAAA')
        self.assertTrue(hit['found'])
        self.assertEqual(hit['focus'], 'line one\nline two')  # full arc, not one line

        miss = presence.peek(self.brain, 'no-such-stream')
        self.assertFalse(miss['found'])
        self.assertEqual(miss['focus'], '')

    def test_peek_enrichment_arc_msgs_activity(self):
        # peek now returns arc + recent conversational msgs + activity window +
        # liveness, so a glance shows where a stream is without interrupting it.
        self.brain.set_config('session_context_streamAAAA', 'arc line one\narc line two')
        self._save_stream('streamAAAA', focus='did the dashboard fix')
        p = presence.peek(self.brain, 'streamAAAA')
        self.assertTrue(p['found'])
        self.assertEqual(p['focus'], 'arc line one\narc line two')  # full arc
        # recent_msgs carries the conversational turn, role-tagged
        self.assertTrue(p['recent_msgs'])
        self.assertEqual(p['recent_msgs'][0]['text'], 'did the dashboard fix')
        self.assertEqual(p['recent_msgs'][0]['role'], 'user_message')
        # a just-written turn ⇒ started + last_active set, liveness active
        self.assertTrue(p['session_started_at'])
        self.assertTrue(p['last_active_at'])
        self.assertEqual(p['liveness'], 'active')

    def test_peek_returns_session_env_cwd_branch(self):
        # cwd/branch/worktree live on the session object (fed from the boot hook)
        # and are surfaced in peek so streams can tell where each other is working.
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='streamENV0')
        ctx.cwd = '/Users/ada/brain/.claude/worktrees/foo'
        ctx.branch = 'claude/foo-123'
        ctx.worktree = 'foo'
        ctx.save(self.brain._session_state)
        p = presence.peek(self.brain, 'streamENV0')
        self.assertEqual(p['cwd'], '/Users/ada/brain/.claude/worktrees/foo')
        self.assertEqual(p['branch'], 'claude/foo-123')
        self.assertEqual(p['worktree'], 'foo')
        # absent ⇒ empty strings, never missing keys (degrades, never half-shaped)
        miss = presence.peek(self.brain, 'no-such-stream')
        self.assertEqual(miss['cwd'], '')
        self.assertEqual(miss['branch'], '')
        self.assertEqual(miss['worktree'], '')

    def test_peek_empty_path_has_all_keys(self):
        # Contract: peek's empty/error path (no stream_id → _empty_peek) must
        # carry EVERY key the full path does — cwd/branch/turn_count included —
        # so bracket-access consumers never KeyError. Guards the regression
        # where new fields are added to peek() but not _empty_peek().
        self._save_stream('streamFULL', focus='real turn')
        full = presence.peek(self.brain, 'streamFULL')
        empty = presence.peek(self.brain, '')          # → _empty_peek
        self.assertEqual(set(empty.keys()), set(full.keys()))
        for k in ('cwd', 'branch', 'turn_count'):
            self.assertIn(k, empty)

    def test_peek_found_from_msgs_when_arc_empty(self):
        # fresh stream: no arc encoded yet, but one real turn ⇒ still peeks
        # usefully (the arc lags S1 Scribe; turns are immediate).
        self._save_stream('streamFRSH', focus='just booted, first message')
        p = presence.peek(self.brain, 'streamFRSH')
        self.assertEqual(p['focus'], '')      # no arc yet
        self.assertTrue(p['found'])           # found via recent_msgs
        self.assertEqual(len(p['recent_msgs']), 1)

    def test_peek_msg_cap(self):
        # a long stored summary is capped to PEEK_MSG_MAX in the glance (append
        # stores summary raw, so this truncation is real, not a no-op).
        self._save_stream('streamLONG', focus='x' * 500)
        p = presence.peek(self.brain, 'streamLONG')
        self.assertEqual(len(p['recent_msgs'][0]['text']), self_contract.PEEK_MSG_MAX)

    def test_peek_pending_inbox_count(self):
        # messages waiting for a stream surface as its reachability backlog
        addr = self_contract.address_for_stream('streamRCPT')
        signal.send(self.brain, 'streamSND1', addr, 'msg one')
        signal.send(self.brain, 'streamSND2', addr, 'msg two')
        p = presence.peek(self.brain, 'streamRCPT')
        self.assertEqual(p['pending_inbox_count'], 2)

    def test_stamp_boot_liveness_makes_stream_present(self):
        # a freshly-booted stream is visible in presence BEFORE its first turn
        self.brain.stamp_boot_liveness('bootSTREAM')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        self.assertIn('bootSTREAM', {r['session_id'] for r in rows})

    def test_cap_is_respected(self):
        for i in range(5):
            self._save_stream('stream%05d' % i, focus='focus %d' % i)
        out = presence.build_presence(self.brain, my_session_id='other', limit=2)
        self.assertEqual(len(out['streams']), 2)

    def test_default_cap_from_contract(self):
        for i in range(self_contract.PRESENCE_MAX_STREAMS + 3):
            self._save_stream('strm%05d' % i, focus='f%d' % i)
        out = presence.build_presence(self.brain, my_session_id='other')  # no limit → contract cap
        self.assertEqual(len(out['streams']), self_contract.PRESENCE_MAX_STREAMS)

    def test_liveness_states_and_lost_surfacing(self):
        """active/dormant/lost by recency. Default active_streams=True keeps the
        live set (active+dormant) and filters lost; active_streams=False surfaces
        lost separately (named in the line), not silently dropped at the edge."""
        self._save_stream('streamACTV', focus='working now')                                  # fresh → active
        self._save_stream('streamDORM', focus='quiet', updated_at=iso_cutoff(minutes=15))      # → dormant
        self._save_stream('streamLOST', focus='vanished', updated_at=iso_cutoff(minutes=45))   # → lost (≤60 grace)

        # Default (active_streams=True): live set only; lost filtered out.
        out = presence.build_presence(self.brain, my_session_id='other', limit=10)
        states = {s['session_id']: s['state'] for s in out['streams']}
        self.assertEqual(states.get('streamACTV'), 'active')
        self.assertEqual(states.get('streamDORM'), 'dormant')   # dormant is live, still shown
        self.assertNotIn('streamLOST', states)                  # lost never in the live roster
        self.assertEqual(out['lost'], [])                       # ...and filtered by default

        # active_streams=False: lost surfaced in the grace window, not dropped.
        out2 = presence.build_presence(self.brain, my_session_id='other', limit=10,
                                       active_streams=False)
        lost_ids = {s['session_id'] for s in out2['lost']}
        self.assertIn('streamLOST', lost_ids)                   # surfaced when asked...
        self.assertNotIn('streamLOST', {s['session_id'] for s in out2['streams']})  # ...not in live
        self.assertIn('lost', out2['line'])                     # and named in the line

    def test_resolve_to_canonical_prefix_and_graceful(self):
        """to= resolution: broadcast + full UUID are canonical; the 8-char short
        (an id-prefix) matches the live roster; unknown is loud (no address)."""
        sid = 'aaaaaaaa-1111-2222-3333-444444444444'
        self._save_stream(sid, focus='dal cleanup')
        self.assertEqual(signal.resolve_to(self.brain, 'broadcast')[0], self_contract.ADDR_BROADCAST)
        addr, err = signal.resolve_to(self.brain, sid)            # full id → canonical
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_stream(sid))
        addr, err = signal.resolve_to(self.brain, 'aaaaaaaa')     # 8-char short (prefix) → that stream
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_stream(sid))
        addr, err = signal.resolve_to(self.brain, 'nope')         # unknown → loud
        self.assertIsNone(addr)
        self.assertIn('no live stream', err)

    def test_resolve_to_ambiguous_is_loud(self):
        self._save_stream('bbbbbbbb-0000-0000-0000-000000000001', focus='x')
        self._save_stream('bbbbbbbb-0000-0000-0000-000000000002', focus='y')
        addr, err = signal.resolve_to(self.brain, 'bbbbbbbb')     # id-prefix → 2 matches
        self.assertIsNone(addr)
        self.assertIn('matches', err)

    def test_presence_shows_short_id(self):
        """No self-labeling — the roster shows the 8-char short id (+ focus + state)."""
        sid = 'cccccccc-0000-0000-0000-000000000003'
        self._save_stream(sid, focus='docs')
        out = presence.build_presence(self.brain, my_session_id='other', limit=10)
        self.assertIn('cccccccc', {s['short'] for s in out['streams']})
        self.assertIn('cccccccc', out['line'])


class TestPresenceCountsWatchers(BrainTestBase):
    """B2 (2026-06-04): a /watch listener emits only `heartbeat` turns, yet it
    is the MOST reachable stream — it can be triggered to act. It must count as
    present, and its focus must not be polluted by the `<task-notification>`
    wake envelope."""
    needs_embedder = False

    def _turn(self, sid, ref_type, summary, updated_at=None, event_type='K'):
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % sid[:8], scale='s0', event_type=event_type,
            ref_type=ref_type, summary=summary, session_id=sid)
        if updated_at is not None:
            self.brain.logs_conn.execute(
                "UPDATE trace_events SET created_at = ? "
                "WHERE session_id = ? AND ref_type = ? AND summary = ?",
                (updated_at, sid, ref_type, summary))
            self.brain.logs_conn.commit()

    def test_heartbeat_only_watcher_is_present(self):
        # A pure listener whose only recent turn is a watch tick (heartbeat).
        self._turn('watcherXX', 'heartbeat', 'Quiet and listening.')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        ids = {r['session_id'] for r in rows}
        self.assertIn('watcherXX', ids,
                      "a heartbeat-only watcher is reachable and must show present")

    def test_real_prompt_wins_over_task_notification_in_focus(self):
        # Recent turns: a real prompt (earlier) then a watch ignition (now).
        # Focus must surface the real prompt, not the <task-notification>.
        self._turn('mixedXXXX', 'user_message', 'fix the recall bug',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('mixedXXXX', 'user_message', '<task-notification>\n<event>twin msg')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        focus = {r['session_id']: r['focus'] for r in rows}.get('mixedXXXX')
        self.assertEqual(focus, 'fix the recall bug',
                         "the watch-ignition envelope must not become the focus")

    def test_pure_watcher_focus_is_empty_not_notification(self):
        # No real prompt ever — only a heartbeat + a task-notification ignition.
        # Present (heartbeat), but focus clean-empty rather than the envelope.
        self._turn('pureWtch0', 'heartbeat', 'listening')
        self._turn('pureWtch0', 'user_message', '<task-notification>\n<event>x')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        row = {r['session_id']: r for r in rows}.get('pureWtch0')
        self.assertIsNotNone(row, "heartbeat keeps the watcher present")
        self.assertEqual(row['focus'], '',
                         "a task-notification must not leak into a watcher's focus")

    def test_assistant_message_can_be_focus(self):
        # Focus is the latest CONVERSATIONAL turn — user OR
        # assistant — so a watcher's own last reply can be the focus. Here the
        # assistant turn is newer than the last user prompt, so it wins.
        self._turn('asstFocus', 'user_message', 'old user prompt',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('asstFocus', 'assistant_message', 'shipped the TTL fix',
                   event_type='delta')   # assistant_message is an (s0, delta) trace
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        focus = {r['session_id']: r['focus'] for r in rows}.get('asstFocus')
        self.assertEqual(focus, 'shipped the TTL fix',
                         "latest conversational turn wins, even when it's the assistant")

    def test_wake_reply_bumps_focus_but_not_recency(self):
        # THE ASYMMETRY. A machine-woken turn is the ENVELOPE (a user_message);
        # the reply it provokes is an ordinary assistant_message carrying no
        # marker. For RANKING that reply must not count — letting it count is
        # what floated a background task above the operator's own session. For
        # DESCRIPTION it must count — it is the stream's own account of what it
        # is doing, and the most informative line available about it.
        self._turn('wokenXXX0', 'user_message', 'adjudicate the eval arms',
                   updated_at=iso_cutoff(minutes=20))
        self._turn('wokenXXX0', 'user_message', '<task-notification>\n<event>log tick',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('wokenXXX0', 'assistant_message', 'Still waiting on the control arm.',
                   event_type='delta')   # assistant_message is an (s0, delta) trace
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        row = {r['session_id']: r for r in rows}.get('wokenXXX0')
        self.assertIsNotNone(row, "a woken stream stays PRESENT — only its rank changes")
        self.assertEqual(row['focus'], 'Still waiting on the control arm.',
                         "focus DESCRIBES the stream — its own latest report is the "
                         "most useful line, even when a wake provoked it")
        # conv_recency is the RANKING key, returned by the DAL and not carried
        # through present_streams' 4-key projection — assert it at its own layer.
        dal_row = {r['session_id']: r for r in self.brain._trace_dal
                   .active_sessions_by_turn(iso_cutoff(minutes=30), limit=10)}['wokenXXX0']
        self.assertLess(dal_row['conv_recency'], iso_cutoff(minutes=10),
                        "the reply must not bump conv_recency — that is what "
                        "outranked the operator's own session")
        # `live_recency` still counts the envelope, so the stream keeps
        # reading `active`. DELIBERATE: a watch-mode stream is the most reachable
        # thing there is. Only the ranking key is corrected here.
        self.assertGreater(row['live_recency'], iso_cutoff(minutes=10),
                           "liveness must still reflect reachability")

    def test_peek_keeps_wake_reply_but_never_the_envelope(self):
        # A peek answers "what is this stream doing", so the answer to a wake is
        # exactly what it should return — while the envelope itself stays noise.
        # Applying the ranking predicate here would swap a live status line for a
        # stale one, so session_activity deliberately filters only the envelope.
        self._turn('peekWoke0', 'user_message', 'run the gate-4 corpus',
                   updated_at=iso_cutoff(minutes=20))
        self._turn('peekWoke0', 'user_message', '<task-notification>\n<event>tick',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('peekWoke0', 'assistant_message', 'Item 6 of 10, waiting.',
                   event_type='delta')
        act = self.brain.session_activity('peekWoke0', msg_limit=4)
        texts = [m['text'] for m in act.get('recent_msgs', [])]
        self.assertIn('Item 6 of 10, waiting.', texts,
                      "the stream's own status line is what a peek exists to show")
        self.assertFalse([t for t in texts if t.startswith('<task-notification>')],
                         "the wake envelope itself is never work")

    def test_woken_stream_ranks_below_attended_stream(self):
        # The operator-visible symptom: a Monitor-woken background task sat at
        # the TOP of the roster, above the worktree actually being worked in.
        # Ranking is by conv_recency, so the woken stream's own replies must not
        # count as recency or it wins the roster on machine traffic alone.
        self._turn('attended0', 'user_message', 'real operator work',
                   updated_at=iso_cutoff(minutes=10))
        self._turn('wokenNOW0', 'user_message', 'launch prompt',
                   updated_at=iso_cutoff(minutes=25))
        self._turn('wokenNOW0', 'user_message', '<task-notification>\n<event>tick',
                   updated_at=iso_cutoff(minutes=2))
        self._turn('wokenNOW0', 'assistant_message', 'Waiting.', event_type='delta')
        rows = self.brain.present_streams(exclude_session='other', window_min=30, limit=10)
        order = [r['session_id'] for r in rows]
        self.assertIn('attended0', order)
        self.assertIn('wokenNOW0', order)
        self.assertLess(order.index('attended0'), order.index('wokenNOW0'),
                        "an attended stream must outrank one kept alive by wakes")


if __name__ == '__main__':
    unittest.main()
