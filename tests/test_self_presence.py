"""Phase 1 self-channel presence — the PULL primitives.

Covers brain.present_sessions (wall-clock roster), presence.build_presence
(roster + rendered line), and presence.peek (drill into one session). Pure
reads; no embedder needed.
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.clock import iso_cutoff
from servers.trace_contract import session_short
from servers.channels.self_channel import presence, self_contract, signal


class TestSelfPresence(BrainTestBase):
    needs_embedder = False

    def _save_session(self, sid, focus='', updated_at=None):
        """Seed a real-turn S0 trace (user_message) for `sid` — the presence
        liveness + focus source (present_sessions reads traces, not
        session_state, so autosave can't fake liveness). `focus` becomes the
        trace summary; `updated_at` backdates created_at to place the turn
        earlier in the window."""
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % session_short(sid), scale='s0', event_type='K',
            ref_type='user_message', summary=focus, session_id=sid)
        if updated_at is not None:
            self.brain.logs_conn.execute(
                "UPDATE trace_events SET created_at = ? "
                "WHERE session_id = ? AND ref_type = 'user_message'",
                (updated_at, sid))
            self.brain.logs_conn.commit()

    def test_present_sessions_excludes_self_and_stale(self):
        self._save_session('sessionAAAA', focus='dashboard fix')
        self._save_session('sessionBBBB', focus='self-channel design')
        # Outside the wall-clock window — awake days ago, not now.
        self._save_session('sessionOLD0', focus='ancient',
                          updated_at=iso_cutoff(days=3))

        rows = self.brain.present_sessions(
            exclude_session='sessionAAAA', window_min=30, limit=5)
        ids = {r['session_id'] for r in rows}

        self.assertIn('sessionBBBB', ids)
        self.assertNotIn('sessionAAAA', ids, "caller must be excluded from its own roster")
        self.assertNotIn('sessionOLD0', ids, "stale session outside window must drop")

    def test_build_presence_renders_line_and_one_line_focus(self):
        self._save_session('sessionAAAA', focus='dashboard fix\nSECOND LINE IGNORED')
        self._save_session('sessionBBBB', focus='self-channel presence build')

        # limit=10 so both of mine surface (the test brain has its own
        # session_state row too — present_sessions correctly returns all live
        # sessions except the caller, so we assert behavior, not an exact count).
        out = presence.build_presence(self.brain, my_session_id='sessionZZZZ', limit=10)

        ids = {s['session_id'] for s in out['sessions']}
        self.assertIn('sessionAAAA', ids)
        self.assertIn('sessionBBBB', ids)
        focuses = {s['focus'] for s in out['sessions']}
        self.assertIn('dashboard fix', focuses)  # first line only
        self.assertNotIn('dashboard fix\nSECOND LINE IGNORED', focuses)
        # the rendered line reflects exactly the sessions returned
        self.assertIn('sessions live: %d' % len(out['sessions']), out['line'])

    def test_build_presence_rich_carries_worktree(self):
        # rich=True roster must forward the per-session worktree — the "where is
        # each session working" surface. peek exposes it; the rich projection (which
        # re-copies peek's fields) must too, or the field ships invisible.
        from servers.session_context import SessionContext
        self._save_session('sessionWT00', focus='worktree work')
        ctx = SessionContext(session_id='sessionWT00')
        ctx.cwd = '/Users/t/brain/.claude/worktrees/emb-bench'
        ctx.branch = 'emb-bench-eval'
        ctx.worktree = 'emb-bench'
        ctx.save(self.brain._session_state)
        out = presence.build_presence(self.brain, my_session_id='other', limit=10, rich=True)
        entry = next(s for s in out['sessions'] if s['session_id'] == 'sessionWT00')
        self.assertEqual(entry['worktree'], 'emb-bench')
        self.assertEqual(entry['cwd'], '/Users/t/brain/.claude/worktrees/emb-bench')

    def test_peek_returns_full_focus(self):
        # peek drills into the full session ARC (session_context_for), a
        # separate source from the roster's trace-derived one-line focus.
        self.brain.set_config('session_context_sessionAAAA', 'line one\nline two')

        hit = presence.peek(self.brain, 'sessionAAAA')
        self.assertTrue(hit['found'])
        self.assertEqual(hit['focus'], 'line one\nline two')  # full arc, not one line

        miss = presence.peek(self.brain, 'no-such-session')
        self.assertFalse(miss['found'])
        self.assertEqual(miss['focus'], '')

    def test_peek_enrichment_arc_msgs_activity(self):
        # peek now returns arc + recent conversational msgs + activity window +
        # liveness, so a glance shows where a session is without interrupting it.
        self.brain.set_config('session_context_sessionAAAA', 'arc line one\narc line two')
        self._save_session('sessionAAAA', focus='did the dashboard fix')
        p = presence.peek(self.brain, 'sessionAAAA')
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
        # and are surfaced in peek so sessions can tell where each other is working.
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='sessionENV0')
        ctx.cwd = '/Users/ada/brain/.claude/worktrees/foo'
        ctx.branch = 'claude/foo-123'
        ctx.worktree = 'foo'
        ctx.save(self.brain._session_state)
        p = presence.peek(self.brain, 'sessionENV0')
        self.assertEqual(p['cwd'], '/Users/ada/brain/.claude/worktrees/foo')
        self.assertEqual(p['branch'], 'claude/foo-123')
        self.assertEqual(p['worktree'], 'foo')
        # absent ⇒ empty strings, never missing keys (degrades, never half-shaped)
        miss = presence.peek(self.brain, 'no-such-session')
        self.assertEqual(miss['cwd'], '')
        self.assertEqual(miss['branch'], '')
        self.assertEqual(miss['worktree'], '')

    def test_peek_empty_path_has_all_keys(self):
        # Contract: peek's empty/error path (no session_id → _empty_peek) must
        # carry EVERY key the full path does — cwd/branch/turn_count included —
        # so bracket-access consumers never KeyError. Guards the regression
        # where new fields are added to peek() but not _empty_peek().
        self._save_session('sessionFULL', focus='real turn')
        full = presence.peek(self.brain, 'sessionFULL')
        empty = presence.peek(self.brain, '')          # → _empty_peek
        self.assertEqual(set(empty.keys()), set(full.keys()))
        for k in ('cwd', 'branch', 'turn_count'):
            self.assertIn(k, empty)

    def test_peek_found_from_msgs_when_arc_empty(self):
        # fresh session: no arc encoded yet, but one real turn ⇒ still peeks
        # usefully (the arc lags S1 Scribe; turns are immediate).
        self._save_session('sessionFRSH', focus='just booted, first message')
        p = presence.peek(self.brain, 'sessionFRSH')
        self.assertEqual(p['focus'], '')      # no arc yet
        self.assertTrue(p['found'])           # found via recent_msgs
        self.assertEqual(len(p['recent_msgs']), 1)

    def test_peek_msg_cap(self):
        # A turn is stored as the hooks store it: summary = its 200-char head,
        # metadata.content = the whole text. The glance reads the whole text and
        # cuts it at PEEK_MSG_MAX loudly, naming the get_trace call that reads it.
        text = 'x' * 500
        tid = self.brain._trace_dal.append(
            chain_id='s0-%s-0' % session_short('sessionLONG'), scale='s0',
            event_type='K', ref_type='user_message', summary=text[:200],
            metadata={'content': text}, session_id='sessionLONG')
        shown = presence.peek(self.brain, 'sessionLONG')['recent_msgs'][0]['text']
        self.assertTrue(shown.startswith('x' * self_contract.PEEK_MSG_MAX + ' …'))
        self.assertNotIn('x' * (self_contract.PEEK_MSG_MAX + 1), shown)
        self.assertIn('(+200 chars — get_trace %s for the whole message)' % tid, shown)

    def test_peek_reads_summary_when_no_content(self):
        # older rows carry the text only in summary
        self._save_session('sessionOLDR', focus='summary only')
        p = presence.peek(self.brain, 'sessionOLDR')
        self.assertEqual(p['recent_msgs'][0]['text'], 'summary only')

    def test_peek_short_msg_is_whole(self):
        self._save_session('sessionSHRT', focus='short and whole')
        p = presence.peek(self.brain, 'sessionSHRT')
        self.assertEqual(p['recent_msgs'][0]['text'], 'short and whole')

    def test_peek_barely_over_the_cap_stays_whole(self):
        # a cut that would read longer than the message is not made
        body = 'y' * (self_contract.PEEK_MSG_MAX + 5)
        self._save_session('sessionEDGE', focus=body)
        p = presence.peek(self.brain, 'sessionEDGE')
        self.assertEqual(p['recent_msgs'][0]['text'], body)

    def test_roster_focus_marks_a_cut(self):
        # the roster's one-line focus keeps its full width and carries a bare
        # '…' when the line was cut.
        wide = 'word ' * 60
        self._save_session('sessionWIDE', focus=wide)
        self._save_session('sessionFITS', focus='fits on one line')
        out = presence.build_presence(self.brain, my_session_id='other', rich=False)
        focus = {s['session_id']: s['focus'] for s in out['sessions']}
        self.assertEqual(focus['sessionWIDE'],
                         wide[:self_contract.ROSTER_FOCUS_MAX].rstrip() + ' …')
        self.assertEqual(focus['sessionFITS'], 'fits on one line')

    def test_peek_pending_inbox_count(self):
        # messages waiting for a session surface as its reachability backlog
        addr = self_contract.address_for_session('sessionRCPT')
        signal.send(self.brain, 'sessionSND1', addr, 'msg one')
        signal.send(self.brain, 'sessionSND2', addr, 'msg two')
        p = presence.peek(self.brain, 'sessionRCPT')
        self.assertEqual(p['pending_inbox_count'], 2)

    def test_stamp_boot_liveness_makes_session_present(self):
        # a freshly-booted session is visible in presence BEFORE its first turn
        self.brain.stamp_boot_liveness('bootSTREAM')
        rows = self.brain.present_sessions(exclude_session='other', window_min=30, limit=10)
        self.assertIn('bootSTREAM', {r['session_id'] for r in rows})

    def test_cap_is_respected(self):
        for i in range(5):
            self._save_session('session%05d' % i, focus='focus %d' % i)
        out = presence.build_presence(self.brain, my_session_id='other', limit=2)
        self.assertEqual(len(out['sessions']), 2)

    def test_default_cap_from_contract(self):
        for i in range(self_contract.PRESENCE_MAX_SESSIONS + 3):
            self._save_session('strm%05d' % i, focus='f%d' % i)
        out = presence.build_presence(self.brain, my_session_id='other')  # no limit → contract cap
        self.assertEqual(len(out['sessions']), self_contract.PRESENCE_MAX_SESSIONS)

    def test_liveness_states_and_lost_surfacing(self):
        """active/dormant/lost by recency. Default active_sessions=True keeps the
        live set (active+dormant) and filters lost; active_sessions=False surfaces
        lost separately (named in the line), not silently dropped at the edge."""
        self._save_session('sessionACTV', focus='working now')                                  # fresh → active
        self._save_session('sessionDORM', focus='quiet', updated_at=iso_cutoff(minutes=15))      # → dormant
        self._save_session('sessionLOST', focus='vanished', updated_at=iso_cutoff(minutes=45))   # → lost (≤60 grace)

        # Default (active_sessions=True): live set only; lost filtered out.
        out = presence.build_presence(self.brain, my_session_id='other', limit=10)
        states = {s['session_id']: s['state'] for s in out['sessions']}
        self.assertEqual(states.get('sessionACTV'), 'active')
        self.assertEqual(states.get('sessionDORM'), 'dormant')   # dormant is live, still shown
        self.assertNotIn('sessionLOST', states)                  # lost never in the live roster
        self.assertEqual(out['lost'], [])                       # ...and filtered by default

        # active_sessions=False: lost surfaced in the grace window, not dropped.
        out2 = presence.build_presence(self.brain, my_session_id='other', limit=10,
                                       active_sessions=False)
        lost_ids = {s['session_id'] for s in out2['lost']}
        self.assertIn('sessionLOST', lost_ids)                   # surfaced when asked...
        self.assertNotIn('sessionLOST', {s['session_id'] for s in out2['sessions']})  # ...not in live
        self.assertIn('lost', out2['line'])                     # and named in the line

    def test_resolve_to_canonical_short_and_graceful(self):
        """to= resolution: broadcast + full UUID are canonical; the session short
        matches the live roster; unknown is loud (no address)."""
        sid = 'aaaaaaaa-1111-2222-3333-444444444444'
        self._save_session(sid, focus='dal cleanup')
        self.assertEqual(signal.resolve_to(self.brain, 'broadcast')[0], self_contract.ADDR_BROADCAST)
        addr, err = signal.resolve_to(self.brain, sid)            # full id → canonical
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_session(sid))
        addr, err = signal.resolve_to(self.brain, '444444444444')  # its session short → that session
        self.assertIsNone(err)
        self.assertEqual(addr, self_contract.address_for_session(sid))
        addr, err = signal.resolve_to(self.brain, 'nope')         # unknown → loud
        self.assertIsNone(addr)
        self.assertIn('no live session', err)

    def test_resolve_to_ambiguous_is_loud(self):
        self._save_session('bbbbbbbb-0000-0000-0000-000000000001', focus='x')
        self._save_session('cccccccc-0000-0000-0000-000000000001', focus='y')
        addr, err = signal.resolve_to(self.brain, '000000000001')  # shared short → 2 matches
        self.assertIsNone(addr)
        self.assertIn('matches', err)

    def test_presence_shows_short_id(self):
        """No self-labeling — the roster shows the session short (the hyphen-free
        tail its chains and log lines carry) + focus + state."""
        sid = 'cccccccc-0000-0000-0000-000000000003'
        self._save_session(sid, focus='docs')
        out = presence.build_presence(self.brain, my_session_id='other', limit=10)
        self.assertIn('000000000003', {s['short'] for s in out['sessions']})
        self.assertIn('000000000003', out['line'])

    def test_same_minute_codex_sessions_each_answer_to_their_shown_short(self):
        """Codex ids are UUIDv7: chats opened in the same minute share their first
        8 characters. Each session is shown by its own short — in presence, in a
        delivered message and in its reply hint — and that short addresses that
        session; the shared head is refused, never delivered to either."""
        a = '01a131e3-005d-7e31-bf02-16ee757e4070'
        b = '01a131e3-29b7-76d2-8003-fbf566884e76'
        self._save_session(a, focus='brain')
        self._save_session(b, focus='he')
        out = presence.build_presence(self.brain, my_session_id='other', limit=10)
        shown = {s['session_id']: s['short'] for s in out['sessions']}
        self.assertEqual(shown, {a: '16ee757e4070', b: 'fbf566884e76'})
        for sid, short in shown.items():
            self.assertEqual(signal.resolve_to(self.brain, short),
                             (self_contract.address_for_session(sid), None))
        signal.send(self.brain, from_session=a,
                    address=self_contract.address_for_session(b), body='hi')
        block, _ = signal.drain_and_render(self.brain, b)
        self.assertIn('other session (id:16ee757e4070) says:', block)
        self.assertIn('reply: self_send to="16ee757e4070"', block)
        addr, err = signal.resolve_to(self.brain, '01a131e3')
        self.assertIsNone(addr)
        self.assertIn('no live session matches', err)


class TestPresenceCountsWatchers(BrainTestBase):
    """B2 (2026-06-04): a /watch listener emits only `heartbeat` turns, yet it
    is the MOST reachable session — it can be triggered to act. It must count as
    present, and its focus must not be polluted by the `<task-notification>`
    wake envelope."""
    needs_embedder = False

    def _turn(self, sid, ref_type, summary, updated_at=None, event_type='K'):
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % session_short(sid), scale='s0', event_type=event_type,
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
        rows = self.brain.present_sessions(exclude_session='other', window_min=30, limit=10)
        ids = {r['session_id'] for r in rows}
        self.assertIn('watcherXX', ids,
                      "a heartbeat-only watcher is reachable and must show present")

    def test_real_prompt_wins_over_task_notification_in_focus(self):
        # Recent turns: a real prompt (earlier) then a watch ignition (now).
        # Focus must surface the real prompt, not the <task-notification>.
        self._turn('mixedXXXX', 'user_message', 'fix the recall bug',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('mixedXXXX', 'user_message', '<task-notification>\n<event>twin msg')
        rows = self.brain.present_sessions(exclude_session='other', window_min=30, limit=10)
        focus = {r['session_id']: r['focus'] for r in rows}.get('mixedXXXX')
        self.assertEqual(focus, 'fix the recall bug',
                         "the watch-ignition envelope must not become the focus")

    def test_pure_watcher_focus_is_empty_not_notification(self):
        # No real prompt ever — only a heartbeat + a task-notification ignition.
        # Present (heartbeat), but focus clean-empty rather than the envelope.
        self._turn('pureWtch0', 'heartbeat', 'listening')
        self._turn('pureWtch0', 'user_message', '<task-notification>\n<event>x')
        rows = self.brain.present_sessions(exclude_session='other', window_min=30, limit=10)
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
        rows = self.brain.present_sessions(exclude_session='other', window_min=30, limit=10)
        focus = {r['session_id']: r['focus'] for r in rows}.get('asstFocus')
        self.assertEqual(focus, 'shipped the TTL fix',
                         "latest conversational turn wins, even when it's the assistant")

    def test_wake_reply_bumps_focus_but_not_recency(self):
        # THE ASYMMETRY. A machine-woken turn is the ENVELOPE (a user_message);
        # the reply it provokes is an ordinary assistant_message carrying no
        # marker. For RANKING that reply must not count — letting it count is
        # what floated a background task above the operator's own session. For
        # DESCRIPTION it must count — it is the session's own account of what it
        # is doing, and the most informative line available about it.
        self._turn('wokenXXX0', 'user_message', 'adjudicate the eval arms',
                   updated_at=iso_cutoff(minutes=20))
        self._turn('wokenXXX0', 'user_message', '<task-notification>\n<event>log tick',
                   updated_at=iso_cutoff(minutes=5))
        self._turn('wokenXXX0', 'assistant_message', 'Still waiting on the control arm.',
                   event_type='delta')   # assistant_message is an (s0, delta) trace
        rows = self.brain.present_sessions(exclude_session='other', window_min=30, limit=10)
        row = {r['session_id']: r for r in rows}.get('wokenXXX0')
        self.assertIsNotNone(row, "a woken session stays PRESENT — only its rank changes")
        self.assertEqual(row['focus'], 'Still waiting on the control arm.',
                         "focus DESCRIBES the session — its own latest report is the "
                         "most useful line, even when a wake provoked it")
        # conv_recency is the RANKING key, returned by the DAL and not carried
        # through present_sessions' 4-key projection — assert it at its own layer.
        dal_row = {r['session_id']: r for r in self.brain._trace_dal
                   .active_sessions_by_turn(iso_cutoff(minutes=30), limit=10)}['wokenXXX0']
        self.assertLess(dal_row['conv_recency'], iso_cutoff(minutes=10),
                        "the reply must not bump conv_recency — that is what "
                        "outranked the operator's own session")
        # `live_recency` still counts the envelope, so the session keeps
        # reading `active`. DELIBERATE: a watch-mode session is the most reachable
        # thing there is. Only the ranking key is corrected here.
        self.assertGreater(row['live_recency'], iso_cutoff(minutes=10),
                           "liveness must still reflect reachability")

    def test_peek_keeps_wake_reply_but_never_the_envelope(self):
        # A peek answers "what is this session doing", so the answer to a wake is
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
                      "the session's own status line is what a peek exists to show")
        self.assertFalse([t for t in texts if t.startswith('<task-notification>')],
                         "the wake envelope itself is never work")

    def test_woken_session_ranks_below_attended_session(self):
        # The operator-visible symptom: a Monitor-woken background task sat at
        # the TOP of the roster, above the worktree actually being worked in.
        # Ranking is by conv_recency, so the woken session's own replies must not
        # count as recency or it wins the roster on machine traffic alone.
        self._turn('attended0', 'user_message', 'real operator work',
                   updated_at=iso_cutoff(minutes=10))
        self._turn('wokenNOW0', 'user_message', 'launch prompt',
                   updated_at=iso_cutoff(minutes=25))
        self._turn('wokenNOW0', 'user_message', '<task-notification>\n<event>tick',
                   updated_at=iso_cutoff(minutes=2))
        self._turn('wokenNOW0', 'assistant_message', 'Waiting.', event_type='delta')
        rows = self.brain.present_sessions(exclude_session='other', window_min=30, limit=10)
        order = [r['session_id'] for r in rows]
        self.assertIn('attended0', order)
        self.assertIn('wokenNOW0', order)
        self.assertLess(order.index('attended0'), order.index('wokenNOW0'),
                        "an attended session must outrank one kept alive by wakes")

    def test_replies_to_new_harness_and_subagent_rows_do_not_count_as_attended(self):
        # The prompt reader writes a notice as env_message and a report as
        # subagent_message, each opening its own chain — no user_message to
        # carry the marker. Replies to them must still not bump the ranking.
        def turn(sid, chain, rt, summary, minutes, event_type='K'):
            self.brain._trace_dal.append(chain_id='s0-%s-%d' % (session_short(sid), chain), scale='s0',
                                         event_type=event_type, ref_type=rt,
                                         summary=summary, session_id=sid)
            self.brain.logs_conn.execute(
                "UPDATE trace_events SET created_at = ? WHERE session_id = ? AND summary = ?",
                (iso_cutoff(minutes=minutes), sid, summary))
            self.brain.logs_conn.commit()
        for sid, rt in (('noticeXX0', 'env_message'), ('reportXX0', 'subagent_message')):
            turn(sid, 0, 'user_message', 'launch prompt ' + sid, 25)
            turn(sid, 1, rt, 'machine row ' + sid, 3)
            turn(sid, 1, 'assistant_message', 'reply ' + sid, 2, event_type='delta')
        # A reminder cut from the operator's prompt rides the operator's chain.
        turn('cutXXXXX0', 0, 'user_message', 'real work with a reminder', 4)
        turn('cutXXXXX0', 0, 'env_message', 'worktree reminder', 4)
        turn('cutXXXXX0', 0, 'assistant_message', 'reply to real work', 3, event_type='delta')
        rows = {r['session_id']: r for r in self.brain._trace_dal
                .active_sessions_by_turn(iso_cutoff(minutes=30), limit=10)}
        for sid in ('noticeXX0', 'reportXX0'):
            self.assertLess(rows[sid]['conv_recency'], iso_cutoff(minutes=10), sid)
        self.assertGreater(rows['cutXXXXX0']['conv_recency'], iso_cutoff(minutes=5))


if __name__ == '__main__':
    unittest.main()
