"""The S0 session stamp — which model produced a turn, on which host.

Contract: trace_contract.S0_SESSION_STAMP_FIELDS ('model', 'host'). The value
is PER SESSION and per turn, so it rides on the SessionContext and is stamped by
the S0 write door (brain_traces.stamp_s0_session), never by the process-wide
TraceDAL identity stamp — one daemon serves streams on different models.

Held in step here:
  • SessionContext carries + persists every field; set_env refreshes on truthy
  • _s0_trace stamps every field from ctx (setdefault — explicit wins; unknown
    stays absent rather than blank)
  • hook_recall / post_response_common feed the stamp from hook args; the
    session mirrors the LATEST value while each row keeps its own
  • the dispatched trace_append door stamps S0 rows from the session env
  • a Stop without a model is loud (errors table), never blocking
  • a dialogue row (user/assistant) written with no text is loud, never blocking
  • presence peek / _empty_peek / session_env_for expose the fields
  • the dashboard's literal mirror of the field list matches the contract and
    query_traces / query_recent_sessions promote every field top-level
  • hook_common.turn_model / host_tells read the value off each host's source
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
_HOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')
sys.path.insert(0, _HOOKS_DIR)

from tests.brain_test_base import BrainTestBase
from servers.trace_contract import session_short
from servers.trace_contract import S0_SESSION_STAMP_FIELDS, READER_STAMP_FIELDS
from servers.session_context import SessionContext
from servers.brain_traces import _s0_trace, stamp_s0_session
from servers.daemon_hooks import (
    hook_recall, post_response_common, hook_post_response_track)
from servers.dispatch_observability import _handle_trace_append
from servers.channels.self_channel import presence


def _s0_meta(brain, session_id, ref_type):
    rows = brain.logs_conn.execute(
        "SELECT metadata FROM trace_events WHERE scale='s0' AND session_id=? "
        "AND ref_type=? ORDER BY created_at", (session_id, ref_type)).fetchall()
    return [json.loads(r[0]) if r[0] else None for r in rows]


class StampHelperTest(unittest.TestCase):
    """stamp_s0_session — the one merge, for ctx objects and env dicts alike."""

    def test_contract_fields(self):
        self.assertEqual(S0_SESSION_STAMP_FIELDS, ('model', 'host'))

    def test_unknown_stamps_nothing(self):
        ctx = SessionContext(session_id='s')
        self.assertIsNone(stamp_s0_session(None, vars(ctx)))
        self.assertEqual(stamp_s0_session({'a': 1}, vars(ctx)), {'a': 1})
        self.assertIsNone(stamp_s0_session(None, {'model': '', 'host': ''}))

    def test_stamps_from_ctx_and_from_env_dict(self):
        ctx = SessionContext(session_id='s')
        ctx.set_env(model='claude-fable-5-1', host='claude-code')
        self.assertEqual(stamp_s0_session(None, vars(ctx)),
                         {'model': 'claude-fable-5-1', 'host': 'claude-code'})
        env = {'model': 'gpt-6-astra', 'host': 'codex', 'cwd': '/x'}
        self.assertEqual(stamp_s0_session({'tool': 'Bash'}, env),
                         {'tool': 'Bash', 'model': 'gpt-6-astra', 'host': 'codex'})

    def test_explicit_per_event_value_wins_and_partial_stamps_partial(self):
        ctx = SessionContext(session_id='s')
        ctx.set_env(model='m1')                      # host unknown
        out = stamp_s0_session({'model': 'explicit'}, vars(ctx))
        self.assertEqual(out, {'model': 'explicit'})  # no blank host key
        self.assertNotIn('host', out)

    def test_non_dict_metadata_passes_through_untouched(self):
        # A wire payload that decodes to a list/scalar is the DAL's to warn
        # about — the stamp must never raise on it (mirrors _stamp_identity).
        env = {'model': 'm', 'host': 'codex'}
        self.assertEqual(stamp_s0_session(['a', 'b'], env), ['a', 'b'])
        self.assertEqual(stamp_s0_session('note', env), 'note')


class SessionContextStampTest(unittest.TestCase):
    def test_every_field_is_an_attribute_and_set_env_refreshes_on_truthy(self):
        ctx = SessionContext(session_id='s')
        for f in S0_SESSION_STAMP_FIELDS:
            self.assertEqual(getattr(ctx, f), '')
        ctx.set_env(model='m1', host='claude-code')
        self.assertEqual((ctx.model, ctx.host), ('m1', 'claude-code'))
        ctx.set_env(model='', host='')               # empty → keep
        self.assertEqual((ctx.model, ctx.host), ('m1', 'claude-code'))
        ctx.set_env(model='m2')                      # latest wins
        self.assertEqual((ctx.model, ctx.host), ('m2', 'claude-code'))

    def test_save_load_roundtrip(self):
        from servers.schema import ensure_logs_schema
        from servers.dal_logs import SessionStateDAL
        conn = sqlite3.connect(':memory:')
        ensure_logs_schema(conn)
        dal = SessionStateDAL(conn)
        ctx = SessionContext(session_id='rt')
        ctx.set_env(model='gpt-6-astra', host='codex')
        ctx.save(dal)
        loaded = SessionContext.load(dal, 'rt')
        for f in S0_SESSION_STAMP_FIELDS:
            self.assertEqual(getattr(loaded, f), getattr(ctx, f), f)
        conn.close()


class S0WriteDoorTest(BrainTestBase):
    """The stamp lands on every S0 row the hooks write, per turn."""

    needs_embedder = False

    def test_s0_trace_stamps_from_ctx(self):
        ctx = self.brain.get_or_create_session('stamp-door')
        ctx.set_env(model='claude-fable-5-1', host='claude-code')
        _s0_trace(self.brain, ctx, event_type='K', ref_type='heartbeat', summary='x')
        meta = _s0_meta(self.brain, 'stamp-door', 'heartbeat')[-1]
        for f in S0_SESSION_STAMP_FIELDS:
            self.assertEqual(meta[f], getattr(ctx, f), f)

    def test_s0_trace_without_stamp_writes_no_keys(self):
        ctx = self.brain.get_or_create_session('stamp-none')
        _s0_trace(self.brain, ctx, event_type='K', ref_type='heartbeat', summary='x')
        meta = _s0_meta(self.brain, 'stamp-none', 'heartbeat')[-1]
        for f in S0_SESSION_STAMP_FIELDS:
            self.assertNotIn(f, meta or {})

    def test_hook_recall_stamps_user_message_and_session_mirrors_latest(self):
        sid = 'stamp-recall'
        hook_recall(self.brain, {"prompt": "yes", "session_id": sid,
                                 "register_only": True,
                                 "model": "claude-fable-5-1", "tells": ["CLAUDE_CODE_SESSION_ID"]}, [])
        ctx = self.brain.get_or_create_session(sid)
        ctx.increment_stop()
        hook_recall(self.brain, {"prompt": "ok", "session_id": sid,
                                 "register_only": True,
                                 "model": "claude-opus-5", "tells": ["CLAUDE_CODE_SESSION_ID"]}, [])
        rows = _s0_meta(self.brain, sid, 'user_message')
        self.assertEqual([r['model'] for r in rows],
                         ['claude-fable-5-1', 'claude-opus-5'])   # per-turn truth
        self.assertEqual(self.brain.session_env_for(sid)['model'],
                         'claude-opus-5')                          # latest mirrored
        self.assertEqual(self.brain.session_env_for(sid)['host'], 'claude-code')

    def test_prompt_rows_carry_the_hosts_turn(self):
        # Every row of one prompt carries the host's name for its turn. A
        # message absorbed mid-turn shares it with the prompt that opened the
        # turn: the conversation reader hands it to the turn rule, and the
        # encode cadence counts the two as one turn. A client that sends no
        # turn writes no key, and its row counts on its own.
        from servers.trace_contract import HOST_TURN_FIELD
        sid = 'host-turn'
        base = {"session_id": sid, "register_only": True, "model": "m",
                "tells": ["CLAUDE_CODE_SESSION_ID"]}
        hook_recall(self.brain, dict(base, prompt="look into the grouping", pieces=[
            ['operator', 'look into the grouping', ''],
            ['harness', 'You are in a worktree.', '<system-reminder>']],
            **{HOST_TURN_FIELD: 'p1'}), [])
        hook_recall(self.brain, dict(base, prompt="also the Surface window",
                                     **{HOST_TURN_FIELD: 'p1'}), [])
        hook_recall(self.brain, dict(base, prompt="an old client"), [])
        self.assertEqual([m.get(HOST_TURN_FIELD) for m in _s0_meta(self.brain, sid, 'user_message')],
                         ['p1', 'p1', None])
        self.assertEqual([m.get(HOST_TURN_FIELD) for m in _s0_meta(self.brain, sid, 'env_message')],
                         ['p1'])
        conv = self.brain.get_conversation(sid, with_judge_output=False)
        self.assertEqual([(t['ref_type'], t[HOST_TURN_FIELD]) for t in conv],
                         [('user_message', 'p1'), ('env_message', 'p1'),
                          ('user_message', 'p1'), ('user_message', '')])
        self.assertEqual(self.brain._trace_dal.conversational_turns_since(sid), 2)
        # After a Stop the host may still name the same turn (a Stop-hook
        # continuation); the Stop closed it, so a row on the next chain is a
        # turn of its own, as the turn rule groups it.
        self.brain.get_or_create_session(sid).increment_stop()
        hook_recall(self.brain, dict(base, prompt="and the backfill too",
                                     **{HOST_TURN_FIELD: 'p1'}), [])
        self.assertEqual(self.brain._trace_dal.conversational_turns_since(sid), 3)

    def test_a_malformed_host_turn_is_loud_and_not_stamped(self):
        from servers.trace_contract import HOST_TURN_FIELD
        sid = 'host-turn-bad'
        with mock.patch.object(self.brain, '_log_warning') as warn:
            hook_recall(self.brain, {"session_id": sid, "register_only": True,
                                     "prompt": "look into it", "model": "m",
                                     "tells": ["CLAUDE_CODE_SESSION_ID"],
                                     HOST_TURN_FIELD: 7}, [])
        self.assertIn('host_turn_malformed', [c.args[0] for c in warn.call_args_list])
        self.assertNotIn(HOST_TURN_FIELD, _s0_meta(self.brain, sid, 'user_message')[0])

    def test_post_response_common_stamps_assistant_message(self):
        sid = 'stamp-stop'
        ctx = self.brain.get_or_create_session(sid)
        ctx.last_recall_stop = ctx.stop_counter
        post_response_common(self.brain, sid, "a response",
                             model='gpt-6-astra', tells=['PLUGIN_DATA'])
        meta = _s0_meta(self.brain, sid, 'assistant_message')[-1]
        self.assertEqual((meta['model'], meta['host']), ('gpt-6-astra', 'codex'))

    def test_stop_carries_the_reply_readers_stamp(self):
        # The contract identity that read the reply rides the row, as on
        # prompt rows; a client that predates the reader sends none.
        for sid, stamp in (('reader-stamped', {'vocab_version': 2, 'impl_identity': 'abc123'}),
                           ('reader-old-client', {})):
            ctx = self.brain.get_or_create_session(sid)
            ctx.last_recall_stop = ctx.stop_counter
            hook_post_response_track(self.brain, dict({
                "session_id": sid, "hook_event_name": "Stop", "model": "m",
                "last_assistant_message": "a reply"}, **stamp), [])
            meta = _s0_meta(self.brain, sid, 'assistant_message')[-1]
            self.assertEqual(meta['content'], 'a reply')
            self.assertEqual({k: meta[k] for k in READER_STAMP_FIELDS if k in meta}, stamp)

    def test_eval_harness_call_without_stamp_keeps_known_value(self):
        sid = 'stamp-keep'
        ctx = self.brain.get_or_create_session(sid)
        ctx.set_env(model='m-known', host='claude-code')
        ctx.last_recall_stop = ctx.stop_counter
        post_response_common(self.brain, sid, "a response")
        meta = _s0_meta(self.brain, sid, 'assistant_message')[-1]
        self.assertEqual(meta['model'], 'm-known')

    def test_trace_append_door_stamps_s0_rows_from_session_env(self):
        sid = 'stamp-tool'
        ctx = self.brain.get_or_create_session(sid)
        ctx.set_env(model='claude-fable-5-1', host='claude-code')
        r = _handle_trace_append(self.brain, {
            "chain_id": ctx.s0_chain(), "scale": "s0", "event_type": "delta",
            "ref_type": "tool_result", "summary": "Bash: ls",
            "metadata": json.dumps({"tool": "Bash"}), "session_id": sid}, [])
        self.assertTrue(r.get('ok'), r)
        meta = _s0_meta(self.brain, sid, 'tool_result')[-1]
        self.assertEqual(meta['tool'], 'Bash')
        self.assertEqual(meta['model'], 'claude-fable-5-1')

    def test_trace_append_door_leaves_non_s0_alone(self):
        sid = 'stamp-s1'
        ctx = self.brain.get_or_create_session(sid)
        ctx.set_env(model='claude-fable-5-1', host='claude-code')
        r = _handle_trace_append(self.brain, {
            "chain_id": ctx.s1r_chain(), "scale": "s1", "event_type": "O",
            "ref_type": "recall", "summary": "x", "session_id": sid}, [])
        self.assertTrue(r.get('ok'), r)
        row = self.brain.logs_conn.execute(
            "SELECT metadata FROM trace_events WHERE scale='s1' AND session_id=?",
            (sid,)).fetchone()
        meta = json.loads(row[0]) if row[0] else {}
        self.assertNotIn('model', meta)

    def _error_rows(self, source):
        return self.brain.logs_conn.execute(
            "SELECT metadata FROM debug_log WHERE event_type='error' AND source=?",
            (source,)).fetchall()

    def test_stop_without_model_is_loud_not_blocking(self):
        sid = 'stamp-loud'
        before = len(self._error_rows('s0_model_unset'))
        out = hook_post_response_track(self.brain, {
            "session_id": sid, "hook_event_name": "Stop",
            "last_assistant_message": "r", "host": "claude-code"}, [])
        self.assertIn('output', out)                          # turn still recorded
        self.assertEqual(len(self._error_rows('s0_model_unset')), before + 1)
        # with a model: silent
        hook_post_response_track(self.brain, {
            "session_id": sid, "hook_event_name": "Stop",
            "last_assistant_message": "r", "model": "m", "host": "claude-code"}, [])
        self.assertEqual(len(self._error_rows('s0_model_unset')), before + 1)

    def test_loud_check_is_per_session_not_deduped_across_streams(self):
        # The error dedup fingerprint includes the message — naming the session
        # keeps one stream's gap from masking another's. Codex ids opened in
        # the same minute share their first 8 chars (UUIDv7 timestamp), so the
        # message must carry the session's tail (session_short).
        before = len(self._error_rows('s0_model_unset'))
        sids = ('01a122b6-9059-7ca1-bb1b-a5e6ff4ce0cf',
                '01a122b6-f3c2-7d10-8e4a-17c09b2d5e61')
        for sid in sids:
            hook_post_response_track(self.brain, {
                "session_id": sid, "hook_event_name": "Stop",
                "last_assistant_message": "r"}, [])
        self.assertEqual(len(self._error_rows('s0_model_unset')), before + 2)
        # the SAME stream again inside the dedup window is collapsed
        hook_post_response_track(self.brain, {
            "session_id": sids[0], "hook_event_name": "Stop",
            "last_assistant_message": "r"}, [])
        self.assertEqual(len(self._error_rows('s0_model_unset')), before + 2)

    def test_empty_dialogue_row_is_loud_not_blocking(self):
        # A turn ending on a tool call hands Stop an empty
        # last_assistant_message; the row still lands, and the gap is logged.
        before = len(self._error_rows('s0_dialogue_empty'))
        for i, (event_type, ref_type) in enumerate(
                (('delta', 'assistant_message'), ('K', 'user_message'))):
            ctx = self.brain.get_or_create_session('dialogue-empty-%d' % i)
            tid = _s0_trace(self.brain, ctx, event_type=event_type,
                            ref_type=ref_type, summary='', content='  ')
            self.assertTrue(tid)
        self.assertEqual(len(self._error_rows('s0_dialogue_empty')), before + 2)
        # text, or a row off the dialogue axis: silent
        ctx = self.brain.get_or_create_session('dialogue-full')
        _s0_trace(self.brain, ctx, event_type='delta',
                  ref_type='assistant_message', summary='r', content='r')
        _s0_trace(self.brain, ctx, event_type='K', ref_type='heartbeat',
                  summary='wakeup re-arm')
        self.assertEqual(len(self._error_rows('s0_dialogue_empty')), before + 2)


class PresenceStampTest(BrainTestBase):
    needs_embedder = False

    def test_peek_and_empty_peek_carry_every_field(self):
        ctx = SessionContext(session_id='sessionMDL0')
        ctx.set_env(model='gpt-6-astra', host='codex')
        ctx.save(self.brain._session_state)
        p = presence.peek(self.brain, 'sessionMDL0')
        self.assertEqual((p['model'], p['host']), ('gpt-6-astra', 'codex'))
        empty = presence.peek(self.brain, '')
        for f in S0_SESSION_STAMP_FIELDS:
            self.assertEqual(empty[f], '')
        for f in S0_SESSION_STAMP_FIELDS:
            self.assertIn(f, self.brain.session_env_for('no-such'))

    def test_rich_presence_forwards_model(self):
        sid = 'sessionMDL1'
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % session_short(sid), scale='s0', event_type='K',
            ref_type='user_message', summary='model work', session_id=sid)
        ctx = SessionContext(session_id=sid)
        ctx.set_env(model='claude-fable-5-1', host='claude-code')
        ctx.save(self.brain._session_state)
        out = presence.build_presence(self.brain, my_session_id='other', limit=10, rich=True)
        entry = next(s for s in out['sessions'] if s['session_id'] == sid)
        self.assertEqual((entry['model'], entry['host']), ('claude-fable-5-1', 'claude-code'))


class DashboardMirrorTest(unittest.TestCase):
    """The dashboard may not import servers.* — its literal field list and its
    reads are held to the contract here."""

    def test_dashboard_field_list_matches_contract(self):
        from dashboard.queries import _meta
        self.assertEqual(tuple(_meta.S0_SESSION_STAMP_FIELDS), S0_SESSION_STAMP_FIELDS)

    def _logs_conn(self):
        from servers.schema import ensure_logs_schema
        conn = sqlite3.connect(':memory:')
        ensure_logs_schema(conn)
        return conn

    def test_query_traces_promotes_every_field(self):
        from dashboard.queries import traces as dq
        from servers.dal_logs import TraceDAL
        from servers.clock import iso_now
        conn = self._logs_conn()
        TraceDAL(conn).append(
            chain_id='s0-abcdefgh-3', scale='s0', event_type='delta',
            ref_type='assistant_message', summary='r', session_id='abcdefgh-x',
            metadata={'model': 'claude-fable-5-1', 'host': 'claude-code'})
        TraceDAL(conn).append(
            chain_id='s0-abcdefgh-4', scale='s0', event_type='K',
            ref_type='heartbeat', summary='h', session_id='abcdefgh-x')
        rows = dq.query_traces.__wrapped__(conn, hours=1)
        by_ref = {r['ref_type']: r for r in rows}
        self.assertEqual(by_ref['assistant_message']['model'], 'claude-fable-5-1')
        self.assertEqual(by_ref['assistant_message']['host'], 'claude-code')
        for f in S0_SESSION_STAMP_FIELDS:
            self.assertEqual(by_ref['heartbeat'][f], '')   # present, empty
        conn.close()

    def test_query_recall_log_joins_turn_stamp_assistant_wins(self):
        from dashboard.queries import recalls as rq
        from servers.dal_logs import TraceDAL
        conn = self._logs_conn()
        dal = TraceDAL(conn)
        dal.append(chain_id='s1r-abcdefgh-3', scale='s1', event_type='O',
                   ref_type='recall', summary='5 candidates for: q',
                   session_id='abcdefgh-x')
        dal.append(chain_id='s0-abcdefgh-3', scale='s0', event_type='K',
                   ref_type='user_message', summary='q', session_id='abcdefgh-x',
                   metadata={'model': 'previous-turn-model', 'host': 'claude-code'})
        dal.append(chain_id='s0-abcdefgh-3', scale='s0', event_type='delta',
                   ref_type='assistant_message', summary='r', session_id='abcdefgh-x',
                   metadata={'model': 'this-turn-model', 'host': 'claude-code'})
        rows = rq.query_recall_log.__wrapped__(conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['model'], 'this-turn-model')
        self.assertEqual(rows[0]['host'], 'claude-code')
        conn.close()

    def test_query_recent_sessions_exposes_latest_model(self):
        from dashboard.queries import sessions as sq
        from servers.dal_logs import TraceDAL, SessionStateDAL
        conn = self._logs_conn()
        TraceDAL(conn).append(chain_id='s0-abcdefgh-0', scale='s0', event_type='K',
                              ref_type='user_message', summary='q', session_id='abcdefgh-x')
        ctx = SessionContext(session_id='abcdefgh-x')
        ctx.set_env(model='gpt-6-astra', host='codex')
        ctx.save(SessionStateDAL(conn))
        rows = sq.query_recent_sessions.__wrapped__(conn)
        self.assertEqual(rows[0]['model'], 'gpt-6-astra')
        self.assertEqual(rows[0]['host'], 'codex')
        conn.close()


class HookSourceTest(unittest.TestCase):
    """hook_common.turn_model / host_tells — where each host exposes the value."""

    def setUp(self):
        import hook_common
        self.hc = hook_common

    def test_host_tells_from_env(self):
        with mock.patch.dict(os.environ, {'CLAUDE_CODE_SESSION_ID': 'x'}, clear=True):
            self.assertEqual(self.hc.host_tells(), ['CLAUDE_CODE_SESSION_ID'])
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.hc.host_tells(), [])
            os.environ['PLUGIN_ROOT'] = '/p'  # our own shim is never a tell
            self.assertEqual(self.hc.host_tells(), [])
            os.environ['PLUGIN_DATA'] = '/d'
            self.assertEqual(self.hc.host_tells(), ['PLUGIN_DATA'])

    def test_codex_payload_model_wins(self):
        self.assertEqual(self.hc.turn_model({'model': ' gpt-6-astra '}), 'gpt-6-astra')

    def _transcript(self, entries):
        f = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
        for e in entries:
            f.write(json.dumps(e) + '\n')
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_claude_code_reads_last_main_thread_assistant_entry(self):
        path = self._transcript([
            {'type': 'assistant', 'message': {'model': 'claude-fable-5-1'}},
            {'type': 'user', 'message': {'content': 'hi'}},
            {'type': 'assistant', 'message': {'model': 'claude-opus-5'},
             'isSidechain': True},                       # a subagent — skipped
            {'type': 'progress', 'data': {}},
        ])
        self.assertEqual(self.hc.turn_model({'transcript_path': path}), 'claude-fable-5-1')

    def test_claude_code_no_assistant_yet_or_missing_file(self):
        path = self._transcript([{'type': 'user', 'message': {'content': 'first prompt'}}])
        self.assertEqual(self.hc.turn_model({'transcript_path': path}), '')
        self.assertEqual(self.hc.turn_model({'transcript_path': '/nope/none.jsonl'}), '')
        self.assertEqual(self.hc.turn_model({}), '')

    def test_tail_window_finds_model_in_a_large_transcript(self):
        big = [{'type': 'user', 'message': {'content': 'x' * 2000}} for _ in range(400)]
        entries = [{'type': 'assistant', 'message': {'model': 'old-model'}}] + big \
            + [{'type': 'assistant', 'message': {'model': 'claude-fable-5-1'}}] \
            + [{'type': 'user', 'message': {'content': 'y' * 2000}} for _ in range(5)]
        path = self._transcript(entries)
        self.assertGreater(os.path.getsize(path), self.hc._TRANSCRIPT_TAIL_BYTES)
        self.assertEqual(self.hc.turn_model({'transcript_path': path}), 'claude-fable-5-1')

    def test_tail_without_assistant_falls_back_to_full_scan(self):
        # One oversized tool result (or a subagent's sidechain) after the last
        # main-thread assistant entry pushes it out of the tail window; the
        # full scan still finds it instead of reporting an unstamped turn.
        entries = [{'type': 'assistant', 'message': {'model': 'claude-fable-5-1'}},
                   {'type': 'user', 'message': {'content': 'x' * (self.hc._TRANSCRIPT_TAIL_BYTES + 5000)}},
                   {'type': 'assistant', 'message': {'model': 'claude-opus-5'}, 'isSidechain': True}]
        path = self._transcript(entries)
        self.assertEqual(self.hc.turn_model({'transcript_path': path}), 'claude-fable-5-1')

    def _box(self, recap):
        return {'type': 'tool_use', 'name': 'mcp__ccd_turn__wrap_up', 'input': {'recap': recap}}

    def test_closing_message_is_every_block_of_the_last_main_thread_message(self):
        # The host writes one entry per content block; the closing message is
        # the run of main-thread entries sharing the last message.id.
        path = self._transcript([
            {'type': 'assistant', 'message': {'id': 'm1', 'model': 'claude-opus-5-5',
                                              'content': [self._box('an earlier turn')]}},
            {'type': 'user', 'message': {'content': 'next'}},
            {'type': 'assistant', 'message': {'id': 'm2', 'model': 'claude-opus-5-5',
                                              'content': [{'type': 'thinking', 'thinking': 'x'}]}},
            {'type': 'assistant', 'message': {'id': 'm2', 'model': 'claude-opus-5-5', 'content': [
                {'type': 'tool_use', 'name': 'mcp__ccd_turn__update_plan', 'input': {'advance': True}}]}},
            {'type': 'assistant', 'message': {'id': 'm2', 'model': 'claude-opus-5-5',
                                              'content': [self._box('this turn')]}},
            {'type': 'assistant', 'message': {'id': 's1', 'model': 'claude-haiku-5-5',
                                              'content': [self._box('a subagent')]},
             'isSidechain': True},
            {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'content': 'ok'}]}},
        ])
        self.assertEqual(self.hc.closing_message({'transcript_path': path}), {
            'model': 'claude-opus-5-5',
            'tool_calls': [('mcp__ccd_turn__update_plan', {'advance': True}),
                           ('mcp__ccd_turn__wrap_up', {'recap': 'this turn'})]})

    def test_text_after_the_box_closes_the_turn_without_it(self):
        path = self._transcript([
            {'type': 'assistant', 'message': {'id': 'm1', 'content': [self._box('box')]}},
            {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'content': 'ok'}]}},
            {'type': 'assistant', 'message': {'id': 'm2', 'model': 'claude-opus-5-5',
                                              'content': [{'type': 'text', 'text': 'Closing.'}]}},
        ])
        self.assertEqual(self.hc.closing_message({'transcript_path': path}),
                         {'model': 'claude-opus-5-5', 'tool_calls': []})

    def test_closing_message_without_ids_or_assistant(self):
        # No message.id: only the last entry is the closing message.
        path = self._transcript([
            {'type': 'assistant', 'message': {'model': 'm', 'content': [self._box('older')]}},
            {'type': 'assistant', 'message': {'content': [self._box('last')]}},
        ])
        self.assertEqual(self.hc.closing_message({'transcript_path': path}), {
            'model': 'm', 'tool_calls': [('mcp__ccd_turn__wrap_up', {'recap': 'last'})]})
        path = self._transcript([{'type': 'user', 'message': {'content': 'first prompt'}}])
        self.assertIsNone(self.hc.closing_message({'transcript_path': path}))
        self.assertIsNone(self.hc.closing_message({'transcript_path': '/nope/none.jsonl'}))
        self.assertIsNone(self.hc.closing_message({}))


if __name__ == '__main__':
    unittest.main()
