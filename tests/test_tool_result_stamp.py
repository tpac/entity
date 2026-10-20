"""S0 tool write boundary: raw facts → daemon stamp, with legacy clients."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest import mock

from tests.brain_test_base import BrainTestBase
from servers.brain_traces import stamp_tool_result
from servers.dispatch_observability import _handle_trace_append
from servers.daemon_hooks import hook_recall, hook_post_response_track
from servers.host_contract import VOCAB_VERSION, contract_fingerprint
from servers.trace_contract import session_short, validate_trace_metadata

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / 'hooks' / 'scripts'


class RawToolHookTest(unittest.TestCase):
    def _wire(self, data, env):
        sys.path.insert(0, str(HOOKS))
        self.addCleanup(sys.path.remove, str(HOOKS))
        import hook_common
        import io
        # get_hook_input caches the payload per process, as one hook process
        # reads one payload; each runpy here is a new hook run.
        hook_common._hook_input_cache = None
        sock = mock.MagicMock()
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(hook_common, 'run_hook', side_effect=lambda name, fn, on_error=None, gated=True: fn()), \
                mock.patch('socket.socket', return_value=sock), \
                mock.patch('sys.stdin', io.StringIO(json.dumps(data))):
            runpy.run_path(str(HOOKS / 'post_tool_trace.py'))
        wire = json.loads(sock.sendall.call_args.args[0])
        wire['args']['metadata'] = json.loads(wire['args']['metadata'])
        return wire

    def test_the_hook_sends_no_chain(self):
        # The daemon names the S0 chain from the session's stop counter; a
        # hook off the daemon's machine has no way to know it.
        wire = self._wire({'session_id': 'session', 'tool_name': 'Read',
                           'tool_input': {'file_path': '/a.py'}}, {})
        self.assertNotIn('chain_id', wire['args'])

    def test_raw_fields_are_additive_and_patch_cap_is_explicit(self):
        patch = '*** Begin Patch\n*** Update File: /a.py\n' + 'x' * 20_000
        data = {'session_id': 'session', 'tool_name': 'apply_patch',
                'tool_input': {'command': patch}, 'tool_use_id': 'call',
                'turn_id': 'turn', 'prompt_id': 'prompt', 'future_field': 'secret'}
        wire = self._wire(data, {'PLUGIN_DATA': 'secret-env-value'})
        args = wire['args']
        self.assertEqual(args['summary'], 'apply_patch: /a.py')
        self.assertEqual(args['ref_type'], 'tool_result')
        md = args['metadata']
        self.assertEqual(md['tool'], 'apply_patch')
        self.assertEqual(md['tells'], ['PLUGIN_DATA'])
        self.assertEqual(md['payload_keys'], sorted(data))
        for key in ('tool_use_id', 'turn_id', 'prompt_id'):
            self.assertEqual(md[key], data[key])
        self.assertEqual(md['patch'], patch[:16_384])
        self.assertEqual(md['patch_truncated_chars'], len(patch) - 16_384)
        self.assertNotIn('kind', md)
        self.assertNotIn('secret', json.dumps(md))

    def test_caller_stamp_is_still_redacted_and_source_ids_are_not_invented(self):
        from servers.dispatch_common import CALLER_SESSION_KEY, CALLER_SIG_KEY
        data = {'tool_name': 'mcp__plugin_x_brain__recall',
                'tool_input': {'topic': 'hello', CALLER_SESSION_KEY: 'caller-secret',
                               CALLER_SIG_KEY: 'replayable-secret'}}
        args = self._wire(data, {})['args']
        self.assertNotIn('secret', json.dumps(args))
        self.assertEqual(set(args['metadata']), {'tool', 'tells', 'payload_keys'})

    # The servers modules the bare tool hook may load: the brain gate, which
    # every gated hook asks first (run_hook), and the two leaves it composes.
    # What the hot path must never load is daemon_config, whose import-time
    # fingerprint of servers/ costs tens of ms on every tool call (D2).
    GATE_LEAVES = {'servers.brain_gate', 'servers.brain_constants', 'servers.session_env'}

    def test_bare_tool_hook_imports_only_the_gate_leaves(self):
        # Exercise the gate's project probe, capture AND sending, with no daemon
        # or package-root env. Empty stdin would miss a future lazy import in
        # the tell probe; a session id and cwd make the gate derive the project.
        import tempfile
        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, tmp, True)
        code = ("import runpy,sys\nfrom unittest.mock import patch\n"
                "with patch('socket.socket'):\n    runpy.run_path(sys.argv[1])\n"
                "print([m for m in sys.modules if m.startswith('servers.')])")
        result = subprocess.run([sys.executable, '-c', code, str(HOOKS / 'post_tool_trace.py')],
                                input=json.dumps({'tool_name': 'apply_patch', 'session_id': 'gate-leaf-test',
                                                  'cwd': str(ROOT), 'tool_input': {'command': 'patch'}}),
                                text=True, capture_output=True,
                                env={'PATH': os.environ['PATH'], 'XDG_CONFIG_HOME': tmp,
                                     'BRAIN_TMP_DIR': tmp})
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = set(json.loads(result.stdout.strip().replace("'", '"')))
        self.assertIn('servers.brain_gate', loaded, 'the gate never ran — the test is vacuous')
        self.assertNotIn('servers.daemon_config', loaded)
        self.assertEqual(loaded - self.GATE_LEAVES, set(),
                         'the bare tool hook loaded servers modules beyond the gate leaves')


class ToolStampTest(unittest.TestCase):
    def test_event_host_wins_without_changing_session_or_input(self):
        env = {'host': 'claude-code', 'model': 'm'}
        raw = {'tool': 'apply_patch', 'tells': ['PLUGIN_DATA'],
               'tool_use_id': 'call', 'turn_id': 'turn', 'prompt_id': 'prompt',
               'payload_keys': ['tool_input'], 'patch': 'body', 'extra': 7}
        out = stamp_tool_result(raw, env)
        self.assertEqual((out['host'], out['host_status'], out['kind']),
                         ('codex', 'family', 'edit'))
        self.assertEqual(out['model'], 'm')
        for k in ('tool', 'tool_use_id', 'turn_id', 'prompt_id', 'payload_keys', 'patch', 'extra'):
            self.assertEqual(out[k], raw[k])
        self.assertNotIn('kind', raw)
        self.assertEqual(env, {'host': 'claude-code', 'model': 'm'})
        self.assertEqual(out['vocab_version'], VOCAB_VERSION)
        self.assertEqual(out['impl_identity'], contract_fingerprint())
        self.assertEqual(validate_trace_metadata('delta', 'tool_result', out), (True, ''))

    def test_only_absent_tells_use_legacy_session_identity(self):
        for raw, status, host, kind in (
            ({'tool': 'Bash'}, 'legacy', 'claude-code', 'shell'),
            ({'tool': 'Bash', 'tells': []}, 'unknown', '', ''),
            ({'tool': 'Bash', 'tells': ['CLAUDE_CODE_SESSION_ID', 'PLUGIN_DATA']},
             'ambiguous', '', ''),
            ({'tool': 'Bash', 'tells': ['not_declared']}, 'unknown', '', ''),
        ):
            with self.subTest(raw=raw):
                out = stamp_tool_result(raw, {'host': 'claude-code'})
                self.assertEqual((out['host_status'], out['host'], out['kind']),
                                 (status, host, kind))

    def test_all_rows_get_stamps_even_without_session_or_metadata(self):
        for raw in (None, [], 'bad JSON', {}, {'tool': 'Bash'}):
            with self.subTest(raw=raw):
                out = stamp_tool_result(raw, {})
                self.assertEqual(out['kind_status'], 'unknown')
                self.assertEqual(out['host_status'], 'legacy')
                self.assertEqual(validate_trace_metadata('delta', 'tool_result', out), (True, ''))
                if not isinstance(raw, dict):
                    self.assertEqual(out['raw'], raw)

    def test_unknown_tell_names_survive_for_future_contract_diagnostics(self):
        tells = ['FUTURE_HOST_TELL']
        out = stamp_tool_result({'tool': 'new_tool', 'tells': tells}, {'host': 'codex'})
        self.assertEqual(out['tells'], tells)
        self.assertEqual((out['host'], out['host_status'], out['kind_status']),
                         ('', 'unknown', 'unknown'))

    def test_daemon_overwrites_client_classification_but_preserves_model(self):
        out = stamp_tool_result({'tool': 'new_tool', 'tells': ['PLUGIN_DATA'],
                                 'kind': 'edit', 'kind_status': 'ok', 'host': 'claude-code',
                                 'impl_identity': 'fake', 'model': 'event-model'}, {'model': 'session-model'})
        self.assertEqual((out['kind'], out['kind_status'], out['host']), ('', 'unknown', 'codex'))
        self.assertEqual(out['model'], 'event-model')
        self.assertEqual(out['impl_identity'], contract_fingerprint())


class ToolWriteDoorTest(BrainTestBase):
    needs_embedder = False

    def _append(self, metadata, sid=''):
        result = _handle_trace_append(self.brain, {
            'chain_id': 's0-tooltest-0', 'scale': 's0', 'event_type': 'delta',
            'ref_type': 'tool_result', 'summary': 'original summary',
            'metadata': json.dumps(metadata), 'session_id': sid}, [])
        self.assertTrue(result['ok'], result)
        return self.brain.get_trace(result['result']['event_id'])

    def test_before_first_prompt_and_without_session_are_stamped(self):
        for sid in ('first-tool', ''):
            row = self._append({'tool': 'apply_patch', 'tells': ['PLUGIN_DATA']}, sid)
            self.assertEqual(row['summary'], 'original summary')
            self.assertEqual(row['metadata']['kind'], 'edit')
            self.assertEqual(row['metadata']['host'], 'codex')
            if sid:
                self.assertFalse(self.brain.session_env_for(sid).get('host'))

    # _append sends a stale client chain ('s0-tooltest-0'): the daemon's own
    # chain for the session wins.
    def test_row_lands_on_the_sessions_current_chain(self):
        ctx = self.brain.get_or_create_session('chain-session')
        ctx.increment_stop()
        ctx.increment_stop()
        row = self._append({'tool': 'Read'}, 'chain-session')
        self.assertEqual(row['chain_id'], ctx.s0_chain())
        self.assertEqual(row['chain_id'], 's0-%s-2' % session_short('chain-session'))

    def test_a_saved_session_is_read_back_after_its_cache_entry_drops(self):
        ctx = self.brain.get_or_create_session('saved-session')
        ctx.increment_stop()
        self.brain.discard_session_context('saved-session')   # saves, then drops
        row = self._append({'tool': 'Read'}, 'saved-session')
        self.assertEqual(row['chain_id'], 's0-%s-1' % session_short('saved-session'))

    def test_an_unseen_session_is_at_stop_zero_and_is_not_created(self):
        from servers.session_context import SessionContext
        for sid, chain in (('never-seen', 's0-%s-0' % session_short('never-seen')), ('', 's0--0')):
            row = self._append({'tool': 'Read'}, sid)
            self.assertEqual(row['chain_id'], chain)
        self.assertNotIn('never-seen', self.brain._session_contexts)
        self.assertIsNone(SessionContext.load(self.brain._session_state, 'never-seen'))

    def test_legacy_client_classifies_from_session_without_mutating_it(self):
        ctx = self.brain.get_or_create_session('old-client')
        ctx.set_env(host='claude-code', model='m')
        row = self._append({'tool': 'Bash'}, ctx.session_id)
        self.assertEqual(row['metadata']['kind'], 'shell')
        self.assertEqual(row['metadata']['host_status'], 'legacy')
        self._append({'tool': 'apply_patch', 'tells': ['PLUGIN_DATA']}, ctx.session_id)
        self.assertEqual(ctx.host, 'claude-code')

    def test_unknown_logs_one_row_per_host_tool_in_existing_dedup_window(self):
        raw = {'tool': 'new_tool', 'tells': ['PLUGIN_DATA']}
        first = self._append(raw)
        self._append(raw)
        errors = self.brain.get_recent_errors()
        errors = [e for e in errors if e['source'] == 'tool_kind_unknown']
        self.assertEqual(len(errors), 1)
        self.assertEqual(first['metadata']['kind_status'], 'unknown')
        # Names sharing >100 chars must not collapse into the same error.
        for tail in ('a', 'b'):
            self._append({'tool': 'x' * 150 + tail, 'tells': ['PLUGIN_DATA']})
        errors = [e for e in self.brain.get_recent_errors() if e['source'] == 'tool_kind_unknown']
        self.assertEqual(len(errors), 3)

    def test_unknown_error_uses_event_session_even_when_sessionless(self):
        for sid, tool in (('actual-event-session', 'unknown_one'), ('', 'unknown_two')):
            with self.subTest(session_id=sid):
                row = self._append({'tool': tool, 'tells': ['PLUGIN_DATA']}, sid)
                errors = [(error_sid, json.loads(metadata)) for error_sid, metadata
                          in self.brain.logs_conn.execute(
                              "SELECT session_id, metadata FROM debug_log "
                              "WHERE source = 'tool_kind_unknown'")
                          if tool in json.loads(metadata)['error']]
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0][0], row['session_id'])
                self.assertEqual(errors[0][0], sid)
                self.assertIn('session_id=%r' % sid, errors[0][1]['context'])


class PromptStopHostTest(BrainTestBase):
    needs_embedder = False

    def test_both_doors_resolve_tells_and_only_confident_hosts_update_session(self):
        for door in (hook_recall, hook_post_response_track):
            with self.subTest(door=door.__name__):
                sid = door.__name__
                args = {'session_id': sid, 'prompt': 'yes', 'register_only': True,
                        'hook_event_name': 'Stop', 'last_assistant_message': 'r',
                        'model': 'm', 'host': 'ignored-old-wire-host'}
                ctx = self.brain.get_or_create_session(sid)
                door(self.brain, dict(args, tells=['PLUGIN_DATA']), [])
                self.assertEqual(ctx.host, 'codex')
                self.assertEqual(ctx.model, 'm')
                door(self.brain, dict(args, tells=['CLAUDE_CODE_SESSION_ID']), [])
                self.assertEqual(ctx.host, 'claude-code')
                # No calls with host= for unknown/ambiguous/old clients. Model
                # and ordinary turn recording continue for every observation.
                for tells in ([], ['PLUGIN_DATA', 'CLAUDE_CODE_SESSION_ID'], None):
                    with mock.patch.object(ctx, 'set_env', wraps=ctx.set_env) as set_env:
                        door(self.brain, dict(args, tells=tells), [])
                    self.assertFalse(any('host' in call.kwargs for call in set_env.call_args_list))
                    self.assertEqual(ctx.host, 'claude-code')

    def test_hooks_send_tells_without_host_and_keep_existing_routing(self):
        sys.path.insert(0, str(HOOKS))
        self.addCleanup(sys.path.remove, str(HOOKS))
        import hook_common
        import types
        # PLUGIN_DATA resolves Codex, whose prompt hook asks its adapter about
        # hook approval; this test is about routing, so the adapter is a stub.
        adapter = types.SimpleNamespace(prompt_notice=lambda root: '')
        for script, command, event in (
            ('pre_response_recall.py', 'hook_recall', 'UserPromptSubmit'),
            ('post_response_track.py', 'hook_post_response_track', 'Stop'),
        ):
            data = {'prompt': 'yes', 'session_id': 'probe-session', 'model': 'm',
                    'hook_event_name': event, 'last_assistant_message': 'r'}
            with mock.patch.dict(os.environ, {'PLUGIN_DATA': 'secret',
                                              'BRAIN_TMP_DIR': os.environ['BRAIN_TMP_DIR']}, clear=True), \
                    mock.patch.dict(sys.modules, {'codex_setup': adapter}), \
                    mock.patch.object(hook_common, 'get_hook_input', return_value=data), \
                    mock.patch.object(hook_common, 'daemon_available', return_value=True), \
                    mock.patch.object(hook_common, 'daemon_call_raw', return_value={'ok': True, 'result': {}}) as call, \
                    mock.patch.object(hook_common, 'brain_debug'), \
                    mock.patch.object(hook_common, 'emit_hook_output'), \
                    mock.patch('os._exit'), \
                    mock.patch.object(hook_common, 'run_hook', side_effect=lambda name, fn, on_error=None, gated=True: fn()):
                runpy.run_path(str(HOOKS / script))
            self.assertEqual(call.call_args.args[0], command)
            args = call.call_args.args[1]
            self.assertEqual(args['tells'], ['PLUGIN_DATA'])
            self.assertNotIn('host', args)
            self.assertEqual(args['model'], 'm')
            self.assertEqual(call.call_args.kwargs['timeout'], 4.0)
            if event == 'UserPromptSubmit':
                self.assertTrue(args['register_only'])
