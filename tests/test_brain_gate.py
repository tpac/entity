"""The brain gate — one owner for "may the brain's automatic mechanics run here?"

Four layers, bottom up: the switch reader/matcher/writer (servers/brain_gate.py),
the hook boundary that asks it (hook_common.run_hook), the per-prompt machine
envelope the recall hook asks the host contract about, and the daemon side —
the `gate` command and hook_recall's first-prompt Thalamus delivery that the
envelope keeps from firing for a scheduler.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts'))

from servers import brain_gate
from servers import host_contract as hc
from tests.brain_test_base import BrainTestBase

_CLEAR = {brain_gate.PAUSED_KEY: '', brain_gate.IGNORE_KEY: ''}


def _env_file(tmp, *lines):
    path = os.path.join(tmp, 'brain', 'env')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


class TestSwitches(unittest.TestCase):
    """The reader, the matcher, the writer — no brain, no daemon."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # The process env must not leak the operator's own switches in.
        self._env = patch.dict(os.environ, _CLEAR)
        self._env.start()
        for k in _CLEAR:
            os.environ.pop(k, None)

    def tearDown(self):
        self._env.stop()

    def test_patterns_and_includes(self):
        self.assertEqual(brain_gate.parse_patterns('a, b:c,,'), ('a', 'b', 'c'))
        pats = brain_gate.parse_patterns('*,!brain,!work')
        self.assertTrue(brain_gate.project_ignored('client-x', pats))
        self.assertFalse(brain_gate.project_ignored('brain', pats))
        self.assertFalse(brain_gate.project_ignored('work', pats))
        # An unscoped session matches `*` like anything else.
        self.assertTrue(brain_gate.project_ignored('', pats))
        self.assertTrue(brain_gate.project_ignored('client-x', ('client-*',)))
        self.assertFalse(brain_gate.project_ignored('brain', ()))

    def test_decide_reads_the_env_file(self):
        env = _env_file(self.tmp, 'ANTHROPIC_API_KEY=sk-x', 'BRAIN_IGNORE_PROJECTS=work,client-*')
        self.assertTrue(brain_gate.decide('brain', env).run)
        d = brain_gate.decide('client-7', env)
        self.assertFalse(d.run)
        self.assertTrue(d.reason.startswith(brain_gate.REASON_IGNORED))
        env = _env_file(self.tmp, 'export BRAIN_PAUSED="1"  # dashboard wrote this')
        self.assertEqual(brain_gate.decide('brain', env).reason, brain_gate.REASON_PAUSED)

    def test_process_env_is_not_a_source(self):
        # The daemon launcher sourced the file at START; a stale BRAIN_PAUSED
        # in its env must not outlive `gate resume` removing the line.
        env = _env_file(self.tmp, 'ANTHROPIC_API_KEY=sk-x')
        with patch.dict(os.environ, {brain_gate.PAUSED_KEY: '1'}):
            self.assertTrue(brain_gate.decide('brain', env).run)
            self.assertFalse(brain_gate.status(env)['paused'])

    def test_writer_keeps_other_lines_and_mode(self):
        env = _env_file(self.tmp, 'ANTHROPIC_API_KEY=sk-x', "BRAIN_AGENT_NAME='Anchor'")
        brain_gate.set_paused(True, env)
        brain_gate.add_ignore('client-*', env)
        brain_gate.add_ignore('client-*', env)   # idempotent
        brain_gate.add_ignore('!brain', env)
        with open(env) as f:
            lines = f.read().splitlines()
        self.assertIn('ANTHROPIC_API_KEY=sk-x', lines)
        self.assertIn("BRAIN_AGENT_NAME='Anchor'", lines)
        self.assertIn('BRAIN_PAUSED=1', lines)
        self.assertIn('BRAIN_IGNORE_PROJECTS=client-*,!brain', lines)
        self.assertEqual(oct(os.stat(env).st_mode & 0o777), '0o600')
        st = brain_gate.status(env)
        self.assertTrue(st['paused'])
        self.assertEqual(st['ignore_projects'], ['client-*', '!brain'])
        # resume / unignore remove, and an emptied list removes its line
        brain_gate.set_paused(False, env)
        brain_gate.remove_ignore('client-*', env)
        brain_gate.remove_ignore('!brain', env)
        with open(env) as f:
            text = f.read()
        self.assertNotIn('BRAIN_PAUSED', text)
        self.assertNotIn('BRAIN_IGNORE_PROJECTS', text)
        self.assertIn('ANTHROPIC_API_KEY=sk-x', text)
        with self.assertRaises(ValueError):
            brain_gate.set_switch('ANTHROPIC_API_KEY', 'x', env)
        with self.assertRaises(ValueError):
            brain_gate.add_ignore('   ', env)

    def test_notice_names_the_switch_and_the_pattern(self):
        env = _env_file(self.tmp, 'BRAIN_IGNORE_PROJECTS=work,client-*')
        self.assertEqual(brain_gate.notice(brain_gate.decide('brain', env), env), '')
        n = brain_gate.notice(brain_gate.decide('client-x', env), env)
        self.assertIn("project 'client-x'", n)
        self.assertIn("pattern 'client-*'", n)
        self.assertIn(env, n)
        self.assertIn('Tell the operator', n)
        env = _env_file(self.tmp, 'BRAIN_PAUSED=1')
        n = brain_gate.notice(brain_gate.decide('brain', env), env)
        self.assertIn('PAUSED', n)
        self.assertIn('BRAIN_PAUSED', n)
        self.assertIn('Config tab', n)

    def test_notice_url_honors_the_env_file_port(self):
        # The boot gate runs before any shell sourced the env file, so the
        # file is the URL's source when the process env has no port.
        from servers.brain_constants import user_env_file
        with patch.dict(os.environ, {'XDG_CONFIG_HOME': self.tmp}):
            os.environ.pop('DASHBOARD_PORT', None)
            env = _env_file(self.tmp, 'BRAIN_PAUSED=1', 'DASHBOARD_PORT=48123')
            self.assertEqual(env, user_env_file())
            n = brain_gate.notice(brain_gate.decide('brain', env), env)
            self.assertIn('http://localhost:48123', n)
            os.environ['DASHBOARD_PORT'] = '48999'
            self.assertIn('http://localhost:48999', brain_gate.notice(brain_gate.decide('brain', env), env))

    def test_verdicts_match_decide_from_one_read(self):
        env = _env_file(self.tmp, 'BRAIN_IGNORE_PROJECTS=*,!brain')
        got = brain_gate.verdicts(['brain', 'client-x', ''], env)
        for p, v in got.items():
            self.assertEqual(v, brain_gate.decide(p, env))
        self.assertTrue(got['brain'].run)
        self.assertFalse(got['client-x'].run)

    def test_gated_boot_prints_the_notice_and_nothing_else(self):
        # boot-brain.sh under a paused env file: exit 0, stdout is exactly the
        # gate's notice — before the runtime, daemon or MCP work it never reaches.
        import subprocess
        _env_file(self.tmp, 'BRAIN_PAUSED=1')
        env = dict(os.environ, XDG_CONFIG_HOME=self.tmp)
        root = os.path.join(os.path.dirname(__file__), '..')
        r = subprocess.run(['bash', os.path.join(root, 'hooks', 'scripts', 'boot-brain.sh')],
                           input=json.dumps({'session_id': 'test-gated-boot', 'cwd': root,
                                             'source': 'startup'}),
                           capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines(),
                         [brain_gate.notice(brain_gate.Decision(False, brain_gate.REASON_PAUSED),
                                            os.path.join(self.tmp, 'brain', 'env'))])
        self.assertIn('gated (paused)', r.stderr)
        # A resume or compaction inside the same conversation is still gated
        # (exit 0, nothing booted) but prints nothing — the operator was told.
        for src in ('resume', 'compact'):
            r = subprocess.run(['bash', os.path.join(root, 'hooks', 'scripts', 'boot-brain.sh')],
                               input=json.dumps({'session_id': 'test-gated-boot', 'cwd': root,
                                                 'source': src}),
                               capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual((r.returncode, r.stdout), (0, ''), (src, r.stderr))
            self.assertIn('gated (paused)', r.stderr)

    def test_gated_boot_imports_the_gate_from_the_plugin_tree(self):
        # boot's pass also imports hook_common, which puts the tree of an
        # inherited BRAIN_SERVER_DIR on sys.path. The gate must still come from
        # this plugin: a decoy servers/ there may not decide the boot.
        import subprocess
        _env_file(self.tmp, 'BRAIN_PAUSED=1')
        decoy = os.path.join(self.tmp, 'decoy', 'servers')
        os.makedirs(decoy)
        with open(os.path.join(decoy, '__init__.py'), 'w') as f:
            f.write("raise ImportError('decoy servers package imported')\n")
        env = dict(os.environ, XDG_CONFIG_HOME=self.tmp, BRAIN_SERVER_DIR=decoy)
        root = os.path.join(os.path.dirname(__file__), '..')
        r = subprocess.run(['bash', os.path.join(root, 'hooks', 'scripts', 'boot-brain.sh')],
                           input=json.dumps({'session_id': 'test-gated-boot', 'cwd': root,
                                             'source': 'startup'}),
                           capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn('decoy', r.stderr)
        self.assertEqual(r.stdout.strip().splitlines(),
                         [brain_gate.notice(brain_gate.Decision(False, brain_gate.REASON_PAUSED),
                                            os.path.join(self.tmp, 'brain', 'env'))])

    def test_session_project_is_derived_once_and_cached(self):
        folder = os.path.join(self.tmp, 'client-x')
        os.makedirs(folder)
        with patch.dict(os.environ, {'BRAIN_TMP_DIR': self.tmp}):
            self.assertEqual(brain_gate.session_project(folder, 'sess-1234abcd'), 'client-x')
            # The marker now answers without the folder (a later hook, no cwd)
            self.assertEqual(brain_gate.session_project('', 'sess-1234abcd'), 'client-x')
            # A different cwd for the same session still reads the marker
            self.assertEqual(brain_gate.session_project(self.tmp, 'sess-1234abcd'), 'client-x')
            # No session id: derive, don't cache
            self.assertEqual(brain_gate.session_project(folder, ''), 'client-x')
            self.assertEqual(brain_gate.session_project('', ''), '')
            # A transient git failure (None project) is NOT cached — the next
            # hook probes again instead of living unscoped for the session.
            with patch('servers.session_env.detect_session_env', return_value=('unknown', None, None)):
                self.assertEqual(brain_gate.session_project(folder, 'sess-transient'), '')
            self.assertEqual(brain_gate.session_project(folder, 'sess-transient'), 'client-x')

    def test_same_minute_codex_sessions_keep_their_own_project(self):
        """Codex ids are UUIDv7: chats opened in the same minute share their
        first 8 chars. Each must still be judged by its own project."""
        a_dir, b_dir = (os.path.join(self.tmp, n) for n in ('client-a', 'client-b'))
        os.makedirs(a_dir)
        os.makedirs(b_dir)
        a = '01a122b6-9059-7ca1-bb1b-a5e6ff4ce0cf'
        b = '01a122b6-f3c2-7d10-8e4a-17c09b2d5e61'
        with patch.dict(os.environ, {'BRAIN_TMP_DIR': self.tmp}):
            self.assertEqual(brain_gate.session_project(a_dir, a), 'client-a')
            self.assertEqual(brain_gate.session_project(b_dir, b), 'client-b')
            self.assertEqual(brain_gate.session_project('', a), 'client-a')
            self.assertEqual(brain_gate.session_project('', b), 'client-b')


class TestRunHookGate(unittest.TestCase):
    """hook_common.run_hook asks the gate before the body runs."""

    def setUp(self):
        import hook_common
        self.hc = hook_common
        self.tmp = tempfile.mkdtemp()
        self._env = patch.dict(os.environ, {
            'XDG_CONFIG_HOME': self.tmp, 'BRAIN_TMP_DIR': self.tmp,
            'HOOK_INPUT': json.dumps({'session_id': 'gate-test-0001', 'cwd': self.tmp}),
        })
        self._env.start()
        for k in _CLEAR:
            os.environ.pop(k, None)
        self.hc._hook_input_cache = None
        self.calls = []

    def tearDown(self):
        self.hc._hook_input_cache = None
        self._env.stop()

    def _body(self):
        self.calls.append('ran')

    def test_paused_brain_skips_the_body(self):
        _env_file(self.tmp, 'BRAIN_PAUSED=1')
        with patch.object(self.hc, 'brain_debug'):
            self.hc.run_hook('x', self._body)
        self.assertEqual(self.calls, [])

    def test_exempt_hook_runs_while_paused(self):
        _env_file(self.tmp, 'BRAIN_PAUSED=1')
        self.hc.run_hook('x', self._body, gated=False)
        self.assertEqual(self.calls, ['ran'])

    def test_ignored_project_skips_the_body(self):
        # cwd is the tmp dir: a non-repo folder, so its basename is its project
        _env_file(self.tmp, 'BRAIN_IGNORE_PROJECTS=%s' % os.path.basename(self.tmp))
        with patch.object(self.hc, 'brain_debug'):
            self.hc.run_hook('x', self._body)
        self.assertEqual(self.calls, [])

    def test_open_gate_runs_the_body(self):
        _env_file(self.tmp, 'ANTHROPIC_API_KEY=sk-x')
        self.hc.run_hook('x', self._body)
        self.assertEqual(self.calls, ['ran'])

    def test_gate_failure_fails_open_and_logs(self):
        with patch('servers.brain_gate.hook_allows', side_effect=RuntimeError('boom')), \
                patch.object(self.hc, 'log_hook_error') as log:
            self.hc.run_hook('x', self._body)
        self.assertEqual(self.calls, ['ran'])
        self.assertTrue(log.called)

    def test_hook_input_falls_back_to_stdin_once(self):
        os.environ.pop('HOOK_INPUT')
        self.hc._hook_input_cache = None
        import io
        with patch.object(sys, 'stdin', io.StringIO(json.dumps({'session_id': 's', 'tool_name': 'Bash'}))):
            first = self.hc.get_hook_input()
            second = self.hc.get_hook_input()   # stdin is spent; the cache answers
        self.assertEqual(first.get('tool_name'), 'Bash')
        self.assertIs(first, second)


class TestMachineEnvelope(unittest.TestCase):
    """The per-prompt axis: which prompts are the harness's, per host
    (host_contract.read_prompt — recorded as the harness's, never recalled on)."""

    def _harness(self, prompt, host):
        intake = hc.read_prompt(prompt, host)
        return (intake.pieces[0][2] if intake.pieces and intake.pieces[0][0] == 'harness'
                and not intake.recall_text else '')

    def test_claude_code_envelopes(self):
        self.assertEqual(self._harness(
            '[SYSTEM NOTIFICATION - NOT USER INPUT]\n<task-notification>\n<task-id>x</task-id>',
            'claude-code'), '<task-notification>')
        self.assertEqual(self._harness(
            '<scheduled-task name="probe" file="/x/SKILL.md">\nThis is an automated run',
            'claude-code'), '<scheduled-task')
        self.assertEqual(self._harness('please check the scheduled task list', 'claude-code'), '')
        # Naming the tag mid-sentence is talking ABOUT it — an operator prompt.
        self.assertEqual(self._harness(
            'why does a <task-notification> skip recall?', 'claude-code'), '')
        self.assertEqual(hc.read_prompt('', 'claude-code').pieces, ())

    def test_unknown_host_reads_every_table_and_codex_keeps_notifications(self):
        self.assertEqual(self._harness('<task-notification>…', hc.HOST_UNKNOWN), '<task-notification>')
        self.assertEqual(self._harness('<task-notification>…', 'codex'), '<task-notification>')
        self.assertEqual(self._harness('<scheduled-task name="x">', 'codex'), '')

    def test_declared_policies_are_vocabulary(self):
        for host, entry in hc.HOST_CONTRACT.items():
            for tag, decl in entry['envelopes'].items():
                self.assertEqual(hc.envelope_policy(host, tag), decl)
        self.assertEqual(hc.validate_host_contract(), [])


class TestDaemonSide(BrainTestBase):
    """The `gate` command, and the first-prompt delivery the envelope guards."""
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.cfg = tempfile.mkdtemp()
        # Not `_env`: BrainTestBase owns that name for its brain_env context.
        self._cfg_patch = patch.dict(os.environ, {'XDG_CONFIG_HOME': self.cfg})
        self._cfg_patch.start()
        for k in _CLEAR:
            os.environ.pop(k, None)

    def tearDown(self):
        self._cfg_patch.stop()
        super().tearDown()

    def test_gate_command_round_trip(self):
        from servers.daemon_dispatch import dispatch_command
        r = dispatch_command(self.brain, 'gate', {'action': 'status'}, [])
        self.assertTrue(r['ok'])
        self.assertFalse(r['result']['paused'])
        r = dispatch_command(self.brain, 'gate', {'action': 'pause'}, [])
        self.assertTrue(r['result']['paused'])
        self.assertIn('note', r['result'])
        r = dispatch_command(self.brain, 'gate', {'action': 'ignore', 'pattern': 'client-*'}, [])
        self.assertEqual(r['result']['ignore_projects'], ['client-*'])
        r = dispatch_command(self.brain, 'gate', {'action': 'resume'}, [])
        self.assertFalse(r['result']['paused'])
        r = dispatch_command(self.brain, 'gate', {'action': 'unignore', 'pattern': 'client-*'}, [])
        self.assertEqual(r['result']['ignore_projects'], [])
        self.assertFalse(dispatch_command(self.brain, 'gate', {'action': 'ignore'}, [])['ok'])
        self.assertFalse(dispatch_command(self.brain, 'gate', {'action': 'explode'}, [])['ok'])
        # It wrote the user env file under XDG_CONFIG_HOME, nowhere else.
        self.assertEqual(r['result']['env_file'], os.path.join(self.cfg, 'brain', 'env'))

    def test_gate_status_answers_verdicts_for_projects(self):
        from servers.daemon_dispatch import dispatch_command
        dispatch_command(self.brain, 'gate', {'action': 'ignore', 'pattern': '*'}, [])
        dispatch_command(self.brain, 'gate', {'action': 'ignore', 'pattern': '!brain'}, [])
        r = dispatch_command(self.brain, 'gate', {'action': 'status',
                                                 'projects': ['brain', 'client-x', '']}, [])
        v = r['result']['verdicts']
        self.assertEqual(v['brain'], {'run': True, 'reason': ''})
        self.assertEqual(v['client-x'], {'run': False, 'reason': 'ignored_project:client-x'})
        self.assertFalse(v['']['run'])
        self.assertNotIn('verdicts', dispatch_command(self.brain, 'gate', {}, [])['result'])
        dispatch_command(self.brain, 'gate', {'action': 'pause'}, [])
        r = dispatch_command(self.brain, 'gate', {'projects': ['brain']}, [])
        self.assertEqual(r['result']['verdicts']['brain']['reason'], brain_gate.REASON_PAUSED)

    def test_set_api_key_writes_the_env_file_once(self):
        from servers.daemon_dispatch import dispatch_command
        from servers.brain_constants import read_env_file_key
        env = os.path.join(self.cfg, 'brain', 'env')
        for bad in ('changeme', 'sk-a', 'sk-ant with space', ''):
            r = dispatch_command(self.brain, 'set_api_key', {'api_key': bad}, [])
            self.assertFalse(r['ok'], bad)
            if bad:
                self.assertNotIn(bad, r['error'])   # never echo the attempt
        self.assertFalse(os.path.exists(env))        # nothing written
        dispatch_command(self.brain, 'gate', {'action': 'pause'}, [])
        r = dispatch_command(self.brain, 'set_api_key', {'api_key': 'sk-ant-test-abc123'}, [])
        self.assertEqual(r, {'ok': True, 'result': {'key_present': True, 'env_file': env}})
        r = dispatch_command(self.brain, 'set_api_key', {'api_key': 'sk-ant-NEW-key-value'}, [])
        self.assertTrue(r['ok'])
        content = open(env).read()
        self.assertEqual(content.count('ANTHROPIC_API_KEY='), 1)   # replaced, not appended
        self.assertEqual(read_env_file_key(env, 'ANTHROPIC_API_KEY'), 'sk-ant-NEW-key-value')
        self.assertIn('BRAIN_PAUSED=1\n', content)                  # other lines kept
        self.assertEqual(os.stat(env).st_mode & 0o777, 0o600)
        # The key crossed the socket and nothing on the daemon side kept it:
        # no trace row, no debug line, no error row carries the value.
        for (table,) in self.brain.logs_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            rows = self.brain.logs_conn.execute('SELECT * FROM "%s"' % table).fetchall()
            self.assertNotIn('sk-ant-NEW-key-value', repr(rows), table)
            self.assertNotIn('sk-ant-test-abc123', repr(rows), table)

    def _user_rows(self, sid):
        return self.brain.logs_conn.execute(
            "SELECT metadata FROM trace_events WHERE scale='s0' AND session_id=? "
            "AND ref_type='user_message'", (sid,)).fetchall()

    def test_first_operator_prompt_delivers_once_and_a_machine_envelope_never(self):
        from servers.daemon_hooks import hook_recall
        from servers.channels.thalamus import thalamus
        thalamus.file(self.brain, 'test', 'a due notice for every session', for_whom='all')

        # A scheduled-task session: envelope → register_only → no delivery,
        # and the row says which envelope.
        sid = 'sched-sess-0001'
        with patch('servers.daemon_hooks._run_surface') as surface, \
                patch.object(self.brain, 'recall') as recall:
            prompt = '<scheduled-task name="x" file="/x">run it</scheduled-task>'
            r = hook_recall(self.brain, {
                "prompt": prompt, "session_id": sid, "register_only": True,
                "pieces": [["harness", prompt, '<scheduled-task']], "recall_text": ""}, [])
        self.assertEqual(r["json"].get("decision"), "approve")
        recall.assert_not_called(); surface.assert_not_called()
        self.assertEqual(self._user_rows(sid), [], 'a scheduled run is not the operator speaking')
        rows = self.brain.logs_conn.execute(
            "SELECT ref_id FROM trace_events WHERE scale='s0' AND session_id=? "
            "AND ref_type='env_message'", (sid,)).fetchall()
        self.assertEqual(rows, [('<scheduled-task',)], 'and the row says which envelope')
        self.assertFalse(self.brain.get_or_create_session(sid).prompt_delivered)
        self.assertEqual(thalamus.pull(self.brain, sid, via='prompt')[1], 1,
                         'the notice must still be due for this session — nothing consumed it')

        # An operator's session: the first prompt carries the notice, the
        # second does not, and the flag persists on the session.
        sid = 'operator-sess-0001'
        with patch('servers.daemon_hooks._run_surface', return_value=None):
            r1 = hook_recall(self.brain, {"prompt": "hello there, what is new", "session_id": sid}, [])
            ctx = self.brain.get_or_create_session(sid)
            self.assertTrue(ctx.prompt_delivered)
            r2 = hook_recall(self.brain, {"prompt": "and now a second question", "session_id": sid}, [])
        self.assertIn('a due notice for every session', r1["json"].get("additionalContext", ""))
        self.assertNotIn('a due notice', r2["json"].get("additionalContext", "") or "")
        rows = self.brain.query_traces(session_id=sid).get('events', [])
        delivered = [e for e in rows if e.get('ref_type') == 'thalamus_delivery']
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].get('ref_id'), 'prompt')

    def test_short_first_prompt_leaves_the_moment_for_the_next_real_one(self):
        from servers.daemon_hooks import hook_recall
        sid = 'short-first-0001'
        with patch('servers.daemon_hooks._run_surface'), patch.object(self.brain, 'recall'):
            hook_recall(self.brain, {"prompt": "yes", "session_id": sid, "register_only": True}, [])
        self.assertFalse(self.brain.get_or_create_session(sid).prompt_delivered)

    def test_boot_render_delivers_nothing(self):
        from servers.channels.thalamus import thalamus
        thalamus.file(self.brain, 'test', 'a reminder nobody should eat at boot', for_whom='all')
        sid = 'boot-only-0001'
        self.brain.reset_session_activity(session_id=sid, cwd='')
        text = self.brain.format_boot_context(session_id=sid)
        self.assertNotIn('a reminder nobody should eat at boot', text)
        self.assertEqual(thalamus.pull(self.brain, sid, via='prompt')[1], 1)


if __name__ == '__main__':
    unittest.main()
