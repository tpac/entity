"""eval/arms.py — real brain daemons that differ only by code.

Arms are keyless here (user_env=None), so no daemon can spend tokens, and each
lives under its own temp root with a scratch HOME: nothing reaches the
operator's brain or config. The daemon tests boot real daemons from this
checkout on seed brains, the path every eval arm takes.
"""
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.eval_optional import require_eval  # noqa: E402

require_eval()  # D-8: the launcher lives in eval/, which the public tree omits

from eval.arms import Arm, ArmError, Fleet, manifest_commands, tree_digest  # noqa: E402
from servers.brain import Brain  # noqa: E402
from servers.daemon_config import get_db_lock_path  # noqa: E402

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Roots(unittest.TestCase):
    """Arm folders are never deleted by the launcher (evals keep brains);
    tests delete their own."""

    def setUp(self):
        self._roots = []

    def tearDown(self):
        for r in self._roots:
            shutil.rmtree(r, ignore_errors=True)

    def _arm(self, label='t', brain='seed', **kw):
        root = tempfile.mkdtemp(prefix='entity-arm-test-')
        self._roots.append(root)
        kw.setdefault('user_env', None)
        return Arm(TREE, brain, label=label, root=root, **kw)


class TestArmSetup(_Roots):

    def test_env_is_launchds_plus_identity_and_nothing_from_the_launcher(self):
        leaks = {'ASPECTS_JSON_PATH': '/elsewhere/aspects.json',
                 'BRAIN_S1E_VIEW_POLICY': 'leak', 'PYTHONPATH': '/elsewhere',
                 'CLAUDE_CODE_SESSION_ID': 'the-launchers-session'}
        saved = {k: os.environ.get(k) for k in leaks}
        os.environ.update(leaks)
        try:
            arm = self._arm()
            arm.port = 49999
            env = arm._build_env()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for k in leaks:
            self.assertNotIn(k, env)
        self.assertEqual(env['HOME'], arm.home)
        self.assertEqual(env['BRAIN_DB_DIR'], arm.data)
        self.assertEqual(env['BRAIN_DAEMON_PORT'], '49999')
        self.assertTrue(env['BRAIN_INSTANCE'].startswith('arm-t-'))
        self.assertEqual(env['ORT_DISABLE_ALL_ACCELERATORS'], '1')

    def test_hooks_come_from_the_trees_own_manifest(self):
        stop = manifest_commands(TREE, 'Stop')
        self.assertEqual(len(stop), 1)
        self.assertIn('post-response-track.sh', stop[0][0])
        self.assertEqual(stop[0][1], 5)
        # Tool-scoped hooks have matchers; a turn makes no tool calls.
        self.assertEqual(manifest_commands(TREE, 'PreToolUse'), [])

    def test_arm_config_is_a_copy_plus_shell_quoted_lines(self):
        src = os.path.join(tempfile.mkdtemp(prefix='arm-env-'), 'env')
        self._roots.append(os.path.dirname(src))
        with open(src, 'w') as f:
            f.write('BRAIN_AGENT_NAME=Jade\n')
        arm = self._arm(user_env=src,
                        extra_env={'BRAIN_OPERATOR_NAME': "Ann O'Brien"})
        arm._prepare_home()
        path = os.path.join(arm.home, '.config', 'brain', 'env')
        out = subprocess.run(
            ['/bin/bash', '-c', '. "$1"; printf "%s|%s" "$BRAIN_AGENT_NAME" '
             '"$BRAIN_OPERATOR_NAME"', '_', path],
            capture_output=True, text=True, env={}).stdout
        self.assertEqual(out, "Jade|Ann O'Brien")
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)

    def test_a_seed_is_an_empty_marked_folder(self):
        arm = self._arm()
        arm._prepare_brain()
        self.assertEqual(os.listdir(arm.data), ['eval-brain'])
        self.assertIsNotNone(Brain.read_eval_mark(arm.data))

    def test_a_copied_brain_is_marked_again(self):
        src = tempfile.mkdtemp(prefix='arm-premade-')
        self._roots.append(src)
        open(os.path.join(src, 'brain.db'), 'w').close()
        Brain.mark_eval_brain(src)
        os.utime(Brain.eval_brain_marker(src), (1_000.0, 1_000.0))
        arm = self._arm(brain=src)
        arm._prepare_brain()
        self.assertTrue(os.path.isfile(os.path.join(arm.data, 'brain.db')))
        self.assertGreater(Brain.read_eval_mark(arm.data), 1_000.0)

    def test_refuses_a_brain_a_daemon_has_open(self):
        src = tempfile.mkdtemp(prefix='arm-live-')
        self._roots.append(src)
        open(os.path.join(src, 'brain.db'), 'w').close()
        with open(get_db_lock_path(os.path.join(src, 'brain.db')), 'a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ArmError, 'running daemon'):
                self._arm(brain=src)._prepare_brain()

    def test_refuses_a_folder_without_a_brain(self):
        src = tempfile.mkdtemp(prefix='arm-empty-')
        self._roots.append(src)
        with self.assertRaisesRegex(ArmError, 'no brain.db'):
            self._arm(brain=src)._prepare_brain()

    def test_proof_refuses_a_daemon_that_is_not_this_arm(self):
        arm = self._arm()
        with self.assertRaisesRegex(ArmError, 'db_dir'):
            arm._prove({'source_dir': TREE, 'db_dir': '/somewhere/else',
                        'code_fingerprint': 'x', 'eval_brain_since': 1.0})


class TestTreeDigest(unittest.TestCase):
    """What an arm executes: servers/ and hooks/ in full, plus the runtime
    lock it syncs. Nothing else of the tree moves the digest."""

    def setUp(self):
        self.tree = tempfile.mkdtemp(prefix='tree-digest-')
        self._write('servers/a.py', 'x = 1\n')
        self._write('hooks/hooks.json', '{}\n')
        self._write('uv.lock', 'lock 1\n')

    def tearDown(self):
        shutil.rmtree(self.tree, ignore_errors=True)

    def _write(self, rel, text):
        path = os.path.join(self.tree, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)

    def test_code_data_hooks_and_runtime_each_move_it(self):
        seen = {tree_digest(self.tree)}
        for rel, text in (('hooks/hooks.json', '{"hooks": {}}\n'),
                          ('servers/scales/aspects.json', '[]\n'),
                          ('uv.lock', 'lock 2\n'),
                          ('pyproject.toml', '[project]\n')):
            self._write(rel, text)
            seen.add(tree_digest(self.tree))
        self.assertEqual(len(seen), 5)

    def test_eval_code_docs_bytecode_and_dot_entries_do_not(self):
        base = tree_digest(self.tree)
        self._write('eval/driver.py', 'y = 2\n')
        self._write('docs/notes.md', 'z\n')
        self._write('servers/__pycache__/a.cpython-311.pyc', 'junk')
        self._write('hooks/.DS_Store', 'finder')
        self._write('servers/.a.py.swp', 'editor')
        self.assertEqual(tree_digest(self.tree), base)


class TestArmDrain(_Roots):
    """drain() retries what drain_embeddings asks to be retried, and fails
    the run on anything else."""

    def test_a_busy_or_loading_tick_is_retried(self):
        arm = self._arm()
        arm.result = mock.Mock(side_effect=[
            {'ok': False, 'busy': True, 'error': 'drain lock held'},
            {'ok': False, 'embedder_ready': False, 'error': 'not ready'},
            {'ok': True, 'pending_after': 0, 'embedder_ready': True}])
        with mock.patch('eval.arms.time.sleep'):
            self.assertTrue(arm.drain()['ok'])
        self.assertEqual(arm.result.call_count, 3)

    def test_vectors_left_behind_fail_the_run_at_once(self):
        arm = self._arm()
        arm.result = mock.Mock(return_value={
            'ok': False, 'embedder_ready': True, 'pending_after': 3,
            'error': '3 ids re-enqueued by a failed batch'})
        with self.assertRaisesRegex(ArmError, 're-enqueued'):
            arm.drain()
        self.assertEqual(arm.result.call_count, 1)


class TestArmDaemons(_Roots):
    """Real daemons from this checkout, keyless, on seed brains."""

    def test_two_arms_on_one_tree_run_a_session_each(self):
        a, b = self._arm('a'), self._arm('b')
        with Fleet([a, b]) as fleet:
            self.assertNotEqual(a.port, b.port)
            for arm in (a, b):
                self.assertIsNotNone(arm.ping['eval_brain_since'])
            sid, boot = fleet['a'].begin()
            self.assertEqual([r['rc'] for r in boot], [0])
            turn = fleet['a'].turn(sid, 'The launcher reads the tree manifest.',
                                   'Noted.')
            self.assertEqual([r['rc'] for r in turn['recall'] + turn['stop']],
                             [0, 0])
            self.assertEqual([r['rc'] for r in fleet['a'].end(sid)], [0])
            # The exchange is named the way the host names a turn, so the
            # turn rule groups it by the host's id, not the unstamped fallback.
            from servers.trace_contract import HOST_TURN_FIELD
            rows = fleet['a'].result('query_traces', scale='s0', session_id=sid,
                                     ref_type='user_message')['events']
            meta = rows[0]['metadata']
            meta = json.loads(meta) if isinstance(meta, str) else meta
            self.assertEqual(meta.get(HOST_TURN_FIELD), fleet['a']._turn_ids[sid])
            said = fleet['a'].command('recall_episodes', session_id=sid,
                                      limit=10)
            self.assertTrue(said.get('ok'), said)
            texts = str(said['result'])
            self.assertIn('The launcher reads the tree manifest.', texts)
            self.assertIn('Noted.', texts)
            # b saw none of a's session: one brain per arm.
            other = fleet['b'].command('recall_episodes', session_id=sid,
                                       limit=10)
            self.assertNotIn('The launcher reads the tree manifest.',
                             str(other.get('result')))
            # A command's own `name` argument reaches the daemon.
            eff = fleet['b'].result('get_interaction_effective',
                                    name='trace_recording')
            self.assertEqual(eff['name'], 'trace_recording')
            # A refused command fails the run; the raw reply stays readable.
            with self.assertRaisesRegex(ArmError, 'no_such_command failed'):
                fleet['b'].result('no_such_command')
            self.assertFalse(fleet['b'].command('no_such_command').get('ok'))
            self.assertTrue(fleet['b'].drain()['ok'])
        self.assertEqual([a.proc.returncode, b.proc.returncode], [0, 0])
        # The arms' copies of the operator config (the API key) are gone;
        # their folders stay.
        for arm in (a, b):
            self.assertFalse(os.path.exists(
                os.path.join(arm.home, '.config', 'brain', 'env')))
            self.assertTrue(os.path.isfile(os.path.join(arm.data, 'brain.db')))

    def test_a_start_that_fails_leaves_nothing_running(self):
        arm = self._arm('s')
        with mock.patch('eval.arms.READY_TIMEOUT_S', 0):
            with self.assertRaisesRegex(ArmError, 'no answer'):
                arm.start()
        self.assertIsNotNone(arm.proc.poll())
        self.assertFalse(os.path.exists(
            os.path.join(arm.home, '.config', 'brain', 'env')))

    def test_a_crashed_arm_fails_the_run(self):
        with Fleet([self._arm('c')]) as fleet:
            arm = fleet['c']
            os.kill(arm.proc.pid, signal.SIGKILL)
            arm.proc.wait()
            with self.assertRaisesRegex(ArmError, 'daemon exited'):
                arm.command('ping')

    def test_a_crash_the_daemon_survives_by_restarting_fails_the_run(self):
        # The supervisor's warm retry keeps the process alive; its
        # 'Daemon started ... restarts=N' line is the only evidence.
        with Fleet([self._arm('w')]) as fleet:
            arm = fleet['w']
            self.assertTrue(arm.command('ping').get('ok'))
            with open(os.path.join(arm.data, 'daemon.log'), 'a') as log:
                log.write('[brain-daemon 00:00:00] Daemon started. PID=1, '
                          'addr=127.0.0.1:1, workers=1, restarts=1, '
                          'threads=1(x)\n')
            with self.assertRaisesRegex(ArmError, 'restarted itself'):
                arm.command('ping')


if __name__ == '__main__':
    unittest.main()
