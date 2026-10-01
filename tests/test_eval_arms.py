"""eval/arms.py — real brain daemons that differ only by code.

Arms are keyless here (user_env=None), so no daemon can spend tokens, and each
lives under its own temp root with a scratch HOME: nothing reaches the
operator's brain or config. The daemon tests boot real daemons from this
checkout on seed brains, the path every eval arm takes.
"""
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.eval_optional import require_eval  # noqa: E402

require_eval()  # D-8: the launcher lives in eval/, which the public tree omits

from eval.arms import Arm, ArmError, Fleet, manifest_commands  # noqa: E402
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
        self.assertEqual([a.proc.returncode, b.proc.returncode], [0, 0])

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
