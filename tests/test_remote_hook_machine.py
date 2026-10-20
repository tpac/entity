"""Hooks on a machine with no route to the daemon — every manifest command, run for real.

Cowork runs plugin command hooks in a cloud Linux VM (CLAUDE_CODE_ENTRYPOINT=
remote_cowork) that cannot reach the operator's Mac, where the daemon lives
(brain node bc9e9e30). There, every command in hooks/hooks.json and
hooks/hooks.codex.json must end at once and quietly: no runtime build, no brain
dir, no daemon contact, and stdout empty except SessionStart's one notice. On a
Mac, and under every other entrypoint, the same commands must go past the rule.

The rule has three copies: host_contract `remote_hooks` (the source),
hook_common (the Python mirror, held exactly by test_host_contract) and
resolve-brain-db.sh (the shell copy, held here). The markers come from the
contract, so a marker added there and not in the resolver fails this file.

The machine is simulated on the Mac: a throwaway plugin root (the real hooks/,
servers/ symlinked, ensure-runtime.sh stubbed to leave a marker instead of
building), a temp HOME, a listener on BRAIN_DAEMON_PORT that counts accepts, and
"Linux" from a `uname` stand-in on PATH plus a sitecustomize.py that patches
os.uname. The sitecustomize sits at the plugin root because boot-brain.sh's
python pass sets PYTHONPATH to it, and it acts only when the test asks.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_HOOKS = os.path.join(ROOT, 'hooks', 'scripts')

from servers import host_contract as hc  # noqa: E402

MANIFESTS = ('hooks/hooks.json', 'hooks/hooks.codex.json')


def _hook_common():
    """hook_common, imported the way test_host_contract does — with
    hooks/scripts on sys.path only for the import, inside the test."""
    sys.path.insert(0, _HOOKS)
    try:
        import hook_common
    finally:
        sys.path.remove(_HOOKS)
    return hook_common
LOCAL_ENTRYPOINTS = ('cli', 'claude-desktop', 'sdk-cli', 'sdk-ts', 'sdk-py')
SID = '74b42362-16e6-50c8-a5ed-b36da71595e8'   # a Cowork-shaped (UUIDv5) session id
BUDGET_S = 0.2

_STUB_RUNTIME = '''#!/bin/sh
# Test stub: records that the runtime bootstrap was reached, builds nothing.
: > "$(cd "$(dirname "$0")/../.." && pwd)/.stub-runtime-ran"
exit 1
'''
_FAKE_UNAME = '''#!/bin/sh
case "${1:-}" in
  ""|-s) echo "$BRAIN_TEST_FAKE_SYSNAME" ;;
  *) exec /usr/bin/uname "$@" ;;
esac
'''
_SITECUSTOMIZE = '''import os
_name = os.environ.get('BRAIN_TEST_FAKE_SYSNAME')
if _name:
    _real = os.uname
    os.uname = lambda: os.uname_result((_name,) + tuple(_real())[1:])
'''


def _manifest_commands():
    """(event, matcher, command) for every command handler in both manifests,
    once each. An mcp_tool handler runs on the MCP host, not here."""
    out = []
    for rel in MANIFESTS:
        with open(os.path.join(ROOT, rel)) as f:
            doc = json.load(f)
        for event, groups in doc['hooks'].items():
            for group in groups:
                for handler in group['hooks']:
                    if handler.get('type') != 'command':
                        continue
                    key = (event, group.get('matcher', '') or '', handler['command'])
                    if key not in out:
                        out.append(key)
    return out


def _payload(event, matcher, cwd, source='startup'):
    """The stdin a host hands the command — shaped so each hook reaches the
    daemon-contact path when the rule lets it (a .py edit passes
    pre_edit_suggest's pre-screen)."""
    data = {'session_id': SID, 'cwd': cwd, 'hook_event_name': event,
            'transcript_path': os.path.join(cwd, SID + '.jsonl')}
    tool = matcher.split('|')[0]
    if tool.startswith('mcp__'):
        tool = 'mcp__brain__recall'
    if event == 'SessionStart':
        data['source'] = source
    elif event == 'UserPromptSubmit':
        data['prompt'] = 'what did we decide about the gate?'
    elif event in ('PreToolUse', 'PostToolUse'):
        data['tool_name'] = tool
        data['tool_input'] = {
            'Bash': {'command': 'ls'},
            'Edit': {'file_path': os.path.join(cwd, 'a.py'), 'old_string': 'a', 'new_string': 'b'},
        }.get(tool, {'query': 'the gate'})
        if event == 'PostToolUse':
            data['tool_response'] = {}
    elif event == 'Stop':
        data['last_assistant_message'] = 'Done.'
        data['stop_hook_active'] = False
    elif event == 'SessionEnd':
        data['reason'] = 'exit'
    elif event == 'WorktreeCreate':
        data['name'] = 'feature'
    elif event == 'WorktreeRemove':
        data['worktree_path'] = os.path.join(cwd, 'wt')
    return data


class _Listener:
    """Stands where the daemon would: counts accepts, answers one line."""

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(('127.0.0.1', 0))
        self._sock.listen(16)
        self._sock.settimeout(0.05)
        self.port = self._sock.getsockname()[1]
        self.accepts = 0
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            self.accepts += 1
            try:
                conn.settimeout(1)
                conn.recv(1 << 20)
                conn.sendall(b'{"ok": true, "result": {}}\n')
            except OSError:
                pass
            finally:
                conn.close()

    def accepted_since(self, before, wait_s):
        """True once an accept beyond `before` is counted, within wait_s."""
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self.accepts > before:
                return True
            time.sleep(0.02)
        return self.accepts > before

    def close(self):
        self._stop = True
        self._thread.join(timeout=2)
        self._sock.close()


# Where Apple's /usr/bin/python3 keeps bytecode instead of __pycache__ — the
# hooks' `python3` when no venv leads PATH, as in the release suite.
_APPLE_PYC_CACHE = ('Library', 'Caches', 'com.apple.python')


def _tree(*dirs):
    """Every file and directory under `dirs` (not following symlinks) — what a
    hook created shows up as the difference. Python's own bytecode cache is
    not brain state and is left out: __pycache__, and Apple's cache with the
    directories leading to it (anything else under them still shows)."""
    out = set()
    for top in dirs:
        for dirpath, dirnames, filenames in os.walk(top):
            rel = () if dirpath == top else tuple(os.path.relpath(dirpath, top).split(os.sep))
            on_cache_path = _APPLE_PYC_CACHE[:len(rel)] == rel and len(rel) < len(_APPLE_PYC_CACHE)
            dirnames[:] = [d for d in dirnames if d != '__pycache__'
                           and rel + (d,) != _APPLE_PYC_CACHE]
            out.update(os.path.join(dirpath, d) + os.sep for d in dirnames
                       if not (on_cache_path and rel + (d,) == _APPLE_PYC_CACHE[:len(rel) + 1]))
            out.update(os.path.join(dirpath, f) for f in filenames)
    return out


class _Machine:
    """A throwaway hook machine: plugin root, HOME, tmp, cwd, and the env a
    host would give the command."""

    def __init__(self, port, entrypoint_env, fake_bin=None):
        self.tmp = tempfile.mkdtemp(prefix='remote-hook-')
        self.root = os.path.join(self.tmp, 'plugin')
        self.home = os.path.join(self.tmp, 'home')
        self.tmpdir = os.path.join(self.tmp, 'tmp')
        self.work = os.path.join(self.tmp, 'work')
        shutil.copytree(os.path.join(ROOT, 'hooks'), os.path.join(self.root, 'hooks'),
                        ignore=shutil.ignore_patterns('__pycache__'))
        os.symlink(os.path.join(ROOT, 'servers'), os.path.join(self.root, 'servers'))
        self._write(os.path.join(self.root, 'hooks', 'scripts', 'ensure-runtime.sh'),
                    _STUB_RUNTIME, 0o755)
        self._write(os.path.join(self.root, 'sitecustomize.py'), _SITECUSTOMIZE)
        for d in (self.home, self.tmpdir, self.work):
            os.makedirs(d)
        path = os.environ['PATH']
        self.env = {
            'HOME': self.home, 'TMPDIR': self.tmpdir, 'BRAIN_TMP_DIR': self.tmpdir,
            'LANG': os.environ.get('LANG', 'en_US.UTF-8'),
            'BRAIN_DAEMON_PORT': str(port),
            'CLAUDE_PLUGIN_ROOT': self.root,
            'CLAUDE_PLUGIN_DATA': os.path.join(self.home, '.claude', 'plugins', 'data', 'entity-synced'),
            'CLAUDE_CODE_SESSION_ID': SID,
            'PYTHONPATH': self.root,
        }
        self.env.update(entrypoint_env)
        if fake_bin:
            path = fake_bin + os.pathsep + path
            self.env['BRAIN_TEST_FAKE_SYSNAME'] = 'Linux'
        self.env['PATH'] = path

    @staticmethod
    def _write(path, text, mode=None):
        with open(path, 'w') as f:
            f.write(text)
        if mode is not None:
            os.chmod(path, mode)

    def run(self, command, payload):
        t0 = time.monotonic()
        r = subprocess.run(['/bin/sh', '-c', command], input=json.dumps(payload),
                           capture_output=True, text=True, timeout=60,
                           env=self.env, cwd=self.work)
        return r, time.monotonic() - t0

    def tree(self):
        return _tree(self.home, self.root, self.tmpdir)

    @property
    def runtime_reached(self):
        return os.path.exists(os.path.join(self.root, '.stub-runtime-ran'))

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestRemoteHookMachine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The `uname` stand-in, made once and run once before any timing:
        # macOS scans a newly written executable at its first launch (~160 ms
        # measured, then ~8 ms) — a cost of this harness, not of the hooks.
        cls._bin = tempfile.mkdtemp(prefix='remote-hook-bin-')
        _Machine._write(os.path.join(cls._bin, 'uname'), _FAKE_UNAME, 0o755)
        subprocess.run([os.path.join(cls._bin, 'uname'), '-s'], capture_output=True,
                       env={'BRAIN_TEST_FAKE_SYSNAME': 'Linux'}, check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._bin, ignore_errors=True)

    def setUp(self):
        self.listener = _Listener()
        self.addCleanup(self.listener.close)
        self.commands = _manifest_commands()
        # Guard against a parse change making the loops below vacuous.
        self.assertGreaterEqual(len(self.commands), 12, self.commands)
        self.assertTrue(hc.all_remote_hook_markers(), 'the contract declares no remote marker')

    def _machine(self, entrypoint_env, fake_linux):
        m = _Machine(self.listener.port, entrypoint_env, self._bin if fake_linux else None)
        self.addCleanup(m.close)
        return m

    def test_every_command_is_inert_and_quiet_on_a_remote_machine(self):
        # No BRAIN_INSTANCE and no BRAIN_DB_DIR: a Cowork VM has neither, and
        # without the rule the resolver would walk the ladder here. A fresh
        # machine per command, so one command's leftovers cannot hide the next's.
        for env_name, value in hc.all_remote_hook_markers():
            for event, matcher, command in self.commands:
                sources = ('startup', 'resume') if event == 'SessionStart' else ('startup',)
                for source in sources:
                    with self.subTest(marker=env_name, event=event, matcher=matcher,
                                      command=command, source=source):
                        m = self._machine({env_name: value}, fake_linux=True)
                        want = (_hook_common().REMOTE_HOOK_NOTICE + '\n'
                                if event == 'SessionStart' and source == 'startup' else '')
                        # The budget is judged on the best of up to three runs,
                        # each held to every other assertion: on the dev Mac the
                        # first read of a freshly copied script stalls once
                        # (measured ~950 ms, then ~30 ms on the same machine), a
                        # cost of the harness's new files, not of the hook.
                        took = []
                        while len(took) < 3 and (not took or min(took) >= BUDGET_S):
                            before, accepts = m.tree(), self.listener.accepts
                            r, t = m.run(command, _payload(event, matcher, m.work, source))
                            took.append(t)
                            self.assertEqual(r.returncode, 0, r.stderr)
                            self.assertEqual(r.stdout, want, r.stderr)
                            self.assertEqual(sorted(m.tree() - before), [], 'the hook created files')
                            self.assertFalse(self.listener.accepted_since(accepts, 0.1),
                                             'the hook connected to the daemon port')
                        self.assertLess(min(took), BUDGET_S,
                                        'took %s ms' % [round(t * 1000) for t in took])

    def test_brain_env_alone_still_bootstraps_on_a_remote_machine(self):
        # ./dev sources brain-env.sh directly, not the resolver: a developer on
        # such a machine needs a runtime and no daemon, so the rule must not
        # sit in brain-env.sh (where it once made `./dev pytest` exit 0 unrun).
        for env_name, value in hc.all_remote_hook_markers():
            m = self._machine({env_name: value}, fake_linux=True)
            r, _ = m.run('. "$CLAUDE_PLUGIN_ROOT/hooks/scripts/brain-env.sh"; echo sourced', {})
            self.assertTrue(m.runtime_reached, r.stderr)
            self.assertIn('sourced', r.stdout, r.stderr)

    def _assert_goes_past_the_rule(self, entrypoint_env, fake_linux, label):
        for event, matcher, command in self.commands:
            with self.subTest(case=label, event=event, matcher=matcher, command=command):
                m = self._machine(entrypoint_env, fake_linux)
                accepts = self.listener.accepts
                r, _ = m.run(command, _payload(event, matcher, m.work))
                if 'boot-brain.sh' in command:
                    # Boot went on to its own work (here: the cold-install branch).
                    self.assertTrue(r.stdout.strip(), r.stderr)
                    self.assertNotIn(_hook_common().REMOTE_HOOK_NOTICE, r.stdout)
                elif command.startswith('python3 '):
                    self.assertTrue(self.listener.accepted_since(accepts, 5),
                                    'no daemon contact: %s' % r.stderr)
                else:
                    self.assertTrue(m.runtime_reached,
                                    'the resolver never reached the runtime bootstrap: %s' % r.stderr)

    def test_local_entrypoints_go_past_the_rule(self):
        # BRAIN_INSTANCE keeps these runs an entity: no ladder walk onto the
        # operator's brain, no launchd.
        for entrypoint in LOCAL_ENTRYPOINTS:
            self._assert_goes_past_the_rule({'CLAUDE_CODE_ENTRYPOINT': entrypoint,
                                             'BRAIN_INSTANCE': 'remote-hook-test'},
                                            True, entrypoint)

    @unittest.skipUnless(sys.platform == 'darwin', 'uses the real uname: needs a Mac')
    def test_a_mac_is_never_a_remote_machine(self):
        # The marker alone does not make the machine remote: a host that runs
        # its hooks on the Mac under the same marker keeps working.
        for env_name, value in hc.all_remote_hook_markers():
            self._assert_goes_past_the_rule({env_name: value, 'BRAIN_INSTANCE': 'remote-hook-test'},
                                            False, 'darwin %s=%s' % (env_name, value))


class TestRule(unittest.TestCase):
    def _unreachable(self, env, sysname):
        real = os.uname()
        fake = os.uname_result((sysname,) + tuple(real)[1:])
        with mock.patch.dict(os.environ, env, clear=True), mock.patch('os.uname', return_value=fake):
            return _hook_common().daemon_unreachable_here()

    def test_marker_on_linux_is_unreachable(self):
        for env_name, value in hc.all_remote_hook_markers():
            self.assertTrue(self._unreachable({env_name: value}, 'Linux'))

    def test_darwin_never(self):
        for env_name, value in hc.all_remote_hook_markers():
            self.assertFalse(self._unreachable({env_name: value}, 'Darwin'))

    def test_other_entrypoints_and_no_marker(self):
        for ep in LOCAL_ENTRYPOINTS:
            self.assertFalse(self._unreachable({'CLAUDE_CODE_ENTRYPOINT': ep}, 'Linux'))
        self.assertFalse(self._unreachable({}, 'Linux'))

    def test_exit_is_silent_success(self):
        hook_common = _hook_common()
        with mock.patch.object(hook_common, 'daemon_unreachable_here', return_value=True):
            for helper in (hook_common.daemon_available,
                           lambda: hook_common.daemon_call('ping'),
                           lambda: hook_common.daemon_call_raw('ping')):
                with self.assertRaises(SystemExit) as cm:
                    helper()
                self.assertEqual(cm.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
