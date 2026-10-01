"""migrate-from-brain.sh in a sandbox: the inventory it prints and the
refusals it raises, for the install shapes old-plugin users have.

Dry runs only, plus --apply and --finish runs that must refuse: HOME here is
not the account's home, and the script refuses every change under a borrowed
HOME (launchd and the daemon's files belong to the account). If that refusal
ever broke, nothing here could reach the real machine anyway: launchctl and
claude are stubs on PATH, and the daemon's port, pid file and maintenance
lock point into the sandbox. The full --apply/--finish path is rehearsed on a
separate account (see the script's header).
"""

import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SCRIPT = os.path.join(REPO, 'migrate-from-brain.sh')

CLAUDE_STUB = '''#!/bin/bash
echo "$*" >> '%(log)s'
case "$*" in
  --version) echo "2.1.281 (Claude Code)" ;;
  "plugin uninstall --help") echo "%(help)s" ;;
esac
exit 0
'''


def _free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestMigrateFromBrain(unittest.TestCase):

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix='brain-migrate-home-')
        self.bin = os.path.join(self.home, 'bin')
        os.makedirs(self.bin)
        self.claude_log = os.path.join(self.home, 'claude.log')
        self.launchctl_log = os.path.join(self.home, 'launchctl.log')
        self._stub_claude('  --keep-data  Preserve the plugin data directory')
        with open(os.path.join(self.bin, 'launchctl'), 'w') as f:
            f.write('#!/bin/bash\necho "$*" >> \'%s\'\n'
                    '[ "$1" = print ] && exit 1\nexit 0\n' % self.launchctl_log)
        os.chmod(os.path.join(self.bin, 'launchctl'), 0o755)
        self.plugins = os.path.join(self.home, '.claude', 'plugins')
        os.makedirs(self.plugins)
        self.conf = os.path.join(self.home, '.config', 'brain')
        os.makedirs(self.conf)
        self.xdg = os.path.join(self.home, '.local', 'share', 'brain')
        self.registry = {}
        self.markets = {}
        # an entity source git can reach without the network
        self.source = os.path.join(self.home, 'entity.git')
        subprocess.run(['git', 'init', '-q', '--bare', self.source], check=True)
        self.lock = os.path.join(self.home, 'm.lock')

    def tearDown(self):
        subprocess.run(['rm', '-rf', self.home])

    # ── sandbox builders ────────────────────────────────────────────────────
    def _stub_claude(self, uninstall_help):
        path = os.path.join(self.bin, 'claude')
        with open(path, 'w') as f:
            f.write(CLAUDE_STUB % {'log': self.claude_log, 'help': uninstall_help})
        os.chmod(path, 0o755)

    def tree(self, rel, name):
        """A plugin tree under the plugins dir that declares `name`."""
        path = os.path.join(self.plugins, rel)
        os.makedirs(os.path.join(path, '.claude-plugin'), exist_ok=True)
        with open(os.path.join(path, '.claude-plugin', 'plugin.json'), 'w') as f:
            json.dump({'name': name, 'version': '9.6.0'}, f)
        return path

    def install(self, pid, tree_name='brain', scope='user', project=''):
        plugin, market = pid.split('@')
        entry = {'scope': scope,
                 'installPath': self.tree('cache/%s/%s/9.6.0' % (market, plugin), tree_name),
                 'version': '9.6.0'}
        if project:
            entry['projectPath'] = project
        self.registry.setdefault(pid, []).append(entry)
        self.markets[market] = {'source': {'source': 'github', 'repo': 'someone/%s' % market}}

    def brain(self, path, rows=3, learned=None):
        """A brain.db with `rows` nodes. With `learned`, the nodes carry
        encoding_source: `rows` from the starting pack, `learned` made since."""
        os.makedirs(path, exist_ok=True)
        conn = sqlite3.connect(os.path.join(path, 'brain.db'))
        if learned is None:
            conn.execute('CREATE TABLE nodes (id TEXT)')
            conn.executemany('INSERT INTO nodes VALUES (?)', [(str(i),) for i in range(rows)])
        else:
            conn.execute('CREATE TABLE nodes (id TEXT, encoding_source TEXT)')
            conn.executemany('INSERT INTO nodes VALUES (?, ?)',
                             [('s%d' % i, 'anchor:seed') for i in range(rows)]
                             + [('l%d' % i, 'anchor') for i in range(learned)])
        conn.commit()
        conn.close()
        return path

    def record(self, name, brain_dir):
        """A shell-grammar config file naming a brain (resolved.env / env)."""
        with open(os.path.join(self.conf, name), 'a') as f:
            f.write("BRAIN_DB_DIR='%s'\n" % brain_dir)

    def run_script(self, *args, port_env=True):
        with open(os.path.join(self.plugins, 'installed_plugins.json'), 'w') as f:
            json.dump({'version': 2, 'plugins': self.registry}, f)
        with open(os.path.join(self.plugins, 'known_marketplaces.json'), 'w') as f:
            json.dump(self.markets, f)
        env = {
            'HOME': self.home,
            'PATH': os.pathsep.join([self.bin, os.path.dirname(sys.executable),
                                     '/usr/bin', '/bin', '/usr/sbin', '/sbin']),
            'TMPDIR': tempfile.gettempdir(),
            'LANG': 'en_US.UTF-8',
            'BRAIN_DAEMON_PIDFILE': os.path.join(self.home, 'd.pid'),
            'BRAIN_MAINTENANCE_LOCK': self.lock,
            'ENTITY_SOURCE': 'file://' + self.source,
        }
        if port_env:
            env['BRAIN_DAEMON_PORT'] = str(_free_port())
        r = subprocess.run(['bash', SCRIPT] + list(args), env=env,
                           capture_output=True, text=True, timeout=120)
        return r, r.stdout + r.stderr

    def parked(self):
        """The 9.6.0 shape: brain@brain with its brain in the plugin data dir."""
        self.install('brain@brain')
        d = self.brain(os.path.join(self.plugins, 'data', 'brain-brain', 'brain'))
        self.record('resolved.env', d)
        return os.path.realpath(d)

    def claude_calls(self):
        try:
            with open(self.claude_log) as f:
                return [l.strip() for l in f if l.strip()]
        except FileNotFoundError:
            return []

    # ── the inventory and plan ──────────────────────────────────────────────
    def test_parked_install_plans_keep_data_uninstall_before_marketplace_remove(self):
        d = self.parked()
        r, out = self.run_script()
        self.assertIn('brain@brain  scope=user', out)
        self.assertIn('live brain (*): ' + d, out)
        self.assertIn('parked', out)
        steps = [l.strip() for l in out.splitlines() if l.strip()[:2].rstrip('.').isdigit()]
        uninstall = next(i for i, s in enumerate(steps)
                         if 'claude plugin uninstall brain@brain --keep-data --scope user' in s)
        remove = next(i for i, s in enumerate(steps)
                      if 'claude plugin marketplace remove brain' in s)
        self.assertLess(uninstall, remove, 'the marketplace must go only after the keep-data uninstall')
        self.assertIn("write BRAIN_DB_DIR='%s'" % d, out)
        self.assertIn('claude plugin marketplace add file://' + self.source, out)
        self.assertIn('claude plugin install entity@anchor', out)
        # a dry run changes nothing and asks claude nothing but its help
        self.assertEqual(self.claude_calls(), ['--version', 'plugin uninstall --help'])
        self.assertFalse(os.path.exists(os.path.join(self.conf, 'env')))
        self.assertFalse([p for p in os.listdir(self.home) if p.startswith('brain-migration-')])

    def test_dev_install_with_empty_project_path_is_left_alone(self):
        # user scope carries no projectPath; that empty field must not shift
        # the tree name into another column, or this dev install would be
        # planned for uninstall with its marketplace removed
        self.install('brain@local-desktop-app-uploads', tree_name='entity')
        self.brain(os.path.join(self.home, 'elsewhere', 'brain'))
        r, out = self.run_script()
        self.assertIn("left alone: brain@local-desktop-app-uploads (user): its tree declares 'entity'", out)
        self.assertNotIn('claude plugin uninstall', out)
        self.assertNotIn('marketplace remove', out)
        self.assertIn('a development install', out)
        self.assertNotEqual(r.returncode, 0)

    def test_renamed_catalog_offer_goes_and_a_foreign_plugin_keeps_the_marketplace(self):
        self.parked()
        self.install('entity@brain', tree_name='entity')
        self.install('other@brain', tree_name='other')
        r, out = self.run_script()
        self.assertIn('claude plugin uninstall entity@brain --keep-data --scope user', out)
        self.assertIn('marketplace brain: kept, other plugins are installed from it', out)
        self.assertNotIn('claude plugin marketplace remove brain', out)

    def test_project_scope_uninstalls_from_its_project(self):
        project = os.path.join(self.home, 'proj')
        os.makedirs(project)
        self.install('brain@brain', scope='project', project=project)
        self.record('resolved.env', self.brain(self.xdg))
        r, out = self.run_script()
        self.assertIn('claude plugin uninstall brain@brain --keep-data --scope project   (from %s)'
                      % project, out)

    def test_disagreeing_records_are_ambiguous_until_named(self):
        self.install('brain@brain')
        a = self.brain(os.path.join(self.plugins, 'data', 'brain-brain', 'brain'))
        b = self.brain(os.path.join(self.home, 'elsewhere', 'brain'))
        self.record('resolved.env', a)
        self.record('env', b)
        r, out = self.run_script()
        self.assertIn('which brain is live is ambiguous', out)
        r, out = self.run_script('--brain', a)
        self.assertNotIn('ambiguous', out)
        self.assertIn('live brain (*): ' + os.path.realpath(a), out)

    def test_second_brain_at_the_standard_location_refuses(self):
        self.parked()
        self.brain(self.xdg)
        r, out = self.run_script()
        self.assertIn('a second brain sits at the standard location', out)
        self.assertNotEqual(r.returncode, 0)

    def test_old_plugin_records_without_a_brain_stop_not_start_fresh(self):
        # a plain uninstall deleted the parked brain: the marketplace and the
        # record remain, and they prove a brain existed
        self.markets['brain'] = {'source': {'source': 'github', 'repo': 'someone/brain'}}
        gone = os.path.join(self.plugins, 'data', 'brain-brain', 'brain')
        self.record('resolved.env', gone)
        r, out = self.run_script()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('a brain was in use on this machine, but none was found', out)
        self.assertIn('names ' + gone, out)
        self.assertIn('the old marketplace "brain" is still declared', out)
        self.assertNotIn('nothing to migrate', out)

    def test_old_install_without_a_brain_stops(self):
        # whoever has the old plugin has a brain; none found is an alarm
        self.install('brain@brain')
        r, out = self.run_script()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('a brain was in use on this machine, but none was found', out)
        self.assertIn('the old plugin is installed: brain@brain (user)', out)

    def test_a_brain_that_learned_nothing_needs_naming(self):
        # a fresh birth — possibly seeded over a lost brain — is not taken
        # for the brain a user had until they name it
        self.install('brain@brain')
        d = self.brain(os.path.join(self.plugins, 'data', 'brain-brain', 'brain'), rows=19, learned=0)
        self.record('resolved.env', d)
        r, out = self.run_script()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('holds only its starting nodes', out)
        r, out = self.run_script('--brain', d)
        self.assertNotIn('holds only its starting nodes', out)

    def test_a_brain_that_learned_is_taken(self):
        self.install('brain@brain')
        d = self.brain(os.path.join(self.plugins, 'data', 'brain-brain', 'brain'), rows=19, learned=1)
        self.record('resolved.env', d)
        r, out = self.run_script()
        self.assertNotIn('holds only its starting nodes', out)
        self.assertIn('1 nodes learned beyond its starting pack', out)

    def test_choosing_the_standard_brain_rewrites_a_knob_naming_another(self):
        # entity reads the knob first: choosing the standard location must
        # repoint it, or entity would serve the other brain
        self.install('brain@brain')
        std = self.brain(self.xdg)
        other = self.brain(os.path.join(self.home, 'brains', 'other'))
        self.record('resolved.env', std)
        self.record('env', other)
        r, out = self.run_script('--brain', std)
        self.assertIn("write BRAIN_DB_DIR='%s'" % os.path.realpath(std), out)

    def test_the_trap_install_goes_even_after_brain_was_removed_by_hand(self):
        # brain@brain uninstalled by hand (--keep-data): entity@brain, the
        # renamed catalog's offer, is still the old catalog's
        self.install('entity@brain', tree_name='entity')
        d = self.brain(os.path.join(self.plugins, 'data', 'brain-brain', 'brain'))
        self.record('resolved.env', d)
        r, out = self.run_script()
        self.assertIn('claude plugin uninstall entity@brain --keep-data --scope user', out)
        self.assertIn('claude plugin marketplace remove brain', out)

    def test_nothing_at_all_is_nothing_to_migrate(self):
        r, out = self.run_script()
        self.assertIn('no old install and no brain found: nothing to migrate', out)
        self.assertNotIn('a brain was in use', out)

    def test_brain_inside_plugin_code_refuses(self):
        self.install('brain@brain')
        d = self.brain(os.path.join(self.plugins, 'cache', 'brain', 'brain', '9.6.0', 'brain'))
        self.record('resolved.env', d)
        r, out = self.run_script()
        self.assertIn("inside a plugin's code folder", out)

    def test_dry_run_inside_a_session_is_not_refused_for_it(self):
        # MIGRATING.md has Claude run step 1 inside a session; only --apply and
        # --finish need every session closed
        self.parked()
        r, out = self.run_script()
        self.assertNotIn('Codex session is running', out)

    def test_rendezvous_copies_match_their_owners(self):
        # the script runs before entity exists and cannot import these, so it
        # carries copies; a change to the owner must fail here, not in the field
        from servers import daemon_config
        with open(SCRIPT) as f:
            text = f.read()
        uid = os.getuid()
        env = {k: v for k, v in os.environ.items()
               if k not in ('BRAIN_INSTANCE', 'BRAIN_DAEMON_PORT')}
        env['XDG_CONFIG_HOME'] = os.path.join(self.home, 'owner-config')
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(daemon_config._resolve_daemon_port(), 47200 + uid % 100)
            os.makedirs(os.path.join(env['XDG_CONFIG_HOME'], 'brain'))
            with open(os.path.join(env['XDG_CONFIG_HOME'], 'brain', 'env'), 'w') as f:
                f.write('BRAIN_DAEMON_PORT=47299\n')
            self.assertEqual(daemon_config._resolve_daemon_port(), 47299,
                             'the owner reads the env file; the script must too')
            self.assertEqual(daemon_config.get_pid_path(), '/tmp/brain-daemon-%d.pid' % uid)
            self.assertEqual(daemon_config.get_maintenance_path(),
                             '/tmp/brain-maintenance-%d.lock' % uid)
        self.assertIn('$((47200 + UID_N % 100))', text)
        self.assertIn('/tmp/brain-daemon-$UID_N.pid', text)
        self.assertIn('/tmp/brain-maintenance-$UID_N.lock', text)
        with open(os.path.join(REPO, 'servers', 'daemon_launch.py')) as f:
            self.assertIn('LAUNCHD_LABEL = "com.brain.daemon"', f.read())
        with open(os.path.join(REPO, 'hooks', 'scripts', 'ensure-dashboard.sh')) as f:
            self.assertIn('LABEL="com.brain.dashboard"', f.read())
        self.assertIn('LABELS="com.brain.daemon com.brain.dashboard"', text)

    def test_daemon_port_from_the_env_file(self):
        port = _free_port()
        with open(os.path.join(self.conf, 'env'), 'w') as f:
            f.write('BRAIN_DAEMON_PORT=%d\n' % port)
        self.parked()
        r, out = self.run_script(port_env=False)
        self.assertIn('daemon on port %d:' % port, out)

    def test_claude_without_keep_data_refuses(self):
        self._stub_claude('  -s, --scope <scope>  Uninstall from scope')
        self.parked()
        r, out = self.run_script()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no 'plugin uninstall --keep-data'", out)

    # ── changes refuse under a borrowed HOME ────────────────────────────────
    def test_apply_refuses_under_a_borrowed_home_and_changes_nothing(self):
        d = self.parked()
        r, out = self.run_script('--apply')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('STOPPED: --apply refuses', out)
        self.assertIn("HOME is not this account's home", out)
        self.assertEqual(self.claude_calls(), ['--version', 'plugin uninstall --help'])
        self.assertFalse(os.path.exists(self.launchctl_log) and
                         'bootout' in open(self.launchctl_log).read())
        self.assertTrue(os.path.exists(os.path.join(d, 'brain.db')))
        self.assertFalse(os.path.exists(self.lock))
        self.assertFalse([p for p in os.listdir(self.home) if p.startswith('brain-migration-')])

    def _stub_ps(self, comm_line):
        """ps on PATH whose process listings hold only comm_line (or nothing);
        per-pid queries still reach the real ps."""
        with open(os.path.join(self.bin, 'ps'), 'w') as f:
            f.write('#!/bin/bash\ncase "$*" in\n'
                    '  "-axo uid=,comm=") %s ;;\n'
                    '  "-axo uid=,command=") ;;\n'
                    '  *) exec /bin/ps "$@" ;;\nesac\nexit 0\n'
                    % ("printf '%%s\\n' '%d %s'" % (os.getuid(), comm_line) if comm_line else ':'))
        os.chmod(os.path.join(self.bin, 'ps'), 0o755)

    def test_a_codex_session_refuses_apply(self):
        # the ChatGPT app runs Codex, which can start a brain daemon too
        self.parked()
        self._stub_ps('/Applications/ChatGPT.app/Contents/Resources/codex')
        r, out = self.run_script('--apply')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Codex session is running', out)
        self.assertIn('ChatGPT app', out)

    def test_no_session_no_session_refusal(self):
        self.parked()
        self._stub_ps('')
        r, out = self.run_script('--apply')
        self.assertNotIn('session is running', out)

    def _pylib(self, inject=''):
        """The script's embedded Python, optionally with `inject` run right
        after the backup copied the source (to model a short copy)."""
        text = open(SCRIPT).read()
        start = text.index('\n', text.index("read -r -d '' PYLIB <<'PY'")) + 1
        body = text[start:text.index('\nPY\n', start)]
        marker = "    d.execute('PRAGMA journal_mode=DELETE')\n"
        self.assertEqual(body.count(marker), 1)
        return body.replace(marker, inject + marker)

    def test_backup_requires_every_tables_rows_with_the_writers_stopped(self):
        src = os.path.join(self.brain(os.path.join(self.home, 'b')), 'brain.db')
        dst = os.path.join(self.home, 'copy.db')

        def backup(pylib, writers):
            if os.path.exists(dst):
                os.remove(dst)
            return subprocess.run([sys.executable, '-c', pylib, 'backup', src, dst, writers],
                                  capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(backup(self._pylib(), 'stopped'), 'ok\t3')
        short = self._pylib("    d.execute('DELETE FROM nodes WHERE rowid = 1'); d.commit()\n")
        self.assertEqual(backup(short, 'stopped'),
                         'table nodes holds 3 rows in the source but 2 in the copy\t2')
        # a live source (the after/ snapshot) may be written meanwhile: only quick_check
        self.assertEqual(backup(short, 'live'), 'ok\t2')

    def test_finish_refuses_under_a_borrowed_home(self):
        self.parked()
        r, out = self.run_script('--finish')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("HOME is not this account's home", out)
        self.assertEqual(self.claude_calls(), [])


if __name__ == '__main__':
    unittest.main()
