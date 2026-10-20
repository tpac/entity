"""Tests for hung-daemon detection + recovery (the consolidated path).

Recovery lives in servers/daemon_client.py — one primitive, recover_daemon(),
called by both the MCP health monitor and the recall hook. The launchd + spawn
MECHANISMS (kickstart, manages, kill_daemon, spawn_detached_daemon) live in
servers/daemon_launch.py — daemon_client and daemon_server both consume them as
public names. These tests lock:
  - liveness (is_daemon_responsive) tells a live daemon from a hung corpse,
  - recover_daemon's guards: responsive no-op, maintenance, cooldown, breaker,
  - _relaunch_daemon prefers launchd kickstart, falls back to kill + spawn,
  - the one hardened spawn primitive — no raw Popen outside daemon_launch,
  - hook_common delegates instead of carrying its own copy.

Pure unit tests — no Brain, no embedder, no real daemon; launchctl + the
relaunch are mocked.

Context: 2026-05-28 host slept mid-request, woke to a daemon holding its port
but servicing nothing. The old liveness check (bare connect / PID-exists) and
the old "restart" (a no-op deferring to launchd, which can't kill a non-exiting
corpse) both missed it.
"""

import os
import sys
import json
import time
import errno
import fcntl
import shutil
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock

from tests.brain_test_base import BrainTestBase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import servers.daemon_client as dc
import servers.daemon_launch as dl

# Every test here drives the recovery and restart ladders. An unpatched
# forced-restart record's evidence would stat the live PID file and probe the
# live port (both keyed by uid, not by brain dir), and the record itself would
# pile up in the per-run brain dir conftest pins. The originals stay reachable
# for the tests that run them against a temp dir.
_real_log_hook_error = dc.log_hook_error
_real_restart_evidence = dc._restart_evidence
_RECORD_GUARDS = [patch.object(dc, "log_hook_error"),
                  patch.object(dc, "_restart_evidence", return_value="evidence")]


def setUpModule():
    for guard in _RECORD_GUARDS:
        guard.start()


def tearDownModule():
    for guard in _RECORD_GUARDS:
        guard.stop()


def write_launchctl_stub(bin_dir, state_path, log_path, loaded=True,
                         bootout_delay=0, on_bootout="", bootout_async=False):
    """Write a launchctl stub that MODELS LOAD STATE, and seed that state.

    `brain_launchd_managed` IS `launchctl print`'s return code, and every
    mutator in launchd-install.sh polls it until it changes — `unload` retries
    five times at one second each before giving up. A stub that answers rc 0
    to everything therefore reads as "still loaded" forever: the caller burns
    the whole budget, warns, and takes its FAILURE branch. Measured: 6.97s per
    unload, and relocate-brain.sh unloads two services.

    So the stub answers `print` from a state file that `bootout` removes and
    `bootstrap` creates, and every invocation is appended to `log_path` so a
    test can assert WHICH services were acted on. `loaded` seeds the starting
    state: True = the service is installed and running (bootout has something
    to do), False = a fresh machine.

    One state file for all labels. The scripts act on one service at a time
    and only ever ask about the one they just touched, so per-label state
    would buy nothing that any current assertion can see.

    `bootout_delay` is the real bootout's wait for the daemon to exit: the
    window in which a concurrent run finds the service still loaded.
    `on_bootout` is shell run inside that window — something another process
    does meanwhile. `bootout_async` is the real launchctl's shape (measured):
    bootout returns at once and the service stays loaded until the process
    exits, `bootout_delay` seconds later.
    """
    if loaded:
        open(state_path, 'w').close()
    elif os.path.exists(state_path):
        os.remove(state_path)
    if bootout_async:
        bootout = '(sleep %s; rm -f "%s") >/dev/null 2>&1 &' % (bootout_delay, state_path)
    else:
        bootout = 'sleep %s; %s rm -f "%s"' % (
            bootout_delay, on_bootout + ";" if on_bootout else "", state_path)
    path = os.path.join(bin_dir, 'launchctl')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\necho "$@" >> "%s"\n'
                'case "$1" in\n'
                '  bootstrap) touch "%s";;\n'
                '  bootout) %s;;\n'
                '  print) [ -f "%s" ] && exit 0; exit 1;;\n'
                'esac\nexit 0\n'
                % (log_path, state_path, bootout, state_path))
    os.chmod(path, 0o755)


def read_launchctl_calls(log_path):
    """What the script asked launchd to do, as [verb, arg...] per call.

    Blank lines are dropped here rather than at each call site, so a reader
    can index `[0]` without guarding — an IndexError raised inside an
    assertion's own expression reports nothing about the behaviour under
    test.
    """
    if not os.path.exists(log_path):
        return []
    return [parts for parts in
            (ln.split() for ln in open(log_path).read().splitlines())
            if parts]


class TestIsDaemonResponsive(unittest.TestCase):
    def test_responsive_when_ping_ok(self):
        with patch.object(dc, "_can_connect", return_value={"ok": True}):
            self.assertTrue(dc.is_daemon_responsive())

    def test_unresponsive_when_no_reply(self):
        # A corpse: connect succeeds but ping never returns ok → empty dict.
        with patch.object(dc, "_can_connect", return_value={}):
            self.assertFalse(dc.is_daemon_responsive())


class TestRecoverDaemon(unittest.TestCase):
    def setUp(self):
        self.state = os.path.join(tempfile.gettempdir(),
                                  "brain-recovery-test-%d.json" % os.getpid())
        self._clear()

    def tearDown(self):
        self._clear()

    def _clear(self):
        if os.path.exists(self.state):
            os.remove(self.state)

    def _seed(self, last_attempt, attempts):
        with open(self.state, "w") as f:
            json.dump({"last_attempt": last_attempt, "attempts": attempts}, f)

    def _read(self):
        with open(self.state) as f:
            return json.load(f)

    def test_responsive_is_noop(self):
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=True), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertTrue(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_responsive_clears_failure_streak(self):
        self._seed(time.time(), 3)
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=True), \
             patch.object(dc, "_relaunch_daemon"):
            dc.recover_daemon()
        self.assertEqual(self._read()["attempts"], 0)

    def test_maintenance_mode_skips(self):
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=True), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_down_and_clean_triggers_restart(self):
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_called_once()
        self.assertEqual(self._read()["attempts"], 1)

    def test_cooldown_blocks_double_restart(self):
        self._seed(time.time(), 1)  # restart issued just now
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_circuit_breaker_opens_at_max(self):
        # Past cooldown, but already at the attempt ceiling within the window.
        self._seed(time.time() - dc._RECOVERY_COOLDOWN_S - 1, dc._RECOVERY_MAX_ATTEMPTS)
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_stale_streak_ages_out_and_restarts(self):
        # Old incident beyond the window → streak resets, recovery resumes.
        self._seed(time.time() - dc._RECOVERY_WINDOW_S - 1, dc._RECOVERY_MAX_ATTEMPTS)
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            self.assertFalse(dc.recover_daemon())
            relaunch.assert_called_once()
        self.assertEqual(self._read()["attempts"], 1)


class TestRecoverDaemonCorpseTest(unittest.TestCase):
    """recover_daemon's kill decision is a CORPSE test, not a liveness test.
    Unresponsive-but-freshly-bound (young PID file) is a daemon warming up;
    unresponsive-with-the-port-held-and-no-PID-file is a restart in flight.
    Kickstarting either SIGKILLs a healthy daemon mid-boot — the double-kill
    that doubled every deploy-restart outage on 2026-08-17/18. Only
    unresponsive + long-past-bind (stale PID file) is a corpse."""

    def setUp(self):
        self.state = os.path.join(tempfile.gettempdir(),
                                  "brain-recovery-corpse-%d.json" % os.getpid())
        if os.path.exists(self.state):
            os.remove(self.state)

    tearDown = setUp

    def _recover(self, age, port_occupied):
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "pid_file_age_s", return_value=age), \
             patch.object(dc, "port_is_occupied", return_value=port_occupied), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            result = dc.recover_daemon()
        return result, relaunch

    def test_young_pid_file_is_booting_not_corpse(self):
        result, relaunch = self._recover(age=3.0, port_occupied=True)
        self.assertFalse(result)
        relaunch.assert_not_called()
        # Declining to kill a booting daemon is NOT a recovery attempt — no
        # cooldown/breaker state is stamped, so a later genuine corpse still
        # gets its full attempt budget.
        self.assertFalse(os.path.exists(self.state))

    def test_missing_pid_file_with_port_held_is_restart_in_flight(self):
        # "Held" includes a stopped daemon's TIME_WAIT connections — after a
        # clean exit they are the only sign of a successor that has not yet
        # taken its lock and claimed the PID file (TestPortProbesOnRealSockets).
        result, relaunch = self._recover(age=None, port_occupied=True)
        self.assertFalse(result)
        relaunch.assert_not_called()

    def test_missing_pid_file_port_free_relaunches(self):
        # No PID file, no recent connections → kickstart. A reload in flight
        # never looks like this: _exec_reload rewrites the PID file just
        # before the exec (the breadcrumb — see
        # test_exec_reload_execs_brain_daemon_launcher), and a fresh start
        # claims it once it holds its lock. Only a start that has not reached
        # its lock yet does (see port_is_occupied).
        result, relaunch = self._recover(age=None, port_occupied=False)
        self.assertFalse(result)
        relaunch.assert_called_once()

    def test_stale_pid_file_is_a_corpse(self):
        result, relaunch = self._recover(
            age=dc._GRACE_DEADLINE_S + 1, port_occupied=True)
        self.assertFalse(result)
        relaunch.assert_called_once()

    def test_a_forced_restart_names_its_attempt_to_the_relaunch(self):
        # _relaunch_daemon records the act (TestRelaunchDaemon); the decision
        # hands it the reason, and records nothing itself.
        dc.log_hook_error.reset_mock()
        with patch.object(dc, "get_recovery_state_path", return_value=self.state), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "is_daemon_responsive", return_value=False), \
             patch.object(dc, "pid_file_age_s", return_value=dc._GRACE_DEADLINE_S + 1), \
             patch.object(dc, "_relaunch_daemon") as relaunch:
            dc.recover_daemon()
        relaunch.assert_called_once_with(None, "recover_daemon: unresponsive, restart attempt 1")
        dc.log_hook_error.assert_not_called()

    def test_declining_to_kill_writes_no_record(self):
        # Every live proxy's monitor passes through here while a daemon boots;
        # a record per decline would fan out the way DAEMON_DOWN does.
        dc.log_hook_error.reset_mock()
        self._recover(age=3.0, port_occupied=True)
        self._recover(age=None, port_occupied=True)
        dc.log_hook_error.assert_not_called()

    def test_pid_file_age_reads_bind_recency(self):
        # daemon_launch.pid_file_age_s: mtime of the PID file, None if absent.
        with tempfile.TemporaryDirectory(prefix="brain-pidage-test-") as tmp:
            pid_path = os.path.join(tmp, "daemon.pid")
            with patch.object(dl, "get_pid_path", return_value=pid_path):
                self.assertIsNone(dl.pid_file_age_s())
                with open(pid_path, "w") as f:
                    f.write("12345")
                age = dl.pid_file_age_s()
                self.assertIsNotNone(age)
                self.assertLess(age, 5.0)


class TestRelaunchDaemon(unittest.TestCase):
    def test_kickstart_success_skips_fallback(self):
        dc.log_hook_error.reset_mock()
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=0)) as run, \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon") as ensure:
            dc._relaunch_daemon(None)
            self.assertEqual(dc.log_hook_error.call_args.args,
                             ("DAEMON_RESTART_FORCED", "recover_daemon: kickstart -k"))
            run.assert_called_once()
            argv = run.call_args[0][0]
            self.assertEqual(argv[:3], ["launchctl", "kickstart", "-k"])
            self.assertIn("com.brain.daemon", argv[3])
            kill.assert_not_called()
            ensure.assert_not_called()

    def test_kickstart_failure_falls_back_to_kill_and_spawn(self):
        # kickstart fails AND we ARE the daemon's source → own the kill + respawn.
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon", return_value=True) as ensure:
            order = []
            dc.log_hook_error.reset_mock()
            dc.log_hook_error.side_effect = lambda *a, **k: order.append(a[1])
            kill.side_effect = lambda: order.append("kill")
            try:
                dc._relaunch_daemon("/tmp/x/brain.db", "why")
            finally:
                dc.log_hook_error.side_effect = None
            self.assertEqual(order, ["why: kill + respawn", "kill"])   # recorded, then killed
            ensure.assert_called_once_with("/tmp/x/brain.db")

    def test_kickstart_failed_but_incumbent_responsive_defers_never_kills(self):
        # Step 3 drift fix. recover_daemon's single 2s ping can call a slow/busy
        # daemon "down"; if kickstart then also fails transiently, the old code
        # went straight to SIGKILL — killing a live daemon mid-request that
        # ensure_daemon's ladder would have deferred to. A daemon that answers
        # the re-ping must be deferred to, never killed.
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon") as ensure:
            dc.log_hook_error.reset_mock()
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_not_called()
            ensure.assert_not_called()
            dc.log_hook_error.assert_not_called()   # deferred: nothing stopped, nothing recorded

    def test_kickstart_failure_from_non_source_defers_never_kills(self):
        # A worktree / 2nd clone must NEVER SIGKILL the shared daemon: kickstart
        # failed AND we are not the source → defer to launchd/source, don't kill.
        with patch.object(dl.subprocess, "run", return_value=MagicMock(returncode=1)), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "ensure_daemon") as ensure:
            dc._relaunch_daemon("/tmp/x/brain.db")
            kill.assert_not_called()
            ensure.assert_not_called()


class TestLaunchdHelpersDegradeWhenAbsent(unittest.TestCase):
    """The launchd helpers must return False — not raise — when the `launchctl`
    binary is absent (i.e. on Linux). The no-launchd direct-spawn fallback
    (test_no_launchd_falls_back_to_direct_spawn) RELIES on this: it mocks both
    helpers to False but never proves they actually degrade that way when
    launchctl is missing. Narrowing the `except Exception` in either helper to a
    subprocess-specific error would let FileNotFoundError propagate, break the
    Linux daemon path, and slip past every other test. This locks the contract.
    """

    def test_kickstart_returns_false_when_launchctl_missing(self):
        with patch.object(dl.subprocess, "run",
                          side_effect=FileNotFoundError("launchctl")):
            self.assertFalse(dl.kickstart())

    def test_manages_returns_false_when_launchctl_missing(self):
        with patch.object(dl.subprocess, "run",
                          side_effect=FileNotFoundError("launchctl")):
            self.assertFalse(dl.manages())


class TestManagesDaemonTransientVsAbsent(unittest.TestCase):
    """manages()==False AUTHORIZES the direct-spawn fallback that
    can orphan a competing daemon (incident 2026-07-03: a transient `launchctl
    print` failure was read as 'launchd absent', spawned an orphan that squatted
    the lock, and launchd's KeepAlive stormed 135k exits). False must mean
    DEFINITIVELY-not-managed; a launchctl that merely failed to answer is
    INDETERMINATE and must defer (True), never spawn a rival.
    """

    def _ok(self, returncode, stderr=""):
        return MagicMock(returncode=returncode, stdout="", stderr=stderr)

    def test_true_when_running(self):
        with patch.object(dl.subprocess, "run", return_value=self._ok(0)):
            self.assertTrue(dl.manages())

    def test_false_on_clean_not_found(self):
        # rc 113 + "Could not find service …" is a definitive absence.
        stderr = 'Could not find service "com.brain.daemon" in domain for user gui: 503'
        with patch.object(dl.subprocess, "run", return_value=self._ok(113, stderr)):
            self.assertFalse(dl.manages())

    def test_true_on_timeout(self):
        # launchctl present but hung → indeterminate → defer, do NOT spawn.
        with patch.object(dl.subprocess, "run",
                          side_effect=dl.subprocess.TimeoutExpired(
                              cmd=["launchctl"], timeout=10)):
            self.assertTrue(dl.manages())

    def test_true_on_unexpected_nonzero(self):
        # non-zero WITHOUT the not-found signature (e.g. gui/<uid> unaddressable)
        # is indeterminate, not absence.
        with patch.object(dl.subprocess, "run",
                          return_value=self._ok(1, "Bootstrap operation not permitted")):
            self.assertTrue(dl.manages())

    def test_retries_transient_then_reads_answer(self):
        # A one-off timeout must not decide the outcome — the retry sees rc 0.
        with patch.object(dl.subprocess, "run",
                          side_effect=[dl.subprocess.TimeoutExpired(cmd=["launchctl"], timeout=10),
                                       self._ok(0)]) as run:
            self.assertTrue(dl.manages())
            self.assertEqual(run.call_count, 2)


class TestHookDelegation(unittest.TestCase):
    """hook_common must delegate, not carry its own recovery copy."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts'))
        import hook_common
        cls.hc = hook_common

    def test_daemon_available_delegates_to_responsive(self):
        with patch("servers.daemon_client.is_daemon_responsive", return_value=True):
            self.assertTrue(self.hc.daemon_available())
        with patch("servers.daemon_client.is_daemon_responsive", return_value=False):
            self.assertFalse(self.hc.daemon_available())

    def test_unavailable_error_calls_recover_daemon(self):
        # Existing brain.db + dead daemon = a crash: recovery must fire.
        with tempfile.TemporaryDirectory(prefix="brain-hook-deleg-") as tmp:
            db = os.path.join(tmp, "brain.db")
            open(db, "w").close()
            with patch.object(self.hc, "db_dir", tmp), \
                 patch.object(self.hc, "db_path", db), \
                 patch("servers.daemon_client.recover_daemon") as recover:
                msg = self.hc.daemon_unavailable_error("recall")
                recover.assert_called_once_with(db)
                self.assertIn("CRITICAL", msg)

    def test_unconfigured_install_exits_without_recovery(self):
        # No brain.db AND no daemon = the brain never came up, not a crash.
        # BRAIN OFFLINE contract (b2aaa1f): log + exit 1, NO recovery attempt —
        # boot creates brains, recovery must not.
        with tempfile.TemporaryDirectory(prefix="brain-hook-deleg-") as tmp:
            with patch.object(self.hc, "db_dir", tmp), \
                 patch.object(self.hc, "db_path", os.path.join(tmp, "brain.db")), \
                 patch("servers.daemon_client.recover_daemon") as recover:
                with self.assertRaises(SystemExit) as cm:
                    self.hc.daemon_unavailable_error("recall")
                self.assertEqual(cm.exception.code, 1)
                recover.assert_not_called()

    def test_hook_common_has_no_local_recovery(self):
        # The duplicate must be gone — recovery lives only in daemon_client.
        self.assertFalse(hasattr(self.hc, "force_restart_daemon"))


class TestEnsureDaemonRoutesThroughLaunchd(unittest.TestCase):
    """ensure_daemon must route every (re)start through launchd (kickstart),
    serialized under the STARTUP lock (never the daemon's own singleton flock,
    which a serving daemon holds for life) — and never Popen alongside KeepAlive.

    Regression guard for the Errno-48 storm (2026-06-04): N concurrent boots
    each saw stale code and independently killed + respawned while launchd's
    KeepAlive also respawned, so several processes raced to bind the port.
    """

    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="brain-ensure-test-")
        self._db = os.path.join(self._dir, "brain.db")
        self._lock = os.path.join(self._dir, "daemon.lock")
        # No ensure_daemon test may ever touch the real wire: the reload
        # branch sends a real `restart` command if send_command is unpatched
        # — against the LIVE daemon on the real port. Class-level autopatch;
        # tests that assert on it re-patch with their own mock.
        self._sc_autopatch = patch.object(
            dc, "send_command", return_value={"ok": True})
        self._sc_mock = self._sc_autopatch.start()
        self.addCleanup(self._sc_autopatch.stop)

    def tearDown(self):
        for p in (self._lock, self._lock + ".startup",
                  os.path.join(self._dir, "daemon.log"), self._db):
            if os.path.exists(p):
                os.remove(p)
        try:
            os.rmdir(self._dir)
        except OSError:
            pass

    def test_healthy_current_code_is_noop(self):
        # Responsive + same code → return True without touching the lifecycle.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_stale_code_reloads_in_place_never_kickstarts(self):
        # Responsive but stale → reload (the daemon execs fresh code, same
        # PID), never a kill. kickstart is for corpses; SIGKILLing a healthy
        # daemon for a code refresh was the deploy-outage class of 2026-08.
        # _code_changed: fast(stale) → recheck(stale) → reload-branch(stale)
        # → post-reload poll(current).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", side_effect=[True, True, True, False]), \
             patch.object(dc, "send_command", return_value={"ok": True}) as sc, \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            sc.assert_called_once_with("restart", timeout=5.0)
            ks.assert_not_called()      # healthy + stale is reload's, not kickstart's
            spawn.assert_not_called()

    def test_reload_poll_converges_on_fingerprint_not_liveness(self):
        # The OLD daemon keeps answering pings for a tick or two after
        # accepting the restart. A poll that returns on bare liveness reads
        # the old image as success and reports ready mid-teardown — the exact
        # connection-error class this design exists to remove. Sequence:
        # poll 1 hits the old daemon (responsive, STALE fingerprint) and must
        # NOT satisfy the poll; poll 2 hits the successor (current) and does.
        # _code_changed: fast, recheck, branch, poll1(stale), poll2(current).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed",
                          side_effect=[True, True, True, True, False]), \
             patch.object(dc, "send_command", return_value={"ok": True}) as sc, \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep") as slp:
            self.assertTrue(dc.ensure_daemon(self._db))
            sc.assert_called_once()
            self.assertEqual(slp.call_count, 2,
                             "poll must iterate past the old daemon's "
                             "responsive-but-stale tick, not return on it")
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_syncing_reply_returns_ready_without_kickstart(self):
        # A daemon whose runtime is behind uv.lock answers `restart` with
        # `syncing`: it keeps serving and swaps itself when the sync ends.
        # Polling the grace budget out and then kickstarting would kill it,
        # and uv with it, mid-sync (node c34a86e0).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=True), \
             patch.object(dc, "send_command",
                          return_value={"ok": True, "result": {"status": "syncing"}}) as sc, \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep") as slp:
            self.assertTrue(dc.ensure_daemon(self._db))
            sc.assert_called_once_with("restart", timeout=5.0)
            slp.assert_not_called()     # no convergence poll to run out
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_stale_reload_not_converging_falls_back_to_kickstart(self):
        # A reload that never comes back on current code within the grace
        # budget IS the corpse case — escalate to kickstart exactly once.
        # The poll loop genuinely runs here (two iterations, both stale)
        # before the fake clock crosses the deadline; the PID file reads old
        # so the booting-successor guard doesn't defer.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed",
                          side_effect=[True, True, True, True, True, False]), \
             patch.object(dc, "send_command", return_value={"ok": True}) as sc, \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc, "_await_responsive", return_value={"ok": True}), \
             patch.object(dc.time, "monotonic",
                          side_effect=[0.0, 1.0, 2.0, 25.0, 25.0, 25.0]), \
             patch.object(dc, "kickstart", return_value=True) as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep") as slp:
            self.assertTrue(dc.ensure_daemon(self._db))
            sc.assert_called_once()     # the reload was tried first
            self.assertEqual(slp.call_count, 2)  # the loop actually polled
            ks.assert_called_once()     # then escalated — once, under the lock
            spawn.assert_not_called()

    def test_reload_non_convergence_defers_to_a_booting_successor(self):
        # Non-convergence while a successor is mid-boot (a degraded reload
        # that fell back to exit + KeepAlive). Kickstarting it now would
        # SIGKILL it mid-boot and reset the throttle — defer instead; the next
        # caller re-evaluates. The sign is a YOUNG PID file, or, once a clean
        # exit unlinked it, the port held by the stopped daemon's TIME_WAIT
        # connections (TestPortProbesOnRealSockets).
        for age, held in ((3.0, False), (None, True)):
            with self.subTest(pid_file_age=age), \
                 patch.object(dc, "is_maintenance_mode", return_value=False), \
                 patch.object(dc, "get_lock_path", return_value=self._lock), \
                 patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
                 patch.object(dc, "_is_daemon_source", return_value=True), \
                 patch.object(dc, "_can_connect", return_value={"ok": True}), \
                 patch.object(dc, "_code_changed", return_value=True), \
                 patch.object(dc, "send_command", return_value={"ok": True}), \
                 patch.object(dc, "pid_file_age_s", return_value=age), \
                 patch.object(dc, "port_is_occupied", return_value=held), \
                 patch.object(dc.time, "monotonic", side_effect=[0.0, 999.0, 999.0]), \
                 patch.object(dc, "kickstart") as ks, \
                 patch.object(dc, "spawn_detached_daemon") as spawn, \
                 patch.object(dc.time, "sleep"):
                self.assertFalse(dc.ensure_daemon(self._db))
                ks.assert_not_called()
                spawn.assert_not_called()

    def test_reload_works_while_daemon_singleton_flock_is_held(self):
        # THE production shape (2026-08-19 root cause): a serving daemon
        # holds get_lock_path()'s flock for its entire life. ensure_daemon
        # must neither block on nor acquire that lock to reload — when it
        # serialized on the daemon's own flock, the stale-code branch was
        # dead code (the boot hook hung against a healthy daemon) and any
        # successor booting while the lock was held exited as a duplicate.
        import fcntl as _fcntl
        holder = open(self._lock, "w")
        _fcntl.flock(holder, _fcntl.LOCK_EX)   # the "serving daemon"
        try:
            with patch.object(dc, "is_maintenance_mode", return_value=False), \
                 patch.object(dc, "get_lock_path", return_value=self._lock), \
                 patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
                 patch.object(dc, "_is_daemon_source", return_value=True), \
                 patch.object(dc, "_can_connect", return_value={"ok": True}), \
                 patch.object(dc, "_code_changed",
                              side_effect=[True, True, True, False]), \
                 patch.object(dc, "send_command", return_value={"ok": True}) as sc, \
                 patch.object(dc, "kickstart") as ks, \
                 patch.object(dc, "spawn_detached_daemon") as spawn, \
                 patch.object(dc.time, "sleep"):
                self.assertTrue(dc.ensure_daemon(self._db))
                sc.assert_called_once_with("restart", timeout=5.0)
                ks.assert_not_called()
                spawn.assert_not_called()
        finally:
            _fcntl.flock(holder, _fcntl.LOCK_UN)
            holder.close()

    def test_concurrent_winner_already_restarted_skips_kickstart(self):
        # The anti-storm guarantee: fast-path sees stale and falls through to
        # the lock, but by the time we hold it another caller already restarted
        # to current code → we re-check and do NOTHING (no second kickstart).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", side_effect=[True, False]), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def _spawn_path(self, reping):
        # Daemon down, nothing serving, AND launchd genuinely not managing it
        # (kickstart fails AND manages() False) → the only case a direct spawn
        # is legitimate (no KeepAlive to race). _can_connect calls:
        # fast-path, under-lock recheck, post-kickstart re-ping, post-spawn ready.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect",
                          side_effect=[{"ok": False}, {"ok": False}, reping, {"ok": True}]), \
             patch.object(dc, "_await_responsive", return_value={"ok": False}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "kickstart", return_value=False) as ks, \
             patch.object(dc, "manages", return_value=False), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
        ks.assert_called_once()
        spawn.assert_called_once_with(self._db)
        return kill

    def test_no_launchd_falls_back_to_direct_spawn(self):
        # Refused: nothing listens (a stopped daemon's TIME_WAIT connections
        # refuse too), so there is nothing to kill first. Nothing was killed,
        # so nothing is recorded — the failed kickstart included.
        dc.log_hook_error.reset_mock()
        kill = self._spawn_path({"ok": False, "transport": "refused"})
        kill.assert_not_called()
        dc.log_hook_error.assert_not_called()

    def test_no_launchd_spawn_kills_a_zombie_first(self):
        # Something accepted the re-ping and did not answer: a zombie holds
        # the port, whatever address it bound. Each kill leaves one record.
        for transport in ("timeout", "reset", "empty", "protocol"):
            with self.subTest(transport=transport):
                dc.log_hook_error.reset_mock()
                kill = self._spawn_path({"ok": False, "transport": transport})
                kill.assert_called_once()
                dc.log_hook_error.assert_called_once()
                name, action = dc.log_hook_error.call_args.args
                self.assertEqual(name, "DAEMON_RESTART_FORCED")
                self.assertIn("zombie", action)
                self.assertIn(transport, action)

    def test_a_kickstart_launchd_accepted_is_recorded(self):
        # The record names the sender, what it saw, and the session's own brain
        # dir as its destination.
        dc.log_hook_error.reset_mock()
        down = {"ok": False, "transport": "timeout"}
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value=down), \
             patch.object(dc, "_await_responsive", side_effect=[down, {"ok": True}]), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "kickstart", return_value=True):
            self.assertTrue(dc.ensure_daemon(self._db))
        dc.log_hook_error.assert_called_once()
        call = dc.log_hook_error.call_args
        self.assertEqual(call.args, ("DAEMON_RESTART_FORCED",
                                     "ensure_daemon: kickstart -k, daemon unresponsive (timeout)"))
        self.assertTrue(call.kwargs["context"].startswith("evidence; caller "), call.kwargs)
        self.assertEqual(call.kwargs["level"], "error")   # visible to level='error' queries
        self.assertEqual(call.kwargs["db_dir"], self._dir)

    def test_kickstart_killed_incumbent_then_failed_does_not_false_defer(self):
        # `kickstart -k` SIGKILLs the incumbent before respawning. If it then
        # returns nonzero/timed-out, the port is FREE — but the pre-kickstart
        # ping said {ok:True}. The defer branch must RE-PING (now {ok:False}) and
        # NOT defer on the stale snapshot (which would report "ready" with
        # nothing serving). Stale code, so the reload is tried first and never
        # converges (monotonic jump skips its poll); then the kickstart path.
        # _can_connect: fast(ok), under-lock(ok), re-ping(DOWN), post-spawn(ok).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect",
                          side_effect=[{"ok": True}, {"ok": True},
                                       {"ok": False, "transport": "refused"}, {"ok": True}]), \
             patch.object(dc, "_code_changed", return_value=True), \
             patch.object(dc, "send_command", return_value={"ok": True}), \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc.time, "monotonic", side_effect=[0.0] + [999.0] * 9), \
             patch.object(dc, "kickstart", return_value=False), \
             patch.object(dc, "manages", return_value=False), \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))
            spawn.assert_called_once()   # spawned — did NOT false-defer on stale resp

    def test_responsive_stale_kickstart_unreachable_defers_not_spawn(self):
        # Regression guard for the 2026-06-05 orphan storm. The daemon looks
        # stale and stays stale (a reload that never converges — monotonic jump
        # skips its poll), and kickstart can't reach launchd from this context.
        # We MUST defer to the responsive incumbent — never kill it and spawn a
        # competitor (the rival orphaned, squatted the port, and crash-looped
        # the real launchd daemon).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=True), \
             patch.object(dc, "send_command", return_value={"ok": True}), \
             patch.object(dc, "pid_file_age_s", return_value=9999.0), \
             patch.object(dc.time, "monotonic", side_effect=[0.0] + [999.0] * 9), \
             patch.object(dc, "kickstart", return_value=False), \
             patch.object(dc, "kill_daemon") as kill, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            kill.assert_not_called()
            spawn.assert_not_called()

    def test_down_but_launchd_managed_kickstart_failed_defers_to_keepalive(self):
        # Daemon down, kickstart failed (transient), but launchd DOES manage the
        # service → don't race a manual spawn; let KeepAlive bring it up.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": False}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "kickstart", return_value=False), \
             patch.object(dc, "manages", return_value=True), \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertFalse(dc.ensure_daemon(self._db))
            spawn.assert_not_called()

    def test_maintenance_mode_skips_everything(self):
        with patch.object(dc, "is_maintenance_mode", return_value=True), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertFalse(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_worktree_client_up_is_noop(self):
        # A non-source checkout (worktree) is a PURE CLIENT: daemon up → return
        # True without ever touching kickstart/spawn, whatever the code version.
        # It's still Anchor (shared brain.db) — just not the daemon's lifecycle owner.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={"ok": True}), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_worktree_client_down_waits_never_kickstarts(self):
        # A worktree never (re)starts the shared daemon — launchd KeepAlive owns
        # recovery (a worktree restart can't converge its code, it only churns).
        # Down → wait out the relaunch grace, return whether it came up; the grace
        # here recovers it → True, with NO kickstart and NO spawn.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": True}), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_owner_slow_daemon_recovers_within_grace_not_kickstarted(self):
        # THE core fix. The source checkout sees a non-responsive 2s ping, but the
        # daemon is only slow / mid-relaunch. The grace probe catches it coming
        # back → NO kickstart (the false-down that started the restart storm).
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_can_connect", return_value={"ok": False}), \
             patch.object(dc, "_await_responsive", return_value={"ok": True}), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()   # slow ≠ dead → not restarted
            spawn.assert_not_called()


class TestReleaseSupersedes(unittest.TestCase):
    """The owner rule for a plugin update: launchd-install.sh
    _brain_launchd_supersedes picks the launchd plist's tree, and
    daemon_launch.release_supersedes asks that same function for a
    plain-process daemon — so this table, run through the Python door,
    pins both consumers at once."""

    C = "/h/.claude/plugins/cache/anchor/entity"
    X = "/h/.codex/plugins/cache/anchor-dev/entity"
    CASES = [
        (C + "/0.9.1", C + "/0.9.0", True),        # an update beside the old dir
        (C + "/0.9.0", C + "/0.9.1", False),       # never back down
        (C + "/0.9.0", C + "/0.9.0", False),       # the same release
        (C + "/0.10.0", C + "/0.9.9", True),       # numeric, not lexical
        (C + "/0.9.10", C + "/0.9.9", True),
        (C + "/1.0", C + "/0.9.9", True),
        (C + "/0.9.0", C + "/0.9", False),         # missing parts are zeros
        (C + "/0.9.1", C + "/0.9", True),
        (C + "/0.9.08", C + "/0.9.7", True),       # a leading zero is decimal
        (X + "/0.9.1", C + "/0.9.0", True),        # names neither host
        (X + "/0.9.0", C + "/0.9.0", False),       # equal across hosts: no ping-pong
        (X + "/0.9.0", C + "/0.9.1", False),
        (C + "/0.9.1", "/Users/me/brain", False),  # a checkout is the dev pin
        ("/Users/me/brain", C + "/0.9.0", False),
        (C + "/0.9.1", "/h/.claude/plugins/cache/brain/brain/9.7.2", False),  # another plugin
        (C + "/0.9.1-rc1", C + "/0.9.0", False),   # release.sh ships X.Y.Z only
        (C + "/abc", C + "/0.9.0", False),
    ]

    def test_rule_table(self):
        for new, old, want in self.CASES:
            with self.subTest(new=new, old=old):
                self.assertIs(dl.release_supersedes(new, old), want)

    def test_a_failed_ask_never_moves_a_daemon(self):
        dl.release_supersedes.cache_clear()
        self.addCleanup(dl.release_supersedes.cache_clear)
        with patch.object(dl.subprocess, "run", side_effect=OSError("no bash")):
            self.assertIs(dl.release_supersedes(self.C + "/0.9.1", self.C + "/0.9.0"), False)


class TestSupersededTreeTakeover(unittest.TestCase):
    """A plugin update installs a NEW version dir beside the old one. A
    daemon launchd does not manage (Linux; macOS before the LaunchAgent is
    bootstrapped) keeps running the old dir's code unless the new tree's
    session takes it over — the reload would exec the old dir's launcher
    again. Under launchd the plist render owns the move, so the client stays
    out of it."""

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-takeover-")
        self.addCleanup(shutil.rmtree, self._home, True)
        self._cache = os.path.join(self._home, ".claude", "plugins", "cache", "anchor", "entity")
        self._db = os.path.join(self._home, "brain.db")
        self._lock = os.path.join(self._home, "daemon.lock")
        sc = patch.object(dc, "send_command", return_value={"ok": True})
        self._send = sc.start()
        self.addCleanup(sc.stop)

    def _tree(self, version, launcher=True):
        tree = os.path.join(self._cache, version)
        scripts = os.path.join(tree, "hooks", "scripts")
        os.makedirs(scripts)
        if launcher:
            p = os.path.join(scripts, "brain-daemon")
            with open(p, "w") as f:
                f.write("#!/bin/bash\n")
            os.chmod(p, 0o755)
        return tree

    @staticmethod
    def _resp(src, fp="old-fp"):
        return {"ok": True, "result": {"source_dir": src, "code_fingerprint": fp}}

    def test_older_release_unmanaged_is_ours(self):
        old, new = self._tree("0.9.0"), self._tree("0.9.1")
        with patch.object(dc, "REPO_ROOT", new), \
             patch.object(dc, "manages", return_value=False):
            self.assertTrue(dc._is_daemon_source(self._resp(old)))

    def test_older_release_under_launchd_is_not_ours(self):
        old, new = self._tree("0.9.0"), self._tree("0.9.1")
        with patch.object(dc, "REPO_ROOT", new), \
             patch.object(dc, "manages", return_value=True):
            self.assertFalse(dc._is_daemon_source(self._resp(old)))

    def test_vanished_tree_unmanaged_is_ours(self):
        new = self._tree("0.9.1")
        with patch.object(dc, "REPO_ROOT", new), \
             patch.object(dc, "manages", return_value=False):
            self.assertTrue(dc._is_daemon_source(self._resp(os.path.join(self._cache, "0.9.0"))))

    def test_equal_newer_or_unversioned_tree_is_not_ours(self):
        mine = self._tree("0.9.1")
        others = [self._tree("0.9.2"),
                  os.path.join(self._home, ".codex", "plugins", "cache", "anchor-dev", "entity", "0.9.1"),
                  os.path.join(self._home, "brain")]
        for tree in others[1:]:
            os.makedirs(os.path.join(tree, "hooks", "scripts"))
            p = os.path.join(tree, "hooks", "scripts", "brain-daemon")
            with open(p, "w") as f:
                f.write("#!/bin/bash\n")
            os.chmod(p, 0o755)
        with patch.object(dc, "REPO_ROOT", mine), \
             patch.object(dc, "manages", return_value=False) as mg:
            for tree in others:
                with self.subTest(tree=tree):
                    self.assertFalse(dc._is_daemon_source(self._resp(tree)))
            mg.assert_not_called()   # the launchctl fork is asked only when the tree lost

    def test_ensure_daemon_replaces_superseded_daemon_never_reloads_it(self):
        old, new = self._tree("0.9.0"), self._tree("0.9.1")
        # Same fingerprint too: a release that changed only hooks/scripts
        # still leaves the daemon reloading through the old launcher.
        for old_fp in ("old-fp", "new-fp"):
            # _can_connect: fast path (old), under-lock recheck (old),
            # post-kickstart re-ping (down, stopped), post-spawn ready (new).
            seq = [self._resp(old, old_fp), self._resp(old, old_fp),
                   {"ok": False, "transport": "refused"},
                   self._resp(new, fp="new-fp")]
            self._send.reset_mock()
            with self.subTest(old_fp=old_fp), \
                 patch.object(dc, "REPO_ROOT", new), \
                 patch.object(dc, "_CODE_FINGERPRINT", "new-fp"), \
                 patch.object(dc, "is_maintenance_mode", return_value=False), \
                 patch.object(dc, "get_lock_path", return_value=self._lock), \
                 patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
                 patch.object(dc, "_can_connect", side_effect=seq), \
                 patch.object(dc, "manages", return_value=False), \
                 patch.object(dc, "kickstart", return_value=False), \
                 patch.object(dc, "spawn_detached_daemon") as spawn, \
                 patch.object(dc.time, "sleep"):
                self.assertTrue(dc.ensure_daemon(self._db))
                spawn.assert_called_once_with(self._db)
                sent = [c.args[0] for c in self._send.call_args_list]
                self.assertEqual(sent, ["shutdown"],
                                 "graceful stop, and never a reload: it would exec "
                                 "the superseded tree's launcher again")

    def test_superseded_daemon_that_will_not_stop_gets_no_rival(self):
        # Shutdown sent but the singleton flock stays held: spawning now
        # would put a second writer on brain.db.
        old, new = self._tree("0.9.0"), self._tree("0.9.1")
        held = open(self._lock, "w")
        self.addCleanup(held.close)
        fcntl.flock(held, fcntl.LOCK_EX)
        with patch.object(dc, "REPO_ROOT", new), \
             patch.object(dc, "_CODE_FINGERPRINT", "new-fp"), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_can_connect", return_value=self._resp(old)), \
             patch.object(dc, "manages", return_value=False), \
             patch.object(dc, "_GRACE_DEADLINE_S", 0.0), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn, \
             patch.object(dc.time, "sleep"):
            self.assertTrue(dc.ensure_daemon(self._db))   # still serving
            spawn.assert_not_called()
            ks.assert_not_called()

    def test_ensure_daemon_under_launchd_leaves_superseded_daemon_to_the_plist(self):
        old, new = self._tree("0.9.0"), self._tree("0.9.1")
        with patch.object(dc, "REPO_ROOT", new), \
             patch.object(dc, "_CODE_FINGERPRINT", "new-fp"), \
             patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "_can_connect", return_value=self._resp(old)), \
             patch.object(dc, "manages", return_value=True), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()
            self._send.assert_not_called()   # no shutdown, no reload


class TestDuplicateDaemonDefers(unittest.TestCase):
    """A duplicate (port already served by a responsive daemon) must raise
    DuplicateDaemonError so the supervisor exits cleanly instead of crash-
    looping on bind. Backstop for the Errno-48 storm (2026-06-05) — closes the
    class regardless of HOW a second daemon was spawned.

    Step 6 removed _run's is_daemon_responsive PRE-check: the flock (acquired in
    start() before the supervisor loop) is the singleton primitive, and while we
    hold it no other same-uid process is past the flock-acquire — so none can be
    serving our port. test_lock_holder_blocks_start_before_run pins that; the
    bind-time EADDRINUSE backstop below covers the only residue (a different uid
    colliding via uid%100, and the acquire→bind race)."""

    def _daemon(self):
        from servers.daemon_server import BrainDaemon
        return BrainDaemon("/tmp/nonexistent-brain-test.db")

    def test_lock_holder_blocks_start_before_run(self):
        # The flock — not a port pre-check — is what rejects a duplicate. With an
        # incumbent holding the singleton lock, start() must return BEFORE ever
        # invoking _run() (which would open DB writer connections). This is the
        # invariant the deleted _run pre-check was redundantly asserting.
        import fcntl
        from servers import daemon_server as ds
        with tempfile.TemporaryDirectory(prefix="brain-lock-test-") as tmp:
            lock_path = os.path.join(tmp, "daemon.lock")
            incumbent = open(lock_path, "w")
            fcntl.flock(incumbent, fcntl.LOCK_EX | fcntl.LOCK_NB)  # hold it
            try:
                d = ds.BrainDaemon.__new__(ds.BrainDaemon)
                d._log = lambda *a, **k: None
                d.pid_path = os.path.join(tmp, "daemon.pid")
                with open(d.pid_path, "w") as f:
                    f.write("999999")                # the incumbent's
                with patch("shutil.rmtree"), \
                     patch("servers.daemon_server.get_lock_path", return_value=lock_path), \
                     patch("servers.daemon_server.get_pid_path",
                           return_value=os.path.join(tmp, "daemon.pid")), \
                     patch.object(d, "_run") as run, \
                     patch.object(d, "_shutdown") as shutdown:
                    d.start()
                run.assert_not_called()       # flock rejected us → never reached _run
                shutdown.assert_not_called()  # early return, before the loop/_shutdown
                with open(d.pid_path) as f:   # the claim comes after the lock
                    self.assertEqual(f.read(), "999999")
            finally:
                fcntl.flock(incumbent, fcntl.LOCK_UN)
                incumbent.close()

    def test_bind_socket_defers_on_eaddrinuse_when_responsive(self):
        from servers import daemon_server as ds
        d = self._daemon()
        fake_sock = MagicMock()
        fake_sock.bind.side_effect = OSError(errno.EADDRINUSE, "in use")
        with patch.object(ds.socket, "socket", return_value=fake_sock), \
             patch("servers.daemon_client.is_daemon_responsive", return_value=True):
            with self.assertRaises(ds.DuplicateDaemonError):
                d._bind_socket()

    def test_bind_socket_retries_when_holder_unresponsive(self):
        # A non-responsive holder (TIME_WAIT / hung corpse) is NOT a duplicate —
        # normal retry, then raise OSError for recovery to handle.
        from servers import daemon_server as ds
        d = self._daemon()
        d.SOCKET_BIND_RETRIES = 2
        fake_sock = MagicMock()
        fake_sock.bind.side_effect = OSError(errno.EADDRINUSE, "in use")
        with patch.object(ds.socket, "socket", return_value=fake_sock), \
             patch("servers.daemon_client.is_daemon_responsive", return_value=False), \
             patch.object(ds.time, "sleep"):
            with self.assertRaises(OSError):
                d._bind_socket()


class TestSupervisorPhaseScoping(unittest.TestCase):
    """Step 6: the supervisor is PHASE-scoped, not exception-type-scoped.
      • Crash while self.brain is None (before/during the brain load) → a brain-
        level fault. Retrying re-runs the same deterministic failure ×MAX, each
        paying the full load cost while holding the flock and serving nothing —
        so exit to KeepAlive for a fresh process (no in-place retry).
      • Crash with the brain up → a transient serve/socket fault. Warm-retry
        (keep the loaded brain) up to MAX, then give up.
      • The crash streak resets only after HEALTHY_UPTIME_RESET_S of serving, so
        an endless serve-crash loop actually reaches MAX (the old reset-on-every-
        bind let it loop forever) while unrelated crashes hours apart don't
        accumulate.
    Every path stays LOUD — _log_crash writes the traceback before any branch."""

    def setUp(self):
        # Structural safety net: any supervisor test that forgets to stub
        # _exec_reload on its instance would otherwise run the REAL one when
        # a reload flag leaks True — which execs/exits the test process with
        # green-looking output. Class-level patch makes that impossible.
        from servers import daemon_server as ds
        self._er_autopatch = patch.object(ds.BrainDaemon, "_exec_reload",
                                          autospec=True)
        self._er_autopatch.start()
        self.addCleanup(self._er_autopatch.stop)

    def _make(self, tmp):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__ — no pool/brain
        d.db_path = os.path.join(tmp, "brain.db")    # start()'s db lock keys on it
        d._restart_count = 0
        d._reload_requested = False
        d._run_started_at = 0
        d.pid_path = os.path.join(tmp, "daemon.pid")  # absent → no unclean predecessor
        d.logs = []
        d._log = lambda m: d.logs.append(m)
        d._log_crash = MagicMock()
        d._shutdown = MagicMock()
        d._close_socket = MagicMock()
        return d

    def _start(self, d, tmp):
        # Run the REAL supervisor loop with only the flock/signal/pycache/sleep
        # machinery stubbed. The flock acquires cleanly (nothing else holds it).
        lock_path = os.path.join(tmp, "daemon.lock")
        with patch("shutil.rmtree"), \
             patch("servers.daemon_server.get_lock_path", return_value=lock_path), \
             patch("servers.daemon_server.get_db_lock_path",
                   return_value=os.path.join(tmp, "db.lock")), \
             patch("servers.daemon_server.signal.signal"), \
             patch("servers.daemon_server.atexit.register"), \
             patch("servers.daemon_server.time.sleep"):
            d.start()

    def test_load_crash_exits_to_keepalive_no_retry(self):
        with tempfile.TemporaryDirectory(prefix="brain-sup-test-") as tmp:
            d = self._make(tmp)
            d.brain = None                       # load-phase fault
            calls = []
            def boom():
                calls.append(1)
                raise RuntimeError("brain load failed")
            d._run = boom
            self._start(d, tmp)
            self.assertEqual(len(calls), 1, "load-phase crash must NOT warm-retry")
            self.assertEqual(d._restart_count, 0, "no restart counted on the exit-to-KeepAlive path")
            self.assertTrue(any("FATAL: crash before the brain loaded" in m for m in d.logs),
                            "must log the FATAL exit reason")
            d._log_crash.assert_called_once()    # LOUD: traceback before the branch
            d._shutdown.assert_called_once()     # clean teardown → flock released → KeepAlive respawns

    def test_serve_crash_warm_retries_then_gives_up(self):
        with tempfile.TemporaryDirectory(prefix="brain-sup-test-") as tmp:
            d = self._make(tmp)
            d.brain = MagicMock()                # brain up → serve-phase fault
            calls = []
            def boom():
                calls.append(1)                  # never sets _run_started_at (crashes "fast")
                raise RuntimeError("serve socket died")
            d._run = boom
            self._start(d, tmp)
            # 1 initial run + MAX warm retries, then FATAL give-up.
            self.assertEqual(len(calls), d.MAX_SUPERVISOR_RESTARTS + 1)
            self.assertTrue(any("Giving up" in m for m in d.logs))

    def test_healthy_uptime_resets_streak(self):
        with tempfile.TemporaryDirectory(prefix="brain-sup-test-") as tmp:
            d = self._make(tmp)
            d.brain = MagicMock()
            seq = []
            def run():
                seq.append(1)
                if len(seq) <= 3:
                    # each crash lands AFTER a full healthy-uptime interval
                    d._run_started_at = time.time() - (d.HEALTHY_UPTIME_RESET_S + 10)
                    raise RuntimeError("crash after healthy uptime")
                return  # 4th run: clean shutdown → loop breaks
            d._run = run
            self._start(d, tmp)
            # 3 healthy-uptime crashes each reset the streak to 0 (→ +1 = 1), so we
            # NEVER approach MAX(5) — they don't accumulate. 4th run returns clean.
            self.assertEqual(len(seq), 4)
            self.assertEqual(d._restart_count, 1)
            self.assertFalse(any("Giving up" in m for m in d.logs),
                             "healthy-uptime crashes must not accumulate toward give-up")


class TestServeLoopExitPaths(unittest.TestCase):
    """_serve's clock, and WHERE the clean-vs-fault question is answered.

    _serve has three exits (while falls through, socket-is-None break, select
    fault break). It answers NONE of them — every exit is a bare break, and
    _run asks once after it returns. The earlier shape asked per arm and one
    arm forgot, so a real socket fault could still read as a clean shutdown:
    no _log_crash, no restart count, MAX_SUPERVISOR_RESTARTS unreachable.

    Also pinned here: the 5s tick is MONOTONIC. On wall clock a backward step
    (NTP, manual set) makes the delta negative and freezes both gate bodies —
    no S2 submit, no Scribe poll — while the loop keeps accepting, so nothing
    looks wrong in daemon.log."""

    def _make(self):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__
        d.running = True
        d.server_socket = MagicMock()
        d._pool = MagicMock()
        d._shutdown = MagicMock()
        d.logs = []
        d._log = lambda m: d.logs.append(m)
        return d

    # ── the clock ──

    def test_tick_survives_backward_wall_clock(self):
        d = self._make()
        d.server_socket = None          # break out after the first tick
        with patch("servers.daemon_server.time.monotonic",
                   side_effect=[0.0, 10.0, 20.0, 30.0]), \
             patch("servers.daemon_server.time.time", return_value=900.0):
            d._serve()
        self.assertEqual(d._pool.submit.call_count, 2,
                         "maintenance + Scribe poll must both submit despite a "
                         "backward wall-clock step")

    # ── _serve answers nothing; it returns ──

    def test_serve_never_raises_and_never_tears_down(self):
        """Both the signal race (socket nulled) and a genuine select fault must
        leave _serve by a plain return. Teardown is the supervisor's — _serve
        calling _shutdown() would run teardown before a warm retry."""
        d = self._make()
        d.server_socket = None          # real select would TypeError on [None]
        d._serve()
        d._shutdown.assert_not_called()

        d2 = self._make()
        with patch("servers.daemon_server.select.select",
                   side_effect=OSError(errno.EBADF, "Bad file descriptor")):
            d2._serve()                 # must NOT raise
        d2._shutdown.assert_not_called()
        self.assertTrue(any("select ended" in m for m in d2.logs),
                        "the errno must be logged where it is known")
        self.assertFalse(any("fail" in m.lower() for m in d2.logs),
                         "worded as an observation, not a verdict — a SIGTERM "
                         "mid-select lands here on a CLEAN shutdown, and "
                         "'failed' in daemon.log would make the normal path "
                         "read as a fault")
        self.assertTrue(d2.running,
                        "_serve must leave running untouched — it is the "
                        "signal _run reads to tell a fault from a shutdown")

    # ── _run answers, once ──

    def _run_with_serve(self, serve, running_after):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)
        d.brain = MagicMock()
        d.logs = []
        d._log = lambda m: d.logs.append(m)
        d._bind_socket = MagicMock()
        d._write_status = MagicMock()
        d._restart_count = 0
        d.daemon_addr = ('127.0.0.1', 0)
        d._pool = MagicMock()
        d._db_maintenance = None

        def _serve():
            serve()
            d.running = running_after
        d._serve = _serve
        with tempfile.TemporaryDirectory(prefix="brain-run-test-") as tmp:
            d.pid_path = os.path.join(tmp, "daemon.pid")
            # A real string db_path so default_backup_dir works; the scheduler
            # and watchdog are patched AT THEIR SOURCE MODULES, because _run
            # imports both inside the function body — patching them on
            # daemon_server is a silent no-op that lets the real ones run.
            d.brain.db_path = os.path.join(tmp, "brain.db")
            with patch("servers.daemon_server.threading.Thread"), \
                 patch("servers.db_maintenance.DBMaintenance"), \
                 patch("servers.memory_watchdog.MemoryWatchdog"):
                d._run()
        return d

    def test_run_raises_when_serve_exits_while_still_running(self):
        """The socket died: nobody asked us to stop, so the exit is a fault.
        Raising is what reaches start()'s handler — _log_crash, the restart
        count, and the give-up threshold all hang off this."""
        with self.assertRaises(RuntimeError) as ctx:
            self._run_with_serve(lambda: None, running_after=True)
        self.assertIn("exited while still running", str(ctx.exception))

    def test_run_returns_cleanly_when_serve_exits_after_shutdown(self):
        """Told to die → plain return, so the supervisor breaks out instead of
        counting a crash and restarting a daemon that was asked to stop."""
        d = self._run_with_serve(lambda: None, running_after=False)
        self.assertFalse(d.running)


class TestShutdownDrainOrder(unittest.TestCase):
    """_shutdown must drain the worker pool AND settle the bg-writer/queues
    BEFORE saving + closing the brain — otherwise an in-flight hook or a mid-drain
    touches a closed connection ('Cannot operate on a closed database', the
    2026-06-06 boot failure, where close() ran before the pool/bg-writer drained).
    Regression guard for the drain ORDER."""

    def test_drains_then_closes_then_releases_resources_last(self):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__ — no real brain/socket
        order = []
        d.brain = MagicMock()
        d.brain.save.side_effect = lambda: order.append("save")
        d.brain.close.side_effect = lambda: order.append("close")
        d._pool = MagicMock()
        d._pool.shutdown.side_effect = lambda **k: order.append("pool")
        d._log = lambda *a, **k: None
        d._log_shutdown_error = lambda *a, **k: None
        d._signal_drain_shutdown = lambda: order.append("signal")   # don't touch the real queues
        d._cleanup = lambda: order.append("release")                # _cleanup = release lock/PID/socket
        with patch("servers.embed_queue.join_worker",
                   side_effect=lambda **k: order.append("join")), \
             patch("_thread.start_new_thread"):                     # don't arm the os._exit backstop
            d._shutdown()
        # signal queues → drain pool → join bg-writer → save+close → release lock/PID LAST.
        # Releasing the singleton lock LAST is what stops a racer opening a 2nd
        # writer on brain.db while our connections were still open.
        self.assertEqual(order, ["signal", "pool", "join", "save", "close", "release"])


class TestTeardownPersistsSessionState(BrainTestBase):
    """A graceful shutdown persists the SessionContext changes that live only in
    the cache. hook_recall's last_recall_stop and the Stop's counter advance
    stay in memory until flushed; a restart that drops them makes the next Stop
    read a stale row and record the reply as a heartbeat. Brain.close() owns
    the flush, so the daemon teardown and every direct closer (the boot
    fallback, scripts) keep them."""
    needs_embedder = False
    SID = 'teardown-sess'

    def _mutate_cached_ctx(self):
        ctx = self.brain.get_or_create_session(self.SID)
        ctx.increment_stop()                        # the Stop's counter advance
        ctx.last_recall_stop = ctx.stop_counter     # hook_recall's classification stamp

    def _persisted_after_close(self):
        """(stop_counter, last_recall_stop) as the row holds them, read over a
        fresh connection — the brain is closed, so the env must not close it
        again."""
        import sqlite3
        from servers.dal_logs import SessionStateDAL
        from servers.session_context import SessionContext
        logs_db = self.brain.logs_db_path
        self._env.brain = None
        conn = sqlite3.connect(logs_db)
        try:
            after = SessionContext.load(SessionStateDAL(conn), self.SID)
        finally:
            conn.close()
        return (after.stop_counter, after.last_recall_stop)

    def test_brain_close_persists_cached_mutations(self):
        self._mutate_cached_ctx()
        self.brain.close()
        self.assertEqual(self._persisted_after_close(), (1, 1))

    def test_unsaved_session_mutations_survive_teardown(self):
        from servers import daemon_server as ds
        self._mutate_cached_ctx()
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)  # no __init__ — no socket, no pool
        d.brain = self.brain
        d._pool = MagicMock()
        errors = []
        d._log_shutdown_error = lambda source, err: errors.append((source, err))
        d._signal_drain_shutdown = lambda: None     # don't touch the real queues
        d._cleanup = lambda: None                   # no lock/PID/socket to release
        with patch("servers.embed_queue.join_worker"):
            d._teardown_brain()
        self.assertEqual(errors, [])
        self.assertEqual(self._persisted_after_close(), (1, 1))


class TestSessionFlushCadence(unittest.TestCase):
    """Session state flushes on its own short cadence; the heavy autosave pass
    (brain.save, the `ps -M` fork, thread walk, status file) keeps its own."""

    def _daemon(self, stop_after_sleeps):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # no __init__ — no socket, no pool
        d.brain = MagicMock()
        d.dirty = False
        d.running = True
        d._log = lambda *a, **k: None
        d._write_status = MagicMock()
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) >= stop_after_sleeps:
                d.running = False
        return ds, d, sleeps, fake_sleep

    def test_autosave_keeps_its_interval_and_no_longer_flushes_sessions(self):
        ds, d, sleeps, fake_sleep = self._daemon(stop_after_sleeps=1)
        with patch.object(ds.time, "sleep", fake_sleep), \
             patch("subprocess.run") as m_run:
            d._autosave_loop()
        self.assertEqual(sleeps, [ds.AUTOSAVE_INTERVAL_SECONDS])
        self.assertIn("-M", m_run.call_args.args[0])  # the native-thread CPU monitor still runs
        d._write_status.assert_called_once()
        d.brain.save_session_contexts.assert_not_called()

    def test_session_flush_runs_on_its_own_interval_without_heavy_work(self):
        ds, d, sleeps, fake_sleep = self._daemon(stop_after_sleeps=2)
        with patch.object(ds.time, "sleep", fake_sleep), \
             patch("subprocess.run") as m_run:
            d._session_flush_loop()
        self.assertEqual(sleeps, [ds.SESSION_FLUSH_INTERVAL_SECONDS] * 2)
        self.assertLess(ds.SESSION_FLUSH_INTERVAL_SECONDS, ds.AUTOSAVE_INTERVAL_SECONDS)
        # One flush: the second wake found the daemon shutting down, and
        # Brain.close() owns that final flush.
        d.brain.save_session_contexts.assert_called_once_with()
        m_run.assert_not_called()
        d._write_status.assert_not_called()


class TestEmbedQueueShutdown(unittest.TestCase):
    """request_shutdown() must wake the drain worker out of its interval wait
    immediately (an Event, not a passive flag), so daemon shutdown doesn't block
    for a full EMBED_DRAIN_INTERVAL and brain.close() can't race a mid-drain."""

    def setUp(self):
        import servers.embed_queue as eq
        self.eq = eq
        self._reset()

    def tearDown(self):
        self._reset()

    def _reset(self):
        eq = self.eq
        with eq._lock:
            eq._worker_started = False
        eq._shutdown_event.clear()      # the single bg-writer shutdown signal
        eq._worker_thread = None

    def test_request_shutdown_wakes_worker_promptly(self):
        eq = self.eq
        # Long interval — only the Event (not the timeout) can make it exit fast.
        with patch.object(eq, "EMBED_DRAIN_INTERVAL", 30):
            eq.start(MagicMock())
            self.assertTrue(eq._worker_thread.is_alive())
            eq.request_shutdown()                 # sets the Event → wakes the worker out of the 30s wait now
            eq.join_worker(timeout=2.0)
            self.assertFalse(eq._worker_thread.is_alive(),
                             "worker must exit on the Event, not wait the full 30s interval")


class TestAwaitResponsive(unittest.TestCase):
    """_await_responsive — the boot-path liveness gate. WALL-CLOCK bounded,
    returns on the first ok ping (no sleep), returns the last resp when never up."""

    def test_returns_on_first_ok_without_sleeping(self):
        with patch.object(dc, "_can_connect", return_value={"ok": True}) as ping, \
             patch.object(dc.time, "sleep") as slept:
            resp = dc._await_responsive(20.0)
        self.assertEqual(resp, {"ok": True})
        ping.assert_called_once()       # one ping, no retry when already up
        slept.assert_not_called()       # never sleeps a live daemon

    def test_wall_clock_bound_returns_last_resp_when_never_up(self):
        # deadline_s=0 → no retry past the first ping; returns the last (down) resp.
        with patch.object(dc, "_can_connect", return_value={}), \
             patch.object(dc.time, "sleep") as slept:
            resp = dc._await_responsive(0.0)
        self.assertEqual(resp, {})      # never ok → returns the last resp (down)
        slept.assert_not_called()       # zero deadline never enters the retry loop


class TestExecReloadInPlace(unittest.TestCase):
    """The `restart` command reloads in place: the handler flags the reload and
    stops the serve loop; start() (main thread) runs the ordered _shutdown()
    THEN execs hooks/scripts/brain-daemon into the same process — the launchd
    entry point, so env flags and the DB ladder re-resolve exactly as on a
    real boot. No spawn, no launchd round-trip — the competing-spawner surface
    (Popen orphans 2026-07-03/04, kickstart two-writer window 2026-07-06)
    does not exist on this path. Exec failure exits loudly; KeepAlive / the
    next ensure_daemon respawns (the pre-reload behavior)."""

    def _make_daemon(self):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__ — no real brain/socket
        d.db_path = "/tmp/nonexistent-brain-test.db"
        d.pid_path = "/tmp/nonexistent-brain-test.pid"  # absent → utime no-ops
        d._reload_requested = True   # the state _exec_reload runs under
        d.logs = []
        d._log = lambda m: d.logs.append(m)
        return d

    def test_restart_command_flags_reload_and_stops_loop(self):
        d = self._make_daemon()
        d.running = True
        d._reload_requested = False
        resp = d._dispatch("restart", {})
        self.assertEqual(resp, {"ok": True, "result": {"status": "restarting"}})
        self.assertTrue(d._reload_requested)
        self.assertFalse(d.running)   # main thread exits _serve → _shutdown → exec

    def test_exec_reload_execs_brain_daemon_launcher(self):
        # The reload execs hooks/scripts/brain-daemon — the launchd entry
        # point — so the resolver chain (brain-env.sh flags, key file, DB
        # ladder) re-runs on every reload exactly as on a real boot. NOT
        # daemon_argv directly: that froze the environment of the first boot
        # forever. PYTHONPATH is stripped (the launcher prepends
        # unconditionally — passing it through grows it every reload).
        from servers.daemon_launch import REPO_ROOT
        with tempfile.TemporaryDirectory(prefix="brain-reload-test-") as tmp:
            d = self._make_daemon()
            d.pid_path = os.path.join(tmp, "daemon.pid")
            with patch("shutil.rmtree") as m_rmtree, \
                 patch("os.path.isdir", return_value=True), \
                 patch("os.chdir") as m_chdir, \
                 patch("os.execve") as m_exec, \
                 patch("os._exit") as m_exit, \
                 patch("subprocess.Popen") as m_popen, \
                 patch.dict("os.environ", {"PYTHONPATH": "/somewhere"}):
                d._exec_reload()
            launcher = os.path.join(REPO_ROOT, "hooks", "scripts", "brain-daemon")
            self.assertEqual(m_exec.call_args[0][0], launcher)
            self.assertEqual(m_exec.call_args[0][1], [launcher])
            self.assertNotIn("PYTHONPATH", m_exec.call_args[0][2])
            m_chdir.assert_called_once_with(REPO_ROOT)
            # pycache clearing may only ever target __pycache__ dirs — a
            # regression to rmtree(project_dir) deletes the repo.
            for call in m_rmtree.call_args_list:
                self.assertTrue(call.args[0].endswith("__pycache__"),
                                "rmtree target must be a __pycache__ dir: %s" % (call.args,))
            # Corpse-test breadcrumb: the PID file is rewritten (fresh mtime,
            # our own PID — truthful, exec keeps the PID) just before exec,
            # so recover_daemon reads the whole boot window as "just bound".
            with open(d.pid_path) as f:
                self.assertEqual(f.read(), str(os.getpid()))
            m_exit.assert_not_called()     # success path never exits
            m_popen.assert_not_called()    # THE invariant — no spawn, ever

    def test_exec_failure_exits_loudly_for_keepalive(self):
        with tempfile.TemporaryDirectory(prefix="brain-reload-test-") as tmp:
            d = self._make_daemon()
            d.pid_path = os.path.join(tmp, "daemon.pid")
            with patch("shutil.rmtree"), \
                 patch("os.chdir"), \
                 patch("os.execve", side_effect=OSError("exec failed")), \
                 patch("os._exit", side_effect=SystemExit) as m_exit, \
                 patch("subprocess.Popen") as m_popen:
                with self.assertRaises(SystemExit):
                    d._exec_reload()
            m_exit.assert_called_once_with(0)
            m_popen.assert_not_called()
            self.assertTrue(any("FAILED" in m for m in d.logs),
                            "a failed exec must be loud before the fallback exit")

    def test_restart_handler_never_spawns_a_thread(self):
        # The 2026-04 execv died SILENTLY because it ran on a handler thread
        # (7b4a01a) — worse, a regression re-adding a thread here would run
        # the real _exec_reload during tests and os._exit the test process
        # with green-looking output. Pin: the handler only flags; the exec
        # belongs to start() on the main thread.
        from servers import daemon_server as ds
        d = self._make_daemon()
        d.running = True
        d._reload_requested = False
        with patch.object(ds.threading, "Thread") as m_thread:
            d._dispatch("restart", {})
        m_thread.assert_not_called()

    def test_restart_command_freshens_pid_file_for_teardown(self):
        # The corpse-test breadcrumb's FIRST half: teardown runs for seconds
        # with the socket open-but-unanswering; without the utime the PID
        # file still carries its hours-old bind mtime and recover_daemon
        # reads exactly a corpse. The utime makes teardown read "just bound".
        with tempfile.TemporaryDirectory(prefix="brain-reload-test-") as tmp:
            d = self._make_daemon()
            d.running = True
            d._reload_requested = False
            d.pid_path = os.path.join(tmp, "daemon.pid")
            with open(d.pid_path, "w") as f:
                f.write("12345")
            old = time.time() - 3600
            os.utime(d.pid_path, (old, old))
            d._dispatch("restart", {})
            self.assertLess(time.time() - os.stat(d.pid_path).st_mtime, 5.0)

    def test_exec_reload_honors_cancellation_at_the_last_instant(self):
        # A SIGTERM landing during teardown clears the flag; _exec_reload
        # must re-check right before execve and exit instead of exec'ing an
        # unmanaged orphan past a launchctl bootout.
        with tempfile.TemporaryDirectory(prefix="brain-reload-test-") as tmp:
            d = self._make_daemon()
            d.pid_path = os.path.join(tmp, "daemon.pid")
            d._reload_requested = False   # cleared mid-teardown
            with patch("shutil.rmtree"), \
                 patch("os.chdir"), \
                 patch("os.execve") as m_exec, \
                 patch("os._exit", side_effect=SystemExit) as m_exit:
                with self.assertRaises(SystemExit):
                    d._exec_reload()
            m_exec.assert_not_called()
            m_exit.assert_called_once_with(0)
            # Teardown removed the PID file; a cancelled reload must not write a
            # breadcrumb that would read as an unclean death to the next start.
            self.assertFalse(os.path.exists(d.pid_path))

    def test_shutdown_is_idempotent_one_backstop(self):
        # _shutdown runs at _serve's tail AND start()'s tail on every clean
        # path; only the FIRST call may tear down and arm the 15s force-exit
        # backstop — two armed backstops obscured the real teardown budget.
        d = self._make_daemon()
        d._arm_force_exit_backstop = MagicMock()
        d._teardown_brain = MagicMock()
        d._shutdown()
        d._shutdown()
        d._arm_force_exit_backstop.assert_called_once()
        d._teardown_brain.assert_called_once()

    def test_shutdown_command_cancels_pending_reload(self):
        d = self._make_daemon()
        d.running = True
        d._reload_requested = False
        d._dispatch("restart", {})
        self.assertTrue(d._reload_requested)
        resp = d._dispatch("shutdown", {})
        self.assertEqual(resp["result"]["status"], "shutting_down")
        self.assertFalse(d._reload_requested,
                         "a shutdown must override an in-flight reload — the "
                         "daemon must not resurrect itself after stop_daemon")

    def test_signal_cancels_pending_reload(self):
        d = self._make_daemon()
        d.running = True
        d._reload_requested = True
        d.server_socket = None
        d._handle_signal(15, None)
        self.assertFalse(d._reload_requested,
                         "SIGTERM after a restart command is a shutdown, not a reload")
        self.assertFalse(d.running)

    def test_start_execs_only_when_reload_requested(self):
        # The wiring: after the supervisor loop + _shutdown, start() execs
        # iff the restart command flagged a reload — and STRICTLY AFTER the
        # teardown. The order is the two-writer invariant (the exec'd
        # successor must be the only opener of brain.db), asserted as an
        # ORDERED call list, not two independent .called checks — a swap of
        # the two lines must fail here.
        from unittest.mock import call
        for requested in (False, True):
            with tempfile.TemporaryDirectory(prefix="brain-reload-test-") as tmp:
                d = self._make_daemon()
                d.brain = MagicMock()
                d._restart_count = 0
                d._run_started_at = 0
                d._log_crash = MagicMock()
                d._close_socket = MagicMock()
                d._reload_requested = requested
                d._run = lambda: None            # clean shutdown → loop breaks
                manager = MagicMock()
                d._shutdown = manager.shutdown
                d._exec_reload = manager.exec_reload
                with patch("servers.daemon_server.get_lock_path",
                           return_value=os.path.join(tmp, "daemon.lock")), \
                     patch("servers.daemon_server.get_db_lock_path",
                           return_value=os.path.join(tmp, "db.lock")), \
                     patch("servers.daemon_server.signal.signal"), \
                     patch("servers.daemon_server.atexit.register"), \
                     patch("servers.daemon_server.time.sleep"):
                    d.start()
                expected = [call.shutdown()] + ([call.exec_reload()] if requested else [])
                self.assertEqual(manager.mock_calls, expected)


class TestSyncBeforeSwap(unittest.TestCase):
    """A restart onto a tree whose runtime is behind its uv.lock syncs FIRST,
    while the old daemon still serves, and reloads when the sync ends (node
    c34a86e0). Synced inside the exec'd launcher, the sync ran for 30s+ with
    nothing serving, and another session's hook recovery kickstarted the
    daemon and killed uv mid-sync. The handler only flags; the serve loop
    (main thread) owns the ensure-runtime child from spawn to reap."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="brain-sync-swap-test-")
        self.addCleanup(shutil.rmtree, self._tmp, True)

    def _make_daemon(self):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__ — no brain/socket
        d.pid_path = os.path.join(self._tmp, "daemon.pid")
        d.running = True
        d._reload_requested = False
        d.brain = MagicMock()
        d.logs = []
        d._log = lambda m: d.logs.append(m)
        return d

    def _finished_sync(self, code):
        child = MagicMock()
        child.poll.return_value = code
        return child

    # ── the handler ──

    def test_stale_runtime_restart_flags_a_sync_and_keeps_serving(self):
        d = self._make_daemon()
        with patch("servers.daemon_server.runtime_ready", return_value=False), \
             patch("servers.daemon_server.start_runtime_sync") as start:
            resp = d._dispatch("restart", {})
        self.assertEqual(resp, {"ok": True, "result": {"status": "syncing"}})
        self.assertTrue(d._sync_requested)
        self.assertTrue(d.running, "the old daemon serves through the sync")
        self.assertFalse(d._reload_requested)
        start.assert_not_called()   # the serve loop starts it, not a handler thread

    def test_restart_during_a_sync_answers_syncing_without_a_second_sync(self):
        d = self._make_daemon()
        d._runtime_sync = self._finished_sync(None)   # still running
        with patch("servers.daemon_server.runtime_ready") as ready:
            resp = d._dispatch("restart", {})
        self.assertEqual(resp["result"]["status"], "syncing")
        ready.assert_not_called()
        self.assertFalse(d._reload_requested)

    def test_ready_runtime_restart_reloads_at_once(self):
        d = self._make_daemon()
        with patch("servers.daemon_server.runtime_ready", return_value=True):
            resp = d._dispatch("restart", {})
        self.assertEqual(resp["result"]["status"], "restarting")
        self.assertTrue(d._reload_requested)
        self.assertFalse(d.running)
        self.assertFalse(d._sync_requested)

    # ── the serve-loop step ──

    def test_step_starts_this_trees_sync_and_keeps_serving(self):
        from servers.daemon_config import REPO_ROOT
        d = self._make_daemon()
        d._sync_requested = True
        with patch("servers.daemon_server.start_runtime_sync") as start:
            d._step_runtime_sync()
        start.assert_called_once_with(REPO_ROOT)
        self.assertIs(d._runtime_sync, start.return_value)
        self.assertFalse(d._sync_requested)
        self.assertTrue(d.running)

    def test_reload_follows_a_finished_sync(self):
        d = self._make_daemon()
        with open(d.pid_path, "w") as f:
            f.write("12345")
        old = time.time() - 3600
        os.utime(d.pid_path, (old, old))
        d._runtime_sync = self._finished_sync(None)
        d._step_runtime_sync()
        self.assertFalse(d._reload_requested, "no swap while uv still runs")
        self.assertTrue(d.running)
        d._runtime_sync = self._finished_sync(0)
        d._step_runtime_sync()
        self.assertIsNone(d._runtime_sync)
        self.assertTrue(d._reload_requested)
        self.assertFalse(d.running)
        # The teardown that follows must read "just bound" to recover_daemon.
        self.assertLess(time.time() - os.stat(d.pid_path).st_mtime, 5.0)

    def test_failed_sync_cancels_the_reload_loudly(self):
        # Non-zero means the exec'd launcher's ensure-runtime would fail the
        # same way (a failed RE-sync already exits 0 and keeps the old venv),
        # so swapping would crash-loop the daemon. Keep serving, loudly.
        d = self._make_daemon()
        d._runtime_sync = self._finished_sync(1)
        d._step_runtime_sync()
        self.assertIsNone(d._runtime_sync)
        self.assertFalse(d._reload_requested)
        self.assertTrue(d.running)
        self.assertTrue(any("FAILED" in m for m in d.logs))
        self.assertEqual(d.brain._log_error.call_args[0][0], "runtime_sync_failed")

    def test_sync_that_cannot_start_keeps_serving(self):
        # The step runs inside _serve; an escaping OSError (fork EAGAIN, a
        # script without its exec bit) would crash-loop the supervisor on the
        # still-set request and end with the daemon exiting.
        d = self._make_daemon()
        d._sync_requested = True
        with patch("servers.daemon_server.start_runtime_sync",
                   side_effect=OSError(errno.EAGAIN, "Resource temporarily unavailable")):
            d._step_runtime_sync()   # must not raise
        self.assertIsNone(d._runtime_sync)
        self.assertFalse(d._sync_requested, "no retry on every loop iteration")
        self.assertTrue(d.running)
        self.assertFalse(d._reload_requested)
        self.assertTrue(any("FAILED" in m for m in d.logs))

    def test_shutdown_during_a_sync_wins_over_the_reload(self):
        d = self._make_daemon()
        d._runtime_sync = self._finished_sync(0)
        d._dispatch("shutdown", {})
        d._step_runtime_sync()   # the loop iteration the shutdown landed in
        self.assertFalse(d._reload_requested,
                         "the daemon must not resurrect itself after a shutdown")

    def test_serve_loop_steps_a_requested_sync(self):
        d = self._make_daemon()
        d.server_socket = None          # break out after the first iteration
        d._pool = MagicMock()
        d._sync_requested = True
        with patch.object(d, "_step_runtime_sync") as step:
            d._serve()
        step.assert_called_once()

    # ── the launch primitives (daemon_launch) ──

    def test_sync_runs_the_trees_ensure_runtime_in_its_own_session(self):
        # BRAIN_RUNTIME_SERVE_STALE makes ensure-runtime skip the sync, the one
        # thing wanted here, so the child must not inherit it. Its own session
        # keeps a signal to the daemon's process group from killing uv mid-sync.
        with patch.dict(os.environ, {"BRAIN_RUNTIME_SERVE_STALE": "1"}), \
             patch.object(dl.subprocess, "Popen") as popen:
            child = dl.start_runtime_sync("/some/tree")
        self.assertIs(child, popen.return_value)
        self.assertEqual(popen.call_args[0][0],
                         ["/some/tree/hooks/scripts/ensure-runtime.sh"])
        kwargs = popen.call_args[1]
        self.assertNotIn("BRAIN_RUNTIME_SERVE_STALE", kwargs["env"])
        self.assertTrue(kwargs["start_new_session"])

    def test_runtime_ready_asks_the_trees_runtime_state(self):
        from servers.daemon_config import REPO_ROOT
        tree = self._tmp
        scripts = os.path.join(tree, "hooks", "scripts")
        os.makedirs(scripts)
        shutil.copy(os.path.join(REPO_ROOT, "hooks", "scripts", "runtime-state.sh"), scripts)
        with open(os.path.join(tree, "uv.lock"), "w") as f:
            f.write("lock v1\n")
        with open(os.path.join(tree, ".python-version"), "w") as f:
            f.write("3.11\n")
        os.makedirs(os.path.join(tree, "venv", "bin"))
        python = os.path.join(tree, "venv", "bin", "python")
        with open(python, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(python, 0o755)
        fingerprint = subprocess.run(
            ["/bin/bash", "-c", '. "$1/hooks/scripts/runtime-state.sh"; '
             'brain_runtime_fingerprint "$1"', "_", tree],
            capture_output=True, text=True, check=True).stdout
        sentinel = os.path.join(tree, ".runtime-ready")
        with open(sentinel, "w") as f:
            f.write(fingerprint)
        self.assertTrue(dl.runtime_ready(tree))
        with open(os.path.join(tree, "uv.lock"), "w") as f:
            f.write("lock v2\n")
        t = time.time()
        os.utime(sentinel, (t - 10, t - 10))
        os.utime(os.path.join(tree, "uv.lock"), (t - 5, t - 5))
        self.assertFalse(dl.runtime_ready(tree))


class TestSpawnDetachedDaemon(unittest.TestCase):
    """spawn_detached_daemon is the ONE spawn, and it is the HARDENED one:
    debugger-friendly interpreter, devnull stdin, log redirect, own session,
    full CPU-only env. Step 2 unified the two drifted spawn sites (ensure_daemon
    fallback vs _perform_restart no-launchd — the latter ran raw sys.executable
    with inherited stdin) into this primitive."""

    def test_spawn_is_hardened(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="brain-spawn-test-") as tmp, \
             patch.object(dl.subprocess, "Popen") as popen, \
             patch.object(dl, "debugger_friendly_python", return_value="/usr/bin/python3"), \
             patch.dict(os.environ, {"BRAIN_DB_DIR": tmp}):
            dl.spawn_detached_daemon(os.path.join(tmp, "brain.db"))
        popen.assert_called_once()
        argv = popen.call_args[0][0]
        kwargs = popen.call_args[1]
        self.assertEqual(argv[0], "/usr/bin/python3")  # debugger-friendly, not raw sys.executable
        # Step 6c: `-m servers.daemon_server <db>`, the same command
        # hooks/scripts/brain-daemon execs. The DB path is an argv element,
        # never interpolated into Python source.
        self.assertEqual(argv[1:3], ["-m", "servers.daemon_server"])
        self.assertEqual(argv[3], os.path.join(tmp, "brain.db"))
        self.assertEqual(len(argv), 4)
        self.assertTrue(kwargs["start_new_session"])
        self.assertIsNotNone(kwargs["stdin"])           # devnull, never inherited
        from servers.daemon_config import DAEMON_CPU_ENV, REPO_ROOT
        self.assertTrue(set(DAEMON_CPU_ENV).issubset(kwargs["env"]),
                        "spawn must merge the full CPU-only env")
        # The two pins that moved out of the old `-c` string and into the env,
        # where they apply before exec instead of after interpreter start.
        self.assertEqual(kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], REPO_ROOT,
                         "`-m` must resolve the package whatever the spawner's cwd is")
        self.assertEqual(kwargs["env"]["BRAIN_DB_DIR"], tmp)

    def test_daemon_env_prepends_inherited_pythonpath(self):
        # Replacing an inherited PYTHONPATH would break a caller that put
        # something on it deliberately (eval harnesses, isolated brains).
        with patch.dict(os.environ, {"PYTHONPATH": "/somewhere/else"}):
            env = dl.daemon_env("/tmp/x/brain.db")
        from servers.daemon_config import REPO_ROOT
        self.assertEqual(env["PYTHONPATH"],
                         REPO_ROOT + os.pathsep + "/somewhere/else")

    def test_daemon_env_pythonpath_prepend_is_idempotent(self):
        # A chained env (boot-brain.sh exports the repo root before
        # ensure_daemon spawns; reload envs chain generations) must not grow
        # PYTHONPATH by one REPO_ROOT per pass.
        from servers.daemon_config import REPO_ROOT
        inherited = REPO_ROOT + os.pathsep + "/somewhere/else"
        with patch.dict(os.environ, {"PYTHONPATH": inherited}):
            env = dl.daemon_env("/tmp/x/brain.db")
        self.assertEqual(env["PYTHONPATH"], inherited)

    def test_daemon_env_falls_back_to_db_parent(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAIN_DB_DIR", None)
            env = dl.daemon_env("/tmp/some-brain/brain.db")
        self.assertEqual(env["BRAIN_DB_DIR"], "/tmp/some-brain")

    def test_entry_point_rejects_bad_argv(self):
        # A daemon that resolved its own DB would be a second resolver; the
        # entry point takes exactly one path and refuses anything else.
        import servers.daemon_server as ds
        self.assertEqual(ds.main([]), 2)
        self.assertEqual(ds.main([""]), 2)
        self.assertEqual(ds.main(["a", "b"]), 2)

    def test_no_raw_popen_outside_daemon_launch(self):
        # Both spawn callers must route through spawn_detached_daemon — a direct
        # Popen in either module is the un-hardened-drift bug class coming back.
        import inspect
        import servers.daemon_server
        for mod in (dc, servers.daemon_server):
            self.assertNotIn(
                "Popen(", inspect.getsource(mod),
                "%s must spawn via daemon_launch.spawn_detached_daemon" % mod.__name__)
        # And the daemon itself must never spawn AT ALL — reload is an exec,
        # never a rival process (the 2026-07-03/04 orphan storms). This holds
        # the line the deleted _perform_restart tests used to hold: even
        # routing through the hardened spawn primitive is forbidden here.
        self.assertNotIn(
            "spawn_detached_daemon", inspect.getsource(servers.daemon_server),
            "the daemon reloads via exec; it never spawns a successor process")


class TestForcedRestartRecord(unittest.TestCase):
    """The hook_errors row a forced restart leaves, written for real into a
    temp brain dir (the module guard stubs it everywhere else)."""

    def test_row_lands_in_the_given_brain_dir(self):
        import sqlite3
        with tempfile.TemporaryDirectory(prefix="brain-restart-record-") as tmp:
            _real_log_hook_error("DAEMON_RESTART_FORCED", "act", context="ctx",
                                 level="warning", db_dir=tmp)
            conn = sqlite3.connect(os.path.join(tmp, "brain_logs.db"))
            try:
                rows = conn.execute(
                    "SELECT hook_name, level, error, context FROM hook_errors").fetchall()
            finally:
                conn.close()
        self.assertEqual(rows, [("DAEMON_RESTART_FORCED", "warning", "act", "ctx")])

    def test_a_dir_that_does_not_exist_gets_no_logs_db(self):
        with tempfile.TemporaryDirectory(prefix="brain-restart-record-") as tmp:
            absent = os.path.join(tmp, "absent")
            _real_log_hook_error("DAEMON_RESTART_FORCED", "act", db_dir=absent)
            self.assertFalse(os.path.exists(absent))

    def test_evidence_names_what_the_corpse_test_reads(self):
        with patch.object(dc, "pid_file_age_s", return_value=42.4):
            self.assertEqual(_real_restart_evidence(), "PID file 42s old")
        for held in (True, False):
            with self.subTest(port_held=held), \
                 patch.object(dc, "pid_file_age_s", return_value=None), \
                 patch.object(dc, "port_is_occupied", return_value=held):
                self.assertEqual(_real_restart_evidence(),
                                 "no PID file, port %s" % ("held" if held else "free"))

    def test_record_reads_evidence_now_and_names_the_caller(self):
        # The module guard stubs _restart_evidence to "evidence".
        dc.log_hook_error.reset_mock()
        dc.record_forced_restart("act", caller="[launchd-install] pid 7")
        self.assertEqual(dc.log_hook_error.call_args.kwargs["context"],
                         "evidence; caller [launchd-install] pid 7")
        dc.record_forced_restart("act")
        self.assertRegex(dc.log_hook_error.call_args.kwargs["context"],
                         r"^evidence; caller \S+ pid %d$" % os.getpid())

    def _bare_daemon(self, tmp, pid_text):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__
        d.db_path = os.path.join(tmp, "brain.db")
        d.pid_path = os.path.join(tmp, "daemon.pid")
        d._log = lambda msg: None
        if pid_text is not None:
            with open(d.pid_path, "w") as f:
                f.write(pid_text)
        return d

    def test_claim_records_an_unclean_predecessor_and_takes_the_file(self):
        # Another process's PID, or garbage, means the last owner never ran
        # _cleanup. Either way the file then names this process, so the same
        # death is reported once.
        for prev in ("999999", "not-a-pid"):
            with self.subTest(prev=prev), \
                 tempfile.TemporaryDirectory(prefix="brain-claim-") as tmp:
                dc.log_hook_error.reset_mock()
                d = self._bare_daemon(tmp, prev)
                d._claim_pid_file()
                dc.log_hook_error.assert_called_once()
                call = dc.log_hook_error.call_args
                self.assertEqual(call.args[0], "DAEMON_DIED_UNCLEAN")
                self.assertIn(repr(prev), call.args[1])
                self.assertEqual(call.kwargs["db_dir"], tmp)
                self.assertTrue(d._owns_pid_file())
                d._claim_pid_file()           # a second claim finds our own PID
                dc.log_hook_error.assert_called_once()

    def test_claim_after_a_reload_or_a_clean_exit_records_nothing(self):
        # A reload's breadcrumb holds our own PID; a clean exit removed the file.
        for prev in (str(os.getpid()), None):
            with self.subTest(prev=prev), \
                 tempfile.TemporaryDirectory(prefix="brain-claim-") as tmp:
                dc.log_hook_error.reset_mock()
                d = self._bare_daemon(tmp, prev)
                d._claim_pid_file()
                dc.log_hook_error.assert_not_called()
                self.assertTrue(d._owns_pid_file())

    def test_only_a_pid_file_naming_this_process_is_owned(self):
        # _cleanup removes the PID and status files only when this holds.
        for prev in ("999999", None):
            with self.subTest(prev=prev), \
                 tempfile.TemporaryDirectory(prefix="brain-claim-") as tmp:
                self.assertFalse(self._bare_daemon(tmp, prev)._owns_pid_file())

    def test_a_claimed_but_unbound_daemon_reads_as_booting(self):
        # Claiming at the lock gives a loading daemon a fresh PID file, so the
        # corpse test declines to kill it even with no TIME_WAIT on the port
        # (a quiet machine), where it used to read "no PID file, port free".
        with tempfile.TemporaryDirectory(prefix="brain-claim-") as tmp:
            d = self._bare_daemon(tmp, None)
            d._claim_pid_file()
            state = os.path.join(tmp, "recovery.json")
            with patch.object(dl, "get_pid_path", return_value=d.pid_path), \
                 patch.object(dc, "get_recovery_state_path", return_value=state), \
                 patch.object(dc, "is_maintenance_mode", return_value=False), \
                 patch.object(dc, "is_daemon_responsive", return_value=False), \
                 patch.object(dc, "port_is_occupied", return_value=False), \
                 patch.object(dc, "_relaunch_daemon") as relaunch:
                self.assertFalse(dc.recover_daemon())
            relaunch.assert_not_called()

    def test_mcp_proxy_logs_through_the_same_writer(self):
        from servers import brain_mcp
        dc.log_hook_error.reset_mock()
        brain_mcp._log_proxy_error("DAEMON_DOWN", "e", "c", level="critical")
        dc.log_hook_error.assert_called_once_with(
            "DAEMON_DOWN", "e", context="c", level="critical")


class TestKillDaemonLockDiscipline(unittest.TestCase):
    """Step 5: NO code path unlinks a lock file. kill_daemon relies on the
    kernel releasing the dead PID's flock; unlinking the path while another
    holder (a daemon, an ensure_daemon mid-ladder) has it open would let a
    third process lock a fresh inode at the same path — two "singleton"
    holders, the two-writer corruption class."""

    def test_kill_daemon_clears_pid_but_never_unlinks_lock(self):
        import inspect
        import tempfile
        self.assertNotIn("get_lock_path", inspect.getsource(dl.kill_daemon),
                         "kill_daemon must not touch the lock path at all")
        with tempfile.TemporaryDirectory(prefix="brain-kill-test-") as tmp:
            pid_path = os.path.join(tmp, "daemon.pid")
            with open(pid_path, "w") as f:
                f.write("999999")
            with patch.object(dl, "get_pid_path", return_value=pid_path), \
                 patch.object(dl.os, "kill", side_effect=ProcessLookupError), \
                 patch.object(dl.time, "sleep"):
                dl.kill_daemon()
            self.assertFalse(os.path.exists(pid_path),
                             "stale PID hint should still be cleared")


class TestPortProbesOnRealSockets(unittest.TestCase):
    """The premises of the port questions, on real sockets (an ephemeral port,
    never the daemon's). After a clean stop the plain bind still reads the
    port as held (the corpse tests' sign of a restart in flight) while a ping
    is refused (the spawn path's "nothing to kill"). A hung listener times the
    ping out whatever address it bound; an older release's daemon binds
    0.0.0.0, which on macOS a SO_REUSEADDR bind to 127.0.0.1 does not see."""

    def _stopped_daemon_port(self):
        # A listener that served one connection and closed, the server closing
        # first as _handle_client does, so its side sits in TIME_WAIT.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as _bind_socket
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        cli = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        conn.sendall(b"{}\n")
        conn.close()
        while cli.recv(16):
            pass
        cli.close()
        srv.close()
        return port

    def test_plain_bind_reads_a_stopped_daemons_port_as_held(self):
        port = self._stopped_daemon_port()
        with patch.object(dl, "get_daemon_addr", return_value=("127.0.0.1", port)):
            self.assertTrue(dl.port_is_occupied())

    def test_ping_to_a_stopped_daemons_port_is_refused(self):
        port = self._stopped_daemon_port()
        resp = dc.send_command("ping", timeout=1.0, port=port)
        self.assertEqual(resp.get("transport"), "refused")

    def test_ping_to_a_hung_wildcard_listener_times_out(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("0.0.0.0", 0))
        srv.listen(1)           # completes the handshake, never answers
        try:
            resp = dc.send_command("ping", timeout=0.3, port=srv.getsockname()[1])
        finally:
            srv.close()
        self.assertEqual(resp.get("transport"), "timeout")


class TestDaemonConfigSingleSource(unittest.TestCase):
    """Step 1: the launch facts (CPU env, log path, launchd timing) are
    single-sourced in daemon_config so no start path drifts from another."""

    def test_cpu_env_is_the_full_set(self):
        from servers.daemon_config import DAEMON_CPU_ENV
        self.assertEqual(
            set(DAEMON_CPU_ENV),
            {"ORT_DISABLE_ALL_ACCELERATORS", "ONNX_PROVIDERS", "PYTORCH_MPS_DISABLE",
             "VECLIB_MAXIMUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK"},
            "DAEMON_CPU_ENV must carry every var the spawn paths used to hand-list")

    def test_log_path_honors_brain_db_dir(self):
        # The divergence fix: ensure_daemon used to ignore BRAIN_DB_DIR, the
        # restart path honored it — now both go through get_daemon_log_path.
        from servers.daemon_config import get_daemon_log_path
        with patch.dict(os.environ, {"BRAIN_DB_DIR": "/tmp/brain-test-logdir"}):
            self.assertEqual(get_daemon_log_path("/other/place/brain.db"),
                             "/tmp/brain-test-logdir/daemon.log")

    def test_recovery_deadlines_derive_from_throttle(self):
        from servers import daemon_client as dc, daemon_config as cfg
        # Deadlines must clear the plist throttle + reload, or a recovering
        # daemon is mistaken for a corpse and re-kickstarted (the storm).
        self.assertGreater(dc._GRACE_DEADLINE_S, cfg.LAUNCHD_THROTTLE_INTERVAL_S)
        self.assertGreater(dc._KICKSTART_DEADLINE_S, dc._GRACE_DEADLINE_S)


class TestDaemonPlistTemplateContract(unittest.TestCase):
    """Step 7: the repo plist template (hooks/scripts/com.brain.daemon.plist,
    installed per-machine by install-daemon-service.sh) DERIVES its CPU env and
    launchd timing from the Step-1 daemon_config constants. Pins the equality so
    the XML can't silently drift from the Python again — the drift Step 1 named,
    where the hand-installed plist carried a stale 4-var CPU set missing
    PYTORCH_ENABLE_MPS_FALLBACK."""

    def _load(self):
        import plistlib
        path = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'com.brain.daemon.plist')
        with open(path, 'rb') as f:
            raw = f.read()
        return plistlib.loads(raw), raw.decode()

    def test_plist_env_matches_daemon_cpu_env(self):
        from servers.daemon_config import DAEMON_CPU_ENV
        plist, _ = self._load()
        env = plist["EnvironmentVariables"]
        # The env is exactly DAEMON_CPU_ENV plus the per-machine BRAIN_DB_DIR path.
        self.assertEqual(set(env), set(DAEMON_CPU_ENV) | {"BRAIN_DB_DIR"},
                         "plist env must be DAEMON_CPU_ENV + BRAIN_DB_DIR, nothing more/less")
        for k, v in DAEMON_CPU_ENV.items():
            self.assertEqual(env.get(k), v,
                             "plist %s=%r must match DAEMON_CPU_ENV" % (k, v))

    def test_plist_throttle_matches_constant(self):
        from servers.daemon_config import LAUNCHD_THROTTLE_INTERVAL_S
        plist, _ = self._load()
        self.assertEqual(plist["ThrottleInterval"], LAUNCHD_THROTTLE_INTERVAL_S,
                         "plist ThrottleInterval must equal LAUNCHD_THROTTLE_INTERVAL_S")

    def test_plist_is_an_unresolved_template(self):
        # The repo copy is a TEMPLATE — install-daemon-service.sh sed-substitutes
        # the tokens. If a resolved (machine-specific) copy is ever committed, the
        # placeholders vanish and a fresh install materializes wrong paths.
        plist, raw = self._load()
        self.assertIn("__PLUGIN_DIR__", raw)
        self.assertIn("__BRAIN_DB_DIR__", raw)
        self.assertEqual(plist["ProgramArguments"],
                         ["__PLUGIN_DIR__/hooks/scripts/brain-daemon"],
                         "entrypoint must be brain-daemon under the plugin-dir token")
        self.assertTrue(plist["KeepAlive"])
        self.assertTrue(plist["RunAtLoad"])


class TestDbDirDivergence(unittest.TestCase):
    """Step 4b (D-13): the daemon ping reports its db_dir; ensure_daemon treats
    a mismatch vs the session's resolved dir like stale code — kickstart. The
    split-brain killer: after the user adopts a new brain location, a daemon
    launched off the old plist-baked path would keep writing the old brain
    forever (the old brain.db still exists per never-auto-move, so the baked
    path stays 'valid')."""

    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="brain-dbdir-test-")
        self._db = os.path.join(self._dir, "brain.db")
        self._lock = os.path.join(self._dir, "daemon.lock")
        # Never the real wire — see TestEnsureDaemonRoutesThroughLaunchd.setUp.
        self._sc_autopatch = patch.object(
            dc, "send_command", return_value={"ok": True})
        self._sc_mock = self._sc_autopatch.start()
        self.addCleanup(self._sc_autopatch.stop)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def _resp(self, db_dir):
        r = {"ok": True, "result": {}}
        if db_dir is not None:
            r["result"]["db_dir"] = db_dir
        return r

    # ── _db_dir_changed unit ──

    def test_mismatch_detected(self):
        with patch.object(dc, "_is_daemon_source", return_value=True):
            self.assertTrue(dc._db_dir_changed(self._resp("/somewhere/else"), self._db))

    def test_match_is_not_changed(self):
        with patch.object(dc, "_is_daemon_source", return_value=True):
            self.assertFalse(dc._db_dir_changed(self._resp(self._dir), self._db))

    def test_missing_db_dir_never_changed(self):
        # A pre-step-4 daemon doesn't report db_dir → conservative no-restart
        # on this signal (the code-fingerprint check already covers it).
        with patch.object(dc, "_is_daemon_source", return_value=True):
            self.assertFalse(dc._db_dir_changed(self._resp(None), self._db))

    def test_non_source_never_changed(self):
        # Same policy as _code_changed: a non-source checkout can't converge a
        # restart, so it never claims divergence.
        with patch.object(dc, "_is_daemon_source", return_value=False):
            self.assertFalse(dc._db_dir_changed(self._resp("/somewhere/else"), self._db))

    def test_symlinked_same_dir_is_not_changed(self):
        # realpath comparison: a symlink to the same brain is not divergence.
        link = self._dir + "-link"
        os.symlink(self._dir, link)
        try:
            with patch.object(dc, "_is_daemon_source", return_value=True):
                self.assertFalse(dc._db_dir_changed(self._resp(link), self._db))
        finally:
            os.unlink(link)

    # ── ensure_daemon flow ──

    def test_db_dir_mismatch_kickstarts_once(self):
        # Healthy, current code, but writing ANOTHER brain → exactly one
        # launchd kickstart (never a spawn); post-kickstart the daemon reports
        # the session's dir (plist re-materialized at boot + brain-daemon
        # re-ran the ladder) → ready.
        stale = self._resp("/somewhere/else")
        fresh = self._resp(self._dir)
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_can_connect", return_value=stale), \
             patch.object(dc, "_await_responsive", return_value=fresh), \
             patch.object(dc, "kickstart", return_value=True) as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_called_once()
            spawn.assert_not_called()

    def test_db_dir_match_is_noop(self):
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_can_connect", return_value=self._resp(self._dir)), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()

    def test_old_daemon_without_db_dir_is_noop(self):
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_code_changed", return_value=False), \
             patch.object(dc, "_can_connect", return_value=self._resp(None)), \
             patch.object(dc, "kickstart") as ks:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()

    def test_stale_code_with_db_dir_divergence_kickstarts_not_reloads(self):
        # The combined case is the realistic one — you move the brain AND
        # redeploy. A reload execs into the same resolution outcome the
        # session disagrees with; the data move must take the spawn ladder.
        # Pin: reload is NOT attempted, kickstart is.
        stale = self._resp("/somewhere/else")
        fresh = self._resp(self._dir)
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "get_lock_path", return_value=self._lock), \
             patch.object(dc, "get_startup_lock_path", return_value=self._lock + ".startup"), \
             patch.object(dc, "_is_daemon_source", return_value=True), \
             patch.object(dc, "_code_changed", side_effect=[True, True, True, False]), \
             patch.object(dc, "_can_connect", return_value=stale), \
             patch.object(dc, "_await_responsive", return_value=fresh), \
             patch.object(dc, "kickstart", return_value=True) as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            self._sc_mock.assert_not_called()   # no reload request on a data move
            ks.assert_called_once()
            spawn.assert_not_called()

    def test_worktree_client_mismatch_never_kickstarts(self):
        # A non-source checkout stays a pure client even when it sees a
        # diverged daemon — the source checkout's boot converges it.
        with patch.object(dc, "is_maintenance_mode", return_value=False), \
             patch.object(dc, "_is_daemon_source", return_value=False), \
             patch.object(dc, "_can_connect", return_value=self._resp("/somewhere/else")), \
             patch.object(dc, "kickstart") as ks, \
             patch.object(dc, "spawn_detached_daemon") as spawn:
            self.assertTrue(dc.ensure_daemon(self._db))
            ks.assert_not_called()
            spawn.assert_not_called()


class TestLaunchdInstallLockBudget(unittest.TestCase):
    """launchd-install.sh's limits are derived by hand from facts owned
    elsewhere; this pins them. A lock waiter can run inside the SessionStart
    hook, so its wait must end inside that hook's timeout. `launchctl bootout`
    returns at once and launchd SIGKILLs a slow daemon only at the end of its
    SIGTERM→SIGKILL window, so the unload must wait past that window. A lock
    is judged dead after the stale ceiling, so the ceiling must exceed the
    longest a live holder takes: its unload, then the forced-restart record
    and the bootstrap retries."""

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')
    HOOKS = os.path.join(os.path.dirname(__file__), '..', 'hooks')
    LIVE_HOLD_MARGIN_S = 15   # after the unload: record + bootstrap retries, generously

    def _shell_const(self, name):
        import re
        src = open(os.path.join(self.SCRIPTS, "launchd-install.sh")).read()
        m = re.search(r'^%s=(\d+)$' % name, src, re.M)
        self.assertIsNotNone(m, "%s not found in launchd-install.sh" % name)
        return int(m.group(1))

    def test_a_waiter_gives_up_inside_the_session_start_timeout(self):
        wait = self._shell_const("_BRAIN_LAUNCHD_LOCK_WAIT_S")
        for manifest in ("hooks.json", "hooks.codex.json"):
            hooks = json.load(open(os.path.join(self.HOOKS, manifest)))["hooks"]
            timeouts = [h.get("timeout") for grp in hooks.get("SessionStart", [])
                        for h in grp.get("hooks", [])
                        if "boot-brain.sh" in h.get("command", "")]
            self.assertTrue(timeouts, "no boot-brain.sh SessionStart hook in " + manifest)
            for t in timeouts:
                self.assertIsNotNone(t, manifest + " gives boot-brain.sh no timeout")
                self.assertLess(wait, t, "%s gives boot-brain.sh %ss; the install "
                                "lock waits %ss" % (manifest, t, wait))

    def test_an_unload_waits_past_launchds_sigkill_window(self):
        import servers.daemon_config as cfg
        self.assertGreater(self._shell_const("_BRAIN_LAUNCHD_UNLOAD_WAIT_S"),
                           cfg.LAUNCHD_SIGKILL_GRACE_S)

    def test_the_plist_keeps_launchds_default_sigkill_window(self):
        # LAUNCHD_SIGKILL_GRACE_S is launchd's default; ExitTimeOut would move it
        with open(os.path.join(self.SCRIPTS, "com.brain.daemon.plist")) as f:
            self.assertNotIn("ExitTimeOut", f.read())

    def test_a_live_holder_is_never_judged_stale(self):
        self.assertGreater(self._shell_const("_BRAIN_LAUNCHD_LOCK_STALE_S"),
                           self._shell_const("_BRAIN_LAUNCHD_UNLOAD_WAIT_S")
                           + self.LIVE_HOLD_MARGIN_S)


class TestInstallerPlistDrift(unittest.TestCase):
    """Step 4c (D-13): the installers render the plist template to a temp file
    on EVERY run and diff against the installed copy — drift (template evolved,
    or the brain moved) re-materializes + re-bootstraps; no drift leaves launchd
    untouched. Runs the real shell scripts against a fake HOME + fake launchctl."""

    DAEMON_SCRIPT = os.path.join(os.path.dirname(__file__), '..',
                                 'hooks', 'scripts', 'install-daemon-service.sh')
    DASH_SCRIPT = os.path.join(os.path.dirname(__file__), '..',
                               'hooks', 'scripts', 'ensure-dashboard.sh')

    def setUp(self):
        if sys.platform != "darwin":
            self.skipTest("launchd installer paths are macOS-only")
        self._home = tempfile.mkdtemp(prefix="brain-plist-home-")
        self._dbdir = os.path.join(self._home, "brain-data")
        os.makedirs(self._dbdir)
        open(os.path.join(self._dbdir, "brain.db"), "w").close()
        self._agents = os.path.join(self._home, "Library", "LaunchAgents")
        os.makedirs(self._agents)
        self._bin = os.path.join(self._home, "bin")
        os.makedirs(self._bin)
        self._log = os.path.join(self._home, "launchctl.log")
        self._mlock = os.path.join(self._home, "maintenance.lock")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _fake_launchctl(self, print_rc=0, bootout_delay=0, on_bootout="",
                        bootout_async=False):
        """The shared state-modelling stub, seeded by `print_rc` (0 = service
        managed, 1 = fresh machine). The installers verify unload-before-copy
        and load-after-bootstrap, so both transitions must be observable."""
        write_launchctl_stub(self._bin,
                             os.path.join(self._home, "launchctl.loaded"),
                             self._log, loaded=(print_rc == 0),
                             bootout_delay=bootout_delay,
                             on_bootout=on_bootout,
                             bootout_async=bootout_async)

    def _fake_curl(self, code="200"):
        """curl stub for ensure-dashboard's _up probe."""
        path = os.path.join(self._bin, "curl")
        with open(path, "w") as f:
            f.write('#!/bin/bash\nprintf "%s"\n' % code)
        os.chmod(path, 0o755)

    def _env(self, **extra):
        env = dict(os.environ,
                   HOME=self._home,
                   PATH=self._bin + os.pathsep + os.environ.get("PATH", ""),
                   BRAIN_DB_DIR=self._dbdir,
                   XDG_CONFIG_HOME=os.path.join(self._home, ".config"),
                   # never the machine's real maintenance lock
                   BRAIN_MAINTENANCE_LOCK=self._mlock)
        # an entity env in the developer's shell must not flip these tests —
        # the entity behaviour has its own tests below, which pass it explicitly
        env.pop("BRAIN_INSTANCE", None)
        env.update(extra)
        return env

    def _run(self, script, **extra):
        return subprocess.run(["bash", script], env=self._env(**extra),
                              capture_output=True, text=True, timeout=60)

    def _run_together(self, script, n=2):
        """`n` runs started at once — a session's installer beside the cold
        chain's. Returns (returncode, stderr) per run."""
        procs = [subprocess.Popen(["bash", script], env=self._env(),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
                 for _ in range(n)]
        return [(p.returncode, err) for p, (_, err) in
                ((p, p.communicate(timeout=60)) for p in procs)]

    def _lock_dir(self, label):
        return os.path.join(self._home, "Library", "Caches",
                            "%s.install.lock" % label)

    def _calls(self):
        return read_launchctl_calls(self._log)

    def _verbs(self):
        return [c[0] for c in self._calls()]

    # ── install-daemon-service.sh ──

    def test_daemon_drift_rematerializes_and_rebootstraps(self):
        self._fake_launchctl(print_rc=0)  # managed
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content, "re-materialized plist must carry the resolved brain dir")
        self.assertNotIn("__BRAIN_DB_DIR__", content)
        self.assertNotIn("__PLUGIN_DIR__", content)
        verbs = self._verbs()
        self.assertIn("bootout", verbs, "drift must re-bootstrap, not kickstart (env is frozen at bootstrap)")
        self.assertIn("bootstrap", verbs)

    def test_daemon_no_drift_leaves_launchd_untouched(self):
        self._fake_launchctl(print_rc=0)
        with open(os.path.join(self._agents, "com.brain.daemon.plist"), "w") as f:
            f.write("seed")
        self._run(self.DAEMON_SCRIPT)  # first run converges the target
        os.remove(self._log)
        r = self._run(self.DAEMON_SCRIPT)  # second run: no drift
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(set(self._verbs()), {"print"},
                         "no drift → only the managed probe, no bootout/bootstrap")

    def _forced_restart_rows(self):
        import sqlite3
        db = os.path.join(self._dbdir, "brain_logs.db")
        if not os.path.exists(db):
            return []
        conn = sqlite3.connect(db)
        try:
            return conn.execute("SELECT hook_name, error, context FROM hook_errors "
                                "WHERE hook_name = 'DAEMON_RESTART_FORCED'").fetchall()
        finally:
            conn.close()

    def test_bootout_of_the_running_daemon_is_recorded(self):
        self._fake_launchctl(print_rc=0)  # managed: the reinstall stops it
        with open(os.path.join(self._agents, "com.brain.daemon.plist"), "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = self._forced_restart_rows()
        self.assertEqual(len(rows), 1, (rows, r.stderr))
        self.assertEqual(rows[0][1], "launchctl bootout com.brain.daemon")
        # the shell's own evidence: nothing read the live PID file or port
        self.assertTrue(rows[0][2].startswith(
            "the service was loaded; bootout unloaded it; caller [install-daemon-service] pid "),
            rows[0][2])

    def test_a_bootout_that_stops_no_daemon_is_not_recorded(self):
        # A fresh install boots nothing out; the dashboard is not the daemon.
        self._fake_launchctl(print_rc=1)
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        with open(os.path.join(self._agents, "com.brain.dashboard.plist"), "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("bootout", self._verbs())
        self.assertEqual(self._forced_restart_rows(), [])

    def test_daemon_fresh_install_bootstraps_without_bootout(self):
        self._fake_launchctl(print_rc=1)  # not managed
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        self.assertTrue(os.path.exists(target))
        self.assertIn(self._dbdir, open(target).read())
        verbs = self._verbs()
        self.assertIn("bootstrap", verbs)
        self.assertNotIn("bootout", verbs)

    # ── concurrent runs: one install at a time per label ──
    # Every session runs install-daemon-service.sh, and a cold one also runs
    # it detached after the runtime bootstrap, so two runs can find the same
    # drift together. Only one of them may boot the service out.

    def _stale_daemon_plist(self):
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        return target

    def test_concurrent_runs_boot_the_daemon_out_once(self):
        self._fake_launchctl(print_rc=0, bootout_delay=2)
        self._stale_daemon_plist()
        runs = self._run_together(self.DAEMON_SCRIPT)
        for rc, err in runs:
            self.assertEqual(rc, 0, err)  # the waiter verified too
        verbs = self._verbs()
        self.assertEqual(verbs.count("bootout"), 1, self._calls())
        self.assertEqual(verbs.count("bootstrap"), 1, self._calls())
        self.assertEqual(len(self._forced_restart_rows()), 1)
        self.assertFalse(os.path.exists(self._lock_dir("com.brain.daemon")))

    def test_concurrent_dashboard_runs_boot_it_out_once(self):
        self._fake_launchctl(print_rc=0, bootout_delay=2)
        self._fake_curl("200")
        with open(os.path.join(self._agents, "com.brain.dashboard.plist"), "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        runs = self._run_together(self.DASH_SCRIPT)
        for rc, err in runs:
            self.assertEqual(rc, 0, err)
        self.assertEqual(self._verbs().count("bootout"), 1, self._calls())

    def test_a_waiter_from_another_tree_keeps_the_first_install(self):
        # Two trees (Claude Code's and Codex's copy of one release) raced: the
        # other tree rendered before this tree's install landed. Under the lock
        # it must render again, find the installed tree kept by the owner rule,
        # and leave launchd alone rather than install its stale render.
        self._fake_launchctl(print_rc=0)
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write("seed")
        setup = self._run(self.DAEMON_SCRIPT)  # this tree's install landed
        self.assertEqual(setup.returncode, 0, setup.stderr)
        installed = open(target).read()
        self.assertIn(self._dbdir, installed)
        os.remove(self._log)
        other = self._fake_owner_tree()
        sd = os.path.dirname(self.DAEMON_SCRIPT)
        stale = os.path.join(self._home, "rendered.plist")
        with open(os.path.join(sd, "com.brain.daemon.plist")) as f:
            tmpl = f.read()
        with open(stale, "w") as f:
            f.write(tmpl.replace("__PLUGIN_DIR__", other)
                        .replace("__BRAIN_DB_DIR__", self._dbdir))
        r = subprocess.run(
            ["bash", "-c",
             'source "$1/launchd-install.sh"; '
             'brain_launchd_install com.brain.daemon "$1/com.brain.daemon.plist" '
             'brain-daemon "$2" "$3" "$4" "[test]"',
             "_", sd, other, self._dbdir, stale],
            env=self._env(), capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("already installed and loaded", r.stderr)
        self.assertEqual([v for v in self._verbs() if v != "print"], [],
                         self._calls())
        self.assertEqual(open(target).read(), installed)

    def test_a_lock_left_by_a_dead_run_is_taken_over(self):
        self._fake_launchctl(print_rc=0)
        target = self._stale_daemon_plist()
        lock = self._lock_dir("com.brain.daemon")
        os.makedirs(lock)
        long_ago = time.time() - 600
        os.utime(lock, (long_ago, long_ago))
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(self._dbdir, open(target).read())
        self.assertIn("bootstrap", self._verbs())
        self.assertFalse(os.path.exists(lock))

    def test_a_gnu_stat_in_path_does_not_break_the_lock(self):
        # Homebrew's gnubin puts GNU stat first: `-f` means file-system mode
        # there, and its text in the age arithmetic aborted the whole script.
        self._fake_launchctl(print_rc=0)
        with open(os.path.join(self._bin, "stat"), "w") as f:
            f.write('#!/bin/bash\n'
                    'if [ "$1" = -f ]; then\n'
                    '  echo "  File: \\"$3\\""; echo "    ID: 1 Namelen: 255 Type: apfs"; exit 1\n'
                    'fi\n'
                    'exec /usr/bin/stat "$@"\n')
        os.chmod(os.path.join(self._bin, "stat"), 0o755)
        self._stale_daemon_plist()
        lock = self._lock_dir("com.brain.daemon")
        os.makedirs(lock)
        long_ago = time.time() - 600
        os.utime(lock, (long_ago, long_ago))
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("syntax error", r.stderr)
        self.assertIn("bootstrap", self._verbs())
        self.assertFalse(os.path.exists(lock))

    def test_a_run_that_cannot_get_the_lock_leaves_launchd_alone(self):
        # A live holder (fresh lock) that outlasts the wait: the run gives up
        # rather than boot out a service another run is installing. The wait
        # is shortened here; the scripts run with the owner's default.
        self._fake_launchctl(print_rc=0)
        target = self._stale_daemon_plist()
        lock = self._lock_dir("com.brain.daemon")
        os.makedirs(lock)
        sd = os.path.dirname(self.DAEMON_SCRIPT)
        r = subprocess.run(
            ["bash", "-c",
             'source "$1/launchd-install.sh"; _BRAIN_LAUNCHD_LOCK_WAIT_S=1; '
             'brain_launchd_install com.brain.daemon "$1/com.brain.daemon.plist" '
             'brain-daemon "$2" "$3" "$4" "[test]"',
             "_", sd, os.path.dirname(os.path.dirname(sd)), self._dbdir,
             os.path.join(self._home, "rendered.plist")],
            env=self._env(), capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("could not take the com.brain.daemon install lock", r.stderr)
        self.assertEqual(self._verbs(), [], "no launchctl call without the lock")
        self.assertEqual(open(target).read(), "<plist>stale frozen snapshot</plist>\n")
        self.assertTrue(os.path.isdir(lock), "the holder's lock is not ours to remove")

    # ── the SessionStart path. A drift reinstall waits for the running daemon to
    # exit, which can take longer than the hook allows, so --boot hands it to a
    # background run that starts once the booting session is done.

    def _run_boot(self):
        # as boot-brain.sh calls it; the installer's parent stands in for the
        # booting hook (`; exit` stops bash exec'ing the installer in its place)
        return subprocess.run(["bash", "-c", 'bash "$1" --boot; exit $?',
                               "_", self.DAEMON_SCRIPT],
                              env=self._env(), capture_output=True, text=True,
                              timeout=60)

    def test_a_slow_shutdown_still_completes_the_reinstall(self):
        # bootout returns at once and the daemon outlasts a 5s poll
        self._fake_launchctl(print_rc=0, bootout_delay=6, bootout_async=True)
        target = self._stale_daemon_plist()
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("did not unload", r.stderr)
        self.assertIn("bootstrap", self._verbs())
        self.assertIn(self._dbdir, open(target).read())

    # The handoff itself (exit 3, the background run, boot-brain.sh skipping
    # ensure_daemon) runs end to end in TestBootImportsPluginTreeNotCwd's tree.

    def test_boot_hands_nothing_off_while_the_maintenance_lock_is_held(self):
        # the background run would refuse; the boot says so instead of
        # promising a reinstall, and ensure_daemon (exit 1, not 3) still runs
        self._fake_launchctl(print_rc=0)
        self._stale_daemon_plist()
        open(self._mlock, "w").close()
        r = self._run_boot()
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("maintenance lock", r.stderr)
        self.assertNotIn("reinstalling in the background", r.stderr)

    def test_boot_installs_a_first_run_in_the_foreground(self):
        # nothing is running to stop, and ensure_daemon, next in the boot,
        # must find the service loaded
        self._fake_launchctl(print_rc=1)
        r = self._run_boot()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("bootstrap", self._verbs())
        self.assertNotIn("reinstalling in the background", r.stderr)

    def test_the_background_run_waits_for_the_boot_to_finish(self):
        self._fake_launchctl(print_rc=0)
        self._stale_daemon_plist()
        # orphaned, so it is reaped the moment it exits (a child of this test
        # would linger as a zombie that `kill -0` still sees)
        boot_pid = subprocess.check_output(
            ["bash", "-c", "sleep 2 >/dev/null 2>&1 & echo $!"], text=True).strip()
        started = time.time()
        r = subprocess.run(["bash", self.DAEMON_SCRIPT, "--after-pid", boot_pid],
                           env=self._env(), capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreaterEqual(time.time() - started, 1.5)
        self.assertIn("bootstrap", self._verbs())

    # ── the maintenance lock: whoever holds it has stopped the daemon to work
    # on the brain's files (relocate-brain.sh copies them) and restores
    # services itself. A bootstrap meanwhile starts a daemon on those files.

    def test_maintenance_lock_blocks_the_install_relocate_left_unloaded(self):
        # relocate booted the service out and left the plist on disk
        self._fake_launchctl(print_rc=1)
        target = self._stale_daemon_plist()
        open(self._mlock, "w").close()
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("maintenance lock %s is held" % self._mlock, r.stderr)
        self.assertNotIn("bootstrap", self._verbs())
        self.assertNotIn("load", self._verbs())
        self.assertEqual(open(target).read(), "<plist>stale frozen snapshot</plist>\n")

    def test_maintenance_lock_leaves_a_loaded_drifted_service_running(self):
        # a maintenance operation that keeps the service loaded: drift must not
        # boot the daemon out from under it
        self._fake_launchctl(print_rc=0)
        target = self._stale_daemon_plist()
        open(self._mlock, "w").close()
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual([v for v in self._verbs() if v != "print"], [],
                         self._calls())
        self.assertEqual(open(target).read(), "<plist>stale frozen snapshot</plist>\n")

    def test_maintenance_lock_taken_during_the_unload_blocks_the_bootstrap(self):
        # a relocation starts while this run waits out the daemon's shutdown
        self._fake_launchctl(print_rc=0, on_bootout='touch "%s"' % self._mlock)
        target = self._stale_daemon_plist()
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 1, r.stderr)
        verbs = self._verbs()
        self.assertIn("bootout", verbs)
        self.assertNotIn("bootstrap", verbs,
                         "the service stays down for the lock's holder to restore")
        self.assertEqual(open(target).read(), "<plist>stale frozen snapshot</plist>\n",
                         "the plist stays as the holder left it")
        self.assertFalse(os.path.exists(self._lock_dir("com.brain.daemon")))

    def test_maintenance_lock_blocks_the_dashboard_install(self):
        self._fake_launchctl(print_rc=1)
        self._fake_curl("000")
        open(self._mlock, "w").close()
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("maintenance lock", r.stderr)
        self.assertNotIn("bootstrap", self._verbs())
        self.assertFalse(os.path.exists(
            os.path.join(self._agents, "com.brain.dashboard.plist")))

    def test_maintenance_lock_path_matches_daemon_config(self):
        # the shell cannot import the owner, so its spelling is pinned here
        env = self._env()
        env.pop("BRAIN_MAINTENANCE_LOCK")
        r = subprocess.run(
            ["bash", "-c", 'source "$1"; brain_maintenance_lock',
             "_", os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                               "launchd-install.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAIN_INSTANCE", None)
            import servers.daemon_config as cfg
            self.assertEqual(r.stdout, cfg.get_maintenance_path())

    # ── ensure-dashboard.sh ──

    def test_dashboard_drift_rematerializes_even_when_up(self):
        # An "up" dashboard whose plist points at a moved brain serves the
        # WRONG data — drift must be fixed before the up fast-path exits.
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        target = os.path.join(self._agents, "com.brain.dashboard.plist")
        with open(target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content)
        self.assertNotIn("__BRAIN_DB_DIR__", content)
        verbs = self._verbs()
        self.assertIn("bootout", verbs)
        self.assertIn("bootstrap", verbs)

    def test_dashboard_no_drift_up_is_noop(self):
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        target = os.path.join(self._agents, "com.brain.dashboard.plist")
        with open(target, "w") as f:
            f.write("seed")
        self._run(self.DASH_SCRIPT)  # converge
        os.remove(self._log)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(set(self._verbs()), {"print"},
                         "up + no drift → early exit, launchd untouched")

    # ── entities never touch launchd (predicate owned by launchd-install.sh) ──
    # The loaded service is uid-wide, the plist path is $HOME-relative: an
    # entity under a scratch $HOME would read "no plist here" as drift and
    # bootout production's job. Both installers must exit before launchctl.

    def test_dashboard_under_instance_touches_no_launchd(self):
        self._fake_launchctl(print_rc=0)
        self._fake_curl("000")
        target = os.path.join(self._agents, "com.brain.dashboard.plist")
        with open(target, "w") as f:
            f.write("<plist>production</plist>\n")
        r = self._run(self.DASH_SCRIPT, BRAIN_INSTANCE="ent1",
                      BRAIN_DAEMON_PORT="47999")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._verbs(), [], "an entity must never reach launchctl")
        self.assertEqual(open(target).read(), "<plist>production</plist>\n")

    def test_daemon_installer_under_instance_touches_no_launchd(self):
        self._fake_launchctl(print_rc=1)
        r = self._run(self.DAEMON_SCRIPT, BRAIN_INSTANCE="ent1",
                      BRAIN_DAEMON_PORT="47999")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self._verbs(), [])
        self.assertFalse(os.path.exists(
            os.path.join(self._agents, "com.brain.daemon.plist")))

    # ── identity preservation (review findings: tree-flip / db-dir hijack) ──

    def _fake_owner_tree(self, rel="other-tree"):
        """A second 'checkout' that owns the installed plist — valid launcher
        scripts under a tree that is NOT the one running the installer."""
        tree = os.path.join(self._home, rel)
        scripts = os.path.join(tree, "hooks", "scripts")
        os.makedirs(scripts)
        for name in ("brain-daemon", "brain-dashboard"):
            p = os.path.join(scripts, name)
            with open(p, "w") as f:
                f.write("#!/bin/bash\n")
            os.chmod(p, 0o755)
        return tree

    def test_daemon_drift_preserves_installed_plugin_dir(self):
        # Hooks run from the plugin copy while the daemon is launchd-pinned to
        # the repo: re-materialization must keep the INSTALLED tree, not flip
        # service ownership to whichever tree ran the installer.
        self._fake_launchctl(print_rc=0)
        owner = self._fake_owner_tree()
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        rendered = open(template).read().replace("__PLUGIN_DIR__", owner) \
                                        .replace("__BRAIN_DB_DIR__", self._dbdir)
        with open(target, "w") as f:
            f.write(rendered + "\n<!-- old template residue -->\n")  # force drift
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(owner + "/hooks/scripts/brain-daemon", content,
                      "installed tree must survive re-materialization")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DAEMON_SCRIPT), "..", ".."))
        self.assertNotIn(repo + "/hooks/scripts/brain-daemon", content,
                         "the caller's tree must not capture the service")

    def test_daemon_legacy_launcher_name_still_preserves_tree(self):
        # An installed plist from BEFORE the start-daemon.sh → brain-daemon
        # rename: extraction is launcher-name-agnostic, validity checks the
        # launcher the CURRENT template ships — so the old tree is preserved
        # and the re-materialized plist execs the new launcher in it.
        self._fake_launchctl(print_rc=0)
        owner = self._fake_owner_tree()  # has brain-daemon, not start-daemon.sh
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        legacy = open(template).read() \
            .replace("__PLUGIN_DIR__/hooks/scripts/brain-daemon",
                     "__PLUGIN_DIR__/hooks/scripts/start-daemon.sh") \
            .replace("__PLUGIN_DIR__", owner) \
            .replace("__BRAIN_DB_DIR__", self._dbdir)
        with open(target, "w") as f:
            f.write(legacy)
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(owner + "/hooks/scripts/brain-daemon", content,
                      "old tree preserved, new launcher name materialized")
        self.assertNotIn("start-daemon.sh", content)
        self.assertIn("bootout", self._verbs())

    def test_daemon_foreign_owned_current_plist_is_no_drift(self):
        # Same template shape, different (valid) owner tree → NOT drift.
        # Without this, plugin-copy boots and repo-side runs would ping-pong
        # the plist with a daemon bootout on every flip.
        self._fake_launchctl(print_rc=0)
        owner = self._fake_owner_tree()
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", owner)
                                         .replace("__BRAIN_DB_DIR__", self._dbdir))
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(set(self._verbs()), {"print"},
                         "foreign-owned but current plist must not re-bootstrap")

    # ── release supersession: a plugin update installs a NEW version dir ──
    # Claude Code keeps the previous <plugin>/<version> dir (launcher still
    # executable); Codex deletes it. Both hosts share this one service, so the
    # owner is decided by version order alone: a strictly newer release of
    # the same plugin takes over, anything else keeps the installed tree.

    def _versioned_caller(self, rel):
        """The installer as a host runs it: from <...>/<plugin>/<version>/,
        here a symlink to this checkout so the real scripts execute."""
        tree = os.path.join(self._home, rel)
        os.makedirs(os.path.dirname(tree), exist_ok=True)
        os.symlink(os.path.realpath(os.path.join(
            os.path.dirname(self.DAEMON_SCRIPT), "..", "..")), tree)
        return tree

    def _write_installed_plist(self, label, tree):
        target = os.path.join(self._agents, label + ".plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT), label + ".plist")
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", tree)
                                         .replace("__BRAIN_DB_DIR__", self._dbdir))
        return target

    def _run_from(self, tree, script, **extra):
        return self._run(os.path.join(tree, "hooks", "scripts",
                                      os.path.basename(script)), **extra)

    def test_daemon_newer_release_replaces_superseded_tree(self):
        # `claude plugin update` 0.9.0 → 0.9.1: 0.9.0 stays on disk with its
        # launcher, so the old keep-while-executable rule pinned the service
        # to the previous release forever.
        self._fake_launchctl(print_rc=0)
        old = self._fake_owner_tree(".claude/plugins/cache/anchor/entity/0.9.0")
        new = self._versioned_caller(".claude/plugins/cache/anchor/entity/0.9.1")
        target = self._write_installed_plist("com.brain.daemon", old)
        r = self._run_from(new, self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(new + "/hooks/scripts/brain-daemon", content)
        self.assertNotIn(old + "/", content)
        verbs = self._verbs()
        self.assertIn("bootout", verbs)
        self.assertIn("bootstrap", verbs)

    def test_daemon_newer_release_from_the_other_host_takes_over(self):
        # The rule reads <plugin>/<version> and names neither host: a Codex
        # 0.9.1 session replaces a Claude Code 0.9.0 tree like its own.
        self._fake_launchctl(print_rc=0)
        old = self._fake_owner_tree(".claude/plugins/cache/anchor/entity/0.9.0")
        new = self._versioned_caller(".codex/plugins/cache/anchor-dev/entity/0.9.1")
        target = self._write_installed_plist("com.brain.daemon", old)
        r = self._run_from(new, self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(new + "/hooks/scripts/brain-daemon", open(target).read())
        self.assertIn("bootout", self._verbs())

    def test_daemon_other_hosts_equal_release_is_kept(self):
        # Both hosts on 0.9.1, each running its own copy: whichever tree the
        # service has stays. Re-pointing here would restart the daemon on
        # every session of the other host (ping-pong).
        self._fake_launchctl(print_rc=0)
        other = self._fake_owner_tree(".codex/plugins/cache/anchor-dev/entity/0.9.1")
        mine = self._versioned_caller(".claude/plugins/cache/anchor/entity/0.9.1")
        target = self._write_installed_plist("com.brain.daemon", other)
        r = self._run_from(mine, self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(other + "/hooks/scripts/brain-daemon", open(target).read())
        self.assertEqual(set(self._verbs()), {"print"})

    def test_daemon_older_release_never_takes_the_service_back(self):
        # A host that has not updated yet must not move the daemon down to
        # its older code — that code would open a brain the newer release
        # already migrated.
        self._fake_launchctl(print_rc=0)
        newer = self._fake_owner_tree(".codex/plugins/cache/anchor-dev/entity/0.9.2")
        older = self._versioned_caller(".claude/plugins/cache/anchor/entity/0.9.1")
        target = self._write_installed_plist("com.brain.daemon", newer)
        r = self._run_from(older, self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(newer + "/hooks/scripts/brain-daemon", open(target).read())
        self.assertEqual(set(self._verbs()), {"print"})

    def test_daemon_repo_pin_survives_a_versioned_caller(self):
        # The dev pin: daemon launchd-pinned to a checkout while hooks run
        # from an installed release (Codex's cache copy on the dev machine).
        # A checkout carries no version, so no release supersedes it.
        self._fake_launchctl(print_rc=0)
        repo = self._fake_owner_tree("brain")
        release = self._versioned_caller(".codex/plugins/cache/anchor-dev/entity/0.9.1")
        target = self._write_installed_plist("com.brain.daemon", repo)
        r = self._run_from(release, self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(repo + "/hooks/scripts/brain-daemon", open(target).read())
        self.assertEqual(set(self._verbs()), {"print"})

    def test_dashboard_newer_release_replaces_superseded_tree(self):
        # ensure-dashboard.sh shares the render: an up dashboard on the
        # previous release must be re-materialized, not left serving it.
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        old = self._fake_owner_tree(".claude/plugins/cache/anchor/entity/0.9.0")
        new = self._versioned_caller(".claude/plugins/cache/anchor/entity/0.9.1")
        target = self._write_installed_plist("com.brain.dashboard", old)
        r = self._run_from(new, self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(new + "/hooks/scripts/brain-dashboard", content)
        self.assertNotIn(old + "/", content)
        self.assertIn("bootout", self._verbs())

    def test_daemon_env_override_does_not_repoint_db_dir(self):
        # An ephemeral BRAIN_DB_DIR (eval run, isolated copy) must not hijack
        # the managed daemon onto its brain: installed BRAIN_DB_DIR wins
        # unless the override came from a durable adoption channel.
        self._fake_launchctl(print_rc=0)
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DAEMON_SCRIPT), "..", ".."))
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", repo)
                                         .replace("__BRAIN_DB_DIR__", prod))
        # self._dbdir (the run's BRAIN_DB_DIR) differs from prod → would be a
        # hijack; no knob, no plugin option → render must keep prod.
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(prod, open(target).read(),
                      "ephemeral env override must not re-point the daemon's brain")
        self.assertEqual(set(self._verbs()), {"print"})

    def test_daemon_knob_adoption_repoints_db_dir(self):
        # The durable channel: BRAIN_DB_DIR in ~/.config/brain/env IS a
        # sanctioned adoption — the plist must converge to it.
        self._fake_launchctl(print_rc=0)
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        cfg = os.path.join(self._home, ".config", "brain")
        os.makedirs(cfg)
        with open(os.path.join(cfg, "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._dbdir)
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        template = os.path.join(os.path.dirname(self.DAEMON_SCRIPT),
                                "com.brain.daemon.plist")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DAEMON_SCRIPT), "..", ".."))
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", repo)
                                         .replace("__BRAIN_DB_DIR__", prod))
        r = self._run(self.DAEMON_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content, "knob adoption must re-point the plist")
        self.assertNotIn(prod, content)
        self.assertIn("bootout", self._verbs())

    # The identity guards below existed for the daemon only — the asymmetry
    # step 6b removed by giving both installers one implementation
    # (launchd-install.sh). They are pinned on the DASHBOARD side because that
    # is the copy that had drifted.

    def _install_dashboard_plist(self, db_dir):
        target = os.path.join(self._agents, "com.brain.dashboard.plist")
        template = os.path.join(os.path.dirname(self.DASH_SCRIPT),
                                "com.brain.dashboard.plist")
        repo = os.path.realpath(os.path.join(os.path.dirname(self.DASH_SCRIPT), "..", ".."))
        with open(target, "w") as f:
            f.write(open(template).read().replace("__PLUGIN_DIR__", repo)
                                         .replace("__BRAIN_DB_DIR__", db_dir))
        return target

    def test_dashboard_env_override_does_not_repoint_db_dir(self):
        # An ephemeral BRAIN_DB_DIR must not hijack the singleton dashboard
        # onto a temp brain any more than it may hijack the daemon.
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        target = self._install_dashboard_plist(prod)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(prod, open(target).read(),
                      "ephemeral env override must not re-point the dashboard's brain")
        self.assertEqual(set(self._verbs()), {"print"})

    def test_dashboard_knob_adoption_repoints_db_dir(self):
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        cfg = os.path.join(self._home, ".config", "brain")
        os.makedirs(cfg, exist_ok=True)
        with open(os.path.join(cfg, "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._dbdir)
        target = self._install_dashboard_plist(prod)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        content = open(target).read()
        self.assertIn(self._dbdir, content, "knob adoption must re-point the plist")
        self.assertNotIn(prod, content)
        self.assertIn("bootout", self._verbs())

    def test_knob_read_ignores_env_file_stdout(self):
        # The env file is shell grammar, so a user `echo`/banner line in it
        # writes to the reader's stdout. The dashboard's copy of the knob read
        # lacked the stdout discard until step 6b unified it: the polluted
        # value then failed the "is this a durable adoption?" comparison and
        # the installer silently kept the OLD brain despite a valid knob.
        self._fake_launchctl(print_rc=0)
        self._fake_curl("200")
        prod = os.path.join(self._home, "prod-brain")
        os.makedirs(prod)
        cfg = os.path.join(self._home, ".config", "brain")
        os.makedirs(cfg, exist_ok=True)
        with open(os.path.join(cfg, "env"), "w") as f:
            f.write("echo 'loading brain env'\nBRAIN_DB_DIR='%s'\n" % self._dbdir)
        target = self._install_dashboard_plist(prod)
        r = self._run(self.DASH_SCRIPT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(self._dbdir, open(target).read(),
                      "a chatty env file must not defeat knob adoption")

    def test_daemon_bootout_failure_keeps_installed_plist(self):
        # bootout that doesn't unload → the plist FILE must stay stale so the
        # next run re-detects drift ("file current, launchd stale" never heals).
        path = os.path.join(self._bin, "launchctl")
        with open(path, "w") as f:
            f.write('#!/bin/bash\necho "$@" >> "%s"\n'
                    'case "$1" in print) exit 0;; esac\nexit 0\n' % self._log)
        os.chmod(path, 0o755)  # print always managed; bootout never unloads
        target = os.path.join(self._agents, "com.brain.daemon.plist")
        with open(target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        # the owner's ritual with its unload wait shortened: the script would sit
        # out the full _BRAIN_LAUNCHD_UNLOAD_WAIT_S for a bootout that never lands
        sd = os.path.dirname(self.DAEMON_SCRIPT)
        r = subprocess.run(
            ["bash", "-c",
             'source "$1/launchd-install.sh"; _BRAIN_LAUNCHD_UNLOAD_WAIT_S=1; '
             'brain_launchd_install com.brain.daemon "$1/com.brain.daemon.plist" '
             'brain-daemon "$2" "$3" "$4" "[test]"',
             "_", sd, os.path.dirname(os.path.dirname(sd)), self._dbdir,
             os.path.join(self._home, "rendered.plist")],
            env=self._env(), capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0, "must fail loudly, not bless the divergence")
        self.assertIn("bootout did not unload", r.stderr)
        self.assertEqual(open(target).read(), "<plist>stale frozen snapshot</plist>\n",
                         "installed plist must be untouched after a failed unload")


class TestResolverEnvHintDemotion(unittest.TestCase):
    """Step 4a refinement (review finding): a BRAIN_DB_DIR env dir WITHOUT
    brain.db is a hint of last resort, not a verdict — a daemon relaunched off
    a stale plist-baked path follows the ladder to the moved brain instead of
    birthing a shadow brain, while a fresh install (dir exists, daemon creates
    brain.db) still lands on the hint. An entity (BRAIN_INSTANCE) never walks
    the ladder at all (the tests at the end)."""

    RESOLVER = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'resolve-brain-db.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-resolve-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _resolve(self, env_extra):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        # production's ladder: no instance unless a test passes one
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA", "BRAIN_INSTANCE"):
            env.pop(k, None)
        env.update(env_extra)
        # cwd in the sandbox: a relative path must never land in the repo
        out = subprocess.run(
            ["bash", "-c",
             'source "%s" >/dev/null 2>&1; printf %%s "$BRAIN_DB_DIR"' % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60, cwd=self._home)
        return out.stdout.strip()

    def test_moved_brain_follows_resolved_env_over_stale_hint(self):
        stale = os.path.join(self._home, "old-brain")   # dir exists, no brain.db
        os.makedirs(stale)
        new = os.path.join(self._home, "new-brain")
        os.makedirs(new)
        open(os.path.join(new, "brain.db"), "w").close()
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % new)
        self.assertEqual(self._resolve({"BRAIN_DB_DIR": stale}), new,
                         "the record of where the brain moved must beat the stale baked hint")

    def test_fresh_install_hint_survives(self):
        hint = os.path.join(self._home, "fresh-brain")  # installer mkdir'd it
        os.makedirs(hint)
        self.assertEqual(self._resolve({"BRAIN_DB_DIR": hint}), hint,
                         "fresh install: nothing else exists → the baked hint wins")

    def test_hint_with_brain_db_is_adopted_outright(self):
        d = os.path.join(self._home, "live-brain")
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        self.assertEqual(self._resolve({"BRAIN_DB_DIR": d}), d)

    def test_no_persist_guard_skips_resolved_env_write(self):
        d = os.path.join(self._home, "live-brain")
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        self._resolve({"BRAIN_DB_DIR": d, "BRAIN_RESOLVE_NO_PERSIST": "1"})
        self.assertFalse(
            os.path.exists(os.path.join(self._xdg, "brain", "resolved.env")),
            "a NO_PERSIST consumer (brain-daemon) must not write the record")

    def _legacy_brain(self):
        # where the ladder finds the operator's brain on the real machine
        d = os.path.join(self._home, "AgentsContext", "brain")
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        return d

    def test_entity_never_adopts_the_brain_the_ladder_finds(self):
        # the test suite's shape: an empty per-run dir beside a real brain
        legacy = self._legacy_brain()
        mine = os.path.join(self._home, "entity-brain")
        os.makedirs(mine)
        self.assertEqual(
            self._resolve({"BRAIN_DB_DIR": mine, "BRAIN_INSTANCE": "ent1"}), mine,
            "an entity that names its dir must not walk onto another brain")
        self.assertEqual(self._resolve({"BRAIN_DB_DIR": mine}), legacy,
                         "control: without the instance the empty dir is a hint")

    def test_entity_dir_is_created_as_its_birthplace(self):
        born = os.path.join(self._home, "not-yet", "brain")
        self.assertEqual(
            self._resolve({"BRAIN_DB_DIR": born, "BRAIN_INSTANCE": "ent1"}), born)
        self.assertTrue(os.path.isdir(born))

    def test_entity_without_an_absolute_dir_has_no_brain(self):
        # the brain the ladder would find is there; an entity never takes it
        self._legacy_brain()
        cases = {
            "unset": {},
            # relative would resolve against each consumer's cwd
            "relative": {"BRAIN_DB_DIR": "rel-brain"},
            # the plugin setting names the operator's brain
            "plugin option": {"CLAUDE_PLUGIN_OPTION_BRAIN_PATH":
                              os.path.join(self._home, "opt-brain")},
        }
        for name, extra in cases.items():
            with self.subTest(name):
                self.assertEqual(
                    self._resolve(dict(extra, BRAIN_INSTANCE="ent1")), "")
        self.assertFalse(os.path.exists(os.path.join(self._home, "rel-brain")))
        self.assertFalse(os.path.exists(os.path.join(self._home, "opt-brain")))

    def test_entity_never_writes_resolved_env(self):
        d = os.path.join(self._home, "entity-brain")
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        self.assertEqual(
            self._resolve({"BRAIN_DB_DIR": d, "BRAIN_INSTANCE": "ent1"}), d)
        self.assertFalse(
            os.path.exists(os.path.join(self._xdg, "brain", "resolved.env")),
            "resolved.env records the operator's brain, not an entity's")


class TestResolverMkdirsSafeUnderSetE(unittest.TestCase):
    """The resolver is sourced under the daemon launcher's `set -e`, and its
    mkdirs are best-effort: a read-only parent must degrade to the next rung
    (or to the existing no-brain path), never abort the whole resolver — that
    would take the daemon down with it. Covers the persist-state dir and the
    plugin-option hint; the Cowork-mount and config-hint mkdirs share the same
    `|| true` shape but need /sessions or rung isolation to exercise."""

    RESOLVER = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'resolve-brain-db.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-resolve-home-")
        self._xdg = os.path.join(self._home, ".config")
        self._ro = []

    def tearDown(self):
        import shutil
        for d in self._ro:
            os.chmod(d, 0o755)
        shutil.rmtree(self._home, ignore_errors=True)

    def _read_only(self, path):
        os.chmod(path, 0o555)
        self._ro.append(path)

    def _shells(self):
        # dash sources via a -c one-liner here, so the brain-env.sh include
        # degrades to its guarded damaged-install fallback (dash's $0 is the
        # caller's path — the include can't self-locate) — fine for these
        # tests: every mkdir under test lives in the resolver itself. The
        # full-chain dash path is TestResolverChainPortability's concern.
        for shell in ("bash", "zsh", "dash"):
            if shutil.which(shell):
                yield shell

    def _source_under_set_e(self, shell, env_extra):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA",
                  "CLAUDE_PLUGIN_OPTION_BRAIN_PATH",
                  "CLAUDE_PLUGIN_OPTION_brain_path", "BRAIN_INSTANCE"):
            env.pop(k, None)
        env.update(env_extra)
        return subprocess.run(
            [shell, "-c",
             'set -e\n. "%s" >/dev/null 2>&1\nprintf %%s "ok:$BRAIN_DB_DIR"'
             % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60)

    def test_unwritable_persist_dir_does_not_abort(self):
        # _brain_persist_state: ~/.config exists read-only and ~/.config/brain
        # does not — its mkdir -p fails. The brain itself is healthy; only the
        # record write must be skipped.
        live = os.path.join(self._home, "live-brain")
        os.makedirs(live)
        open(os.path.join(live, "brain.db"), "w").close()
        os.makedirs(self._xdg)
        self._read_only(self._xdg)
        for shell in self._shells():
            with self.subTest(shell=shell):
                r = self._source_under_set_e(shell, {"BRAIN_DB_DIR": live})
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout, "ok:%s" % live, r.stderr)

    def test_uncreatable_plugin_option_hint_does_not_abort(self):
        # The plugin-option hint points under a read-only parent — its
        # selection-time mkdir fails; the resolution chain must still run.
        ro_parent = os.path.join(self._home, "ro")
        os.makedirs(ro_parent)
        self._read_only(ro_parent)
        hint = os.path.join(ro_parent, "brain")
        for shell in self._shells():
            with self.subTest(shell=shell):
                r = self._source_under_set_e(
                    shell, {"CLAUDE_PLUGIN_OPTION_BRAIN_PATH": hint,
                            "BRAIN_RESOLVE_NO_PERSIST": "1"})
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(r.stdout.startswith("ok:"), r.stderr)


class TestAdoptionNetAndXdgCreate(unittest.TestCase):
    """5.0a: new brains are born at the XDG service dir (D-13), and the
    create branch refuses to run while an existing brain sits unreachable at
    an old host-owned default (plugin renamed/reinstalled) — it names the
    candidate for boot to surface instead. An explicit choice (the
    ~/.config/brain/env knob) always beats the net."""

    RESOLVER = os.path.join(os.path.dirname(__file__), '..',
                            'hooks', 'scripts', 'resolve-brain-db.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-resolve-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        self._native = os.path.join(self._home, ".local", "share", "brain")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _resolve(self, env_extra=None):
        """Returns (BRAIN_DB_DIR, BRAIN_ADOPTION_CANDIDATE, BRAIN_HOST_PARKED)
        after sourcing."""
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA", "XDG_DATA_HOME",
                  "BRAIN_ADOPTION_CANDIDATE", "BRAIN_HOST_PARKED",
                  "BRAIN_PARKED_ACK", "BRAIN_INSTANCE"):
            env.pop(k, None)
        env.update(env_extra or {})
        out = subprocess.run(
            ["bash", "-c",
             'source "%s" >/dev/null 2>&1; '
             'printf "%%s\\n%%s\\n%%s" "$BRAIN_DB_DIR" '
             '"$BRAIN_ADOPTION_CANDIDATE" "$BRAIN_HOST_PARKED"'
             % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60)
        parts = (out.stdout.split("\n") + ["", ""])[:3]
        return parts[0].strip(), parts[1].strip(), parts[2].strip()

    def _make_brain(self, *segments):
        d = os.path.join(self._home, *segments)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "brain.db"), "w").close()
        return d

    def test_fresh_install_creates_at_xdg(self):
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, self._native,
                         "nothing anywhere: a new brain is born at the XDG service dir")
        self.assertEqual(cand, "")
        self.assertTrue(os.path.isdir(self._native), "create branch mkdirs the dir")

    def test_xdg_existing_brain_adopted_over_legacy(self):
        native = self._make_brain(".local", "share", "brain")
        self._make_brain("AgentsContext", "brain")
        db_dir, _, _parked = self._resolve()
        self.assertEqual(db_dir, native)

    def test_candidate_at_plugin_data_blocks_create(self):
        cand_dir = self._make_brain(".claude", "plugins", "data",
                                    "old-plugin-name", "brain")
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, "",
                         "the net must refuse to create while a candidate exists")
        self.assertEqual(cand, cand_dir)
        self.assertFalse(os.path.exists(self._native),
                         "refusal means nothing is created at the XDG dir")

    def test_candidate_via_cpd_sibling_scan(self):
        # rename case with a NON-default plugin-data root: only the live
        # $CLAUDE_PLUGIN_DATA sibling scan can find the old plugin's brain
        cand_dir = self._make_brain("custom-root", "data", "old-name", "brain")
        cpd = os.path.join(self._home, "custom-root", "data", "new-name")
        os.makedirs(cpd, exist_ok=True)
        db_dir, cand, _parked = self._resolve({"CLAUDE_PLUGIN_DATA": cpd})
        self.assertEqual(db_dir, "")
        self.assertEqual(cand, cand_dir)

    def test_config_knob_with_brain_adopted(self):
        d = self._make_brain("my-brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        db_dir, _, _parked = self._resolve()
        self.assertEqual(db_dir, d)

    def test_config_knob_beats_adoption_net(self):
        # the user's explicit fresh-start choice defeats the repeating notice
        self._make_brain(".claude", "plugins", "data", "old-plugin", "brain")
        fresh = os.path.join(self._home, "chosen-fresh")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % fresh)
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, fresh,
                         "an explicit knob is a choice — the net must not fire")
        self.assertEqual(cand, "")
        self.assertTrue(os.path.isdir(fresh), "knob dir is honored (mkdir'd)")

    def test_config_knob_beats_stale_env_hint(self):
        stale = os.path.join(self._home, "stale-baked")
        os.makedirs(stale)
        knob = os.path.join(self._home, "knob-choice")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % knob)
        db_dir, _, _parked = self._resolve({"BRAIN_DB_DIR": stale})
        self.assertEqual(db_dir, knob,
                         "durable knob beats a stale plist-baked empty dir")

    def test_cpd_existing_brain_still_adopted(self):
        # pre-D-13 install: brain at the live $CLAUDE_PLUGIN_DATA — adopted,
        # no net, no XDG create — and flagged parked: `claude plugin
        # uninstall` deletes that tree by default (verified 2026-08-27)
        cpd = os.path.join(self._home, ".claude", "plugins", "data", "brain-x")
        d = self._make_brain(".claude", "plugins", "data", "brain-x", "brain")
        db_dir, cand, parked = self._resolve({"CLAUDE_PLUGIN_DATA": cpd})
        self.assertEqual(db_dir, d)
        self.assertEqual(cand, "")
        self.assertEqual(parked, d,
                         "a plugin-data brain must be flagged for relocation")

    def test_knob_into_plugin_data_flags_parked(self):
        # a knob written by the OLD adoption notice points INTO plugins/data:
        # resolves at the knob rung, never rung 3 — parked detection is by
        # path shape, so it must fire on this route too
        d = self._make_brain(".claude", "plugins", "data", "brain-x", "brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        db_dir, _, parked = self._resolve()
        self.assertEqual(db_dir, d)
        self.assertEqual(parked, d,
                         "the knob route into plugins/data is parked all the same")

    def test_resolved_env_into_plugin_data_flags_parked(self):
        # the non-hook-shell route (4b): resolved.env persisted by a prior
        # hook points at a plugin-data brain the shell can't see via env
        d = self._make_brain(".claude", "plugins", "data", "brain-x", "brain")
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        db_dir, _, parked = self._resolve()
        self.assertEqual(db_dir, d)
        self.assertEqual(parked, d)

    def test_xdg_and_legacy_brains_not_parked(self):
        # the two safe locations must never trip the relocation warning
        native = self._make_brain(".local", "share", "brain")
        db_dir, _, parked = self._resolve()
        self.assertEqual(db_dir, native)
        self.assertEqual(parked, "")
        os.remove(os.path.join(native, "brain.db"))
        legacy = self._make_brain("AgentsContext", "brain")
        db_dir, _, parked = self._resolve()
        self.assertEqual(db_dir, legacy)
        self.assertEqual(parked, "")

    def test_parked_sibling_under_custom_cpd_root(self):
        # the adoption net's own population: a renamed SIBLING plugin dir
        # under a custom plugins root (no literal plugins/data segment).
        # Detection must use the net's root — dirname of CPD — not just CPD.
        d = self._make_brain("custom-root", "data", "old-name", "brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        cpd = os.path.join(self._home, "custom-root", "data", "new-name")
        # trailing slash on CPD must not break the match either
        db_dir, _, parked = self._resolve({"CLAUDE_PLUGIN_DATA": cpd + "/"})
        self.assertEqual(db_dir, d)
        self.assertEqual(parked, d,
                         "sibling dirs under the CPD root are on the same trapdoor")

    def test_parked_brain_exactly_at_cpd(self):
        # brain.db directly AT the plugin-data dir (option/knob can name it)
        cpd = os.path.join(self._home, "custom-root", "data", "brain-x")
        d = self._make_brain("custom-root", "data", "brain-x")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        db_dir, _, parked = self._resolve({"CLAUDE_PLUGIN_DATA": cpd})
        self.assertEqual(db_dir, d)
        self.assertEqual(parked, d)

    def test_unrelated_plugins_data_path_not_parked(self):
        # a dev checkout that merely CONTAINS plugins/data must not warn —
        # the pattern is anchored to real host roots, not a bare glob
        d = self._make_brain("dev", "myapp", "plugins", "data", "brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        db_dir, _, parked = self._resolve()
        self.assertEqual(db_dir, d)
        self.assertEqual(parked, "",
                         "unanchored substring match would false-positive here")

    def test_parked_ack_suppresses(self):
        # the informed-stay opt-out: BRAIN_PARKED_ACK silences the flag
        d = self._make_brain(".claude", "plugins", "data", "brain-x", "brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % d)
        db_dir, _, parked = self._resolve({"BRAIN_PARKED_ACK": "1"})
        self.assertEqual(db_dir, d)
        self.assertEqual(parked, "")

    # -- review-finding regressions (2026-08-12 multi-lens pass) --

    def test_knob_file_stdout_does_not_pollute_value(self):
        # a user `echo` in the sourced config file must not prepend to the
        # captured knob value
        d = self._make_brain("my-brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write('echo "loading brain env"\nBRAIN_DB_DIR=\'%s\'\n' % d)
        db_dir, _, _parked = self._resolve()
        self.assertEqual(db_dir, d)

    def test_quoted_tilde_knob_is_expanded(self):
        # BRAIN_DB_DIR="~/x" (quoted, so the shell won't expand it) — the
        # resolver expands the ~/ prefix like the Python reader does
        d = self._make_brain("tilde-brain")
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write('BRAIN_DB_DIR="~/tilde-brain"\n')
        db_dir, _, _parked = self._resolve()
        self.assertEqual(db_dir, d)

    def test_relative_knob_is_ignored(self):
        # a relative path would resolve against each consumer's cwd — ignore
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR=relative-dir\n")
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, self._native)
        self.assertEqual(cand, "")

    def test_plugin_data_without_brains_does_not_block_create(self):
        # populated plugins/data with no brain.db anywhere: the scan must
        # come up empty (no glob abort, no false candidate) and create at XDG
        os.makedirs(os.path.join(self._home, ".claude", "plugins", "data",
                                 "some-other-plugin"), exist_ok=True)
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, self._native)
        self.assertEqual(cand, "")

    def test_resolver_survives_zsh_sourcing(self):
        # zsh nomatch treats a no-match glob in a sourced file as fatal —
        # the net's scan must not be a glob; whole ladder must complete
        import subprocess
        os.makedirs(os.path.join(self._home, ".claude", "plugins", "data",
                                 "some-other-plugin"), exist_ok=True)
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA", "XDG_DATA_HOME",
                  "BRAIN_ADOPTION_CANDIDATE", "BRAIN_INSTANCE"):
            env.pop(k, None)
        out = subprocess.run(
            ["zsh", "-c",
             'source "%s" >/dev/null 2>&1; printf %%s "$BRAIN_DB_DIR"'
             % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), self._native)

    # -- the 2026-08-08 rename sandbox matrix, automated --

    def test_rename_with_current_record_rescues_silently(self):
        # old plugin's brain + resolved.env pointing at it → 4b rescue, no net
        old = self._make_brain(".claude", "plugins", "data", "old-name", "brain")
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % old)
        cpd = os.path.join(self._home, ".claude", "plugins", "data", "new-name")
        os.makedirs(cpd, exist_ok=True)
        db_dir, cand, _parked = self._resolve({"CLAUDE_PLUGIN_DATA": cpd})
        self.assertEqual(db_dir, old)
        self.assertEqual(cand, "")

    def test_rename_with_stale_record_hits_the_net(self):
        # resolved.env points at a deleted dir (the uninstall shape) — the
        # 2026-08-08 silent-fresh-brain case; now the net refuses instead
        old = self._make_brain(".claude", "plugins", "data", "old-name", "brain")
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % os.path.join(self._home, "deleted"))
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, "")
        self.assertEqual(cand, old)

    def _missing_brain(self, env_extra=None):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA", "XDG_DATA_HOME",
                  "BRAIN_MISSING_BRAIN", "BRAIN_INSTANCE"):
            env.pop(k, None)
        env.update(env_extra or {})
        return subprocess.run(
            ["bash", "-c", 'source "%s" >/dev/null 2>&1; printf %%s "$BRAIN_MISSING_BRAIN"'
             % self.RESOLVER],
            env=env, capture_output=True, text=True, timeout=60).stdout.strip()

    def test_stale_record_and_no_brain_anywhere_refuses_create(self):
        # a plain uninstall deleted the parked brain; its record remains. The
        # record proves a brain existed, so a fresh one would hide the loss
        gone = os.path.join(self._home, ".claude", "plugins", "data", "brain-brain", "brain")
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % gone)
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, "", "a missing brain must not be replaced by a fresh one")
        self.assertEqual(cand, "")
        self.assertEqual(self._missing_brain(), gone)
        self.assertFalse(os.path.exists(self._native), "nothing is created at the standard location")

    def _service_naming(self, db_dir):
        la = os.path.join(self._home, "Library", "LaunchAgents")
        os.makedirs(la, exist_ok=True)
        with open(os.path.join(la, "com.brain.daemon.plist"), "w") as f:
            f.write("<plist><dict><key>EnvironmentVariables</key><dict>\n"
                    "\t<key>BRAIN_DB_DIR</key>\n\t<string>%s</string>\n"
                    "</dict></dict></plist>\n" % db_dir)

    def test_service_naming_a_gone_brain_refuses_create(self):
        # 9.6.0 wrote no resolved.env: its service definition is the only
        # record of the parked brain a plain uninstall deleted
        gone = os.path.join(self._home, ".claude", "plugins", "data", "brain-brain", "brain")
        self._service_naming(gone)
        db_dir, cand, _parked = self._resolve()
        self.assertEqual(db_dir, "")
        self.assertEqual(self._missing_brain(), gone)
        self.assertFalse(os.path.exists(self._native))

    def test_service_over_a_brain_being_born_is_not_a_missing_brain(self):
        # a first install renders the service over the standard dir it just
        # created, before the daemon writes brain.db there
        os.makedirs(self._native)
        self._service_naming(self._native)
        db_dir, _cand, _parked = self._resolve()
        self.assertEqual(db_dir, self._native)
        self.assertEqual(self._missing_brain(), "")

    def test_starting_fresh_on_purpose_beats_the_missing_brain(self):
        # the notice's escape: the knob naming the standard location is an
        # explicit choice, taken before the net
        with open(os.path.join(self._xdg, "brain", "resolved.env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % os.path.join(self._home, "deleted"))
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._native)
        db_dir, _cand, _parked = self._resolve()
        self.assertEqual(db_dir, self._native)
        self.assertEqual(self._missing_brain(), "")


class TestDaemonPortEnvFirst(unittest.TestCase):
    """Step 2 (D-13 family): BRAIN_DAEMON_PORT is a real contract — the
    daemon's own DAEMON_PORT reads the env override first, exactly like every
    shell/hook client, so setting it can't split clients and daemon across
    two ports."""

    _REPO = os.path.join(os.path.dirname(__file__), '..')

    def setUp(self):
        # Sandbox XDG so the real ~/.config/brain/env can't leak into asserts.
        self._xdg = tempfile.mkdtemp(prefix="brain-port-xdg-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._xdg, ignore_errors=True)

    def _port(self, env_extra):
        import subprocess
        # production's resolution: no instance (the suite's own would make
        # daemon_config refuse an implicit port)
        env = {k: v for k, v in os.environ.items()
               if k not in ("BRAIN_DAEMON_PORT", "BRAIN_INSTANCE")}
        env["XDG_CONFIG_HOME"] = self._xdg
        env.update(env_extra)
        out = subprocess.run(
            [sys.executable, "-c",
             "from servers.daemon_config import DAEMON_PORT; print(DAEMON_PORT)"],
            env=env, capture_output=True, text=True, cwd=self._REPO, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.strip().splitlines()[-1]

    def test_env_override_wins(self):
        self.assertEqual(self._port({"BRAIN_DAEMON_PORT": "47299"}), "47299")

    def test_formula_when_unset(self):
        self.assertEqual(self._port({}), str(47200 + os.getuid() % 100))

    def test_env_file_fallback_covers_bare_env_processes(self):
        # The MCP server is spawned by CC without brain-env.sh — it must read
        # the same knob from the user env file or the daemon (which sources it)
        # binds one port while the MCP health monitor pings another, and
        # recover_daemon kickstart-storms a healthy daemon forever.
        os.makedirs(os.path.join(self._xdg, "brain"))
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("export BRAIN_DAEMON_PORT=47288\n")
        self.assertEqual(self._port({}), "47288")

    def test_malformed_value_warns_and_falls_back(self):
        # A garbage knob must not crash-loop the daemon under KeepAlive.
        self.assertEqual(self._port({"BRAIN_DAEMON_PORT": "auto"}),
                         str(47200 + os.getuid() % 100))



class TestApiKeyEnvHelper(unittest.TestCase):
    """Step 6a: api-key-env.sh is the ONE definition of where
    ANTHROPIC_API_KEY comes from — boot-brain.sh and brain-env.sh both source
    it instead of carrying the copy that made a casing fix land on one side
    only (the 2026-07-15 keyless-daemon failure).

    The helper is sourced transitively by brain-daemon, which runs `set -e`,
    and by resolvers that zsh also sources — so the shape is pinned here too.
    """

    HELPER = os.path.join(os.path.dirname(__file__), '..',
                          'hooks', 'scripts', 'api-key-env.sh')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-apikey-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._home, ignore_errors=True)

    def _write_env_file(self, body):
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write(body)

    def _run(self, script, env_extra=None, shell="bash"):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("CLAUDE_PLUGIN_OPTION_API_KEY", None)
        env.pop("CLAUDE_PLUGIN_OPTION_api_key", None)
        env.update(env_extra or {})
        return subprocess.run(
            [shell, "-c", '. "%s"\n%s' % (self.HELPER, script)],
            env=env, capture_output=True, text=True, timeout=30)

    def _key(self, env_extra=None, shell="bash", prelude=""):
        r = self._run(
            '%sbrain_source_user_env\nbrain_api_key_from_plugin_option\n'
            'printf %%s "${ANTHROPIC_API_KEY:-}"' % prelude,
            env_extra, shell)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_env_file_supplies_key(self):
        self._write_env_file("ANTHROPIC_API_KEY=sk-from-file\n")
        self.assertEqual(self._key(), "sk-from-file")

    def test_plugin_option_fills_when_env_file_silent(self):
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-upper"}), "sk-upper")

    def test_plugin_option_lowercase_casing_also_works(self):
        # The plugins-reference does not pin <KEY>'s case; a one-sided fix here
        # is the exact drift this extraction exists to prevent.
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_api_key": "sk-lower"}), "sk-lower")

    def test_env_file_beats_plugin_option(self):
        self._write_env_file("ANTHROPIC_API_KEY=sk-from-file\n")
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-option"}),
            "sk-from-file", "the file is the durable channel — it must win")

    def test_shell_export_beats_plugin_option(self):
        self.assertEqual(
            self._key({"ANTHROPIC_API_KEY": "sk-shell",
                       "CLAUDE_PLUGIN_OPTION_API_KEY": "sk-option"}),
            "sk-shell")

    def test_source_exports_every_var_not_just_the_key(self):
        # brain-env.sh depends on this: identity tokens in the file must reach
        # the daemon, so the source is `set -a`, not a key-only read.
        self._write_env_file("BRAIN_OPERATOR_NAME=ada\n")
        r = self._run('brain_source_user_env\n'
                      'bash -c \'printf %s "$BRAIN_OPERATOR_NAME"\'')
        self.assertEqual(r.stdout, "ada", r.stderr)

    def test_missing_env_file_is_not_an_error(self):
        r = self._run('brain_source_user_env; printf %s "$?"')
        self.assertEqual(r.stdout, "0", r.stderr)

    def test_safe_under_set_e(self):
        # brain-daemon runs `set -e` and sources this transitively: a bare
        # failing test as a standalone command would abort the whole resolver.
        r = self._run('set -e\nbrain_source_user_env\n'
                      'brain_api_key_from_plugin_option\nprintf ok')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "ok")

    def test_safe_under_zsh(self):
        if not shutil.which("zsh"):
            self.skipTest("zsh not available")
        self.assertEqual(
            self._key({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-zsh"}, shell="zsh"),
            "sk-zsh")

    def test_no_trailing_newline_env_file_still_reads(self):
        # Hand-edited files without a trailing newline exist in production
        # (2026-08-12 finding) — `.`-sourcing must still see the last line.
        self._write_env_file("ANTHROPIC_API_KEY=sk-no-newline")
        self.assertEqual(self._key(), "sk-no-newline")

    def test_failing_command_in_env_file_survives_set_e(self):
        # A failing command in the hand-edited file (FOO=$(typo-cmd)) used to
        # crash-loop the daemon: on macOS bash 3.2 errexit stays ACTIVE inside
        # a sourced file even when the `.` call is guarded with `|| true`
        # (verified 2026-08-13). The owner must drop errexit around the source
        # so the good variables on either side of the bad line still load.
        self._write_env_file(
            "BRAIN_BEFORE=loaded-before\n"
            "BRAIN_BROKEN=$(definitely-not-a-command-xyz)\n"
            "BRAIN_AFTER=loaded-after\n")
        for shell in ("bash", "zsh", "dash"):
            if not shutil.which(shell):
                continue
            with self.subTest(shell=shell):
                r = self._run(
                    'set -e\nbrain_source_user_env\n'
                    'printf %s "${BRAIN_BEFORE:-}:${BRAIN_AFTER:-}"',
                    shell=shell)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout, "loaded-before:loaded-after",
                                 r.stderr)

    def test_errexit_restored_after_source(self):
        # The errexit drop is scoped to the source itself: a caller that had
        # `set -e` on must get it back, or the daemon launcher's own failure
        # discipline silently degrades after the first env-file read.
        self._write_env_file("BRAIN_OK=1\n")
        r = self._run('set -e\nbrain_source_user_env\nfalse\nprintf leaked')
        self.assertNotEqual(r.returncode, 0,
                            "errexit was not restored after the source")
        self.assertEqual(r.stdout, "")

    def test_source_never_enables_errexit(self):
        # Hooks run WITHOUT set -e; the restore must be conditional on the
        # caller's prior state, not an unconditional `set -e`.
        self._write_env_file("BRAIN_OK=1\n")
        r = self._run('brain_source_user_env\nfalse\nprintf ok')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "ok")


class TestBootKeyMirror(unittest.TestCase):
    """Step 6a: boot-brain.sh mirrors a userConfig-resolved key into the env
    file, and ONLY a userConfig-resolved one.

    CLAUDE_PLUGIN_OPTION_* exists only inside hook executions, so without the
    mirror a user who fills the plugin's key field still runs a keyless daemon
    (launchd spawns it in a separate process tree that resolves the key from
    the env file alone — first laptop install, 2026-07-15). A key that came
    from the shell or the file itself must NOT be written back: the file is
    already the daemon's channel, and rewriting it would persist an ephemeral
    eval/session key into the user's durable config.

    Runs the real boot-brain.sh on its cold-install branch (no .runtime-ready),
    where key resolution has already happened and the provisioning chain is
    stubbed out.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-bootkey-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        # A plugin tree carrying the real boot path + stubs for everything the
        # cold branch would otherwise launch for real (runtime bootstrap...).
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("boot-brain.sh", "api-key-env.sh", "runtime-state.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        for name in ("ensure-runtime.sh", "install-daemon-service.sh",
                     "ensure-dashboard.sh"):
            stub = os.path.join(self._sd, name)
            with open(stub, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")
            os.chmod(stub, 0o755)

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    @property
    def _env_file(self):
        return os.path.join(self._xdg, "brain", "env")

    def _boot(self, env_extra):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("ANTHROPIC_API_KEY", "CLAUDE_PLUGIN_OPTION_API_KEY",
                  "CLAUDE_PLUGIN_OPTION_api_key"):
            env.pop(k, None)
        env.update(env_extra)
        return subprocess.run(
            ["bash", os.path.join(self._sd, "boot-brain.sh")],
            input="{}", env=env, capture_output=True, text=True, timeout=60)

    def _file_body(self):
        if not os.path.exists(self._env_file):
            return None
        with open(self._env_file) as f:
            return f.read()

    def test_plugin_option_key_is_mirrored(self):
        r = self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-ant-mirrored"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ANTHROPIC_API_KEY=sk-ant-mirrored", self._file_body() or "",
                      "a userConfig key must reach the launchd-spawned daemon")
        self.assertEqual(os.stat(self._env_file).st_mode & 0o777, 0o600)

    def test_lowercase_plugin_option_key_is_mirrored(self):
        self._boot({"CLAUDE_PLUGIN_OPTION_api_key": "sk-ant-lower"})
        self.assertIn("ANTHROPIC_API_KEY=sk-ant-lower", self._file_body() or "")

    def test_shell_key_is_not_mirrored(self):
        # An ephemeral shell key (eval run, isolated copy) must not be written
        # into the user's durable config.
        self._boot({"ANTHROPIC_API_KEY": "sk-ant-ephemeral"})
        self.assertIsNone(self._file_body(),
                          "a shell-supplied key must never be persisted")

    def test_existing_key_line_is_never_overwritten(self):
        with open(self._env_file, "w") as f:
            f.write("ANTHROPIC_API_KEY=sk-ant-original\n")
        self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-ant-newer"})
        self.assertEqual(self._file_body(), "ANTHROPIC_API_KEY=sk-ant-original\n")

    def test_mirror_does_not_glue_onto_a_file_without_a_trailing_newline(self):
        # Hand-edited config files with no trailing newline exist in
        # production. A bare `>>` would produce
        # `BRAIN_AGENT_NAME=xxxANTHROPIC_API_KEY=sk-...`, corrupting both lines.
        with open(self._env_file, "w") as f:
            f.write("BRAIN_AGENT_NAME=anchor")   # no trailing \n
        self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-ant-appended"})
        body = self._file_body()
        self.assertIn("\nANTHROPIC_API_KEY=sk-ant-appended\n", body)
        self.assertNotIn("anchorANTHROPIC_API_KEY", body)
        # and the file still parses as shell, which is how everything reads it
        import subprocess
        r = subprocess.run(
            ["bash", "-c", '. "%s"; printf "%%s:%%s" "$BRAIN_AGENT_NAME" "$ANTHROPIC_API_KEY"'
             % self._env_file],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(r.stdout, "anchor:sk-ant-appended")

    def test_non_sk_plugin_option_is_not_mirrored(self):
        self._boot({"CLAUDE_PLUGIN_OPTION_API_KEY": "not-a-key"})
        self.assertIsNone(self._file_body())


def _write_synced_runtime(tree, scripts_dir):
    """Mark a fake plugin tree's runtime as synced to its own uv.lock, with the
    fingerprint computed by the tree's copy of runtime-state.sh — the predicate
    boot reads — so the fixture cannot drift from it."""
    with open(os.path.join(tree, "uv.lock"), "w") as f:
        f.write("# fixture lock\n")
    import subprocess
    fingerprint = subprocess.run(
        ["bash", "-c", 'source "$1"; brain_runtime_fingerprint "$2"', "_",
         os.path.join(scripts_dir, "runtime-state.sh"), tree],
        capture_output=True, text=True, check=True).stdout
    with open(os.path.join(tree, ".runtime-ready"), "w") as f:
        f.write(fingerprint)


class TestBootImportsPluginTreeNotCwd(unittest.TestCase):
    """boot-brain.sh's `python3 -c` sites must import the PLUGIN tree's
    servers/ package, never one sitting in the hook's cwd.

    `python3 -c` puts the cwd at sys.path[0], AHEAD of PYTHONPATH — so a user
    project carrying its own top-level `servers/` package shadowed the plugin
    tree at the ensure_daemon and MCP-verify calls: confusing tracebacks and a
    false "MCP SERVER BROKEN" notice every session booted from that project.
    restart-daemon.sh already pins cwd with a `cd "$PLUGIN_ROOT"`; these tests
    hold boot-brain.sh to the same contract.

    Runs the real boot-brain.sh warm path (runtime-ready sentinel + resolved
    brain) in a fake plugin tree whose servers/ package is a minimal stub —
    the invariant under test is import resolution order, not daemon behavior.
    The same tree carries the real install-daemon-service.sh for the boot's
    handoff contract with it (exit 3 → ensure_daemon skipped).
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-bootcwd-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

        # Fake plugin tree: real boot chain, stubs for everything it launches.
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("boot-brain.sh", "api-key-env.sh", "runtime-state.sh",
                     "resolve-brain-db.sh", "brain-env.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        for name in ("ensure-runtime.sh", "install-daemon-service.sh",
                     "ensure-dashboard.sh"):
            stub = os.path.join(self._sd, name)
            with open(stub, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")
            os.chmod(stub, 0o755)
        with open(os.path.join(self._sd, "boot_brain.py"), "w") as f:
            f.write("")  # boot's final exec — a no-op here

        # Warm path: a runtime synced to the tree's lock + a venv python
        # (brain-env.sh prepends venv/bin to PATH, so link python3 too — the
        # boot script invokes bare `python3`).
        _write_synced_runtime(self._tree, self._sd)
        vbin = os.path.join(self._tree, "venv", "bin")
        os.makedirs(vbin)
        os.symlink(sys.executable, os.path.join(vbin, "python"))
        os.symlink(sys.executable, os.path.join(vbin, "python3"))

        # The plugin tree's servers/ package — what boot MUST import.
        srv = os.path.join(self._tree, "servers")
        os.makedirs(srv)
        open(os.path.join(srv, "__init__.py"), "w").close()
        with open(os.path.join(srv, "daemon_client.py"), "w") as f:
            f.write("def ensure_daemon(db):\n    return True\n")
        with open(os.path.join(srv, "brain_mcp.py"), "w") as f:
            f.write("TOOLS = ['a', 'b']\n")

        # A resolved brain, so boot reaches the daemon + MCP-verify blocks.
        self._db_dir = os.path.join(self._home, "braindb")
        os.makedirs(self._db_dir)
        open(os.path.join(self._db_dir, "brain.db"), "w").close()

        # The user project boot runs from — carrying the decoy package.
        self._project = os.path.join(self._home, "project")
        os.makedirs(os.path.join(self._project, "servers"))
        with open(os.path.join(self._project, "servers", "__init__.py"), "w") as f:
            f.write("raise ImportError('decoy servers package imported "
                    "(cwd leaked into sys.path)')\n")

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _env(self, **extra):
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg,
                   ANTHROPIC_API_KEY="sk-ant-test",
                   BRAIN_DB_DIR=self._db_dir)
        for k in ("PYTHONPATH", "PYTHONHOME", "CLAUDE_PLUGIN_DATA",
                  "CLAUDE_PLUGIN_OPTION_BRAIN_PATH",
                  "CLAUDE_PLUGIN_OPTION_brain_path"):
            env.pop(k, None)
        env.update(extra)
        return env

    def _boot(self, cwd, **extra):
        return subprocess.run(
            ["bash", os.path.join(self._sd, "boot-brain.sh")],
            input="{}", cwd=cwd, env=self._env(**extra),
            capture_output=True, text=True, timeout=60)

    def _real_installer(self, lock_wait_s=None):
        """The real install-daemon-service.sh in this tree, against a fake
        launchctl whose loaded service has a drifted plist. `lock_wait_s`
        lowers the boot's install-lock wait in this tree's copy. Returns the
        env the installer needs."""
        if sys.platform != "darwin":
            self.skipTest("launchd installer paths are macOS-only")
        for name in ("install-daemon-service.sh", "launchd-install.sh",
                     "com.brain.daemon.plist"):
            shutil.copy(os.path.join(self.SCRIPTS, name), os.path.join(self._sd, name))
        if lock_wait_s is not None:
            path = os.path.join(self._sd, "launchd-install.sh")
            src = open(path).read()
            self.assertIn("_BRAIN_LAUNCHD_LOCK_WAIT_S=10\n", src)
            with open(path, "w") as f:
                f.write(src.replace("_BRAIN_LAUNCHD_LOCK_WAIT_S=10\n",
                                    "_BRAIN_LAUNCHD_LOCK_WAIT_S=%d\n" % lock_wait_s))
        bin_dir = os.path.join(self._home, "bin")
        os.makedirs(bin_dir)
        self._log = os.path.join(self._home, "launchctl.log")
        write_launchctl_stub(bin_dir, os.path.join(self._home, "launchctl.loaded"),
                             self._log, loaded=True)
        agents = os.path.join(self._home, "Library", "LaunchAgents")
        os.makedirs(agents)
        self._target = os.path.join(agents, "com.brain.daemon.plist")
        with open(self._target, "w") as f:
            f.write("<plist>stale frozen snapshot</plist>\n")
        # BRAIN_INSTANCE="": the suite runs as an entity, and entities never
        # touch launchd — this test is the non-entity boot
        return dict(PATH=bin_dir + os.pathsep + os.environ.get("PATH", ""),
                    BRAIN_INSTANCE="",
                    BRAIN_MAINTENANCE_LOCK=os.path.join(self._home, "maintenance.lock"))

    def _verbs(self):
        return [c[0] for c in read_launchctl_calls(self._log)]

    def test_a_handed_off_reinstall_skips_ensure_daemon(self):
        # The real installer hands the drift off (exit 3) and boot-brain.sh
        # boots on the running daemon: a restart now would relaunch it on the
        # plist not yet replaced. The background run then replaces the service
        # once the boot has exited.
        env = self._real_installer()
        marker = os.path.join(self._home, "ensure-daemon-called")
        with open(os.path.join(self._tree, "servers", "daemon_client.py"), "w") as f:
            f.write("def ensure_daemon(db):\n    open(%r, 'w').close()\n    return True\n"
                    % marker)
        r = self._boot(cwd=self._project, **env)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("reinstalling in the background", out)
        self.assertIn("booting on the running daemon", out)
        self.assertFalse(os.path.exists(marker), "ensure_daemon ran after a handoff")

        def verified():
            v = self._verbs()
            return "bootstrap" in v and "print" in v[v.index("bootstrap") + 1:]
        deadline = time.time() + 30
        while time.time() < deadline and not verified():
            time.sleep(0.2)
        self.assertTrue(verified(), read_launchctl_calls(self._log))
        self.assertEqual(self._verbs().count("bootout"), 1)
        self.assertIn(self._db_dir, open(self._target).read())

    def test_the_background_run_outwaits_a_holder_the_boot_would_give_up_on(self):
        # This tree's boot gives up on the install lock after 2s; a background
        # run, off the SessionStart budget, keeps waiting for the holder.
        env = self._real_installer(lock_wait_s=2)
        lock = os.path.join(self._home, "Library", "Caches", "com.brain.daemon.install.lock")
        os.makedirs(lock)
        release = threading.Timer(5, shutil.rmtree, [lock])
        release.start()
        gone = subprocess.Popen(["true"])
        gone.wait()  # a boot PID that has already exited
        try:
            r = subprocess.run(
                ["bash", os.path.join(self._sd, "install-daemon-service.sh"),
                 "--after-pid", str(gone.pid)],
                env=self._env(**env), capture_output=True, text=True, timeout=60)
        finally:
            release.cancel()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("could not take", r.stderr)
        self.assertIn("bootstrap", self._verbs())

    def test_boot_imports_plugin_tree_when_cwd_has_decoy_servers(self):
        r = self._boot(cwd=self._project)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("MCP server OK — 2 tools available", out)
        self.assertIn("Daemon ready", out)
        self.assertNotIn("MCP SERVER BROKEN", out)
        self.assertNotIn("decoy", out)

    def test_boot_succeeds_from_neutral_cwd(self):
        # Control: proves a decoy-test failure means the decoy leaked in,
        # not that the harness itself rotted.
        r = self._boot(cwd=self._home)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertIn("MCP server OK — 2 tools available", out)
        self.assertIn("Daemon ready", out)


class TestBothSpawnPathsExecIdentically(unittest.TestCase):
    """Step 6c: the daemon has ONE boot incantation.

    hooks/scripts/brain-daemon (launchd's entry) and
    daemon_launch.daemon_argv (the no-launchd fallback) used to be a shell
    heredoc and a Python `-c` string that had already drifted apart on env
    pinning — and the heredoc interpolated $DB_PATH into Python source
    unquoted, so a brain path with a space, quote or backslash produced a
    SyntaxError and a launchd respawn loop.

    Runs the real brain-daemon against a stub interpreter that records its
    argv, and compares it to what daemon_argv builds.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-spawnpath-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("brain-daemon", "resolve-brain-db.sh", "brain-env.sh",
                     "api-key-env.sh", "launchd-install.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        stub = os.path.join(self._sd, "ensure-runtime.sh")
        with open(stub, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(stub, 0o755)
        # The venv python brain-env.sh wires in — records argv + PYTHONPATH
        # instead of starting a daemon.
        venv = os.path.join(self._tree, "venv", "bin")
        os.makedirs(venv)
        self._record = os.path.join(self._home, "argv.txt")
        py = os.path.join(venv, "python")
        with open(py, "w") as f:
            f.write('#!/bin/bash\nprintf "%s\\n" "$@" > "{rec}"\n'
                    'printf "PYTHONPATH=%s\\n" "$PYTHONPATH" >> "{rec}"\n'
                    .format(rec=self._record))
        os.chmod(py, 0o755)

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _run_launcher(self, db_dir):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg,
                   BRAIN_DB_DIR=db_dir)
        env.pop("CLAUDE_PLUGIN_DATA", None)
        env.pop("PYTHONPATH", None)
        r = subprocess.run(["bash", os.path.join(self._sd, "brain-daemon")],
                           env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(self._record) as f:
            lines = f.read().splitlines()
        argv = [l for l in lines if not l.startswith("PYTHONPATH=")]
        pythonpath = [l for l in lines if l.startswith("PYTHONPATH=")][0][11:]
        return argv, pythonpath

    def _make_brain(self, name):
        d = os.path.join(self._home, name)
        os.makedirs(d)
        open(os.path.join(d, "brain.db"), "w").close()
        return d

    def test_launcher_execs_the_module_entry_point(self):
        db_dir = self._make_brain("brain-data")
        argv, pythonpath = self._run_launcher(db_dir)
        self.assertEqual(argv, ["-m", "servers.daemon_server",
                                os.path.join(db_dir, "brain.db")])
        self.assertIn(os.path.realpath(self._tree),
                      [os.path.realpath(p) for p in pythonpath.split(os.pathsep)],
                      "the plugin tree must be on PYTHONPATH for `-m` to resolve")

    def test_both_paths_build_the_same_command(self):
        db_dir = self._make_brain("brain-data")
        db_path = os.path.join(db_dir, "brain.db")
        shell_argv, _ = self._run_launcher(db_dir)
        python_argv = dl.daemon_argv(db_path)[1:]  # drop the interpreter
        self.assertEqual(shell_argv, python_argv,
                         "launchd and the direct-spawn fallback must exec the "
                         "same daemon command — divergence here is the class "
                         "step 6c removed")

    def test_path_with_spaces_and_quotes_survives(self):
        # The old heredoc pasted this straight into Python source.
        db_dir = self._make_brain("my brain's data")
        argv, _ = self._run_launcher(db_dir)
        self.assertEqual(argv[-1], os.path.join(db_dir, "brain.db"))

class TestSendCommandTransportContract(unittest.TestCase):
    """Step 6d: `transport` is the STABLE classification of a WIRE failure.

    Three consumers branch on it — hook_common.daemon_call_raw (which error
    text Claude sees and which hook_errors row is written), brain_mcp.daemon_send
    and restart-daemon.sh. They must never match on the prose in `error`, which
    is for humans; and the key must be ABSENT when the daemon answered, because
    a daemon-level ok=false reported as a transport failure turns every
    daemon-side error into "cannot reach the daemon".
    """

    def _serve(self, handler):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        host, port = srv.getsockname()

        def run():
            try:
                conn, _ = srv.accept()
                try:
                    conn.recv(4096)
                    handler(conn)
                finally:
                    conn.close()
            except OSError:
                pass
            finally:
                srv.close()

        threading.Thread(target=run, daemon=True).start()
        return host, port

    def _send(self, handler, timeout=3.0):
        host, port = self._serve(handler)
        with patch.object(dc, "get_daemon_addr", return_value=(host, port)):
            return dc.send_command("probe", timeout=timeout)

    def test_port_addresses_another_daemon_on_loopback(self):
        # One process driving several eval arms names each arm's port; the
        # ambient address (DAEMON_PORT, fixed at import) is not consulted.
        host, port = self._serve(lambda c: c.sendall(b'{"ok": true, "result": "arm"}\n'))
        with patch.object(dc, "get_daemon_addr",
                          side_effect=AssertionError("ambient address used")):
            resp = dc.send_command("probe", timeout=3.0, port=port)
        self.assertEqual(resp, {"ok": True, "result": "arm"})
        self.assertEqual(host, dc.DAEMON_HOST)

    def test_default_address_is_unchanged(self):
        with patch.object(dc.socket, "socket") as sock, \
             patch.object(dc, "get_daemon_addr", return_value=("127.0.0.1", 1)) as ambient:
            sock.return_value.recv.return_value = b'{"ok": true}\n'
            dc.send_command("probe")
        ambient.assert_called_once()
        sock.return_value.connect.assert_called_once_with(("127.0.0.1", 1))

    def test_success_carries_no_transport(self):
        resp = self._send(lambda c: c.sendall(b'{"ok": true, "result": {"x": 1}}\n'))
        self.assertTrue(resp.get("ok"))
        self.assertIsNone(resp.get("transport"))

    def test_daemon_level_error_carries_no_transport(self):
        # THE invariant: the daemon answered. Marking this a transport failure
        # would make every daemon-side error read as an unreachable daemon.
        resp = self._send(lambda c: c.sendall(b'{"ok": false, "error": "no such node"}\n'))
        self.assertFalse(resp.get("ok"))
        self.assertIsNone(resp.get("transport"),
                          "a daemon that answers is not a wire failure")
        self.assertEqual(resp.get("error"), "no such node")

    def test_empty_reply_is_transport_empty(self):
        resp = self._send(lambda c: None)  # accept, then close
        self.assertEqual(resp.get("transport"), "empty")
        self.assertIn("empty response", resp.get("error", ""))

    def test_garbled_reply_is_transport_protocol(self):
        resp = self._send(lambda c: c.sendall(b"not json at all\n"))
        self.assertEqual(resp.get("transport"), "protocol")

    def test_timeout_is_transport_timeout(self):
        resp = self._send(lambda c: time.sleep(2), timeout=0.3)
        self.assertEqual(resp.get("transport"), "timeout")

    def test_refused_is_transport_refused(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        host, port = srv.getsockname()
        srv.close()  # nothing listening
        with patch.object(dc, "get_daemon_addr", return_value=(host, port)):
            resp = dc.send_command("probe", timeout=1.0)
        self.assertEqual(resp.get("transport"), "refused")

    def test_unread_request_cut_off_by_a_closing_listener_is_reset(self):
        # A reloading daemon keeps its socket open through teardown without
        # accepting; the exec closes it under the queued connection. The
        # daemon never read the request, unlike `empty`, where it read then
        # closed.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        threading.Timer(0.2, srv.close).start()
        resp = dc.send_command("probe", timeout=3.0, port=srv.getsockname()[1])
        self.assertEqual(resp.get("transport"), "reset")

    def test_non_object_reply_is_a_protocol_failure_not_a_raise(self):
        # Valid JSON that isn't an object (a foreign listener on the port).
        # The contract is "returns a response dict" — callers must not each
        # discover separately that it sometimes doesn't.
        resp = self._send(lambda c: c.sendall(b'"OK"\n'))
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp.get("transport"), "protocol")
        self.assertIn("non-object", resp.get("error", ""))


class TestBrainEnvCallsTheApiKeyOwner(unittest.TestCase):
    """Step 6a left brain-env.sh — the file EVERY hook sources — as a one-line
    caller of api-key-env.sh. Deleting that one line is far easier to do by
    accident than deleting the inline block it replaced, and nothing else
    covers this wiring: TestApiKeyEnvHelper tests the owner in isolation and
    TestBootKeyMirror tests boot-brain.sh's separate copy of the call.

    Blast radius if it goes: the 2026-07-15 failure, where a user filled in the
    plugin's key field and the daemon ran keyless anyway."""

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-envwire-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _source(self, env_extra, echo):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg)
        for k in ("ANTHROPIC_API_KEY", "CLAUDE_PLUGIN_OPTION_API_KEY",
                  "CLAUDE_PLUGIN_OPTION_api_key"):
            env.pop(k, None)
        env.update(env_extra)
        r = subprocess.run(
            ["bash", "-c", '. "%s" >/dev/null 2>&1; printf %%s "%s"'
             % (os.path.join(self.SCRIPTS, "brain-env.sh"), echo)],
            env=env, capture_output=True, text=True, timeout=90)
        return r.stdout

    def test_brain_env_resolves_the_key_from_the_plugin_option(self):
        self.assertEqual(
            self._source({"CLAUDE_PLUGIN_OPTION_API_KEY": "sk-wired"},
                         "$ANTHROPIC_API_KEY"),
            "sk-wired",
            "brain-env.sh must call the api-key-env.sh owner, not skip it")

    def test_brain_env_sources_the_user_config_file(self):
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_OPERATOR_NAME=ada\nANTHROPIC_API_KEY=sk-from-file\n")
        out = self._source({}, "$BRAIN_OPERATOR_NAME:$ANTHROPIC_API_KEY")
        self.assertEqual(out, "ada:sk-from-file",
                         "every var in the user config must reach the daemon, "
                         "not just the key")


class TestKnobReadSurvivesDamagedInstall(unittest.TestCase):
    """Step 6 review finding: reading the BRAIN_DB_DIR knob must depend on
    NOTHING but $HOME.

    Before step 6 all three knob readers used the literal
    `${XDG_CONFIG_HOME:-$HOME/.config}/brain/env` — a string that cannot fail.
    Extraction (a) routed that read through `brain_user_env_file`, creating a
    failure mode that did not exist: if `api-key-env.sh` is missing or
    unreadable, the function is undefined. The first repair REFUSED in that
    state, which made a brain the user explicitly named unreachable and told
    them it wasn't configured. The correct repair is a fallback to the literal
    path, so a damaged install still resolves the right brain.

    Second half of the same finding: `.` on a missing file is a SPECIAL BUILTIN
    failure that `|| true` cannot rescue — dash exits 2 and bash under `set -e`
    (which brain-daemon sets) exits 1, both before the resolver can do
    anything. Only a readability guard survives, so that is what both sourcing
    sites use.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-damaged-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        # A plugin tree we can damage without touching the repo.
        self._sd = os.path.join(self._home, "plugin", "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("resolve-brain-db.sh", "brain-env.sh", "api-key-env.sh",
                     "launchd-install.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name), os.path.join(self._sd, name))
        stub = os.path.join(self._sd, "ensure-runtime.sh")
        with open(stub, "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(stub, 0o755)
        # A real brain, named only by the knob.
        self._brain = os.path.join(self._home, "knobbrain")
        os.makedirs(self._brain)
        open(os.path.join(self._brain, "brain.db"), "w").close()
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._brain)

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _resolve(self, shell="bash", errexit=False):
        import subprocess
        resolver = os.path.join(self._sd, "resolve-brain-db.sh")
        script = '%s. "%s" >/dev/null 2>&1 %s printf %%s "$BRAIN_DB_DIR"' % (
            "set -e; " if errexit else "",
            resolver,
            "|| true;" if errexit else ";")
        r = subprocess.run([shell, "-c", script],
                           cwd="/tmp",
                           env={"HOME": self._home, "PATH": "/usr/bin:/bin",
                                "XDG_CONFIG_HOME": self._xdg},
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()

    def _shadow_brain_created(self):
        return os.path.exists(os.path.join(self._home, ".local", "share", "brain"))

    def _damage(self, mode="remove"):
        p = os.path.join(self._sd, "api-key-env.sh")
        if mode == "remove":
            os.remove(p)
        else:
            os.chmod(p, 0o000)

    def test_healthy_install_adopts_the_knob_brain(self):
        for shell in ("bash", "zsh", "dash"):
            with self.subTest(shell=shell):
                if not shutil.which(shell):
                    self.skipTest("%s not available" % shell)
                self.assertEqual(self._resolve(shell), self._brain)

    def test_missing_api_key_env_still_adopts_the_knob_brain(self):
        self._damage("remove")
        for shell in ("bash", "zsh", "dash"):
            with self.subTest(shell=shell):
                if not shutil.which(shell):
                    self.skipTest("%s not available" % shell)
                self.assertEqual(
                    self._resolve(shell), self._brain,
                    "a damaged install must not make the user's named brain "
                    "unreachable — fall back to the literal config path")
        self.assertFalse(self._shadow_brain_created(),
                         "and it must certainly not birth a second brain")

    def test_unreadable_api_key_env_still_adopts_the_knob_brain(self):
        self._damage("chmod")
        self.assertEqual(self._resolve("bash"), self._brain)
        self.assertFalse(self._shadow_brain_created())

    def test_damaged_install_does_not_kill_the_shell_under_errexit(self):
        # brain-daemon runs `set -e`; a special-builtin failure there exits
        # before any FATAL can print, and launchd respawns every 10s forever.
        self._damage("remove")
        self.assertEqual(self._resolve("bash", errexit=True), self._brain,
                         "the resolver must survive `set -e` on a damaged install")

    def test_damaged_install_does_not_kill_dash(self):
        if not shutil.which("dash"):
            self.skipTest("dash not available")
        self._damage("remove")
        self.assertEqual(self._resolve("dash"), self._brain,
                         "`.` on a missing file exits dash outright unless guarded")


class TestResolverChainPortability(unittest.TestCase):
    """The resolver chain (resolve-brain-db.sh → brain-env.sh →
    api-key-env.sh) is SOURCED, never executed, so it runs in the consumer's
    shell, not its shebang: bash hooks, zsh skills/terminals, and dash
    (/bin/sh on Linux, D-3). The chain must be POSIX — `source` at the
    brain-env.sh include was a bashism, so under dash the chain ran
    permanently in its damaged-install fallback (brain-env.sh never loaded,
    the api-key owner's accessor never in scope) and died 127 under `set -e`.

    Everything here drives the chain through a SIBLING driver script — the
    real consumer shape, and the only shape dash supports at all: a sourced
    file's $0 under dash is the CALLER's path (there is no BASH_SOURCE), so
    the chain self-locates only when sourced from its own directory. cwd is
    pinned elsewhere so cwd-relative resolution cannot fake a pass.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-chainport-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("resolve-brain-db.sh", "brain-env.sh", "api-key-env.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        stub = os.path.join(self._sd, "ensure-runtime.sh")
        with open(stub, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(stub, 0o755)
        # A real brain, named by the config knob — resolving it through the
        # NON-fallback path requires brain-env.sh to have actually loaded.
        self._brain = os.path.join(self._home, "knobbrain")
        os.makedirs(self._brain)
        open(os.path.join(self._brain, "brain.db"), "w").close()
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._brain)

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _shells(self):
        for shell in ("bash", "zsh", "dash"):
            if shutil.which(shell):
                yield shell

    def _run_driver(self, shell, body, errexit=False):
        import subprocess
        driver = os.path.join(self._sd, "driver.sh")
        with open(driver, "w") as f:
            if errexit:
                f.write("set -e\n")
            f.write('. "$(dirname "$0")/resolve-brain-db.sh"\n')
            f.write(body)
        return subprocess.run(
            [shell, driver], cwd="/tmp",
            env={"HOME": self._home, "XDG_CONFIG_HOME": self._xdg,
                 "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=60)

    PROBE = ('printf %s "$BRAIN_DB_DIR|${PLUGIN_DIR:-UNSET}|'
             '$(command -v brain_user_env_file >/dev/null 2>&1 '
             '&& echo fn || echo nofn)"\n')

    def test_chain_loads_fully_in_each_shell(self):
        # Not just "resolution succeeded" — the fallback also resolves the
        # knob. FULLY loaded means brain-env.sh ran (PLUGIN_DIR exported) and
        # api-key-env.sh ran (the accessor in scope), with no damaged-install
        # WARN on stderr.
        for shell in self._shells():
            with self.subTest(shell=shell):
                r = self._run_driver(shell, self.PROBE)
                self.assertEqual(r.returncode, 0, r.stderr)
                db, plug, fn = r.stdout.split("|")
                self.assertEqual(db, self._brain, r.stderr)
                self.assertEqual(os.path.realpath(plug),
                                 os.path.realpath(self._tree), r.stderr)
                self.assertEqual(fn, "fn",
                                 "api-key owner not in scope — the chain ran "
                                 "in its damaged-install fallback")
                self.assertNotIn("WARN", r.stderr)
                self.assertNotIn("not found", r.stderr)

    def test_chain_survives_set_e_in_each_shell(self):
        # brain-daemon sources the resolver under `set -e`; a bashism there
        # (dash: `source` → 127) kills the launcher before any FATAL prints.
        for shell in self._shells():
            with self.subTest(shell=shell):
                r = self._run_driver(shell, 'printf ok:%s "$BRAIN_DB_DIR"\n',
                                     errexit=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(r.stdout, "ok:%s" % self._brain, r.stderr)

    def test_persist_record_stable_across_sources(self):
        # _brain_persist_state compares before rewriting. The bash-only
        # $(<file) read expanded EMPTY under dash, so the comparison always
        # mismatched and every source rewrote resolved.env. tmp+mv replaces
        # the inode, so a stable inode across sources proves the no-op path.
        state = os.path.join(self._xdg, "brain", "resolved.env")
        for shell in self._shells():
            with self.subTest(shell=shell):
                r = self._run_driver(shell, 'printf done\n')
                self.assertEqual(r.returncode, 0, r.stderr)
                first = os.stat(state).st_ino
                r = self._run_driver(shell, 'printf done\n')
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(os.stat(state).st_ino, first,
                                 "unchanged record was rewritten — the "
                                 "existing-content read is broken in %s"
                                 % shell)


class TestParkedNoticeRender(unittest.TestCase):
    """The parked-brain relocation notice, rendered by a real boot-brain.sh.

    The flag has resolver-level tests; this is the only coverage of the
    render path — the guard line, the source gate, and the heredoc. Without
    it, dropping the `&& _parked_notice` call (or a heredoc regression)
    ships the whole feature silently disabled: boot still exits 0 and
    memory still works.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..',
                           'hooks', 'scripts')

    def setUp(self):
        self._home = tempfile.mkdtemp(prefix="brain-parkednotice-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        self._tree = os.path.join(self._home, "plugin")
        self._sd = os.path.join(self._tree, "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("boot-brain.sh", "api-key-env.sh", "runtime-state.sh",
                     "resolve-brain-db.sh", "brain-env.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        for name in ("ensure-runtime.sh", "install-daemon-service.sh",
                     "ensure-dashboard.sh"):
            stub = os.path.join(self._sd, name)
            with open(stub, "w") as f:
                f.write("#!/bin/bash\nexit 0\n")
            os.chmod(stub, 0o755)
        with open(os.path.join(self._sd, "boot_brain.py"), "w") as f:
            f.write("")
        _write_synced_runtime(self._tree, self._sd)
        vbin = os.path.join(self._tree, "venv", "bin")
        os.makedirs(vbin)
        os.symlink(sys.executable, os.path.join(vbin, "python"))
        os.symlink(sys.executable, os.path.join(vbin, "python3"))
        srv = os.path.join(self._tree, "servers")
        os.makedirs(srv)
        open(os.path.join(srv, "__init__.py"), "w").close()
        with open(os.path.join(srv, "daemon_client.py"), "w") as f:
            f.write("def ensure_daemon(db):\n    return True\n")
        with open(os.path.join(srv, "brain_mcp.py"), "w") as f:
            f.write("TOOLS = ['a', 'b']\n")
        # a parked brain (host plugin-data shape) and a safe one
        self._parked = os.path.join(self._home, ".claude", "plugins",
                                    "data", "brain-x", "brain")
        os.makedirs(self._parked)
        open(os.path.join(self._parked, "brain.db"), "w").close()
        self._safe = os.path.join(self._home, "braindb")
        os.makedirs(self._safe)
        open(os.path.join(self._safe, "brain.db"), "w").close()

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _boot(self, db_dir, source="startup", extra_env=None):
        import json
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg,
                   ANTHROPIC_API_KEY="sk-ant-test", BRAIN_DB_DIR=db_dir)
        for k in ("PYTHONPATH", "PYTHONHOME", "CLAUDE_PLUGIN_DATA",
                  "CLAUDE_PLUGIN_OPTION_BRAIN_PATH",
                  "CLAUDE_PLUGIN_OPTION_brain_path", "BRAIN_PARKED_ACK",
                  "XDG_DATA_HOME"):
            env.pop(k, None)
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", os.path.join(self._sd, "boot-brain.sh")],
            input=json.dumps({"source": source}), cwd=self._home, env=env,
            capture_output=True, text=True, timeout=60)

    def test_parked_brain_renders_relocation_notice_on_startup(self):
        r = self._boot(self._parked)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("relocate-brain.sh", r.stdout)
        self.assertIn(self._parked, r.stdout)
        self.assertIn("BRAIN_PARKED_ACK", r.stdout)
        # the notice names the data FOLDER and points at --keep-data; since the
        # entity rename the folder is not the plugin id, so the line is not
        # paste-ready by design (`claude plugin list` shows which plugin)
        self.assertIn("--keep-data", r.stdout)
        self.assertIn("data folder is `brain-x`", r.stdout)

    def test_notice_quiet_on_resume_and_compact(self):
        for source in ("resume", "compact"):
            r = self._boot(self._parked, source=source)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotIn("relocate-brain.sh", r.stdout,
                             "notice must not re-inject on %s" % source)

    def test_notice_quiet_under_ack(self):
        r = self._boot(self._parked, extra_env={"BRAIN_PARKED_ACK": "1"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("relocate-brain.sh", r.stdout)

    def test_notice_quiet_for_safe_location(self):
        r = self._boot(self._safe)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn("relocate-brain.sh", r.stdout)


class TestRelocateBrain(unittest.TestCase):
    """relocate-brain.sh end to end in a sandbox: copy, verify, switch,
    retire the source, comment the knob, re-render services, re-resolve.

    launchctl is stubbed via PATH and the pid/lock files are pointed into
    the sandbox — the test must never touch the real daemon on this
    machine.
    """

    SCRIPTS = os.path.join(os.path.dirname(__file__), '..',
                           'hooks', 'scripts')

    def setUp(self):
        import sqlite3
        self._home = tempfile.mkdtemp(prefix="brain-relocate-home-")
        self._xdg = os.path.join(self._home, ".config")
        os.makedirs(os.path.join(self._xdg, "brain"))
        self._sd = os.path.join(self._home, "plugin", "hooks", "scripts")
        os.makedirs(self._sd)
        for name in ("relocate-brain.sh", "resolve-brain-db.sh",
                     "brain-env.sh", "api-key-env.sh", "launchd-install.sh"):
            shutil.copy(os.path.join(self.SCRIPTS, name),
                        os.path.join(self._sd, name))
        self._svc_log = os.path.join(self._home, "svc-invoked")
        with open(os.path.join(self._sd, "install-daemon-service.sh"), "w") as f:
            f.write("#!/bin/bash\ntouch '%s'\nexit 0\n" % self._svc_log)
        self._dash_log = os.path.join(self._home, "dash-invoked")
        with open(os.path.join(self._sd, "ensure-dashboard.sh"), "w") as f:
            f.write("#!/bin/bash\ntouch '%s'\nexit 0\n" % self._dash_log)
        for name in ("install-daemon-service.sh", "ensure-dashboard.sh",
                     "relocate-brain.sh"):
            os.chmod(os.path.join(self._sd, name), 0o755)
        with open(os.path.join(self._sd, "ensure-runtime.sh"), "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(os.path.join(self._sd, "ensure-runtime.sh"), 0o755)
        # fake launchctl so bootout can never reach the real services. It
        # starts LOADED and models the transition, so an unload can succeed —
        # a stub that always answers "still managed" sends every caller down
        # its bootout-failed branch, five seconds of retry at a time.
        self._bin = os.path.join(self._home, "bin")
        os.makedirs(self._bin)
        self._lc_log = os.path.join(self._home, "launchctl.log")
        write_launchctl_stub(self._bin,
                             os.path.join(self._home, "launchctl.loaded"),
                             self._lc_log)
        # a real (tiny) brain at a plugins-data path, knob pointing at it
        self._src = os.path.join(self._home, ".claude", "plugins", "data",
                                 "brain-b", "brain")
        os.makedirs(self._src)
        for db in ("brain.db", "brain_logs.db"):
            conn = sqlite3.connect(os.path.join(self._src, db))
            conn.execute("CREATE TABLE t (v TEXT)")
            conn.execute("INSERT INTO t VALUES ('payload-%s')" % db)
            conn.commit()
            conn.close()
        with open(os.path.join(self._xdg, "brain", "env"), "w") as f:
            f.write("BRAIN_DB_DIR='%s'\n" % self._src)
        self._target = os.path.join(self._home, ".local", "share", "brain")

    def tearDown(self):
        shutil.rmtree(self._home, ignore_errors=True)

    def _run(self, *args):
        import subprocess
        env = dict(os.environ, HOME=self._home, XDG_CONFIG_HOME=self._xdg,
                   PATH=self._bin + os.pathsep + os.environ.get("PATH", ""),
                   BRAIN_MAINTENANCE_LOCK=os.path.join(self._home, "m.lock"),
                   BRAIN_DAEMON_PIDFILE=os.path.join(self._home, "d.pid"))
        # BRAIN_INSTANCE: relocation is an operator act that refuses to run
        # from an entity, and the suite is one.
        for k in ("BRAIN_DB_DIR", "CLAUDE_PLUGIN_DATA", "XDG_DATA_HOME",
                  "BRAIN_PARKED_ACK", "CLAUDE_PLUGIN_OPTION_BRAIN_PATH",
                  "CLAUDE_PLUGIN_OPTION_brain_path", "BRAIN_INSTANCE"):
            env.pop(k, None)
        return subprocess.run(
            ["bash", os.path.join(self._sd, "relocate-brain.sh")] + list(args),
            env=env, capture_output=True, text=True, timeout=120)

    def test_relocates_verifies_and_retires_source(self):
        import glob
        import sqlite3
        r = self._run()
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        # the writers were stopped first, and the stop SUCCEEDED. Moving a
        # brain out from under a live daemon is the failure this step exists
        # to prevent, and nothing here checked it: under the always-rc-0 stub
        # this assertion's subject spent five seconds per service retrying,
        # warned, and carried on — green either way.
        booted = sorted(c[1].rsplit("/", 1)[-1]
                        for c in read_launchctl_calls(self._lc_log)
                        if c[0] == "bootout" and len(c) > 1)
        self.assertEqual(booted, ["com.brain.daemon", "com.brain.dashboard"],
                         "both writers must be booted out before the copy; "
                         "saw %r" % booted)
        self.assertNotIn("did not unload", out,
                         "bootout gave up after its retry budget — the move "
                         "ran against a service launchd still considers "
                         "loaded")
        # moved and intact
        for db in ("brain.db", "brain_logs.db"):
            conn = sqlite3.connect(os.path.join(self._target, db))
            self.assertEqual(conn.execute("SELECT v FROM t").fetchone()[0],
                             "payload-" + db)
            conn.close()
        # source retired (renamed, never deleted), nothing left at the path
        self.assertFalse(os.path.exists(self._src))
        self.assertEqual(len(glob.glob(self._src + ".relocated-*")), 1)
        # knob commented with a backup, services re-rendered, ladder verified
        env_text = open(os.path.join(self._xdg, "brain", "env")).read()
        self.assertIn("# relocated to the standard location:", env_text)
        self.assertTrue(os.path.exists(
            os.path.join(self._xdg, "brain", "env.bak-relocate")))
        self.assertTrue(os.path.exists(self._svc_log),
                        "install-daemon-service.sh must be re-run")
        # the dashboard was running, and the stop above took it down with the
        # daemon: it must come back on the new path, not wait for /dashboard
        self.assertTrue(os.path.exists(self._dash_log),
                        "ensure-dashboard.sh must bring the stopped dashboard back")
        self.assertIn("verified: the resolver finds the brain at", out)
        # the maintenance lock is gone
        self.assertFalse(os.path.exists(os.path.join(self._home, "m.lock")))

    def test_dashboard_not_started_when_it_was_not_running(self):
        # a machine that never ran the dashboard must not gain one from a move
        write_launchctl_stub(self._bin,
                             os.path.join(self._home, "launchctl.loaded"),
                             self._lc_log, loaded=False)
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(os.path.exists(self._dash_log))

    def test_second_run_is_a_noop(self):
        self.assertEqual(self._run().returncode, 0)
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("nothing to do", r.stdout)

    def test_occupied_target_refuses_loudly_and_touches_nothing(self):
        import sqlite3
        os.makedirs(self._target)
        conn = sqlite3.connect(os.path.join(self._target, "brain.db"))
        conn.execute("CREATE TABLE other (v TEXT)")
        conn.commit()
        conn.close()
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("already holds a brain.db", r.stderr)
        # both brains untouched
        self.assertTrue(os.path.exists(os.path.join(self._src, "brain.db")))
        conn = sqlite3.connect(os.path.join(self._target, "brain.db"))
        self.assertEqual(
            conn.execute("SELECT name FROM sqlite_master").fetchone()[0],
            "other")
        conn.close()

    def _lsof_stub(self, access, from_call):
        """lsof on PATH reporting pid 4242 holding a source file with this
        access mode ('u' read-write, 'r' read-only) from call N on. Calls 1
        and 2 are the check before the copy, call 3 the check before the
        switch."""
        counter = os.path.join(self._home, "lsof.calls")
        with open(os.path.join(self._bin, "lsof"), "w") as f:
            f.write("#!/bin/bash\n"
                    "n=$(( $(cat '%s' 2>/dev/null || echo 0) + 1 )); echo $n > '%s'\n"
                    "[ $n -ge %d ] && printf 'p4242\\nf5\\na%s\\n'\n"
                    "exit 0\n" % (counter, counter, from_call, access))
        os.chmod(os.path.join(self._bin, "lsof"), 0o755)

    def test_writer_that_opens_the_source_during_the_copy_moves_nothing(self):
        # a session booting mid-copy can re-bootstrap a daemon on the source;
        # the copy is then stale, so nothing may switch
        import glob
        self._lsof_stub("u", from_call=3)
        env_before = open(os.path.join(self._xdg, "brain", "env")).read()
        r = self._run()
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("during the copy", r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self._src, "brain.db")))
        self.assertFalse(os.path.exists(self._target))
        self.assertEqual(glob.glob(self._target + ".incoming.*"), [],
                         "the stale staging copy must be removed")
        self.assertEqual(glob.glob(self._src + ".relocated-*"), [])
        self.assertEqual(open(os.path.join(self._xdg, "brain", "env")).read(),
                         env_before, "the knob must still name the source")
        self.assertFalse(os.path.exists(os.path.join(self._home, "m.lock")))

    def test_a_reader_holding_the_source_does_not_stop_the_move(self):
        # antivirus, Spotlight and backup agents open files read-only; they
        # cannot make the copy stale
        self._lsof_stub("r", from_call=1)
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(os.path.join(self._target, "brain.db")))

    def test_knob_rewrite_keeps_a_symlinked_env_file_its_mode_and_the_key(self):
        # the env file also holds the API key and may be a dotfiles symlink;
        # the knob may be written in export form
        envf = os.path.join(self._xdg, "brain", "env")
        real = os.path.join(self._home, "dotfiles", "brain-env")
        os.makedirs(os.path.dirname(real))
        with open(real, "w") as f:
            f.write("ANTHROPIC_API_KEY=sk-test-not-real\n"
                    "export BRAIN_DB_DIR='%s'\n" % self._src)
        os.chmod(real, 0o600)
        os.remove(envf)
        os.symlink(real, envf)
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.islink(envf), "the symlink must survive")
        text = open(real).read()
        self.assertIn("ANTHROPIC_API_KEY=sk-test-not-real\n", text)
        self.assertIn("# relocated to the standard location: export BRAIN_DB_DIR=", text)
        self.assertEqual(os.stat(real).st_mode & 0o777, 0o600)
        self.assertFalse([p for p in os.listdir(os.path.dirname(real))
                          if p.endswith(".tmp-relocate")])


if __name__ == "__main__":
    unittest.main()
