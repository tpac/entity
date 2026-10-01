"""Pytest fixtures + dev guard rails.

Runs once per pytest invocation. Refuses to proceed if the wrong Python
interpreter is active — catches the "tests pass here but the daemon
runs a different Python" class of bug that bit us on 2026-04-19.

Expected interpreter: the bundled CPython 3.11 at
`<repo>/venv/bin/python`. That's the Python the daemon uses, the hooks
use via brain-env.sh, and the one not blocked by macOS SIP so debuggers
(py-spy, lldb) can attach.

To bypass (rare — e.g. quick smoke test on a machine without the venv):
    BRAIN_ALLOW_ANY_PYTHON=1 pytest ...

To run with the right Python without thinking about it:
    ./dev pytest ...
"""
import atexit
import errno
import fcntl
import os
import re
import shutil
import socket
import sys
import tempfile
import time

import pytest


# Environment isolation. os.environ is process-global and pytest runs the whole
# suite in one process, so a test that sets a variable and doesn't put it back
# hands that value to every test after it. The bite that motivated this: four
# classes set $BRAIN_DB_DIR in setUpClass and rmtree'd the directory in
# tearDownClass, so the rest of the suite inherited a pointer to a directory
# that no longer existed — and resolve_db_dir() trusts that variable
# unconditionally (D-13's top rung, on the premise that the hook wrappers
# validated it).
#
# Two fixtures because test setup nests two deep, and they nest the same way:
# the class-scoped one wraps unittest's setUpClass/tearDownClass, the
# function-scoped one wraps each test. So a variable set in setUpClass survives
# that class's tests and dies with the class; one set inside a test dies with
# the test.
#
# This is the chokepoint, not a fifth convention: callers keep using whatever
# they already use — monkeypatch, patch.dict, explicit save/restore — and none
# of them can leak past its own scope even when the restore is missing or a
# tearDown raises before reaching it.
#
# LIMIT, and it is a real one. A guard can only restore what its own scope
# opened, so anything that outlives the scope but is first created INSIDE it
# gets clobbered. Two shapes to keep out of the suite:
#   - a module/session-scoped fixture that writes env and is first requested by
#     a LATER test of a class rather than its first (the class guard snapshotted
#     before the fixture existed, so its restore deletes env the fixture still
#     owns). A brain env as a module-scoped fixture is exactly this shape, and
#     every file using it is safe only because its class requests it from the
#     FIRST test — verified by watching BRAIN_DB_DIR across test_fetch_tools.py.
#     Nothing enforces that: it is a real, unguarded limit, because the only
#     check available is textual and a grep for this shape matches its own
#     description.
#   - a module whose IMPORT writes env, first imported inside a test: the write
#     is reverted and sys.modules caching means it never happens again. That is
#     why daemon_config is imported below rather than left to whichever test
#     file happens to pull it in first.
#
# REPAIRING SILENTLY IS ITS OWN DRIFT. A guard that quietly absorbs an
# unrestored write means the next sloppy setUpClass is invisible forever, so
# every repair is recorded and the session FAILS on any of them. The suite sits
# at zero — the four sites that motivated this turned out to be writing
# $BRAIN_DB_DIR that nothing read, and were deleted rather than annotated — so
# a non-empty report means someone just introduced a leak, named with the test
# and the keys. Set BRAIN_TEST_ENV_LEAKS_OK=1 to downgrade it to a report while
# mid-refactor.
#
# pytest owns PYTEST_CURRENT_TEST and rewrites it per test; leave it alone.
_ENV_NOT_OURS = frozenset({'PYTEST_CURRENT_TEST'})
_ENV_LEAKS = []
# The one file whose job is to leak: its probes exist to prove the guard puts
# things back, so reporting them would make the report permanently non-empty
# and train everyone to ignore it.
_ENV_LEAK_EXEMPT = 'test_env_isolation.py'


_SECRET_KEY = re.compile(r'KEY|TOKEN|SECRET|PASSWORD', re.I)


def _shown(key, value):
    """A leaked value as the report may print it. A credential is shown as
    its fingerprint — the report lands in terminals and CI logs, where a raw
    key must never appear (the daemon's own rule, key_fingerprint)."""
    if value is None:
        return 'unset'
    if _SECRET_KEY.search(key):
        from servers.scales.dispatch import key_fingerprint
        return 'fp:%s' % key_fingerprint(value)
    return repr(value)


def _restore_env(saved, where=''):
    # Re-add before delete: a restore that dies half-done must not leave a
    # deliberately-popped baseline (the ASPECTS_JSON_PATH pin) missing, and
    # pop-with-default cannot raise on a key another thread already removed.
    leaked = []
    for key, value in saved.items():
        if key not in _ENV_NOT_OURS and os.environ.get(key) != value:
            leaked.append('%s: %s -> %s' % (key, _shown(key, value),
                                            _shown(key, os.environ.get(key))))
            os.environ[key] = value
    for key in [k for k in os.environ if k not in saved and k not in _ENV_NOT_OURS]:
        leaked.append('%s: unset -> %s' % (key, _shown(key, os.environ.get(key))))
        os.environ.pop(key, None)
    if leaked and where and _ENV_LEAK_EXEMPT not in where:
        _ENV_LEAKS.append((where, leaked))


@pytest.fixture(autouse=True, scope='class')
def _env_isolation_class(request):
    saved = dict(os.environ)
    yield
    _restore_env(saved, 'setUpClass of %s' % request.node.nodeid)


@pytest.fixture(autouse=True)
def _env_isolation_function(request):
    saved = dict(os.environ)
    yield
    _restore_env(saved, request.node.nodeid)


def pytest_terminal_summary(terminalreporter):
    if _OUTSIDE_TMP:
        terminalreporter.section('database opened outside a temp dir',
                                 red=True)
        for where in _OUTSIDE_TMP:
            terminalreporter.write_line('    %s' % where)
        terminalreporter.write_line(
            '%d database file(s) opened by a test on a path the suite does not '
            'own. Every test brain is stamped into a temp dir; anything else '
            "is either the operator's own databases or a write that outlives "
            'the run. Use brain_env() — see tests/test_isolated_brain.py.'
            % len(_OUTSIDE_TMP))
    if not _ENV_LEAKS:
        return
    terminalreporter.section('environment leaks', red=True)
    for where, leaked in _ENV_LEAKS:
        terminalreporter.write_line('%s' % where)
        for change in leaked:
            terminalreporter.write_line('    %s' % change)
    terminalreporter.write_line(
        '%d scope(s) mutated os.environ without restoring it. The guard put it '
        'back, so nothing downstream broke — but restore it at the source '
        '(monkeypatch, patch.dict, or setUp/tearDown). '
        'BRAIN_TEST_ENV_LEAKS_OK=1 downgrades this to a report.'
        % len(_ENV_LEAKS))


def pytest_sessionfinish(session, exitstatus):
    if _ENV_LEAKS and not os.environ.get('BRAIN_TEST_ENV_LEAKS_OK'):
        session.exitstatus = 1
    if _OUTSIDE_TMP:
        session.exitstatus = 1


# Every database the suite opens lives in a temp dir. THE WALL, resource side.
#
# tests/test_isolated_brain.py forbids reaching the production clone, but it
# reads source: it cannot see an import inside an eval function that a test
# calls, nor any other route to the operator's databases. So the check that
# matters is not "who imported what" but "what did we open": every legitimate
# test brain is stamped into mkdtemp, and the operator's is not.
#
# The check sits on sqlite3.connect, the call every sqlite open in this
# process goes through. Brain opens with it, and so does the clone — which
# never opens a Brain on the operator's files: it snapshots them into mkdtemp
# through the backup API and opens the copy, so a check on the Brain
# constructor accepts it. What it cannot see: a plain file copy of a .db
# (shutil, cp) and anything a child process opens.
#
# This wraps the resource rather than auditing call sites, for the reason in
# id:24a48377 — a guard on one caller is bypassed by the next; the guard
# belongs on the thing guarded. Wrapping is done ONCE at import, before any
# test runs, and the wrapper only observes.
#
# Failures are collected and reported at the end rather than raised in place:
# raising inside connect() would surface as whatever that test was doing, and
# the report names the path and the test instead.
_OUTSIDE_TMP = []


def _install_brain_location_guard():
    import functools
    import sqlite3
    from urllib.parse import unquote

    orig = sqlite3.connect
    if getattr(orig, '_location_guarded', False):
        return
    tmp_root = os.path.realpath(tempfile.gettempdir())

    def file_of(database, uri):
        """The file a connect() call opens; None for an in-memory database or
        SQLite's private temporary one."""
        path = os.fsdecode(os.fspath(database))
        if uri and path.startswith('file:'):
            path, _, query = path[len('file:'):].partition('?')
            if 'mode=memory' in query:
                return None
            path = unquote(path)
        return None if path in ('', ':memory:') else path

    # `database` is required, as on the real connect(), and functools.wraps
    # carries __wrapped__ so inspect.signature reports the real signature: a
    # guard must not change the contract it is watching.
    @functools.wraps(orig)
    def guarded(database, *a, **kw):
        path = file_of(database, kw.get('uri', False))
        real = path and os.path.realpath(path)
        if real and not real.startswith(tmp_root + os.sep):
            where = '%s\n      opened by %s' % (
                real, os.environ.get('PYTEST_CURRENT_TEST', '?'))
            # One entry per file and test: a Brain opens several connections.
            if where not in _OUTSIDE_TMP:
                _OUTSIDE_TMP.append(where)
        return orig(database, *a, **kw)

    guarded._location_guarded = True
    # The report list rides on the wrapper, so a test can read the list this
    # wrapper appends to whatever name the conftest module was loaded under.
    guarded.outside_tmp = _OUTSIDE_TMP
    sqlite3.connect = guarded


# Live-brain aspects file guard. aspects_json_path() resolves from env at
# CALL time and falls back to the OPERATOR'S LIVE $BRAIN_DB_DIR/
# aspects_v1.json — and AspectRegistry (constructed by every Brain.__init__)
# runs reconcile_working_copy, which WRITES that file (seed heal). Any test
# that builds a raw Brain(db_path=tmp) without pinning ASPECTS_JSON_PATH
# therefore heals the live taxonomy from the repo seed under test
# (observed 2026-07-28: a worktree suite run stamped unmerged schema fields
# into the live file via test_prompt_sync / test_daemon).
#
# Pinned at import — before collection — rather than from a fixture, because
# setUpClass runs inside the class fixture and a function-scoped guard is too
# late to cover the raw Brain(db_path=tmp) that a setUpClass builds. Fills only
# when ABSENT, so deliberate overrides (BrainTestBase, a BrainEnv,
# run_aspect_cycles_on_clone) still win; _env_isolation restores this baseline
# after each of them, which is what makes the pin self-healing.
# The pin is also ANNOUNCED, in BRAIN_TEST_ASPECTS_DEFAULT_PIN. A BrainEnv
# honors a pre-set ASPECTS_JSON_PATH as "the caller owns the aspects location"
# and otherwise pins its own seeded copy — a safety default it cannot
# distinguish from a deliberate override would silently disable that, leaving
# every brain env in the process sharing this one file. Publishing the
# value lets it tell the two apart; absent (no conftest, e.g. an eval script),
# it falls back to the is-None test.
if not os.environ.get('ASPECTS_JSON_PATH'):
    _guard_dir = tempfile.mkdtemp(prefix='pytest_aspects_guard_')
    # Swept at exit: one dir per pytest invocation adds up silently — the
    # unconditional version of this line had left 1131 of them behind.
    atexit.register(shutil.rmtree, _guard_dir, True)
    os.environ['ASPECTS_JSON_PATH'] = os.path.join(_guard_dir, 'aspects_v1.json')
    os.environ['BRAIN_TEST_ASPECTS_DEFAULT_PIN'] = os.environ['ASPECTS_JSON_PATH']


# Hermetic config home + key. brain.llm_available gates surface/encode/S2/warms
# on a resolved sk-* key (keyless first-run onboarding); without pinning,
# real-Brain tests would behave differently on machines with vs without a key.
# A fake sk- prefix keeps the suite deterministic; unit tests mock LLM calls,
# so the fake never reaches the API.
#
# The pin covers the FILE as well as the variable: resolve_api_key() reads
# ${XDG_CONFIG_HOME:-~/.config}/brain/env on every llm_available check and
# writes the file's key into os.environ (the file wins — that is what makes
# key replacement work without a restart), so a real env file in reach swaps
# the fake for a real credential on the first test that touches it, and the
# leak guard above then reports what it caught. So the config home is a
# per-run dir holding the fake key and nothing secret — set unconditionally
# (at import no test has run; one that wants its own config home sets it
# later) and BEFORE servers.daemon_config is imported below, which reads the
# same file at import.
#
# The config this writes holds the fake key and NOTHING ELSE. It used to carry
# the operator's BRAIN_DB_DIR pointer forward so the production clone could
# find their brain from inside the suite; the suite no longer reaches that
# brain at all (tests/test_isolated_brain.py holds the wall), so the pointer
# has no consumer here and leaving it would keep the address in reach of
# every test for nothing. Tests that need a brain location set their own.
_HERMETIC_KEY = 'sk-test-hermetic-not-a-key'
_cfg_home = tempfile.mkdtemp(prefix='pytest_xdg_config_')
atexit.register(shutil.rmtree, _cfg_home, True)
os.makedirs(os.path.join(_cfg_home, 'brain'))
with open(os.path.join(_cfg_home, 'brain', 'env'), 'w') as _f:
    _f.write('ANTHROPIC_API_KEY=%s\n' % _HERMETIC_KEY)
os.environ['XDG_CONFIG_HOME'] = _cfg_home
os.environ['ANTHROPIC_API_KEY'] = _HERMETIC_KEY

# The brain location, pinned for the same reason. With BRAIN_DB_DIR unset,
# resolve_db_dir() walks its ladder: the config rungs are hermetic (the dir
# above), but the HOME rungs (~/.local/share/brain, then the legacy home-dir
# location) still find the operator's brain. Code that logs to or
# opens "the brain" without a test-supplied location reached production that
# way. An empty per-run dir now answers "where should the brain be" for all of
# it. Tests of the ladder set or clear the variable themselves, and
# resolve_db_dir(trust_env=False) (the eval clone's "where is an existing
# brain") skips this rung by design. Set unconditionally: a session's shell
# can carry the live location.
_db_home = tempfile.mkdtemp(prefix='pytest_brain_db_')
atexit.register(shutil.rmtree, _db_home, True)
os.environ['BRAIN_DB_DIR'] = _db_home


# The suite is an entity. The pins above keep it off the operator's config and
# brain; this keeps it off the operator's DAEMON. Left alone, `./dev` hands the
# suite production's port (brain-env.sh exports the uid formula), and the
# launchd label and every /tmp rendezvous file (PID, singleton and startup
# locks, recovery state, maintenance lock, status) are production's — so one
# test that forgets a patch kickstarts, kills or writes to the live daemon.
# BRAIN_INSTANCE keys all of them, both installers refuse an instance, and the
# resolver never walks its ladder for one: its brain is its own BRAIN_DB_DIR.
# daemon_config refuses an instance on production's port, hence a free one. An
# unpatched daemon call then reaches nothing: a dead port, a label launchd
# never loaded. BRAIN_TMP_DIR moves the per-session ephemera off /tmp.
#
# Set unconditionally — an entity's shell carries that entity's identity — and
# before the imports below: DAEMON_PORT and LAUNCHD_LABEL are fixed at import.
# A test of production's unkeyed behavior pops BRAIN_INSTANCE in the env of
# the subprocess it runs (the installer, relocate, port and ladder helpers).
def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]
    finally:
        s.close()


os.environ['BRAIN_INSTANCE'] = 'pytest-%d' % os.getpid()
os.environ['BRAIN_DAEMON_PORT'] = str(_free_port())
_tmp_home = tempfile.mkdtemp(prefix='pytest_brain_tmp_')
atexit.register(shutil.rmtree, _tmp_home, True)
os.environ['BRAIN_TMP_DIR'] = _tmp_home


# CPU-only pin. daemon_config is the one module under servers/ that writes
# os.environ at IMPORT (DAEMON_CPU_ENV — ORT_DISABLE_ALL_ACCELERATORS is the
# load-bearing SIGABRT guard on Apple Silicon), and most test files reach it
# through a lazy in-function import. A process-wide invariant established
# inside a test is one _env_isolation restore away from being deleted for good,
# because sys.modules caching means the module body never runs twice — so
# import it here, before anything can snapshot, and the invariant is part of
# every baseline instead of a side effect of which files got collected.
import servers.daemon_config  # noqa: E402,F401
# daemon_launch fixes LAUNCHD_LABEL at import from the instance above; imported
# here so the label is the suite's for the whole run, whichever test (or a
# test's env change) would otherwise import it first.
import servers.daemon_launch  # noqa: E402,F401



_BYPASS_ENV = 'BRAIN_ALLOW_ANY_PYTHON'


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _expected_python():
    return os.path.join(_repo_root(), 'venv', 'bin', 'python')


def pytest_configure(config):
    """Fail fast if the wrong Python is running the tests."""
    if os.environ.get(_BYPASS_ENV):
        return

    expected = _expected_python()
    actual = os.path.realpath(sys.executable)
    expected_real = os.path.realpath(expected) if os.path.exists(expected) else expected

    # If the bundled venv doesn't exist yet (fresh clone), just warn.
    if not os.path.exists(expected):
        sys.stderr.write(
            '\n[conftest] WARN: bundled venv not found at %s — '
            'running under %s (%s).\n'
            '  Bootstrap via hooks/scripts/ensure-runtime.sh or set BRAIN_ALLOW_ANY_PYTHON=1.\n\n'
            % (expected, actual, '.'.join(map(str, sys.version_info[:3]))))
        return

    if actual != expected_real:
        ver = '.'.join(map(str, sys.version_info[:3]))
        raise RuntimeError(
            '\n'
            '========================================================\n'
            ' WRONG PYTHON — tests refuse to run\n'
            '========================================================\n'
            '  expected: %s  (Python 3.11, not SIP-protected)\n'
            '  got:      %s  (Python %s)\n'
            '\n'
            '  Run tests via:    ./dev pytest ...\n'
            '  Or source:        source hooks/scripts/brain-env.sh\n'
            '  Or bypass once:   %s=1 pytest ...\n'
            '========================================================\n'
            % (expected_real, actual, ver, _BYPASS_ENV))

    # Sanity: reject SIP-protected system Pythons even by chance.
    if '/Xcode.app/' in actual or actual.startswith('/Applications/'):
        raise RuntimeError(
            '[conftest] Tests running under SIP-protected Python (%s). '
            'Debuggers cannot attach. Use ./dev pytest ...' % actual)


# ── One heavy suite run at a time, per machine ───────────────────────
#
# Several streams work this repo at once, each in its own worktree, and a
# `pytest` in one is invisible to a `pytest` in another. Measured 2026-09-21:
# two suites overlapping took 26:14 at 27% CPU against a 3:55 baseline — a 6x
# wall-clock loss for work that was already serial on the same 12 cores. The
# same overlap also produced a test failure, because a shared tmp root gets
# written by whichever run happens to be alive (fixed in e1ad9c7, but the
# class of collision is what this lock removes).
#
# So a big run takes an exclusive lock and a second one waits for it. Small
# runs — the targeted tier everyone lives in — are let straight through: they
# do not cause the thrash, and making a 20-test check queue behind a full
# suite is how a guard rail gets bypassed.
#
# The lock lives in the per-user tmp root, which is precisely the sharing
# scope we want: every stream of this operator, no matter which worktree.
_SUITE_LOCK_MIN_ITEMS = 300
_SUITE_LOCK_WAIT_S = 20 * 60
_SUITE_LOCK_ENV = 'BRAIN_TEST_NO_SUITE_LOCK'
_suite_lock_fd = None


def _suite_lock_path():
    return os.path.join(tempfile.gettempdir(), 'brain-suite.lock')


def _suite_lock_holder():
    """Whatever the holder wrote about itself — best effort, never raises."""
    try:
        with open(_suite_lock_path()) as fh:
            return fh.read().strip() or 'unknown'
    except Exception:
        return 'unknown'


# trylast, and it is load-bearing: `-k`/`-m` deselection is itself a
# pytest_collection_modifyitems implementation, and a conftest hook runs BEFORE
# the builtin ones. Without trylast, `pytest tests/ -k one_test` counts 3770
# items and takes the lock this threshold exists to skip — measured, it waited.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    """Serialize suite-scale runs across worktrees. See the block comment."""
    global _suite_lock_fd
    if os.environ.get(_SUITE_LOCK_ENV) or os.environ.get('PYTEST_XDIST_WORKER'):
        return
    if config.getoption('collectonly', False):
        return          # collection runs nothing; it has no cores to fight over
    if len(items) < _SUITE_LOCK_MIN_ITEMS:
        return

    # Every failure below falls through to running UNLOCKED. This is a
    # scheduling optimization: a suite that refuses to start because a lock
    # file misbehaved is a worse outcome than two suites sharing the cores.
    try:
        fd = os.open(_suite_lock_path(), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        sys.stderr.write('[conftest] suite lock unavailable (%s) — running unlocked.\n' % exc)
        return

    deadline = time.time() + _SUITE_LOCK_WAIT_S
    attempts = 0
    announced = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                # Not "someone holds it" — no advisory locking on this
                # filesystem, or an interrupted call. Waiting 20 minutes for a
                # holder that cannot exist is the worst of both worlds.
                sys.stderr.write(
                    '[conftest] suite lock not supported here (%s) — running unlocked.\n' % exc)
                os.close(fd)
                return
            attempts += 1
            # Announce on the SECOND failure: the holder writes its identity
            # just after it acquires, so a read in that window reports the
            # PREVIOUS holder. One sleep is enough to be reading the live one.
            if attempts > 1 and not announced:
                sys.stderr.write(
                    '\n[conftest] another suite run holds the machine lock — waiting.\n'
                    '  holder: %s\n'
                    '  skip with %s=1 (both runs then share the cores)\n\n'
                    % (_suite_lock_holder(), _SUITE_LOCK_ENV))
                sys.stderr.flush()
                announced = True
            if time.time() >= deadline:
                sys.stderr.write(
                    '[conftest] still locked after %d min — running anyway.\n'
                    % (_SUITE_LOCK_WAIT_S // 60))
                sys.stderr.flush()
                os.close(fd)
                return
            time.sleep(2)

    # Held for the life of the process; the OS drops it when we exit, which is
    # also what makes a crashed run release it.
    _suite_lock_fd = fd
    os.ftruncate(fd, 0)
    os.write(fd, ('pid %d · %s · %d tests · started %s\n'
                  % (os.getpid(), _repo_root(), len(items),
                     time.strftime('%Y-%m-%d %H:%M:%S'))).encode())
    if announced:
        sys.stderr.write('[conftest] lock acquired, starting.\n')
        sys.stderr.flush()


# Installed at IMPORT, like the pins above and for the same reason: a
# pytest_configure hook on a non-initial conftest is too late and was measured
# not to run at all here, and a fixture is later still — setUpClass builds
# brains before any function-scoped fixture exists.
_install_brain_location_guard()
