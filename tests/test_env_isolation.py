"""Contract: conftest's env guard makes an unrestored env write impossible.

os.environ is process-global and pytest runs the suite in one process, so a
test that sets a variable and doesn't put it back hands that value to every
test after it. That is not hypothetical here — four classes set $BRAIN_DB_DIR
in setUpClass and rmtree'd the directory in tearDownClass, leaving the rest of
the suite pointed at a directory that no longer existed, and resolve_db_dir()
trusts that variable unconditionally (D-13's top rung).

conftest closes it with a snapshot/restore pair — one class-scoped (wraps
setUpClass/tearDownClass), one function-scoped (wraps each test). These tests
assert the guard's BEHAVIOUR, so deleting either fixture from conftest fails
them: a gate that goes green when the mechanism it watches disappears is not a
gate.

Ordering: each leak source is followed by the item that checks it escaped
nothing, and pytest runs items in collection (file) order — requirements-test
pins pytest + pytest-timeout only, no randomizer. A check whose source was
deselected (`-k`, `--lf`, a pasted node id) would pass while asserting about
state nothing perturbed, so each source records that it ran and each check
skips rather than going green on an unexercised guard.
"""
import os
import unittest

import pytest

from servers.daemon_config import resolve_db_dir

_CLASS_PROBE = 'BRAIN_TEST_ENV_PROBE_CLASS'
_FUNC_PROBE = 'BRAIN_TEST_ENV_PROBE_FUNC'

_RAN = set()


def _needs(source):
    if source not in _RAN:
        pytest.skip('leak source %r was not collected — this check would pass '
                    'vacuously' % source)


class ClassScopeLeakSource(unittest.TestCase):
    """The exact shape of the bug: setUpClass writes, tearDownClass forgets."""

    @classmethod
    def setUpClass(cls):
        os.environ[_CLASS_PROBE] = 'written-by-setUpClass'
        _RAN.add('class')

    def test_class_scope_write_is_visible_within_the_class(self):
        self.assertEqual(os.environ.get(_CLASS_PROBE), 'written-by-setUpClass')


def test_class_scope_write_did_not_escape_the_class():
    """Fails if the class-scoped guard is removed — setUpClass ran before this."""
    _needs('class')
    assert _CLASS_PROBE not in os.environ, (
        'setUpClass wrote %s=%r and it survived tearDownClass — the '
        'class-scoped guard in conftest is missing or disarmed'
        % (_CLASS_PROBE, os.environ.get(_CLASS_PROBE)))


def test_function_scope_leak_source():
    os.environ[_FUNC_PROBE] = 'written-by-a-test-body'
    _RAN.add('func')


def test_function_scope_write_did_not_escape_the_test():
    """Fails if the function-scoped guard is removed."""
    _needs('func')
    assert _FUNC_PROBE not in os.environ, (
        'a test body wrote %s=%r and it survived into the next test — the '
        'function-scoped guard in conftest is missing or disarmed'
        % (_FUNC_PROBE, os.environ.get(_FUNC_PROBE)))


def test_deletion_leak_source():
    """The other direction: a test that POPS a var the suite relies on."""
    os.environ.pop('ASPECTS_JSON_PATH', None)
    _RAN.add('deletion')


def test_deleted_var_was_restored():
    """Fails if the guard only strips additions and never re-adds removals."""
    _needs('deletion')
    assert os.environ.get('ASPECTS_JSON_PATH'), (
        'the previous test popped ASPECTS_JSON_PATH and it was not put back — '
        'the guard restores additions but not deletions')


def test_aspects_pin_never_points_at_the_live_taxonomy():
    """conftest pins ASPECTS_JSON_PATH at import, before collection, because
    setUpClass builds Brains before any fixture could pin it — and every
    Brain.__init__ reconciles (WRITES) the resolved aspects file."""
    pinned = os.environ.get('ASPECTS_JSON_PATH')
    assert pinned, 'ASPECTS_JSON_PATH is unpinned — a raw Brain() in a ' \
                   'setUpClass would heal the operator\'s live taxonomy'
    # trust_env=True mirrors aspect_store.aspects_json_path(), which is the
    # call that would land the heal — resolving it any other way tests a
    # directory the heal would never touch.
    live_dir = resolve_db_dir()
    assert os.path.dirname(os.path.abspath(pinned)) != os.path.abspath(live_dir), (
        'ASPECTS_JSON_PATH points into the live brain dir (%s)' % live_dir)


def test_cpu_only_pin_survives_the_env_guards():
    """conftest imports daemon_config so its import-time DAEMON_CPU_ENV write
    lands in every snapshot. Without that, the first lazy import inside a test
    has the vars stripped at teardown and sys.modules caching means they never
    come back — silently disarming the SIGABRT guard for the rest of the run."""
    from servers.daemon_config import DAEMON_CPU_ENV
    missing = [k for k in DAEMON_CPU_ENV if not os.environ.get(k)]
    assert not missing, (
        'CPU-only invariant stripped from the environment: %s — conftest must '
        'import servers.daemon_config before anything snapshots' % missing)


def test_the_suite_runs_as_its_own_entity():
    """conftest gives the suite its own instance, so a test that forgets a
    patch reaches nothing of the operator's: the daemon port, the launchd label
    and every /tmp rendezvous file are the suite's, never production's."""
    from servers import daemon_config as dc
    from servers import daemon_launch as dl
    inst = os.environ.get('BRAIN_INSTANCE', '')
    assert inst == 'pytest-%d' % os.getpid(), 'the suite runs unkeyed: %r' % inst
    assert dc.DAEMON_PORT == int(os.environ['BRAIN_DAEMON_PORT'])
    assert dc.DAEMON_PORT != 47200 + os.getuid() % 100, "production's port"
    assert dl.LAUNCHD_LABEL == 'com.brain.daemon.' + inst
    for path in (dc.get_pid_path(), dc.get_lock_path(),
                 dc.get_startup_lock_path(), dc.get_status_path(),
                 dc.get_maintenance_path(), dc.get_recovery_state_path()):
        assert path.endswith('-%s%s' % (inst, os.path.splitext(path)[1])), (
            "production's rendezvous file: %s" % path)
    assert dc.brain_tmp_dir() != '/tmp'
