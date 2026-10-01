"""Contract: the machine-wide suite lock skips what it promises to skip.

conftest serializes suite-scale pytest runs across worktrees, because two
overlapping suites on the same twelve cores measured 26:14 against a 3:55
baseline. The lock only works if it is invisible to everything else — the
targeted tier everyone lives in, a collection listing, an xdist worker — and
every one of those exemptions is a silent failure when it breaks: the suite
still passes, it just waits when it should not, and the fix people reach for
is to set BRAIN_TEST_NO_SUITE_LOCK=1 forever.

One of these started as a real defect. `-k` deselection is itself a
pytest_collection_modifyitems implementation and a conftest hook runs BEFORE
the builtin ones, so the first version counted the unfiltered 3770 items and
made `pytest tests/ -k one_test` wait for a full suite. `trylast` is what puts
the count after deselection.

NO TEST HERE ACQUIRES THE LOCK, deliberately. flock is per open-file-
description, so a second descriptor in this same process cannot take a lock
the running suite already holds — a positive-path test would block for the
full 20-minute timeout rather than fail. The exemptions and the give-up paths
are what these cover; the acquire path is exercised by every full-suite run.
"""
import errno
import os
import unittest
from unittest import mock

from tests import conftest


class _Config:
    """Minimal stand-in for pytest's Config — getoption is all the hook uses."""

    def __init__(self, collectonly=False):
        self._collectonly = collectonly

    def getoption(self, name, default=None):
        if name == 'collectonly':
            return self._collectonly
        return default


def _items(n):
    return [object()] * n


class TestSuiteLockExemptions(unittest.TestCase):
    """Each case must return without touching the lock file at all."""

    def setUp(self):
        self.addCleanup(setattr, conftest, '_suite_lock_fd',
                        conftest._suite_lock_fd)

    def _assert_no_lock_attempt(self, config, items, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch.object(conftest.os, 'open') as opened:
            conftest.pytest_collection_modifyitems(None, config, items)
        self.assertFalse(
            opened.called,
            'the hook opened the lock file for a run it should have skipped')

    def test_opt_out_env_skips_the_lock(self):
        self._assert_no_lock_attempt(
            _Config(), _items(conftest._SUITE_LOCK_MIN_ITEMS + 1),
            {conftest._SUITE_LOCK_ENV: '1'})

    def test_xdist_worker_skips_the_lock(self):
        # The controller takes it; a worker waiting on its own controller is a
        # deadlock, not a queue.
        self._assert_no_lock_attempt(
            _Config(), _items(conftest._SUITE_LOCK_MIN_ITEMS + 1),
            {'PYTEST_XDIST_WORKER': 'gw0'})

    def test_collect_only_skips_the_lock(self):
        self._assert_no_lock_attempt(
            _Config(collectonly=True),
            _items(conftest._SUITE_LOCK_MIN_ITEMS + 1))

    def test_a_run_below_the_threshold_skips_the_lock(self):
        self._assert_no_lock_attempt(
            _Config(), _items(conftest._SUITE_LOCK_MIN_ITEMS - 1))


class TestSuiteLockRunsAfterDeselection(unittest.TestCase):

    def test_the_hook_is_trylast(self):
        """Counting before `-k` deselection makes every targeted run wait.

        White-box on purpose: the behaviour only shows up in a nested pytest
        run, and the marker is the whole mechanism. Dropping it is a one-word
        edit whose symptom is a stall nobody attributes to this file.
        """
        opts = getattr(conftest.pytest_collection_modifyitems, 'pytest_impl', {})
        self.assertTrue(
            opts.get('trylast'),
            'pytest_collection_modifyitems must be trylast, or it counts '
            'items before -k/-m deselection and locks for a one-test run')


class TestSuiteLockDegradesRatherThanBlocks(unittest.TestCase):

    def setUp(self):
        self.addCleanup(setattr, conftest, '_suite_lock_fd',
                        conftest._suite_lock_fd)

    def test_a_filesystem_without_locking_runs_unlocked(self):
        """ENOLCK is not "someone holds it" — waiting 20 min would be absurd.

        A container or NFS-mounted TMPDIR has no advisory locking, so treating
        every OSError as contention stalls every suite-scale run on that host
        for the full timeout before doing any work.
        """
        conftest._suite_lock_fd = None
        with mock.patch.object(conftest.fcntl, 'flock',
                               side_effect=OSError(errno.ENOLCK, 'no locks')), \
                mock.patch.object(conftest.time, 'sleep') as slept:
            conftest.pytest_collection_modifyitems(
                None, _Config(), _items(conftest._SUITE_LOCK_MIN_ITEMS + 1))
        self.assertFalse(slept.called, 'it waited on a lock that cannot exist')
        self.assertIsNone(conftest._suite_lock_fd)

    def test_an_unopenable_lock_file_runs_unlocked(self):
        """The lock is a scheduling optimization; it must never fail a run."""
        conftest._suite_lock_fd = None
        with mock.patch.object(conftest.os, 'open',
                               side_effect=OSError(errno.EROFS, 'read-only')):
            conftest.pytest_collection_modifyitems(
                None, _Config(), _items(conftest._SUITE_LOCK_MIN_ITEMS + 1))
        self.assertIsNone(conftest._suite_lock_fd)


if __name__ == '__main__':
    unittest.main()
