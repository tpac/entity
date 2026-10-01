"""run_unit_in_background's on_complete callback contract.

The S1 Scribe (and any in-process scale unit) counts its run toward the S2
activity gate ON COMPLETION (not dispatch), and only when it actually wrote
material. run_unit_in_background fires on_complete(write_actions) after a
successful unit.run(), in the background thread. These pin that contract without
a real DB or flakiness: the runner releases `lock` in its finally, so
re-acquiring it deterministically waits for the thread to finish.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales import runner


class _StubBrain:
    """Minimal brain — run_unit_in_background only touches brain._log_error,
    and only on the crash path."""
    def _log_error(self, *a, **k):
        pass


class _FakeUnit:
    """An IntegrationUnit stand-in: run() returns a result dict (or raises)."""
    def __init__(self, run_fn):
        self._run_fn = run_fn
        self.brain = _StubBrain()

    def run(self):
        return self._run_fn()


def _drive(run_fn, on_complete):
    """Run run_unit_in_background to completion (sync via the lock) and return."""
    lock = threading.Lock()
    lock.acquire()  # caller holds it; the runner releases in finally
    runner.run_unit_in_background(
        _FakeUnit(run_fn), name='test', lock=lock, on_complete=on_complete)
    # Block until the thread's finally releases the lock → run complete.
    assert lock.acquire(timeout=5.0), "background thread never finished"


def test_on_complete_receives_write_actions():
    got = []
    _drive(lambda: {'write_actions': 5, 'actions': 7}, on_complete=got.append)
    assert got == [5], "on_complete must receive the run's write_actions"


def test_on_complete_zero_when_no_writes():
    got = []
    _drive(lambda: {'actions': 2},  # no write_actions key → 0
           on_complete=got.append)
    assert got == [0], "a run that wrote nothing reports write_actions=0"


def test_on_complete_not_called_when_run_raises():
    got = []

    def boom():
        raise RuntimeError("encoder died")

    _drive(boom, on_complete=got.append)
    assert got == [], "on_complete must NOT fire when run() raised"


def test_callback_error_does_not_break_thread():
    # A raising callback must be swallowed (logged) — the lock still releases.
    def bad(_write_actions):
        raise ValueError("callback boom")

    # _drive asserts the lock is reacquired, i.e. the thread finished cleanly
    # despite the callback raising.
    _drive(lambda: {'write_actions': 1}, on_complete=bad)
