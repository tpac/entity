"""TrackedRLock — RLock that exposes its current holder for diagnostics.

Stand-in for `threading.RLock()` at any site where you want to answer
"who is holding this and for how long?" from another thread. The brain's
`write_lock` uses this so the bg_writer's stall watchdog can log who
was holding when it tried to drain.

Implementation note: threading.RLock doesn't expose its recursion count
to Python. We track it ourselves alongside the underlying lock. Holder
state is cleared only when the outermost release runs (count → 0), so
re-entrant acquires by the same thread don't fool the diagnostic into
thinking the lock was released and reacquired.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class TrackedRLock:
    """RLock with holder + held-since diagnostics."""

    def __init__(self) -> None:
        self._rlock = threading.RLock()
        self._count = 0
        self._holder: Optional[str] = None
        self._held_since: Optional[float] = None
        # Read access to _holder / _held_since must be coherent across
        # threads. The underlying RLock guards writes; this Lock guards
        # the metadata reads/writes that diagnostics rely on.
        self._meta_lock = threading.Lock()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if timeout == -1:
            ok = self._rlock.acquire(blocking)
        else:
            ok = self._rlock.acquire(blocking, timeout)
        if ok:
            with self._meta_lock:
                if self._count == 0:
                    self._holder = threading.current_thread().name
                    self._held_since = time.time()
                self._count += 1
        return ok

    def release(self) -> None:
        # Clear metadata when the outermost release runs. The underlying
        # RLock raises on imbalanced release; we don't catch that — let
        # it surface, that's a bug in the caller.
        with self._meta_lock:
            self._count -= 1
            if self._count == 0:
                self._holder = None
                self._held_since = None
        self._rlock.release()

    def __enter__(self) -> 'TrackedRLock':
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def holder(self) -> Optional[str]:
        """Name of the thread currently holding the lock, or None."""
        with self._meta_lock:
            return self._holder

    @property
    def held_for_ms(self) -> Optional[int]:
        """Milliseconds since the current holder first acquired, or None."""
        with self._meta_lock:
            if self._held_since is None:
                return None
            return int((time.time() - self._held_since) * 1000)

    def snapshot(self) -> dict:
        """Atomic view of (holder, held_for_ms, recursion_depth). Use
        this from diagnostics — single lock acquisition, consistent
        view, no torn reads."""
        with self._meta_lock:
            if self._held_since is None:
                return {'holder': None, 'held_for_ms': None, 'depth': 0}
            return {
                'holder': self._holder,
                'held_for_ms': int((time.time() - self._held_since) * 1000),
                'depth': self._count,
            }
