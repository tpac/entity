"""Tests for TrackedRLock — the holder-tracking RLock used by brain.write_lock.

The lock-holder snapshot is the diagnostic that lets the bg_writer
stall watchdog answer "who was holding when we tried to drain?" If
the snapshot lies (e.g., stale holder after release, lost holder on
re-entrant release), the watchdog logs misleading info. These tests
lock the contract.
"""

import threading
import time
import unittest

from servers.tracked_lock import TrackedRLock


class TestTrackedRLockSingleThread(unittest.TestCase):
    def test_initial_snapshot_is_empty(self):
        lock = TrackedRLock()
        snap = lock.snapshot()
        self.assertIsNone(snap['holder'])
        self.assertIsNone(snap['held_for_ms'])
        self.assertEqual(snap['depth'], 0)

    def test_acquired_snapshot_names_main_thread(self):
        lock = TrackedRLock()
        with lock:
            snap = lock.snapshot()
            self.assertEqual(snap['holder'], threading.current_thread().name)
            self.assertEqual(snap['depth'], 1)
            self.assertIsNotNone(snap['held_for_ms'])

    def test_release_clears_holder(self):
        lock = TrackedRLock()
        with lock:
            pass
        snap = lock.snapshot()
        self.assertIsNone(snap['holder'])
        self.assertEqual(snap['depth'], 0)

    def test_reentrant_acquire_tracks_depth(self):
        lock = TrackedRLock()
        with lock:
            self.assertEqual(lock.snapshot()['depth'], 1)
            with lock:
                self.assertEqual(lock.snapshot()['depth'], 2)
                with lock:
                    self.assertEqual(lock.snapshot()['depth'], 3)
                self.assertEqual(lock.snapshot()['depth'], 2)
            self.assertEqual(lock.snapshot()['depth'], 1)
        self.assertEqual(lock.snapshot()['depth'], 0)

    def test_reentrant_holds_holder_across_inner_release(self):
        """Holder must NOT change when a nested release runs — only when
        the outermost release brings depth back to 0."""
        lock = TrackedRLock()
        with lock:
            initial_holder = lock.snapshot()['holder']
            with lock:
                pass  # inner release
            # depth back to 1; holder unchanged
            snap = lock.snapshot()
            self.assertEqual(snap['holder'], initial_holder)
            self.assertEqual(snap['depth'], 1)


class TestTrackedRLockCrossThread(unittest.TestCase):
    def test_snapshot_sees_other_thread_as_holder(self):
        """When thread A holds, thread B's snapshot reports A as holder."""
        lock = TrackedRLock()
        worker_grabbed = threading.Event()
        let_worker_go = threading.Event()

        def worker():
            with lock:
                worker_grabbed.set()
                let_worker_go.wait(timeout=2.0)

        t = threading.Thread(target=worker, name='worker-thread', daemon=True)
        t.start()
        self.assertTrue(worker_grabbed.wait(timeout=2.0))
        # While worker holds, our snapshot sees it.
        snap = lock.snapshot()
        self.assertEqual(snap['holder'], 'worker-thread')
        self.assertEqual(snap['depth'], 1)
        let_worker_go.set()
        t.join(timeout=2.0)
        # After release, snapshot is empty.
        self.assertIsNone(lock.snapshot()['holder'])

    def test_held_for_ms_grows_over_time(self):
        """held_for_ms reflects time since acquire, not last snapshot."""
        lock = TrackedRLock()
        with lock:
            snap1 = lock.snapshot()
            time.sleep(0.1)
            snap2 = lock.snapshot()
            self.assertGreaterEqual(snap2['held_for_ms'], snap1['held_for_ms'] + 90)


class TestTrackedRLockSerialization(unittest.TestCase):
    def test_blocks_until_released(self):
        """Two threads cannot hold the lock simultaneously."""
        lock = TrackedRLock()
        order = []

        def first():
            with lock:
                order.append('first-in')
                time.sleep(0.1)
                order.append('first-out')

        def second():
            with lock:
                order.append('second-in')

        t1 = threading.Thread(target=first, daemon=True)
        t2 = threading.Thread(target=second, daemon=True)
        t1.start()
        time.sleep(0.02)  # ensure t1 grabs first
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        # Either ordering proves serialization (first-out must come before second-in).
        self.assertIn('first-out', order)
        self.assertIn('second-in', order)
        self.assertLess(order.index('first-out'), order.index('second-in'))


if __name__ == '__main__':
    unittest.main()
