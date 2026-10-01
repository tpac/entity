"""Tests for the unified brain.write_lock — protects against the regression
where _write_lock lived on the daemon and only daemon-mediated writes were
serialized. Now ALL writers (daemon dispatch, S2 encoder, embed_queue,
autosave) take the same brain.write_lock, so concurrent writers can't
interleave at the cross-statement level.

Specifically covers:
- brain.write_lock exists and is an RLock (re-entrant — same-thread re-acquire
  must not deadlock, since some paths legitimately re-enter)
- get_or_create_session is race-safe under concurrent first-touch
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class WriteLockTests(BrainTestBase):
    needs_embedder = False

    def test_write_lock_exists_on_brain(self):
        """The lock is a brain attribute, not daemon-only."""
        self.assertTrue(hasattr(self.brain, 'write_lock'),
                        "Brain.write_lock must exist for cross-writer serialization")

    def test_write_lock_is_reentrant(self):
        """RLock so a thread that holds the lock can call into a brain method
        that also wants to acquire it — without deadlocking on itself."""
        with self.brain.write_lock:
            # Re-entry — must not block. If this is a regular Lock, this
            # acquire would block the thread on itself and the test hangs
            # until pytest's timeout.
            acquired = self.brain.write_lock.acquire(blocking=False)
            try:
                self.assertTrue(acquired,
                    "write_lock must be re-entrant — same-thread re-acquire failed")
            finally:
                if acquired:
                    self.brain.write_lock.release()

    def test_concurrent_holders_block(self):
        """Two threads must not hold the lock at the same time (cross-thread).

        Verifies the lock is a real lock — not a no-op stub. Holds the lock
        from thread A, asserts a thread B's non-blocking acquire fails.
        """
        held = threading.Event()
        release_now = threading.Event()
        b_acquired = []

        def thread_a():
            with self.brain.write_lock:
                held.set()
                release_now.wait(timeout=2.0)

        def thread_b():
            held.wait(timeout=1.0)
            # Non-blocking acquire — should fail because thread A holds it.
            got = self.brain.write_lock.acquire(blocking=False)
            b_acquired.append(got)
            if got:
                self.brain.write_lock.release()

        a = threading.Thread(target=thread_a)
        b = threading.Thread(target=thread_b)
        a.start()
        b.start()
        b.join(timeout=2.0)
        release_now.set()
        a.join(timeout=2.0)

        self.assertEqual(b_acquired, [False],
            "thread B got the lock while thread A held it — write_lock is not actually serializing")


class GetOrCreateSessionRaceTests(BrainTestBase):
    """get_or_create_session must be safe under concurrent first-touch with
    the same session_id — INSERT OR IGNORE makes the row creation atomic."""

    needs_embedder = False

    def test_concurrent_first_touch_same_id_yields_one_row(self):
        session_id = 'race-test-' + os.urandom(4).hex()
        N = 8
        contexts = []
        errors = []
        barrier = threading.Barrier(N)

        def worker():
            try:
                barrier.wait(timeout=2.0)
                ctx = self.brain.get_or_create_session(session_id)
                contexts.append(ctx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(errors, [],
            "concurrent get_or_create_session raised: %s" % errors)
        self.assertEqual(len(contexts), N,
            "expected %d contexts back, got %d" % (N, len(contexts)))

        # All workers must see the same session_id.
        ids = {c.session_id for c in contexts}
        self.assertEqual(ids, {session_id})

        # Exactly one row exists in session_state (INSERT OR IGNORE was
        # the atomic guard).
        rows = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM session_state WHERE session_id = ? AND key = ?",
            (session_id, '_session_context')).fetchone()
        self.assertEqual(rows[0], 1,
            "expected exactly 1 session_state row, got %d — INSERT OR IGNORE failed to atomicize creation" % rows[0])

    def test_default_state_has_zero_stop_counter(self):
        """Brand-new session has stop_counter=0 and an empty fatigue dict."""
        session_id = 'default-test-' + os.urandom(4).hex()
        ctx = self.brain.get_or_create_session(session_id)
        self.assertEqual(ctx.stop_counter, 0)
        self.assertEqual(ctx.fatigue, {})

    def test_existing_session_state_preserved(self):
        """If a session_state row already exists with non-default values,
        get_or_create_session must NOT overwrite it (INSERT OR IGNORE)."""
        session_id = 'preserve-test-' + os.urandom(4).hex()
        # Create + mutate
        ctx = self.brain.get_or_create_session(session_id)
        ctx.stop_counter = 42
        ctx.fatigue = {'node-x': 7}
        ctx.save(self.brain._session_state)
        # Re-fetch via get_or_create — should see the modified state
        ctx2 = self.brain.get_or_create_session(session_id)
        self.assertEqual(ctx2.stop_counter, 42,
            "get_or_create_session overwrote existing stop_counter — INSERT OR IGNORE is broken")
        self.assertEqual(ctx2.fatigue, {'node-x': 7})

    def test_autosave_and_create_share_logs_conn_safely(self):
        """save_session_contexts (autosave path) and get_or_create_session both
        write session_state through the DAL write boundary (logs_write_lock +
        logs_conn_w). Under concurrent unserialized access, same-connection
        writers collide on transaction state ("cannot start a transaction
        within a transaction") — the boundary must serialize them.

        Barrier-synced N threads, each looping over BOTH writers, to force the
        overlapping-transaction window deterministically (loose 2-thread timing
        is a false-green — it rarely hits the window).
        """
        # Seed several cached contexts so each save_session_contexts() flush
        # does real per-row work — widens the transaction window.
        for _ in range(10):
            self.brain.get_or_create_session('seed-' + os.urandom(4).hex())

        N = 8
        errors = []
        barrier = threading.Barrier(N)

        def worker():
            try:
                barrier.wait(timeout=2.0)
                for _ in range(20):
                    self.brain.get_or_create_session('race2-' + os.urandom(4).hex())
                    self.brain.save_session_contexts()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        self.assertEqual(errors, [],
            "concurrent autosave + create raised (logs_conn transaction race): %s" % errors)


if __name__ == '__main__':
    unittest.main()
