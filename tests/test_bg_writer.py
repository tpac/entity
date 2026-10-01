"""Tests for the background-writer architecture (Phases 1-5, 2026-05-18).

Covers:
  - Brain opens `conn_bg_writer` on boot with correct PRAGMAs.
  - `_mark_accessed` is pure enqueue (no DB I/O on hot path).
  - Dedup is keyed on (node_id, session_id).
  - Drain produces atomic +1 UPDATEs per dedup'd (node, session) pair.
  - mark_accessed already uses atomic +1 (no read-modify-write).
  - Drain rollback on transaction failure + loud log.
  - Concurrent enqueue from multiple threads doesn't lose work.
  - Worker loop survives an exception in drain_once.
"""

import threading
import time
import unittest

from tests.brain_test_base import BrainTestBase

from servers import recall_write_queue


class TestBgWriterConnection(BrainTestBase):
    """Phase 1 — conn_bg_writer opens correctly on boot."""

    needs_embedder = False

    def test_bg_writer_opens_on_boot(self):
        self.assertIsNotNone(self.brain.conn_bg_writer)
        # Basic query works
        r = self.brain.conn_bg_writer.execute('SELECT 1').fetchone()
        self.assertEqual(r, (1,))

    def test_bg_writer_pragmas(self):
        wal = self.brain.conn_bg_writer.execute(
            'PRAGMA journal_mode').fetchone()[0].lower()
        self.assertEqual(wal, 'wal')
        bt = self.brain.conn_bg_writer.execute(
            'PRAGMA busy_timeout').fetchone()[0]
        self.assertEqual(bt, 30000)
        fk = self.brain.conn_bg_writer.execute(
            'PRAGMA foreign_keys').fetchone()[0]
        self.assertEqual(fk, 1)


class TestEnqueueNoDbIo(BrainTestBase):
    """Phase 5 — hot path enqueues without touching the DB."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_enqueue_access_does_not_touch_db(self):
        # sqlite3.Connection.execute is read-only; use the round-trip
        # observation instead. Snapshot WAL frame counter before enqueue,
        # confirm no commits happened post-enqueue. Equivalent semantic:
        # "the call did not push any data through SQLite".
        before_changes = self.brain.conn.execute(
            "PRAGMA data_version").fetchone()[0]

        recall_write_queue.enqueue_access(
            'abc12345', 'sess1', '2026-05-18T12:00:00')

        after_changes = self.brain.conn.execute(
            "PRAGMA data_version").fetchone()[0]
        # data_version increments on every commit to the database.
        # enqueue_access is pure in-memory — data_version unchanged.
        self.assertEqual(before_changes, after_changes,
                         'enqueue_access touched the DB (data_version moved)')
        self.assertEqual(recall_write_queue.queue_depth(), 1)


class TestDedupPerSession(BrainTestBase):
    """Phase 5 — dedup key is (node_id, session_id)."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_same_node_same_session_dedups(self):
        for ts in ('2026-05-18T12:00:00', '2026-05-18T12:00:01',
                   '2026-05-18T12:00:02'):
            recall_write_queue.enqueue_access('node1234', 'sess1', ts)
        self.assertEqual(recall_write_queue.queue_depth(), 1)
        stats = recall_write_queue.get_stats()
        self.assertEqual(stats['access_enqueued_total'], 3)

    def test_same_node_different_session_separates(self):
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:00')
        recall_write_queue.enqueue_access(
            'node1234', 'sess2', '2026-05-18T12:00:00')
        self.assertEqual(recall_write_queue.queue_depth(), 2)

    def test_newer_ts_wins_in_dedup(self):
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:00')
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:05')
        recall_write_queue.enqueue_access(
            'node1234', 'sess1', '2026-05-18T12:00:02')  # older
        # Snapshot and verify the latest ts wins
        acc = recall_write_queue._snapshot_and_clear()
        self.assertEqual(acc[('node1234', 'sess1')], '2026-05-18T12:00:05')


class TestDrainProcessesAccess(BrainTestBase):
    """Phase 5 — drain writes atomic +1 UPDATEs to the nodes table."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_drain_increments_access_count(self):
        # Create a real node so the UPDATE WHERE id=? finds it.
        result = self.brain.remember(
            type='test', title='drain-target',
            content='content',
            encoding_source='anchor:test')
        nid = result['id']
        before = self.brain.conn.execute(
            'SELECT access_count FROM nodes WHERE id = ?', (nid,)
        ).fetchone()[0]

        # Enqueue 3 accesses for same (node, session) — dedup to 1 UPDATE
        for ts in ('2026-05-18T12:00:00', '2026-05-18T12:00:01',
                   '2026-05-18T12:00:02'):
            recall_write_queue.enqueue_access(nid, 'sess1', ts)

        recall_write_queue.drain_once(self.brain)

        after = self.brain.conn.execute(
            'SELECT access_count, last_accessed FROM nodes WHERE id = ?',
            (nid,)).fetchone()
        # Dedup collapses 3 enqueues to 1 increment (by design)
        self.assertEqual(after[0], before + 1)
        self.assertEqual(after[1], '2026-05-18T12:00:02')

    def test_drain_never_bumps_updated_at(self):
        # Contract (2026-07-27): reads must never look like writes.
        # updated_at means "a write mutated this row" — access marks carry
        # their semantics in last_accessed/access_count only. The old
        # access-bump of updated_at broke the community idle gate
        # (always-firing), churned consolidation SKIP fingerprints, and got
        # the recall_recent tool deleted. Regression here = all of that back.
        result = self.brain.remember(
            type='test', title='updated-at-invariant',
            content='content',
            encoding_source='anchor:test')
        nid = result['id']
        before = self.brain.conn.execute(
            'SELECT updated_at FROM nodes WHERE id = ?', (nid,)).fetchone()[0]

        recall_write_queue.enqueue_access(nid, 'sess1', '2026-05-18T12:00:00')
        recall_write_queue.drain_once(self.brain)

        after = self.brain.conn.execute(
            'SELECT updated_at FROM nodes WHERE id = ?', (nid,)).fetchone()[0]
        self.assertEqual(after, before,
                         'access drain bumped updated_at — reads must never '
                         'look like writes')

    def test_drain_two_sessions_two_increments(self):
        result = self.brain.remember(
            type='test', title='drain-target-2sess',
            content='content',
            encoding_source='anchor:test')
        nid = result['id']
        before = self.brain.conn.execute(
            'SELECT access_count FROM nodes WHERE id = ?', (nid,)
        ).fetchone()[0]

        recall_write_queue.enqueue_access(nid, 'sess1', '2026-05-18T12:00:00')
        recall_write_queue.enqueue_access(nid, 'sess2', '2026-05-18T12:00:00')

        recall_write_queue.drain_once(self.brain)

        after = self.brain.conn.execute(
            'SELECT access_count FROM nodes WHERE id = ?', (nid,)).fetchone()[0]
        # Two sessions = two increments
        self.assertEqual(after, before + 2)


class TestDrainRollbackOnFailure(BrainTestBase):
    """Phase 5 — drain rolls back + logs loudly on transaction failure."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_drain_rollback_logged_and_batch_dropped(self):
        # Failure-injection via a connection proxy whose executemany
        # raises (sqlite3.Connection methods are read-only, so we swap
        # brain.conn_bg_writer for a delegating wrapper instead).
        # Raising forces drain_once into its outer except branch,
        # which calls conn_bg_writer.rollback() and logs via
        # bg_writer_batch_rollback.

        real_conn = self.brain.conn_bg_writer

        class FailingExecutemany:
            def __getattr__(self, name):
                return getattr(real_conn, name)

            def executemany(self, *args, **kwargs):
                raise RuntimeError('synthetic drain failure for rollback test')

        logged = []
        original_log = self.brain._log_error

        def log_spy(source, error, context='', ctx=None):
            logged.append((source, str(error)[:60], context[:60]))
            return original_log(source, error, context, ctx)

        self.brain._log_error = log_spy
        self.brain.conn_bg_writer = FailingExecutemany()

        try:
            # Enqueue an access item so the failing path is reached.
            recall_write_queue.enqueue_access(
                'a' * 12, 'sess1', '2026-05-18T12:00:00')

            recall_write_queue.drain_once(self.brain)

            stats = recall_write_queue.get_stats()
            self.assertGreaterEqual(stats['errors_total'], 1)
            self.assertGreaterEqual(stats['rollbacks_total'], 1)

            sources = [s for s, _, _ in logged]
            self.assertIn('bg_writer_batch_rollback', sources)

            self.assertEqual(recall_write_queue.queue_depth(), 0)
        finally:
            self.brain._log_error = original_log
            self.brain.conn_bg_writer = real_conn


class TestConcurrentEnqueue(BrainTestBase):
    """Phase 5 — concurrent enqueues from multiple threads don't lose work."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_concurrent_enqueue_preserves_unique_pairs(self):
        N_THREADS = 8
        N_PER_THREAD = 50

        def producer(thread_idx):
            for i in range(N_PER_THREAD):
                # Distinct (node, session) pairs per thread
                recall_write_queue.enqueue_access(
                    'n%07d' % (thread_idx * 1000 + i),
                    'sess%d' % thread_idx,
                    '2026-05-18T12:00:00')

        threads = [threading.Thread(target=producer, args=(t,))
                   for t in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All pairs unique → all enqueued
        expected = N_THREADS * N_PER_THREAD
        self.assertEqual(recall_write_queue.queue_depth(), expected)
        stats = recall_write_queue.get_stats()
        self.assertEqual(stats['access_enqueued_total'], expected)


class TestQueueDepthIntrospection(BrainTestBase):
    """Phase 6 — get_stats surfaces what dashboard / health checks need."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_stats_keys(self):
        stats = recall_write_queue.get_stats()
        for key in ('access_enqueued_total',
                    'drains_total', 'drains_skipped_empty',
                    'last_drain_at', 'last_drain_took_ms', 'last_drain_size',
                    'access_drained_total',
                    'errors_total', 'rollbacks_total',
                    'overlong_drains_total', 'access_queue_depth'):
            self.assertIn(key, stats, 'missing stat: %s' % key)


class TestEmptyDrainTimestampsAreFresh(BrainTestBase):
    """Empty-queue drain ticks must still update last_drain_at — otherwise
    a long idle period followed by a burst of enqueues looks like a stall
    the moment the burst lands (the stall watchdog reads a stale timestamp
    from the last batch with actual work).

    Real regression: 2026-05-19, bg_writer_worker_stalled fired with
    "no drain in 197s, embed_depth=0 rwq_depth=96, write_lock=free" — the
    worker was healthy, the queue had just been empty for 197s. Without
    this contract, every burst-after-idle triggers a false-positive."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        recall_write_queue._clear_for_test()

    def test_empty_drain_updates_last_drain_at(self):
        # Establish a baseline: an actual drain runs and stamps last_drain_at.
        result = self.brain.remember(
            type='test', title='baseline', content='c',
            encoding_source='anchor:test')
        recall_write_queue.enqueue_access(result['id'], 'sess', '2026-05-19T12:00:00')
        recall_write_queue.drain_once(self.brain)
        first_stamp = recall_write_queue.get_stats()['last_drain_at']
        self.assertIsNotNone(first_stamp)

        # Simulate "queue was empty for a while" by sleeping briefly and
        # then running an empty drain — pre-fix this was a no-op for
        # last_drain_at, leaving the watchdog with a stale timestamp.
        time.sleep(0.05)
        recall_write_queue.drain_once(self.brain)  # nothing queued

        second_stamp = recall_write_queue.get_stats()['last_drain_at']
        self.assertIsNotNone(second_stamp)
        self.assertGreater(second_stamp, first_stamp,
                           "Empty-drain tick must refresh last_drain_at")

    def test_empty_drain_counts_as_skipped_empty(self):
        # Empty drain still increments drains_skipped_empty (existing
        # behavior preserved); we're only adding the timestamp side-effect.
        before = recall_write_queue.get_stats()['drains_skipped_empty']
        recall_write_queue.drain_once(self.brain)
        after = recall_write_queue.get_stats()['drains_skipped_empty']
        self.assertEqual(after, before + 1)


if __name__ == '__main__':
    unittest.main()
