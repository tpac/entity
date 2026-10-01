"""Tests for db_maintenance scheduler + db_backends.sqlite pragma application.

What this locks down:
1. apply_pragmas sets the documented per-connection knobs (not the
   SQLite defaults) on every fresh sqlite3.connect.
2. checkpoint() actually shrinks the WAL file after writes.
3. The scheduler fires registered ops at the configured cadence and
   catches per-op failures without killing the loop.
"""

import os
import sqlite3
import tempfile
import threading
import time
import unittest

from servers.db_backends import sqlite as sqlite_backend
from servers.db_maintenance import DBMaintenance


class TestApplyPragmas(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'test.db')

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        sqlite_backend.apply_pragmas(conn)
        return conn

    def test_pragmas_match_documented_values(self):
        conn = self._open()
        try:
            self.assertEqual(conn.execute('PRAGMA busy_timeout').fetchone()[0], 30000)
            # cache_size in negative form = -kibibytes; -65536 = 64 MB
            self.assertEqual(conn.execute('PRAGMA cache_size').fetchone()[0], -65536)
            self.assertEqual(conn.execute('PRAGMA mmap_size').fetchone()[0], 268435456)
            # temp_store: 0=DEFAULT, 1=FILE, 2=MEMORY
            self.assertEqual(conn.execute('PRAGMA temp_store').fetchone()[0], 2)
            # synchronous: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
            self.assertEqual(conn.execute('PRAGMA synchronous').fetchone()[0], 1)
            self.assertEqual(conn.execute('PRAGMA foreign_keys').fetchone()[0], 1)
            self.assertEqual(
                conn.execute('PRAGMA journal_mode').fetchone()[0].lower(), 'wal')
        finally:
            conn.close()

    def test_pragmas_idempotent(self):
        conn = self._open()
        try:
            sqlite_backend.apply_pragmas(conn)
            sqlite_backend.apply_pragmas(conn)
            # Still correct after double-apply.
            self.assertEqual(conn.execute('PRAGMA synchronous').fetchone()[0], 1)
        finally:
            conn.close()


class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_truncates_wal(self):
        """A held reader keeps the WAL from auto-truncating, so we can
        observe checkpoint(TRUNCATE) actually shrinking it. Without the
        held reader, SQLite's auto-checkpoint trigger fires on commit
        once WAL exceeds 1000 pages and the test becomes flaky."""
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'test.db')

        writer = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(writer)
        writer.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, payload BLOB)')
        # Reader on a separate connection — its existence prevents the
        # auto-checkpoint from truncating WAL during writes.
        reader = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(reader)
        reader.execute('SELECT COUNT(*) FROM t').fetchone()

        try:
            blob = b'x' * 4096
            for _ in range(500):
                writer.execute('INSERT INTO t (payload) VALUES (?)', (blob,))
            writer.commit()

            wal_path = db_path + '-wal'
            wal_before = os.path.getsize(wal_path)
            self.assertGreater(wal_before, 0,
                               "Held reader should leave a non-empty WAL")

            # Release the reader so checkpoint can truncate.
            reader.close()
            reader = None

            result = sqlite_backend.checkpoint(db_path)
            self.assertEqual(result['busy'], 0)
            self.assertEqual(result['wal_size_after'], 0)
            self.assertGreater(result['wal_size_before'], 0)
        finally:
            writer.close()
            if reader is not None:
                reader.close()


def _indexed_db(rows):
    """A db whose one indexed table holds `rows` rows, analyzed at that size."""
    db_path = os.path.join(tempfile.mkdtemp(), 'test.db')
    conn = sqlite3.connect(db_path)
    sqlite_backend.apply_pragmas(conn)
    conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER)')
    conn.execute('CREATE INDEX idx_t_x ON t(x)')
    conn.executemany('INSERT INTO t (x) VALUES (?)', [(i % 7,) for i in range(rows)])
    conn.execute('ANALYZE')
    conn.commit()
    conn.close()
    return db_path


def _grow(db_path, rows):
    conn = sqlite3.connect(db_path)
    conn.executemany('INSERT INTO t (x) VALUES (?)', [(i % 7,) for i in range(rows)])
    conn.commit()
    conn.close()


def _stat_rows(db_path):
    """Row count sqlite_stat1 records for idx_t_x (what the planner believes)."""
    conn = sqlite3.connect(db_path)
    try:
        stat = conn.execute(
            "SELECT stat FROM sqlite_stat1 WHERE idx = 'idx_t_x'").fetchone()[0]
    finally:
        conn.close()
    return int(stat.split()[0])


class TestOptimize(unittest.TestCase):
    def test_optimize_refreshes_stats_of_a_grown_table(self):
        # The maintenance connection is fresh, so it has no query history —
        # the case where bare `PRAGMA optimize` re-analyzes nothing. The
        # 10x-grown table must still get fresh statistics.
        db_path = _indexed_db(20)
        _grow(db_path, 5000)
        self.assertEqual(_stat_rows(db_path), 20)
        self.assertTrue(sqlite_backend.optimize(db_path)['stats_refreshed'])
        self.assertEqual(_stat_rows(db_path), 5020)

    def test_optimize_leaves_a_mildly_grown_table_to_analyze(self):
        db_path = _indexed_db(1000)
        _grow(db_path, 1000)   # 2x: below optimize's 10x trigger
        sqlite_backend.optimize(db_path)
        self.assertEqual(_stat_rows(db_path), 1000)


    def test_optimize_signals_open_connections_only_when_stats_change(self):
        db_path = _indexed_db(20)
        reader = sqlite3.connect(db_path)
        try:
            version = lambda: reader.execute('PRAGMA schema_version').fetchone()[0]
            v0 = version()
            self.assertFalse(sqlite_backend.optimize(db_path)['stats_refreshed'])
            self.assertEqual(version(), v0, 'a no-op optimize must not make '
                             'every connection re-read its schema')
            _grow(db_path, 5000)
            self.assertTrue(sqlite_backend.optimize(db_path)['stats_refreshed'])
            self.assertGreater(version(), v0)
        finally:
            reader.close()


class TestAnalyze(unittest.TestCase):
    def test_analyze_writes_exact_stats_below_the_10x_trigger(self):
        db_path = _indexed_db(1000)
        _grow(db_path, 1000)
        self.assertEqual(sqlite_backend.analyze(db_path), {'ok': True})
        self.assertEqual(_stat_rows(db_path), 2000)

    def test_an_open_connection_plans_on_the_new_stats(self):
        # A long-lived connection keeps the statistics it loaded when it
        # parsed the schema; analyze() must make it re-read them. The IN-list
        # plan flips from the `archived` index to the primary key once the
        # stats say the table is big.
        db_path = os.path.join(tempfile.mkdtemp(), 'test.db')
        conn = sqlite3.connect(db_path)
        conn.execute('CREATE TABLE n (id TEXT PRIMARY KEY, archived INTEGER)')
        conn.execute('CREATE INDEX idx_n_archived ON n(archived)')
        conn.executemany('INSERT INTO n VALUES (?, ?)',
                         [('a%05d' % i, i % 2) for i in range(60)])
        conn.execute('ANALYZE')
        conn.executemany('INSERT INTO n VALUES (?, ?)',
                         [('b%05d' % i, i % 2) for i in range(20000)])
        conn.commit()
        conn.close()
        ids = ['b%05d' % i for i in range(900)]
        reader = sqlite3.connect(db_path)
        try:
            def plan(tag):
                # An executed statement is what notices a schema change; the
                # tag keeps each EXPLAIN out of the statement cache.
                reader.execute('SELECT 1 FROM n LIMIT 1').fetchall()
                return reader.execute(
                    'EXPLAIN QUERY PLAN SELECT id FROM n WHERE archived = 0 '
                    'AND id IN (%s) /* %s */' % (','.join('?' * len(ids)), tag),
                    ids).fetchall()[0][3]
            self.assertIn('idx_n_archived', plan('before'))
            sqlite_backend.analyze(db_path)
            self.assertIn('sqlite_autoindex_n_1', plan('after'))
        finally:
            reader.close()


class TestStats(unittest.TestCase):
    def test_stats_returns_documented_fields(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'test.db')
        conn = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()
        s = sqlite_backend.stats(db_path)
        for key in ('db_size_bytes', 'wal_size_bytes', 'page_count',
                    'page_size', 'freelist_pages', 'pages_in_use_pct'):
            self.assertIn(key, s)


class TestSchedulerFiresOps(unittest.TestCase):
    def test_due_op_fires_within_one_tick(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'test.db')
        conn = sqlite3.connect(db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()

        fired = {'checkpoint': 0, 'optimize': 0}

        class StubBackend:
            def apply_pragmas(self, conn):
                pass
            def checkpoint(self, p):
                fired['checkpoint'] += 1
                return {}
            def quick_check(self, p):
                return {'ok': True}
            def optimize(self, p):
                fired['optimize'] += 1
                return {}
            def analyze(self, p):
                return {'ok': True}
            def stats(self, p):
                return {}

        m = DBMaintenance(
            checkpoint_interval_s=0.05,   # ~immediately due
            optimize_interval_s=0.05,
            tick_interval_s=0.05,
        )
        m._backend = StubBackend()
        m.register('test', db_path)
        m.start()
        try:
            # Two ticks gives the scheduler time to fire both ops.
            time.sleep(0.25)
        finally:
            m.stop()

        self.assertGreaterEqual(fired['checkpoint'], 1)
        self.assertGreaterEqual(fired['optimize'], 1)

    def test_op_exception_does_not_kill_loop(self):
        """A failing op must be caught + logged; loop continues."""
        fired = {'count': 0}

        class FlakeyBackend:
            def apply_pragmas(self, conn):
                pass
            def checkpoint(self, p):
                fired['count'] += 1
                if fired['count'] == 1:
                    raise RuntimeError('first call fails')
                return {}
            def quick_check(self, p):
                return {'ok': True}
            def optimize(self, p):
                return {}
            def analyze(self, p):
                return {'ok': True}
            def stats(self, p):
                return {}

        logged = []
        m = DBMaintenance(
            log_fn=lambda msg: logged.append(msg),
            log_error_fn=lambda origin, exc, ctx: logged.append('ERR %s' % origin),
            checkpoint_interval_s=0.05,
            optimize_interval_s=999.0,
            tick_interval_s=0.05,
        )
        m._backend = FlakeyBackend()
        m.register('test', '/dev/null')
        m.start()
        try:
            time.sleep(0.4)
        finally:
            m.stop()

        # First call raised; subsequent calls must have happened.
        self.assertGreaterEqual(fired['count'], 2)
        # The error was logged via log_error_fn (not silently swallowed).
        self.assertTrue(any('ERR db_maintenance_checkpoint' in s for s in logged),
                        "Expected ERR log, got: %s" % logged)

    def test_failed_op_reschedules_by_interval_not_hot_retry(self):
        """Regression: a failing op must advance its schedule by one full
        interval, NOT retry every tick. The original bug stamped the
        last-run timestamp only on success, so a contended `database is
        locked` stayed perpetually 'due' and the 30s tick loop hammered an
        80s-blocking optimize back-to-back for ~20 min — starving recall
        and tripping DAEMON_DOWN. Fix: stamp on attempt, so a long interval
        means the failed op waits that interval before its next attempt."""
        fired = {'count': 0}

        class AlwaysFailsBackend:
            def apply_pragmas(self, conn):
                pass
            def checkpoint(self, p):
                fired['count'] += 1
                raise RuntimeError('database is locked')
            def quick_check(self, p):
                return {'ok': True}
            def optimize(self, p):
                return {}
            def analyze(self, p):
                return {'ok': True}
            def stats(self, p):
                return {}

        m = DBMaintenance(
            log_fn=lambda msg: None,
            log_error_fn=lambda origin, exc, ctx: None,
            checkpoint_interval_s=999.0,   # long: due once, then not again
            optimize_interval_s=999.0,
            tick_interval_s=0.05,          # many ticks within the window
        )
        m._backend = AlwaysFailsBackend()
        m.register('test', '/dev/null')
        m.start()
        try:
            # ~16 ticks elapse here. Pre-fix this would fire ~16 times
            # (every tick, since failure left it 'due'). Post-fix it fires
            # exactly once — the attempt stamp pushes the next run 999s out.
            time.sleep(0.8)
        finally:
            m.stop()

        self.assertEqual(fired['count'], 1,
                         "Failed op should fire once then wait the full "
                         "interval, not hot-retry every tick. Got %d fires."
                         % fired['count'])


if __name__ == '__main__':
    unittest.main()


class TestQuickCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'qc.db')
        conn = sqlite3.connect(self.db_path)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, body TEXT)')
        conn.executemany('INSERT INTO t (body) VALUES (?)',
                         [('x' * 500,) for _ in range(200)])
        conn.commit()
        conn.close()

    def test_healthy_db_returns_ok(self):
        self.assertEqual(sqlite_backend.quick_check(self.db_path), {'ok': True})

    def test_corruption_raises_into_the_error_path(self):
        # Flip bytes in the middle of a data page. quick_check must RAISE
        # (RuntimeError for structural findings; sqlite may also throw its
        # own DatabaseError first) — never return a quiet not-ok dict.
        size = os.path.getsize(self.db_path)
        with open(self.db_path, 'r+b') as f:
            f.seek(size // 2)
            f.write(b'\xde\xad\xbe\xef' * 64)
        with self.assertRaises((RuntimeError, sqlite3.DatabaseError)):
            sqlite_backend.quick_check(self.db_path)


class TestTableSizes(unittest.TestCase):
    def test_returns_per_table_mb(self):
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'ts.db')
        conn = sqlite3.connect(db_path)
        conn.execute('CREATE TABLE big (body TEXT)')
        conn.executemany('INSERT INTO big VALUES (?)',
                         [('y' * 1000,) for _ in range(500)])
        conn.commit()
        conn.close()
        sizes = sqlite_backend.table_sizes(db_path)
        if not sizes:
            self.skipTest('dbstat vtab not compiled into this sqlite build')
        self.assertIn('big', sizes)
        self.assertGreater(sizes['big'], 0)


class TestWalPinnedAlert(unittest.TestCase):
    """_track_wal_pinning: alert exactly once at the streak threshold; any
    successful truncation resets the streak."""

    def _maintenance_with_capture(self):
        errors = []
        m = DBMaintenance(
            log_fn=lambda msg: None,
            log_error_fn=lambda origin, exc, ctx: errors.append(origin))
        return m, errors

    def test_alert_fires_once_at_threshold(self):
        m, errors = self._maintenance_with_capture()
        entry = {'name': 'test', 'wal_pinned_streak': 0}
        pinned = {'wal_size_before': 4096, 'wal_size_after': 4096}
        for _ in range(5):
            m._track_wal_pinning(entry, pinned)
        self.assertEqual(errors.count('db_maintenance_wal_pinned'), 1)

    def test_success_resets_streak(self):
        m, errors = self._maintenance_with_capture()
        entry = {'name': 'test', 'wal_pinned_streak': 0}
        pinned = {'wal_size_before': 4096, 'wal_size_after': 4096}
        truncated = {'wal_size_before': 4096, 'wal_size_after': 0}
        m._track_wal_pinning(entry, pinned)
        m._track_wal_pinning(entry, pinned)
        m._track_wal_pinning(entry, truncated)   # reset
        m._track_wal_pinning(entry, pinned)
        m._track_wal_pinning(entry, pinned)
        self.assertEqual(errors, [], 'streak did not reset on success')

    def test_empty_wal_is_not_pinned(self):
        m, errors = self._maintenance_with_capture()
        entry = {'name': 'test', 'wal_pinned_streak': 0}
        for _ in range(5):
            m._track_wal_pinning(entry, {'wal_size_before': 0, 'wal_size_after': 0})
        self.assertEqual(errors, [])


class TestQuickCheckScheduled(unittest.TestCase):
    def test_quick_check_fires_on_schedule(self):
        fired = {'quick_check': 0}

        class StubBackend:
            def apply_pragmas(self, conn):
                pass
            def checkpoint(self, p):
                return {}
            def quick_check(self, p):
                fired['quick_check'] += 1
                return {'ok': True}
            def optimize(self, p):
                return {}
            def analyze(self, p):
                return {'ok': True}
            def stats(self, p):
                return {}

        m = DBMaintenance(
            log_fn=lambda msg: None,
            checkpoint_interval_s=999.0,
            optimize_interval_s=999.0,
            quick_check_interval_s=0.05,
            tick_interval_s=0.05,
        )
        m._backend = StubBackend()
        m.register('test', '/dev/null')
        m.start()
        try:
            time.sleep(0.25)
        finally:
            m.stop()
        self.assertGreaterEqual(fired['quick_check'], 1)


class TestAnalyzeScheduled(unittest.TestCase):
    def test_analyze_fires_after_boot_then_waits_its_interval(self):
        # The daemon restarts more often than weekly, so the first run must
        # not wait a full interval after boot or it would never come due.
        fired = {'analyze': 0}

        class StubBackend:
            def apply_pragmas(self, conn):
                pass
            def checkpoint(self, p):
                return {}
            def quick_check(self, p):
                return {'ok': True}
            def optimize(self, p):
                return {}
            def analyze(self, p):
                fired['analyze'] += 1
                return {'ok': True}
            def stats(self, p):
                return {}

        m = DBMaintenance(
            log_fn=lambda msg: None,
            checkpoint_interval_s=999.0,
            optimize_interval_s=999.0,
            quick_check_interval_s=999.0,
            analyze_interval_s=999.0,
            tick_interval_s=0.05,
        )
        m._backend = StubBackend()
        m.register('test', '/dev/null')
        m.start()
        try:
            time.sleep(0.4)
        finally:
            m.stop()
        self.assertEqual(fired['analyze'], 1)
