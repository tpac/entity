"""Tests for db_backup — GFS retention policy + the snapshot round-trip.

What this locks down:
1. select_retained() implements GFS correctly per tier (daily/weekly/
   monthly) and dedups the union.
2. list_backups() parses only this module's `{base}.{ts}.gz` files and
   ignores hand-made `.bak*` files (retention must never touch them).
3. backup_database() produces a gzip that decompresses to a valid SQLite
   DB with the original rows (the online-backup + gzip path works), and
   prunes to the retained set.
4. seconds_since_last_backup() reads the newest snapshot's age.
"""

import gzip
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from servers import db_backup
from servers.db_backends import sqlite as sqlite_backend

UTC = timezone.utc


class TestGFSRetention(unittest.TestCase):
    def test_daily_tier_keeps_newest_per_day(self):
        base = datetime(2026, 6, 25, tzinfo=UTC)
        # Three snapshots on the same day; daily=1 keeps only the newest.
        ts = [base.replace(hour=9), base.replace(hour=12), base.replace(hour=15)]
        retained = db_backup.select_retained(ts, keep_daily=1, keep_weekly=0,
                                             keep_monthly=0)
        self.assertEqual(retained, {base.replace(hour=15)})

    def test_weekly_tier_keeps_newest_n_weeks(self):
        # One snapshot per ISO week (7-day spacing), 6 weeks back.
        base = datetime(2026, 6, 25, 12, tzinfo=UTC)
        ts = [base - timedelta(days=7 * i) for i in range(6)]
        retained = db_backup.select_retained(ts, keep_daily=0, keep_weekly=4,
                                             keep_monthly=0)
        self.assertEqual(retained, set(ts[:4]))   # 4 most recent weeks

    def test_monthly_tier_keeps_newest_n_months(self):
        ts = [datetime(2026, m, 25, 12, tzinfo=UTC) for m in (6, 5, 4, 3, 2, 1)]
        retained = db_backup.select_retained(ts, keep_daily=0, keep_weekly=0,
                                             keep_monthly=3)
        self.assertEqual(retained, set(ts[:3]))   # Jun, May, Apr

    def test_union_dedups_and_bounds(self):
        # 120 consecutive daily snapshots. Union of 7+4+3 tiers with heavy
        # overlap (recent days are also recent weeks/months).
        base = datetime(2026, 6, 25, 3, tzinfo=UTC)
        ts = [base - timedelta(days=i) for i in range(120)]
        retained = db_backup.select_retained(ts, keep_daily=7, keep_weekly=4,
                                             keep_monthly=3)
        # Newest is always kept; the 7 most recent days are all kept.
        for i in range(7):
            self.assertIn(base - timedelta(days=i), retained)
        # Union never exceeds the sum of tier sizes.
        self.assertLessEqual(len(retained), 7 + 4 + 3)
        # A mid-range daily that is not any tier's representative is dropped.
        self.assertNotIn(base - timedelta(days=40), retained)


class TestListBackups(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ignores_handmade_bak_files(self):
        db_path = os.path.join(self.tmp, 'brain.db')
        # Two real auto-backups + clutter that must be ignored.
        for name in ('brain.db.20260625T120000Z.gz',
                     'brain.db.20260624T120000Z.gz',
                     'brain.db.bak-20260101-orphan-audit',     # hand-made
                     'brain.db.bak-20260101-orphan-audit.gz',  # no ts segment
                     'brain_logs.db.20260625T120000Z.gz'):     # different DB
            open(os.path.join(self.tmp, name), 'w').close()
        found = db_backup.list_backups(db_path, self.tmp)
        names = sorted(os.path.basename(p) for _, p in found)
        self.assertEqual(names, ['brain.db.20260624T120000Z.gz',
                                 'brain.db.20260625T120000Z.gz'])
        # Newest first ordering.
        self.assertEqual(os.path.basename(found[0][1]),
                         'brain.db.20260625T120000Z.gz')

    def test_seconds_since_last_backup(self):
        db_path = os.path.join(self.tmp, 'brain.db')
        self.assertEqual(
            db_backup.seconds_since_last_backup(db_path, self.tmp), float('inf'))
        open(os.path.join(self.tmp, 'brain.db.20260625T120000Z.gz'), 'w').close()
        now = datetime(2026, 6, 25, 13, tzinfo=UTC)   # 1h after the snapshot
        age = db_backup.seconds_since_last_backup(db_path, self.tmp, now=now)
        self.assertEqual(age, 3600.0)


class TestBackupRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        conn = sqlite3.connect(self.db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)')
        conn.executemany('INSERT INTO t (v) VALUES (?)',
                         [('row-%d' % i,) for i in range(200)])
        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_snapshot_decompresses_to_valid_db_with_rows(self):
        backup_dir = os.path.join(self.tmp, 'backups')
        now = datetime(2026, 6, 25, 12, tzinfo=UTC)
        result = db_backup.backup_database(self.db_path, backup_dir, now=now)

        gz_path = os.path.join(backup_dir, result['dest'])
        self.assertTrue(os.path.exists(gz_path))
        self.assertGreater(result['raw_bytes'], 0)
        self.assertGreater(result['gz_bytes'], 0)
        self.assertEqual(result['kept'], 1)
        self.assertEqual(result['deleted'], 0)
        # No stray intermediates left behind.
        self.assertFalse(os.path.exists(gz_path + '.tmp.db'))
        self.assertFalse(os.path.exists(gz_path + '.part'))

        # Decompress and verify it's a real SQLite DB with all the rows.
        restored = os.path.join(self.tmp, 'restored.db')
        with gzip.open(gz_path, 'rb') as f_in, open(restored, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        conn = sqlite3.connect(restored)
        try:
            count = conn.execute('SELECT COUNT(*) FROM t').fetchone()[0]
            self.assertEqual(count, 200)
            self.assertEqual(
                conn.execute('SELECT v FROM t WHERE id=1').fetchone()[0], 'row-0')
        finally:
            conn.close()

    def test_prune_drops_non_retained_on_second_run(self):
        backup_dir = os.path.join(self.tmp, 'backups')
        # Two snapshots on the same day; daily-keep collapses to the newest,
        # so the second run should prune the first.
        db_backup.backup_database(self.db_path, backup_dir,
                                  now=datetime(2026, 6, 25, 9, tzinfo=UTC))
        res2 = db_backup.backup_database(
            self.db_path, backup_dir,
            keep_daily=1, keep_weekly=0, keep_monthly=0,
            now=datetime(2026, 6, 25, 15, tzinfo=UTC))
        self.assertEqual(res2['kept'], 1)
        self.assertEqual(res2['deleted'], 1)
        remaining = sorted(os.listdir(backup_dir))
        self.assertEqual(remaining, ['brain.db.20260625T150000Z.gz'])

    def test_prune_never_deletes_newest_even_with_zero_keep(self):
        """Defensive guard: an all-zero keep config must not wipe the
        just-created snapshot — the newest is always retained."""
        backup_dir = os.path.join(self.tmp, 'backups')
        db_backup.backup_database(self.db_path, backup_dir,
                                  now=datetime(2026, 6, 24, 12, tzinfo=UTC))
        res = db_backup.backup_database(
            self.db_path, backup_dir,
            keep_daily=0, keep_weekly=0, keep_monthly=0,
            now=datetime(2026, 6, 25, 12, tzinfo=UTC))
        # Older one pruned, newest survives — never zero backups.
        self.assertEqual(res['kept'], 1)
        self.assertEqual(os.listdir(backup_dir), ['brain.db.20260625T120000Z.gz'])


class TestSnapshotTo(unittest.TestCase):
    """The raw primitive every backup and working clone routes through."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_captures_committed_wal_tail_where_file_copy_loses_it(self):
        """The mutation contrast that justifies the primitive: committed rows
        living only in the -wal (daemon SIGKILLed before any checkpoint) must
        be in the snapshot; a raw file copy of the main file loses them."""
        writer = sqlite3.connect(self.db_path)
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE t (v TEXT)')
        writer.executemany('INSERT INTO t VALUES (?)',
                           [('r%d' % i,) for i in range(50)])
        writer.commit()   # committed, but only into the -wal
        self.assertGreater(os.path.getsize(self.db_path + '-wal'), 0)

        # Writer stays open (no close-time checkpoint) AND holds a fresh
        # open transaction — the snapshot must depend on neither.
        writer.execute("INSERT INTO t VALUES ('uncommitted')")

        dest = os.path.join(self.tmp, 'clone.db')
        size = sqlite_backend.snapshot_to(self.db_path, dest)
        self.assertGreater(size, 0)
        rows = sqlite3.connect(dest).execute(
            'SELECT COUNT(*) FROM t').fetchone()[0]
        self.assertEqual(rows, 50, 'snapshot missed the committed WAL tail '
                                   'or included uncommitted rows')

        # The contrast: copy2 of the main file alone has no tables at all.
        torn = os.path.join(self.tmp, 'torn.db')
        shutil.copy2(self.db_path, torn)
        tables = sqlite3.connect(torn).execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        self.assertEqual(tables, 0, 'if a raw copy now works, the primitive '
                                    'may be over-engineered — re-probe')

    def test_missing_source_fails_loudly(self):
        with self.assertRaises(sqlite3.Error):
            sqlite_backend.snapshot_to(
                os.path.join(self.tmp, 'missing', 'nope.db'),
                os.path.join(self.tmp, 'out.db'))


class TestBackupBeforeDestructive(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        conn = sqlite3.connect(self.db_path)
        conn.execute('CREATE TABLE t (v TEXT)')
        conn.execute("INSERT INTO t VALUES ('original')")
        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_tagged_gz_and_roundtrips(self):
        path = db_backup.backup_before_destructive(self.db_path, 'pre-op')
        self.assertEqual(path, self.db_path + '.pre-op.bak.gz')
        self.assertTrue(os.path.exists(path))
        restored = db_backup.materialize_backup(path)
        row = sqlite3.connect(restored).execute(
            'SELECT v FROM t').fetchone()[0]
        self.assertEqual(row, 'original')

    def test_idempotent_per_tag_keeps_first_attempt_state(self):
        first = db_backup.backup_before_destructive(self.db_path, 'pre-op')
        # Mutate the DB — a retry must NOT refresh the backup with this.
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE t SET v = 'clobbered'")
        conn.commit()
        conn.close()
        second = db_backup.backup_before_destructive(self.db_path, 'pre-op')
        self.assertEqual(first, second)
        restored = db_backup.materialize_backup(second)
        row = sqlite3.connect(restored).execute(
            'SELECT v FROM t').fetchone()[0]
        self.assertEqual(row, 'original',
                         'retry overwrote the pre-first-attempt backup')

    def test_failure_returns_none_and_leaves_no_artifact(self):
        missing = os.path.join(self.tmp, 'ghost', 'brain.db')
        self.assertIsNone(
            db_backup.backup_before_destructive(missing, 'pre-op'))
        self.assertFalse(os.path.exists(missing + '.pre-op.bak.gz'))

    def test_uncompressed_shape_is_directly_openable(self):
        """compress=False (boot-path callers): raw .bak, no materialize step."""
        path = db_backup.backup_before_destructive(
            self.db_path, 'v29', compress=False)
        self.assertEqual(path, self.db_path + '.v29.bak')
        row = sqlite3.connect(path).execute('SELECT v FROM t').fetchone()[0]
        self.assertEqual(row, 'original')
        # materialize_backup must pass a raw backup through untouched.
        self.assertEqual(db_backup.materialize_backup(path), path)


class TestEnsureBackupFresh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.backup_dir = os.path.join(self.tmp, 'backups')
        conn = sqlite3.connect(self.db_path)
        conn.execute('CREATE TABLE t (v TEXT)')
        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stale_dir_snapshots_and_returns_true(self):
        self.assertTrue(
            db_backup.ensure_backup_fresh(self.db_path, self.backup_dir))
        self.assertEqual(
            len(db_backup.list_backups(self.db_path, self.backup_dir)), 1)

    def test_fresh_snapshot_short_circuits(self):
        db_backup.ensure_backup_fresh(self.db_path, self.backup_dir)
        before = db_backup.list_backups(self.db_path, self.backup_dir)
        self.assertTrue(
            db_backup.ensure_backup_fresh(self.db_path, self.backup_dir))
        self.assertEqual(
            db_backup.list_backups(self.db_path, self.backup_dir), before,
            'a fresh snapshot existed — the gate must not re-snapshot')

    def test_unsnapshottable_db_returns_false(self):
        missing = os.path.join(self.tmp, 'ghost', 'brain.db')
        self.assertFalse(
            db_backup.ensure_backup_fresh(missing, self.backup_dir))


class TestTaggedBakTTL(unittest.TestCase):
    """TTL retention for tagged pre-destructive backups — the generic
    reap_by_ttl primitive, the tagged-set lister, and the wire into
    backup_database's rotation."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain_logs.db')
        conn = sqlite3.connect(self.db_path)
        sqlite_backend.apply_pragmas(conn)
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _touch(self, name, age_days=0.0):
        path = os.path.join(self.tmp, name)
        open(path, 'w').close()
        if age_days:
            old = datetime.now(UTC).timestamp() - age_days * 86400
            os.utime(path, (old, old))
        return path

    def test_lister_matches_only_tagged_bak_scheme(self):
        want_old = self._touch('brain_logs.db.v1.bak', age_days=20)
        want_gz = self._touch('brain_logs.db.pre-rebuild.bak.gz')
        self._touch('brain_logs.db.20260625T120000Z.gz')   # rolling scheme
        self._touch('brain_logs.db-wal')                   # sibling
        self._touch('other.db.v1.bak')                     # different DB
        self._touch('brain_logs.db.bak')      # hand-made copy — no tag segment
        self._touch('brain_logs.db.bak.gz')   # hand-made compressed copy
        self._touch('brain_logs.db.v1.bak-shm')  # orphan sibling of a bak
        found = {p for _, p in db_backup.list_tagged_backups(self.db_path)}
        self.assertEqual(found, {want_old, want_gz})
        # the live DB itself is never listed
        self.assertNotIn(self.db_path, found)

    def test_reap_by_ttl_reports_failures_loudly(self):
        missing = os.path.join(self.tmp, 'ghost', 'brain_logs.db.v9.bak')
        res = db_backup.reap_by_ttl([missing], ttl_days=0)
        self.assertEqual(res['failed'], ['brain_logs.db.v9.bak'])
        self.assertEqual(res['reaped'], [])

    def test_reap_by_ttl_deletes_old_keeps_young(self):
        old = self._touch('brain_logs.db.v1.bak', age_days=20)
        young = self._touch('brain_logs.db.v2.bak', age_days=2)
        res = db_backup.reap_by_ttl([old, young], ttl_days=14)
        self.assertEqual(res['reaped'], ['brain_logs.db.v1.bak'])
        self.assertEqual(res['kept'], 1)
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(young))

    def test_rotation_reaps_expired_tagged_bak(self):
        old = self._touch('brain_logs.db.v1.bak',
                          age_days=db_backup.TAGGED_BAK_TTL_DAYS + 1)
        young = self._touch('brain_logs.db.v2.bak', age_days=1)
        handmade = self._touch('brain_logs.db.bak',
                               age_days=db_backup.TAGGED_BAK_TTL_DAYS + 1)
        backup_dir = os.path.join(self.tmp, 'backups')
        result = db_backup.backup_database(self.db_path, backup_dir)
        self.assertEqual(result.get('tagged_reaped'),
                         ['brain_logs.db.v1.bak'])
        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(young))
        # hand-made copy (no tag segment) never enters the reap set
        self.assertTrue(os.path.exists(handmade))

    def test_rotation_with_no_tagged_baks_is_silent(self):
        backup_dir = os.path.join(self.tmp, 'backups')
        result = db_backup.backup_database(self.db_path, backup_dir)
        self.assertNotIn('tagged_reaped', result)


if __name__ == '__main__':
    unittest.main()
