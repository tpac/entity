"""The versioned migration runner — one mechanism, three streams.

Every test here exists because attempt 1 (dfc74ee, reverted) shipped a runner
that could not run anything and had 8 green tests. The load-bearing ones:

  - `test_pending_step_actually_runs` is the exact repro of the CRITICAL that
    caused the revert: `ensure_schema` stamped BRAIN_VERSION before calling the
    runner, which re-read the version, saw itself current, and returned. With
    an empty ladder that bug is invisible, so every test that matters here
    drives a NON-EMPTY ladder.
  - `test_two_consecutive_bumps` covers the actual fleet path (an install that
    skips releases), which nothing covered before.
  - `test_preversioning_logs_db_is_backed_up` covers the oldest DBs in the
    fleet — stored version 0 with real tables — which a naive `current > 0`
    backup gate skips precisely when a rewrite is most dangerous.
"""

import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import schema as schema_mod
from servers.schema import (
    BRAIN_VERSION,
    BRAIN_VERSION_KEY,
    LOGS_VERSION,
    LOGS_VERSION_KEY,
    ensure_logs_schema,
    ensure_schema,
    read_schema_version,
    run_versioned_migrations,
    stamp_schema_version,
)


def _meta_conn(table='brain_meta'):
    """Connection with just a meta table — the runner's minimum substrate."""
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE %s (key TEXT PRIMARY KEY, value TEXT, '
                 'updated_at TEXT)' % table)
    return conn


class RunnerContractTest(unittest.TestCase):
    """The runner in isolation: what runs, what stamps, what refuses."""

    def test_pending_step_actually_runs_and_stamps(self):
        conn = _meta_conn()
        stamp_schema_version(conn, 'brain_meta', 'v', 30)
        ran = []
        entry = run_versioned_migrations(
            conn, 'brain_meta', 'v', 31, [(31, lambda c: ran.append(31))])
        self.assertEqual(ran, [31])
        self.assertEqual(entry, 30)
        self.assertEqual(read_schema_version(conn, 'brain_meta', 'v'), 31)

    def test_stamp_before_run_would_kill_the_step(self):
        """The inverted order, asserted as a defect so nobody reintroduces it.

        This is the shape of the reverted bug: stamp first, then call the
        runner. The runner correctly early-returns — which is exactly why the
        stamp must belong to it and nothing may stamp ahead of it.
        """
        conn = _meta_conn()
        stamp_schema_version(conn, 'brain_meta', 'v', 30)
        ran = []
        stamp_schema_version(conn, 'brain_meta', 'v', 31)  # the inversion
        run_versioned_migrations(
            conn, 'brain_meta', 'v', 31, [(31, lambda c: ran.append(31))])
        self.assertEqual(ran, [], 'a pre-stamped version silently skips steps')

    def test_two_consecutive_bumps_run_each_step_once(self):
        conn = _meta_conn()
        stamp_schema_version(conn, 'brain_meta', 'v', 30)
        ran = []
        run_versioned_migrations(conn, 'brain_meta', 'v', 31,
                                 [(31, lambda c: ran.append(31))])
        # Next release adds a second step and bumps again. The v31 step must
        # NOT re-run; only v32 is pending.
        run_versioned_migrations(conn, 'brain_meta', 'v', 32,
                                 [(31, lambda c: ran.append(31)),
                                  (32, lambda c: ran.append(32))])
        self.assertEqual(ran, [31, 32])
        self.assertEqual(read_schema_version(conn, 'brain_meta', 'v'), 32)

    def test_install_skipping_a_release_runs_both_steps(self):
        """The real fleet path: v30 install opens code that already ships v32."""
        conn = _meta_conn()
        stamp_schema_version(conn, 'brain_meta', 'v', 30)
        ran = []
        run_versioned_migrations(conn, 'brain_meta', 'v', 32,
                                 [(31, lambda c: ran.append(31)),
                                  (32, lambda c: ran.append(32))])
        self.assertEqual(ran, [31, 32])

    def test_current_db_does_no_writes(self):
        conn = _meta_conn()
        stamp_schema_version(conn, 'brain_meta', 'v', 31)
        before = conn.execute(
            "SELECT updated_at FROM brain_meta WHERE key='v'").fetchone()[0]
        ran = []
        run_versioned_migrations(conn, 'brain_meta', 'v', 31,
                                 [(31, lambda c: ran.append(31))])
        after = conn.execute(
            "SELECT updated_at FROM brain_meta WHERE key='v'").fetchone()[0]
        self.assertEqual(ran, [])
        self.assertEqual(before, after)

    def test_failing_step_leaves_stream_unstamped_and_retries(self):
        conn = _meta_conn()
        stamp_schema_version(conn, 'brain_meta', 'v', 30)
        conn.commit()
        attempts = []

        def flaky(c):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('boom')

        with self.assertRaises(RuntimeError):
            run_versioned_migrations(conn, 'brain_meta', 'v', 31,
                                     [(31, flaky)])
        self.assertEqual(read_schema_version(conn, 'brain_meta', 'v'), 30,
                         'a failed migration must not mark the DB current')
        # Next open retries and succeeds.
        run_versioned_migrations(conn, 'brain_meta', 'v', 31, [(31, flaky)])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(read_schema_version(conn, 'brain_meta', 'v'), 31)

    def test_failing_step_rolls_back_its_partial_writes(self):
        conn = _meta_conn()
        conn.execute('CREATE TABLE t (x INTEGER)')
        stamp_schema_version(conn, 'brain_meta', 'v', 30)
        conn.commit()

        def half_writes(c):
            c.execute('INSERT INTO t (x) VALUES (1)')
            raise RuntimeError('boom')

        with self.assertRaises(RuntimeError):
            run_versioned_migrations(conn, 'brain_meta', 'v', 31,
                                     [(31, half_writes)])
        self.assertEqual(
            conn.execute('SELECT COUNT(*) FROM t').fetchone()[0], 0,
            'partial work must not survive on the shared connection')

    def test_logs_v2_step_is_atomic_across_its_ddl(self):
        """Regression: DDL autocommits in legacy isolation mode, so without
        the step's own BEGIN IMMEDIATE a second-half failure rolled back only
        the DML — the ledger rows stayed stranded in thalamus_deliveries_old
        while the retry's column probe skipped the first half: permanent
        ledger loss. The step must fail whole and retry whole."""
        conn = _meta_conn(table='logs_meta')
        conn.execute("""CREATE TABLE thalamus_deliveries (
            item_id TEXT NOT NULL, session_id TEXT NOT NULL,
            delivered_at TEXT NOT NULL, via TEXT NOT NULL,
            PRIMARY KEY (item_id, session_id))""")
        conn.execute("INSERT INTO thalamus_deliveries VALUES "
                     "('th_1', 'sessA', '2026-01-01', 'boot')")
        stamp_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY, 1)
        conn.commit()
        # No thalamus_items table → the step's second half raises.
        from servers.schema import _migrate_logs_v2_thalamus_armed_epoch
        with self.assertRaises(sqlite3.OperationalError):
            run_versioned_migrations(
                conn, 'logs_meta', LOGS_VERSION_KEY, 2,
                [(2, _migrate_logs_v2_thalamus_armed_epoch)])
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertNotIn('thalamus_deliveries_old', tables,
                         'a failed step must not strand the renamed ledger')
        self.assertEqual(  # old shape + its row fully restored
            conn.execute('SELECT COUNT(*) FROM thalamus_deliveries'
                         ).fetchone()[0], 1)
        self.assertEqual(
            read_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY), 1)
        # Retry with the missing table present: the whole step reruns.
        conn.execute("""CREATE TABLE thalamus_items (id TEXT PRIMARY KEY)""")
        conn.commit()
        run_versioned_migrations(
            conn, 'logs_meta', LOGS_VERSION_KEY, 2,
            [(2, _migrate_logs_v2_thalamus_armed_epoch)])
        row = conn.execute('SELECT item_id, armed_epoch FROM '
                           'thalamus_deliveries').fetchone()
        self.assertEqual((row[0], row[1]), ('th_1', 0))
        self.assertEqual(
            read_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY), 2)

    def test_fresh_db_is_baselined_without_running_steps(self):
        conn = _meta_conn()
        ran = []
        run_versioned_migrations(conn, 'brain_meta', 'v', 31,
                                 [(31, lambda c: ran.append(31))], fresh=True)
        self.assertEqual(ran, [], 'a fresh DB is born at the current shape')
        self.assertEqual(read_schema_version(conn, 'brain_meta', 'v'), 31)

    def test_preversioning_db_runs_every_step(self):
        """Version 0 WITH tables is not fresh — it needs the whole ladder."""
        conn = _meta_conn()
        ran = []
        run_versioned_migrations(conn, 'brain_meta', 'v', 32,
                                 [(31, lambda c: ran.append(31)),
                                  (32, lambda c: ran.append(32))],
                                 fresh=False)
        self.assertEqual(ran, [31, 32])

    def test_unsorted_or_duplicate_steps_refused(self):
        conn = _meta_conn()
        noop = lambda c: None
        with self.assertRaises(ValueError):
            run_versioned_migrations(conn, 'brain_meta', 'v', 33,
                                    [(32, noop), (31, noop)])
        with self.assertRaises(ValueError):
            run_versioned_migrations(conn, 'brain_meta', 'v', 33,
                                    [(31, noop), (31, noop)])

    def test_step_above_target_refused(self):
        """Would run early, stamp low, then run a second time after the bump."""
        conn = _meta_conn()
        with self.assertRaises(ValueError):
            run_versioned_migrations(conn, 'brain_meta', 'v', 31,
                                    [(31, lambda c: None),
                                     (32, lambda c: None)])

    def test_read_version_raises_rather_than_reading_as_fresh(self):
        """0 means "row absent". An operational error read as 0 would let the
        caller baseline-stamp a populated brain and skip every step forever."""
        conn = _meta_conn()
        conn.execute("INSERT INTO brain_meta (key, value) VALUES ('v', 'junk')")
        with self.assertRaises(ValueError):
            read_schema_version(conn, 'brain_meta', 'v')


class BackupTest(unittest.TestCase):
    """Requirement 7: the stream that migrates must be able to back up."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _db(self, name='brain_logs.db'):
        return os.path.join(self.tmp, name)

    def test_preversioning_logs_db_is_backed_up_before_the_step(self):
        path = self._db()
        conn = sqlite3.connect(path)
        # A pre-versioning logs DB: real tables, no version row.
        conn.execute('CREATE TABLE logs_meta (key TEXT PRIMARY KEY, '
                     'value TEXT, updated_at TEXT)')
        conn.execute('CREATE TABLE debug_log (x INTEGER)')
        conn.commit()

        seen = {}

        def step(c):
            seen['backup_existed_when_step_ran'] = os.path.exists(
                path + '.v0.bak')

        run_versioned_migrations(conn, 'logs_meta', LOGS_VERSION_KEY, 2,
                                 [(2, step)], db_path=path, fresh=False)
        self.assertTrue(seen['backup_existed_when_step_ran'],
                        'version-0 logs DBs are the oldest in the fleet and '
                        'must be backed up before the first rewrite')

    def test_backup_captures_committed_rows_still_in_the_wal(self):
        """The backup must be self-contained, not a pre-checkpoint husk.

        In WAL mode a COMMIT lands in the -wal file; the main DB gets it only at
        checkpoint. Copying db_path alone therefore captures the last CHECKPOINT,
        not the last COMMIT — measured here as total loss, not partial: with
        12KB of committed data in the WAL the copy has no tables at all.

        This is reachable in production because the daemon is SIGKILLed by
        `launchctl kickstart -k` (watchdog recovery), which skips the clean-close
        checkpoint and leaves committed rows in the WAL for the next boot to
        back up.
        """
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'logs.db')
        conn = sqlite3.connect(path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE logs_meta (key TEXT PRIMARY KEY, '
                     'value TEXT, updated_at TEXT)')
        conn.execute('CREATE TABLE canary (x TEXT)')
        conn.commit()
        conn.execute("INSERT INTO canary VALUES ('committed')")
        conn.commit()   # committed, but only into the -wal
        self.assertGreater(os.path.getsize(path + '-wal'), 0,
                           'precondition: committed data must be in the WAL')

        from servers.db_backup import backup_before_destructive
        backup_before_destructive(path, 'v0', compress=False)

        bak = path + '.v0.bak'
        self.assertTrue(os.path.exists(bak), 'no backup was taken')
        # Read the backup ALONE — no -wal beside it, exactly as a restore would.
        rows = sqlite3.connect(bak).execute(
            'SELECT COUNT(*) FROM canary').fetchone()[0]
        self.assertEqual(rows, 1,
                         'backup missed committed rows still in the WAL — '
                         'restoring it silently loses committed data')

    def test_backup_survives_an_open_transaction_at_the_call_site(self):
        """The backup must not depend on the caller's transaction state.

        Schema work in `ensure_logs_schema` can leave a write transaction
        open when it hands off to the runner (the shape that motivated this:
        the since-deleted interaction_active backstop DML). The backup opens
        its own read-only source connection and must capture the last
        COMMITTED state whatever the caller's connection is doing — asserted
        end-to-end through a real Brain boot migrating a pre-versioning
        logs DB.
        """
        from servers.brain import Brain

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'brain.db')
        first = Brain(db_path, skip_embedder=True)
        first.logs_conn.execute(
            "INSERT INTO interactions (name, version, template, parameters, "
            "created_at, created_by) VALUES ('x', 1, 't', '{}', 'now', 'x')")
        first.logs_conn.commit()
        first.logs_conn.execute('DELETE FROM logs_meta')   # → pre-versioning
        first.logs_conn.commit()
        logs_db_path = first.logs_db_path
        first.logs_conn.close()
        first.conn.close()

        orig_version = schema_mod.LOGS_VERSION
        orig_ladder = schema_mod.LOGS_MIGRATIONS
        try:
            schema_mod.LOGS_VERSION = orig_version + 1
            schema_mod.LOGS_MIGRATIONS = [(orig_version + 1, lambda c: None)]
            second = Brain(db_path, skip_embedder=True)
            second.logs_conn.close()
            second.conn.close()
        finally:
            schema_mod.LOGS_VERSION = orig_version
            schema_mod.LOGS_MIGRATIONS = orig_ladder

        bak = logs_db_path + '.v0.bak'
        self.assertTrue(os.path.exists(bak),
                        'pre-versioning logs DB migrated without a backup')
        row = sqlite3.connect(bak).execute(
            "SELECT COUNT(*) FROM interactions WHERE name = 'x'").fetchone()[0]
        self.assertEqual(row, 1,
                         'backup is missing committed pre-migration data')

    def test_fresh_db_is_not_backed_up(self):
        path = self._db()
        conn = sqlite3.connect(path)
        conn.execute('CREATE TABLE logs_meta (key TEXT PRIMARY KEY, '
                     'value TEXT, updated_at TEXT)')
        conn.commit()
        run_versioned_migrations(conn, 'logs_meta', LOGS_VERSION_KEY, 2,
                                 [(2, lambda c: None)], db_path=path,
                                 fresh=True)
        self.assertFalse(os.path.exists(path + '.v0.bak'))

    def test_no_backup_when_nothing_is_pending(self):
        path = self._db()
        conn = sqlite3.connect(path)
        conn.execute('CREATE TABLE logs_meta (key TEXT PRIMARY KEY, '
                     'value TEXT, updated_at TEXT)')
        stamp_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY, 1)
        conn.commit()
        run_versioned_migrations(conn, 'logs_meta', LOGS_VERSION_KEY, 2, [],
                                 db_path=path, fresh=False)
        self.assertFalse(os.path.exists(path + '.v1.bak'))

    def test_logs_split_backs_up_before_dropping_legacy_tables(self):
        """migrate_logs_to_separate_db runs at EVERY Brain open, outside the
        version-gated backup — so it must decide its own: a backup exactly
        when a legacy log table is about to be DROPped, and no file touched
        on the common boot where none exist."""
        main_path = self._db('brain.db')
        main = sqlite3.connect(main_path)
        logs = sqlite3.connect(self._db('brain_logs.db'))

        # Common boot: no legacy tables in main → no backup, early exit.
        migrated = schema_mod.migrate_logs_to_separate_db(
            main, logs, main_db_path=main_path)
        self.assertEqual(migrated, [])
        self.assertFalse(os.path.exists(main_path + '.pre-logs-split.bak'))

        # Legacy boot: a log table still lives in brain.db.
        main.execute(schema_mod.LOG_TABLES['debug_log']['create'])
        main.commit()
        logs.execute(schema_mod.LOG_TABLES['logs_meta']['create'])
        logs.execute(schema_mod.LOG_TABLES['debug_log']['create'])
        logs.commit()
        migrated = schema_mod.migrate_logs_to_separate_db(
            main, logs, main_db_path=main_path)
        self.assertIn('debug_log', migrated)
        self.assertTrue(os.path.exists(main_path + '.pre-logs-split.bak'),
                        'legacy log tables were DROPped with no backup')
        self.assertIsNone(main.execute(
            "SELECT name FROM sqlite_master WHERE name='debug_log'"
        ).fetchone())

    def test_nodes_rebuild_backs_up_on_its_own_trigger(self):
        """The nodes rebuild (DROP TABLE nodes) is triggered by a DDL probe,
        not a version change — a current-version brain with legacy CHECK DDL
        gets rebuilt. The backup must key on the rebuild trigger, not on the
        version delta (which is zero here)."""
        path = self._db('brain.db')
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL CHECK(type IN ('person','concept')),
            title TEXT NOT NULL
        )""")
        conn.execute("INSERT INTO nodes VALUES ('n1', 'person', 'T')")
        conn.execute('CREATE TABLE brain_meta (key TEXT PRIMARY KEY, '
                     'value TEXT, updated_at TEXT)')
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY,
                             schema_mod.BRAIN_VERSION)   # already current
        conn.commit()

        ensure_schema(conn, db_path=path)

        bak = path + '.nodes-rebuild.bak'
        self.assertTrue(os.path.exists(bak),
                        'nodes table rebuilt (DROPped) with no backup')
        # The rebuild itself must still have worked.
        row = conn.execute("SELECT title FROM nodes WHERE id='n1'").fetchone()
        self.assertEqual(row[0], 'T')


class EnsureSchemaIntegrationTest(unittest.TestCase):
    """The runner wired into the two real entry points."""

    def test_brain_db_migration_runs_when_ladder_is_non_empty(self):
        """The reverted CRITICAL, at the real call site.

        Attempt 1 passed its equivalent of this suite because MAIN_MIGRATIONS
        was empty — the one condition that hides a stamp-before-run inversion.
        """
        conn = sqlite3.connect(':memory:')
        ensure_schema(conn)  # fresh brain, stamped at BRAIN_VERSION
        # Pretend this install is one release behind.
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY,
                             BRAIN_VERSION - 1)
        conn.commit()

        ran = []
        orig_version = schema_mod.BRAIN_VERSION
        orig_ladder = schema_mod.MAIN_MIGRATIONS
        try:
            schema_mod.BRAIN_VERSION = orig_version + 1
            schema_mod.MAIN_MIGRATIONS = [
                (orig_version + 1, lambda c: ran.append('v_next'))]
            ensure_schema(conn)
        finally:
            schema_mod.BRAIN_VERSION = orig_version
            schema_mod.MAIN_MIGRATIONS = orig_ladder

        self.assertEqual(ran, ['v_next'],
                         'ensure_schema must let the runner own the stamp')
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            orig_version + 1)

    def test_fresh_brain_db_runs_no_ladder_steps(self):
        """Guards `fresh=` in the TRUE direction: hardcoding fresh=False here
        would run v31 migrations against a brand-new DB that never had a v30."""
        conn = sqlite3.connect(':memory:')
        ran = []
        orig_version = schema_mod.BRAIN_VERSION
        orig_ladder = schema_mod.MAIN_MIGRATIONS
        try:
            schema_mod.BRAIN_VERSION = orig_version + 1
            schema_mod.MAIN_MIGRATIONS = [
                (orig_version + 1, lambda c: ran.append('v_next'))]
            ensure_schema(conn)
        finally:
            schema_mod.BRAIN_VERSION = orig_version
            schema_mod.MAIN_MIGRATIONS = orig_ladder
        self.assertEqual(ran, [], 'a brand-new brain.db needs no ladder step')
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            orig_version + 1)

    def test_preversioning_brain_db_runs_the_ladder(self):
        """Guards `fresh=` in the FALSE direction: hardcoding fresh=True would
        baseline a populated brain and skip every pending step."""
        conn = sqlite3.connect(':memory:')
        ensure_schema(conn)
        conn.execute("INSERT INTO nodes (id, type, title) "
                     "VALUES ('aaaaaaaa', 'fact', 'real data')")
        conn.execute('DELETE FROM brain_meta WHERE key = ?',
                     (BRAIN_VERSION_KEY,))
        conn.commit()

        ran = []
        orig_ladder = schema_mod.MAIN_MIGRATIONS
        try:
            schema_mod.MAIN_MIGRATIONS = [
                (BRAIN_VERSION, lambda c: ran.append('ladder'))]
            ensure_schema(conn)
        finally:
            schema_mod.MAIN_MIGRATIONS = orig_ladder
        self.assertEqual(ran, ['ladder'],
                         'version 0 with real tables is not a fresh DB')

    def test_backfill_runs_before_the_stamp(self):
        """Ordering guard: the stamp must land after `_backfill_data`.

        With the stamp first (the old order), a crash mid-backfill left the DB
        marked current with the backfill half-applied and no retry — forward-only
        logic guarantees it never runs again.
        """
        conn = sqlite3.connect(':memory:')
        ensure_schema(conn)
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY,
                             BRAIN_VERSION - 1)
        conn.commit()

        seen = {}
        orig = schema_mod._backfill_data

        def spy(c, from_version):
            seen['version_during_backfill'] = read_schema_version(
                c, 'brain_meta', BRAIN_VERSION_KEY)
            return orig(c, from_version)

        try:
            schema_mod._backfill_data = spy
            ensure_schema(conn)
        finally:
            schema_mod._backfill_data = orig

        self.assertEqual(seen['version_during_backfill'], BRAIN_VERSION - 1,
                         'the stamp must not be visible while backfill runs, '
                         'or a crash there can never be retried')
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            BRAIN_VERSION)

    def test_fresh_brain_db_is_stamped_current(self):
        conn = sqlite3.connect(':memory:')
        ensure_schema(conn)
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            BRAIN_VERSION)

    def test_fresh_logs_db_is_stamped_and_runs_no_steps(self):
        conn = sqlite3.connect(':memory:')
        ran = []
        orig = schema_mod.LOGS_MIGRATIONS
        try:
            schema_mod.LOGS_MIGRATIONS = [(1, lambda c: ran.append(1))]
            ensure_logs_schema(conn)
        finally:
            schema_mod.LOGS_MIGRATIONS = orig
        self.assertEqual(ran, [], 'a fresh logs DB needs no structural steps')
        self.assertEqual(read_schema_version(conn, 'logs_meta',
                                             LOGS_VERSION_KEY), LOGS_VERSION)

    def test_logs_schema_is_idempotent(self):
        conn = sqlite3.connect(':memory:')
        ensure_logs_schema(conn)
        ensure_logs_schema(conn)
        self.assertEqual(read_schema_version(conn, 'logs_meta',
                                             LOGS_VERSION_KEY), LOGS_VERSION)

    def test_real_brain_backs_up_a_preversioning_logs_db(self):
        """End-to-end through the real constructor, on disk.

        The runner-level backup and the db_path wiring are each covered above,
        but nothing joined them through `Brain()` — and requirement 7 exists
        because attempt 1's `ensure_logs_schema(conn)` took no db_path at all,
        so the first real logs migration would have run with no backup.
        """
        from servers.brain import Brain

        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, 'brain.db')
        logs_path = os.path.join(tmp, 'brain_logs.db')

        first = Brain(db_path, skip_embedder=True)      # builds both DBs
        first.logs_conn.execute('DELETE FROM logs_meta')  # → pre-versioning
        first.logs_conn.commit()
        first.logs_conn.close()
        first.conn.close()

        ran = []
        orig_version = schema_mod.LOGS_VERSION
        orig_ladder = schema_mod.LOGS_MIGRATIONS
        try:
            schema_mod.LOGS_VERSION = orig_version + 1
            schema_mod.LOGS_MIGRATIONS = [
                (orig_version + 1, lambda c: ran.append('logs_step'))]
            second = Brain(db_path, skip_embedder=True)
            second.logs_conn.close()
            second.conn.close()
        finally:
            schema_mod.LOGS_VERSION = orig_version
            schema_mod.LOGS_MIGRATIONS = orig_ladder

        self.assertEqual(ran, ['logs_step'])
        self.assertTrue(os.path.exists(logs_path + '.v0.bak'),
                        'Brain() must back up a pre-versioning logs DB before '
                        'a structural step rewrites it')

    def test_preversioning_logs_db_runs_its_ladder(self):
        """A logs DB from before versioning: tables present, no counter."""
        conn = sqlite3.connect(':memory:')
        ensure_logs_schema(conn)                       # build current shape
        conn.execute("DELETE FROM logs_meta")          # erase the counter
        conn.commit()

        ran = []
        orig_version = schema_mod.LOGS_VERSION
        orig_ladder = schema_mod.LOGS_MIGRATIONS
        try:
            schema_mod.LOGS_VERSION = orig_version + 1
            schema_mod.LOGS_MIGRATIONS = [
                (orig_version + 1, lambda c: ran.append('v_next'))]
            ensure_logs_schema(conn)
        finally:
            schema_mod.LOGS_VERSION = orig_version
            schema_mod.LOGS_MIGRATIONS = orig_ladder

        self.assertEqual(ran, ['v_next'],
                         'an unstamped logs DB with tables is not fresh')


class LogsV4ToolResultVectorPurgeTest(unittest.TestCase):
    """The v4 step's semantics: vectors go, rows stay.

    The step is a one-shot purge of ~137k rows on a live brain, so what it
    LEAVES matters as much as what it removes — the encoder timeline, the
    `contains=` lens and node source_refs all read trace_events, not vectors.
    """

    def _logs_db(self):
        conn = sqlite3.connect(':memory:')
        ensure_logs_schema(conn)
        rows = [('tr000001', 'tool_result'), ('tr000002', 'tool_result'),
                ('um000001', 'user_message'), ('am000001', 'assistant_message')]
        for tid, rt in rows:
            conn.execute(
                'INSERT INTO trace_events (id, chain_id, scale, event_type, '
                'ref_type, summary, created_at) VALUES (?,?,?,?,?,?,?)',
                (tid, 'c1', 's0', 'delta', rt, rt + ' row',
                 '2026-09-22T00:00:00+00:00'))
            conn.execute(
                'INSERT INTO trace_embeddings (trace_id, vector, text, model, '
                'created_at) VALUES (?,?,?,?,?)',
                (tid, b'\x00' * 8, rt, 'm', '2026-09-22T00:00:00+00:00'))
        conn.commit()
        return conn

    def _vector_ref_types(self, conn):
        return sorted(r[0] for r in conn.execute(
            'SELECT te.ref_type FROM trace_embeddings tem '
            'JOIN trace_events te ON te.id = tem.trace_id'))

    def test_only_tool_result_vectors_are_deleted(self):
        conn = self._logs_db()
        schema_mod._migrate_logs_v4_drop_tool_result_vectors(conn)
        conn.commit()
        self.assertEqual(self._vector_ref_types(conn),
                         ['assistant_message', 'user_message'],
                         'dialogue vectors must survive the purge')

    def test_trace_event_rows_are_untouched(self):
        conn = self._logs_db()
        schema_mod._migrate_logs_v4_drop_tool_result_vectors(conn)
        conn.commit()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) FROM trace_events "
                         "WHERE ref_type = 'tool_result'").fetchone()[0], 2,
            'the purge drops vectors only — the rows carry the timeline, the '
            'contains= lens and node source_refs')

    def test_idempotent_on_retry(self):
        conn = self._logs_db()
        schema_mod._migrate_logs_v4_drop_tool_result_vectors(conn)
        conn.commit()
        schema_mod._migrate_logs_v4_drop_tool_result_vectors(conn)
        conn.commit()
        self.assertEqual(self._vector_ref_types(conn),
                         ['assistant_message', 'user_message'])

    def test_ladder_purges_a_v3_db_and_stamps_v4(self):
        """End-to-end through the runner, not the step directly.

        The direct-call tests above prove the DELETE; this proves the WIRING —
        that a real v3 install reaching ensure_logs_schema actually runs it and
        lands stamped. Registering the step at the wrong version, or the
        fresh-DB guard swallowing it, is invisible to a direct call.
        """
        conn = self._logs_db()
        stamp_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY, 3)
        conn.commit()

        ensure_logs_schema(conn)

        self.assertEqual(self._vector_ref_types(conn),
                         ['assistant_message', 'user_message'],
                         'the v4 step did not run for a stamped-v3 logs DB')
        self.assertEqual(
            read_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY),
            LOGS_VERSION)

    def test_step_is_registered_at_the_current_logs_version(self):
        self.assertIn((4, schema_mod._migrate_logs_v4_drop_tool_result_vectors),
                      schema_mod.LOGS_MIGRATIONS)
        self.assertGreaterEqual(
            LOGS_VERSION, 4,
            'a registered step above LOGS_VERSION is refused by the runner')


if __name__ == '__main__':
    unittest.main()
