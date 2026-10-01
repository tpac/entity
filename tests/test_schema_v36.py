"""brain.db v36 and brain_logs.db v5 — drop the structure nothing reads.

v36 drops DEAD_MAIN_TABLES (and DEAD_LOG_TABLES left in a pre-split brain.db),
DEAD_COLUMNS and DEAD_INDEXES, after moving COLUMNS_TO_KV values into
node_metadata_kv. Logs v5 drops DEAD_LOG_TABLES and DEAD_LOG_COLUMNS. These
tests lock the drops against legacy shapes (including an index no schema
declared), that live rows and column values survive, idempotency, that fresh
DBs never create the dead structure, and that upgraded DBs reach the current
version through the real ensure_* ladders.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.schema import (  # noqa: E402
    BRAIN_VERSION,
    BRAIN_VERSION_KEY,
    COLUMNS_TO_KV,
    DEAD_COLUMNS,
    DEAD_INDEXES,
    DEAD_LOG_COLUMNS,
    DEAD_LOG_TABLES,
    DEAD_MAIN_TABLES,
    LOG_TABLES,
    LOGS_MIGRATIONS,
    LOGS_VERSION,
    LOGS_VERSION_KEY,
    MAIN_MIGRATIONS,
    TABLES,
    _migrate_logs_v5_drop_dead_state,
    _migrate_v36_drop_dead_state,
    ensure_logs_schema,
    ensure_schema,
    read_schema_version,
    stamp_schema_version,
)


def _names(conn, kind='table'):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def _columns(conn, table):
    return {r[1] for r in conn.execute('PRAGMA table_info(%s)' % table)}


def _fresh_brain():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    ensure_schema(conn)
    return conn


def _add_legacy_tables(conn):
    """Every dead table (reasoning_steps with a row whose FK points at
    reasoning_chains, so the drop order is exercised) and the dead log tables
    a pre-split brain.db still holds, each with an index."""
    for table in DEAD_MAIN_TABLES + DEAD_LOG_TABLES:
        if table == 'reasoning_steps':
            conn.execute(
                "CREATE TABLE reasoning_steps (id INTEGER PRIMARY KEY, "
                "chain_id TEXT NOT NULL, "
                "FOREIGN KEY (chain_id) REFERENCES reasoning_chains(id) "
                "ON DELETE CASCADE)")
            continue
        conn.execute("CREATE TABLE IF NOT EXISTS %s (id TEXT PRIMARY KEY, "
                     "v TEXT)" % table)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_legacy_%s ON %s(v)"
                     % (table, table))
    conn.execute("INSERT INTO reasoning_chains (id) VALUES ('c1')")
    conn.execute("INSERT INTO reasoning_steps (chain_id) VALUES ('c1')")
    conn.execute("INSERT INTO embedding_fidelity (id, v) VALUES ('n1', 'x')")


def _add_legacy_columns(conn):
    """The dead columns and indexes as upgraded brains still carry them: the
    declared idx_nodes_activation / idx_nodes_scope, one index no schema ever
    declared (installs migrated from older plugins can carry those), and
    idx_edge_relations_active. n1 holds column values with no KV row; n2
    holds a scope that KV already has differently."""
    for table, columns in DEAD_COLUMNS:
        for column in columns:
            conn.execute("ALTER TABLE %s ADD COLUMN %s TEXT" % (table, column))
    conn.execute("CREATE INDEX idx_nodes_activation ON nodes(activation DESC)")
    conn.execute("CREATE INDEX idx_nodes_scope ON nodes(scope)")
    conn.execute("CREATE INDEX idx_unknown_due ON nodes(due_date, type)")
    conn.execute("CREATE INDEX idx_unknown_expr ON nodes(lower(scope))")
    conn.execute("CREATE INDEX idx_unknown_partial ON nodes(type) "
                 "WHERE stability IS NOT NULL")
    conn.execute("CREATE INDEX idx_edge_relations_active "
                 "ON edge_relations(edge_id, archived)")
    conn.execute("INSERT INTO nodes (id, type, title, content, scope, "
                 "source_attribution, due_date, activation) VALUES ('n1', "
                 "'decision', 't', 'c', 'module', 'user_stated', "
                 "'2026-03-01', '1.0')")
    conn.execute("INSERT INTO nodes (id, type, title, content, scope) "
                 "VALUES ('n2', 'decision', 't2', 'c', 'file')")
    conn.execute("INSERT INTO node_metadata_kv (node_id, key, value) "
                 "VALUES ('n2', 'scope', 'system')")
    conn.execute("INSERT INTO edges (edge_id, source_id, target_id, "
                 "co_access_count) VALUES ('e1', 'n1', 'n2', '3')")
    conn.execute("INSERT INTO edge_relations (edge_id, relation, decay_rate) "
                 "VALUES ('e1', 'refines', '0.1')")


class DropDeadMainStateTest(unittest.TestCase):

    def test_drops_dead_tables_and_their_indexes(self):
        conn = _fresh_brain()
        _add_legacy_tables(conn)

        _migrate_v36_drop_dead_state(conn)

        self.assertFalse(set(DEAD_MAIN_TABLES + DEAD_LOG_TABLES)
                         & _names(conn))
        self.assertFalse({i for i in _names(conn, 'index')
                          if i.startswith('idx_legacy_')})

    def test_drops_dead_columns_and_every_index_on_them(self):
        conn = _fresh_brain()
        _add_legacy_columns(conn)

        _migrate_v36_drop_dead_state(conn)

        for table, columns in DEAD_COLUMNS:
            self.assertFalse(set(columns) & _columns(conn, table), table)
        self.assertFalse({'idx_nodes_activation', 'idx_nodes_scope',
                          'idx_unknown_due', 'idx_unknown_expr',
                          'idx_unknown_partial'} & _names(conn, 'index'))
        self.assertIn('idx_nodes_type', _names(conn, 'index'),
                      'an index on live columns only must survive')
        self.assertFalse(set(DEAD_INDEXES) & _names(conn, 'index'))

    def test_live_rows_survive_the_column_drops(self):
        conn = _fresh_brain()
        _add_legacy_columns(conn)

        _migrate_v36_drop_dead_state(conn)

        self.assertEqual(
            conn.execute("SELECT id, title FROM nodes ORDER BY id").fetchall(),
            [('n1', 't'), ('n2', 't2')])
        self.assertEqual(
            conn.execute("SELECT edge_id, source_id, target_id FROM edges")
            .fetchall(), [('e1', 'n1', 'n2')])
        self.assertEqual(
            conn.execute("SELECT edge_id, relation FROM edge_relations")
            .fetchall(), [('e1', 'refines')])

    def test_moves_kv_backed_column_values_into_kv(self):
        conn = _fresh_brain()
        _add_legacy_columns(conn)

        _migrate_v36_drop_dead_state(conn)

        kv = {(n, k): v for n, k, v in conn.execute(
            "SELECT node_id, key, value FROM node_metadata_kv WHERE key IN "
            "(%s)" % ','.join('?' * len(COLUMNS_TO_KV)), COLUMNS_TO_KV)}
        self.assertEqual(kv, {
            ('n1', 'scope'): 'module',
            ('n1', 'source_attribution'): 'user_stated',
            ('n2', 'scope'): 'system',   # the KV row already there wins
        })

    def test_live_tables_are_untouched(self):
        conn = _fresh_brain()
        _add_legacy_tables(conn)

        _migrate_v36_drop_dead_state(conn)

        self.assertEqual(set(TABLES) - _names(conn), set())

    def test_rerun_is_a_no_op(self):
        conn = _fresh_brain()
        _add_legacy_tables(conn)
        _add_legacy_columns(conn)
        _migrate_v36_drop_dead_state(conn)
        shape = (_names(conn), _names(conn, 'index'), _columns(conn, 'nodes'))

        _migrate_v36_drop_dead_state(conn)

        self.assertEqual(
            (_names(conn), _names(conn, 'index'), _columns(conn, 'nodes')),
            shape)

    def test_fresh_brain_never_creates_dead_structure(self):
        conn = _fresh_brain()
        self.assertFalse(set(DEAD_MAIN_TABLES) & _names(conn))
        for table, columns in DEAD_COLUMNS:
            self.assertFalse(set(columns) & _columns(conn, table), table)
        self.assertFalse(set(DEAD_INDEXES) & _names(conn, 'index'))
        self.assertFalse({'idx_nodes_activation', 'idx_nodes_scope'}
                         & _names(conn, 'index'))

    def test_no_dead_table_is_declared(self):
        # Declared AND dropped means every boot recreates what v36 removed.
        self.assertFalse(set(DEAD_MAIN_TABLES) & set(TABLES))

    def test_upgraded_brain_reaches_v36_through_the_ladder(self):
        conn = _fresh_brain()
        _add_legacy_tables(conn)
        _add_legacy_columns(conn)
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY, 35)
        conn.commit()

        ensure_schema(conn)

        self.assertFalse(set(DEAD_MAIN_TABLES) & _names(conn))
        self.assertFalse({'activation', 'scope', 'due_date'}
                         & _columns(conn, 'nodes'))
        self.assertNotIn('co_access_count', _columns(conn, 'edges'))
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            BRAIN_VERSION)


class NewerBrainLeftAloneTest(unittest.TestCase):
    """Older code opening a brain a later version migrated must not converge
    it back to the older declaration (a hook fallback on a stale install)."""

    def test_ensure_schema_leaves_a_newer_brain_untouched(self):
        conn = _fresh_brain()
        conn.execute("DROP TABLE s2_rejections")
        conn.execute("ALTER TABLE nodes DROP COLUMN emotion_label")
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY,
                             BRAIN_VERSION + 1)
        conn.commit()

        ensure_schema(conn)

        self.assertNotIn('s2_rejections', _names(conn))
        self.assertNotIn('emotion_label', _columns(conn, 'nodes'))
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            BRAIN_VERSION + 1)

    def test_ensure_logs_schema_leaves_a_newer_logs_db_untouched(self):
        conn = sqlite3.connect(':memory:')
        ensure_logs_schema(conn)
        conn.execute("DROP TABLE boot_renders")
        stamp_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY,
                             LOGS_VERSION + 1)
        conn.commit()

        ensure_logs_schema(conn)

        self.assertNotIn('boot_renders', _names(conn))
        self.assertEqual(
            read_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY),
            LOGS_VERSION + 1)


def _fresh_logs():
    conn = sqlite3.connect(':memory:')
    ensure_logs_schema(conn)
    return conn


def _add_legacy_logs_state(conn):
    """Every dead log table; debug_log with its dead columns and a row; and
    the self_inflight courier as installs that predate the intent removal
    still carry it, a row included."""
    for table in DEAD_LOG_TABLES:
        conn.execute("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY, "
                     "session_id TEXT)" % table)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_legacy_%s ON %s"
                     "(session_id)" % (table, table))
    for column in dict(DEAD_LOG_COLUMNS)['debug_log']:
        conn.execute("ALTER TABLE debug_log ADD COLUMN %s TEXT" % column)
    conn.execute("INSERT INTO debug_log (event_type, source, metadata, "
                 "created_at) VALUES ('pre_edit', 'hook', '{}', "
                 "'2026-09-28T00:00:00+00:00')")
    conn.execute("DROP TABLE self_inflight")
    conn.execute("""CREATE TABLE self_inflight (
        id TEXT PRIMARY KEY,
        from_session TEXT NOT NULL,
        address TEXT NOT NULL,
        intent TEXT NOT NULL DEFAULT 'signal',
        body TEXT NOT NULL,
        refs TEXT DEFAULT '',
        created_at TEXT NOT NULL
    , expires_at TEXT)""")
    conn.execute(
        "INSERT INTO self_inflight (id, from_session, address, body, "
        "created_at, expires_at) VALUES ('m1', 's1', 'self:s2', 'hi', "
        "'2026-09-28T00:00:00+00:00', '2026-09-29T00:00:00+00:00')")
    conn.commit()


class DropDeadLogsStateTest(unittest.TestCase):

    def test_drops_dead_log_tables_and_columns(self):
        conn = _fresh_logs()
        _add_legacy_logs_state(conn)

        _migrate_logs_v5_drop_dead_state(conn)
        conn.commit()

        self.assertFalse(set(DEAD_LOG_TABLES) & _names(conn))
        for table, columns in DEAD_LOG_COLUMNS:
            self.assertFalse(set(columns) & _columns(conn, table), table)
        self.assertEqual(
            conn.execute("SELECT id, body FROM self_inflight").fetchall(),
            [('m1', 'hi')], 'the courier row must survive its column drop')
        self.assertEqual(
            conn.execute("SELECT event_type FROM debug_log").fetchall(),
            [('pre_edit',)])

    def test_rerun_is_a_no_op(self):
        conn = _fresh_logs()
        _add_legacy_logs_state(conn)
        _migrate_logs_v5_drop_dead_state(conn)
        conn.commit()
        shape = (_names(conn), _columns(conn, 'debug_log'))

        _migrate_logs_v5_drop_dead_state(conn)
        conn.commit()

        self.assertEqual((_names(conn), _columns(conn, 'debug_log')), shape)

    def test_fresh_logs_db_has_no_dead_structure(self):
        conn = _fresh_logs()
        self.assertFalse(set(DEAD_LOG_TABLES) & _names(conn))
        for table, columns in DEAD_LOG_COLUMNS:
            self.assertFalse(set(columns) & _columns(conn, table), table)

    def test_no_dead_log_table_is_declared(self):
        self.assertFalse(set(DEAD_LOG_TABLES) & set(LOG_TABLES))

    def test_upgraded_logs_db_reaches_v5_through_the_ladder(self):
        conn = _fresh_logs()
        _add_legacy_logs_state(conn)
        stamp_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY, 4)
        conn.commit()

        ensure_logs_schema(conn)

        self.assertFalse(set(DEAD_LOG_TABLES) & _names(conn))
        self.assertNotIn('intent', _columns(conn, 'self_inflight'))
        self.assertNotIn('file_target', _columns(conn, 'debug_log'))
        self.assertEqual(
            read_schema_version(conn, 'logs_meta', LOGS_VERSION_KEY),
            LOGS_VERSION)


class LadderRegistrationTest(unittest.TestCase):

    def test_v36_is_on_the_brain_db_ladder(self):
        self.assertIn((36, _migrate_v36_drop_dead_state), MAIN_MIGRATIONS)
        self.assertGreaterEqual(BRAIN_VERSION, 36)

    def test_v5_is_on_the_logs_ladder(self):
        self.assertIn((5, _migrate_logs_v5_drop_dead_state), LOGS_MIGRATIONS)
        self.assertGreaterEqual(LOGS_VERSION, 5)


if __name__ == '__main__':
    unittest.main()
