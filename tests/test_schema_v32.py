"""v32 — drop dead edges columns (relation / edge_type / description /
stability / decay_rate).

The columns are v22 leftovers: constant or NULL on every row, zero
production readers (edge_relations holds the live relation data, including
its own decay_rate). The migration converges a drifted install to the
declared 7-column edges shape. These tests drive the migration against the
real drifted 12-column shape and lock idempotency, data preservation, and
ladder registration.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.schema import (  # noqa: E402
    BRAIN_VERSION,
    MAIN_MIGRATIONS,
    _migrate_v32_drop_dead_edge_columns,
)


def _edge_cols(conn):
    return {row[1] for row in
            conn.execute("PRAGMA table_info(edges)").fetchall()}


def _legacy_conn():
    """In-memory edges table in the drifted 12-column live shape + dead index."""
    conn = sqlite3.connect(':memory:')
    conn.execute("""CREATE TABLE edges (
        edge_id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        weight REAL DEFAULT 0.5,
        co_access_count INTEGER DEFAULT 0,
        last_strengthened TEXT,
        created_at TEXT,
        relation TEXT DEFAULT 'related',
        stability REAL DEFAULT 0.5,
        edge_type TEXT DEFAULT 'related',
        decay_rate REAL,
        description TEXT DEFAULT '',
        UNIQUE(source_id, target_id)
    )""")
    conn.execute("CREATE INDEX idx_edges_type ON edges(edge_type)")
    conn.execute(
        "INSERT INTO edges (edge_id, source_id, target_id, weight, "
        "co_access_count, created_at) VALUES "
        "('e1', 'a', 'b', 0.7, 3, '2026-01-01T00:00:00+00:00'), "
        "('e2', 'b', 'c', 0.4, 0, '2026-02-01T00:00:00+00:00')")
    return conn


class DropDeadEdgeColumnsTest(unittest.TestCase):

    def test_drops_all_five_columns_and_the_index(self):
        conn = _legacy_conn()
        _migrate_v32_drop_dead_edge_columns(conn)
        cols = _edge_cols(conn)
        self.assertFalse({'relation', 'edge_type', 'description',
                          'stability', 'decay_rate'} & cols)
        self.assertEqual(cols, {'edge_id', 'source_id', 'target_id',
                                'weight', 'co_access_count',
                                'last_strengthened', 'created_at'})
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        self.assertNotIn('idx_edges_type', indexes)

    def test_rows_and_live_columns_survive(self):
        conn = _legacy_conn()
        _migrate_v32_drop_dead_edge_columns(conn)
        rows = conn.execute(
            "SELECT edge_id, source_id, target_id, weight, co_access_count, "
            "created_at FROM edges ORDER BY edge_id").fetchall()
        self.assertEqual(rows, [
            ('e1', 'a', 'b', 0.7, 3, '2026-01-01T00:00:00+00:00'),
            ('e2', 'b', 'c', 0.4, 0, '2026-02-01T00:00:00+00:00')])

    def test_rerun_is_a_no_op(self):
        conn = _legacy_conn()
        _migrate_v32_drop_dead_edge_columns(conn)
        before = _edge_cols(conn)
        _migrate_v32_drop_dead_edge_columns(conn)
        self.assertEqual(_edge_cols(conn), before)

    def test_clean_shape_table_is_untouched(self):
        conn = sqlite3.connect(':memory:')
        conn.execute("""CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            co_access_count INTEGER DEFAULT 0,
            last_strengthened TEXT,
            created_at TEXT,
            UNIQUE(source_id, target_id)
        )""")
        conn.execute("INSERT INTO edges (edge_id, source_id, target_id) "
                     "VALUES ('e1', 'a', 'b')")
        _migrate_v32_drop_dead_edge_columns(conn)
        self.assertEqual(
            _edge_cols(conn),
            {'edge_id', 'source_id', 'target_id', 'weight',
             'co_access_count', 'last_strengthened', 'created_at'})

    def test_partial_drift_only_drops_what_exists(self):
        conn = sqlite3.connect(':memory:')
        conn.execute("""CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT DEFAULT 'related',
            weight REAL DEFAULT 0.5,
            created_at TEXT
        )""")
        _migrate_v32_drop_dead_edge_columns(conn)
        self.assertNotIn('edge_type', _edge_cols(conn))


class LadderRegistrationTest(unittest.TestCase):

    def test_v32_is_on_the_brain_db_ladder_at_the_current_version(self):
        self.assertIn(32, [v for v, _ in MAIN_MIGRATIONS])
        self.assertGreaterEqual(BRAIN_VERSION, 32)


if __name__ == '__main__':
    unittest.main()
