"""Schema v27 — episodic references substrate.

Locks the contract for two new tables:
  - node_source_refs (brain.db) — multi-ref pointer from node → trace_event
  - trace_embeddings (brain_logs.db) — per-trace vector with embedded-text snapshot

These tests are pure-schema: structural shape, idempotency, PK enforcement,
FK cascade. They don't require a live Brain or embedder.

Background: v27 replaced the single-ref `nodes.source_turn_id` column (v9,
dropped in v36) with a multi-ref join table.

v29 update (2026-05-25): trace_id columns migrated INTEGER → TEXT (8-char
hex) for brain-wide ID consistency with nodes and edges. This file's
assertions reflect the v29 column types; the v27 substrate (tables, indexes,
FK cascade) is unchanged.
"""

import os
import sqlite3
import sys
import unittest

# Ensure servers package importable when run via pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.schema import (
    BRAIN_VERSION,
    ensure_schema,
    ensure_logs_schema,
)


def _open_in_memory():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def _table_columns(conn, table_name):
    """Return {column_name: column_type} for a table."""
    cur = conn.execute('PRAGMA table_info(%s)' % table_name)
    return {row[1]: row[2] for row in cur.fetchall()}


def _index_exists(conn, index_name):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name = ?",
        (index_name,))
    return cur.fetchone() is not None


class SchemaVersionTest(unittest.TestCase):
    def test_brain_version_is_at_least_27(self):
        # v27 substrate (this file's scope) still installed; subsequent
        # migrations (v28 drop-keywords, etc.) layer on top. Was an
        # equality pin before 2026-05-24; floor-check keeps this test's
        # intent stable across upgrades.
        self.assertGreaterEqual(BRAIN_VERSION, 27)


class NodeSourceRefsTest(unittest.TestCase):
    """node_source_refs lives in brain.db (TABLES via ensure_schema)."""

    def setUp(self):
        self.conn = _open_in_memory()
        ensure_schema(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_table_exists(self):
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name = 'node_source_refs'")
        self.assertIsNotNone(cur.fetchone())

    def test_columns_shape(self):
        cols = _table_columns(self.conn, 'node_source_refs')
        self.assertEqual(cols.get('node_id'), 'TEXT')
        # v29: trace_id migrated INTEGER → TEXT (8-char hex)
        self.assertEqual(cols.get('trace_id'), 'TEXT')
        self.assertEqual(cols.get('position'), 'INTEGER')
        self.assertEqual(cols.get('created_at'), 'TEXT')

    def test_idx_nsr_trace_exists(self):
        self.assertTrue(_index_exists(self.conn, 'idx_nsr_trace'))

    def test_primary_key_prevents_duplicate_ref(self):
        # Seed a node so the FK doesn't fire
        self.conn.execute(
            "INSERT INTO nodes (id, type, title) VALUES (?, ?, ?)",
            ('node_a', 'concept', 'A'))
        self.conn.execute(
            "INSERT INTO node_source_refs (node_id, trace_id, position, created_at) "
            "VALUES (?, ?, ?, ?)",
            ('node_a', 42, 1, '2026-05-23T00:00:00Z'))
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO node_source_refs (node_id, trace_id, position, created_at) "
                "VALUES (?, ?, ?, ?)",
                ('node_a', 42, 2, '2026-05-23T00:01:00Z'))

    def test_fk_cascade_on_node_delete(self):
        self.conn.execute(
            "INSERT INTO nodes (id, type, title) VALUES (?, ?, ?)",
            ('node_b', 'concept', 'B'))
        self.conn.execute(
            "INSERT INTO node_source_refs (node_id, trace_id, position, created_at) "
            "VALUES (?, ?, ?, ?)",
            ('node_b', 99, 1, '2026-05-23T00:00:00Z'))
        self.conn.execute("DELETE FROM nodes WHERE id = ?", ('node_b',))
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM node_source_refs WHERE node_id = ?",
            ('node_b',))
        self.assertEqual(cur.fetchone()[0], 0)

    def test_multiple_refs_per_node_allowed(self):
        self.conn.execute(
            "INSERT INTO nodes (id, type, title) VALUES (?, ?, ?)",
            ('node_c', 'concept', 'C'))
        # v29: trace_id is TEXT (8-char hex); test uses canonical hex form.
        for trace_id, position in [('00000001', 1), ('00000002', 2), ('00000003', 3)]:
            self.conn.execute(
                "INSERT INTO node_source_refs (node_id, trace_id, position, created_at) "
                "VALUES (?, ?, ?, ?)",
                ('node_c', trace_id, position, '2026-05-23T00:00:00Z'))
        cur = self.conn.execute(
            "SELECT trace_id FROM node_source_refs WHERE node_id = ? "
            "ORDER BY position",
            ('node_c',))
        self.assertEqual([r[0] for r in cur.fetchall()],
                         ['00000001', '00000002', '00000003'])


class TraceEmbeddingsTest(unittest.TestCase):
    """trace_embeddings lives in brain_logs.db (LOG_TABLES via ensure_logs_schema)."""

    def setUp(self):
        self.conn = _open_in_memory()
        ensure_logs_schema(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_table_exists(self):
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name = 'trace_embeddings'")
        self.assertIsNotNone(cur.fetchone())

    def test_columns_shape(self):
        cols = _table_columns(self.conn, 'trace_embeddings')
        # v29: trace_id migrated INTEGER → TEXT (8-char hex)
        self.assertEqual(cols.get('trace_id'), 'TEXT')
        self.assertEqual(cols.get('vector'), 'BLOB')
        self.assertEqual(cols.get('text'), 'TEXT')
        self.assertEqual(cols.get('model'), 'TEXT')
        self.assertEqual(cols.get('created_at'), 'TEXT')

    def test_primary_key_prevents_duplicate_trace(self):
        # v29: trace_id is TEXT (hex). Use canonical hex form for test inputs.
        self.conn.execute(
            "INSERT INTO trace_embeddings (trace_id, vector, text, model, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ('00000001', b'\x00\x01', 'first render', 'nomic', '2026-05-23T00:00:00Z'))
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO trace_embeddings (trace_id, vector, text, model, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ('00000001', b'\x02\x03', 'second render', 'nomic', '2026-05-23T00:01:00Z'))

    def test_vector_not_null(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO trace_embeddings (trace_id, vector) VALUES (?, ?)",
                ('00000001', None))


class IdempotencyTest(unittest.TestCase):
    """Running ensure_schema / ensure_logs_schema twice must be a no-op."""

    def test_brain_schema_idempotent(self):
        conn = _open_in_memory()
        try:
            ensure_schema(conn)
            ensure_schema(conn)
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name = 'node_source_refs'")
            self.assertIsNotNone(cur.fetchone())
        finally:
            conn.close()

    def test_logs_schema_idempotent(self):
        conn = _open_in_memory()
        try:
            ensure_logs_schema(conn)
            ensure_logs_schema(conn)
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name = 'trace_embeddings'")
            self.assertIsNotNone(cur.fetchone())
        finally:
            conn.close()


if __name__ == '__main__':
    unittest.main()
