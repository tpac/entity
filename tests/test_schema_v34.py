"""v34 — drop the TF-IDF index (node_vectors, doc_freq).

Its only reader was recall's keyword step, retired with it. These tests lock
the drop against the real pre-v34 table shapes, idempotency, that no other
table is touched, that a fresh brain never creates the index, and that an
upgraded brain reaches v34 through the ensure_schema ladder.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.schema import (  # noqa: E402
    BRAIN_VERSION,
    BRAIN_VERSION_KEY,
    MAIN_MIGRATIONS,
    _migrate_v34_drop_tfidf_index,
    ensure_schema,
    read_schema_version,
    stamp_schema_version,
)

TFIDF_TABLES = {'node_vectors', 'doc_freq'}
TFIDF_INDEXES = {'idx_vectors_term', 'idx_vectors_node'}


def _names(conn, kind):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def _add_pre_v34_index(conn):
    """The TF-IDF tables and indexes exactly as schema v33 declared them."""
    conn.execute("""CREATE TABLE node_vectors (
        node_id TEXT NOT NULL,
        term TEXT NOT NULL,
        tf REAL NOT NULL,
        tfidf REAL DEFAULT 0,
        PRIMARY KEY (node_id, term),
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )""")
    conn.execute("""CREATE TABLE doc_freq (
        term TEXT PRIMARY KEY,
        count INTEGER NOT NULL DEFAULT 1
    )""")
    conn.execute("CREATE INDEX idx_vectors_term ON node_vectors(term)")
    conn.execute("CREATE INDEX idx_vectors_node ON node_vectors(node_id)")


def _fresh_brain():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    ensure_schema(conn)
    return conn


class DropTfidfIndexTest(unittest.TestCase):

    def test_drops_both_tables_and_their_indexes(self):
        conn = _fresh_brain()
        _add_pre_v34_index(conn)
        conn.execute("INSERT INTO nodes (id, type, title, content) "
                     "VALUES ('n1', 'decision', 't', 'c')")
        conn.execute("INSERT INTO node_vectors (node_id, term, tf) "
                     "VALUES ('n1', 'term', 1.0)")
        conn.execute("INSERT INTO doc_freq (term, count) VALUES ('term', 1)")

        _migrate_v34_drop_tfidf_index(conn)

        self.assertFalse(TFIDF_TABLES & _names(conn, 'table'))
        self.assertFalse(TFIDF_INDEXES & _names(conn, 'index'))
        self.assertEqual(
            conn.execute("SELECT id FROM nodes").fetchall(), [('n1',)],
            'the drop must not touch the nodes it indexed')

    def test_rerun_is_a_no_op(self):
        conn = _fresh_brain()
        _add_pre_v34_index(conn)
        _migrate_v34_drop_tfidf_index(conn)
        tables = _names(conn, 'table')
        _migrate_v34_drop_tfidf_index(conn)
        self.assertEqual(_names(conn, 'table'), tables)

    def test_fresh_brain_never_creates_the_index(self):
        conn = _fresh_brain()
        self.assertFalse(TFIDF_TABLES & _names(conn, 'table'))
        self.assertFalse(TFIDF_INDEXES & _names(conn, 'index'))

    def test_upgraded_brain_reaches_v34_through_the_ladder(self):
        conn = _fresh_brain()
        _add_pre_v34_index(conn)
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY, 33)
        conn.commit()

        ensure_schema(conn)

        self.assertFalse(TFIDF_TABLES & _names(conn, 'table'))
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            BRAIN_VERSION)


class LadderRegistrationTest(unittest.TestCase):

    def test_v34_is_on_the_brain_db_ladder_at_the_current_version(self):
        self.assertIn(34, [v for v, _ in MAIN_MIGRATIONS])
        self.assertGreaterEqual(BRAIN_VERSION, 34)


if __name__ == '__main__':
    unittest.main()
