"""brain.db v38 — drop entity_dates, the index of dates mined from node and
edge text.

Locks the drop against the upgraded shape (table + its three indexes, as the
v37 schema created them), that other tables survive, idempotency, that fresh
DBs never create the table, and that a v37 brain reaches the current version
through the real ladder.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.schema import (  # noqa: E402
    BRAIN_VERSION,
    BRAIN_VERSION_KEY,
    ENTITY_DATES_TABLE,
    MAIN_MIGRATIONS,
    _migrate_v38_drop_entity_dates,
    ensure_schema,
    read_schema_version,
    stamp_schema_version,
)


def _names(conn, kind='table'):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def _v37_brain():
    """A current brain given back the v37 shape: the table, its indexes and a
    sentinel row."""
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    ensure_schema(conn)
    conn.execute("""CREATE TABLE entity_dates (
        entity_kind TEXT NOT NULL CHECK(entity_kind IN ('node','edge')),
        entity_id TEXT NOT NULL,
        start_ts INTEGER NOT NULL,
        end_ts INTEGER NOT NULL,
        extraction_source TEXT NOT NULL,
        raw_text TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (entity_kind, entity_id, start_ts, end_ts, extraction_source)
    )""")
    for sql in (
            'CREATE INDEX idx_entity_dates_start ON entity_dates(start_ts)',
            'CREATE INDEX idx_entity_dates_end ON entity_dates(end_ts)',
            'CREATE INDEX idx_entity_dates_kind_id '
            'ON entity_dates(entity_kind, entity_id)'):
        conn.execute(sql)
    conn.execute("INSERT INTO nodes (id, type, title, content) "
                 "VALUES ('n1', 'decision', 't', 'c')")
    conn.execute("INSERT INTO entity_dates (entity_kind, entity_id, start_ts, "
                 "end_ts, extraction_source) VALUES ('node', 'n1', 0, 0, "
                 "'_no_dates_found')")
    return conn


class DropEntityDatesTest(unittest.TestCase):

    def test_drops_the_table_and_its_indexes(self):
        conn = _v37_brain()

        _migrate_v38_drop_entity_dates(conn)

        self.assertNotIn(ENTITY_DATES_TABLE, _names(conn))
        self.assertFalse({i for i in _names(conn, 'index')
                          if i.startswith('idx_entity_dates')})

    def test_other_rows_survive(self):
        conn = _v37_brain()

        _migrate_v38_drop_entity_dates(conn)

        self.assertEqual(conn.execute(
            "SELECT title FROM nodes WHERE id = 'n1'").fetchone(), ('t',))

    def test_rerun_is_a_no_op(self):
        conn = _v37_brain()
        _migrate_v38_drop_entity_dates(conn)

        _migrate_v38_drop_entity_dates(conn)

        self.assertNotIn(ENTITY_DATES_TABLE, _names(conn))

    def test_fresh_brain_never_creates_the_table(self):
        conn = sqlite3.connect(':memory:')
        ensure_schema(conn)
        self.assertNotIn(ENTITY_DATES_TABLE, _names(conn))

    def test_v37_brain_reaches_the_current_version_through_the_ladder(self):
        conn = _v37_brain()
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY, 37)
        conn.commit()

        ensure_schema(conn)

        self.assertNotIn(ENTITY_DATES_TABLE, _names(conn))
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            BRAIN_VERSION)

    def test_v38_is_on_the_brain_db_ladder(self):
        self.assertIn((38, _migrate_v38_drop_entity_dates), MAIN_MIGRATIONS)
        self.assertGreaterEqual(BRAIN_VERSION, 38)


if __name__ == '__main__':
    unittest.main()
