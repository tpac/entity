"""v35 — each relation row carries its own ends (claim_source_id/claim_target_id).

The migration copies every row's ends from its pair, so every existing claim
reads exactly as it did before. These tests lock the copy, idempotency (a row
that already has ends keeps them), and that an upgraded brain reaches v35
through the ensure_schema ladder.
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
    _migrate_v35_claim_ends,
    ensure_schema,
    read_schema_version,
    stamp_schema_version,
)


def _fresh_brain():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    ensure_schema(conn)
    for nid in ('a', 'b'):
        conn.execute("INSERT INTO nodes (id, type, title, content) "
                     "VALUES (?, 'decision', ?, 'c')", (nid, nid))
    # The pair was first written b → a.
    conn.execute("INSERT INTO edges (edge_id, source_id, target_id) "
                 "VALUES ('e1', 'b', 'a')")
    return conn


def _ends(conn):
    return dict((r[0], (r[1], r[2])) for r in conn.execute(
        "SELECT relation, claim_source_id, claim_target_id "
        "FROM edge_relations WHERE edge_id = 'e1'"))


class ClaimEndsTest(unittest.TestCase):

    def test_rows_without_ends_take_their_pairs_ends(self):
        conn = _fresh_brain()
        conn.execute("INSERT INTO edge_relations (edge_id, relation) "
                     "VALUES ('e1', 'grounds'), ('e1', 'supersedes')")

        _migrate_v35_claim_ends(conn)

        self.assertEqual(_ends(conn), {'grounds': ('b', 'a'),
                                       'supersedes': ('b', 'a')})

    def test_a_row_whose_pair_is_gone_is_left_alone(self):
        conn = _fresh_brain()
        conn.execute("INSERT INTO edge_relations (edge_id, relation) "
                     "VALUES ('gone', 'grounds')")

        _migrate_v35_claim_ends(conn)

        self.assertEqual(conn.execute(
            "SELECT claim_source_id, claim_target_id FROM edge_relations "
            "WHERE edge_id = 'gone'").fetchone(), (None, None))

    def test_a_row_with_ends_keeps_them(self):
        conn = _fresh_brain()
        conn.execute("INSERT INTO edge_relations "
                     "(edge_id, relation, claim_source_id, claim_target_id) "
                     "VALUES ('e1', 'supersedes', 'a', 'b')")

        _migrate_v35_claim_ends(conn)
        _migrate_v35_claim_ends(conn)

        self.assertEqual(_ends(conn), {'supersedes': ('a', 'b')})

    def test_upgraded_brain_reaches_v35_through_the_ladder(self):
        conn = _fresh_brain()
        conn.execute("INSERT INTO edge_relations (edge_id, relation) "
                     "VALUES ('e1', 'grounds')")
        stamp_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY, 34)
        conn.commit()

        ensure_schema(conn)

        self.assertEqual(_ends(conn), {'grounds': ('b', 'a')})
        self.assertEqual(
            read_schema_version(conn, 'brain_meta', BRAIN_VERSION_KEY),
            BRAIN_VERSION)


class LadderRegistrationTest(unittest.TestCase):

    def test_v35_is_on_the_brain_db_ladder_at_the_current_version(self):
        self.assertIn(35, [v for v, _ in MAIN_MIGRATIONS])
        self.assertGreaterEqual(BRAIN_VERSION, 35)


if __name__ == '__main__':
    unittest.main()
