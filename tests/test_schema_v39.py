"""Schema v39 — a community's key decisions move into its `thought`, then
every retired community key's rows go.

`community_key_decisions` is retired (COMMUNITY_RETIRED_KEYS): the encoder
records a community's defining members in the standard `thought` field. The
step carries an old brain's picks over so they stay visible, then deletes the
rows of every retired key — views render each stored key, so a kept row would
show a frozen value as current.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s2.community_contract import COMMUNITY_RETIRED_KEYS
from servers.schema import (
    BRAIN_VERSION,
    MAIN_MIGRATIONS,
    _key_decisions_as_thought,
    _migrate_v39_community_metadata,
    ensure_schema,
)


def _brain_db():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    ensure_schema(conn)
    return conn


class KeyDecisionsAsThoughtTest(unittest.TestCase):

    def test_pairs_become_cited_titles(self):
        self.assertEqual(
            _key_decisions_as_thought(
                '18374a5c: Overnight brain vision, 32ab5545: Partnership, now shared'),
            'Defining moments: Overnight brain vision (id:18374a5c); '
            'Partnership, now shared (id:32ab5545)')

    def test_unparseable_value_is_kept_verbatim(self):
        self.assertEqual(_key_decisions_as_thought('  the hook arc and its fix '),
                         'Defining moments: the hook arc and its fix')


class MigrateV39Test(unittest.TestCase):

    def setUp(self):
        self.conn = _brain_db()
        for nid in ('c1', 'c2', 'c3'):
            self.conn.execute("INSERT INTO nodes (id, type, title, content) "
                              "VALUES (?, 'community', 't', 'c')", (nid,))
        self._kv('c1', 'community_key_decisions', 'aaaa1111: First pick')
        self._kv('c2', 'community_key_decisions', 'bbbb2222: Second pick')
        self._kv('c2', 'thought', 'An earlier read')
        self._kv('c3', 'community_key_decisions', '   ')

    def _kv(self, nid, key, value):
        self.conn.execute("INSERT INTO node_metadata_kv (node_id, key, value) "
                          "VALUES (?, ?, ?)", (nid, key, value))

    def _value(self, nid, key):
        row = self.conn.execute("SELECT value FROM node_metadata_kv "
                                "WHERE node_id = ? AND key = ?", (nid, key)).fetchone()
        return row[0] if row else None

    def test_moves_into_thought_and_leaves_no_old_rows(self):
        _migrate_v39_community_metadata(self.conn)

        self.assertEqual(self._value('c1', 'thought'),
                         'Defining moments: First pick (id:aaaa1111)')
        self.assertEqual(self._value('c2', 'thought'),
                         'An earlier read\n\nDefining moments: Second pick (id:bbbb2222)')
        self.assertIsNone(self._value('c3', 'thought'))
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM node_metadata_kv "
            "WHERE key = 'community_key_decisions'").fetchone()[0], 0)

    def test_row_of_a_deleted_node_is_dropped_not_moved(self):
        # Production carried such rows; a thought written for a missing node
        # fails the foreign key, and a failed step blocks the boot.
        self.conn.commit()   # the pragma is a no-op inside a transaction
        self.conn.execute('PRAGMA foreign_keys = OFF')
        self._kv('gone', 'community_key_decisions', 'cccc3333: Orphaned pick')
        self.conn.commit()
        self.conn.execute('PRAGMA foreign_keys = ON')

        _migrate_v39_community_metadata(self.conn)

        self.assertIsNone(self._value('gone', 'thought'))
        self.assertIsNone(self._value('gone', 'community_key_decisions'))
        self.assertEqual(self._value('c1', 'thought'),
                         'Defining moments: First pick (id:aaaa1111)')

    def test_every_retired_key_goes_and_live_keys_stay(self):
        for key in COMMUNITY_RETIRED_KEYS - {'community_key_decisions'}:
            self._kv('c1', key, 'frozen %s' % key)
        self._kv('c1', 'community_internal_fraction', '0.42')
        self._kv('c1', 'community_learning_arc', 'Vision to validation')
        self.conn.commit()
        self.conn.execute('PRAGMA foreign_keys = OFF')
        self._kv('gone', 'community_size', '7')
        self.conn.commit()
        self.conn.execute('PRAGMA foreign_keys = ON')

        _migrate_v39_community_metadata(self.conn)

        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM node_metadata_kv WHERE key IN (%s)"
            % ','.join('?' * len(COMMUNITY_RETIRED_KEYS)),
            sorted(COMMUNITY_RETIRED_KEYS)).fetchone()[0], 0)
        self.assertEqual(self._value('c1', 'community_internal_fraction'), '0.42')
        self.assertEqual(self._value('c1', 'community_learning_arc'),
                         'Vision to validation')
        self.assertEqual(self._value('c1', 'thought'),
                         'Defining moments: First pick (id:aaaa1111)')

    def test_internal_fraction_takes_the_stamps_two_decimal_form(self):
        # The stamp compares stored strings against str(round(x, 2)); a
        # three-decimal value left behind would read as moved on the first
        # cycle and be revised for nothing.
        self._kv('c1', 'community_internal_fraction', '0.444')
        self._kv('c2', 'community_internal_fraction', '0.5')
        self._kv('c3', 'community_internal_fraction', 'n/a')

        _migrate_v39_community_metadata(self.conn)

        self.assertEqual(self._value('c1', 'community_internal_fraction'), '0.44')
        self.assertEqual(self._value('c2', 'community_internal_fraction'), '0.5')
        self.assertEqual(self._value('c3', 'community_internal_fraction'), 'n/a')

    def test_v39_is_on_the_brain_db_ladder(self):
        self.assertIn((39, _migrate_v39_community_metadata), MAIN_MIGRATIONS)
        self.assertGreaterEqual(BRAIN_VERSION, 39)

    def test_rerun_changes_nothing(self):
        _migrate_v39_community_metadata(self.conn)
        before = self.conn.execute(
            "SELECT node_id, key, value FROM node_metadata_kv ORDER BY 1, 2").fetchall()
        _migrate_v39_community_metadata(self.conn)
        self.assertEqual(self.conn.execute(
            "SELECT node_id, key, value FROM node_metadata_kv ORDER BY 1, 2").fetchall(),
            before)


if __name__ == '__main__':
    unittest.main()
