"""Schema v31 — voice-quote field rename, and the primitive it rides on.

`rename_kv_field` is the reusable half: a promoted metadata field stores its
NAME in two places, and both must move together —

  node_metadata_kv.key          the value
  node_enrichments.vector_type  the per-field embedding lane

Renaming only the kv key is the failure this file exists to prevent: the
activation kernel would look up the new name, miss, fall back to the
`high_meta` blend, and a lazy backfill would re-embed thousands of rows that
were never stale. The blobs stay valid across a rename because field names
never enter the embedded text — the builders join values only.

v31's own step is the first customer: user_raw_quote → their_raw_quote,
anchor_raw_quote → my_raw_quote.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.schema import (
    BRAIN_VERSION,
    _migrate_v31_voice_fields,
    ensure_schema,
    rename_kv_field,
)


def _brain_db():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    ensure_schema(conn)
    return conn


def _node(conn, node_id):
    conn.execute("INSERT INTO nodes (id, type, title, content) "
                 "VALUES (?, 'decision', 't', 'c')", (node_id,))


def _kv(conn, node_id, key, value):
    conn.execute("INSERT INTO node_metadata_kv (node_id, key, value) "
                 "VALUES (?, ?, ?)", (node_id, key, value))


def _vector(conn, node_id, vector_type, text='txt', blob=b'\x01\x02'):
    conn.execute(
        "INSERT INTO node_enrichments (id, node_id, vector_type, text, "
        "embedding) VALUES (?, ?, ?, ?, ?)",
        ('%s-%s' % (node_id, vector_type), node_id, vector_type, text, blob))


def _kv_keys(conn, node_id):
    return {r[0] for r in conn.execute(
        "SELECT key FROM node_metadata_kv WHERE node_id = ?", (node_id,))}


def _vector_types(conn, node_id):
    return {r[0] for r in conn.execute(
        "SELECT vector_type FROM node_enrichments WHERE node_id = ?",
        (node_id,))}


class RenameKvFieldTest(unittest.TestCase):
    """The primitive, in isolation."""

    def setUp(self):
        self.conn = _brain_db()
        _node(self.conn, 'n1')

    def test_renames_both_the_kv_key_and_the_vector_lane(self):
        _kv(self.conn, 'n1', 'old_field', 'the value')
        _vector(self.conn, 'n1', 'old_field')

        counts = rename_kv_field(self.conn, 'old_field', 'new_field')

        self.assertEqual(counts,
                         {'node_metadata_kv': 1, 'node_enrichments': 1})
        self.assertEqual(_kv_keys(self.conn, 'n1'), {'new_field'})
        self.assertEqual(_vector_types(self.conn, 'n1'), {'new_field'})

    def test_value_and_embedding_survive_untouched(self):
        """A rename is a relabel. Nothing about the payload may change."""
        _kv(self.conn, 'n1', 'old_field', 'the exact words')
        _vector(self.conn, 'n1', 'old_field', text='embedded text',
                blob=b'\xde\xad\xbe\xef')

        rename_kv_field(self.conn, 'old_field', 'new_field')

        value = self.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE key = 'new_field'"
        ).fetchone()[0]
        text, blob = self.conn.execute(
            "SELECT text, embedding FROM node_enrichments "
            "WHERE vector_type = 'new_field'").fetchone()
        self.assertEqual(value, 'the exact words')
        self.assertEqual(text, 'embedded text')
        self.assertEqual(blob, b'\xde\xad\xbe\xef')

    def test_leaves_other_fields_and_lanes_alone(self):
        _kv(self.conn, 'n1', 'old_field', 'v')
        _kv(self.conn, 'n1', 'situation', 'when x')
        _vector(self.conn, 'n1', 'old_field')
        _vector(self.conn, 'n1', 'high_meta')

        rename_kv_field(self.conn, 'old_field', 'new_field')

        self.assertEqual(_kv_keys(self.conn, 'n1'), {'new_field', 'situation'})
        self.assertEqual(_vector_types(self.conn, 'n1'),
                         {'new_field', 'high_meta'})

    def test_rerun_is_a_no_op(self):
        """Idempotent by construction — the UPDATEs match the OLD name, so a
        retried migration (crash before the stamp) finds nothing to do."""
        _kv(self.conn, 'n1', 'old_field', 'v')
        _vector(self.conn, 'n1', 'old_field')
        rename_kv_field(self.conn, 'old_field', 'new_field')

        counts = rename_kv_field(self.conn, 'old_field', 'new_field')

        self.assertEqual(counts,
                         {'node_metadata_kv': 0, 'node_enrichments': 0})
        self.assertEqual(_kv_keys(self.conn, 'n1'), {'new_field'})

    def test_field_with_no_vector_lane_renames_cleanly(self):
        """Most promoted fields have no field-cohort vector. The vector UPDATE
        must be a harmless zero, not a reason to special-case callers."""
        _kv(self.conn, 'n1', 'correction_pattern', 'p')

        counts = rename_kv_field(self.conn, 'correction_pattern', 'renamed')

        self.assertEqual(counts['node_metadata_kv'], 1)
        self.assertEqual(counts['node_enrichments'], 0)


class MigrateV31Test(unittest.TestCase):
    """The v31 step: both voice fields, both tables, one pass."""

    def setUp(self):
        self.conn = _brain_db()
        _node(self.conn, 'n1')
        _kv(self.conn, 'n1', 'user_raw_quote', 'what they said')
        _kv(self.conn, 'n1', 'anchor_raw_quote', 'what I said')
        _kv(self.conn, 'n1', 'situation', 'when x')
        _vector(self.conn, 'n1', 'user_raw_quote')
        _vector(self.conn, 'n1', 'anchor_raw_quote')
        _vector(self.conn, 'n1', 'high_meta')

    def test_both_voice_fields_move_in_kv_and_in_the_vector_lanes(self):
        _migrate_v31_voice_fields(self.conn)

        self.assertEqual(_kv_keys(self.conn, 'n1'),
                         {'their_raw_quote', 'my_raw_quote', 'situation'})
        self.assertEqual(_vector_types(self.conn, 'n1'),
                         {'their_raw_quote', 'my_raw_quote', 'high_meta'})

    def test_the_two_voices_do_not_cross(self):
        """Their words must land in their field. A swapped pair would be a
        silent, unrecoverable corruption of who said what."""
        _migrate_v31_voice_fields(self.conn)

        rows = dict(self.conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id = 'n1'"))
        self.assertEqual(rows['their_raw_quote'], 'what they said')
        self.assertEqual(rows['my_raw_quote'], 'what I said')

    def test_rerun_is_a_no_op(self):
        _migrate_v31_voice_fields(self.conn)
        _migrate_v31_voice_fields(self.conn)

        self.assertEqual(_kv_keys(self.conn, 'n1'),
                         {'their_raw_quote', 'my_raw_quote', 'situation'})

    def test_a_node_carrying_only_one_voice_migrates(self):
        _node(self.conn, 'n2')
        _kv(self.conn, 'n2', 'user_raw_quote', 'only theirs')

        _migrate_v31_voice_fields(self.conn)

        self.assertEqual(_kv_keys(self.conn, 'n2'), {'their_raw_quote'})


class LadderWiringTest(unittest.TestCase):
    """v31 must be reachable from the runner, not just callable by hand."""

    def test_v31_is_on_the_brain_db_ladder_at_the_current_version(self):
        from servers.schema import MAIN_MIGRATIONS
        self.assertIn(31, [v for v, _ in MAIN_MIGRATIONS])
        self.assertGreaterEqual(BRAIN_VERSION, 31)

    def test_the_contract_names_the_new_fields_and_not_the_old(self):
        """The schema and the field contract have to agree, or the migration
        renames rows into a name the write boundary rejects."""
        from servers.contract import PROMOTED_FIELDS
        kv = {k for k, v in PROMOTED_FIELDS.items() if v.get('store') == 'metadata_kv'}
        self.assertIn('their_raw_quote', kv)
        self.assertIn('my_raw_quote', kv)
        self.assertNotIn('user_raw_quote', kv)
        self.assertNotIn('anchor_raw_quote', kv)

    def test_the_embedding_lanes_follow_the_field_names(self):
        """`vectors_affected_by` is what invalidates a node's vectors on
        revise. If it still answered on the old names, an edited quote would
        keep its stale vector forever."""
        from servers.pipeline_contract import (FIELD_VECTOR_FALLBACK,
                                               field_vector_types,
                                               vectors_affected_by)
        for field in ('their_raw_quote', 'my_raw_quote'):
            self.assertIn(field, field_vector_types())
            self.assertIn(field, FIELD_VECTOR_FALLBACK)
            self.assertEqual(vectors_affected_by(field),
                             {field, 'high_meta'})


if __name__ == '__main__':
    unittest.main()
