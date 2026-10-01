"""DAL methods for episodic references (v27 substrate; v29 trace_id hex).

  - TraceDAL: store_embeddings / find_unembedded
  - SourceRefDAL: add_source_refs / get_source_refs / get_nodes_referencing

Pure-DAL tests: in-memory SQLite, no live Brain, no embedder.

v29 update (2026-05-25): trace_id migrated INTEGER → TEXT (8-char hex).
Tests use canonical hex form. DAL int→hex coercion is preserved for legacy
callers but the contract is hex strings end-to-end.
"""

import os
import secrets
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.dal import SourceRefDAL
from servers.dal_logs import TraceDAL
from servers.schema import ensure_schema, ensure_logs_schema


def _open_in_memory():
    conn = sqlite3.connect(':memory:')
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def _seed_node(conn, node_id):
    conn.execute(
        "INSERT INTO nodes (id, type, title) VALUES (?, ?, ?)",
        (node_id, 'concept', node_id))


def _seed_trace(conn, scale='s0', ref_type='user_message',
                summary='hello', session_id='sess-1'):
    """Insert a trace_event row and return its hex id (v29)."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    trace_id = secrets.token_hex(4)
    conn.execute(
        'INSERT INTO trace_events '
        '(id, chain_id, scale, event_type, ref_type, summary, '
        ' metadata, session_id, created_at) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (trace_id, 'chain-x', scale, 'K' if ref_type == 'user_message' else 'delta',
         ref_type, summary, None, session_id, now))
    conn.commit()
    return trace_id


class TraceEmbeddingsDALTest(unittest.TestCase):
    def setUp(self):
        self.conn = _open_in_memory()
        ensure_logs_schema(self.conn)
        self.dal = TraceDAL(self.conn)

    def tearDown(self):
        self.conn.close()

    def _read_embeddings(self, ids):
        """Direct-SQL readback of stored trace embeddings ({trace_id: vector}).

        Replaces the removed TraceDAL.get_embeddings readback helper; these
        tests verify store_embeddings (live), not the deleted getter.
        """
        if not ids:
            return {}
        ph = ','.join('?' * len(ids))
        rows = self.conn.execute(
            'SELECT trace_id, vector FROM trace_embeddings WHERE trace_id IN (%s)' % ph,
            list(ids)).fetchall()
        return {r[0]: r[1] for r in rows}

    def test_store_and_get_single(self):
        n = self.dal.store_embeddings(
            [('00000001', b'\x01\x02', 'ada said hi')], model='nomic')
        self.assertEqual(n, 1)
        got = self._read_embeddings(['00000001'])
        self.assertEqual(got, {'00000001': b'\x01\x02'})

    def test_store_skips_null_vectors(self):
        n = self.dal.store_embeddings(
            [('00000001', b'\x00', 'ok'),
             ('00000002', None, 'skip'),
             ('00000003', b'\x02', 'ok2')],
            model='nomic')
        self.assertEqual(n, 2)
        self.assertEqual(
            set(self._read_embeddings(['00000001', '00000002', '00000003']).keys()),
            {'00000001', '00000003'})

    def test_store_replaces_on_conflict(self):
        self.dal.store_embeddings([('00000001', b'\xaa', 'first')], model='nomic')
        self.dal.store_embeddings([('00000001', b'\xbb', 'second')], model='nomic')
        got = self._read_embeddings(['00000001'])
        self.assertEqual(got['00000001'], b'\xbb')

    def test_text_truncation_500_chars(self):
        long_text = 'x' * 1000
        self.dal.store_embeddings([('00000007', b'\x01', long_text)], model='nomic')
        row = self.conn.execute(
            "SELECT text FROM trace_embeddings WHERE trace_id = '00000007'"
        ).fetchone()
        self.assertEqual(len(row[0]), 500)

    def test_find_unembedded_returns_newest_first(self):
        t1 = _seed_trace(self.conn, summary='first')
        t2 = _seed_trace(self.conn, summary='second')
        t3 = _seed_trace(self.conn, summary='third')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'],
            ref_types=['user_message', 'assistant_message', 'tool_result'])
        self.assertEqual([r['id'] for r in rows], [t3, t2, t1])

    def test_find_unembedded_skips_already_embedded(self):
        t1 = _seed_trace(self.conn, summary='first')
        t2 = _seed_trace(self.conn, summary='second')
        self.dal.store_embeddings([(t1, b'\x01', 'first')], model='nomic')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertEqual([r['id'] for r in rows], [t2])

    def test_find_unembedded_respects_limit(self):
        for _ in range(10):
            _seed_trace(self.conn)
        rows = self.dal.find_unembedded(
            limit=3, scales=['s0'], ref_types=['user_message'])
        self.assertEqual(len(rows), 3)

    def test_find_unembedded_filters_by_scale(self):
        _seed_trace(self.conn, scale='s0')
        _seed_trace(self.conn, scale='s1', ref_type='recall')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['scale'], 's0')

    def test_find_unembedded_filters_by_ref_type(self):
        _seed_trace(self.conn, ref_type='user_message')
        _seed_trace(self.conn, ref_type='assistant_message')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['ref_type'], 'user_message')

    def test_find_unembedded_requires_scales(self):
        with self.assertRaises(ValueError):
            self.dal.find_unembedded(
                limit=5, scales=[], ref_types=['user_message'])

    def test_find_unembedded_requires_ref_types(self):
        with self.assertRaises(ValueError):
            self.dal.find_unembedded(
                limit=5, scales=['s0'], ref_types=[])

    def test_find_unembedded_respects_since_cutoff(self):
        """The `since` window excludes traces older than the cutoff —
        bounds embed scope to traces created after identity stamping
        went live (avoids polluting the vector neighborhood with
        sentinel-tagged historical traces)."""
        import time as _time
        t1 = _seed_trace(self.conn, summary='old')
        # Force old created_at on t1
        self.conn.execute(
            "UPDATE trace_events SET created_at = '2025-01-01T00:00:00+00:00' "
            "WHERE id = ?", (t1,))
        self.conn.commit()
        t2 = _seed_trace(self.conn, summary='recent')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'],
            since='2025-06-01T00:00:00+00:00')
        ids = [r['id'] for r in rows]
        self.assertIn(t2, ids)
        self.assertNotIn(t1, ids)

    def test_find_unembedded_no_since_returns_all(self):
        """since=None preserves legacy behavior (no time filter)."""
        t1 = _seed_trace(self.conn, summary='one')
        rows = self.dal.find_unembedded(
            limit=5, scales=['s0'], ref_types=['user_message'])
        self.assertIn(t1, [r['id'] for r in rows])

    def test_decode_metadata_single_encoded(self):
        """Post-fix metadata is single-encoded JSON — one json.loads pass."""
        meta = self.dal._decode_metadata('{"tool": "Bash"}')
        self.assertEqual(meta, {'tool': 'Bash'})

    def test_decode_metadata_double_encoded_legacy(self):
        """Pre-fix tool_result traces stored '"{\\"tool\\": \\"Bash\\"}"'
        (the dispatch json.dumps'd a string client payload, then
        TraceDAL.append json.dumps'd that again). Defensive decode
        unwraps both layers."""
        import json as _json
        double = _json.dumps('{"tool": "Bash"}')  # produces '"{\"tool\": \"Bash\"}"'
        self.assertEqual(self.dal._decode_metadata(double),
                         {'tool': 'Bash'})

    def test_decode_metadata_handles_garbage(self):
        self.assertEqual(self.dal._decode_metadata(None), {})
        self.assertEqual(self.dal._decode_metadata(''), {})
        self.assertEqual(self.dal._decode_metadata('not-json'), {})
        # JSON that decodes to a non-dict primitive — return {} not crash
        self.assertEqual(self.dal._decode_metadata('42'), {})


class GetByIdsTest(unittest.TestCase):
    """Trace point/batch lookup by id — brain.get_trace / get_traces
    backend. Returns ordered list with missing ids silently skipped
    (mirrors NodeDAL.get_bulk semantics)."""

    def setUp(self):
        self.conn = _open_in_memory()
        ensure_logs_schema(self.conn)
        self.dal = TraceDAL(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_single_id(self):
        t = _seed_trace(self.conn, summary='hi')
        rows = self.dal.get_by_ids([t])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['id'], t)
        self.assertEqual(rows[0]['summary'], 'hi')

    def test_batch_returns_in_id_order(self):
        t1 = _seed_trace(self.conn, summary='first')
        t2 = _seed_trace(self.conn, summary='second')
        t3 = _seed_trace(self.conn, summary='third')
        # Input order shouldn't matter — result is ascending id (v29: lex
        # ascending on 8-char hex). Random hex ids mean ascending id != insertion
        # order — assert against sorted() to keep the contract precise.
        rows = self.dal.get_by_ids([t3, t1, t2])
        self.assertEqual([r['id'] for r in rows], sorted([t1, t2, t3]))

    def test_missing_ids_skipped(self):
        t = _seed_trace(self.conn)
        # 'deadbeef' is a 8-char hex placeholder that won't match anything
        rows = self.dal.get_by_ids([t, 'deadbeef', 'cafef00d'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['id'], t)

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(self.dal.get_by_ids([]), [])

    def test_metadata_decoded(self):
        # Insert a row with double-encoded metadata (legacy shape) and
        # verify get_by_ids returns it as a dict via _decode_metadata.
        # v29: insert with explicit hex id (TEXT PK has no autoincrement).
        import json as _json
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        double = _json.dumps('{"tool": "Bash"}')
        tid = secrets.token_hex(4)
        self.conn.execute(
            'INSERT INTO trace_events '
            '(id, chain_id, scale, event_type, ref_type, summary, metadata, '
            ' session_id, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (tid, 'c', 's0', 'delta', 'tool_result', 'Bash: ls',
             double, '', now))
        self.conn.commit()
        rows = self.dal.get_by_ids([tid])
        self.assertEqual(rows[0]['metadata'], {'tool': 'Bash'})


class SourceRefsDALTest(unittest.TestCase):
    def setUp(self):
        self.conn = _open_in_memory()
        ensure_schema(self.conn)
        self.dal = SourceRefDAL(self.conn)
        _seed_node(self.conn, 'node_a')
        _seed_node(self.conn, 'node_b')

    def tearDown(self):
        self.conn.close()

    def test_add_and_get_preserves_order(self):
        n = self.dal.add_source_refs('node_a', ['0000002a', '00000011', '00000063'])
        self.assertEqual(n, 3)
        self.assertEqual(self.dal.get_source_refs('node_a'),
                         ['0000002a', '00000011', '00000063'])

    def test_add_ignores_duplicates(self):
        first = self.dal.add_source_refs('node_a', ['0000000a', '00000014'])
        second = self.dal.add_source_refs('node_a', ['0000000a', '0000001e'])
        self.assertEqual(first, 2)
        self.assertEqual(second, 1)  # only 0000001e new; 0000000a ignored
        self.assertEqual(self.dal.get_source_refs('node_a'),
                         ['0000000a', '00000014', '0000001e'])

    def test_add_empty_inputs_returns_zero(self):
        self.assertEqual(self.dal.add_source_refs('', ['00000001', '00000002']), 0)
        self.assertEqual(self.dal.add_source_refs('node_a', []), 0)

    def test_get_source_refs_empty_node(self):
        self.assertEqual(self.dal.get_source_refs('node_b'), [])
        self.assertEqual(self.dal.get_source_refs('nonexistent'), [])

    def test_get_nodes_referencing_cohort(self):
        self.dal.add_source_refs('node_a', ['00000064', '000000c8'])
        self.dal.add_source_refs('node_b', ['000000c8', '0000012c'])
        cohort = self.dal.get_nodes_referencing('000000c8')
        self.assertEqual(sorted(cohort), ['node_a', 'node_b'])
        self.assertEqual(self.dal.get_nodes_referencing('00000064'), ['node_a'])
        self.assertEqual(self.dal.get_nodes_referencing('deadbeef'), [])

    def test_int_input_rejected_loudly(self):
        """v29 contract: trace_ids are 8-char hex strings end-to-end.
        Coercion was removed because random hex generation made silent
        int→hex unsafe (collision with token_hex output). Int input
        must raise ValueError so the encoder fails loud rather than
        landing colliding refs."""
        with self.assertRaises(ValueError) as ctx:
            self.dal.add_source_refs('node_a', [42, 17])
        self.assertIn('strings', str(ctx.exception))
        with self.assertRaises(ValueError):
            self.dal.get_nodes_referencing(42)

    def test_replace_source_refs_clears_then_inserts(self):
        """replace_source_refs is the revise() persistence path:
        atomic DELETE existing + INSERT new (field-level REPLACE per
        decision 995ffeb1)."""
        self.dal.add_source_refs('node_a', ['00000001', '00000002', '00000003'])
        self.assertEqual(self.dal.get_source_refs('node_a'),
                         ['00000001', '00000002', '00000003'])
        # Replace with two new ones
        n = self.dal.replace_source_refs('node_a', ['deadbeef', 'cafef00d'])
        self.assertEqual(n, 2)
        self.assertEqual(self.dal.get_source_refs('node_a'),
                         ['deadbeef', 'cafef00d'])
        # Replace with empty list clears all refs
        n = self.dal.replace_source_refs('node_a', [])
        self.assertEqual(n, 0)
        self.assertEqual(self.dal.get_source_refs('node_a'), [])

    def test_replace_source_refs_rejects_ints(self):
        """Same v29 contract as add_source_refs."""
        with self.assertRaises(ValueError):
            self.dal.replace_source_refs('node_a', [42, 17])

    def test_fk_cascade_clears_refs(self):
        self.dal.add_source_refs('node_a',
                                 ['00000005', '00000006', '00000007'])
        self.conn.execute("DELETE FROM nodes WHERE id = 'node_a'")
        self.assertEqual(self.dal.get_source_refs('node_a'), [])


if __name__ == '__main__':
    unittest.main()
