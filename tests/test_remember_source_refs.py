"""Integration tests for source_refs through the full brain.remember()
and brain.revise() write paths (v29 / Phase B Steps 2-4).

These tests exercise the MCP-schema → dispatch validation → brain.remember
kwarg → SourceRefDAL persistence path end-to-end via an isolated Brain
instance. Pure-DAL tests live in test_episodic_refs_dal.py; this file
locks the integration contract.
"""

import os
import shutil
import tempfile
import unittest

from servers.brain import Brain
from servers.dal import SourceRefDAL
from servers.dispatch_write import (
    _validate_source_refs,
    _maybe_warn_source_refs_hex_format,
    _maybe_warn_source_refs_sparseness,
)


class RememberSourceRefsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='brain_source_refs_test_')
        cls.brain = Brain(db_path=os.path.join(cls.tmpdir, 'brain.db'))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.brain.conn.close()
            cls.brain.logs_conn.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        # Seed a couple of trace_event rows so we have real ids to anchor against
        self.brain.logs_conn.execute(
            "INSERT OR REPLACE INTO trace_events "
            "(id, chain_id, scale, event_type, ref_type, summary, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ('a3f5e2b1', 'chain-test', 's0', 'K', 'user_message',
             'ada probe', 'sess-test', '2026-05-25T00:00:00Z'))
        self.brain.logs_conn.execute(
            "INSERT OR REPLACE INTO trace_events "
            "(id, chain_id, scale, event_type, ref_type, summary, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ('b8c9d0e1', 'chain-test', 's0', 'delta', 'assistant_message',
             'anchor reply', 'sess-test', '2026-05-25T00:00:01Z'))
        self.brain.logs_conn.commit()
        self.created_node_ids = []

    def tearDown(self):
        gd = SourceRefDAL(self.brain.conn)
        for nid in self.created_node_ids:
            try:
                gd.replace_source_refs(nid, [])
                self.brain.conn.execute('DELETE FROM nodes WHERE id = ?', (nid,))
                self.brain.conn.commit()
            except Exception:
                pass

    def _create_node(self, **kwargs):
        result = self.brain.remember(
            type=kwargs.pop('type', 'principle'),
            title=kwargs.pop('title', 'test'),
            content=kwargs.pop('content', 'test content'),
            encoding_source='anchor',
            **kwargs,
        )
        nid = result['id']
        self.created_node_ids.append(nid)
        return nid, result

    # ── remember() ────────────────────────────────────────────

    def test_remember_persists_source_refs(self):
        nid, _ = self._create_node(source_refs=['a3f5e2b1', 'b8c9d0e1'])
        gd = SourceRefDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), ['a3f5e2b1', 'b8c9d0e1'])

    def test_remember_without_source_refs_leaves_table_empty(self):
        nid, _ = self._create_node()
        gd = SourceRefDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), [])

    def test_remember_empty_list_is_noop(self):
        nid, _ = self._create_node(source_refs=[])
        gd = SourceRefDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), [])

    def test_engram_cohort_via_shared_ref(self):
        """Two nodes anchored to the same trace_id form a cohort retrievable
        via get_nodes_referencing — the substrate for the future co_anchored
        edge (Step 7)."""
        nid1, _ = self._create_node(title='node 1', source_refs=['a3f5e2b1'])
        nid2, _ = self._create_node(title='node 2', source_refs=['a3f5e2b1'])
        gd = SourceRefDAL(self.brain.conn)
        cohort = sorted(gd.get_nodes_referencing('a3f5e2b1'))
        self.assertEqual(cohort, sorted([nid1, nid2]))

    # ── revise() — field-level REPLACE per decision 995ffeb1 ──

    def test_revise_with_source_refs_replaces_not_appends(self):
        """When source_refs IS in the revise payload → REPLACE entire list."""
        nid, _ = self._create_node(source_refs=['a3f5e2b1'])
        gd = SourceRefDAL(self.brain.conn)
        self.assertEqual(gd.get_source_refs(nid), ['a3f5e2b1'])

        result = self.brain.revise(nid, reason='swap anchor', source_refs=['b8c9d0e1'])
        self.assertIn('source_refs', result.get('fields_updated', []))
        self.assertEqual(gd.get_source_refs(nid), ['b8c9d0e1'])

    def test_revise_without_source_refs_preserves_existing(self):
        """When source_refs is ABSENT from the revise payload → preserve."""
        nid, _ = self._create_node(source_refs=['a3f5e2b1', 'b8c9d0e1'])
        gd = SourceRefDAL(self.brain.conn)
        self.brain.revise(nid, reason='content only',
                          content='new content (refs untouched)')
        self.assertEqual(gd.get_source_refs(nid), ['a3f5e2b1', 'b8c9d0e1'])

    def test_revise_with_empty_list_explicitly_clears(self):
        """Empty list is the explicit clear signal (per unified contract:
        present-field replaces, even with empty value)."""
        nid, _ = self._create_node(source_refs=['a3f5e2b1', 'b8c9d0e1'])
        gd = SourceRefDAL(self.brain.conn)
        result = self.brain.revise(nid, reason='clear refs', source_refs=[])
        self.assertIn('source_refs', result.get('fields_updated', []))
        self.assertEqual(gd.get_source_refs(nid), [])
        self.assertEqual(result.get('source_refs_replaced'), 0)

    # ── Step 7: co_anchored auto-edge ─────────────────────────

    def _has_co_anchored_edge(self, a: str, b: str) -> bool:
        """Check whether a co_anchored relation exists between a and b
        (direction-agnostic — physical edges are single-row per pair)."""
        gd = SourceRefDAL(self.brain.conn)
        # Try both directions; physical edges store one row per pair
        for (s, t) in [(a, b), (b, a)]:
            rows = self.brain.conn.execute(
                'SELECT er.relation FROM edges e '
                'JOIN edge_relations er ON er.edge_id = e.edge_id '
                'WHERE e.source_id = ? AND e.target_id = ? '
                'AND er.relation = ? AND er.archived = 0',
                (s, t, 'co_anchored')).fetchall()
            if rows:
                return True
        return False

    def test_co_anchored_fires_for_shared_ref(self):
        """Two nodes sharing one trace_id get a co_anchored edge auto-written
        at remember() time."""
        nid1, _ = self._create_node(title='cohort A', source_refs=['a3f5e2b1'])
        nid2, _ = self._create_node(title='cohort B', source_refs=['a3f5e2b1'])
        self.assertTrue(self._has_co_anchored_edge(nid1, nid2))

    def test_co_anchored_skips_self(self):
        """A node with refs but no siblings produces no co_anchored edges to
        itself."""
        nid, _ = self._create_node(title='solo', source_refs=['a3f5e2b1'])
        # No siblings means no co_anchored edges from this node at all
        rows = self.brain.conn.execute(
            'SELECT er.relation FROM edges e '
            'JOIN edge_relations er ON er.edge_id = e.edge_id '
            'WHERE (e.source_id = ? OR e.target_id = ?) '
            'AND er.relation = ? AND er.archived = 0',
            (nid, nid, 'co_anchored')).fetchall()
        self.assertEqual(rows, [])

    def test_co_anchored_three_node_cohort(self):
        """Three nodes anchored to the same trace get co_anchored edges
        to each other (full clique within the cohort)."""
        nid1, _ = self._create_node(title='A', source_refs=['a3f5e2b1'])
        nid2, _ = self._create_node(title='B', source_refs=['a3f5e2b1'])
        nid3, _ = self._create_node(title='C', source_refs=['a3f5e2b1'])
        self.assertTrue(self._has_co_anchored_edge(nid1, nid2))
        self.assertTrue(self._has_co_anchored_edge(nid1, nid3))
        self.assertTrue(self._has_co_anchored_edge(nid2, nid3))

    def test_co_anchored_no_edge_for_disjoint_refs(self):
        """Two nodes with disjoint source_refs share no anchor — no
        co_anchored edge."""
        nid1, _ = self._create_node(title='X', source_refs=['a3f5e2b1'])
        nid2, _ = self._create_node(title='Y', source_refs=['b8c9d0e1'])
        self.assertFalse(self._has_co_anchored_edge(nid1, nid2))

    def test_co_anchored_fires_on_revise_replace(self):
        """When revise REPLACES source_refs to include a shared ref, the
        new cohort gets co_anchored edges."""
        nid1, _ = self._create_node(title='target', source_refs=['a3f5e2b1'])
        nid2, _ = self._create_node(title='isolated', source_refs=['b8c9d0e1'])
        self.assertFalse(self._has_co_anchored_edge(nid1, nid2))
        # Now revise nid2 to share nid1's anchor
        self.brain.revise(nid2, reason='swap anchor to join cohort',
                          source_refs=['a3f5e2b1'])
        self.assertTrue(self._has_co_anchored_edge(nid1, nid2))

    # ── dispatch validator ────────────────────────────────────

    def test_validator_accepts_well_formed_list(self):
        ok, err = _validate_source_refs(['a3f5e2b1', 'b8c9d0e1'], 'test')
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_validator_accepts_none(self):
        ok, err = _validate_source_refs(None, 'test')
        self.assertTrue(ok)

    def test_validator_accepts_empty_list(self):
        """Empty list is a legitimate shape (explicit clear on revise;
        pure-synthesis no-anchor pattern on remember)."""
        ok, err = _validate_source_refs([], 'test')
        self.assertTrue(ok)

    def test_validator_rejects_non_list(self):
        ok, err = _validate_source_refs('a3f5e2b1', 'test')
        self.assertFalse(ok)
        self.assertIn('must be a list', err)

    def test_validator_rejects_int_in_list(self):
        """v29: ints rejected loudly (reviewer F2)."""
        ok, err = _validate_source_refs([42], 'test')
        self.assertFalse(ok)
        self.assertIn('hex string', err)

    def test_validator_rejects_non_string_element(self):
        ok, err = _validate_source_refs(['valid', None], 'test')
        self.assertFalse(ok)

    def test_validator_rejects_empty_string_element(self):
        ok, err = _validate_source_refs(['valid', '  '], 'test')
        self.assertFalse(ok)
        self.assertIn('empty', err)

    # ── Layer 1 soft-warn validators (v22 eval gate) ──────────

    def test_hex_format_warn_fires_for_placeholder(self):
        """Encoder copied a literal `<trace-...>` from an example into
        production — hex-format warn fires; write proceeds (refs persist)."""
        warnings_logged = []
        original_log = self.brain._log_warning
        self.brain._log_warning = lambda kind, msg, **kw: warnings_logged.append((kind, msg))
        try:
            _maybe_warn_source_refs_hex_format(
                self.brain, ['<trace-placeholder>', 'a3f5e2b1'], 'test')
        finally:
            self.brain._log_warning = original_log
        kinds = [k for k, _ in warnings_logged]
        self.assertIn('source_refs_hex_format', kinds)

    def test_hex_format_warn_silent_on_valid_hex(self):
        """Well-formed 8-char hex refs produce no hex-format warn."""
        warnings_logged = []
        original_log = self.brain._log_warning
        self.brain._log_warning = lambda kind, msg, **kw: warnings_logged.append((kind, msg))
        try:
            _maybe_warn_source_refs_hex_format(
                self.brain, ['a3f5e2b1', 'b8c9d0e1'], 'test')
        finally:
            self.brain._log_warning = original_log
        self.assertEqual(warnings_logged, [])

    def test_sparseness_warn_threshold_lowered_to_5(self):
        """v22 lowered the sparsity threshold from 10 to 5 to match the
        prompt's §7.5 teaching (1-3 typical, second-guess at 5-6)."""
        warnings_logged = []
        original_log = self.brain._log_warning
        self.brain._log_warning = lambda kind, msg, **kw: warnings_logged.append((kind, msg))
        try:
            # 5 refs = silent (boundary)
            _maybe_warn_source_refs_sparseness(
                self.brain, ['a3f5e2b1'] * 5, 'test_5')
            # 6 refs = fires
            _maybe_warn_source_refs_sparseness(
                self.brain, ['a3f5e2b1'] * 6, 'test_6')
        finally:
            self.brain._log_warning = original_log
        kinds = [k for k, _ in warnings_logged]
        self.assertNotIn('test_5', ' '.join(m for _, m in warnings_logged))
        self.assertEqual(kinds.count('source_refs_sparseness'), 1)


if __name__ == '__main__':
    unittest.main()
