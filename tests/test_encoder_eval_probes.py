"""Smoke tests for eval/encoder_eval/quality_probes.py.

Creates a tiny synthetic brain with hand-crafted nodes + source_refs, then
runs every probe and asserts the result shapes are sane. Catches regressions
in the probe code path before any real eval burns cost.
"""
import os
import shutil
import tempfile
import unittest

from servers.brain import Brain
from servers.dal import SourceRefDAL
from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

from eval.encoder_eval.quality_probes import (
    ALL_PROBES,
    run_all_probes,
    probe_brain_presence,
    probe_source_refs_coverage,
    probe_atomization_shape,
    probe_edge_structure,
    probe_voice_balance,
    probe_specificity_preservation,
)


class EncoderEvalProbesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix='encoder_eval_probes_test_')
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
        # Seed a couple of trace_events so source_refs has real targets
        for tid, summary in [('aa11bb22', 'ada: 15 days total'),
                              ('cc33dd44', 'anchor: noted')]:
            self.brain.logs_conn.execute(
                "INSERT OR REPLACE INTO trace_events "
                "(id, chain_id, scale, event_type, ref_type, summary, "
                " session_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (tid, 'chain-probe', 's0', 'K', 'user_message', summary,
                 'sess-probe', '2026-05-25T00:00:00Z'))
        self.brain.logs_conn.commit()
        self.created = []

    def tearDown(self):
        gd = SourceRefDAL(self.brain.conn)
        for nid in self.created:
            try:
                gd.replace_source_refs(nid, [])
                self.brain.conn.execute('DELETE FROM nodes WHERE id = ?', (nid,))
                self.brain.conn.commit()
            except Exception:
                pass

    def _make(self, **kwargs):
        kwargs.setdefault('type', 'fact')
        kwargs.setdefault('encoding_source', 'encoder:sonnet')
        result = self.brain.remember(**kwargs)
        self.created.append(result['id'])
        return result['id']

    # ─── Probe smoke tests ───

    def test_brain_presence_finds_gold_atom(self):
        self._make(title='15-day total across Hawaii and NYC',
                    content='Ada spent 15 days on the combined trip',
                    source_refs=['aa11bb22'])
        item = {'question': 'how many total days?',
                 'answer': '15 days', 'haystack_sessions': []}
        r = probe_brain_presence(self.brain, item)
        self.assertTrue(r['found'])
        self.assertGreater(r['score'], 0.5)
        self.assertEqual(r['nodes_encoded'], 1)

    def test_brain_presence_misses_when_atom_absent(self):
        self._make(title='unrelated topic',
                    content='something else entirely')
        item = {'question': 'how many total days?',
                 'answer': '15 days', 'haystack_sessions': []}
        r = probe_brain_presence(self.brain, item)
        self.assertFalse(r['found'])

    def test_source_refs_coverage_counts_correctly(self):
        self._make(title='node A', source_refs=['aa11bb22'])
        self._make(title='node B', source_refs=['aa11bb22', 'cc33dd44'])
        self._make(title='node C (no refs)')
        item = {'question': 'x', 'answer': 'y', 'haystack_sessions': []}
        r = probe_source_refs_coverage(self.brain, item)
        self.assertEqual(r['nodes_encoded'], 3)
        self.assertEqual(r['nodes_with_refs'], 2)
        self.assertAlmostEqual(r['coverage_pct'], 66.7, places=0)
        self.assertEqual(r['hex_format_failures'], 0)
        self.assertEqual(r['sparsity_violations_gt5'], 0)

    def test_source_refs_coverage_catches_hex_format_failure(self):
        # Create node with malformed ref using DAL directly (bypasses validator)
        self._make(title='ok node', source_refs=['aa11bb22'])
        # Manually insert a malformed ref so the probe can catch it
        bad_node = self._make(title='bad ref node')
        self.brain.conn.execute(
            'INSERT INTO node_source_refs (node_id, trace_id, position, created_at) '
            'VALUES (?, ?, ?, ?)',
            (bad_node, '<trace-placeholder-not-hex>', 1, '2026-05-25T00:00:00Z'))
        self.brain.conn.commit()
        item = {'question': 'x', 'answer': 'y', 'haystack_sessions': []}
        r = probe_source_refs_coverage(self.brain, item)
        self.assertGreater(r['hex_format_failures'], 0)

    def test_atomization_shape_returns_sane_score(self):
        # Empty haystack means score = 0 / sane
        item = {'question': 'x', 'answer': 'y', 'haystack_sessions': []}
        r = probe_atomization_shape(self.brain, item)
        self.assertIn('score', r)

        self._make(title='node1')
        self._make(title='node2')
        item2 = {'question': 'x', 'answer': 'y',
                  'haystack_sessions': [[{'role': 'user', 'content': '...'},
                                          {'role': 'assistant', 'content': '...'},
                                          {'role': 'user', 'content': '...'},
                                          {'role': 'assistant', 'content': '...'}]]}
        r = probe_atomization_shape(self.brain, item2)
        # 2 nodes / 4 turns = 0.5 (in sweet spot 0.3-0.8)
        self.assertEqual(r['total_nodes'], 2)
        self.assertEqual(r['total_turns'], 4)
        self.assertAlmostEqual(r['nodes_per_turn'], 0.5)
        self.assertEqual(r['score'], 1.0)

    def test_edge_structure_counts_co_anchored(self):
        # Two nodes sharing a ref → co_anchored auto-edge fires (Step 7)
        self._make(title='cohort A', source_refs=['aa11bb22'])
        self._make(title='cohort B', source_refs=['aa11bb22'])
        item = {'question': 'x', 'answer': 'y', 'haystack_sessions': []}
        r = probe_edge_structure(self.brain, item)
        self.assertGreaterEqual(r['co_anchored_pairs'], 1)

    def test_voice_balance_computes_symmetry(self):
        self._make(type='principle', title='balanced node',
                    their_raw_quote='operator said something',
                    my_raw_quote='anchor reframed it')
        self._make(type='principle', title='user-only node',
                    their_raw_quote='only operator voice')
        item = {'question': 'x', 'answer': 'y', 'haystack_sessions': []}
        r = probe_voice_balance(self.brain, item)
        self.assertEqual(r['identity_bearing_total'], 2)
        self.assertEqual(r['identity_bearing_with_user'], 2)
        self.assertEqual(r['identity_bearing_with_anchor'], 1)
        self.assertEqual(r['identity_bearing_symmetry'], 0.5)

    def test_specificity_preservation_detects_dropped_numerics(self):
        self._make(title='partial preservation',
                    content='trip duration was about a week')  # smoothed
        item = {'question': 'x', 'answer': 'y', 'haystack_sessions': [
            [{'role': 'user', 'content': '15 days total — 7 in Hawaii, 8 in NYC'}]
        ]}
        r = probe_specificity_preservation(self.brain, item)
        # 15, 7, 8 are in haystack but not in node content → all dropped
        self.assertLess(r['score'], 1.0)
        self.assertGreater(len(r['dropped_examples']), 0)

    def test_run_all_probes_returns_full_dict(self):
        self._make(title='node1', source_refs=['aa11bb22'])
        item = {'question': 'q', 'answer': 'a',
                 'haystack_sessions': [[{'role': 'user', 'content': 'hi'}]]}
        r = run_all_probes(self.brain, item)
        self.assertEqual(set(r.keys()), set(ALL_PROBES.keys()))
        for probe_name, result in r.items():
            self.assertIsInstance(result, dict,
                                   f'probe {probe_name} returned non-dict')


if __name__ == '__main__':
    unittest.main()
