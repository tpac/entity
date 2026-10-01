"""Contract tests for the eval's read/write path onto brain state.

Two halves of docs/EVAL-BRAIN-PATH-MIGRATION.md step 1:

* the eval's local dispatch routes through `dispatch_command` — the one
  execution chokepoint — so eval brains emit mutation traces exactly like
  production (`node_created` is the run delta the artifacts layer reads);
* `artifacts.dump_nodes`/`dump_edges` source that delta through the brain
  API (`query_traces` + `get_node`), never raw SQL.

The failure class pinned here: an eval brain whose writes bypass the emitter
carries zero `node_created` rows, dump_nodes writes an empty delta, and every
downstream metric computes over [] while reporting real-looking numbers.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tests.brain_test_base import BrainTestBase
from tests.eval_optional import require_eval

require_eval()  # D-8: this module reaches into eval/ inside its test bodies


def _remember_args(title, content, connect_to=None):
    op = {'op': 'remember', 'type': 'fact', 'title': title, 'content': content}
    if connect_to:
        op['connect_to'] = connect_to
    return {'operations': [op], 'encoding_source': 'encoder:test'}


class TestEvalDispatchEmitsMutationTraces(BrainTestBase):
    needs_embedder = False

    def test_local_dispatch_routes_through_chokepoint(self):
        from eval.longmem.replay import _make_local_dispatch
        dispatch = _make_local_dispatch(self.brain)

        res = dispatch('brain_batch', _remember_args('Delta probe', 'x'))
        self.assertTrue(res.get('ok'), res)
        # dispatch_command pops the manifest after emission — a leaked
        # 'mutations' key means the handler was called around the chokepoint.
        self.assertNotIn('mutations', res)

        traces = self.brain.query_traces(ref_type='node_created', hours=None)
        events = traces.get('events', [])
        self.assertEqual(len(events), 1, events)
        self.assertTrue(events[0].get('ref_id'))

    def test_unknown_command_still_errors(self):
        from eval.longmem.replay import _make_local_dispatch
        res = _make_local_dispatch(self.brain)('no_such_command', {})
        self.assertFalse(res.get('ok'))


class TestArtifactsDumpDelta(BrainTestBase):
    needs_embedder = False

    def _dumper(self):
        from eval.longmem.artifacts import EvalArtifactsDumper
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return EvalArtifactsDumper('testrun', 'q1', reports_root=self._tmp.name)

    def test_dump_nodes_is_the_created_delta_not_a_snapshot(self):
        from eval.longmem.replay import _make_local_dispatch
        # A pre-existing node written OUTSIDE dispatch (the seed-pack shape:
        # loaded at init, no trace) must not appear in the delta.
        seed_id = self.brain.remember(
            type='fact', title='Seed node', content='seeded',
            encoding_source='anchor:seed')['id']

        dispatch = _make_local_dispatch(self.brain)
        res = dispatch('brain_batch', _remember_args(
            'Run node', 'created by the run',
            connect_to=[{'title': 'Seed node', 'relation': 'grounds',
                         'why': 'run node grounds itself on the seed pack '
                                'for the edge-dump assertion below'}]))
        self.assertTrue(res.get('ok'), res)

        # A seed the run REVISES (gold absorbed into a pre-existing node)
        # must enter the delta as 'revised' — invisible otherwise.
        res = dispatch('brain_batch', {'operations': [
            {'op': 'revise', 'node_id': seed_id, 'reason': 'test',
             'content': 'seeded plus gold fact'}],
            'encoding_source': 'encoder:test'})
        self.assertTrue(res.get('ok'), res)

        dumper = self._dumper()
        dumper.dump_nodes(self.brain)
        dumper.dump_edges(self.brain)

        node_lines = [json.loads(l) for l in
                      Path(dumper.path('nodes.jsonl')).read_text().splitlines()]
        by_op = {n['delta_op']: n for n in node_lines}
        self.assertEqual(set(by_op), {'created', 'revised'})
        self.assertEqual(by_op['revised']['id'], seed_id)
        self.assertIn('gold fact', by_op['revised']['content'])
        rec = by_op['created']
        self.assertEqual(rec['title'], 'Run node')
        self.assertNotEqual(rec['id'], seed_id)
        # The consumer contract: analyzer._node_text reads kv as a dict;
        # connections stay out of nodes.jsonl (edges.jsonl carries them).
        self.assertIsInstance(rec.get('kv'), dict)
        self.assertNotIn('connections', rec)
        self.assertEqual(rec.get('encoding_source'), 'encoder:test')

        edge_lines = [json.loads(l) for l in
                      Path(dumper.path('edges.jsonl')).read_text().splitlines()]
        rels = [(e['relation'], e['source_title'], e['target_title'])
                for e in edge_lines]
        self.assertIn(('grounds', 'Run node', 'Seed node'), rels)
        # Edges are created-nodes-only: the revised seed contributes none
        # beyond the one the created node owns.
        for e in edge_lines:
            self.assertIn(rec['id'], (e['source_id'], e['target_id']))

    def test_empty_delta_writes_empty_files_not_errors(self):
        dumper = self._dumper()
        dumper.dump_nodes(self.brain)
        dumper.dump_edges(self.brain)
        self.assertEqual(Path(dumper.path('nodes.jsonl')).read_text(), '')
        self.assertEqual(Path(dumper.path('edges.jsonl')).read_text(), '')
        self.assertFalse(Path(str(dumper.path('nodes.jsonl')) + '.error').exists())


if __name__ == '__main__':
    unittest.main()
