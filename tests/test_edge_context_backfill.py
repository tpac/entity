"""edge_context backfill — the regression guard for a silent, long-standing
bug: edge_context (a 0.55-weighted recall scoring group) NEVER produced a
node_enrichments row in production. Two halves, both required:

1. find_missing(require_described_edge=True) must queue ONLY nodes that have a
   qualifying described edge — else the ~edgeless nodes (which can never yield
   edge_context text) sit at the front of the last_accessed queue forever and
   starve the edged nodes out of the backfill. (Sibling of the kv-key filter
   in test_find_missing_kv_filter.py — same starvation class, edge-shaped.)

2. backfill_vectors must actually WRITE an edge_context row for an edged node.
   The original bug: the backfill loop had no handler for the _edge_descriptions
   marker, so it built an empty embed-text and skipped every node → 0 rows.
"""

import unittest
from tests.brain_test_base import BrainTestBase
from servers.dal import VectorDAL


class TestFindMissingDescribedEdgeFilter(BrainTestBase):
    """The eligibility filter — pure SQL, no embedder needed."""
    needs_embedder = False

    def _node(self, title):
        return self.brain.remember(type='fact', title=title, content='c')['id']

    def _edge(self, src, tgt, relation, description):
        self.brain._graph.add_relation(src, tgt, relation, description=description)
        self.brain.conn.commit()

    def test_filter_returns_only_nodes_with_a_qualifying_described_edge(self):
        # A—B: real relation + long description → BOTH endpoints eligible
        a, b = self._node('A'), self._node('B')
        self._edge(a, b, 'extends', 'a sufficiently long edge description')
        # C—D: only edge is a noise relation (community_member) → neither eligible
        c, d = self._node('C'), self._node('D')
        self._edge(c, d, 'community_member', 'a sufficiently long edge description')
        # S—T: real relation but description too short (<= min length) → neither
        s, t = self._node('S'), self._node('T')
        self._edge(s, t, 'extends', 'tiny')
        # E: edgeless → never eligible
        e = self._node('E')

        vdal = VectorDAL(self.brain.conn)

        # Baseline: WITHOUT the filter, every node is missing edge_context
        # (it's never created at write time — the node has no edges yet), so all
        # of them come back. This is the starvation state the filter prevents.
        unfiltered = {r['id'] for r in vdal.find_missing('edge_context', limit=50)}
        for nid in (a, b, c, d, s, t, e):
            self.assertIn(nid, unfiltered,
                          'unfiltered must return all edge_context-missing nodes')

        # WITH the filter: only A and B (the qualifying described edge).
        filtered = {r['id'] for r in vdal.find_missing(
            'edge_context', limit=50, require_described_edge=True,
            exclude_relations=self.brain._edge_context_excluded())}
        self.assertIn(a, filtered, 'source of a described edge must qualify')
        self.assertIn(b, filtered, 'target of a described edge must qualify (both directions)')
        for nid, why in [(c, 'noise relation only'), (d, 'noise relation only'),
                         (s, 'description too short'), (t, 'description too short'),
                         (e, 'edgeless')]:
            self.assertNotIn(nid, filtered, 'must exclude node: %s' % why)


class TestEdgeContextBackfillWritesRow(BrainTestBase):
    """End-to-end: an edged node gets a real edge_context vector after backfill.
    This is the assertion that was false in production (0 rows)."""
    needs_embedder = True

    def test_backfill_writes_edge_context_for_edged_node_only(self):
        a = self.brain.remember(type='fact', title='Edged', content='c')['id']
        b = self.brain.remember(type='fact', title='Partner', content='c')['id']
        self.brain._graph.add_relation(
            a, b, 'extends', description='a sufficiently long edge description')
        edgeless = self.brain.remember(type='fact', title='Edgeless', content='c')['id']
        self.brain.conn.commit()

        # Isolate backfill_vectors as the producer: clear any edge_context rows
        # the write/embed-queue path may have created, then prove the backfill
        # recreates one for the edged node and none for the edgeless one.
        self.brain.conn.execute(
            "DELETE FROM node_enrichments WHERE vector_type='edge_context'")
        self.brain.conn.commit()

        self.brain.backfill_vectors(batch_size=200)

        # The edged node now has a non-null edge_context embedding.
        row = self.brain.conn.execute(
            "SELECT embedding, text FROM node_enrichments "
            "WHERE node_id=? AND vector_type='edge_context'", (a,)).fetchone()
        self.assertIsNotNone(row, 'edged node must get an edge_context row')
        self.assertIsNotNone(row[0], 'edge_context embedding must be non-null')
        # The stored text is composed from the node's edge descriptions.
        self.assertIn('a sufficiently long edge description', row[1])

        # The edgeless node must NOT get an edge_context row.
        none_row = self.brain.conn.execute(
            "SELECT 1 FROM node_enrichments WHERE node_id=? AND vector_type='edge_context'",
            (edgeless,)).fetchone()
        self.assertIsNone(none_row, 'edgeless node must not get an edge_context row')


class TestDeadHandlerTrip(BrainTestBase):
    """The loud-by-default detector: a configured embedding group that returns
    eligible candidates but stores zero vectors must scream. This is the alarm
    that was MISSING when edge_context silently held 0 rows for its whole
    history — nothing watched 'configured group produces nothing'."""
    needs_embedder = True

    def _make_edged(self, n):
        for i in range(n):
            a = self.brain.remember(type='fact', title='DH-A%d' % i, content='c')['id']
            b = self.brain.remember(type='fact', title='DH-B%d' % i, content='c')['id']
            self.brain._graph.add_relation(
                a, b, 'extends', description='a sufficiently long edge description %d' % i)
        self.brain.conn.commit()

    def _edge_context_dead_errors(self):
        return [e for e in self.brain.get_recent_errors(hours=1, limit=100)
                if e.get('source') == 'embedding_handler_dead'
                and 'edge_context' in (e.get('error', '') + e.get('context', ''))]

    def test_dead_handler_trips_alert(self):
        self._make_edged(4)  # + seed-pack edged nodes → well over the 5-candidate floor
        self.brain.conn.execute("DELETE FROM node_enrichments WHERE vector_type='edge_context'")
        self.brain.conn.commit()
        # Break the handler: eligible candidates remain, but it yields no text.
        orig = self.brain._graph.get_edge_descriptions_for
        self.brain._graph.get_edge_descriptions_for = lambda *a, **k: []
        try:
            self.brain.backfill_vectors(batch_size=50)
        finally:
            self.brain._graph.get_edge_descriptions_for = orig
        self.assertTrue(self._edge_context_dead_errors(),
                        'a dead edge_context handler must log embedding_handler_dead')

    def test_healthy_handler_does_not_trip(self):
        self._make_edged(4)
        self.brain.conn.execute("DELETE FROM node_enrichments WHERE vector_type='edge_context'")
        self.brain.conn.commit()
        # Real handler — produces text, stores vectors → no alarm.
        self.brain.backfill_vectors(batch_size=50)
        self.assertFalse(self._edge_context_dead_errors(),
                         'a healthy handler must not false-trip the dead-handler alarm')


if __name__ == '__main__':
    unittest.main()
