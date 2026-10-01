"""Edge embedding stability + async write-path coverage (schema v26).

Edge embedding is ASYNC (2026-06): write paths only INVALIDATE — add_relation /
rename_relation NULL the embedding + enqueue_edge — and the embed_queue worker
re-embeds via Brain.backfill_edge_embeddings, the same mechanism that embeds
nodes and traces. No write function embeds inline; nothing outside the brain
layer (e.g. S2 reclassify) touches embedding. These tests model the worker by
calling backfill_edge_embeddings explicitly.

Properties locked:

1. **Async populate.** A connect on a non-excluded relation leaves embedding
   NULL at write; backfill_edge_embeddings then populates embedding + model.

2. **Description change invalidates + re-embeds.** A desc change NULLs the
   stored embedding (so the worker re-embeds); backfill produces a new vector.

3. **Stability across partner-title revisions.** Composed edge text is INTRINSIC
   (`[relation] description`, no partner title), so revising a
   partner node's title does NOT invalidate the edge embedding.

4. **Archive NULLs; revive re-embeds.**
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.brain_test_base import BrainTestBase  # noqa: E402


class TestEdgeEmbedding(BrainTestBase):
    needs_embedder = True

    def _create_pair(self, title_a='Source node', title_b='Target node'):
        a = self.brain.remember(type='fact', title=title_a, content='A')
        b = self.brain.remember(type='fact', title=title_b, content='B')
        return a['id'], b['id']

    def _edge_id(self, source_id, target_id):
        """Edge id for the pair (edges are single-direction in v22+)."""
        row = self.brain.conn.execute(
            'SELECT edge_id FROM edges '
            'WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (source_id, target_id, target_id, source_id)).fetchone()
        return row[0] if row else None

    def _read_embedding(self, source_id, target_id, relation):
        """Return (blob, model) for the (source, target, relation) edge."""
        eid = self._edge_id(source_id, target_id)
        if not eid:
            return None, None
        row = self.brain.conn.execute(
            'SELECT embedding, embedding_model FROM edge_relations '
            'WHERE edge_id = ? AND relation = ?', (eid, relation)).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def _backfill(self, source_id, target_id):
        """Model the embed_queue worker: re-embed this edge's NULL relations."""
        eid = self._edge_id(source_id, target_id)
        if eid:
            self.brain.backfill_edge_embeddings([eid])

    def test_async_backfill_populates_embedding(self):
        """A connect leaves embedding NULL at write (async); backfill — the
        worker step — stores embedding + model. If the worker never ran, recall
        falls back to live compose, so NULL-at-write is correct, not a bug."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A extends B')
        self.assertIsNone(self._read_embedding(a, b, 'extends')[0],
                          'embedding is NOT written synchronously at connect (async)')
        self._backfill(a, b)
        blob, model = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob, 'backfill should store the embedding')
        self.assertGreater(len(blob), 0)
        self.assertTrue(model, 'embedding_model should be populated by backfill')

    def test_select_edges_reads_stored_embedding_from_get_node(self):
        """get_node's connections carry the top claim's edge_id, so
        select_edges reads the stored vector — it must not re-embed the text.
        Without the id every hook turn re-embedded ~150 edge texts (94% of the
        candidates phase, measured 2026-09-23)."""
        from unittest import mock
        import numpy as np
        from servers.embedder import embed_query
        from servers.scales.s1 import surface_contract as sc
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A extends B')
        self._backfill(a, b)
        conns = self.brain.get_node(a)['connections']
        self.assertEqual(conns[0]['edge_id'], self._edge_id(a, b))
        query = np.frombuffer(embed_query('what does A extend'), dtype=np.float32)
        with mock.patch.object(sc, '_desc_vecs_batched',
                               side_effect=AssertionError('live re-embed')):
            picked = sc.select_edges(conns, query, limit=3,
                                     brain_conn=self.brain.conn, brain=self.brain)
        self.assertEqual([c['id'] for c in picked], [b])

    def test_select_edges_reads_stored_embeddings_for_claims_both_ways(self):
        """A pair with claims pointing both ways gives the owner one entry per
        direction; each carries its top claim's edge_id, and none re-embeds."""
        from unittest import mock
        import numpy as np
        from servers.embedder import embed_query
        from servers.scales.s1 import surface_contract as sc
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A extends B')
        self.brain.connect_typed(b, a, relation='supports', description='B supports A')
        self._backfill(a, b)
        conns = self.brain.get_node(a)['connections']
        self.assertEqual(sorted(c['direction'] for c in conns),
                         ['incoming', 'outgoing'])
        for c in conns:
            self.assertEqual(c['edge_id'], c['relations'][0]['edge_id'])
            self.assertEqual(c['edge_id'], self._edge_id(a, b))
        query = np.frombuffer(embed_query('how are A and B linked'), dtype=np.float32)
        with mock.patch.object(sc, '_desc_vecs_batched',
                               side_effect=AssertionError('live re-embed')):
            picked = sc.select_edges(conns, query, limit=3,
                                     brain_conn=self.brain.conn, brain=self.brain)
        self.assertEqual(len(picked), 2)

    def test_edge_key_without_edge_id_is_logged(self):
        """A connection that lost its edge_id still ranks (live re-embed) but is
        reported — the silent version of this cost every hook turn for months."""
        from unittest import mock
        import numpy as np
        from servers.embedder import embed_query
        from servers.scales.s1 import surface_contract as sc
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A extends B')
        self._backfill(a, b)
        conns = self.brain.get_node(a)['connections']
        for c in conns:
            c['edge_id'] = None
        query = np.frombuffer(embed_query('what does A extend'), dtype=np.float32)
        with mock.patch.object(self.brain, '_log_error') as log:
            picked = sc.select_edges(conns, query, limit=3,
                                     brain_conn=self.brain.conn, brain=self.brain)
        self.assertEqual([c['id'] for c in picked], [b])
        self.assertIn('edge_embedding_key_missing',
                      [call.args[0] for call in log.call_args_list])

    def test_partner_title_revision_does_not_stale_embedding(self):
        """Revising a partner node's title must not change the edge embedding —
        composed text is intrinsic to (relation, description), no partner
        title. A node revise does not invalidate the edge, so a re-run of the
        worker is a no-op (embedding already non-NULL)."""
        a, b = self._create_pair(title_a='Original A', title_b='Original B')
        self.brain.connect_typed(a, b, relation='extends', description='A extends B')
        self._backfill(a, b)
        blob_before, _ = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_before)

        self.brain.revise(b, title='Renamed B')
        self._backfill(a, b)  # node revise does not enqueue the edge; no-op here
        blob_after, _ = self._read_embedding(a, b, 'extends')
        self.assertEqual(blob_before, blob_after,
                         'edge embedding must be stable across partner title '
                         'revisions — it is intrinsic to (relation, description)')

    def test_description_change_invalidates_and_reembeds(self):
        """A description change NULLs the stored embedding (invalidate), and
        backfill produces a new vector for the new text."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='first description')
        self._backfill(a, b)
        blob_v1, _ = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_v1)

        self.brain.connect_typed(a, b, relation='extends',
                                 description='entirely different rationale')
        # Description change invalidated the embedding — NULL until re-embed.
        self.assertIsNone(self._read_embedding(a, b, 'extends')[0],
                          'desc change must NULL the embedding (invalidate)')
        self._backfill(a, b)
        blob_v2, _ = self._read_embedding(a, b, 'extends')
        self.assertNotEqual(blob_v1, blob_v2,
                            'description change should produce a new vector')

    def test_archive_clears_embedding(self):
        """Archiving a relation NULLs its embedding (parallel to node archive's
        DELETE FROM node_enrichments). Archived rows are never read."""
        from servers.dal_graph import GraphDAL
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A extends B')
        self._backfill(a, b)
        blob_before, model_before = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_before, 'embedding should be populated by backfill')
        self.assertIsNotNone(model_before)

        gdal = GraphDAL(self.brain.conn)
        gdal.remove_relation(a, b, 'extends', archived_by='test')

        blob_after, model_after = self._read_embedding(a, b, 'extends')
        self.assertIsNone(blob_after,
                          'archived edge_relations.embedding must be NULL')
        self.assertIsNone(model_after,
                          'archived embedding_model must be NULL too')

        # Soft-archive (row still exists), not hard-delete.
        edge_row = self.brain.conn.execute(
            'SELECT er.archived FROM edges e '
            'JOIN edge_relations er ON er.edge_id = e.edge_id '
            'WHERE er.relation = ? '
            'AND ((e.source_id = ? AND e.target_id = ?) '
            '     OR (e.source_id = ? AND e.target_id = ?))',
            ('extends', a, b, b, a)).fetchone()
        self.assertIsNotNone(edge_row, 'archived row should still exist (soft-archive)')
        self.assertEqual(edge_row[0], 1, 'row should be archived=1')

    def test_revive_re_embeds_after_archive(self):
        """A revived edge (add_relation Branch 3, created=True) starts NULL and
        backfill re-embeds it — archived→revived edges must not stay NULL."""
        from servers.dal_graph import GraphDAL
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='original description')
        self._backfill(a, b)
        blob_v1, _ = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_v1)

        gdal = GraphDAL(self.brain.conn)
        gdal.remove_relation(a, b, 'extends', archived_by='test')
        self.assertIsNone(self._read_embedding(a, b, 'extends')[0])

        # Revive with a new description (created=True via Branch 3).
        self.brain.connect_typed(a, b, relation='extends', description='revived description')
        self._backfill(a, b)
        blob_v2, model_v2 = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_v2, 'revived edge must get a fresh embedding')
        self.assertIsNotNone(model_v2)
        self.assertNotEqual(blob_v1, blob_v2,
                            'revived edge with new description → new vector')

    def test_revive_drops_a_vector_left_on_the_archived_row(self):
        """Archived rows from before every archive path NULLed still hold their
        old vector. A revive is a fresh row: it must drop that vector, or the
        worker (which skips non-NULL current-model rows) keeps the OLD
        description's geometry on the new description for good."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='original description')
        self._backfill(a, b)
        blob_v1, _ = self._read_embedding(a, b, 'extends')
        eid = self._edge_id(a, b)
        # An archive that kept its vector — the historical shape.
        self.brain.conn.execute(
            "UPDATE edge_relations SET archived = 1 "
            "WHERE edge_id = ? AND relation = 'extends'", (eid,))
        self.brain.conn.commit()

        self.brain.connect_typed(a, b, relation='extends', description='revived description')
        self.assertIsNone(self._read_embedding(a, b, 'extends')[0],
                          'revive must drop the vector the archived row kept')
        self._backfill(a, b)
        blob_v2, _ = self._read_embedding(a, b, 'extends')
        self.assertIsNotNone(blob_v2)
        self.assertNotEqual(blob_v1, blob_v2,
                            'revived edge carries its new description\'s vector')

    def test_archive_during_compute_leaves_no_vector(self):
        """The embed write only lands on a live row: an archive between the
        worker's SELECT and its UPDATE must leave the archived row NULL."""
        from servers import embedder
        from servers.dal_graph import GraphDAL
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A extends B')
        eid = self._edge_id(a, b)
        orig = embedder.embed_batch

        def archiving_embed(texts, **kw):
            GraphDAL(self.brain.conn).remove_relation(a, b, 'extends', archived_by='test')
            return orig(texts, **kw)

        embedder.embed_batch = archiving_embed
        try:
            self.brain.backfill_edge_embeddings([eid])
        finally:
            embedder.embed_batch = orig
        self.assertIsNone(self._read_embedding(a, b, 'extends')[0],
                          'an archived row must not receive the worker\'s blob')

    def test_concurrent_desc_change_during_compute_does_not_clobber(self):
        """Write is description-guarded: if a concurrent revise changes the
        description during the compute window, the worker must NOT write its now-
        stale blob (clobbering the fresh-NULL row, which would then never
        re-embed). Simulate the race by mutating the row inside embed_batch,
        which runs between backfill's SELECT and its guarded UPDATE."""
        from servers import embedder
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A')
        eid = self._edge_id(a, b)
        orig = embedder.embed_batch

        def racing_embed(texts, **kw):
            # Foreground revise lands mid-compute: change desc + re-NULL.
            self.brain.conn.execute(
                "UPDATE edge_relations SET description = 'B', embedding = NULL "
                "WHERE edge_id = ? AND relation = 'extends'", (eid,))
            self.brain.conn.commit()
            return orig(texts, **kw)

        embedder.embed_batch = racing_embed
        try:
            self.brain.backfill_edge_embeddings([eid])
        finally:
            embedder.embed_batch = orig
        self.assertIsNone(self._read_embedding(a, b, 'extends')[0],
                          'description-guard must reject the stale-desc write')
        # A clean re-drain embeds the current description.
        self.brain.backfill_edge_embeddings([eid])
        self.assertIsNotNone(self._read_embedding(a, b, 'extends')[0],
                             'next drain embeds the changed description')

    def test_stale_model_rows_are_reembedded(self):
        """A model swap leaves rows whose embedding_model the read path can't use
        (it filters embedding_model = active). backfill must re-embed them —
        treat stale-model as missing — not skip because embedding is non-NULL."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A')
        self._backfill(a, b)
        eid = self._edge_id(a, b)
        _, model1 = self._read_embedding(a, b, 'extends')
        self.assertTrue(model1)
        self.brain.conn.execute(
            "UPDATE edge_relations SET embedding_model = 'OLD-MODEL' "
            "WHERE edge_id = ? AND relation = 'extends'", (eid,))
        self.brain.conn.commit()
        n = self.brain.backfill_edge_embeddings([eid])
        self.assertGreaterEqual(n, 1, 'stale-model row must be re-embedded')
        _, model2 = self._read_embedding(a, b, 'extends')
        self.assertEqual(model2, model1, 'embedding_model refreshed to current')

    def test_embedder_not_ready_reenqueues_not_drops(self):
        """If the embedder isn't ready (e.g. boot), backfill re-enqueues the
        edges instead of dropping them — rows stay NULL (live-fallback) until a
        later drain embeds them."""
        from servers import embedder, embed_queue
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', description='A')
        eid = self._edge_id(a, b)
        embed_queue._edge_queue.discard(eid)
        orig_ready = embedder.is_ready
        embedder.is_ready = lambda: False
        try:
            n = self.brain.backfill_edge_embeddings([eid])
        finally:
            embedder.is_ready = orig_ready
        self.assertEqual(n, 0, 'no embeds while embedder not ready')
        self.assertIn(eid, embed_queue._edge_queue, 're-enqueued, not dropped')
        embed_queue._edge_queue.discard(eid)


if __name__ == '__main__':
    unittest.main()
