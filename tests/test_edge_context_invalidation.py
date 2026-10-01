"""edge_context invalidation — an edge write must invalidate the endpoints'
edge_context vector the way revise() invalidates a kv-sourced one.

edge_context embeds a node's top-5 edge descriptions. Its source lives on
EDGES, its artifact on the NODE, so no single write path owned both ends and
nothing refreshed it: ~40% of production rows embedded a pre-edge snapshot.
The fix: GraphDAL (which every edge write passes through) REPORTS a changed
endpoint via `on_edge_text_changed`; Brain._edge_text_changed runs the one
invalidation path (`invalidate_source_fields`, shared with revise()). The
coverage sweep then rebuilds the row as missing.

Three layers pinned here:
  1. the write side — which writes report, which stay silent (noise-aspect
     relations, short descriptions, renames within the set)
  2. the repair — a stale vector is rebuilt with the new text end to end
  3. the v33 migration — every edge_context row is rebuilt (the text
     definition changed with this version)
"""

import sqlite3
import unittest

from servers import embed_queue
from servers.clock import iso_now
from servers.dal_graph import GraphDAL
from servers.schema import (BRAIN_VERSION, MAIN_MIGRATIONS,
                            _migrate_v33_edge_context_rebuild)
from tests.brain_test_base import BrainTestBase

LONG = 'a sufficiently long edge description'


class TestEdgeWriteInvalidatesEdgeContext(BrainTestBase):
    """Write side: seeded edge_context rows disappear on the writes that change
    the text and survive the ones that cannot."""
    needs_embedder = False

    def setUp(self):
        super().setUp()
        embed_queue._queue.clear()

    def _node(self, title):
        return self.brain.remember(type='fact', title=title, content='c')['id']

    def _seed(self, *node_ids):
        for nid in node_ids:
            for vt in ('edge_context', 'title'):
                self.brain.conn.execute(
                    'INSERT OR REPLACE INTO node_enrichments '
                    '(id, node_id, vector_type, text, embedding, model, created_at) '
                    "VALUES (?, ?, ?, 'old text', x'00', 'test-model', ?)",
                    ('%s__%s' % (nid, vt), nid, vt, iso_now()))
        self.brain.conn.commit()
        embed_queue._queue.clear()

    def _types(self, nid):
        return {r[0] for r in self.brain.conn.execute(
            'SELECT vector_type FROM node_enrichments WHERE node_id = ?',
            (nid,)).fetchall()}

    def _assert_invalidated(self, *node_ids):
        for nid in node_ids:
            types = self._types(nid)
            self.assertNotIn('edge_context', types, 'edge_context must be deleted')
            self.assertIn('title', types, 'only edge_context is edge-sourced')
            self.assertIn(nid, embed_queue._queue, 'endpoint must be re-queued')

    def _assert_untouched(self, *node_ids):
        for nid in node_ids:
            self.assertIn('edge_context', self._types(nid))
            self.assertNotIn(nid, embed_queue._queue)

    def test_connect_typed_described_edge_invalidates_both_endpoints(self):
        a, b = self._node('A'), self._node('B')
        self._seed(a, b)
        self.brain.connect_typed(a, b, 'extends', description=LONG)
        self._assert_invalidated(a, b)

    def test_direct_dal_write_is_covered_without_a_brain_door(self):
        # The absorb / co_anchored writers call self._graph.add_relation
        # directly. The hook lives on the DAL every one of them holds.
        a, b = self._node('A'), self._node('B')
        self._seed(a, b)
        self.brain._graph.add_relation(a, b, 'absorbed_into', description=LONG)
        self._assert_invalidated(a, b)

    def test_policy_tracks_a_live_registry_reload(self):
        # structural_exclusions is rebound on every registry adopt (the S2
        # aspect encoder adopts at runtime). The DAL must read the policy,
        # not a copy taken at construction.
        before = self.brain._graph.edge_context_excluded
        self.assertEqual(before, self.brain.aspects.structural_exclusions)
        self.brain.aspects.structural_exclusions = frozenset(before | {'made_up_noise'})
        self.assertIn('made_up_noise', self.brain._graph.edge_context_excluded)

    def test_policy_is_empty_when_the_registry_failed_to_load(self):
        # A registry that failed to construct leaves brain.aspects unset;
        # excluding nothing keeps producer and backfill filter agreeing.
        aspects = self.brain.aspects
        del self.brain.aspects
        try:
            self.assertEqual(self.brain._graph.edge_context_excluded, frozenset())
            self.assertEqual(self.brain._edge_context_excluded(), frozenset())
        finally:
            self.brain.aspects = aspects

    def test_noise_relations_do_not_invalidate(self):
        # The exclusion set is the noise aspect, resolved through Brain.
        self.assertEqual(self.brain._graph.edge_context_excluded,
                         self.brain.aspects.structural_exclusions)
        self.assertIn('community_member', self.brain._graph.edge_context_excluded)
        self.assertIn('co_anchored', self.brain._graph.edge_context_excluded)
        for rel in ('community_member', 'co_anchored'):
            a, b = self._node('A-' + rel), self._node('B-' + rel)
            self._seed(a, b)
            self.brain.connect_typed(a, b, rel, description=LONG)
            self._assert_untouched(a, b)

    def test_producer_reads_top_k_from_config_and_excludes_noise(self):
        hub = self._node('Hub')
        for i in range(20):
            self.brain.connect_typed(hub, self._node('N%d' % i), 'extends',
                                     description='%s %02d' % (LONG, i), weight=0.5 + i / 100)
        self.brain.connect_typed(hub, self._node('noise'), 'co_anchored',
                                 description=LONG + ' noise', weight=0.99)
        top_k = self.brain.get_interaction_config('edge_context')['top_k']
        self.assertEqual(top_k, 15)
        descs = self.brain._graph.get_edge_descriptions_for(hub, limit=top_k)
        self.assertEqual(len(descs), 15)
        self.assertTrue(all('noise' not in d for d in descs))
        self.assertIn(LONG + ' 19', descs)   # heaviest first
        self.assertNotIn(LONG + ' 00', descs)  # 20 described, 15 kept

    def test_short_description_does_not_invalidate(self):
        a, b = self._node('A'), self._node('B')
        self._seed(a, b)
        self.brain.connect_typed(a, b, 'extends', description='tiny')
        self._assert_untouched(a, b)

    def test_non_feeding_relation_moving_the_edge_weight_invalidates(self):
        # The text is the top-5 by EDGE weight (max over its relations). A
        # community_member row feeds nothing, but raising the edge's weight
        # reorders the described sibling riding the same edge.
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG, weight=0.3)
        self._seed(a, b)
        self.brain.connect_typed(a, b, 'community_member', description='', weight=0.9)
        self._assert_invalidated(a, b)

    def test_non_feeding_relation_on_an_undescribed_edge_does_not_invalidate(self):
        a, b = self._node('A'), self._node('B')
        self._seed(a, b)
        self.brain.connect_typed(a, b, 'community_member', description='', weight=0.9)
        self._assert_untouched(a, b)

    def test_idempotent_reconnect_does_not_invalidate(self):
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG, weight=0.6)
        self._seed(a, b)
        self.brain.connect_typed(a, b, 'extends', description=LONG, weight=0.6)
        self._assert_untouched(a, b)

    def test_revise_edge_description_change_invalidates(self):
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG)
        self._seed(a, b)
        r = self.brain.revise_edge(a, b, 'extends', description=LONG + ' revised')
        self.assertTrue(r['ok'], r)
        self._assert_invalidated(a, b)

    def test_revise_edge_weight_change_invalidates(self):
        # The text is the top-5 by weight — a reorder changes it.
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG, weight=0.5)
        self._seed(a, b)
        r = self.brain.revise_edge(a, b, 'extends', weight=0.9)
        self.assertTrue(r['ok'], r)
        self._assert_invalidated(a, b)

    def test_rename_within_the_text_set_does_not_invalidate(self):
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG)
        self._seed(a, b)
        r = self.brain.revise_edge(a, b, 'extends', new_relation='refines')
        self.assertTrue(r['ok'], r)
        self._assert_untouched(a, b)

    def test_rename_across_the_excluded_set_invalidates(self):
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG)
        self._seed(a, b)
        r = self.brain.revise_edge(a, b, 'extends', new_relation='community_member')
        self.assertTrue(r['ok'], r)
        self._assert_invalidated(a, b)

    def test_disconnect_of_a_described_relation_invalidates(self):
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG)
        self._seed(a, b)
        r = self.brain._graph.remove_relation(a, b, 'extends', archived_by='test')
        self.assertTrue(r['flipped'], r)
        self._assert_invalidated(a, b)

    def test_disconnect_of_a_non_feeding_relation_lowering_edge_weight_invalidates(self):
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG, weight=0.3)
        self.brain.connect_typed(a, b, 'community_member', description='', weight=0.9)
        self._seed(a, b)
        self.brain._graph.remove_relation(a, b, 'community_member', archived_by='test')
        self._assert_invalidated(a, b)

    def test_archive_primitive_does_not_commit_the_flip(self):
        # bulk_archive_relations' contract: callers own the commit. The hook's
        # vector delete must ride that commit, not force one mid-primitive.
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG)
        self._seed(a, b)
        conn = self.brain.conn
        commits = []
        orig = conn.commit
        conn.commit = lambda: (commits.append(1), orig())[1]
        try:
            self.brain._graph.bulk_archive_relations(
                'edge_id = ?', [self.brain._graph.get_edge_id(a, b)], 'test',
                recompute_weight=False)
        finally:
            conn.commit = orig
        self.assertEqual(commits, [], 'the primitive must not commit')
        self.assertFalse(conn.in_batch, 'envelope flag restored')
        self.assertNotIn('edge_context', self._types(a))
        conn.commit()

    def test_node_archive_invalidates_the_live_neighbor(self):
        a, b = self._node('A'), self._node('B')
        self.brain.connect_typed(a, b, 'extends', description=LONG)
        self._seed(a, b)
        self.brain.archive_node(a, archived_by="test", reason="test")
        # b's text lost a's description; a's own vectors go with the archive.
        self.assertNotIn('edge_context', self._types(b))
        self.assertIn(b, embed_queue._queue)

    def test_standalone_graphdal_reports_to_nobody(self):
        a, b = self._node('A'), self._node('B')
        self._seed(a, b)
        GraphDAL(self.brain.conn).add_relation(a, b, 'extends', description=LONG)
        self.brain.conn.commit()
        self.assertIn('edge_context', self._types(a))

    def test_invalidation_failure_is_logged_and_does_not_fail_the_write(self):
        a, b = self._node('A'), self._node('B')
        self._seed(a, b)
        orig = self.brain._vec_dal.delete_for_node

        def boom(*args, **kwargs):
            raise RuntimeError('cache down')
        self.brain._vec_dal.delete_for_node = boom
        try:
            res = self.brain.connect_typed(a, b, 'extends', description=LONG)
        finally:
            self.brain._vec_dal.delete_for_node = orig
        self.assertTrue(res['created'])
        errs = [e for e in self.brain.get_recent_errors(hours=1, limit=100)
                if e.get('source') == 'edge_write_vector_invalidate']
        self.assertTrue(errs, 'a failed invalidation must be logged loudly')


class TestEdgeContextRepairEndToEnd(BrainTestBase):
    """A stale vector is rebuilt with the new text — write invalidates, the
    backfill (the sweep's producer) recreates."""
    needs_embedder = True

    def test_new_edge_re_embeds_edge_context_with_its_description(self):
        a = self.brain.remember(type='fact', title='Hub', content='c')['id']
        b = self.brain.remember(type='fact', title='First', content='c')['id']
        c = self.brain.remember(type='fact', title='Second', content='c')['id']
        self.brain.connect_typed(a, b, 'extends', description=LONG + ' one')
        self.brain.backfill_vectors(batch_size=200)
        row = self.brain.conn.execute(
            "SELECT text FROM node_enrichments WHERE node_id=? AND vector_type='edge_context'",
            (a,)).fetchone()
        self.assertIsNotNone(row)
        self.assertIn('one', row[0])
        self.assertNotIn('two', row[0])

        self.brain.connect_typed(a, c, 'grounds', description=LONG + ' two', weight=0.9)
        gone = self.brain.conn.execute(
            "SELECT 1 FROM node_enrichments WHERE node_id=? AND vector_type='edge_context'",
            (a,)).fetchone()
        self.assertIsNone(gone, 'the edge write must delete the stale row')

        self.brain.backfill_vectors(batch_size=200)
        row = self.brain.conn.execute(
            "SELECT text FROM node_enrichments WHERE node_id=? AND vector_type='edge_context'",
            (a,)).fetchone()
        self.assertIsNotNone(row, 'the backfill must rebuild the row as missing')
        self.assertIn('two', row[0])


class TestMigrateV33(unittest.TestCase):
    """The rebuild: every edge_context row goes (the text definition changed
    with v33); nothing else is touched."""

    def _conn(self):
        conn = sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE node_enrichments (id TEXT PRIMARY KEY, node_id TEXT, '
                     'vector_type TEXT, text TEXT, embedding BLOB, created_at TEXT)')
        for nid, vt in (('a', 'edge_context'), ('b', 'edge_context'),
                        ('a', 'title'), ('c', '_primary')):
            conn.execute("INSERT INTO node_enrichments VALUES (?, ?, ?, 't', x'00', '2026-01-01T00:00:00+00:00')",
                         ('%s__%s' % (nid, vt), nid, vt))
        return conn

    def test_deletes_every_edge_context_row_and_nothing_else(self):
        conn = self._conn()
        _migrate_v33_edge_context_rebuild(conn)
        rows = conn.execute(
            'SELECT node_id, vector_type FROM node_enrichments ORDER BY 1, 2').fetchall()
        self.assertEqual(rows, [('a', 'title'), ('c', '_primary')])

    def test_idempotent(self):
        conn = self._conn()
        _migrate_v33_edge_context_rebuild(conn)
        _migrate_v33_edge_context_rebuild(conn)
        self.assertEqual(conn.execute(
            "SELECT COUNT(*) FROM node_enrichments WHERE vector_type='edge_context'"
        ).fetchone()[0], 0)

    def test_registered_on_the_ladder(self):
        self.assertIn((33, _migrate_v33_edge_context_rebuild), MAIN_MIGRATIONS)
        self.assertGreaterEqual(BRAIN_VERSION, 33)


if __name__ == '__main__':
    unittest.main()
