"""Tests for multi-relation edge model (v22: edge_id + single-direction).

The edge model separates physical edges (one per pair, direction in source/target)
from semantic relations (multiple per edge via edge_id). Two nodes connect once
but carry multiple relation/description pairs.

Test IDs map to the plan: T1=migration, T2=multi-relation, T3=encoding,
T4=query, T9=cascade, T10=compat.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


def _get_relations_for_pair(conn, node_a, node_b):
    """Helper: get all relations between two nodes (either direction)."""
    row = conn.execute(
        'SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) '
        'OR (source_id = ? AND target_id = ?)',
        (node_a, node_b, node_b, node_a)
    ).fetchone()
    if not row:
        return []
    return conn.execute(
        'SELECT relation, description, weight FROM edge_relations WHERE edge_id = ? ORDER BY weight DESC',
        (row[0],)
    ).fetchall()


class T1_Migration(BrainTestBase):
    """Edge table has new v22 structure."""

    needs_embedder = False

    def test_edge_relations_table_exists(self):
        tables = [r[0] for r in self.brain.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        self.assertIn('edge_relations', tables)

    def test_edges_table_has_edge_id(self):
        cols = {r[1] for r in self.brain.conn.execute("PRAGMA table_info(edges)").fetchall()}
        self.assertIn('edge_id', cols)
        self.assertNotIn('relation', cols)
        self.assertNotIn('edge_type', cols)
        self.assertNotIn('description', cols)
        self.assertNotIn('stability', cols)

    def test_edge_relations_has_edge_id(self):
        cols = {r[1] for r in self.brain.conn.execute("PRAGMA table_info(edge_relations)").fetchall()}
        self.assertIn('edge_id', cols)
        self.assertIn('encoding_source', cols)
        self.assertNotIn('source_id', cols)
        self.assertNotIn('target_id', cols)

    def test_single_direction_storage(self):
        """connect(A, B) creates ONE edge row, not two mirrors."""
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']
        self.brain.connect(a, b, 'extends', 0.8)

        # Only one row in edges
        count = self.brain.conn.execute(
            'SELECT COUNT(*) FROM edges WHERE '
            '(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)',
            (a, b, b, a)
        ).fetchone()[0]
        self.assertEqual(count, 1)


class T2_MultiRelation(BrainTestBase):
    """New edges accumulate multiple relations."""

    needs_embedder = False

    def _create_pair(self):
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']
        return a, b

    def test_two_relations_one_edge(self):
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', weight=0.8,
                                 description='A extends B')
        self.brain.connect_typed(a, b, relation='corrects', weight=0.7,
                                 description='A corrects B')

        # One physical edge
        edge_count = self.brain.conn.execute(
            'SELECT COUNT(*) FROM edges WHERE '
            '(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)',
            (a, b, b, a)).fetchone()[0]
        self.assertEqual(edge_count, 1)

        # Two relations
        rels = _get_relations_for_pair(self.brain.conn, a, b)
        self.assertEqual(len(rels), 2)

    def test_aggregate_weight_is_max(self):
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', weight=0.8,
                                 description='A extends B')
        self.brain.connect_typed(a, b, relation='corrects', weight=0.5,
                                 description='A corrects B')

        edge_weight = self.brain.conn.execute(
            'SELECT weight FROM edges WHERE (source_id = ? AND target_id = ?) '
            'OR (source_id = ? AND target_id = ?)',
            (a, b, b, a)).fetchone()[0]
        self.assertAlmostEqual(edge_weight, 0.8, places=1)

    def test_same_relation_idempotent(self):
        """Stage 1B (Option α): re-connecting the same relation is idempotent.
        Repeated connect does NOT auto-strengthen weight. A later description
        replaces the earlier one (field-preserving upsert), and the pair
        stays a single row.
        """
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', weight=0.5,
                                 description='first')
        self.brain.connect_typed(a, b, relation='extends', weight=0.5,
                                 description='second')

        rels = _get_relations_for_pair(self.brain.conn, a, b)
        extends_rels = [r for r in rels if r[0] == 'extends']
        self.assertEqual(len(extends_rels), 1)            # one row, not two
        self.assertEqual(extends_rels[0][2], 0.5)         # weight unchanged (no auto-strengthen)
        self.assertEqual(extends_rels[0][1], 'second')    # later description replaces earlier

    def test_direction_preserved(self):
        """Source of connect call should be source_id in edges table."""
        a, b = self._create_pair()
        self.brain.connect_typed(a, b, relation='extends', weight=0.8,
                                 description='A extends B')

        edge = self.brain.conn.execute(
            'SELECT source_id, target_id FROM edges WHERE '
            '(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)',
            (a, b, b, a)).fetchone()
        # a should be source (it was passed first to connect_typed)
        self.assertEqual(edge[0], a)
        self.assertEqual(edge[1], b)


class T3_EncodingOutput(BrainTestBase):
    """Encoder connect_to creates typed relations."""

    needs_embedder = False

    def test_connect_to_old_format(self):
        a = self.brain.remember(type='decision', title='Target Node', content='target')['id']
        result = self.brain.remember_batch(
            nodes=[{'type': 'decision', 'title': 'Source Node', 'content': 'source'}],
            connect_to=[{'title': 'Target Node', 'why': 'extends the pattern', 'relation': 'extends'}]
        )
        created_id = result['results'][0]['id']
        rels = _get_relations_for_pair(self.brain.conn, created_id, a)
        self.assertTrue(any(r[0] == 'extends' for r in rels))

    def test_connect_to_new_format_multi_relations(self):
        a = self.brain.remember(type='decision', title='Target Node', content='target')['id']
        result = self.brain.remember_batch(
            nodes=[{'type': 'decision', 'title': 'Source Node', 'content': 'source'}],
            connect_to=[{
                'title': 'Target Node',
                'relations': [
                    {'relation': 'extends', 'why': 'builds on the pattern'},
                    {'relation': 'depends_on', 'why': 'requires this to function'},
                ]
            }]
        )
        created_id = result['results'][0]['id']
        rels = _get_relations_for_pair(self.brain.conn, created_id, a)
        rel_dict = {r[0]: r[1] for r in rels}

        self.assertIn('extends', rel_dict)
        self.assertIn('depends_on', rel_dict)
        self.assertEqual(rel_dict['extends'], 'builds on the pattern')
        self.assertEqual(rel_dict['depends_on'], 'requires this to function')


class T4_EdgeQuery(BrainTestBase):
    """Readers get relations via edge_id."""

    needs_embedder = False

    def test_get_relations_returns_list(self):
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']
        self.brain.connect_typed(a, b, relation='extends', weight=0.8,
                                 description='A extends B')
        self.brain.connect_typed(a, b, relation='corrects', weight=0.7,
                                 description='A corrects B')

        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        edge_id = dal.get_edge_id(a, b)
        self.assertIsNotNone(edge_id)
        rels = dal.get_relations(edge_id)
        self.assertEqual(len(rels), 2)
        rel_types = {r['relation'] for r in rels}
        self.assertEqual(rel_types, {'extends', 'corrects'})

    def test_edge_id_and_relations(self):
        # get_edge was removed in the DAL Phase-A cleanup; the live path is
        # get_edge_id (the pair → edge_id lookup) + get_relations (relations
        # attached to that edge).
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']
        self.brain.connect_typed(a, b, relation='extends', weight=0.8,
                                 description='desc')

        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        edge_id = dal.get_edge_id(a, b)
        self.assertIsNotNone(edge_id)
        rels = dal.get_relations(edge_id)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]['relation'], 'extends')

    def test_edge_id_either_direction(self):
        """get_edge_id finds the same edge regardless of query direction."""
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']
        self.brain.connect_typed(a, b, relation='extends', weight=0.8)

        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        eid_ab = dal.get_edge_id(a, b)
        eid_ba = dal.get_edge_id(b, a)
        self.assertIsNotNone(eid_ab)
        self.assertIsNotNone(eid_ba)
        # Single-direction storage: either query order resolves to one edge.
        self.assertEqual(eid_ab, eid_ba)


class T6_MultiRelationPreservation(BrainTestBase):
    """Adding a second relation doesn't overwrite the first."""

    needs_embedder = False

    def test_second_relation_adds_not_overwrites(self):
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']

        self.brain.connect_typed(a, b, relation='extends', weight=0.8,
                                 description='A extends B')
        self.brain.connect(a, b, relation='refines', weight=0.3)

        rels = _get_relations_for_pair(self.brain.conn, a, b)
        rel_dict = {r[0]: r[1] for r in rels}
        self.assertIn('extends', rel_dict)
        self.assertIn('refines', rel_dict)
        self.assertEqual(rel_dict['extends'], 'A extends B')


class T9_Cascade(BrainTestBase):
    """Delete/archive cascades to edge_relations."""

    needs_embedder = False

    def test_delete_node_removes_relations(self):
        """delete_node_edges() soft-archives all edge_relations for a node.

        v25 contract change (see GraphDAL.delete_node_edges docstring):
        was a hard DELETE, now sets archived=1 on the edge_relations and
        leaves the edges aggregate row intact. The asymmetry with node
        archive previously destroyed edge provenance forever; soft-archive
        preserves it.

        So the new contract is:
          - edges row: still present (the physical pair survives)
          - edge_relations: archived=1, still queryable for provenance
          - active relation count for this edge: 0
        """
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']
        self.brain.connect_typed(a, b, relation='extends', weight=0.8,
                                 description='A extends B')

        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        edge_id = dal.get_edge_id(a, b)
        self.assertIsNotNone(edge_id)

        dal.delete_node_edges([a])

        # No ACTIVE relations for this edge (soft-archive contract).
        active_rels = self.brain.conn.execute(
            'SELECT COUNT(*) FROM edge_relations '
            'WHERE edge_id = ? AND archived = 0',
            (edge_id,)).fetchone()[0]
        self.assertEqual(active_rels, 0,
                         "Active edge_relations should be 0 after delete_node_edges")

        # The provenance row should still exist (archived=1).
        archived_rels = self.brain.conn.execute(
            'SELECT COUNT(*) FROM edge_relations '
            'WHERE edge_id = ? AND archived = 1',
            (edge_id,)).fetchone()[0]
        self.assertGreaterEqual(archived_rels, 1,
                                "Soft-archived relation row should survive for provenance")


class T10_BackwardCompat(BrainTestBase):
    """Existing callers still work with single-relation calls."""

    needs_embedder = False

    def test_connect_single_relation(self):
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']

        self.brain.connect(a, b, 'related', 0.5)

        # Edge exists
        edge = self.brain.conn.execute(
            'SELECT edge_id FROM edges WHERE '
            '(source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)',
            (a, b, b, a)).fetchone()
        self.assertIsNotNone(edge)

        # One relation exists
        rels = self.brain.conn.execute(
            'SELECT relation FROM edge_relations WHERE edge_id = ?',
            (edge[0],)).fetchall()
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0][0], 'related')

    def test_encoding_source_tracked(self):
        """Relations should carry encoding_source."""
        a = self.brain.remember(type='decision', title='Node A', content='A')['id']
        b = self.brain.remember(type='decision', title='Node B', content='B')['id']

        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        dal.add_relation(a, b, 'extends', 'test', 0.8, encoding_source='encoder:sonnet')

        edge_id = dal.get_edge_id(a, b)
        rels = dal.get_relations(edge_id)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]['encoding_source'], 'encoder:sonnet')


class T5_DanglingArchiveTimestampFormat(BrainTestBase):
    """2026-06-12 — archive_dangling_edges must stamp archived_at in the
    brain's ISO-T format (clock.iso_now), like every other edge_relations
    writer. It was the lone unix-ms writer into the TEXT column, which
    broke lexicographic time reads and rendered as 1970 epoch dates."""

    needs_embedder = False

    def test_archived_at_is_iso(self):
        from servers.dal_graph import GraphDAL

        a = self.brain.remember(type='test', title='dangle_a', content='c',
                                encoding_source='anchor:test')
        b = self.brain.remember(type='test', title='dangle_b', content='c',
                                encoding_source='anchor:test')
        dal = GraphDAL(self.brain.conn)
        dal.add_relation(a['id'], b['id'], 'relates_to',
                         encoding_source='anchor:test')

        # Simulate the leak the restorer exists for: node archived without
        # its edges (bypassing archive_node's own edge sweep).
        self.brain.conn.execute(
            'UPDATE nodes SET archived = 1 WHERE id = ?', (b['id'],))

        r = dal.archive_dangling_edges('anchor:test')
        self.assertGreaterEqual(r['archived'], 1, 'restorer archived nothing')

        row = self.brain.conn.execute(
            "SELECT er.archived, er.archived_at FROM edge_relations er "
            "JOIN edges e ON e.edge_id = er.edge_id "
            "WHERE e.source_id = ? AND e.target_id = ? "
            "  AND er.relation = 'relates_to'",
            (a['id'], b['id'])).fetchone()
        self.assertIsNotNone(row)
        archived, archived_at = row
        self.assertEqual(archived, 1)
        # ISO-T, not unix-ms: starts with a year and carries the T separator.
        self.assertIsInstance(archived_at, str)
        self.assertTrue(archived_at.startswith('20'),
                        'archived_at not ISO: %r' % archived_at)
        self.assertIn('T', archived_at,
                      'archived_at missing ISO T separator: %r' % archived_at)


class T11_SharedFlipPrimitive(BrainTestBase):
    """Step 9 — bulk_archive_relations is THE one soft-archive flip; the four
    archiving writers route through it with explicit per-caller policy, and
    the sweeps return observed [edge_id, relation] flips for trace rows."""

    needs_embedder = False

    def _pair(self, ta, tb, relation, **kw):
        from servers.dal_graph import GraphDAL
        a = self.brain.remember(type='test', title=ta, content='c',
                                encoding_source='anchor:test')['id']
        b = self.brain.remember(type='test', title=tb, content='c',
                                encoding_source='anchor:test')['id']
        dal = GraphDAL(self.brain.conn)
        res = dal.add_relation(a, b, relation,
                               encoding_source='anchor:test', **kw)
        return dal, a, b, res['edge_id']

    def test_dangling_sweep_returns_flipped_pairs_and_commits(self):
        dal, a, b, edge_id = self._pair('fp_a', 'fp_b', 'relates_to')
        self.brain.conn.execute(
            'UPDATE nodes SET archived = 1 WHERE id = ?', (b,))
        self.brain.conn.commit()

        r = dal.archive_dangling_edges('anchor:test')
        self.assertEqual(r['archived'], len(r['edge_relations']))
        self.assertIn([edge_id, 'relates_to'], r['edge_relations'])
        # The sweep owns its durability now — no open transaction rides
        # after it (any trace emitted after it used to be orphanable).
        self.assertFalse(self.brain.conn.in_transaction,
                         'archive_dangling_edges left its flip uncommitted')

    def test_dangling_sweep_pairs_exclude_exempt_relations(self):
        dal, a, b, edge_id = self._pair('ex_a', 'ex_b', 'absorbed_into')
        dal.add_relation(a, b, 'relates_to', encoding_source='anchor:test')
        self.brain.conn.execute(
            'UPDATE nodes SET archived = 1 WHERE id = ?', (b,))
        self.brain.conn.commit()

        r = dal.archive_dangling_edges(
            'anchor:test', exempt_relations=['absorbed_into'])
        pairs = r['edge_relations']
        self.assertIn([edge_id, 'relates_to'], pairs)
        self.assertNotIn([edge_id, 'absorbed_into'], pairs,
                         'the exempt redirect must never be claimed flipped')

    def test_primitive_select_matches_update(self):
        """The observed-truth property at the primitive level: the returned
        pairs are exactly the rows flipped, already-archived rows excluded."""
        dal, a, b, edge_id = self._pair('sm_a', 'sm_b', 'relates_to')
        dal.add_relation(a, b, 'extends', encoding_source='anchor:test')
        # Pre-archive one of the two relations.
        dal.remove_relation(a, b, 'extends', archived_by='anchor:test')

        flipped = dal.bulk_archive_relations(
            'edge_id = ?', [edge_id], 'anchor:test',
            recompute_weight=False)
        self.assertEqual(flipped, [[edge_id, 'relates_to']],
                         'already-archived rows must not be claimed')


if __name__ == '__main__':
    unittest.main()
