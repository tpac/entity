"""Supersession handling in S2 consolidation — heal + render contract.

Two halves of the survivor-ladder age-bias fix (journal audit 2026-07-25):

1. Healing pass (decoder): a live handoff with a live handoff successor
   (`supersedes` edge) is archived before clustering — so consolidation can
   never again absorb a session opener into its stale predecessor. Keyed on
   the edge AND both endpoint types, fail-safe for untyped openers.
   Superseded KNOWLEDGE (decision/fact/...) deliberately stays live — its
   staleness is annotated at read time by correction_enrich, and its
   reasoning keeps recall value.

2. Render contract (encoder): intra-cluster edges appear in the cluster
   text WITH direction (each claim as written: source → relation → target). They used to be dropped
   by the external-edges-only filter and collapsed to a directionless
   CORRECTION_EDGE flag — which is exactly why the encoder merged
   supersession pairs backwards (it couldn't see which node supersedes
   which). A future change that re-filters intra-member edges out of
   _format_clusters must fail here.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class SupersessionBase(BrainTestBase):
    needs_embedder = False

    def _node(self, title, type='handoff', locked=False, **kw):
        r = self.brain.remember(type=type, title=title, content='content of %s' % title,
                                locked=locked, encoding_source='anchor', **kw)
        return r['id']

    def _archived(self, nid):
        return self.brain.conn.execute(
            'SELECT archived FROM nodes WHERE id = ?', (nid,)).fetchone()[0]

    def _decoder(self):
        from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
        return ConsolidationDecoder(self.brain)


class TestHealSupersededHandoffs(SupersessionBase):
    """Decoder healing pass: predecessor openers are retired pre-clustering."""

    def test_superseded_handoff_archived_successor_live(self):
        old = self._node('opener 07-21')
        new = self._node('opener 07-23')
        self.brain.connect(new, old, relation='supersedes')

        dec = self._decoder()
        healed = dec._heal_graph()

        self.assertEqual(self._archived(old), 1)
        self.assertEqual(self._archived(new), 0)
        self.assertTrue(any(h['id'] == old and 'superseded' in h['reason']
                            for h in healed))
        # Step 10: the heal routes through dispatch — survivor lineage must
        # survive the routing (the batch archive op carries survivor_id) and
        # the archive must trace on the decoder's run chain.
        ptr = self.brain._meta_kv.get_all_bulk([old])[old].get(
            '_sys_archived_survivor_id')
        self.assertEqual(ptr, new, 'lineage lost in dispatch routing')
        rows = [t for t in self.brain._trace_dal.get_chain(dec.chain_id())
                if t['ref_type'] == 'node_archived' and t['ref_id'] == old]
        self.assertEqual(len(rows), 1,
                         'superseded-handoff heal must emit node_archived')

    def test_chain_archives_every_predecessor(self):
        # a ← b ← c (c newest): both a and b retire, head stays.
        a = self._node('opener 1')
        b = self._node('opener 2')
        c = self._node('opener 3')
        self.brain.connect(b, a, relation='supersedes')
        self.brain.connect(c, b, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(a), 1)
        self.assertEqual(self._archived(b), 1)
        self.assertEqual(self._archived(c), 0)

    def test_inverse_verb_superseded_by_also_retires(self):
        # Same semantics, inverse verb: old --superseded_by--> new.
        # The heal handles both canonical verbs — but ONLY these two,
        # never the (LLM-grown) correction_improvement aspect list.
        old = self._node('opener written inverse')
        new = self._node('opener successor')
        self.brain.connect(old, new, relation='superseded_by')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 1)
        self.assertEqual(self._archived(new), 0)

    def test_inverted_stored_orientation_still_retires_older(self):
        # Finding 1 of the a4f934c review: add_relation reuses the pair's
        # physical edge row in either orientation (Hebbian co-access creates
        # them in recall order), so `supersedes` can be STORED inverted —
        # old as source, new as target. The heal must not trust orientation:
        # created_at decides, and the OLDER node retires regardless.
        old = self._node('opener corrupt-direction old')
        new = self._node('opener corrupt-direction new')
        self.brain.connect(old, new, relation='supersedes')  # inverted on purpose

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 1)
        self.assertEqual(self._archived(new), 0)

    def test_self_loop_never_archives(self):
        # Finding 2: connect(x, x, 'supersedes') succeeds at the DAL; the
        # heal must not archive a node as superseded by itself.
        x = self._node('opener self-loop')
        self.brain.connect(x, x, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(x), 0)

    def test_equal_created_at_skips(self):
        # No date ground truth → never guess an archive.
        a = self._node('opener twin a')
        b = self._node('opener twin b')
        ts = self.brain.conn.execute(
            'SELECT created_at FROM nodes WHERE id = ?', (a,)).fetchone()[0]
        self.brain.conn.execute(
            'UPDATE nodes SET created_at = ? WHERE id = ?', (ts, b))
        self.brain.conn.commit()
        self.brain.connect(b, a, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(a), 0)
        self.assertEqual(self._archived(b), 0)

    def test_parallel_openers_untouched(self):
        # Two live threads, deliberately NO supersedes edge — both stay live.
        t1 = self._node('opener thread A')
        t2 = self._node('opener thread B')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(t1), 0)
        self.assertEqual(self._archived(t2), 0)

    def test_superseded_knowledge_stays_live(self):
        # supersedes between non-handoff types is knowledge lifecycle —
        # handled by correction_enrich at read time, never auto-archived.
        old = self._node('old decision', type='decision')
        new = self._node('new decision', type='decision')
        self.brain.connect(new, old, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 0)
        self.assertEqual(self._archived(new), 0)

    def test_locked_predecessor_skipped_quietly(self):
        old = self._node('locked opener', locked=True)
        new = self._node('successor opener')
        self.brain.connect(new, old, relation='supersedes')

        healed = self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 0)
        # Excluded in the query, not bounced off archive_node's guard —
        # so it never appears in the healed list (and never error-logs).
        self.assertFalse(any(h['id'] == old for h in healed))

    def test_mixed_type_supersession_untouched(self):
        # Edge exists but only one endpoint is a handoff → not opener
        # lifecycle; leave it alone.
        old = self._node('old opener')
        new = self._node('new decision', type='decision')
        self.brain.connect(new, old, relation='supersedes')

        self._decoder()._heal_graph()

        self.assertEqual(self._archived(old), 0)


class TestIntraClusterEdgeRenderContract(SupersessionBase):
    """Encoder contract: intra-member edges render WITH direction.

    Pins the fix for the survivor-direction blindness. Do not weaken:
    if intra-cluster edges disappear from the render, the encoder is
    back to guessing survivor direction from age-correlated signals.
    """

    def _cluster(self, a, b, edge_details):
        return {
            'nodes': [a, b],
            'size': 2,
            'pre_class': 'needs_judgment',
            'content_cosine_max': 0.95,
            'title_cosine_max': 0.97,
            'node_details': {
                a: {'title': 'opener old', 'type': 'handoff'},
                b: {'title': 'opener new', 'type': 'handoff'},
            },
            'co_recall_count': 0,
            'judge_preference': {},
            'catalog_blind': {},
            'shared_edge_count': 0,
            'same_community': False,
            'has_correction_edge': True,
            'edge_details': edge_details,
            'communities': {},
        }

    def _encoder(self):
        from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
        return ConsolidationEncoder(self.brain)

    def test_intra_cluster_edge_rendered_with_direction(self):
        old = self._node('opener old')
        new = self._node('opener new')
        # Production shape (get_connections_bulk, grouped per owner): the
        # intra-cluster edge appears under BOTH members, outgoing from the
        # actor and incoming on the target — the render must show it once.
        edge_details = {
            new: {old: [{'id': old, 'title': 'opener old', 'type': 'handoff',
                         'direction': 'outgoing',
                         'relations': [{'relation': 'supersedes',
                                        'description': 'newer opener'}]}]},
            old: {new: [{'id': new, 'title': 'opener new', 'type': 'handoff',
                         'direction': 'incoming',
                         'relations': [{'relation': 'supersedes',
                                        'description': 'newer opener'}]}]},
        }
        text = self._encoder()._format_clusters([self._cluster(old, new, edge_details)])

        expected = '%s → supersedes → %s' % (new[:8], old[:8])
        self.assertIn(expected, text,
                      'intra-cluster supersedes edge must render with direction '
                      '(actor first) — without it the encoder merges chains backwards')
        # Dedup: the mirrored incoming entry must NOT produce a reversed line.
        reversed_line = '%s → supersedes → %s' % (old[:8], new[:8])
        self.assertNotIn(reversed_line, text)
        self.assertEqual(text.count('supersedes →'), 1)

    def test_edge_data_arrives_whole_with_the_relations_age(self):
        """The decoder used to cut descriptions to 80 and titles to 60 chars
        before the encoder ever saw them — a description a reader copies as
        a swap's `old` has to arrive whole, and the edge line's age is the
        relation's, so the loader carries it."""
        a = self._node('member')
        b = self._node('a neighbor title long enough to have been cut at sixty characters by the old loader', type='fact')
        long_desc = 'd' * 120 + ' — the tail past eighty that used to vanish'
        self.brain.connect_typed(a, b, relation='extends', weight=0.6,
                                 description=long_desc, encoding_source='test')
        data = self._decoder()._load_edge_data([a])
        [conn] = data[a][b]
        (rel,) = conn['relations']
        self.assertEqual(rel['description'], long_desc)
        self.assertEqual(conn['title'], self.brain.get_node(b)['title'])
        self.assertTrue(rel.get('created_at'))
        # and the encoder's External block renders it whole, in the one grammar
        text = self._encoder()._format_clusters([self._cluster(a, self._node('other'), {a: {b: [conn]}})])
        self.assertIn('this extends "%s' % conn['title'][:100], text)
        self.assertIn(' — ' + long_desc, text)

    def test_edge_between_members_of_two_clusters_reaches_both(self):
        """The decoder loads edges for every cluster's ids at once. An edge
        whose endpoints sit in two different clusters must appear under BOTH
        owners — the flat loader gave it to the source only, and with the
        rich block's edges off the target member rendered edge-blind."""
        a = self._node('member of cluster one')
        n = self._node('member of cluster two', type='fact')
        self.brain.connect_typed(n, a, relation='depends_on', weight=0.6,
                                 description='cluster two leans on cluster one',
                                 encoding_source='test')
        data = self._decoder()._load_edge_data([a, n])
        self.assertEqual([c['direction'] for c in data[a][n]], ['incoming'])
        self.assertEqual([c['direction'] for c in data[n][a]], ['outgoing'])
        text = self._encoder()._format_clusters([self._cluster(a, self._node('other'), {a: data[a]})])
        self.assertIn('"member of cluster two" depends_on this — cluster two leans on cluster one', text)

    def test_claims_both_ways_between_members_each_render_from_their_source(self):
        """A handoff replacement writes "new supersedes old" and then "old
        absorbed_into new" on the same pair. Each claim keeps its writer's
        order, so the loader lists the neighbor once per direction and the
        intra block prints both claims, each from its own source."""
        old = self._node('opener old')
        new = self._node('opener new')
        self.brain.connect_typed(new, old, relation='supersedes', weight=0.6,
                                 description='newer opener', encoding_source='test')
        self.brain.connect_typed(old, new, relation='absorbed_into', weight=0.6,
                                 description='folded into the newer opener',
                                 encoding_source='test')
        data = self._decoder()._load_edge_data([old, new])
        self.assertEqual(sorted(c['direction'] for c in data[old][new]),
                         ['incoming', 'outgoing'])
        text = self._encoder()._format_clusters([self._cluster(old, new, data)])
        self.assertIn('%s → supersedes → %s' % (new[:8], old[:8]), text)
        self.assertIn('%s → absorbed_into → %s' % (old[:8], new[:8]), text)
        self.assertNotIn('%s → supersedes → %s' % (old[:8], new[:8]), text)

    def test_edge_loader_excludes_noise_like_get_node(self):
        """The loader applies the registry's structural_exclusions — the same
        set get_node uses — so community_member and the other plumbing
        relations are neither rendered as External edges nor offered to the
        encoder as migration candidates. Community placement still reaches
        the cluster block, through _load_community_membership."""
        a = self._node('member')
        comm = self._node('a community', type='community')
        peer = self._node('a peer', type='fact')
        self.brain.connect_typed(comm, a, relation='community_member', weight=0.9,
                                 encoding_source='test')
        self.brain.connect_typed(a, peer, relation='co_anchored', weight=0.9,
                                 description='shared episodic anchor', encoding_source='test')
        self.brain.connect_typed(a, peer, relation='extends', weight=0.5,
                                 description='the semantic claim', encoding_source='test')
        data = self._decoder()._load_edge_data([a])
        self.assertEqual(set(data[a]), {peer})
        self.assertEqual([r['relation'] for c in data[a][peer] for r in c['relations']], ['extends'])
        self.assertEqual(self._decoder()._load_community_membership([a]),
                         {a: [{'id': comm, 'title': 'a community'}]})

    def test_cluster_block_prints_one_header_per_node_keeping_the_flags(self):
        """render_rich_node prints the `[type] "title" (id:…)` header; the
        encoder's own `[type] "title"` line above it was a duplicate. The
        separator keeps its LOCKED flag."""
        old = self._node('opener old', locked=True)
        new = self._node('opener new')
        cluster = self._cluster(old, new, {old: {}, new: {}})
        cluster['node_details'][old]['locked'] = True
        text = self._encoder()._format_clusters([cluster])
        self.assertEqual(text.count('"opener old"'), 1, text)
        self.assertEqual(text.count('"opener new"'), 1, text)
        self.assertIn('    --- %s --- [LOCKED]' % old[:8], text)
        self.assertIn('    --- %s ---\n' % new[:8], text)
        self.assertIn('[handoff] "opener old" (id:%s, locked' % old[:8], text)
        # the rich header already carries src: and the age — the separator
        # block does not repeat them
        self.assertNotIn('src=', text)
        self.assertNotIn('created=', text)
        # a member that vanished between decode and encode still reads as
        # `[type] "title"`, never as an anonymous body
        gone = 'deadbeef'
        cluster = self._cluster(old, gone, {old: {}, gone: {}})
        cluster['node_details'][gone] = {'title': 'vanished', 'type': 'handoff',
                                         'content': 'its body'}
        text = self._encoder()._format_clusters([cluster])
        self.assertIn('      [handoff] "vanished"\n      Content: its body', text)

    def test_cluster_block_renders_member_content_whole(self):
        """A merge or keep is decided on the claim, not a gist: the rich node
        block and both fallbacks render the member's content whole."""
        long_body = 'c' * 900 + ' THE TAIL PAST SIX HUNDRED'
        a = self._node('long member')
        self.brain.revise(a, content=long_body)
        cluster = self._cluster(a, self._node('other'), {})
        self.assertIn(long_body, self._encoder()._format_clusters([cluster]))
        gone = 'deadbeef'
        cluster = self._cluster(a, gone, {})
        cluster['node_details'][gone] = {'title': 'vanished', 'type': 'handoff',
                                         'content': long_body}
        self.assertIn('      Content: ' + long_body, self._encoder()._format_clusters([cluster]))

    def test_external_edges_still_external_only(self):
        # The intra block must not leak external edges, and vice versa.
        old = self._node('opener old')
        new = self._node('opener new')
        outsider = self._node('elsewhere', type='fact')
        edge_details = {
            new: {
                old: [{'id': old, 'title': 'opener old', 'type': 'handoff',
                       'direction': 'outgoing',
                       'relations': [{'relation': 'supersedes', 'description': ''}]}],
                outsider: [{'id': outsider, 'title': 'elsewhere', 'type': 'fact',
                            'direction': 'outgoing',
                            'relations': [{'relation': 'extends', 'description': ''}]}],
            },
            old: {},
        }
        text = self._encoder()._format_clusters([self._cluster(old, new, edge_details)])

        self.assertIn('%s → supersedes → %s' % (new[:8], old[:8]), text)
        # The external neighbor renders in the per-node External block,
        # not in the intra-cluster block.
        intra_section = text.split('Intra-cluster edges')[1].split('Nodes:')[0]
        self.assertNotIn(outsider[:8], intra_section)


if __name__ == '__main__':
    unittest.main()
