"""Tests for brain.absorb() — the lossless merge primitive.

absorb folds `absorbed` INTO `survivor` (source_refs union, edge re-point,
access_count sum, KV fill, optional content override) then archives `absorbed`.
Transfer-by-default: the merge must NOT silently drop information the imperative
revise+connect+archive path lost (preservation audit, node 988de522).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestAbsorb(BrainTestBase):
    needs_embedder = False

    def _node(self, title, content='content', locked=False, **kw):
        r = self.brain.remember(type='fact', title=title, content=content,
                                locked=locked, encoding_source='anchor', **kw)
        return r['id']

    # ── structural transfers (lossless-by-default) ──

    def test_source_refs_union(self):
        survivor = self._node('survivor', source_refs=['aaaaaaaa'])
        absorbed = self._node('absorbed', source_refs=['bbbbbbbb', 'cccccccc'])
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        refs = set(self.brain._source_refs.get_source_refs(survivor))
        self.assertEqual(refs, {'aaaaaaaa', 'bbbbbbbb', 'cccccccc'})

    def test_external_edge_migrated_intra_dropped(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        neighbor = self._node('neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on',
                                       description='external — should migrate')
        self.brain._graph.add_relation(absorbed, survivor, 'similar_to',
                                       description='intra — should die with absorbed')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        conns = self.brain._graph.get_connections_bulk([survivor]).get(survivor, [])
        pairs = {(c['id'], rel['relation'])
                 for c in conns for rel in c['relations']}
        self.assertIn((neighbor, 'depends_on'), pairs)       # external migrated
        self.assertNotIn((absorbed, 'similar_to'), pairs)    # intra-pair gone

    def test_community_member_not_migrated(self):
        """Ordinary node merges leave placement to the community unit."""
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        neighbor = self._node('neighbor')
        community = self.brain.remember(
            type='community', title='Test community', content='cluster',
            encoding_source='s2:community_detection')['id']
        self.brain._graph.add_relation(community, absorbed, 'community_member',
                                       weight=0.3)
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on',
                                       description='semantic — should migrate')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        conns = self.brain._graph.get_connections_bulk([survivor]).get(survivor, [])
        pairs = {(c['id'], rel['relation'])
                 for c in conns for rel in c['relations']}
        self.assertNotIn((community, 'community_member'), pairs)
        self.assertIn((neighbor, 'depends_on'), pairs)
        self.assertEqual(r['edges_migrated'], 1)

    def test_access_count_summed(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        self.brain.conn.execute(
            "UPDATE nodes SET access_count = 10 WHERE id = ?", (survivor,))
        self.brain.conn.execute(
            "UPDATE nodes SET access_count = 7 WHERE id = ?", (absorbed,))
        self.brain.conn.commit()
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        ac = self.brain.conn.execute(
            "SELECT access_count FROM nodes WHERE id = ?", (survivor,)).fetchone()[0]
        self.assertEqual(ac, 17)

    def test_kv_filled_where_survivor_lacks_survivor_wins(self):
        survivor = self._node('survivor', situation='survivor situation')
        absorbed = self._node('absorbed', situation='absorbed situation',
                              their_raw_quote='the peer quote')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertEqual(kv.get('their_raw_quote'), 'the peer quote')  # filled
        self.assertEqual(kv.get('situation'), 'survivor situation')   # survivor wins

    def test_distinct_voice_quotes_merge_appended(self):
        """Voice exception: when survivor AND absorbed both carry a distinct
        quote, the absorbed quote is APPENDED, not dropped (it's meaning
        paraphrase can't recover). Non-voice fields stay survivor-wins."""
        survivor = self._node('survivor', their_raw_quote='survivor said this',
                              situation='survivor situation')
        absorbed = self._node('absorbed', their_raw_quote='absorbed said that',
                              my_raw_quote='anchor reflected here',
                              situation='absorbed situation')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        # both present + distinct → appended
        self.assertEqual(kv.get('their_raw_quote'),
                         'survivor said this\n\nabsorbed said that')
        # survivor lacked my_raw_quote → filled (not appended)
        self.assertEqual(kv.get('my_raw_quote'), 'anchor reflected here')
        # non-voice field unchanged: survivor still wins
        self.assertEqual(kv.get('situation'), 'survivor situation')
        self.assertEqual(r.get('voice_merged'), ['their_raw_quote'])

    def test_duplicate_voice_quote_not_appended(self):
        survivor = self._node('survivor', their_raw_quote='same words')
        absorbed = self._node('absorbed', their_raw_quote='same words')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertEqual(kv.get('their_raw_quote'), 'same words')  # no dup

    def test_substring_voice_quote_not_appended(self):
        survivor = self._node('survivor', their_raw_quote='the full long quote here')
        absorbed = self._node('absorbed', their_raw_quote='long quote')  # substring
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertEqual(kv.get('their_raw_quote'), 'the full long quote here')

    def test_voice_caller_override_beats_append(self):
        survivor = self._node('survivor', their_raw_quote='survivor quote')
        absorbed = self._node('absorbed', their_raw_quote='absorbed quote')
        r = self.brain.absorb(survivor, absorbed, their_raw_quote='explicit override')
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertEqual(kv.get('their_raw_quote'), 'explicit override')

    def test_voice_drop_respected(self):
        survivor = self._node('survivor', their_raw_quote='survivor quote')
        absorbed = self._node('absorbed', their_raw_quote='absorbed quote')
        r = self.brain.absorb(survivor, absorbed, drop_fields=['their_raw_quote'])
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertEqual(kv.get('their_raw_quote'), 'survivor quote')  # not appended

    def test_content_override_replaces(self):
        survivor = self._node('survivor', content='old survivor content')
        absorbed = self._node('absorbed')
        r = self.brain.absorb(survivor, absorbed, content='merged synthesis')
        self.assertTrue(r['ok'], r)
        c = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (survivor,)).fetchone()[0]
        self.assertEqual(c, 'merged synthesis')

    def test_content_preserved_when_not_overridden(self):
        survivor = self._node('survivor', content='survivor keeps this')
        absorbed = self._node('absorbed')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        c = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (survivor,)).fetchone()[0]
        self.assertEqual(c, 'survivor keeps this')

    # ── archive + provenance ──

    def test_absorbed_is_archived_with_provenance(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        self.assertTrue(r['absorbed_archived'])
        archived = self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0]
        self.assertEqual(archived, 1)
        prov = self.brain.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE node_id = ? "
            "AND key = '_sys_archived_survivor_id'", (absorbed,)).fetchone()
        self.assertIsNotNone(prov)
        self.assertEqual(prov[0], survivor)

    # ── guards ──

    def test_locked_absorbed_refused_not_archived(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed', locked=True)
        r = self.brain.absorb(survivor, absorbed)
        self.assertFalse(r['ok'])
        self.assertIn('locked', r['error'])
        archived = self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0]
        self.assertEqual(archived, 0)  # locked node never archived

    def test_survivor_may_be_locked(self):
        # The whole point of locked-absorb: you absorb INTO a locked node.
        survivor = self._node('survivor', locked=True)
        absorbed = self._node('absorbed', their_raw_quote='peer quote')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertEqual(kv.get('their_raw_quote'), 'peer quote')

    def test_self_absorb_refused(self):
        n = self._node('n')
        r = self.brain.absorb(n, n)
        self.assertFalse(r['ok'])

    # ── full field flexibility (revise-shape overrides) ──

    def test_field_override_via_kwargs(self):
        survivor = self._node('survivor', content='old')
        absorbed = self._node('absorbed')
        r = self.brain.absorb(survivor, absorbed, content='synthesis',
                              title='merged title', confidence=0.99)
        self.assertTrue(r['ok'], r)
        row = self.brain.conn.execute(
            "SELECT content, title, confidence FROM nodes WHERE id = ?",
            (survivor,)).fetchone()
        self.assertEqual(row[0], 'synthesis')
        self.assertEqual(row[1], 'merged title')
        self.assertEqual(row[2], 0.99)

    def test_field_override_via_updates_dict(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        r = self.brain.absorb(survivor, absorbed,
                              updates={'title': 'X', 'confidence': 0.5})
        self.assertTrue(r['ok'], r)
        row = self.brain.conn.execute(
            "SELECT title, confidence FROM nodes WHERE id = ?", (survivor,)).fetchone()
        self.assertEqual(row[0], 'X')
        self.assertEqual(row[1], 0.5)

    def test_caller_override_beats_kv_fill(self):
        # survivor lacks situation; absorbed has one; caller sets a third —
        # the caller's explicit override wins over the auto KV-fill.
        survivor = self._node('survivor')
        absorbed = self._node('absorbed', situation='from absorbed')
        r = self.brain.absorb(survivor, absorbed, situation='from caller')
        self.assertTrue(r['ok'], r)
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertEqual(kv.get('situation'), 'from caller')

    def test_field_override_via_brain_batch_op(self):
        from servers.daemon_dispatch import COMMAND_TABLE
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        res = COMMAND_TABLE['brain_batch'].handler(self.brain, {'operations': [
            {'op': 'absorb', 'survivor_id': survivor, 'absorbed_id': absorbed,
             'title': 'op-merged', 'confidence': 0.88}]}, [])
        self.assertTrue(res.get('ok'), res)
        row = self.brain.conn.execute(
            "SELECT title, confidence FROM nodes WHERE id = ?", (survivor,)).fetchone()
        self.assertEqual(row[0], 'op-merged')
        self.assertEqual(row[1], 0.88)

    # ── guards + atomicity + fidelity (the review's gaps) ──

    def test_already_archived_absorbed_refused(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        self.brain.archive_node(absorbed, archived_by='test', reason='pre')
        r = self.brain.absorb(survivor, absorbed)
        self.assertFalse(r['ok'])
        self.assertIn('already archived', r['error'])

    def test_edge_weight_preserved(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        neighbor = self._node('neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on', weight=0.9)
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        conns = self.brain._graph.get_connections_bulk([survivor]).get(survivor, [])
        w = next(rel['weight'] for c in conns if c['id'] == neighbor
                 for rel in c['relations'] if rel['relation'] == 'depends_on')
        self.assertEqual(w, 0.9)  # not defaulted to 0.5

    def test_atomicity_rollback_on_failure(self):
        # A raise mid-absorb must leave NO partial merge: survivor unchanged,
        # absorbed not archived.
        from unittest import mock
        survivor = self._node('survivor', source_refs=['aaaaaaaa'])
        absorbed = self._node('absorbed', source_refs=['bbbbbbbb'])
        neighbor = self._node('neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on')
        with mock.patch.object(self.brain, 'archive_node',
                               side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                self.brain.absorb(survivor, absorbed)
        # source_refs NOT unioned, edge NOT migrated, absorbed NOT archived
        self.assertEqual(set(self.brain._source_refs.get_source_refs(survivor)),
                         {'aaaaaaaa'})
        conns = self.brain._graph.get_connections_bulk([survivor]).get(survivor, [])
        self.assertNotIn(neighbor, {c['id'] for c in conns})
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0], 0)


class TestCommunityAbsorb(BrainTestBase):
    needs_embedder = False

    def node(self, title, type='finding', **kw):
        return self.brain.remember(type=type, title=title, content=title,
                                   **kw)['id']

    def members(self, cid):
        return {m['id'] for m in self.brain._graph.get_members_bulk([cid]).get(cid, [])}

    def pair(self):
        # The observed failure: 13 + 11 members with 9 in common must yield 15.
        members = [self.node('Member %d' % i) for i in range(15)]
        # Communities are the S2 unit's own, as its decoder and stamp read them.
        survivor = self.node('Larger', 'community', community_maturity='settled',
                             encoding_source='s2:community_detection')
        absorbed = self.node('Smaller', 'community',
                             encoding_source='s2:community_detection')
        for member in members[:13]:
            self.brain.connect(survivor, member, relation='community_member')
        for i, member in enumerate(members[:9] + members[13:]):
            # Both directions are valid in the membership reader.
            src, dst = (absorbed, member) if i % 2 else (member, absorbed)
            self.brain.connect(src, dst, relation='community_member', weight=0.7)
        return survivor, absorbed, members

    def test_merge_unions_live_members_preserving_other_relations(self):
        survivor, absorbed, members = self.pair()
        self.brain._graph.add_relation(absorbed, members[13], 'extends', weight=0.8,
                                        description='Distinct evidence for the combined story')
        retired = self.node('Retired member')
        self.brain.connect(absorbed, retired, relation='community_member')
        self.brain.archive_node(retired, archived_by='test', reason='retired')
        legacy = self.node('Legacy nested community', 'community')
        self.brain.connect(absorbed, legacy, relation='community_member')

        result = self.brain.absorb(survivor, absorbed, content='Combined story')

        self.assertTrue(result['ok'], result)
        self.assertEqual(self.members(survivor), set(members))
        conns = self.brain._graph.get_connections_bulk([survivor])[survivor]
        self.assertNotIn(legacy, {c['id'] for c in conns})
        self.assertNotIn(retired, {c['id'] for c in conns})
        relation = next(r for c in conns if c['id'] == members[13]
                        for r in c['relations'] if r['relation'] == 'extends')
        self.assertEqual(relation['weight'], 0.8)
        self.assertEqual(relation['description'], 'Distinct evidence for the combined story')
        for mid in members[13:]:
            relation = next(r for c in conns if c['id'] == mid
                            for r in c['relations'] if r['relation'] == 'community_member')
            self.assertEqual(relation['weight'], 0.7)
        self.assertTrue(self.brain._nodes.get_bulk([absorbed])[absorbed]['archived'])
        self.assertEqual(self.brain._meta_kv.get_all_bulk([survivor])[survivor]
                         ['community_maturity'], 'settled')

    def test_mixed_node_types_do_not_inherit_membership(self):
        for survivor_type, absorbed_type in [('community', 'finding'),
                                              ('finding', 'community')]:
            with self.subTest(survivor_type=survivor_type):
                survivor = self.node('Survivor', survivor_type)
                absorbed = self.node('Absorbed', absorbed_type)
                member = self.node('Member')
                self.brain.connect(absorbed, member, relation='community_member')
                self.assertTrue(self.brain.absorb(survivor, absorbed)['ok'])
                self.assertNotIn(member, {c['id'] for c in
                    self.brain._graph.get_connections_bulk([survivor]).get(survivor, [])})

    def test_explicit_retype_preserves_transfer_then_applies_override(self):
        survivor, absorbed, members = self.pair()
        result = self.brain.absorb(survivor, absorbed, type='finding',
                                   content='Synthesized finding')
        self.assertTrue(result['ok'], result)
        rows = self.brain._nodes.get_bulk([survivor, absorbed])
        self.assertEqual(rows[survivor]['type'], 'finding')
        self.assertTrue(rows[absorbed]['archived'])
        links = self.brain._graph.get_connections_bulk(
            [survivor], include_relations=['community_member'])[survivor]
        self.assertEqual({c['id'] for c in links}, set(members))
        # Retyping does not change the community reader's typed contract.
        self.assertEqual(self.members(survivor), set())

    def test_encoder_merge_stamps_union_and_next_decode_has_no_old_merge(self):
        from unittest.mock import patch
        from servers.scales.s2.community_encoder import CommunityEncoder
        from servers.scales.s2.community_decoder import CommunityDecoder
        survivor, absorbed, members = self.pair()
        decoder = CommunityDecoder(self.brain)
        operations = [{'op': 'absorb', 'survivor_id': survivor,
                       'absorbed_id': absorbed, 'content': 'Combined story'}]

        def respond(**kw):
            response = kw['dispatch_fn']('brain_batch', {'operations': operations})
            self.assertTrue(response['result']['results'][0]['ok'], response)
            return {'rounds': 1, 'actions': 1, 'write_actions': 1,
                    'action_details': [{'tool': 'brain_batch',
                                        'input': {'operations': operations}}],
                    'final_text': '## Review\n```\n```'}

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.runner.make_client', return_value=object()), \
                patch('servers.scales.runner.run_llm_loop', side_effect=respond):
            result = CommunityEncoder(self.brain).run([
                {'type': 'merge_communities', 'larger_id': survivor,
                 'smaller_id': absorbed}], decoder._read_community_state())
        self.assertFalse(result.get('error'), result)
        self.assertEqual(result['rejection_skipped_count'], 0)
        self.assertEqual(self.members(survivor), set(members))
        # Members share no edges, so the stamp derives no internal cohesion.
        self.assertEqual(self.brain._meta_kv.get_all_bulk([survivor])[survivor]
                         ['community_internal_fraction'], '0.0')
        next_run = decoder.run()
        self.assertFalse(any(p['type'] == 'merge_communities'
                             and absorbed in (p['larger_id'], p['smaller_id'])
                             for p in next_run['proposals']))

    def test_explicit_membership_pruning_remains_supported(self):
        survivor, absorbed, members = self.pair()
        result = self.brain.absorb(survivor, absorbed, prune_edges=['community_member'])
        self.assertTrue(result['ok'], result)
        self.assertEqual(self.members(survivor), set(members[:13]))

    def test_batch_failure_rolls_back_membership_content_and_archive(self):
        from unittest.mock import patch
        from servers.daemon_dispatch import COMMAND_TABLE
        survivor, absorbed, members = self.pair()
        with patch.object(self.brain, 'archive_node', side_effect=RuntimeError('archive failed')):
            response = COMMAND_TABLE['brain_batch'].handler(self.brain, {'operations': [
                {'op': 'absorb', 'survivor_id': survivor, 'absorbed_id': absorbed,
                 'content': 'Combined story'}]}, [])
        self.assertFalse(response['result']['results'][0]['ok'], response)
        self.assertEqual(self.members(survivor), set(members[:13]))
        self.assertEqual(self.members(absorbed), set(members[:9] + members[13:]))
        self.assertEqual(self.brain.get_node(survivor)['content'], 'Larger')
        self.assertFalse(self.brain._nodes.get_bulk([absorbed])[absorbed]['archived'])

    def test_incomplete_transfer_refuses_archive_and_rolls_back(self):
        from unittest.mock import patch
        survivor, absorbed, members = self.pair()
        add = self.brain._graph.add_relation

        def drop_member(source_id, target_id, relation, *args, **kw):
            if relation == 'community_member' and members[13] in (source_id, target_id):
                return {'ok': False, 'error': 'injected missing link'}
            return add(source_id, target_id, relation, *args, **kw)

        with patch.object(self.brain._graph, 'add_relation', side_effect=drop_member):
            for overrides in ({}, {'type': 'finding'}):
                with self.subTest(overrides=overrides):
                    with self.assertRaisesRegex(RuntimeError, 'community absorb lost member links'):
                        self.brain.absorb(survivor, absorbed, content='Combined story',
                                          **overrides)
        self.assertEqual(self.members(survivor), set(members[:13]))
        self.assertEqual(self.members(absorbed), set(members[:9] + members[13:]))
        self.assertEqual(self.brain.get_node(survivor)['content'], 'Larger')


class TestAbsorbEmbedding(BrainTestBase):
    needs_embedder = True

    def _node(self, title, content='content', **kw):
        return self.brain.remember(type='fact', title=title, content=content,
                                   encoding_source='anchor', **kw)['id']

    def test_situation_fill_is_re_embedded(self):
        # The review's #1: KV-fill must route through revise() so the filled
        # `situation` (an embedding-bearing field) gets a vector — set_many
        # alone would leave it unembedded and invisible to recall.
        survivor = self._node('survivor', content='survivor body')
        absorbed = self._node('absorbed', content='absorbed body',
                              situation='zylophone quasar absorbent retrieval marker')
        before = self.brain.conn.execute(
            "SELECT COUNT(*) FROM node_enrichments WHERE node_id = ?",
            (survivor,)).fetchone()[0]
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)
        # situation filled onto survivor...
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertIn('zylophone', kv.get('situation', ''))
        # ...AND it produced a new enrichment vector (re-embedded, not raw set_many)
        after = self.brain.conn.execute(
            "SELECT COUNT(*) FROM node_enrichments WHERE node_id = ?",
            (survivor,)).fetchone()[0]
        self.assertGreater(after, before)

    def test_missing_nodes_refused(self):
        survivor = self._node('survivor')
        self.assertFalse(self.brain.absorb(survivor, 'deadbeef')['ok'])
        self.assertFalse(self.brain.absorb('deadbeef', survivor)['ok'])

    # ── end-to-end through the brain_batch op (dispatcher branch + atomicity) ──

    def test_absorb_via_brain_batch_op(self):
        from servers.daemon_dispatch import COMMAND_TABLE
        survivor = self._node('survivor', source_refs=['aaaaaaaa'])
        absorbed = self._node('absorbed', source_refs=['bbbbbbbb'],
                              their_raw_quote='peer quote')
        res = COMMAND_TABLE['brain_batch'].handler(self.brain, {'operations': [
            {'op': 'absorb', 'survivor_id': survivor, 'absorbed_id': absorbed,
             'encoding_source': 's2:consolidation'}]}, [])
        self.assertTrue(res.get('ok'), res)
        # absorbed archived, refs unioned, KV filled — all in one atomic batch
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0], 1)
        self.assertEqual(set(self.brain._source_refs.get_source_refs(survivor)),
                         {'aaaaaaaa', 'bbbbbbbb'})
        self.assertEqual(
            self.brain._meta_kv.get_all_bulk([survivor])[survivor].get('their_raw_quote'),
            'peer quote')

    def test_content_override_via_brain_batch_op(self):
        """REGRESSION: a content-rewriting absorb inside brain_batch must be
        atomic AND succeed. The content override re-embeds the survivor; the
        embed backfill (_store_batch in backfill_vectors) used to self-commit
        unconditionally, and a COMMIT releases every open SAVEPOINT — so the
        absorb's `absorb_sp` vanished mid-merge and `RELEASE absorb_sp` threw
        `no such savepoint: absorb_sp`. The absorb sub-op returned ok=False
        while the batch reported ok=True overall: a silent merge loss. This is
        S2 Consolidation's primary path (synthesised content + absorb).

        The pre-existing test_field_override_via_brain_batch_op only exercised
        title/confidence (no re-embed), so it never tripped the backfill commit
        and this regression slipped through."""
        from servers.daemon_dispatch import COMMAND_TABLE
        survivor = self._node('survivor', content='old survivor body')
        absorbed = self._node('absorbed', content='folded body')
        res = COMMAND_TABLE['brain_batch'].handler(self.brain, {'operations': [
            {'op': 'absorb', 'survivor_id': survivor, 'absorbed_id': absorbed,
             'content': 'merged synthesis content', 'reason': 'consolidation',
             'encoding_source': 's2:consolidation'}]}, [])
        # The batch reports ok (it always did) — but now the SUB-OP must too.
        self.assertTrue(res.get('ok'), res)
        sub = res['result']['results'][0]
        self.assertTrue(sub.get('ok'), sub)               # the bug: this was False
        self.assertNotIn('savepoint', str(sub).lower())   # specifically not absorb_sp
        self.assertEqual(res['result']['failed'], 0, res['result'])
        # survivor content actually rewritten (the merge landed, not no-op'd)...
        c = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (survivor,)).fetchone()[0]
        self.assertEqual(c, 'merged synthesis content')
        # ...and absorbed archived
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0], 1)

    def test_content_override_brain_batch_atomic_on_midmerge_failure(self):
        """A forced failure AFTER the content re-embed must roll the whole
        merge back cleanly — no half-merge committed. Before the fix the
        backfill commit had already released absorb_sp and flushed the
        partial merge, so the survivor content was rewritten-and-committed
        even though the absorb failed. With the savepoint intact, the unwind
        (ROLLBACK TO absorb_sp) restores the survivor exactly."""
        from unittest import mock
        survivor = self._node('survivor', content='untouched survivor body',
                              source_refs=['aaaaaaaa'])
        absorbed = self._node('absorbed', content='folded body',
                              source_refs=['bbbbbbbb'])
        from servers.daemon_dispatch import COMMAND_TABLE
        with mock.patch.object(self.brain, 'archive_node',
                               side_effect=RuntimeError('boom')):
            res = COMMAND_TABLE['brain_batch'].handler(self.brain, {'operations': [
                {'op': 'absorb', 'survivor_id': survivor, 'absorbed_id': absorbed,
                 'content': 'should not persist', 'reason': 't',
                 'encoding_source': 's2:consolidation'}]}, [])
        # Per-op failure is recorded; the batch as a whole doesn't raise.
        sub = res['result']['results'][0]
        self.assertFalse(sub.get('ok'), sub)
        self.assertNotIn('savepoint', str(sub).lower())  # clean unwind, not a sp error
        # No half-merge: content unchanged, absorbed not archived, refs not unioned.
        c = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (survivor,)).fetchone()[0]
        self.assertEqual(c, 'untouched survivor body')
        self.assertEqual(self.brain.conn.execute(
            "SELECT archived FROM nodes WHERE id = ?", (absorbed,)).fetchone()[0], 0)
        self.assertEqual(set(self.brain._source_refs.get_source_refs(survivor)),
                         {'aaaaaaaa'})


if __name__ == '__main__':
    unittest.main()
