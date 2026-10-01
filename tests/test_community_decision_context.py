"""Decision targets survive recall misses; multi-batch feedback stays current."""
import os
from unittest.mock import patch

from tests.brain_test_base import BrainTestBase
from servers.scales.s2.community_encoder import CommunityEncoder
from servers.scales.s2.community_contract import COMMUNITY_DETECTION
from servers.channels.thalamus import thalamus


class TestCommunityDecisionContext(BrainTestBase):
    needs_embedder = False

    def node(self, title, type='finding', **kw):
        return self.brain.remember(title=title, type=type,
                                   content=kw.pop('content', title),
                                   **kw)['id']

    def encoder(self, **kw):
        return CommunityEncoder(self.brain, config={**COMMUNITY_DETECTION, **kw})

    def test_recall_miss_keeps_full_target_and_candidate_evidence(self):
        narrative = 'Historical context. ' * 20 + 'Current account is still v4.'
        target = self.node('Onboarding', 'community', content=narrative)
        candidate = self.node('v5 validated', content='The desktop test passed.')
        member = self.node('Original ruling')
        self.brain.connect(member, target, relation='community_member')
        self.brain.connect(member, target, relation='extends')
        props = [{'type': 'add_to_existing', 'node_id': candidate,
                  'node_title': 'v5 validated', 'communities': [{'id': target}]}]
        enc = self.encoder()
        for result in ({'results': []}, RuntimeError('recall unavailable')):
            with patch.object(self.brain, 'recall', **(
                    {'side_effect': result} if isinstance(result, Exception)
                    else {'return_value': result})):
                text = enc._find_relevant_communities(props, [])
            self.assertIn(narrative, text)
            self.assertIn('The desktop test passed.', text)
            self.assertIn(member, text)
            self.assertIn('Live members: 1', text)
            self.assertIn('extends', text)

    def test_every_decision_role_is_loaded_without_a_search_query(self):
        ids = [self.node('Community %d' % i, 'community') for i in range(6)]
        props = [
            {'type': 'health_update', 'community_id': ids[0]},
            {'type': 'drift', 'home_id': ids[1], 'foreign': [{'id': ids[2]}]},
            {'type': 'merge_communities', 'larger_id': ids[3], 'smaller_id': ids[4]},
            {'type': 'new_community', 'overlaps_existing': {'id': ids[5]}},
        ]
        with patch.object(self.brain, 'recall', side_effect=AssertionError('no query')):
            text = self.encoder()._find_relevant_communities(props, [])
        for cid in ids:
            self.assertIn(cid, text)

    def test_nearby_context_yields_to_full_decision_evidence(self):
        narrative = 'This is the decision account. ' * 100
        target = self.node('Target', 'community', content=narrative)
        nearby = self.node('Optional neighbour', 'community',
                           situation='Large optional context. ' * 300)
        proposal = {'type': 'health_update', 'community_id': target,
                    'node_title': 'a query for nearby communities'}
        with patch.object(self.brain, 'recall', return_value={
                'results': [{'id': nearby}]}):
            text = self.encoder()._find_relevant_communities(
                [proposal], [], max_chars=5000)
        self.assertIn(narrative, text)
        self.assertNotIn('Large optional context', text)
        self.assertIn('1 nearby communities omitted', text)
        self.assertLessEqual(len(text), 5000)

    def test_redirected_target_uses_survivor_members_and_write_id(self):
        old = self.node('Old community', 'community')
        survivor = self.node('Surviving community', 'community')
        member = self.node('Live member')
        self.brain.connect(survivor, member, relation='community_member')
        self.assertTrue(self.brain.absorb(survivor, old)['ok'])
        text, ids = self.encoder()._decision_context([
            {'type': 'health_update', 'community_id': old},
            {'type': 'merge_communities', 'larger_id': survivor, 'smaller_id': old},
        ])
        self.assertEqual(ids, {survivor})
        self.assertIn('ID MOVED: %s → %s' % (old, survivor), text)
        self.assertIn('Live members: 1', text)
        self.assertIn(member, text)
        self.assertIn('do not merge it with itself', text)

    def test_retired_target_is_explicitly_archived(self):
        target = self.node('Retired community', 'community')
        self.brain.archive_node(target, archived_by='test', reason='retired')
        text, ids = self.encoder()._decision_context([
            {'type': 'health_update', 'community_id': target}])
        self.assertFalse(ids)
        self.assertIn('ARCHIVED', text)
        self.assertIn(target, text)

    def test_merge_transfer_is_complete_beyond_the_member_preview(self):
        large = self.node('Parent', 'community')
        small = self.node('Merged story', 'community')
        mids = [self.node('Member %d' % i) for i in range(12)]
        for mid in mids:
            self.brain.connect(small, mid, relation='community_member')
        self.brain.connect(large, mids[0], relation='community_member')
        text, _ = self.encoder()._decision_context([
            {'type': 'merge_communities', 'larger_id': large, 'smaller_id': small}])
        self.assertIn('12; 8 newest shown, 4 not shown', text)
        transfer_line = next(l for l in text.splitlines() if 'members over:' in l)
        self.assertIn('absorb moves ALL 11 members', transfer_line)
        self.assertNotIn(mids[0], transfer_line)
        for mid in mids[1:]:
            self.assertIn(mid, transfer_line)

    def test_missing_target_is_explicit(self):
        text, _ = self.encoder()._decision_context([
            {'type': 'health_update', 'community_id': 'deadbeef'}])
        self.assertIn('UNAVAILABLE: deadbeef', text)

    def test_oversized_batch_splits_without_losing_narrative_or_proposals(self):
        narrative = 'A concrete community account. ' * 200
        targets = [self.node('Target %d' % i, 'community', content=narrative)
                   for i in range(2)]
        props = [{'type': 'health_update', 'community_id': cid} for cid in targets]
        seen = []

        def run(**kwargs):
            seen.append(kwargs['user_content'])
            return {'actions': 0, 'write_actions': 0, 'rounds': 1,
                    'final_text': '## Review\n```\n```'}

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.runner.make_client', return_value=object()), \
                patch('servers.scales.runner.run_llm_loop', side_effect=run):
            self.encoder(max_batch_context_chars=9000)._encode(props, [])
        self.assertEqual(len(seen), 2)
        for cid, text in zip(targets, seen):
            self.assertIn(cid, text)
            self.assertIn(narrative, text)
            self.assertLessEqual(len(text), 9000)
            self.assertEqual(text.count('HEALTH UPDATE'), 1)

    def test_second_batch_reads_new_message_but_not_same_run_residue(self):
        source = 's2:community_detection'
        self.brain.write_journal_notes(
            final_text='## Review\n```\nopen · deployment · old pending claim\n```',
            chain_id='s2-previous-community_detection', scale='s2')
        thalamus.file(self.brain, source, 'old report', needs_answer=True,
                      dedup_key='maintenance')
        seen = []

        def run(**kwargs):
            seen.append(kwargs['user_content'])
            review = ('ask · maintenance · updated after first batch\n'
                      'resolved · deployment · verified deployed\n'
                      'open · fresh-residue · first batch private note') if len(seen) == 1 else ''
            return {'actions': 0, 'write_actions': 0, 'rounds': 1,
                    'final_text': '## Review\n```\n%s\n```' % review}

        props = [{'type': 'new_community', 'member_count': 0,
                  'internal_fraction': 0, 'all_members': []}] * 2
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.runner.make_client', return_value=object()), \
                patch('servers.scales.runner.run_llm_loop', side_effect=run):
            self.encoder(max_proposals_per_call=1)._encode(props, [])
        self.assertEqual(len(seen), 2)
        self.assertIn('old report', seen[0])
        self.assertIn('updated after first batch', seen[1])
        self.assertNotIn('old report', seen[1])
        self.assertNotIn('first batch private note', seen[1])
        self.assertIn('old pending claim', seen[0])
        self.assertNotIn('old pending claim', seen[1])
        self.assertIn('resolved · deployment · verified deployed', seen[1])

    def test_packing_keeps_shared_target_together_without_reordering(self):
        targets = [self.node('Target %d' % i, 'community',
                             content=('Account %d. ' % i) * 300) for i in range(3)]
        order = [targets[0], targets[1], targets[1], targets[2]]
        props = [{'type': 'health_update', 'community_id': cid} for cid in order]
        seen = []

        def run(**kw):
            seen.append(kw['user_content'])
            return {'rounds': 1, 'actions': 0, 'write_actions': 0, 'final_text': ''}

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.runner.make_client', return_value=object()), \
                patch('servers.scales.runner.run_llm_loop', side_effect=run):
            result = self.encoder(max_batch_context_chars=8500)._encode(props, [])
        self.assertEqual([text.count('HEALTH UPDATE') for text in seen], [3, 1])
        self.assertIn(targets[0], seen[0])
        self.assertIn(targets[1], seen[0])
        self.assertNotIn(targets[1], seen[1])
        self.assertIn(targets[2], seen[1])
        for text, parts in zip(seen, result['context_parts']):
            self.assertEqual(sum(parts.values()), len(text))
            self.assertLessEqual(len(text), 8500)
