"""Failed batches remain retryable through suppression, scan gates and scheduling."""
import os
from contextlib import contextmanager
from unittest.mock import patch

from tests.brain_test_base import BrainTestBase
from servers.scales.s2.base import IntegrationUnit
from servers.scales.s2.community import CommunityDetection
from servers.scales.s2.community_decoder import CommunityDecoder
from servers.scales.s2.community_contract import COMMUNITY_DETECTION
from servers.scales.s2.consolidation import Consolidation
from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
from servers.scales.s2.consolidation_contract import CONSOLIDATION
from servers.scales.s2.rejection_table import filter_rejected


class TestBatchCompletion(BrainTestBase):
    needs_embedder = False

    def node(self, title, **kw):
        return self.brain.remember(
            title=title, type=kw.pop('type', 'finding'),
            content=kw.pop('content', title), **kw)['id']

    @contextmanager
    def model(self, respond):
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.runner.make_client', return_value=object()), \
                patch('servers.scales.runner.run_llm_loop', side_effect=respond), \
                patch.object(self.brain, 'recall', return_value={'results': []}):
            yield

    def rejection(self):
        # These words in the model's reasoning are not execution failures.
        return {'rounds': 1, 'actions': 0, 'write_actions': 0,
                'final_text': 'ERROR handling is relevant, but the old test FAILED. '
                              'Reject this proposal.\n## Review\n```\n```'}

    def test_partial_run_config_preserves_the_resolved_batch_budget(self):
        from tests.interaction_override import interaction_override
        from servers.scales.s2.community_encoder import CommunityEncoder
        with interaction_override(self.brain, 's2_community', parameters={
                'max_batch_context_chars': 12345}):
            encoder = CommunityEncoder(self.brain, config={'max_proposals_per_call': 1})
        self.assertEqual(encoder.config['max_batch_context_chars'], 12345)
        self.assertEqual(encoder.config['max_proposals_per_call'], 1)

    def test_batch_failure_cannot_be_erased_by_later_success(self):
        unit = IntegrationUnit(self.brain)
        total = {'rounds': 0, 'actions': 0, 'write_actions': 0,
                 'action_details': [], 'read_calls': [], 'final_text': ''}
        self.assertFalse(unit._fold_batch_result(
            total, {'rounds': 2, 'error': 'provider unavailable'}, 1, 'test'))
        self.assertTrue(unit._fold_batch_result(
            total, {'rounds': 1, 'write_actions': 1, 'actions': 1}, 2, 'test'))
        self.assertIn('batch 1: provider unavailable', total['error'])
        self.assertEqual(total['write_actions'], 1)
        self.assertEqual(total['rounds'], 3)

    def test_zero_round_batch_is_an_error(self):
        unit = IntegrationUnit(self.brain)
        total = {'rounds': 0, 'actions': 0, 'write_actions': 0,
                 'action_details': [], 'read_calls': [], 'final_text': ''}
        self.assertFalse(unit._fold_batch_result(total, {'rounds': 0}, 1, 'test'))
        self.assertIn('no completed rounds', total['error'])

    def community_scan(self):
        targets = [self.node('Community %d' % i, type='community',
                             content='A concrete community account. ' * 1100)
                   for i in range(2)]
        pending = [self.node('Pending %d' % i) for i in range(2)]
        proposals = [{'type': 'health_update', 'community_id': cid,
                      'community_title': 'Community %d' % i,
                      'signal': 'corridor_maturing',
                      'fraction': 0.4}
                     for i, cid in enumerate(targets)]
        return {'proposals': proposals,
                'community_state': [{'id': cid, 'members': [], 'meta': {}}
                                    for cid in targets],
                'pending_probes': [{'type': 'unplaceable', 'node_id': nid,
                                    'neighborhood': ''} for nid in pending]}

    def test_split_community_failure_retries_without_new_graph_changes(self):
        # Real default context budget forces two calls. Exercise failures in
        # either order, then a completed retry through the same idle gate.
        for failed_batch in (0, 1):
            with self.subTest(failed_batch=failed_batch):
                decoded = self.community_scan()
                unit = CommunityDetection(self.brain, config={
                    **COMMUNITY_DETECTION, 'min_run_interval_seconds': 0})
                self.brain.set_config(unit.LAST_RUN_KEY, '1')
                calls = []

                def respond(**kw):
                    calls.append(kw['user_content'])
                    if len(calls) - 1 == failed_batch:
                        raise RuntimeError('provider unavailable')
                    return self.rejection()

                with patch.object(CommunityDecoder, 'run', return_value=decoded), \
                        self.model(respond):
                    result = unit.run()
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(len(c) <= 48000 for c in calls))
                self.assertIn('provider unavailable', result['error'])
                self.assertEqual(self.brain.get_config(unit.LAST_RUN_KEY), '1')
                for proposals in (decoded['proposals'], decoded['pending_probes']):
                    self.assertEqual(filter_rejected(self.brain, proposals)[1], 0)
                self.assertIsNone(unit._should_skip())

                # No new node or edge is needed to make the retry eligible.
                with patch.object(CommunityDecoder, 'run', return_value=decoded), \
                        self.model(lambda **kw: self.rejection()):
                    result = unit.run()
                self.assertFalse(result.get('error'))
                self.assertNotEqual(self.brain.get_config(unit.LAST_RUN_KEY), '1')
                for proposals in (decoded['proposals'], decoded['pending_probes']):
                    self.assertEqual(filter_rejected(self.brain, proposals)[1], 2)

    def test_community_partial_write_survives_later_batch_failure(self):
        decoded = self.community_scan()
        cid = decoded['community_state'][0]['id']
        nid = decoded['pending_probes'][0]['node_id']
        decoded['proposals'][0] = {'type': 'add_to_existing', 'node_id': nid,
                                   'communities': [{'id': cid}]}
        operations = [{'op': 'connect', 'source_id': cid, 'target_id': nid,
                       'relation': 'community_member'}]
        calls = []

        def respond(**kw):
            calls.append(kw['user_content'])
            if len(calls) == 2:
                raise RuntimeError('second batch unavailable')
            response = kw['dispatch_fn']('brain_batch', {'operations': operations})
            self.assertTrue(response['ok'])
            return {'rounds': 2, 'actions': 1, 'write_actions': 1,
                    'action_details': [{'tool': 'brain_batch',
                                        'input': {'operations': operations}}]}

        unit = CommunityDetection(self.brain)
        with patch.object(CommunityDecoder, 'run', return_value=decoded), \
                self.model(respond):
            result = unit.run()
        self.assertIn('second batch unavailable', result['error'])
        self.assertEqual(result['actions'], 1)
        self.assertIn(cid, [c['id'] for c in self.brain._graph.get_communities_for([nid])[nid]])
        self.assertIsNone(self.brain.get_config(unit.LAST_RUN_KEY))

    def test_community_setup_exception_does_not_advance_cutoff(self):
        unit = CommunityDetection(self.brain)
        with patch.object(CommunityDecoder, 'run', side_effect=RuntimeError('read failed')):
            with self.assertRaisesRegex(RuntimeError, 'read failed'):
                unit.run()
        self.assertIsNone(self.brain.get_config(unit.LAST_RUN_KEY))

    def test_consolidation_partial_failure_preserves_fingerprints_and_cutoff(self):
        ids = [self.node('Member %d' % i) for i in range(4)]
        clusters = [{'nodes': ids[i:i + 2], 'size': 2,
                     'node_details': self.brain.get_node(ids[i:i + 2]),
                     'content_cosine_max': 0.95, 'title_cosine_max': 0.9,
                     'pre_class': 'needs_judgment'} for i in (0, 2)]
        decoded = {'clusters': clusters, 'stats': {},
                   '_stamp': {'ts': 12345.0, 'threshold': '0.89'}}
        unit = Consolidation(self.brain, config={
            **CONSOLIDATION, 'max_proposals_per_call': 1})
        self.brain.set_config(unit.LAST_RUN_TS_KEY, '1')
        self.brain.set_config(unit.LAST_THRESHOLD_KEY, '0.8')
        calls = []

        def respond(**kw):
            calls.append(kw['user_content'])
            if len(calls) == 2:
                raise RuntimeError('provider unavailable')
            return self.rejection()

        with patch.object(ConsolidationDecoder, 'run', return_value=decoded), \
                self.model(respond):
            result = unit.run()
        self.assertEqual(len(calls), 2)
        self.assertIn('provider unavailable', result['error'])
        self.assertEqual(self.brain.get_config(unit.LAST_RUN_TS_KEY), '1')
        self.assertEqual(self.brain.get_config(unit.LAST_THRESHOLD_KEY), '0.8')
        self.assertEqual(self.brain.conn.execute('SELECT COUNT(*) FROM s2_rejections').fetchone()[0], 0)
        with patch.object(ConsolidationDecoder, 'run', return_value=decoded), \
                self.model(lambda **kw: self.rejection()):
            result = unit.run()
        self.assertFalse(result.get('error'))
        self.assertEqual(result['skipped_recorded'], 2)
        self.assertEqual(self.brain.get_config(unit.LAST_RUN_TS_KEY), '12345.0')

    def test_coordinator_counts_returned_and_raised_errors_and_recovers(self):
        from servers.scales.s2.aspect_integration import AspectIntegration
        from servers.scales.s2.healer import Healer
        key = 's2_community_detection_consecutive_failures'
        partial = {'error': 'batch 2 unavailable', 'actions': 3}
        with patch.object(AspectIntegration, 'run', return_value={'skipped': 'empty'}), \
                patch.object(Healer, 'run', return_value={'skipped': 'empty'}), \
                patch.object(Consolidation, 'run', return_value={'skipped': 'empty'}), \
                patch.object(CommunityDetection, 'run', side_effect=[
                    partial, RuntimeError('raised failure'), partial,
                    {'skipped': 'throttled'}, {'actions': 0}]), \
                patch.object(self.brain, '_log_error') as log:
            for expected in ('1', '2', '3', '3', '0'):
                result = self.brain.run_s2()['units']['community_detection']
                self.assertEqual(self.brain.get_config(key), expected)
                if result.get('error') == partial['error']:
                    self.assertEqual(result['actions'], 3)
        self.assertTrue(any(c.args[0] == 's2_community_detection_persistent_failure'
                            for c in log.call_args_list))


class TestRunsRecordTheirInteraction(BrainTestBase):
    """Each S2 unit's run delta names the interaction IT resolved — the
    required stamp proves one is present, this proves it is the right one."""
    needs_embedder = False
    node = TestBatchCompletion.node
    model = TestBatchCompletion.model
    rejection = TestBatchCompletion.rejection
    community_scan = TestBatchCompletion.community_scan

    def delta(self, ref_type):
        # The run's delta is the one carrying the run record; the unit also
        # writes bare marker deltas under the same ref_type.
        events = self.brain.query_traces(
            scale='s2', event_type='delta', ref_type=ref_type,
            hours=None, limit=50)['events']
        runs = [e['metadata'] for e in events
                if isinstance(e.get('metadata'), dict) and 'rounds' in e['metadata']]
        self.assertTrue(runs, 'no %s run delta written' % ref_type)
        return runs[0]

    def fingerprint(self, name):
        return self.brain.get_interaction_stamp(name)['fingerprint']

    def test_community_records_its_prompt_and_its_config(self):
        unit = CommunityDetection(self.brain)
        with patch.object(CommunityDecoder, 'run', return_value=self.community_scan()), \
                self.model(lambda **kw: self.rejection()):
            unit.run()
        md = self.delta('community_enriched')
        self.assertEqual(md['interaction_fingerprint'],
                         self.fingerprint('s2_community_enrichment'))
        self.assertEqual(md['config_interaction_fingerprint'],
                         self.fingerprint('s2_community'))

    def run_community(self, **kw):
        """One community cycle; returns (the rounds budget each encoder call
        ran on, the names the resolver was asked for during the cycle)."""
        rounds = []

        def respond(**call):
            rounds.append(call['max_rounds'])
            return self.rejection()

        with patch.object(self.brain, '_resolve_interaction',
                          wraps=self.brain._resolve_interaction) as spy, \
                patch.object(CommunityDecoder, 'run',
                             return_value=self.community_scan()), \
                self.model(respond):
            CommunityDetection(self.brain, **kw).run()
        return rounds, [c.args[0] for c in spy.call_args_list]

    def test_community_resolves_its_config_once_per_cycle(self):
        # One read feeds the decoder, the encoder and the stamp — two reads
        # could straddle a pointer flip and run version N while recording N+1.
        from tests.interaction_override import interaction_override
        with interaction_override(self.brain, 's2_community',
                                  parameters={'max_rounds': 2}) as version:
            stamp = self.brain.get_interaction_stamp('s2_community')
            rounds, resolved = self.run_community()
        self.assertEqual(resolved.count('s2_community'), 1)
        self.assertEqual(set(rounds), {2})
        md = self.delta('community_enriched')
        self.assertEqual(md['config_interaction_fingerprint'], stamp['fingerprint'])
        self.assertEqual(md['config_interaction_source'], 'override')
        self.assertEqual(md['config_interaction_version'], version)

    def test_community_stamps_a_caller_config_as_what_ran(self):
        # A caller's partial config lays over the resolved one in decoder and
        # encoder alike, and the stamp names the config that ran.
        from servers.interaction_defaults import interaction_fingerprint
        rounds, _ = self.run_community(config={'max_rounds': 2})
        self.assertEqual(set(rounds), {2})
        md = self.delta('community_enriched')
        self.assertEqual(md['config_interaction_fingerprint'], interaction_fingerprint(
            's2_community', '', {**COMMUNITY_DETECTION, 'max_rounds': 2}))
        self.assertEqual(md['config_interaction_source'], 'caller')

    def test_consolidation_records_its_prompt(self):
        ids = [self.node('Member %d' % i) for i in range(2)]
        decoded = {'clusters': [{'nodes': ids, 'size': 2,
                                 'node_details': self.brain.get_node(ids),
                                 'content_cosine_max': 0.95, 'title_cosine_max': 0.9,
                                 'pre_class': 'needs_judgment'}],
                   'stats': {}, '_stamp': {'ts': 12345.0, 'threshold': '0.89'}}
        with patch.object(ConsolidationDecoder, 'run', return_value=decoded), \
                self.model(lambda **kw: self.rejection()):
            Consolidation(self.brain).run()
        self.assertEqual(self.delta('consolidated')['interaction_fingerprint'],
                         self.fingerprint('s2_consolidation_enrichment'))

    def test_healer_records_its_prompt(self):
        from servers.scales.s2.healer_encoder import HealerEncoder
        nid = self.node('Needs a question')
        tel = {'elapsed_ms': 1, 'model': 'stub', 'input_tokens': 1,
               'output_tokens': 1, 'cache_read_tokens': 0,
               'cache_creation_tokens': 0}
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.s2.base.run_llm_once', return_value=('[]', tel)), \
                patch('servers.scales.s2.base.make_client', return_value=object()):
            HealerEncoder(self.brain).run([{'node_id': nid, 'needs_question': True}])
        self.assertEqual(self.delta('healer_generated')['interaction_fingerprint'],
                         self.fingerprint('s2_healer'))

    def test_healer_resolves_its_prompt_once_per_run(self):
        # Every batch runs on one read and the run records it — a read per
        # batch could run version N on early batches and record N+1.
        from servers.scales.s2.healer_encoder import HealerEncoder
        from servers.scales.s2.healer_contract import HEALER
        ids = [self.node('Needs a question %d' % i) for i in range(3)]
        tel = {'elapsed_ms': 1, 'model': 'stub', 'input_tokens': 1,
               'output_tokens': 1, 'cache_read_tokens': 0,
               'cache_creation_tokens': 0}
        with patch.object(self.brain, '_resolve_interaction',
                          wraps=self.brain._resolve_interaction) as spy, \
                patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.s2.base.run_llm_once',
                      return_value=('[]', tel)) as llm, \
                patch('servers.scales.s2.base.make_client', return_value=object()):
            HealerEncoder(self.brain, config={**HEALER, 'max_nodes_per_call': 1}).run(
                [{'node_id': nid, 'needs_question': True} for nid in ids])
        self.assertEqual(llm.call_count, 3)
        self.assertEqual([c.args[0] for c in spy.call_args_list].count('s2_healer'), 1)
        self.assertEqual(self.delta('healer_generated')['interaction_fingerprint'],
                         self.fingerprint('s2_healer'))

    def test_an_override_run_records_the_override(self):
        from servers.scales.s2.healer_encoder import HealerEncoder
        from tests.interaction_override import interaction_override
        nid = self.node('Needs a situation')
        tel = {'elapsed_ms': 1, 'model': 'stub', 'input_tokens': 1,
               'output_tokens': 1, 'cache_read_tokens': 0,
               'cache_creation_tokens': 0}
        with interaction_override(self.brain, 's2_healer',
                                  template='Heal the fields.') as version, \
                patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.s2.base.run_llm_once', return_value=('[]', tel)), \
                patch('servers.scales.s2.base.make_client', return_value=object()):
            stamp = self.brain.get_interaction_stamp('s2_healer')
            HealerEncoder(self.brain).run([{'node_id': nid, 'needs_situation': True}])
        md = self.delta('healer_generated')
        self.assertEqual(md['interaction_source'], 'override')
        self.assertEqual(md['interaction_version'], version)
        self.assertEqual(md['interaction_fingerprint'], stamp['fingerprint'])

    def test_aspect_records_its_prompt(self):
        from servers.scales.s2.aspect_encoder import AspectEncoder
        tel = {'elapsed_ms': 1, 'model': 'stub', 'input_tokens': 1,
               'output_tokens': 1, 'cache_read_tokens': 0,
               'cache_creation_tokens': 0}
        proposals = [{'category': 'edge_relations', 'value': 'stub_rel',
                      'count': 1, 'examples': []}]
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                patch('servers.scales.s2.base.run_llm_once',
                      return_value=('{"classifications": []}', tel)), \
                patch('servers.scales.s2.base.make_client', return_value=object()):
            AspectEncoder(self.brain).run(proposals)
        self.assertEqual(self.delta('aspect_classified')['interaction_fingerprint'],
                         self.fingerprint('s2_aspects'))
