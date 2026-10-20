"""AspectDecoder run() behavior — empty-batch must NOT write an O trace.

The decoder's O trace ('aspect_scan') marks "I observed unclassified work."
Writing it on empty batches caused a runaway S2 cascade in production
(rolled back 2026-05-08, see coordinator.py). The fix is an early-out
BEFORE the trace write. This test locks the contract so the regression
can't sneak back in.
"""

from unittest.mock import patch

from tests.brain_test_base import BrainTestBase
from servers.scales.s2.aspect_contract import ASPECT
from servers.scales.s2.aspect_decoder import AspectDecoder

# What the real _call_llm returns beside the tokens: the K stamp it resolved.
_STUB_STAMP = {'fingerprint': 'stub', 'source': 'default', 'version': 0, 'id': None}


def _count_aspect_scan_traces(brain):
    rows = brain._trace_dal.conn.execute(
        "SELECT COUNT(*) FROM trace_events "
        "WHERE scale='s2' AND ref_type='aspect_scan'"
    ).fetchone()
    return rows[0] if rows else 0


class TestAspectDecoderEmptyBatch(BrainTestBase):
    """Empty-batch decoder is a true no-op — no proposals, no trace."""

    needs_embedder = False

    def _count_aspect_scan_traces(self):
        return _count_aspect_scan_traces(self.brain)

    def test_shipped_defaults(self):
        # A change here is a cost / latency / quality decision (brain nodes
        # dea14302, d736cc49): make it deliberately.
        from servers.scales.s2.aspect_contract import ASPECT_INTERACTION_DEFAULT
        self.assertEqual((ASPECT['min_pending_to_call'], ASPECT['call_now_at_count']), (20, 5))
        self.assertEqual(ASPECT_INTERACTION_DEFAULT,
                         {'model': 'claude-opus-5-5', 'effort': 'medium', 'max_tokens': 32000})

    def test_empty_batch_returns_skipped(self):
        # Fresh BrainTestBase has only the 14 seeded aspect-nodes; their
        # types ('aspect') are classified in aspects_v1.json. So the
        # decoder finds nothing unclassified.
        decoder = AspectDecoder(self.brain)
        result = decoder.run()

        self.assertEqual(result.get('proposals'), [])
        self.assertEqual(result.get('skipped'), 'nothing to classify or fill')

    def test_empty_batch_writes_no_trace(self):
        # The original bug: O trace fired even when batch was empty,
        # downstream units treated it as fresh work.
        before = self._count_aspect_scan_traces()
        AspectDecoder(self.brain).run()
        after = self._count_aspect_scan_traces()

        self.assertEqual(after, before,
                         'aspect_scan trace must NOT be written on empty batch '
                         '— this is the cascade-prevention contract')

    def test_unclassified_string_DOES_write_trace(self):
        # Symmetric proof: when there IS work, the O trace fires as expected.
        # Add a node with an unclassified type, above min_count_threshold.
        threshold = AspectDecoder(self.brain).config['min_count_threshold']
        for i in range(threshold + 1):
            self.brain.remember(
                type='zzz_novel_unclassified_type',
                title='novel-type node %d' % i,
                content='content %d' % i,
                source='test',
            )

        before = self._count_aspect_scan_traces()
        config = dict(ASPECT, min_pending_to_call=1)      # one string is enough here
        result = AspectDecoder(self.brain, config=config).run()
        after = self._count_aspect_scan_traces()

        self.assertGreater(len(result.get('proposals', [])), 0,
                           'unclassified type should produce a proposal')
        self.assertEqual(after, before + 1,
                         'aspect_scan trace fires when batch is non-empty')


class TestAspectDecoderFillLane(BrainTestBase):
    """The unfilled lane: a classified verb this brain has written whose
    member carries no `orientation` is proposed with `fill: [orientation]`
    and its fixed `aspects`; once the value is stored it is not. Each test
    owns its aspects working copy so a write here never leaks into the
    session-wide guard file."""

    needs_embedder = False

    def setUp(self):
        import json
        import os
        import shutil
        import tempfile
        from servers.aspect_store import SEED_ASPECTS_JSON_PATH
        self._tmp = tempfile.mkdtemp()
        self._orig_env = os.environ.get('ASPECTS_JSON_PATH')
        self._work = os.path.join(self._tmp, 'aspects_v1.json')
        shutil.copy(SEED_ASPECTS_JSON_PATH, self._work)
        os.environ['ASPECTS_JSON_PATH'] = self._work
        super().setUp()
        # A pre-orientation working copy — bare strings, the state every brain
        # is in before the fill lane runs — so the lane has work to do here.
        # Stripped AFTER boot: the registry reconciles against the oriented repo
        # seed at construction and job 5 would refill anything stripped before.
        with open(self._work) as f:
            data = json.load(f)
        for name, entry in data.items():
            if name.startswith('_'):
                continue
            entry['edge_relations'] = [m['verb'] if isinstance(m, dict) else m
                                       for m in entry.get('edge_relations', []) or []]
        with open(self._work, 'w') as f:
            json.dump(data, f)
        self.brain.aspects._load()
        self.assertIsNone(self.brain.aspects.fact_of('corrects', 'orientation'))
        # These tests drive the lanes with one to three strings; the batch
        # gate would hold them back, so it is off here and tested on its own
        # below with an explicit config.
        gate_off = patch.dict(ASPECT, {'min_pending_to_call': 1})
        gate_off.start()
        self.addCleanup(gate_off.stop)

    def tearDown(self):
        import os
        import shutil
        super().tearDown()
        if self._orig_env is None:
            os.environ.pop('ASPECTS_JSON_PATH', None)
        else:
            os.environ['ASPECTS_JSON_PATH'] = self._orig_env
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _edge(self, relation, n=1):
        for i in range(n):
            a = self.brain.remember(type='lesson', title='%s src %d' % (relation, i),
                                    content='source content', source='test')
            b = self.brain.remember(type='lesson', title='%s tgt %d' % (relation, i),
                                    content='target content', source='test')
            self.brain.connect(a['id'], b['id'], relation)

    def test_bare_classified_verb_is_proposed_for_fill(self):
        self._edge('corrects')
        result = AspectDecoder(self.brain).run()
        fills = [p for p in result['proposals'] if p['value'] == 'corrects']
        self.assertEqual(len(fills), 1, result['proposals'])
        p = fills[0]
        self.assertEqual(p['fill'], ['orientation'])
        self.assertEqual(p['aspects'],
                         list(self.brain.aspects.homes_of('corrects', 'edge_relations')))
        self.assertIn('correction_improvement', p['aspects'])
        self.assertTrue(p['examples'], 'a fill candidate carries its graph evidence')
        self.assertEqual(result['stats']['unfilled'], 1)

    def test_verb_absent_from_this_graph_is_not_proposed(self):
        # 200+ seed verbs are bare, but the lane reads the graph: no edge, no
        # examples, nothing to judge from — the seed carries their value.
        result = AspectDecoder(self.brain).run()
        self.assertEqual(result.get('skipped'), 'nothing to classify or fill')

    def test_noise_member_is_not_proposed_for_fill(self):
        # co_anchored is noise-vetoed machinery: no semantic claim, no
        # dimension to fill, and it would otherwise lead every backlog by count.
        self._edge('co_anchored', n=3)
        self.assertIn('co_anchored', self.brain.aspects.structural_exclusions)
        result = AspectDecoder(self.brain).run()
        self.assertEqual(result.get('skipped'), 'nothing to classify or fill')

    def test_oriented_verb_is_not_proposed(self):
        self._edge('corrects')
        self.brain.aspects.add_members(
            [{'category': 'edge_relations', 'value': 'corrects',
              'aspects': ['correction_improvement'], 'orientation': 'source_actor'}],
            source='test')
        result = AspectDecoder(self.brain).run()
        self.assertEqual(result.get('skipped'), 'nothing to classify or fill')

    def test_cleared_verb_is_proposed_again(self):
        # A reviewed wrong value is cleared, not hand-set: the lane then
        # re-proposes the verb and the classifier derives the value anew.
        self._edge('corrects')
        self.brain.aspects.add_members(
            [{'category': 'edge_relations', 'value': 'corrects',
              'aspects': ['correction_improvement'], 'orientation': 'target_actor'}],
            source='test')
        self.brain.aspects.clear_fact('corrects', 'orientation', source='test')
        fills = [p for p in AspectDecoder(self.brain).run()['proposals']
                 if p['value'] == 'corrects']
        self.assertEqual([p['fill'] for p in fills], [['orientation']])

    def test_unclassified_first_then_fills_under_their_own_cap(self):
        self._edge('corrects', n=3)
        self._edge('supersedes', n=2)
        self._edge('zzz_brand_new_rel', n=1)
        config = dict(AspectDecoder(self.brain).config, max_fill_per_call=1)
        result = AspectDecoder(self.brain, config=config).run()
        values = [p['value'] for p in result['proposals']]
        # the new string leads despite the lowest count; one fill rides behind
        self.assertEqual(values, ['zzz_brand_new_rel', 'corrects'])
        self.assertEqual(result['proposals'][0]['fill'], ['aspects', 'orientation'])
        self.assertNotIn('aspects', result['proposals'][0])
        self.assertEqual(result['remaining'], 1)          # supersedes waits

    def test_round_trip_decode_classify_write(self):
        """A stubbed model answer flows decode → validate → add_members: the
        verb's member becomes an object in every home, the registry reads the
        value, and the next scan has nothing to fill."""
        from servers.scales.s2.aspect_encoder import AspectEncoder
        from servers.scales.s2.aspect_integration import AspectIntegration
        self._edge('supersedes')
        homes = self.brain.aspects.homes_of('supersedes', 'edge_relations')
        self.assertEqual(len(homes), 3, homes)
        seen = {}

        def fake_call(self_, interaction_name, user_content, journal=False):
            seen['prompt'] = user_content
            return ({'classifications': [
                {'category': 'edge_relations', 'value': 'supersedes',
                 'orientation': 'source_actor',
                 'rationale': 'orientation: the source is what stands afterwards.'}]},
                    {'elapsed_ms': 1, 'model': 'stub', 'input_tokens': 1, 'output_tokens': 1,
                     'interaction': _STUB_STAMP})

        orig = AspectEncoder._call_llm
        AspectEncoder._call_llm = fake_call
        try:
            result = AspectIntegration(self.brain).run()
        finally:
            AspectEncoder._call_llm = orig
        self.assertEqual(result.get('classified'), 1, result)
        self.assertEqual(result.get('calls'), 1, result)
        menu, body = seen['prompt']                 # two blocks, stable first
        self.assertIn('ASPECT MENU', menu)
        self.assertIn('FIELD DEFINITIONS', menu)
        self.assertNotIn('CANDIDATES', menu)
        self.assertIn('  fill: orientation\n', body)
        self.assertIn('  aspects: %s\n' % ', '.join(homes), body)
        self.assertEqual(self.brain.aspects.fact_of('supersedes', 'orientation'),
                         'source_actor')
        import json
        with open(self._work) as f:
            data = json.load(f)
        for h in homes:
            self.assertIn({'verb': 'supersedes', 'orientation': 'source_actor'},
                          data[h]['edge_relations'], h)
        self.assertEqual(AspectDecoder(self.brain).run().get('skipped'),
                         'nothing to classify or fill')

    def _stub(self, answers, fail_on=None):
        """Patch the model call: `answers` is {value: orientation}; the stub
        answers every candidate it is shown, records each call's blocks, and
        raises the failure path (returns None) on call number `fail_on`."""
        from servers.scales.s2.aspect_encoder import AspectEncoder
        calls = []

        def fake_call(self_, interaction_name, user_content, journal=False):
            calls.append(user_content)
            if fail_on and len(calls) == fail_on:
                return None, {'elapsed_ms': 1, 'model': 'stub', 'interaction': _STUB_STAMP,
                              'failure': 'LLM call failed'}
            import re
            body = user_content[1]
            values = re.findall(r'^  value: "(.*)"$', body, re.M)
            return ({'classifications': [
                {'category': 'edge_relations', 'value': v, 'rationale': 'orientation: stub.',
                 **({'orientation': answers[v]} if v in answers else {})}
                for v in values]},
                    {'elapsed_ms': 1, 'model': 'stub', 'input_tokens': 1, 'output_tokens': 1,
                     'interaction': _STUB_STAMP})
        orig = AspectEncoder._call_llm
        AspectEncoder._call_llm = fake_call
        self.addCleanup(setattr, AspectEncoder, '_call_llm', orig)
        return calls

    def test_cycle_loops_batches_until_nothing_is_left(self):
        """Fill cap 1, three bare verbs, budget 6: three calls in one cycle,
        the menu block byte-identical across them, one audit for the cycle
        with every row carrying its fill."""
        import json
        from servers.scales.s2.aspect_contract import aspects_proposed_path
        from servers.scales.s2.aspect_integration import AspectIntegration
        self._edge('corrects', n=3)
        self._edge('supersedes', n=2)
        self._edge('resolves', n=1)
        calls = self._stub({'corrects': 'source_actor', 'supersedes': 'source_actor',
                            'resolves': 'source_actor'})
        config = dict(AspectIntegration(self.brain).config, max_fill_per_call=1)
        result = AspectIntegration(self.brain, config=config).run()
        self.assertEqual(result['calls'], 3, result)
        self.assertEqual(result['classified'], 3)
        self.assertEqual(result['remaining'], 0)
        self.assertEqual(len({c[0] for c in calls}), 1, 'menu block must not change between batches')
        self.assertEqual(len({c[1] for c in calls}), 3, 'candidate block changes per batch')
        for v in ('corrects', 'supersedes', 'resolves'):
            self.assertEqual(self.brain.aspects.fact_of(v, 'orientation'), 'source_actor', v)
        with open(aspects_proposed_path()) as f:
            audit = json.load(f)
        rows = audit['classifications_accepted']
        self.assertEqual(sorted(r['value'] for r in rows), ['corrects', 'resolves', 'supersedes'])
        self.assertTrue(all(r['fill'] == ['orientation'] for r in rows), rows)
        self.assertEqual(AspectDecoder(self.brain).run().get('skipped'),
                         'nothing to classify or fill')

    def test_cycle_stops_at_budget_and_on_failure(self):
        from servers.scales.s2.aspect_integration import AspectIntegration
        self._edge('corrects', n=3)
        self._edge('supersedes', n=2)
        self._edge('resolves', n=1)
        answers = {'corrects': 'source_actor', 'supersedes': 'source_actor',
                   'resolves': 'source_actor'}
        # budget of 2 leaves one verb for the next cycle
        self._stub(answers)
        config = dict(AspectIntegration(self.brain).config, max_fill_per_call=1,
                      max_calls_per_cycle=2)
        result = AspectIntegration(self.brain, config=config).run()
        self.assertEqual((result['calls'], result['classified'], result['remaining']), (2, 2, 1))
        # a failed call ends the cycle — no blind retry loop
        calls = self._stub(answers, fail_on=1)
        result = AspectIntegration(self.brain, config=dict(config, max_calls_per_cycle=6)).run()
        self.assertEqual(result['calls'], 1, result)
        self.assertEqual(result['errors'], ['LLM call failed'])
        self.assertEqual(result['error'], 'LLM call failed')   # the coordinator's key
        self.assertIsNone(self.brain.aspects.fact_of('resolves', 'orientation'))

    # ── batch gate ──

    def _gated(self, **over):
        return {**ASPECT, 'min_pending_to_call': 3, 'call_now_at_count': 5, **over}

    def test_gate_waits_below_min_pending_and_says_so_in_a_trace(self):
        import json
        self._edge('corrects')
        self._edge('supersedes')
        before = _count_aspect_scan_traces(self.brain)
        result = AspectDecoder(self.brain, config=self._gated()).run()
        self.assertEqual(result['proposals'], [])
        self.assertEqual(result['skipped'], 'waiting for a batch (2 of 3 pending)')
        # it observed work, so unlike the empty batch it leaves an O trace
        self.assertEqual(_count_aspect_scan_traces(self.brain), before + 1)
        summary, meta = self.brain._trace_dal.conn.execute(
            "SELECT summary, metadata FROM trace_events WHERE ref_type='aspect_scan' "
            "ORDER BY rowid DESC LIMIT 1").fetchone()
        self.assertIn('waiting for a batch (2 of 3 pending)', summary)
        meta = json.loads(meta)
        self.assertEqual((meta['pending'], meta['gated'], meta['min_pending_to_call']), (2, True, 3))

    def test_forced_cycle_runs_what_waits_below_the_gate(self):
        from servers.scales.s2.aspect_integration import AspectIntegration
        self._edge('corrects')
        calls = self._stub({'corrects': 'source_actor'})
        unit = AspectIntegration(self.brain, config=self._gated())
        unit.forced = True
        result = unit.run()
        self.assertEqual((result['calls'], result['classified']), (1, 1), result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.brain.aspects.fact_of('corrects', 'orientation'), 'source_actor')

    def test_run_s2_hands_force_to_every_unit(self):
        from unittest.mock import patch as _patch
        from servers.scales.s2.aspect_integration import AspectIntegration
        seen = []

        def _record(unit):
            seen.append(unit.forced)
            return {'actions': 0}
        with _patch.object(AspectIntegration, 'run', autospec=True, side_effect=_record):
            self.brain.run_s2(units=['aspect_integration'])
            self.brain.run_s2(units=['aspect_integration'], force=True)
        self.assertEqual(seen, [False, True])


    def test_gate_opens_at_min_pending(self):
        for rel in ('corrects', 'supersedes', 'resolves'):
            self._edge(rel)
        result = AspectDecoder(self.brain, config=self._gated()).run()
        self.assertEqual(sorted(p['value'] for p in result['proposals']),
                         ['corrects', 'resolves', 'supersedes'])

    def test_gate_opens_for_one_string_the_brain_leans_on(self):
        self._edge('corrects', n=5)                 # call_now_at_count
        result = AspectDecoder(self.brain, config=self._gated(min_pending_to_call=20)).run()
        self.assertEqual([p['value'] for p in result['proposals']], ['corrects'])

    def test_gate_holds_the_whole_cycle(self):
        from servers.scales.s2.aspect_integration import AspectIntegration
        self._edge('corrects')
        calls = self._stub({'corrects': 'source_actor'})
        result = AspectIntegration(self.brain, config=self._gated()).run()
        self.assertEqual(result, {'actions': 0, 'skipped': 'waiting for a batch (1 of 3 pending)',
                                  'stats': result['stats']})
        self.assertEqual(calls, [])

    def test_cycle_that_called_drains_below_the_gate(self):
        """The gate opens the cycle; later passes run with the prefix cached,
        so the last verb goes too instead of waiting for the next batch."""
        from servers.scales.s2.aspect_integration import AspectIntegration
        for rel, n in (('corrects', 3), ('supersedes', 2), ('resolves', 1)):
            self._edge(rel, n=n)
        calls = self._stub({'corrects': 'source_actor', 'supersedes': 'source_actor',
                            'resolves': 'source_actor'})
        config = self._gated(max_fill_per_call=1)      # one verb per pass
        result = AspectIntegration(self.brain, config=config).run()
        self.assertEqual((result['calls'], result['classified'], result['remaining']), (3, 3, 0))
        self.assertEqual(len(calls), 3)

    def test_cycle_stops_when_a_pass_writes_nothing(self):
        # Every entry leaves its field open: accepted, but the door writes 0.
        # The next decode would propose the same batch — one call, not six.
        from servers.scales.s2.aspect_integration import AspectIntegration
        self._edge('corrects', n=2)
        self._edge('supersedes', n=1)
        calls = self._stub({})            # answers no field for anyone
        config = dict(AspectIntegration(self.brain).config, max_fill_per_call=1)
        result = AspectIntegration(self.brain, config=config).run()
        self.assertEqual(result['calls'], 1, result)
        self.assertNotIn('error', result)
        self.assertEqual(len(calls), 1)
