"""Decode Pipeline Transition Tests — verify wiring between stages.

Each test checks that the output of one pipeline stage has the shape
the next stage expects. These catch silent breakage when a format
changes in one place but the consumer doesn't update.

Stages tested:
  remember → recall (embedding roundtrip)
  recall → build_surface_prompt (candidate fields)
  judge output → format_judge_output (voice surface)
  correction_enrich (correction chain lookup)
  recall → build_surface_prompt end-to-end (candidate format compatibility)
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestDecodeTransitions(BrainTestBase):
    """Tests for wiring between decode pipeline stages."""

    needs_embedder = True

    def test_remember_recall_embedding_roundtrip(self):
        """remember → recall: stored node must be retrievable by semantic query."""
        node = self.brain.remember(
            type='decision',
            title='Use PostgreSQL for the analytics database',
            content='Chose PostgreSQL over MongoDB for analytics because we need '
                    'complex joins and ACID transactions for financial data.',
            keywords='postgres analytics database',
        )
        node_id = node['id']

        result = self.brain.recall(query='which database for analytics', limit=10)
        results = result.get('results', [])
        found_ids = [r['id'] for r in results]

        self.assertIn(node_id, found_ids,
                      'Stored node not found in recall results — embedding storage or '
                      'retrieval is broken')

    def test_recall_result_shape_for_surface(self):
        """recall → surface: recall results must have all fields build_surface_prompt expects.

        build_surface_prompt calls format_candidate_for_surface which reads:
        id, type, title, content, confidence, locked, score (via effective_activation),
        keywords. If recall drops any of these, the surface gets malformed candidates.
        """
        self.brain.remember(
            type='rule',
            title='Always validate input before processing',
            content='Input validation prevents injection attacks and data corruption.',
            keywords='validation security input',
        )
        result = self.brain.recall(query='input validation', limit=5)
        results = result.get('results', [])
        self.assertTrue(len(results) > 0, 'No recall results — test data not stored')

        # Fields that format_candidate_for_surface reads from each candidate
        required_fields = {'id', 'type', 'title', 'content', 'confidence', 'locked'}
        node = results[0]
        missing = required_fields - set(node.keys())
        self.assertEqual(missing, set(),
                         'Recall result missing fields needed by surface: %s' % missing)

        # score comes from effective_activation — surface reads it as 'score'
        # The hook_recall code in daemon_hooks enriches candidates before passing
        # to build_surface_prompt. But the raw recall result must have the base fields.
        self.assertIn('effective_activation', node,
                      'Missing effective_activation — surface needs this as candidate score')

    def test_differential_project_threads_to_candidate_menu(self):
        """build_surface_prompt(scope=) → candidate render: a
        foreign-project candidate carries the ⚠ mark in the Haiku menu, a
        same-project candidate carries no project line. This pins the
        THREADING (the feature existing but nobody passing the parameter is
        the failure class), not just the render."""
        from servers.scales.s1.surface_contract import build_surface_prompt

        foreign = self.brain.remember(
            type='fact', title='People Inc deadline passed',
            content='External legal deadline expired.', project='acme-site')
        home = self.brain.remember(
            type='fact', title='Daemon restart shipped',
            content='Deploy verified.', project='brain')
        candidates = [self.brain.get_node(foreign['id']),
                      self.brain.get_node(home['id'])]
        for c in candidates:
            c['score'] = 0.8

        text, _ = build_surface_prompt(
            candidates, 'status of the initiative',
            scope={'project': 'brain'})
        self.assertIn('⚠ From another project: acme-site', text)
        self.assertNotIn('Project: brain', text)

        # Undeclared (unscoped session) → no marks, legacy render.
        text_unscoped, _ = build_surface_prompt(
            candidates, 'status of the initiative')
        self.assertNotIn('From another project', text_unscoped)

    def test_correction_enrich_finds_correction_chains(self):
        """brain.correction_enrich: node B corrects node A must be discoverable.

        The encoder writes correction-aspect edges (corrects, supersedes,
        reframes, ...). brain.correction_enrich walks those edges so the
        surface and downstream consumers can warn Claude about superseded
        knowledge.
        """
        node_a = self.brain.remember(
            type='decision',
            title='Use REST API for all endpoints',
            content='REST for everything, no GraphQL.',
        )
        node_b = self.brain.remember(
            type='decision',
            title='Use GraphQL for complex queries',
            content='Switched to GraphQL for queries needing joins across 3+ tables.',
        )
        # Edge: node_b corrects node_a
        self.brain.connect_typed(
            source_id=node_b['id'], target_id=node_a['id'],
            relation='corrects', weight=0.5,
            description='GraphQL handles the cross-table queries REST struggled with',
            encoding_source='test:correction_chains')

        # brain.correction_enrich finds node_a's correction (node_b corrects it)
        corrections = self.brain.correction_enrich({node_a['id']})

        # Result keyed by both full and short id — accept either
        chain = corrections.get(node_a['id']) or corrections.get(node_a['id'][:8])
        self.assertTrue(chain, 'correction_enrich did not find correction for node A')

        # Verify the correction info has the expected shape
        self.assertTrue(len(chain) > 0, 'No correction entries found')

        # Check all entries carry the required fields downstream consumers read
        for entry in chain:
            self.assertIn('id', entry, 'Correction entry missing "id"')
            self.assertIn('title', entry, 'Correction entry missing "title"')
            self.assertIn('direction', entry, 'Correction entry missing "direction"')
            self.assertIn(entry['direction'], ('corrector', 'corrected'),
                          'Unknown correction direction: %s' % entry['direction'])
            # Heavy payload now also carries relation + edge_description
            self.assertIn('relation', entry, 'Correction entry missing "relation"')
            self.assertIn('edge_description', entry,
                          'Correction entry missing "edge_description"')

        # node_b's ID must appear somewhere in the correction chain
        correction_ids = {e['id'] for e in chain}
        self.assertIn(node_b['id'][:8], correction_ids,
                      'Correcting node B not found in correction chain for node A')

    def test_correction_enrich_walks_every_replacement_verb(self):
        """Each replacement verb in the seed must actually surface end-to-end.

        The membership contract test asserts these verbs are in the
        correction_improvement aspect; this asserts the consequence — that a
        node joined by one of them surfaces its counterpart through
        correction_enrich. Membership alone passes the moment someone appends a
        string, even if the aspect plumbing or the relation filter in
        get_connections_bulk regresses underneath it. The verb list is the same
        constant the membership contract test pins (REPLACEMENT_VERBS in
        tests/test_aspects_contract.py), so adding a verb there covers it here
        too — one place, not two tuples drifting apart.
        """
        from tests.test_aspects_contract import REPLACEMENT_VERBS
        for verb in REPLACEMENT_VERBS:
            with self.subTest(relation=verb):
                old = self.brain.remember(
                    type='handoff', title='Opener superseded by %s' % verb,
                    content='The previous state, replaced via %s.' % verb)
                new = self.brain.remember(
                    type='handoff', title='Opener that replaced via %s' % verb,
                    content='The current state, arrived via %s.' % verb)
                # Written the way the verb reads: an inverse-voice verb
                # (`old superseded_by new`) puts the replaced node first.
                passive = self.brain.aspects.fact_of(verb, 'orientation') == 'target_actor'
                src, tgt = (old, new) if passive else (new, old)
                self.brain.connect_typed(
                    source_id=src['id'], target_id=tgt['id'],
                    relation=verb, weight=0.5,
                    description='the newer opener replaces the older one',
                    encoding_source='test:replacement_verbs')

                enriched = self.brain.correction_enrich({old['id']})
                chain = (enriched.get(old['id'])
                         or enriched.get(old['id'][:8]) or [])
                self.assertTrue(
                    chain,
                    'correction_enrich found nothing for a node replaced via '
                    '%s — the replacement is invisible to recall' % verb)
                self.assertIn(
                    new['id'][:8], {e['id'] for e in chain},
                    'the replacing node is missing from the chain for %s' % verb)
                # The verb itself must survive the walk. Without this, a
                # relation-mislabel regression (emitting a generic 'related',
                # or collapsing the per-relation loop) keeps the id and loses
                # the meaning every consumer renders.
                self.assertIn(
                    verb, {e.get('relation') for e in chain},
                    'chain for %s carries the wrong relation label' % verb)

    def test_recall_candidates_feed_into_surface_prompt(self):
        """recall → build_surface_prompt: recall output must produce a valid surface prompt.

        This is the full transition: recall results (enriched as candidates)
        passed through build_surface_prompt. The prompt must contain all candidate
        IDs and not raise errors. Tests that the recall output shape is compatible
        with the surface input shape.
        """
        from servers.scales.s1.surface_contract import build_surface_prompt

        # Store several nodes to get real recall results
        self.brain.remember(
            type='decision', title='Deploy to AWS us-east-1',
            content='Primary region is us-east-1 for latency to east coast users.',
            keywords='aws deploy region')
        self.brain.remember(
            type='mechanism', title='Blue-green deployment strategy',
            content='Zero-downtime deploys using blue-green with ALB target group switching.',
            keywords='deploy blue green zero downtime')

        result = self.brain.recall(query='how do we deploy', limit=10)
        results = result.get('results', [])
        self.assertTrue(len(results) > 0, 'No recall results for judge prompt test')

        # Enrich candidates the way hook_recall does: add 'score' from effective_activation
        candidates = []
        for r in results:
            c = dict(r)
            c['score'] = c.get('effective_activation', 0)
            candidates.append(c)

        # build_surface_prompt should not raise and should produce a string.
        # 2026-05-03: session_context parameter removed (per-session leak fix +
        # Frame replaces it). Frame becomes the prior; here we pass empty Frame
        # since the test is about candidate-shape compatibility, not Frame content.
        prompt, max_tokens = build_surface_prompt(
            candidates,
            user_message='how do we deploy to production?',
            recent_messages=[
                {'role': 'user', 'content': 'tell me about our deploy process'},
                {'role': 'assistant', 'content': 'Let me check the brain for deployment info.'},
            ],
        )

        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 100,
                        'Surface prompt suspiciously short: %d chars' % len(prompt))
        self.assertIsInstance(max_tokens, int)

        # Every candidate ID (first 8 chars) should appear in the prompt
        for c in candidates:
            short_id = str(c['id'])[:8]
            self.assertIn(short_id, prompt,
                          'Candidate %s not found in surface prompt — '
                          'format_candidate_for_surface dropped it' % short_id)

    def test_current_message_is_own_block_not_history(self):
        """build_surface_prompt: current prompt renders once, in its own block.

        Regression pin for the 94b4642 fallout: the user_message trace is
        written at prompt-arrival, so upstream excludes the current chain and
        this function renders `user_message` ONLY in the "Current message"
        block — never appended to the history window (which duplicated it
        and ate a history slot). Window = SURFACE['recent_messages'] previous
        messages, newest kept.
        """
        from servers.scales.s1.surface_contract import build_surface_prompt, SURFACE

        window = SURFACE['recent_messages']
        # More history than the window holds — oldest must fall off.
        # Zero-padded labels: 'message 01' must never be a substring of
        # 'message 010' if the window config grows past single digits.
        history = []
        for i in range(window + 2):
            role = 'user' if i % 2 == 0 else 'assistant'
            history.append({'role': role, 'content': 'history message %03d' % i})

        current = 'CURRENT-PROMPT-SENTINEL what changed in the deploy?'
        prompt, _ = build_surface_prompt([], user_message=current,
                                         recent_messages=history)

        # Exactly one occurrence, on the turn marked current. (Asserted the
        # legacy "Current message" block until 2026-09-21; the invariant is
        # the same, the shape is the one layout that ships.)
        self.assertEqual(prompt.count('CURRENT-PROMPT-SENTINEL'), 1,
                         'current prompt must appear exactly once')
        self.assertGreater(prompt.index('CURRENT-PROMPT-SENTINEL'),
                           prompt.index('current_msg="true"'),
                           'current prompt must render on the current turn')
        self.assertIn('<conversation>', prompt)

        # Window slice: newest `window` messages kept, oldest dropped.
        self.assertNotIn('history message 000', prompt)
        self.assertNotIn('history message 001', prompt)
        self.assertIn('history message 002', prompt)
        self.assertIn('history message %03d' % (window + 1), prompt)

    def test_xml_layout_structure(self):
        """layout='xml_v13': turn grouping, current_msg last, <shown> dedup,
        no score floats, candidate id-attribute grammar.
        """
        from servers.scales.s1.surface_contract import build_surface_prompt

        n = self.brain.remember(
            type='decision', title='XML layout ships behind interaction config',
            content='Layout rides in the surface interaction parameters.')
        node = self.brain.get_node(n['id'] if isinstance(n, dict) else n)
        candidate = dict(node)
        candidate['score'] = 0.87
        candidate['discovery'] = 'embedding'

        history = [
            {'role': 'user', 'content': 'lets fix the window',
             'surfaced': [{'id': 'aaaabbbb', 'title': 'window fix decision'}]},
            {'role': 'assistant', 'content': 'Done, tests green.'},
            {'role': 'assistant', 'content': 'lone assistant (envelope-filtered user half)'},
            {'role': 'user', 'content': 'now the XML restructure'},
        ]
        prompt, _ = build_surface_prompt(
            [candidate], user_message='CURRENT-XML-SENTINEL ready to eval?',
            recent_messages=history)

        # Structure: all four turn groups render; current turn is last and marked.
        self.assertIn('<conversation>', prompt)
        self.assertIn('current_msg="true"', prompt)
        last_turn_pos = prompt.rindex('<turn ')
        self.assertIn('current_msg="true"', prompt[last_turn_pos:prompt.index('>', last_turn_pos) + 1],
                      'the last turn must be the current_msg turn')
        self.assertEqual(prompt.count('CURRENT-XML-SENTINEL'), 1)

        # <shown> rides the turn it surfaced for, same id-attribute grammar.
        # Id only — the dedup rule matches on the id; the title was unread
        # weight and is gone (2026-09-21).
        self.assertIn('<shown id="aaaabbbb"/>', prompt)
        self.assertNotIn('window fix decision</shown>', prompt)

        # Lone assistant gets its own turn (no strict alternation assumption).
        self.assertIn('<assistant>lone assistant', prompt)

        # Candidates: id attribute, no score floats, no legacy #N header.
        short_id = str(candidate['id'])[:8]
        self.assertIn('<candidate id="%s"' % short_id, prompt)
        self.assertNotIn('match:', prompt)
        self.assertNotIn('#1', prompt)

        # Legacy sections must not leak into the XML layout.
        self.assertNotIn('Recently surfaced', prompt)
        self.assertNotIn('Query type:', prompt)

    def test_fetch_tool_names_in_sync(self):
        """SURFACE_FETCH_TOOLS must cover exactly the tools Haiku can fire —
        a new tool that isn't listed would render without source_tool, and a
        recall MODE (laf_v1) must never count as tool provenance."""
        from servers.scales.s1.surface_contract import SURFACE_FETCH_TOOLS
        from servers.scales.s1.fetch_tools import TOOL_DEFINITIONS
        tool_names = {t['name'] for t in TOOL_DEFINITIONS}
        self.assertEqual(tool_names, set(SURFACE_FETCH_TOOLS))

    def test_xml_source_tool_only_for_real_tools(self):
        """laf_v1 (recall mode) must not render source_tool; tool names must."""
        from servers.scales.s1.surface_contract import format_candidate_for_surface
        base = {'id': 'aaaabbbbcccc', 'type': 'fact', 'title': 't', 'content': 'c',
                'confidence': 1.0, 'created_at': '2026-07-01T00:00:00+00:00'}
        laf = format_candidate_for_surface(dict(base, discovery='laf_v1'))
        self.assertNotIn('source_tool', laf)
        tool = format_candidate_for_surface(
            dict(base, discovery='recall_topical'))
        self.assertIn('source_tool="true"', tool)

    def test_layout_config_reaches_the_capture_stamp(self):
        """Transition: interaction config → the replay capture's layout
        stamp. The layout the resolver hands back (K-store override overlaid
        on the code default) must be the one the capture records —
        sentinel-checked because a wrong overlay is silent."""
        from servers.scales.s1 import surface as surface_mod
        from servers.scales.s1 import surface_capture

        self.brain.logs_conn.execute('DELETE FROM interactions')
        self.brain.logs_conn.commit()
        self.brain._interaction_dal.register(
            'surface', template='',
            parameters=json.dumps({'layout': 'sentinel-layout'}))
        self.brain._interaction_dal.set_active('surface', 1, set_by='test')

        captured = {}

        class _Stop(Exception):
            pass

        def capturing_begin(*args, **kwargs):
            captured['layout'] = kwargs.get('layout')
            raise _Stop()

        real_begin = surface_capture.begin
        surface_capture.begin = capturing_begin
        try:
            with self.assertRaises(_Stop):
                surface_mod._call_surface(
                    self.brain, [], 'a message', [], 'sess-layout', {})
        finally:
            surface_capture.begin = real_begin

        self.assertEqual(captured['layout'], 'sentinel-layout')

    def test_archived_selection_dropped_live_kept(self):
        from servers.scales.s1.surface import _drop_archived_selected

        live = self.brain.remember(type='test', title='gate_live',
                                   content='c',
                                   encoding_source='anchor:test')
        dead = self.brain.remember(type='test', title='gate_dead',
                                   content='c',
                                   encoding_source='anchor:test')
        arch = self.brain.archive_node(dead['id'], archived_by='anchor:test',
                                       reason='liveness gate test')
        self.assertTrue(arch.get('ok'))

        selected_mode = {live['id']: 'arc', dead['id']: 'arc'}

        dropped = _drop_archived_selected(self.brain, selected_mode)

        self.assertEqual(dropped, [dead['id']])
        self.assertNotIn(dead['id'], selected_mode)
        self.assertIn(live['id'], selected_mode)

    def test_all_live_selection_untouched(self):
        from servers.scales.s1.surface import _drop_archived_selected

        a = self.brain.remember(type='test', title='gate_a', content='c',
                                encoding_source='anchor:test')
        selected_mode = {a['id']: 'arc'}

        dropped = _drop_archived_selected(self.brain, selected_mode)

        self.assertEqual(dropped, [])
        self.assertIn(a['id'], selected_mode)

    def test_run_surface_drops_archived_end_to_end(self):
        """Wiring test: an archived node Haiku selects must not reach the
        surface_selected trace when it goes through the REAL run_surface path.

        The other gate tests call _drop_archived_selected directly — they'd
        still pass if the gate call were deleted from run_surface. This one
        drives run_surface with a canned Haiku selection (live + archived)
        and asserts the archived id never lands in the surface_selected
        trace — the source of the recently-surfaced block, so a leak here
        is the self-perpetuating dead-node loop. A wiring regression (gate
        removed / called in the wrong place / trace written before the
        gate) fails loudly. query_vec=None skips spread expansion, so no
        embedder is needed."""
        from servers.scales.s1 import surface as surface_mod

        live = self.brain.remember(type='test', title='wire_live', content='c',
                                   encoding_source='anchor:test')
        dead = self.brain.remember(type='test', title='wire_dead', content='c',
                                   encoding_source='anchor:test')
        self.assertTrue(self.brain.archive_node(
            dead['id'], archived_by='anchor:test',
            reason='wiring test').get('ok'))

        session_id = 'test-run-surface-wiring'
        ctx = self.brain.get_or_create_session(session_id)
        candidates_data = [
            {'id': live['id'], 'title': 'wire_live', 'type': 'test', 'score': 0.9},
            {'id': dead['id'], 'title': 'wire_dead', 'type': 'test', 'score': 0.9},
        ]

        # Canned Haiku selection: both the live node and the archived one.
        # 5-tuple matches the real _call_surface contract (the trailing dict is
        # the run-cost telemetry the K trace now carries; this wiring test
        # doesn't assert on it).
        def _fake_call_surface(brain, cands, user_message, recent_messages,
                               sid, result, frame='', scope=None):
            return ({'selected': [
                {'id': live['id'][:8], 'why': 'relevant'},
                {'id': dead['id'][:8], 'why': 'stale'},
            ]}, 'prompt', 100, brain.get_interaction_stamp('surface'),
                {'input_tokens': 50, 'output_tokens': 10, 'cache_read_tokens': 0,
                 'cache_creation_tokens': 0, 'elapsed_ms': 5, 'rounds': 1,
                 'truncated': 0})

        orig = surface_mod._call_surface
        surface_mod._call_surface = _fake_call_surface
        try:
            surface_mod.run_surface(
                self.brain, ctx, candidates_data, 'user msg', [], {},
                'enriched query', [], 'test-recall-ref', session_id, None,
                query_vec=None)
            evts = self.brain.query_traces(
                scale='s1', ref_type='surface_selected',
                session_id=session_id, hours=None).get('events') or []
            self.assertTrue(evts, 'no surface_selected K trace written')
            surfaced = set(json.loads(evts[0]['ref_id']))
            self.assertIn(live['id'][:8], surfaced,
                          'live node missing from surface_selected trace')
            self.assertNotIn(
                dead['id'][:8], surfaced,
                'archived node reached the surface_selected trace — '
                'liveness gate is not wired into run_surface')
        finally:
            surface_mod._call_surface = orig

    def test_run_surface_writes_cost_telemetry_to_k_trace(self):
        """Cost-telemetry gap closer: a real run_surface threads _call_surface's
        run-cost dict (input/output tokens + elapsed_ms + rounds) FLAT into the
        surface_selected K trace via build_run_telemetry, and records the
        served/empty outcome — so surface cost is queryable from traces, not
        absent (the gap that made the recall-timeout diagnosis painful)."""
        from servers.scales.s1 import surface as surface_mod

        node = self.brain.remember(type='test', title='tel_node', content='c',
                                   encoding_source='anchor:test')
        session_id = 'test-surface-telemetry'
        ctx = self.brain.get_or_create_session(session_id)
        candidates_data = [
            {'id': node['id'], 'title': 'tel_node', 'type': 'test', 'score': 0.9}]

        # Canned Haiku selection + KNOWN telemetry (the 5-tuple contract). The
        # read_usage→telemetry mapping inside _call_surface is covered by the
        # builder/guard unit tests; this asserts the threading into the K trace.
        def _fake_call_surface(brain, cands, user_message, recent_messages,
                               sid, result, frame='', scope=None):
            return ({'selected': [{'id': node['id'][:8], 'why': 'relevant'}]},
                    'prompt', 100, brain.get_interaction_stamp('surface'),
                    {'input_tokens': 1234, 'output_tokens': 56,
                     'cache_read_tokens': 7, 'cache_creation_tokens': 0,
                     'elapsed_ms': 88, 'rounds': 1, 'truncated': 0})

        orig = surface_mod._call_surface
        surface_mod._call_surface = _fake_call_surface
        try:
            surface_mod.run_surface(
                self.brain, ctx, candidates_data, 'user msg', [], {},
                'enriched query', [], 'test-tel-ref', session_id, None,
                query_vec=None)
            evts = self.brain._trace_dal.get_by_ref_type(
                'surface_selected', scale='s1', hours=None, session_id=session_id)
            self.assertTrue(evts, 'no surface_selected K trace written')
            meta = evts[0]['metadata']
            self.assertEqual(meta.get('input_tokens'), 1234)
            self.assertEqual(meta.get('output_tokens'), 56)
            self.assertEqual(meta.get('cache_read_tokens'), 7)
            self.assertEqual(meta.get('elapsed_ms'), 88)
            self.assertEqual(meta.get('rounds'), 1)
            # Phase 4 fold-ins also land on the K trace.
            self.assertIn(meta.get('outcome'), ('served', 'empty'))
            self.assertIn('phase_timing', meta)   # [] when pt not threaded (this path)
        finally:
            surface_mod._call_surface = orig


class TestCorrectionDirection(BrainTestBase):
    """correction_enrich derives who corrected whom from the verb's
    orientation (aspect fact), not the stored arrow, and keeps only the
    neighbours that corrected this node — one entry per corrector. The nodes
    this one corrected are history and render as ordinary edges. Verbs are the
    repo seed's: `resolved_by` is target_actor, `resolves`/`supersedes`/
    `corrects` source_actor, `restates` bare."""

    needs_embedder = False

    def _pair(self):
        a = self.brain.remember(type='bug', title='Hook emits the wrong JSON shape',
                                content='additionalContext is silently dropped.')
        b = self.brain.remember(type='fix', title='Hook wraps hookSpecificOutput',
                                content='The documented shape reaches the model.')
        return a['id'], b['id']

    def _connect(self, src, tgt, verb, why):
        self.brain.connect_typed(source_id=src, target_id=tgt, relation=verb,
                                 weight=0.5, description=why,
                                 encoding_source='test:correction_direction')

    def _chain(self, enriched, nid):
        return enriched.get(nid) or []

    def test_passive_verb_warns_the_corrected_endpoint_only(self):
        """`A --resolved_by--> B` means B resolved A: A shows ⚠ Resolved by B
        (the arrow alone would say the reverse); B carries no warning — the
        edge is history for it."""
        from servers.contract import render_corrections
        self.assertEqual(self.brain.aspects.fact_of('resolved_by', 'orientation'),
                         'target_actor')
        bug, fix = self._pair()
        self._connect(bug, fix, 'resolved_by', 'the wrapper is the fix')
        enriched = self.brain.correction_enrich({bug, fix})

        on_bug = self._chain(enriched, bug)
        self.assertEqual([(e['id'], e['direction']) for e in on_bug],
                         [(fix[:8], 'corrector')])
        self.assertIn('⚠ Resolved by "', render_corrections(on_bug)[0])
        self.assertEqual(self._chain(enriched, fix), [])

    def test_unoriented_verb_renders_neutral(self):
        """A correction verb with no orientation names no actor: direction is
        None and the header is the neutral ⚠ Correction:, never a guess. The
        seed curates every unoriented verb it ships (`restates` is
        `unchanged`), so the test files a fresh bare one."""
        from servers.contract import render_corrections
        self.brain.aspects.add_members(
            [{'category': 'edge_relations', 'value': 'zz_unoriented_rel',
              'aspects': ['correction_improvement']}], source='test')
        self.assertIsNone(self.brain.aspects.fact_of('zz_unoriented_rel', 'orientation'))
        self.assertIsNone(self.brain.aspects.fact_of('zz_unoriented_rel', 'flavor'))
        a, b = self._pair()
        self._connect(a, b, 'zz_unoriented_rel', 'the same claim, reworded')
        enriched = self.brain.correction_enrich({a, b})
        for nid in (a, b):
            chain = self._chain(enriched, nid)
            self.assertEqual([e['direction'] for e in chain], [None])
            self.assertIn('⚠ Correction:', render_corrections(chain)[0])

    def test_unchanged_verb_is_an_ordinary_edge(self):
        """A verb curated `unchanged` changed nothing in the other node's
        claim: no ⚠ entry on either endpoint — the pair reads as an edge."""
        self.assertEqual(self.brain.aspects.fact_of('addresses', 'flavor'), 'unchanged')
        self.assertEqual(self.brain.aspects.fact_of('restates', 'flavor'), 'unchanged')
        a, b = self._pair()
        self._connect(b, a, 'addresses', 'the wrapper is being tried on the bug')
        self._connect(a, b, 'restates', 'the same claim, reworded')
        enriched = self.brain.correction_enrich({a, b})
        self.assertEqual(self._chain(enriched, a), [])
        self.assertEqual(self._chain(enriched, b), [])

    def test_strongest_flavor_first(self):
        """Entries come strongest flavor first — Replaced before Corrected
        before an uncurated corrector — so a reader that shows only some (the
        inject's cap) keeps the ones that change what the reader does."""
        from servers.contract import render_corrections
        self.brain.aspects.add_members(
            [{'category': 'edge_relations', 'value': 'zz_uncurated_fix',
              'aspects': ['correction_improvement'], 'orientation': 'source_actor'}],
            source='test')
        old = self.brain.remember(type='decision', title='Old rollout order',
                                  content='auth first.')['id']
        others = [self.brain.remember(type='decision', title=t, content='c.')['id']
                  for t in ('Uncurated fixer', 'Corrector', 'Replacement')]
        # written weakest first, so edge order alone would put them backwards
        for src, verb in zip(others, ('zz_uncurated_fix', 'corrects', 'supersedes')):
            self._connect(src, old, verb, 'why %s' % verb)
        chain = self._chain(self.brain.correction_enrich({old}), old)
        self.assertEqual([e['flavor'] for e in chain], ['replaced', 'corrected', None])
        lines = render_corrections(chain, mode='lean', limit=2)
        self.assertIn('⚠ Replaced by "', lines[0])
        self.assertIn('⚠ Corrected by "', lines[1])
        self.assertIn('+1 more correction', lines[2])

    def test_agreeing_verbs_are_one_corrector(self):
        """Two verbs on one arrow that resolve to the same direction are ONE
        entry — verbs and descriptions joined — so lean mode prints one line
        and the inject's correction limit counts one corrector."""
        from servers.contract import render_corrections
        old, new = self._pair()
        self._connect(new, old, 'resolves', 'the wrapper fixes the shape')
        self._connect(new, old, 'supersedes', 'the old shape is retired')
        chain = self._chain(self.brain.correction_enrich({old}), old)

        self.assertEqual(len(chain), 1, chain)
        entry = chain[0]
        self.assertEqual(entry['direction'], 'corrector')
        self.assertEqual(set(entry['relation'].split(', ')), {'resolves', 'supersedes'})
        for why in ('the wrapper fixes the shape', 'the old shape is retired'):
            self.assertIn(why, entry['edge_description'])
        self.assertEqual(len(render_corrections(chain, mode='lean')), 1)

    def test_entry_reads_like_its_edges_line(self):
        """The ⚠ line shows what the Edges line for the same relation shows —
        the neighbour's type and the relation's age in the reference, the verb
        after a dash, the description whole on its own line — in the reader's
        separator (the legacy ', ' when the reader names none)."""
        from servers.contract import NODE_REF_SEP, relation_age, render_corrections
        bug, fix = self._pair()
        self._connect(fix, bug, 'resolves', 'the wrapper is the fix')
        entry = self._chain(self.brain.correction_enrich({bug}), bug)[0]
        conn = next(c for c in self.brain.get_node(bug)['connections']
                    if c['id'] == fix)
        self.assertEqual(entry['relation_at'], relation_age(conn['relations'][0], conn))
        self.assertTrue(entry['relation_at'])

        lines = render_corrections([entry], mode='balanced',
                                   cfg={'node_ref_sep': NODE_REF_SEP})
        self.assertEqual(lines[0], '  ⚠ Resolved by "Hook wraps hookSpecificOutput" '
                                   '(id:%s · fix · %s) — resolves'
                         % (fix[:8], entry['relation_at'][:10]))
        self.assertEqual(lines[1], '     description: the wrapper is the fix')
        self.assertIn('(id:%s, fix, ' % fix[:8],
                      render_corrections([entry], mode='balanced')[0])

    def test_claim_against_the_pairs_first_order_names_its_actor(self):
        """Who acts comes from the claim's own ends (actor_of), not from the
        order the pair was first written in: `a supersedes b` hung on a pair
        born `b extends a` still makes a the one that replaced b."""
        a, b = self._pair()
        self._connect(b, a, 'extends', 'b builds on a')
        self._connect(a, b, 'supersedes', 'a replaces b')
        enriched = self.brain.correction_enrich({a, b})
        self.assertEqual([(e['id'], e['direction'], e['flavor'])
                          for e in self._chain(enriched, b)],
                         [(a[:8], 'corrector', 'replaced')])
        self.assertEqual(self._chain(enriched, a), [])

    def test_agreeing_claims_written_both_ways_are_one_corrector(self):
        """`new resolves old` and `old resolved_by new` name the same actor
        from opposite ends. The DAL hands them over as two entries (one per
        direction); the ⚠ block still shows one corrector, so a capped reader
        does not spend two slots on it."""
        old, new = self._pair()
        self._connect(new, old, 'resolves', 'the wrapper fixes the shape')
        self._connect(old, new, 'resolved_by', 'the shape is fixed by the wrapper')
        enriched = self.brain.correction_enrich({old, new})
        chain = self._chain(enriched, old)
        self.assertEqual(len(chain), 1, chain)
        self.assertEqual(chain[0]['direction'], 'corrector')
        self.assertEqual(set(chain[0]['relation'].split(', ')), {'resolves', 'resolved_by'})
        self.assertEqual(self._chain(enriched, new), [])

    def test_disagreeing_verbs_warn_both_endpoints(self):
        """`corrects` and `corrected_by` on one arrow claim opposite actors;
        the data contradicts itself, so each endpoint is warned by the verb
        that names the other as its corrector — not merged, not dropped."""
        a, b = self._pair()
        self._connect(a, b, 'corrects', 'a corrects b')
        self._connect(a, b, 'corrected_by', 'b corrects a')
        enriched = self.brain.correction_enrich({a, b})
        self.assertEqual([(e['direction'], e['relation'])
                          for e in self._chain(enriched, a)],
                         [('corrector', 'corrected_by')])
        self.assertEqual([(e['direction'], e['relation'])
                          for e in self._chain(enriched, b)],
                         [('corrector', 'corrects')])


class TestSelectedIdRecovery(BrainTestBase):
    """2026-07-03 — Haiku occasionally emits whitespace-corrupted ids in its
    selection JSON ('9 9a 2e ' for candidate 99a2e…). The raw [:8] lookup
    matched nothing, the pick died silently, and the surfaced context came
    out empty (eval run v12_1_full, item d7c942c3-r1). The resolution layer
    must sanitize + unique-prefix-recover against the candidate pool, and
    any id that still resolves nowhere must log a drift warning the
    scoreboard's drift section counts."""

    needs_embedder = False

    def _run_with_selection(self, session_id, candidates_data, selection):
        """Drive the REAL run_surface with a canned Haiku selection; return
        the surfaced-ids set the surface_selected K trace carries."""
        from servers.scales.s1 import surface as surface_mod

        ctx = self.brain.get_or_create_session(session_id)

        def _fake_call_surface(brain, cands, user_message, recent_messages,
                               sid, result, frame='', scope=None):
            return ({'selected': selection}, 'prompt', 100, brain.get_interaction_stamp('surface'),
                    {'input_tokens': 50, 'output_tokens': 10,
                     'cache_read_tokens': 0, 'cache_creation_tokens': 0,
                     'elapsed_ms': 5, 'rounds': 1, 'truncated': 0})

        orig = surface_mod._call_surface
        surface_mod._call_surface = _fake_call_surface
        try:
            surface_mod.run_surface(
                self.brain, ctx, candidates_data, 'user msg', [], {},
                'enriched query', [], 'test-recall-ref', session_id, None,
                query_vec=None)
            evts = self.brain.query_traces(
                scale='s1', ref_type='surface_selected',
                session_id=session_id, hours=None).get('events') or []
            return set(json.loads(evts[0]['ref_id'])) if evts else set()
        finally:
            surface_mod._call_surface = orig

    def _warnings(self, source):
        rows = self.brain.logs_conn.execute(
            "SELECT metadata FROM debug_log "
            "WHERE source = ? AND event_type = 'warning'", (source,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def _two_candidates(self, tag):
        a = self.brain.remember(type='test', title='%s_a' % tag, content='c',
                                encoding_source='anchor:test')
        b = self.brain.remember(type='test', title='%s_b' % tag, content='c',
                                encoding_source='anchor:test')
        cands = [
            {'id': a['id'], 'title': '%s_a' % tag, 'type': 'test', 'score': 0.9},
            {'id': b['id'], 'title': '%s_b' % tag, 'type': 'test', 'score': 0.8},
        ]
        return a, b, cands

    def test_space_corrupted_full_id_sanitized(self):
        """All 8 chars present but space-riddled → sanitize alone recovers,
        and the file carries the REAL short id, not the corrupted emission."""
        a, _b, cands = self._two_candidates('sani')
        nid8 = a['id'][:8]
        corrupted = ' '.join(nid8[i:i + 2] for i in range(0, 8, 2))

        on_disk = self._run_with_selection(
            'test-id-sanitize', cands,
            [{'id': corrupted, 'why': 'relevant'}])

        self.assertEqual(on_disk, {nid8},
                         'sanitized id did not resolve to its candidate')
        self.assertFalse(self._warnings('surface_unknown_selected_id'),
                         'exact sanitized match must not log unknown-id drift')

    def test_space_corrupted_fragment_prefix_recovered(self):
        """Corruption dropped chars ('d 6d3 f8' shape): the sanitized
        fragment is a unique prefix of one candidate → recovered + warned."""
        a, _b, cands = self._two_candidates('fuzz')
        nid8 = a['id'][:8]
        frag = nid8[:6]
        corrupted = '%s %s %s' % (frag[0], frag[1:4], frag[4:6])

        on_disk = self._run_with_selection(
            'test-id-fuzzy', cands,
            [{'id': corrupted, 'why': 'relevant'}])

        self.assertEqual(on_disk, {nid8},
                         'unique-prefix fragment was not recovered')
        recov = self._warnings('surface_id_fuzzy_recovered')
        self.assertTrue(recov, 'fuzzy recovery must log its drift warning')
        self.assertIn(nid8, recov[-1]['message'])

    def test_selection_schema_id_pattern(self):
        """Layer-1 source fix: the structured-outputs schema constrains the
        id field so constrained decoding cannot emit the whitespace
        corruption the parse layer recovers from. Pins the pattern's
        semantics AND that the schema actually carries it."""
        import re
        from servers.scales.s1.surface_contract import (
            SURFACE_SELECTION_SCHEMA, SURFACE_SELECTED_ID_PATTERN)

        id_prop = (SURFACE_SELECTION_SCHEMA['properties']['selected']
                   ['items']['properties']['id'])
        self.assertEqual(id_prop.get('pattern'), SURFACE_SELECTED_ID_PATTERN,
                         'schema id field lost its pattern constraint')

        pat = re.compile(SURFACE_SELECTED_ID_PATTERN)
        # The observed corruption class must be unrepresentable.
        for bad in ('9 9a 2e ', 'd 6d3 f8', '', 'zzzz9999', 'a3f0c5e1x',
                    'A3F0C5E1'):
            self.assertIsNone(pat.match(bad),
                              'pattern must reject %r' % bad)
        # Known-legitimate emissions stay representable: full 8-char,
        # 7-char leading-zero drop, short honest fragments (>=4).
        for good in ('a3f0c5e1', '95c2b96', 'd6d3'):
            self.assertIsNotNone(pat.match(good),
                                 'pattern must accept %r' % good)

    def test_unknown_id_dropped_with_drift_warning(self):
        """An id that matches no candidate and no brain node is dropped —
        but never silently: surface_unknown_selected_id must fire."""
        a, _b, cands = self._two_candidates('unkn')
        nid8 = a['id'][:8]

        on_disk = self._run_with_selection(
            'test-id-unknown', cands, [
                {'id': nid8, 'why': 'relevant'},
                {'id': 'zzzz 9999', 'why': 'confabulated'},
            ])

        self.assertEqual(on_disk, {nid8},
                         'good pick lost or unknown id leaked into the file')
        warns = self._warnings('surface_unknown_selected_id')
        self.assertTrue(warns, 'unknown selected id was dropped silently')
        self.assertIn('zzzz 9999', warns[-1]['message'])


class TestPresentationShuffle(unittest.TestCase):
    """§20.12 A2 (2026-07-14) — the candidate menu Haiku sees is a seeded
    shuffle of the scorer's ranking. These pin the contract: presentation
    order shuffles, scorer order (traces, caller's list) never does."""

    @staticmethod
    def _cands(n=12):
        return [{'id': ('%02x' % i) * 16, 'title': 'node %d' % i,
                 'type': 'fact', 'content': 'content %d' % i,
                 'score': 1.0 - i * 0.01,
                 'created_at': '2026-07-01T00:00:00+00:00'}
                for i in range(n)]

    def test_seeded_shuffle_deterministic_membership_preserved(self):
        from servers.scales.s1.surface_contract import (
            prepare_presented_candidates)
        cands = self._cands()
        a = prepare_presented_candidates(cands, shuffle_seed=42)
        b = prepare_presented_candidates(cands, shuffle_seed=42)
        plain = prepare_presented_candidates(cands, shuffle_seed=None)
        self.assertEqual([c['id'] for c in a], [c['id'] for c in b],
                         'same seed must reproduce the same order')
        self.assertEqual({c['id'] for c in a}, {c['id'] for c in plain},
                         'shuffle must not change top-N membership')
        self.assertNotEqual([c['id'] for c in a], [c['id'] for c in plain],
                            'seed 42 must actually permute 12 candidates')

    def test_none_seed_preserves_scorer_order(self):
        from servers.scales.s1.surface_contract import (
            prepare_presented_candidates)
        cands = self._cands()
        out = prepare_presented_candidates(cands, shuffle_seed=None)
        self.assertEqual([c['id'] for c in out], [c['id'] for c in cands])

    def test_caller_list_never_reordered(self):
        """candidates_data is the trace/mapping substrate — prepare must
        work on a copy, never mutate the caller's scorer-ordered list."""
        from servers.scales.s1.surface_contract import (
            prepare_presented_candidates)
        cands = self._cands()
        before = [c['id'] for c in cands]
        prepare_presented_candidates(cands, shuffle_seed=42)
        self.assertEqual([c['id'] for c in cands], before)

    def test_xml_prompt_renders_presented_order(self):
        """build_surface_prompt(shuffle_seed) must render candidate id
        attributes in exactly prepare_presented_candidates' order — the
        presented_order recorded in the K trace IS what Haiku saw."""
        import re
        from servers.scales.s1.surface_contract import (
            build_surface_prompt, prepare_presented_candidates)
        cands = self._cands()
        expected = [c['id'][:8] for c in
                    prepare_presented_candidates(cands, shuffle_seed=7)]
        prompt, _ = build_surface_prompt(
            cands, 'a question', shuffle_seed=7)
        rendered = re.findall(r'<candidate id="([0-9a-f]{8})"', prompt)
        self.assertEqual(rendered, expected)

    def test_turn_seed_stable_and_message_sensitive(self):
        from servers.scales.s1.surface_contract import (
            presentation_shuffle_seed)
        s1 = presentation_shuffle_seed('sess-a', 'hello')
        s2 = presentation_shuffle_seed('sess-a', 'hello')
        s3 = presentation_shuffle_seed('sess-a', 'other message')
        self.assertEqual(s1, s2, 'seed must be deterministic per turn')
        self.assertNotEqual(s1, s3)


class TestShuffleTraceRecord(BrainTestBase):
    """The K trace must carry shuffle_seed + presented_order (the exact
    menu Haiku saw) while the O trace keeps scorer order — the join that
    makes picked/dropped rows propensity-exact for the LAF P3 fit."""

    needs_embedder = False

    def test_run_surface_writes_presented_order_to_k_trace(self):
        from servers.scales.s1 import surface as surface_mod

        node = self.brain.remember(type='test', title='shuf_node', content='c',
                                   encoding_source='anchor:test')
        session_id = 'test-surface-shuffle'
        ctx = self.brain.get_or_create_session(session_id)
        candidates_data = [
            {'id': node['id'], 'title': 'shuf_node', 'type': 'test',
             'score': 0.9}]

        # Fake _call_surface that stashes the presentation record the way
        # the real one does (the stash, not the call, is under test here).
        def _fake_call_surface(brain, cands, user_message, recent_messages,
                               sid, result, frame='', scope=None):
            brain._surface_presented = {
                sid: {'shuffle_seed': 12345,
                      'presented_order': [node['id'][:8]]}}
            return ({'selected': [{'id': node['id'][:8], 'why': 'r'}]},
                    'prompt', 100, brain.get_interaction_stamp('surface'),
                    {'input_tokens': 10, 'output_tokens': 5,
                     'cache_read_tokens': 0, 'cache_creation_tokens': 0,
                     'elapsed_ms': 5, 'rounds': 1, 'truncated': 0})

        orig = surface_mod._call_surface
        surface_mod._call_surface = _fake_call_surface
        try:
            surface_mod.run_surface(
                self.brain, ctx, candidates_data, 'user msg', [], {},
                'enriched query', [], 'test-shuf-ref', session_id, None,
                query_vec=None)
            evts = self.brain._trace_dal.get_by_ref_type(
                'surface_selected', scale='s1', hours=None,
                session_id=session_id)
            self.assertTrue(evts, 'no surface_selected K trace written')
            meta = evts[0]['metadata']
            self.assertEqual(meta.get('shuffle_seed'), 12345)
            self.assertEqual(meta.get('presented_order'), [node['id'][:8]])
        finally:
            surface_mod._call_surface = orig


class TestSelectionDeltaShape(BrainTestBase):
    """The additionalContext delta a REAL run_surface writes satisfies the
    shape TraceDAL enforces for it — on a served turn and on an empty one.
    The builder unit tests would still pass if _write_traces stopped calling
    the builder or threaded an ill-typed stamp; this is the wiring check."""

    needs_embedder = False

    def _run(self, session_id, pick):
        import contextlib
        import io
        from servers.scales.s1 import surface as surface_mod

        node = self.brain.remember(type='test', title=session_id, content='c',
                                   auto_connect=False,
                                   encoding_source='anchor:test')
        ctx = self.brain.get_or_create_session(session_id)
        candidates_data = [
            {'id': node['id'], 'title': session_id, 'type': 'test', 'score': 0.9}]
        selection = [{'id': node['id'][:8], 'why': 'r'}] if pick else []

        # The real stamp, so the builder sees the production-typed values.
        def _fake_call_surface(brain, cands, user_message, recent_messages,
                               sid, result, frame='', scope=None):
            return ({'selected': selection}, 'prompt', 100,
                    brain.get_interaction_stamp('surface'),
                    {'input_tokens': 10, 'output_tokens': 5,
                     'cache_read_tokens': 0, 'cache_creation_tokens': 0,
                     'elapsed_ms': 5, 'rounds': 1, 'truncated': 0})

        orig = surface_mod._call_surface
        surface_mod._call_surface = _fake_call_surface
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                surface_mod.run_surface(
                    self.brain, ctx, candidates_data, 'user msg', [], {},
                    'enriched query', [], 'test-shape-ref', session_id, None,
                    query_vec=None)
        finally:
            surface_mod._call_surface = orig
        evts = self.brain._trace_dal.get_by_ref_type(
            'additionalContext', scale='s1', hours=None, session_id=session_id)
        # Not vacuous: "no warning" also describes "nothing was written".
        self.assertTrue(evts, 'no additionalContext delta written')
        self.assertNotIn('trace metadata invalid', err.getvalue())
        return evts[0]['metadata']

    def test_served_turn_delta_validates(self):
        meta = self._run('test-shape-served', pick=True)
        self.assertTrue(meta['content'])

    def test_empty_turn_delta_validates(self):
        meta = self._run('test-shape-empty', pick=False)
        self.assertEqual(meta['selected'], [])

    def _judge_output(self, session_id):
        """The K trace's pointer, followed to the judge payload it names."""
        evts = self.brain._trace_dal.get_by_ref_type(
            'surface_selected', scale='s1', hours=None, session_id=session_id)
        self.assertTrue(evts, 'no surface_selected K trace written')
        pointer = evts[0]['metadata'].get('payload_pointer')
        self.assertTrue(pointer, 'K trace carries no judge payload pointer')
        raw = self.brain.read_payload(pointer)
        self.assertIsNotNone(raw, 'pointer names no judge payload: %s' % pointer)
        return json.loads(raw)['surface_output']

    def test_k_trace_points_at_the_judge_payload(self):
        """The dashboard's Live feed takes a recall's output from the payload
        its K trace points at, on the one delivery it reads — so the payload
        is recorded before the trace batch, and its pointer rides K."""
        served = self._run('test-judge-served', pick=True)
        self.assertEqual(self._judge_output('test-judge-served'),
                         served['content'])
        self._run('test-judge-empty', pick=False)
        self.assertEqual(self._judge_output('test-judge-empty'),
                         '(no selection)')
