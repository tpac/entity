"""Tests for the spreading-activation kernel and activation-driven rendering.

Contract checks (fast, no LLM, no daemon):
  • Per-field vector split exists in EMBEDDING_GROUPS
  • _compose_enriched_edge_text produces expected text
  • _allocate_budget_softmax respects minimum per node + total budget

Behavioral checks (use r3 preserved brain from N=5 run — read-only):
  • spread_activation from Hawaii seed activates NYC
  • render_surface_inject produces rendered output with
    "5 days" (from NYC content) and "10 days" (from Hawaii content)
  • Trace metadata contains activation_count + kernel_trace

Skipped when:
  • r3 brain isn't preserved on disk
  • embedder can't load the nomic model

TestIterFamiliesShape (4 tests) removed 2026-05-04 — exercised iter_families
which lived in servers/scales/s2/edge_families.py, deleted alongside
EdgeFamilyIntegration in Step 12 of unified-aspects. Aspect data now flows
through brain.aspects.all() directly.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _r3_corpus():
    """The r3 eval corpus, if `$BRAIN_R3_CORPUS` says where it is.

    Named explicitly rather than derived. The old form walked the brain
    location ladder and took a sibling of wherever the operator's brain
    resolved to, which made a test file reach for that brain to find
    something that is not in it — the suite does not go there (see
    tests/test_isolated_brain.py). The classes below already skip on any
    machine without the corpus; now that includes a machine that has one but
    has not said so.
    """
    env = os.environ.get('BRAIN_R3_CORPUS')
    return os.path.expanduser(env) if env else ''


R3_BRAIN = _r3_corpus()
HAS_R3 = bool(R3_BRAIN) and os.path.isdir(R3_BRAIN)


# ─────────────────────────────────────────────────────────────────
# Contract tests — fast, no external state
# ─────────────────────────────────────────────────────────────────

class TestEmbeddingGroupsExtension(unittest.TestCase):
    """Part C: the field-cohort EXTENSION specifics — vector ordering + fallback.

    Cohort presence/membership and the field-weight=0 invariant are owned by
    test_pipeline_contract.py::TestEmbeddingGroups (test_cohort_assignment asserts
    exact legacy/field membership; test_cohort_weight_invariants asserts legacy>0
    AND field==0 — strictly stronger than the checks formerly duplicated here).
    Removed from this file:
      test_both_cohorts_present       — subset of test_cohort_assignment
      test_field_cohort_has_zero_weights — subset of test_cohort_weight_invariants
    """

    def test_field_vector_types_stable_order(self):
        from servers.pipeline_contract import field_vector_types
        order = field_vector_types()
        self.assertEqual(order, ['title', 'content', 'situation', 'reasoning',
                                 'their_raw_quote', 'my_raw_quote', 'question'])

    def test_field_fallback_chain_covers_all(self):
        from servers.pipeline_contract import FIELD_VECTOR_FALLBACK, field_vector_types
        for t in field_vector_types():
            self.assertIn(t, FIELD_VECTOR_FALLBACK)

    def test_situation_served_by_legacy_situation_vector(self):
        # Dedup: situation has no field-cohort vector; the kernel resolves it
        # to the legacy `_situation` vector first (then high_meta). There must
        # be no `field_situation` group writing a duplicate `situation` vector.
        from servers.pipeline_contract import FIELD_VECTOR_FALLBACK, EMBEDDING_GROUPS
        self.assertEqual(FIELD_VECTOR_FALLBACK['situation'][0], '_situation')
        self.assertNotIn('field_situation', EMBEDDING_GROUPS)
        self.assertFalse(any(g.get('vector_type') == 'situation'
                             for g in EMBEDDING_GROUPS.values()))


class TestEnrichedEdgeText(unittest.TestCase):
    """Edge text composition (post-v26): relation + description only.

    The composer lives at AspectRegistry.compose_edge_text. The embedded text
    intentionally EXCLUDES two things:
      - partner_title — including it would cascade-stale every edge embedding
        when a partner node's title revised; partner content contributes via
        the partner node's own embedding.
      - the relation's aspect-family `meaning` — verbose classifier guidance
        (authored for AspectIntegration) that, baked into every edge, dominated
        the description and blunted per-edge disambiguation. It still feeds the
        classifier via aspects_v1.json, just not the embedding geometry.
    """

    def test_composes_relation_and_description(self):
        from servers.aspects import AspectRegistry
        # compose_edge_text no longer consults aspects — call it duck-typed
        # with a bare object as `self`.
        text = AspectRegistry.compose_edge_text(
            object(), 'contextualizes', 'Hawaii is the contrastive frame')
        self.assertEqual(text, '[contextualizes] Hawaii is the contrastive frame')
        # Family meaning is NOT embedded.
        self.assertNotIn('family:', text)
        # Partner title is NOT embedded.
        self.assertNotIn('NYC trip', text)

    def test_missing_parts_drop_silently(self):
        from servers.aspects import AspectRegistry
        text = AspectRegistry.compose_edge_text(object(), 'related', '')
        # No description — just [relation].
        self.assertEqual(text.strip(), '[related]')
        self.assertNotIn('family:', text)


class TestFieldMaskingRemoved(unittest.TestCase):
    """Contract evolution 2026-05-17 — _mask_node_by_field_activation removed.

    The renderer used to strip fields below a 0.3 cosine threshold per
    spread_activation's field_activation. That stripped voice quotes,
    reasoning, and situation from Haiku's primary picks systematically
    (short fields have intrinsically lower cosines than long fields).

    Render layer now trusts the encoder's attached fields — every field
    present on the node renders, subject only to char-budget truncation
    in render_rich_node. Field selection is the encoder's job, not the
    renderer's.

    This test guards the contract: the masking function must stay deleted.
    """

    def test_mask_function_removed(self):
        from servers.scales.s1 import surface_contract
        self.assertFalse(
            hasattr(surface_contract, '_mask_node_by_field_activation'),
            'Field-masking renderer regression — function must stay deleted. '
            'Render layer trusts encoder-attached fields; restoring this '
            'function would re-introduce systematic voice/reasoning stripping.')

    def test_field_render_threshold_removed(self):
        from servers.scales.s1 import surface_contract
        self.assertFalse(
            hasattr(surface_contract, '_FIELD_RENDER_THRESHOLD'),
            'Threshold constant must stay deleted alongside the mask function.')


# ─────────────────────────────────────────────────────────────────
# Behavioral tests — need r3 preserved brain + embedder
# ─────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_R3, 'r3 preserved brain not on disk (run N=5 first)')
class TestSpreadActivationOnR3(unittest.TestCase):
    """End-to-end: spread_activation from Hawaii activates NYC (the multi_session
    case that failed pre-redesign)."""

    @classmethod
    def setUpClass(cls):
        from servers import embedder
        if embedder._model is None:
            embedder.load_model()
        cls.tmp = tempfile.mkdtemp(prefix='test_spread_')
        for f in os.listdir(R3_BRAIN):
            shutil.copy2(os.path.join(R3_BRAIN, f), cls.tmp)
        from servers.brain import Brain
        cls.brain = Brain(db_path=os.path.join(cls.tmp, 'brain.db'))
        cls.brain.backfill_vectors(batch_size=50)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.brain.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_nyc_activates_from_hawaii_seed(self):
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '62f666c0'").fetchone()[0]
        nyc = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '4e8acb7a'").fetchone()[0]

        query = ("How many days did I spend in total traveling "
                 "in Hawaii and in New York City?")
        qv = np.frombuffer(embed_query(query), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        self.assertIn(nyc, result['node_activation'],
            'NYC did not activate — spreading failed for multi_session case')
        self.assertGreater(result['node_activation'][nyc], 0.5,
            f'NYC activation too weak: {result["node_activation"][nyc]}')

    def test_community_bridge_activates_on_relevant_query(self):
        """Community activates when query semantics match community meaning.
        Query-dependent by design: 'travel style' triggers the Solo Travel
        Profile community naturally; a narrow day-counting query wouldn't."""
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '62f666c0'").fetchone()[0]
        community = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '3d25ca4c'").fetchone()[0]

        query = "What's my travel style?"
        qv = np.frombuffer(embed_query(query), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        self.assertIn(community, result['node_activation'],
            'community bridge did not activate for a style-relevant query')

    def test_field_activations_populated_for_activated_nodes(self):
        """Each activated node should have per-field activations for rendering."""
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '62f666c0'").fetchone()[0]
        qv = np.frombuffer(embed_query("Hawaii trip"), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        # At least the seed should have field activations
        self.assertIn(hawaii, result['field_activation'])
        self.assertGreater(len(result['field_activation'][hawaii]), 0,
            'seed has no field activations')

    def test_kernel_trace_present(self):
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '62f666c0'").fetchone()[0]
        qv = np.frombuffer(embed_query("travel"), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        self.assertGreater(len(result['trace']), 0,
            'no kernel trace entries')
        first_step = result['trace'][0]
        for key in ('step', 'new_nodes', 'edges_considered',
                    'edges_transmitted', 'max_act'):
            self.assertIn(key, first_step,
                f'trace step missing key: {key}')


@unittest.skipUnless(HAS_R3, 'r3 preserved brain not on disk (run N=5 first)')
class TestActivationRenderingOnR3(unittest.TestCase):
    """render_surface_inject produces the 5-day/10-day composition
    content that the old pipeline missed."""

    @classmethod
    def setUpClass(cls):
        from servers import embedder
        if embedder._model is None:
            embedder.load_model()
        cls.tmp = tempfile.mkdtemp(prefix='test_render_')
        for f in os.listdir(R3_BRAIN):
            shutil.copy2(os.path.join(R3_BRAIN, f), cls.tmp)
        from servers.brain import Brain
        cls.brain = Brain(db_path=os.path.join(cls.tmp, 'brain.db'))
        cls.brain.backfill_vectors(batch_size=50)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.brain.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_nyc_content_visible_in_rendered_output(self):
        """The r3 failing case: full pipeline renders NYC's 5-day fact."""
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import (
            spread_activation, render_surface_inject)
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '62f666c0'").fetchone()[0]
        nyc = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '4e8acb7a'").fetchone()[0]

        query = ("How many days did I spend in total traveling in "
                 "Hawaii and in New York City?")
        qv = np.frombuffer(embed_query(query), dtype=np.float32)

        activation_result = spread_activation([hawaii], qv, self.brain)
        all_ids = list(activation_result['node_activation'].keys())
        rich_nodes = self.brain.get_node(all_ids)

        rendered = render_surface_inject(
            node_activation=activation_result['node_activation'],
            field_activation=activation_result['field_activation'],
            rich_nodes=rich_nodes,
            selected_mode={hawaii: 'arc'},
            query_vec=qv,
            brain=self.brain,
            total_budget=4000,
        )['text']

        self.assertGreater(len(rendered), 500,
            'rendered output too small')
        has_nyc_ref = 'NYC' in rendered or 'New York' in rendered
        self.assertTrue(has_nyc_ref, 'NYC not mentioned in rendered output')
        has_five = '5 days' in rendered or '5-day' in rendered or 'five days' in rendered
        self.assertTrue(has_five, 'NYC 5-day fact not in rendered output')
        has_ten = '10 days' in rendered or '10-day' in rendered or 'ten days' in rendered
        self.assertTrue(has_ten, 'Hawaii 10-day fact not in rendered output')

    def test_rendered_output_stays_under_budget(self):
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import (
            spread_activation, render_surface_inject)
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id = '62f666c0'").fetchone()[0]
        qv = np.frombuffer(embed_query("travel"), dtype=np.float32)

        act = spread_activation([hawaii], qv, self.brain)
        rich = self.brain.get_node(list(act['node_activation'].keys()))

        budget = 2000
        rendered = render_surface_inject(
            node_activation=act['node_activation'],
            field_activation=act['field_activation'],
            rich_nodes=rich,
            selected_mode={hawaii: 'arc'},
            query_vec=qv,
            brain=self.brain,
            total_budget=budget,
        )['text']
        # Allow overrun: MIN_NODE_BUDGET ensures each activated node gets
        # ~150 chars of content minimum; with 10+ nodes that adds up. What
        # we guard against is runaway output, not per-node slack.
        self.assertLess(len(rendered), budget * 2,
            f'rendered output {len(rendered)} ballooned beyond 2x budget {budget}')



class TestSurfacerJsonParser(unittest.TestCase):
    """Part-K hardening: Haiku's response occasionally has trailing prose
    after the JSON object. raw_decode consumes the first valid object
    and ignores the tail — no more 'Extra data' JSONDecodeError crashes
    that produce empty additionalContext."""

    def test_bare_json(self):
        from servers.scales.s1.surface import _parse_surfacer_json
        result = _parse_surfacer_json('{"selected":[{"id":"abc","why":"x"}]}')
        self.assertEqual(result, {"selected": [{"id": "abc", "why": "x"}]})

    def test_fenced_json(self):
        from servers.scales.s1.surface import _parse_surfacer_json
        result = _parse_surfacer_json('```json\n{"selected":[]}\n```')
        self.assertEqual(result, {"selected": []})

    def test_json_plus_trailing_prose(self):
        """The exact failure mode that broke fishing test — valid JSON
        followed by an explanation paragraph. Must parse cleanly."""
        from servers.scales.s1.surface import _parse_surfacer_json
        raw = ('{"selected":[{"id":"abc","why":"b"}]}\n\n'
               'Here is why I picked these memories: ...')
        result = _parse_surfacer_json(raw)
        self.assertEqual(result, {"selected": [{"id": "abc", "why": "b"}]})

    def test_fenced_plus_trailing_prose(self):
        from servers.scales.s1.surface import _parse_surfacer_json
        result = _parse_surfacer_json(
            '```\n{"selected":[]}\n```\nLet me explain...')
        self.assertEqual(result, {"selected": []})

    def test_multiline_json_plus_prose(self):
        """The actual Haiku response shape that caused the 'Extra data'
        error at char 627 — formatted multi-line JSON + explanation."""
        from servers.scales.s1.surface import _parse_surfacer_json
        raw = ('{"selected": [\n'
               '  {"id": "abc12345", "why": "reason one"},\n'
               '  {"id": "def67890", "why": "reason two"}\n'
               ']}\n\n'
               'The above shows my selections based on the query.')
        result = _parse_surfacer_json(raw)
        self.assertEqual(result, {"selected": [
            {"id": "abc12345", "why": "reason one"},
            {"id": "def67890", "why": "reason two"},
        ]})

    def test_no_json_returns_none(self):
        from servers.scales.s1.surface import _parse_surfacer_json
        self.assertIsNone(_parse_surfacer_json('I cannot find anything relevant.'))

    def test_empty_string_returns_none(self):
        from servers.scales.s1.surface import _parse_surfacer_json
        self.assertIsNone(_parse_surfacer_json(''))

    def test_non_dict_toplevel_returns_none(self):
        """If Haiku returns an array or scalar, we reject — contract
        requires a dict with 'selected' key."""
        from servers.scales.s1.surface import _parse_surfacer_json
        self.assertIsNone(_parse_surfacer_json('[1,2,3]'))



if __name__ == '__main__':
    unittest.main()
