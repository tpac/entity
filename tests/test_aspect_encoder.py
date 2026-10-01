"""Tests for the S2 Aspect Encoder's classification validation.

Focus: the noise-exclusivity boundary (2026-06-08). `noise` means "no
semantic claim"; if the encoder dual-routes a string to noise AND a real
aspect, the guard strips noise and keeps the meaning. That preserves the
invariant noise ∩ {any other aspect} = ∅, which is what lets downstream
exclusion filters trust "not in noise" == "is real knowledge".

These exercise _validate_classifications directly with a stub brain — no
LLM, no daemon. Validation derives its closed list + accepts-categories from
brain.aspects (the per-aspect `routable`/`accepts` facts, Step 4), so the
stub carries a real registry built from the repo seed.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.aspect_store import SEED_ASPECTS_JSON_PATH
from servers.aspects import AspectRegistry
from servers.scales.s2.aspect_contract import ASPECT
from servers.scales.s2.aspect_encoder import AspectEncoder

with open(SEED_ASPECTS_JSON_PATH) as _f:
    _SEED_REGISTRY = AspectRegistry.from_dict(None, json.load(_f))


class _StubBrain:
    aspects = _SEED_REGISTRY

    def __init__(self):
        self.errors = []
        self.warnings = []

    def _log_error(self, name, exc, msg):
        self.errors.append((str(exc), msg))

    def _log_warning(self, name, message, context=''):
        self.warnings.append((message, context))


def _encoder():
    # Bypass __init__: _validate_classifications only touches self.brain and
    # the class-level self.NAME — no dispatch_fn / config needed.
    enc = AspectEncoder.__new__(AspectEncoder)
    enc.brain = _StubBrain()
    enc.config = ASPECT
    return enc


def _prompt(enc, aspects, proposals, usage=None):
    """The user message run() sends, as one string (menu block + candidates
    block; run() puts the journal continuity between them)."""
    return enc._format_menu(aspects, usage) + enc._format_candidates(proposals)


def _props(*values):
    return [{'category': 'edge_relations', 'value': v, 'count': 3, 'examples': []}
            for v in values]


class TestMenuDerivation(unittest.TestCase):
    """The ASPECT MENU derives from the per-aspect facts (routable / accepts /
    prompt_visible) instead of the deleted ASPECT_ACCEPTS + `order` literals.
    Pins that the derivation reproduces the hand-curated menu this replaced —
    verified byte-identical against the pre-Step-4 prompt when it shipped."""

    def setUp(self):
        self.enc = _encoder()
        self.prompt = _prompt(
            self.enc, _SEED_REGISTRY.all(),
            [{'category': 'edge_relations', 'value': 'x', 'count': 1,
              'examples': []}])

    def test_menu_order_matches_the_replaced_literal(self):
        expected = [
            'identity_bearing', 'episodic_anchor', 'active_thread',
            'lesson_insight', 'wisdom',
            'correction_improvement',
            'extension_refinement', 'explanation_causation', 'dependency_flow',
            'contradiction_conflict', 'validation_evidence',
            'hierarchical_structure', 'temporal_sequence',
            'generic_relation', 'noise',
        ]
        shown = [line[3:-3].strip() for line in self.prompt.splitlines()
                 if line.startswith('── ') and line.endswith(' ──')
                 and not line.startswith('── #')]
        self.assertEqual(shown, expected)

    def test_menu_header_counts_routable_aspects(self):
        self.assertIn('ASPECT MENU — 15 aspects', self.prompt)

    def test_non_routable_aspect_never_offered(self):
        self.assertNotIn('survivor_lineage', self.prompt)


class TestNoiseDualRouting(unittest.TestCase):
    """noise + semantic is a LEGAL classification, kept as returned.
    This boundary used to strip noise on the reasoning that a
    semantic claim refutes it — which deleted the half that does the
    filtering, silently un-noising machinery the classifier had flagged.
    Noise vetoes at read time instead; here we only record and preserve.
    """

    def test_both_kept_when_dual_routed_with_semantic(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'foo_rel',
                'aspects': ['noise', 'correction_improvement']}]
        accepted, rejected = enc._validate_classifications(cls, _props('foo_rel'))
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['aspects'],
                         ['noise', 'correction_improvement'])

    def test_both_kept_when_noise_listed_secondary(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'baz_rel',
                'aspects': ['correction_improvement', 'noise']}]
        accepted, _ = enc._validate_classifications(cls, _props('baz_rel'))
        self.assertEqual(accepted[0]['aspects'],
                         ['correction_improvement', 'noise'])

    def test_pure_noise_preserved(self):
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'bar_rel', 'aspects': ['noise']}]
        accepted, _ = enc._validate_classifications(cls, _props('bar_rel'))
        self.assertEqual(accepted[0]['aspects'], ['noise'])

    def test_two_semantic_aspects_untouched(self):
        # The guard targets noise only — legitimate semantic multi-membership
        # (e.g. dependency_flow + temporal_sequence) must pass through intact.
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'qux_rel',
                'aspects': ['dependency_flow', 'temporal_sequence']}]
        accepted, _ = enc._validate_classifications(cls, _props('qux_rel'))
        self.assertEqual(accepted[0]['aspects'], ['dependency_flow', 'temporal_sequence'])

    def test_dual_route_recorded_as_warning_not_error(self):
        """Visibility without a false alarm: dual routing is legal now, so it
        must not land in the errors table — but it stays recorded, since it is
        the only signal telling us whether the classifier uses it at all."""
        enc = _encoder()
        cls = [{'category': 'edge_relations', 'value': 'foo_rel',
                'aspects': ['noise', 'correction_improvement']}]
        enc._validate_classifications(cls, _props('foo_rel'))
        self.assertTrue(any('noise vetoes on read' in m
                            for m, _ctx in enc.brain.warnings),
                        enc.brain.warnings)
        self.assertEqual(enc.brain.errors, [])


class TestFreeTextShapes(unittest.TestCase):
    """The reply is free JSON text, so a field can arrive as any JSON type.
    A wrong type rejects that entry; the rest of the reply still lands."""

    def _run(self, bad):
        enc = _encoder()
        good = {'category': 'edge_relations', 'value': 'ok_rel',
                'aspects': ['dependency_flow']}
        return enc._validate_classifications([bad, good],
                                             _props('ok_rel', 'bad_rel'))

    def test_non_string_value_rejects_one_entry(self):
        accepted, rejected = self._run(
            {'category': 'edge_relations', 'value': ['bad_rel'],
             'aspects': ['dependency_flow']})
        self.assertEqual([a['value'] for a in accepted], ['ok_rel'])
        self.assertIn('non-string', rejected[0]['reason'])

    def test_non_string_aspect_rejects_one_entry(self):
        accepted, rejected = self._run(
            {'category': 'edge_relations', 'value': 'bad_rel',
             'aspects': [{'name': 'dependency_flow'}]})
        self.assertEqual([a['value'] for a in accepted], ['ok_rel'])
        self.assertIn('non-string', rejected[0]['reason'])

    def test_single_aspect_key_rejects_one_entry(self):
        # The prompt teaches only `aspects: [...]`; a singular `aspect`
        # answer is counted as rejected rather than coerced.
        accepted, rejected = self._run(
            {'category': 'edge_relations', 'value': 'bad_rel',
             'aspect': 'dependency_flow'})
        self.assertEqual([a['value'] for a in accepted], ['ok_rel'])
        self.assertIn('(aspects)', rejected[0]['reason'])


class TestFillFields(unittest.TestCase):
    """Fill-fields: the FIELD DEFINITIONS block and the candidate rendering
    derive from CLASSIFIER_FIELDS (a curated field is never taught), the
    per-field validation from MEMBER_FIELDS whole, and every word on the
    candidate's page is one the encoder writes back."""

    def setUp(self):
        self.enc = _encoder()

    # ── schema ──

    # ── rendering ──

    def _prompt(self, proposals):
        return _prompt(self.enc, _SEED_REGISTRY.all(), proposals)

    def test_field_definitions_block_rendered_from_spec(self):
        from servers.aspect_store import CLASSIFIER_FIELDS
        prompt = self._prompt(_props('x'))
        self.assertIn('FIELD DEFINITIONS', prompt)
        self.assertIn('--value-->', prompt)
        for name, spec in CLASSIFIER_FIELDS.items():
            self.assertIn('\n%s (' % name, prompt)
            for val in spec.get('values') or ():
                self.assertIn('\n  %s' % val, prompt)
        # flavor is curated by hand: the classifier is never taught it
        self.assertNotIn('\nflavor (', prompt)
        self.assertLess(prompt.index('ASPECT MENU'), prompt.index('FIELD DEFINITIONS'))
        self.assertLess(prompt.index('FIELD DEFINITIONS'), prompt.index('CANDIDATES'))

    def test_new_candidate_lists_every_field_in_fill(self):
        prompt = self._prompt(_props('new_rel'))
        self.assertIn('  fill: aspects, orientation\n', prompt)
        self.assertNotIn('  aspects:', prompt)
        node = [{'category': 'node_types', 'value': 'new_type', 'count': 1, 'examples': []}]
        self.assertIn('  fill: aspects\n', self._prompt(node))

    def test_routed_candidate_shows_aspects_and_fill_only_the_gap(self):
        p = {'category': 'edge_relations', 'value': 'resolved_by', 'count': 43,
             'aspects': ['correction_improvement', 'temporal_sequence'],
             'fill': ['orientation'], 'examples': []}
        prompt = self._prompt([p])
        self.assertIn('  aspects: correction_improvement, temporal_sequence\n', prompt)
        self.assertIn('  fill: orientation\n', prompt)
        self.assertNotIn('homes', prompt)

    def test_example_lines_say_source_and_target(self):
        p = {'category': 'edge_relations', 'value': 'r', 'count': 1, 'examples': [{
            'src_id': 'aaaaaaaa', 'src_title': 'A', 'src_type': 'x', 'src_content_snippet': 'sc',
            'tgt_id': 'bbbbbbbb', 'tgt_title': 'B' * 200, 'tgt_type': 'y', 'tgt_content_snippet': 'tc',
            'description': 'd', 'weight': 1}]}
        prompt = self._prompt([p])
        self.assertIn('      source content: sc', prompt)
        self.assertIn('      target content: tc', prompt)
        self.assertNotIn('src content', prompt)
        self.assertNotIn('tgt content', prompt)
        # standard node reference on both ends, long title cut with the mark
        self.assertIn('    - [x] "A" (id:aaaaaaaa) --r--> [y] "%s…" (id:bbbbbbbb)' % ('B' * 119), prompt)
        self.assertNotIn('[strong]', prompt)
        node = [{'category': 'node_types', 'value': 't', 'count': 1, 'examples': [
            {'id': 'cccccccc', 'type': 't', 'title': 'N', 'content_snippet': 'c', 'situation': 's'}]}]
        nprompt = self._prompt(node)
        self.assertIn('    - [t] "N" (id:cccccccc)', nprompt)
        self.assertIn('      content: c', nprompt)
        self.assertIn('      situation: s', nprompt)

    def test_every_candidate_key_is_a_word_the_encoder_writes(self):
        """The vocabulary contract: a `key:` printed on a candidate block is a
        key of the returned entry, or one of the three context words the
        return rules name (count, fill, examples)."""
        import re
        from servers.aspect_store import MEMBER_FIELDS
        props = {'category', 'value', 'rationale', *MEMBER_FIELDS}
        p = {'category': 'edge_relations', 'value': 'resolved_by', 'count': 43,
             'aspects': ['correction_improvement'], 'fill': ['orientation'],
             'examples': []}
        prompt = self._prompt([p, _props('new_rel')[0]])
        block = prompt[prompt.index('── #1 ──'):]
        keys = set(re.findall(r'^  (\w+):', block, re.M))
        self.assertTrue(keys, block[:200])
        self.assertEqual(keys - props, {'count', 'fill', 'examples'}, keys)

    # ── validation ──

    def test_full_entry_keeps_aspects_and_orientation(self):
        cls = [{'category': 'edge_relations', 'value': 'new_rel',
                'aspects': ['correction_improvement'], 'orientation': 'target_actor',
                'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, _props('new_rel'))
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['orientation'], 'target_actor')
        self.assertEqual(accepted[0]['aspects'], ['correction_improvement'])

    def test_fill_only_entry_keeps_candidate_homes(self):
        p = {'category': 'edge_relations', 'value': 'resolved_by', 'count': 43,
             'aspects': ['correction_improvement', 'settlement'],
             'fill': ['orientation'], 'examples': []}
        cls = [{'category': 'edge_relations', 'value': 'resolved_by',
                'orientation': 'target_actor', 'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, [p])
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['aspects'], ['correction_improvement', 'settlement'])
        self.assertEqual(accepted[0]['orientation'], 'target_actor')

    def test_fill_only_entry_cannot_reroute(self):
        p = {'category': 'edge_relations', 'value': 'resolved_by', 'count': 43,
             'aspects': ['correction_improvement'], 'fill': ['orientation'], 'examples': []}
        cls = [{'category': 'edge_relations', 'value': 'resolved_by',
                'aspects': ['noise'], 'orientation': 'target_actor', 'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, [p])
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['aspects'], ['correction_improvement'])
        self.assertTrue(any('not in fill' in w[0] for w in self.enc.brain.warnings),
                        self.enc.brain.warnings)

    def test_missing_field_is_a_warning_and_stays_open(self):
        cls = [{'category': 'edge_relations', 'value': 'new_rel',
                'aspects': ['correction_improvement'], 'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, _props('new_rel'))
        self.assertEqual(rejected, [])
        self.assertNotIn('orientation', accepted[0])
        self.assertTrue(any('left open' in w[0] for w in self.enc.brain.warnings),
                        self.enc.brain.warnings)
        self.assertEqual(self.enc.brain.errors, [])

    def test_unknown_value_rejects_the_entry(self):
        cls = [{'category': 'edge_relations', 'value': 'new_rel',
                'aspects': ['correction_improvement'], 'orientation': 'forwards',
                'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, _props('new_rel'))
        self.assertEqual(accepted, [])
        self.assertIn('orientation not in closed list', rejected[0]['reason'])

    def test_missing_aspects_on_a_new_string_still_rejects(self):
        cls = [{'category': 'edge_relations', 'value': 'new_rel',
                'orientation': 'symmetric', 'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, _props('new_rel'))
        self.assertEqual(accepted, [])
        self.assertIn('aspects', rejected[0]['reason'])

    def test_curated_field_from_the_model_is_dropped(self):
        # No fill line ever names flavor; a value the model writes anyway is
        # dropped with a warning, never stored.
        cls = [{'category': 'edge_relations', 'value': 'new_rel',
                'aspects': ['correction_improvement'], 'orientation': 'source_actor',
                'flavor': 'replaced', 'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, _props('new_rel'))
        self.assertEqual(rejected, [])
        self.assertNotIn('flavor', accepted[0])
        self.assertTrue(any('not in fill' in w[0] for w in self.enc.brain.warnings),
                        self.enc.brain.warnings)

    def test_node_type_never_carries_an_edge_field(self):
        node = [{'category': 'node_types', 'value': 'new_type', 'count': 1, 'examples': []}]
        cls = [{'category': 'node_types', 'value': 'new_type',
                'aspects': ['lesson_insight'], 'orientation': 'symmetric', 'rationale': 'r'}]
        accepted, rejected = self.enc._validate_classifications(cls, node)
        self.assertEqual(rejected, [])
        self.assertNotIn('orientation', accepted[0])
        self.assertTrue(any('not in fill' in w[0] for w in self.enc.brain.warnings))

    def test_menu_shows_most_used_members_with_a_labelled_sample(self):
        from servers.aspects import rank_members as _menu_members
        cap = ASPECT['menu_members_per_aspect']
        members = ['v%02d' % i for i in range(cap + 5)]
        counts = {m: i for i, m in enumerate(members)}      # v24 most used
        label, shown = _menu_members(members, counts, cap)
        self.assertEqual(label, 'top %d of %d by use' % (cap, cap + 5))
        self.assertEqual(shown.split(', ')[0], 'v%02d' % (cap + 4))
        self.assertEqual(len(shown.split(', ')), cap)
        self.assertEqual(_menu_members(members, {}, cap)[0], 'first %d of %d' % (cap, cap + 5))
        self.assertEqual(_menu_members(['a', 'b'], {}, cap), ('all 2', 'a, b'))

    def test_menu_ranks_by_usage_and_says_so(self):
        usage = {'edge_relations': {'supersedes': 500}, 'node_types': {}}
        prompt = _prompt(self.enc, _SEED_REGISTRY.all(), _props('x'), usage)
        line = [l for l in prompt.splitlines()
                if l.startswith('  current edge_relations') and 'supersedes' in l
                and 'top ' in l][0]
        self.assertTrue(line.split(': ', 1)[1].startswith('supersedes,'), line)
        self.assertIn('most-used members, not every member', prompt)

    def test_menu_block_and_candidates_block_are_separable(self):
        # run() sends them as two blocks.
        menu = self.enc._format_menu(_SEED_REGISTRY.all())
        body = self.enc._format_candidates(_props('x'))
        self.assertTrue(menu.endswith('\n'))
        self.assertIn('FIELD DEFINITIONS', menu)
        self.assertNotIn('CANDIDATES', menu)
        self.assertTrue(body.startswith('═'))

    def test_system_prompt_names_the_fill_vocabulary(self):
        from servers.scales.s2.aspect_prompt import SYSTEM_PROMPT
        for word in ('`fill`', 'FIELD DEFINITIONS', '"orientation"', 'Do not re-route'):
            self.assertIn(word, SYSTEM_PROMPT, word)
        self.assertNotIn('homes', SYSTEM_PROMPT)


if __name__ == '__main__':
    unittest.main()
