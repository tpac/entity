"""Tests for pipeline_contract.py — surface prompt, output formatting, embedding groups.

Covers:
- EMBEDDING_GROUPS contract integrity
- format_candidate_for_surface output structure
- build_surface_prompt assembly
"""

import unittest
import sys

sys.path.insert(0, '.')

from servers.pipeline_contract import (
    EDGE_DESCRIPTIONS_FIELD,
    EMBEDDING_GROUPS,
    EMBEDDING_SKIP_FIELDS,
    INVALIDATED_BY_EDGE_WRITE,
    INVALIDATED_BY_REVISE,
    vectors_affected_by,
)
from servers.scales.s1.surface_contract import (
    format_candidate_for_surface,
    build_surface_prompt,
    SURFACE,
)


class TestTruncationInvariants(unittest.TestCase):
    """Cross-contract truncation invariants — loud instead of comment-enforced."""

    def test_upstream_cap_covers_display_limits(self):
        """PIPELINE['recent_message_content'] must not clip below the per-role
        display truncation in build_surface_prompt. The upstream trace pull in
        daemon_hooks caps every message at recent_message_content chars BEFORE
        the per-role limits apply — a smaller upstream value silently starves
        the role limits (anchor turns clipped at the cap, the configured limit
        never reached).
        """
        from servers.pipeline_contract import PIPELINE
        self.assertGreaterEqual(
            PIPELINE['recent_message_content'],
            max(SURFACE['user_message_limit'], SURFACE['anchor_message_limit'],
                SURFACE['last_anchor_message_limit'],
                SURFACE['current_message_limit']),
            "recent_message_content (upstream cap) clips messages below the "
            "per-role display limits in build_surface_prompt")


class TestEmbeddingGroups(unittest.TestCase):
    """Verify embedding group contract integrity."""

    def test_all_groups_present(self):
        self.assertEqual(set(EMBEDDING_GROUPS.keys()), {
            'title', 'blend', 'high_meta', 'other_meta', 'edge_context', 'question',
            'field_content', 'field_reasoning',
            'field_their_raw_quote', 'field_my_raw_quote',
        })

    def test_cohort_assignment(self):
        legacy = {k for k, v in EMBEDDING_GROUPS.items() if v['cohort'] == 'legacy'}
        field = {k for k, v in EMBEDDING_GROUPS.items() if v['cohort'] == 'field'}
        self.assertEqual(legacy, {
            'title', 'blend', 'high_meta', 'other_meta', 'edge_context', 'question'
        })
        self.assertEqual(field, {
            'field_content', 'field_reasoning',
            'field_their_raw_quote', 'field_my_raw_quote'
        })

    def test_cohort_weight_invariants(self):
        """Legacy weights > 0 (participate in recall top2_avg).
        Field weights = 0 (kernel reads directly, must not ripple into scoring)."""
        for name, group in EMBEDDING_GROUPS.items():
            cohort = group.get('cohort')
            if cohort == 'legacy':
                self.assertGreater(group['weight'], 0, f"{name}: legacy must have weight>0")
            elif cohort == 'field':
                self.assertEqual(group['weight'], 0, f"{name}: field must have weight=0")
            else:
                self.fail(f"{name}: unknown cohort {cohort!r}")

    def test_weights_ordered(self):
        """Title > blend > high_meta > other_meta."""
        self.assertGreater(EMBEDDING_GROUPS['title']['weight'],
                          EMBEDDING_GROUPS['blend']['weight'])
        self.assertGreater(EMBEDDING_GROUPS['blend']['weight'],
                          EMBEDDING_GROUPS['high_meta']['weight'])
        self.assertGreater(EMBEDDING_GROUPS['high_meta']['weight'],
                          EMBEDDING_GROUPS['other_meta']['weight'])

    def test_every_group_declares_who_keeps_it_fresh(self):
        # `fields` says what a vector is MADE OF; `invalidated_by` says which
        # write path deletes it when a source changes. edge_context carried a
        # correct `fields` entry with no consumer for months — this is the
        # assertion that would have failed the day it was added.
        for name, group in EMBEDDING_GROUPS.items():
            self.assertIn(group.get('invalidated_by'),
                          {INVALIDATED_BY_REVISE, INVALIDATED_BY_EDGE_WRITE},
                          '%s must declare an invalidation owner' % name)

    def test_edge_write_owner_matches_the_edge_field_dependency(self):
        # The declaration and the dependency map must agree: exactly the
        # groups built from edge descriptions are invalidated by edge writes,
        # and no revise-owned group depends on the edge field.
        edge_owned = {g['vector_type'] for g in EMBEDDING_GROUPS.values()
                      if g['invalidated_by'] == INVALIDATED_BY_EDGE_WRITE}
        self.assertEqual(edge_owned, vectors_affected_by(EDGE_DESCRIPTIONS_FIELD))
        self.assertEqual(edge_owned, {'edge_context'})
        for name, g in EMBEDDING_GROUPS.items():
            if g['invalidated_by'] == INVALIDATED_BY_REVISE:
                self.assertNotIn(EDGE_DESCRIPTIONS_FIELD, g['fields'], name)

    def test_title_always_computed(self):
        self.assertTrue(EMBEDDING_GROUPS['title']['always_compute'])

    def test_blend_is_primary(self):
        self.assertEqual(EMBEDDING_GROUPS['blend']['vector_type'], '_primary')

    def test_skip_fields(self):
        self.assertIn('metadata_created_at', EMBEDDING_SKIP_FIELDS)
        self.assertIn('validation_count', EMBEDDING_SKIP_FIELDS)


class TestFormatCandidateForSurface(unittest.TestCase):
    """Verify candidate formatting for the surface prompt."""

    def test_basic_candidate(self):
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test rule',
             'content': 'Some content', 'confidence': 0.9, 'score': 0.75}
        result = format_candidate_for_surface(c)
        # The element is the pick key's home; there is no list-position
        # header any more (the score/#N header went with the pre-v13 arm).
        self.assertIn('<candidate id="abc12345">', result)
        self.assertTrue(result.endswith('</candidate>'))
        self.assertNotIn('match:', result)
        self.assertIn('[rule]', result)
        self.assertIn('Test rule', result)
        self.assertIn('abc12345', result)

    def test_metadata_lean_default_vs_full(self):
        """Selection render contract (2026-06-12, Area 2 lean default):
        the lean render keeps situation (the selection signal) and
        deliberately SKIPS encoder/recall scaffolding — reasoning,
        question, voice quotes. BRAIN_HAIKU_RENDER=full restores the
        heavy render with all metadata. Ablation: ab_render_ablation.py
        (gold-neutral at −41% tokens; divergence at the same-prompt
        noise floor)."""
        import os
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content',
             'situation': 'When debugging',  # situation is top-level for ergonomics
             '_metadata': {'reasoning': 'Important because...',
                           'question': 'Why does this matter?',
                           'their_raw_quote': 'verbatim operator words'}}
        # Lean (default): situation in, scaffolding out
        result = format_candidate_for_surface(c)
        self.assertIn('Situation:', result)
        self.assertNotIn('reasoning:', result.lower())
        self.assertNotIn('question:', result.lower())
        self.assertNotIn('verbatim operator words', result)
        # Full (explicit env): all metadata restored
        os.environ['BRAIN_HAIKU_RENDER'] = 'full'
        try:
            result_full = format_candidate_for_surface(c)
            self.assertIn('Situation:', result_full)
            self.assertIn('reasoning:', result_full.lower())
            self.assertIn('verbatim operator words', result_full)
        finally:
            os.environ.pop('BRAIN_HAIKU_RENDER', None)

    def test_metadata_omitted_when_empty(self):
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content'}
        result = format_candidate_for_surface(c)
        self.assertNotIn('Situation:', result)
        self.assertNotIn('Reasoning:', result)

    def test_locked_flag(self):
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content', 'locked': True}
        result = format_candidate_for_surface(c)
        self.assertIn('locked', result)

    def test_edges_included(self):
        """Edges live in `connections` (get_rich_node shape) and render as
        natural-language relations like `this depends_on "Related node"`.

        Renderer contract (servers/contract.py:render_rich_node):
        - When connection has multiple relations → use `relations` array.
        - When single relation → use top-level `relation` + `description`
          on the connection (not nested in a `relations` list of one).

        Was: pinned on the literal substring `Related node` from a top-level
        `top_edges` array, which the formatter no longer reads.
        """
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content',
             'connections': [{
                 'id': 'def67890',
                 'title': 'Related node',
                 'type': 'concept',
                 'direction': 'outgoing',
                 # Single-relation form: top-level relation + description.
                 'relation': 'depends_on',
                 'description': 'because',
             }]}
        result = format_candidate_for_surface(c)
        self.assertIn('Related node', result)
        self.assertIn('depends_on', result)


    def test_render_depth_moves_the_body_not_the_element(self):
        """BRAIN_HAIKU_RENDER and the <candidate> element are two different
        axes, and only one of them may move. The env var picks the BODY
        (HAIKU_FORMAT vs HAIKU_FORMAT_LEAN); the element line carries the
        pick key. If a depth change ever reaches the attributes, Haiku's
        id source changes with an ablation flag — a third behavior neither
        format intends."""
        import os
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content', 'locked': True,
             'discovery': 'recall_topical',
             '_metadata': {'reasoning': 'heavy scaffolding'}}
        lean = format_candidate_for_surface(c)
        os.environ['BRAIN_HAIKU_RENDER'] = 'full'
        try:
            full = format_candidate_for_surface(c)
        finally:
            os.environ.pop('BRAIN_HAIKU_RENDER', None)
        self.assertEqual(lean.split('\n')[0], full.split('\n')[0],
                         'render depth moved the candidate element')
        self.assertTrue(lean.endswith('</candidate>'))
        self.assertTrue(full.endswith('</candidate>'))
        # ... while the body it wraps is exactly what the depth decides.
        self.assertNotIn('heavy scaffolding', lean)
        self.assertIn('heavy scaffolding', full)

    def test_scope_does_not_leak_into_the_shared_format(self):
        """`scope` is per-session; the format constants are module-global.
        The formatter must copy before adding it — a mutation would mark
        every later candidate in the process against one session's scope,
        including candidates rendered for a different session."""
        from servers.scales.s1.surface_contract import (
            HAIKU_FORMAT, HAIKU_FORMAT_LEAN)
        c = {'id': 'abc12345', 'type': 'rule', 'title': 'Test',
             'content': 'Content', 'project': 'other-project'}
        marked = format_candidate_for_surface(c, scope={'project': 'brain'})
        self.assertIn('From another project', marked,
                      'differential exposure did not mark a foreign node')
        self.assertNotIn('scope', HAIKU_FORMAT_LEAN)
        self.assertNotIn('scope', HAIKU_FORMAT)
        self.assertNotIn('From another project',
                         format_candidate_for_surface(c),
                         'a scoped render left its scope on the constant')


class TestBuildSurfacePrompt(unittest.TestCase):
    """Verify surface prompt USER-content assembly.

    2026-05-03 (Frame Phase 2.5 / surface prompt v2): instructions moved
    to the cached system block (registered `surface` interaction template),
    assembled by `_call_surface`. `build_surface_prompt` now builds ONLY
    the per-turn user content. Instruction-content tests live in
    test_frame.py's TestSurfacePromptAcceptsFrame and the registered
    template itself; they're not this function's contract.
    """

    def test_prompt_includes_user_message(self):
        prompt, _ = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'how does the daemon work?')
        self.assertIn('how does the daemon work?', prompt)

    def test_prompt_uses_generic_operator_label(self):
        # 2026-05-03: prompts must not hardcode operator names — different
        # Anchor will have a different operator. Conversation should use
        # "Operator:" generically.
        prompt, _ = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'how does the daemon work?')
        # The live render labels the speaker structurally (<user>), so the
        # rule is the same and the assertion follows the one layout that
        # ships — this test asserted the deleted legacy label until 2026-09-21.
        self.assertIn('<user>', prompt)
        self.assertNotIn('Tom:', prompt)

    # test_prompt_includes_frame_when_provided — REMOVED (redundant). Asserted the
    # identical build_surface_prompt(frame=...) behavior as
    # test_frame.py::TestSurfacePromptAcceptsFrame::test_frame_renders_as_partnership_context.
    # This class's own docstring already defers frame/instruction-content tests to
    # test_frame.py; coverage lives there (and test_frame adds the empty-frame
    # degraded-marker case this never had).

    def test_max_tokens_from_config(self):
        _, max_tokens = build_surface_prompt(
            [{'id': 'a', 'type': 'rule', 'title': 'T', 'content': 'C'}],
            'test')
        self.assertEqual(max_tokens, SURFACE['max_tokens'])


class TestFormatSurfaceOutput(unittest.TestCase):
    """Verify structured output formatting."""
