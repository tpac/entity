"""Tests for the Frame Constructor — Anchor's structured awareness object.

Covers (post 2026-08-05 lean Frame: deterministic session-state only):
- build_frame returns markdown with Session / Current focus
- removed queried-node sections (wisdom / Operator / Partnership) never appear
- the Session header carries project / clock / worktree from session env
- ctx.get_frame(brain) is the session-scoped entry point
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.scales.s1.frame import build_frame


class TestFrameConstruction(BrainTestBase):
    """Frame builds the expected three sections; removed sections are gone."""
    needs_embedder = False

    def test_build_frame_returns_markdown_with_all_sections(self):
        # A wisdom-type node exists, but the wisdom section is disabled
        # (2026-07-16 ruling: no queried nodes in the Frame — deterministic
        # session state only, pending the identity-prior redesign)
        self.brain.remember(
            type='insight', title='Test insight for wisdom',
            content='Some insight content')

        frame = build_frame(self.brain, 'test_session')

        self.assertIsInstance(frame, str)
        self.assertIn('## Session', frame)
        self.assertIn('## Current focus', frame)
        # Removed / disabled sections must NOT appear
        self.assertNotIn('## Recent moves', frame)
        self.assertNotIn("## What I've learned", frame)
        self.assertNotIn('## Operator', frame)
        self.assertNotIn('## Partnership', frame)
        self.assertNotIn('## Active threads', frame)

    def test_build_frame_handles_empty_brain(self):
        # No nodes — should still render the kept sections without crashing
        frame = build_frame(self.brain, 'test_session')

        self.assertIn('## Current focus', frame)
        # Empty sections render gracefully, not as errors
        self.assertNotIn('Traceback', frame)
        self.assertNotIn('Error', frame)


class TestSessionHeader(BrainTestBase):
    """The Session header carries the deterministic situational anchor."""
    needs_embedder = False

    def test_header_shows_project_and_worktree_from_session_env(self):
        ctx = self.brain.get_or_create_session('hdr-sess')
        ctx.set_env(cwd='/x/y', branch='main', worktree='wt-1', project='brain')
        ctx.save(self.brain._session_state)

        frame = build_frame(self.brain, 'hdr-sess')
        self.assertIn('- Project: brain', frame)
        self.assertIn('- Worktree: wt-1', frame)

    def test_header_unscoped_project_is_explicit(self):
        # '' project renders as (unscoped) — itself a signal (no project
        # pressure applies), never a silently missing line.
        frame = build_frame(self.brain, 'hdr-fresh')
        self.assertIn('- Project: (unscoped)', frame)
        self.assertNotIn('- Worktree:', frame)   # no worktree → no line

    def test_header_clock_uses_passed_at(self):
        # Replays pass their injected conversation-time; the header must
        # honor it (bare wall-clock would corrupt time-anchored replays).
        import datetime as dt
        at = dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.timezone.utc)
        frame = build_frame(self.brain, 'hdr-at', at=at)
        self.assertIn('- Now: 2026-01-02 03:04 UTC (Friday)', frame)


class TestSessionContextPerSession(BrainTestBase):
    """session_context must be per-session — no parallel-session leak.
    (Restored 2026-08-05: collaterally dropped in the lean-Frame rewrite —
    these guard the historical session-A-arc-bleeds-into-session-B bug and
    were never about the wisdom section.)"""
    needs_embedder = False

    def test_session_context_for_returns_per_session(self):
        # Two distinct sessions write different contexts via direct config
        self.brain.set_config('session_context_session_a', 'arc for session A')
        self.brain.set_config('session_context_session_b', 'arc for session B')

        self.assertEqual(
            self.brain.session_context_for('session_a'),
            'arc for session A')
        self.assertEqual(
            self.brain.session_context_for('session_b'),
            'arc for session B')

    def test_session_context_for_unknown_returns_empty(self):
        self.assertEqual(
            self.brain.session_context_for('never_written'), '')

    def test_session_context_for_no_session_id_returns_empty(self):
        # Defensive: don't read a global key when session_id is missing
        self.assertEqual(self.brain.session_context_for(''), '')

    def test_global_session_context_property_removed(self):
        # The leaky global property must not exist — replaced by session_context_for
        self.assertFalse(hasattr(self.brain, 'session_context'),
                        "brain.session_context property should be removed (leaked across parallel sessions)")


class TestSurfacePromptAcceptsFrame(BrainTestBase):
    """build_surface_prompt accepts frame and renders it as 'Partnership
    context:'. (Restored 2026-08-05 — test_pipeline_contract.py:236 deleted
    its own copy as redundant, pointing here by name; dropping frame= from
    the prompt builder must not pass a green suite.)"""
    needs_embedder = False

    def test_frame_renders_as_partnership_context(self):
        from servers.scales.s1.surface_contract import build_surface_prompt

        candidates = [{'id': 'abcd1234', 'title': 'Test',
                       'type': 'fact', 'content': 'x',
                       'confidence': 0.5, 'score': 0.8}]
        prompt, _ = build_surface_prompt(
            candidates, "test query",
            frame="THE FRAME PRIOR CONTENT")
        self.assertIn('THE FRAME PRIOR CONTENT', prompt)
        self.assertIn('<partnership_context>', prompt)

    def test_empty_frame_renders_explicit_degraded_marker(self):
        from servers.scales.s1.surface_contract import build_surface_prompt

        candidates = [{'id': 'abcd1234', 'title': 'Test',
                       'type': 'fact', 'content': 'x',
                       'confidence': 0.5, 'score': 0.8}]
        prompt, _ = build_surface_prompt(
            candidates, "test query",
            frame="")
        # Explicit degraded marker — no silent fallback to old layout
        # The live marker's own wording — legacy said "no partnership
        # context", the xml render says this. Same contract: the absence is
        # stated, never a silent fallback.
        self.assertIn('<partnership_context>', prompt)
        self.assertIn('fresh session or frame unavailable', prompt.lower())
        # Make sure no Phase 1 layout artifacts appear
        self.assertNotIn('Session arc', prompt)
        self.assertNotIn("Encoder's recent journal", prompt)


class TestSessionContextGetFrame(BrainTestBase):
    """ctx.get_frame(brain) is the public session-scoped entry point."""
    needs_embedder = False

    def test_ctx_get_frame_returns_same_as_build_frame(self):
        # Deterministic sections match; the clock line can tick between the
        # two calls, so compare with the Now line stripped.
        ctx = self.brain.get_or_create_session('test_session_ctx')
        strip = lambda f: '\n'.join(
            l for l in f.splitlines() if not l.startswith('- Now:'))
        from_ctx = strip(ctx.get_frame(self.brain))
        from_direct = strip(build_frame(self.brain, 'test_session_ctx'))
        self.assertEqual(from_ctx, from_direct)
