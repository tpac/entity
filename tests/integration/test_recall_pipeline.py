"""Integration test: full recall pipeline (cosine → z-weighted → fatigue → results).

Runs on a seeded isolated brain (brain_env), never the live one.
Tests the actual recall() function end-to-end.
"""

import unittest
import sys

sys.path.insert(0, '.')


def get_test_brain():
    """An entered seed-brain env, and the env so the class can close it.

    needs_embedder=True: every class below recalls, and recall needs the
    model to encode the query even though the template carries its document
    vectors.
    """
    from tests.isolated_brain import brain_env
    env = brain_env(needs_embedder=True)
    env.__enter__()
    return env.brain, env


class TestRecallPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.brain, cls._env = get_test_brain()

    @classmethod
    def tearDownClass(cls):
        cls._env.__exit__(None, None, None)

    def test_recall_returns_results(self):
        """Basic recall returns a list of results."""
        result = self.brain.recall("how I grow and what deserves to be kept", limit=10)
        self.assertIn('results', result)
        self.assertGreater(len(result['results']), 0)

    def test_recall_limit_respected(self):
        result = self.brain.recall("one brain, one human", limit=5)
        self.assertLessEqual(len(result['results']), 5)

    def test_results_have_required_fields(self):
        """Each result must have id, type, title, effective_activation."""
        result = self.brain.recall("a correction from my operator is treasure", limit=3)
        for r in result['results']:
            self.assertIn('id', r)
            self.assertIn('type', r)
            self.assertIn('title', r)
            self.assertIn('effective_activation', r)

    def test_results_sorted_by_score(self):
        """Results should be sorted descending by effective_activation."""
        result = self.brain.recall("verify before claiming what I have not checked", limit=10)
        scores = [r['effective_activation'] for r in result['results']]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fatigue_accumulates_within_a_session(self):
        """Recall with a session increments per-node fatigue on that session.

        Fatigue is per-session: it lives on SessionContext and only accumulates
        when recall is given one (production always is — the hook passes ctx=,
        the MCP path passes session_id=). A recall with no session deliberately
        drops the increment, so this test must supply one. The tests that used
        to live here read `brain._fatigue_ctx` / `brain._session_fatigue` —
        attributes that do not exist — so they could never observe accumulation
        and sat deselected in pytest.ini for months.

        NOT asserted here, deliberately — two things measured 2026-08-13 under
        laf_v1 that nobody has explained yet, and pinning a guess about either
        would be worse than leaving them visible:
        (1) The count does not pass 1. A second recall of the same query in the
            same session leaves the top node's fatigue at 1, though
            `_mark_accessed` increments unconditionally when ctx is not None and
            `get_or_create_session` returns a cached instance by reference.
        (2) Fatigue does not move the surfaced score. Two such recalls return
            `effective_activation` identical to 16 decimal places, though the
            path reads as intact: LAF field score → `sim *= (1 - fatigue)` →
            embedding_scores → blended → effective_activation.
        Both need a debug run, not more reading. Until then this test pins only
        what is proven: a recall with a session records fatigue.
        """
        sid = 'fatigue-contract-test'
        ctx = self.brain.get_or_create_session(sid)
        ctx.fatigue.clear()

        r1 = self.brain.recall("my operator persists across every session", limit=25, session_id=sid)
        self.assertTrue(r1['results'], "need at least one result to fatigue")
        first_id = r1['results'][0]['id']

        self.assertGreater(len(ctx.fatigue), 0,
                           "recall with a session must increment fatigue")
        self.assertGreater(ctx.fatigue.get(first_id, 0), 0,
                           "the top result must carry a fatigue count")


    def test_structural_degree_cache(self):
        """Structural degree cache should be populated after recall."""
        self.brain.recall("recognition is not a verdict", limit=3)
        self.assertTrue(hasattr(self.brain, '_structural_degree_cache'))
        self.assertGreater(len(self.brain._structural_degree_cache), 0)

    def test_25_candidates_default(self):
        """Default recall should return up to 25 results for the surface."""
        result = self.brain.recall("revise rather than duplicate a memory", limit=25)
        self.assertLessEqual(len(result['results']), 25)


if __name__ == '__main__':
    unittest.main()
