#!/usr/bin/env python3
"""
brain — Recall Quality Tests

Tests the scoring pipeline that determines what the brain surfaces:
- Scoring weights (relevance, frequency, emotion, locked decay)
- Intent detection and type boosting
- Project filtering (dict filter on kv provenance)
- Spreading activation (multi-hop, decay, max hops)

Run: python -m unittest tests.test_recall_quality -v
"""

import sys
import os
import unittest
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


# ═══════════════════════════════════════════════════════════════
# TestScoringWeights — verify that the scoring formula respects
# the intended priority of relevance, frequency, emotion, and locked status
# ═══════════════════════════════════════════════════════════════

class TestScoringWeights(BrainTestBase):
    """Verify that scoring weights produce correct ranking behavior."""

    def test_relevance_dominates(self):
        """A node highly relevant to the query should outrank a recent but irrelevant node.

        Creates a Clerk authentication decision (matches query) and a database
        migration timeline decision (does not match). Even though both are fresh,
        the Clerk node should rank first because relevance weight (0.35) is the
        highest single factor and keyword overlap is strong.
        """
        self.brain.remember(
            type='decision',
            title='Authentication: adopt Clerk with magic links for passwordless login',
            content='We evaluated Auth0, Firebase Auth, and Clerk. Clerk won because its magic link flow '
                    'eliminates password fatigue, integrates with Next.js middleware out of the box, '
                    'and supports organization-scoped sessions for our multi-tenant architecture.',
            keywords='auth clerk login magic-link passwordless authentication nextjs middleware'
        )
        self.brain.remember(
            type='decision',
            title='Database migration timeline for Q2 schema overhaul',
            content='The PostgreSQL migration from v12 to v16 is scheduled for late April. We need to '
                    'coordinate with the DevOps team on blue-green deployment and ensure pgbouncer '
                    'connection pooling is configured before the cutover window.',
            keywords='database migration postgresql devops deployment pgbouncer schema'
        )
        self.brain.save()

        results = self.brain.recall('auth login clerk', limit=5)
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 1, 'Should find at least one result')
        self.assertIn('Clerk', result_list[0]['title'],
                      f'Clerk node should rank first, got: {result_list[0]["title"]}')

    def test_locked_never_decays(self):
        """A locked rule node should appear in recall even if its last_accessed is 60 days old.

        Locked nodes have infinite half-life and skip decay calculations entirely.
        This verifies the locked code path in the scoring pipeline.
        """
        r = self.brain.remember(
            type='rule',
            title='Never commit secrets to version control',
            content='All API keys, tokens, and credentials must live in environment variables or '
                    'a secrets manager (AWS SSM Parameter Store or 1Password CLI). Pre-commit hooks '
                    'with detect-secrets must be enabled on every repository. Violations trigger '
                    'immediate secret rotation and a post-mortem.',
            keywords='secrets security api-keys credentials pre-commit detect-secrets',
            locked=True
        )
        self.brain.save()

        # Manually backdate last_accessed to 60 days ago
        sixty_days_ago = (datetime.utcnow() - timedelta(days=60)).isoformat() + 'Z'
        self.brain.conn.execute(
            'UPDATE nodes SET last_accessed = ? WHERE id = ?',
            (sixty_days_ago, r['id'])
        )
        self.brain.conn.commit()

        results = self.brain.recall('secrets credentials api keys security', limit=10)
        result_list = results.get('results', [])
        found_ids = [n['id'] for n in result_list]
        self.assertIn(r['id'], found_ids,
                      'Locked node should still appear despite being 60 days old')


# ═══════════════════════════════════════════════════════════════
# TestProjectFilter — the dict filter hard-scopes recall by project
# ═══════════════════════════════════════════════════════════════

class TestProjectFilter(BrainTestBase):
    """Project is kv provenance; a dict filter on it scopes recall."""

    def test_project_filtering(self):
        """Project is kv provenance (2026-07-03): the dict filter hard-scopes
        by it, replacing the removed recall(project=) soft-sort. The soft
        version is the LAF proj lane (gain-dialed, session-derived —
        tests/test_recall_laf.py::TestProjLane)."""
        a = self.brain.remember(
            type='decision',
            title='Alpha: use GraphQL for the dashboard API layer',
            content='The alpha project dashboard needs flexible queries for widgets. GraphQL lets '
                    'the frontend request exactly the fields it needs without over-fetching. We use '
                    'Apollo Server with DataLoader for N+1 query prevention.',
            project='alpha'
        )
        self.brain.remember(
            type='decision',
            title='Beta: use REST for the public developer API',
            content='The beta project public API must be REST because our developer audience expects '
                    'standard HTTP methods, predictable URLs, and OpenAPI documentation. GraphQL would '
                    'add unnecessary complexity for third-party consumers.',
            project='beta'
        )
        self.brain.save()

        # project rides **extra_fields into node_metadata_kv, and get_node
        # promotes it back onto the payload (kv wins over the legacy column)
        node = self.brain.get_node(a['id'])
        self.assertEqual(node.get('project'), 'alpha')
        self.assertEqual(
            self.brain._meta_kv.get_field(a['id'], 'project'), 'alpha')

        # dict filter routes project to the kv lookup (left _NODE_COLUMNS)
        results = self.brain.recall(
            'API design decisions', limit=5,
            filter={'project': {'equals': 'alpha'}})
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 1,
                        f'Should find at least 1 result, got {len(result_list)}')
        for n in result_list:
            self.assertEqual(
                self.brain._meta_kv.get_field(n['id'], 'project'), 'alpha',
                'project filter must hard-scope to alpha nodes only')


# Coverage moved 2026-04-26 to tests/test_spread_activation.py.
# The old TestSpreadingActivation class called brain.spread_activation([seeds])
# — a method-on-Brain API that no longer exists. The kernel was rewritten as
# a query-aware module function `servers.scales.s1.surface_contract.spread_activation(
# seed_ids, query_vec, brain, prior_vecs=None)`. The new contract requires a
# query embedding (activation originates from max field-cosine with query),
# making the old "spread without query" tests not translatable. The new
# test_spread_activation.py covers the full kernel: edge enrichment text
# composition, softmax budget allocation, field activation masking, and
# end-to-end Hawaii→NYC seed propagation. The replacement tests assert
# stricter properties than the originals (edge text compositions, budget
# distributions, field masks) — not weaker.

class TestSpreadingActivationCoverageMoved(unittest.TestCase):
    """Sentinel — placeholder pointing to the new home (test_spread_activation.py).

    A plain TestCase: the import IS the assertion (it raises if the move is
    undone), and an import needs no brain — as a BrainTestBase this
    provisioned a seeded fixture per run to execute two lines that never
    touch it. Kept rather than deleted because that import is a live
    regression lock, which a 2026-06 review established when it reclassified
    this same test away from the cut list (brain 728643b1).
    """

    def test_kernel_module_importable(self):
        """The new spread_activation lives in surface_contract — verify the move landed."""
        from servers.scales.s1.surface_contract import spread_activation
        self.assertTrue(callable(spread_activation))




# ═══════════════════════════════════════════════════════════════
# TestEmbedderDownNoFallback — embedder unavailable/failed must yield EMPTY
# results + a reported condition, never a silent keyword substitute
# ═══════════════════════════════════════════════════════════════

class TestEmbedderDownNoFallback(BrainTestBase):
    """Recall's no-embedding exits: empty results, consistent shape, loud report.

    Guards the no-fallback contract — a future
    edit that re-introduces a silent keyword substitute, drops the _log_error
    report, or diverges the empty-result shape fails here."""

    def test_embedder_down_returns_empty_and_reports(self):
        from unittest.mock import patch
        from servers import embedder

        original_model = embedder._model
        original_loaded = embedder.stats['model_loaded']
        embedder._model = None
        embedder.stats['model_loaded'] = False
        try:
            with patch.object(self.brain, '_log_error') as log_spy:
                results = self.brain.recall('embedder down empty contract', limit=5)
        finally:
            embedder._model = original_model
            embedder.stats['model_loaded'] = original_loaded

        self.assertEqual(results.get('results'), [],
                         'no keyword substitute — results must be EMPTY')
        self.assertEqual(results.get('_recall_mode'), 'embedder_unavailable')
        stats = results.get('_embedding_stats')
        self.assertTrue(stats, '_embedding_stats must be present so the MCP '
                               'footer renders the failure mode inline')
        self.assertFalse(stats.get('embedder_ready'))
        sources = [c.args[0] for c in log_spy.call_args_list]
        self.assertIn('recall_embedder_unavailable', sources,
                      'the condition must be REPORTED via _log_error')

    def test_embed_failure_returns_empty_and_reports(self):
        from unittest.mock import patch
        from servers import embedder

        with patch.object(embedder, 'embed_query',
                          side_effect=RuntimeError('embed boom')), \
             patch.object(self.brain, '_log_error') as log_spy:
            results = self.brain.recall('embed failure empty contract', limit=5)

        self.assertEqual(results.get('results'), [],
                         'no keyword substitute — results must be EMPTY')
        self.assertEqual(results.get('_recall_mode'), 'embed_failed')
        self.assertTrue(results.get('_embedding_stats'),
                        'embed_failed must carry _embedding_stats too — '
                        'same shape as embedder_unavailable')
        sources = [c.args[0] for c in log_spy.call_args_list]
        self.assertIn('recall_embed_failed', sources,
                      'the condition must be REPORTED via _log_error')

    def test_laf_unavailable_returns_empty_and_reports(self):
        """The third RECALL_FAILURE_MODE: LAF is the only scorer, so a raising
        engine returns empty + a loud mode, never a degraded substitute."""
        from unittest.mock import patch
        import servers.recall_laf as recall_laf
        from servers.brain_recall import RECALL_FAILURE_MODES

        with patch.object(recall_laf, 'get_engine',
                          side_effect=RuntimeError('laf boom')), \
             patch.object(self.brain, '_log_error') as log_spy:
            results = self.brain.recall('laf unavailable empty contract', limit=5)

        self.assertEqual(results.get('results'), [],
                         'no fallback scorer — results must be EMPTY')
        self.assertEqual(results.get('_recall_mode'), 'laf_unavailable')
        self.assertIn('laf_unavailable', RECALL_FAILURE_MODES,
                      'the mode must be flagged downstream as infrastructure '
                      'failure, not as "the brain knows nothing"')
        self.assertTrue(results.get('_embedding_stats'),
                        'laf_unavailable must carry _embedding_stats too — '
                        'same shape as the embedder exits')
        sources = [c.args[0] for c in log_spy.call_args_list]
        self.assertIn('recall_laf', sources,
                      'the condition must be REPORTED via _log_error')


class TestArchivedRecallNotSupported(BrainTestBase):
    """include_archived=True is a REFUSAL, not a failure.

    Deliberately separate from TestEmbedderDownNoFallback: a healthy brain
    declining an unimplemented argument must not be booked as degraded
    infrastructure, and that distinction is what keeps eval validity from
    marking a good run suspect."""

    def test_returns_empty_with_an_explaining_note(self):
        results = self.brain.recall('archived probe', limit=10,
                                    include_archived=True)
        self.assertEqual(results.get('results'), [])
        self.assertEqual(results.get('_recall_mode'), 'archived_not_supported')
        self.assertIn('not supported', results.get('_note', ''),
                      'the caller gets a sentence, not just a mode string')

    def test_is_not_an_infrastructure_failure_mode(self):
        from servers.brain_recall import RECALL_FAILURE_MODES
        self.assertNotIn('archived_not_supported', RECALL_FAILURE_MODES)

    def test_refusal_precedes_the_embedder_gate(self):
        """The refusal must not depend on infrastructure state: with the
        embedder down, the answer is still 'not supported', not
        'embedder_unavailable' (which WOULD be marked degraded)."""
        from unittest.mock import patch
        from servers import embedder

        with patch.object(embedder, 'is_ready', return_value=False):
            results = self.brain.recall('archived probe cold embedder',
                                        limit=10, include_archived=True)
        self.assertEqual(results.get('_recall_mode'), 'archived_not_supported')


if __name__ == '__main__':
    unittest.main()
