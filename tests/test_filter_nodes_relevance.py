"""filter_nodes relevance path — rank-then-enrich decoupling.

The relevance branch ranks the skinny candidate pool by embedding/structural
order FIRST, then correction-enriches only the <=limit winners — not the whole
(limit * relevance_pool_multiplier) pool it discards most of. This pins that
contract.

needs_embedder=False on purpose: without an embedder, _rerank_by_relevance
falls back to structural [:limit] — but the trim still happens BEFORE
enrichment, so the winners-only-enrich property holds regardless.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestFilterNodesRelevanceEnrichesOnlyWinners(BrainTestBase):
    needs_embedder = False

    def test_relevance_rich_enriches_only_winners(self):
        # Seed more candidates than the limit (pool would be limit*mult = 6).
        for i in range(8):
            self.brain.remember(type='insight', title='insight %d' % i,
                                content='body %d' % i)

        # Spy on get_node — the enrichment call — to count enriched ids.
        enriched_ids = []
        orig = self.brain.get_node

        def spy(ids_or_id, *a, **k):
            ids = ids_or_id if isinstance(ids_or_id, list) else [ids_or_id]
            enriched_ids.extend(ids)
            return orig(ids_or_id, *a, **k)

        self.brain.get_node = spy
        try:
            res = self.brain.filter_nodes(
                field='type', include=['insight'],
                rich=True, relevance_query='insight about something',
                limit=2, relevance_pool_multiplier=3)
        finally:
            self.brain.get_node = orig

        nodes = res.get('nodes', [])
        self.assertLessEqual(len(nodes), 2, 'must return <= limit')
        # The decoupling: enrichment touched at most `limit` ids, NOT the
        # full pool (limit * multiplier = 6).
        self.assertLessEqual(
            len(enriched_ids), 2,
            'relevance+rich must enrich only the <=limit winners, not the pool')

    def test_non_relevance_rich_unchanged(self):
        for i in range(5):
            self.brain.remember(type='insight', title='n %d' % i,
                                content='c %d' % i)
        res = self.brain.filter_nodes(field='type', include=['insight'],
                                      rich=True, limit=3)
        nodes = res.get('nodes', [])
        self.assertLessEqual(len(nodes), 3)
        # rich shape — enrichment ran (content present)
        if nodes:
            self.assertIn('content', nodes[0])


if __name__ == '__main__':
    unittest.main()
