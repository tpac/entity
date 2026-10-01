"""query_traces saturated-limit flag — the loud-by-default truncation guard.

A bounded trace query that hits its limit covers only the most recent slice
of the requested window while looking complete. The 2026-08-06 cost tally
aggregated 'a week' of deltas that were actually 2 days (limit=5000 saturated
by S0 tool_result volume) — plausible numbers, wrong window. query_traces now
fetches limit+1 and attaches a 'truncated' payload when the extra row proves
the window holds more than the result carries.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


def _events(n, session_id='trunctest-session'):
    return [{
        'chain_id': f's0-trunc-{i}',
        'scale': 's0',
        'event_type': 'delta',
        'ref_type': 'tool_result',
        'summary': f'event {i}',
        'metadata': {'tool': 'probe', 'i': i},   # tool_result rows carry `tool` (trace_contract shape)
        'session_id': session_id,
    } for i in range(n)]


class TestQueryTracesTruncation(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.brain._trace_dal.append_batch(_events(10))

    def test_saturated_limit_carries_truncated_payload(self):
        res = self.brain.query_traces(event_type='delta', hours=24, limit=4)
        self.assertEqual(len(res['events']), 4)
        self.assertIn('truncated', res)
        t = res['truncated']
        self.assertEqual(t['limit'], 4)
        # coverage_start is the oldest row actually returned
        self.assertEqual(t['coverage_start'],
                         min(e['created_at'] for e in res['events']))
        self.assertIn('limit=4', t['note'])

    def test_unsaturated_limit_has_no_truncated_key(self):
        res = self.brain.query_traces(event_type='delta', hours=24, limit=50)
        self.assertEqual(len(res['events']), 10)
        self.assertNotIn('truncated', res)

    def test_exact_fit_is_not_truncated(self):
        """limit == row count: the +1 probe proves exhaustion, no false alarm."""
        res = self.brain.query_traces(event_type='delta', hours=24, limit=10)
        self.assertEqual(len(res['events']), 10)
        self.assertNotIn('truncated', res)

    def test_ref_type_branch_flags_too(self):
        res = self.brain.query_traces(ref_type='tool_result', hours=24, limit=3)
        self.assertEqual(len(res['events']), 3)
        self.assertIn('truncated', res)


class TestRecallEpisodesHonestLimit(BrainTestBase):
    """recall_episodes honest limit: None → all events in the window (no silent
    500 cap); a number is an honest page that flags loudly when the window
    holds more. The agent-facing default+cap live at the dispatch door.

    Scoped by session_id (bypasses the 7-day default window) and ref_type
    (the tool_result events are excluded from the s0-conversational default).
    """
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.brain._trace_dal.append_batch(_events(15))

    def _episodes(self, **kw):
        return self.brain.recall_episodes(
            session_id='trunctest-session', ref_type='tool_result', **kw)

    def test_limit_none_is_unbounded_past_the_old_500_cap(self):
        # The removed cap was EPISODE_MAX_LIMIT=500 — limit=None must return
        # every matching event PAST that boundary, not the old silent 500.
        # (15 < 500 would prove None!=default but never exercise the cap.)
        from servers.brain_constants import EPISODE_MAX_LIMIT
        n = EPISODE_MAX_LIMIT + 50
        self.brain._trace_dal.append_batch(
            _events(n, session_id='bigcap-session'))
        res = self.brain.recall_episodes(
            session_id='bigcap-session', ref_type='tool_result', limit=None)
        self.assertEqual(len(res['episodes']), n)    # all 550, no 500 clamp
        self.assertFalse(res['truncated'])           # asked for all, got all
        self.assertEqual(res['ranked_by'], 'time')

    def test_numeric_limit_is_honest_page_and_flags(self):
        res = self._episodes(limit=5)
        self.assertEqual(len(res['episodes']), 5)
        self.assertTrue(res['truncated'])            # 15 > 5 → loud payload
        self.assertEqual(res['truncated']['limit'], 5)

    def test_unsaturated_limit_no_flag(self):
        res = self._episodes(limit=50)
        self.assertEqual(len(res['episodes']), 15)
        self.assertFalse(res['truncated'])           # +1 probe proves exhaustion

    def test_omitted_limit_is_the_bounded_default_not_unbounded(self):
        # Omitting limit → EPISODE_DEFAULT_LIMIT (bounded), NOT unbounded — the
        # signature default matches filter_nodes; unbounded is explicit None.
        from servers.brain_constants import EPISODE_DEFAULT_LIMIT
        res = self._episodes()                       # no limit → default page
        self.assertEqual(len(res['episodes']), EPISODE_DEFAULT_LIMIT)  # 10, not 15
        self.assertTrue(res['truncated'])            # 15 > 10 → loud, not silent


if __name__ == '__main__':
    unittest.main()
