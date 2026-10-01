"""The bounded-read truncation contract (contract.py) — flag-or-exempt.

A windowed read that hits its row limit covers only the most recent slice of
the requested window while looking complete (the 2026-08-06 cost tally read
2 days as 7). The contract: every read door that claims WINDOW coverage
attaches the `truncated` payload when saturated; ranked top-k doors are
exempt — there, truncation IS the contract.

Two halves:
- Behavior: each flagged door produces the payload under a saturating
  fixture and stays silent when the window fits.
- Enumeration: every MCP read tool whose schema accepts a window param plus
  `limit` must be in FLAGGED or EXEMPT below — a new door can't ship silent.

(query_traces door behaviors are pinned in test_query_traces_truncation.py.)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from tests.test_query_traces_truncation import _events


# ── The contract roster ────────────────────────────────────────────────────
# Every MCP read tool with (window param + limit) belongs to exactly one set.

FLAGGED = {
    'query_traces',      # hours window — flags via +1 probe (brain_traces)
    'query_logs',        # hours window — flags via DAL's exact COUNT(*)
    'filter_nodes',      # gt/lt bounds — flags via +1 probe (brain_recall)
    'recall_episodes',   # older/younger bounds — time path flags via +1
                         # probe; semantic path is ranked top-k (bare bool)
    'get_traces',        # no window, but a TRACE_BATCH_MAX_IDS cap on the id
                         # list — flags via truncation_payload_ids (dispatch).
                         # The enumeration test below can't see this one: the
                         # cap isn't a schema `limit`, it's the list length.
    'get_nodes',         # same id-list shape (GET_NODES_MAX_IDS); its result
                         # is a list, so the payload's banner rides an
                         # `actions` reply note (dispatch_read) — pinned in
                         # test_mcp_roundtrip.
}

EXEMPT = {
    # tool: reason (truncation is the contract, not a lie)
    'recall': 'ranked top-k — relevance claim, not coverage',
    'recall_batch': 'ranked top-k per query',
    'get_trace': 'one row by id — nothing to cover, nothing to cut',
    'count_traces': 'returns counts, not rows',
}

WINDOW_PARAMS = {'hours', 'older_than', 'younger_than', 'gt', 'lt'}


class TestEnumeration(unittest.TestCase):
    """Every windowed+limited MCP read tool made a flag-or-exempt decision."""

    def test_every_windowed_read_tool_is_classified(self):
        from servers.brain_mcp import TOOLS
        unclassified = []
        for tool in TOOLS:
            props = set((tool.get('inputSchema') or {})
                        .get('properties', {}).keys())
            if 'limit' in props and props & WINDOW_PARAMS:
                name = tool['name']
                if name not in FLAGGED and name not in EXEMPT:
                    unclassified.append(name)
        self.assertEqual(
            unclassified, [],
            'read tools with a window param + limit must flag truncation '
            'or be EXEMPT with a reason (see contract.py truncation '
            'contract): %r' % unclassified)


class TestQueryLogs(BrainTestBase):
    needs_embedder = False

    def _seed_errors(self, n):
        for i in range(n):
            self.brain._logs_dal.log_hook_error(
                'trunc_probe', 'e%d' % i, context='ctx')

    def test_saturated_logs_flag_with_exact_counts(self):
        self._seed_errors(8)
        res = self.brain.query_logs(source='errors', hours=1, limit=3)
        self.assertIn('truncated', res)
        self.assertEqual(res['truncated']['limit'], 3)
        self.assertIn('of', res['truncated']['note'])  # 'N of M matching'

    def test_unsaturated_logs_stay_silent(self):
        self._seed_errors(2)
        res = self.brain.query_logs(source='errors', hours=1, limit=50)
        self.assertNotIn('truncated', res)


class TestFilterNodes(BrainTestBase):
    needs_embedder = False

    def _seed_nodes(self, n):
        for i in range(n):
            self.brain.remember(type='truncprobe', title='n%d' % i,
                                content='c%d' % i)

    def test_saturated_filter_flags(self):
        self._seed_nodes(6)
        res = self.brain.filter_nodes(field='type', include=['truncprobe'],
                                      limit=4, rich=False)
        self.assertEqual(len(res['nodes']), 4)
        self.assertIn('truncated', res)
        self.assertEqual(res['truncated']['limit'], 4)

    def test_exact_fit_filter_stays_silent(self):
        self._seed_nodes(5)
        res = self.brain.filter_nodes(field='type', include=['truncprobe'],
                                      limit=5, rich=False)
        self.assertEqual(len(res['nodes']), 5)
        self.assertNotIn('truncated', res)


class TestRecallEpisodesTimePath(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.brain._trace_dal.append_batch(_events(10))

    def test_saturated_time_path_carries_payload(self):
        res = self.brain.recall_episodes(
            contains='event', scale='s0', ref_type='tool_result', limit=4)
        self.assertEqual(len(res['episodes']), 4)
        self.assertIsInstance(res['truncated'], dict)
        self.assertEqual(res['truncated']['limit'], 4)

    def test_exact_fit_time_path_is_false(self):
        res = self.brain.recall_episodes(
            contains='event', scale='s0', ref_type='tool_result', limit=10)
        self.assertEqual(len(res['episodes']), 10)
        self.assertIs(res['truncated'], False)


class TestProbeAliveAtTheCap(BrainTestBase):
    """The +1 probe must fire the truncation flag at exactly limit=500
    (2026-08-07 finding 1: a clamp without probe headroom made the flag
    structurally dead at that operating point — silent backlog skip). The
    door is now honest — it no longer clamps over-cap requests; the
    agent-facing clamp lives at the dispatch boundary — but the probe at
    the cap value stays intact."""
    needs_embedder = False

    def test_recall_episodes_flags_at_episode_max_limit(self):
        from servers.brain_constants import EPISODE_MAX_LIMIT
        self.brain._trace_dal.append_batch(_events(EPISODE_MAX_LIMIT + 1))
        res = self.brain.recall_episodes(
            contains='event', scale='s0', ref_type='tool_result',
            limit=EPISODE_MAX_LIMIT)
        self.assertEqual(len(res['episodes']), EPISODE_MAX_LIMIT)
        self.assertIsInstance(res['truncated'], dict)
        self.assertEqual(res['truncated']['limit'], EPISODE_MAX_LIMIT)

    def test_over_cap_request_is_honest_at_the_door(self):
        # The door no longer clamps an over-cap request — that clamp moved to
        # the dispatch boundary. An internal caller asking above the cap gets
        # every matching row honestly: asked for MAX+50, MAX+1 exist → all
        # MAX+1 returned, nothing truncated (you got everything).
        from servers.brain_constants import EPISODE_MAX_LIMIT
        self.brain._trace_dal.append_batch(_events(EPISODE_MAX_LIMIT + 1))
        res = self.brain.recall_episodes(
            contains='event', scale='s0', ref_type='tool_result',
            limit=EPISODE_MAX_LIMIT + 50)
        self.assertEqual(len(res['episodes']), EPISODE_MAX_LIMIT + 1)
        self.assertFalse(res['truncated'])
        # (The relocated dispatch clamp is covered by TestAgentLimit in
        # test_mcp_roundtrip.py — no need to restate it here.)


class TestS2ReadTracesWindow(BrainTestBase):
    """S2's _read_traces_window (scales/s2/base.py) is uncapped: the window
    is the claim and consumers aggregate over the result (co-recall counts) —
    at 633 recall rows/168h a limit=500 cap truncated every consolidation
    cycle (s2_consolidation_traces_truncated wallpaper, 2026-09) and biased
    the aggregates toward the newest slice."""
    needs_embedder = False

    def test_windowed_read_covers_whole_window_beyond_500(self):
        from servers.scales.s2.base import IntegrationUnit
        self.brain._trace_dal.append_batch(_events(507))
        unit = IntegrationUnit(self.brain)
        rows = unit._read_traces_window(
            's0', hours=24, ref_types=['tool_result'])
        self.assertEqual(len(rows), 507)


if __name__ == '__main__':
    unittest.main()
