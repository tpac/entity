"""Unit tests for servers/scales/s1/fetch_tools.py.

Each tool is exercised against a seeded test brain with the embedder loaded —
the tools recall, so they need a graph that carries vectors and a model to
encode the query. Tests verify:
  - Tool returns the expected candidate shape (id/title/type/score/discovery).
  - Tool handles edge cases (empty query, unknown aspect, no FTS5 hits).
  - Natural-language parsing produces reasonable date windows.
"""
import os
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from tests.brain_test_base import BrainTestBase
from tests.isolated_brain import brain_env
from servers.scales.s1.fetch_tools import (
    TOOL_DEFINITIONS,
    _parse_window,
    _parse_date_expr,
    recall_topical,
    recall_by_time,
    expand_node,
    execute_tool,
    format_tool_result_for_haiku,
)


# ─── Natural-language parsers (pure, no brain needed) ────────────────────

class TestWindowParser:
    def test_last_n_hours(self):
        now = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
        start, end = _parse_window('last 10 hours', now=now)
        assert end == now
        assert start == now - timedelta(hours=10)

    def test_last_nh_shorthand(self):
        now = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
        start, end = _parse_window('24h', now=now)
        assert start == now - timedelta(hours=24)

    def test_yesterday(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('yesterday', now=now)
        assert start == datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)
        assert end == datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc)

    def test_today(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('today', now=now)
        assert start == datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc)
        assert end == now

    def test_last_session_heuristic(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('since last session', now=now)
        # Heuristic: last 24h
        assert (now - start).total_seconds() == 24 * 3600

    def test_unknown_falls_back_to_24h(self):
        now = datetime(2026, 5, 10, 14, 30, tzinfo=timezone.utc)
        start, end = _parse_window('blah random', now=now)
        # 24h is the conservative fallback
        assert (now - start).total_seconds() == 24 * 3600

    def test_n_units_ago_is_a_point(self):
        # A point in time; a digit is never read as a day of the current month.
        now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        for phrase, point in [
                ('2 weeks ago', now - timedelta(weeks=2)),
                ('2w ago', now - timedelta(weeks=2)),
                ('about 3 days ago', now - timedelta(days=3)),
                ('a few days ago', now - timedelta(days=3)),
                ('an hour ago', now - timedelta(hours=1)),
                ('2 months ago', datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc))]:
            assert _parse_window(phrase, now=now) == (point, point), phrase

    def test_last_n_months_and_last_month_are_rolling(self):
        now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        assert _parse_window('last 2 months', now=now) == \
            (datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc), now)
        assert _parse_window('last month', now=now) == \
            (datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc), now)
        assert _parse_window('last few hours', now=now) == (now - timedelta(hours=3), now)

    def test_conversation_phrases_use_session_heuristic(self):
        now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        for phrase in ('last 3 turns', 'this session', 'last message',
                       'turn 6', 'just now'):
            assert _parse_window(phrase, now=now) == \
                (now - timedelta(hours=24), now), phrase
        assert _parse_window('last 2 sessions', now=now) == \
            (now - timedelta(hours=48), now)

    def test_digits_in_free_text_are_not_days(self):
        now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        unrecognized = []
        assert _parse_window('step 7 shipped', now=now,
                             unrecognized=unrecognized) == \
            (now - timedelta(hours=24), now)
        assert unrecognized == ['step 7 shipped']
        # An out-of-range count is unrecognized too, not a silent None upstream.
        assert _parse_window('last 100000 years', now=now,
                             unrecognized=unrecognized) == \
            (now - timedelta(hours=24), now)
        assert unrecognized == ['step 7 shipped', 'last 100000 years']

    def test_literal_datetime_parses_and_is_not_unrecognized(self):
        now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        unrecognized = []
        start, _ = _parse_window('2026-09-23T20:00Z', now=now,
                                 unrecognized=unrecognized)
        assert start == datetime(2026, 9, 23, 20, 0, tzinfo=timezone.utc)
        assert unrecognized == []


class TestDateParser:
    def test_since_iso(self):
        since, until = _parse_date_expr('since 2026-05-01')
        assert since is not None
        assert since.year == 2026
        assert since.month == 5
        assert since.day == 1

    def test_before_iso(self):
        since, until = _parse_date_expr('before 2026-04-30')
        assert until is not None
        assert until.year == 2026
        assert until.month == 4
        assert until.day == 30

    def test_on_specific_day(self):
        since, until = _parse_date_expr('on 2026-05-09')
        assert since.day == 9
        assert until.day == 10  # day after

    def test_dict_form(self):
        since, until = _parse_date_expr({'since': '2026-05-01', 'until': '2026-05-10'})
        assert since.day == 1
        assert until.day == 10

    def test_before_relative_phrase(self):
        now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
        assert _parse_date_expr('before today', now=now) == \
            (None, datetime(2026, 9, 29, 0, 0, tzinfo=timezone.utc))
        assert _parse_date_expr('before now', now=now) == (None, now)


# ─── Tool definitions are well-formed ────────────────────────────────────

class TestToolDefinitions:
    def test_two_tools_defined(self):
        # Haiku's surface tool set. expand_node stays callable (tested below)
        # but is not offered — re-offering it is an open decision (6a9062d8).
        names = {t['name'] for t in TOOL_DEFINITIONS}
        assert names == {'recall_topical', 'recall_by_time'}

    def test_each_tool_has_input_schema(self):
        for tool in TOOL_DEFINITIONS:
            assert 'input_schema' in tool
            assert tool['input_schema']['type'] == 'object'
            assert 'properties' in tool['input_schema']

    def test_descriptions_substantive(self):
        for tool in TOOL_DEFINITIONS:
            assert len(tool['description']) > 80, \
                f"{tool['name']} description too short"


# ─── Tool execution against a seeded brain ───────────────────────────────

@pytest.fixture(scope='module')
def env():
    with brain_env(needs_embedder=True) as env:
        yield env


def _event_ids(b, event_type, source=None) -> set:
    """Set of debug_log row ids for an event_type (optionally a specific source).

    The discussed-anchor tests assert on what a single recall_by_time call
    WROTE, and a global COUNT(*) delta is not a safe proxy for that: anything
    that DELETEs debug_log rows concurrently — `Brain._check_logs_db_size`
    prunes rows older than 7 days once the logs db passes 50MB — moves the
    count without the call under test writing anything. That was observed to
    flip the sign of the delta day to day (proven flaky 2026-06-25).

    `debug_log.id` is INTEGER PRIMARY KEY AUTOINCREMENT, so ids are monotonic
    and never reused. Set-differencing this across a call yields EXACTLY the
    rows that call inserted, immune to any prune (a DELETE removes old, low
    ids; it can never produce a new one). event_type='error' is the loud,
    dashboard-surfaced severity; 'warning' is the quiet §6 signal — diffing
    these (not the deleted 'fetch_by_time_archived_leak' string, which now has
    zero writers) is what keeps the 'quiet' check non-tautological."""
    if source is not None:
        rows = b.logs_conn.execute(
            "SELECT id FROM debug_log WHERE event_type=? AND source=?",
            (event_type, source)).fetchall()
    else:
        rows = b.logs_conn.execute(
            "SELECT id FROM debug_log WHERE event_type=?",
            (event_type,)).fetchall()
    return {r[0] for r in rows}


class TestRecallTopical:
    def test_returns_candidates(self, env):
        results = recall_topical(env.brain, query='partnership target function', k=5)
        assert isinstance(results, list)
        # Non-vacuous: the per-candidate checks below are a loop, and a loop
        # over nothing asserts nothing. The seed pack answers this query.
        assert results, 'recall_topical returned nothing — the shape checks below would be vacuous'
        for r in results:
            assert 'id' in r
            assert 'title' in r
            assert 'score' in r
            assert r['discovery'] == 'recall_topical'

    def test_empty_query_handled(self, env):
        results = recall_topical(env.brain, query='', k=5)
        assert isinstance(results, list)
        # Empty query must not crash AND must not yield malformed candidates:
        # anything returned still carries the tool's source tag.
        for r in results:
            assert r['discovery'] == 'recall_topical'

    def test_score_maps_effective_activation(self, env):
        """Score-contract pin: the tool's score MUST be recall_score(result)
        (= effective_activation), the same field daemon_hooks gives the
        cosine pool. When these fork, the admission floor in surface.py
        silently drops 100% of fetches (the 3-week death of 2026-07)."""
        real_recall = env.brain.recall
        try:
            env.brain.recall = lambda **kw: {'results': [
                {'id': 'aaaa1111', 'title': 't1', 'effective_activation': 0.42},
                {'id': 'bbbb2222', 'title': 't2', 'effective_activation': 0.17},
            ]}
            out = recall_topical(env.brain, query='anything', k=5)
        finally:
            env.brain.recall = real_recall
        assert [c['score'] for c in out] == [0.42, 0.17]

    def test_zero_raw_results_logs_warning(self, env):
        """recall_topical returning 0 raw results means the tool broke, not
        'no matches' — cosine top-k always fills k. Must warn loudly."""
        before = _event_ids(env.brain, 'warning', source='fetch_topical_zero_raw')
        real_recall = env.brain.recall
        try:
            env.brain.recall = lambda **kw: {'results': []}
            out = recall_topical(env.brain, query='anything', k=5)
        finally:
            env.brain.recall = real_recall
        assert out == []
        after = _event_ids(env.brain, 'warning', source='fetch_topical_zero_raw')
        assert len(after - before) == 1


class TestScoreContractSync:
    """Every score producer must route through surface_contract.recall_score.

    Source-level pin: a refactor that reverts any of them to a raw field
    read re-opens the silent-drift class this function exists to close."""

    def _source(self, relpath):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, relpath)) as f:
            return f.read()

    def test_daemon_hooks_uses_recall_score(self):
        src = self._source('servers/daemon_hooks.py')
        assert 'node_data["score"] = recall_score(r)' in src

    def test_fetch_tools_uses_recall_score(self):
        src = self._source('servers/scales/s1/fetch_tools.py')
        assert 'recall_score(r)' in src
        # No raw 'score' read in any form: brain.recall results carry no
        # 'score' key, so whatever default the read names becomes the score.
        assert not re.search(r"""\.get\(\s*['"]score['"]""", src)
        # Nor a raw effective_activation read: right field, but it skips
        # recall_score's type guard — the one-owner fork of encode's pin.
        assert not re.search(r"""\.get\(\s*['"]effective_activation['"]""", src)

    def test_encode_stub_ranking_uses_recall_score(self):
        # third producer (associated-stub turnmax) — same pin, same drift class
        src = self._source('servers/scales/s1/encode.py')
        assert 'recall_score(h)' in src
        assert "h.get('effective_activation')" not in src


class TestRecallByTimeDiscussed:
    # recall_recent was DELETED 2026-06-12 — its use case ('the thing we
    # talked about 3 weeks ago') moved to recall_by_time(time_anchor=
    # 'discussed'), which reads surface-selection traces (s1r K events).
    def test_discussed_anchor_finds_traced_node(self, env):
        import json as _json
        b = env.brain
        rows = b.conn.execute("SELECT id FROM nodes LIMIT 1").fetchall()
        assert rows, "fixture brain has no nodes"
        node_id = rows[0][0]
        # Far-past window so ONLY this synthetic trace matches — a rolling
        # window ('yesterday') competes with real production traces in the
        # copied logs db and the limit cut can drop the test node.
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES ('te5td15c', 's1r-test-1', 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', "
            "'2009-01-15T12:00:00+00:00')",
            (_json.dumps([node_id]),))
        b.logs_conn.commit()
        results = recall_by_time(b, start_when='January 2009',
                                 end_when='February 2009', time_anchor='discussed')
        ids = {r.get('id') for r in results}
        assert node_id in ids, "discussed anchor should surface the traced node"

    def test_discussed_anchor_empty_window_no_crash(self, env):
        results = recall_by_time(env.brain, start_when='January 2001',
                                 end_when='February 2001', time_anchor='discussed')
        assert isinstance(results, list)

    def test_discussed_anchor_redirects_archived_to_survivor(self, env):
        # TRACE-NODE-RESOLUTION site #1: a trace can point at a node S2
        # absorbed since. The anchor must resolve forward to the live
        # survivor (the thing we discussed survives as its descendant) — the
        # survivor surfaces, the dead id does not — and do it QUIETLY (the
        # old loud `fetch_by_time_archived_leak` stopgap is gone; §6).
        import json as _json
        b = env.brain
        live = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 LIMIT 1").fetchone()[0]
        src = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 AND id != ? "
            "LIMIT 1", (live,)).fetchone()[0]
        # Absorb `src` into `live`: archive it + stamp the survivor pointer.
        b.conn.execute("UPDATE nodes SET archived=1 WHERE id=?", (src,))
        b.conn.execute(
            "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) "
            "VALUES (?, '_sys_archived_survivor_id', ?)", (src, live))
        b.conn.commit()
        # Isolated far-past window so only this synthetic trace matches.
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES ('te5tredir', 's1r-test-3', 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', "
            "'2007-05-15T12:00:00+00:00')",
            (_json.dumps([src]),))
        b.logs_conn.commit()
        # `live in ids` is the positive control (get_node is a raw point
        # lookup that does NOT itself resolve survivors, so `live` can only
        # appear if resolve_live actually redirected). Quietness = no loud
        # error row added by this call (event_type='error' is what the
        # dashboard surfaces); a routine redirect logs nothing.
        err_before = _event_ids(b, 'error')
        results = recall_by_time(b, start_when='May 2007',
                                 end_when='June 2007', time_anchor='discussed')
        ids = {r.get('id') for r in results}
        assert src not in ids, "absorbed source must not surface"
        assert live in ids, "survivor must surface in place of the absorbed node"
        new_errs = _event_ids(b, 'error') - err_before
        assert not new_errs, \
            "routine redirect must not write a loud error row (new ids=%s)" % sorted(new_errs)

    def test_discussed_anchor_drops_orphan_quietly(self, env):
        # An archived node with NO survivor pointer is a true orphan: dropped,
        # never surfaced, with no loud error — but COUNTED via a low-severity
        # warning (§6). `ctrl` is a LIVE sibling in the same trace: the
        # positive control. It must still surface, so the test fails if
        # resolve_live returns nothing (rather than passing vacuously on an
        # empty result, e.g. a swallowed crash).
        import json as _json
        b = env.brain
        src = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 LIMIT 1").fetchone()[0]
        ctrl = b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 AND id != ? "
            "LIMIT 1", (src,)).fetchone()[0]
        b.conn.execute("UPDATE nodes SET archived=1 WHERE id=?", (src,))
        b.conn.execute(
            "DELETE FROM node_metadata_kv WHERE node_id=? "
            "AND key='_sys_archived_survivor_id'", (src,))
        b.conn.commit()
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES ('te5torph', 's1r-test-4', 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', "
            "'2006-07-15T12:00:00+00:00')",
            (_json.dumps([src, ctrl]),))
        b.logs_conn.commit()
        err_before = _event_ids(b, 'error')
        warn_before = _event_ids(b, 'warning', 'fetch_by_time_orphans_dropped')
        results = recall_by_time(b, start_when='July 2006',
                                 end_when='August 2006', time_anchor='discussed')
        ids = {r.get('id') for r in results}
        assert src not in ids, "true orphan must be dropped"
        assert ctrl in ids, "live sibling must still surface (positive control)"
        new_errs = _event_ids(b, 'error') - err_before
        assert not new_errs, \
            "orphan drop must not write a loud error row (new ids=%s)" % sorted(new_errs)
        new_warns = _event_ids(b, 'warning', 'fetch_by_time_orphans_dropped') - warn_before
        assert new_warns, \
            "orphan drop must be COUNTED via a low-severity warning (§6)"


class TestRecallByTime:
    def test_open_ended_start_returns_results(self, env):
        # recall_by_date(when='since 2026-04-01') → recall_by_time with an
        # open-ended upper bound. Needs a query for tiers 1/2; time-only is tier 3.
        results = recall_by_time(env.brain, start_when='2026-04-01',
                                 query='partnership', limit=10)
        assert isinstance(results, list)
        assert results, 'open-ended start returned nothing — the name of this test is the assertion'
        for r in results:
            assert r['discovery'] == 'recall_by_time'

    def test_bounded_range(self, env):
        # recall_by_date(when={'since':..., 'until':...}) → start_when/end_when.
        results = recall_by_time(env.brain, start_when='2026-04-01',
                                 end_when='2026-05-10', query='partnership', limit=5)
        assert isinstance(results, list)
        for r in results:
            assert r['discovery'] == 'recall_by_time'

    def _one_node_in_window(self, b, trace_id, created_at):
        """Two live node ids; the first is placed in a far-past 'discussed'
        window (a synthetic surface_selected trace) so it is the only
        time match there."""
        import json as _json
        a, c = [r[0] for r in b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 LIMIT 2")]
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES (?, ?, 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', ?)",
            (trace_id, 's1r-' + trace_id, _json.dumps([a]), created_at))
        b.logs_conn.commit()
        return a, c

    def test_query_tier_scores_are_recall_score(self, env):
        """Score-contract pin for the query tiers: each candidate's score is
        recall_score(result), the pool's semantic, with no tier discount.
        Tier precedence is the emission order — here the query-only hit
        outscores the query+time hit and still comes second."""
        b = env.brain
        a, c = self._one_node_in_window(b, 'te5tqtr1', '2005-03-15T12:00:00+00:00')
        real_recall = b.recall
        try:
            b.recall = lambda **kw: {'results': [
                {'id': c, 'title': 't-query', 'effective_activation': 0.88},
                {'id': a, 'title': 't-both', 'effective_activation': 0.31},
            ]}
            out = recall_by_time(b, start_when='March 2005', end_when='April 2005',
                                 time_anchor='discussed', query='anything')
        finally:
            b.recall = real_recall
        assert [(r['id'], r['tier'], r['score']) for r in out] == \
            [(a, 'query+time', 0.31), (c, 'query', 0.88)]

    def _trace(self, b, trace_id, ids, created_at):
        import json as _json
        b.logs_conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, ref_id, summary, session_id, created_at) "
            "VALUES (?, ?, 's1', 'K', 'surface_selected', ?, 'test', 'test-sess', ?)",
            (trace_id, 's1r-' + trace_id, _json.dumps(ids), created_at))
        b.logs_conn.commit()

    def _live_ids(self, b, n):
        return [r[0] for r in b.conn.execute(
            "SELECT id FROM nodes WHERE COALESCE(archived,0)=0 ORDER BY id LIMIT ?", (n,))]

    def test_time_tier_discussed_is_newest_discussion_first(self, env):
        """Tier 3 emits in time order, the same in every process: a node's key
        is its LATEST discussion in the window."""
        b = env.brain
        a, c, d = self._live_ids(b, 3)
        self._trace(b, 'te5tord1', [a], '2003-01-10T12:00:00+00:00')
        self._trace(b, 'te5tord2', [c], '2003-01-11T12:00:00+00:00')
        self._trace(b, 'te5tord3', [d, a], '2003-01-12T12:00:00+00:00')
        ids = lambda lim: [r['id'] for r in recall_by_time(
            b, start_when='January 2003', end_when='January 2003',
            time_anchor='discussed', limit=lim)]
        assert ids(3) == [d, a, c]
        assert ids(2) == [d, a]

    def test_time_tier_created_is_newest_first(self, env):
        b = env.brain
        picked = self._live_ids(b, 3)
        before = dict(b.conn.execute(
            "SELECT id, created_at FROM nodes WHERE id IN (?,?,?)", picked).fetchall())
        try:
            for i, nid in enumerate(picked):
                b.conn.execute("UPDATE nodes SET created_at=? WHERE id=?",
                               ('2002-03-0%dT12:00:00+00:00' % (i + 1), nid))
            b.conn.commit()
            out = recall_by_time(b, start_when='March 2002', end_when='March 2002',
                                 time_anchor='created', limit=2)
            assert [(r['id'], r['tier']) for r in out] == \
                [(picked[2], 'time'), (picked[1], 'time')]
        finally:
            for nid, ca in before.items():
                b.conn.execute("UPDATE nodes SET created_at=? WHERE id=?", (ca, nid))
            b.conn.commit()

    @contextmanager
    def _with_event_times(self, b, values):
        """Set event_time on live nodes for the test, restoring after. Yields
        the node ids in `values` order."""
        ids = self._live_ids(b, len(values))
        before = {nid: b._meta_kv.get_field(nid, 'event_time') for nid in ids}
        try:
            for nid, v in zip(ids, values):
                b._meta_kv.set(nid, 'event_time', v)
            b.conn.commit()
            yield ids
        finally:
            for nid, v in before.items():
                if v is None:
                    b._meta_kv.delete(nid, 'event_time')
                else:
                    b._meta_kv.set(nid, 'event_time', v)
            b.conn.commit()

    def test_time_tier_event_reads_event_time_newest_first(self, env):
        """The event anchor is the node's own event_time: date-only, datetime
        and offset forms all count, newest first, and no edge tier."""
        b = env.brain
        with self._with_event_times(b, ['1998-06-10', '1998-06-20T09:00:00Z',
                                        '1998-06-05T23:30:00-04:00']) as (
                day, dt, offset):
            out = recall_by_time(b, start_when='June 1998', end_when='June 1998',
                                 time_anchor='event', limit=5)
        assert [(r['id'], r['tier']) for r in out] == \
            [(dt, 'time'), (day, 'time'), (offset, 'time')]

    def test_event_date_only_value_covers_its_day(self, env):
        b = env.brain
        noon = int(datetime(1998, 7, 4, 12, tzinfo=timezone.utc).timestamp())
        with self._with_event_times(b, ['1998-07-04', '1998-07-04T10:00:00Z']) as (
                day, morning):
            hits = b._meta_kv.node_ids_with_time_in('event_time', noon, noon + 3600)
        assert hits == [day]

    def test_event_excludes_archived_nodes(self, env):
        b = env.brain
        with self._with_event_times(b, ['1997-02-02']) as (nid,):
            try:
                b.conn.execute("UPDATE nodes SET archived=1 WHERE id=?", (nid,))
                hits = b._meta_kv.node_ids_with_time_in(
                    'event_time', 0, 9_999_999_999)
            finally:
                b.conn.execute("UPDATE nodes SET archived=0 WHERE id=?", (nid,))
                b.conn.commit()
        assert nid not in hits

    def test_edge_tier_emits_in_callers_order(self, env):
        from servers.scales.s1.fetch_tools import _fetch_edges_with_endpoints
        b = env.brain
        e1, e2 = [r[0] for r in b.conn.execute(
            "SELECT DISTINCT edge_id FROM edge_relations "
            "WHERE COALESCE(archived,0)=0 ORDER BY edge_id LIMIT 2")]
        order = lambda ids: [c['id'] for c in _fetch_edges_with_endpoints(b, ids)
                             if c.get('tier') == 'edge_time']
        assert order([e2, e1])[0] == e2
        assert order([e1, e2])[0] == e1

    def test_unrecognized_window_is_logged(self, env):
        b = env.brain
        before = _event_ids(b, 'warning', 'fetch_by_time_unparsed_window')
        recall_by_time(b, start_when='step 7 shipped', time_anchor='discussed', limit=1)
        assert _event_ids(b, 'warning', 'fetch_by_time_unparsed_window') - before

    def test_recall_failure_logs_and_keeps_time_tier(self, env):
        """A raising brain.recall drops the query tiers LOUDLY: one error row,
        and the time tier still answers."""
        b = env.brain
        a, _ = self._one_node_in_window(b, 'te5tqtr2', '2004-03-15T12:00:00+00:00')
        before = _event_ids(b, 'error', source='fetch_by_time_semantic')

        def _boom(**kw):
            raise RuntimeError('recall broke')
        real_recall = b.recall
        try:
            b.recall = _boom
            out = recall_by_time(b, start_when='March 2004', end_when='April 2004',
                                 time_anchor='discussed', query='anything')
        finally:
            b.recall = real_recall
        assert [(r['id'], r['tier']) for r in out] == [(a, 'time')]
        after = _event_ids(b, 'error', source='fetch_by_time_semantic')
        assert len(after - before) == 1


class TestExpandNode:
    def test_unknown_ref_returns_empty(self, env):
        results = expand_node(env.brain, node_ref='zzzzzzzz_not_a_node', hops=1)
        assert results == []


class TestExpandNodeEndToEnd(BrainTestBase):
    """expand_node against a small fresh brain with a known seed→neighbor
    constellation. Regression for the traverse() shape bug: traverse returns
    {'neighbors': [...], 'corrections': {...}, 'metadata': {...}} but
    expand_node iterated the dict itself (string keys → AttributeError,
    swallowed by the except) — so the tool returned 0 candidates on every
    production call."""
    needs_embedder = False

    def _node(self, title):
        r = self.brain.remember(type='fact', title=title, content='content of %s' % title,
                                encoding_source='anchor')
        return r['id']

    def setUp(self):
        super().setUp()
        self.seed = self._node('seed node about daemon restarts')
        self.nb1 = self._node('neighbor one — restart ladder')
        self.nb2 = self._node('neighbor two — launchd kickstart')
        self.brain.connect(self.seed, self.nb1, relation='extends', weight=0.8)
        self.brain.connect(self.seed, self.nb2, relation='grounds', weight=0.6)

    def test_expand_by_id_returns_neighbor_candidates(self):
        results = expand_node(self.brain, node_ref=self.seed, hops=1)
        ids = {r['id'] for r in results}
        self.assertEqual(ids, {self.nb1, self.nb2})
        for r in results:
            self.assertEqual(r['discovery'], 'expand_node')
            self.assertEqual(r['discovery'], 'expand_node')
            self.assertTrue(r.get('title'))

    def test_expand_by_title_returns_neighbor_candidates(self):
        results = expand_node(self.brain, node_ref='seed node about daemon restarts',
                              hops=1)
        ids = {r['id'] for r in results}
        self.assertEqual(ids, {self.nb1, self.nb2})

    def test_expand_via_execute_tool_dispatch(self):
        # Full path: dispatch → tool → boundary enrichment. No error key,
        # and no fetch_expand_node row logged (the old failure mode logged
        # one per call while returning []).
        out = execute_tool(self.brain, 'expand_node',
                           {'node_ref': self.seed, 'hops': 1})
        self.assertNotIn('error', out)
        ids = {r['id'] for r in out['results']}
        self.assertEqual(ids, {self.nb1, self.nb2})
        self.assertFalse(_event_ids(self.brain, 'error', source='fetch_expand_node'))


# ─── Dispatch + format helpers ───────────────────────────────────────────

class TestExecuteDispatch:
    def test_known_tool(self, env):
        result = execute_tool(env.brain, 'recall_topical',
                              {'query': 'brain', 'k': 3})
        assert 'results' in result
        assert 'latency_ms' in result
        assert isinstance(result['results'], list)

    def test_unknown_tool_returns_error(self, env):
        result = execute_tool(env.brain, 'nonexistent_tool', {})
        assert result['results'] == []
        assert 'error' in result
        assert 'unknown_tool' in result['error']

    def test_by_time_dispatch_without_session(self, env):
        # recall_recent (and its session_id special-case) deleted 2026-06-12.
        result = execute_tool(env.brain, 'recall_by_time',
                              {'start_when': 'yesterday', 'time_anchor': 'discussed'},
                              session_id='test-session')
        assert 'results' in result


class TestFormatToolResult:
    def test_format_with_results(self):
        result = {'results': [
            {'id': 'abc12345', 'title': 'Test node', 'type': 'lesson', 'score': 0.85},
            {'id': 'def67890', 'title': 'Another', 'type': 'principle', 'score': 0.72},
        ]}
        s = format_tool_result_for_haiku(result)
        assert '2 results' in s
        assert 'abc12345' in s
        assert 'Test node' in s

    def test_format_empty(self):
        s = format_tool_result_for_haiku({'results': []})
        assert 'No results' in s

    def test_format_error(self):
        s = format_tool_result_for_haiku({'results': [], 'error': 'boom'})
        assert 'ERROR' in s
        assert 'boom' in s

    def test_tool_result_block_is_the_candidate_render(self):
        """One renderer for the whole menu: a tool-fetched node must render
        byte-identically to the same node in the initial cosine pool. The
        two halves arrive in one prompt and Haiku is told to judge them on
        the same merit — a second render here would teach it two grammars
        and hide a data-richness asymmetry (which is what the `layout`
        argument this function used to take could silently introduce)."""
        from servers.scales.s1.surface_contract import (
            format_candidate_for_surface)
        node = {'id': 'abc12345', 'type': 'lesson', 'title': 'Test node',
                'content': 'body', 'locked': True,
                'discovery': 'recall_topical', 'situation': 'When testing'}
        block = format_tool_result_for_haiku({'results': [node]})
        assert format_candidate_for_surface(node) in block
        # Scope rides through the same one door.
        scope = {'project': 'brain'}
        scoped = format_tool_result_for_haiku({'results': [
            dict(node, project='other-project')]}, scope=scope)
        assert format_candidate_for_surface(
            dict(node, project='other-project'), scope=scope) in scoped
