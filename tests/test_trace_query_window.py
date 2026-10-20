"""query_traces names the window it applied on scoped ref_type pulls.

The ref_type branch outranks the session modes and stays hours-bound — that
composition is deliberate (get_by_ref_type: "hours composes WITH a session
scope here"). The hazard is that it clips as silently as the limit used to: a
2026-08-16 cost audit read 0 rows for one session and 386 of 1,163 for another
and took both for session totals. `window_hours` makes the bound visible
without changing which rows come back.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestQueryTracesWindowFlag(BrainTestBase):
    needs_embedder = False

    def _append(self, **kw):
        # Seed through the DAL: the write door is dispatch/TraceDAL, and these
        # tests are about what the READ door reports back about its own bounds.
        self.brain._trace_dal.append(scale='s0', event_type='delta',
                                     ref_type='tool_result', summary='x', **kw)

    def test_scoped_pull_names_its_window(self):
        self._append(chain_id='s0-w-1', session_id='sess-window')
        out = self.brain.query_traces(ref_type='tool_result',
                                      session_id='sess-window', hours=24)
        assert out.get('window_hours') == 24, (
            "a session-scoped ref_type pull is hours-bound; the caller must be "
            "able to see that from the payload")

    def test_hours_none_carries_no_window_key(self):
        self._append(chain_id='s0-w-2', session_id='sess-nowindow')
        out = self.brain.query_traces(ref_type='tool_result',
                                      session_id='sess-nowindow', hours=None)
        assert 'window_hours' not in out, (
            "hours=None means unbounded — naming a window would be a lie")

    def test_session_ids_scope_the_ref_type_pull(self):
        # A ref_type pull scoped to several sessions returns only those
        # sessions, as hours-bound as a single-session pull.
        for s in ('sess-many-a', 'sess-many-b', 'sess-many-c'):
            self._append(chain_id='s0-m-' + s, session_id=s)
        out = self.brain.query_traces(ref_type='tool_result', hours=24,
                                      session_ids=['sess-many-a', 'sess-many-b'])
        assert {e['session_id'] for e in out['events']} == {
            'sess-many-a', 'sess-many-b'}
        assert out.get('window_hours') == 24, (
            "a multi-session ref_type pull is as hours-bound as a single one")

    def test_unscoped_pull_is_unchanged(self):
        self._append(chain_id='s0-w-3', session_id='sess-plain')
        out = self.brain.query_traces(ref_type='tool_result', hours=24)
        assert 'window_hours' not in out, (
            "the flag is for scoped pulls, where the session/ref_id reads as "
            "the authority and the window hides behind it")


class TestGroupedQueryTraces(BrainTestBase):
    """grouped=true is honored or refused, never dropped for a flat pull."""
    needs_embedder = False

    def _append(self, session_id, chain_id):
        self.brain._trace_dal.append(scale='s0', event_type='delta',
                                     ref_type='tool_result', summary='g',
                                     chain_id=chain_id, session_id=session_id)

    def test_grouped_session_ids_returns_those_sessions_chains(self):
        for s in ('sess-grp-a', 'sess-grp-b', 'sess-grp-c'):
            self._append(s, 's0-grp-' + s)
        out = self.brain.query_traces(grouped=True, hours=24,
                                      session_ids=['sess-grp-a', 'sess-grp-b'])
        assert 'chains' in out, "grouped + session_ids fell through to a flat pull"
        assert {c['session_id'] for c in out['chains']} == {'sess-grp-a', 'sess-grp-b'}
        assert out.get('window_hours') == 24, (
            "a grouped pull is hours-bound; the payload must say so")

    def test_grouped_refuses_what_it_cannot_honor(self):
        self._append('sess-grp-r', 's0-grp-r')
        for kw in ({},                                   # no session scope
                   {'session_id': 'sess-grp-r', 'ref_type': 'tool_result'},
                   {'session_id': 'sess-grp-r', 'event_type': 'delta'},
                   {'session_id': 'sess-grp-r', 'ref_id': 'n1'}):
            with self.assertRaises(ValueError, msg=repr(kw)):
                self.brain.query_traces(grouped=True, **kw)
