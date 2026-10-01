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

    def test_unscoped_pull_is_unchanged(self):
        self._append(chain_id='s0-w-3', session_id='sess-plain')
        out = self.brain.query_traces(ref_type='tool_result', hours=24)
        assert 'window_hours' not in out, (
            "the flag is for scoped pulls, where the session/ref_id reads as "
            "the authority and the window hides behind it")
