"""Arg-sanity guard at the execute_tool chokepoint.

Haiku's tool_use serialization occasionally leaks function-call markup into
string arg values (2×/1,848 production calls, both recall_by_time end_when).
The guard drops the corrupted arg, keeps the call, and logs loudly
(surface_malformed_tool_arg). These tests pin: detection on the two REAL
production values, non-breakage (results still flow), loud logging, the
dropped_args passthrough, and no false positives on ordinary queries.
"""

from servers.scales.s1.fetch_tools import _MARKUP_ARG_RE, execute_tool
from tests.brain_test_base import BrainTestBase

# The two real production corruptions, verbatim (traces 2026-06-30 / 2026-07-15).
PROD_BAD_1 = '</antml parameter>\n<parameter name="time_anchor">event'
PROD_BAD_2 = '</antml :parameter>\n<parameter name="time_anchor">discussed'


class TestMarkupArgRegex:
    def test_matches_both_production_corruptions(self):
        assert _MARKUP_ARG_RE.search(PROD_BAD_1)
        assert _MARKUP_ARG_RE.search(PROD_BAD_2)

    def test_matches_generic_xml_tag_and_control_chars(self):
        assert _MARKUP_ARG_RE.search('<tag attr="x">')
        assert _MARKUP_ARG_RE.search('before\x07after')

    def test_no_false_positive_on_ordinary_args(self):
        # Real production queries that contain guard-adjacent words.
        for ok in (
            '0.7 decay threshold running field fatigue prior justification '
            'parameter sensitivity',
            'P3 variations testing all hyperparameters grid sweep',
            'last week',
            '2026-07-14',
            'score < threshold comparison',   # bare '<' is not a tag
        ):
            assert not _MARKUP_ARG_RE.search(ok), ok


class TestExecuteToolGuard(BrainTestBase):
    needs_embedder = False

    def _error_rows(self):
        out = self.brain.query_logs(source='debug', level='error', limit=50)
        return [e for e in out.get('entries', [])
                if e.get('origin') == 'surface_malformed_tool_arg']

    def test_corrupted_optional_arg_dropped_call_continues(self):
        out = execute_tool(
            self.brain, 'recall_by_time',
            {'start_when': 'last week', 'end_when': PROD_BAD_1,
             'query': 'anything', 'limit': 5},
            session_id='test-sess')
        # Non-breaking: no error, results shape intact (list, possibly empty).
        assert out.get('error') is None
        assert isinstance(out.get('results'), list)
        # The drop is visible on the result for traces/captures.
        assert out.get('dropped_args') == {'end_when': PROD_BAD_1[:200]}

    def test_corruption_logs_loud(self):
        before = len(self._error_rows())
        execute_tool(
            self.brain, 'recall_by_time',
            {'start_when': 'yesterday', 'end_when': PROD_BAD_2},
            session_id='test-sess')
        rows = self._error_rows()
        assert len(rows) > before
        assert 'end_when' in str(rows[0])

    def test_clean_args_untouched_no_dropped_key(self):
        out = execute_tool(
            self.brain, 'recall_by_time',
            {'start_when': 'last week', 'query': 'parameter sensitivity'},
            session_id='test-sess')
        assert 'dropped_args' not in out
        assert out.get('error') is None

    def test_corrupted_required_arg_degrades_to_error_not_crash(self):
        # Dropping a REQUIRED arg (query) → fn raises TypeError → error
        # result → Haiku falls back to the candidate pool. Never an exception.
        out = execute_tool(
            self.brain, 'recall_topical',
            {'query': 'find the <parameter name="x">antml</parameter> leak'},
            session_id='test-sess')
        assert isinstance(out.get('results'), list)
        assert out['results'] == []
        assert out.get('error')
        assert out.get('dropped_args', {}).get('query')
