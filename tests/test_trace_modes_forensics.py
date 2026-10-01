"""Tests for the trace-modes forensics contract (commits 8ef9431 + 1f26db8).

Locks the failure-path forensics added after the 2026-07-31 1M-token 400s
(finding 74dfb59c: one ~6M-char brain_batch tool result killed the run):

- runner truncation guard: oversized tool results are capped at
  trace_detail()['tool_result_cap'] BEFORE entering the LLM conversation,
  with result_chars / result_truncated / result_head on the action record
- RunLoopError: mid-run failures carry partial_actions + the original
  exception type name in the message, original exception as __cause__
- retry_on_transient_api_error: matches transients wrapped in RunLoopError
  via __cause__ (the S2 mid-run retry regression), never retries
  non-transients
- build_failed_run_metadata: salvaged forensics stay bounded even when the
  run died because something was enormous
- trace_detail(): mode selection, unknown-mode fallback, and the
  observation-neutrality invariant (tool_result_cap identical across modes)

No real Brain, no live API — same fake-client harness as
test_runner_per_round_stats.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.runner import (RunLoopError, retry_on_transient_api_error,
                                   run_llm_loop)
from servers.trace_contract import (TRACE_MODES, build_failed_run_metadata,
                                    trace_detail)
from tests.test_runner_per_round_stats import (FakeBlock, FakeClient,
                                               FakeMessage, FakeUsage)

# A K stamp for builder calls — the builders require one.
_STAMP = {'fingerprint': 'f0f0f0f0f0f0', 'source': 'default', 'version': 0, 'id': None}


class TestTraceDetail(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop('BRAIN_TRACE_MODE', None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop('BRAIN_TRACE_MODE', None)
        else:
            os.environ['BRAIN_TRACE_MODE'] = self._saved

    def test_default_is_normal(self):
        self.assertEqual(trace_detail(), TRACE_MODES['normal'])

    def test_debug_selected_by_env(self):
        os.environ['BRAIN_TRACE_MODE'] = 'debug'
        self.assertEqual(trace_detail(), TRACE_MODES['debug'])

    def test_unknown_mode_falls_back_to_normal(self):
        os.environ['BRAIN_TRACE_MODE'] = 'vrebose-typo'
        self.assertEqual(trace_detail(), TRACE_MODES['normal'])

    def test_tool_result_cap_is_mode_invariant(self):
        """Observation-neutrality: tool_result_cap bounds what the LLM SEES,
        so debug mode must not change it — a debug run must be the same
        conversation as the normal run it reproduces."""
        self.assertEqual(TRACE_MODES['normal']['tool_result_cap'],
                         TRACE_MODES['debug']['tool_result_cap'])

    def test_debug_widens_row_side_caps(self):
        for cap in ('failed_action_input_cap', 'result_head_cap'):
            self.assertGreater(TRACE_MODES['debug'][cap],
                               TRACE_MODES['normal'][cap])

    def test_recording_shapes_cover_exactly_the_contract_kinds(self):
        """A kind rename in PAYLOAD_KIND_EXT with a stale _NORMAL_ON_KINDS
        would silently flip that kind's normal default to off — pin the
        derivations to the contract."""
        from servers.trace_contract import (_NORMAL_ON_KINDS,
                                            PAYLOAD_KIND_EXT,
                                            TRACE_RECORDING_DEBUG,
                                            TRACE_RECORDING_NORMAL)
        self.assertLessEqual(set(_NORMAL_ON_KINDS), set(PAYLOAD_KIND_EXT))
        self.assertEqual(set(TRACE_RECORDING_NORMAL['kinds']),
                         set(PAYLOAD_KIND_EXT))
        self.assertEqual(set(TRACE_RECORDING_DEBUG['kinds']),
                         set(PAYLOAD_KIND_EXT))
        self.assertTrue(all(TRACE_RECORDING_DEBUG['kinds'].values()))


class TestTruncationGuard(unittest.TestCase):
    """Oversized tool results are truncated before the next API round."""

    def _run_with_result_chars(self, n_chars):
        cap = trace_detail()['tool_result_cap']
        r0 = FakeMessage(
            content=[FakeBlock(type='tool_use', id='t1', name='get_nodes',
                               input={'node_ids': ['x']})],
            stop_reason='tool_use', usage=FakeUsage())
        r1 = FakeMessage(content=[FakeBlock(type='text', text='done')],
                         stop_reason='end_turn', usage=FakeUsage())
        seen_messages = []

        class SpyAPI:
            def __init__(self):
                self._responses = [r0, r1]

            def stream(self, **kwargs):
                seen_messages.append(kwargs.get('messages'))
                from tests.test_runner_per_round_stats import FakeStream
                return FakeStream(self._responses.pop(0))

        client = FakeClient([])
        client.messages = SpyAPI()
        result = run_llm_loop(
            client=client, model='claude-test', max_tokens=4096,
            max_rounds=3, system_prompt='S', user_content='U', tools=[],
            dispatch_fn=lambda name, args: {
                'ok': True, 'result': {'id': 'A' * n_chars}},
            log_fn=lambda m: None)
        return result, seen_messages, cap

    def test_oversized_result_truncated_with_marker(self):
        cap = trace_detail()['tool_result_cap']
        result, seen, _ = self._run_with_result_chars(cap + 50_000)
        # Round-1 request carries the tool result the model will see.
        tool_content = seen[1][-1]['content'][0]['content']
        self.assertLess(len(tool_content), cap + 200)  # cap + marker only
        self.assertIn('[TRUNCATED by runner', tool_content)
        # Forensics ride the action record.
        act = (result['action_details'] + result['read_calls'])[0]
        self.assertTrue(act['result_truncated'])
        self.assertGreater(act['result_chars'], cap)
        self.assertLessEqual(len(act['result_head']),
                             trace_detail()['result_head_cap'])

    def test_normal_sized_result_untouched(self):
        result, seen, cap = self._run_with_result_chars(1_000)
        tool_content = seen[1][-1]['content'][0]['content']
        self.assertNotIn('[TRUNCATED by runner', tool_content)
        act = (result['action_details'] + result['read_calls'])[0]
        self.assertNotIn('result_truncated', act)
        self.assertNotIn('result_head', act)
        self.assertGreater(act['result_chars'], 0)


class TestRunLoopError(unittest.TestCase):

    def _run_failing_round1(self, exc):
        """Round 0 dispatches one write, then the round-1 API call raises."""
        r0 = FakeMessage(
            content=[FakeBlock(type='tool_use', id='t1', name='remember',
                               input={'title': 'A', 'content': 'c'})],
            stop_reason='tool_use', usage=FakeUsage())

        class ExplodingAPI:
            def __init__(self):
                self._responses = [r0]

            def stream(self, **kwargs):
                if not self._responses:
                    raise exc
                from tests.test_runner_per_round_stats import FakeStream
                return FakeStream(self._responses.pop(0))

        client = FakeClient([])
        client.messages = ExplodingAPI()
        run_llm_loop(
            client=client, model='claude-test', max_tokens=4096,
            max_rounds=3, system_prompt='S', user_content='U', tools=[],
            dispatch_fn=lambda name, args: {
                'ok': True, 'result': {'id': 'n1'},
                'affected': {'created': ['n1']}},
            log_fn=lambda m: None)

    def test_midrun_failure_wraps_with_partial_actions_and_cause(self):
        original = ValueError('prompt is too long')
        with self.assertRaises(RunLoopError) as ctx:
            self._run_failing_round1(original)
        err = ctx.exception
        self.assertIs(err.__cause__, original)
        # The original type name rides the message (error-log greppability).
        self.assertIn('ValueError', str(err))
        self.assertIn('prompt is too long', str(err))
        # The round-0 write survives on partial_actions.
        self.assertEqual(len(err.partial_actions), 1)
        self.assertEqual(err.partial_actions[0]['tool'], 'remember')
        self.assertEqual(err.partial_actions[0]['created'], ['n1'])


class TestRetryMatchesWrappedTransients(unittest.TestCase):
    """The S2 regression: retry_on_transient_api_error must see through
    RunLoopError to the transient __cause__ — a mid-run 5xx/timeout used to
    retry (pre-wrap) and must keep retrying."""

    def _transient(self):
        import anthropic
        import httpx
        req = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
        return anthropic.APIConnectionError(request=req)

    def test_wrapped_transient_is_retried(self):
        calls = {'n': 0}

        def fn():
            calls['n'] += 1
            if calls['n'] == 1:
                err = RunLoopError('APIConnectionError: boom',
                                   partial_actions=[{'tool': 'remember'}])
                err.__cause__ = self._transient()
                raise err
            return 'ok'

        out = retry_on_transient_api_error(fn, attempts=2, base_backoff_s=0)
        self.assertEqual(out, 'ok')
        self.assertEqual(calls['n'], 2)

    def test_bare_transient_still_retried(self):
        calls = {'n': 0}

        def fn():
            calls['n'] += 1
            if calls['n'] == 1:
                raise self._transient()
            return 'ok'

        out = retry_on_transient_api_error(fn, attempts=2, base_backoff_s=0)
        self.assertEqual(out, 'ok')
        self.assertEqual(calls['n'], 2)

    def test_wrapped_non_transient_raises_immediately(self):
        calls = {'n': 0}

        def fn():
            calls['n'] += 1
            err = RunLoopError('ValueError: client bug')
            err.__cause__ = ValueError('client bug')
            raise err

        with self.assertRaises(RunLoopError):
            retry_on_transient_api_error(fn, attempts=3, base_backoff_s=0)
        self.assertEqual(calls['n'], 1)

    def test_exhausted_retries_reraise_last(self):
        def fn():
            raise self._transient()

        import anthropic
        with self.assertRaises(anthropic.APIConnectionError):
            retry_on_transient_api_error(fn, attempts=2, base_backoff_s=0)


class TestBuildFailedRunMetadata(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop('BRAIN_TRACE_MODE', None)

    def tearDown(self):
        if self._saved is not None:
            os.environ['BRAIN_TRACE_MODE'] = self._saved

    def test_salvaged_forensics_are_bounded(self):
        huge = 'X' * 5_000_000
        actions = [{
            'tool': 'brain_batch',
            'input': {'operations': [{'op': 'remember', 'content': huge}]},
            'result_chars': 6_000_000,
            'result_truncated': True,
            'result_head': huge[:trace_detail()['result_head_cap']],
            'error': None,
        }]
        md = build_failed_run_metadata(interaction=_STAMP,
            error=RuntimeError('E' * 10_000), stop_counter=48,
            inputs_processed=1, partial_actions=actions)
        d = trace_detail()
        self.assertLessEqual(len(md['error']), 500)
        pa = md['partial_actions'][0]
        self.assertLessEqual(len(pa['input_head']),
                             d['failed_action_input_cap'])
        self.assertLessEqual(len(pa['result_head']), d['result_head_cap'])
        self.assertEqual(pa['ops'], 1)
        self.assertEqual(pa['result_chars'], 6_000_000)
        self.assertEqual(md['stop_counter'], 48)

    def test_untruncated_action_carries_no_result_head(self):
        md = build_failed_run_metadata(interaction=_STAMP,
            error='boom', stop_counter=1, inputs_processed=0,
            partial_actions=[{'tool': 'remember', 'input': {'title': 't'},
                              'result_chars': 120}])
        self.assertEqual(md['partial_actions'][0]['result_head'], '')

    def test_no_partial_actions_is_fine(self):
        md = build_failed_run_metadata(interaction=_STAMP, error='x', stop_counter=0,
                                       inputs_processed=0)
        self.assertEqual(md['partial_actions'], [])

    def test_payload_pointer_rides_when_present_absent_when_not(self):
        with_ptr = build_failed_run_metadata(interaction=_STAMP,
            error='x', stop_counter=0, inputs_processed=0,
            payload_pointer='payloads/2026-08-02/s1e-a-1/000-failed_run.json')
        self.assertIn('payload_pointer', with_ptr)
        without = build_failed_run_metadata(interaction=_STAMP, error='x', stop_counter=0,
                                            inputs_processed=0)
        self.assertNotIn('payload_pointer', without)


if __name__ == '__main__':
    unittest.main()
