"""Tests for run_llm_loop's per-round timing + token instrumentation.

The encoder's wall-clock budget (~3 min per cycle) is split across 2-5 LLM
rounds. Pre-change the summary log only had totals; we had no way to know
whether r1's 100s was generation, prefill, or server wait. These tests
lock in the per-round breakdown contract added 2026-05-19.

These tests intentionally do NOT touch any real Brain or hit the live
Anthropic API. All Brain interaction is unnecessary — run_llm_loop only
needs a client object that quacks like anthropic.Anthropic.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.runner import run_llm_loop


# ─── Fake Anthropic SDK surface ──────────────────────────────────────


class FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50,
                 cache_read=0, cache_creation=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation


class FakeBlock:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input


class FakeMessage:
    def __init__(self, content, stop_reason='end_turn', usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or FakeUsage()


class FakeStream:
    """Context manager mimicking client.messages.stream(...).

    Yields N synthetic events (the first one triggers TTFT measurement);
    get_final_message() returns the canned final response."""
    def __init__(self, final_msg, num_events=2):
        self.final_msg = final_msg
        self.num_events = num_events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __iter__(self):
        for i in range(self.num_events):
            yield {'type': 'event_%d' % i}

    def get_final_message(self):
        return self.final_msg


class FakeMessagesAPI:
    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)

    def stream(self, **kwargs):
        if not self._responses:
            raise RuntimeError(
                'FakeMessagesAPI: ran out of scripted responses')
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, scripted_responses):
        self.messages = FakeMessagesAPI(scripted_responses)


# ─── Tests ───────────────────────────────────────────────────────────


class TestPerRoundStats(unittest.TestCase):

    def _run(self, scripted, **kwargs):
        logs = []
        client = FakeClient(scripted)
        result = run_llm_loop(
            client=client, model='claude-test', max_tokens=4096,
            max_rounds=kwargs.get('max_rounds', 3),
            system_prompt='SYSTEM',
            user_content='USER',
            tools=[],
            dispatch_fn=lambda name, args: {'ok': True, 'result': {}},
            log_fn=logs.append,
        )
        return result, logs

    def test_single_round_logs_per_round_breakdown(self):
        """One round, no tool_use → log includes r0 ttft + out + rate."""
        final = FakeMessage(
            content=[FakeBlock(type='text', text='done')],
            stop_reason='end_turn',
            usage=FakeUsage(input_tokens=120, output_tokens=42,
                            cache_read=200, cache_creation=300))
        result, logs = self._run([FakeStream(final)])

        per_round_log = [l for l in logs if l.startswith('  r')]
        self.assertEqual(len(per_round_log), 1,
                         "Expected one per-round line, got: %s" % logs)
        line = per_round_log[0]
        self.assertIn('r0 ttft=', line)
        self.assertIn('out=42tok', line)
        self.assertIn('in=120', line)
        self.assertIn('cr=200', line)
        self.assertIn('cw=300', line)

    def test_two_rounds_logs_two_breakdowns(self):
        """First response has tool_use → loop calls API twice. Per-round
        log lines must appear for r0 AND r1, each with their own usage."""
        r0_response = FakeMessage(
            content=[FakeBlock(type='tool_use', id='tu_1', name='remember',
                               input={'title': 'A', 'content': 'first'})],
            stop_reason='tool_use',
            usage=FakeUsage(input_tokens=500, output_tokens=300))
        r1_response = FakeMessage(
            content=[FakeBlock(type='text', text='wrap-up')],
            stop_reason='end_turn',
            usage=FakeUsage(input_tokens=900, output_tokens=120))
        result, logs = self._run([
            FakeStream(r0_response), FakeStream(r1_response)])

        per_round_log = [l for l in logs if l.startswith('  r')]
        self.assertEqual(len(per_round_log), 2,
                         "Expected r0+r1 lines, got: %s" % per_round_log)
        self.assertIn('out=300tok', per_round_log[0])
        self.assertIn('out=120tok', per_round_log[1])
        # Totals line should still match the sum of rounds.
        totals = [l for l in logs if l.startswith('Rounds:')]
        self.assertEqual(len(totals), 1)
        self.assertIn('420 out', totals[0],
                      "Total output should be 300+120=420")

    def test_ttft_is_captured_when_stream_yields_events(self):
        """TTFT is the time from request issue to first server event. The
        FakeStream yields events synchronously, so ttft should be tiny
        but non-None (>=0)."""
        final = FakeMessage(
            content=[FakeBlock(type='text', text='done')],
            usage=FakeUsage(output_tokens=10))
        result, logs = self._run([FakeStream(final, num_events=1)])

        per_round_log = [l for l in logs if l.startswith('  r')]
        self.assertEqual(len(per_round_log), 1)
        # Must show a numeric ttft, not '?'
        self.assertRegex(per_round_log[0], r'ttft=\d+ms')

    def test_empty_stream_falls_back_gracefully(self):
        """If the stream yields zero events, ttft stays None and the log
        line shows '?'. This is a defensive path — should never happen
        in production but must not crash if it does."""
        final = FakeMessage(
            content=[FakeBlock(type='text', text='ok')],
            usage=FakeUsage(output_tokens=5))
        # num_events=0 → no events, ttft never set
        result, logs = self._run([FakeStream(final, num_events=0)])

        per_round_log = [l for l in logs if l.startswith('  r')]
        self.assertEqual(len(per_round_log), 1)
        self.assertIn('ttft=?', per_round_log[0])
        # Loop still completes, no exception escaped.
        self.assertEqual(result['final_text'], 'ok')


class TestPerToolDetail(unittest.TestCase):
    """run_llm_loop records per-tool latency_ms / result_count / error on every
    action record (input = the args), so action_details (writes) and read_calls
    (reads) — already threaded into build_delta_metadata by the run_llm_loop
    encoders — gain the same per-call observability Surface's tool_trace has.
    Option A: extend the existing action record, no new delta field."""

    def _dispatch(self, name, args):
        # Production-faithful shapes (see servers/dispatch_read.py):
        #   recall_batch → {'result': [{'query', 'results':[...]}, ...]} (groups)
        #   get_nodes    → {'result': [node, node, ...]}                 (flat)
        #   remember     → {'result': {'id': ...}}                       (one id)
        #   boom         → {'ok': False, 'error': ...}
        # WRITE_TOOLS routes 'remember' → action_details, reads → read_calls.
        if name == 'recall_batch':
            return {'ok': True, 'result': [
                {'query': 'q1', 'results': [{'id': 'a'}, {'id': 'b'}]},
                {'query': 'q2', 'results': [{'id': 'c'}]},
            ]}
        if name == 'get_nodes':
            return {'ok': True, 'result': [{'id': 'x'}, {'id': 'y'}]}
        if name == 'remember':
            return {'ok': True, 'result': {'id': 'n1'},
                    'affected': {'created': ['n1']}}
        if name == 'boom':
            return {'ok': False, 'error': 'kaboom'}
        return {'ok': True, 'result': {}}

    def _run(self, scripted):
        return run_llm_loop(
            client=FakeClient(scripted), model='claude-test', max_tokens=4096,
            max_rounds=3, system_prompt='S', user_content='U', tools=[],
            dispatch_fn=self._dispatch, log_fn=lambda m: None)

    def test_read_and_write_carry_per_tool_detail(self):
        r0 = FakeMessage(
            content=[
                FakeBlock(type='tool_use', id='t1', name='recall_batch',
                          input={'queries': ['x']}),
                FakeBlock(type='tool_use', id='t2', name='get_nodes',
                          input={'node_ids': ['x', 'y']}),
                FakeBlock(type='tool_use', id='t3', name='remember',
                          input={'title': 'A', 'content': 'c'}),
            ],
            stop_reason='tool_use', usage=FakeUsage())
        r1 = FakeMessage(content=[FakeBlock(type='text', text='done')],
                         stop_reason='end_turn', usage=FakeUsage())
        result = self._run([FakeStream(r0), FakeStream(r1)])

        reads = {r['tool']: r for r in result['read_calls']}
        self.assertEqual(set(reads), {'recall_batch', 'get_nodes'})

        # recall_batch → per-query groups: result_count = SUM of nested hits
        # (2 + 1 = 3), NOT the number of queries (would be 2).
        rb = reads['recall_batch']
        self.assertEqual(rb['result_count'], 3)
        self.assertIsNone(rb['error'])
        self.assertIsInstance(rb['latency_ms'], int)
        self.assertGreaterEqual(rb['latency_ms'], 0)
        self.assertEqual(rb['input'], {'queries': ['x']})  # args captured

        # get_nodes → flat result list: result_count = its length (2)
        self.assertEqual(reads['get_nodes']['result_count'], 2)

        # write tool → action_details, result_count = 1
        writes = result['action_details']
        self.assertEqual(len(writes), 1)
        wr = writes[0]
        self.assertEqual(wr['tool'], 'remember')
        self.assertEqual(wr['result_count'], 1)
        self.assertIsNone(wr['error'])
        self.assertIsInstance(wr['latency_ms'], int)

    def test_failed_tool_records_error_and_zero_count(self):
        r0 = FakeMessage(
            content=[FakeBlock(type='tool_use', id='t1', name='boom',
                               input={'q': 1})],
            stop_reason='tool_use', usage=FakeUsage())
        r1 = FakeMessage(content=[FakeBlock(type='text', text='done')],
                         stop_reason='end_turn', usage=FakeUsage())
        result = self._run([FakeStream(r0), FakeStream(r1)])

        reads = result['read_calls']  # 'boom' isn't a WRITE_TOOL
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0]['error'], 'kaboom')
        self.assertEqual(reads[0]['result_count'], 0)
        self.assertIsInstance(reads[0]['latency_ms'], int)


if __name__ == '__main__':
    unittest.main()
