"""Per-round capture in run_llm_loop (record_round_fn — TRACE-MODES step 2).

run_llm_loop hands the LITERAL per-round request pieces to the caller's
record_round_fn (brain.round_recorder closure) before each API call, so the
actual prompt is recoverable without an unfaithful rebuild. The runner has no
brain — the closure owns gating, shape (build_round_payload), and file writes.

Contract locked here:
  - full system TEXT reaches the callback (not a length, the legacy-dump bug)
  - one callback per round with the correct round index; the tool-result
    continuation (r1) carries the round-0 assistant + tool_result blocks
  - no callback (None) → no capture, zero behavior change
  - a raising callback never kills the loop (capture must not break a cycle)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.runner import run_llm_loop


class _Usage:
    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 5
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _Block:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type, self.text, self.id, self.name, self.input = type, text, id, name, input


class _Msg:
    def __init__(self, content, stop_reason='end_turn'):
        self.content, self.stop_reason, self.usage = content, stop_reason, _Usage()


class _Stream:
    def __init__(self, final):
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        yield {'type': 'e0'}

    def get_final_message(self):
        return self.final


class _Messages:
    def __init__(self, scripted):
        self._s = list(scripted)
        self.calls = []          # kwargs of every stream() request, in order

    def stream(self, **kw):
        self.calls.append(kw)
        return self._s.pop(0)


class _Client:
    def __init__(self, scripted):
        self.messages = _Messages(scripted)


def _two_round_script():
    # r0: one tool_use → forces a second round; r1: end_turn.
    r0 = _Msg([_Block('tool_use', id='t1', name='remember',
                      input={'title': 'A', 'content': 'c'})], stop_reason='tool_use')
    r1 = _Msg([_Block('text', text='done')], stop_reason='end_turn')
    return [_Stream(r0), _Stream(r1)]


def _run(client, record_round_fn):
    return run_llm_loop(
        client=client, model='claude-test', max_tokens=256, max_rounds=3,
        system_prompt='FULL SYSTEM PROMPT «with unicode»', user_content='BODY',
        user_preamble='PREAMBLE', tools=[],
        dispatch_fn=lambda name, args: {'ok': True, 'result': {'id': 'n1'},
                                        'affected': {'created': ['n1']}},
        log_fn=lambda m: None, record_round_fn=record_round_fn)


class TestRecordRoundFn(unittest.TestCase):

    def test_callback_fires_per_round_with_full_pieces(self):
        seen = []
        _run(_Client(_two_round_script()),
             lambda idx, parts: seen.append((idx, parts)))
        self.assertEqual([idx for idx, _ in seen], [0, 1])

        by_round = {idx: parts for idx, parts in seen}
        # Full system TEXT (not a length) — the legacy-dump bug this pins.
        for parts in by_round.values():
            self.assertEqual(parts['system'], 'FULL SYSTEM PROMPT «with unicode»')
            self.assertEqual(parts['model'], 'claude-test')
            self.assertEqual(parts['tools'], [])

        # r0 = the initial user turn (preamble + body blocks).
        r0_texts = [b['text'] for m in by_round[0]['messages']
                    for b in m['content'] if b.get('type') == 'text']
        self.assertIn('PREAMBLE', r0_texts)
        self.assertIn('BODY', r0_texts)

        # r1 carries the round-0 assistant tool_use + the tool_result
        # continuation — i.e. the ACTUAL later-round prompt, not a rebuild.
        roles = [m['role'] for m in by_round[1]['messages']]
        self.assertEqual(roles, ['user', 'assistant', 'user'])

    def test_none_callback_no_capture_loop_unchanged(self):
        result = _run(_Client(_two_round_script()), None)
        self.assertEqual(result['rounds'], 2)
        self.assertEqual(result['final_text'], 'done')

    def test_raising_callback_never_kills_the_loop(self):
        def boom(idx, parts):
            raise RuntimeError('capture exploded')
        result = _run(_Client(_two_round_script()), boom)
        self.assertEqual(result['final_text'], 'done')


class TestEffortPassthrough(unittest.TestCase):
    """`effort` reaches every API request as output_config; None omits it.
    The value originates in the encoder interaction's parameters JSON (the
    K-store) — this pins the runner half of that wire."""

    def _loop(self, client, **kw):
        return run_llm_loop(
            client=client, model='claude-test', max_tokens=256, max_rounds=3,
            system_prompt='SYS', user_content='BODY', tools=[],
            dispatch_fn=lambda name, args: {'ok': True, 'result': {'id': 'n1'},
                                            'affected': {'created': ['n1']}},
            log_fn=lambda m: None, **kw)

    def test_effort_set_rides_every_round(self):
        client = _Client(_two_round_script())
        self._loop(client, effort='medium')
        calls = client.messages.calls
        self.assertEqual(len(calls), 2)
        for kw in calls:
            self.assertEqual(kw.get('output_config'), {'effort': 'medium'})

    def test_effort_none_omits_output_config(self):
        client = _Client(_two_round_script())
        self._loop(client)                      # default: no effort kwarg
        for kw in client.messages.calls:
            self.assertNotIn('output_config', kw)


if __name__ == '__main__':
    unittest.main()
