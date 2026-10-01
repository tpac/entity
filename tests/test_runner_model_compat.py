"""Runner behavior on models that think by default (Sonnet 5, Opus 5.5).

Such models return thinking blocks alongside text and tool_use, and the
Opus 5.x / Fable 5.x safety classifiers can stop a reply with
stop_reason='refusal'. Contract locked here:
  - the loop echoes a thinking block back unchanged in the assistant turn
  - reply text (loop final_text, run_llm_once's raw text) skips thinking
    blocks instead of reading content[0]
  - a refusal raises ModelRefusal (wrapped in RunLoopError after round 0),
    never a clean zero-write result
  - a reply cut off by max_tokens before any write fails the run; after a
    write it stays a logged truncation
  - run_llm_once passes effort as output_config, and S2's _call_llm passes
    the unit config's effort through (None when unset)
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.runner import (ModelRefusal, RunLoopError, run_llm_loop,
                                   run_llm_once)
from servers.scales.s2 import base as s2_base
from servers.scales.s2.base import IntegrationUnit


THINKING = {'type': 'thinking', 'thinking': '', 'signature': 'sig-abc'}


class _Usage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Block:
    def __init__(self, type, text=None, id=None, name=None, input=None):
        self.type, self.text, self.id, self.name, self.input = type, text, id, name, input


class _Thinking:
    type = 'thinking'

    def to_dict(self):
        return dict(THINKING)


class _Msg:
    def __init__(self, content, stop_reason='end_turn', stop_details=None):
        self.content, self.stop_reason = content, stop_reason
        self.stop_details = stop_details
        self.usage = _Usage()


class _Details:
    category = 'cyber'


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
        self.calls = []

    def stream(self, **kw):
        self.calls.append(kw)
        return _Stream(self._s.pop(0))

    def create(self, **kw):
        self.calls.append(kw)
        return self._s.pop(0)


class _Client:
    def __init__(self, scripted):
        self.messages = _Messages(scripted)


def _loop(client):
    return run_llm_loop(
        client=client, model='claude-test', max_tokens=256, max_rounds=3,
        system_prompt='SYS', user_content='BODY', tools=[],
        dispatch_fn=lambda name, args: {'ok': True, 'result': {'id': 'n1'},
                                        'affected': {'created': ['n1']}},
        log_fn=lambda m: None)


def _thinking_then_tool():
    return _Msg([_Thinking(),
                 _Block('tool_use', id='t1', name='remember',
                        input={'title': 'A', 'content': 'c'})],
                stop_reason='tool_use')


class TestLoopOnThinkingModels(unittest.TestCase):

    def test_thinking_block_is_echoed_unchanged(self):
        client = _Client([_thinking_then_tool(),
                          _Msg([_Thinking(), _Block('text', text='done')])])
        _loop(client)
        assistant = client.messages.calls[1]['messages'][1]
        self.assertEqual(assistant['role'], 'assistant')
        self.assertEqual(assistant['content'][0], THINKING)
        self.assertEqual(assistant['content'][1]['type'], 'tool_use')

    def test_final_text_skips_thinking_blocks(self):
        client = _Client([_thinking_then_tool(),
                          _Msg([_Thinking(), _Block('text', text='done')])])
        self.assertEqual(_loop(client)['final_text'], 'done')

    def test_round_zero_refusal_raises(self):
        client = _Client([_Msg([], stop_reason='refusal',
                               stop_details=_Details())])
        with self.assertRaises(ModelRefusal) as cm:
            _loop(client)
        self.assertIn('cyber', str(cm.exception))

    def test_later_round_refusal_fails_the_run(self):
        client = _Client([_thinking_then_tool(),
                          _Msg([], stop_reason='refusal')])
        with self.assertRaises(RunLoopError) as cm:
            _loop(client)
        self.assertIsInstance(cm.exception.__cause__, ModelRefusal)
        self.assertEqual(len(cm.exception.partial_actions), 1)

    def test_budget_spent_before_any_write_fails_the_run(self):
        # The observed Sonnet 5 case: round 0 thinks through max_tokens and
        # never reaches its tool call.
        client = _Client([_Msg([_Thinking()], stop_reason='max_tokens')])
        with self.assertRaises(RunLoopError) as cm:
            _loop(client)
        self.assertIn('max_tokens', str(cm.exception))
        self.assertEqual(cm.exception.partial_actions, [])

    def test_truncated_wrap_up_after_a_write_keeps_the_result(self):
        client = _Client([_thinking_then_tool(),
                          _Msg([_Block('text', text='journal…')],
                               stop_reason='max_tokens')])
        result = _loop(client)
        self.assertEqual(result['final_text'], 'journal…')
        self.assertEqual(len(result['truncations']), 1)


class TestRunLlmOnce(unittest.TestCase):

    def test_text_after_thinking_block(self):
        client = _Client([_Msg([_Thinking(), _Block('text', text=' {"ok": 1} ')])])
        raw, _tel = run_llm_once(client, 'm', 10, 'SYS', 'BODY')
        self.assertEqual(raw, '{"ok": 1}')

    def test_effort_rides_as_output_config(self):
        client = _Client([_Msg([_Block('text', text='x')])])
        run_llm_once(client, 'm', 10, 'SYS', 'BODY', effort='high')
        self.assertEqual(client.messages.calls[0]['output_config'],
                         {'effort': 'high'})

    def test_no_effort_omits_output_config(self):
        client = _Client([_Msg([_Block('text', text='x')])])
        run_llm_once(client, 'm', 10, 'SYS', 'BODY')
        self.assertNotIn('output_config', client.messages.calls[0])

    def test_refusal_raises(self):
        client = _Client([_Msg([], stop_reason='refusal')])
        with self.assertRaises(ModelRefusal):
            run_llm_once(client, 'm', 10, 'SYS', 'BODY')


class _StubBrain:
    def __init__(self, config):
        self.config = config

    def get_interaction_prompt(self, name):
        return 'SYS'

    def get_interaction_config(self, name):
        return dict(self.config)

    def get_interaction_effective(self, name):
        return {'template': self.get_interaction_prompt(name),
                'config': self.get_interaction_config(name),
                'stamp': {'fingerprint': 'stub', 'source': 'default',
                          'version': 0, 'id': None}}

    def _log_error(self, *a):
        pass


class TestS2EffortPassthrough(unittest.TestCase):
    """_call_llm hands run_llm_once the unit config's effort; unset is None,
    which run_llm_once turns into no output_config at all."""

    def setUp(self):
        self._real_once = s2_base.run_llm_once
        self._real_make_client = s2_base.make_client
        self.kwargs = []

        def capturing_once(client, model, max_tokens, system, user, **kw):
            self.kwargs.append(kw)
            return '{"ok": true}', {'elapsed_ms': 1}

        s2_base.run_llm_once = capturing_once
        s2_base.make_client = lambda: object()
        # A key in the env keeps _call_llm from loading the operator's real
        # one from disk into this process.
        self._env = mock.patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test'})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        s2_base.run_llm_once = self._real_once
        s2_base.make_client = self._real_make_client

    def _call(self, config):
        u = IntegrationUnit.__new__(IntegrationUnit)
        u.brain, u.dispatch, u.NAME = _StubBrain(config), None, 'healer'
        u._call_llm('s2_healer', 'payload')

    def test_configured_effort_reaches_the_call(self):
        self._call({'model': 'm', 'max_tokens': 10, 'effort': 'high'})
        self.assertEqual(self.kwargs[0], {'effort': 'high'})

    def test_unset_effort_passes_none(self):
        self._call({'model': 'm', 'max_tokens': 10})
        self.assertEqual(self.kwargs[0], {'effort': None})


if __name__ == '__main__':
    unittest.main()
