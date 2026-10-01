"""brain_mcp.render_reply — the one render the MCP proxy and the encoder
runner share — and the reply notes (contract.REPLY_NOTE_KEYS) that ride
beside a reply's result or error.

A reply with no notes renders its body alone: the body tests below pin it
against the pieces the render is built from. Both paths show the truncation
banner, so an encoder never reads a cut result as complete.
"""
import copy
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import brain_mcp
from servers.brain_mcp import (_format_error, _format_result, render_reply,
                               render_reply_body, render_reply_notes)
from servers.contract import (REPLY_NOTE_KEYS, add_reply_note, reply_notes,
                              truncation_banner)
from servers.scales.runner import run_llm_loop
from tests.test_runner_per_round_stats import (FakeBlock, FakeMessage,
                                               FakeStream)

_WRITE = {'ok': True, 'result': {'total': 1, 'succeeded': 1, 'results': [
    {'op': 'revise', 'index': 0, 'ok': True, 'result': {'id': 'bd01343d'}}]},
    'affected': {'revised': ['bd01343d']}}
_CUT = {'truncated': {'limit': 2, 'note': 'hit limit=2 before exhausting '
                      'the requested window'}, 'rows': [{'id': 1}, {'id': 2}]}
_REFUSED = {'ok': False, 'error': 'content swap[0]: `old` not found',
            'current': {'content': 'As of 2023-05-25, 124 points.'}}


class TestBodyRendersAsBefore(unittest.TestCase):
    """No notes → the text each path produced before render_reply."""

    def test_ok_reply_is_the_formatted_result(self):
        self.assertEqual(render_reply('brain_batch', _WRITE),
                         _format_result('brain_batch', _WRITE['result']))

    def test_cut_result_is_headed_by_the_banner(self):
        reply = {'ok': True, 'result': _CUT}
        self.assertEqual(render_reply('query_logs', reply), '%s\n\n%s' % (
            truncation_banner(_CUT['truncated']),
            _format_result('query_logs', _CUT)))

    def test_refusal_is_the_error_with_its_pointers(self):
        self.assertEqual(render_reply('revise', _REFUSED),
                         _format_error(_REFUSED['error'], _REFUSED))

    def test_caller_named_error_replaces_the_replys(self):
        self.assertEqual(render_reply('revise', _REFUSED, error='timeout'),
                         _format_error('timeout', _REFUSED))


class TestReplyNotes(unittest.TestCase):

    def test_notes_follow_the_body_in_contract_order(self):
        reply = dict(_WRITE)
        add_reply_note(reply, 'actions', 'Repair what still holds 124.')
        add_reply_note(reply, 'information', 'Situation: 124 is her best.')
        add_reply_note(reply, 'information', 'Question: her best score?')
        self.assertEqual(REPLY_NOTE_KEYS, ('information', 'actions'))
        self.assertEqual(
            render_reply('brain_batch', reply),
            _format_result('brain_batch', _WRITE['result'])
            + '\n\nInformation:\nSituation: 124 is her best.'
              '\n\nQuestion: her best score?'
            + '\n\nActions:\nRepair what still holds 124.')

    def test_notes_never_enter_the_result_body(self):
        reply = dict(_WRITE, information=['beside, not inside'])
        body = render_reply('brain_batch', reply).split('\n\nInformation:')[0]
        self.assertEqual(body, _format_result('brain_batch', _WRITE['result']))

    def test_an_error_reply_carries_notes_after_its_pointers(self):
        reply = dict(_REFUSED, actions=['Re-send the swap from `current`.'])
        self.assertEqual(render_reply('revise', reply),
                         _format_error(_REFUSED['error'], _REFUSED)
                         + '\n\nActions:\nRe-send the swap from `current`.')

    def test_empty_entries_and_lists_render_nothing(self):
        reply = dict(_WRITE, information=['', None], actions=[])
        self.assertEqual(render_reply('brain_batch', reply),
                         _format_result('brain_batch', _WRITE['result']))

    def test_add_reply_note_refuses_an_unknown_key(self):
        with self.assertRaises(ValueError):
            add_reply_note({}, 'hints', 'x')

    def test_add_reply_note_takes_text_only(self):
        for bad in ('', None, {'id': 'bd01343d'}, 7):
            with self.assertRaises(ValueError):
                add_reply_note({}, 'information', bad)

    def test_notes_set_without_the_door_neither_vanish_nor_break(self):
        reply = dict(_WRITE, information='one string', actions=[{'id': 'x'}, 3])
        self.assertEqual(reply_notes(reply), {
            'information': ['one string'], 'actions': ['{"id": "x"}', '3']})
        self.assertTrue(render_reply('brain_batch', reply).endswith(
            '\n\nInformation:\none string\n\nActions:\n{"id": "x"}\n\n3'))

    def test_a_count_under_a_note_key_is_not_a_note(self):
        # `actions` is also the name unit results give their tool-call count.
        self.assertEqual(reply_notes({'ok': True, 'actions': 3,
                                      'information': {'a': 1}}), {})

    def test_render_reply_is_its_body_then_its_notes(self):
        reply = dict(_REFUSED, actions=['Re-send the swap from `current`.'])
        self.assertEqual(render_reply('revise', reply),
                         render_reply_body('revise', reply)
                         + render_reply_notes(reply_notes(reply)))

    def test_a_raw_payload_carries_no_notes(self):
        raw = {'healthy': True, 'actions': ['Auto-archived 3 context nodes']}
        self.assertEqual(reply_notes(raw), {})
        self.assertNotIn('Actions:', render_reply('health_check', raw,
                                                  error='missing envelope'))


class TestProxyRendersNotes(unittest.TestCase):

    def _call(self, reply):
        with mock.patch.object(brain_mcp, 'daemon_send', return_value=reply):
            return brain_mcp.handle_tools_call(
                'req-1', {'name': 'brain_batch', 'arguments': {}})['result']

    def test_ok_reply_notes_reach_anchor(self):
        reply = dict(_WRITE, information=['Situation: 124 is her best.'])
        out = self._call(reply)
        self.assertNotIn('isError', out)
        self.assertEqual(out['content'][0]['text'],
                         render_reply('brain_batch', reply))

    def test_a_missing_envelope_shows_no_payload_actions(self):
        out = self._call({'healthy': True, 'actions': ['Auto-archived 3 nodes']})
        self.assertTrue(out['isError'])
        self.assertIn('envelope', out['content'][0]['text'])
        self.assertNotIn('Actions:', out['content'][0]['text'])

    def test_error_reply_notes_reach_anchor(self):
        reply = dict(_REFUSED, actions=['Re-send the swap from `current`.'])
        out = self._call(reply)
        self.assertTrue(out['isError'])
        self.assertEqual(out['content'][0]['text'],
                         render_reply('brain_batch', reply))


class _Recording:
    """A client whose stream() records the conversation each round sees."""

    def __init__(self, scripted):
        self.seen, self._scripted = [], list(scripted)
        outer = self

        class _API:
            def stream(self, **kw):
                outer.seen.append(copy.deepcopy(kw['messages']))
                return FakeStream(outer._scripted.pop(0))

        self.messages = _API()


def _tool_call(name='brain_batch', inp=None):
    return FakeMessage([FakeBlock('tool_use', id='t0', name=name,
                                  input=inp or {'operations': []})],
                       stop_reason='tool_use')


def _loop(reply, annotate_fn=None, name='brain_batch', logs=None):
    client = _Recording([_tool_call(name), FakeMessage([FakeBlock('text', text='done')])])
    result = run_llm_loop(
        client=client, model='claude-test', max_tokens=256, max_rounds=5,
        system_prompt='S', user_content='U', tools=[],
        dispatch_fn=lambda n, a: copy.deepcopy(reply),
        log_fn=(logs.append if logs is not None else None),
        annotate_fn=annotate_fn)
    shown = client.seen[-1][-1]['content'][0]['content']
    return result, shown


class TestRunnerAnnotates(unittest.TestCase):

    def test_without_a_hook_the_encoder_reads_the_reply_as_rendered(self):
        _, shown = _loop(_WRITE)
        self.assertEqual(shown, render_reply('brain_batch', _WRITE))

    def test_hook_sees_each_call_and_its_notes_reach_the_encoder(self):
        calls = []

        def annotate(name, inp, reply):
            calls.append((name, inp, reply.get('affected')))
            if (reply.get('affected') or {}).get('revised'):
                add_reply_note(reply, 'information', 'Situation: 124 is her best.')
                add_reply_note(reply, 'actions', 'Repair what still holds 124.')

        _, shown = _loop(_WRITE, annotate_fn=annotate)
        self.assertEqual(calls, [('brain_batch', {'operations': []},
                                  {'revised': ['bd01343d']})])
        self.assertEqual(shown, render_reply('brain_batch', dict(
            _WRITE, information=['Situation: 124 is her best.'],
            actions=['Repair what still holds 124.'])))

    def test_the_notes_shown_ride_the_action_record(self):
        def annotate(name, inp, reply):
            add_reply_note(reply, 'actions', 'Repair what still holds 124.')

        result, _ = _loop(_WRITE, annotate_fn=annotate)
        self.assertEqual(result['action_details'][0]['notes'],
                         {'actions': ['Repair what still holds 124.']})
        plain, _ = _loop(_WRITE)
        self.assertNotIn('notes', plain['action_details'][0])

    def test_a_failing_hook_costs_its_notes_not_the_run(self):
        logs = []

        def annotate(name, inp, reply):
            raise RuntimeError('render broke')

        result, shown = _loop(_WRITE, annotate_fn=annotate, logs=logs)
        self.assertEqual(result['final_text'], 'done')
        self.assertEqual(shown, render_reply('brain_batch', _WRITE))
        self.assertTrue(any('annotation failed' in l and 'render broke' in l
                            for l in logs))
        self.assertEqual(result['annotation_errors'], [
            {'tool': 'brain_batch', 'error': "RuntimeError('render broke')"}])

    def test_the_size_cap_cuts_the_body_and_keeps_the_notes(self):
        reply = {'ok': True, 'result': {'rows': ['x' * 400]},
                 'actions': ['Repair what still holds 124.']}
        caps = {'tool_result_cap': 100, 'result_head_cap': 50}
        with mock.patch('servers.trace_contract.trace_detail', return_value=caps):
            result, shown = _loop(reply, name='query_logs')
        self.assertIn('[TRUNCATED by runner:', shown)
        self.assertTrue(shown.endswith('\n\nActions:\nRepair what still holds 124.'))
        self.assertTrue(result['read_calls'][0]['result_truncated'])

    def test_notes_longer_than_the_cap_are_cut_too(self):
        reply = {'ok': True, 'result': {'rows': ['x' * 400]},
                 'actions': ['y' * 300]}
        caps = {'tool_result_cap': 100, 'result_head_cap': 50}
        with mock.patch('servers.trace_contract.trace_detail', return_value=caps):
            _, shown = _loop(reply, name='query_logs')
        self.assertTrue(shown.endswith('[TRUNCATED by runner: the notes were %d chars.]'
                                       % len('\n\nActions:\n' + 'y' * 300)))
        self.assertLess(len(shown), 100 + 250)   # the cap plus the two markers

    def test_encoder_sees_the_truncation_banner(self):
        _, shown = _loop({'ok': True, 'result': _CUT}, name='query_logs')
        self.assertTrue(shown.startswith(truncation_banner(_CUT['truncated'])))


class TestHandlerNotesPassThroughDispatch(unittest.TestCase):
    """A note a command's handler sets reaches the reader: dispatch_command
    hands the handler's reply on with its top-level keys intact."""

    def test_handler_note_survives_dispatch_and_renders(self):
        from servers import daemon_dispatch
        from servers.dispatch_common import CmdEntry

        def handler(brain, args, graph_changes):
            reply = {'ok': True, 'result': {'done': True}}
            add_reply_note(reply, 'information', 'Situation: 124 is her best.')
            return reply

        with mock.patch.dict(daemon_dispatch.COMMAND_TABLE,
                             {'reply_note_probe': CmdEntry(handler, is_write=False)}):
            reply = daemon_dispatch.dispatch_command(None, 'reply_note_probe', {}, [])
        self.assertEqual(reply_notes(reply),
                         {'information': ['Situation: 124 is her best.']})
        self.assertTrue(render_reply('reply_note_probe', reply).endswith(
            '\n\nInformation:\nSituation: 124 is her best.'))


class TestAnnotationFailuresReachTheErrorsTable(unittest.TestCase):
    """run_llm_loop returns annotate_fn failures; each encoder family logs
    them where it logs its other loop losses."""

    _FAILED = [{'tool': 'brain_batch', 'error': "RuntimeError('render broke')"}]

    def test_an_s1_run_logs_them(self):
        from servers.scales.s1 import encode
        from tests.isolated_brain import brain_env
        loop_result = {'rounds': 1, 'actions': 0, 'write_actions': 0,
                       'action_details': [], 'read_calls': [], 'final_text': 'done',
                       'round_texts': ['done'], 'elapsed_ms': 1, 'input_tokens': 0,
                       'output_tokens': 0, 'cache_read_tokens': 0,
                       'cache_creation_tokens': 0, 'model': 'claude-test',
                       'truncations': [], 'undispatched_tool_calls': [],
                       'annotation_errors': self._FAILED}
        messages = [{'role': 'user', 'content': 'hi', 'trace_id': 'aaaaaaaa',
                     'judge_output': None},
                    {'role': 'assistant', 'content': 'hello', 'trace_id': 'bbbbbbbb'}]
        with brain_env(needs_embedder=False) as env:
            brain = env.brain
            with mock.patch.object(encode, '_gather_messages', return_value=messages), \
                    mock.patch('servers.scales.runner.make_client', return_value=object()), \
                    mock.patch.object(encode, 'run_llm_loop', return_value=loop_result), \
                    mock.patch.object(brain, '_log_error', wraps=brain._log_error) as spy:
                encode.run_encoding(brain, lambda cmd, args: {'ok': True}, 1,
                                    'sess-annotate-test')
        rows = [c.args for c in spy.call_args_list
                if c.args[0] == 's1e_reply_annotation_failed']
        self.assertEqual(len(rows), 1)
        self.assertIn('render broke', str(rows[0][1]))

    def test_an_s2_batch_logs_them(self):
        from types import SimpleNamespace
        from servers.scales.s2.base import IntegrationUnit
        logged = []
        unit = SimpleNamespace(
            NAME='consolidation', journal=None, chain_id=lambda: 'chain',
            brain=SimpleNamespace(_log_error=lambda *a: logged.append(a)),
            _accumulate_run=lambda total, result: None)
        self.assertTrue(IntegrationUnit._fold_batch_result(
            unit, {}, {'rounds': 1, 'annotation_errors': self._FAILED}, 1, 'src'))
        self.assertEqual([a[0] for a in logged], ['reply_annotation_failed'])


if __name__ == '__main__':
    unittest.main()
