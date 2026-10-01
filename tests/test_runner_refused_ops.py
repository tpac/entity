"""run_llm_loop copies the refused ops dispatch reports (top-level
`failed_ops`, beside `affected`) onto each action record, and
build_delta_metadata turns them into the run's refused_ops / errors. How
dispatch collects them is tested in test_brain_batch_transaction.py.

The shape replayed here is production run s1e-926d2c57-8 (id:dde56599): a
brain_batch whose two revises were refused, a retry that repaired one and was
refused again on the other, then a whole-content revise that repaired it. The
batch answered ok=True every time, so before failed_ops the run's delta said
'3 writes, errors []' and nothing distinguished it from a clean run.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.runner import run_llm_loop
from servers.trace_contract import build_delta_metadata

# A K stamp for builder calls — the builders require one.
_STAMP = {'fingerprint': 'f0f0f0f0f0f0', 'source': 'default', 'version': 0, 'id': None}


class _Usage:
    input_tokens, output_tokens = 10, 5
    cache_read_input_tokens = cache_creation_input_tokens = 0


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


class _Client:
    def __init__(self, scripted):
        streams = [_Stream(m) for m in scripted]
        self.messages = type('M', (), {'stream': lambda _s, **kw: streams.pop(0)})()


def _tool(i, name, inp):
    return _Msg([_Block('tool_use', id='t%d' % i, name=name, input=inp)],
                stop_reason='tool_use')


_ERR = 'content swap[0]: `old` not found'


def _refused(index, nid):
    return {'index': index, 'op': 'revise', 'node_id': nid, 'error': _ERR}


# Dispatch answers, in call order — dispatch_command's reply shape.
_ANSWERS = [
    {'ok': True, 'result': {'total': 2, 'succeeded': 0, 'failed': 2, 'results': [
        {'op': 'revise', 'index': 0, 'ok': False, 'error': _ERR},
        {'op': 'revise', 'index': 1, 'ok': False, 'error': _ERR}]},
     'affected': {}, 'failed_ops': [_refused(0, 'f2ee7c16'), _refused(1, 'fd1e97ea')]},
    {'ok': True, 'result': {'total': 2, 'succeeded': 1, 'failed': 1, 'results': [
        {'op': 'revise', 'index': 0, 'ok': True},
        {'op': 'revise', 'index': 1, 'ok': False, 'error': _ERR}]},
     'affected': {'revised': ['f2ee7c16']}, 'failed_ops': [_refused(1, 'fd1e97ea')]},
    {'ok': True, 'result': {'total': 1, 'revised': 1, 'results': [
        {'op': 'revise', 'index': 0, 'ok': True, 'node_id': 'fd1e97ea'}]},
     'affected': {'revised': ['fd1e97ea']}},
]


def _run(answers, script):
    answers = list(answers)
    return run_llm_loop(
        client=_Client(script), model='claude-test', max_tokens=256,
        max_rounds=5, system_prompt='S', user_content='U', tools=[],
        dispatch_fn=lambda name, args: answers.pop(0), log_fn=lambda m: None)


class TestRefusedOpsThroughTheLoop(unittest.TestCase):

    def test_production_shape_every_refusal_repaired(self):
        result = _run(_ANSWERS, [
            _tool(0, 'brain_batch', {'operations': []}),
            _tool(1, 'brain_batch', {'operations': []}),
            _tool(2, 'revise_batch', {'revisions': []}),
            _Msg([_Block('text', text='All writes succeeded.')])])
        ad = result['action_details']
        self.assertEqual([len(a.get('failed_ops', [])) for a in ad], [2, 1, 0])
        self.assertEqual(ad[0]['failed_ops'][1], _refused(1, 'fd1e97ea'))
        m = build_delta_metadata(interaction=_STAMP, action_details=ad)
        self.assertEqual([(r['node_id'], r['repaired']) for r in m['refused_ops']],
                         [('f2ee7c16', True), ('fd1e97ea', True), ('fd1e97ea', True)])
        self.assertEqual(m['errors'], [])   # every first attempt was fixed
        self.assertEqual(result['undispatched_tool_calls'], [])

    def test_repair_cut_off_by_max_rounds_is_reported(self):
        """The refusal comes back in the last round, so the repair call in the
        final reply never runs; the result names it for the caller to log."""
        answers = list(_ANSWERS[:1])
        result = run_llm_loop(
            client=_Client([_tool(0, 'brain_batch', {'operations': []}),
                            _tool(1, 'brain_batch', {'operations': []})]),
            model='claude-test', max_tokens=256, max_rounds=1,
            system_prompt='S', user_content='U', tools=[],
            dispatch_fn=lambda name, args: answers.pop(0), log_fn=lambda m: None)
        self.assertEqual(len(result['action_details']), 1)
        self.assertEqual(result['undispatched_tool_calls'], ['brain_batch'])

    def test_run_ending_before_the_retry_leaves_an_error(self):
        result = _run(_ANSWERS[:1], [
            _tool(0, 'brain_batch', {'operations': []}),
            _Msg([_Block('text', text='done')])])
        m = build_delta_metadata(interaction=_STAMP, action_details=result['action_details'])
        self.assertEqual([r['repaired'] for r in m['refused_ops']], [False, False])
        self.assertEqual(len(m['errors']), 2)
        self.assertTrue(m['errors'][0].startswith(
            'revise f2ee7c16 refused, not repaired:'))

    def test_reads_and_clean_writes_carry_no_failed_ops(self):
        result = _run([{'ok': True, 'result': [{'id': 'n1'}]},
                       {'ok': True, 'result': {'results': [
                           {'op': 'remember', 'index': 0, 'ok': True}]},
                        'affected': {'created': ['n2']}}], [
            _tool(0, 'get_nodes', {'node_ids': ['n1']}),
            _tool(1, 'brain_batch', {'operations': []}),
            _Msg([_Block('text', text='done')])])
        self.assertNotIn('failed_ops', result['read_calls'][0])
        self.assertNotIn('failed_ops', result['action_details'][0])

    def test_refused_call_shows_the_agent_its_pointers(self):
        """An ok=False reply's `current` reaches the model beside the error
        (brain_mcp._format_error), whole; the action record keeps the error."""
        body = 'Paragraph one. ' * 400
        seen = []

        class _Recording(_Client):
            def __init__(self, scripted):
                streams = [_Stream(m) for m in scripted]
                self.messages = type('M', (), {'stream': lambda _s, **kw: (
                    seen.append(kw['messages']), streams.pop(0))[1]})()

        answers = [{'ok': False, 'error': _ERR, 'current': {'content': body},
                    'failed_ops': [_refused(None, 'f2ee7c16')]}]
        result = run_llm_loop(
            client=_Recording([_tool(0, 'revise', {'node_id': 'f2ee7c16'}),
                               _Msg([_Block('text', text='done')])]),
            model='claude-test', max_tokens=256, max_rounds=5,
            system_prompt='S', user_content='U', tools=[],
            dispatch_fn=lambda name, args: answers.pop(0), log_fn=lambda m: None)
        shown = seen[-1][-1]['content'][0]['content']
        self.assertTrue(shown.startswith('ERROR: %s' % _ERR))
        self.assertIn(json.dumps({'current': {'content': body}}, indent=2), shown)
        self.assertEqual(result['action_details'][0]['error'], _ERR)


if __name__ == '__main__':
    unittest.main()
