"""surface_turns_bench selects through production's surface loop.

The bench once built its own Haiku call and drifted from production: no
fetch tools, while the shipping prompt describes them. These pin that a rep
goes through _call_surface_agentic (tools offered, a fetch answered with
production's empty result, final round forced to select) and that a reply
with no usable selection is never scored.

Scripted client only — no network.
"""
import json
from types import SimpleNamespace

from tests.eval_optional import require_eval
require_eval()  # D-8: eval/ is absent from the public tree

from eval import surface_turns_bench as bench  # noqa: E402
from servers.scales.s1.fetch_tools import TOOL_DEFINITIONS  # noqa: E402


def _usage():
    return SimpleNamespace(input_tokens=3000, output_tokens=50,
                           cache_read_input_tokens=0,
                           cache_creation_input_tokens=0)


def _text(text, stop_reason='end_turn'):
    return SimpleNamespace(stop_reason=stop_reason, usage=_usage(),
                           content=[SimpleNamespace(type='text', text=text)])


def _tool_use(name='recall_topical', tool_input=None):
    return SimpleNamespace(
        stop_reason='tool_use', usage=_usage(),
        content=[SimpleNamespace(type='tool_use', id='tu_1', name=name,
                                 input=tool_input or {'query': 'timeout fix'})])


class _Client:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _scenario(name='shown_repick_trap'):
    with open(bench.FIXTURES) as f:
        return next(s for s in json.load(f)['scenarios'] if s['name'] == name)


PICK = json.dumps({'selected': [{'id': 'f1f2f3f4', 'mode': 'arc'}]})


def test_fetch_runs_the_production_loop_and_scores_the_final_pick():
    client = _Client(_tool_use(), _text(PICK))
    result = bench.run_one(client, _scenario(), 'SYSTEM')

    first, final = client.calls
    assert first['tools'] == TOOL_DEFINITIONS
    assert 'tool_choice' not in first
    assert final['tools'] == TOOL_DEFINITIONS
    assert final['tool_choice'] == {'type': 'none'}
    tool_result = final['messages'][-1]['content'][0]
    assert tool_result['type'] == 'tool_result'
    assert tool_result['content'] == 'No results.'

    assert result['pass'] is True
    assert result['picked'] == ['f1f2f3f4']
    assert result['tool_calls'] == [
        {'tool': 'recall_topical', 'args': {'query': 'timeout fix'}}]


def test_direct_selection_is_one_call_with_tools_offered():
    client = _Client(_text(PICK))
    result = bench.run_one(client, _scenario(), 'SYSTEM')

    assert len(client.calls) == 1
    assert client.calls[0]['tools'] == TOOL_DEFINITIONS
    assert result['pass'] is True
    assert result['tool_calls'] == []


def test_cut_off_reply_is_unusable_not_scored():
    client = _Client(_text(PICK, stop_reason='max_tokens'))
    result = bench.run_one(client, _scenario(), 'SYSTEM')

    assert result['pass'] is False
    assert 'cut off' in result['error']
    assert result['picked'] == []


def test_api_failure_reports_what_the_loop_logged():
    client = _Client(RuntimeError('overloaded'))
    result = bench.run_one(client, _scenario(), 'SYSTEM')

    assert result['pass'] is False
    assert 'surface_agentic_api' in result['error']
    assert 'overloaded' in result['error']
