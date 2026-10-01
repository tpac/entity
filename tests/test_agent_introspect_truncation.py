"""Eval scripts must not report an unusable model reply as a result.

A reply cut off before finishing, or one that is not the JSON the probe asked
for, parsed leniently reads as "Findings: 0" — indistinguishable from a clean
prompt — a cut-off encoder replay reads as a sparser encoding, and a cut-off
judge verdict reads as a measured miss. These tests hold the refusals (the
agent_introspect helpers and the eval scripts that share them), the
exclusions, and the record-and-continue paths that keep one bad reply from
discarding a run. No API calls: the client is faked.
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

import anthropic  # noqa: E402

from eval.agent_introspect._common import (  # noqa: E402
    ProbeOutputError, call_sonnet, parse_json_reply,
)


def _fake_client(monkeypatch, text, stop_reason, tokens_out):
    resp = SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=tokens_out),
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    monkeypatch.setattr(anthropic, 'Anthropic', lambda: client)


def _fake_sequence(monkeypatch, *replies):
    """Each call returns the next (text, stop_reason) reply."""
    queue = list(replies)
    def create(**kw):
        text, stop = queue.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(text=text)],
                               stop_reason=stop,
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    monkeypatch.setattr(anthropic, 'Anthropic',
                        lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_truncated_reply_raises(monkeypatch):
    _fake_client(monkeypatch, '[{"severity": "high", "kind": "contr',
                 'max_tokens', 6000)
    with pytest.raises(ProbeOutputError, match='max_tokens=6000'):
        call_sonnet('sys', 'user', max_tokens=6000)


def test_context_window_cut_off_raises_and_keeps_the_partial_text(monkeypatch):
    _fake_client(monkeypatch, 'The first half of an answer',
                 'model_context_window_exceeded', 900)
    with pytest.raises(ProbeOutputError, match='model_context_window_exceeded') as e:
        call_sonnet('sys', 'user')
    assert e.value.text == 'The first half of an answer'


def test_parse_failure_keeps_the_reply_text():
    with pytest.raises(ProbeOutputError) as e:
        parse_json_reply('no json here', list)
    assert e.value.text == 'no json here'


def test_complete_reply_carries_stop_reason(monkeypatch):
    _fake_client(monkeypatch, '[]', 'end_turn', 2)
    out = call_sonnet('sys', 'user')
    assert out['stop_reason'] == 'end_turn'
    assert out['text'] == '[]'


def test_empty_array_is_a_result():
    assert parse_json_reply('[]', list) == []
    assert parse_json_reply('```json\n[]\n```', list) == []


def test_fenced_and_prose_wrapped_json_parses():
    assert parse_json_reply('```json\n[{"a": 1}]\n```', list) == [{'a': 1}]
    assert parse_json_reply('Here:\n[{"a": 1}]\nDone [x].', list) == [{'a': 1}]
    assert parse_json_reply('```json\n{"prediction": "yes"}\n```', dict) == {
        'prediction': 'yes'}


@pytest.mark.parametrize('text, expect', [
    ('[{"severity": "high", "kind": "contr', list),   # truncated array
    ('I found no issues worth reporting.', list),     # prose, no JSON
    ('{"prediction": "yes", "confid', dict),          # truncated object
    ('["yes"]', dict),                                # wrong JSON type
])
def test_unparseable_reply_raises(text, expect):
    with pytest.raises(ProbeOutputError, match='not a JSON'):
        parse_json_reply(text, expect)


@pytest.mark.parametrize('text, expect', [
    ('{"findings": [], "note": "could not audit"}', list),  # declined, not empty
    ('[{"prediction": "yes"}, {"prediction": "no"}]', dict),  # not one verdict
])
def test_reply_of_the_wrong_json_type_raises(text, expect):
    with pytest.raises(ProbeOutputError, match='got a JSON'):
        parse_json_reply(text, expect)


def test_payload_fence_wins_over_brackets_and_numbers_in_prose():
    fenced = '```json\n[{"a": 1}]\n```'
    assert parse_json_reply('Findings (see [rule 3]):\n' + fenced, list) == [{'a': 1}]
    assert parse_json_reply('1. Findings below.\n' + fenced, list) == [{'a': 1}]
    # the payload is the largest value, not the first that decodes
    assert parse_json_reply('{"note": "1 finding"}\n' + fenced, list) == [{'a': 1}]


def test_fence_split_inside_a_quoted_string_does_not_win():
    # Probes quote prompts verbatim, and prompts are full of ``` fences.
    reply = ('```json\n[{"q": "use ```a``` [] ```b``` here", '
             '"severity": "high"}]\n```')
    assert parse_json_reply(reply, list) == [
        {'q': 'use ```a``` [] ```b``` here', 'severity': 'high'}]


# ── Replay consumers: a run cut off at max_tokens is not a sparser run ──

from eval.agent_introspect import content_diff, quality_probe  # noqa: E402


def _row(prompt, nodes, **extra):
    return {"prompt": prompt, "qid": "q1", "trial": 0,
            "nodes_count": nodes, **extra}


def test_quality_aggregate_measures_complete_runs_only():
    agg = quality_probe._aggregate([
        _row("a", 4), _row("a", 6),
        _row("a", 1, truncated=True),
        {"prompt": "a", "qid": "q1", "trial": 3, "error": "APIError: boom"},
    ])
    assert agg["mean_nodes_count"] == 5
    assert (agg["n"], agg["n_truncated"], agg["n_errored"]) == (2, 1, 1)


def test_quality_report_shows_unmeasured_prompt_as_dash_not_zero():
    rows = [_row("a", 4), _row("b", 1, truncated=True)]
    md = quality_probe._format_report({"a": "", "b": ""}, ["q1"], 1, rows)
    assert "| nodes/run | 4.0 | — |" in md
    assert "| truncated (excluded) | 0 | 1 |" in md
    assert "1 of 2 runs are excluded" in md


def test_content_diff_marks_truncated_trials():
    conv = {"question": "q?", "gold": "g", "turns": []}
    md = content_diff._format_report(
        "q1", {"a": ""}, conv,
        {"a": [{"actions": []}, {"actions": [], "truncated": True}]})
    assert "1 of 2 trials were cut off" in md
    assert "| `a` | 1 ⚠ truncated |" in md
    assert "| `a` | 0 |" in md


# ── Consolidation grader: an unscored dimension must not pass the gate ──

from eval.agent_introspect import consolidation_contract_eval as cce  # noqa: E402


def _grade(dims):
    import json
    return json.dumps({"per_dim": [{"dim": d, "status": "satisfied",
                                    "evidence": "x"} for d in dims],
                       "summary": "ok"})


def test_grader_reply_covering_every_dim_is_scored(monkeypatch):
    _fake_client(monkeypatch, _grade(cce.DIMENSIONS), 'end_turn', 50)
    assert len(cce.score_decision('contract', 'cluster', [])['per_dim']) == len(
        cce.DIMENSIONS)


@pytest.mark.parametrize('reply', [
    '{"summary": "ok"}',                   # no per_dim at all
    _grade(list(cce.DIMENSIONS)[1:]),      # one dim missing
    _grade(['c1_action_fit'] + list(cce.DIMENSIONS)[1:]),  # mis-cased dim
    _grade(list(cce.DIMENSIONS) + [list(cce.DIMENSIONS)[3]]),  # scored twice
])
def test_grader_reply_with_an_unscored_dim_raises(monkeypatch, reply):
    _fake_client(monkeypatch, reply, 'end_turn', 50)
    with pytest.raises(ProbeOutputError, match='unscored'):
        cce.score_decision('contract', 'cluster', [])


def test_content_diff_marks_errored_trials():
    conv = {"question": "q?", "gold": "g", "turns": []}
    md = content_diff._format_report(
        "q1", {"a": ""}, conv, {"a": [{"actions": [], "error": "APIError: 529"}]})
    assert "| `a` | 0 ⚠ errored |" in md
    assert "a trial errored and has no nodes: APIError: 529" in md


# ── replay_one owns the truncation flag ──

from eval.agent_introspect import encoder_replay  # noqa: E402


@pytest.mark.parametrize('stop, truncated', [
    ('max_tokens', True), ('model_context_window_exceeded', True),
    ('tool_use', False), ('end_turn', False),
])
def test_replay_one_flags_every_cut_off_stop_reason(monkeypatch, stop, truncated):
    import servers.scales.s1.encode as encode
    monkeypatch.setattr(encode, '_get_tool_schemas', lambda: [])
    monkeypatch.setattr(encoder_replay, '_load_conversation', lambda qid: {
        'question': 'q', 'gold': 'g', 'conversation_now': '2026-01-01', 'turns': []})
    _fake_client(monkeypatch, '', stop, 10)
    assert encoder_replay.replay_one(None, 'q1', 'prompt')['truncated'] is truncated


def test_replay_report_keeps_errored_items_and_rates_complete_ones_only():
    full = {'qid': 'a', 'conversation_now': 'd', 'question': 'q', 'gold': 'g',
            'call_ms': 1, 'tokens_in': 1, 'tokens_out': 1, 'stop_reason': 'end_turn',
            'final_text': '', 'actions': [{'tool': 'remember_batch', 'input': {
                'nodes': [{'type': 'fact', 'title': 't', 'event_time': '2026'}]}}]}
    cut = dict(full, qid='b', truncated=True, actions=[{'tool': 'remember_batch',
               'input': {'nodes': [{'type': 'fact', 'title': 't'}]}}])
    md = encoder_replay.render_report(
        [full, cut, {'qid': 'c', 'error': 'boom'}], 'p.md')
    assert '1/1 (100%) across 1 complete items — excluded: 1 truncated, 1 errored' in md
    assert '## `c` — ERROR' in md


# ── One unusable reply is recorded; the rest of the run survives ──

from eval.agent_introspect import compliance_probe, counterfactual_probe  # noqa: E402

_ITEM = {'qid': 'q', 'conversation': [], 'encoder_prompt': 'p', 'meta': {},
         'action_details': [{'tool': 'remember_batch', 'input': {}}]}


def test_compliance_records_an_unusable_reply_and_continues(monkeypatch):
    monkeypatch.setattr(compliance_probe, 'load_item_artifact', lambda d, q: _ITEM)
    _fake_sequence(monkeypatch, ('no json', 'end_turn'),
                   ('[{"rule_id": "r1", "status": "comply"}]', 'end_turn'))
    results = compliance_probe.run_probe('run', ['q1', 'q2'],
                                         [{'id': 'r1', 'text': 'rule'}])
    assert 'not a JSON list' in results['q1']['error']
    assert results['q2']['verdicts'][0]['status'] == 'comply'


def test_counterfactual_off_vocabulary_prediction_is_an_error_not_a_gap(monkeypatch):
    monkeypatch.setattr(counterfactual_probe, 'load_item_artifact', lambda d, q: _ITEM)
    _fake_sequence(monkeypatch, ('{"prediction": "Yes"}', 'end_turn'),
                   ('{"prediction": "yes", "confidence": "high"}', 'end_turn'))
    changes = [{'name': 'c1'}, {'name': 'c2'}]
    results = counterfactual_probe.run_probe('run', ['q1'], changes)
    md = counterfactual_probe.render_markdown(changes, results)
    c1, c2 = md.split('## Change: `c2`')
    assert '| error | 1 |' in c1 and '| yes | 0 |' in c1
    assert '| yes | 1 |' in c2


def test_gate_cannot_pass_with_unscored_decisions():
    counts = {d: {} for d in cce.DIMENSIONS}
    assert cce.gate({}, {}, counts, 0)[0] is True
    passed, reasons = cce.gate({}, {}, counts, 2)
    assert passed is False and reasons[0].startswith('UNSCORED 2')


def test_surface_review_report_renders_errored_items():
    from eval.agent_introspect import surface_review
    ok = {'qid': 'a', 'verdict': 'PASS', 'question': 'q', 'gold': 'g',
          'hypothesis': 'h', 'sonnet_review': 'r', 'haiku_introspect': 'i'}
    md = surface_review.render_md([ok, {'qid': 'b', 'error': 'cut off'}], 'p.md')
    assert '2 items, 1 passing, 0 failing, 1 errored' in md
    assert '## `b` — **ERROR**' in md


# ── Eval scripts outside agent_introspect refuse a cut-off the same way ──

from eval.agent_introspect._common import raise_if_cut_off  # noqa: E402


def _resp(stop, text='partial'):
    return SimpleNamespace(content=[SimpleNamespace(type='text', text=text)],
                           stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=1, output_tokens=7))


def _fake_create(monkeypatch, create):
    monkeypatch.setattr(anthropic, 'Anthropic', lambda **kw: SimpleNamespace(
        messages=SimpleNamespace(create=create)))


@pytest.mark.parametrize('stop', ['max_tokens', 'model_context_window_exceeded'])
def test_raise_if_cut_off_refuses_both_stop_reasons(stop):
    with pytest.raises(ProbeOutputError, match=stop) as e:
        raise_if_cut_off(_resp(stop), 'm', 100)
    assert e.value.text == 'partial'


@pytest.mark.parametrize('stop', ['end_turn', 'tool_use', 'stop_sequence'])
def test_raise_if_cut_off_passes_finished_replies(stop):
    raise_if_cut_off(_resp(stop), 'm', 100)


def test_empty_system_is_not_sent(monkeypatch):
    seen = {}
    _fake_create(monkeypatch, lambda **kw: seen.update(kw) or _resp('end_turn'))
    call_sonnet('', 'user')
    assert 'system' not in seen


from eval.longmem import answerer, classifier, judge as longmem_judge  # noqa: E402


def test_longmem_judge_cut_off_verdict_is_ungraded(monkeypatch):
    _fake_client(monkeypatch, '{"verdict": "YES", "compar', 'max_tokens', 300)
    j = longmem_judge.judge_one('q', 'gold', 'hyp')
    assert j['judge_parse_failed'] is True and j['correct'] is False


def test_longmem_judge_failed_call_is_ungraded_not_a_measured_no(monkeypatch):
    def boom(**kw):
        raise RuntimeError('529 overloaded')
    _fake_create(monkeypatch, boom)
    assert longmem_judge.judge_one('q', 'gold', 'hyp')['judge_parse_failed'] is True


def test_longmem_judge_complete_verdict_is_graded(monkeypatch):
    _fake_client(monkeypatch, '{"verdict": "YES", "comparison": "equivalent", '
                 '"reasoning": "same fact"}', 'end_turn', 20)
    j = longmem_judge.judge_one('q', 'gold', 'hyp')
    assert j['correct'] is True and 'judge_parse_failed' not in j


def test_longmem_answerer_cut_off_is_an_answerer_error(monkeypatch):
    _fake_client(monkeypatch, 'The answer is', 'max_tokens', 400)
    a = answerer.answer_question('q', 'memories')
    assert a['hypothesis'] == '' and 'cut off' in a['error']


@pytest.mark.parametrize('text, stop, present', [
    ('MISSING', 'end_turn', False),
    ('PRESENT', 'end_turn', True),
    ('Looking at the context', 'end_turn', True),  # neither word: loud default
    ('PRES', 'max_tokens', True),                  # cut off: loud default
])
def test_sufficiency_judge_defaults_loudly_on_an_unusable_verdict(
        monkeypatch, capsys, text, stop, present):
    _fake_client(monkeypatch, text, stop, 1)
    assert classifier._context_has_gold('q', 'gold', 'context') is present
    warned = 'WARN sufficiency judge failed' in capsys.readouterr().out
    assert warned is (text not in ('MISSING', 'PRESENT'))


def test_schema_gate_cut_off_round_is_not_a_skip():
    from eval import mcp_schema_gate
    client = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: _resp('max_tokens', 'Weighing the cluster')))
    with pytest.raises(ProbeOutputError, match='max_tokens'):
        mcp_schema_gate.decide_turn(
            client, {'user': 'u', 'model': 'm', 'system': 's', 'tools': []}, None)


def test_skill_eval_cut_off_turn_fails_the_run_and_failed_runs_exit():
    from eval import skill_eval
    client = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: _resp('max_tokens')))
    with pytest.raises(ProbeOutputError):
        skill_eval.run_single(client, 'm', 'skill',
                              {'messages': [{'role': 'user', 'content': 'hi'}]})
    failed = {'v': {'name': 'V', 'scenarios': {},
                    'errors': [{'scenario': 's', 'run': 0, 'error': 'cut off'}]}}
    with pytest.raises(SystemExit, match='1 runs failed'):
        skill_eval.exit_on_failed_runs(failed)
    skill_eval.exit_on_failed_runs({'v': {'name': 'V', 'scenarios': {}, 'errors': []}})
