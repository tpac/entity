"""The paired-code gate rejects input drift, including the observed clock trap."""
import json

import pytest

from tests.eval_optional import require_eval
require_eval()

from eval.host_contract_eval import compare_inputs


def _pair(tmp_path):
    report = dict(snapshot_hashes={'brain.db': 'frozen'}, session='session',
                  render_now='2026-09-08T17:25:48+00:00', flags={}, messages=[],
                  catalog_ids=[], interaction_stamp={}, interaction_config={},
                  eval_execution={'model': 'same-model', 'effort': 'API default'},
                  stamped_edits=1,
                  calls=[{'is_tail': True,
                          'episodes': [{'metadata': {'kind': 'edit', 'kind_status': 'ok'}}],
                          'parsed': [{'protected': True, 'label': 'edit: src/code.py'}]}])
    before, after = tmp_path / 'before', tmp_path / 'after'
    for p, action in ((before, 'rolled up'), (after, 'edit: src/code.py')):
        p.mkdir()
        report['calls'][0]['lines'] = [action]
        (p / 'input.json').write_text(json.dumps(report))
        (p / 'system.txt').write_text('same system')
        (p / 'tools.json').write_text('[]')
        (p / 'prompt.txt').write_text('catalog age 5m\n<actions>' + action + '</actions>')
    return before, after, report


def test_pair_allows_only_action_changes(tmp_path):
    before, after, _ = _pair(tmp_path)
    assert compare_inputs(before, after)['paired'] is True
    prompt = (after / 'prompt.txt').read_text()
    (after / 'prompt.txt').write_text(prompt.replace('age 5m', 'age 12m'))
    with pytest.raises(ValueError, match='outside actions'):
        compare_inputs(before, after)


@pytest.mark.parametrize('field', ['snapshot_hashes', 'render_now', 'messages',
                                   'interaction_config', 'eval_execution'])
def test_pair_refuses_changed_substrate(tmp_path, field):
    before, after, report = _pair(tmp_path)
    report[field] = 'changed'
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match=field):
        compare_inputs(before, after)


def test_pair_refuses_unprotected_stamped_edits(tmp_path):
    before, after, report = _pair(tmp_path)
    report['calls'][0]['parsed'][0]['protected'] = False
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not all protected'):
        compare_inputs(before, after)


def test_pair_refuses_changed_budget_input(tmp_path):
    before, after, report = _pair(tmp_path)
    report['calls'][0]['is_tail'] = False
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='different action windows'):
        compare_inputs(before, after)


def test_pair_accepts_legacy_metadata_absence(tmp_path):
    before, after, _ = _pair(tmp_path)
    for path in (before, after):
        report = json.loads((path / 'input.json').read_text())
        report['calls'][0]['episodes'].append({'metadata': None})
        report['calls'][0]['parsed'].append({'protected': False})
        (path / 'input.json').write_text(json.dumps(report))
    assert compare_inputs(before, after)['protected_edits'] == 1


def test_pair_refuses_protected_edits_lost_by_condenser(tmp_path):
    before, after, report = _pair(tmp_path)
    report['calls'][0]['lines'] = ['all actions rolled up']
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not all retained'):
        compare_inputs(before, after)


def test_pair_refuses_action_blocks_lost_by_prompt_renderer(tmp_path):
    before, after, _ = _pair(tmp_path)
    (after / 'prompt.txt').write_text('catalog age 5m\n<actions>ALL ACTIONS LOST</actions>')
    with pytest.raises(ValueError, match='omits a measured action block'):
        compare_inputs(before, after)


def test_strict_retention_gate_refuses_global_limit_omissions(tmp_path):
    before, after, report = _pair(tmp_path)
    report['action_allocation'] = {'notice': '<action_limit>1 action omitted</action_limit>'}
    (after / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='exceeded the timeline action limit'):
        compare_inputs(before, after)


def test_pair_refuses_different_effective_action_settings(tmp_path):
    before, after, report = _pair(tmp_path)
    for path, profile in ((before, 'thin'), (after, 'balanced')):
        record = json.loads((path / 'input.json').read_text())
        record['action_allocation'] = {'policy': {'profile': profile}, 'notice': ''}
        (path / 'input.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='action policy'):
        compare_inputs(before, after)


def test_eval_settings_apply_to_initial_and_followup_requests(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from eval import s1_encode_eval as runner

    monkeypatch.setattr(runner, 'EVAL_MODEL', 'fixture-model')
    monkeypatch.setattr(runner, 'EVAL_MAX_TOKENS', 128)
    monkeypatch.setattr(runner, 'EVAL_TOOL_ROUNDS', 2)
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason='tool_use',
        content=[SimpleNamespace(type='tool_use', id='call', name='remember_batch',
                                 input={'nodes': []})])
    client = Mock()
    client.messages.create.return_value = response
    result = runner.run_encoding(client, 'system', 'content', [], object())
    assert client.messages.create.call_count == 3
    assert len(result['actions']) == 2
    for call in client.messages.create.call_args_list:
        assert call.kwargs['model'] == 'fixture-model'
        assert call.kwargs['max_tokens'] == 128
        assert 'output_config' not in call.kwargs


@pytest.mark.parametrize('caption', ['edit: src/code.py (2 edit calls)',
                                     'Closing: edit: src/code.py ×2'])
def test_retention_gate_counts_grouped_and_closing_edit_cues(tmp_path, caption):
    before, after, _ = _pair(tmp_path)
    for path in (before, after):
        report = json.loads((path / 'input.json').read_text())
        call = report['calls'][0]
        call['episodes'] *= 2
        call['parsed'] *= 2
        report['stamped_edits'] = 2
        if path == after:
            call['lines'] = [caption]
            (path / 'prompt.txt').write_text('catalog age 5m\n<actions>' + caption + '</actions>')
        (path / 'input.json').write_text(json.dumps(report))
    assert compare_inputs(before, after)['retained_edits'] == 2


def test_retention_gate_rejects_undercounted_edit_group(tmp_path):
    before, after, _ = _pair(tmp_path)
    for path in (before, after):
        report = json.loads((path / 'input.json').read_text())
        call = report['calls'][0]
        call['episodes'] *= 3
        call['parsed'] *= 3
        report['stamped_edits'] = 3
        if path == after:
            call['lines'] = ['edit: src/code.py (2 edit calls)']
        (path / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not all retained'):
        compare_inputs(before, after)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('caption_length', [40, 179, 180, 181])
def test_retention_gate_accepts_actual_group_with_mixed_multiline_markers(tmp_path, reverse, caption_length):
    from servers.action_policy import ActionPolicy
    from servers.scales.s1.encoder_actions import parse_action, prepare_action_block
    before, after, report = _pair(tmp_path)
    def ep(summary, kind='edit'):
        return {'summary': summary, 'metadata': {'tool': 'Edit' if kind == 'edit' else 'Bash',
                'kind': kind, 'kind_status': 'ok', 'vocab_version': 1, 'impl_identity': 'test'}}
    caption = 'Edit: src/' + 'a' * (caption_length - len('Edit: src/.py')) + '.py'
    edits = [ep(caption), ep(caption + '\nold: a\nnew: b')]
    if reverse:
        edits.reverse()
    episodes = [edits[0], ep('Bash: pytest middle', 'shell'), edits[1],
                ep('Bash: pytest end', 'shell'), ep('Bash: deploy', 'shell')]
    block = prepare_action_block(episodes, is_tail=True, encoded=False,
                                 view_policy=True, policy=ActionPolicy())
    report['stamped_edits'] = 2
    report['calls'] = [{'is_tail': True, 'episodes': episodes,
                       'parsed': [{'protected': (a := parse_action(e)).protected, 'label': a.label}
                                  for e in episodes],
                       'lines': [line.text for line in block.lines]}]
    prompt = 'catalog age 5m\n<actions>' + '\n'.join(report['calls'][0]['lines']) + '</actions>'
    for path in (before, after):
        (path / 'input.json').write_text(json.dumps(report))
        (path / 'prompt.txt').write_text(prompt)
    assert compare_inputs(before, after)['retained_edits'] == 2


@pytest.mark.parametrize('tool_collision', [False, True])
def test_retention_gate_refuses_ambiguous_shortened_captions(tmp_path, tool_collision):
    before, after, _ = _pair(tmp_path)
    for path in (before, after):
        report = json.loads((path / 'input.json').read_text())
        call = report['calls'][0]
        call['episodes'] = [{'summary': 'Edit: /root/%s/servers/code.py' % prefix,
                             'metadata': {'kind': 'edit', 'kind_status': 'ok', 'tool': tool}}
                            for prefix, tool in [('one', 'editor_one'),
                                                 ('one' if tool_collision else 'two', 'editor_two')]]
        call['parsed'] *= 2
        report['stamped_edits'] = 2
        (path / 'input.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='ambiguous rendered edit captions'):
        compare_inputs(before, after)


def test_s1_encode_eval_counts_landed_and_refused_revises(monkeypatch):
    """Writes are answered as production would answer them: brain_batch ops
    count (the encoder's main write tool), and a revise whose swap would miss
    is refused — counted as a failed attempt, not as a revision."""
    from types import SimpleNamespace
    from unittest.mock import Mock
    from eval import s1_encode_eval as runner

    class Copy:
        def get_node(self, ids):
            return {i + '0' * 24: {'id': i + '0' * 24, 'title': 't',
                                   'content': 'stored text here'} for i in ids}

    monkeypatch.setattr(runner, 'EVAL_TOOL_ROUNDS', 2)
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    write = SimpleNamespace(usage=usage, stop_reason='tool_use', content=[
        SimpleNamespace(type='tool_use', id='w', name='brain_batch', input={'operations': [
            {'op': 'remember', 'type': 'fact', 'title': 'n', 'content': 'c', 'situation': 's'},
            {'op': 'revise', 'node_id': 'abcdef01', 'reason': 'r',
             'content': {'old': 'stored text here', 'new': 'fixed'}},
            {'op': 'revise', 'node_id': 'abcdef02', 'reason': 'r',
             'content': {'old': 'text that was never stored', 'new': 'x'}}]})])
    done = SimpleNamespace(usage=usage, stop_reason='end_turn',
                           content=[SimpleNamespace(type='text', text='done')])
    client = Mock()
    client.messages.create.side_effect = [write, done]
    result = runner.run_encoding(client, 'system', 'content', [], Copy())
    assert (result['node_count'], result['revision_count'],
            result['revisions_refused']) == (1, 1, 1)
    assert result['rich_fields'] == [['situation']]
