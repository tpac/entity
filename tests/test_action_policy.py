"""User settings and literal Git-call exclusion, including quoted-data controls."""
from unittest.mock import Mock

import pytest

from servers.action_policy import ActionPolicy, has_git_command, load_action_policy
from servers.dispatch_observability import _handle_trace_append
from tests import test_tool_result_stamp as hook_tests


@pytest.mark.parametrize('command', [
    'git status', '/usr/bin/git log', 'git status && rg needle .',
    'rg needle .; git status', 'cat README | git hash-object --stdin',
    'X=1 env -u Y git status', 'sudo -u someone git log', 'timeout 10 git status',
    'command git log', './dev git status', "bash -lc 'git status && echo done'",
    'echo "$(git status)"', 'echo "$(echo $(git status))"', 'echo `git status`',
    "cat <<'EOF'\nnot a command\nEOF\ngit status\n",
    'cat <<EOF\n$(git status)\nEOF\n',
    'if true; then git status; fi', 'echo ok\n/usr/bin/git diff',
    'echo ' + 'x' * 1000 + '; git status',
    'sudo -n git status', 'caffeinate -t 10 git status', 'env --unset FOO git status',
    'diff <(git show HEAD:a) a', 'cat <<EOF\n$(\ngit status\n)\nEOF\n',
    'echo x#$(git status)', r'g\it status', "g'i't status",
    'g\\\nit status',
    'printf function; git status',
])
def test_git_anywhere_in_a_call_excludes_the_whole_call(command):
    assert has_git_command(command)


@pytest.mark.parametrize('command', [
    'rg git servers/', "printf '%s' 'git status'", 'echo "git status"',
    'printf "%s" ";" git', "echo '$(git status)'", 'echo git status',
    "cat <<'EOF'\ngit status\n$(git diff)\nEOF\n",
    "python3 - <<'PY'\nprint('git status')\nPY\n",
    'cat git-notes.md', 'git-lfs version', 'echo ok # git status',
    "python3 -c 'print(\"git status\")'", 'echo "escaped \\$(git status)"',
    'echo hi # $(git status)', 'echo hi # `git status`',
    'cat <<\\EOF\n$(git status)\nEOF\n', "'FOO=bar' git status", "'if' git status",
    'command -v git', 'command -V git',
    'bash script.sh -c git', 'command -pv git', 'command -pV git',
    r"printf '%s\n' \( git status \)", 'sudo -l git status', 'env --help git',
    "'>' ignored git status", 'echo \\\ngit status', "printf '%s\\n' \\\ngit status",
    'git() { echo wrapper; }', 'helper() { git status; }',
    'function helper {\ngit status\n}',
    "bash -n -c 'git status'", "sh -nc 'git status'",
])
def test_mentions_and_file_bodies_are_not_git_commands(command):
    assert not has_git_command(command)


def test_large_multiline_script_is_opaque_but_later_git_is_seen():
    command = "python -c '\n" + 'print(1)\n' * 10000 + "'"
    assert not has_git_command(command)
    assert has_git_command(command + '\ngit status')


@pytest.mark.parametrize('values', [
    {}, {'PROFILE': 'not-a-profile'}, {'MAX_LINES': '0'}, {'MAX_LINES': '-1'},
    {'MAX_LINES': '9999999'}, {'MAX_BYTES': 'unlimited'}, {'MAX_BYTES': '0'},
    {'MAX_BYTES': '99999999'}, {'EXCLUDE_GIT': 'perhaps'},
])
def test_bad_settings_never_mean_unlimited(values):
    policy = load_action_policy({'BRAIN_ACTIONS_' + k: v for k, v in values.items()})
    assert policy == ActionPolicy()


def test_full_profile_still_has_bounded_limits():
    policy = load_action_policy({'BRAIN_ACTIONS_PROFILE': 'full',
                                 'BRAIN_ACTIONS_EXCLUDE_GIT': 'false',
                                 'BRAIN_ACTIONS_MAX_LINES': '200',
                                 'BRAIN_ACTIONS_MAX_BYTES': '24000'})
    assert policy == ActionPolicy('full', False, 200, 24000)


def _capture(command, **extra):
    brain = Mock()
    brain.session_env_for.return_value = {}
    brain._trace_dal.append.return_value = 'abcdef12'
    args = {'scale': 's0', 'event_type': 'delta', 'ref_type': 'tool_result',
            'summary': 'Bash: ' + command[:200], 'tool_command': command,
            'metadata': {'tool': 'Bash', 'tells': ['PLUGIN_DATA']}, **extra}
    return brain, _handle_trace_append(brain, args, [])


def test_capture_uses_full_command_and_never_mints_an_excluded_trace(monkeypatch):
    monkeypatch.setenv('BRAIN_ACTIONS_EXCLUDE_GIT', '1')
    brain, result = _capture('echo ' + 'x' * 400 + '; git status')
    assert result == {'ok': True, 'result': {'excluded': 'git_call'}}
    brain._trace_dal.append.assert_not_called()


def test_disabled_exclusion_records_fact_but_not_full_command(monkeypatch):
    monkeypatch.setenv('BRAIN_ACTIONS_EXCLUDE_GIT', '0')
    command = 'echo ' + 'x' * 400 + '; git status'
    brain, result = _capture(command)
    assert result['result']['event_id'] == 'abcdef12'
    written = brain._trace_dal.append.call_args.kwargs
    assert written['metadata']['git_invocation'] is True
    assert command not in str(written)


def test_oversized_wire_input_is_explicitly_unclassified_and_retained(monkeypatch):
    monkeypatch.setenv('BRAIN_ACTIONS_EXCLUDE_GIT', '1')
    brain, result = _capture('git status', tool_command_omitted=True)
    assert 'event_id' in result['result']
    assert brain._trace_dal.append.call_args.kwargs['metadata']['capture_filter_incomplete'] is True
    brain._log_error.assert_called_once()


def test_unknown_kind_is_retained_even_when_cue_mentions_git(monkeypatch):
    monkeypatch.setenv('BRAIN_ACTIONS_EXCLUDE_GIT', '1')
    brain, result = _capture('git status', metadata={'tool': 'future-tool', 'tells': ['PLUGIN_DATA']})
    assert 'event_id' in result['result']
    assert brain._trace_dal.append.call_args.kwargs['metadata']['kind_status'] == 'unknown'


def test_wire_carries_full_fact_without_changing_cue_or_metadata():
    command = 'echo ' + 'x' * 400 + '; git status'
    helper = hook_tests.RawToolHookTest()
    try:
        wire = helper._wire({'tool_name': 'Bash', 'tool_input': {'command': command}}, {})
        assert wire['args']['tool_command'] == command
        assert wire['args']['summary'] == 'Bash: ' + command[:200]
        assert 'tool_command' not in wire['args']['metadata']
    finally:
        helper.doCleanups()


def test_wire_marks_oversized_commands_instead_of_breaking_transport():
    helper = hook_tests.RawToolHookTest()
    try:
        wire = helper._wire({'tool_name': 'Bash',
                             'tool_input': {'command': 'echo ' + 'x' * 910000}}, {})
        assert wire['args']['tool_command_omitted'] is True
        assert 'tool_command' not in wire['args']
        assert len(wire['args']['summary']) < 500
    finally:
        helper.doCleanups()
