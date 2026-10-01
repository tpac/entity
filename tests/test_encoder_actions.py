"""Actions condenser (servers/scales/s1/encoder_actions.py) — parse →
condense → render over typed records.

Pins the invariants:
  • SELF-MARKING — '×N' only for true repeats of the same recorded action;
    the '(N more actions, not shown: …)' accounting line; ' …' on every
    multi-line trim; '+k more' on every capped breakdown
  • COUNT-CONSERVING — rendered ×N counts + accounting count = true total
  • TOTAL PARSE — unknown tools degrade to a generic record, never an error
  • PROTECTION — closing actions keep their outcome slot; writes never roll

Fixtures mirror the production timelines the design was reviewed against
(review-sweep flood, build session, journal-check mix) and the four-angle
Opus review's reproduced findings (dea6cdd review, 2026-08-18).
"""
import os
import re
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.action_policy import load_action_policy  # noqa: E402
from servers.scales.s1.encoder_actions import (  # noqa: E402
    parse_action, prepare_action_block)
from servers.scales.s1.encoder_view import (  # noqa: E402
    ACTION_BUDGETS, ACTIONS_BUDGET_SOFT_EDGE, ACTIONS_KEEP_LAST)

ACTIONS_BUDGET, ACTIONS_BUDGET_TAIL = ACTION_BUDGETS['balanced']


def condense_actions(episodes, is_tail=False):
    """One turn's condensed action lines, through the production door the
    lived timeline uses (encode._render_lived_sequence_timeline)."""
    return [line.text for line in prepare_action_block(
        episodes, is_tail=is_tail, encoded=False, view_policy=True,
        policy=load_action_policy()).lines]

# The plugin-adapter tool prefix, built from the manifests — never the literal
# (test_deploy_contract's containment gate keeps the adapter shape out of
# source; .claude/settings.json is its one legitimate home). CC's shape is
# mcp__plugin_<plugin>_<server>__<tool>: the plugin name comes from plugin.json,
# the server name is the .mcp.json key — two namespaces (D-11), not one.
import json  # noqa: E402
with open(os.path.join(ROOT, '.claude-plugin', 'plugin.json')) as _f:
    _PLUGIN = json.load(_f)['name']
with open(os.path.join(ROOT, '.mcp.json')) as _f:
    _servers = list(json.load(_f))
assert _servers == ['brain'], f'.mcp.json servers changed: {_servers}'
_SERVER = _servers[0]
PLUGIN_TOOL = ('mcp__plugin_%s_%s__' % (_PLUGIN, _SERVER)) + '%s'
ROLLUP_RE = re.compile(r'^\((\d+) more actions, not shown: ')


@pytest.fixture(autouse=True)
def balanced_action_policy(monkeypatch):
    monkeypatch.setenv('BRAIN_ACTIONS_PROFILE', 'balanced')
    monkeypatch.setenv('BRAIN_ACTIONS_EXCLUDE_GIT', '0')
    monkeypatch.setenv('BRAIN_ACTIONS_MAX_LINES', '200')
    monkeypatch.setenv('BRAIN_ACTIONS_MAX_BYTES', '24000')


def _ep(summary, tool='Bash'):
    return {'summary': summary, 'metadata': {'tool': tool}}


def _accounted_total(lines):
    """Apply the module's own audit rule: sum ×N counts + rollup counts."""
    total = 0
    for ln in lines:
        m = ROLLUP_RE.match(ln)
        if m:
            total += int(m.group(1))
            continue
        m = re.search(r' ×(\d+)$', ln)
        total += int(m.group(1)) if m else 1
    return total


# ── parse: tool families ──

def test_plain_bash_single_line():
    a = parse_action(_ep('Bash: git status --short'))
    assert a.tool == 'Bash' and a.sub == 'git'
    assert a.label == 'Bash: git status --short'


def test_bash_verb_skips_wrappers_and_quoted_cd():
    assert parse_action(_ep('Bash: cd "/a b" && git status')).sub == 'git'
    assert parse_action(_ep('Bash: ./dev python3 -m pytest tests/ -q')).sub == 'pytest'
    assert parse_action(_ep('Bash: /usr/bin/env python3 foo.py')).sub == 'foo.py'
    assert parse_action(_ep('Bash: ../scripts/build.sh')).sub == 'build.sh'


def test_heredoc_harvests_intent_comment_and_marks_trim():
    a = parse_action(_ep(
        "Bash: python3 - <<'EOF'\n"
        '# dal_graph.py: delete decay_edges + DEFAULT_EXCLUDED_RELATIONS\n'
        "p = 'servers/dal_graph.py'\nsrc = open(p).read()"))
    assert 'delete decay_edges + DEFAULT_EXCLUDED_RELATIONS' in a.label
    assert a.label.endswith(' …')          # body dropped, marked — always
    assert '\n' not in a.label


def test_script_without_comment_falls_back_to_first_code_line():
    a = parse_action(_ep(
        'Bash: ./dev python3 -c "\n'
        'from servers.db_backup import backup_before_destructive\n'
        "p = backup_before_destructive('/x/brain.db', 'tag')"))
    assert 'backup_before_destructive' in a.label   # the evidence line
    assert a.label.endswith(' …')


def test_script_skips_shebang_and_coding_cookie():
    a = parse_action(_ep(
        "Bash: cat > /tmp/x.py <<'EOF'\n# -*- coding: utf-8 -*-\nimport os"))
    assert '-*-' not in a.label
    assert 'import os' in a.label


def test_git_commit_harvests_subject():
    a = parse_action(_ep(
        'Bash: git add -A && git commit -m "$(cat <<\'EOF\'\n'
        'refactor(edges): phase 2 — purge retired rows\n\nbody text'))
    assert 'refactor(edges): phase 2 — purge retired rows' in a.label
    assert a.protected                      # commit is a write verb


def test_compound_does_not_misattribute_later_heredoc_comment():
    # The heredoc opens on line 2; its comment must NOT read as the intent
    # of the leading `git rm` (encoder-eye finding 10).
    a = parse_action(_ep(
        'Bash: git rm -q scripts/old.py\n'
        "python3 - <<'EOF'\n# brain_constants: strip dead keys\nx = 1"))
    assert '·' not in a.label
    assert a.label.endswith(' …')


def test_edit_drops_old_new_heads_but_stays_distinct():
    e1 = _ep('Edit: /Users/x/brain/tests/t.py\n  old: 4: aaa\n  new: 4: aaa\n 5: b',
             tool='Edit')
    e2 = _ep('Edit: /Users/x/brain/tests/t.py\n  old: 5: ccc\n  new: 5: ddd',
             tool='Edit')
    a1 = parse_action(e1)
    assert 'old:' not in a1.label and a1.label.endswith(' …')
    # same rendered label, different raw → never folds into a false ×2
    lines = condense_actions([e1, e2])
    assert len(lines) == 2 and not any('×' in ln for ln in lines)


def test_long_paths_squeeze_keeps_leading_slash():
    a = parse_action(_ep(
        'Read: /Users/dev/repo/.claude/worktrees/wt-x/servers/scales/s2/community.py',
        tool='Read'))
    assert a.label == 'Read: /…/scales/s2/community.py'


def test_urls_are_never_squeezed():
    a = parse_action(_ep(
        'WebFetch: https://docs.python.org/3/library/re.html#module-re',
        tool='WebFetch'))
    assert 'https://docs.python.org/3/library/re.html' in a.label


def test_targets_keep_leading_slash_and_skip_urls():
    a = parse_action(_ep(
        'Bash: cp /tmp/x1.txt servers/dal.py && curl https://h.com/a/b/c.py'))
    assert '/tmp/x1.txt' in a.targets
    assert 'servers/dal.py' in a.targets
    assert not any('h.com' in t or 'a/b/c.py' in t for t in a.targets)


def test_mcp_tool_name_shortens():
    a = parse_action(_ep(PLUGIN_TOOL % 'query_traces' + ': {"scale": "s1"}',
                         tool=PLUGIN_TOOL % 'query_traces'))
    assert a.tool == 'query_traces'


def test_unknown_tool_and_empty_summary_are_total():
    a = parse_action(_ep('SomeFutureTool: whatever args', tool='SomeFutureTool'))
    assert a.label.startswith('SomeFutureTool:')
    b = parse_action(_ep('', tool=None))
    assert b.label == 'tool (no cue)'


def test_existing_policy_drop_and_stub_respected():
    dropped = parse_action(_ep(PLUGIN_TOOL % 'remember' + ': {...}',
                               tool=PLUGIN_TOOL % 'remember'))
    assert dropped is None                 # provenance owns node-ops
    stubbed = parse_action(_ep(PLUGIN_TOOL % 'recall' + ': {"query": "x"}',
                               tool=PLUGIN_TOOL % 'recall'))
    assert stubbed.label.endswith('→ results in provenance')


# ── condense: dedup identity, protection, budget ──

def test_exact_repeats_fold_but_closing_slot_survives():
    # A closing action that duplicates a mid-turn one must still render at
    # the END (correctness finding 1: the outcome slot). The fold happens
    # only among body actions.
    eps = ([_ep('Bash: grep -rn "S%d" servers/' % i) for i in range(18)]
           + [_ep('Bash: ./dev pytest tests/ -q')]        # mid-turn run
           + [_ep('Read: /a/b/c/d.py', tool='Read')]
           + [_ep('Bash: ./dev pytest tests/ -q')])       # closing re-run
    lines = condense_actions(eps)
    assert lines[-1] == 'Bash: ./dev pytest tests/ -q'    # outcome, in place
    assert _accounted_total(lines) == len(eps)


def test_different_scripts_same_first_line_never_fold():
    # Dedup keys on the RAW summary — three different heredocs sharing an
    # opener line render as three lines (invariants finding 1).
    eps = [_ep("Bash: python3 - <<'EOF'\nimport os\nos.remove('x')"),
           _ep("Bash: python3 - <<'EOF'\nimport sys\nprint(sys.path)"),
           _ep("Bash: python3 - <<'EOF'\nopen('y','w').write('z')")]
    lines = condense_actions(eps)
    assert len(lines) == 3
    assert not any('×' in ln for ln in lines)


def test_worktree_and_main_tree_same_suffix_never_fold():
    eps = [_ep('Read: /Users/x/brain/servers/scales/s1/encode.py', tool='Read'),
           _ep('Read: /Users/x/brain/.claude/worktrees/wt/servers/scales/s1/encode.py',
               tool='Read')]
    lines = condense_actions(eps)
    assert len(lines) == 2


def test_writes_never_roll_up():
    # 40 greps + 5 Edits scattered through the middle: every Edit renders.
    eps = []
    for i in range(40):
        eps.append(_ep('Bash: grep -rn "S%d" servers/' % i))
        if i % 8 == 0:
            eps.append(_ep('Edit: /Users/x/brain/servers/f%d.py\n  old: a\n  new: b' % i,
                           tool='Edit'))
    lines = condense_actions(eps)
    edit_lines = [ln for ln in lines if ln.startswith('Edit:')]
    assert len(edit_lines) == 5
    assert _accounted_total(lines) == len(eps)


def test_under_soft_edge_renders_verbatim_no_rollup():
    n = ACTIONS_BUDGET + ACTIONS_BUDGET_SOFT_EDGE
    lines = condense_actions([_ep('Bash: step %d' % i) for i in range(n)])
    assert len(lines) == n
    assert not any(ROLLUP_RE.match(ln) for ln in lines)


def test_one_past_soft_edge_condenses():
    n = ACTIONS_BUDGET + ACTIONS_BUDGET_SOFT_EDGE + 1
    lines = condense_actions([_ep('Bash: step %d' % i) for i in range(n)])
    assert sum(1 for ln in lines if ROLLUP_RE.match(ln)) == 1


def test_over_budget_rolls_middle_and_keeps_last():
    n = 60
    eps = [_ep('Bash: grep sweep %d servers/dal_graph.py' % i) for i in range(n - 1)]
    eps.append(_ep('Bash: git status --short'))
    lines = condense_actions(eps)
    assert lines[-1] == 'Bash: git status --short'
    rollup = [ln for ln in lines if ROLLUP_RE.match(ln)]
    assert len(rollup) == 1 and 'touched:' in rollup[0]
    assert 'servers/dal_graph.py' in rollup[0]
    assert _accounted_total(lines) == n


def test_count_conservation_over_mixed_flood_with_folds():
    # The ×N branch of the audit rule must be exercised on RENDERED lines
    # too (invariants finding 6): identical greps land in the kept head.
    eps = ([_ep('Bash: git show HEAD --stat')] * 3          # folds, in head
           + [_ep('Bash: grep -rn "SYM_%d" servers/' % i) for i in range(70)]
           + [_ep('Read: /Users/x/brain/servers/f%d.py' % i, tool='Read')
              for i in range(20)]
           + [_ep('Edit: /Users/x/brain/servers/g%d.py\n  old: a\n  new: b' % i,
                  tool='Edit') for i in range(12)])
    lines = condense_actions(eps)
    assert any(re.search(r' ×3$', ln) for ln in lines)      # rendered fold
    assert _accounted_total(lines) == len(eps)


def test_tail_turn_gets_larger_budget():
    eps = [_ep('Bash: step %d' % i) for i in range(ACTIONS_BUDGET_TAIL)]
    assert len(condense_actions(eps, is_tail=True)) == ACTIONS_BUDGET_TAIL
    assert any(ROLLUP_RE.match(ln)
               for ln in condense_actions(eps, is_tail=False))


def test_rollup_marks_every_internal_cap():
    # 8 distinct Bash verbs (cap 4) and 12 distinct targets (cap 8): both
    # breakdowns must mark their truncation (invariants findings 4+5).
    verbs = ['cat', 'awk', 'find', 'sort', 'cut', 'tr', 'wc', 'head']
    eps = []
    for v in verbs:
        for i in range(5):
            eps.append(_ep('Bash: %s /Users/x/brain/servers/deep/dir/t_%s_%d.py'
                           % (v, v, i)))
    lines = condense_actions(eps)
    rollup = next(ln for ln in lines if ROLLUP_RE.match(ln))
    assert re.search(r', \+\d+ more\)', rollup)             # verbs cap marked
    assert re.search(r'touched: .*\+\d+ more', rollup)      # targets cap marked
    assert _accounted_total(lines) == len(eps)


def test_keep_last_constant_is_honored():
    eps = [_ep('Bash: a%d' % i) for i in range(50)]
    lines = condense_actions(eps)
    tail_labels = ['Bash: a%d' % i for i in range(50 - ACTIONS_KEEP_LAST, 50)]
    assert lines[-ACTIONS_KEEP_LAST:] == tail_labels


def test_rollup_subcounts_belong_to_each_display_tool():
    from servers.scales.s1.encoder_actions import _Action, _rollup_line
    # Records exercise D7(a) before a kind-aware parser exists. Counts from
    # one shell display name must never contaminate another shell's entry.
    one = _Action('one', 'terminal_one', 'pytest', 'one', (), False)
    one.count = 3
    two = _Action('two', 'terminal_two', 'git', 'two', (), False)
    two.count = 2
    line = _rollup_line([one, two])
    assert line == ('(5 more actions, not shown: '
                    'terminal_one ×3 (pytest ×3), terminal_two ×2 (git ×2))')


def _stamped(summary, tool='apply_patch', kind='edit', **overrides):
    # Literal write vocabulary, deliberately independent of the input map.
    md = {'tool': tool, 'kind': kind, 'kind_status': 'ok',
          'vocab_version': 1, 'impl_identity': 'fixture-implementation'}
    md.update(overrides)
    return {'summary': summary, 'metadata': md}


def test_legacy_summary_behavior_is_frozen():
    for head in ('Edit', 'Write', 'NotebookEdit'):
        a = parse_action(_ep(head + ': src/code.py', tool='unrelated_metadata'))
        assert (a.kind, a.kind_status, a.protected) == ('edit', 'legacy', True)
    old_patch = _ep('apply_patch: src/code.py', tool='apply_patch')
    old_patch['metadata'].update(host='codex', tool_use_id='old-id')
    a = parse_action(old_patch)
    assert (a.kind, a.kind_status, a.protected) == ('', 'legacy', False)
    a = parse_action(_ep('Bash: ./dev python3 -m pytest', tool='unrelated_metadata'))
    assert (a.kind, a.kind_status, a.sub) == ('shell', 'legacy', 'pytest')


def test_reader_supports_current_write_vocabulary():
    from servers.host_contract import VOCAB_VERSION
    from servers.scales.s1.encoder_view import SUPPORTED_ACTION_VOCAB_VERSIONS
    assert VOCAB_VERSION in SUPPORTED_ACTION_VOCAB_VERSIONS


def test_normalized_kinds_override_summary_names_and_keep_raw_names():
    a = parse_action(_stamped('Read: src/code.py', tool='future_editor'))
    assert (a.kind, a.tool, a.raw_tool, a.protected) == (
        'edit', 'future_editor', 'future_editor', True)
    a = parse_action(_stamped('Edit: src/code.py', tool='future_reader', kind='read'))
    assert a.kind == 'read' and not a.protected
    assert a.label == 'Edit: src/code.py'  # summary capture is unchanged
    a = parse_action(_stamped('Bash: git status', tool='future_reader', kind='read'))
    assert a.sub == ''


def test_normalized_shell_intent_and_rollup_use_kind():
    a = parse_action(_stamped('terminal: git commit -m "$(cat <<\'EOF\'\n'
                             'Ship parser change\nEOF', tool='terminal', kind='shell'))
    assert a.sub == 'git' and a.protected
    assert '· Ship parser change' in a.label
    from servers.scales.s1.encoder_actions import _rollup_line
    one = parse_action(_stamped('terminal: pytest tests', tool='terminal', kind='shell'))
    two = parse_action(_stamped('runner: rg code', tool='runner', kind='shell'))
    assert _rollup_line([one, two]) == (
        '(2 more actions, not shown: terminal ×1 (pytest ×1), runner ×1 (rg ×1))')


@pytest.mark.parametrize('overrides', [
    {'kind': '', 'kind_status': 'unknown'},
    {'kind': 'edit', 'kind_status': 'unknown'},
    {'kind': 'unrecognized'}, {'kind': []}, {'kind_status': []},
    {'kind_status': 'oops'}, {'vocab_version': 999}, {'vocab_version': True},
    {'vocab_version': '1'}, {'impl_identity': None}, {'impl_identity': ''},
    {'tool': []}, {'tool': ''},
])
def test_invalid_stamps_never_use_legacy_behavior(overrides):
    ep = _stamped('Bash: python3 - <<EOF\n# should not harvest\nEOF',
                  tool='Bash', kind='shell')
    ep['metadata'].update(overrides)
    a = parse_action(ep)
    assert a.kind == '' and a.sub == '' and a.protected
    assert 'tool kind ' in a.label and 'action unclassified' in a.label
    assert 'should not harvest' not in a.label
    assert a.label.endswith(' …')
    assert a.kind_status in ('unknown', 'malformed')
    # Invalid raw names must also survive the condenser's hash key.
    assert condense_actions([ep] * 4)


@pytest.mark.parametrize('key,value', [
    ('kind', 'edit'), ('kind_status', 'ok'), ('vocab_version', 1),
    ('impl_identity', 'fixture-implementation')])
def test_partial_stamp_is_diagnostic_not_legacy(key, value):
    ep = _ep('Edit: src/code.py', tool='Edit')
    ep['metadata'][key] = value
    a = parse_action(ep)
    assert a.kind == '' and a.kind_status == 'malformed'
    assert 'action unclassified' in a.label


def test_mixed_states_do_not_dedup_or_lose_diagnostics():
    summary = 'apply_patch: src/middle.py'
    old = _ep(summary, tool='apply_patch')
    valid = _stamped(summary)
    unknown = _stamped(summary, kind='', kind_status='unknown')
    partial = _ep(summary, tool='apply_patch')
    partial['metadata']['kind'] = 'edit'
    eps = ([_ep('Bash: step %d' % i) for i in range(40)]
           + [old, valid, valid, unknown, partial]
           + [_ep('Bash: done'), _ep('Bash: finished')])
    lines = condense_actions(eps)
    assert summary + ' ×2' in lines
    assert sum('action unclassified' in ln for ln in lines) == 2
    assert _accounted_total(lines) == len(eps)
    assert lines[-2:] == ['Bash: done', 'Bash: finished']


def test_normalized_mcp_namespace_policy():
    for base in ('remember', 'recall'):
        brain_tool = 'mcp__plugin_x_brain__' + base
        other_tool = 'mcp__other__' + base
        summary = base + ': {"query": "kept intent"}'
        brain = parse_action(_stamped(summary, tool=brain_tool, kind='mcp'))
        other = parse_action(_stamped(summary, tool=other_tool, kind='mcp'))
        assert other.label == summary
        if base == 'remember':
            assert brain is None
        else:
            assert brain.label.endswith('→ results in provenance')
        unknown = parse_action(_stamped(summary, tool=brain_tool, kind='', kind_status='unknown'))
        assert unknown.protected and 'action unclassified' in unknown.label
    eps = [_stamped('query: same', tool='mcp__one__query', kind='mcp'),
           _stamped('query: same', tool='mcp__two__query', kind='mcp'),
           _ep('tail: one'), _ep('tail: two')]
    assert len(condense_actions(eps)) == 4


def test_normalized_patch_survives_busy_turn_without_reclassifying_history():
    flood = [_ep('Bash: step %d' % i) for i in range(40)]
    tail = [_ep('Bash: outcome'), _ep('Bash: final')]
    summary = 'apply_patch: src/retained.py'
    assert summary not in condense_actions(flood + [_ep(summary)] + tail)
    lines = condense_actions(flood + [_stamped(summary)] + tail)
    assert summary in lines
    assert _accounted_total(lines) == 43
