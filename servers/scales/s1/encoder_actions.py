"""Shared S1 action view: parse → group/condense → allocate → serialize.

ActionLine retains event count, priority and text until the entire timeline is
allocated. Exact-summary dedup stays separate from grouping distinct edits.
Thin groups within a turn; closing cues stay separate. Every omission is counted;
priority never bypasses the final line/escaped-byte ceiling. Git capture/render exclusion and
settings live in action_policy; vocabulary and provenance live in their existing
contracts. The prompt's action glossary describes this output.
"""
import re
from collections import Counter
from dataclasses import dataclass
from html import escape

from servers.action_policy import has_git_command
from servers.scales.s1.encoder_view import (
    ACTION_BUDGETS, ACTIONS_KEEP_LAST, ACTION_LABEL_CAP,
    ACTIONS_BUDGET_SOFT_EDGE, COMMENT_SCAN_DEPTH, PATH_KEEP_SEGMENTS,
    ROLLUP_SUBS_CAP, ROLLUP_TOOLS_CAP, ROLLUP_TARGET_CAP,
    SUPPORTED_ACTION_VOCAB_VERSIONS, action_mode, action_stub, actions_stub_line)
from servers.trace_contract import ACTION_KINDS

# Frozen pre-cutover SUMMARY behavior, only for rows with no classification
# stamp. Never derive this from host_contract: that would reclassify history.
LEGACY_SUMMARY_KINDS = {'Bash': 'shell', 'Edit': 'edit', 'Write': 'edit',
                        'NotebookEdit': 'edit'}
_KIND_STAMP_KEYS = ('kind', 'kind_status', 'vocab_version', 'impl_identity')

# File-ish tokens (optional leading '/', a directory part, basename with a
# short extension) — the rollup's target vocabulary. Extension-gated: bare
# dirs and flag soup stay out. URL innards are rejected at the call site.
_TARGET_RE = re.compile(r'/?(?:[\w.@~-]+/)+[\w.@-]+\.[A-Za-z0-9]{1,5}\b')

# Absolute paths ≥4 segments squeeze to '/…/<last PATH_KEEP_SEGMENTS>' — the
# worktree/session prefixes that repeat on every line carry no per-line
# signal. The lookbehind keeps the match off URL innards ('https://host/a/b')
# and protocol-relative forms: a real path start is never preceded by a word
# char, ':' or another '/'.
_LONG_PATH_RE = re.compile(r'(?<![\w:/])(?:/[\w.@~+-]+){4,}')

# Leading shell noise before the command word that names the intent.
# cd targets may be quoted (paths with spaces).
_SHELL_PREFIX_RE = re.compile(
    r'^(?:cd\s+(?:"[^"]*"|\'[^\']*\'|\S+)\s*(?:&&|;)\s*'
    r'|sleep\s+\d+\s*(?:&&|;)\s*)+')

# Wrapper words the Bash verb extractor walks past to find the real command
# ('./dev python3 -m pytest …' counts as pytest, not dev).
_VERB_WRAPPERS = frozenset({'dev', 'python', 'python3', 'uv', 'env', 'nice',
                            'time', 'timeout', 'caffeinate'})

# Heredoc / inline-script openers: the first line is scaffolding; the intent
# lives in the body (a '#' comment, or failing that the first code line).
_SCRIPT_OPENER_RE = re.compile(r'''(?:<<-?\s*['"]?\w+['"]?\s*$|-c\s+["']\s*$)''')


class _Action:
    __slots__ = ('raw', 'tool', 'sub', 'label', 'targets', 'protected', 'count',
                 'kind', 'kind_status', 'raw_tool', 'routine')

    def __init__(self, raw, tool, sub, label, targets, protected,
                 kind='', kind_status='legacy', raw_tool=''):
        self.raw, self.tool, self.sub = raw, tool, sub
        self.label, self.targets = label, targets
        self.protected, self.count = protected, 1
        self.kind, self.kind_status, self.raw_tool = kind, kind_status, raw_tool
        self.routine = False


def _squeeze_paths(s):
    return _LONG_PATH_RE.sub(
        lambda m: '/…/' + '/'.join(m.group(0).split('/')[-PATH_KEEP_SEGMENTS:]), s)


def _one_line(s):
    return ' '.join(str(s or '').split())


def _cap(s):
    return s if len(s) <= ACTION_LABEL_CAP else s[:ACTION_LABEL_CAP] + '…'


def _short_tool(head):
    """'mcp__<server>__query_traces' → 'query_traces'."""
    return head.split('__')[-1] if head.startswith('mcp__') else head


def _bash_verb(args):
    """The command word that names a Bash action's intent: walk past shell
    prefixes (quoted-cd, sleep) and wrapper words ('./dev python3 -m pytest'
    → 'pytest'); path-shaped commands reduce to their basename. Bounded —
    inspects at most the first 6 tokens; falls back to the first token."""
    args = _SHELL_PREFIX_RE.sub('', args.strip())
    tokens = args.split()
    if not tokens:
        return ''
    fallback = tokens[0].split('/')[-1] or tokens[0]
    take_next = False
    for tok in tokens[:6]:
        if take_next:
            return tok.split('/')[-1]
        base = tok.split('/')[-1]
        if tok == '-m':
            take_next = True            # `python -m pytest` → pytest
            continue
        if tok.startswith('-') or tok.startswith('(') or '=' in tok:
            continue
        if base in _VERB_WRAPPERS:
            continue
        return base or fallback
    return fallback


def _extract_targets(summary):
    """All file-ish tokens in the summary (deduped downstream by the union).
    URL innards rejected: a match preceded by '//' or ':' is protocol
    territory, not a filesystem path."""
    out = []
    for m in _TARGET_RE.finditer(summary):
        s = m.start()
        if summary[max(0, s - 2):s] in ('//', ':/') or \
                summary[max(0, s - 1):s] == ':':
            continue
        out.append(_squeeze_paths(m.group(0)))
    return tuple(out)


def _commit_subject(lines):
    """The commit subject from a `git commit` action — the densest
    what-did-this-turn-do string in the stream. Heredoc form: the first
    non-empty body line; inline form: the head of the -m string."""
    for ln in lines[1:COMMENT_SCAN_DEPTH]:
        s = ln.strip().strip('"\'')
        if s and not s.startswith(('$(', 'EOF', '<<')):
            return s
    m = re.search(r'-m\s+["\']([^"\']+)', lines[0])
    return m.group(1).strip() if m else ''


def _script_intent(lines):
    """A script body's stated intent: its first '#' comment (skipping
    shebangs and editor cookies), else its first non-empty code line — the
    evidence line ('from servers.db_backup import backup_before_destructive'
    tells the reader what was verified)."""
    for ln in lines[1:COMMENT_SCAN_DEPTH]:
        s = ln.strip()
        if s.startswith('#') and not s.startswith('#!') and '-*-' not in s:
            return s.lstrip('# ').strip()
    for ln in lines[1:COMMENT_SCAN_DEPTH]:
        s = ln.strip()
        if s and not s.startswith('#'):    # rejected comments don't fall back
            return s
    return ''


def _label(kind, first, lines):
    """One line per action. Every multi-line trim is marked with ' …'.
    Intent harvest (the '·' segment) fires only when it can be attributed
    honestly: a commit subject for `git commit` actions; a script body's
    comment/first line only when the FIRST command opens the script (a
    compound whose heredoc comes later would mis-attribute the comment to
    the leading command)."""
    intent = ''
    if len(lines) > 1:
        if kind == 'shell' and 'git commit' in first:
            intent = _commit_subject(lines)
        elif _SCRIPT_OPENER_RE.search(first):
            intent = _script_intent(lines)
    label = ('%s · %s' % (first, _one_line(intent))) if intent else first
    if len(lines) > 1:
        label += ' …'
    return label


def _read_kind(md, summary_tool):
    """Absent / valid / unknown-or-malformed: never backfill a stamped row.

    Only classification fields mark the cutover. Historical source IDs or
    session host metadata alone do not constitute a kind stamp. Join and host
    diagnostics are the write door's concern; this reader validates the fields
    it consumes plus the classification version and implementation identity.
    """
    if not isinstance(md, dict) or not any(k in md for k in _KIND_STAMP_KEYS):
        return LEGACY_SUMMARY_KINDS.get(summary_tool, ''), 'legacy'
    kind, status = md.get('kind'), md.get('kind_status')
    if (not isinstance(md.get('tool'), str) or not md['tool']
            or not isinstance(kind, str) or not isinstance(status, str)
            or type(md.get('vocab_version')) is not int
            or md['vocab_version'] not in SUPPORTED_ACTION_VOCAB_VERSIONS
            or not isinstance(md.get('impl_identity'), str) or not md['impl_identity']):
        return '', 'malformed'
    if status == 'ok' and kind in ACTION_KINDS:
        return kind, 'ok'
    if status == 'unknown' and kind == '':
        return '', 'unknown'
    return '', 'malformed'


def parse_action(episode):
    """One tool_result episode → an _Action, or None when existing policy
    drops the line (node-ops provenance already shows). Total: an unseen
    tool shape falls through to the generic one-line record."""
    summary = str(episode.get('summary') or '')
    md = episode.get('metadata')
    raw_tool = md.get('tool') if isinstance(md, dict) else None
    raw_tool = raw_tool if isinstance(raw_tool, str) else ''

    lines = summary.split('\n')
    first = _one_line(lines[0])
    head, sep, args = first.partition(': ')
    summary_tool = _short_tool(head) if sep else 'tool'
    kind, status = _read_kind(md, summary_tool)
    tool = (_short_tool(raw_tool) if raw_tool and status != 'legacy' else summary_tool)

    # A mapping failure must remain visible even in a flood or on a brain
    # node-op that the ordinary provenance policy would drop. No shell intent
    # or write inference from summary text in this state.
    if status in ('unknown', 'malformed'):
        diagnostic = 'tool kind %s; action unclassified: ' % status
        label = _cap(diagnostic + (_squeeze_paths(first) or '(no cue)')
                     + (' …' if len(lines) > 1 else ''))
        return _Action(summary, tool, '', label, _extract_targets(summary), True,
                       kind=kind, kind_status=status, raw_tool=raw_tool)

    if isinstance(md, dict) and md.get('capture_filter_incomplete') is True:
        return _Action(summary, tool, '', _cap('capture filter incomplete; action retained: '
                       + _squeeze_paths(first)), (), True,
                       kind=kind, kind_status='malformed', raw_tool=raw_tool)

    mode = action_mode(raw_tool) if status == 'legacy' or kind == 'mcp' else 'full'
    if mode == 'drop':
        return None
    if mode == 'stub':
        stub = action_stub(summary)
        return _Action(summary, stub.split(':', 1)[0], '', stub, (), False,
                       kind=kind, kind_status=status, raw_tool=raw_tool)

    sub = _bash_verb(args) if kind == 'shell' else ''
    # Transport names remain in identity/dedup, not in every visible label.
    display_first = '%s: %s' % (_short_tool(head), args) if sep else first
    label = _cap(_squeeze_paths(_label(kind, display_first, lines)))
    if not label.strip():
        label = '%s (no cue)' % (tool or 'tool')
    intent = _script_intent(lines) if _SCRIPT_OPENER_RE.search(first) else ''
    scaffold = bool(re.match(r'^(?:import\s|from\s+[\w.]+\s+import\s)', intent))
    # An import is a poor cue, not evidence that the script only reads.
    protected = (kind == 'edit'
                 or (kind == 'shell' and sub in {'rm', 'mv', 'cp', 'touch', 'mkdir'})
                 or (' · ' in label and not scaffold))
    action = _Action(summary, tool, sub, label, _extract_targets(summary),
                     protected, kind=kind, kind_status=status, raw_tool=raw_tool)
    from servers.dispatch_common import is_brain_tool
    action.routine = (scaffold or kind == 'read'
                      or (kind == 'shell' and sub in {
                          'cat', 'sed', 'rg', 'grep', 'head', 'tail', 'wc', 'ls', 'pwd', 'find'})
                      or (is_brain_tool(raw_tool) and tool in {
                          'self_inbox', 'self_outbox', 'self_presence', 'self_peek'}))
    if scaffold:
        action.sub = 'script'  # count an opaque script, never invent its effect
    return action


def _dedup(actions, *, legacy_summary_identity=False):
    """Exact-repeat dedup keyed on the RAW summary — never the rendered
    label (a rendered label is lossy: squeezed paths, trimmed bodies, the
    180-char cap; folding on it would claim two different actions were the
    same). '×N' therefore always means the identical recorded action."""
    kept, by_raw = [], {}
    for a in actions:
        # Same summary across read states is not the same action: folding a
        # stamped edit into a legacy row would lose its protection. Preserve
        # raw MCP namespace/operation too (display names may be identical).
        key = (a.raw, a.kind, a.kind_status,
               None if legacy_summary_identity and a.kind_status == 'legacy' else a.raw_tool)
        prior = by_raw.get(key)
        if prior is not None:
            prior.count += a.count
        else:
            by_raw[key] = a
            kept.append(a)
    return kept


def _rollup_line(mid, *, grouped=False):
    """The accounting line for the unrendered middle. Every internal cap
    marks itself ('+k more') — this line's entire job is auditability."""
    total = sum(a.count for a in mid)
    tools, subs = Counter(), {}
    targets, seen = [], set()
    for a in mid:
        tools[a.tool] += a.count
        if a.sub:
            subs.setdefault(a.tool, Counter())[a.sub] += a.count
        for t in a.targets:
            if t not in seen:
                seen.add(t)
                targets.append(t)
    parts = []
    top_tools = tools.most_common(ROLLUP_TOOLS_CAP)
    for tool, cnt in top_tools:
        part = '%s ×%d' % (tool, cnt)
        tool_subs = subs.get(tool)
        if tool_subs:
            top_subs = tool_subs.most_common(ROLLUP_SUBS_CAP)
            sub_txt = ', '.join('%s ×%d' % (s, c) for s, c in top_subs)
            if len(tool_subs) > len(top_subs):
                sub_txt += ', +%d more' % (len(tool_subs) - len(top_subs))
            part += ' (%s)' % sub_txt
        parts.append(part)
    if len(tools) > len(top_tools):
        parts.append('+%d more tools' % (len(tools) - len(top_tools)))
    description = ('other actions grouped; details omitted' if grouped
                   else 'more actions, not shown')
    line = '(%d %s: %s' % (total, description, ', '.join(parts))
    if targets:
        shown = targets[:ROLLUP_TARGET_CAP]
        line += ' — touched: %s' % ', '.join(shown)
        if len(targets) > len(shown):
            line += ', +%d more' % (len(targets) - len(shown))
    return line + ')'


def _render(a):
    return a.label + (' ×%d' % a.count if a.count > 1 else '')


@dataclass(frozen=True)
class ActionLine:
    text: str
    count: int = 1
    priority: int = 0


@dataclass
class ActionBlock:
    lines: list
    inline: bool = False


def filter_action_episodes(episodes, policy):
    """Apply the capture exclusion to history without changing stored rows."""
    if not policy.exclude_git:
        return list(episodes)
    kept = []
    for episode in episodes:
        summary = str(episode.get('summary') or '')
        head, _, command = summary.partition(': ')
        kind, status = _read_kind(episode.get('metadata'), _short_tool(head))
        md = episode.get('metadata')
        known_git = isinstance(md, dict) and md.get('git_invocation') is True
        incomplete = isinstance(md, dict) and md.get('capture_filter_incomplete') is True
        if (kind == 'shell' and status in ('legacy', 'ok') and not incomplete
                and (known_git or has_git_command(command))):
            continue
        kept.append(episode)
    return kept


def _action_line(action, closing=False):
    priority = (3 if action.kind_status in ('unknown', 'malformed') else
                2 if action.protected else 1 if closing and not action.routine else 0)
    return ActionLine(_render(action), action.count, priority)


def _group_edits(actions):
    """Group edit calls by full recorded target cue, keeping other cues separate.

    A caption may name only the first file of a multi-file patch. Counts describe
    calls sharing that caption, not identical edits or complete per-file touches.
    Group before shortening paths, and never merge classification diagnostics.
    """
    edits, other = {}, []
    for action in actions:
        if action.kind == 'edit' and action.kind_status not in ('unknown', 'malformed'):
            key = (action.raw.split('\n', 1)[0], action.kind_status, action.raw_tool)
            edits.setdefault(key, []).append(action)
        else:
            other.append(_action_line(action))
    out = []
    for members in edits.values():
        count = sum(a.count for a in members)
        label = members[0].label
        if count > 1:
            label += ' (%d edit calls)' % count
        out.append(ActionLine(label, count, 2))
    return out + other


def _condense_lines(episodes, is_tail, policy):
    """Episodes of one turn → the lines its <actions> element renders.
    `is_tail`: the newest turn — the encoder's actual working material —
    gets the larger budget; older unencoded turns the smaller."""
    actions = []
    for episode in episodes:
        action = parse_action(episode)
        if action:
            actions.append(action)
    if policy.profile == 'full':
        return [_action_line(a, closing=i >= len(actions) - ACTIONS_KEEP_LAST)
                for i, a in enumerate(actions)]

    # The closing actions are the turn's outcome: split them off BEFORE
    # dedup so a final action that repeats an earlier one can never be
    # folded forward out of its outcome slot.
    closing = actions[-ACTIONS_KEEP_LAST:] if len(actions) > ACTIONS_KEEP_LAST \
        else actions
    body = actions[:-len(closing)] if closing is not actions else []
    # Balanced retains its historical summary-only identity for unstamped rows.
    # Thin must keep raw tools separate before grouping edit captions.
    body = _dedup(body, legacy_summary_identity=policy.profile == 'balanced')

    budget = ACTION_BUDGETS[policy.profile][bool(is_tail)]
    # Soft edge: an accounting line for one or two actions costs more than
    # it saves — only condense when the middle is worth a line.
    thin = policy.profile == 'thin'
    if not thin and len(body) + len(closing) <= budget + ACTIONS_BUDGET_SOFT_EDGE:
        return [_action_line(a) for a in body] + [_action_line(a, True) for a in closing]

    # Writes render regardless of budget; the budget's head slots go to the
    # leading regular actions; everything else rolls into the accounting
    # line. Thin groups the whole body; balanced keeps its relative order.
    head_slots = max(0, budget - len(closing))
    out, kept, mid, regular_kept = [], [], [], 0

    def flush_kept():
        out.extend(_group_edits(kept) if thin else map(_action_line, kept))
        kept.clear()

    def flush_mid():
        if mid:
            out.append(ActionLine(_rollup_line(mid, grouped=thin), sum(a.count for a in mid)))
            mid.clear()

    for a in body:
        keep = a.protected or (not (thin and a.routine) and regular_kept < head_slots)
        if keep:
            if not thin:
                flush_mid()
            kept.append(a)
            if not a.protected:
                regular_kept += 1
        else:
            if not thin:
                flush_kept()
            mid.append(a)
    flush_kept()
    # Closing boilerplate joins the turn's one rollup. Meaningful closing cues
    # stay separate even when a body action has exactly the same summary.
    if thin:
        mid.extend(a for a in closing if a.routine)
        closing = [a for a in closing if not a.routine]
    flush_mid()
    for a in closing:
        line = _action_line(a, True)
        out.append(ActionLine('Closing: ' + line.text, line.count, line.priority)
                   if thin else line)
    return out


def prepare_action_block(episodes, *, is_tail, encoded, view_policy, policy):
    episodes = filter_action_episodes(episodes, policy)
    if not episodes:
        return ActionBlock([])
    if encoded and view_policy:
        return ActionBlock([ActionLine(actions_stub_line(len(episodes)), len(episodes))], True)
    if view_policy:
        return ActionBlock(_condense_lines(episodes, is_tail, policy))
    # The old view may show raw multiline cues, but cannot bypass exclusion or
    # the final cap. Classification diagnostics still get allocation priority.
    lines = []
    for i, episode in enumerate(episodes):
        action = parse_action(episode)
        prepared = _action_line(action, i >= len(episodes) - ACTIONS_KEEP_LAST) if action else None
        diagnostic = prepared and prepared.priority == 3
        lines.append(ActionLine(prepared.text if diagnostic else str(episode.get('summary') or ''),
                                1, prepared.priority if prepared else 0))
    return ActionBlock(lines)


def _line_text(line):
    # XML normalizes CR and CRLF to LF. Count and render the same visible lines.
    return line.text.replace('\r\n', '\n').replace('\r', '\n')


def _block_xml(block, selected):
    if not selected:
        return ''
    if block.inline and len(selected) == 1:
        return '  <actions>%s</actions>\n' % escape(_line_text(selected[0]), quote=False)
    return ('  <actions>\n' + ''.join('    %s\n' % escape(_line_text(line), quote=False)
                                     for line in selected) + '  </actions>\n')


def render_action_blocks(blocks, policy):
    """Hard whole-timeline limits, including escaped UTF-8 and omission markup.

    Allocate structured records by priority and turn recency, preserving block order.
    Even a flood of protected edits or diagnostics must fit; one global notice
    accounts for omitted records across turns without spending a marker per turn.
    """
    full = [_block_xml(block, block.lines) for block in blocks]
    n_lines = sum(_line_text(line).count('\n') + 1 for b in blocks for line in b.lines)
    if n_lines <= policy.max_lines and sum(len(s.encode('utf-8')) for s in full) <= policy.max_bytes:
        return full, ''
    candidates = [(ti, li, line) for ti, block in enumerate(blocks)
                  for li, line in enumerate(block.lines)]
    total = sum(line.count for _, _, line in candidates)
    priority_total = sum(line.count for _, _, line in candidates if line.priority)

    def notice(count, turns, priority):
        return ('<action_limit>%d actions omitted across %d turns by the timeline '
                'action limit (%d priority actions); omissions are not absence '
                'of activity.</action_limit>\n' % (count, turns, priority))

    reserved = len(notice(total, len(blocks), priority_total).encode('utf-8'))
    remaining_bytes, remaining_lines = policy.max_bytes - reserved, policy.max_lines - 1
    selected = [{} for _ in blocks]
    for ti, li, line in sorted(candidates, key=lambda item: (item[2].priority, item[0], item[1]), reverse=True):
        # Charge a turn's wrapper once, and the exact incremental serialized cost.
        before = _block_xml(blocks[ti], list(selected[ti].values()))
        after = _block_xml(blocks[ti], list(selected[ti].values()) + [line])
        cost = len(after.encode('utf-8')) - len(before.encode('utf-8'))
        lines = _line_text(line).count('\n') + 1
        if cost <= remaining_bytes and lines <= remaining_lines:
            selected[ti][li] = line
            remaining_bytes -= cost
            remaining_lines -= lines
    omitted = [(ti, line) for ti, li, line in candidates if li not in selected[ti]]
    marker = notice(sum(line.count for _, line in omitted),
                    len({ti for ti, _ in omitted}),
                    sum(line.count for _, line in omitted if line.priority))
    rendered = [_block_xml(block, [line for _, line in sorted(kept.items())])
                for block, kept in zip(blocks, selected)]
    assert sum(len(s.encode('utf-8')) for s in rendered) + len(marker.encode('utf-8')) <= policy.max_bytes
    return rendered, marker
