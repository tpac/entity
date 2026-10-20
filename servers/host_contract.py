"""Host contract — one declaration per harness, the brain's only knowledge of a
host's dialect.

One concern: everything the brain must know about a HARNESS (Claude Code,
Codex, a later Grok or local model) to translate what it observes into the
host-neutral vocabulary of trace_contract — which env tells identify it, what
its tool names mean, what its prompt envelopes are, which hook events it
registers, what it is known to be blind to, and which build the entry was
verified against. Consumers past the boundary read the OUTPUT vocabulary
(trace_contract.ACTION_KINDS, KIND_STATUS, HOST_STATUS, ENVELOPE_POLICIES)
and never a host's name; this is the only file where a NEW host literal may
appear — tests/test_host_shape_guardrail.py ratchets every other file's
remaining sites down, both ways, as the design's steps retire them. The ratchet
knows only names this contract declares; a genuinely new host name is caught by
the write door stamping kind_status 'unknown' into the errors table (step 1).

Three precedents, one shape: contract.PROMOTED_FIELDS (entries carry
behaviour, consumers derive), interaction_defaults (validators plus a content
fingerprint), aspects (a closed registry validated at boot — loud, no
auto-heal). Design: docs/HOST-CONTRACT-DESIGN.md.

A LEAF, by measurement: imports only trace_contract, itself a leaf with no path
to daemon_config (whose import-time fingerprint of servers/ costs ~22 ms). The
prompt hook may import this; the PostToolUse hook never does — see below.
Pinned by tests/test_host_contract.py::TestLeaf.

WHERE CLASSIFICATION RUNS (D9 — classify at the first consumer's boundary):
- tool kinds: the DAEMON, at the S0 write door, from the raw name the hook
  sent. The first consumer of a kind is the encoder hours later; classifying
  there is restart-deployable and every row from every client gets the same
  vocabulary on the same day. The hook stays a bare socket send.
- envelopes: the PROMPT HOOK, which consumes the class itself — read_prompt
  decides what of a prompt gets in, who each piece is from and what recall
  searches, before the daemon is called; the daemon writes one row per piece
  and never parses prompt text.
- reply_tools: the STOP HOOK — read_reply decides what of the assistant's
  turn gets in, from the host's reply text and the turn's closing tool calls,
  before the daemon is called; the daemon writes the reply it is sent.
- identity: the hook OBSERVES (which tell env vars are present — names only),
  the daemon RESOLVES (resolve_host). The env is visible only in the hook
  process; the rule lives here.
- turn: the PROMPT HOOK reads the host's own name for the turn from its
  payload (read_turn, the entry's turn_key); the daemon stamps it on the
  prompt's rows and never decides it. trace_contract.joins_open_turn reads it.

ADDING A HOST — the complete recipe (a step skipped ships a host that looks
supported and classifies as unknown):
1. Entry below: identity.tells, tools (+ tool_patterns), matcher_aliases when
   the manifest matches names other than the ones the payload carries,
   envelopes, repeating_commands (slash commands whose prompt recurs on its
   own, never recorded; () when none), transcript, events (== the host's
   manifest), engine_events (the host's documented event list, dated by
   verified.host_version), blind, turn_key (the hook-payload field naming
   the turn; None when the host names none), remote_hooks (env markers of a
   hook process on a machine with no route to the daemon; () when the host
   never runs hooks off the operator's machine),
   unattended (env markers of a hook in a session no person attends, whose
   prompts are not the operator's; () when the host gives none), record,
   manifest, adapter (the host's setup module in hooks/adapters/, or None),
   verified.
2. Manifest: hooks/hooks.<host>.json — tests/test_hooks_manifest_sync.py holds
   it to hooks.json AND to this entry (D4: declared, test-verified).
3. Run tests/test_host_contract.py and test_hooks_manifest_sync.py: validator,
   manifest parity, hook mirror and leaf pin all fail loudly on a partial entry.
4. Bump VOCAB_VERSION only when an existing host's tools/envelopes CHANGE
   meaning — rows stamp the version that classified them.

TWO DOORS: validate_host_contract() is the write door — the sync test refuses
a violating entry before merge, and Brain.__init__ logs every violation to
the errors table at boot (loud, non-fatal, no auto-heal). classify_tool /
resolve_host / envelope_policy / read_prompt / read_reply / read_turn are the
read doors: an unknown name yields an explicit 'unknown' status, never a guess
and never a KeyError.
"""
import re
import textwrap
from collections import namedtuple

from servers.trace_contract import (ACTION_KINDS, ENVELOPE_EXTRACT_PREFIX,
                                    ENVELOPE_POLICIES, SHORT_ANSWER_MAX_LEN,
                                    SPEAKER_REF_TYPES)

# Bumped when an existing host's tools/envelopes change meaning (never on a
# host being added). Every stamped row carries the version that classified it.
# 2: Claude Code prompts that 1 read as the operator's — the desktop app's
# wake-ups, briefs and session messages, an undeclared whole-prompt wrapper,
# an unattended session's text — are not; tool kinds are unchanged from 1.
VOCAB_VERSION = 2

# Explicit unresolved identity; never a contract key or a guessed host.
HOST_UNKNOWN = ''

TELL_STRENGTHS = ('strong', 'family')

# A host adapter is a module filename inside hooks/adapters/ — the one shape
# the validator accepts and the MCP proxy's loader will load.
ADAPTER_FILENAME = r'[a-z][a-z0-9_]*\.py'

# Every entry carries exactly these keys — the validator refuses more or fewer.
ENTRY_KEYS = ('identity', 'tools', 'tool_patterns', 'matcher_aliases', 'envelopes',
              'repeating_commands', 'transcript', 'events', 'engine_events', 'blind',
              'turn_key', 'remote_hooks', 'unattended', 'record', 'manifest', 'adapter',
              'verified')

# A declared command name: what follows the slash and any plugin prefix.
COMMAND_NAME = r'[a-z0-9][a-z0-9_-]*'

def _agent_handback(prompt, frame):
    """Every report in Claude Code's subagent hand-backs: each is a frame
    line, a one-line notice, then the report with every line indented by the
    harness. A report ends at the first line back at column zero, and only a
    frame line opens the next — the harness indents every report line, so a
    frame-like line at column zero inside a report would be forged. Returns
    the reports un-indented, in order, each stored and recalled as it reads;
    () when no frame is followed by an indented report."""
    reports, body = [], None
    for line in prompt.split('\n') + ['']:
        if body and line[:1] not in (' ', '\t'):
            reports.append(textwrap.dedent('\n'.join(body)).strip())
            body = None
        if line.startswith(frame):
            body = []
        elif body is not None and line[:1] in (' ', '\t'):
            body.append(line)
    return tuple((r, r) for r in reports if r)


# The desktop app's frames for the first prompt of a session another session
# started: a fresh session's brief carries the starting session's background
# before its task; an offered or forked one carries the task alone.
_BRIEF_WITH_BACKGROUND = 'Background (from the session that started this one):'
_BRIEF_TASK_ONLY = 'Task (from the session that started this one):'
_BRIEF_TASK_LINE = '\n\nTask:\n'


def _session_brief(prompt, header):
    """The brief another session wrote for this one, without the app's frame
    lines. Returns the task, then the background when the brief carries one —
    what this session was asked to do leads what recall reads, since a long
    background would otherwise fill the query before the task begins; () when
    the frame holds nothing. The app appends the task line after the
    background, so a background that quotes one is split at the last."""
    body = prompt[len(header):].strip()
    if header == _BRIEF_WITH_BACKGROUND:
        background, found, task = body.rpartition(_BRIEF_TASK_LINE)
        if found:
            return tuple((t, t) for t in (task.strip(), background.strip()) if t)
    return ((body, body),) if body else ()


def _session_messages(prompt, tag):
    """Every message another session sent, as Claude Code delivers them: one
    closed block each, back to back, the opening tag on its own line. Each is
    stored without its closing tag — the opening line names the sender — and
    recalled on the lines after it, what the sender wrote; a block without
    that shape is stored and recalled on nothing, which the daemon logs.
    Whatever follows the last block is the harness's note, stored as its own
    piece and never recalled; () when the prompt is empty."""
    found, close, rest = [], _closing_tag(tag), prompt
    while rest.startswith(tag) and _opens(rest, 0, tag):
        end = rest.find(close)
        if end < 0:
            break
        block = rest[:end].strip()
        head, newline, words = block.partition('\n')
        found.append((block, words.strip() if newline and head.endswith('>') else ''))
        rest = rest[end + len(close):].strip()
    if rest:
        found.append((rest, ''))
    return tuple(found)


# Pure extractors for `extract:<name>` envelope policies — closed registry:
# an envelope naming an unregistered extractor is a validator violation. Each
# takes the prompt from its envelope's line on and the envelope's tag, and
# returns what it reads out, in order, as (stored, recalled) pairs — one piece
# each, holding the stored text; recall reads the recalled texts together, and
# '' keeps a piece out of it; () when it finds nothing.
EXTRACTORS = {
    'agent_handback': _agent_handback,
    'session_brief': _session_brief,
    'session_messages': _session_messages,
}

# The MCP tool-name pattern is host-neutral in shape (dispatch_common owns
# which server is the brain's); both hosts prefix every MCP tool with it.
_MCP_PATTERN = (r'^mcp__', 'mcp')

HOST_CONTRACT = {
    'claude-code': {
        # The one strong tell: Claude Code exports its session id into every
        # hook process. Bare PLUGIN_ROOT is NOT a tell — resolve-brain-db.sh
        # exports it on both hosts.
        'identity': {'tells': ({'env': 'CLAUDE_CODE_SESSION_ID', 'strength': 'strong'},)},
        'tools': {
            'Bash': 'shell',
            'Edit': 'edit', 'Write': 'edit', 'NotebookEdit': 'edit',
            'Read': 'read', 'WebFetch': 'read',
            'Glob': 'search', 'Grep': 'search', 'WebSearch': 'search',
            'Agent': 'agent',
        },
        'tool_patterns': (_MCP_PATTERN,),
        'matcher_aliases': {},
        # Prompt envelopes — tag → (speaker, policy); read_prompt applies
        # them. A background task's completion, a scheduled task's run and
        # the desktop app's wake-ups (its side sessions settled, its Auto-fix
        # watch on a pull request) are the harness's whole prompt (the
        # notification arrives behind a "[SYSTEM NOTIFICATION …]" preamble
        # line, the scheduled task carries attributes). A reminder glued to
        # the operator's text is cut out into its own piece. A subagent's
        # hand-back is its report, framed and indented by the harness; the
        # piece is the report itself. Another session's words — the brief it
        # wrote when it started this one, a message it sent — are not the
        # operator's and are recalled on; they are 'harness' until the
        # session row (trace_contract, self_message) is turned on. The
        # operator's own attachments lead their words and stay theirs.
        'envelopes': {
            '<task-notification>': ('harness', 'machine'),
            '<scheduled-task': ('harness', 'machine'),
            '<side-sessions-event>': ('harness', 'machine'),
            '<ci-monitor-event>': ('harness', 'machine'),
            '<system-reminder>': ('harness', 'drop'),
            '<agent-message': ('subagent', 'extract:agent_handback'),
            '<cross-session-message': ('harness', 'extract:session_messages'),
            _BRIEF_WITH_BACKGROUND: ('harness', 'extract:session_brief'),
            _BRIEF_TASK_ONLY: ('harness', 'extract:session_brief'),
            '<artifact-view-context': ('operator', 'keep'),
            '<pasted_content': ('operator', 'keep'),
        },
        # A slash command reaches UserPromptSubmit only when it starts a model
        # turn: a skill or custom command arrives as the line typed, a built-in
        # like /model or /cost runs locally and fires no hook (measured
        # 2026-10-19, CLI 2.1.303). Each is the operator's prompt, except these,
        # whose line recurs on its own: a /loop iteration re-submits its /loop
        # line, a watch is re-armed by typing it again (brain node 7d615634).
        'repeating_commands': ('loop', 'watch'),
        # The transcript is the JSONL the hook payload's transcript_path names;
        # hook_common.closing_message reads the turn's closing main-thread
        # assistant message — its model, and the tool calls that close it.
        # reply_tools: a tool whose input is what the operator reads when the
        # turn ends → the input fields read_reply records, in order. The
        # desktop app's next-step box closes a turn with no text after it, so
        # last_assistant_message is empty there (brain node 9483a773).
        'transcript': {'grammar': 'claude_code_jsonl', 'readable': True,
                       'reply_tools': {'mcp__ccd_turn__wrap_up': ('recap', 'ask')}},
        'events': ('SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse',
                   'Stop', 'StopFailure', 'SessionEnd',
                   'WorktreeCreate', 'WorktreeRemove'),
        # code.claude.com/docs/en/hooks, read against verified.host_version.
        'engine_events': frozenset({
            'SessionStart', 'Setup', 'UserPromptSubmit', 'UserPromptExpansion',
            'PreToolUse', 'PermissionRequest', 'PermissionDenied', 'PostToolUse',
            'PostToolUseFailure', 'PostToolBatch', 'Notification', 'MessageDisplay',
            'SubagentStart', 'SubagentStop', 'TaskCreated', 'TaskCompleted', 'Stop',
            'StopFailure', 'TeammateIdle', 'InstructionsLoaded', 'ConfigChange',
            'CwdChanged', 'DirectoryAdded', 'FileChanged', 'WorktreeCreate',
            'WorktreeRemove', 'PreCompact', 'PostCompact', 'PreModelSwitch',
            'PostModelSwitch', 'Elicitation', 'ElicitationResult', 'SessionEnd',
        }),
        # Nothing known. A message typed mid-turn, tool call running or not,
        # reaches UserPromptSubmit when the turn absorbs it (measured
        # 2026-10-18: every one from CC 2.1.260 on, none before; brain node
        # 4b8ed058).
        'blind': (),
        # The hook-payload field naming the turn. Every hook of one turn
        # carries the same prompt_id, and a message the turn absorbs arrives
        # with the running turn's, not a new one; a new prompt gets a new one,
        # also after an interrupted turn, whose next prompt shares its s0
        # chain (probe 2026-10-19, CC 2.1.303; brain node a15cada5).
        'turn_key': 'prompt_id',
        # Hook processes that run where no daemon can be reached: Cowork runs
        # plugin command hooks in a cloud Linux VM with no route to the
        # operator's Mac, while the plugin's MCP server runs on the Mac (brain
        # node bc9e9e30). NOT an identity tell — the same process carries
        # CLAUDE_CODE_SESSION_ID, so a second tell would resolve every VM hook
        # 'ambiguous'. The hooks hold zero-import copies (hook_common,
        # resolve-brain-db.sh); they never treat macOS as such a machine.
        'remote_hooks': ({'env': 'CLAUDE_CODE_ENTRYPOINT', 'value': 'remote_cowork'},),
        # A session nobody sits at: Claude Code exports whether a person
        # attends it into every hook process — '0' for a `claude -p` run, an
        # SDK script or a background job, even one started from inside an
        # attended session ('1' there; measured 2026-10-16, CC 2.1.301).
        # CLAUDE_CODE_ENTRYPOINT is inherited from the parent, so it can't say.
        'unattended': ({'env': 'CLAUDE_CODE_SESSION_ATTENDED', 'value': '0'},),
        'record': {'kind': 'transcript_jsonl'},
        'manifest': 'hooks/hooks.json',
        'adapter': None,
        'verified': {'host_version': '2.1.263'},
    },
    'codex': {
        # No host-specific tell is guaranteed in a Codex hook process (brain
        # node 2f1ee97e): bare PLUGIN_DATA marks the CC-compatible-plugin-host
        # FAMILY (Codex sets Claude Code's aliases deliberately, so a fork
        # would too); CODEX_INTERNAL_ORIGINATOR_OVERRIDE is host-specific but
        # observed only on the Desktop app.
        'identity': {'tells': ({'env': 'PLUGIN_DATA', 'strength': 'family'},
                               {'env': 'CODEX_INTERNAL_ORIGINATOR_OVERRIDE', 'strength': 'strong'})},
        'tools': {
            'Bash': 'shell',
            'apply_patch': 'edit',
            'spawn_agent': 'agent',
        },
        'tool_patterns': (_MCP_PATTERN,),
        # Codex's hook engine matches Claude Code's names for its own tools;
        # the payload carries the real leaf name (tests/test_hooks_manifest_sync).
        'matcher_aliases': {'Edit': 'apply_patch', 'Write': 'apply_patch', 'Agent': 'spawn_agent'},
        # Codex sessions receive the same background-task completion envelope
        # (machine-woken Codex sessions carried task-notification rows, brain
        # node 9b462caa). No scheduled-task envelope observed on Codex. The
        # desktop attachment headers and the request boundary frame the
        # operator's own text; the question reply carries the operator's
        # answer beside the assistant's question — kept whole as the
        # operator's until its extractor has real samples to be written from.
        'envelopes': {
            '<task-notification>': ('harness', 'machine'),
            '# Files mentioned by the user:': ('operator', 'keep'),
            '# Files pasted by the user:': ('operator', 'keep'),
            '# Response annotations:': ('operator', 'keep'),
            '# Diff comments:': ('operator', 'keep'),
            '# Browser comments:': ('operator', 'keep'),
            '# Selected text:': ('operator', 'keep'),
            '# Failing PR checks:': ('operator', 'keep'),
            '## My request': ('operator', 'marker'),
            '<send_user_message_question_reply>': ('operator', 'keep'),
        },
        # The entity watch skill re-arms the same way on both hosts; Codex's
        # own slash commands are not probed.
        'repeating_commands': ('watch',),
        # The rollout nests turns as item_completed records; our transcript
        # readers match 0 rows on it (measured) — declared, not read.
        'transcript': {'grammar': 'codex_rollout', 'readable': False},
        'events': ('SessionStart', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse',
                   'Stop', 'SessionEnd'),
        # learn.chatgpt.com/docs/hooks, read against verified.host_version.
        'engine_events': frozenset({
            'SessionStart', 'SessionEnd', 'UserPromptSubmit', 'PreToolUse', 'PostToolUse',
            'PermissionRequest', 'PreCompact', 'PostCompact', 'SubagentStart',
            'SubagentStop', 'Stop', 'Interrupt',
        }),
        # Hosted web Extension calls never reach a hook: 4 in a 25 s rollout
        # window, zero in S0 (measured by a Codex-side session).
        'blind': ('hosted_web',),
        # Turn-scoped hook payloads carry turn_id (docs/CODEX-ADAPTER-RESEARCH.md).
        # A message typed into a running turn and an async-question answer both
        # stay inside that turn: no new turn starts, the prompt hook fires for
        # each, and the work around them carries the running turn_id (probe
        # 2026-10-19, Codex Desktop; brain node 0132ab77).
        'turn_key': 'turn_id',
        'remote_hooks': (),
        # No hook-process signal for an unattended run is known on Codex.
        'unattended': (),
        'record': {'kind': 'rollout'},
        'manifest': 'hooks/hooks.codex.json',
        # Setup and onboarding behaviour for this host (hooks/adapters/). The
        # MCP proxy loads it from the plugin manifest's --adapter argument, the
        # prompt hook from here; tests/test_hooks_manifest_sync.py holds the two
        # equal.
        'adapter': 'codex_setup.py',
        'verified': {'host_version': '0.153.4'},
    },
}

_PATTERNS = {host: tuple((re.compile(p), k) for p, k in entry['tool_patterns'])
             for host, entry in HOST_CONTRACT.items()}


def all_tell_env_vars(contract=None):
    """Every env var name any host's identity rests on — the probe list a hook
    reports presence for (names only, never values)."""
    contract = HOST_CONTRACT if contract is None else contract
    return tuple(sorted({t['env'] for e in contract.values() for t in e['identity']['tells']}))


def all_remote_hook_markers(contract=None):
    """Every (env, value) pair any host declares for a hook process on a
    machine with no route to the daemon — the list the hooks mirror."""
    contract = HOST_CONTRACT if contract is None else contract
    return tuple(sorted({(m['env'], m['value']) for e in contract.values()
                         for m in e['remote_hooks']}))


def resolve_host(present, contract=None):
    """Which harness a hook process ran under, from the tells it saw.

    `present` maps env var name → truthy when the hook observed it set.
    Returns (host, status, fired): status is one of trace_contract.HOST_STATUS
    minus 'legacy' — 'strong' when a host-specific tell fired, 'family' when
    only a family tell did, 'ambiguous' when tells of more than one host fired
    (host is '' — never pick one), 'unknown' when none did."""
    contract = HOST_CONTRACT if contract is None else contract
    fired = {}
    for host, entry in contract.items():
        hit = tuple(t['env'] for t in entry['identity']['tells'] if present.get(t['env']))
        if hit:
            fired[host] = hit
    if not fired:
        return HOST_UNKNOWN, 'unknown', ()
    if len(fired) > 1:
        return HOST_UNKNOWN, 'ambiguous', tuple(sorted(e for hit in fired.values() for e in hit))
    (host, hit), = fired.items()
    strength = {t['env']: t['strength'] for t in contract[host]['identity']['tells']}
    status = 'strong' if any(strength[e] == 'strong' for e in hit) else 'family'
    return host, status, hit


def classify_tool(host, tool_name):
    """(kind, status) for a host's raw tool name: kind ∈ ACTION_KINDS with
    status 'ok', or ('', 'unknown') — unknown host, empty name, or a name the
    host's map and patterns do not know. Never a guess."""
    entry = HOST_CONTRACT.get(host)
    name = str(tool_name or '')
    if entry is None or not name:
        return '', 'unknown'
    kind = entry['tools'].get(name)
    if kind:
        return kind, 'ok'
    for pattern, pattern_kind in _PATTERNS.get(host, ()):
        if pattern.match(name):        # patterns anchor at the start of the name
            return pattern_kind, 'ok'
    return '', 'unknown'


def _entries(host):
    """The entries a host's prompts are read against: its own, or every
    host's when the hook could not tell which harness it is on — a harness
    prompt must not be recorded as the operator's for want of a name."""
    return [HOST_CONTRACT[host]] if host in HOST_CONTRACT else list(HOST_CONTRACT.values())


def envelope_policy(host, tag):
    """The declared (speaker, policy) for a host's envelope tag, or None when
    undeclared. An unknown or ambiguous host reads every host's table, as
    read_prompt does."""
    for entry in _entries(host):
        if tag in entry['envelopes']:
            return entry['envelopes'][tag]
    return None


def session_unattended(host, env):
    """True when the host marks this hook's session as one no person attends
    (`unattended`), so its prompts are not the operator's. `env` is the hook
    process's environment."""
    return any(env.get(m['env']) == m['value']
               for entry in _entries(host) for m in entry['unattended'])


# A shell escape is the host's own and is not recorded at all. A slash command
# that reaches the hook is the operator's prompt, unless the host declares it
# repeating (repeating_commands) or it is the session's last command again.
SHELL_PREFIX = '!'
_COMMAND = re.compile(r'/([\w:-]+)(?=\s|$)')


def _command_name(text):
    """The name of the slash command `text` opens with, or '' when it opens
    with none (a path like /etc/hosts is not one). Named wide, so the same
    command is one name whatever its args, plugin prefix or case:
    `/Entity:Watch now` → watch."""
    m = _COMMAND.match(text)
    return m.group(1).rsplit(':', 1)[-1].lower() if m else ''

# What the prompt reader returns. `pieces` — (speaker, text, tag) in the order
# they are written: the piece that opens the turn first (the operator's text,
# a whole-prompt envelope, or the first text its extractor read out of it),
# then the rest, which ride in that turn — further extracted texts, then the
# blocks cut from the front of the prompt; tag is the envelope that declared
# the piece, the undeclared wrapper's own tag, or '' for text no envelope
# wrapped. `recall_text` — what recall and the surface read; '' records the
# turn without recalling on it. No pieces means nothing is recorded.
# `command` — the name of the slash command the prompt opens with
# (_command_name), '' when none: the hook keeps it as the session's last
# command for the next prompt's read.
Intake = namedtuple('Intake', ('pieces', 'recall_text', 'command'), defaults=('',))

_OPERATOR = 'operator'
_HARNESS = 'harness'


def _line_start(text, tag):
    """Index where `tag` opens a line of `text`, or -1. Line-start, not
    substring: the harness emits its tags at the head of a line (after its
    notice preamble for notifications), while an operator naming the tag
    mid-sentence is talking ABOUT it. An open tag ('<name', attributes to
    follow) matches only that name, never a longer one that begins with it."""
    at = 0 if text.startswith(tag) else _next_line_start(text, tag, 0)
    while at >= 0 and not _opens(text, at, tag):
        at = _next_line_start(text, tag, at + 1)
    return at


def _next_line_start(text, tag, start):
    at = text.find('\n' + tag, start)
    return at + 1 if at >= 0 else -1


def _opens(text, at, tag):
    """True when `tag` at `at` is the whole tag it declares: a text header or
    a closed tag always is; an open tag ('<name') only when its name ends
    there. A malformed tag of the declared name still is, so its extractor
    fails on it loudly instead of the operator's text absorbing it."""
    if not tag.startswith('<') or tag.endswith('>'):
        return True
    after = text[at + len(tag):at + len(tag) + 1]
    return not (after.isalnum() or after in ('_', '-'))


def _harness_preamble(text):
    """True when `text` — what precedes a whole-prompt envelope — could be the
    harness's: nothing, or a notice whose first line is bracketed ("[SYSTEM
    NOTIFICATION - NOT USER INPUT]", which may run on for a few lines).
    Anything else is the operator writing, with the envelope quoted inside
    their prompt."""
    first = text.strip().split('\n', 1)[0].strip()
    return not first or (first.startswith('[') and first.endswith(']'))


def _tag_name(tag):
    return tag.lstrip('<').rstrip('>').split()[0]


def _closing_tag(tag):
    return '</%s>' % _tag_name(tag)


_TAG_OPEN = re.compile(r'<([A-Za-z][\w-]*)[\s>]')


def _undeclared_wrapper(text, declared):
    """The tag of an undeclared block that is the whole of `text`, or ''.
    Every wrapper the harness sends as a prompt of its own opens the prompt
    and closes at its end, while the operator's own attachments are followed
    by their words (the 2026-10-16 census) — so only the first shape is taken
    from the operator, and only when no envelope declares the tag."""
    m = _TAG_OPEN.match(text)
    if not m or not text.endswith('</%s>' % m.group(1)):
        return ''
    if any(t.startswith('<') and _tag_name(t) == m.group(1) for t in declared):
        return ''
    return '<%s>' % m.group(1)


def _cut_leading(text, dropped):
    """Cut the `drop` envelopes' blocks off the FRONT of `text` — where the
    harness glues them (the 2026-10-05..11 census: every reminder led its
    prompt). A tag further in is the operator writing about it, and an
    unclosed block is not a block; both stay. Returns (rest, cut pieces)."""
    cut = []
    while True:
        text = text.lstrip()
        for tag, speaker in dropped:
            if text.startswith(tag):
                close = _closing_tag(tag)
                end = text.find(close)
                if end >= 0:
                    inner = text[text.index('>') + 1:end].strip()
                    if inner:
                        cut.append((speaker, inner, tag))
                    text = text[end + len(close):]
                    break
        else:
            return text, cut


def read_prompt(prompt, host=HOST_UNKNOWN, unattended=False, last_command=''):
    """What of `prompt` gets in, who each piece is from, and what recall
    reads — the one decision site for every incoming prompt (the prompt
    hook calls it; the daemon writes what it returns). With a resolved host
    only that host's envelopes are read; an unknown or ambiguous host reads
    every host's — a harness prompt must not be recorded as the operator's
    because the hook could not tell which harness it is on.

    First the `drop` envelopes' blocks are cut off the front of the prompt
    into pieces of their own; a shell escape behind them is still not
    recorded, and neither is a slash command the host declares repeating or
    that is `last_command` again — the session's last command, with no words
    of the operator's since (a /loop or cron job firing the same command).
    Then a `machine` or `extract:` envelope opening what
    remains, after at most the harness's notice preamble, is its speaker's;
    the preamble is the harness's frame and is left out. A `machine`
    envelope is one piece, recalled on nothing. An `extract:` envelope gives
    one piece per text its extractor reads out of it, and recall reads the
    texts it marks for recall together — or the envelope itself when the
    extractor finds nothing, so nothing is lost (the daemon logs it). A block no envelope declares that
    is the whole of what remains is a wrapper the contract has not met yet:
    the harness's, recalled on nothing, its tag left on the piece for the
    daemon to log. Otherwise the rest is the operator's text, recalled on
    unless it is a short answer — or, in a session no person attends
    (`unattended`, session_unattended), the harness's, recalled on nothing."""
    if not prompt or not prompt.strip():
        return Intake((), '')
    declared = {}
    for entry in _entries(host):
        for tag, decl in entry['envelopes'].items():
            declared.setdefault(tag, decl)
    text, cut = _cut_leading(prompt, [(tag, speaker) for tag, (speaker, policy)
                                      in declared.items() if policy == 'drop'])
    text = text.strip()
    if text.startswith(SHELL_PREFIX):
        return Intake((), '')
    # A command behind a reminder the harness glued on is still a command.
    command = _command_name(text)
    if command and (command == last_command or
                    any(command in e['repeating_commands'] for e in _entries(host))):
        return Intake((), '', command)
    for tag, (speaker, policy) in declared.items():
        if policy != 'machine' and not policy.startswith(ENVELOPE_EXTRACT_PREFIX):
            continue
        at = _line_start(text, tag)
        if at < 0 or not _harness_preamble(text[:at]):
            continue
        envelope = text[at:]
        if policy == 'machine':
            return Intake(((speaker, envelope, tag),) + tuple(cut), '')
        found = EXTRACTORS[policy[len(ENVELOPE_EXTRACT_PREFIX):]](envelope, tag)
        pieces = tuple((speaker, stored, tag) for stored, _ in found) or ((speaker, envelope, tag),)
        return Intake(pieces + tuple(cut), '\n\n'.join(r for _, r in found if r))
    if not text:
        return Intake(tuple(cut), '')
    wrapper = _undeclared_wrapper(text, declared)
    if wrapper or unattended:
        return Intake(((_HARNESS, text, wrapper),) + tuple(cut), '', command)
    recall_text = text if len(text) >= SHORT_ANSWER_MAX_LEN else ''
    return Intake(((_OPERATOR, text, ''),) + tuple(cut), recall_text, command)


def reply_tools(host=HOST_UNKNOWN):
    """The declared reply tools (tool name → input fields) of every readable
    transcript the hook could be on — an unknown or ambiguous host reads
    every host's, as read_prompt does. Empty means the Stop hook has no
    transcript to read for the reply."""
    found = {}
    for entry in _entries(host):
        tr = entry['transcript']
        if tr['readable']:
            for name, fields in tr.get('reply_tools', {}).items():
                found.setdefault(name, fields)
    return found


def read_reply(text, closing_calls=(), host=HOST_UNKNOWN):
    """What of the assistant's turn gets in — the reply side's one decision
    site, as read_prompt is the prompt's (the Stop hook calls it; the daemon
    writes what it returns). `text` is the host's own reply text (Claude
    Code's last_assistant_message: the text of the turn's closing message);
    `closing_calls` the (name, input) tool calls of that same closing message,
    read from the transcript. The reply is the text, then each declared field
    of each declared reply tool among the calls, in order — what the operator
    read when the turn ended. A closing message with no reply tool records
    the host's text verbatim."""
    tools = reply_tools(host)
    shown = []
    for name, args in closing_calls or ():
        for field in tools.get(name, ()):
            value = args.get(field) if isinstance(args, dict) else None
            if isinstance(value, str) and value.strip():
                shown.append(value.strip())
    if not shown:
        return text or ''
    return '\n\n'.join(([text.rstrip()] if text and text.strip() else []) + shown)


def read_turn(payload, host=HOST_UNKNOWN):
    """The host's own name for the turn a hook ran in: the payload field its
    entry declares (`turn_key`), or '' when it declares none or the payload
    lacks it. An unknown or ambiguous host reads every host's field, as
    read_prompt does — each host's payload carries only its own."""
    for entry in _entries(host):
        key = entry['turn_key']
        value = payload.get(key) if key and isinstance(payload, dict) else None
        if isinstance(value, str) and value:
            return value
    return ''


_FINGERPRINT = None


def contract_fingerprint():
    """12-hex content identity of THIS module — entries, resolver and
    extractors alike, so a changed rule is a changed identity even when no
    registry name moved (design D6). Stamped on every normalized row, so it is
    computed once per process. The output vocabularies live in trace_contract
    and are not hashed here: a change to what a kind MEANS is VOCAB_VERSION's
    lever, a change to how a name is classified is this one's."""
    global _FINGERPRINT
    if _FINGERPRINT is None:
        import hashlib
        with open(__file__, 'rb') as f:
            _FINGERPRINT = hashlib.sha256(f.read()).hexdigest()[:12]
    return _FINGERPRINT


def _env_markers(markers):
    """True for a tuple of {env, value} markers with non-empty strings."""
    return isinstance(markers, tuple) and all(
        isinstance(m, dict) and set(m) == {'env', 'value'}
        and all(isinstance(m[k], str) and m[k] for k in m) for m in markers)


def validate_host_contract(contract=None, extractors=None):
    """Every violation in the contract, as 'host: what'. Empty means clean.
    Pure; the sync test asserts empty and Brain.__init__ logs each entry."""
    contract = HOST_CONTRACT if contract is None else contract
    extractors = EXTRACTORS if extractors is None else extractors
    out = []
    seen_tells = {}
    # Every host's tells, up front: a remote-hook marker on any of them is the
    # ambiguity trap the field exists to avoid, whichever host declares it.
    all_tells = {t.get('env') for e in contract.values() if isinstance(e, dict)
                 and isinstance(e.get('identity'), dict)
                 for t in (e['identity'].get('tells') or ()) if isinstance(t, dict)}
    for host, entry in contract.items():
        say = lambda msg, host=host: out.append('%s: %s' % (host, msg))
        if not isinstance(host, str) or not host:
            say('host key must be a non-empty string')
        if not isinstance(entry, dict):
            say('entry must be a dict')
            continue
        missing = [k for k in ENTRY_KEYS if k not in entry]
        extra = [k for k in entry if k not in ENTRY_KEYS]
        if missing:
            say('missing keys %s' % missing)
        if extra:
            say('unknown keys %s' % extra)
        if missing:
            continue
        tells = entry['identity'].get('tells') if isinstance(entry['identity'], dict) else None
        if not tells:
            say('identity.tells must name at least one env tell')
        else:
            for t in tells:
                env, strength = t.get('env'), t.get('strength')
                if not env or not isinstance(env, str):
                    say('identity tell without an env name: %r' % (t,))
                elif env in seen_tells:
                    # Across hosts it is always ambiguous; within one host the
                    # resolver's last-wins strength map would silently re-weight it.
                    who = 'this host' if seen_tells[env] == host else seen_tells[env]
                    say('tell %s is declared twice (%s)' % (env, who))
                else:
                    seen_tells[env] = host
                if strength not in TELL_STRENGTHS:
                    say('tell %s strength %r not in %s' % (env, strength, TELL_STRENGTHS))
        tools = entry['tools']
        if not tools or not isinstance(tools, dict):
            say('tools must be a non-empty dict')
            tools = {}
        for name, kind in tools.items():
            if kind not in ACTION_KINDS:
                say('tool %r has kind %r not in ACTION_KINDS %s' % (name, kind, ACTION_KINDS))
        patterns = entry['tool_patterns']
        if not isinstance(patterns, (tuple, list)) or not all(
                isinstance(p, (tuple, list)) and len(p) == 2 for p in patterns):
            say('tool_patterns must be a tuple of (pattern, kind) pairs')
            patterns = ()
        for pat, kind in patterns:
            try:
                re.compile(pat)
            except (re.error, TypeError) as e:
                say('tool_pattern %r does not compile: %s' % (pat, e))
            if kind not in ACTION_KINDS:
                say('tool_pattern %r has kind %r not in ACTION_KINDS' % (pat, kind))
        aliases = entry['matcher_aliases']
        if not isinstance(aliases, dict):
            say('matcher_aliases must be a dict of alias → declared tool')
            aliases = {}
        for alias, target in aliases.items():
            if target not in tools:
                say('matcher alias %r → %r, which is not one of its tools' % (alias, target))
            if alias in tools:
                say('matcher alias %r is also a declared tool' % alias)
        envelopes = entry['envelopes']
        if not isinstance(envelopes, dict):
            say('envelopes must be a dict of tag → (speaker, policy)')
            envelopes = {}
        for tag, decl in envelopes.items():
            if not (isinstance(decl, tuple) and len(decl) == 2):
                say('envelope %r must declare (speaker, policy), got %r' % (tag, decl))
                continue
            speaker, policy = decl
            if speaker not in SPEAKER_REF_TYPES:
                say('envelope %r speaker %r not in %s' % (tag, speaker, sorted(SPEAKER_REF_TYPES)))
            if policy == 'drop' and not (isinstance(tag, str) and tag.startswith('<')):
                say('envelope %r is dropped as a block, so it must be a <tag>' % tag)
            if policy in ENVELOPE_POLICIES:
                continue
            if isinstance(policy, str) and policy.startswith(ENVELOPE_EXTRACT_PREFIX):
                name = policy[len(ENVELOPE_EXTRACT_PREFIX):]
                if name not in extractors:
                    say('envelope %r names extractor %r, not registered in EXTRACTORS' % (tag, name))
                continue
            say('envelope %r policy %r not in %s or %s<name>' % (tag, policy, ENVELOPE_POLICIES, ENVELOPE_EXTRACT_PREFIX))
        tr = entry['transcript']
        if not isinstance(tr, dict) or not tr.get('grammar') or not isinstance(tr.get('readable'), bool):
            say('transcript must carry a grammar name and a readable bool')
        else:
            replies = tr.get('reply_tools', {})
            if not (isinstance(replies, dict) and all(
                    isinstance(name, str) and name and isinstance(fields, tuple) and fields
                    and all(isinstance(f, str) and f for f in fields)
                    for name, fields in replies.items())):
                say('transcript reply_tools must map a tool name to a tuple of its input fields')
            elif replies and not tr['readable']:
                say('transcript reply_tools are read from the transcript, so it must be readable')
        events, engine = entry['events'], entry['engine_events']
        if not isinstance(events, (tuple, list)) or not events \
                or not all(isinstance(e, str) for e in events):
            say('events must be a non-empty tuple of the manifest\'s registered event names')
            events = ()
        if not isinstance(engine, (set, frozenset)) or not engine \
                or not all(isinstance(e, str) for e in engine):
            say('engine_events must be a non-empty set of the host\'s documented events')
            engine = frozenset()
        unsupported = sorted(set(events) - set(engine)) if events and engine else []
        if unsupported:
            say('events %s are not in engine_events' % unsupported)
        if not isinstance(entry['blind'], tuple) or not all(isinstance(b, str) for b in entry['blind']):
            say('blind must be a tuple of family names')
        turn_key = entry['turn_key']
        if turn_key is not None and not (isinstance(turn_key, str) and turn_key):
            say('turn_key must be None or the payload field naming the host\'s turn')
        remote = entry['remote_hooks']
        if not _env_markers(remote):
            say('remote_hooks must be a tuple of {env, value} markers with non-empty strings')
        else:
            for m in remote:
                if m['env'] in all_tells:
                    say('remote_hooks marker %s is an identity tell — a hook carrying both '
                        'would resolve ambiguous' % m['env'])
        if not _env_markers(entry['unattended']):
            say('unattended must be a tuple of {env, value} markers with non-empty strings')
        repeating = entry['repeating_commands']
        if not isinstance(repeating, tuple) or not all(
                isinstance(n, str) and re.fullmatch(COMMAND_NAME, n) for n in repeating):
            say('repeating_commands must be a tuple of command names (no slash or plugin prefix)')
        if not isinstance(entry['record'], dict) or not entry['record'].get('kind'):
            say('record must carry a kind')
        if not isinstance(entry['manifest'], str) or not entry['manifest'].endswith('.json'):
            say('manifest must be the hooks manifest path')
        adapter = entry['adapter']
        if adapter is not None and not (isinstance(adapter, str) and re.fullmatch(ADAPTER_FILENAME, adapter)):
            say('adapter must be None or a module filename in hooks/adapters/')
        if not isinstance(entry['verified'], dict) or not entry['verified'].get('host_version'):
            say('verified must carry the host_version the entry was read against')
    return out
