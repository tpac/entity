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
- envelopes: the PROMPT HOOK, which consumes the class itself — a wake envelope
  is routed register_only with a short timeout before the daemon is called.
- identity: the hook OBSERVES (which tell env vars are present — names only),
  the daemon RESOLVES (resolve_host). The env is visible only in the hook
  process; the rule lives here.

ADDING A HOST — the complete recipe (a step skipped ships a host that looks
supported and classifies as unknown):
1. Entry below: identity.tells, tools (+ tool_patterns), matcher_aliases when
   the manifest matches names other than the ones the payload carries,
   envelopes, transcript, events (== the host's manifest), engine_events (the
   host's documented event list, dated by verified.host_version), blind,
   record, manifest, verified.
2. Manifest: hooks/hooks.<host>.json — tests/test_hooks_manifest_sync.py holds
   it to hooks.json AND to this entry (D4: declared, test-verified).
3. Run tests/test_host_contract.py and test_hooks_manifest_sync.py: validator,
   manifest parity, hook mirror and leaf pin all fail loudly on a partial entry.
4. Bump VOCAB_VERSION only when an existing host's tools/envelopes CHANGE
   meaning — rows stamp the version that classified them.

TWO DOORS: validate_host_contract() is the write door — the sync test refuses
a violating entry before merge, and Brain.__init__ logs every violation to
the errors table at boot (loud, non-fatal, no auto-heal). classify_tool /
resolve_host / envelope_policy are the read doors: an unknown name yields an
explicit 'unknown' status, never a guess and never a KeyError.
"""
import re

from servers.trace_contract import (ACTION_KINDS, ENVELOPE_EXTRACT_PREFIX,
                                    ENVELOPE_POLICIES)

# Bumped when an existing host's tools/envelopes change meaning (never on a
# host being added). Every stamped row carries the version that classified it.
VOCAB_VERSION = 1

# Explicit unresolved identity; never a contract key or a guessed host.
HOST_UNKNOWN = ''

TELL_STRENGTHS = ('strong', 'family')

# Every entry carries exactly these keys — the validator refuses more or fewer.
ENTRY_KEYS = ('identity', 'tools', 'tool_patterns', 'matcher_aliases', 'envelopes',
              'transcript', 'events', 'engine_events', 'blind', 'record',
              'manifest', 'verified')

# Pure extractors for `extract:<name>` envelope policies — closed registry:
# an envelope naming an unregistered extractor is a validator violation.
# Populated when envelope handling lands (design step 3).
EXTRACTORS = {}

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
        'envelopes': {},   # populated at design step 3 (system-reminder, task-notification)
        # The transcript is the JSONL the hook payload's transcript_path names;
        # hook_common.turn_model reads its last main-thread assistant entry.
        'transcript': {'grammar': 'claude_code_jsonl', 'readable': True},
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
        # Messages typed while a tool call is running are dropped by the host
        # and never reach UserPromptSubmit (brain node 4b8ed058).
        'blind': ('mid_tool_call_message',),
        'record': {'kind': 'transcript_jsonl'},
        'manifest': 'hooks/hooks.json',
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
        'envelopes': {},   # populated at design step 3 (desktop headers, question reply)
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
        # window, zero in S0 (measured by a Codex-side stream).
        'blind': ('hosted_web',),
        'record': {'kind': 'rollout'},
        'manifest': 'hooks/hooks.codex.json',
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


def envelope_policy(host, tag):
    """The declared policy for a host's envelope tag, or None when undeclared."""
    entry = HOST_CONTRACT.get(host)
    return entry['envelopes'].get(tag) if entry else None


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


def validate_host_contract(contract=None, extractors=None):
    """Every violation in the contract, as 'host: what'. Empty means clean.
    Pure; the sync test asserts empty and Brain.__init__ logs each entry."""
    contract = HOST_CONTRACT if contract is None else contract
    extractors = EXTRACTORS if extractors is None else extractors
    out = []
    seen_tells = {}
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
            say('envelopes must be a dict of tag → policy')
            envelopes = {}
        for tag, policy in envelopes.items():
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
        if not isinstance(entry['record'], dict) or not entry['record'].get('kind'):
            say('record must carry a kind')
        if not isinstance(entry['manifest'], str) or not entry['manifest'].endswith('.json'):
            say('manifest must be the hooks manifest path')
        if not isinstance(entry['verified'], dict) or not entry['verified'].get('host_version'):
            say('verified must carry the host_version the entry was read against')
    return out
