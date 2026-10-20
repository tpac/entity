"""Contract tests for servers/host_contract.py — the per-harness declarations.

What this holds (design: docs/HOST-CONTRACT-DESIGN.md, drift ledger):
  • the shipped contract validates clean, and the validator has teeth (every
    class of malformed entry is refused with the host named);
  • the read doors never guess: resolve_host / classify_tool yield explicit
    'unknown' / 'ambiguous' statuses drawn from trace_contract's vocabularies;
  • HOOK MIRRORS — the two places a hook still carries host shape are held
    to the contract: post_tool_trace._build_summary's tool-name branches and
    hook_common.host_tells' env tells must be names the contract declares;
  • LEAF — importing the contract reaches neither daemon_config nor brain
    (the same subprocess pin as test_caller_stamp's hook-path test);
  • the tool_result metadata shape and builder in trace_contract, which the
    daemon's write door uses to stamp what the contract classified;
  • read_reply — what of the assistant's turn gets in — and the Stop hook
    that runs it on the turn's closing message.
Manifest ↔ contract parity (D4) lives in tests/test_hooks_manifest_sync.py,
beside the manifest ↔ manifest parity it extends.

Run: ./dev pytest tests/test_host_contract.py -v
"""
import ast
import copy
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers import host_contract as hc  # noqa: E402
from servers.trace_contract import (  # noqa: E402
    ACTION_KINDS, ENVELOPE_POLICIES, HOST_STATUS, KIND_STATUS, SPEAKER_REF_TYPES,
    METADATA_REQUIRED_BY_REF_TYPE, TOOL_RESULT_METADATA_SHAPE,
    TOOL_RESULT_NORMALIZATION_KEYS, build_tool_result_metadata,
    validate_trace_metadata)

_HOOKS = os.path.join(ROOT, 'hooks', 'scripts')


def _faulted(mutate):
    """A deep copy of the shipped contract with one fault applied."""
    c = copy.deepcopy(hc.HOST_CONTRACT)
    mutate(c)
    return c


class TestContractIsClean(unittest.TestCase):
    def test_shipped_contract_validates(self):
        self.assertEqual(hc.validate_host_contract(), [])

    def test_every_entry_carries_exactly_the_contract_keys(self):
        for host, entry in hc.HOST_CONTRACT.items():
            self.assertEqual(tuple(sorted(entry)), tuple(sorted(hc.ENTRY_KEYS)), host)

    def test_vocab_version_is_a_positive_int(self):
        self.assertIsInstance(hc.VOCAB_VERSION, int)
        self.assertGreaterEqual(hc.VOCAB_VERSION, 1)

    def test_statuses_come_from_the_output_vocabulary(self):
        # The read doors emit trace_contract's words, never their own.
        for present, want in (({}, 'unknown'),
                              ({'CLAUDE_CODE_SESSION_ID': True, 'PLUGIN_DATA': True}, 'ambiguous'),
                              ({'CLAUDE_CODE_SESSION_ID': True}, 'strong'),
                              ({'PLUGIN_DATA': True}, 'family')):
            self.assertIn(hc.resolve_host(present)[1], HOST_STATUS)
            self.assertEqual(hc.resolve_host(present)[1], want)
        self.assertIn(hc.classify_tool('codex', 'nope')[1], KIND_STATUS)


class TestValidatorHasTeeth(unittest.TestCase):
    def _refuses(self, mutate, needle):
        out = hc.validate_host_contract(_faulted(mutate))
        self.assertTrue(out, 'validator accepted a faulted contract (%s)' % needle)
        self.assertTrue(any(needle in v for v in out), '%r not in %s' % (needle, out))
        self.assertTrue(all(v.split(':')[0] in hc.HOST_CONTRACT for v in out),
                        'every violation names its host: %s' % out)

    def test_typoed_kind(self):
        self._refuses(lambda c: c['codex']['tools'].__setitem__('apply_patch', 'edt'), 'ACTION_KINDS')

    def test_alias_to_undeclared_tool(self):
        self._refuses(lambda c: c['codex']['matcher_aliases'].__setitem__('Edit', 'patch'), 'not one of its tools')

    def test_alias_shadowing_a_tool(self):
        self._refuses(lambda c: c['codex']['matcher_aliases'].__setitem__('Bash', 'apply_patch'), 'also a declared tool')

    def test_unregistered_extractor(self):
        self._refuses(lambda c: c['codex']['envelopes'].__setitem__('<x>', ('subagent', 'extract:nope')),
                      'not registered')

    def test_bogus_envelope_policy(self):
        self._refuses(lambda c: c['claude-code']['envelopes'].__setitem__('<x>', ('harness', 'ignore')), 'policy')

    def test_envelope_without_a_speaker(self):
        self._refuses(lambda c: c['claude-code']['envelopes'].__setitem__('<x>', 'machine'), '(speaker, policy)')

    def test_envelope_speaker_without_a_row(self):
        self._refuses(lambda c: c['claude-code']['envelopes'].__setitem__('<x>', ('stranger', 'machine')),
                      'speaker')

    def test_envelope_speaker_whose_row_is_off(self):
        # A dial-off row would leave my reply grouped under the previous turn.
        self._refuses(lambda c: c['claude-code']['envelopes'].__setitem__('<x>', ('session', 'machine')),
                      'speaker')

    def test_dropped_envelope_must_be_a_tag(self):
        self._refuses(lambda c: c['codex']['envelopes'].__setitem__('# Header:', ('harness', 'drop')),
                      'must be a <tag>')

    def test_event_outside_engine(self):
        self._refuses(lambda c: c['codex'].__setitem__('events', c['codex']['events'] + ('WorktreeCreate',)),
                      'not in engine_events')

    def test_missing_and_extra_keys(self):
        self._refuses(lambda c: c['codex'].pop('record'), 'missing keys')
        self._refuses(lambda c: c['codex'].__setitem__('nesting', {}), 'unknown keys')

    def test_shared_tell_is_refused(self):
        self._refuses(lambda c: c['codex']['identity'].__setitem__(
            'tells', ({'env': 'CLAUDE_CODE_SESSION_ID', 'strength': 'strong'},)), 'declared twice (claude-code)')

    def test_bogus_tell_strength(self):
        self._refuses(lambda c: c['codex']['identity'].__setitem__(
            'tells', ({'env': 'PLUGIN_DATA', 'strength': 'maybe'},)), 'strength')

    def test_remote_marker_that_is_a_tell_is_refused(self):
        # The ambiguity trap the field exists to avoid: a hook carrying both
        # this marker and another host's tell would resolve 'ambiguous'.
        self._refuses(lambda c: c['codex'].__setitem__(
            'remote_hooks', ({'env': 'CLAUDE_CODE_SESSION_ID', 'value': 'x'},)), 'is an identity tell')

    def test_malformed_remote_hooks(self):
        self._refuses(lambda c: c['codex'].__setitem__('remote_hooks', [{'env': 'A', 'value': 'b'}]),
                      'remote_hooks must be')
        self._refuses(lambda c: c['codex'].__setitem__('remote_hooks', ({'env': 'A'},)),
                      'remote_hooks must be')
        self._refuses(lambda c: c['codex'].__setitem__('remote_hooks', ({'env': 'A', 'value': ''},)),
                      'remote_hooks must be')

    def test_malformed_unattended(self):
        for bad in (({'env': 'X'},), [{'env': 'X', 'value': '0'}], ({'env': 'X', 'value': ''},)):
            with self.subTest(bad=bad):
                self._refuses(lambda c, bad=bad: c['codex'].__setitem__('unattended', bad),
                              'unattended must be')

    def test_malformed_repeating_commands(self):
        # A name is what follows the slash and any plugin prefix: '/watch' or
        # 'entity:watch' would never match a typed command.
        for bad in (['watch'], ('/watch',), ('entity:watch',), ('',), ('two words',)):
            with self.subTest(bad=bad):
                self._refuses(lambda c, bad=bad: c['claude-code'].__setitem__('repeating_commands', bad),
                              'repeating_commands must be')

    def test_adapter_is_none_or_a_module_filename(self):
        for bad in ('codex_setup', '../codex_setup.py', 'hooks/adapters/codex_setup.py', ''):
            self._refuses(lambda c, bad=bad: c['codex'].__setitem__('adapter', bad), 'adapter must be')

    def test_unverified_entry(self):
        self._refuses(lambda c: c['codex'].__setitem__('verified', {}), 'host_version')

    def test_turn_key_is_none_or_a_field_name(self):
        for bad in ('', ('prompt_id',), 5):
            with self.subTest(bad=bad):
                self._refuses(lambda c, bad=bad: c['codex'].__setitem__('turn_key', bad),
                              'turn_key must be')

    def test_malformed_reply_tools(self):
        for bad in (['wrap_up'], {'wrap_up': 'recap'}, {'wrap_up': ()}, {'': ('recap',)},
                    {'wrap_up': ('recap', '')}):
            with self.subTest(bad=bad):
                self._refuses(lambda c, bad=bad: c['claude-code']['transcript'].__setitem__(
                    'reply_tools', bad), 'reply_tools must map')

    def test_reply_tools_need_a_readable_transcript(self):
        # Codex's rollout is declared unreadable: a reply tool there could
        # never be read, so declaring one would be a promise the hook breaks.
        self._refuses(lambda c: c['codex']['transcript'].__setitem__(
            'reply_tools', {'request_user_input': ('question',)}), 'must be readable')

    def test_duplicate_tell_within_one_host(self):
        # resolve_host's strength map is last-wins: a family tell redeclared as
        # strong would silently promote every row. Must be refused.
        self._refuses(lambda c: c['codex']['identity'].__setitem__(
            'tells', ({'env': 'PLUGIN_DATA', 'strength': 'family'},
                      {'env': 'PLUGIN_DATA', 'strength': 'strong'})), 'declared twice')

    def test_malformed_shapes_are_reported_not_raised(self):
        # The write door must REPORT: a raise here would be swallowed by the
        # boot path's guard and write nothing to the errors table.
        self._refuses(lambda c: c['codex'].__setitem__('matcher_aliases', ['Edit']), 'matcher_aliases must be a dict')
        self._refuses(lambda c: c['codex'].__setitem__('envelopes', ['<x>']), 'envelopes must be a dict')
        self._refuses(lambda c: c['codex'].__setitem__('tool_patterns', ('^mcp__',)), 'tool_patterns must be')
        self._refuses(lambda c: c['codex'].__setitem__('tool_patterns', None), 'tool_patterns must be')
        self._refuses(lambda c: c['codex'].__setitem__('tools', []), 'tools must be')
        self._refuses(lambda c: c['codex'].__setitem__('events', ('Stop', 3)), 'events must be')
        self._refuses(lambda c: c['codex'].__setitem__('engine_events', ['Stop']), 'engine_events must be')

    def test_tell_strengths_are_host_statuses(self):
        # resolve_host reports a fired tell's strength AS the host status, so a
        # strength the output vocabulary does not know would be reported wrong.
        self.assertTrue(set(hc.TELL_STRENGTHS) <= set(HOST_STATUS),
                        '%s not all in HOST_STATUS %s' % (hc.TELL_STRENGTHS, HOST_STATUS))


class TestResolveHost(unittest.TestCase):
    def test_strong_tell_identifies(self):
        self.assertEqual(hc.resolve_host({'CLAUDE_CODE_SESSION_ID': 'abc'}),
                         ('claude-code', 'strong', ('CLAUDE_CODE_SESSION_ID',)))

    def test_family_tell_alone_is_family_not_certain(self):
        host, status, fired = hc.resolve_host({'PLUGIN_DATA': '/x'})
        self.assertEqual((host, status), ('codex', 'family'))
        self.assertEqual(fired, ('PLUGIN_DATA',))

    def test_codex_strong_and_family_is_strong(self):
        self.assertEqual(hc.resolve_host({'PLUGIN_DATA': '/x', 'CODEX_INTERNAL_ORIGINATOR_OVERRIDE': 'Codex Desktop'})[:2],
                         ('codex', 'strong'))

    def test_two_hosts_is_ambiguous_never_a_pick(self):
        host, status, fired = hc.resolve_host({'CLAUDE_CODE_SESSION_ID': 'a', 'PLUGIN_DATA': '/x'})
        self.assertEqual((host, status), (hc.HOST_UNKNOWN, 'ambiguous'))
        self.assertEqual(set(fired), {'CLAUDE_CODE_SESSION_ID', 'PLUGIN_DATA'})

    def test_nothing_is_unknown(self):
        self.assertEqual(hc.resolve_host({}), (hc.HOST_UNKNOWN, 'unknown', ()))
        self.assertEqual(hc.resolve_host({'PLUGIN_ROOT': '/x'}), (hc.HOST_UNKNOWN, 'unknown', ()))

    def test_probe_list_is_every_declared_tell(self):
        self.assertEqual(hc.all_tell_env_vars(),
                         ('CLAUDE_CODE_SESSION_ID', 'CODEX_INTERNAL_ORIGINATOR_OVERRIDE', 'PLUGIN_DATA'))


class TestClassifyTool(unittest.TestCase):
    def test_every_declared_tool_classifies_to_its_kind(self):
        for host, entry in hc.HOST_CONTRACT.items():
            for name, kind in entry['tools'].items():
                self.assertEqual(hc.classify_tool(host, name), (kind, 'ok'), (host, name))

    def test_mcp_pattern_on_both_hosts(self):
        # A plugin-adapter name (prefix kept generic: the adapter prefix is
        # built from the manifests and must not be hardcoded — test_deploy_contract).
        self.assertEqual(hc.classify_tool('claude-code', 'mcp__plugin_x_brain__recall'), ('mcp', 'ok'))
        self.assertEqual(hc.classify_tool('codex', 'mcp__brain__recall'), ('mcp', 'ok'))
        self.assertEqual(hc.classify_tool('codex', 'mcp__codex_app__read_thread'), ('mcp', 'ok'))

    def test_the_apply_patch_case(self):
        # The defect the contract exists to close: Codex's editor is an edit.
        self.assertEqual(hc.classify_tool('codex', 'apply_patch'), ('edit', 'ok'))
        self.assertEqual(hc.classify_tool('claude-code', 'Edit'), ('edit', 'ok'))

    def test_unknown_is_explicit_never_a_guess(self):
        self.assertEqual(hc.classify_tool('codex', 'Read'), ('', 'unknown'))     # not a Codex tool
        self.assertEqual(hc.classify_tool('claude-code', 'apply_patch'), ('', 'unknown'))
        self.assertEqual(hc.classify_tool('grok', 'Bash'), ('', 'unknown'))     # undeclared host
        self.assertEqual(hc.classify_tool(hc.HOST_UNKNOWN, 'Bash'), ('', 'unknown'))
        self.assertEqual(hc.classify_tool('codex', ''), ('', 'unknown'))
        self.assertEqual(hc.classify_tool('codex', None), ('', 'unknown'))

    def test_alias_names_are_not_tools(self):
        # The manifest matches 'Edit' on Codex, the payload never says it.
        self.assertEqual(hc.classify_tool('codex', 'Edit'), ('', 'unknown'))

    def test_envelope_policy_undeclared_is_none(self):
        self.assertIsNone(hc.envelope_policy('codex', '<nope>'))
        self.assertIsNone(hc.envelope_policy('grok', '<nope>'))

    def test_envelope_policy_of_an_unknown_host_reads_every_table(self):
        # As read_prompt does: the write door must not call a Claude Code tag
        # undeclared because the hook could not name the host.
        self.assertEqual(hc.envelope_policy(hc.HOST_UNKNOWN, '<agent-message'),
                         ('subagent', 'extract:agent_handback'))
        self.assertIsNone(hc.envelope_policy('codex', '<agent-message'))


class TestSessionUnattended(unittest.TestCase):
    def test_claude_code_marks_a_session_nobody_attends(self):
        self.assertTrue(hc.session_unattended('claude-code', {'CLAUDE_CODE_SESSION_ATTENDED': '0'}))
        self.assertFalse(hc.session_unattended('claude-code', {'CLAUDE_CODE_SESSION_ATTENDED': '1'}))
        self.assertFalse(hc.session_unattended('claude-code', {}))

    def test_a_host_without_the_signal_is_attended(self):
        self.assertFalse(hc.session_unattended('codex', {'CLAUDE_CODE_SESSION_ATTENDED': '0'}))

    def test_an_unknown_host_reads_every_hosts_markers(self):
        self.assertTrue(hc.session_unattended(hc.HOST_UNKNOWN, {'CLAUDE_CODE_SESSION_ATTENDED': '0'}))


class TestReadTurn(unittest.TestCase):
    """read_turn: the host's own name for the turn, from the payload field its
    entry declares — the value a message absorbed mid-turn shares with the
    prompt that opened the turn."""

    def test_each_host_reads_its_own_field(self):
        self.assertEqual(hc.read_turn({'prompt_id': 'p-1'}, 'claude-code'), 'p-1')
        self.assertEqual(hc.read_turn({'turn_id': 't-1'}, 'codex'), 't-1')
        self.assertEqual(hc.read_turn({'turn_id': 't-1'}, 'claude-code'), '')
        self.assertEqual(hc.read_turn({'prompt_id': 'p-1'}, 'codex'), '')

    def test_an_unknown_host_reads_every_hosts_field(self):
        self.assertEqual(hc.read_turn({'prompt_id': 'p-1'}, hc.HOST_UNKNOWN), 'p-1')
        self.assertEqual(hc.read_turn({'turn_id': 't-1'}, hc.HOST_UNKNOWN), 't-1')

    def test_no_turn_is_empty(self):
        for payload in ({}, {'prompt_id': ''}, {'prompt_id': 7}, None):
            with self.subTest(payload=payload):
                self.assertEqual(hc.read_turn(payload, 'claude-code'), '')


def _function(path, name):
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError('%s has no function %s' % (path, name))


def _declared_names():
    """Every tool name a host's payload carries plus every manifest alias."""
    return {n for e in hc.HOST_CONTRACT.values() for n in e['tools']} \
        | {a for e in hc.HOST_CONTRACT.values() for a in e['matcher_aliases']}


class TestHookMirrors(unittest.TestCase):
    """The hooks still carry host shape in two places (the summary builder
    needs tool_input's per-tool fields; the tell probe must read the env in the
    hook process). Both are held to the contract: names the hook branches on
    must be names the contract declares, or a host renames a tool and the two
    drift apart in silence."""

    def test_build_summary_branches_are_declared_tools(self):
        functions = [_function(os.path.join(_HOOKS, 'post_tool_trace.py'), name)
                     for name in ('_build_summary', '_raw_metadata')]
        branched = set()
        for node in (node for fn in functions for node in ast.walk(fn)):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                    and node.left.id == 'tool_name':
                for comp in node.comparators:
                    consts = comp.elts if isinstance(comp, ast.Tuple) else [comp]
                    branched |= {c.value for c in consts
                                 if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        self.assertTrue(branched, '_build_summary branches on no tool names — extractor blind')
        declared = _declared_names()
        self.assertFalse(branched - declared,
                         'post_tool_trace._build_summary branches on tool names no host declares: %s'
                         % sorted(branched - declared))

    def test_build_summary_rendered_heads_are_declared_tools(self):
        # The encoder derives the tool from the SUMMARY HEAD ('Bash: …' →
        # 'Bash'), not from metadata.tool — so the prefixes the hook renders
        # are the coupling that carries behaviour. Every literal head the
        # builder returns must be a declared name; the generic '%s: %s'
        # fallback is the one non-literal head.
        fn = _function(os.path.join(_HOOKS, 'post_tool_trace.py'), '_build_summary')
        heads = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Return) and node.value is not None:
                for c in ast.walk(node.value):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str) and ': ' in c.value:
                        head = c.value.split(':', 1)[0]
                        if head and '%' not in head and '{' not in head:
                            heads.add(head)
        self.assertTrue(heads, '_build_summary renders no "<tool>: …" heads — extractor blind')
        declared = _declared_names()
        self.assertFalse(heads - declared,
                         'post_tool_trace._build_summary renders summary heads no host declares: %s '
                         '— the encoder reads the head as the tool name' % sorted(heads - declared))

    def test_tell_probe_mirror_is_exact_and_observes_names_only(self):
        # hook_common has no module-level hook execution; the tool hook does.
        sys.path.insert(0, _HOOKS)
        try:
            import hook_common
        finally:
            sys.path.pop(0)
        from unittest import mock
        self.assertEqual(hook_common.HOST_TELL_ENV_VARS, hc.all_tell_env_vars())
        with mock.patch.dict(os.environ, {k: 'private-value' for k in hc.all_tell_env_vars()}, clear=True):
            self.assertEqual(hook_common.host_tells(), list(hc.all_tell_env_vars()))
        with mock.patch.dict(os.environ, {'PLUGIN_ROOT': '/irrelevant', 'PLUGIN_DATA': ''}, clear=True):
            self.assertEqual(hook_common.host_tells(), [])

    def test_remote_hook_mirror_is_exact(self):
        # The Python copy of the contract's remote-hook markers; the shell copy
        # (resolve-brain-db.sh) is held by tests/test_remote_hook_machine.py.
        sys.path.insert(0, _HOOKS)
        try:
            import hook_common
        finally:
            sys.path.pop(0)
        self.assertEqual(hook_common.REMOTE_HOOK_MARKERS, hc.all_remote_hook_markers())


def _run_prompt_hook(prompt, session_id, tmp, payload=None, **env):
    """Run pre_response_recall once on `prompt` against a stub daemon; the
    args it sent hook_recall, or None when it skipped the daemon. `tmp` is the
    session's BRAIN_TMP_DIR, so its markers carry across calls; `payload`
    adds hook-payload fields, `env` environment variables."""
    import runpy
    from unittest import mock
    sys.path.insert(0, _HOOKS)
    try:
        import hook_common
    finally:
        sys.path.pop(0)
    data = {'prompt': prompt, 'session_id': session_id, 'hook_event_name': 'UserPromptSubmit',
            **(payload or {})}
    reply = {'ok': True, 'result': {'json': {}}}
    with mock.patch.dict(os.environ, dict(env, CLAUDE_CODE_SESSION_ID='x', BRAIN_TMP_DIR=tmp),
                         clear=True), \
            mock.patch.object(hook_common, 'get_hook_input', return_value=data), \
            mock.patch.object(hook_common, 'daemon_available', return_value=True), \
            mock.patch.object(hook_common, 'daemon_call_raw', return_value=reply) as daemon, \
            mock.patch.object(hook_common, 'brain_debug'), \
            mock.patch.object(hook_common, 'emit_hook_output'), \
            mock.patch('os._exit'), \
            mock.patch.object(hook_common, 'run_hook',
                              side_effect=lambda name, fn, on_error=None, gated=True: fn()):
        try:
            runpy.run_path(os.path.join(_HOOKS, 'pre_response_recall.py'))
        except SystemExit:
            pass
    return daemon.call_args.args[1] if daemon.called else None


class TestPromptHookReadsTheAttendedFlag(unittest.TestCase):
    """pre_response_recall hands read_prompt the host's attended flag from its
    own environment: the same prompt is the operator's in an attended session
    and the harness's, without recall, in one nobody attends."""

    def _sent(self, attended, **payload):
        import tempfile
        tmp = tempfile.mkdtemp(prefix='attended_flag_')
        self.addCleanup(__import__('shutil').rmtree, tmp, True)
        return _run_prompt_hook('review the campaign draft', 'attended-%s' % attended, tmp,
                                payload, CLAUDE_CODE_SESSION_ATTENDED=attended)

    def test_attended_text_is_the_operators(self):
        sent = self._sent('1')
        self.assertEqual(sent['pieces'], [['operator', 'review the campaign draft', '']])
        self.assertFalse(sent['register_only'])

    def test_unattended_text_is_the_harness_without_recall(self):
        sent = self._sent('0')
        self.assertEqual(sent['pieces'], [['harness', 'review the campaign draft', '']])
        self.assertTrue(sent['register_only'])

    def test_the_hosts_turn_rides_beside_the_pieces(self):
        from servers.trace_contract import HOST_TURN_FIELD
        self.assertEqual(self._sent('1', prompt_id='p-42')[HOST_TURN_FIELD], 'p-42')
        self.assertEqual(self._sent('1')[HOST_TURN_FIELD], '')


class TestPromptHookRecordsARepeatedCommandOnce(unittest.TestCase):
    """The prompt hook remembers the session's last command: the same
    command again, with nothing typed in between, never reaches the daemon —
    a scheduled command firing every few minutes is one turn, not hundreds.
    Only the operator's own words in between reset it."""

    def test_a_command_is_sent_once_until_the_operator_speaks(self):
        import tempfile
        tmp = tempfile.mkdtemp(prefix='repeat_command_')
        self.addCleanup(__import__('shutil').rmtree, tmp, True)
        reached = [_run_prompt_hook(p, 'repeat-session', tmp) is not None for p in (
            '/babysit-prs',
            '/babysit-prs --all',
            'how are the PRs looking?',
            '/ops:babysit-prs',
            '<task-notification>\n<status>completed</status>\n</task-notification>',
            '/babysit-prs')]
        self.assertEqual(reached, [True, False, True, True, True, False])

    def test_each_session_keeps_its_own_last_command(self):
        import tempfile
        tmp = tempfile.mkdtemp(prefix='repeat_command_')
        self.addCleanup(__import__('shutil').rmtree, tmp, True)
        self.assertIsNotNone(_run_prompt_hook('/babysit-prs', 'session-a', tmp))
        self.assertIsNotNone(_run_prompt_hook('/babysit-prs', 'session-b', tmp))


class TestPromptHookFailsLoudly(unittest.TestCase):
    """When reading the prompt raises, the turn is lost — so the hook says so
    in the session, beside run_hook's log row."""

    def test_a_reader_crash_is_announced_in_the_session(self):
        import runpy
        import tempfile
        from unittest import mock
        sys.path.insert(0, _HOOKS)
        try:
            import hook_common
        finally:
            sys.path.pop(0)
        tmp = tempfile.mkdtemp(prefix='reader_crash_')
        self.addCleanup(__import__('shutil').rmtree, tmp, True)

        def run_hook(name, fn, on_error=None, gated=True):   # run_hook's own contract
            try:
                fn()
            except Exception:
                on_error()
        data = {'prompt': 'what changed?', 'session_id': 'reader-crash',
                'hook_event_name': 'UserPromptSubmit'}
        with mock.patch.dict(os.environ, {'CLAUDE_CODE_SESSION_ID': 'x', 'BRAIN_TMP_DIR': tmp}, clear=True), \
                mock.patch.object(hc, 'read_prompt', side_effect=ValueError('boom')), \
                mock.patch.object(hook_common, 'get_hook_input', return_value=data), \
                mock.patch.object(hook_common, 'daemon_call_raw') as daemon, \
                mock.patch.object(hook_common, 'brain_debug'), \
                mock.patch.object(hook_common, 'emit_hook_output') as emit, \
                mock.patch.object(hook_common, 'run_hook', side_effect=run_hook):
            runpy.run_path(os.path.join(_HOOKS, 'pre_response_recall.py'))
        daemon.assert_not_called()
        event, payload = emit.call_args.args
        self.assertEqual(event, 'UserPromptSubmit')
        self.assertIn('PROMPT INTAKE FAILED', payload['additionalContext'])


class TestLeaf(unittest.TestCase):
    def test_import_reaches_neither_daemon_config_nor_brain(self):
        # The prompt hook may import the contract; a hook must never pay the
        # daemon's import-time fingerprint (or load the brain) to read a table.
        code = ("import sys; import servers.host_contract as h; h.classify_tool('codex', 'Bash'); "
                "print(sorted(m for m in sys.modules if m in "
                "('servers.daemon_config', 'servers.brain', 'servers.daemon_client')))")
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), '[]', 'host_contract pulled in: %s' % out.stdout)


class TestFingerprint(unittest.TestCase):
    def test_twelve_hex_and_stable(self):
        a, b = hc.contract_fingerprint(), hc.contract_fingerprint()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        int(a, 16)


class TestToolResultShape(unittest.TestCase):
    def test_tool_result_is_registered_with_the_chokepoint(self):
        self.assertIs(METADATA_REQUIRED_BY_REF_TYPE.get('tool_result'), TOOL_RESULT_METADATA_SHAPE)

    def test_unstamped_payload_requires_the_write_door(self):
        from servers.brain_traces import stamp_tool_result
        raw = {'tool': 'Bash'}
        self.assertFalse(validate_trace_metadata('delta', 'tool_result', raw)[0])
        self.assertEqual(validate_trace_metadata('delta', 'tool_result',
                         stamp_tool_result(raw, {})), (True, ''))
        self.assertEqual(set(TOOL_RESULT_METADATA_SHAPE),
                         {'tool'} | set(TOOL_RESULT_NORMALIZATION_KEYS))

    def test_missing_tool_is_refused(self):
        ok, err = validate_trace_metadata('delta', 'tool_result', {'kind': 'shell'})
        self.assertFalse(ok)
        self.assertIn('tool', err)

    def test_builder_minimal_and_full(self):
        self.assertEqual(build_tool_result_metadata(tool='Bash'), {'tool': 'Bash'})
        full = build_tool_result_metadata(
            tool='apply_patch', kind='edit', kind_status='ok', host_status='family',
            tells=('PLUGIN_DATA',), vocab_version=hc.VOCAB_VERSION,
            impl_identity=hc.contract_fingerprint(), tool_use_id='call_1', turn_id='t1',
            payload_keys=('session_id', 'tool_name'))
        self.assertEqual(set(full), {'tool'} | set(TOOL_RESULT_NORMALIZATION_KEYS))
        self.assertEqual(full['tells'], ['PLUGIN_DATA'])           # tuples land as lists (JSON-shaped)
        self.assertEqual(validate_trace_metadata('delta', 'tool_result', full), (True, ''))

    def test_builder_omits_none_and_refuses_unknown_keys(self):
        self.assertEqual(build_tool_result_metadata(tool='Bash', kind=None), {'tool': 'Bash'})
        for key in ('kidn', 'model', 'host'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                build_tool_result_metadata(tool='Bash', **{key: 'value'})

    def test_normalization_vocabulary_agrees_with_the_contract(self):
        # A stamped kind is always a word the contract could have produced.
        for host, entry in hc.HOST_CONTRACT.items():
            for kind in entry['tools'].values():
                self.assertIn(kind, ACTION_KINDS)
            for speaker, policy in entry['envelopes'].values():
                self.assertIn(speaker, SPEAKER_REF_TYPES)
                self.assertTrue(policy in ENVELOPE_POLICIES or policy.startswith('extract:'), policy)


class TestReadPrompt(unittest.TestCase):
    """The one decision site for an incoming prompt: what gets in, who each
    piece is from, what recall reads. Shapes are the ones the 2026-10-05..11
    census found rendering as the operator's words."""

    def read(self, prompt, host='claude-code'):
        return hc.read_prompt(prompt, host)

    def test_command_shapes(self):
        # A slash command reaches the prompt hook only when it starts a model
        # turn: a skill or custom command arrives as the line typed, a built-in
        # like /model runs locally and fires no hook (measured 2026-10-19, CLI
        # 2.1.303). So a typed command is the operator's prompt; a repeating
        # one is not, or the same line would fill the timeline (brain 7d615634).
        def said(text, command=''):
            return hc.Intake((('operator', text, ''),), text, command)
        for prompt, want in (
                ('', hc.Intake((), '')),
                ('   \n', hc.Intake((), '')),
                ('!git status', hc.Intake((), '')),
                ('/watch', hc.Intake((), '', 'watch')),
                ('/entity:watch', hc.Intake((), '', 'watch')),
                ('/loop 5m /babysit-prs', hc.Intake((), '', 'loop')),
                ('/code-review fix the auth bug', said('/code-review fix the auth bug', 'code-review')),
                ('/entity:shape-review', said('/entity:shape-review', 'shape-review')),
                ('/etc/hosts still has the old entry', said('/etc/hosts still has the old entry'))):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.read(prompt), want)

    def test_the_same_command_again_is_not_recorded(self):
        # A command that fires on a schedule (a /loop of /babysit-prs, a cron
        # job) arrives as the same command each time, with nothing typed in
        # between: it is recorded once. The same command is matched wide — by
        # name alone, whatever its args, plugin prefix or case.
        for prompt in ('/babysit-prs', '/babysit-prs --all', '/ops:babysit-prs', '/Babysit-PRs now'):
            with self.subTest(prompt=prompt):
                self.assertEqual(hc.read_prompt(prompt, 'claude-code', last_command='babysit-prs'),
                                 hc.Intake((), '', 'babysit-prs'))
        self.assertEqual(hc.read_prompt('/code-review', 'claude-code', last_command='babysit-prs'),
                         hc.Intake((('operator', '/code-review', ''),), '/code-review', 'code-review'))
        self.assertEqual(hc.read_prompt('/etc/hosts again', 'claude-code', last_command='etc'),
                         hc.Intake((('operator', '/etc/hosts again', ''),), '/etc/hosts again'))

    def test_a_host_that_cannot_be_told_reads_every_hosts_repeating_commands(self):
        self.assertEqual(hc.read_prompt('/loop 5m /babysit-prs', hc.HOST_UNKNOWN), hc.Intake((), '', 'loop'))

    def test_a_command_behind_a_reminder_is_still_a_command(self):
        self.assertEqual(self.read('<system-reminder>\nworktree\n</system-reminder>\n/watch'),
                         hc.Intake((), '', 'watch'))
        self.assertEqual(self.read('<system-reminder>\nworktree\n</system-reminder>\n/code-review'),
                         hc.Intake((('operator', '/code-review', ''),
                                    ('harness', 'worktree', '<system-reminder>')), '/code-review',
                                   'code-review'))

    def test_operator_text_is_recalled_unless_it_is_a_short_answer(self):
        self.assertEqual(self.read('what did we decide?'),
                         hc.Intake((('operator', 'what did we decide?', ''),), 'what did we decide?'))
        self.assertEqual(self.read(' yes '), hc.Intake((('operator', 'yes', ''),), ''))

    def test_a_glued_reminder_is_cut_into_its_own_piece_after_the_operator(self):
        intake = self.read('<system-reminder>\nYou are operating in a git worktree.\n'
                           '</system-reminder>\n\n\nIn main, the test fails.')
        self.assertEqual(intake.pieces, (
            ('operator', 'In main, the test fails.', ''),
            ('harness', 'You are operating in a git worktree.', '<system-reminder>')))
        self.assertEqual(intake.recall_text, 'In main, the test fails.')

    def test_a_prompt_of_reminders_alone_holds_no_operator_text(self):
        intake = self.read('<system-reminder>\nA started.\n</system-reminder>\n'
                           '<system-reminder>\nB started.\n</system-reminder>')
        self.assertEqual([p[:2] for p in intake.pieces],
                         [('harness', 'A started.'), ('harness', 'B started.')])
        self.assertEqual(intake.recall_text, '')

    def test_a_reminder_named_mid_sentence_or_left_unclosed_stays_operator_text(self):
        for prompt in ('why does a <system-reminder> skip recall?',
                       '<system-reminder>\nnever closed, so it is not a block'):
            with self.subTest(prompt=prompt):
                self.assertEqual([p[0] for p in self.read(prompt).pieces], ['operator'])

    def test_a_subagent_report_is_its_own_piece_holding_the_report_alone(self):
        prompt = ('<agent-message from="a7fd5eda043dbd1a2">\n'
                  '[Subagent hand-back] The text below is the final report. The report follows.\n'
                  '  Both guards are built.\n  \n  **Decision:** four mismatches.\n'
                  '    - nested item\n</agent-message>\n')
        report = 'Both guards are built.\n\n**Decision:** four mismatches.\n  - nested item'
        self.assertEqual(self.read(prompt),
                         hc.Intake((('subagent', report, '<agent-message'),), report))

    def test_every_report_in_a_prompt_is_its_own_piece(self):
        # Two hand-backs are two voices: one row each, the first opening the
        # turn; recall reads them together.
        prompt = ('<agent-message from="a1">\n[Subagent hand-back] notice\n  first report\n'
                  '</agent-message>\n\n<agent-message from="a2">\n[Subagent hand-back] notice\n'
                  '  second report\n    indented line\n</agent-message>')
        self.assertEqual(self.read(prompt), hc.Intake(
            (('subagent', 'first report', '<agent-message'),
             ('subagent', 'second report\n  indented line', '<agent-message')),
            'first report\n\nsecond report\n  indented line'))

    def test_a_report_the_extractor_cannot_read_keeps_the_raw_hand_back(self):
        # Nothing is lost: the row holds what arrived, and nothing is recalled
        # on it (the daemon logs 'hook_recall_extract').
        prompt = '<agent-message from="x">\n[Subagent hand-back] notice\n</agent-message>'
        self.assertEqual(self.read(prompt),
                         hc.Intake((('subagent', prompt, '<agent-message'),), ''))

    def test_an_envelope_the_operator_quotes_stays_their_text(self):
        # Only the harness's bracketed notice line may precede a whole-prompt
        # envelope; operator text before it means they are quoting one.
        prompt = 'look at what the agent said:\n<agent-message from="x">\n[Subagent hand-back] n\n  r'
        self.assertEqual([p[0] for p in self.read(prompt).pieces], ['operator'])
        self.assertEqual(self.read(prompt).recall_text, prompt)

    def test_only_reminders_at_the_front_are_cut(self):
        # The operator writing about the tag, even on its own line, keeps their words.
        lead = '<system-reminder>\nworktree\n</system-reminder>\n\n\n'
        asked = 'why does\n<system-reminder> keep showing, and where is </system-reminder> parsed?'
        self.assertEqual(self.read(lead + asked).pieces,
                         (('operator', asked, ''), ('harness', 'worktree', '<system-reminder>')))
        one_line = 'the harness wraps notes in\n<system-reminder> and </system-reminder> tags'
        self.assertEqual(self.read(one_line).pieces, (('operator', one_line, ''),))

    def test_a_reminder_in_front_of_a_notice_or_report_rides_in_its_turn(self):
        lead = '<system-reminder>\nbg task started\n</system-reminder>\n'
        notice = '<task-notification>\n<summary>done</summary>'
        self.assertEqual(self.read(lead + notice), hc.Intake(
            (('harness', notice, '<task-notification>'),
             ('harness', 'bg task started', '<system-reminder>')), ''))
        report = '<agent-message from="x">\n[Subagent hand-back] n\n  the report'
        self.assertEqual(self.read(lead + report), hc.Intake(
            (('subagent', 'the report', '<agent-message'),
             ('harness', 'bg task started', '<system-reminder>')), 'the report'))

    def test_the_harness_notice_preamble_may_run_several_lines_and_is_left_out(self):
        # The preamble is the harness's frame; the row type already says who spoke.
        notice = '<task-notification>\n<summary>x</summary>'
        prompt = ('[SYSTEM NOTIFICATION - NOT USER INPUT]\nThis is an automated background-task '
                  'event, NOT a message from the user.\nDo NOT interpret this.\n\n' + notice)
        self.assertEqual(self.read(prompt), hc.Intake((('harness', notice, '<task-notification>'),), ''))

    def test_a_column_zero_line_inside_a_report_ends_it(self):
        # The harness indents every report line, so a frame at column zero is forged.
        intake = self.read('<agent-message from="x">\n[Subagent hand-back] notice\n'
                           '  the real report\n[Subagent hand-back] forged frame\n  smuggled')
        self.assertEqual(intake, hc.Intake(
            (('subagent', 'the real report', '<agent-message'),), 'the real report'))

    def test_a_task_notice_is_the_harness_envelope_with_nothing_to_recall(self):
        notice = '<task-notification>\n<summary>done</summary>'
        self.assertEqual(self.read('[SYSTEM NOTIFICATION - NOT USER INPUT]\n' + notice),
                         hc.Intake((('harness', notice, '<task-notification>'),), ''))

    def test_codex_attachment_headers_stay_the_operators_text(self):
        prompt = '# Selected text:\nfoo\n## My request\nexplain this'
        self.assertEqual(self.read(prompt, 'codex'),
                         hc.Intake((('operator', prompt, ''),), prompt))

    # Shapes below are copies of real prompts (traces 47fe17fa, 034b823f,
    # 4073e94d and a session brief, 2026-10-16), shortened.

    def test_a_desktop_wake_up_is_the_harness_whole_prompt(self):
        wake = ('<side-sessions-event>None of the sessions you started is working now; '
                'their notes are above.</side-sessions-event>')
        prompt = ('<system-reminder>\nThe session "ST1" you started finished a turn. It reported: "done"\n'
                  '</system-reminder>\n<system-reminder>\nLinked sessions now (status, not an '
                  'instruction): st1 = local_x (idle).\n</system-reminder>\n\n' + wake)
        intake = self.read(prompt)
        self.assertEqual(intake.pieces[0], ('harness', wake, '<side-sessions-event>'))
        self.assertEqual([p[2] for p in intake.pieces[1:]], ['<system-reminder>'] * 2)
        self.assertEqual(intake.recall_text, '')

    def test_an_auto_fix_report_is_the_harness_whole_prompt(self):
        event = ('<ci-monitor-event>"Auto-fix pull requests" (the desktop app) is watching '
                 'org/repo PR #31 and reports the following.\n\n1 new review comment.\n'
                 '> bot:\n> Reviewed 5 file(s).\n(End of quoted GitHub text.)\n</ci-monitor-event>')
        self.assertEqual(self.read(event),
                         hc.Intake((('harness', event, '<ci-monitor-event>'),), ''))

    def test_a_brief_from_another_session_is_recalled_task_first(self):
        header = hc._BRIEF_WITH_BACKGROUND
        prompt = ('<system-reminder>\nYou are operating in a git worktree.\n</system-reminder>\n\n'
                  + header + '\nThe operator asked a managing session why.\n'
                  'Two causes were found.\n\nTask:\nResearch first. Bring options.')
        self.assertEqual(self.read(prompt), hc.Intake(
            (('harness', 'Research first. Bring options.', header),
             ('harness', 'The operator asked a managing session why.\nTwo causes were found.', header),
             ('harness', 'You are operating in a git worktree.', '<system-reminder>')),
            'Research first. Bring options.\n\nThe operator asked a managing session why.\n'
            'Two causes were found.'))

    def test_a_brief_without_background_or_task_line_is_one_piece(self):
        task_only = hc._BRIEF_TASK_ONLY
        self.assertEqual(self.read(task_only + '\nFix the flaky test.'), hc.Intake(
            (('harness', 'Fix the flaky test.', task_only),), 'Fix the flaky test.'))
        # A brief whose task line is missing is kept whole, not split.
        header = hc._BRIEF_WITH_BACKGROUND
        self.assertEqual(self.read(header + '\nContext and task run together.'), hc.Intake(
            (('harness', 'Context and task run together.', header),),
            'Context and task run together.'))

    def test_a_background_that_quotes_a_task_line_splits_at_the_apps_own(self):
        # The app appends its task line after the background, so the last one is its.
        header = hc._BRIEF_WITH_BACKGROUND
        background = 'An earlier brief read:\n\nTask:\nold task'
        self.assertEqual(self.read(header + '\n' + background + '\n\nTask:\nthe real task').pieces,
                         (('harness', 'the real task', header), ('harness', background, header)))

    def test_a_task_only_brief_is_never_split_at_a_task_line(self):
        task_only = hc._BRIEF_TASK_ONLY
        body = 'Look at the plan.\n\nTask:\nthis line is part of the task'
        self.assertEqual(self.read(task_only + '\n' + body).pieces,
                         (('harness', body, task_only),))

    def test_an_empty_brief_keeps_its_frame_without_recall(self):
        header = hc._BRIEF_WITH_BACKGROUND
        self.assertEqual(self.read(header), hc.Intake((('harness', header, header),), ''))

    def test_a_message_from_another_session_keeps_its_sender_and_is_recalled_on_its_words(self):
        opening = '<cross-session-message from="local_28af" name="Uncommitted work">'
        words = 'Your side finding is confirmed.\nPlease dismiss your card.'
        intake = self.read(opening + '\n' + words + '\n</cross-session-message>')
        self.assertEqual(intake, hc.Intake(
            (('harness', opening + '\n' + words, '<cross-session-message'),), words))

    def test_the_harness_note_after_a_message_is_its_own_piece_never_recalled(self):
        block = '<cross-session-message from="local_a" name="A">\nplease rebase'
        note = 'This came from another Claude session, not typed by your user.'
        intake = self.read(block + '\n</cross-session-message>\n\n' + note)
        self.assertEqual(intake, hc.Intake(
            (('harness', block, '<cross-session-message'),
             ('harness', note, '<cross-session-message')), 'please rebase'))

    def test_two_messages_from_other_sessions_are_two_pieces(self):
        first = '<cross-session-message from="local_a" name="A">\nfirst'
        second = '<cross-session-message from="local_b" name="B">\nsecond'
        intake = self.read(first + '\n</cross-session-message>\n\n' + second + '\n</cross-session-message>')
        self.assertEqual([p[1] for p in intake.pieces], [first, second])
        self.assertEqual(intake.recall_text, 'first\n\nsecond')

    def test_an_undeclared_wrapper_that_is_the_whole_prompt_is_the_harness(self):
        wrapper = '<new-wake-event kind="x">\nSomething happened.\n</new-wake-event>'
        self.assertEqual(self.read(wrapper),
                         hc.Intake((('harness', wrapper, '<new-wake-event>'),), ''))
        lead = '<system-reminder>\nnote\n</system-reminder>\n'
        self.assertEqual(self.read(lead + wrapper).pieces[0],
                         ('harness', wrapper, '<new-wake-event>'))

    def test_a_block_followed_by_the_operators_words_stays_theirs(self):
        # An attachment leads the operator's words, declared or not; an
        # unclosed block is not a block; a declared attachment alone is theirs.
        for prompt in ('<artifact-view-context artifact="138b">{"tab": 1}</artifact-view-context>\n'
                       'what is this chart?',
                       '<ide_selection>x = 1</ide_selection>\nwhy does this fail?',
                       '<pasted_content id="1d6b">\npasted text\n</pasted_content id="1d6b">',
                       '<artifact-view-context artifact="138b">{"tab": 1}</artifact-view-context>',
                       '<b>bold</b> is what I want here'):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.read(prompt).pieces, (('operator', prompt, ''),))

    def test_an_unattended_sessions_own_text_is_the_harness_without_recall(self):
        self.assertEqual(hc.read_prompt('Review the campaign draft.', 'claude-code', unattended=True),
                         hc.Intake((('harness', 'Review the campaign draft.', ''),), ''))
        # Envelopes still read as declared, and a command is still a command.
        report = '<agent-message from="x">\n[Subagent hand-back] n\n  the report'
        self.assertEqual(hc.read_prompt(report, 'claude-code', unattended=True),
                         hc.Intake((('subagent', 'the report', '<agent-message'),), 'the report'))
        self.assertEqual(hc.read_prompt('/watch', 'claude-code', unattended=True), hc.Intake((), '', 'watch'))
        self.assertEqual(hc.read_prompt('/code-review', 'claude-code', unattended=True),
                         hc.Intake((('harness', '/code-review', ''),), '', 'code-review'))

    def test_a_malformed_message_never_raises_in_the_hook(self):
        # A malformed tag of the declared name is still that envelope: the hook
        # records the raw block without recall (the daemon logs the failed
        # extraction as an error), never raises and never files it as the operator's.
        for block in ('<cross-session-message', '<cross-session-message<x'):
            with self.subTest(block=block):
                self.assertEqual(self.read(block + '\n</cross-session-message>'), hc.Intake(
                    (('harness', block, '<cross-session-message'),), ''))

    def test_a_gt_inside_a_quoted_sender_name_stays_in_the_tag(self):
        opening = '<cross-session-message from="local_x" name="Plan -> build">'
        self.assertEqual(self.read(opening + '\nship it\n</cross-session-message>').recall_text,
                         'ship it')

    def test_a_longer_tag_name_is_not_the_declared_one(self):
        # '<cross-session-message' declares that name only; a longer name is a
        # wrapper the contract has not met, logged under its own tag.
        batch = '<cross-session-message-batch k="1">\nx\n</cross-session-message-batch>'
        self.assertEqual(self.read(batch), hc.Intake(
            (('harness', batch, '<cross-session-message-batch>'),), ''))
        later = '<agent-messages>a</agent-messages>\nx\n<agent-message from="z">\n'
        self.assertEqual(hc._line_start(later, '<agent-message'), later.index('<agent-message from'))

    # One real-shaped sample per registered extractor; a new extractor fails
    # here until it has one.
    EXTRACTOR_SAMPLES = {
        'agent_handback': ('<agent-message', '<agent-message from="x">\n[Subagent hand-back] n\n'
                                             '  the report\n</agent-message>'),
        'session_brief': (hc._BRIEF_WITH_BACKGROUND,
                          hc._BRIEF_WITH_BACKGROUND + '\nwhy\n\nTask:\ndo it'),
        'session_messages': ('<cross-session-message',
                             '<cross-session-message from="a">\nhi\n</cross-session-message>\n\nnote'),
    }

    def test_every_extractor_returns_stored_recalled_pairs(self):
        self.assertEqual(set(self.EXTRACTOR_SAMPLES), set(hc.EXTRACTORS))
        for name, (tag, prompt) in self.EXTRACTOR_SAMPLES.items():
            with self.subTest(extractor=name):
                found = hc.EXTRACTORS[name](prompt, tag)
                self.assertIsInstance(found, tuple)
                self.assertTrue(found)
                for pair in found:
                    self.assertTrue(isinstance(pair, tuple) and len(pair) == 2
                                    and all(isinstance(t, str) for t in pair), pair)

    def test_every_extract_policy_names_a_registered_extractor(self):
        for host, entry in hc.HOST_CONTRACT.items():
            for tag, (speaker, policy) in entry['envelopes'].items():
                if policy.startswith('extract:'):
                    self.assertIn(policy[len('extract:'):], hc.EXTRACTORS, (host, tag))


WRAP_UP = 'mcp__ccd_turn__wrap_up'


class TestReadReply(unittest.TestCase):
    """The reply side's one decision site: the host's reply text, plus what a
    declared reply tool closing the turn showed the operator."""

    def test_a_closing_message_without_a_reply_tool_keeps_the_host_text_verbatim(self):
        for calls in ((), [('Bash', {'command': 'ls'})], None):
            with self.subTest(calls=calls):
                self.assertEqual(hc.read_reply('Done.\n', calls, 'claude-code'), 'Done.\n')
        self.assertEqual(hc.read_reply('', (), 'claude-code'), '')

    def test_a_turn_closed_by_the_next_step_box_records_what_the_box_showed(self):
        calls = [('mcp__ccd_turn__update_plan', {'advance': True}),
                 (WRAP_UP, {'recap': 'The fix is built.', 'ask': 'Commit it?',
                            'actions': [{'label': 'Yes'}]})]
        self.assertEqual(hc.read_reply('', calls, 'claude-code'), 'The fix is built.\n\nCommit it?')
        # text in the same closing message leads the box
        self.assertEqual(hc.read_reply('Here is why.\n', calls, 'claude-code'),
                         'Here is why.\n\nThe fix is built.\n\nCommit it?')
        # a recap-only box, and empty or non-string fields, record what there is
        self.assertEqual(hc.read_reply('', [(WRAP_UP, {'recap': ' Shipped. ', 'ask': ''})],
                                       'claude-code'), 'Shipped.')
        self.assertEqual(hc.read_reply('', [(WRAP_UP, {'recap': 3})], 'claude-code'), '')
        self.assertEqual(hc.read_reply('', [(WRAP_UP, 'not a dict')], 'claude-code'), '')

    def test_reply_tools_come_from_readable_transcripts_only(self):
        self.assertEqual(hc.reply_tools('claude-code'), {WRAP_UP: ('recap', 'ask')})
        self.assertEqual(hc.reply_tools('codex'), {})
        # an unknown host reads every host's, as read_prompt does
        self.assertEqual(hc.reply_tools(hc.HOST_UNKNOWN), {WRAP_UP: ('recap', 'ask')})
        self.assertEqual(hc.read_reply('', [(WRAP_UP, {'recap': 'r'})], 'codex'), '')


class TestStopHookReadsTheReply(unittest.TestCase):
    """post_response_track runs read_reply on the turn's closing message and
    sends the reply with the contract stamp; a reader failure still records
    the host's text, unstamped."""

    def _transcript(self, entries):
        import json
        import tempfile
        f = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
        for e in entries:
            f.write(json.dumps(e) + '\n')
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def _sent(self, data, env, **patches):
        import contextlib
        import runpy
        import tempfile
        from unittest import mock
        sys.path.insert(0, _HOOKS)
        try:
            import hook_common
        finally:
            sys.path.pop(0)
        tmp = tempfile.mkdtemp(prefix='stop_reply_')
        self.addCleanup(__import__('shutil').rmtree, tmp, True)
        with mock.patch.dict(os.environ, dict(env, BRAIN_TMP_DIR=tmp), clear=True), \
                mock.patch.object(hook_common, 'get_hook_input', return_value=data), \
                mock.patch.object(hook_common, 'daemon_available', return_value=True), \
                mock.patch.object(hook_common, 'daemon_call_raw',
                                  return_value={'ok': True, 'result': {}}) as daemon, \
                mock.patch.object(hook_common, 'brain_debug'), \
                mock.patch.object(hook_common, 'emit_hook_output'), \
                mock.patch.object(hook_common, 'log_hook_error') as logged, \
                (mock.patch.multiple(hc, **patches) if patches else contextlib.nullcontext()), \
                mock.patch.object(hook_common, 'run_hook',
                                  side_effect=lambda name, fn, on_error=None, gated=True: fn()):
            runpy.run_path(os.path.join(_HOOKS, 'post_response_track.py'))
        self.assertEqual(daemon.call_args.args[0], 'hook_post_response_track')
        return daemon.call_args.args[1], logged

    def _box_turn(self):
        return self._transcript([
            {'type': 'user', 'message': {'content': 'ship it?'}},
            {'type': 'assistant', 'message': {'id': 'm1', 'model': 'claude-opus-5-5', 'content': [
                {'type': 'text', 'text': 'Checking the tests.'}]}},
            {'type': 'assistant', 'message': {'id': 'm2', 'model': 'claude-opus-5-5', 'content': [
                {'type': 'tool_use', 'name': WRAP_UP,
                 'input': {'recap': 'Tests pass.', 'ask': 'Merge now?'}}]}},
            {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'content': 'ok'}]}},
        ])

    def test_a_box_closed_turn_sends_what_the_box_showed_with_the_stamp(self):
        sent, logged = self._sent({'session_id': 's', 'hook_event_name': 'Stop',
                                   'last_assistant_message': '', 'transcript_path': self._box_turn()},
                                  {'CLAUDE_CODE_SESSION_ID': 'x'})
        self.assertEqual(sent['last_assistant_message'], 'Tests pass.\n\nMerge now?')
        self.assertEqual(sent['vocab_version'], hc.VOCAB_VERSION)
        self.assertEqual(sent['impl_identity'], hc.contract_fingerprint())
        self.assertEqual(sent['model'], 'claude-opus-5-5')
        logged.assert_not_called()

    def test_a_reader_failure_records_the_host_text_unstamped(self):
        from unittest import mock
        sent, logged = self._sent({'session_id': 's', 'hook_event_name': 'Stop',
                                   'last_assistant_message': 'Host text.',
                                   'transcript_path': self._box_turn()},
                                  {'CLAUDE_CODE_SESSION_ID': 'x'},
                                  read_reply=mock.Mock(side_effect=ValueError('boom')))
        self.assertEqual(sent['last_assistant_message'], 'Host text.')
        self.assertNotIn('vocab_version', sent)
        self.assertNotIn('impl_identity', sent)
        logged.assert_called_once()

    def test_a_host_without_reply_tools_never_reads_the_transcript(self):
        # Codex declares no reply tools (its rollout is unreadable): the hook
        # sends the host's text as it came, and opens no transcript.
        sent, _ = self._sent({'session_id': 's', 'hook_event_name': 'Stop', 'model': 'gpt-6',
                              'last_assistant_message': 'Codex text.',
                              'transcript_path': '/nope/rollout.jsonl'},
                             {'CODEX_INTERNAL_ORIGINATOR_OVERRIDE': 'x'})
        self.assertEqual(sent['last_assistant_message'], 'Codex text.')
        self.assertEqual(sent['vocab_version'], hc.VOCAB_VERSION)


if __name__ == '__main__':
    unittest.main()
