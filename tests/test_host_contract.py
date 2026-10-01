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
    daemon's write door uses to stamp what the contract classified.
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
    ACTION_KINDS, ENVELOPE_POLICIES, HOST_STATUS, KIND_STATUS,
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
        self._refuses(lambda c: c['codex']['envelopes'].__setitem__('<x>', 'extract:nope'), 'not registered')

    def test_bogus_envelope_policy(self):
        self._refuses(lambda c: c['claude-code']['envelopes'].__setitem__('<x>', 'ignore'), 'policy')

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

    def test_unverified_entry(self):
        self._refuses(lambda c: c['codex'].__setitem__('verified', {}), 'host_version')

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
            for policy in entry['envelopes'].values():
                self.assertTrue(policy in ENVELOPE_POLICIES or policy.startswith('extract:'), policy)


if __name__ == '__main__':
    unittest.main()
