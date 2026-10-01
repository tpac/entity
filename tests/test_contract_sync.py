"""
Contract Sync Tests — ensures all brain layers stay in sync.

The brain has 5 layers that must agree on method names, parameters, and defaults:
  1. Brain methods (servers/brain_*.py) — the source of truth
  2. Daemon dispatch (servers/daemon_dispatch.py) — command routing
  3. MCP server (servers/brain_mcp.py) — tool definitions for Claude
  4. Hook scripts (hooks/scripts/*.py) — automated triggers
  5. Scale dispatch/runner (servers/scales/) — background agent infrastructure

Run: python3 -m pytest tests/test_contract_sync.py -v
"""

import inspect
import json
import os
import re
import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "servers"))


class TestBrainMethodsExist(unittest.TestCase):
    """Layer 1: Brain methods are the source of truth. Verify they exist and have correct signatures."""

    @classmethod
    def setUpClass(cls):
        """Introspect all brain modules for public encoding/connection methods."""
        from servers import brain_reminders, brain_connections, brain_remember

        cls.methods = {}
        for mod_name, mod in [
            ('brain_reminders', brain_reminders),
            ('brain_connections', brain_connections),
            ('brain_remember', brain_remember),
        ]:
            for name in dir(mod):
                cls_obj = getattr(mod, name)
                if not inspect.isclass(cls_obj):
                    continue
                for method_name, method in inspect.getmembers(cls_obj, predicate=inspect.isfunction):
                    if method_name.startswith('_'):
                        continue
                    if any(method_name.startswith(p) for p in
                           ('remember', 'create_', 'connect', 'record_', 'learn_', 'recall', 'revise')):
                        sig = inspect.signature(method)
                        params = [k for k in sig.parameters.keys() if k != 'self']
                        defaults = {
                            k: v.default for k, v in sig.parameters.items()
                            if k != 'self' and v.default is not inspect.Parameter.empty
                        }
                        cls.methods[method_name] = {
                            'module': mod_name,
                            'params': params,
                            'defaults': defaults,
                        }

    def test_core_methods_exist(self):
        """Critical write methods must exist in brain."""
        required = [
            'remember', 'remember_batch',
            'revise', 'revise_batch',
            'connect',
        ]
        for method in required:
            self.assertIn(method, self.methods,
                          f"Brain missing core method: {method}")

    def test_remember_has_required_params(self):
        """remember() must accept type, title, content, locked."""
        params = self.methods['remember']['params']
        for p in ['type', 'title', 'content', 'locked']:
            self.assertIn(p, params, f"remember() missing param: {p}")

    def test_connect_has_required_params(self):
        """connect() must accept source_id, target_id, relation, weight."""
        params = self.methods['connect']['params']
        for p in ['source_id', 'target_id', 'relation', 'weight']:
            self.assertIn(p, params, f"connect() missing param: {p}")


class TestDaemonDispatchSync(unittest.TestCase):
    """Layer 2: Every core brain method must have a daemon dispatch command."""

    @classmethod
    def setUpClass(cls):
        from servers.daemon_dispatch import COMMAND_TABLE
        cls.commands = set(COMMAND_TABLE.keys())

    def test_core_write_commands_exist(self):
        """All write methods must be in daemon dispatch."""
        required_commands = [
            'remember', 'remember_batch',
            'revise', 'revise_batch',
            'connect', 'connect_batch',
            'brain_batch',
        ]
        missing = [cmd for cmd in required_commands if cmd not in self.commands]
        self.assertEqual(missing, [],
                         f"Daemon dispatch missing commands: {missing}")
        # Consolidated from the former standalone test_connection_commands_exist
        # — 'connect' is in required_commands above; this preserves its explicit
        # presence assertion.
        self.assertIn('connect', self.commands)

    def test_core_read_commands_exist(self):
        """Essential read commands must exist."""
        for cmd in ['recall', 'ping', 'context_boot', 'save', 'eval']:
            self.assertIn(cmd, self.commands, f"Missing core command: {cmd}")

    def test_removed_commands_gone(self):
        """Deprecated commands should not be in dispatch."""
        removed = [
            'remember_lesson', 'remember_impact', 'remember_mechanism',
            'remember_convention', 'remember_uncertainty', 'remember_mental_model',
            'record_divergence', 'learn_vocabulary',
            'auto_heal', 'auto_tune',
            'consciousness', 'dismiss_signal', 'queue_state',
            'migrate_to_aspects',  # retired 2026-05-29 — obsolete aspect-node migration
            # no sender (2026-09-29):
            'validate_config', 'procedure_trigger', 'get_debug_status',
            'health_check', 'log_debug', 'promote_staged', 'backfill_vectors',
            'graph_expand', 'diagnose', 'scan_host',
        ]
        present = [cmd for cmd in removed if cmd in self.commands]
        self.assertEqual(present, [],
                         f"Removed commands still in dispatch: {present}")


class TestMCPToolSync(unittest.TestCase):
    """Layer 3: MCP tools must match daemon commands and brain signatures."""

    @classmethod
    def setUpClass(cls):
        from servers.brain_mcp import TOOLS

        cls.mcp_tools = set(t["name"] for t in TOOLS)
        cls.mcp_params = {}
        for t in TOOLS:
            schema = t.get("inputSchema", {})
            props = list(schema.get("properties", {}).keys())
            cls.mcp_params[t["name"]] = props

    def test_mcp_tools_have_daemon_commands(self):
        """Every MCP tool must have a matching daemon command."""
        from servers.daemon_dispatch import COMMAND_TABLE
        daemon_cmds = set(COMMAND_TABLE.keys())
        # These commands are handled directly by daemon_server before COMMAND_TABLE dispatch
        daemon_direct = {'restart', 'shutdown'}

        for tool in self.mcp_tools:
            self.assertTrue(tool in daemon_cmds or tool in daemon_direct,
                          f"MCP tool '{tool}' has no daemon command — will fail at runtime")

    def test_core_write_tools_exposed(self):
        """Core write tools should be MCP-native."""
        should_be_mcp = [
            'remember', 'remember_batch',
            'revise', 'revise_batch',
            'connect', 'connect_batch',
            'brain_batch',
            'recall', 'eval',
        ]
        for tool in should_be_mcp:
            self.assertIn(tool, self.mcp_tools,
                          f"Core tool '{tool}' not in MCP — Claude can't use it directly")

    def test_removed_tools_gone(self):
        """Deprecated tools should not be in MCP."""
        removed = [
            'remember_lesson', 'remember_impact', 'remember_mechanism',
            'remember_convention', 'remember_uncertainty', 'remember_mental_model',
            'record_divergence', 'learn_vocabulary',
            'consciousness', 'dismiss_signal', 'queue_state',
        ]
        present = [t for t in removed if t in self.mcp_tools]
        self.assertEqual(present, [],
                         f"Removed tools still in MCP: {present}")

    def test_remember_tool_params_match_brain(self):
        """MCP remember tool params must be a subset of brain.remember() params."""
        if 'remember' not in self.mcp_params:
            self.skipTest("remember not found in MCP params")

        from servers import brain_remember
        for name in dir(brain_remember):
            cls = getattr(brain_remember, name)
            if inspect.isclass(cls):
                for mname, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    if mname == 'remember':
                        sig = inspect.signature(method)
                        brain_params = {k for k in sig.parameters.keys() if k != 'self'}
                        mcp_params = set(self.mcp_params.get('remember', []))
                        invalid = mcp_params - brain_params
                        self.assertEqual(invalid, set(),
                                         f"MCP remember has params not in brain: {invalid}")

    def test_connect_uses_ids_not_titles(self):
        """MCP connect tool must use source_id/target_id, not source_title/target_title."""
        if 'connect' in self.mcp_params:
            params = self.mcp_params['connect']
            self.assertNotIn('source_title', params,
                             "MCP connect uses source_title but brain needs source_id")
            self.assertNotIn('target_title', params,
                             "MCP connect uses target_title but brain needs target_id")


class TestDefaultsSync(unittest.TestCase):
    """Cross-layer: parameter defaults must agree."""

    def test_confidence_default_consistent(self):
        """remember() confidence default must match across brain, daemon, and MCP."""
        from servers import brain_remember
        brain_default = None
        for name in dir(brain_remember):
            cls = getattr(brain_remember, name)
            if inspect.isclass(cls):
                for mname, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    if mname == 'remember':
                        sig = inspect.signature(method)
                        if 'confidence' in sig.parameters:
                            brain_default = sig.parameters['confidence'].default

        # Daemon default
        dispatch_path = ROOT / "servers" / "daemon_dispatch.py"
        dispatch_content = dispatch_path.read_text()
        daemon_match = re.search(r'confidence.*?args\.get\(["\']confidence["\'],?\s*([\d.]+)', dispatch_content)
        daemon_default = float(daemon_match.group(1)) if daemon_match else None

        # MCP default — read from contract (single source of truth)
        from servers.contract import ALL_FIELDS
        mcp_default = ALL_FIELDS.get('confidence', {}).get('default')

        if brain_default is not None and daemon_default is not None:
            self.assertEqual(brain_default, daemon_default,
                             f"confidence default: brain={brain_default} vs daemon={daemon_default}")
        if brain_default is not None and mcp_default is not None:
            self.assertEqual(brain_default, mcp_default,
                             f"confidence default: brain={brain_default} vs MCP={mcp_default}")


class TestHookCommandSync(unittest.TestCase):
    """Layer 4: Hook scripts must only call commands that exist in daemon."""

    @classmethod
    def setUpClass(cls):
        hooks_dir = ROOT / "hooks" / "scripts"
        cls.hook_commands = {}
        if hooks_dir.exists():
            for script in hooks_dir.glob("*.py"):
                content = script.read_text()
                calls = re.findall(r'daemon_call_raw\(["\'](\w+)["\']', content)
                calls += re.findall(r'send_command\(["\'](\w+)["\']', content)
                if calls:
                    cls.hook_commands[script.name] = calls

    def test_hook_commands_exist_in_daemon(self):
        """Every command a hook calls must exist in daemon dispatch or hook table."""
        from servers.daemon_dispatch import COMMAND_TABLE
        daemon_cmds = set(COMMAND_TABLE.keys())

        for script, commands in self.hook_commands.items():
            for cmd in commands:
                valid = cmd in daemon_cmds or cmd.startswith('hook_')
                self.assertTrue(valid,
                                f"Hook {script} calls '{cmd}' which doesn't exist in daemon")


class TestScaleDispatchSync(unittest.TestCase):
    """Layer 5: Scale dispatch WRITE_COMMANDS must be a subset of daemon COMMAND_TABLE."""

    def test_write_commands_exist_in_daemon(self):
        from servers.scales.dispatch import WRITE_COMMANDS
        from servers.daemon_dispatch import COMMAND_TABLE
        daemon_cmds = set(COMMAND_TABLE.keys())

        missing = [cmd for cmd in WRITE_COMMANDS if cmd not in daemon_cmds]
        self.assertEqual(missing, [],
                         f"Scale WRITE_COMMANDS not in daemon: {missing}")


class TestConnectToDescriptionContract(unittest.TestCase):
    """The s1e prompt DELEGATES the edge vocabulary, the never-use rule, and
    sibling resolution to the connect_to tool description (C3 coupling,
    measured-deliberate). These strings are load-bearing: if one drifts, the
    prompt's pointer dereferences to text that no longer says what it
    promises. Guardrail from the v-next.5 ship package."""

    def test_connect_to_load_bearing_strings(self):
        from servers.contract import CONNECT_TO_ITEM_SCHEMA
        relation_desc = CONNECT_TO_ITEM_SCHEMA['properties']['relation']['description']
        target_desc = CONNECT_TO_ITEM_SCHEMA['properties']['target']['description']
        # The never-use rule (the prompt's pointer target).
        self.assertIn('NEVER', relation_desc)
        self.assertIn('related_to', relation_desc)
        # A vocabulary list exists (spot-check two verbs the prompt cites).
        self.assertIn('grounds', relation_desc)
        self.assertIn('supersedes', relation_desc)
        # Sibling resolution lives on `target`, the one key the schema
        # offers; the old key `title` is read but never offered.
        self.assertIn('same batch may be named by its exact title', target_desc)
        self.assertIn('8-char hex id', target_desc)
        self.assertEqual(CONNECT_TO_ITEM_SCHEMA['required'], ['target'])
        self.assertNotIn('title', CONNECT_TO_ITEM_SCHEMA['properties'])

    def test_connect_to_description_admission_floor(self):
        from servers.contract import CONNECT_TO_ITEM_SCHEMA
        why_desc = CONNECT_TO_ITEM_SCHEMA['properties']['description']['description']
        self.assertIn('30', why_desc)

    def test_batch_connect_to_carries_double_emit_rule(self):
        from servers import brain_mcp
        tools = {t['name']: t for t in brain_mcp.TOOLS}
        rb = tools['remember_batch']['inputSchema']['properties']['nodes'][
            'items']['properties']['connect_to']['description']
        self.assertIn('DOUBLE-EMIT', rb)


class TestNodeIdInvariant(unittest.TestCase):
    """Every node id is exactly 8 lowercase hex chars. Exact-match id
    resolution at every door rests on this invariant; if the minting
    scheme ever changes width or alphabet, every exact-match lookup
    breaks at once. This is the loud tripwire."""

    def test_generated_node_ids_are_8_lowercase_hex(self):
        from servers.brain import Brain
        pat = re.compile(r'^[0-9a-f]{8}$')
        for _ in range(100):
            nid = Brain._generate_id(None)
            self.assertRegex(nid, pat)


if __name__ == '__main__':
    unittest.main()
