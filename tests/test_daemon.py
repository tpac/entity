#!/usr/bin/env python3
"""
Daemon Architecture Tests

Tests for the split daemon modules, including:
  - Integrity: dispatch table ↔ MCP tools ↔ hook table consistency
  - Concurrency: parallel reads don't block, writes serialize
  - Degradation: recall fallback flagging
  - Worktree: hooks.json has required hooks
  - Schema: encoding_version column exists and is set
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from servers.daemon_config import (
    get_pid_path, _code_fingerprint,
)
from servers.daemon_dispatch import COMMAND_TABLE, CmdEntry
from servers.daemon_client import (
    send_command, is_daemon_running, ensure_daemon, stop_daemon,
)
from servers.brain_mcp import TOOLS as MCP_TOOLS


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: Integrity — dispatch, MCP, hooks all in sync
# ══════════════════════════════════════════════════════════════════════════

class TestIntegrity(unittest.TestCase):
    """Verify that dispatch table, MCP tools, and hook table stay in sync.

    These tests catch the case where someone adds a Brain method,
    exposes it in one place, but forgets the others.
    """

    def test_all_mcp_tools_have_dispatch_entries(self):
        """Every MCP tool must have a matching daemon command."""
        mcp_names = {t['name'] for t in MCP_TOOLS}
        dispatch_names = set(COMMAND_TABLE.keys())
        # Commands handled directly by daemon_server before COMMAND_TABLE dispatch
        # (they end the serve loop — restart reloads in place, shutdown exits)
        daemon_direct = {'restart', 'shutdown'}

        for name in mcp_names:
            self.assertTrue(name in dispatch_names or name in daemon_direct,
                          f"MCP tool '{name}' has no dispatch entry in COMMAND_TABLE or daemon_direct")

    def test_dispatch_table_has_valid_entries(self):
        """Every dispatch entry must be a CmdEntry with callable handler."""
        for cmd, entry in COMMAND_TABLE.items():
            self.assertIsInstance(entry, CmdEntry,
                                 f"'{cmd}' is not a CmdEntry")
            self.assertTrue(callable(entry.handler),
                            f"'{cmd}' handler is not callable")
            self.assertIsInstance(entry.is_write, bool,
                                 f"'{cmd}' is_write must be bool")

    def test_read_commands_dont_mark_dirty(self):
        """Read commands should never mark dirty (they don't mutate state)."""
        for cmd, entry in COMMAND_TABLE.items():
            if not entry.is_write:
                self.assertFalse(entry.marks_dirty,
                                 f"Read command '{cmd}' marks_dirty=True — reads shouldn't mutate")

    def test_hook_table_matches_daemon_hooks_module(self):
        """Every hook in HOOK_TABLE must exist as a function in daemon_hooks.

        HOOK_TABLE entries are (is_write, marks_dirty); the daemon resolves
        the handler via getattr(_hooks, cmd), so the cmd name itself must
        be a callable attribute on daemon_hooks.
        """
        from servers.daemon_server import BrainDaemon
        import servers.daemon_hooks as hooks_module

        for hook_cmd, (is_write, marks_dirty) in BrainDaemon.HOOK_TABLE.items():
            self.assertIsInstance(is_write, bool,
                                  f"HOOK_TABLE['{hook_cmd}'][0] must be bool (is_write)")
            self.assertIsInstance(marks_dirty, bool,
                                  f"HOOK_TABLE['{hook_cmd}'][1] must be bool (marks_dirty)")
            self.assertTrue(hasattr(hooks_module, hook_cmd),
                            f"Hook '{hook_cmd}' has no matching function in daemon_hooks")
            self.assertTrue(callable(getattr(hooks_module, hook_cmd)),
                            f"'{hook_cmd}' in daemon_hooks is not callable")

    def test_mcp_tool_descriptions_not_empty(self):
        """Every MCP tool must have a non-empty description."""
        for tool in MCP_TOOLS:
            self.assertTrue(len(tool.get('description', '')) > 10,
                            f"MCP tool '{tool['name']}' has no/short description")

    def test_dispatch_covers_all_hook_commands(self):
        """Hook commands (hook_*) should NOT be in COMMAND_TABLE.
        They're routed via HOOK_TABLE in daemon_server."""
        from servers.daemon_server import BrainDaemon
        for cmd in COMMAND_TABLE:
            self.assertFalse(cmd.startswith('hook_'),
                             f"'{cmd}' is in COMMAND_TABLE but should be in HOOK_TABLE")
        for hook_cmd in BrainDaemon.HOOK_TABLE:
            self.assertNotIn(hook_cmd, COMMAND_TABLE,
                             f"'{hook_cmd}' is in both COMMAND_TABLE and HOOK_TABLE")


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: Worktree hooks integrity
# ══════════════════════════════════════════════════════════════════════════

class TestWorktreeHooks(unittest.TestCase):
    """Verify worktree hooks.json has required hooks for brain functionality."""

    REQUIRED_WORKTREE_HOOKS = {
        'SessionStart',       # Boot brain
        'UserPromptSubmit',   # Recall before responding — CRITICAL
        'PreToolUse',         # Pre-edit suggestions
    }

    def _load_worktree_hooks(self):
        """Load all worktree hooks.json files, normalized to event level.

        Worktree hooks.json mirrors the plugin format (`hooks/hooks.json`): a
        `{"description": ..., "hooks": {<event>: [...]}}` envelope. Unwrap the
        envelope so callers always see the event-level dict — matching how
        `test_main_settings_has_all_hooks` reads `hooks_config.get('hooks')`.
        A flat (already event-level) file is returned as-is for back-compat.
        """
        worktree_dir = os.path.join(PROJECT_ROOT, '.claude', 'worktrees')
        if not os.path.isdir(worktree_dir):
            return {}
        results = {}
        for wt in os.listdir(worktree_dir):
            hooks_path = os.path.join(worktree_dir, wt, 'hooks', 'hooks.json')
            if os.path.isfile(hooks_path):
                with open(hooks_path) as f:
                    config = json.load(f)
                # Unwrap the plugin envelope; no hook event is named "hooks",
                # so a dict-valued "hooks" key unambiguously marks the wrapper.
                inner = config.get('hooks')
                results[wt] = inner if isinstance(inner, dict) else config
        return results

    def test_worktree_hooks_have_required_events(self):
        """Every worktree hooks.json must include critical hook events."""
        worktrees = self._load_worktree_hooks()
        if not worktrees:
            self.skipTest("No worktree hooks.json found")

        for wt_name, hooks in worktrees.items():
            for event in self.REQUIRED_WORKTREE_HOOKS:
                self.assertIn(event, hooks,
                              f"Worktree '{wt_name}' missing required hook event: {event}")

    def test_main_settings_has_all_hooks(self):
        """Plugin hooks.json must have all hook events."""
        hooks_path = os.path.join(PROJECT_ROOT, 'hooks', 'hooks.json')
        if not os.path.isfile(hooks_path):
            self.skipTest("No hooks/hooks.json found")

        with open(hooks_path) as f:
            hooks_config = json.load(f)

        hooks = hooks_config.get('hooks', {})
        required_main = {
            'SessionStart', 'UserPromptSubmit', 'PreToolUse',
            'SessionEnd', 'Stop',
        }
        for event in required_main:
            self.assertIn(event, hooks,
                          f"hooks/hooks.json missing hook event: {event}")

    def test_worktree_hooks_reference_valid_scripts(self):
        """Hook commands in worktree hooks.json must reference scripts that
        exist IN THAT WORKTREE's own hooks/scripts/. Each worktree is a
        coherent checkout — validating stale sibling branches against main's
        scripts dir made every legitimate script deletion on main go red
        until all siblings merged (idle-maintenance.sh, 2026-08-18)."""
        worktrees = self._load_worktree_hooks()
        worktree_dir = os.path.join(PROJECT_ROOT, '.claude', 'worktrees')

        for wt_name, hooks in worktrees.items():
            scripts_dir = os.path.join(worktree_dir, wt_name, 'hooks', 'scripts')
            for event, entries in hooks.items():
                for entry in entries:
                    for hook in entry.get('hooks', []):
                        if hook.get('type') == 'command':
                            cmd = hook['command']
                            # Extract script name from "bash ${...}/scripts/foo.sh"
                            # or the quoted form `bash "${...}/scripts/foo.sh"`
                            # (98fe784 quoted all commands for spaced paths).
                            parts = cmd.split('/')
                            script_name = parts[-1].strip('"\'') if parts else ''
                            if script_name:
                                script_path = os.path.join(scripts_dir, script_name)
                                self.assertTrue(os.path.isfile(script_path),
                                                f"Worktree '{wt_name}' event '{event}' "
                                                f"references missing script: {script_name}")


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: Daemon module structure
# ══════════════════════════════════════════════════════════════════════════

class TestDaemonModuleStructure(unittest.TestCase):
    """Verify the split daemon modules are importable and properly structured."""

    def test_split_modules_export_all_public_symbols(self):
        """Split daemon modules must export all expected symbols in their respective homes."""
        from servers.daemon_server import BrainDaemon
        from servers.daemon_client import (send_command, is_daemon_running,
                                           ensure_daemon, stop_daemon)
        from servers.daemon_launch import (kickstart, manages, kill_daemon,
                                           port_is_occupied, spawn_detached_daemon)
        from servers.daemon_config import (get_pid_path,
                                           get_lock_path, get_startup_lock_path,
                                           get_status_path)
        from servers.daemon_dispatch import COMMAND_TABLE
        # Verify they're all importable and non-None
        for sym_name, sym in [
            ('BrainDaemon', BrainDaemon),
            ('send_command', send_command),
            ('is_daemon_running', is_daemon_running),
            ('ensure_daemon', ensure_daemon),
            ('stop_daemon', stop_daemon),
            ('kickstart', kickstart),
            ('manages', manages),
            ('kill_daemon', kill_daemon),
            ('port_is_occupied', port_is_occupied),
            ('spawn_detached_daemon', spawn_detached_daemon),
            ('get_pid_path', get_pid_path),
            ('get_lock_path', get_lock_path),
            ('get_startup_lock_path', get_startup_lock_path),
            ('get_status_path', get_status_path),
            ('COMMAND_TABLE', COMMAND_TABLE),
        ]:
            self.assertIsNotNone(sym, f"Split daemon modules missing symbol: {sym_name}")

    def test_daemon_dispatch_is_readable(self):
        """daemon_dispatch.py should stay under 1120 lines."""
        path = os.path.join(PROJECT_ROOT, 'servers', 'daemon_dispatch.py')
        with open(path) as f:
            lines = len(f.readlines())
        self.assertLess(lines, 1120,
                        f"daemon_dispatch.py is {lines} lines — should be <1120")

    def test_no_circular_imports(self):
        """Importing daemon modules in any order should not cause circular imports."""
        # These should all succeed without ImportError
        import importlib
        for mod_name in ['daemon_config', 'daemon_launch', 'daemon_dispatch',
                         'daemon_client', 'daemon_log', 'daemon_server']:
            mod = importlib.import_module(f'servers.{mod_name}')
            self.assertIsNotNone(mod)

    def test_code_fingerprint_changes_on_file_modification(self):
        """Code fingerprint should detect file changes."""
        fp1 = _code_fingerprint()
        self.assertNotEqual(fp1, "unknown")
        self.assertEqual(len(fp1), 16)  # MD5 hex truncated to 16

    def test_fingerprint_is_content_based_not_mtime(self):
        """The churn fix: fingerprint must be STABLE across mtime-only changes
        and SENSITIVE to content changes — so a worktree looks 'stale' to the
        daemon only when its code actually differs, not just because checkout
        gave its files fresh mtimes (the 2026-06-05 restart-churn root cause)."""
        import tempfile, shutil
        from servers.daemon_config import _fingerprint_dir
        d = tempfile.mkdtemp(prefix="brain-fp-test-")
        try:
            p = os.path.join(d, "a.py")
            with open(p, "w") as f:
                f.write("x = 1\n")
            fp1 = _fingerprint_dir(d)
            os.utime(p, (10**9, 10**9))   # mtime-only change
            self.assertEqual(_fingerprint_dir(d), fp1,
                             "mtime-only change must NOT change the fingerprint")
            with open(p, "w") as f:        # content change
                f.write("x = 2\n")
            fp2 = _fingerprint_dir(d)
            self.assertNotEqual(fp2, fp1,
                                "content change MUST change the fingerprint")
            # recursion: a .py added in a SUBPACKAGE must change the fingerprint
            os.makedirs(os.path.join(d, "sub"))
            with open(os.path.join(d, "sub", "s.py"), "w") as f:
                f.write("y = 1\n")
            self.assertNotEqual(_fingerprint_dir(d), fp2,
                                "subpackage .py change MUST change the fingerprint")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_worktree_checkout_detection(self):
        """A linked worktree has `.git` as a FILE (gitdir: pointer); the primary
        checkout has `.git` as a DIRECTORY; a tarball install has neither. This
        is the signal that suppresses staleness restarts for worktree sessions
        (2026-06-06 churn fix)."""
        import tempfile, shutil
        from servers.daemon_config import _is_worktree_checkout
        d = tempfile.mkdtemp(prefix="brain-wt-test-")
        try:
            os.makedirs(os.path.join(d, ".git"))            # primary checkout
            self.assertFalse(_is_worktree_checkout(d),
                             "primary checkout (.git dir) is NOT a worktree")
            shutil.rmtree(os.path.join(d, ".git"))
            with open(os.path.join(d, ".git"), "w") as f:    # linked worktree
                f.write("gitdir: /repo/.git/worktrees/wt\n")
            self.assertTrue(_is_worktree_checkout(d),
                            "linked worktree (.git file) IS a worktree")
            os.remove(os.path.join(d, ".git"))               # tarball install
            self.assertFalse(_is_worktree_checkout(d),
                             "missing .git → not a worktree (safe default)")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_code_changed_warranted_only_for_daemon_source(self):
        """_code_changed gates on the daemon's reported source_dir: a restart only
        converges if THIS checkout is the daemon's launch source, so a different
        checkout (worktree, second clone) never reports changed even on a
        fingerprint mismatch. Daemons that omit source_dir fall back to the
        linked-worktree heuristic. (2026-06-06 non-convergent churn fix.)"""
        import servers.daemon_client as dc
        from unittest import mock

        # launchd manages the daemon: the case this describes. Unpatched, the
        # superseded-tree exception asks the real launchctl about the label.
        with mock.patch.object(dc, "REPO_ROOT", "/repo/main"), \
             mock.patch.object(dc, "_CODE_FINGERPRINT", "FP_A"), \
             mock.patch.object(dc, "manages", return_value=True):
            # different source_dir → never restart, even with a fingerprint mismatch
            self.assertFalse(dc._code_changed(
                {"result": {"source_dir": "/repo/.wt/feat", "code_fingerprint": "FP_B"}}),
                "a checkout that isn't the daemon's source must never report changed")
            # same source_dir + mismatch → restart warranted
            self.assertTrue(dc._code_changed(
                {"result": {"source_dir": "/repo/main", "code_fingerprint": "FP_B"}}),
                "the daemon's own source running changed code IS code-changed")
            # same source_dir + matching fingerprint → no restart
            self.assertFalse(dc._code_changed(
                {"result": {"source_dir": "/repo/main", "code_fingerprint": "FP_A"}}),
                "same source and same fingerprint is not code-changed")
            # fallback (daemon omits source_dir): the worktree heuristic decides
            with mock.patch.object(dc, "_IS_WORKTREE", True):
                self.assertFalse(dc._code_changed({"result": {"code_fingerprint": "FP_B"}}),
                                 "old daemon + worktree caller falls back to suppression")
            with mock.patch.object(dc, "_IS_WORKTREE", False):
                self.assertTrue(dc._code_changed({"result": {"code_fingerprint": "FP_B"}}),
                                "old daemon + primary checkout + mismatch IS code-changed")


# ══════════════════════════════════════════════════════════════════════════
# TEST 7: Relevance floor
# ══════════════════════════════════════════════════════════════════════════

class TestVersionAwareFloor(unittest.TestCase):
    """Verify the relevance floor constant."""

    def test_floor_constant_exists(self):
        """The relevance floor must be defined and inside (0, 1)."""
        from servers.brain_constants import RELEVANCE_FLOOR_PRIMARY
        self.assertGreater(RELEVANCE_FLOOR_PRIMARY, 0.0)
        self.assertLess(RELEVANCE_FLOOR_PRIMARY, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# TEST 8: SKILL.md exists and is loadable
# ══════════════════════════════════════════════════════════════════════════

class TestSkillAvailability(unittest.TestCase):
    """Verify SKILL.md exists and contains critical sections."""

    def test_skill_md_exists(self):
        """SKILL.md must exist at the expected path."""
        skill_path = os.path.join(PROJECT_ROOT, 'skills', 'brain', 'SKILL.md')
        self.assertTrue(os.path.isfile(skill_path),
                        f"SKILL.md not found at {skill_path}")

    def test_skill_md_documents_core_tools(self):
        """SKILL.md mentions the core write/read tools by name.

        Was `test_skill_md_has_api_reference` which asserted SKILL.md is
        a tool catalog (every MCP tool name present). The current SKILL.md
        is intentionally identity-first prose, not a catalog —
        `consciousness` is a meta tool not used in normal flow and
        doesn't belong in the identity doc. Narrowed to the genuinely
        load-bearing names that define how Anchor uses the brain.
        """
        skill_path = os.path.join(PROJECT_ROOT, 'skills', 'brain', 'SKILL.md')
        with open(skill_path) as f:
            content = f.read()
        for tool in ['recall', 'remember', 'connect']:
            self.assertIn(tool, content,
                          f"SKILL.md should mention core tool '{tool}'")


if __name__ == '__main__':
    unittest.main()
