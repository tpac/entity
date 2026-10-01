"""Contract: hook stdout is host-portable (Claude Code and Codex) and never gates.

Both hosts parse a hook's stdout against strict per-event JSON schemas. Codex
marks a run FAILED for `{"decision":"approve"}` on UserPromptSubmit / PreToolUse
and for plain text on Stop; on Claude Code that approve meant "allow, skip the
permission prompt" — a decision a memory plugin must not make. So "nothing to
say" is silence, a block is honored only on Stop (self-message delivery),
`hook_common.emit_hook_output` is the single writer of what the brain SAYS and
`hook_common.emit_updated_input` the single writer of the one tool input it
REWRITES (the caller-identity stamp on the brain's own MCP tools), and this
locks (1) the event-by-event mapping of both, (2) that they stay loud on shapes
they do not recognise, and (3) that no hook entrypoint — Python OR shell shim —
writes a decision or rewrite shape around them. Pure unit + file inspection —
no brain, no daemon.
"""
import contextlib
import glob
import io
import json
import os
import re
import sys
import unittest
from unittest import mock

_HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks", "scripts")
sys.path.insert(0, _HOOKS_DIR)
import hook_common  # noqa: E402

# Matches the approve decision in any spacing/quoting a script could print it:
# `{"decision":"approve"}`, `"decision": "approve"`, `'decision': 'approve'`.
_APPROVE_SHAPE = re.compile(r"""decision["']?\s*:\s*["']?approve""")


def _emit(event, payload):
    """Run the emitter with its sinks stubbed; return (stdout, logged_warnings)."""
    buf = io.StringIO()
    logged = []
    with mock.patch.object(hook_common, "brain_debug", lambda *_a, **_k: None), \
            mock.patch.object(hook_common, "log_hook_error",
                              lambda *a, **k: logged.append((a, k))), \
            contextlib.redirect_stdout(buf):
        hook_common.emit_hook_output(event, payload)
    return buf.getvalue(), logged


def _out(event, payload):
    return _emit(event, payload)[0]


def _context(out):
    return json.loads(out)["hookSpecificOutput"]["additionalContext"]


class TestEmitMapping(unittest.TestCase):
    def test_approve_is_silence(self):
        for event in ("UserPromptSubmit", "PreToolUse", "Stop", "SessionStart"):
            self.assertEqual(_out(event, {"decision": "approve"}), "", event)
            self.assertEqual(_out(event, {}), "", event)
            self.assertEqual(_out(event, None), "", event)

    def test_approve_with_reason_becomes_context(self):
        out = json.loads(_out("PreToolUse", {"decision": "approve", "reason": "mind the rule"}))
        self.assertEqual(out, {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "additionalContext": "mind the rule"}})

    def test_additional_context_payload(self):
        out = json.loads(_out("UserPromptSubmit", {"additionalContext": "recalled"}))
        self.assertEqual(out["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")
        self.assertEqual(out["hookSpecificOutput"]["additionalContext"], "recalled")
        self.assertNotIn("decision", out)

    def test_stop_block_is_the_delivery_form(self):
        out = json.loads(_out("Stop", {"decision": "block", "reason": "wait"}))
        self.assertEqual(out, {"decision": "block", "reason": "wait"})

    def test_stop_block_without_reason_still_carries_one(self):
        # Both hosts reject a block with an empty reason — the delivery must not
        # silently degrade into a no-op.
        out = json.loads(_out("Stop", {"decision": "block"}))
        self.assertEqual(out["decision"], "block")
        self.assertTrue(out["reason"].strip())

    def test_block_outside_stop_is_downgraded_and_logged(self):
        # The brain informs, it never gates: a block on a tool or prompt event
        # is a daemon-side mistake to surface, and its text still reaches the
        # model as context.
        for event in ("PreToolUse", "UserPromptSubmit"):
            out, logged = _emit(event, {"decision": "block", "reason": "careful"})
            self.assertEqual(len(logged), 1, event)
            self.assertIn("never gates", logged[0][0][1])
            self.assertEqual(_context(out), "careful", event)
            self.assertNotIn("permissionDecision", out)
            self.assertNotIn('"decision"', out)

    def test_stop_drops_context_loudly(self):
        # Codex's Stop schema has no additionalContext channel; the two-host
        # contract drops it — and says so in hook_errors.
        for payload in ({"decision": "approve", "reason": "(stored)"}, {"additionalContext": "x"}):
            out, logged = _emit("Stop", payload)
            self.assertEqual(out, "")
            self.assertEqual(len(logged), 1, payload)
            self.assertEqual(logged[0][1].get("level"), "warning")

    def test_unknown_decision_is_logged_and_informational(self):
        for bad in ("deny", "block ", "ALLOW"):
            out, logged = _emit("PreToolUse", {"decision": bad, "reason": "careful"})
            self.assertEqual(len(logged), 1, bad)
            self.assertIn("unknown hook decision", logged[0][0][1])
            self.assertEqual(_context(out), "careful", bad)

    def test_non_dict_payload_is_logged_not_swallowed(self):
        for bad in ("not a dict", ["a", "b"], 7):
            out, logged = _emit("UserPromptSubmit", bad)
            self.assertEqual(out, "")
            self.assertEqual(len(logged), 1, bad)
            self.assertIn("non-dict", logged[0][0][1])

    def test_non_string_fields_are_rendered_not_fatal(self):
        out = json.loads(_out("Stop", {"decision": "block", "reason": ["line one", "line two"]}))
        self.assertIn("line one", out["reason"])
        self.assertIn('"k"', _context(_out("PreToolUse", {"additionalContext": {"k": 1}})))

    def test_never_emits_approve_or_deny(self):
        for event in ("UserPromptSubmit", "PreToolUse", "Stop"):
            for payload in ({"decision": "approve"}, {"decision": "approve", "reason": "r"},
                            {"additionalContext": "c"}, {"decision": "block", "reason": "b"},
                            {"decision": "deny", "reason": "d"}):
                out = _out(event, payload)
                self.assertIsNone(_APPROVE_SHAPE.search(out), (event, payload))
                self.assertNotIn("permissionDecision", out, (event, payload))


def _emit_updated(event, updated, tool_name="mcp__brain__recall"):
    buf = io.StringIO()
    logged = []
    with mock.patch.object(hook_common, "log_hook_error",
                           lambda *a, **k: logged.append((a, k))), \
            contextlib.redirect_stdout(buf):
        hook_common.emit_updated_input(event, tool_name, updated)
    return buf.getvalue(), logged


class TestEmitUpdatedInput(unittest.TestCase):
    def test_pretooluse_rewrite_shape(self):
        # Both hosts: `allow` + `updatedInput` replaces the tool's arguments.
        out, logged = _emit_updated("PreToolUse", {"query": "q", "_caller_session": "s"})
        self.assertEqual(json.loads(out), {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {"query": "q", "_caller_session": "s"}}})
        self.assertEqual(logged, [])

    def test_only_pretooluse_carries_a_rewrite(self):
        for event in ("UserPromptSubmit", "PostToolUse", "Stop", "SessionStart"):
            out, logged = _emit_updated(event, {"a": 1})
            self.assertEqual(out, "", event)
            self.assertEqual(len(logged), 1, event)
            self.assertEqual(logged[0][1].get("level"), "warning", event)

    def test_only_brain_tools_are_rewritten(self):
        # The `allow` is the brain approving its OWN tools; a widened matcher
        # must not turn the writer into an auto-approve of a user's tool.
        for tool in ("Bash", "apply_patch", "mcp__github__create_issue", "mcp__brainy__x", "", None):
            out, logged = _emit_updated("PreToolUse", {"a": 1}, tool_name=tool)
            self.assertEqual(out, "", tool)
            self.assertEqual(len(logged), 1, tool)
            self.assertIn("refused", logged[0][0][1], tool)
        for tool in ("mcp__brain__recall", "mcp__plugin_x_brain__remember"):
            out, logged = _emit_updated("PreToolUse", {"a": 1}, tool_name=tool)
            self.assertIn('"updatedInput"', out, tool)
            self.assertEqual(logged, [], tool)

    def test_non_dict_input_is_logged_not_emitted(self):
        for bad in (None, "x", ["a"], 3):
            out, logged = _emit_updated("PreToolUse", bad)
            self.assertEqual(out, "", bad)
            self.assertEqual(len(logged), 1, bad)
            self.assertIn("non-dict", logged[0][0][1])


class TestStripCallerStamp(unittest.TestCase):
    def test_drops_only_the_reserved_pair(self):
        from servers.dispatch_common import CALLER_SESSION_KEY, CALLER_SIG_KEY
        # Both keys share the prefix the stripper keys on — the convention the
        # trace hook relies on without importing the owner.
        self.assertTrue(CALLER_SESSION_KEY.startswith("_caller_"))
        self.assertTrue(CALLER_SIG_KEY.startswith("_caller_"))
        stripped = hook_common.strip_caller_stamp(
            {"query": "q", "_x": 1, CALLER_SESSION_KEY: "s", CALLER_SIG_KEY: "0" * 64})
        self.assertEqual(stripped, {"query": "q", "_x": 1})

    def test_non_dict_passes_through(self):
        for v in (None, "patch text", ["a"], 7):
            self.assertEqual(hook_common.strip_caller_stamp(v), v)


class TestToolTargetFile(unittest.TestCase):
    def test_claude_code_shape(self):
        self.assertEqual(hook_common.tool_target_file({"file_path": "a/b.py"}), "a/b.py")

    def test_codex_apply_patch_shape(self):
        patch = "*** Begin Patch\n*** Update File: servers/brain.py\n@@\n-a\n+b\n*** End Patch"
        self.assertEqual(hook_common.tool_target_file({"command": patch}), "servers/brain.py")
        self.assertEqual(hook_common.tool_target_file({"command": "*** Add File: new.txt\n+x"}), "new.txt")

    def test_no_target(self):
        self.assertEqual(hook_common.tool_target_file({"command": "ls -la"}), "")
        self.assertEqual(hook_common.tool_target_file("nope"), "")


class TestScriptsRouteThroughEmitter(unittest.TestCase):
    # Every hook entrypoint — the bash shims Claude Code / Codex actually run
    # AND the Python bodies they exec — is in scope. Only the shared module that
    # implements the emitter may spell the shapes.
    _SHARED = {"hook_common.py"}

    @classmethod
    def setUpClass(cls):
        paths = glob.glob(os.path.join(_HOOKS_DIR, "*.py")) + glob.glob(os.path.join(_HOOKS_DIR, "*.sh"))
        cls.sources = {}
        for p in sorted(paths):
            name = os.path.basename(p)
            if name in cls._SHARED:
                continue
            with open(p) as f:
                cls.sources[name] = f.read()

    def test_scan_sees_both_shims_and_bodies(self):
        # Guard against a glob typo making the scan vacuous.
        self.assertIn("pre-bash-safety.sh", self.sources)
        self.assertIn("pre_bash_safety.py", self.sources)

    def test_no_script_prints_approve(self):
        for name, src in self.sources.items():
            self.assertIsNone(_APPROVE_SHAPE.search(src),
                              "%s emits an approve decision — exit 0 with no output instead" % name)
            self.assertNotIn("APPROVE", src, "%s keeps an APPROVE constant" % name)

    def test_no_script_hand_rolls_hook_json(self):
        # The hookSpecificOutput envelope, and the permission/rewrite keys inside
        # it, are composed in exactly one module. The needles are the quoted JSON
        # keys, so a docstring may still NAME the fields.
        for name, src in self.sources.items():
            for needle in ('"hookSpecificOutput"', '"permissionDecision"', '"updatedInput"'):
                self.assertNotIn(needle, src,
                                 "%s spells %s itself — route it through hook_common's emitters"
                                 % (name, needle))


if __name__ == "__main__":
    unittest.main()
