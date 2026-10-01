"""Contract: brain tool calls stay attributable on a host whose MCP proxy gets
no session identity.

Codex passes stdio MCP servers no thread id, so the PreToolUse hook
(hooks/scripts/stamp_caller_session.py) signs the `session_id` it receives and
rewrites the tool input with `_caller_session` + `_caller_sig`; the proxy
(brain_mcp._stamp_caller_session) honors a `_caller_session` it did not write
only when the signature verifies. Locked here: (1) sign/verify and the secret's
provisioning in servers.dispatch_common, (2) the proxy rule — env wins, then a
verified stamp, else scrub and note once per reason, (3) the hook end to end:
its stdout, fed to the proxy, resolves to the same session, and (4) that the
proxy never forwards the signature. No brain, no daemon; every test signs under
a throwaway XDG_CONFIG_HOME, never the install's real secret.
"""
import hashlib
import hmac
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHIM = os.path.join(_ROOT, "hooks", "scripts", "stamp-caller-session.sh")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from servers import dispatch_common as dc  # noqa: E402
from servers import brain_mcp  # noqa: E402

SID = "01920000-aaaa-7bbb-8ccc-ddddeeeeffff"


class TestBrainToolPredicate(unittest.TestCase):
    def test_server_name_is_the_manifests_key(self):
        # A plugin .mcp.json lists servers at the top level; the Codex manifest
        # nests them under mcpServers. Both must register the proxy under the
        # name the tool-name predicate looks for.
        with open(os.path.join(_ROOT, ".mcp.json")) as f:
            self.assertIn(dc.BRAIN_MCP_SERVER, json.load(f), ".mcp.json")
        with open(os.path.join(_ROOT, ".codex-plugin", "plugin.json")) as f:
            self.assertIn(dc.BRAIN_MCP_SERVER, json.load(f)["mcpServers"], ".codex-plugin/plugin.json")

    def test_matches_both_hosts_namings_and_nothing_else(self):
        for name in ("mcp__brain__recall", "mcp__plugin_x_brain__recall",
                     "mcp__plugin_my-plugin2_brain__recall", "mcp__brain__self_send"):
            self.assertTrue(dc.is_brain_tool(name), name)
        # Somebody else's server whose name merely contains or ends in "brain",
        # a foreign tool called "brain", and non-MCP tools are all refused.
        for name in ("Bash", "apply_patch", "mcp__github__brain", "mcp__github__brain__issue",
                     "mcp__second_brain__recall", "mcp__brainy__x", "mcp__notbrain__x",
                     "brain__recall", "", None, 3):
            self.assertFalse(dc.is_brain_tool(name), name)

    def test_hook_path_never_imports_the_daemon_config(self):
        # The hook signs on every brain tool call; daemon_config fingerprints
        # servers/ at import (tens of ms). The config dir comes from the leaf
        # constants module, so the hot path must stay clear of daemon_config.
        code = ("import sys; from servers import dispatch_common as dc; dc.hook_secret_path(); "
                "print('servers.daemon_config' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True,
                             text=True, timeout=60, env={**os.environ, "XDG_CONFIG_HOME": self._tmp()})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False", "hook_secret_path pulled in servers.daemon_config")

    def _tmp(self):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


class _TmpSecret(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": self.tmp.name})
        self._env.start()
        # The test process runs inside a Claude Code session: drop its identity
        # so the env-wins branch is exercised only where a test sets it.
        os.environ.pop("CLAUDE_CODE_SESSION_ID", None)

    def tearDown(self):
        self._env.stop()
        self.tmp.cleanup()


class TestSignVerify(_TmpSecret):
    def test_secret_is_created_on_first_use_under_xdg_mode_0600(self):
        path = dc.hook_secret_path()
        self.assertTrue(path.startswith(self.tmp.name))
        self.assertFalse(os.path.exists(path))
        sig = dc.sign_caller_session(SID)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        with open(path) as f:
            secret = f.read().strip()
        self.assertEqual(len(secret), 64)
        int(secret, 16)  # hex
        self.assertEqual(sig, hmac.new(secret.encode(), SID.encode(), hashlib.sha256).hexdigest())

    def test_verify_roundtrip_and_tamper(self):
        sig = dc.sign_caller_session(SID)
        self.assertEqual(sig, dc.sign_caller_session(SID))
        self.assertTrue(dc.verify_caller_session(SID, sig))
        self.assertFalse(dc.verify_caller_session(SID + "x", sig))
        self.assertFalse(dc.verify_caller_session(SID, sig[:-1] + ("0" if sig[-1] != "0" else "1")))
        self.assertFalse(dc.verify_caller_session("", sig))
        self.assertFalse(dc.verify_caller_session(SID, None))
        self.assertFalse(dc.verify_caller_session(None, sig))
        self.assertFalse(dc.verify_caller_session(SID, 12))
        # Model-controlled text: a non-ASCII "signature" is False, not a TypeError.
        self.assertFalse(dc.verify_caller_session(SID, "é" * 64))
        self.assertFalse(dc.verify_caller_session(SID, ""))

    def test_existing_secret_is_reused_not_rotated(self):
        path = dc.hook_secret_path()
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write("ab" * 32 + "\n")
        self.assertEqual(dc.sign_caller_session(SID),
                         hmac.new(b"ab" * 32, SID.encode(), hashlib.sha256).hexdigest())
        with open(path) as f:
            self.assertEqual(f.read().strip(), "ab" * 32)

    def test_lost_create_race_reads_the_winner(self):
        # Hook and proxy both find no secret on a fresh install; the loser of
        # the publish must sign under the winner's secret, not its own. The
        # winner's file is complete before its name exists, so the loser can
        # never read a half-written key.
        path = dc.hook_secret_path()

        def racing_link(src, dst):
            with open(dst, "wb") as f:
                f.write(b"cd" * 32 + b"\n")
            raise FileExistsError(dst)

        with mock.patch.object(dc.os, "link", racing_link):
            self.assertEqual(dc._hook_secret(), b"cd" * 32)
        with open(path, "rb") as f:
            self.assertEqual(f.read().strip(), b"cd" * 32)
        self.assertEqual([n for n in os.listdir(os.path.dirname(path)) if n.endswith(".tmp")], [],
                         "the loser's temp file was not cleaned up")

    def test_empty_secret_file_is_loud(self):
        path = dc.hook_secret_path()
        os.makedirs(os.path.dirname(path))
        open(path, "w").close()
        with self.assertRaises(RuntimeError):
            dc.sign_caller_session(SID)


class TestProxyRule(_TmpSecret):
    """brain_mcp._stamp_caller_session: env wins → verified stamp → scrub."""

    def _stamped(self, sid=SID, **extra):
        args = {dc.CALLER_SESSION_KEY: sid, dc.CALLER_SIG_KEY: dc.sign_caller_session(sid)}
        args.update(extra)
        return args

    def test_env_wins_and_strips_the_signature(self):
        notes = []
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": "env-sess"}):
            out = brain_mcp._stamp_caller_session(self._stamped(q=1), note=notes.append)
        self.assertEqual(out, {dc.CALLER_SESSION_KEY: "env-sess", "q": 1})
        self.assertEqual(notes, [])

    def test_verified_stamp_is_accepted_without_the_signature(self):
        notes = []
        out = brain_mcp._stamp_caller_session(self._stamped(q=1), note=notes.append)
        self.assertEqual(out, {dc.CALLER_SESSION_KEY: SID, "q": 1})
        self.assertEqual(notes, [])

    def test_bad_signature_is_scrubbed_and_noted(self):
        notes = []
        args = self._stamped(q=1)
        args[dc.CALLER_SIG_KEY] = "0" * 64
        out = brain_mcp._stamp_caller_session(args, note=notes.append)
        self.assertEqual(out, {"q": 1})
        self.assertEqual(len(notes), 1)
        self.assertIn("a bad signature", notes[0])

    def test_unverifiable_stamp_degrades_to_unattributed(self):
        # A broken secret file (unreadable, empty) must cost attribution, never
        # the tool call: the proxy scrubs, notes the real error, and proceeds.
        notes = []
        with mock.patch.object(brain_mcp, "verify_caller_session",
                               side_effect=PermissionError("denied")):
            out = brain_mcp._stamp_caller_session(self._stamped(q=1), note=notes.append)
        self.assertEqual(out, {"q": 1})
        self.assertEqual(len(notes), 1)
        self.assertIn("could not verify", notes[0])
        self.assertIn("PermissionError", notes[0])

    def test_unsigned_claim_is_scrubbed_and_noted(self):
        notes = []
        out = brain_mcp._stamp_caller_session({dc.CALLER_SESSION_KEY: "forged", "q": 1},
                                              note=notes.append)
        self.assertEqual(out, {"q": 1})
        self.assertEqual(len(notes), 1)
        self.assertIn("no signature", notes[0])

    def test_headless_call_is_noted_and_otherwise_untouched(self):
        notes = []
        out = brain_mcp._stamp_caller_session({"q": 1}, note=notes.append)
        self.assertEqual(out, {"q": 1})
        self.assertEqual(len(notes), 1)
        self.assertIn("no identity source", notes[0])

    def test_without_note_the_rule_is_silent(self):
        self.assertEqual(brain_mcp._stamp_caller_session({dc.CALLER_SESSION_KEY: "forged"}), {})

    def test_session_id_filter_is_never_touched(self):
        # Identity ≠ filter: an explicit caller-supplied session_id survives
        # every branch, and the stamp never becomes one.
        out = brain_mcp._stamp_caller_session(self._stamped(session_id="other"))
        self.assertEqual(out["session_id"], "other")
        self.assertEqual(out[dc.CALLER_SESSION_KEY], SID)
        out = brain_mcp._stamp_caller_session({"session_id": "other", dc.CALLER_SESSION_KEY: "forged"})
        self.assertEqual(out, {"session_id": "other"})

    def test_identity_gap_is_logged_once_per_reason(self):
        logged = []
        with mock.patch.object(brain_mcp, "_log_proxy_error",
                               lambda *a, **k: logged.append((a, k))), \
                mock.patch.object(brain_mcp, "_noted_identity_gaps", set()):
            brain_mcp._note_identity_gap("reason A")
            brain_mcp._note_identity_gap("reason A")
            brain_mcp._note_identity_gap("reason B")
        self.assertEqual([a[1] for a, _k in logged], ["reason A", "reason B"])
        self.assertEqual({a[0] for a, _k in logged}, {"mcp_caller_identity"})
        self.assertEqual({k.get("level") for _a, k in logged}, {"warning"})


class TestHookEndToEnd(_TmpSecret):
    """The shim Codex actually runs, fed Codex-shaped stdin, judged by the proxy."""

    def setUp(self):
        super().setUp()
        self.db_dir = os.path.join(self.tmp.name, "brain")
        os.makedirs(self.db_dir)
        open(os.path.join(self.db_dir, "brain.db"), "w").close()

    def _run(self, payload):
        env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_SESSION_ID"}
        env["BRAIN_DB_DIR"] = self.db_dir
        proc = subprocess.run(["bash", _SHIM], input=json.dumps(payload), env=env,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stdout

    def test_stamp_reaches_the_proxy_as_the_same_session(self):
        out = self._run({"session_id": SID, "hook_event_name": "PreToolUse",
                         "tool_name": "mcp__brain__recall",
                         "tool_input": {"query": "x", "limit": 3}})
        spec = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(spec["hookEventName"], "PreToolUse")
        self.assertEqual(spec["permissionDecision"], "allow")
        updated = spec["updatedInput"]
        self.assertEqual(updated["query"], "x")
        self.assertEqual(updated["limit"], 3)
        self.assertEqual(updated[dc.CALLER_SESSION_KEY], SID)
        self.assertTrue(dc.verify_caller_session(SID, updated[dc.CALLER_SIG_KEY]))
        # The contract itself: what the hook wrote is what the proxy accepts,
        # with the signature stripped before dispatch.
        notes = []
        args = brain_mcp._stamp_caller_session(dict(updated), note=notes.append)
        self.assertEqual(args, {"query": "x", "limit": 3, dc.CALLER_SESSION_KEY: SID})
        self.assertEqual(notes, [])

    def _warnings(self):
        conn = sqlite3.connect(os.path.join(self.db_dir, "brain_logs.db"))
        try:
            rows = conn.execute("SELECT hook_name, level, error FROM hook_errors").fetchall()
        finally:
            conn.close()
        self.assertEqual({r[0] for r in rows}, {"stamp_caller_session"}, rows)
        self.assertEqual({r[1] for r in rows}, {"warning"}, rows)
        return [r[2] for r in rows]

    def test_local_tool_is_never_rewritten_and_the_matcher_gap_is_loud(self):
        # A matcher wide enough to reach a user's tool is a manifest bug worth
        # hearing about; the tool itself proceeds untouched.
        self.assertEqual(self._run({"session_id": SID, "tool_name": "Bash",
                                    "tool_input": {"command": "ls"}}), "")
        self.assertEqual(self._run({"session_id": SID, "tool_name": "mcp__github__create_issue",
                                    "tool_input": {"title": "t"}}), "")
        errors = self._warnings()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(all("refused" in e for e in errors), errors)

    def test_missing_session_is_silent_and_logged(self):
        out = self._run({"tool_name": "mcp__brain__recall", "tool_input": {"query": "x"}})
        self.assertEqual(out, "")
        errors = self._warnings()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("session_id", errors[0])
        self.assertIn("unattributed", errors[0])

    def test_missing_tool_input_leaves_the_call_intact(self):
        # updatedInput REPLACES the arguments — a payload without them must
        # not be rewritten into a stamp-only call that erases the model's args.
        for payload in ({"session_id": SID, "tool_name": "mcp__brain__remember"},
                        {"session_id": SID, "tool_name": "mcp__brain__remember", "tool_input": None},
                        {"session_id": SID, "tool_name": "mcp__brain__remember", "tool_input": "x"}):
            self.assertEqual(self._run(payload), "", payload)
        errors = self._warnings()
        self.assertEqual(len(errors), 3, errors)
        self.assertTrue(all("tool_input" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()
