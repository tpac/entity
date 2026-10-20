"""Contract: the daemon notices an installed copy that runs different code.

Hooks and the MCP proxy stamp each request with their copy's build identity
(daemon_client.send_command `role=`); the daemon's door judges it
(dispatch_common.check_client_code). Locked here: the verdicts, the grace that
keeps a redeploy's copy-by-copy window quiet, one errors row per (session,
copy) and one notice per stale hook copy, the client-side stamps, and that the
door never lets the check cost a request. Isolated brains only.
"""
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "hooks", "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "hooks", "scripts"))

from tests.brain_test_base import BrainTestBase  # noqa: E402
from servers import daemon_config as dcfg  # noqa: E402
from servers.channels.thalamus import thalamus  # noqa: E402
from servers.dispatch_common import (  # noqa: E402
    CALLER_SESSION_KEY, ClientCodeWatch, check_client_code)

DAEMON_FP = "d" * 16
OLD_FP = "0" * 16
HOOKS = frozenset({"hook_recall", "hook_session_end", "context_boot", "reset_session"})
T0 = 1_000_000.0
AFTER = T0 + dcfg.STALE_CLIENT_GRACE_S


def _stamp(role=dcfg.CLIENT_ROLE_HOOK, fp=OLD_FP, root="/copies/old"):
    return {"role": role, "fp": fp, "root": root}


class TestStaleCopyRule(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.watch = ClientCodeWatch()

    def check(self, cmd="hook_recall", args=None, client=None, now=T0, daemon_fp=DAEMON_FP):
        return check_client_code(self.brain, cmd, args if args is not None else {"session_id": "s1"},
                                 client, self.watch, HOOKS, daemon_fp=daemon_fp, now=now)

    def errors(self):
        out = self.brain.query_logs(source="debug", level="error", limit=50)
        return [e for e in out.get("entries", []) if e.get("origin") == "daemon:client_check"]

    def notices(self):
        return [i for i in thalamus.list_items(self.brain, limit=50)["items"]
                if i["source"] == "daemon:client_check"]

    def test_matching_and_unjudgeable_clients_pass(self):
        for client in (_stamp(fp=DAEMON_FP), _stamp(fp="unknown"), _stamp(fp="")):
            self.check(client=client)
            self.assertIsNone(self.check(client=client, now=AFTER), client)
        # A dev tool that sends hook commands declares itself and is never judged.
        dev = _stamp(role=dcfg.CLIENT_ROLE_DEV)
        self.check(client=dev)
        self.assertIsNone(self.check(client=dev, now=AFTER))
        # No stamp on a command only dev tools and the proxy send: never judged.
        self.check(cmd="ping")
        self.assertIsNone(self.check(cmd="ping", now=AFTER))
        self.assertEqual(self.errors(), [])
        self.assertEqual(self.notices(), [])

    def test_stale_hook_copy_waits_out_the_grace_then_reports(self):
        self.assertIsNone(self.check(client=_stamp()))
        self.assertIsNone(self.check(client=_stamp(), now=AFTER - 1))
        self.assertEqual(self.errors(), [])
        verdict = self.check(client=_stamp(), now=AFTER)
        self.assertEqual(verdict, {"key": OLD_FP + "@/copies/old", "role": "hook",
                                   "session": "s1", "reported": True, "noticed": True})
        [row] = self.errors()
        self.assertIn("session=s1", json.dumps(row))
        self.assertIn(OLD_FP + "@/copies/old", json.dumps(row))
        [item] = self.notices()
        self.assertIn("/copies/old", item["body"])
        self.assertEqual(item["audience"], "every_session")

    def test_two_sessions_one_copy_two_rows_one_notice(self):
        self.check(client=_stamp())
        self.check(client=_stamp(), args={"session_id": "s1"}, now=AFTER)
        self.check(client=_stamp(), args={"session_id": "s1"}, now=AFTER + 5)
        self.check(client=_stamp(), args={"session_id": "s2"}, now=AFTER + 6)
        self.assertEqual(len(self.errors()), 2)
        self.assertEqual(len(self.notices()), 1)

    def test_notice_refreshes_while_the_copy_keeps_calling(self):
        self.check(client=_stamp())
        self.assertTrue(self.check(client=_stamp(), now=AFTER)["noticed"])
        self.assertFalse(self.check(client=_stamp(), now=AFTER + 60)["noticed"])
        later = AFTER + dcfg.STALE_CLIENT_NOTICE_REFRESH_S
        self.assertTrue(self.check(client=_stamp(), now=later)["noticed"])
        self.assertEqual(len(self.notices()), 1)  # same item, window extended

    def test_a_notice_that_fails_to_file_is_retried_on_the_next_call(self):
        self.check(client=_stamp())
        with mock.patch.object(thalamus, "file", side_effect=RuntimeError("db busy")):
            self.assertFalse(self.check(client=_stamp(), now=AFTER)["noticed"])
        self.assertTrue(self.check(client=_stamp(), now=AFTER + 1)["noticed"])
        self.assertEqual(len(self.notices()), 1)

    def test_nothing_is_judged_before_the_brain_loads(self):
        self.assertIsNone(check_client_code(None, "hook_recall", {}, _stamp(), self.watch,
                                            HOOKS, daemon_fp=DAEMON_FP, now=AFTER))
        self.assertEqual(self.watch.first_seen, {})

    def test_stale_proxy_gets_an_errors_row_and_no_notice(self):
        proxy = _stamp(role=dcfg.CLIENT_ROLE_MCP)
        args = {CALLER_SESSION_KEY: "s9"}
        self.check(cmd="recall", args=args, client=proxy)
        verdict = self.check(cmd="recall", args=args, client=proxy, now=AFTER)
        self.assertEqual((verdict["session"], verdict["noticed"]), ("s9", False))
        self.assertEqual(len(self.errors()), 1)
        self.assertEqual(self.notices(), [])

    def test_unstamped_hook_command_is_an_old_copy(self):
        # The Oct 9 upload's hook_recall, replayed: an `envelope` key, no
        # `pieces`, no build stamp.
        args = {"session_id": "s3", "prompt": "hi", "envelope": "user"}
        self.check(args=args)
        verdict = self.check(args=args, now=AFTER)
        self.assertEqual((verdict["key"], verdict["noticed"]), ("unstamped", True))
        [item] = self.notices()
        self.assertIn("older installed copy", item["body"])

    def test_unknown_daemon_fingerprint_switches_the_check_off_loudly_once(self):
        for now in (T0, AFTER, AFTER + 1):
            self.assertIsNone(self.check(client=_stamp(), now=now, daemon_fp="unknown"))
        warns = [e for e in self.brain.query_logs(source="debug", level="warning", limit=50)["entries"]
                 if e.get("origin") == "daemon:client_check"]
        self.assertEqual(len(warns), 1)
        self.assertEqual(self.errors(), [])


class _Capture:
    """A one-shot daemon stand-in on a free port: records the request line,
    answers ok."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.request = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        conn, _ = self.sock.accept()
        with conn:
            self.request = json.loads(conn.makefile().readline())
            conn.sendall(b'{"ok": true, "result": {}}\n')
        self.sock.close()


class TestClientStamps(unittest.TestCase):
    def test_send_command_stamps_only_when_given_a_role(self):
        from servers import daemon_client
        for role, expect in ((dcfg.CLIENT_ROLE_HOOK, True), (None, False)):
            cap = _Capture()
            daemon_client.send_command("ping", {"a": 1}, timeout=5, port=cap.port, role=role)
            cap.thread.join(5)
            self.assertEqual(cap.request["args"], {"a": 1})
            if expect:
                self.assertEqual(cap.request[dcfg.CLIENT_STAMP_KEY],
                                 {"role": role, "fp": dcfg._CODE_FINGERPRINT,
                                  "root": dcfg.REPO_ROOT})
            else:
                self.assertNotIn(dcfg.CLIENT_STAMP_KEY, cap.request)

    def test_hook_calls_carry_the_hook_role(self):
        import hook_common
        sent = []
        fake = lambda cmd, args=None, timeout=10.0, port=None, role=None: (  # noqa: E731
            sent.append(role) or {"ok": True, "result": {}})
        with mock.patch.object(hook_common, "exit_if_daemon_unreachable"), \
                mock.patch("servers.daemon_client.send_command", fake):
            hook_common.daemon_call("hook_recall", {})
            hook_common.daemon_call_raw("hook_recall", {})
        self.assertEqual(sent, [dcfg.CLIENT_ROLE_HOOK] * 2)

    def test_proxy_calls_carry_the_mcp_role(self):
        from servers import brain_mcp
        with mock.patch.object(brain_mcp, "send_command",
                               return_value={"ok": True, "result": {}}) as send:
            brain_mcp.daemon_send("ping")
        self.assertEqual(send.call_args.kwargs["role"], dcfg.CLIENT_ROLE_MCP)


class TestDaemonDoor(unittest.TestCase):
    """_handle_client hands the stamp to the check and serves the request
    whatever the check does; unknown top-level keys never reach dispatch."""

    def _serve(self, msg, check):
        from servers import daemon_server as ds
        with tempfile.TemporaryDirectory() as tmp:
            d = ds.BrainDaemon(os.path.join(tmp, "brain.db"))
        d._log = lambda *a, **k: None
        dispatched = []
        d._dispatch = lambda cmd, args: dispatched.append((cmd, args)) or {"ok": True, "result": "served"}
        a, b = socket.socketpair()
        with a, mock.patch.object(ds, "check_client_code", check):
            a.sendall((json.dumps(msg) + "\n").encode())
            d._handle_client(b)
            reply = json.loads(a.makefile().readline())
        d._pool.shutdown(wait=False)
        return reply, dispatched

    def test_stamp_reaches_the_check_and_the_request_is_served(self):
        seen = []
        stamp = _stamp()
        reply, dispatched = self._serve(
            {"cmd": "hook_session_end", "args": {"session_id": "s1"}, "client": stamp, "extra": 1},
            lambda brain, cmd, args, client, watch, hooks: seen.append((cmd, client, cmd in hooks)))
        self.assertEqual(seen, [("hook_session_end", stamp, True)])
        self.assertEqual(dispatched, [("hook_session_end", {"session_id": "s1"})])
        self.assertEqual(reply.get("result"), "served")

    def test_a_failing_check_never_costs_the_request(self):
        def boom(*a, **k):
            raise RuntimeError("check broke")
        reply, dispatched = self._serve({"cmd": "ping", "args": {}}, boom)
        self.assertEqual(dispatched, [("ping", {})])
        self.assertEqual(reply.get("result"), "served")


if __name__ == "__main__":
    unittest.main()
