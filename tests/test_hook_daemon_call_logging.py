"""daemon_call_raw observability contract.

Regression guard for the silent-failure class: a recall failure that surfaced
to Claude (via additionalContext) but never reached hook_errors, so it was
invisible in the dashboard and at boot. Two gaps caused it:

  1. The empty/garbled-read case let json.loads("") throw the cryptic
     "Expecting value: line 1 column 1 (char 0)".
  2. daemon_call_raw's catch-all `except` returned ok=false WITHOUT logging
     (unlike its timeout and ok=false siblings).

The fix lives in the shared daemon path (single source of truth — every hook
calls daemon_call_raw), so these tests exercise that one function with a fake
daemon and assert BOTH halves: a clear error string AND a hook_errors row.

Since step 6d the WIRE belongs to daemon_client.send_command and only the
observability belongs to daemon_call_raw, so the fake daemon is now pointed at
by steering the owner's address (get_daemon_addr) rather than module constants
on hook_common — which no longer keeps a host/port of its own.
"""
import os
import sys
import socket
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts'))
import hook_common


def _serve_once(handler):
    """Start a throwaway TCP server on an ephemeral port. `handler(conn)` runs
    for the single accepted connection. Returns (host, port, stop_fn)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()

    def run():
        try:
            conn, _ = srv.accept()
            try:
                conn.recv(4096)  # drain the client's request so its send completes
                handler(conn)
            finally:
                conn.close()
        except OSError:
            pass
        finally:
            srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return host, port, t


class TestDaemonCallRawLogging(unittest.TestCase):
    def setUp(self):
        # Point hook_common's logger at a temp brain_logs.db.
        self._tmp = tempfile.mkdtemp()
        self._saved_db_dir = hook_common.db_dir
        hook_common.db_dir = self._tmp
        self._addr_patch = None

    def tearDown(self):
        hook_common.db_dir = self._saved_db_dir
        if self._addr_patch is not None:
            self._addr_patch.stop()

    def _point_at(self, host, port):
        """Aim the wire owner at the fake daemon for this test."""
        self._addr_patch = mock.patch("servers.daemon_client.get_daemon_addr",
                                      return_value=(host, port))
        self._addr_patch.start()

    def _hook_errors(self):
        db = os.path.join(self._tmp, "brain_logs.db")
        if not os.path.isfile(db):
            return []
        conn = sqlite3.connect(db)
        try:
            return conn.execute(
                "SELECT hook_name, error FROM hook_errors ORDER BY id").fetchall()
        finally:
            conn.close()

    def test_empty_response_returns_clear_error_and_logs(self):
        """Daemon accepts then closes without writing → clear message, logged."""
        host, port, _ = _serve_once(lambda conn: None)  # close without sending
        self._point_at(host, port)

        resp = hook_common.daemon_call_raw("probe_empty", timeout=3.0)

        self.assertFalse(resp.get("ok"))
        self.assertIn("empty response", resp.get("error", ""),
                      "empty read should yield a clear message, not a cryptic JSON error")
        rows = self._hook_errors()
        self.assertTrue(any(r[0] == "probe_empty" for r in rows),
                        "empty-response failure must be logged to hook_errors")

    def test_wire_failure_records_the_caller_stack(self):
        """A failure handed over as a RETURN VALUE has no sys.exc_info(), so
        these rows used to persist an empty traceback column. log_hook_error
        falls back to the CALLER stack — the part worth reading; the
        exception's own frames are one socket call the error string names."""
        host, port, _ = _serve_once(lambda conn: conn.sendall(b"not json\n"))
        self._point_at(host, port)

        hook_common.daemon_call_raw("probe_stack", timeout=3.0)

        db = os.path.join(self._tmp, "brain_logs.db")
        conn = sqlite3.connect(db)
        try:
            tb = conn.execute(
                "SELECT traceback FROM hook_errors WHERE hook_name = 'probe_stack'"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertTrue(tb.strip(), "wire failures must not persist an empty traceback")
        self.assertIn("daemon_call_raw", tb,
                      "the caller chain is what makes the row worth reading")

    def test_garbage_response_is_logged(self):
        """Non-JSON reply → JSON parse error caught by the catch-all AND logged."""
        host, port, _ = _serve_once(lambda conn: conn.sendall(b"not json at all\n"))
        self._point_at(host, port)

        resp = hook_common.daemon_call_raw("probe_garbage", timeout=3.0)

        self.assertFalse(resp.get("ok"))
        rows = self._hook_errors()
        self.assertTrue(any(r[0] == "probe_garbage" for r in rows),
                        "garbled-response failure must be logged to hook_errors")


class TestDaemonUnavailableLogging(unittest.TestCase):
    """daemon_unavailable_error must PERSIST the outage, not just relay it.

    The canonical daemon-down handler (every hook calls it) logged via
    log_hook_output — deprecated to a no-op (2026-04-03) — so a real
    daemon-down event left hook_errors empty: invisible in the dashboard, at
    boot, and to query_logs, even though the operator saw the CRITICAL relay.

    Two arms since b2aaa1f (BRAIN OFFLINE gate): an existing brain.db means
    a crashed daemon — relay + persist + recover; no brain.db means Anchor
    never came up — persist BRAIN OFFLINE + exit 1, no recovery.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._saved = (hook_common.db_dir, hook_common.db_path)
        hook_common.db_dir = self._tmp
        hook_common.db_path = os.path.join(self._tmp, "brain.db")

    def tearDown(self):
        hook_common.db_dir, hook_common.db_path = self._saved

    def _hook_errors(self):
        db = os.path.join(self._tmp, "brain_logs.db")
        if not os.path.isfile(db):
            return []
        conn = sqlite3.connect(db)
        try:
            return conn.execute(
                "SELECT hook_name, level, error FROM hook_errors ORDER BY id").fetchall()
        finally:
            conn.close()

    def test_outage_relayed_and_persisted(self):
        # A brain.db exists → this is a crashed daemon, the recovery arm.
        open(hook_common.db_path, "w").close()
        # Neutralize step 3 (recover_daemon) so the test never touches a real
        # daemon — daemon_unavailable_error imports it lazily at call time.
        with mock.patch("servers.daemon_client.recover_daemon", lambda *a, **k: None):
            msg = hook_common.daemon_unavailable_error("recall")

        # 1. still returns the CRITICAL relay Claude must surface to the operator
        self.assertIn("CRITICAL", msg)

        # 2. and now persists the outage to hook_errors (was a silent no-op before)
        db = os.path.join(self._tmp, "brain_logs.db")
        self.assertTrue(os.path.isfile(db), "daemon-down must create/write brain_logs.db")
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT hook_name, level, error FROM hook_errors ORDER BY id").fetchall()
        finally:
            conn.close()
        down = [r for r in rows if r[0] == "DAEMON_DOWN"]
        self.assertTrue(down, "daemon-down event must be logged to hook_errors")
        self.assertEqual(down[0][1], "critical", "daemon-down must log at level=critical")
        self.assertIn("recall", down[0][2], "the detecting hook should be named in the error")

    def test_unconfigured_install_logs_offline_and_exits(self):
        # No brain.db → BRAIN OFFLINE arm: persist a critical error, exit 1,
        # and never attempt recovery (boot creates brains, recovery must not).
        with mock.patch("servers.daemon_client.recover_daemon") as recover:
            with self.assertRaises(SystemExit) as cm:
                hook_common.daemon_unavailable_error("recall")
        self.assertEqual(cm.exception.code, 1)
        recover.assert_not_called()

        offline = [r for r in self._hook_errors() if "BRAIN OFFLINE" in (r[2] or "")]
        self.assertTrue(offline, "BRAIN OFFLINE must be persisted to hook_errors")
        self.assertEqual(offline[0][0], "recall", "the detecting hook is the error's hook_name")
        self.assertEqual(offline[0][1], "critical", "BRAIN OFFLINE must log at level=critical")


if __name__ == "__main__":
    unittest.main()
