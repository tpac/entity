"""Contract tests for daemon instance keying (BRAIN_INSTANCE).

Locks the multi-entity boundary: with BRAIN_INSTANCE unset every rendezvous
path is byte-identical to the historical form (production unaffected); with it
set, every path and the launchd label carry the key, so an eval entity can
never acquire production's daemon/maintenance locks or target its launchd job.
Also locks the daemon address to loopback, and the deterministic background-loop triggers (s2_force /
drain_embeddings) that entity harnesses call instead of re-implementing the
daemon's loops.
"""
import ipaddress
import os
import re
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import daemon_config as dc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_FUNCS = [dc.get_pid_path, dc.get_lock_path,
              dc.get_startup_lock_path, dc.get_status_path,
              dc.get_maintenance_path, dc.get_recovery_state_path,
              dc.get_mcp_crash_path]


class TestInstancePaths(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("BRAIN_INSTANCE")
        os.environ.pop("BRAIN_INSTANCE", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("BRAIN_INSTANCE", None)
        else:
            os.environ["BRAIN_INSTANCE"] = self._saved

    def test_no_instance_paths_are_legacy_exact(self):
        uid = os.getuid()
        expected = ["/tmp/brain-daemon-%d.pid" % uid,
                    "/tmp/brain-daemon-%d.lock" % uid,
                    "/tmp/brain-startup-%d.lock" % uid,
                    "/tmp/brain-status-%d.json" % uid,
                    "/tmp/brain-maintenance-%d.lock" % uid,
                    "/tmp/brain-recovery-%d.json" % uid,
                    "/tmp/brain-mcp-crash-%d.txt" % uid]
        self.assertEqual([f() for f in PATH_FUNCS], expected,
                         "production paths must be byte-identical to the "
                         "pre-instance form")

    def test_instance_suffixes_every_path(self):
        os.environ["BRAIN_INSTANCE"] = "evalx"
        uid = os.getuid()
        # Exact ordered list — a substring/set check would survive a mutant
        # where two rendezvous concerns collapse onto one inode (the "one
        # inode cannot answer both questions" bug get_startup_lock_path's
        # docstring records).
        expected = ["/tmp/brain-daemon-%d-evalx.pid" % uid,
                    "/tmp/brain-daemon-%d-evalx.lock" % uid,
                    "/tmp/brain-startup-%d-evalx.lock" % uid,
                    "/tmp/brain-status-%d-evalx.json" % uid,
                    "/tmp/brain-maintenance-%d-evalx.lock" % uid,
                    "/tmp/brain-recovery-%d-evalx.json" % uid,
                    "/tmp/brain-mcp-crash-%d-evalx.txt" % uid]
        instance_paths = [f() for f in PATH_FUNCS]
        self.assertEqual(instance_paths, expected)
        os.environ.pop("BRAIN_INSTANCE")
        self.assertTrue(set(expected).isdisjoint({f() for f in PATH_FUNCS}),
                        "instance paths must never collide with production's")

    def test_boot_hook_reads_the_crash_file_the_mcp_server_writes(self):
        """boot-brain.sh builds the crash path in shell; it must land on
        get_mcp_crash_path() with and without an instance."""
        with open(os.path.join(REPO, "hooks", "scripts", "boot-brain.sh")) as f:
            line = next(l for l in f if l.strip().startswith("MCP_CRASH_FILE="))
        for inst in ("", "evalx"):
            env = {k: v for k, v in os.environ.items() if k != "BRAIN_INSTANCE"}
            if inst:
                env["BRAIN_INSTANCE"] = inst
                os.environ["BRAIN_INSTANCE"] = inst
            else:
                os.environ.pop("BRAIN_INSTANCE", None)
            shell = subprocess.run(["bash", "-c", line.strip() + '; printf %s "$MCP_CRASH_FILE"'],
                                   env=env, capture_output=True, text=True, check=True).stdout
            self.assertEqual(shell, dc.get_mcp_crash_path())

    def test_db_lock_path_keyed_on_realpath_not_instance(self):
        os.environ["BRAIN_INSTANCE"] = "evalx"
        a = dc.get_db_lock_path("/tmp/some-brain/brain.db")
        os.environ.pop("BRAIN_INSTANCE")
        b = dc.get_db_lock_path("/tmp/some-brain/brain.db")
        self.assertEqual(a, b, "the DB writer lock is keyed on the brain, not "
                               "the label — that is its entire point")
        self.assertNotEqual(a, dc.get_db_lock_path("/tmp/other-brain/brain.db"))


class TestInstanceImportGuard(unittest.TestCase):
    """The guard fires at import — needs a fresh interpreter per case."""

    def _import_probe(self, env_extra, code="import servers.daemon_config"):
        env = {k: v for k, v in os.environ.items()
               if k not in ("BRAIN_INSTANCE", "BRAIN_DAEMON_PORT",
                            "BRAIN_DB_DIR")}
        env.update(env_extra)
        return subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, %r); %s"
             % (REPO, code)],
            env=env, capture_output=True, text=True, cwd=REPO)

    def test_instance_without_port_and_db_refuses(self):
        r = self._import_probe({"BRAIN_INSTANCE": "evalx"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requires explicit", r.stderr)

    def test_bad_instance_name_refuses(self):
        r = self._import_probe({"BRAIN_INSTANCE": "eval x",
                                "BRAIN_DAEMON_PORT": "47999",
                                "BRAIN_DB_DIR": "/tmp/x"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("1-32 chars", r.stderr)

    def test_production_port_value_refuses(self):
        # brain-env.sh fills the uid-formula port in when unset, so presence
        # alone cannot prove the operator chose a port — the VALUE must differ.
        prod_port = str(47200 + (os.getuid() % 100))
        r = self._import_probe({"BRAIN_INSTANCE": "evalx",
                                "BRAIN_DAEMON_PORT": prod_port,
                                "BRAIN_DB_DIR": "/tmp/x",
                                # keep the probe hermetic from the user env file
                                "XDG_CONFIG_HOME": "/tmp/nonexistent-xdg"})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("production's port", r.stderr)

    def test_hyphenated_instance_label(self):
        r = self._import_probe(
            {"BRAIN_INSTANCE": "eval-a", "BRAIN_DAEMON_PORT": "47999",
             "BRAIN_DB_DIR": "/tmp/x"},
            code="from servers.daemon_launch import LAUNCHD_LABEL; "
                 "print(LAUNCHD_LABEL)")
        self.assertEqual(r.returncode, 0, r.stderr)
        # Only the leading suffix dash becomes a dot; inner dashes are legal
        # launchd label characters and must survive.
        self.assertEqual(r.stdout.strip(), "com.brain.daemon.eval-a")

    def test_instance_with_port_and_db_imports_and_keys_label(self):
        r = self._import_probe(
            {"BRAIN_INSTANCE": "evalx", "BRAIN_DAEMON_PORT": "47999",
             "BRAIN_DB_DIR": "/tmp/x"},
            code="from servers.daemon_launch import LAUNCHD_LABEL; "
                 "print(LAUNCHD_LABEL)")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "com.brain.daemon.evalx")

    def test_production_label_unchanged(self):
        r = self._import_probe(
            {}, code="from servers.daemon_launch import LAUNCHD_LABEL; "
                     "print(LAUNCHD_LABEL)")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "com.brain.daemon")


class TestDaemonAddressIsLoopback(unittest.TestCase):
    """The daemon has no access control of its own (the hook HMAC is
    attribution checked in the MCP proxy; `eval` runs code), so binding
    loopback is the only thing keeping other hosts off the port."""

    def test_host_is_a_literal_ipv4_loopback_address(self):
        # A literal, not a name: "localhost" can resolve to ::1 on macOS while
        # the server is AF_INET-only. Not "" or "0.0.0.0": that is every
        # interface.
        try:
            addr = ipaddress.ip_address(dc.DAEMON_HOST)
        except ValueError:
            self.fail("DAEMON_HOST=%r is not a literal IP address" % dc.DAEMON_HOST)
        self.assertEqual(addr.version, 4)
        self.assertTrue(addr.is_loopback, dc.DAEMON_HOST)
        self.assertEqual(dc.get_daemon_addr()[0], dc.DAEMON_HOST)

    def test_server_bind_lands_on_loopback(self):
        from servers import daemon_server as ds
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)
        d.server_socket = None
        d._log = lambda m: None
        d.daemon_addr = (dc.get_daemon_addr()[0], 0)  # the owner's host, an ephemeral port
        d._bind_socket()
        try:
            bound = d.server_socket.getsockname()[0]
        finally:
            d._close_socket()
        self.assertTrue(ipaddress.ip_address(bound).is_loopback,
                        "daemon listens on %s — reachable beyond this host" % bound)

    def test_post_tool_trace_connects_to_the_owner_host(self):
        # The hook copies the host instead of importing daemon_config (why: the
        # raw-socket row in tests/test_deploy_contract.py, which also keeps the
        # set of socket owners from growing unnoticed). The dashboard's copy is
        # pinned with its other mirrors in tests/test_dashboard_disconnection.py.
        with open(os.path.join(REPO, "hooks", "scripts", "post_tool_trace.py")) as f:
            hosts = re.findall(r"\.connect\(\(\s*['\"]([^'\"]*)['\"]", f.read())
        self.assertEqual(hosts, [dc.DAEMON_HOST])


class TestBackgroundLoopTriggers(unittest.TestCase):
    def test_commands_registered_lock_free_and_wired(self):
        from servers.daemon_dispatch import COMMAND_TABLE
        from servers import dispatch_ops
        # Identity, not just registration — a swapped wiring would return
        # ok:True while running the wrong loop (an "S2 arm" that never ran).
        self.assertIs(COMMAND_TABLE["s2_force"].handler,
                      dispatch_ops._handle_s2_force)
        self.assertIs(COMMAND_TABLE["drain_embeddings"].handler,
                      dispatch_ops._handle_drain_embeddings)
        for cmd in ("s2_force", "drain_embeddings"):
            # Both own their serialization internally (run_s2 single-flight;
            # drain batches take brain.write_lock) — holding the dispatch
            # exclusive lock for a multi-minute S2 cycle would deadlock S2's
            # own encoder dispatch.
            self.assertFalse(COMMAND_TABLE[cmd].is_write, cmd)

    def test_s2_force_passes_units_through_to_the_door(self):
        """`units` rides dispatch_command → _handle_s2_force → brain.run_s2
        unchanged; omitted — or args null on the wire — the door gets None
        (every unit). A forced cycle is marked as one (force=True), so a unit
        that batches work runs whatever is waiting."""
        from unittest.mock import MagicMock
        from servers.daemon_dispatch import dispatch_command
        for args, expected in (({"units": ["aspect_integration"]},
                                ["aspect_integration"]),
                               ({}, None), (None, None)):
            with self.subTest(args=args):
                brain = MagicMock()
                brain.run_s2.return_value = {"units": {}, "elapsed_ms": 0}
                out = dispatch_command(brain, "s2_force", args, None)
                self.assertTrue(out["ok"])
                brain.run_s2.assert_called_once_with(units=expected, force=True)

    def test_drain_now_runs_the_tick_under_the_lock(self):
        """drain_now must run the SAME stage function the worker runs, while
        holding _drain_busy — the lock that keeps two threads from
        interleaving transactions on conn_bg_writer."""
        from servers import embed_queue
        seen = {}

        def fake_stages(brain, rwq):
            seen["locked"] = embed_queue._drain_busy.locked()
            seen["brain"] = brain

        orig = embed_queue._worker_tick_stages
        embed_queue._worker_tick_stages = fake_stages
        try:
            sentinel = object()
            result = embed_queue.drain_now(sentinel)
        finally:
            embed_queue._worker_tick_stages = orig
        self.assertTrue(seen["locked"], "stages must run under _drain_busy")
        self.assertIs(seen["brain"], sentinel)
        self.assertIn("pending_after", result)
        self.assertIn("embedder_ready", result)

    def test_drain_now_reports_busy_instead_of_blocking_forever(self):
        from servers import embed_queue
        embed_queue._drain_busy.acquire()
        old_timeout = embed_queue.DRAIN_NOW_TIMEOUT_S
        embed_queue.DRAIN_NOW_TIMEOUT_S = 0.05
        try:
            result = embed_queue.drain_now(object())
        finally:
            embed_queue.DRAIN_NOW_TIMEOUT_S = old_timeout
            embed_queue._drain_busy.release()
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("busy"))


if __name__ == "__main__":
    unittest.main()
