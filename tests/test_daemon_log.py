#!/usr/bin/env python3
"""What the daemon writes to daemon.log — the THREADS alarm, the line stamp, and
the rotation that bounds the file's history.

Pinned here: the bounded pool never trips the THREADS alarm (its workers are
non-daemon by design); every line reaching the log carries exactly one dated
stamp, whichever path wrote it; and rotation keeps at most keep_days of history
without losing a line across the switch.
"""

import io
import os
import shutil
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import servers.daemon_log as dlog
import servers.daemon_server as ds
from servers.daemon_config import POOL_THREAD_PREFIX, THREAD_ALARM_MAX_OTHERS

# The daemon's own named threads, as a healthy production inventory shows them.
_DAEMON_THREADS = ("embed-queue-drain", "autosave", "memory-watchdog",
                   "db-maintenance", "keepalive", "s1e")


class TestThreadAlarm(unittest.TestCase):
    """Real threads, parked on an event — is_alive(), daemon and name are the
    attributes the check reads, so fakes would test the fakes."""

    def setUp(self):
        self._release = threading.Event()
        self._threads = []
        self._pools = []

    def tearDown(self):
        self._release.set()
        for t in self._threads:
            t.join(timeout=5)
        for p in self._pools:
            p.shutdown(wait=True)

    def _thread(self, name, daemon):
        t = threading.Thread(target=self._release.wait, name=name, daemon=daemon)
        t.start()
        self._threads.append(t)
        return t

    def _pool(self, n, busy_cmd=None):
        """n live pool workers, named the way the daemon's pool names them.
        busy_cmd renames one as _handle_client does mid-request."""
        pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix=POOL_THREAD_PREFIX)
        self._pools.append(pool)
        workers, lock = [], threading.Lock()
        started = threading.Barrier(n + 1)

        def park():
            with lock:
                workers.append(threading.current_thread())
            started.wait()
            self._release.wait()

        for _ in range(n):
            pool.submit(park)
        started.wait(timeout=5)
        if busy_cmd:
            workers[0].name = "%s:%s" % (POOL_THREAD_PREFIX, busy_cmd)
        return workers

    def _healthy(self, pool_size=6):
        return ([threading.main_thread()]
                + self._pool(pool_size, busy_cmd="hook_recall")
                + [self._thread(n, daemon=True) for n in _DAEMON_THREADS])

    def test_healthy_inventory_logs_nothing(self):
        self.assertIsNone(ds._thread_alarm(self._healthy()))

    def test_large_pool_is_not_a_runaway(self):
        # THREAD_POOL_SIZE is cpu_count // 2: a 32-core host runs 16 workers,
        # past the limit on its own. The pool is bounded; only the rest counts.
        self.assertIsNone(ds._thread_alarm(self._healthy(pool_size=16)))

    def test_unexpected_non_daemon_thread_fires(self):
        threads = self._healthy() + [self._thread("leaky", daemon=False)]
        line = ds._thread_alarm(threads)
        self.assertIsNotNone(line)
        self.assertIn("1 unexpected non-daemon [leaky]", line)
        self.assertNotIn(POOL_THREAD_PREFIX, line.split("outside the pool:")[1],
                         "pool workers are not listed among the rest")

    def test_too_many_threads_outside_the_pool_fires(self):
        threads = ([threading.main_thread()] + self._pool(2)
                   + [self._thread("spawned-%d" % i, daemon=True)
                      for i in range(THREAD_ALARM_MAX_OTHERS + 1)])
        line = ds._thread_alarm(threads)
        self.assertIsNotNone(line)
        self.assertIn("%d threads outside the pool (limit %d)"
                      % (THREAD_ALARM_MAX_OTHERS + 1, THREAD_ALARM_MAX_OTHERS), line)
        self.assertNotIn("unexpected non-daemon", line)


class TestStampedStream(unittest.TestCase):
    """Every line reaching daemon.log starts with one iso_now() stamp."""

    def _stream(self):
        buf = io.StringIO()
        return dlog.StampedStream(buf), buf

    def test_each_line_of_a_write_is_stamped_once(self):
        s, buf = self._stream()
        with patch.object(dlog, "iso_now", return_value="T"):
            s.write("CRASH: boom\nTraceback line\n")
        self.assertEqual(buf.getvalue(), "[T] CRASH: boom\n[T] Traceback line\n")

    def test_print_partial_and_writelines_stamp_at_line_start_only(self):
        # print() writes the text and the newline as two calls.
        s, buf = self._stream()
        with patch.object(dlog, "iso_now", return_value="T"):
            print("[embed_queue] trace_embed 1", file=s)
            s.write("abc")
            s.write("def\n")
            s.writelines(["one\n", "two\n"])
        self.assertEqual(buf.getvalue(),
                         "[T] [embed_queue] trace_embed 1\n[T] abcdef\n"
                         "[T] one\n[T] two\n")

    def test_write_returns_the_input_length(self):
        s, _ = self._stream()
        self.assertEqual(s.write("abc\n"), 4)

    def test_stream_attributes_pass_through(self):
        # faulthandler and subprocess need the real fd.
        with tempfile.TemporaryFile("w") as f:
            self.assertEqual(dlog.StampedStream(f).fileno(), f.fileno())

    def test_main_stamps_stdout_and_stderr(self):
        with patch.object(sys, "stdout", io.StringIO()), \
             patch.object(sys, "stderr", io.StringIO()), \
             patch.object(ds, "BrainDaemon") as daemon:
            self.assertEqual(ds.main(["brain.db"]), 0)
            daemon.return_value.start.assert_called_once()
            self.assertIsInstance(sys.stdout, dlog.StampedStream)
            self.assertIsInstance(sys.stderr, dlog.StampedStream)


class TestDaemonLogRotation(unittest.TestCase):
    """daemon.log retention. A spare fd opened O_APPEND on the log stands in
    for launchd's fds 1 and 2, so dup2 is exercised for real without touching
    the test process's own stdout/stderr."""

    NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.path = os.path.join(self._dir, "daemon.log")
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def tearDown(self):
        os.close(self.fd)
        shutil.rmtree(self._dir, ignore_errors=True)

    def _write(self, text):
        os.write(self.fd, text.encode())

    def _rotate(self, keep_days=60, now=None):
        return dlog.rotate_if_due(self.path, keep_days, fds=(self.fd,),
                                     now=now or self.NOW)

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def _stamped(self, days_ago, msg):
        return "[%s] %s\n" % ((self.NOW - timedelta(days=days_ago)).isoformat(), msg)

    def test_log_younger_than_half_keep_days_is_left_alone(self):
        self._write(self._stamped(29, "[brain-daemon] Daemon started"))
        self.assertEqual(self._rotate(), {'rotated': False, 'age_days': 29.0})
        self.assertFalse(os.path.exists(self.path + ".1"))

    def test_due_log_rotates_and_the_fd_follows_the_fresh_file(self):
        old = self._stamped(31, "[brain-daemon] Daemon started")
        self._write(old)
        result = self._rotate()
        self.assertTrue(result['rotated'])
        self.assertEqual(self._read(self.path + ".1"), old)
        self._write("after the switch\n")
        fresh = self._read(self.path)
        self.assertRegex(fresh.splitlines()[0],
                         r"^\[\d{4}-\d\d-\d\dT[^\]]+\] \[brain-daemon\] daemon\.log rotated")
        self.assertTrue(fresh.endswith("after the switch\n"),
                        "writes after the rotation reach the fresh file")
        self.assertEqual(self._read(self.path + ".1"), old,
                         "nothing new lands in the backup")
        self.assertFalse(os.path.exists(self.path + ".new"))

    def test_failed_rotation_leaves_the_log_as_it_was(self):
        old = self._stamped(31, "previous month")
        self._write(old)
        with patch.object(dlog.os, "write", side_effect=OSError(28, "No space left")):
            with self.assertRaises(OSError):
                self._rotate()
        self.assertFalse(os.path.exists(self.path + ".1"))
        self._write("still here\n")
        self.assertEqual(self._read(self.path), old + "still here\n",
                         "the fd still writes into daemon.log")

    def test_pre_stamp_log_counts_as_due(self):
        # The log this change first meets: HH:MM:SS, no date.
        self._write("[brain-daemon 11:33:19] THREADS: 12 alive (6 non-daemon)\n")
        self.assertTrue(self._rotate()['rotated'])

    def test_rotation_replaces_the_previous_backup(self):
        with open(self.path + ".1", "w") as f:
            f.write("oldest month\n")
        self._write(self._stamped(31, "previous month"))
        self._rotate()
        self.assertNotIn("oldest month", self._read(self.path + ".1"))

    def test_fresh_file_is_not_due_again(self):
        self._write(self._stamped(31, "previous month"))
        self._rotate()
        self.assertFalse(self._rotate(now=datetime.now(timezone.utc))['rotated'])

    def test_fd_on_another_file_leaves_the_log_alone(self):
        # A daemon run from a terminal, or a stale path: never dup2 its fds away.
        other = os.open(os.path.join(self._dir, "other.log"),
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            self._write(self._stamped(31, "previous month"))
            result = dlog.rotate_if_due(self.path, 60, fds=(other,), now=self.NOW)
        finally:
            os.close(other)
        self.assertFalse(result['rotated'])
        self.assertFalse(os.path.exists(self.path + ".1"))

    def test_keep_days_comes_from_config_with_a_floor(self):
        d = ds.BrainDaemon.__new__(ds.BrainDaemon)   # skip __init__
        d.db_path = os.path.join(self._dir, "brain.db")
        d.brain = MagicMock()
        for configured, used in ((30, 30), (0, 2)):
            d.brain.get_config.return_value = configured
            with patch.object(dlog, "rotate_if_due") as rotate, \
                 patch.dict(os.environ, {"BRAIN_DB_DIR": self._dir}):
                d._rotate_daemon_log()
            rotate.assert_called_once_with(self.path, used)
        d.brain.get_config.assert_called_with('daemon_log.keep_days', 60)


if __name__ == "__main__":
    unittest.main()
