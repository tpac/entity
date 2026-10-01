"""
brain — Daemon Server

BrainDaemon class: loads Brain, serves commands over TCP localhost.
Thread pool (THREAD_POOL_SIZE workers) handles concurrent connections.
Reads run without lock; all writers (daemon dispatch, S2 encoder, embed
queue, autosave) serialize via brain.write_lock — the lock lives on the
brain itself, not the daemon, so any caller with a brain reference takes
the same lock.
"""

import sys

if __name__ == '__main__':
    # `python -m servers.daemon_server <brain.db>` — the daemon's single entry
    # point (see main() at the bottom of this file).
    #
    # This guard sits at the TOP, before the module body, on purpose: `-m`
    # executes this file under the name `__main__`, so importing it back under
    # its canonical name would otherwise run the whole body a SECOND time —
    # two module objects, two sets of class identities, every module-level
    # side effect doubled (visibly: the trust-store warning below printed
    # twice). Delegating here means the body runs exactly once, under the name
    # every other importer sees.
    from servers.daemon_server import main as _main
    sys.exit(_main())

# OS trust store injection — corporate networks TLS-intercept HTTPS with a
# proxy CA that lives in the system keychain, which Python's bundled certifi
# ignores; every outbound call (Anthropic API, embedding-model fetch) then
# dies with UnknownIssuer. Patch ssl.SSLContext to defer to the OS store
# before anything can build an SSL context. Missing package (a venv
# bootstrapped before truststore was declared) falls back to certifi — the
# prior behavior — and says so in daemon.log.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception as _e:
    print(f"[daemon] WARN: OS trust store injection unavailable ({_e}) — TLS uses bundled certifi", file=sys.stderr)

import os
import json
import socket
import select
import signal
import time
import threading
import traceback
import atexit
import fcntl
import errno
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from . import daemon_log
from .brain_constants import LOGS_MAINTENANCE_INTERVAL_S
from .daemon_config import (
    AUTOSAVE_INTERVAL_SECONDS, SESSION_FLUSH_INTERVAL_SECONDS,
    SOCKET_BACKLOG, MAX_MESSAGE_SIZE, THREAD_POOL_SIZE,
    POOL_THREAD_PREFIX, THREAD_ALARM_MAX_OTHERS, DAEMON_LOG_KEEP_DAYS,
    REPO_ROOT,
    get_daemon_addr, get_pid_path, get_lock_path, get_status_path,
    get_db_lock_path, get_daemon_log_path,
)
from .daemon_dispatch import COMMAND_TABLE, dispatch_command
from .daemon_launch import runtime_ready, start_runtime_sync
from .dispatch_common import caller_session


def _thread_alarm(threads):
    """The THREADS log line for a thread inventory, or None while it is healthy.

    Pool workers are left out of both checks: the pool is bounded by
    THREAD_POOL_SIZE, so it cannot run away, and concurrent.futures workers are
    non-daemon by design. Among the rest, the alarm is too many threads, or any
    non-daemon one besides MainThread — it keeps the interpreter from exiting
    at shutdown."""
    alive = [t for t in threads if t.is_alive()]
    pool = [t for t in alive if t.name.startswith(POOL_THREAD_PREFIX)]
    others = [t for t in alive if t not in pool and t.name != "MainThread"]
    unexpected = [t for t in others if not t.daemon]
    reasons = []
    if unexpected:
        reasons.append("%d unexpected non-daemon [%s]" % (
            len(unexpected), ", ".join(t.name for t in unexpected)))
    if len(others) > THREAD_ALARM_MAX_OTHERS:
        reasons.append("%d threads outside the pool (limit %d)" % (
            len(others), THREAD_ALARM_MAX_OTHERS))
    if not reasons:
        return None
    return "THREADS: %s — %d alive, %d pool workers; outside the pool: %s" % (
        "; ".join(reasons), len(alive), len(pool),
        ", ".join(t.name for t in others))


class DuplicateDaemonError(Exception):
    """The port is already served by a responsive brain daemon, so this process
    is a duplicate. Signals the supervisor to exit cleanly (defer to the
    incumbent) instead of restarting into the same bind collision — which would
    turn a single duplicate into a perpetual Errno-48 crash storm (2026-06-05).
    """


class BrainDaemon:
    """Persistent Brain daemon that listens on TCP localhost."""

    MAX_SUPERVISOR_RESTARTS = 5      # Max restarts before giving up
    SUPERVISOR_RESTART_COOLDOWN = 2   # Seconds between restart attempts
    HEALTHY_UPTIME_RESET_S = 300      # A crash after this much healthy serving is
                                      # a FRESH incident, not part of a rapid crash-
                                      # loop — reset the streak. Without this the
                                      # count either reset on every bind (an endless
                                      # serve-crash loop never hit MAX) or accrued
                                      # across unrelated crashes hours apart.
    SHUTDOWN_BACKSTOP_S = 15          # Force-exit if teardown wedges. > a typical
                                      # in-flight recall (so it finishes on open
                                      # conns), < launchd's ~20s SIGTERM→SIGKILL
                                      # window (so we exit on our own terms).
    SOCKET_BIND_RETRIES = 10          # Retries for port binding after crash
    SOCKET_BIND_RETRY_DELAY = 1.0     # Seconds between bind retries

    # Sync-before-swap state (see _step_runtime_sync). Class-level defaults;
    # instances assign their own. The `restart` handler only sets the request;
    # the main thread owns the child from spawn to reap.
    _sync_requested = False           # a restart found the runtime behind uv.lock
    _runtime_sync = None              # the running ensure-runtime.sh child

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.daemon_addr = get_daemon_addr()
        self.pid_path = get_pid_path()
        self.brain = None
        self.server_socket = None
        self.running = False
        # The S2/keepalive gating signals (last_user_activity + encode-runs
        # counter) live on brain.activity (ActivityState) — single source of
        # truth, mutated here via record_* and read by run_maintenance_if_due.
        self.dirty = False
        self.graph_changes = []  # In-memory graph mutation log
        # Write serialization lives on the brain (brain.write_lock) — see
        # Brain.__init__. Daemon acquires it via _locked_exec / autosave.
        self._pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE,
                                        thread_name_prefix=POOL_THREAD_PREFIX)
        # S1 Scribe concurrency: a hard cap of MAX_CONCURRENT_ENCODES encodes at
        # once across all sessions (BoundedSemaphore — an over-count release
        # raises rather than silently inflating the cap). The daemon owns encode
        # concurrency now that the Scribe is poll-driven (brain.scribe_due decides
        # who's due; this semaphore caps how many run; the encode thread releases a
        # permit when it completes). Per-session single-flight rides on top via
        # _encode_inflight: a session already encoding is excluded from selection,
        # so it can't double-encode even though an encode outlives the retry
        # cooldown. _encode_inflight: {session_id: monotonic start} (also the
        # hung-alarm baseline); _encode_hung_warned: {session_id: True} once warned.
        from servers.scales.s1.encode_contract import MAX_CONCURRENT_ENCODES
        self._encode_sema = threading.BoundedSemaphore(MAX_CONCURRENT_ENCODES)
        self._encode_inflight = {}
        self._encode_hung_warned = {}
        # Per-session Scribe retry cooldown. A failed/skipped encode never resets
        # the cadence, so the session stays "due" — without this the ~5s poll
        # would re-fire it every tick. {session_id: last-attempt monotonic}; a
        # successful encode pops its entry. _scribe_failures counts re-fires that
        # never advanced (wedged encode) → loud escalation. GIL makes the dict
        # set/pop atomic; the cooldown is advisory, so the poll-thread rebuild vs
        # encode-thread pop race is benign.
        self._scribe_attempts = {}
        self._scribe_failures = {}
        self._restart_count = 0
        self._reload_requested = False  # set by the `restart` command; start()
                                        # execs fresh code in place after teardown
        self._run_started_at = 0  # wall-clock when the current _run() began serving
                                  # (set after bind) — the supervisor's healthy-uptime
                                  # streak reset reads it.

    # Hook dispatch table: hook_name → (is_write, marks_dirty)
    #   is_write     — True takes _write_lock; False runs concurrently
    #   marks_dirty  — True sets self.dirty so autosave persists soon
    # Single source of truth for hook routing — no parallel "read_hooks"
    # list to keep in sync. Function name in daemon_hooks always matches
    # the cmd name; getattr(_hooks, cmd) resolves it.
    HOOK_TABLE = {
        "hook_recall":               (False, True),
        "hook_post_response_track":  (True,  True),
        "hook_pre_edit":             (False, True),
        "hook_pre_bash_safety":      (False, False),
        "hook_session_end":          (True,  True),
        "hook_stop_failure_log":     (True,  True),
        "hook_worktree_context":     (False, True),
        "hook_worktree_cleanup":     (True,  True),
    }

    def start(self):
        """Supervisor loop — start daemon and restart on internal crashes.

        Handles: brain errors, socket errors, thread pool crashes.
        Does NOT handle: SIGKILL, OOM (external watchdog needed — MCP plugin).
        Gives up after MAX_SUPERVISOR_RESTARTS consecutive crashes.
        """
        # Clear pycache so launchd restarts always use latest code
        import shutil
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '__pycache__')
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

        # Acquire exclusive lock (one daemon per user). flock IS the singleton
        # primitive: the kernel auto-releases it when the holder dies, so a
        # LOCK_NB failure ALWAYS means a live holder — there is no "stale flock".
        # Do NOT unlink + recreate the lock on failure: that let the incumbent
        # keep its lock on the old (unlinked) inode while a recreator acquired a
        # SECOND lock on a fresh inode — defeating mutual exclusion and feeding
        # the Errno-48 storm (2026-06-05). A lingering lock file is harmless:
        # flock on the reused inode succeeds once the dead holder's lock is gone.
        lock_path = get_lock_path()
        self._lock_fd = open(lock_path, 'w')
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            self._lock_fd.close()
            pid_hint = ""
            try:
                pid_hint = " (PID %s)" % open(get_pid_path()).read().strip()
            except Exception:
                pass
            self._log("Another daemon holds the singleton lock%s. Exiting duplicate." % pid_hint)
            return

        # Second flock, keyed on the resolved BRAIN rather than the instance
        # label: with BRAIN_INSTANCE keying the singleton lock above, only this
        # lock still guarantees one writer per brain.db machine-wide. A
        # mis-enved entity whose ladder resolved onto another instance's (or
        # production's) DB exits here, loudly, instead of corrupting an index.
        self._db_lock_fd = open(get_db_lock_path(self.db_path), 'w')
        try:
            fcntl.flock(self._db_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            self._db_lock_fd.close()
            self._lock_fd.close()
            self._log("Another daemon already writes %s (db lock held). "
                      "Exiting duplicate — check BRAIN_DB_DIR/BRAIN_INSTANCE." %
                      self.db_path)
            return

        self._claim_pid_file()

        # Register cleanup + signal handlers (once).
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_signal)
        atexit.register(self._cleanup)

        # ── Supervisor loop ──
        while self._restart_count <= self.MAX_SUPERVISOR_RESTARTS:
            try:
                self._run()
                # _run() returns normally on clean shutdown (signal)
                break
            except DuplicateDaemonError as e:
                # Another responsive daemon owns the port. Restarting would just
                # hit the same bind collision — exit cleanly and defer to it.
                self._log("DEFER: %s — exiting cleanly (no restart)." % e)
                break
            except Exception as e:
                # LOUD first: the traceback goes to daemon.log (and the brain error
                # table if the brain is up) before any branch decision.
                self._log_crash(e)

                # Phase-scoped, not exception-type-scoped. self.brain is set ONLY
                # after Brain() constructs (see _load_brain). So brain is None ⟺
                # the crash happened before/during the load — a brain-level fault.
                # Warm-retrying that re-runs the SAME deterministic failure
                # ×MAX, each paying the full load cost while holding the flock and
                # serving nothing. Exit for a fresh process (a clean reload)
                # instead of crash-looping in place: on macOS launchd KeepAlive
                # respawns it (throttled); off-launchd the next client
                # ensure_daemon does. Either way, exit beats in-place retry.
                if self.brain is None:
                    self._log("FATAL: crash before the brain loaded — exiting for a "
                              "fresh reload (KeepAlive / next ensure_daemon respawns; "
                              "no in-place retry).")
                    break

                # Brain is up: a transient serve/bind/socket fault, where the warm-
                # brain retry is the real latency win (no reload). Reset the streak
                # if the daemon had been serving healthily — a crash after sustained
                # uptime is a fresh incident, not a continuation of a rapid loop.
                if self._run_started_at and (
                        time.time() - self._run_started_at) > self.HEALTHY_UPTIME_RESET_S:
                    self._restart_count = 0
                self._restart_count += 1

                if self._restart_count > self.MAX_SUPERVISOR_RESTARTS:
                    self._log("FATAL: %d consecutive crashes. Giving up." % self._restart_count)
                    break

                self._log("SUPERVISOR: Restart %d/%d in %ds..." % (
                    self._restart_count, self.MAX_SUPERVISOR_RESTARTS,
                    self.SUPERVISOR_RESTART_COOLDOWN))

                # Clean up before restart
                self._close_socket()
                time.sleep(self.SUPERVISOR_RESTART_COOLDOWN)

        self._shutdown()
        if self._reload_requested:
            self._exec_reload()

    def _run(self):
        """Single daemon lifecycle — load brain, bind socket, serve until stopped.

        Raises on fatal errors so the supervisor can restart.
        Normal shutdown (signal) returns cleanly.

        Singleton identity is the flock, acquired in start() BEFORE this runs.
        A serving daemon holds its flock for its whole serving life (released in
        _cleanup, after the socket closes), so while WE hold it no other same-uid
        process can be past the flock-acquire — none is binding or serving our
        port. The only residual port owner is a DIFFERENT uid colliding via
        uid%100, and the EADDRINUSE backstop in _bind_socket defers cleanly on
        that (and on the acquire→bind race window). So there is no reachable
        "responsive daemon already owns the port" state here to pre-check for.
        """
        # Load brain if not loaded (first run or after crash that corrupted it)
        if not self.brain:
            self._load_brain()

        # Bind socket with retry (handles TIME_WAIT after crash)
        self._bind_socket()

        # Bound: rewrite the PID file start() claimed. The fresh mtime is the
        # corpse test's "just bound", and the rewrite restores a file someone
        # removed while this process loaded.
        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))
        # Status file immediately — teardown unlinked it, and the next natural
        # writer is a hook command or the 60s autosave tick, so without this
        # the statusline reads "Brain offline" for up to a minute after a
        # 2-4s reload.
        self._write_status()

        self.running = True
        self._run_started_at = time.time()  # healthy-uptime clock for the supervisor's
                                            # streak reset (see the except handler).
        threads = threading.enumerate()
        self._log("Daemon started. PID={}, addr={}:{}, workers={}, restarts={}, threads={}({})".format(
            os.getpid(), self.daemon_addr[0], self.daemon_addr[1],
            THREAD_POOL_SIZE, self._restart_count, len(threads),
            ", ".join(t.name for t in threads if t.name != "MainThread")))

        # Start autosave thread
        autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True, name="autosave")
        autosave_thread.start()
        session_flush_thread = threading.Thread(
            target=self._session_flush_loop, daemon=True, name="session-flush")
        session_flush_thread.start()

        # Memory watchdog — opt-in via config (memory_watchdog.enabled). Off
        # by default. When the daemon leaks (it has — see brain memory
        # `b6e32edd` category), flip the config to start RSS sampling so the
        # next leak shows up in daemon.log without instrumenting allocations.
        # Allocation-level profiling is intentionally NOT in the watchdog —
        # tracemalloc inside the daemon turned recall into a 5-min spin
        # (2026-05-08); use a one-shot diagnostic script or `py-spy` instead.
        # See servers/memory_watchdog.py for config keys.
        try:
            from .memory_watchdog import MemoryWatchdog
            self._memory_watchdog = MemoryWatchdog.maybe_start(
                self.brain, log_fn=lambda m: self._log("[mem] " + m))
        except Exception as _wd_e:
            # Watchdog must never block daemon startup.
            self._log("memory_watchdog start failed: %s" % _wd_e)
            self._memory_watchdog = None

        # DB maintenance — WAL checkpoint every 5 min, PRAGMA optimize
        # every 30 min, a weekly full ANALYZE, plus a daily compressed
        # rolling backup, on brain.db and brain_logs.db. Lives in its own
        # backend-agnostic scheduler (servers/db_maintenance.py) that calls into
        # db_backends.current for SQLite-specific work; if we ever migrate
        # off SQLite, only the backend module changes. Always-on (no
        # config gate) — periodic checkpointing caps WAL growth, which
        # directly reduces writer-slot wait time. Failures caught + logged
        # via brain._log_error; loop never dies.
        #
        # Auto-backups land in a `backups/` subdir beside the DBs (keeps
        # the rolling snapshots out of the cluttered top-level dir) under
        # GFS retention (7 daily + 4 weekly + 3 monthly per DB). These are
        # distinct from the named pre-destructive-op `.bak` files a human
        # makes by hand — this scheduler only manages its own snapshots.
        # A warm retry re-enters _run with the PREVIOUS scheduler still
        # ticking — its thread outlives the fault, and overwriting the reference
        # below makes it unstoppable. It would keep running WAL checkpoint,
        # optimize, quick_check and BACKUPS on its own 30s tick, concurrently
        # with the new one (node:c7f8ce33 is what concurrent maintenance ops
        # cost). stop() only clears a flag; the thread exits at its next check.
        # OUTSIDE the try below, and before it: everything in there can raise,
        # and that except sets _db_maintenance = None — which would orphan this
        # scheduler beyond reach, the exact bug being fixed.
        _prev_dbm = getattr(self, '_db_maintenance', None)
        if _prev_dbm is not None:
            _prev_dbm.stop()
        try:
            from .db_backup import default_backup_dir
            from .db_maintenance import DBMaintenance
            backup_dir = default_backup_dir(self.brain.db_path)
            self._db_maintenance = DBMaintenance(
                log_fn=self._log,
                log_error_fn=getattr(self.brain, '_log_error', None))
            self._db_maintenance.register(
                'brain', self.brain.db_path, backup_dir=backup_dir)
            self._db_maintenance.register(
                'brain_logs', self.brain.logs_db_path, backup_dir=backup_dir)
            # Own try: a task registration failure must not take
            # checkpoint/optimize/backup down with it. Backups are the one
            # mechanism that survived the six-week idle-hook outage precisely
            # because nothing else gated them. Registered BEFORE start() —
            # _tasks is only safe to mutate while the worker isn't iterating it.
            try:
                self._db_maintenance.register_task(
                    'logs_retention', self.brain.run_logs_maintenance,
                    LOGS_MAINTENANCE_INTERVAL_S)
            except Exception as _t_e:
                self._log("logs_retention task registration failed: %s" % _t_e)
            try:
                self._db_maintenance.register_task(
                    'daemon_log_rotation', self._rotate_daemon_log,
                    LOGS_MAINTENANCE_INTERVAL_S)
            except Exception as _t_e:
                self._log("daemon_log_rotation task registration failed: %s" % _t_e)
            try:
                from .brain_constants import DB_SIZE_TELEMETRY_INTERVAL_S
                self._db_maintenance.register_task(
                    'db_size_telemetry', self._collect_db_size_telemetry,
                    DB_SIZE_TELEMETRY_INTERVAL_S)
            except Exception as _t_e:
                self._log("db_size_telemetry task registration failed: %s" % _t_e)
            self._db_maintenance.start()
        except Exception as _dm_e:
            # Scheduler must never block daemon startup.
            self._log("db_maintenance start failed: %s" % _dm_e)
            self._db_maintenance = None

        # Brain warmup — build the LAF engine caches and the structural degree
        # cache off the user's critical path (see Brain.warm_up). Without it
        # the first recall pays ~1 s of cache builds; that bill belongs to
        # boot, not the user's first prompt.
        #
        # Background thread: boot-blocking would re-introduce latency at a
        # different lifecycle point. The recall hot path's lazy paths are
        # idempotence guards — if the user prompts before warmup finishes,
        # recall finishes the work itself.
        warmup_thread = threading.Thread(
            target=self._run_warmup, daemon=True, name="warmup")
        warmup_thread.start()

        # Connection keepalive — re-warm the Anthropic httpx pool during idle
        # so the first recall after a quiet period doesn't pay a cold-TLS tax
        # (idle inflates surface_haiku ~6s->~10s; see
        # Brain.warm_anthropic_connection). Always-on thread; the work is both
        # config-gated and idle-gated inside the loop, so flipping
        # surface_keepalive.enabled off makes every tick a no-op — no restart.
        keepalive_thread = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="keepalive")
        keepalive_thread.start()

        self._serve()

        # THE one place that decides what the serve loop's exit meant. Every
        # exit path inside _serve is a bare break; asking here — once — is what
        # stops the question being re-answered per arm (it was asked in three
        # arms, and one of them forgot, which is how a real socket fault could
        # read as a clean shutdown).
        #
        # running still True => nobody asked us to stop, so the loop ended
        # because the socket died. Raise: start()'s supervisor logs the crash,
        # counts it against MAX_SUPERVISOR_RESTARTS and warm-retries with the
        # brain still loaded. Returning normally here would make a real fault
        # indistinguishable from a clean shutdown — no _log_crash, no count,
        # give-up threshold unreachable. Teardown is NOT done here: _shutdown()
        # runs once at the supervisor's tail, on every path.
        if self.running:
            raise RuntimeError(
                'serve loop exited while still running — the listening socket '
                'failed (see the "select failed" line above for the errno)')

    def _collect_db_size_telemetry(self):
        """Weekly durable size snapshot for both DBs — file-level stats plus
        per-table breakdown (dbstat, best-effort) — written as one debug_log
        row per DB via the DAL write boundary. Turns "why did the DB grow?"
        into a query over event_type='telemetry' instead of archaeology.
        Runs on the db-maintenance thread (register_task)."""
        from . import db_backends
        backend = db_backends.current
        out = {}
        for name, db_path in (('brain', self.brain.db_path),
                              ('brain_logs', self.brain.logs_db_path)):
            payload = backend.stats(db_path)
            payload['tables_mb'] = backend.table_sizes(db_path)
            self.brain._logs_dal.write_event(
                'telemetry', 'db_size_%s' % name, payload)
            out[name] = payload['db_size_bytes']
        return out

    def _rotate_daemon_log(self):
        """daemon.log retention, run on the db-maintenance thread
        (register_task). keep_days comes from brain_meta
        `daemon_log.keep_days`, floored at 2 so a bad value rotates at most
        daily instead of emptying the history every run."""
        keep_days = max(2, int(self.brain.get_config(
            'daemon_log.keep_days', DAEMON_LOG_KEEP_DAYS)))
        return daemon_log.rotate_if_due(get_daemon_log_path(self.db_path), keep_days)

    def _run_warmup(self):
        """Background warmup. See Brain.warm_up() for what's covered."""
        try:
            timings = self.brain.warm_up()
            self._log("Warmup done: %s" % timings)
        except Exception as e:
            # Warmup failure must never affect the daemon. Falling through
            # to the recall hot path's lazy guards is acceptable degradation.
            self._log("Warmup failed: %s" % e)

    # Connection keepalive ---------------------------------------------------
    # Config (brain config table, hot-read each tick so edits take effect live
    # without a daemon restart):
    #   surface_keepalive.enabled           bool  default True
    #   surface_keepalive.interval_seconds  int   default 300 (5 min)

    @staticmethod
    def _keepalive_due(idle_s: float, since_last_ping_s: float,
                       interval_s: float) -> bool:
        """Whether to re-warm the Anthropic connection this tick.

        Fire only when the daemon has been idle a full interval AND we haven't
        already warmed within the last interval. Active sessions keep the pool
        hot via real recalls, so nothing fires while prompts are flowing; once
        idle, it re-warms once per interval to hold the connection open. Pure
        function — no clock, no I/O — so the gating stays unit-testable.
        """
        return idle_s >= interval_s and since_last_ping_s >= interval_s

    def _keepalive_loop(self):
        """Keep the Anthropic connection warm during idle (see _keepalive_due
        and Brain.warm_anthropic_connection).

        Gated on last_user_activity (the UserPromptSubmit clock), so a
        keepalive tick never resets the clock S2 idle maintenance gates on,
        and so quiet stretches between prompts (Anchor working, or the
        operator away) correctly count as idle and re-warm.

        Scope: this warms `brain.anthropic_client`, the client the S1 surface
        step reuses. The encoders / S2 units (via runner.make_client) build
        their own client per use and are NOT covered here — by design, since
        the recall timeout this addresses is a surface-path problem.
        """
        CHECK_CADENCE_S = 30.0    # how often to re-check idle; cheap, bounded
        last_ping = time.time()   # avoid double-warming right after boot warmup
        while self.running:
            time.sleep(CHECK_CADENCE_S)
            if not self.running or not self.brain:
                continue
            last_ping = self._keepalive_tick(time.time(), last_ping)

    def _keepalive_tick(self, now: float, last_ping: float) -> float:
        """One keepalive decision, factored out of _keepalive_loop so the
        gating and backoff are unit-testable without driving the thread.
        Returns the (possibly advanced) last_ping.

        Config (hot-read each tick): surface_keepalive.enabled (bool, default
        True), surface_keepalive.interval_seconds (int, default 300). A
        non-numeric interval falls back to the default rather than disabling
        the loop.
        """
        try:
            if not self.brain.get_config('surface_keepalive.enabled', True):
                return last_ping
            interval = self.brain.get_config(
                'surface_keepalive.interval_seconds', 300)
            try:
                interval = float(interval)
            except (TypeError, ValueError):
                # Bad config value -> fall back to the default; never let a
                # typo silently disable the keepalive.
                interval = 300.0
            if interval <= 0:
                return last_ping
            if self._keepalive_due(now - self.brain.activity.last_user_activity,
                                   now - last_ping, interval):
                # Advance last_ping BEFORE warming so a raised API error still
                # backs off to one attempt per interval (not one per tick).
                # warm_anthropic_connection raises on API error and returns
                # False if there's no client yet or a warm is already in
                # flight — all three outcomes back off the same way.
                # Keyless: the warm can only 401 — skip (noted once), keep the
                # interval cadence so the check stays cheap.
                last_ping = now
                if self.brain.llm_available:
                    self.brain.warm_anthropic_connection()
                else:
                    self.brain.note_llm_unavailable('keepalive warm')
            return last_ping
        except Exception as e:
            # A keepalive tick must never crash the daemon.
            try:
                self.brain._log_error(
                    'keepalive_tick', e, 'keepalive tick failed')
            except Exception:
                pass
            return last_ping

    def _bind_socket(self):
        """Bind TCP socket with retry for TIME_WAIT recovery."""
        for attempt in range(self.SOCKET_BIND_RETRIES):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # DO NOT set SO_REUSEPORT — it allows duplicate daemons to bind the same port
                self.server_socket.bind(self.daemon_addr)
                self.server_socket.listen(SOCKET_BACKLOG)
                self.server_socket.setblocking(False)
                return  # Success
            except OSError as e:
                self._close_socket()
                # If a healthy daemon already owns the port, we're a duplicate —
                # defer (clean exit) rather than retry into a crash storm. Backstop
                # for the race where an incumbent binds between the _run pre-check
                # and here. (A non-responsive holder — TIME_WAIT or a hung corpse —
                # falls through to the normal retry; recovery handles corpses.)
                if e.errno == errno.EADDRINUSE:
                    from .daemon_client import is_daemon_responsive
                    if is_daemon_responsive(timeout=1.0):
                        raise DuplicateDaemonError(
                            "Port %d already served by a responsive daemon." %
                            self.daemon_addr[1])
                if attempt < self.SOCKET_BIND_RETRIES - 1:
                    self._log("BIND: Port %d busy (attempt %d/%d): %s" % (
                        self.daemon_addr[1], attempt + 1, self.SOCKET_BIND_RETRIES, e))
                    time.sleep(self.SOCKET_BIND_RETRY_DELAY)
                else:
                    raise  # Give up — supervisor will handle

    def _close_socket(self):
        """Close the server socket safely."""
        try:
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception:
            pass

    def _log_crash(self, error: Exception):
        """Log crash details to daemon.log and brain error log."""
        tb = traceback.format_exc()
        self._log("CRASH: %s\n%s" % (error, tb))
        # Also log to brain's error table if brain is alive
        try:
            if self.brain:
                self.brain._log_error('daemon_crash', error,
                                       'restart_count=%d' % self._restart_count)
        except Exception:
            pass

    def _load_brain(self):
        """Load the Brain instance + embedder."""
        try:
            import torch
            torch.backends.mps.is_available = lambda: False
            torch.backends.mps.is_built = lambda: False
        except ImportError:
            pass

        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)

        from servers.brain import Brain
        self.brain = Brain(self.db_path)
        self._log("Brain loaded from {}".format(self.db_path))
        if self.brain.eval_brain_since() is not None:
            self._log("Eval brain: S2 and idle-tail encodes start only on "
                      "s2_force / scribe_flush")

        # Reclassify pre-override-model seeded rows as "no override deployed",
        # once per install (its own version stamp; a stamped install no-ops).
        # Deliberately here and NOT in Brain.__init__: this is the one Brain
        # that owns its DB exclusively (launchd singleton, before the port
        # opens), while eval corpora, IsolatedBrain copies and the boot_brain
        # fallback all construct a Brain that must never be mutated.
        try:
            from servers.interaction_collapse import collapse_seeded_overrides
            collapse_seeded_overrides(self.brain)
        except Exception as e:
            self._log("override collapse failed: {}".format(e))

        # Feedback at the moment a default edit is expected to land: a live
        # override pointer SHADOWS its code default (the whole template when
        # the row carries one; config key-by-key), so an edit to a shadowed
        # value doesn't take. Named per boot — the restart IS the deployment
        # step, so this is where an operator looks when an edit doesn't land.
        # The two permanent COLLAPSE_POLICY pointers (PIN/SKIP) are excluded:
        # printing them every boot would habituate the line into noise, and
        # they are kept by design — a real anomaly must arrive on a line that
        # is otherwise absent.
        try:
            from servers.interaction_collapse import COLLAPSE_POLICY, PIN, SKIP
            permanent = {n for n, v in COLLAPSE_POLICY.items()
                         if v in (PIN, SKIP)}
            shadowing = [e for e in self.brain.list_interactions()
                         if e.get('active_version') is not None
                         and e['name'] not in permanent]
            if shadowing:
                self._log(
                    "override pointers shadowing code defaults (template "
                    "whole, config keys they set — an edit to a shadowed "
                    "value won't land until the override is cleared): %s"
                    % ", ".join(
                        "%s v%s (set_by=%s)" % (e['name'], e['active_version'],
                                                e.get('active_set_by'))
                        for e in shadowing))
        except Exception as e:
            self._log("override shadow report failed: {}".format(e))

        # Start the embed queue drain worker. remember/revise/remember_batch
        # enqueue dirty node_ids; this worker embeds them in batches every
        # EMBED_DRAIN_INTERVAL seconds, and sweeps for unqueued gaps on a
        # much slower cadence (COVERAGE_SWEEP_INTERVAL — hourly; the sweep is
        # a net for writers that bypassed the enqueue paths, not a routine
        # pass). It owns vector coverage end to end — nothing else repairs a
        # missing embedding.
        try:
            from servers import embed_queue
            embed_queue.start(self.brain)
            self._log("Embed queue worker started (drain every {}s)".format(
                embed_queue.EMBED_DRAIN_INTERVAL))
        except Exception as e:
            self._log("embed_queue start failed: {}".format(e))

        # Cold-start vector re-queue: a versioned migration can delete
        # vectors (the schema layer has no queue), and the in-memory queue
        # does not survive a restart, so a node written just before one lost
        # its enqueue. A gap that exists at boot is known, not leaked.
        # Routing it through the enqueue path keeps the coverage sweep's
        # `embed_coverage_gap` alarm meaning what it says — a writer bypassed
        # the hooks — and drains at the worker's batch size instead of the
        # sweep's.
        try:
            self._enqueue_vector_backfill_gaps()
        except Exception as e:
            self._log("vector backfill enqueue failed: {}".format(e))

    def _enqueue_vector_backfill_gaps(self):
        """Enqueue nodes missing their _primary vector (a write whose enqueue
        died with the previous process) and nodes whose edge_context vector is
        missing though a described edge exists (the group a migration deletes,
        v33). No `model=`: a stale-model row counts as present, so a model swap
        stays the sweep's bulk job instead of queueing the whole corpus here."""
        from servers import embed_queue
        primary = self.brain._vec_dal.find_missing('_primary', limit=100000)
        edge_context = self.brain._vec_dal.find_missing(
            'edge_context', limit=100000, require_described_edge=True,
            exclude_relations=self.brain._edge_context_excluded())
        for row in primary + edge_context:
            embed_queue.enqueue(row['id'])
        if primary or edge_context:
            self._log("Vector backfill enqueued {} nodes missing _primary + {} "
                      "missing edge_context".format(len(primary), len(edge_context)))

    def _serve(self):
        """Main event loop — accept connections, dispatch to thread pool."""
        # Monotonic: the tick is a pure elapsed-time question. On wall clock a
        # backward step (NTP correction, manual set) makes the delta negative
        # and freezes BOTH bodies of the gate — no S2 maintenance submit, no
        # Scribe poll — until wall clock climbs back, while the loop keeps
        # accepting connections so nothing looks wrong in daemon.log.
        last_poll_tick = time.monotonic()
        self._maintenance_poll_running = False
        self._scribe_poll_running = False
        while self.running:
            # Periodic tick — maintenance + Scribe poll, not every loop
            mono = time.monotonic()
            if mono - last_poll_tick > 5.0:
                last_poll_tick = mono

                # Maintenance decision lives in brain.run_maintenance_if_due().
                # Daemon owns the polling cadence; brain owns BOTH "is it time?"
                # (idle threshold + min interval + persisted last-run timestamp)
                # AND the single-flight lock (brain._s2_lock, taken by run_s2).
                # The flag below is only a submission throttle — it stops a
                # second pool task being queued on this loop; it does NOT make
                # the cycle single-flight and must not be mistaken for the
                # guard (node:24a48377 — a guard held by one caller cannot
                # protect the others). Keeping "is it time?" in the brain means
                # restarting the daemon doesn't forget when maintenance last
                # ran — the brain_meta record survives.
                if not self._maintenance_poll_running:
                    self._maintenance_poll_running = True
                    try:
                        self._pool.submit(self._run_idle_maintenance)
                    except Exception:
                        # submit can raise on pool shutdown/saturation; clear the
                        # flag so the next tick retries instead of wedging S2.
                        self._maintenance_poll_running = False

                # S1 Scribe reactor — poll for a session whose encode is due
                # (mid-session ENCODE_EVERY turns, or the idle tail). Distinct
                # from idle maintenance: the Scribe fires DURING active sessions,
                # gated per-session, not on the global idle clock. The poll task
                # returns fast (the encode runs on its own thread); the flag just
                # prevents poll-task pile-up if the pool is momentarily saturated.
                if not self._scribe_poll_running:
                    self._scribe_poll_running = True
                    try:
                        self._pool.submit(self._run_scribe_poll)
                    except Exception:
                        # submit can raise if the pool is shutting down/saturated;
                        # clear the flag so the next tick retries instead of
                        # wedging the reactor permanently.
                        self._scribe_poll_running = False

            if self._sync_requested or self._runtime_sync is not None:
                self._step_runtime_sync()

            try:
                # Snapshot: _handle_signal sets running=False and THEN clears
                # server_socket, so a signal landing between the while test and
                # here would otherwise hand select [None] → TypeError, which is
                # not in the caught tuple below.
                server_sock = self.server_socket
                if server_sock is None:
                    break
                # 0.5s select — balances responsiveness to shutdown vs CPU usage
                readable, _, _ = select.select([server_sock], [], [], 0.5)
            except (select.error, OSError) as e:
                # LOG here (this is where the errno is), DECIDE at the boundary
                # (_run, after _serve returns). Every exit from this loop is a
                # bare break; whether it meant "told to die" or "something
                # broke" is asked ONCE, where the answer can't be forgotten.
                # Worded as an observation, not a verdict: a SIGTERM landing
                # mid-select closes the fd under us and lands here on a
                # perfectly clean shutdown, and "failed" in daemon.log would
                # make that normal path read as a fault — the mirror of the
                # bug this shape exists to fix.
                self._log("select ended: {}".format(e))
                break

            for sock in readable:
                try:
                    client, _ = sock.accept()
                    client.settimeout(30.0)
                    # Submit to thread pool — non-blocking
                    self._pool.submit(self._handle_client, client)
                except Exception as e:
                    self._log("Accept error: {}".format(e))

    def _handle_client(self, client: socket.socket):
        """Handle a single client connection (runs in thread pool)."""
        t = threading.current_thread()
        t_name = t.name
        try:
            data = b""
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data or len(data) > MAX_MESSAGE_SIZE:
                    break

            if not data:
                return

            try:
                msg = json.loads(data.decode('utf-8').strip())
            except json.JSONDecodeError as e:
                self._send_error(client, "Invalid JSON: {}".format(e))
                return

            if not isinstance(msg, dict):
                self._send_error(client, "Message must be a JSON object, got: {}".format(type(msg).__name__))
                return

            cmd = msg.get("cmd")
            if cmd is None:
                # Common mistake: using "command" instead of "cmd"
                alt_cmd = msg.get("command")
                if alt_cmd:
                    self._send_error(client, "Wrong key: use 'cmd' not 'command'. Got: {}".format(alt_cmd))
                else:
                    self._send_error(client, "Missing 'cmd' field. Message keys: {}".format(list(msg.keys())))
                return

            if not isinstance(cmd, str):
                self._send_error(client, "Field 'cmd' must be a string, got: {} ({})".format(type(cmd).__name__, str(cmd)[:100]))
                return

            args = msg.get("args", {})

            # The user-prompt clock gates S2 maintenance. Tool-use hooks,
            # internal IPC, pings are noise from S2's perspective.
            # See run_maintenance_if_due in brain.py.
            if cmd == "hook_recall":
                self.brain.activity.record_user_activity()

            # Direct dispatch — no watchdog thread.
            # The old pattern spawned a thread per request and joined with 20s timeout.
            # If it timed out, the thread kept running forever → thread leak → CPU spiral.
            # Now: dispatch runs inline in the pool worker. Client has its own timeout (30s).
            # Pool worker finishes and returns to pool. No orphans.
            t.name = "%s:%s" % (POOL_THREAD_PREFIX, cmd)
            t0 = time.time()
            result = self._dispatch(cmd, args)
            elapsed_ms = int((time.time() - t0) * 1000)

            if elapsed_ms > 2000:
                self._log("[thread:%s] %s took %dms" % (t_name, cmd, elapsed_ms))

            self._send_response(client, result)

        except Exception as e:
            self._log("[thread:%s] EXCEPTION in %s: %s" % (
                t_name, msg.get("cmd", "?") if 'msg' in dir() else "?", e))
            try:
                self._send_error(client, "Internal error: {}".format(e))
            except Exception:
                pass
        finally:
            t.name = t_name
            try:
                client.close()
            except Exception:
                pass

    def _locked_exec(self, fn, cmd, args):
        """Acquire brain's write lock with 10s timeout, execute fn.

        The 10s timeout is for client-facing dispatch — a hung write
        returns an error rather than blocking the client. Background
        writers (S2 encoder dispatch, embed_queue) acquire the same
        brain.write_lock without timeout via `with brain.write_lock:`.
        """
        if not self.brain.write_lock.acquire(timeout=10.0):
            self._log("Write lock timeout (10s) for: {}".format(cmd))
            return {"ok": False, "error": "Write lock timeout — another operation is holding the lock"}
        try:
            return fn()
        finally:
            self.brain.write_lock.release()

    def _dispatch(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route command to handler with appropriate locking."""
        try:
            if cmd == "shutdown":
                # A shutdown overrides any in-flight reload — otherwise the
                # daemon would resurrect itself after stop_daemon() reported
                # success (the unmanaged-orphan shape).
                self._reload_requested = False
                self.running = False
                return {"ok": True, "result": {"status": "shutting_down"}}

            if cmd == "restart":
                # Reload in place: flag it and stop the serve loop. The MAIN
                # thread exits _serve within one select tick, runs the ordered
                # _shutdown() (this response finishes flushing during the pool
                # drain), and start() execs fresh code into this same process
                # (_exec_reload — same PID, no launchd round-trip, no down
                # window beyond teardown + boot). Main thread deliberately:
                # the 2026-04 execv ran on a handler thread and died silently
                # (7b4a01a). Exec failure falls back to exit + KeepAlive —
                # the pre-reload restart behavior.
                #
                # Sync before swap: a tree whose runtime is behind its uv.lock
                # syncs FIRST, while this daemon keeps serving, and reloads
                # when the sync ends. Synced inside the exec'd launcher it ran
                # with nothing serving, long enough for other sessions' hook
                # recovery to kickstart the daemon and kill uv mid-sync
                # (node c34a86e0). `syncing` tells the caller the daemon is
                # healthy and will swap itself.
                if self._sync_requested or self._runtime_sync is not None:
                    return {"ok": True, "result": {"status": "syncing"}}
                if not runtime_ready(REPO_ROOT):
                    self._log("Restart requested — runtime is behind uv.lock; "
                              "syncing while serving, then reloading...")
                    self._sync_requested = True
                    return {"ok": True, "result": {"status": "syncing"}}
                self._log("Restart requested — reloading in place after teardown...")
                self._begin_reload()
                return {"ok": True, "result": {"status": "restarting"}}

            if cmd == "scribe_flush":
                return self._scribe_flush(float(args.get("timeout_s", 600)))

            # Hook commands — HOOK_TABLE is the single source of truth
            # for is_write (lock or concurrent) and marks_dirty.
            if cmd.startswith("hook_"):
                entry = self.HOOK_TABLE.get(cmd)
                if entry is None:
                    return {"ok": False, "error": "Unknown hook: %s" % cmd}
                is_write, _marks_dirty = entry
                if is_write:
                    return self._locked_exec(lambda: self._dispatch_hook(cmd, args), cmd, args)
                return self._dispatch_hook(cmd, args)

            # Table-driven dispatch
            entry = COMMAND_TABLE.get(cmd)
            if entry is None:
                return {"ok": False, "error": "Unknown command: {}".format(cmd)}

            # Caller identity for the touched accumulator — resolved BEFORE the
            # handler runs, because handlers that pass **args to a brain method
            # pop `_caller_session` (see dispatch_common._pop_session_ctx), so it
            # would be gone by the time _accumulate_touched runs. caller_session
            # prefers an explicit session_id filter, else the proxy-stamped
            # `_caller_session` — the write tools send only the latter.
            caller_sess = caller_session(args)
            # Execution goes through dispatch_command (the single chokepoint —
            # arg-contract validation + per-op loudness live there). The registry
            # lookup above stays here because lock/dirty POLICY is the daemon's.
            if entry.is_write:
                def _write():
                    result = dispatch_command(self.brain, cmd, args, self.graph_changes)
                    if entry.marks_dirty:
                        self.dirty = True
                    return result
                result = self._locked_exec(_write, cmd, args)
            else:
                result = dispatch_command(self.brain, cmd, args, self.graph_changes)
            # Record what Anchor's own tools touched this turn (Piece 3a). Only
            # Anchor's TCP calls reach _dispatch — the in-process encoder bypasses
            # it (calls COMMAND_TABLE handlers directly via _make_encoder_dispatch)
            # — so this is structurally Anchor-only, no encoding_source check.
            self._accumulate_touched(caller_sess, cmd, result)
            return result

        except Exception as e:
            tb = traceback.format_exc()
            self._log("Command '{}' failed: {}".format(cmd, tb))
            self.brain._log_error("daemon_dispatch", e,
                                   "cmd={}, args={}".format(cmd, str(args)[:200]))
            return {"ok": False, "error": str(e)}

    # Lookup tools → the touched key their result ids land in. get_node[s] →
    # `recalled` (deliberate by-id reads; the encoder catalog folds them as
    # full bodies). The search tools → `looked_up` (the encoder's <provenance>
    # shows them; NEVER catalog-folded — recall returns whole result pages).
    # Keys must exist in ANCHOR_TOUCHED_KEYS (trace_contract).
    _LOOKUP_KEY = {'get_node': 'recalled', 'get_nodes': 'recalled',
                   'recall': 'looked_up', 'recall_batch': 'looked_up',
                   'find_node_by_title': 'looked_up',
                   'filter_nodes': 'looked_up'}

    @staticmethod
    def _lookup_result_ids(cmd, payload):
        """Node ids a lookup command's result payload names — one extractor per
        result shape, defensive on every branch (a malformed payload yields
        nothing, never raises past the caller's guard)."""
        if cmd == 'recall':                    # by-query AND by-id: {'results': […]}
            rows = payload.get('results') if isinstance(payload, dict) else []
        elif cmd == 'recall_batch':            # [{'query', 'results': […]}, …]
            rows = [r for g in (payload if isinstance(payload, list) else [])
                    if isinstance(g, dict) for r in (g.get('results') or [])]
        elif cmd == 'filter_nodes':            # {'nodes': […]}
            rows = payload.get('nodes') if isinstance(payload, dict) else []
        else:                                  # get_node[s], find_node_by_title:
            rows = payload if isinstance(payload, list) else [payload]
        # get_nodes mixes resolved node dicts with not-found entries
        # {'id': <raw>, 'error': …} — skip those so a bad ref doesn't land
        # in the accumulator as a phantom node id.
        return [n['id'] for n in (rows or [])
                if isinstance(n, dict) and n.get('id') and 'error' not in n]

    def _accumulate_touched(self, sess, cmd, result):
        """Append the node ids Anchor touched this turn to the session's per-turn
        accumulator (flushed as one `anchor_touched` S0 delta in
        post_response_common). Writes contribute their dispatch-authoritative
        `affected` (created/revised/archived); lookups contribute the resolved
        ids they returned, keyed per _LOOKUP_KEY (recalled vs looked_up).
        `sess` is the caller identity resolved before the handler ran.
        Failure-isolated, but LOUD: a real error here is logged (the silent
        version once hid the session-keying bug)."""
        if not isinstance(result, dict):
            return
        aff = result.get('affected')
        lookup_key = self._LOOKUP_KEY.get(cmd)
        # Early-out BEFORE the session lookup, so the recall HOOK path and any
        # other non-contributing command pay nothing.
        if not isinstance(aff, dict) and not lookup_key:
            return
        if not sess:
            return
        try:
            touched = self.brain.get_or_create_session(sess).touched
            if isinstance(aff, dict):
                for k in ('created', 'revised', 'archived'):
                    ids = aff.get(k)
                    if ids:
                        touched[k].extend(ids)
            elif lookup_key:
                touched[lookup_key].extend(
                    self._lookup_result_ids(cmd, result.get('result')))
        except Exception as e:
            try:
                self.brain._log_error('accumulate_touched', e,
                                      'cmd=%s sess=%s' % (cmd, str(sess)[:16]))
            except Exception:
                pass

    def _dispatch_hook(self, cmd: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch hook with telemetry. Caller (_dispatch) handles locking
        based on HOOK_TABLE.is_write — this function trusts that decision."""
        import servers.daemon_hooks as _hooks

        entry = self.HOOK_TABLE.get(cmd)
        if not entry:
            return {"error": "Unknown hook: %s" % cmd}

        _is_write, marks_dirty = entry
        hook_func = getattr(_hooks, cmd)

        start_t = time.time()
        result = hook_func(self.brain, args, self.graph_changes)
        latency_ms = (time.time() - start_t) * 1000

        if marks_dirty:
            self.dirty = True

        # Measure injection volume
        injection_chars = 0
        if isinstance(result, dict):
            reason = result.get("json", {}).get("reason", "") if "json" in result else ""
            output = result.get("output", "")
            injection_chars = len(reason) + len(output)

        # Log telemetry (best-effort)
        try:
            self.brain.log_debug(
                event_type=cmd, source="hook_telemetry",
                latency_ms=latency_ms,
                metadata=json.dumps({
                    "injection_chars": injection_chars,
                    "decision": result.get("json", {}).get("decision", "")
                    if isinstance(result, dict) and "json" in result else "",
                }))
        except Exception as e:
            self._log("Telemetry write failed for %s: %s" % (cmd, e))

        self._write_status()
        return {"ok": True, "result": result}

    def _send_response(self, client: socket.socket, data: Dict[str, Any]):
        """Send JSON response to client."""
        try:
            response = json.dumps(data, default=str) + "\n"
            client.sendall(response.encode('utf-8'))
        except Exception as e:
            self._log("Send error: {}".format(e))

    def _send_error(self, client: socket.socket, message: str):
        """Send error response."""
        self._send_response(client, {"ok": False, "error": message})

    def _write_status(self):
        """Write brain status JSON for statusline script."""
        try:
            brain = self.brain
            if not brain:
                return

            # Semantics: nodes/edges report total store size (incl. archived);
            # locked/tensions report the active (non-archived) subset via the DAL
            # defaults — the identity-meaningful live counts.
            node_count = brain._nodes.count(archived=True)
            locked_count = brain._nodes.count_locked()
            edge_count = brain._graph.count_total()
            tension_count = brain._nodes.count_by_type('tension')

            from servers import embedder
            emb_ready = embedder.is_ready()
            emb_stats = embedder.get_stats() if emb_ready else {}

            last_encode = brain.conn.execute(
                "SELECT created_at FROM nodes ORDER BY created_at DESC LIMIT 1").fetchone()

            status = {
                "nodes": node_count, "edges": edge_count,
                "locked": locked_count, "tensions": tension_count,
                "embedder_ready": emb_ready,
                "model_name": emb_stats.get("model_name", ""),
                "last_encode_at": last_encode[0] if last_encode else None,
                "pid": os.getpid(),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            status_path = get_status_path()
            tmp_path = status_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(status, f)
            os.replace(tmp_path, status_path)
        except Exception:
            pass  # Status file is best-effort

    def _autosave_loop(self):
        """Periodically save brain if dirty + health check + thread monitor.
        SessionContexts flush on their own cadence (_session_flush_loop).

        Recovering a hung daemon is not this loop's job — that's reactive,
        via the ping-based nets (`ensure_daemon` at session start, the MCP
        health monitor during a session), both force-restarting through
        `launchctl kickstart -k`.
        """
        while self.running:
            time.sleep(AUTOSAVE_INTERVAL_SECONDS)
            if self.dirty:
                if self.brain.write_lock.acquire(timeout=5.0):
                    try:
                        self.brain.save()
                        self.dirty = False
                        self._log("Autosaved")
                    except Exception as e:
                        self._log("Autosave error: {}".format(e))
                    finally:
                        self.brain.write_lock.release()
            # Internal health check — verify SQLite alive (skip during shutdown)
            if self.running and self.brain:
                try:
                    self.brain.conn.execute("SELECT 1").fetchone()
                except Exception as e:
                    self._log("HEALTH: SQLite check failed: {}".format(e))
            alarm = _thread_alarm(threading.enumerate())
            if alarm:
                self._log(alarm)
            # Native thread CPU monitor — catches onnxruntime/tokenizers spin
            try:
                import subprocess
                r = subprocess.run(
                    ['ps', '-M', '-p', str(os.getpid()), '-o', 'pcpu='],
                    capture_output=True, text=True, timeout=3)
                hot = [float(x) for x in r.stdout.strip().split('\n') if x.strip() and float(x.strip()) > 50]
                if hot:
                    self._log("CPU SPIRAL DETECTED: %d native threads at %s%%" % (
                        len(hot), "+".join("%.0f" % h for h in hot)))
            except Exception:
                pass
            self._write_status()

    def _session_flush_loop(self):
        """Flush changed SessionContexts every SESSION_FLUSH_INTERVAL_SECONDS.

        Its own loop, not a faster autosave: the autosave pass also forks
        `ps -M`, walks threads and writes the status file, which stay at
        AUTOSAVE_INTERVAL_SECONDS. This bounds what a crash loses; a graceful
        shutdown flushes in Brain.close(). Ctx mutations don't toggle
        self.dirty, so the flush runs regardless of it.
        """
        while self.running:
            time.sleep(SESSION_FLUSH_INTERVAL_SECONDS)
            if not self.running:
                break  # shutting down — Brain.close() does the final flush
            try:
                self.brain.save_session_contexts()
            except Exception as e:
                self._log("SessionContext flush error: {}".format(e))

    def _begin_reload(self):
        """Flag the in-place reload and stop the serve loop; start() runs the
        teardown and _exec_reload on the main thread."""
        # Freshen the PID file NOW, not only at the pre-exec rewrite:
        # teardown (pool drain + save/close, seconds) runs with the
        # socket open-but-unanswering and the PID file otherwise
        # carrying its hours-old bind mtime — exactly the signature
        # recover_daemon's corpse test kills. The utime makes the
        # WHOLE teardown→exec→boot span read "just bound".
        try:
            os.utime(self.pid_path, None)
        except OSError:
            pass
        self._reload_requested = True
        self.running = False

    def _step_runtime_sync(self):
        """Sync before swap, one serve-loop step: start ensure-runtime.sh for
        this tree, or reap it and reload once it has exited.

        The sync runs while this daemon keeps serving, so the swap itself costs
        only teardown + boot, and the launcher's own ensure-runtime.sh finds the
        runtime ready.

        A non-zero exit cancels the reload: ensure-runtime already keeps a
        previously synced venv serving (exit 0), so non-zero means the launcher
        would fail the same way and the daemon would crash-loop. It keeps
        serving the old code instead, loudly; so does a sync that cannot start.
        Nothing here may raise: this runs inside _serve, and an escape would
        crash-loop the supervisor on the still-set request. A shutdown during
        the sync just ends the serve loop; the child finishes on its own.
        """
        if not self.running:
            return  # a shutdown landed this iteration; it wins over the reload
        if self._runtime_sync is None:
            try:
                self._runtime_sync = start_runtime_sync(REPO_ROOT)
            except OSError as e:
                self._cancel_reload("could not start: %s" % e)
            self._sync_requested = False
            return
        code = self._runtime_sync.poll()
        if code is None:
            return
        self._runtime_sync = None
        if code != 0:
            self._cancel_reload("exit %d" % code)
            return
        self._log("Runtime synced — reloading in place after teardown...")
        self._begin_reload()

    def _cancel_reload(self, why: str):
        self._log("Runtime sync FAILED (%s) — reload cancelled; "
                  "still serving the old code" % why)
        try:
            self.brain._log_error(
                'runtime_sync_failed', RuntimeError('ensure-runtime.sh: %s' % why),
                'restart cancelled; daemon serves its previous code')
        except Exception:
            pass

    def _exec_reload(self):
        """Reload in place: exec fresh code into THIS process. Same PID, no
        spawn, platform-uniform — launchd (where present) never notices, so
        there is no KeepAlive respawn to race and no competing-spawner
        surface at all (the disease that killed every earlier restart shape —
        Popen orphans 2026-07-03/04, the two-writer kickstart window
        2026-07-06). The unresponsive window shrinks to teardown + boot
        (~2-4s measured); the structural win is that a deliberate reload can
        no longer be mistaken for a death — no DAEMON_DOWN bursts, nothing
        for recover_daemon to kill.

        Runs on the MAIN thread only, called by start() AFTER the ordered
        `_shutdown()` teardown (drain → save → CLOSE DB → release lock) — so
        the exec'd successor is the only opener of brain.db, and a swallowed
        exec failure can't hide (the 2026-04 execv died silently on a handler
        thread — 7b4a01a).

        Execs hooks/scripts/brain-daemon — the launchd entry point — NOT
        daemon_argv directly: the launcher re-runs the resolver chain
        (resolve-brain-db.sh → brain-env.sh → the user env file), so runtime
        flags, a replaced key file, and a moved brain re-resolve on every
        reload exactly as on a real boot, then it execs the identical
        `python -m servers.daemon_server` incantation. PYTHONPATH is stripped
        so the launcher rebuilds it (it prepends unconditionally — passing it
        through would grow it by one entry per reload).

        The PID file is freshened twice — utime when the restart command is
        accepted (covers the teardown seconds, when the socket is open but
        unanswering) and a rewrite just before the exec (covers boot). Its
        mtime is what recover_daemon's corpse test reads, so the whole
        teardown→exec→boot span reads "just bound — booting, not a corpse".
        A failed exec leaves it to age past the grace budget on its own.

        Caveat: the launcher re-resolves the DB through the ladder, so a
        hand-started daemon (`python -m servers.daemon_server /custom/brain.db`
        with no BRAIN_DB_DIR) that receives `restart` re-points at whatever
        the ladder resolves. Both production routes bake BRAIN_DB_DIR, so the
        hint always matches there.

        Failure falls through to a loud exit: on launchd KeepAlive respawns
        (fresh code from disk anyway); off-launchd the next ensure_daemon
        spawns. Either way the fallback IS the pre-reload restart behavior.
        """
        self._log("Reloading: exec fresh code in place (same PID)...")

        # Clear bytecode cache so the exec'd image compiles fresh code (the
        # point of a reload).
        import shutil
        servers_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(servers_dir)
        for cache_dir in [os.path.join(servers_dir, '__pycache__'),
                          os.path.join(project_dir, 'hooks', 'scripts', '__pycache__')]:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)

        try:
            launcher = os.path.join(REPO_ROOT, 'hooks', 'scripts', 'brain-daemon')
            env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
            os.chdir(REPO_ROOT)
            # Last-instant cancellation check: a SIGTERM (launchctl bootout,
            # logout) landing during teardown cleared the flag — honor it here
            # rather than exec into an unmanaged orphan. Checked before the
            # breadcrumb, so a cancelled reload leaves no PID file naming it.
            if not self._reload_requested:
                self._log("Reload cancelled by a shutdown signal — exiting.")
                os._exit(0)
            with open(self.pid_path, 'w') as f:   # corpse-test breadcrumb
                f.write(str(os.getpid()))
            os.execve(launcher, [launcher], env)
        except Exception as e:
            self._log("Reload exec FAILED: %s — exiting; KeepAlive/"
                      "ensure_daemon respawns fresh code from disk." % e)
            os._exit(0)

    def _run_idle_maintenance(self):
        """Poll brain's maintenance decision. Runs in thread pool.

        The polling cadence is the daemon's responsibility; the decision
        (idle threshold, min interval, persisted last-run timestamp) is
        the brain's. If the brain declines (returns None), this is a
        cheap no-op and we try again on the next poll. Previously the
        daemon owned this logic and lost its last-run state on restart —
        brain.run_maintenance_if_due() persists via brain_meta instead.
        """
        try:
            # Gate reads brain.activity (last_user_activity + encode-runs) and
            # consumes the encode runs on fire — single source of truth.
            result = self.brain.run_maintenance_if_due()
            if result is None:
                return  # not due — quiet no-op
            self._log("Maintenance ran (idle %.0fs): %s" % (
                result.get('idle_seconds', 0),
                ", ".join("%s=%s" % (u, str(r)[:60])
                          for u, r in result.get('units', {}).items())))
            self.brain.save()
        except Exception as e:
            self._log("Maintenance error: {}".format(e))
            try:
                self.brain._log_error('daemon_maintenance_poll', e,
                                      'brain.run_maintenance_if_due raised')
            except Exception:
                pass
        finally:
            self._maintenance_poll_running = False

    def _run_scribe_poll(self):
        """Poll for a session whose S1 Scribe is due and fire it. Thread pool.

        Capped concurrency via _encode_sema — at most MAX_CONCURRENT_ENCODES
        encodes at once across sessions — plus per-session single-flight
        (_encode_inflight) so one session can't double-encode. All permits busy,
        or the only due session already encoding, makes this a cheap no-op: the
        skipped session isn't lost, it re-qualifies and drains on a later poll
        (most-overdue first — the multi-session "queue"). The daemon owns
        concurrency; brain.scribe_due() owns the decision (who's due) via higher
        session functions. The encode runs on its own thread
        (run_unit_in_background), so this poll returns fast; the encode thread
        releases its permit and clears its in-flight slot when it completes.
        """
        import time as _time
        from servers.scales.s1.encode_contract import (
            SCRIBE_RETRY_COOLDOWN_SECONDS, SCRIBE_MAX_FAILED_RETRIES,
            SCRIBE_CANDIDATE_WINDOW_MIN, SCRIBE_HUNG_ALARM_SECONDS,
            MAX_CONCURRENT_ENCODES)
        spawned = False
        acquired = False
        claimed_sid = None
        try:
            # Wall clock for scribe_due only — it compares presence and idle
            # against real time (see its docstring). Every daemon-local elapsed
            # check below is monotonic: those deltas are written and read right
            # here, and on wall clock a backward step would defeat the retry
            # cooldown (re-firing a failing session every tick) and mute the
            # hung-encode alarm. A running scribe_flush moves this clock
            # ahead by the tail-idle period, and says so, since on an eval
            # brain only a flush may encode an idle tail (getattr: the tests
            # drive this method on a bare namespace).
            offset = getattr(self, '_scribe_fast_forward_s', 0)
            now = _time.time() + offset
            mono = _time.monotonic()
            # Hung-encode surfacing: a thread stuck past the alarm still holds one
            # of the MAX_CONCURRENT_ENCODES permits and can't be killed — make the
            # lost slot visible once per incident (per session), not silent. Runs
            # every poll (not only when all permits are busy), so a single stuck
            # encode among free permits is still surfaced.
            for _hsid, _hstart in list(self._encode_inflight.items()):
                if (mono - _hstart > SCRIBE_HUNG_ALARM_SECONDS
                        and not self._encode_hung_warned.get(_hsid)):
                    self._encode_hung_warned[_hsid] = True
                    try:
                        self.brain._log_error(
                            'scribe_hung',
                            RuntimeError('encode running %ds — thread presumed '
                                         'hung; holds 1 of %d encode permits'
                                         % (int(mono - _hstart), MAX_CONCURRENT_ENCODES)),
                            'session=%s' % _hsid[:8])
                    except Exception:
                        pass

            if not self._encode_sema.acquire(blocking=False):
                return  # all MAX_CONCURRENT_ENCODES permits in use — next poll retries
            acquired = True

            # Prune attempts older than the candidate window — those sessions
            # have aged out of consideration anyway (keeps the dicts bounded).
            horizon = SCRIBE_CANDIDATE_WINDOW_MIN * 60
            self._scribe_attempts = {s: t for s, t in self._scribe_attempts.items()
                                     if mono - t < horizon}
            # Keep _scribe_failures keyed to live attempts — a session aged out
            # of _scribe_attempts has no live cooldown, so its failure count is
            # stale (otherwise _scribe_failures would leak entries forever).
            self._scribe_failures = {s: f for s, f in self._scribe_failures.items()
                                     if s in self._scribe_attempts}
            # Cooling = attempted within the cooldown → exclude from selection so
            # a failing (still-"due") session can't monopolize the poll. In-flight
            # sessions are excluded too (per-session single-flight): a session
            # mid-encode stays most-overdue until it completes, so without this it
            # would be re-selected every poll and double-encode.
            cooling = {s for s, t in self._scribe_attempts.items()
                       if mono - t < SCRIBE_RETRY_COOLDOWN_SECONDS}
            skip = cooling | set(self._encode_inflight)

            due = self.brain.scribe_due(now=now, skip_sessions=skip,
                                        flush=bool(offset))
            if not due:
                return
            sid = due['session_id']
            # Claim the in-flight slot before recording the attempt: a session
            # with an attempt and no slot reads as a failed encode (scribe_flush
            # stops waiting for it), so the slot must come first.
            self._encode_inflight[sid] = mono  # per-session single-flight + hung baseline
            self._encode_hung_warned.pop(sid, None)
            claimed_sid = sid

            # Re-firing a session we already attempted (past its cooldown but
            # STILL due) means the prior attempt never advanced the cadence — the
            # encode is crashing or skipping. Count + escalate loudly: the
            # starvation alarm can't catch this (turns is frozen at a fixed value,
            # so its `% ENCODE_EVERY` rate-limit never trips).
            if sid in self._scribe_attempts:
                fails = self._scribe_failures.get(sid, 0) + 1
                self._scribe_failures[sid] = fails
                if fails >= SCRIBE_MAX_FAILED_RETRIES:
                    try:
                        self.brain._log_error(
                            'scribe_repeated_failure',
                            RuntimeError('session %s re-fired %d times without '
                                         'advancing the encode cadence — encode '
                                         'is crashing or skipping' % (sid[:8], fails)),
                            'check s1e_* errors; turns_since_last_encode is stuck')
                    except Exception:
                        pass
            self._scribe_attempts[sid] = mono

            from servers.scales.s1.scribe import S1Scribe
            from servers.scales.runner import run_unit_in_background

            def _count_encode(write_actions, _sid=sid):
                # on_complete fires only when the encode COMPLETED — a crash
                # skips it (run_unit_in_background's except). So any call here
                # means the cadence advanced (encoding_prompt was written), even
                # for a 0-write encode. Clear the cooldown/failure state
                # unconditionally — only a genuine crash (no on_complete) leaves
                # the session cooling, which is exactly what should bound the
                # retry. (Clearing only on write>0 falsely escalated a healthy
                # 0-write encode on its next legit re-fire.) record_encode_run
                # stays gated on write_actions — the S2 gate cares only about
                # material actually written.
                self._scribe_attempts.pop(_sid, None)
                self._scribe_failures.pop(_sid, None)
                if write_actions > 0:
                    self.brain.activity.record_encode_run()

            def _release_slot(_sid=sid):
                # Runs in the encode thread's finally (success OR crash), beside
                # the permit release, so a crashed encode can't leave a session
                # wedged as in-flight (blocked from ever re-encoding).
                self._encode_inflight.pop(_sid, None)
                self._encode_hung_warned.pop(_sid, None)

            scribe = S1Scribe(self.brain, session_id=sid, counter=due['counter'])
            run_unit_in_background(scribe, name='s1e', lock=self._encode_sema,
                                   on_complete=_count_encode, on_release=_release_slot)
            spawned = True  # permit + in-flight slot ownership transferred to the thread
        except Exception as e:
            try:
                self.brain._log_error('scribe_poll', e, 'Scribe reactor poll')
            except Exception:
                pass
        finally:
            # Release the permit only if we acquired it but did NOT hand it to an
            # encode thread (nothing due, or an error before spawn). Undo any
            # in-flight slot claimed in the same breath — a pre-spawn failure
            # would otherwise wedge that session out of encoding forever. A failed
            # release means the semaphore state is corrupt and the encoder is
            # jammed — log loud, don't swallow.
            if acquired and not spawned:
                if claimed_sid is not None:
                    self._encode_inflight.pop(claimed_sid, None)
                    self._encode_hung_warned.pop(claimed_sid, None)
                try:
                    self._encode_sema.release()
                except Exception as _re:
                    try:
                        self.brain._log_error(
                            'scribe_lock_release_failed', _re,
                            'encode permit release failed — encoder may be jammed')
                    except Exception:
                        pass
            self._scribe_poll_running = False

    def _scribe_flush(self, timeout_s: float) -> Dict[str, Any]:
        """Fast-forward idle: the Scribe poll runs as if every live stream had
        gone quiet for the tail-idle period, until nothing is due and nothing is
        encoding. The settle step of an eval golden needs it: a snapshot taken
        mid-exchange holds a turn production encodes only after an hour of
        quiet, and every arm cloned from it would otherwise encode that turn on
        its own, differently.

        Same selection as production's tail (more than SCRIBE_TAIL_MIN_TURNS
        pending; a stream with fewer stays pending, as it would in production),
        through the one poll: the regular tick keeps firing it, with its permits,
        per-session single-flight and retry cooldown, on the moved clock. The
        boot grace passes too. A session whose encode fails is cooling when the
        wait ends and comes back as failed.

        On an eval brain this is the only way an idle tail encodes, and only
        the eval's own sessions are candidates (Brain.eval_brain_since).
        """
        import time as _time
        from servers.scales.s1.encode_contract import (
            SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_RETRY_COOLDOWN_SECONDS)
        if getattr(self, '_scribe_fast_forward_s', 0):
            return {"ok": False, "error": "a scribe_flush is already running"}
        offset = SCRIBE_TAIL_IDLE_SECONDS + 1

        def _due(skip):
            return self.brain.scribe_due(now=_time.time() + offset,
                                         skip_sessions=skip, flush=True)

        due_at_start = []
        while True:
            d = _due(set(due_at_start))
            if not d:
                break
            due_at_start.append(d['session_id'])

        t0 = _time.monotonic()
        timed_out = False
        cooling = set()
        self._scribe_fast_forward_s = offset
        try:
            while True:
                # Order matters: the poll claims a session's in-flight slot
                # before its attempt, so reading attempts, then due, then the
                # slots last cannot miss an encode that is just starting.
                # list(): one C-level copy — the poll and the encode threads
                # write this dict while the flush reads it.
                mono = _time.monotonic()
                cooling = {s for s, t in list(self._scribe_attempts.items())
                           if mono - t < SCRIBE_RETRY_COOLDOWN_SECONDS}
                if not _due(cooling) and not self._encode_inflight:
                    break
                if mono - t0 > timeout_s:
                    timed_out = True
                    break
                _time.sleep(0.5)
        finally:
            self._scribe_fast_forward_s = 0
        return {"ok": True, "result": {
            "due_at_start": due_at_start,
            "failed": sorted(cooling - set(self._encode_inflight)),
            "timed_out": timed_out,
            "llm_available": bool(self.brain.llm_available),
            "elapsed_s": round(_time.monotonic() - t0, 1)}}

    def _claim_pid_file(self):
        """Write this process's PID into the instance's PID file. Called once
        both flocks are held, so no other daemon of this instance is alive
        and the file is this process's to write, from here until _cleanup
        removes it.

        Claiming at the lock rather than at bind is what makes the file true
        for the whole life: a loading daemon reads to recover_daemon's corpse
        test as freshly claimed (booting), not as absent or as its dead
        predecessor's stale file. _run rewrites it at bind, and _exec_reload
        rewrites it just before the exec.

        What it held before says how the last owner ended. Its own PID means
        an exec reload handed over. Any other content means the last owner
        never ran _cleanup (a SIGKILL, a crash or a forced exit). A SIGKILL
        leaves no line in daemon.log, so that is recorded where the
        forced-restart rows are."""
        try:
            with open(self.pid_path) as f:
                prev = f.read().strip()
            age = time.time() - os.stat(self.pid_path).st_mtime
        except OSError:
            prev = ""
        if prev and prev != str(os.getpid()):
            msg = ("previous daemon (PID file said %r) exited without cleanup; "
                   "the file was %.0fs old" % (prev, age))
            self._log("Unclean predecessor: " + msg)
            from .daemon_client import log_hook_error
            log_hook_error("DAEMON_DIED_UNCLEAN", msg,
                           context="SIGKILL, a crash, or a forced exit; daemon.log "
                                   "before this start says which, if anything",
                           level="error", db_dir=os.path.dirname(self.db_path))
        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))

    def _owns_pid_file(self):
        """Whether the PID file names this process — the one ownership test
        for the PID and status files."""
        try:
            with open(self.pid_path) as f:
                return f.read().strip() == str(os.getpid())
        except OSError:
            return False

    def _handle_signal(self, signum, frame):
        self._log("Received signal {}".format(signum))
        # A signal-driven shutdown is a shutdown: cancel any in-flight reload
        # so a SIGTERM (launchctl bootout, logout, system shutdown) landing
        # after a `restart` command can't be overridden by the exec.
        self._reload_requested = False
        self.running = False
        # Close server socket immediately to unblock select() and reject new connections
        try:
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception:
            pass

    def _arm_force_exit_backstop(self):
        """Last-resort guard: if a drain wedges (e.g. an embedder CPU spin), force
        the process to exit so teardown always terminates. SHUTDOWN_BACKSTOP_S is
        sized to let a normal in-flight command finish first, yet stay under
        launchd's SIGTERM→SIGKILL window so we exit on our own terms."""
        import _thread
        def _force_exit():
            time.sleep(self.SHUTDOWN_BACKSTOP_S)
            self._log("Teardown wedged — forcing exit after %ds" % self.SHUTDOWN_BACKSTOP_S)
            os._exit(0)
        _thread.start_new_thread(_force_exit, ())

    def _signal_drain_shutdown(self):
        """Signal the single bg-writer drain worker to stop (it drains BOTH the
        embed and recall-write queues, so one signal covers both). Wakes it out
        of its interval wait so it exits at the next check. Idempotent (the Event
        is the single signal). Loud on failure."""
        try:
            from servers import embed_queue
            embed_queue.request_shutdown()
        except Exception as e:
            self._log_shutdown_error('queue_shutdown', e)

    def _teardown_brain(self):
        """The single ordered teardown: drain everything holding a DB connection,
        save+close, release the singleton lock/PID LAST. Never raises — every step
        is guarded and logged. Order is load-bearing:
          1. signal queues    → wakes the bg-writer so join_worker can return
          2. drain pool       → in-flight commands finish on OPEN connections
          3. join bg-writer   → its in-flight drain settles off conn_bg_writer
          4. save + close     → no live consumer touches the connections now;
                                close() flushes the cached SessionContexts first
          5. release lock/PID → LAST: a racer's ensure_daemon blocks on this lock
             before spawning, so it can't open a 2nd writer on brain.db while our
             connections were still open (two writers corrupt the indexes).
        This is the fix for the 'Cannot operate on a closed database' boot failure
        of 2026-06-06, where close() ran before the pool + bg-writer were drained."""
        self._signal_drain_shutdown()
        # Drain the pool. Catch EVERYTHING (not only the cancel_futures TypeError)
        # so a pool-drain error can never skip the save+close below.
        try:
            try:
                self._pool.shutdown(wait=True, cancel_futures=False)
            except TypeError:                       # Python <3.9 has no cancel_futures
                self._pool.shutdown(wait=True)
        except Exception as e:
            self._log_shutdown_error('pool_drain', e)
        try:
            from servers import embed_queue
            embed_queue.join_worker(timeout=3.0)
        except Exception as e:
            self._log_shutdown_error('bg_writer_join', e)
        try:
            if self.brain:
                self.brain.save()
                self.brain.close()
                self.brain = None
        except Exception as e:
            self._log_shutdown_error('brain_save_close', e)
        # Resources LAST — see step 5. _cleanup re-signals the queues (idempotent)
        # and releases the socket, PID file, and the singleton fcntl lock.
        self._cleanup()

    def _shutdown(self):
        """Clean shutdown: arm the force-exit backstop, then run the shared
        drain-then-close teardown (see _teardown_brain for the ordering).

        Idempotent — it runs at _serve's tail AND at start()'s tail on every
        clean path. Before this guard the second call re-tore-down (harmless)
        but armed a SECOND 15s force-exit backstop; on the reload path that
        obscured the real teardown budget (one backstop, one clock)."""
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        self._log("Shutting down...")
        self._arm_force_exit_backstop()
        self._teardown_brain()

    def _cleanup(self):
        """Close server socket, observer channel, remove PID and lock files.
        Also signals background-writer queues to stop draining so the worker
        thread exits cleanly at its next interval check (daemon thread is
        force-killed on process exit anyway, but signaling lets a partial
        drain finish without crash-rollback).
        Idempotent — safe to call multiple times (signal + atexit + explicit)."""
        self._signal_drain_shutdown()
        self._close_socket()
        # These files belong to the process the PID file names. A duplicate
        # that exited at the flock never claimed it, so it must not unlink the
        # incumbent's PID/status out from under it.
        if self._owns_pid_file():
            for path in [self.pid_path, get_status_path()]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as _ue:
                    # Stale PID/status files cause "Another daemon running"
                    # errors on next boot. Silent failure here hides that
                    # class of bug. Stderr is the right channel (we may
                    # not have a brain handle at this point in shutdown).
                    print('[brain-daemon] failed to remove %s: %s' %
                          (path, _ue), file=sys.stderr)
        try:
            # Idempotent: _cleanup runs more than once (explicit _shutdown +
            # atexit). A closed file object is still truthy, so the bare
            # `and self._lock_fd` guard let the 2nd call re-flock an
            # already-closed fd → "I/O operation on closed file" — dozens of
            # false alarms that looked like a lock leak but weren't. Skip when
            # already closed; null in finally so later calls short-circuit.
            if getattr(self, '_lock_fd', None) and not self._lock_fd.closed:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                self._lock_fd.close()
        except Exception as _le:
            # A GENUINE release failure (not the harmless double-call above).
            # A leaked lock prevents daemon restart, so surface it where the
            # operator can SEE it (the dashboard reads debug_log, not stderr).
            self._log_shutdown_error('release_lock_fd', _le)
        finally:
            self._lock_fd = None
        try:
            # Same idempotence contract as the singleton lock above.
            if getattr(self, '_db_lock_fd', None) and not self._db_lock_fd.closed:
                fcntl.flock(self._db_lock_fd, fcntl.LOCK_UN)
                self._db_lock_fd.close()
        except Exception as _le:
            self._log_shutdown_error('release_db_lock_fd', _le)
        finally:
            self._db_lock_fd = None

    def _log_shutdown_error(self, source, err):
        """Surface a shutdown-time error to debug_log (which the dashboard
        reads) AND stderr. brain is already closed during _cleanup, so write
        via a fresh connection — the way log_hook_error logs without a Brain
        handle. Never raises: logging must not crash shutdown."""
        print('[brain-daemon] %s: %s' % (source, err), file=sys.stderr)
        try:
            import sqlite3
            from .clock import iso_now
            logs_db = os.path.join(os.path.dirname(self.db_path), 'brain_logs.db')
            conn = sqlite3.connect(logs_db, timeout=5)
            conn.execute(
                "INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) "
                "VALUES ('daemon', 'error', ?, ?, ?)",
                (source, json.dumps({'error': str(err), 'type': type(err).__name__}), iso_now()))
            conn.commit()
            conn.close()
        except Exception:
            pass  # stderr already carries it

    def _log(self, message: str):
        print("[brain-daemon] {}".format(message), file=sys.stderr)


def main(argv=None):
    """The daemon's entry point: `python -m servers.daemon_server <brain.db>`.

    The SINGLE boot incantation. Both spawn routes exec exactly this — launchd
    through hooks/scripts/brain-daemon, and the no-launchd fallback through
    daemon_launch's hardened detached spawn (daemon_argv + daemon_env).

    The caller supplies the environment (PYTHONPATH, BRAIN_DB_DIR, the CPU-only
    invariant) — see daemon_launch.daemon_env. Nothing is resolved here: a
    daemon that picked its own DB would be a second resolver.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or not argv[0]:
        sys.stderr.write(
            "usage: python -m servers.daemon_server <path to brain.db>\n")
        return 2
    daemon_log.stamp_std_streams()
    BrainDaemon(argv[0]).start()
    return 0
