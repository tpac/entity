"""Database maintenance scheduler — backend-agnostic.

Owns a background thread that fires checkpoint, optimize and analyze
operations on registered databases at fixed cadences. Backend-specific work
(SQLite's `PRAGMA wal_checkpoint`, `PRAGMA optimize`, `ANALYZE`) is
delegated to a backend module that implements the BackendOps protocol. If
the brain ever moves off SQLite, swap the backend module and this scheduler
stays as-is.

Scope:
- `checkpoint` every 5 min — flush WAL into DB and truncate the WAL file.
  A checkpoint that cannot shrink the WAL for consecutive runs raises a
  loud wal-pinned alert (a long-lived reader — historically the
  dashboard's ro connections — is starving truncation).
- `optimize` every 30 min — refresh planner statistics for tables whose
  size moved ~10x (cheap; usually a no-op).
- `analyze` weekly and once after each boot — exact planner statistics for
  every table, bounding the drift optimize's 10x trigger allows (2–4s of
  write lock across both DBs at 2026-09 sizes).
- `quick_check` weekly — structural integrity scan; corruption raises
  into the error path instead of surfacing months later as weird recall.
- `stats` collected every checkpoint tick and logged for observability.

Out of scope (call out if you add them):
- `VACUUM` (manual via maintenance-lock file; locks the DB — pair with
  post-purge migrations, useless on a DB with an empty freelist)

The thread is `daemon=True` so it never blocks shutdown. All operation
failures are caught and logged via brain._log_error — a single bad tick
must not kill the loop.

This module imports the backend lazily via `db_backends.current` so the
scheduler doesn't need to know which backend is active.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Protocol

# The backup cadence is backup policy and lives in db_backup (the freshness
# gate there is defined as a multiple of it); imported, not duplicated.
from .db_backup import BACKUP_INTERVAL_S as _DEFAULT_BACKUP_INTERVAL_S


class BackendOps(Protocol):
    """Interface every backend implements. Methods take a db identifier
    (path for SQLite, conn string for others) and return a dict of
    diagnostic info or raise on error."""

    def apply_pragmas(self, conn) -> None: ...
    def checkpoint(self, db_path: str) -> Dict: ...
    def optimize(self, db_path: str) -> Dict: ...
    def analyze(self, db_path: str) -> Dict: ...
    def quick_check(self, db_path: str) -> Dict: ...
    def stats(self, db_path: str) -> Dict: ...


# Default cadences (seconds). Override per-DB via register() if needed.
_DEFAULT_CHECKPOINT_INTERVAL_S = 5 * 60
_DEFAULT_OPTIMIZE_INTERVAL_S = 30 * 60
_DEFAULT_QUICK_CHECK_INTERVAL_S = 7 * 24 * 60 * 60
_DEFAULT_ANALYZE_INTERVAL_S = 7 * 24 * 60 * 60
# Consecutive checkpoints that failed to shrink a non-empty WAL before the
# wal-pinned alert fires. 3 × 5 min ≈ 15 minutes of continuous pinning —
# past any legitimate long read, short enough to catch a leaked read txn
# while the holder is still alive to inspect.
_WAL_PINNED_ALERT_STREAK = 3
# Worker wakes every ~30s and checks "is anything due". Short enough to
# stay responsive to shutdown; long enough that the scheduler is cheap.
_TICK_INTERVAL_S = 30.0


class DBMaintenance:
    """Background scheduler. One per daemon process.

    Usage:
        m = DBMaintenance(log_fn=daemon._log,
                          log_error_fn=brain._log_error)
        m.register('brain', brain.db_path)
        m.register('brain_logs', logs_db_path)
        m.start()
        ...
        m.stop()  # signals worker to exit at next tick
    """

    def __init__(self,
                 log_fn: Optional[Callable[[str], None]] = None,
                 log_error_fn: Optional[Callable[[str, Exception, str], None]] = None,
                 checkpoint_interval_s: float = _DEFAULT_CHECKPOINT_INTERVAL_S,
                 optimize_interval_s: float = _DEFAULT_OPTIMIZE_INTERVAL_S,
                 quick_check_interval_s: float = _DEFAULT_QUICK_CHECK_INTERVAL_S,
                 analyze_interval_s: float = _DEFAULT_ANALYZE_INTERVAL_S,
                 tick_interval_s: float = _TICK_INTERVAL_S):
        self._log = log_fn or (lambda m: print('[db_maintenance] %s' % m, flush=True))
        self._log_error = log_error_fn  # may be None during early boot
        self._checkpoint_interval_s = checkpoint_interval_s
        self._optimize_interval_s = optimize_interval_s
        self._quick_check_interval_s = quick_check_interval_s
        self._analyze_interval_s = analyze_interval_s
        self._tick_interval_s = tick_interval_s

        self._registered: List[Dict] = []
        self._tasks: List[Dict] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Resolve backend lazily so unit tests can monkey-patch.
        self._backend = None

    def register(self, name: str, db_path: str,
                 backup_dir: Optional[str] = None,
                 backup_interval_s: Optional[float] = None,
                 backup_keep: Optional[Dict[str, int]] = None) -> None:
        """Add a database to the maintenance schedule.

        `name` is for log readability ('brain', 'brain_logs'). `db_path`
        is what the backend operates on. Subsequent registers with the
        same name replace the entry (idempotent boot).

        When `backup_dir` is given, the DB is also snapshotted (compressed,
        GFS-pruned) every `backup_interval_s` into that dir. `backup_keep`
        is `{'daily': N, 'weekly': N, 'monthly': N}` (defaults applied by
        db_backup). The backup clock is seeded from the newest existing
        snapshot so a daemon restart doesn't re-snapshot a just-backed-up
        DB — no persisted scheduler state needed."""
        self._registered = [r for r in self._registered if r['name'] != name]
        entry: Dict = {
            'name': name,
            'db_path': db_path,
            'last_checkpoint_at': 0.0,
            'last_optimize_at': 0.0,
            # 0.0 → first quick_check fires on the first due tick after boot:
            # cheap, and every boot gets a fresh integrity baseline.
            'last_quick_check_at': 0.0,
            # Monotonic (process time: stats drift only while the daemon
            # writes). -inf = never ran, so it fires on the first due tick
            # after boot — the daemon restarts more often than weekly, so a
            # clock started at boot would never come due. That boot run also
            # replaces statistics left by older builds.
            'last_analyze_at': float('-inf'),
            'wal_pinned_streak': 0,
        }
        if backup_dir:
            from . import db_backup
            entry['backup_dir'] = backup_dir
            entry['backup_interval_s'] = backup_interval_s or _DEFAULT_BACKUP_INTERVAL_S
            # Merge any override onto the defaults so all three tiers are
            # always present — _run_backup can then read keys directly.
            entry['backup_keep'] = {**db_backup._DEFAULT_KEEP, **(backup_keep or {})}
            age = db_backup.seconds_since_last_backup(db_path, backup_dir)
            entry['last_backup_at'] = (
                time.time() - age) if age != float('inf') else 0.0
        self._registered.append(entry)

    def register_task(self, name: str, fn, interval_s: float) -> None:
        """Schedule a periodic callable on this thread.

        For maintenance that isn't a backend DB op — policy work that needs
        a reliable cadence but knows about the brain (log retention, orphan
        sweeps). The scheduler owns timing, stamping and error isolation and
        never inspects what `fn` does, so `register`'s backend contract stays
        db_path-only and the abstraction survives a backend swap.

        Same replace-by-name idempotence as `register`, so a re-run of boot
        wiring doesn't double-schedule.
        """
        self._tasks = [t for t in self._tasks if t['name'] != name]
        self._tasks.append({'name': name, 'fn': fn,
                            'interval_s': interval_s, 'last_run_at': 0.0})

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='db-maintenance')
        self._thread.start()
        self._log('started — %d DBs registered, checkpoint=%ds optimize=%ds '
                  'quick_check=%ds analyze=%ds' % (
                      len(self._registered),
                      int(self._checkpoint_interval_s),
                      int(self._optimize_interval_s),
                      int(self._quick_check_interval_s),
                      int(self._analyze_interval_s)))

    def stop(self) -> None:
        self._running = False

    # ─── Internal ─────────────────────────────────────────────────────

    def _resolve_backend(self):
        if self._backend is None:
            from . import db_backends
            self._backend = db_backends.current
        return self._backend

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._tick_interval_s)
            if not self._running:
                break
            try:
                self._tick()
            except Exception as e:
                # Tick failures must never kill the loop.
                self._report_error('db_maintenance_tick', e, 'top-level tick caught')

    def _tick(self) -> None:
        now = time.time()
        mono = time.monotonic()
        backend = self._resolve_backend()
        for entry in self._registered:
            if now - entry['last_checkpoint_at'] >= self._checkpoint_interval_s:
                self._run_op(backend, entry, 'checkpoint', now)
            if now - entry['last_optimize_at'] >= self._optimize_interval_s:
                self._run_op(backend, entry, 'optimize', now)
            if now - entry['last_quick_check_at'] >= self._quick_check_interval_s:
                self._run_op(backend, entry, 'quick_check', now)
            if mono - entry['last_analyze_at'] >= self._analyze_interval_s:
                self._run_op(backend, entry, 'analyze', mono)
            if (entry.get('backup_dir') and
                    now - entry.get('last_backup_at', 0.0) >= entry['backup_interval_s']):
                self._run_backup(entry, now)
        for task in self._tasks:
            if now - task['last_run_at'] >= task['interval_s']:
                self._run_task(task, now)

    def _run_task(self, task: Dict, now: float) -> None:
        # Stamp before running, for the same reason _run_op does — a failing
        # task must wait out its interval rather than retry every tick.
        task['last_run_at'] = now
        t0 = time.time()
        try:
            result = task['fn']()
            took_ms = int((time.time() - t0) * 1000)
            self._log('task %s ok in %dms: %s' % (
                task['name'], took_ms, _short(result)))
        except Exception as e:
            took_ms = int((time.time() - t0) * 1000)
            self._report_error(
                'db_maintenance_task', e,
                'task=%s took=%dms' % (task['name'], took_ms))

    def _run_op(self, backend, entry: Dict, op_name: str, now: float) -> None:
        # Stamp the attempt time BEFORE running, not on success. A failed
        # op (e.g. a contended `database is locked`) must reschedule to the
        # next interval — not stay "due" and retry every tick. Stamping on
        # success turned a transient lock into a hot retry loop that ran an
        # 80s lock-hammer back-to-back for ~20 min, starving foreground
        # recall and tripping DAEMON_DOWN. See git log for the incident.
        entry['last_%s_at' % op_name] = now
        t0 = time.time()
        try:
            result = getattr(backend, op_name)(entry['db_path'])
            took_ms = int((time.time() - t0) * 1000)
            self._log('%s %s ok in %dms: %s' % (
                op_name, entry['name'], took_ms, _short(result)))
            if op_name == 'checkpoint':
                self._track_wal_pinning(entry, result)
        except Exception as e:
            took_ms = int((time.time() - t0) * 1000)
            self._report_error(
                'db_maintenance_%s' % op_name, e,
                'db=%s took=%dms' % (entry['name'], took_ms))

    def _track_wal_pinning(self, entry: Dict, result: Dict) -> None:
        """Loud alert when checkpoints stop shrinking a non-empty WAL.

        A TRUNCATE checkpoint that leaves the WAL at its prior size means a
        long-lived read transaction is pinning it (historically: dashboard
        ro connections, or a leaked cursor). One occurrence is normal; a
        streak means the WAL grows without bound and recovery time with it.
        Alert once per streak (at the threshold), reset on any success.
        """
        before = result.get('wal_size_before') or 0
        after = result.get('wal_size_after') or 0
        pinned = before > 0 and after >= before
        if not pinned:
            entry['wal_pinned_streak'] = 0
            return
        entry['wal_pinned_streak'] += 1
        if entry['wal_pinned_streak'] == _WAL_PINNED_ALERT_STREAK:
            self._report_error(
                'db_maintenance_wal_pinned',
                RuntimeError(
                    '%s WAL pinned at %d bytes across %d checkpoints — a '
                    'long-lived reader is starving truncation' % (
                        entry['name'], after, _WAL_PINNED_ALERT_STREAK)),
                'db=%s; suspect a long-lived read txn (dashboard ro conn, '
                'leaked cursor)' % entry['name'])

    def _run_backup(self, entry: Dict, now: float) -> None:
        # Attempt-stamp first, same discipline as _run_op: a failed backup
        # waits the full interval rather than re-snapshotting every tick.
        entry['last_backup_at'] = now
        t0 = time.time()
        try:
            from . import db_backup
            keep = entry['backup_keep']
            result = db_backup.backup_database(
                entry['db_path'], entry['backup_dir'],
                keep_daily=keep['daily'],
                keep_weekly=keep['weekly'],
                keep_monthly=keep['monthly'])
            took_ms = int((time.time() - t0) * 1000)
            self._log('backup %s ok in %dms: %s' % (
                entry['name'], took_ms, _short(result)))
        except Exception as e:
            took_ms = int((time.time() - t0) * 1000)
            self._report_error(
                'db_maintenance_backup', e,
                'db=%s took=%dms' % (entry['name'], took_ms))

    def _report_error(self, origin: str, e: Exception, context: str) -> None:
        if self._log_error is not None:
            try:
                self._log_error(origin, e, context)
                return
            except Exception:
                pass
        # Fallback: stderr via the regular log function.
        self._log('ERROR %s: %s (%s)' % (origin, e, context))


def _short(d: Dict) -> str:
    """Compact dict repr for log lines — values truncated."""
    parts = []
    for k, v in (d or {}).items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + '...'
        parts.append('%s=%s' % (k, s))
    return ' '.join(parts)
