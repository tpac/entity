"""SQLite backend ops + pragma centralization.

Single source of truth for:
- Connection pragmas applied at every `sqlite3.connect` site in the brain
- Maintenance operations (checkpoint, optimize, analyze) invoked by the scheduler

If we ever swap SQLite for another store, this is the file that gets
replaced. The brain calls `apply_pragmas(conn)` from this module; the
scheduler calls `checkpoint(db_path)` / `optimize(db_path)` /
`analyze(db_path)`. Neither caller knows about SQLite-specific syntax.

Pragma reasoning:
- busy_timeout=30000: tolerate short-lived WAL writer-slot contention
  before raising "database is locked". Default 0 fails immediately.
- cache_size=-65536: 64 MB per-connection page cache (default is 2 MB).
  Brain.db is ~340 MB; this caches the hot working set.
- mmap_size=256 MB: memory-map the DB file so reads go through page
  cache instead of read() syscalls. Big win for read-heavy recall.
- temp_store=MEMORY: temp tables (ORDER BY without index, complex
  joins) live in RAM, not on disk.
- synchronous=NORMAL: with WAL, NORMAL only fsyncs at checkpoint, not
  at every commit. 5-10x faster commits with no durability loss for a
  daemon that can replay from journal. This pragma matters most for
  reducing time-under-write-lock during commit-heavy workloads.
- journal_mode=WAL: concurrent readers + faster commits. Set per-file
  (not per-connection) so we only need it once per DB; included here
  for completeness and so fresh DBs get it on first connect.
- foreign_keys=ON: enforce FK constraints (off by default in SQLite!).
"""

from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
from typing import Any, Dict


# ─── Transaction discipline ───────────────────────────────────────────
# brain_batch wraps many sub-ops in one BEGIN IMMEDIATE / COMMIT envelope,
# and multi-statement writers (cascade delete, revise, edge-text notify)
# commit once at their end. For either to be atomic, the DAL writers running
# inside must NOT self-commit. A per-call `commit` kwarg is one missed caller
# away from silent corruption, so the batch state is a property of the
# CONNECTION, not of each call. `in_batch` lives on the connection; the envelope owner flips
# it; writers consult it via commit_unless_batched(). No kwarg to forget.
#
# Why a subclass and not `conn.in_transaction`: under SQLite's default
# deferred isolation `in_transaction` is True after ANY DML, so it can't
# distinguish "inside an explicit batch" from "mid standalone write" — a
# naive check there would never commit standalone writes (data loss).
# And a plain sqlite3.Connection rejects arbitrary attributes, so the
# flag needs a subclass carrying a real __dict__.


class BatchAwareConnection(sqlite3.Connection):
    """sqlite3 connection that knows whether it's inside a batch envelope.

    `in_batch=False` by default → writers self-commit (standalone behavior).
    A batch owner (dispatch_write._handle_brain_batch, and the multi-statement
    writers in brain_remember / brain_connections / dal_graph) sets
    `in_batch=True` for the duration of its envelope and resets it in a
    finally. Writers gate their commit on this via
    commit_unless_batched() — see that helper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_batch = False


def commit_unless_batched(conn: sqlite3.Connection) -> None:
    """Commit `conn` unless it's inside a batch envelope (conn.in_batch).

    The single source of truth for write-path commit discipline. EVERY DAL
    writer ends with this instead of a bare `conn.commit()`, so a caller can
    no longer break batch atomicity by forgetting a kwarg — the writer reads
    the connection's batch state itself.

    Non-BatchAware connections (some maintenance/test paths) have no
    `in_batch` attribute → getattr returns False → treated as standalone →
    commit. That's the safe default: a stray writer on a plain connection
    behaves exactly as before this change.
    """
    if not getattr(conn, 'in_batch', False):
        conn.commit()


def rollback_unless_batched(conn: sqlite3.Connection) -> None:
    """Discard `conn`'s pending statements unless it's inside a batch envelope.

    The mirror of `commit_unless_batched`, and it exists for the same reason: a
    writer that raises part-way through a multi-statement write must not leave
    an uncommitted prefix behind. On a default-isolation connection those rows
    stay pending and are committed by the NEXT unrelated write on the same
    connection — silently publishing a partial set.

    Batch-aware: inside an envelope the OUTER transaction owns rollback, so
    this is a no-op there (rolling back would destroy the caller's other work).
    A rollback with no open transaction is itself a no-op, so this is safe to
    call unconditionally on the failure path.
    """
    if not getattr(conn, 'in_batch', False):
        conn.rollback()


# Per-connection pragmas. Applied at every sqlite3.connect site via
# apply_pragmas(conn). Reset when the connection closes, so every new
# connection needs them.
_CONNECTION_PRAGMAS = (
    'PRAGMA busy_timeout = 30000',
    'PRAGMA cache_size = -65536',          # 64 MB
    'PRAGMA mmap_size = 268435456',        # 256 MB
    'PRAGMA temp_store = MEMORY',
    'PRAGMA synchronous = NORMAL',
    'PRAGMA foreign_keys = ON',
)

# Per-database pragmas. Persisted in the DB file; setting from any
# connection sticks across processes.
_DATABASE_PRAGMAS = (
    'PRAGMA journal_mode = WAL',
)


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply the standard pragma set to a SQLite connection.

    Idempotent. Call once per connection, immediately after connect().
    Order matters: journal_mode goes first because cache_size and
    mmap_size behave differently in DELETE vs WAL.
    """
    for stmt in _DATABASE_PRAGMAS:
        conn.execute(stmt)
    for stmt in _CONNECTION_PRAGMAS:
        conn.execute(stmt)


# ─── Maintenance ops ──────────────────────────────────────────────────
# Each op opens a short-lived connection, runs its work, closes. This
# keeps the maintenance thread decoupled from the daemon's primary
# connections — no shared state, no contention with the worker, no
# lock plumbing across the module boundary.

# Maintenance must yield FAST under contention. The daemon's primary
# connections wait up to busy_timeout=30s for the lock — correct for a
# user-facing write that must not be dropped. A background ANALYZE or
# checkpoint has no such obligation: if it can't get the lock quickly it
# should give up and let the next cycle retry. With the 30s default,
# `PRAGMA optimize` on a hot DB blocked ~80s (multiple internal ANALYZE
# statements, each waiting the full timeout) and starved the foreground.
# Cap maintenance connections far lower so a contended op fails in seconds.
_MAINTENANCE_BUSY_TIMEOUT_MS = 5000


def connect_maintenance(db_path: str) -> sqlite3.Connection:
    """Open a connection for a background maintenance op with a short
    busy_timeout — it yields fast under contention instead of blocking
    the foreground writers. Applies the standard pragma set, then lowers
    busy_timeout from the 30s default to the maintenance cap.

    This is how maintenance runs against a LIVE database: WAL admits one
    writer alongside readers, and a private connection means a private
    transaction — so a sweep can never join, commit or roll back work a
    foreground writer has open. Never hand a maintenance op `brain.conn`;
    its bare commits would land on whatever envelope that connection is
    holding.
    """
    conn = sqlite3.connect(db_path, timeout=_MAINTENANCE_BUSY_TIMEOUT_MS / 1000.0)
    apply_pragmas(conn)
    conn.execute('PRAGMA busy_timeout = %d' % _MAINTENANCE_BUSY_TIMEOUT_MS)
    return conn


def checkpoint(db_path: str) -> Dict[str, Any]:
    """`PRAGMA wal_checkpoint(TRUNCATE)` — flush WAL into the DB and
    truncate the WAL file. Without periodic checkpointing the WAL
    grows unbounded; growing WAL means longer recovery on every
    connection open, which compounds writer-slot wait time.

    Returns: dict with 'busy', 'log_pages', 'checkpointed_pages',
    plus 'wal_size_before' / 'wal_size_after' for observability.
    """
    wal_path = db_path + '-wal'
    size_before = _file_size_or_zero(wal_path)
    conn = connect_maintenance(db_path)
    try:
        row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
    finally:
        conn.close()
    size_after = _file_size_or_zero(wal_path)
    busy, log_pages, ckpt_pages = (row or (None, None, None))
    return {
        'busy': busy,
        'log_pages': log_pages,
        'checkpointed_pages': ckpt_pages,
        'wal_size_before': size_before,
        'wal_size_after': size_after,
    }


def optimize(db_path: str) -> Dict[str, Any]:
    """`PRAGMA optimize=0x10002` — re-analyze any table whose size moved
    ~10x since its statistics were gathered, or that has none yet.

    The mask is load-bearing. Bare `PRAGMA optimize` only considers tables
    the planner used on THIS connection, and a maintenance connection is
    fresh every call, so the bare form never re-analyzes a populated table.
    0x10000 checks every table's size instead. Leaving out 0x10 drops the
    temporary row cap: capped sampling truncates the per-value estimates
    (the `... 2001` rows in sqlite_stat1), and those mis-plan IN-list
    lookups as index scans. Usually a ~0.1ms no-op; when a table does
    qualify it is an exact ANALYZE of that table, and open connections are
    told to re-read the statistics. `analyze` bounds the drift this 10x
    trigger allows.
    """
    conn = connect_maintenance(db_path)
    try:
        before = _stat1_rows(conn)
        conn.execute('PRAGMA optimize=0x10002')
        refreshed = _stat1_rows(conn) != before
        if refreshed:
            _signal_stats_reload(conn)
    finally:
        conn.close()
    return {'ok': True, 'stats_refreshed': refreshed}


def analyze(db_path: str) -> Dict[str, Any]:
    """Full `ANALYZE` — exact planner statistics for every table and index,
    then a signal so open connections re-read them.

    Holds the write lock for its duration (0.2–1.5s brain.db, ~2s
    brain_logs.db at 2026-09 sizes, growing with the data), so it runs on a
    slow cadence; the maintenance busy_timeout lets it yield when contended.
    """
    conn = connect_maintenance(db_path)
    try:
        conn.execute('ANALYZE')
        conn.commit()
        _signal_stats_reload(conn)
    finally:
        conn.close()
    return {'ok': True}


def _stat1_rows(conn: sqlite3.Connection) -> list:
    """sqlite_stat1's rows, or [] before anything was ever analyzed."""
    if not conn.execute("SELECT 1 FROM sqlite_master "
                        "WHERE name = 'sqlite_stat1'").fetchone():
        return []
    return conn.execute('SELECT tbl, idx, stat FROM sqlite_stat1 '
                        'ORDER BY tbl, idx').fetchall()


def _signal_stats_reload(conn: sqlite3.Connection) -> None:
    """Make every open connection re-read the planner statistics.

    A connection loads sqlite_stat1 into memory when it parses the schema,
    and ANALYZE on another connection does not tell it to reload — so the
    daemon's long-lived connections would keep planning on the statistics
    they opened with until the next restart. A schema change does: every
    connection re-parses schema and statistics on its next statement. One
    transaction creates and drops a throwaway table, so no connection ever
    sees it; the cost is one schema re-parse per connection.
    """
    conn.execute('BEGIN IMMEDIATE')
    try:
        conn.execute('CREATE TABLE _planner_stats_reload (x)')
        conn.execute('DROP TABLE _planner_stats_reload')
        conn.execute('COMMIT')
    except Exception:
        conn.execute('ROLLBACK')
        raise


def stats(db_path: str) -> Dict[str, Any]:
    """Diagnostic snapshot — DB size, WAL size, page count, freelist.
    Cheap to collect; useful for tracking DB growth between ticks.
    """
    db_size = _file_size_or_zero(db_path)
    wal_size = _file_size_or_zero(db_path + '-wal')
    shm_size = _file_size_or_zero(db_path + '-shm')

    conn = connect_maintenance(db_path)
    try:
        page_count = conn.execute('PRAGMA page_count').fetchone()[0]
        page_size = conn.execute('PRAGMA page_size').fetchone()[0]
        freelist = conn.execute('PRAGMA freelist_count').fetchone()[0]
    finally:
        conn.close()
    return {
        'db_size_bytes': db_size,
        'wal_size_bytes': wal_size,
        'shm_size_bytes': shm_size,
        'page_count': page_count,
        'page_size': page_size,
        'freelist_pages': freelist,
        'pages_in_use_pct': round(
            (page_count - freelist) / max(1, page_count) * 100, 1),
    }


def quick_check(db_path: str) -> Dict[str, Any]:
    """`PRAGMA quick_check` — structural integrity scan (skips index-content
    verification, so it's fast enough for a live DB on the maintenance
    connection). RAISES on any finding: corruption must go through the
    scheduler's error path into the brain errors table, not sit in a log
    line nobody reads. Healthy result is {'ok': True}.
    """
    conn = connect_maintenance(db_path)
    try:
        rows = conn.execute('PRAGMA quick_check(20)').fetchall()
    finally:
        conn.close()
    issues = [r[0] for r in rows if r and r[0] != 'ok']
    if issues:
        raise RuntimeError('quick_check found %d issue(s) in %s: %s' % (
            len(issues), db_path, '; '.join(issues[:5])))
    return {'ok': True}


def table_sizes(db_path: str, top_n: int = 12) -> Dict[str, Any]:
    """Per-table size breakdown via the dbstat virtual table, in MB,
    largest first. Returns {} when dbstat isn't compiled in — callers
    treat the breakdown as best-effort telemetry, never load-bearing.
    """
    conn = connect_maintenance(db_path)
    try:
        try:
            rows = conn.execute(
                'SELECT name, SUM(pgsize) FROM dbstat '
                'GROUP BY name ORDER BY 2 DESC LIMIT ?', (top_n,)).fetchall()
        except sqlite3.OperationalError:
            return {}
    finally:
        conn.close()
    return {name: round(size / 1048576.0, 1) for name, size in rows}


def snapshot_to(db_path: str, dest_db_path: str,
                pages: int = 4000, sleep_s: float = 0.0) -> int:
    """Consistent raw-.db snapshot of a LIVE SQLite DB, via the online
    backup API — NOT a file copy. `cp` of a live WAL-mode DB can capture
    a torn state (DB file + a partial WAL the copy missed); the backup
    API produces a transactionally-consistent, self-contained file with
    no such hazard, including committed rows still sitting in the -wal.

    Copies in page batches (`pages`), optionally pausing `sleep_s`
    between batches to throttle I/O against a busy daemon (background
    callers pass a pause; boot-path and clone callers take the default
    and finish fast). If a write lands mid-copy SQLite re-copies the
    changed pages — correctness preserved.

    The source is opened READ-ONLY here — never the caller's connection,
    and never a writer. A source connection holding an open write
    transaction deadlocks the backup against itself; a separate
    read-only connection is immune, captures last-committed state even
    while another connection's transaction is open, and cannot touch the
    source (no pragmas, no close-time checkpoint) — clones of a live
    production DB must be a pure read.

    Returns the snapshot's size in bytes.
    """
    if not os.path.exists(db_path):
        # A read-write connect() would CREATE an empty DB and "back it up"
        # successfully — a silent no-op snapshot is worse than a loud failure.
        raise sqlite3.OperationalError('snapshot source missing: %s' % db_path)
    from urllib.parse import quote
    src = sqlite3.connect('file:%s?mode=ro' % quote(db_path), uri=True,
                          timeout=_MAINTENANCE_BUSY_TIMEOUT_MS / 1000.0)
    try:
        dst = sqlite3.connect(dest_db_path)
        try:
            src.backup(dst, pages=pages, sleep=sleep_s)
        finally:
            dst.close()
    finally:
        src.close()
    return _file_size_or_zero(dest_db_path)


def backup_snapshot(db_path: str, dest_gz_path: str,
                    pages: int = 4000, sleep_s: float = 0.05,
                    compresslevel: int = 6) -> Dict[str, Any]:
    """Consistent, gzip-compressed snapshot of a LIVE SQLite DB.

    The copy itself is `snapshot_to` (online backup API — see there for
    the consistency and writer-friendliness story). This wrapper owns
    the durable-artifact shape: the snapshot lands in a temp .db, is
    gzipped to a `.part` file, then atomically renamed to
    `dest_gz_path` — so the canonical .gz appears only once fully
    written. A mid-gzip crash (disk full, process kill) never leaves a
    truncated .gz that list_backups would treat as a valid snapshot.
    Both intermediates are removed on every exit path. Returns
    raw/compressed sizes + ratio for observability.

    Intermediates carry a per-call random suffix: two unserialized
    processes (or threads) snapshotting the same destination must not
    interleave writes into one shared temp file — each builds its own and
    the last atomic rename wins whole.

    `compresslevel` trades CPU for size: 6 suits background snapshots;
    time-budgeted callers pass 1.
    """
    nonce = os.urandom(4).hex()
    tmp_db = '%s.tmp.%s.db' % (dest_gz_path, nonce)
    part_gz = '%s.part.%s' % (dest_gz_path, nonce)
    try:
        raw_bytes = snapshot_to(db_path, tmp_db, pages=pages, sleep_s=sleep_s)
        with open(tmp_db, 'rb') as f_in, \
                gzip.open(part_gz, 'wb', compresslevel=compresslevel) as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        os.replace(part_gz, dest_gz_path)   # atomic publish
    finally:
        # Always clear intermediates. On success tmp_db is stale and
        # part_gz was renamed away (both removes no-op); on any failure
        # this prevents orphaned ~hundreds-of-MB temp files accumulating.
        for tmp in (tmp_db, part_gz):
            try:
                os.remove(tmp)
            except OSError:
                pass

    gz_bytes = _file_size_or_zero(dest_gz_path)
    return {
        'raw_bytes': raw_bytes,
        'gz_bytes': gz_bytes,
        'ratio': round(gz_bytes / max(1, raw_bytes), 3),
    }


# ─── Helpers ──────────────────────────────────────────────────────────

def _file_size_or_zero(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
