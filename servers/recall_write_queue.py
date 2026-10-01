"""
brain — Recall Write Queue

Deferred-write queue for recall-side bookkeeping that does NOT need realtime
persistence. The recall hot path enqueues; the embed_queue worker drains
every EMBED_DRAIN_INTERVAL seconds via `brain.conn_bg_writer`.

One signal rides on this queue:

  enqueue_access(node_id, session_id, ts)
    Recognition signal. Per-(node, session, drain-window) dedup'd by Dict
    semantics — the value held is the most recent timestamp seen. Drain
    produces ONE atomic UPDATE per unique (node, session) pair:
        access_count = access_count + 1
        last_accessed = ts (max within window)
    Multiple accesses to the same node from the same session within one
    drain window collapse to a single +1. Cross-session accesses to the
    same node still produce one increment per session.

Design constraints:
  - No silent errors. Every except logs via brain._log_error.
  - No read-modify-write where SQL can do +1 atomically.
  - Hot path is pure enqueue — no DB I/O.

Loss semantics (accepted by operator):
  - Daemon crash mid-drain: pending queue (≤ EMBED_DRAIN_INTERVAL seconds
    of recalls) lost. access_count is approximate.
  - Transaction failure: ROLLBACK + drop the batch + log loudly. No retry
    queue (avoids poison-pill loops).

Stats (get_stats) feed dashboard observability.
"""

import sys
import threading
import time
from typing import Dict, Tuple


# ─── Module-level state ─────────────────────────────────────────────
# All access guarded by `_lock`. The queue is mutated only via
# enqueue_access and _snapshot_and_clear().

# Access queue: maps (node_id, session_id) → most-recent timestamp string.
# Dict (not Set) so we preserve the latest `last_accessed` value without
# extra state. Within a drain window, multiple accesses to the same
# (node, session) collapse to one +1; the timestamp stored is the most
# recent the queue has seen.
_access_queue: Dict[Tuple[str, str], str] = {}

_lock = threading.Lock()

# NOTE: this queue has NO shutdown signal of its own. There is ONE background
# drain worker (it lives in embed_queue and drains this queue too); its single
# shutdown signal — embed_queue.request_shutdown() / _shutdown_event — stops
# draining of this queue too. recall_write_queue is a passive data structure.

_stats = {
    'access_enqueued_total': 0,
    'drains_total': 0,
    'drains_skipped_empty': 0,
    'last_drain_at': None,            # epoch seconds
    'last_drain_took_ms': 0,
    'last_drain_size': 0,             # access items in batch
    'last_begin_wait_ms': 0,          # time spent in BEGIN IMMEDIATE (WAL slot wait)
    'access_drained_total': 0,
    'errors_total': 0,                # exceptions during drain (caught + logged)
    'rollbacks_total': 0,             # transactions that hit ROLLBACK
    'overlong_drains_total': 0,       # drains that took > _OVERLONG_THRESHOLD_MS
    'enqueue_errors_total': 0,        # exceptions during enqueue (lock contention)
}

# Threshold for the "drain took too long" warning emission (ms). Mirrors
# embed_queue's 10s overlong gate. Reviewed alongside drain interval if
# the queue ever changes producer/consumer ratio.
_OVERLONG_THRESHOLD_MS = 10_000

# Threshold for the "producer outpacing drain" warning emission. If we
# ever cross this, something pathological is happening upstream — emit a
# loud warning so we notice before the queue eats RAM.
_QUEUE_DEPTH_WARN = 5_000


# ─── Public API: enqueue ─────────────────────────────────────────────

def enqueue_access(node_id: str, session_id: str, ts: str) -> None:
    """Mark a node as accessed from a specific session at time `ts`.

    Cheap — dict update under lock. No DB I/O. Dedups multiple accesses
    to the same (node, session) within the current drain window down to
    one +1 increment, with last_accessed = the most recent ts seen.

    Caller MUST NOT touch the DB on the hot path. session_id of '' is
    coerced to 'unknown' (still tracked; access_count doesn't depend on
    attribution).
    """
    if not node_id:
        return
    sid = session_id or 'unknown'
    try:
        with _lock:
            key = (node_id, sid)
            existing = _access_queue.get(key)
            # ISO-8601 timestamp strings compare correctly as strings.
            if existing is None or ts > existing:
                _access_queue[key] = ts
            _stats['access_enqueued_total'] += 1
    except Exception as e:
        # Loud-by-default: we don't have a `brain` handle here for
        # _log_error, so stderr is the fallback. The caller (e.g.
        # brain_recall) can also wrap this and route through
        # brain._log_error for session attribution.
        with _lock:
            _stats['enqueue_errors_total'] += 1
        print('[recall_write_queue] enqueue_access failed: %s' % e,
              file=sys.stderr)


# ─── Public API: introspection ───────────────────────────────────────

def queue_depth() -> int:
    """Pending access items. Cheap; for stall heartbeat."""
    with _lock:
        return len(_access_queue)


def get_stats() -> dict:
    """Snapshot of queue stats. Safe to call any time."""
    with _lock:
        return {
            **_stats,
            'access_queue_depth': len(_access_queue),
            'overlong_threshold_ms': _OVERLONG_THRESHOLD_MS,
            'queue_depth_warn': _QUEUE_DEPTH_WARN,
        }


# ─── Shutdown ────────────────────────────────────────────────────────
# No per-queue shutdown here: the single bg-writer drain worker (in embed_queue)
# owns the one shutdown signal; stopping it stops draining of this queue too.
# See embed_queue.request_shutdown() / join_worker().


# ─── Internal: snapshot/swap ─────────────────────────────────────────

def _snapshot_and_clear() -> Dict[Tuple[str, str], str]:
    """Atomically swap queue contents to a local snapshot and clear the
    module-level structure. Subsequent enqueues land in the fresh
    structure while the drain proceeds against the snapshot.
    """
    global _access_queue
    with _lock:
        access_snap = _access_queue
        _access_queue = {}
    return access_snap


# ─── Public API: drain ───────────────────────────────────────────────

def drain_once(brain) -> None:
    """Drain the access queue against `brain.conn_bg_writer` in a single
    transaction. Called by the embed_queue worker once per cycle
    (wired in Phase 3).

    An empty queue is a no-op (counted in drains_skipped_empty).
    On any exception during the transaction: ROLLBACK + log via
    `bg_writer_batch_rollback` + drop the snapshot. The dropped batch
    is NOT re-queued — preserves the loss-semantic contract and avoids
    poison-pill loops (one corrupt row could otherwise stall the
    queue forever).

    This function's contract is: never raise out — log and return.
    Top-level errors are caught here so the worker loop's outer guard
    is a backstop, not the primary safety net.
    """
    t0 = time.time()

    access_snap = _snapshot_and_clear()
    batch_size = len(access_snap)

    if batch_size == 0:
        # Empty tick — worker is alive, just had nothing to drain. Update
        # last_drain_at so the stall watchdog doesn't read a stale timestamp
        # from the last batch with actual work. Without this, a long idle
        # period followed by a burst of enqueues looks like a "stall" the
        # moment the burst arrives.
        with _lock:
            _stats['drains_skipped_empty'] += 1
            _stats['last_drain_at'] = t0
        return

    conn = brain.conn_bg_writer
    rolled_back = False
    begin_wait_ms = 0
    try:
        # BEGIN IMMEDIATE so we grab the WAL writer slot upfront; without
        # this SQLite auto-begins on first write and could deadlock on
        # busy_timeout mid-batch if a foreground write started first.
        # We measure this separately because stalls usually point here —
        # the WAL slot is contended, BEGIN IMMEDIATE waits, everything
        # downstream pays. The wait time is exported in stats so the
        # stall watchdog can attribute time accurately.
        _bi_t0 = time.time()
        conn.execute('BEGIN IMMEDIATE')
        begin_wait_ms = int((time.time() - _bi_t0) * 1000)

        # Atomic +1 per (node, session) pair via executemany. The
        # `archived = 0` filter makes the UPDATE a silent no-op on
        # archived nodes (race between enqueue and drain — intended,
        # not an error).
        # Deliberately does NOT touch updated_at: reads must never look
        # like writes. Access semantics live in last_accessed; updated_at
        # means "a write mutated this row" (contract.py field spec) —
        # the old access-bump broke that for every consumer (community
        # idle gate always-firing, consolidation fingerprint churn, the
        # deleted recall_recent tool). Pinned by test_bg_writer.
        conn.executemany(
            'UPDATE nodes SET '
            '    access_count = access_count + 1, '
            '    last_accessed = ? '
            'WHERE id = ? AND archived = 0',
            [(ts, nid) for (nid, _sid), ts in access_snap.items()])
        with _lock:
            _stats['access_drained_total'] += len(access_snap)

        conn.commit()

    except Exception as e:
        # Primary path failed. Attempt rollback; either way, log loudly
        # and drop the batch.
        try:
            conn.rollback()
            rolled_back = True
        except Exception as re:
            # Double-fault. Best-effort log via _log_error; if that also
            # fails, stderr.
            try:
                brain._log_error('bg_writer_drain_rollback_failed', re,
                                 'rollback after primary drain exception failed')
            except Exception as le:
                print('[recall_write_queue] rollback failed AND _log_error '
                      'failed: rollback=%s log=%s' % (re, le),
                      file=sys.stderr)
        try:
            brain._log_error(
                'bg_writer_batch_rollback', e,
                'drain_once dropped batch: access=%d' % len(access_snap))
        except Exception as le:
            print('[recall_write_queue] _log_error itself failed: drain=%s '
                  'log=%s' % (e, le), file=sys.stderr)
        with _lock:
            _stats['errors_total'] += 1
            if rolled_back:
                _stats['rollbacks_total'] += 1
        # NOTE: do NOT re-enqueue the snapshot. The data is intentionally
        # lost — preserves the loss contract and avoids poison-pill loops.

    took_ms = int((time.time() - t0) * 1000)
    with _lock:
        _stats['drains_total'] += 1
        _stats['last_drain_at'] = t0
        _stats['last_drain_took_ms'] = took_ms
        _stats['last_drain_size'] = batch_size
        _stats['last_begin_wait_ms'] = begin_wait_ms
        if took_ms > _OVERLONG_THRESHOLD_MS:
            _stats['overlong_drains_total'] += 1

    # Overlong warning — log AFTER stats are updated so observers see
    # the increment regardless of whether _log_error succeeds.
    if took_ms > _OVERLONG_THRESHOLD_MS:
        try:
            brain._log_error(
                'bg_writer_drain_overlong',
                RuntimeError('drain took %dms (>%dms threshold) — BEGIN IMMEDIATE waited %dms' %
                             (took_ms, _OVERLONG_THRESHOLD_MS, begin_wait_ms)),
                'access=%d begin_wait_ms=%d' %
                (len(access_snap), begin_wait_ms))
        except Exception as le:
            print('[recall_write_queue] overlong log failed: %s' % le,
                  file=sys.stderr)

    depth_now = queue_depth()
    if depth_now > _QUEUE_DEPTH_WARN:
        try:
            brain._log_error(
                'bg_writer_queue_overflow',
                RuntimeError('post-drain queue depth %d > %d threshold' %
                             (depth_now, _QUEUE_DEPTH_WARN)),
                'producer outpacing drain — investigate')
        except Exception as le:
            print('[recall_write_queue] overflow log failed: %s' % le,
                  file=sys.stderr)


# ─── Test-only helpers ───────────────────────────────────────────────

def _clear_for_test() -> None:
    """Reset module state. Production code MUST NOT call this; it bypasses
    the normal enqueue → drain cycle and risks contention with the worker.
    """
    global _access_queue
    with _lock:
        _access_queue = {}
        for k in _stats:
            _stats[k] = 0 if isinstance(_stats[k], int) else None
