"""
brain — Embed Queue

Nodes that get written (remember/revise/connect) enqueue themselves here.
A single worker thread wakes every EMBED_DRAIN_INTERVAL seconds, drains the
set, and runs one pass of backfill_vectors() scoped to the drained ids.

Design constraints (from operator):
  - Single worker (no pool). One embed_batch call already uses ORT internally.
  - Skip-tick if the previous drain is still running. No overlapping drains.
  - Queue is in-memory. On daemon restart, S2 Heal sweeps up any gaps.
  - Latency bound: ~EMBED_DRAIN_INTERVAL seconds from write to indexed.

Exposed stats (via get_stats) feed embedder diagnostics so "my write isn't
indexed" is legible instead of silent.
"""

import sys
import threading
import time
from typing import Dict, Optional, Set

from .clock import iso_cutoff
from .trace_contract import CONVERSATIONAL_REF_TYPES

# Drain every N seconds if queue is non-empty. Empty queue = no-op, no work.
EMBED_DRAIN_INTERVAL = 5.0
# Max entities processed under a single write_lock acquisition. Lock is
# released between batches so concurrent writers can interleave. Set to
# handle the cold-start case (~20K entities at first boot) without
# holding the lock for the full drain.
DRAIN_BATCH_SIZE = 500
# Stall threshold — if pending work exists AND no drain has happened
# within STALL_THRESHOLD seconds, log loudly. Self-reported by the
# worker; covers "worker is alive but blocked / slow" not "worker is
# dead". Thread death is detected by daemon's thread-count watchdog.
STALL_THRESHOLD_S = EMBED_DRAIN_INTERVAL * 3

# Unscoped vector-coverage sweep. The queue only repairs what reached it;
# a node written by a path that skips the enqueue hooks, or a crash between
# insert and enqueue, leaves vectors missing forever. This sweep is the
# only automatic repair for that class, so it lives beside the queue that
# owns embedding rather than in a separate maintenance pass.
# Hourly, not per-minute: this is a NET, not the primary path. Writes reach
# their vectors within one drain interval, and edge writes invalidate through
# the brain-level door (Brain.invalidate_source_fields), so a gap here means a
# writer bypassed both — rare by construction. Measured: one pass costs 0.276s
# of find_missing scans across six vector types and, with the enqueue paths
# healthy, returns nothing. At 60s that was ~0.46% of a core spent learning
# "no" 1,440 times a day. Nothing rides on noticing a gap within a minute
# rather than an hour; a node missing `_primary` is invisible to recall either
# way, and the write paths are what keep that from happening.
COVERAGE_SWEEP_INTERVAL = 3600.0
COVERAGE_SWEEP_BATCH = 30
# How soon after worker start the first sweep may run. NOT the full interval:
# boot is when coverage is least trustworthy — the queue is in-memory, so a
# restart drops every id it held, and boot-time re-enqueues (a migration's
# rebuild) ride that same volatile queue. Leaving the cold-start anchor at a
# full interval would have pushed first recovery from ~1min to ~1h the moment
# the interval changed. Long enough that cold-start drain isn't competing.
COVERAGE_SWEEP_BOOT_GRACE_S = 120.0
# Ceiling on deferring to a non-empty queue. Without it, "let scoped work go
# first" degrades into "never sweep" on a brain that always has scoped work.
# The invariant is the ABSOLUTE bound — how long coverage may go unchecked
# under sustained load — not its ratio to the interval; ×10 would have turned
# a 10-minute guarantee into a 10-hour one. Kept above the interval (or the
# interval gate decides every tick and this becomes dead code) and no higher.
# Derived at import: overriding COVERAGE_SWEEP_INTERVAL at runtime does not
# move it.
COVERAGE_SWEEP_MAX_STALENESS = COVERAGE_SWEEP_INTERVAL * 2

# Pull-reconciliation policy for trace embeddings (v27 episodic refs).
# Each tick: find up to N S0 trace events with no embedding yet, render
# per §5.3, embed in one batch, store. Newest-first so recent
# conversation is anchorable immediately; backfill of older traces works
# backwards on its own across subsequent deep passes, which stay hot while
# they keep writing (see TRACE_DEEP_SCAN_INTERVAL). No queue state;
# restart-safe — the LEFT JOIN finds whatever's still missing.
TRACE_DRAIN_LIMIT = 5
EAGER_TRACE_SCALES = ('s0',)
# Conversation only — deliberately NOT the encoder's SAID_AND_DID set, which
# also carries tool_result. Both trace-vector consumers read conversation:
# recall_laf's matrix is pinned to OPERATOR_DIALOGUE_REF_TYPES and
# recall_episodes' semantic scan defaults to it, so no lane ever ranks a
# tool_result vector. Tool rows still enter the encoder's timeline and stay
# queryable by row (recall_episodes ref_type='tool_result' with contains=);
# they just carry no vector. Dial-derived, so flipping a correspondent on in
# S0_CONVERSATIONAL_INCOMING widens what is embedded together with what the
# conversation means.
EAGER_TRACE_REF_TYPES = CONVERSATIONAL_REF_TYPES
# Embed window default — only consider traces newer than this many days.
# Runtime value comes from config key 'embed_queue.trace_window_days'
# (read per tick — flips take effect without a restart); <= 0 means NO
# window (full history). The window's original architectural reason —
# pre-identity-stamping traces would render OPERATOR/ANCHOR sentinels,
# which Decision 19 rules out — is extinct: the historical identity
# backfill stamped every trace, so on an established brain no unembedded
# dialogue row lacks human_identity. The render's sentinel fallback remains
# for fresh installs only.
TRACE_EMBED_WINDOW_DAYS = 30
# The window is scanned at two cadences. A FAST tick (every drain interval)
# asks only about traces newer than the watermark — the created_at that a
# previous empty scan proved everything above; a DEEP pass drops the watermark
# and re-reads the whole window, so a row the watermark stepped over waits
# minutes rather than forever. Without the watermark every idle tick walked the
# whole window: the scan is cheap exactly when it finds work — it stops at
# TRACE_DRAIN_LIMIT — and most expensive when there is none, which is what an
# idle daemon does all day. Measured at 44.5K rows in window: ~0.3s of CPU
# every 5s for an answer that was "nothing" on 6010 of 6020 ticks.
# A deep pass that WRITES stays hot (next tick is deep again), so a real
# backlog below the watermark — a widened window, an embedder outage — still
# drains at TRACE_DRAIN_LIMIT per tick instead of per interval. Note what that
# makes of the window knob: widening trace_window_days to re-include older
# traces is now deep-pass-paced, since fast ticks keep using the newer
# watermark — recovery starts within one interval, not on the next tick.
TRACE_DEEP_SCAN_INTERVAL = 600.0
# How far behind the scan's start the watermark lands. A trace committed while
# the scan runs carries a created_at from before it; the lag keeps such a row
# in the next fast tick's range instead of leaving it to the deep pass.
TRACE_WATERMARK_LAG_MINUTES = 1.0

# ─── State ───
_queue: Set[str] = set()           # node ids needing vector recomputation
_edge_queue: Set[str] = set()      # edge ids needing embedding recomputation
_lock = threading.Lock()
_drain_busy = threading.Lock()  # serializes a WHOLE tick's stages (worker skips
                                # its tick non-blocking; drain_now waits up to
                                # DRAIN_NOW_TIMEOUT_S) — two threads must never
                                # interleave transactions on conn_bg_writer
_worker_started = False
# Coverage-sweep throttle. Stamped at worker start, NOT left at 0.0: the
# staleness floor below compares elapsed time, and 0.0 makes "time since last
# sweep" ~epoch seconds, which clears every ceiling and would fire a full
# unscoped sweep during cold-start against a 20K-entity queue.
_last_sweep_at = 0.0
# Trace-scan watermark: the created_at above which the window is known fully
# embedded. None = nothing proven yet, so the scan falls back to the window.
_trace_watermark: Optional[str] = None
# Deep-pass clock — time.monotonic(), because this is a tick: a backward wall
# step would park it until the clock caught up (servers/clock.py, category C).
# The 0.0 seed is not load-bearing the way _last_sweep_at's is: a fresh process
# has no watermark, so its first scan reads the whole window whether or not the
# deep slot has opened yet.
_last_trace_deep_at = 0.0
_shutdown_event = threading.Event()    # the SINGLE shutdown signal: set() stops the worker and wakes it out of its interval wait at once
_worker_thread: Optional[threading.Thread] = None
_stats = {
    'nodes_enqueued_total': 0,
    'edges_enqueued_total': 0,
    'drains_total': 0,
    'drains_skipped_busy': 0,
    'drains_skipped_empty': 0,
    'nodes_processed_total': 0,
    'edges_processed_total': 0,
    'vectors_written_total': 0,
    'traces_processed_total': 0,
    'traces_deep_scans_total': 0,
    'traces_errors_total': 0,
    'traces_skipped_embedder_not_ready': 0,
    'last_trace_drain_at': None,
    'last_drain_at': None,   # epoch seconds
    'last_drain_took_ms': 0,
    'last_drain_size': 0,
    'worker_loop_errors_total': 0,
    'stalls_logged_total': 0,
}


def enqueue(node_id: str) -> None:
    """Mark a node as needing vector (re)computation. Cheap — set.add under lock."""
    if not node_id:
        return
    with _lock:
        _queue.add(node_id)
        _stats['nodes_enqueued_total'] += 1


def enqueue_edge(edge_id: str) -> None:
    """Mark an edge as needing embedding re-computation. Cheap (set.add under
    lock). Called from edge-write paths (GraphDAL.add_relation /
    rename_relation), which NULL the stale embedding first; the worker drains
    via Brain.backfill_edge_embeddings (the central re-embed for new/stale rows).
    """
    if not edge_id:
        return
    with _lock:
        _edge_queue.add(edge_id)
        _stats['edges_enqueued_total'] += 1


def get_stats() -> dict:
    with _lock:
        return {**_stats,
                'queue_depth': len(_queue),
                'edge_queue_depth': len(_edge_queue),
                'trace_watermark': _trace_watermark,
                'interval_s': EMBED_DRAIN_INTERVAL}


def start(brain) -> None:
    """Start the single drain worker. Idempotent — safe to call multiple times."""
    global _worker_started, _worker_thread, _last_sweep_at
    with _lock:
        if _worker_started:
            return
        _worker_started = True
        # Anchor the sweep clock to worker start so both the interval and the
        # staleness floor measure real elapsed time — offset so the first sweep
        # lands one BOOT GRACE from here, not one full interval. Cold-start
        # still gets a window to drain the queue before an unscoped sweep
        # competes; what it no longer gets is an hour of unchecked coverage at
        # the one moment the in-memory queue was just lost.
        _last_sweep_at = (time.time() - COVERAGE_SWEEP_INTERVAL
                          + COVERAGE_SWEEP_BOOT_GRACE_S)
    if _shutdown_event.is_set():
        # A prior lifecycle latched the shutdown signal. Clear it, or the fresh
        # worker would exit on its first wait() and silently never drain.
        sys.stderr.write("[embed_queue] start: clearing a latched shutdown signal "
                         "from a prior lifecycle so the new worker drains.\n")
        _shutdown_event.clear()
    t = threading.Thread(target=_worker_loop, args=(brain,),
                         name='embed-queue-drain', daemon=True)
    _worker_thread = t
    t.start()


def request_shutdown() -> None:
    """Stop the single bg-writer drain worker. It drains BOTH embed_queue and
    recall_write_queue, so this is the ONE shutdown signal for both. Sets an
    Event (atomic, lock-free, one source of truth) that wakes the worker out of
    its interval wait immediately — so daemon shutdown isn't blocked for up to
    EMBED_DRAIN_INTERVAL and brain.close() can't race an in-flight drain."""
    _shutdown_event.set()


def _is_shutdown_requested() -> bool:
    return _shutdown_event.is_set()


# How long an explicit drain waits for the tick lock before reporting busy.
# Bounded: drain_now runs on a dispatch-pool thread, and an unbounded wait
# behind a cold-start multi-minute worker drain would park dispatch threads
# until the pool starves and the health monitor kickstarts a busy-but-healthy
# daemon. A busy answer is honest; the caller retries.
DRAIN_NOW_TIMEOUT_S = 60.0


def drain_now(brain) -> dict:
    """One synchronous worker tick, on the caller's thread — the same four
    stages as the background worker (_worker_tick_stages), under the same
    _drain_busy lock, so the two can never interleave transactions on
    conn_bg_writer.

    The door for callers that must not race the worker's interval: an eval
    entity's harness turn ("everything written is embedded before the next
    recall"), or a deploy step that needs the queues settled.

    Honesty contract: the stages themselves log-and-continue (worker
    semantics), so success is judged by OBSERVABLE state, not absence of
    exceptions — `ok` is True only when the embedder is ready AND the
    node/edge queues drained to empty. A not-ready embedder or a re-enqueued
    failure batch yields ok=False with the reason, never a green signal over
    semantically invisible nodes."""
    from servers import recall_write_queue
    from servers import embedder
    if not _drain_busy.acquire(timeout=DRAIN_NOW_TIMEOUT_S):
        return {"ok": False, "busy": True,
                "error": "drain lock held >%ds (background drain in flight) — "
                         "retry" % int(DRAIN_NOW_TIMEOUT_S)}
    try:
        _worker_tick_stages(brain, recall_write_queue)
    finally:
        _drain_busy.release()
    with _lock:
        pending = len(_queue) + len(_edge_queue)
    ready = embedder.is_ready()
    result = {"ok": ready and pending == 0, "pending_after": pending,
              "embedder_ready": ready}
    if not ready:
        result["error"] = "embedder not ready — nothing was embedded"
    elif pending:
        result["error"] = ("%d ids re-enqueued by a failed batch — see the "
                           "errors table" % pending)
    return result


def join_worker(timeout: float = 3.0) -> None:
    """Block until the drain worker has exited (or `timeout` elapses). Called from
    daemon shutdown AFTER request_shutdown() so the worker settles OFF
    brain.conn_bg_writer — its in-flight drain finishes and the thread exits —
    before brain.close() runs. Without it, close() races a mid-drain →
    'Cannot operate on a closed database' + a dropped batch (2026-06-06)."""
    t = _worker_thread
    if t is not None and t.is_alive():
        t.join(timeout)


def _worker_loop(brain) -> None:
    """Worker thread body. One thread drives two queues every
    EMBED_DRAIN_INTERVAL seconds:

      1. `embed_queue` — node vectors + edge embeddings.
         Heavier work; uses `_drain_busy` skip-tick semantics to avoid
         overlapping drains on cold-start (~20K entities at first boot).
      2. `recall_write_queue` — access marks. Lightweight (atomic SQL
         +1s on `brain.conn_bg_writer`); always called per cycle since
         its own drain_once self-checks for an empty queue.

    Top-level try/except is the load-bearing safety net for the
    "no silent errors" mandate — the worker thread MUST never die.
    Any unexpected exception is logged (origin `embed_worker_loop`)
    and the loop continues after a brief backoff.

    Shutdown: request_shutdown() sets the Event, waking this loop out of its
    interval wait so it exits at the next check. ONE worker, ONE shutdown signal
    — it drains both queues, so both stop together.
    """
    # Import here so module-load order doesn't matter.
    from servers import recall_write_queue

    while True:
        try:
            # Interruptible wait — request_shutdown() sets _shutdown_event to wake
            # us at once, so shutdown isn't blocked for a full interval and
            # brain.close() won't race an in-flight drain.
            _shutdown_event.wait(EMBED_DRAIN_INTERVAL)

            if _is_shutdown_requested():     # the single bg-writer shutdown signal
                break

            # Liveness self-check — if pending work exists and last drain
            # was suspiciously long ago, log loudly. Helps surface a
            # blocked / slow worker without an external watchdog.
            _check_stall(brain, recall_write_queue)

            # The whole four-stage tick runs under _drain_busy — the same lock
            # drain_now() (a dispatch thread) takes — because a stage
            # (recall_write_queue.drain_once) issues BEGIN IMMEDIATE/commit on
            # the single conn_bg_writer, and two threads interleaving on that
            # connection nest transactions or cross-commit each other's
            # half-written batches. Skip-tick if a previous tick (or an
            # explicit drain_now) still holds it.
            if not _drain_busy.acquire(blocking=False):
                with _lock:
                    _stats['drains_skipped_busy'] += 1
                continue
            try:
                _worker_tick_stages(brain, recall_write_queue)
            finally:
                _drain_busy.release()

        except Exception as e:
            # Worker thread MUST never die silently. Catch everything,
            # log loudly, sleep a beat, continue.
            with _lock:
                _stats['worker_loop_errors_total'] += 1
            try:
                brain._log_error(
                    'embed_worker_loop', e,
                    'worker loop caught unexpected exception — '
                    'continuing after backoff')
            except Exception as le:
                print('[embed_queue] worker loop fatal (log failed): '
                      'original=%s log=%s' % (e, le), file=sys.stderr)
            # Extra sleep on error so we don't spin if the cause is
            # immediate-and-repeating. time.sleep does not raise in
            # normal operation; if it does (signal handler reraises),
            # let it propagate — the OUTER while True will resume the
            # loop body at the next iteration.
            time.sleep(EMBED_DRAIN_INTERVAL)


def _worker_tick_stages(brain, recall_write_queue) -> None:
    """The four drain stages of one tick, in order. Caller holds _drain_busy.
    Each stage keeps its own try/except: one failing step must not cost the
    others (the worker's loud-logging semantics; drain_now() reports outcomes
    to its caller instead — see its docstring)."""
    with _lock:
        embed_pending = bool(_queue or _edge_queue)
    if embed_pending:
        try:
            _drain_once(brain)
        except Exception as e:
            try:
                brain._log_error(
                    'embed_queue_drain', e,
                    'top-level embed drain caught')
            except Exception as le:
                print('[embed_queue] drain error: %s '
                      '(log failed: %s)' % (e, le),
                      file=sys.stderr)
    else:
        with _lock:
            _stats['drains_skipped_empty'] += 1

    # Pull-reconciliation: trace embeddings. Independent of
    # node/edge queues — reads via the shared logs conn, writes
    # through TraceDAL's write boundary (dedicated write conn under
    # write_lock). Caps per-tick work via TRACE_DRAIN_LIMIT; runs on
    # every tick so newly-written S0 trace events get anchored
    # within ~5s.
    try:
        _drain_trace_embeddings_once(brain)
    except Exception as e:
        try:
            brain._log_error(
                'embed_queue_trace_top', e,
                'top-level trace embed drain caught')
        except Exception as le:
            print('[embed_queue] trace embed error: %s '
                  '(log failed: %s)' % (e, le), file=sys.stderr)

    # Unscoped coverage sweep — every tick, throttled internally.
    # Runs here rather than inside _drain_once so sustained write load
    # can't starve it (the drain's empty-tick branch is never reached
    # while work keeps arriving), and LAST in the tick because a sweep
    # can embed a full batch — ahead of the trace drain it would push
    # S0 anchoring past its ~5s contract. Own try/except, like every
    # other step here: one failing step must not cost the others.
    try:
        _coverage_sweep(brain)
    except Exception as e:
        try:
            brain._log_error(
                'embed_queue_coverage_top', e,
                'top-level coverage sweep caught')
        except Exception as le:
            print('[embed_queue] coverage sweep error: %s '
                  '(log failed: %s)' % (e, le), file=sys.stderr)

    # Drain recall_write_queue (access marks). Fast,
    # separate connection, separate transaction. drain_once is
    # contract-bound never to raise out — this try/except is
    # belt-and-suspenders.
    try:
        recall_write_queue.drain_once(brain)
    except Exception as e:
        try:
            brain._log_error(
                'recall_write_queue_drain', e,
                'top-level recall_write drain caught — '
                'drain_once is supposed to not raise')
        except Exception as le:
            print('[embed_queue] recall_write drain error: %s '
                  '(log failed: %s)' % (e, le), file=sys.stderr)


def _check_stall(brain, recall_write_queue) -> None:
    """Log a stall warning if there's pending work AND last drain was
    longer than STALL_THRESHOLD_S ago. Self-reported by the worker —
    only fires when the worker thread is alive enough to execute this.
    For "worker thread dead entirely" detection, rely on the daemon's
    thread-count watchdog (separate signal).

    Log payload includes the write_lock holder (if TrackedRLock is in
    use) and the last drain's BEGIN IMMEDIATE wait time. These two
    signals usually answer "who's holding the WAL writer slot and for
    how long" — the root cause of most stalls.
    """
    try:
        embed_snap = get_stats()
        rwq_snap = recall_write_queue.get_stats()

        embed_depth = (embed_snap.get('queue_depth', 0)
                       + embed_snap.get('edge_queue_depth', 0))
        rwq_depth = rwq_snap.get('access_queue_depth', 0)
        total_depth = embed_depth + rwq_depth

        if total_depth == 0:
            return

        # "Last drain" = the most recent of either queue's last drain.
        embed_last = embed_snap.get('last_drain_at')
        rwq_last = rwq_snap.get('last_drain_at')
        last_drain_at = max(t for t in (embed_last, rwq_last) if t is not None) \
            if (embed_last or rwq_last) else None

        if last_drain_at is None:
            return  # never drained — first cycle, ignore

        age_s = time.time() - last_drain_at
        if age_s < STALL_THRESHOLD_S:
            return

        with _lock:
            _stats['stalls_logged_total'] += 1

        # Diagnostic enrichment: who holds write_lock right now, and
        # how long did the most recent BEGIN IMMEDIATE wait? Both are
        # best-effort — the snapshot() method only exists on
        # TrackedRLock and get_stats may not have the wait yet.
        holder_info = ''
        try:
            wl = getattr(brain, 'write_lock', None)
            if wl is not None and hasattr(wl, 'snapshot'):
                snap = wl.snapshot()
                if snap.get('holder'):
                    holder_info = ', write_lock_held_by=%s for %dms (depth=%d)' % (
                        snap['holder'], snap.get('held_for_ms') or 0,
                        snap.get('depth') or 0)
                else:
                    holder_info = ', write_lock=free'
        except Exception:
            pass

        last_wait = rwq_snap.get('last_begin_wait_ms', 0)
        wait_info = ', last_begin_wait=%dms' % last_wait if last_wait else ''

        try:
            brain._log_error(
                'bg_writer_worker_stalled',
                RuntimeError('no drain in %ds, embed_depth=%d rwq_depth=%d%s%s'
                             % (int(age_s), embed_depth, rwq_depth,
                                holder_info, wait_info)),
                'worker appears blocked or slow — investigate drain duration')
        except Exception as le:
            print('[embed_queue] stall log failed: %s' % le, file=sys.stderr)
    except Exception as e:
        # Stall check is best-effort observability — never raise.
        print('[embed_queue] _check_stall failed: %s' % e, file=sys.stderr)


def _render_trace_for_embedding(row: Dict) -> str:
    """Render a trace_event row to embedding text per docs/EPISODIC-REFERENCES.md §5.3.

    Concrete identity tokens at the embedding layer (revised decision
    19, biology-grounded): same individual's traces land in the same
    vector neighborhood regardless of role/context changes. Falls back
    to OPERATOR / ANCHOR sentinels if identity isn't configured in the
    trace metadata yet (so the pipeline keeps producing usable vectors
    on fresh installs).
    """
    meta = row.get('metadata') or {}
    ref_type = row.get('ref_type', '')
    human = meta.get('human_identity') or 'OPERATOR'
    agent = meta.get('agent_identity') or 'ANCHOR'
    # Prefer the longer metadata.content over the truncated summary;
    # falls back to summary when content isn't present (tool_result,
    # historical traces, etc.).
    content = meta.get('content') or row.get('summary') or ''
    if ref_type == 'user_message':
        return '%s: %s' % (human, content)
    if ref_type == 'assistant_message':
        return '%s: %s' % (agent, content)
    if ref_type == 'tool_result':
        tool = meta.get('tool') or 'tool'
        summary = row.get('summary') or ''
        return '%s via %s: %s' % (agent, tool, summary)
    # Unknown ref_type: best-effort tag so the embedder still gets
    # signal. Future ref_types (S1 recall, S1 encoding) can override
    # by adding a branch above; the design doc §5.3 has templates.
    return '%s: %s' % (ref_type or 'event', content)


def _newer_iso(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """The later of two ISO cutoffs — i.e. the tighter lower bound.

    None means "no bound", so it loses to any real timestamp. Both operands
    come from iso_cutoff, so they share one format and compare lexicographically
    (the cross-format trap iso_cutoff's docstring describes needs two producers).
    """
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _drain_trace_embeddings_once(brain) -> None:
    """Pull-reconciliation tick: find recent S0 traces with no embedding,
    render → embed in one batch → store. Independent from node/edge
    drains (different table; store goes through TraceDAL's serialized
    write boundary); runs even when those queues are empty. Skip-tick on
    prior overlap is unnecessary — TRACE_DRAIN_LIMIT caps per-tick work.

    The scan is bounded from below by the watermark on a fast tick and by the
    window alone on a deep pass (see TRACE_DEEP_SCAN_INTERVAL). An empty result
    is what moves the watermark up: it proves nothing above `since` is missing
    an embedding, which is the same claim at either cadence. A tick that FINDS
    work leaves the watermark where it is — those rows are still unembedded
    when the scan ends, and the next tick must see them again.
    """
    global _trace_watermark, _last_trace_deep_at
    try:
        try:
            window_days = float(brain.get_config(
                'embed_queue.trace_window_days', TRACE_EMBED_WINDOW_DAYS))
        except Exception:
            window_days = TRACE_EMBED_WINDOW_DAYS
        window_iso = iso_cutoff(days=window_days) if window_days > 0 else None
        # Taken BEFORE the scan: a trace committed while it runs was stamped
        # before this instant, so adopting this floor cannot step over it.
        # Wall clock, because it is compared against stored created_at. The
        # two clock-step directions are not symmetric: a backward step only
        # widens the next scans, while a forward step parks the watermark ahead
        # of the traces written after it, which fast ticks then skip. The deep
        # pass is what bounds that — it ignores the watermark entirely.
        scan_floor = iso_cutoff(minutes=TRACE_WATERMARK_LAG_MINUTES)
        mono = time.monotonic()
        with _lock:
            deep = (mono - _last_trace_deep_at) >= TRACE_DEEP_SCAN_INTERVAL
            since = window_iso if deep else _newer_iso(window_iso,
                                                       _trace_watermark)
        pending = brain._trace_dal.find_unembedded(
            limit=TRACE_DRAIN_LIMIT,
            scales=list(EAGER_TRACE_SCALES),
            ref_types=list(EAGER_TRACE_REF_TYPES),
            since=since)
        if deep:
            # Stamped only once the scan RETURNED. A scan that raised read
            # nothing, so it must neither burn the slot — that would leave the
            # window unexamined for a full interval per failure — nor count as
            # coverage that happened. Stamped with the pre-scan reading: the
            # interval measures from when the pass started.
            with _lock:
                _last_trace_deep_at = mono
                _stats['traces_deep_scans_total'] += 1
    except Exception as e:
        with _lock:
            _stats['traces_errors_total'] += 1
        try:
            brain._log_error('embed_queue_trace_find', e,
                             'find_unembedded raised')
        except Exception:
            pass
        return
    if not pending:
        with _lock:
            _trace_watermark = scan_floor
            _stats['last_trace_drain_at'] = time.time()
        return

    try:
        from servers import embedder
        if not embedder.is_ready():
            # First few ticks during boot are expected — embedder
            # loads asynchronously. After that, "not ready" usually
            # means it failed to load and the worker would spin
            # forever silently. Track a per-process count and log
            # loudly when it persists.
            with _lock:
                _stats['traces_skipped_embedder_not_ready'] = (
                    _stats.get('traces_skipped_embedder_not_ready', 0) + 1)
                skips = _stats['traces_skipped_embedder_not_ready']
            # Loud once at the 5-tick mark (~25s), then every 50 ticks
            # (~4 min). Catches genuine boot grace; surfaces stuck state.
            if skips == 5 or (skips > 5 and skips % 50 == 0):
                try:
                    brain._log_error(
                        'embed_queue_trace_embedder_not_ready',
                        RuntimeError('embedder.is_ready() False after %d ticks' % skips),
                        'worker can\'t produce trace embeddings; check embedder.get_model_status()')
                except Exception:
                    pass
            return
        t0 = time.time()
        texts = [_render_trace_for_embedding(row) for row in pending]
        vectors = embedder.embed_batch(texts, kind='document')
        if not vectors or len(vectors) != len(pending):
            with _lock:
                _stats['traces_errors_total'] += 1
            try:
                brain._log_error(
                    'embed_queue_trace_embed_mismatch',
                    RuntimeError(
                        'embed_batch returned %d vectors for %d texts' %
                        (len(vectors) if vectors else 0, len(pending))),
                    'embedder returned partial or empty result; traces will be retried next tick')
            except Exception:
                pass
            return
        rows_to_store = []
        for row, vec, text in zip(pending, vectors, texts):
            if vec is not None:
                rows_to_store.append((row['id'], vec, text))
        if rows_to_store:
            model = embedder.get_config().get('model_name', 'unknown')
            n_written = brain._trace_dal.store_embeddings(
                rows_to_store, model=model)
            elapsed_ms = int((time.time() - t0) * 1000)
            with _lock:
                _stats['traces_processed_total'] += n_written
                _stats['last_trace_drain_at'] = time.time()
                if deep and n_written:
                    # A deep pass that wrote found a backlog older than the
                    # watermark; hold the deep cadence until it comes back
                    # empty. Keyed on rows WRITTEN, not rows found, so a trace
                    # that can never be embedded — it is found on every pass
                    # and written by none — cannot pin every tick to a
                    # full-window scan, which is the cost this watermark exists
                    # to remove.
                    _last_trace_deep_at = (time.monotonic()
                                           - TRACE_DEEP_SCAN_INTERVAL)
            print('[embed_queue] trace_embed %d in %dms (render+embed+store)' %
                  (n_written, elapsed_ms), flush=True)
    except Exception as e:
        with _lock:
            _stats['traces_errors_total'] += 1
        try:
            brain._log_error('embed_queue_trace_phase', e,
                             'render/embed/store failed')
        except Exception:
            pass


def _coverage_sweep(brain) -> None:
    """Unscoped vector sweep — driven by the worker loop, throttled.

    The queue repairs only what reached it. A node written by a path that
    skips the enqueue hooks, or one lost to a crash between insert and
    enqueue, has no other route back to being embedded.

    Called from `_worker_loop`, NOT from `_drain_once`: the drain's own
    early-return only fires on an empty tick, so hanging the sweep there
    starved it under sustained write load — exactly when an enqueue miss is
    most likely. The loop runs every tick regardless, and the emptiness
    check below is what defers to scoped work. Keeping it out of
    `_drain_once` also keeps it out of the test harness, which calls that
    function directly.

    BOTH outcomes are reported, because they mean different things:
      - repaired > 0  → a node reached a drained queue unembedded; the
        enqueue path missed it.
      - repaired == 0 while `_primary` is still missing → a node CANNOT be
        embedded. That is the one worth waking up for, and reporting only
        on successful repair would have hidden it forever.
    """
    global _last_sweep_at
    now = time.time()
    with _lock:
        if now - _last_sweep_at < COVERAGE_SWEEP_INTERVAL:
            return
        # Scoped work takes precedence — a non-empty queue means the drain is
        # mid-flight and its ids are about to be covered anyway. But not
        # forever: under sustained load the queue is rarely empty at the moment
        # of the check, and that is exactly when a writer is most likely to
        # have bypassed the enqueue hooks. MAX_STALENESS is the floor that
        # stops "defer to scoped work" from becoming "never run".
        if (_queue or _edge_queue) and (
                now - _last_sweep_at < COVERAGE_SWEEP_MAX_STALENESS):
            return
        _last_sweep_at = now
    try:
        outcome = brain.vector_coverage_sweep(COVERAGE_SWEEP_BATCH)
        if outcome['repaired']:
            brain._log_error(
                'embed_coverage_gap', None,
                'unscoped sweep repaired %d vector(s) the enqueue path never '
                'queued: %s' % (outcome['repaired'], outcome['by_type']))
        elif outcome['stuck']:
            brain._log_error(
                'embed_coverage_stuck', None,
                'node(s) missing _primary that the sweep could not embed — '
                'first: %s' % outcome['stuck'][0].get('id'))
        if outcome['remaining']:
            # A type filled its batch, so more waits behind it — clear the
            # throttle rather than rate-limiting recovery to one batch per
            # interval. Keyed on the repair's own report, so a permanently
            # unembeddable node (which repairs nothing) cannot spin this.
            # Rewind by one interval rather than zeroing: 0.0 is not "due
            # now", it reads as ~epoch seconds of staleness and would punch
            # through the queue-deference floor as well as the throttle.
            with _lock:
                _last_sweep_at = time.time() - COVERAGE_SWEEP_INTERVAL
    except Exception as e:
        brain._log_error('embed_coverage_sweep', e,
                         'unscoped vector coverage sweep failed')


def _drain_once(brain) -> None:
    """Drain each queued id once per call, in BATCHES.

    Processes every queued id NOT already attempted this drain, then returns.
    Ids re-enqueued mid-drain — an edge-embed retry when the embedder isn't
    ready, or the failure path below — deliberately stay in the queue for the
    NEXT _drain_once rather than being re-consumed here (re-consuming them spun
    the loop forever on a persistent failure). The worker loop re-picks them on
    its next tick. So the queues are NOT guaranteed empty on return.

    Holds brain.write_lock per-batch so cold-start (~20K entities at
    first boot) doesn't block other writers for the full drain. Each
    batch:
      1. backfill_vectors(node_ids) — writes node_enrichments embeddings
         (idempotent: vdal.find_missing skips already-vectorized nodes)
      2. backfill_edge_embeddings(edge_ids) — re-embeds invalidated edges
    """
    t0 = time.time()
    total_nodes = 0
    total_edges = 0
    total_vectors = 0
    total_edge_vectors = 0
    batches = 0
    # Ids already attempted in THIS drain. A batch can be re-enqueued mid-drain —
    # by the edge-embed retry (backfill_edge_embeddings re-queues when the
    # embedder isn't ready) or the failure path below. Re-enqueue means "retry on
    # a LATER drain", so we must not re-consume it within this `while True`, or a
    # persistent failure (embedder unavailable) spins the loop forever. Tracking
    # what we've attempted lets re-enqueued ids sit in the queue for the next
    # _drain_once instead.
    processed_nodes: Set[str] = set()
    processed_edges: Set[str] = set()

    while True:
        # Pull one batch out of the queues under the queue lock — only ids not
        # already attempted this drain.
        with _lock:
            fresh_nodes = _queue - processed_nodes
            fresh_edges = _edge_queue - processed_edges
            if not fresh_nodes and not fresh_edges:
                break
            node_batch = list(fresh_nodes)[:DRAIN_BATCH_SIZE]
            budget = DRAIN_BATCH_SIZE - len(node_batch)
            edge_batch = list(fresh_edges)[:budget] if budget > 0 else []
            # Remove what we took from the live queues and mark it attempted, so
            # a mid-drain re-enqueue stays queued for the NEXT drain.
            _queue.difference_update(node_batch)
            _edge_queue.difference_update(edge_batch)
            processed_nodes.update(node_batch)
            processed_edges.update(edge_batch)
        if not node_batch and not edge_batch:
            break

        # ─── Vectors phase ─────────────────────────────────────────
        # Vectors stay on the primary connection (CachedVectorDAL holds
        # invalidation state keyed to self.conn — migrating to
        # conn_bg_writer is a separate refactor). The write_lock used to
        # wrap this whole call, which held the lock for the duration of
        # embed_batch — seconds to minutes of CPU work blocking every
        # other writer. backfill_vectors now self-locks only around the
        # DB write+commit per batch, so other writers get the slot
        # between vector types and between batches.
        if node_batch:
            try:
                result = brain.backfill_vectors(
                    batch_size=len(node_batch), node_ids=node_batch)
            except TypeError:
                result = brain.backfill_vectors(
                    batch_size=len(node_batch))
            total_vectors += sum(
                v for v in (result or {}).values() if isinstance(v, int))

        # ─── Edge embedding phase ──────────────────────────────────
        # Re-embed edges whose embedding was invalidated (NULLed) by a
        # relation/description change. backfill_edge_embeddings self-locks
        # around its writes (compute runs outside the lock) and is idempotent
        # (skips rows already embedded). Same async mechanism that embeds nodes
        # and traces — write paths only invalidate; this is where the work lands.
        if edge_batch:
            try:
                total_edge_vectors += brain.backfill_edge_embeddings(edge_batch)
            except Exception as e:
                # Re-enqueue so a transient failure doesn't permanently drop the
                # batch — the rows stay NULL (recall live-fallback) until a later
                # drain retries. Safe to re-enqueue into the live queue: the
                # process-once guard above won't re-consume these ids within the
                # current drain, so a persistent failure can't spin the loop.
                for _eid in edge_batch:
                    enqueue_edge(_eid)
                try:
                    brain._log_error(
                        'bg_writer_drain_edge_embed', e,
                        'edge embedding batch re-enqueued after error: edges=%d'
                        % len(edge_batch))
                except Exception as le:
                    print('[embed_queue] edge embedding error: %s '
                          '(log failed: %s)' % (e, le), file=sys.stderr)

        total_nodes += len(node_batch)
        total_edges += len(edge_batch)
        batches += 1

    if batches == 0:
        # Empty tick — see matching note in recall_write_queue.drain_once.
        # Worker is alive, just had nothing to drain. Stamp last_drain_at
        # so the stall watchdog doesn't false-positive when a burst of
        # enqueues lands after a long idle period.
        with _lock:
            _stats['last_drain_at'] = t0
        return

    elapsed_ms = int((time.time() - t0) * 1000)
    with _lock:
        _stats['drains_total'] += 1
        _stats['nodes_processed_total'] += total_nodes
        _stats['edges_processed_total'] += total_edges
        _stats['vectors_written_total'] += total_vectors
        _stats['edge_vectors_written_total'] = (
            _stats.get('edge_vectors_written_total', 0) + total_edge_vectors)
        _stats['last_drain_at'] = t0
        _stats['last_drain_took_ms'] = elapsed_ms
        _stats['last_drain_size'] = total_nodes + total_edges
    if batches > 1 or total_edge_vectors > 0:
        print('[embed_queue] drained %d entities in %d batches (%dms): '
              'nodes=%d edges=%d vectors=%d edge_vectors=%d'
              % (total_nodes + total_edges, batches, elapsed_ms,
                 total_nodes, total_edges, total_vectors, total_edge_vectors),
              flush=True)
