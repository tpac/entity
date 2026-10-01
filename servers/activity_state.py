"""Daemon-wide activity state — the signals that gate background work.

Single source of truth for the runtime counters that decide when S2
maintenance (and the connection keepalive) may run. Held on the brain
(`brain.activity`) so the gate logic reads one cohesive object instead of
parameters threaded through call sites.

GLOBAL, not per-session. S2 operates on the whole graph and one daemon serves
many concurrent streams, so these counters aggregate across all of them — they
are deliberately NOT keyed by session_id.

Mutated by the daemon / hooks as events arrive (record_*); read — and consumed —
by Brain.run_maintenance_if_due.
"""

import time


class ActivityState:
    """Aggregate activity signals consumed by the S2 gate and keepalive.

    Why encode runs, not surface/recall count: S2's material is *encoded
    nodes*. A recall (S1 Surface) reads the graph; it creates nothing for S2
    to consolidate. Only the S1 Encoder (Scribe) writes new nodes — so the
    "is there new work for S2?" signal is encoder runs since the last S2 fire.
    The encode count is recorded on the Scribe's COMPLETION and only when it
    actually wrote material (write_actions > 0), not at dispatch — a run that
    failed or wrote nothing produces nothing for S2 to do.

    Concurrency: deliberately lock-free. record_* runs in request/hook threads,
    consume_encode_runs in the maintenance pool thread. The int += and the
    read-subtract-write are not atomic, so a racing increment can be lost — but
    the only effect is the counter drifting LOW (S2 fires slightly less often,
    which is the intended direction), never negative (max(0) floors it) and
    never an over-fire. A lock would buy nothing here; do not add one.

    Lifetime: in-memory and ephemeral — resets on daemon restart. The S2
    min-interval timestamp IS persisted (brain_meta), but the encode count is
    intentionally not: after a restart we want fresh material to accrue before
    S2 fires (boot-grace already suppresses early S2), so starting from 0 is
    correct, not a bug.
    """

    def __init__(self) -> None:
        # Epoch seconds of the last real user prompt (hook_recall). 0.0 means
        # "no prompt since boot" → the gate treats it as infinitely idle.
        self.last_user_activity: float = 0.0
        # S1 Encoder (Scribe) runs since the last S2 maintenance fire.
        self.encode_runs_since_maintenance: int = 0

    def record_user_activity(self, now: float = None) -> None:
        """Mark a real user prompt (UserPromptSubmit / hook_recall)."""
        self.last_user_activity = now if now is not None else time.time()

    def record_encode_run(self) -> None:
        """Mark that an S1 Encoder (Scribe) run was dispatched — new material
        for S2 to consolidate."""
        self.encode_runs_since_maintenance += 1

    def consume_encode_runs(self, n: int) -> None:
        """Subtract the count the gate decided on when S2 fires.

        Subtract — not zero — so encoder runs that complete during the
        multi-minute S2 cycle still accrue toward the next cycle rather than
        being silently dropped.
        """
        self.encode_runs_since_maintenance = max(
            0, self.encode_runs_since_maintenance - n)
