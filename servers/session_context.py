"""SessionContext — per-request session identity.

The brain is a singleton. Sessions are not. SessionContext flows with every
brain call — hooks, MCP, encoding. The brain serves requests tagged with
context, like a database server.

SessionContext carries:
- session_id: identity (from Claude Code hook args)
- stop_counter: which stop we're on
- fatigue: {node_id: access_count} for synaptic fatigue (resets between sessions)
- boot_time: when the session booted (empty means never booted)

Usage:
    # In a hook handler:
    ctx = brain.get_or_create_session(session_id)
    chain = ctx.s0_chain()  # 's0-{session_short}-{stop}'

    # Persist across daemon restarts (dal = brain._session_state):
    ctx.save(dal)
    ctx = SessionContext.load(dal, session_id)
"""
import json
import uuid
from typing import Dict, Optional


class SessionContextCorrupt(Exception):
    """Raised by SessionContext.load when a session_state row EXISTS but its blob
    won't parse (corrupt) — as opposed to absent, which returns None. The
    brain-holding caller catches this and logs via the canonical
    brain._log_error; SessionContext stays a pure data carrier (no brain ref, no
    DAL reach, no logging of its own)."""


class SessionContext:
    """Per-session state that flows with every brain call."""

    def __init__(self, session_id: str = '', stop_counter: int = 0):
        self.session_id = session_id or uuid.uuid4().hex
        self.stop_counter = stop_counter
        # Turn classification (see trace_contract S0 TURN CLASSIFICATION):
        # last_recall_stop = the stop_counter value at which a real UserPromptSubmit
        # last ran hook_recall. A turn is "conversational" iff recall ran THIS stop
        # (last_recall_stop == stop_counter); a /watch wakeup skips recall
        # client-side, so its stop never updates this → it reads as a heartbeat.
        # -1 = recall has never run for this session.
        self.last_recall_stop: int = -1
        # Twin of last_recall_stop for delivery continuations: the stop
        # number a dial-on delivery-block armed (+ when, for the freshness
        # window) — classified and consumed in post_response_common, cleared
        # on boot/resume. -1/'' = none armed.
        self.last_delivery_stop: int = -1
        self.last_delivery_armed_at: str = ''
        # Integration cadence (every Nth conversational turn → S1 Scribe) is NOT
        # a stored counter anymore — it's derived live from traces via
        # turns_since_last_encode(). A maintained counter desynced across resume
        # (boot reset it while the traces stayed truthful), starving the Scribe.
        # stop_counter stays a real counter: it names chain IDs and must be
        # assigned synchronously, before its trace exists.
        # Transient (not persisted): set by post_response_common each turn so the
        # Stop hook's Scribe gate only fires on conversational turns.
        self.last_turn_conversational: bool = True
        self.fatigue: Dict[str, int] = {}  # {node_id: access_count} — resets between sessions
        # Per-turn "Anchor touched" accumulator — the S0 mirror of the encoder's
        # ops-delta. Anchor's own MCP tools append node ids here as they fire
        # (writes via dispatch `affected`, deliberate reads via get_node(s));
        # post_response_common flushes it as one `anchor_touched` S0 delta at the
        # turn boundary, then resets. Persisted with the row, so a restart
        # mid-turn keeps the touches made before it. Keyset driven by
        # ANCHOR_TOUCHED_KEYS (the contract) so it can't drift from the builder
        # it feeds via **touched.
        from .trace_contract import ANCHOR_TOUCHED_KEYS
        self.touched: Dict[str, list] = {k: [] for k in ANCHOR_TOUCHED_KEYS}
        # Per-session node activity — the parallel-session replacement for
        # global nodes.{last_accessed, access_count}
        # in reads that should be session-scoped (spreading-activation kernel,
        # recency filtering, live-session Frame composition). Global nodes
        # columns stay populated by the drain for S2 maintenance + dashboard
        # analytics. See bump_node_activity() for write semantics.
        # Shape: {node_id: {'activation': float, 'recency_score': float,
        #                   'access_count': int, 'last_accessed': iso_str}}
        self.node_activity: Dict[str, Dict[str, object]] = {}
        self.boot_time: str = ''  # ISO timestamp; empty means not booted yet
        # The Thalamus delivered at this session's first operator prompt
        # (daemon_hooks.hook_recall → channels/delivery.PROMPT). Once per
        # session: later prompts are mid-thread, and Stop carries the rest.
        self.prompt_delivered: bool = False
        # Session env — the Claude-side facts (working dir + git branch) FED IN
        # from the boot hook; the daemon never introspects Claude to learn them.
        # Per-session identity (not activity), surfaced in peek so sessions can
        # tell where each other is working. Stable for the session's life.
        self.cwd: str = ''
        self.branch: str = ''
        # The linked worktree this session works in (CC WorktreeCreate `name`),
        # or '' for the main working tree. Per-session — replaces the global
        # `current_worktree` config that was last-writer-wins across parallel
        # sessions (two sessions in different worktrees clobbered each other).
        # Stamped at boot (derived from cwd) and refreshed on WorktreeCreate/Remove.
        self.worktree: str = ''
        # Deterministic project identity — the main repo's directory name
        # (worktree-resolved via git common-dir), '' for non-repo sessions.
        # PROVENANCE, not aboutness: "learned while working in <repo>". Never
        # written by an agent — derived from cwd at boot, stamped onto every
        # S0/S1 write by the dispatch chokepoints. Read by the LAF proj lane.
        self.project: str = ''
        # What the session rides on — the model that produced the latest turn
        # and the host runtime resolved by host_contract. Fed in per turn by the
        # UserPromptSubmit / Stop hooks; stamped onto every S0 row by the S0
        # write door (trace_contract.S0_SESSION_STAMP_FIELDS). The value here is
        # the LATEST one — presence reads it; the per-turn record is the trace.
        self.model: str = ''
        self.host: str = ''
        # Transient (not persisted): the blob last written to or read from this
        # session's row — save(if_changed=True) skips a write that would not
        # change it. None until the first save/load.
        self._saved_blob: Optional[str] = None

    @property
    def session_short(self) -> str:
        """The session part of chain IDs — trace_contract.session_short."""
        from .trace_contract import session_short
        return session_short(self.session_id)

    def increment_stop(self):
        """Increment stop counter. Called by the Stop hook."""
        self.stop_counter += 1

    # ── Chain ID generators ──

    def s0_chain(self) -> str:
        """S0 chain: one per stop — messages + tools."""
        return 's0-%s-%d' % (self.session_short, self.stop_counter)

    def s1r_chain(self) -> str:
        """S1 recall chain: recall/surface for this stop."""
        return 's1r-%s-%d' % (self.session_short, self.stop_counter)

    def s1e_chain(self) -> str:
        """S1 encode chain: encoding run triggered at this stop."""
        return 's1e-%s-%d' % (self.session_short, self.stop_counter)

    # ── Frame ──

    def get_frame(self, brain, at=None) -> str:
        """Build and return Anchor's structured awareness Frame for this session.

        Frame Constructor — markdown text with two sections (Session /
        Current focus), composed from per-session state: the deterministic
        situational header (project, counterpart, clock, worktree) plus this
        session's arc. No LLM call, no queried nodes.

        `at`: conversation-time for the header clock. Production omits it
        (wall-clock is the conversation's now); REPLAYS MUST PASS their
        injected time or the header renders today's date into a historical
        conversation and corrupts temporal judgments.

        Brain is passed as a dependency rather than stored on SessionContext —
        SessionContext is a per-request data carrier, brain is the singleton.
        Rebuilt fresh on every call (no caching).

        See servers/scales/s1/frame.py.
        """
        from servers.scales.s1.frame import build_frame
        return build_frame(brain, self.session_id, at=at)

    # ── Persistence ──

    def increment_fatigue(self, node_id: str) -> int:
        """Increment fatigue for a node. Returns new count."""
        self.fatigue[node_id] = self.fatigue.get(node_id, 0) + 1
        return self.fatigue[node_id]


    def bump_node_activity(self, node_id: str, ts: str) -> Dict[str, object]:
        """Mark a node as accessed by this session at time `ts`.

        Semantics mirror the global `nodes` UPDATE in recall_write_queue.drain:
          access_count += 1
          activation   = min(1.0, activation + 0.1)
          recency_score = 1.0  (reset on access)
          last_accessed = ts   (caller passes wall-clock or conversation-time)

        First access creates the entry with access_count=1, activation=1.0
        (capped at the bump-from-zero ceiling — matches `remember()` initial).
        Returns the updated record.
        """
        if not node_id:
            return {}
        rec = self.node_activity.get(node_id)
        if rec is None:
            rec = {
                'activation': 1.0,
                'recency_score': 1.0,
                'access_count': 1,
                'last_accessed': ts,
            }
        else:
            current = float(rec.get('activation', 0.0))
            rec['activation'] = min(1.0, current + 0.1)
            rec['recency_score'] = 1.0
            rec['access_count'] = int(rec.get('access_count', 0)) + 1
            # Monotonic ts: ISO-8601 strings sort lexicographically.
            existing_ts = rec.get('last_accessed', '')
            if not existing_ts or ts > existing_ts:
                rec['last_accessed'] = ts
        self.node_activity[node_id] = rec
        return rec

    def set_env(self, cwd: str = '', branch: str = '', worktree=None,
                project=None, model: str = '', host: str = '') -> None:
        """Stamp the Claude-side session env — where this session is working and
        what it rides on.

        Single mutator for the per-session identity fed in from the boot hook, the
        WorktreeCreate/Remove hooks and the per-turn recall/Stop hooks (the daemon
        never introspects the host). The per-session replacement for the global
        cwd/branch/worktree config that was last-writer-wins across parallel
        sessions.

        cwd/branch/model/host refresh only on a truthy value — falsy (''/None)
        leaves the existing value, so a failed detection never clobbers a known
        one. worktree and project are three-state: None leaves them unchanged
        (detection failed — keep what we have), '' CLEARS (main tree / non-repo),
        a name SETS. That None-vs-'' distinction is why detect_git_env returns
        None on git failure.
        """
        if cwd:
            self.cwd = cwd
        if branch:
            self.branch = branch
        if worktree is not None:
            self.worktree = worktree
        if project is not None:
            self.project = project
        if model:
            self.model = model
        if host:
            self.host = host

    def save(self, dal, *, if_changed: bool = False) -> bool:
        """Save session context to DB. Creates or updates. Returns whether it
        wrote.

        One JSON row per session — fatigue, env, node activity and the
        per-turn bookkeeping below. `if_changed` skips a write whose blob
        matches the one last saved or loaded (the periodic flush uses it, so an
        idle session's row is not rewritten every pass).
        """
        data = json.dumps({
            'stop_counter': self.stop_counter,
            'last_recall_stop': self.last_recall_stop,
            'last_delivery_stop': self.last_delivery_stop,
            'last_delivery_armed_at': self.last_delivery_armed_at,
            'fatigue': self.fatigue,
            'boot_time': self.boot_time,
            'prompt_delivered': self.prompt_delivered,
            'cwd': self.cwd,
            'branch': self.branch,
            'worktree': self.worktree,
            'project': self.project,
            'model': self.model,
            'host': self.host,
            'node_activity': self.node_activity,
            'touched': self.touched,
        })
        if if_changed and data == self._saved_blob:
            return False
        dal.set(self.session_id, '_session_context', data)
        self._saved_blob = data
        return True

    @classmethod
    def load(cls, dal, session_id: str) -> 'SessionContext':
        """Load session context from DB. Returns None if not found.

        `dal` is a SessionStateDAL — production callers pass the brain's held
        instance (brain._session_state); brain-free tests construct one over
        their own connection.
        """
        value = dal.get(session_id, '_session_context')
        if not value:
            return None
        try:
            data = json.loads(value)
            ctx = cls(
                session_id=session_id,
                stop_counter=data.get('stop_counter', 0),
            )
            ctx.last_recall_stop = int(data.get('last_recall_stop', -1))
            ctx.last_delivery_stop = int(data.get('last_delivery_stop', -1))
            ctx.last_delivery_armed_at = data.get('last_delivery_armed_at', '') or ''
            ctx.fatigue = {k: int(v) for k, v in data.get('fatigue', {}).items()}
            ctx.boot_time = data.get('boot_time', '') or ''
            ctx.prompt_delivered = bool(data.get('prompt_delivered', False))
            ctx.cwd = data.get('cwd', '') or ''
            ctx.branch = data.get('branch', '') or ''
            ctx.worktree = data.get('worktree', '') or ''
            ctx.project = data.get('project', '') or ''
            ctx.model = data.get('model', '') or ''
            ctx.host = data.get('host', '') or ''
            raw_activity = data.get('node_activity', {}) or {}
            # Coerce types — JSON round-trip preserves dicts but the inner
            # numerics need conversion when an older session_state row
            # predates this field (raw_activity will be {} then; loop no-op).
            for nid, rec in raw_activity.items():
                if isinstance(rec, dict):
                    ctx.node_activity[nid] = {
                        'activation': float(rec.get('activation', 0.0)),
                        'recency_score': float(rec.get('recency_score', 0.0)),
                        'access_count': int(rec.get('access_count', 0)),
                        'last_accessed': str(rec.get('last_accessed', '') or ''),
                    }
            # The contract's keyset, from __init__: a key the row lacks (an
            # older row, a newer key) starts empty; one it has beyond it drops.
            raw_touched = data.get('touched') or {}
            for key in ctx.touched:
                ctx.touched[key] = list(raw_touched.get(key) or [])
            ctx._saved_blob = value
            return ctx
        except (json.JSONDecodeError, TypeError) as e:
            # The row EXISTED but won't parse — corrupt, not absent. Signal it so
            # the brain-holding caller logs via the canonical brain._log_error.
            # (Pure data carrier: no brain ref, no DAL reach, no logging here.)
            # Named by the owner directly: a parse failure leaves no ctx.
            from .trace_contract import session_short
            raise SessionContextCorrupt(
                'corrupt session_state blob for %s: %s' % (session_short(session_id), e)
            ) from e
