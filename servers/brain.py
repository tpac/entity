"""The Brain class: core infrastructure, with everything else mixed in.

This file holds construction and DAL wiring, the singleton registry, session
and scope context, interaction prompt/config resolution, configuration,
maintenance and S2 gating, error and event logging, LLM availability,
embedder configuration, and save/close. Recall, traces, remember, connections,
reminders, assembly and corrections come from the mixins the class inherits,
each in its own brain_*.py file.

Storage: sqlite3 (WAL mode, FK on; see db_backends/sqlite.py). Embeddings are
written off the request path by the embed_queue worker.
"""

import sys
import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
import sqlite3
import uuid
import json
import time
import os
import threading
from typing import Dict, List, Optional, Any
from .schema import ensure_schema, ensure_logs_schema, migrate_logs_to_separate_db
from .dal import BrainMetaDAL
from .dal_logs import LOG_SWEEP_SAMPLES, LogsDAL
from .db_backends.sqlite import commit_unless_batched
from .clock import iso_cutoff, iso_now
from .brain_recall import BrainRecallMixin
from .brain_traces import BrainTracesMixin
from .brain_remember import BrainRememberMixin
from .brain_connections import BrainConnectionsMixin
from .brain_reminders import BrainRemindersMixin
from .brain_assembly import BrainAssemblyMixin
from .brain_corrections import BrainCorrectionsMixin
from . import embedder



# ═══════════════════════════════════════════════════════════════
# BRAIN CLASS
# ═══════════════════════════════════════════════════════════════

class Brain(
    BrainRecallMixin,
    BrainTracesMixin,
    BrainRememberMixin,
    BrainConnectionsMixin,
    BrainRemindersMixin,
    BrainAssemblyMixin,
    BrainCorrectionsMixin,
):
    """
    Core brain engine.

    Manages:
    - Node storage (memories, thoughts, rules, etc.)
    - Edge connections
    - Semantic recall (embeddings)
    - Session activity tracking

    Singleton pattern: Use Brain.get_instance(db_path) to reuse an existing
    warm Brain for the same db_path. Direct __init__ always creates a new instance
    (useful for tests, simulations, and fresh brains).
    """

    # ─── Singleton registry ───
    _instances: Dict[str, 'Brain'] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: str) -> 'Brain':
        """
        Get or create a singleton Brain for the given db_path.

        Returns an existing warm instance if one is already open for this path,
        avoiding repeated schema checks and embedder loads.
        Thread-safe.

        Args:
            db_path: Path to brain.db file

        Returns:
            Brain instance (cached or newly created)
        """
        canonical = os.path.realpath(db_path)
        with cls._lock:
            instance = cls._instances.get(canonical)
            if instance is not None:
                # Verify the connection is still alive
                try:
                    instance.conn.execute('SELECT 1')
                    return instance
                except Exception:
                    # Connection died — remove stale entry and recreate
                    del cls._instances[canonical]

            instance = cls(db_path)
            cls._instances[canonical] = instance
            return instance

    def __init__(self, db_path: str, skip_embedder: bool = False):
        """
        Initialize Brain with SQLite3 database.

        NOTE: For production use, prefer Brain.get_instance(db_path) which
        reuses warm instances. Direct __init__ always creates a fresh connection
        (appropriate for tests, simulations, and temporary brains).

        Args:
            db_path: Path to brain.db file
            skip_embedder: If True, skip loading the embedding model (~1GB).
                          Useful for tests that don't need semantic search.
        """
        self.db_path = db_path
        # Read once: a brain is an eval brain from the moment it opens, and
        # marking it while it runs changes nothing until it reopens.
        self._eval_brain_since = self.read_eval_mark(
            os.path.dirname(os.path.abspath(db_path)))

        # Batch state lives on the CONNECTION (self.conn.in_batch), set by
        # BatchAwareConnection below — not a brain-level flag. The batch
        # owner (_handle_brain_batch) flips conn.in_batch for the duration
        # of its BEGIN IMMEDIATE / COMMIT; DAL writers consult it via
        # commit_unless_batched() and brain._maybe_commit() does the same.
        # One source of truth on the resource it describes, so a writer
        # can't break batch atomicity by forgetting a kwarg. (Replaces the
        # old by-convention brain._batch_mode + per-call commit= kwarg.)

        # Serializes all writers to brain.db / brain_logs.db. The lock lives
        # on the brain (the resource it protects), not the daemon — this lets
        # any caller that holds a brain reference participate in
        # serialization: daemon dispatch, S2 encoder dispatch, embed_queue
        # backfill, autosave. RLock so a thread that already holds it can
        # call into a brain method that also wants to acquire it without
        # deadlock. TrackedRLock exposes the current holder for the
        # bg_writer stall watchdog — when a drain times out we want to
        # know which thread was holding when.
        from .tracked_lock import TrackedRLock
        self.write_lock = TrackedRLock()

        # Serializer for brain_logs.db writes (the DAL write boundary's
        # `_wlock`). Deliberately SEPARATE from write_lock: logs writes are
        # small and frequent (traces, error logs) and must not queue behind
        # multi-second graph batches on the other database. Lock ordering is
        # strictly write_lock -> logs_write_lock (the logs lock is a leaf:
        # no holder ever acquires write_lock, a graph lock, or calls back
        # out of the DAL while holding it), so no inversion is possible.
        self.logs_write_lock = TrackedRLock()

        # Non-blocking guard for the Anthropic connection warm (warm_up at boot
        # + the idle keepalive loop). With try-acquire(blocking=False) it makes
        # warm_anthropic_connection idempotent under concurrency: if a warm is
        # already in flight, a second caller skips rather than piling a
        # redundant models.retrieve on top. (httpx.Client is itself thread-safe,
        # so a warm racing a real recall is harmless — this only stops two
        # *warms* overlapping.)
        self._anthropic_warm_lock = threading.Lock()

        # S2 single-flight. Held for a WHOLE S2 cycle by run_s2() — the one door
        # to S2 activation. Same try-acquire(blocking=False) shape as the warm
        # lock: a second caller skips rather than queueing, because two
        # concurrent cycles are never wanted (consolidation is a multi-minute
        # LLM run; overlapping passes duplicate work and race each other's
        # fingerprints).
        #
        # WHY IT LIVES ON THE BRAIN, not the daemon: the daemon's `_s2_running`
        # is an instance attribute of the server, so it could only ever block a
        # second POLL entry. That is exactly how the 2026-06 parallel-run bug
        # happened — a second caller (the idle hook) couldn't see it and wasn't
        # seen by it. The guard belongs with the thing it guards, so ANY caller
        # in this process is serialized: the poll thread, and anything arriving
        # over TCP dispatch (including `eval`). Plain Lock, not RLock — S2 must
        # never re-enter itself.
        self._s2_lock = threading.Lock()

        # Open SQLite connection with WAL mode for concurrency. Pragma
        # set comes from db_backends.current — single source for every
        # connection in the daemon.
        from . import db_backends
        self.conn = sqlite3.connect(
            db_path, check_same_thread=False,
            factory=db_backends.current.BatchAwareConnection)
        db_backends.current.apply_pragmas(self.conn)

        # Background-writer connection (2026-05-18). Owned exclusively by the
        # embed_queue worker thread for batched, deferred, non-realtime writes:
        # vector backfill and recall-side access marks (`_mark_accessed`).
        # The recall hot path is read-only at
        # SQLite. Foreground writes via `self.conn` no longer race with
        # background batches at the WAL writer slot.
        #
        # Open failure here is a hard boot crash — the daemon cannot serve
        # writes without this connection. sqlite3 raises OperationalError,
        # which propagates to daemon_server boot path and surfaces in daemon.log.
        self.conn_bg_writer = sqlite3.connect(
            db_path, check_same_thread=False,
            factory=db_backends.current.BatchAwareConnection)
        db_backends.current.apply_pragmas(self.conn_bg_writer)

        # Daemon boot timestamp — used by run_maintenance_if_due to enforce
        # MAINTENANCE_BOOT_GRACE_SECONDS so maintenance never fires during
        # the first N seconds after start. Boot already runs schema +
        # migrations + S0/S1 startup paths; piling consolidation on top
        # makes first-user-recall consistently time out.
        import time as _time
        self._boot_time = _time.time()

        # Daemon-wide activity signals that gate background work (S2
        # maintenance + keepalive). Single source of truth; mutated by the
        # daemon/hooks, read by run_maintenance_if_due. Global, not per-session.
        from .activity_state import ActivityState
        self.activity = ActivityState()

        # Create schema if needed
        ensure_schema(self.conn, db_path=db_path)

        # Open separate logs database (brain_logs.db)
        db_dir = os.path.dirname(db_path) or '.'
        self.logs_db_path = os.path.join(db_dir, 'brain_logs.db')
        self.logs_conn = sqlite3.connect(
            self.logs_db_path, check_same_thread=False,
            factory=db_backends.current.BatchAwareConnection)
        db_backends.current.apply_pragmas(self.logs_conn)
        ensure_logs_schema(self.logs_conn, db_path=self.logs_db_path)

        # Dedicated logs WRITE connection. logs_conn serves concurrent reads
        # (read dispatch, embed drain scans) whose open cursors hold WAL read
        # snapshots; writing on that same connection is a read->write upgrade
        # that fails INSTANTLY with 'database is locked' whenever an external
        # process (hooks, MCP monitor) committed since the snapshot —
        # busy_timeout never applies to snapshot upgrades.
        # This connection is used only inside the DAL write methods under
        # write_lock, so it never holds a read cursor and every write
        # transaction begins at the WAL head.
        self.logs_conn_w = sqlite3.connect(
            self.logs_db_path, check_same_thread=False,
            factory=db_backends.current.BatchAwareConnection)
        db_backends.current.apply_pragmas(self.logs_conn_w)

        # One-time migration: move log tables from brain.db to brain_logs.db.
        # This runs AFTER ensure_logs_schema has stamped a version, so any rows
        # it imports arrive behind that stamp — it resets the logs counter when
        # it imports anything, so the migration ladder faces the legacy data on
        # the next open instead of treating it as already-current.
        migrate_logs_to_separate_db(self.conn, self.logs_conn,
                                    main_db_path=self.db_path)

        # DAL instances — incremental adoption, brain.py migrates one method at a time
        self._meta = BrainMetaDAL(self.conn)
        self._logs_dal = LogsDAL(self.logs_conn, write_conn=self.logs_conn_w,
                                 write_lock=self.logs_write_lock)
        from .dal_logs import TraceDAL, InteractionDAL, SessionStateDAL
        self._trace_dal = TraceDAL(self.logs_conn, write_conn=self.logs_conn_w,
                                   write_lock=self.logs_write_lock)
        self._interaction_dal = InteractionDAL(
            self.logs_conn, write_conn=self.logs_conn_w,
            write_lock=self.logs_write_lock)
        # logs-bound: session_state lives in brain_logs.db, not brain.db
        self._session_state = SessionStateDAL(
            self.logs_conn, write_conn=self.logs_conn_w,
            write_lock=self.logs_write_lock)

        # Repository aggregate (DAL cleanup Phase 2): hold the brain.db DALs
        # foreground-conn-bound so methods use them by construction instead of
        # re-instantiating XDAL(self.conn) ad hoc.
        from .dal import NodeDAL, Fts5DAL, SourceRefDAL
        from .dal_graph import GraphDAL
        from .dal_metadata import MetadataDAL
        self._nodes = NodeDAL(self.conn)
        self._graph = GraphDAL(self.conn)
        self._meta_kv = MetadataDAL(self.conn)
        self._fts = Fts5DAL(self.conn)
        self._source_refs = SourceRefDAL(self.conn)  # node_source_refs (not edges)

        # Identity binding — concrete names of the human partner and the
        # agent at this brain's "current moment." Stamped onto every S0
        # trace event so each row independently records who said what
        # (matches biology's per-utterance speaker binding; survives
        # partner changes without rewriting history). Sourced from env
        # at boot; empty when unset → DAL skips stamping.
        #
        # The "identity missing" loud signal lives at the write boundary
        # (TraceDAL._maybe_warn_identity_unset), not here — boot is one
        # moment; trace writes are continuous, and that's where the gap
        # actually manifests.
        from .daemon_config import (
            get_operator_name, get_agent_name, agent_name_is_default)
        self.operator_name = get_operator_name()
        self.agent_name = get_agent_name()
        self.agent_name_is_default = agent_name_is_default()
        self._trace_dal.set_identity(self.operator_name, self.agent_name,
                                     agent_is_default=self.agent_name_is_default)

        # Shared vector DAL — cache-backed by default. Set env
        # BRAIN_DISABLE_VECTOR_CACHE=1 to fall back to raw VectorDAL for
        # emergency rollback or A/B benchmarking. Brain consumers use
        # self._vec_dal either way.
        if os.environ.get('BRAIN_DISABLE_VECTOR_CACHE'):
            from .dal import VectorDAL
            self._vec_dal = VectorDAL(self.conn)
        else:
            from .dal_vector_cached import CachedVectorDAL
            self._vec_dal = CachedVectorDAL(self.conn)
        # Edge writes report changed endpoints; the brain invalidates their
        # edge_context through the revise() path. Registered after _vec_dal
        # exists — the handler deletes through it.
        self._graph.on_edge_text_changed = self._edge_text_changed
        # edge_context policy: the DAL is the mechanism (text producer,
        # eligibility fragment, write-side reports); which relations are
        # excluded is the brain's to derive, from the noise aspect. Installed
        # as a callable so the DAL reads it live — the registry rebinds the
        # set on every adopt.
        self._graph._edge_context_excluded_fn = self._edge_context_excluded

        # Init rate limiter for error logging (DDoS protection)
        self._init_rate_limiter()

        # Init human-readable log file (brain.log with rotation)
        self._init_file_logger(db_dir)

        # Load embedder with config from brain_meta (falls back to the pinned defaults)
        if not skip_embedder:
            try:
                embedder_config = self._get_embedder_config()
                embedder.load_model(embedder_config)
            except Exception as e:
                print(f'[brain] Embedder load failed (optional): {e}')
            # ASK whether it worked, OUTSIDE the try above: load_model does not
            # raise — it records the reason in stats['load_error'], leaves
            # _model None and returns — so that except almost never fires and a
            # dead embedder passed construction in silence. Silence is the
            # expensive part: the module state is process-global, so every later
            # recall returns empty and every backfill writes nothing, and the
            # first visible symptom is unrelated work failing on absent vectors.
            # Outside, because a failure in the LOGGING path caught by that
            # except would be reported as an embedder load failure it isn't.
            if not embedder.is_ready():
                self._log_error(
                    'embedder_load_failed',
                    RuntimeError(embedder.get_model_status()),
                    'embedder dead after load_model — vectors cannot be '
                    'written or queried; recall degrades to empty results')

        # AspectRegistry — first-class semantic-role API exposed as
        # brain.aspects, and the ONE door for aspects_v1.json writes.
        # Construction materializes the working copy (seed copy on first
        # boot + additive heal — reconcile_working_copy), then loads and
        # validates: checks REQUIRED_ASPECTS present, logs a warning if any
        # are missing (doesn't block). A Brain built with skip_embedder=True
        # still validates — the load is a cheap JSON read.
        #
        # AspectRegistry must be ready before any edge embedding runs: the
        # embed_queue worker (Brain.backfill_edge_embeddings) dereferences
        # brain.aspects to compose edge text. It's initialized here at __init__,
        # before the worker starts and before seed_baby_brain enqueues its
        # seed edges, so async edge embedding always has aspects available. (Edge
        # embedding moved async 2026-06 — seed's connect_typed no longer embeds
        # inline, so the old 'seed edges silently fail to embed' ordering hazard
        # is gone; keeping this init position is harmless + defensive.)
        try:
            from .aspects import AspectRegistry
            self.aspects = AspectRegistry(self)
        except Exception as _e:
            # Never let registry init crash Brain init. Log + leave aspects
            # unset; consumers that read brain.aspects will get AttributeError
            # which is louder than a silent empty registry.
            print('[brain] WARNING: AspectRegistry init failed: %s' % _e, flush=True)

        # Host contract: validate the per-harness declarations on every Brain
        # construction (daemon boot, background scale runs) — loud, never
        # fatal, no auto-heal (the aspects posture). The sync test is the write
        # door; this is the last line for an install whose code drifted past
        # it. One errors-table row per violation.
        try:
            from .host_contract import validate_host_contract
            for _violation in validate_host_contract():
                self._log_error('host_contract_invalid', ValueError(_violation))
        except Exception as _e:
            print('[brain] WARNING: host contract validation failed: %s' % _e, flush=True)

        # Seed baby brain nodes if missing (runs AFTER embedder — remember() needs it)
        if not skip_embedder:
            try:
                from .seed_pack import seed_baby_brain
                seed_baby_brain(self)
            except Exception as _e:
                print('[brain] WARNING: seed_pack failed: %s' % _e, flush=True)

    def _init_rate_limiter(self):
        """Initialize the same-condition dedup state for error logging."""
        self._error_fingerprints = {}  # fingerprint -> last_seen (monotonic)
        self._last_fingerprint_sweep = 0.0
        self._last_db_size_check = 0.0
        # SessionContext in-memory cache — keyed by session_id. Mutations
        # (fatigue, counters) live here across hooks within the same session;
        # the daemon's session flush persists changed entries every
        # SESSION_FLUSH_INTERVAL_SECONDS, and close() persists them before the
        # connections close. Cleared entries on SessionEnd hook.
        from typing import Dict as _Dict
        self._session_contexts: _Dict[str, 'SessionContext'] = {}
        # Limits
        self._error_dedup_window = 60      # seconds
        self._max_logs_db_size = 50 * 1024 * 1024  # 50MB

    def _init_file_logger(self, db_dir: str):
        """Initialize rotating file logger for human-readable brain.log."""
        import logging
        from logging.handlers import RotatingFileHandler
        self._file_logger = logging.getLogger('brain_%s' % id(self))
        self._file_logger.setLevel(logging.DEBUG)
        # Avoid duplicate handlers on re-init
        if not self._file_logger.handlers:
            try:
                handler = RotatingFileHandler(
                    os.path.join(db_dir, 'brain.log'),
                    maxBytes=5 * 1024 * 1024,  # 5MB per file
                    backupCount=2,              # keep brain.log.1 and brain.log.2
                )
                handler.setFormatter(logging.Formatter('%(message)s'))
                self._file_logger.addHandler(handler)
            except Exception as e:
                print('[brain] Could not init file logger: %s' % e)

    def _check_rate_limit(self, fingerprint: str) -> bool:
        """Suppress a repeat of the SAME condition inside the dedup window.

        Returns True if the event should be SUPPRESSED. Only an identical
        fingerprint seen inside the window is suppressed — a burst of
        *distinct* conditions is information and always reaches the table.
        Bounding the logs DB is LogsDAL retention's job (`run_maintenance`),
        not the write path's.
        """
        now = time.monotonic()
        last = self._error_fingerprints.get(fingerprint)
        if last is not None and (now - last) < self._error_dedup_window:
            return True
        if now - self._last_fingerprint_sweep > self._error_dedup_window:
            # An entry is dead once the window passes, so the map holds at most
            # the conditions seen in the last two windows — bounded without a
            # size knob. Swept on a timer, not on size: a storm of distinct
            # fingerprints has nothing expirable to drop, and a size trigger
            # would rebuild the whole map on every call instead.
            # `.copy()` is one uninterruptible C call, so the comprehension
            # walks a private snapshot: a concurrent logger from another daemon
            # thread cannot make it raise "dictionary changed size", which
            # _log_event's except would turn into a dropped error row.
            self._last_fingerprint_sweep = now
            snapshot = self._error_fingerprints.copy()
            self._error_fingerprints = {
                f: t for f, t in snapshot.items()
                if now - t < self._error_dedup_window}
        self._error_fingerprints[fingerprint] = now
        return False

    def _write_to_file_log(self, level: str, source: str, message: str, traceback_str: str = ''):
        """Write a formatted entry to brain.log."""
        try:
            ts = self.now()
            parts = ['[%s] %s %s: %s' % (ts, level, source, message)]
            if traceback_str:
                for line in traceback_str.strip().split('\n')[-5:]:
                    parts.append('  ' + line)
            parts.append('---')
            self._file_logger.info('\n'.join(parts))
        except Exception:
            pass

    def _check_logs_db_size(self):
        """Rotate logs DB if it exceeds size limit. Checked at most once per minute."""
        now = time.monotonic()
        if now - self._last_db_size_check < 60:
            return
        self._last_db_size_check = now
        try:
            size = os.path.getsize(self.logs_db_path)
            if size > self._max_logs_db_size:
                # Delete entries older than 7 days
                self._logs_dal.prune_oversize(iso_cutoff(days=7))
                self._write_to_file_log('INFO', 'logs_db', 'Pruned entries older than 7 days (DB was %dMB)' % (size // (1024*1024)))
        except Exception:
            pass

    def now(self) -> str:
        """Return current UTC ISO timestamp.

        Routes through ``iso_now()`` — single source of truth for the
        write-side timestamp format (``'…+00:00'``). Pre-2026-05-24 this
        emitted ``'…Z'``; historical rows with that suffix remain valid
        (consumers normalize via ``.replace('Z', '+00:00')`` before
        parsing). See ``servers/clock.py:iso_now``.
        """
        return iso_now()

    def _generate_id(self, node_type: str = None) -> str:
        """Generate 8-char hex node ID. ~4.3B combinations, collision-free at brain scale."""
        return uuid.uuid4().hex[:8]

    # ─── Session State ───
    # Per-session state lives on SessionContext, persisted via session_state.
    # See reset_session_activity — session-keyed.

    def session_context_for(self, session_id: str) -> str:
        """Per-session running journey summary (encoder's session arc).

        2026-05-02 (Frame Phase 2.5): replaces the global `session_context`
        property. The previous shape (single brain_meta key, leaked across
        parallel sessions — last-writer-wins, two Claude Code instances
        scrambled each other's arc) is gone. Each session writes/reads
        its own `session_context_{session_id}` key.

        Cross-session continuity (e.g., boot pulling the previous session's
        context) becomes a deliberate query — not an accidental side-effect
        of a leaky global. See docs/FRAME-DESIGN.md Phase 3 work.

        Returns empty string if no context for this session yet.
        """
        if not session_id:
            return ''
        return self.get_config('session_context_' + session_id, '') or ''

    def detect_git_env(self, cwd: str):
        """Branch + worktree + project for a session cwd — thin delegate to the
        host-adapter layer (session_env.detect_session_env). The brain RECEIVES
        session identity; deriving it from the host (git, marker files, cwd)
        lives in servers/session_env.py so a different host swaps that module,
        not Brain. Returns (branch, worktree, project); None fields mean
        'detection failed — keep what we have' (set_env's three-state)."""
        from .session_env import detect_session_env
        return detect_session_env(cwd, log=self._log_error)

    @property
    def scopes(self):
        """Scope policy (servers/scopes.py) — the operator's separation
        contract per scope dimension. Fresh parse of the 'scopes'
        interaction config; enforcement callers use scope_veil (cached),
        this property is for introspection/config paths."""
        from .scopes import load_scope_policy
        return load_scope_policy(self)

    def scope_veil(self, session_id: str) -> frozenset:
        """The hidden-set for this session — node ids an active isolation
        config walls off (servers/scopes.py docstring for semantics). THE
        enforcement object: consumers do one set-membership check, never
        per-candidate policy evaluation.

        Self-invalidating cache, no TTL: keyed on (active 'scopes' config
        version, MetadataDAL.change_probe(), the session's scope signature)
        — a config flip or a newly stamped node invalidates on the next
        read. change_probe is MAX(rowid) (O(1)), NOT change_key's COUNT(*)
        full scan — this probe runs on every gate touch of the recall hot
        path. Sessions with the same scope share an entry. Rebuild failure
        keeps the last good veil and logs CRITICAL (never silently
        un-walls); a first-build failure raises — a loud dead recall beats
        a silent leak.

        Sessionless callers ('' / unknown session) get the OUTWARD-ONLY
        veil (empty scope signature): every isolated value is hidden, no
        inward wall. Never substitute another session's veil (a borrowed
        INWARD veil is the complement of a wall, i.e. a leak)."""
        from .scopes import load_scope_policy, build_veil
        scope = self.session_scope(session_id) or {}
        sig = tuple(sorted(
            (k, (v or '').strip().lower()) for k, v in scope.items()))
        try:
            row = self._interaction_dal.get_active('scopes') or {}
            version = row.get('version', 0)
            change = self._meta_kv.change_probe()
        except Exception as e:
            self._log_error('scope_veil_probe', e,
                            'staleness probe failed — treating as changed')
            version, change = -1, None
        cache = getattr(self, '_scope_veils', None)
        if cache is None:
            cache = self._scope_veils = {}
        hit = cache.get(sig)
        if hit is not None and change is not None \
                and hit[0] == version and hit[1] == change:
            return hit[2]
        try:
            policy = load_scope_policy(self)
            veil = (build_veil(self, policy, scope)
                    if policy.has_isolation else frozenset())
        except Exception as e:
            if hit is not None:
                self._log_error(
                    'scope_veil_rebuild', e,
                    'CRITICAL: veil rebuild failed — serving last good '
                    'veil (%d ids); isolation may be stale' % len(hit[2]))
                return hit[2]
            raise
        cache[sig] = (version, change, veil)
        return veil

    def counterpart_for(self, session_id: str) -> str:
        """Who this session is with — the ONE site that answers it, so the
        speaker arc's F4 (counterpart on SessionContext) is a one-line change
        here instead of a sweep. Today: the install default (a constant);
        '' when unset. `session_id` is accepted-but-unused by design — the
        forward-compatibility seam, same shape as speaker_for."""
        from .daemon_config import get_operator_name
        return get_operator_name() or ''

    def _scope_value(self, dim: str, session_id: str) -> str:
        """Session-side value of one scope dimension ('' when none). The one
        dimension→resolver mapping both scope forms below share."""
        if dim == 'project':
            return (self.session_env_for(session_id) or {}).get('project', '')
        if dim == 'counterpart':
            return self.counterpart_for(session_id)
        return ''

    def session_scope(self, session_id: str):
        """RENDER form — the session's declared side of every scope
        dimension, threaded as plain data (never re-derived at depth) for
        differential exposure (contract.scope_marks). Only truthy dimensions
        are declared: an unscoped session applies no project pressure,
        matching the lane's unknown-is-neutral semantics. Returns None when
        nothing is declared (legacy render everywhere)."""
        from .contract import SCOPE_PROVENANCE_FIELDS
        scope = {}
        for dim in SCOPE_PROVENANCE_FIELDS:
            value = self._scope_value(dim, session_id)
            if value:
                scope[dim] = value
        return scope or None

    def scope_policy_for(self, session_id: str):
        """POLICY form — the same dimensions for stamp_scope_provenance,
        keeping '' (authoritative strip: this session has none, drop
        agent-supplied values). The falsy convention differs from
        session_scope on purpose and lives only here + there. None when the
        caller has no session (no authority — pass through)."""
        if not session_id:
            return None
        from .contract import SCOPE_PROVENANCE_FIELDS
        return {dim: self._scope_value(dim, session_id)
                for dim in SCOPE_PROVENANCE_FIELDS}

    def session_env_for(self, session_id: str) -> dict:
        """Per-session env (cwd, branch, worktree, project, model, host) for a
        stream — fed in from the host side (boot hook; per-turn recall/Stop hooks
        for model/host), surfaced in peek so streams identify where each other
        work and what they ride on. Reads the live cached SessionContext if
        present, else the persisted row. Empty strings when unknown. Mirrors
        session_context_for's per-session pattern (no global key — parallel
        sessions don't clobber)."""
        _empty = {'cwd': '', 'branch': '', 'worktree': '', 'project': '',
                  'model': '', 'host': ''}
        if not session_id:
            return _empty
        ctx = self._session_contexts.get(session_id)
        if ctx is None:
            from .session_context import SessionContext, SessionContextCorrupt
            try:
                ctx = SessionContext.load(self._session_state, session_id)
            except SessionContextCorrupt as e:
                self._log_error('session_context_load', e,
                                'session_env_for session=%s' % (session_id or '')[:8])
                ctx = None
        if ctx is None:
            return _empty
        return {'cwd': ctx.cwd, 'branch': ctx.branch, 'worktree': ctx.worktree,
                'project': ctx.project, 'model': ctx.model, 'host': ctx.host}

    def _resolve_interaction(self, name: str):
        """Resolve `name` to its effective K: (template, config, row, shadows).

        The single resolution seam of the override model: the code default
        from INTERACTION_DEFAULTS is the base; the active DB row (when
        present) overlays it — template as a whole when non-empty, config
        key-level (partial overrides are the normal case: one knob, not a
        snapshot). `shadows` is True only when the row actually contributed
        something. Guards, in order:
        - unknown name → KeyError. An unregistered boundary would otherwise
          run on an empty prompt with no signal; the registry completeness
          test (tests/test_interaction_defaults.py) is what makes raising
          safe.
        - no row → the code default, silently (the normal state once
          overrides collapse).
        - unparseable parameters JSON → _log_error + code default. A typo'd
          override must be distinguishable from "no override" — swallowing
          it would silently revert the boundary.
        - validator violations (INTERACTION_VALIDATORS) → _log_error + code
          default. Same registry the write door uses to refuse; read-time
          it degrades loudly instead of raising.
        """
        from .interaction_defaults import (
            INTERACTION_DEFAULTS, INTERACTION_VALIDATORS)
        if name not in INTERACTION_DEFAULTS:
            raise KeyError(
                'unknown interaction %r — no code default registered; '
                'add it to servers/interaction_defaults.py' % name)
        default_template, default_config = INTERACTION_DEFAULTS[name]
        row = self._interaction_dal.get_active(name)
        if not row:
            return default_template, dict(default_config), None, False
        override = {}
        params = row.get('parameters')
        if params:
            try:
                parsed = json.loads(params)
                if not isinstance(parsed, dict):
                    raise TypeError('parameters JSON is %s, not an object'
                                    % type(parsed).__name__)
                override = parsed
            except (json.JSONDecodeError, TypeError) as e:
                self._log_error(
                    'interaction_resolve', e,
                    '%s: unparseable override parameters — running on the '
                    'code default' % name)
        if override:
            validator = INTERACTION_VALIDATORS.get(name)
            if validator:
                violations = validator(override)
                if violations:
                    self._log_error(
                        'interaction_resolve',
                        ValueError('; '.join(violations)),
                        '%s: invalid override config — running on the code '
                        'default' % name)
                    override = {}
        template = row.get('template') or default_template
        shadows = bool(row.get('template')) or bool(override)
        return template, {**default_config, **override}, row, shadows

    def get_interaction_config(self, name: str) -> dict:
        """Effective config for an interaction: the code default with the
        active DB override (if any) overlaid key-level. Total by
        construction — readers subscript; caller-side fallbacks are the
        a6dfcfe3 trap. Raises KeyError for a name with no code default.

        Registering a new version does NOT change what this returns —
        call set_interaction_active() to flip the runtime to a new version.
        """
        return self._resolve_interaction(name)[1]

    def get_interaction_prompt(self, name: str) -> str:
        """Effective prompt for an LLM interaction: the active DB row's
        template when non-empty, else the code default. Raises KeyError
        for a name with no code default. See get_interaction_config() for
        the activation model.
        """
        return self._resolve_interaction(name)[0]

    def get_interaction_stamp(self, name: str) -> dict:
        """K-provenance stamp for the EFFECTIVE prompt+config behind `name`.

        Returns {'fingerprint', 'source', 'version', 'id'} — the block trace
        writers put on delta/selection metadata (fingerprint + source +
        version) and the trace row (id). `fingerprint` content-addresses the
        RESOLVED (overlaid) value a run of `name` actually uses, so it stays
        comparable across installs and unchanged when an override row is
        collapsed into a byte-identical code default. `source` is 'override'
        only when the active row actually shadows the default (non-empty
        template or ≥1 config key); a vacuous row stamps 'default'
        (version 0, id None) like no row at all.
        """
        return self.get_interaction_effective(name)['stamp']

    def get_interaction_effective(self, name: str, overlay: dict = None) -> dict:
        """Template, config and stamp from ONE resolution — mutually
        consistent by construction. Three separate accessor calls can
        straddle a concurrent set_interaction_active and return a template
        from the old row with a config from the new; this is the door for
        callers that need the whole effective value.

        `overlay` is a caller's explicit config (evals), laid key-level over
        the resolved config the way an override lies over the default. When
        it changes the effective value, the stamp fingerprints what runs and
        `source` is 'caller' — no registered version holds that value, so
        `version`/`id` still name the row it was laid over."""
        from .interaction_defaults import interaction_fingerprint
        template, config, row, shadows = self._resolve_interaction(name)
        stamp = {'fingerprint': interaction_fingerprint(name, template, config),
                 'source': 'default', 'version': 0, 'id': None}
        if shadows:
            stamp.update(source='override',
                         version=int(row.get('version') or 0),
                         id=row.get('id'))
        if overlay:
            laid = {**config, **overlay}
            fingerprint = interaction_fingerprint(name, template, laid)
            if fingerprint != stamp['fingerprint']:
                config = laid
                stamp.update(fingerprint=fingerprint, source='caller')
        return {'name': name, 'template': template, 'config': config,
                'stamp': stamp}

    def get_interaction(self, name: str, version: int = 0) -> dict:
        """Get interaction by name. version=0 (default) returns active, else specific version."""
        if version:
            return self._interaction_dal.get_version(name, version)
        return self._interaction_dal.get_active(name)

    def set_interaction_active(self, name: str, version: int,
                                set_by: str = 'anchor') -> dict:
        """Flip the active version pointer for `name`. Runtime picks up
        the new active version on the next read of get_interaction_prompt
        or get_interaction_config. See InteractionDAL.set_active.
        """
        result = self._interaction_dal.set_active(name, version, set_by)
        self.invalidate_interaction_caches(name)
        return result

    def clear_interaction_override(self, name: str) -> dict:
        """Delete the active pointer for `name` — revert to the code default.

        The inverse of set_interaction_active: "no pointer" means "no
        override deployed", so the resolver serves the code default on the
        next read. Registered versions stay on record for re-activation.
        `cleared` is False when no pointer existed (already on the default).

        A name that deleted nothing AND has no code default is refused: a
        typo'd clear ('trace_recoding') reporting "already on the default"
        would leave the real override live behind a success message. Legacy
        names without a code default still clear fine when their pointer
        exists — the delete-first order is what allows that cleanup.
        """
        cleared = self._interaction_dal.clear_active(name)
        if not cleared:
            from .interaction_defaults import INTERACTION_DEFAULTS
            if name not in INTERACTION_DEFAULTS:
                raise KeyError(
                    'unknown interaction %r — no pointer to clear and no '
                    'code default registered; check the name' % name)
        self.invalidate_interaction_caches(name)
        return {'name': name, 'cleared': cleared}

    def invalidate_interaction_caches(self, name: str) -> None:
        """Drop every cache holding `name`'s resolved value, so a pointer
        flip or clear reaches the next read immediately — not after a TTL.

        The next-read promise is what makes clear-then-measure workflows
        (eval overrides, the trace_recording debug switch) trustworthy.

        _INTERACTION_CACHE_INVALIDATORS below is the registry of every cache
        keyed on an interaction name — a new TTL cache over a resolved
        interaction MUST add its name here or a deploy/clear runs late for
        that reader, silently. (scope_veil needs no entry: its cache key
        includes the scopes pointer version, so a flip or clear misses the
        key and rebuilds naturally.)
        """
        for invalidate in self._INTERACTION_CACHE_INVALIDATORS.get(name, ()):
            invalidate(self)

    _INTERACTION_CACHE_INVALIDATORS = {
        # The payload recorder TTL-caches this config (performance charter) —
        # the one config that gates live capture.
        'trace_recording': (
            lambda self: self.invalidate_trace_recording_cache(),
        ),
        # The LAF engine TTL-caches its gain overlay; the recall RESULT cache
        # keys on the query alone (no config fingerprint), so it must purge
        # too or an identical query re-asked within its TTL returns the
        # pre-flip result.
        'recall_laf': (
            lambda self: (getattr(self, '_laf_engine', None) is not None
                          and self._laf_engine.invalidate_config()),
            lambda self: self._recall_cache_purge(),
        ),
    }

    def register_interaction(self, name: str, template: str = '',
                             parameters: str = '',
                             created_by: str = 'anchor') -> dict:
        """Register a new version of an interaction (prompt + config).

        Never activates — a write is not a deployment decision; every name
        runs on its code default until set_interaction_active deploys an
        override. Completes the interaction-registry door: callers never
        touch _interaction_dal directly.

        Configs with a registered validator (INTERACTION_VALIDATORS) are
        validated AT THIS DOOR and refused on violations — for scopes, a
        typo'd mode silently meaning "less isolation than configured" is the
        one failure a separation contract must not defer to read-time
        logging.
        """
        from .interaction_defaults import INTERACTION_VALIDATORS
        validator = INTERACTION_VALIDATORS.get(name)
        if validator:
            try:
                config = json.loads(parameters) if parameters else {}
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError('%s config is not valid JSON: %s' % (name, e))
            violations = validator(config)
            if violations:
                raise ValueError(
                    '%s config refused: %s' % (name, '; '.join(violations)))
        return self._interaction_dal.register(
            name=name, template=template, parameters=parameters,
            created_by=created_by)

    def get_or_create_session(self, session_id: str) -> 'SessionContext':
        """Get or create a SessionContext for a given session_id.

        Single entry point for session state. Hooks send session_id from
        Claude Code args. The brain holds the state. An empty session_id
        raises: session state belongs to one stream, and there is no other
        stream's state to substitute for it.

        In-memory cache (2026-05-17): SessionContext instances are cached
        on `self._session_contexts` and returned by reference. Mutations
        to fatigue / counters live in memory across hooks in the same
        session; `save_session_contexts()` persists them — every
        `SESSION_FLUSH_INTERVAL_SECONDS` from the daemon's session flush,
        and once more in close(). The cache entry is cleared on SessionEnd
        hook (clean shutdown of that session) — see
        daemon_hooks.hook_session_end.

        Race-safe on first load: two threads with the same brand-new
        session_id can call concurrently. The create path takes `write_lock`
        to guard the `_session_contexts` cache (check-then-create must be
        atomic or two threads mint two instances); the DB write itself is
        serialized inside the DAL write boundary (logs_write_lock + wconn),
        and `INSERT OR IGNORE` keeps the row idempotent. After first load,
        both threads operate on the same cached instance (Python attribute
        access — `write_lock` serializes actual mutations).
        """
        from .session_context import SessionContext
        if not session_id:
            raise ValueError('get_or_create_session needs a session_id')
        # Fast path: already cached (lock-free — the common case).
        cached = self._session_contexts.get(session_id)
        if cached is not None:
            return cached
        # First touch: serialize the create under write_lock so the cache
        # check-then-create is atomic (two racing threads must not mint two
        # SessionContext instances). The DB write is serialized separately by
        # the DAL write boundary (logs_write_lock + logs_conn_w).
        with self.write_lock:
            cached = self._session_contexts.get(session_id)  # re-check under lock
            if cached is not None:
                return cached
            default_data = json.dumps({'stop_counter': 0, 'fatigue': {}})
            self._session_state.ensure_default(session_id, '_session_context', default_data)
            # Row is guaranteed to exist now — load reads our default or a
            # racing thread's already-modified state.
            from .session_context import SessionContextCorrupt
            try:
                ctx = SessionContext.load(self._session_state, session_id)
            except SessionContextCorrupt as e:
                self._log_error('session_context_load', e,
                                'get_or_create_session session=%s' % (session_id or '')[:8])
                ctx = None
            if ctx is None:
                ctx = SessionContext(session_id=session_id)
            self._session_contexts[session_id] = ctx
            return ctx

    def save_session_contexts(self) -> int:
        """Persist the cached SessionContexts that changed since their last
        write to session_state. Returns the count of rows written.

        Called by the daemon's session flush every
        SESSION_FLUSH_INTERVAL_SECONDS and by close(). Mutations (fatigue,
        counters, the turn-classification stamps) accumulate in memory between
        flushes; this is the persistence boundary. An unchanged session costs
        one serialize and a compare, and its row (updated_at included) is left
        alone.
        """
        n = 0
        # write_lock guards the _session_contexts cache iteration against a
        # concurrent create/discard; the DB writes themselves serialize inside
        # the DAL write boundary (logs_write_lock + logs_conn_w).
        with self.write_lock:
            for ctx in list(self._session_contexts.values()):
                try:
                    if ctx.save(self._session_state, if_changed=True):
                        n += 1
                except Exception as _e:
                    try:
                        self._log_error('session_context_autosave', _e,
                                        'persisting cached SessionContext')
                    except Exception:
                        pass
        return n

    def scribe_due(self, now: Optional[float] = None,
                   skip_sessions=None,
                   flush: bool = False) -> Optional[Dict[str, Any]]:
        """Decide whether any active session's S1 Scribe is due to encode — the
        poll-driven cadence trigger (mid-session ENCODE_EVERY turns, or the idle
        tail). The daemon polls this every few seconds; it owns spawning + the
        single-flight lock + the per-session retry cooldown, this method only
        DECIDES.

        Reads only higher session functions — `present_streams` (who's awake +
        each one's last-turn time), `turns_since_last_encode` (per-session
        cadence count) and `get_conversation` (is the newest exchange
        complete?) — never SQL/DAL directly. Wall-clock is correct here
        (presence + idle are real-time "is the operator away", like
        run_maintenance_if_due), so it's exempt from the conversation_now rule.

        `skip_sessions` (set) — sessions the daemon is cooling down after a recent
        attempt; excluded from selection so a session whose encode keeps failing
        (and thus stays "due") can't monopolize the poll and starve the others.

        `flush` — a scribe_flush is asking. On an eval brain (eval_brain_since)
        that is the only way an idle tail comes due, and sessions last active
        before the mark are never due at all.

        Returns the MOST-OVERDUE due session's {'session_id', 'counter'} so a
        multi-session backlog drains one-per-poll (most-behind first); None when
        nothing is due.
        """
        # Keyless onboarding window: encoding is Sonnet-driven — nothing is due
        # until a key resolves. Gate here (the DECIDE step) so the daemon poll
        # never spawns an encode destined to 401.
        if not self.llm_available:
            self.note_llm_unavailable('S1 Scribe')
            return None
        import time as _time
        from datetime import datetime as _datetime
        from .scales.s1.encode_contract import (
            ENCODE_EVERY, SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS,
            SCRIBE_CANDIDATE_WINDOW_MIN, SCRIBE_ACTIVE_WINDOW_SECONDS,
            scribe_is_starved)
        from .brain_constants import MAINTENANCE_BOOT_GRACE_SECONDS

        now = now if now is not None else _time.time()
        # Boot-grace: don't sweep/encode during the daemon's warmup after a
        # (re)start — let it settle, and don't flush backlogs the instant it
        # comes up. Same settle window the S2 maintenance gate uses.
        if now - getattr(self, '_boot_time', now) < MAINTENANCE_BOOT_GRACE_SECONDS:
            return None
        skip = skip_sessions or ()
        eval_since = self.eval_brain_since()
        best = None  # (turns, session_id, counter) — highest turns = most overdue

        for stream in self.present_streams(
                window_min=SCRIBE_CANDIDATE_WINDOW_MIN, limit=50):
            sid = stream.get('session_id', '')
            if not sid or sid in skip:
                continue
            try:
                last_turn = _datetime.fromisoformat(
                    stream.get('live_recency', '')).timestamp()
            except (ValueError, TypeError):
                # An eval brain can't tell whose this session is, so it isn't
                # the eval's to pay for.
                if eval_since is not None:
                    continue
                last_turn = now
            # An eval brain encodes only its own conversation. A session last
            # active before the mark (the operator's live tails in a copy, a
            # previous eval's leftovers) is not the eval's to pay for: it never
            # comes due and never raises the starvation alarm.
            if eval_since is not None and last_turn <= eval_since:
                continue
            turns = self.turns_since_last_encode(sid)
            if turns <= 0:
                continue

            # Loud signal if a session is wedged (turns kept climbing past the
            # cadence — encoder erroring or never firing). Preserved from the
            # old hook gate; rate-limited inside scribe_is_starved.
            if scribe_is_starved(turns):
                try:
                    self._log_error(
                        'scribe_starvation',
                        RuntimeError('%d conversational turns since last encode '
                                     '— Scribe not completing runs' % turns),
                        'session=%s' % sid)
                except Exception:
                    pass

            # Idle (wall-clock) since this session's last turn gates BOTH
            # clauses: 5+ fires only while ACTIVELY conversing (recent turn);
            # the tail only once the session has gone quiet. A session in
            # between waits for the tail — or re-qualifies for 5+ on its next
            # turn. This is what stops a restart from sweeping every recent
            # session's backlog at once.
            idle = now - last_turn
            five_plus = turns >= ENCODE_EVERY and idle < SCRIBE_ACTIVE_WINDOW_SECONDS
            if five_plus:
                # The turn count crosses the threshold ON the user prompt (a
                # turn == one user_message), so an immediate fire snapshots a
                # window ending on an unanswered question — the answer's Stop
                # trace lands seconds later. Wait for the exchange to complete;
                # the next poll fires it. The tail is exempt: a question still
                # dangling once the session went quiet is genuinely unanswered
                # (interrupt/disconnect) and belongs in the encode as-is.
                last = self.get_conversation(sid, limit=1,
                                             with_judge_output=False)
                # ref_type, not role: role maps every incoming correspondent
                # to 'user', so a dial-on delivery row would read as an
                # unanswered question and stall the trigger (a boot delivery
                # never gets a reaction).
                five_plus = bool(last) and (
                    last[0].get('ref_type') == 'assistant_message')
            # On an eval brain time alone starts no encode: its idle tail
            # waits for scribe_flush.
            tail = (turns > SCRIBE_TAIL_MIN_TURNS and idle > SCRIBE_TAIL_IDLE_SECONDS
                    and (flush or eval_since is None))
            if not (five_plus or tail):
                continue

            if best is None or turns > best[0]:
                ctx = self.get_or_create_session(sid)
                best = (turns, sid, ctx.stop_counter)

        if best:
            return {'session_id': best[1], 'counter': best[2]}
        return None

    def stamp_boot_liveness(self, session_id: str) -> None:
        """Write ONE S0 heartbeat trace at boot so a freshly-booted stream is
        visible in presence IMMEDIATELY — before it takes its first turn.

        present_streams reads real-turn S0 traces, so without a boot trace a
        stream is invisible until its first hook_recall/Stop — which is exactly
        why two just-booted streams can't find each other at boot (the
        rendezvous gap, 2026-06-06). 'heartbeat' is already counted by
        active_sessions_by_turn, so the read side is unchanged. Loud on failure,
        never blocks boot."""
        if not session_id:
            return
        try:
            ctx = self.get_or_create_session(session_id)
            self._trace_dal.append(
                chain_id=ctx.s0_chain(), scale='s0', event_type='K',
                ref_type='heartbeat', session_id=session_id,
                summary='boot — stream online')
        except Exception as e:
            try:
                self._log_error('stamp_boot_liveness', e,
                                'session=%s' % (session_id or '')[:8])
            except Exception:
                pass

    def discard_session_context(self, session_id: str) -> None:
        """Save + drop a session's cached SessionContext. Called from
        SessionEnd hook for clean shutdown of that session.
        """
        ctx = self._session_contexts.pop(session_id, None)
        if ctx is not None:
            try:
                with self.write_lock:  # guard the cache entry vs autosave
                    ctx.save(self._session_state)
            except Exception as _e:
                try:
                    self._log_error('session_context_discard', _e,
                                    'final save before dropping cache entry')
                except Exception:
                    pass

    def reset_session_activity(self, session_id: str = '', cwd: str = '') -> bool:
        """Boot a session. Resume-aware: CONTINUES an existing session's counters
        instead of zeroing them. Returns True on resume, False for a new session.

        New-vs-resume is decided from the SESSION OBJECT itself (ctx.boot_time —
        empty means never booted), NOT a global brain_meta flag. The old design
        let the boot hook gate resume-detection on the global `last_booted_session`
        key, which is last-writer-wins across parallel sessions: with concurrent
        streams every session but the most-recent booter failed the guard, so its
        re-boot (resume / host-wake / compaction) fell through to a full reset.
        That reset wiped the session's accumulated state — back then the S1
        Scribe's cadence was a stored counter, so it got zeroed and the Scribe
        starved. The cadence is trace-derived now (turns_since_last_encode), but
        preserving the rest still matters: a reset would reset stop_counter (→
        duplicate chain IDs), fatigue, and node_activity.

        An empty session_id raises, as in get_or_create_session: a boot that
        names no session has no stream to boot.
        """
        if not session_id:
            raise ValueError('reset_session_activity needs a session_id')
        sid = session_id
        from .session_context import SessionContext, SessionContextCorrupt
        # Derive branch + worktree + project in ONE git call BEFORE the lock
        # (don't hold write_lock across a subprocess). detect_git_env returns
        # None fields on git failure, so a transient hiccup on resume can't
        # wipe a known worktree/project.
        branch, worktree, project = (self.detect_git_env(cwd) if cwd
                                     else ('', None, None))
        # The whole read-decide-write must be atomic under write_lock — same
        # double-checked-locking discipline as get_or_create_session. Doing the
        # resume read outside the lock lets two concurrent boots of the SAME
        # session_id both observe `None`, both build a fresh ctx, and clobber the
        # accumulated state. write_lock is reentrant (TrackedRLock), so the nested
        # SessionContext.load / _log_error / save are safe.
        with self.write_lock:
            # Resume detection from the session object. Prefer the live cached ctx;
            # fall back to the persisted row (a daemon restart empties the cache
            # but the row survives, so a post-restart re-boot is still a resume).
            existing = self._session_contexts.get(sid)
            if existing is None:
                try:
                    existing = SessionContext.load(self._session_state, sid)
                except SessionContextCorrupt as e:
                    self._log_error('session_context_load', e,
                                    'reset_session_activity session=%s' % (sid or '')[:8])
                    existing = None
            is_resume = existing is not None and bool(existing.boot_time)
            if is_resume:
                # RESUME — keep the accumulated state (stop counter, fatigue,
                # node activity). Only per-boot facts get refreshed below.
                ctx = existing
            else:
                # NEW session — fresh state (the SessionContext defaults).
                ctx = SessionContext(session_id=sid)
            ctx.boot_time = self.now()
            # A delivery continuation cannot survive a boot/resume — the
            # blocked turn's process is gone (same per-boot reasoning as
            # fatigue). Disarm so a resumed session's first stop can never
            # classify as a days-old delivery's reaction.
            ctx.last_delivery_stop = -1
            ctx.last_delivery_armed_at = ''
            # cwd/branch/worktree are session IDENTITY (where this stream works),
            # fed in from the boot hook and stamped through the session object's
            # single env mutator. Surfaced via session_env_for / peek.
            if cwd:
                ctx.set_env(cwd=cwd, branch=branch, worktree=worktree,
                            project=project)
            ctx.save(self._session_state)
            self._session_contexts[sid] = ctx
        return is_resume

    # ─── Utilities ───

    def _get_node_count(self) -> int:
        """Get count of non-archived nodes."""
        return self._nodes.count()

    # ─── EMBEDDER CONFIG: Model-agnostic configuration ───

    # Default embedder config — used when brain_meta has no overrides. The
    # model and its pinned_revision / pinned_onnx_sha256 come from
    # embedder.EMBEDDER_PIN ([tool.brain.embedder] in pyproject.toml).
    # embedder.load_model verifies the LOADED snapshot against both pins and
    # refuses to serve a different artifact — a silently refreshed model would
    # split the vector space in two (operator ruling 2026-08-07: "Can we not
    # auto update embedders?"). Changing the model or accepting an upstream
    # refresh = update the pin deliberately, then re-embed. A brain can also
    # override per key via set_embedder_config() (takes effect on next boot).
    _EMBEDDER_DEFAULTS = {
        **embedder.EMBEDDER_PIN,
        'cache_dir': None,   # None → durable default next to brain.db (below)
    }

    def _get_embedder_config(self) -> Dict[str, Any]:
        """
        Read embedder config from brain_meta, falling back to defaults.
        Config keys: embedder_model_name, embedder_dim, embedder_pooling,
                     embedder_model_file, embedder_model_path, embedder_cache_dir.

        Users can override any of these via set_config() to switch models
        without changing code.
        """
        config = {}
        for key, default in self._EMBEDDER_DEFAULTS.items():
            meta_key = f'embedder_{key}'
            val = self.get_config(meta_key, default)
            # Handle None stored as string
            if val == 'None' or val == 'null':
                val = None
            config[key] = val
        if not config.get('cache_dir') and self.db_path:
            # Durable default next to brain.db. fastembed's own default lives
            # in $TMPDIR, which macOS purges after ~3 days unused — every purge
            # re-downloaded the model from HF at whatever revision `main`
            # pointed to that day (observed 2026-08-07: cache dated Aug 3,
            # 137MB onnx re-pulled at boot).
            config['cache_dir'] = os.path.join(
                os.path.dirname(os.path.abspath(self.db_path)), 'fastembed_cache')
        return config

    def set_embedder_config(self, **kwargs) -> Dict[str, Any]:
        """
        Update embedder config in brain_meta. Takes effect on next boot.

        Example:
            brain.set_embedder_config(model_name="<hf-org>/<model>", dim=768)
        """
        updated = {}
        for key, value in kwargs.items():
            if key in self._EMBEDDER_DEFAULTS:
                meta_key = f'embedder_{key}'
                self.set_config(meta_key, value)
                updated[key] = value
        return {'updated': updated, 'takes_effect': 'next boot'}

    def set_config(self, key: str, value: Any) -> Dict[str, Any]:
        """Set a config value in brain_meta via DAL. Persists across restarts.

        Holds write_lock. set_config is a foreground self.conn write, and S2
        maintenance calls it lock-free on a pool thread (gating timestamps,
        failure counters, journals). Without the lock those writes interleave
        with a concurrent client brain_batch on the shared connection: the
        INSERT joins the batch's open transaction and BrainMetaDAL.set's
        commit_unless_batched then no-ops (in_batch=True), so the config write
        is lost if the batch rolls back. The lock serializes set_config against
        brain_batch (which resets in_batch before releasing the lock), so a
        non-owner thread never observes in_batch. write_lock is an RLock, so a
        caller already holding it (encoder dispatch, a batch) re-acquires safely.
        """
        with self.write_lock:
            self._meta.set(key, str(value))
        return {'key': key, 'value': value, 'updated_at': self.now()}

    # ══════════════════════════════════════════════════════════════════
    # Maintenance scheduling — brain owns the decision, daemon owns only
    # the polling cadence. State lives in brain_meta (s2_last_run_ts) so it
    # survives daemon restarts.
    # ══════════════════════════════════════════════════════════════════

    # Thresholds imported from brain_constants (contract-first; tunables
    # live next to other brain-wide config, not as class constants).

    def _maintenance_last_run_ts(self) -> float:
        """Epoch of the most recent maintenance run. Persisted in brain_meta."""
        raw = self.get_config('s2_last_run_ts') or '0'
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def _maintenance_set_last_run_ts(self, ts: float) -> None:
        self.set_config('s2_last_run_ts', str(ts))

    @property
    def s2_running(self) -> bool:
        """True while an S2 cycle is in flight. Cheap, non-consuming check.

        `run_maintenance_if_due` consults this BEFORE it stamps the last-run
        timestamp — see the note at that call.
        """
        return self._s2_lock.locked()

    def run_s2(self, units=None, force=False) -> Dict[str, Any]:
        """Run every S2 unit once, NOW. **THE single door to S2 activation.**

        `force` marks the cycle as requested rather than scheduled (s2_force,
        an eval's final flush); the coordinator hands it to every unit as
        `forced`, and a unit that batches work to save calls (the aspect
        classifier's batch gate) runs whatever is waiting.

        `units` optionally narrows the cycle to the named units (their
        `NAME`s), still in the coordinator's fixed order and still under the
        single-flight lock. None runs every unit. The selection is validated
        BEFORE the lock (coordinator.select_units raises ValueError), so a bad
        one fails the same way whether or not a cycle is live, and the result
        says whether it left any unit out — the scheduler-stamping caller
        (s2_force) reads that rather than re-deriving it from the argument.

        Every caller comes through here — the daemon's maintenance poll (via
        run_maintenance_if_due, which owns the "is it time?" policy), evals,
        benchmarks, and IsolatedBrain. Nothing calls
        `scales.s2.coordinator.run_s2` directly any more; this method owns the
        single-flight guarantee, and a caller that bypasses it forfeits that.

        POLICY vs EXECUTION — the split that makes one door possible:
          run_maintenance_if_due()  answers "SHOULD we?"  (idle, min interval,
                                    encode count, boot grace, force-fire)
          run_s2()                  answers "do it, exactly ONCE"

        A second concurrent call does not queue and does not block — it returns
        immediately with `skipped`. Two overlapping cycles are never wanted:
        consolidation is a multi-minute LLM run, and a second pass would
        duplicate its work and race its own cluster fingerprints. That was the
        2026-06 parallel-run bug (`node:daaf63a9`).

        Deliberately UNGATED: the five fire gates are the poll's scheduling
        policy, not a safety property. On a LIVE brain this therefore does real
        work — Sonnet calls and graph mutations — possibly while the operator is
        typing, which the idle gate exists to avoid. Calling it directly is a
        deliberate act; the poll should be left to its cadence.

        Scope note: `threading.Lock` is per-process, and that is sufficient —
        the daemon is a launchd singleton holding THE Brain, and every caller
        (poll thread, TCP dispatch, `eval`) shares it. A second OS process on
        the same `brain.db` is already forbidden for a stronger reason: two
        writer connections corrupt indexes.

        Returns:
            {'units': {unit_name: result}, 'elapsed_ms': int, 'partial': bool}
            when it ran, or {'units': {}, 'skipped': 'already running'} when a
            cycle was live.
        """
        from .scales.s2.coordinator import run_s2 as _run_units, select_units
        partial = len(select_units(units)) < len(select_units())
        if not self._s2_lock.acquire(blocking=False):
            return {'units': {}, 'skipped': 'already running'}
        try:
            import time as _time
            t0 = _time.time()
            results = _run_units(self, units=units, force=force)
            return {'units': results,
                    'elapsed_ms': int((_time.time() - t0) * 1000),
                    'partial': partial}
        finally:
            self._s2_lock.release()

    def run_logs_maintenance(self) -> Dict[str, Any]:
        """Log retention + graph orphan sweep. Safe against a live DB.

        The orphan arm runs on a PRIVATE maintenance connection, never
        `self.conn`. That is the whole safety property: WAL admits this
        writer alongside the foreground one, the short busy_timeout makes it
        yield rather than block, and a private transaction means its commit
        can never land on a `brain_batch` envelope the foreground is holding.
        Passing `self.conn` here would reintroduce exactly the stray-commit
        class that killed a savepoint mid-merge once already.

        Retention pruning is never gated — it reclaims space, and skipping it
        because the freshness gate fails on a full disk is a death spiral.
        The destructive orphan arm is gated on a restorable snapshot;
        `graph_conn=None` disables it for the cycle.
        """
        from .brain_constants import SESSION_LIVE_DAYS
        from .db_backup import ensure_backup_fresh
        # Session live time (brain_meta `session.live_days`), floored at 1 so a
        # bad override cannot prune live sessions. An unparseable value falls
        # back to the default (get_config logged it) rather than failing the
        # whole retention pass.
        try:
            live_days = max(1, int(self.get_config('session.live_days',
                                                   SESSION_LIVE_DAYS)))
        except (TypeError, ValueError):
            live_days = SESSION_LIVE_DAYS
        try:
            fresh = (ensure_backup_fresh(self.db_path)
                     and ensure_backup_fresh(self.logs_db_path))
        except Exception as e:
            fresh = False
            self._log_error('logs_maintenance_backup_gate', e,
                            'freshness check failed — orphan sweep skipped')
        if not fresh:
            return self._logs_dal.run_maintenance(graph_conn=None,
                                                  session_live_days=live_days)

        from . import db_backends
        conn = db_backends.current.connect_maintenance(self.db_path)
        try:
            return self._logs_dal.run_maintenance(graph_conn=conn,
                                                  session_live_days=live_days)
        finally:
            conn.close()

    def sweep_channels_if_due(self, now: Optional[float] = None) -> None:
        """Expiry sweeps for the two channel stores — the courier reap
        (self-channel dead messages) and the Thalamus window sweep (expired
        items; an unanswered ask dead-letters LOUDLY). Like the payload
        prune, these must NOT sit behind the S2 fire conditions: on a
        keyless brain (llm_available False) the S2 cycle never runs, and a
        gated sweep would mean the ask dead-letter never fires and expired
        items linger. Each sweep is behind its own cheap hourly in-memory
        throttle and failure-isolated (loud via _log_error)."""
        import time as _time
        now = now if now is not None else _time.time()

        # Imports stay inside each sweep: a broken channel module costs that
        # sweep alone, never the other one or the S2 cycle that follows.
        def reap_expired():
            from .channels.self_channel import signal
            return signal.reap_expired(self)

        def expire_due():
            from .channels.thalamus import thalamus
            return thalamus.expire_due(self)

        # (throttle stamp, sweep, log line, error tag) — one row per channel
        # store; the hourly throttle and the failure isolation are the policy.
        for stamp, sweep, line, tag in (
                ('_courier_reap_checked', reap_expired,
                 'self-channel: reaped %d expired message(s)', 'self_signal_reap'),
                ('_thalamus_sweep_checked', expire_due,
                 'thalamus: expired %d item(s) past their window', 'thalamus_expire')):
            if now - getattr(self, stamp, 0) < 3_600:
                continue
            setattr(self, stamp, now)
            try:
                n = sweep()
                if n:
                    print('[brain] %s' % (line % n), flush=True)
            except Exception as e:
                self._log_error(tag, e, '%s in channel sweep' % sweep.__name__)

    def run_maintenance_if_due(self, now: Optional[float] = None
                               ) -> Optional[Dict[str, Any]]:
        """Run S2 maintenance iff idle, min-interval, and activity conditions are met.

        Reads the gating signals from `self.activity` (ActivityState) — the
        single source of truth the daemon/hooks keep updated:
          - last_user_activity → idle gate (operator not typing)
          - encode_runs_since_maintenance → activity gate (new encoded material)

        S2's material is encoded nodes, so the activity gate counts S1 Encoder
        (Scribe) runs, NOT surfaces/recalls — a recall reads the graph and
        creates nothing for S2 to consolidate. The FORCE_FIRE valve overrides
        both normal-trigger gates so a stale graph still gets maintenance.

        Args:
            now: Epoch seconds (default: time.time()). Injectable for tests.

        This is the SINGLE production activation path for S2 — it owns the
        POLICY ("is it time?"). Execution and single-flight belong to
        `run_s2()`, the one door every caller shares. No other caller triggers
        *scheduled* maintenance; the daemon poll owns this method.

        Returns: S2 coordinator results dict when a run fired, None when
            not due. Caller (daemon) is responsible for serializing calls;
            this method is safe to call frequently — it no-ops cheaply when
            not due.
        """
        # Payload-file retention self-gates (in-memory hourly throttle +
        # daily brain_meta stamp) and must NOT sit behind the S2 fire
        # conditions below — a keyless brain (llm_available False) or one
        # whose idle/encode gates rarely fire still needs its payloads/
        # pruned, or debug-mode capture grows unbounded.
        self.prune_payloads_if_due()
        # Channel expiry sweeps self-gate the same way and for the same
        # reason — a keyless brain still needs its dead-letters.
        self.sweep_channels_if_due(now)

        # Keyless onboarding window: every S2 unit's encoder is LLM-driven —
        # skip the whole cycle until a key resolves (noted once, not per poll).
        if not self.llm_available:
            self.note_llm_unavailable('S2 maintenance')
            return None

        # An eval brain never schedules S2 itself; the eval runs it with
        # s2_force, which does not pass through this gate.
        if self.eval_brain_since() is not None:
            return None

        # A cycle is already in flight — bail BEFORE the gates, and crucially
        # before the stamp + encode-run consumption below. run_s2() would skip
        # anyway, but by then this method would have already advanced the
        # last-run timestamp and eaten the encode runs it gated on: a cycle
        # burned that never ran, and the next one starved.
        if self.s2_running:
            return None

        import time as _time
        from .brain_constants import (
            MAINTENANCE_IDLE_THRESHOLD_SECONDS,
            MAINTENANCE_MIN_INTERVAL_SECONDS,
            MAINTENANCE_MIN_ENCODE_RUNS,
            MAINTENANCE_FORCE_FIRE_SECONDS,
            MAINTENANCE_BOOT_GRACE_SECONDS,
        )
        now = now if now is not None else _time.time()

        # Boot grace gate (2026-05-08): never fire maintenance for the first
        # N seconds after daemon start, so the first user recall isn't blocked
        # behind a long consolidation cycle.
        boot_age = now - getattr(self, '_boot_time', now)
        if boot_age < MAINTENANCE_BOOT_GRACE_SECONDS:
            return None

        last_user_activity = self.activity.last_user_activity
        encode_runs = self.activity.encode_runs_since_maintenance

        # last_user_activity == 0.0 means "daemon just booted, no user prompts
        # yet" — treat as infinitely idle so S2 can fire (subject to the gates).
        if not last_user_activity:
            idle_seconds = float('inf')
        else:
            idle_seconds = now - last_user_activity
        last_run_ts = self._maintenance_last_run_ts()
        since_last_run = now - last_run_ts if last_run_ts else float('inf')

        # Min-interval gate is absolute — never fire more often than this.
        if since_last_run < MAINTENANCE_MIN_INTERVAL_SECONDS:
            return None
        # Normal-trigger gates: idle (operator not typing) AND enough new
        # encoded material (Scribe runs) since the last run. A stale-S2 safety
        # valve overrides BOTH: if maintenance hasn't fired in
        # FORCE_FIRE_SECONDS the graph is going stale regardless, so fire.
        if since_last_run < MAINTENANCE_FORCE_FIRE_SECONDS:
            if idle_seconds < MAINTENANCE_IDLE_THRESHOLD_SECONDS:
                return None
            if encode_runs < MAINTENANCE_MIN_ENCODE_RUNS:
                return None

        # Mark the run BEFORE executing so concurrent callers (via a second
        # poll) see the same timestamp and skip. The daemon's
        # _maintenance_poll_running flag is NOT a second layer of this — it
        # throttles duplicate pool SUBMISSIONS and is not mutual exclusion;
        # self._s2_lock (taken by run_s2) is what makes the cycle single-flight.
        self.mark_s2_ran(now, encode_runs)

        # Execution goes through the one door, which owns single-flight.
        outcome = self.run_s2()
        return {
            'ran_at_epoch': now,
            'idle_seconds': idle_seconds,
            'units': outcome.get('units', {}),
        }


    def mark_s2_ran(self, now: float, encode_runs: int) -> None:
        """Record that an S2 cycle ran: stamp the scheduler clock and consume
        the encode runs that cycle covered.

        Both activation paths write this pair, and both must write it the same
        way — the maintenance poll (before delegating, so a concurrent poll
        sees the stamp and skips) and the forced `s2_force` dispatch (after
        completion, since during the run the poll bails on `s2_running`). The
        timing differs; the bookkeeping must not.

        `encode_runs` MUST be the count snapshotted BEFORE the cycle.
        `consume_encode_runs` subtracts rather than zeroes precisely so runs
        completing during the multi-minute cycle accrue toward the next one —
        passing a count re-read afterwards subtracts everything and silently
        drops them, starving the next cycle's gate. That is the bug this method
        exists to make unrepeatable (dispatch_ops had its own copy and drifted
        into exactly it).
        """
        self._maintenance_set_last_run_ts(now)
        self.activity.consume_encode_runs(encode_runs)

    def get_debug_status(self) -> Dict[str, Any]:
        """Check if debug mode is enabled via DAL, falls back to env var."""
        try:
            val = self._meta.get('debug_enabled', '')
            if val:
                return {'debug_enabled': val == '1'}
        except Exception:
            pass
        return {'debug_enabled': os.environ.get('BRAIN_DEBUG') == '1'}

    def _log_event(self, event_type: str, source: str, *,
                   metadata: Dict[str, Any], file_level: str, file_text: str,
                   file_detail: str = '', fingerprint: Optional[str] = None,
                   ctx=None, session_id: Optional[str] = None):
        """Shared policy + write path for a debug_log row.

        Owns the cross-cutting policy that `_log_error` / `_log_warning`
        share: optional rate-limiting (skipped when `fingerprint is None`),
        the logs-db size check, per-session attribution, the human-readable
        file-log mirror, and the stderr last-resort fallback. The SQL lives
        in the DAL (`self._logs_dal.write_event`). The public loggers build
        their payload and call here — one INSERT path, so a per-type footgun
        (e.g. an error=None traceback) can't silently skip the write.

        Parallel-session attribution: hot-path callers (hook_recall, S1
        encode, MCP dispatch) pass `ctx` for correct session attribution;
        a row logged without one is sessionless (''). A raw event may supply
        `session_id` directly, including an empty string for a sessionless
        event; this overrides either context source.
        """
        try:
            if fingerprint is not None and self._check_rate_limit(fingerprint):
                return  # suppressed
            _sid = session_id if session_id is not None else (
                ctx.session_id if ctx is not None else '')
            self._check_logs_db_size()
            self._logs_dal.write_event(event_type, source, metadata, session_id=_sid)
            self._write_to_file_log(file_level, source, file_text, file_detail)
        except Exception:
            # Last resort — can't even log. Print to stderr.
            print('brain: %s in %s: %s' % (event_type, source, file_text),
                  file=sys.stderr)

    def loud(self, source: str, context: str = ''):
        """Context manager: run a stage loudly — any exception is logged to the
        errors table (via _log_error, which never raises) and SWALLOWED so the
        caller's next stage still runs. The one primitive for "this step may
        fail but must not take the rest down, and must never fail silently":

            with brain.loud('s1e_postprocess', 'journal/arc write'):
                ...

        For a failure that must ABORT (log + stop), just let it raise and log
        at the catch site — this helper is only for the continue case.
        """
        from contextlib import contextmanager

        @contextmanager
        def _stage():
            try:
                yield
            except Exception as e:
                self._log_error(source, e, context)
        return _stage()

    def _log_error(self, source: str, error: Optional[BaseException | str],
                   context: str = '',
                   ctx=None, *, session_id: Optional[str] = None):
        """Log an error to brain_logs.db + brain.log with rate limiting.
        NEVER raises — _log_event guards the whole write with a stderr
        last-resort, so callers need no try/except around this.

        Replaces silent `except: pass` blocks. Errors are stored in the logs
        DB and surfaced at boot. `error=None` or a message string is a valid
        call — callers log a *condition* (not an exception); only an
        exception carries a traceback, so the build is guarded and the write
        still proceeds.

        Not only a log: an exception the provider classifier reads as a
        refusal (a 401/403, or a quota 400/402/429) pauses EVERY LLM feature,
        process-wide, via _note_llm_failure. That is how the rejection latch
        engages without each LLM call site threading a classification. To
        record a provider refusal WITHOUT pausing the brain, pass error=None
        and put the text in `context`: wrapping the exception does not help,
        since the classifier reads `__cause__` and the message's status code.
        """
        import traceback
        error_str = str(error)
        error_type = type(error).__name__
        if isinstance(error, BaseException):
            tb = traceback.format_exception(type(error), error, error.__traceback__)
            tb_short = ''.join(tb[-3:]) if len(tb) > 3 else ''.join(tb)
        else:
            tb_short = ''
        self._log_event(
            'error', source,
            metadata={
                'error': error_str,
                'type': error_type,
                'context': context,
                'traceback': tb_short[:500],
            },
            file_level='ERROR',
            file_text='%s: %s' % (error_type, error_str),
            file_detail=tb_short,
            # context is the caller's stable description of the CONDITION, so
            # it belongs in the dedup identity: without it, two different
            # conditions with lookalike messages inside the dedup window
            # collapse into one log row (the s2 truncation twin-pull case).
            fingerprint='%s:%s:%s:%s' % (source, error_type,
                                         error_str[:100], context[:100]),
            ctx=ctx,
            session_id=session_id,
        )
        # After the row is written, never before: a provider refusal must be
        # recorded even if the latch itself misbehaves. Only an exception can
        # be one — a condition's message text must never trip the latch.
        if isinstance(error, BaseException):
            self._note_llm_failure(error, source)

    def _log_warning(self, source: str, message: str, context: str = '',
                     ctx=None):
        """Log a non-blocking warning to brain_logs.db + brain.log.

        For signals worth surfacing but not errors — empty-husk required
        aspect, auto-heal events, deprecated path used, etc. Writes
        event_type='warning' so consumers can distinguish severity.
        """
        self._log_event(
            'warning', source,
            metadata={'message': message, 'context': context},
            file_level='WARNING',
            file_text=message,
            file_detail=context,
            fingerprint='%s:warning:%s:%s' % (source, message[:100],
                                              context[:100]),
            ctx=ctx,
        )

    def get_recent_errors(self, hours: int = 24, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors from brain_logs.db via DAL. Default limit
        matches LogsDAL.get_recent_errors — no caller ever wanted 10."""
        try:
            return self._logs_dal.get_recent_errors(hours=hours, limit=limit)
        except Exception as e:
            # Can't log this to the errors table (it's the thing that failed)
            # — stderr/stdout is the minimum-loudness channel, so an empty
            # result is distinguishable from a broken errors read.
            print('[brain] get_recent_errors failed: %r' % e, flush=True)
            return []

    def log_watermark(self) -> Optional[int]:
        """Newest debug_log row id — the open end of a log window.

        None on failure, never 0: 0 is a VALID watermark meaning "empty
        table", so a failed read that returned it would widen the window to
        the whole table. On a clone of the operator's brain that turns a
        run's log report into months of someone else's history. A caller that
        cannot bound its window has to know it cannot.
        """
        try:
            return self._logs_dal.debug_log_watermark()
        except Exception as e:
            print('[brain] log_watermark failed: %r' % e, flush=True)
            return None

    def log_activity_since(self, row_id: int = 0,
                           samples: int = LOG_SWEEP_SAMPLES
                           ) -> List[Dict[str, Any]]:
        """Grouped debug_log activity after `row_id` — every origin and level,
        not only errors. The door an isolated measurement run reads to see
        what its own brain said while it ran.

        Raises rather than returning a placeholder. A synthesised row would
        enter the same list real rows do, and a reader counting them cannot
        tell an invented one from a logged one — the reporting surface must
        not be the thing that invents entries.
        """
        return self._logs_dal.rows_since(row_id, samples=samples)

    def log_debug(self, event_type: str, source: str, **kwargs) -> Dict[str, Any]:
        """Log a debug event to brain_logs.db + brain.log.

        Lightweight path (no rate-limit; caller-supplied event_type; returns a
        status dict), so it doesn't share _log_event — but the SQL still routes
        through the DAL writer rather than raw logs_conn.execute.
        """
        try:
            self._logs_dal.write_event(event_type, source, kwargs, session_id='unknown')
            # Also write to file log for non-error events
            self._write_to_file_log('DEBUG', source, '%s: %s' % (event_type, json.dumps(kwargs)[:200]))
            return {'logged': True}
        except Exception as e:
            return {'logged': False, 'error': str(e)}

    def get_config(self, key: str, default_val: Any = None) -> Any:
        """
        Get a config value from brain_meta via DAL.
        Auto-parses numbers and booleans based on default_val type.
        """
        try:
            val = self._meta.get(key, "")
            if not val:
                return default_val
            # Bool first — bool is a subclass of int in Python, so the
            # isinstance(default_val, (int, float)) check below would catch
            # bool defaults and try to parse "true"/"false" as int, throwing
            # a benign-but-noisy error every read. Handle bool explicitly.
            if isinstance(default_val, bool):
                if val in ('true', '1', 'yes', 'on', True):
                    return True
                if val in ('false', '0', 'no', 'off', False):
                    return False
                return default_val
            # Auto-parse numbers
            if default_val is not None and isinstance(default_val, (int, float)):
                try:
                    return float(val) if '.' in str(val) else int(val)
                except (ValueError, TypeError) as _e:
                    self._log_error("get_config", _e, "parsing numeric config value for key '%s'" % key)
            if val == 'true':
                return True
            if val == 'false':
                return False
            return val
        except Exception as _e:
            self._log_error("get_config", _e, "reading config key '%s' from brain_meta" % key)
        return default_val

    def warm_up(self) -> Dict[str, Any]:
        """First-call-only work, paid at boot instead of on the first prompt.

        Daemon boot announces ready and binds the socket immediately. This
        method runs in a background thread (see `BrainDaemon._run_warmup`)
        so the user never waits on it. By the time their first prompt
        arrives — they have to open Claude Code, see the splash, type —
        warmup is typically already done.

        What's covered:

          1. LAF engine caches. Recall scores against the engine's field
             matrices, title idf, project map and trace matrix; a cold
             engine builds them inside the first recall (measured ~0.8 s at
             12.5k nodes — the bulk of a ~1.2 s first recall). Vector bytes
             need no warming: VectorCache loads them at Brain construction.

          2. Structural degree cache. Used by the fatigue formula in the
             cosine loop. A single SQL UNION ALL across edges × edge_relations.
             Pre-built here so the recall hot path skips the lazy guard.

          3. Anthropic SDK + httpx pool. The S1 surface step calls Haiku
             on every recall via the Anthropic Python SDK. The first call
             in any process pays import (~350ms) + httpx pool init + TLS
             handshake to api.anthropic.com. We pay that here, store the
             warm client on `self.anthropic_client`, and `surface.py`
             reuses it across the daemon's ThreadPoolExecutor workers
             (httpx.Client is documented thread-safe).

             Two sub-phases: (a) eager import + client construction;
             (b) `models.retrieve` — free, warms TLS + httpx pool + DNS.
             That's enough.

             A 1-token `messages.create` was tried (2026-05-09) as a
             "Haiku route warmup". Removed: phase-timer data showed the
             daemon's surface_haiku phase is a stable ~5s whether the
             route is warmed or not. The ping wasn't buying real work.

        Note: an in-memory edge-text embedding pre-warm was tried here
        (2026-05-09, removed same day). It populated `_DESC_VEC_CACHE`
        with 1500 highest-weight edge texts so the first surface_spread
        phase wouldn't pay fastembed cost. Removed because it treated
        the symptom (cold cache on restart) instead of the underlying
        asymmetry: nodes have stored embeddings (one-time write, free
        reads forever), but edges did not — every spread call recomputed
        them. The fix was to give edges first-class stored embeddings
        like nodes (Option B in the design discussion); see
        `_compose_enriched_edge_text` in surface_contract.py and the
        `embedding` column on `edge_relations`.

        Note: cache_control / prompt-caching warmup is NOT here. The
        surface system block is ~2390 tokens, below Haiku 4.5's 4096-token
        minimum cacheable prefix. A cache warmup call would silently
        no-op. If the surface prefix grows past 4096, add it back.

        Idempotent and safe to race against a concurrent first recall —
        the recall hot path's lazy paths are guards, not initializers, so
        if warmup hasn't finished when the prompt arrives, recall will
        finish the work itself and warmup becomes a no-op on its second
        try. Last-writer-wins on dict assignment is fine because the data
        is identical. The Anthropic phase is similarly graceful: surface.py
        falls back to constructing its own client if `anthropic_client`
        isn't set yet.

        Failures are caught and logged. A warmup failure must never
        affect the daemon — falling through to lazy paths is acceptable
        degradation.

        Returns a dict of timings/sizes for daemon.log telemetry.
        """
        import time as _time
        t0 = _time.monotonic()
        timings: Dict[str, Any] = {}

        # 3 runs on its own thread: it is network-bound and shares nothing
        # with 1-2 (SQLite + numpy), so boot pays the slower side, not the sum.
        def _warm_anthropic():
            # 3. Anthropic SDK + Haiku connection.
            try:
                t = _time.monotonic()
                # Build + cache the shared client. _ensure_anthropic_client is
                # the single construction site (load_env + Anthropic());
                # httpx.Client is thread-safe, so one instance is shared across
                # the daemon's worker threads instead of each recall building
                # its own.
                self._ensure_anthropic_client()
                timings['anthropic_client_ms'] = int(
                    (_time.monotonic() - t) * 1000)

                # Free warmup: warms TLS handshake + httpx connection pool + DNS
                # to api.anthropic.com. Doesn't bill. Same call the idle
                # keepalive loop makes — one shared primitive
                # (warm_anthropic_connection). Keyless: skip — the warm can only
                # 401; note once instead.
                t = _time.monotonic()
                if self.llm_available:
                    self.warm_anthropic_connection()
                else:
                    self.note_llm_unavailable('boot warm-up')
                timings['anthropic_models_retrieve_ms'] = int(
                    (_time.monotonic() - t) * 1000)
                # No Haiku route ping. The ping was here briefly (2026-05-09)
                # but phase-timer data showed surface_haiku is a stable ~5s
                # whether the route is "warmed" or not — the ping wasn't
                # buying real latency. models.retrieve() warms TLS + pool +
                # DNS, which is the only first-call-only work that matters.
            except Exception as e:
                # SDK warmup failure must not crash the daemon. The client is
                # left unset; the next surface recall or keepalive tick rebuilds
                # it via _ensure_anthropic_client (self-heal) — no restart
                # needed.
                self._log_error(
                    'warmup_anthropic', e, 'Anthropic SDK warmup failed')
                timings['anthropic_error'] = str(e)
                self.anthropic_client = None

        anthropic_warm = threading.Thread(
            target=_warm_anthropic, daemon=True, name='warmup-anthropic')
        anthropic_warm.start()

        # 1. LAF engine caches. Same model value recall passes — the engine's
        #    matrices are not keyed by model. No embedder → recall exits
        #    before LAF, so there is nothing to warm.
        try:
            t = _time.monotonic()
            if embedder.is_ready():
                from .recall_laf import get_engine
                timings['laf_rows'] = get_engine(self).warm(
                    self, model=embedder.stats.get('model_name') or None)
                timings['laf_ms'] = int((_time.monotonic() - t) * 1000)
        except Exception as e:
            self._log_error('warmup_laf', e, 'LAF engine warm failed')
            timings['laf_error'] = str(e)

        # 2. Structural degree cache.
        try:
            t = _time.monotonic()
            self._ensure_structural_degree_cache()
            timings['degree_cache_size'] = len(
                getattr(self, '_structural_degree_cache', {}) or {})
            timings['degree_cache_ms'] = int((_time.monotonic() - t) * 1000)
        except Exception as e:
            self._log_error(
                'warmup_degree_cache', e, 'degree cache build failed')
            timings['degree_cache_error'] = str(e)

        anthropic_warm.join()

        timings['total_ms'] = int((_time.monotonic() - t0) * 1000)
        return timings

    @property
    def llm_available(self) -> bool:
        """True when a plausibly-valid Anthropic key (sk-*) is resolved.

        The single keyless-mode gate: S1 surface, S1 Scribe, S2 maintenance,
        and the connection warms all check this before spending an API call
        that can only 401. resolve_api_key() re-reads the env file on every
        check (the FILE wins over a stale os.environ value — that's what
        makes key replacement via /setup or the env file take effect with no
        restart, not just first-key). sk-* matches boot-brain.sh's gate: a
        placeholder value counts as missing, not as a key.

        NOT a pure read — reading it also rewrites os.environ's key and can
        clear the rejection latch. Both effects are load-bearing today (the
        encoder lane gets key freshness ONLY via this property, since
        s2/base.py reloads only on an empty environ), and both belong in an
        explicit refresh this property calls rather than hidden behind an
        attribute access. Named here so the next reader isn't surprised;
        lifting them out is sequenced work, not a drive-by.
        """
        from .scales.dispatch import resolve_api_key
        key = resolve_api_key()
        if key.startswith('sk-'):
            # Keep os.environ in sync so anthropic.Anthropic() (which reads
            # the env var at construction) builds with the CURRENT key.
            if os.environ.get('ANTHROPIC_API_KEY') != key:
                os.environ['ANTHROPIC_API_KEY'] = key
            # A key can be PRESENT and REFUSED — disabled in the console,
            # rotated, or past a spend cap. Presence alone kept the Scribe
            # firing at every poll against a dead key for a day (external
            # install, 2026-08-13: 293 consecutive 401s). The latch expires on
            # its own, so a key re-enabled by hand resumes with no restart and
            # no change to the key's VALUE — which is what the operator
            # actually does, and what a value-change trigger would miss.
            if time.time() < getattr(self, '_llm_rejected_until', 0.0):
                # ...unless the key ITSELF changed. The refusal was a verdict
                # on the old credential, so a replaced one deserves an
                # immediate try — otherwise pasting a fresh key into /setup
                # does nothing for up to an hour, breaking the "picked up
                # automatically, no restart needed" promise the keyless notice
                # makes. Re-enabling the SAME key still waits for the clock,
                # which is the case a value-change trigger alone would miss.
                if self.llm_rejection() is not None:
                    return False
                self._llm_rejected_until = 0.0
                self._llm_reject_strikes = 0     # new credential, fresh ladder
                self._file_logger.info(
                    'llm_available: API key replaced — clearing the %s latch'
                    % getattr(self, '_llm_rejected_kind', 'rejection'))
            if getattr(self, '_llm_unavailable_noted', False):
                self._llm_unavailable_noted = False
                self._file_logger.info(
                    'llm_available: API key resolved — LLM features live')
            return True
        return False

    def note_llm_rejected(self, outcome: dict, where: str = '') -> None:
        """Pause LLM features after the provider REFUSED a call.

        `outcome` is a `classify_llm_failure` result; only auth/quota kinds
        reach here. Escalates through LLM_REJECT_BACKOFF_MINUTES per
        consecutive rejection. A quota refusal rides the same ladder even when
        the provider names the date access returns: the operator can raise the
        limit on the same key long before that date, and only a re-probe
        notices — a refused probe costs no tokens.

        System-wide on purpose: a refused credential fails every session and
        every scale at once, so a per-session backoff would still let four
        concurrent sessions hammer the endpoint.
        """
        from .brain_constants import (LLM_REJECT_BACKOFF_MINUTES,
                                      LLM_REJECT_STRIKE_RESET_SECONDS)
        now = time.time()
        # Already latched → this is a straggler from a call that was in flight
        # when the gate closed, not a fresh probe. It must not advance the
        # ladder: four concurrent encodes failing in the same second would walk
        # straight to the ceiling, turning a three-second blip into an hour of
        # paused encoding. The ladder counts failed PROBES, one per window.
        if now < getattr(self, '_llm_rejected_until', 0.0):
            self._llm_rejected_at = now
            return
        if now - getattr(self, '_llm_rejected_at', 0.0) > LLM_REJECT_STRIKE_RESET_SECONDS:
            self._llm_reject_strikes = 0
        strikes = getattr(self, '_llm_reject_strikes', 0) + 1
        ladder = LLM_REJECT_BACKOFF_MINUTES
        until = now + ladder[min(strikes, len(ladder)) - 1] * 60

        from .scales.dispatch import key_fingerprint, resolve_api_key
        self._llm_reject_strikes = strikes
        self._llm_rejected_at = now
        self._llm_rejected_until = until
        self._llm_rejected_kind = outcome.get('kind', '')
        self._llm_rejected_detail = outcome.get('detail', '')
        # WHICH credential was refused — llm_available lifts the latch early
        # when the operator swaps in a different one. Fingerprint, never the key.
        self._llm_rejected_key = key_fingerprint(resolve_api_key())
        # One row per ENGAGE (the early return above makes every call that
        # reaches here one): the underlying failure logged its own error, and
        # the poll can rediscover a dead key many times inside one window.
        self._log_error(
            'llm_rejected', None,
            'LLM features paused %.0f min (%s, strike %d, first hit: %s) — '
            'provider refused the call: %s' % (
                (until - now) / 60.0, self._llm_rejected_kind, strikes,
                where or 'unknown', self._llm_rejected_detail))

    def llm_rejection(self) -> Optional[dict]:
        """The provider refusal pausing LLM features right now, or None.

        {'kind', 'detail', 'minutes_left'} — the one answer to "is this pause
        a refused key or a missing one?". Every channel that explains the pause
        (the errors row, the boot banner, the per-turn notice) asks here, so
        none of them can tell an operator whose key was refused to go paste a
        key. Read-only.
        """
        left = getattr(self, '_llm_rejected_until', 0.0) - time.time()
        if left <= 0:
            return None
        # A refusal is a verdict on ONE credential. Once the current key is a
        # different one — replaced or removed — this refusal no longer explains
        # the pause, and naming it would send the operator to the wrong fix.
        from .scales.dispatch import key_fingerprint, resolve_api_key
        if key_fingerprint(resolve_api_key()) != getattr(self, '_llm_rejected_key', ''):
            return None
        return {'kind': getattr(self, '_llm_rejected_kind', ''),
                'detail': getattr(self, '_llm_rejected_detail', ''),
                'minutes_left': left / 60.0}

    def _note_llm_failure(self, error: Exception, source: str) -> None:
        """Latch the LLM gate when `error` is a provider refusal.

        Called from `_log_error` — the sink every failing LLM path already
        reaches — so no caller has to thread a classification through. Anything
        that isn't a refusal classifies as `unknown` and returns silently, so a
        disk error or a JSON bug passes straight through.

        Only auth and quota latch. A rate limit is already paced by the SDK's
        Retry-After, a transient error by the retry helper, and an invalid
        request is per-payload — none of them mean "stop trying entirely".
        Narrow on purpose: a misclassification here can only cost at most one
        ladder step of paused encoding, never a wedged brain.
        """
        try:
            from .scales.dispatch import (classify_llm_failure,
                                          LLM_AUTH_REJECTED,
                                          LLM_QUOTA_EXHAUSTED)
            outcome = classify_llm_failure(error)
            if outcome['kind'] in (LLM_AUTH_REJECTED, LLM_QUOTA_EXHAUSTED):
                self.note_llm_rejected(outcome, where=source)
        except Exception as e:
            # Last-resort stderr, never a recursive _log_error: this runs
            # INSIDE the error sink.
            sys.stderr.write('[brain] llm-failure classification failed: %s\n' % e)

    # A Brain built with __new__ (tests) is never an eval brain.
    _eval_brain_since: Optional[float] = None

    @staticmethod
    def eval_brain_marker(db_dir: str) -> str:
        """Path of the eval-brain marker (EVAL_BRAIN_MARKER) for the brain in
        `db_dir`. The one construction the writer and the reader share."""
        from .brain_constants import EVAL_BRAIN_MARKER
        return os.path.join(db_dir, EVAL_BRAIN_MARKER)

    @staticmethod
    def mark_eval_brain(db_dir: str) -> None:
        """Mark the brain in `db_dir` an eval brain from now on. Mark it before
        any daemon opens it: an open brain keeps the moment it read at open.
        Every copy handed to an eval is marked again, which moves the moment
        forward, so the sessions the copy carries (the operator's live tails, a
        previous eval's leftovers) never count as the new eval's own.

        The moment is this host's clock, stamped explicitly: the daemon
        compares it with trace times from the same clock, and a network or
        FUSE mount would otherwise stamp its server's."""
        path = Brain.eval_brain_marker(db_dir)
        with open(path, 'a'):
            pass
        now = time.time()
        os.utime(path, (now, now))

    @staticmethod
    def read_eval_mark(db_dir: str) -> Optional[float]:
        """The marker's moment (epoch seconds) for the brain in `db_dir`, or
        None when it is not an eval brain, as production never is."""
        try:
            return os.stat(Brain.eval_brain_marker(db_dir)).st_mtime
        except FileNotFoundError:
            return None

    def eval_brain_since(self) -> Optional[float]:
        """When this brain became an eval brain, as read when it opened, or
        None. The one check for self-scheduled token spend:
        run_maintenance_if_due and scribe_due read it, and any new work that
        starts an LLM call on a timer must read it too."""
        return self._eval_brain_since

    def note_llm_unavailable(self, where: str) -> None:
        """Record the keyless state ONCE (errors table), not per attempt.

        Keyless mode is the designed first-run onboarding window — a single
        marker row keeps the dashboard honest without burying real errors
        under per-turn/per-tick auth-failure spam. Reset by llm_available
        when the key appears, so a later key removal notes again.
        """
        if getattr(self, '_llm_unavailable_noted', False):
            return
        # A latched brain has a key; it was refused. Saying "not resolved"
        # would send the operator to fix the one thing that isn't broken —
        # note_llm_rejected already wrote the accurate row.
        if self.llm_rejection() is not None:
            return
        self._llm_unavailable_noted = True
        self._log_error(
            'llm_unavailable',
            RuntimeError('ANTHROPIC_API_KEY not resolved'),
            'LLM features paused (first hit: %s) — memory storage, traces and '
            'direct recall unaffected. Set the key in ~/.config/brain/env; '
            'picked up automatically, no restart.' % where)

    def _ensure_anthropic_client(self):
        """Return the shared Anthropic client, building it if absent.

        The single construction site for the daemon's shared client. Three
        callers route through here so it's built once and cached on the brain:
        warm_up() (boot), warm_anthropic_connection() (keepalive self-heal after
        a boot failure), and surface.py (which no longer keeps a per-call
        throwaway fallback). Idempotent.

        load_env() resolves ANTHROPIC_API_KEY from the supported sources — the
        daemon is launched by launchd with no shell env, so without it the
        client constructs fine but fails on first use with "Could not resolve
        authentication method."

        Thread-safety: last-writer-wins on the assignment is fine — two racing
        constructions yield equivalent clients and the loser is GC'd
        (httpx.Client is cheap). Construction is lazy (no network), so this
        rarely raises; auth/network errors surface on the first real API call,
        where callers already handle them.
        """
        from .scales.dispatch import resolve_api_key
        from .brain_constants import (ANTHROPIC_CLIENT_TIMEOUT,
                                      ANTHROPIC_CONNECT_TIMEOUT)
        key = resolve_api_key()
        client = getattr(self, 'anthropic_client', None)
        # Key-stamped cache: reuse the client only while it was built with
        # the key that is CURRENT now. A replaced key (via /setup or the env
        # file — e.g. the first key was mistyped/revoked) rebuilds on the
        # next call, keeping the "picked up automatically, no restart"
        # promise true for replacement, not just first-key (code review
        # 2026-07-17). One file read per call — LLM calls only, never the
        # recall hot path.
        if client is not None and getattr(self, '_anthropic_client_key', None) == key:
            return client
        if os.environ.get('ANTHROPIC_API_KEY') != key and key:
            os.environ['ANTHROPIC_API_KEY'] = key
        import anthropic
        import httpx
        # Granular timeout: 600s is a read budget; connect gets its own 10s
        # bound so a dead network fails fast (see ANTHROPIC_CONNECT_TIMEOUT).
        client = anthropic.Anthropic(
            timeout=httpx.Timeout(ANTHROPIC_CLIENT_TIMEOUT,
                                  connect=ANTHROPIC_CONNECT_TIMEOUT))
        # Keyless boot (first-run onboarding): don't cache a client that can
        # never authenticate — the next call re-resolves. sk-* (not
        # truthiness) so a placeholder like 'changeme' is never cached —
        # same predicate as llm_available and boot-brain.sh.
        if not key.startswith('sk-'):
            return client
        self.anthropic_client = client
        self._anthropic_client_key = key
        return client

    def warm_anthropic_connection(self) -> bool:
        """Warm the Anthropic httpx connection (TLS + DNS) — the single free,
        no-bill `models.retrieve` primitive shared by warm_up() at boot and the
        daemon keepalive loop on idle.

        Why it matters: the S1 surface Haiku call reuses self.anthropic_client's
        connection pool across the daemon's worker threads. After the daemon
        sits idle, that pool's keep-alive sockets expire — the server FINs idle
        connections (the same stale-socket shape a post-host-sleep wake can
        leave behind) — and the next recall pays a fresh TLS+DNS handshake. Phase-timer data shows this as inflated
        surface_haiku after idle: median ~6s warm vs ~10s after >60m idle, while
        local DB/embedder phases stay flat (so it's the connection, not compute).

        Why models.retrieve and not a `messages.create` ping: the ping was tried
        and removed 2026-05-09 — it aimed to cut the ~5s baseline inference,
        which a ping can't do. models.retrieve warms TLS+pool+DNS without an
        inference and doesn't bill, which is exactly the idle (connection
        cold-start) penalty, not the inference.

        Self-heal: builds the client via _ensure_anthropic_client if it's unset
        (e.g. a boot-warmup failure left it None), so a transient boot failure
        recovers on a later keepalive tick — no daemon restart.

        Concurrency: a non-blocking lock makes the warm idempotent — if a warm
        is already in flight, a second caller returns False instead of double-
        warming. httpx.Client is itself thread-safe, so a warm racing a live
        recall is harmless; the lock only prevents two *warms* at once.

        Returns True if warmed, False if a warm is already in flight. RAISES on
        client-construction or API/SDK error — it deliberately does NOT swallow,
        because the two callers apply different failure policies: warm_up() nulls
        the client (the next call rebuilds it); the keepalive loop logs and
        retries. Each wraps this in its own try/except.
        """
        client = self._ensure_anthropic_client()
        # Skip if a warm is already running (boot warmup overlapping a keepalive
        # tick, or two ticks racing) — non-blocking so the caller never waits.
        if not self._anthropic_warm_lock.acquire(blocking=False):
            return False
        try:
            from .scales.s1.surface_contract import SURFACE_MODEL
            client.models.retrieve(SURFACE_MODEL)
            return True
        finally:
            self._anthropic_warm_lock.release()

    def _maybe_commit(self):
        """Commit self.conn unless we're inside a brain_batch transaction.

        Write methods on Brain (remember, revise, connect, archive_node)
        used to call self.conn.commit() directly. Inside a brain_batch
        with many ops, this meant N separate commits hitting the WAL
        writer slot — bad for parallel-session contention, and there was
        no rollback semantic if op #37 failed.

        Now those methods call _maybe_commit() instead. It defers to the
        connection's batch state (self.conn.in_batch): a no-op inside
        _handle_brain_batch (which owns one BEGIN IMMEDIATE / COMMIT with
        ROLLBACK on failure), a real commit on the normal single-op path.
        Same gate the DAL writers use — commit_unless_batched is the one
        source of truth.
        """
        commit_unless_batched(self.conn)

    def save(self):
        """
        Commit pending changes.

        Holds write_lock around the self.conn commit. save() is called
        lock-free from the daemon's S2 idle-maintenance path (_run_idle_
        maintenance) on a pool thread — the SAME pool that handles client
        commands. Without the lock, its commit() can land mid-flight on a
        concurrent client brain_batch (BEGIN IMMEDIATE + many writes on the
        shared self.conn), committing the batch's PARTIAL transaction and
        breaking its all-or-nothing atomicity. write_lock serializes save
        against brain_batch; it's an RLock, so the primary autosave path
        (daemon_server, which already holds write_lock before calling save)
        re-acquires safely. The logs write connection commits under
        logs_write_lock: logs_conn_w transactions only ever exist while that
        lock is held (the DAL write-boundary invariant), so an unlocked commit
        here could land mid-flight on another thread's append_batch and
        publish a partial trace set.
        """
        with self.write_lock:
            self.conn.commit()  # commit-ok: explicit durability point (save/autosave)
        try:
            with self.logs_write_lock:
                self.logs_conn_w.commit()
        except Exception:
            pass

    def close(self):
        """Persist cached SessionContexts, commit, close all database
        connections, and remove from singleton cache.

        The session flush runs first, while every connection is open: the
        cache holds changes no periodic flush has written yet, and a closer
        (daemon teardown, the boot fallback, a script) would otherwise drop
        them. Its failures are logged per session by save_session_contexts.

        Per-connection close is wrapped so a failure on one doesn't skip the
        others. Errors are logged via `_log_error` (no silent except: pass)
        with distinct origins per connection / phase so post-mortem can tell
        commit failures from close failures.
        """
        self.save_session_contexts()
        try:
            self.conn.commit()  # commit-ok: final commit on shutdown
        except Exception as e:
            self._log_error('brain_close_commit', e, 'primary conn final commit')
        try:
            self.conn.close()
        except Exception as e:
            self._log_error('brain_close', e, 'primary conn close')

        # Background-writer connection (vector backfill, access marks).
        try:
            self.conn_bg_writer.commit()
        except Exception as e:
            self._log_error('bg_writer_close_commit', e,
                            'bg_writer final commit')
        try:
            self.conn_bg_writer.close()
        except Exception as e:
            self._log_error('bg_writer_close', e, 'bg_writer close')

        # logs connections close — logged via stderr fallback because
        # _log_error itself writes via the logs write connection. Closed last
        # so prior _log_error calls in this method stay functional. The wconn
        # commit+close must hold logs_write_lock (a straggler thread — an
        # embed drain that outlived join_worker's 3s best-effort join, an S2
        # unit thread — may be mid-append_batch; an unlocked commit would
        # publish its partial trace set). Bounded acquire: shutdown must not
        # hang behind a wedged holder — on timeout, close anyway and say so.
        _got_logs_lock = self.logs_write_lock.acquire(timeout=5.0)
        try:
            self.logs_conn_w.commit()
            self.logs_conn_w.close()
        except Exception as e:
            import sys as _sys
            print('[brain] logs_conn_w close failed: %s' % e, file=_sys.stderr)
        finally:
            if _got_logs_lock:
                self.logs_write_lock.release()
            else:
                import sys as _sys
                print('[brain] logs_conn_w closed WITHOUT lock '
                      '(holder timeout) — a straggler write may have been '
                      'interrupted', file=_sys.stderr)
        try:
            self.logs_conn.commit()
            self.logs_conn.close()
        except Exception as e:
            import sys as _sys
            print('[brain] logs_conn close failed: %s' % e, file=_sys.stderr)
        # Remove from singleton registry if present
        canonical = os.path.realpath(self.db_path)
        with Brain._lock:
            if Brain._instances.get(canonical) is self:
                del Brain._instances[canonical]
