"""
brain — Data Access Layer (DAL): logs & traces (brain_logs.db)

Access layer for the four brain_logs.db-backed concerns: raw debug/hook
logs (LogsDAL), versioned prompt/config templates (InteractionDAL), the
fractal trace-event chain (TraceDAL), and per-session ephemeral state
(SessionStateDAL). Split out of dal.py (which now holds the brain.db
classes) along the one structural boundary that matters: which SQLite
file owns the table.

Usage in brain.py:
    from servers.dal_logs import LogsDAL, TraceDAL

    self._logs = LogsDAL(self.logs_conn, write_conn=self.logs_conn_w,
                         write_lock=self.write_lock)
    self._trace_dal = TraceDAL(self.logs_conn, write_conn=self.logs_conn_w,
                               write_lock=self.write_lock)

Read/write connection split (2026-08-18): brain_logs.db writes were silently
dropped by SQLITE_BUSY_SNAPSHOT — the shared connection held read snapshots
(concurrent read dispatch + embed-drain cursors) while writing, and a
read->write upgrade from a stale snapshot fails INSTANTLY, bypassing
busy_timeout entirely. The invariant that kills the
mechanism: `wconn` is touched ONLY inside `_wlock` (the daemon passes
brain.logs_write_lock — a leaf lock, separate from the graph write_lock so
log writes never queue behind graph batches), and every statement on it is
FULLY CONSUMED before the next write — fetchall(), or a single-step lookup
whose cursor is discarded (CPython refcount finalizes it immediately; do not
bind such a cursor to a name). No open cursor means no held snapshot, so
every write transaction begins at the WAL head. Multi-row table scans belong
on the shared read `conn`, where open cursors are harmless. Single-connection
construction (write_conn omitted) keeps the old behavior for standalone use.

See servers/dal.py for the brain.db-backed classes (nodes, edges, vectors, ...).
"""

import json
import secrets
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from .brain_constants import SESSION_LIVE_DAYS
from .clock import iso_cutoff, iso_now
from .db_backends.sqlite import commit_unless_batched, rollback_unless_batched


class _LogsWriteBase:
    """Shared read/write plumbing for the brain_logs.db DAL classes.

    `conn` serves reads; `wconn` serves writes and is only ever used while
    holding `_wlock` — that pairing is the snapshot-upgrade fix (see module
    docstring). Callers embedded in the daemon pass brain.write_lock (a
    reentrant TrackedRLock, so dispatch paths already holding it re-enter
    safely); standalone users get a private RLock and, with write_conn
    omitted, single-connection semantics identical to the pre-split DAL.
    """

    def __init__(self, conn: sqlite3.Connection,
                 write_conn: Optional[sqlite3.Connection] = None,
                 write_lock=None):
        self.conn = conn
        self.wconn = write_conn if write_conn is not None else conn
        self._wlock = write_lock if write_lock is not None else threading.RLock()


def _new_trace_id(conn) -> str:
    """Generate a fresh 8-char hex trace id, retrying on the vanishingly rare
    collision with an existing id. Matches node id shape (also 8-char hex via
    secrets.token_hex(4)). Collision space is 4 billion vs ~60K existing rows
    — first-try success is ~99.9985%."""
    for _ in range(5):
        candidate = secrets.token_hex(4)
        row = conn.execute(
            'SELECT 1 FROM trace_events WHERE id = ? LIMIT 1',
            (candidate,)
        ).fetchone()
        if row is None:
            return candidate
    raise RuntimeError("_new_trace_id: 5 consecutive collisions — investigate")


# Hard cap on a single query_logs pull — shared with the truncation payload
# in brain_recall.query_logs so the flagged 'limit' is the EFFECTIVE one (a
# note advising "raise limit" past this cap prescribed an impossible fix —
# 2026-08-07 review, finding 10).
LOG_QUERY_MAX_LIMIT = 200


# Sample messages carried per (origin, level) group by `rows_since`. One
# default, because three layers read this door and a per-layer default is
# how two callers quietly get different answers to the same question.
LOG_SWEEP_SAMPLES = 3


class LogsDAL(_LogsWriteBase):
    """Access layer for brain_logs.db's operational logs: debug_log,
    hook_errors, boot_renders."""

    # ── debug_log ──
    # Schema: id, session_id, event_type, source, metadata, created_at
    # Errors use event_type='error' with details in metadata JSON.

    def write_event(self, event_type: str, source: str,
                    metadata: Dict[str, Any], session_id: str = "") -> None:
        """Single INSERT path for debug_log rows (error / warning / debug).

        Owns the SQL + the batch-aware commit gate; the caller owns the
        policy and builds `metadata`. Brain._log_event (and the typed
        delegators below) route here so there is exactly ONE debug_log
        writer — a per-type footgun can't silently skip the INSERT, and
        every log row commits immediately (durable + visible to the
        dashboard's separate connection) unless inside a batch.
        """
        with self._wlock:
            self.wconn.execute(
                'INSERT INTO debug_log (session_id, event_type, source, metadata, created_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (session_id, event_type, source, json.dumps(metadata), iso_now())
            )
            commit_unless_batched(self.wconn)

    # ── hook_errors ──
    # The daemon-independent error table the dashboard + boot read. In-process
    # callers (e.g. the MCP health monitor) route here so the hook_errors write
    # lives in the DAL, not raw in the MCP layer. The table is canonically
    # defined in schema.py (LOG_TABLES['hook_errors']) — referenced here
    # defensively, never re-declared, so the schema can't drift.
    def log_hook_error(self, hook_name: str, error: str, context: str = "",
                       level: str = "error", traceback_str: str = "") -> None:
        """Append a hook_errors row (creating the table from the canonical schema
        if absent) and prune to the newest LOG_TABLES['hook_errors']['keep_rows']."""
        from servers.schema import LOG_TABLES
        spec = LOG_TABLES['hook_errors']
        with self._wlock:
            self.wconn.execute(spec['create'])
            self.wconn.execute(
                "INSERT INTO hook_errors (created_at, hook_name, level, error, context, traceback) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (iso_now(), hook_name, level, str(error), context[:500],
                 (traceback_str or "")[:2000]))
            self.wconn.execute(
                "DELETE FROM hook_errors WHERE id NOT IN "
                "(SELECT id FROM hook_errors ORDER BY id DESC LIMIT ?)", (spec['keep_rows'],))
            commit_unless_batched(self.wconn)

    def record_boot_render(self, session_id: str, user: str, project: str,
                           text: str) -> None:
        """Persist the exact boot text served to a session (boot_renders).
        Policy (skip-empty, never-raise) stays with the caller."""
        with self._wlock:
            self.wconn.execute(
                'INSERT INTO boot_renders '
                '(session_id, user, project, char_count, text, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (session_id, user or '', project or '', len(text), text,
                 iso_now()))
            commit_unless_batched(self.wconn)

    def prune_oversize(self, cutoff_iso: str) -> None:
        """Emergency size prune: drop debug_log rows older than
        `cutoff_iso`. Policy (size threshold, cutoff choice) stays with the
        caller (Brain._check_logs_db_size); this owns the SQL."""
        with self._wlock:
            self.wconn.execute(
                "DELETE FROM debug_log WHERE created_at < ?", (cutoff_iso,))
            commit_unless_batched(self.wconn)

    def clear_errors(self, cutoff_iso: str = '',
                     include_debug_log: bool = False) -> Dict[str, int]:
        """Clear hook_errors (and optionally debug_log). Empty cutoff clears
        everything. Returns per-table cleared counts."""
        cleared = {}
        with self._wlock:
            if cutoff_iso:
                c = self.wconn.execute(
                    "DELETE FROM hook_errors WHERE created_at < ?", (cutoff_iso,))
            else:
                c = self.wconn.execute("DELETE FROM hook_errors")
            cleared['hook_errors'] = c.rowcount
            if include_debug_log:
                if cutoff_iso:
                    c = self.wconn.execute(
                        "DELETE FROM debug_log WHERE created_at < ?", (cutoff_iso,))
                else:
                    c = self.wconn.execute("DELETE FROM debug_log")
                cleared['debug_log'] = c.rowcount
            commit_unless_batched(self.wconn)
        return cleared

    def get_recent_errors(self, hours: int = 24, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors from debug_log."""
        rows = self.conn.execute(
            "SELECT source, metadata, created_at FROM debug_log "
            "WHERE event_type = 'error' AND created_at > ? "
            "ORDER BY created_at DESC LIMIT ?",
            (iso_cutoff(hours=hours), limit)
        ).fetchall()
        results = []
        for source, metadata, created_at in rows:
            try:
                meta = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                meta = {'error': str(metadata)}
            results.append({
                'source': source,
                'error': meta.get('error', ''),
                'type': meta.get('type', ''),
                'context': meta.get('context', ''),
                'created_at': created_at,
            })
        return results

    # ── DB maintenance ──

    def debug_log_watermark(self) -> int:
        """Newest debug_log row id, or 0 when the table is empty.

        Paired with `rows_since` to bound a window by ID rather than by time:
        retention prunes mid-run, so a time window's contents shift under a
        long measurement while ids only grow.
        """
        row = self.conn.execute('SELECT MAX(id) FROM debug_log').fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def rows_since(self, row_id: int = 0,
                   samples: int = LOG_SWEEP_SAMPLES) -> List[Dict[str, Any]]:
        """Everything debug_log recorded after `row_id`, grouped by origin.

        The eval validity sweep's door. A measurement run wants its whole log
        window — not just errors — because the signal that invalidates a
        measurement is as often a warning or a gauge reading as an error. It
        is grouped rather than listed because a run can write thousands of
        rows and a reader needs the shape: which origins spoke, how loudly,
        over what span, and a few messages to judge them by.

        `query_logs` cannot serve this: its entries cap at
        LOG_QUERY_MAX_LIMIT and its counts are per-table, so a run noisier
        than the cap would report a truncated picture as a complete one.

        Returns one dict per (origin, level), busiest first.
        """
        groups = self.conn.execute(
            'SELECT source, event_type, COUNT(*), MIN(created_at), '
            '       MAX(created_at) '
            'FROM debug_log WHERE id > ? '
            'GROUP BY source, event_type ORDER BY COUNT(*) DESC', (row_id,)
        ).fetchall()
        out = []
        for origin, level, count, first_at, last_at in groups:
            rows = self.conn.execute(
                'SELECT metadata FROM debug_log '
                'WHERE id > ? AND source IS ? AND event_type IS ? '
                'ORDER BY id DESC LIMIT ?',
                (row_id, origin, level, max(samples, 0))
            ).fetchall()
            msgs = []
            for (raw,) in rows:
                try:
                    meta = json.loads(raw) if raw else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {'raw': str(raw)[:200]}
                # Explicit None, not truthiness: a producer that emitted an
                # EMPTY message is a thing the reader wants to see, and an
                # `or` chain hides it behind a dump of the whole row.
                text = meta.get('error')
                if text is None:
                    text = meta.get('message')
                if text is None:
                    text = meta
                msgs.append(str(text)[:200])
            out.append({'origin': origin or '', 'level': level or '',
                        'count': int(count), 'first_at': first_at,
                        'last_at': last_at, 'samples': msgs})
        return out

    def run_maintenance(self, graph_conn: sqlite3.Connection,
                        session_live_days: int = SESSION_LIVE_DAYS) -> Dict[str, Any]:
        """Run DB maintenance: prune old logs, clean orphans, reindex.

        Retention policy (Option C — significance-based):
          - debug_log errors: keep forever (errors are always signal)
          - debug_log telemetry: 30 days

        Also cleans orphaned graph data (vectors, edges, embeddings for deleted nodes).
        """
        stats = {}

        # --- Logs DB retention ---
        # debug_log: keep errors forever, prune telemetry/other after 30 days
        with self._wlock:
            cur = self.wconn.execute(
                "DELETE FROM debug_log WHERE event_type != 'error' "
                "AND created_at < ?",
                (iso_cutoff(days=30),))
            stats['debug_log_pruned'] = cur.rowcount

            # hook_errors: 30 days (surfaced ones only — unsurfaced kept until shown)
            try:
                cur = self.wconn.execute(
                    "DELETE FROM hook_errors WHERE surfaced = 1 "
                    "AND created_at < ?",
                    (iso_cutoff(days=30),))
                stats['hook_errors_pruned'] = cur.rowcount
            except Exception:
                stats['hook_errors_pruned'] = 0

            # session_state: a session unchanged for session_live_days is dead.
            # Its row is rewritten whenever its state changes (the daemon
            # flush writes only changed contexts; a resume re-boot rewrites
            # it), so only dead sessions age past the window. Guarded like
            # hook_errors — a missing table on an old install must not kill
            # the whole pass.
            try:
                cur = self.wconn.execute(
                    "DELETE FROM session_state WHERE updated_at < ?",
                    (iso_cutoff(days=session_live_days),))
                stats['session_state_pruned'] = cur.rowcount
            except Exception:
                stats['session_state_pruned'] = 0

            # boot_renders: 30 days. Full boot text per session start —
            # observability with a shelf life, not a record.
            try:
                cur = self.wconn.execute(
                    "DELETE FROM boot_renders WHERE created_at < ?",
                    (iso_cutoff(days=30),))
                stats['boot_renders_pruned'] = cur.rowcount
            except Exception:
                stats['boot_renders_pruned'] = 0

            commit_unless_batched(self.wconn)

        # --- Graph DB orphan cleanup ---
        if graph_conn:
            # edge_relations FIRST, scoped to exactly the edges this pass is
            # about to delete. Deleting edges alone strands their relation
            # rows as permanent orphans — invisible to every JOIN-based read
            # but polluting counts and blocking recovery (2026-06-12 audit:
            # 17,982 stranded rows, 84% of active relations, accumulated by
            # this one-sided delete). Scoped IN (...) — NOT a blanket
            # "NOT IN (SELECT edge_id FROM edges)" — so pre-existing orphans
            # stay untouched for the trace-based recovery effort.
            cur = graph_conn.execute(
                "DELETE FROM edge_relations WHERE edge_id IN ("
                "SELECT edge_id FROM edges "
                "WHERE source_id NOT IN (SELECT id FROM nodes) "
                "OR target_id NOT IN (SELECT id FROM nodes))")
            stats['orphaned_edge_relations'] = cur.rowcount

            cur = graph_conn.execute(
                "DELETE FROM edges WHERE source_id NOT IN (SELECT id FROM nodes) "
                "OR target_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_edges'] = cur.rowcount

            # Sanctioned raw-SQL exception (docs/VECTOR-DELETE-CONSOLIDATION-
            # PLAN.md §4.6): janitorial orphan sweep, structurally unable to
            # touch live nodes, runs on a caller-supplied graph_conn — not
            # worth a set-based VectorDAL surface for one caller. All LIVE-
            # node enrichment deletion goes through VectorDAL.delete_for_node.
            # (No archived-node residue sweep: archive_node de-indexes INSIDE
            # its transaction — a failure rolls the archive back, so there is
            # no crash window that leaves index rows on an archived node.)
            cur = graph_conn.execute(
                "DELETE FROM node_enrichments WHERE node_id NOT IN (SELECT id FROM nodes)")
            stats['orphaned_enrichments'] = cur.rowcount

            # FTS orphans: nodes_fts has no delete trigger, so rows for nodes
            # no longer in the table (legacy hard deletes left some) stay until
            # swept here. Orphan-only (NOT IN nodes); archive deletes its own
            # FTS row inline via _deindex_node. Guarded: test DBs may lack FTS5.
            has_fts = graph_conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
            ).fetchone() is not None
            if has_fts:
                cur = graph_conn.execute(
                    "DELETE FROM nodes_fts WHERE node_id NOT IN (SELECT id FROM nodes)")
                stats['orphaned_fts'] = cur.rowcount

            graph_conn.commit()

        # Summarize
        total_pruned = sum(v for k, v in stats.items() if 'pruned' in k)
        total_orphans = sum(v for k, v in stats.items() if 'orphaned' in k)
        stats['total_pruned'] = total_pruned
        stats['total_orphans'] = total_orphans

        return stats

    def query_logs(self, source: str = 'all', hours: int = 24,
                   level: str = 'all', hook_name: str = '',
                   limit: int = 50) -> Dict[str, Any]:
        """Unified log query across hook_errors and debug_log.

        Args:
            source: 'errors' (hook_errors), 'debug' (debug_log), or 'all'
                    (merged, sorted by time).
            hours: look back window (default 24).
            level: filter by level — 'error', 'critical', 'warning', or 'all'.
            hook_name: filter hook_errors by hook name (e.g. 'hook_recall').
            limit: max results per source (capped at 200).

        Returns: dict with 'entries' list and 'counts' summary.
        """
        limit = min(max(limit, 1), LOG_QUERY_MAX_LIMIT)
        cutoff = iso_cutoff(hours=hours)
        entries = []
        counts = {}

        # ── hook_errors ──
        if source in ('errors', 'all'):
            try:
                conditions = ['created_at > ?']
                params = [cutoff]
                if level != 'all':
                    conditions.append('level = ?')
                    params.append(level)
                if hook_name:
                    conditions.append('hook_name = ?')
                    params.append(hook_name)
                where = ' AND '.join(conditions)
                rows = self.conn.execute(
                    'SELECT id, hook_name, level, error, context, created_at '
                    'FROM hook_errors WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
                    params + [limit]
                ).fetchall()
                count_row = self.conn.execute(
                    'SELECT COUNT(*) FROM hook_errors WHERE %s' % where, params
                ).fetchone()
                counts['hook_errors'] = count_row[0] if count_row else 0
                for r in rows:
                    entries.append({
                        'source': 'hook_errors', 'id': r[0], 'hook_name': r[1],
                        'level': r[2], 'message': r[3], 'context': r[4] or '',
                        'created_at': r[5]
                    })
            except Exception:
                counts['hook_errors'] = 0

        # ── debug_log ──
        if source in ('debug', 'all'):
            try:
                conditions = ['created_at > ?']
                params = [cutoff]
                if level == 'error':
                    conditions.append("event_type = 'error'")
                elif level != 'all':
                    conditions.append('event_type = ?')
                    params.append(level)
                where = ' AND '.join(conditions)
                rows = self.conn.execute(
                    'SELECT id, source, event_type, metadata, created_at '
                    'FROM debug_log WHERE %s ORDER BY created_at DESC LIMIT ?' % where,
                    params + [limit]
                ).fetchall()
                count_row = self.conn.execute(
                    'SELECT COUNT(*) FROM debug_log WHERE %s' % where, params
                ).fetchone()
                counts['debug_log'] = count_row[0] if count_row else 0
                for r in rows:
                    meta = {}
                    try:
                        meta = json.loads(r[3]) if r[3] else {}
                    except (json.JSONDecodeError, TypeError):
                        meta = {'raw': str(r[3])[:200]}
                    entries.append({
                        'source': 'debug_log', 'id': r[0], 'origin': r[1] or '',
                        'level': r[2], 'message': meta.get('error', meta.get('message', str(meta)[:200])),
                        'context': meta.get('context', ''),
                        'created_at': r[4]
                    })
            except Exception:
                counts['debug_log'] = 0

        # Sort merged entries by time descending
        entries.sort(key=lambda e: e.get('created_at', ''), reverse=True)
        if source == 'all':
            entries = entries[:limit]

        return {'entries': entries, 'counts': counts}


# Provenance values that mean "the system put this here", as opposed to a human
# making a deployment decision. Named here, at the table's owner — the one-time
# override collapse and the pointer audits read them to tell
# system-placed pointers from a human's deployment decisions.
#
# None is minted anymore: RECONCILE belonged to the deleted shipped-prompt
# reconcile, AUTO_V1 to register's old v1 auto-activate, BACKSTOP to the old
# ensure_logs_schema pointer fill-in. All three survive to classify the live
# rows they stamped.
#
# Closed vocabulary: callers coming through the MCP door may not supply any of
# them, or a stray call could relabel a human's deployment decision as an
# untouched default and get published over.
AUTO_V1_PROVENANCE = 'register:auto_v1'
RECONCILE_PROVENANCE = 'seed:reconcile'
BACKSTOP_PROVENANCE = 'migration:initial_active'
SYSTEM_PROVENANCE = (AUTO_V1_PROVENANCE, RECONCILE_PROVENANCE,
                     BACKSTOP_PROVENANCE)


class InteractionDAL(_LogsWriteBase):
    """Access layer for interactions — versioned templates for system boundaries.

    Every learnable boundary (surface prompt, encoding prompt, voice format,
    signal assembly) is an interaction. Versioned, traceable, optimizable
    by higher scales.

    Active-version model (2026-05-10):
      - `register()` inserts a new version row. Does NOT change which version
        the runtime reads. Decoupled by design.
      - `set_active()` flips the per-name active pointer to a chosen version.
      - `clear_active()` deletes the pointer — reverts to the code default.
      - `get_active()` reads via the active pointer only; None when no
        pointer exists — "no pointer" means "no override deployed".
      - `get_version()` reads a specific version (used by eval overrides).
    """

    def register(self, name: str, template: str, parameters: str = '',
                 created_by: str = 'anchor') -> Dict[str, Any]:
        """Register a new version of an interaction. Auto-increments version.

        Never activates — a write is not a deployment decision. Every name
        has a code default to run on, so a registered-but-inactive version
        is readable state, not a dead name. Flip the runtime pointer with
        `set_active()`.
        """
        now = iso_now()
        with self._wlock:
            # Get current max version — on wconn so the read is atomic with the
            # INSERT under the same lock hold (no register can interleave).
            # fetchall(): a wconn statement must be exhausted, never left open
            # (module docstring — an open cursor pins a snapshot).
            rows = self.wconn.execute(
                'SELECT MAX(version) FROM interactions WHERE name = ?', (name,)
            ).fetchall()
            version = (rows[0][0] or 0) + 1 if rows else 1
            parent = version - 1 if version > 1 else None
            self.wconn.execute(
                'INSERT INTO interactions (name, version, template, parameters, created_at, created_by, parent_version) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (name, version, template, parameters, now, created_by, parent))
            new_id = self.wconn.execute('SELECT last_insert_rowid()').fetchall()[0][0]
            commit_unless_batched(self.wconn)
        return {'name': name, 'version': version, 'id': new_id}

    def set_active(self, name: str, version: int,
                   set_by: str = 'anchor') -> Dict[str, Any]:
        """Flip the active pointer for `name` to `version`.

        UPSERT into interaction_active. Verifies the (name, version) pair
        actually exists in `interactions` before flipping — refuses to activate
        a non-existent version.
        """
        with self._wlock:
            # Verify the target version exists — on wconn, atomic with the
            # flip. fetchall(): wconn statements must be exhausted (module
            # docstring).
            row = self.wconn.execute(
                'SELECT 1 FROM interactions WHERE name = ? AND version = ?',
                (name, version)).fetchall()
            if not row:
                raise ValueError(
                    "Cannot activate %s v%d: no such version registered" % (name, version))
            now = iso_now()
            self.wconn.execute(
                'INSERT INTO interaction_active (name, version, set_at, set_by) '
                'VALUES (?, ?, ?, ?) '
                'ON CONFLICT(name) DO UPDATE SET '
                'version = excluded.version, set_at = excluded.set_at, set_by = excluded.set_by',
                (name, version, now, set_by))
            commit_unless_batched(self.wconn)
        return {'name': name, 'version': version, 'set_at': now, 'set_by': set_by}

    def clear_active(self, name: str) -> bool:
        """Delete the active pointer for `name` — the inverse of `set_active`.

        Pure mechanics: the caller owns what "no pointer" means (the resolver
        serves the code default) and any cache invalidation. Registered
        versions stay on record. Returns True when a pointer was deleted,
        False when none existed.
        """
        with self._wlock:
            cur = self.wconn.execute(
                'DELETE FROM interaction_active WHERE name = ?', (name,))
            deleted = cur.rowcount > 0
            commit_unless_batched(self.wconn)
        return deleted

    def get_active(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the currently-active version of an interaction.

        Reads via the interaction_active pointer only. Returns None when no
        pointer exists (or it dangles) — pointer presence is the single
        "an override is deployed" bit; what runs then is the caller's policy
        (the resolver falls through to the code default).
        """
        row = self.conn.execute(
            'SELECT i.id, i.name, i.version, i.template, i.parameters, '
            'i.created_at, i.created_by '
            'FROM interaction_active a '
            'JOIN interactions i ON i.name = a.name AND i.version = a.version '
            'WHERE a.name = ?', (name,)).fetchone()
        if not row:
            return None
        return {'id': row[0], 'name': row[1], 'version': row[2],
                'template': row[3], 'parameters': row[4],
                'created_at': row[5], 'created_by': row[6]}

    def get_version(self, name: str, version: int) -> Optional[Dict[str, Any]]:
        """Get a specific version of an interaction."""
        row = self.conn.execute(
            'SELECT id, name, version, template, parameters, created_at, created_by, parent_version '
            'FROM interactions WHERE name = ? AND version = ?',
            (name, version)).fetchone()
        if not row:
            return None
        return {'id': row[0], 'name': row[1], 'version': row[2],
                'template': row[3], 'parameters': row[4],
                'created_at': row[5], 'created_by': row[6],
                'parent_version': row[7]}

    def list_all(self) -> List[Dict[str, Any]]:
        """List all interactions: name, max_version, total_versions,
        active_version, active_set_by, active_set_at.

        `active_set_by` is who flipped the pointer — the provenance the
        shipped-prompt reconcile reads to tell a system default apart from a
        human's deployment decision. See SYSTEM_PROVENANCE. `active_set_at`
        is when, which is what distinguishes a recent deliberate deployment
        from years-old seed residue when a collapse is being reviewed.
        """
        rows = self.conn.execute(
            'SELECT i.name, MAX(i.version) as v, COUNT(*) as versions, '
            'a.version, a.set_by, a.set_at '
            'FROM interactions i '
            'LEFT JOIN interaction_active a ON a.name = i.name '
            'GROUP BY i.name ORDER BY i.name').fetchall()
        return [{'name': r[0], 'max_version': r[1], 'total_versions': r[2],
                 'active_version': r[3], 'active_set_by': r[4],
                 'active_set_at': r[5]}
                for r in rows]

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """Every registered version of `name`, ascending: version + created_by.

        Registry fact, not policy: callers decide what a given `created_by`
        means. Used to distinguish a human's dormant candidate (which must
        never be published over) from a crashed reconcile's residue.
        """
        rows = self.conn.execute(
            'SELECT version, created_by FROM interactions '
            'WHERE name = ? ORDER BY version', (name,)).fetchall()
        return [{'version': r[0], 'created_by': r[1]} for r in rows]


def _attended_sql(alias: str) -> str:
    """SQL predicate: the row at `alias` is an OPERATOR-ATTENDED turn — the
    latest user_message at-or-before it lacks the wake-envelope marker. Takes
    one bound param (the marker LIKE pattern).

    For RANKING only. Testing the marker on a row's OWN summary catches half a
    machine-woken exchange: the envelope is the user_message, but the reply to
    it is an ordinary assistant_message carrying no marker, so a self-test
    passes the reply through and every wake refreshes the stream's recency.

    Do NOT reach for this wherever an envelope filter appears. It answers "did
    the operator provoke this turn", which is the right question for ranking and
    the WRONG one for description: a machine-woken reply is unattended yet is
    usually the most informative line about that stream, so focus and peek keep
    it and filter only the envelope itself.

    The envelope self-excludes (it is its own latest predecessor); its reply is
    excluded by that predecessor. COALESCE keeps a conversational row that has
    no preceding user_message at all ATTENDED rather than silently dropping it.
    Prefix-matched (`marker%`) to stay index-friendly, matching every other
    envelope filter in this module; trace_contract.is_machine_turn tests the
    marker anywhere in the text, so the two agree only while envelopes lead
    with it — which is how the harness writes them.
    """
    return ("COALESCE((SELECT p.summary FROM trace_events p "
            "  WHERE p.scale = 's0' AND p.session_id = {a}.session_id "
            "    AND p.ref_type = 'user_message' "
            "    AND p.created_at <= {a}.created_at "
            "  ORDER BY p.created_at DESC LIMIT 1), '') NOT LIKE ?").format(a=alias)


class TraceDAL(_LogsWriteBase):
    """Access layer for trace_events — the fractal learning loop.

    Append-only event chains. Each chain tracks one integrate() cycle:
    what was observed, what knowledge was selected, what was produced,
    and what happened next (corrections, recalls, outcomes).

    Over time, all small log tables migrate into trace_events.
    Each becomes a different event_type + ref_type in one table.
    """

    def __init__(self, conn: sqlite3.Connection,
                 write_conn: Optional[sqlite3.Connection] = None,
                 write_lock=None):
        super().__init__(conn, write_conn=write_conn, write_lock=write_lock)
        # Identity stamped onto every trace event's metadata at write time.
        # Empty strings = unset → no stamping (graceful degradation; no
        # placeholder sentinel tokens). Set once at Brain init via
        # set_identity() from daemon_config env vars.
        self._human_identity: str = ''
        self._agent_identity: str = ''
        # Whether the agent name was CHOSEN. The agent name always has a
        # value now (it falls back to a shipped default), so its presence no
        # longer tells us anything — this does.
        self._agent_identity_is_default: bool = False
        # One-shot guard for the "identity unset on write" warning.
        # The gap doesn't change per-write, only per-restart — logging
        # every write would flood; logging once per TraceDAL lifetime
        # surfaces the gap without noise.
        self._identity_missing_logged: bool = False

    def set_identity(self, human_identity: str, agent_identity: str,
                     agent_is_default: bool = False) -> None:
        """Configure the identity tokens stamped onto every trace event.

        Source values come from BRAIN_OPERATOR_NAME / BRAIN_AGENT_NAME env
        vars resolved at Brain construction. Identity is a property of the
        substrate — each trace_event records the speakers present when it
        was written. Per-event override is possible by passing
        human_identity / agent_identity directly in `metadata`.

        `agent_is_default` says the agent name was issued rather than chosen.
        It is stamped all the same — an entity running under its shipped name
        genuinely IS that name, and a rename later leaves earlier events
        honestly recording who was speaking then — but it keeps the
        unconfigured-identity warning alive, which a non-empty default would
        otherwise silence forever.
        """
        self._human_identity = (human_identity or '').strip()
        self._agent_identity = (agent_identity or '').strip()
        self._agent_identity_is_default = bool(agent_is_default)

    def _maybe_warn_identity_unset(self, scale: str, ref_type: str) -> None:
        """Loud-by-default at the scale-write boundary: when a trace
        gets written but identity stamping is empty, surface it once.

        The check lives at the write site (not at Brain.__init__)
        because boot is a moment but writes are continuous — every
        scale write through this DAL passes here. The first one that
        fires with unset identity tells the operator the gap exists.
        Subsequent writes stay silent (one-shot via
        `_identity_missing_logged`) — the config didn't suddenly
        re-break, no value in spamming.

        Output goes to stderr → daemon.log (launchd-managed
        StandardErrorPath). TraceDAL has no Brain reference, so we
        don't route through brain._log_error here.
        """
        unchosen_agent = not self._agent_identity or self._agent_identity_is_default
        if self._human_identity and not unchosen_agent:
            return
        if self._identity_missing_logged:
            return
        import sys as _sys
        missing = []
        if not self._human_identity:
            missing.append('BRAIN_OPERATOR_NAME')
        if unchosen_agent:
            missing.append('BRAIN_AGENT_NAME')
        # Mirror what the writer actually does (see _stamp below): it skips
        # entirely only when BOTH names are empty, and otherwise stamps
        # whichever it has. Saying "no identity metadata" for a partial stamp
        # sends the operator hunting for a bug that isn't there.
        if not self._agent_identity and not self._human_identity:
            effect = 'Trace events will be written without identity metadata'
        elif not self._human_identity:
            effect = 'Trace events are stamped with the agent name only'
        elif not self._agent_identity:
            effect = 'Trace events are stamped with the operator name only'
        else:
            effect = 'Trace events are stamped with the default agent name'
        print('[trace_dal] identity unconfigured on first scale-%s write '
              '(ref_type=%s) — %s not set in env. %s until '
              '~/.config/brain/env is configured and the daemon '
              'restarts.' %
              (scale, ref_type or '?', ', '.join(missing), effect),
              file=_sys.stderr, flush=True)
        self._identity_missing_logged = True

    def _stamp_identity(self, metadata: Optional[Dict]) -> Optional[Dict]:
        """Inject configured identity tokens into a metadata dict.

        setdefault semantics — explicit per-event values win. Returns
        metadata unchanged if neither identity is configured, or if
        metadata is a non-dict value (defensive: callers that pass
        unexpected shapes don't crash the trace write — the daemon
        dispatch layer is responsible for normalizing to dict).
        """
        if not self._human_identity and not self._agent_identity:
            return metadata
        if metadata is not None and not isinstance(metadata, dict):
            return metadata
        meta = dict(metadata) if metadata else {}
        if self._human_identity:
            meta.setdefault('human_identity', self._human_identity)
        if self._agent_identity:
            meta.setdefault('agent_identity', self._agent_identity)
        return meta

    def _warn_metadata_invalid(self, ref_type, error):
        """Loud-by-default at the trace-write chokepoint: a ref_type with a
        declared payload schema carried a malformed metadata dict. NEVER blocks
        the write (a malformed trace beats a lost one) — logs to stderr →
        daemon.log, the same channel as _maybe_warn_identity_unset (TraceDAL has
        no Brain reference). Fires per occurrence — each malformed trace is a
        distinct signal."""
        import sys as _sys
        print('[trace_dal] trace metadata invalid (ref_type=%s): %s — wrote anyway'
              % (ref_type, error), file=_sys.stderr, flush=True)

    def append(self, chain_id: str, scale: str, event_type: str,
               ref_type: str = '', ref_id: str = '', summary: str = '',
               metadata: Optional[Dict] = None, session_id: str = '',
               interaction_id: int = None) -> str:
        """Append an event to a trace chain. Returns event id (8-char hex).

        Validates against trace_contract before writing. Configured
        identity tokens (set_identity) are stamped into metadata via
        setdefault — explicit per-event values win.
        """
        return self.append_batch([{
            'chain_id': chain_id, 'scale': scale, 'event_type': event_type,
            'ref_type': ref_type, 'ref_id': ref_id,
            'summary': summary if summary else '',
            'metadata': metadata, 'session_id': session_id,
            'interaction_id': interaction_id}])[0]

    def append_batch(self, events: list) -> List[str]:
        """Append multiple trace events in a single transaction.

        Reduces WAL lock contention — one commit instead of N.
        Each event dict uses the same keys as append() (which delegates here —
        this per-event body is the write path's ONE implementation).
        Identity stamping applies per-event via setdefault semantics.

        ATOMIC: all events land or none do. Contract validation is pre-flight,
        so a violation writes nothing; anything raising mid-insert rolls back
        the pending rows. Callers may therefore batch a whole command's traces
        in one append_batch without risking a partially-published set.

        Two validations, deliberately different severities: `validate_trace_event`
        BLOCKS (an unregistered triple is a programmer bug and must not reach the
        table), while metadata-shape validation is loud but never blocking — the
        row still lands. This is the SINGLE chokepoint every writer passes
        (inline S1/S2 + the dispatched command), so the guard actually fires in
        production: the command-boundary check missed every in-process delta
        write (S2 units + S1 Scribe run with dispatch=None).
        """
        from .trace_contract import validate_trace_event, validate_trace_metadata

        # PRE-FLIGHT: validate every event before mutating anything, so a
        # contract violation is a pure no-op — no INSERT work started and
        # discarded, and no per-row stderr warnings emitted for rows that will
        # never land. This is NOT what makes the batch atomic (the rollback
        # below is); it's "don't start what you can't finish".
        for ev in events:
            ok, error = validate_trace_event(ev['scale'], ev['event_type'], ev.get('ref_type', ''))
            if not ok:
                raise ValueError("Trace contract violation: %s" % error)

        ids = []
        with self._wlock:
            # Stamped under the lock, so created_at follows the order this
            # process writes in: readers page with a `created_at > cursor`
            # bookmark (the dashboard's recall feed, whose rows only the daemon
            # writes), and a row stamped before waiting here would commit behind
            # a newer one and fall below a cursor already past it.
            now = iso_now()
            try:
                for ev in events:
                    # Non-blocking by design: shape drift warns (stderr) and the
                    # row still lands. Stays in this loop — it must not gate writes.
                    meta_ok, meta_err = validate_trace_metadata(
                        ev['event_type'], ev.get('ref_type', ''), ev.get('metadata'))
                    if not meta_ok:
                        self._warn_metadata_invalid(ev.get('ref_type', ''), meta_err)
                    self._maybe_warn_identity_unset(ev['scale'], ev.get('ref_type', ''))
                    metadata = self._stamp_identity(ev.get('metadata'))
                    meta_json = json.dumps(metadata) if metadata else None
                    # Collision check on wconn: same-transaction, so it sees
                    # ids inserted earlier in THIS batch (the read conn's
                    # snapshot cannot); single-step lookup, cursor discarded —
                    # the safe wconn read pattern (module docstring).
                    trace_id = _new_trace_id(self.wconn)
                    self.wconn.execute(
                        'INSERT INTO trace_events '
                        '(id, chain_id, scale, event_type, ref_type, ref_id, summary, metadata, session_id, interaction_id, created_at) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (trace_id, ev['chain_id'], ev['scale'], ev['event_type'], ev.get('ref_type', ''),
                         ev.get('ref_id', ''), ev.get('summary', ''), meta_json,
                         ev.get('session_id', ''), ev.get('interaction_id'), now))
                    ids.append(trace_id)
            except Exception:
                # THIS is the atomicity guarantee. Without it, a raise part-way
                # through leaves the already-inserted rows PENDING: the commit below
                # is skipped, but `self.wconn` is default-isolation, so the next
                # unrelated write on it — including the error log reporting this
                # very failure — commits that prefix. The result is a partial trace
                # set that lies by omission, which is strictly worse than no traces
                # (a missing trace is recoverable from the graph; a partial one
                # misrepresents it). Safe to fire: every logs writer in the repo
                # ends with commit_unless_batched, so there is never another
                # caller's pending work here to discard. Known limit: this does
                # NOT protect against a nested logs write from INSIDE this loop
                # (reentrant lock, same connection — its commit would publish
                # the partial batch). No such caller exists: the two warn
                # helpers are stderr-only by that exact design.
                rollback_unless_batched(self.wconn)
                raise
            commit_unless_batched(self.wconn)
        return ids

    def _decode_metadata(self, raw: Optional[str]) -> Dict[str, Any]:
        """Decode a trace_events.metadata JSON cell into a dict.

        Defensive against pre-v27 tool_result rows where the metadata
        was double-encoded (the dispatch handler json.dumps'd a string
        client payload, then trace_dal.append json.dumps'd it again).
        Single-encoded rows (post-2026-05-23 dispatch fix) decode in
        one pass; double-encoded legacy rows decode in two. Returns
        an empty dict on any failure so callers can rely on dict shape.
        """
        if not raw:
            return {}
        try:
            meta = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                return {}
        return meta if isinstance(meta, dict) else {}

    # The canonical 10-column projection _row_to_event() decodes. Interpolate
    # one of these into every trace_events SELECT — never restate the list
    # (the hand-maintained copies drifted into different orders before).
    _CANON_FIELDS = ('id', 'chain_id', 'scale', 'event_type', 'ref_type',
                     'ref_id', 'summary', 'metadata', 'session_id', 'created_at')
    _CANON_COLS = ', '.join(_CANON_FIELDS)
    _CANON_COLS_TE = ', '.join('te.' + f for f in _CANON_FIELDS)

    def _row_to_event(self, r) -> Dict[str, Any]:
        """Map ONE canonical-order trace_events row tuple → the event dict.

        The single column→dict mapping for the whole DAL: every reader's SELECT
        interpolates _CANON_COLS (or the te.-qualified variant) so the
        index→field binding lives in ONE place.

        ref_type/ref_id/summary are ''-guarded (they're '' at write, never NULL
        in practice — the guard keeps the dict total). metadata is decoded.
        """
        if len(r) != 10:
            # Loud-by-default: a mis-arity SELECT would otherwise mis-index
            # silently (over-select ignores extra cols; under-select IndexErrors
            # at a confusing offset). Name the contract at the boundary instead.
            raise ValueError(
                "_row_to_event expects 10 columns in canonical order "
                "(id, chain_id, scale, event_type, ref_type, ref_id, summary, "
                "metadata, session_id, created_at), got %d" % len(r))
        return {
            'id': r[0], 'chain_id': r[1], 'scale': r[2], 'event_type': r[3],
            'ref_type': r[4] or '', 'ref_id': r[5] or '', 'summary': r[6] or '',
            'metadata': self._decode_metadata(r[7]),
            'session_id': r[8] or '', 'created_at': r[9],
        }

    def get_by_ids(self, trace_ids: List[str]) -> List[Dict[str, Any]]:
        """Point/batch lookup by trace_event.id (v29: 8-char hex strings).
        Returns rows in ascending id order (deterministic); missing ids are
        silently skipped (mirrors NodeDAL.get_bulk behavior — caller checks
        len(result) vs len(input) if presence matters).

        Rejects int input loudly per v29 contract.
        """
        if not trace_ids:
            return []
        for tid in trace_ids:
            if not isinstance(tid, str):
                raise ValueError(
                    "get_by_ids: trace_ids must be strings, got %s (%r). "
                    "v29 trace ids are 8-char hex." % (
                        type(tid).__name__, tid))
        placeholders = ','.join('?' * len(trace_ids))
        rows = self.conn.execute(
            'SELECT %s FROM trace_events WHERE id IN (%s) '
            'ORDER BY id ASC' % (self._CANON_COLS, placeholders),
            list(trace_ids)).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_chain(self, chain_id: str,
                  ref_type: str = '') -> List[Dict[str, Any]]:
        """Get all events in a trace chain, ordered by time. Each event is the
        full canonical row (incl. its own chain_id — = the queried id).
        Optional ref_type narrows to one event kind — the bounded form for
        point lookups (e.g. a run's encoding_prompt), avoiding the full-chain
        metadata decode."""
        extra = ' AND ref_type = ?' if ref_type else ''
        params = (chain_id, ref_type) if ref_type else (chain_id,)
        rows = self.conn.execute(
            'SELECT %s FROM trace_events WHERE chain_id = ?%s '
            'ORDER BY created_at ASC' % (self._CANON_COLS, extra),
            params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_recent(self, scale: str = '', hours: Optional[int] = 24,
                   event_type: str = '', session_id: str = '',
                   session_ids: Optional[List[str]] = None,
                   limit: int = 100, chain_suffix: str = '',
                   exclude_ref_types: Optional[List[str]] = None,
                   older_than: str = '') -> List[Dict[str, Any]]:
        """Get recent trace events, optionally filtered by scale/type/session.

        Session filtering — three modes:

        - **session_id** (singular, str): authoritative for one session. The
          `hours` window is ignored — historical sessions older than the
          default cutoff must not silently truncate to empty. Zero-row
          results log loud to stderr.

        - **session_ids** (plural, List[str]): authoritative for N sessions.
          Returns events from any session in the list, ordered by time.
          The `hours` window is ignored (same reasoning). Zero-row results
          log loud. Useful for cross-session audits, eval cohorts,
          partnership-arc lookups across distinct conversations.

        - **neither**: default hours-based recent window.

        Passing both session_id and session_ids raises ValueError — the
        caller's intent is ambiguous and we don't guess.
        """
        # Session scope is authoritative — the hours window is skipped so
        # historical sessions don't silently truncate to empty. exclude_ref_types
        # drops residue (journal_note) so "recent integration deltas" don't
        # count encoder notes. The XOR guard + predicate build live in
        # _event_where (the one WHERE source for generic trace_events readers).
        where, params = self._event_where(
            scale=scale, event_type=event_type,
            session_id=session_id, session_ids=session_ids,
            hours=None if (session_id or session_ids) else hours,
            chain_suffix=chain_suffix, exclude_ref_types=exclude_ref_types,
            older_than=older_than)
        rows = self.conn.execute(
            'SELECT %s FROM trace_events te '
            'WHERE %s ORDER BY te.created_at DESC LIMIT ?'
            % (self._CANON_COLS_TE, where),
            params + [limit]).fetchall()
        out = [self._row_to_event(r) for r in rows]
        # Loud-by-default: explicit session filter with zero rows is a real
        # signal (typo, archived session, wrong DB), not a quiet noop.
        if (session_id or session_ids) and not out:
            import sys as _sys
            who = session_id[:12] if session_id else ('[' + ','.join(s[:8] for s in session_ids[:3]) +
                                                       (',…' if len(session_ids) > 3 else '') + ']')
            print('[query_traces] WARN session filter %s returned 0 rows '
                  '(scale=%r event_type=%r limit=%d)'
                  % (who, scale, event_type, limit),
                  file=_sys.stderr)
        return out

    def _event_where(self, contains: str = '', scale: str = '',
                     event_type: str = '', ref_types: Optional[List[str]] = None,
                     session_id: str = '', session_ids: Optional[List[str]] = None,
                     younger_than: str = '', older_than: str = '',
                     ref_type: str = '', ref_id: str = '',
                     hours: Optional[int] = None, chain_suffix: str = '',
                     exclude_ref_types: Optional[List[str]] = None):
        """Build the shared WHERE clause + params for every trace_events reader
        (get_recent, get_by_ref_type, get_chains, filter_events,
        filter_event_vectors). Columns are `te.`-qualified so the same clause
        works under the trace_embeddings JOIN — plain readers alias the table
        (`FROM trace_events te`).

        Purely mechanical: applies exactly the predicates it is given. Door
        SEMANTICS stay at each caller — e.g. get_recent skips `hours` under a
        session scope while get_by_ref_type applies both; each door passes (or
        withholds) `hours` accordingly.

        Needles: contains → (summary OR metadata) LIKE %s% (metadata is JSON
        text, so it searches the full body, not the 200-char summary).
        Structural: scale/event_type/ref_type/
        ref_id equality. ref_types: an INCLUDE whitelist (te.ref_type IN (...));
        None/empty = no filter (the recall_episodes caller sources its default
        whitelist from the trace_contract dial, so there's no hardcoded list
        here to drift). exclude_ref_types: NOT IN, NULL-safe (NULL ref_type is
        kept — treated as non-residue). Scope: session_id XOR session_ids (both
        raises). Time: younger_than/older_than are absolute ISO bounds; hours
        is the relative window (created_at > now-hours). chain_suffix matches
        chains ENDING in '-{suffix}' via _like_suffix_param.
        """
        if session_id and session_ids:
            raise ValueError(
                "Pass either session_id (single str) or session_ids (List[str]), "
                "not both. Got session_id=%r session_ids=%r"
                % (session_id, session_ids))
        conditions: list = []
        params: list = []
        if session_ids:
            placeholders = ','.join(['?'] * len(session_ids))
            conditions.append('te.session_id IN (%s)' % placeholders)
            params.extend(session_ids)
        elif session_id:
            conditions.append('te.session_id = ?')
            params.append(session_id)
        if scale:
            conditions.append('te.scale = ?')
            params.append(scale)
        if event_type:
            conditions.append('te.event_type = ?')
            params.append(event_type)
        if ref_type:
            conditions.append('te.ref_type = ?')
            params.append(ref_type)
        if ref_id:
            conditions.append('te.ref_id = ?')
            params.append(ref_id)
        if ref_types:
            placeholders = ','.join(['?'] * len(ref_types))
            conditions.append('te.ref_type IN (%s)' % placeholders)
            params.extend(ref_types)
        if exclude_ref_types:
            ph = ','.join(['?'] * len(exclude_ref_types))
            conditions.append(
                '(te.ref_type IS NULL OR te.ref_type NOT IN (%s))' % ph)
            params.extend(exclude_ref_types)
        if contains:
            like = '%' + contains + '%'
            conditions.append('(te.summary LIKE ? OR te.metadata LIKE ?)')
            params.extend([like, like])
        if hours is not None:
            conditions.append('te.created_at > ?')
            params.append(iso_cutoff(hours=hours))
        if younger_than:
            conditions.append('te.created_at > ?')
            params.append(younger_than)
        if older_than:
            conditions.append('te.created_at < ?')
            params.append(older_than)
        if chain_suffix:
            conditions.append("te.chain_id LIKE ? ESCAPE '\\'")
            params.append(self._like_suffix_param(chain_suffix))
        return (' AND '.join(conditions) if conditions else '1=1'), params

    def filter_events(self, contains: str = '', scale: str = '',
                      event_type: str = '', ref_types: Optional[List[str]] = None,
                      session_id: str = '', session_ids: Optional[List[str]] = None,
                      younger_than: str = '', older_than: str = '',
                      sort_order: str = 'desc',
                      limit: Optional[int] = 10) -> List[Dict[str, Any]]:
        """Structured + lexical query over trace_events — the filter_nodes
        analog for the traces layer. Returns full DECODED records (same shape
        as get_by_ids), the substance recall_episodes returns to the caller.
        The time/no-query path: indexed WHERE + ORDER BY created_at + LIMIT
        early-exits, so only `limit` rows are decoded (limit=None → all matching
        rows). See _event_where for the filter semantics (ref_types is an
        INCLUDE whitelist). Semantic ranking lives in
        BrainTracesMixin.recall_episodes (brain_traces.py).
        """
        # Honest limit (the filter_nodes analog): None → no LIMIT clause (every
        # matching event in the window — internal window pulls take all rows);
        # a number is an honest page with no silent ceiling. The agent-facing
        # cap lives at the recall_episodes dispatch door, not here. The +1 probe
        # recall_episodes runs is just limit+1, honored directly now that
        # nothing clamps it away.
        if limit is not None:
            limit = max(int(limit), 1)
        where, params = self._event_where(
            contains, scale, event_type, ref_types, session_id, session_ids,
            younger_than, older_than)
        order = 'ASC' if sort_order == 'asc' else 'DESC'
        base_sql = ('SELECT %s FROM trace_events te '
                    'WHERE %s ORDER BY te.created_at %s'
                    % (self._CANON_COLS_TE, where, order))
        if limit is None:
            rows = self.conn.execute(base_sql, params).fetchall()
        else:
            rows = self.conn.execute(
                base_sql + ' LIMIT ?', params + [limit]).fetchall()
        return [self._row_to_event(r) for r in rows]

    def filter_event_vectors(self, contains: str = '', scale: str = '',
                             event_type: str = '', ref_types: Optional[List[str]] = None,
                             session_id: str = '',
                             session_ids: Optional[List[str]] = None,
                             younger_than: str = '', older_than: str = '',
                             limit: int = 500) -> List[tuple]:
        """Lean candidate scan for semantic recall_episodes: returns
        [(trace_id, vector)] for embedded traces matching the same filter as
        filter_events, INNER JOINed to trace_embeddings (only embedded traces
        are rankable). No metadata decode and no second query — the embedding
        rides the JOIN, and only the top-k full records get hydrated (via
        get_by_ids) after ranking. Newest-first so the cap keeps recent ones.
        """
        # The RANKING POOL's own ceiling, not the agent's result ceiling —
        # they are both 500 today, and clamping with EPISODE_MAX_LIMIT here
        # meant raising EPISODE_SEMANTIC_CANDIDATE_CAP would silently do
        # nothing. Two policies, two constants, each applied where it means
        # something.
        from .brain_constants import EPISODE_SEMANTIC_CANDIDATE_CAP
        limit = min(max(int(limit), 1), EPISODE_SEMANTIC_CANDIDATE_CAP)
        where, params = self._event_where(
            contains, scale, event_type, ref_types, session_id, session_ids,
            younger_than, older_than)
        rows = self.conn.execute(
            'SELECT te.id, tem.vector FROM trace_events te '
            'JOIN trace_embeddings tem ON tem.trace_id = te.id '
            'WHERE %s ORDER BY te.created_at DESC LIMIT ?' % where,
            params + [limit]).fetchall()
        return [(r[0], r[1]) for r in rows if r[1]]

    def event_vector_rows(self, scale: str = 's0',
                          ref_types: Optional[List[str]] = None,
                          since: str = None) -> List[tuple]:
        """UNCAPPED embedded-trace pull for field consumers (LAF episodic matrix).

        Returns [(chain_id, session_id, created_at, vector, trace_id, ref_type)]
        for every embedded trace matching scale/ref_types, created_at ASC.
        trace_id is trace_events.id (the same 8-char hex get_session_turns
        returns as trace_id — the LAF moment stack joins turn→vector on it).
        `since` (exclusive ISO bound) makes refreshes incremental — callers
        keep a resident matrix and append only new rows. Deliberately separate
        from filter_event_vectors: that is the recall_episodes BROWSING scan
        (newest-first, EPISODE_MAX_LIMIT capped); this is the substrate pull
        for a scorer that must see the whole history (the newest-500 cap was a
        coverage ceiling, not a feature — 2026-07-02,
        eval/laf/composition_probe.md).
        """
        where, params = self._event_where(
            scale=scale, ref_types=ref_types, younger_than=since or '')
        conditions = ['tem.vector IS NOT NULL', where]
        rows = self.conn.execute(
            'SELECT te.chain_id, te.session_id, te.created_at, tem.vector, '
            'te.id, te.ref_type '
            'FROM trace_events te '
            'JOIN trace_embeddings tem ON tem.trace_id = te.id '
            'WHERE %s ORDER BY te.created_at ASC' % ' AND '.join(conditions),
            params).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows]

    def get_chains(self, session_id: str = '', scale: str = '',
                   hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """Get complete chains grouped, with all events and metadata.

        Returns: [{chain_id, scale, session_id, events: [{id, event_type,
        ref_type, ref_id, summary, metadata, created_at}]}] — chain_id/scale/
        session_id are chain-level; each event is the chain-relative subset
        (no chain_id/scale/session_id). Ordered by most recent chain first.
        """
        where, params = self._event_where(
            scale=scale, session_id=session_id, hours=hours)
        rows = self.conn.execute(
            'SELECT %s FROM trace_events te '
            'WHERE %s ORDER BY te.created_at DESC'
            % (self._CANON_COLS_TE, where),
            params).fetchall()

        # Group by chain_id, preserve order of first appearance.
        # A chain belongs to one session; we record the session_id from
        # the first event seen in each chain. Per-event `id` is included so a
        # grouped view stays drillable (get_trace by id); chain_id/scale/
        # session_id are chain-level — deliberately NOT repeated per event (the
        # render layer propagates them onto events). The full canonical row is
        # built once via _row_to_event, then projected to the chain-relative
        # event subset, so the decode/guard logic still lives in one place.
        EVENT_KEYS = ('id', 'event_type', 'ref_type', 'ref_id',
                      'summary', 'metadata', 'created_at')
        chains = {}
        chain_order = []
        for r in rows:
            ev = self._row_to_event(r)
            cid = ev['chain_id']
            if cid not in chains:
                chains[cid] = {'chain_id': cid, 'scale': ev['scale'],
                               'session_id': ev['session_id'], 'events': []}
                chain_order.append(cid)
            chains[cid]['events'].append({k: ev[k] for k in EVENT_KEYS})

        # Reverse events within each chain to chronological order
        for cid in chain_order:
            chains[cid]['events'].reverse()

        result = [chains[cid] for cid in chain_order[:limit]]
        return result

    @staticmethod
    def _like_suffix_param(suffix: str) -> str:
        """LIKE param matching chains ENDING in '-{suffix}', with LIKE metachars
        in the suffix escaped so a '_' in a unit name (community_detection)
        matches literally, not as a single-char wildcard. Pair with the clause
        `chain_id LIKE ? ESCAPE '\\'`."""
        esc = suffix.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        return '%-' + esc

    def get_by_ref_type(self, ref_type: str, scale: str = '',
                        hours: Optional[int] = 24, limit: int = 100,
                        session_id: str = '', ref_id: str = '',
                        chain_suffix: str = '',
                        older_than: str = '') -> List[Dict[str, Any]]:
        """Get events filtered by ref_type.

        Use: "all corrections", "all recall_hits", "all encoding_runs".
        Pass hours=None to disable the time-window filter (caller controls
        recency purely via `limit` + `ORDER BY created_at DESC`).
        Pass older_than (ISO, strict `created_at <`) to position the
        newest-first LIMIT window at a historical instant — replay as-of
        cuts belong in SQL, not a Python post-filter that the LIMIT already
        clipped.
        Pass session_id to scope results to a single session — required for
        per-session reads (e.g. surface's recently-surfaced dedup list).
        Pass ref_id to scope to a single subject (e.g. journal notes about one
        node — `ref_type='journal_note', ref_id=<node>`).
        Pass chain_suffix to scope to chains ENDING in '-{suffix}' — the S2 unit
        identity lives in the chain suffix (`s2-{ts}-{unit}`). LIMIT then bounds
        the per-unit result, not the global stream.
        """
        # Loud on a falsy ref_type: this door's contract REQUIRES one (the
        # builder would silently drop the predicate and return every type).
        # Un-typed pulls are get_recent's job.
        if not ref_type:
            raise ValueError(
                "get_by_ref_type: ref_type is required — use get_recent for "
                "un-typed pulls")
        # Unlike get_recent, hours composes WITH a session scope here — this
        # door's contract is "exactly these predicates", no authority rule.
        where, params = self._event_where(
            scale=scale, session_id=session_id, ref_type=ref_type,
            ref_id=ref_id, hours=hours, chain_suffix=chain_suffix,
            older_than=older_than)
        rows = self.conn.execute(
            'SELECT %s FROM trace_events te '
            'WHERE %s ORDER BY te.created_at DESC LIMIT ?'
            % (self._CANON_COLS_TE, where),
            params + [limit]).fetchall()

        return [self._row_to_event(r) for r in rows]

    def journal_page(self, *, scale, session_id='', unit='', after=None,
                     limit):
        """Read committed journal events in append order, with a resume cursor.

        Initial reads select the newest bounded history; incremental reads
        drain oldest-first so a limit never skips intervening resolutions.
        rowid is an internal cursor, not a trace identity: trace IDs are random
        and an append_batch gives every row the same timestamp. Cursors live
        only for an active encoder invocation, never across DB maintenance.
        """
        where, params = self._event_where(
            scale=scale, session_id=session_id, chain_suffix=unit,
            ref_type='journal_note', hours=None)
        if after is not None:
            where += ' AND te.rowid > ?'
            params.append(after)
        order = 'DESC' if after is None else 'ASC'
        rows = self.conn.execute(
            'SELECT %s, te.rowid FROM trace_events te WHERE %s '
            'ORDER BY te.rowid %s LIMIT ?' % (self._CANON_COLS_TE, where, order),
            params + [limit + 1]).fetchall()
        events = [dict(self._row_to_event(r[:-1]), journal_cursor=r[-1])
                  for r in rows[:limit]]
        return {'events': events, 'truncated': len(rows) > limit,
                'cursor': max((e['journal_cursor'] for e in events),
                              default=after or 0)}

    def count_by(self, field: str, scale: str = '', hours: int = 24) -> Dict[str, int]:
        """Count events grouped by a field.

        field: 'event_type', 'ref_type', or 'chain_id'
        Returns: {value: count} dict
        """
        allowed = {'event_type', 'ref_type', 'chain_id', 'scale'}
        if field not in allowed:
            return {}

        where, params = self._event_where(scale=scale, hours=hours)
        rows = self.conn.execute(
            'SELECT te.%s, COUNT(*) FROM trace_events te '
            'WHERE %s GROUP BY te.%s' % (field, where, field),
            params).fetchall()

        return {r[0] or '': r[1] for r in rows}

    def active_sessions_by_turn(self, cutoff_iso: str, exclude_session: str = '',
                                limit: int = 5,
                                sort_by: str = 'recency') -> List[Dict[str, Any]]:
        """Sessions reachable RIGHT NOW since `cutoff_iso`, newest first — the
        wall-clock presence signal, sourced from S0 traces.

        Counts user_message / assistant_message AND heartbeat turns. Heartbeats
        are emitted by /watch listeners on every quiet tick — a stream living
        purely in watch mode is the MOST reachable (it can be triggered to act),
        so it must count as present, not vanish after 30 min (2026-06-04).
        Session-state flushes write no traces at all, so this still reflects
        ACTUAL turns — not session_state.updated_at, which moves on boots and
        state saves that are not turns. A stream that stops taking turns of any
        kind ages out once its last turn passes the cutoff; a sid relaunched
        under a new id drops its stale sid.

        Returns [{'session_id', 'live_recency', 'focus'}] where `focus` is the
        latest CONVERSATIONAL turn — user_message OR assistant_message, per
        trace_contract.OPERATOR_DIALOGUE_REF_TYPES (not user-only): a watcher's
        last real work is often its own last reply. `focus` drops only the
        wake-envelope turn itself (a `<task-notification>` ignition), so a
        woken stream still DESCRIBES itself by its own latest report.
        `conv_recency` — the RANKING key — is stricter: it counts only
        OPERATOR-ATTENDED turns, meaning the latest user_message at-or-before
        the row lacks the marker (see _attended_sql). That asymmetry is
        deliberate. A machine-woken stream answers its own wake, and those
        replies are unmarked, so counting them as recency let a background task
        outrank a session the operator was actually working in; but the same
        reply is the most informative line available ABOUT that stream. So it
        is discounted for "how much should I care" and kept for "what is this".
        Both the conversational set and the marker come from the contract — no
        filters reproduced here. (Raw; the render layer first-lines/truncates.) Caller
        computes the cutoff (wall-clock vs conversation-time is the caller's
        policy, not the DAL's)."""
        # PINNED to operator dialogue (not the dial): presence focus and
        # ranking must show WORK even after a delivery correspondent flips on
        # in S0_CONVERSATIONAL_INCOMING — a brain notice is not a focus.
        from .trace_contract import (OPERATOR_DIALOGUE_REF_TYPES,
                                     PRESENCE_LIVE_REF_TYPES, WAKE_ENVELOPE_MARKER)
        conv_ph = ','.join('?' * len(OPERATOR_DIALOGUE_REF_TYPES))
        # TWO RECENCIES, and they are NOT interchangeable. `live_recency` spans
        # PRESENCE_LIVE_REF_TYPES (heartbeats INCLUDED) and proves the stream can
        # be REACHED — a watch listener living purely on heartbeats is the most
        # reachable stream there is (B2, 2026-06-04). `conv_recency` spans attended
        # operator dialogue and proves WORK happened. A heartbeat moves the first
        # and cannot move the second, which is exactly why a gate asking "has this
        # gone quiet" must not read live_recency: its counter (see
        # conversational_turns_since) spans the second row set, not the first.
        live_types = PRESENCE_LIVE_REF_TYPES
        live_ph = ','.join('?' * len(live_types))
        # scale='s0' is a behavior-preserving predicate (every conversational +
        # heartbeat ref_type is s0-only per trace_contract.REF_TYPES) that lets
        # the query use idx_trace_scope_created (scale, ref_type, created_at) —
        # the composite leads with scale, so without the equality it can't engage.
        # Sort key whitelist (never interpolate raw sort_by into SQL):
        #  recency (default) — most recent CONVERSATIONAL turn first, so a fresh
        #    boot (heartbeat only, conv_recency NULL → sorts last in DESC) can't
        #    crowd out a substantial stream. Membership still counts heartbeats
        #    (live_types below) so watch listeners / fresh boots stay VISIBLE,
        #    just ranked below real work.
        #  length — by number of conversational turns (user_message count).
        order = ("turn_count DESC, conv_recency DESC"
                 if sort_by == 'length'
                 else "conv_recency DESC, live_recency DESC")
        # RANKING (conv_recency) counts only OPERATOR-ATTENDED turns — that is
        # what a machine-woken stream was winning the roster on. FOCUS keeps the
        # older, weaker rule (drop the envelope, keep everything else): the
        # envelope is noise, but the reply to it is the stream's own account of
        # what it just did, and that is usually the single most informative line
        # about it. Discounting the reply for RANK and keeping it for DESCRIPTION
        # is the whole point — one of these answers "how much should I care", the
        # other "what is this". Membership (the outer WHERE) still counts
        # heartbeats and envelopes, so watch-mode streams stay VISIBLE.
        rows = self.conn.execute(
            "SELECT t.session_id, MAX(t.created_at) AS live_recency, "
            "  (SELECT u.summary FROM trace_events u "
            "   WHERE u.scale = 's0' AND u.session_id = t.session_id AND u.ref_type IN (%s) "
            "     AND u.summary NOT LIKE ? "
            "   ORDER BY u.created_at DESC LIMIT 1) AS focus, "
            "  (SELECT MAX(c.created_at) FROM trace_events c "
            "   WHERE c.scale = 's0' AND c.session_id = t.session_id AND c.ref_type IN (%s) "
            "     AND c.summary NOT LIKE ? AND %s) AS conv_recency, "
            "  (SELECT COUNT(*) FROM trace_events c2 "
            "   WHERE c2.scale = 's0' AND c2.session_id = t.session_id "
            "     AND c2.ref_type = 'user_message' AND c2.summary NOT LIKE ?) AS turn_count "
            "FROM trace_events t "
            "WHERE t.scale = 's0' AND t.ref_type IN (%s) "
            "  AND t.created_at > ? AND t.session_id != ? "
            "GROUP BY t.session_id "
            "ORDER BY %s LIMIT ?" % (conv_ph,
                                     conv_ph, _attended_sql('c'),
                                     live_ph, order),
            (*OPERATOR_DIALOGUE_REF_TYPES, WAKE_ENVELOPE_MARKER + '%',
             *OPERATOR_DIALOGUE_REF_TYPES, WAKE_ENVELOPE_MARKER + '%',
             WAKE_ENVELOPE_MARKER + '%',
             WAKE_ENVELOPE_MARKER + '%',
             *live_types, cutoff_iso, exclude_session or '', limit)).fetchall()
        return [{'session_id': r[0], 'live_recency': r[1], 'focus': r[2] or '',
                 'conv_recency': r[3] or '', 'turn_count': r[4] or 0}
                for r in rows]

    def session_activity(self, session_id: str, msg_limit: int = 2) -> Dict[str, Any]:
        """Activity snapshot for ONE session — self_peek's enrichment data.

        Returns {'started_at', 'last_active_at', 'recent_msgs'} from this
        session's S0 traces:
          started_at     — first conversational turn (MIN), tenure
          last_active_at — most recent turn of ANY live kind incl. heartbeats
                           (MAX), for liveness; a watch listener's quiet ticks
                           ARE activity (same rule as active_sessions_by_turn)
          recent_msgs    — last `msg_limit` conversational turns, newest first,
                           [{'ts','ref_type','text'}], wake-envelope turns
                           skipped so a peek shows work not ignition. The
                           stream's own REPLY to a wake is kept: it is the
                           status line a peek exists to surface. (Contrast
                           active_sessions_by_turn's conv_recency, which
                           discounts those replies — ranking and description
                           want opposite things here.) Raw; the render layer
                           truncates. Read-only.
        """
        # PINNED to operator dialogue — see active_sessions_by_turn.
        from .trace_contract import (OPERATOR_DIALOGUE_REF_TYPES,
                                     PRESENCE_LIVE_REF_TYPES, WAKE_ENVELOPE_MARKER)
        conv_ph = ','.join('?' * len(OPERATOR_DIALOGUE_REF_TYPES))
        live_types = PRESENCE_LIVE_REF_TYPES
        live_ph = ','.join('?' * len(live_types))
        agg = self.conn.execute(
            "SELECT "
            "  (SELECT MIN(created_at) FROM trace_events "
            "     WHERE scale='s0' AND session_id=? AND ref_type IN (%s)), "
            "  (SELECT MAX(created_at) FROM trace_events "
            "     WHERE scale='s0' AND session_id=? AND ref_type IN (%s)), "
            "  (SELECT COUNT(*) FROM trace_events "
            "     WHERE scale='s0' AND session_id=? AND ref_type='user_message')"
            % (conv_ph, live_ph),
            (session_id, *OPERATOR_DIALOGUE_REF_TYPES,
             session_id, *live_types, session_id)).fetchone()
        started_at = (agg[0] or '') if agg else ''
        last_active_at = (agg[1] or '') if agg else ''
        turn_count = (agg[2] or 0) if agg else 0
        # Envelope-only filter here, deliberately NOT the attendedness rule that
        # ranks the roster: a peek asks "what is this stream doing", and the
        # stream's answer to a wake IS that — "Longmem prod arm at item 6 of 10"
        # is the most useful line a peek can return. Discounting machine-woken
        # turns is a RANKING concern (active_sessions_by_turn's conv_recency);
        # applied here it would swap the live status line for a stale one.
        rows = self.conn.execute(
            "SELECT created_at, ref_type, summary FROM trace_events "
            "WHERE scale='s0' AND session_id=? AND ref_type IN (%s) "
            "  AND summary NOT LIKE ? "
            "ORDER BY created_at DESC LIMIT ?" % conv_ph,
            (session_id, *OPERATOR_DIALOGUE_REF_TYPES,
             WAKE_ENVELOPE_MARKER + '%', msg_limit)).fetchall()
        recent = [{'ts': r[0], 'ref_type': r[1], 'text': r[2] or ''} for r in rows]
        return {'started_at': started_at, 'last_active_at': last_active_at,
                'turn_count': turn_count, 'recent_msgs': recent}

    def conversational_turns_since(self, session_id: str, since_iso: str = '') -> int:
        """Count this session's conversational turns — optionally only those after
        `since_iso`. A turn == one s0 `user_message` trace, wake-envelope
        (`<task-notification>`) ignitions excluded — the same definition
        active_sessions_by_turn's turn_count uses. This is the S1 Scribe's cadence
        primitive: turns-since-last-encode, read live from traces instead of a
        maintained counter (which desynced across resume). Uses idx_trace_session
        + the scale/ref_type predicates. since_iso='' counts all turns."""
        from .trace_contract import WAKE_ENVELOPE_MARKER
        sql = ("SELECT COUNT(*) FROM trace_events WHERE scale='s0' "
               "AND session_id=? AND ref_type='user_message' AND summary NOT LIKE ?")
        params: List[Any] = [session_id, WAKE_ENVELOPE_MARKER + '%']
        if since_iso:
            sql += " AND created_at > ?"
            params.append(since_iso)
        return self.conn.execute(sql, params).fetchone()[0]

    def get_node_creation_traces(self, node_id: str) -> List[Dict[str, str]]:
        """Exact creation evidence, oldest first, across recorded sessions.

        A node_created ref_id or an encoding_run's structured `created` list
        proves creation. Mentions elsewhere and revisions do not. Return all
        matches so the caller can detect conflicting provenance; a row limit
        could hide a contradictory session. Historic malformed metadata cannot
        establish a link. This is independent of generic time-based searches.
        """
        if not node_id:
            return []
        rows = self.conn.execute(
            "SELECT session_id, created_at, ref_type, id FROM trace_events te "
            "WHERE event_type = 'delta' AND ("
            "  (ref_type = 'node_created' AND ref_id = ?) OR "
            "  (scale = 's1' AND ref_type = 'encoding_run' AND EXISTS ("
            "    SELECT 1 FROM json_each(CASE WHEN json_valid(te.metadata) "
            "      THEN te.metadata ELSE '{}' END) field, "
            "      json_each(CASE WHEN field.type = 'array' "
            "        THEN field.value ELSE '[]' END) item "
            "    WHERE field.key = 'created' AND field.type = 'array' "
            "      AND item.type = 'text' AND item.value = ?))) "
            "ORDER BY created_at ASC, rowid ASC", (node_id, node_id)).fetchall()
        return [dict(zip(('session_id', 'created_at', 'ref_type', 'id'), row))
                for row in rows]

    def get_session_turns(self, session_id: str, limit: int = 20,
                          around_timestamp: str = None,
                          before: int = None, after: int = None,
                          with_judge_output: bool = True,
                          exclude_trace_id: str = None,
                          with_surfaced: bool = False,
                          older_than: str = None,
                          include_timestamp_ties: bool = False) -> List[Dict[str, Any]]:
        """Get chronological turns for a session from S0 + S1 traces.

        Returns: [{role, ref_type, content, timestamp, trace_id, judge_output}]
        (brain_traces.py get_conversation and the encoder's
        _gather_messages derive their shapes from this — a key added here
        must be passed through get_conversation's row build too.)

        One turn per trace row — never grouped by chain. A chain can hold N
        user messages (an interrupted turn never fires Stop, so stop_counter
        never advances and the next prompt lands in the same chain); keying
        on chain would overwrite all but the last user message.

        Cross-references S1 delta (additionalContext) via recall_chain in
        metadata to fill judge_output on user turns.

        Args:
            session_id: Full session UUID
            limit: Max turns to return (most recent if no around_timestamp)
            around_timestamp: ISO timestamp to center the window on.
                If provided, returns `before` turns before + `after` turns after
                this timestamp instead of the most recent `limit`.
            before: Turns before around_timestamp (default 10)
            after: Turns after around_timestamp (default 5)
            with_judge_output: fill judge_output on user turns (one extra
                query over the window's recall chains). Callers that only
                read role/content pass False.
            with_surfaced: fill `surfaced` on user turns — the memories the
                surface selected for that turn ([{id, title}], from the s1
                surface_selected K trace's 'id8|title' detail). Consumed by
                the v13 XML surface layout as per-turn <shown> elements.
                Shares the recall-chain collection with judge_output; adds
                one query over the window's chains when enabled.
            exclude_trace_id: trace_event id of one row to drop. The
                user_message trace is written at prompt-arrival (not Stop),
                so mid-turn readers that want PREVIOUS turns only — the
                surface conversation window, prior-query embeddings — pass
                the current prompt's trace id (returned by the append).
                Keyed on the trace row, NOT the chain: after an interrupt
                the current chain also holds the previous real prompt,
                which must stay in the window. Readers that want the
                conversation as-is (Scribe, historic lookups) omit it.
            older_than: ISO timestamp, strict `created_at <` bound applied
                in SQL — positions the newest-first LIMIT window at a
                historical instant (the LAF replay as-of cut) instead of
                clipping at wall-now and post-filtering the wrong rows.
                Strict on purpose: a replay's cue row sits exactly AT as_of
                and must not enter its own window.
            include_timestamp_ties: expand historic window boundaries to keep
                whole timestamp groups. Cited context uses this to retain every
                source row without imposing a source priority on tied events.
        """
        # v29: select `id` (8-char hex trace_event.id) so callers can render
        # the encoder timeline's trace="…" attributes — the encoder copies
        # these into source_refs.
        # Conversation window = the conversational ref_types defined by the S0
        # turn-classification contract (single source of truth for what the
        # encoder reads — see trace_contract.CONVERSATIONAL_REF_TYPES).
        from .trace_contract import CONVERSATIONAL_REF_TYPES
        _refs = ",".join("?" * len(CONVERSATIONAL_REF_TYPES))
        base_sql = (
            "SELECT id, ref_type, summary, metadata, created_at "
            "FROM trace_events WHERE scale = 's0' AND session_id = ? "
            "AND event_type IN ('K', 'delta') AND ref_type IN (%s) " % _refs)
        params = (session_id, *CONVERSATIONAL_REF_TYPES)
        if exclude_trace_id:
            # In SQL (not post-filter) so the LIMIT below still returns
            # `limit` full rows of previous turns.
            base_sql += "AND id != ? "
            params = (*params, exclude_trace_id)
        if older_than:
            # Also in SQL: a post-filter after the DESC LIMIT keeps only the
            # newest rows and then discards them — a deep-history bound
            # returned zero turns while the session plainly had them.
            base_sql += "AND created_at < ? "
            params = (*params, older_than)
        if around_timestamp:
            # Historic window — needs the whole session to locate the center.
            rows = self.conn.execute(
                base_sql + "ORDER BY created_at ASC, rowid ASC",
                params).fetchall()
        else:
            # Hot path (hook_recall, once per user prompt): newest-first with
            # a SQL LIMIT so cost is O(limit), not O(session length) —
            # idx_trace_session_created walks created_at DESC and stops.
            # rowid breaks same-timestamp ties in insert order.
            rows = self.conn.execute(
                base_sql + "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (*params, limit)).fetchall()
            rows.reverse()

        # Build result in encoding_agent._gather_messages() shape —
        # one turn per row, chronological.
        turns = []
        user_refs = []  # (turn index, recall_chain) for the judge_output fill
        for r in rows:
            meta = self._decode_metadata(r[3])
            # Content lives in metadata (full), summary is truncated for display
            content = meta.get('content', '') or r[2] or ''
            # ref_type rides every turn as the CORRESPONDENT axis — who the
            # incoming side is (user_message = operator, self_message = a
            # stream, thalamus_delivery = the brain). The encoder's render
            # decides the element names; this layer only passes it through.
            if r[1] == 'assistant_message':
                turns.append({
                    'role': 'assistant',
                    'ref_type': r[1],
                    'trace_id': r[0],
                    'content': content,
                    'timestamp': r[4],
                    'judge_output': None,
                })
            else:
                # Incoming side (user_message today; self_message /
                # thalamus_delivery once dial-on in S0_CONVERSATIONAL_INCOMING)
                rc = meta.get('recall_chain', '')
                if rc and (with_judge_output or with_surfaced):
                    user_refs.append((len(turns), rc))
                turns.append({
                    'role': 'user',
                    'ref_type': r[1],
                    'trace_id': r[0],
                    'content': content,
                    'timestamp': r[4],
                    'judge_output': '',
                })

        # Cross-reference S1 traces over the window's recall chains:
        # delta/additionalContext → judge_output, K/surface_selected →
        # surfaced (each gated by its flag; the chain collection is shared).
        if user_refs:
            recall_chains = {rc for _, rc in user_refs}
            placeholders = ','.join('?' for _ in recall_chains)
            if with_judge_output:
                s1_rows = self.conn.execute(
                    "SELECT chain_id, metadata FROM trace_events "
                    "WHERE scale = 's1' AND event_type = 'delta' AND ref_type = 'additionalContext' "
                    "AND chain_id IN (%s)" % placeholders,
                    list(recall_chains)).fetchall()
                judge_outputs = {r[0]: self._decode_metadata(r[1]).get('content', '')
                                 for r in s1_rows}
                for i, rc in user_refs:
                    turns[i]['judge_output'] = judge_outputs.get(rc, '')
            if with_surfaced:
                sel_rows = self.conn.execute(
                    "SELECT chain_id, metadata FROM trace_events "
                    "WHERE scale = 's1' AND event_type = 'K' AND ref_type = 'surface_selected' "
                    "AND chain_id IN (%s)" % placeholders,
                    list(recall_chains)).fetchall()
                surfaced_by_chain = {}
                for r in sel_rows:
                    entries = []
                    for item in (self._decode_metadata(r[1]).get('selected') or []):
                        if not isinstance(item, str):
                            continue
                        sid, _, title = item.partition('|')
                        if sid:
                            entries.append({'id': sid, 'title': title})
                    # UNION per chain, not last-row-wins: after an interrupt
                    # two prompts share one s1r chain, each with its own
                    # surface_selected row — overwriting dropped one turn's
                    # selections from every consumer (<shown> render AND the
                    # seen-dedup filter). Dedup by id, order preserved.
                    prev = surfaced_by_chain.setdefault(r[0], [])
                    have = {e['id'] for e in prev}
                    prev.extend(e for e in entries if e['id'] not in have)
                for i, rc in user_refs:
                    turns[i]['surfaced'] = surfaced_by_chain.get(rc, [])

        # Apply windowing (default mode is already limited in SQL)
        if around_timestamp:
            # Find the turn closest to around_timestamp, then take window
            _before = before if before is not None else 10
            _after = after if after is not None else 5
            center_idx = 0
            for i, t in enumerate(turns):
                if t.get('timestamp', '') <= around_timestamp:
                    center_idx = i
            start = max(0, center_idx - _before * 2)  # ×2 because user+assistant = 2 turns per exchange
            end = min(len(turns), center_idx + _after * 2 + 1)
            if include_timestamp_ties and start < end:
                while start > 0 and turns[start - 1]['timestamp'] == turns[start]['timestamp']:
                    start -= 1
                while end < len(turns) and turns[end]['timestamp'] == turns[end - 1]['timestamp']:
                    end += 1
            turns = turns[start:end]

        return turns

    # --- Embeddings (v27: episodic references) ---

    def store_embeddings(self, rows: List[tuple], model: str) -> int:
        """Upsert per-trace embeddings (one row per unique trace_id).

        Args:
            rows: iterable of (trace_id, vector_blob, text). Rows with
                  vector=None are skipped.
            model: embedder model tag stored on each row.

        Returns: count of rows actually written.

        INSERT OR REPLACE handles both new inserts and updates to the
        same trace_id (e.g., re-embed after rendering change).
        """
        now = iso_now()
        prepared = []
        for trace_id, vector, text in rows:
            if vector is None or trace_id is None:
                continue
            prepared.append((trace_id, vector, (text or '')[:500], model, now))
        if not prepared:
            return 0
        with self._wlock:
            self.wconn.executemany(
                'INSERT OR REPLACE INTO trace_embeddings '
                '(trace_id, vector, text, model, created_at) '
                'VALUES (?, ?, ?, ?, ?)',
                prepared)
            commit_unless_batched(self.wconn)
        return len(prepared)

    def find_unembedded(self, limit: int, scales: List[str],
                        ref_types: List[str],
                        since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find recent trace events with no embedding row yet.

        Pull-reconciliation primitive: the worker calls this every tick,
        gets up to `limit` traces in newest-first order, embeds them.
        No queue state; restart-safe by construction.

        Args:
            limit: max rows to return.
            scales: required scale filter (e.g., ['s0']). Empty raises ValueError.
            ref_types: required ref_type filter (e.g., ['user_message',
                       'assistant_message', 'tool_result']). Empty raises
                       ValueError.
            since: optional ISO timestamp lower bound on created_at.
                   Worker uses this to skip historical traces that
                   pre-date identity stamping — they'd render with
                   OPERATOR/ANCHOR sentinels and pollute the embedding
                   neighborhood that decision 19 keeps concrete.

        Returns full canonical trace rows (id/chain_id/scale/event_type/ref_type/
        ref_id/summary/metadata/session_id/created_at). Caller renders to text
        and embeds (it reads metadata/ref_type/summary; chain_id/ref_id ride
        along for free via the shared row mapping).
        """
        if not scales:
            raise ValueError("find_unembedded: scales required")
        if not ref_types:
            raise ValueError("find_unembedded: ref_types required")
        scale_ph = ','.join('?' * len(scales))
        ref_ph = ','.join('?' * len(ref_types))
        params: List[Any] = list(scales) + list(ref_types)
        since_clause = ''
        if since:
            since_clause = ' AND te.created_at > ? '
            params.append(since)
        params.append(limit)
        rows = self.conn.execute(
            'SELECT %s FROM trace_events te '
            'LEFT JOIN trace_embeddings tem ON tem.trace_id = te.id '
            'WHERE tem.trace_id IS NULL '
            '  AND te.scale IN (%s) '
            '  AND te.ref_type IN (%s) '
            '  %s'
            'ORDER BY te.created_at DESC '
            'LIMIT ?' % (self._CANON_COLS_TE, scale_ph, ref_ph, since_clause),
            params).fetchall()
        return [self._row_to_event(r) for r in rows]


class SessionStateDAL(_LogsWriteBase):
    """Access layer for session_state table in brain_logs.db.

    Backs SessionContext's save/load/seed: each session collapses to a single
    `_session_context` JSON blob row, keyed by (session_id, key, node_id).
    get/set are the read/upsert pair; ensure_default seeds without clobbering
    a racing thread's already-mutated row.
    """

    def get(self, session_id: str, key: str, node_id: str = '') -> Optional[str]:
        """Get a single session state value."""
        row = self.conn.execute(
            "SELECT value FROM session_state WHERE session_id = ? AND key = ? AND node_id = ?",
            (session_id, key, node_id)).fetchone()
        return row[0] if row else None

    def set(self, session_id: str, key: str, value: str, node_id: str = ''):
        """Set a session state value (upsert)."""
        ts = iso_now()
        with self._wlock:
            self.wconn.execute(
                """INSERT INTO session_state (session_id, key, node_id, value, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, key, node_id)
                   DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (session_id, key, node_id, value, ts))
            commit_unless_batched(self.wconn)

    # --- session-context lifecycle (the rows keyed '_session_context') ---

    def ensure_default(self, session_id: str, key: str, value: str,
                       node_id: str = '') -> None:
        """Insert a default row only if absent (INSERT OR IGNORE).

        Distinct from set(): set() overwrites via upsert; this preserves an
        existing row, so a racing thread's already-mutated state is never
        clobbered on first touch.
        """
        with self._wlock:
            self.wconn.execute(
                'INSERT OR IGNORE INTO session_state '
                '(session_id, key, node_id, value, updated_at) VALUES (?, ?, ?, ?, ?)',
                (session_id, key, node_id, value, iso_now()))
            commit_unless_batched(self.wconn)
