"""
brain — Data Access Layer (DAL): nodes & everything else (brain.db)

Thin abstraction over SQLite tables. Each table has read/write methods.
Only this module knows which connection (brain.db vs brain_logs.db) owns
which table.

Sibling modules split out along the one structural boundary that matters
(which SQLite file owns the table, or — for GraphDAL — sheer size):
  - `dal_logs.py` — LogsDAL, InteractionDAL, TraceDAL, SessionStateDAL (brain_logs.db)
  - `dal_graph.py` — GraphDAL (edges + edge_relations, brain.db)
  - `dal_metadata.py` — MetadataDAL (node_metadata_kv, brain.db)
  - `dal_vector_cached.py` — CachedVectorDAL (wraps VectorDAL below)

This module holds the rest of brain.db: config (BrainMetaDAL), nodes
(NodeDAL), full-text search (Fts5DAL), episodic anchors
(SourceRefDAL), and embeddings (VectorDAL).

Usage in brain.py:
    from servers.dal import BrainMetaDAL, NodeDAL

    self._meta = BrainMetaDAL(self.conn)
    self._nodes = NodeDAL(self.conn)

Incrementally adoptable: brain.py can migrate one table at a time.
Direct self.conn.execute() calls continue to work alongside the DAL.
"""

import re
import sqlite3
from typing import Any, Dict, List, Optional

from .clock import iso_now

# v29 trace-id shape — the write boundary's gate in add/replace_source_refs.
_TRACE_ID_HEX = re.compile(r'[0-9a-f]{8}')
from .dal_graph import edge_context_relation_sql
from .db_backends.sqlite import commit_unless_batched


def _like_escape(s: str) -> str:
    """Escape LIKE metachars (\\, %, _) so a caller-supplied value matches
    literally under `... LIKE ? ESCAPE '\\'`. A '_' in a source like
    's2:community_detection' must not act as a single-char wildcard. Mirrors
    TraceDAL._like_suffix_param (dal_logs.py) for node-side text filters."""
    return s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


class BrainMetaDAL:
    """Access layer for brain_meta table — key-value config store."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, key: str, default: str = "") -> str:
        """Get a config value."""
        row = self.conn.execute(
            'SELECT value FROM brain_meta WHERE key = ?', (key,)
        ).fetchone()
        return row[0] if row else default

    def set(self, key: str, value: str) -> None:
        """Set a config value."""
        now = iso_now()
        self.conn.execute(
            'INSERT OR REPLACE INTO brain_meta (key, value, updated_at) VALUES (?, ?, ?)',
            (key, str(value), now)
        )
        commit_unless_batched(self.conn)


class NodeDAL:
    """Access layer for brain.db nodes table.

    ALL node SQL lives here. When we move to in-memory graph,
    swap this implementation — nothing else changes.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # --- Reads ---

    def change_key(self) -> tuple:
        """(row count, max rowid) — cheap staleness key for node-derived caches
        (LAF title-idf lane). Count catches deletions, rowid catches inserts;
        pure UPDATEs are invisible — callers that care about edits pair this
        with a TTL."""
        return tuple(self.conn.execute(
            'SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM nodes').fetchone())

    def title_rows(self) -> List[tuple]:
        """[(id, title)] for live titled nodes — the LAF title-idf substrate."""
        return self.conn.execute(
            "SELECT id, title FROM nodes "
            "WHERE archived = 0 AND title IS NOT NULL AND title != ''"
        ).fetchall()

    def created_rows(self) -> List[tuple]:
        """[(id, created_at)] for live nodes — the LAF as_of node-mask
        substrate (§20.11 read-side time travel)."""
        return self.conn.execute(
            'SELECT id, created_at FROM nodes WHERE archived = 0').fetchall()

    def project_rows(self) -> List[tuple]:
        """[(id, project)] for live nodes that carry a project — the LAF proj
        lane substrate. project is kv provenance (node_metadata_kv['project']);
        the legacy nodes.project column was dropped in schema v30."""
        return self.conn.execute(
            "SELECT k.node_id, k.value FROM node_metadata_kv k "
            "JOIN nodes n ON n.id = k.node_id "
            "WHERE k.key = 'project' AND n.archived = 0 "
            "  AND k.value IS NOT NULL AND k.value != ''"
        ).fetchall()

    def get_naked_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a single node row by ID. Returns all columns as a dict.

        This is the bare DB row — no metadata, no corrections, no connections.
        For the full assembled node, use brain.get_node().

        Column names come from the query cursor's .description (SELECT *), so
        new columns are automatically included. Boolean fields (locked,
        archived) coerced to Python bool.
        """
        cur = self.conn.execute(
            'SELECT * FROM nodes WHERE id = ?', (node_id,))
        row = cur.fetchone()
        if not row:
            return None

        # Column names from the live query cursor — no throwaway LIMIT 0 probe.
        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))

        # Coerce booleans (SQLite stores as 0/1 or NULL)
        for bool_field in ('locked', 'archived'):
            d[bool_field] = d.get(bool_field) == 1
        # Defaults for nullable fields
        d['emotion'] = d.get('emotion') or 0
        d['emotion_label'] = d.get('emotion_label') or 'neutral'
        return d

    def get_bulk(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Bulk-fetch naked node rows. Returns {node_id: row_dict}.

        Same shape as get_naked_node() per row. Missing/invalid ids are
        silently omitted from the result. Used by callers that need many
        nodes in one query (recall enrichment, correction enrichment,
        rich-node assembly) — replaces the N+1 get_naked_node loop.
        """
        if not node_ids:
            return {}
        ph = ','.join('?' * len(node_ids))
        cur = self.conn.execute(
            'SELECT * FROM nodes WHERE id IN (%s)' % ph,
            list(node_ids))
        # Column names from the live query cursor — no throwaway LIMIT 0 probe.
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            d = dict(zip(cols, row))
            for bool_field in ('locked', 'archived'):
                d[bool_field] = d.get(bool_field) == 1
            d['emotion'] = d.get('emotion') or 0
            d['emotion_label'] = d.get('emotion_label') or 'neutral'
            out[d['id']] = d
        return out

    def get_title(self, node_id: str) -> Optional[str]:
        """Get just the title of a node. Exact id match."""
        row = self.conn.execute(
            'SELECT title FROM nodes WHERE id = ?', (node_id,)
        ).fetchone()
        return row[0] if row else None

    def resolve_id_prefix(self, prefix: str) -> List[str]:
        """Node ids starting with `prefix` (lowercased), at most two — enough
        to tell one match from none or ambiguity. The connect_to resolvers
        decide what each case means."""
        rows = self.conn.execute(
            'SELECT id FROM nodes WHERE id LIKE ? LIMIT 2',
            (prefix.strip().lower() + '%',)).fetchall()
        return [r[0] for r in rows]

    def archived_subset(self, node_ids) -> set:
        """Return the subset of `node_ids` that are archived.

        Single source for liveness checks (surface selection gate).
        Exact-id match — no prefix resolution;
        unknown ids are simply absent from the result. Empty input
        returns an empty set (no `IN ()` SQL).
        """
        ids = [nid for nid in node_ids if nid]
        if not ids:
            return set()
        ph = ','.join('?' * len(ids))
        rows = self.conn.execute(
            'SELECT id FROM nodes WHERE id IN (%s) AND archived = 1' % ph,
            ids).fetchall()
        return {r[0] for r in rows}

    # --- Survivor-pointer resolution (read-only) ---

    # Metadata key recording where an archived node's content survived to.
    # Today this is the ONLY survivor source: absorb/consolidation stamps it
    # on the absorbed node before archiving. A future `absorbed_into` graph
    # edge would be a SECOND source — fold it into `_survivor_pointers_bulk`
    # below (read the edge, prefer/merge with the kv value) without touching
    # `resolve_live`'s walk.
    _SURVIVOR_META_KEY = '_sys_archived_survivor_id'

    def _live_status_bulk(self, node_ids) -> Dict[str, str]:
        """Batched `_live_status`: {id: 'live'|'archived'} for the ids that
        exist in `nodes`. Missing ids are absent from the result, which the
        resolve_live walk treats as orphan. One query for the whole frontier —
        no per-id probing on the hot path.
        """
        ids = [n for n in node_ids if n]
        if not ids:
            return {}
        ph = ','.join('?' * len(ids))
        rows = self.conn.execute(
            'SELECT id, archived FROM nodes WHERE id IN (%s)' % ph, ids
        ).fetchall()
        return {r[0]: ('archived' if r[1] == 1 else 'live') for r in rows}

    def _survivor_pointers_bulk(self, node_ids) -> Dict[str, str]:
        """Batched survivor-pointer lookup: {archived_id: survivor_id} for the
        ids that carry a pointer (others simply absent).

        SEAM: reads `_sys_archived_survivor_id` from node_metadata_kv today via
        the canonical batch getter (one IN-query, PK-covered on (node_id, key)).
        When the `absorbed_into` edge becomes the survivor source, swap the body
        HERE — `resolve_live`'s walk never changes.
        """
        ids = [n for n in node_ids if n]
        if not ids:
            return {}
        from .dal_metadata import MetadataDAL
        got = MetadataDAL(self.conn).get_fields_bulk(
            ids, [self._SURVIVOR_META_KEY])
        return {nid: kv[self._SURVIVOR_META_KEY]
                for nid, kv in got.items()
                if kv.get(self._SURVIVOR_META_KEY)}

    def resolve_live(self, ids, *, on_orphan: str = 'drop',
                     max_hops: int = None) -> Dict[str, Any]:
        """Resolve a set of node ids to their live survivors. READ-ONLY.

        For each input id: a LIVE node passes through unchanged; an ARCHIVED
        node is followed forward along its survivor pointer (see
        `_survivor_pointers_bulk`) until a live terminal, an orphan (missing
        node or no pointer), a cycle, or `max_hops` redirects. Many inputs
        collapsing to one survivor are deduped, first-seen order preserved.

        Batched walk: all in-flight inputs advance in lockstep, so the DB is
        hit twice per chain LEVEL (liveness + survivor lookup), not twice per
        id per hop. Cost is O(max chain depth) queries, independent of the
        input count — the hot-path shape the 6 history→node sites in
        docs/TRACE-NODE-RESOLUTION.md need.

        Returns ids, not hydrated nodes — callers hydrate via get_node():
            {
              'live':       [live ids, deduped, order-preserved],
              'redirected': {input_id: survivor_id},   # only redirected inputs
              'orphans':    [input ids with no live terminal],
            }

        `on_orphan='drop'` (default) returns `orphans: []`; `'mark'` returns
        the orphan input ids in `orphans`. Either way orphans never appear in
        `live`.
        """
        if max_hops is None:
            from .contract import RESOLVE_LIVE_MAX_HOPS
            max_hops = RESOLVE_LIVE_MAX_HOPS
        inputs = [i for i in (ids or []) if i]
        if not inputs:
            return {'live': [], 'redirected': {}, 'orphans': []}

        pos = {i: i for i in inputs}        # input_id -> current node in its walk
        hops = {i: 0 for i in inputs}       # redirects taken (== round reached)
        visited = {i: {i} for i in inputs}  # per-input cycle guard
        terminal: Dict[str, str] = {}       # input_id -> live terminal id
        orphaned: set = set()
        pending = list(inputs)

        while pending:
            # Round liveness: one query for every distinct current position.
            status = self._live_status_bulk({pos[i] for i in pending})
            advancers = []                  # inputs sitting on an archived node
            for i in pending:
                st = status.get(pos[i])     # None => id not in nodes (missing)
                if st == 'live':
                    terminal[i] = pos[i]
                elif st is None:
                    orphaned.add(i)         # missing node
                elif hops[i] >= max_hops:
                    orphaned.add(i)         # hop budget exhausted
                else:
                    advancers.append(i)     # archived, may still redirect
            if not advancers:
                break
            # Round survivor lookup: one query for the archived frontier.
            survivors = self._survivor_pointers_bulk(
                {pos[i] for i in advancers})
            next_pending = []
            for i in advancers:
                surv = survivors.get(pos[i])
                if not surv:
                    orphaned.add(i)         # archived, no pointer
                elif surv in visited[i]:
                    orphaned.add(i)         # cycle
                else:
                    visited[i].add(surv)
                    pos[i] = surv
                    hops[i] += 1
                    next_pending.append(i)
            pending = next_pending

        # Assemble in first-seen input order; dedup live; mark redirects.
        live_out: List[str] = []
        seen_live: set = set()
        redirected: Dict[str, str] = {}
        for i in inputs:
            t = terminal.get(i)
            if t is None:
                continue
            if hops[i] > 0:
                redirected[i] = t
            if t not in seen_live:
                seen_live.add(t)
                live_out.append(t)

        return {
            'live': live_out,
            'redirected': redirected,
            'orphans': [i for i in inputs if i in orphaned]
                       if on_orphan == 'mark' else [],
        }

    def survivor_of(self, node_id: str) -> Optional[str]:
        """Live terminal for an ABSORBED node id, else None (live, retired,
        orphaned, or missing all return None). One-id convenience over
        resolve_live for write-door refusal messages — writes never redirect,
        they point the producer at the survivor and refuse."""
        if not node_id:
            return None
        return self.resolve_live([node_id])['redirected'].get(node_id)

    def count(self, archived: bool = False) -> int:
        """Count nodes, optionally excluding archived."""
        if archived:
            row = self.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()
        else:
            row = self.conn.execute(
                'SELECT COUNT(*) FROM nodes WHERE archived = 0'
            ).fetchone()
        return row[0] if row else 0

    def count_locked(self, include_archived: bool = False) -> int:
        """Count locked nodes. Excludes archived by default (the
        identity-meaningful count); pass include_archived=True for the raw
        lock count regardless of archive state.

        Default (non-archived) matches all current call sites — brain.py,
        brain_assembly.py, and daemon_server's status count (migrated 2026-05-30,
        which intentionally drops archived-locked nodes from the status total).
        Pass include_archived=True for the raw all-state lock count."""
        sql = 'SELECT COUNT(*) FROM nodes WHERE locked = 1'
        if not include_archived:
            sql += ' AND archived = 0'
        row = self.conn.execute(sql).fetchone()
        return row[0] if row else 0

    def nursery_gate_stats(self, age_cutoff: str) -> Dict[str, Any]:
        """The Zero-Memory boot gate's three facts (brain_assembly.boot_stats).

        lived: live nodes NOT from the seed pack (COALESCE guards legacy
        NULL encoding_source rows out of silent exclusion). born: any seed
        node exists at all — deliberately ignores archived, so archiving the
        seeds doesn't erase the birth record. old_enough: a seed node exists
        older than age_cutoff (an ISO-T string — always bind one from
        clock.iso_cutoff, per the time-window contract)."""
        lived = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived = 0 "
            "AND COALESCE(encoding_source, '') != 'anchor:seed'"
        ).fetchone()[0]
        born = self.conn.execute(
            "SELECT 1 FROM nodes WHERE encoding_source = 'anchor:seed' LIMIT 1"
        ).fetchone() is not None
        old_enough = self.conn.execute(
            "SELECT 1 FROM nodes WHERE encoding_source = 'anchor:seed' "
            "AND created_at < ? LIMIT 1", (age_cutoff,)
        ).fetchone() is not None
        return {'lived': lived, 'born': born, 'old_enough': old_enough}

    def count_by_type(self, node_type: str) -> int:
        """Count nodes of a specific type."""
        row = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE type = ? AND archived = 0',
            (node_type,)
        ).fetchone()
        return row[0] if row else 0

    def filter_nodes(self, field: str, include=None, exclude=None,
                     lt=None, gt=None, contains=None, prefix=None, limit: int = 50,
                     sort_by: str = 'created_at', sort_order: str = 'desc'):
        """Filter nodes by any structural OR promoted-kv field.

        Args:
            field: a nodes-table column (STRUCTURAL_FIELDS) or any
                metadata_kv key — promoted keys accepted directly, other
                keys accepted when any node stores them (else the error
                carries `unknown_field: True`). kv keys are matched via a
                node_metadata_kv subquery with the key BOUND as a parameter.
            include: list of values to match (exact, IN).
            exclude: list of values to exclude (exact, NOT IN).
            lt/gt: numeric comparisons — structural fields only.
            contains: substring match on `field` (LIKE '%value%'). Works on
                structural and kv fields. Same operator vocabulary as recall's
                _apply_filter dict-filter (brain_recall.py); LIKE metachars in
                the value are escaped so they match literally.
            prefix: prefix match on `field` (LIKE 'value%'), escaped as above.
            limit: max results. None → unbounded (all matches — an honest
                id-set extraction). A number is an honest page with NO silent
                ceiling: the caller's page is the caller's. The agent-facing
                cap lives at the dispatch boundary (NODE_QUERY_MAX_LIMIT), not
                here — a skinny internal scan is not a context-flood risk.
            sort_by: column to sort by.
            sort_order: 'asc' or 'desc'.

        Returns: dict with 'nodes' list and 'total_count'.
        """
        from .contract import STRUCTURAL_FIELDS, PROMOTED_FIELDS

        # kv_field: any metadata_kv key (promoted or free-form) — matched via
        # node_metadata_kv, not a nodes column. The key is BOUND as a query
        # parameter everywhere below, so no whitelist is needed for safety;
        # unknown keys still error loudly (a typo'd key must not read as a
        # measured zero) via a cheap existence probe, with the promoted set
        # accepted without probing.
        _kv_fields = {k for k, v in PROMOTED_FIELDS.items()
                      if v.get('store') == 'metadata_kv'}
        kv_field = field not in STRUCTURAL_FIELDS
        if kv_field and field not in _kv_fields:
            known = self.conn.execute(
                'SELECT 1 FROM node_metadata_kv WHERE key = ? LIMIT 1',
                (field,)).fetchone()
            if not known:
                # unknown_field lets a programmatic caller distinguish "no
                # node stores this key" (a legitimate empty for a sweep over
                # optional kv keys) from a malformed query — without parsing
                # the message.
                return {"error": "Unknown field '%s' — not a nodes column and "
                                 "no node stores it as a kv key. Structural: "
                                 "%s; promoted kv: %s"
                                 % (field, ', '.join(sorted(STRUCTURAL_FIELDS)),
                                    ', '.join(sorted(_kv_fields))),
                        "unknown_field": True}
        if kv_field and (lt is not None or gt is not None):
            return {"error": "lt/gt not supported for metadata field '%s' "
                             "(text-valued)" % field}

        # Whitelist sort_by
        allowed_sort = {'created_at', 'confidence', 'access_count', 'title', 'type',
                        'updated_at', 'last_accessed', 'revised_at'}
        if sort_by not in allowed_sort:
            sort_by = 'created_at'
        if sort_order not in ('asc', 'desc'):
            sort_order = 'desc'

        # None → unbounded (no LIMIT clause); a number is an honest floor-of-1
        # page with no silent ceiling. Flooding is bounded at the agent door
        # (dispatch), not here — see the `limit` arg note above.
        if limit is not None:
            limit = max(int(limit), 1)

        # Build WHERE clauses
        conditions = ['archived = 0']
        params = []

        if include and exclude:
            return {"error": "Cannot use both include and exclude"}

        if include:
            placeholders = ','.join('?' for _ in include)
            if kv_field:
                conditions.append(
                    "id IN (SELECT node_id FROM node_metadata_kv "
                    "WHERE key = ? AND value IN (%s))" % placeholders)
                params.append(field)
            else:
                conditions.append('%s IN (%s)' % (field, placeholders))
            params.extend(include)
        elif exclude:
            placeholders = ','.join('?' for _ in exclude)
            if kv_field:
                conditions.append(
                    "id NOT IN (SELECT node_id FROM node_metadata_kv "
                    "WHERE key = ? AND value IN (%s))" % placeholders)
                params.append(field)
            else:
                conditions.append('%s NOT IN (%s)' % (field, placeholders))
            params.extend(exclude)

        if lt is not None:
            conditions.append('%s < ?' % field)
            params.append(lt)
        if gt is not None:
            conditions.append('%s > ?' % field)
            params.append(gt)

        # Text ops (contains/prefix): a LIKE on `field`, kv-routed the same way
        # include/exclude are. Additive (AND) with the clauses above. The
        # metachars in the value are escaped so 's2:' or 'a_b' match literally.
        for op_val, wrap in ((contains, '%%%s%%'), (prefix, '%s%%')):
            if not op_val:
                continue  # empty == not-provided, matching include/exclude
            like = wrap % _like_escape(str(op_val))
            if kv_field:
                conditions.append(
                    "id IN (SELECT node_id FROM node_metadata_kv "
                    "WHERE key = ? AND value LIKE ? ESCAPE '\\')")
                params.append(field)
            else:
                conditions.append("%s LIKE ? ESCAPE '\\'" % field)
            params.append(like)

        where = ' AND '.join(conditions)

        # Count total matches
        count_row = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE %s' % where, params
        ).fetchone()
        total_count = count_row[0] if count_row else 0

        # kv fields select their value via a correlated subquery with the key
        # BOUND — its placeholder appears in the SELECT clause, i.e. before the
        # WHERE placeholders in bind order; structural fields select direct.
        field_select = field
        select_params: List[Any] = []
        if kv_field:
            field_select = ("(SELECT value FROM node_metadata_kv "
                            "WHERE node_id = nodes.id AND key = ?)")
            select_params = [field]
        base_sql = 'SELECT id, title, type, confidence, created_at, %s FROM nodes WHERE %s ORDER BY %s %s' % (
            field_select, where, sort_by, sort_order)
        if limit is None:
            rows = self.conn.execute(base_sql, select_params + params).fetchall()
        else:
            rows = self.conn.execute(base_sql + ' LIMIT ?',
                                     select_params + params + [limit]).fetchall()

        nodes = []
        for r in rows:
            node = {'id': r[0], 'title': r[1], 'type': r[2],
                    'confidence': r[3], 'created_at': r[4]}
            if field not in ('id', 'title', 'type', 'confidence', 'created_at'):
                node[field] = r[5]
            nodes.append(node)

        return {"nodes": nodes, "total_count": total_count}

    # --- Writes ---

    def set_locked(self, node_id: str, locked: bool) -> None:
        """Flip the locked flag. Sole caller: Brain.set_node_lock — the
        two-phase confirmed lock door. revise() keeps treating locked as
        immutable; nothing else writes this column.

        Deliberately does NOT bump updated_at: a lock flip is metadata, not
        content, and updated_at is the S2 community delta-gate wake signal
        and a recency-ordering key — a flip must trigger neither. The
        node_lock_changed trace is the record of when it happened."""
        self.conn.execute(
            'UPDATE nodes SET locked = ? WHERE id = ?',
            (1 if locked else 0, node_id))
        commit_unless_batched(self.conn)

    # No field write-helpers here: revise() is the only content path
    # (b2f97fb1) — content/title/confidence go through
    # brain_remember's revise (locked via set_locked above), access via
    # recall_write_queue's drain.


class Fts5DAL:
    """Access layer for nodes_fts (FTS5 full-text search).

    FTS5 provides word-level search alongside embedding similarity.
    Different signal: embeddings match meaning, FTS5 matches words.
    Both feed into the surfacer which decides relevance.
    """

    def __init__(self, conn):
        self.conn = conn

    def search(self, query: str, limit: int = 30,
               include_archived: bool = False,
               prefix: bool = False, column: str = '',
               min_token_len: int = 2) -> List[str]:
        """Full-text search. Returns node_ids ranked by BM25 relevance.

        Title matches weighted 10x over content.
        bm25() column weights: (node_id=0, title=10, content=1)

        Excludes archived nodes by default. FTS5 (nodes_fts) is a separate
        virtual table with no `archived` column — historically the ONE recall
        candidate lane that didn't filter liveness, so a lingering FTS entry for
        an archived node surfaced it in recall (the dead-node leak; see
        docs/TRACE-NODE-RESOLUTION.md). JOINing `nodes` and filtering
        `archived = 0` makes the flag the single source of truth at READ time —
        so the FTS-delete on archive becomes hygiene, not a correctness
        requirement, and `LIMIT` now returns live hits instead of spending
        slots on dead ones. The survivor-redirect reader passes
        include_archived=True to SEE an archived hit and resolve_live it to its
        living survivor rather than drop it.

        """
        safe_query = self._sanitize_query(query, prefix=prefix, column=column,
                                          min_token_len=min_token_len)
        if not safe_query:
            return []
        try:
            if include_archived:
                sql = """SELECT node_id FROM nodes_fts
                         WHERE nodes_fts MATCH ?
                         ORDER BY bm25(nodes_fts, 0, 10.0, 1.0)
                         LIMIT ?"""
            else:
                sql = """SELECT nodes_fts.node_id FROM nodes_fts
                         JOIN nodes ON nodes.id = nodes_fts.node_id
                         WHERE nodes_fts MATCH ? AND nodes.archived = 0
                         ORDER BY bm25(nodes_fts, 0, 10.0, 1.0)
                         LIMIT ?"""
            rows = self.conn.execute(sql, (safe_query, limit)).fetchall()
            return [r[0] for r in rows]
        except Exception as e:
            # Loud-by-default: a malformed query or corrupt FTS5 index must not
            # look identical to "no matches" — FTS5 is one of two recall signals.
            # Log before degrading (matches add_relation's de-silenced pattern).
            import sys as _sys
            print('[Fts5DAL.search] FTS5 query failed (%r): %s'
                  % (safe_query, e), file=_sys.stderr)
            return []

    def upsert(self, node_id: str, title: str, content: str):
        """Insert or update a node in the FTS5 index."""
        self.delete(node_id)
        self.conn.execute(
            "INSERT INTO nodes_fts (node_id, title, content) VALUES (?, ?, ?)",
            (node_id, title, content or ''))

    def delete(self, node_id: str):
        """Remove a node from FTS5 index."""
        try:
            self.conn.execute(
                "DELETE FROM nodes_fts WHERE node_id = ?", (node_id,))
        except Exception as e:
            # Loud-by-default: a failed delete leaves a stale index entry.
            # Lower stakes than search (self-heals on next upsert) but log
            # rather than swallow silently.
            import sys as _sys
            print('[Fts5DAL.delete] FTS5 delete failed for %s: %s'
                  % (node_id, e), file=_sys.stderr)

    @staticmethod
    def _sanitize_query(query: str, prefix: bool = False,
                        column: str = '', min_token_len: int = 2) -> str:
        """Sanitize query for FTS5 MATCH syntax.

        Wraps each meaningful term in quotes, joins with OR.
        Caps at 8 terms to prevent explosion.

        prefix=True appends the FTS5 prefix operator to each term
        (`"conf"*` matches 'configuration') — used by title-match candidate
        generation so a partial-word query token still finds its titles.

        column scopes every term to one indexed column (`title:"conf"*`) —
        title-match candidate generation uses it so nodes that merely MENTION
        a probe token in content can't enter (or crowd) the pool. Without it,
        bm25 ranking + LIMIT makes the pool a relevance cut over
        title-and-content matches, which broke the pigeonhole recall
        guarantee (bug 69c2cbab #1).
        """
        from .brain_constants import STOP_WORDS
        words = query.strip().split()
        # min_token_len=2 (default) drops single-char noise for general
        # search. The title-match door passes 1: its probes are chosen by
        # the pigeonhole argument, and a silently-dropped probe is a DEAD
        # probe that shrinks the recall guarantee (bug 69c2cbab #4).
        terms = [w for w in words if w.lower() not in STOP_WORDS
                 and len(w) >= min_token_len]
        if not terms:
            terms = [w for w in words if len(w) >= min_token_len]
        if not terms:
            return ''
        star = '*' if prefix else ''
        col = '%s:' % column if column else ''
        # Quote each term, join with OR (any match, not all)
        return ' OR '.join('%s"%s"%s' % (col, t.replace('"', ''), star)
                           for t in terms[:8])


class SourceRefDAL:
    """Access layer for `node_source_refs` — node→trace_event pointers (v29:
    8-char hex) anchoring a node to the S0 moments it came from (episodic
    references). Extracted from GraphDAL: source_refs are NOT edges — they're
    the engram-cohort substrate (get_nodes_referencing). One row per
    (node_id, trace_id, position). Writers gate on conn.in_batch like every
    other DAL writer."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add_source_refs(self, node_id: str, trace_ids: List[str]) -> int:
        """Append trace_event pointers to a node (v29: 8-char hex strings).
        Used at NEW-node creation (remember()). Position derived from list
        order (1-indexed); first ref is the primary anchor.

        INSERT OR IGNORE — first-write-wins. Re-calling with the same refs
        is a no-op. For revise() use replace_source_refs() instead — that's
        where field-level replace semantics belong (decision 995ffeb1).

        Reject int input loudly per the v29 contract — coercion was removed
        because random hex generation made it unsafe.

        Returns count of refs newly inserted (existing ignored).
        """
        if not node_id or not trace_ids:
            return 0
        # Reject int input loudly — v29 contract is hex strings end-to-end.
        # Shape gate: non-hex refs (leftover example placeholders like
        # "{trace-...}") used to store silently and point at no moment —
        # loud at the write boundary, same contract as the int rejection.
        for tid in trace_ids:
            if not isinstance(tid, str):
                raise ValueError(
                    "add_source_refs: trace_ids must be strings, got "
                    "%s (%r). v29 trace ids are 8-char hex." % (
                        type(tid).__name__, tid))
            if not _TRACE_ID_HEX.fullmatch(tid):
                raise ValueError(
                    "add_source_refs: %r is not an 8-char hex trace id — "
                    "placeholders and malformed refs store silently and "
                    "resolve to no moment; copy the id from the input's "
                    "trace markers." % (tid,))
        now = _now()
        rows = [(node_id, tid, idx + 1, now)
                for idx, tid in enumerate(trace_ids)]
        cur = self.conn.executemany(
            'INSERT OR IGNORE INTO node_source_refs '
            '(node_id, trace_id, position, created_at) '
            'VALUES (?, ?, ?, ?)',
            rows)
        commit_unless_batched(self.conn)
        return cur.rowcount

    def replace_source_refs(self, node_id: str, trace_ids: List[str]) -> int:
        """Replace the node's source_refs with the given list. v29: 8-char hex.

        Per the unified 2-class revise contract (decision 995ffeb1):
        - field present → REPLACE entire value
        - field absent → preserve (caller decides whether to call this)
        Called only by revise() when source_refs is in the update payload.
        Atomic: DELETE old rows, then INSERT new ones in a single transaction.

        Pass empty list to clear all refs. Returns count inserted.
        """
        if not node_id:
            return 0
        # Reject int input loudly — v29 contract is hex strings end-to-end.
        # Coercion was removed (reviewer F2) — silent int→hex was unsafe
        # against random hex generation; loud rejection is the doctrine.
        for tid in (trace_ids or []):
            if not isinstance(tid, str):
                raise ValueError(
                    "replace_source_refs: trace_ids must be strings, got "
                    "%s (%r). v29 trace ids are 8-char hex." % (
                        type(tid).__name__, tid))
            if not _TRACE_ID_HEX.fullmatch(tid):
                raise ValueError(
                    "replace_source_refs: %r is not an 8-char hex trace id — "
                    "placeholders and malformed refs store silently and "
                    "resolve to no moment; copy the id from the input's "
                    "trace markers." % (tid,))
        now = _now()
        self.conn.execute(
            'DELETE FROM node_source_refs WHERE node_id = ?', (node_id,))
        rows = [(node_id, tid, idx + 1, now)
                for idx, tid in enumerate(trace_ids or [])]
        if rows:
            self.conn.executemany(
                'INSERT INTO node_source_refs '
                '(node_id, trace_id, position, created_at) '
                'VALUES (?, ?, ?, ?)',
                rows)
        commit_unless_batched(self.conn)
        return len(rows)

    def get_source_refs(self, node_id: str) -> List[str]:
        """Trace ids anchoring this node, ordered by encoder-written
        position (primary first). v29: returns 8-char hex strings."""
        if not node_id:
            return []
        rows = self.conn.execute(
            'SELECT trace_id FROM node_source_refs '
            'WHERE node_id = ? ORDER BY position ASC',
            (node_id,)).fetchall()
        return [r[0] for r in rows]

    def get_source_refs_bulk(self, node_ids) -> Dict[str, List[str]]:
        """Trace ids per node for a set of nodes, each list in encoder-written
        position order — one query for the whole set. Nodes without refs are
        absent from the result (callers `.get(id, [])`). v29: 8-char hex."""
        ids = [i for i in dict.fromkeys(node_ids or []) if i]
        if not ids:
            return {}
        placeholders = ','.join('?' for _ in ids)
        rows = self.conn.execute(
            'SELECT node_id, trace_id FROM node_source_refs '
            'WHERE node_id IN (%s) ORDER BY node_id, position ASC' % placeholders,
            ids).fetchall()
        out: Dict[str, List[str]] = {}
        for node_id, trace_id in rows:
            out.setdefault(node_id, []).append(trace_id)
        return out

    def get_nodes_referencing(self, trace_id: str) -> List[str]:
        """All node_ids anchored to a given trace (v29: 8-char hex). Engram
        cohort detection primitive — nodes that share a trace are part of
        the same memory at the substrate level. Rejects int input loudly."""
        if trace_id is None:
            return []
        if not isinstance(trace_id, str):
            raise ValueError(
                "get_nodes_referencing: trace_id must be a string, got "
                "%s (%r). v29 trace ids are 8-char hex." % (
                    type(trace_id).__name__, trace_id))
        rows = self.conn.execute(
            'SELECT node_id FROM node_source_refs WHERE trace_id = ?',
            (trace_id,)).fetchall()
        return [r[0] for r in rows]


def _now() -> str:
    """UTC ISO timestamp for edge operations."""
    return iso_now()


class VectorDAL:
    """Unified access layer for all node vectors (node_enrichments table, v23+).

    After v23 migration, ALL vectors live in node_enrichments with vector_type:
      _primary    — title+content blend (was in node_embeddings)
      _situation  — situation embedding (was in node_embeddings.situation_embedding)
                    NOTE: text column is DEPRECATED for _situation rows —
                    kv is canonical (see contract.py PROMOTED_FIELDS.situation).
                    Callers should pass empty string for text when storing _situation.
      title       — title-only diagnostic pointer
      high_meta   — situation + quotes
      other_meta  — reasoning + correction_pattern
      edge_context — edge descriptions
      question    — the "what would someone ask" field
      content, reasoning, their_raw_quote, my_raw_quote — per-field vectors
                    (pipeline_contract EMBEDDING_GROUPS, field cohort)
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def store(self, node_id: str, vector_type: str, text: str,
              embedding: Optional[bytes], model: str) -> None:
        """Store or replace a single vector for a node.

        `model` is the name of the embedder that produced the vector — the
        stamp recall filters on and the coverage sweep re-embeds by. Callers
        pass it from the live embedder; there is no default to go stale.

        Uses deterministic ID '{node_id}__{vector_type}' for INSERT OR REPLACE.
        For bulk writes, prefer store_batch() — one round-trip instead of N.
        """
        vid = '%s__%s' % (node_id, vector_type)
        now = iso_now()
        try:
            self.conn.execute(
                '''INSERT OR REPLACE INTO node_enrichments
                   (id, node_id, vector_type, text, embedding, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (vid, node_id, vector_type, text[:500] if text else '',
                 embedding, model, now))
        except Exception as e:
            import sys
            print('[VectorDAL] store error for %s/%s: %s' % (
                node_id[:12], vector_type, e), file=sys.stderr)
            return
        # Commit OUTSIDE the try: a COMMIT failure must propagate loud, never
        # be swallowed by the insert's stderr-catch (which would leave an
        # uncommitted row + the cache updated ahead of the DB).
        commit_unless_batched(self.conn)

    def store_batch(self, rows, model: str) -> int:
        """Batch insert-or-update many vectors in one executemany round-trip.

        Args:
            rows: iterable of (node_id, vector_type, text, embedding_blob).
                  Rows with embedding=None are skipped.
            model: model tag stored on each row.

        Returns: count of rows actually written.

        INSERT OR REPLACE handles both new inserts and updates to existing
        (node_id, vector_type) rows via the deterministic id key.
        """
        now = iso_now()
        prepared = []
        for node_id, vector_type, text, blob in rows:
            if blob is None or not node_id or not vector_type:
                continue
            vid = '%s__%s' % (node_id, vector_type)
            prepared.append((vid, node_id, vector_type,
                             text[:500] if text else '',
                             blob, model, now))
        if not prepared:
            return 0
        try:
            self.conn.executemany(
                '''INSERT OR REPLACE INTO node_enrichments
                   (id, node_id, vector_type, text, embedding, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                prepared)
        except Exception as e:
            import sys
            print('[VectorDAL] store_batch error (%d rows): %s' % (len(prepared), e),
                  file=sys.stderr)
            return 0
        # Commit outside the try — a COMMIT failure propagates loud (see store).
        commit_unless_batched(self.conn)
        return len(prepared)

    def get_all_with_context(self, exclude_archived: bool = True,
                             types: List[str] = None,
                             model: str = None) -> List[Dict[str, Any]]:
        """Get all primary embeddings with node context for recall STEP 3 scan.

        When `model` is given, only vectors produced by that model are returned.
        Stale-model rows are invisible — prevents cosine noise after a swap.

        Project is kv provenance, scored by the LAF proj lane / filtered via
        the dict filter — not a scan pre-filter.
        """
        where = ["ne.vector_type = '_primary'"]
        params: List[Any] = []
        if exclude_archived:
            where.append('n.archived = 0')
        if types:
            where.append('n.type IN (%s)' % ','.join('?' * len(types)))
            params.extend(types)
        if model:
            where.append('ne.model = ?')
            params.append(model)
        where_sql = ' WHERE ' + ' AND '.join(where)
        rows = self.conn.execute(
            'SELECT ne.node_id, ne.embedding, n.personal, n.personal_context '
            'FROM node_enrichments ne '
            'JOIN nodes n ON n.id = ne.node_id' + where_sql,
            params).fetchall()
        return [{'node_id': r[0], 'embedding': r[1], 'personal': r[2],
                 'personal_context': r[3]}
                for r in rows]

    def get_all_vectors(self, exclude_archived: bool = True,
                        vector_types: Optional[List[str]] = None,
                        model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get vectors for unified recall scan, optionally filtered.

        Args:
            exclude_archived: skip archived nodes (default True)
            vector_types: restrict to these types, e.g. ['_primary']. None = all.
            model: restrict to rows produced by this model. None = all.

        Returns: [{node_id, vector_type, embedding}] for rows with non-null embeddings.
        """
        sql = ('SELECT ne.node_id, ne.vector_type, ne.embedding '
               'FROM node_enrichments ne '
               'JOIN nodes n ON n.id = ne.node_id '
               'WHERE ne.embedding IS NOT NULL')
        params: List[Any] = []
        if exclude_archived:
            sql += ' AND n.archived = 0'
        if vector_types:
            ph = ','.join('?' * len(vector_types))
            sql += f' AND ne.vector_type IN ({ph})'
            params.extend(vector_types)
        if model:
            sql += ' AND ne.model = ?'
            params.append(model)
        rows = self.conn.execute(sql, params).fetchall()
        return [{'node_id': r[0], 'vector_type': r[1], 'embedding': r[2]}
                for r in rows]

    def change_key(self) -> tuple:
        """(row count, max rowid) on node_enrichments — cheap staleness key for
        vector-derived caches (LAF field matrices). INSERT OR REPLACE bumps
        rowid, so replaced vectors are visible; count-shrink signals deletion
        (caller falls back to a full rebuild)."""
        return tuple(self.conn.execute(
            'SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM node_enrichments'
        ).fetchone())

    def vectors_since(self, rowid: int,
                      vector_types: Optional[List[str]] = None,
                      model: Optional[str] = None) -> List[tuple]:
        """[(rowid, node_id, vector_type, embedding)] for enrichment rows with
        rowid > watermark, live nodes only — the LAF incremental matrix append.
        INSERT OR REPLACE gives updated vectors a fresh rowid, so upserts ride
        the same watermark. Ordered by rowid so the caller's last-seen tracking
        is monotone."""
        sql = ('SELECT ne.rowid, ne.node_id, ne.vector_type, ne.embedding '
               'FROM node_enrichments ne '
               'JOIN nodes n ON n.id = ne.node_id '
               'WHERE ne.rowid > ? AND ne.embedding IS NOT NULL '
               'AND n.archived = 0')
        params: List[Any] = [int(rowid)]
        if vector_types:
            ph = ','.join('?' * len(vector_types))
            sql += f' AND ne.vector_type IN ({ph})'
            params.extend(vector_types)
        if model:
            sql += ' AND ne.model = ?'
            params.append(model)
        return self.conn.execute(sql + ' ORDER BY ne.rowid', params).fetchall()

    def find_missing(self, vector_type: str, limit: int = 50,
                     model: Optional[str] = None,
                     node_ids: Optional[set] = None,
                     require_kv_keys_any: Optional[List[str]] = None,
                     source_kv_keys: Optional[List[str]] = None,
                     require_described_edge: bool = False,
                     exclude_relations=()) -> List[Dict[str, Any]]:
        """Find active nodes whose vector for `vector_type` is missing or stale.

        A row is "present" only if it has a non-null embedding AND (if `model`
        is given) was produced by the same model. On model swaps, rows
        embedded by prior models become eligible for re-embedding.

        When `node_ids` is given, scope the scan to just those IDs (queue
        drain path — don't re-scan the whole graph on every tick).

        When `require_kv_keys_any` is given, restrict to nodes that have at
        least one of those keys present in node_metadata_kv with a non-empty
        value. **OR semantics** — node passes if ANY listed key matches; the
        list is *required-any-of*, not *all-of-these*. This prevents the
        field-cohort backfill from filling its top-N batch with nodes that
        lack the source field — older nodes-with-the-field would otherwise
        be stuck below the LIMIT cutoff and never embedded.

        `source_kv_keys` is an accepted alias for `require_kv_keys_any`
        (the prior name was misleading — sounded like "the keys that ARE
        the source" rather than "keys that must exist"). Either kwarg works;
        if both are provided, `require_kv_keys_any` wins.

        Returns [{id, title, content}] ordered by recency of access.
        """
        # Resolve alias: prefer the new explicit name; fall back to the old.
        if require_kv_keys_any is None:
            require_kv_keys_any = source_kv_keys
        where = ['n.archived = 0']
        params: list = []

        if model:
            where.append('''n.id NOT IN (
                SELECT ne.node_id FROM node_enrichments ne
                WHERE ne.vector_type = ?
                  AND ne.embedding IS NOT NULL
                  AND ne.model = ?
            )''')
            params.extend([vector_type, model])
        else:
            where.append('''n.id NOT IN (
                SELECT ne.node_id FROM node_enrichments ne
                WHERE ne.vector_type = ? AND ne.embedding IS NOT NULL
            )''')
            params.append(vector_type)

        if node_ids:
            ids = list(node_ids)
            ph = ','.join('?' * len(ids))
            where.append('n.id IN (%s)' % ph)
            params.extend(ids)

        if require_kv_keys_any:
            ph = ','.join('?' * len(require_kv_keys_any))
            where.append(
                'EXISTS (SELECT 1 FROM node_metadata_kv kv '
                'WHERE kv.node_id = n.id AND kv.key IN (%s) '
                # trim() mirrors the text-builder's `val.strip()` — a
                # whitespace-only value is NOT eligible (it yields no embed
                # text), so it neither clogs the batch nor false-trips the
                # dead-handler alarm. Keeps "eligible <=> yields text" exact.
                "AND kv.value IS NOT NULL AND trim(kv.value) != '')" % ph)
            params.extend(require_kv_keys_any)

        if require_described_edge:
            # edge_context group: its only source is _edge_descriptions, which
            # lives on edges, not node_metadata_kv — so require_kv_keys_any can't
            # gate it. Without this clause the edgeless nodes (no described edge,
            # never get a vector) sit at the front of the last_accessed queue
            # forever and starve the edged nodes. Same fragment as the text
            # producer (edge_context_relation_sql) so "eligible" ⇔ "yields text";
            # `exclude_relations` is the noise set the caller reads from aspects.
            frag, frag_params = edge_context_relation_sql(exclude_relations, 'er')
            where.append(
                'EXISTS (SELECT 1 FROM edges e '
                'JOIN edge_relations er ON er.edge_id = e.edge_id '
                'WHERE (e.source_id = n.id OR e.target_id = n.id) '
                'AND er.archived = 0 AND ' + frag + ')')
            params.extend(frag_params)

        sql = ('SELECT n.id, n.title, n.content FROM nodes n '
               'WHERE ' + ' AND '.join(where) +
               ' ORDER BY n.last_accessed DESC LIMIT ?')
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [{'id': r[0], 'title': r[1] or '', 'content': r[2] or ''} for r in rows]

    def delete_for_node(self, node_id: str, vector_types=None) -> int:
        """Delete a node's enrichment vectors — all rows (archive/delete
        path), or only the given vector_types (revise invalidation, per
        pipeline_contract.vectors_affected_by). THE one deletion path for
        node_enrichments; no caller raw-SQLs this table (raw-SQL ratchet
        enforces). Returns rows deleted."""
        if vector_types is not None:
            vts = list(vector_types)
            if not vts:
                return 0
            ph = ','.join('?' * len(vts))
            self.conn.execute(
                'DELETE FROM node_enrichments WHERE node_id = ? '
                'AND vector_type IN (%s)' % ph, [node_id, *vts])
        else:
            self.conn.execute(
                'DELETE FROM node_enrichments WHERE node_id = ?', (node_id,))
        n = self.conn.execute('SELECT changes()').fetchone()[0]
        commit_unless_batched(self.conn)
        return n

    def get_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all vectors for a single node."""
        rows = self.conn.execute(
            'SELECT vector_type, text, embedding FROM node_enrichments WHERE node_id = ?',
            (node_id,)).fetchall()
        return [{'vector_type': r[0], 'text': r[1], 'embedding': r[2]} for r in rows]
