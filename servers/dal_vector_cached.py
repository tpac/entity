"""CachedVectorDAL — drop-in decorator over VectorDAL.

Same public surface as VectorDAL (see servers/dal.py). Brain operations
don't know this exists — brain.py wires self._vec_dal to this class
instead of VectorDAL, nothing else changes.

Architecture:
  - Writes pass through to the inner VectorDAL first. On SQL success,
    the cache is updated. On SQL failure, cache is not touched — SQLite
    is always the truth, cache can never hold data the DB doesn't.
  - Reads serve from VectorCache. For get_all_with_context, the cache
    supplies the vectors and one scan of the `nodes` table supplies the
    fields recall reads (personal, personal_context) — no vector bytes
    leave SQLite on the recall path.
  - find_missing delegates to inner VectorDAL — it's the backfill path,
    not the hot recall path, and it needs node titles/content the cache
    doesn't track.
  - Deletion: delete_for_node(node_id, vector_types=None) is the ONE
    deletion surface — DB delete + exactly-mirrored cache drop in a single
    call (all rows, or the typed subset for revise invalidation). Archive
    and revise both route here via brain_remember; there is no
    cache-only public drop (that split surface caused the 2026-07-17
    healer-invisibility bug).

Thread safety: a single RLock guards cache mutations. Reads snapshot
dicts under the lock then release — negligible contention.
"""
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Set

from .dal import VectorDAL
from .vector_cache import VectorCache


class CachedVectorDAL:
    """VectorDAL decorator with an in-memory vector cache.

    Thread safety model:
      - VectorCache has its own RLock — queries against the in-memory store
        are safe concurrently.
      - This class also guards the shared SQLite connection with a lock.
        Python's sqlite3 is built in SERIALIZED mode (with per-process
        mutex) but macOS ARM builds have historically crashed under
        concurrent execute() on one connection. The daemon's dispatch
        serializes commands, so production never hits concurrency here,
        but this lock makes the class standalone-safe (tests, benchmarks,
        future callers).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._sql_lock = threading.Lock()
        self._inner = VectorDAL(conn)
        self._cache = VectorCache()
        self._load_cache_from_db()

    # ── Boot ────────────────────────────────────────────────────────

    def _load_cache_from_db(self) -> int:
        """One-shot SELECT at construction time. ~200 ms for 8k rows."""
        with self._sql_lock:
            rows = self.conn.execute(
                'SELECT node_id, vector_type, embedding, text, model '
                'FROM node_enrichments WHERE embedding IS NOT NULL'
            ).fetchall()
        return self._cache.load(rows)

    def reload(self) -> int:
        """Public: drop cache and rebuild from SQL. Callers (migration
        scripts, manual SQL writes) should invoke this after bypassing
        the write path."""
        return self._load_cache_from_db()

    # ── Writes (pass through + update cache) ────────────────────────

    def store(self, node_id: str, vector_type: str, text: str,
              embedding: Optional[bytes], model: str) -> None:
        """Upsert one vector. DB first, cache after."""
        with self._sql_lock:
            self._inner.store(node_id, vector_type, text, embedding, model)
        if embedding is not None:
            self._cache.add(node_id, vector_type, embedding,
                            (text or '')[:500], model)

    def store_batch(self, rows, model: str) -> int:
        """Upsert many. DB first, cache after. Returns count written to DB."""
        rows_list = list(rows)
        with self._sql_lock:
            written = self._inner.store_batch(rows_list, model=model)
        if written:
            self._cache.add_batch(
                (nid, vt, blob, (text or '')[:500], model)
                for (nid, vt, text, blob) in rows_list
                if blob is not None)
        return written

    def delete_for_node(self, node_id: str, vector_types=None) -> int:
        """DB delete + cache drop, mirrored EXACTLY (one vector_types set
        feeds both) — the invariant whose violation was the 2026-07-17
        healer-invisibility bug (a typed DB delete paired with a whole-node
        cache drop orphaned the surviving vectors out of every cache-served
        scan until restart). All-rows (vector_types=None: archive/delete)
        or typed subset (revise invalidation)."""
        with self._sql_lock:
            n = self._inner.delete_for_node(node_id, vector_types=vector_types)
        self._cache.drop_node(node_id, vector_types=vector_types)
        return n

    # ── Reads (serve from cache) ────────────────────────────────────

    def get_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        return self._cache.get_for_node(node_id)

    def get_all_vectors(self, exclude_archived: bool = True,
                        vector_types: Optional[List[str]] = None,
                        model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Cache-served. Shape matches VectorDAL exactly."""
        exclude_ids = self._archived_ids() if exclude_archived else None
        return self._cache.query(vector_types=vector_types, model=model,
                                 exclude_node_ids=exclude_ids)

    def get_all_with_context(self, exclude_archived: bool = True,
                             types: List[str] = None,
                             model: str = None) -> List[Dict[str, Any]]:
        """Primary vectors + the node fields recall's STEP 4 reads
        (personal, personal_context).

        Cache supplies the vectors, one scan of the nodes table supplies the
        fields; rows are joined here by id, in cache order.
        """
        primary = self._cache.query_primary_with_text(model=model)
        if not primary:
            return []
        ctx_by_id = self._fetch_node_context(
            exclude_archived=exclude_archived, types=types)
        # Same dict shape as VectorDAL.get_all_with_context.
        out = []
        for nid, blob in primary:
            ctx = ctx_by_id.get(nid)
            if ctx is None:
                # Filtered out by archived/type → skip.
                continue
            out.append({
                'node_id': nid,
                'embedding': blob,
                'personal': ctx[0],
                'personal_context': ctx[1],
            })
        return out

    def find_missing(self, vector_type: str, limit: int = 50,
                     model: Optional[str] = None,
                     node_ids: Optional[set] = None,
                     require_kv_keys_any: Optional[List[str]] = None,
                     source_kv_keys: Optional[List[str]] = None,
                     require_described_edge: bool = False,
                     exclude_relations=()) -> List[Dict[str, Any]]:
        """Delegate — backfill path, cold, needs node.title/content.

        Mirrors VectorDAL.find_missing's full signature so the canonical
        `require_kv_keys_any` / `require_described_edge` kwargs reach the inner
        DAL. Production wires CachedVectorDAL, so a caller passing either would
        otherwise hit a TypeError on the missing parameter.
        """
        with self._sql_lock:
            return self._inner.find_missing(vector_type, limit,
                                            model=model, node_ids=node_ids,
                                            require_kv_keys_any=require_kv_keys_any,
                                            source_kv_keys=source_kv_keys,
                                            require_described_edge=require_described_edge,
                                            exclude_relations=exclude_relations)

    def change_key(self) -> tuple:
        """Delegate — LAF matrix staleness key (one scalar SELECT)."""
        with self._sql_lock:
            return self._inner.change_key()

    def vectors_since(self, rowid: int,
                      vector_types: Optional[List[str]] = None,
                      model: Optional[str] = None) -> List[tuple]:
        """Delegate — LAF incremental matrix append (watermarked pull)."""
        with self._sql_lock:
            return self._inner.vectors_since(rowid, vector_types=vector_types,
                                             model=model)

    # ── Diagnostics ─────────────────────────────────────────────────

    def cache_stats(self) -> Dict[str, Any]:
        """Expose VectorCache stats — useful for `brain diagnose` later."""
        return self._cache.stats()

    # ── Internal helpers ────────────────────────────────────────────

    def _archived_ids(self) -> Set[str]:
        """Fetch the set of archived node ids — a small indexed query."""
        with self._sql_lock:
            rows = self.conn.execute(
                'SELECT id FROM nodes WHERE archived = 1').fetchall()
        return {r[0] for r in rows}

    def _fetch_node_context(self, *, exclude_archived: bool,
                            types: Optional[List[str]]) -> Dict[str, tuple]:
        """{id → (personal, personal_context)} from ONE scan of
        nodes, filtered by archived/type.

        A scan, not `id IN (...)` over the cache's ids: nearly every live
        node is wanted, and an IN list over them must be chunked under
        SQLite's bind limit, which SQLite can plan as a full pass over the
        live rows per chunk — cost growing with N²/chunk (~165 ms at 12.5k
        nodes, against ~12 ms for the one scan).
        """
        where: List[str] = []
        params: List[Any] = []
        if exclude_archived:
            where.append('archived = 0')
        if types:
            where.append('type IN (%s)' % ','.join('?' * len(types)))
            params.extend(types)
        sql = 'SELECT id, personal, personal_context FROM nodes'
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        with self._sql_lock:
            rows = self.conn.execute(sql, params).fetchall()
        return {r[0]: r[1:] for r in rows}
