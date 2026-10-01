"""In-memory store for node vectors.

Pure storage + query layer. Knows nothing about SQLite, brain structure,
or callers. Backs CachedVectorDAL (see dal_vector_cached.py).

Why: recall scans all primary + enrichment + situation vectors on every
call. On SQLite that's 25 MB per call, cache-thrashing under concurrent
reads. In memory it's a dict lookup. Same data shape, 100×+ faster.

Stored per (node_id, vector_type):
- embedding bytes (raw, as stored in DB — callers unpack as needed)
- text (first 500 chars of source text)
- model (the name of the embedder that produced the vector)

No JOINs, no archived flag here — CachedVectorDAL handles those by
cross-referencing the nodes table on reads.
"""
import threading
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class VectorCache:
    """Vectors-only in-memory store.

    Thread safety: a single RLock guards all mutations. Readers snapshot
    the relevant dicts under the lock, then release before processing —
    keeps read-path lock contention minimal.
    """

    def __init__(self):
        # (node_id, vector_type) → {'embedding': bytes, 'text': str, 'model': str}
        self._rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # node_id → set of vector_types present for that node (for fast drop)
        self._by_node: Dict[str, Set[str]] = {}
        # Mutation counter — reload triggers can check this for staleness.
        self._version: int = 0
        self._lock = threading.RLock()

    # ── Build / reload ──────────────────────────────────────────────

    def load(self, rows: Iterable[Tuple[str, str, bytes, str, str]]) -> int:
        """Replace cache contents with the given rows.

        Each row: (node_id, vector_type, embedding_bytes, text, model).
        Rows with no embedding are skipped — consistent with VectorDAL.

        Called by CachedVectorDAL at boot and after manual reloads.
        Returns count loaded.
        """
        new_rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
        new_by_node: Dict[str, Set[str]] = {}
        for node_id, vector_type, blob, text, model in rows:
            if blob is None or not node_id or not vector_type:
                continue
            new_rows[(node_id, vector_type)] = {
                'embedding': blob, 'text': text or '', 'model': model or '',
            }
            new_by_node.setdefault(node_id, set()).add(vector_type)
        with self._lock:
            self._rows = new_rows
            self._by_node = new_by_node
            self._version += 1
        return len(new_rows)

    # ── Mutations ───────────────────────────────────────────────────

    def add(self, node_id: str, vector_type: str,
            embedding: bytes, text: str, model: str) -> None:
        """Upsert a single vector. INSERT OR REPLACE semantics."""
        if embedding is None or not node_id or not vector_type:
            return
        with self._lock:
            self._rows[(node_id, vector_type)] = {
                'embedding': embedding,
                'text': text or '',
                'model': model or '',
            }
            self._by_node.setdefault(node_id, set()).add(vector_type)
            self._version += 1

    def add_batch(self, rows: Iterable[Tuple[str, str, bytes, str, str]]) -> int:
        """Upsert many vectors. Returns count written."""
        prepared = [(nid, vt, blob, text or '', model or '')
                    for (nid, vt, blob, text, model) in rows
                    if blob is not None and nid and vt]
        if not prepared:
            return 0
        with self._lock:
            for nid, vt, blob, text, model in prepared:
                self._rows[(nid, vt)] = {
                    'embedding': blob, 'text': text, 'model': model,
                }
                self._by_node.setdefault(nid, set()).add(vt)
            self._version += 1
        return len(prepared)

    def drop_node(self, node_id: str, vector_types=None) -> int:
        """Remove a node's vectors — ALL (vector_types=None: archive/delete
        path) or just the given types (revise invalidation: only the vectors
        whose source text changed). Returns count dropped.

        The types parameter exists because a whole-node drop after a
        partial-DB delete ORPHANS the surviving vectors: revise deletes only
        the affected DB rows, the backfill re-embeds only what the DB is
        missing — so anything dropped here beyond the DB delete never comes
        back until a process restart, and the node silently vanishes from
        every cache-served recall scan (found live 2026-07-17, pooled smoke
        probes; the healer made freshly-encoded nodes recall-invisible)."""
        if not node_id:
            return 0
        with self._lock:
            have = self._by_node.get(node_id, set())
            types = set(have) if vector_types is None else (set(vector_types) & have)
            for vt in types:
                self._rows.pop((node_id, vt), None)
            if vector_types is None:
                self._by_node.pop(node_id, None)
            else:
                have -= types
                if not have:
                    self._by_node.pop(node_id, None)
            if types:
                self._version += 1
            return len(types)

    # ── Queries (reads) ─────────────────────────────────────────────

    def get(self, node_id: str, vector_type: str) -> Optional[Dict[str, Any]]:
        """Get one row. Returns None if missing."""
        with self._lock:
            v = self._rows.get((node_id, vector_type))
            return dict(v) if v else None

    def get_embedding(self, node_id: str, vector_type: str) -> Optional[bytes]:
        """Get just the embedding bytes for a single vector."""
        with self._lock:
            v = self._rows.get((node_id, vector_type))
            return v['embedding'] if v else None

    def get_for_node(self, node_id: str) -> List[Dict[str, Any]]:
        """All vectors for a node. Shape matches VectorDAL.get_for_node()."""
        with self._lock:
            types = list(self._by_node.get(node_id, ()))
            return [{'vector_type': vt,
                     'text': self._rows[(node_id, vt)]['text'],
                     'embedding': self._rows[(node_id, vt)]['embedding']}
                    for vt in types
                    if (node_id, vt) in self._rows]

    def query(self, vector_types: Optional[List[str]] = None,
              model: Optional[str] = None,
              exclude_node_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """Return all rows matching the filters. Shape:
        [{node_id, vector_type, embedding}] — matches VectorDAL.get_all_vectors().

        exclude_node_ids: set of archived ids to mask out (callers pass this
        from the nodes table). Keeps archive logic out of the cache.
        """
        vt_filter = set(vector_types) if vector_types else None
        with self._lock:
            items = list(self._rows.items())
        out = []
        for (nid, vt), v in items:
            if vt_filter and vt not in vt_filter:
                continue
            if model and v['model'] != model:
                continue
            if exclude_node_ids and nid in exclude_node_ids:
                continue
            out.append({'node_id': nid, 'vector_type': vt,
                        'embedding': v['embedding']})
        return out

    def query_primary_with_text(self, model: Optional[str] = None,
                                 exclude_node_ids: Optional[Set[str]] = None
                                 ) -> List[Tuple[str, bytes]]:
        """Primary vectors only. Used by get_all_with_context — which then
        joins to the nodes table for per-node metadata.
        """
        with self._lock:
            items = [(nid, v) for (nid, vt), v in self._rows.items()
                     if vt == '_primary']
        out = []
        for nid, v in items:
            if model and v['model'] != model:
                continue
            if exclude_node_ids and nid in exclude_node_ids:
                continue
            out.append((nid, v['embedding']))
        return out

    # ── Diagnostics ─────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Row counts per vector_type + memory footprint estimate."""
        with self._lock:
            by_type: Dict[str, int] = {}
            total_bytes = 0
            for (_, vt), v in self._rows.items():
                by_type[vt] = by_type.get(vt, 0) + 1
                total_bytes += len(v['embedding']) if v['embedding'] else 0
            return {
                'total_rows': len(self._rows),
                'total_nodes': len(self._by_node),
                'by_vector_type': by_type,
                'embedding_bytes': total_bytes,
                'version': self._version,
            }
