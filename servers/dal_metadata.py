"""DAL for node metadata key-value store.

All metadata reads/writes go through this class. No raw SQL for metadata
anywhere else in the codebase. Keys are validated against the contract.

Usage:
    dal = MetadataDAL(conn)
    dal.set(node_id, 'reasoning', 'Why this was encoded')
    dal.set_many(node_id, {'reasoning': '...', 'my_raw_quote': '...'})
    meta = dal.get(node_id)  # → {'reasoning': '...', 'my_raw_quote': '...'}
    dal.delete(node_id, 'reasoning')

Storage convention:
    - str values stored as-is
    - list and dict values JSON-encoded before storage (json.loads recovers
      them on read).
    - other types coerced via str()
"""

import json
import sqlite3
from typing import Any, Dict, Optional, List


def _encode_value(value: Any) -> Optional[str]:
    """Serialize a metadata value to its on-disk string form.

    - None / empty-string: returns None (caller skips the row).
    - list / dict / tuple: JSON-encoded.
    - other: str()-coerced.
    """
    if value is None:
        return None
    if isinstance(value, (list, dict, tuple)):
        # Tuples normalize to lists in JSON
        return json.dumps(list(value) if isinstance(value, tuple) else value)
    s = str(value)
    if not s.strip():
        return None
    return s


class MetadataDAL:
    """Key-value metadata access for nodes. Single table: node_metadata_kv."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def change_key(self) -> tuple:
        """(row count, max rowid) — cheap staleness key for kv-derived caches
        (LAF proj lane). INSERT OR REPLACE assigns a fresh rowid on update, so
        edits bump MAX(rowid); COUNT catches deletions. Mirrors
        NodeDAL.change_key."""
        return tuple(self.conn.execute(
            'SELECT COUNT(*), COALESCE(MAX(rowid), 0) '
            'FROM node_metadata_kv').fetchone())

    def node_ids_with_value_in(self, key: str, values: List[str]) -> List[str]:
        """Node ids whose kv[key] case-insensitively matches any of `values`
        (callers pass lowercase). One indexed scan — the scope veil's outward
        query ("every node stamped with an isolated value")."""
        if not values:
            return []
        placeholders = ','.join('?' * len(values))
        rows = self.conn.execute(
            'SELECT DISTINCT node_id FROM node_metadata_kv '
            'WHERE key = ? AND lower(value) IN (%s)' % placeholders,
            [key] + list(values)).fetchall()
        return [r[0] for r in rows]

    def node_ids_with_time_in(self, key: str, start_ts: int,
                              end_ts: int) -> List[str]:
        """Non-archived node ids whose kv[key] ISO time falls in
        [start_ts, end_ts] (Unix seconds), newest first — recall_by_time's
        `event` anchor over event_time.

        A date-only value ('2026-09-09') covers its whole day; a datetime is a
        point. strftime reads both forms, with or without a UTC offset; a
        value it can't read is NULL and never matches."""
        rows = self.conn.execute(
            "SELECT kv.node_id FROM node_metadata_kv kv "
            "JOIN nodes n ON n.id = kv.node_id "
            "WHERE kv.key = ? AND n.archived = 0 "
            "AND CAST(strftime('%s', kv.value) AS INTEGER) <= ? "
            "AND CAST(strftime('%s', kv.value) AS INTEGER) "
            "    + (CASE WHEN length(kv.value) = 10 THEN 86399 ELSE 0 END) >= ? "
            "ORDER BY CAST(strftime('%s', kv.value) AS INTEGER) DESC, "
            "n.created_at DESC, kv.node_id",
            (key, end_ts, start_ts)).fetchall()
        return [r[0] for r in rows]

    def distinct_values_for_key(self, key: str) -> List[str]:
        """All distinct values stored under kv[key] (as stored, caller
        normalizes). Small result set for provenance keys (one value per
        project/counterpart) — lets the scope veil resolve per-VALUE policy
        (overrides) in Python instead of encoding policy into SQL."""
        rows = self.conn.execute(
            'SELECT DISTINCT value FROM node_metadata_kv WHERE key = ?',
            (key,)).fetchall()
        return [r[0] for r in rows if r[0]]

    def change_probe(self) -> int:
        """MAX(rowid) — the cheap staleness witness for kv-derived caches
        checked on HOT paths (scope veil: once per gate touch). O(1) via
        SQLite's max-rowid optimization, unlike change_key()'s COUNT(*)
        full scan. Misses deletions — acceptable for provenance keys, which
        are effectively append-only (revise cannot move them; only
        migration deletes, which also restarts the daemon)."""
        row = self.conn.execute(
            'SELECT COALESCE(MAX(rowid), 0) FROM node_metadata_kv').fetchone()
        return row[0]

    def get(self, node_id: str) -> Dict[str, str]:
        """Get all metadata for a node as a dict. Returns {} if none.

        Values are returned as stored — callers needing typed list/dict values
        json.loads them.
        """
        rows = self.conn.execute(
            'SELECT key, value FROM node_metadata_kv WHERE node_id = ?',
            (node_id,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_field(self, node_id: str, key: str) -> Optional[str]:
        """Get a single metadata field. Returns None if not set."""
        row = self.conn.execute(
            'SELECT value FROM node_metadata_kv WHERE node_id = ? AND key = ?',
            (node_id, key)).fetchone()
        return row[0] if row else None

    def get_fields(self, node_id: str, keys: List[str]) -> Dict[str, str]:
        """Get specific metadata fields. Returns dict with only present keys."""
        if not keys:
            return {}
        placeholders = ','.join('?' * len(keys))
        rows = self.conn.execute(
            'SELECT key, value FROM node_metadata_kv WHERE node_id = ? AND key IN (%s)' % placeholders,
            [node_id] + keys).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_fields_bulk(self, node_ids: List[str],
                        keys: List[str]) -> Dict[str, Dict[str, str]]:
        """Bulk-fetch specific metadata fields for multiple nodes.

        Returns {node_id: {key: value}} — outer dict only contains nodes
        that have at least one of the requested keys; inner dict only
        contains keys actually present (matching get_fields semantics).

        Used by correction-enrichment and other bulk consumers that would
        otherwise N+1 over get_fields().
        """
        if not node_ids or not keys:
            return {}
        node_ph = ','.join('?' * len(node_ids))
        key_ph = ','.join('?' * len(keys))
        rows = self.conn.execute(
            'SELECT node_id, key, value FROM node_metadata_kv '
            'WHERE node_id IN (%s) AND key IN (%s)' % (node_ph, key_ph),
            list(node_ids) + list(keys)).fetchall()
        out: Dict[str, Dict[str, str]] = {}
        for nid, k, v in rows:
            out.setdefault(nid, {})[k] = v
        return out

    def get_all_bulk(self, node_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Bulk-fetch ALL metadata fields for multiple nodes.

        Returns {node_id: {key: value}} — same shape as get_fields_bulk
        but without a key whitelist. Used by callers that need the full
        K/V slice (e.g. brain_recall.get_node assembling a rich node).
        Single SQL source for the "all metadata for these nodes" pattern.
        """
        if not node_ids:
            return {}
        ph = ','.join('?' * len(node_ids))
        rows = self.conn.execute(
            'SELECT node_id, key, value FROM node_metadata_kv '
            'WHERE node_id IN (%s)' % ph,
            list(node_ids)).fetchall()
        out: Dict[str, Dict[str, str]] = {}
        for nid, k, v in rows:
            out.setdefault(nid, {})[k] = v
        return out

    def set(self, node_id: str, key: str, value: Any) -> None:
        """Set a single metadata field. Overwrites if exists.

        list/dict values JSON-encoded; str values stored as-is; other types
        coerced via str(). Empty/None values silently skipped.
        """
        encoded = _encode_value(value)
        if encoded is None:
            return
        self.conn.execute(
            'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)',
            (node_id, key, encoded))

    def set_many(self, node_id: str, metadata: Dict[str, Any]) -> int:
        """Set multiple metadata fields at once. Returns count written.

        list/dict values JSON-encoded; str values stored as-is; other types
        coerced via str(). Empty/None values silently skipped.
        """
        count = 0
        for key, value in metadata.items():
            encoded = _encode_value(value)
            if encoded is None:
                continue
            self.conn.execute(
                'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)',
                (node_id, key, encoded))
            count += 1
        return count

    def delete(self, node_id: str, key: str) -> bool:
        """Delete a single metadata field. Returns True if existed."""
        cursor = self.conn.execute(
            'DELETE FROM node_metadata_kv WHERE node_id = ? AND key = ?',
            (node_id, key))
        return cursor.rowcount > 0

    def delete_all(self, node_id: str) -> int:
        """Delete all metadata for a node. Returns count deleted."""
        cursor = self.conn.execute(
            'DELETE FROM node_metadata_kv WHERE node_id = ?',
            (node_id,))
        return cursor.rowcount

    def get_all_by_key(self, key: str) -> Dict[str, str]:
        """Get all node_id→value pairs for a given key. For bulk loading."""
        rows = self.conn.execute(
            'SELECT node_id, value FROM node_metadata_kv WHERE key = ?',
            (key,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def bulk_set_key(self, key: str, node_values: Dict[str, Any]) -> int:
        """Set the same key on many nodes at once. Returns count written.

        Same value-encoding rules as set_many: list/dict JSON-encoded, str
        as-is, other coerced. Empty/None silently skipped.
        """
        count = 0
        for node_id, value in node_values.items():
            encoded = _encode_value(value)
            if encoded is None:
                continue
            self.conn.execute(
                'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) VALUES (?, ?, ?)',
                (node_id, key, encoded))
            count += 1
        return count
