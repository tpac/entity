"""Explorer tab + node-detail panel — node listing and per-node enrichment.

Corrections (via brain.correction_enrich → `_corrections` payload) live behind
a separate endpoint that goes through the daemon TCP socket. The dashboard
can't import the brain mixin (see test_dashboard_disconnection), and replaying
the aspect-edge walk client-side would duplicate logic that's free to change
in `servers/brain_corrections.py`. So we ask the daemon for the canonical
enrichment and render whatever it returns.
"""

from ..daemon_client import daemon_send
from ..db import brain_db_path, direct_query, fetch_by_id, logs_db_path, ro_connect
from ..log import warn


def query_node_list(limit: int = 50, node_type: str = None, search: str = None):
    """Filtered node listing for the Explorer tab."""
    # Search matches title + content.
    sql = (
        "SELECT id, type, title, content, locked, emotion, "
        "access_count, created_at, confidence, encoding_source "
        "FROM nodes WHERE archived = 0"
    )
    args = []
    if node_type:
        sql += " AND type = ?"
        args.append(node_type)
    if search:
        sql += " AND (title LIKE ? OR content LIKE ?)"
        pat = "%%%s%%" % search
        args.extend([pat, pat])
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)

    rows = direct_query(sql, tuple(args), db_path=brain_db_path())
    return [
        {
            "id": r[0], "type": r[1], "title": r[2],
            "content": (r[3] or "")[:500],
            "locked": bool(r[4]),
            "emotion": r[5], "access_count": r[6], "created_at": r[7],
            "confidence": r[8], "encoding_source": r[9],
        }
        for r in rows
    ]


def query_node_detail(node_id: str):
    """Full node detail + KV metadata + active connections (up to 20)."""
    db = brain_db_path()
    row = direct_query(
        "SELECT id, type, title, content, locked, emotion, "
        "access_count, confidence, encoding_source, created_at, last_accessed, "
        "revised_at, personal, personal_context, evolution_status "
        "FROM nodes WHERE id = ?",
        args=(node_id,), db_path=db,
    )
    if not row:
        return None
    r = row[0]
    node = {
        "id": r[0], "type": r[1], "title": r[2], "content": r[3],
        "locked": bool(r[4]), "emotion": r[5],
        "access_count": r[6], "confidence": r[7], "encoding_source": r[8],
        "created_at": r[9], "last_accessed": r[10],
        "revised_at": r[11], "personal": r[12], "personal_context": r[13],
        "evolution_status": r[14],
    }
    meta_kv = direct_query(
        "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
        args=(node_id,), db_path=db,
    )
    if meta_kv:
        kv = {row[0]: row[1] for row in meta_kv if row[1]}
        node["metadata"] = kv
        if kv.get('situation'):
            node["situation"] = kv['situation']

    edges = direct_query(
        "SELECT n.id, er.relation, e.weight, n.type, n.title, "
        "CASE WHEN er.claim_source_id = ? THEN 'outgoing' ELSE 'incoming' END as direction "
        "FROM edges e "
        "JOIN edge_relations er ON er.edge_id = e.edge_id "
        "JOIN nodes n ON n.id = CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END "
        "WHERE (e.source_id = ? OR e.target_id = ?) AND n.archived = 0 "
        "AND er.archived = 0 "
        "ORDER BY e.weight DESC LIMIT 20",
        args=(node_id, node_id, node_id, node_id), db_path=db,
    )
    connections = [
        {"id": e[0], "relation": e[1], "weight": e[2], "type": e[3],
         "title": e[4], "direction": e[5]}
        for e in edges
    ]
    return {"node": node, "connections": connections}


def query_node_source_refs(node_id: str):
    """Trace events this node was encoded from — the episodic-refs substrate.

    `node_source_refs` (schema v27, table 9015636) is the canonical link
    between an encoded node and the traces it crystallized out of. Returns
    each referenced trace's chain_id / scale / event_type / ref_type /
    short summary + the position field (which utterance in the source
    conversation this came from).
    """
    if not node_id:
        return []
    # node_source_refs lives in brain.db; trace_events lives in brain_logs.db.
    # SQLite doesn't cross-DB JOIN cheaply, so two queries + a Python merge.
    refs = direct_query(
        "SELECT trace_id, position, created_at "
        "FROM node_source_refs WHERE node_id = ? "
        "ORDER BY position ASC, created_at ASC",
        args=(node_id,), db_path=brain_db_path(),
    )
    if not refs:
        return []
    trace_ids = [r[0] for r in refs]
    trace_by_id = {}
    with ro_connect(logs_db_path()) as conn:
        if conn is not None:
            try:
                trace_by_id = fetch_by_id(
                    conn, 'trace_events',
                    'id, chain_id, scale, event_type, ref_type, summary, session_id, created_at',
                    trace_ids)
            except Exception as e:
                warn('queries.explorer', 'trace_events join for source_refs failed', exc=e)
    out = []
    for trace_id, position, ref_created in refs:
        t = trace_by_id.get(trace_id)
        if not t:
            out.append({
                "trace_id": trace_id, "position": position,
                "ref_created_at": ref_created,
                "missing": True,  # trace was archived / never written / log-rotated
            })
            continue
        _id, chain_id, scale, event_type, ref_type, summary, session_id, created_at = t
        out.append({
            "trace_id": trace_id,
            "position": position,
            "ref_created_at": ref_created,
            "chain_id": chain_id,
            "scale": scale,
            "event_type": event_type,
            "ref_type": ref_type or "",
            "summary": (summary or "")[:200],
            "session_id": session_id or "",
            "trace_created_at": created_at,
        })
    return out


def query_node_corrections(node_id: str):
    """Fetch correction-edge enrichment for one node via the daemon.

    Returns the raw `_corrections` list — direction / relation / edge_description
    / content / reasoning / their_raw_quote / my_raw_quote per neighbor. The
    daemon's brain.get_node already attaches this on every canonical pull, so
    we just unwrap it from the rich response. When the daemon is down, returns
    [] silently — the UI shows the node detail without a corrections section
    rather than failing.
    """
    if not node_id:
        return []
    result = daemon_send('get_node', {'node_id': node_id}, timeout=5)
    if not result or not isinstance(result, dict):
        return []
    # brain.get_node() with a single id attaches _corrections as a flat
    # LIST of correction dicts on the returned node (already resolved from
    # the multi-id dict by brain_recall.py:408). Defensive: a future
    # signature change could expose the dict shape — handle both.
    raw = result.get('_corrections')
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in (node_id[:8], node_id):
            if key in raw and raw[key]:
                return raw[key]
        for v in raw.values():
            if v:
                return v
    return []
