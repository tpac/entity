"""Generic trace_events query — used by the Traces tab.

Surfaces identity (`human_identity` / `agent_identity`) and the S0 session
stamp (`model` / `host`) from trace metadata as top-level fields so the UI
doesn't have to re-parse JSON: every trace records who was speaking when it was
written, and S0 rows record which model produced the turn on which host.
"""

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query
from ._meta import extract_meta_fields, S0_SESSION_STAMP_FIELDS

_PROMOTED = ('human_identity', 'agent_identity') + S0_SESSION_STAMP_FIELDS


@safe_query('queries.traces', logs_db_path)
def query_traces(conn, hours: int = 24, scale: str = '', limit: int = 500, session_id: str = ''):
    """Read trace_events from brain_logs.db, filtered by time window + scale + session.

    Order/limit: DESC keeps the newest `limit` events; ASC would drop the
    newest events in a busy session and break source-ref navigation. The
    Traces tab re-sorts within chains after fetching, so wire order doesn't
    affect display.
    """
    conditions = ["created_at > ?"]
    params = [utc_cutoff(hours=hours)]
    if scale:
        conditions.append('scale = ?')
        params.append(scale)
    if session_id:
        conditions.append('session_id = ?')
        params.append(session_id)
    where = ' AND '.join(conditions)
    rows = conn.execute(
        "SELECT id, chain_id, scale, event_type, ref_type, ref_id, "
        "summary, metadata, session_id, created_at, interaction_id "
        "FROM trace_events WHERE %s ORDER BY created_at DESC LIMIT ?" % where,
        params + [limit],
    ).fetchall()
    out = []
    for r in rows:
        row = {
            'id': r[0], 'chain_id': r[1], 'scale': r[2],
            'event_type': r[3], 'ref_type': r[4] or '', 'ref_id': r[5] or '',
            'summary': r[6] or '', 'metadata': r[7], 'session_id': r[8] or '',
            'created_at': r[9], 'interaction_id': r[10],
        }
        # One parse per row for every promoted field.
        row.update(zip(_PROMOTED, extract_meta_fields(r[7], *_PROMOTED)))
        out.append(row)
    return out


# The two s0 ref_types that hold what was actually SAID in a session's own
# conversation — the operator's prompt and the session's reply.
_SAID = ('user_message', 'assistant_message')


@safe_query('queries.traces', logs_db_path)
def query_session_messages(conn, session_id: str, limit: int = 8):
    """The last few turns of ONE session's own operator conversation.

    Identity, not content: the Sessions tab shows these so the operator can
    recognise WHICH session it is ("oh, that's the one where I asked
    about X"). A session's handle, model and turn count say what it is; a
    couple of its own lines say what it's for.

    Deliberately narrow — `query_traces` above would haul the session's whole
    event stream (hundreds of rows in a busy session) to find five messages.
    Newest first; the caller reverses for reading order."""
    if not session_id:
        return []
    rows = conn.execute(
        "SELECT ref_type, summary, metadata, created_at FROM trace_events "
        "WHERE session_id = ? AND ref_type IN (?, ?) "
        "ORDER BY created_at DESC LIMIT ?",
        (session_id,) + _SAID + (int(limit),)).fetchall()
    out = []
    for ref_type, summary, metadata, created_at in rows:
        content, model = extract_meta_fields(metadata, 'content', 'model')
        out.append({
            "role": 'operator' if ref_type == 'user_message' else 'session',
            "text": content or summary or '',
            "model": model or '',
            "created_at": created_at,
        })
    return out
