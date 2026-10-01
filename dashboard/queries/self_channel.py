"""Self-channel views — stream↔stream messages + faithful boot captures.

Two read surfaces for the Streams dashboard tab, both over brain_logs.db:

  - query_messages: the directed-signal courier log (self_inflight, with the
    self_delivered fan-out folded in) — every message, who sent it, to whom,
    and which streams have consumed it. This is the COMPLETE log of live
    traffic; the s0 `self_message` traces (queries.traces) are only the
    delivered-into-Observation subset, so the chat view leans on this.

  - query_boot_renders: the exact `for_claude` text the daemon served per
    session at SessionStart (boot_renders), written by _handle_context_boot.
    "What actually got to boot."

Read-only, like every queries.* module. SENDING is a write — it goes through
the daemon (server.py do_POST → daemon_send('self_send')), never from here.
"""

import json

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query


@safe_query('queries.self_channel', logs_db_path)
def query_messages(conn, hours: int = 48, limit: int = 200):
    """The courier log: every in-flight self-message + its delivery fan-out.

    self_inflight holds un-reaped messages. TTL is now per-message (broadcast ~1h
    / directed ~24h, by address) enforced via `expires_at`; the daemon's idle reap
    deletes expired rows (they survive only as s0 `self_message` traces), so this
    is effectively the live courier. `expires_at` shows each message's death time.
    Newest first."""
    cutoff = utc_cutoff(hours=hours)
    rows = conn.execute(
        "SELECT id, from_session, address, body, refs, created_at, expires_at "
        "FROM self_inflight WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
        (cutoff, limit)).fetchall()

    # Delivery fan-out per message (broadcast → many recipients, each once).
    delivered = {}
    for mid, to_session, dat in conn.execute(
            "SELECT message_id, to_session, delivered_at FROM self_delivered").fetchall():
        delivered.setdefault(mid, []).append({
            "to": (to_session or '')[:8],
            "to_full": to_session or '',
            "at": dat,
        })

    out = []
    for r in rows:
        mid = r[0]
        try:
            refs = json.loads(r[4]) if r[4] else []
        except (ValueError, TypeError):
            refs = []
        out.append({
            "id": mid,
            "from": (r[1] or '')[:8],
            "from_full": r[1] or '',
            "address": r[2] or '',
            "body": r[3] or '',
            "refs": refs,
            "created_at": r[5],
            "expires_at": r[6],
            "delivered": delivered.get(mid, []),
        })
    return out


@safe_query('queries.self_channel', logs_db_path)
def query_boot_renders(conn, session_id: str = '', limit: int = 30):
    """Faithful boot captures — the exact text served per session, newest first.
    Optional session filter (defaults to all sessions)."""
    base = ("SELECT id, session_id, user, project, char_count, text, created_at "
            "FROM boot_renders ")
    if session_id:
        rows = conn.execute(
            base + "WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit)).fetchall()
    else:
        rows = conn.execute(
            base + "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [{
        "id": r[0],
        "session_id": r[1] or '',
        "session_short": (r[1] or '')[:8],
        "user": r[2] or '',
        "project": r[3] or '',
        "char_count": r[4] or 0,
        "text": r[5] or '',
        "created_at": r[6],
    } for r in rows]
