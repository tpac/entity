"""Top-bar stats + insights — high-level summaries."""


from ..clock import utc_cutoff
from ..daemon_client import DAEMON_PORT, daemon_alive
from ..db import brain_db_path, direct_query, logs_db_path
from ..log import warn
from ._meta import WAKE_ENVELOPE_MARKER
from .system import daemon_record


def query_stats():
    """Top-bar counters: nodes, edges, locked, last 24h, orphans, type histogram,
    daemon status, turns since the last encode against the daemon's cadence."""
    db = brain_db_path()
    nodes = direct_query("SELECT COUNT(*) FROM nodes WHERE archived = 0", db_path=db)
    edges = direct_query("SELECT COUNT(*) FROM edges", db_path=db)
    locked = direct_query("SELECT COUNT(*) FROM nodes WHERE locked = 1", db_path=db)
    types = direct_query(
        "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 GROUP BY type ORDER BY COUNT(*) DESC",
        db_path=db,
    )
    recent = direct_query(
        "SELECT COUNT(*) FROM nodes WHERE created_at > ?",
        args=(utc_cutoff(hours=24),), db_path=db,
    )
    orphans = direct_query(
        "SELECT COUNT(*) FROM nodes n WHERE archived = 0 "
        "AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = n.id OR e.target_id = n.id)",
        db_path=db,
    )

    # Encoding cadence indicator — turns since the active session's last S1
    # encode, read live from traces (the gate's level trigger:
    # turns_since_last_encode >= the cadence). Mirrors
    # dal_logs.conversational_turns_since: active session = latest s0 user
    # turn; anchor on that session's latest encoding_prompt; count s0
    # user_message turns since, excluding wake-envelope ignitions. The cadence
    # is the one the running daemon read at start (its ping reports it,
    # system.daemon_record carries it) — None while the daemon is down.
    daemon = daemon_record()
    encode_every = daemon.get('encode_every')
    enc_position = 0
    try:
        logs = logs_db_path()
        sess = direct_query(
            "SELECT session_id FROM trace_events WHERE scale='s0' "
            "AND ref_type='user_message' AND session_id != '' "
            "ORDER BY created_at DESC LIMIT 1", db_path=logs)
        if sess and sess[0][0]:
            sid = sess[0][0]
            last = direct_query(
                "SELECT created_at FROM trace_events WHERE scale='s1' "
                "AND ref_type='encoding_prompt' AND session_id=? "
                "ORDER BY created_at DESC LIMIT 1", args=(sid,), db_path=logs)
            since = last[0][0] if last else ''
            sql = ("SELECT COUNT(*) FROM trace_events WHERE scale='s0' "
                   "AND session_id=? AND ref_type='user_message' "
                   "AND summary NOT LIKE ?")
            params = [sid, WAKE_ENVELOPE_MARKER + '%']
            if since:
                sql += " AND created_at > ?"
                params.append(since)
            r = direct_query(sql, args=tuple(params), db_path=logs)
            enc_position = r[0][0] if r else 0
    except Exception as e:
        warn('queries.stats', 'computing turns-since-last-encode failed', exc=e)

    return {
        "nodes": nodes[0][0] if nodes else 0,
        "edges": edges[0][0] if edges else 0,
        "locked": locked[0][0] if locked else 0,
        "recent_24h": recent[0][0] if recent else 0,
        "orphans": orphans[0][0] if orphans else 0,
        "types": {t: cnt for t, cnt in types},
        "daemon": "alive" if daemon['alive'] else "unavailable",
        "encoding": {
            "turns_since": enc_position,
            "every": encode_every,
            "due": encode_every is not None and enc_position >= encode_every,
        },
    }


def query_insights():
    """Anchor-facing diagnostic insights — orphan-locked, thin nodes, zero quotes,
    trace coverage, daemon liveness."""
    db = brain_db_path()
    insights = []

    orphan_locked = direct_query(
        "SELECT id, title, type, created_at FROM nodes "
        "WHERE locked = 1 AND archived = 0 "
        "AND NOT EXISTS (SELECT 1 FROM edges e WHERE e.source_id = nodes.id OR e.target_id = nodes.id)",
        db_path=db,
    )
    if orphan_locked:
        insights.append({
            "severity": "high", "icon": "\U0001f512",
            "title": "%d locked nodes are orphaned" % len(orphan_locked),
            "detail": "Important memories disconnected from everything. Recall can't find them through graph traversal.",
            "nodes": [{"id": r[0], "title": r[1], "type": r[2]} for r in orphan_locked],
        })

    thin = direct_query(
        "SELECT COUNT(*), AVG(LENGTH(content)) FROM nodes "
        "WHERE archived = 0 AND LENGTH(content) < 100 "
        "AND created_at > ?",
        args=(utc_cutoff(days=7),), db_path=db,
    )
    if thin and thin[0][0] > 5:
        insights.append({
            "severity": "medium", "icon": "\U0001f4cf",
            "title": "%d thin nodes this week (avg %d chars)" % (thin[0][0], thin[0][1] or 0),
            "detail": "Nodes under 100 chars lack context for future recall.",
        })

    try:
        s1_traces = direct_query(
            "SELECT COUNT(*) FROM trace_events WHERE scale = 's1' "
            "AND created_at > ?",
            args=(utc_cutoff(hours=24),), db_path=logs_db_path(),
        )
        s1_count = s1_traces[0][0] if s1_traces else 0
        if s1_count == 0:
            insights.append({
                "severity": "high", "icon": "\U0001f4ca",
                "title": "No S1 traces in 24h",
                "detail": "No recall or encoding traces. Check daemon and hook pipeline.",
            })
    except Exception as e:
        warn('queries.stats', 'S1 trace coverage insight failed', exc=e)

    _7d = utc_cutoff(days=7)
    # Quote capture is detected via the structured fields (their_raw_quote /
    # my_raw_quote in node_metadata_kv) and quote-typed nodes — never
    # name-string content heuristics: operator/agent names are per-install.
    quotes = direct_query(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0 "
        "AND created_at > ? "
        "AND (type = 'quote' OR EXISTS ("
        "SELECT 1 FROM node_metadata_kv kv WHERE kv.node_id = nodes.id "
        "AND kv.key IN ('their_raw_quote', 'my_raw_quote')))",
        args=(_7d,), db_path=db,
    )
    types = dict(direct_query(
        "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 AND created_at > ? GROUP BY type",
        args=(_7d,), db_path=db,
    ))
    total_recent = sum(types.values())
    if quotes and quotes[0][0] == 0 and total_recent > 5:
        insights.append({
            "severity": "high", "icon": "\U0001f4ad",
            "title": "Zero quotes preserved this week",
            "detail": "No node this week carries a verbatim quote (their_raw_quote / my_raw_quote).",
        })

    if not daemon_alive():
        insights.insert(0, {
            "severity": "high", "icon": "⚠️",
            "title": "Daemon is not running",
            "detail": "Brain daemon on port %d is not responding. Dashboard is showing read-only data from SQLite directly. Live features (recall, encoding) are unavailable." % DAEMON_PORT,
        })

    return insights
