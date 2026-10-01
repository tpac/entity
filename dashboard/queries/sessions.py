"""Recent sessions — the one place a session_id becomes something readable.

A session is a stream of thought working somewhere. The brain already persists
where: `SessionContext.save()` writes cwd / branch / worktree / project into
`session_state` under the `_session_context` key. This module joins that to the
trace counts so every session_id the dashboard shows can carry a human handle
and a hover with enough to find the thing again.

Identity convention (matched to the Streams tab and the brain's own
"one stream, one worktree — your handle is your branch name"): the handle is
the worktree name, else the branch tail with the shared `claude/` namespace
dropped, else the 8-char hex. Never invented — if the brain never learned
where a session was working, its handle stays the hex, honestly.
"""

import json

from ..clock import utc_cutoff
from ..db import logs_db_path
from ..query import safe_query


def _handle(env: dict, session_id: str) -> str:
    """Readable handle for one session, from its persisted env."""
    worktree = (env.get('worktree') or '').strip()
    if worktree:
        return worktree
    branch = (env.get('branch') or '').strip()
    if branch and branch != 'unknown':
        # Worktree branches are `claude/<name>`; the namespace is shared by
        # every stream, so the distinctive tail is the handle.
        return branch.split('/', 1)[1] if '/' in branch else branch
    return (session_id or '')[:8]


@safe_query('queries.sessions', logs_db_path)
def query_recent_sessions(conn, limit: int = 80, days: int = 7):
    """Recent sessions, newest activity first, each with its readable handle.

    The limit is generous (80) because this feeds a REGISTRY, not a dropdown:
    every session_id the feed can show must resolve to a name, and the Live
    tab's 48h encode window routinely spans more streams than a picker-sized
    list would cover — a session past the cut renders as raw hex.

    Per session: `id`, `short`, `handle` (the display name), `first`/`last`
    seen, `events` count, plus whatever env the brain recorded (`branch`,
    `worktree`, `project`, `cwd`, and the LATEST `model` / `host` the stream
    rode on — per-turn values live on the S0 rows) and its accumulated `arc`
    line. The frontend renders `handle` and puts the rest in the hover — one
    shape, so no caller re-derives a label.
    """
    rows = conn.execute(
        "SELECT session_id, MIN(created_at) as first_seen, "
        "MAX(created_at) as last_seen, COUNT(*) as event_count "
        "FROM trace_events WHERE session_id != '' "
        "AND created_at > ? "
        "GROUP BY session_id ORDER BY last_seen DESC LIMIT ?",
        (utc_cutoff(days=days), limit),
    ).fetchall()
    if not rows:
        return []
    ids = [r[0] for r in rows]

    # Per-session env, one query for the whole page. session_state is keyed
    # (session_id, key, node_id) — `_session_context` uses the default ''
    # node_id, so session_id alone identifies the row.
    env_by_session: dict = {}
    for sid, value in conn.execute(
        "SELECT session_id, value FROM session_state "
        "WHERE key = '_session_context' AND session_id IN (%s)"
        % ','.join('?' * len(ids)),
        ids,
    ).fetchall():
        try:
            parsed = json.loads(value) if value else {}
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            env_by_session[sid] = parsed

    out = []
    for r in rows:
        sid = r[0]
        env = env_by_session.get(sid, {})
        out.append({
            "id": sid,
            "short": sid[:8],
            "handle": _handle(env, sid),
            "first": r[1],
            "last": r[2],
            "events": r[3],
            "branch": env.get('branch') or '',
            "worktree": env.get('worktree') or '',
            "project": env.get('project') or '',
            "cwd": env.get('cwd') or '',
            "model": env.get('model') or '',
            "host": env.get('host') or '',
            "turns": env.get('stop_counter') or 0,
        })
    return out
