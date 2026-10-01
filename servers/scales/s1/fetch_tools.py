"""Fetch toolbox for the agentic surface.

Two tools offered to Haiku (TOOL_DEFINITIONS), each a thin wrapper over an
existing brain primitive, plus expand_node, callable but not offered. The
tool NAMES carry intent vocabulary — Haiku picking a tool IS the intent
classification. No separate classifier.

DESIGN CONTRACT
---------------
- Brain primitives are single source of truth (servers/brain.py).
  Tools wrap, never duplicate.
- Each tool returns the same candidate shape so spread_activation + render
  don't need new branches.
- Inputs accept natural language ('last 10 hours', 'yesterday') — tools
  parse to timestamps. Haiku doesn't compute dates.
- Tools never raise; failures return [] with a logged warning.

LOOP CONTROL
------------
- Max 2 rounds (surface.py _call_surface_agentic max_rounds=2).
- Tools are offered on EVERY round byte-identical — the tools+system
  prefix is prompt-cached. The final round is sent with tool_choice='none',
  so Haiku must finalize with the selection JSON (max_rounds is the hard
  API-call cap; no extra forced call).
- Parallel tool calls per round encouraged (Anthropic native).
- Behavioral discipline (in surface prompt) prevents iterating same query.
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


# ─── Tool definitions exposed to Haiku via Anthropic `tools` param ───────

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "recall_topical",
        "description": (
            "Topical semantic recall — embeddings plus rare title words. Use when the user "
            "asks about a topic or concept. NOT for time-bounded ('yesterday'), "
            "verbatim quotes ('what did X say'), or session-continuation queries. "
            "This is the default fall-back when no other tool's intent matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The semantic query — what to find similar to."},
                # k lowered 25 → 8 (2026-06-12, operator call): at k=25 each
                # topical call could flood the 25 cosine candidates 1:1 —
                # volume + last-position = tool results overshadowing the
                # original pool (finding dfb4691e). Precision over recall:
                # the 25 already cover the broad query; topical exists for
                # the specific gap. Hard cap enforced in the function.
                "k": {"type": "integer", "description": "Max results (default 8, max 10)", "default": 8},
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall_by_time",
        "description": (
            "THE time tool — any time-anchored ask, rolling or absolute: "
            "'what did we do in March 2023', 'Q1 2024 launch work', "
            "'yesterday', 'last week', 'the thing we talked about 3 weeks "
            "ago'. Optionally combined with a semantic query.\n\n"
            "Pick the anchor by what the time refers to: when events "
            "HAPPENED (event), when we TALKED about it (discussed), or when "
            "it was encoded/revised (created/updated).\n\n"
            "Ranking tiers when both `query` and time are given:\n"
            "  1. Entities matching BOTH query and time range (top)\n"
            "  2. Entities matching just the query (no time match)\n"
            "  3. Entities matching just the time range\n\n"
            "If an edge is matched by time, the response also includes its "
            "source and target nodes so the relation has context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_when": {
                    "type": "string",
                    "description": (
                        "Range start. Natural language ok: 'January 2023', "
                        "'last month', 'Q1 2024', '2024-05'. Omit / empty "
                        "string = open-ended past."
                    ),
                },
                "end_when": {
                    "type": "string",
                    "description": (
                        "Range end. Natural language ok: 'before May', "
                        "'March 2024', '2024-03-31'. Omit / empty string = "
                        "open-ended future."
                    ),
                },
                "time_anchor": {
                    "type": "string",
                    "enum": ["event", "created", "updated", "discussed"],
                    "default": "event",
                    "description": (
                        "Which date to filter on:\n"
                        "  event     — the node's event_time (default; when "
                        "what it describes happened)\n"
                        "  discussed — when the CONVERSATION touched it "
                        "(trace time). Use for 'we talked about / worked on "
                        "X <time> ago' — finds re-discussed old nodes that "
                        "created/updated miss\n"
                        "  created   — when the node/edge was first encoded\n"
                        "  updated   — when last revised / strengthened"
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional semantic filter. When provided, results "
                        "are tiered: query∩time first, query-only second, "
                        "time-only third."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max results (default 20)",
                },
            },
        },
    },
    # expand_node is callable but not offered to Haiku: re-offering it is an
    # open operator decision (6a9062d8).
]


# ─── Natural-language window/date parser ─────────────────────────────────

# Units for 'last N <unit>' and 'N <unit> ago'. Months and years step on the
# calendar (relativedelta); 'm' stays minutes, months need 'mo'.
_UNIT_RE = (r"(months?|mo|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|wks?|w"
            r"|years?|yrs?|y)")
_NUM_RE = r"(\d+|an?|(?:a\s+)?couple(?:\s+of)?|(?:a\s+)?few)"
_NUM_WORDS = {'a': 1, 'an': 1, 'couple': 2, 'few': 3}


def _num(tok: str) -> int:
    words = [w for w in tok.split() if w not in ('a', 'of')] or ['a']
    return int(words[-1]) if words[-1].isdigit() else _NUM_WORDS[words[-1]]


def _back(now: datetime, n: int, unit: str) -> datetime:
    """`now` minus n units."""
    from dateutil.relativedelta import relativedelta
    u = unit.rstrip('s')
    if u in ('mo', 'month'):
        return now - relativedelta(months=n)
    if u in ('y', 'yr', 'year'):
        return now - relativedelta(years=n)
    if u in ('m', 'min', 'minute'):
        return now - timedelta(minutes=n)
    if u in ('h', 'hr', 'hour'):
        return now - timedelta(hours=n)
    if u in ('d', 'day'):
        return now - timedelta(days=n)
    return now - timedelta(weeks=n)


def _strict_datetime(s: str) -> Optional[datetime]:
    """A literal date/time ('2026-09-23', 'September 23', '2026-09-23T20:00Z'),
    or None. Never fuzzy: fuzzy parsing reads any digit in free text as a day
    of the current month ('2 weeks ago' → the 2nd)."""
    try:
        from dateutil import parser as _dp
        dt = _dp.parse(s)
    except Exception:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _parse_window(window: str, now: Optional[datetime] = None,
                  unrecognized: Optional[List[str]] = None) -> Tuple[datetime, datetime]:
    """Parse a natural-language 'recent window' string into (start, end) UTC.

    Always returns (start <= end). End is `now` for rolling windows; 'N units
    ago' is a point, returned as (point, point), so it opens a window as a
    start and closes one as an end. Phrases about the current conversation
    (turns, messages, this session) get the 24h session heuristic. Anything
    else falls back to the last 24h and is appended to `unrecognized` when
    the caller passes a list.
    """
    if now is None:
        from servers.clock import brain_now
        now = brain_now()
    s = (window or '').strip().lower()

    # 'last N hours', 'past 2 months', 'last few days', 'last 10h'
    m = re.match(r"(?:last|past)\s+%s\s*%s$" % (_NUM_RE, _UNIT_RE), s)
    if m:
        try:
            return (_back(now, _num(m.group(1)), m.group(2)), now)
        except (OverflowError, ValueError):
            pass  # out of the date range: unrecognized, below

    # '3 weeks ago', '2w ago', 'about 3 months ago', 'a few days ago'
    m = re.match(r"(?:about\s+|around\s+|~)?%s\s*%s\s+ago$" % (_NUM_RE, _UNIT_RE), s)
    if m:
        try:
            point = _back(now, _num(m.group(1)), m.group(2))
            return (point, point)
        except (OverflowError, ValueError):
            pass  # out of the date range: unrecognized, below

    # 'last 24h' / '24h' / '10h' shorthand
    m = re.match(r"(\d+)\s*h$", s)
    if m:
        return (now - timedelta(hours=int(m.group(1))), now)

    if s in ('now',):
        return (now, now)
    if s in ('today', 'earlier today'):
        return (now.replace(hour=0, minute=0, second=0, microsecond=0), now)
    if s in ('yesterday',):
        start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start_today - timedelta(days=1), start_today)
    if s in ('this morning',):
        return (now.replace(hour=0, minute=0, second=0, microsecond=0),
                now.replace(hour=12, minute=0, second=0, microsecond=0))
    if s in ('this afternoon',):
        return (now.replace(hour=12, minute=0, second=0, microsecond=0), now)
    if s in ('this week',):
        return (now - timedelta(days=now.weekday()), now)
    if s in ('last week',):
        end = now - timedelta(days=now.weekday())
        return (end - timedelta(days=7), end)
    if s in ('this month',):
        return (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), now)
    if s in ('last month', 'past month'):
        return (_back(now, 1, 'month'), now)

    # 'last N sessions' — the session heuristic, N times over.
    m = re.match(r"(?:last|past)\s+(\d+)\s+sessions?$", s)
    if m:
        return (now - timedelta(hours=24 * int(m.group(1))), now)

    # 'since last session' / 'last session', and phrases about the current
    # conversation ('last 3 turns', 'this session', 'last message', 'just
    # now') — heuristic: last 24h
    if ('last session' in s or 'previous session' in s or s == 'just now'
            or re.search(r"\b(turns?|messages?|session|conversation|chatted)\b", s)):
        return (now - timedelta(hours=24), now)

    dt = _strict_datetime(s)
    if dt is not None:
        return (dt, now)

    # Final fallback: last 24h
    if unrecognized is not None:
        unrecognized.append(window)
    return (now - timedelta(hours=24), now)


def _parse_date_expr(expr: Any, now: Optional[datetime] = None,
                     unrecognized: Optional[List[str]] = None
                     ) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse a date expression into (since, until) UTC. Either may be None
    (open-ended). Accepts strings OR dicts {since: ..., until: ...}. A
    'since' / 'before' / 'on' prefix takes the start point of whatever
    follows it ('since 2026-05-01', 'before today', 'since last session')."""
    if now is None:
        from servers.clock import brain_now
        now = brain_now()

    if isinstance(expr, dict):
        since = expr.get('since')
        until = expr.get('until')
        return (_strict_datetime(str(since)) if since else None,
                _strict_datetime(str(until)) if until else None)

    s = (str(expr) if expr is not None else '').strip().lower()

    for prefix in ('since ', 'before ', 'on '):
        if s.startswith(prefix):
            point = _parse_window(s[len(prefix):].strip(), now=now,
                                  unrecognized=unrecognized)[0]
            if prefix == 'since ':
                return (point, now)
            if prefix == 'before ':
                return (None, point)
            day_start = point.replace(hour=0, minute=0, second=0, microsecond=0)
            return (day_start, day_start + timedelta(days=1))

    # Reuse window parser for common shortcuts
    start, end = _parse_window(s, now=now, unrecognized=unrecognized)
    return (start, end)


# ─── Candidate shape normalization ──────────────────────────────────────

def _to_candidate(node: Dict[str, Any], score: float, source_tool: str) -> Dict[str, Any]:
    """Attach the recall-specific fields (score, discovery) to a node dict.

    Preserves the full rich shape — every field the fetcher provided rides
    through to the formatter. execute_tool() runs a batched brain.get_node
    pass after this so the final candidate is fully enriched (_corrections,
    _metadata, connections), keeping tool-fetched results symmetric with
    the initial cosine 25.
    """
    if not isinstance(node, dict):
        return None
    nid = node.get('id') or node.get('node_id')
    if not nid:
        return None
    cand = dict(node)
    cand['id'] = nid
    cand['score'] = float(score)
    # ONE provenance field. `discovery` is what format_candidate_for_surface
    # matches against SURFACE_FETCH_TOOLS to decide the `source_tool="true"`
    # flag, so it is the only field that can carry a tool's provenance to
    # Haiku. Assignment, like id and score above: the tool that fetched this
    # node is the truth, and a pre-existing value could only ever mislabel it.
    cand['discovery'] = source_tool
    return cand


# ─── The tools ───────────────────────────────────────────────────────────

def recall_topical(brain, query: str, k: int = 8, **_) -> List[Dict[str, Any]]:
    """Topical semantic recall — wraps brain.recall() (LAF field scoring).
    Default fallback when no other tool's intent fires.

    k default 8, hard-capped at 10 (2026-06-12): tool results join the
    selection pool LAST (position bias) — at k=25 they flooded the original
    cosine candidates. Archived nodes never returned (brain.recall filters
    archived — LAF scores the live graph only)."""
    try:
        from servers.scales.s1.surface_contract import recall_score
        k = min(int(k), 10)
        results = brain.recall(query=query, limit=k)
        if isinstance(results, dict):
            results = results.get('results') or results.get('items') or []
        if not results:
            # Cosine top-k against a populated brain essentially always
            # returns k results — zero raw results means the tool broke
            # (drift, empty index, recall regression), not "no matches".
            brain._log_warning(
                'fetch_topical_zero_raw',
                'recall_topical returned 0 raw results — brain.recall drift?',
                'query=%r k=%d' % (query[:120], k))
        out = []
        for r in (results or []):
            cand = _to_candidate(r, recall_score(r) if isinstance(r, dict) else 0.0,
                                  'recall_topical')
            if cand:
                out.append(cand)
        return out[:int(k)]
    except Exception as e:
        brain._log_error('fetch_recall_topical', e, 'surface fetch tool failed; returned no candidates')
        print('[fetch_tools] recall_topical failed: %s' % e, file=sys.stderr)
        return []


def _parse_when_to_ts(when: str, default_to_end: bool = False,
                      unrecognized: Optional[List[str]] = None) -> Optional[int]:
    """Convert a natural-language time expression to Unix seconds.

    `default_to_end`: when the parsed expression has month/year precision
    and represents an endpoint, push to end-of-period (e.g. 'March 2023'
    as end_when -> 2023-03-31T23:59:59 vs 2023-03-01 as start_when).

    Returns None if `when` is empty or unparseable. A phrase the window
    parser does not know is appended to `unrecognized` (see _parse_window).
    """
    if not when or not when.strip():
        return None
    try:
        from servers.temporal_extraction import (
            extract_intervals_from_text,
        )
        intervals = extract_intervals_from_text(when)
        if intervals:
            start_ts, end_ts, _ = intervals[0]
            return end_ts if default_to_end else start_ts
    except Exception:
        pass
    # Relative phrases ('yesterday', 'last week', '3 weeks ago', ...).
    try:
        since, until = _parse_date_expr(when, unrecognized=unrecognized)
        if default_to_end:
            target = until or since
        else:
            target = since or until
        if target is None:
            return None
        return int(target.timestamp())
    except Exception:
        return None


def _fetch_edges_with_endpoints(brain, edge_ids: List[str]) -> List[Dict[str, Any]]:
    """For each edge_id, return [edge_synthetic_candidate, source_node,
    target_node]. The edge candidate is synthetic — title carries the
    relation, content carries the description, type='edge_relation'.

    Endpoint nodes are fetched in ONE batch via brain.get_node(list),
    not per-endpoint, to avoid N+1 lookups when an operator's question
    surfaces many edges in the same time window.
    """
    if not edge_ids:
        return []
    out = []
    conn = brain.conn
    placeholders = ','.join('?' * len(edge_ids))
    rows = conn.execute(
        f'''SELECT er.edge_id, er.claim_source_id, er.claim_target_id,
                   er.relation, er.description
            FROM edge_relations er
            WHERE er.edge_id IN ({placeholders})
              AND (er.archived IS NULL OR er.archived = 0)''',
        edge_ids,
    ).fetchall()
    if not rows:
        return []
    # Emit in the caller's order (newest first); SQL returns index order.
    _pos = {eid: i for i, eid in enumerate(edge_ids)}
    rows.sort(key=lambda r: _pos.get(r[0], len(_pos)))

    # One batch fetch for every endpoint we'll need.
    endpoint_ids: set = set()
    for _, source_id, target_id, _, _ in rows:
        endpoint_ids.add(source_id)
        endpoint_ids.add(target_id)
    try:
        nodes_map = brain.get_node(list(endpoint_ids))
        if not isinstance(nodes_map, dict):
            nodes_map = {}
    except Exception as e:
        brain._log_error('fetch_by_time_edge_endpoints', e,
                         'endpoint get_node failed; edges emitted without '
                         'source/target candidates')
        nodes_map = {}

    # Endpoint liveness: get_node follows an ABSORBED endpoint to its live
    # survivor (the canonical-pull contract) — the survivor IS the right
    # candidate for Haiku. Only RETIRED corpses (archived, no successor)
    # are dropped, counted via a low-severity warning: routine retrieval,
    # not a producer bug (docs/TRACE-NODE-RESOLUTION.md §6).
    _retired_dropped: set = set()

    seen_endpoint_ids: set = set()
    for edge_id, source_id, target_id, relation, description in rows:
        # Synthetic edge candidate.
        out.append({
            'id': edge_id,
            'title': '%s → %s : %s' % (
                source_id[:8], target_id[:8], (relation or '')[:60]),
            'type': 'edge_relation',
            'content': description or '',
            'score': 0.5,
            'discovery': 'recall_by_time',
            'kv': {
                'edge_relation': relation,
                'source_id': source_id,
                'target_id': target_id,
            },
            'tier': 'edge_time',
        })
        # Source / target as context candidates. Dedup on the node that
        # actually becomes the candidate (node['id']) — ONE id-space: it
        # covers both a repeated endpoint across edges (same live id) and
        # two absorbed endpoints collapsing onto one survivor. Deduping on
        # the requested endpoint_id as well was the 2026-08-30 regression:
        # for a live endpoint the two ids are equal, so the second check
        # saw the first's insert and dropped every live candidate.
        for endpoint_id, role in [(source_id, 'edge_source'),
                                    (target_id, 'edge_target')]:
            node = nodes_map.get(endpoint_id)
            if not node:
                continue
            if node.get('archived'):
                _retired_dropped.add(endpoint_id)
                continue
            if node['id'] in seen_endpoint_ids:
                continue
            seen_endpoint_ids.add(node['id'])
            cand = _to_candidate(node, 0.4, 'recall_by_time')
            if cand:
                cand['tier'] = role
                out.append(cand)
    if _retired_dropped:
        brain._log_warning(
            'fetch_by_time_retired_endpoint',
            '%d retired endpoint nodes dropped (archived, no live successor)'
            % len(_retired_dropped),
            'sample=%s' % sorted(_retired_dropped)[:5])
    return out


def recall_by_time(brain, start_when: str = '', end_when: str = '',
                    time_anchor: str = 'event', query: Optional[str] = None,
                    limit: int = 20, **_) -> List[Dict[str, Any]]:
    """Time-bounded recall with optional semantic ranking.

    Ranking tiers (when query provided):
      1. query AND time match — top
      2. query only — middle
      3. time only — bottom, newest first on the anchor's own time (edges
         include their source/target nodes)

    When query is empty/None, returns only tier 3 (time matches).
    When both start_when and end_when are empty, returns []
    (no scope to filter on).
    """
    try:
        # 1. Parse time range. Use very wide bounds for open-ended sides.
        _unparsed: List[str] = []
        start_ts = _parse_when_to_ts(start_when, default_to_end=False,
                                     unrecognized=_unparsed)
        end_ts = _parse_when_to_ts(end_when, default_to_end=True,
                                   unrecognized=_unparsed)
        if _unparsed:
            brain._log_warning(
                'fetch_by_time_unparsed_window',
                '%d time phrase(s) not understood; the 24h default was used'
                % len(_unparsed), 'values=%r' % [v[:120] for v in _unparsed])
        if start_ts is None and end_ts is None:
            return []
        if start_ts is None:
            start_ts = 0
        if end_ts is None:
            end_ts = 9_999_999_999  # year 2286

        # 2. Get time-matching entity ids based on anchor, newest first on
        # the anchor's own time (tier 3 and the edge tier emit in this order).
        time_node_ids: List[str] = []
        time_edge_ids: List[str] = []
        conn = brain.conn
        if time_anchor == 'event':
            # The node's own event_time field (archived filtered in the DAL).
            # Edges carry no event time, so this anchor feeds no edge tier.
            time_node_ids = brain._meta_kv.node_ids_with_time_in(
                'event_time', start_ts, end_ts)
        elif time_anchor == 'created':
            # Point-in-time anchors use exclusive bounds (gt/lt).
            # Second-level boundary precision is irrelevant for date windows.
            from datetime import datetime as _dt, timezone as _tz
            start_iso = _dt.fromtimestamp(start_ts, tz=_tz.utc).isoformat()
            end_iso = _dt.fromtimestamp(end_ts, tz=_tz.utc).isoformat()
            time_node_ids = [r[0] for r in conn.execute(
                "SELECT id FROM nodes WHERE archived = 0 "
                "AND created_at > ? AND created_at < ? "
                "ORDER BY created_at DESC, id",
                (start_iso, end_iso),
            )]
            time_edge_ids = [r[0] for r in conn.execute(
                "SELECT edge_id FROM edges "
                "WHERE created_at > ? AND created_at < ? "
                "ORDER BY created_at DESC, edge_id",
                (start_iso, end_iso),
            )]
        elif time_anchor == 'updated':
            from datetime import datetime as _dt, timezone as _tz
            start_iso = _dt.fromtimestamp(start_ts, tz=_tz.utc).isoformat()
            end_iso = _dt.fromtimestamp(end_ts, tz=_tz.utc).isoformat()
            time_node_ids = [r[0] for r in conn.execute(
                "SELECT id FROM nodes WHERE archived = 0 "
                "AND COALESCE(revised_at, created_at) > ? "
                "AND COALESCE(revised_at, created_at) < ? "
                "ORDER BY COALESCE(revised_at, created_at) DESC, id",
                (start_iso, end_iso),
            )]
            time_edge_ids = [r[0] for r in conn.execute(
                "SELECT edge_id FROM edges "
                "WHERE COALESCE(last_strengthened, created_at) > ? "
                "AND COALESCE(last_strengthened, created_at) < ? "
                "ORDER BY COALESCE(last_strengthened, created_at) DESC, edge_id",
                (start_iso, end_iso),
            )]
        elif time_anchor == 'discussed':
            # Trace-time anchor (2026-06-12, replaces recall_recent): nodes
            # the CONVERSATION actually touched in the window — surface
            # selections (s1r K events) carry the surfaced node ids. 'The
            # thing we talked about 3 weeks ago' is a trace property:
            # created/updated miss re-discussions of old nodes, and
            # updated_at is bumped by encoder/S2 churn (the recall_recent
            # bug class). Read-only scan of the daemon's own logs handle.
            from datetime import datetime as _dt, timezone as _tz
            start_iso = _dt.fromtimestamp(start_ts, tz=_tz.utc).isoformat()
            end_iso = _dt.fromtimestamp(end_ts, tz=_tz.utc).isoformat()
            import json as _json
            try:
                # Newest trace first, so an id's first appearance is its
                # latest discussion.
                _seen: set = set()
                for r in brain.logs_conn.execute(
                    "SELECT ref_id FROM trace_events "
                    "WHERE scale='s1' AND event_type='K' "
                    "AND ref_type='surface_selected' "
                    "AND created_at > ? AND created_at < ? "
                    "ORDER BY created_at DESC",
                    (start_iso, end_iso),
                ):
                    try:
                        for nid in _json.loads(r[0] or '[]'):
                            if nid not in _seen:
                                _seen.add(nid)
                                time_node_ids.append(nid)
                    except Exception:
                        continue
            except Exception as e:
                brain._log_error('fetch_by_time_discussed', e,
                                 'trace scan for discussed anchor failed')
        else:
            return []  # unknown anchor

        # Resolve history-sourced ids forward to their live survivors
        # (docs/TRACE-NODE-RESOLUTION.md, site #1). The 'discussed' anchor reads ids from
        # immutable surface_selected traces; a node live when surfaced may
        # have been absorbed/archived since. resolve_live redirects each to
        # the live node it became — "the thing we discussed survives as its
        # descendant" — and marks true orphans (no live survivor). Orphans are
        # dropped and COUNTED via a low-severity warning (§6: routine retrieval,
        # NOT the loud error the dashboard surfaces) so survivor-pointer rot
        # stays observable during the recovery window without re-spamming.
        # Backstop scope: the execute_tool tripwire downstream guards the NODE
        # feed only — it matches ids against the nodes table, and edge
        # candidates carry edge_ids it can't inspect, so the edge feed keeps
        # its own archived-endpoint gate in _fetch_edges_with_endpoints.
        # 'created'/'updated'/'event' already filter archived in SQL, so for
        # them this is a single passthrough query. resolve_live keeps input
        # order, so a survivor takes its newest input's place.
        if time_node_ids:
            _resolved = brain.resolve_live(time_node_ids, on_orphan='mark')
            time_node_ids = list(_resolved['live'])
            _orphans = _resolved.get('orphans') or []
            if _orphans:
                brain._log_warning(
                    'fetch_by_time_orphans_dropped',
                    '%d discussed-anchor ids had no live survivor (anchor=%s)'
                    % (len(_orphans), time_anchor),
                    'dropped; sample=%s' % sorted(_orphans)[:5])

        # 3. Semantic matches (if query). Over-fetch so tiers can fill.
        semantic_results: List[Dict[str, Any]] = []
        if query and query.strip():
            try:
                raw = brain.recall(query=query, limit=int(limit) * 3)
                if isinstance(raw, dict):
                    raw = raw.get('results') or raw.get('items') or []
                semantic_results = list(raw or [])
            except Exception as e:
                brain._log_error('fetch_by_time_semantic', e,
                                 'recall failed; query tiers dropped, time tier only')
                semantic_results = []
        semantic_node_ids = {
            r.get('id') for r in semantic_results if r.get('id')
        }

        time_node_set = set(time_node_ids)

        # 4. Compute tiers. Query-tier scores are recall's own (recall_score),
        # pool-comparable like recall_topical's; tier precedence lives in the
        # emission order and cand['tier'], not in the score.
        from servers.scales.s1.surface_contract import recall_score
        out: List[Dict[str, Any]] = []

        def _emit(node_or_dict, tier: str, score: float) -> None:
            if len(out) >= limit:
                return
            cand = _to_candidate(node_or_dict, score, 'recall_by_time')
            if cand:
                cand['tier'] = tier
                out.append(cand)

        # Tier 1: query ∩ time (semantic_results filtered to time-matched ids)
        for r in semantic_results:
            if len(out) >= limit:
                break
            if r.get('id') in time_node_set:
                _emit(r, 'query+time', recall_score(r))

        # Tier 2: query only (semantic_results not in time)
        for r in semantic_results:
            if len(out) >= limit:
                break
            if r.get('id') not in time_node_set:
                _emit(r, 'query', recall_score(r))

        # Tier 3: time only — nodes not already covered + edges with endpoints
        if len(out) < limit:
            tier3_node_ids = [nid for nid in time_node_ids
                              if nid not in semantic_node_ids][:limit]
            if tier3_node_ids:
                try:
                    nodes_map = brain.get_node(tier3_node_ids)
                    if isinstance(nodes_map, dict):
                        for nid in tier3_node_ids:
                            if len(out) >= limit:
                                break
                            if nid in nodes_map:
                                _emit(nodes_map[nid], 'time', 0.5)
                except Exception as e:
                    brain._log_error('fetch_by_time_tier3', e,
                                     'time-only tier dropped')

        if len(out) < limit and time_edge_ids:
            edge_cands = _fetch_edges_with_endpoints(
                brain, time_edge_ids[:limit])
            for c in edge_cands:
                if len(out) >= limit:
                    break
                out.append(c)

        return out[:int(limit)]
    except Exception as e:
        brain._log_error('fetch_recall_by_time', e, 'surface fetch tool failed; returned no candidates')
        print('[fetch_tools] recall_by_time failed: %s' % e, file=sys.stderr)
        return []


def expand_node(brain, node_ref: str = '', hops: int = 1, **_) -> List[Dict[str, Any]]:
    """Constellation expansion from a node — wraps traverse() from
    pipeline_contract.py. Accepts node_id (≥8 char) or fuzzy title."""
    try:
        from servers.pipeline_contract import traverse
        # Resolve node_ref to a real id
        node_id = None
        if node_ref and len(node_ref) >= 8 and re.match(r'^[0-9a-f]+$', node_ref[:8]):
            # Looks like an id (full or 8-char prefix)
            n = brain.get_node(node_ref)
            if n:
                node_id = n.get('id') if isinstance(n, dict) else None
        if not node_id and node_ref:
            # Fuzzy title lookup
            found = brain.find_node_by_title(node_ref)
            if isinstance(found, dict):
                node_id = found.get('id')
            elif isinstance(found, list) and found:
                node_id = found[0].get('id') if isinstance(found[0], dict) else None
        if not node_id:
            return []
        traversed = traverse(brain, [node_id], depth=int(hops), limit_per_seed=20)
        # traverse() returns {'neighbors': [...], 'corrections': {...},
        # 'metadata': {...}} — only neighbors matter here: execute_tool's
        # boundary enrichment re-attaches _corrections + metadata via a
        # batched get_node pass on every tool result.
        neighbors = (traversed or {}).get('neighbors') or []
        out = []
        seen = set()
        for n in neighbors:
            nid = n.get('id') if isinstance(n, dict) else None
            if not nid or nid == node_id or nid in seen:
                continue
            seen.add(nid)
            cand = _to_candidate(n, 0.65, 'expand_node')
            if cand:
                out.append(cand)
        return out
    except Exception as e:
        brain._log_error('fetch_expand_node', e, 'surface fetch tool failed; returned no candidates')
        print('[fetch_tools] expand_node failed: %s' % e, file=sys.stderr)
        return []


# ─── Tool dispatch ───────────────────────────────────────────────────────

_TOOL_FN_MAP = {
    'recall_topical':    recall_topical,
    'recall_by_time':    recall_by_time,
    'expand_node':       expand_node,
}


# Haiku's tool_use serialization can leak function-call markup into string
# arg values (observed 2×/1,848 calls, both recall_by_time end_when, shape
# '</antml parameter>\n<parameter name=...>'). The values are schema-valid
# strings, so no upstream layer catches them; unguarded they degrade silently
# (date parse → None → unbounded window). Matches XML-ish tags, the antml
# literal, and control chars — not bare '<' or the word 'parameter'.
_MARKUP_ARG_RE = re.compile(r'</?[a-zA-Z_][^>]*>|antml|[\x00-\x08\x0b-\x1f]')


def execute_tool(brain, tool_name: str, tool_input: Dict[str, Any],
                 session_id: str = '') -> Dict[str, Any]:
    """Execute a single tool call. Returns {results, latency_ms, error?,
    dropped_args?}.

    Every tool's output passes through one batched brain.get_node() pass —
    so _corrections, full metadata, and connections come along regardless
    of which tool fired. Tool authors only need to return ID-bearing dicts;
    they can't forget to enrich because the boundary does it.
    """
    fn = _TOOL_FN_MAP.get(tool_name)
    if fn is None:
        return {'results': [], 'latency_ms': 0,
                'error': 'unknown_tool: %s' % tool_name}
    t0 = time.time()
    kwargs = dict(tool_input or {})
    # Arg-sanity guard: drop markup-corrupted string args, keep the call.
    # Behavior matches the pre-guard degradation (the corrupt value never
    # parsed anyway) — the difference is the loud error. Dropping a REQUIRED
    # arg (e.g. query) makes fn raise TypeError below → error result → Haiku
    # falls back to the candidate pool; still non-breaking.
    dropped_args = {}
    for k, v in list(kwargs.items()):
        if isinstance(v, str) and _MARKUP_ARG_RE.search(v):
            dropped_args[k] = v[:200]
            del kwargs[k]
    if dropped_args and brain is not None:
        try:
            brain._log_error(
                'surface_malformed_tool_arg',
                ValueError('markup in tool_use args: %r' % dropped_args),
                'tool=%s session=%s — arg(s) dropped, call continued'
                % (tool_name, session_id))
        except Exception:
            pass
    try:
        results = fn(brain, **kwargs)
    except Exception as e:
        out = {'results': [], 'latency_ms': int((time.time() - t0) * 1000),
               'error': str(e)[:200]}
        if dropped_args:
            out['dropped_args'] = dropped_args
        return out

    # Unified enrichment — single source of "tool results are fully rich".
    # Rich fields from get_node fill missing keys; recall-specific fields
    # set by _to_candidate (score, discovery) are preserved.
    if results and brain is not None:
        ids = [r.get('id') for r in results
               if isinstance(r, dict) and r.get('id')]
        if ids:
            try:
                rich_map = brain.get_node(ids) or {}
            except Exception as e:
                # Best-effort: if enrichment fails, fall back to whatever the
                # tool returned. Log so the failure surfaces.
                try:
                    brain._log_error(
                        'execute_tool_enrich', e,
                        'batched get_node enrichment for tool=%s' % tool_name)
                except Exception:
                    pass
                rich_map = {}
            for r in results:
                rid = r.get('id') if isinstance(r, dict) else None
                if rid and rid in rich_map:
                    for k, v in rich_map[rid].items():
                        if k not in r:
                            r[k] = v
        # Tripwire — NOT a filter layer. Every producer is responsible for
        # never emitting archived ids (gated at source in each tool). If one
        # ever slips through, that is a producer BUG: log as an error EVERY
        # time and drop it rather than feed Haiku a dead node (the
        # spread_seed_no_vectors incident class). This should never fire in
        # healthy operation — any entry in the errors table here means a
        # producer regressed.
        try:
            _dead = brain._nodes.archived_subset(
                [r.get('id') for r in results if isinstance(r, dict)])
        except Exception:
            _dead = set()
        if _dead:
            try:
                brain._log_error(
                    'fetch_tool_archived_tripwire',
                    RuntimeError('tool %s emitted %d archived node ids — '
                                 'producer bug, fix the tool not this '
                                 'tripwire' % (tool_name, len(_dead))),
                    'dropped; sample=%s' % sorted(_dead)[:5])
            except Exception:
                pass
            results = [r for r in results
                       if not (isinstance(r, dict) and r.get('id') in _dead)]

    # Scope veil — same wall brain.recall applies, here because tool
    # functions reach brain primitives that may bypass recall
    # (recall_by_time's trace lane, etc.). One set-membership pass; a veil
    # failure FAILS CLOSED (empty results + loud error) — a degraded tool
    # beats a leaked wall (default-deny is the contract).
    if results and brain is not None:
        try:
            _veil = brain.scope_veil(session_id)
            if _veil:
                from servers.scopes import scrub_node
                results = [r for r in results
                           if isinstance(r, dict) and r.get('id') not in _veil]
                # Survivors were get_node-enriched above — scrub their edge
                # attachments (connections carry walled titles to Haiku's
                # menu; _corrections carry the walled corrector's FULL text).
                for r in results:
                    scrub_node(r, _veil)
        except Exception as e:
            try:
                brain._log_error(
                    'fetch_tool_scope_gate', e,
                    'veil unavailable for tool=%s — failing CLOSED '
                    '(results withheld)' % tool_name)
            except Exception:
                pass
            results = []

    out = {'results': results or [],
           'latency_ms': int((time.time() - t0) * 1000)}
    if dropped_args:
        out['dropped_args'] = dropped_args
    return out


def format_tool_result_for_haiku(result: Dict[str, Any], scope=None) -> str:
    """Format a tool's output using the SAME renderer as the initial 25
    cosine candidates — `format_candidate_for_surface`. Tool results have
    already been fully enriched by `execute_tool()` (batched brain.get_node
    pass attaches _corrections, _metadata, connections), so this function
    just hands the rich shape to the formatter. One renderer, no arguments
    to disagree on: the pool and the tool results cannot speak different
    grammars inside one prompt.
    """
    results = result.get('results') or []
    if result.get('error'):
        return 'ERROR: %s' % result['error']
    if not results:
        return 'No results.'

    from servers.scales.s1.surface_contract import format_candidate_for_surface

    lines = ['%d results (evaluate on the same merit as the initial 25):'
             % len(results)]

    for r in results[:25]:
        if not isinstance(r, dict):
            continue
        lines.append('')
        lines.append(format_candidate_for_surface(r, scope=scope))

    if len(results) > 25:
        lines.append('')
        lines.append('... (%d more truncated)' % (len(results) - 25))
    return '\n'.join(lines)


__all__ = [
    'TOOL_DEFINITIONS',
    'execute_tool',
    'format_tool_result_for_haiku',
    'recall_topical', 'recall_by_time',
    'expand_node',
]
