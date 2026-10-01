"""Unified logs view — pulls from every component that records errors AND
warnings.

Sources: brain `debug_log` (event_type='error' and 'warning') and `hook_errors`
(any level — incl. DAEMON_DOWN, written by both the hook-side detector and the
MCP health monitor). Each source contributes a row with a uniform shape so the
UI can render them in one feed; the per-row `level` ('error'/'warning'/...) lets
the Logs tab colour and filter by severity.

Each row carries full `error` / `context` / `traceback` text (capped, not
list-truncated) so the Logs tab's click-to-expand can show the whole story —
the hook traceback in particular was recorded but never surfaced.

Sources are declared as a config list and one helper runs each with
loud-by-default warnings on any failure.
"""

import json

from ..clock import utc_cutoff
from ..db import logs_db_path, ro_connect
from ..log import warn


def _decode_meta(raw):
    """Parse a JSON metadata blob, return {} on failure. Inner-row silence
    is intentional — old debug_log rows often have non-JSON `metadata`."""
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}


# Shape: a row mapper takes the SQL row tuple and returns the uniform error
# dict the UI consumes. Each source has its own mapper because column order
# differs per table, but the OUTPUT shape is identical.

# Caps: generous, not list-truncation. The list view clamps display in CSS;
# the click-to-expand wants the whole message / context / traceback.
_ERR_CAP, _CTX_CAP, _TB_CAP = 2000, 1000, 6000


def _shape_brain(r):
    meta = _decode_meta(r[3])
    event_type = r[4] if len(r) > 4 else 'error'
    # _log_error stores the text under 'error'; _log_warning under 'message'.
    # Fall back to the raw metadata blob only if neither is present.
    message = meta.get('error') or meta.get('message') or (r[3] or '')
    return {
        'source': 'brain', 'component': r[2], 'timestamp': r[1],
        'error': message[:_ERR_CAP],
        'context': (meta.get('context', '') or '')[:_CTX_CAP],
        'traceback': (meta.get('traceback', '') or '')[:_TB_CAP],
        # Warnings carry event_type='warning' with no metadata level; errors
        # default to 'error'. An explicit metadata level, if ever set, wins.
        'level': meta.get('level') or event_type or 'error',
    }


def _shape_hook(r):
    return {
        'source': 'hook', 'component': r[2], 'timestamp': r[1],
        'error': (r[4] or '')[:_ERR_CAP], 'context': (r[5] or '')[:_CTX_CAP],
        'traceback': (r[6] or '')[:_TB_CAP],
        'level': r[3] or 'error',
    }


# (db_path_fn, sql, shape_fn, label) — label is for the warn() message.
# Sources are intentionally explicit (not auto-discovered): adding a new
# error source is a deliberate one-line addition here.
_SOURCES = [
    (logs_db_path,
     "SELECT id, created_at, source, metadata, event_type FROM debug_log "
     "WHERE event_type IN ('error', 'warning') AND created_at > ? "
     "ORDER BY created_at DESC LIMIT ?",
     _shape_brain, 'brain debug_log'),
    (logs_db_path,
     "SELECT id, created_at, hook_name, level, error, context, traceback FROM hook_errors "
     "WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
     _shape_hook, 'hook_errors'),
]


def _pull_source(db_path_fn, sql, shape_fn, label, since_ts, limit):
    """Run one source's query. On any failure: warn loudly, return []."""
    try:
        with ro_connect(db_path_fn()) as conn:
            if conn is None:
                return []
            rows = conn.execute(sql, (since_ts, limit)).fetchall()
            return [shape_fn(r) for r in rows]
    except Exception as e:
        warn('queries.errors', '%s pull failed' % label, exc=e)
        return []


def query_all_errors(limit: int = 50, hours: int = 24):
    """Aggregate errors from every component into one timestamp-sorted list."""
    since_ts = utc_cutoff(hours=hours)
    errors = []
    for db_fn, sql, shape, label in _SOURCES:
        errors.extend(_pull_source(db_fn, sql, shape, label, since_ts, limit))
    errors.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
    return errors[:limit]
