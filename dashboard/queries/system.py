"""System status — live/dead checks across every brain component.

Every DB read routes through `db.ro_connect`.
"""

import json
import os

from ..clock import utc_cutoff
from ..daemon_client import daemon_send
from ..db import brain_db_path, logs_db_path, ro_connect
from ..log import warn
from ._meta import EMPTY_SELECTION_REF_ID, REGISTER_ONLY_FIELD

# Surface runs once per user turn, so its card reads Down after a quiet stretch
# longer than this rather than staying green on old rows.
SURFACE_LIVE_HOURS = 24

# A surface run lands seconds after its turn (the recall hook's budget is ~20s),
# so a turn older than this with no run after it was skipped or failed.
SURFACE_GRACE_MINUTES = 1

# One turn can legitimately skip surface (every candidate already shown, no
# recall hits); this many consecutive turns without a run is an outage — a
# refused API key pauses the LLM layer and surface stops writing at all.
SURFACE_MISSED_TURNS = 2


def daemon_record() -> dict:
    """The dashboard's one view of the daemon, from its ping — the status
    card, the stats bar's encode chip and the Config tab all read this record
    rather than mapping ping fields themselves.

    Through `daemon_send`, which returns exactly the ping payload and takes a
    timeout — a hand-rolled socket here would miss the newline framing the
    module's client has. `uptime_seconds` is the loaded code image's age,
    from dispatch_ops._handle_ping — an in-place reload resets it while the
    PID stays the same. `db_dir` is the brain the daemon actually writes;
    `encode_every` the Scribe cadence it read at start. {'alive': False}
    with an error when the daemon does not answer within 3 s — long enough
    for a daemon mid-dispatch to answer, short enough for the 5 s status
    poll; the stats bar carried this value before the record was shared."""
    result = daemon_send("ping", timeout=3)
    if result is None:
        return {'alive': False, 'error': 'no response'}
    return {
        'alive': True, 'pid': result.get('pid', '?'),
        'uptime': result.get('uptime_seconds', 0),
        'code_fingerprint': (result.get('code_fingerprint') or '')[:12],
        'db_dir': result.get('db_dir') or '',
        'source_dir': result.get('source_dir') or '',
        'encode_every': result.get('encode_every'),
    }


def _check_brain_db() -> dict:
    """Brain DB liveness: open + count nodes + report size."""
    path = brain_db_path()
    try:
        with ro_connect(path, timeout=2) as conn:
            if conn is None:
                return {'alive': False, 'error': 'File not found'}
            count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        size_mb = round(os.path.getsize(path) / 1048576, 1)
        return {'alive': True, 'nodes': count, 'path': path, 'size_mb': size_mb}
    except Exception as e:
        warn('queries.system', 'brain_db check failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def _check_logs_db() -> dict:
    """Logs DB liveness: open + trivial SELECT + report size."""
    path = logs_db_path()
    try:
        with ro_connect(path, timeout=2) as conn:
            if conn is None:
                return {'alive': False, 'error': 'File not found'}
            conn.execute("SELECT 1").fetchone()
        size_mb = round(os.path.getsize(path) / 1048576, 1)
        return {'alive': True, 'path': path, 'size_mb': size_mb}
    except Exception as e:
        warn('queries.system', 'logs_db check failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def _surface_stopped(conn) -> bool:
    """True when each of the newest SURFACE_MISSED_TURNS user turns that should
    have run surface (not registered without it; past the grace window, inside
    the live window) has no surface run after it in its own session."""
    turns = conn.execute(
        "SELECT session_id, created_at FROM trace_events "
        "WHERE scale = 's0' AND ref_type = 'user_message' "
        "AND created_at > ? AND created_at < ? "
        "AND (CASE WHEN json_valid(metadata) THEN json_extract(metadata, ?) END) IS NULL "
        "ORDER BY created_at DESC LIMIT ?",
        (utc_cutoff(hours=SURFACE_LIVE_HOURS),
         utc_cutoff(minutes=SURFACE_GRACE_MINUTES), '$.' + REGISTER_ONLY_FIELD,
         SURFACE_MISSED_TURNS)).fetchall()
    return len(turns) == SURFACE_MISSED_TURNS and all(conn.execute(
        "SELECT 1 FROM trace_events "
        "WHERE scale = 's1' AND event_type = 'K' AND ref_type = 'surface_selected' "
        "AND session_id = ? AND created_at >= ? LIMIT 1",
        (session_id, created_at)).fetchone() is None
        for session_id, created_at in turns)


def _check_surface() -> dict:
    """Surface health over its last 20 S1 K `surface_selected` runs: when the
    newest landed, how many picked something, and the model that ran it.
    Down when nothing ran in SURFACE_LIVE_HOURS, or when the newest user turns
    got no run — surface stopped while the operator kept talking."""
    try:
        with ro_connect(logs_db_path(), timeout=2) as conn:
            if conn is None:
                return {'alive': False, 'error': 'logs DB unavailable'}
            rows = conn.execute(
                "SELECT ref_id, created_at, "
                "CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.model') END "
                "FROM trace_events "
                "WHERE scale = 's1' AND event_type = 'K' AND ref_type = 'surface_selected' "
                "ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            if not rows or (rows[0][1] or '') < utc_cutoff(hours=SURFACE_LIVE_HOURS):
                return {'alive': False, 'error': 'no run in %dh' % SURFACE_LIVE_HOURS}
            if _surface_stopped(conn):
                return {'alive': False,
                        'error': 'no run for the last %d turns' % SURFACE_MISSED_TURNS}
        picked = sum(1 for r in rows if r[0] != EMPTY_SELECTION_REF_ID)
        return {
            'alive': True,
            'last_run': rows[0][1],
            'model': rows[0][2] or '',
            'sample': '%d/%d picked something' % (picked, len(rows)),
        }
    except Exception as e:
        warn('queries.system', 'surface health check failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def _check_embedder() -> dict:
    """Embedder status — read from /tmp status file written by the daemon."""
    try:
        # Instance-keyed like daemon_config.get_status_path() (the dashboard is
        # deliberately servers-decoupled, so it mirrors the env protocol).
        inst = os.environ.get("BRAIN_INSTANCE", "").strip()
        status_path = "/tmp/brain-status-%d%s.json" % (
            os.getuid(), "-" + inst if inst else "")
        if not os.path.exists(status_path):
            return {'alive': False, 'error': 'No status file'}
        with open(status_path) as f:
            ds = json.load(f)
        return {
            'alive': ds.get('embedder_ready', False),
            'model': ds.get('model_name', '?'),
        }
    except Exception as e:
        warn('queries.system', 'embedder status read failed', exc=e)
        return {'alive': False, 'error': str(e)[:100]}


def check_system_status() -> dict:
    """Probe each component and return a dict of {component: {alive, ...}}.

    Each checker is its own function — adding a new component is one new
    `_check_X()` + one dict entry below.
    """
    return {
        'daemon':   daemon_record(),
        'brain_db': _check_brain_db(),
        'logs_db':  _check_logs_db(),
        'surface':  _check_surface(),
        'embedder': _check_embedder(),
    }
