"""Logs tab surfaces WARNINGS as well as errors.

The dashboard's brain source used to query `debug_log WHERE event_type='error'`,
so every `_log_warning` row (event_type='warning') was invisible. These tests
pin both halves of the fix:

  * `_shape_brain` — reads the warning's `message` field and derives `level`
    from the `event_type` column (errors store text under `error`, warnings
    under `message`); an explicit metadata `level` (e.g. 'critical') still wins.
  * `query_all_errors` — the SQL now pulls BOTH event types, so a revert to
    `='error'` is caught here, not in the browser.

The dashboard's own error ring is pinned here too: its `total` keeps counting
after the ring drops its oldest entries — the Logs badge counts from it.
"""
import json
import sqlite3
from datetime import datetime, timezone

from dashboard import log
from dashboard.queries import errors


# ── _shape_brain (pure row → uniform dict) ──────────────────────────────────
# Row shape from the SELECT: (id, created_at, source, metadata, event_type).

def _row(metadata: dict, event_type: str):
    return (1, "2026-06-25T00:00:00+00:00", "some_component",
            json.dumps(metadata), event_type)


def test_shape_error_row():
    out = errors._shape_brain(_row(
        {"error": "boom", "context": "ctx", "traceback": "tb"}, "error"))
    assert out["level"] == "error"
    assert out["error"] == "boom"        # error text read from `error`
    assert out["context"] == "ctx"
    assert out["traceback"] == "tb"


def test_shape_warning_row_reads_message_field():
    # _log_warning stores text under `message`, not `error`, with no traceback.
    out = errors._shape_brain(_row(
        {"message": "deprecated path used", "context": "where"}, "warning"))
    assert out["level"] == "warning"     # derived from event_type column
    assert out["error"] == "deprecated path used"
    assert out["context"] == "where"
    assert out["traceback"] == ""


def test_shape_explicit_metadata_level_wins():
    # If a row ever carries an explicit level, it overrides the event_type.
    out = errors._shape_brain(_row(
        {"error": "fatal", "level": "critical"}, "error"))
    assert out["level"] == "critical"


def test_shape_legacy_row_without_event_type_defaults_error():
    # Backward-compat: a 4-tuple (pre-change SELECT) must not blow up and
    # defaults to 'error'.
    legacy = (1, "2026-06-25T00:00:00+00:00", "comp", json.dumps({"error": "x"}))
    out = errors._shape_brain(legacy)
    assert out["level"] == "error"
    assert out["error"] == "x"


# ── query_all_errors (guards the SQL `IN ('error','warning')`) ──────────────

def _seed_logs_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE debug_log (id INTEGER PRIMARY KEY, session_id TEXT, "
        "event_type TEXT, source TEXT, metadata TEXT, created_at TEXT)")
    conn.execute(
        "CREATE TABLE hook_errors (id INTEGER PRIMARY KEY, created_at TEXT, "
        "hook_name TEXT, level TEXT, error TEXT, context TEXT, traceback TEXT)")
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT INTO debug_log (event_type, source, metadata, created_at) "
        "VALUES (?,?,?,?)",
        [
            ("error",   "comp_a", json.dumps({"error": "an error"}),      now),
            ("warning", "comp_b", json.dumps({"message": "a warning"}),   now),
            ("debug",   "comp_c", json.dumps({"message": "just debug"}),  now),
        ])
    conn.commit()
    conn.close()


def test_query_all_errors_includes_warnings(tmp_path, monkeypatch):
    _seed_logs_db(str(tmp_path / "brain_logs.db"))
    monkeypatch.setenv("BRAIN_DB_DIR", str(tmp_path))

    rows = errors.query_all_errors(limit=50, hours=24)
    levels = {r["level"] for r in rows}
    messages = {r["error"] for r in rows}

    assert "warning" in levels        # the fix — previously absent
    assert "error" in levels
    assert "a warning" in messages    # warning message surfaced (not raw JSON)
    assert "an error" in messages
    # 'debug' event_type stays out — only error/warning are log-tab signal.
    assert "just debug" not in messages


# ── The dashboard's own error ring ──────────────────────────────────────────

def test_ring_total_keeps_counting_past_the_ring_cap():
    log.clear()
    try:
        cap = log._RING.maxlen
        for i in range(cap + 5):
            log.warn('test', 'failure %d' % i)
        assert len(log.recent(limit=cap + 5)) == cap
        assert log.total() == cap + 5
    finally:
        log.clear()
