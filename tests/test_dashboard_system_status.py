"""Surface health on the dashboard: the Overview status card reads the live
`surface_selected` trace and goes Down when no run landed recently (never green
on old rows) or when the newest user turns got no run; it and the Live
tab's empty-selection insight count empties by the one shared empty ref_id.
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.queries import insights_scanner, system
from dashboard.queries._meta import EMPTY_SELECTION_REF_ID, REGISTER_ONLY_FIELD


def _logs_db(tmp_path, rows, turns=()):
    """rows: list of (ref_type, ref_id, age_hours, metadata_dict[, session_id])
    s1 K rows; turns: list of (summary, age_minutes, session_id[, metadata])
    s0 user turns."""
    conn = sqlite3.connect(tmp_path / "brain_logs.db")
    conn.execute(
        "CREATE TABLE trace_events (id TEXT, scale TEXT, event_type TEXT, "
        "ref_type TEXT, ref_id TEXT, summary TEXT, created_at TEXT, metadata TEXT, "
        "session_id TEXT)")
    now = datetime.now(timezone.utc)
    for i, (ref_type, ref_id, age_h, meta, *session) in enumerate(rows):
        conn.execute(
            "INSERT INTO trace_events VALUES (?,?,?,?,?,?,?,?,?)",
            ("t%d" % i, "s1", "K", ref_type, ref_id, "",
             (now - timedelta(hours=age_h)).isoformat(), json.dumps(meta),
             session[0] if session else "sess-a"))
    for i, (summary, age_min, session_id, *meta) in enumerate(turns):
        conn.execute(
            "INSERT INTO trace_events VALUES (?,?,?,?,?,?,?,?,?)",
            ("u%d" % i, "s0", "K", "user_message", "", summary,
             (now - timedelta(minutes=age_min)).isoformat(),
             json.dumps(meta[0]) if meta else None, session_id))
    conn.commit()
    conn.close()


def _check(tmp_path, monkeypatch, rows, turns=()):
    _logs_db(tmp_path, rows, turns)
    monkeypatch.setenv("BRAIN_DB_DIR", str(tmp_path))
    return system._check_surface()


def _run(age_min, session_id="sess-a"):
    return ("surface_selected", '["a1b2c3d4"]', age_min / 60, {}, session_id)


_MISSED = {"alive": False,
           "error": "no run for the last %d turns" % system.SURFACE_MISSED_TURNS}


def _two_turns(session_id="sess-a"):
    """Two user turns that ran recall, 10 and 5 minutes old (unstamped, with
    the metadata a real row carries)."""
    meta = {"recall_chain": "s1r-x-1"}
    return [("why did the encoder skip that turn?", 10, session_id, meta),
            ("try it again with the new prompt", 5, session_id, meta)]


def test_live_run_reports_rate_and_model(tmp_path, monkeypatch):
    s = _check(tmp_path, monkeypatch, [
        ("surface_selected", '["a1b2c3d4"]', 0.1, {"model": "claude-haiku-4-5"}),
        ("surface_selected", "[]", 0.2, {"model": "claude-haiku-4-5"}),
        ("surface_selected", '["e5f6a7b8", "c9d0e1f2"]', 0.3, {}),
    ])
    assert s["alive"] is True
    assert s["sample"] == "2/3 picked something"
    assert s["model"] == "claude-haiku-4-5"


def test_empty_selection_counts_as_nothing_picked(tmp_path, monkeypatch):
    s = _check(tmp_path, monkeypatch, [
        ("surface_selected", EMPTY_SELECTION_REF_ID, 0.1, {}),
        ("surface_selected", '["a1b2c3d4"]', 0.2, {}),
    ])
    assert s["sample"] == "1/2 picked something"


def test_old_rows_read_down(tmp_path, monkeypatch):
    s = _check(tmp_path, monkeypatch, [
        ("surface_selected", '["a1b2c3d4"]', system.SURFACE_LIVE_HOURS + 1, {}),
    ])
    assert s == {"alive": False, "error": "no run in %dh" % system.SURFACE_LIVE_HOURS}


def test_other_ref_types_do_not_count(tmp_path, monkeypatch):
    # The card read a renamed ref_type for six months and stayed green on
    # rows nothing writes any more; only the live writer's rows count.
    s = _check(tmp_path, monkeypatch, [
        ("judge_selected", '["a1b2c3d4"]', 0.1, {}),
    ])
    assert s == {"alive": False, "error": "no run in %dh" % system.SURFACE_LIVE_HOURS}


@pytest.mark.parametrize("run_age_min, down", [(30, True), (9, False)])
def test_real_turns_without_a_run_read_down(tmp_path, monkeypatch, run_age_min, down):
    # A refused API key pauses the LLM layer: the operator keeps talking, surface
    # writes nothing, and the last run is minutes old — the 24h bound says Live.
    # A run after the older turn leaves one unsurfaced turn: still Live.
    s = _check(tmp_path, monkeypatch, [_run(run_age_min)], turns=_two_turns())
    assert (s == _MISSED) is down and s["alive"] is not down


def test_turn_inside_the_grace_window_is_not_counted(tmp_path, monkeypatch):
    # The run lands seconds after its turn; a turn this fresh is still in flight.
    s = _check(tmp_path, monkeypatch, [_run(9)], turns=_two_turns() + [
        ("and now?", system.SURFACE_GRACE_MINUTES / 3, "sess-a"),
    ])
    assert s["alive"] is True


def test_turns_registered_without_surface_do_not_count(tmp_path, monkeypatch):
    # Short answers and wake envelopes: the hook stamps them and runs no surface.
    stamp = {"recall_chain": "s1r-x-1", REGISTER_ONLY_FIELD: True}
    s = _check(tmp_path, monkeypatch, [_run(19)], turns=[
        ("why did the encoder skip that turn?", 20, "sess-a"),
        ("<task-notification><task-id>t1</task-id>", 10, "sess-a", stamp),
        ("ok", 5, "sess-a", stamp),
    ])
    assert s["alive"] is True


def test_a_run_in_another_session_does_not_cover_the_turn(tmp_path, monkeypatch):
    s = _check(tmp_path, monkeypatch, [_run(1, "sess-a")], turns=_two_turns("sess-b"))
    assert s == _MISSED


def test_empty_selection_insight_counts_the_shared_empty_ref_id():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE trace_events (scale TEXT, event_type TEXT, "
                 "ref_type TEXT, ref_id TEXT, created_at TEXT)")
    now = datetime.now(timezone.utc).isoformat()
    n = insights_scanner.EMPTY_SELECTION_MIN_SAMPLES
    for _ in range(n):
        conn.execute("INSERT INTO trace_events VALUES ('s1','K','surface_selected',?,?)",
                     (EMPTY_SELECTION_REF_ID, now))
    (card,) = insights_scanner._scan_empty_selections.__wrapped__(conn)
    assert card["evidence"] == {"empty_count": n, "total_count": n, "rate": 1.0}
