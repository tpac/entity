"""Insights scanner surfaces surface-id corruption as a proactive alert.

The Logs tab already shows surface_id_fuzzy_recovered / surface_unknown_selected_id
rows (event_type='warning'), but that's passive. This rule promotes a recurring
rate to the Live tab's insight panel so the drift ALERTS, not just logs — the
'if I don't see it in the dashboard it doesn't exist' requirement. Empty-selection
scanning can't cover it: recovered picks keep ref_id non-empty, so the pipeline
looks healthy while the emission degrades.

Tests call the undecorated rule (`.__wrapped__`) with an in-memory debug_log so
they exercise the SQL + severity logic without a real brain.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

from dashboard.queries import insights_scanner as ins


def _conn(rows):
    """In-memory debug_log with (source, event_type, created_at) rows."""
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE debug_log (id INTEGER PRIMARY KEY, source TEXT, "
              "event_type TEXT, created_at TEXT, metadata TEXT)")
    now = datetime.now(timezone.utc)
    for i, (source, event_type, age_h) in enumerate(rows):
        ts = (now - timedelta(hours=age_h)).isoformat()
        c.execute("INSERT INTO debug_log (id, source, event_type, created_at, "
                  "metadata) VALUES (?,?,?,?,?)",
                  (i, source, event_type, ts, "{}"))
    c.commit()
    return c


scan = ins._scan_surface_id_drift.__wrapped__   # bypass @safe_query conn-open


def test_below_min_is_quiet():
    # 2 combined events < SURFACE_ID_DRIFT_MIN (3) → no card (Logs tab still
    # has them; the insight panel is a rate signal, not a per-event alarm).
    c = _conn([('surface_id_fuzzy_recovered', 'warning', 1),
               ('surface_unknown_selected_id', 'warning', 2)])
    assert scan(c) == []


def test_recovered_only_flags_medium():
    # 3 recovered, 0 lost → flag, medium (working but degrading).
    c = _conn([('surface_id_fuzzy_recovered', 'warning', 1)] * 3)
    out = scan(c)
    assert len(out) == 1
    assert out[0]['severity'] == 'medium'
    assert out[0]['evidence'] == {'lost': 0, 'recovered': 3}
    assert '0 lost, 3 recovered' in out[0]['title']


def test_lost_picks_escalate_to_high():
    # 5 lost (unresolvable) picks ≥ SURFACE_ID_DRIFT_HIGH → high — the silent
    # context-loss class that produced the v12.1 wrong abstention.
    c = _conn([('surface_unknown_selected_id', 'warning', 3)] * 5)
    out = scan(c)
    assert len(out) == 1
    assert out[0]['severity'] == 'high'
    assert out[0]['evidence'] == {'lost': 5, 'recovered': 0}


def test_ignores_errors_and_other_sources_and_old_rows():
    # An error-typed row of the same source name, an unrelated warning, and a
    # 30h-old (outside 24h) drift row must all be excluded — leaving 2 in-window
    # drift warnings, which is below MIN → quiet.
    c = _conn([('surface_unknown_selected_id', 'error', 1),       # wrong type
               ('some_other_warning', 'warning', 1),              # wrong source
               ('surface_id_fuzzy_recovered', 'warning', 30),     # too old
               ('surface_id_fuzzy_recovered', 'warning', 1),
               ('surface_unknown_selected_id', 'warning', 2)])
    assert scan(c) == []


def test_wired_into_scan_all():
    # Regression guard: the rule must be in the aggregator, or it never renders.
    import inspect
    assert '_scan_surface_id_drift' in inspect.getsource(ins.scan_all)
