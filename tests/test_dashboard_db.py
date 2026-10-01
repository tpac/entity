"""Unit tests for dashboard.db.fetch_by_id — the shared by-id fetch idiom
that replaced hand-rolled placeholder + `WHERE id IN` + by-id-dict code in
queries/{s2_runs,encoding,explorer}.py.
"""
import sqlite3

from dashboard.db import fetch_by_id


def _nodes_conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT, "
              "archived INTEGER DEFAULT 0)")
    c.executemany("INSERT INTO nodes VALUES (?,?,?)",
                  [("a", "principle", 0), ("b", "fact", 1), ("c", "lesson", 0)])
    c.commit()
    return c


def test_basic_fetch_keyed_by_id():
    out = fetch_by_id(_nodes_conn(), "nodes", "id, type, archived", ["a", "c"])
    assert set(out) == {"a", "c"}
    assert out["a"] == ("a", "principle", 0)  # first column is the dict key


def test_empty_ids_short_circuits():
    # No query issued, no empty `IN ()` syntax error — just {}.
    assert fetch_by_id(_nodes_conn(), "nodes", "id, type", []) == {}


def test_falsy_ids_dropped():
    out = fetch_by_id(_nodes_conn(), "nodes", "id, type", ["a", None, "", "c"])
    assert set(out) == {"a", "c"}


def test_missing_ids_simply_absent():
    out = fetch_by_id(_nodes_conn(), "nodes", "id, type", ["a", "nope"])
    assert set(out) == {"a"}  # unknown id is just not in the map, no error


def test_liveness_neutral_returns_archived():
    # Contract: a point-fetch-by-id returns archived rows too — callers gate
    # liveness themselves. The helper must never silently filter by archived.
    out = fetch_by_id(_nodes_conn(), "nodes", "id, type, archived", ["b"])
    assert "b" in out and out["b"][2] == 1  # b is archived=1, still returned


def test_arbitrary_table_and_columns():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE trace_events (id TEXT PRIMARY KEY, chain_id TEXT)")
    c.execute("INSERT INTO trace_events VALUES ('t1','chainX')")
    c.commit()
    out = fetch_by_id(c, "trace_events", "id, chain_id", ["t1", "t2"])
    assert out == {"t1": ("t1", "chainX")}
