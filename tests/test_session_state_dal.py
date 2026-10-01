"""Unit tests for SessionStateDAL's session-context lifecycle methods.

These three methods were added when the raw session_state SQL in brain.py
(get_or_create_session / present_streams / live_sessions) was migrated onto
the DAL (DAL cleanup Phase 4). The load-bearing semantic is ensure_default's
INSERT-OR-IGNORE: it must NOT clobber a racing thread's already-mutated row,
which is exactly what distinguishes it from set()'s upsert.

Run: BRAIN_ALLOW_ANY_PYTHON=1 python3 -m pytest tests/test_session_state_dal.py -v
"""
import sqlite3
import pytest

from servers.dal_logs import SessionStateDAL

_DDL = """CREATE TABLE session_state (
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    node_id TEXT NOT NULL DEFAULT '',
    value TEXT,
    updated_at TEXT,
    PRIMARY KEY (session_id, key, node_id)
)"""


@pytest.fixture
def dal():
    conn = sqlite3.connect(":memory:")
    conn.execute(_DDL)
    conn.commit()
    return SessionStateDAL(conn)


class TestEnsureDefault:
    def test_inserts_when_absent(self, dal):
        dal.ensure_default("s1", "_session_context", '{"stop_counter": 0}')
        assert dal.get("s1", "_session_context") == '{"stop_counter": 0}'

    def test_does_not_clobber_existing(self, dal):
        # A racing thread already wrote live state under this key...
        dal.set("s1", "_session_context", '{"stop_counter": 7}')
        # ...ensure_default must preserve it, not reset to the default.
        dal.ensure_default("s1", "_session_context", '{"stop_counter": 0}')
        assert dal.get("s1", "_session_context") == '{"stop_counter": 7}'

    def test_set_does_clobber(self, dal):
        # Contrast: set() is an upsert — it overwrites. This is why a
        # separate ensure_default method exists.
        dal.set("s1", "_session_context", '{"stop_counter": 7}')
        dal.set("s1", "_session_context", '{"stop_counter": 0}')
        assert dal.get("s1", "_session_context") == '{"stop_counter": 0}'
