"""query_insights() quote detection — structured metadata, not name strings.

The "Zero quotes preserved this week" insight must key on the canonical quote
storage (their_raw_quote / my_raw_quote in node_metadata_kv, plus
quote-typed nodes). Operator/agent names are per-install, so content LIKE
'%<name> said%'-style heuristics cannot be used — pinned here.

query_stats()'s encode chip shows the cadence the running daemon reports in
its ping, pinned here from both ends.
"""
import sqlite3
from types import SimpleNamespace

import pytest

from dashboard.clock import utc_cutoff
from dashboard.queries import stats
from dashboard.queries.stats import query_insights

ZERO_QUOTES_TITLE = "Zero quotes preserved this week"


def _make_brain_db(tmp_path, nodes, kv=()):
    """nodes: list of (id, title, type, content). kv: list of (node_id, key, value).
    All nodes are recent (inside the 7d window), unlocked, unarchived."""
    recent = utc_cutoff(days=1)
    conn = sqlite3.connect(tmp_path / "brain.db")
    conn.execute(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, title TEXT, type TEXT, "
        "content TEXT, created_at TEXT, archived INTEGER DEFAULT 0, "
        "locked INTEGER DEFAULT 0)")
    conn.execute("CREATE TABLE edges (source_id TEXT, target_id TEXT)")
    conn.execute(
        "CREATE TABLE node_metadata_kv (node_id TEXT, key TEXT, value TEXT)")
    conn.executemany(
        "INSERT INTO nodes (id, title, type, content, created_at) "
        "VALUES (?,?,?,?,?)",
        [(i, t, ty, c, recent) for i, t, ty, c in nodes])
    conn.executemany("INSERT INTO node_metadata_kv VALUES (?,?,?)", kv)
    conn.commit()
    conn.close()


def _plain_nodes(n):
    # Long content so the thin-nodes insight stays out of the way.
    return [("n%d" % i, "node %d" % i, "fact", "x" * 200) for i in range(n)]


def _zero_quotes_fired(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DB_DIR", str(tmp_path))
    return any(i["title"] == ZERO_QUOTES_TITLE for i in query_insights())


def test_fires_when_no_quote_metadata(tmp_path, monkeypatch):
    _make_brain_db(tmp_path, _plain_nodes(6))
    assert _zero_quotes_fired(tmp_path, monkeypatch)


@pytest.mark.parametrize("key", ["their_raw_quote", "my_raw_quote"])
def test_kv_quote_suppresses(tmp_path, monkeypatch, key):
    _make_brain_db(tmp_path, _plain_nodes(6), kv=[("n0", key, "exact words")])
    assert not _zero_quotes_fired(tmp_path, monkeypatch)


def test_quote_typed_node_suppresses(tmp_path, monkeypatch):
    nodes = _plain_nodes(6) + [("q1", "a line kept", "quote", "x" * 200)]
    _make_brain_db(tmp_path, nodes)
    assert not _zero_quotes_fired(tmp_path, monkeypatch)


def test_operator_name_in_content_does_not_count(tmp_path, monkeypatch):
    # A content LIKE '%<name> said%' heuristic is install-specific and wrong
    # on any brain whose operator has a different name. Name mentions in
    # content are NOT quote capture; only the structured fields count.
    nodes = _plain_nodes(6)
    nodes[0] = ("n0", "node 0", "fact", "Ada said this. Claude: replied. " + "x" * 200)
    _make_brain_db(tmp_path, nodes)
    assert _zero_quotes_fired(tmp_path, monkeypatch)


def test_quiet_below_activity_threshold(tmp_path, monkeypatch):
    # <=5 recent nodes: too little activity to judge — insight stays quiet.
    _make_brain_db(tmp_path, _plain_nodes(3))
    assert not _zero_quotes_fired(tmp_path, monkeypatch)


# ── Encode chip: the daemon's cadence ─────────────────────────────────────

def test_ping_reports_the_scribe_cadence():
    from servers.dispatch_ops import _handle_ping
    from servers.scales.s1.encode_contract import ENCODE_EVERY

    brain = SimpleNamespace(db_path='/brain/brain.db', eval_brain_since=lambda: None)
    assert _handle_ping(brain, {}, None)['result']['encode_every'] == ENCODE_EVERY


@pytest.mark.parametrize('ping, every, daemon', [
    ({'status': 'alive', 'encode_every': 7}, 7, 'alive'),
    (None, None, 'unavailable'),
])
def test_encode_chip_reads_the_daemon_cadence(tmp_path, monkeypatch, ping, every, daemon):
    monkeypatch.setenv("BRAIN_DB_DIR", str(tmp_path))
    monkeypatch.setattr(stats, 'daemon_send', lambda cmd, timeout=3: ping)
    d = stats.query_stats()
    assert d['daemon'] == daemon
    assert d['encoding'] == {'turns_since': 0, 'every': every, 'due': False}
