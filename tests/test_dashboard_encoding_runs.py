"""Encoding-run card reads the authoritative delta, not the first one.

Regression guard for the 2026-07-03 "0 actions" bug: S1E (v29) writes many
delta rows per chain (edge_relation_revised per edge, node_revised per node,
journal_note per residue item, plus the encoding_run summary). The card query
must select the `encoding_run` delta — an unfiltered fetchone() grabs the first
row (an edge revision, no created/revised rollup) and the card shows "0 actions"
for runs that actually wrote. Also pins the journal-notes collection (rendered
first in the card details).

Calls the undecorated `_query_encoding_chains.__wrapped__` with an in-memory
logs_db so it exercises the SQL without a real brain.
"""
import json
import sqlite3

from dashboard.queries import encoding as enc


def _conn(rows, path=":memory:"):
    """trace_events with (chain_id, event_type, ref_type, summary, metadata,
    session_id, created_at) rows, inserted in list order (so rowid order ==
    list order — the ordering the bug depended on). In memory unless `path`
    names a file, for the tests that go through the real db resolution."""
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE trace_events (id INTEGER PRIMARY KEY, chain_id TEXT, "
              "scale TEXT, event_type TEXT, ref_type TEXT, ref_id TEXT, "
              "summary TEXT, metadata TEXT, session_id TEXT, created_at TEXT)")
    for i, r in enumerate(rows):
        c.execute("INSERT INTO trace_events (id, chain_id, scale, event_type, "
                  "ref_type, ref_id, summary, metadata, session_id, created_at) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (i, r['chain_id'], r.get('scale', 's1'), r['event_type'],
                   r['ref_type'], r.get('ref_id', ''), r.get('summary', ''),
                   json.dumps(r.get('metadata', {})), r.get('session_id', ''),
                   r.get('created_at', '')))
    c.commit()
    return c


run_chains = enc._query_encoding_chains.__wrapped__   # bypass @safe_query conn-open

CHAIN = 's1e-testsess-7'
TS = '2026-07-03T16:37:00+00:00'


def _base_rows():
    # Order matters: the O/K first, then an edge_relation_revised delta BEFORE
    # the encoding_run summary delta — this is the ordering that made the old
    # unfiltered fetchone() pick the wrong row.
    return [
        {'chain_id': CHAIN, 'event_type': 'O', 'ref_type': 'encoding_prompt',
         'ref_id': '/tmp/none.json', 'summary': '19 turns', 'created_at': TS,
         'session_id': 'testsess'},
        {'chain_id': CHAIN, 'event_type': 'K', 'ref_type': 'node_catalog',
         'summary': '0 unique nodes', 'created_at': TS},
        {'chain_id': CHAIN, 'event_type': 'delta', 'ref_type': 'edge_relation_revised',
         'summary': '3 field(s): description, weight, encoding_source',
         'metadata': {'edge_id': 'e1', 'relation': 'co_anchored'}, 'created_at': TS},
        {'chain_id': CHAIN, 'event_type': 'delta', 'ref_type': 'journal_note',
         'summary': 'a doubt', 'metadata': {'note': 'ships to a corpus with zero cues',
                                            'tag': 'doubt'}, 'created_at': TS},
        {'chain_id': CHAIN, 'event_type': 'delta', 'ref_type': 'encoding_run',
         'summary': '1 actions (1 writes) in 2 rounds',
         'metadata': {'created': ['aaaa1111', 'bbbb2222'], 'revised': ['cccc3333']},
         'created_at': '2026-07-03T16:40:00+00:00'},
    ]


def test_reads_encoding_run_delta_not_first():
    runs = run_chains(_conn(_base_rows()), limit=10, hours=999999)
    assert len(runs) == 1
    r = runs[0]
    # Summary + rollup come from the encoding_run delta, NOT the edge revision.
    assert r['summary'].startswith('1 actions')
    assert r['created_ids'] == ['aaaa1111', 'bbbb2222']
    assert r['revised_ids'] == ['cccc3333']
    # delta_ts is the run's completion, used to bound the edge-window query.
    assert r['delta_ts'] == '2026-07-03T16:40:00+00:00'


def test_collects_journal_notes():
    """Journal notes now come from the ONE journal reader (queries.journals),
    shared by S1E and all four S2 units, so the row carries that reader's full
    shape rather than the two keys this view used to build for itself.

    Asserted field-by-field on the load-bearing values instead of by whole-dict
    equality: the extra fields (`unit`, `subject`, `open_runs`, `since`) are
    what the cross-encoder Journals view and the standing-item escalation rank
    on, and a dict-equality assertion would fail again the next time the shared
    reader learns a field — without anything actually being wrong.
    """
    runs = run_chains(_conn(_base_rows()), limit=10, hours=999999)
    notes = runs[0]['journal_notes']
    assert len(notes) == 1
    n = notes[0]
    assert n['note'] == 'ships to a corpus with zero cues'
    assert n['tag'] == 'doubt'
    # Attribution the shared reader adds: which encoder wrote it, and the
    # lifecycle counters that drive the "needs you" escalation (0 here — a
    # plain tag with no `×N since` suffix).
    assert n['unit'] == 's1e'
    assert n['chain_id'] == 's1e-testsess-7'
    assert n['open_runs'] == 0 and n['since'] == ''


def test_in_progress_run_has_no_summary_delta():
    # A chain whose encoding_run delta hasn't landed yet (still encoding) →
    # graceful fallback, empty rollup, no crash.
    rows = [r for r in _base_rows() if r['ref_type'] != 'encoding_run']
    runs = run_chains(_conn(rows), limit=10, hours=999999)
    r = runs[0]
    assert r['summary'] == '(encoding in progress or no actions)'
    assert r['created_ids'] == [] and r['revised_ids'] == []
    # journal notes still surface even before the summary delta lands.
    assert len(r['journal_notes']) == 1


def test_run_edges_carry_their_ends_by_id(tmp_path, monkeypatch):
    """The graph lights the memories a finished run wove its writes into: its
    own new memories aren't in the graph loaded before they existed, so the
    firing starts from the far ends of the run's edges — which it can only
    find by id."""
    monkeypatch.setenv('BRAIN_DB_DIR', str(tmp_path))
    _conn(_base_rows(), str(tmp_path / 'brain_logs.db')).close()
    b = sqlite3.connect(str(tmp_path / 'brain.db'))
    b.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT, title TEXT, "
              "content TEXT, created_at TEXT, encoding_source TEXT)")
    b.execute("CREATE TABLE edges (edge_id INTEGER, source_id TEXT, "
              "target_id TEXT, weight REAL, created_at TEXT)")
    b.execute("CREATE TABLE edge_relations (edge_id INTEGER, relation TEXT, "
              "archived INTEGER DEFAULT 0, claim_source_id TEXT, claim_target_id TEXT)")
    for nid in ('aaaa1111', 'bbbb2222', 'cccc3333', 'dddd4444'):
        b.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?)",
                  (nid, 'lesson', 'Node ' + nid, 'c', TS, 's1e'))
    # New memory aaaa1111 woven into the older dddd4444, inside the run's span.
    # The pair row stores its ends in the other order: direction is the
    # claim's, so the shipped ends must follow claim_source/claim_target.
    b.execute("INSERT INTO edges VALUES (1,'dddd4444','aaaa1111',0.8,?)",
              ('2026-07-03T16:38:00+00:00',))
    b.execute("INSERT INTO edge_relations VALUES (1,'builds_on',0,'aaaa1111','dddd4444')")
    b.commit()
    b.close()

    runs = enc.query_encoding_runs(limit=10, hours=999999)
    assert len(runs) == 1
    [edge] = runs[0]['edges']
    assert (edge['source_id'], edge['target_id']) == ('aaaa1111', 'dddd4444')
    assert (edge['source_title'], edge['target_title']) == ('Node aaaa1111', 'Node dddd4444')
