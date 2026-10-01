"""Tests for NodeDAL.resolve_live() — the read-only survivor-pointer walk.

Two layers:
  1. Real-data tests against a seeded brain_env() — live
     passthrough, a real archived→live redirect, and an aggregate report of
     how the 340-odd real `_sys_archived_survivor_id` nodes actually resolve.
  2. Deterministic walk-logic tests — A→B→C chain, cycle safety, orphan
     drop-vs-mark, dedup, max_hops. These are built INSIDE the isolated copy
     by cloning a real node row (so the schema is always valid) and flipping
     id/archived, then stamping survivor pointers. They exercise the real
     helper SQL (`SELECT archived ...`, `SELECT value FROM node_metadata_kv`),
     not a mock.

resolve_live is READ-ONLY; the synthetic rows are written directly to the
throwaway isolated DB by the tests, never by the function under test.
"""
import pytest

from tests.isolated_brain import brain_env

SURVIVOR_KEY = '_sys_archived_survivor_id'
PFX = 'zztest-resolvelive-'  # synthetic ids — never collide with real ids


@pytest.fixture(scope='module')
def env():
    with brain_env() as e:
        yield e


@pytest.fixture(scope='module')
def ndal(env):
    return env.brain._nodes


# --- helpers -----------------------------------------------------------------

def _real_survivor_sources(brain):
    """{archived_node_id: survivor_value} for every real node carrying the key,
    excluding any synthetic ids this test file inserts."""
    rows = brain.conn.execute(
        'SELECT node_id, value FROM node_metadata_kv WHERE key = ?',
        (SURVIVOR_KEY,)).fetchall()
    return {nid: val for nid, val in rows if not nid.startswith(PFX)}


def _a_live_node(brain):
    row = brain.conn.execute(
        'SELECT id FROM nodes WHERE archived = 0 LIMIT 1').fetchone()
    return row[0] if row else None


def _clone_node(brain, new_id, archived):
    """Insert a node row with valid schema by cloning a real row's columns,
    overriding only id + archived. Returns new_id."""
    cols = [d[0] for d in brain.conn.execute(
        'SELECT * FROM nodes LIMIT 0').description]
    template = brain.conn.execute(
        'SELECT * FROM nodes WHERE archived = 0 LIMIT 1').fetchone()
    vals = dict(zip(cols, template))
    vals['id'] = new_id
    vals['archived'] = 1 if archived else 0
    ph = ','.join('?' * len(cols))
    brain.conn.execute(
        'INSERT OR REPLACE INTO nodes (%s) VALUES (%s)' % (','.join(cols), ph),
        [vals[c] for c in cols])
    return new_id


def _set_survivor(brain, node_id, survivor_id):
    brain.conn.execute(
        'INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) '
        'VALUES (?, ?, ?)', (node_id, SURVIVOR_KEY, survivor_id))


def _build(brain, spec):
    """spec: {id_suffix: (archived_bool, survivor_suffix_or_None)}.
    Builds the nodes + pointers, returns {suffix: full_id}."""
    ids = {sfx: PFX + sfx for sfx in spec}
    for sfx, (archived, _) in spec.items():
        _clone_node(brain, ids[sfx], archived)
    for sfx, (_, survivor_sfx) in spec.items():
        if survivor_sfx is not None:
            # survivor may be another synthetic node or a literal full id
            target = ids.get(survivor_sfx, survivor_sfx)
            _set_survivor(brain, ids[sfx], target)
    brain.conn.commit()
    return ids


# --- real-data tests ---------------------------------------------------------

def test_live_node_passes_through(env, ndal):
    live = _a_live_node(env.brain)
    assert live is not None, 'isolated brain has no live nodes — copy broken?'
    out = ndal.resolve_live([live])
    assert out['live'] == [live]
    assert out['redirected'] == {}
    assert out['orphans'] == []


def test_real_archived_node_redirects_to_live_survivor(env, ndal):
    """Find a real archived survivor-source that resolves to a live terminal
    and assert the redirect is recorded and the source is not in `live`."""
    sources = _real_survivor_sources(env.brain)
    if not sources:
        pytest.skip('no real _sys_archived_survivor_id nodes in this copy')

    found = None
    for src in sources:
        out = ndal.resolve_live([src])
        if src in out['redirected']:
            found = (src, out)
            break
    if found is None:
        pytest.skip('no real survivor source resolves to a live terminal')

    src, out = found
    terminal = out['redirected'][src]
    assert out['live'] == [terminal]
    assert src not in out['live']
    # terminal really is live
    assert ndal._live_status_bulk([terminal]).get(terminal) == 'live'


def test_real_chain_a_b_c_to_live_terminal(env, ndal):
    """A real two-hop chain from prod (supervisor-verified):
    d44bb207 (archived) → 37cb583d (archived) → 28e92124 (LIVE)."""
    a, b, c = 'd44bb207', '37cb583d', '28e92124'
    present = {nid for (nid,) in env.brain.conn.execute(
        'SELECT id FROM nodes WHERE id IN (?,?,?)', (a, b, c))}
    if {a, b, c} - present:
        pytest.skip('real chain fixture not present in this copy')

    out = ndal.resolve_live([a])
    assert out['live'] == [c]
    assert out['redirected'] == {a: c}      # input → FINAL terminal, not first hop
    assert out['orphans'] == []
    # b is genuinely an intermediate archived hop, not the terminal
    statuses = ndal._live_status_bulk([a, b, c])
    assert statuses.get(a) == 'archived'
    assert statuses.get(b) == 'archived'
    assert statuses.get(c) == 'live'


def test_real_corpus_resolution_report(env, ndal, capsys):
    """Run resolve_live over EVERY real survivor source and report the split.
    Not a strict assertion beyond 'it runs and buckets cleanly' — the numbers
    are the deliverable."""
    sources = list(_real_survivor_sources(env.brain))
    if not sources:
        pytest.skip('no real _sys_archived_survivor_id nodes in this copy')

    out = ndal.resolve_live(sources, on_orphan='mark')
    redirected = set(out['redirected'])
    orphans = set(out['orphans'])
    # The remainder: sources that carry a survivor stamp but are themselves
    # still LIVE (a stale pointer left from a since-reverted archive). They
    # correctly pass through unchanged rather than redirecting.
    passthrough = [s for s in sources if s not in redirected and s not in orphans]

    # Every source falls into exactly one bucket, disjointly.
    assert len(redirected) + len(orphans) + len(passthrough) == len(sources)
    # Passthrough sources really are live.
    passthrough_status = ndal._live_status_bulk(passthrough)
    for s in passthrough:
        assert passthrough_status.get(s) == 'live'

    with capsys.disabled():
        print(
            '\n[resolve_live real-corpus report] '
            'sources=%d redirected=%d orphans=%d live_passthrough=%d '
            'distinct_live_survivors=%d'
            % (len(sources), len(redirected), len(orphans), len(passthrough),
               len(out['live'])))


# --- deterministic walk-logic tests (synthetic rows in the isolated copy) ----

def test_single_redirect(env, ndal):
    ids = _build(env.brain, {
        'sr-arch': (True, 'sr-live'),
        'sr-live': (False, None),
    })
    out = ndal.resolve_live([ids['sr-arch']])
    assert out['live'] == [ids['sr-live']]
    assert out['redirected'] == {ids['sr-arch']: ids['sr-live']}
    assert out['orphans'] == []


def test_chain_a_b_c(env, ndal):
    ids = _build(env.brain, {
        'ch-a': (True, 'ch-b'),
        'ch-b': (True, 'ch-c'),
        'ch-c': (False, None),
    })
    out = ndal.resolve_live([ids['ch-a']])
    assert out['live'] == [ids['ch-c']]
    assert out['redirected'] == {ids['ch-a']: ids['ch-c']}
    assert out['orphans'] == []


def test_cycle_is_safe(env, ndal):
    ids = _build(env.brain, {
        'cy-a': (True, 'cy-b'),
        'cy-b': (True, 'cy-a'),  # loop back
    })
    out = ndal.resolve_live([ids['cy-a']], on_orphan='mark')
    assert out['live'] == []
    assert out['orphans'] == [ids['cy-a']]


def test_orphan_no_pointer_drop_vs_mark(env, ndal):
    ids = _build(env.brain, {
        'orph': (True, None),  # archived, no survivor pointer
    })
    drop = ndal.resolve_live([ids['orph']])  # default on_orphan='drop'
    assert drop['live'] == []
    assert drop['orphans'] == []

    mark = ndal.resolve_live([ids['orph']], on_orphan='mark')
    assert mark['live'] == []
    assert mark['orphans'] == [ids['orph']]


def test_missing_node_is_orphan(ndal):
    out = ndal.resolve_live([PFX + 'never-inserted'], on_orphan='mark')
    assert out['live'] == []
    assert out['orphans'] == [PFX + 'never-inserted']


def test_dedup_many_inputs_one_survivor(env, ndal):
    ids = _build(env.brain, {
        'dd-1': (True, 'dd-live'),
        'dd-2': (True, 'dd-live'),
        'dd-live': (False, None),
    })
    out = ndal.resolve_live([ids['dd-1'], ids['dd-2']])
    assert out['live'] == [ids['dd-live']]  # deduped to one
    assert out['redirected'] == {
        ids['dd-1']: ids['dd-live'],
        ids['dd-2']: ids['dd-live'],
    }


def test_first_seen_order_preserved(env, ndal):
    ids = _build(env.brain, {
        'ord-a': (False, None),
        'ord-b-arch': (True, 'ord-c'),
        'ord-c': (False, None),
    })
    # input order: c-via-redirect first, then a — output should reflect
    # first-seen of each distinct survivor in input order.
    out = ndal.resolve_live([ids['ord-b-arch'], ids['ord-a']])
    assert out['live'] == [ids['ord-c'], ids['ord-a']]


def test_max_hops_exhausted_is_orphan(env, ndal):
    ids = _build(env.brain, {
        'mh-a': (True, 'mh-b'),
        'mh-b': (True, 'mh-c'),
        'mh-c': (False, None),
    })
    # 2 redirects needed (a→b→c); max_hops=1 can't reach the live terminal.
    out = ndal.resolve_live([ids['mh-a']], on_orphan='mark', max_hops=1)
    assert out['live'] == []
    assert out['orphans'] == [ids['mh-a']]
    # with enough budget it resolves
    ok = ndal.resolve_live([ids['mh-a']], max_hops=2)
    assert ok['live'] == [ids['mh-c']]


def test_empty_input(ndal):
    out = ndal.resolve_live([])
    assert out == {'live': [], 'redirected': {}, 'orphans': []}
    assert ndal.resolve_live(None)['live'] == []
