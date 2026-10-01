"""Canonical-pull redirect — absorbed ids resolve at the door (ruling d42a49ce).

The contract under test, end to end:
  - READS: get_node / get_nodes / filter_nodes(field='id') follow an absorbed
    id to its live survivor, keyed by the REQUESTED id, carrying
    `_redirected_from` so every renderer marks the redirect. A RETIRED node
    (archived, no survivor) returns its own record, honestly archived.
    `follow_absorbed=False` is the audit hatch.
  - WRITES never redirect: revise / connect_typed / absorb refuse an absorbed
    id loudly WITH the pointer.
  - RENDER: render_rich_node shows `⚠ ARCHIVED` on a corpse and the
    `src ↦ survivor` line on a redirect; thalamus ref lines mark redirects.

Synthetic rows are built inside a seeded brain_env() (the
test_resolve_live.py pattern — real schema, real helper SQL, throwaway DB).
"""
import pytest

from tests.isolated_brain import brain_env
from tests.test_resolve_live import _clone_node, _set_survivor

SURVIVOR_KEY = '_sys_archived_survivor_id'
PFX = 'zztest-canonpull-'


@pytest.fixture(scope='module')
def env():
    with brain_env() as e:
        yield e


@pytest.fixture(scope='module')
def brain(env):
    return env.brain


def _build(brain, spec):
    """spec: {suffix: (archived, survivor_suffix_or_None, title_or_None)}.
    Returns {suffix: full_id}."""
    ids = {sfx: PFX + sfx for sfx in spec}
    for sfx, (archived, _, title) in spec.items():
        _clone_node(brain, ids[sfx], archived)
        if title:
            brain.conn.execute('UPDATE nodes SET title = ? WHERE id = ?',
                               (title, ids[sfx]))
    for sfx, (_, survivor_sfx, _) in spec.items():
        if survivor_sfx is not None:
            _set_survivor(brain, ids[sfx], ids.get(survivor_sfx, survivor_sfx))
    brain.conn.commit()
    return ids


# --- reads: get_node ---------------------------------------------------------

def test_get_node_single_absorbed_redirects(brain):
    ids = _build(brain, {
        'g1-arch': (True, 'g1-live', None),
        'g1-live': (False, None, 'the survivor'),
    })
    node = brain.get_node(ids['g1-arch'])
    assert node is not None
    assert node['id'] == ids['g1-live']
    assert node['_redirected_from'] == [ids['g1-arch']]
    assert node['title'] == 'the survivor'


def test_get_node_chain_redirects_to_terminal(brain):
    ids = _build(brain, {
        'g2-a': (True, 'g2-b', None),
        'g2-b': (True, 'g2-c', None),
        'g2-c': (False, None, None),
    })
    node = brain.get_node(ids['g2-a'])
    assert node['id'] == ids['g2-c']
    assert node['_redirected_from'] == [ids['g2-a']]


def test_get_node_retired_returns_corpse(brain):
    ids = _build(brain, {'g3-retired': (True, None, None)})
    node = brain.get_node(ids['g3-retired'])
    assert node is not None
    assert node['id'] == ids['g3-retired']
    assert node['archived'] is True
    assert '_redirected_from' not in node


def test_get_node_batch_keys_by_requested_id(brain):
    ids = _build(brain, {
        'g4-live': (False, None, None),
        'g4-arch': (True, 'g4-surv', None),
        'g4-surv': (False, None, None),
        'g4-retired': (True, None, None),
    })
    missing = PFX + 'g4-never-inserted'
    out = brain.get_node([ids['g4-live'], ids['g4-arch'],
                          ids['g4-retired'], missing])
    assert set(out) == {ids['g4-live'], ids['g4-arch'], ids['g4-retired']}
    assert out[ids['g4-live']]['id'] == ids['g4-live']
    # redirected request keys the SURVIVOR's dict
    assert out[ids['g4-arch']]['id'] == ids['g4-surv']
    assert out[ids['g4-arch']]['_redirected_from'] == [ids['g4-arch']]
    # retired keeps its corpse, honestly archived
    assert out[ids['g4-retired']]['archived'] is True


def test_get_node_two_absorbed_one_survivor(brain):
    """Copy-on-stamp: each redirected request gets its OWN entry carrying
    only its OWN requested id — never one shared dict under two keys."""
    ids = _build(brain, {
        'g5-a1': (True, 'g5-surv', None),
        'g5-a2': (True, 'g5-surv', None),
        'g5-surv': (False, None, None),
    })
    out = brain.get_node([ids['g5-a1'], ids['g5-a2']])
    assert out[ids['g5-a1']] is not out[ids['g5-a2']]
    assert out[ids['g5-a1']]['id'] == ids['g5-surv']
    assert out[ids['g5-a1']]['_redirected_from'] == [ids['g5-a1']]
    assert out[ids['g5-a2']]['_redirected_from'] == [ids['g5-a2']]


def test_directly_requested_survivor_carries_no_banner(brain):
    """A survivor the caller also asked for by its own live id must NOT
    carry another request's redirect marker (the false-banner bug)."""
    ids = _build(brain, {
        'g7-arch': (True, 'g7-surv', None),
        'g7-surv': (False, None, None),
    })
    out = brain.get_node([ids['g7-arch'], ids['g7-surv']])
    assert '_redirected_from' not in out[ids['g7-surv']]
    assert out[ids['g7-arch']]['_redirected_from'] == [ids['g7-arch']]


def test_get_node_follow_absorbed_false_returns_corpse(brain):
    ids = _build(brain, {
        'g6-arch': (True, 'g6-live', None),
        'g6-live': (False, None, None),
    })
    node = brain.get_node(ids['g6-arch'], follow_absorbed=False)
    assert node['id'] == ids['g6-arch']
    assert node['archived'] is True
    assert '_redirected_from' not in node


# --- reads: filter_nodes(field='id') -----------------------------------------

def test_filter_by_id_redirects_and_stamps_skinny(brain):
    ids = _build(brain, {
        'f1-arch': (True, 'f1-surv', None),
        'f1-surv': (False, None, 'stamped survivor'),
        'f1-live': (False, None, None),
    })
    res = brain.filter_nodes(field='id',
                             include=[ids['f1-arch'], ids['f1-live']],
                             rich=False, limit=10)
    by_id = {n['id']: n for n in res['nodes']}
    assert ids['f1-surv'] in by_id
    assert by_id[ids['f1-surv']]['_redirected_from'] == [ids['f1-arch']]
    assert ids['f1-live'] in by_id
    assert '_redirected_from' not in by_id[ids['f1-live']]


def test_filter_by_id_redirects_and_stamps_rich(brain):
    ids = _build(brain, {
        'f2-arch': (True, 'f2-surv', None),
        'f2-surv': (False, None, None),
    })
    res = brain.filter_nodes(field='id', include=[ids['f2-arch']],
                             rich=True, limit=10)
    assert len(res['nodes']) == 1
    node = res['nodes'][0]
    assert node['id'] == ids['f2-surv']
    assert node['_redirected_from'] == [ids['f2-arch']]


def test_filter_by_id_retired_still_drops(brain):
    ids = _build(brain, {'f3-retired': (True, None, None)})
    res = brain.filter_nodes(field='id', include=[ids['f3-retired']],
                             rich=False, limit=10)
    assert res.get('nodes') in ([], None) or not res.get('nodes')


# --- writes refuse with the pointer ------------------------------------------

def test_revise_absorbed_refuses_with_pointer(brain):
    ids = _build(brain, {
        'w1-arch': (True, 'w1-surv', None),
        'w1-surv': (False, None, None),
    })
    out = brain.revise(ids['w1-arch'], reason='test',
                       updates={'situation': 'x'})
    assert 'error' in out
    assert ids['w1-surv'][:8] in out['error']
    assert out.get('survivor_id') == ids['w1-surv']


def test_revise_retired_refuses_generic(brain):
    ids = _build(brain, {'w2-retired': (True, None, None)})
    out = brain.revise(ids['w2-retired'], reason='test',
                       updates={'situation': 'x'})
    assert out == {'error': 'Cannot revise archived node',
                   'node_id': ids['w2-retired']}


def test_connect_absorbed_endpoint_raises_with_pointer(brain):
    ids = _build(brain, {
        'w3-arch': (True, 'w3-surv', None),
        'w3-surv': (False, None, None),
        'w3-live': (False, None, None),
    })
    with pytest.raises(ValueError, match='absorbed into %s' %
                                         ids['w3-surv'][:8]):
        brain.connect_typed(ids['w3-live'], ids['w3-arch'],
                            relation='test_rel')
    with pytest.raises(ValueError, match='absorbed into'):
        brain.connect_typed(ids['w3-arch'], ids['w3-live'],
                            relation='test_rel')


def test_absorb_archived_survivor_error_carries_pointer(brain):
    ids = _build(brain, {
        'w4-arch': (True, 'w4-term', None),
        'w4-term': (False, None, None),
        'w4-other': (False, None, None),
    })
    out = brain.absorb(survivor_id=ids['w4-arch'],
                       absorbed_id=ids['w4-other'], reason='test')
    assert out['ok'] is False
    assert ids['w4-term'][:8] in out['error']


# --- render -------------------------------------------------------------------

def test_render_marks_redirect_and_archived(brain):
    from servers.contract import render_rich_node
    ids = _build(brain, {
        'r1-arch': (True, 'r1-surv', None),
        'r1-surv': (False, None, None),
    })
    redirected = brain.get_node(ids['r1-arch'])
    text = render_rich_node(redirected)
    assert '%s ↦ %s' % (ids['r1-arch'][:8], ids['r1-surv'][:8]) in text
    assert 'absorbed into this node' in text

    corpse = brain.get_node(ids['r1-arch'], follow_absorbed=False)
    text = render_rich_node(corpse)
    assert '⚠ ARCHIVED' in text
    assert '↦' not in text


def test_thalamus_ref_lines_mark_redirect(brain):
    from servers.channels.thalamus.thalamus import _attach_ref_lines
    ids = _build(brain, {
        't1-arch': (True, 't1-surv', None),
        't1-surv': (False, None, 'ref survivor title'),
        't1-live': (False, None, 'live ref title'),
    })
    items = [{'refs': [ids['t1-arch'], ids['t1-live']]}]
    _attach_ref_lines(brain, items, session_id='')
    lines = items[0]['ref_lines']
    assert '%s ↦ %s · ref survivor title (absorbed)' % (
        ids['t1-arch'][:8], ids['t1-surv'][:8]) in lines
    assert '%s · live ref title' % ids['t1-live'][:8] in lines


def test_redirect_door_parity(brain):
    """Every id-keyed read door reports the SAME redirect shape — survivor
    identity under the requested handle, REDIRECTED_FROM_KEY naming the
    requested id. The doors diverging silently is the class this test
    exists to catch (the review found three hand-rolled copies)."""
    from servers.contract import REDIRECTED_FROM_KEY
    ids = _build(brain, {
        'p1-arch': (True, 'p1-surv', 'parity survivor'),
        'p1-surv': (False, None, 'parity survivor'),
    })
    arch, surv = ids['p1-arch'], ids['p1-surv']

    via_get = brain.get_node(arch)
    via_filter = brain.filter_nodes(field='id', include=[arch],
                                    rich=False, limit=5)['nodes']
    via_recall = brain.recall_node(arch)['results']

    assert via_get['id'] == surv
    assert via_get[REDIRECTED_FROM_KEY] == [arch]
    assert len(via_filter) == 1 and via_filter[0]['id'] == surv
    assert via_filter[0][REDIRECTED_FROM_KEY] == [arch]
    assert len(via_recall) == 1 and via_recall[0]['id'] == surv
    assert via_recall[0][REDIRECTED_FROM_KEY] == [arch]


def test_canonicalize_swaps_identity_for_absorbed_row(brain):
    """The ONE door: a result row holding an absorbed id becomes the
    survivor wholesale — never the corpse wearing the survivor's
    attachments (the chimera)."""
    ids = _build(brain, {
        'c1-arch': (True, 'c1-surv', 'corpse title'),
        'c1-surv': (False, None, 'door survivor'),
    })
    row = {'id': ids['c1-arch'], 'title': 'corpse title',
           '_source': 'test_seed'}
    brain.canonicalize_results([row])
    assert row['id'] == ids['c1-surv']
    assert row['title'] == 'door survivor'
    assert row['_redirected_from'] == [ids['c1-arch']]
    assert row['_source'] == 'test_seed'  # scoring fields survive the swap


def test_write_doors_share_the_refusal_contract(brain):
    """revise / set_node_lock refuse an absorbed id with the shared message
    AND the structured survivor_id (connect raises the same message)."""
    ids = _build(brain, {
        'w5-arch': (True, 'w5-surv', None),
        'w5-surv': (False, None, None),
    })
    r = brain.revise(ids['w5-arch'], reason='t', updates={'situation': 'x'})
    lock = brain.set_node_lock(ids['w5-arch'], locked=True)
    for out in (r, lock):
        assert 'absorbed into %s' % ids['w5-surv'][:8] in out['error']
        assert out.get('survivor_id') == ids['w5-surv']


def test_recall_node_follows_redirect_coherently(brain):
    """The by-id recall door must not chimera: corpse content with survivor
    attachments (canonicalize_results pulls through the redirecting
    get_node, so the row itself has to be the survivor too)."""
    ids = _build(brain, {
        'rn-arch': (True, 'rn-surv', None),
        'rn-surv': (False, None, 'recall survivor'),
    })
    out = brain.recall_node(ids['rn-arch'])
    assert out['results'], 'redirect lost the node entirely'
    r = out['results'][0]
    assert r['id'] == ids['rn-surv']
    assert r['title'] == 'recall survivor'
    assert r['_redirected_from'] == [ids['rn-arch']]

    ids2 = _build(brain, {'rn-retired': (True, None, None)})
    out2 = brain.recall_node(ids2['rn-retired'])
    assert out2['results'][0]['id'] == ids2['rn-retired']
    assert out2['results'][0]['archived'] is True


def test_edge_endpoints_emit_live_candidates(brain):
    """Regression for the 2026-08-30 dedup bug: LIVE edge endpoints must
    emit edge_source/edge_target candidates (the two-id-space seen-set
    dropped every live endpoint); absorbed endpoints emit their survivor;
    retired endpoints drop."""
    from servers.scales.s1.fetch_tools import _fetch_edges_with_endpoints
    ids = _build(brain, {
        'e1-src': (False, None, None),
        'e1-tgt': (False, None, None),
        'e1-arch': (True, 'e1-surv', None),
        'e1-surv': (False, None, None),
        'e1-retired': (True, None, None),
    })
    live_edge = brain._graph.add_relation(
        ids['e1-src'], ids['e1-tgt'], 'test_endpoint_rel')['edge_id']
    mixed_edge = brain._graph.add_relation(
        ids['e1-arch'], ids['e1-retired'], 'test_endpoint_rel')['edge_id']

    out = _fetch_edges_with_endpoints(brain, [live_edge])
    by_tier = {}
    for c in out:
        by_tier.setdefault(c.get('tier'), []).append(c['id'])
    assert by_tier.get('edge_source') == [ids['e1-src']]
    assert by_tier.get('edge_target') == [ids['e1-tgt']]
    # Provenance: EVERY candidate this tool builds — the synthetic edge row
    # included, which is built inline rather than through _to_candidate —
    # must carry `discovery`, the one field the surface renderer reads to
    # emit source_tool="true". A candidate without it tells Haiku a
    # tool-fetched node came from the brain's own retrieval.
    assert {c.get('discovery') for c in out} == {'recall_by_time'}, \
        'a fetch-tool candidate reached the renderer with no provenance: %r' \
        % sorted({(c.get('tier'), c.get('discovery')) for c in out})

    out = _fetch_edges_with_endpoints(brain, [mixed_edge])
    endpoint_ids = {c['id'] for c in out
                    if c.get('tier') in ('edge_source', 'edge_target')}
    assert endpoint_ids == {ids['e1-surv']}, (
        'absorbed endpoint must emit its survivor; retired must drop')


# --- dispatch keeps request order through redirects ---------------------------

def test_dispatch_get_nodes_preserves_order_through_redirect(brain):
    from servers.dispatch_read import _handle_get_nodes
    ids = _build(brain, {
        'd1-live': (False, None, None),
        'd1-arch': (True, 'd1-surv', None),
        'd1-surv': (False, None, None),
    })
    out = _handle_get_nodes(
        brain, {'node_ids': [ids['d1-live'], ids['d1-arch']]}, [])
    assert out['ok'] is True
    results = out['result']
    assert results[0]['id'] == ids['d1-live']
    assert results[1]['id'] == ids['d1-surv']
    assert results[1]['_redirected_from'] == [ids['d1-arch']]
