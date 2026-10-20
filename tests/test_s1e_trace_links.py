"""Piece 2 of the S1E code-half rebuild — the trace↔node link capability.

Verifies servers/scales/s1/trace_links.py: the PURE composer joins S1R surface
selections and S1E encode runs to target traces by stop_counter (the structural
chain-suffix key, never timestamps). Driven entirely by a raw synthetic trace
dataset — plain dicts, no brain, no DB — which is exactly the robustness the
architecture is for (eval replay / sequential historical data drive it the same).

See docs/S1-SCRIBE-REDESIGN.md §10.3.2.
"""
import os
import sys

import pytest

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s1.trace_links import (  # noqa: E402
    nodes_for_traces, gather, session_node_ids, _stop_of, _surface_ids,
    _candidate_outcomes, _tool_provenance,
)

SHORT = 'abcd1234'  # session_short in every chain id


def _turn(stop, tid=None):
    return {'id': tid or ('u%d' % stop), 'chain_id': 's0-%s-%d' % (SHORT, stop)}


def _surface(stop, ids, tid=None, tool_trace=None):
    import json
    rec = {'id': tid or ('sr%d' % stop), 'chain_id': 's1r-%s-%d' % (SHORT, stop),
           'ref_id': json.dumps(ids)}
    if tool_trace is not None:
        rec['metadata'] = {'tool_trace': tool_trace}
    return rec


def _tool_call(tool, result_ids=None, dropped_ids=None):
    return {'tool': tool, 'args': {}, 'result_count': len(result_ids or []),
            'result_ids': result_ids or [], 'dropped_ids': dropped_ids or [],
            'latency_ms': 5, 'error': None}


def _encode(stop, created, revised, tid=None):
    return {'id': tid or ('e%d' % stop), 'chain_id': 's1e-%s-%d' % (SHORT, stop),
            'metadata': {'created': created, 'revised': revised}}


# ── the stop parser (the join key) ──

def test_stop_of_parses_trailing_counter():
    assert _stop_of('s0-abcd1234-5') == 5
    assert _stop_of('s1r-abcd1234-12') == 12
    assert _stop_of('s1e-abcd1234-0') == 0
    assert _stop_of('') is None
    assert _stop_of(None) is None
    assert _stop_of('s0-abcd1234-notanint') is None


def test_stop_of_round_trips_session_context_chains():
    # Pin the build↔parse coupling: the stop SessionContext stamps into a chain id
    # must be the stop _stop_of recovers. Guards a silent join break if the
    # chain-id format ever changes on one side only.
    from servers.session_context import SessionContext
    ctx = SessionContext(session_id='abcd1234deadbeef', stop_counter=7)
    assert _stop_of(ctx.s0_chain()) == 7
    assert _stop_of(ctx.s1r_chain()) == 7
    assert _stop_of(ctx.s1e_chain()) == 7


def test_surface_ids_resilient():
    assert _surface_ids('["a","b"]') == ['a', 'b']
    assert _surface_ids('') == []
    assert _surface_ids('not json') == []      # malformed → [], no raise
    assert _surface_ids('{"a":1}') == []        # not a list → []


# ── surfaced: 1:1 with the turn, same stop ──

def test_surfaced_joins_by_stop():
    links = nodes_for_traces(
        surface_traces=[_surface(5, ['nodeA', 'nodeB']),
                        _surface(6, ['nodeC'])],
        encode_traces=[],
        target_traces=[_turn(5), _turn(6)])
    assert links['u5']['surfaced'] == ['nodeA', 'nodeB']
    assert links['u6']['surfaced'] == ['nodeC']
    # no encode runs → nothing encoded, no owning run
    assert links['u5']['encoded'] == [] and links['u5']['encoded_by'] is None


# ── encoded: per-run, the owning run closes the turn-range at/after the turn ──

def test_encoded_owning_run_is_first_run_strictly_after_turn():
    # STRICT convention: turn chains stamp PRE-increment, run chains POST- —
    # a run at stop S saw exactly the turns with chain-stop < S, never == S.
    # runs at stop 5 and 10: turn 3 → run@5; turns 5 and 7 → run@10.
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(5, ['c5'], ['r5'], tid='run5'),
                       _encode(10, ['c10'], [], tid='run10')],
        target_traces=[_turn(3), _turn(5), _turn(7)])
    assert links['u3']['encoded_by'] == 'run5'
    assert links['u5']['encoded_by'] == 'run10'   # run@5 did NOT see chain-stop 5
    assert links['u7']['encoded_by'] == 'run10'
    # the run's COMPLETE node-set (created + revised), deduped & order-preserved
    assert links['u3']['encoded'] == ['c5', 'r5']
    assert links['u7']['encoded'] == ['c10']


def test_unencoded_tail_has_no_owning_run():
    # a turn past the last run = the emergent unencoded boundary.
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(5, ['c5'], [])],
        target_traces=[_turn(8)])
    assert links['u8']['encoded'] == []
    assert links['u8']['encoded_by'] is None


def test_created_and_revised_merged_and_deduped():
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(6, ['x', 'y'], ['y', 'z'])],  # y in both
        target_traces=[_turn(5)])
    assert links['u5']['encoded'] == ['x', 'y', 'z']


def test_full_link_shape_surfaced_and_encoded_together():
    links = nodes_for_traces(
        surface_traces=[_surface(5, ['surfA', 'surfB'])],
        encode_traces=[_encode(6, ['encC'], ['encD'], tid='run6')],
        target_traces=[_turn(5)])
    assert links['u5'] == {
        'surfaced': ['surfA', 'surfB'],
        'encoded': ['encC', 'encD'],
        'encoded_by': 'run6', 'encoded_by_stop': 6,
        'authored': [], 'created': [], 'revised': [], 'archived': [],
        'recalled': [], 'looked_up': [], 'endo': [],  # no touched stream here
        'dropped': [],                               # no recall stream here
        'fetched_by': {}, 'floored_by': {},          # no tool_trace here
    }


# ── resilience: a raw dataset with odd rows must not crash the join ──

def test_malformed_rows_are_survivable():
    links = nodes_for_traces(
        surface_traces=[{'id': 'bad', 'chain_id': 's1r-x-5', 'ref_id': 'garbage'},
                        {'id': 'nochain'},  # no chain_id → skipped
                        _surface(5, ['ok'])],
        encode_traces=[{'id': 'badrun'}],   # no chain_id → skipped
        target_traces=[_turn(5), {'id': 'orphan'}])  # orphan has no chain_id
    assert links['u5']['surfaced'] == ['ok']           # the good row survived
    assert links['orphan'] == {'surfaced': [], 'encoded': [], 'encoded_by': None,
                               'encoded_by_stop': None,
                               'authored': [], 'created': [], 'revised': [],
                               'archived': [], 'recalled': [], 'looked_up': [],
                               'endo': [], 'dropped': [],
                               'fetched_by': {}, 'floored_by': {}}


def test_runs_out_of_order_still_pick_earliest_owning_run():
    # encode_traces fed newest-first; the composer sorts by stop internally.
    links = nodes_for_traces(
        surface_traces=[],
        encode_traces=[_encode(10, ['c10'], [], tid='run10'),
                       _encode(5, ['c5'], [], tid='run5')],
        target_traces=[_turn(6)])
    assert links['u6']['encoded_by'] == 'run10'   # first run with stop >= 6


# ── the live adapter pulls both streams via the query_traces door ──

def _touched(stop, created=None, revised=None, recalled=None, endo=None, tid=None):
    return {'id': tid or ('at%d' % stop), 'chain_id': 's0-%s-%d' % (SHORT, stop),
            'metadata': {'created': created or [], 'revised': revised or [],
                         'archived': [], 'recalled': recalled or [], 'endo': endo or []}}


class _StubBrain:
    def __init__(self):
        self.calls = []

    def query_traces(self, **kw):
        self.calls.append(kw)
        if kw.get('ref_type') == 'surface_selected':
            return {'events': [_surface(5, ['s'])]}
        if kw.get('ref_type') == 'encoding_run':
            return {'events': [_encode(6, ['e'], [])]}   # 6 > 5: owns turn@5 (strict)
        if kw.get('ref_type') == 'anchor_touched':
            return {'events': [_touched(5, created=['w'], recalled=['r'])]}
        return {'events': []}


def test_gather_pulls_all_three_streams_via_door():
    brain = _StubBrain()
    st = gather(brain, 'sess-xyz')          # default streams: surface/encode/touched
    surf, enc, touched = st['surface'], st['encode'], st['touched']
    assert len(surf) == 1 and len(enc) == 1 and len(touched) == 1
    # all three pulls scoped to the session, window disabled (hours=None);
    # surface/encode at s1, anchor_touched at s0.
    by_rt = {c['ref_type']: c for c in brain.calls}
    assert set(by_rt) == {'surface_selected', 'encoding_run', 'anchor_touched'}
    assert by_rt['surface_selected']['scale'] == 's1'
    assert by_rt['anchor_touched']['scale'] == 's0'
    for c in brain.calls:
        assert c['session_id'] == 'sess-xyz' and c['hours'] is None
    # end-to-end through the pure core
    links = nodes_for_traces(surf, enc, [_turn(5)], touched_traces=touched)
    assert links['u5']['surfaced'] == ['s'] and links['u5']['encoded'] == ['e']
    assert links['u5']['authored'] == ['w'] and links['u5']['recalled'] == ['r']


# ── the touched feed: anchor_touched joins 1:1 with the turn by stop ──

def test_touched_authored_and_recalled_join_by_stop():
    links = nodes_for_traces(
        surface_traces=[], encode_traces=[],
        target_traces=[_turn(5), _turn(6)],
        touched_traces=[_touched(5, created=['nA'], revised=['nB'], recalled=['nC']),
                        _touched(6, created=['nD'])])
    assert links['u5']['authored'] == ['nA', 'nB']   # created ∪ revised
    assert links['u5']['recalled'] == ['nC']
    assert links['u6']['authored'] == ['nD'] and links['u6']['recalled'] == []


def test_touched_archived_not_in_authored():
    # archived is recorded in the delta but a gone node must not surface as authored.
    t = {'id': 'at5', 'chain_id': 's0-%s-5' % SHORT,
         'metadata': {'created': ['live'], 'revised': [], 'archived': ['dead'],
                      'recalled': [], 'endo': []}}
    links = nodes_for_traces([], [], [_turn(5)], touched_traces=[t])
    assert links['u5']['authored'] == ['live']
    assert 'dead' not in links['u5']['authored']
    # ...but it IS carried on its own key (the provenance verb split renders it)
    assert links['u5']['archived'] == ['dead']


def test_touched_verb_split_keys_join_by_stop():
    # created/revised/archived ride the link SPLIT per verb alongside the merged
    # authored — the encoder's <provenance> renders the verbs (view policy).
    links = nodes_for_traces(
        surface_traces=[], encode_traces=[],
        target_traces=[_turn(5)],
        touched_traces=[_touched(5, created=['nA'], revised=['nB', 'nA'])])
    assert links['u5']['created'] == ['nA']
    assert links['u5']['revised'] == ['nB', 'nA']
    assert links['u5']['authored'] == ['nA', 'nB']   # merged view unchanged


def test_touched_looked_up_joins_but_never_reaches_catalog():
    # `looked_up` (search-tool results) joins per turn for the provenance line,
    # but session_node_ids deliberately ignores it — folding recall result
    # pages into the catalog would flood what the aging work just cut.
    t = {'id': 'at5', 'chain_id': 's0-%s-5' % SHORT,
         'metadata': {'created': [], 'revised': [], 'archived': [],
                      'recalled': ['byid1'], 'looked_up': ['srch1', 'srch2'],
                      'endo': []}}
    links = nodes_for_traces([], [], [_turn(5)], touched_traces=[t])
    assert links['u5']['looked_up'] == ['srch1', 'srch2']
    assert links['u5']['recalled'] == ['byid1']
    ids = session_node_ids([], [t])
    assert ids['recalled'] == {'byid1'}
    assert 'srch1' not in ids['recalled']
    assert 'looked_up' not in ids                     # never a catalog category


def test_lookup_result_id_extractor_per_shape():
    # The daemon accumulator's per-tool result parsing (S0 capture side of the
    # looked_up stream) — one branch per dispatch result shape, all defensive.
    from servers.daemon_server import BrainDaemon
    x = BrainDaemon._lookup_result_ids
    node = {'id': 'aaaa1111', 'title': 't'}
    assert x('recall', {'results': [node, {'no_id': 1}]}) == ['aaaa1111']
    assert x('recall_batch', [{'query': 'q', 'results': [node]},
                              {'query': 'q2', 'results': None}]) == ['aaaa1111']
    assert x('filter_nodes', {'nodes': [node]}) == ['aaaa1111']
    assert x('find_node_by_title', node) == ['aaaa1111']       # top_k=1: dict
    assert x('find_node_by_title', [node, node]) == ['aaaa1111', 'aaaa1111']
    assert x('get_nodes', [node, {'id': 'bad', 'error': 'not found'}]) == ['aaaa1111']
    # malformed payloads yield nothing, never raise
    for cmd in ('recall', 'recall_batch', 'filter_nodes', 'get_node'):
        assert x(cmd, None) == []
        assert x(cmd, 'garbage') == []


def test_touched_absent_yields_empty_relations():
    # No touched stream → authored/recalled/endo are empty, never missing.
    links = nodes_for_traces([_surface(5, ['s'])], [], [_turn(5)])
    assert links['u5']['authored'] == [] and links['u5']['endo'] == []


def test_delta_ids_shared_parser_symmetry():
    # The SAME parser reads the S1 encode delta and the S0 touched delta — both
    # carry created/revised. Proven by feeding one meta dict through both lenses.
    from servers.scales.s1.trace_links import _delta_ids
    meta = {'created': ['a', 'b'], 'revised': ['b', 'c'], 'recalled': ['d']}
    assert _delta_ids(meta, 'created', 'revised') == ['a', 'b', 'c']  # encode lens
    assert _delta_ids(meta, 'recalled') == ['d']                       # touched-only lens
    assert _delta_ids(None, 'created') == []


def test_session_node_ids_unions_encode_and_touched():
    # The catalog's session-level view: union across all encode + touched traces,
    # no turn keying. encoded = created∪revised over runs; authored/recalled over
    # touched. `surfaced` is empty when no surface stream is passed.
    ids = session_node_ids(
        encode_traces=[_encode(5, ['e1'], ['e2']), _encode(10, ['e3'], [])],
        touched_traces=[_touched(5, created=['a1'], revised=['a2'], recalled=['r1']),
                        _touched(6, created=['a3'], recalled=['r1'])])  # r1 dup
    assert ids['encoded'] == {'e1', 'e2', 'e3'}
    assert ids['authored'] == {'a1', 'a2', 'a3'}
    assert ids['recalled'] == {'r1'}                  # deduped across turns
    assert ids['surfaced'] == set()                   # no surface stream passed
    assert 'endo' not in ids


def test_session_node_ids_preserves_recency():
    # Catalog aging needs to know WHICH stop each id was last touched at and
    # where the encode rounds sit — `stops` (newest wins) and `run_stops`.
    # ONE coordinate system, 1-based: run stops are post-increment (already
    # 1-based); touched stops are pre-increment turn chains, normalized +1.
    ids = session_node_ids(
        encode_traces=[_encode(5, ['e1'], ['e2']), _encode(10, ['e3'], ['e1'])],
        touched_traces=[_touched(3, created=['a1'], recalled=['r1']),
                        _touched(12, recalled=['r1'])])   # r1 re-looked-up later
    assert ids['run_stops'] == [5, 10]                    # ascending, as stamped
    assert ids['stops']['e1'] == 10                       # revised later → newest
    assert ids['stops']['e2'] == 5
    assert ids['stops']['a1'] == 4                        # chain 3 → turn 4
    assert ids['stops']['r1'] == 13                       # newest touch wins, +1


def test_session_node_ids_empty_streams():
    ids = session_node_ids([], [])
    assert ids == {'encoded': set(), 'authored': set(), 'recalled': set(),
                   'surfaced': set(), 'stops': {}, 'run_stops': []}


def test_session_node_ids_takes_surfaced_from_the_surface_trace():
    """The catalog's surfaced ids come from the S1R `surface_selected` picks —
    the same structured trace the timeline's `surfaced:` provenance reads.

    Regression guard for the 37% gap (2026-09-16): the catalog used to derive
    them ONLY by regex-scraping the RENDERED inject text, so every id the
    inject render cut or never wrote reached the timeline's provenance with no
    catalog body. 180 of 483 window-surfaced ids across 26 captured payloads.
    One fact, one source.
    """
    ids = session_node_ids(
        encode_traces=[_encode(5, ['e1'], [])],
        touched_traces=[],
        surface_traces=[_surface(7, ['s1', 's2']),
                        _surface(9, ['s2', 's3'])])      # s2 recurs
    assert ids['surfaced'] == {'s1', 's2', 's3'}          # deduped across turns
    # and they carry recency, or aging would sort every surfaced node as
    # unknown-stop and trim the ones the window just lit
    assert ids['stops']['s1'] == 8                        # chain 7 → turn 8
    assert ids['stops']['s2'] == 10                       # newest pick wins
    assert ids['encoded'] == {'e1'}                       # other streams intact


def test_session_node_ids_surface_trace_malformed_ref_is_quiet():
    # One bad row never sinks the join — the live writer validates its own side.
    ids = session_node_ids([], [], surface_traces=[
        {'chain_id': 's1r-abcd1234-7', 'ref_id': 'not json'},
        _surface(8, ['ok1'])])
    assert ids['surfaced'] == {'ok1'}


def test_anchor_touched_metadata_builder_shape_and_dedup():
    # The contract builder: all five keys present, order-preserving dedup, and it
    # mirrors the encode delta's created/revised/archived key names (the symmetry).
    from servers.trace_contract import (build_anchor_touched_metadata,
                                        ANCHOR_TOUCHED_KEYS, DELTA_METADATA_SHAPE)
    m = build_anchor_touched_metadata(created=['a', 'a', 'b'], recalled=['c'])
    assert set(m) == set(ANCHOR_TOUCHED_KEYS)
    assert m['created'] == ['a', 'b'] and m['recalled'] == ['c']
    assert m['revised'] == [] and m['archived'] == [] and m['endo'] == []
    # the shared keys are exactly the encode delta's id-list field names
    assert {'created', 'revised', 'archived'} <= set(DELTA_METADATA_SHAPE)


# ── the dropped role: recall_traces + _candidate_outcomes (episodic view) ──

def _outcomes(stop, outcomes, tid=None):
    """A Δ additionalContext candidate-pool trace (authoritative verdict map)."""
    return {'id': tid or ('ac%d' % stop), 'chain_id': 's1r-%s-%d' % (SHORT, stop),
            'metadata': {'outcomes_per_candidate': outcomes}}


def _recall_pool(stop, cand_short_ids, tid=None):
    """An O recall candidate-pool trace (raw pool, no verdict)."""
    cands = ['%s|some title|0.70|finding' % c for c in cand_short_ids]
    return {'id': tid or ('rc%d' % stop), 'chain_id': 's1r-%s-%d' % (SHORT, stop),
            'metadata': {'candidates': cands}}


def test_candidate_outcomes_reads_verdict_map():
    cands, dropped = _candidate_outcomes(
        {'outcomes_per_candidate': {'a': 'selected', 'b': 'dropped', 'c': 'dropped'}})
    assert set(cands) == {'a', 'b', 'c'}
    assert set(dropped) == {'b', 'c'}


def test_candidate_outcomes_falls_back_to_mirror_lists():
    cands, dropped = _candidate_outcomes({'selected': ['a'], 'dropped': ['b', 'c']})
    assert set(cands) == {'a', 'b', 'c'} and set(dropped) == {'b', 'c'}


def test_candidate_outcomes_raw_pool_has_no_verdict():
    cands, dropped = _candidate_outcomes(
        {'candidates': ['a|t|0.7|x', 'b|t|0.6|y']})
    assert cands == ['a', 'b'] and dropped is None   # caller derives dropped


def test_candidate_outcomes_malformed_is_empty():
    assert _candidate_outcomes(None) == ([], None)
    assert _candidate_outcomes({}) == ([], None)
    assert _candidate_outcomes({'candidates': 'not a list'}) == ([], None)


def test_dropped_authoritative_verdict_split():
    # surfaced from K surface_selected; dropped from the Δ verdict map; encoded
    # from the owning encode run at the same stop — ONE join, all roles.
    links = nodes_for_traces(
        [_surface(5, ['p1'])],
        # run@6 owns turn 5 (strict ownership, 4171a2e — run@5 never saw stop 5)
        [_encode(6, ['enc_full_1'], ['enc_full_2'])],
        [_turn(5)],
        recall_traces=[_outcomes(5, {'p1': 'selected', 'd1': 'dropped',
                                     'd2': 'dropped'})])
    r = links['u5']
    assert r['surfaced'] == ['p1']
    assert set(r['dropped']) == {'d1', 'd2'}
    assert set(r['encoded']) == {'enc_full_1', 'enc_full_2'}


def test_dropped_raw_pool_derives_pool_minus_surfaced():
    # No verdict map — only the raw O pool. dropped = candidates − surfaced.
    links = nodes_for_traces(
        [_surface(7, ['p1'])], [], [_turn(7)],
        recall_traces=[_recall_pool(7, ['p1', 'd1', 'd2', 'd3'])])
    r = links['u7']
    assert r['surfaced'] == ['p1']
    assert set(r['dropped']) == {'d1', 'd2', 'd3'}
    assert r['encoded'] == []


def test_surfaced_never_appears_in_dropped():
    # Even if a stale pool lists a surfaced id as a candidate, surfaced wins.
    links = nodes_for_traces(
        [_surface(3, ['p1', 'p2'])], [], [_turn(3)],
        recall_traces=[_recall_pool(3, ['p1', 'p2', 'd1'])])
    r = links['u3']
    assert set(r['surfaced']) == {'p1', 'p2'}
    assert r['dropped'] == ['d1']


def test_dropped_empty_without_recall_stream():
    # recall_traces omitted → dropped is empty, never missing (mirrors touched).
    links = nodes_for_traces([_surface(5, ['s'])], [], [_turn(5)])
    assert links['u5']['dropped'] == []


def test_dropped_missing_stop_yields_empty():
    links = nodes_for_traces([], [], [{'id': 'x', 'chain_id': 'no-stop-here'}],
                             recall_traces=[])
    assert links['x']['surfaced'] == [] and links['x']['dropped'] == []
    assert links['x']['encoded'] == []
    assert links['x']['fetched_by'] == {} and links['x']['floored_by'] == {}


# ── tool provenance (fetched_by / floored_by, from the K rows' tool_trace) ──

def test_tool_provenance_parses_result_and_dropped_ids():
    fetched, floored = _tool_provenance({'tool_trace': [
        {'round': 0, 'stop_reason': 'tool_use', 'tool_calls': [
            _tool_call('recall_by_time', result_ids=['aaaa1111', 'bbbb2222']),
            _tool_call('recall_topical', result_ids=['cccc3333'],
                       dropped_ids=['dddd4444', 'eeee5555']),
        ]},
        {'round': 1, 'stop_reason': 'end_turn', 'tool_calls': []},
    ]})
    assert fetched == {'aaaa1111': 'recall_by_time', 'bbbb2222': 'recall_by_time',
                       'cccc3333': 'recall_topical'}
    assert floored == {'dddd4444': 'recall_topical', 'eeee5555': 'recall_topical'}


def test_tool_provenance_survives_old_and_malformed_traces():
    # pre-2026-07-02 traces: tool_calls without result_ids/dropped_ids
    fetched, floored = _tool_provenance({'tool_trace': [
        {'round': 0, 'tool_calls': [{'tool': 'recall_by_time', 'result_count': 9}]}]})
    assert fetched == {} and floored == {}
    assert _tool_provenance(None) == ({}, {})
    assert _tool_provenance({'tool_trace': 'garbage'}) == ({}, {})


def test_fetched_and_floored_roles_join_by_stop():
    surface = [_surface(3, ['aaaa1111'], tool_trace=[
        {'round': 0, 'stop_reason': 'tool_use', 'tool_calls': [
            _tool_call('recall_topical', result_ids=['aaaa1111', 'ffff6666'],
                       dropped_ids=['dddd4444'])]}])]
    links = nodes_for_traces(surface, [], [_turn(3), _turn(4)])
    l3 = links['u3']
    # aaaa1111 was fetched AND picked; ffff6666 fetched, pooled, not picked.
    assert l3['fetched_by'] == {'aaaa1111': 'recall_topical',
                                'ffff6666': 'recall_topical'}
    assert l3['floored_by'] == {'dddd4444': 'recall_topical'}
    # other turns unaffected
    assert links['u4']['fetched_by'] == {} and links['u4']['floored_by'] == {}


class _RolesStubBrain:
    """Stub serving the three doors the episodic gather pulls."""
    def __init__(self):
        self.calls = []

    def query_traces(self, **kw):
        self.calls.append(kw)
        rt = kw.get('ref_type')
        if rt == 'additionalContext':
            return {'events': [_outcomes(5, {'p': 'selected', 'd': 'dropped'})]}
        if rt == 'surface_selected':
            return {'events': [_surface(5, ['p'])]}
        if rt == 'encoding_run':
            # stop 6: a run owns turns with chain-stop < its own (strict
            # ownership, 4171a2e) — run@6 is turn 5's owning run.
            return {'events': [_encode(6, ['e'], [])]}
        return {'events': []}


def test_gather_recall_streams_and_dropped_compose():
    brain = _RolesStubBrain()
    st = gather(brain, 'sess-xyz', streams=('surface', 'encode', 'recall'))
    assert len(st['surface']) == 1 and len(st['encode']) == 1 and len(st['recall']) == 1
    rts = {c['ref_type'] for c in brain.calls}
    assert rts == {'additionalContext', 'surface_selected', 'encoding_run'}
    for c in brain.calls:
        assert c['scale'] == 's1' and c['session_id'] == 'sess-xyz' and c['hours'] is None
    links = nodes_for_traces(st['surface'], st['encode'], [_turn(5)],
                             recall_traces=st['recall'])
    r = links['u5']
    assert r['surfaced'] == ['p'] and r['encoded'] == ['e'] and r['dropped'] == ['d']


def test_gather_unknown_stream_is_loud():
    import pytest
    with pytest.raises(KeyError):
        gather(_RolesStubBrain(), 'sess-xyz', streams=('surface', 'bogus'))


# ── render integration: <provenance> in the lived-sequence timeline (encode.py) ──

from servers.scales.s1.encode import _render_lived_sequence_timeline  # noqa: E402


def _msg(ref_type, tid, stop, content, ts):
    return {'id': tid, 'ref_type': ref_type, 'chain_id': 's0-%s-%d' % (SHORT, stop),
            'summary': content[:60], 'created_at': ts, 'metadata': {'content': content}}


def _window(eps):
    """The episodes grouped into the encoder's window, oldest first — what
    encode._window hands the render (the whole conversation here)."""
    from servers.trace_contract import group_turns
    return group_turns(sorted(eps, key=lambda e: e['created_at']))


class _ProvBrain:
    """A brain stub for the timeline's provenance: query_traces serves
    gather(); the turns come from _window."""
    def __init__(self, episodes, surface_traces, encode_traces):
        self._episodes = episodes
        self._surface = surface_traces
        self._encode = encode_traces

    def speaker_for(self, session_id, ref_type):
        # An install that declares no counterpart: no speaker is named.
        return ''

    def query_traces(self, **kw):
        if kw.get('ref_type') == 'surface_selected':
            return {'events': self._surface}
        if kw.get('ref_type') == 'encoding_run':
            return {'events': self._encode}
        return {'events': []}


# Four turns (stops 5–8). A prior run@7 covers stops 5,6 (STRICT: a run's
# post-increment stop is > every chain-stop it saw); stops 7,8 are the
# unencoded tail. Surface fired (only) on turn 5.
_EPS = [
    _msg('user_message', 'u5', 5, 'turn five user', '2026-06-29T00:00:01'),
    _msg('assistant_message', 'a5', 5, 'turn five reply', '2026-06-29T00:00:02'),
    _msg('user_message', 'u6', 6, 'turn six user', '2026-06-29T00:00:03'),
    _msg('assistant_message', 'a6', 6, 'turn six reply', '2026-06-29T00:00:04'),
    _msg('user_message', 'u7', 7, 'turn seven user', '2026-06-29T00:00:05'),
    _msg('assistant_message', 'a7', 7, 'turn seven reply', '2026-06-29T00:00:06'),
    _msg('user_message', 'u8', 8, 'turn eight user', '2026-06-29T00:00:07'),
    _msg('assistant_message', 'a8', 8, 'turn eight reply', '2026-06-29T00:00:08'),
]
_SURF = [_surface(5, ['nodeAAAA1111', 'nodeBBBB2222'])]
_ENC = [_encode(7, ['nodeCCCC3333'], [], tid='run7')]


def test_turns_carry_encoded_attr_and_frontier_ids():
    # Coverage lives on the turn (encoded="true|false"); provenance carries only
    # REAL refs — no ✓ marker. The frontier turn still shows
    # the owning run's full id-list, once.
    brain = _ProvBrain(_EPS, _SURF, _ENC)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(_EPS))
    assert '<turn n="1" encoded="true">' in out       # covered turns say so
    assert '<turn n="2" encoded="true">' in out       # frontier is also covered
    assert '<turn n="3" encoded="false">' in out      # first turn PAST the run
    assert '<turn n="4" encoded="false">' in out      # the unencoded tail
    assert '✓' not in out                              # marker retired
    t5, t6, t7 = (out.split('<turn n="%d"' % n)[1].split('</turn>')[0]
                  for n in range(1, 4))
    assert 'surfaced: id:nodeAAAA id:nodeBBBB' in t5
    assert 'encoded(S1S)' not in t5                    # covered-not-frontier: no marker
    assert 'encoded(S1S): id:nodeCCCC' in t6           # full id-list at frontier only
    assert 'encoded(S1S)' not in t7                    # past the run: unencoded


def test_provenance_unencoded_tail_has_no_encoded_marker():
    brain = _ProvBrain(_EPS, _SURF, _ENC)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(_EPS))
    t8 = out.split('<turn n="4"')[1].split('</turn>')[0]
    # turn 8 is past the last run → no encoded marker at all (the boundary)
    assert 'encoded(S1S)' not in t8
    # full id-list appears exactly once across the whole timeline (frontier only)
    assert out.count('id:nodeCCCC') == 1


def test_provenance_edge_only_run_renders_no_marker_but_attr_says_covered():
    # A run that wrote only edges (connect) has empty created+revised, so encoded
    # is [] even though encoded_by is set. No provenance marker renders (nothing
    # to dereference), but the turn attr still states coverage.
    enc = [_encode(7, [], [], tid='run7')]  # edge-only run
    out = _render_lived_sequence_timeline(_ProvBrain(_EPS, [], enc), 'sess', _window(_EPS))
    assert 'encoded(S1S)' not in out                   # no marker at all
    assert '✓' not in out
    assert 'id:' not in out                            # nothing to dereference
    assert '<turn n="2" encoded="true">' in out        # coverage on the attr
    assert '<turn n="3" encoded="false">' in out       # strictly past the run
    assert '<turn n="4" encoded="false">' in out


def test_encoded_turns_keep_full_text():
    # encoded_turn_trim retired at activation: covered turns keep their full
    # text on both arms — coverage is stated by the `encoded` attr, and (on the
    # policy arm) it is the turn's <actions> that stub, never the message.
    long_body = 'x' * 800
    eps = [
        _msg('user_message', 'u5', 5, 'covered ' + long_body, '2026-06-29T00:00:01'),
        _msg('assistant_message', 'a5', 5, 'reply five', '2026-06-29T00:00:02'),
        _msg('user_message', 'u6', 6, 'tail ' + long_body, '2026-06-29T00:00:03'),
        _msg('assistant_message', 'a6', 6, 'reply six', '2026-06-29T00:00:04'),
    ]
    enc = [_encode(6, ['nodeCCCC3333'], [], tid='run6')]   # covers stop 5 only (6 > 5)
    out = _render_lived_sequence_timeline(_ProvBrain(eps, [], enc), 'sess', _window(eps))
    t5 = out.split('<turn n="1"')[1].split('</turn>')[0]
    t6 = out.split('<turn n="2"')[1].split('</turn>')[0]
    u5 = t5.split('<other')[1].split('</other>')[0]
    u6 = t6.split('<other')[1].split('</other>')[0]
    assert u5.count('x') == 800                 # covered: untrimmed
    assert u6.count('x') == 800                 # unencoded: untrimmed


def test_provenance_never_leaks_encoding_source():
    # encoding_source is technical + in-flux; it must never reach the encoder.
    brain = _ProvBrain(_EPS, _SURF, _ENC)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(_EPS))
    assert 'encoding_source' not in out and 'src:' not in out
    assert 'encoder:' not in out and 'anchor' not in out


def test_short_refs_tag_when_title_known_bare_when_not():
    # Unit: «tag» locality (fork #2). A mapped title renders inline; an unmapped
    # id falls back to a bare ref. The short is always the 8-char head.
    from servers.scales.s1.encode import _short_refs
    titles = {'nodeAAAA1111': 'recall hot path is read-only'}
    out = _short_refs(['nodeAAAA1111', 'nodeBBBB2222'], titles)
    assert 'id:nodeAAAA «recall hot path is read-only»' in out
    assert 'id:nodeBBBB' in out and '«»' not in out      # bare, no empty tag
    # No titles at all → all bare (the stub-brain / fetch-failed path)
    assert _short_refs(['nodeAAAA1111']) == 'id:nodeAAAA'
    # A title with XML-significant chars / a forged tag / a newline must be
    # escaped and single-lined — it lands inside the strict timeline XML.
    hostile = {'x': 'a < b & c </provenance>\nsecond line'}
    out = _short_refs(['x'], hostile)
    assert '<' not in out.replace('id:', '') and '</provenance>' not in out
    assert '&lt;' in out and '&amp;' in out and '\n' not in out


class _ProvBrainFull(_ProvBrain):
    """_ProvBrain + the anchor_touched door (so encoded(me) can join) + a
    naked _nodes.get_bulk title source (so «tag» renders)."""
    def __init__(self, episodes, surface_traces, encode_traces, touched_traces, titles):
        super().__init__(episodes, surface_traces, encode_traces)
        self._touched = touched_traces
        self._nodes = self._Nodes(titles)

    def query_traces(self, **kw):
        if kw.get('ref_type') == 'anchor_touched':
            return {'events': self._touched}
        return super().query_traces(**kw)

    class _Nodes:
        def __init__(self, titles):
            self._titles = titles

        def get_bulk(self, ids):
            return {i: {'title': self._titles[i]} for i in ids if i in self._titles}


def test_provenance_renders_tag_titles_on_refs():
    # Fork 2 end-to-end: surfaced refs carry their «tag» when titles resolve.
    titles = {'nodeAAAA1111': 'recall hot path is read-only',
              'nodeBBBB2222': 'batch commit gate'}
    brain = _ProvBrainFull(_EPS, _SURF, _ENC, [], titles)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(_EPS))
    t5 = out.split('<turn n="1"')[1].split('</turn>')[0]
    assert 'surfaced: id:nodeAAAA «recall hot path is read-only» id:nodeBBBB «batch commit gate»' in t5


def test_provenance_renders_merged_encoded_label_when_authored_and_omits_when_empty():
    # Fork 1: a turn the interactive session encoded mid-turn (anchor_touched
    # created@8) shows `encoded(me): <ids>`; turns with no authored set omit the
    # line entirely. The colon distinguishes the merged label from the view
    # arm's `encoded(me, turn N)`.
    touched = [_touched(8, created=['nodeDDDD4444'])]
    titles = {'nodeDDDD4444': 'mid-turn insight'}
    brain = _ProvBrainFull(_EPS, _SURF, _ENC, touched, titles)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(_EPS))
    t5, t8 = (out.split('<turn n="%d"' % n)[1].split('</turn>')[0] for n in (1, 4))
    assert 'encoded(me): id:nodeDDDD «mid-turn insight»' in t8       # filled at turn 8
    assert 'encoded(me):' not in t5                                  # omitted elsewhere
    assert out.count('encoded(me):') == 1                            # exactly the one turn


@pytest.mark.parametrize('view_policy', [False, True])
def test_turns_sharing_a_stop_show_its_provenance_once_on_the_last(view_policy):
    # Several prompts before one reply share a chain stop, and the link join is
    # by stop, so every turn on it holds the same refs. The line renders once,
    # on the stop's last turn; the earlier turns keep their messages, no line.
    eps = [
        # stop 5: two operator prompts, then the reply (an interrupted turn)
        _msg('user_message', 'u5a', 5, 'first ask', '2026-06-29T00:00:01'),
        _msg('user_message', 'u5b', 5, 'and another thing', '2026-06-29T00:00:02'),
        _msg('assistant_message', 'a5', 5, 'both done', '2026-06-29T00:00:03'),
        # stop 6: a subagent's report opens it, the operator answers on it
        _msg('subagent_message', 'r6', 6, 'report: two findings', '2026-06-29T00:00:04'),
        _msg('user_message', 'u6', 6, 'fix the first', '2026-06-29T00:00:05'),
        _msg('assistant_message', 'a6', 6, 'fixed', '2026-06-29T00:00:06'),
    ]
    surf = [_surface(5, ['nodeAAAA1111']), _surface(6, ['nodeBBBB2222'])]
    enc = [_encode(6, ['nodeCCCC3333'], [], tid='run6')]   # covers stop 5
    touched = [_touched(5, created=['nodeDDDD4444'])]
    brain = _ProvBrainFull(eps, surf, enc, touched, {})
    out = _render_lived_sequence_timeline(brain, 'sess', _window(eps),
                                          view_policy=view_policy)

    turns = out.split('</turn>')[:-1]
    assert ['<provenance>' in t for t in turns] == [False, True, False, True]
    assert 'trace="u5a"' in turns[0] and 'trace="u5b"' in turns[1]
    assert 'nodeAAAA' in turns[1] and 'nodeDDDD' in turns[1]   # the stop's refs
    assert 'nodeCCCC' in turns[1]                     # the run's list, on its last turn
    assert 'trace="r6"' in turns[2] and 'nodeBBBB' in turns[3]
    for nid in ('nodeAAAA', 'nodeBBBB', 'nodeCCCC', 'nodeDDDD'):
        assert out.count(nid) == 1


def test_timeline_degrades_to_piece1_when_provenance_unavailable():
    # A brain with no query_traces (the piece-1 stub shape) → guarded fallback:
    # the timeline still renders, just without any <provenance> line.
    class _NoDoor:
        def speaker_for(self, session_id, ref_type):
            return ''

    out = _render_lived_sequence_timeline(_NoDoor(), 'sess', _window(_EPS))
    assert '<turn n="1">' in out and '<provenance>' not in out
    # coverage unknown on the degraded path → no encoded attr either
    assert 'encoded=' not in out
