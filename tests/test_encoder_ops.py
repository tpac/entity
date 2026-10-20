"""eval/encoder_ops — the one reader of encoder op dumps, swap-aware, shared by
the A/B harness and the shape scorer. Every text it returns is the NEW text a
revise proposes; the removed `old` must never satisfy a fact check."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree
from eval.encoder_ops import (edge_entries, kind, new_text, ops_of,  # noqa: E402
                              revise_surfaces, revise_text, swaps_of)


def test_ops_of_flattens_every_batch_shape():
    assert ops_of({'args': {'operations': [{'op': 'remember', 'title': 'a'}, 'junk']}}) \
        == [{'op': 'remember', 'title': 'a'}]
    # brain_batch operations that arrived JSON-encoded, unwrapped as dispatch does
    assert ops_of({'args': {'operations': '[{"op": "archive", "node_id": "abcdef01"}]'}}) \
        == [{'op': 'archive', 'node_id': 'abcdef01'}]
    assert ops_of({'args': {'revisions': [{'node_id': 'abcdef01', 'reason': 'r'}]}}) \
        == [{'node_id': 'abcdef01', 'reason': 'r'}]
    assert ops_of({'args': {'connections': [{'source_id': 'a' * 8, 'target_id': 'b' * 8}]}}) \
        [0]['source_id'] == 'a' * 8
    assert ops_of({'args': {'node_id': 'abcdef01', 'reason': 'single'}}) \
        == [{'node_id': 'abcdef01', 'reason': 'single'}]
    assert ops_of({'args': {}}) == []


def test_ops_of_folds_batch_level_connect_to_into_each_node():
    w = {'args': {'nodes': [{'title': 'n1', 'connect_to': [
                                {'target': 'aaaaaaaa', 'relation': 'grounds', 'why': 'w'}]},
                            {'title': 'n2'}],
                  'connect_to': [{'target': 'bbbbbbbb', 'relation': 'during', 'why': 'shared'}]}}
    ops = ops_of(w)
    assert [len(o['connect_to']) for o in ops] == [2, 1]
    assert ops[1]['connect_to'][0]['target'] == 'bbbbbbbb'
    assert w['args']['nodes'][1].get('connect_to') is None   # copies — the recorded write is untouched


def test_kind_declared_or_inferred():
    assert kind({'op': 'absorb'}) == 'absorb'
    assert kind({'node_id': 'abcdef01'}) == 'revise'
    assert kind({'source_id': 'a' * 8, 'target_id': 'b' * 8}) == 'connect'
    assert kind({'title': 't', 'content': 'c'}) == 'remember'


def test_new_text_is_the_proposed_text_only():
    assert new_text('plain') == 'plain'
    assert new_text({'old': 'stale 9.6.0', 'new': 'fresh 9.7.2'}) == 'fresh 9.7.2'
    assert new_text([{'old': 'a', 'new': 'x'}, {'old': 'b', 'new': 'y'}]) == 'x y'
    assert new_text(['1a2b3c4d', '5e6f7a8b']) == '1a2b3c4d 5e6f7a8b'
    assert new_text(None) == ''


def test_swaps_of_normalizes_the_alias_onto_content():
    op = {'op': 'revise', 'node_id': 'abcdef01', 'reason': 'r',
          'title': {'old': 'week 6', 'new': 'completed'},
          'content_edits': [{'old': 'a', 'new': 'b'}],
          'situation': 'a bare value is not a swap',
          'connect_to': [{'target': 'aaaaaaaa', 'relation': 'x', 'why': 'y'}]}
    s = swaps_of(op)
    assert set(s) == {'title', 'content'}
    assert s['content'] == [{'old': 'a', 'new': 'b'}]
    assert s['title'] == [{'old': 'week 6', 'new': 'completed'}]


def test_edge_entries_read_target_or_title_and_swap_new_text():
    rem = {'op': 'remember', 'title': 'new node', 'connect_to': [
        {'title': 'aaaaaaaa', 'relation': 'grounds', 'why': 'why-a'},
        {'target': 'bbbbbbbb', 'relations': [{'relation': 'after', 'why': 'w1'},
                                             {'relation': 'during', 'why': 'w2'}]}]}
    e = edge_entries(rem)
    assert [x['target'] for x in e] == ['aaaaaaaa', 'bbbbbbbb', 'bbbbbbbb']
    assert [x['relation'] for x in e] == ['grounds', 'after', 'during']
    assert all(x['source'] is None and x['via'] == 'connect_to' for x in e)

    rev = {'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r', 'connect_to': [
        {'target': '15bbfd64', 'relation': {'old': 'gaps_in', 'new': 'blocks'},
         'why': {'old': 'both manifests still say 9.6.0',
                 'new': 'manifests moved to 9.7.2, still short of 0.9.0'}}]}
    (x,) = edge_entries(rev)
    assert x == {'via': 'connect_to', 'source': 'd827d22f', 'target': '15bbfd64',
                 'relation': 'blocks',
                 'why': 'manifests moved to 9.7.2, still short of 0.9.0'}

    con = {'op': 'connect', 'source_id': 'a' * 8, 'target_id': 'b' * 8,
           'relation': 'similar_to', 'description': 'desc'}
    (y,) = edge_entries(con)
    assert y == {'via': 'connect', 'source': 'a' * 8, 'target': 'b' * 8,
                 'relation': 'similar_to', 'why': 'desc'}
    # a why-less edge is still an edge (empty why), never dropped
    (z,) = edge_entries({'op': 'remember', 'connect_to': [{'target': 'c' * 8, 'relation': 'r'}]})
    assert z['why'] == ''
    # a disconnect removes an edge and asserts none — no entry, so it can
    # never blank out the pair's real why in the surfaces scorer
    assert edge_entries({'op': 'disconnect', 'source_id': 'a' * 8,
                         'target_id': 'b' * 8, 'relation': 'x'}) == []
    # `description` is the write path's alias of `why` on a connect_to entry
    (w,) = edge_entries({'op': 'remember', 'connect_to': [
        {'target': 'd' * 8, 'relation': 'r', 'description': 'the alias why'}]})
    assert w['why'] == 'the alias why'


def test_revise_surfaces_and_text_carry_new_values_only():
    op = {'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
          'title': {'old': 'version 9.6.0', 'new': 'version 9.7.2'},
          'content': [{'old': 'say 9.6.0', 'new': 'say 9.7.2 (was 9.6.0)'}],
          'situation': 'when the manifest version is asked'}
    assert revise_surfaces(op) == {'title': 'version 9.7.2',
                                   'situation': 'when the manifest version is asked',
                                   'content': 'say 9.7.2 (was 9.6.0)'}
    t = revise_text(op)
    assert 'version 9.6.0' not in t and 'version 9.7.2' in t
    alias = {'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
             'content_edits': [{'old': 'a', 'new': 'b'}]}
    assert revise_surfaces(alias) == {'content': 'b'}


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))


def _harness():
    import eval.encoder_prompt_ab as harness
    return harness


def test_score_arm_counts_ops_by_kind():
    """kind() decides, not key presence: a connect or disconnect is not a
    create, an archive is not a revise."""
    score_arm = _harness().score_arm
    log = {'reads': [], 'rounds': 1, 'final_text': '',
           'usage': {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0},
           'writes': [{'tool': 'brain_batch', 'args': {'operations': [
               {'op': 'remember', 'title': 'n', 'content': 'c'},
               {'op': 'revise', 'node_id': 'abcdef01', 'reason': 'r', 'title': 't'},
               {'op': 'connect', 'source_id': 'a' * 8, 'target_id': 'b' * 8,
                'relation': 'similar_to', 'description': 'd'},
               {'op': 'archive', 'node_id': 'abcdef02', 'reason': 'gone'},
               {'op': 'disconnect', 'source_id': 'a' * 8, 'target_id': 'b' * 8,
                'relation': 'x'}]}}]}
    s = score_arm(log, set(), {})
    assert (s['creates'], s['revises'], s['connects'], s['archives']) == (1, 1, 1, 2)


def test_gist_flag_never_swallows_a_capture_path():
    """`--gist` is a flag: a capture path written after it is a capture, not
    a gist override file. `--gist-file FILE` is the override door and implies
    the splice."""
    build_parser = _harness().build_parser
    a = build_parser().parse_args(['--gist', 'payloads/x/000-prompt.md', '--arms', 'F'])
    assert a.gist is True and a.gist_file is None
    assert a.captures == ['payloads/x/000-prompt.md']
    b = build_parser().parse_args(['--gist-file', 'gist_b.md', 'payloads/x/000-prompt.md'])
    assert b.gist is False and b.gist_file == 'gist_b.md'
    assert b.captures == ['payloads/x/000-prompt.md']


def test_splice_gist_fills_or_strips_the_slot():
    """Enabled: the effective text sits between the last block closer and
    <timeline>, whatever the capture carried there. Disabled: the slot is
    empty — byte-for-byte what production assembles with `enabled: false`."""
    splice_gist = _harness().splice_gist
    # production: "<node_catalog>…</node_catalog>\n\n" + gist + "\n\n" + <timeline
    cap = '<node_catalog>\nnodes\n</node_catalog>\n\nold gist words\n\n<timeline now="x">\n…'
    filled, replaced = splice_gist(cap, 'NEW GIST')
    assert filled == '<node_catalog>\nnodes\n</node_catalog>\n\nNEW GIST\n\n<timeline now="x">\n…'
    assert replaced == len('old gist words')
    stripped, replaced = splice_gist(cap, None)
    assert stripped == '<node_catalog>\nnodes\n</node_catalog>\n\n<timeline now="x">\n…'
    assert replaced == len('old gist words')
    assert splice_gist(filled, 'NEW GIST') == (filled, len('NEW GIST'))   # idempotent
    # production: scout legend + "\n" + gist — no blank line after that closer
    legend = '</scout_legend>\nold words\n\n<timeline now="x">\n…'
    assert splice_gist(legend, 'G') == ('</scout_legend>\nG\n\n<timeline now="x">\n…', len('old words'))
    assert splice_gist(legend, None) == ('</scout_legend>\n<timeline now="x">\n…', len('old words'))
    assert splice_gist('no timeline here', 'g') == (None, 0)


def test_swap_fidelity_refuses_like_production_on_fields_and_edges():
    """The advisory uses contract.apply_swaps — the primitive brain.revise
    applies — so its refusals are production's: exactly-once matching, in
    order, a failure fails the field. connect_to swaps are checked against
    the stored edge (why → description, relation → an active row), resolved
    the way brain.revise resolves the edge."""
    score_swap_fidelity = _harness().score_swap_fidelity
    stored = {'d827d22f': {'id': 'd827d22f' + '0' * 24, 'title': 'brain/9.6.0',
                           'content': 'say 9.6.0 here and 9.6.0 there',
                           '_metadata': {'reasoning': 'measured at 9.6.0'}}}
    edges = {frozenset(('d827d22f', '15bbfd64')):
             {'gaps_in': 'both manifests still say 9.6.0'}}
    ops = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
            'title': {'old': '9.6.0', 'new': '9.7.2'},                   # lands
            'reasoning': [{'old': 'at 9.6.0', 'new': 'at 9.7.2'}],       # lands (KV field)
            'content': [{'old': '9.6.0', 'new': '9.7.2'}],               # ambiguous — two places
            'situation': {'old': 'a', 'new': 'b'},                       # nothing stored
            'connect_to': [
                {'target': '15bbfd64', 'why': {'old': 'still say 9.6.0',
                                               'new': 'moved to 9.7.2'}},      # lands: one relation
                {'target': '15bbfd64', 'relation': {'old': 'blocks', 'new': 'gaps_in'}},  # no such row
                {'target': '15bbfd64', 'relation': 'gaps_in',
                 'why': {'old': 'both manifests', 'new': 'both manifests'}},    # old == new
                {'target': 'cccccccc', 'why': {'old': 'a', 'new': 'b'}},         # no edge
                {'target': '15bbfd64', 'relation': 'gaps_in', 'why': 'bare text is not a swap'}]}]
    rows = score_swap_fidelity(ops, stored, edges)
    by_field = {}
    for r in rows:
        by_field.setdefault(r['field'], []).append(r['error'])
    assert set(by_field) == {'content', 'situation', 'connect_to:15bbfd64', 'connect_to:cccccccc'}
    assert 'matches 2 places' in by_field['content'][0]
    assert by_field['situation'] == ['no stored value to swap into']
    assert by_field['connect_to:cccccccc'] == ['no edge between the pair to swap into']
    assert len(by_field['connect_to:15bbfd64']) == 2
    assert "no active relation 'blocks'" in by_field['connect_to:15bbfd64'][0]
    assert 'identical' in by_field['connect_to:15bbfd64'][1]
    # a later op patches what the earlier one produced — the why swap landed
    later = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
              'connect_to': [{'target': '15bbfd64',
                              'why': {'old': 'moved to 9.7.2', 'new': 'at 9.7.2'}}]}]
    assert score_swap_fidelity(ops + later, stored, edges) == rows


def test_swap_fidelity_edge_corners_match_production():
    """A relation swap LIST of any length but one is a refusal in production
    ('a relation takes one swap'), not a bare relation; `description` is the
    write path's alias of `why`; and a revise on a node the copy no longer
    holds says nothing about its edges — skipped whole, no spurious rows."""
    h = _harness()
    stored = {'d827d22f': {'id': 'd827d22f' + '0' * 24, 'title': 't', 'content': 'c',
                           'connections': [{'id': '15bbfd64' + '0' * 24, 'relations': [
                               {'relation': 'gaps_in', 'description': 'both manifests still say 9.6.0'}]}]}}
    edges = h.stored_edges(stored)
    assert edges == {frozenset(('d827d22f', '15bbfd64')): {'gaps_in': 'both manifests still say 9.6.0'}}
    two = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
            'connect_to': [{'target': '15bbfd64',
                            'relation': [{'old': 'gaps_in', 'new': 'blocks'}, {'old': 'x', 'new': 'y'}],
                            'why': {'old': 'still say 9.6.0', 'new': 'moved to 9.7.2'}}]}]
    (row,) = h.score_swap_fidelity(two, stored, edges)
    assert 'one swap' in row['error']
    alias = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
              'connect_to': [{'target': '15bbfd64', 'description': {'old': 'still say 9.6.0',
                                                                    'new': 'moved to 9.7.2'}}]}]
    assert h.score_swap_fidelity(alias, stored, edges) == []          # lands, via the alias
    gone = [{'op': 'revise', 'node_id': 'zzzzzzzz', 'reason': 'r',
             'connect_to': [{'target': '15bbfd64', 'why': {'old': 'a', 'new': 'b'}}]}]
    assert h.score_swap_fidelity(gone, stored, edges) == []           # node absent → skipped whole


class _FrozenCopy:
    """brain.get_node's shape over a dict — what write_answerer reads."""
    def __init__(self, nodes):
        self.nodes = nodes

    def get_node(self, ids):
        return {i + '0' * 24: dict(self.nodes[i]) for i in ids if i in self.nodes}


def test_write_answerer_refuses_like_production_and_patches_what_lands():
    """A revise whose swap would miss is answered with production's refusal
    and writes nothing; a landed one patches the view the next op sees; a
    node the copy does not hold (created this run) is not checked."""
    answer = _harness().write_answerer(_FrozenCopy(
        {'d827d22f': {'id': 'd827d22f' + '0' * 24, 'title': 't',
                      'content': 'none of the trial verbs are seeded.'}}))
    ops = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
            'content': {'old': 'None of the trial verbs are seeded.', 'new': 'x'}},
           {'op': 'remember', 'title': 'fresh', 'content': 'c'},
           {'op': 'revise', 'node_id': 'ffffffff', 'reason': 'r',
            'content': {'old': 'anything', 'new': 'y'}}]
    result, refused = answer('brain_batch', {'operations': ops})
    assert refused == [0]
    rows = result['result']['results']
    assert rows[0]['ok'] is False and 'Nothing from this op was written' in rows[0]['error']
    assert 'That text reads: "none of the trial verbs are seeded."' in rows[0]['error']
    assert rows[0]['current'] == {'content': 'none of the trial verbs are seeded.'}
    assert [r['ok'] for r in rows[1:]] == [True, True]
    assert (result['result']['succeeded'], result['result']['failed']) == (2, 1)
    # the retry with the real text lands, and the next swap sees its result
    ok = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
           'content': {'old': 'none of the trial', 'new': 'none of the problem'}}]
    assert answer('brain_batch', {'operations': ok})[1] == []
    again = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
              'content': {'old': 'none of the problem verbs', 'new': 'z'}}]
    assert answer('brain_batch', {'operations': again})[1] == []
    # a bare revise tool refuses the whole call, as dispatch does
    bare, refused = answer('revise', {'node_id': 'd827d22f', 'reason': 'r',
                                      'content': {'old': 'not there at all', 'new': 'q'}})
    assert refused == [0] and bare['ok'] is False
    assert bare['current'] == {'content': 'z are seeded.'}   # the running view


class _HistoryCopy:
    """A copy that has moved on since a capture: get_node (absorbs followed
    unless follow_absorbed=False) and the node_revised / node_archived traces
    the as-of view rolls back through."""
    def __init__(self, nodes, absorbed=None, events=None):
        self.nodes, self.absorbed, self.events = nodes, absorbed or {}, events or []

    def get_node(self, ids, follow_absorbed=True):
        import copy
        out = {}
        for i in ids:
            j = self.absorbed.get(i, i) if follow_absorbed else i
            if j in self.nodes:
                out[i + '0' * 24] = copy.deepcopy(self.nodes[j])
        return out

    def query_traces(self, ref_type, ref_id, hours=None, limit=100):
        return {'events': [e for e in self.events
                           if e['ref_type'] == ref_type and e['ref_id'] == ref_id]}


def test_write_answerer_as_of_answers_the_node_the_capture_showed():
    """The frozen arm's answers come from each node as it stood at the
    capture: a revision after it is rolled back (the payload's patch lands,
    a miss never quotes the later text), an absorb after it answers with the
    node's own record, and a node created after it is not checked; a node
    already archived at the capture is absent too."""
    h = _harness()
    as_of = '2026-10-09T18:48:00+00:00'
    later = '2026-10-09T23:34:00+00:00'
    copy_ = _HistoryCopy(
        {'a7b8a9c2': {'id': 'a7b8a9c2', 'created_at': '2026-10-01T00:00:00+00:00',
                      'title': 'REPLACED, shipped', 'content': 'resolved and deployed'},
         '545a96dd': {'id': '545a96dd', 'created_at': '2026-10-01T00:00:00+00:00',
                      'archived': 1, 'title': 'own',
                      'content': 'the absorbed node said this'},
         'c3c3c3c3': {'id': 'c3c3c3c3', 'created_at': '2026-09-01T00:00:00+00:00',
                      'archived': 1, 'title': 'archived long before', 'content': 'old'},
         'f0f0f0f0': {'id': 'f0f0f0f0', 'created_at': '2026-10-01T00:00:00+00:00',
                      'title': 'survivor', 'content': 'the survivor says that'},
         'b1b1b1b1': {'id': 'b1b1b1b1', 'created_at': '2026-10-10T00:00:00+00:00',
                      'title': 'born after', 'content': 'x'}},
        absorbed={'545a96dd': 'f0f0f0f0'},
        events=[{'ref_type': 'node_revised', 'ref_id': 'a7b8a9c2', 'created_at': later,
                 'metadata': {'deltas': [
                     {'field': 'content', 'old': 'redesign running, 6 questions',
                      'new': 'resolved and deployed'},
                     {'field': 'title', 'old': 'redesign running', 'new': 'REPLACED, shipped'}]}},
                {'ref_type': 'node_archived', 'ref_id': '545a96dd', 'created_at': later,
                 'metadata': {'deltas': [{'field': 'archived', 'old': 0, 'new': 1}]}}])
    then = h.stored_nodes(copy_, ['a7b8a9c2', '545a96dd', 'b1b1b1b1', 'c3c3c3c3'],
                          as_of)
    assert then['a7b8a9c2']['content'] == 'redesign running, 6 questions'
    assert then['a7b8a9c2']['title'] == 'redesign running'
    assert then['545a96dd']['content'] == 'the absorbed node said this'
    assert not then['545a96dd']['archived']          # archived after the capture
    assert 'b1b1b1b1' not in then and 'c3c3c3c3' not in then
    patch = [{'op': 'revise', 'node_id': 'a7b8a9c2', 'reason': 'r',
              'content': {'old': '6 questions', 'new': '6 answers'}},
             {'op': 'revise', 'node_id': '545a96dd', 'reason': 'r',
              'content': {'old': 'absorbed node said', 'new': 'absorbed node says'}},
             {'op': 'revise', 'node_id': 'b1b1b1b1', 'reason': 'r',
              'content': {'old': 'not checked', 'new': 'y'}}]
    assert h.write_answerer(copy_, as_of)('brain_batch', {'operations': patch})[1] == []
    # without as_of the copy as it is answers: every patch misses, the
    # later-born node's included (the copy holds it)
    assert h.write_answerer(copy_)('brain_batch', {'operations': patch})[1] == [0, 1, 2]
    # a miss quotes the node as the capture showed it, never the later text
    miss = [{'op': 'revise', 'node_id': 'a7b8a9c2', 'reason': 'r',
             'content': {'old': 'not in either version', 'new': 'z'}}]
    result, refused = h.write_answerer(copy_, as_of)('brain_batch', {'operations': miss})
    assert refused == [0]
    assert result['result']['results'][0]['current'] == {
        'content': 'redesign running, 6 questions'}
    # reads are served from the copy as it is — the score names the ones
    # that showed later text; query strings are not node ids
    log = {'rounds': 1, 'final_text': '', 'writes': [],
           'usage': {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0},
           'reads': [{'tool': 'get_nodes', 'args': {'node_ids': ['a7b8a9c2', 'f0f0f0f0']}},
                     {'tool': 'recall', 'args': {'query': 'deadbeef and more words'}}]}
    assert h.score_arm(log, set(), {}, copy_, as_of=as_of)['reads_drifted'] == ['a7b8a9c2']
    assert h.score_arm(log, set(), {}, copy_)['reads_drifted'] == []


def test_score_arm_counts_refused_revises_apart():
    """`revises` is what the run tried; a refused one is a failed attempt."""
    log = {'reads': [], 'rounds': 1, 'final_text': '',
           'usage': {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0},
           'writes': [{'tool': 'brain_batch', 'refused': [1], 'args': {'operations': [
               {'op': 'revise', 'node_id': 'abcdef01', 'reason': 'r', 'title': 't'},
               {'op': 'revise', 'node_id': 'abcdef02', 'reason': 'r',
                'content': {'old': 'a', 'new': 'b'}}]}}]}
    s = _harness().score_arm(log, set(), {})
    assert (s['revises'], s['revises_refused'], s['revises_landed']) == (2, 1, 1)
