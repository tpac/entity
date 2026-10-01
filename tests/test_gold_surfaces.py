"""score_gold's `surfaces_required` criterion — the field-coverage instrument.

The defect it exists to measure (id:450650d5): the encoder repairs a stale
value in `title` and `content` while the SAME value survives in `situation`
and in an edge description. Every text-only check scores that as a pass,
because the node WAS revised and the op text DOES carry the fresh fact — so
before this criterion, the candidate and the incumbent both passed and the
A/B delta read zero.

The v41 ops below are the real ones, copied from trace 703e70a5 (encode run
80f21f0d, 2026-08-31, interaction_fingerprint 3817564d21c4 = production).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eval'))

from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

from eval.encoder_prompt_ab import score_gold  # noqa: E402

CORR = {'corrects', 'supersedes', 'reframes', 'resolves'}

# The canonical item: 9.6.0 lived in FOUR surfaces; the run reached two.
GOLD = {
    'revise_or_correct': [{
        'id': 'd827d22f',
        'note': 'plugin manifests moved 9.6.0 -> 9.7.2',
        'stale_token': '9.6.0',
        'surfaces_required': ['title', 'content', 'situation', 'edge:15bbfd64'],
    }],
    'invalid_if_read': ['d827d22f', '15bbfd64'],
}


def _log(ops, reads=()):
    return {'writes': [{'args': {'operations': ops}}],
            'reads': [{'args': r} for r in reads]}


V41_OPS = [{
    'op': 'revise', 'node_id': 'd827d22f',
    'reason': 'plugin.json version field is now 9.7.2, not 9.6.0; title carries '
              'a stale version number that embeds and ranks against live queries',
    'title': "D-6 and D-10 decided but never applied to files — plugin.json "
             "name still 'brain', version stale (9.7.2, not yet 0.9.0)",
    'content_edits': [{'old': '- `D-10` (v0.9.0): both manifests still say `9.6.0`.',
                       'new': '- `D-10` (v0.9.0): both manifests still say `9.7.2` '
                              '(was `9.6.0` — bumped but not to the locked 0.9.0).'}],
}]


def test_v41_scores_two_of_four_and_fails():
    """The recorded production run: title+content only."""
    g = score_gold(GOLD, _log(V41_OPS), CORR)
    t = g['targets'][0]
    assert t['via'] == 'revise', t          # it DID revise the node
    assert t['surface_score'] == '2/4', t['surfaces']
    assert t['surfaces'] == {'title': True, 'content': True,
                             'situation': False, 'edge:15bbfd64': False}
    assert not t['pass']
    assert not g['pass']


def test_without_the_criterion_the_same_run_passes():
    """Why the criterion had to exist: strip it and the defect is invisible."""
    blind = {'revise_or_correct': [{'id': 'd827d22f',
                                    'content_any': ['9.7.2']}]}
    assert score_gold(blind, _log(V41_OPS), CORR)['pass']


def test_full_repair_scores_four_of_four():
    ops = [dict(V41_OPS[0], situation='…both manifests read 9.7.2 as of 2026-08-31…'),
           {'op': 'connect', 'source_id': 'd827d22f', 'target_id': '15bbfd64',
            'relation': 'gaps_in', 'description': '…manifests read 9.7.2…'}]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surface_score'] == '4/4', t['surfaces']
    assert t['pass']


def test_edge_surface_is_source_aware():
    """An edge from a DIFFERENT node onto the same target is not this node's
    repair — counting it would score someone else's edge as ours."""
    ops = [dict(V41_OPS[0], situation='fresh'),
           {'op': 'connect', 'source_id': 'aaaaaaaa', 'target_id': '15bbfd64',
            'relation': 'gaps_in', 'description': 'unrelated'}]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surfaces']['edge:15bbfd64'] is False
    assert t['surface_score'] == '3/4'


def test_content_counts_whole_or_patched():
    for key, val in (('content', 'rewritten whole'),
                     ('content_edits', [{'old': 'a', 'new': 'b'}])):
        ops = [{'op': 'revise', 'node_id': 'd827d22f', key: val,
                'title': 't', 'situation': 's'},
               {'op': 'connect', 'source_id': 'd827d22f',
                'target_id': '15bbfd64', 'relation': 'gaps_in'}]
        t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
        assert t['surfaces']['content'] is True, key


def test_connect_to_on_a_revise_counts_as_the_edge():
    ops = [dict(V41_OPS[0], situation='fresh',
                connect_to=[{'title': '15bbfd64', 'relation': 'gaps_in',
                             'why': 'x' * 40}])]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surfaces']['edge:15bbfd64'] is True


def test_reading_either_endpoint_voids_the_run():
    """Both endpoints moved after the capture — d827d22f was repaired and the
    edge description with it — so a live read of either leaks post-capture
    state into a run the frozen payload is supposed to control."""
    for rid in ('d827d22f', '15bbfd64'):
        g = score_gold(GOLD, _log(V41_OPS, reads=[{'node_ids': [rid]}]), CORR)
        assert g['invalid_reads'] == [rid], rid
        assert not g['pass']


def test_rewriting_a_surface_that_still_asserts_the_stale_value_is_not_repair():
    """Writing a surface is not fixing it. Caught in review: without this the
    instrument scores its own defect as clean."""
    ops = [{'op': 'revise', 'node_id': 'd827d22f', 'title': 'now 9.7.2',
            'content': 'now 9.7.2',
            'situation': '9.6.0 is still the version in both manifests'},
           {'op': 'connect', 'source_id': 'd827d22f', 'target_id': '15bbfd64',
            'relation': 'gaps_in', 'description': 'both manifests still say 9.6.0'}]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surfaces']['situation'] is False, t['surfaces']
    assert t['surfaces']['edge:15bbfd64'] is False, t['surfaces']
    assert t['surface_score'] == '2/4'
    assert not t['pass']


def test_a_disconnect_on_the_pair_is_not_an_edge_assertion():
    """A disconnect carries source_id+target_id but asserts nothing. Counted
    as an edge it entered the pair with an empty why — and an empty why never
    carries the stale token, so an UNREPAIRED edge scored as repaired."""
    ops = [{'op': 'revise', 'node_id': 'd827d22f', 'title': 'now 9.7.2',
            'content': 'now 9.7.2', 'situation': 'now 9.7.2',
            'connect_to': [{'target': '15bbfd64', 'relation': 'gaps_in',
                            'why': 'both manifests still say 9.6.0'}]},
           {'op': 'disconnect', 'source_id': 'd827d22f', 'target_id': '15bbfd64',
            'relation': 'noise'}]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surfaces']['edge:15bbfd64'] is False, t['surfaces']
    # and on its own it writes no edge surface at all
    alone = [{'op': 'revise', 'node_id': 'd827d22f', 'title': 't', 'content': 'c',
              'situation': 's'},
             {'op': 'disconnect', 'source_id': 'd827d22f', 'target_id': '15bbfd64',
              'relation': 'gaps_in'}]
    t = score_gold(GOLD, _log(alone), CORR)['targets'][0]
    assert t['surfaces']['edge:15bbfd64'] is False and t['surface_score'] == '3/4'


def test_content_may_keep_the_old_value_as_history():
    """E17: history rides in `content` and only there — a patch writing
    '9.7.2 (was 9.6.0)' is correct and must not be scored as unrepaired."""
    ops = [{'op': 'revise', 'node_id': 'd827d22f', 'title': 'now 9.7.2',
            'content_edits': [{'old': 'x', 'new': '9.7.2 (was 9.6.0 — bumped)'}],
            'situation': 'both manifests read 9.7.2'},
           {'op': 'connect', 'source_id': 'd827d22f', 'target_id': '15bbfd64',
            'relation': 'gaps_in', 'description': 'manifests read 9.7.2'}]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surfaces']['content'] is True, t['surfaces']
    assert t['surface_score'] == '4/4'
    assert t['pass']


def test_swapped_surfaces_score_on_the_new_text():
    """Value-or-swap (REVISE_RULE): a swap whose `old` carries the stale token
    and whose `new` does not is a repair on that surface — the removed text
    must never be read as still asserted. Edge included: `connect_to` on the
    revise, keyed `target`, its `why` a swap."""
    ops = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
            'title': {'old': 'version stale (9.6.0)', 'new': 'version 9.7.2, short of 0.9.0'},
            'content': [{'old': 'say `9.6.0`', 'new': 'say `9.7.2`'}],
            'situation': [{'old': '9.6.0', 'new': '9.7.2'}],
            'connect_to': [{'target': '15bbfd64', 'relation': 'gaps_in',
                            'why': {'old': 'both manifests still say 9.6.0',
                                    'new': 'manifests moved to 9.7.2 and still miss 0.9.0'}}]}]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surface_score'] == '4/4', t['surfaces']
    assert t['pass']


def test_swap_whose_new_text_still_asserts_the_stale_value_is_not_repair():
    ops = [{'op': 'revise', 'node_id': 'd827d22f', 'reason': 'r',
            'title': 'now 9.7.2', 'content': 'now 9.7.2',
            'situation': {'old': 'x', 'new': 'still 9.6.0 in both manifests'},
            'connect_to': [{'target': '15bbfd64', 'relation': 'gaps_in',
                            'why': {'old': 'a', 'new': 'still 9.6.0'}}]}]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surfaces'] == {'title': True, 'content': True,
                             'situation': False, 'edge:15bbfd64': False}


def test_connect_to_target_key_counts_like_the_title_alias():
    ops = [dict(V41_OPS[0], situation='fresh',
                connect_to=[{'target': '15bbfd64', 'relation': 'gaps_in',
                             'why': 'x' * 40}])]
    t = score_gold(GOLD, _log(ops), CORR)['targets'][0]
    assert t['surfaces']['edge:15bbfd64'] is True


def test_specs_without_surfaces_required_are_unchanged():
    """Back-compat: the run-44 spec and every existing item keep their
    semantics — surfaces is None, not an empty pass."""
    spec = {'revise_or_correct': [{'id': 'd827d22f', 'content_any': ['9.7.2']}]}
    t = score_gold(spec, _log(V41_OPS), CORR)['targets'][0]
    assert t['surfaces'] is None and t['surface_score'] is None
    assert t['pass']


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('  PASS  %s' % name)
            except AssertionError as e:
                fails += 1
                print('  FAIL  %s — %s' % (name, e))
    print('\n%s' % ('ALL PASS' if not fails else '%d FAILED' % fails))
    sys.exit(1 if fails else 0)


def test_a_refused_revise_is_a_failed_attempt_not_a_revise():
    """The harness answers a would-be-refused revise as production does; gold
    must not credit it — and must say it was tried, not skipped."""
    log = _log(V41_OPS)
    log['writes'][0]['refused'] = [0]
    t = score_gold(GOLD, log, CORR)['targets'][0]
    assert t['via'] is None and t['pass'] is False
    assert t['refused'] is True
    assert score_gold(GOLD, _log(V41_OPS), CORR)['targets'][0]['refused'] is False
