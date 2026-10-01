"""S1 Scribe — converged in-process IntegrationUnit + the attribution chokepoint.

S1 Scribe now writes through the same `_make_encoder_dispatch` the S2 units use,
so its revise / edge traces carry its run chain (s1e-{session}-{stop}) instead of
falling to dispatch_write's '{scale}-{date}-revise' phantom chain. These pin the
attribution contract the legacy bg-thread + TCP path dropped, plus the s1e chain
format.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s2.base import apply_encoder_attribution, CHAIN_AWARE_WRITES
from servers.scales.dispatch import ATTRIBUTED_WRITE_COMMANDS
from tests.brain_test_base import BrainTestBase


# ── S1Scribe identity / chain ──

def test_scribe_chain_id_is_session_scoped():
    """The Scribe's run chain is s1e-{session_short}-{stop} — NOT S2's time
    format — sourced from SessionContext.s1e_chain, and cached/stable."""
    from servers.scales.s1.scribe import S1Scribe
    unit = S1Scribe(brain=None, session_id='abcd1234efgh5678', counter=5)
    assert unit.chain_id() == 's1e-abcd1234-5'
    assert unit.chain_id() == 's1e-abcd1234-5'   # cached → stable within a run


def test_scribe_identity():
    from servers.scales.s1.scribe import S1Scribe
    assert S1Scribe.SCALE == 's1'
    assert S1Scribe.ENCODING_SOURCE == 'encoder:sonnet'


# ── the attribution chokepoint (shared by S1 + S2) ──

WRITE_SAMPLES = {
    'brain_batch':    {'operations': []},
    'revise':         {'node_id': 'x', 'reason': 'r'},
    'revise_batch':   {'revisions': []},
    'connect':        {'source_id': 'a', 'target_id': 'b'},
    'connect_batch':  {'connections': []},
    'remember':       {'title': 't', 'content': 'c', 'type': 'fact'},
    'remember_batch': {'nodes': []},
    'revise_edge':    {'source_id': 'a', 'target_id': 'b', 'relation': 'r'},
}
# get_nodes / recall_batch are the encoder's reads — they must stay un-attributed.
READ_SAMPLES = {
    'get_nodes':    {'node_ids': ['x']},
    'recall_batch': {'queries': ['q']},
}


def test_every_write_carries_the_run_chain():
    for cmd, args in WRITE_SAMPLES.items():
        a = dict(args)
        apply_encoder_attribution(cmd, a, encoding_source='encoder:sonnet',
                                  run_chain_id='s1e-abcd1234-5')
        assert a.get('chain_id') == 's1e-abcd1234-5', cmd


def test_reads_carry_neither_chain_nor_source():
    for cmd, args in READ_SAMPLES.items():
        a = dict(args)
        apply_encoder_attribution(cmd, a, encoding_source='encoder:sonnet',
                                  run_chain_id='s1e-abcd1234-5')
        assert 'chain_id' not in a, '%s must not carry a chain' % cmd
        assert 'encoding_source' not in a, '%s must not be attributed' % cmd


def test_explicit_chain_id_is_preserved():
    # setdefault: a caller that set its own chain (e.g. a future direct path)
    # keeps it; the run chain only FILLS a missing one.
    a = {'operations': [], 'chain_id': 's2-20260101000000-consolidation'}
    apply_encoder_attribution('brain_batch', a, encoding_source='x',
                              run_chain_id='s1e-abcd1234-5')
    assert a['chain_id'] == 's2-20260101000000-consolidation'


def test_no_chain_when_run_chain_absent():
    # An empty run_chain_id (no run context) must not stamp a falsy chain that
    # would silently re-orphan the trace.
    a = {'operations': []}
    apply_encoder_attribution('brain_batch', a, encoding_source='s2:x',
                              run_chain_id='')
    assert 'chain_id' not in a
    assert a['encoding_source'] == 's2:x'   # attribution still applies


def test_non_dict_args_is_a_noop():
    apply_encoder_attribution('brain_batch', None,
                              encoding_source='x', run_chain_id='y')  # must not raise


def test_chain_aware_is_attributed_plus_revise_edge():
    # Single source: chain-awareness tracks attribution (+ the standalone edge
    # revise), so a new attributed write becomes chain-aware for free.
    assert CHAIN_AWARE_WRITES == ATTRIBUTED_WRITE_COMMANDS | {'revise_edge'}


class TestScribeInProcessAttribution(BrainTestBase):
    """End-to-end: a revise through the Scribe's in-process dispatch lands its
    node_revised trace on the s1e run chain — not dispatch_write's
    '{scale}-{date}-revise' fallback. Exercises the REAL path against an
    isolated brain: dispatch_command → _handle_brain_batch → mutations
    manifest → mutation_emitter. No Sonnet — just the chain routing the
    bg+TCP path used to drop."""

    needs_embedder = False

    def test_revise_lands_on_s1e_run_chain(self):
        from servers.scales.s1.scribe import S1Scribe

        node = self.brain.remember(type='fact', title='attr probe',
                                   content='before')
        nid = node['id']

        # dispatch_fn=None → S1Scribe builds the in-process _make_encoder_dispatch
        scribe = S1Scribe(self.brain, session_id='deadbeefcafe0000', counter=3)
        dispatch = scribe._make_encoder_dispatch()
        res = dispatch('brain_batch', {'operations': [
            {'op': 'revise', 'node_id': nid, 'reason': 'attribution probe',
             'content': 'after'}]})
        self.assertTrue(res.get('ok'), res)

        events = self.brain.query_traces(
            ref_type='node_revised', ref_id=nid, hours=24, limit=10
        ).get('events', [])
        self.assertTrue(events, 'no node_revised trace was emitted')
        chains = {e['chain_id'] for e in events}
        self.assertEqual(
            chains, {'s1e-deadbeef-3'},
            'revise must land on the s1e run chain, not a date fallback')
        self.assertTrue(all(e['scale'] == 's1' for e in events))


if __name__ == '__main__':
    unittest.main()
