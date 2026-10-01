"""Invalid brain_batch op → retry next cycle, never a silent SKIP-suppression.

When an S2 encoder emits a concept-verb (`absorb` / `reject` / `keep` / `skip`)
as a brain_batch op NAME instead of translating it into one of the 5 real ops,
dispatch drops the op and logs `brain_batch_invalid_op`. The encoder *tried* to
act and was thwarted — so the proposal/cluster must be RETRIED next cycle, never
stamped as a clean SKIP rejection. Stamping would both abandon the intended
action (a merge / a drift-threshold raise) AND suppress the proposal until a
member node's `updated_at` changes — silent capability loss.

These tests lock the detection helper + the matcher/detector composition that
`community_encoder` and `consolidation` use to exclude failed attempts.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.contract import VALID_BATCH_OPS
from servers.scales.s2.rejection_table import (
    node_ids_touched_by_invalid_ops,
    node_ids_touched_by_valid_ops,
    had_rejected_batch_call,
    get_proposed_ids,
    match_proposals_to_actions,
)


def _batch(*ops):
    """Shape an action_details entry the way runner.py records a brain_batch call."""
    return [{'tool': 'brain_batch', 'input': {'operations': list(ops)}}]


class TestDetector:
    def test_invalid_op_node_id_collected(self):
        # `consolidate` is a concept-verb, NOT a valid brain_batch op — dispatch
        # drops it. Its node id is collected so the cluster is retried, not
        # stamped. (`absorb` used to live here; it is now a VALID op — see
        # test_valid_absorb_op_not_collected.)
        ad = _batch({'op': 'consolidate', 'node_id': 'aaaa1111', 'reason': 'merge'})
        assert node_ids_touched_by_invalid_ops(ad) == {'aaaa1111'}

    def test_valid_absorb_op_not_collected(self):
        # absorb shipped into VALID_BATCH_OPS — a successful absorb is a real
        # merge, never a thwarted attempt, so the detector must NOT flag it.
        ad = _batch({'op': 'absorb', 'survivor_id': 'aaaa1111',
                     'absorbed_id': 'bbbb2222', 'reason': 'merge'})
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_valid_ops_return_empty(self):
        ad = _batch(
            {'op': 'revise', 'node_id': 'aaaa1111'},
            {'op': 'archive', 'node_id': 'bbbb2222'},
        )
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_mixed_collects_only_invalid(self):
        ad = _batch(
            {'op': 'revise', 'node_id': 'good0001'},
            {'op': 'reject', 'node_id': 'bad00001', 'reason': 'drift'},
        )
        assert node_ids_touched_by_invalid_ops(ad) == {'bad00001'}

    def test_collects_edge_and_member_ids(self):
        ad = _batch(
            {'op': 'keep', 'source_id': 's1', 'target_id': 't1'},
            {'op': 'skip', 'members': ['m1', 'm2']},
        )
        assert node_ids_touched_by_invalid_ops(ad) == {'s1', 't1', 'm1', 'm2'}

    def test_ignores_non_brain_batch_tools(self):
        ad = [{'tool': 'get_nodes',
               'input': {'operations': [{'op': 'reject', 'node_id': 'x'}]}}]
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_tolerates_malformed_entries(self):
        ad = [
            'garbage',
            {'tool': 'brain_batch', 'input': {'operations': ['not-a-dict', {}]}},
            {'tool': 'brain_batch', 'input': None},
        ]
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_empty_and_none(self):
        assert node_ids_touched_by_invalid_ops([]) == set()
        assert node_ids_touched_by_invalid_ops(None) == set()


class TestDriftRejectIsRetriedNotSuppressed:
    """The exact case from the error log: `op: reject` on a drift proposal."""

    def _drift(self):
        return {'type': 'drift', 'node_id': 'ff18ae35',
                'foreign': [{'id': 'bc3780cd'}]}

    def test_matcher_alone_marks_invalid_reject_as_skipped(self):
        # The matcher only counts a drift-reject when op==revise carries
        # _sys_drift_threshold — so an `op: reject` lands in skipped. Without
        # the fix this is what gets a suppression fingerprint.
        p = self._drift()
        ad = _batch({'op': 'reject', 'node_id': 'ff18ae35',
                     'reason': 'DRIFT rejected'})
        acted, skipped = match_proposals_to_actions([p], ad)
        assert acted == []
        assert skipped == [p]

    def test_invalid_reject_node_is_pulled_out_of_suppression(self):
        # The fix: detector flags ff18ae35, so the encoder's exclusion filter
        # moves the proposal to retry instead of record_rejections.
        p = self._drift()
        ad = _batch({'op': 'reject', 'node_id': 'ff18ae35',
                     'reason': 'DRIFT rejected'})
        _, skipped = match_proposals_to_actions([p], ad)
        touched = node_ids_touched_by_invalid_ops(ad)
        retry = [q for q in skipped if set(get_proposed_ids(q)) & touched]
        to_suppress = [q for q in skipped
                       if not (set(get_proposed_ids(q)) & touched)]
        assert retry == [p]          # retried next cycle
        assert to_suppress == []     # nothing stamped → no silent abandonment

    def test_agent_authored_threshold_no_longer_counts_as_acting(self):
        # Contract CHANGED: the agent does not author _sys_drift_threshold any
        # more — declining is "take no action", and the encoder's algorithmic Δ
        # writes the number (id:e0b7f842). A legacy agent still emitting the
        # revise must land in SKIPPED, so the stamp overwrites its value; if
        # this still read as acted-on, the agent's string would survive — which
        # is how "1.6x" reached production. (This fixture's own '0.6' is a
        # sub-1.0 value the decoder now clamps.)
        p = self._drift()
        ad = _batch({'op': 'revise', 'node_id': 'ff18ae35',
                     '_sys_drift_threshold': '0.6'})
        acted, skipped = match_proposals_to_actions([p], ad)
        assert acted == []
        assert skipped == [p]
        # Still a VALID op — never routed to the invalid-op retry path.
        assert node_ids_touched_by_invalid_ops(ad) == set()

    def test_drift_accept_reads_as_acted_on(self):
        # The accept branch is what keeps the threshold stamp OFF a node the
        # agent deliberately moved. If this regresses, the accepted proposal
        # falls into skipped and _raise_drift_thresholds ratchets the node's
        # drift bar up for a decision that was a YES — silently penalising
        # correct judgment, with every other test still green.
        p = self._drift()
        ad = _batch({'op': 'connect', 'source_id': 'bc3780cd',
                     'target_id': 'ff18ae35', 'relation': 'community_member'})
        acted, skipped = match_proposals_to_actions([p], ad)
        assert acted == [p]
        assert skipped == []


class TestConsolidationAbsorbIsRealMerge:
    def test_valid_absorb_cluster_not_pulled_into_retry(self):
        # A consolidation cluster merged via the VALID `absorb` op is a real
        # merge — the absorbed node gets archived. It must NOT land in the
        # invalid-op retry path (that is only for concept-verbs dispatch drops).
        # The orchestrator detects the successful merge via the archived member,
        # not via this helper — see test_s2_consolidation.py.
        ad = _batch({'op': 'absorb', 'survivor_id': '7463a2aa',
                     'absorbed_id': 'c3571f66', 'content': 'merged'})
        members = ['7463a2aa', 'c3571f66']
        assert set(members) & node_ids_touched_by_invalid_ops(ad) == set()


class TestRejectedBatchCallDetector:
    """A call-level dispatch rejection (empty operations, a non-list, a
    string that doesn't parse) writes nothing and names no node ids —
    node_ids_touched_by_invalid_ops can't shield anything. The detector is
    the dispatch verdict: the runner records `error` on EVERY action whose
    handler returned ok=False, so the verdict is always present in real
    action_details and shape-guessing would only be less accurate."""

    def test_empty_operations_rejection_detected(self):
        ad = [{'tool': 'brain_batch', 'input': {'operations': []},
               'error': 'operations array is required'}]
        assert had_rejected_batch_call(ad) is True

    def test_error_verdict_detected_for_stringified_ops(self):
        # `operations: "[]"` is a truthy string (a shape check couldn't see
        # it) but dispatch rejects the call — the recorded error is the
        # authoritative signal.
        ad = [{'tool': 'brain_batch', 'input': {'operations': '[]'},
               'error': 'operations array is required — ...'}]
        assert had_rejected_batch_call(ad) is True

    def test_error_verdict_detected_for_dict_ops(self):
        ad = [{'tool': 'brain_batch',
               'input': {'operations': {'0': {'op': 'absorb'}}},
               'error': 'operations must be an array, got dict'}]
        assert had_rejected_batch_call(ad) is True

    def test_shape_alone_is_not_a_verdict(self):
        # Verdict-only by design: an entry with empty/missing operations but
        # no recorded error never occurs in runner output (error is set on
        # every ok=False action) — the detector must not guess from shape.
        assert had_rejected_batch_call(_batch()) is False
        assert had_rejected_batch_call(
            [{'tool': 'brain_batch', 'input': {}}]) is False

    def test_successful_call_not_flagged(self):
        ad = [{'tool': 'brain_batch',
               'input': {'operations': [{'op': 'connect', 'source_id': 'a',
                                         'target_id': 'b'}]},
               'error': None}]
        assert had_rejected_batch_call(ad) is False

    def test_ignores_other_tools_and_empty_input(self):
        assert had_rejected_batch_call([{'tool': 'get_nodes', 'input': {},
                                         'error': 'boom'}]) is False
        assert had_rejected_batch_call([]) is False
        assert had_rejected_batch_call(None) is False

    def test_tolerates_malformed_entries(self):
        assert had_rejected_batch_call(['garbage']) is False
        assert had_rejected_batch_call(
            [{'tool': 'brain_batch', 'input': None}]) is False


class TestValidOpAttribution:
    """node_ids_touched_by_valid_ops scopes the rejected-call shield: a
    cluster a real op touched was acted on and must not be pulled into
    retry when a stray rejected call appears in the same run."""

    def test_collects_ids_from_valid_ops(self):
        ad = _batch(
            {'op': 'absorb', 'survivor_id': 's1', 'absorbed_id': 'a1'},
            {'op': 'connect', 'source_id': 'x1', 'target_id': 'y1',
             'relation': 'similar_to'},
        )
        assert node_ids_touched_by_valid_ops(ad) == {'s1', 'a1', 'x1', 'y1'}

    def test_stringified_executed_ops_are_attributed(self):
        # Dispatch unwraps a parseable stringified array and EXECUTES it
        # (ok=True, no error) — attribution must unwrap the same way, or the
        # acted-on cluster is pulled back into retry by a stray rejected call.
        ad = [{'tool': 'brain_batch',
               'input': {'operations':
                         '[{"op": "absorb", "survivor_id": "s1", '
                         '"absorbed_id": "a1"}]'}}]
        assert node_ids_touched_by_valid_ops(ad) == {'s1', 'a1'}

    def test_invalid_ops_and_errored_calls_contribute_nothing(self):
        ad = _batch({'op': 'keep', 'source_id': 'x1', 'target_id': 'y1'})
        assert node_ids_touched_by_valid_ops(ad) == set()
        errored = [{'tool': 'brain_batch',
                    'input': {'operations': [{'op': 'archive', 'node_id': 'n1'}]},
                    'error': 'something rejected the whole call'}]
        assert node_ids_touched_by_valid_ops(errored) == set()

    def test_tolerates_malformed_entries(self):
        assert node_ids_touched_by_valid_ops(None) == set()
        assert node_ids_touched_by_valid_ops(
            [{'tool': 'brain_batch', 'input': None}]) == set()
        assert node_ids_touched_by_valid_ops(
            [{'tool': 'brain_batch', 'input': {'operations': '[]'}}]) == set()


class TestContractSync:
    def test_valid_batch_ops_is_the_closed_six(self):
        # brain_mcp's enum is built from this set and the dispatcher's if/elif
        # must match it. Lock the membership so a new op can't silently diverge.
        # absorb (lossless merge) joined the closed set — see S2-ABSORB-OP-DESIGN.md.
        assert set(VALID_BATCH_OPS) == {
            'remember', 'revise', 'connect', 'disconnect', 'archive', 'absorb'}
