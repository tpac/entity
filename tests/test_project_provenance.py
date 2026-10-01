"""Project provenance — deterministic, session-derived, never agent-authored.

The contract (2026-07-03): `project` on a node is PROVENANCE — the repo the
session was working in when the node was learned (SessionContext.project,
derived from cwd at boot). Two chokepoints enforce it:

  - encoder attribution (scales/s2/base.apply_encoder_attribution): the Scribe
    stamps its session's project; S2 units strip agent-supplied values
    (graph-scope work never invents provenance).
  - MCP write handlers (dispatch_write._stamp_session_scope): the ambient
    session's project is force-stamped on node-creating payloads; agent
    values on revise are dropped (a revise never moves provenance).

These tests pin the policy semantics and the two wirings.
"""
import pytest

from tests.brain_test_base import BrainTestBase
from servers.scales.dispatch import stamp_scope_provenance


class TestStampPolicy:
    """Pure-function semantics of stamp_scope_provenance (project field)."""

    def test_force_on_remember(self):
        args = {'title': 't', 'project': 'agent-invented'}
        warnings = stamp_scope_provenance('remember', args, {'project': 'brain'})
        assert args['project'] == 'brain'
        assert len(warnings) == 1          # override is surfaced, not silent

    def test_force_without_supplied_is_quiet(self):
        args = {'title': 't'}
        warnings = stamp_scope_provenance('remember', args, {'project': 'brain'})
        assert args['project'] == 'brain'
        assert warnings == []

    def test_strip_when_session_has_no_project(self):
        # '' is authoritative: non-repo session / S2 unit — agent value drops.
        args = {'title': 't', 'project': 'agent-invented'}
        warnings = stamp_scope_provenance('remember', args, {'project': ''})
        assert 'project' not in args
        assert len(warnings) == 1

    def test_none_is_no_op(self):
        # No session authority — an upstream chokepoint may already have
        # stamped (encoder path). Args pass through untouched.
        args = {'title': 't', 'project': 'encoder-stamped'}
        assert stamp_scope_provenance('remember', args, None) == []
        assert args['project'] == 'encoder-stamped'

    def test_revise_strips_even_with_project(self):
        # Revise never moves provenance — strip regardless of session project.
        args = {'node_id': 'n', 'reason': 'r', 'project': 'anything'}
        warnings = stamp_scope_provenance('revise', args, {'project': 'brain'})
        assert 'project' not in args
        assert len(warnings) == 1

    def test_empty_string_project_cannot_wipe_provenance(self):
        # REGRESSION (review finding): `project: ''` is falsy but present —
        # a truthy-only strip let it through to revise, where validate_field
        # accepts '' and the column write WIPED birth provenance. Strip must
        # be presence-based on every path.
        args = {'node_id': 'n', 'reason': 'r', 'project': ''}
        stamp_scope_provenance('revise', args, {'project': 'brain'})
        assert 'project' not in args
        # remember with no session project: explicit '' must also be popped
        args = {'title': 't', 'project': ''}
        stamp_scope_provenance('remember', args, {'project': ''})
        assert 'project' not in args

    def test_remember_batch_stamps_each_node(self):
        args = {'nodes': [{'title': 'a'}, {'title': 'b', 'project': 'x'}]}
        stamp_scope_provenance('remember_batch', args, {'project': 'brain'})
        assert [n['project'] for n in args['nodes']] == ['brain', 'brain']

    def test_brain_batch_force_on_remember_strip_elsewhere(self):
        args = {'operations': [
            {'op': 'remember', 'title': 'x'},
            {'op': 'revise', 'node_id': 'n', 'reason': 'r', 'project': 'bad'},
            {'op': 'absorb', 'survivor_id': 's', 'absorbed_id': 'a',
             'project': 'bad'},
        ]}
        stamp_scope_provenance('brain_batch', args, {'project': 'brain'})
        ops = args['operations']
        assert ops[0]['project'] == 'brain'
        assert 'project' not in ops[1]
        assert 'project' not in ops[2]

    def test_counterpart_stamps_through_same_machinery(self):
        # The stamp is field-generic over SCOPE_PROVENANCE_FIELDS — the
        # counterpart dimension force/strips exactly like project, with no
        # dimension-specific code path.
        args = {'title': 't', 'counterpart': 'agent-invented'}
        warnings = stamp_scope_provenance(
            'remember', args, {'project': 'brain', 'counterpart': 'Ada'})
        assert args['counterpart'] == 'Ada'
        assert args['project'] == 'brain'
        assert len(warnings) == 1

    def test_counterpart_strips_on_revise(self):
        args = {'node_id': 'n', 'reason': 'r', 'counterpart': 'anything'}
        stamp_scope_provenance('revise', args, {'counterpart': 'Ada'})
        assert 'counterpart' not in args

    def test_batch_branch_derives_from_op_contract(self):
        # force-vs-strip comes from BATCH_OP_SPECS' creates_node flag, not a
        # local op enumeration — pins the derivation so a future node-creating
        # op added via the contract path inherits the stamp automatically.
        from servers.contract import BATCH_OP_SPECS
        creating = {op for op, spec in BATCH_OP_SPECS.items()
                    if spec.get('creates_node')}
        assert creating == {'remember'}   # today; grows via the contract only


class TestUnitPolicies(BrainTestBase):
    """The two encoder-side policies: Scribe = session project, S2 = strip."""

    needs_embedder = False

    def test_s2_unit_default_policy_strips(self):
        from servers.scales.s2.base import IntegrationUnit
        unit = IntegrationUnit.__new__(IntegrationUnit)  # policy is state-free
        assert unit.scope_policy() == {'project': '', 'counterpart': ''}

    def test_scribe_policy_reads_session_project(self):
        from servers.scales.s1.scribe import S1Scribe
        sid = 'scribe-proj-sess'
        ctx = self.brain.get_or_create_session(sid)
        ctx.set_env(cwd='/tmp/x', project='brain')
        scribe = S1Scribe(self.brain, sid, counter=1)
        assert scribe.scope_policy()['project'] == 'brain'

    def test_encoder_attribution_carries_project(self):
        from servers.scales.s2.base import apply_encoder_attribution
        args = {'nodes': [{'type': 'lesson', 'title': 't', 'content': 'c'}]}
        warnings = apply_encoder_attribution(
            'remember_batch', args, encoding_source='encoder:sonnet',
            run_chain_id='s1e-x-1', scope={'project': 'brain'})
        assert warnings == []
        assert args['nodes'][0]['project'] == 'brain'
        assert args['encoding_source'] == 'encoder:sonnet'


class TestDispatchStamping(BrainTestBase):
    """MCP write boundary: the ambient session's project lands on the node."""

    needs_embedder = False

    def _session_with_project(self, sid, project):
        ctx = self.brain.get_or_create_session(sid)
        ctx.set_env(cwd='/tmp/repo', project=project)
        return ctx

    def test_remember_stamps_session_project(self):
        from servers.dispatch_write import _handle_remember
        self._session_with_project('proj-sess-1', 'brain')
        result = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'stamped node', 'content': 'c',
            '_caller_session': 'proj-sess-1',
        }, [])
        assert result['ok'], result
        node = self.brain.get_node(result['result']['id'])
        assert node['project'] == 'brain'

    def test_remember_overrides_agent_supplied(self):
        from servers.dispatch_write import _handle_remember
        self._session_with_project('proj-sess-2', 'brain')
        result = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'override node', 'content': 'c',
            'project': 'S1Scribe',                # the drift this kills
            '_caller_session': 'proj-sess-2',
        }, [])
        assert result['ok'], result
        node = self.brain.get_node(result['result']['id'])
        assert node['project'] == 'brain'
        assert any('provenance' in w for w in result['result'].get('warnings', []))

    def test_non_repo_session_gets_no_project(self):
        from servers.dispatch_write import _handle_remember
        self._session_with_project('proj-sess-3', '')
        result = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'chat node', 'content': 'c',
            'project': 'invented',
            '_caller_session': 'proj-sess-3',
        }, [])
        assert result['ok'], result
        node = self.brain.get_node(result['result']['id'])
        assert not node.get('project')

    def test_revise_cannot_move_provenance(self):
        from servers.dispatch_write import _handle_remember, _handle_revise
        self._session_with_project('proj-sess-4', 'brain')
        made = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'immovable', 'content': 'c',
            '_caller_session': 'proj-sess-4',
        }, [])
        nid = made['result']['id']
        rev = _handle_revise(self.brain, {
            'node_id': nid, 'reason': 'attempt provenance move',
            'project': 'elsewhere', 'content': 'c2',
            '_caller_session': 'proj-sess-4',
        }, [])
        assert rev['ok'], rev
        node = self.brain.get_node(nid)
        assert node['project'] == 'brain'      # unchanged
        assert node['content'] == 'c2'         # other fields still applied

    def test_brain_batch_absorb_cannot_move_provenance(self):
        # REGRESSION (review finding): MCP brain_batch never stamped, and
        # _op_absorb forwards non-control keys as survivor field overrides —
        # an agent-supplied project on an absorb op silently moved birth
        # provenance. The batch-boundary stamp closes it.
        from servers.dispatch_write import _handle_remember, _handle_brain_batch
        self._session_with_project('proj-sess-5', 'brain')
        a = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'survivor node', 'content': 'keep me',
            '_caller_session': 'proj-sess-5'}, [])['result']['id']
        b = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'absorbed node', 'content': 'fold me',
            '_caller_session': 'proj-sess-5'}, [])['result']['id']
        res = _handle_brain_batch(self.brain, {
            'operations': [{'op': 'absorb', 'survivor_id': a,
                            'absorbed_id': b, 'project': 'elsewhere'}],
            '_caller_session': 'proj-sess-5',
        }, [])
        assert res['ok'], res
        assert res['result']['succeeded'] == 1, res['result']
        node = self.brain.get_node(a)
        assert node['project'] == 'brain'       # provenance NOT moved
        assert any('provenance' in w for w in res['result'].get('warnings', []))

    def test_revise_batch_surfaces_stamp_warnings(self):
        # REGRESSION (review finding): revise_batch discarded the stamp's
        # warnings while the other three handlers surface them — the agent
        # never learned its project field was dropped.
        from servers.dispatch_write import _handle_remember, _handle_revise_batch
        self._session_with_project('proj-sess-6', 'brain')
        nid = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'batch revised', 'content': 'c',
            '_caller_session': 'proj-sess-6'}, [])['result']['id']
        res = _handle_revise_batch(self.brain, {
            'revisions': [{'node_id': nid, 'reason': 'r',
                           'project': 'elsewhere', 'content': 'c2'}],
            '_caller_session': 'proj-sess-6',
        }, [])
        assert res['ok'], res
        assert any('provenance' in w
                   for w in res['result'].get('warnings', []))
        assert self.brain.get_node(nid)['project'] == 'brain'

    def test_sessionless_caller_passes_through(self):
        # No ambient session (encoder path / direct handler call): the handler
        # must not touch a project an upstream chokepoint stamped.
        from servers.dispatch_write import _handle_remember
        result = _handle_remember(self.brain, {
            'type': 'lesson', 'title': 'upstream stamped', 'content': 'c',
            'project': 'brain', 'encoding_source': 'encoder:sonnet',
        }, [])
        assert result['ok'], result
        node = self.brain.get_node(result['result']['id'])
        assert node['project'] == 'brain'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
