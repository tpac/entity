"""BATCH_OP_SPECS contract — the brain_batch discriminated-op single source.

Three sites derive from contract.BATCH_OP_SPECS and must not drift:
  1. VALID_BATCH_OPS (membership checks in dispatch + S2 rejection table)
  2. brain_mcp's brain_batch inputSchema (oneOf branch per op — generation-
     time signal; probe-validated 2026-06-12, eval/mcp_variants/probe_v2_*)
  3. dispatch_write._handle_brain_batch's per-op required pre-check
     (dispatch-time backstop)

These tests pin each derivation plus the revise reason/reasoning
disambiguation surviving through the batch path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.contract import BATCH_OP_SPECS, VALID_BATCH_OPS
from tests.brain_test_base import BrainTestBase


def test_valid_ops_derive_from_specs():
    assert VALID_BATCH_OPS == frozenset(BATCH_OP_SPECS)


_JSON_TYPES = {'string', 'number', 'boolean', 'array', 'object'}


def test_mcp_schema_derives_from_specs():
    """One oneOf branch per op, in dict order, with const discriminator and
    required = ['op'] + spec's required list."""
    from servers import brain_mcp
    tool = next(t for t in brain_mcp.TOOLS if t['name'] == 'brain_batch')
    items = tool['inputSchema']['properties']['operations']['items']
    branches = items['oneOf']
    assert [b['properties']['op']['const'] for b in branches] == \
        list(BATCH_OP_SPECS)
    for branch, (op, spec) in zip(branches, BATCH_OP_SPECS.items()):
        assert branch['required'] == ['op'] + spec['required'], op
        # Every contract property fragment is present in the branch
        for prop in spec['properties']:
            assert prop in branch['properties'], '%s.%s' % (op, prop)
        # additionalProperties stays open — remember/revise/absorb accept
        # open-ended node fields by design. A branch may describe what those
        # fields take (revise shows the swap shapes), but never narrow them.
        extra = branch.get('additionalProperties')
        if extra is not None:
            admitted = {alt.get('type') for alt in extra['anyOf']}
            assert admitted >= _JSON_TYPES, (op, _JSON_TYPES - admitted)


def test_absorb_fields_are_built_from_revises():
    """absorb applies its fields through revise(), so its schema is revise's:
    the same `content` shape, the same `connect_to` items and the same
    open-field schema."""
    absorb, revise = BATCH_OP_SPECS['absorb'], BATCH_OP_SPECS['revise']
    assert absorb['additionalProperties'] == revise['additionalProperties']
    assert (absorb['properties']['connect_to']['items']
            == revise['properties']['connect_to']['items'])

    def shape(spec):
        return {k: v for k, v in spec['properties']['content'].items()
                if k != 'description'}
    assert shape(absorb) == shape(revise)


def test_specs_required_fields_are_declared_properties():
    """Every required field must have a property fragment (the schema can't
    require a field it doesn't describe)."""
    for op, spec in BATCH_OP_SPECS.items():
        for field in spec['required']:
            assert field in spec['properties'], '%s.%s' % (op, field)


class TestDispatchEnforcesRequired(BrainTestBase):
    needs_embedder = False

    # Minimal valid op payloads (ids are fake — the pre-check fires before
    # any handler touches the DB, so broken variants never write).
    VALID = {
        'remember': {'type': 'concept', 'title': 'T', 'content': 'C'},
        'revise': {'node_id': 'aaaabbbb', 'reason': 'r'},
        'connect': {'source_id': 'aaaabbbb', 'target_id': 'ccccdddd',
                    'relation': 'extends'},
        'disconnect': {'source_id': 'aaaabbbb', 'target_id': 'ccccdddd',
                       'relation': 'extends'},
        'archive': {'node_id': 'aaaabbbb'},
        'absorb': {'survivor_id': 'aaaabbbb', 'absorbed_id': 'ccccdddd'},
    }

    def test_each_missing_required_field_errors(self):
        from servers.daemon_dispatch import _handle_brain_batch

        for op, spec in BATCH_OP_SPECS.items():
            for field in spec['required']:
                payload = dict(self.VALID[op])
                payload.pop(field)
                r = _handle_brain_batch(
                    self.brain, {'operations': [{'op': op, **payload}]}, [])
                op_result = r['result']['results'][0]
                self.assertFalse(
                    op_result['ok'],
                    '%s without %s must fail: %r' % (op, field, op_result))
                self.assertIn(
                    field, op_result['error'],
                    '%s-missing error must name the field: %r'
                    % (field, op_result['error']))

    def test_revise_missing_reason_keeps_disambiguation(self):
        """The pre-check must not flatten the rich reason/reasoning error
        shipped 2026-06-12 (tests/test_revise_unified.py Class H)."""
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(self.brain, {'operations': [
            {'op': 'revise', 'node_id': 'aaaabbbb',
             'reasoning': 'meant as audit', 'content': 'x'},
        ]}, [])
        err = r['result']['results'][0]['error']
        self.assertIn('reason is required', err)
        self.assertIn('reasoning', err)
        self.assertIn('FIELD', err)

    def test_revise_missing_reason_and_node_id_keeps_disambiguation(self):
        """`reason` among MULTIPLE missing fields still gets the rich error,
        with the other missing fields appended (review 2026-06-12 #2)."""
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(self.brain, {'operations': [
            {'op': 'revise', 'reasoning': 'meant as audit', 'content': 'x'},
        ]}, [])
        err = r['result']['results'][0]['error']
        self.assertIn('reason is required', err)
        self.assertIn('reasoning', err)
        self.assertIn('Also missing: node_id', err)

    def test_unknown_op_still_hits_invalid_op_guard(self):
        """Ops outside BATCH_OP_SPECS bypass the pre-check and land in the
        loud invalid-op branch (S2 rejection table depends on that error)."""
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(
            self.brain, {'operations': [{'op': 'consolidate'}]}, [])
        err = r['result']['results'][0]['error']
        self.assertIn('Unknown op', err)


class TestStringifiedOperationsUnwrap(BrainTestBase):
    """Lossless unwrap of JSON-string-serialized operations (2026-08-10/11
    incidents: a stringified array iterated char-by-char fanned out 27k+
    per-op errors and a 3.3MB tool result). Recoverable strings proceed;
    unrecoverable ones collapse to ONE whole-call error — never a
    per-character fan-out."""
    needs_embedder = False

    def test_string_array_unwraps_and_dispatches(self):
        """A JSON-encoded ops array behaves exactly like the array itself —
        the op reaches per-op validation (field error, not type error)."""
        import json
        from servers.daemon_dispatch import _handle_brain_batch

        ops = [{'op': 'revise', 'node_id': 'aaaabbbb',
                'reasoning': 'meant as audit', 'content': 'x'}]
        r = _handle_brain_batch(
            self.brain, {'operations': json.dumps(ops, indent=2)}, [])
        self.assertTrue(r['ok'])
        self.assertEqual(len(r['result']['results']), 1)
        # Proof the unwrap happened: the revise pre-check fired (rich
        # reason/reasoning error), not "operation must be a dict".
        self.assertIn('reason is required', r['result']['results'][0]['error'])

    def test_unparseable_string_fails_whole_call_once(self):
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(
            self.brain, {'operations': 'not json at all ' * 2000}, [])
        self.assertFalse(r['ok'])
        self.assertIn('JSON array', r['error'])
        self.assertNotIn('results', r)  # one error, zero fan-out

    def test_string_parsing_to_non_list_fails_whole_call_once(self):
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(
            self.brain, {'operations': '{"op": "archive"}'}, [])
        self.assertFalse(r['ok'])
        self.assertIn('JSON array', r['error'])

    def test_non_list_non_string_fails_whole_call_once(self):
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(
            self.brain, {'operations': {'op': 'archive'}}, [])
        self.assertFalse(r['ok'])
        self.assertIn('must be an array, got dict', r['error'])

    def test_string_element_parsing_to_dict_unwraps(self):
        """A string ELEMENT that parses to a dict is the intended op
        (S2 community precedent)."""
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(self.brain, {'operations': [
            '{"op": "archive"}',
        ]}, [])
        op_result = r['result']['results'][0]
        # Reached per-op validation as an archive op missing node_id —
        # not the non-dict type error.
        self.assertIn('node_id', op_result['error'])
        self.assertNotIn('must be a dict', op_result['error'])

    def test_unparseable_string_element_keeps_per_op_error(self):
        from servers.daemon_dispatch import _handle_brain_batch

        r = _handle_brain_batch(self.brain, {'operations': [
            'garbage', {'op': 'consolidate'},
        ]}, [])
        results = r['result']['results']
        self.assertIn('must be a dict, got str', results[0]['error'])
        self.assertIn('Unknown op', results[1]['error'])  # sibling still ran


class TestCommunityCreateNeedsMembers(BrainTestBase):
    """A community is its community_member edges, so every create path
    refuses a community whose connect_to names no member — before anything is
    written, so the writer can re-send it whole."""
    needs_embedder = False

    def _member(self, title):
        return self.brain.remember(type='finding', title=title, content=title)['id']

    def _communities(self):
        return self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE type = 'community'").fetchone()[0]

    def _op(self, connect_to):
        return {'op': 'remember', 'type': 'community', 'title': 'Story',
                'content': 'A story', 'connect_to': connect_to}

    def test_brain_batch_refuses_only_the_memberless_community(self):
        from servers.daemon_dispatch import _handle_brain_batch
        member = self._member('Member')
        r = _handle_brain_batch(self.brain, {'operations': [
            self._op(None),
            self._op([{'target': member, 'relation': 'extends'}]),
            dict(self._op([{'target': member, 'relation': 'community_member'}]),
                 title='Story with members'),
            {'op': 'remember', 'type': 'concept', 'title': 'T', 'content': 'C'},
        ]}, [])
        results = r['result']['results']
        for refused in results[:2]:
            self.assertFalse(refused['ok'], refused)
            self.assertIn('community_member', refused['error'])
        self.assertTrue(results[2]['ok'], results[2])
        self.assertTrue(results[3]['ok'], results[3])
        self.assertEqual(self._communities(), 1)

    def test_member_named_inside_a_relations_list_counts(self):
        from servers.daemon_dispatch import _handle_brain_batch
        member = self._member('Member')
        r = _handle_brain_batch(self.brain, {'operations': [self._op([
            {'target': member, 'relations': [
                {'relation': 'extends', 'description': 'd'},
                {'relation': 'community_member', 'description': 'd'}]}])]}, [])
        self.assertTrue(r['result']['results'][0]['ok'], r)

    def test_the_check_reads_the_fields_envelope_like_the_door(self):
        from servers.scales.s2.community_contract import community_create_refusal
        member = self._member('Member')
        members = [{'target': member, 'relation': 'community_member'}]
        self.assertIsNone(community_create_refusal(
            {'type': 'community', 'title': 'T', 'fields': {'connect_to': members}}))
        self.assertIsNotNone(community_create_refusal(
            {'title': 'T', 'fields': {'type': 'community'}}))

    def test_remember_command_refuses_a_memberless_community(self):
        from servers.daemon_dispatch import COMMAND_TABLE
        op = self._op(None)
        op.pop('op')
        r = COMMAND_TABLE['remember'].handler(self.brain, op, [])
        self.assertFalse(r['ok'], r)
        self.assertIn('community_member', r['error'])
        self.assertEqual(self._communities(), 0)

    def test_remember_batch_refuses_before_writing_any_node(self):
        from servers.daemon_dispatch import COMMAND_TABLE
        community = self._op(None)
        community.pop('op')
        r = COMMAND_TABLE['remember_batch'].handler(self.brain, {'nodes': [
            {'type': 'concept', 'title': 'T', 'content': 'C'}, community]}, [])
        self.assertFalse(r['ok'], r)
        self.assertIn('nodes[1]', r['error'])
        self.assertEqual(self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE title = 'T'").fetchone()[0], 0)

    def test_remember_batch_members_named_batch_wide_count(self):
        # The batch-level connect_to is written from every created node, so
        # it gives each community created its members.
        from servers.daemon_dispatch import COMMAND_TABLE
        member = self._member('Member')
        community = self._op(None)
        community.pop('op')
        r = COMMAND_TABLE['remember_batch'].handler(self.brain, {
            'nodes': [community],
            'connect_to': [{'target': member, 'relation': 'community_member'}]}, [])
        self.assertTrue(r['ok'], r)
        self.assertEqual(self._communities(), 1)
