"""Integration tests for trace write paths.

Simulates what hooks do, then verifies traces are correct.
All tests use brain_env() — a seeded brain, never the production DB.

Run: python3 -m pytest tests/test_trace_integration.py -v
"""
import json
import os
import time
import pytest
from servers.trace_contract import build_selection_metadata
from tests.isolated_brain import brain_env

# A K stamp for builder calls — the builders require one.
_STAMP = {'fingerprint': 'f0f0f0f0f0f0', 'source': 'default', 'version': 0, 'id': None}


class TestS0StopTraces:
    """Verify the Stop hook's S0 trace writes."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            # Clear traces for clean assertions
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def _simulate_stop(self, session_id, stop, user_msg, assistant_msg,
                       recall_log_id=None):
        """Simulate what hook_post_response_track writes for S0 traces."""
        chain = 's0-%s-%s' % (session_id[:8], stop)
        recall_chain = 's1r-%s-%s' % (session_id[:8], stop) if recall_log_id else ''

        self.dal.append(
            chain_id=chain, scale='s0', event_type='K',
            ref_type='user_message',
            summary=user_msg[:200],
            metadata={'content': user_msg, 'recall_chain': recall_chain},
            session_id=session_id)
        self.dal.append(
            chain_id=chain, scale='s0', event_type='delta',
            ref_type='assistant_message',
            summary=assistant_msg[:200],
            metadata={'content': assistant_msg},
            session_id=session_id)
        return chain

    def test_writes_k_and_delta(self):
        """Stop produces K (user_message) and delta (assistant_message) in same chain."""
        chain = self._simulate_stop('sess-a', '10', 'hello', 'hi there')
        events = self.dal.get_chain(chain)
        assert len(events) == 2
        assert events[0]['event_type'] == 'K'
        assert events[0]['ref_type'] == 'user_message'
        assert events[1]['event_type'] == 'delta'
        assert events[1]['ref_type'] == 'assistant_message'

    def test_summary_is_short(self):
        """Summary is truncated for display, full content in metadata."""
        long_msg = 'A' * 1000
        chain = self._simulate_stop('sess-a', '11', long_msg, 'reply')
        events = self.dal.get_chain(chain)
        assert len(events[0]['summary']) <= 200
        assert events[0]['metadata']['content'] == long_msg

    def test_recall_chain_cross_reference(self):
        """K metadata includes recall_chain linking to S1."""
        session_id = 'abcdef1234567890'
        chain = self._simulate_stop(session_id, '12', 'question?', 'answer',
                                    recall_log_id='9999')
        events = self.dal.get_chain(chain)
        k_event = events[0]
        expected = 's1r-%s-12' % session_id[:8]
        assert k_event['metadata']['recall_chain'] == expected


class TestS0ToolTraces:
    """Verify PostToolUse tool_result traces attach to correct chain."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_tool_shares_stop_chain(self):
        """Tool results share the same s0-{session}-{stop} chain as messages."""
        session_id = 'sess-tool-test'
        stop = '15'
        chain = 's0-%s-%s' % (session_id[:8], stop)

        # Simulate tool call (what PostToolUse writes)
        self.dal.append(
            chain_id=chain, scale='s0', event_type='delta',
            ref_type='tool_result',
            summary='Edit: /path/to/file.py',
            metadata={'tool': 'Edit'},
            session_id=session_id)

        # Simulate stop (what hook_post_response_track writes)
        self.dal.append(
            chain_id=chain, scale='s0', event_type='K',
            ref_type='user_message',
            summary='check the file',
            metadata={'content': 'check the file'},
            session_id=session_id)
        self.dal.append(
            chain_id=chain, scale='s0', event_type='delta',
            ref_type='assistant_message',
            summary='file looks good',
            metadata={'content': 'file looks good'},
            session_id=session_id)

        events = self.dal.get_chain(chain)
        assert len(events) == 3
        ref_types = [e['ref_type'] for e in events]
        assert 'tool_result' in ref_types
        assert 'user_message' in ref_types
        assert 'assistant_message' in ref_types


class TestS1RecallTraces:
    """Verify the recall hook's S1 trace writes."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def _simulate_recall(self, session_id, stop, query, candidates, selected,
                         additional_context):
        """Simulate what hook_recall writes for S1 traces."""
        chain = 's1r-%s-%s' % (session_id[:8], stop)

        cand_detail = ['%s|%s|%.2f|%s' % (c['id'], c['title'], c['score'], c['type'])
                        for c in candidates]
        sel_detail = ['%s|%s' % (s['id'], s['title']) for s in selected]

        self.dal.append(
            chain_id=chain, scale='s1', event_type='O',
            ref_type='recall', ref_id=str(stop),
            summary='%d candidates for: %s' % (len(candidates), query[:100]),
            metadata={'source': 'hook', 'query': query, 'candidates': cand_detail},
            session_id=session_id)
        self.dal.append(
            chain_id=chain, scale='s1', event_type='K',
            ref_type='surface_selected',
            ref_id=json.dumps([s['id'] for s in selected]),
            summary='%d selected' % len(selected),
            metadata={'selected': sel_detail, 'expanded': []},
            session_id=session_id)
        self.dal.append(
            chain_id=chain, scale='s1', event_type='delta',
            ref_type='additionalContext',
            summary='%d nodes surfaced' % len(selected),
            metadata=build_selection_metadata(interaction=_STAMP, content=additional_context),
            session_id=session_id)
        return chain

    def test_writes_okd(self):
        """Recall produces O, K, delta in s1r chain."""
        candidates = [{'id': 'abc', 'title': 'Node A', 'score': 0.85, 'type': 'lesson'},
                      {'id': 'def', 'title': 'Node B', 'score': 0.72, 'type': 'decision'}]
        selected = [{'id': 'abc', 'title': 'Node A'}]

        chain = self._simulate_recall('sess-r', '20', 'test query', candidates,
                                      selected, 'Brain recalled: Node A')
        events = self.dal.get_chain(chain)
        assert len(events) == 3
        assert events[0]['event_type'] == 'O'
        assert events[1]['event_type'] == 'K'
        assert events[2]['event_type'] == 'delta'

    def test_o_has_candidates_in_metadata(self):
        """O event metadata contains full candidate list."""
        candidates = [{'id': 'abc', 'title': 'Node A', 'score': 0.85, 'type': 'lesson'}]
        chain = self._simulate_recall('sess-r', '21', 'test', candidates, [], '')
        events = self.dal.get_chain(chain)
        o_event = events[0]
        assert 'candidates' in o_event['metadata']
        assert len(o_event['metadata']['candidates']) == 1
        assert 'abc' in o_event['metadata']['candidates'][0]

    def test_k_has_selected_in_metadata(self):
        """K event metadata contains selected node details."""
        selected = [{'id': 'abc', 'title': 'Node A'}, {'id': 'def', 'title': 'Node B'}]
        chain = self._simulate_recall('sess-r', '22', 'test',
                                      [{'id': 'abc', 'title': 'A', 'score': 0.8, 'type': 'x'}],
                                      selected, 'context')
        events = self.dal.get_chain(chain)
        k_event = events[1]
        assert 'selected' in k_event['metadata']
        assert len(k_event['metadata']['selected']) == 2

    def test_delta_has_content_in_metadata(self):
        """Delta event metadata contains full additionalContext."""
        context = 'Brain recalled 3 memories:\n[lesson] "Test" (id:abc)...'
        chain = self._simulate_recall('sess-r', '23', 'test',
                                      [{'id': 'abc', 'title': 'T', 'score': 0.9, 'type': 'lesson'}],
                                      [{'id': 'abc', 'title': 'T'}], context)
        events = self.dal.get_chain(chain)
        delta = events[2]
        assert delta['metadata']['content'] == context

    def test_source_in_metadata(self):
        """O event metadata includes source='hook'."""
        chain = self._simulate_recall('sess-r', '24', 'test',
                                      [{'id': 'a', 'title': 'T', 'score': 0.8, 'type': 'x'}],
                                      [], '')
        events = self.dal.get_chain(chain)
        assert events[0]['metadata'].get('source') == 'hook'


class TestS1EncodeTraces:
    """Verify the encoding agent's S1 trace writes."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def _simulate_encode(self, session_id, counter, turn_count, chars,
                         node_ids, actions, final_text):
        """Simulate what encoding_agent + daemon_hooks write for S1 encode traces."""
        chain = 's1e-%s-%d' % (session_id[:8], counter)

        # O: from encode.py — ref_id is the db_dir-relative payload pointer
        # (TRACE-MODES step 2 flipped the semantics from an absolute tmp path).
        self.dal.append(
            chain_id=chain, scale='s1', event_type='O',
            ref_type='encoding_prompt',
            ref_id='payloads/2026-08-03/%s/000-prompt.md' % chain,
            summary='%d turns, %d chars context' % (turn_count, chars),
            session_id=session_id)
        # K: from encoding_agent.py
        self.dal.append(
            chain_id=chain, scale='s1', event_type='K',
            ref_type='node_catalog',
            ref_id=','.join(node_ids[:20]),
            summary='%d unique nodes in catalog' % len(node_ids),
            session_id=session_id)
        # delta: from daemon_hooks.py
        action_lines = ['%s: %s' % (a['tool'], a['summary']) for a in actions]
        self.dal.append(
            chain_id=chain, scale='s1', event_type='delta',
            ref_type='encoding_run', ref_id=str(counter),
            summary='%d actions:\n%s\n---\n%s' % (
                len(actions), '\n'.join(action_lines), final_text[:2000]),
            session_id=session_id)
        return chain

    def test_writes_okd(self):
        """Encode produces O, K, delta in s1e chain."""
        chain = self._simulate_encode(
            'sess-e', 100, 5, 20000, ['abc', 'def'],
            [{'tool': 'remember_batch', 'summary': 'New insight'}],
            'Good — 1 node created.')
        events = self.dal.get_chain(chain)
        assert len(events) == 3
        assert events[0]['event_type'] == 'O'
        assert events[1]['event_type'] == 'K'
        assert events[2]['event_type'] == 'delta'

    def test_delta_includes_actions_and_reasoning(self):
        """Delta summary includes specific actions AND final_text reasoning."""
        chain = self._simulate_encode(
            'sess-e', 101, 5, 15000, ['abc'],
            [{'tool': 'remember_batch', 'summary': 'Architecture decision'},
             {'tool': 'revise', 'summary': 'Updated stale node'}],
            'Two changes: one new architecture node, one revision of outdated info.')
        events = self.dal.get_chain(chain)
        delta = events[2]
        assert 'remember_batch: Architecture decision' in delta['summary']
        assert 'revise: Updated stale node' in delta['summary']
        assert 'Two changes' in delta['summary']

    def test_o_references_prompt_payload(self):
        """O event ref_id is the prompt payload's db_dir-relative pointer."""
        chain = self._simulate_encode('sess-e', 102, 3, 10000, [], [], '')
        events = self.dal.get_chain(chain)
        ref = events[0]['ref_id']
        assert ref.startswith('payloads/'), ref
        assert not os.path.isabs(ref)
        assert ref.endswith('-prompt.md'), ref


# TestS0S1CrossReference — REMOVED (redundant). Its sole test
# (test_get_session_turns_cross_references) asserted the identical behavior as
# test_trace_system.py::TestGetSessionTurns::test_cross_reference_surface_output:
# get_session_turns resolving surface_output from the S1 delta via recall_chain.
# That assertion is kept in trace_system, where it completes the cohesive 5-facet
# get_session_turns unit suite (shape / chronological / cross-reference / content /
# limit). The "integration" framing here was thin — both do DAL-append-then-read,
# not a more faithful hook path — so the lone single-test class is dropped.
