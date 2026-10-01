"""Conversation ownership comes from source refs or creation evidence.

Fresh isolated brains reproduce the two observed wrong-session shapes and pin
the boundary with general MCP trace/episode search. No model calls are needed.
"""
from unittest.mock import patch

from tests.brain_test_base import BrainTestBase
from servers.trace_contract import build_delta_metadata, build_node_created_metadata

# A K stamp for builder calls — the builders require one.
_STAMP = {'fingerprint': 'f0f0f0f0f0f0', 'source': 'default', 'version': 0, 'id': None}


class TestConversationProvenance(BrainTestBase):
    needs_embedder = False
    center = '2026-09-10T20:16:36.781948+00:00'

    def _trace(self, session, timestamp, ref_type='user_message', *,
               ref_id='', metadata=None, content='', scale='s0'):
        with patch('servers.dal_logs.iso_now', return_value=timestamp):
            return self.brain._trace_dal.append(
                chain_id='s0-%s-1' % session, scale=scale,
                event_type='K' if ref_type == 'user_message' else 'delta',
                ref_type=ref_type, ref_id=ref_id, session_id=session,
                summary=content, metadata=metadata)

    def _creation(self, node, session, timestamp=None):
        return self._trace(
            session, timestamp or self.center, 'node_created', ref_id=node,
            metadata=build_node_created_metadata(node_id=node, encoding_source='anchor'))

    def _run(self, session, timestamp=None, **metadata):
        delta = build_delta_metadata(interaction=_STAMP)
        delta.update(metadata)
        return self._trace(session, timestamp or self.center, 'encoding_run',
                           metadata=delta, scale='s1')

    def _ids(self, **kwargs):
        context = self.brain.get_conversation_around(before=0, after=0, **kwargs)
        return [turn['trace_id'] for conversation in context['conversations']
                for window in conversation['windows'] for turn in window['turns']]

    def _sequence(self, session, count):
        return [self._trace(session, '2026-09-10T20:%02d:00+00:00' % i,
                            'assistant_message' if i % 2 else 'user_message',
                            content='%s turn %d' % (session, i))
                for i in range(count)]

    def _node_with_refs(self, refs):
        return self.brain.remember(
            type='fact', title='Cited context test node',
            content='A memory with stored source references.', source_refs=refs)['id']

    def test_source_refs_select_cited_session_and_time_over_creation(self):
        cited = self._trace('cited', '2026-09-10T19:00:00+00:00', content='source exchange')
        self._trace('cited', self.center, content='later unrelated exchange')
        self._trace('writer', self.center, content='memory was written here')
        node = self._node_with_refs([cited])
        self._creation(node, 'writer')
        self._run('writer', created=[node])
        assert self._ids(node_id=node) == [cited]

    def test_all_cited_sessions_survive_reordering_and_overlapping_clocks(self):
        first = self._trace('first', self.center, content='first session')
        second = self._trace('second', self.center, content='second session')
        node = self._node_with_refs([second, first])
        context = self.brain.get_conversation_around(node_id=node, before=0, after=0)
        assert context['basis'] == 'source_refs'
        assert context['missing_trace_ids'] == []
        assert [(c['session_id'], [w['anchor_trace_ids'] for w in c['windows']])
                for c in context['conversations']] == [('first', [[first]]), ('second', [[second]])]
        assert self._ids(node_id=node) == [first, second]
        self.brain.revise(node, source_refs=[first, second], reason='Reorder equivalent citations')
        assert self.brain.get_conversation_around(node_id=node, before=0, after=0) == context

    def test_overlapping_windows_merge_transitively_without_duplicate_turns(self):
        turns = self._sequence('one', 16)
        anchors = [turns[i] for i in (3, 7, 11)]
        node = self._node_with_refs(list(reversed(anchors)))
        context = self.brain.get_conversation_around(node_id=node, before=1, after=1)
        assert len(context['conversations']) == 1
        windows = context['conversations'][0]['windows']
        assert len(windows) == 1
        assert windows[0]['anchor_trace_ids'] == anchors
        assert [t['trace_id'] for t in windows[0]['turns']] == turns[1:14]

    def test_distant_windows_preserve_gaps_within_a_session(self):
        turns = self._sequence('one', 16)
        node = self._node_with_refs([turns[12], turns[2]])
        context = self.brain.get_conversation_around(node_id=node, before=1, after=1)
        windows = context['conversations'][0]['windows']
        assert [w['anchor_trace_ids'] for w in windows] == [[turns[2]], [turns[12]]]
        assert [[t['trace_id'] for t in w['turns']] for w in windows] == [turns[:5], turns[10:15]]

    def test_zero_width_cited_window_retains_all_timestamp_ties(self):
        first = self._trace('one', self.center, content='cited question')
        second = self._trace('one', self.center, 'assistant_message', content='same-clock answer')
        node = self._node_with_refs([first, second])
        context = self.brain.get_conversation_around(node_id=node, before=0, after=0)
        windows = context['conversations'][0]['windows']
        assert len(windows) == 1
        assert set(windows[0]['anchor_trace_ids']) == {first, second}
        assert [t['trace_id'] for t in windows[0]['turns']] == [first, second]
        self.brain.revise(node, source_refs=[first], reason='Cite only the question')
        assert self._ids(node_id=node) == [first, second]
        # Single-session readers retain their established sizing by default.
        default = self.brain.get_conversation(
            'one', around_timestamp=self.center, before=0, after=0)
        assert [t['trace_id'] for t in default] == [second]

    def test_cited_windows_expand_ties_at_both_boundaries(self):
        timestamps = [0, 1, 1, 1, 2, 3, 3, 3, 4]
        turns = [self._trace('one', '2026-09-10T20:%02d:00+00:00' % minute,
                             content='row %d' % i) for i, minute in enumerate(timestamps)]
        node = self._node_with_refs([turns[4]])
        context = self.brain.get_conversation_around(node_id=node, before=1, after=1)
        windows = context['conversations'][0]['windows']
        assert len(windows) == 1
        assert [t['trace_id'] for t in windows[0]['turns']] == turns[1:8]

    def test_windows_with_tied_centers_merge_in_reader_order(self):
        timestamps = [0, 1, 1, 1, 2, 3, 3, 3, 4]
        turns = [self._trace('one', '2026-09-10T20:%02d:00+00:00' % minute,
                             content='row %d' % i) for i, minute in enumerate(timestamps)]
        node = self._node_with_refs([turns[i] for i in (7, 1, 6, 3)])
        context = self.brain.get_conversation_around(node_id=node, before=1, after=1)
        windows = context['conversations'][0]['windows']
        assert len(windows) == 1
        assert [t['trace_id'] for t in windows[0]['turns']] == turns[1:]

    def test_explicit_session_overrides_a_nodes_cited_session(self):
        cited = self._trace('cited', self.center, content='source')
        explicit = self._trace('explicit', self.center, content='requested session')
        node = self._node_with_refs([cited])
        assert self._ids(node_id=node, session_id='explicit', timestamp=self.center) == [explicit]

    def test_absorb_preserves_both_cited_conversations_despite_tied_ref_positions(self):
        primary = self._trace('survivor', self.center, content='survivor source')
        secondary = self._trace('absorbed', self.center, content='absorbed source')
        survivor = self._node_with_refs([primary])
        absorbed = self.brain.remember(
            type='fact', title='Absorbed cited memory', content='Additional evidence',
            source_refs=[secondary])['id']
        self.brain.absorb(survivor, absorbed)
        assert set(self.brain.get_source_refs(survivor)) == {primary, secondary}
        context = self.brain.get_conversation_around(node_id=survivor, before=0, after=0)
        assert {c['session_id'] for c in context['conversations']} == {'survivor', 'absorbed'}
        assert set(self._ids(node_id=survivor)) == {primary, secondary}

    def test_missing_source_is_visible_while_valid_source_survives(self):
        secondary = self._trace('secondary', self.center, content='secondary source')
        self._trace('writer', self.center, content='creation context')
        node = self._node_with_refs(['ffffffff', secondary])
        self._creation(node, 'writer')
        context = self.brain.get_conversation_around(node_id=node, before=0, after=0)
        assert context['basis'] == 'source_refs'
        assert context['missing_trace_ids'] == ['ffffffff']
        assert self._ids(node_id=node) == [secondary]

    def test_sessionless_source_is_visible_without_using_creation_session(self):
        primary = self._trace('', self.center, content='unscoped source')
        self._trace('writer', self.center, content='creation context')
        node = self._node_with_refs([primary])
        self._creation(node, 'writer')
        context = self.brain.get_conversation_around(node_id=node)
        assert context == {'basis': 'source_refs', 'conversations': [],
                           'missing_trace_ids': [primary]}

    def test_all_missing_sources_do_not_fall_back_to_creation(self):
        self._trace('writer', self.center, content='creation context')
        node = self._node_with_refs(['ffffffff', 'eeeeeeee'])
        self._creation(node, 'writer')
        assert self.brain.get_conversation_around(node_id=node) == {
            'basis': 'source_refs', 'conversations': [],
            'missing_trace_ids': ['eeeeeeee', 'ffffffff']}

    def test_failed_trace_batch_preserves_unavailable_source_ids(self):
        cited = self._trace('cited', self.center)
        node = self._node_with_refs([cited])
        with patch.object(self.brain, 'get_traces', side_effect=RuntimeError('trace read failed')):
            with patch.object(self.brain, '_log_error') as error:
                context = self.brain.get_conversation_around(node_id=node)
        assert context == {'basis': 'source_refs', 'conversations': [],
                           'missing_trace_ids': [cited]}
        assert 'trace read failed' in str(error.call_args.args[1])

    def test_empty_cited_session_does_not_hide_another_sessions_context(self):
        empty = self._run('empty')
        valid = self._trace('valid', self.center, content='available context')
        node = self._node_with_refs([empty, valid])
        context = self.brain.get_conversation_around(node_id=node)
        assert context['missing_trace_ids'] == [empty]
        assert self._ids(node_id=node) == [valid]

    def test_failed_session_read_does_not_hide_another_sessions_context(self):
        failed = self._trace('failed', self.center, content='unreadable context')
        valid = self._trace('valid', self.center, content='available context')
        node = self._node_with_refs([failed, valid])
        reader = self.brain._trace_dal.get_session_turns

        def read(session_id, **kwargs):
            if session_id == 'failed':
                raise RuntimeError('session read failed')
            return reader(session_id, **kwargs)

        with patch.object(self.brain._trace_dal, 'get_session_turns', side_effect=read):
            with patch.object(self.brain, '_log_error') as error:
                context = self.brain.get_conversation_around(node_id=node)
        assert context['missing_trace_ids'] == [failed]
        assert [c['session_id'] for c in context['conversations']] == ['valid']
        assert 'session read failed' in str(error.call_args.args[1])

    def test_failed_source_ref_read_is_reported_without_creation_substitute(self):
        self._trace('writer', self.center, content='creation context')
        self._creation('authored', 'writer')
        with patch.object(self.brain, 'get_source_refs', side_effect=RuntimeError('refs unavailable')):
            with patch.object(self.brain, '_log_error') as error:
                assert self._ids(node_id='authored') == []
        assert error.call_count == 1
        assert 'refs unavailable' in str(error.call_args.args[1])

    def test_healer_renders_sessions_gaps_and_missing_sources_without_creation_marker(self):
        from servers.scales.s2.healer_decoder import HealerDecoder
        from servers.scales.s2.healer_encoder import HealerEncoder

        turns = self._sequence('cited', 55)
        other = self._trace('another', self.center, content='Independent cited evidence')
        self._trace('writer', self.center, content='Wrong creation conversation')
        node = self._node_with_refs([turns[0], turns[50], other, 'ffffffff'])
        self._creation(node, 'writer')
        proposals = HealerDecoder(self.brain)._build_proposals([node])
        assert len(proposals) == 1
        context = proposals[0]['conversation']
        assert context['basis'] == 'source_refs'
        assert len(context['conversations']) == 2
        assert 'encoding_timestamp' not in proposals[0]
        prompt = HealerEncoder(self.brain)._format_batch(proposals)
        assert 'Conversation cited' in prompt
        assert 'Conversation another' in prompt
        assert 'CONTEXT BASIS: source_refs' in prompt
        assert '[Separate excerpt]' in prompt
        assert 'Unavailable sources: ffffffff' in prompt
        assert '[operator] cited turn 0' in prompt
        assert '[assistant] cited turn 1' in prompt
        assert 'cited turn 20' not in prompt
        assert 'Independent cited evidence' in prompt
        for anchor in (turns[0], turns[50], other):
            assert anchor in prompt
        assert 'Wrong creation conversation' not in prompt
        assert 'ENCODED AROUND HERE' not in prompt
        assert 'around when this node was encoded' not in prompt

    def test_healer_availability_counts_excerpts_not_empty_envelopes(self):
        from servers.scales.s2.healer_decoder import HealerDecoder

        cited = self._trace('cited', self.center, content='available context')
        available = self._node_with_refs([cited])
        missing = self._node_with_refs(['ffffffff'])
        decoder = HealerDecoder(self.brain)
        with patch.object(decoder, '_find_targets', return_value=[available, missing]):
            with patch.object(decoder, 'trace') as trace:
                result = decoder.run()
        assert len(result['proposals']) == 2
        metadata = next(call.kwargs['metadata'] for call in trace.call_args_list
                        if call.args[1] == 'healer_proposals')
        assert metadata['with_conversation'] == 1

    def test_observed_authored_memories_resolve_to_their_recorded_sessions(self):
        s1e = '01a08172-63d2-7913-be97-3a99ac8d0830'
        host = '01a08cf2-be83-78b2-a59a-4c97a173859c'
        cases = (
            ('e1642b9a', s1e, host, '2026-09-11T17:50:12.995636+00:00'),
            ('7ed4d8ae', host, s1e, self.center),
        )
        for node, origin, other, timestamp in cases:
            with self.subTest(node=node):
                expected = self._trace(origin, timestamp, content='origin conversation')
                self._trace(other, timestamp, content='unrelated concurrent conversation')
                self._run(other, timestamp, created=['different-node'])
                self._creation(node, origin, timestamp)
                assert self._ids(node_id=node) == [expected]
                context = self.brain.get_conversation_around(node_id=node)
                assert context['basis'] == 'creation_trace'
                assert context['conversations'][0]['session_id'] == origin

    def test_encoded_origin_uses_exact_created_membership_and_run_center(self):
        self._creation('encoded-node', 'origin', '2026-09-10T20:00:00+00:00')
        self._trace('origin', '2026-09-10T20:00:00+00:00', content='earlier turn')
        expected = self._trace('origin', self.center, content='run context')
        self._trace('other', self.center, content='wrong conversation')
        self._run('other', created=['other-node'], revised=['encoded-node'],
                  journal_entry='Mentioning encoded-node does not mean creating it')
        self._run('origin', created=['encoded-node'])
        assert self._ids(node_id='encoded-node') == [expected]

    def test_encoding_run_alone_is_sufficient_creation_evidence(self):
        expected = self._trace('origin', self.center, content='encoded here')
        self._run('origin', created=['encoded-node'])
        assert self._ids(node_id='encoded-node') == [expected]

    def test_mentions_revisions_prefixes_and_non_array_created_are_not_origins(self):
        self._trace('other', self.center, content='tempting nearby conversation')
        for metadata in (
            {'created': ['node1234-longer']},
            {'created': 'node1234'},
            {'created': {'node1234': True}},
            {'created': [{'id': 'node1234'}]},
            {'created': None},
            {'revised': ['node1234']},
            {'journal_entry': 'node1234'},
        ):
            with self.subTest(metadata=metadata):
                self._run('other', **metadata)
                assert self._ids(node_id='node1234') == []

    def test_creation_ref_id_must_match_exactly(self):
        self._trace('other', self.center, content='wrong conversation')
        self._creation('node1234-longer', 'other')
        assert self._ids(node_id='node1234') == []

    def test_historic_malformed_metadata_cannot_establish_creation(self):
        trace = self._run('other')
        for raw in ('broken JSON', 'null', '["node1234"]', '"node1234"'):
            with self.subTest(raw=raw):
                # Raw legacy cells bypass the writer deliberately, in this
                # test's fresh database only. They must not break the reader.
                self.brain.logs_conn_w.execute(
                    'UPDATE trace_events SET metadata=? WHERE id=?', (raw, trace))
                self.brain.logs_conn_w.commit()
                assert self.brain._trace_dal.get_node_creation_traces('node1234') == []

    def test_conflicting_recorded_sessions_are_reported_without_guessing(self):
        self._trace('one', self.center, content='one')
        self._trace('two', self.center, content='two')
        self._creation('conflict', 'one')
        self._run('two', created=['conflict'])
        with patch.object(self.brain, '_log_error') as error:
            assert self._ids(node_id='conflict') == []
        assert error.call_count == 1
        assert 'conflicting sessions' in str(error.call_args.args[1])

    def test_sessionless_creation_cannot_select_a_nearby_session(self):
        self._trace('other', self.center, content='wrong conversation')
        self._creation('unscoped', '')
        assert self._ids(node_id='unscoped') == []

    def test_empty_explicit_or_resolved_session_stays_empty(self):
        self._trace('other', self.center, content='wrong conversation')
        self._creation('authored', 'empty')
        assert self._ids(session_id='empty', timestamp=self.center) == []
        assert self._ids(node_id='authored') == []

    def test_timestamp_alone_cannot_select_a_conversation(self):
        self._trace('other', self.center, content='wrong conversation')
        assert self._ids(timestamp=self.center) == []
        assert self.brain.get_conversation('') == []

    def test_explicit_session_and_timestamp_remain_authoritative(self):
        expected = self._trace('explicit', self.center, content='selected context')
        self._creation('authored', 'another')
        assert self._ids(node_id='authored', session_id='explicit',
                         timestamp=self.center) == [expected]

    def test_explicit_timestamp_positions_window_in_recorded_session(self):
        timestamp = '2026-09-10T20:00:00+00:00'
        expected = self._trace('origin', timestamp, content='requested earlier point')
        self._trace('origin', self.center, content='creation point')
        self._creation('authored', 'origin')
        assert self._ids(node_id='authored', timestamp=timestamp) == [expected]

    def test_recent_and_centered_readers_preserve_correspondents_and_order(self):
        first = self._trace('origin', '2026-09-10T20:00:00+00:00', content='question')
        second = self._trace('origin', self.center, 'assistant_message', content='answer')
        self._trace('other', self.center, content='not part of this conversation')
        recent = self.brain.get_conversation('origin', with_judge_output=False)
        context = self.brain.get_conversation_around(
            session_id='origin', timestamp=self.center)
        assert context['basis'] == 'explicit_session'
        assert context['missing_trace_ids'] == []
        assert len(context['conversations']) == 1
        assert context['conversations'][0]['session_id'] == 'origin'
        centered = context['conversations'][0]['windows'][0]['turns']
        assert [row['trace_id'] for row in centered] == [first, second]
        assert [row['ref_type'] for row in centered] == ['user_message', 'assistant_message']
        assert centered == [{k: v for k, v in row.items() if k != 'judge_output'}
                            for row in recent]

    def test_failed_origin_read_is_reported_without_fallback(self):
        self._trace('other', self.center, content='wrong conversation')
        with patch.object(self.brain._trace_dal, 'get_node_creation_traces',
                          side_effect=RuntimeError('origin read failed')):
            with patch.object(self.brain, '_log_error') as error:
                assert self._ids(node_id='authored') == []
        assert error.call_count == 1
        assert 'origin read failed' in str(error.call_args.args[1])

    def test_failed_conversation_read_uses_shared_error_reporting(self):
        with patch.object(self.brain._trace_dal, 'get_session_turns',
                          side_effect=RuntimeError('conversation read failed')):
            with patch.object(self.brain, '_log_error') as error:
                assert self._ids(session_id='origin', timestamp=self.center) == []
        assert error.call_args.args[0] == 'get_conversation'

    def test_general_trace_and_episode_search_remain_cross_session_and_time_filtered(self):
        from servers.clock import iso_now, iso_cutoff
        old = self._trace('old', iso_cutoff(days=30), content='search-boundary old')
        expected = {
            self._trace(session, iso_now(), content='search-boundary recent')
            for session in ('one', 'two')
        }
        traces = self.brain.query_traces(ref_type='user_message', hours=24)['events']
        episodes = self.brain.recall_episodes(
            contains='search-boundary', younger_than='1d')['episodes']
        assert {row['id'] for row in traces} == expected
        assert {row['id'] for row in episodes} == expected
        all_traces = self.brain.query_traces(ref_type='user_message', hours=None)['events']
        all_episodes = self.brain.recall_episodes(
            contains='search-boundary', younger_than='60d')['episodes']
        assert {row['id'] for row in all_traces} == expected | {old}
        assert {row['id'] for row in all_episodes} == expected | {old}
