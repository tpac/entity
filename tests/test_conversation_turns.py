"""The shared conversation reader: trace_contract.group_turns / row_speaker
and brain_traces.conversation_turns — the turns the encoder timeline, the
Surface window and the LAF moment stack all read."""
import pytest

from servers.trace_contract import (TURN_READ_ROWS_PER_TURN, WAKE_ENVELOPE_MARKER,
                                    group_turns, row_speaker, row_text)
from tests.isolated_brain import brain_env

SID = 'convturn-aaaa-bbbb-cccc-0123456789ab'


def _row(ref_type, text='', chain='s0-x-1', host_turn=None, **extra):
    meta = {'content': text}
    if host_turn:
        meta['host_turn'] = host_turn
    return dict({'ref_type': ref_type, 'chain_id': chain, 'metadata': meta,
                 'summary': text[:200]}, **extra)


class TestGroupTurns:

    def test_absorbed_message_joins_its_opener(self):
        a = _row('user_message', 'A', host_turn='P1')
        b = _row('user_message', 'B', host_turn='P1')
        turns = group_turns([a, _row('tool_result', 'Bash: x'), b,
                             _row('assistant_message', 'reply')])
        assert len(turns) == 1
        assert turns[0]['user'] is a and turns[0]['with'] == [b]
        assert turns[0]['assistant']['metadata']['content'] == 'reply'
        assert len(turns[0]['actions']) == 1

    def test_interrupted_and_unstamped_prompts_each_open_a_turn(self):
        for b_turn in ('P2', None):
            turns = group_turns([_row('user_message', 'A', host_turn='P1'),
                                 _row('user_message', 'B', host_turn=b_turn),
                                 _row('assistant_message', 'reply')])
            assert [t['user']['metadata']['content'] for t in turns] == ['A', 'B']
            assert turns[0]['assistant'] is None

    def test_new_chain_opens_a_turn_even_on_the_same_host_turn(self):
        turns = group_turns([_row('user_message', 'A', host_turn='P1'),
                             _row('assistant_message', 'r1'),
                             _row('user_message', 'B', chain='s0-x-2', host_turn='P1')])
        assert len(turns) == 2

    def test_machine_opened_turn_keeps_its_opener_and_riders(self):
        notice = _row('env_message', '<cross-session-message from="x">hi',
                      chain='s0-x-2')
        status = _row('env_message', 'Linked sessions now', chain='s0-x-2')
        turns = group_turns([notice, status, _row('assistant_message', 'ok')])
        assert len(turns) == 1
        assert turns[0]['user'] is notice and turns[0]['with'] == [status]

    def test_leading_reply_opens_an_orphan_turn(self):
        turns = group_turns([_row('assistant_message', 'r0'),
                             _row('user_message', 'A')])
        assert turns[0]['user'] is None and turns[0]['assistant'] is not None
        assert turns[1]['user']['metadata']['content'] == 'A'


class TestRowSpeaker:

    @pytest.mark.parametrize('ref_type,text,speaker', [
        ('user_message', 'hello', 'operator'),
        ('user_message', WAKE_ENVELOPE_MARKER + '<task-id>x', 'harness'),
        ('user_message', 'quoting ' + WAKE_ENVELOPE_MARKER, 'operator'),
        ('env_message', 'Linked sessions now', 'harness'),
        ('subagent_message', 'report', 'subagent'),
        ('assistant_message', 'reply', None),
        ('tool_result', 'Bash: ls', None),
    ])
    def test_speaker(self, ref_type, text, speaker):
        assert row_speaker(_row(ref_type, text)) == speaker

    def test_reads_a_flat_content_row(self):
        row = {'ref_type': 'user_message', 'content': WAKE_ENVELOPE_MARKER,
               'metadata': {}, 'summary': 'operator words'}
        assert row_speaker(row) == 'harness'


class TestConversationTurns:

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            yield

    def _in(self, chain, text, ref_type='user_message', host_turn=None,
            recall_chain=None):
        meta = {'content': text}
        if host_turn:
            meta['host_turn'] = host_turn
        if recall_chain:
            meta['recall_chain'] = recall_chain
        return self.dal.append(chain_id=chain, scale='s0', event_type='K',
                               ref_type=ref_type, summary=text[:200],
                               metadata=meta, session_id=SID)

    def _reply(self, chain, text):
        return self.dal.append(chain_id=chain, scale='s0', event_type='delta',
                               ref_type='assistant_message', summary=text[:200],
                               metadata={'content': text}, session_id=SID)

    def _tool(self, chain, cue):
        return self.dal.append(chain_id=chain, scale='s0', event_type='delta',
                               ref_type='tool_result', summary=cue,
                               session_id=SID)

    def _texts(self, turns):
        return [[row_text(r) for r in ([t['user']] if t['user'] else []) + t['with']]
                + [row_text(t['assistant']) if t['assistant'] else None]
                for t in turns]

    def test_last_n_whole_turns(self):
        for i in range(5):
            self._in('s0-c-%d' % i, 'q%d' % i)
            self._reply('s0-c-%d' % i, 'a%d' % i)
        turns = self.brain.conversation_turns(SID, 3)
        assert self._texts(turns) == [['q2', 'a2'], ['q3', 'a3'], ['q4', 'a4']]

    def test_a_page_never_returns_a_cut_turn(self):
        # Turn A outgrows the first page (TURN_READ_ROWS_PER_TURN × 2 rows):
        # the page holds A's tail without its opener, so the reader must grow
        # the page rather than hand back a turn opened by a status note.
        self._in('s0-c-1', 'A')
        for i in range(2 * TURN_READ_ROWS_PER_TURN + 4):
            self._in('s0-c-1', 'status %d' % i, ref_type='env_message')
        self._reply('s0-c-1', 'reply A')
        self._in('s0-c-2', 'B')
        self._reply('s0-c-2', 'reply B')
        turns = self.brain.conversation_turns(SID, 2)
        assert [row_text(t['user']) for t in turns] == ['A', 'B']
        assert len(turns[0]['with']) == 2 * TURN_READ_ROWS_PER_TURN + 4

    def test_before_trace_id_drops_the_current_prompts_pieces(self):
        self._in('s0-c-1', 'A')
        self._reply('s0-c-1', 'reply A')
        current = self._in('s0-c-2', 'C')
        self._in('s0-c-2', 'Linked sessions now', ref_type='env_message')
        turns = self.brain.conversation_turns(SID, 4, before_trace_id=current)
        assert self._texts(turns) == [['A', 'reply A']]

    def test_absorbed_message_still_sees_its_opener_turn(self):
        self._in('s0-c-1', 'A', host_turn='P1')
        self._tool('s0-c-1', 'Bash: grep')
        b = self._in('s0-c-1', 'B', host_turn='P1')
        turns = self.brain.conversation_turns(SID, 4, before_trace_id=b)
        assert self._texts(turns) == [['A', None]]
        after = self.brain.conversation_turns(SID, 4, with_actions=True)
        assert self._texts(after) == [['A', 'B', None]]
        assert [row_text(r) for r in after[0]['actions']] == ['Bash: grep']

    def test_a_turn_with_more_tool_rows_than_a_page_stays_whole(self):
        # Tool rows never share the grouping page: a turn whose tools outrun
        # TURN_READ_MAX_ROWS keeps its prompt and reply, and carries the
        # newest tools.
        from unittest.mock import patch
        import servers.trace_contract as tc
        self._in('s0-c-1', 'A')
        for i in range(30):
            self._tool('s0-c-1', 'Bash: step %02d' % i)
        self._reply('s0-c-1', 'reply A')
        with patch.object(tc, 'TURN_READ_MAX_ROWS', 16):
            turns = self.brain.conversation_turns(SID, 2, with_actions=True)
        assert self._texts(turns) == [['A', 'reply A']]
        assert [row_text(r) for r in turns[0]['actions']] == [
            'Bash: step %02d' % i for i in range(14, 30)]

    def test_actions_only_when_asked(self):
        self._in('s0-c-1', 'A')
        self._tool('s0-c-1', 'Read: x.py')
        self._reply('s0-c-1', 'reply')
        assert self.brain.conversation_turns(SID, 1)[0]['actions'] == []
        assert len(self.brain.conversation_turns(
            SID, 1, with_actions=True)[0]['actions']) == 1

    def test_recall_chain_fill_on_incoming_rows(self):
        rc = 's1r-convturn-1'
        self._in('s0-c-1', 'what did we decide?', recall_chain=rc)
        self._reply('s0-c-1', 'X')
        self.dal.append(chain_id=rc, scale='s1', event_type='K',
                        ref_type='surface_selected', summary='1 surfaced',
                        metadata={'selected': ['aaaabbbb|the decision']},
                        session_id=SID)
        turn = self.brain.conversation_turns(SID, 1, with_surfaced=True,
                                             with_judge_output=True)[0]
        assert turn['user']['surfaced'] == [{'id': 'aaaabbbb', 'title': 'the decision'}]
        assert turn['user']['judge_output'] == ''
        assert 'surfaced' not in turn['assistant']
        plain = self.brain.conversation_turns(SID, 1)[0]
        assert 'surfaced' not in plain['user']

    def test_older_than_is_the_as_of_cut(self):
        self._in('s0-c-1', 'A')
        self._reply('s0-c-1', 'reply A')
        b = self._in('s0-c-2', 'B')
        at = self.dal.get_by_ids([b])[0]['created_at']
        turns = self.brain.conversation_turns(SID, 4, older_than=at)
        assert self._texts(turns) == [['A', 'reply A']]
