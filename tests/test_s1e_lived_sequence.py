"""Piece 1 of the S1E code-half rebuild — the lived-sequence timeline.

Verifies the A/B flag's two arms (docs/S1-SCRIBE-REDESIGN.md §10.3.1):
  - OFF → the markdown messages-only timeline (control), byte-identical to before.
  - ON  → the XML lived sequence (messages + tool actions interleaved), read
          through brain.conversation_turns (encode._window), grouped into turns.

Both the renderer alone and the ASSEMBLED body (through _build_user_content, where
the <timeline> wrapper and its `now=` stamp are added) — a renderer that works but
assembles into an empty shell is the gap the wrapper-only checks can't see.

No DB needed — a tiny stub brain feeds the readers the controlled inputs.
"""
import os
import sys
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.brain_traces import BrainTracesMixin  # noqa: E402
from servers.trace_contract import group_turns, row_text  # noqa: E402
from servers.scales.s1.encode import (  # noqa: E402
    _render_lived_sequence_timeline, _window, _xml_escape, _build_user_content,
)


class _StubNodes:
    def get_title(self, rid):
        return 'Title-' + rid


def _no_selections(chains, **kwargs):
    return {}, {}


class _StubBrain:
    """Feeds recall_episodes a fixed episode list, read by the real
    conversation_turns; serves get_title. No earlier encode unless an anchor
    is given."""
    conversation_turns = BrainTracesMixin.conversation_turns

    def __init__(self, episodes, anchor=''):
        self._episodes = episodes
        self._nodes = _StubNodes()
        self._anchor = anchor
        self._trace_dal = SimpleNamespace(get_by_ids=lambda ids: [],
                                          recall_chain_selections=_no_selections)
        self.errors = []

    def last_encode_anchor(self, session_id, older_than=None):
        return self._anchor

    def _log_error(self, source, exc, context=''):
        self.errors.append((source, str(exc), context))

    def recall_episodes(self, **kwargs):
        # Honor the ref_type whitelist and the older_than cut the caller
        # passes, like the real one.
        rts = kwargs.get('ref_type') or []
        eps = [e for e in self._episodes if e.get('ref_type') in rts] if rts else self._episodes
        if kwargs.get('older_than'):
            eps = [e for e in eps if e['created_at'] < kwargs['older_than']]
        return {'episodes': eps, 'ranked_by': 'time', 'truncated': False}

    def speaker_for(self, session_id, ref_type):
        # An install that declares no counterpart: no speaker is named.
        return ''


def _ep(rt, summary, ts, tid, tool=None, content=None, chain=None):
    # Real message traces store the capped body in metadata['content'] and a
    # 200-char display `summary`; tool_result traces store only the cue in summary.
    meta = {'tool': tool} if tool else {}
    if content is not None:
        meta['content'] = content
    ep = {'id': tid, 'ref_type': rt, 'summary': summary, 'created_at': ts, 'metadata': meta}
    if chain is not None:
        ep['chain_id'] = chain
    return ep


# Two turns: turn 1 (lock fix) edits a file; turn 2 (run tests) runs bash.
# Deliberately out of created_at order in the list to prove the sort.
# Messages carry a distinct metadata['content'] (fuller than summary) to prove the
# render reads content, not the truncated summary.
EPISODES = [
    _ep('assistant_message', 'found it', '2026-06-29T00:00:02', 'a1',
        content='found it — the bg writer holds the lock through the whole batch'),
    _ep('user_message', 'the recall keeps locking', '2026-06-29T00:00:01', 'u1',
        content='the recall keeps locking — can you check the wal-index path?'),
    _ep('tool_result', 'Edit: servers/dal.py', '2026-06-29T00:00:03', 't1', 'Edit'),
    _ep('user_message', 'run the tests', '2026-06-29T00:00:04', 'u2', content='run the tests'),
    _ep('tool_result', 'Bash: pytest test_write_txn.py', '2026-06-29T00:00:05', 't2', 'Bash'),
    _ep('assistant_message', '12 passed', '2026-06-29T00:00:06', 'a2', content='12 passed'),
]

def test_lived_sequence_interleaves_actions_into_turns():
    brain = _StubBrain(EPISODES)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(brain, 'sess'))

    assert out.count('<turn n="') == 2
    assert '<turn n="1">' in out and '<turn n="2">' in out
    assert '<other trace="u1">' in out and '<me trace="a1">' in out

    # The Edit action lands in turn 1, the Bash action in turn 2 — split on the
    # turn-2 boundary and check each action is in its own turn and nowhere else.
    t1, t2 = out.split('<turn n="2">')
    assert 'Edit: servers/dal.py' in t1 and 'Edit: servers/dal.py' not in t2
    assert 'Bash: pytest test_write_txn.py' in t2 and 'Bash: pytest test_write_txn.py' not in t1
    assert '<actions>' in out


def test_messages_render_full_content_not_summary():
    # Regression guard for the code-review HIGH finding: render metadata['content']
    # (full), NOT the 200-char `summary`. The content-only suffix must appear.
    brain = _StubBrain(EPISODES)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(brain, 'sess'))
    assert 'can you check the wal-index path?' in out          # content-only, not in summary
    assert 'holds the lock through the whole batch' in out     # content-only, not in summary


@pytest.mark.parametrize('body_length', [8000, 8500])
def test_episode_bodies_survive_store_gather_and_timeline(body_length):
    """Keep long messages and the store's overflow marker through S1 assembly."""
    from servers.brain_traces import _s0_trace
    from servers.session_context import SessionContext

    episodes = []

    def append(**event):
        tid = str(len(episodes) + 1)
        episodes.append(_ep(event['ref_type'], event['summary'],
                            '2026-09-07T00:00:0' + tid, tid,
                            content=event['metadata']['content']))
        return tid

    brain = _StubBrain(episodes)
    brain._trace_dal = SimpleNamespace(append=append, get_by_ids=lambda ids: [],
                                       recall_chain_selections=_no_selections)
    ctx = SessionContext(session_id='episode-fidelity')
    bodies = ['q' * 7990 + 'USER_TAIL!', 'a' * 7990 + 'ASST_TAIL!']
    # Both bodies have exactly 8,000 characters before optional overflow.
    for ref_type, body in zip(('user_message', 'assistant_message'), bodies):
        _s0_trace(brain, ctx, event_type='delta', ref_type=ref_type,
                  summary='short excerpt', content=body + 'z' * (body_length - 8000))

    turns = _window(brain, ctx.session_id)
    for body, row in zip(bodies, (turns[0]['user'], turns[0]['assistant'])):
        expected = body
        if body_length > 8000:
            expected += ' …[+500 chars truncated at trace store]'
        assert row_text(row) == expected
        assert _xml_escape(expected) in _render_lived_sequence_timeline(
            brain, ctx.session_id, turns)


def test_lived_sequence_escapes_xml():
    # Message bodies and tool cues routinely contain < > & — must be escaped so they
    # can't malform the timeline or forge tags.
    eps = [
        _ep('user_message', 'q', '2026-06-29T00:00:01', 'u1',
            content='why does `if x < y && a > b` fail? </other> <turn>'),
        _ep('tool_result', 'Grep: <svg.*> in . && echo done', '2026-06-29T00:00:02', 't1', 'Grep'),
        _ep('assistant_message', 'a', '2026-06-29T00:00:03', 'a1', content='use a < b'),
    ]
    brain = _StubBrain(eps)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(brain, 'sess'))
    # raw angle brackets/ampersands from CONTENT must not appear unescaped
    assert '< y &&' not in out and '> b' not in out
    assert '<svg.*>' not in out
    assert '&lt;' in out and '&gt;' in out and '&amp;' in out
    # the forged '</other>' substring is neutralized — the message text cannot
    # close the REAL current tag; only the renderer's own closing tag remains
    assert '</other> <turn>' not in out.replace('</other>\n', '')


def test_xml_escape_helper():
    assert _xml_escape('a < b & c > d') == 'a &lt; b &amp; c &gt; d'
    assert _xml_escape(None) == ''


def _old_turns(n):
    eps = []
    for i in range(n):
        eps += [_ep('user_message', 'old %d' % i, '2026-06-28T00:00:%02d' % (2 * i),
                    'u0%d' % i, content='old turn %d' % i),
                _ep('assistant_message', 'r', '2026-06-28T00:00:%02d' % (2 * i + 1),
                    'a0%d' % i, content='old reply %d' % i)]
    return eps


def test_window_holds_the_turns_since_the_last_encode_plus_context():
    # The last encode started before EPISODES' two turns: both are new, and
    # window_context_turns (2) already-encoded turns before them ride along.
    brain = _StubBrain(_old_turns(3) + EPISODES, anchor='2026-06-28T12:00:00')
    turns = _window(brain, 'sess')
    assert [t['user']['id'] for t in turns] == ['u01', 'u02', 'u1', 'u2']
    out = _render_lived_sequence_timeline(brain, 'sess', turns)
    assert out.count('<turn n="') == 4 and 'old turn 0' not in out
    assert brain.errors == []


def test_nothing_new_since_the_last_encode_is_an_empty_window():
    brain = _StubBrain(EPISODES, anchor='2026-06-30T00:00:00')
    assert _window(brain, 'sess') == []


def test_a_message_typed_after_the_anchor_makes_its_turn_new():
    # The encode started inside turn 2 (after u2, before the reply); the
    # operator then typed into that turn. Its turn is new, turn 1 is context.
    import copy
    eps = copy.deepcopy(EPISODES) + [_ep('user_message', 'also this', '2026-06-29T00:00:05.5',
                                         'u3', content='also this', chain='s0-sess-1')]
    for e in eps:
        if e['id'] in ('u2', 'u3'):
            e['chain_id'] = 's0-sess-1'
            e['metadata']['host_turn'] = 'p2'
    brain = _StubBrain(eps, anchor='2026-06-29T00:00:04.5')
    turns = _window(brain, 'sess')
    assert [t['user']['id'] for t in turns] == ['u1', 'u2']
    assert [r['id'] for r in turns[1]['with']] == ['u3']


def test_the_window_as_of_an_earlier_instant():
    calls = []

    class _Asof(_StubBrain):
        def last_encode_anchor(self, session_id, older_than=None):
            calls.append(older_than)
            return ''
    brain = _Asof(EPISODES)
    turns = _window(brain, 'sess', older_than='2026-06-29T00:00:03.5')
    assert calls == ['2026-06-29T00:00:03.5']
    assert [t['user']['id'] for t in turns] == ['u1']


def test_first_encode_reads_every_turn():
    brain = _StubBrain(_old_turns(3) + EPISODES)
    assert len(_window(brain, 'sess')) == 5


def test_a_backlog_past_the_cap_keeps_the_newest_turns_loudly():
    from unittest.mock import patch
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    brain = _StubBrain(_old_turns(3) + EPISODES, anchor='2026-06-27T00:00:00')
    with patch.dict(ENCODING_AGENT, {'window_max_turns': 3}):
        turns = _window(brain, 'sess')
    assert [t['user']['id'] for t in turns] == ['u02', 'u1', 'u2']
    (source, _exc, context), = brain.errors
    assert source == 's1e_window_capped' and '2 unencoded turns left out' in context


# ── incoming speakers (trace_contract.INCOMING_SPEAKERS) ──

def test_every_incoming_speaker_has_a_timeline_element():
    from servers.scales.s1.encoder_view import SPEAKER_ELEMENTS
    from servers.trace_contract import INCOMING_SPEAKERS
    assert set(SPEAKER_ELEMENTS) == set(INCOMING_SPEAKERS.values())
    assert SPEAKER_ELEMENTS['operator'] == 'other'   # the prompt's partner


def _row(rt, tid, chain, n, content, turn=None):
    ep = _ep(rt, content, '2026-06-29T00:00:%02d' % n, tid, content=content,
             chain='s0-sess-%d' % chain)
    if turn:   # the host's name for the turn, as the write door stamps it
        ep['metadata']['host_turn'] = turn
    return ep


# Row shapes as a dial-on daemon writes them (seed-arm run 2026-10-10): a
# first-prompt delivery shares the operator prompt's chain; a Stop delivery
# opens the successor chain, and a composed Stop block puts the stream row and
# the brain row on that one chain.
SPEAKER_SCENARIOS = {
    'prompt_delivery_joins_the_operator_turn': (
        [_row('user_message', 'u1', 0, 1, 'where were we?'),
         _row('thalamus_delivery', 'd1', 0, 2, 'the nightly backup did not run'),
         _row('assistant_message', 'a1', 0, 3, 'on the lock fix; checking the backup')],
        [['<other trace="u1">', '<brain trace="d1">', '<me trace="a1">']]),
    'stop_delivery_opens_its_own_turn': (
        [_row('user_message', 'u1', 0, 1, 'run the tests'),
         _row('assistant_message', 'a1', 0, 2, '12 passed'),
         _row('thalamus_delivery', 'd1', 1, 3, 'rerun the soak test'),
         _row('assistant_message', 'a2', 1, 4, 'starting the soak test')],
        [['<other trace="u1">', '<me trace="a1">'],
         ['<brain trace="d1">', '<me trace="a2">']]),
    'composed_stop_block_is_one_turn': (
        [_row('user_message', 'u1', 0, 1, 'ship it'),
         _row('assistant_message', 'a1', 0, 2, 'merging'),
         _row('self_message', 's1', 1, 3, 'other session says: hold your merge'),
         _row('thalamus_delivery', 'd1', 1, 4, 'the soak test is still running'),
         _row('assistant_message', 'a2', 1, 5, 'holding the merge')],
        [['<other trace="u1">', '<me trace="a1">'],
         ['<session trace="s1">', '<brain trace="d1">', '<me trace="a2">']]),
    'interrupted_operator_prompts_stay_two_turns': (
        [_row('user_message', 'u1', 0, 1, 'delete the old folder'),
         _row('user_message', 'u2', 0, 2, 'wait, list it first'),
         _row('assistant_message', 'a1', 0, 3, 'listed, nothing deleted')],
        [['<other trace="u1">'], ['<other trace="u2">', '<me trace="a1">']]),
    # The host names each prompt's turn: after an interrupt the next prompt is
    # a new turn on the same chain; a message typed mid-turn is absorbed into
    # the running turn and carries its name, so my reply answers both.
    'interrupted_turn_the_host_named_apart_stays_two_turns': (
        [_row('user_message', 'u1', 0, 1, 'delete the old folder', turn='p1'),
         _row('user_message', 'u2', 0, 2, 'wait, list it first', turn='p2'),
         _row('assistant_message', 'a1', 0, 3, 'listed, nothing deleted')],
        [['<other trace="u1">'], ['<other trace="u2">', '<me trace="a1">']]),
    'message_absorbed_mid_turn_joins_its_turn': (
        [_row('user_message', 'u1', 0, 1, 'look into the grouping code', turn='p1'),
         _row('user_message', 'u2', 0, 2, 'also check the Surface window', turn='p1'),
         _row('assistant_message', 'a1', 0, 3, 'both readers share one rule now')],
        [['<other trace="u1">', '<other trace="u2">', '<me trace="a1">']]),
    # A Stop-hook continuation keeps the host's turn across the Stop, but the
    # Stop closed the turn: a message typed during the continuation is on the
    # successor chain and opens its own turn, so each reply keeps its prompt.
    'message_in_a_continuation_after_a_stop_opens_its_own_turn': (
        [_row('user_message', 'u1', 0, 1, 'run the migration', turn='p1'),
         _row('assistant_message', 'a1', 0, 2, 'migration done'),
         _row('user_message', 'u2', 1, 3, 'and the backfill too', turn='p1'),
         _row('assistant_message', 'a2', 1, 4, 'backfill done')],
        [['<other trace="u1">', '<me trace="a1">'],
         ['<other trace="u2">', '<me trace="a2">']]),
    # The prompt hook writes the operator's text first, then the reminder it
    # cut out, on the same chain — the reminder rides in the operator's turn.
    'glued_reminder_rides_in_the_operator_turn': (
        [_row('user_message', 'u1', 0, 1, 'the test fails'),
         _row('env_message', 'e1', 0, 2, 'You are operating in a git worktree.'),
         _row('assistant_message', 'a1', 0, 3, 'fixed it')],
        [['<other trace="u1">', '<harness trace="e1">', '<me trace="a1">']]),
    # A subagent's report and a task notice each arrive as their own prompt:
    # each opens a turn that holds my reply to it.
    'subagent_report_opens_its_own_turn': (
        [_row('user_message', 'u1', 0, 1, 'audit the branch'),
         _row('assistant_message', 'a1', 0, 2, 'delegated the audit'),
         _row('subagent_message', 'r1', 1, 3, 'two findings, both fixed'),
         _row('assistant_message', 'a2', 1, 4, 'read the report')],
        [['<other trace="u1">', '<me trace="a1">'],
         ['<subagent trace="r1">', '<me trace="a2">']]),
    # Two hand-backs in one prompt are two rows on one chain: the second
    # rides in the first's turn, each under its own element.
    'two_reports_in_one_prompt_share_a_turn': (
        [_row('user_message', 'u1', 0, 1, 'get two reviews'),
         _row('assistant_message', 'a1', 0, 2, 'two reviewers are on it'),
         _row('subagent_message', 'r1', 1, 3, 'blocking: batch the backfill'),
         _row('subagent_message', 'r2', 1, 4, 'no blocking issues'),
         _row('assistant_message', 'a2', 1, 5, 'the reviewers disagree')],
        [['<other trace="u1">', '<me trace="a1">'],
         ['<subagent trace="r1">', '<subagent trace="r2">', '<me trace="a2">']]),
    'task_notice_opens_its_own_turn': (
        [_row('user_message', 'u1', 0, 1, 'run the suite in the background'),
         _row('assistant_message', 'a1', 0, 2, 'started'),
         _row('env_message', 'n1', 1, 3, 'background command completed'),
         _row('assistant_message', 'a2', 1, 4, 'suite is green')],
        [['<other trace="u1">', '<me trace="a1">'],
         ['<harness trace="n1">', '<me trace="a2">']]),
}


def test_the_partner_is_named_and_other_voices_are_not():
    class _Named(_StubBrain):
        def speaker_for(self, session_id, ref_type):
            return 'Sam "S" <P>' if ref_type == 'user_message' else ''
    eps, _ = SPEAKER_SCENARIOS['glued_reminder_rides_in_the_operator_turn']
    brain = _Named(eps)
    out = _render_lived_sequence_timeline(brain, 'sess', _window(brain, 'sess'))
    assert '<other name="Sam &quot;S&quot; &lt;P&gt;" trace="u1">' in out   # escaped as an attribute
    assert '<harness trace="e1">' in out


@pytest.mark.parametrize('name', sorted(SPEAKER_SCENARIOS))
def test_the_turn_rule_groups_the_timeline(name):
    # The turn rule (trace_contract.joins_open_turn) for every speaker, dial-on
    # ones simulated by widening the lived read's whitelist as a flip would.
    from unittest.mock import patch
    import servers.trace_contract as tc
    eps, expected = SPEAKER_SCENARIOS[name]
    brain = _StubBrain(eps)
    flipped = ('thalamus_delivery', 'self_message')
    with patch.object(tc, 'CONVERSATIONAL_REF_TYPES', tc.CONVERSATIONAL_REF_TYPES + flipped), \
            patch.object(tc, 'SAID_AND_DID_REF_TYPES', tc.SAID_AND_DID_REF_TYPES + flipped):
        out = _render_lived_sequence_timeline(brain, 'sess', _window(brain, 'sess'))

    turns = out.split('</turn>')[:-1]
    assert len(turns) == len(expected)
    for turn, tags in zip(turns, expected):
        positions = [turn.index(tag) for tag in tags]       # each present…
        assert positions == sorted(positions)                # …in this order
        assert turn.count(' trace="') == len(tags)           # and nothing else


# ── the assembled body (_build_user_content wraps + stamps the render) ──

_NOW = datetime(2026, 6, 29, 14, 5, tzinfo=timezone.utc)   # 14h after the fixtures


def _assembly_brain():
    """_StubBrain + the doors _build_user_content knocks on beyond the
    renderer: continuity notes, session arc, failed-encode residue (all
    empty) and the interaction resolver serving code defaults — so the
    assembled body is the gist and the timeline section."""
    from servers.interaction_defaults import INTERACTION_DEFAULTS
    brain = _StubBrain(EPISODES)
    brain.journal_view = lambda **kw: {'notes': []}
    brain.log_debug = lambda *args, **kw: None
    brain.session_context_for = lambda sid: ''
    brain.query_traces = lambda **kw: {'events': []}
    brain.get_interaction_effective = lambda name: {
        'template': INTERACTION_DEFAULTS[name][0],
        'config': dict(INTERACTION_DEFAULTS[name][1])}
    return brain


def _assemble(view_now):
    # precomputed=('', set(), None) skips _build_catalog — the catalog has its own
    # tests; this one is about the timeline surviving assembly.
    brain = _assembly_brain()
    _pre, body, _cat, _ids = _build_user_content(
        brain, _window(brain, 'sess'), counter=2, session_id='sess',
        precomputed=('', set(), None),
        view_policy=True, view_now=view_now)
    return body


def test_assembled_body_stamps_timeline_and_carries_the_render():
    # The stamp (view policy): the absolute anchor that makes the relative `age=`
    # labels invertible. Deterministic here — view_now is passed, not derived.
    body = _assemble(_NOW)
    assert '<timeline now="2026-06-29 14:05 UTC">' in body

    # …and the render must be INSIDE that wrapper. An empty <timeline></timeline>
    # (assembly dropping the render) satisfies a wrapper-only check but fails here.
    inner = body.split('<timeline now="2026-06-29 14:05 UTC">')[1].split('</timeline>')[0]
    assert inner.count('<turn n="') == 2
    assert '<turn n="1" age="14h ago" encoded="false">' in inner
    assert '<turn n="2" age="14h ago" encoded="false">' in inner
    # full metadata['content'] survives assembly, not the 200-char summary
    assert '<other trace="u1">the recall keeps locking — can you check the ' \
           'wal-index path?</other>' in inner
    assert '<me trace="a1">found it — the bg writer holds the lock through the ' \
           'whole batch</me>' in inner
    # tool actions survive too, each in its own turn
    t1, t2 = inner.split('<turn n="2"')
    assert 'Edit: servers/dal.py' in t1 and 'Bash: pytest test_write_txn.py' in t2


def test_assembled_timeline_degrades_to_bare_tag_when_unstampable():
    # No conversation time resolvable (_conversation_now_safe logged a failure
    # and returned None) → the stamp and the per-turn ages drop, but the
    # timeline still renders. A degraded render, never a broken one — this is
    # why bare-'<timeline>' checks elsewhere still pass.
    from unittest.mock import patch
    with patch('servers.scales.s1.encode._conversation_now_safe', return_value=None):
        body = _assemble(None)
    assert '<timeline>' in body and 'now=' not in body and 'age=' not in body
    inner = body.split('<timeline>')[1].split('</timeline>')[0]
    assert inner.count('<turn n="') == 2
    assert '<turn n="1" encoded="false">' in inner
    assert 'Edit: servers/dal.py' in inner


def test_lived_read_failure_fails_the_window():
    # No fallback: the timeline IS the encoder's input, so a failed trace read
    # raises out of the window read and fails the run (the runner logs it; the
    # turns retry) instead of encoding from a substitute.
    class _BrokenEpisodes(_StubBrain):
        def recall_episodes(self, **kwargs):
            raise RuntimeError('trace read failed')
    with pytest.raises(RuntimeError, match='trace read failed'):
        _window(_BrokenEpisodes([]), 'sess')


def test_system_prompt_injects_arc_review_and_closure():
    # The WRITE-side closing instructions come from the contract in §7.2
    # order (Arc → Review → closure LAST).
    from servers.scales.s1.encode import _build_system_prompt
    from servers.trace_contract import (render_journal_arc_block,
                                        render_journal_review_block,
                                        render_prompt_closure)
    prompt = _build_system_prompt('SYSTEM RULES HERE')
    assert render_journal_arc_block() in prompt
    assert render_journal_review_block() in prompt
    assert render_prompt_closure() in prompt
    assert prompt.rstrip().endswith(render_prompt_closure().rstrip())  # closure LAST
    assert prompt.index(render_journal_arc_block()) < prompt.index(
        render_journal_review_block())        # §7.2: Arc before Review
