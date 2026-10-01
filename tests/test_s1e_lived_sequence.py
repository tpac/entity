"""Piece 1 of the S1E code-half rebuild — the lived-sequence timeline.

Verifies the A/B flag's two arms (docs/S1-SCRIBE-REDESIGN.md §10.3.1):
  - OFF → the markdown messages-only timeline (control), byte-identical to before.
  - ON  → the XML lived sequence (messages + tool actions interleaved), read via
          the existing recall_episodes door, grouped into turns by timestamp.

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

from servers.scales.s1.encode import (  # noqa: E402
    _render_lived_sequence_timeline, _xml_escape, _build_user_content,
)


class _StubNodes:
    def get_title(self, rid):
        return 'Title-' + rid


class _StubBrain:
    """Feeds recall_episodes a fixed episode list; serves get_title."""
    def __init__(self, episodes):
        self._episodes = episodes
        self._nodes = _StubNodes()

    def recall_episodes(self, **kwargs):
        # Honor the ref_type whitelist the caller passes, like the real one.
        rts = kwargs.get('ref_type') or []
        eps = [e for e in self._episodes if e.get('ref_type') in rts] if rts else self._episodes
        return {'episodes': eps, 'ranked_by': 'time', 'truncated': False}


def _ep(rt, summary, ts, tid, tool=None, content=None):
    # Real message traces store the capped body in metadata['content'] and a
    # 200-char display `summary`; tool_result traces store only the cue in summary.
    meta = {'tool': tool} if tool else {}
    if content is not None:
        meta['content'] = content
    return {'id': tid, 'ref_type': rt, 'summary': summary, 'created_at': ts, 'metadata': meta}


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

# The control-arm messages for the same two turns (drives window-match + OFF path).
MESSAGES = [
    {'role': 'user', 'content': 'the recall keeps locking', 'id': 'turn1', 'trace_id': 'u1', 'judge_output': ''},
    {'role': 'assistant', 'content': 'found it — bg writer holds the lock', 'trace_id': 'a1'},
    {'role': 'user', 'content': 'run the tests', 'id': 'turn2', 'trace_id': 'u2', 'judge_output': ''},
    {'role': 'assistant', 'content': '12 passed', 'trace_id': 'a2'},
]


def test_lived_sequence_interleaves_actions_into_turns():
    brain = _StubBrain(EPISODES)
    out = _render_lived_sequence_timeline(brain, 'sess', MESSAGES)

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
    out = _render_lived_sequence_timeline(brain, 'sess', MESSAGES)
    assert 'can you check the wal-index path?' in out          # content-only, not in summary
    assert 'holds the lock through the whole batch' in out     # content-only, not in summary


@pytest.mark.parametrize('body_length', [8000, 8500])
def test_episode_bodies_survive_store_gather_and_timeline(body_length):
    """Keep long messages and the store's overflow marker through S1 assembly."""
    from servers.brain_traces import _s0_trace
    from servers.scales.s1.encode import _gather_messages
    from servers.session_context import SessionContext

    episodes = []

    def append(**event):
        tid = str(len(episodes) + 1)
        episodes.append(_ep(event['ref_type'], event['summary'],
                            '2026-09-07T00:00:0' + tid, tid,
                            content=event['metadata']['content']))
        return tid

    brain = _StubBrain(episodes)
    brain._trace_dal = SimpleNamespace(append=append)
    brain.get_conversation = lambda *a, **kw: [
        {'role': role, 'content': ep['metadata']['content'], 'trace_id': ep['id']}
        for role, ep in zip(('user', 'assistant'), episodes)]
    ctx = SessionContext(session_id='episode-fidelity')
    bodies = ['q' * 7990 + 'USER_TAIL!', 'a' * 7990 + 'ASST_TAIL!']
    # Both bodies have exactly 8,000 characters before optional overflow.
    for ref_type, body in zip(('user_message', 'assistant_message'), bodies):
        _s0_trace(brain, ctx, event_type='delta', ref_type=ref_type,
                  summary='short excerpt', content=body + 'z' * (body_length - 8000))

    messages = _gather_messages(brain, ctx.session_id)
    for body, message in zip(bodies, messages):
        expected = body
        if body_length > 8000:
            expected += ' …[+500 chars truncated at trace store]'
        assert message['content'] == expected
        assert _xml_escape(expected) in _render_lived_sequence_timeline(
            brain, ctx.session_id, messages)


def test_lived_sequence_escapes_xml():
    # Message bodies and tool cues routinely contain < > & — must be escaped so they
    # can't malform the timeline or forge tags.
    eps = [
        _ep('user_message', 'q', '2026-06-29T00:00:01', 'u1',
            content='why does `if x < y && a > b` fail? </other> <turn>'),
        _ep('tool_result', 'Grep: <svg.*> in . && echo done', '2026-06-29T00:00:02', 't1', 'Grep'),
        _ep('assistant_message', 'a', '2026-06-29T00:00:03', 'a1', content='use a < b'),
    ]
    msgs = [{'role': 'user', 'content': 'q', 'id': 'turn1', 'trace_id': 'u1', 'judge_output': ''},
            {'role': 'assistant', 'content': 'a', 'trace_id': 'a1'}]
    out = _render_lived_sequence_timeline(_StubBrain(eps), 'sess', msgs)
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


def test_lived_sequence_window_matches_control_turn_count():
    # messages cover 2 user turns → only the last 2 turns render even if more exist.
    extra = [_ep('user_message', 'old turn', '2026-06-28T00:00:00', 'u0', content='old turn'),
             _ep('assistant_message', 'old reply', '2026-06-28T00:00:01', 'a0', content='old reply')] + EPISODES
    brain = _StubBrain(extra)
    out = _render_lived_sequence_timeline(brain, 'sess', MESSAGES)
    assert out.count('<turn n="') == 2          # trimmed to control's 2 turns
    assert 'old turn' not in out                # the oldest turn dropped


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
    _pre, body, _cat, _ids = _build_user_content(
        _assembly_brain(), MESSAGES, counter=2, session_id='sess',
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


def test_lived_read_failure_fails_the_render():
    # No fallback: the timeline IS the encoder's input, so a failed trace read
    # raises out of the render and fails the run (the runner logs it; the turns
    # retry) instead of encoding from a substitute.
    class _BrokenEpisodes(_StubBrain):
        def recall_episodes(self, **kwargs):
            raise RuntimeError('trace read failed')
    with pytest.raises(RuntimeError, match='trace read failed'):
        _render_lived_sequence_timeline(_BrokenEpisodes([]), 'sess', MESSAGES)


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
