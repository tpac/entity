"""Piece 4 of the S1E code-half rebuild — residue wiring (the journal port).

The S1E journal blob → the `## Review` note contract, SESSION-BOUND. Covers:
  - the continuity READ side in _build_user_content (the notes prefix);
  - the write→read round-trip via the public doors, session-scoped (a run in
    session A never surfaces in session B);
  - the S1E continuity K default (5 runs, scale='s1').

The system-prompt injection (write-side instructions) is covered in
test_s1e_lived_sequence.py. See docs/S1-SCRIBE-REDESIGN.md §10.3.4.
"""
import os
import re

from tests.brain_test_base import BrainTestBase
from servers.scales.s1.encode import _build_user_content
from servers.trace_contract import session_short

# A parseable `## Review` section: a fenced ``` block of `tag · subject · note`
# lines (the format extract_review_block requires — the review-block instructions
# tell the encoder to emit exactly this).
_REVIEW = (
    "Some narrative the encoder wrote.\n\n"
    "## Review\n```\n"
    "watch · dedup-risk · unsure if the LAF node duplicates 9a3017ea\n"
    "```\n"
)


def _review(note_line):
    return "## Review\n```\n%s\n```\n" % note_line


def _turns():
    # A minimal one-turn window as encode._window hands it over; judge_output
    # empty so the catalog stays small.
    def row(tid, ref_type, text):
        return {'id': tid, 'ref_type': ref_type, 'chain_id': 's0-residue-0',
                'created_at': '2026-10-19T00:00:00+00:00', 'summary': text,
                'metadata': {'content': text}, 'content': text}
    user = dict(row('u1', 'user_message', 'do a thing'), judge_output='')
    return [{'user': user, 'with': [], 'assistant': row('a1', 'assistant_message', 'done'),
             'actions': []}]


class TestResidueWiring(BrainTestBase):
    needs_embedder = False

    def test_session_bound_roundtrip_via_public_doors(self):
        # Write a run's review notes scoped to session A; they read back for A,
        # and are WALLED from session B (the S1E session-bound continuity).
        self.brain.write_journal_notes(
            final_text=_REVIEW, chain_id='s1e-sessAxxx-5',
            scale='s1', session_id='sessA')

        a = self.brain.journal_notes(scale='s1', session_id='sessA')
        assert any('dedup-risk' in n['subject'] for n in a)
        assert any(n['tag'] == 'watch' for n in a)

        b = self.brain.journal_notes(scale='s1', session_id='sessB')
        assert b == []                      # session-walled: B sees nothing of A's

    def test_continuity_read_uses_notes(self):
        # _build_user_content injects the residue notes (self-labeled
        # 'RECENT REVIEW NOTES').
        sid = 'sess-lived'
        self.brain.write_journal_notes(
            final_text=_REVIEW, chain_id='s1e-%s-3' % session_short(sid),
            scale='s1', session_id=sid)
        _pre, body, _cat, _ids = _build_user_content(
            self.brain, _turns(), counter=8, session_id=sid)
        assert 'RECENT REVIEW NOTES' in body
        assert 'dedup-risk' in body

    def test_fresh_session_lived_has_no_continuity_block(self):
        # No prior notes this session → the continuity block is simply absent
        # (no 'first run' filler), and the encode still assembles.
        _pre, body, _cat, _ids = _build_user_content(
            self.brain, _turns(), counter=1, session_id='sess-fresh')
        assert 'RECENT REVIEW NOTES' not in body

    def test_lived_body_uses_xml_section_wrappers(self):
        # Fork 5: sections are wrapped in the XML labels the system prompt
        # names — <continuity>/<node_catalog>/<timeline>. The session arc
        # folds into <continuity>.
        sid = 'sess-xml'
        self.brain.set_config('session_context_%s' % sid, 'building S1E reconciliation')
        self.brain.write_journal_notes(
            final_text=_REVIEW, chain_id='s1e-%s-3' % session_short(sid),
            scale='s1', session_id=sid)
        pre, body, _cat, _ids = _build_user_content(
            self.brain, _turns(), counter=8, session_id=sid)
        assert '<continuity>' in body and '</continuity>' in body
        # prefix — the view policy stamps <timeline now="…">; this test's concern
        # is the XML wrappers, not the tag's attributes (exact shape + contents
        # pinned in test_s1e_lived_sequence).
        assert '<timeline' in body and '</timeline>' in body
        assert 'Session arc: building S1E' in body      # arc folded into continuity
        assert 'RECENT REVIEW NOTES' in body            # residue also in continuity
        # the preamble carries no section legend (the system prompt owns it —
        # two voices describing the layout would drift)
        assert 'Encoding Journal' not in pre and 'Conversation Timeline' not in pre

    def test_lived_timeline_stamp_resolves_against_a_real_brain(self):
        # The other half of the stamp: test_s1e_lived_sequence pins its SHAPE by
        # handing view_now in, which proves nothing about production being able to
        # DERIVE the time. This is the only real-brain assembly site, so it owns
        # the _conversation_now_safe wiring — if that regresses to None the tag
        # silently loses its temporal anchor and every relative `age=` below it
        # stops being invertible. Pattern, not a literal: the value is real time.
        _pre, body, _cat, _ids = _build_user_content(
            self.brain, _turns(), counter=8, session_id='sess-stamp')
        assert re.search(r'<timeline now="\d{4}-\d\d-\d\d \d\d:\d\d UTC">', body)

    def test_s1e_continuity_k_default_is_five(self):
        # S1E keeps the last 5 note-bearing runs of THIS session (the 's1e' K).
        sid = 'sess-k'
        for i in range(1, 7):  # 6 runs, one note each
            self.brain.write_journal_notes(
                final_text=_review("note · subj%d · run %d residue" % (i, i)),
                chain_id='s1e-%s-%d' % (session_short(sid), i), scale='s1', session_id=sid)
        notes = self.brain.journal_notes(scale='s1', session_id=sid)
        runs = {n['note'] for n in notes}
        assert 'run 1 residue' not in runs          # oldest dropped (K=5)
        assert 'run 6 residue' in runs
        assert len({n['subject'] for n in notes}) == 5


def _arc(line):
    return "narrative text\n\n## Arc\n```\n%s\n```\n" % line


class TestSessionArcWriteDoor(BrainTestBase):
    """write_session_arc — the journal mechanism's arc component (the v26
    arc-regression fix). Extracts the `## Arc` fence and ACCUMULATES it into
    session_context_{sid}, which the Frame's 'Current focus', the next run's
    context block, and recall ranking all read via session_context_for."""
    needs_embedder = False

    def test_arc_accumulates_into_session_context(self):
        sid = 'sess-arc'
        r1 = self.brain.write_session_arc(final_text=_arc('dashboard fix'),
                                          session_id=sid)
        assert r1 == {'written': True, 'status': 'ok'}
        r2 = self.brain.write_session_arc(final_text=_arc('judge moved to daemon'),
                                          session_id=sid)
        assert r2['written'] is True
        ctx = self.brain.session_context_for(sid)
        assert ctx == 'dashboard fix\njudge moved to daemon'   # journey, newest last

    def test_arc_is_session_walled(self):
        self.brain.write_session_arc(final_text=_arc('A-only movement'),
                                     session_id='sessA2')
        assert self.brain.session_context_for('sessB2') in ('', None)

    def test_empty_fence_is_legit_nothing_progressed(self):
        sid = 'sess-arc-empty'
        r = self.brain.write_session_arc(
            final_text="text\n\n## Arc\n```\n```\n", session_id=sid)
        assert r == {'written': False, 'status': 'empty_arc'}
        assert self.brain.session_context_for(sid) in ('', None)

    def test_missing_section_and_broken_fence_are_drift(self):
        # An opted-in encoder always emits `## Arc` — absence/broken fence is
        # drift, reported loud (status), never an exception.
        sid = 'sess-arc-drift'
        r = self.brain.write_session_arc(final_text='no arc at all DONE',
                                         session_id=sid)
        assert r == {'written': False, 'status': 'no_arc_section'}
        r = self.brain.write_session_arc(final_text='## Arc\nbare, no fence',
                                         session_id=sid)
        assert r == {'written': False, 'status': 'no_arc_extracted'}
        assert self.brain.session_context_for(sid) in ('', None)

    def test_multiline_fence_keeps_first_line_only(self):
        # The contract is ONE line; a drifting encoder that writes two must not
        # flood the digest.
        sid = 'sess-arc-multi'
        self.brain.write_session_arc(
            final_text="## Arc\n```\nreal movement\nspurious second line\n```",
            session_id=sid)
        assert self.brain.session_context_for(sid) == 'real movement'

    def test_limit_truncates_oldest_from_front(self):
        # Same rolling-journey shape as the legacy path: over-limit drops the
        # OLDEST lines; the newest movement always survives.
        sid = 'sess-arc-limit'
        self.brain.write_session_arc(final_text=_arc('x' * 50), session_id=sid,
                                     limit=40)
        self.brain.write_session_arc(final_text=_arc('newest movement'),
                                     session_id=sid, limit=40)
        ctx = self.brain.session_context_for(sid)
        assert len(ctx) <= 40
        assert ctx == 'newest movement'             # oldest line truncated away

    def test_arc_and_review_coexist_on_one_final_reply(self):
        # §7.2: the final reply carries Arc + Review; each write door consumes
        # only its own fence.
        sid = 'sess-arc-both'
        text = ('## Arc\n```\narc write-path built\n```\n\n'
                '## Review\n```\ndoubt · arc-fence · watch for drift\n```\nDONE')
        self.brain.write_session_arc(final_text=text, session_id=sid)
        self.brain.write_journal_notes(final_text=text,
                                       chain_id='s1e-%s-5' % session_short(sid),
                                       scale='s1', session_id=sid)
        assert self.brain.session_context_for(sid) == 'arc write-path built'
        notes = self.brain.journal_notes(scale='s1', session_id=sid)
        assert any(n['subject'] == 'arc-fence' for n in notes)
        # and the arc line never leaks into the notes
        assert not any('arc write-path built' in n['note'] for n in notes)

    def test_failed_lived_read_fails_the_run_before_any_trace(self):
        # The timeline is the encoder's input: when the trace read fails, the
        # run raises (the background runner logs it; the turns retry) — and it
        # raises BEFORE the O/K prompt traces or any write, so a failed run
        # never records a prompt it did not send.
        from unittest import mock
        from servers.scales.s1 import encode
        dispatched = []
        with mock.patch('servers.scales.runner.make_client', return_value=object()), \
                mock.patch.object(self.brain, 'conversation_turns',
                                  side_effect=RuntimeError('trace read failed')), \
                mock.patch.object(encode, 'run_llm_loop') as loop:
            with self.assertRaises(RuntimeError):
                encode.run_encoding(self.brain,
                                    lambda cmd, args: dispatched.append(cmd),
                                    8, 'sess-read-fail')
        assert dispatched == []
        loop.assert_not_called()
