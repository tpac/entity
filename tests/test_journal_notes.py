"""brain.journal_notes() — the encoder-journal read door.

Reads journal_note traces through the public trace API (query_traces), never
raw SQL. Covers the two modes: subject pull (every note about one ref_id) and
continuity (notes from the last K note-bearing runs of an encoder, grouped by
chain_id). Part of the encoder-journal redesign Phase 2.
"""
from tests.brain_test_base import BrainTestBase
from servers.trace_contract import build_journal_note_metadata


class TestJournalNotesRead(BrainTestBase):
    needs_embedder = False

    def _note(self, chain_id, scale, ref_id, note, tag='', session_id=''):
        """Append one journal_note trace via the DAL (the write path #6 will
        wrap this; here we exercise the READ)."""
        self.brain._trace_dal.append(
            chain_id=chain_id, scale=scale, event_type='delta',
            ref_type='journal_note', ref_id=ref_id, summary=note[:80],
            metadata=build_journal_note_metadata(note=note, tag=tag),
            session_id=session_id)

    def test_subject_pull_returns_all_notes_about_subject(self):
        self._note('s2-20260101000001-consolidation', 's2', 'nodeA', 'first', tag='doubt')
        self._note('s2-20260101000002-consolidation', 's2', 'nodeA', 'second', tag='friction')
        self._note('s2-20260101000003-consolidation', 's2', 'nodeB', 'other')
        out = self.brain.journal_notes(subject='nodeA')
        assert {n['note'] for n in out} == {'first', 'second'}
        assert all(n['subject'] == 'nodeA' for n in out)
        assert {n['tag'] for n in out} == {'doubt', 'friction'}

    def test_continuity_keeps_only_last_k_runs(self):
        for i in range(1, 5):  # 4 runs, one note each, appended oldest→newest
            self._note('s2-2026010100000%d-consolidation' % i, 's2',
                       'node%d' % i, 'run %d' % i)
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=3)
        assert {n['note'] for n in out} == {'run 2', 'run 3', 'run 4'}  # oldest dropped

    def test_continuity_groups_multi_note_run_as_one(self):
        # An older run with TWO notes + a newer run with one; k=1 → only newest run.
        self._note('s2-20260101000001-consolidation', 's2', 'a', 'old1')
        self._note('s2-20260101000001-consolidation', 's2', 'b', 'old2')
        self._note('s2-20260101000002-consolidation', 's2', 'c', 'new1')
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=1)
        assert {n['note'] for n in out} == {'new1'}

    def test_unit_scoping_excludes_other_units_same_scale(self):
        self._note('s2-20260101000001-consolidation', 's2', 'a', 'consol-note')
        self._note('s2-20260101000001-community_detection', 's2', 'b', 'community-note')
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {n['note'] for n in out} == {'consol-note'}

    def test_continuity_k_defaults_from_contract(self):
        # No explicit k → JOURNAL_CONTINUITY_RUNS['consolidation'] == 3.
        for i in range(1, 6):  # 5 runs
            self._note('s2-2026010100000%d-consolidation' % i, 's2',
                       'n%d' % i, 'r%d' % i)
        out = self.brain.journal_notes(scale='s2', unit='consolidation')
        assert len({n['chain_id'] for n in out}) == 3  # default K

    def test_s1_continuity_scoped_by_session(self):
        self._note('s1e-aaaa-5', 's1', 'nodeX', 'sess-a note', session_id='sess-a')
        self._note('s1e-bbbb-5', 's1', 'nodeY', 'sess-b note', session_id='sess-b')
        out = self.brain.journal_notes(scale='s1', session_id='sess-a', k=5)
        assert {n['note'] for n in out} == {'sess-a note'}

    def test_empty_when_no_notes(self):
        assert self.brain.journal_notes(scale='s2', unit='consolidation') == []

    def test_underscore_unit_matches_literally(self):
        # The unit filter is a LIKE under the hood; a '_' in the unit name must
        # match literally, not as a single-char wildcard.
        self._note('s2-20260101000001-community_detection', 's2', 'a', 'real')
        self._note('s2-20260101000001-communityXdetection', 's2', 'b', 'decoy')
        out = self.brain.journal_notes(scale='s2', unit='community_detection', k=5)
        assert {n['note'] for n in out} == {'real'}

    def test_unit_filter_pushed_down_so_limit_bounds_per_unit(self):
        # One (older) community note + many (newer) consolidation notes. A small
        # limit must STILL surface the community note: the unit filter runs in
        # SQL, so LIMIT bounds the filtered set — not the global delta stream
        # (the old Python post-filter would have lost it behind the limit).
        self._note('s2-20260101000001-community_detection', 's2', 'c', 'community')
        for i in range(2, 8):
            self._note('s2-2026010100000%d-consolidation' % i, 's2',
                       'n%d' % i, 'consol %d' % i)
        out = self.brain.journal_notes(scale='s2', unit='community_detection', limit=2)
        assert {n['note'] for n in out} == {'community'}


class TestJournalNotesWrite(BrainTestBase):
    """brain.write_journal_notes() — the write door. Extracts the encoder's
    ## Review fenced block, parses it, writes journal_note rows. Mirror of the
    read door; round-trips through brain.journal_notes."""
    needs_embedder = False

    def test_write_then_read_roundtrip(self):
        final = (
            "Encoded 3 nodes, merged 1.\n\n"
            "## Review\n"
            "```\n"
            "friction · temporal-scout · misread a number again — 3rd run\n"
            "doubt · nodeA · merged but unsure the claims match\n"
            "```\n"
        )
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000005-consolidation', scale='s2')
        assert r['written'] == 2 and r['status'] == 'ok'
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['subject'] for x in out} == {'temporal-scout', 'nodeA'}
        assert {x['tag'] for x in out} == {'friction', 'doubt'}

    def test_no_review_section_warns_loud(self):
        # No `## Review` at all → loud signal (not a silent 0), distinct status.
        r = self.brain.write_journal_notes(
            final_text="Encoded 3 nodes. No review here.",
            chain_id='s2-20260101000006-consolidation', scale='s2')
        assert r == {'written': 0, 'malformed': 0, 'status': 'no_review_section',
                     'addressed': [], 'resolved': []}

    def test_marker_present_but_no_fence_warns(self):
        # `## Review` present but no fenced block → format drift, distinct status.
        r = self.brain.write_journal_notes(
            final_text="## Review\nfriction · nodeA · misread (no fence!)\n",
            chain_id='s2-20260101000066-consolidation', scale='s2')
        assert r['status'] == 'no_review_extracted' and r['written'] == 0

    def test_empty_fenced_review_is_clean_not_drift(self):
        # `## Review` with an empty fence = a legit clean run, NOT drift.
        r = self.brain.write_journal_notes(
            final_text="## Review\n```\n```\n",
            chain_id='s2-20260101000067-consolidation', scale='s2')
        assert r == {'written': 0, 'malformed': 0, 'status': 'empty_review',
                     'addressed': [], 'resolved': []}

    def test_prose_outside_fence_not_parsed(self):
        # Review #4: prose before the fence with a stray '·' must NOT become a
        # malformed note — only the fenced block is parsed.
        final = (
            "## Review\n"
            "Some reflection · with a stray middot in prose, not a note.\n"
            "```\n"
            "surprise · recall-ranking · IDF boost helped, unexpectedly\n"
            "```\n"
        )
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000007-consolidation', scale='s2')
        assert r['written'] == 1
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['note'] for x in out} == {'IDF boost helped, unexpectedly'}

    def test_malformed_line_in_fence_isolated(self):
        # One good note + one delimiter-less line: the good note is written, the
        # malformed one skipped (logged loud), the batch survives.
        final = (
            "## Review\n"
            "```\n"
            "doubt · nodeX · a real note\n"
            "this line has no delimiter and is malformed\n"
            "```\n"
        )
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000008-consolidation', scale='s2')
        assert r['written'] == 1 and r['malformed'] == 1
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['note'] for x in out} == {'a real note'}

    def test_bullet_prefixed_notes_write_clean_tags(self):
        # LLMs love bullet lists; a leading '- '/'* ' must not pollute the tag.
        final = ("## Review\n```\n"
                 "- friction · nodeA · misread again\n"
                 "* doubt · nodeB · unsure the merge is coherent\n"
                 "```\n")
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000009-consolidation', scale='s2')
        assert r['written'] == 2
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['tag'] for x in out} == {'friction', 'doubt'}   # not '- friction'

    def test_headingless_valid_fence_salvaged(self):
        # Observed Haiku drift (community batch 2, chain s2-20260729180005):
        # a perfectly formed notes fence with the `## Review` heading dropped.
        # Salvage harvests it — status 'salvaged', still a (loud) warning.
        final = (
            "All eight proposals acted:\n"
            "- **[1-6]** connected as community members.\n\n"
            "```\n"
            "resolved · proposal_1_2_3 · six members connected across three communities\n"
            "drift_none_detected · proposals_4_5 · all placements above threshold\n"
            "```\n"
        )
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000010-consolidation', scale='s2')
        assert r['written'] == 2 and r['status'] == 'salvaged'
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['subject'] for x in out} == {'proposal_1_2_3', 'proposals_4_5'}

    def test_headingless_code_fence_not_salvaged(self):
        # A code fence must never be harvested as notes — strict gate: every
        # line must parse, so one delimiter-less line disqualifies the fence.
        final = ("Rejected everything.\n\n"
                 "```\n"
                 "def foo():\n"
                 "    return 1\n"
                 "```\n")
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000011-consolidation', scale='s2')
        assert r == {'written': 0, 'malformed': 0, 'status': 'no_review_section',
                     'addressed': [], 'resolved': []}

    def test_headingless_mixed_fence_not_salvaged(self):
        # One valid note + one malformed line → all-or-nothing gate rejects.
        final = ("```\n"
                 "doubt · nodeX · a real note\n"
                 "this line has no delimiter\n"
                 "```\n")
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000012-consolidation', scale='s2')
        assert r['status'] == 'no_review_section' and r['written'] == 0

    def test_headingless_empty_fence_not_salvaged(self):
        # An empty heading-less fence is indistinguishable from a stray code
        # block — NOT treated as a clean run (that requires the heading).
        r = self.brain.write_journal_notes(
            final_text="Nothing to do.\n```\n```\n",
            chain_id='s2-20260101000013-consolidation', scale='s2')
        assert r['status'] == 'no_review_section' and r['written'] == 0

    def test_salvage_takes_last_qualifying_fence(self):
        # The closure puts the review at the end — with several qualifying
        # fences, the LAST one is the review.
        final = ("```\n"
                 "old · nodeA · earlier fence, not the review\n"
                 "```\n"
                 "Some prose between.\n"
                 "```\n"
                 "resolved · nodeB · the actual review notes\n"
                 "```\n")
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000014-consolidation', scale='s2')
        assert r['written'] == 1 and r['status'] == 'salvaged'
        out = self.brain.journal_notes(scale='s2', unit='consolidation', k=5)
        assert {x['subject'] for x in out} == {'nodeB'}

    def test_arc_fence_not_mistaken_for_review(self):
        # A well-formed `## Arc` section is stripped before the salvage scan —
        # its fence must never be harvested as review notes, even when the
        # arc line happens to parse as one.
        final = ("## Arc\n"
                 "```\n"
                 "milestone · nodeC · arc line that looks like a note\n"
                 "```\n"
                 "No review this run.\n")
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000015-consolidation', scale='s2')
        assert r['status'] == 'no_review_section' and r['written'] == 0

    def test_broken_review_section_with_valid_fence_elsewhere_salvaged(self):
        # Notes fenced BEFORE a fenceless `## Review` heading: the extractor
        # finds no fence after the marker (drift), but salvage recovers the
        # notes from the earlier fence.
        final = ("```\n"
                 "resolved · nodeD · notes landed before the heading\n"
                 "```\n"
                 "## Review\n"
                 "(forgot the fence here)\n")
        r = self.brain.write_journal_notes(
            final_text=final, chain_id='s2-20260101000016-consolidation', scale='s2')
        assert r['written'] == 1 and r['status'] == 'salvaged'


class TestJournalNotesAddressed(BrainTestBase):
    """The write door PARTITIONS: tell/ask lines come back under 'addressed'
    (messages to the people working, not residue — the binding routes them)
    and this run's resolve lines under 'resolved'; the door holds no routing
    policy and returns one shape on every path."""
    needs_embedder = False

    TEXT = ('## Review\n```\n'
            'friction · abc12345 · drift\n'
            'tell · segment 6.a · confirm first\n'
            'ask · 7e6decd2 · revise, or leave?\n'
            'resolved · 9a9a9a9a · done\n'
            '```\n')

    def test_addressed_lines_returned_not_written(self):
        r = self.brain.write_journal_notes(
            final_text=self.TEXT, chain_id='s1e-aaaaaaaa-1', scale='s1',
            session_id='sess-a')
        self.assertEqual((r['written'], r['malformed'], r['status']),
                         (2, 0, 'ok'))
        self.assertEqual([(n['tag'], n['subject']) for n in r['addressed']],
                         [('tell', 'segment 6.a'), ('ask', '7e6decd2')])
        self.assertEqual([(n['tag'], n['subject'], n['note'])
                          for n in r['resolved']],
                         [('resolved', '9a9a9a9a', 'done')])
        tags = sorted(n['tag'] for n in
                      self.brain.journal_notes(scale='s1', session_id='sess-a'))
        self.assertEqual(tags, ['friction', 'resolved'])

    def test_two_field_addressed_line_is_about_the_run(self):
        """`tell · message` (no subject — the shape a model reaches for) is
        a message about the run itself, never a residue note whose subject
        is the word 'tell'."""
        from servers.trace_contract import (parse_journal_notes,
                                            JOURNAL_RUN_SUBJECT)
        notes, malformed = parse_journal_notes(
            'tell · you are proceeding on "I wonder if", not a yes\n'
            'ask · 7e6decd2\n')
        self.assertEqual(malformed, [])
        self.assertEqual(
            [(n['tag'], n['subject'], n['note']) for n in notes],
            [('tell', JOURNAL_RUN_SUBJECT,
              'you are proceeding on "I wonder if", not a yes'),
             ('ask', JOURNAL_RUN_SUBJECT, '7e6decd2')])
        r = self.brain.write_journal_notes(
            final_text='## Review\n```\ntell · confirm first\n```\n',
            chain_id='s1e-aaaaaaaa-5', scale='s1', session_id='sess-e')
        self.assertEqual([n['subject'] for n in r['addressed']],
                         [JOURNAL_RUN_SUBJECT])
        self.assertEqual(r['written'], 0)

    def test_one_shape_on_every_path(self):
        r = self.brain.write_journal_notes(
            final_text='## Review\n```\n```\n', chain_id='s1e-aaaaaaaa-3',
            scale='s1', session_id='sess-c')
        self.assertEqual(r, {'written': 0, 'malformed': 0,
                             'status': 'empty_review',
                             'addressed': [], 'resolved': []})

    def test_undelivered_rides_the_note_row_and_renders(self):
        """A rejected addressed line kept as residue carries the door's
        reason as a FIELD — the note stays the line the encoder wrote and the
        reason cannot be eaten by the note cap."""
        from servers.trace_contract import render_journal_notes_prefix
        n = self.brain.write_journal_note_rows(
            [{'tag': 'ask', 'subject': '7e6decd2', 'note': 'revise, or leave?',
              'undelivered': 'thalamus budget: cap 8'}],
            chain_id='s1e-aaaaaaaa-4', scale='s1', session_id='sess-d')
        self.assertEqual(n, 1)
        notes = self.brain.journal_notes(scale='s1', session_id='sess-d')
        self.assertEqual((notes[0]['note'], notes[0]['undelivered']),
                         ('revise, or leave?', 'thalamus budget: cap 8'))
        self.assertIn('— not delivered: thalamus budget: cap 8',
                      render_journal_notes_prefix(notes))


class TestJournalNotesRecallGuard(BrainTestBase):
    """journal_note must never leak into Anchor's recall. Structural guard:
    EAGER_TRACE_SCALES=('s0',) means s1/s2 traces are never embedded, so they
    can't surface in recall() (node + s0-trace vectors) or recall_episodes()
    (s0-scoped). Locked here so a future scope change can't silently break it."""
    needs_embedder = False

    def test_journal_note_scales_never_embedded(self):
        from servers.embed_queue import EAGER_TRACE_SCALES
        # journal_note rides on s1/s2; the embed worker only ever pulls s0.
        assert EAGER_TRACE_SCALES == ('s0',)
        assert 's1' not in EAGER_TRACE_SCALES and 's2' not in EAGER_TRACE_SCALES

    def test_journal_note_absent_from_recall_episodes(self):
        # A real s2 journal_note with a unique marker must not surface in
        # recall_episodes (which scans s0 only).
        self.brain._trace_dal.append(
            chain_id='s2-20260101000009-consolidation', scale='s2',
            event_type='delta', ref_type='journal_note', ref_id='nodeZ',
            summary='zebra-unique-marker',
            metadata=build_journal_note_metadata(note='zebra-unique-marker note'))
        res = self.brain.recall_episodes(contains='zebra-unique-marker')
        assert 'zebra-unique-marker' not in str(res)


class TestGatingExcludesResidue(BrainTestBase):
    """S2 idle-gating counts REAL integration deltas, not residue notes — so a
    run that only journaled doesn't read as a completed integration. Routed
    through query_traces(exclude_ref_types=...), not raw SQL (#11)."""
    needs_embedder = False

    def _unit(self):
        from servers.scales.s2.base import IntegrationUnit

        class U(IntegrationUnit):
            NAME = 'consolidation'
            SCALE = 's2'
        return U(self.brain)

    def test_query_traces_exclude_ref_types(self):
        ta = self.brain._trace_dal
        ta.append(chain_id='s2-20260101000010-consolidation', scale='s2',
                  event_type='delta', ref_type='consolidated', summary='ran')
        ta.append(chain_id='s2-20260101000010-consolidation', scale='s2',
                  event_type='delta', ref_type='journal_note', ref_id='nodeA',
                  summary='note', metadata=build_journal_note_metadata(note='a note'))
        events = self.brain.query_traces(
            scale='s2', event_type='delta', chain_suffix='consolidation',
            exclude_ref_types=['journal_note'], hours=None)['events']
        assert {e['ref_type'] for e in events} == {'consolidated'}

    def test_last_run_timestamp_excludes_journal_note(self):
        import time
        ta = self.brain._trace_dal
        ta.append(chain_id='s2-20260101000010-consolidation', scale='s2',
                  event_type='delta', ref_type='consolidated', summary='ran')
        ops_ts = self.brain.query_traces(
            ref_type='consolidated', scale='s2',
            chain_suffix='consolidation', hours=None)['events'][0]['created_at']
        time.sleep(0.01)  # journal_note lands strictly later
        ta.append(chain_id='s2-20260101000010-consolidation', scale='s2',
                  event_type='delta', ref_type='journal_note', ref_id='nodeA',
                  summary='note', metadata=build_journal_note_metadata(note='a later note'))
        # Must be the ops-delta's time, NOT the later journal_note's.
        assert self._unit()._last_run_timestamp() == ops_ts

    def test_last_run_timestamp_empty_when_only_notes(self):
        self.brain._trace_dal.append(
            chain_id='s2-20260101000011-consolidation', scale='s2',
            event_type='delta', ref_type='journal_note', ref_id='nodeB',
            summary='note', metadata=build_journal_note_metadata(note='only a note'))
        assert self._unit()._last_run_timestamp() == ''  # no real integration yet
