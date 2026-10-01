"""Journal lifecycle verbs (audit finding #6) + boot standing items.

Read-time only — traces stay append-only:
  • `resolved · subject · why` drops older same-subject notes from the
    continuity prefix (normalized exact match). The resolve note itself
    stays until it ages out.
  • `open · subject · note` pins the newest note per subject beyond the
    K-run window until resolved; the READER computes ×N persistence
    (distinct runs mentioning the subject) — never the encoder.
  • Past JOURNAL_OPEN_NUDGE_RUNS the render nudges: resolve it, or hand it
    up as an `ask` — a Thalamus item the live work answers (the boot
    standing-items default ships empty; BRAIN_BOOT_INJECT_TYPES is the
    operator's own extension point).

The hotspot view (journal_notes(subject=...)) stays UNFILTERED — full
history for investigation.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


def _review(*lines):
    return '## Review\n```\n' + '\n'.join(lines) + '\n```\n'


class JournalLifecycleBase(BrainTestBase):
    needs_embedder = False

    UNIT = 'consolidation'

    def _write_run(self, n, *lines):
        """Write one run's notes under a distinct chain_id."""
        r = self.brain.write_journal_notes(
            final_text=_review(*lines),
            chain_id='s2-2026072800%04d-%s' % (n, self.UNIT),
            scale='s2')
        self.assertEqual(r['status'], 'ok')
        return r

    def _continuity(self, k=None):
        return self.brain.journal_notes(scale='s2', unit=self.UNIT, k=k)

    def _subjects(self, notes):
        return [n['subject'] for n in notes]


class TestResolveVerb(JournalLifecycleBase):

    def test_resolved_drops_older_same_subject(self):
        self._write_run(1, 'doubt · pin-42 · might be wrong')
        self._write_run(2, 'resolved · pin-42 · verified correct')

        notes = self._continuity(k=5)
        tags = [(n['tag'], n['subject']) for n in notes]
        self.assertIn(('resolved', 'pin-42'), tags)
        self.assertNotIn(('doubt', 'pin-42'), tags)

    def test_matching_is_normalized(self):
        self._write_run(1, 'doubt · Pin-42 · might be wrong')
        self._write_run(2, 'resolved ·   pin-42  · verified')

        notes = self._continuity(k=5)
        self.assertNotIn('Pin-42', self._subjects(notes))

    def test_other_subjects_untouched(self):
        self._write_run(1, 'doubt · pin-42 · might be wrong')
        self._write_run(2, 'resolved · other-thing · done')

        notes = self._continuity(k=5)
        self.assertIn('pin-42', self._subjects(notes))

    def test_resolve_does_not_reach_newer_notes(self):
        self._write_run(1, 'resolved · pin-42 · done')
        self._write_run(2, 'doubt · pin-42 · new doubt, after the resolve')

        notes = self._continuity(k=5)
        tags = [(n['tag'], n['subject']) for n in notes]
        self.assertIn(('doubt', 'pin-42'), tags)

    def test_hotspot_view_unfiltered(self):
        self._write_run(1, 'doubt · pin-42 · might be wrong')
        self._write_run(2, 'resolved · pin-42 · verified correct')

        history = self.brain.journal_notes(subject='pin-42', scale='s2',
                                           unit=self.UNIT)
        self.assertEqual(len(history), 2)


class TestOpenPins(JournalLifecycleBase):

    def test_open_survives_beyond_k_window(self):
        self._write_run(1, 'open · repo-question · still undecided')
        for i in range(2, 6):
            self._write_run(i, 'friction · run-%d · unrelated note' % i)

        notes = self._continuity(k=2)   # window holds runs 5,4 only
        opens = [n for n in notes if n['subject'] == 'repo-question']
        self.assertEqual(len(opens), 1)
        self.assertTrue(opens[0].get('open_runs'))

    def test_resolved_unpins(self):
        self._write_run(1, 'open · repo-question · still undecided')
        self._write_run(2, 'resolved · repo-question · promoted to abc12345')
        for i in range(3, 7):
            self._write_run(i, 'friction · run-%d · unrelated' % i)

        notes = self._continuity(k=2)
        self.assertNotIn('repo-question', self._subjects(notes))

    def test_open_runs_counts_distinct_runs(self):
        # Old-habit re-assertion across 3 runs → one line, ×3.
        for i in (1, 2, 3):
            self._write_run(i, 'open · repo-question · still undecided')
        self._write_run(4, 'friction · other · noise')

        notes = self._continuity(k=2)
        opens = [n for n in notes if n['subject'] == 'repo-question']
        self.assertEqual(len(opens), 1)          # deduped to newest
        self.assertEqual(opens[0]['open_runs'], 3)

    def test_still_open_alias_pins(self):
        self._write_run(1, 'still-open · legacy-item · from the wild corpus')
        for i in range(2, 6):
            self._write_run(i, 'friction · run-%d · unrelated' % i)

        notes = self._continuity(k=2)
        self.assertIn('legacy-item', self._subjects(notes))

    def test_pin_cap_bounds_carryover(self):
        # Exact-count both sides: the cap is hit (not zero pins vacuously),
        # and never exceeded. All open runs are pushed outside the k=2 window
        # by the trailing friction runs, so every pin is carry-over.
        from servers.trace_contract import JOURNAL_OPEN_PIN_CAP
        for i in range(JOURNAL_OPEN_PIN_CAP + 4):
            self._write_run(i + 1, 'open · item-%02d · lingering' % i)
        for j in range(30, 33):
            self._write_run(j, 'friction · run-%d · unrelated' % j)

        notes = self._continuity(k=2)
        pinned = [n for n in notes if n.get('open_runs')]
        self.assertEqual(len(pinned), JOURNAL_OPEN_PIN_CAP)

    def test_reopen_after_resolve_starts_fresh_epoch(self):
        # A resolve closes the epoch: a re-opened subject counts ×1 with a
        # fresh first_seen — runs retired by the resolution don't bleed into
        # the new count (review finding 3).
        self._write_run(1, 'open · repo-question · first epoch')
        self._write_run(2, 'open · repo-question · first epoch again')
        self._write_run(3, 'resolved · repo-question · settled for now')
        self._write_run(4, 'open · repo-question · re-opened, new grounds')
        self._write_run(5, 'friction · other · noise')

        notes = self._continuity(k=2)
        opens = [n for n in notes if n['subject'] == 'repo-question'
                 and n.get('open_runs')]
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0]['open_runs'], 1)
        self.assertEqual(opens[0]['note'], 're-opened, new grounds')


class TestRenderLifecycle(JournalLifecycleBase):

    def test_render_shows_count_and_nudge_at_threshold(self):
        """Past the threshold the nudge hands the item UP through the
        addressed verb — the Thalamus ask — never to a node type."""
        from servers.trace_contract import (render_journal_notes_prefix,
                                             JOURNAL_OPEN_NUDGE_RUNS)
        note = {'tag': 'open', 'subject': 'repo-question', 'note': 'undecided',
                'open_runs': JOURNAL_OPEN_NUDGE_RUNS,
                'first_seen': '2026-07-17T00:00:00+00:00'}
        from servers.trace_contract import JOURNAL_ASK_TAG, JOURNAL_NOTE_DELIMITER as D
        text = render_journal_notes_prefix([note])
        self.assertIn('open ×%d since 07-17' % JOURNAL_OPEN_NUDGE_RUNS, text)
        self.assertIn('hand it up: `%s %s repo-question %s <the question>`'
                      % (JOURNAL_ASK_TAG, D, D), text)
        # …and clear the pin — otherwise the nudge repeats every run while
        # the item already carries the question.
        self.assertIn('`resolved %s repo-question %s handed up`' % (D, D), text)

    def test_render_below_threshold_no_nudge(self):
        from servers.trace_contract import render_journal_notes_prefix
        note = {'tag': 'open', 'subject': 'repo-question', 'note': 'undecided',
                'open_runs': 2, 'first_seen': '2026-07-17T00:00:00+00:00'}
        text = render_journal_notes_prefix([note])
        self.assertIn('open ×2', text)
        self.assertNotIn('hand it up', text)

    def test_instruction_teaches_both_verbs(self):
        from servers.trace_contract import JOURNAL_REVIEW_INSTRUCTION
        from servers.trace_contract import (render_journal_review_block,
                                            JOURNAL_ADDRESSED_LIVE,
                                            JOURNAL_ADDRESSED_INSTRUCTION,
                                            JOURNAL_TELL_TAG, JOURNAL_ASK_TAG)
        # The addressed verbs are LIVE (one flag for every encoder): the one
        # block every encoder reads carries the paragraph between the `open`
        # line and the output-format close, and the render IS the constant.
        self.assertTrue(JOURNAL_ADDRESSED_LIVE)
        block = render_journal_review_block()
        self.assertEqual(block, JOURNAL_REVIEW_INSTRUCTION)
        self.assertIn(JOURNAL_ADDRESSED_INSTRUCTION, block)
        self.assertIn('`%s · subject · note`' % JOURNAL_TELL_TAG, block)
        self.assertLess(block.index('`%s · subject' % JOURNAL_ASK_TAG),
                        block.index('Put the notes under a `## Review`'))
        self.assertGreater(block.index('`%s · subject' % JOURNAL_ASK_TAG),
                           block.index('open · subject · note'))
        self.assertTrue(block.endswith('Stay sharp.'))
        self.assertIn('resolved · <its exact subject> · why', block)


class TestBootStandingItems(BrainTestBase):
    needs_embedder = False

    def test_default_injects_no_type(self):
        """The boot default ships EMPTY: a long-lived journal item reaches a
        human through the Thalamus ask (budgeted, expiring, answerable), not
        through a node type printed at every boot. The mechanism stays for
        operators who name their own types."""
        from servers.scales.s1.frame import (render_standing_items,
                                             BOOT_INJECT_TYPES_DEFAULT)
        self.assertEqual(BOOT_INJECT_TYPES_DEFAULT, '')
        self.brain.remember(type='some-standing-type',
                            title='repo-question — open 3 sessions',
                            content='a standing item of an unconfigured type',
                            encoding_source='encoder:sonnet')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('BRAIN_BOOT_INJECT_TYPES', None)
            self.assertEqual(render_standing_items(self.brain), '')

    def test_env_var_extends_types(self):
        from servers.scales.s1.frame import render_standing_items
        self.brain.remember(type='my-custom-boot-type', title='custom item',
                            content='c', encoding_source='anchor')
        with mock.patch.dict(os.environ,
                             {'BRAIN_BOOT_INJECT_TYPES':
                              'some-standing-type, my-custom-boot-type'}):
            text = render_standing_items(self.brain)
        self.assertIn('custom item', text)

    def test_empty_when_nothing_qualifies(self):
        from servers.scales.s1.frame import render_standing_items
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('BRAIN_BOOT_INJECT_TYPES', None)
            self.assertEqual(render_standing_items(self.brain), '')

    def test_archived_items_leave_the_boot(self):
        from servers.scales.s1.frame import render_standing_items
        r = self.brain.remember(type='my-custom-boot-type', title='handled item',
                                content='c', encoding_source='anchor')
        self.brain.archive_node(r['id'], archived_by='anchor', reason='handled')
        with mock.patch.dict(os.environ,
                             {'BRAIN_BOOT_INJECT_TYPES': 'my-custom-boot-type'}):
            self.assertNotIn('handled item', render_standing_items(self.brain))


if __name__ == '__main__':
    unittest.main()
