"""Working continuity: committed lifecycle, bounded context, unchanged history."""
from unittest.mock import patch
import os

from tests.brain_test_base import BrainTestBase
from servers.scales.journal import JournalBinding
from servers.scales.s1.scribe import S1Scribe
from servers.scales.s2.community_encoder import CommunityEncoder
from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
from servers.scales.s2.healer_encoder import HealerEncoder
from servers.scales.s2.aspect_encoder import AspectEncoder
from servers.trace_contract import (JOURNAL_VIEW_MAX_CHARS,
                                    render_journal_view, parse_journal_notes)


def review(*lines):
    return '## Review\n```\n' + '\n'.join(lines) + '\n```'


class TestJournalContinuity(BrainTestBase):
    needs_embedder = False

    def binding(self, unit='community_detection', session_id=''):
        return JournalBinding(self.brain, scale='s1' if session_id else 's2',
                              unit='' if session_id else unit,
                              session_id=session_id)

    def write(self, binding, chain, *lines):
        binding.harvest(review(*lines), chain)

    def test_all_five_bindings_share_lifecycle_and_next_run_selection(self):
        units = [CommunityEncoder(self.brain), ConsolidationEncoder(self.brain),
                 HealerEncoder(self.brain), AspectEncoder(self.brain),
                 S1Scribe(self.brain, 'session-a', 1)]
        for unit in units:
            with self.subTest(unit=unit.NAME):
                b = unit.journal
                before = ('s1e-session-0' if unit.SCALE == 's1'
                          else 's2-before-' + unit.NAME)
                self.write(b, before, 'open · deployment · old pending claim',
                           'doubt · unrelated · original private observation')
                chain = unit.chain_id()
                self.assertIn('old pending claim', b.continuity(chain_id=chain))
                self.write(b, chain, 'resolved · deployment · deployment confirmed',
                           'open · new-item · same-run private note',
                           'doubt · unrelated · new thought about old subject')
                text = b.continuity(chain_id=chain)
                self.assertIn('resolved · deployment · deployment confirmed', text)
                self.assertNotIn('old pending claim', text)
                self.assertNotIn('same-run private note', text)
                self.assertNotIn('new thought about old subject', text)
                self.assertIn('original private observation', text)
                # A new invocation on a reused binding admits new residue.
                text = b.continuity(chain_id=chain + '-next')
                self.assertIn('same-run private note', text)
                self.assertIn('new thought about old subject', text)

    def test_compact_recent_opens_without_losing_distinct_observations(self):
        b = self.binding()
        for i in range(3):
            self.write(b, 's2-%d-community_detection' % i,
                       'open · item · repeated assertion',
                       'open · item · repeated assertion',
                       'doubt · item · uncertainty',
                       'friction · item · different observation')
        history = self.brain.journal_notes(subject='item', scale='s2')
        self.assertEqual(len(history), 12)
        text = b.continuity(chain_id='current')
        self.assertEqual(text.count('repeated assertion'), 1)
        self.assertEqual(text.count('uncertainty'), 1)
        self.assertEqual(text.count('different observation'), 1)
        self.assertIn('Persistence: ×3', text)
        line = next(line for line in text.splitlines() if line.startswith('- open'))
        self.assertEqual(parse_journal_notes(line)[0][0]['tag'], 'open')
        self.assertEqual(self.brain.journal_notes(subject='item', scale='s2'), history)

    def test_unchanged_refresh_keeps_old_resolution_outside_run_window(self):
        b = self.binding()
        self.write(b, 's2-0-community_detection', 'resolved · item · old closure')
        for i, subject in enumerate(('other-a', 'other-b', 'item'), start=1):
            self.write(b, 's2-%d-community_detection' % i,
                       'doubt · %s · current observation' % subject)
        first = b.continuity(chain_id='run')
        self.assertNotIn('old closure', first)
        self.assertEqual(b.continuity(chain_id='run'), first)
        self.write(b, 's2-4-community_detection', 'resolved · item · new closure')
        updated = b.continuity(chain_id='run')
        self.assertIn('new closure', updated)
        self.assertNotIn('old closure', updated)
        self.assertEqual(b.continuity(chain_id='run'), updated)

    def test_unchanged_refresh_keeps_old_open_beyond_pin_cap(self):
        from servers.trace_contract import JOURNAL_OPEN_PIN_CAP

        b = self.binding()
        # Three newest runs are in the window; more than ten older open
        # subjects overflow the pin cap. The oldest subject is selected
        # only through its newest ordinary observation.
        for i in range(JOURNAL_OPEN_PIN_CAP + 3):
            self.write(b, 's2-%d-community_detection' % i,
                       'open · item-%d · old open %d' % (i, i))
        self.write(b, 's2-latest-community_detection',
                   'doubt · item-0 · current observation')
        first = b.continuity(chain_id='run')
        self.assertIn('current observation', first)
        self.assertNotIn('old open 0', first)
        self.assertEqual(b.continuity(chain_id='run'), first)

    def test_healer_and_consolidation_real_batch_entry_points_refresh(self):
        from servers.scales.s2.healer_contract import HEALER
        from servers.scales.s2.consolidation_contract import CONSOLIDATION
        units = [HealerEncoder(self.brain, config={**HEALER, 'max_nodes_per_call': 1}),
                 ConsolidationEncoder(self.brain, config={
                     **CONSOLIDATION, 'max_proposals_per_call': 1})]
        for unit in units:
            with self.subTest(unit=unit.NAME):
                self.write(unit.journal, 's2-prior-' + unit.NAME,
                           'open · deployment · pending before batch one')
                seen = []

                def response(user):
                    seen.append(user)
                    return review('resolved · deployment · confirmed by first batch',
                                  'doubt · fresh · new private thought')

                def once(client, model, max_tokens, system, user, effort=None):
                    return '[]\n' + response(user), {}

                def loop(**kw):
                    return {'rounds': 1, 'actions': 0, 'write_actions': 0,
                            'final_text': response(kw['user_content'])}

                with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-not-a-key'}), \
                        patch('servers.scales.s2.base.make_client', return_value=object()), \
                        patch('servers.scales.s2.base.run_llm_once', side_effect=once), \
                        patch('servers.scales.runner.make_client', return_value=object()), \
                        patch('servers.scales.runner.run_llm_loop', side_effect=loop):
                    if unit.NAME == 'healer':
                        result = unit.run([{'node_id': 'aaaaaaaa'}, {'node_id': 'bbbbbbbb'}])
                        self.assertEqual(result['errors'], [])
                    else:
                        with patch.object(unit, '_format_clusters', return_value='clusters'):
                            unit._encode([{'nodes': []}, {'nodes': []}])
                self.assertEqual(len(seen), 2)
                self.assertIn('pending before batch one', seen[0])
                self.assertNotIn('pending before batch one', seen[1])
                self.assertIn('confirmed by first batch', seen[1])
                self.assertNotIn('new private thought', seen[1])

    def test_reopen_is_new_epoch_and_repeated_batches_count_once(self):
        b = self.binding()
        self.write(b, 's2-0-community_detection', 'open · item · first epoch')
        b.continuity(chain_id='run')
        self.write(b, 's2-1-community_detection', 'resolved · item · confirmed')
        self.assertNotIn('first epoch', b.continuity(chain_id='run'))
        self.write(b, 's2-2-community_detection', 'open · item · new evidence')
        self.write(b, 's2-2-community_detection', 'open · item · new evidence')
        text = b.continuity(chain_id='run')
        self.assertEqual(text.count('new evidence'), 1)
        self.assertIn('Persistence: ×1', text)
        self.assertNotIn('resolved · item', text)

    def test_typo_stays_distinct_and_inversion_recovery_still_works(self):
        b = self.binding()
        self.write(b, 's2-0-community_detection',
                   'open · 482d2a1c_deploy_pending · pending')
        b.continuity(chain_id='run')
        self.write(b, 's2-1-community_detection',
                   'resolved · 482a2a1c_deploy_pending · wrong spelling')
        self.assertIn('pending', b.continuity(chain_id='run'))
        self.write(b, 's2-2-community_detection',
                   'resolved · open · 482d2a1c_deploy_pending · verified')
        text = b.continuity(chain_id='run')
        self.assertIn('resolved · 482d2a1c_deploy_pending', text)
        self.assertNotIn('open · 482d2a1c_deploy_pending', text)

    def test_session_and_unit_isolation_including_foreign_receipt(self):
        bindings = [self.binding(session_id='a'), self.binding(session_id='b'),
                    self.binding('community_detection'), self.binding('healer')]
        for i, b in enumerate(bindings):
            chain = ('s1e-%s-0' % b.session_id if b.session_id
                     else 's2-0-' + b.unit)
            self.write(b, chain, 'open · shared-name · pending-%d' % i)
            self.assertIn('pending-%d' % i, b.continuity(chain_id='run'))
        self.write(bindings[0], 's1e-a-1', 'resolved · shared-name · confirmed')
        for i, b in enumerate(bindings):
            text = b.continuity(chain_id='run')
            self.assertEqual('confirmed' in text, i == 0)
        with self.assertRaises(ValueError):
            self.brain.journal_view(scale='s1', session_id='b',
                                    previous=bindings[0]._view)
        with self.assertRaises(ValueError):
            self.brain.journal_view(scale='s1')

    def test_failed_write_never_updates_working_state(self):
        b = self.binding()
        self.write(b, 's2-0-community_detection', 'open · item · pending')
        b.continuity(chain_id='run')
        with patch.object(self.brain, 'write_journal_note_rows',
                          side_effect=RuntimeError('write failed')):
            self.write(b, 's2-1-community_detection', 'resolved · item · lost write')
        text = b.continuity(chain_id='run')
        self.assertIn('open · item · pending', text)
        self.assertNotIn('lost write', text)

    def test_read_failure_retains_state_and_recovers_without_skipping(self):
        b = self.binding()
        self.write(b, 's2-0-community_detection', 'open · item · pending')
        b.continuity(chain_id='run')
        self.write(b, 's2-1-community_detection', 'resolved · item · confirmed')
        with patch.object(self.brain, 'journal_view', side_effect=RuntimeError('offline')):
            text = b.continuity(chain_id='run')
        self.assertIn('pending', text)
        self.assertIn('Latest journal read failed', text)
        text = b.continuity(chain_id='run')
        self.assertIn('confirmed', text)
        self.assertNotIn('Latest journal read failed', text)

    def test_equal_timestamp_pages_drain_without_echoing_new_subjects(self):
        b = self.binding()
        self.write(b, 's2-0-community_detection', 'open · item · pending')
        b.continuity(chain_id='run')
        # All four notes share one append timestamp. Page size two puts the
        # resolution across the boundary; a timestamp-only cursor loses it.
        self.write(b, 's2-1-community_detection',
                   'doubt · fresh-a · not for this run',
                   'doubt · fresh-b · not for this run',
                   'resolved · item · confirmed',
                   'doubt · fresh-c · not for this run')
        with patch('servers.trace_contract.JOURNAL_VIEW_PAGE_SIZE', 2), \
                patch('servers.trace_contract.JOURNAL_VIEW_MAX_PAGES', 1):
            text = b.continuity(chain_id='run')
            self.assertIn('updates remain unread', text)
            self.assertIn('pending', text)
            text = b.continuity(chain_id='run')
        self.assertIn('confirmed', text)
        self.assertNotIn('updates remain unread', text)
        self.assertNotIn('not for this run', text)

    def test_initial_read_failure_does_not_admit_this_runs_new_notes(self):
        b = self.binding()
        with patch.object(self.brain, 'journal_view', side_effect=RuntimeError('offline')):
            self.assertIn('Initial journal selection unavailable',
                          b.continuity(chain_id='run'))
        self.write(b, 's2-1-community_detection', 'doubt · fresh · new private thought')
        self.assertNotIn('new private thought', b.continuity(chain_id='run'))
        self.assertIn('new private thought', b.continuity(chain_id='next-run'))

    def test_bad_render_is_failure_isolated(self):
        b = self.binding()
        with patch.object(self.brain, 'journal_view', return_value={
                'notes': [{'subject': 'bad', 'tag': object(), 'note': 'malformed'}]}):
            self.assertIn('Journal rendering failed', b.continuity(chain_id='run'))

    def test_render_budget_omits_whole_rows_and_names_partial_coverage(self):
        notes = [{'tag': 'open', 'subject': 'item-%d' % i,
                  'note': 'x' * 600, 'open_runs': 6} for i in range(40)]
        notes.insert(0, {'tag': 'doubt', 'subject': 'huge-' + 's' * 9000,
                         'note': 'oversized subject'})
        text, stats = render_journal_view({'notes': notes, 'history_truncated': True})
        self.assertLessEqual(len(text), JOURNAL_VIEW_MAX_CHARS)
        self.assertGreater(stats['omitted_rows'], 0)
        self.assertIn('entries omitted', text)
        self.assertIn('Older journal history', text)
        self.assertNotIn('huge-', text)
        self.assertIn('open · item-0', text)
        self.assertEqual(len(notes), 41)
