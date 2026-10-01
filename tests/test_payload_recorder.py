"""Tests for the payload recorder (docs/TRACE-MODES-DESIGN.md, rollout step 1).

brain.record_payload / brain.read_payload — the ONE capture writer/reader.
Locks: db_dir-derived root (isolated by construction), per-kind K-store
gating with modes as config versions, append-only chain dirs (O_EXCL +
attempt ordinals), pointer safety, pruned-read semantics, unknown-kind
loudness, midnight chain-dir reuse.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestPayloadRecorder(BrainTestBase):
    needs_embedder = False

    def _abs(self, pointer):
        return os.path.join(self.tmp, pointer)

    # ── write + read ────────────────────────────────────────────

    def test_roundtrip_prompt(self):
        ptr = self.brain.record_payload('s1e-abc-3', 'prompt', '# The prompt')
        self.assertIsNotNone(ptr)
        self.assertFalse(os.path.isabs(ptr))
        self.assertTrue(ptr.startswith('payloads' + os.sep))
        self.assertTrue(ptr.endswith('000-prompt.md'))
        self.assertIn('s1e-abc-3', ptr)
        # File lives inside THIS brain's db_dir (isolated by construction)
        self.assertTrue(os.path.exists(self._abs(ptr)))
        self.assertEqual(self.brain.read_payload(ptr), '# The prompt')

    def test_dict_content_serialized_as_json(self):
        ptr = self.brain.record_payload(
            's1e-abc-4', 'failed_run', {'messages': [{'role': 'user'}]},
            seq=2)
        self.assertTrue(ptr.endswith('002-failed_run.json'))
        self.assertIn('"messages"', self.brain.read_payload(ptr))

    def test_empty_content_records_nothing(self):
        for empty in (None, '', {}, []):
            self.assertIsNone(
                self.brain.record_payload('s1e-abc-5', 'prompt', empty))

    # ── gating ──────────────────────────────────────────────────

    def test_normal_config_gates_round_payload_off(self):
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-6', 'round_payload', 'x'))

    def test_debug_override_opens_round_payload(self):
        """Entering debug is an ordinary override carrying
        TRACE_RECORDING_DEBUG (nothing pre-registers versions anymore);
        clear_interaction_override is the whole exit — no restart, no env."""
        import json
        from servers.trace_contract import TRACE_RECORDING_DEBUG
        reg = self.brain.register_interaction(
            'trace_recording', template='',
            parameters=json.dumps(TRACE_RECORDING_DEBUG))
        self.brain.set_interaction_active('trace_recording', reg['version'])
        ptr = self.brain.record_payload('s1e-abc-7', 'round_payload',
                                        {'round': 0})
        self.assertIsNotNone(ptr)
        # Clear the override: back on the NORMAL code default, gated again.
        self.brain.clear_interaction_override('trace_recording')
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-7', 'round_payload',
                                      {'round': 1}, seq=1))

    def test_unknown_kind_returns_none_and_logs_loud_once(self):
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-8', 'promt-typo', 'x'))
        self.assertIsNone(
            self.brain.record_payload('s1e-abc-8', 'promt-typo', 'x'))
        errs = [e for e in self.brain.get_recent_errors(hours=1, limit=50)
                if e.get('source') == 'record_payload_unknown_kind']
        self.assertEqual(len(errs), 1)

    def test_stale_config_missing_kind_falls_back_to_contract_default(self):
        """A kind added to the contract after a brain's config was seeded
        must resolve to its contract default (and log the gap), never
        silently not-record."""
        import json as _json
        reg = self.brain.register_interaction(
            'trace_recording', template='',
            parameters=_json.dumps({'kinds': {'prompt': True},
                                    'retention_days': 14}))
        self.brain.set_interaction_active('trace_recording', reg['version'])
        # failed_run is absent from the active config but ON in the contract
        # defaults — it must still record.
        ptr = self.brain.record_payload('s1e-abc-14', 'failed_run',
                                        {'error': 'x', 'messages': [1]})
        self.assertIsNotNone(ptr)
        errs = [e for e in self.brain.get_recent_errors(hours=1, limit=50)
                if e.get('source') == 'record_payload_config_missing_kind']
        self.assertGreaterEqual(len(errs), 1)

    def test_record_failed_run_owns_shape_and_skips_msgless(self):
        class Boom(Exception):
            pass
        err = Boom('it died')
        err.msgs = [{'role': 'user', 'content': 'hi'}]
        ptr = self.brain.record_failed_run('s1e-abc-15', err)
        self.assertTrue(ptr.endswith('000-failed_run.json'))
        body = self.brain.read_payload(ptr)
        self.assertIn('"it died"', body)
        self.assertIn('"messages"', body)
        # No msgs (round-0 / unwrapped failure) → nothing recorded.
        self.assertIsNone(self.brain.record_failed_run('s1e-abc-15',
                                                       Boom('bare')))

    def test_chain_id_is_sanitized_into_a_path_segment(self):
        """A traversal chain must not create ANY path outside its own
        chain dir — assert the exact dirs the unsanitized joins would have
        created (a deep-escape chain and the payloads-root 'evil' dir),
        so this test goes red if the sanitizer is deleted."""
        deep = '../' * 6 + 'escaped'  # enough .. to climb out of db_dir
        ptr = self.brain.record_payload(deep, 'prompt', 'x')
        self.assertIsNotNone(ptr)
        self.assertTrue(ptr.startswith('payloads' + os.sep))
        self.assertEqual(self.brain.read_payload(ptr), 'x')
        self.assertFalse(
            os.path.exists(os.path.join(os.path.dirname(self.tmp),
                                        'escaped')))
        ptr2 = self.brain.record_payload('s1e/../../evil', 'prompt', 'y')
        self.assertEqual(self.brain.read_payload(ptr2), 'y')
        self.assertFalse(
            os.path.exists(os.path.join(self.tmp, 'payloads', 'evil')))

    def test_dot_only_chain_stays_inside_date_layout(self):
        """'.' and '..' are path syntax, not names — unsanitized they land
        files where the date-pattern pruner never looks."""
        for chain in ('.', '..'):
            ptr = self.brain.record_payload(chain, 'prompt', 'dots')
            # Pointer shape: payloads/{date}/{chain}/{file} — 4 segments,
            # so the file is inside a date dir the pruner can reach.
            self.assertEqual(len(ptr.split(os.sep)), 4, ptr)
            self.assertEqual(self.brain.read_payload(ptr), 'dots')
        root_files = [f for f in os.listdir(
            os.path.join(self.tmp, 'payloads'))
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', f)]
        self.assertEqual(root_files, [])

    # ── append-only ─────────────────────────────────────────────

    def test_collision_gets_attempt_ordinal_never_overwrites(self):
        p1 = self.brain.record_payload('s1e-abc-9', 'prompt', 'attempt one')
        p2 = self.brain.record_payload('s1e-abc-9', 'prompt', 'attempt two')
        self.assertNotEqual(p1, p2)
        self.assertTrue(p2.endswith('000-prompt.2.md'))
        self.assertEqual(self.brain.read_payload(p1), 'attempt one')
        self.assertEqual(self.brain.read_payload(p2), 'attempt two')

    # ── pointer safety + pruned reads ───────────────────────────

    def test_read_rejects_absolute_and_traversal(self):
        secret = os.path.join(self.tmp, 'secret.txt')
        with open(secret, 'w') as f:
            f.write('nope')
        self.assertIsNone(self.brain.read_payload(secret))
        self.assertIsNone(self.brain.read_payload('../secret.txt'))
        self.assertIsNone(
            self.brain.read_payload('payloads/../secret.txt'))
        self.assertIsNone(self.brain.read_payload(''))
        self.assertIsNone(self.brain.read_payload(None))

    def test_deleted_file_reads_as_pruned_none(self):
        ptr = self.brain.record_payload('s1e-abc-10', 'prompt', 'gone soon')
        os.remove(self._abs(ptr))
        self.assertIsNone(self.brain.read_payload(ptr))

    # ── midnight reuse ──────────────────────────────────────────

    def test_existing_chain_dir_under_yesterday_is_reused(self):
        from datetime import datetime, timedelta
        from servers.clock import iso_now
        yesterday = (datetime.strptime(iso_now()[:10], '%Y-%m-%d')
                     - timedelta(days=1)).strftime('%Y-%m-%d')
        pre = os.path.join(self.tmp, 'payloads', yesterday, 's1e-abc-11')
        os.makedirs(pre)
        ptr = self.brain.record_payload('s1e-abc-11', 'prompt', 'late run')
        self.assertIn(yesterday, ptr)
        self.assertEqual(self.brain.read_payload(ptr), 'late run')

    # ── retention ───────────────────────────────────────────────

    def test_prune_removes_expired_date_dirs_only(self):
        old = os.path.join(self.tmp, 'payloads', '2026-01-01', 's1e-old-1')
        os.makedirs(old)
        with open(os.path.join(old, '000-prompt.md'), 'w') as f:
            f.write('ancient')
        keep_ptr = self.brain.record_payload('s1e-abc-13', 'prompt', 'fresh')
        removed = self.brain.prune_payloads_if_due()
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(old))
        self.assertEqual(self.brain.read_payload(keep_ptr), 'fresh')
        # Daily stamp: a second call the same day is a no-op.
        os.makedirs(os.path.join(self.tmp, 'payloads', '2026-01-02'))
        self.assertEqual(self.brain.prune_payloads_if_due(), 0)

    # ── recorder never raises ───────────────────────────────────

    def test_failure_is_loud_not_raised(self):
        # Make the payload root unwritable-as-a-dir by occupying its path
        # with a FILE — os.makedirs inside record_payload must fail.
        with open(os.path.join(self.tmp, 'payloads'), 'w') as f:
            f.write('blocker')
        ptr = self.brain.record_payload('s1e-abc-12', 'prompt', 'x')
        self.assertIsNone(ptr)
        errs = [e for e in self.brain.get_recent_errors(hours=1, limit=50)
                if e.get('source') == 'record_payload']
        self.assertEqual(len(errs), 1)


if __name__ == '__main__':
    unittest.main()
