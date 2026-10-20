"""Payload readers for TRACE-MODES rollout step 2 — writers record via
brain.record_payload; the dashboard reads pointer-carrying rows and refuses
absolute ref_ids; the runner's record_round_fn round-trips through
brain.round_recorder; a failed encode leaves residue the next run can see.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.clock import iso_now
from tests.brain_test_base import BrainTestBase


class TestDashboardPayloadHelpers(unittest.TestCase):
    """dashboard/db.py — the sanctioned direct payload reader."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix='dashpay-')
        self._prev = os.environ.get('BRAIN_DB_DIR')
        os.environ['BRAIN_DB_DIR'] = self.tmp
        chain = os.path.join(self.tmp, 'payloads', '2026-08-03', 's2-x-consolidation')
        os.makedirs(chain)
        with open(os.path.join(chain, '001-prompt.md'), 'w') as f:
            f.write('BATCH ONE')
        with open(os.path.join(chain, '002-prompt.md'), 'w') as f:
            f.write('BATCH TWO')
        with open(os.path.join(chain, '000-round_payload.json'), 'w') as f:
            f.write('{}')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        if self._prev is None:
            os.environ.pop('BRAIN_DB_DIR', None)
        else:
            os.environ['BRAIN_DB_DIR'] = self._prev

    def test_read_payload_pointer_guards(self):
        from dashboard import db
        ptr = os.path.join('payloads', '2026-08-03',
                           's2-x-consolidation', '001-prompt.md')
        self.assertEqual(db.read_payload_pointer(ptr), 'BATCH ONE')
        self.assertIsNone(db.read_payload_pointer('/etc/passwd'))
        self.assertIsNone(db.read_payload_pointer('payloads/../secret'))
        self.assertIsNone(db.read_payload_pointer('payloads/2026/none.md'))
        self.assertIsNone(db.read_payload_pointer(''))
        self.assertIsNone(db.read_payload_pointer(None))

    def test_chain_payload_files_kind_filter_and_order(self):
        from dashboard import db
        files = db.chain_payload_files('s2-x-consolidation', kind='prompt')
        self.assertEqual([os.path.basename(p) for p in files],
                         ['001-prompt.md', '002-prompt.md'])
        self.assertEqual(db.chain_payload_files('no-such-chain'), [])
        self.assertEqual(db.chain_payload_files('../evil'), [])

    def test_attempt_ordinal_sorts_after_base_file(self):
        """'000-judge.2.json' < '000-judge.json' lexically — the sort key
        must order (seq, attempt) so [-1] is the NEWEST attempt, not the
        oldest (the retried-chain wrong-file bug)."""
        from dashboard import db
        chain = os.path.join(self.tmp, 'payloads', '2026-08-03', 's1r-y-7')
        os.makedirs(chain)
        for name in ('000-judge.json', '000-judge.2.json',
                     '000-judge.3.json'):
            with open(os.path.join(chain, name), 'w') as f:
                f.write('{}')
        files = db.chain_payload_files('s1r-y-7', kind='judge')
        self.assertEqual([os.path.basename(p) for p in files],
                         ['000-judge.json', '000-judge.2.json',
                          '000-judge.3.json'])


class TestEncodingPromptReader(BrainTestBase):
    """query_encoding_prompt — pointer rows, absolute rows, pruned rows.
    BrainTestBase's tmp dir holds brain_logs.db, so pointing BRAIN_DB_DIR at
    it lets the dashboard query read the same trace rows the DAL writes."""
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self._prev = os.environ.get('BRAIN_DB_DIR')
        os.environ['BRAIN_DB_DIR'] = self.tmp

    def tearDown(self):
        if self._prev is None:
            os.environ.pop('BRAIN_DB_DIR', None)
        else:
            os.environ['BRAIN_DB_DIR'] = self._prev
        super().tearDown()

    def _o_trace(self, chain, ref_id):
        self.brain._trace_dal.append(
            chain_id=chain, scale='s1', event_type='O',
            ref_type='encoding_prompt', ref_id=ref_id,
            summary='x', session_id='sess')

    def test_pointer_row_reads_payload(self):
        from dashboard.queries.encoding import query_encoding_prompt
        ptr = self.brain.record_payload('s1e-abcd1234-5', 'prompt',
                                        'PREAMBLE\n\nBODY')
        self._o_trace('s1e-abcd1234-5', ptr)
        out = query_encoding_prompt(chain_id='s1e-abcd1234-5')
        self.assertEqual(out, {'user_content': 'PREAMBLE\n\nBODY'})

    def test_pointer_row_pruned_payload_reports_pruned(self):
        from dashboard.queries.encoding import query_encoding_prompt
        self._o_trace('s1e-abcd1234-6',
                      'payloads/2026-01-01/s1e-abcd1234-6/000-prompt.md')
        out = query_encoding_prompt(chain_id='s1e-abcd1234-6')
        self.assertIn('pruned', out.get('error', ''))

    def test_absolute_row_is_refused_without_a_file_read(self):
        # Only db_dir-relative pointers are read; an absolute ref_id (the
        # retired /tmp dump shape) answers an error even when the file exists.
        from dashboard.queries.encoding import query_encoding_prompt
        absolute = os.path.join(self.tmp, 'absolute-prompt.json')
        with open(absolute, 'w') as f:
            json.dump({'user_content': 'OUTSIDE PAYLOADS'}, f)
        self._o_trace('s1e-abcd1234-7', absolute)
        out = query_encoding_prompt(chain_id='s1e-abcd1234-7')
        self.assertIn('error', out)
        self.assertNotIn('user_content', out)


class TestRoundRecorderEndToEnd(BrainTestBase):
    """brain.round_recorder — the runner-seam closure over (brain, chain)."""
    needs_embedder = False

    PARTS = {'model': 'claude-test', 'effort': None, 'system': 'SYS',
             'messages': [{'role': 'user', 'content': 'hi'}],
             'tools': ['remember']}

    def test_gated_off_in_normal_config(self):
        rec = self.brain.round_recorder('s1e-abcd1234-9')
        rec(0, dict(self.PARTS))
        self.assertEqual(self.brain.read_payload(
            'payloads'), None)  # nothing recorded → nothing readable
        root = os.path.join(self.tmp, 'payloads')
        chains = [c for _, dirs, _ in os.walk(root) for c in dirs] \
            if os.path.isdir(root) else []
        self.assertNotIn('s1e-abcd1234-9', chains)

    def _enter_debug(self):
        """Deploy TRACE_RECORDING_DEBUG as an ordinary override — the
        post-collapse recipe (nothing pre-registers a dormant v2 anymore)."""
        import json
        from servers.trace_contract import TRACE_RECORDING_DEBUG
        reg = self.brain.register_interaction(
            'trace_recording', template='',
            parameters=json.dumps(TRACE_RECORDING_DEBUG))
        self.brain.set_interaction_active('trace_recording', reg['version'])

    def test_debug_flip_records_pinned_shape(self):
        self._enter_debug()
        rec = self.brain.round_recorder('s1e-abcd1234-10')
        rec(0, dict(self.PARTS))
        rec(1, dict(self.PARTS))
        ptr0 = 'payloads/%s/s1e-abcd1234-10/000-round_payload.json' % \
            iso_now()[:10]
        body = self.brain.read_payload(ptr0)
        self.assertIsNotNone(body)
        payload = json.loads(body)
        # The contract-pinned dict — ab_encode's parsers read these keys.
        self.assertEqual(
            set(payload),
            {'label', 'round', 'seq', 'model', 'effort', 'system',
             'messages', 'tools'})
        self.assertEqual(payload['label'], 's1e-abcd1234-10')
        self.assertEqual(payload['round'], 0)
        self.assertEqual(payload['system'], 'SYS')

    def test_ttl_cache_invalidated_by_flip(self):
        """set_interaction_active promises next-read pickup — the recorder's
        TTL cache must not serve the stale gate after a flip."""
        rec = self.brain.round_recorder('s1e-abcd1234-11')
        rec(0, dict(self.PARTS))            # primes the cache (normal: off)
        self._enter_debug()
        rec(1, dict(self.PARTS))            # must record despite warm cache
        self.brain.clear_interaction_override('trace_recording')
        found = []
        for dirpath, _dirs, names in os.walk(
                os.path.join(self.tmp, 'payloads')):
            if dirpath.endswith('s1e-abcd1234-11'):
                found = sorted(names)
        self.assertEqual(found, ['001-round_payload.json'])


class TestFailedEncodeResidue(BrainTestBase):
    """_render_failed_encodes — the retry sees its predecessor's death."""
    needs_embedder = False

    def _append(self, ref_type, chain, session, metadata=None):
        # append() stamps created_at at write time (iso_now, µs resolution),
        # so ordering in these tests = insertion order.
        self.brain._trace_dal.append(
            chain_id=chain, scale='s1', event_type='delta',
            ref_type=ref_type, ref_id='', summary='x',
            metadata=metadata, session_id=session)

    def test_failed_after_last_success_renders(self):
        from servers.scales.s1.encode import _render_failed_encodes
        sid = 'sess-fail'
        self._append('encoding_run', 's1e-sess-fai-5', sid)
        self._append('encoding_run_failed', 's1e-sess-fai-10', sid,
                     metadata={'error': 'API exploded', 'stop_counter': 10,
                               'partial_actions': [{'tool': 'brain_batch'}],
                               'payload_pointer':
                                   'payloads/2026-08-03/s1e-sess-fai-10/'
                                   '000-failed_run.json'})
        block = _render_failed_encodes(self.brain, sid)
        self.assertIn('stop 10 FAILED: API exploded', block)
        self.assertIn('1 action(s) completed', block)
        self.assertIn('000-failed_run.json', block)

    def test_failure_before_success_is_silent(self):
        from servers.scales.s1.encode import _render_failed_encodes
        sid = 'sess-heal'
        self._append('encoding_run_failed', 's1e-sess-hea-5', sid,
                     metadata={'error': 'old wound', 'stop_counter': 5})
        self._append('encoding_run', 's1e-sess-hea-10', sid)
        self.assertEqual(_render_failed_encodes(self.brain, sid), '')

    def test_no_traces_no_block(self):
        from servers.scales.s1.encode import _render_failed_encodes
        self.assertEqual(_render_failed_encodes(self.brain, 'sess-clean'), '')

    def test_read_failure_is_logged_per_session(self):
        """Codex ids opened in the same minute share their first 8 chars; the
        error names each session by its own short, so the 60 s dedup
        fingerprint keeps the two failures as two rows."""
        from unittest import mock
        from servers.scales.s1.encode import _render_failed_encodes

        def rows():
            return self.brain.logs_conn.execute(
                "SELECT COUNT(*) FROM debug_log WHERE event_type='error' "
                "AND source='s1e_failed_encodes_block'").fetchone()[0]

        before = rows()
        with mock.patch.object(self.brain, 'query_traces',
                               side_effect=RuntimeError('database is locked')):
            for sid in ('01a122b6-9059-7ca1-bb1b-a5e6ff4ce0cf',
                        '01a122b6-f3c2-7d10-8e4a-17c09b2d5e61'):
                self.assertEqual(_render_failed_encodes(self.brain, sid), '')
        self.assertEqual(rows(), before + 2)


if __name__ == '__main__':
    unittest.main()
