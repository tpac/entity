"""Error-log suppression contract: same condition deduped, distinct conditions never.

Guards the 2026-09-21 reshape of `Brain._check_rate_limit`. The function used to
carry four layers (circuit breaker, 60s dedup, 50/source/hour, 200/hour global).
The global layer summed per-source timestamp lists that only the currently-logging
source ever pruned, so once ~200 timestamps accumulated across sources that then
went quiet, EVERY error from EVERY source was dropped for the life of the daemon —
silently, with no counter and no recovery. `test_no_global_latch` is that exact
reproduction: it must never pass again.
"""
import time

from tests.brain_test_base import BrainTestBase


class TestErrorLogDedup(BrainTestBase):
    needs_embedder = False

    def _errors(self, source=None):
        sql = ("SELECT COUNT(*) FROM debug_log WHERE event_type = 'error'"
               + (" AND source = ?" if source else ""))
        cur = self.brain.logs_conn.execute(sql, (source,) if source else ())
        return cur.fetchone()[0]

    # ── the guard ──────────────────────────────────────────────────────────
    def test_no_global_latch(self):
        """Many distinct conditions must not silence an unrelated later source.

        Pre-reshape this failed: 5 sources x 40 distinct errors filled the global
        200/hour budget with timestamps nothing ever pruned, and the next source
        to log was suppressed forever.
        """
        for s in range(5):
            for i in range(40):
                self.brain._log_error('flood_source_%d' % s,
                                      ValueError('distinct condition %d' % i))
        before = self._errors('late_arriving_source')
        self.brain._log_error('late_arriving_source', RuntimeError('the one that matters'))
        self.assertEqual(self._errors('late_arriving_source'), before + 1,
                         'a new source was silenced by unrelated earlier errors')

    def test_distinct_conditions_all_recorded(self):
        """A burst of different errors from ONE source is information, not a flood."""
        for i in range(60):   # past the old 50/source/hour cap
            self.brain._log_error('busy_source', ValueError('condition %d' % i))
        self.assertEqual(self._errors('busy_source'), 60)

    # ── the one thing that is still suppressed ─────────────────────────────
    def test_same_condition_deduped_in_window(self):
        for _ in range(10):
            self.brain._log_error('repeating', ValueError('identical'), 'same ctx')
        self.assertEqual(self._errors('repeating'), 1)

    def test_same_condition_logs_again_after_window(self):
        self.brain._log_error('expiring', ValueError('identical'), 'same ctx')
        # Age every fingerprint past the window without sleeping.
        aged = time.monotonic() - self.brain._error_dedup_window - 1
        self.brain._error_fingerprints = {
            f: aged for f in self.brain._error_fingerprints}
        self.brain._log_error('expiring', ValueError('identical'), 'same ctx')
        self.assertEqual(self._errors('expiring'), 2)

    def test_context_distinguishes_conditions(self):
        """Same message, different condition — both recorded (the s2 twin-pull case)."""
        self.brain._log_error('twin', ValueError('hit limit=500'), 'recall pull')
        self.brain._log_error('twin', ValueError('hit limit=500'), 'surface_selected pull')
        self.assertEqual(self._errors('twin'), 2)

    def test_warnings_share_the_same_path(self):
        self.brain._log_warning('warner', 'identical message', 'same ctx')
        self.brain._log_warning('warner', 'identical message', 'same ctx')
        self.brain._log_warning('warner', 'a different message', 'same ctx')
        cur = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type = 'warning' AND source = 'warner'")
        self.assertEqual(cur.fetchone()[0], 2)

    # ── memory safety ──────────────────────────────────────────────────────
    def test_fingerprint_map_drops_expired_entries(self):
        """The dedup map holds live conditions only — it must not grow forever.

        Live entries are retained by contract (dropping one would break dedup),
        so the bound is time, not count: once the window passes, the next call
        sweeps them.
        """
        for i in range(50):
            self.brain._log_error('sweeper', ValueError('unique %d' % i))
        self.assertEqual(len(self.brain._error_fingerprints), 50)
        # Age everything past the window, then log once to trigger the sweep.
        aged = time.monotonic() - self.brain._error_dedup_window - 1
        self.brain._error_fingerprints = {
            f: aged for f in self.brain._error_fingerprints}
        self.brain._last_fingerprint_sweep = aged
        self.brain._log_error('sweeper', ValueError('the trigger'))
        self.assertEqual(len(self.brain._error_fingerprints), 1,
                         'expired fingerprints were not swept')

    def test_sweep_is_rate_limited_not_per_call(self):
        """A storm of distinct fingerprints must not rebuild the map every call."""
        self.brain._log_error('storm', ValueError('first'))
        sweep_at = self.brain._last_fingerprint_sweep
        for i in range(200):
            self.brain._log_error('storm', ValueError('unique %d' % i))
        self.assertEqual(self.brain._last_fingerprint_sweep, sweep_at,
                         'swept more than once inside a single window')

    # ── contract edges a future "improvement" could break ──────────────────
    def test_unfingerprinted_events_are_never_deduped(self):
        """`_log_event` skips dedup when fingerprint is None — raw events must
        not collapse into each other (a None key would collide across all)."""
        for _ in range(3):
            self.brain._log_event('error', 'raw_source',
                                  metadata={'error': 'identical'},
                                  file_level='ERROR', file_text='identical')
        self.assertEqual(self._errors('raw_source'), 3)

    def test_suppressed_repeats_do_not_extend_the_window(self):
        """The window runs from the last LOGGED occurrence, not the last attempt.

        If a hit refreshed the stored timestamp, a condition firing faster than
        the window would log once and then be suppressed forever.
        """
        self.brain._log_error('steady', ValueError('identical'), 'same ctx')
        for _ in range(5):                       # hits inside the window
            self.brain._log_error('steady', ValueError('identical'), 'same ctx')
        self.assertEqual(self._errors('steady'), 1)
        # Age the ORIGINAL entry past the window. If any hit had refreshed it,
        # this rewind would not be enough and the next call would stay silent.
        aged = time.monotonic() - self.brain._error_dedup_window - 1
        self.brain._error_fingerprints = {
            f: aged for f in self.brain._error_fingerprints}
        self.brain._log_error('steady', ValueError('identical'), 'same ctx')
        self.assertEqual(self._errors('steady'), 2)

    # ── attribution ────────────────────────────────────────────────────────
    def test_row_without_ctx_is_sessionless_even_after_a_boot(self):
        """A row logged with no ctx and no session_id belongs to no session,
        not to whichever stream booted last."""
        self.brain.reset_session_activity('booted-stream')
        self.brain._log_error('ctxless_source', ValueError('no ctx here'))
        row = self.brain.logs_conn.execute(
            "SELECT session_id FROM debug_log WHERE source = 'ctxless_source'"
        ).fetchone()
        self.assertEqual(row[0], '')
