"""Tests for brain.run_maintenance_if_due() — the S2 fire decision.

Covers the gating contract:
- min_interval is absolute (never fire more often than that)
- idle threshold is the normal trigger
- 24h force-fire safety valve overrides idle when S2 is stale

Background: S2 was firing every 2.5 days because daemon's `last_activity`
was reset by every IPC call (Claude editing files, internal pings). The
fix split `last_user_activity` (only hook_recall) from `last_activity`
(any IPC), but the brain-side gate still needed a stale-S2 escape valve
for the case where even prompt cadence keeps the idle window closed.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_constants import (
    MAINTENANCE_IDLE_THRESHOLD_SECONDS,
    MAINTENANCE_MIN_INTERVAL_SECONDS,
    MAINTENANCE_MIN_ENCODE_RUNS,
    MAINTENANCE_FORCE_FIRE_SECONDS,
    MAINTENANCE_BOOT_GRACE_SECONDS,
)


class MaintenanceGateTests(BrainTestBase):
    needs_embedder = False

    def _set_last_run(self, ts: float):
        self.brain._maintenance_set_last_run_ts(ts)

    def _call(self, idle_seconds: float, since_last_run: float,
              encode_runs: int = None):
        """Invoke gate with controlled idle + since_last_run via brain.activity.

        encode_runs defaults to MAINTENANCE_MIN_ENCODE_RUNS so the idle/interval
        tests aren't incidentally blocked by the activity gate — they set it
        high enough to pass and exercise the gate they care about. Tests that
        target the activity gate pass an explicit value.

        Patches run_s2 so a real fire returns a sentinel (no actual S2 work).
        """
        now = 1_000_000.0
        # Clear the boot-grace gate (2026-05-08): it suppresses maintenance for
        # the first BOOT_GRACE seconds after _boot_time. Anchor _boot_time to
        # the fake `now` so boot_age is well past the grace window — otherwise
        # the gate short-circuits before the idle/interval logic under test.
        self.brain._boot_time = now - MAINTENANCE_BOOT_GRACE_SECONDS - 60
        self._set_last_run(now - since_last_run)
        self.brain.activity.last_user_activity = now - idle_seconds
        self.brain.activity.encode_runs_since_maintenance = (
            MAINTENANCE_MIN_ENCODE_RUNS if encode_runs is None else encode_runs)
        with patch('servers.scales.s2.coordinator.run_s2',
                   return_value={'fired': True}):
            return self.brain.run_maintenance_if_due(now=now)

    # ── min_interval is absolute ──────────────────────────────────────

    def test_min_interval_blocks_even_when_idle(self):
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 60,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS - 60,
        )
        self.assertIsNone(result, "min_interval must block even when idle")

    def test_min_interval_blocks_force_fire_window(self):
        # Even at 24h+ stale, if min_interval not satisfied we still skip.
        # Defensive: ensures the safety valve doesn't undermine min_interval.
        result = self._call(
            idle_seconds=10,  # not idle
            since_last_run=10,  # min_interval not satisfied
        )
        self.assertIsNone(result)

    # ── idle threshold is the normal trigger ──────────────────────────

    def test_idle_satisfied_fires(self):
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result['units'], {'fired': True})

    def test_not_idle_no_force_fire_blocks(self):
        # Active session, min_interval satisfied, but not stale enough to
        # force-fire. Should skip.
        result = self._call(
            idle_seconds=10,  # user is active
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 60,
        )
        self.assertIsNone(result)

    # ── 24h force-fire safety valve ───────────────────────────────────

    def test_stale_s2_force_fires_despite_active_user(self):
        # The 2.5-day-gap class: user prompts every minute (idle_seconds
        # always small), but S2 is 24h+ stale. Safety valve must override.
        result = self._call(
            idle_seconds=10,  # user actively typing
            since_last_run=MAINTENANCE_FORCE_FIRE_SECONDS + 60,
        )
        self.assertIsNotNone(result,
            "stale S2 must fire even when user is active")

    def test_force_fire_persists_last_run(self):
        before = self.brain._maintenance_last_run_ts()
        self._call(
            idle_seconds=10,
            since_last_run=MAINTENANCE_FORCE_FIRE_SECONDS + 60,
        )
        after = self.brain._maintenance_last_run_ts()
        self.assertGreater(after, before,
            "fire must update s2_last_run_ts")

    # ── cold-boot path (last_activity_ts == 0) ────────────────────────

    def test_zero_last_activity_treated_as_infinite_idle(self):
        # last_user_activity = 0.0 (no prompts yet) must map to "infinitely
        # idle" so the gate fires — once past the boot-grace window. (Boot
        # grace, 2026-05-08, deliberately suppresses the very first post-start
        # poll so the first user recall isn't blocked behind consolidation.)
        now = 1_000_000.0
        self.brain._boot_time = now - MAINTENANCE_BOOT_GRACE_SECONDS - 60
        self._set_last_run(now - MAINTENANCE_MIN_INTERVAL_SECONDS - 60)
        self.brain.activity.last_user_activity = 0.0
        self.brain.activity.encode_runs_since_maintenance = MAINTENANCE_MIN_ENCODE_RUNS
        with patch('servers.scales.s2.coordinator.run_s2',
                   return_value={'fired': True}):
            result = self.brain.run_maintenance_if_due(now=now)
        self.assertIsNotNone(result,
            "past boot-grace, a daemon with no user activity (0.0) is treated "
            "as infinitely idle and fires")


    # ── encode-runs activity gate (≥N Scribe runs since last run) ─────

    def test_insufficient_encode_runs_blocks(self):
        # Idle + interval both satisfied, but only 1 encoder run since last S2.
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
            encode_runs=MAINTENANCE_MIN_ENCODE_RUNS - 1,
        )
        self.assertIsNone(result,
            "too few encoder runs since last run must block the fire")

    def test_sufficient_encode_runs_fires(self):
        result = self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
            encode_runs=MAINTENANCE_MIN_ENCODE_RUNS,
        )
        self.assertIsNotNone(result, "enough encoder runs + idle + interval fires")

    def test_force_fire_overrides_encode_runs(self):
        # 24h stale with zero encoder runs — the safety valve still fires.
        result = self._call(
            idle_seconds=10,
            since_last_run=MAINTENANCE_FORCE_FIRE_SECONDS + 60,
            encode_runs=0,
        )
        self.assertIsNotNone(result,
            "force-fire must override the encode-runs gate for stale S2")

    def test_fire_consumes_encode_runs(self):
        # When S2 fires, it consumes exactly the encode runs it gated on, so
        # the counter resets toward the next cycle.
        self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 1,
            since_last_run=MAINTENANCE_MIN_INTERVAL_SECONDS + 1,
            encode_runs=MAINTENANCE_MIN_ENCODE_RUNS,
        )
        self.assertEqual(self.brain.activity.encode_runs_since_maintenance, 0,
            "a fire must consume the encode runs it gated on")

    # ── eval brain: S2 never starts itself; s2_force still runs ──────────

    def _mark(self):
        """Mark this test brain's folder and take the mark the way
        Brain.__init__ does when a brain opens (the brain here is already
        open, like _call's _boot_time)."""
        from servers.brain import Brain
        db_dir = os.path.dirname(self.brain.db_path)
        Brain.mark_eval_brain(db_dir)
        self.brain._eval_brain_since = Brain.read_eval_mark(db_dir)

    def test_eval_brain_never_schedules_s2(self):
        """Every gate is open and S2 is a day stale, the case of an eval brain
        booted hours after it was made (directive c1c7f680): a marked brain
        still starts nothing on its own."""
        self._mark()
        self.assertIsNone(self._call(
            idle_seconds=MAINTENANCE_IDLE_THRESHOLD_SECONDS + 60,
            since_last_run=MAINTENANCE_FORCE_FIRE_SECONDS + 60))

    def test_s2_force_still_runs_on_an_eval_brain(self):
        from servers.dispatch_ops import _handle_s2_force
        self._mark()
        with patch('servers.scales.s2.coordinator.run_s2',
                   side_effect=lambda brain_arg, units=None, force=False: {}) as run:
            out = _handle_s2_force(self.brain, {}, None)
        self.assertTrue(out['ok'])
        run.assert_called_once()
        self.assertIs(run.call_args.kwargs['force'], True)

    def test_eval_mark_is_read_once_when_the_brain_opens(self):
        """The marker sits beside brain.db and carries this host's clock. An
        open brain keeps what it read at open, so marking a running brain
        cannot orphan the sessions it already drove; ping reports the value
        the brain opened with."""
        from servers.brain import Brain
        from servers.dispatch_ops import _handle_ping
        db_dir = os.path.dirname(self.brain.db_path)
        self.assertIsNone(Brain.read_eval_mark(db_dir))
        before = time.time()
        Brain.mark_eval_brain(db_dir)
        self.assertGreaterEqual(Brain.read_eval_mark(db_dir), before - 1)
        self.assertIsNone(self.brain.eval_brain_since(),
                          'an open brain keeps the mark it read at open')
        self.assertIsNone(
            _handle_ping(self.brain, {}, None)['result']['eval_brain_since'])
        os.utime(Brain.eval_brain_marker(db_dir), (1_000.0, 1_000.0))
        Brain.mark_eval_brain(db_dir)
        self.assertGreater(Brain.read_eval_mark(db_dir), 1_000.0,
                           're-marking moves the moment forward')
        self._mark()
        self.assertEqual(
            _handle_ping(self.brain, {}, None)['result']['eval_brain_since'],
            self.brain.eval_brain_since())

if __name__ == '__main__':
    unittest.main()


class S2SingleDoorTests(BrainTestBase):
    """brain.run_s2() — the one door, and the single-flight it owns.

    The guard used to live on the daemon, an attribute of the SERVER, so it
    could only block a second poll entry. That is how the 2026-06 parallel-run
    bug happened (node:daaf63a9): a second caller could neither see it nor be
    seen by it. These tests pin the guard to the brain, where every in-process
    caller shares it. (The daemon still carries a flag — renamed
    `_maintenance_poll_running` — but it throttles duplicate pool submissions
    and is not this guard.)
    """
    needs_embedder = False

    def test_second_concurrent_call_skips_instead_of_overlapping(self):
        """Two cycles must never overlap.

        Exercised by re-entering the door from inside a running cycle. That is a
        PROXY for cross-thread contention, not the real thing — but for a plain
        (non-reentrant) Lock the two are the same failure: acquire(blocking=False)
        returns False either way. It also pins that S2 must never re-enter
        itself, which is why the lock is a Lock and not an RLock.

        NEGATIVE-TESTED: disabling the guard fails this test.
        """
        inner = {}

        def _reenter(brain_arg, units=None, force=False):
            inner['result'] = self.brain.run_s2()
            return {'healer': {'actions': 1}}

        with patch('servers.scales.s2.coordinator.run_s2', side_effect=_reenter):
            outer = self.brain.run_s2()

        self.assertEqual(outer['units'], {'healer': {'actions': 1}},
                         "the first caller must complete normally")
        self.assertEqual(inner['result'].get('skipped'), 'already running',
                         "the second caller must skip, not run or block")
        self.assertEqual(inner['result']['units'], {})

    def test_forced_s2_does_not_eat_encodes_that_land_during_the_cycle(self):
        """dispatch_ops._handle_s2_force kept its OWN copy of the post-fire
        bookkeeping and drifted into a bug: it re-read
        encode_runs_since_maintenance AFTER the multi-minute cycle and
        subtracted all of it — i.e. zeroed it, which is precisely what
        consume_encode_runs' docstring forbids ("Subtract — not zero — so
        encoder runs that complete during the multi-minute S2 cycle still
        accrue toward the next cycle"). Encodes finishing during a forced cycle
        were silently dropped, starving the next cycle's MIN_ENCODE_RUNS gate.

        Both callers now route through brain.mark_s2_ran(now, snapshot), where
        the snapshot is taken BEFORE the cycle.

        NEGATIVE-TESTED: restoring the re-read (consuming
        encode_runs_since_maintenance after the run) fails this.
        """
        from servers.dispatch_ops import _handle_s2_force
        self.brain.activity.encode_runs_since_maintenance = 2

        def _cycle_with_a_concurrent_encode(brain_arg, units=None, force=False):
            # An S1 encode completes while S2 is still mid-cycle — the whole
            # reason consume subtracts instead of zeroing.
            self.brain.activity.record_encode_run()
            return {'healer': {'actions': 1}}

        with patch('servers.scales.s2.coordinator.run_s2',
                   side_effect=_cycle_with_a_concurrent_encode):
            out = _handle_s2_force(self.brain, {}, None)

        self.assertTrue(out['ok'])
        self.assertEqual(
            self.brain.activity.encode_runs_since_maintenance, 1,
            "the 2 runs the cycle covered are consumed; the 1 that landed "
            "mid-cycle must survive toward the NEXT cycle, not be zeroed")

    def test_gate_does_not_stamp_or_consume_while_a_cycle_is_running(self):
        """THE subtlety. run_maintenance_if_due stamps the last-run timestamp
        BEFORE executing (so a concurrent poller skips on min-interval). If it
        stamped and then run_s2() skipped because the lock was held, the cycle
        would be burned without running AND the encode runs it gated on would
        be eaten — starving the next one. So the gate must check first."""
        now = 1_000_000.0
        self.brain._boot_time = now - MAINTENANCE_BOOT_GRACE_SECONDS - 60
        before_ts = now - MAINTENANCE_MIN_INTERVAL_SECONDS - 60
        self.brain._maintenance_set_last_run_ts(before_ts)
        self.brain.activity.last_user_activity = (
            now - MAINTENANCE_IDLE_THRESHOLD_SECONDS - 60)
        self.brain.activity.encode_runs_since_maintenance = (
            MAINTENANCE_MIN_ENCODE_RUNS + 1)
        expected_runs = self.brain.activity.encode_runs_since_maintenance

        # Every gate is open — the ONLY thing that should stop it is the lock.
        self.brain._s2_lock.acquire()
        try:
            result = self.brain.run_maintenance_if_due(now=now)
        finally:
            self.brain._s2_lock.release()

        self.assertIsNone(result, "must not report a run that did not happen")
        self.assertEqual(self.brain.activity.encode_runs_since_maintenance,
                         expected_runs,
                         "encode runs were consumed for a cycle that never ran")
        self.assertAlmostEqual(
            self.brain._maintenance_last_run_ts(), before_ts, places=3,
            msg="last-run timestamp advanced for a cycle that never ran")


class S2UnitsFilterTests(BrainTestBase):
    """brain.run_s2(units=...) — force a subset through the same door.

    Every unit's run() is patched in these tests, so a filter that leaked
    records the extra unit instead of doing real S2 work.
    """
    needs_embedder = False

    def _run_recording(self, ran, **kwargs):
        """Run the door with every unit's run() recording into `ran`.

        The caller owns `ran`, so it still holds what ran when the door raises.
        """
        from contextlib import ExitStack
        from servers.scales.s2.aspect_integration import AspectIntegration
        from servers.scales.s2.community import CommunityDetection
        from servers.scales.s2.healer import Healer
        from servers.scales.s2.consolidation import Consolidation

        def _record(unit):
            ran.append(unit.NAME)
            return {'actions': 0}

        with ExitStack() as stack:
            for cls in (AspectIntegration, CommunityDetection, Healer,
                        Consolidation):
                stack.enter_context(patch.object(
                    cls, 'run', autospec=True, side_effect=_record))
            return self.brain.run_s2(**kwargs)

    def test_default_runs_every_unit_in_fixed_order(self):
        ran = []
        out = self._run_recording(ran)
        order = ['aspect_integration', 'community_detection', 'healer',
                 'consolidation']
        self.assertEqual(ran, order)
        self.assertEqual(list(out['units']), order)
        self.assertFalse(out['partial'])

    def test_filter_runs_only_the_named_units_in_fixed_order(self):
        """The selection arrives reversed and still runs in coordinator order."""
        ran = []
        out = self._run_recording(
            ran, units=['consolidation', 'aspect_integration'])
        self.assertEqual(ran, ['aspect_integration', 'consolidation'])
        self.assertEqual(list(out['units']),
                         ['aspect_integration', 'consolidation'])
        self.assertTrue(out['partial'])

    def test_listing_every_unit_is_not_partial(self):
        ran = []
        out = self._run_recording(
            ran, units=['consolidation', 'healer', 'community_detection',
                        'aspect_integration'])
        self.assertEqual(len(ran), 4)
        self.assertFalse(out['partial'])

    def test_bad_selection_raises_and_runs_nothing(self):
        """Unknown name, empty list, bare string, dict, unhashable element:
        each is loud, never a silent or unintended cycle — and the
        single-flight lock is left free."""
        for bad in (['aspect_integration', 'no_such_unit'], [],
                    'aspect_integration', {'aspect_integration': False},
                    [['healer']]):
            with self.subTest(units=bad):
                ran = []
                with self.assertRaises(ValueError):
                    self._run_recording(ran, units=bad)
                self.assertEqual(ran, [])
                self.assertFalse(self.brain.s2_running)

    def test_bad_selection_raises_even_while_a_cycle_is_running(self):
        """Validation precedes the lock: a typo sent during a live cycle is
        reported as a typo, not masked as 'already running'."""
        self.brain._s2_lock.acquire()
        try:
            with self.assertRaises(ValueError):
                self.brain.run_s2(units=['aspect'])
            self.assertEqual(
                self.brain.run_s2(units=['aspect_integration']).get('skipped'),
                'already running')
        finally:
            self.brain._s2_lock.release()

    def test_partial_forced_cycle_does_not_stamp_the_scheduler(self):
        """A unit subset did not do the full cycle's work, so s2_force must not
        advance last_run_ts or consume encode runs — the poll's next natural
        cycle still owes every unit. The full force afterwards is the control:
        same handler, same brain, and it does stamp.

        NEGATIVE-TESTED: stamping regardless of `units` fails this.
        """
        from servers.dispatch_ops import _handle_s2_force
        before_ts = 12345.0
        self.brain._maintenance_set_last_run_ts(before_ts)
        self.brain.activity.encode_runs_since_maintenance = 3
        seen = []

        def _cycle(brain_arg, units=None, force=False):
            seen.append(units)
            return {'aspect_integration': {'actions': 1}}

        with patch('servers.scales.s2.coordinator.run_s2', side_effect=_cycle):
            out = _handle_s2_force(
                self.brain, {'units': ['aspect_integration']}, None)
            self.assertTrue(out['ok'])
            self.assertAlmostEqual(
                self.brain._maintenance_last_run_ts(), before_ts, places=3,
                msg="a partial forced cycle advanced the scheduler clock")
            self.assertEqual(self.brain.activity.encode_runs_since_maintenance,
                             3, "a partial forced cycle consumed encode runs")

            _handle_s2_force(self.brain, {}, None)

        self.assertEqual(seen, [['aspect_integration'], None])
        self.assertGreater(self.brain._maintenance_last_run_ts(), before_ts)
        self.assertEqual(self.brain.activity.encode_runs_since_maintenance, 0)

    def test_forcing_every_unit_by_name_stamps_like_a_full_cycle(self):
        """Listing all four units runs a full cycle, so it must stamp —
        otherwise the poll's next tick fires a second full cycle
        (node:cbc43e30).

        NEGATIVE-TESTED: gating the stamp on `units is None` fails this.
        """
        from servers.dispatch_ops import _handle_s2_force
        self.brain._maintenance_set_last_run_ts(12345.0)
        self.brain.activity.encode_runs_since_maintenance = 3
        every = ['aspect_integration', 'community_detection', 'healer',
                 'consolidation']
        with patch('servers.scales.s2.coordinator.run_s2',
                   side_effect=lambda brain_arg, units=None, force=False: {}):
            _handle_s2_force(self.brain, {'units': every}, None)
        self.assertGreater(self.brain._maintenance_last_run_ts(), 12345.0)
        self.assertEqual(self.brain.activity.encode_runs_since_maintenance, 0)
