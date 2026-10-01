"""Unscoped vector-coverage sweep (embed_queue._coverage_sweep).

The queue repairs only what reached it. A writer that skips the enqueue
hooks, or a crash between insert and enqueue, leaves a node unembedded
with no other route back — and an unembedded node is invisible to LAF
entirely. This sweep is that route.

It previously lived in `hook_idle_maintenance`, which is driven by a
Claude Code Notification/idle_prompt event. That event stopped firing
2026-07-04 and took vector coverage with it, silently, for six weeks.
These tests hold the contract that it now lives beside the queue that
owns embedding, that it cannot be starved, and that BOTH a gap and a
stuck node are reported.

Pure-function tests — stub brain, no DB, no embedder.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import embed_queue


class _StubVecDal:
    """Mirrors find_missing's DOCUMENTED model contract.

    dal.py: "A row is 'present' only if it has a non-null embedding AND (if
    `model` is given) was produced by the same model."

    A stub that ignored `model` is exactly what let a probe omitting it ship
    green: stale-model rows read as present, so a model swap — the whole
    corpus — looked like an empty backlog. Rows are (node_id, embedded_by)
    where embedded_by is None for "no embedding at all".
    """

    def __init__(self, rows=None):
        self.rows = rows or []
        self.seen_kwargs = {}

    def find_missing(self, vector_type, limit=50, **kw):
        self.seen_kwargs = dict(kw)
        model = kw.get('model')
        out = [{'id': nid} for nid, by in self.rows
               if by is None or (model and by != model)]
        return out[:limit]


class _StubBrain:
    """Stands in for the Brain door, recording what the sweep asked for."""

    def __init__(self, outcome=None, raises=None):
        self.outcome = outcome or {'repaired': 0, 'by_type': {},
                                   'remaining': False, 'stuck': []}
        self.raises = raises
        self.calls = []
        self.errors = []

    def vector_coverage_sweep(self, batch_size=None):
        self.calls.append({'batch_size': batch_size})
        if self.raises is not None:
            raise self.raises
        return self.outcome

    def _log_error(self, source, error, context='', ctx=None):
        self.errors.append({'source': source, 'error': error,
                            'context': context})

    def sources(self):
        return [e['source'] for e in self.errors]


def _outcome(repaired=0, by_type=None, remaining=False, stuck=()):
    return {'repaired': repaired, 'by_type': by_type or {},
            'remaining': remaining, 'stuck': list(stuck)}


def _reset():
    # "Due now, but not stale" — the ordinary steady state. NOT 0.0: that is
    # ~epoch seconds of apparent staleness, which punches through the
    # queue-deference floor and hid a real boot-time bug until this changed.
    embed_queue._last_sweep_at = (
        time.time() - embed_queue.COVERAGE_SWEEP_INTERVAL - 1)
    embed_queue._queue.clear()
    embed_queue._edge_queue.clear()


class CoverageSweepTest(unittest.TestCase):

    setUp = tearDown = staticmethod(_reset)

    def test_asks_the_door_with_the_configured_batch(self):
        brain = _StubBrain()
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 1)
        self.assertEqual(brain.calls[0]['batch_size'],
                         embed_queue.COVERAGE_SWEEP_BATCH)

    def test_throttled_within_interval(self):
        brain = _StubBrain()
        embed_queue._coverage_sweep(brain)
        embed_queue._coverage_sweep(brain)
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 1)

    def test_runs_again_after_interval_elapses(self):
        brain = _StubBrain()
        embed_queue._coverage_sweep(brain)
        embed_queue._last_sweep_at -= (embed_queue.COVERAGE_SWEEP_INTERVAL + 1)
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 2)

    def test_defers_to_scoped_work(self):
        brain = _StubBrain()
        embed_queue._queue.add('node-1')
        embed_queue._coverage_sweep(brain)
        self.assertEqual(brain.calls, [])

    def test_defers_to_scoped_edge_work(self):
        brain = _StubBrain()
        embed_queue._edge_queue.add('edge-1')
        embed_queue._coverage_sweep(brain)
        self.assertEqual(brain.calls, [])

    def test_staleness_floor_beats_a_permanently_busy_queue(self):
        """Deferring to scoped work must not become never running.

        On a brain that always has queued work the sweep would otherwise be
        skipped forever — and a busy brain is exactly where a writer is most
        likely to have bypassed the enqueue hooks.
        """
        brain = _StubBrain()
        embed_queue._queue.add('always-busy')
        embed_queue._last_sweep_at = (
            time.time() - embed_queue.COVERAGE_SWEEP_MAX_STALENESS - 1)
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 1,
                         'staleness floor must force a sweep past the queue guard')

    def test_repair_is_reported_as_an_error(self):
        brain = _StubBrain(_outcome(repaired=5, by_type={'_primary': 5}))
        embed_queue._coverage_sweep(brain)
        self.assertIn('embed_coverage_gap', brain.sources())
        self.assertIn('5', brain.errors[0]['context'])

    def test_stuck_node_is_reported_when_nothing_was_repaired(self):
        """The case reporting-on-repair-only would hide forever."""
        brain = _StubBrain(_outcome(repaired=0, stuck=[{'id': 'abc123'}]))
        embed_queue._coverage_sweep(brain)
        self.assertIn('embed_coverage_stuck', brain.sources())
        self.assertIn('abc123', brain.errors[0]['context'])

    def test_remaining_clears_the_throttle(self):
        brain = _StubBrain(_outcome(repaired=30, remaining=True))
        embed_queue._coverage_sweep(brain)
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 2,
                         'a filled batch must be followed immediately')

    def test_stuck_alone_does_not_clear_the_throttle(self):
        """A node that can never embed must not spin the sweep every tick.

        This is why `remaining` comes from the repair's own batch counts and
        not from a probe: a permanently-stuck node repairs nothing, so it can
        never keep the loop hot.
        """
        brain = _StubBrain(_outcome(repaired=0, remaining=False,
                                    stuck=[{'id': 'never'}]))
        embed_queue._coverage_sweep(brain)
        embed_queue._coverage_sweep(brain)
        self.assertEqual(len(brain.calls), 1,
                         'a stuck node must not re-arm the sweep every tick')

    def test_clean_sweep_is_silent(self):
        brain = _StubBrain(_outcome())
        embed_queue._coverage_sweep(brain)
        self.assertEqual(brain.errors, [])

    def test_failure_is_logged_and_swallowed(self):
        brain = _StubBrain(raises=RuntimeError('embedder down'))
        embed_queue._coverage_sweep(brain)
        self.assertIn('embed_coverage_sweep', brain.sources())


class CoverageDoorTest(unittest.TestCase):
    """Brain.vector_coverage_sweep — repair and detection in ONE place.

    They were split across two callers once; the repair passed `model=` and
    the probe did not, so stale-model rows read as present and a model swap
    looked like an empty backlog.
    """

    def _door(self, backfill_result, rows):
        from servers.brain import Brain
        vec = _StubVecDal(rows)

        class _Self:
            _vec_dal = vec

            def backfill_vectors(self, batch_size=None):
                return backfill_result

        return Brain.vector_coverage_sweep(_Self(), 30), vec

    def test_probe_is_given_the_model(self):
        out, vec = self._door({}, [('n1', None)])
        self.assertIn('model', vec.seen_kwargs,
                      'probe must ask the same question the repair asked')
        self.assertTrue(out['stuck'])

    def test_remaining_comes_from_batch_fill_not_a_probe(self):
        out, vec = self._door({'_primary': 30}, [])
        self.assertTrue(out['remaining'])
        self.assertEqual(vec.seen_kwargs, {},
                         'no probe needed when repair reports work')

    def test_partial_batch_means_backlog_drained(self):
        out, _ = self._door({'_primary': 7}, [])
        self.assertFalse(out['remaining'])

    def test_non_int_values_are_not_counted_as_repairs(self):
        out, _ = self._door({'error': 'embedder not ready'}, [])
        self.assertEqual(out['repaired'], 0)
        self.assertFalse(out['remaining'])


class SweepIsNotInTheDrainTest(unittest.TestCase):
    """The sweep belongs to the worker loop, not to _drain_once.

    _drain_once's only sweep-reachable branch was its empty-tick early
    return, so sustained write load silently switched the safety net off.
    It is also called directly by the test harness (brain_test_base,
    isolated_brain) — sweeping there runs an unscoped backfill against
    whatever graph the fixture handed out.
    """

    setUp = tearDown = staticmethod(_reset)

    def test_drain_does_not_sweep(self):
        brain = _StubBrain()
        embed_queue._drain_once(brain)
        self.assertEqual(brain.calls, [],
                         '_drain_once must not trigger the coverage sweep')

    def test_worker_loop_actually_invokes_the_sweep(self):
        """Behavioural, not a substring match — a rename or an unreachable
        call site must fail this, since a repair path nothing calls is the
        exact failure the whole change exists to prevent."""
        brain = _StubBrain()
        called = []
        orig = {
            '_coverage_sweep': embed_queue._coverage_sweep,
            '_drain_trace_embeddings_once': embed_queue._drain_trace_embeddings_once,
            '_check_stall': embed_queue._check_stall,
            'EMBED_DRAIN_INTERVAL': embed_queue.EMBED_DRAIN_INTERVAL,
        }

        def _stop_after_this_tick(*a, **kw):
            embed_queue._shutdown_event.set()

        embed_queue._coverage_sweep = lambda b: called.append(b)
        embed_queue._drain_trace_embeddings_once = lambda b: None
        embed_queue._check_stall = _stop_after_this_tick
        embed_queue.EMBED_DRAIN_INTERVAL = 0.0
        try:
            embed_queue._worker_loop(brain)
        finally:
            embed_queue._shutdown_event.clear()
            for k, v in orig.items():
                setattr(embed_queue, k, v)

        self.assertEqual(len(called), 1,
                         'worker loop must invoke the coverage sweep each tick')


class HookNoLongerOwnsVectorsTest(unittest.TestCase):
    """Directional guardrail: vector coverage must not drift back behind
    the Claude Code notification hook."""

    def test_idle_maintenance_does_not_backfill_vectors(self):
        path = os.path.join(os.path.dirname(__file__), '..',
                            'servers', 'daemon_hooks.py')
        with open(path) as fh:
            src = fh.read()
        self.assertNotIn('brain.backfill_vectors(', src,
                         'vector backfill belongs to embed_queue, not the '
                         'Notification/idle_prompt hook')


class LogsMaintenanceUsesPrivateConnTest(unittest.TestCase):
    """The orphan sweep must never run on brain.conn.

    run_maintenance ends in a bare graph_conn.commit(). On the foreground
    connection that commit lands on whatever brain_batch envelope is open —
    the stray-commit class that killed a savepoint mid-merge once already.
    A private maintenance connection makes it structurally impossible.
    """

    def test_brain_door_uses_a_maintenance_connection(self):
        import inspect
        from servers.brain import Brain
        src = inspect.getsource(Brain.run_logs_maintenance)
        self.assertIn('connect_maintenance', src)
        self.assertNotIn('graph_conn=self.conn', src)

    def test_daemon_does_not_reach_into_logs_dal(self):
        path = os.path.join(os.path.dirname(__file__), '..',
                            'servers', 'daemon_server.py')
        with open(path) as fh:
            src = fh.read()
        self.assertNotIn('_logs_dal.run_maintenance', src,
                         'route through Brain.run_logs_maintenance')


class BootRequeueTest(unittest.TestCase):
    """BrainDaemon._enqueue_vector_backfill_gaps — the boot route back for a
    node whose enqueue died with the previous process (the in-memory queue
    does not survive a restart). Without it the node waits for the coverage
    sweep, which then reports a hook bypass that never happened."""

    setUp = tearDown = staticmethod(_reset)

    def _requeue(self, primary_rows, edge_context_rows=()):
        from servers.daemon_server import BrainDaemon
        asked = []

        class _Vec:
            def find_missing(self, vector_type, limit=50, **kw):
                asked.append((vector_type, kw))
                rows = primary_rows if vector_type == '_primary' else edge_context_rows
                return [{'id': nid} for nid in rows][:limit]

        class _Brain:
            _vec_dal = _Vec()

            def _edge_context_excluded(self):
                return ()

        class _Daemon:
            brain = _Brain()

            def _log(self, msg):
                pass

        BrainDaemon._enqueue_vector_backfill_gaps(_Daemon())
        return asked

    def test_node_missing_primary_is_queued(self):
        self._requeue(['n_lost'], ['n_ctx'])
        self.assertEqual(embed_queue._queue, {'n_lost', 'n_ctx'})

    def test_primary_probe_leaves_model_swaps_to_the_sweep(self):
        asked = self._requeue([])
        primary_kw = [kw for vt, kw in asked if vt == '_primary']
        self.assertEqual(len(primary_kw), 1)
        self.assertNotIn('model', primary_kw[0],
                         'a model= probe would queue the whole corpus on a swap')


if __name__ == '__main__':
    unittest.main()
