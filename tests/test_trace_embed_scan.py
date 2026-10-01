"""Trace pull-reconciliation scan cadence (embed_queue._drain_trace_embeddings_once).

The worker asks "any S0 trace still missing an embedding?" every drain
interval. Unbounded, that question costs the most precisely when the answer
is "no": it walks the whole window — measured at 44.5K rows, ~0.3s of CPU
every 5s, and "nothing" on 6010 of 6020 ticks — which is what an idle daemon
does all day.

These tests hold the two-cadence contract that replaced it: a fast tick asks
only above the watermark, a deep pass re-reads the whole window every
TRACE_DEEP_SCAN_INTERVAL, an empty scan is what moves the watermark, and a
deep pass that is finding real backlog stays hot instead of dripping
TRACE_DRAIN_LIMIT rows per ten minutes.

Pure-function tests — stub brain, stub DAL, stub embedder, no DB.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import embed_queue, embedder
from servers.clock import iso_cutoff


class _StubTraceDal:
    """Records the lower bound each scan was given, and serves canned rows.

    The rows are FILTERED by `since`, like the real anti-join — a stub that
    ignored the bound would return its canned rows however wrong the bound
    was, which is precisely the bug class a watermark can introduce.
    """

    def __init__(self, replies=None, rows=None):
        self.since_seen = []
        self.replies = list(replies or [])
        self.rows = list(rows or [])      # (created_at, row) — the "table"
        self.stored = []

    def find_unembedded(self, limit, scales, ref_types, since=None):
        self.since_seen.append(since)
        if self.replies:
            return self.replies.pop(0)[:limit]
        return [row for created_at, row in self.rows
                if since is None or created_at > since][:limit]

    def store_embeddings(self, rows, model=''):
        self.stored.extend(rows)
        return len(rows)


class _StubBrain:
    def __init__(self, dal, window_days=None):
        self._trace_dal = dal
        self.window_days = window_days
        self.errors = []

    def get_config(self, key, default=None):
        if key == 'embed_queue.trace_window_days' and self.window_days is not None:
            return self.window_days
        return default

    def _log_error(self, source, error, context='', ctx=None):
        self.errors.append({'source': source, 'error': error,
                            'context': context})

    def sources(self):
        return [e['source'] for e in self.errors]


def _trace_row(trace_id):
    return {'id': trace_id, 'ref_type': 'user_message', 'summary': 'hi',
            'metadata': {'content': 'hi', 'human_identity': 'Ada'}}


def _reset():
    embed_queue._trace_watermark = None
    # 0.0 — a fresh process has proven nothing, so its first tick is deep.
    embed_queue._last_trace_deep_at = 0.0
    for key in ('traces_deep_scans_total', 'traces_processed_total',
                'traces_errors_total'):
        embed_queue._stats[key] = 0


def _deep_scans():
    return embed_queue._stats['traces_deep_scans_total']


class NewerIsoTest(unittest.TestCase):
    """The bound picker: later wins, None means 'no bound' and loses."""

    def test_later_of_two_wins(self):
        self.assertEqual(embed_queue._newer_iso('2026-09-01', '2026-09-18'),
                         '2026-09-18')

    def test_none_loses_to_a_real_bound(self):
        self.assertEqual(embed_queue._newer_iso(None, '2026-09-18'),
                         '2026-09-18')
        self.assertEqual(embed_queue._newer_iso('2026-09-18', None),
                         '2026-09-18')

    def test_both_none_is_no_bound(self):
        self.assertIsNone(embed_queue._newer_iso(None, None))


class TraceScanCadenceTest(unittest.TestCase):

    setUp = tearDown = staticmethod(_reset)

    def test_first_scan_is_deep_and_reads_the_whole_window(self):
        dal = _StubTraceDal()
        embed_queue._drain_trace_embeddings_once(_StubBrain(dal))
        self.assertEqual(_deep_scans(), 1)
        # ~30 days back, not ~1 minute back.
        self.assertLess(dal.since_seen[0], iso_cutoff(days=29))

    def test_empty_scan_moves_the_watermark_just_behind_the_scan(self):
        dal = _StubTraceDal()
        embed_queue._drain_trace_embeddings_once(_StubBrain(dal))
        self.assertIsNotNone(embed_queue._trace_watermark)
        self.assertGreater(embed_queue._trace_watermark, iso_cutoff(minutes=5))
        self.assertLess(embed_queue._trace_watermark, iso_cutoff())

    def test_fast_tick_asks_above_the_watermark_not_across_the_window(self):
        dal = _StubTraceDal()
        brain = _StubBrain(dal)
        embed_queue._drain_trace_embeddings_once(brain)   # deep, empty
        mark = embed_queue._trace_watermark               # what the fast tick gets
        embed_queue._drain_trace_embeddings_once(brain)   # fast
        self.assertEqual(_deep_scans(), 1)
        self.assertEqual(dal.since_seen[1], mark)
        self.assertGreater(dal.since_seen[1], dal.since_seen[0])
        # …and coming back empty again walks the watermark forward.
        self.assertGreater(embed_queue._trace_watermark, mark)

    def test_deep_pass_returns_once_the_interval_has_elapsed(self):
        dal = _StubTraceDal()
        brain = _StubBrain(dal)
        embed_queue._drain_trace_embeddings_once(brain)
        embed_queue._last_trace_deep_at -= (
            embed_queue.TRACE_DEEP_SCAN_INTERVAL + 1)
        embed_queue._drain_trace_embeddings_once(brain)
        self.assertEqual(_deep_scans(), 2)
        # The deep pass drops the watermark and re-reads the window.
        self.assertLess(dal.since_seen[1], iso_cutoff(days=29))

    def test_a_tick_that_found_work_leaves_the_watermark_alone(self):
        """Rows are still unembedded when the scan ends — the next tick has
        to see them again, so an empty result is the only thing that may
        advance the bound."""
        dal = _StubTraceDal(replies=[[], [_trace_row('t1')]])
        brain = _StubBrain(dal)
        embed_queue._drain_trace_embeddings_once(brain)
        mark = embed_queue._trace_watermark
        embed_queue._drain_trace_embeddings_once(brain)
        self.assertEqual(embed_queue._trace_watermark, mark)

    def test_window_disabled_means_a_deep_pass_has_no_lower_bound(self):
        dal = _StubTraceDal()
        embed_queue._drain_trace_embeddings_once(_StubBrain(dal, window_days=0))
        self.assertIsNone(dal.since_seen[0])

    def test_window_disabled_still_gets_a_watermark_on_fast_ticks(self):
        dal = _StubTraceDal()
        brain = _StubBrain(dal, window_days=0)
        embed_queue._drain_trace_embeddings_once(brain)
        mark = embed_queue._trace_watermark
        embed_queue._drain_trace_embeddings_once(brain)
        self.assertIsNone(dal.since_seen[0])
        self.assertEqual(dal.since_seen[1], mark)

    def test_a_trace_written_after_the_watermark_is_still_found(self):
        """The failure this whole mechanism risks: a watermark that walks past
        rows the scan should still see. Driven through the drain against a DAL
        that honours `since`, so the bound the tick chose is what selects —
        and the watermark is the tell, since only an EMPTY scan advances it.
        """
        dal = _StubTraceDal()
        brain = _StubBrain(dal)
        embed_queue._drain_trace_embeddings_once(brain)   # empty → watermark set
        mark = embed_queue._trace_watermark
        dal.rows.append((iso_cutoff(), _trace_row('fresh')))   # stamped now
        embed_queue._drain_trace_embeddings_once(brain)   # fast tick
        self.assertEqual(embed_queue._trace_watermark, mark,
                         'the fast tick must have found the fresh trace — a '
                         'watermark that advanced means the scan missed it')

    def test_a_fast_tick_does_not_re_offer_rows_below_the_watermark(self):
        """The other half, and where the CPU saving comes from: rows a previous
        scan already cleared fall out of the fast tick's range."""
        dal = _StubTraceDal()
        brain = _StubBrain(dal)
        embed_queue._drain_trace_embeddings_once(brain)   # empty → watermark set
        mark = embed_queue._trace_watermark
        dal.rows.append((iso_cutoff(days=2), _trace_row('old')))
        embed_queue._drain_trace_embeddings_once(brain)   # fast tick, deep spent
        self.assertEqual(_deep_scans(), 1)
        self.assertGreater(embed_queue._trace_watermark, mark,
                           'a 2-day-old row is below the watermark, so the '
                           'fast tick should have come back empty')

    def test_a_failing_scan_does_not_move_the_watermark(self):
        class _Raises(_StubTraceDal):
            def find_unembedded(self, *a, **kw):
                raise RuntimeError('logs db gone')

        brain = _StubBrain(_Raises())
        embed_queue._drain_trace_embeddings_once(brain)
        self.assertIsNone(embed_queue._trace_watermark)
        self.assertIn('embed_queue_trace_find', brain.sources())


class DeepPassBacklogTest(unittest.TestCase):
    """A backlog below the watermark — a widened window, an embedder outage —
    must still drain at the worker's cadence, not TRACE_DRAIN_LIMIT rows per
    deep interval."""

    def setUp(self):
        _reset()
        self._saved = (embedder.is_ready, embedder.embed_batch,
                       embedder.get_config)
        embedder.is_ready = lambda: True
        embedder.get_config = lambda: {'model_name': 'stub-model'}

    def tearDown(self):
        (embedder.is_ready, embedder.embed_batch,
         embedder.get_config) = self._saved
        _reset()

    def test_a_deep_pass_that_wrote_keeps_the_next_tick_deep(self):
        embedder.embed_batch = lambda texts, kind=None: [[0.1]] * len(texts)
        dal = _StubTraceDal(replies=[[_trace_row('t1')], [_trace_row('t2')]])
        brain = _StubBrain(dal)
        embed_queue._drain_trace_embeddings_once(brain)
        embed_queue._drain_trace_embeddings_once(brain)
        self.assertEqual(_deep_scans(), 2)
        self.assertEqual(embed_queue._stats['traces_processed_total'], 2)

    def test_a_deep_pass_that_wrote_nothing_does_not_stay_hot(self):
        """A trace that can never be embedded is found by every deep pass and
        written by none. Keying the rewind on rows FOUND would let one such
        row pin every tick to a full-window scan — the exact cost the
        watermark exists to remove."""
        embedder.embed_batch = lambda texts, kind=None: []
        dal = _StubTraceDal(replies=[[_trace_row('stuck')],
                                     [_trace_row('stuck')]])
        brain = _StubBrain(dal)
        embed_queue._drain_trace_embeddings_once(brain)
        embed_queue._drain_trace_embeddings_once(brain)
        self.assertEqual(_deep_scans(), 1)
        self.assertIn('embed_queue_trace_embed_mismatch', brain.sources())


if __name__ == '__main__':
    unittest.main()
