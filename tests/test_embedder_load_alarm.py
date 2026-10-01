"""Brain construction must be LOUD about a dead embedder.

`embedder.load_model` never raises — it records the reason in
`stats['load_error']`, leaves `_model` None, and returns. So the `except`
around it in `Brain.__init__` almost never fires, and a brain whose embedder
failed to load used to finish construction in silence.

Silence is what costs: the embedder module is process-global, so from that
point every recall returns empty and every backfill writes nothing, while the
only trace is a `print` — which under pytest's capture is shown for the test
that FAILS, never the test that broke it. On 2026-09-18 that cost a session
~90 minutes chasing 29 bystander failures in four unrelated test files.

The contract these tests hold: construction asks `is_ready()` and, on a dead
embedder, writes a row through `_log_error` — durable, queryable, surfaced at
boot, and immune to captured stdout.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import embedder
from servers.brain import Brain


class _DeadEmbedderTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self._saved = (embedder.load_model, embedder.is_ready,
                       dict(embedder.stats))

    def tearDown(self):
        embedder.load_model, embedder.is_ready = self._saved[0], self._saved[1]
        embedder.stats.clear()
        embedder.stats.update(self._saved[2])
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _errors_for(self, brain, source):
        rows = brain.query_logs(source='debug', hours=1, level='error',
                                limit=50).get('entries', [])
        return [r for r in rows if r.get('origin') == source]

    def test_dead_embedder_is_logged_not_swallowed(self):
        """The failure mode that hid: load_model returns normally with the
        model dead."""
        def _fails_quietly(config=None):
            embedder.stats['model_loaded'] = False
            embedder.stats['load_error'] = 'ONNXRuntimeError NO_SUCHFILE'

        embedder.load_model = _fails_quietly
        embedder.is_ready = lambda: False

        brain = Brain(self.db_path)
        try:
            logged = self._errors_for(brain, 'embedder_load_failed')
            self.assertTrue(logged, 'a dead embedder must reach the errors '
                                    'table — print alone is invisible under '
                                    'pytest capture')
            self.assertIn('NO_SUCHFILE', str(logged[0]),
                          'the recorded load_error must ride along, or the '
                          'row says something failed without saying what')
        finally:
            brain.close()

    def test_healthy_embedder_logs_nothing(self):
        """The alarm must not fire on the normal path, or it habituates into
        noise and stops being read."""
        embedder.load_model = lambda config=None: None
        embedder.is_ready = lambda: True

        brain = Brain(self.db_path)
        try:
            self.assertEqual(self._errors_for(brain, 'embedder_load_failed'), [],
                             'a healthy load must be silent')
        finally:
            brain.close()

    def test_skip_embedder_does_not_alarm(self):
        """A brain constructed with skip_embedder never asked for a model —
        the background scale runs and read-only brains use this, and they are
        not broken."""
        embedder.is_ready = lambda: False

        brain = Brain(self.db_path, skip_embedder=True)
        try:
            self.assertEqual(self._errors_for(brain, 'embedder_load_failed'), [],
                             'skip_embedder is a declaration, not a failure')
        finally:
            brain.close()


if __name__ == '__main__':
    unittest.main()
