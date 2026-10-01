"""Regression: a test brain must NOT leak aspect-registry heals into the
aspects_v1.json outside it.

Bug (observed 2026-06-16): ASPECTS_JSON_PATH was a module-level constant
resolved at IMPORT time — before the brain env set BRAIN_DB_DIR in __enter__.
So a top-level `from servers.scales.s2... import ...` froze the path at the
file the process resolved on startup, and the seed reconcile
(reconcile_working_copy, run at Brain.__init__ via AspectRegistry) healed that
file even though the DB was isolated.

Fix: resolve the aspects path at CALL time (aspects_json_path()), so a later
os.environ change takes effect. The brain env additionally pins
ASPECTS_JSON_PATH into its own temp dir.

These assert both halves: the pin lands INSIDE the env's temp dir, and the
file outside it is byte-for-byte unchanged across a session that builds a
Frame (which loads the registry and runs the seed/self-heal path).
"""
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s2.aspect_contract import aspects_json_path  # noqa: E402
from servers.scales.s1.frame import build_frame  # noqa: E402 — top-level import is the repro trigger
from tests.isolated_brain import brain_env  # noqa: E402


def _outer_aspects_path():
    """The aspects file this process resolves when no brain env is active.

    Under pytest that is conftest's announced session pin; the operator's own
    taxonomy is never in reach of the suite (the production clone lives in
    eval/). Either way it is "the file a leak would land in", which is the
    thing this module exists to protect.
    """
    return (os.environ.get('BRAIN_TEST_ASPECTS_DEFAULT_PIN')
            or os.environ.get('ASPECTS_JSON_PATH'))


def _snapshot(path):
    """(bytes, mtime) or None when the file does not exist.

    Absence is a state worth pinning too: a heal that CREATES the outer file
    is the same leak as one that rewrites it.
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return f.read(), os.path.getmtime(path)


class TestAspectsPathIsolation(unittest.TestCase):
    def test_test_brain_does_not_touch_the_outer_aspects_file(self):
        outer = _outer_aspects_path()
        self.assertTrue(outer, 'no outer ASPECTS_JSON_PATH — nothing to protect')
        before = _snapshot(outer)

        with brain_env(load_env=False) as env:
            # The aspects path must resolve INTO the temp dir. Asserting
            # containment rather than "not the outer file" is what keeps this
            # honest: a path that is neither would also leak, just elsewhere.
            resolved = os.path.abspath(aspects_json_path())
            self.assertTrue(
                resolved.startswith(os.path.abspath(env.db_dir) + os.sep),
                'aspects_json_path() resolved outside the isolated dir: %s'
                % resolved)
            # Exercise the registry load + seed/heal path. (Was build_frame()
            # alone until 2026-08-05 — the lean Frame no longer touches
            # brain.aspects, so the repro trigger moved to an explicit
            # registry access plus a Frame build for the original
            # import-time-freeze shape.)
            self.assertIsNotNone(env.brain.aspects.by_name('wisdom'))
            build_frame(env.brain, session_id='isolation-test')

        self.assertEqual(before, _snapshot(outer),
                         'the test brain modified the aspects file outside it')

    def test_test_brain_honors_preset_aspects_override(self):
        """A caller that pins ASPECTS_JSON_PATH before entering owns the
        aspects location — the brain env must NOT clobber it.

        run_aspect_cycles_on_clone.py relies on this: it points
        ASPECTS_JSON_PATH at its own work file so the encoder classifies into
        that file. If the env overwrites the variable, the eval reads back an
        unchanged file and silently reports wrong results.
        """
        import tempfile
        work = os.path.join(tempfile.mkdtemp(prefix='aspects_override_'),
                            'aspects_v1.json')
        prior = os.environ.get('ASPECTS_JSON_PATH')
        os.environ['ASPECTS_JSON_PATH'] = work
        try:
            with brain_env(load_env=False):
                self.assertEqual(os.environ.get('ASPECTS_JSON_PATH'), work,
                                 "the brain env clobbered the caller's "
                                 "ASPECTS_JSON_PATH override")
                self.assertEqual(os.path.abspath(aspects_json_path()),
                                 os.path.abspath(work))
            # Restored to the caller's value on exit (not popped).
            self.assertEqual(os.environ.get('ASPECTS_JSON_PATH'), work)
        finally:
            # Restore, don't pop — popping would strip the conftest
            # live-file guard from every test after this one.
            if prior is None:
                os.environ.pop('ASPECTS_JSON_PATH', None)
            else:
                os.environ['ASPECTS_JSON_PATH'] = prior
            shutil.rmtree(os.path.dirname(work), ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
