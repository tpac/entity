"""
System-level tests — hook functions + daemon dispatch integrity.

Tests the hook functions DIRECTLY (no subprocess, no daemon required)
and verifies the daemon's dispatch table is complete.

Run: python3 -m pytest tests/test_system.py -v
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from servers.brain import Brain
from servers.daemon_server import BrainDaemon
from servers.daemon_hooks import (
    hook_recall, hook_pre_edit,
)
from tests.brain_test_base import BrainTestBase


def _seed_realistic_brain(brain):
    """Seed a brain with realistic data for hook tests."""
    nodes = {}
    nodes['auth_rule'] = brain.remember(
        type='rule', title='Auth: never store raw passwords',
        content='Always hash passwords with bcrypt. Never store plaintext.',
        keywords='auth password security bcrypt', locked=True)
    nodes['decision'] = brain.remember(
        type='decision', title='Use Arctic v1.5 for embeddings',
        content='Chose Arctic v1.5 because it balances quality and speed.',
        keywords='embedder arctic decision')
    nodes['lesson'] = brain.remember(
        type='lesson', title='Silent failures are the worst kind',
        content='Always log errors loudly. Silent except:pass hides bugs.',
        keywords='error handling logging silent')
    brain.save()
    return nodes


class TestHookFunctions(BrainTestBase):
    """Direct tests of hook functions with a real Brain instance."""

    def setUp(self):
        super().setUp()
        _seed_realistic_brain(self.brain)

    def test_recall_returns_json(self):
        """hook_recall should return a dict with 'json' key."""
        result = hook_recall(self.brain, {'prompt': 'embedding model choice',
                                          'session_id': 'test-system-recall'}, [])
        self.assertIn('json', result)

    def test_pre_edit_never_decides(self):
        """The tool moment informs, it never decides: a json envelope with at
        most additionalContext, never a permission decision."""
        result = hook_pre_edit(self.brain, {
            'path': 'auth.py', 'session_id': 'test-system-edit'}, [])
        self.assertIn('json', result)
        self.assertNotIn('decision', result['json'])


class TestDaemonDispatch(unittest.TestCase):
    """Daemon dispatch table integrity — no subprocess needed."""

    def test_hook_table_covers_all_hooks(self):
        """HOOK_TABLE has entries for all hook functions."""
        expected_hooks = [
            'hook_recall', 'hook_post_response_track',
            'hook_pre_edit',
            'hook_session_end', 'hook_stop_failure_log',
            'hook_worktree_context', 'hook_worktree_cleanup',
        ]
        for hook_name in expected_hooks:
            self.assertIn(hook_name, BrainDaemon.HOOK_TABLE,
                         f"Missing hook in HOOK_TABLE: {hook_name}")

    def test_hook_table_functions_exist(self):
        """All cmd names in HOOK_TABLE resolve to callables in daemon_hooks.

        HOOK_TABLE entries are (is_write, marks_dirty); the daemon does
        getattr(_hooks, cmd), so the cmd name itself must be the function
        name in daemon_hooks.
        """
        import servers.daemon_hooks as dh
        for hook_name, _meta in BrainDaemon.HOOK_TABLE.items():
            self.assertTrue(hasattr(dh, hook_name),
                           f"daemon_hooks missing: {hook_name}")

    def test_hook_table_dirty_flags(self):
        """Every hook marks the brain dirty."""
        non_dirty = [name for name, (_iw, dirty) in BrainDaemon.HOOK_TABLE.items() if not dirty]
        self.assertEqual(non_dirty, [])


if __name__ == '__main__':
    unittest.main()
