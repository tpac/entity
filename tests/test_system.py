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
    hook_recall, hook_pre_edit, hook_pre_bash_safety,
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

    def test_pre_edit_approves(self):
        """Pre-edit hook should approve and return decision."""
        result = hook_pre_edit(self.brain, {
            'filename': 'auth.py', 'tool_name': 'Edit'}, [])
        self.assertIn('json', result)
        self.assertEqual(result['json'].get('decision'), 'approve')

    def test_pre_bash_returns_decision(self):
        """Pre-bash safety should return a decision for destructive commands."""
        result = hook_pre_bash_safety(self.brain, {
            'command': 'rm -rf /tmp/important'}, [])
        self.assertIn('json', result)
        self.assertIn(result['json'].get('decision'), ['approve', 'block'])

    def test_pre_bash_approves_safe(self):
        """Non-destructive commands should be approved — assert the decision,
        not just the envelope. The old check only verified 'json' was present,
        so it passed regardless of whether a safe command was approved."""
        result = hook_pre_bash_safety(self.brain, {
            'command': 'ls -la /tmp'}, [])
        self.assertIn('json', result)
        self.assertEqual(result['json'].get('decision'), 'approve')


class TestDaemonDispatch(unittest.TestCase):
    """Daemon dispatch table integrity — no subprocess needed."""

    def test_hook_table_covers_all_hooks(self):
        """HOOK_TABLE has entries for all hook functions."""
        expected_hooks = [
            'hook_recall', 'hook_post_response_track',
            'hook_pre_edit', 'hook_pre_bash_safety',
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
        """Only pre_bash_safety should be non-dirty (read-only)."""
        non_dirty = [name for name, (_iw, dirty) in BrainDaemon.HOOK_TABLE.items() if not dirty]
        self.assertEqual(non_dirty, ['hook_pre_bash_safety'])


if __name__ == '__main__':
    unittest.main()
