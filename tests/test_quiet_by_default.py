#!/usr/bin/env python3
"""A brain born from this repo ships with every debug and diagnostic switch off.

Each switch is an operator opt-in that grows a log, a table or a directory
for as long as it stays on, so a fresh install must start quiet: turning one
on is a choice made on that brain (brain_meta, an interaction version or an
env var), never a default the plugin carries. One place lists them all — a
new switch belongs here too.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts'))

from tests.brain_test_base import BrainTestBase

# Env overrides an operator may have exported; a default is what's left without them.
_DEBUG_ENV = ('BRAIN_DEBUG', 'BRAIN_TRACE_MODE', 'BRAIN_SURFACE_CAPTURE')


class TestFreshBrainShipsQuiet(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        # Before the base setUp: cleanups run after its tearDown, so this
        # snapshot restores the env the test started with.
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in _DEBUG_ENV:
            os.environ.pop(key, None)
        super().setUp()

    def test_memory_watchdog_is_off(self):
        from servers.memory_watchdog import MemoryWatchdog
        self.assertIsNone(MemoryWatchdog.maybe_start(self.brain))

    def test_debug_mode_is_off(self):
        self.assertEqual(self.brain.get_debug_status(), {'debug_enabled': False})

    def test_hooks_see_debug_mode_off(self):
        import hook_common
        with patch.object(hook_common, 'db_path', self.brain.db_path), \
             patch.object(hook_common, '_debug_mode_cache', None):
            self.assertFalse(hook_common.is_debug_mode())

    def test_trace_rows_use_normal_caps(self):
        from servers.trace_contract import TRACE_MODES, trace_detail
        self.assertEqual(trace_detail(), TRACE_MODES['normal'])

    def test_surface_capture_is_off(self):
        from servers.scales.s1 import surface_capture
        self.assertIsNone(surface_capture.capture_dir(self.brain))

    def test_surface_capture_opt_in_turns_it_on(self):
        # The other half of the switch, through the real brain_meta read — an
        # "off" that nothing can turn on would pass the test above vacuously.
        from servers.scales.s1 import surface_capture
        self.brain.set_config('surface_capture.enabled', 'true')
        self.assertIsNotNone(surface_capture.capture_dir(self.brain))

    def test_payload_recording_is_normal(self):
        from servers.trace_contract import TRACE_RECORDING_NORMAL
        self.assertEqual(self.brain.get_interaction_effective('trace_recording')['config'],
                         TRACE_RECORDING_NORMAL)


if __name__ == '__main__':
    unittest.main()
