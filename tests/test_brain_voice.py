"""Tests for BrainVoice — the text the brain hands Claude.

Tests cover:
- format_recall_results
- render_boot() and the format_boot_context() wrapper
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_voice import BrainVoice


class TestBrainVoiceFormatters(unittest.TestCase):
    """Formatting helpers: recall results, encoding warning, suggestions."""

    def test_format_recall_results_includes_all_types(self):
        """Evolution separation removed — all nodes use unified format."""
        results = [
            {"type": "tension", "title": "T1", "content": "c1", "effective_activation": 0.5},
            {"type": "rule", "title": "R1", "content": "c2", "effective_activation": 0.8},
        ]
        lines = []
        BrainVoice.format_recall_results(results, lines)
        text = "\n".join(lines)
        self.assertIn("T1", text)
        self.assertIn("[rule]", text)

    def test_format_recall_results_empty(self):
        lines = []
        BrainVoice.format_recall_results([], lines)
        self.assertEqual(lines, [])


class TestBrainVoiceRenderBoot(BrainTestBase):
    """render_boot() produces valid output and matches wrapper."""

    def test_render_boot_returns_text(self):
        voice = BrainVoice(self.brain)
        result = voice.render_boot()
        self.assertIsInstance(result, str)
        self.assertIn("memories", result)

    def test_context_boot_command_serves_and_records_the_text(self):
        """The daemon command the boot hook calls: the hook's real args in,
        the boot string out, and exactly that string recorded for the
        dashboard's Boot view."""
        from servers.dispatch_read import _handle_context_boot
        with patch.object(self.brain, 'record_boot_render') as record:
            r = _handle_context_boot(self.brain, {'session_id': 'cmd-boot'}, [])
        self.assertTrue(r['ok'])
        self.assertIsInstance(r['result'], str)
        self.assertIn("memories", r['result'])
        record.assert_called_once_with('cmd-boot', r['result'])

    def test_boot_record_stamps_the_sessions_project_and_counterpart(self):
        """The boot record carries where and with whom the session booted —
        the session's own scope (its git project, the operator identity), not
        values a caller sends: stale user/project args are ignored."""
        from servers.dispatch_read import _handle_context_boot
        sid = 'cmd-boot-stamp'
        self.brain.get_or_create_session(sid).set_env(project='entity-test')
        with patch.dict(os.environ, {'BRAIN_OPERATOR_NAME': 'Ada'}):
            _handle_context_boot(
                self.brain, {'session_id': sid, 'user': 'Stale', 'project': 'Glo'}, [])
        row = self.brain.logs_conn.execute(
            "SELECT user, project FROM boot_renders WHERE session_id = ?",
            (sid,)).fetchone()
        self.assertEqual(tuple(row), ('Ada', 'entity-test'))

    def test_direct_fallback_boots_without_delivering(self):
        """The no-daemon fallback prints a boot but never delivers: the hook
        can be killed before it prints, and a delivered-but-unseen ask is lost."""
        import importlib, io, contextlib
        scripts = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'hooks', 'scripts'))
        sys.path.insert(0, scripts)
        try:
            import hook_common
            # boot_brain runs the hook and its error surfacing at import time;
            # stub both so importing it only defines the boot paths.
            with patch.object(hook_common, 'run_hook', lambda name, fn, on_error=None, gated=True: None), \
                 patch.object(hook_common, 'get_unsurfaced_hook_errors',
                              lambda limit=10: []):
                sys.modules.pop('boot_brain', None)
                boot_brain = importlib.import_module('boot_brain')
        finally:
            sys.path.remove(scripts)
        brain = self.brain

        class _Kept:
            # The test's brain, minus close(): BrainTestBase owns teardown.
            def __init__(self, path):
                pass
            def __getattr__(self, name):
                return getattr(brain, name)
            def close(self):
                pass

        out = io.StringIO()
        with patch('servers.brain.Brain', _Kept), \
             patch.object(boot_brain, '_hook_session_id', 'fallback-sess'), \
             patch('servers.channels.delivery.deliver') as deliver, \
             contextlib.redirect_stdout(out):
            boot_brain._boot_via_direct()
        self.assertIn("memories", out.getvalue())
        deliver.assert_not_called()

    def test_render_boot_has_no_brain_tags(self):
        voice = BrainVoice(self.brain)
        text = voice.render_boot()
        # Contract evolved (operator-directed, 2026-08-19): the [BRAIN]
        # envelope is gone from boot. The SKILL.md stance is injected FIRST,
        # the state block follows bare.
        self.assertTrue(text.startswith(voice._load_stance()),
                        "boot must open with the SKILL.md stance")
        self.assertNotIn("[BRAIN]", text)
        self.assertNotIn("[/BRAIN]", text)

    def test_render_boot_keyless_daemon_states_llm_paused(self):
        """The banner carries the DAEMON's LLM state: a keyless daemon must
        say so in the boot output (first laptop install: hook resolved a
        userConfig key the launchd daemon never saw — boot looked healthy
        while the daemon was keyless). Keyed daemons show no such line."""
        from unittest.mock import patch, PropertyMock
        voice = BrainVoice(self.brain)

        keyed = voice.render_boot()
        self.assertNotIn("LLM layer: PAUSED", keyed)

        with patch.object(type(self.brain), 'llm_available',
                          new_callable=PropertyMock, return_value=False):
            keyless = voice.render_boot()
        self.assertIn("LLM layer: PAUSED", keyless)

    def test_llm_paused_notice_names_the_cause_and_its_fix(self):
        """One text for every channel: a missing key → paste it; a refused
        key → check it, re-enabling resumes on its own; a usage limit →
        raising it is enough. A refused key must never read as a missing one."""
        from servers.scales.dispatch import LLM_AUTH_REJECTED, LLM_QUOTA_EXHAUSTED
        keyless = BrainVoice.llm_paused_notice(self.brain)
        self.assertIn("no API key", keyless)
        self.assertIn("paste the key", keyless)

        def refused(kind):
            self.brain._llm_rejected_until = 0.0   # a fresh probe, not a straggler
            self.brain.note_llm_rejected(
                {'kind': kind, 'retry_after': 0, 'detail': 'provider words'},
                where='test')
            return BrainVoice.llm_paused_notice(self.brain)

        auth, quota = refused(LLM_AUTH_REJECTED), refused(LLM_QUOTA_EXHAUSTED)
        for text in (auth, quota):
            self.assertIn("REFUSED", text)
            self.assertIn("provider words", text)
            self.assertNotIn("no API key", text)
        self.assertIn("disabled or rotated", auth)
        self.assertIn("Raising the limit", quota)

    def test_render_boot_refused_key_banner_names_the_refusal(self):
        """The boot channel renders the shared text end to end: a spend-cap
        refusal boots saying so, with the tell-the-operator line, and never
        as a missing key."""
        from servers.scales.dispatch import LLM_QUOTA_EXHAUSTED
        self.brain.note_llm_rejected(
            {'kind': LLM_QUOTA_EXHAUSTED, 'retry_after': 0,
             'detail': 'You have reached your specified API usage limits.'},
            where='test')
        boot = BrainVoice(self.brain).render_boot()
        self.assertIn("REFUSED", boot)
        self.assertIn("Raising the limit", boot)
        self.assertIn("Tell the operator before starting work", boot)
        self.assertNotIn("no API key", boot)

    def test_wrapper_delegates_to_render_boot(self):
        """format_boot_context() produces structurally valid boot output."""
        wrapper = self.brain.format_boot_context()
        # Identity line — "I'm {name} and I have N memories, N locked."
        # (name-less form when BRAIN_AGENT_NAME is unset)
        self.assertIn("I have", wrapper)
        self.assertIn("memories", wrapper)
        self.assertIn("locked", wrapper)

    def test_render_boot_structure(self):
        """Boot output has the first-person identity line, no envelope."""
        voice = BrainVoice(self.brain)
        text = voice.render_boot()
        self.assertNotIn("[BRAIN]", text)
        # Identity line — first person, name-aware
        self.assertIn("I have", text)
        # Stats embedded in the identity line
        self.assertIn("memories", text)

    def test_render_boot_identity_line_uses_agent_name(self):
        """BRAIN_AGENT_NAME fills the identity line; unset → name-less form."""
        from unittest.mock import patch
        voice = BrainVoice(self.brain)
        with patch.dict(os.environ, {"BRAIN_AGENT_NAME": "Anchor"}):
            text = voice.render_boot()
        self.assertIn("I'm Anchor and I have", text)
        with patch.dict(os.environ, {"BRAIN_AGENT_NAME": ""}):
            text = voice.render_boot()
        self.assertNotIn("I'm ", text.split("\n\n")[-1])
        self.assertIn("I have", text)

    def test_boot_section_isolates_a_raise_and_logs_it(self):
        """A boot part that raises is logged under boot_<what>_failed and the
        section yields its default — boot never dies on one part's read."""
        voice = BrainVoice(self.brain)

        def boom():
            raise RuntimeError('part exploded')
        with patch.object(self.brain, '_log_error') as log:
            self.assertEqual(voice._boot_section('frame_build', boom), '')
            self.assertIsNone(voice._boot_section('session_ctx', boom, default=None))
            self.assertEqual(voice._boot_section('fine', lambda: 'text'), 'text')
        self.assertEqual([c.args[0] for c in log.call_args_list],
                         ['boot_frame_build_failed', 'boot_session_ctx_failed'])

    def test_render_boot_stance_first(self):
        """The SKILL.md stance is injected FIRST, before the state block."""
        voice = BrainVoice(self.brain)
        stance = voice._load_stance()
        self.assertTrue(stance, "stance should load from skills/brain/SKILL.md")

        text = voice.render_boot(session_id="stance-order-check")

        # Boot opens with the stance, the state block follows
        self.assertTrue(text.startswith(stance),
                        "boot must open with the identity stance")
        self.assertLess(text.find(stance), text.find("memories"),
                        "stance must sit before the state block")


if __name__ == '__main__':
    unittest.main()
