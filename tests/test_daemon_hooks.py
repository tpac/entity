"""Tests for daemon_hooks.py — hook logic layer.

Tests cover:
- hook_recall() output format (surface-formatted additionalContext)
- Early return behavior (no results = approve)
- Surface integration (mock — no API key in test env)
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.trace_contract import REGISTER_ONLY_FIELD, session_short
from servers.daemon_hooks import (
    hook_recall, post_response_common,
    hook_worktree_context, hook_worktree_cleanup,
)


# Stand-in for the inject text run_surface returns
_MOCK_SURFACE_OUTPUT = (
    'Brain recalled 1 memories:\n\n'
    '[rule] "Test rule for recall" (id:abcd1234, conf:1.0)\n'
    'Content: Important test content\n'
)


def _mock_run_surface(brain, ctx, candidates_data, user_message, **kwargs):
    """Mock surface that returns formatted output for any non-empty candidates."""
    if not candidates_data:
        return None
    return _MOCK_SURFACE_OUTPUT


class TestHookRecallOutput(BrainTestBase):
    """Verify hook_recall() output format."""

    def _call_recall(self, message="test query"):
        """Helper to call hook_recall with standard args."""
        args = {"prompt": message,
                "session_id": "test-hook-recall-output"}
        return hook_recall(self.brain, args, [])

    def _seed_data(self):
        """Add test data so recall has results (avoids early-return approve)."""
        self.brain.remember(type="rule", title="Test rule for recall", content="Important test content")
        self.brain.remember(type="lesson", title="Test lesson", content="We learned something")

    def test_hook_recall_early_return_when_empty(self):
        """No results/signals -> returns approve (no-op)."""
        result = self._call_recall("xyzzy gibberish")
        self.assertEqual(result["json"], {"decision": "approve"})

    def test_hook_recall_keyless_injects_paused_notice_max_3_per_session(self):
        """Keyless daemon: the recall injection must TELL Claude the LLM
        layer is paused (with the instruction to alert the operator) — not
        inject nothing — but at most 3 times per session (operator-set cap:
        can't be missed, doesn't nag a deliberately-keyless session).
        First laptop install: the session never mentioned the missing key;
        the operator found it in the dashboard."""
        from unittest.mock import patch, PropertyMock
        self._seed_data()
        with patch.object(type(self.brain), 'llm_available',
                          new_callable=PropertyMock, return_value=False):
            for i in range(3):
                result = self._call_recall("Test rule for recall")
                ctx = result["json"].get("additionalContext", "")
                self.assertIn("LLM layer: PAUSED", ctx,
                              "notice %d of 3 missing" % (i + 1))
                self.assertIn("tell them", ctx)
            # 4th turn: capped — no notice (plain approve, nothing injected).
            result = self._call_recall("Test rule for recall")
            self.assertNotIn("LLM layer: PAUSED",
                             result["json"].get("additionalContext", ""))

    def test_hook_recall_refused_key_notice_names_the_refusal(self):
        """A key the provider REFUSED is not a missing key: the per-turn
        notice must say which, or it sends the operator to paste the key they
        already have (2026-09-25: a usage-limit refusal read as 'no API key')."""
        from servers.scales.dispatch import LLM_QUOTA_EXHAUSTED
        self._seed_data()
        self.brain.note_llm_rejected(
            {'kind': LLM_QUOTA_EXHAUSTED, 'retry_after': 0,
             'detail': 'You have reached your specified API usage limits.'},
            where='test')
        result = self._call_recall("Test rule for recall")
        ctx = result["json"].get("additionalContext", "")
        self.assertIn("REFUSED", ctx)
        self.assertIn("Raising the limit", ctx)
        self.assertIn("tell them", ctx)
        self.assertNotIn("no API key", ctx)
        self.assertNotIn("paste the key", ctx)

    def test_hook_recall_writes_user_message_trace(self):
        """hook_recall writes the user_message S0 trace at prompt-arrival. The
        write moved here from post_response_common (it precedes recall, so it
        holds even on the empty/early-return path) so presence/peek can surface a
        session's current prompt mid-turn instead of only after the turn completes."""
        sid = 'test-recall-usermsg'
        hook_recall(self.brain, {"prompt": "operator prompt here",
                                 "session_id": sid}, [])
        rows = self.brain.logs_conn.execute(
            "SELECT summary, metadata FROM trace_events WHERE scale='s0' "
            "AND session_id=? AND ref_type='user_message'", (sid,)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIn("operator prompt here", rows[0][0])
        self.assertNotIn(REGISTER_ONLY_FIELD, json.loads(rows[0][1]))

    def test_hook_recall_cues_recall_on_the_prompts_own_row(self):
        """The prompt's first row is the recall's cue: the LAF moment stack
        ends its window just before it, as the Surface's window does."""
        from unittest.mock import patch
        sid = 'test-recall-cue'
        with patch.object(self.brain, 'recall',
                          wraps=self.brain.recall) as rec:
            hook_recall(self.brain, {"prompt": "operator prompt here",
                                     "session_id": sid}, [])
        row = self.brain.logs_conn.execute(
            "SELECT id FROM trace_events WHERE scale='s0' AND session_id=? "
            "AND ref_type='user_message'", (sid,)).fetchone()
        self.assertEqual(rec.call_args.kwargs['cue_trace_id'], row[0])

    def test_register_only_registers_turn_but_skips_recall_and_surface(self):
        """Short answers ('yes') routed with register_only=True must REGISTER the
        turn — user_message S0 trace + conversational classification
        (last_recall_stop == stop_counter) — while skipping the expensive recall
        and Haiku surface. The pre-fix client dropped these turns entirely (no
        trace, misfiled as a heartbeat); the regression here is that registration
        happens without paying for recall/surface."""
        sid = 'test-register-only'
        with patch('servers.daemon_hooks._run_surface') as mock_surface, \
                patch.object(self.brain, 'recall') as mock_recall:
            result = hook_recall(self.brain, {"prompt": "yes", "session_id": sid,
                                              "register_only": True}, [])
        # registered: user_message trace written, stamped as having no
        # recall/surface behind it (the dashboard's Surface card reads this)...
        rows = self.brain.logs_conn.execute(
            "SELECT summary, metadata FROM trace_events WHERE scale='s0' "
            "AND session_id=? AND ref_type='user_message'", (sid,)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIn("yes", rows[0][0])
        self.assertIs(json.loads(rows[0][1])[REGISTER_ONLY_FIELD], True)
        # ...and classified conversational (so Stop writes assistant_message, not heartbeat)
        ctx = self.brain.get_or_create_session(sid)
        self.assertEqual(ctx.last_recall_stop, ctx.stop_counter)
        # ...but recall + Haiku surface were NOT run (the efficiency win)
        mock_recall.assert_not_called()
        mock_surface.assert_not_called()
        self.assertEqual(result["json"].get("decision"), "approve")

    def test_register_only_turn_is_conversational_through_stop(self):
        """Downstream proof: a register-only turn classifies conversational at
        Stop — post_response_common writes the assistant_message half (NOT a
        heartbeat), so the Scribe sees both halves of a short exchange.

        The assert_not_called pair pins this to the register-only branch: turn
        classification (last_recall_stop) is set unconditionally upstream, so the
        Stop assertions alone would pass even if the branch were deleted. Without
        the branch, recall/surface WOULD run on 'yes' — so these two assertions
        are what make the test fail if register-only regresses."""
        sid = 'test-register-only-stop'
        with patch('servers.daemon_hooks._run_surface') as mock_surface, \
                patch.object(self.brain, 'recall') as mock_recall:
            hook_recall(self.brain, {"prompt": "yes", "session_id": sid,
                                     "register_only": True}, [])
        # branch-dependent: register-only must skip the expensive K stages
        mock_recall.assert_not_called()
        mock_surface.assert_not_called()
        post_response_common(self.brain, sid, "ok, shipping it")
        refs = [r[0] for r in self.brain.logs_conn.execute(
            "SELECT ref_type FROM trace_events WHERE scale='s0' "
            "AND session_id=?", (sid,)).fetchall()]
        self.assertIn('user_message', refs)        # the short answer registered
        self.assertIn('assistant_message', refs)   # reply registered (conversational)
        self.assertNotIn('heartbeat', refs)        # NOT misfiled as a wakeup

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_returns_additional_context(self, mock_surface):
        """When results exist and surface selects, returns {'json': {'additionalContext': str}}."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertIn("json", result)
        self.assertIn("additionalContext", result["json"])

    def test_hook_recall_no_system_message(self):
        """systemMessage key is never present in output (dead channel removed)."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertNotIn("systemMessage", result.get("json", {}))

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_has_brain_recalled_header(self, mock_surface):
        """additionalContext contains 'Brain recalled' header from surface output."""
        self._seed_data()
        result = self._call_recall("test rule")
        ctx = result["json"]["additionalContext"]
        self.assertIn("Brain recalled", ctx)
        self.assertIn("memories:", ctx)

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_contains_node_content(self, mock_surface):
        """additionalContext includes node type, title, and content from surface formatting."""
        self._seed_data()
        result = self._call_recall("test rule")
        ctx = result["json"]["additionalContext"]
        self.assertIn("[rule]", ctx)
        self.assertIn("Test rule for recall", ctx)

    @patch('servers.daemon_hooks._run_surface', side_effect=_mock_run_surface)
    def test_hook_recall_judge_failure_returns_approve(self, mock_judge):
        """When judge raises an exception, hook_recall returns approve."""
        mock_judge.side_effect = RuntimeError("API key missing")
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertEqual(result["json"].get("decision"), "approve")

    @patch('servers.daemon_hooks._run_surface', return_value=None)
    def test_hook_recall_judge_returns_none_means_approve(self, mock_judge):
        """When judge returns None (no selection), hook_recall returns approve."""
        self._seed_data()
        result = self._call_recall("test rule")
        self.assertEqual(result["json"].get("decision"), "approve")


class TestTurnClassification(BrainTestBase):
    """post_response_common classifies each stop: conversational (a real prompt
    ran recall this stop → last_recall_stop == stop_counter) vs heartbeat (a
    /watch wakeup re-arm, recall skipped client-side). It advances stop_counter
    (the per-stop SEQUENCE → unique chain IDs), records last_turn_conversational
    (the classification), and writes the right s0 trace type. The
    Scribe's CADENCE is no longer a counter here — it's derived live from the
    user_message traces (see test_turns_since_last_encode_trace_pull); a heartbeat
    writes a `heartbeat` trace, not a user_message, so it can't drag the cadence.
    See trace_contract S0 TURN CLASSIFICATION."""

    needs_embedder = False

    def _s0_refs(self, session_id):
        rows = self.brain.logs_conn.execute(
            "SELECT ref_type FROM trace_events WHERE scale='s0' AND session_id=? "
            "ORDER BY created_at", (session_id,)).fetchall()
        return [r[0] for r in rows]

    def test_conversational_turn_advances_sequence_and_writes_assistant_message(self):
        # post_response_common writes the ASSISTANT half at Stop; the user_message
        # half is now written upstream by hook_recall at prompt-arrival (see
        # TestHookRecallOutput.test_hook_recall_writes_user_message_trace). This
        # test exercises post_response_common in isolation, so only the assistant
        # trace appears here — both share the same chain_id (stop_counter is
        # unchanged between hook_recall and this Stop), so the pair stays grouped.
        sid = 'test-turn-conv'
        ctx = self.brain.get_or_create_session(sid)
        seq_before = ctx.stop_counter
        ctx.last_recall_stop = ctx.stop_counter   # simulate hook_recall having run this stop
        post_response_common(self.brain, sid, "a response")
        self.assertEqual(ctx.stop_counter, seq_before + 1)           # sequence advances
        self.assertTrue(ctx.last_turn_conversational)                # classified conversational
        refs = self._s0_refs(sid)
        self.assertIn('assistant_message', refs)
        self.assertNotIn('heartbeat', refs)

    def test_heartbeat_advances_sequence_but_not_cadence_no_user_message(self):
        sid = 'test-turn-heartbeat'
        ctx = self.brain.get_or_create_session(sid)
        seq_before = ctx.stop_counter
        post_response_common(self.brain, sid, "(watching — inbox empty)")
        self.assertEqual(ctx.stop_counter, seq_before + 1)           # sequence advances → unique chain IDs
        self.assertFalse(ctx.last_turn_conversational)               # classified heartbeat
        refs = self._s0_refs(sid)
        self.assertIn('heartbeat', refs)
        self.assertNotIn('user_message', refs)                       # no user_message → cadence untouched
        self.assertNotIn('assistant_message', refs)

    def test_mixed_sequence_classifies_each_turn(self):
        # real → heartbeat → real: stop_counter advances on ALL (unique chain IDs),
        # but only real turns enter the conversation stream (assistant_message);
        # the heartbeat writes a `heartbeat` trace and never a user/assistant one,
        # so it can't drag the trace-derived cadence.
        sid = 'test-turn-mixed'
        ctx = self.brain.get_or_create_session(sid)
        ctx.last_recall_stop = ctx.stop_counter
        post_response_common(self.brain, sid, "r")
        seq_after_real = ctx.stop_counter
        post_response_common(self.brain, sid, "(watching)")   # heartbeat: no recall mark
        self.assertFalse(ctx.last_turn_conversational)
        self.assertEqual(ctx.stop_counter, seq_after_real + 1)          # sequence advanced
        ctx.last_recall_stop = ctx.stop_counter                         # next real turn
        post_response_common(self.brain, sid, "r")
        self.assertTrue(ctx.last_turn_conversational)
        refs = self._s0_refs(sid)
        self.assertEqual(refs.count('assistant_message'), 2)           # two real turns
        self.assertEqual(refs.count('heartbeat'), 1)                   # one heartbeat, off the cadence


class TestScribeReactor(unittest.TestCase):
    """The poll-driven S1 Scribe trigger — brain.scribe_due() (the decision) and
    BrainDaemon._run_scribe_poll() (single-flight). Replaces the old Stop-hook
    encoder gate: the trigger moved to the daemon poll in the S1 convergence.

    Preserves the guarantees the gate proved — each session fires on ITS OWN
    conversational count, sub-threshold never fires, one encode runs at a time
    across sessions (a busy session re-qualifies and drains on a later poll) —
    and adds the idle-tail clause. scribe_due reads only HIGHER session
    functions (present_sessions + turns_since_last_encode + get_conversation),
    so we stub those: no DB, no SQL.
    """

    def _due(self, sessions, turns_map, now=1_000_000.0, skip=None, boot_time=0.0,
             last_role='assistant', eval_since=None, flush=False):
        """Brain.scribe_due bound to a fake brain whose higher session
        functions are stubbed (present_sessions + turns_since_last_encode +
        get_conversation — the latter two are BrainTracesMixin methods, so
        the fake provides them directly). boot_time defaults ancient (past
        the boot-grace); pass ~now to test it. last_role is the newest
        conversational row's role (default 'assistant' = complete exchange,
        so the 5+ clause's wait-for-answer gate passes); None = empty
        conversation."""
        import types
        from servers.brain import Brain

        class FakeBrain:
            _boot_time = 0.0   # ancient → past the boot-grace window
            # Models a KEYED brain — scribe_due gates on llm_available first
            # (keyless onboarding); the keyless path has its own test below.
            llm_available = True

            def present_sessions(self, window_min, limit):
                return sessions

            def turns_since_last_encode(self, sid):
                return turns_map.get(sid, 0)

            def get_conversation(self, sid, limit=20, with_judge_output=True):
                # Mirrors the real DAL row shape: ref_type is the field the
                # five-plus gate keys on (role maps every incoming
                # correspondent to 'user', so it can't discriminate).
                return ([{'role': last_role,
                          'ref_type': ('assistant_message'
                                       if last_role == 'assistant'
                                       else 'user_message')}]
                        if last_role else [])

            def get_or_create_session(self, sid):
                return types.SimpleNamespace(stop_counter=42)

            def eval_brain_since(self):
                return eval_since

            def _log_error(self, *a, **k):
                pass

        fb = FakeBrain()
        fb._boot_time = boot_time
        return Brain.scribe_due(fb, now=now, skip_sessions=skip, flush=flush)

    def _poll(self, scribe_due_fn, attempts=None, failures=None, fast_forward=0):
        """Drive BrainDaemon._run_scribe_poll against a fake daemon (scribe_due
        stubbed, run_unit_in_background patched to release the lock = instant
        completion). Returns a dict with the spawned units, the captured
        on_complete, logged error sources, and the fake daemon for inspection."""
        import threading
        import types
        from unittest.mock import patch
        from servers.daemon_server import BrainDaemon
        cap = {'spawned': [], 'logged': [], 'on_complete': None, 'encode_runs': 0}

        def fake_run(unit, name, lock, on_complete=None, on_release=None):
            cap['spawned'].append(unit)
            cap['on_complete'] = on_complete
            lock.release()   # mimic the encode thread's finally
            if on_release is not None:
                on_release()  # clears the in-flight slot, as the real finally does

        def _record_encode():
            cap['encode_runs'] += 1

        from servers.scales.s1.encode_contract import MAX_CONCURRENT_ENCODES
        fake = types.SimpleNamespace(
            _encode_sema=threading.BoundedSemaphore(MAX_CONCURRENT_ENCODES),
            _encode_inflight={},
            _encode_hung_warned={},
            _scribe_poll_running=True,
            _scribe_attempts=dict(attempts or {}),
            _scribe_failures=dict(failures or {}),
            _scribe_fast_forward_s=fast_forward,
            brain=types.SimpleNamespace(
                scribe_due=scribe_due_fn,
                _log_error=lambda *a, **k: cap['logged'].append(a[0] if a else None),
                activity=types.SimpleNamespace(record_encode_run=_record_encode)),
        )
        with patch('servers.scales.runner.run_unit_in_background',
                   side_effect=fake_run):
            BrainDaemon._run_scribe_poll(fake)
        cap['fake'] = fake
        return cap

    def test_scribe_due_skips_cooldown_sessions(self):
        # par-B is more overdue but cooling down → par-A is picked instead (a
        # failing/cooling session can't monopolize the poll).
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'par-A', 'live_recency': self._iso(now, 10)},
             {'session_id': 'par-B', 'live_recency': self._iso(now, 10)}],
            {'par-A': ENCODE_EVERY, 'par-B': ENCODE_EVERY + 4}, now=now,
            skip={'par-B'})
        self.assertEqual(due['session_id'], 'par-A')

    def test_poll_records_attempt_and_first_poll_has_empty_cooldown(self):
        seen = {}

        def sd(now=None, skip_sessions=None, flush=False):
            seen['skip'] = set(skip_sessions or ())
            return {'session_id': 'sX', 'counter': 1}
        cap = self._poll(sd)
        self.assertEqual(seen['skip'], set())               # nothing cooling yet
        self.assertIn('sX', cap['fake']._scribe_attempts)   # attempt recorded
        self.assertEqual([u.session_id for u in cap['spawned']], ['sX'])

    def test_poll_cools_down_a_recently_attempted_session(self):
        import time
        # sX attempted just now → must be in the skip set scribe_due receives.
        seen = {}

        def sd(now=None, skip_sessions=None, flush=False):
            seen['skip'] = set(skip_sessions or ())
            return None
        self._poll(sd, attempts={'sX': time.monotonic()})
        self.assertIn('sX', seen['skip'])

    def test_poll_escalates_repeated_failure(self):
        import time
        from servers.scales.s1.encode_contract import (
            SCRIBE_RETRY_COOLDOWN_SECONDS, SCRIBE_MAX_FAILED_RETRIES)
        # sX attempted past its cooldown and STILL due (cadence never advanced)
        # → a re-fire that didn't progress → failures climbs; at threshold, loud.
        old = time.monotonic() - SCRIBE_RETRY_COOLDOWN_SECONDS - 10

        def sd(now=None, skip_sessions=None, flush=False):
            return {'session_id': 'sX', 'counter': 1}   # not cooling → re-fired
        cap = self._poll(sd, attempts={'sX': old},
                         failures={'sX': SCRIBE_MAX_FAILED_RETRIES - 1})
        self.assertEqual(cap['fake']._scribe_failures['sX'],
                         SCRIBE_MAX_FAILED_RETRIES)
        self.assertIn('scribe_repeated_failure', cap['logged'])

    def test_poll_clears_cooldown_on_successful_encode(self):
        # on_complete(write_actions>0) clears the session's cooldown + failure
        # state (and counts the encode toward the S2 gate).
        def sd(now=None, skip_sessions=None, flush=False):
            return {'session_id': 'sX', 'counter': 1}
        cap = self._poll(sd, failures={'sX': 1})
        self.assertIn('sX', cap['fake']._scribe_attempts)   # recorded on fire
        cap['on_complete'](3)                                # encode wrote material
        self.assertNotIn('sX', cap['fake']._scribe_attempts)
        self.assertNotIn('sX', cap['fake']._scribe_failures)
        self.assertEqual(cap['encode_runs'], 1)

    def test_poll_clears_cooldown_on_zero_write_completion(self):
        # A COMPLETED encode that wrote nothing still advanced the cadence
        # (encoding_prompt was written) — so it must clear the cooldown too, or
        # its next legit re-fire would false-trip scribe_repeated_failure. It
        # must NOT count toward the S2 gate (no material written).
        def sd(now=None, skip_sessions=None, flush=False):
            return {'session_id': 'sX', 'counter': 1}
        cap = self._poll(sd, failures={'sX': 2})
        cap['on_complete'](0)                                # completed, 0 writes
        self.assertNotIn('sX', cap['fake']._scribe_attempts)
        self.assertNotIn('sX', cap['fake']._scribe_failures)
        self.assertEqual(cap['encode_runs'], 0)              # no S2-gate credit

    def test_poll_prunes_orphaned_failures(self):
        # _scribe_failures is kept keyed to live _scribe_attempts — an entry for
        # a session no longer being attempted is dropped (no unbounded leak).
        def sd(now=None, skip_sessions=None, flush=False):
            return None                                      # nothing due this poll
        cap = self._poll(sd, failures={'ghost': 5})          # ghost not in attempts
        self.assertNotIn('ghost', cap['fake']._scribe_failures)

    @staticmethod
    def _iso(now, ago_s):
        import datetime
        return datetime.datetime.fromtimestamp(
            now - ago_s, datetime.timezone.utc).isoformat()

    def test_fires_at_threshold(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'gate-A', 'live_recency': self._iso(now, 10)}],
            {'gate-A': ENCODE_EVERY}, now=now)
        self.assertEqual(due, {'session_id': 'gate-A', 'counter': 42})

    def test_skips_below_threshold(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        # Sub-threshold AND recently active → neither clause fires.
        due = self._due(
            [{'session_id': 'gate-B', 'live_recency': self._iso(now, 10)}],
            {'gate-B': ENCODE_EVERY - 2}, now=now)
        self.assertIsNone(due)

    def test_keyless_scribe_never_due(self):
        """Keyless onboarding window: encoding is LLM-driven — scribe_due
        gates on llm_available BEFORE any cadence logic (no presence/turn
        reads) and notes the state via note_llm_unavailable."""
        from servers.brain import Brain

        class KeylessBrain:
            llm_available = False

            def __init__(self):
                self.notes = []

            def note_llm_unavailable(self, where):
                self.notes.append(where)

        kb = KeylessBrain()
        self.assertIsNone(Brain.scribe_due(kb, now=1_000_000.0))
        self.assertEqual(kb.notes, ['S1 Scribe'])

    def test_each_session_on_its_own_count_most_overdue_first(self):
        # Two sessions both at/over threshold, independently — the MOST-overdue
        # (most turns since encode) is picked first; the backlog then drains
        # one-per-poll (single-flight), so neither rides the other's count.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'par-A', 'live_recency': self._iso(now, 10)},
             {'session_id': 'par-B', 'live_recency': self._iso(now, 10)}],
            {'par-A': ENCODE_EVERY, 'par-B': ENCODE_EVERY + 4}, now=now)
        self.assertEqual(due['session_id'], 'par-B')

    def test_idle_tail_fires(self):
        # A session that went quiet below threshold still gets its tail encoded
        # after SCRIBE_TAIL_IDLE_SECONDS — if it has at least
        # SCRIBE_TAIL_MIN_TURNS (two pending turns encode).
        from servers.scales.s1.encode_contract import (
            SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS)
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'tail',
              'live_recency': self._iso(now, SCRIBE_TAIL_IDLE_SECONDS + 60)}],
            {'tail': SCRIBE_TAIL_MIN_TURNS}, now=now)
        self.assertEqual(due['session_id'], 'tail')

    def test_idle_tail_guard_below_min_turns(self):
        # The tail skips a lone leftover turn (below SCRIBE_TAIL_MIN_TURNS) —
        # not worth a Sonnet call.
        from servers.scales.s1.encode_contract import (
            SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS)
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'trivial',
              'live_recency': self._iso(now, SCRIBE_TAIL_IDLE_SECONDS + 60)}],
            {'trivial': SCRIBE_TAIL_MIN_TURNS - 1}, now=now)
        self.assertIsNone(due)

    def test_idle_tail_not_yet_idle_enough(self):
        # Sub-threshold but only briefly idle → the tail hasn't matured yet.
        from servers.scales.s1.encode_contract import SCRIBE_TAIL_MIN_TURNS
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'recent', 'live_recency': self._iso(now, 120)}],
            {'recent': SCRIBE_TAIL_MIN_TURNS}, now=now)
        self.assertIsNone(due)

    def test_5plus_not_swept_when_session_gone_quiet(self):
        # The fix for "picks up tons of old conversations": a session with 5+
        # unencoded turns but idle past the active window is NOT 5+-encoded — it
        # waits for the 1h tail. The same session within the window DOES fire,
        # proving the bound is what gates it, not the turn count.
        from servers.scales.s1.encode_contract import (
            ENCODE_EVERY, SCRIBE_ACTIVE_WINDOW_SECONDS)
        now = 1_000_000.0
        stale = self._iso(now, SCRIBE_ACTIVE_WINDOW_SECONDS + 60)  # quiet, under 1h
        self.assertIsNone(
            self._due([{'session_id': 'quiet', 'live_recency': stale}],
                      {'quiet': ENCODE_EVERY + 3}, now=now),
            '5+ must not sweep a session that has gone quiet')
        fresh = self._iso(now, 10)
        self.assertEqual(
            self._due([{'session_id': 'active', 'live_recency': fresh}],
                      {'active': ENCODE_EVERY + 3}, now=now)['session_id'],
            'active', 'an actively-conversing 5+ session still fires')

    def test_5plus_waits_for_the_answer_to_the_threshold_prompt(self):
        # The turn count crosses ENCODE_EVERY on the USER prompt, so without a
        # completeness gate the 5+ clause always fires mid-turn and the encode
        # window ends on an unanswered question (the 2026-07-11 incident: the
        # Scribe journaled "Turn 5 has no <me> response" because the answer's
        # Stop trace landed ~30s after the snapshot). The 5+ clause must wait
        # until the newest conversational row is the assistant's answer.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        fresh = [{'session_id': 'midturn', 'live_recency': self._iso(now, 10)}]
        turns = {'midturn': ENCODE_EVERY + 1}
        self.assertIsNone(
            self._due(fresh, turns, now=now, last_role='user'),
            '5+ must not fire while the newest turn is unanswered')
        self.assertEqual(
            self._due(fresh, turns, now=now, last_role='assistant')
            ['session_id'], 'midturn',
            'the same session fires once the answer trace lands')

    def test_idle_tail_encodes_a_dangling_question(self):
        # The tail is EXEMPT from the completeness gate: a question still
        # unanswered once the session went quiet (interrupt / disconnect —
        # Stop never fires, so no assistant trace ever lands) is genuinely
        # dangling and belongs in the encode as-is; the journal records the
        # gap honestly.
        from servers.scales.s1.encode_contract import (
            SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS)
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'dangling',
              'live_recency': self._iso(now, SCRIBE_TAIL_IDLE_SECONDS + 60)}],
            {'dangling': SCRIBE_TAIL_MIN_TURNS}, now=now, last_role='user')
        self.assertEqual(due['session_id'], 'dangling')

    def test_boot_grace_suppresses_the_poll(self):
        # Just after a (re)start, the poll is a no-op for the settle window — no
        # backlog flush the instant the daemon comes up.
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        due = self._due(
            [{'session_id': 'active', 'live_recency': self._iso(now, 10)}],
            {'active': ENCODE_EVERY + 3}, now=now, boot_time=now)   # just booted
        self.assertIsNone(due)

    def test_poll_skips_when_all_permits_busy(self):
        # Hard cap: when every encode permit is in use, the poll is a no-op — it
        # never even asks scribe_due. Skipped sessions re-qualify and drain on a
        # later poll (level trigger). Replaces the old global single-flight guard.
        import threading
        import types
        from servers.daemon_server import BrainDaemon
        decided = []
        sema = threading.BoundedSemaphore(1)
        sema.acquire()   # the only permit is taken → cap reached
        fake = types.SimpleNamespace(
            _encode_sema=sema, _encode_inflight={}, _encode_hung_warned={},
            _scribe_poll_running=True,
            brain=types.SimpleNamespace(
                scribe_due=lambda **k: decided.append(1),
                _log_error=lambda *a, **k: None))
        BrainDaemon._run_scribe_poll(fake)
        self.assertEqual(decided, [], 'all permits busy must skip the decision')
        self.assertFalse(fake._scribe_poll_running, 'poll flag must clear')
        self.assertEqual(sema._value, 0, 'the busy permit is untouched')

    def test_poll_excludes_in_flight_session_from_selection(self):
        # Per-session single-flight: a session already encoding is passed to
        # scribe_due in the skip set, so it can't be re-selected mid-encode. Its
        # cadence hasn't reset, so without this it would stay most-overdue and
        # double-encode. A free permit remains, so the decision still runs.
        import threading
        import time
        import types
        from servers.daemon_server import BrainDaemon
        from servers.scales.s1.encode_contract import MAX_CONCURRENT_ENCODES
        seen = {}

        def sd(now=None, skip_sessions=None, flush=False):
            seen['skip'] = set(skip_sessions or ())
            return None   # nothing else due
        fake = types.SimpleNamespace(
            _encode_sema=threading.BoundedSemaphore(MAX_CONCURRENT_ENCODES),
            _encode_inflight={'busy-sess': time.monotonic()},  # mid-encode, not hung
            _encode_hung_warned={},
            _scribe_poll_running=True,
            _scribe_attempts={}, _scribe_failures={},
            brain=types.SimpleNamespace(
                scribe_due=sd, _log_error=lambda *a, **k: None))
        BrainDaemon._run_scribe_poll(fake)
        self.assertIn('busy-sess', seen['skip'],
                      'an in-flight session must be excluded from selection')
        self.assertFalse(fake._scribe_poll_running, 'poll flag must clear')

    # ── eval brain: its own conversation only; tails wait for a flush ────

    def _eval_brain(self, now):
        """A marked brain whose mark is three tail-idle periods old. The
        'inherited' session last spoke before the mark (the operator's tail in
        a copy); 'eval' spoke after it. Both are quiet past the tail-idle
        period."""
        from servers.scales.s1.encode_contract import SCRIBE_TAIL_IDLE_SECONDS
        mark = now - 3 * SCRIBE_TAIL_IDLE_SECONDS
        sessions = [
            {'session_id': 'inherited',
             'live_recency': self._iso(now, 4 * SCRIBE_TAIL_IDLE_SECONDS)},
            {'session_id': 'eval',
             'live_recency': self._iso(now, 2 * SCRIBE_TAIL_IDLE_SECONDS)}]
        return mark, sessions

    def test_eval_brain_idle_tail_waits_for_a_flush(self):
        from servers.scales.s1.encode_contract import SCRIBE_TAIL_MIN_TURNS
        now = 1_000_000.0
        mark, sessions = self._eval_brain(now)
        turns = {'eval': SCRIBE_TAIL_MIN_TURNS}
        self.assertIsNone(self._due(sessions, turns, now=now, eval_since=mark),
                          'time alone must not start an encode on an eval brain')
        due = self._due(sessions, turns, now=now, eval_since=mark, flush=True)
        self.assertEqual(due['session_id'], 'eval')

    def test_eval_brain_never_encodes_a_session_from_before_the_mark(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        mark, sessions = self._eval_brain(now)
        turns = {'inherited': ENCODE_EVERY + 3}
        self.assertIsNone(
            self._due(sessions, turns, now=now, eval_since=mark, flush=True),
            'a flush must not pay for the operator\'s leftover tail')
        # Mid-flow when the copy was made: the five-turn clause's case.
        recent = [{'session_id': 'inherited',
                   'live_recency': self._iso(now, 60)}]
        self.assertIsNone(self._due(recent, turns, now=now, eval_since=now - 30))

    def test_eval_brain_skips_a_session_whose_time_it_cannot_read(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        unreadable = [{'session_id': 'unknown', 'live_recency': ''}]
        turns = {'unknown': ENCODE_EVERY + 3}
        self.assertIsNone(self._due(unreadable, turns, now=now,
                                    eval_since=now - 600))
        # Production keeps its old reading: an unreadable time is "just now".
        self.assertEqual(self._due(unreadable, turns, now=now)['session_id'],
                         'unknown')

    def test_eval_brain_keeps_the_five_turn_rhythm_on_its_own_conversation(self):
        from servers.scales.s1.encode_contract import ENCODE_EVERY
        now = 1_000_000.0
        sessions = [{'session_id': 'eval', 'live_recency': self._iso(now, 10)}]
        due = self._due(sessions, {'eval': ENCODE_EVERY}, now=now,
                        eval_since=now - 600)
        self.assertEqual(due['session_id'], 'eval')

    def test_poll_on_an_eval_brain_starts_no_encode_by_itself(self):
        """The Scribe half of a daemon tick on a marked brain: the operator's
        leftover and the eval's quiet tail are both waiting, and a poll starts
        nothing."""
        import time
        from servers.scales.s1.encode_contract import (
            ENCODE_EVERY, SCRIBE_TAIL_MIN_TURNS)
        mark, sessions = self._eval_brain(time.time())
        turns = {'inherited': ENCODE_EVERY + 3, 'eval': SCRIBE_TAIL_MIN_TURNS}
        cap = self._poll(lambda now=None, skip_sessions=None, flush=False: self._due(
            sessions, turns, now=now, skip=skip_sessions, eval_since=mark,
            flush=flush))
        self.assertEqual(cap['spawned'], [])

    def test_poll_during_a_flush_encodes_only_the_eval_tail(self):
        import time
        from servers.scales.s1.encode_contract import (
            ENCODE_EVERY, SCRIBE_TAIL_IDLE_SECONDS, SCRIBE_TAIL_MIN_TURNS)
        mark, sessions = self._eval_brain(time.time())
        turns = {'inherited': ENCODE_EVERY + 3, 'eval': SCRIBE_TAIL_MIN_TURNS}
        cap = self._poll(lambda now=None, skip_sessions=None, flush=False: self._due(
            sessions, turns, now=now, skip=skip_sessions, eval_since=mark,
            flush=flush), fast_forward=SCRIBE_TAIL_IDLE_SECONDS + 1)
        self.assertEqual([u.session_id for u in cap['spawned']], ['eval'])


class TestScribeFlush(unittest.TestCase):
    """scribe_flush, the settle step's trigger: 'fast-forward idle'. The regular
    poll keeps firing encodes with its permits, single-flight and cooldown; the
    flush only moves its clock ahead by the tail-idle period and waits until
    nothing is due and nothing is encoding (plan 00d314be)."""

    def _daemon(self, pending, attempts=None, inflight=None):
        """A BrainDaemon shell whose brain reports `pending` (a mutable set) as
        due under any clock at or past the tail-idle offset."""
        import types
        from servers.daemon_server import BrainDaemon
        from servers.scales.s1.encode_contract import SCRIBE_TAIL_IDLE_SECONDS
        d = BrainDaemon.__new__(BrainDaemon)     # skip __init__ — no brain/socket
        d._scribe_attempts = dict(attempts or {})
        d._encode_inflight = dict(inflight or {})
        d.clocks = []

        def scribe_due(now=None, skip_sessions=None, flush=False):
            import time as _t
            d.clocks.append(now - _t.time())
            if now - _t.time() < SCRIBE_TAIL_IDLE_SECONDS:
                return None                      # the real clock: nothing tail-due yet
            left = sorted(set(pending) - set(skip_sessions or ()))
            return {'session_id': left[0], 'counter': 1} if left else None
        d.brain = types.SimpleNamespace(scribe_due=scribe_due, llm_available=True)
        return d

    def test_nothing_due_returns_at_once(self):
        d = self._daemon(pending=set())
        r = d._scribe_flush(5)
        self.assertTrue(r['ok'])
        self.assertEqual(r['result']['due_at_start'], [])
        self.assertFalse(r['result']['timed_out'])
        self.assertEqual(getattr(d, '_scribe_fast_forward_s', 0), 0)

    def test_waits_while_the_poll_encodes_then_resets_the_clock(self):
        import threading
        import time
        from servers.scales.s1.encode_contract import SCRIBE_TAIL_IDLE_SECONDS
        pending = {'s1', 's2'}
        d = self._daemon(pending)
        seen_offset = []

        def poll_encodes():
            # Stand-in for the regular tick: it can only find tail work once the
            # flush has moved its clock.
            while not getattr(d, '_scribe_fast_forward_s', 0):
                time.sleep(0.05)
            seen_offset.append(d._scribe_fast_forward_s)
            time.sleep(0.3)
            pending.clear()
        t = threading.Thread(target=poll_encodes)
        t.start()
        r = d._scribe_flush(10)
        t.join()
        self.assertEqual(r['result']['due_at_start'], ['s1', 's2'])
        self.assertEqual(r['result']['failed'], [])
        self.assertFalse(r['result']['timed_out'])
        self.assertGreater(seen_offset[0], SCRIBE_TAIL_IDLE_SECONDS)
        self.assertEqual(d._scribe_fast_forward_s, 0, 'the clock must go back')
        self.assertTrue(all(c > SCRIBE_TAIL_IDLE_SECONDS for c in d.clocks),
                        'every flush decision runs on the moved clock')

    def test_a_failed_encode_comes_back_as_failed_not_a_loop(self):
        import time
        # The poll tried s1, the encode crashed: its attempt stays (cooling), no
        # slot. Still due, but excluded while cooling, so the flush ends.
        d = self._daemon({'s1'}, attempts={'s1': time.monotonic()})
        r = d._scribe_flush(5)
        self.assertEqual(r['result']['failed'], ['s1'])
        self.assertFalse(r['result']['timed_out'])

    def test_an_encode_in_flight_is_waited_for_until_the_timeout(self):
        import time
        d = self._daemon(set(), inflight={'s1': time.monotonic()})
        r = d._scribe_flush(0.6)
        self.assertTrue(r['result']['timed_out'])
        self.assertEqual(r['result']['failed'], [], 'in flight is not failed')
        self.assertEqual(d._scribe_fast_forward_s, 0)

    def test_a_second_flush_is_refused_while_one_runs(self):
        d = self._daemon(set())
        d._scribe_fast_forward_s = 3601
        self.assertFalse(d._scribe_flush(5)['ok'])

    def test_dispatch_routes_scribe_flush_with_its_timeout(self):
        from unittest.mock import patch
        from servers.daemon_server import BrainDaemon
        d = BrainDaemon.__new__(BrainDaemon)
        d.logs = []
        d._log = lambda m: d.logs.append(m)
        with patch.object(BrainDaemon, '_scribe_flush',
                          return_value={'ok': True, 'result': {}}) as f:
            d._dispatch('scribe_flush', {'timeout_s': 42})
        f.assert_called_once_with(42.0)

    def test_poll_runs_on_the_moved_clock(self):
        import threading
        import time
        import types
        from servers.daemon_server import BrainDaemon
        from servers.scales.s1.encode_contract import MAX_CONCURRENT_ENCODES
        seen = {}

        def sd(now=None, skip_sessions=None, flush=False):
            seen['offset'] = now - time.time()
            return None
        fake = types.SimpleNamespace(
            _encode_sema=threading.BoundedSemaphore(MAX_CONCURRENT_ENCODES),
            _encode_inflight={}, _encode_hung_warned={}, _scribe_poll_running=True,
            _scribe_attempts={}, _scribe_failures={}, _scribe_fast_forward_s=3601,
            brain=types.SimpleNamespace(scribe_due=sd, _log_error=lambda *a, **k: None))
        BrainDaemon._run_scribe_poll(fake)
        self.assertGreater(seen['offset'], 3600)

    def test_poll_claims_the_slot_before_recording_the_attempt(self):
        # An attempt with no slot reads as a failed encode to the flush, so the
        # slot must already be held when the attempt lands.
        import threading
        import types
        from unittest.mock import patch
        from servers.daemon_server import BrainDaemon
        from servers.scales.s1.encode_contract import MAX_CONCURRENT_ENCODES
        order = []

        class Slots(dict):          # the poll never rebinds _encode_inflight
            def __setitem__(self, k, v):
                order.append(('slot', k in fake._scribe_attempts))
                super().__setitem__(k, v)
        fake = types.SimpleNamespace(
            _encode_sema=threading.BoundedSemaphore(MAX_CONCURRENT_ENCODES),
            _encode_inflight=Slots(), _encode_hung_warned={}, _scribe_poll_running=True,
            _scribe_attempts={}, _scribe_failures={},
            brain=types.SimpleNamespace(
                scribe_due=lambda now=None, skip_sessions=None, flush=False: {'session_id': 's1', 'counter': 1},
                _log_error=lambda *a, **k: None,
                activity=types.SimpleNamespace(record_encode_run=lambda: None)))

        def fake_run(unit, name, lock, on_complete=None, on_release=None):
            lock.release()
        with patch('servers.scales.s1.scribe.S1Scribe'), \
             patch('servers.scales.runner.run_unit_in_background', side_effect=fake_run):
            BrainDaemon._run_scribe_poll(fake)
        self.assertIn('s1', fake._scribe_attempts)
        self.assertEqual(order, [('slot', False)], 'slot claimed before the attempt')

    def test_asks_as_a_flush(self):
        """On an eval brain only a flush may encode an idle tail, so every
        question the flush itself asks says it is one."""
        import types
        from servers.daemon_server import BrainDaemon
        d = BrainDaemon.__new__(BrainDaemon)
        d._scribe_attempts, d._encode_inflight = {}, {}
        asked = []

        def scribe_due(now=None, skip_sessions=None, flush=False):
            asked.append(flush)
            return None
        d.brain = types.SimpleNamespace(scribe_due=scribe_due, llm_available=True)
        self.assertTrue(d._scribe_flush(1)['ok'])
        self.assertTrue(asked and all(asked), asked)


class TestWorktreeHooks(BrainTestBase):
    """WorktreeCreate/Remove write the per-session worktree identity and NEVER
    emit context on stdout. Claude Code consumes a WorktreeCreate hook's stdout
    as the new worktree path, so the `[BRAIN] GIT CONTEXT` block that used to
    print here got chdir'd into → the `ENOENT chdir '<repo>' -> '[/BRAIN]'`."""

    needs_embedder = False

    def test_worktree_create_stamps_session_and_emits_no_output(self):
        sid = 'wt-create-sess'
        self.brain.reset_session_activity(session_id=sid, cwd='/tmp')  # boot first
        result = hook_worktree_context(
            self.brain,
            {"session_id": sid, "name": "emb-bench", "cwd": "/tmp"}, [])
        # NO stdout leak — empty output, no [BRAIN] marker for CC to chdir into.
        self.assertEqual(result["output"], "")
        # Recorded on the SESSION OBJECT, not the old global config key.
        self.assertEqual(self.brain.session_env_for(sid)["worktree"], "emb-bench")
        self.assertEqual(self.brain.get_config("current_worktree", "SENTINEL"), "SENTINEL")
        # Persisted immediately (not just cached): a fresh load from logs_conn sees it.
        from servers.session_context import SessionContext
        reloaded = SessionContext.load(self.brain._session_state, sid)
        self.assertEqual(reloaded.worktree, "emb-bench")

    def test_worktree_remove_clears_session_worktree(self):
        sid = 'wt-remove-sess'
        self.brain.reset_session_activity(session_id=sid, cwd='/tmp')
        hook_worktree_context(self.brain, {"session_id": sid, "name": "emb-bench", "cwd": "/tmp"}, [])
        self.assertEqual(self.brain.session_env_for(sid)["worktree"], "emb-bench")
        result = hook_worktree_cleanup(self.brain, {"session_id": sid}, [])
        self.assertEqual(result["output"], "")
        self.assertEqual(self.brain.session_env_for(sid)["worktree"], "")

    def test_worktree_create_without_session_id_is_safe(self):
        # No session_id → must NOT touch any session: never fall back to the
        # singleton (which would mis-attribute the worktree to another session) and
        # never create a phantom row. Asserts the invariant, not just output==''.
        from unittest.mock import patch
        with patch.object(self.brain, 'get_or_create_session',
                          wraps=self.brain.get_or_create_session) as m:
            result = hook_worktree_context(self.brain, {"name": "x", "cwd": "/tmp"}, [])
        self.assertEqual(result["output"], "")
        m.assert_not_called()


class TestAnchorTouched(BrainTestBase):
    """The per-turn 'Anchor touched' feed (Piece 3a), driven end-to-end through
    REAL _dispatch — never by hand-appending to ctx.touched (that shortcut stays
    green even when the feature is fully dead, which is exactly how the original
    feature-killer — keying on args['session_id'] instead of the proxy-stamped
    _caller_session — slipped through). Each test exercises the production path:
    dispatch keys the touch onto the caller's ctx; the Stop flush writes it."""

    def _daemon(self):
        from servers.daemon_server import BrainDaemon
        d = BrainDaemon('/tmp/unused-anchor-touched.db')  # no start() → no sockets
        d.brain = self.brain
        return d

    def _touched_rows(self, sid):
        return self.brain.logs_conn.execute(
            "SELECT metadata FROM trace_events WHERE scale='s0' "
            "AND session_id=? AND ref_type='anchor_touched'", (sid,)).fetchall()

    def test_write_keys_by_caller_session_then_flushes_and_resets(self):
        # remember sends NO session_id — only _caller_session. Full path: dispatch
        # keys the new id onto the caller's ctx (the bug: it used to key on the
        # absent session_id → recorded nothing), then the Stop flush writes one
        # anchor_touched delta on the turn chain and resets the accumulator.
        import json
        sid = 'sid-e2e'
        d = self._daemon()
        res = d._dispatch('remember', {'type': 'note', 'title': 'e2e',
                                       'content': 'b', '_caller_session': sid})
        new_id = (res.get('affected') or {}).get('created', [None])[0]
        self.assertTrue(new_id)
        ctx = self.brain.get_or_create_session(sid)
        self.assertIn(new_id, ctx.touched['created'])        # keyed correctly

        post_response_common(self.brain, sid, "done")
        rows = self._touched_rows(sid)
        self.assertEqual(len(rows), 1)
        self.assertIn(new_id, json.loads(rows[0][0])['created'])
        self.assertTrue(all(not v for v in ctx.touched.values()))  # reset

    def test_get_node_records_recalled(self):
        nid = self.brain.remember(type='note', title='look me up', content='x')['id']
        d = self._daemon()
        d._dispatch('get_node', {'node_id': nid, '_caller_session': 'sid-read'})
        self.assertIn(nid, self.brain.get_or_create_session('sid-read').touched['recalled'])

    def test_get_nodes_skips_not_found_error_entries(self):
        nid = self.brain.remember(type='note', title='real one', content='x')['id']
        d = self._daemon()
        d._dispatch('get_nodes', {'node_ids': [nid, 'bogusid0'],
                                  '_caller_session': 'sid-batch'})
        touched = self.brain.get_or_create_session('sid-batch').touched
        self.assertIn(nid, touched['recalled'])
        self.assertNotIn('bogusid0', touched['recalled'])   # error entry skipped

    def test_recall_hot_path_does_not_create_session(self):
        # A non-contributing read must early-out BEFORE get_or_create_session, so
        # it adds no session row and no touched state on the recall hot path.
        from unittest.mock import patch
        d = self._daemon()
        with patch.object(self.brain, 'get_or_create_session',
                          wraps=self.brain.get_or_create_session) as m:
            d._dispatch('query_traces', {'ref_type': 'recall',
                                         '_caller_session': 'sid-hot'})
        m.assert_not_called()

    def test_no_delta_when_nothing_touched(self):
        # A turn with no Anchor tool activity writes no anchor_touched delta.
        sid = 'sid-empty'
        post_response_common(self.brain, sid, "hello")
        self.assertEqual(len(self._touched_rows(sid)), 0)


class TestSeenCandidateDedup(BrainTestBase):
    """Already-surfaced nodes must not re-enter the candidate pool.

    The v13 <shown> element + never-select-again prompt rule did NOT stop
    re-injection (Haiku re-picked a shown node with the element in-prompt,
    2026-07-27 capture) — so the dedup is structural: hook_recall drops
    seen ids from the pool before the final cap. This test pins it.
    """

    def _recall(self, sid, message):
        from servers.daemon_hooks import hook_recall
        return hook_recall(self.brain, {"prompt": message,
                                        "session_id": sid}, [])

    def _wire_prior_surfaced(self, sid, node_id, title):
        """Simulate a prior turn on `sid` whose surface selected node_id:
        a user_message s0 trace carrying recall_chain + the matching
        surface_selected K trace (the shapes get_session_turns joins on)."""
        import json as _json
        chain = 's1r-%s-77' % session_short(sid)
        tconn = self.brain._trace_dal.conn
        tconn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, "
            "metadata, summary, session_id, created_at) "
            "VALUES ('te5t5een', ?, 's0', 'K', 'user_message', ?, 'prior turn', ?, ?)",
            ('s0-%s-77' % session_short(sid),
             _json.dumps({'content': 'earlier question', 'recall_chain': chain}),
             sid, '2026-01-01T00:00:00+00:00'))
        tconn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, "
            "metadata, summary, session_id, created_at) "
            "VALUES ('te5t5el3', ?, 's1', 'K', 'surface_selected', ?, 'sel', ?, ?)",
            (chain,
             _json.dumps({'selected': ['%s|%s' % (node_id[:8], title)]}),
             sid, '2026-01-01T00:00:01+00:00'))
        tconn.commit()

    def test_seen_node_dropped_from_pool(self):
        from unittest.mock import patch
        r = self.brain.remember(type='rule', title='Seen dedup target rule',
                                content='Distinctive dedup target content')
        other = self.brain.remember(type='lesson', title='Seen dedup other lesson',
                                    content='Distinctive dedup target content too')
        sid = 'seen-dedup-sess'
        self._wire_prior_surfaced(sid, r['id'], 'Seen dedup target rule')

        captured = {}

        def spy(brain, ctx, candidates_data, user_message, **kwargs):
            captured['ids'] = [c.get('id') for c in candidates_data]
            return None

        with patch('servers.daemon_hooks._run_surface', side_effect=spy):
            self._recall(sid, 'Distinctive dedup target content')

        ids = captured.get('ids') or []
        self.assertTrue(ids, 'no candidates reached the surface call')
        self.assertNotIn(r['id'], ids,
                         'already-surfaced node re-entered the candidate pool')
        self.assertIn(other['id'], ids,
                      'unseen node should still be in the pool')

    def test_all_seen_pool_skips_surface(self):
        # Every candidate already surfaced this window → skip the Haiku
        # call entirely (approve), never render an empty candidate menu.
        from unittest.mock import patch
        r = self.brain.remember(type='rule', title='All seen rule',
                                content='All seen distinctive content')
        sid = 'all-seen-sess'
        self._wire_prior_surfaced(sid, r['id'], 'All seen rule')

        called = {}

        def spy(brain, ctx, candidates_data, user_message, **kwargs):
            called['yes'] = True
            return None

        fake = {'results': [{'id': r['id'], 'title': 'All seen rule',
                             'score': 0.9, 'type': 'rule'}],
                '_retrieval_stats': {}}
        with patch.object(self.brain, 'recall', return_value=fake), \
                patch('servers.daemon_hooks._run_surface', side_effect=spy):
            out = self._recall(sid, 'all seen distinctive content')

        self.assertEqual(out['json'], {'decision': 'approve'})
        self.assertNotIn('yes', called,
                         'surface must not run on a fully-seen pool')

    def test_fresh_session_pool_untouched(self):
        from unittest.mock import patch
        r = self.brain.remember(type='rule', title='Fresh pool rule',
                                content='Distinctive fresh pool content')
        captured = {}

        def spy(brain, ctx, candidates_data, user_message, **kwargs):
            captured['ids'] = [c.get('id') for c in candidates_data]
            return None

        with patch('servers.daemon_hooks._run_surface', side_effect=spy):
            self._recall('fresh-sess-no-history', 'Distinctive fresh pool content')

        self.assertIn(r['id'], captured.get('ids') or [],
                      'no prior surfacing — nothing should be dropped')




class TestToolMoment(BrainTestBase):
    """hook_pre_edit — memories at an agent's first edit of a file: one
    recall per cue per agent, at most TOOL_MOMENT_MAX_NODES, never a node the
    agent was already shown. Its shows are its own record, not Haiku's picks:
    the prompt surface's shown window never holds them."""
    needs_embedder = False

    PATH = 'servers/auth.py'

    def _edit(self, sid, path=PATH, agent_id=''):
        from servers.daemon_hooks import hook_pre_edit
        return hook_pre_edit(self.brain, {'path': path, 'session_id': sid,
                                          'agent_id': agent_id}, [])

    def _nodes(self, n, tag='auth'):
        return [self.brain.remember(
            type='lesson', title='Tool moment %s node %d' % (tag, i),
            content='About %s, note %d. ' % (tag, i) * 30) for i in range(n)]

    def _recall_returning(self, nodes):
        from unittest.mock import patch
        fake = {'results': [{'id': n['id']} for n in nodes],
                '_retrieval_stats': {}}
        return patch.object(self.brain, 'recall', return_value=fake)

    def _context(self, out):
        return out['json'].get('additionalContext', '')

    def _wire_user_turn(self, sid, chain, shown=()):
        """A user turn whose recall chain is `chain`, plus (when `shown`) the
        prompt surface's surface_selected row on it."""
        self.brain._trace_dal.append(
            chain_id='s0-%s-0' % session_short(sid), scale='s0', event_type='K',
            ref_type='user_message', summary='edit the auth module',
            metadata={'content': 'edit the auth module', 'recall_chain': chain},
            session_id=sid)
        if shown:
            self.brain._trace_dal.append(
                chain_id=chain, scale='s1', event_type='K',
                ref_type='surface_selected',
                ref_id=json.dumps([n['id'][:8] for n in shown]),
                metadata={'selected': ['%s|%s' % (n['id'][:8], 'shown')
                                       for n in shown]},
                session_id=sid)

    def _surfaced(self, sid):
        turns = self.brain.get_conversation(sid, with_surfaced=True,
                                            with_judge_output=False)
        return {s['id'] for t in turns for s in (t.get('surfaced') or [])}

    def test_first_edit_shows_capped_memories(self):
        from servers.scales.s1.surface_contract import TOOL_MOMENT_MAX_NODES
        nodes = self._nodes(TOOL_MOMENT_MAX_NODES + 2)
        with self._recall_returning(nodes) as recall:
            text = self._context(self._edit('tm-first'))
        self.assertEqual(recall.call_args.kwargs['query'],
                         'editing %s' % self.PATH)
        self.assertTrue(text.startswith('Memories about %s:' % self.PATH))
        shown = [n for n in nodes if n['id'][:8] in text]
        self.assertEqual(len(shown), TOOL_MOMENT_MAX_NODES)
        # A bare cut: no count, no get_nodes pointer, no fields beyond the
        # header, content and corrections.
        self.assertIn('…', text)
        self.assertNotIn('get_nodes', text)
        self.assertNotIn('Edges', text)

    def test_second_edit_of_a_file_does_not_recall_again(self):
        nodes = self._nodes(2)
        with self._recall_returning(nodes) as recall:
            self.assertTrue(self._context(self._edit('tm-twice')))
            self.assertEqual(self._edit('tm-twice'), {'json': {}})
        self.assertEqual(recall.call_count, 1)

    def test_an_empty_recall_is_remembered_too(self):
        with self._recall_returning([]) as recall:
            self.assertEqual(self._edit('tm-empty'), {'json': {}})
            self.assertEqual(self._edit('tm-empty'), {'json': {}})
        self.assertEqual(recall.call_count, 1)

    def test_a_node_the_agent_was_shown_is_not_shown_again(self):
        sid = 'tm-seen'
        nodes = self._nodes(3)
        ctx = self.brain.get_or_create_session(sid)
        self._wire_user_turn(sid, ctx.s1r_chain(), shown=nodes[:1])
        with self._recall_returning(nodes):
            text = self._context(self._edit(sid))
        self.assertNotIn(nodes[0]['id'][:8], text)
        self.assertIn(nodes[1]['id'][:8], text)

    def test_a_redirect_to_a_shown_node_is_not_shown_again(self):
        # Recall returns an absorbed id; the canonical pull hands back its
        # survivor, which the agent already saw.
        from unittest.mock import patch
        sid = 'tm-redirect'
        absorbed, survivor = self._nodes(2)
        ctx = self.brain.get_or_create_session(sid)
        self._wire_user_turn(sid, ctx.s1r_chain(), shown=[survivor])
        survivor_node = self.brain.get_node(survivor['id'])
        with self._recall_returning([absorbed]), \
                patch.object(self.brain, 'get_node',
                             return_value={absorbed['id']: survivor_node}):
            out = self._edit(sid)
        self.assertEqual(out, {'json': {}})

    def test_the_moment_shows_stay_out_of_the_surface_window(self):
        # Haiku may still pick what the tool moment showed: its shows are not
        # in the per-turn shown set the prompt surface reads.
        sid = 'tm-own-record'
        nodes = self._nodes(2)
        ctx = self.brain.get_or_create_session(sid)
        self._wire_user_turn(sid, ctx.s1r_chain())
        with self._recall_returning(nodes):
            self.assertTrue(self._context(self._edit(sid)))
        self.assertEqual(self._surfaced(sid), set())

    def test_a_node_shown_for_one_file_is_not_shown_for_the_next(self):
        nodes = self._nodes(3)
        with self._recall_returning(nodes[:2]):
            self._edit('tm-next')
        with self._recall_returning(nodes):
            text = self._context(self._edit('tm-next', path='servers/b.py'))
        self.assertNotIn(nodes[0]['id'][:8], text)
        self.assertIn(nodes[2]['id'][:8], text)

    def test_an_agents_own_working_files_and_data_recall_nothing(self):
        # Shapes from production cues: a path TOOL_MOMENT_SKIP matches never
        # reaches recall and leaves no row, so its first edit stays silent.
        skipped = (
            '/private/tmp/claude-503/-Users-u-repo/0f1e/scratchpad/probe.py',
            '/tmp/entity-audit-x/audit.py',
            '/var/folders/p8/v9/T/tmp.Ey5/drive.py',
            '/Users/u/.claude/projects/-Users-u-repo/memory/MEMORY.md',
            '/Users/u/.claude-second/projects/-Users-u-repo/memory/rule.md',
            '/Users/u/.claude/plans/radiant-otter.md',
            '/Users/u/.claude-second/side-session-notes/local_1/report.md',
            '/Users/u/Library/Application Support/Claude/side-session-notes/'
            'local_1/reader-a.md',
            'web/node_modules/left-pad/index.js',
            '.venv/lib/python3.11/site-packages/x/y.py',
            'uv.lock', 'servers/app.log', 'dist/app.js.map',
            'config/settings.json',
        )
        nodes = self._nodes(1)
        with self._recall_returning(nodes) as recall:
            for i, path in enumerate(skipped):
                with self.subTest(path=path):
                    self.assertEqual(self._edit('tm-skip-%d' % i, path=path),
                                     {'json': {}})
                    self.assertEqual(
                        self.brain.tool_moment_rows('tm-skip-%d' % i), [])
        self.assertEqual(recall.call_count, 0)

    def test_work_files_outside_the_working_directory_still_recall(self):
        recalled = (
            'package.json', 'tmp/fixture.py', '/Users/u/.claude/CLAUDE.md',
            '/Users/u/.claude/skills/brief/SKILL.md',
            '/Users/u/.claude/projects/-Users-u-repo/research/sim/replay.py',
            '/Users/u/repo/.claude/worktrees/w1/servers/a.py',
            '/Users/u/work/.worktrees/km1/api/lib/a.ts',
        )
        nodes = self._nodes(1)
        for i, path in enumerate(recalled):
            with self.subTest(path=path), \
                    self._recall_returning(nodes) as recall:
                self.assertTrue(self._context(
                    self._edit('tm-work-%d' % i, path=path)))
                self.assertEqual(recall.call_count, 1)

    def test_a_subagent_has_its_own_cues_and_shown_set(self):
        sid = 'tm-agent'
        ctx = self.brain.get_or_create_session(sid)
        self._wire_user_turn(sid, ctx.s1r_chain())
        main_nodes = self._nodes(2, tag='main')
        with self._recall_returning(main_nodes):
            self._edit(sid)
        # The subagent's first edit of the same file recalls for it: it never
        # saw what the session's agent was shown.
        with self._recall_returning(main_nodes) as recall:
            text = self._context(self._edit(sid, agent_id='agent-1'))
        self.assertEqual(recall.call_count, 1)
        self.assertIn(main_nodes[0]['id'][:8], text)
        # Its own shows dedup its next file.
        with self._recall_returning(main_nodes + self._nodes(1, tag='b')):
            text = self._context(self._edit(sid, path='servers/b.py',
                                            agent_id='agent-1'))
        self.assertNotIn(main_nodes[0]['id'][:8], text)


class TestToolMomentRecall(BrainTestBase):
    """The real wire: recall on the file cue finds a memory about the file,
    and the render carries its correction."""

    def test_a_memory_about_the_file_arrives_with_its_correction(self):
        from servers.daemon_hooks import hook_pre_edit
        claim = self.brain.remember(
            type='mechanism', title='servers/vault.py hashes secrets with md5',
            content='editing servers/vault.py: the vault hashes every secret '
                    'with md5 before it is stored.')['id']
        fix = self.brain.remember(
            type='correction', title='servers/vault.py moved to argon2',
            content='The md5 hashing in servers/vault.py was replaced.')['id']
        self.brain.connect_typed(fix, claim, relation='corrects',
                                 description='the hash changed after a review')
        out = hook_pre_edit(self.brain, {'path': 'servers/vault.py',
                                         'session_id': 'tm-real'}, [])
        text = out['json'].get('additionalContext', '')
        self.assertIn(claim[:8], text)
        self.assertIn('servers/vault.py moved to argon2', text)


class TestStopHookMintsNoEdges(BrainTestBase):
    """co_accessed retirement guard: the Stop hook's post-response path must
    not create edge_relations rows. This is the writer the retirement
    removed (surface picks → Hebbian pairs → drain); re-wiring any co-access
    writer into the Stop hook must fail here, not pass silently."""
    needs_embedder = False

    def test_conversational_stop_creates_no_edge_relations(self):
        from servers.daemon_hooks import post_response_common
        from servers import recall_write_queue

        session_id = 'test-stop-no-edges'
        ctx = self.brain.get_or_create_session(session_id)
        # Make the turn classify as conversational (a real UserPromptSubmit
        # would have set last_recall_stop via hook_recall).
        ctx.last_recall_stop = ctx.stop_counter

        before = self.brain.conn.execute(
            'SELECT COUNT(*) FROM edge_relations').fetchone()[0]

        post_response_common(self.brain, session_id, 'assistant response')
        recall_write_queue.drain_once(self.brain)

        after = self.brain.conn.execute(
            'SELECT COUNT(*) FROM edge_relations').fetchone()[0]
        self.assertEqual(after, before,
                         'Stop-hook path created edge_relations rows — '
                         'a co-access-style writer is back on the '
                         'post-response path')


class TestSessionlessHookPayload(BrainTestBase):
    """A hook payload that names no session names no session to record under. Both per-turn
    hooks raise (the daemon logs it and the hook scripts fail open) instead of
    recording the turn under another session."""
    needs_embedder = False

    def test_recall_and_stop_hooks_raise_without_session(self):
        from servers.daemon_hooks import hook_post_response_track
        self.brain.reset_session_activity('another-session')
        with self.assertRaises(ValueError):
            hook_recall(self.brain, {'prompt': 'sessionless prompt'}, [])
        with self.assertRaises(ValueError):
            hook_post_response_track(self.brain, {
                'last_assistant_message': 'sessionless reply'}, [])
        rows = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM trace_events WHERE scale='s0' "
            "AND summary LIKE '%sessionless%'").fetchone()[0]
        self.assertEqual(rows, 0)


class TestDispatchFailureReachesTheErrorsTable(BrainTestBase):
    """A command that raises inside the daemon's dispatcher is answered with
    {'ok': False} AND lands in the errors table as its real exception — type
    and traceback — so a failure the caller never reports is still visible."""

    def test_a_raising_command_is_logged_with_its_real_type(self):
        from servers.daemon_server import BrainDaemon
        d = BrainDaemon('/tmp/unused-dispatch-failure.db')  # no start() → no sockets
        d.brain = self.brain
        with patch('servers.daemon_server.dispatch_command',
                   side_effect=RuntimeError('handler exploded')):
            res = d._dispatch('get_node', {'node_id': 'abcd1234'})
        self.assertEqual(res, {'ok': False, 'error': 'handler exploded'})
        match = [e for e in self.brain.get_recent_errors(hours=1)
                 if e['source'] == 'daemon_dispatch']
        self.assertTrue(match, 'a failed command must reach the errors table')
        self.assertEqual(match[0]['type'], 'RuntimeError')
        self.assertEqual(match[0]['error'], 'handler exploded')
        self.assertIn('cmd=get_node', match[0]['context'])


class TestBusyWriteLockIsMarkedNotRun(unittest.TestCase):
    """A write that cannot take the write lock is declined before it runs, and
    the reply says so (NOT_RUN_FIELD) — the MCP proxy resends only replies
    that carry it, so a reply where the command ran must never carry it."""

    def _daemon(self, lock_free):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from servers.daemon_server import BrainDaemon
        d = BrainDaemon('/tmp/unused-busy-lock.db')  # no start() → no sockets
        d._log = lambda *a: None
        d.brain = SimpleNamespace(write_lock=Mock(acquire=Mock(return_value=lock_free)))
        return d

    def test_a_busy_lock_declines_the_write_and_marks_it_not_run(self):
        from servers.contract import NOT_RUN_FIELD
        d = self._daemon(lock_free=False)
        ran = []
        res = d._locked_exec(lambda: ran.append(1), 'remember', {})
        self.assertEqual(ran, [])
        self.assertFalse(res['ok'])
        self.assertIs(res[NOT_RUN_FIELD], True)

    def test_a_write_that_ran_carries_no_mark(self):
        from servers.contract import NOT_RUN_FIELD
        d = self._daemon(lock_free=True)
        res = d._locked_exec(lambda: {'ok': False, 'error': 'refused'}, 'remember', {})
        self.assertNotIn(NOT_RUN_FIELD, res)


class TestHookRecallWritesThePromptReadersPieces(BrainTestBase):
    """The write door for a prompt the hook read (host_contract.read_prompt):
    one S0 row per piece under its speaker's ref_type, the operator's text
    first, recall on the reader's recall text — and a client that predates
    the reader still records the whole prompt as one operator row."""

    STAMP = {"vocab_version": 1, "impl_identity": "abc123abc123"}

    def _rows(self, sid):
        return [(r[0], r[1], json.loads(r[2] or '{}'), r[3]) for r in self.brain.logs_conn.execute(
            "SELECT ref_type, ref_id, metadata, summary FROM trace_events "
            "WHERE scale='s0' AND session_id=? AND event_type='K' ORDER BY rowid", (sid,))]

    def test_a_glued_reminder_becomes_its_own_row_after_the_operator_text(self):
        sid = 'pieces-reminder-0001'
        with patch('servers.daemon_hooks._run_surface'), \
                patch.object(self.brain, 'recall', return_value={'results': []}) as recall:
            hook_recall(self.brain, {
                "prompt": "<system-reminder>\nworktree notice\n</system-reminder>\nthe test fails",
                "session_id": sid, "register_only": False,
                "pieces": [["operator", "the test fails", ""],
                           ["harness", "worktree notice", "<system-reminder>"]],
                "recall_text": "the test fails", **self.STAMP}, [])
        rows = self._rows(sid)
        self.assertEqual([(r[0], r[1]) for r in rows],
                         [('user_message', ''), ('env_message', '<system-reminder>')])
        self.assertEqual(rows[0][3], 'the test fails')       # the operator's row holds only their text
        self.assertIn('recall_chain', rows[0][2])             # the turn's row carries the recall chain
        self.assertNotIn('recall_chain', rows[1][2])
        for _, _, meta, _ in rows:
            self.assertEqual((meta['vocab_version'], meta['impl_identity']), (1, 'abc123abc123'))
        self.assertEqual(recall.call_args.kwargs['query'], 'the test fails')

    def test_a_subagent_report_recalls_on_the_report_and_is_not_the_operator(self):
        sid = 'pieces-handback-0001'
        from servers.channels.thalamus import thalamus
        thalamus.file(self.brain, 'test', 'a due notice for every session', for_whom='all')
        whole = '<agent-message from="x">\n[Subagent hand-back] notice\n  the report body'
        with patch('servers.daemon_hooks._run_surface'), \
                patch.object(self.brain, 'recall', return_value={'results': []}) as recall:
            hook_recall(self.brain, {
                "prompt": whole, "session_id": sid, "register_only": False,
                "pieces": [["subagent", "the report body", "<agent-message"]],
                "recall_text": "the report body", **self.STAMP}, [])
        rows = self._rows(sid)
        self.assertEqual([(r[0], r[3]) for r in rows], [('subagent_message', 'the report body')])
        self.assertEqual(recall.call_args.kwargs['query'], 'the report body')
        # The first-prompt moment waits for the operator: a report is not them arriving.
        self.assertFalse(self.brain.get_or_create_session(sid).prompt_delivered)

    def test_two_reports_in_one_prompt_are_two_rows_in_one_turn(self):
        sid = 'pieces-handback-0002'
        with patch('servers.daemon_hooks._run_surface'), \
                patch.object(self.brain, 'recall', return_value={'results': []}) as recall:
            hook_recall(self.brain, {
                "prompt": "two hand-backs", "session_id": sid, "register_only": False,
                "pieces": [["subagent", "first report", "<agent-message"],
                           ["subagent", "second report", "<agent-message"]],
                "recall_text": "first report\n\nsecond report", **self.STAMP}, [])
        rows = self._rows(sid)
        self.assertEqual([(r[0], r[3]) for r in rows],
                         [('subagent_message', 'first report'), ('subagent_message', 'second report')])
        self.assertIn('recall_chain', rows[0][2])
        self.assertNotIn('recall_chain', rows[1][2])
        chains = {r[0] for r in self.brain.logs_conn.execute(
            "SELECT chain_id FROM trace_events WHERE scale='s0' AND session_id=? "
            "AND event_type='K'", (sid,))}
        self.assertEqual(len(chains), 1)      # the second rides in the first's turn
        self.assertEqual(recall.call_args.kwargs['query'], 'first report\n\nsecond report')

    def test_an_unreadable_hand_back_warns_on_its_own_session(self):
        sid = 'pieces-handback-0003'
        raw = '<agent-message from="x">\n[Subagent hand-back] notice\n</agent-message>'
        with patch('servers.daemon_hooks._run_surface'), patch.object(self.brain, 'recall'):
            hook_recall(self.brain, {
                "prompt": raw, "session_id": sid, "register_only": True,
                "pieces": [["subagent", raw, "<agent-message"]], "recall_text": "", **self.STAMP}, [])
        sessions = [r[0] for r in self.brain.logs_conn.execute(
            "SELECT session_id FROM debug_log WHERE source='hook_recall_extract'")]
        self.assertEqual(sessions, [sid])

    def test_a_harness_notice_registers_without_recall_and_keeps_the_reply(self):
        sid = 'pieces-notice-0001'
        notice = '<task-notification>\n<summary>done</summary>\n</task-notification>'
        with patch('servers.daemon_hooks._run_surface') as surface, \
                patch.object(self.brain, 'recall') as recall:
            hook_recall(self.brain, {
                "prompt": notice, "session_id": sid, "register_only": True,
                "pieces": [["harness", notice, "<task-notification>"]],
                "recall_text": "", **self.STAMP}, [])
        recall.assert_not_called(); surface.assert_not_called()
        rows = self._rows(sid)
        self.assertEqual([r[0] for r in rows], ['env_message'])
        self.assertIs(rows[0][2][REGISTER_ONLY_FIELD], True)
        post_response_common(self.brain, sid, "the background run finished green")
        refs = [r[0] for r in self.brain.logs_conn.execute(
            "SELECT ref_type FROM trace_events WHERE scale='s0' AND session_id=?", (sid,))]
        self.assertIn('assistant_message', refs)    # my reply is conversation, not a heartbeat
        self.assertNotIn('user_message', refs)      # and nothing was recorded as the operator's

    def _warned(self, source):
        return [r[0] for r in self.brain.logs_conn.execute(
            "SELECT session_id FROM debug_log WHERE source=?", (source,))]

    def test_an_undeclared_wrapper_is_recorded_and_logged_on_its_session(self):
        sid = 'pieces-unknown-0001'
        wrapper = '<new-wake-event>\nSomething happened.\n</new-wake-event>'
        with patch('servers.daemon_hooks._run_surface'), patch.object(self.brain, 'recall') as recall:
            hook_recall(self.brain, {
                "prompt": wrapper, "session_id": sid, "register_only": True,
                "pieces": [["harness", wrapper, "<new-wake-event>"]],
                "recall_text": "", **self.STAMP}, [])
        recall.assert_not_called()
        self.assertEqual([(r[0], r[1]) for r in self._rows(sid)],
                         [('env_message', '<new-wake-event>')])
        self.assertEqual(self._warned('unknown_envelope'), [sid])

    def test_a_declared_wrapper_and_untagged_text_are_not_logged(self):
        sid = 'pieces-unknown-0002'
        with patch('servers.daemon_hooks._run_surface'), patch.object(self.brain, 'recall'):
            hook_recall(self.brain, {
                "prompt": "x", "session_id": sid, "register_only": True,
                "pieces": [["harness", "<side-sessions-event>x</side-sessions-event>",
                            "<side-sessions-event>"], ["harness", "unattended text", ""]],
                "recall_text": "", **self.STAMP}, [])
        self.assertEqual(self._warned('unknown_envelope'), [])

    def test_tags_are_checked_against_the_table_this_prompt_was_read_with(self):
        # The session last resolved as codex; this prompt's tells resolve no
        # host, so the hook read every host's table — and so must the check.
        sid = 'pieces-unknown-0003'
        self.brain.get_or_create_session(sid).set_env(host='codex')
        with patch('servers.daemon_hooks._run_surface'), \
                patch.object(self.brain, 'recall', return_value={'results': []}):
            hook_recall(self.brain, {
                "prompt": "a report", "session_id": sid, "register_only": False, "tells": [],
                "pieces": [["subagent", "the report", "<agent-message"]],
                "recall_text": "the report", **self.STAMP}, [])
        self.assertEqual(self._warned('unknown_envelope'), [])

    def test_a_brief_from_another_session_is_recalled_task_first_and_is_not_the_operator(self):
        sid = 'pieces-brief-0001'
        from servers import host_contract as hc
        header = hc._BRIEF_WITH_BACKGROUND
        with patch('servers.daemon_hooks._run_surface'), \
                patch.object(self.brain, 'recall', return_value={'results': []}) as recall:
            hook_recall(self.brain, {
                "prompt": "a brief", "session_id": sid, "register_only": False,
                "pieces": [["harness", "Bring options.", header],
                           ["harness", "Two causes were found.", header],
                           ["harness", "worktree notice", "<system-reminder>"]],
                "recall_text": "Bring options.\n\nTwo causes were found.", **self.STAMP}, [])
        self.assertEqual([(r[0], r[3]) for r in self._rows(sid)],
                         [('env_message', 'Bring options.'),
                          ('env_message', 'Two causes were found.'),
                          ('env_message', 'worktree notice')])
        self.assertEqual(recall.call_args.kwargs['query'], 'Bring options.\n\nTwo causes were found.')
        self.assertFalse(self.brain.get_or_create_session(sid).prompt_delivered)
        self.assertEqual(self._warned('unknown_envelope') + self._warned('hook_recall_extract'), [])

    def test_an_empty_brief_is_an_error_like_an_unreadable_hand_back(self):
        sid = 'pieces-brief-0002'
        from servers import host_contract as hc
        header = hc._BRIEF_WITH_BACKGROUND
        with patch('servers.daemon_hooks._run_surface'), patch.object(self.brain, 'recall'):
            hook_recall(self.brain, {
                "prompt": header, "session_id": sid, "register_only": True,
                "pieces": [["harness", header, header]], "recall_text": "", **self.STAMP}, [])
        self.assertEqual(self._warned('hook_recall_extract'), [sid])
        levels = {r[0] for r in self.brain.logs_conn.execute(
            "SELECT event_type FROM debug_log WHERE source='hook_recall_extract'")}
        self.assertEqual(levels, {'error'})       # surfaced at boot, not just listed

    def test_pieces_the_door_cannot_place_record_the_prompt_as_before_loudly(self):
        sid = 'pieces-bad-0001'
        with patch('servers.daemon_hooks._run_surface'), \
                patch.object(self.brain, 'recall', return_value={'results': []}), \
                patch.object(self.brain, '_log_error') as logged:
            hook_recall(self.brain, {
                "prompt": "hello there", "session_id": sid,
                "pieces": [["stranger", "hello there", ""]], "recall_text": "hello there"}, [])
        self.assertEqual([(r[0], r[3]) for r in self._rows(sid)], [('user_message', 'hello there')])
        self.assertIn('hook_recall_pieces', [c.args[0] for c in logged.call_args_list])

    def _report(self, sid, n):
        whole = '<agent-message from="x">\n[Subagent hand-back] notice\n  report %d' % n
        hook_recall(self.brain, {
            "prompt": whole, "session_id": sid,
            "pieces": [["subagent", "report %d" % n, "<agent-message"]],
            "recall_text": "report %d" % n, **self.STAMP}, [])
        post_response_common(self.brain, sid, "read report %d" % n)

    def test_subagent_reports_bring_an_encode_due_and_harness_notices_do_not(self):
        sid = 'pieces-cadence-0001'
        notice = '<task-notification>\n<summary>done</summary>'
        with patch('servers.daemon_hooks._run_surface', return_value=None), \
                patch.object(self.brain, 'recall', return_value={'results': []}):
            for n in range(3):
                self._report(sid, n)
            hook_recall(self.brain, {
                "prompt": notice, "session_id": sid, "register_only": True,
                "pieces": [["harness", notice, "<task-notification>"]], "recall_text": ""}, [])
            post_response_common(self.brain, sid, "noted")
        self.assertEqual(self.brain.turns_since_last_encode(sid), 3)

    def test_haikus_window_keeps_operator_turns_after_a_run_of_reports(self):
        sid = 'pieces-window-0002'
        with patch('servers.daemon_hooks._run_surface', return_value=None), \
                patch.object(self.brain, 'recall', return_value={'results': []}):
            hook_recall(self.brain, {"prompt": "the operator's earlier question", "session_id": sid}, [])
            post_response_common(self.brain, sid, "earlier answer")
            for n in range(6):
                self._report(sid, n)
        self.brain.remember(type="rule", title="Test rule for recall", content="Important test content")
        with patch('servers.daemon_hooks._run_surface', return_value=None) as surface:
            hook_recall(self.brain, {"prompt": "Test rule for recall", "session_id": sid}, [])
        from servers.trace_contract import row_text, turn_incoming
        window = [row_text(r) for t in surface.call_args.kwargs['recent_turns']
                  for r in turn_incoming(t)]
        self.assertIn("the operator's earlier question", window)

    def test_haikus_conversation_shows_a_report_under_its_speaker(self):
        sid = 'pieces-window-0001'
        self.brain.remember(type="rule", title="Test rule for recall", content="Important test content")
        with patch('servers.daemon_hooks._run_surface', return_value=None), \
                patch.object(self.brain, 'recall', return_value={'results': []}):
            hook_recall(self.brain, {"prompt": "first question about recall", "session_id": sid}, [])
            post_response_common(self.brain, sid, "first answer")
            whole = '<agent-message from="x">\n[Subagent hand-back] notice\n  SUBAGENT REPORT TEXT'
            hook_recall(self.brain, {
                "prompt": whole, "session_id": sid,
                "pieces": [["subagent", "SUBAGENT REPORT TEXT", "<agent-message"]],
                "recall_text": "SUBAGENT REPORT TEXT", **self.STAMP}, [])
            post_response_common(self.brain, sid, "read the report")
        with patch('servers.daemon_hooks._run_surface', return_value=None) as surface:
            hook_recall(self.brain, {"prompt": "Test rule for recall", "session_id": sid}, [])
        from servers.scales.s1.surface_contract import build_surface_prompt
        window = surface.call_args.kwargs['recent_turns']
        self.assertEqual(len(window), 2)        # the operator's turn and the report's
        prompt, _ = build_surface_prompt([], 'Test rule for recall', recent_turns=window)
        conv = prompt[prompt.index('<conversation>\n'):prompt.index('</conversation>')]
        self.assertIn('<user>first question about recall</user>', conv)
        self.assertIn('<subagent>SUBAGENT REPORT TEXT</subagent>\n'
                      '<assistant>read the report</assistant>', conv)


class TestPromptShapesThroughTheEncoderTimeline(BrainTestBase):
    """Every prompt shape the reader types, end to end through the real parts:
    host_contract.read_prompt (as the prompt hook calls it) → hook_recall's
    write door → my reply at Stop → the encoder's lived timeline, read back
    through recall_episodes. Nothing downstream knows the new wrappers; each
    row renders under its speaker's element because the reader typed it.
    Shapes are shortened copies of real prompts (2026-10-16 census)."""
    needs_embedder = False

    WORKTREE = '<system-reminder>\nYou are operating in a git worktree.\n</system-reminder>\n\n'

    def _prompt(self, sid, prompt, reply, unattended=False):
        from servers import host_contract as hc
        intake = hc.read_prompt(prompt, 'claude-code', unattended)
        with patch('servers.daemon_hooks._run_surface'), \
                patch.object(self.brain, 'recall', return_value={'results': []}):
            hook_recall(self.brain, {
                "prompt": prompt, "session_id": sid,
                "register_only": not intake.recall_text,
                "pieces": [list(p) for p in intake.pieces],
                "recall_text": intake.recall_text, "tells": ["CLAUDE_CODE_SESSION_ID"],
                "vocab_version": hc.VOCAB_VERSION, "impl_identity": hc.contract_fingerprint()}, [])
        post_response_common(self.brain, sid, reply)

    def _timeline(self, sid):
        """[[(element, text), …] per turn] as the encoder renders its window."""
        import html
        import re
        from servers.scales.s1.encode import _render_lived_sequence_timeline, _window
        out = _render_lived_sequence_timeline(self.brain, sid, _window(self.brain, sid))
        voices = re.compile(r'  <(other|harness|subagent|session|brain|me)\b[^>]*>(.*?)</\1>\n', re.S)
        return [[(el, html.unescape(text)) for el, text in voices.findall(turn)]
                for turn in out.split('</turn>')[:-1]]

    def _check(self, sid, steps, expected):
        for prompt, unattended in steps:
            self._prompt(sid, prompt, 'my reply to: %s' % prompt[:20], unattended)
        turns = self._timeline(sid)
        self.assertEqual([[el for el, _ in t] for t in turns], [[el for el, _ in t] for t in expected])
        for turn, want in zip(turns, expected):
            for (el, text), (_, snippet) in zip(turn, want):
                self.assertIn(snippet, text, (el, text))

    def test_the_operators_words_and_their_reminder(self):
        self._check('tl-operator', [(self.WORKTREE + 'Why does the soak test hang?', False)], [
            [('other', 'Why does the soak test hang?'),
             ('harness', 'You are operating in a git worktree.'), ('me', 'my reply')]])

    def test_an_attachment_leading_the_operators_words_is_theirs(self):
        prompt = ('<artifact-view-context artifact="138b">{"tab": 1}</artifact-view-context>\n'
                  'what is this chart?')
        self._check('tl-attachment', [(prompt, False)], [
            [('other', 'what is this chart?'), ('me', '')]])

    def test_a_side_session_wake_is_the_harness_turn_with_its_notes(self):
        prompt = ('<system-reminder>\nThe session "ST1" you started finished a turn.\n</system-reminder>\n'
                  '<system-reminder>\nLinked sessions now (status, not an instruction): st1 (idle).\n'
                  '</system-reminder>\n\n<side-sessions-event>None of the sessions you started is '
                  'working now.</side-sessions-event>')
        self._check('tl-wake', [(prompt, False)], [
            [('harness', '<side-sessions-event>None of the sessions'),
             ('harness', 'The session "ST1" you started'), ('harness', 'Linked sessions now'),
             ('me', '')]])

    def test_an_auto_fix_report_is_a_harness_turn(self):
        prompt = ('<ci-monitor-event>"Auto-fix pull requests" (the desktop app) is watching org/repo '
                  'PR #31.\n1 new review comment.\n</ci-monitor-event>')
        self._check('tl-ci', [(prompt, False)], [[('harness', 'is watching org/repo'), ('me', '')]])

    def test_a_brief_from_another_session_leads_with_its_task(self):
        from servers import host_contract as hc
        prompt = (self.WORKTREE + hc._BRIEF_WITH_BACKGROUND + '\nThe operator asked why.\n\n'
                  'Task:\nResearch first. Bring options.')
        self._check('tl-brief', [(prompt, False)], [
            [('harness', 'Research first. Bring options.'), ('harness', 'The operator asked why.'),
             ('harness', 'You are operating in a git worktree.'), ('me', '')]])

    def test_a_message_from_another_session_keeps_its_sender_and_the_note(self):
        prompt = ('<cross-session-message from="local_a" name="A">\nplease rebase\n'
                  '</cross-session-message>\n\nThis came from another Claude session.')
        self._check('tl-message', [(prompt, False)], [
            [('harness', '<cross-session-message from="local_a" name="A">\nplease rebase'),
             ('harness', 'This came from another Claude session.'), ('me', '')]])

    def test_an_undeclared_wrapper_is_a_harness_turn(self):
        self._check('tl-unknown', [('<new-wake-event>\nSomething happened.\n</new-wake-event>', False)],
                    [[('harness', '<new-wake-event>'), ('me', '')]])

    def test_an_unattended_sessions_text_is_a_harness_turn(self):
        self._check('tl-unattended', [('Review the campaign draft.', True)],
                    [[('harness', 'Review the campaign draft.'), ('me', '')]])

    def test_a_subagent_report_is_a_subagent_turn(self):
        prompt = '<agent-message from="x">\n[Subagent hand-back] notice\n  the report body\n</agent-message>'
        self._check('tl-subagent', [(prompt, False)], [[('subagent', 'the report body'), ('me', '')]])

    def test_machine_turns_sit_between_the_operators_turns_without_borrowing_them(self):
        wake = '<side-sessions-event>None of the sessions you started is working now.</side-sessions-event>'
        self._check('tl-sequence', [('Start the two side sessions.', False), (wake, False),
                                    ('What did they find?', False)], [
            [('other', 'Start the two side sessions.'), ('me', '')],
            [('harness', '<side-sessions-event>'), ('me', '')],
            [('other', 'What did they find?'), ('me', '')]])

    def test_only_the_operators_words_are_operator_dialogue(self):
        from servers import host_contract as hc
        sid = 'tl-dialogue'
        for prompt in ('<ci-monitor-event>x</ci-monitor-event>',
                       hc._BRIEF_TASK_ONLY + '\nFix the flaky test.',
                       '<cross-session-message from="local_a" name="A">\nhi\n</cross-session-message>',
                       'Is the fix in?'):
            self._prompt(sid, prompt, 'ok')
        said = [e['summary'] for e in self.brain.recall_episodes(session_id=sid)['episodes']
                if e.get('ref_type') == 'user_message']
        self.assertEqual(said, ['Is the fix in?'])


if __name__ == '__main__':
    unittest.main()
