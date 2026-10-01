"""Tests for the LLM rejection latch — pausing on a REFUSED provider call.

Background: an external install's key was disabled mid-session on 2026-08-13.
`llm_available` tested key PRESENCE only, so the gate stayed open and the S1
Scribe re-fired against a dead key every 120s for over a day — 293 consecutive
401s, ~1700 error rows, and a brain that silently stopped remembering.

Two halves, tested separately:
- `classify_llm_failure` (dispatch): provider exception → closed vocabulary.
  Pure, SDK-free, so every branch is exercisable without a network or a client.
- the brain latch: auth/quota refusals pause every LLM feature at once, expire
  on a ladder, and reset without the key's VALUE ever changing (the operator
  re-enables the same key — a value-change trigger would never fire).
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_constants import (
    LLM_REJECT_BACKOFF_MINUTES,
    LLM_REJECT_STRIKE_RESET_SECONDS,
)
from servers.scales.dispatch import (
    classify_llm_failure,
    LLM_AUTH_REJECTED, LLM_QUOTA_EXHAUSTED, LLM_RATE_LIMITED,
    LLM_INVALID_REQUEST, LLM_TRANSIENT, LLM_UNKNOWN,
)


class _FakeResponse:
    def __init__(self, status_code=None, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _FakeAPIError(Exception):
    """Stands in for an SDK error: message + status, like every provider's."""

    def __init__(self, message, status_code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(status_code, headers)


# Verbatim from the daemon's 2026-09-25 surface failure (request id elided).
_QUOTA_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': "
    "'invalid_request_error', 'message': 'You have reached your specified API "
    "usage limits. You will regain access on 2026-10-01 at 00:00 UTC.'}, "
    "'request_id': 'req_x'}")


class APIConnectionError(Exception):
    """Name-matched by the classifier — the connectivity family carries no
    status code, so the class name is the only portable signal."""


class ClassifyLLMFailureTests(unittest.TestCase):
    """The vocabulary mapping. Messages are the real ones seen in production."""

    def test_401_is_auth_rejected(self):
        exc = _FakeAPIError(
            "Error code: 401 - {'error': {'message': 'API key is invalid'}}",
            status_code=401)
        self.assertEqual(classify_llm_failure(exc)['kind'], LLM_AUTH_REJECTED)

    def test_403_is_auth_rejected(self):
        self.assertEqual(
            classify_llm_failure(_FakeAPIError('forbidden', 403))['kind'],
            LLM_AUTH_REJECTED)

    def test_usage_limit_400_is_quota_and_detail_keeps_the_reset_date(self):
        # The notice shows the provider's own words, so the date it names
        # must survive the detail cut — this is the production message.
        exc = _FakeAPIError(_QUOTA_400, status_code=400)
        outcome = classify_llm_failure(exc)
        self.assertEqual(outcome['kind'], LLM_QUOTA_EXHAUSTED)
        self.assertIn('regain access on 2026-10-01', outcome['detail'])

    def test_plain_400_is_invalid_request_not_quota(self):
        # The prompt-too-long case: same status as a quota refusal, and it must
        # NOT pause the whole brain — it's a property of one payload.
        exc = _FakeAPIError('prompt is too long: 1768459 tokens > 1000000',
                            status_code=400)
        self.assertEqual(classify_llm_failure(exc)['kind'], LLM_INVALID_REQUEST)

    def test_429_is_rate_limited_and_carries_retry_after(self):
        exc = _FakeAPIError('rate limited', 429, headers={'retry-after': '42'})
        outcome = classify_llm_failure(exc)
        self.assertEqual(outcome['kind'], LLM_RATE_LIMITED)
        self.assertEqual(outcome['retry_after'], 42)

    def test_5xx_and_connection_family_are_transient(self):
        self.assertEqual(
            classify_llm_failure(_FakeAPIError('overloaded', 529))['kind'],
            LLM_TRANSIENT)
        self.assertEqual(
            classify_llm_failure(APIConnectionError('connection refused'))['kind'],
            LLM_TRANSIENT)

    def test_non_llm_exception_is_unknown(self):
        # _log_error hands every exception to the classifier, so a disk error
        # or a JSON bug must pass straight through.
        self.assertEqual(classify_llm_failure(ValueError('bad json'))['kind'],
                         LLM_UNKNOWN)
        self.assertEqual(classify_llm_failure(None)['kind'], LLM_UNKNOWN)

    def test_status_recovered_from_message_when_attribute_is_absent(self):
        # A wrapper that preserved only the string still classifies.
        bare = Exception("Error code: 401 - authentication_error")
        self.assertEqual(classify_llm_failure(bare)['kind'], LLM_AUTH_REJECTED)

    def test_wrapped_cause_is_classified(self):
        # RunLoopError wraps mid-run failures and has no status of its own; a
        # round-2 401 would otherwise read as unknown and never latch.
        wrapper = RuntimeError('AuthenticationError: run failed at round 2')
        wrapper.__cause__ = _FakeAPIError('API key is invalid', 401)
        self.assertEqual(classify_llm_failure(wrapper)['kind'], LLM_AUTH_REJECTED)


class LLMRejectionLatchTests(BrainTestBase):
    """The brain-side gate. Requires a key to be PRESENT — the whole point is
    that presence is no longer sufficient."""

    needs_embedder = False

    def setUp(self):
        super().setUp()
        # Point key resolution at an empty config dir so the env var (not the
        # developer's real ~/.config/brain/env) is what resolves.
        self._env_backup = {k: os.environ.get(k)
                            for k in ('XDG_CONFIG_HOME', 'ANTHROPIC_API_KEY')}
        os.environ['XDG_CONFIG_HOME'] = self.tmp
        os.environ['ANTHROPIC_API_KEY'] = 'sk-test-key-present'

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()

    def _reject(self, kind=LLM_AUTH_REJECTED):
        self.brain.note_llm_rejected(
            {'kind': kind, 'retry_after': 0,
             'detail': 'API key is invalid'}, where='test')

    def test_present_key_is_available_until_rejected(self):
        self.assertTrue(self.brain.llm_available)
        self._reject()
        self.assertFalse(self.brain.llm_available)

    def test_latch_expires_without_the_key_value_changing(self):
        # The operator disables and re-enables the SAME key — nothing about the
        # key's value changes, so only the clock can lift the pause.
        self._reject()
        self.assertFalse(self.brain.llm_available)
        self.brain._llm_rejected_until = time.time() - 1
        self.assertTrue(self.brain.llm_available)

    def _expire_latch(self):
        """Simulate the window elapsing, so the next rejection is a failed
        PROBE rather than a straggler."""
        self.brain._llm_rejected_until = time.time() - 1

    def test_ladder_escalates_per_failed_probe(self):
        seen = []
        for _ in range(len(LLM_REJECT_BACKOFF_MINUTES) + 2):
            before = time.time()
            self._reject()
            seen.append(round((self.brain._llm_rejected_until - before) / 60.0))
            self._expire_latch()
        expected = list(LLM_REJECT_BACKOFF_MINUTES) + [
            LLM_REJECT_BACKOFF_MINUTES[-1]] * 2      # ceiling holds
        self.assertEqual(seen, expected)

    def test_concurrent_stragglers_do_not_advance_the_ladder(self):
        # Four concurrent encodes fail in the same second when a key dies.
        # Only the first is a probe; the rest were already in flight. Without
        # this, one blip would jump straight to the hour-long ceiling.
        self._reject()
        first_until = self.brain._llm_rejected_until
        for _ in range(3):
            self._reject()
        self.assertEqual(self.brain._llm_reject_strikes, 1)
        self.assertEqual(self.brain._llm_rejected_until, first_until)

    def test_strikes_age_out_after_a_quiet_stretch(self):
        self._reject()
        self._expire_latch()
        self._reject()
        self._expire_latch()
        self.brain._llm_rejected_at = time.time() - LLM_REJECT_STRIKE_RESET_SECONDS - 1
        before = time.time()
        self._reject()
        self.assertEqual(
            round((self.brain._llm_rejected_until - before) / 60.0),
            LLM_REJECT_BACKOFF_MINUTES[0])

    def test_replacing_the_key_lifts_the_latch_immediately(self):
        # Pasting a fresh key into /setup must work now, not in an hour — the
        # refusal was a verdict on the OLD credential.
        self._reject()
        self.assertFalse(self.brain.llm_available)
        os.environ['ANTHROPIC_API_KEY'] = 'sk-a-different-key'
        self.assertTrue(self.brain.llm_available)
        self.assertEqual(self.brain._llm_reject_strikes, 0)   # fresh ladder

    def test_re_enabling_the_same_key_still_waits_for_the_clock(self):
        # The case a value-change trigger alone would miss: the operator
        # disables and re-enables the same key, so nothing about it changes.
        self._reject()
        self.assertFalse(self.brain.llm_available)
        self.assertFalse(self.brain.llm_available)   # still latched, not lifted

    def test_quota_refusal_naming_a_date_still_reprobes_on_the_ladder(self):
        # 2026-09-25: the limit was raised on the SAME key, but the latch had
        # parked until the date the provider named, six days out — the brain
        # stayed paused until a restart. A refused probe costs nothing, so a
        # quota refusal rides the ladder and notices the raised limit.
        before = time.time()
        self.brain._log_error('surface_agentic_api',
                              _FakeAPIError(_QUOTA_400, 400), 'ctx')
        self.assertFalse(self.brain.llm_available)
        self.assertEqual(
            round((self.brain._llm_rejected_until - before) / 60.0),
            LLM_REJECT_BACKOFF_MINUTES[0])

    def test_llm_rejection_reports_the_live_refusal_only(self):
        self.assertIsNone(self.brain.llm_rejection())
        self._reject(kind=LLM_QUOTA_EXHAUSTED)
        refusal = self.brain.llm_rejection()
        self.assertEqual(refusal['kind'], LLM_QUOTA_EXHAUSTED)
        self.assertEqual(refusal['detail'], 'API key is invalid')
        self.assertAlmostEqual(refusal['minutes_left'],
                               LLM_REJECT_BACKOFF_MINUTES[0], delta=0.1)
        self._expire_latch()
        self.assertIsNone(self.brain.llm_rejection())

    def test_a_removed_key_is_not_reported_as_refused(self):
        # A refusal is a verdict on one credential. With no key at all the
        # operator must be told to set one, not to check a key that is gone.
        self._reject()
        os.environ['ANTHROPIC_API_KEY'] = ''
        self.assertFalse(self.brain.llm_available)
        self.assertIsNone(self.brain.llm_rejection())
        self.brain.note_llm_unavailable('S1 Scribe')
        self.assertTrue(self.brain._llm_unavailable_noted)

    def test_log_error_latches_on_a_refusal_and_ignores_other_errors(self):
        # The wiring that matters: no caller threads a classification through,
        # so every failing LLM path latches by virtue of logging its failure.
        self.brain._log_error('s1e_run_failed', ValueError('unrelated'), 'ctx')
        self.assertTrue(self.brain.llm_available)

        self.brain._log_error(
            's1e_run_failed',
            _FakeAPIError('Error code: 401 - API key is invalid', 401), 'ctx')
        self.assertFalse(self.brain.llm_available)

    def test_a_condition_message_never_latches(self):
        # Only an exception can be a provider refusal: a condition's message
        # text is not one, whatever status code it happens to quote.
        self.brain._log_error(
            's1e_truncation', 'Error code: 401 - API key is invalid', 'ctx')
        self.assertTrue(self.brain.llm_available)

    def test_keyless_notice_suppressed_while_latched(self):
        # "No API key resolved" would send the operator to fix the one thing
        # that isn't broken.
        self._reject()
        self.brain.note_llm_unavailable('S1 Scribe')
        self.assertFalse(getattr(self.brain, '_llm_unavailable_noted', False))


if __name__ == '__main__':
    unittest.main()
