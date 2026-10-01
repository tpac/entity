"""Tests for retry_on_transient_api_error in servers/scales/runner.py.

Ensures the S2 retry wrapper:
- Retries on APITimeoutError, APIConnectionError, InternalServerError
- Does NOT retry on BadRequestError, AuthenticationError, RateLimitError
- Respects `attempts` ceiling
- Returns the function's result on success (no wrapping)
- Exponential backoff is applied between attempts

The wrapper is the fix for 'encode batch 1 FAILED: The read operation
timed out' — a mid-stream stall the SDK's max_retries can't cover.
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class _Counter:
    """Helper: callable that tracks invocations and raises N times first."""
    def __init__(self, raise_n, exc, return_value='ok'):
        self.raise_n = raise_n
        self.exc = exc
        self.return_value = return_value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.raise_n:
            raise self.exc
        return self.return_value


def _make_api_error(cls):
    """Construct an anthropic.APIError subclass safely — SDK requires args."""
    import anthropic
    if cls is anthropic.APITimeoutError:
        # APITimeoutError requires a request arg
        class _FakeReq:
            pass
        return cls(request=_FakeReq())
    if cls is anthropic.APIConnectionError:
        class _FakeReq:
            pass
        return cls(message='connection failed', request=_FakeReq())
    if cls is anthropic.InternalServerError:
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.status_code = 500
        resp.headers = {}
        return cls(message='internal error', response=resp, body=None)
    if cls is anthropic.BadRequestError:
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.status_code = 400
        resp.headers = {}
        return cls(message='bad request', response=resp, body=None)
    if cls is anthropic.RateLimitError:
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {}
        return cls(message='rate limit', response=resp, body=None)
    raise ValueError(f'unsupported: {cls}')


class TestRetryTransient(unittest.TestCase):

    def setUp(self):
        # Short backoff keeps tests under 1s
        from servers.scales import runner
        self.runner = runner
        self._orig_sleep = time.sleep
        # Monkey-patch sleep in the runner module to avoid real delay
        self._sleep_patcher = patch.object(runner.time, 'sleep')
        self.mock_sleep = self._sleep_patcher.start()

    def tearDown(self):
        self._sleep_patcher.stop()

    # ── RETRY path ──

    def test_success_first_try_no_retry(self):
        counter = _Counter(raise_n=0, exc=None, return_value='payload')
        out = self.runner.retry_on_transient_api_error(counter)
        self.assertEqual(out, 'payload')
        self.assertEqual(counter.calls, 1)
        self.mock_sleep.assert_not_called()

    def test_retries_on_api_timeout_error(self):
        import anthropic
        err = _make_api_error(anthropic.APITimeoutError)
        counter = _Counter(raise_n=1, exc=err)
        out = self.runner.retry_on_transient_api_error(
            counter, attempts=2, base_backoff_s=0.1)
        self.assertEqual(out, 'ok')
        self.assertEqual(counter.calls, 2)
        self.mock_sleep.assert_called_once()

    def test_retries_on_api_connection_error(self):
        import anthropic
        err = _make_api_error(anthropic.APIConnectionError)
        counter = _Counter(raise_n=1, exc=err)
        out = self.runner.retry_on_transient_api_error(
            counter, attempts=2, base_backoff_s=0.1)
        self.assertEqual(out, 'ok')
        self.assertEqual(counter.calls, 2)

    def test_retries_on_internal_server_error(self):
        import anthropic
        err = _make_api_error(anthropic.InternalServerError)
        counter = _Counter(raise_n=1, exc=err)
        out = self.runner.retry_on_transient_api_error(
            counter, attempts=2, base_backoff_s=0.1)
        self.assertEqual(out, 'ok')
        self.assertEqual(counter.calls, 2)

    def test_retries_on_httpx_timeout(self):
        import httpx
        err = httpx.ReadTimeout('read timed out')
        counter = _Counter(raise_n=1, exc=err)
        out = self.runner.retry_on_transient_api_error(
            counter, attempts=2, base_backoff_s=0.1)
        self.assertEqual(out, 'ok')
        self.assertEqual(counter.calls, 2)

    # ── NO RETRY path ──

    def test_does_not_retry_bad_request(self):
        import anthropic
        err = _make_api_error(anthropic.BadRequestError)
        counter = _Counter(raise_n=99, exc=err)
        with self.assertRaises(anthropic.BadRequestError):
            self.runner.retry_on_transient_api_error(
                counter, attempts=3, base_backoff_s=0.1)
        # Only one call — no retries on non-transient
        self.assertEqual(counter.calls, 1)
        self.mock_sleep.assert_not_called()

    def test_does_not_retry_rate_limit(self):
        # Rate limit has its own Retry-After handling at SDK layer; our
        # wrapper should not retry it to avoid stacked backoffs
        import anthropic
        err = _make_api_error(anthropic.RateLimitError)
        counter = _Counter(raise_n=99, exc=err)
        with self.assertRaises(anthropic.RateLimitError):
            self.runner.retry_on_transient_api_error(
                counter, attempts=3, base_backoff_s=0.1)
        self.assertEqual(counter.calls, 1)

    def test_non_api_exception_raises_immediately(self):
        err = ValueError('something else')
        counter = _Counter(raise_n=99, exc=err)
        with self.assertRaises(ValueError):
            self.runner.retry_on_transient_api_error(
                counter, attempts=3, base_backoff_s=0.1)
        self.assertEqual(counter.calls, 1)

    # ── Exhaustion ──

    def test_exhaustion_raises_last_transient(self):
        import anthropic
        err = _make_api_error(anthropic.APITimeoutError)
        counter = _Counter(raise_n=99, exc=err)
        with self.assertRaises(anthropic.APITimeoutError):
            self.runner.retry_on_transient_api_error(
                counter, attempts=2, base_backoff_s=0.1)
        self.assertEqual(counter.calls, 2)  # attempts=2 => 1 retry
        # One sleep call (between attempt 1 and 2)
        self.assertEqual(self.mock_sleep.call_count, 1)

    def test_exponential_backoff_doubles(self):
        import anthropic
        err = _make_api_error(anthropic.APITimeoutError)
        counter = _Counter(raise_n=99, exc=err)
        with self.assertRaises(anthropic.APITimeoutError):
            self.runner.retry_on_transient_api_error(
                counter, attempts=3, base_backoff_s=2.0)
        # attempts=3 → 2 sleeps: 2s, 4s
        self.assertEqual(self.mock_sleep.call_count, 2)
        calls = [c.args[0] for c in self.mock_sleep.call_args_list]
        self.assertEqual(calls, [2.0, 4.0])

    # ── Logging ──

    def test_log_fn_called_on_retry(self):
        import anthropic
        err = _make_api_error(anthropic.APITimeoutError)
        counter = _Counter(raise_n=1, exc=err)
        logs = []
        self.runner.retry_on_transient_api_error(
            counter, attempts=2, base_backoff_s=0.1,
            log_fn=lambda msg: logs.append(msg))
        self.assertEqual(len(logs), 1)
        self.assertIn('retrying', logs[0])
        self.assertIn('APITimeoutError', logs[0])

    def test_log_fn_not_called_on_success(self):
        counter = _Counter(raise_n=0, exc=None)
        logs = []
        self.runner.retry_on_transient_api_error(
            counter, log_fn=lambda msg: logs.append(msg))
        self.assertEqual(logs, [])


if __name__ == '__main__':
    unittest.main()
