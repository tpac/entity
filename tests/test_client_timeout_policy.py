"""Transport policy of the two shared Anthropic clients (plan Step 5).

`ANTHROPIC_CLIENT_TIMEOUT`'s 600s is a READ budget; passed as a bare float it
also governed connect, so a dead network was granted the slow-generation
allowance — a flapping network held an encode permit ~53 min that way
(e2dc24d3, 2026-08-18: 600s read × 3 SDK attempts, with the cooperative run
deadline unable to preempt a blocked read). These tests pin the granular
shape at both construction sites — connect must stay far below read — and
the encoder lane's max_retries=1 (its callers own retry policy: the Scribe
cooldown above S1E, retry_on_transient_api_error above the S2 loops).

Construction is lazy (no network), so building real clients here is free.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.brain_constants import (ANTHROPIC_CLIENT_TIMEOUT,
                                     ANTHROPIC_CONNECT_TIMEOUT)
from servers.scales.runner import make_client


class TimeoutPolicyTest(unittest.TestCase):
    def _assert_granular(self, client):
        t = client.timeout
        self.assertEqual(t.connect, ANTHROPIC_CONNECT_TIMEOUT)
        self.assertEqual(t.read, ANTHROPIC_CLIENT_TIMEOUT)
        # The point of the split — connect must never inherit the read budget.
        self.assertLess(t.connect, t.read / 10)

    def test_encoder_lane_client(self):
        client = make_client()
        self._assert_granular(client)
        self.assertEqual(client.max_retries, 1)

    def test_daemon_shared_client(self):
        from servers.brain import Brain
        # Keyless path: the method returns the built client without caching,
        # so a bare stub suffices and no attribute writes are expected.
        with patch('servers.scales.dispatch.resolve_api_key', return_value=''):
            client = Brain._ensure_anthropic_client(SimpleNamespace())
        self._assert_granular(client)


if __name__ == '__main__':
    unittest.main()
