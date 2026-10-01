"""IntegrationUnit builds ONE Anthropic client per run, not one per call.

`_call_llm` used to evaluate `make_client()` inline on every invocation.
Aspect calls it once per run so it never noticed, but healer calls it once
per BATCH — so a run with N batches built N clients, each with its own
httpx pool and a cold TLS handshake. That is the throwaway-client shape
already removed from `surface.py`; the loop encoders avoid it by hoisting
`client = make_client()` above their batch loop.

`_llm_client()` caches on the instance, which is per-run because units are
constructed inside their unit's `run()` (healer.py, aspect_integration.py).
These tests pin that: one construction per unit, the same object handed to
every call, and no cross-unit sharing that would outlive a run.

There is no daemon.log signal for any of this — `_call_llm` goes through
`run_llm_once`, which is single-shot and emits no per-round lines. A test
is the only way this claim gets checked at all.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.scales.s2 import base as s2_base
from servers.scales.s2.base import IntegrationUnit


class _FakeClient:
    """Distinguishable by identity — that's the whole point of the test."""

    def __init__(self, serial):
        self.serial = serial


class _StubBrain:
    def __init__(self):
        self.errors = []

    def get_interaction_prompt(self, name):
        return 'system prompt for %s' % name

    def get_interaction_config(self, name):
        return {'model': 'test-model', 'max_tokens': 128}

    def get_interaction_effective(self, name):
        # Composed from the two accessors so a test overriding one of them
        # still reaches _call_llm.
        return {'template': self.get_interaction_prompt(name),
                'config': self.get_interaction_config(name),
                'stamp': {'fingerprint': 'stub', 'source': 'default',
                          'version': 0, 'id': None}}

    def _log_error(self, name, exc, msg):
        self.errors.append((name, str(exc), msg))


def _unit():
    """A bare IntegrationUnit — bypass __init__, which wants a real brain."""
    u = IntegrationUnit.__new__(IntegrationUnit)
    u.brain = _StubBrain()
    u.dispatch = None
    return u


class ClientLifetimeTest(unittest.TestCase):
    def setUp(self):
        self._real_make_client = s2_base.make_client
        self.constructions = 0

        def counting_make_client():
            self.constructions += 1
            return _FakeClient(self.constructions)

        s2_base.make_client = counting_make_client

    def tearDown(self):
        s2_base.make_client = self._real_make_client

    def test_one_construction_per_unit_however_many_calls(self):
        """Healer's batch loop must not pay a new TLS pool per batch."""
        unit = _unit()
        clients = [unit._llm_client() for _ in range(5)]

        self.assertEqual(
            self.constructions, 1,
            'a unit must build its client once per run — %d constructions '
            'means every batch is paying a cold TLS handshake again'
            % self.constructions)
        self.assertEqual(
            len({id(c) for c in clients}), 1,
            'every call must receive the same client instance')

    def test_units_do_not_share_a_client_across_runs(self):
        """Per-run lifetime, not a process-wide singleton.

        The cache is deliberately instance-scoped: units are constructed per
        run, so a run is the lifetime. A shared client would outlive a key
        rotation, which is exactly what brain._ensure_anthropic_client
        key-stamps against for the daemon's own client.
        """
        first, second = _unit()._llm_client(), _unit()._llm_client()

        self.assertEqual(self.constructions, 2)
        self.assertIsNot(
            first, second,
            'separate unit instances must not share one client — that would '
            'silently extend a client past the run it was built for')

    def test_failed_construction_is_not_cached(self):
        """A construction failure must not poison the rest of the run."""
        def exploding_make_client():
            self.constructions += 1
            raise RuntimeError('SDK unavailable')

        s2_base.make_client = exploding_make_client
        unit = _unit()

        with self.assertRaises(RuntimeError):
            unit._llm_client()

        # Recover: the next call must retry construction, not return None
        # or re-raise from a poisoned cache.
        s2_base.make_client = lambda: _FakeClient('recovered')
        self.assertEqual(unit._llm_client().serial, 'recovered')


class CallLlmWiringTest(unittest.TestCase):
    """The hoist only matters if _call_llm actually uses it."""

    def setUp(self):
        self._real_make_client = s2_base.make_client
        self._real_run_llm_once = s2_base.run_llm_once
        self.constructions = 0
        self.clients_seen = []
        self.calls = []

        def counting_make_client():
            self.constructions += 1
            return _FakeClient(self.constructions)

        def capturing_run_llm_once(client, model, max_tokens, system, user,
                                   effort=None):
            self.clients_seen.append(client)
            self.calls.append(
                {'model': model, 'max_tokens': max_tokens, 'system': system})
            return '{"ok": true}', {'elapsed_ms': 1}

        s2_base.make_client = counting_make_client
        s2_base.run_llm_once = capturing_run_llm_once

    def tearDown(self):
        s2_base.make_client = self._real_make_client
        s2_base.run_llm_once = self._real_run_llm_once

    def test_batch_loop_hands_the_same_client_to_every_call(self):
        """Simulates healer: several _call_llm calls on one unit instance."""
        unit = _unit()
        unit.NAME = 'healer'

        for batch in range(3):
            result, _tel = unit._call_llm('s2_healer', 'batch %d' % batch)
            self.assertEqual(result, {'ok': True})

        self.assertEqual(
            self.constructions, 1,
            'three batches built %d clients — _call_llm is still constructing '
            'per call' % self.constructions)
        self.assertEqual(len(self.clients_seen), 3)
        self.assertEqual(
            len({id(c) for c in self.clients_seen}), 1,
            'every batch must be handed the same client instance')

    def test_resolved_config_and_prompt_reach_the_call(self):
        """Sentinel: the model/max_tokens the brain's resolver hands back —
        and its prompt — must be exactly what run_llm_once receives. A
        caller-side fallback would eat an override silently (the a6dfcfe3
        failure shape); _call_llm subscripts the resolved config instead."""
        unit = _unit()
        unit.NAME = 'healer'
        unit.brain.get_interaction_config = lambda name: {
            'model': 'sentinel-s2-model', 'max_tokens': 4242}

        unit._call_llm('s2_healer', 'payload')

        self.assertEqual(self.calls[0]['model'], 'sentinel-s2-model')
        self.assertEqual(self.calls[0]['max_tokens'], 4242)
        self.assertEqual(self.calls[0]['system'],
                         'system prompt for s2_healer')


if __name__ == '__main__':
    unittest.main()
