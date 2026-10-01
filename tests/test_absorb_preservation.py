"""Preservation gate — one absorb on a richly-populated fixture must be lossless
across EVERY transfer dimension at once.

test_absorb.py unit-tests each dimension in isolation; this is the integration
guarantee. The auditor (eval/absorb_preservation_probe.py) is the same tool used
to verify a LIVE merge before trusting the consolidation prompt in production
(S2-ABSORB-OP-DESIGN.md roadmap step 3).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from tests.eval_optional import require_eval  # noqa: E402
require_eval()  # D-8: eval/ is absent from the public tree

from eval.absorb_preservation_probe import build_rich_fixture, snapshot_pre, audit


class TestAbsorbPreservationGate(BrainTestBase):
    needs_embedder = False

    def test_rich_fixture_absorb_is_lossless(self):
        survivor, absorbed = build_rich_fixture(self.brain)
        pre = snapshot_pre(self.brain, survivor, absorbed)
        r = self.brain.absorb(survivor, absorbed,
                              content='merged synthesis', reason='gate')
        self.assertTrue(r['ok'], r)
        report = audit(self.brain, survivor, absorbed, pre, overrides=['content'])
        # Per-dimension assert so a failure names the lost dimension.
        for name, d in report['dimensions'].items():
            if d.get('diagnostic'):
                continue
            self.assertTrue(d['ok'], '%s lost information: %s' % (name, d['detail']))
        self.assertTrue(report['lossless'], report)

    def test_drop_fields_and_prune_edges_are_honored(self):
        # Caller-named drops are intentional loss, not silent — the auditor
        # excludes them, so the merge still scores lossless.
        survivor, absorbed = build_rich_fixture(self.brain)
        pre = snapshot_pre(self.brain, survivor, absorbed)
        r = self.brain.absorb(survivor, absorbed,
                              drop_fields=['emergent_key'])
        self.assertTrue(r['ok'], r)
        report = audit(self.brain, survivor, absorbed, pre,
                       drop_fields=['emergent_key'])
        self.assertTrue(report['lossless'], report)
        # The dropped field really did NOT transfer.
        kv = self.brain._meta_kv.get_all_bulk([survivor])[survivor]
        self.assertNotIn('emergent_key', kv)


if __name__ == '__main__':
    unittest.main()
