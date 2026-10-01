"""The last-mile leg (channels/delivery.py) — moments × sources.

Component layer for the leg both hooks ride: the serves() eligibility
predicate (the boot ruling as code), per-source failure isolation, the
per-source s0 K trace at both moments, the trace-after-keep ordering (a
trace failure costs the trace, never an already-ledgered delivery), and the
composite warn. Hook wiring is covered one layer up (test_self_delivery.py
drives the real Stop handler).
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.channels import delivery
from servers.channels.thalamus import thalamus
from servers.channels.self_channel import signal, self_contract


class TestServes(unittest.TestCase):

    def test_serves_is_the_boot_ruling(self):
        # Forcing moment: everyone speaks. Passive moment: only a source
        # that survives a miss (ruling id:bb0513ae as a predicate).
        self.assertTrue(delivery.serves(delivery.COURIER, delivery.STOP))
        self.assertTrue(delivery.serves(delivery.THALAMUS, delivery.STOP))
        self.assertTrue(delivery.serves(delivery.THALAMUS, delivery.BOOT))
        self.assertFalse(delivery.serves(delivery.COURIER, delivery.BOOT))


class DeliveryBase(BrainTestBase):
    needs_embedder = False

    def _ctx(self, session_id):
        return self.brain.get_or_create_session(session_id)

    def _traces(self, session_id, ref_type):
        events = self.brain.query_traces(session_id=session_id).get('events', [])
        return [e for e in events if e.get('ref_type') == ref_type]


class TestDeliver(DeliveryBase):

    def test_boot_delivers_thalamus_and_writes_the_trace(self):
        # A boot delivery must be joinable to the S0 stream — the boot leg
        # writes the same thalamus_delivery K event the Stop leg does.
        thalamus.file(self.brain, 'test', 'a due notice', for_whom='all')
        block, shown = delivery.deliver(self.brain, self._ctx('S1'), delivery.BOOT)
        self.assertIn('a due notice', block)
        self.assertEqual(shown, ('thalamus_delivery',))
        events = self._traces('S1', 'thalamus_delivery')
        self.assertTrue(events)
        # The trace carries the rendered block — the content a dial-on
        # correspondent surfaces as the turn's incoming side — and the moment
        # as its ref_id (indexed: the boot-prelude vs Stop-turn discriminator).
        meta = events[0].get('metadata') or {}
        self.assertIn('a due notice', meta.get('content', ''))
        self.assertEqual(events[0].get('ref_id'), 'boot')

    def test_boot_does_not_drain_the_courier(self):
        # The courier declines the passive moment — the tap must SURVIVE for
        # the forcing Stop leg (no consume-once against a channel that can
        # miss). Ruling id:bb0513ae.
        signal.send(self.brain, from_session='other',
                    address=self_contract.address_for_stream('S2'),
                    body='a tap for stop')
        block, _ = delivery.deliver(self.brain, self._ctx('S2'), delivery.BOOT)
        self.assertNotIn('a tap for stop', block)
        self.assertEqual(len(signal.peek_inbox(self.brain, 'S2')), 1)

    def test_stop_composes_both_sources_in_order(self):
        signal.send(self.brain, from_session='other',
                    address=self_contract.address_for_stream('S3'),
                    body='stream speech first')
        thalamus.file(self.brain, 'test', 'brain item second', for_whom='all')
        block, shown = delivery.deliver(self.brain, self._ctx('S3'), delivery.STOP)
        self.assertIn('stream speech first', block)
        self.assertIn('brain item second', block)
        self.assertLess(block.index('stream speech first'),
                        block.index('brain item second'))
        self.assertEqual(shown, ('self_message', 'thalamus_delivery'))
        self.assertTrue(self._traces('S3', 'self_message'))
        self.assertTrue(self._traces('S3', 'thalamus_delivery'))

    def test_nothing_due_is_empty(self):
        self.assertEqual(
            delivery.deliver(self.brain, self._ctx('S4'), delivery.STOP), ('', ()))

    def test_no_trace_when_nothing_shows(self):
        delivery.deliver(self.brain, self._ctx('S5'), delivery.BOOT)
        self.assertFalse(self._traces('S5', 'thalamus_delivery'))

    def test_source_failure_is_isolated(self):
        # One source raising must not take the moment down — the other still
        # delivers, and the failure lands in the errors table.
        thalamus.file(self.brain, 'test', 'survives the crash', for_whom='all')
        broken = delivery.COURIER._replace(
            render=lambda *a: (_ for _ in ()).throw(RuntimeError('boom')))
        orig = delivery.SOURCES
        delivery.SOURCES = (broken, delivery.THALAMUS)
        try:
            block, _ = delivery.deliver(self.brain, self._ctx('S6'), delivery.STOP)
        finally:
            delivery.SOURCES = orig
        self.assertIn('survives the crash', block)

    def test_trace_failure_keeps_the_block_but_drops_the_stamp(self):
        # Sources ledger/consume inside render — a trace-write failure after
        # that must cost the trace, never the block: a dropped block would
        # leave items marked delivered that nobody ever saw. And the failed
        # source must NOT ride the returned traced set: a stamped reaction
        # whose incoming row doesn't exist is a turn with no incoming side.
        thalamus.file(self.brain, 'test', 'ledgered then shown', for_whom='all')
        orig = delivery._s0_trace
        delivery._s0_trace = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('trace substrate down'))
        try:
            block, traced = delivery.deliver(
                self.brain, self._ctx('S8'), delivery.BOOT)
        finally:
            delivery._s0_trace = orig
        self.assertIn('ledgered then shown', block)
        self.assertEqual(traced, ())
        self.assertFalse(self._traces('S8', 'thalamus_delivery'))

    def test_composite_over_budget_warns_and_still_delivers(self):
        big = delivery.THALAMUS._replace(
            render=lambda *a: ('x' * (delivery.COMPOSITE_WARN + 1), 1))
        orig = delivery.SOURCES
        delivery.SOURCES = (big,)
        try:
            block, _ = delivery.deliver(self.brain, self._ctx('S7'), delivery.STOP)
        finally:
            delivery.SOURCES = orig
        self.assertGreater(len(block), delivery.COMPOSITE_WARN)  # warn, no cap
        rows = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log "
            "WHERE source = 'delivery_composite_over_budget'").fetchone()
        self.assertTrue(rows[0] >= 1)


if __name__ == '__main__':
    unittest.main()
