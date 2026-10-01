"""Thalamus Phase 1 — the door, the pull, the verbs, the sweep.

Covers the core contracts:
- file(): routing (queue vs live delegation), defaults by kind, dedup upsert,
  never refused for volume, when/for_whom resolution errors.
- pull(): the delivery predicate (due/window/audience/target/ledger),
  asks-at-boot-only, push-once-per-session idempotence, once vs all.
- resolve(): answer / defer (re-arm) / dismiss, exactly-one-action guard.
- withdraw(): source-scoped producer retraction.
- expire_due(): window sweep; an unanswered ask dies LOUDLY (dead-letter).
- render: pre-filled resolve call, loud truncation.
- dispatch handlers return the {"ok", "result"} envelope (the dispatch_self
  lesson — a raw payload surfaces as "Unknown daemon error").
"""
import unittest

from tests.brain_test_base import BrainTestBase
from servers.channels.thalamus import thalamus, thalamus_contract as tc
from servers.dispatch_thalamus import (
    _handle_remind, _handle_thalamus_list, _handle_thalamus_resolve)
from servers.clock import iso_now, iso_cutoff

S1 = 'aaaaaaaa-1111-2222-3333-444444444444'
S2 = 'bbbbbbbb-1111-2222-3333-444444444444'


class ThalamusBase(BrainTestBase):
    needs_embedder = False

    def _file(self, body='check the thing', **kw):
        kw.setdefault('source', 'test')
        return thalamus.file(self.brain, kw.pop('source'), body, **kw)

    def _row(self, item_id):
        return self.brain.logs_conn.execute(
            'SELECT state, audience, needs_answer, deliver_at, expires_at,'
            ' target_session FROM thalamus_items WHERE id = ?',
            (item_id,)).fetchone()

    def _age_out(self, item_id):
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET expires_at = ? WHERE id = ?',
                (iso_cutoff(hours=1), item_id))
            self.brain.logs_conn_w.commit()

    def _age_deliveries(self):
        """Push every ledger stamp past the saturation grace — the sweep
        waits for the session still holding the delivery that saturated."""
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_deliveries SET delivered_at = ?',
                (iso_cutoff(hours=tc.SATURATION_GRACE_HOURS + 1),))
            self.brain.logs_conn_w.commit()

    def _saturation_warnings(self):
        """The loud rows for an item the CAP closed that asked nothing."""
        return [r[0] for r in self.brain.logs_conn.execute(
            "SELECT metadata FROM debug_log"
            " WHERE source='thalamus_saturated' ORDER BY rowid DESC")]

    def _dead_letters(self):
        """The loud unanswered-ask rows, newest first. `_log_error` packs the
        message into debug_log.metadata."""
        return [r[0] for r in self.brain.logs_conn.execute(
            "SELECT metadata FROM debug_log"
            " WHERE source='thalamus_ask_expired' ORDER BY rowid DESC")]


class TestFile(ThalamusBase):

    def test_notice_defaults(self):
        r = self._file()
        self.assertTrue(r['ok'])
        self.assertEqual(r['route'], 'queue')
        state, audience, needs_answer, deliver_at, expires_at, _ = self._row(r['id'])
        self.assertEqual(state, tc.STATE_OPEN)
        self.assertEqual(audience, tc.AUDIENCE_FIRST)
        self.assertEqual(needs_answer, 0)
        self.assertIsNone(deliver_at)
        self.assertGreater(expires_at, iso_now())

    def test_ask_defaults_to_all_audience(self):
        r = self._file('should X be configurable?', needs_answer=True)
        _, audience, needs_answer, _, expires_at, _ = self._row(r['id'])
        self.assertEqual(audience, tc.AUDIENCE_EVERY)
        self.assertEqual(needs_answer, 1)
        # Ask window is the long one.
        self.assertGreater(expires_at, iso_cutoff(days=-(tc.ASK_EXPIRES_DAYS - 1)))

    def test_when_shorthand_sets_future_deliver_at(self):
        r = self._file(when='2h')
        _, _, _, deliver_at, _, _ = self._row(r['id'])
        self.assertIsNotNone(deliver_at)
        self.assertGreater(deliver_at, iso_now())

    def test_bad_when_rejects_loudly(self):
        r = self._file(when='next full moon')
        self.assertFalse(r['ok'])
        self.assertIn('when', r['error'])

    def test_bad_for_whom_rejects_loudly(self):
        r = self._file(for_whom='everyone-ish')
        self.assertFalse(r['ok'])
        self.assertIn('for_whom', r['error'])

    def test_directed_requires_full_uuid_and_stamps_target(self):
        r = self._file(for_whom=S1)
        _, audience, _, _, _, target = self._row(r['id'])
        self.assertEqual(target, S1)
        self.assertEqual(audience, tc.AUDIENCE_FIRST)
        # An 8-char short is a display convention, not a key.
        r2 = self._file(for_whom=S1[:8])
        self.assertFalse(r2['ok'])

    def test_empty_body_rejects(self):
        self.assertFalse(self._file('  ')['ok'])

    def test_directed_ask_files_and_targets(self):
        # A nameable session has already booted, so a directed ask is only
        # deliverable because its moment is Stop (TestPull pins the moment).
        r = self._file('judge this?', needs_answer=True, for_whom=S1)
        self.assertTrue(r['ok'])
        _, audience, needs_answer, _, _, target = self._row(r['id'])
        self.assertEqual((audience, needs_answer, target),
                         (tc.AUDIENCE_FIRST, 1, S1))

    def test_expires_before_when_rejects(self):
        r = self._file(when='3d', expires='1d')
        self.assertFalse(r['ok'])
        self.assertIn('before it ever becomes due', r['error'])

    def test_expires_now_or_past_rejects_named(self):
        # 'now' resolves to no instant, a past ISO or '0m' to one already
        # gone: each is refused at the door, not left for the insert or the
        # sweep to fail on.
        for expires in ('now', '2020-01-01T00:00:00+00:00', '0m'):
            r = self._file(expires=expires)
            self.assertFalse(r['ok'], expires)
            self.assertIn('before it ever becomes due', r['error'])

    def test_dedup_identity_includes_the_target(self):
        """(source, dedup_key, target_session): the same key filed for two
        sessions is two items; withdraw by key closes only the named
        target's item. One producer string serves every session."""
        a = self._file('for A', dedup_key='k', for_whom=S1)
        b = self._file('for B', dedup_key='k', for_whom=S2)
        c = self._file('broadcast', dedup_key='k')
        self.assertEqual(len({a['id'], b['id'], c['id']}), 3)
        self.assertFalse(b.get('updated'))
        r = thalamus.withdraw(self.brain, 'test', dedup_key='k',
                              target_session=S2)
        self.assertTrue(r['ok'])
        self.assertEqual([self._row(x['id'])[0] for x in (a, b, c)],
                         [tc.STATE_OPEN, tc.STATE_WITHDRAWN, tc.STATE_OPEN])
        # Without a target, the key names the broadcast item.
        self.assertTrue(thalamus.withdraw(self.brain, 'test',
                                          dedup_key='k')['ok'])
        self.assertEqual(self._row(c['id'])[0], tc.STATE_WITHDRAWN)

    def test_dedup_key_updates_not_duplicates(self):
        r1 = self._file('v1 of the concern', dedup_key='concern-x')
        r2 = self._file('v2 of the concern', dedup_key='concern-x')
        self.assertTrue(r2.get('updated'))
        self.assertEqual(r1['id'], r2['id'])
        body = self.brain.logs_conn.execute(
            'SELECT body FROM thalamus_items WHERE id = ?',
            (r1['id'],)).fetchone()[0]
        self.assertEqual(body, 'v2 of the concern')
        n = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM thalamus_items WHERE dedup_key='concern-x'"
        ).fetchone()[0]
        self.assertEqual(n, 1)

    def test_dedup_refile_escalates_a_note_into_an_ask(self):
        """The change-gate covers every producer-controlled delivery
        attribute, so escalating a standing note into an ask actually lands.
        Comparing only (body, refs, deliver_at) left needs_answer=0 — still a
        note, still Stop-delivered, never dead-lettering — while the door had
        already granted it the ask's longer window."""
        r1 = self._file('the recurring thing', dedup_key='esc')
        self.assertEqual(self._row(r1['id'])[2], 0)
        r2 = self._file('the recurring thing', dedup_key='esc',
                        needs_answer=True)
        self.assertEqual(r2['id'], r1['id'])
        self.assertTrue(r2['rearmed'])
        _, audience, needs_answer, _, _, _ = self._row(r2['id'])
        self.assertEqual(needs_answer, 1)
        # ...including the audience the new kind implies — an ask left on the
        # notice audience would deliver to exactly one session, ever.
        self.assertEqual(audience, tc.AUDIENCE_EVERY)

    def test_dedup_refile_to_another_reader_is_another_item(self):
        """The reader is part of an item's identity — (source, dedup_key,
        target_session). Re-filing a key for a different
        session is a new item, never a retarget: one producer string serves
        every session's runs, so a retarget would let one session silently
        take another's item. Moving an item = withdraw, then file."""
        r1 = self._file('heads up', dedup_key='aim')
        self.assertEqual(self._row(r1['id'])[1], tc.AUDIENCE_FIRST)
        r2 = self._file('heads up', dedup_key='aim', for_whom=S1)
        self.assertNotEqual(r1['id'], r2['id'])
        self.assertFalse(r2.get('updated'))
        self.assertEqual(self._row(r1['id'])[5], '')
        self.assertEqual(self._row(r2['id'])[5], S1)
        # Within one reader the key still updates, not duplicates.
        r3 = self._file('heads up, louder', dedup_key='aim', for_whom=S1)
        self.assertEqual((r3['id'], r3['updated'], r3['rearmed']),
                         (r2['id'], True, True))

    def test_dated_item_keeps_full_window(self):
        """Expiry anchors at deliver_at, not now — an ask due in 3 weeks must
        not expire after 2 (it would be undeliverable and fire a FALSE loud
        dead-letter)."""
        r = self._file('late ask', needs_answer=True, when='3w')
        _, _, _, deliver_at, expires_at, _ = self._row(r['id'])
        self.assertGreater(expires_at, deliver_at)
        r2 = self._file('late notice', for_whom='all', when='1w')
        _, _, _, deliver_at2, expires_at2, _ = self._row(r2['id'])
        self.assertGreater(expires_at2, deliver_at2)

    def test_dedup_with_second_matching_row_still_files(self):
        """Regression for the write-conn snapshot hazard: a second open row
        sharing (source, dedup_key) must not break file() — the dedup lookup
        is LIMIT 1 and its cursor unbound."""
        r1 = self._file('first', dedup_key='dup')
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'INSERT INTO thalamus_items (id, source, body, refs, audience,'
                ' target_session, needs_answer, dedup_key, deliver_at,'
                ' expires_at, state, answer, created_at)'
                " VALUES ('th_dupdup', 'test', 'second', '', 'first_session',"
                " '', 0,"
                " 'dup', NULL, ?, 'open', '', ?)",
                (iso_cutoff(days=-7), iso_now()))
            self.brain.logs_conn_w.commit()
        r2 = self._file('updated', dedup_key='dup')
        self.assertTrue(r2['ok'])
        self.assertTrue(r2.get('updated'))

    def test_live_rejects_queue_shaped_params(self):
        r = self._file('fyi', for_whom='live', session_id=S1, when='2h')
        self.assertFalse(r['ok'])
        self.assertIn('live', r['error'])
        r = self._file('fyi', for_whom='live', session_id=S1,
                       needs_answer=True)
        self.assertFalse(r['ok'])

    def test_filing_is_never_refused_for_volume(self):
        """The Thalamus is always open for work: a producer may hold any
        number of open items, broadcast or directed, dated far out or due
        now. What a reader is shown is bounded at the read."""
        for i in range(30):
            self.assertTrue(self._file('broadcast %d' % i)['ok'])
            self.assertTrue(self._file('s1 %d' % i, for_whom=S1)['ok'])
            self.assertTrue(self._file('later %d' % i, when='30d')['ok'])
        self.assertEqual(thalamus.list_items(self.brain, limit=500)['total'], 90)

    def test_list_pages_through_every_item(self):
        for i in range(7):
            self._file('item %d' % i)
        first = thalamus.list_items(self.brain, limit=3)
        self.assertEqual((first['count'], first['total']), (3, 7))
        seen = [i['id'] for i in first['items']]
        for offset in (3, 6):
            page = thalamus.list_items(self.brain, limit=3, offset=offset)
            self.assertEqual(page['total'], 7)
            seen += [i['id'] for i in page['items']]
        self.assertEqual(len(seen), 7)
        self.assertEqual(len(set(seen)), 7)
        self.assertEqual(thalamus.list_items(self.brain, limit=3,
                                             offset=9)['items'], [])

    def test_list_orders_open_then_due_newest_then_scheduled_soonest(self):
        now = self._file('due now')['id']
        later = self._file('due in 2d', when='2d')['id']
        soon = self._file('due in 1h', when='1h')['id']
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET deliver_at = ? WHERE id = ?',
                (iso_cutoff(hours=5), now))
            self.brain.logs_conn_w.commit()
        fresh = self._file('due now, just filed')['id']
        closed = self._file('dismissed')['id']
        thalamus.resolve(self.brain, closed, dismiss=True)
        order = [i['id'] for i in thalamus.list_items(
            self.brain, include_closed=True)['items']]
        self.assertEqual(order, [fresh, now, soon, later, closed])

    def test_equal_due_dates_page_without_repeats(self):
        for i in range(4):
            self._file('same slot %d' % i, when='2030-01-01T09:00:00+00:00')
        seen = [thalamus.list_items(self.brain, limit=1, offset=k)['items'][0]['id']
                for k in range(4)]
        self.assertEqual(len(set(seen)), 4)

    def test_dashboard_copies_the_list_order(self):
        """The dashboard may not import servers.* (its disconnection
        contract), so it carries a copy of the order; this pins it."""
        from dashboard.queries import thalamus as dash
        self.assertEqual(dash._LIST_ORDER_SQL, tc.LIST_ORDER_SQL)

    def test_live_now_requires_filing_session(self):
        r = self._file(for_whom='live')
        self.assertFalse(r['ok'])
        self.assertIn('session', r['error'])

    def test_live_now_delegates_to_courier_and_goes_terminal(self):
        r = self._file('main moved — rebase', for_whom='live', session_id=S1)
        self.assertTrue(r['ok'])
        self.assertEqual(r['route'], 'live')
        self.assertEqual(self._row(r['id'])[0], tc.STATE_SENT)
        # The courier holds the actual broadcast.
        courier = self.brain.logs_conn.execute(
            'SELECT address, from_session FROM self_inflight WHERE id = ?',
            (r['courier_id'],)).fetchone()
        self.assertEqual(courier[0], 'self:broadcast')
        self.assertEqual(courier[1], S1)
        # Terminal rows never enter the pull path.
        block, n = thalamus.pull(self.brain, S2, via='stop')
        self.assertEqual(n, 0)


class TestDoorContract(ThalamusBase):
    """The door's producer-facing contract: ONE result envelope across the
    module, and `source` as vocabulary rather than free text."""

    def test_one_envelope_across_the_module(self):
        """file() answers in resolve()/withdraw()'s shape — a Phase 2 producer
        learns one result contract, not two."""
        filed = self._file('x')
        self.assertIs(filed['ok'], True)
        self.assertIn('id', filed)
        self.assertIs(thalamus.resolve(self.brain, filed['id'],
                                       dismiss=True)['ok'], True)
        # A rejection carries the same key, inverted.
        self.assertIs(self._file('  ')['ok'], False)

    def test_free_text_source_rejects_loudly(self):
        """source is the budget key AND the withdraw-ownership key — a typo'd
        one gets a fresh budget and orphans its own items."""
        for bad in ('', 'Anchor', 'my source', 's2:', ':unit', 'a:b:c'):
            r = self._file('x', source=bad)
            self.assertIs(r['ok'], False, 'source %r should not file' % bad)
            self.assertIn('source', r['error'])

    def test_category_process_grammar_files(self):
        for good in ('anchor', 'encoder:sonnet', 's2:consolidation', 'hook'):
            self.assertIs(self._file('x', source=good)['ok'], True,
                          'source %r should file' % good)


class TestPull(ThalamusBase):

    def test_notice_delivers_once_across_sessions(self):
        r = self._file('heads up')
        block, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 1)
        self.assertIn(r['id'], block)
        # audience first_session: a second session gets nothing.
        _, n2 = thalamus.pull(self.brain, S2, via='stop')
        self.assertEqual(n2, 0)

    def test_push_once_per_session_is_ledger_idempotent(self):
        self._file('standing note', for_whom='all')
        _, n1 = thalamus.pull(self.brain, S1, via='boot')
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual((n1, n2), (1, 0))
        # A DIFFERENT session still gets its copy.
        _, n3 = thalamus.pull(self.brain, S2, via='boot')
        self.assertEqual(n3, 1)

    def test_cap_rotates_fewest_delivered_first(self):
        """Under PULL_MAX_ITEMS a stable age order hands the SAME items to
        every session and the rest starve (measured: 5 of 8 due asks
        rendered 61 times each, 3 never). Fewest-delivered-first makes the
        cap rotate: the items a first session did not get lead the next."""
        ids = [self._file('ask %d' % i, needs_answer=True)['id']
               for i in range(tc.PULL_MAX_ITEMS + 2)]
        block1, n1 = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n1, tc.PULL_MAX_ITEMS)
        starved = [i for i in ids if i not in block1]
        self.assertEqual(len(starved), 2)
        block2, n2 = thalamus.pull(self.brain, S2, via='boot')
        self.assertEqual(n2, tc.PULL_MAX_ITEMS)
        for i in starved:
            self.assertIn(i, block2)

    def test_saturation_stops_delivery_at_once_and_closes_after_the_grace(self):
        """MAX_DELIVERIES_PER_ITEM sessions saw an every_session item: it is
        due to no one more. The CLOSE waits out the grace — then the sweep
        expires it like a window end — loud for an ask (naming the sessions
        that passed), and warned for a notice: an item every session scrolled
        past is not a silent write-off either."""
        ask = self._file('decide X?', needs_answer=True, dedup_key='k')
        note = self._file('standing note', for_whom='all', source='other')
        for k in range(tc.MAX_DELIVERIES_PER_ITEM):
            _, n = thalamus.pull(self.brain, 'sess-%02d' % k, via='boot')
            self.assertEqual(n, 2, k)
        block, n = thalamus.pull(self.brain, 'sess-late', via='boot')
        self.assertEqual((block, n), ('', 0))
        self._age_deliveries()
        before = len(self._dead_letters())
        self.assertEqual(thalamus.expire_due(self.brain), 2)
        self.assertEqual(self._row(ask['id'])[0], tc.STATE_EXPIRED)
        self.assertEqual(self._row(note['id'])[0], tc.STATE_EXPIRED)
        letters = self._dead_letters()
        self.assertEqual(len(letters), before + 1)
        self.assertIn('%d sessions saw it' % tc.MAX_DELIVERIES_PER_ITEM,
                      letters[0])
        warnings = self._saturation_warnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn(note['id'], warnings[0])
        self.assertIn('none acted', warnings[0])

    def test_the_saturating_session_can_still_answer(self):
        """Saturation is reached BY a delivery, so the session that tipped it
        is mid-thread when the sweep runs. Closing then would reject its
        answer and dead-letter a question that was about to be answered — a
        window end has no reader, saturation always does."""
        r = self._file('decide X?', needs_answer=True)
        for k in range(tc.MAX_DELIVERIES_PER_ITEM):
            thalamus.pull(self.brain, 'sess-%02d' % k, via='boot')
        self.assertEqual(thalamus.expire_due(self.brain), 0)
        self.assertEqual(self._row(r['id'])[0], tc.STATE_OPEN)
        out = thalamus.resolve(self.brain, r['id'], answer='here it is')
        self.assertEqual((out['ok'], out['state']), (True, tc.STATE_ANSWERED))
        self.assertEqual(self._dead_letters(), [])

    def test_a_question_closed_unanswered_cannot_be_re_asked_at_once(self):
        """Expiry frees the dedup key, so without the cool-off a cyclic
        producer's next run inserts a fresh item with a fresh epoch and a
        fresh cap — the treadmill the cap exists to end. Its OWN withdrawal
        does not gate: that is the producer correcting itself."""
        r = self._file('decide X?', needs_answer=True, dedup_key='k')
        self._age_out(r['id'])
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        again = self._file('decide X?', needs_answer=True, dedup_key='k')
        self.assertFalse(again['ok'])
        self.assertIn('cool-off', again['error'])
        self.assertIn(r['id'], again['error'])
        # Its OWN withdrawal does not gate — that is the producer
        # correcting itself.
        w = self._file('another question', needs_answer=True, dedup_key='k2')
        thalamus.withdraw(self.brain, 'test', item_id=w['id'])
        self.assertTrue(self._file('another question', needs_answer=True,
                                   dedup_key='k2')['ok'])
        # Nor does a DISMISSAL: the reader acted on a live item, and
        # re-asking in a better form is a designed flow.
        d = self._file('third question', needs_answer=True, dedup_key='k3')
        thalamus.resolve(self.brain, d['id'], dismiss=True)
        self.assertTrue(self._file('third question, sharper',
                                   needs_answer=True, dedup_key='k3')['ok'])

    def test_delivery_cap_is_a_live_knob_with_a_floor(self):
        """The config key overrides the code default at read time, no
        restart. The cap drives a TERMINAL transition, so an unusable value
        (0, a negative, a typo) is refused LOUDLY and the default stands —
        `count >= 0` would expire every standing item at zero deliveries."""
        self.brain.set_config(tc.MAX_DELIVERIES_CONFIG_KEY, 2)
        r = self._file('standing note', for_whom='all')
        for k in range(2):
            _, n = thalamus.pull(self.brain, 'sess-%02d' % k, via='stop')
            self.assertEqual(n, 1, k)
        _, n = thalamus.pull(self.brain, 'sess-late', via='stop')
        self.assertEqual(n, 0)
        self._age_deliveries()
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        self.assertEqual(self._row(r['id'])[0], tc.STATE_EXPIRED)
        for bad in (0, -3, 'not a number'):
            self.brain.set_config(tc.MAX_DELIVERIES_CONFIG_KEY, bad)
            self.assertEqual(thalamus._delivery_cap(self.brain),
                             tc.MAX_DELIVERIES_PER_ITEM, bad)
        self.assertTrue(self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log"
            " WHERE source='thalamus_cap_unusable'").fetchone()[0])
        fresh = self._file('never delivered', for_whom='all', source='other')
        self.assertEqual(thalamus.expire_due(self.brain), 0)
        self.assertEqual(self._row(fresh['id'])[0], tc.STATE_OPEN)

    def test_rotation_covers_every_repeating_item_not_just_asks(self):
        """The count key is keyed on the AUDIENCE that makes an item repeat:
        every_session items compete for the same slots for their whole
        window, and under a stable age order the newest of them never render
        at all — an ask-only key leaves standing notices starving."""
        ids = [self._file('standing %d' % i, for_whom='all')['id']
               for i in range(tc.PULL_MAX_ITEMS + 2)]
        block1, n1 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1, tc.PULL_MAX_ITEMS)
        starved = [i for i in ids if i not in block1]
        self.assertEqual(len(starved), 2)
        block2, _ = thalamus.pull(self.brain, S2, via='stop')
        for i in starved:
            self.assertIn(i, block2)

    def test_a_standing_item_never_sinks_below_fresh_one_shots(self):
        """The invariant the ask-class key once pinned, dropped when the key
        moved to the audience and its test went with it: a standing item one
        session has seen must not fall behind every fresh one-shot, which
        counts 0 for as long as it is due. Measured at the regression: the
        standing notice reached 0 of 6 sessions while one-shots arrived."""
        standing = self._file('standing note', for_whom='all')
        thalamus.pull(self.brain, S1, via='stop')          # count -> 1
        for i in range(tc.PULL_MAX_ITEMS):
            self._file('reminder %d' % i, source='other')
        block, n = thalamus.pull(self.brain, S2, via='stop')
        self.assertEqual(n, tc.PULL_MAX_ITEMS)
        self.assertIn(standing['id'], block)

    def test_both_audiences_progress_when_they_compete(self):
        """The mixed case — untested through two regressions, and the one
        that regressed both times. Standing items and one-shots contend for
        the same slots at the same moment; neither class may be shut out
        while the other drains."""
        standing = [self._file('standing %d' % i, for_whom='all')['id']
                    for i in range(3)]
        fresh = [self._file('one-shot %d' % i, source='p%d' % i)['id']
                 for i in range(4)]
        seen = ''.join(thalamus.pull(self.brain, s, via='stop')[0]
                       for s in (S1, S2))
        for i in standing:
            self.assertIn(i, seen, 'a standing item never rendered')
        for i in fresh:
            self.assertIn(i, seen, 'a one-shot never rendered')

    def test_the_budget_belongs_to_the_identity_not_the_generation(self):
        """A re-file that CHANGES the item re-arms it — sessions that saw the
        old text are owed the new one — but it does not renew the delivery
        budget. Counted per epoch, a producer whose body drifts renewed the
        cap on every run and the cap never bound it (measured: 32 deliveries
        against a cap of 16, beside an idempotent producer that stopped at
        16)."""
        cap = tc.MAX_DELIVERIES_PER_ITEM
        drift = self._file('question, run 0', needs_answer=True, dedup_key='d')
        seen = 0
        for k in range(cap * 2):
            block, _ = thalamus.pull(self.brain, 'sess-%02d' % k, via='boot')
            seen += drift['id'] in block
            self._file('question, run %d' % (k + 1), needs_answer=True,
                       dedup_key='d')
        self.assertEqual(seen, cap)

    def test_a_defer_grants_the_question_a_fresh_budget(self):
        """A defer is the reader saying "ask me again later" — it must not be
        answered by a dead-letter. The budget counts deliveries since the
        item last came due, and a defer moves that date, so the sessions that
        passed before it stop counting. Without this the sweep closed a
        just-deferred ask inside the grace and dead-lettered it as unanswered
        while it sat scheduled for next week."""
        r = self._file('decide X?', needs_answer=True, dedup_key='k')
        for k in range(tc.MAX_DELIVERIES_PER_ITEM):
            thalamus.pull(self.brain, 'sess-%02d' % k, via='boot')
        self.assertTrue(thalamus.resolve(self.brain, r['id'],
                                         defer_until='1w')['ok'])
        self._age_deliveries()
        self.assertEqual(thalamus.expire_due(self.brain), 0)
        self.assertEqual(self._row(r['id'])[0], tc.STATE_OPEN)
        self.assertEqual(self._dead_letters(), [])
        # and it delivers again once the deferral is up
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET deliver_at = ? WHERE id = ?',
                (iso_cutoff(hours=1), r['id']))
            self.brain.logs_conn_w.commit()
        _, n = thalamus.pull(self.brain, 'sess-next-week', via='boot')
        self.assertEqual(n, 1)

    def test_a_directed_asks_fate_carries_its_delivery_count(self):
        """The journal's producer view reads DIRECTED rows, which can never
        saturate — a fate derived from the cap was permanently silent there,
        so every expired directed ask read as "nobody was around". The count
        is raw, so it now says how many sessions passed on the question."""
        r = self._file('decide X?', needs_answer=True, dedup_key='k',
                       for_whom=S1, source='enc')
        _, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 1)
        self._age_out(r['id'])
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        item = thalamus.producer_items(self.brain, 'enc', S1,
                                       settled_days=7)[0]
        self.assertEqual(item['deliveries'], 1)
        self.assertEqual(tc.fate_of(item), tc.FATE_PASSED_OVER % 1)

    def test_a_window_end_notice_closes_quietly(self):
        """Only the CAP warns for an item that asked nothing. A notice
        reaching its window end is a natural death and must stay silent, or
        every routine expiry becomes an error row at every session's boot."""
        r = self._file('standing note', for_whom='all')
        thalamus.pull(self.brain, S1, via='stop')
        self._age_out(r['id'])
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        self.assertEqual(self._saturation_warnings(), [])
        self.assertEqual(self._dead_letters(), [])

    def test_a_refile_carrying_a_new_when_moves_the_budget_baseline(self):
        """The budget counts deliveries since the item last came due, so a
        producer re-filing with a new `when` moves the baseline and starts
        over — narrower than the armed_epoch door it replaced, since a
        body-only re-file cannot, but real. Pinned so it cannot drift
        without someone deciding to let it."""
        cap = tc.MAX_DELIVERIES_PER_ITEM
        r = self._file('decide X?', needs_answer=True, dedup_key='k')
        for k in range(cap):
            thalamus.pull(self.brain, 'sess-%02d' % k, via='boot')
        _, n = thalamus.pull(self.brain, 'sess-late', via='boot')
        self.assertEqual(n, 0, 'should be saturated at the cap')
        # a body-only re-file re-arms delivery but does NOT renew the budget
        self._file('decide X, sharper?', needs_answer=True, dedup_key='k')
        _, n = thalamus.pull(self.brain, 'sess-late2', via='boot')
        self.assertEqual(n, 0, 'a body change must not buy a fresh budget')
        # one carrying a new `when` moves the date the budget counts from
        again = self._file('decide X, sharper?', needs_answer=True,
                           dedup_key='k', when=iso_now())
        self.assertEqual(again['id'], r['id'])
        _, n = thalamus.pull(self.brain, 'sess-late3', via='boot')
        self.assertEqual(n, 1, 'a new `when` resets the budget baseline')

    def test_the_reask_cooloff_binds_questions_not_rearming_notices(self):
        """The gate exists so a dead QUESTION does not come back on a loop.
        A notice, a dated reminder and a directed item reach their window end
        as part of their normal cycle — re-filing them IS the re-arm — and a
        gate keyed on the expired STATE alone refused every one of them."""
        for label, kw in (('notice', {'for_whom': 'all'}),
                          ('reminder', {'when': '1h'}),
                          ('directed', {'for_whom': S1})):
            r = self._file('cycle %s' % label, dedup_key=label, **kw)
            self._age_out(r['id'])
            thalamus.expire_due(self.brain)
            again = self._file('cycle %s' % label, dedup_key=label, **kw)
            self.assertTrue(again['ok'],
                            '%s refused: %s' % (label, again.get('error')))
            self.assertNotEqual(again['id'], r['id'], label)
        ask = self._file('decide X?', needs_answer=True, dedup_key='ask')
        self._age_out(ask['id'])
        thalamus.expire_due(self.brain)
        blocked = self._file('decide X?', needs_answer=True, dedup_key='ask')
        self.assertFalse(blocked['ok'])
        self.assertIn('cool-off', blocked['error'])

    def test_refs_refresh_on_refile_but_never_rearm_or_vanish(self):
        """The anchor is derived and may vary run to run: a re-file that
        differs only in refs updates them silently — no epoch bump, no
        re-delivery. A re-file carrying NO refs leaves the stored anchor
        alone; omitting means "this run derived none", never "clear it"."""
        r = self._file('same body', for_whom='all', dedup_key='k')
        _, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 1)
        r2 = self._file('same body', for_whom='all', dedup_key='k',
                        refs=['d4ed89c5'])
        self.assertEqual((r2['id'], r2['updated'], r2['rearmed']),
                         (r['id'], True, False))
        item = next(i for i in thalamus.list_items(self.brain)['items']
                    if i['id'] == r['id'])
        self.assertEqual((item['refs'], item['armed_epoch']), (['d4ed89c5'], 0))
        _, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 0)
        # neither path may erase it — the quiet refresh or the re-arm
        self._file('same body', for_whom='all', dedup_key='k')
        self._file('a changed body', for_whom='all', dedup_key='k')
        item = next(i for i in thalamus.list_items(self.brain)['items']
                    if i['id'] == r['id'])
        self.assertEqual(item['refs'], ['d4ed89c5'])

    def test_broadcast_ask_boot_only(self):
        r = self._file('decide X?', needs_answer=True)
        _, n_stop = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n_stop, 0)
        block, n_boot = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n_boot, 1)
        self.assertIn('thalamus_resolve("%s"' % r['id'], block)

    def test_directed_ask_stop_only_and_only_its_target(self):
        """The Scribe asking the session it just encoded: Stop, not boot
        (the named session has already had its boot), and never another
        session."""
        r = self._file('revise, or leave?', needs_answer=True, for_whom=S1)
        _, n_boot = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n_boot, 0)
        _, n_other = thalamus.pull(self.brain, S2, via='stop')
        self.assertEqual(n_other, 0)
        block, n_stop = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n_stop, 1)
        self.assertIn('thalamus_resolve("%s", answer=' % r['id'], block)

    def test_future_deliver_at_not_due_yet(self):
        self._file('later', when='1w')
        _, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 0)

    def test_directed_reaches_only_its_target(self):
        self._file('for one stream', for_whom=S1)
        _, n_other = thalamus.pull(self.brain, S2, via='boot')
        self.assertEqual(n_other, 0)
        _, n_target = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n_target, 1)

    def test_ledger_records_via_and_session(self):
        r = self._file('note')
        thalamus.pull(self.brain, S1, via='stop')
        row = self.brain.logs_conn.execute(
            'SELECT session_id, via FROM thalamus_deliveries WHERE item_id = ?',
            (r['id'],)).fetchone()
        self.assertEqual((row[0], row[1]), (S1, 'stop'))

    def test_empty_session_is_noop(self):
        self._file('note')
        self.assertEqual(thalamus.pull(self.brain, '', via='boot'), ('', 0))

    def test_overflow_names_the_true_count(self):
        for i in range(tc.PULL_MAX_ITEMS + 4):
            self._file('item %d' % i, source='src-%d' % i)  # distinct budgets
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, tc.PULL_MAX_ITEMS)
        self.assertIn('(+4 more due', block)  # the real number, not "+1"


class TestResolveWithdraw(ThalamusBase):

    def test_answer_closes_and_stores_payload(self):
        r = self._file('decide X?', needs_answer=True)
        out = thalamus.resolve(self.brain, r['id'], answer='derive from aspects')
        self.assertTrue(out['ok'])
        state, answer = self.brain.logs_conn.execute(
            'SELECT state, answer FROM thalamus_items WHERE id = ?',
            (r['id'],)).fetchone()
        self.assertEqual(state, tc.STATE_ANSWERED)
        self.assertEqual(answer, 'derive from aspects')

    def test_exactly_one_action(self):
        r = self._file('x')
        out = thalamus.resolve(self.brain, r['id'],
                               answer='a', dismiss=True)
        self.assertFalse(out['ok'])
        out = thalamus.resolve(self.brain, r['id'])
        self.assertFalse(out['ok'])

    def test_empty_answer_rejected(self):
        r = self._file('decide?', needs_answer=True)
        out = thalamus.resolve(self.brain, r['id'], answer='   ')
        self.assertFalse(out['ok'])
        self.assertIn('empty', out['error'])
        # Item stays open and answerable.
        self.assertEqual(self._row(r['id'])[0], tc.STATE_OPEN)

    def test_defer_rearms_delivery(self):
        r = self._file('note')
        thalamus.pull(self.brain, S1, via='stop')  # delivered
        out = thalamus.resolve(self.brain, r['id'], defer_until='1h')
        self.assertTrue(out['ok'])
        # Re-armed (new epoch), but not due yet → still nothing.
        _, n = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n, 0)
        deliver_at = self._row(r['id'])[3]
        self.assertGreater(deliver_at, iso_now())
        # The ledger is APPEND-ONLY: defer preserves the delivery history —
        # "delivered, then deferred" must stay distinguishable from "never
        # delivered" (Phase 3 retry gates on unacked).
        ledger = self.brain.logs_conn.execute(
            'SELECT COUNT(*) FROM thalamus_deliveries WHERE item_id = ?',
            (r['id'],)).fetchone()[0]
        self.assertEqual(ledger, 1)
        epoch = self.brain.logs_conn.execute(
            'SELECT armed_epoch FROM thalamus_items WHERE id = ?',
            (r['id'],)).fetchone()[0]
        self.assertEqual(epoch, 1)

    def test_rearm_redelivers_to_same_session_and_keeps_history(self):
        r = self._file('note')
        _, n1 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1, 1)
        thalamus.resolve(self.brain, r['id'], defer_until='now')
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n2, 1)  # same session, new epoch → delivers again
        rows = self.brain.logs_conn.execute(
            'SELECT armed_epoch FROM thalamus_deliveries WHERE item_id = ?'
            ' ORDER BY armed_epoch', (r['id'],)).fetchall()
        self.assertEqual([e for (e,) in rows], [0, 1])  # both generations kept
        out = thalamus.list_items(self.brain)
        item = next(i for i in out['items'] if i['id'] == r['id'])
        self.assertEqual(item['deliveries'], 2)        # all-time
        self.assertEqual(item['deliveries_epoch'], 1)  # current generation

    def test_dedup_refile_rearms_delivery(self):
        """A re-file under the same (source, dedup_key) is a re-arm: a
        once-item already delivered must deliver again with the updated
        content — not stay suppressed by the old generation's ledger row."""
        self._file('v1', dedup_key='concern-x')
        _, n1 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1, 1)
        r2 = self._file('v2 updated concern', dedup_key='concern-x')
        self.assertTrue(r2.get('updated'))
        block, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n2, 1)
        self.assertIn('v2 updated concern', block)

    def test_identical_refile_is_idempotent(self):
        """An UNCHANGED re-file (a cyclic producer re-asserting its standing
        item) must not re-arm — bumping on a no-op would re-deliver the same
        text every producer cycle, unbounded. It refreshes the window only."""
        self._file('standing concern', dedup_key='concern-x')
        _, n1 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1, 1)
        r2 = self._file('standing concern', dedup_key='concern-x')
        self.assertTrue(r2['updated'])
        self.assertFalse(r2['rearmed'])
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n2, 0)  # no re-delivery
        # Window refreshed: expires_at moved forward with the re-file.
        self.assertGreater(self._row(r2['id'])[4], iso_now())

    def test_deferred_ask_keeps_ask_window(self):
        """extend_window composes through window_for — a deferred ask keeps
        its kind's full span (14d) past the new due date, not a reminder's
        7-day grace."""
        from datetime import datetime, timedelta
        r = self._file('decide X?', needs_answer=True)
        out = thalamus.resolve(self.brain, r['id'], defer_until='30d')
        self.assertTrue(out['ok'])
        due = datetime.fromisoformat(out['deliver_at'])
        expires = datetime.fromisoformat(out['expires_at'])
        self.assertGreaterEqual(expires - due,
                                timedelta(days=tc.ASK_EXPIRES_DAYS))

    def test_resolve_unknown_or_closed(self):
        self.assertFalse(thalamus.resolve(self.brain, 'th_nope',
                                          dismiss=True)['ok'])
        r = self._file('x')
        thalamus.resolve(self.brain, r['id'], dismiss=True)
        again = thalamus.resolve(self.brain, r['id'], dismiss=True)
        self.assertFalse(again['ok'])
        self.assertIn('already', again['error'])

    def test_withdraw_is_source_scoped(self):
        r = self._file('mine', source='s2:consolidation')
        stolen = thalamus.withdraw(self.brain, 'someone-else', item_id=r['id'])
        self.assertFalse(stolen['ok'])
        own = thalamus.withdraw(self.brain, 's2:consolidation', item_id=r['id'])
        self.assertTrue(own['ok'])
        self.assertEqual(self._row(r['id'])[0], tc.STATE_WITHDRAWN)

    def test_withdraw_by_dedup_key(self):
        self._file('mine', source='s2:x', dedup_key='k1')
        out = thalamus.withdraw(self.brain, 's2:x', dedup_key='k1')
        self.assertTrue(out['ok'])


class TestExpiry(ThalamusBase):

    def test_notice_expires_naturally(self):
        r = self._file('old news')
        self._age_out(r['id'])
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        self.assertEqual(self._row(r['id'])[0], tc.STATE_EXPIRED)

    def test_expired_ask_is_loud_dead_letter(self):
        r = self._file('never answered?', needs_answer=True)
        self._age_out(r['id'])
        before = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source='thalamus_ask_expired'"
        ).fetchone()[0]
        self.assertEqual(thalamus.expire_due(self.brain), 1)
        after = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source='thalamus_ask_expired'"
        ).fetchone()[0]
        self.assertEqual(after, before + 1)

    def test_expired_never_delivers(self):
        r = self._file('too late')
        self._age_out(r['id'])
        _, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 0)


class TestVocabulary(ThalamusBase):
    """Contract-first: moments and audiences are closed vocabulary — unknown
    values fail LOUDLY instead of silently behaving as a default."""

    def test_unknown_via_is_loud(self):
        self._file('x')
        with self.assertRaises(ValueError):
            thalamus.pull(self.brain, S1, via='bot')  # the ledgered-typo case

    def test_internal_audience_drift_is_loud(self):
        """An audience outside tc.AUDIENCES matches neither pull-predicate
        branch (open forever, silent death at expiry) — the door rejects."""
        from unittest import mock
        with mock.patch.object(tc, 'resolve_for_whom',
                               return_value=('queue', 'weekly', '')):
            r = self._file('x')
        self.assertFalse(r['ok'])
        self.assertIn('audience', r['error'])

    def test_resolve_for_whom_outputs_stay_inside_audiences(self):
        """The contract half of the audience guard (gate 4): every reachable
        resolve_for_whom audience is in the closed set, and the pull
        predicate binds BOTH members — resolver, vocabulary, and predicate
        cannot drift apart without this failing at merge time. (The runtime
        tripwire in file() covers the future-new-branch case a fixed
        enumeration can't see.)"""
        cases = [(None, False), (None, True), ('', False), ('', True),
                 ('all', False), ('all', True),
                 ('123e4567-e89b-12d3-a456-426614174000', False)]
        for fw, na in cases:
            route, audience, _ = tc.resolve_for_whom(fw, na)
            if route == 'queue':
                self.assertIn(
                    audience, tc.AUDIENCES,
                    'resolve_for_whom(%r, %r) -> %r escaped the closed set'
                    % (fw, na, audience))
        _, params = thalamus._due_filter(
            S1, 'boot', '2027-01-01T00:00:00+00:00',
            tc.MAX_DELIVERIES_PER_ITEM)
        for member in tc.AUDIENCES:
            self.assertIn(member, params,
                          'pull predicate does not bind audience %r' % member)

    def test_ddl_audience_default_is_inside_the_closed_set(self):
        """The CREATED table's audience default must be a member of the closed
        set. A default outside it matches neither pull-predicate branch, so any
        insert omitting the column writes a row that never delivers and dies
        silently at expiry — below the reach of file()'s runtime tripwire. The
        v3 rename updated existing rows but left the column default at 'once'.
        """
        default = None
        for col in self.brain.logs_conn.execute(
                'PRAGMA table_info(thalamus_items)').fetchall():
            if col[1] == 'audience':
                default = (col[4] or '').strip("'")
        self.assertIn(
            default, tc.AUDIENCES,
            'thalamus_items.audience DDL default %r is not in %s — an insert '
            'omitting the column would be undeliverable' % (default,
                                                            tc.AUDIENCES))

    def test_ask_moments_cover_every_audience_with_real_moments(self):
        """ASK_MOMENTS is keyed by audience: an audience without an entry
        would make its asks due at NO moment (the directed-ask dead-letter
        of 2026-09-01, id:178f4727, reborn), and a moment outside MOMENTS
        would never be pulled."""
        self.assertEqual(set(tc.ASK_MOMENTS), set(tc.AUDIENCES))
        for moments in tc.ASK_MOMENTS.values():
            self.assertTrue(moments)
            self.assertTrue(set(moments) <= set(tc.MOMENTS))

    def test_kind_is_one_derivation_for_verb_and_span(self):
        self.assertEqual(tc.kind_of({'needs_answer': 1}), tc.KIND_ASK)
        self.assertEqual(tc.kind_of({'deliver_at': '2027-01-01T00:00:00+00:00'}),
                         tc.KIND_REMINDER)
        self.assertEqual(tc.kind_of({}), tc.KIND_NOTICE)
        # Both partitions read the same derivation.
        self.assertEqual(set(tc.KIND_VERB), set(tc.KIND_EXPIRES_DAYS))
        # window_for anchors a dated item's span at its due date.
        due = '2027-01-01T00:00:00+00:00'
        self.assertGreater(tc.window_for(False, due), due)

    def test_kind_verbs_render(self):
        self._file('decide?', needs_answer=True)
        self._file('fyi', source='s3')
        block, _ = thalamus.pull(self.brain, S1, via='boot')
        self.assertIn('asks', block)
        self.assertIn('notes', block)


class TestMaintenanceSweep(ThalamusBase):

    def test_sweep_fires_without_s2_conditions(self):
        """The channel sweeps run AHEAD of the S2 fire gate inside
        run_maintenance_if_due — a brain whose S2 cycle declines (keyless,
        boot grace, idle gates) still expires its items and dead-letters
        unanswered asks."""
        r = self._file('never answered?', needs_answer=True)
        self._age_out(r['id'])
        result = self.brain.run_maintenance_if_due()
        # S2 itself must have declined (test brain: boot grace / no LLM) —
        # the sweep firing anyway is exactly the point.
        self.assertIsNone(result)
        self.assertEqual(self._row(r['id'])[0], tc.STATE_EXPIRED)

    def test_sweep_is_hourly_throttled(self):
        self.brain.run_maintenance_if_due()
        r = self._file('expires between polls')
        self._age_out(r['id'])
        self.brain.run_maintenance_if_due()  # within the hour — no sweep
        self.assertEqual(self._row(r['id'])[0], tc.STATE_OPEN)
        self.brain._thalamus_sweep_checked = 0  # an hour passes
        self.brain.run_maintenance_if_due()
        self.assertEqual(self._row(r['id'])[0], tc.STATE_EXPIRED)


class TestRefs(ThalamusBase):
    """pull() owns ref resolution — ONE veil-aware batch; the contract only
    formats what it is handed (ref_lines)."""

    def test_refs_resolve_in_one_batched_call(self):
        from unittest import mock
        from tests.test_scopes import _mint
        a = _mint(self.brain, 'sess-a', '', 'first ref title')
        b = _mint(self.brain, 'sess-a', '', 'second ref title')
        self._file('note one', refs=[a, b])
        self._file('note two', refs=[a], source='other')
        with mock.patch.object(self.brain, 'filter_nodes',
                               wraps=self.brain.filter_nodes) as fn:
            block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 2)
        self.assertEqual(fn.call_count, 1)  # batched: one call for the block
        self.assertIn('first ref title', block)
        self.assertIn('second ref title', block)

    def test_walled_ref_renders_bare_not_title(self):
        """A globally-filed item can ref a walled node — its title must not
        print into another session's pull (default-deny outward veil)."""
        from tests.test_scopes import _mint, _set_scopes, ISOLATE_CLIENT_X
        walled = _mint(self.brain, 'sess-client', 'client-x',
                       'client secret plan')
        _set_scopes(self.brain, ISOLATE_CLIENT_X)
        self._file('global note', refs=[walled])
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 1)
        self.assertNotIn('client secret plan', block)
        self.assertIn(walled[:8], block)  # the ref itself renders bare

    def test_bad_ref_renders_bare(self):
        self._file('note', refs=['deadbeef'])
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertEqual(n, 1)
        self.assertIn('deadbeef', block)


class TestRender(ThalamusBase):

    def test_long_body_truncates_loudly(self):
        r = self._file('x' * (tc.BODY_MAX + 500))
        block, _ = thalamus.pull(self.brain, S1, via='boot')
        self.assertIn('chars', block)          # the loud marker, not a bare cut
        self.assertIn('thalamus_list', block)  # points at the full text
        self.assertLess(len(block), tc.BLOCK_MAX + 500)

    def test_cap_dropped_items_not_ledgered(self):
        """Only items the block actually shows are ledgered — a cap-dropped
        item was never delivered and stays armed for the next moment."""
        for i in range(4):
            self._file('x' * 1400, source='src-%d' % i)  # distinct budgets
        _, n1 = thalamus.pull(self.brain, S1, via='boot')
        self.assertLess(n1, 4)  # the block cap dropped at least one
        ledgered = self.brain.logs_conn.execute(
            'SELECT COUNT(*) FROM thalamus_deliveries').fetchone()[0]
        self.assertEqual(ledgered, n1)
        _, n2 = thalamus.pull(self.brain, S1, via='stop')
        self.assertEqual(n1 + n2, 4)  # the dropped items deliver next moment

    def test_block_head_counts_shown_items(self):
        """Head, tail, ledger, and pull's count all say `kept` — the head
        must not claim fetched items the cap dropped."""
        for i in range(4):
            self._file('x' * 1400, source='src-%d' % i)
        block, n = thalamus.pull(self.brain, S1, via='boot')
        self.assertLess(n, 4)
        self.assertIn('— %d item(s)' % n, block)
        self.assertIn('(+%d more due' % (4 - n), block)

    def test_block_head_carries_relay_note(self):
        """The reader's standing instruction sits between the head and the
        first item — plain words to the operator, never id/moment/producer —
        and names the read routes; it is reserved from the budget."""
        self._file('note')
        block, _ = thalamus.pull(self.brain, S1, via='boot')
        head, rest = block.split('\n', 1)
        self.assertIn('— 1 item(s)', head)
        self.assertTrue(rest.startswith(tc._HEAD_NOTE + '\n\n• th_'))
        for phrase in ('plain words', 'never the id, the moment, or the producer',
                       'thalamus_list', "recall_episodes(ref_type='thalamus_delivery')",
                       'thalamus_resolve'):
            self.assertIn(phrase, tc._HEAD_NOTE)
        self.assertLessEqual(len(block), tc.BLOCK_MAX)

    def test_list_items_filters_by_source_and_target(self):
        """The producer's own-items read behind the journal continuity
        join: one producer, one directed recipient, closed items on request."""
        a = self._file('for s1', source='encoder:sonnet', for_whom=S1)
        self._file('for s2', source='encoder:sonnet', for_whom=S2)
        self._file('broadcast', source='encoder:sonnet')
        self._file('other producer', source='anchor', for_whom=S1)
        ids = lambda **kw: {i['id'] for i in
                            thalamus.list_items(self.brain, **kw)['items']}
        self.assertEqual(len(ids(source='encoder:sonnet')), 3)
        self.assertEqual(ids(source='encoder:sonnet', target_session=S1),
                         {a['id']})
        self.assertEqual(len(ids(target_session=S1)), 2)
        thalamus.resolve(self.brain, a['id'], dismiss=True)
        self.assertEqual(ids(source='encoder:sonnet', target_session=S1),
                         set())
        self.assertEqual(ids(source='encoder:sonnet', target_session=S1,
                             include_closed=True), {a['id']})

    def test_producer_items_is_identity_exact_windowed_and_open_first(self):
        """The producer's own read: (source, target) exact — '' IS the
        broadcast recipient; every open item regardless of age; settled items
        only within the window and only in a fate the producer hears about
        (withdrawn/sent excluded in SQL); open rows first, then by when they
        last changed."""
        b = self._file('broadcast', dedup_key='b')
        d = self._file('directed', dedup_key='d', for_whom=S1)
        ids = lambda *a, **kw: [i['id'] for i in
                                thalamus.producer_items(self.brain, 'test',
                                                        *a, **kw)]
        self.assertEqual(ids(''), [b['id']])
        self.assertEqual(ids(S1), [d['id']])
        old_open = self._file('ancient but open', dedup_key='o', for_whom=S1)
        thalamus.resolve(self.brain, d['id'], answer='yes')
        w = self._file('withdrawn', dedup_key='w', for_whom=S1)
        thalamus.withdraw(self.brain, 'test', dedup_key='w', target_session=S1)
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET created_at = ?, updated_at = ? '
                'WHERE id = ?', (iso_cutoff(days=40), iso_cutoff(days=40),
                                 old_open['id']))
            self.brain.logs_conn_w.commit()
        # open only (no window)
        self.assertEqual(ids(S1), [old_open['id']])
        # windowed: open first, then the answered one; withdrawn never
        self.assertEqual(ids(S1, settled_days=7), [old_open['id'], d['id']])
        with self.brain.logs_write_lock:
            self.brain.logs_conn_w.execute(
                'UPDATE thalamus_items SET updated_at = ? WHERE id = ?',
                (iso_cutoff(days=8), d['id']))
            self.brain.logs_conn_w.commit()
        self.assertEqual(ids(S1, settled_days=7), [old_open['id']])
        self.assertNotIn(w['id'], ids(S1, settled_days=365))

    def test_fate_of_mirrors_kind_of(self):
        self.assertEqual(tc.fate_of({'state': tc.STATE_OPEN}), tc.FATE_OPEN)
        self.assertEqual(tc.fate_of({'state': tc.STATE_ANSWERED}),
                         tc.FATE_ANSWERED)
        self.assertEqual(tc.fate_of({'state': tc.STATE_EXPIRED}),
                         tc.FATE_EXPIRED)
        self.assertEqual(tc.fate_of({'state': tc.STATE_EXPIRED,
                                     'needs_answer': True}),
                         tc.FATE_EXPIRED_UNANSWERED)
        for own_act in (tc.STATE_WITHDRAWN, tc.STATE_SENT):
            self.assertIsNone(tc.fate_of({'state': own_act}))
        self.assertEqual(set(tc.FATE_STATES),
                         {tc.STATE_OPEN, tc.STATE_ANSWERED,
                          tc.STATE_DISMISSED, tc.STATE_EXPIRED})

    def test_touch_rejects_unknown_and_owned_columns_before_sql(self):
        r = self._file('x')
        for bad in ({'staet': tc.STATE_DISMISSED}, {'updated_at': iso_now()},
                    {'armed_epoch': 3}, {'id': 'th_x'}):
            with self.assertRaises(ValueError, msg=str(bad)):
                with self.brain.logs_write_lock:
                    thalamus._touch(self.brain.logs_conn_w, r['id'], iso_now(),
                                    **bad)

    def test_list_items_carries_updated_at(self):
        """Settled-recently reads key on updated_at: stamped at insert, moved
        by every state change."""
        r = self._file('note')
        item = thalamus.list_items(self.brain)['items'][0]
        self.assertTrue(item['updated_at'])
        thalamus.resolve(self.brain, r['id'], dismiss=True)
        closed = thalamus.list_items(self.brain, include_closed=True)['items'][0]
        self.assertGreaterEqual(closed['updated_at'], item['updated_at'])

    def test_list_items_shows_delivery_counts(self):
        r = self._file('note')
        thalamus.pull(self.brain, S1, via='boot')
        out = thalamus.list_items(self.brain)
        item = next(i for i in out['items'] if i['id'] == r['id'])
        self.assertEqual(item['deliveries'], 1)


class TestFiledTrace(ThalamusBase):
    """Step 13(d): a filing made from a producer's run leaves ONE
    `thalamus_filed` delta row on the run chain (ref_id = item id) through
    the traces door — the filed→delivered→answered join. No chain, no row."""

    CHAIN = 's1e-aaaaaaaa-3'

    def _filed_rows(self, chain=CHAIN):
        return [e for e in self.brain.query_traces(chain_id=chain)['chain']
                if e['ref_type'] == 'thalamus_filed']

    def test_run_filing_writes_one_row_on_the_run_chain(self):
        r = self._file('you are proceeding on "I wonder if", not a yes',
                       source='encoder:sonnet', for_whom=S1, session_id=S1,
                       run_chain=self.CHAIN)
        rows = self._filed_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        # scale is the chain's — derived, never a second parameter
        self.assertEqual((row['scale'], row['event_type'], row['ref_id'],
                          row['session_id']),
                         ('s1', 'delta', r['id'], S1))
        meta = row['metadata']
        from servers.trace_contract import THALAMUS_FILED_METADATA_SHAPE
        # the boundary adds its provenance stamps; the shape's keys all land
        self.assertTrue(set(THALAMUS_FILED_METADATA_SHAPE) <= set(meta))
        self.assertEqual((meta['source'], meta['target_session'],
                          meta['needs_answer'], meta['route'], meta['filing']),
                         ('encoder:sonnet', S1, False, 'queue', 'new'))
        self.assertIn('I wonder if', meta['body'])

    def test_no_chain_no_row(self):
        self._file('interactive remind', session_id=S1)
        self.assertEqual(
            self.brain.query_traces(ref_type='thalamus_filed', scale='s1',
                                    hours=None)['events'], [])

    def test_rejected_filing_leaves_no_row(self):
        r = self._file('  ', run_chain=self.CHAIN)
        self.assertFalse(r['ok'])
        self.assertEqual(self._filed_rows(), [])

    def test_refile_rows_name_what_the_door_did(self):
        """Three runs, one item: new → refresh (identical re-file, window
        only) → rearm (changed re-file, delivers again). Each run's row says
        what THAT run did; the 13(g) dedup-updates-vs-inserts count reads
        `filing`."""
        chains = ['s1e-aaaaaaaa-3', 's1e-aaaaaaaa-4', 's1e-aaaaaaaa-5']
        for chain, body in zip(chains, ('v1', 'v1', 'v2')):
            self._file(body, dedup_key='7e6decd2', needs_answer=True,
                       for_whom=S1, run_chain=chain)
        rows = [self._filed_rows(c) for c in chains]
        self.assertEqual([len(r) for r in rows], [1, 1, 1])
        self.assertEqual(len({r[0]['ref_id'] for r in rows}), 1)
        self.assertEqual([r[0]['metadata']['filing'] for r in rows],
                         ['new', 'refresh', 'rearm'])
        self.assertTrue(rows[2][0]['metadata']['needs_answer'])
        self.assertEqual(rows[2][0]['metadata']['dedup_key'], '7e6decd2')

    def test_s2_run_filing_is_traced_on_its_chain(self):
        """An S2 unit files broadcast (no session) — its row lands on the
        s2 run chain with the derived scale."""
        r = self._file('s2 asks', source='s2:consolidation', needs_answer=True,
                       run_chain='s2-20260905120000-consolidation')
        rows = self._filed_rows('s2-20260905120000-consolidation')
        self.assertEqual([(e['scale'], e['ref_id']) for e in rows],
                         [('s2', r['id'])])
        self.assertEqual(rows[0]['metadata']['target_session'], '')

    def test_unregistered_scale_is_loud_and_never_masks_the_filing(self):
        """A filing is a run's act, not a turn's: an s0 chain fails at the
        write boundary, is logged, and the item still files."""
        r = self._file('s0 tried', run_chain='s0-aaaaaaaa-3')
        self.assertTrue(r['ok'])
        self.assertEqual(self._filed_rows('s0-aaaaaaaa-3'), [])
        n = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='error' "
            "AND source = ?", ('thalamus_filed_trace_failed',)).fetchone()[0]
        self.assertEqual(n, 1)


class TestDispatchEnvelope(ThalamusBase):
    """Every handler returns {"ok", ...} — the dispatch_self lesson."""

    def test_remind_enveloped(self):
        r = _handle_remind(self.brain, {'what': 'check X', 'when': '1d'}, [])
        self.assertIs(r.get('ok'), True)
        self.assertIn('result', r)
        self.assertTrue(r['result']['id'].startswith('th_'))

    def test_remind_rejection_is_error_envelope(self):
        r = _handle_remind(self.brain, {'what': '   '}, [])
        self.assertIs(r.get('ok'), False)
        self.assertIn('empty body', r.get('error', ''))

    def test_list_and_resolve_enveloped(self):
        filed = _handle_remind(self.brain, {'what': 'x'}, [])
        r = _handle_thalamus_list(self.brain, {}, [])
        self.assertIs(r.get('ok'), True)
        self.assertEqual(r['result']['count'], 1)
        rid = filed['result']['id']
        r = _handle_thalamus_resolve(self.brain, {'id': rid, 'dismiss': True}, [])
        self.assertIs(r.get('ok'), True)
        r = _handle_thalamus_resolve(self.brain, {'id': rid, 'dismiss': True}, [])
        self.assertIs(r.get('ok'), False)


if __name__ == '__main__':
    unittest.main()
