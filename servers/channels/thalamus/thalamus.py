"""Thalamus mechanics — file / pull / resolve / withdraw / expire.

Every verb answers in ONE envelope — {'ok': bool, 'id': …} — so a producer
learns one result contract for the whole module.

One producer door (`file()`), three entrances: Anchor's MCP tools
(dispatch_thalamus.py), agent toolsets (Phase 2), direct code calls. Routing
lives inside the door, which validates and resolves the grammars and then
hands off to one storage branch: a live-now FYI delegates to the courier
broadcast and goes terminal immediately (`_file_live`); everything with a
clock, a window, or an answer queues (`_file_queued`). Delivery is PULL-ONLY
at the two moments that provably land — the Stop
drain and the first-prompt render — so the Thalamus never enumerates sessions, never
pushes, holds no roster: each session self-serves against the item list and
records its own delivery in the ledger at render time (annotate-at-render —
the only visibility mechanism that survives courier receipt expiry, brain
node 8a170558).

Storage: thalamus_items / thalamus_deliveries in brain_logs.db (DDL in
servers/schema.py). Reads via brain.logs_conn; writes serialize on
brain.logs_write_lock through brain.logs_conn_w — the self-channel pattern.
Time is WALL-CLOCK throughout (courier-class deadlines; see thalamus_contract).

Design: docs/THALAMUS-DESIGN.md
"""

import json
import uuid

from servers.clock import iso_now, iso_cutoff
from servers.channels.thalamus import thalamus_contract as tc


# The item's columns, declared ONCE as names. Everything else derives: the
# SELECT text, the INSERT text, and _row_to_item's mapping — which is BY NAME,
# so a column may be added anywhere in this tuple without silently shifting
# every field after it into the wrong key.
_ITEM_COLS = ('id', 'source', 'body', 'refs', 'audience', 'target_session',
              'needs_answer', 'dedup_key', 'deliver_at', 'expires_at',
              'state', 'answer', 'answered_at', 'created_at', 'updated_at',
              'armed_epoch')

_ITEM_SELECT = ', '.join(_ITEM_COLS)

# The door supplies every column it writes; armed_epoch is the exception —
# it takes its DDL default, generation 0. Derived rather than re-listed so a
# new column reaches the INSERT too, where _insert_item's KeyError makes the
# author decide what to write instead of defaulting silently.
_INSERT_COLS = tuple(c for c in _ITEM_COLS if c != 'armed_epoch')


def _row_to_item(row):
    """A SELECT of _ITEM_COLS → item dict, mapped by NAME. Extra trailing
    columns (list_items' delivery counts) are ignored — zip stops at the
    names — and the caller attaches them itself."""
    item = dict(zip(_ITEM_COLS, row))
    item['refs'] = json.loads(item['refs'] or '[]')
    item['needs_answer'] = bool(item['needs_answer'])
    return item


def _ok(**fields):
    """The door's success envelope — `ok` + `id`, the shape resolve() and
    withdraw() already return, so a producer learns ONE result contract."""
    return {'ok': True, **fields}


def _reject(error):
    """The door's LOUD synchronous rejection — same envelope, inverted."""
    return {'ok': False, 'error': error}


def _insert_item(conn, **fields):
    """The one INSERT and the one id-minting site, shared by both routes.
    Every column is named by the caller: a missing field is a KeyError here,
    not a silent NULL in the row. The caller holds the write lock and
    commits."""
    item_id = 'th_%s' % uuid.uuid4().hex[:8]
    # updated_at is stamped HERE and in _touch_where — the two write helpers
    # — so no caller carries the clock: a birth is the first update.
    row = dict(fields, id=item_id, answer='', answered_at=None,
               updated_at=fields['created_at'])
    conn.execute(
        'INSERT INTO thalamus_items (%s) VALUES (%s)'
        % (', '.join(_INSERT_COLS), ', '.join('?' * len(_INSERT_COLS))),
        tuple(row[col] for col in _INSERT_COLS))
    return item_id


# Columns a caller may never pass to _touch_where: the row's identity, the
# two clocks the helpers stamp themselves, and the epoch (bump_epoch is the
# only way it moves). Passing `updated_at=` would silently lose to the stamp.
_TOUCH_OWNED = {'id', 'created_at', 'updated_at', 'armed_epoch'}


def _touch_where(conn, where, params, now, bump_epoch=False, **cols):
    """The one UPDATE shape: set `cols` on every row matching `where`, stamp
    updated_at=now, optionally bump the delivery generation (armed_epoch — a
    re-arm, see file()/resolve). Every state change, re-file and sweep goes
    through here or _touch, so updated_at can never be forgotten at a new
    write site (the producer view's settled-recently window reads it). A
    column outside _ITEM_COLS is a programmer error and raises before SQL
    does. The caller holds the write lock and commits. Returns rowcount."""
    bad = set(cols) - (set(_ITEM_COLS) - _TOUCH_OWNED)
    if bad:
        raise ValueError('thalamus._touch_where: column(s) %s are unknown or '
                         'owned by the helper (identity, clocks, epoch)'
                         % sorted(bad))
    sets = ['%s = ?' % c for c in cols] + ['updated_at = ?']
    if bump_epoch:
        sets.append('armed_epoch = armed_epoch + 1')
    cur = conn.execute(
        'UPDATE thalamus_items SET %s WHERE %s' % (', '.join(sets), where),
        list(cols.values()) + [now] + list(params))
    return cur.rowcount or 0


def _touch(conn, item_id, now, bump_epoch=False, **cols):
    """Per-item form of _touch_where."""
    return _touch_where(conn, 'id = ?', [item_id], now, bump_epoch, **cols)


def file(brain, source, body, *, needs_answer=False, when=None, for_whom=None,
         refs=None, dedup_key=None, expires=None, session_id='',
         run_chain=''):
    """The single producer door: validate → resolve the grammars → route.

    Returns {'ok': True, 'id': …} or a LOUD synchronous rejection
    {'ok': False, 'error': <guidance>} — every guard fires HERE, at the write
    boundary, where the rejection lands in the caller's loop while it can
    still adapt (never in a sweeper hours later).

    Tracing: a producer filing from a RUN passes its chain (`run_chain`; the
    scale is the chain's) and every ACCEPTED filing leaves one
    `thalamus_filed` row on that chain through the traces door
    (brain.write_thalamus_filed) — the filed→delivered→answered join across
    scales. No chain, no row: an interactive `remind` call is already in the
    caller's own tool trail.

    Routing: for_whom 'live' delegates to the courier broadcast (requires a
    filing session — the locked session-speech render must stay honest) and the
    row goes terminal 'sent'. Everything else queues for pull delivery.

    Identity: a repeat (source, dedup_key, target_session) UPDATES the open
    item instead of inserting — identity is producer-owned or absent, never
    derived from text. The target is part of it because one producer string
    (the Scribe's) serves every session, and a key re-used for another
    session is another item, not a retarget of the first.
    Both forms return updated=True; they differ in `rearmed`. A repeat that
    CHANGES any producer-controlled delivery attribute — body, when,
    needs_answer, for_whom — rewrites the item to what this call describes
    and re-arms delivery (rearmed=True), so an edited item delivers again to
    sessions that already saw the old one. An identical repeat only refreshes
    the window (rearmed=False) — a cyclic producer re-asserting its standing
    item must never re-notify every session on every cycle. `refs` refresh
    on every repeat but never re-arm: the anchor is derived (a subject's
    ids, a model-supplied ref) and can vary run to run without the question
    changing.
    """
    body = (body or '').strip()
    if not body:
        return _reject('thalamus.file: empty body')
    # `source` is the withdraw-ownership key — a typo'd one orphans its own
    # items, so it is vocabulary (the repo-wide encoding_source grammar), not
    # free text.
    from servers.contract import validate_encoding_source
    ok, err = validate_encoding_source(source)
    if not ok:
        return _reject('thalamus.file: %s' % err)

    try:
        route, audience, target_session = tc.resolve_for_whom(
            for_whom, needs_answer)
        deliver_at = tc.resolve_when(when)
        expires_at = (tc.resolve_when(expires) if expires
                      else tc.window_for(needs_answer, deliver_at))
    except ValueError as e:
        return _reject(str(e))
    if route == 'queue' and audience not in tc.AUDIENCES:
        # Internal drift guard at the write boundary: an audience outside the
        # closed set matches neither pull-predicate branch — the row would
        # stay open forever and die silently at expiry.
        return _reject('thalamus.file: internal audience %r is not one of %s'
                       % (audience, tc.AUDIENCES))
    now = iso_now()
    if expires and (not expires_at or expires_at <= max(deliver_at or '', now)):
        # window_for anchors the DEFAULT window at deliver_at exactly to
        # prevent expiry-before-due; an explicit `expires` must meet the same
        # bar — after `when`, and after now ('now' itself names no instant) —
        # or the item never becomes due (a false loud dead-letter for asks).
        return _reject(
            'thalamus.file: expires=%r (%s) is not after when (%s) and now — '
            'the item would expire before it ever becomes due'
            % (expires, expires_at, deliver_at or 'now'))
    refs_json = json.dumps(list(refs or []))

    if route == 'live':
        # A queue-shaped param on a live send is an intent the route cannot
        # honor — reject loudly rather than silently dropping it. Routing
        # validation stays at the door; the branch only stores.
        if deliver_at or needs_answer or dedup_key or expires:
            return _reject(
                "thalamus.file: for_whom='live' is fire-and-forget — it "
                "cannot honor when/needs_answer/dedup_key/expires; drop "
                "them, or queue instead (for_whom='all' or omit)")
        if not session_id:
            return _reject(
                "thalamus.file: for_whom='live' requires a filing session "
                "(the session-speech render needs an honest sender); "
                "machine live-now is Phase 3")
        result = _file_live(brain, source, body, refs, refs_json, session_id,
                            now)
    else:
        result = _file_queued(brain, source, body, refs, refs_json, now,
                              audience=audience, target_session=target_session,
                              needs_answer=needs_answer, dedup_key=dedup_key,
                              deliver_at=deliver_at, expires_at=expires_at)
    if run_chain and result.get('ok'):
        # The door owns its result envelope, so it names what it did — the
        # traces layer never learns the envelope's shape.
        filing = ('rearm' if result.get('rearmed') else
                  'refresh' if result.get('updated') else 'new')
        brain.write_thalamus_filed(
            chain_id=run_chain, session_id=session_id, item_id=result['id'],
            source=source, body=body, target_session=target_session,
            needs_answer=needs_answer, dedup_key=dedup_key or '',
            route=result['route'], filing=filing)
    return result


def _file_live(brain, source, body, refs, refs_json, session_id, now):
    """Route 'live' — fire-and-forget to whoever is alive NOW, the courier's
    contract. The terminal 'sent' row is kept for observability
    (thalamus_list) and has no ledger lifecycle: the courier owns delivery and
    death (1h TTL). Phase 3's machine live-now rewrites exactly this."""
    from servers.channels.self_channel import signal, self_contract
    try:
        sent = signal.send(brain, from_session=session_id,
                           address=self_contract.ADDR_BROADCAST,
                           body=body, refs=refs)
    except ValueError as e:  # the courier's door refused the body
        return _reject(str(e))
    with brain.logs_write_lock:
        item_id = _insert_item(
            brain.logs_conn_w, source=source, body=body, refs=refs_json,
            audience='', target_session='', needs_answer=0, dedup_key='',
            deliver_at=None, expires_at=sent['expires_at'],
            state=tc.STATE_SENT, created_at=now)
        brain.logs_conn_w.commit()
    return _ok(id=item_id, route='live', courier_id=sent['id'])


def _file_queued(brain, source, body, refs, refs_json, now, *, audience,
                 target_session, needs_answer, dedup_key, deliver_at,
                 expires_at):
    """Route 'queue' — the durable path, in order: dedup upsert, re-ask gate,
    insert. The door owns the identity contract; the epoch mechanics
    are below."""
    # Refresh the anchor only when this call carries one. An omitted refs
    # means "this run derived none", never "clear it" — no producer can
    # express clearing, and a silent wipe would undo the very anchor the
    # item exists to carry.
    anchor = {'refs': refs_json} if refs else {}
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        if dedup_key:
            # Unbound one-step lookup + LIMIT 1 — the dal_logs.py write-conn
            # invariant: a bound, unexhausted cursor on logs_conn_w holds a
            # read snapshot and the next write fails INSTANTLY on a concurrent
            # commit (SQLITE_BUSY_SNAPSHOT, busy_timeout bypassed).
            existing = conn.execute(
                'SELECT id, body, deliver_at, needs_answer, audience,'
                ' target_session FROM thalamus_items '
                'WHERE source = ? AND dedup_key = ? AND target_session = ?'
                ' AND state = ? LIMIT 1',
                (source, dedup_key, target_session, tc.STATE_OPEN)).fetchone()
            if existing:
                # The gate compares EVERY producer-controlled delivery
                # attribute, and the UPDATE sets each one it compares: a
                # re-file leaves the row in the state the producer just
                # described, or it is a no-op. Comparing a subset silently
                # keeps stale attributes — a note re-filed as an ask kept
                # needs_answer=0 (still a note, still Stop-delivered, never
                # dead-lettering) while taking the ask's longer window.
                # A CHANGED re-file is a re-arm: bump the generation (the
                # same mechanism as resolve's defer) so an item already
                # delivered at the old epoch delivers again with the updated
                # content — without the bump the current-epoch ledger row
                # suppresses the update forever while file() reports success.
                # An UNCHANGED re-file (a cyclic producer re-asserting its
                # standing item) is idempotent: it refreshes the window and
                # nothing else — bumping on a no-op would re-deliver the same
                # text to every session on every producer cycle, unbounded.
                # `refs` sit outside the gate and refresh on both paths: the
                # anchor is derived and may vary run to run while the
                # question stands — re-arming on it would be that unbounded
                # re-delivery by another door.
                needs_answer_col = 1 if needs_answer else 0
                changed = tuple(existing[1:]) != (
                    body, deliver_at, needs_answer_col, audience,
                    target_session)
                if changed:
                    _touch(conn, existing[0], now, bump_epoch=True,
                           body=body, deliver_at=deliver_at, **anchor,
                           expires_at=expires_at, needs_answer=needs_answer_col,
                           audience=audience, target_session=target_session)
                else:
                    _touch(conn, existing[0], now, expires_at=expires_at,
                           **anchor)
                conn.commit()
                return _ok(id=existing[0], updated=True, rearmed=changed,
                           route='queue')

            # No OPEN item under this identity — but an ASK that EXPIRED
            # still binds (tc.REASK_GATED_STATES). Expiry frees the dedup
            # key, so a cyclic producer's next run would insert a fresh item
            # with a fresh window: the treadmill both the cap and the window
            # exist to end. ASKS only, read off the stored row: for a notice
            # or a dated reminder, re-filing after the window IS the re-arm
            # the producer exists to do, and gating it refused the next cycle
            # of a recurring reminder — nobody's question was being repeated.
            expired = conn.execute(
                'SELECT id, updated_at FROM thalamus_items '
                'WHERE source = ? AND dedup_key = ? AND target_session = ?'
                ' AND needs_answer = 1 AND state IN (%s) AND updated_at > ?'
                ' ORDER BY updated_at DESC LIMIT 1'
                % ', '.join('?' * len(tc.REASK_GATED_STATES)),
                (source, dedup_key, target_session, *tc.REASK_GATED_STATES,
                 iso_cutoff(days=tc.REASK_COOLOFF_DAYS))).fetchone()
            if expired:
                return _reject(
                    'thalamus re-ask: %s (%r) ran its whole course on %s '
                    'with no answer — re-filing the same identity now '
                    'restarts its delivery cycle; wait out the %d-day '
                    'cool-off, or use a different dedup_key for a different '
                    'question' % (expired[0], dedup_key,
                                  (expired[1] or '')[:10],
                                  tc.REASK_COOLOFF_DAYS))

        item_id = _insert_item(
            conn, source=source, body=body, refs=refs_json, audience=audience,
            target_session=target_session,
            needs_answer=1 if needs_answer else 0, dedup_key=dedup_key or '',
            deliver_at=deliver_at, expires_at=expires_at,
            state=tc.STATE_OPEN, created_at=now)
        conn.commit()
    return _ok(id=item_id, route='queue', audience=audience,
               deliver_at=deliver_at, expires_at=expires_at)


# The ledger rows of one item, and of its CURRENT generation — the one
# spelling behind every reader of "delivered" / "delivered at this epoch":
# the due predicate, the saturation predicate, the pull order and the audit
# counts (the signal._PENDING_INBOX_SQL pattern).
_ITEM_DELIVERIES = (' FROM thalamus_deliveries d'
                    ' WHERE d.item_id = thalamus_items.id')
_EPOCH_DELIVERIES = (_ITEM_DELIVERIES
                     + ' AND d.armed_epoch = thalamus_items.armed_epoch')
_EPOCH_DELIVERY_COUNT = '(SELECT COUNT(*)%s)' % _EPOCH_DELIVERIES
_ITEM_DELIVERY_COUNT = '(SELECT COUNT(*)%s)' % _ITEM_DELIVERIES
# Deliveries since the item's current due date — the BUDGET's span. A defer
# is the reader saying "ask me again later", and it already leaves its mark
# on the row by moving deliver_at: counting from there, the sessions that
# passed before the deferral stop counting against it, with no second record
# of who re-armed. An item that was never deferred has deliver_at NULL (or a
# past due date every delivery follows), so this counts its whole life.
_BUDGET_DELIVERIES = (_ITEM_DELIVERIES
                      + " AND d.delivered_at > COALESCE(thalamus_items.deliver_at, '')")
_BUDGET_COUNT = '(SELECT COUNT(*)%s)' % _BUDGET_DELIVERIES
# The two axes, and why each reads a different span of the same ledger:
# ELIGIBILITY is per-epoch — a re-arm is a new generation, so the sessions
# that saw the old text become eligible again, WHILE BUDGET REMAINS. The
# BUDGET spans every delivery since the item last came due; it belongs to
# the identity, and only the reader extends it. A re-armed item whose budget
# is spent is saturated: it delivers to no one and the sweep closes it.
# Saturated: an every_session item that `cap` sessions saw since it last
# came due. Params: [tc.AUDIENCE_EVERY, cap]. Delivery stops here at once
# (the due predicate); the CLOSE waits — see expire_due.
_SATURATED = '(audience = ? AND %s >= ?)' % _BUDGET_COUNT
# Settled: no delivery at all since `cutoff`. Param: [cutoff]. The delivery
# that reaches the cap is by definition one a session has just been handed,
# so the sweep needs this before it may close what someone is still holding —
# a window end has no reader, saturation always does. Spans EVERY delivery,
# deliberately wider than the budget: the guard asks whether a session is
# holding this right now, and neither a re-file nor a defer may answer that
# by emptying the span it is read over.
_SETTLED = 'NOT EXISTS (SELECT 1%s AND d.delivered_at > ?)' % _ITEM_DELIVERIES


def _delivery_cap(brain):
    """tc.MAX_DELIVERIES_PER_ITEM, overridable live through the config key.

    Clamped to >= 1 because the cap drives a TERMINAL transition: a 0, a
    negative or a typo would otherwise expire every standing item at zero
    deliveries. An unusable value is LOUD and the code default stands."""
    raw = tc.MAX_DELIVERIES_PER_ITEM
    try:
        raw = brain.get_config(tc.MAX_DELIVERIES_CONFIG_KEY,
                               tc.MAX_DELIVERIES_PER_ITEM)
        cap = int(raw)
        if cap < 1:
            raise ValueError('a delivery cap is a session count >= 1')
    except Exception as e:
        brain._log_error(
            'thalamus_cap_unusable', e,
            'thalamus._delivery_cap: %s=%r ignored — using the code default %d'
            % (tc.MAX_DELIVERIES_CONFIG_KEY, raw,
               tc.MAX_DELIVERIES_PER_ITEM))
        return tc.MAX_DELIVERIES_PER_ITEM
    return cap


def _due_filter(session_id, via, now, cap):
    """The due predicate — WHERE clause + params, shared by pull()'s fetch and
    its overflow count so the two cannot drift (the signal._PENDING_INBOX_SQL
    pattern). Asks (needs_answer) are due at their audience's
    tc.ASK_MOMENTS only; a saturated item (_SATURATED) is due to no one."""
    sql = (' WHERE state = ?'
           ' AND (deliver_at IS NULL OR deliver_at <= ?)'
           ' AND expires_at > ?'
           " AND (target_session = '' OR target_session = ?)")
    params = [tc.STATE_OPEN, now, now, session_id]
    for audience, moments in tc.ASK_MOMENTS.items():
        if via not in moments:
            sql += ' AND NOT (needs_answer = 1 AND audience = ?)'
            params.append(audience)
    # The ledger is append-only across re-arms: only CURRENT-epoch rows
    # block delivery — a defer bumps the item's armed_epoch, so prior
    # generations' deliveries stay as history without suppressing the re-arm.
    sql += (' AND ((audience = ? AND NOT EXISTS (SELECT 1%s))'
            '  OR (audience = ? AND NOT EXISTS (SELECT 1%s AND d.session_id = ?)))'
            ' AND NOT %s'
            % (_EPOCH_DELIVERIES, _EPOCH_DELIVERIES, _SATURATED))
    params.extend([tc.AUDIENCE_FIRST, tc.AUDIENCE_EVERY, session_id,
                   tc.AUDIENCE_EVERY, cap])
    return sql, params


def _attach_ref_lines(brain, items, session_id):
    """Resolve every item's refs in ONE veil-aware batch and attach
    'ref_lines' for the render — the contract only FORMATS (its own rule);
    pull() holds both `brain` and `session_id`, so resolution lives here.
    Routes through filter_nodes (the existing veil door, default-deny): a
    globally-filed item can ref a walled node, and its title must not print
    into another session's boot — an unreturned ref (walled, retired, or a
    bad id) renders bare. A ref whose node was ABSORBED rides filter_nodes'
    canonical redirect: the survivor comes back stamped `_redirected_from`
    and renders as `old ↦ new · Title (absorbed)`.
    Failure-isolated and LOUD: refs left untitled by a
    failed batch render bare AND the failure is logged — bare ids are also
    the normal walled/bad-id output, so an unlogged dead resolution path
    would be invisible forever."""
    want = []
    for item in items:
        for ref in (item['refs'] or [])[:tc.RENDER_REFS_MAX]:
            if ref not in want:
                want.append(ref)
    titles = {}
    redirects = {}  # stored ref id -> (survivor id, survivor title)
    if want:
        try:
            res = brain.filter_nodes(
                field='id', include=want, rich=False,
                session_id=session_id, limit=len(want))
            from servers.contract import REDIRECTED_FROM_KEY
            for n in res.get('nodes') or []:
                titles[n['id']] = n.get('title') or ''
                # Canonical-pull redirect: a stored ref whose node was
                # absorbed resolves to its live survivor — render both ids
                # so the consumer knows the ref migrated.
                for src in n.get(REDIRECTED_FROM_KEY) or ():
                    redirects[src] = (n['id'], n.get('title') or '')
        except Exception as e:
            brain._log_error('thalamus_ref_resolve', e,
                             'pull ref batch failed — refs render bare')
    for item in items:
        refs = item['refs'] or []
        lines = []
        for ref in refs[:tc.RENDER_REFS_MAX]:
            if ref in redirects:
                from servers.contract import redirect_ref_line
                sid, title = redirects[ref]
                lines.append(redirect_ref_line(ref, sid, title))
                continue
            title = titles.get(ref)
            lines.append('%s · %s' % (ref[:8], title) if title else ref[:8])
        extra = len(refs) - tc.RENDER_REFS_MAX
        if extra > 0:
            lines.append('(+%d more refs — get_nodes)' % extra)
        item['ref_lines'] = lines


def pull(brain, session_id, via):
    """Due items for this session at a delivery moment ('prompt' | 'stop') —
    rendered block + count of items actually shown. Writes the ledger at
    render for exactly those items (INSERT OR IGNORE: the PK makes re-render
    idempotent per session and epoch). Asks (needs_answer) deliver at their
    audience's tc.ASK_MOMENTS — broadcast at the first prompt only, directed at Stop.

    The caller owns tracing (channels/delivery.py holds the chain at both
    moments); the ledger here is the delivery-policy record. Two sessions
    racing a once-item can each deliver it — rare and harmless (a reminder
    seen twice beats one lost); the ledger stays truthful about both.
    """
    if via not in tc.MOMENTS:
        # via is written to the ledger verbatim — vocabulary, not free text.
        # A typo'd moment would behave as Stop (asks silently withheld) and
        # ledger the typo.
        raise ValueError('thalamus.pull: via=%r is not a delivery moment %s'
                         % (via, tc.MOMENTS))
    if not session_id:
        return '', 0
    now = iso_now()
    where, params = _due_filter(session_id, via, now, _delivery_cap(brain))
    # True due count first, so the render's overflow tail names the real
    # number — a "+1" hiding fifteen is the wallpaper defect one level down.
    # Predicate shared with the item fetch below via _due_filter, so the two
    # cannot drift.
    total = brain.logs_conn.execute(
        'SELECT COUNT(*) FROM thalamus_items' + where, params).fetchone()[0]
    if not total:
        return '', 0
    # Asks first, then fewest deliveries this epoch, then age — with the
    # count taken RELATIVE TO ITS OWN AUDIENCE's floor, so a number is never
    # compared across classes. It cannot be: for an every_session item the
    # count measures coverage against a whole window of sessions, for a
    # first_session item one delivery is the entire contract and the count is
    # 0 for as long as it is due. Compared directly, one class starves
    # whichever way the key runs; against its own floor, each class's
    # least-delivered item ties at 0 and age breaks the tie, so standing
    # items rotate among themselves without sinking below the one-shots.
    # Both directions are pinned in tests/test_thalamus.py.
    rows = brain.logs_conn.execute(
        'SELECT %s FROM thalamus_items%s'
        ' ORDER BY needs_answer DESC,'
        ' (%s - MIN(%s) OVER (PARTITION BY audience)) ASC,'
        ' COALESCE(deliver_at, created_at) ASC'
        ' LIMIT ?' % (_ITEM_SELECT, where, _EPOCH_DELIVERY_COUNT,
                      _EPOCH_DELIVERY_COUNT),
        params + [tc.PULL_MAX_ITEMS]).fetchall()
    overflow = total - len(rows)
    items = [_row_to_item(r) for r in rows]
    _attach_ref_lines(brain, items, session_id)
    # Render FIRST, then ledger only what the block actually shows — a
    # cap-dropped item was never delivered, and a ledger row would suppress
    # it forever; unledgered, it stays armed for the next moment.
    block, kept = tc.render_block(
        items, overflow=overflow,
        counterpart=brain.counterpart_for(session_id))
    delivered_at = iso_now()
    with brain.logs_write_lock:
        brain.logs_conn_w.executemany(
            'INSERT OR IGNORE INTO thalamus_deliveries '
            '(item_id, session_id, delivered_at, via, armed_epoch) '
            'VALUES (?, ?, ?, ?, ?)',
            [(i['id'], session_id, delivered_at, via, i['armed_epoch'])
             for i in items[:kept]])
        brain.logs_conn_w.commit()
    return block, kept


def producer_items(brain, source, target_session, settled_days=None):
    """A producer's OWN items for one recipient — the binding's read behind
    the journal (open keys for withdraw; the producer view). Identity-exact:
    (source, target_session) with '' meaning the broadcast items. Every open
    item, plus — when `settled_days` is given — the items that reached a fate
    a producer hears about (tc.FATE_STATES) within that many days. Ordered
    open-first, then by when they last changed, so a consumer that cuts rows
    never cuts an open one and sees the freshest answer first. Carries the
    item's all-time delivery count — LIFETIME coverage, which is what the
    fate reports; the cap spends a narrower span (deliveries since the item
    last came due) and the per-generation split is the audit view's
    (list_items). Wall-clock over updated_at, a transaction-time column —
    the channel's documented exemption (thalamus_contract docstring)."""
    if settled_days is None:
        state_sql, params = 'state = ?', [tc.STATE_OPEN]
    else:
        fate_states = tuple(s for s in tc.FATE_STATES if s != tc.STATE_OPEN)
        state_sql = ('(state = ? OR (state IN (%s) AND updated_at >= ?))'
                     % ', '.join('?' * len(fate_states)))
        params = [tc.STATE_OPEN, *fate_states, iso_cutoff(days=settled_days)]
    rows = brain.logs_conn.execute(
        'SELECT %s, %s AS deliveries FROM thalamus_items'
        ' WHERE source = ? AND target_session = ?'
        ' AND %s ORDER BY (state <> ?), updated_at DESC'
        % (_ITEM_SELECT, _ITEM_DELIVERY_COUNT, state_sql),
        [source, target_session or '', *params, tc.STATE_OPEN]).fetchall()
    items = []
    for r in rows:
        item = _row_to_item(r)            # the aggregate falls off the zip
        item['deliveries'] = r[-1]
        items.append(item)
    return items


def list_items(brain, include_closed=False, limit=50, source='',
               target_session='', offset=0):
    """The audit view — open items (default) with their delivery counts in
    tc.LIST_ORDER_SQL's order, a page at a time: `offset` skips that many and `total`
    counts every match, so a reader can walk the whole list. include_closed
    adds terminal items; `source` / `target_session` narrow to one producer /
    one directed recipient."""
    clauses, params = [], []
    if not include_closed:
        clauses.append('state = ?')
        params.append(tc.STATE_OPEN)
    if source:
        clauses.append('source = ?')
        params.append(source)
    if target_session:
        clauses.append('target_session = ?')
        params.append(target_session)
    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    now = iso_now()
    total = brain.logs_conn.execute(
        'SELECT COUNT(*) FROM thalamus_items ' + where, params).fetchone()[0]
    rows = brain.logs_conn.execute(
        'SELECT %s,'
        ' (SELECT COUNT(*)%s) AS deliveries,'
        ' %s AS deliveries_epoch'
        ' FROM thalamus_items %s ORDER BY %s LIMIT ? OFFSET ?'
        % (_ITEM_SELECT, _ITEM_DELIVERIES, _EPOCH_DELIVERY_COUNT, where,
           tc.LIST_ORDER_SQL),
        params + [now, now, int(limit), int(offset)]).fetchall()
    items = []
    for r in rows:
        item = _row_to_item(r)            # the two aggregates fall off the zip
        item['deliveries'] = r[-2]        # all-time, across re-arms
        item['deliveries_epoch'] = r[-1]  # since the last re-arm
        items.append(item)
    return {'items': items, 'count': len(items), 'total': total}


def _get_item(conn, item_id):
    row = conn.execute(
        'SELECT %s FROM thalamus_items WHERE id = ?' % _ITEM_SELECT,
        (item_id,)).fetchone()
    return _row_to_item(row) if row else None


def resolve(brain, item_id, answer=None, defer_until=None, dismiss=False):
    """Anchor's exit verb — exactly one of answer / defer_until / dismiss.

    answer      → state 'answered'; the payload rides the item for the
                  producer's return surface (Phase 2 render-join).
    defer_until → deliver_at moves forward and the item RE-ARMS as a new
                  generation (armed_epoch bump — prior deliveries stay in the
                  ledger as history but stop blocking); the window stretches
                  to cover the new date.
    dismiss     → state 'dismissed' (closed without an answer).
    """
    actions = [a for a in (answer, defer_until, True if dismiss else None)
               if a is not None]
    if len(actions) != 1:
        return {'ok': False, 'error':
                'thalamus_resolve: pass exactly one of answer / defer_until / '
                'dismiss'}
    if answer is not None and not str(answer).strip():
        # Same guard file() puts on the body — an ask must not close on an
        # empty payload; the producer's return surface would render nothing.
        return {'ok': False, 'error':
                'thalamus_resolve: empty answer — pass real text, or dismiss'}
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        item = _get_item(conn, item_id)
        if not item:
            return {'ok': False, 'error': 'thalamus_resolve: no item %r'
                    % item_id}
        if item['state'] != tc.STATE_OPEN:
            return {'ok': False, 'error':
                    'thalamus_resolve: %s is already %r'
                    % (item_id, item['state'])}
        now = iso_now()
        if answer is not None:
            _touch(conn, item_id, now, state=tc.STATE_ANSWERED,
                   answer=str(answer), answered_at=now)
            result = {'ok': True, 'id': item_id, 'state': tc.STATE_ANSWERED}
        elif defer_until is not None:
            try:
                new_deliver = tc.resolve_when(defer_until)
            except ValueError as e:
                return {'ok': False, 'error': str(e)}
            if new_deliver is None:
                new_deliver = iso_now()
            new_expires = tc.extend_window(item['needs_answer'], new_deliver,
                                           item['expires_at'])
            # Re-arm is a GENERATION, not a deletion: bumping armed_epoch
            # makes the pull predicate ignore prior-epoch ledger rows, so the
            # item delivers again when due — while the ledger keeps truthful
            # history ("delivered, then deferred" ≠ "never delivered";
            # Phase 3 retry gates on unacked).
            _touch(conn, item_id, now, bump_epoch=True,
                   deliver_at=new_deliver, expires_at=new_expires)
            result = {'ok': True, 'id': item_id, 'state': tc.STATE_OPEN,
                      'deliver_at': new_deliver, 'expires_at': new_expires}
        else:
            _touch(conn, item_id, now, state=tc.STATE_DISMISSED)
            result = {'ok': True, 'id': item_id, 'state': tc.STATE_DISMISSED}
        conn.commit()
    return result


def withdraw(brain, source, item_id=None, dedup_key=None, target_session=''):
    """Producer retraction — a producer may close ITS OWN open item (by id, or
    by the (dedup_key, target_session) identity it filed under). Without
    this, a condition that resolved itself waits as a stale ask: the
    wallpaper defect one level up. Source must match."""
    if not (item_id or dedup_key):
        return {'ok': False, 'error':
                'thalamus.withdraw: pass item_id or dedup_key'}
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        if item_id:
            row = conn.execute(
                'SELECT source, state FROM thalamus_items WHERE id = ?',
                (item_id,)).fetchone()
        else:
            row = conn.execute(
                'SELECT source, state, id FROM thalamus_items WHERE '
                'source = ? AND dedup_key = ? AND target_session = ?'
                ' AND state = ?',
                (source, dedup_key, target_session or '',
                 tc.STATE_OPEN)).fetchone()
            item_id = row[2] if row else None
        if not row:
            return {'ok': False, 'error': 'thalamus.withdraw: no such item'}
        if row[0] != source:
            return {'ok': False, 'error':
                    'thalamus.withdraw: %s belongs to %r, not %r — only the '
                    'filing producer may withdraw' % (item_id, row[0], source)}
        if row[1] != tc.STATE_OPEN:
            return {'ok': False, 'error': 'thalamus.withdraw: %s is already %r'
                    % (item_id, row[1])}
        _touch(conn, item_id, iso_now(), state=tc.STATE_WITHDRAWN)
        conn.commit()
    return {'ok': True, 'id': item_id, 'state': tc.STATE_WITHDRAWN}


def expire_due(brain):
    """The window sweep (Brain.sweep_channels_if_due, hourly, ahead of the
    S2 fire gate). An item is done when its window ends, or when it is
    _SATURATED and _SETTLED — delivery stops at the cap, but the close waits
    for the session still holding the delivery that reached it.

    Two closes are LOUD to the errors table (which has a reader: query_logs,
    and the boot hook): an unanswered ASK, however it closed, so a question
    never dissolves silently; and any item the CAP closed, so a standing
    notice that every session scrolled past is not a silent write-off either
    — a producer filing something nobody ever acts on should hear that. A
    notice reaching its window end stays a natural death.

    The cause is named HERE, from the same cap this sweep selected on, and
    saturation wins a tie: "N sessions saw it and moved on" is the actionable
    half. tc.fate_of never re-derives it — it reports LIFETIME coverage, while
    the cap spends only what an item collected since it last came due, so both
    name their span and a deferred item's two numbers cannot read as a
    contradiction. Returns count expired."""
    now = iso_now()
    cap = _delivery_cap(brain)
    done = '(expires_at <= ? OR (%s AND %s))' % (_SATURATED, _SETTLED)
    done_params = [now, tc.AUDIENCE_EVERY, cap,
                   iso_cutoff(hours=tc.SATURATION_GRACE_HOURS)]
    with brain.logs_write_lock:
        conn = brain.logs_conn_w
        closing = conn.execute(
            'SELECT id, source, body, needs_answer, audience, %s'
            ' FROM thalamus_items WHERE state = ? AND %s'
            % (_BUDGET_COUNT, done),
            [tc.STATE_OPEN, *done_params]).fetchall()
        expired = _touch_where(conn, 'state = ? AND ' + done,
                               [tc.STATE_OPEN, *done_params], now,
                               state=tc.STATE_EXPIRED)
        conn.commit()
    for item_id, source, body, needs_answer, audience, seen in closing:
        saturated = audience == tc.AUDIENCE_EVERY and seen >= cap
        if needs_answer:
            why = ('%d sessions saw it since it last came due, none '
                   'answered' % seen if saturated else 'hit its window')
            brain._log_error(
                'thalamus_ask_expired',
                RuntimeError('ask %s from %s expired unanswered (%s): %.200s'
                             % (item_id, source, why, body)),
                'thalamus.expire_due — dead-letter (unanswered ask)')
        elif saturated:
            brain._log_error(
                'thalamus_saturated',
                RuntimeError('%s from %s spent its whole delivery budget: '
                             '%d sessions saw it since it last came due, none '
                             'acted: %.200s' % (item_id, source, seen, body)),
                'thalamus.expire_due — standing item closed on the cap')
    return expired
