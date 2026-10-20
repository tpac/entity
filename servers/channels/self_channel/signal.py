"""Self-channel signal — the directed message (Phase 2a, pull courier).

The "hearing" half of self<->self: one session sends a message addressed to
another live session (self:<sid>) or to self:broadcast; the recipient consumes
it once. Presence is the passive look; a signal is a deliberate reach — Anchor
authors it, unlike the encoder-written letter.

Storage: the `self_inflight` courier in brain_logs.db (schema.LOG_TABLES), with
consumption tracked in `self_delivered` (one row per (message, recipient) — so
broadcast fans out and each recipient consumes exactly once). Phase 2a is PULL:
a recipient drains its inbox on demand. Phase 2b wires auto-delivery into
Observation at a hook (traced as the s0 `self_message` marker).

Writes go through `brain.logs_write_lock` around `brain.logs_conn_w` — the
dedicated logs write connection (see dal_logs.py: the shared read connection's
open cursors poison write snapshots; writes on it get dropped by
SQLITE_BUSY_SNAPSHOT). Reads inside a write envelope also use `logs_conn_w`
so same-transaction visibility holds; reads outside envelopes stay on the
shared `logs_conn`. TTL is
per-message: send() resolves it by ADDRESS (broadcast vs directed, config-
tunable) and stamps `expires_at` via the clock contract (iso_now/iso_after);
readers filter `expires_at > now` and the reaper deletes `expires_at <= now`.
"""
import json
import uuid

from servers.clock import iso_now, iso_after
from servers.trace_contract import session_short
from servers.channels.self_channel import self_contract


def _resolve_ttl_hours(brain, address):
    """Per-message TTL in hours by address, with runtime config override.
    Defaults are documented in self_contract (BROADCAST/DIRECTED_TTL_HOURS);
    operators tune via brain.get_config('self_channel.{kind}_ttl_hours')."""
    kind = self_contract.ttl_kind_for(address)
    default = (self_contract.BROADCAST_TTL_HOURS if kind == 'broadcast'
               else self_contract.DIRECTED_TTL_HOURS)
    raw = brain.get_config('self_channel.%s_ttl_hours' % kind, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        # get_config returns the raw string when its numeric auto-parse fails (a
        # typo'd config value). Don't let one bad key crash every send — fall
        # back to the documented default, loudly.
        brain._log_error('self_ttl_config_invalid', ValueError(
            'self_channel.%s_ttl_hours=%r is not numeric; using default %s'
            % (kind, raw, default)), 'self-channel TTL resolution')
        return float(default)


def send(brain, from_session, address, body, refs=None):
    """Place a directed/broadcast self-message in the courier. Returns its record.

    The courier's one write door, for every sender (self_send, Thalamus's live
    route). Per the self-channel size contract (self_contract), a body over
    SIGNAL_BODY_MAX is refused here, loudly, and nothing is ever cut later:
    what is accepted is stored and delivered whole. An empty body is refused
    too. A message addressed to the sender's own id carries
    `note_to_self: True`, so a mis-addressed reply is visible at send time."""
    body = (body or '').strip()
    if not body:
        raise ValueError('self.signal.send: empty body')
    if len(body) > self_contract.SIGNAL_BODY_MAX:
        raise ValueError(
            'self.signal.send: the message is {:,} characters and the limit is '
            '{:,}. Shorten it, or put the detail in a file or a brain node and '
            'send its path or id.'.format(len(body), self_contract.SIGNAL_BODY_MAX))
    refs_json = json.dumps(list(refs or []))
    mid = uuid.uuid4().hex[:12]
    created_at = iso_now()
    # Per-message TTL: resolved by address (broadcast ephemeral, directed waits)
    # and stamped as a future expires_at. Readers filter on it; nothing
    # recomputes a cutoff. The sub-ms skew vs created_at is irrelevant (TTL is
    # hours) and nothing compares the two columns.
    expires_at = iso_after(hours=_resolve_ttl_hours(brain, address))
    # Serialize on logs_write_lock, write via the dedicated logs write conn.
    with brain.logs_write_lock:
        brain.logs_conn_w.execute(
            'INSERT INTO self_inflight '
            '(id, from_session, address, body, refs, created_at, expires_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (mid, from_session or '', address, body, refs_json,
             created_at, expires_at))
        brain.logs_conn_w.commit()
    rec = {'id': mid, 'address': address,
           'created_at': created_at, 'expires_at': expires_at}
    if from_session and address == self_contract.address_for_session(from_session):
        rec['note_to_self'] = True
    return rec


def _recent_sender_ids(brain):
    """Distinct senders of still-unexpired courier messages — the pool that lets
    you reply-by-short to a session that messaged you even after it drops off the
    live roster (its from_session is still in the courier). Read-only."""
    rows = brain.logs_conn.execute(
        'SELECT DISTINCT from_session FROM self_inflight '
        'WHERE expires_at > ? AND from_session != ?',
        (iso_now(), '')).fetchall()
    return [r[0] for r in rows]


def resolve_session(brain, ref, exclude_session='', pool='live'):
    """Resolve a session REFERENCE to its canonical FULL session id — the single
    resolver every "name a session" caller routes through (self_send's target,
    self_peek, remind's for_whom, and the session filters of query_traces and
    recall_episodes). Returns (full_id, None) on a unique
    match, (None, error) on ambiguity / no match. Errors name FULL ids so "use the
    full session id" is actionable.

    A session is named one of two ways: its full session UUID, honored directly
    (works even when the target isn't in the live roster this instant — it drains
    within TTL), or its session short (trace_contract.session_short), the one form
    every surface shows, matched EXACTLY — never a start, a partial tail or a
    guess, so a stale or mistyped reference fails loudly rather than landing on
    another session (Codex ids open with a timestamp, so same-minute chats share
    their start). Two distinct ids sharing a short (vanishingly rare) are refused
    as ambiguous.

    `pool` is where a short is looked for.

    'live' — addressing: a message or a reminder aims at a session that is here.
    The pool is the live roster ∪ recent courier senders. The roster alone misses
    a session gone dormant since it messaged you, but its from_session lingers in
    the courier, so you can always reply by short to someone who reached you.
    Candidates are filtered to is_session_id-shaped ids. That filter (not a bare
    `if s`) is the structural guard against id contamination: a short / non-UUID
    `from_session` that leaked into the courier (db79e0c1 — now stopped at the
    write boundary by sender_id) simply never enters the pool, so it can't read
    as a phantom second match (the '37a32ee9 matches 2' bug) NOR silently merge
    into a different session. dict.fromkeys collapses the same id arriving from
    both the roster AND the courier. `exclude_session` drops that session from
    the pool — self_peek passes the caller, so a short never resolves a peek to
    the session doing the looking.

    'history' — filtering: a trace filter may name any session ever recorded.
    The pool is every stored session id, UUID or not: trace history keeps eval
    and test sessions whose ids are not UUIDs, and trace headers show a short
    for those too. A reference names a stored id exactly or by its short; one
    that does both for different sessions is ambiguous like any other.

    NOT handled here: 'broadcast' (a delivery address, not a session — resolve_to
    owns that)."""
    if pool not in ('live', 'history'):
        raise ValueError("resolve_session: pool must be 'live' or 'history', got %r"
                         % (pool,))
    ref = (ref or '').strip()
    if not ref:
        return None, "empty session reference"
    if self_contract.is_session_id(ref):
        return ref, None
    if pool == 'history':
        matches = [s for s in brain.trace_session_ids()
                   if s != exclude_session and (s == ref or session_short(s) == ref)]
        kind = 'sessions in the trace history'
        missing = ("no session in the trace history matches '%s' — use the short "
                   "a trace header shows, or the session's full id." % ref)
    else:
        window = self_contract.ROSTER_LIVE_WINDOW_MIN + self_contract.ROSTER_LOST_GRACE_MIN
        candidates = [r.get('session_id', '')
                      for r in brain.present_sessions(
                          exclude_session=exclude_session, window_min=window, limit=50)]
        candidates += _recent_sender_ids(brain)
        matches = list(dict.fromkeys(
            s for s in candidates
            if self_contract.is_session_id(s) and s != exclude_session
            and session_short(s) == ref))
        kind = 'live sessions'
        missing = ("no live session matches '%s' — it may be dormant or lost. Use its "
                   "full session id, or self_presence to see who's live." % ref)
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        # Distinct ids sharing this reference. Name the FULL ids: the reference
        # is exactly what's ambiguous, so echoing it can't help.
        return None, ("'%s' matches %d %s (%s) — use the full session id "
                      "to disambiguate" % (ref, len(matches), kind, ", ".join(matches)))
    return None, missing


def resolve_to(brain, to):
    """Resolve an MCP self_send `to` to a delivery ADDRESS, gracefully — returns
    (address, error) with exactly one non-None. 'broadcast' is the only non-session
    target; every other form is a session reference handed to resolve_session (full
    id or session short) and wrapped as a directed address. The sender stays in the
    candidate pool: a session may address its own id (a note to itself)."""
    to = (to or '').strip()
    if not to:
        return None, "self_send: empty 'to'"
    if to == 'broadcast':
        return self_contract.ADDR_BROADCAST, None
    full_id, error = resolve_session(brain, to)
    if error:
        return None, "self_send: " + error
    return self_contract.address_for_session(full_id), None


# The pending-inbox SELECT — the SINGLE source for drain (consume-once) and peek
# (read-only). A row is pending for `to_session` when it's addressed to it
# (directed or broadcast), unexpired, not its own broadcast, and not already
# delivered. Only the broadcast route drops the sender: a directed message to
# your own id is a note to yourself and delivers like any other. drain wraps
# this in write_lock + a self_delivered stamp; peek just reads. One query means
# the two can't drift (they used to carry a "keep in lockstep" note — this
# removes the hazard).
_PENDING_INBOX_SQL = (
    'SELECT id, from_session, body, created_at FROM self_inflight '
    'WHERE address IN (?, ?) AND expires_at > ? '
    'AND NOT (address = ? AND from_session = ?) '  # not your own broadcast
    'AND id NOT IN (SELECT message_id FROM self_delivered WHERE to_session = ?) '
    'ORDER BY created_at')


def _pending_rows(conn, to_session, now):
    """Pending-inbox rows for `to_session` at time `now`, oldest first
    (id, from_session, body, created_at, own). Read-only; the single query
    shared by drain_inbox and peek_inbox. `own` marks a note this session sent
    itself — decided here once, so every surface that shows a message (the Stop
    drain, the /watch poller) attributes it the same way."""
    directed, broadcast = self_contract.routes_at_turn(to_session)
    rows = conn.execute(
        _PENDING_INBOX_SQL,
        (directed, broadcast, now, broadcast, to_session, to_session)).fetchall()
    return [(mid, frm, body, created_at, frm == to_session)
            for mid, frm, body, created_at in rows]


def _has_prior_delivery(conn, to_session, from_session):
    """True if `to_session` has ALREADY drained a message from `from_session`
    (any time). Drives first-contact enrichment: the first message from a session
    carries its peek as intro, the rest stay lean. Read-only."""
    if not from_session:
        return False
    return conn.execute(
        'SELECT 1 FROM self_delivered d JOIN self_inflight i ON d.message_id = i.id '
        'WHERE d.to_session = ? AND i.from_session = ? LIMIT 1',
        (to_session, from_session)).fetchone() is not None


def _pending_messages(conn, to_session, now):
    """The pending inbox as delivery records, oldest first — what both drains
    consume from. Read-only; the caller holds the write envelope and marks
    what it consumes with _mark_delivered."""
    out, seen = [], set()
    for mid, from_session, body, created_at, own in _pending_rows(conn, to_session, now):
        # First contact = no message from this sender delivered to me before,
        # and none earlier in this batch. A note to self is your own words:
        # no introduction.
        first_contact = (not own and from_session not in seen
                         and not _has_prior_delivery(conn, to_session, from_session))
        if from_session:
            seen.add(from_session)
        short = self_contract.sender_short(from_session)
        out.append({
            'id': mid,
            'from': short,
            'from_full': from_session or '',
            'body': body,
            'created_at': created_at,
            'first_contact': first_contact,
            'own': own,
            'rendered': self_contract.render_signal(body, short=short, own=own),
        })
    return out


def _mark_delivered(conn, to_session, messages, now):
    """Consume `messages` for `to_session` — the one write behind consume-once
    (the self_delivered PK makes a repeat a no-op)."""
    conn.executemany(
        'INSERT OR IGNORE INTO self_delivered (message_id, to_session, delivered_at) '
        'VALUES (?, ?, ?)', [(m['id'], to_session, now) for m in messages])


def drain_inbox(brain, to_session):
    """Consume-once: deliver undelivered, unexpired messages addressed to this
    session (directed self:<id> + self:broadcast). Returns all of them in full,
    oldest first. Marking-delivered is atomic vs concurrent drains (write_lock +
    the self_delivered PK)."""
    if not to_session:
        return []
    now = iso_now()
    with brain.logs_write_lock:
        # Reads and the consume share logs_conn_w, one envelope.
        out = _pending_messages(brain.logs_conn_w, to_session, now)
        _mark_delivered(brain.logs_conn_w, to_session, out, now)
        brain.logs_conn_w.commit()
    return out


def peek_inbox(brain, to_session):
    """Read-only twin of drain_inbox: return pending (undelivered, unexpired)
    messages addressed to this session WITHOUT consuming them.

    Shares the pending-inbox query with drain_inbox (_pending_rows) — same
    filter (directed + broadcast routes, unexpired, not your own broadcast, not
    already delivered) — but NO self_delivered write and NO write_lock. The
    /watch-live poller calls this every ~1.5s to detect arrivals; the real
    consume-once drain still happens in drain_inbox at the Stop hook."""
    if not to_session:
        return []
    now = iso_now()
    rows = _pending_rows(brain.logs_conn, to_session, now)
    return [{'id': mid, 'from': self_contract.sender_short(from_session),
             'body': body, 'created_at': created_at, 'own': own}
            for mid, from_session, body, created_at, own in rows]


def pending_count(brain, to_session):
    """How many messages wait in `to_session`'s inbox (undelivered, unexpired,
    addressed to it). Shares the pending-inbox filter with drain/peek via
    _pending_rows (no SQL duplicated). Read-only — for peek's reachability hint
    ('is my message joining a backlog?')."""
    if not to_session:
        return 0
    return len(_pending_rows(brain.logs_conn, to_session, iso_now()))


def drain_and_render(brain, to_session):
    """Delivery primitive — render this session's pending messages into one
    budgeted block and consume exactly the ones it shows. Returns
    (block, n_shown), or ("", 0) when empty. A message past the block budget
    is not consumed: it stays in the inbox, and the block's tail tells the
    reader to call self_inbox for it (or it arrives at the next Stop).

    The only caller is the courier's render adapter in channels/delivery.py,
    and only at the forcing Stop moment (`decision:block` — a turn cannot end
    on an unread tap): the courier declares `survives_a_miss=False` there, so
    `serves()` keeps this drain off every passive moment, where consume-once
    against a channel that can be missed would lose messages. The delivery
    leg owns tracing (it holds the chain)."""
    if not to_session:
        return "", 0
    now = iso_now()
    with brain.logs_write_lock:
        pending = _pending_messages(brain.logs_conn_w, to_session, now)
        if not pending:
            return "", 0
        # First contact from a session → attach its peek so the intro carries
        # context (who, since when, what they're working on). Lazy import:
        # presence imports signal at module load, so importing presence here
        # (call time, not load time) keeps the enrichment in the delivery layer
        # without a top-level import cycle.
        if any(m.get('first_contact') for m in pending):
            from servers.channels.self_channel import presence
            for m in pending:
                if m.get('first_contact') and m.get('from_full'):
                    m['sender_peek'] = presence.peek(brain, m['from_full'])
        block, shown = self_contract.render_received_block(pending)
        _mark_delivered(brain.logs_conn_w, to_session, pending[:shown], now)
        brain.logs_conn_w.commit()
    return block, shown


def reap_expired(brain):
    """Delete messages past their TTL (the dead-letter sweep) + orphan delivery
    rows. Returns count reaped. Wired into Brain.sweep_channels_if_due
    (hourly, ahead of the S2 fire gate); safe to call anytime."""
    now = iso_now()
    with brain.logs_write_lock:
        # expires_at <= now → dead. IS NULL → a pre-expires_at legacy row (column
        # added to the existing courier); send() always stamps it, so a NULL can
        # only be legacy — treat as dead and sweep it.
        cur = brain.logs_conn_w.execute(
            'DELETE FROM self_inflight WHERE expires_at <= ? OR expires_at IS NULL', (now,))
        reaped = cur.rowcount or 0
        brain.logs_conn_w.execute(
            'DELETE FROM self_delivered '
            'WHERE message_id NOT IN (SELECT id FROM self_inflight)')
        brain.logs_conn_w.commit()
    return reaped


def outbox(brain, from_session, limit=20):
    """Delivery status of messages THIS session SENT — the sender-side view that
    until now only the dashboard had. Per recent in-flight message: which sessions
    have drained (read) it and when, and — for a DIRECTED send — whether the named
    target is still pending. Closes the gap a sender hit when a reply never came:
    it tells "delivered, not acted on" apart from "never delivered". Read-only —
    pure SELECT, no write_lock.

    Broadcast has no fixed recipient set (any live session may yet drain it), so
    `pending` is reported only for a directed address; broadcasts just list who has
    drained so far."""
    if not from_session:
        return {'messages': []}
    now = iso_now()
    rows = brain.logs_conn.execute(
        'SELECT id, address, body, created_at FROM self_inflight '
        'WHERE from_session = ? AND expires_at > ? '
        'ORDER BY created_at DESC LIMIT ?',
        (from_session, now, limit)).fetchall()
    out = []
    for mid, address, body, created_at in rows:
        delivered = brain.logs_conn.execute(
            'SELECT to_session, delivered_at FROM self_delivered '
            'WHERE message_id = ? ORDER BY delivered_at', (mid,)).fetchall()
        rec = {
            'id': mid,
            'address': address,
            'created_at': created_at,
            'preview': (body or '')[:120] + (' …' if len(body or '') > 120 else ''),
            'delivered_to': [{'to': session_short(ts), 'at': at} for ts, at in delivered],
        }
        target = address.split(':', 1)[1] if ':' in address else address
        if target == 'broadcast':
            rec['broadcast'] = True
        else:
            rec['target'] = session_short(target)
            rec['pending'] = not any((ts or '') == target for ts, _ in delivered)
        out.append(rec)
    return {'messages': out}
