"""Self-channel signal — the directed message (Phase 2a, pull courier).

The "hearing" half of self<->self: one stream sends a message addressed to
another live stream (self:<sid>) or to self:broadcast; the recipient consumes
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

    Stores the body and refs IN FULL — no truncation here. Per the self-channel
    truncation contract (self_contract), the SINGLE truncation point is delivery
    render (render_received_block), and it is loud; storage keeps everything so
    the dashboard always shows the message untruncated. The only guard is a
    non-empty body. A message addressed to the sender's own id carries
    `note_to_self: True`, so a mis-addressed reply is visible at send time."""
    body = (body or '').strip()
    if not body:
        raise ValueError('self.signal.send: empty body')
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
    if from_session and address == self_contract.address_for_stream(from_session):
        rec['note_to_self'] = True
    return rec


def _recent_sender_ids(brain):
    """Distinct senders of still-unexpired courier messages — the pool that lets
    you reply-by-short to a stream that messaged you even after it drops off the
    live roster (its from_session is still in the courier). Read-only."""
    rows = brain.logs_conn.execute(
        'SELECT DISTINCT from_session FROM self_inflight '
        'WHERE expires_at > ? AND from_session != ?',
        (iso_now(), '')).fetchall()
    return [r[0] for r in rows]


def resolve_stream(brain, ref, exclude_session=''):
    """Resolve a stream REFERENCE to its canonical FULL session id — the single
    resolver every "name a stream" caller routes through (self_send's target,
    self_peek, and any future fuzzy reference). Returns (full_id, None) on a unique
    match, (None, error) on ambiguity / no match. Errors name FULL ids so "use the
    full session id" is actionable.

    A full session UUID is canonical, honored directly (works even when the target
    isn't in the live roster this instant — it drains within TTL). Anything shorter
    is an id-PREFIX (the 8-char short you see in a message) matched against the live
    set: live roster ∪ recent courier senders. The roster alone misses a stream
    gone dormant since it messaged you, but its from_session lingers in the courier,
    so you can always reply-by-short to someone who reached you.

    Candidates are filtered to is_session_id-shaped ids — only a canonical full
    UUID is a resolvable stream. That filter (not a bare `if s`) is the structural
    guard against id contamination: a short / non-UUID `from_session` that leaked
    into the courier (db79e0c1 — now stopped at the write boundary by sender_id)
    simply never enters the pool, so it can't read as a phantom second match (the
    '37a32ee9 matches 2' bug) NOR silently merge into a different stream. No full
    UUID is a prefix of another (same length), so every surviving multi-match is
    GENUINE ambiguity; dict.fromkeys still collapses the same id arriving from both
    the roster AND the courier. `exclude_session` drops that stream from the
    prefix pool — self_peek passes the caller, so a prefix never resolves a peek
    to the stream doing the looking.

    NOT handled here: 'broadcast' (a delivery address, not a stream — resolve_to
    owns that). Fuzzy refs (a partial worktree/branch label) are the natural next
    match dimension — the candidate pool already carries those via the rich peek —
    and slot in right where the prefix match is, with the same ambiguity contract."""
    ref = (ref or '').strip()
    if not ref:
        return None, "empty stream reference"
    if self_contract.is_session_id(ref):
        return ref, None
    window = self_contract.ROSTER_LIVE_WINDOW_MIN + self_contract.ROSTER_LOST_GRACE_MIN
    candidates = [r.get('session_id', '')
                  for r in brain.present_streams(
                      exclude_session=exclude_session, window_min=window, limit=50)]
    candidates += _recent_sender_ids(brain)
    matches = list(dict.fromkeys(
        s for s in candidates
        if self_contract.is_session_id(s) and s != exclude_session and s.startswith(ref)))
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        # GENUINE ambiguity — distinct full ids sharing this prefix. Name the FULL
        # ids: the prefix is exactly what's ambiguous, so echoing it can't help.
        return None, ("'%s' matches %d live streams (%s) — use the full session id "
                      "to disambiguate" % (ref, len(matches), ", ".join(matches)))
    return None, ("no live stream matches '%s' — it may be dormant or lost. Use its "
                  "full session id, or self_presence to see who's live." % ref)


def resolve_to(brain, to):
    """Resolve an MCP self_send `to` to a delivery ADDRESS, gracefully — returns
    (address, error) with exactly one non-None. 'broadcast' is the only non-stream
    target; every other form is a stream reference handed to resolve_stream (full
    id / id-prefix) and wrapped as a directed address. The sender stays in the
    candidate pool: a stream may address its own id (a note to itself)."""
    to = (to or '').strip()
    if not to:
        return None, "self_send: empty 'to'"
    if to == 'broadcast':
        return self_contract.ADDR_BROADCAST, None
    full_id, error = resolve_stream(brain, to)
    if error:
        return None, "self_send: " + error
    return self_contract.address_for_stream(full_id), None


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
    shared by drain_inbox and peek_inbox. `own` marks a note this stream sent
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
    (any time). Drives first-contact enrichment: the first message from a stream
    carries its peek as intro, the rest stay lean. Read-only."""
    if not from_session:
        return False
    return conn.execute(
        'SELECT 1 FROM self_delivered d JOIN self_inflight i ON d.message_id = i.id '
        'WHERE d.to_session = ? AND i.from_session = ? LIMIT 1',
        (to_session, from_session)).fetchone() is not None


def drain_inbox(brain, to_session):
    """Consume-once: deliver undelivered, unexpired messages addressed to this
    stream (directed self:<id> + self:broadcast). Returns rendered messages,
    oldest first. Marking-delivered is atomic vs concurrent drains (write_lock +
    the self_delivered PK)."""
    if not to_session:
        return []
    now = iso_now()
    out = []
    with brain.logs_write_lock:
        # Everything in this envelope runs on logs_conn_w — the first-contact
        # check must see this transaction's own uncommitted inserts.
        rows = _pending_rows(brain.logs_conn_w, to_session, now)
        for mid, from_session, body, created_at, own in rows:
            # First contact = no PRIOR delivered message from this sender to me.
            # Checked BEFORE the insert below; within a multi-message batch the
            # same-transaction insert makes the 2nd+ from one sender non-first.
            # A note to self is your own words: no introduction.
            first_contact = not own and not _has_prior_delivery(
                brain.logs_conn_w, to_session, from_session)
            brain.logs_conn_w.execute(
                'INSERT OR IGNORE INTO self_delivered (message_id, to_session, delivered_at) '
                'VALUES (?, ?, ?)', (mid, to_session, now))
            short = (from_session or '')[:8]
            out.append({
                'id': mid,
                'from': short,
                'from_full': from_session or '',
                'body': body,
                'created_at': created_at,
                'first_contact': first_contact,
                'own': own,
                'rendered': self_contract.render_signal(body, stream_short=short, own=own),
            })
        brain.logs_conn_w.commit()
    return out


def peek_inbox(brain, to_session):
    """Read-only twin of drain_inbox: return pending (undelivered, unexpired)
    messages addressed to this stream WITHOUT consuming them.

    Shares the pending-inbox query with drain_inbox (_pending_rows) — same
    filter (directed + broadcast routes, unexpired, not your own broadcast, not
    already delivered) — but NO self_delivered write and NO write_lock. The
    /watch-live poller calls this every ~1.5s to detect arrivals; the real
    consume-once drain still happens in drain_inbox at the Stop hook."""
    if not to_session:
        return []
    now = iso_now()
    rows = _pending_rows(brain.logs_conn, to_session, now)
    return [{'id': mid, 'from': (from_session or '')[:8],
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
    """Delivery primitive — drain this stream's inbox and render the
    pending messages into one budgeted block. Returns (block, n_drained), or
    ("", 0) when empty. Consume-once is inherited from drain_inbox.

    The only caller is the courier's render adapter in channels/delivery.py,
    and only at the forcing Stop moment (`decision:block` — a turn cannot end
    on an unread tap): the courier declares `survives_a_miss=False` there, so
    `serves()` keeps this drain off every passive moment, where consume-once
    against a channel that can be missed would lose messages. The delivery
    leg owns tracing (it holds the chain)."""
    pending = drain_inbox(brain, to_session)
    if not pending:
        return "", 0
    # First contact from a stream → attach its peek so the intro carries context
    # (who, since when, what they're working on). Lazy import: presence imports
    # signal at module load, so importing presence here (call time, not load time)
    # keeps the enrichment in the delivery layer without a top-level import cycle.
    if any(m.get('first_contact') for m in pending):
        from servers.channels.self_channel import presence
        for m in pending:
            if m.get('first_contact') and m.get('from_full'):
                m['sender_peek'] = presence.peek(brain, m['from_full'])
    return self_contract.render_received_block(pending), len(pending)


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
    """Delivery status of messages THIS stream SENT — the sender-side view that
    until now only the dashboard had. Per recent in-flight message: which streams
    have drained (read) it and when, and — for a DIRECTED send — whether the named
    target is still pending. Closes the gap a sender hit when a reply never came:
    it tells "delivered, not acted on" apart from "never delivered". Read-only —
    pure SELECT, no write_lock.

    Broadcast has no fixed recipient set (any live stream may yet drain it), so
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
            'delivered_to': [{'to': (ts or '')[:8], 'at': at} for ts, at in delivered],
        }
        target = address.split(':', 1)[1] if ':' in address else address
        if target == 'broadcast':
            rec['broadcast'] = True
        else:
            rec['target'] = target[:8]
            rec['pending'] = not any((ts or '') == target for ts, _ in delivered)
        out.append(rec)
    return {'messages': out}
