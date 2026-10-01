"""Dispatch handlers for the self channel — presence (pull) + signal (reach).

Thin command surface over servers/channels/self_channel/{presence,signal}.py.
- presence (self_presence / self_peek): read-only look at other streams.
- signal   (self_send / self_inbox):    directed message + consume-once drain.
Signal's brain_logs.db writes go through brain.logs_write_lock inside signal.py;
none of these touch brain.db, so all register is_write=False.

Handler contract (shared by all daemon commands):
    handler(brain, args, graph_changes) -> {"ok": True, "result": <payload>}

Every handler MUST return the {"ok", "result"} envelope, like every other
dispatch_*.py handler. The table dispatch in daemon_server sends the return
verbatim — a raw, un-enveloped dict reaches the MCP client (brain_mcp) as a
falsy `ok` with no `error`, surfacing as the misleading "Unknown daemon error"
even though the handler succeeded. test_self_dispatch.py locks this.
"""

from servers.channels.self_channel import presence, signal
from servers.dispatch_common import caller_session, sender_id


def _handle_self_presence(brain, args, graph_changes):
    """Roster of streams of thought awake right now + the rendered presence line.

    args.session_id     = the caller's session (excluded from its own roster).
    args.limit          = optional cap (default PRESENCE_MAX_STREAMS).
    args.rich           = include per-stream detail (default True for this op —
                          interactive callers want to know WHO each stream is).
    args.active_streams = only reachable streams (default True).
    args.sort_by        = 'recency' (default) | 'length'.
    """
    return {"ok": True, "result": presence.build_presence(
        brain,
        my_session_id=caller_session(args),
        limit=args.get('limit'),
        rich=args.get('rich', True),
        active_streams=args.get('active_streams', True),
        sort_by=args.get('sort_by', 'recency'))}


def _handle_self_peek(brain, args, graph_changes):
    """Look into one stream of thought — its current focus. Read-only.

    args.stream_id = the TARGET stream to peek (distinct from the caller's
    session_id, so peeking never collides with the caller identity) — a full
    session id OR an id-prefix (the 8-char short you see in a message), resolved to
    the full id via signal.resolve_stream (the same resolver self_send's target
    uses). A prefix that doesn't resolve uniquely falls through to an empty
    (found:false) peek — a peek is a glance, not a delivery, so a miss is "nobody
    there", never a hard error.
    """
    ref = args.get('stream_id', '') or ''
    full_id, _ = signal.resolve_stream(brain, ref, exclude_session=caller_session(args))
    return {"ok": True, "result": presence.peek(brain, full_id or ref)}


def _handle_self_send(brain, args, graph_changes):
    """Send a directed/broadcast self-message into the courier — the deliberate reach.

    args.to           = target: id-prefix, full session id, or 'broadcast'.
    args.body         = the message.
    args.from_session = OPTIONAL attribution override; the proxy-stamped caller id
                        wins over it (see sender_id) so a short can't be stored as
                        the sender — honored only headless.
    args.refs         = optional.

    `to` resolves gracefully (signal.resolve_to): canonical id / broadcast pass
    through; the 8-char short (an id-prefix) matches the live roster; ambiguous or
    no match is a LOUD error so silence is never mistaken for delivery. The
    caller's own id is a valid target — a note to itself, delivered like any other.
    """
    sid = sender_id(args)
    address, error = signal.resolve_to(brain, args.get('to', '') or '')
    if error:
        return {"ok": False, "error": error}
    return {"ok": True, "result": signal.send(
        brain,
        from_session=sid,
        address=address,
        body=args.get('body', '') or '',
        refs=args.get('refs'))}


def _handle_self_inbox(brain, args, graph_changes):
    """Drain the caller's inbox — consume-once pending self-messages.

    args.session_id = the caller's own session, to fetch messages addressed to it.
    """
    return {"ok": True, "result": {'messages': signal.drain_inbox(brain, to_session=caller_session(args))}}


def _handle_self_inbox_peek(brain, args, graph_changes):
    """Read-only inbox peek — pending self-messages WITHOUT consuming them.

    Powers the /watch-live poller's arrival detection. The consume-once drain
    stays in self_inbox (the Stop hook). args.session_id = the caller's session.
    """
    return {"ok": True, "result": {'messages': signal.peek_inbox(brain, to_session=caller_session(args))}}


def _handle_self_outbox(brain, args, graph_changes):
    """Delivery status of the caller's SENT messages — who's drained each, and
    whether a directed target is still pending. Read-only (sender-side receipt).

    args.from_session = OPTIONAL; the proxy-stamped caller id wins (see sender_id),
                        so you read YOUR OWN outbox — honored only headless.
    args.limit        = optional cap (default 20).
    """
    return {"ok": True, "result": signal.outbox(
        brain,
        from_session=sender_id(args),
        limit=args.get('limit', 20))}
