"""Dispatch handlers for the self channel — presence (pull) + signal (reach).

Thin command surface over servers/channels/self_channel/{presence,signal}.py.
- presence (self_presence / self_peek): read-only look at other sessions.
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


def _no_identity(cmd):
    """The refusal for a call that must know who is asking and was not told.

    An unattributed proxy call arrives with an empty stamp; answering it would
    read as a true answer about nobody ("no messages", "nothing sent") or
    store a message no one can reply to. The host-specific remedy rides on
    the proxy's side (brain_mcp's identity-gap note and the adapter notice)."""
    return {"ok": False,
            "error": "%s: this call carries no session identity, so there is no "
                     "caller to answer for (an MCP call the host gave no session "
                     "id, or a direct call naming none)" % cmd}


def _handle_self_presence(brain, args, graph_changes):
    """Roster of sessions awake right now + the rendered presence line.

    The caller (caller_session: the stamped identity, or session_id when
    headless) is excluded from its own roster.

    args.limit          = optional cap (default PRESENCE_MAX_SESSIONS).
    args.rich           = include per-session detail (default True for this op —
                          interactive callers want to know WHO each session is).
    args.active_sessions = only reachable sessions (default True).
    args.sort_by        = 'recency' (default) | 'length'.
    """
    return {"ok": True, "result": presence.build_presence(
        brain,
        my_session_id=caller_session(args),
        limit=args.get('limit'),
        rich=args.get('rich', True),
        active_sessions=args.get('active_sessions', True),
        sort_by=args.get('sort_by', 'recency'))}


def _handle_self_peek(brain, args, graph_changes):
    """Look into one session — its current focus. Read-only.

    args.session_id = the TARGET session to peek (the caller is the
    proxy-stamped _caller_session, never this arg — dispatch_common.
    caller_session) — a full session id OR its short (the one you see in a message), resolved to the full
    id via signal.resolve_session (the same resolver self_send's target uses). A
    short that doesn't resolve uniquely falls through to an empty (found:false)
    peek — a peek is a glance, not a delivery, so a miss is "nobody there",
    never a hard error.
    """
    ref = args.get('session_id', '') or ''
    full_id, _ = signal.resolve_session(brain, ref, exclude_session=caller_session(args))
    return {"ok": True, "result": presence.peek(brain, full_id or ref)}


def _handle_self_send(brain, args, graph_changes):
    """Send a directed/broadcast self-message into the courier — the deliberate reach.

    args.to           = target: short id, full session id, or 'broadcast'.
    args.body         = the message.
    args.from_session = OPTIONAL attribution override; the proxy-stamped caller id
                        wins over it (see sender_id) so a short can't be stored as
                        the sender — honored only headless.
    args.refs         = optional.

    `to` resolves gracefully (signal.resolve_to): canonical id / broadcast pass
    through; a short id matches the live roster; ambiguous or no match is a
    LOUD error so silence is never mistaken for delivery. The
    caller's own id is a valid target — a note to itself, delivered like any other.
    """
    sid = sender_id(args)
    if not sid:
        return _no_identity('self_send')
    address, error = signal.resolve_to(brain, args.get('to', '') or '')
    if error:
        return {"ok": False, "error": error}
    try:
        sent = signal.send(brain, from_session=sid, address=address,
                           body=args.get('body', '') or '', refs=args.get('refs'))
    except ValueError as e:  # the courier's door refused the body
        return {"ok": False, "error": str(e)}
    return {"ok": True, "result": sent}


def _handle_self_inbox(brain, args, graph_changes):
    """Drain the caller's inbox — consume-once pending self-messages.

    The caller is caller_session — the stamped identity, or session_id when
    headless; a model-written session_id never redirects the drain.
    """
    sid = caller_session(args)
    if not sid:
        return _no_identity('self_inbox')
    return {"ok": True, "result": {'messages': signal.drain_inbox(brain, to_session=sid)}}


def _handle_self_inbox_peek(brain, args, graph_changes):
    """Read-only inbox peek — pending self-messages WITHOUT consuming them.

    Powers the /watch-live poller's arrival detection; the consume-once drains
    are self_inbox and Stop-time delivery (signal.drain_and_render). The caller
    is caller_session — the poller runs headless and names itself with
    session_id.
    """
    sid = caller_session(args)
    if not sid:
        return _no_identity('self_inbox_peek')
    return {"ok": True, "result": {'messages': signal.peek_inbox(brain, to_session=sid)}}


def _handle_self_outbox(brain, args, graph_changes):
    """Delivery status of the caller's SENT messages — who's drained each, and
    whether a directed target is still pending. Read-only (sender-side receipt).

    args.from_session = OPTIONAL; the proxy-stamped caller id wins (see sender_id),
                        so you read YOUR OWN outbox — honored only headless.
    args.limit        = optional cap (default 20).
    """
    sid = sender_id(args)
    if not sid:
        return _no_identity('self_outbox')
    return {"ok": True, "result": signal.outbox(
        brain,
        from_session=sid,
        limit=args.get('limit', 20))}
