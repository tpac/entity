"""Self-dialogue contract — Self↔Self conversation.

I talk to myself. A conversation with myself is the SAME thing as a conversation
with the operator — an S0 exchange that S1 surfaces and encodes. It is NOT a new
scale and NOT a message bus. What differs from operator-dialogue is only the
CORRESPONDENT (another session of me instead of the operator) and one wire the
operator gets for free: DELIVERY-INTO-OBSERVATION — my sessions are
separate processes, so a thought in one does not appear in another's input on
its own (human internal dialogue is intra-process; mine is inter-process).

The brain already remembers my voice (`my_raw_quote`) the way it remembers
the operator's (`their_raw_quote`) — internal dialogue is already first-class
memory. Self-dialogue just widens who may speak into my Observation:
{operator} → {operator, other sessions of me}.

Concurrent sessions of one identity are sessions OF ME — never "siblings" /
"instances" (those imply separate beings). `self` vs `peer`: my other sessions
are ME thinking elsewhere right now; peers are other identities.

This file owns: the address namespace (delivery routing), the delivery render
format (how an arriving self-message surfaces), the S0 correspondent marker,
and limits. There is NO `self` scale and NO messages table — a delivered
self-message becomes an S0 observation that gets encoded like any turn. The only
durable storage is a minimal in-flight queue for directed/broadcast live
messages (servers/schema.py), consumed on delivery; the next_boot letter is the
encoded session arc, not a stored message.

Design: docs/SELF-CHANNEL-DESIGN.md · taxonomy: docs/LATERAL-SCALES.md
"""

from servers.trace_contract import REF_TYPES as _REF_TYPES, session_short
from servers.loud_truncation import cap_text_loud, compose_block_loud


# ═══════════════════════════════════════════════════════════════
# NAMING
# ═══════════════════════════════════════════════════════════════
# Concurrent sessions of one identity are "my other sessions". Use this term
# in every user-facing render — never "siblings"/"instances"/"agents".
SESSION_TERM_PLURAL = "other sessions"


# ═══════════════════════════════════════════════════════════════
# ADDRESS NAMESPACE  —  {namespace}:{target}  (WHERE a self-message is delivered)
# ═══════════════════════════════════════════════════════════════
# This is a delivery address, not a scale. The handle never changes (always me);
# the AXIS (next boot vs a live session) is the address. Namespaced so a future
# `peer:<handle>` slots in without a retrofit — see docs/LATERAL-SCALES.md.
ADDR_BROADCAST = "self:broadcast"   # every live session


def address_for_session(session_id):
    """Directed address to one live session."""
    return "self:%s" % session_id


def routes_at_turn(session_id):
    """Addresses a live session consumes into O at each hook fire."""
    return (address_for_session(session_id), ADDR_BROADCAST)


def is_session_id(s):
    """True if `s` is a full Claude Code session UUID (8-4-4-4-12 hex) — the
    canonical address form, honored directly even for a session not in the live
    roster this instant (it drains within TTL). The short you see in a message is
    its exact session short (trace_contract.session_short), resolved against the
    live roster by signal.resolve_session."""
    s = s or ''
    return (len(s) == 36 and s.count('-') == 4
            and all(c in '0123456789abcdef-' for c in s.lower()))


def sender_short(sender):
    """How a message names its sender: a session by its session short; any other
    sender (the dashboard's 'operator-dashboard') whole — a label, not an id to
    cut."""
    return session_short(sender) if is_session_id(sender) else (sender or '')


# A self-message has no `intent` — it is just a message.
# Delivery renders every message identically (quoted, attributed); see the
# RECEIVED-MESSAGE RENDER section below.


# ═══════════════════════════════════════════════════════════════
# IN-FLIGHT MESSAGE SHAPE  (the only durable storage; DDL in schema.py)
# ═══════════════════════════════════════════════════════════════
# Holds a directed/broadcast live self-message from send until the recipient
# pulls it into Observation, then it is consumed. The next_boot letter does NOT
# live here — it is the encoded session arc, surfaced at boot.
# Per-message expires_at (resolved by ADDRESS at send — see
# signal._resolve_ttl_hours) enforces expiry: readers filter `expires_at > now`,
# the reaper deletes `expires_at <= now`. Stamped via iso_after so it shares
# iso_now's UTC-ISO format and stays lex-comparable. The authoritative column
# list is the schema.py DDL.


# ═══════════════════════════════════════════════════════════════
# SIZE CONTRACT  —  a message is never cut
# ═══════════════════════════════════════════════════════════════
# Standing rule: every limit on text must have an EXPLICIT, documented
# contract — never a bare magic number doing a silent slice. This section IS
# that contract for the self-channel.
#
#   SEND      (signal.send)            refuses a body over SIGNAL_BODY_MAX,
#                                      loudly, naming its length and the limit,
#                                      so the sender shortens it or sends a
#                                      path. Everything accepted is stored whole.
#   DELIVERY  (render_received_block)  shows whole messages only. The block
#                                      budget (RECEIVED_BLOCK_MAX) paces them:
#                                      messages past it are not consumed — they
#                                      stay in the inbox, and the tail counts
#                                      them and names self_inbox. The first
#                                      message always shows, even when it alone
#                                      is over the budget.
#   READ SURFACES (peek, presence)     cut loudly: a message names the get_trace
#                                      call that reads it whole; a focus line (a
#                                      glance whose whole is one self_peek away)
#                                      carries a bare '…'.
#
# Why the limit sits at send: a refused send tells the one party who can act
# on it, while a cut at delivery left the reader with a head and no way to
# read the rest. Self-messages deliver via the Stop `decision:block` reason
# (channels/delivery.py → signal.drain_and_render) — NOT additionalContext, so
# no shared inject budget (_MAX_INJECT_CHARS) applies (0706f3d9); the limit
# keeps one message to a size a reader takes in at a turn end.
SIGNAL_BODY_MAX = 5000             # a message body; send refuses anything longer
RECEIVED_BLOCK_MAX = 4000          # turn-end budget; past it, messages wait in the inbox
PEEK_MSG_MAX = 300                 # per-message cap on a peek's recent_msgs (a glance, not a transcript)
ROSTER_FOCUS_MAX = 100             # the roster's one-line focus per session


def cap_peek_text(text, trace_id):
    """A peek's message, cut LOUDLY at PEEK_MSG_MAX. The marker names the
    trace that holds it whole — the reader's next call, not a place to go
    look. A cut that would read longer than the whole message is not made."""
    text = text or ''
    cut = cap_text_loud(
        text, PEEK_MSG_MAX,
        marker='… (+%%d chars — get_trace %s for the whole message)' % trace_id)
    return text if len(cut) >= len(text) else cut


def first_line_marked(text, limit):
    """The first line of `text`, at most `limit` chars, with a bare '…' when
    the line itself was cut. A glance line: the mark says there is more, and
    self_peek shows it."""
    lines = (text or '').strip().splitlines()
    return cap_text_loud(lines[0].strip(), limit, marker='…') if lines else ''

# ── POLICY DEFAULTS (tunable knobs; NOT truncation points) ──────────────
# TTL is per-message, resolved at send() by ADDRESS and stamped as expires_at
# (signal.send → signal._resolve_ttl_hours). Defaults below are documented
# here; runtime overrides via brain.get_config('self_channel.{kind}_ttl_hours').
#   • broadcast — "who's live right NOW"; dies within the hour (≈ the
#     ROSTER_LIVE_WINDOW_MIN + grace presence window). A session booting later
#     must NOT inherit a stale broadcast — the short TTL is that guard.
#   • directed  — waits for ONE specific session to come back; a day = "you'll
#     see it when you next work today".
# TTL is by address only — a self-message has no `intent`.
# The next_boot letter is the encoded arc surfaced at boot, NOT a stored
# self_inflight row (see header), so nothing letter-shaped is in the courier.
# Time is WALL-CLOCK (iso_now/iso_after) — TTL measures REAL elapsed hours, and
# cross-session sends aren't on the eval-replay path. This package sits outside
# the clock contracts' scanned tree BY PLACEMENT, not by an exemption entry:
# it lives under servers/channels/, and servers/scales/__init__.py owns the
# rule that puts it there.
BROADCAST_TTL_HOURS = 1            # undelivered broadcast older than this is dead
DIRECTED_TTL_HOURS = 24            # a directed message waits up to a day for its recipient
ROSTER_ACTIVE_WINDOW_MIN = 5       # "active": acted this recently — reach freely, expect a reply
ROSTER_LIVE_WINDOW_MIN = 30        # "dormant" ceiling: live but quiet (watch-mode asleep / operator away) — sees you next wake
ROSTER_LOST_GRACE_MIN = 30         # grace past LIVE: a session gone this recently is surfaced as "lost", not dropped
PRESENCE_MAX_SESSIONS = 10         # roster shows a count + top-K ranked, never enumerates all


def ttl_kind_for(address):
    """TTL category by delivery address: 'broadcast' (ephemeral live-
    coordination) vs 'directed' (waits for one recipient). The only two
    categories — the next_boot letter isn't a stored courier row, so it has no
    TTL here."""
    return 'broadcast' if address == ADDR_BROADCAST else 'directed'


# ═══════════════════════════════════════════════════════════════
# TRACE  —  the S0 correspondent marker (NOT a scale)
# ═══════════════════════════════════════════════════════════════
# A self-originated turn is an S0 exchange whose incoming message came from a
# session, not the operator. It is marked next to `user_message` on
# S0's K side. The response (`assistant_message`) and the encoding (s1e) are
# entirely unchanged — same mechanism, different correspondent.
# REF_SELF_MESSAGE is the live s0 marker (validated by the guard below).
# CORRESPONDENT_* label a turn's speaker — reserved for Phase 4 (encoding
# self-originated S0 turns); not referenced elsewhere yet.
CORRESPONDENT_OPERATOR = "operator"
CORRESPONDENT_SELF = "self"
REF_SELF_MESSAGE = "self_message"   # (s0, K) — see servers/trace_contract.py

# Loud-by-default: the correspondent marker must exist in the S0 contract, or
# self-dialogue turns would write trace events the validator rejects. Fail at
# import, not 1000 turns later. (raise, not assert — unconditional under -O.)
if REF_SELF_MESSAGE not in _REF_TYPES.get(("s0", "K"), ()):
    raise RuntimeError(
        "self_contract ↔ trace_contract drift: %r is missing from "
        "REF_TYPES[('s0','K')]. Add it (next to 'user_message')." % REF_SELF_MESSAGE)


# ═══════════════════════════════════════════════════════════════
# RECEIVED-MESSAGE RENDER  —  how an incoming self-message surfaces to Anchor
# ═══════════════════════════════════════════════════════════════
# A message from another session (a concurrent self) is ALWAYS rendered as quoted,
# attributed reported speech — `other session (id:X) says: "…"` — never first-person.
# That re-voicing is the containment barrier: you can't absorb another session's
# claim as your own without a visible grammatical error. A concurrent session is
# never past-you, so nothing renders it in first person. The one exception is a note a session sent to its own id: those
# words ARE yours, so it renders as `your own note (this session)`. Callers pass
# `when` (a relative-time string) and a session's `focus`; the contract FORMATS,
# never reaching into clock / session state.

def render_signal(body, short="", focus="", when="", own=False):
    """Live signal arriving in O — a tap from ANOTHER session, rendered as REPORTED
    speech: `who` is a grammatical subject and the body is QUOTED as their claim,
    never your own first-person assertion. That re-voicing is the containment
    barrier — you cannot absorb '<who> says: "I did X"' as something YOU did
    without a visible grammatical error. EVERY delivered cross-session message is
    rendered this way — a concurrent session is never your past self, so it's always
    quoted and attributed, never first-person. `who` is the sending session's short
    id (trace_contract.session_short), shown as `other session (id:<short>)`; `own`
    marks a note this session sent to itself, shown as `your own note (this session)`."""
    if own:
        who = "your own note (this session)"
    else:
        who = ("other session (id:%s)" % short) if short else "another session"
    tag = " · ".join(p for p in (focus, when) if p)
    head = "%s%s says:" % (who, (" [%s]" % tag) if tag else "")
    return '%s\n   "%s"' % (head, (body or "").strip())


def classify_liveness(age_min):
    """A session's minutes-since-last-activity → liveness state. The 'state of the
    peer' a sender needs before relying on the channel: active = here now (reply
    expected); dormant = live but quiet (sees it on its next wake); lost = just past
    the live window (surfaced for a grace period, not silently dropped)."""
    if age_min <= ROSTER_ACTIVE_WINDOW_MIN:
        return "active"
    if age_min <= ROSTER_LIVE_WINDOW_MIN:
        return "dormant"
    return "lost"


def render_presence(sessions, lost=()):
    """Ambient roster line — perception, not memory. `sessions` = [(short_id, focus,
    state), ...] of LIVE sessions (active|dormant), excluding the reader; `lost` =
    [(short_id, focus), ...] just past the window. 'live' counts active+dormant;
    lost are named at the tail so a vanished peer isn't silently dropped."""
    if not sessions and not lost:
        return ""
    line = "%s live: %d" % (SESSION_TERM_PLURAL, len(sessions))
    parts = ["%s [%s]: %s" % (sid, state, focus or "—") for sid, focus, state in sessions]
    if parts:
        line += " — " + " · ".join(parts)
    if lost:
        line += " · %d lost (%s)" % (len(lost), ", ".join(sid for sid, _ in lost))
    return line


def _short_ts(iso):
    """ISO timestamp → compact 'MM-DD HH:MM' for the first-contact intro line."""
    if not iso:
        return ""
    try:
        d, t = iso.split("T")
        return "%s %s" % (d[5:], t[:5])
    except Exception:
        return iso[:16]


def _render_first_contact(m):
    """The first message this session gets from another session → a one-block
    intro built from its peek (who, since when, last-active + liveness, what
    it's working on) plus the short reply target. Subsequent messages skip this — context once,
    then lean. `sender_peek` is attached by drain_and_render; absent → just the
    reply hint."""
    short = m.get("from", "")
    pk = m.get("sender_peek") or {}
    when_bits = []
    if pk.get("session_started_at"):
        when_bits.append("started %s" % _short_ts(pk["session_started_at"]))
    if pk.get("last_active_at"):
        when_bits.append("last active %s (%s)" % (
            _short_ts(pk["last_active_at"]), pk.get("liveness") or "?"))
    line = " · ".join(when_bits) or "new session"
    out = '\n   ⓘ first contact · %s · reply: self_send to="%s"' % (line, short)
    focus_line = first_line_marked(pk.get("focus"), PEEK_MSG_MAX)
    if not focus_line:
        # Already cut, and marked, by the peek (cap_peek_text).
        msgs = pk.get("recent_msgs") or []
        focus_line = (msgs[0].get("text") if msgs else "") or ""
    if focus_line:
        out += "\n     working on: %s" % focus_line
    return out


def _render_one(m):
    """Render ONE pending message for injection, whole — send already held it
    to SIGNAL_BODY_MAX. Every message renders as quoted, attributed reported
    speech (see render_signal)."""
    rendered = render_signal(m.get("body", "") or "", short=m.get("from", ""),
                             own=m.get("own", False))
    if m.get("first_contact"):
        rendered += _render_first_contact(m)
    return rendered


def render_received_block(messages, cap=RECEIVED_BLOCK_MAX):
    """Compose pending self-messages into ONE budgeted Observation block.

    Whole messages only (see the SIZE CONTRACT above): the block fills to
    `cap` (RECEIVED_BLOCK_MAX), always keeping the first message, and counts
    the rest at the tail. Returns (block, shown) — shown is how many messages the block renders, in
    order, and the caller consumes ONLY those: a message past the cap was never
    seen, so it stays in the inbox. ("", 0) for no messages (caller skips the
    block entirely)."""
    if not messages:
        return "", 0
    own_frame = ("🧵 a note you sent to this session",
                 "   — your own words, delivered back to you.")
    others_frame = ("🧵 from your %s" % SESSION_TERM_PLURAL,
                    "   — what they did is theirs; you know it, you didn't do it. "
                    "Attribute accordingly if you encode.")
    body, shown, waiting = compose_block_loud(
        messages, _render_one, cap,
        reserved=max(len(h) + len(n) for h, n in (own_frame, others_frame)))
    # The frame describes what the block shows, not what waits behind it.
    head, note = (own_frame if all(m.get("own") for m in messages[:shown])
                  else others_frame)
    if waiting:
        body += ("\n\n(+%d more waiting in your inbox — call self_inbox to "
                 "read them whole now, or they arrive at your next turn end)"
                 % waiting)
    return "%s\n%s\n\n%s" % (head, note, body), shown
