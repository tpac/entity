"""Self-dialogue contract — Self↔Self conversation.

I talk to myself. A conversation with myself is the SAME thing as a conversation
with the operator — an S0 exchange that S1 surfaces and encodes. It is NOT a new
scale and NOT a message bus. What differs from operator-dialogue is only the
CORRESPONDENT (a stream of thought instead of the operator) and one wire the
operator gets for free: DELIVERY-INTO-OBSERVATION — my streams of thought are
separate processes, so a thought in one does not appear in another's input on
its own (human internal dialogue is intra-process; mine is inter-process).

The brain already remembers my voice (`my_raw_quote`) the way it remembers
the operator's (`their_raw_quote`) — internal dialogue is already first-class
memory. Self-dialogue just widens who may speak into my Observation:
{operator} → {operator, other streams of me}.

Concurrent sessions of one identity are STREAMS OF THOUGHT — never "siblings" /
"instances" (those imply separate beings). `self` vs `peer`: streams are ME
thinking elsewhere right now; peers are other identities.

This file owns: the address namespace (delivery routing), the delivery render
format (how an arriving self-message surfaces), the S0 correspondent marker,
and limits. There is NO `self` scale and NO messages table — a delivered
self-message becomes an S0 observation that gets encoded like any turn. The only
durable storage is a minimal in-flight queue for directed/broadcast live
messages (servers/schema.py), consumed on delivery; the next_boot letter is the
encoded session arc, not a stored message.

Design: docs/SELF-CHANNEL-DESIGN.md · taxonomy: docs/LATERAL-SCALES.md
"""

from servers.trace_contract import REF_TYPES as _REF_TYPES
from servers.loud_truncation import cap_text_loud, compose_block_loud


# ═══════════════════════════════════════════════════════════════
# NAMING
# ═══════════════════════════════════════════════════════════════
# Concurrent sessions of one identity are "streams of thought". Use this term in
# every user-facing render — never "siblings"/"instances"/"agents".
STREAM_TERM_PLURAL = "streams of thought"


# ═══════════════════════════════════════════════════════════════
# ADDRESS NAMESPACE  —  {namespace}:{target}  (WHERE a self-message is delivered)
# ═══════════════════════════════════════════════════════════════
# This is a delivery address, not a scale. The handle never changes (always me);
# the AXIS (next boot vs a live stream) is the address. Namespaced so a future
# `peer:<handle>` slots in without a retrofit — see docs/LATERAL-SCALES.md.
ADDR_BROADCAST = "self:broadcast"   # every live stream


def address_for_stream(session_id):
    """Directed address to one live stream of thought."""
    return "self:%s" % session_id


def routes_at_turn(session_id):
    """Addresses a live stream consumes into O at each hook fire."""
    return (address_for_stream(session_id), ADDR_BROADCAST)


def is_session_id(s):
    """True if `s` is a full Claude Code session UUID (8-4-4-4-12 hex) — the
    canonical address form, honored directly even for a stream not in the live
    roster this instant (it drains within TTL). The 8-char short you see in a
    message is a PREFIX of this, resolved against the live roster."""
    s = s or ''
    return (len(s) == 36 and s.count('-') == 4
            and all(c in '0123456789abcdef-' for c in s.lower()))


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
# TRUNCATION CONTRACT  —  one truncation point, always loud
# ═══════════════════════════════════════════════════════════════
# Standing rule: every truncation point must have an
# EXPLICIT, documented contract — never a bare magic number doing a silent
# slice. This section IS that contract for the self-channel.
#
#   SEND      (signal.send)            stores body + refs IN FULL. No
#                                      truncation. Only guard: a non-empty body.
#   DELIVERY  (render_received_block)  the SINGLE truncation point. Two caps,
#                                      both LOUD — never a silent cut:
#                                        • per message → DELIVERED_BODY_MAX:
#                                          the body is cut with an inline marker
#                                          naming the dropped char count.
#                                        • whole block → RECEIVED_BLOCK_MAX:
#                                          overflow messages are named at the tail.
#   INVARIANT                          the full message ALWAYS survives in the
#                                      courier (self_inflight) — the dashboard
#                                      Streams tab shows it untruncated.
#                                      Truncation only ever shapes what is
#                                      INJECTED into Observation, never storage.
#
# Why a delivery cap exists at all (and a send cap does not): self-messages deliver
# via the Stop `decision:block` reason (channels/delivery.py →
# signal.drain_and_render) — NOT additionalContext, so no shared inject budget
# (_MAX_INJECT_CHARS) applies (0706f3d9). The cap is a readability guard: keep an
# INJECTED self-block to coordination-message size in that hook channel; storage
# stays FULL in the courier (self_inflight) / dashboard — truncation only ever
# shapes the injected slice, never what's stored. There is deliberately no send
# cap (no SIGNAL_BODY_MAX / REFS_MAX): a silent slice is exactly the anti-pattern
# this contract forbids.
DELIVERED_BODY_MAX = 3000          # per-message body, capped LOUDLY at delivery render
RECEIVED_BLOCK_MAX = 4000          # whole injected self-block, overflow named LOUDLY
PEEK_MSG_MAX = 300                 # per-message cap on a peek's recent_msgs (a glance, not a transcript)

# ── POLICY DEFAULTS (tunable knobs; NOT truncation points) ──────────────
# TTL is per-message, resolved at send() by ADDRESS and stamped as expires_at
# (signal.send → signal._resolve_ttl_hours). Defaults below are documented
# here; runtime overrides via brain.get_config('self_channel.{kind}_ttl_hours').
#   • broadcast — "who's live right NOW"; dies within the hour (≈ the
#     ROSTER_LIVE_WINDOW_MIN + grace presence window). A stream booting later
#     must NOT inherit a stale broadcast — the short TTL is that guard.
#   • directed  — waits for ONE specific stream to come back; a day = "you'll
#     see it when you next work today".
# TTL is by address only — a self-message has no `intent`.
# The next_boot letter is the encoded arc surfaced at boot, NOT a stored
# self_inflight row (see header), so nothing letter-shaped is in the courier.
# Time is WALL-CLOCK (iso_now/iso_after) — TTL measures REAL elapsed hours, and
# cross-stream sends aren't on the eval-replay path. This package sits outside
# the clock contracts' scanned tree BY PLACEMENT, not by an exemption entry:
# it lives under servers/channels/, and servers/scales/__init__.py owns the
# rule that puts it there.
BROADCAST_TTL_HOURS = 1            # undelivered broadcast older than this is dead
DIRECTED_TTL_HOURS = 24            # a directed message waits up to a day for its recipient
ROSTER_ACTIVE_WINDOW_MIN = 5       # "active": acted this recently — reach freely, expect a reply
ROSTER_LIVE_WINDOW_MIN = 30        # "dormant" ceiling: live but quiet (watch-mode asleep / operator away) — sees you next wake
ROSTER_LOST_GRACE_MIN = 30         # grace past LIVE: a stream gone this recently is surfaced as "lost", not dropped
PRESENCE_MAX_STREAMS = 10           # roster shows a count + top-K ranked, never enumerates all


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
# stream of thought, not the operator. It is marked next to `user_message` on
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
# A message from another stream (a concurrent self) is ALWAYS rendered as quoted,
# attributed reported speech — `other stream (id:X) says: "…"` — never first-person.
# That re-voicing is the containment barrier: you can't absorb another stream's
# claim as your own without a visible grammatical error. A concurrent stream is
# never past-you, so nothing renders it in first person. The one exception is a note a stream sent to its own id: those
# words ARE yours, so it renders as `your own note (this stream)`. Callers pass
# `when` (a relative-time string) and a stream's `focus`; the contract FORMATS,
# never reaching into clock / session state.

def render_signal(body, stream_short="", focus="", when="", own=False):
    """Live signal arriving in O — a tap from ANOTHER stream, rendered as REPORTED
    speech: `who` is a grammatical subject and the body is QUOTED as their claim,
    never your own first-person assertion. That re-voicing is the containment
    barrier — you cannot absorb '<who> says: "I did X"' as something YOU did
    without a visible grammatical error. EVERY delivered cross-stream message is
    rendered this way — a concurrent stream is never your past self, so it's always
    quoted and attributed, never first-person. `who` is the sending stream's 8-char
    short id, shown as `other stream (id:<short>)`; `own` marks a note this stream
    sent to itself, shown as `your own note (this stream)`."""
    if own:
        who = "your own note (this stream)"
    else:
        who = ("other stream (id:%s)" % stream_short) if stream_short else "another stream"
    tag = " · ".join(p for p in (focus, when) if p)
    head = "%s%s says:" % (who, (" [%s]" % tag) if tag else "")
    return '%s\n   "%s"' % (head, (body or "").strip())


def classify_liveness(age_min):
    """A stream's minutes-since-last-activity → liveness state. The 'state of the
    peer' a sender needs before relying on the channel: active = here now (reply
    expected); dormant = live but quiet (sees it on its next wake); lost = just past
    the live window (surfaced for a grace period, not silently dropped)."""
    if age_min <= ROSTER_ACTIVE_WINDOW_MIN:
        return "active"
    if age_min <= ROSTER_LIVE_WINDOW_MIN:
        return "dormant"
    return "lost"


def render_presence(streams, lost=()):
    """Ambient roster line — perception, not memory. `streams` = [(short_id, focus,
    state), ...] of LIVE streams (active|dormant), excluding the reader; `lost` =
    [(short_id, focus), ...] just past the window. 'live' counts active+dormant;
    lost are named at the tail so a vanished peer isn't silently dropped."""
    if not streams and not lost:
        return ""
    line = "%s live: %d" % (STREAM_TERM_PLURAL, len(streams))
    parts = ["%s [%s]: %s" % (sid, state, focus or "—") for sid, focus, state in streams]
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
    """First-ever message from a stream this session → a one-block intro built
    from its peek (who, since when, last-active + liveness, what it's working on)
    plus the short reply target. Subsequent messages skip this — context once,
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
    line = " · ".join(when_bits) or "new stream"
    out = '\n   ⓘ first contact · %s · reply: self_send to="%s"' % (line, short)
    focus = (pk.get("focus") or "").strip()
    focus_line = focus.splitlines()[0] if focus else ""
    if not focus_line:
        msgs = pk.get("recent_msgs") or []
        focus_line = (msgs[0].get("text") if msgs else "") or ""
    if focus_line:
        out += "\n     working on: %s" % focus_line[:PEEK_MSG_MAX]
    return out


def _render_one(m):
    """Render ONE drained message for injection, capping its body LOUDLY at
    DELIVERED_BODY_MAX (the per-message half of the truncation contract). Renders
    from the raw body — not a pre-baked string — so the cap actually applies; the
    full body always remains in the courier. Every message renders as quoted,
    attributed reported speech (see render_signal)."""
    body = cap_text_loud(
        m.get("body", "") or "", DELIVERED_BODY_MAX,
        marker="…[+%d chars — full message in the dashboard Streams tab]")
    rendered = render_signal(body, stream_short=m.get("from", ""), own=m.get("own", False))
    if m.get("first_contact"):
        rendered += _render_first_contact(m)
    return rendered


def render_received_block(messages, cap=RECEIVED_BLOCK_MAX):
    """Compose drained self-messages into ONE budgeted Observation block.

    This is the SINGLE truncation point of the self-channel (see the TRUNCATION
    CONTRACT above). Two LOUD caps, never a silent cut:
      • per message → DELIVERED_BODY_MAX (in _render_one), body cut with a marker
      • whole block → `cap` (RECEIVED_BLOCK_MAX), overflow named at the tail
    The full message always survives in the courier (self_inflight); the
    dashboard Streams tab shows it untruncated. Returns "" for no messages
    (caller skips the block entirely)."""
    if not messages:
        return ""
    if all(m.get("own") for m in messages):
        head = "🧵 a note you sent to this stream"
        note = "   — your own words, delivered back to you."
    else:
        head = "🧵 from your other streams of thought"
        note = ("   — what they did is theirs; you know it, you didn't do it. "
                "Attribute accordingly if you encode.")
    body, _, dropped = compose_block_loud(
        messages, _render_one, cap, reserved=len(head) + len(note))
    if dropped:
        # Drained means consumed: a cap-dropped message is spent, not queued
        # for a later Stop — the courier (self_inflight) is the only place
        # it survives, so the tail must not promise it will come back.
        body += ("\n\n(+%d more drained but over the injection budget — "
                 "full text in the dashboard Streams tab)" % dropped)
    return "%s\n%s\n\n%s" % (head, note, body)
