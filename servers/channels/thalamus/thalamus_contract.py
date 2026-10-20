"""Thalamus contract — the brain speaking to its sessions.

The Thalamus is a durable store of STANDING INTENTS with delivery policy: one
item can spawn N deliveries over its life (once per session inside a window),
knows who already got it (the ledger), and can carry an answer back to its
producer. It sits BESIDE the self-channel, not inside it — the msgs layer is
sessions speaking to each other (ephemeral, consume-once, TTL); the Thalamus is
the brain speaking to its sessions (durable, windowed, ledgered). A live-now
fire-and-forget FYI never enters the queue: `file()` delegates it to the
courier broadcast, where irrelevance-at-send is already the contract.

This file owns: states, audiences, caps and default windows, the `when` /
`for_whom` resolution grammar, the delivery render, and the trace-contract
drift guard. Mechanics live in thalamus.py; DDL in servers/schema.py
(thalamus_items / thalamus_deliveries, logs DB).

Time is WALL-CLOCK (iso_now / iso_after / iso_cutoff) — delivery windows and
the producer's settled-recently window are courier-class real-elapsed spans
over transaction-time columns, the same documented exemption as the
self-channel; nothing here is on the eval-replay conversation-time path.

Design: docs/THALAMUS-DESIGN.md
"""

from datetime import datetime as _dt

from servers.loud_truncation import cap_text_loud, compose_block_loud
from servers.clock import iso_after, resolve_offset, FUTURE
from servers.channels.delivery import PROMPT as _PROMPT, STOP as _STOP


# ═══════════════════════════════════════════════════════════════
# STATES  —  the item lifecycle (delivery state, NOT producer-condition state)
# ═══════════════════════════════════════════════════════════════
# Whether the underlying condition is still true stays the producer's business
# (two orthogonal state machines — brain node e63c41dd). Retry, when it ships
# (Phase 3), gates on unacked delivery, never on a still-open condition.
STATE_OPEN = 'open'            # active: pullable inside its window
STATE_ANSWERED = 'answered'    # closed by Anchor with an answer payload
STATE_DISMISSED = 'dismissed'  # closed by Anchor without an answer
STATE_WITHDRAWN = 'withdrawn'  # closed by its OWN producer (retraction)
STATE_EXPIRED = 'expired'      # window ended — LOUD for an unanswered ask
                               # (the dead-letter fix), natural for a notice
STATE_SENT = 'sent'            # terminal at file(): delegated live-now
                               # broadcast — the courier owns its death
STATES = (STATE_OPEN, STATE_ANSWERED, STATE_DISMISSED, STATE_WITHDRAWN,
          STATE_EXPIRED, STATE_SENT)  # the closed lifecycle — every
                                      # partition of it (fate_of) is checked
                                      # against this tuple


# ═══════════════════════════════════════════════════════════════
# AUDIENCES  —  the item's RECIPIENT SET (who the pull predicate serves).
# NOT a frequency: delivery cardinality is constant — the ledger PK caps
# every item at one delivery per session per armed_epoch, and repetition
# over time is the epoch axis (re-arm). This enum only picks who is
# eligible. A directed item narrows first_session to one named session
# via the target_session column.
# ═══════════════════════════════════════════════════════════════
AUDIENCE_FIRST = 'first_session'  # the first session to pull after due —
                                  # a dynamic singleton (reminders/notices)
AUDIENCE_EVERY = 'every_session'  # every session inside the window, once
                                  # each (standing notices and asks), until
                                  # MAX_DELIVERIES_PER_ITEM sessions saw it

AUDIENCES = (AUDIENCE_FIRST, AUDIENCE_EVERY)  # the closed set the pull
                                              # predicate matches — a value
                                              # outside it would never deliver
                                              # and die silently at expiry;
                                              # file() guards the door

def default_audience(needs_answer):
    """An ask renders at each new session's first prompt until answered, expired, or
    MAX_DELIVERIES_PER_ITEM sessions passed on it; a reminder/notice fires
    once. Both overridable via for_whom."""
    return AUDIENCE_EVERY if needs_answer else AUDIENCE_FIRST


# ═══════════════════════════════════════════════════════════════
# MOMENTS  —  the delivery moments a session pulls at. The vocabulary is
# OWNED by channels/delivery.py (the last-mile leg both channels ride);
# these are its names, kept here so the pull predicate and ledger speak
# them without reaching around the contract. `via` is written to the
# ledger verbatim, so it must be vocabulary, not free text — pull()
# validates against MOMENTS loudly (a typo'd via would behave as Stop and
# ledger the typo).
# ═══════════════════════════════════════════════════════════════
VIA_PROMPT = _PROMPT.name
VIA_STOP = _STOP.name
MOMENTS = (VIA_PROMPT, VIA_STOP)
# Asks deliver at moments chosen by their AUDIENCE. A broadcast ask (every
# session) renders at the session's first operator prompt only — an
# architecture question arriving mid-thread trains reflex-deferral; at the
# first prompt there is no thread yet (it was boot until 2026-10-02, ruling
# id:15e46fa9: boot cannot tell an operator from a scheduler, the first
# prompt can). A directed ask (one named session) renders at that session's
# Stop: the Scribe asking the session it just encoded IS mid-thread by
# design, and a session you can name has already had its first prompt —
# first-prompt-only would dead-letter it.
ASK_MOMENTS = {AUDIENCE_EVERY: (VIA_PROMPT,),
               AUDIENCE_FIRST: (VIA_STOP,)}


# ═══════════════════════════════════════════════════════════════
# CAPS & WINDOWS  —  policy as data. The door never refuses for volume: the
# Thalamus is always open for work, and what a reader is shown is bounded at
# the read — per moment, and per item across sessions.
# ═══════════════════════════════════════════════════════════════
PULL_MAX_ITEMS = 5         # per render moment (prompt / stop), overflow named;
                           # repeat-eligible items rotate fewest-first under it
MAX_DELIVERIES_PER_ITEM = 16  # an every_session item that this many
                           # sessions saw SINCE IT LAST CAME DUE is DONE: the
                           # sweep expires it as a window end does (loud for
                           # an ask, warned for a notice). Sessions that
                           # passed on an item are the signal; more slots are
                           # noise. The budget belongs to the IDENTITY, not
                           # to a generation of it: a re-file that only
                           # changes the body cannot renew it, and the
                           # reader's defer always can. One carrying a new
                           # `when` also does, by moving the due date the
                           # budget counts from — a narrower door than the
                           # armed_epoch one it replaced, left open pending
                           # evidence that any producer walks through it.
                           # Code default; the config key below overrides it
                           # live
MAX_DELIVERIES_CONFIG_KEY = 'thalamus.max_deliveries_per_item'
                           # brain.set_config(key, n) — read at every pull
                           # and sweep, no restart (the embed_queue
                           # trace_window_days pattern). Clamped to >= 1 at
                           # the read: the cap drives a TERMINAL transition,
                           # so there is no <= 0 "unlimited" sentinel here —
                           # a very large number is how you switch it off
SATURATION_GRACE_HOURS = 24  # the delivery that REACHES the cap is one a
                           # session is holding right now, so the sweep waits
                           # this long past the last delivery before closing.
                           # A window end has no reader; saturation always
                           # does. Delivery itself stops at the cap regardless.
                           # Read over EVERY delivery, wider than the budget's
                           # own span: the guard asks whether a session holds
                           # the item right now, which neither a re-file nor a
                           # defer may answer by emptying what it reads
REASK_COOLOFF_DAYS = 7     # an ASK that closed with no answer cannot be
                           # re-filed under that identity for this long.
                           # Expiry frees the dedup key, so without it a
                           # cyclic producer's next run inserts a fresh item
                           # with a fresh window — the treadmill the cap
                           # exists to end. Asks only: a notice or a dated
                           # reminder reaching its window end is a producer
                           # RE-ARMING on its own cycle, the designed flow —
                           # gating those refused the re-arm that is the
                           # whole point of a recurring reminder
BLOCK_MAX = 4000           # whole injected block — loud cap, mirror of the
                           # self-channel RECEIVED_BLOCK_MAX discipline
BODY_MAX = 1500            # per-item body at render — loud cap; storage is FULL
RENDER_REFS_MAX = 3        # refs resolved inline per item, rest named

# The queue as a list (thalamus_list, and the dashboard's copy of this string:
# its separation contract forbids importing servers.*, and a test pins the
# copy). Open before closed; then what is due, the most recently due first;
# then what is scheduled, the soonest first; `id` last, so equal due dates
# keep one order and a page never repeats or skips a row. Both `?` bind now
# (iso_now).
_LIST_DUE = 'COALESCE(deliver_at, created_at)'
LIST_ORDER_SQL = ("(state <> '{open}'), ({d} > ?), "
                  "CASE WHEN {d} > ? THEN {d} END, {d} DESC, id"
                  .format(open=STATE_OPEN, d=_LIST_DUE))

ASK_EXPIRES_DAYS = 14      # needs_answer window; expiry past it is LOUD
NOTICE_EXPIRES_DAYS = 7    # undated-notice window
REMIND_GRACE_DAYS = 7      # dated items: expiry = deliver_at + grace
PRODUCER_VIEW_SETTLED_DAYS = 7  # a producer sees its item's settled outcome
                                # this long after it settled; open items it
                                # sees regardless (the open-pin rule)


# ═══════════════════════════════════════════════════════════════
# KIND  —  ONE derivation of what an item IS, feeding both the render verb
# and the expiry-span lookup (three unnamed partitions of the item space
# disagreeing was the root of the false-dead-letter defect). Internal and
# DERIVED — never producer-set, never stored: the rejected producer-facing
# kind vocabulary stays rejected. Producers state needs_answer / when /
# for_whom; the kind falls out.
# ═══════════════════════════════════════════════════════════════
KIND_ASK = 'ask'            # needs an answer (needs_answer)
KIND_REMINDER = 'reminder'  # carries a clock (deliver_at)
KIND_NOTICE = 'notice'      # undated FYI

# How a part of me is heard when its item renders — first person, from the
# whole toward the part ("I need to help my healer"), never a producer label.
# %s is the part's name (part_of). Items I filed myself get no lead.
KIND_LEAD = {KIND_ASK: 'I need to help my %s:',
             KIND_REMINDER: 'My %s reminds me:',
             KIND_NOTICE: 'My %s noticed:'}
KIND_EXPIRES_DAYS = {KIND_ASK: ASK_EXPIRES_DAYS,
                     KIND_REMINDER: REMIND_GRACE_DAYS,
                     KIND_NOTICE: NOTICE_EXPIRES_DAYS}


def kind_of(item):
    """item dict (needs_answer / deliver_at suffice) → its derived kind."""
    if item.get('needs_answer'):
        return KIND_ASK
    if item.get('deliver_at'):
        return KIND_REMINDER
    return KIND_NOTICE


# FATE — how an item ENDED, as its producer should hear it: the one
# derivation mirroring kind_of, so no consumer re-partitions the state space
# (an expired ASK is the dead-letter case and says so). None = the producer's
# own act (withdrawn) or the courier's (sent): nothing to report back.
FATE_OPEN = 'open'
FATE_ANSWERED = 'answered'
FATE_DISMISSED = 'dismissed'
FATE_EXPIRED = 'expired'
FATE_EXPIRED_UNANSWERED = 'expired, unanswered'
FATE_PASSED_OVER = 'expired, unanswered — %d sessions passed'
_FATE_BY_STATE = {STATE_OPEN: FATE_OPEN, STATE_ANSWERED: FATE_ANSWERED,
                  STATE_DISMISSED: FATE_DISMISSED, STATE_EXPIRED: FATE_EXPIRED}
FATE_STATES = tuple(_FATE_BY_STATE)  # the states a producer hears about
NO_FATE_STATES = (STATE_WITHDRAWN, STATE_SENT)  # its own act / the courier's
assert set(FATE_STATES) | set(NO_FATE_STATES) == set(STATES), \
    'fate_of must partition STATES — a new state needs a fate or a reason'

# The states the door's re-ask gate (REASK_COOLOFF_DAYS) keys on, for an
# ASK only: the question ran its whole course and NOBODY engaged. The gate
# reads the STORED item's needs_answer, not the new call's — what it
# protects is a dead question coming back, not a key being reused.
# Deliberately not `dismissed` —
# that is the reader acting on a live item, and a producer re-asking in a
# better form is a designed flow (the producer view shows one row per
# subject, live state first) — nor `withdrawn`, the producer's own act.
REASK_GATED_STATES = (STATE_EXPIRED,)


def fate_of(item):
    """item dict (state + needs_answer suffice) → FATE_* or None.

    `deliveries` — the item's own all-time count, read RAW — sharpens an
    expired ask: "nobody was around" and "N sessions read it and moved on"
    call for opposite producer behaviour, and one state carries both. Raw on
    purpose: derived by comparing that count against the cap read NOW, it
    restated the fate of long-closed items whenever the knob moved, and it
    was silent for a directed item, which cannot saturate and is the only
    kind the journal's producer view reads. Naming the CAUSE of the close
    belongs to the sweep, which holds the cap it actually used; the fate
    reports COVERAGE, so the two cannot contradict each other."""
    fate = _FATE_BY_STATE.get(item.get('state'))
    if fate == FATE_EXPIRED and kind_of(item) == KIND_ASK:
        passed = item.get('deliveries') or 0
        return FATE_PASSED_OVER % passed if passed else FATE_EXPIRED_UNANSWERED
    return fate


# ═══════════════════════════════════════════════════════════════
# `when` RESOLUTION  —  the door's time grammar (a presentation concern,
# kept out of the mechanics; the table only ever sees resolved ISO)
# ═══════════════════════════════════════════════════════════════

def resolve_when(value):
    """Resolve a producer's `when` to a future ISO deliver_at, or None for
    "next opportunity". Accepts relative shorthand ('30m', '2h', '3d', '1w'),
    an ISO timestamp literal, or ''/None/'now'. Anything else raises
    ValueError — a malformed deadline must fail loud at the door, not
    silently become an immediate delivery.

    The grammar itself lives in clock.py, shared with recall_episodes'
    lookback bounds; what stays here is what is genuinely the door's — the
    'now'/empty convention (a queue has a "next opportunity"; a lookback
    bound does not) and the door-flavoured error message."""
    if not value or str(value).strip().lower() == 'now':
        return None
    try:
        return resolve_offset(value, direction=FUTURE)
    except ValueError as e:
        raise ValueError('thalamus: when=%r %s' % (value, e))


def window_for(needs_answer, deliver_at):
    """The item's expiry when the producer names none: ANCHOR (deliver_at,
    or now) × SPAN (per-kind days) — one composition, never branch-dependent.
    Anchoring at the due date keeps a dated item's full window; expiry
    before due would make it undeliverable and fire a FALSE loud
    dead-letter."""
    span = KIND_EXPIRES_DAYS[kind_of(
        {'needs_answer': needs_answer, 'deliver_at': deliver_at})]
    anchor = _dt.fromisoformat(deliver_at) if deliver_at else None
    return iso_after(days=span, at=anchor)


def extend_window(needs_answer, new_deliver, current_expires):
    """Defer's window rule: literally window_for's ANCHOR × SPAN composition
    re-anchored at the new due date — a deferred item keeps its KIND's full
    span (an ask stays an ask: 14 days past due, not a reminder's 7). A
    window already past that keeps its length."""
    return max(current_expires or '', window_for(needs_answer, new_deliver))


# ═══════════════════════════════════════════════════════════════
# `for_whom` RESOLUTION  —  producer intent → route
# ═══════════════════════════════════════════════════════════════
FOR_WHOM_LIVE = ('live', 'live-now', 'all-live')  # → courier broadcast, no queue row lifecycle
FOR_WHOM_ALL = 'all'                              # → AUDIENCE_EVERY (window)
# Every audience word; any other non-empty for_whom names one session.
FOR_WHOM_WORDS = FOR_WHOM_LIVE + (FOR_WHOM_ALL,)


def resolve_for_whom(for_whom, needs_answer):
    """Map a producer's `for_whom` to (route, audience, target_session).

    route is 'live' (delegate to the courier broadcast — fire-and-forget) or
    'queue'. A full session UUID targets one session (directed). None/'' means
    the default audience for the item's kind. Unknown strings raise — a typo'd
    audience must not silently become a default."""
    fw = (for_whom or '').strip()
    if not fw:
        return 'queue', default_audience(needs_answer), ''
    if fw.lower() in FOR_WHOM_LIVE:
        return 'live', '', ''
    if fw.lower() == FOR_WHOM_ALL:
        return 'queue', AUDIENCE_EVERY, ''
    from servers.channels.self_channel.self_contract import is_session_id
    if is_session_id(fw):
        # Directed: the session-keyed correction (b474ccbd) — full UUID only
        # here; the remind door (dispatch_thalamus) resolves a short first.
        return 'queue', AUDIENCE_FIRST, fw
    raise ValueError(
        "thalamus: for_whom=%r is not 'live', 'all', or a full session UUID"
        % (for_whom,))


# ═══════════════════════════════════════════════════════════════
# TRACE  —  the s0 delivery marker
# ═══════════════════════════════════════════════════════════════
# A Thalamus delivery is an incoming K to the receiving session, next to
# self_message. Untraced delivery IS the visibility problem this system
# exists to fix. channels/delivery.py writes it at BOTH moments (the caller
# owns tracing — it holds the chain); the ledger stays the delivery-policy
# record (who got which item, which epoch), the trace is the S0-stream join.
# The constant LIVES in trace_contract (which registers it in REF_TYPES by
# construction and is import-cycle-free from channels/); re-exported here for
# this package's callers.
from servers.trace_contract import REF_THALAMUS_DELIVERY


# ═══════════════════════════════════════════════════════════════
# RENDER  —  how due items surface. Answerable without a fetch: body +
# resolved refs + the pre-filled resolve call, inline (brain node 70016ed3).
# Single truncation point, always loud; storage keeps everything.
# ═══════════════════════════════════════════════════════════════

# The interactive channel's source — what the remind tool files under on
# every install. A channel label, not the agent's name: 'anchor' is the
# daemon's main-chat stamp whatever the agent is called (brain node
# efafd723; renaming it is a deferred, deliberate change).
INTERACTIVE_SOURCE = 'anchor'
PART_NAMES = {'encoder:sonnet': 'scribe'}  # sources that don't name a part


def part_of(source):
    """The part of me a producer is, by name — 's2:healer' → 'healer',
    's2:community_detection' → 'community detection'. The Scribe's source
    names its model, not its part, so it is mapped. '' for items I filed
    myself (INTERACTIVE_SOURCE): they are already mine and render without a
    lead."""
    if source == INTERACTIVE_SOURCE:
        return ''
    if source in PART_NAMES:
        return PART_NAMES[source]
    return (source or '').split(':')[-1].replace('_', ' ')


def render_item(item):
    """One due item: the id to recognize it by, then the note as my own —
    a part of me speaking ("I need to help my healer: …"), the exit inline.
    Formats what it is handed: refs arrive pre-resolved as 'ref_lines'
    (pull() owns resolution — batched, veil-aware; the contract never
    reaches into the brain)."""
    body = cap_text_loud(
        (item.get('body') or '').strip(), BODY_MAX,
        marker='…[+%d chars — thalamus_list shows it in full]')
    part = part_of(item.get('source', ''))
    if part:
        body = '%s %s' % (KIND_LEAD[kind_of(item)] % part, body)
    lines = ['• %s' % item.get('id', '?'), '  %s' % body]
    for ref_line in item.get('ref_lines') or []:
        lines.append('  ↳ %s' % ref_line)
    if item.get('needs_answer'):
        lines.append('  → thalamus_resolve("%s", answer=…) · defer_until=… · '
                     'dismiss=true' % item.get('id', '?'))
    else:
        lines.append('  → thalamus_resolve("%s", dismiss=true) when handled'
                     % item.get('id', '?'))
    return '\n'.join(lines)


def _head(n):
    return '📝 note to self' if n == 1 else '📝 notes to self (%d)' % n


# My standing instruction for what arrives: these are my own notes, mine to
# handle unless I need help — a part reports to the whole, nobody relays
# (brain node ee325c5c). %s is the counterpart's name. The second line names
# the read routes so an item is explorable without guessing at tools.
_HEAD_NOTE = (
    '   If I need help I should ask %s, otherwise I should quickly look into '
    'this on the next pause and handle it.\n'
    '     Explore: get_nodes(↳ refs) or recall(the question) for context · '
    'thalamus_list for the queue · '
    "recall_episodes(ref_type='thalamus_delivery') for who saw it · "
    'thalamus_resolve to act.')


def render_block(items, overflow=0, cap=BLOCK_MAX, counterpart=''):
    """Compose due items into ONE budgeted block. Two loud caps — per item
    (BODY_MAX, in render_item) and whole block (`cap`); overflow items are
    named at the tail, never silently cut. `counterpart` is who I'd ask
    for help ('' → 'my operator').

    Returns (block, kept) — kept is how many items actually rendered ('' and
    0 when items is empty). The caller must ledger ONLY the kept items: a
    cap-dropped item was never shown, and recording it as delivered would
    suppress it forever (it stays armed for the next moment instead)."""
    if not items:
        return '', 0
    # Budget against the widest possible head, then rebuild it from the
    # kept count — the head must claim what the block SHOWS, never what
    # was fetched (head, tail, ledger, and pull's count all say `kept`).
    note = _HEAD_NOTE % (counterpart or 'my operator')
    body, kept, dropped = compose_block_loud(
        items, render_item, cap,
        reserved=len(_head(len(items))) + len(note))
    head = _head(kept)
    tail = dropped + max(0, overflow)
    if tail:
        body += '\n\n(+%d more due — thalamus_list shows them)' % tail
    return '%s\n%s\n\n%s' % (head, note, body), kept
