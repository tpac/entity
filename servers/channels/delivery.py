"""Delivery — the last mile: every channel that touches a live stream rides this.

One owner for the leg: at a delivery MOMENT, walk the SOURCES, skip the
ineligible, render each failure-isolated, keep the blocks, trace each shown
delivery as an s0 K event on the receiving session's chain, and join. The
channel packages own storage and policy (courier: consume-once + TTL;
Thalamus: queue + ledger + windows); this module owns only the leg, so both
hooks call it instead of composing sources themselves.

Eligibility is a ruling as a predicate (operator rulings id:7c7e805c,
id:bb0513ae): a source speaks at a moment iff the moment FORCES a read or
the source SURVIVES A MISS. The courier declines the passive boot moment —
a consumed message that goes unread leaves only a `self_delivered` row and
silence — while a Thalamus item stays open after a missed render and an
unanswered ask expires loudly, so it can ride any moment.

MOMENT VOCABULARY lives here (thalamus_contract derives its names from
these). Sources import their channel packages CALL-TIME inside the render
adapters: thalamus_contract imports this module at load for the moment
names, so a load-time import back into `channels/` would close the cycle.
The per-source trace ref_types come from modules OUTSIDE channels/
(trace_contract, self_contract), keeping their import-time contract guards.

The caller owns the chain: `deliver` takes the receiving session's
SessionContext and stamps every trace with its s0 chain — the same
symmetric record at boot and Stop, which is what makes a delivery joinable
to the rest of the session's S0 stream.
"""

from collections import namedtuple

from servers.brain_traces import _s0_trace
from servers.trace_contract import REF_THALAMUS_DELIVERY
from servers.channels.self_channel.self_contract import REF_SELF_MESSAGE


# ═══════════════════════════════════════════════════════════════
# MOMENTS  —  when a live stream is touched. `forcing` is whether the
# channel COMPELS a read: a Stop `decision:block` reason cannot be skipped;
# boot `additionalContext` is passive and a miss is silent.
# ═══════════════════════════════════════════════════════════════
Moment = namedtuple('Moment', ('name', 'forcing'))

BOOT = Moment('boot', False)  # additionalContext at session start
STOP = Moment('stop', True)   # decision:block at turn end

MOMENTS = (BOOT, STOP)


# ═══════════════════════════════════════════════════════════════
# SOURCES  —  who may speak at a moment. `render(brain, session_id, moment)
# -> (block, n)` — n is what the block actually shows (the source ledgers /
# consumes for exactly those). Channel imports are CALL-TIME (see header).
# ═══════════════════════════════════════════════════════════════
Source = namedtuple('Source', ('name', 'noun', 'ref_type', 'survives_a_miss',
                               'render'))


def _courier_render(brain, session_id, moment):
    """The courier drain — consume-once, so a delivery here is spent."""
    from servers.channels.self_channel import signal
    return signal.drain_and_render(brain, session_id)


def _thalamus_render(brain, session_id, moment):
    """The Thalamus pull — the item stays open; the ledger records the show."""
    from servers.channels.thalamus import thalamus
    return thalamus.pull(brain, session_id, via=moment.name)


COURIER = Source('self', 'self-message(s)', REF_SELF_MESSAGE,
                 survives_a_miss=False, render=_courier_render)
THALAMUS = Source('thalamus', 'thalamus item(s)', REF_THALAMUS_DELIVERY,
                  survives_a_miss=True, render=_thalamus_render)

# Join order is render order: stream speech first, the brain's items after —
# the order the Stop hook always composed.
SOURCES = (COURIER, THALAMUS)

# The composed leg WARNS above this (operator ruling id:1e22a2f0 — no cap):
# each source already caps itself at 4000, so 5000 means more than one
# channel is contributing meaningfully. To the errors table, which has a
# reader (query_logs).
COMPOSITE_WARN = 5000


def serves(source, moment):
    """The eligibility ruling as a predicate: speak iff the moment forces a
    read, or the source can afford the miss."""
    return moment.forcing or source.survives_a_miss


def deliver(brain, ctx, moment):
    """Render everything due for `ctx.session_id` at `moment` — returns
    (composite, traced_ref_types): the joined block ('' when nothing is due)
    and the ref_type of each source that showed something AND had its trace
    recorded, in render order. The Stop hook gates its continuation stamp on
    those — an untraced delivery must never classify a reaction, or the
    timeline gets a reaction whose incoming side does not exist.

    Each source is failure-isolated: a raise is logged to the errors table
    and the walk continues. A source that shows something is traced as one
    s0 K event on the session's chain AFTER its block is kept — the sources
    ledger/consume inside render, so a trace failure must cost the trace,
    never a delivery the substrate already recorded as shown. The trace
    carries the rendered block as its content (capped loudly by _s0_trace,
    same as every turn trace) and the moment name as its ref_id — indexed,
    so query_traces(ref_type=..., ref_id='boot') separates boot preludes
    from Stop turns; the incoming side a dial-on correspondent surfaces in
    the encoder timeline (trace_contract.S0_CONVERSATIONAL_INCOMING)."""
    shown, traced = [], []  # (source, block) kept; ref_types whose trace landed
    for source in SOURCES:
        if not serves(source, moment):
            continue
        try:
            block, n = source.render(brain, ctx.session_id, moment)
        except Exception as e:
            brain._log_error(
                '%s_delivery_%s' % (source.name, moment.name), e,
                'delivery.deliver: %s render raised at %s (session=%s) — '
                'moment continues without it'
                % (source.name, moment.name, ctx.session_id))
            continue
        if not n:
            continue
        if not block:
            # The source consumed/ledgered n items yet rendered nothing —
            # they are spent and the model will never see them. Loud.
            brain._log_error(
                '%s_delivery_%s' % (source.name, moment.name), None,
                'delivery.deliver: %s reported %d delivered at %s but an '
                'empty block (session=%s)'
                % (source.name, n, moment.name, ctx.session_id))
            continue
        shown.append((source, block))  # kept FIRST — the block delivers even
        try:                           # if its trace below fails
            _s0_trace(
                brain, ctx, event_type='K', ref_type=source.ref_type,
                summary='delivered %d %s at %s' % (n, source.noun, moment.name),
                ref_id=moment.name, content=block)
            traced.append(source.ref_type)
        except Exception as e:
            brain._log_error(
                '%s_delivery_%s' % (source.name, moment.name), e,
                'delivery.deliver: trace write failed at %s — the block is '
                'still delivered, but this source is dropped from the '
                'continuation stamp (session=%s)'
                % (moment.name, ctx.session_id))
    composite = '\n\n'.join(b for _, b in shown)
    if len(composite) > COMPOSITE_WARN:
        brain._log_warning(
            'delivery_composite_over_budget',
            'composed %s leg is %d chars (warn at %d) — %s'
            % (moment.name, len(composite), COMPOSITE_WARN,
               ', '.join('%s: %d chars' % (s.name, len(b)) for s, b in shown)),
            'delivery.deliver — delivered anyway')
    return composite, tuple(traced)
