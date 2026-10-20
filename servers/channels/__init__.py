"""Channels — the correspondents that ADDRESS a live session.

A channel is a CORRESPONDENT plus a WIRE, indexed the way
`docs/LATERAL-SCALES.md` indexes laterals — by who is on the other end:

    self_channel/   another session of me (`peer:` once a persistent one exists)
    thalamus/       the brain itself, speaking to its sessions

"Addresses" is the boundary, not "reaches". Recall, the Frame and the boot
render all reach a live session and live elsewhere (`scales/s1`,
`brain_voice.py`); they are output, not correspondence. Nor is this the
`additionalContext` sense of "channel" used in CLAUDE.md's Conventions — that
names the injection pipe, this names who is speaking.

One piece here also serves a door that is not correspondence: the session
resolver, `self_channel/signal.resolve_session`, names the session behind the
trace filters too (its `history` pool), so a session reference resolves one
way everywhere.

NOT under `scales/`: neither package runs an `integrate(O, K) → Δ` loop, and
the delivery windows here are real-elapsed by design (TTLs, roster staleness).
The placement rule and why it is load-bearing live in one place —
`servers/scales/__init__.py`, the axis these packages are not on.

A delivery is traced as an s0 K event (`self_message`, `thalamus_delivery`),
so a channel rides the S0 loop as an incoming correspondent rather than
introducing a grain of its own. The leg itself — which source speaks at which
moment, the render walk, the trace — has ONE owner: `delivery.py`, which both
hooks call; the channel packages own storage and policy only.

`world` is NOT a channel — it is ingestion, a source you read rather than a
correspondent you address, and must not ride this transport
(`docs/LATERAL-SCALES.md`, which owns that call).

Taxonomy: docs/LATERAL-SCALES.md
"""
