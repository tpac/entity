"""Channels — the correspondents that ADDRESS a live stream.

A channel is a CORRESPONDENT plus a WIRE, indexed the way
`docs/LATERAL-SCALES.md` indexes laterals — by who is on the other end:

    self_channel/   another stream of me (`peer:` once a persistent one exists)
    thalamus/       the brain itself, speaking to its streams

"Addresses" is the boundary, not "reaches". Recall, the Frame and the boot
render all reach a live stream and live elsewhere (`scales/s1`,
`brain_voice.py`); they are output, not correspondence. Nor is this the
`additionalContext` sense of "channel" used in CLAUDE.md's Conventions — that
names the injection pipe, this names who is speaking.

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
