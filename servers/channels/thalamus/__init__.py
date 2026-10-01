"""The Thalamus — the brain speaking to its streams (durable, windowed).

A channel, not a scale: it runs no integrate(O, K) → Δ loop of its own. A
delivery is traced as an **s0 K event**
(`thalamus_contract.REF_THALAMUS_DELIVERY`, registered beside `self_message`
and checked at import), so the Thalamus rides the S0 loop as an incoming
correspondent rather than introducing a grain of its own. What it adds is
storage and policy, not a cycle. Why that puts it under `channels/`, and what
the split buys: the package docstring one level up.

Not the `operator` lateral either — that one is the async Anchor↔operator
channel considered and deprioritized in docs/LATERAL-SCALES.md, and it is the
one carrying the "prove it's not just async S0" burden. Here the recipient is
ANCHOR; the operator sees an item only when Anchor surfaces it.

Sibling, not parent, to the self-channel: msgs are streams speaking to EACH
OTHER (ephemeral, consume-once, TTL); the Thalamus is the brain speaking to its
streams (durable, windowed, ledgered). One item can spawn N deliveries over its
life, knows who already got it, and can carry an answer back to its producer.
Δ from any scale becomes a future O for a session; this is where that hand-off
waits. The two layers meet at exactly one place — a live-now FYI, which
`file()` delegates to the courier and marks terminal instead of queueing.

Delivery is PULL-ONLY, at the moments that provably land: the Thalamus never
enumerates sessions, never pushes, and holds no roster. A session self-serves
against the item list and records its own delivery at render time.

Design: docs/THALAMUS-DESIGN.md · taxonomy: docs/LATERAL-SCALES.md
"""
