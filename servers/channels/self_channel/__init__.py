"""The self channel — one identity messaging itself (self<->self).

Package is named `self_channel` so the directory doesn't shadow the ubiquitous
`self` receiver. There is NO "self" trace scale — self<->self turns are traced
as s0 with the `self_message` correspondent marker
(self_contract.REF_SELF_MESSAGE). It rides the S0/S1 loop with a self-
correspondent, not a new scale.

Design: docs/SELF-CHANNEL-DESIGN.md · taxonomy: docs/LATERAL-SCALES.md
"""
