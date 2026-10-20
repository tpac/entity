"""S2 Healer — Contract and Configuration.

Generates missing findability fields (question, situation, reasoning) for
nodes that lack them. Uses Haiku with S0/S1 trace context for high-quality
generation.

The healer gap: nodes created before the current field contract, or by paths
that skipped field population, end up missing findability context. This unit
scans for those gaps and fills them in batch each S2 cycle. Re-healing a node
after a revise/connect is not wired yet (`staleness_triggers` below is unread);
the planned healer redo decides that trigger.
"""

HEALER = {
    # LLM config — must match what production runs. Thinking counts against
    # max_tokens (~1.8k output on a 5-node batch, ~2/3 of it thinking); 16k is
    # headroom, not a target, and a max_tokens stop is logged by _call_llm.
    # Model and effort travel together: overrides merge key by key, so an
    # override moving `model` to one without effort support (Haiku 4.5)
    # must also set `effort` to null or every call is rejected.
    'model': 'claude-haiku-5-5',
    'effort': 'high',
    'max_tokens': 16384,

    # Batch sizing
    'max_nodes_per_call': 10,        # Nodes per Haiku call
    'max_nodes_per_run': 50,         # Total nodes per S2 run (graduated: increase over time)

    # What triggers staleness (node needs re-healing) — not wired yet; the
    # planned healer redo owns this and needs_healing_key.
    'staleness_triggers': ['revise', 'connect'],

    # Metadata flag key
    'needs_healing_key': 'needs_healing',
}

# How the healer prompt renders a target's edges (contract.render_edge_lines
# over NODE_FORMAT_DEFAULTS): every relation, the description whole, ages
# relative.
HEALER_EDGE_FORMAT = {
    'time_format': 'relative',
}

# Interaction config default for the `s2_healer` K — sliced from HEALER so
# model/effort/max_tokens have exactly one home in this file.
HEALER_INTERACTION_DEFAULT = {
    'model': HEALER['model'],
    'effort': HEALER['effort'],
    'max_tokens': HEALER['max_tokens'],
}
