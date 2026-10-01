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
    # LLM config — must match what production runs; a wrong max_tokens here is
    # silent (a healer batch quietly truncates instead of erroring).
    'model': 'claude-haiku-4-5',
    'max_tokens': 4096,

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
# model/max_tokens have exactly one home in this file.
HEALER_INTERACTION_DEFAULT = {
    'model': HEALER['model'],
    'max_tokens': HEALER['max_tokens'],
}
