"""S2 Consolidation contract — cluster shape, config, and rendering.

Three contracts:
1. CLUSTER_SHAPE — what the decoder produces, what the encoder can rely on
2. CONSOLIDATION — config (thresholds, encoder settings)
3. CONSOLIDATION_NODE_FORMAT — how cluster member nodes are rendered for the encoder
"""

from servers.contract import META_BRIEF


# ═══════════════════════════════════════════════════════════════
# CLUSTER SHAPE — decoder→encoder contract
#
# Each cluster is a dict with these fields. The encoder RELIES on
# this shape — if the decoder changes what it produces, update here
# and the encoder's _format_clusters() must adapt.
#
# This is the equivalent of S1's CANDIDATE_POOL contract.
# ═══════════════════════════════════════════════════════════════

CLUSTER_SHAPE = {
    # ── Identity ──
    'nodes': 'list[str]',              # Node IDs in the cluster (sorted)
    'size': 'int',                     # len(nodes)

    # ── Similarity (two independent dimensions) ──
    'content_cosine_max': 'float',     # Highest content embedding similarity
    'content_cosine_avg': 'float',     # Average content similarity
    'title_cosine_max': 'float',       # Highest title-only embedding similarity
    'title_cosine_avg': 'float',       # Average title similarity
    'pair_scores': 'dict[str, dict]',  # "idA-idB" → {content: float, title: float}

    # ── Pre-classification (decoder's algorithmic guess) ──
    'pre_class': 'str',               # likely_consolidate | likely_evolve | likely_keep | needs_judgment

    # ── Node details (per member) ──
    'node_details': 'dict[str, dict]',  # node_id → {
    #   title, type, content, confidence, encoding_source,
    #   locked, created_at, updated_at,
    #   reasoning*, their_raw_quote*, my_raw_quote*, situation*
    #   (* = from node_metadata_kv, may be absent)
    # }

    # ── Behavioral evidence (from S1 traces) ──
    'co_recall_count': 'int',          # Times pair appeared as candidates for same query
    'judge_preference': 'dict[str, int]',  # node_id → times selected by surfacer
    'recall_counts': 'dict[str, int]',     # node_id → total times appeared as candidate
    'query_coverage': 'dict[str, list]',   # node_id → [query strings that found it]

    # ── Catalog blindness (from S1E traces) ──
    'catalog_blind': 'dict[str, bool]',  # node_id → was created without seeing cluster mates?

    # ── Graph structure ──
    'shared_edge_count': 'int',        # Neighbors shared by ALL cluster members
    'unique_edges': 'dict[str, int]',  # node_id → edges not in shared set
    'edge_details': 'dict[str, dict]', # node_id → {neighbor_id: [get_connections_bulk entry, one per
                                        #   direction — id, type, title, direction, edge_created_at,
                                        #   relations: [{relation, description, weight, created_at,
                                        #   updated_at}]]} (render_edge_lines' input, per entry)
    'communities': 'dict[str, list]',  # node_id → [{id, title}] community memberships
    'same_community': 'bool',          # Any pair shares a community?
    'shared_community_ids': 'list[str]',

    # ── Correction ──
    'has_correction_edge': 'bool',     # Correction/supersedes edge between members?

    # ── Tension ──
    'has_tension_edge': 'bool',        # Contradicts/challenges edge between members?
                                        # NEVER consolidate — tensions are productive
}

# Fields the encoder MUST see to make good decisions.
# If any of these are missing, the formatter should warn.
CLUSTER_REQUIRED_FIELDS = {
    'nodes', 'size', 'pre_class',
    'content_cosine_max', 'title_cosine_max',
    'node_details',
    'co_recall_count', 'judge_preference', 'catalog_blind',
    'shared_edge_count', 'same_community', 'has_correction_edge',
    # the prompt's ONLY edge source (the rich node block renders none) — a
    # cluster without it hands the encoder edge-blind nodes
    'edge_details',
}


# ═══════════════════════════════════════════════════════════════
# NODE RENDERING FORMAT
#
# How cluster member nodes are rendered for the consolidation encoder.
# Consolidation needs FULL depth for 2-5 nodes — Sonnet must read the whole
# content to decide synthesize vs keep, and a survivor's content it did not
# fully see is content a merge can silently drop.
# ═══════════════════════════════════════════════════════════════

CONSOLIDATION_NODE_FORMAT = {
    'content_limit': None,      # the whole node: a merge or keep is decided on the claim, not a gist
    # Edges render ONCE per cluster, in the Intra-cluster and External blocks
    # (_format_clusters) — every edge, with direction and the whole
    # description — so the rich node block shows none.
    'edge_limit': 0,
    'metadata_limit': META_BRIEF,   # reasoning, raw quotes
    'time_format': 'relative',
    # `communities` stays off: the cluster block prints one Communities
    # line per cluster (_format_clusters).
}


# ═══════════════════════════════════════════════════════════════
# CONSOLIDATION ENRICHMENT LLM CONFIG
# (stored in interactions table as 's2_consolidation_enrichment' parameters)
# ═══════════════════════════════════════════════════════════════

CONSOLIDATION_ENRICHMENT = {
    'model': 'claude-sonnet-4-6',
    'max_tokens': 32768,
}


# ═══════════════════════════════════════════════════════════════
# CONSOLIDATION CONFIG
# ═══════════════════════════════════════════════════════════════

CONSOLIDATION = {
    # ── Embedding scan (decoder) ──
    'similarity_threshold': 0.89,       # Minimum cosine for candidate pair.
                                        # Raised from 0.82 after Nomic-Q migration: at 0.82 the
                                        # similarity graph collapsed into a 2165-node giant
                                        # component (86% of corpus) — unprocessable as clusters.
                                        # 0.89 yields ~250 well-formed clusters with biggest ~12.
    'max_cluster_size': 10,             # Larger clusters get dropped + logged.
                                        # Historical note: 5 was too tight —
                                        # genuine convergence across sessions
                                        # routinely produced 6-9 member clusters
                                        # that were silently lost. 10 covers the
                                        # observed backlog; if clusters hit 15+,
                                        # threshold is probably too permissive.

    # ── Behavioral trace lookback (decoder) ──
    'co_recall_lookback_hours': 168,    # 7 days of S1R traces
    'encoding_lookback_hours': 168,     # 7 days of S1E traces

    # ── Suppression (decoder) ──
    # LIVE SOURCE: the `settlement` aspect (aspects_v1.json) — the decoder
    # derives its suppression set from brain.aspects.settlement.edge_relations.
    # This constant is the fallback for a degraded/empty registry only, and
    # mirrors the aspect's seed membership. A node touched by any of these
    # edges is "reviewed"; reviewed↔reviewed pairs are skipped.
    # similar_to / consolidated_into = KEEP/merge outcomes; corrects /
    # supersedes / resolves (+ inverse forms — stored orientation is advisory)
    # = one endpoint settled the other. Membership changes are a human edit
    # to the aspect entry, never a code edit here.
    'suppression_relations': {
        'similar_to', 'consolidated_into',
        'corrects', 'corrected_by',
        'supersedes', 'superseded_by',
        'resolves', 'resolved_by',
    },

    # ── Pre-classification thresholds (decoder) ──
    'likely_consolidate_cosine': 0.90,  # Above this + structural signal = likely consolidate

    # ── Encoder ──
    'model': 'claude-sonnet-4-6',
    'max_tokens': 32768,
    'max_proposals_per_call': 10,       # Clusters per Sonnet call
    'max_rounds': 2,                    # Tool-use rounds per call — read then write, done

    # ── Cold start / run cap ──
    # (suppression_relations fallback above; live derivation is
    # suppression_relations() at the bottom of this file — the ONE
    # source both the decoder scan and the encoder payload render use.)
    # Max clusters the encoder processes per idle cycle.
    # Cold start (164 clusters) spreads across ~6 cycles.
    # Each cycle: suppression edges prevent re-processing.
    # Order: likely_consolidate → likely_evolve → likely_keep → needs_judgment
    # Easy cases first — validates prompt quality before harder decisions.
    'max_clusters_per_run': 10,
}


def suppression_relations(brain):
    """The decoder's suppression verb set, derived from the settlement aspect.

    The ONE derivation both consumers use: the decoder's embedding-scan
    reviewed filter and the encoder's payload 'Settlement relations' line.
    They must never disagree — the encoder describes to Sonnet exactly the
    set the decoder scanned with. Falls back to the config constant when
    the registry serves an empty aspect (degraded working copy).
    """
    rels = set(brain.aspects.settlement.edge_relations)
    return rels or set(CONSOLIDATION['suppression_relations'])
