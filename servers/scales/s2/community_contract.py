"""S2 Community Detection contract — config, metadata schema, rendering.

Three contracts in one file:
1. COMMUNITY_DETECTION — decoder config (z-scores, thresholds)
2. COMMUNITY_METADATA_KEYS — flat keys for community nodes
3. S2CE_NODE_FORMAT — how representative nodes are rendered for the encoder
"""

from servers.contract import (
    META_BRIEF, META_NONE, connect_to_relations, connect_to_target,
    normalize_connect_to, unpack_fields)


# ═══════════════════════════════════════════════════════════════
# DECODER CONFIG
# ═══════════════════════════════════════════════════════════════

COMMUNITY_DETECTION = {
    # Minimum nodes to form a community proposal
    'min_community_size': 3,

    # ── Z-score pair scoring ──
    # z = (raw_shared - bucket_mean) / bucket_std
    # z≥1.0 = 1 std above average for pairs at this degree
    'z_seed_threshold': 1.0,

    # ── Add-to-existing affinity threshold ──
    # Fraction of node's typed neighbors that must be in the community.
    # 15% was too low (noise). 25% = at least 1 in 4 neighbors.
    'add_to_existing_min_affinity': 0.25,

    # Max candidate communities on one add_to_existing proposal. Enforced by
    # the Step 5b emitter AND the Step 9c merge — they must agree, or the
    # merge silently re-truncates what the emitter allowed.
    'add_candidates_cap': 3,

    # ── Embedding placement for orphans ──
    # Paused (PZ-1): the 0.50 raw-cosine gate is below the measured random
    # node↔centroid baseline (0.78) and ~27% of accepted placements scored
    # at noise level. A wrong member corrupts a community boundary; an
    # unplaced node costs nothing and re-enters via S1E edges. Flip to True
    # to restore the channel (the threshold below then applies again).
    'orphan_placement_enabled': False,
    'embedding_placement_threshold': 0.50,

    # ── Cluster-to-community overlap (duplicate disclosure) ──
    # Before proposing new_community, check if cluster members mostly
    # connect to an existing community. If the fraction of cluster nodes
    # with neighbors in the same existing community >= this threshold, the
    # proposal carries `overlaps_existing` and the encoder judges
    # new-story-vs-extension. (PZ-1: this used to CONVERT the cluster into
    # add_to_existing proposals — measured eating 94% of births.)
    'cluster_overlap_threshold': 0.60,

    # ── Drift detection ──
    # foreign_affinity > max(home_affinity, floor) × drift_ratio triggers a
    # drift proposal (the floor keeps the ratio a real lever at home_aff 0).
    # Targets are existing communities only — never same-run clusters.
    # Per-node override: _sys_drift_threshold in metadata_kv, raised by the
    # encoder's algorithmic Δ (never the agent) each time a drift proposal is
    # declined, so weak drift doesn't re-propose. Ratchets up only — the
    # decoder clamps any stored value below drift_ratio.
    'drift_ratio': 1.5,
    'drift_ratio_step': 0.1,        # how much encoder raises per rejection
    'drift_min_foreign_affinity': 0.15,  # absolute floor for foreign affinity

    # ── Community merge detection ──
    # Merge when overlap >= this AND unique members in smaller < min_unique
    'merge_overlap_threshold': 0.80,    # % of smaller community's members shared
    'merge_min_unique_members': 3,      # smaller must have < this many unique to merge

    # ── Cross-cutting detection ──
    'cross_cutting_min_degree': 15,
    'cross_cutting_max_top_affinity': 0.35,

    # ── Community health seam ──
    # ONE tunable (the review zone) + ONE structural fact (disconnected):
    #   typed int_frac < low_cohesion_threshold → the community is loose enough
    #       to act on. Within that zone:
    #         · DISCONNECTED (no internal edge of any real-cohesion relation —
    #           see non_cohesion_relations) → deterministic auto-archive: no
    #           encoder round, no rejection fingerprint, so it can never get
    #           stuck "suppressed but undead". A structural fact, not a tuned
    #           number — and counting ALL relations (incl. similar_to) means a
    #           community cohesive only via similar_to is NOT auto-archived.
    #         · otherwise → encoder JUDGES archive-or-keep (the review surface).
    # The threshold is a pure volume dial — int_frac is low population-wide
    # (median ~0.49, p10 ~0.22), so it stays conservative; above ~0.15 it would
    # sweep in normal communities.
    'low_cohesion_threshold': 0.10,  # typed int_frac below this → act on it
    # Relations that do NOT count as internal cohesion for the disconnected
    # check: structural membership edges, dream artifacts, and pure-generic
    # 'related'. Everything else (similar_to, synthesizes, and all typed
    # relations) counts — so a community linked only by these is
    # "disconnected" → safe to auto-archive.
    'non_cohesion_relations': (
        'community_member',
        'dream_observation', 'dreamed_from', 'related', 'related_to'),

    # ── Encoder ── (its model and max_tokens: COMMUNITY_ENRICHMENT below)
    'max_proposals_per_call': 8,   # Smaller batches = richer per-community
                                    # context AND lower round-2 generation
                                    # load (10 was hitting the 180s client
                                    # timeout on batch 1 consistently).
    # Split rich decision batches before dispatch instead of truncating away
    # their evidence. A single oversized proposal is retained and logged.
    'max_batch_context_chars': 48000,
    'max_actionable_per_run': 30,   # Cap per idle cycle
    # Per-type quotas — priority order: merge > new_community > add_to_existing > health > drift
    'type_quotas': {
        'merge_communities': 3,
        'new_community': 10,
        'add_to_existing': 12,
        'health_update': 3,
        'drift': 2,
    },
    'max_rounds': 3,                # tool rounds per batch — get_nodes, brain_batch, one repair of refused ops

    # ── Idle-run gating (Phase 1, 2026-05-29) ──
    # The decode is a pure function of graph state — re-running it on an
    # unchanged graph re-derives identical, already-rejected proposals. Skip
    # the whole unit unless the graph changed AND this interval has elapsed
    # since the last actual decode. See CommunityDetection._should_skip.
    'min_run_interval_seconds': 30 * 60,
}


# ═══════════════════════════════════════════════════════════════
# CORRIDOR CLASSIFICATION (shared: decoder + structural helper)
# ═══════════════════════════════════════════════════════════════
#
# A "corridor" is a low-cohesion cluster that bridges areas of the graph
# rather than forming a tight community. The decoder classifies corridors
# when it validates/summarizes fresh clusters, through
# community_structural.structural_metrics — the one cohesion formula the
# decoder and the structural stamp share. One named source for the rule.
CORRIDOR_INTERNAL_FRACTION_MAX = 0.20  # int_frac strictly below this …
CORRIDOR_MIN_SIZE = 3                   # … AND member count strictly above this → corridor


# ═══════════════════════════════════════════════════════════════
# COHESION ADJACENCY FILTERS (shared: decoder + structural helper)
# ═══════════════════════════════════════════════════════════════
#
# The typed-edge adjacency that internal/external cohesion is computed over
# excludes structural/Hebbian edges (not semantic cohesion) and edges whose
# aspect is noise/generic. The decoder builds this adjacency for its own fresh
# computation; the structural helper builds it (scoped) when stamping. Both
# MUST exclude the same relations/aspects — else the stamped cohesion metrics
# disagree with the decoder's. One named source so the parity-critical filters
# can't drift. (The SQL JOIN structure stays in each builder; the parity test
# guards that.)
ADJACENCY_EXCLUDED_RELATIONS = ('community_member',)
ADJACENCY_SKIP_ASPECTS = ('generic_relation', 'noise')


# ═══════════════════════════════════════════════════════════════
# COMMUNITY ENRICHMENT LLM CONFIG
# (code default of the 's2_community_enrichment' interaction config)
# ═══════════════════════════════════════════════════════════════

# Sonnet 4.6 after the 2026-08-12 model A/B (eval/ab_community_model.py):
# equal-or-better completion than Haiku 4.5, markedly more consistent across
# identical inputs, richer journal discipline.
COMMUNITY_ENRICHMENT = {
    'model': 'claude-sonnet-4-6',
    'max_tokens': 32768,
}


# ═══════════════════════════════════════════════════════════════
# PROPOSAL TYPE PRIORITY
# ═══════════════════════════════════════════════════════════════
#
# Order proposals by structural impact so the encoder sees the
# highest-leverage work first:
#   1. merges simplify the community landscape
#   2. new_community creates frontier for future inside-out scans
#   3. add_to_existing extends communities at the frontier
#   4. health_update corrects community metadata
#   5. drift moves a single node between communities (lowest impact)
#
# Used by rejection_table.sort_proposals_by_priority() when capping
# quotas and sorting the batch sent to the encoder.

TYPE_PRIORITY = {
    'merge_communities': 0,
    'new_community': 1,
    'add_to_existing': 2,
    'health_update': 3,
    'drift': 4,
}


# ═══════════════════════════════════════════════════════════════
# COMMUNITY NODE METADATA KEYS
#
# The community_* keys code writes and reads: the structural stamp writes the
# structural keys, the encoder the judgment keys, and the decoder's state read
# takes both. A community node is still open KV like any node — whatever else
# a writer put on it renders — except the retired keys below.
#
# Membership is not metadata: it is the community_member edges.
# ═══════════════════════════════════════════════════════════════

# Computed by code from the member edges, never authored by the encoder. A fact
# is stored only when a reader cannot get it cheaply from the graph: the member
# count, the corridor flag and the dominant member type are each one query
# away; internal_fraction is a walk over every member's typed edges.
COMMUNITY_STRUCTURAL_KEYS = (
    'community_internal_fraction',    # Float 0-1
)

# Structural-field revises per brain_batch call. Each call holds the write lock
# every other writer waits on. On a production-size brain (2026-09-29) one
# metadata revise costs ~12 ms — revise re-syncs the node's FTS row, a scan of
# the FTS table — so a chunk holds the lock for ~0.25 s, and the first cycle's
# fill of every community interleaves with other writes instead of blocking.
STRUCTURAL_STAMP_CHUNK = 20

# Judgment the community encoder writes and maintains.
COMMUNITY_JUDGMENT_KEYS = (
    'community_maturity',             # "forming" / "active" / "settled" / "corridor"
)

COMMUNITY_METADATA_KEYS = frozenset(
    COMMUNITY_STRUCTURAL_KEYS + COMMUNITY_JUDGMENT_KEYS)

# Keys nothing writes any more. Schema v38 deleted their rows, and every view
# renders each stored key, so retiring a key means adding it here AND a schema
# step that deletes its rows — a kept row would show a frozen value as current.
# tests/test_retired_fields.py keeps every prompt from teaching one.
COMMUNITY_RETIRED_KEYS = frozenset({
    'community_members',          # membership is the community_member edges
    'community_key_decisions',    # moved to the standard `thought` field
    'community_size',             # member-edge count, one query away
    'community_is_corridor',      # internal_fraction + size
    'community_dominant_type',    # one GROUP BY over the members
    'community_narrative',        # retired 2026-04: duplicated `content`
    'community_open_questions',   # retired 2026-04: became `question`
    'community_latest_development',  # the frontier content already tells;
                                     # the decision view lists the newest members
})


def _community_create(spec):
    """A write spec with its `fields` envelope merged — the way every create
    door reads it — or None when it does not create a community (a malformed
    envelope is the door's own refusal to make)."""
    spec, err = unpack_fields(spec) if isinstance(spec, dict) else (None, None)
    return spec if not err and spec and spec.get('type') == 'community' else None


def community_member_targets(spec):
    """The member ids a community-creating write names in its connect_to;
    empty for any other write. The create refusal and the S2 rejection
    matcher both read a create's membership through here."""
    spec = _community_create(spec)
    if spec is None:
        return set()
    return {connect_to_target(e)
            for e in normalize_connect_to(spec.get('connect_to')) or []
            if any(isinstance(r, dict) and r.get('relation') == 'community_member'
                   for r in connect_to_relations(e))
            and isinstance(connect_to_target(e), str) and connect_to_target(e)}


def community_create_refusal(spec):
    """The refusal for a write that creates a community without naming its
    members in connect_to, or None when the write may proceed.

    A community is its community_member edges; created without them it is a
    node no view can show members for, and consolidation archives it on the
    next cycle. Refusing at the write lets the writer re-send it whole — the
    community encoder's repair round exists for exactly that."""
    if _community_create(spec) is None or community_member_targets(spec):
        return None
    return ('a community is created with its members: name each one in '
            'connect_to with relation "community_member" — nothing was written')


# ═══════════════════════════════════════════════════════════════
# S2CE NODE RENDERING FORMAT
#
# How member nodes are rendered for the community encoder: its get_nodes
# drill-ins and the decision evidence's node detail. The encoder needs to
# understand what the community IS about — not every detail of every
# member, but enough to write a narrative. S1E writes about individual
# nodes and needs full depth; S2CE writes about communities and needs the
# gist.
# ═══════════════════════════════════════════════════════════════

# Used when encoder calls get_nodes() to inspect specific members, and for
# non-community nodes in the decision evidence
S2CE_NODE_FORMAT = {
    'content_limit': 800,       # Generous gist — the encoder needs real member content to write a quality narrative
    'edge_limit': 5,            # Relations matter — keep 5 with descriptions
    'metadata_limit': META_BRIEF,   # key metadata only
    'correction_render': 'balanced',  # relation verb + edge desc + 150-char excerpt; NOT heavy (the per-node correction firehose that blew the context to 217K)
    'time_format': 'relative',  # "2d ago" not "2026-04-09"
    'communities': 'title',     # where a rep already sits — context for placement, no ids
}

# Existing communities in the context listing (the ~15 recall picks). The
# encoder REVISES these — a merge rewrites the narrative, an add extends it —
# so it reads each one whole: a 150-char gist cut 73% of them (community
# content averages ~1,100 chars) and a rewrite from a gist loses the story.
# ~15 × 1.1K chars per batch is the price; edges and metadata stay off.
S2CE_COMMUNITY_FORMAT = {
    'content_limit': None,      # the whole narrative — it is what gets revised
    'edge_limit': 0,            # No edges
    'metadata_limit': META_NONE,  # no metadata — content is the narrative
    'time_format': 'relative',
}

# Above the shared ladder deliberately: a community named by a proposal is the
# object under decision, and the encoder rewrites its meaning fields from this
# view. Owned here rather than in contract.py's ladder — this depth serves one
# consumer and no other reader should inherit it.
S2CE_DECISION_META = 800

# Communities named by a proposal are decision evidence, rather than the
# compact neighbours above. Membership is attached from live edges separately.
S2CE_DECISION_FORMAT = {
    'content_limit': None,
    'edge_limit': 4,
    'metadata_limit': S2CE_DECISION_META,
    'show_edge_total': True,
    'correction_render': 'balanced',
    'time_format': 'relative',
}
S2CE_MEMBER_PREVIEW = 8

# What the encoder sees per batch (community_encoder._build_batch_context):
# journal continuity, the decision evidence for the communities and nodes the
# proposals name, compact nearby communities, then the proposals themselves
# (_format_proposals: one block per proposal — a new community shows its
# counts, overlap flag, edge signature and one line per member).
