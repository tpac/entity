"""Pipeline Contract — shared constants and utilities for the recall/encoding pipeline.

This file contains cross-boundary constants used by multiple stages.
Boundary-specific logic lives in dedicated contracts:
- scales/s1/surface_contract.py — S1 Surface (relevance surfacing, formatting, enrichment)
- scales/s1/encode_contract.py — S1 turn encoder (Sonnet config, node formatting, catalog)

contract.py defines what fields a node HAS.
pipeline_contract.py defines what fields FLOW at each stage.
"""


# ═══════════════════════════════════════════════════════════════
# EMBEDDING GROUPS — multi-vector architecture
#
# Two cohorts coexist (the `cohort` value 'legacy' names the blended one):
#
#   BLENDED — whole-node views, recall's candidate ranking.
#     `title`, `_primary` (title+content), `high_meta` (situation+quotes),
#     `other_meta` (reasoning+patterns), `edge_context`, `question`.
#     A non-zero `weight` makes the group one of LAF recall's maxsim views
#     (recall_laf.MAXSIM_VIEWS); only weight > 0 is read, not its size.
#
#   FIELD (per-field, unblended) — used by S1 Surface's activation kernel.
#     One vector per semantic field. Weight = 0 keeps them out of recall's
#     views; the kernel reads them directly.
#
# Why both: blended vectors capture whole-node semantics well for retrieval
# ranking. Field vectors capture per-field signal so activation can fade per
# field (query about "what did they say" → their_raw_quote field lights up;
# query about "when does this apply" → situation field lights up). Blending
# them together in `high_meta` made each query score muddled.
#
# Migration: new nodes get both cohorts at encode time. Old nodes get lazy
# backfill when the activation kernel encounters missing field vectors.
# The kernel has a fallback chain (field → blend) so old nodes still work
# with reduced granularity until backfill.
# ═══════════════════════════════════════════════════════════════

# Who keeps a group's vector FRESH. `fields` says what a vector is made of;
# this says which write path deletes it when a source changes — the half
# of the declaration that edge_context lacked for months (a correct
# `fields` entry that no write path ever consumed, so ~44% of its vectors
# embedded an outdated graph snapshot). Both paths converge on
# Brain.invalidate_source_fields → vectors_affected_by → delete_for_node.
#   INVALIDATED_BY_REVISE      — the source is node kv/columns; revise()
#                                 owns the field write and the delete.
#   INVALIDATED_BY_EDGE_WRITE  — the source lives on EDGES, not the node;
#                                 GraphDAL reports the changed endpoints via
#                                 on_edge_text_changed and the brain deletes.
# tests/test_pipeline_contract.py holds every group to one of these.
INVALIDATED_BY_REVISE = 'revise_kv'
INVALIDATED_BY_EDGE_WRITE = 'graph_edge_write'
# The virtual field edge_context is built from — not a kv key; the marker
# backfill_vectors resolves via GraphDAL.get_edge_descriptions_for.
EDGE_DESCRIPTIONS_FIELD = '_edge_descriptions'

# edge_context producer policy — how many of a node's edge descriptions (by
# edge weight, noise-aspect relations excluded) are embedded. Read only
# through brain.get_interaction_config('edge_context'); a DB override wins
# over this default, and changing it stales every edge_context vector (the
# text changes for every node), so pair a change with a rebuild.
EDGE_CONTEXT_INTERACTION_DEFAULT = {'top_k': 15}


def validate_edge_context_config(cfg) -> list:
    """Violations for an edge_context config — a typo'd override must not
    become the running producer policy."""
    top_k = (cfg or {}).get('top_k')
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        return ['edge_context.top_k must be a positive int, got %r' % (top_k,)]
    return []

EMBEDDING_GROUPS = {
    # ── BLENDED COHORT (recall's maxsim views) ─────────────────────────

    # The diagnostic pointer. Always exists. Highest weight.
    'title': {
        'weight': 1.00,
        'fields': ['title'],
        'vector_type': 'title',
        'always_compute': True,
        'cohort': 'legacy',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
    # Title+content blend. Stored as `_primary` in node_enrichments.
    'blend': {
        'weight': 0.85,
        'fields': ['title', 'content'],
        'vector_type': '_primary',
        'always_compute': True,
        'cohort': 'legacy',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
    # When is this relevant + who said it (situation + quotes blend).
    'high_meta': {
        'weight': 0.70,
        'fields': ['situation', 'their_raw_quote', 'my_raw_quote'],
        'vector_type': 'high_meta',
        'always_compute': False,
        'cohort': 'legacy',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
    # Why was this stored + behavioral patterns + emergent (reasoning blend).
    'other_meta': {
        'weight': 0.40,
        'fields': ['reasoning', 'correction_pattern', 'source_context', '_emergent'],
        'vector_type': 'other_meta',
        'always_compute': False,
        'cohort': 'legacy',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
    # Edge context — concatenated edge descriptions. Enables recall to find
    # nodes through their relationship descriptions, not just their own content.
    'edge_context': {
        'weight': 0.55,
        'fields': [EDGE_DESCRIPTIONS_FIELD],
        'vector_type': 'edge_context',
        'always_compute': False,
        'cohort': 'legacy',
        'invalidated_by': INVALIDATED_BY_EDGE_WRITE,
    },
    # "What would someone ask to find this?" — Haiku-generated by S2 Healer,
    # stored in node_metadata_kv as 'question'. Bridges vocabulary gap.
    'question': {
        'weight': 0.90,
        'fields': ['question'],
        'vector_type': 'question',
        'always_compute': False,
        'cohort': 'legacy',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },

    # ── FIELD COHORT (used by S1 Surface activation kernel) ────────────
    # Weight=0 keeps them out of recall's views — they exist solely to give
    # the activation kernel per-field resolution.

    # Content body alone — split from `_primary` blend.
    'field_content': {
        'weight': 0.0,
        'fields': ['content'],
        'vector_type': 'content',
        'always_compute': False,
        'cohort': 'field',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
    # situation has NO field-cohort vector: it's served by the dedicated
    # `_situation` vector (full-text, shared with recall's situation-boost) via
    # FIELD_VECTOR_FALLBACK['situation']. One situation vector, two consumers.
    # Reasoning alone — split from `other_meta`.
    'field_reasoning': {
        'weight': 0.0,
        'fields': ['reasoning'],
        'vector_type': 'reasoning',
        'always_compute': False,
        'cohort': 'field',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
    # Their verbatim voice — split from `high_meta`.
    'field_their_raw_quote': {
        'weight': 0.0,
        'fields': ['their_raw_quote'],
        'vector_type': 'their_raw_quote',
        'always_compute': False,
        'cohort': 'field',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
    # My own verbatim voice — split from `high_meta`. Distinct channel from
    # their quote; a query about my reasoning should find this cleanly
    # without interference from their words.
    'field_my_raw_quote': {
        'weight': 0.0,
        'fields': ['my_raw_quote'],
        'vector_type': 'my_raw_quote',
        'always_compute': False,
        'cohort': 'field',
        'invalidated_by': INVALIDATED_BY_REVISE,
    },
}


# Fallback chain for the activation kernel: when a field vector is missing
# on an old node (not yet backfilled), fall back to the blended vector that
# contains the field's content. Ordered best-match-first.
FIELD_VECTOR_FALLBACK = {
    'title':              ['title'],  # no fallback needed — always computed
    'content':            ['_primary'],
    'situation':          ['_situation', 'high_meta'],
    'reasoning':          ['other_meta'],
    'their_raw_quote':    ['high_meta'],
    'my_raw_quote':       ['high_meta'],
    'question':           ['question'],  # no blend, direct
}


def field_vector_types():
    """Return the list of per-field vector_types the activation kernel reads.

    Stable order: title → content → situation → reasoning → their quote
    → my quote → question. Determinism matters for test assertions and
    trace readability.
    """
    return ['title', 'content', 'situation', 'reasoning',
            'their_raw_quote', 'my_raw_quote', 'question']


def fields_invalidated_by(owner: str) -> list:
    """Source fields whose vectors `owner` keeps fresh — the runtime reader of
    `invalidated_by`. Brain._edge_text_changed invalidates exactly these for
    INVALIDATED_BY_EDGE_WRITE, so declaring a new edge-sourced group is enough
    to have edge writes invalidate it."""
    seen = []
    for group in EMBEDDING_GROUPS.values():
        if group.get('invalidated_by') == owner:
            for f in group['fields']:
                if f not in seen:
                    seen.append(f)
    return seen


def vectors_affected_by(field_name: str) -> set:
    """Given a field name, return vector_types whose embedding text depends
    on it — these go stale when the field is updated and must be deleted
    so embed_queue's backfill creates fresh ones.

    Derived from EMBEDDING_GROUPS — single source of truth. Special case:
    `_situation` derives from node_metadata_kv['situation'] but is NOT an
    EMBEDDING_GROUPS entry (it's the dedicated situation vector, written
    by its own backfill scan), so it's added explicitly. Updating `situation`
    also invalidates `high_meta` (which blends situation + quotes); there is no
    longer a `field_situation`/`situation` vector to invalidate.

    Used by `brain.revise()` to invalidate stale vectors. Without this,
    `VectorDAL.find_missing()` skips rows that exist (text in kv changed
    but the row carries old embedding), and the vector permanently
    encodes outdated text.
    """
    affected = set()
    for spec in EMBEDDING_GROUPS.values():
        if field_name in spec.get('fields', []):
            affected.add(spec['vector_type'])
    # `_situation` is the dedicated situation vector — derived from
    # kv['situation'] by its own backfill scan, NOT an EMBEDDING_GROUPS entry,
    # so the loop above never picks it up. There is no `field_situation` cohort:
    # situation is served by `_situation` via FIELD_VECTOR_FALLBACK (one vector,
    # two consumers — see EMBEDDING_GROUPS). So `_situation` + `high_meta` are
    # the only situation-dependent vectors. Add `_situation` explicitly.
    if field_name == 'situation':
        affected.add('_situation')
    return affected

# KV fields to skip when building embedding text (not semantic content)
EMBEDDING_SKIP_FIELDS = {
    'metadata_created_at', 'validation_count', 'last_validated',
    'alternatives', 'change_impacts',
}

# Max chars per field when building group embedding text
EMBEDDING_FIELD_CHAR_LIMIT = 300

# Dead-handler detection floor. If backfill_vectors finds at least this many
# ELIGIBLE candidates for a group (find_missing now filters to nodes that
# should yield text) but stores ZERO embeddings, the group's handler built no
# text — a dead/missing handler or a total embed failure. This is the exact
# silent-partial that hid edge_context's 0-row bug (a configured 0.55-weighted
# group that never produced a row). The floor avoids false trips on a 1-2 node
# tail; rely on _log_error's rate-limiter for repeat suppression.
EMBEDDING_DEAD_HANDLER_MIN_CANDIDATES = 5


# ═══════════════════════════════════════════════════════════════
# NODE TYPE EXCLUSIONS — which types are excluded from which operations
# Central registry so exclusions don't scatter across files as hardcoded strings.
# ═══════════════════════════════════════════════════════════════

NODE_TYPE_EXCLUSIONS = {
    # Recall: community nodes are S2 navigation/consolidation structure,
    # not surfaceable memories — excluded from the recall
    # pool. Explicit dict filters on type bypass this (deliberate community
    # queries — dashboard, S2, direct asks — still work).
    'recall': {'community'},
}


def get_excluded_types(operation: str) -> set:
    """Get node types excluded from a given operation.

    Args:
        operation: Key from NODE_TYPE_EXCLUSIONS (e.g. 'recall')

    Returns: set of type strings to exclude, or empty set if no exclusions.
    """
    return NODE_TYPE_EXCLUSIONS.get(operation, set())


# ═══════════════════════════════════════════════════════════════
# TRUNCATION LIMITS — per stage, per field
# ═══════════════════════════════════════════════════════════════

PIPELINE = {
    'user_message_query': 4000,
    'assistant_response_store': 8000,  # S0 episode bodies, both conversation roles
}


# ═══════════════════════════════════════════════════════════════
# TRAVERSE — the enriched cluster atom
# ═══════════════════════════════════════════════════════════════
#
# Given seed node IDs, light them up: follow edges (noise excluded),
# attach correction chains, attach metadata. One function, every consumer.
#
# Callers: boot (S0), MCP recall, S1 Surface, get_node/get_nodes.

# Traversal exclusions derive from the noise aspect at load time —
# brain.aspects.structural_exclusions (see AspectRegistry._adopt). The old
# TRAVERSE_EXCLUDED_EDGES literal was a dated snapshot of "what noise meant".


def traverse(brain, seed_ids, depth=1, limit_per_seed=3):
    """The atom: seed IDs → enriched cluster.

    Returns: {
        'neighbors': [{id, type, title, content, edge_type, edge_weight,
                        edge_description, confidence, seed_id}, ...],
        'corrections': {node_id: [{id, title, direction}, ...]},
        'metadata': {node_id: {key: value, ...}},
    }
    """
    mdal = brain._meta_kv

    resolved_ids = {sid for sid in seed_ids if sid}

    if not resolved_ids:
        return {'neighbors': [], 'corrections': {}, 'metadata': {}}

    # ── Graph expansion via GraphDAL (v25 — archived + noise centralized) ──
    graph_dal = brain._graph
    seen = set(resolved_ids)
    neighbors = []
    excluded = set(brain.aspects.traversal_exclusions)

    for full_id in resolved_ids:
        rows = graph_dal.get_neighbors(
            full_id,
            limit=limit_per_seed,
            exclude_relations=excluded,
            exclude_node_ids=seen,
            content_preview_chars=300,
        )
        for r in rows:
            if r['id'] not in seen:
                seen.add(r['id'])
                neighbors.append({
                    "id": r['id'], "type": r['type'], "title": r['title'],
                    "content": r.get('content_preview', ''),
                    "edge_type": r['relation'],
                    "edge_weight": r['weight'],
                    "edge_description": r.get('edge_description') or '',
                    "confidence": r['confidence'],
                    "locked": r['locked'] == 1,
                    "created_at": r['created_at'], "revised_at": r['revised_at'],
                    "direction": r['direction'],
                    "seed_id": full_id,
                })

    # ── Correction chains (seeds + neighbors) — aspect-edge walk
    all_ids = set(resolved_ids)
    for nb in neighbors:
        all_ids.add(nb['id'])

    corrections = brain.correction_enrich(all_ids)

    # ── Metadata for seeds ──
    metadata = {}
    for sid in resolved_ids:
        meta = mdal.get(sid)
        if meta:
            metadata[sid] = meta
            metadata[sid[:8]] = meta

    return {
        'neighbors': neighbors,
        'corrections': corrections,
        'metadata': metadata,
    }
