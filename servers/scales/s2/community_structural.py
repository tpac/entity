"""Community structural fields — derived from community_member edges.

The stored structural fields (COMMUNITY_STRUCTURAL_KEYS) are arithmetic over
the graph, not judgment — never authored by the encoder LLM. Two moments write
them, through one writer (write_structural_fields): the decode seam, from the
values the decoder computes for every community each cycle, and the encoder,
after each batch, for the communities that batch touched.

This module is the single owner of the typed-edge adjacency
(build_member_adjacency) and of the cohesion formula (structural_metrics):
the decoder and the stamp both route through them, so a stored value can
never disagree with the decoder's. See docs/S2-COMMUNITY-DESIGN.md.
"""

from collections import defaultdict

from .community_contract import (
    CORRIDOR_INTERNAL_FRACTION_MAX, CORRIDOR_MIN_SIZE,
    ADJACENCY_EXCLUDED_RELATIONS, ADJACENCY_SKIP_ASPECTS,
    STRUCTURAL_STAMP_CHUNK)


# Aspect names whose edges never count as cohesion.
_SKIP_FAMS = frozenset(ADJACENCY_SKIP_ASPECTS)
_SCOPED_BIND_LIMIT = 800  # above this, scan whole-graph instead of an IN-scope


def structural_metrics(members, edges_by_node):
    """Internal/external edge counts, internal_fraction, is_corridor for one
    community's member set over a typed-edge adjacency.

    `edges_by_node` is the symmetric adjacency the decoder builds: each
    surviving typed edge_relation appears in BOTH endpoints' lists as
    (neighbor, aspect, relation), with community/archived nodes and
    noise/structural relations already excluded. Internal edges are
    double-counted (//2); external are counted once from the member side.

    internal_fraction is RAW (unrounded) — callers round only at the
    display/stamp boundary, so the decoder's exact threshold comparisons are
    preserved. This is the single source of the cohesion formula; both the
    decoder and the structural stamp route through it.
    """
    ms = members if isinstance(members, set) else set(members)
    internal = sum(1 for n in ms
                   for nbr, _, _ in edges_by_node.get(n, [])
                   if nbr in ms) // 2
    external = sum(1 for n in ms
                   for nbr, _, _ in edges_by_node.get(n, [])
                   if nbr not in ms)
    int_frac = internal / (internal + external) if (internal + external) else 0.0
    is_corridor = (int_frac < CORRIDOR_INTERNAL_FRACTION_MAX
                   and len(ms) > CORRIDOR_MIN_SIZE)
    return {
        'internal': internal,
        'external': external,
        'internal_fraction': int_frac,
        'is_corridor': is_corridor,
    }


def build_member_adjacency(brain, member_ids=None):
    """The typed-edge adjacency community detection runs on, optionally scoped.

    Excludes archived/community nodes, the non-cohesion relations, and any
    relation whose aspect is noise/generic_relation, so community detection
    routes on specific semantic edges. Symmetric — each surviving
    edge_relation row is added to BOTH endpoints, so a multi-relation pair is
    counted once per relation. The decoder takes it whole-graph; the stamp
    scopes it.

    member_ids scopes the scan to edges incident to those nodes when the set
    is within SQLite's bind limit; otherwise the whole graph is scanned.
    Extra non-member adjacency is harmless — structural_metrics only walks
    member nodes' lists.

    Sanctioned raw-SQL exception: a graph-wide typed-edge scan has no DAL
    method and fits community detection specifically — give it
    GraphDAL.iter_semantic_edges() only if a caller outside community
    detection appears.
    """
    # PRIMARY (first-claimant) family per relation — the registry's reverse
    # lookup contract. A comprehension over aspects.all() lets the LAST
    # claimant win and flips multi-homed verbs (similar_to: generic_relation →
    # settlement), turning a generic edge into a cohesion edge.
    rel_to_fam = brain.aspects.primary_edge_map()

    excl = ','.join('?' * len(ADJACENCY_EXCLUDED_RELATIONS))
    sql = """
        SELECT e.source_id, e.target_id, er.relation
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        JOIN nodes ns ON ns.id = e.source_id AND ns.archived = 0
            AND ns.type != 'community'
        JOIN nodes nt ON nt.id = e.target_id AND nt.archived = 0
            AND nt.type != 'community'
        WHERE er.archived = 0
        AND er.relation NOT IN (%s)
    """ % excl
    params = tuple(ADJACENCY_EXCLUDED_RELATIONS)
    ids = [m for m in (member_ids or []) if m]
    if ids and len(ids) <= _SCOPED_BIND_LIMIT:
        ph = ','.join('?' * len(ids))
        sql += " AND (e.source_id IN (%s) OR e.target_id IN (%s))" % (ph, ph)
        params = (*params, *ids, *ids)

    edges_by_node = defaultdict(list)
    for src, tgt, rel in brain.conn.execute(sql, params).fetchall():
        fam = rel_to_fam.get(rel, 'unclassified')
        if fam in _SKIP_FAMS:
            continue
        edges_by_node[src].append((tgt, fam, rel))
        edges_by_node[tgt].append((src, fam, rel))
    return edges_by_node


def structural_fields(metrics):
    """The stored structural fields for one community, from its
    structural_metrics(), as the strings node_metadata_kv holds — exactly
    COMMUNITY_STRUCTURAL_KEYS. Two decimals, the precision every reader shows,
    so a drift too small to see writes nothing."""
    return {'community_internal_fraction': str(round(metrics['internal_fraction'], 2))}


def compute_community_structural(brain, community_ids):
    """Derive the stored structural fields for each community from its live
    member edges: {community_id: structural_fields(...)}.

    Pure arithmetic over community_member edges + the typed adjacency — no LLM.
    Matches the decoder's fresh computation exactly: archived members carry no
    edges in the (archived-excluding) adjacency, so they can't shift it.
    """
    ids = [c for c in (community_ids or []) if c]
    if not ids:
        return {}

    members_by_comm = brain._graph.get_members_bulk(ids)
    union = {m['id'] for ms in members_by_comm.values() for m in ms}
    edges_by_node = build_member_adjacency(brain, union if union else None)
    return {cid: structural_fields(structural_metrics(
                {m['id'] for m in members_by_comm.get(cid, [])}, edges_by_node))
            for cid in ids}


def write_structural_fields(brain, dispatch_fn, fields_by_id, stored_by_id,
                            encoding_source):
    """Write each community's structural fields where they differ from what
    it stores, as metadata revises through the unit's encoder dispatch.

    Unchanged values write nothing — no revise, no node_revised trace — so a
    cycle's cost follows what actually moved. Chunked by
    STRUCTURAL_STAMP_CHUNK: every brain_batch holds the write lock every other
    writer waits on, and a fill of the whole community set in one call would
    hold it for the length of the fill. A refused chunk is logged here — the
    dispatch leaves a whole-call refusal to its caller; a refused op inside an
    accepted chunk is logged by the dispatch (batch_op_failed). Returns the
    communities actually written.
    """
    ops = []
    for cid, fields in fields_by_id.items():
        stored = stored_by_id.get(cid) or {}
        changed = {k: v for k, v in fields.items() if stored.get(k) != v}
        if changed:
            ops.append({'op': 'revise', 'node_id': cid,
                        'reason': 'structural fields derived from member edges',
                        **changed})
    written = 0
    for i in range(0, len(ops), STRUCTURAL_STAMP_CHUNK):
        chunk = ops[i:i + STRUCTURAL_STAMP_CHUNK]
        reply = dispatch_fn('brain_batch', {
            'operations': chunk, 'encoding_source': encoding_source}) or {}
        if not reply.get('ok'):
            brain._log_error(
                'community_structural_stamp',
                RuntimeError(str(reply.get('error') or 'no reply')),
                'structural stamp chunk of %d refused' % len(chunk))
            continue
        written += sum(1 for r in (reply.get('result') or {}).get('results', [])
                       if r.get('ok'))
    return written
