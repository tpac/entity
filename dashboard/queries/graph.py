"""Graph data for the Live tab's canvas graph — nodes, their community,
deduped real edges (no noise-aspect bookkeeping), the live taxonomy's primary
aspect per node type and relation (graph.js colours by aspect), and the
brain's semantic depth axis (graph.js spends it on the Z axis). Carries only
the fields tabs/graph.js reads."""

from ..db import brain_db_path, direct_query
from ..query import safe_query
from .aspects import primary_aspects

# A member's depth leans on its community's topic so a knot stays a knot;
# the rest is the member's own position on the axis.
_KNOT_PULL = 0.65
# Depth is a z-score clipped at this many standard deviations, then scaled
# to [-1, 1].
_DEPTH_CLIP = 2.5
_MIN_KNOT = 3
_POLE_TITLES = 3
_FLAT = ({}, {}, None)

_depth_cache = {'key': None, 'value': None}


def _semantic_depth(db, nodes, node_aspect):
    """The brain's main topic axis: the first principal axis of its community
    centroids over the `_primary` embeddings (of its memories when it has
    fewer than three communities), oriented so identity memories sit on the
    positive side. Returns ({node_id: depth in [-1, 1]}, {community_id:
    {"coherence": rank in [0, 1]}}, {"up": [titles], "down": [titles]}).

    One embedding space: the model most `_primary` vectors were made with, so
    a partial re-embed or an embedder swap never mixes spaces. Coherence is
    each community's mean member-to-centroid cosine, ranked among the brain's
    communities (only once it has three or more — a rank of one or two would
    invent contrast), so graph.js can pack a focused topic tight and let a
    grab-bag spread. Cached until the vectors, the memories or their
    memberships change; a brain without vectors gets no depth and graph.js
    keeps its flat disc."""
    per_model = direct_query(
        "SELECT model, COUNT(*), MAX(created_at) FROM node_enrichments "
        "WHERE vector_type = '_primary' GROUP BY model", db_path=db)
    if not per_model:
        return _FLAT
    model = max(per_model, key=lambda r: r[1])
    shape = hash(tuple((n['id'], n['type'], n['name'], n['community']) for n in nodes))
    key = (db, model, shape)
    if key != _depth_cache['key']:
        _depth_cache.update(key=key, value=_compute_depth(model[0], nodes, node_aspect))
    return _depth_cache['value']


@safe_query('queries.graph', brain_db_path, default=_FLAT)
def _compute_depth(conn, model, nodes, node_aspect):
    import numpy as np

    live = {n['id'] for n in nodes}
    blobs = {nid: blob for nid, blob in conn.execute(
        "SELECT node_id, embedding FROM node_enrichments WHERE vector_type = '_primary' "
        "AND model = ? AND embedding IS NOT NULL", (model,)) if nid in live}
    if not blobs:
        return _FLAT
    V = np.frombuffer(b''.join(blobs.values()), dtype=np.float32).reshape(len(blobs), -1)
    norms = np.linalg.norm(V, axis=1)
    keep = norms > 0
    ids = [nid for nid, k in zip(blobs, keep) if k]
    V = V[keep] / norms[keep, None]
    if len(ids) < _MIN_KNOT:
        return _FLAT
    row = {nid: k for k, nid in enumerate(ids)}

    groups = {}
    for n in nodes:
        if n['community'] and n['id'] in row:
            groups.setdefault(n['community'], []).append(row[n['id']])
    groups = {c: m for c, m in groups.items() if len(m) >= _MIN_KNOT}
    centroid = {c: V[m].mean(axis=0) for c, m in groups.items()}

    by_community = len(centroid) >= _MIN_KNOT
    basis = np.stack(list(centroid.values())) if by_community else V
    mean = basis.mean(axis=0)
    axis = np.linalg.svd(basis - mean, full_matrices=False)[2][0]
    own = dict(zip(ids, ((V - mean) @ axis).tolist()))
    knot = {c: float((v - mean) @ axis) for c, v in centroid.items()}
    identity = [own[n['id']] for n in nodes
                if n['id'] in own and node_aspect.get(n['type']) == 'identity_bearing']
    sign = -1.0 if identity and np.mean(identity) < 0 else 1.0

    raw = {}
    for n in nodes:
        nid, c = n['id'], n['community']
        if nid in knot:
            raw[nid] = knot[nid]
        elif c in knot:
            raw[nid] = (_KNOT_PULL * knot[c] + (1 - _KNOT_PULL) * own[nid]
                        if nid in own else knot[c])
        elif nid in own:
            raw[nid] = own[nid]
    values = np.array(list(raw.values()))
    mid, spread = float(values.mean()), float(values.std()) or 1.0
    depth = {nid: round(max(-_DEPTH_CLIP, min(_DEPTH_CLIP, sign * (r - mid) / spread))
                        / _DEPTH_CLIP, 3)
             for nid, r in raw.items()}

    communities = {}
    if by_community:
        cohesion = {c: float(np.mean(V[m] @ (centroid[c] / np.linalg.norm(centroid[c]))))
                    for c, m in groups.items()}
        ranked = sorted(cohesion, key=cohesion.get)
        communities = {c: {'coherence': round(k / (len(ranked) - 1), 3)}
                       for k, c in enumerate(ranked)}

    titles = {n['id']: n['name'] for n in nodes}
    along = knot if by_community else own
    poles = sorted(along, key=lambda k: sign * along[k])
    depth_axis = {'up': [titles.get(k, k) for k in poles[::-1][:_POLE_TITLES]],
                  'down': [titles.get(k, k) for k in poles[:_POLE_TITLES]]}
    return depth, communities, depth_axis


def query_graph3d():
    """All non-archived nodes with their community, deduped edges, and the
    primary-aspect maps."""
    db = brain_db_path()
    aspects = primary_aspects()

    rows = direct_query(
        "SELECT id, type, title, access_count, created_at, last_accessed "
        "FROM nodes WHERE archived = 0",
        db_path=db,
    )

    # Community membership from active community_member edges
    comm_edges = direct_query(
        "SELECT er.claim_source_id, er.claim_target_id "
        "FROM edge_relations er "
        "JOIN nodes n ON n.id = er.claim_source_id AND n.type = 'community' AND n.archived = 0 "
        "WHERE er.relation = 'community_member' "
        "AND er.archived = 0",
        db_path=db,
    )
    member_to_community = {tgt: src for src, tgt in comm_edges}

    node_ids = set()
    nodes = []
    for nid, ntype, title, access_count, created_at, last_accessed in rows:
        node_ids.add(nid)
        nodes.append({
            "id": nid,
            "name": (title or nid[:8])[:80],
            "type": ntype,
            "access_count": access_count or 1,
            "created_at": created_at,
            "last_accessed": last_accessed,
            "community": member_to_community.get(nid),
        })

    edges = []
    if node_ids:
        placeholders = ",".join("?" * len(node_ids))
        id_list = list(node_ids)
        # Noise-aspect relations (community membership, co-anchoring, …) are
        # bookkeeping the graph never draws or counts; membership already
        # arrives as each node's `community`. They stay in the database.
        noise = [r for r, a in aspects['edge_relations'].items() if a == 'noise']
        skip = ("AND er.relation NOT IN (%s) " % ",".join("?" * len(noise))) if noise else ""
        edge_rows = direct_query(
            "SELECT er.claim_source_id, er.claim_target_id, er.relation, "
            "er.weight, er.created_at "
            "FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE er.claim_source_id IN (%s) AND er.claim_target_id IN (%s) "
            "%sAND er.archived = 0" % (placeholders, placeholders, skip),
            id_list * 2 + noise, db_path=db,
        )
        # One relation per (src, tgt) pair, the one its line shows: the
        # strongest, then the earliest stated.
        best = {}
        for src, tgt, rel, weight, created_at in edge_rows:
            rank = (-(weight or 0), created_at or '', rel)
            best[(src, tgt)] = min(best.get((src, tgt), rank), rank)
        edges = [{"source": s, "target": t, "relation": rank[-1]}
                 for (s, t), rank in best.items()]

    depth, communities, depth_axis = _semantic_depth(db, nodes, aspects['node_types'])
    for n in nodes:
        if n['id'] in depth:
            n['depth'] = depth[n['id']]
    return {"nodes": nodes, "edges": edges, "aspects": aspects,
            "communities": communities, "depth_axis": depth_axis}
