"""
brain — BrainRecall Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from datetime import datetime
from typing import Any, Dict, Optional
import json
import sys
import time
# Recall result cache (2026-05-08) — short-TTL dedup on top of the
# single-flight gate. 10s window covers multi-hook bursts (pre_edit +
# hook_recall land within seconds of each other) without
# letting suggestions go stale on natural pauses. They all go through this
# single recall layer, so one TTL knob covers every caller.
_RECALL_CACHE_TTL_S = 10.0
_RECALL_CACHE_MAX_ENTRIES = 100


from .brain_constants import (
    MAX_PAGE_SIZE,
    RELEVANCE_FLOOR_PRIMARY,
    NOISE_FLOOR_THRESHOLD,
)
from .db_backends.sqlite import commit_unless_batched


# Node table columns — filter checks these on the result dict.
# Anything not in this set is looked up as a metadata key.
# `project` is not a column: provenance lives in node_metadata_kv, so
# filter {'project': ...} routes to the KV lookup like other promoted fields.
_NODE_COLUMNS = frozenset({
    'id', 'type', 'title', 'content', 'access_count', 'locked', 'archived',
    'emotion', 'emotion_label', 'confidence', 'personal',
    'personal_context', 'evolution_status', 'content_summary',
    'encoding_source', 'revised_at',
    'last_accessed', 'created_at',
})


def _apply_filter(nodes: list, filter_dict: dict, conn) -> list:
    """Apply dict filter to a list of node dicts.

    Filter format:
        {"type": {"in": ["moment", "reflection"]}}
        {"my_raw_quote": {"exists": True}}
        {"content": {"contains": "daemon"}}
        {"confidence": {"gte": 0.9}}
        {"locked": {"equals": 1}}

    Keys in _NODE_COLUMNS are read from the node dict.
    Other keys are looked up in node_metadata_kv.
    """
    if not filter_dict:
        return nodes

    from .dal_metadata import MetadataDAL
    mdal = MetadataDAL(conn)

    def _matches(node):
        for key, conditions in filter_dict.items():
            # Get the value
            if key in _NODE_COLUMNS:
                val = node.get(key)
            else:
                val = mdal.get_field(node['id'], key)

            # Apply conditions
            for op, expected in conditions.items():
                if op == 'exists':
                    if expected and not val:
                        return False
                    if not expected and val:
                        return False
                elif op == 'equals':
                    if val != expected:
                        return False
                elif op == 'in':
                    if val not in expected:
                        return False
                elif op == 'contains':
                    if not val or expected not in str(val):
                        return False
                elif op == 'gte':
                    if val is None or val < expected:
                        return False
                elif op == 'lte':
                    if val is None or val > expected:
                        return False
                else:
                    print('[brain] WARNING: unknown filter operator %r for key %r' % (op, key), file=sys.stderr)
        return True

    out = []
    for n in nodes:
        try:
            if _matches(n):
                out.append(n)
        except Exception as e:
            print('[brain] ERROR in _apply_filter for node %s: %s' % (n.get('id', '?'), e), file=sys.stderr)
    return out


def _rerank_by_relevance(conn, rich_nodes: list, query_text: str, limit: int,
                         vector_type: str = '_primary') -> list:
    """Re-rank a list of rich nodes by cosine similarity to the query embedding.

    Frame Phase 2.5 helper. Hybrid output: top half by relevance, remainder by
    the order the caller passed in (which is the structural sort — recency,
    typically). Lets relevance lift the most-meaningful candidates while
    preserving the structural recency floor.

    vector_type: which `node_enrichments.vector_type` to score against.
    Default `_primary` (canonical single-vector encoding, full coverage). v2
    callers may pass other types or, eventually, request multi-field
    z-weighted scoring (mirroring surface).

    Nodes without an embedding fall through to the recency tail. Cost: 1
    embed call + N cosine ops + 1 SQL query for embeddings (single batch).
    """
    if not query_text or not rich_nodes:
        return rich_nodes[:limit]
    try:
        import numpy as np
        from . import embedder as _emb
        if not _emb.is_ready():
            return rich_nodes[:limit]
        query_blob = _emb.embed_query(query_text)
        if not query_blob:
            return rich_nodes[:limit]
        query_vec = np.frombuffer(query_blob, dtype=np.float32) \
            if isinstance(query_blob, bytes) else np.array(query_blob, dtype=np.float32)
        norm = float(np.linalg.norm(query_vec))
        if norm < 1e-9:
            return rich_nodes[:limit]
        query_vec = query_vec / norm

        # Batch fetch _primary embeddings for all rich nodes
        ids = [n['id'] for n in rich_nodes if n.get('id')]
        if not ids:
            return rich_nodes[:limit]
        placeholders = ','.join('?' * len(ids))
        rows = conn.execute(
            'SELECT node_id, embedding FROM node_enrichments '
            'WHERE node_id IN (%s) AND vector_type = ?' % placeholders,
            ids + [vector_type]).fetchall()
        emb_by_id = {nid: blob for nid, blob in rows}

        # Score
        scored = []
        unscored = []
        for n in rich_nodes:
            blob = emb_by_id.get(n.get('id'))
            if not blob:
                unscored.append(n)
                continue
            nv = np.frombuffer(blob, dtype=np.float32)
            nnorm = float(np.linalg.norm(nv))
            if nnorm < 1e-9:
                unscored.append(n)
                continue
            score = float(np.dot(query_vec, nv / nnorm))
            scored.append((score, n))

        # Hybrid: top half by relevance, remainder by original (structural)
        # order. limit=None (unbounded relevance rank) → every scored node is
        # a slot, so the whole matching set comes back relevance-first.
        relevance_slots = len(scored) if limit is None else max(1, limit // 2)
        relevant = [n for _, n in sorted(scored, key=lambda x: -x[0])[:relevance_slots]]
        relevant_ids = {n.get('id') for n in relevant}
        # Remainder preserves the input order (which is the structural sort)
        remainder = [n for n in rich_nodes if n.get('id') not in relevant_ids]
        return (relevant + remainder)[:limit]
    except Exception as e:
        # Loud but not fatal — rerank failure falls back to structural order
        print('[brain] _rerank_by_relevance failed: %s' % e, file=sys.stderr)
        return rich_nodes[:limit]


# The INFRASTRUCTURE failure modes `_recall_mode` can carry — an empty result
# under one of these means recall could not run, NOT "the brain knows nothing".
# Single source: consumers (hook exits, eval validity marking) import this
# instead of copying the strings, so a new failure mode cannot silently stop
# being flagged downstream.
#
# Not every non-empty mode belongs here, and the distinction is the point:
# `archived_not_supported` is a healthy brain refusing an unimplemented
# request, so marking it degraded would libel a working run. Read this list as
# "recall is broken", never as "recall actually ran" — an exit can be
# deliberate and still return nothing.
RECALL_FAILURE_MODES = ('embedder_unavailable', 'embed_failed',
                        'laf_unavailable')


class BrainRecallMixin:
    """Recall methods for Brain."""

    def _ensure_structural_degree_cache(self) -> None:
        """Build the per-node structural degree cache used by fatigue.

        Idempotent — returns immediately if already built. Called by:
          - the recall hot path (lazy on first use)
          - Brain.warm_up()  (eager at boot, off the user's critical path)

        Reads `edges` × `edge_relations`. The result feeds the fatigue formula
        `K = 10 / (1 + degree/10)` — hubs fatigue fast, peripherals slow.

        Hoisted out of the per-node cosine loop (was an `if not hasattr`
        check inside an N-row loop) so the lazy guard is paid once, not
        N times. Stores the dict on `self._structural_degree_cache`.
        Failures are logged and result in an empty cache — fatigue then
        defaults to maximum K (no dampening), which degrades gracefully.
        """
        if hasattr(self, '_structural_degree_cache'):
            return
        cache: Dict[str, int] = {}
        try:
            for row in self.conn.execute("""
                SELECT node_id, COUNT(*) FROM (
                    SELECT e.source_id as node_id FROM edges e
                    JOIN edge_relations er ON er.edge_id = e.edge_id
                    UNION ALL
                    SELECT e.target_id as node_id FROM edges e
                    JOIN edge_relations er ON er.edge_id = e.edge_id
                ) GROUP BY node_id"""):
                cache[row[0]] = cache.get(row[0], 0) + row[1]
        except Exception as e:
            self._log_error(
                'fatigue_degree_cache', e,
                'building structural degree cache')
        self._structural_degree_cache = cache

    def get_source_refs(self, node_id: str) -> list:
        """Source refs (trace event ids) for one node — the public door to
        SourceRefDAL. Readers (eval, tooling) route here, never _source_refs."""
        return self._source_refs.get_source_refs(node_id) if node_id else []

    def get_source_refs_bulk(self, node_ids) -> dict:
        """Source refs for many nodes in one query — {node_id: [trace ids]},
        nodes without refs absent. The surface attaches these to Haiku's
        picks so the inject can name the conversation behind a memory."""
        return (self._source_refs.get_source_refs_bulk(node_ids)
                if node_ids else {})

    def resolve_live(self, ids, *, on_orphan: str = 'drop'):
        """Owner door for the survivor walk (NodeDAL.resolve_live) — id-SET
        conversion for callers that transform an id collection BEFORE any
        node fetch (candidate feeds, ref lists). Node-returning reads never
        need it: get_node / filter_nodes(field='id') resolve themselves."""
        return self._nodes.resolve_live(ids, on_orphan=on_orphan)

    def get_node(self, node_id_or_ids, follow_absorbed: bool = True):
        """Fully assembled node(s): content + metadata + correction chain + connections.

        Accepts a single node_id (str) or a list of node_ids.
        - Single ID → returns one rich node dict, or None if not found.
        - List of IDs → returns dict keyed by the REQUESTED id. Missing nodes omitted.

        Absorbed ids resolve to their live survivor (the canonical-pull
        contract): an absorb is the store's own identity claim, so an
        id-keyed read of the absorbed id returns the survivor node, keyed
        by the requested id, carrying `_redirected_from` so the renderer
        marks the redirect — the consumer always learns the id moved. A
        RETIRED node (archived, no survivor) returns itself, honestly
        archived. `follow_absorbed=False` is the audit hatch: return the
        absorbed node's own record instead of walking the pointer.
        Writes never follow this redirect — the write doors refuse with
        the pointer instead.

        When given a list, uses batched queries (5 queries total instead of N×4).
        This is the canonical way to get a node. For the bare DB row, use NodeDAL.get_naked_node().
        """
        ndal = self._nodes

        # ── Dispatch: single vs batch ──
        single = isinstance(node_id_or_ids, str)
        raw_ids = [node_id_or_ids] if single else list(node_id_or_ids)

        if not raw_ids:
            return None if single else {}

        full_ids = [nid for nid in raw_ids if nid]

        if not full_ids:
            return None if single else {}

        # ── 1. Batch fetch all nodes (via NodeDAL — single SQL source) ──
        nodes = ndal.get_bulk(full_ids)

        if not nodes:
            return None if single else {}

        # ── 1b. Absorbed-id redirect. Zero extra queries when every hit is
        # live (the common case): detection rides get_bulk's own archived
        # flag; only actual archived hits pay the survivor walk. The stamp
        # is applied per-REQUEST at return time (copy-on-stamp), never onto
        # the shared survivor row — a survivor the caller also asked for
        # directly must not carry another request's redirect banner.
        alias = {}  # requested id -> live terminal id (redirects only)
        if follow_absorbed:
            archived_hits = [nid for nid, n in nodes.items()
                             if n.get('archived')]
            if archived_hits:
                redirected = ndal.resolve_live(archived_hits)['redirected']
                missing = [t for t in set(redirected.values())
                           if t not in nodes]
                if missing:
                    nodes.update(ndal.get_bulk(missing))
                alias = {src: t for src, t in redirected.items()
                         if t in nodes}
                for src in alias:
                    # The corpse leaves the working set — its requested id
                    # re-enters at return time, keyed onto the survivor.
                    nodes.pop(src, None)
                # Orphans (retired / dead chain) keep their corpse row —
                # honest archived render, nothing to resolve to.

        found_ids = list(nodes.keys())

        # ── 2. Batch fetch all metadata via MetadataDAL.get_all_bulk.
        # Single SQL source for KV reads — same DAL the correction-enrich
        # pipeline uses for its scoped fetch.
        meta_by_node = self._meta_kv.get_all_bulk(found_ids)
        for nid in found_ids:
            if nid in meta_by_node:
                nodes[nid]['_metadata'] = meta_by_node[nid]
                # Promote situation to top-level node dict — it's a first-class
                # field consumers expect alongside title/content. Previously
                # a separate SELECT from node_enrichments; v24 collapsed it
                # into the same kv fetch above.
                sit = meta_by_node[nid].get('situation')
                if sit:
                    nodes[nid]['situation'] = sit
                # Promote project the same way — provenance is first-class.
                proj = meta_by_node[nid].get('project')
                if proj:
                    nodes[nid]['project'] = proj

        # ── 4. Batch corrections via aspect-edge walk ──
        # Renderer slices the heavy payload per consumer (HAIKU_FORMAT
        # balanced, ENCODER_FORMAT heavy).
        corrections = self.correction_enrich(found_ids)
        for nid in found_ids:
            node_corrs = corrections.get(nid) or []
            if node_corrs:
                nodes[nid]['_corrections'] = node_corrs

        # ── 5. Batch fetch all connections via GraphDAL (v25) ──
        # DAL centralizes: archived=0 default, direction detection,
        # per-neighbor relation grouping. The DAL holds no policy; the read
        # exclusion is the registry's `structural_exclusions` — the full
        # noise aspect (id:49d734ad: flat reads hide, graph dynamics keep
        # conducting through `traversal_exclusions`). Every reader of a
        # node's connections — Anchor, the recall surface, the encoder
        # catalog, the healer, consolidation — gets the same exclusion here,
        # so none of them re-filters. Community membership is not lost to
        # readers that want it: it rides as its own `communities` attachment.
        connections_by_owner = self._graph.get_connections_bulk(
            found_ids, exclude_relations=self.aspects.structural_exclusions)

        from .contract import relation_age
        for nid in found_ids:
            conns = connections_by_owner.get(nid, [])
            # Order: weight desc, then relation_age desc — the same recency
            # the edge line prints. Weights are flat in practice (0.5/0.6
            # everywhere, id:abbb5b26), so without the second key the top-N
            # cut every reader applies is a tie broken by SQL row order; the
            # newest claim is the better tie-break. The pair's weight is the
            # max over the relations that SURVIVED the exclusion, not the
            # stored aggregate — that one counts the hidden noise rows too,
            # and a 0.6 community_member must not lift a 0.5 pair above its
            # semantic peers. 'relation'/'description' = the top relation,
            # for compat.
            for c in conns:
                rels = sorted(c['relations'],
                              key=lambda r: (r.get('weight') or 0, relation_age(r, c)),
                              reverse=True)
                c['relations'] = rels
                c['relation'] = rels[0]['relation'] if rels else 'related'
                c['description'] = rels[0]['description'] if rels else ''
                # (edge_id, relation) keys the top claim's stored embedding —
                # select_edges reads it instead of re-embedding the text.
                c['edge_id'] = rels[0]['edge_id'] if rels else None
                c['weight'] = max((r.get('weight') or 0 for r in rels),
                                  default=c.get('weight') or 0)
                c['_age'] = max((relation_age(r, c) for r in rels), default='')
            conns.sort(key=lambda x: (x['weight'], x.pop('_age')), reverse=True)
            nodes[nid]['connections'] = conns

        # ── 6. Community membership — the one exception to the noise hide:
        # Anchor and the recall surface want to know which communities a
        # node sits in, but not as edge lines competing for the Edges cut.
        # It rides as its own attachment, [{id, title}], and
        # renders as a `Communities:` line where a format opts in
        # (cfg `communities`: 'title' for most readers, 'ref' for Anchor's tools).
        communities_by_node = self._graph.get_communities_for(found_ids)
        for nid in found_ids:
            nodes[nid]['communities'] = communities_by_node.get(nid, [])

        # ── Return: keyed by the REQUESTED id. A redirected request gets a
        # per-request SHALLOW COPY of the survivor carrying its own
        # REDIRECTED_FROM_KEY (`node['id']` names the survivor); the
        # survivor's own entry, if also requested, stays unstamped. Nested
        # structures (connections, _metadata) are shared across copies —
        # renderers replace keys, they don't mutate in place. ──
        from .contract import REDIRECTED_FROM_KEY
        if single:
            fid = full_ids[0]
            t = alias.get(fid)
            if t is None:
                return nodes.get(fid)
            node = nodes.get(t)
            return (dict(node, **{REDIRECTED_FROM_KEY: [fid]})
                    if node else None)
        if not alias:
            return nodes
        out = {}
        for fid in full_ids:
            t = alias.get(fid)
            if t is None:
                if fid in nodes:
                    out[fid] = nodes[fid]
            elif t in nodes:
                out[fid] = dict(nodes[t], **{REDIRECTED_FROM_KEY: [fid]})
        return out

    def canonicalize_results(self, results, session_id: str = '') -> None:
        """Overlay the canonical pull onto recall results, in place.

        The ONE door that makes a recall result complete. get_node is the
        canonical pull — metadata, situation, corrections, connections — and
        this lays exactly its attachments (CANONICAL_ATTACHMENT_KEYS) over
        each result, leaving recall's own scoring fields untouched. Then the
        session veil scrubs those attachments: a connection or correction
        line carries a neighbor's id and title, and a correction its full
        text, which is the payload a wall exists to stop.

        Every recall door routes through here — by-query, by-id and batch —
        so no caller can hand back a node with its corrections missing.
        Bypassing it is how `recall(node_id=…)` used to return a superseded
        claim with no correction marker attached.

        EVERY result, not a slice. A cap here reads as a cost saving and acts
        as a correctness boundary: the renderer draws whatever it is handed,
        so capped-off results render as authoritative with their correction
        chain silently absent — the exact failure this door exists to remove.
        Measured at ~2.5ms/node, bounded per call by recall's MAX_PAGE_SIZE.
        recall_batch multiplies that by its query count; if that ever needs
        bounding, the lever is the recall LIMIT — how many nodes a caller
        asked for — never a cap on how many of them come back whole.

        A failed pull RAISES rather than degrading. Returning a node whose
        corrections could not be read is indistinguishable, downstream, from
        a node that has none — so silence here would reintroduce the hazard
        in its worst form. The dispatcher turns the raise into a loud error.

        A sessionless caller ('') gets the outward-only veil, the same one
        the by-query door ranks under: the doors resolving the veil
        differently is the divergence this method was written to end.
        """
        from .contract import CANONICAL_ATTACHMENT_KEYS
        from .scopes import scrub_node
        rich = self.get_node([r['id'] for r in results
                              if isinstance(r, dict) and r.get('id')])
        if not rich:
            return
        veil = self.scope_veil(session_id or '')
        for r in results:
            node = rich.get(r.get('id')) if isinstance(r, dict) else None
            if not node:
                continue
            if node.get('id') and node['id'] != r.get('id'):
                # The requested id was absorbed — get_node returned its live
                # survivor (marked). The row BECOMES the survivor wholesale:
                # overlaying only the attachments would dress the corpse in
                # the survivor's corrections and edges, the exact chimera the
                # one-door contract exists to prevent. Recall's own scoring
                # fields aren't node fields, so they survive the update.
                r.update(node)
            for key in CANONICAL_ATTACHMENT_KEYS:
                if key in node:
                    r[key] = node[key]
            scrub_node(r, veil)

    def filter_nodes(self, field: str, include=None, exclude=None,
                     lt=None, gt=None, contains=None, prefix=None, limit: int = 50,
                     sort_by: str = 'created_at', sort_order: str = 'desc',
                     rich: bool = True,
                     relevance_query: str = None,
                     relevance_pool_multiplier: int = 3,
                     relevance_vector_type: str = '_primary',
                     session_id: str = ''):
        """Structured query: filter nodes by any structural field.

        rich=True (default): full content, metadata, corrections, connections
        via batched get_node() — 5 queries regardless of N. The consumer is a
        reasoner; richness is the advantage (see node 9b938b91).
        rich=False: skinny shape (id/title/type/confidence/created_at), for
        discovery scans or feeding IDs to other ops.

        contains: substring LIKE match on `field` (recall's dict-filter speaks
        the same op). prefix: prefix LIKE match (prefix is new here — recall's
        dict-filter has no prefix). limit=None returns every match unbounded
        (e.g. every s2:* node for an internal id-set scan); a number is an
        honest page. The agent-facing cap lives at the dispatch door. Pair
        limit=None with rich=False: with the default rich=True it enriches
        every match (get_node per row), which for a large set is the firehose
        the cap exists to prevent.

        relevance_query (Frame Phase 2.5): when provided, the result is
        re-ranked by semantic relevance to the query text. The DAL pulls a
        wider candidate pool (limit * relevance_pool_multiplier), the brain
        embeds the query once and scores each candidate by cosine similarity
        against its `_primary` embedding, then returns the top-`limit`
        candidates as a hybrid: top half by relevance + remainder by the
        original sort order (deduped).

        Lets consumers (Frame, future agentic recall) ask for "nodes of type
        X most relevant to my current arc" without owning the embedding
        mechanics. Cost: 1 embed call (~50ms) + N cosine ops (cheap matrix
        math). When relevance_query is empty/None, behavior is unchanged.
        """
        from .contract import truncation_payload, REDIRECTED_FROM_KEY
        node_dal = self._nodes

        def _stamp_redirects(node_list):
            if not _absorbed_from:
                return
            for n in node_list:
                srcs = _absorbed_from.get(n.get('id'))
                if srcs:
                    n[REDIRECTED_FROM_KEY] = list(srcs)
        # Widen the pool when relevance ranking is requested. The structural
        # path's truncation detection rides the DAL's exact total_count (an
        # unclamped COUNT(*) on the same WHERE) — no +1 probe: a probe row is
        # eaten by the veil cut and capped by the DAL clamp, both of which
        # silently un-flag saturated results (2026-08-07 review, finding 3).
        # The relevance path is ranked top-k — truncation is its contract.
        # limit=None → unbounded pull (all matches); the pool-widening below
        # (relevance pool, veil backfill) only applies to a finite page.
        dal_limit = limit
        if relevance_query and dal_limit is not None:
            dal_limit = dal_limit * relevance_pool_multiplier
        # Scope veil: filter_nodes is an ambient ENUMERATION surface (a
        # structural sweep, not a reach for a known id) — rich=True would
        # otherwise hand out the walled corpus's full content in one call,
        # and skinny results feed boot's standing items. Over-fetch so
        # walled drops backfill. Sessionless callers ('' — never another
        # session, whose borrowed INWARD veil would be the complement of a
        # wall) get the default-deny outward veil.
        # The veil cut does NOT slice on the relevance path — the widened
        # pool belongs to _rerank_by_relevance, which owns the trim.
        _veil = self.scope_veil(session_id or '')
        if _veil and dal_limit is not None:
            dal_limit = dal_limit * 2
        result = node_dal.filter_nodes(
            field=field, include=include, exclude=exclude,
            lt=lt, gt=gt, contains=contains, prefix=prefix,
            limit=dal_limit, sort_by=sort_by, sort_order=sort_order)
        # Absorbed-id redirect for the id-keyed lookup shape (the canonical-
        # pull contract) — POST-HOC and missing-only, so the all-live case
        # (every boot/Stop ref pull) pays zero extra queries: the DAL's
        # archived=0 already said which ids are gone, and only those pay the
        # survivor walk. Retired/missing ids stay dropped, as before. This
        # door returns a LIST, so a stamp means "these requested ids resolve
        # here" — a survivor already in the page is stamped in place rather
        # than duplicated.
        _absorbed_from = {}  # survivor id -> [requested ids]
        if field == 'id' and include and result.get('nodes') is not None:
            _returned = {n.get('id') for n in result['nodes']}
            _gone = [i for i in include if i and i not in _returned]
            if _gone:
                for _src, _dst in node_dal.resolve_live(
                        _gone)['redirected'].items():
                    _absorbed_from.setdefault(_dst, []).append(_src)
                _need = [t for t in dict.fromkeys(_absorbed_from)
                         if t not in _returned]
                if _need:
                    _rows = node_dal.filter_nodes(
                        field='id', include=_need,
                        limit=len(_need)).get('nodes') or []
                    result['nodes'].extend(_rows)
                    result['total_count'] = (
                        result.get('total_count', 0) + len(_rows))
        if _veil and result.get('nodes'):
            kept = [n for n in result['nodes'] if n.get('id') not in _veil]
            # The relevance path's trim belongs to _rerank_by_relevance; the
            # structural trim happens in the truncation block below.
            result['nodes'] = kept
        if 'error' in result or not result.get('nodes'):
            return result

        # Truncation contract (contract.py): trim the structural pool to
        # `limit`; saturation detection is the DAL's exact, unclamped
        # total_count — veil-independent and alive at every limit. When a
        # veil hides rows, total_count still counts them, so a veiled page
        # can over-flag (report truncated when the hidden remainder is all
        # walled) — conservative in the right direction: isolation governs
        # what rises, not what the count admits exists.
        if not relevance_query and limit is not None:
            result['nodes'] = result['nodes'][:limit]
            if result.get('total_count', 0) > limit:
                result['truncated'] = truncation_payload(
                    limit, result['nodes'])
        # limit is None → unbounded: every matching row is kept and total_count
        # == len(nodes), so there is nothing to trim or flag.

        # Relevance ranking happens BEFORE enrichment. _rerank_by_relevance
        # scores by embedding (looked up by id) + structural order — it needs
        # only ids, not rich content — so we rank the skinny candidate pool and
        # trim to `limit` here, then enrich only the winners below. This is the
        # decoupling: correction-enrich `limit` nodes, not the whole
        # (limit × relevance_pool_multiplier) candidate pool we discard most of.
        nodes = result['nodes']
        if relevance_query:
            nodes = _rerank_by_relevance(
                self.conn, nodes, relevance_query, limit,
                vector_type=relevance_vector_type)
        # No relevance: the truncation block above already trimmed to `limit`.

        if not rich:
            _stamp_redirects(nodes)
            result['nodes'] = nodes
            return result

        # Rich: enrich only the (already-ranked, <= limit) winners.
        ids = [n['id'] for n in nodes]
        rich_map = self.get_node(ids)
        result['nodes'] = [rich_map[i] for i in ids if i in rich_map]
        # After enrichment — get_node returns fresh dicts, so the skinny
        # rows' stamps don't survive it.
        _stamp_redirects(result['nodes'])
        return result

    def query_logs(self, source: str = 'all', hours: int = 24,
                   level: str = 'all', hook_name: str = '',
                   limit: int = 50):
        """Query brain logs: hook errors and debug events.

        Windowed read → truncation contract (contract.py): the DAL already
        counts each source's TRUE window total alongside the limited fetch,
        so saturation detection is exact with no extra query — flagged
        loudly instead of buried in `counts` for the caller to notice.
        """
        result = self._logs_dal.query_logs(
            source=source, hours=hours, level=level,
            hook_name=hook_name, limit=limit)
        from .contract import truncation_payload
        from .dal_logs import LOG_QUERY_MAX_LIMIT
        # Report the EFFECTIVE limit — the DAL clamps at LOG_QUERY_MAX_LIMIT,
        # and a note advising "raise limit" past the cap prescribes an
        # impossible remedy (2026-08-07 review, finding 10).
        effective = min(max(limit, 1), LOG_QUERY_MAX_LIMIT)
        entries = result.get('entries', [])
        total = sum(result.get('counts', {}).values())
        if total > len(entries):
            result['truncated'] = truncation_payload(
                effective, entries,
                reason='%d of %d matching rows returned (effective limit=%d,'
                       ' hard cap %d)' % (len(entries), total, effective,
                                          LOG_QUERY_MAX_LIMIT))
        return result

    def list_interactions(self):
        """List all registered interactions with latest versions."""
        return self._interaction_dal.list_all()

    def list_interaction_versions(self, name: str):
        """Every registered version of one interaction: version + created_by."""
        return self._interaction_dal.list_versions(name)

    def vector_coverage_sweep(self, batch_size: int = 30) -> dict:
        """Repair vector gaps the enqueue path missed; report what it couldn't.

        ONE door, because repair and detection must ask the same question.
        Split across two callers they drifted: the repair passed `model=` and a
        separate probe did not, and `find_missing` counts a stale-model row as
        present without it and missing with it. A model swap — the bulk case,
        the whole corpus — therefore read as "nothing left to do" while
        thousands of rows waited.

        `remaining` is derived from the repair's own per-type counts rather than
        a second query: a type that filled its batch has more behind it. The two
        halves then cannot disagree, and a node that can never be embedded —
        which repairs nothing — can never hold a caller in a re-sweep loop.

        Returns {repaired, by_type, remaining, stuck}. `stuck` is only probed
        when repair achieved nothing, which is the one state worth waking for:
        `_primary` is the LAF-visibility invariant, and a node without it is
        invisible to the field entirely.
        """
        result = self.backfill_vectors(batch_size=batch_size) or {}
        counts = {k: v for k, v in result.items() if isinstance(v, int)}
        repaired = sum(counts.values())
        remaining = any(v >= batch_size for v in counts.values())
        stuck = []
        if not repaired:
            stuck = self._vec_dal.find_missing(
                '_primary', 1, model=embedder.stats.get('model_name', ''))
        return {'repaired': repaired, 'by_type': result,
                'remaining': remaining, 'stuck': stuck}

    def backfill_vectors(self, batch_size: int = 20,
                          node_ids=None) -> dict:
        """Backfill ALL missing vectors for nodes — batched for throughput.

        Primary path: called by embed_queue worker, scoped to a set of
        recently-written node_ids so we don't rescan the graph every tick.
        Safety path: called by S2 Heal during idle, full scan (node_ids=None).

        Each vector type is resolved with one `embed_batch` call (which embeds
        every text alone, so a vector never depends on its batch mates).
        Handles: _primary, _situation, title, high_meta, other_meta, edge_context.

        Returns dict with counts per vector type.
        """
        if not embedder.is_ready():
            return {'error': 'embedder not ready'}

        import time as _time
        _t0 = _time.time()
        print('[backfill_vectors] started (batch_size=%d)' % batch_size, flush=True)

        from .pipeline_contract import (EDGE_DESCRIPTIONS_FIELD,
                                        EMBEDDING_GROUPS, EMBEDDING_SKIP_FIELDS,
                                        EMBEDDING_FIELD_CHAR_LIMIT,
                                        EMBEDDING_DEAD_HANDLER_MIN_CANDIDATES)
        from .dal_metadata import MetadataDAL

        vdal = self._vec_dal
        mdal = self._meta_kv
        model = embedder.stats.get('model_name', '')
        result = {}

        def _store_batch(items, vector_type):
            """items = list of (node_id, text). One embed_batch + one
            executemany store. Returns count of rows written.

            Embedding runs OUTSIDE brain.write_lock — `embedder.embed_batch`
            is CPU-heavy ONNX work that historically held the lock for
            seconds-to-minutes on large batches, blocking every other
            writer (multi-session contention). The lock is acquired
            only around the actual DB write + commit, which is brief.
            write_lock is an RLock, so callers that already hold it
            (none today) wouldn't deadlock.

            For vector_type='_situation', the text column is deprecated — kv
            is the single source of truth. Text is used to generate the
            embedding, then discarded (empty string stored). Other vector_types
            keep the text column (used by dashboard, debugging, recall group
            reconstruction)."""
            if not items:
                return 0
            texts = [t for _, t in items]
            blobs = embedder.embed_batch(texts, kind='document')
            store_text = '' if vector_type == '_situation' else None
            rows = [(nid, vector_type,
                     store_text if store_text is not None else text,
                     blob)
                    for (nid, text), blob in zip(items, blobs)]
            with self.write_lock:
                stored = vdal.store_batch(rows, model=model)
                if stored:
                    # Gate on conn.in_batch — NOT an unconditional commit. A bare
                    # self.conn.commit() here releases ALL open SAVEPOINTs, which
                    # silently broke a content-rewriting absorb run inside a
                    # brain_batch: the embed drain backfilled the survivor's
                    # re-embedded vector mid-merge and its commit killed
                    # `absorb_sp`, so the later RELEASE threw `no such savepoint`
                    # and the merge no-op'd while the batch reported ok. When a
                    # batch/savepoint envelope owns self.conn (in_batch=True), the
                    # owner commits; standalone backfill commits as before.
                    commit_unless_batched(self.conn)
            return stored

        # 1. Primary vectors — nodes missing _primary for the active model
        try:
            missing = vdal.find_missing('_primary', batch_size,
                                        model=model, node_ids=node_ids)
            items = [(n['id'], '%s %s' % (n['title'], n['content'])) for n in missing]
            stored = _store_batch(items, '_primary')
            if stored:
                result['_primary'] = stored
        except Exception as e:
            self._log_error('backfill_primary_scan', e, 'scanning for missing primaries')

        # 2. Situation vectors — nodes with situation in kv but no _situation
        # embedding for the active model. Stale-model rows count as missing.
        # Source: node_metadata_kv['situation'] (canonical, single source of
        # truth). The enrichments row stores the embedding only; its text
        # column is deprecated and written empty for _situation.
        try:
            sit_sql = '''
                SELECT n.id, kv.value as situation_text
                FROM nodes n
                JOIN node_metadata_kv kv
                  ON kv.node_id = n.id AND kv.key = 'situation'
                WHERE n.archived = 0
                  AND kv.value IS NOT NULL AND kv.value != ''
                  AND n.id NOT IN (
                      SELECT node_id FROM node_enrichments
                      WHERE vector_type = '_situation'
                        AND embedding IS NOT NULL
                        AND model = ?
                  )
            '''
            sit_params = [model]
            if node_ids:
                ids = list(node_ids)
                sit_sql += ' AND n.id IN (%s)' % ','.join('?' * len(ids))
                sit_params.extend(ids)
            sit_sql += ' LIMIT ?'
            sit_params.append(batch_size)
            sit_missing = self.conn.execute(sit_sql, sit_params).fetchall()
            items = [(nid, txt) for nid, txt in sit_missing if txt]
            stored = _store_batch(items, '_situation')
            if stored:
                result['_situation'] = stored
        except Exception as e:
            self._log_error('backfill_situation_scan', e, 'scanning for missing situations')

        # 3. Group vectors — title, high_meta, other_meta, edge_context, field cohort
        # Derive which fields live in node_metadata_kv (vs the nodes table) so
        # find_missing can pre-filter to nodes that actually have at least one
        # source field populated. Without this filter, the top-N batch fills
        # with nodes lacking the field, the backfill skips them, and they stay
        # in the missing pool while older nodes-WITH-the-field never reach the
        # front (last_accessed DESC ordering). Result: incomplete coverage even
        # after many passes.
        from .contract import STRUCTURAL_FIELDS as _STRUCT_FIELDS
        nodes_table_fields = set(_STRUCT_FIELDS.keys())

        # edge_context producer policy — one config read per pass, not per node.
        edge_context_top_k = self.get_interaction_config('edge_context')['top_k']
        for group_name, group_config in EMBEDDING_GROUPS.items():
            vector_type = group_config.get('vector_type')
            if not vector_type or vector_type == '_primary':
                continue  # blend is the primary, already handled

            # Identify which fields in this group are kv-stored. Skip:
            #  - fields on the nodes table (always present, no filter needed)
            #  - special markers like `_emergent` and `_edge_descriptions`
            kv_source_keys = [
                f for f in group_config.get('fields', [])
                if f not in nodes_table_fields and not f.startswith('_')
            ]

            # edge_context's only source (_edge_descriptions) lives on edges,
            # not kv — gate find_missing on edge existence so edgeless nodes
            # don't clog the batch and starve the edged ones (see find_missing).
            needs_edge_filter = EDGE_DESCRIPTIONS_FIELD in group_config.get('fields', [])

            try:
                missing = vdal.find_missing(
                    vector_type, batch_size, model=model, node_ids=node_ids,
                    source_kv_keys=kv_source_keys if kv_source_keys else None,
                    require_described_edge=needs_edge_filter,
                    exclude_relations=self._edge_context_excluded())
                if not missing:
                    continue

                items = []
                for node in missing:
                    try:
                        field_values = {'title': node['title'], 'content': node['content']}
                        kv = mdal.get(node['id'])
                        for k, v in kv.items():
                            if k not in EMBEDDING_SKIP_FIELDS and v and str(v).strip():
                                field_values[k] = str(v)

                        parts = []
                        for field in group_config.get('fields', []):
                            if field == '_emergent':
                                continue  # Emergent: any KV field not in other groups
                            if field == EDGE_DESCRIPTIONS_FIELD:
                                # Edge context: descriptions live on edges, not the
                                # node — the DAL producer ranks by edge weight
                                # (both directions; noise/archived/short-desc
                                # filtered there); how many is the `edge_context`
                                # interaction's top_k.
                                for desc in self._graph.get_edge_descriptions_for(
                                        node['id'], limit=edge_context_top_k):
                                    parts.append(desc[:EMBEDDING_FIELD_CHAR_LIMIT])
                                continue
                            val = field_values.get(field)
                            if val and val.strip():
                                parts.append(val[:EMBEDDING_FIELD_CHAR_LIMIT])
                        if not parts:
                            continue
                        items.append((node['id'], '. '.join(parts)))
                    except Exception as e:
                        self._log_error('backfill_group_%s' % vector_type, e,
                                        'node %s' % node['id'][:8])

                stored = _store_batch(items, vector_type)
                if stored:
                    result[vector_type] = stored

                # Loud-by-default dead-handler trip. Two gates keep it precise so
                # it only fires on the real silent-partial that hid edge_context's
                # 0-row bug, not on healthy edge cases:
                #  (a) the group must have an ELIGIBILITY FILTER (kv-key or
                #      described-edge) — only then is a returned candidate a
                #      guarantee it SHOULD yield text. An unfiltered group like
                #      field_content can return nodes with empty content that
                #      legitimately build nothing.
                #  (b) ZERO text built (`not items`), not merely zero stored — a
                #      transient embed failure (embed_batch returns []) builds
                #      items but stores 0; that's an embedder problem, not a dead
                #      handler, and would mis-fire the alarm.
                has_eligibility_filter = bool(kv_source_keys) or needs_edge_filter
                if (has_eligibility_filter
                        and len(missing) >= EMBEDDING_DEAD_HANDLER_MIN_CANDIDATES
                        and not items):
                    self._log_error(
                        'embedding_handler_dead',
                        RuntimeError(
                            'group %s: %d eligible candidates but built no embed '
                            'text — handler is dead/missing'
                            % (vector_type, len(missing))),
                        'embedding backfill dead-handler check')
            except Exception as e:
                self._log_error('backfill_%s_scan' % vector_type, e,
                                'scanning for missing %s vectors' % vector_type)

        elapsed = _time.time() - _t0
        total = sum(v for v in result.values() if isinstance(v, int))
        if total > 0:
            parts = ['%s:%d' % (k, v) for k, v in result.items() if isinstance(v, int) and v > 0]
            print('[backfill_vectors] %d vectors in %.1fs (%s)' % (total, elapsed, ', '.join(parts)), flush=True)
        else:
            print('[backfill_vectors] no missing vectors (%.1fs)' % elapsed, flush=True)

        return result

    def recall(self, query: str, filter: Optional[Dict[str, Any]] = None,
               limit: int = 20,
               include_archived: bool = False,
               session_id: Optional[str] = None, source: str = 'unknown',
               ctx=None, as_of: Optional[str] = None,
               mark_accessed: bool = True,
               cue_trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Recall: LAF field scoring over the live graph. A node is recallable
        once its embedding lands; there is no lexical fallback path.

        Project is provenance in node_metadata_kv: hard-scope with
        filter {"project": {"equals": ...}} (routes to the KV lookup), or let
        the LAF proj lane score the session's project as a soft field.

        Args:
            query: Search query
            filter: Dict filter on node/metadata fields. Examples:
                {"type": {"in": ["moment", "reflection"]}}
                {"my_raw_quote": {"exists": True}}
                {"content": {"contains": "daemon"}}
                {"confidence": {"gte": 0.9}}
            limit: Max results
            include_archived: NOT SUPPORTED — LAF scores the live graph only.
                Passing True returns an empty result with
                _recall_mode='archived_not_supported' (backlog #17).
            session_id: Optional session ID
            cue_trace_id: the trace row that fired this recall (the prompt
                hook passes its first incoming row) — the LAF moment stack's
                window ends just before it. Every other caller passes none.

        Returns:
            Dict with results, _embedding_stats, _recall_mode

        Two layers of dedup (2026-05-08):
        1. Result cache (5s TTL) — repeat identical recalls return cached
           result without re-running. Deepcopy on read so callers are
           isolated.
        2. Single-flight gate — concurrent identical recalls share work:
           one becomes leader, others wait for its result.

        Both keyed by (query, filter, limit, include_archived, session_id,
        as_of, mark_accessed, cue_trace_id). session_id is in the key because
        synaptic fatigue is per-session; cue_trace_id because the moment
        stack's window ends at it.

        Every recall caller (hook_recall, the tool moment, MCP) shares them.
        """
        # Resolve session_id from ctx when caller passed an object; ctx wins
        # over session_id if both supplied (the convention is to pass the object).
        if ctx is not None and not session_id:
            session_id = ctx.session_id

        # Build dedup key from result-affecting params.
        try:
            filter_key = json.dumps(filter, sort_keys=True, default=str) if filter else None
            dedup_key = (
                query, int(min(limit, MAX_PAGE_SIZE)),
                bool(include_archived),
                session_id, filter_key, as_of,
                # mark_accessed is in the key even though it can't change the
                # RESULT: without it an observer's read-only recall could win
                # the single-flight race and a real recall waiting behind it
                # would silently lose its access + fatigue marking.
                bool(mark_accessed),
                cue_trace_id,
            )
        except Exception:
            # If key construction fails, skip dedup — better to do
            # redundant work than crash the recall path.
            dedup_key = None

        # Layer 1: result cache fast path
        if dedup_key is not None:
            cached = self._recall_cache_get(dedup_key)
            if cached is not None:
                return cached

        # Layer 2: single-flight gate
        if dedup_key is not None:
            inflight, leader = self._recall_inflight_acquire(dedup_key)
            if not leader:
                # Wait for leader. .result() blocks until set_result/
                # set_exception fires; deepcopy keeps callers isolated.
                import copy as _copy
                return _copy.deepcopy(inflight.result())

            # Leader path — compute, populate cache, fan out, clean up.
            # Phase 5 (2026-05-18): _run_recall_with_commit removed — recall
            # is now read-only at SQLite (writes deferred to recall_write_queue
            # drained by the bg_writer worker), so no commit step is needed.
            try:
                result = self._recall_impl(
                    query=query, filter=filter, limit=limit,
                    include_archived=include_archived,
                    session_id=session_id, ctx=ctx,
                    source=source, as_of=as_of,
                    mark_accessed=mark_accessed, cue_trace_id=cue_trace_id)
                self._recall_cache_put(dedup_key, result)
                inflight.set_result(result)
                return result
            except Exception as e:
                inflight.set_exception(e)
                raise
            finally:
                self._recall_inflight_release(dedup_key, inflight)

        # Dedup disabled (key construction failed) — fall through.
        return self._recall_impl(
            query=query, filter=filter, limit=limit,
            include_archived=include_archived,
            session_id=session_id, ctx=ctx,
            source=source, as_of=as_of,
            mark_accessed=mark_accessed, cue_trace_id=cue_trace_id)

    # ─── recall result cache (5s TTL) ─────────────────────────────────

    def _recall_cache_get(self, key):
        """Return cached recall result if present + within TTL, else None.

        Returns deepcopy so callers can mutate freely.
        """
        import time as _time
        import copy as _copy
        if not hasattr(self, '_recall_cache'):
            return None
        now = _time.time()
        with self._recall_cache_lock:
            entry = self._recall_cache.get(key)
            if entry is None:
                return None
            result, ts = entry
            if now - ts > _RECALL_CACHE_TTL_S:
                del self._recall_cache[key]
                return None
            return _copy.deepcopy(result)

    def _recall_cache_put(self, key, result) -> None:
        """Cache a recall result with current timestamp. Lazy-init + LRU cap."""
        import time as _time
        import threading
        if not hasattr(self, '_recall_cache'):
            self._recall_cache = {}
            self._recall_cache_lock = threading.Lock()
        now = _time.time()
        with self._recall_cache_lock:
            self._recall_cache[key] = (result, now)
            if len(self._recall_cache) > _RECALL_CACHE_MAX_ENTRIES:
                oldest = min(self._recall_cache,
                             key=lambda k: self._recall_cache[k][1])
                del self._recall_cache[oldest]

    def _recall_cache_purge(self) -> None:
        """Drop every cached recall result — called by
        Brain.invalidate_interaction_caches when a recall-shaping interaction
        (recall_laf) flips or clears. The dedup key
        carries no config fingerprint, so without the purge an identical query
        re-asked within the TTL returns the pre-flip result verbatim."""
        if not hasattr(self, '_recall_cache'):
            return
        with self._recall_cache_lock:
            self._recall_cache.clear()

    def _recall_inflight_acquire(self, key):
        """Return (future, is_leader). Lazy-inits inflight state on first call.

        is_leader=True → caller computes the result + sets the future.
        is_leader=False → caller waits on the existing future for the result.
        """
        import threading
        from concurrent.futures import Future
        if not hasattr(self, '_recall_inflight'):
            self._recall_inflight = {}
            self._recall_inflight_lock = threading.Lock()
        with self._recall_inflight_lock:
            existing = self._recall_inflight.get(key)
            if existing is not None:
                return existing, False
            fut = Future()
            self._recall_inflight[key] = fut
            return fut, True

    def _recall_inflight_release(self, key, fut) -> None:
        """Drop the inflight slot. Defensive: ensure future is set so any
        waiters that arrived AFTER set_result/set_exception don't deadlock
        (shouldn't happen — they'd see existing fut and wait — but the
        cost of guarding is one cheap check)."""
        with self._recall_inflight_lock:
            # Only delete if it's still the same future (defensive against
            # a different recall having taken the slot, which shouldn't
            # be possible but guard anyway).
            if self._recall_inflight.get(key) is fut:
                del self._recall_inflight[key]
        if not fut.done():
            fut.set_exception(RuntimeError(
                'recall leader exited without setting result'))

    def _empty_recall(self, mode: str, note: str = '') -> Dict[str, Any]:
        """The one empty-result shape for recall's no-result exits.

        Every exit carries `_embedding_stats` so the MCP diagnostic footer
        (brain_mcp renders it only when stats is non-empty) shows the failure
        mode inline — an exit without it reads as "brain knows nothing" instead
        of "embedding is broken". `note` is a caller-facing sentence for exits
        the mode string alone doesn't explain (brain_mcp renders it verbatim)."""
        out = {'results': [], '_recall_mode': mode,
               '_embedding_stats': {'embedder_ready': embedder.is_ready(),
                                    'embedder_status': embedder.get_model_status()}}
        if note:
            out['_note'] = note
        return out

    def _recall_impl(self, query: str, filter=None, limit: int = 20,
                     include_archived: bool = False,
                     session_id=None,
                     source: str = 'unknown', ctx=None,
                     as_of: Optional[str] = None,
                     mark_accessed: bool = True,
                     cue_trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Actual recall implementation — hot path. Single-flight wrapper
        in recall() ensures only one of these runs per (query, scope) at
        a time across the daemon."""
        t0 = time.time()
        limit = min(limit, MAX_PAGE_SIZE)

        # Request validation first: an unsupported ARGUMENT must not be
        # answered by infrastructure state. Below the embedder gate this
        # returns embedder_unavailable — a RECALL_FAILURE_MODE — booking a
        # healthy refusal as degraded, and burning an embed + a cache slot
        # on a request that was never going to run.
        # LAF's universe is deliberately live-only (vectors_since /
        # get_all_vectors exclude archived — a performance choice), so it has
        # no scorer for archived nodes. Say so instead of returning a silently
        # live-only result set. Backlog #17 tracks archived-mode support.
        if include_archived:
            self._log_error('recall_archived_unsupported', None,
                            'include_archived=True requested — LAF has no '
                            'archived universe; returning empty')
            return self._empty_recall(
                'archived_not_supported',
                'include_archived=True is not supported — LAF scores the live '
                'graph only, so there is no archived recall path (backlog #17). '
                'Re-run without it for live results.')

        # ── No embedder → no recall. Deliberately NO keyword fallback: a silent
        #    keyword substitute masks that semantic recall is down. Report the
        #    condition (dashboard errors view + boot unsurfaced-errors notice)
        #    and return empty — the recall hook degrades gracefully on empty
        #    results, and the embedder is warmed at daemon boot so this window
        #    is small.
        if not embedder.is_ready():
            self._log_error('recall_embedder_unavailable', None,
                            'embedder not ready — returning empty (no keyword fallback)')
            return self._empty_recall('embedder_unavailable')

        # ── PRIMARY PATH: Embeddings-first ──

        _active_model = embedder.stats.get('model_name') or None

        # STEP 1: Embed the query. No fallback — a failed embed reports the
        # condition and returns empty (see the not-ready branch above).
        try:
            query_vec = embedder.embed_query(query)
        except Exception as e:
            self._log_error('recall_embed_failed', e,
                            'embed_query raised — returning empty (no keyword fallback)')
            return self._empty_recall('embed_failed')
        if not query_vec:
            self._log_error('recall_embed_failed', None,
                            'embed_query returned empty — returning empty (no keyword fallback)')
            return self._empty_recall('embed_failed')

        # STEP 2: LAF scoring. recall_laf computes the full field score per
        # node (maxsim + episodic pick/enc + idf + situation + proj lanes,
        # sigmoid-squashed to (0,1)); STEP 3 injects it as each node's `sim`,
        # and filters, fatigue, the floor, hydration and tracing all run on
        # top of it.
        # Session context — resolved ONCE for the whole recall: the proj
        # lane's query-side project (STEP 2) and the fatigue snapshot
        # (STEP 3) both read it. Previously the ctx-less MCP path loaded the
        # SessionContext twice (session_env_for here + get_or_create_session
        # in STEP 3), a redundant logs-db blob parse per recall.
        _recall_ctx = ctx
        if _recall_ctx is None and session_id:
            try:
                _recall_ctx = self.get_or_create_session(session_id)
            except Exception as _ferr:
                self._log_error('recall_fatigue_ctx', _ferr,
                                'loading session ctx for fatigue dampening — '
                                'falling back to empty fatigue')

        # LAF is the recall scorer. There is no second one: a scorer failure
        # returns empty + a loud mode rather than degrading to a historic
        # path (same rule as the embedder-down exits above).
        try:
            try:
                from .recall_laf import get_engine as _laf_get_engine
            except ImportError:
                from recall_laf import get_engine as _laf_get_engine
            # Session project — the query-side source for the proj lane
            # (deterministic provenance from ctx, never from query text).
            _session_project = (_recall_ctx.project
                                if _recall_ctx is not None else None)
            _laf_scores, _laf_fields = _laf_get_engine(self).scores(
                self, query, query_vec, model=_active_model,
                session_project=_session_project,
                as_of=as_of, session_id=session_id,
                cue_trace_id=cue_trace_id)
        except Exception as _laf_e:
            self._log_error('recall_laf', _laf_e,
                            'laf scoring failed — returning empty (no fallback scorer)')
            return self._empty_recall('laf_unavailable')
        if not _laf_scores:
            self._log_error('recall_laf', None,
                            'laf returned no field scores (empty universe or '
                            'scorer failure) — returning empty')
            return self._empty_recall('laf_unavailable')

        # STEP 3: Inject each embedded node's LAF field score as its
        # similarity, then dampen it by this session's fatigue; STEP 4 applies
        # the floors and boosts on top of it.
        embedding_scores = {}  # node_id → field score
        node_personal_data = {}  # node_id → (personal, personal_context) for pre-sort penalty
        nodes_with_embeddings = 0

        # v9: Brain size for retrieval stats (computed once)
        try:
            _brain_size = self._nodes.count()
        except Exception:
            _brain_size = 0

        # Pre-compute query terms for contextual qualifier matching (applied in STEP 4)
        _query_terms_set = set(query.lower().split()) if query else set()

        try:
            _vec_dal = self._vec_dal
            # Extract type filter from dict filter for embedding scan pre-filter
            _types_filter = None
            if filter and 'type' in filter and 'in' in filter['type']:
                _types_filter = filter['type']['in']
            emb_rows = _vec_dal.get_all_with_context(
                exclude_archived=True,          # refused above if requested
                types=_types_filter,
                model=_active_model)
            # Hoisted out of the per-row loop — these were `if not hasattr`
            # guards behind the cosine inner loop, so paid N times for no
            # reason. Now paid once. Idempotent if Brain.warm_up() already
            # built the structural cache during daemon boot.
            self._ensure_structural_degree_cache()
            # Per-session fatigue snapshot — _recall_ctx was resolved once
            # before STEP 2 (shared with the proj lane's session project).
            # ctx is saved at end of _recall_impl so fatigue increments (via
            # _mark_accessed below) persist for the next recall this session.
            _recall_fatigue: Dict[str, int] = {}
            if _recall_ctx is not None:
                _recall_fatigue = _recall_ctx.fatigue

            for row in emb_rows:
                node_id = row['node_id']
                blob = row['embedding']
                node_personal_data[node_id] = (row['personal'], row['personal_context'])
                if blob:
                    # The field score IS the similarity.
                    # RANGE CONTRACT: values are sigmoid ∈ (0,1) (validated in
                    # recall_laf.scores) — the fatigue multiply below and
                    # STEP 4's floors assume cosine-like magnitudes. A row
                    # outside the field's universe (masked out under as_of)
                    # is not a candidate.
                    sim = _laf_scores.get(node_id)
                    if sim is None:
                        continue

                    # v10: Synaptic fatigue.
                    # Dampens the surprise-score for nodes recalled repeatedly
                    # this session. _recall_fatigue is the per-call snapshot
                    # of this session's ctx.fatigue (loaded above). Resets
                    # between sessions.
                    # K (fatigue resistance) scales with structural degree:
                    #   Hub (30 edges): K=2.5, fatigues fast
                    #   Peripheral (3 edges): K=7.7, fatigues slow
                    #   New node (0 edges): K=10, barely fatigues
                    _fatigue_count = _recall_fatigue.get(node_id, 0)
                    if _fatigue_count > 0:
                        _degree = self._structural_degree_cache.get(node_id, 0)
                        _K = 10.0 / (1.0 + _degree / 10.0)
                        _fatigue = _fatigue_count / (_fatigue_count + _K)
                        sim *= (1.0 - _fatigue)

                    embedding_scores[node_id] = sim
                    nodes_with_embeddings += 1

        except Exception as e:
            print(f'[brain] Embedding scan error: {e}', file=sys.stderr)

        # STEP 4: Score each candidate — its field score, then the
        # contextual-qualifier penalty.
        scored_results = []
        for nid, emb_score in embedding_scores.items():
            blended = emb_score

            # v4 FIX: Contextual qualifier penalty — nodes marked 'contextual' with
            # a personal_context that doesn't overlap query terms get penalized.
            _context_mismatch = False
            _np, _npc = node_personal_data.get(nid, (None, ''))
            if _np == 'contextual' and _npc:
                qualifier_terms = set(_npc.lower().split())
                if not qualifier_terms & _query_terms_set:
                    blended *= 0.7
                    _context_mismatch = True

            # Minimum threshold — don't return noise
            # v9: Raised from 0.05 to NOISE_FLOOR_THRESHOLD (0.15)
            if blended < NOISE_FLOOR_THRESHOLD:
                continue

            scored_results.append({
                'node_id': nid,
                'blended_score': blended,
                'embedding_similarity': round(emb_score * 1000) / 1000,
                '_source': 'laf_v1',
                '_context_mismatch': _context_mismatch,
            })

        # Sort by blended score descending
        scored_results.sort(key=lambda x: -x['blended_score'])

        # Scope veil — BEFORE the [:limit] cut, so eligible lower-ranked
        # candidates backfill the slots walled nodes vacate (gating after
        # truncation starves isolated sessions — the seen_dedup_headroom
        # class, a98143f finding 1). One set-membership check per candidate;
        # empty frozenset (no isolation configured) costs nothing.
        _veil = self.scope_veil(session_id or '')
        _iso_dropped = []
        if _veil:
            _kept = [r for r in scored_results if r['node_id'] not in _veil]
            _iso_dropped = [r['node_id'] for r in scored_results
                            if r['node_id'] in _veil]
            scored_results = _kept

        scored_results = scored_results[:limit]
        # STEP 5: Per-result relevance floor — each result must clear it
        # on its own (v8.7: was all-or-nothing, gated by the top result).
        scored_results = [sr for sr in scored_results
                          if sr['blended_score'] >= RELEVANCE_FLOOR_PRIMARY]

        # STEP 6: Hydrate full node data for top results
        final_results = []
        # One bulk query for the survivors instead of N get_naked_node calls.
        _hydrate_ids = [sr['node_id'] for sr in scored_results]
        try:
            _hydrated = self._nodes.get_bulk(_hydrate_ids) if _hydrate_ids else {}
        except Exception as e:
            self._log_error("recall_hydrate", e,
                            "Failed to bulk-hydrate %d nodes" % len(_hydrate_ids))
            _hydrated = {}
        # Default type exclusions (pipeline_contract registry). An explicit
        # dict filter on type overrides — a deliberate community query must
        # not come back empty because of the default.
        from .pipeline_contract import get_excluded_types
        _excluded_types = get_excluded_types('recall') \
            if not (filter and 'type' in filter) else set()
        for sr in scored_results:
            nid = sr['node_id']
            node = _hydrated.get(nid)

            if node and (node.get('type') or '') in _excluded_types:
                continue
            if node:
                node['effective_activation'] = sr['blended_score']
                node['embedding_similarity'] = sr['embedding_similarity']
                node['_source'] = sr['_source']
                if sr.get('_context_mismatch'):
                    node['_context_mismatch'] = True

                # v4: Brain→Host communication dimensions.
                # The brain expresses WHAT it needs to communicate. The host adapter
                # translates HOW. Four dimensions (all 0-1):
                #   priority: how important (locked=high, evolution=high, regular=medium)
                #   confidence: how certain (locked=1.0, hypothesis=its confidence, regular=0.7)
                #   action_expected: should host act on it? (rule/constraint=yes, context=no)
                #   feedback_needed: does brain need a response? (evolution=yes, fact=no)
                ntype = node.get('type', '')
                is_locked = node.get('locked', False)
                is_evolution = ntype in ('tension', 'hypothesis', 'pattern', 'catalyst', 'aspiration')
                is_rule = ntype in ('rule', 'arch_constraint', 'bug_lesson', 'failure_mode',
                                    'constraint', 'lesson', 'convention')  # v5 engineering memory
                is_cognitive = ntype in ('mental_model', 'reasoning_trace', 'uncertainty',
                                         'correction', 'validation')  # v5 cognitive layer
                is_engineering = ntype in ('purpose', 'mechanism', 'impact', 'concept')  # v5 engineering

                # Temporal freshness — how old is this info?
                _freshness = 'unknown'
                _created = node.get('created_at', '')
                if _created:
                    try:
                        from datetime import timezone
                        _cdt = datetime.fromisoformat(_created.replace('Z', '+00:00'))
                        _age_hours = (datetime.now(timezone.utc) - _cdt).total_seconds() / 3600
                        if _age_hours < 1:
                            _freshness = 'just_now'
                        elif _age_hours < 24:
                            _freshness = 'today'
                        elif _age_hours < 168:
                            _freshness = 'this_week'
                        elif _age_hours < 720:
                            _freshness = 'this_month'
                        else:
                            _freshness = 'older'
                    except Exception as e:
                        self._log_error('freshness_parse', e, 'parsing created_at for freshness classification')

                node['_brain_to_host'] = {
                    'priority': 0.9 if is_locked or is_evolution else (
                        0.8 if is_engineering or is_cognitive else (0.7 if is_rule else 0.5)),
                    'confidence': node.get('confidence') or (1.0 if is_locked else 0.7),
                    'action_expected': is_rule or is_locked or ntype == 'impact',
                    'feedback_needed': is_evolution or ntype in ('failure_mode', 'uncertainty', 'correction'),
                    'freshness': _freshness,
                }

                node['_discovery'] = sr.get('_source', 'embedding')

                # v4: Contextual qualifier matching.
                # Penalty is applied to blended_score BEFORE sorting in STEP 4.
                # Here we only apply confidence reduction and set the qualifier label.
                node_personal = node.get('personal')
                if node_personal == 'contextual':
                    pctx = node.get('personal_context', '')
                    if pctx and query:
                        qualifier_terms = set(pctx.lower().split())
                        overlap = qualifier_terms & _query_terms_set
                        if not overlap:
                            node['_context_mismatch'] = True
                            node['_context_qualifier'] = pctx
                            # Score penalty already applied in STEP 4 — only reduce confidence here
                            node['_brain_to_host']['confidence'] *= 0.6

                # v9: All nodes are primary results (vocab→concept migration complete)
                final_results.append(node)

        # STEP 7: Apply dict filter on final results
        final_results = _apply_filter(final_results, filter, self.conn)

        # (Scope veil already applied pre-[:limit] — candidates were gated
        # before truncation, and _iso_dropped carries the walled ids.)

        # No enrichment here: recall scores and returns, and each door
        # canonicalizes what it hands out (canonicalize_results). The hook
        # path runs its own get_node pass over the capped candidate pool, so
        # enriching a top-3 slice here only ever produced a shape a
        # downstream consumer immediately overwrote.

        # STEP 8: Mark accessed (recognition signal + fatigue)
        # Per-session: _recall_ctx (loaded at top of this function) is passed
        # to _mark_accessed so fatigue increments land on the right session.
        #
        # mark_accessed=False makes this a pure READ. It exists for observers
        # (the dashboard's recall probe) that need the real pipeline's answer
        # without becoming part of the brain's own history: access_count,
        # last_accessed and fatigue are what the graph renders as recall heat
        # (and fatigue dampens this session's next STEP 3), so an observer that
        # marked them would be measuring its own looking. Every in-brain caller leaves this True.
        if mark_accessed:
            for node in final_results:
                try:
                    self._mark_accessed(node['id'], session_id, ctx=_recall_ctx)
                except Exception as _e:
                    self._log_error("recall", _e, "marking node as accessed")

        # Fatigue increments live on the cached SessionContext (mutations
        # in memory). Persistence happens via the daemon's session flush
        # every SESSION_FLUSH_INTERVAL_SECONDS — fatigue is approximate and
        # self-healing, so per-recall saves are unnecessary churn.

        # STEP 9: Build the result. Recall events are traced by the callers
        # (the S1 hook's O/K/Δ traces), not here.
        recall_ms = (time.time() - t0) * 1000
        result = {
            'results': final_results,
            '_recall_mode': 'laf_v1',
            '_embedding_stats': {
                'embedder_ready': True,
                'nodes_with_embeddings': nodes_with_embeddings,
                'recall_ms': round(recall_ms, 1),
                'results_by_source': {
                    'laf_v1': sum(1 for r in final_results
                                  if r.get('_discovery') == 'laf_v1'),
                },
            },
            # v9: Retrieval stats for surfacer — distribution-aware context
            '_retrieval_stats': {
                'brain_size': _brain_size,
                'candidates_after_floor': len(scored_results),
                'top_score': round(scored_results[0]['blended_score'], 3) if scored_results else 0,
                'median_score': round(scored_results[len(scored_results)//2]['blended_score'], 3) if scored_results else 0,
                'source_breakdown': {
                    'laf_v1': sum(1 for sr in scored_results
                                  if sr['_source'] == 'laf_v1'),
                },
            },
        }

        # Scope-veil observability — present only when the wall dropped
        # something, so the legacy result shape is unchanged for unscoped
        # brains.
        if _iso_dropped:
            result['_scope_isolated_dropped'] = len(_iso_dropped)

        # laf telemetry: per-candidate field z-scores for the top nodes —
        # rides the result into the S1R trace so the P2 dataset walker accretes
        # (query, per-field scores, outcome) rows in production for free.
        result['_laf_fields'] = _laf_fields

        # Gap detection: when no results pass the relevance floor,
        # flag it so the voice layer can prompt encoding
        if not final_results and query:
            result['_gap'] = {
                'query': query,
                'top_score': result['_retrieval_stats']['top_score'],
            }

        # Return the query embedding for the recall hook's query-aware edge
        # selection. Zero cost — already computed in STEP 1
        result['_query_embedding'] = query_vec

        return result

    def recall_node(self, node_id: str, session_id: str = '') -> Dict[str, Any]:
        """Recall a specific node by ID with full enrichment.

        Returns same shape as recall() so callers get a
        consistent interface regardless of how the node was found.

        By-id reach is the veil's sanctioned open door — the NODE returns
        regardless of scope. Its ATTACHMENTS are still scrubbed, inside
        canonicalize_results: reaching for a known id says nothing about
        the neighbors and correctors it happens to be linked to.
        """
        node = self._nodes.get_naked_node(node_id)
        if not node:
            return {'results': []}
        # An absorbed id needs no handling here: canonicalize_results below
        # swaps the row to the live survivor (marked) — the one door owns
        # the redirect. A retired corpse passes through, honestly archived.

        # Set display fields for format compatibility
        node['effective_activation'] = 1.0  # a by-id reach is an exact hit
        node['embedding_similarity'] = None
        node['_source'] = 'direct_lookup'

        results = [node]
        self.canonicalize_results(results, session_id)

        return {
            'results': results,
            '_recall_mode': 'by_id',
        }

    def _mark_accessed(self, node_id: str, session_id: str, ctx=None):
        """Enqueue a recognition signal for this node + session pair.

        Architecture (2026-05-18, Phase 5 of bg_writer migration):
        - No DB I/O on the hot path. Enqueues (node_id, session_id, ts)
          into `recall_write_queue`. The background worker drains every
          EMBED_DRAIN_INTERVAL seconds via `brain.conn_bg_writer`,
          producing one atomic +1 UPDATE per unique (node, session)
          pair (dedup'd by Dict semantics within the drain window).
        - Recall is now read-only at SQLite. No `conn_recall_write`,
          no per-recall commit, no busy_timeout exposure on the hot
          path.

        Parallel-session correctness: fatigue increments go to the
        SessionContext passed in by the caller. When ctx is None,
        fatigue is dropped for this access — the access mark still
        enqueues, just no per-session fatigue feedback.
        """
        try:
            from . import recall_write_queue
            recall_write_queue.enqueue_access(node_id, session_id, self.now())
        except Exception as e:
            self._log_error('mark_accessed_enqueue', e,
                            'enqueue failed for node=%s session=%s' %
                            (node_id[:12] if node_id else '', session_id))

        # Increment session fatigue counter — next recall will dampen this
        # node's cosine. ctx is the per-call SessionContext; mutations are
        # saved by the caller at end of recall (see _recall_impl).
        if ctx is not None:
            try:
                ctx.increment_fatigue(node_id)
            except Exception as e:
                self._log_error('mark_accessed_fatigue', e,
                                'fatigue increment failed for node=%s' %
                                (node_id[:12] if node_id else ''))
            # Per-session node activity — parallel-session replacement for
            # global nodes.{last_accessed,access_count}
            # in reads that should be session-scoped (spreading-activation
            # kernel, recency filtering, live-session Frame composition).
            # In-memory; persisted with the SessionContext save at end of
            # recall. Global nodes columns still bumped by recall_write_queue
            # drain for S2 maintenance + dashboard analytics.
            try:
                ctx.bump_node_activity(node_id, self.now())
            except Exception as e:
                self._log_error('mark_accessed_activity', e,
                                'node_activity bump failed for node=%s' %
                                (node_id[:12] if node_id else ''))
