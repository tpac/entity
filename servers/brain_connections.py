"""
brain — BrainConnections Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from typing import Optional


class BrainConnectionsMixin:
    """Connections methods for Brain."""

    def backfill_edge_embeddings(self, edge_ids) -> int:
        """Re-embed edge relations whose stored embedding is NULL — the async,
        centralized counterpart of the old synchronous per-write embed.

        Edge writes only INVALIDATE: GraphDAL.add_relation / rename_relation NULL
        the embedding and enqueue_edge. This pass — driven by the embed_queue
        worker, the SAME mechanism that embeds nodes and traces — recomputes off
        the write hot path. Idempotent: skips rows already embedded. Nothing
        outside the brain layer ever touches embedding (S2 units just mutate the
        graph and stay embedding-ignorant).

        Stored on edge_relations.embedding (v26+), read by select_edges /
        _build_edge_coeffs; recall falls back to live compose_edge_text for any
        row still NULL, so the brief pre-drain window is harmless. Returns the
        number of relations embedded.

        Layer note: lives here (not GraphDAL) because the compute needs the
        embedder + brain.aspects.compose_edge_text. GraphDAL stays storage-only.
        """
        ids = [e for e in (edge_ids or []) if e]
        if not ids:
            return 0
        from . import embedder as _embedder
        if not _embedder.is_ready():
            # Embedder still loading (e.g. at boot before warmup). Don't drop
            # these — re-enqueue so a later drain embeds them. Rows stay NULL
            # meanwhile and recall falls back to live compose, so this is a perf
            # safety net, not data loss.
            from . import embed_queue
            for e in ids:
                embed_queue.enqueue_edge(e)
            return 0
        model = _embedder.stats.get('model_name') or ''
        ph = ','.join('?' * len(ids))
        # Re-embed rows that are unembedded OR embedded by a DIFFERENT model
        # (a model swap makes the stored blob unreadable — the read path filters
        # embedding_model = active — so stale-model rows must re-embed; matches
        # node find_missing semantics).
        rows = self.conn.execute(
            'SELECT edge_id, relation, description FROM edge_relations '
            'WHERE edge_id IN (%s) AND archived = 0 '
            'AND (embedding IS NULL OR embedding_model IS NOT ?)'
            % ph,
            [*ids, model]).fetchall()
        if not rows:
            return 0
        # Compute OUTSIDE the write lock — fastembed is CPU-heavy. 'document'
        # kind matches the prefix used at recall time (_desc_vecs_batched);
        # a mismatched prefix would break the read path's cosine score. Keep the
        # description each blob was computed from for the concurrency guard below.
        # Compose all texts, then embed them in one embed_batch call (mirrors
        # backfill_vectors._store_batch).
        triples = []  # (edge_id, relation, description, text)
        for edge_id, relation, description in rows:
            text = self.aspects.compose_edge_text(relation, description or '')
            if text:
                triples.append((edge_id, relation, description, text))
        if not triples:
            return 0
        try:
            blobs = _embedder.embed_batch([t[3] for t in triples], kind='document')
        except Exception as e:
            self._log_error('edge_embedding_backfill', e, 'batch of %d' % len(triples))
            return 0
        pending = [(eid, rel, desc, blob)            # (edge_id, relation, description, blob)
                   for (eid, rel, desc, _t), blob in zip(triples, blobs)
                   if blob]
        if not pending:
            return 0
        # Write under write_lock (fast) — mirrors backfill_vectors' self-locking.
        # Optimistic guard: only write if the description STILL matches what we
        # embedded (`description IS ?`). A concurrent connect/revise that changed
        # the description has already NULLed + re-enqueued the row; without this
        # guard we'd clobber that fresh-NULL row with the stale blob, and the
        # next drain's filter would no longer see it as needing re-embed —
        # permanently stale geometry. On a description change we match 0 rows and
        # leave the row for the next drain to embed against the new text. The
        # same holds for an archive landing mid-compute: archived rows hold no
        # vector, so `archived = 0` keeps the blob off them.
        written = 0
        with self.write_lock:
            for edge_id, relation, description, blob in pending:
                cur = self.conn.execute(
                    'UPDATE edge_relations SET embedding = ?, embedding_model = ? '
                    'WHERE edge_id = ? AND relation = ? AND description IS ? '
                    'AND archived = 0',
                    (blob, model, edge_id, relation, description))
                written += max(cur.rowcount, 0) if cur.rowcount is not None else 0
            self._maybe_commit()
        return written

    def _edge_context_excluded(self) -> frozenset:
        """Relations the edge_context text and its invalidation both ignore —
        the noise aspect. THE one reader of that policy: GraphDAL resolves its
        eligibility through this (installed as a callable, never copied), the
        backfill candidate filter passes it to find_missing, and the daemon's
        boot re-queue uses it. Read live, because AspectRegistry rebinds
        structural_exclusions on every adopt.

        frozenset() when the registry failed to load: excluding nothing keeps
        the producer and the filter agreeing (both then see every described
        relation), where an AttributeError here would take down the whole
        vector backfill, not just this group.
        """
        aspects = getattr(self, 'aspects', None)
        return aspects.structural_exclusions if aspects is not None else frozenset()

    def _edge_text_changed(self, node_id: str) -> None:
        """GraphDAL.on_edge_text_changed target — an edge write changed the
        descriptions `node_id`'s edge_context vector is embedded from.

        The symmetry with revise(): a node's kv write owns both the field and
        the vector, so revise() invalidates inline. Edge descriptions live on
        a DIFFERENT entity, so no edge writer owns the node's vector — the
        DAL, which every edge write passes through, reports the endpoint and
        this brain-level handler runs the one invalidation path. Registered
        once in Brain.__init__; never called by hand.
        """
        # Fires inside the DAL write — nothing may escape, or a write that
        # already committed reports failure. invalidate_source_fields logs
        # its own failures; this guards the logger itself.
        try:
            from .pipeline_contract import fields_invalidated_by, INVALIDATED_BY_EDGE_WRITE
            self.invalidate_source_fields(
                node_id, fields_invalidated_by(INVALIDATED_BY_EDGE_WRITE),
                origin='edge_write')
        except Exception as e:
            import sys
            print('[brain] edge_write invalidation failed for %s: %s'
                  % (node_id[:8], e), file=sys.stderr)

    def connect(self, source_id: str, target_id: str, relation: str = 'related',
                weight: Optional[float] = None):
        """Add a relation between two nodes (idempotent upsert).

        Stage 1B: add_relation is field-preserving — repeated calls do NOT
        auto-strengthen weight, and unspecified fields preserve existing
        values on update.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation type (e.g., 'related', 'depends_on')
            weight: Edge weight (0-1). None preserves existing on update
                (add_relation's 0.5 on create).

        Returns:
            The result dict from GraphDAL.add_relation() (edge_id, created,
            revived_from_archive, updated, deltas, warnings) — used by callers
            that want to emit trace events.
        """
        graph_dal = self._graph
        # description and an unset weight are omitted so add_relation's
        # sentinel defaults kick in (preserve existing on update; '' and 0.5
        # on create).
        kwargs = {} if weight is None else {'weight': weight}
        result = graph_dal.add_relation(source_id, target_id, relation, **kwargs)
        # Embedding is async: add_relation NULLs + enqueues the EDGE row, and
        # reports changed endpoints to _edge_text_changed (their edge_context);
        # the embed_queue worker re-embeds both. No sync embed here.
        return result

    def connect_typed(self, source_id: str, target_id: str, relation: str = 'related',
                     weight: Optional[float] = None,
                     description: Optional[str] = None,
                     encoding_source: Optional[str] = None):
        """Add a typed relation (idempotent upsert).

        Stage 1B: field-preserving upsert. None defaults mean 'preserve existing
        on update' — pass empty string '' explicitly if you want to clear a
        field. Repeated calls do NOT auto-strengthen weight.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation name (open text — any string)
            weight: Edge weight. None preserves existing on update
                (add_relation's 0.5 on create).
            description: Why this relation exists. None preserves existing on update.
            encoding_source: Who created this edge. None preserves existing on update.

        Returns:
            The result dict from GraphDAL.add_relation() (edge_id, created,
            revived_from_archive, updated, deltas, warnings) — used by callers
            that want to emit trace events.
        """
        # Writes never redirect (the canonical-pull contract): an edge aimed
        # at an absorbed node is a producer holding a stale alias — refuse
        # with the pointer so it re-aims, never silently re-point. Machinery
        # below this door (absorb's own edge migration) keeps its latitude by
        # calling GraphDAL.add_relation directly. Retired endpoints (archived,
        # no survivor) keep add_relation's existing behavior. ONE batched
        # walk answers liveness and redirect for both endpoints — 1 query
        # when both are live, no per-id re-walk.
        from .contract import absorbed_refusal
        for nid, surv in self._nodes.resolve_live(
                [source_id, target_id])['redirected'].items():
            raise ValueError(absorbed_refusal('connect', nid, surv))

        graph_dal = self._graph
        # Build kwargs: only pass explicitly-provided fields so add_relation's
        # sentinel-based field-preservation propagates cleanly through this
        # layer.
        kwargs = {}
        if weight is not None:
            kwargs['weight'] = weight
        if description is not None:
            kwargs['description'] = description
        if encoding_source is not None:
            kwargs['encoding_source'] = encoding_source
        result = graph_dal.add_relation(source_id, target_id, relation, **kwargs)
        # Embedding is async: the edge row is NULLed + enqueued, the endpoints'
        # edge_context invalidated via _edge_text_changed; the worker re-embeds.
        return result

    def revise_edge(self, source_id, target_id, relation,
                    new_relation=None, description=None, weight=None,
                    encoding_source=None, reason=''):
        """Revise an existing edge relation IN PLACE. Mirrors revise()'s contract:
        identify the edge-relation row by (source_id, target_id, relation), then
        update only the fields you pass — omit a field to preserve it.

          - new_relation: rename the relation via GraphDAL.rename_relation (in
            place — keeps the same row, its weight, and created_at; no
            delete+recreate). rename_relation NULLs the stale embedding +
            enqueues the edge; the embed_queue worker re-embeds async (this
            method does no embedding work — embedding is a brain-layer concern).
          - description / weight: field-preserving update via add_relation
            (which likewise invalidates + enqueues on a description change).
          - endpoints' edge_context: GraphDAL reports a changed endpoint to
            _edge_text_changed, which deletes the vector the way revise()
            does. Not this method's job either — same one path.

        Loud (ok=False) on a missing edge / missing relation / rename collision,
        rather than a silent no-op. Returns {ok, edge_id, relation, deltas}.
        """
        gdal = self._graph
        edge_id = gdal.get_edge_id(source_id, target_id)
        if not edge_id:
            return {'ok': False, 'error': 'no edge between %s and %s' % (
                str(source_id)[:8], str(target_id)[:8])}
        active = {r['relation']: r for r in gdal.get_relations(edge_id)}
        if relation not in active:
            return {'ok': False, 'error': 'edge has no active relation %r (has: %s)' % (
                relation, sorted(active))}
        # The pair identifies the claim in either order; the edit keeps the
        # claim's own order. add_relation treats its argument order as the
        # claim's (latest order wins), so the caller's order must not reach it.
        source_id = active[relation]['source_id']
        target_id = active[relation]['target_id']

        # rename_relation and add_relation each self-commit standalone — a
        # rename+update revise would get TWO durability points, and a failure
        # between them persists half the revise. Own the envelope: flip
        # in_batch for the duration (save/restore — we may already be inside
        # a brain_batch that owns it), commit once at the end if we own it.
        _owned_batch = not self.conn.in_batch
        if _owned_batch:
            self.conn.in_batch = True

        try:
            deltas, final_relation, err = self._revise_edge_ops(
                gdal, edge_id, active, source_id, target_id, relation,
                new_relation, description, weight, encoding_source)
        except Exception:
            # A raise between the rename and the field-update must not leave
            # uncommitted writes on the connection — the next batch's
            # entry-flush would silently commit half a revise.
            if _owned_batch:
                self.conn.in_batch = False
                self.conn.rollback()
                # The edge_context hook dropped cache rows eagerly inside the
                # envelope; the rollback restored the DB rows, not the cache.
                self._resync_vector_cache('revise_edge rollback')
            raise
        if _owned_batch:
            self.conn.in_batch = False
            if not err:
                # revise_edge owns the rename+update envelope — one durability
                # point for what was two independent commits pre-step-5.
                self.conn.commit()  # commit-ok: envelope owner
            # No rollback on the err path: the only error return is the rename
            # collision check, which runs SELECTs only — nothing of ours to
            # roll back, and a rollback here would silently destroy a PRIOR
            # leaked transaction's writes (the class brain_batch's entry-flush
            # commits loudly instead). Leaked txns stay open, as pre-step-5.
        if err:
            return err

        # Embedding is handled async: rename_relation and add_relation both NULL
        # the stored embedding + enqueue_edge, and the embed_queue worker
        # re-embeds via backfill_edge_embeddings. revise_edge does no embed work.
        # The claim's own ends, echoed so the trace manifest carries the
        # claim's direction without a second lookup (edge_id is one-way).
        return {'ok': True, 'edge_id': edge_id, 'relation': final_relation,
                'source_id': source_id, 'target_id': target_id,
                'deltas': deltas, 'warnings': []}

    def _revise_edge_ops(self, gdal, edge_id, active, source_id, target_id,
                         relation, new_relation, description, weight,
                         encoding_source):
        """The rename + field-update ops of revise_edge, envelope-agnostic.
        Returns (deltas, final_relation, error_dict_or_None)."""
        deltas = []
        final_relation = relation
        if new_relation and new_relation != relation:
            # Collision must consider ARCHIVED rows too: the (edge_id, relation)
            # primary key spans active + archived, so renaming onto an archived
            # relation name would violate the PK (uncaught IntegrityError).
            all_relations = {r['relation']
                             for r in gdal.get_relations(edge_id, include_archived=True)}
            if new_relation in all_relations:
                return deltas, final_relation, {
                    'ok': False, 'error': 'edge already has relation %r (active or '
                    'archived) — rename would collide on the (edge_id, relation) '
                    'primary key' % new_relation}
            # rename_relation ALWAYS writes encoding_source. Preserve the row's
            # existing provenance when the caller didn't pass one — defaulting to
            # 'anchor' would silently clobber a field the caller never asked to
            # change (e.g. an 's2:reclassify' edge becoming 'anchor' on rename).
            rename_src = (encoding_source if encoding_source is not None
                          else active[relation].get('encoding_source') or '')
            gdal.rename_relation(edge_id, relation, new_relation, rename_src)
            deltas.append({'field': 'relation', 'old': relation, 'new': new_relation})
            final_relation = new_relation

        if description is not None or weight is not None:
            kwargs = {}
            if description is not None:
                kwargs['description'] = description
            if weight is not None:
                kwargs['weight'] = weight
            if encoding_source is not None:
                kwargs['encoding_source'] = encoding_source
            res = gdal.add_relation(source_id, target_id, final_relation, **kwargs)
            deltas.extend(res.get('deltas') or [])

        return deltas, final_relation, None

