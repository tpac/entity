"""S2 Consolidation Decoder — finds and characterizes convergent node clusters.

Phase 1: detection + behavioral enrichment only. No LLM, no writes.
Produces proposals with enough evidence for human review (now)
and future encoder (phase 2).

O = graph embeddings + S1 behavioral traces
K = similarity thresholds + edge families + community membership
Δ = consolidation proposals (written to traces, not to graph)
"""

import json
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone

import numpy as np

from .base import IntegrationUnit
from .consolidation_contract import CONSOLIDATION
from servers.embedder import cosine_similarity


class ConsolidationDecoder(IntegrationUnit):
    NAME = 'consolidation'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:consolidation'

    # brain_meta keys for the scan gate (2026-05-29). One cold-start covers the
    # existing backlog; subsequent runs scan only nodes changed since last run
    # (incremental) and skip when nothing changed. A similarity-threshold
    # change forces a fresh cold-start.
    LAST_RUN_TS_KEY = 's2_consolidation_last_run_ts'
    LAST_THRESHOLD_KEY = 's2_consolidation_last_threshold'

    O_SOURCES = ['graph_embeddings', 's1_recall_traces', 's1_encode_traces']
    K_SOURCES = ['similarity_threshold', 'edge_families', 'community_membership']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or CONSOLIDATION

    def run(self):
        """Find convergent clusters, enrich with behavioral evidence.

        Gating (2026-05-29): the embedding scan is the expensive step. The
        first run (no recorded last-run timestamp) does a full cold-start scan
        that covers the entire existing backlog; afterwards each run scans only
        nodes created/revised since the last run (incremental) and skips
        entirely when nothing changed. A similarity-threshold change forces one
        fresh cold-start so previously sub-threshold pairs are re-evaluated.
        _heal_graph runs every cycle (cheap) — only the scan is gated.

        Returns:
            dict with: clusters (list), stats (dict), skipped (str or None)
        """
        # Step 0: Graph health — archive broken artifacts (cheap, always runs)
        healed = self._heal_graph()

        # Mode selection. Wall-clock is correct here — real time since the last
        # scan, same basis as Brain._maintenance_last_run_ts; not conversation
        # time, so no `at=` anchor.
        raw_ts = self.brain.get_config(self.LAST_RUN_TS_KEY) or '0'
        try:
            last_run_ts = float(raw_ts)
        except (TypeError, ValueError):
            last_run_ts = 0.0
        cur_threshold = str(self.config['similarity_threshold'])
        prev_threshold = self.brain.get_config(self.LAST_THRESHOLD_KEY)
        cold_start = (last_run_ts <= 0) or (prev_threshold != cur_threshold)

        cutoff_iso = (datetime.fromtimestamp(last_run_ts, tz=timezone.utc).isoformat()
                      if last_run_ts > 0 else '')  # clock-ok — gate cutoff, not conversation time
        changed_ids = None
        if not cold_start:
            changed_ids = self._get_changed_node_ids(cutoff_iso)
            if not changed_ids:
                # Nothing changed since the last scan — skip the expensive
                # pass. Leave last_run_ts untouched so changes keep accruing.
                return {'clusters': [], 'stats': {'healed': healed},
                        'skipped': 'no graph change'}

        # Step 1: Find candidate pairs by embedding similarity
        scan_started = time.time()  # clock-ok — idle-cycle wall-clock duration
        candidates, scan_stats = self._scan_embeddings(
            cutoff_iso, cold_start, changed_ids)
        scan_stats['scan_ms'] = int((time.time() - scan_started) * 1000)

        self.trace('O', 'consolidation_candidates',
                   'Scanned %d nodes, found %d pairs above %.2f (%d reviewed, %s %dms)' % (
                       scan_stats['nodes_scanned'],
                       scan_stats['pairs_found'],
                       self.config['similarity_threshold'],
                       scan_stats.get('reviewed_nodes', 0),
                       scan_stats.get('mode', '?'),
                       scan_stats['scan_ms']),
                   metadata=scan_stats)

        # Baseline to record IF the full run completes. The orchestrator
        # stamps this only AFTER the encoder finishes — so a mid-run failure
        # (encoder hang/timeout) leaves the cutoff untouched and the next cycle
        # retries the same work instead of skipping past it. Captured at scan
        # start so changes arriving during a slow encode aren't missed.
        stamp = {'ts': scan_started, 'threshold': cur_threshold}

        if not candidates:
            return {'clusters': [], 'stats': scan_stats, '_stamp': stamp}

        # Step 2: Cluster connected components
        clusters = self._cluster_pairs(candidates)

        # Step 3: Enrich with behavioral evidence
        enriched = self._enrich_clusters(clusters)

        # Step 4: Pre-classify
        for cluster in enriched:
            cluster['pre_class'] = self._pre_classify(cluster)

        # Write K trace with full proposal data
        class_counts = Counter(c['pre_class'] for c in enriched)
        self.trace('K', 'consolidation_proposals',
                   '%d clusters: %s' % (
                       len(enriched),
                       ', '.join('%d %s' % (v, k) for k, v in class_counts.most_common())),
                   metadata={'clusters': self._serialize_clusters(enriched)})

        return {
            'clusters': enriched,
            'stats': {**scan_stats, 'clusters_formed': len(enriched),
                      'class_counts': dict(class_counts),
                      'healed': healed},
            '_stamp': stamp,
        }

    # ══════════════════════════════════════════════════════════
    # Step 0: Graph health
    # ══════════════════════════════════════════════════════════

    def _heal_graph(self):
        """Archive broken graph artifacts. Runs before every scan.

        Currently detects:
        - Community nodes with 0 members (failed edge creation)
        - Superseded handoff nodes (live successor exists) — see below
        """
        archived = []

        # Find community nodes with no active community_member edges.
        # Sanctioned raw-SQL exception: the NOT EXISTS shape doesn't fit any
        # existing GraphDAL method cleanly and this is the only caller.
        # Raw with an archived=0 filter until a second caller surfaces.
        orphan_communities = self.brain.conn.execute("""
            SELECT n.id, n.title FROM nodes n
            WHERE n.type = 'community' AND n.archived = 0
            AND n.locked = 0
            AND NOT EXISTS (
                SELECT 1 FROM edges e
                JOIN edge_relations er ON er.edge_id = e.edge_id
                WHERE (e.source_id = n.id OR e.target_id = n.id)
                AND er.relation = 'community_member'
                AND er.archived = 0
            )
        """).fetchall()

        # Through dispatch, not brain.archive_node directly — the direct call
        # left no trace (plan step 10). UNGUARDED dispatch built on this
        # DECODER instance: heal targets are out-of-cluster by definition,
        # and the ENCODER's archive_guard closure drops out-of-scope archives
        # while reporting success — reusing it would silently stop the heal.
        heal_dispatch = self._make_encoder_dispatch()

        def _archive_via_dispatch(node_id, reason, **extra_op):
            ops = [{'op': 'archive', 'node_id': node_id,
                    'reason': reason, **extra_op}]
            batch = heal_dispatch('brain_batch', {'operations': ops}) or {}
            results = (batch.get('result') or {}).get('results', [])
            return results[0] if results else {'ok': False,
                                               'error': 'no batch result'}

        for nid, title in orphan_communities:
            r = _archive_via_dispatch(nid, 'community with 0 members')
            if r.get('ok'):
                archived.append({'id': nid, 'title': title or '',
                                 'reason': 'community with 0 members'})
                print('[consolidation] Healed: archived orphan community "%s" (%s)' % (
                    title[:50], nid[:8]), flush=True)

        # Superseded handoffs: a handoff (session opener) is a consumable —
        # a directive list addressed to one future session. Once a live
        # successor exists (`supersedes` edge, both endpoints handoff), the
        # predecessor's surface value is negative: it competes with the real
        # opener in recall and, left live, clusters with its successor here —
        # which is how consolidation absorbed a 07-23 opener INTO its 07-21
        # predecessor (survivor-ladder age-bias, journal audit 2026-07-25).
        # Retiring it pre-clustering is deterministic lifecycle, not judgment:
        # archive is soft (content + audit metadata kept), and unlike knowledge
        # types there is no reasoning worth re-surfacing — the successor was
        # written complete. Keyed on the edge, not the type alone, so untyped
        # openers are simply left alone (fail-safe). Locked nodes excluded
        # here to avoid archive_node's guard logging errors on every cycle.
        # Deliberately NOT the correction_improvement aspect: that list is
        # LLM-grown (58 verbs incl. improves/flags/restates) and would
        # silently widen an archive trigger as the classifier files new
        # verbs. Exact replacement semantics only: `supersedes` + its
        # inverse `superseded_by`.
        #
        # The claim's order is ADVISORY here, never trusted for the archive
        # direction: claims keep their writer's order since v35, but one
        # written earlier onto a pair first written the other way can still
        # read reversed until repaired. The edge tells us a supersession
        # relationship EXISTS for the pair; created_at decides
        # which node retires. Older-by-created_at is ground truth for opener
        # chains (a successor is by definition newer). Equal timestamps or
        # self-loops → skip, fail-safe.
        supersession_pairs = self.brain.conn.execute("""
            SELECT s.id, s.created_at, s.locked, s.title,
                   t.id, t.created_at, t.locked, t.title
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes s ON s.id = e.source_id
            JOIN nodes t ON t.id = e.target_id
            WHERE er.relation IN ('supersedes', 'superseded_by')
            AND er.archived = 0
            AND s.type = 'handoff' AND t.type = 'handoff'
            AND s.archived = 0 AND t.archived = 0
            AND s.id != t.id
        """).fetchall()

        seen_pairs = set()
        for row in supersession_pairs:
            a = {'id': row[0], 'created': row[1], 'locked': row[2],
                 'title': row[3]}
            b = {'id': row[4], 'created': row[5], 'locked': row[6],
                 'title': row[7]}
            key = frozenset((a['id'], b['id']))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            if not a['created'] or not b['created'] or a['created'] == b['created']:
                continue  # no date ground truth — never guess an archive
            older, newer = (a, b) if a['created'] < b['created'] else (b, a)
            if older['locked']:
                continue  # quiet skip; don't bounce off archive_node's guard
            reason = 'handoff superseded by %s' % newer['id'][:8]
            r = _archive_via_dispatch(older['id'], reason,
                                      survivor_id=newer['id'])
            if r.get('ok'):
                archived.append({'id': older['id'], 'title': older['title'] or '',
                                 'reason': reason})
                print('[consolidation] Healed: archived superseded handoff "%s" (%s)' % (
                    (older['title'] or '')[:50], older['id'][:8]), flush=True)

        if archived:
            # Use 'O' not 'delta' — heal is an observation, not a consolidation action.
            # 'delta' would pollute _last_run_timestamp() and prevent cold start scan.
            # Distinct ref_type from the embedding-scan O trace so heal is
            # greppable (prior design reused 'consolidation_candidates' for both).
            self.trace('O', 'heal_archive',
                       'Healed %d broken artifacts: %s' % (
                           len(archived),
                           ', '.join(a['title'][:30] for a in archived[:5])),
                       metadata={'healed': archived})

        return archived

    # ══════════════════════════════════════════════════════════
    # Step 1: Embedding scan
    # ══════════════════════════════════════════════════════════

    def _suppression_relations(self):
        """Suppression verbs — the contract's shared derivation.

        Delegates to consolidation_contract.suppression_relations() so the
        decoder scan and the encoder's payload 'Settlement relations' line
        can never disagree.
        """
        from .consolidation_contract import suppression_relations
        return suppression_relations(self.brain)

    def _scan_embeddings(self, last_ts, is_cold_start, changed_ids=None):
        """Find all node pairs above similarity threshold.

        Uses two similarity dimensions:
        - content_cosine: _primary vector (title + content) from node_enrichments
        - title_cosine: title-only vector from node_enrichments

        A pair is a candidate if EITHER dimension exceeds threshold.
        Both scores are reported per pair for the encoder.

        Cold start: full pairwise scan.
        Incremental: new + revised nodes vs all existing.
        """
        threshold = self.config['similarity_threshold']
        suppression = self._suppression_relations()

        # Load content (blend) embeddings — v23: from node_enrichments _primary
        content_rows = self.brain.conn.execute("""
            SELECT ne.node_id, ne.embedding
            FROM node_enrichments ne
            JOIN nodes n ON n.id = ne.node_id
            WHERE n.archived = 0 AND n.type != 'community'
            AND ne.vector_type = '_primary'
            AND ne.embedding IS NOT NULL AND typeof(ne.embedding) = 'blob'
        """).fetchall()

        ids = []
        content_vecs = []
        content_decode_failures = 0
        for nid, emb in content_rows:
            try:
                v = np.frombuffer(emb, dtype=np.float32)
                if len(v) > 0:
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        ids.append(nid)
                        content_vecs.append(v / norm)
            except Exception:
                content_decode_failures += 1

        if content_decode_failures:
            self.brain._log_error(
                's2_consolidation_embedding_decode',
                ValueError('%d content embeddings failed to decode'
                           % content_decode_failures),
                'total scanned=%d; kept=%d' % (len(content_rows), len(ids)))

        if len(content_vecs) < 2:
            return [], {'nodes_scanned': len(content_vecs), 'pairs_found': 0, 'mode': 'empty'}

        id_to_idx = {nid: i for i, nid in enumerate(ids)}

        # Load title embeddings from node_enrichments
        title_vecs_by_id = {}
        title_decode_failures = 0
        for row in self.brain.conn.execute("""
            SELECT node_id, embedding FROM node_enrichments
            WHERE vector_type = 'title'
            AND embedding IS NOT NULL AND typeof(embedding) = 'blob'
        """).fetchall():
            if row[0] in id_to_idx:
                try:
                    v = np.frombuffer(row[1], dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        title_vecs_by_id[row[0]] = v / norm
                except Exception:
                    title_decode_failures += 1

        if title_decode_failures:
            self.brain._log_error(
                's2_consolidation_embedding_decode',
                ValueError('%d title embeddings failed to decode'
                           % title_decode_failures),
                'kept=%d titles out of %d decoded ids'
                % (len(title_vecs_by_id), len(id_to_idx)))

        # Build title matrix aligned with ids (zeros for missing)
        dim = len(content_vecs[0])
        title_vecs = []
        has_title = set()
        for nid in ids:
            if nid in title_vecs_by_id:
                title_vecs.append(title_vecs_by_id[nid])
                has_title.add(nid)
            else:
                title_vecs.append(np.zeros(dim, dtype=np.float32))

        # State-based suppression — "Unreviewed Node" pattern.
        # Following the same design Community detection uses with
        # community_member: a node is "reviewed" iff it has any edge of a
        # suppression-relation type. A cluster is surfaced to the encoder
        # iff at least one member is unreviewed. Nodes whose content
        # changes don't auto-reset (explicit re-review is a separate flow).
        # This replaces pair-level suppression which required N² edges per
        # cluster and relied on prompt compliance to stay consistent.
        already_reviewed = set()
        if suppression:
            already_reviewed = self.brain._graph.nodes_touched_by_relations(suppression)

        content_mat = np.stack(content_vecs)
        title_mat = np.stack(title_vecs)

        def _find_pairs(content_sim, title_sim, row_ids, col_ids):
            """Find pairs where either similarity exceeds threshold.

            Applies the Unreviewed-Node pattern: only pairs with at least one
            unreviewed endpoint survive. Reviewed↔reviewed pairs are skipped
            because both endpoints already have suppression edges from prior
            runs — re-proposing would re-do settled work.
            """
            pairs = []
            yi, xi = np.where(content_sim > threshold)
            pair_set = set()
            for y, x in zip(yi, xi):
                a, b = row_ids[y], col_ids[x]
                if a == b:
                    continue
                if a in already_reviewed and b in already_reviewed:
                    continue
                key = (min(a, b), max(a, b))
                if key not in pair_set:
                    pair_set.add(key)
                    t_sim = float(title_sim[y, x]) if (a in has_title and b in has_title) else 0.0
                    pairs.append((a, b, float(content_sim[y, x]), t_sim))

            # Title matches (may find pairs content missed)
            if title_sim is not None:
                yi, xi = np.where(title_sim > threshold)
                for y, x in zip(yi, xi):
                    a, b = row_ids[y], col_ids[x]
                    if a == b:
                        continue
                    if a in already_reviewed and b in already_reviewed:
                        continue
                    key = (min(a, b), max(a, b))
                    if key not in pair_set:
                        if a in has_title and b in has_title:
                            pair_set.add(key)
                            c_sim = float(content_sim[y, x])
                            pairs.append((a, b, c_sim, float(title_sim[y, x])))
            return pairs

        if is_cold_start:
            content_sim = content_mat @ content_mat.T
            np.fill_diagonal(content_sim, 0)
            title_sim = title_mat @ title_mat.T
            np.fill_diagonal(title_sim, 0)

            # Only keep upper triangle
            content_sim_upper = np.triu(content_sim, k=1)
            title_sim_upper = np.triu(title_sim, k=1)

            pairs_raw = _find_pairs(content_sim_upper, title_sim_upper, ids, ids)
            mode = 'cold_start'
        else:
            if changed_ids is None:
                changed_ids = self._get_changed_node_ids(last_ts)
            if not changed_ids:
                return [], {'nodes_scanned': len(content_vecs), 'pairs_found': 0,
                            'mode': 'incremental', 'changed_nodes': 0}

            # A changed node must always re-evaluate, even if it already carries
            # a suppression edge — exclude it from the reviewed filter so its
            # pairs survive (a revise() can create a new near-duplicate).
            already_reviewed.difference_update(changed_ids)

            changed_indices = [id_to_idx[nid] for nid in changed_ids if nid in id_to_idx]
            if not changed_indices:
                return [], {'nodes_scanned': len(content_vecs), 'pairs_found': 0,
                            'mode': 'incremental', 'changed_nodes': len(changed_ids)}

            changed_content = content_mat[changed_indices]
            changed_title = title_mat[changed_indices]
            changed_id_list = [ids[i] for i in changed_indices]

            content_sim = changed_content @ content_mat.T
            title_sim = changed_title @ title_mat.T

            # Zero out self-matches
            for ci, full_idx in enumerate(changed_indices):
                content_sim[ci, full_idx] = 0
                title_sim[ci, full_idx] = 0

            pairs_raw = _find_pairs(content_sim, title_sim, changed_id_list, ids)
            mode = 'incremental'

        # Deduplicate
        seen = set()
        unique_pairs = []
        for a, b, c_score, t_score in pairs_raw:
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((a, b, c_score, t_score))

        stats = {
            'nodes_scanned': len(content_vecs),
            'title_embeddings': len(has_title),
            'pairs_found': len(unique_pairs),
            'reviewed_nodes': len(already_reviewed),
            'unreviewed_nodes': len(content_vecs) - len(already_reviewed),
            'mode': mode,
        }
        if mode == 'incremental':
            stats['changed_nodes'] = len(changed_ids) if not is_cold_start else 0

        return unique_pairs, stats

    def _get_changed_node_ids(self, since_iso):
        """Node IDs created or revised since the cutoff, from node timestamps.

        Keys on created_at/revised_at — NOT updated_at. revised_at moves only
        on a real revise()/absorb — exactly when re-checking for new
        near-duplicates is warranted; updated_at also moves on non-content
        writes (archive, metadata) that don't change what the dedup scan
        judges. (Historically updated_at was also bumped by the recall
        access-mark drain — fixed 2026-07-27, reads no longer look like
        writes — but revised_at remains the tighter key here.) Community
        nodes are excluded (consolidation scans non-community nodes only).
        Empty cutoff returns the full active set.
        """
        c = self.brain.conn
        if not since_iso:
            rows = c.execute(
                "SELECT id FROM nodes WHERE archived = 0 AND type != 'community'"
            ).fetchall()
            return {r[0] for r in rows}
        rows = c.execute(
            "SELECT id FROM nodes WHERE archived = 0 AND type != 'community' "
            "AND (created_at > ? OR revised_at > ?)",
            (since_iso, since_iso)).fetchall()
        return {r[0] for r in rows}

    # ══════════════════════════════════════════════════════════
    # Step 2: Cluster connected components
    # ══════════════════════════════════════════════════════════

    def _cluster_pairs(self, pairs):
        """Group pairs into connected components, capped at max_cluster_size.

        Each pair carries (a, b, content_cosine, title_cosine).
        """
        max_size = self.config['max_cluster_size']

        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        pair_scores = {}  # (a, b) → {content: float, title: float}
        for a, b, c_score, t_score in pairs:
            union(a, b)
            pair_scores[(min(a, b), max(a, b))] = {
                'content': c_score, 'title': t_score}

        groups = defaultdict(set)
        all_nodes = set()
        for a, b, _, _ in pairs:
            all_nodes.add(a)
            all_nodes.add(b)
        for nid in all_nodes:
            groups[find(nid)].add(nid)

        clusters = []
        oversized_dropped = []
        for members in groups.values():
            if len(members) > max_size:
                oversized_dropped.append(sorted(members))
                continue
            member_list = sorted(members)
            content_scores = []
            title_scores = []
            for i in range(len(member_list)):
                for j in range(i + 1, len(member_list)):
                    key = (min(member_list[i], member_list[j]),
                           max(member_list[i], member_list[j]))
                    if key in pair_scores:
                        content_scores.append(pair_scores[key]['content'])
                        title_scores.append(pair_scores[key]['title'])

            clusters.append({
                'nodes': member_list,
                'size': len(member_list),
                'content_cosine_max': max(content_scores) if content_scores else 0,
                'content_cosine_avg': sum(content_scores) / len(content_scores) if content_scores else 0,
                'title_cosine_max': max(title_scores) if title_scores else 0,
                'title_cosine_avg': sum(title_scores) / len(title_scores) if title_scores else 0,
                'pair_scores': {
                    '%s-%s' % k: v for k, v in pair_scores.items()
                    if k[0] in members and k[1] in members
                },
            })

        # Surface oversized-cluster drops so a bad encoder run or pool
        # saturation is visible instead of silent. Log once per run with
        # sample IDs — the full list is too noisy.
        if oversized_dropped:
            try:
                sample = [m[:3] for m in oversized_dropped[:3]]
                self.brain._log_error(
                    's2_consolidation_oversized_cluster',
                    ValueError(
                        '%d clusters exceeded max_size=%d (dropped)'
                        % (len(oversized_dropped), max_size)),
                    'sample member IDs: %s; total members dropped=%d' % (
                        sample,
                        sum(len(m) for m in oversized_dropped)))
            except Exception as log_err:
                # _log_error itself failing is rare; surface to stderr so
                # daemon.log captures it rather than silently swallowing
                # the visibility we were trying to add. Matches the
                # brain_remember.py:41-44 pattern.
                import sys
                print(
                    '[consolidation] _log_error failed while reporting '
                    'oversized clusters: %r' % log_err,
                    file=sys.stderr, flush=True)

        # Sort by highest similarity (max of either dimension)
        clusters.sort(key=lambda c: -max(c['content_cosine_max'], c['title_cosine_max']))
        return clusters

    # ══════════════════════════════════════════════════════════
    # Step 3: Enrich with behavioral evidence
    # ══════════════════════════════════════════════════════════

    def _enrich_clusters(self, clusters):
        """Add S1 behavioral traces, edge data, and community membership."""
        if not clusters:
            return []

        # Collect all node IDs across clusters
        all_ids = set()
        for c in clusters:
            all_ids.update(c['nodes'])

        # Batch-load node data
        node_data = self._load_node_data(all_ids)

        # Load S1 behavioral data
        recall_data = self._load_recall_data(all_ids)
        catalog_data = self._load_catalog_data(all_ids)

        # Load community membership
        community_map = self._load_community_membership(all_ids)

        # Load edge data per node
        edge_data = self._load_edge_data(all_ids)

        for cluster in clusters:
            nids = cluster['nodes']

            # Node details
            cluster['node_details'] = {
                nid: node_data.get(nid, {}) for nid in nids
            }

            # Recall behavioral signals
            cluster['co_recall_count'] = recall_data['co_recall'].get(
                tuple(sorted(nids)), 0) if len(nids) == 2 else 0
            cluster['judge_preference'] = {
                nid: recall_data['selections'].get(nid, 0) for nid in nids
            }
            cluster['recall_counts'] = {
                nid: recall_data['candidates'].get(nid, 0) for nid in nids
            }
            cluster['query_coverage'] = {
                nid: recall_data['queries'].get(nid, []) for nid in nids
            }

            # Catalog blindness
            cluster['catalog_blind'] = {}
            for nid in nids:
                # Was this node created without seeing the others?
                blind = catalog_data.get(nid, {}).get('blind_to', set())
                cluster['catalog_blind'][nid] = bool(blind & set(nids) - {nid})

            # Edge comparison
            shared_neighbors = set()
            all_neighbors = {}
            for nid in nids:
                neighbors = set(edge_data.get(nid, {}).keys())
                all_neighbors[nid] = neighbors
                if not shared_neighbors:
                    shared_neighbors = neighbors
                else:
                    shared_neighbors &= neighbors
            # Remove cluster members from neighbor sets
            for nid in nids:
                all_neighbors[nid] -= set(nids)
                shared_neighbors -= set(nids)

            cluster['shared_edge_count'] = len(shared_neighbors)
            cluster['unique_edges'] = {
                nid: len(all_neighbors[nid] - shared_neighbors) for nid in nids
            }
            cluster['edge_details'] = {
                nid: edge_data.get(nid, {}) for nid in nids
            }

            # Community membership
            cluster['communities'] = {
                nid: community_map.get(nid, []) for nid in nids
            }
            # Check if any pair shares a community
            comm_sets = {nid: set(c['id'] for c in comms)
                         for nid, comms in cluster['communities'].items()}
            shared_comms = set()
            for nid in nids:
                if not shared_comms:
                    shared_comms = comm_sets.get(nid, set())
                else:
                    shared_comms &= comm_sets.get(nid, set())
            cluster['same_community'] = bool(shared_comms)
            cluster['shared_community_ids'] = list(shared_comms)

            # Correction relationship
            cluster['has_correction_edge'] = self._has_correction_edge(nids)

            # Tension relationship (contradicts, challenges)
            cluster['has_tension_edge'] = self._has_tension_edge(nids)

        return clusters

    def _load_node_data(self, node_ids):
        """Batch-load title, type, content, confidence, encoding_source, timestamps.

        """
        data = {}
        placeholders = ','.join('?' * len(node_ids))
        for row in self.brain.conn.execute("""
            SELECT id, title, type, content, confidence, encoding_source,
                   locked, created_at, updated_at
            FROM nodes WHERE id IN (%s)
        """ % placeholders, list(node_ids)).fetchall():
            data[row[0]] = {
                'title': row[1], 'type': row[2], 'content': row[3],
                'confidence': row[4], 'encoding_source': row[5] or '',
                'locked': bool(row[6]),
                'created_at': row[7] or '', 'updated_at': row[8] or '',
            }

        # Load raw quotes from metadata
        for row in self.brain.conn.execute("""
            SELECT node_id, key, value FROM node_metadata_kv
            WHERE node_id IN (%s)
            AND key IN ('their_raw_quote', 'my_raw_quote', 'situation',
                        'reasoning')
        """ % placeholders, list(node_ids)).fetchall():
            if row[0] in data:
                data[row[0]][row[1]] = row[2]

        return data

    def _load_recall_data(self, node_ids):
        """Load S1R trace data: co-recall, judge preference, queries."""
        lookback = self.config['co_recall_lookback_hours']
        candidates_count = defaultdict(int)  # node_id → times appeared as candidate
        selections_count = defaultdict(int)  # node_id → times selected by judge
        queries_by_node = defaultdict(list)  # node_id → [query strings]
        co_recall = defaultdict(int)         # (sorted node pair) → co-occurrence count

        # Read recall O traces (candidates)
        recall_traces = self._read_traces_window(
            's1', hours=lookback, ref_types=['recall'])
        for t in recall_traces:
            meta = t.get('metadata')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(meta, dict):
                continue

            query = meta.get('query', '')
            cands = meta.get('candidates', [])
            # Candidates are "id|title|score|type" strings
            cand_ids = set()
            for c in cands:
                if isinstance(c, str) and '|' in c:
                    cid = c.split('|')[0]
                    if cid in node_ids:
                        cand_ids.add(cid)
                        candidates_count[cid] += 1
                        if query:
                            queries_by_node[cid].append(query[:100])

            # Co-recall: pairs of our target nodes that appeared together
            target_cands = sorted(cand_ids & node_ids)
            for i in range(len(target_cands)):
                for j in range(i + 1, len(target_cands)):
                    co_recall[tuple(sorted([target_cands[i], target_cands[j]]))] += 1

        # Read surface_selected K traces (judge selections)
        selected_traces = self._read_traces_window(
            's1', hours=lookback, ref_types=['surface_selected'])
        for t in selected_traces:
            ref_id = t.get('ref_id', '')
            if not ref_id:
                continue
            try:
                selected_ids = json.loads(ref_id)
                if isinstance(selected_ids, list):
                    for sid in selected_ids:
                        if sid in node_ids:
                            selections_count[sid] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            'candidates': dict(candidates_count),
            'selections': dict(selections_count),
            'queries': {nid: list(set(qs))[:5] for nid, qs in queries_by_node.items()},
            'co_recall': dict(co_recall),
        }

    def _load_catalog_data(self, node_ids):
        """Check S1E catalog traces for blindness — was node B created without A in catalog?"""
        lookback = self.config['encoding_lookback_hours']
        catalog_data = {}

        catalog_traces = self._read_traces_window(
            's1', hours=lookback, ref_types=['node_catalog'])
        encoding_traces = self._read_traces_window(
            's1', hours=lookback, ref_types=['encoding_run'])

        # Catalog ref_id contains 8-char prefix IDs (comma-separated).
        # Build prefix→full_id lookup for our target nodes.
        prefix_to_full = {}
        for nid in node_ids:
            prefix_to_full[nid[:8]] = nid

        # Build: for each encoding run chain, what full node IDs were in the catalog?
        catalogs_by_chain = {}
        for t in catalog_traces:
            chain = t.get('chain_id', '')
            ref_id = t.get('ref_id', '')
            if ref_id:
                catalog_prefixes = set(ref_id.split(','))
                # Resolve prefixes to full IDs where they match our targets
                catalog_full = set()
                for prefix in catalog_prefixes:
                    prefix = prefix.strip()
                    if prefix in prefix_to_full:
                        catalog_full.add(prefix_to_full[prefix])
                catalogs_by_chain[chain] = catalog_full

        for t in encoding_traces:
            chain = t.get('chain_id', '')
            meta = t.get('metadata')
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(meta, dict):
                continue

            created = meta.get('created', [])
            catalog = catalogs_by_chain.get(chain, set())

            for nid in created:
                if nid in node_ids:
                    # Which of our target nodes were NOT in the catalog when this was created?
                    blind_to = node_ids - catalog - {nid}
                    if blind_to:
                        catalog_data[nid] = {'blind_to': blind_to, 'catalog_size': len(catalog)}

        return catalog_data

    def _load_community_membership(self, node_ids):
        """Load community membership for target nodes via GraphDAL."""
        ids = list(node_ids)
        if not ids:
            return {}
        return self.brain._graph.get_communities_for(ids)

    def _load_edge_data(self, node_ids):
        """Load typed edges per node via GraphDAL.

        Same read exclusion as get_node — the registry's structural_exclusions
        (the noise aspect): community_member and the other plumbing relations
        are not edges the encoder should read or migrate. Community placement
        reaches the cluster block through _load_community_membership's
        `Communities:` line instead.

        Returns {member_id: {neighbor_id: [connection, ...]}} — one
        connection per direction, since a neighbor whose claims point both
        ways ("A supersedes B" beside "B absorbed_into A") is two
        get_connections_bulk entries. Each is the entry unchanged — id, type, title,
        direction, edge_created_at, relations: [{relation, description,
        weight, created_at, updated_at}] — the shape contract.render_edge_lines reads,
        so the encoder passes it through without reshaping. Grouped per
        owner: an edge whose BOTH endpoints are in `node_ids` appears under
        each of them with its own direction (the flat loader assigns such an
        edge to its source only, which left the target member edge-blind
        when its neighbor sat in another cluster of the same run). Nothing
        is truncated: a description a reader might copy as a swap's `old`
        has to arrive whole.
        """
        ids = list(node_ids)
        if not ids:
            return {}
        # archived=0 is the DAL default (v25).
        grouped = self.brain._graph.get_connections_bulk(
            ids, exclude_relations=self.brain.aspects.structural_exclusions)
        out = {}
        for member, conns in grouped.items():
            by_neighbor = out.setdefault(member, {})
            for c in conns:
                by_neighbor.setdefault(c['id'], []).append(c)
        return out

    def _has_correction_edge(self, node_ids):
        """Check if any correction edge exists between cluster members.

        Reads the `correction_improvement` aspect — the same one
        `brain.correction_enrich()` walks, so the decoder and recall agree on
        what counts as "these two already resolved each other".

        SCOPE: correction only. Containment relations (`part_of`, `abstracts`,
        `contains`, ~364 live edges) are NOT correction evidence and are not
        flagged here. They DO reach the encoder now — the intra-cluster edge
        block in the encoder render shows every relation between members,
        verbatim with direction — so the encoder can see "part and its whole"
        even though no boolean flag fires for it.
        """
        correction_rels = set(
            self.brain.aspects.correction_improvement.edge_relations)
        if not correction_rels:
            correction_rels = {'corrects', 'corrected_by', 'supersedes', 'superseded_by'}
        return self.brain._graph.has_edge_between(
            node_ids, node_ids, relations=correction_rels)

    def _has_tension_edge(self, node_ids):
        """Check if any contradiction/challenge edge exists between cluster members.

        Tensions are productive — they represent opposing views on the same topic.
        NEVER consolidate these. Always KEEP both sides of a tension.
        """
        tension_rels = set(self.brain.aspects.contradiction_conflict.edge_relations)
        if not tension_rels:
            tension_rels = {'contradicts', 'challenges', 'conflicts_with',
                            'contrasts', 'undermines', 'violates'}
        return self.brain._graph.has_edge_between(
            node_ids, node_ids, relations=tension_rels)

    # ══════════════════════════════════════════════════════════
    # Step 4: Pre-classify
    # ══════════════════════════════════════════════════════════

    def _pre_classify(self, cluster):
        """Algorithmic pre-classification to guide future encoder.

        Uses both title and content similarity dimensions:
        - High title sim = likely same thing encoded twice (catalog blindness)
        - High content sim = same knowledge, possibly different framing
        - Both high = strongest consolidation signal
        - Type mismatch = needs_judgment, NOT a blanket KEEP — cross-type can be
          one claim (bug+fact = one incident, mechanism+architecture = one system)
          or distinct perspectives; the encoder's claim test decides.
        """
        likely_cosine = self.config.get('likely_consolidate_cosine', 0.90)
        content_max = cluster['content_cosine_max']
        title_max = cluster['title_cosine_max']

        # Correction and tension edges are EVIDENCE for the encoder, not mandates.
        # The encoder sees has_correction_edge and has_tension_edge in the cluster
        # data and reasons about them with full content context.
        # Pre-classifier uses them as soft signals for priority sorting only.

        # Type mismatch → route to needs_judgment (not a blanket KEEP). Pre-classifying
        # cross-type clusters as likely_keep flipped the encoder's burden of proof
        # ("why override the keep?") and was the dominant under-merge cause — corpus
        # targets 5,6,7,8,12,13 were all type_mismatch + likely_keep. needs_judgment
        # hands the call to the claim test, backed by the cross-type-duplicate
        # example in the prompt.
        types = set()
        for nid in cluster['nodes']:
            t = cluster.get('node_details', {}).get(nid, {}).get('type', '')
            if t:
                types.add(t)
        if len(types) > 1:
            # Different types + high similarity → encoder runs the claim test
            type_mismatch = True
        else:
            type_mismatch = False

        # Both dimensions agree (high title + high content) → strong signal
        if title_max >= 0.95 and content_max >= 0.50:
            return 'needs_judgment' if type_mismatch else 'likely_consolidate'

        # High title but content diverges significantly → same name, different knowledge
        # Could be evolution, naming collision, or shallow vs rich
        if title_max >= 0.95 and content_max < 0.50:
            return 'likely_evolve'

        # Very high content similarity (regardless of title)
        if content_max >= 0.95:
            return 'needs_judgment' if type_mismatch else 'likely_consolidate'

        # High similarity + structural signal
        if content_max >= likely_cosine or title_max >= likely_cosine:
            if (cluster['shared_edge_count'] > 0
                    or cluster['same_community']
                    or any(cluster['catalog_blind'].values())):
                return 'needs_judgment' if type_mismatch else 'likely_consolidate'

        return 'needs_judgment'

    # ══════════════════════════════════════════════════════════
    # Serialization (for traces)
    # ══════════════════════════════════════════════════════════

    def _serialize_clusters(self, clusters):
        """Serialize clusters for trace metadata (JSON-safe, within 4000 chars)."""
        serialized = []
        for c in clusters:
            entry = {
                'nodes': c['nodes'],
                'size': c['size'],
                'content_cosine': round(c['content_cosine_max'], 3),
                'title_cosine': round(c['title_cosine_max'], 3),
                'pre_class': c['pre_class'],
                'co_recall_count': c.get('co_recall_count', 0),
                'judge_preference': c.get('judge_preference', {}),
                'catalog_blind': c.get('catalog_blind', {}),
                'shared_edge_count': c.get('shared_edge_count', 0),
                'unique_edges': c.get('unique_edges', {}),
                'same_community': c.get('same_community', False),
                'has_correction_edge': c.get('has_correction_edge', False),
                'has_tension_edge': c.get('has_tension_edge', False),
                'node_titles': {
                    nid: c.get('node_details', {}).get(nid, {}).get('title', '')[:60]
                    for nid in c['nodes']
                },
            }
            serialized.append(entry)
        return serialized
