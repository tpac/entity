"""S2CD — Community Decoder.

Reads graph structure → produces community proposals. Pure algorithmic,
no LLM calls. Runs in <1s on a 2500-node graph.

Algorithm:
1. Build typed adjacency (edge families, skip noise)
2. Z-score pair scoring within degree buckets
3. Seed clusters from high-z pairs + direct edges
4. Validate: dissolve fragments, flag corridors
5. Compute affinities (new clusters + existing communities)
6. Detect cross-cutting nodes (high-degree, thin spread)
7. Embedding placement for orphans (paused by default: orphan_placement_enabled)
8. Tie analysis (genuine overlap vs split)
9. Build proposals for encoder

Incremental: placed nodes excluded from seeding. New nodes matched
against existing communities. Drift detected when foreign affinity
exceeds max(home affinity, a floor) × the node's drift ratio (default 1.5).
"""

import time
from collections import defaultdict, Counter

import numpy as np

from .base import IntegrationUnit
from .community_contract import COMMUNITY_METADATA_KEYS
from .rejection_table import filter_rejected
from .community_structural import (
    build_member_adjacency, structural_fields, structural_metrics)
from servers.embedder import cosine_similarity


class CommunityDecoder(IntegrationUnit):
    NAME = 'community_detection'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:community_detection'

    O_SOURCES = ['s1_delta', 'graph_structure']
    K_SOURCES = ['semantic_similarity', 'relational_signals', 'usage_traces']

    def __init__(self, brain, dispatch_fn=None, config=None, config_effective=None):
        super().__init__(brain, dispatch_fn)
        # s2_community resolves once per cycle: the code default is
        # community_contract's COMMUNITY_DETECTION unless an override is
        # deployed, and a caller's `config=` (evals) lays over it. The encoder
        # takes this `config_effective` whole, so the config every stage runs
        # on and the stamp the run records come from the same read.
        self.config_effective = (config_effective or
                                 brain.get_interaction_effective(
                                     's2_community', overlay=config))
        self.config = self.config_effective['config']

    def run(self):
        """Decode only — returns proposals for the encoder.

        Returns:
            dict with keys: proposals, stats, community_state, unplaced_count,
                            pending_probes, auto_archive_dead — or
                            {proposals: [], skipped: reason} when nothing waits
        """
        # Always incremental — find unplaced nodes, not trace deltas.
        # Survives partial failures: if encoder placed 50/100 before crash,
        # next run finds the remaining 50.
        community_state = self._read_community_state()

        # Count unplaced nodes for the trace
        already_placed = set()
        for comm in community_state:
            for mid in comm.get('members', []):
                already_placed.add(mid)
        all_active = set(r[0] for r in self.brain.conn.execute(
            "SELECT id FROM nodes WHERE archived = 0 AND type != 'community'"
        ).fetchall())
        unplaced = all_active - already_placed
        unplaced_count = len(unplaced)

        if unplaced_count == 0:
            return {'proposals': [], 'skipped': 'all nodes placed'}

        # ── Phase 2: unplaceable-marking gate ───────────────────────────
        # A node that was examined and couldn't be placed is marked
        # "unplaceable", fingerprinted on its 1-hop neighborhood (each neighbor
        # + that neighbor's community). filter_rejected drops nodes whose
        # fingerprint is unchanged since they were marked, so `pending` is just
        # the genuinely new-or-moved nodes. When none remain the whole decode
        # rests — the old rest condition `unplaced_count == 0` could never
        # reach, since ~28% of nodes never cluster. (The marking itself lives
        # in the pipeline, which sees which pending nodes a surviving proposal
        # actually places.)
        node_to_comm = self._node_to_comm(community_state)
        neighbors = self._load_neighbors(unplaced)
        probes = [{'type': 'unplaceable', 'node_id': nid,
                   'neighborhood': self._neighborhood_str(nid, neighbors, node_to_comm)}
                  for nid in unplaced]
        pending, _suppressed = filter_rejected(self.brain, probes)
        if not pending:
            return {'proposals': [],
                    'skipped': 'all %d unplaced nodes marked unplaceable' % unplaced_count}

        # Cluster detection still scans the full unplaced set (a pending node
        # may cluster with an already-marked one); `pending` drives only the
        # gate and the marking.
        self.trace('O', 's1_delta',
                   '%d pending of %d unplaced, %d communities' % (
                       len(pending), unplaced_count, len(community_state)))

        _t_decode = time.time()  # clock-ok — idle-cycle wall-clock duration
        decode_result = self._decode(community_state)
        proposals = decode_result['proposals']
        decode_ms = int((time.time() - _t_decode) * 1000)

        if not proposals:
            self.trace('K', 'community_proposals',
                       'No proposals generated (decode %dms)' % decode_ms)

        # Rich trace with decode state
        if proposals:
            self.trace('K', 'community_proposals',
                       '%d proposals: %d clusters, %d affinities, %d cross-cutting (decode %dms)' % (
                           len(proposals),
                           decode_result['stats'].get('valid_clusters', 0),
                           decode_result['stats'].get('nodes_with_affinities', 0),
                           decode_result['stats'].get('cross_cutting', 0),
                           decode_ms),
                       metadata={
                           'decode_ms': decode_ms,
                           'decode_stats': decode_result['stats'],
                           'cluster_summaries': decode_result.get('cluster_summaries', []),
                       })

        return {
            'proposals': proposals,
            'stats': decode_result.get('stats', {}),
            'community_state': community_state,
            'unplaced_count': unplaced_count,
            'pending_probes': pending,
            'auto_archive_dead': decode_result.get('auto_archive_dead', []),
            'structural': decode_result.get('structural', {}),
        }

    # ══════════════════════════════════════════════════════════
    # Phase 2: unplaceable-marking helpers
    # ══════════════════════════════════════════════════════════

    def _node_to_comm(self, community_state):
        """Map each node to a deterministic sorted-join of ALL its community
        ids. NOT a dict last-wins comprehension: a node in >1 community would
        then take whichever community community_state listed last, and the
        SELECT behind community_state has no ORDER BY — so the mapping (and the
        node's fingerprint) could flip between runs and wake it spuriously.
        Sorting all of a node's communities makes it order-independent.
        """
        nc = {}
        for c in community_state:
            for m in c.get('members', []):
                nc.setdefault(m, set()).add(c['id'])
        return {m: ','.join(sorted(cids)) for m, cids in nc.items()}

    def _load_neighbors(self, node_ids):
        """neighbor-id set per node, both edge directions, active non-noise.

        A cheap 1-hop load for the unplaceable fingerprint — not the typed
        adjacency _decode builds. Noise/generic relations (e.g. co_anchored,
        related_to) are excluded so they can't churn a node's placeability
        fingerprint and wake it spuriously.
        """
        result = {nid: set() for nid in node_ids}
        if not result:
            return result
        try:
            noise = set(self.brain.aspects.relations_in(['noise', 'generic_relation']))
        except Exception as e:
            # Don't swallow: an empty noise set INCLUDES noise edges in the
            # fingerprint, churning it. Log so the (rare) transient churn is
            # visible rather than a silent spurious-wakeup wave.
            noise = set()
            self.brain._log_error('s2_community_noise_aspect_load', e,
                                  'unplaceable fingerprint may churn this cycle')
        ids = list(node_ids)
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            ph = ','.join('?' * len(chunk))
            rows = self.brain.conn.execute(
                "SELECT e.source_id, e.target_id, er.relation "
                "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
                "WHERE er.archived = 0 "
                "AND (e.source_id IN (%s) OR e.target_id IN (%s))" % (ph, ph),
                chunk + chunk).fetchall()
            for src, tgt, rel in rows:
                if rel in noise:
                    continue
                if src in result:
                    result[src].add(tgt)
                if tgt in result:
                    result[tgt].add(src)
        return result

    def _neighborhood_str(self, nid, neighbors, node_to_comm):
        """Stable 1-hop fingerprint basis for a node: each neighbor + that
        neighbor's community ('' if unplaced), sorted. It changes iff the node
        gains/loses an edge or a neighbor's community assignment flips — the
        only things that change whether the node can cluster or attach.
        compute_fingerprint() hashes this into the rejection fingerprint.
        """
        items = sorted((nbr, node_to_comm.get(nbr, ''))
                       for nbr in neighbors.get(nid, ()))
        return ';'.join('%s>%s' % (nbr, comm) for nbr, comm in items)

    # ══════════════════════════════════════════════════════════
    # _read_community_state
    # ══════════════════════════════════════════════════════════

    def _read_community_state(self):
        communities = []
        rows = self.brain.conn.execute(
            "SELECT id, title, content, confidence "
            "FROM nodes WHERE type = 'community' AND archived = 0 "
            "AND encoding_source = ?",
            (self.ENCODING_SOURCE,)).fetchall()

        # The stored community fields, one bulk query. `meta` holds the raw
        # stored strings — the values as of the last write, before this cycle
        # recomputes anything.
        meta_by_node = self.brain._meta_kv.get_fields_bulk(
            [r[0] for r in rows], list(COMMUNITY_METADATA_KEYS))

        for nid, title, content, conf in rows:
            # Member IDs via GraphDAL (archived=0 default, v25).
            member_dicts = self.brain._graph.get_community_members(
                nid, require_active_member=False)
            members = {m['id'] for m in member_dicts}

            communities.append({
                'id': nid, 'title': title, 'content': content,
                'confidence': conf,
                'members': members,
                'meta': meta_by_node.get(nid, {}),
            })

        return communities

    # ══════════════════════════════════════════════════════════
    # _decode — the 9-step pipeline
    # ══════════════════════════════════════════════════════════

    def _typed_adjacency(self):
        """Step 1: the typed adjacency over ALL nodes (placed nodes' edges
        matter) and each node's typed neighbours — from the one builder the
        structural stamp also reads, so a stored value and the decoder's own
        number cannot disagree."""
        edges_by_node = build_member_adjacency(self.brain)
        return edges_by_node, {nid: {nbr for nbr, _, _ in el}
                               for nid, el in edges_by_node.items()}

    def _decode(self, community_state):
        edges_by_node, typed_neighbors = self._typed_adjacency()

        if len(edges_by_node) < self.config['min_community_size']:
            return {'proposals': [], 'stats': {}}

        # Identify already-placed nodes (members of existing communities)
        already_placed = set()
        for comm in community_state:
            already_placed.update(comm['members'])

        unplaced = set(edges_by_node.keys()) - already_placed

        # Step 2: Z-score pairs — only pairs with at least one UNPLACED node
        pair_zscores, degrees, bucket_stats = self._compute_pair_scores(
            typed_neighbors, edges_by_node)

        # Always filter to pairs with at least one unplaced node
        pair_zscores = {
            (a, b): z for (a, b), z in pair_zscores.items()
            if a in unplaced or b in unplaced
        }

        # Step 3: Seed clusters from unplaced nodes
        direct_pairs = self._get_direct_pairs(edges_by_node)
        direct_pairs = {
            (a, b) for a, b in direct_pairs
            if a in unplaced or b in unplaced
        }
        clusters = self._seed_clusters(pair_zscores, direct_pairs)

        # Step 4: Validate
        valid_clusters, corridors, dissolved_count = self._validate_clusters(
            clusters, edges_by_node, typed_neighbors)

        # Step 4b: Absorb subsets — when one cluster is fully contained
        # in another, merge the smaller into the larger. The larger cluster
        # keeps all members (no information loss). The smaller is dissolved.
        valid_clusters, absorbed_count = self._absorb_subsets(valid_clusters)

        # Step 5: Affinities for unplaced nodes to NEW clusters
        node_affinities = self._compute_affinities(
            valid_clusters, typed_neighbors)

        # Step 5b: Affinities for unplaced nodes to EXISTING communities
        add_min_affinity = self.config.get('add_to_existing_min_affinity', 0.25)
        existing_additions = {}
        if community_state:
            for nid in unplaced:
                nbrs = typed_neighbors.get(nid, set())
                if not nbrs:
                    continue
                for comm in community_state:
                    shared = len(nbrs & comm['members'])
                    if shared > 0:
                        aff = shared / len(nbrs)
                        if aff >= add_min_affinity:
                            if nid not in existing_additions:
                                existing_additions[nid] = []
                            existing_additions[nid].append(
                                (comm['id'], comm['title'], aff))

            for nid in existing_additions:
                existing_additions[nid].sort(key=lambda x: -x[2])

        # Step 5c: Drift detection for PLACED nodes
        # Per-node threshold: the encoder's algorithmic Δ raises
        # _sys_drift_threshold each time a drift proposal is declined
        default_drift_ratio = self.config.get('drift_ratio', 1.5)
        min_foreign_aff = self.config.get('drift_min_foreign_affinity', 0.15)

        # Bulk-load per-node drift thresholds
        node_drift_thresholds = {}
        drift_rows = self.brain._meta_kv.get_all_by_key('_sys_drift_threshold').items()
        for _nid, _val in drift_rows:
            try:
                # The threshold only ever RATCHETS UP from the default — the
                # encoder raises it to suppress, nothing lowers it. A stored
                # value below the default has no author, and honoring one
                # would invert the lever: the node would drift on WEAKER
                # evidence than an untouched one. Clamp, don't trust.
                node_drift_thresholds[_nid] = max(float(_val),
                                                  default_drift_ratio)
            except (ValueError, TypeError):
                # Legacy rows only — the encoder no longer authors this value
                # (_raise_drift_thresholds stamps it). Left loud because a
                # silent fallback here is what hid the corruption: the
                # rejection suppressed nothing and the proposal returned every
                # run with no error row to show for it.
                self.brain._log_error(
                    's2_community_drift_threshold',
                    ValueError('unparseable _sys_drift_threshold %r on node %s'
                               % (_val, _nid)),
                    'drift threshold falls back to default %s'
                    % default_drift_ratio)

        drift_candidates = {}
        if community_state:
            for nid in already_placed:
                if nid not in typed_neighbors:
                    continue
                nbrs = typed_neighbors[nid]
                if not nbrs:
                    continue

                drift_ratio = node_drift_thresholds.get(nid, default_drift_ratio)
                # Home = the node's STRONGEST community. A multi-home node
                # has several; basing the ratio on an arbitrary one makes the
                # drift test (and the reported home) row-order dependent —
                # the same node could flip drifting/settled between runs on
                # an unchanged graph.
                home = max((c for c in community_state if nid in c['members']),
                           key=lambda c: len(nbrs & c['members']))
                home_aff = len(nbrs & home['members']) / len(nbrs)

                # Drift targets are EXISTING communities only — targets must
                # have persistent ids so rejection fingerprints can suppress
                # re-proposals. A same-run cluster is already a new_community
                # proposal; once created it becomes a drift target next cycle.
                for comm in community_state:
                    # A community the node already belongs to can't be a
                    # drift target — `home` is only the strongest of a
                    # multi-home node's communities.
                    if nid in comm['members']:
                        continue
                    foreign_aff = len(nbrs & comm['members']) / len(nbrs)
                    # max(home_aff, floor) keeps the ratio a real lever when
                    # home_aff == 0: a fully-estranged node still surfaces,
                    # but raising _sys_drift_threshold raises its bar too
                    # (a ratio on a zero base was inert).
                    if (foreign_aff > max(home_aff, min_foreign_aff) * drift_ratio
                            and foreign_aff > min_foreign_aff):
                        if nid not in drift_candidates:
                            drift_candidates[nid] = {
                                'home': home, 'home_aff': home_aff,
                                'drift_ratio': drift_ratio,
                                'foreign': []}
                        drift_candidates[nid]['foreign'].append(
                            (comm['id'], comm['title'], foreign_aff))

            # foreign[0] is the fingerprint key and the quota confidence —
            # it must be the STRONGEST target, not community-row order.
            for drift in drift_candidates.values():
                drift['foreign'].sort(key=lambda t: -t[2])

        # Step 5d: Community health seam. ONE tunable + ONE fact.
        #   typed int_frac < low_cohesion → the community is loose enough to act
        #     on. Within that zone:
        #       · DISCONNECTED (no internal edge of any real-cohesion relation)
        #         → deterministic auto-archive (returned as auto_archive_dead;
        #         the orchestrator writes it. No encoder round, no rejection
        #         fingerprint → can't get stuck "suppressed but undead"; and the
        #         all-relation check means a similar_to-cohesive community is NOT
        #         archived — closes the typed-int_frac blind spot).
        #       · otherwise → 'dead' encoder proposal: the encoder JUDGES
        #         archive-or-keep (prompt frames it as "low cohesion"; internal
        #         signal stays 'dead' so the fingerprint/matcher are unchanged).
        #   corridor_maturing → unchanged.
        # No signal for diffused-but-alive communities: it would fire every
        # cycle, and the encoder's rote maturity='forming' never clears the
        # trigger (creation-frozen baseline) → zero-value churn.
        low_cohesion = self.config.get('low_cohesion_threshold', 0.10)
        non_cohesion = self.config.get('non_cohesion_relations', ())
        community_health_updates = []
        auto_archive_dead = []
        # The stored structural fields for every community, from this cycle's
        # graph — the orchestrator writes the ones that moved before the
        # encoder reads any community.
        structural = {}
        for comm in community_state:
            if not comm['members']:
                continue
            ms = comm['members']
            metrics = structural_metrics(ms, edges_by_node)
            structural[comm['id']] = structural_fields(metrics)
            int_frac = metrics['internal_fraction']

            # The stored fraction is last cycle's stamp, not a baseline from
            # the encoder's last visit, so a health signal carries only the
            # current value.
            old_maturity = comm['meta'].get('community_maturity', '')

            if int_frac < low_cohesion:
                if self._community_disconnected(ms, non_cohesion):
                    # No internal cohesion of any kind — structurally dead.
                    auto_archive_dead.append({
                        'id': comm['id'],
                        'title': comm['title'],
                        'int_frac': round(int_frac, 3),
                    })
                else:
                    # Loose but still internally linked — the encoder judges.
                    community_health_updates.append({
                        'community': comm,
                        'fraction': int_frac,
                        'signal': 'dead',
                    })
            elif old_maturity == 'corridor' and int_frac > 0.3:
                community_health_updates.append({
                    'community': comm,
                    'fraction': int_frac,
                    'signal': 'corridor_maturing',
                })

        # Step 5e: Community merge detection
        merge_candidates = self._detect_merge_candidates(community_state)

        # Step 6: Cross-cutting
        cross_cutting = self._detect_cross_cutting(
            node_affinities, degrees)

        # Step 7: Orphan embedding affinities. Gated off by default (PZ-1):
        # raw-cosine placement was measured noise-level for ~27% of accepts,
        # and a wrong member corrupts a community boundary while an unplaced
        # node costs nothing — orphans re-enter via S1E edges.
        if self.config.get('orphan_placement_enabled', True):
            orphan_affinities = self._compute_orphan_affinities(
                valid_clusters, typed_neighbors)
        else:
            orphan_affinities = {}
            print('[s2cd] orphan placement paused '
                  '(orphan_placement_enabled=False)', flush=True)

        # Step 8: Tie analysis
        tie_analysis = self._analyze_ties(
            node_affinities, cross_cutting, valid_clusters,
            edges_by_node)

        # Step 9: Build proposals
        proposals = self._build_proposals(
            valid_clusters, corridors, node_affinities,
            orphan_affinities, cross_cutting, tie_analysis,
            edges_by_node, typed_neighbors, community_state,
            already_placed)

        # Step 9b: Add incremental proposals
        titles = {}
        types_map = {}
        for row in self.brain.conn.execute(
                "SELECT id, title, type FROM nodes WHERE archived = 0"):
            titles[row[0]] = row[1][:60]
            types_map[row[0]] = row[2]

        add_cap = self.config.get('add_candidates_cap', 3)
        for nid, additions in existing_additions.items():
            proposals.append({
                'type': 'add_to_existing',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'communities': [
                    {'id': cid, 'title': ctitle, 'affinity': aff}
                    for cid, ctitle, aff in additions[:add_cap]],
            })

        for nid, drift in drift_candidates.items():
            proposals.append({
                'type': 'drift',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'home_community': drift['home']['title'],
                'home_id': drift['home']['id'],
                'home_affinity': drift['home_aff'],
                'current_drift_threshold': drift.get('drift_ratio', default_drift_ratio),
                'foreign': [
                    {'id': cid, 'title': ctitle, 'affinity': aff}
                    for cid, ctitle, aff in drift['foreign'][:3]],
            })

        for update in community_health_updates:
            proposals.append({
                'type': 'health_update',
                'community_id': update['community']['id'],
                'community_title': update['community']['title'],
                'signal': update['signal'],
                'fraction': update['fraction'],
            })

        # Merge candidates
        for merge in merge_candidates:
            proposals.append({
                'type': 'merge_communities',
                'larger_id': merge['larger']['id'],
                'larger_title': merge['larger']['title'],
                'larger_size': len(merge['larger']['members']),
                'smaller_id': merge['smaller']['id'],
                'smaller_title': merge['smaller']['title'],
                'smaller_size': len(merge['smaller']['members']),
                'shared_count': merge['shared_count'],
                'overlap_pct': merge['overlap_pct'],
                'unique_in_smaller': merge['unique_in_smaller'],
            })

        # Step 9c: batch contract for add_to_existing — one proposal per
        # node, no candidate the node is already a member of.
        proposals = self._finalize_add_proposals(proposals, community_state)

        # Cluster summaries for trace (S3 consumption)
        cluster_summaries = []
        for cid, members in sorted(valid_clusters.items(),
                                    key=lambda x: -len(x[1]))[:30]:
            ms = set(members)
            metrics = structural_metrics(ms, edges_by_node)
            edge_sig = self._compute_edge_signature(members, edges_by_node)

            cluster_summaries.append({
                'cluster_id': cid,
                'size': len(members),
                'internal_edges': metrics['internal'],
                'external_edges': metrics['external'],
                'internal_fraction': round(metrics['internal_fraction'], 3),
                'edge_signature': edge_sig,
                'is_corridor': cid in corridors,
            })

        stats = {
            'nodes_with_typed_edges': len(edges_by_node),
            'total_pairs_scored': len(pair_zscores),
            'bucket_stats': {str(k): {'mean': round(v['mean'], 2),
                                       'std': round(v['std'], 2)}
                             for k, v in bucket_stats.items()},
            'clusters_seeded': len(clusters),
            'fragments_dissolved': dissolved_count,
            'subsets_absorbed': absorbed_count,
            'valid_clusters': len(valid_clusters),
            'corridors': len(corridors),
            'nodes_with_affinities': len(node_affinities) - len(cross_cutting),
            'cross_cutting': len(cross_cutting),
            'orphans_placed': len(orphan_affinities),
            'genuine_overlaps': tie_analysis['genuine_overlaps'],
            'possible_splits': tie_analysis['possible_splits'],
        }

        return {
            'proposals': proposals,
            'stats': stats,
            'cluster_summaries': cluster_summaries,
            'auto_archive_dead': auto_archive_dead,
            'structural': structural,
        }

    def _community_disconnected(self, members, non_cohesion=()):
        """True if the members share NO internal edge carrying a real-cohesion
        relation — i.e. they have no semantic link at all, only (at most)
        structural edges (`non_cohesion`: community_member, related, ...). This is the deterministic 'truly dead' signal: a
        structural fact, not a tuned int_frac. It counts edges over ALL
        relations (unlike typed int_frac, which drops generic_relation/noise) —
        that's what closes the blind spot: a community held together by
        similar_to is NOT disconnected and is routed to the encoder, not
        auto-archived.

        TODO(v25-dal): a one-off internal-edge existence check; no DAL method
        fits and this is the only caller. Raw SELECT with archived=0, mirroring
        _sample_internal_edges in this file.
        """
        ms = list(members)
        if len(ms) < 2:
            return True  # 0 or 1 member can't carry an internal edge
        ph = ','.join('?' * len(ms))
        params = ms + ms
        excl = ''
        if non_cohesion:
            excl = ' AND er.relation NOT IN (%s)' % ','.join('?' * len(non_cohesion))
            params = ms + ms + list(non_cohesion)
        row = self.brain.conn.execute(
            "SELECT 1 FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE er.archived = 0 AND e.source_id IN (%s) AND e.target_id IN (%s)%s "
            "LIMIT 1" % (ph, ph, excl), params).fetchone()
        return row is None

    # ── Step 2 ──

    def _compute_pair_scores(self, typed_neighbors, edges_by_node):
        degrees = {nid: len(nbrs) for nid, nbrs in typed_neighbors.items()}

        neighbor_to_nodes = defaultdict(set)
        for nid, nbrs in typed_neighbors.items():
            for nbr in nbrs:
                neighbor_to_nodes[nbr].add(nid)

        raw_shared = defaultdict(int)
        for shared_node, node_set in neighbor_to_nodes.items():
            nodes = list(node_set)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    key = (min(nodes[i], nodes[j]), max(nodes[i], nodes[j]))
                    raw_shared[key] += 1

        for nid, edge_list in edges_by_node.items():
            for nbr, fam, rel in edge_list:
                key = (min(nid, nbr), max(nid, nbr))
                raw_shared[key] += 2

        def _bucket(degree):
            if degree <= 1: return 1
            if degree <= 2: return 2
            if degree <= 4: return 3
            if degree <= 7: return 5
            if degree <= 12: return 8
            return 13

        bucket_scores = defaultdict(list)
        pair_to_bucket = {}
        for (a, b), raw in raw_shared.items():
            bucket = _bucket(min(degrees.get(a, 1), degrees.get(b, 1)))
            bucket_scores[bucket].append(raw)
            pair_to_bucket[(a, b)] = bucket

        bucket_stats = {}
        for bucket, scores in bucket_scores.items():
            arr = np.array(scores, dtype=float)
            bucket_stats[bucket] = {
                'mean': float(arr.mean()),
                'std': max(float(arr.std()), 0.5),
            }

        pair_zscores = {}
        for (a, b), raw in raw_shared.items():
            stats = bucket_stats[pair_to_bucket[(a, b)]]
            pair_zscores[(a, b)] = (raw - stats['mean']) / stats['std']

        return pair_zscores, degrees, bucket_stats

    # ── Step 3 ──

    def _get_direct_pairs(self, edges_by_node):
        direct = set()
        for nid, edge_list in edges_by_node.items():
            for nbr, fam, rel in edge_list:
                direct.add((min(nid, nbr), max(nid, nbr)))
        return direct

    def _seed_clusters(self, pair_zscores, direct_pairs):
        """Direct edges always seed. Z>=1.0 for neighbor-found pairs."""
        z_threshold = self.config.get('z_seed_threshold', 1.0)

        seed_pairs = []
        for (a, b), z in pair_zscores.items():
            is_direct = (a, b) in direct_pairs
            if is_direct or z >= z_threshold:
                seed_pairs.append((a, b, z, is_direct))
        # Deterministic tie-break: z-scores bucket heavily (60 distinct
        # values across ~15K pairs), and without the (a, b) suffix the order
        # within a tie group inherits str-hash-dependent dict insertion
        # order — a different process yields a different partition (F19).
        seed_pairs.sort(key=lambda x: (-x[2], -x[3], x[0], x[1]))

        clusters = {}
        node_home = {}
        cluster_counter = 0
        used = set()

        for a, b, z, is_direct in seed_pairs:
            if a in used and b in used:
                continue
            a_c = node_home.get(a)
            b_c = node_home.get(b)
            if a_c and not b_c:
                clusters[a_c].add(b); node_home[b] = a_c
            elif b_c and not a_c:
                clusters[b_c].add(a); node_home[a] = b_c
            elif not a_c and not b_c:
                cluster_counter += 1
                clusters[cluster_counter] = {a, b}
                node_home[a] = cluster_counter
                node_home[b] = cluster_counter
            used.add(a); used.add(b)

        return clusters

    # ── Step 4 ──

    def _validate_clusters(self, clusters, edges_by_node, typed_neighbors):
        valid = {}
        corridors = set()
        dissolved = 0
        min_size = self.config['min_community_size']

        for cid, members in clusters.items():
            if len(members) < min_size:
                continue

            ms = set(members)
            metrics = structural_metrics(ms, edges_by_node)

            if metrics['internal'] < 2:
                dissolved += 1
                continue

            valid[cid] = ms

            if metrics['is_corridor']:
                corridors.add(cid)

        return valid, corridors, dissolved

    # ── Step 4b ──

    def _absorb_subsets(self, valid_clusters):
        """When one cluster is fully contained in another, absorb it.

        The larger cluster keeps all its members (no information loss).
        The smaller cluster is dissolved — its members are already
        covered by the larger one.

        Handles chains: if A ⊂ B ⊂ C, both A and B dissolve into C.

        Returns: (filtered_clusters, absorbed_count)
        """
        absorbed = set()
        cluster_ids = sorted(valid_clusters.keys(),
                             key=lambda k: -len(valid_clusters[k]))

        for i in range(len(cluster_ids)):
            if cluster_ids[i] in absorbed:
                continue
            larger = valid_clusters[cluster_ids[i]]
            for j in range(i + 1, len(cluster_ids)):
                if cluster_ids[j] in absorbed:
                    continue
                smaller = valid_clusters[cluster_ids[j]]
                if smaller <= larger:  # set subset check
                    absorbed.add(cluster_ids[j])

        filtered = {cid: members for cid, members in valid_clusters.items()
                    if cid not in absorbed}
        return filtered, len(absorbed)

    # ── Step 5 ──

    def _compute_affinities(self, valid_clusters, typed_neighbors):
        node_affinities = {}
        for nid, nbrs in typed_neighbors.items():
            if not nbrs:
                continue
            affinities = []
            for cid, members in valid_clusters.items():
                shared = len(nbrs & members)
                if shared > 0:
                    affinities.append((cid, shared / len(nbrs)))
            if affinities:
                affinities.sort(key=lambda x: -x[1])
                node_affinities[nid] = affinities
        return node_affinities

    # ── Step 5e ──

    def _detect_merge_candidates(self, community_state):
        """Detect community pairs with high member overlap.

        Merge when:
        - overlap >= threshold (% of smaller community's members)
        - AND smaller community has < min_unique unique members

        This adaptive condition prevents merging in young brains where
        small communities naturally overlap due to few nodes.
        """
        threshold = self.config.get('merge_overlap_threshold', 0.80)
        min_unique = self.config.get('merge_min_unique_members', 3)
        candidates = []

        # Only check communities with members
        active = [c for c in community_state if c['members']]

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                shared = a['members'] & b['members']
                if not shared:
                    continue

                # Overlap relative to the smaller community
                smaller, larger = (a, b) if len(a['members']) <= len(b['members']) else (b, a)
                overlap_pct = len(shared) / len(smaller['members'])
                unique_in_smaller = len(smaller['members'] - shared)

                if overlap_pct >= threshold and unique_in_smaller < min_unique:
                    candidates.append({
                        'larger': larger,
                        'smaller': smaller,
                        'shared_count': len(shared),
                        'overlap_pct': round(overlap_pct, 3),
                        'unique_in_smaller': unique_in_smaller,
                    })

        candidates.sort(key=lambda c: -c['overlap_pct'])
        return candidates

    # ── Step 6 ──

    def _detect_cross_cutting(self, node_affinities, degrees):
        cross_cutting = set()
        min_deg = self.config.get('cross_cutting_min_degree', 15)
        max_top = self.config.get('cross_cutting_max_top_affinity', 0.35)

        for nid, affs in node_affinities.items():
            deg = degrees.get(nid, 0)
            top = affs[0][1] if affs else 0
            if deg >= min_deg and top < max_top:
                cross_cutting.add(nid)
        return cross_cutting

    # ── Step 7 ──

    def _compute_orphan_affinities(self, valid_clusters, typed_neighbors):
        centroids = {}
        for cid, members in valid_clusters.items():
            blobs = []
            for nid in members:
                row = self.brain.conn.execute(
                    "SELECT embedding FROM node_enrichments "
                    "WHERE node_id = ? AND vector_type = '_primary' AND embedding IS NOT NULL",
                    (nid,)).fetchone()
                if row and row[0]:
                    blobs.append(row[0])
            if blobs:
                vecs = [np.frombuffer(b, dtype=np.float32) for b in blobs]
                c = np.mean(vecs, axis=0)
                n = np.linalg.norm(c)
                if n > 0:
                    c = c / n
                centroids[cid] = c.astype(np.float32).tobytes()

        if not centroids:
            return {}

        all_nodes = set(r[0] for r in self.brain.conn.execute(
            "SELECT id FROM nodes WHERE archived = 0 "
            "AND type != 'community'").fetchall())
        orphans = all_nodes - set(typed_neighbors.keys())
        threshold = self.config.get('embedding_placement_threshold', 0.50)

        orphan_affinities = {}
        for row in self.brain.conn.execute(
                "SELECT node_id, embedding FROM node_enrichments "
                "WHERE vector_type = '_primary' AND embedding IS NOT NULL"):
            nid, emb = row
            if nid not in orphans:
                continue
            affinities = []
            for cid, centroid in centroids.items():
                sim = cosine_similarity(emb, centroid)
                if sim >= threshold:
                    affinities.append((cid, sim))
            if affinities:
                affinities.sort(key=lambda x: -x[1])
                orphan_affinities[nid] = affinities

        return orphan_affinities

    # ── Step 8 ──

    def _analyze_ties(self, node_affinities, cross_cutting,
                      valid_clusters, edges_by_node):
        genuine = 0
        splits = 0

        for nid, affs in node_affinities.items():
            if nid in cross_cutting:
                continue
            if len(affs) < 2 or abs(affs[0][1] - affs[1][1]) >= 0.03:
                continue

            c1, c2 = affs[0][0], affs[1][0]
            fams1 = set()
            fams2 = set()
            for nbr, fam, rel in edges_by_node.get(nid, []):
                if c1 in valid_clusters and nbr in valid_clusters[c1]:
                    fams1.add(fam)
                if c2 in valid_clusters and nbr in valid_clusters[c2]:
                    fams2.add(fam)

            if (fams1 - fams2) or (fams2 - fams1):
                genuine += 1
            else:
                splits += 1

        return {'genuine_overlaps': genuine, 'possible_splits': splits}

    # ── Step 9 ──

    def _build_proposals(self, valid_clusters, corridors,
                         node_affinities, orphan_affinities,
                         cross_cutting, tie_analysis,
                         edges_by_node, typed_neighbors,
                         community_state=None, already_placed=None):
        proposals = []

        titles = {}
        types_map = {}
        for row in self.brain.conn.execute(
                "SELECT id, title, type FROM nodes WHERE archived = 0"):
            titles[row[0]] = row[1][:60]
            types_map[row[0]] = row[2]

        # Build community membership lookup for overlap check
        comm_members = {}  # comm_id -> set of member ids
        if community_state:
            for comm in community_state:
                comm_members[comm['id']] = comm['members']
        if already_placed is None:  # direct callers (tests, evals)
            already_placed = set().union(*comm_members.values()) \
                if comm_members else set()

        overlap_threshold = self.config.get('cluster_overlap_threshold', 0.60)
        flagged_overlap = 0

        # ── Community proposals ──
        for cid, members in sorted(valid_clusters.items(),
                                    key=lambda x: -len(x[1])):
            ms = set(members)
            edge_sig = self._compute_edge_signature(members, edges_by_node)
            metrics = structural_metrics(ms, edges_by_node)
            internal, external = metrics['internal'], metrics['external']
            int_frac = metrics['internal_fraction']

            # Timeline
            member_dates = []
            for nid in members:
                row = self.brain.conn.execute(
                    "SELECT created_at FROM nodes WHERE id = ?",
                    (nid,)).fetchone()
                if row and row[0]:
                    member_dates.append((nid, row[0]))
            member_dates.sort(key=lambda x: x[1])

            origin = member_dates[0] if member_dates else None
            latest = member_dates[-1] if member_dates else None

            transitions = []
            seen_trans = set()
            for nid in ms:
                for nbr, fam, rel in edges_by_node.get(nid, []):
                    if nbr in ms and fam == 'replacement_correction':
                        pair = tuple(sorted([nid, nbr]))
                        if pair in seen_trans:
                            continue
                        seen_trans.add(pair)
                        nid_d = next((d for n, d in member_dates if n == nid), '')
                        nbr_d = next((d for n, d in member_dates if n == nbr), '')
                        corrector = nid if nid_d > nbr_d else nbr
                        corrected = nbr if corrector == nid else nid
                        transitions.append({
                            'corrector_id': corrector,
                            'corrected_id': corrected,
                            'corrector_title': titles.get(corrector, '?'),
                            'corrected_title': titles.get(corrected, '?'),
                            'date': max(nid_d, nbr_d)[:10],
                            'relation': rel,
                        })
            transitions.sort(key=lambda t: t['date'])

            timeline = {
                'origin': {'id': origin[0], 'date': origin[1][:10],
                           'title': titles.get(origin[0], '?'),
                           'type': types_map.get(origin[0], '?')}
                if origin else None,
                'latest': {'id': latest[0], 'date': latest[1][:10],
                           'title': titles.get(latest[0], '?'),
                           'type': types_map.get(latest[0], '?')}
                if latest else None,
                'transitions': transitions[:5],
                'date_range': '%s to %s' % (
                    member_dates[0][1][:10] if member_dates else '?',
                    member_dates[-1][1][:10] if member_dates else '?'),
            }

            # Structural hubs
            rep_scores = {}
            for nid in members:
                rep_scores[nid] = sum(
                    1 for nbr in typed_neighbors.get(nid, set())
                    if nbr in ms)
            reps = sorted(rep_scores.items(), key=lambda x: -x[1])[:3]
            representatives = [
                {'id': nid, 'title': titles.get(nid, nid[:8]),
                 'type': types_map.get(nid, '?'),
                 'internal_edges': score}
                for nid, score in reps]

            sample_edges = self._sample_internal_edges(members, limit=5)

            # All members with titles (chronological)
            all_member_info = [
                {'id': nid, 'title': titles.get(nid, nid[:8]),
                 'type': types_map.get(nid, '?'),
                 'date': next((d[:10] for n, d in member_dates if n == nid), '?')}
                for nid in [n for n, _ in member_dates]]

            latest_id = latest[0] if latest else None
            hub_ids = {r['id'] for r in representatives}
            render_latest = latest_id and latest_id not in hub_ids

            # ── Overlap check: would this cluster duplicate an existing community? ──
            # For each existing community, count how many cluster members have
            # neighbors in that community. If >= threshold, flag overlaps_existing
            # for the encoder to judge.
            best_overlap_comm = None
            best_overlap_frac = 0.0
            if comm_members:
                for comm_id, comm_mids in comm_members.items():
                    connecting = sum(
                        1 for nid in ms
                        if typed_neighbors.get(nid, set()) & comm_mids)
                    frac = connecting / len(ms) if ms else 0
                    if frac > best_overlap_frac:
                        best_overlap_frac = frac
                        best_overlap_comm = comm_id

            # Births stay births (PZ-1): the overlap used to CONVERT this
            # cluster into add_to_existing proposals for the overlapping
            # community — feeding the accretion giants (measured: 94% of
            # fresh clusters converted). Now it rides the proposal as
            # evidence and the encoder judges new-story-vs-extension.
            overlaps_existing = None
            if best_overlap_frac >= overlap_threshold and best_overlap_comm:
                overlaps_existing = {
                    'id': best_overlap_comm,
                    'title': next(
                        (c['title'] for c in community_state
                         if c['id'] == best_overlap_comm), '?'),
                    'connected_frac': round(best_overlap_frac, 2),
                }
                flagged_overlap += 1

            proposals.append({
                'type': 'new_community',
                'overlaps_existing': overlaps_existing,
                'cluster_id': cid,
                'members': list(members),
                'member_count': len(members),
                'all_members': all_member_info,
                'internal_edges': internal,
                'external_edges': external,
                'internal_fraction': round(int_frac, 3),
                'is_corridor': cid in corridors,
                'timeline': timeline,
                'edge_signature': edge_sig,
                'representatives': representatives,
                'render_latest': render_latest,
                'sample_edges': sample_edges,
            })

        # ── Node affinity proposals ──
        for nid, affinities in node_affinities.items():
            if nid in cross_cutting or not affinities:
                continue
            proposals.append({
                'type': 'node_affinities',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'affinities': [{'cluster_id': c, 'affinity': a}
                                for c, a in affinities[:5]],
                'method': 'structural',
            })

        # ── Cross-cutting proposals ──
        for nid in cross_cutting:
            affs = node_affinities.get(nid, [])
            proposals.append({
                'type': 'cross_cutting',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'cluster_count': len(affs),
                'top_affinity': affs[0][1] if affs else 0,
            })

        # ── Orphan proposals ──
        for nid, affinities in orphan_affinities.items():
            proposals.append({
                'type': 'node_affinities',
                'node_id': nid,
                'node_title': titles.get(nid, nid[:8]),
                'node_type': types_map.get(nid, '?'),
                'affinities': [{'cluster_id': c, 'affinity': sim}
                                for c, sim in affinities[:5]],
                'method': 'embedding',
            })

        if flagged_overlap:
            print('[s2cd] Overlap check: flagged %d new_community proposals '
                  'with an overlapping existing community (threshold %.0f%%)'
                  % (flagged_overlap, overlap_threshold * 100), flush=True)

        return proposals

    # ── Step 9c ──

    def _finalize_add_proposals(self, proposals, community_state):
        """Batch contract for add_to_existing: the encoder must never see
        (a) two proposals for the same node, or (b) a candidate community
        the node is already a member of.

        Emitters produce add_to_existing without seeing each other (today
        only the Step 5b affinity loop; any new emitter joins before this
        contract), so the same (node, community) can otherwise reach the
        encoder twice in one batch. Merge to one proposal per node: candidate
        lists union by community id keeping the HIGHER-affinity entry — the
        rejection fingerprint tier, the quota rank, and prior suppressions
        all key on the strongest candidate, so preferring a weaker duplicate
        would re-arm old rejections and demote strong placements. The
        proposal-level algorithmic-source label is recomputed from the
        winning head candidate. Non-add proposals pass through untouched.

        The member filter is a defensive belt: every live emitter is already
        unplaced-only, so a drop here means a new emitter violated the
        placed-nodes-go-through-drift policy — surfaced loudly, never silent.
        """
        members_by_comm = {c['id']: c['members'] for c in community_state or []}
        cap = self.config.get('add_candidates_cap', 3)

        finalized = []
        held_by_node = {}
        for p in proposals:
            if p.get('type') != 'add_to_existing':
                finalized.append(p)
                continue
            nid = p.get('node_id', '')
            comms, dropped = [], []
            for c in p.get('communities', []):
                if not c.get('id'):
                    continue
                if nid and nid in members_by_comm.get(c['id'], ()):
                    dropped.append(c['id'])
                else:
                    comms.append(c)
            if dropped:
                print('[s2cd] add_to_existing for %s dropped member '
                      'candidate(s) %s — an emitter bypassed the '
                      'placed-nodes-go-through-drift policy'
                      % (nid[:8] or '?', ','.join(d[:8] for d in dropped)),
                      flush=True)
            if not comms:
                continue
            held = held_by_node.get(nid) if nid else None
            if held is None:
                merged = dict(p, communities=comms)
                if nid:
                    held_by_node[nid] = merged
                finalized.append(merged)
                continue
            cands = {c['id']: c for c in held['communities']}
            for c in comms:
                prev = cands.get(c['id'])
                if prev is None or c.get('affinity', 0) > prev.get('affinity', 0):
                    cands[c['id']] = c
            ranked = sorted(cands.values(),
                            key=lambda c: -c.get('affinity', 0))[:cap]
            held['communities'] = ranked
        return finalized

    # ── Decode helpers ──

    def _compute_edge_signature(self, members, edges_by_node):
        counts = Counter()
        ms = set(members)
        for nid in members:
            for nbr, fam, rel in edges_by_node.get(nid, []):
                if nbr in ms:
                    counts[fam] += 1
        total = sum(counts.values())
        if total == 0:
            return {}
        return {fam: round(cnt / total, 2)
                for fam, cnt in counts.most_common(10)}

    def _sample_internal_edges(self, members, limit=5):
        ms = set(members)
        # Sanctioned raw-SQL exception: internal-edge rendering with
        # descriptions. The shape is close to has_edge_between but returns
        # metadata, not a bool. Raw with archived=0 until a second caller.
        #
        # Skips what the disconnected check skips: this is the EVIDENCE for
        # cohesion, so it must not show an edge that does not count as
        # cohesion. Reads the one named source (`non_cohesion_relations`), the
        # way _community_disconnected does, so no narrower spelling of the
        # same concept can drift from it.
        non_cohesion = list(self.config.get('non_cohesion_relations', ()))
        placeholders = ','.join('?' * len(ms))
        id_list = list(ms)
        excl = ''
        params = id_list * 2
        if non_cohesion:
            excl = ' AND er.relation NOT IN (%s)' % ','.join('?' * len(non_cohesion))
            params = params + non_cohesion
        rows = self.brain.conn.execute("""
            SELECT ns.title, nt.title, er.relation, er.description
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            JOIN nodes ns ON ns.id = er.claim_source_id
            JOIN nodes nt ON nt.id = er.claim_target_id
            WHERE er.claim_source_id IN (%s) AND er.claim_target_id IN (%s)
            AND er.archived = 0%s
            AND er.description IS NOT NULL AND er.description != ''
            ORDER BY e.weight DESC LIMIT ?
        """ % (placeholders, placeholders, excl),
            params + [limit]).fetchall()
        return [{'source': r[0][:50], 'target': r[1][:50],
                 'relation': r[2], 'description': r[3][:100]}
                for r in rows]
