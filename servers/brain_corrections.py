"""
brain — BrainCorrections Mixin

Walks correction-aspect edges (corrects, supersedes, reframes, resolves, ...)
to attach corrector context to any set of node ids. Every canonical node pull
(brain.get_node) attaches `_corrections` via this method.

If you're writing a new fetcher that bypasses brain.get_node, decide
explicitly whether you need corrections and call brain.correction_enrich(ids)
if you do. Don't re-implement the aspect-edge walk.

Mixed into the Brain class via multiple inheritance. References self.conn,
self.aspects, self._log_error — provided by Brain.__init__.
"""

import sys
from typing import Any, Dict, Iterable, List

from servers.aspect_store import field_values
from servers.contract import relation_age

# Flavor rank, strongest first — the declaration lists its values in that
# order. An uncurated verb (no flavor) ranks after every curated one.
_FLAVOR_RANK = {f: i for i, f in enumerate(field_values('flavor'))}


class BrainCorrectionsMixin:
    """Correction-aspect graph walks for Brain."""

    def correction_enrich(self, node_ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Find the nodes that corrected each of these nodes, via the
        correction_improvement aspect.

        Walks edge_relations whose `relation` is in
        `self.aspects.correction_improvement.edge_relations` (corrects,
        supersedes, reframes, resolves, addresses, fixes, ...) in both stored
        directions, and keeps the neighbours the verb's orientation names as
        the actor on this node — the ⚠ warnings. The inverse (this node
        corrected the neighbour) is history and is left to the node's Edges,
        as is a verb whose curated flavor is `unchanged` (nothing in this
        node's claim changed yet). A verb with no orientation yet is kept on
        both endpoints as a neutral entry. Entries come strongest flavor
        first, so a reader that shows only some keeps the ones that matter.

        Returns dict {node_id: [correction_dict, ...]} keyed by BOTH the
        8-char short form AND the full id, so callers indexing by either form
        resolve.

        Each correction_dict carries the HEAVY payload — renderer slices it
        per consumer (HAIKU_FORMAT: balanced, ENCODER/HEALER_FORMAT: full).

            id              (str)  neighbor short id
            title           (str)  neighbor title
            type            (str)  neighbor node type
            direction       (str|None)  the NEIGHBOUR's role: 'corrector'
                                   (it corrected this node) | None (the verb
                                   carries no orientation) — from the verb's
                                   orientation, not the stored arrow:
                                   `A resolved_by B` makes B the corrector.
                                   A role word, not a verb, so no consumer
                                   mistakes the label for a relation.
            relation        (str)  the aspect verb(s); a neighbour whose verbs
                                   share a role is ONE entry, whichever end
                                   wrote each claim, its verbs joined with ', '
            flavor          (str|None)  what happened to this node's claim —
                                   the strongest curated flavor among the
                                   entry's verbs (MEMBER_FIELDS['flavor']);
                                   None when no verb is curated
            edge_description (str) the edge's `why` text, the verbs'
                                   distinct descriptions joined with '; '
            relation_at     (str)  when the entry's claim last changed —
                                   contract.relation_age, the newest among
                                   its verbs (the age an Edges line shows)
            content        (str)  neighbor's full content (renderer slices)
            reasoning       (str)  neighbor's reasoning metadata
            their_raw_quote  (str)  neighbor's their_raw_quote metadata
            my_raw_quote (str) neighbor's my_raw_quote metadata

        Loud-by-default: any failure inside the aspect walk gets logged via
        self._log_error AND returns an empty dict so the calling pipeline
        (get_node, traverse, future fetchers) degrades gracefully instead of
        taking the whole query down.
        """
        if not node_ids:
            return {}

        try:
            return self._correction_enrich_impl(node_ids)
        except Exception as e:
            try:
                self._log_error(
                    'correction_enrich', e,
                    'aspect-edge walk failed for %d ids' % len(list(node_ids)))
            except Exception:
                # _log_error itself failing is rare (DB / rate limit) — don't
                # cascade; print so it shows in daemon.log.
                print('[correction_enrich] error + _log_error failed: %r' % e,
                      file=sys.stderr, flush=True)
            return {}

    def _correction_enrich_impl(self, node_ids):
        """Body of correction_enrich — separated so the public method can wrap
        it in a single try/except without indentation pyramid.
        """
        full_ids = list(dict.fromkeys(nid for nid in node_ids if nid))
        if not full_ids:
            return {}

        correction_relations = tuple(self.aspects.correction_improvement.edge_relations)
        if not correction_relations:
            return {}

        graph_dal = self._graph
        connections = graph_dal.get_connections_bulk(
            full_ids,
            include_relations=correction_relations,
            include_archived=False,
            include_neighbor_archived=False)

        # Keep only the neighbours that corrected this node, plus verbs with
        # no orientation yet (either side, so they stay visible as a neutral
        # warning). A neighbour THIS node corrected is history, and a verb
        # curated `unchanged` changed nothing in this node's claim: both render
        # as ordinary edges in the node's Edges, not as a warning. Who acts is
        # aspects.actor_of on the claim's own ends — never read off the arrow.
        kept = {}
        for owner_full, conns in connections.items():
            per_neighbour = {}
            for c in conns:
                src, tgt = c['claim_source_id'], c['claim_target_id']
                # One entry per role per neighbour: two verbs that agree the
                # neighbour corrected this node are one corrector, not two ⚠
                # lines (lean mode would print them byte-identical and spend
                # the inject's correction budget twice) — also when they were
                # written from opposite ends and arrive as two entries, one
                # per direction.
                _, by_role = per_neighbour.setdefault(c['id'], (c, {}))
                for rel in c.get('relations', []) or []:
                    verb = rel.get('relation') or ''
                    desc = rel.get('description') or ''
                    actor = self.aspects.actor_of(verb, src, tgt)
                    if actor == owner_full:
                        continue  # this node acts: history, left to Edges
                    flavor = self.aspects.fact_of(verb, 'flavor')
                    if flavor == 'unchanged':
                        continue  # nothing changed in this claim: left to Edges
                    role = None if actor is None else 'corrector'
                    item = by_role.setdefault(role, {'verbs': [], 'descs': [],
                                                     'flavors': [], 'ats': []})
                    if verb not in item['verbs']:
                        item['verbs'].append(verb)
                    if desc not in item['descs']:
                        item['descs'].append(desc)
                    if flavor:
                        item['flavors'].append(flavor)
                    item['ats'].append(relation_age(rel, c))
            pairs = [(c, by_role) for c, by_role in per_neighbour.values() if by_role]
            if pairs:
                kept[owner_full] = pairs

        if not kept:
            return {}

        neighbor_ids = list({c['id'] for pairs in kept.values() for c, _ in pairs})
        # Keys must mirror render_corrections's heavy-mode render list, otherwise
        # the renderer reads a key the data layer never fetched.
        naked_by_id = self._nodes.get_bulk(neighbor_ids)
        meta_by_id = self._meta_kv.get_fields_bulk(
            neighbor_ids, ['reasoning', 'their_raw_quote', 'my_raw_quote'])

        corrections = {}
        for owner_full, pairs in kept.items():
            bucket = []
            for c, by_role in pairs:
                neighbor_id = c['id']
                naked = naked_by_id.get(neighbor_id) or {}
                meta = meta_by_id.get(neighbor_id) or {}
                for role, item in by_role.items():
                    bucket.append({
                        'id': neighbor_id[:8],
                        # Full id alongside the display short-form: the
                        # scope veil (scopes.scrub_node) matches full ids —
                        # a short-only entry would be silently unfilterable.
                        'node_id': neighbor_id,
                        'title': c.get('title') or naked.get('title') or '',
                        'type': naked.get('type') or c.get('type') or '',
                        'direction': role,
                        'relation': ', '.join(v for v in item['verbs'] if v),
                        'flavor': min(item['flavors'], key=_FLAVOR_RANK.get,
                                      default=None),
                        'edge_description': '; '.join(d for d in item['descs'] if d),
                        'relation_at': max(item['ats'], default=''),
                        'content': naked.get('content') or '',
                        'reasoning': meta.get('reasoning') or '',
                        'their_raw_quote': meta.get('their_raw_quote') or '',
                        'my_raw_quote': meta.get('my_raw_quote') or '',
                    })
            # Strongest first: correctors by flavor rank, uncurated correctors
            # after them, neutral entries last. Stable, so the edge order
            # (weight, then recency) holds within a rank.
            bucket.sort(key=lambda e: (e['direction'] is None,
                                       _FLAVOR_RANK.get(e['flavor'], len(_FLAVOR_RANK))))
            corrections[owner_full] = bucket

        return corrections
