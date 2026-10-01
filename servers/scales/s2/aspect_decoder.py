"""S2 Aspect Decoder — finds strings with a field still to fill.

Two lanes over the brain's distinct types/relations, one candidate shape:
  · unclassified — a string no aspect lists yet; it needs `aspects` and
    every MEMBER_FIELDS enum field of its category (`fill` names them all)
  · unfilled — a classified string, present in this brain's graph, whose
    member carries no value for a field the spec declares (`orientation`);
    `fill` names the gap and `aspects` carries its fixed homes
Filters by count threshold (1 = every string, singletons included). Each
lane sorts by count DESC; unclassified take the first
`max_candidates_per_call`, unfilled ride behind up to `max_fill_per_call`
so a backlog never starves new vocabulary. For each candidate, loads N
example records (nodes for types, edges for relations) to give the encoder
real evidence of usage — which is also why the unfilled lane reads the
graph, not the taxonomy: a verb this brain has never written has no
examples to judge from, and the seed carries its value.

The encoder reads the menu via brain.aspects — decoder only produces the
candidate list + examples. No suppression, no clustering — closed-list
classification means every string gets a home and every field a value.
The decoder also decides WHEN to call: a cycle with fewer than
`min_pending_to_call` candidates, none written `call_now_at_count` times,
skips and lets the work accumulate, because each call re-sends the whole
instruction and menu prefix.
"""

from servers.aspect_store import filled_fields, new_member_fill
from servers.contract import truncate_marked

from .base import IntegrationUnit
from .aspect_contract import ASPECT


class AspectDecoder(IntegrationUnit):
    NAME = 'aspect_integration'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:aspect_integration'

    O_SOURCES = ['nodes.type', 'edge_relations.relation', 'aspects_v1.json']
    K_SOURCES = ['count_threshold', 'examples_per_candidate']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or ASPECT

    def run(self, gate=True):
        """Find strings with a field to fill and build proposals.

        `gate=False` skips the batch gate — a forced cycle, and the
        orchestrator's later passes in a cycle that already called, which
        drain what is left while the prefix is still cached.

        Returns: {proposals: [...], stats: {...}, skipped: str|None}
        """
        classified = self._load_classified_strings()
        type_rows, relation_rows = self._scan_distinct()
        threshold = self.config['min_count_threshold']
        reg = self.brain.aspects

        # A member whose primary aspect is noise is machinery with no semantic
        # claim (noise wins the reverse maps whatever else claims it). A
        # dimension is a semantic fact about a member, so the fill lane does
        # not ask for one — `co_anchored` and `community_member` would
        # otherwise lead every backlog by count.
        primary_of = {'node_types': reg.by_node_type,
                      'edge_relations': reg.by_edge_relation}
        unclassified = []
        unfilled = []
        for category, rows in (('node_types', type_rows),
                               ('edge_relations', relation_rows)):
            fields = filled_fields(category)
            for r in rows:
                if r['count'] < threshold:
                    continue
                if r['value'] not in classified[category]:
                    unclassified.append({'category': category, **r,
                                         'fill': new_member_fill(category)})
                    continue
                primary = primary_of[category](r['value'])
                if primary is not None and primary.name == 'noise':
                    continue
                gap = [f for f in fields if reg.fact_of(r['value'], f) is None]
                if gap:
                    unfilled.append({'category': category, **r, 'fill': gap,
                                     'aspects': list(reg.homes_of(r['value'], category))})

        stats = {
            'distinct_types': len(type_rows),
            'distinct_relations': len(relation_rows),
            'classified_types': len(classified['node_types']),
            'classified_relations': len(classified['edge_relations']),
            'unclassified_types': sum(1 for c in unclassified if c['category'] == 'node_types'),
            'unclassified_relations': sum(1 for c in unclassified if c['category'] == 'edge_relations'),
            'unfilled': len(unfilled),
        }

        # Each lane sorted by count DESC — highest-evidence candidates first
        # regardless of category. Unclassified strings fill the batch first;
        # unfilled members ride behind under their own cap, so the day a
        # field is declared (every verb in the graph becomes unfilled at
        # once) new vocabulary still gets its aspect home in the same cycle.
        unclassified.sort(key=lambda r: r['count'], reverse=True)
        unfilled.sort(key=lambda r: r['count'], reverse=True)
        batch = (unclassified[:self.config['max_candidates_per_call']]
                 + unfilled[:self.config['max_fill_per_call']])
        remaining = len(unclassified) + len(unfilled) - len(batch)

        # Per-string write counts, for the encoder's menu ranking (each
        # aspect shows its most-used members) — the same scan, no second query.
        usage = {'node_types': {r['value']: r['count'] for r in type_rows},
                 'edge_relations': {r['value']: r['count'] for r in relation_rows}}

        # Empty-batch early-out BEFORE any trace write. The O trace marks
        # "I observed work" — when there's none, writing it is noise that
        # historically caused a runaway S2 cascade (downstream units re-fired
        # treating empty scans as fresh work, rolled back 2026-05-08 in
        # coordinator.py). The unit is a true no-op when everything is
        # classified and every field is filled.
        if not batch:
            return {'proposals': [], 'stats': stats, 'usage': usage,
                    'skipped': 'nothing to classify or fill'}

        # Batch gate: too little waiting, and nothing the brain already leans
        # on, means the work waits for the next cycle. Unlike the empty batch
        # this did observe work, so it says so in the O trace (nothing gates
        # on S2 traces: the healer reads S1 encoding runs, community reads the
        # graph).
        pending = unclassified + unfilled
        if gate and not (len(pending) >= self.config['min_pending_to_call']
                         or any(r['count'] >= self.config['call_now_at_count']
                                for r in pending)):
            skipped = 'waiting for a batch (%d of %d pending)' % (
                len(pending), self.config['min_pending_to_call'])
            self.trace('O', 'aspect_scan',
                       '%d unclassified types + %d unclassified relations + %d unfilled → %s' % (
                           stats['unclassified_types'], stats['unclassified_relations'],
                           stats['unfilled'], skipped),
                       metadata={**stats, 'pending': len(pending), 'gated': True,
                                 'min_pending_to_call': self.config['min_pending_to_call']})
            return {'proposals': [], 'stats': stats, 'usage': usage, 'skipped': skipped}

        self.trace('O', 'aspect_scan',
                   '%d unclassified types + %d unclassified relations + %d unfilled → batch of %d' % (
                       stats['unclassified_types'],
                       stats['unclassified_relations'],
                       stats['unfilled'],
                       len(batch)),
                   metadata=stats)

        # Attach example records to each candidate
        proposals = self._build_proposals(batch)

        self.trace('K', 'aspect_proposals',
                   '%d proposals built with %d examples each' % (
                       len(proposals), self.config['examples_per_candidate']),
                   metadata={
                       'proposal_count': len(proposals),
                       'examples_per_proposal': self.config['examples_per_candidate'],
                       'remaining': max(0, remaining),
                   })

        return {
            'proposals': proposals,
            'stats': stats,
            'usage': usage,
            'remaining': remaining,
        }

    # ─── helpers ─────────────────────────────────────────────────────

    def _load_classified_strings(self):
        """Registry view of every classified string: {node_types: set, edge_relations: set}.

        Strings that already appear in any aspect's member list are
        considered classified and won't be re-proposed. Reads brain.aspects —
        fresh by construction, since every taxonomy write goes through the
        registry's door and re-derives its maps.
        """
        reg = self.brain.aspects
        names = list(reg.all().keys())
        return {'node_types': set(reg.types_in(names)),
                'edge_relations': set(reg.relations_in(names))}

    def _scan_distinct(self):
        """Return (type_rows, relation_rows) sorted by count DESC.

        Each row: {value: str, count: int}. Archived rows excluded.
        """
        type_rows = self.brain.conn.execute("""
            SELECT type, COUNT(*)
            FROM nodes
            WHERE archived = 0 AND type IS NOT NULL AND type != ''
            GROUP BY type
            ORDER BY COUNT(*) DESC
        """).fetchall()

        relation_rows = self.brain.conn.execute("""
            SELECT relation, COUNT(*)
            FROM edge_relations
            WHERE archived = 0 AND relation IS NOT NULL AND relation != ''
            GROUP BY relation
            ORDER BY COUNT(*) DESC
        """).fetchall()

        return (
            [{'value': r[0], 'count': r[1]} for r in type_rows],
            [{'value': r[0], 'count': r[1]} for r in relation_rows],
        )

    def _build_proposals(self, batch):
        """Attach example records to each candidate, carrying its `fill` and
        (for a routed candidate) its fixed `aspects` through to the encoder.

        For node_types: 3 nodes that have that type — title + content snippet.
        For edge_relations: 3 edges with that relation — source/target titles
            + edge description.

        Examples are picked deterministically (highest access_count /
        edge weight first) for reproducibility across cycles.
        """
        n_examples = self.config['examples_per_candidate']
        proposals = []
        for c in batch:
            if c['category'] == 'node_types':
                examples = self._sample_node_examples(c['value'], n_examples)
            else:
                examples = self._sample_relation_examples(c['value'], n_examples)
            proposal = {
                'category': c['category'],
                'value': c['value'],
                'count': c['count'],
                'fill': c['fill'],
                'examples': examples,
            }
            if c.get('aspects'):
                proposal['aspects'] = c['aspects']
            proposals.append(proposal)
        return proposals

    def _sample_node_examples(self, type_value, n):
        """Pick examples spanning utility range — strong/typical/edge.

        Ordering: access_count DESC (how often the brain reaches for it)
        then length(content) DESC (richer text). Confidence dropped as
        ordering signal — it tracks content truthness, not how
        representative an example is of its type.

        Loads situation alongside content because situation often signals
        the type's role more clearly than content (e.g., "When designing
        the recall pipeline" → architecture/design type).
        """
        rows = self.brain.conn.execute("""
            SELECT n.id, n.title, n.content, n.type,
                   (SELECT value FROM node_metadata_kv
                    WHERE node_id = n.id AND key = 'situation' LIMIT 1)
            FROM nodes n
            WHERE n.archived = 0 AND n.type = ?
            ORDER BY n.access_count DESC, LENGTH(n.content) DESC
        """, (type_value,)).fetchall()
        # Indices: 0=id, 1=title, 2=content, 3=type, 4=situation
        # Cuts end in '…' (contract.truncate_marked) — a reader must be able to tell
        # a clipped snippet from a short one.
        return [
            {
                'id': (r[0] or '')[:8],
                'type': r[3] or '',
                'title': r[1] or '',
                'content_snippet': truncate_marked(r[2] or '', 400),
                'situation': truncate_marked(r[4] or '', 300),
            }
            for r in self._pick_diverse(rows, n)
        ]

    def _sample_relation_examples(self, relation_value, n):
        """Pick relation examples spanning weight range — strong/typical/edge.

        Ordering: weight DESC (Hebbian utility) then length(description)
        DESC (richer text). Includes src + tgt content snippets so the
        encoder can reason about what the relation actually expresses,
        not just what its name suggests.
        """
        rows = self.brain.conn.execute("""
            SELECT er.description, er.weight,
                   src.title, src.type, src.content,
                   tgt.title, tgt.type, tgt.content,
                   src.id, tgt.id
            FROM edge_relations er
            JOIN nodes src ON er.claim_source_id = src.id
            JOIN nodes tgt ON er.claim_target_id = tgt.id
            WHERE er.archived = 0 AND er.relation = ?
              AND src.archived = 0 AND tgt.archived = 0
            ORDER BY er.weight DESC, LENGTH(er.description) DESC
        """, (relation_value,)).fetchall()
        # Indices: 0=desc, 1=weight, 2=src_title, 3=src_type, 4=src_content,
        #          5=tgt_title, 6=tgt_type, 7=tgt_content, 8=src_id, 9=tgt_id
        # The description travels whole, as every edge render does (a reader
        # may quote it); content cuts end in '…'.
        return [
            {
                'src_id': (r[8] or '')[:8],
                'src_title': r[2] or '',
                'src_type': r[3] or '',
                'src_content_snippet': truncate_marked(r[4] or '', 150),
                'tgt_id': (r[9] or '')[:8],
                'tgt_title': r[5] or '',
                'tgt_type': r[6] or '',
                'tgt_content_snippet': truncate_marked(r[7] or '', 150),
                'description': r[0] or '',
                'weight': r[1],
            }
            for r in self._pick_diverse(rows, n)
        ]

    @staticmethod
    def _pick_diverse(rows, n):
        """Pick n rows spanning the input list's range: the strongest, the
        median, the weakest (evenly spaced for other n). Input rows are
        pre-sorted (highest score first). Shorter inputs return whole. The
        spread is the point — the encoder sees the relation at its best and
        at its thinnest — and it needs no label on the page.
        """
        if not rows:
            return []
        if len(rows) <= n:
            return list(rows)
        if n == 1:
            return [rows[0]]
        step = (len(rows) - 1) / (n - 1)
        return [rows[int(round(i * step))] for i in range(n)]
