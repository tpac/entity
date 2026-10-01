"""S2 Healer Decoder — finds nodes needing healing and loads full context.

Scans for nodes missing question, situation, or reasoning.
Loads full context via get_rich_node() + conversation from S0 traces.
Produces proposals for the encoder.
"""

import json

from .base import IntegrationUnit
from .healer_contract import HEALER


class HealerDecoder(IntegrationUnit):
    NAME = 'healer'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:healer'

    O_SOURCES = ['healer_gaps', 's1_encoding_runs']
    K_SOURCES = ['get_rich_node', 's0_conversation']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or HEALER

    def run(self):
        """Scan for healing needs and build proposals.

        Returns: {proposals: [...], stats: {...}, skipped: str|None}
        """
        # Check if there's work to do
        if not self._has_new_traces('s1', ref_type='encoding_run'):
            # Still check for gaps even without new traces
            gaps = self._count_gaps()
            if gaps['total'] == 0:
                return {'proposals': [], 'stats': gaps, 'skipped': 'no gaps'}

        gaps = self._count_gaps()

        # Find nodes needing healing, prioritized
        target_ids = self._find_targets(gaps)

        # O trace
        self.trace('O', 'healer_scan',
                   '%d need question, %d need situation, %d need reasoning → %d targets' % (
                       gaps['question'], gaps['situation'], gaps['reasoning'],
                       len(target_ids)),
                   metadata=gaps)

        if not target_ids:
            return {'proposals': [], 'stats': gaps, 'skipped': 'no targets'}

        # Build proposals with full context
        proposals = self._build_proposals(target_ids)

        # K trace
        self.trace('K', 'healer_proposals',
                   '%d proposals built' % len(proposals),
                   metadata={
                       'proposal_count': len(proposals),
                       'with_conversation': sum(
                           1 for p in proposals
                           if (p.get('conversation') or {}).get('conversations')),
                       'node_ids': [p['node_id'][:8] for p in proposals[:20]],
                   })

        return {
            'proposals': proposals,
            'stats': gaps,
        }

    def _count_gaps(self):
        """Count nodes missing each field.

        Includes community nodes — they're first-class nodes that need
        question/situation/reasoning like any other node.
        """
        total = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived=0"
        ).fetchone()[0]

        question = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes n WHERE n.archived=0 "
            "AND n.id NOT IN (SELECT node_id FROM node_metadata_kv WHERE key='question')"
        ).fetchone()[0]

        situation = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes n WHERE n.archived=0 "
            "AND n.id NOT IN ("
            "  SELECT node_id FROM node_metadata_kv "
            "  WHERE key='situation' AND length(value) > 5)"
        ).fetchone()[0]

        reasoning = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes n WHERE n.archived=0 "
            "AND n.id NOT IN ("
            "  SELECT node_id FROM node_metadata_kv WHERE key='reasoning' AND length(value) > 5)"
        ).fetchone()[0]

        return {
            'total': question + situation + reasoning,
            'question': question,
            'situation': situation,
            'reasoning': reasoning,
            'nodes': total,
        }

    def _find_targets(self, gaps):
        """Find nodes to heal, prioritized by how many fields they're missing.

        Includes community nodes — they need question/situation/reasoning
        like any other node. The conversation loader handles missing
        conversation gracefully (communities have no conversation context).

        Priority tiers (each capped at max_nodes_per_run, fall-through
        fills remaining slots). Situation and reasoning are first-class
        targets alongside question — earlier versions only targeted
        question-missing nodes, which left a 600+ node backlog of
        situation/reasoning gaps invisible to healing.
        """
        max_nodes = self.config['max_nodes_per_run']

        targets = []
        seen = set()

        def _run(sql, limit):
            if limit <= 0:
                return
            # Already-taken ids are excluded in SQL, before the LIMIT. The
            # tiers are supersets (tier 2's two-missing predicates re-match
            # every tier-1 node), so the old post-LIMIT Python dedup let
            # already-seen rows consume the fetch budget — runs configured
            # for max_nodes healed far fewer (fetch-then-filter clip class,
            # 2026-08-07 sweep).
            exclude, params = '', []
            if seen:
                exclude = ' AND n.id NOT IN (%s)' % ','.join('?' * len(seen))
                params = list(seen)
            rows = self.brain.conn.execute(
                sql + exclude + ' ORDER BY n.access_count DESC LIMIT ?',
                params + [limit]).fetchall()
            # SQL owns dedup + budget (NOT IN + LIMIT=remaining), so this
            # can't overflow max_nodes — plain accumulation.
            targets.extend(nid for (nid,) in rows)
            seen.update(nid for (nid,) in rows)

        # Shared subqueries
        missing_q = ("n.id NOT IN (SELECT node_id FROM node_metadata_kv "
                     "WHERE key='question')")
        missing_s = ("n.id NOT IN (SELECT node_id FROM node_metadata_kv "
                     "WHERE key='situation' AND length(value) > 5)")
        missing_r = ("n.id NOT IN (SELECT node_id FROM node_metadata_kv "
                     "WHERE key='reasoning' AND length(value) > 5)")
        base = "SELECT n.id FROM nodes n WHERE n.archived=0 AND %s"

        # Tier 1: missing all three
        _run(base % ' AND '.join([missing_q, missing_s, missing_r]), max_nodes)

        # Tier 2: missing any two
        for combo in [(missing_q, missing_s), (missing_q, missing_r),
                      (missing_s, missing_r)]:
            _run(base % ' AND '.join(combo), max_nodes - len(targets))
            if len(targets) >= max_nodes:
                return targets

        # Tier 3: missing any one
        for single in (missing_q, missing_s, missing_r):
            _run(base % single, max_nodes - len(targets))
            if len(targets) >= max_nodes:
                return targets

        return targets

    def _build_proposals(self, target_ids):
        """Build proposals with full context for each target node."""
        # Batch load all rich nodes
        rich_nodes = self.brain.get_node(target_ids)
        if isinstance(rich_nodes, dict) is False:
            rich_nodes = {}

        proposals = []
        for nid in target_ids:
            rich = rich_nodes.get(nid)
            if not rich:
                continue

            # Determine what's missing
            metadata = rich.get('_metadata', {})
            has_question = bool(metadata.get('question', '').strip())
            has_situation = bool(rich.get('situation', '').strip())
            has_reasoning = bool(metadata.get('reasoning', '').strip())

            if has_question and has_situation and has_reasoning:
                continue  # Nothing to heal

            # Load conversation context
            conversation = self.brain.get_conversation_around(
                node_id=nid, before=10, after=5)

            proposals.append({
                'node_id': nid,
                'rich_node': rich,
                'title': rich.get('title', ''),
                'needs_question': not has_question,
                'needs_situation': not has_situation,
                'needs_reasoning': not has_reasoning,
                'conversation': conversation,
            })

        return proposals
