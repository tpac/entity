"""
brain — BrainSurface Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from . import embedder
from .text_processing import split_identifier
from .clock import iso_cutoff
from typing import Any, Dict, List, Optional
import os
import re

from .brain_constants import (
    ZERO_MEMORY_MIN_DAYS,
    ZERO_MEMORY_MIN_LIVED,
)

DESTRUCTIVE_PATTERNS = [
    re.compile(r'rm\s+(-[rf]+\s+|.*--force)', re.IGNORECASE),
    re.compile(r'git\s+worktree\s+remove', re.IGNORECASE),
    re.compile(r'git\s+reset\s+--hard', re.IGNORECASE),
    re.compile(r'git\s+clean\s+-[fd]', re.IGNORECASE),
    re.compile(r'git\s+checkout\s+--\s', re.IGNORECASE),
    re.compile(r'git\s+push\s+.*--force', re.IGNORECASE),
    re.compile(r'DROP\s+TABLE', re.IGNORECASE),
    re.compile(r'DELETE\s+FROM', re.IGNORECASE),
    re.compile(r'TRUNCATE', re.IGNORECASE),
    re.compile(r'\brmdir\b', re.IGNORECASE),
    re.compile(r'xargs\s+rm', re.IGNORECASE),
]


class BrainAssemblyMixin:
    """Assembly methods — gathers context for hooks (boot, pre-edit, procedures)."""

    def suggest(self, context: Optional[str] = None, file: Optional[str] = None,
               screen: Optional[str] = None, action: Optional[str] = None,
               limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-query recall with type boosts, locked node boost, file-term relevance.
        Returns dict with suggestions list and query_count.
        """
        if limit is None:
            limit = self.get_config('suggestion_limit', 5)

        # Pre_edit cost surgery (2026-05-08): suggest() previously fanned out
        # the inputs into up to 7 separate full recalls per call (context +
        # joined filename tokens + each individual token > 2 chars + screen +
        # action). Single-word tokens like 'servers', 'scales', 'edit'
        # produced thousands of low-signal matches per query, and 7×O(N)
        # cosine scans per Edit was the dominant CPU + memory pressure
        # source (watchdog confirmed). Now: ONE query — the most-specific
        # input available. context (e.g., "editing path/to/file.py") wins
        # because it's the whole signal in one string; the fan-out lost
        # precision while burning compute.
        queries = []
        if context:
            queries.append(context)
        elif file:
            # Fallback when caller didn't compose a context — use the joined
            # tokens. Skip the per-token fan-out for the same precision-vs-
            # cost reason.
            file_tokens = split_identifier(file)
            if file_tokens:
                queries.append(' '.join(file_tokens))
        elif screen:
            queries.append(screen)
        elif action:
            queries.append(action)

        if not queries:
            return {'suggestions': [], 'reason': 'no context provided'}

        # Run recall for each query
        seen = set()
        all_results = []
        pool_multiplier = self.get_config('recall_pool_multiplier', 2)
        recall_limit = max(limit * pool_multiplier, 15)

        for q in queries:
            result = self.recall(query=q, limit=recall_limit, source='internal')
            results = result.get('results', [])
            for r in results:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    all_results.append(r)

        # Second pass: check edge neighbors of top results for locked nodes
        try:
            top_ids = [r['id'] for r in all_results[:10]]
            if top_ids:
                placeholders = ','.join('?' * len(top_ids))
                neighbor_rows = self.conn.execute(f'''
                    SELECT DISTINCT n.id, n.type, n.title, n.content,
                           n.access_count, n.locked, n.archived, n.last_accessed, n.created_at
                    FROM edges e
                    JOIN nodes n ON (n.id = CASE WHEN e.source_id = n.id THEN e.target_id ELSE e.source_id END)
                    WHERE (e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders}))
                      AND n.locked = 1 AND n.archived = 0
                      AND n.id NOT IN ({placeholders})
                    LIMIT 20
                ''', top_ids + top_ids + top_ids).fetchall()

                # Scope veil (outward-only — suggest has no session):
                # the locked-neighbor pull bypasses recall's gate.
                _veil = self.scope_veil('')
                for row in neighbor_rows:
                    nid = row[0]
                    if nid in _veil:
                        continue
                    if nid not in seen:
                        seen.add(nid)
                        all_results.append({
                            'id': row[0], 'type': row[1], 'title': row[2], 'content': row[3],
                            'access_count': row[4], 'locked': row[5] == 1, 'archived': row[6] == 1,
                            'last_accessed': row[7], 'created_at': row[8],
                            '_edge_neighbor': True
                        })
        except Exception as e:
            self._log_error('suggest_edge_neighbors', e, 'fetching edge neighbors for suggestions')

        # Scoring
        rule_boost = self.get_config('boost_rule', 1.3)
        decision_boost = self.get_config('boost_decision', 1.2)
        locked_boost = self.get_config('boost_locked', 1.5)
        edge_neighbor_penalty = self.get_config('penalty_edge_neighbor', 0.85)
        file_relevance_max = self.get_config('file_relevance_bonus', 0.15)

        # File-specific terms
        file_terms = set()
        if file:
            clean = file.replace('/', ' ').replace('\\', ' ').replace(os.path.splitext(file)[1], '')
            clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
            clean = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', clean)
            for t in clean.lower().split(r'[\s\-_]+'):
                if len(t) > 2:
                    file_terms.add(t)
        if screen:
            file_terms.add(screen.lower())

        ranked = []
        for r in all_results:
            boost = 1.0
            if r.get('type') == 'rule':
                boost = rule_boost
            if r.get('type') == 'decision':
                boost = decision_boost
            if r.get('locked'):
                boost *= locked_boost
            if r.get('_edge_neighbor'):
                boost *= edge_neighbor_penalty

            # File relevance — match against title + content snippet
            file_relevance = 0
            if file_terms:
                node_text = f"{r.get('title', '')} {(r.get('content') or '')[:300]}".lower()
                for term in file_terms:
                    if term in node_text:
                        file_relevance += 1
                file_relevance = (file_relevance / len(file_terms)) * file_relevance_max

            r['suggest_score'] = ((r.get('effective_activation') or 0.5) + file_relevance) * boost
            ranked.append(r)

        ranked.sort(key=lambda r: -r['suggest_score'])

        # Locked node promotion
        selected = ranked[:limit]
        selected_ids = {r['id'] for r in selected}

        if file_terms:
            missed_locked = [
                r for r in ranked
                if r['id'] not in selected_ids and r.get('locked') and
                   r.get('type') in ('rule', 'decision') and
                   any(t in f"{r.get('title', '')} {(r.get('content') or '')[:300]}".lower() for t in file_terms)
            ]

            for locked_node in missed_locked:
                worst_idx = -1
                worst_score = float('inf')
                for i in range(len(selected) - 1, -1, -1):
                    if not selected[i].get('locked') and selected[i]['suggest_score'] < worst_score:
                        worst_idx = i
                        worst_score = selected[i]['suggest_score']
                if worst_idx >= 0:
                    selected[worst_idx] = locked_node
                    selected_ids.add(locked_node['id'])
                else:
                    break

            selected.sort(key=lambda r: -r['suggest_score'])

        suggestions = [
            {
                'id': r['id'],
                'type': r.get('type'),
                'title': r.get('title'),
                'content': r.get('content', '')[:300] if r.get('content') else None,
                'locked': r.get('locked', False),
                'relevance': r['suggest_score'],
                'reason': self._suggest_reason(r, queries)
            }
            for r in selected
        ]

        return {'suggestions': suggestions, 'query_count': len(queries)}

    def _suggest_reason(self, node: Dict[str, Any], queries: List[str]) -> str:
        """Generate reason string for suggestion."""
        lower_title = (node.get('title') or '').lower()
        for q in queries:
            terms = [w for w in q.lower().split() if len(w) > 2]
            for t in terms:
                if t in lower_title:
                    return f'matches "{t}" from context'
        return 'related via graph connections'

    def boot_stats(self) -> Dict[str, Any]:
        """Brain totals and the Zero-Memory gate, for render_boot's header."""
        # Zero-Memory gate — the Nursery's spoken half; brain_voice renders
        # the block. Young = fewer than ZERO_MEMORY_MIN_LIVED non-seed
        # memories OR younger than ZERO_MEMORY_MIN_DAYS since the seeds were
        # born; it retires only when both are exceeded. The age test is a
        # bound iso_cutoff against created_at (transaction-time column →
        # wall-clock anchor is correct; no timestamp parsing in Python).
        zero_memory = {'active': False, 'lived': 0}
        try:
            stats = self._nodes.nursery_gate_stats(
                iso_cutoff(days=ZERO_MEMORY_MIN_DAYS))
            zero_memory = {
                'active': stats['born'] and (
                    stats['lived'] < ZERO_MEMORY_MIN_LIVED
                    or not stats['old_enough']),
                'lived': stats['lived'],
            }
        except Exception as e:
            self._log_error('zero_memory_gate', e,
                            'boot_stats zero-memory gate — block skipped')

        return {
            'zero_memory': zero_memory,
            'total_nodes': self._get_node_count(),
            'total_locked': self._nodes.count_locked(),
        }

    def health_check(self, session_id: str = 'boot', auto_fix: bool = True) -> Dict[str, Any]:
        """
        Check brain health: orphaned locked nodes, stale contexts, stale staged learnings.
        Auto-fix: enrich missed nodes, promote staged learnings.
        """
        issues = []
        actions = []
        ts = self.now()

        # Check for orphaned locked nodes
        orphaned = self.conn.execute('''
            SELECT n.id, n.title FROM nodes n
            WHERE n.locked = 1 AND n.archived = 0
            AND n.id NOT IN (SELECT source_id FROM edges)
            AND n.id NOT IN (SELECT target_id FROM edges)
            LIMIT 5
        ''').fetchall()

        if orphaned:
            issues.append({
                'type': 'orphaned_locked_nodes',
                'severity': 'low',
                'message': f'{len(orphaned)} locked nodes with no connections.',
                'nodes': [{'id': r[0], 'title': r[1]} for r in orphaned]
            })

        # 4. Check for stale context nodes
        stale_count_row = self.conn.execute('''
            SELECT COUNT(*) FROM nodes
            WHERE type = 'context' AND locked = 0 AND archived = 0
            AND created_at < ?
        ''', (iso_cutoff(days=7),)).fetchone()
        stale_count = stale_count_row[0] if stale_count_row else 0

        if stale_count > 10:
            issues.append({
                'type': 'stale_contexts',
                'severity': 'low',
                'message': f'{stale_count} context nodes older than 7 days.'
            })
            if auto_fix:
                stale_ids = [r[0] for r in self.conn.execute('''
                    SELECT id FROM nodes
                    WHERE type = 'context' AND locked = 0 AND archived = 0
                    AND created_at < ?
                ''', (iso_cutoff(days=14),)).fetchall()]
                # Per-archive node_archived rows via the ONE trace writer.
                # This path runs at every session boot and never crosses the
                # dispatch chokepoint, so without a direct emit these archives
                # became invisible when archive_node's inline trace died
                # (plan step 8, review 2026-08-06). Emitted per node, after
                # that node's archive commits — same rationale as the junk
                # purge in daemon_hooks.
                from servers.mutation_emitter import emit_mutation_traces
                from servers.clock import brain_today
                # Chain deliberately DISTINCT from the junk purge's
                # maint-{date}-mutation (s2): these rows land at s0
                # (hook: prefix), and reusing the same literal string would
                # make one chain_id span two scales — get_chains stamps a
                # chain's scale from the first event seen, so every
                # prefix-based reader assumes scale-homogeneity per chain.
                maint_chain = ('maint-%s-integrity'
                               % brain_today(self).strftime('%Y%m%d'))
                for sid in stale_ids:
                    arch = self.archive_node(
                        sid, archived_by='hook:integrity',
                        reason='context node older than 14 days')
                    if arch.get('ok'):
                        emit_mutation_traces(
                            self, 'health_check',
                            {'nodes': {'archived': [{
                                'node_id': arch.get('node_id') or sid,
                                'type': arch.get('type') or '',
                                'title': arch.get('title') or '',
                                'archived_by': 'hook:integrity',
                                'encoding_source': 'hook:integrity',
                                'reason': arch.get('reason') or '',
                                'edge_relations': arch.get('edge_relations', []),
                                'vectors_deleted': arch.get('vectors_deleted', 0),
                            }]}},
                            chain_id=maint_chain)
                actions.append('Auto-archived %d context nodes older than 14 days' % len(stale_ids))

        return {
            'healthy': not any(i['severity'] == 'high' for i in issues),
            'issues': issues,
            'actions': actions,
            'checked_at': ts
        }

    def pre_edit(self, file: str, tool_name: str = '',
                 ctx=None) -> dict:
        """
        Batch pre-edit call combining all lookups into one.
        Replaces 8 sequential HTTP calls from the old architecture.

        Args:
            file: Filename being edited
            tool_name: Raw tool name when supplied; empty when absent.
            ctx: The caller's SessionContext, resolved at dispatch from the
                session id the call carries. Nothing here reads it today.

        Returns:
            Dict with suggestions, procedures, context_files, timings

        No cache here: brain.recall's result cache + single-flight gate
        covers every recall caller, keyed by the actual query.
        """
        import time as _time
        t0 = _time.time()
        timings = {}

        # 1. Suggest
        t1 = _time.time()
        suggest_result = self.suggest(
            context=f"editing {file}",
            file=file,
            action=tool_name.lower(),
            limit=10
        )
        timings['suggest_ms'] = round((_time.time() - t1) * 1000)

        # 2. Procedures
        t2 = _time.time()
        proc_result = self.procedure_trigger('pre_edit', {'file': file, 'tool': tool_name})
        timings['procedures_ms'] = round((_time.time() - t2) * 1000)

        # 3. Context files (nodes of type 'file' matching the edited filename)
        context_files = []
        try:
            # File-type nodes encode their topic in the title; content
            # snippet carries the body.
            cursor = self.conn.execute(
                "SELECT id, title, content, updated_at FROM nodes WHERE type = 'file' AND archived = 0 AND title LIKE ? LIMIT 3",
                (f'%{file}%',)
            )
            for row in cursor.fetchall():
                context_files.append({
                    'id': row[0], 'title': row[1], 'summary': (row[2] or '')[:200],
                    'topic': '', 'last_updated': row[3],
                })
        except Exception as _e:
            self._log_error("pre_edit", _e, "cursor = self.conn.execute(")

        timings['total_ms'] = round((_time.time() - t0) * 1000)

        return {
            'suggestions': suggest_result.get('suggestions', []),
            'procedures': proc_result.get('matched', []),
            'context_files': context_files,
            'embedder_ready': embedder.is_ready(),
            'debug_enabled': self.get_debug_status(),
            'timings': timings,
        }

    def safety_check(self, command: str) -> dict:
        """
        Check a bash command against destructive patterns and brain safety nodes.

        Args:
            command: The bash command string to check

        Returns:
            Dict with destructive (bool), risk_level, warnings
        """
        # Check against destructive patterns
        matched_patterns = []
        for pattern in DESTRUCTIVE_PATTERNS:
            if pattern.search(command):
                matched_patterns.append(pattern.pattern)

        if not matched_patterns:
            return {'destructive': False, 'risk_level': 'none'}

        # Destructive command detected — query brain for safety context
        warnings = []

        try:
            # Recall relevant safety context
            recall_result = self.recall(command, limit=5, source='internal')
            results = recall_result.get('results', [])

            safety_types = {'rule', 'decision', 'constraint', 'convention', 'lesson'}
            for r in results:
                node_type = r.get('type', '')
                if node_type in safety_types or r.get('locked'):
                    warnings.append({
                        'id': r.get('id'),
                        'type': node_type,
                        'title': r.get('title', ''),
                        'content': (r.get('content', '') or '')[:300]
                    })

        except Exception as e:
            self._log_error("safety_check", e, "querying brain for safety context")

        # Determine risk level
        if warnings:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'destructive': True,
            'risk_level': risk_level,
            'warnings': warnings,
            'matched_patterns': matched_patterns,
        }

    def procedure_trigger(self, trigger_type: str, context: dict = None) -> dict:
        """
        Find and return procedures matching a trigger type.

        Args:
            trigger_type: 'session_start', 'pre_edit', 'pre_compact', etc.
            context: Optional context dict with trigger-specific data

        Returns:
            Dict with matched procedures list
        """
        context = context or {}
        matched = []

        try:
            # Procedure-trigger matching: trigger types
            # ("session_start", "pre_edit", "pre_compact") and file_name
            # filters match against title + content only. Procedure
            # nodes that need explicit tagging should include the trigger
            # in their content body (e.g. "Run on session_start").
            cursor = self.conn.execute(
                "SELECT id, title, content FROM nodes WHERE type = 'procedure' AND archived = 0 AND locked = 1"
            )
            for row in cursor.fetchall():
                node_id, title, content = row
                content_lower = (content or '').lower()
                title_lower = (title or '').lower()
                # Combined search surface for trigger matching.
                search_text = content_lower + ' ' + title_lower

                # Check if procedure matches trigger type
                if trigger_type in search_text:
                    # Parse procedure content for steps
                    steps = content or ''
                    category = 'general'
                    if 'session_start' in search_text:
                        category = 'session_start'
                    elif 'pre_edit' in search_text:
                        category = 'pre_edit'
                    elif 'pre_compact' in search_text:
                        category = 'pre_compact'

                    # Check file-specific procedures
                    if trigger_type == 'pre_edit' and 'file' in context:
                        file_name = context['file'].lower()
                        if file_name not in search_text:
                            # Check for wildcard patterns
                            if '*' not in content_lower:
                                continue

                    matched.append({
                        'id': node_id,
                        'title': title,
                        'steps': steps,
                        'category': category,
                    })
        except Exception as _e:
            self._log_error("procedure_trigger", _e, "matching procedure nodes to trigger context")

        return {'matched': matched}

    # ── Formatted boot context ──────────────

    def format_boot_context(self, session_id: str = '') -> str:
        """The boot text for Claude's context window: BrainVoice.render_boot
        for this session (its Frame comes from session_id)."""
        from .brain_voice import BrainVoice
        return BrainVoice(self).render_boot(session_id=session_id)

    def record_boot_render(self, session_id: str, text: str) -> None:
        """Persist the exact boot text served to a session at SessionStart,
        stamped with the session's project and counterpart — read through
        the same per-dimension resolver (_scope_value) that stamps node
        provenance, so the record says where and with whom the session booted.

        The context_boot command is read-only re: the knowledge graph, but it
        logs what it served here. The dashboard's Streams→Boot view reads this to show "what
        actually got to boot." Writes through LogsDAL's write boundary
        (dedicated write connection under write_lock — see dal_logs.py).

        Best-effort: a failure here must NEVER break boot (loud to stderr, not
        raised). Skips when there's no session or no text (non-SessionStart
        renders shouldn't pollute the log)."""
        if not session_id or not text:
            return
        try:
            scope = self.session_scope(session_id) or {}
            self._logs_dal.record_boot_render(
                session_id, scope.get('counterpart', ''),
                scope.get('project', ''), text)
        except Exception as e:
            import sys
            print('[brain] record_boot_render failed: %s' % e,
                  file=sys.stderr, flush=True)
