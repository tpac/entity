"""
brain — BrainAssembly Mixin

Extracted from brain.py monolith. Methods are mixed into the Brain class
via multiple inheritance. All methods reference self.conn, self.get_config, etc.
which are provided by Brain.__init__.
"""

from .clock import iso_cutoff
from typing import Any, Dict

from .brain_constants import (
    ZERO_MEMORY_MIN_DAYS,
    ZERO_MEMORY_MIN_LIVED,
)


class BrainAssemblyMixin:
    """Assembly methods — gathers context for hooks (boot stats, health check,
    the boot render)."""

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
        logs what it served here. The dashboard's Sessions→Boot view reads this to show "what
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
