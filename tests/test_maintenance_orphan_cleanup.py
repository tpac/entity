"""LogsDAL.run_maintenance graph-orphan cleanup — the two-sided delete contract.

2026-06-12 audit: the cleanup deleted `edges` rows whose endpoints vanished
but left their `edge_relations` rows behind — 17,982 stranded relation rows
(84% of active) accumulated, invisible to every JOIN-based read. The contract
now:

1. Deleting an orphaned edge also deletes ITS relation rows (scoped).
2. Pre-existing orphan relations (no parent edge at all) are NOT touched —
   they are the recovery corpus for the trace-based restoration effort.
3. Healthy edges and their relations survive untouched.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


class TestMaintenanceOrphanCleanup(BrainTestBase):
    needs_embedder = False

    def _node(self, title):
        return self.brain.remember(type='fact', title=title, content='c',
                                   encoding_source='anchor')['id']

    def test_orphan_edge_delete_takes_relations_along(self):
        a, b, c = self._node('A'), self._node('B'), self._node('C')
        # Healthy edge A—B and soon-to-be-orphaned edge A—C
        self.brain._graph.add_relation(a, b, 'extends', description='healthy')
        self.brain._graph.add_relation(a, c, 'extends', description='doomed')
        doomed_edge = self.brain._graph.get_edge_id(a, c)
        healthy_edge = self.brain._graph.get_edge_id(a, b)
        # Pre-existing orphan relation (no parent edge at all) — must SURVIVE
        # cleanup; it's recovery corpus, not garbage.
        self.brain.conn.execute(
            "INSERT INTO edge_relations (edge_id, relation, description, "
            "weight, encoding_source, created_at) "
            "VALUES ('edg_deadbeef', 'extends', 'pre-existing orphan', "
            "0.5, 'test', '2026-01-01T00:00:00+00:00')")
        # Hard-delete node C (simulates the condition the cleanup targets)
        self.brain.conn.execute("DELETE FROM nodes WHERE id = ?", (c,))
        self.brain.conn.commit()

        stats = self.brain._logs_dal.run_maintenance(graph_conn=self.brain.conn)

        def edge_exists(eid):
            return self.brain.conn.execute(
                "SELECT 1 FROM edges WHERE edge_id = ?", (eid,)).fetchone()

        def relations_for(eid):
            return self.brain.conn.execute(
                "SELECT COUNT(*) FROM edge_relations WHERE edge_id = ?",
                (eid,)).fetchone()[0]

        # Doomed edge AND its relations are gone — no new orphans
        self.assertIsNone(edge_exists(doomed_edge))
        self.assertEqual(relations_for(doomed_edge), 0)
        self.assertGreaterEqual(stats.get('orphaned_edge_relations', 0), 1)
        # Healthy edge untouched
        self.assertIsNotNone(edge_exists(healthy_edge))
        self.assertEqual(relations_for(healthy_edge), 1)
        # Pre-existing orphan preserved for recovery
        self.assertEqual(relations_for('edg_deadbeef'), 1)


class TestLogsRetention(BrainTestBase):
    """run_maintenance's logs-side retention: session_state and boot_renders
    age out at 30 days; live (recently-updated) rows survive. Added with the
    2026-08-18 DB-stewardship pass — before it, both tables grew unbounded
    (measured: 2,099 sessions back four months, 1,441 full boot texts)."""
    needs_embedder = False

    OLD = '2026-01-01T00:00:00+00:00'

    def test_dead_session_state_pruned_live_kept(self):
        dal = self.brain._session_state
        dal.set('dead-session', '_session_context', '{}')
        dal.set('live-session', '_session_context', '{}')
        # Backdate the dead session past the 30-day window.
        self.brain.logs_conn_w.execute(
            "UPDATE session_state SET updated_at = ? WHERE session_id = 'dead-session'",
            (self.OLD,))
        self.brain.logs_conn_w.commit()

        stats = self.brain._logs_dal.run_maintenance(graph_conn=None)

        self.assertEqual(stats.get('session_state_pruned'), 1)
        rows = {r[0] for r in self.brain.logs_conn.execute(
            "SELECT session_id FROM session_state").fetchall()}
        self.assertNotIn('dead-session', rows)
        self.assertIn('live-session', rows)

    def _session_idle_for(self, sid, days):
        from servers.clock import iso_cutoff
        self.brain._session_state.set(sid, '_session_context', '{}')
        self.brain.logs_conn_w.execute(
            "UPDATE session_state SET updated_at = ? WHERE session_id = ?",
            (iso_cutoff(days=days), sid))
        self.brain.logs_conn_w.commit()

    def _sessions_after_retention(self):
        # The brain-level pass reads `session.live_days`; the stale backup
        # gate only skips the orphan arm, never retention.
        from unittest.mock import patch
        with patch('servers.db_backup.ensure_backup_fresh', return_value=False):
            self.brain.run_logs_maintenance()
        return {r[0] for r in self.brain.logs_conn.execute(
            "SELECT session_id FROM session_state").fetchall()}

    def test_session_live_days_sets_the_prune_window(self):
        self._session_idle_for('idle-45d', 45)
        self.brain.set_config('session.live_days', '60')
        self.assertIn('idle-45d', self._sessions_after_retention())
        self.brain.set_config('session.live_days', '')   # back to the 30-day default
        self.assertNotIn('idle-45d', self._sessions_after_retention())

    def test_session_live_days_bad_override_is_bounded(self):
        self._session_idle_for('idle-2d', 2)
        self._session_idle_for('idle-12h', 0.5)
        self.brain.set_config('session.live_days', '0')     # floored at 1 day
        rows = self._sessions_after_retention()
        self.assertNotIn('idle-2d', rows)
        self.assertIn('idle-12h', rows)
        self._session_idle_for('idle-20d', 20)
        self.brain.set_config('session.live_days', 'soon')  # unparseable: the default
        self.assertIn('idle-20d', self._sessions_after_retention())

    def test_old_boot_renders_pruned_recent_kept(self):
        self.brain._logs_dal.record_boot_render('old-sess', 'u', 'p', 'old boot text')
        self.brain._logs_dal.record_boot_render('new-sess', 'u', 'p', 'new boot text')
        self.brain.logs_conn_w.execute(
            "UPDATE boot_renders SET created_at = ? WHERE session_id = 'old-sess'",
            (self.OLD,))
        self.brain.logs_conn_w.commit()

        stats = self.brain._logs_dal.run_maintenance(graph_conn=None)

        self.assertEqual(stats.get('boot_renders_pruned'), 1)
        rows = {r[0] for r in self.brain.logs_conn.execute(
            "SELECT session_id FROM boot_renders").fetchall()}
        self.assertNotIn('old-sess', rows)
        self.assertIn('new-sess', rows)
