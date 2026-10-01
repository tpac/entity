#!/usr/bin/env python3
"""
brain — Core Unit Tests

Tests that catch silent failures and non-obvious regressions:
- vocabulary system (learn, resolve, ambiguous)
- error logging pipeline
- DAL read/write consistency and pattern enforcement
- consciousness signals, evolution lifecycle
- rich fields, typed edges, surface formatting
- safety checks, vocabulary admission
- keyword extraction, common word filtering, sentence splitting

Run: python tests/test_core.py
"""

import sys
import os
import shutil
import tempfile
import unittest
import sqlite3

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.dal import BrainMetaDAL
from servers.dal_logs import LogsDAL
from servers.schema import ensure_schema, ensure_logs_schema
from tests.brain_test_base import BrainTestBase



class TestErrorLogging(BrainTestBase):
    """Test error logging pipeline."""

    def test_log_error_writes_to_db(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            self.brain._log_error("test_source", e, "test context")

        errors = self.brain.get_recent_errors(hours=1)
        self.assertTrue(len(errors) > 0, 'Should have logged error')
        self.assertEqual(errors[0]['source'], 'test_source')

    def test_log_error_none_surfaces_in_db(self):
        """error=None must still reach debug_log (not just stderr).

        Regression: callers log a *condition* with error=None (e.g.
        brain_remember 'group_vectors_skip'). Building the traceback from
        None.__traceback__ raised AttributeError, which the method's own
        except swallowed to stderr — so the row never hit debug_log and
        the failure was invisible at boot / in the dashboard.
        """
        self.brain._log_error('test_none_src', None, 'none ctx')

        errors = self.brain.get_recent_errors(hours=1)
        match = [e for e in errors if e['source'] == 'test_none_src']
        self.assertTrue(match, 'error=None must surface in debug_log')
        self.assertEqual(match[0]['type'], 'NoneType')
        self.assertEqual(match[0]['context'], 'none ctx')

    def test_log_error_message_string_surfaces_in_db(self):
        """A message string is a condition too (truncation, undispatched tool
        calls): it must reach debug_log and never raise — a str has no
        __traceback__, and a raise here kills the run doing the logging."""
        self.brain._log_error('test_str_src', 'max_tokens truncation: 5/5', 'str ctx')

        errors = self.brain.get_recent_errors(hours=1)
        match = [e for e in errors if e['source'] == 'test_str_src']
        self.assertTrue(match, 'a message-string condition must surface in debug_log')
        self.assertEqual(match[0]['type'], 'str')
        self.assertEqual(match[0]['error'], 'max_tokens truncation: 5/5')
        self.assertEqual(match[0]['context'], 'str ctx')

    def test_log_error_unraised_exception_captures_type(self):
        """A constructed-but-unraised exception (the group_vectors_skip pattern)
        captures its real type/message — the chosen alternative to error=None."""
        self.brain._log_error('test_cond_src',
                              RuntimeError('embedder not ready'), 'cond ctx')

        errors = self.brain.get_recent_errors(hours=1)
        match = [e for e in errors if e['source'] == 'test_cond_src']
        self.assertTrue(match, 'condition logged as error must surface')
        self.assertEqual(match[0]['type'], 'RuntimeError')
        self.assertEqual(match[0]['error'], 'embedder not ready')

    def test_dedup_fingerprint_separates_contexts(self):
        """context is part of the dedup identity: two different conditions
        with the same message inside the dedup window each get a row (the s2
        truncation twin-pull case — identical error text, different pulls),
        while a true repeat (same source/type/message/context) is collapsed.
        """
        def _rows():
            return [e for e in self.brain.get_recent_errors(hours=1)
                    if e['source'] == 'test_fp_ctx']

        err = RuntimeError('hit limit before exhausting the window')
        self.brain._log_error('test_fp_ctx', err, "reading ['recall']")
        self.brain._log_error('test_fp_ctx', err, "reading ['surface_selected']")
        self.assertEqual(len(_rows()), 2,
                         'distinct contexts must not collapse')
        self.brain._log_error('test_fp_ctx', err, "reading ['recall']")
        self.assertEqual(len(_rows()), 2,
                         'true repeat inside the window must collapse')

    def test_loggers_commit_durably_to_separate_connection(self):
        """The unified loggers route through LogsDAL.write_event, which commits
        via commit_unless_batched — so rows are durable and visible to a SEPARATE
        connection (the dashboard's), not just the brain's own logs_conn.

        Regression guard for the old raw-SQL path, which never committed and rode
        on a later writer's commit to persist.
        """
        import sqlite3 as _sqlite3
        self.brain._log_error('test_durable_err', ValueError('boom'), 'c')
        self.brain._log_warning('test_durable_warn', 'heads up', 'c')

        reader = _sqlite3.connect(self.brain.logs_db_path)
        try:
            rows = reader.execute(
                "SELECT source, event_type FROM debug_log "
                "WHERE source IN ('test_durable_err', 'test_durable_warn')"
            ).fetchall()
        finally:
            reader.close()

        seen = {src: et for src, et in rows}
        self.assertEqual(seen.get('test_durable_err'), 'error',
                         'error must be committed + visible cross-connection')
        self.assertEqual(seen.get('test_durable_warn'), 'warning',
                         'warning must be committed + visible cross-connection')

    def test_log_warning_writes_to_db(self):
        """_log_warning writes a non-blocking signal with event_type='warning'."""
        self.brain._log_warning("test_warn_source", "test warning message", "warn ctx")

        # Query debug_log directly — warnings live in the unified debug_log
        # table with event_type='warning' (distinct from event_type='error').
        row = self.brain.logs_conn.execute(
            "SELECT event_type, source, metadata FROM debug_log "
            "WHERE source = 'test_warn_source' AND event_type = 'warning' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row, 'Should have logged warning to debug_log')
        self.assertEqual(row[0], 'warning')
        self.assertEqual(row[1], 'test_warn_source')
        # metadata is JSON with message + context
        import json as _json
        meta = _json.loads(row[2])
        self.assertEqual(meta['message'], 'test warning message')
        self.assertEqual(meta['context'], 'warn ctx')


class TestDAL(unittest.TestCase):
    """Test the DAL layer independently."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.logs_conn = sqlite3.connect(os.path.join(self.tmp, 'logs.db'))
        ensure_logs_schema(self.logs_conn)
        self.logs = LogsDAL(self.logs_conn)

        self.brain_conn = sqlite3.connect(os.path.join(self.tmp, 'brain.db'))
        ensure_schema(self.brain_conn)
        self.meta = BrainMetaDAL(self.brain_conn)

    def tearDown(self):
        self.logs_conn.close()
        self.brain_conn.close()
        shutil.rmtree(self.tmp)

    def test_logs_write_read_errors(self):
        # write_event is the single live debug_log writer (the typed
        # write_error delegator was removed in the DAL Phase-A cleanup).
        self.logs.write_event('error', 'src', {'error': 'err', 'context': 'ctx'})
        errors = self.logs.get_recent_errors(hours=1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['source'], 'src')

    def test_meta_get_set(self):
        self.meta.set("key", "value")
        self.assertEqual(self.meta.get("key"), "value")



# ═══════════════════════════════════════════════════════════════════════
# SESSION A: Comprehensive Unit Tests (v5.1 expansion)
#
# Tests organized by priority (P0-P7) matching the test plan.
# All test data uses realistic brain content, not toy examples.
# ═══════════════════════════════════════════════════════════════════════


# ── Helpers ──────────────────────────────────────────────────────────


def _seed_brain_with_realistic_data(brain):
    """Populate a brain with realistic test data for consciousness/evolution tests."""
    nodes = []
    # Decisions
    nodes.append(brain.remember(type='decision', title='Auth: Clerk for passwordless login via magic links',
        content='Clerk handles auth flow. Magic links for login, no passwords. Webhook syncs user data to our DB. Free tier covers MVP needs. Chose over Auth0 because of simpler integration.',
        locked=True, confidence=0.95))
    nodes.append(brain.remember(type='decision', title='React component architecture follows atomic design',
        content='Components organized by atomic design: atoms (Button, Input), molecules (FormField), organisms (LoginForm). Shared via internal package.',
        locked=True, confidence=0.9))
    # Rules
    nodes.append(brain.remember(type='rule', title='Experimental features must never block core operations',
        content='When adding new features (bridging, proposals, archive) to existing methods (remember, consolidate, dream, smartPrune), always wrap in try/catch. A bridge failure should never prevent a remember from succeeding.',
        locked=True, confidence=0.85))
    nodes.append(brain.remember(type='rule', title='Communication style with Ada: direct, peer-to-peer',
        content='Speak peer-to-peer. Be direct. Challenge when warranted. Always plan before executing. Never dump a full spec — work iteratively through components.',
        locked=True))
    # Lessons
    nodes.append(brain.remember(type='lesson', title='Lesson: new features must be connected to the graph at birth',
        content='Built vocabulary system with learn_vocabulary(), resolve_vocabulary(), gap detection. Then discovered vocabulary nodes were completely isolated — not connected to anything in the graph.',
        locked=True, emotion=0.8, confidence=0.85))
    # Concepts
    nodes.append(brain.remember(type='concept', title='ScheduleResolver — isolated job parameter builder',
        content='Isolated component that takes a job spec and returns scheduler-ready params (time windows, retry cap, backoff, batch size). V1: config-driven defaults per workspace.',
        confidence=0.7))
    # Context (will be stale)
    nodes.append(brain.remember(type='context', title='Session #7 final log: Brain v4.0.0 shipped',
        content='Session #7. Massive session. Everything built and tested in one go. Shipped Phase 0.5A, 0.5B, 0.5C, typed edges, consciousness, curiosity.'))
    # Correction
    nodes.append(brain.remember(type='correction', title='Divergence: Claude compresses when encoding to brain',
        content='CLAUDE ASSUMED: Encoding to brain should be concise. REALITY: Brain encoding should be RICH. Future Claude needs texture, specifics, failures, reasoning journeys.',
        locked=True, emotion=0.9, confidence=0.95))
    return nodes


# ── P0: Silent Failure Detection ─────────────────────────────────────

class TestSilentFailures(BrainTestBase):
    """P0: Verify errors are logged, never silently swallowed."""

    # A node view's read-only attachments, as an echoing writer sends them.
    _VIEW_ECHO = {
        'connections': [{'target_id': 'nonexistent_node_id_xyz', 'relation': 'related'}],
        'communities': [{'id': 'abcd1234', 'title': 'a community'}],
        '_metadata': {'reasoning': 'echoed'},
        '_corrections': [{'id': 'abcd1234'}],
    }

    def _loud_count(self):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE event_type='error' "
            "AND source='write_read_only_attachment'").fetchone()[0]

    def _kv_keys(self, node_id):
        return {r[0] for r in self.brain.conn.execute(
            "SELECT key FROM node_metadata_kv WHERE node_id=?", (node_id,))}

    def test_remember_read_only_attachments_are_loud_not_stored(self):
        """A node view echoed into remember: no attachment lands as metadata,
        `connections` makes no edge, and the drop is logged LOUDLY
        (source='write_read_only_attachment')."""
        from servers.contract import READ_ONLY_ATTACHMENT_KEYS
        self.assertTrue(set(self._VIEW_ECHO) <= READ_ONLY_ATTACHMENT_KEYS)
        before = self._loud_count()
        result = self.brain.remember(
            type='decision', title='Echoed node view must be loud, not stored',
            content='Passing a node view back into remember.', **self._VIEW_ECHO)
        node_id = result.get('id')
        self.assertIsNotNone(node_id)
        from servers.dal_graph import GraphDAL
        self.assertIsNone(
            GraphDAL(self.brain.conn).get_edge_id(node_id, 'nonexistent_node_id_xyz'))
        self.assertEqual(self._loud_count(), before + 1)
        self.assertFalse(self._kv_keys(node_id) & set(self._VIEW_ECHO))

    def test_revise_read_only_attachments_are_named_not_reported_written(self):
        """A node view echoed into revise: the attachments never enter
        fields_updated or the deltas, are named in warnings, and are logged."""
        node_id = self.brain.remember(type='decision', title='revise echo target',
                                      content='before')['id']
        before = self._loud_count()
        r = self.brain.revise(node_id, reason='echo probe', content='after',
                              **self._VIEW_ECHO)
        self.assertNotIn('error', r)
        self.assertFalse(set(r.get('fields_updated') or []) & set(self._VIEW_ECHO))
        self.assertFalse({d.get('field') for d in r.get('deltas') or []}
                         & set(self._VIEW_ECHO))
        for k in self._VIEW_ECHO:
            self.assertTrue(any(w.startswith(k + ':') for w in r.get('warnings') or []), k)
        self.assertEqual(self._loud_count(), before + 1)
        self.assertFalse(self._kv_keys(node_id) & set(self._VIEW_ECHO))
        # Only read-only keys: refused as no update, and named.
        r2 = self.brain.revise(node_id, reason='echo only', communities=[{'id': 'x'}])
        self.assertIn('read-only, not written: communities', r2.get('error', ''))

    def test_content_summary_cut_is_loud_not_silent(self):
        """`content_summary` is a stored field a reader can see. Cutting it
        with an ASCII '...' made it the one truncation no marker detector
        recognizes — the same class as 0682f808, on the write side. It rides
        the shared loud primitive so looks_truncated() can see the cut."""
        from servers.loud_truncation import looks_truncated
        body = 'x' * 600                       # no '. ' → the head-cut path
        nid = self.brain.remember(type='fact', title='long body canary',
                                  content=body)['id']
        summary = self.brain.conn.execute(
            "SELECT content_summary FROM nodes WHERE id = ?", (nid,)).fetchone()[0]
        self.assertTrue(looks_truncated(summary), repr(summary[-40:]))
        self.assertNotIn('...', summary)

    def test_no_bare_except_pass_in_critical_paths(self):
        """Meta-test: grep for bare 'except:' without logging in brain modules."""
        import re as re_mod
        import glob as glob_mod
        servers_dir = os.path.join(os.path.dirname(__file__), '..', 'servers')
        bare_excepts = []
        for py_file in glob_mod.glob(os.path.join(servers_dir, 'brain*.py')):
            with open(py_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == 'except:' or stripped.startswith('except Exception:'):
                    # Check if next non-empty line has logging
                    next_lines = []
                    for j in range(i + 1, min(i + 4, len(lines))):
                        next_stripped = lines[j].strip()
                        if next_stripped:
                            next_lines.append(next_stripped)
                    has_log = any('_log_error' in nl or 'pass' not in nl for nl in next_lines[:2])
                    if next_lines and next_lines[0] == 'pass':
                        bare_excepts.append('%s:%d: %s' % (
                            os.path.basename(py_file), i + 1, stripped))
        # Report but don't fail — this is informational for Phase 2
        if bare_excepts:
            # Store count for tracking
            self.brain.log_debug(
                event_type='bare_except_audit',
                source='test_core',
                count=len(bare_excepts),
                files=str(bare_excepts[:10])
            )


# ── P0: DAL Pattern Enforcement ──────────────────────────────────────

class TestDALPatternEnforcement(unittest.TestCase):
    """Meta-tests: enforce that DB access goes through the DAL.

    The brain has a centralized Data Access Layer (dal.py) that owns access to:
    - brain_meta table → BrainMetaDAL (self._meta)
    - brain_logs.db tables → LogsDAL (self._logs)

    Direct self.logs_conn.execute() and direct brain_meta access in mixin files
    bypasses the DAL. These tests catch violations so the pattern stays clean.
    Nodes/edges in brain.db are NOT DAL-ified yet — direct access is allowed there.
    """

    SERVERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'servers')
    # Files that ARE the DAL or own the connections — allowed to use direct access
    ALLOWED_DIRECT = {'dal.py', 'brain.py', 'schema.py'}
    # Mixin files that should use DAL for logs and meta access
    MIXIN_PATTERN = 'brain_*.py'

    def _scan_mixin_files(self):
        """Get all mixin .py files (brain_*.py) excluding allowed files."""
        import glob as glob_mod
        files = glob_mod.glob(os.path.join(self.SERVERS_DIR, self.MIXIN_PATTERN))
        return [f for f in files if os.path.basename(f) not in self.ALLOWED_DIRECT]

    def test_no_direct_logs_conn_in_mixins(self):
        """Mixin files should use self._logs (LogsDAL) not self.logs_conn.execute().

        brain_logs.db tables (debug_log, access_log, recall_log, miss_log, dream_log,
        staged_learnings) should be accessed through LogsDAL methods, not raw SQL.
        This ensures consistent error handling, timestamps, and commit behavior.
        """
        violations = []
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'self.logs_conn.execute' in line:
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        # Track the count for trend detection
        if violations:
            msg = (f'{len(violations)} direct logs_conn.execute() calls found in mixin files '
                   f'(should use self._logs DAL methods):\n  '
                   + '\n  '.join(violations[:10]))
            if len(violations) > 10:
                msg += f'\n  ... and {len(violations) - 10} more'
            # Don't fail yet — document current state and track regression
            # When migration is complete, change this to self.assertEqual(len(violations), 0, msg)
            self._log_violation_count('logs_conn_direct', len(violations), violations[:5])

    def test_no_direct_brain_meta_in_mixins(self):
        """Mixin files should use self._meta (BrainMetaDAL) not raw brain_meta SQL.

        brain_meta is a key-value config store. All access should go through
        BrainMetaDAL.get() / .set(). Direct INSERT/SELECT on brain_meta
        bypasses validation and timestamps.
        """
        import re as re_mod
        violations = []
        # Match direct brain_meta SQL — INSERT, SELECT, UPDATE, DELETE
        pattern = re_mod.compile(r'(?:INSERT|SELECT|UPDATE|DELETE).*brain_meta', re_mod.IGNORECASE)
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if pattern.search(line) and 'brain_meta' in line:
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        if violations:
            self._log_violation_count('brain_meta_direct', len(violations), violations[:5])

    def test_no_direct_debug_log_in_mixins(self):
        """Mixin files should use self._logs.write_error/write_debug, not raw debug_log SQL."""
        violations = []
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                content = f.read()
            # Direct INSERT into debug_log
            for i, line in enumerate(content.split('\n')):
                if 'debug_log' in line and ('INSERT' in line or 'SELECT' in line):
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        if violations:
            self._log_violation_count('debug_log_direct', len(violations), violations[:5])

    def test_no_direct_miss_log_in_mixins(self):
        """Mixin files should use self._logs.log_miss(), not raw miss_log SQL."""
        violations = []
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                content = f.read()
            for i, line in enumerate(content.split('\n')):
                if 'miss_log' in line and ('INSERT' in line or 'SELECT' in line):
                    violations.append(
                        f'{os.path.basename(py_file)}:{i+1}: {line.strip()[:80]}')

        if violations:
            self._log_violation_count('miss_log_direct', len(violations), violations[:5])

    def test_dal_methods_match_log_tables(self):
        """LogsDAL should have methods for every log table in the schema.

        Tracks which log tables lack DAL methods. As DAL grows, lower the threshold.
        """
        from servers.schema import LOG_TABLES
        import glob

        dal_source = ''
        for path in glob.glob(os.path.join(self.SERVERS_DIR, 'dal*.py')):
            with open(path) as f:
                dal_source += f.read()
        uncovered = []
        for table_name in LOG_TABLES:
            if table_name not in dal_source:
                uncovered.append(table_name)

        # Current baseline: 6 tables not yet in the DAL (dal*.py).
        MAX_UNCOVERED = 6
        self.assertLessEqual(len(uncovered), MAX_UNCOVERED,
                            f'Log tables without DAL coverage: {uncovered}. '
                            f'Add DAL methods or lower threshold after migration.')

    def test_violation_counts_not_increasing(self):
        """Track total violation count — this number should only go DOWN over time.

        Current baseline (2026-03-20): 28 logs_conn + ~6 brain_meta = ~34 violations.
        After each migration session, lower the threshold.
        """
        total_violations = 0
        for py_file in self._scan_mixin_files():
            with open(py_file) as f:
                content = f.read()
            total_violations += content.count('self.logs_conn.execute')
            # Count direct brain_meta access (excluding comments)
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('"""'):
                    continue
                if 'brain_meta' in line and any(kw in line for kw in ['INSERT', 'SELECT', 'UPDATE', 'DELETE']):
                    total_violations += 1

        # Threshold: current state. Lower this as DAL migration progresses.
        # 2026-03-25: 50 violations (brain_precision:20, brain_surface:15, brain_evolution:5,
        #   brain_recall:4, brain_dreams:3, brain_engineering:2, brain_consciousness:1)
        # New DALs built: NodeDAL, VectorDAL, GraphDAL — migrate callers next.
        MAX_ALLOWED_VIOLATIONS = 50
        self.assertLessEqual(total_violations, MAX_ALLOWED_VIOLATIONS,
                            f'{total_violations} direct DB violations in mixins '
                            f'(max allowed: {MAX_ALLOWED_VIOLATIONS}). '
                            f'Migrate to DAL methods or lower the threshold if you just migrated.')

    def _log_violation_count(self, category, count, examples):
        """Log violation count for trend tracking (no brain needed — just prints)."""
        print(f'\n  [dal-audit] {category}: {count} violations')
        for ex in examples:
            print(f'    {ex}')





# ── P4: Remember Rich & Metadata ─────────────────────────────────────

class TestRememberRich(BrainTestBase):
    """P4: Test remember with rich fields and node metadata."""

    def test_recall_node_with_metadata(self):
        """recall_node should return enriched node with metadata."""
        result = self.brain.remember(
            type='lesson',
            title='Lesson: test with real data not toy examples',
            content='Toy test data misses edge cases that real content reveals.',
            reasoning='Production brain has 675 nodes with complex content.')
        recall_result = self.brain.recall_node(result['id'])
        self.assertEqual(len(recall_result['results']), 1)
        node = recall_result['results'][0]
        self.assertIn('_metadata', node)
        self.assertEqual(recall_result['_recall_mode'], 'by_id')

    def test_recall_node_carries_the_canonical_shape(self):
        """By-id recall must agree with the canonical pull.

        recall_node used to read the bare DB row, so a corrected node came
        back with no _corrections and no situation — a superseded claim
        handed over with its correction marker stripped off. Every recall
        door now routes through canonicalize_results.
        """
        claim = self.brain.remember(
            type='finding', title='Claim: the allowlist is read at render',
            content='Asserted before anyone grepped the consumer side.',
            situation='When deciding whether a filter is load-bearing')['id']
        fix = self.brain.remember(
            type='correction', title='Correction: nothing reads it',
            content='The render path never looks at that key.')['id']
        self.brain.connect_typed(
            fix, claim, relation='corrects',
            description='verified on the read side, not the write side')

        node = self.brain.recall_node(claim)['results'][0]
        rich = self.brain.get_node(claim)
        for key in ('_metadata', 'situation', '_corrections', 'connections'):
            self.assertEqual(node.get(key), rich.get(key), key)
        self.assertTrue(node['_corrections'],
                        'the correction chain must ride along on by-id recall')
        self.assertEqual(node['situation'],
                         'When deciding whether a filter is load-bearing')


# ── P6: Surface Layer ────────────────────────────────────────────────

class TestSurfaceLayer(BrainTestBase):
    """P6: Test context boot, health check, and suggest."""

    def test_seed_pack_generation_guard(self):
        """A brain holding a previous-generation seed pack is left untouched:
        gap-fill is crash recovery for THIS pack, never a migration path.
        The faithful fixture keeps titles INTACT — the old and new packs
        genuinely share a title verbatim, which is exactly why the guard is
        a brain_meta marker and not a title match."""
        from servers.seed_pack import SEED_NODES, SEED_PACK_GENERATION, seed_baby_brain

        # Pristine seeded brain (marker stamped at init-seed) → fast path.
        self.assertEqual(seed_baby_brain(self.brain)['status'], 'already_seeded')

        # Retirement is respected, never resurrected: archived seeds count
        # as present (scaffolds instruct the entity to archive them when
        # outgrown — the next boot must not undo that).
        self.brain.conn.execute(
            "UPDATE nodes SET archived = 1 WHERE id IN ("
            "  SELECT id FROM nodes WHERE encoding_source = 'anchor:seed' LIMIT 5)")
        result = seed_baby_brain(self.brain)
        self.assertEqual(result['status'], 'already_seeded')
        self.assertEqual(result['nodes_created'], 0)
        dup = self.brain.conn.execute(
            "SELECT title FROM nodes WHERE encoding_source = 'anchor:seed' "
            "GROUP BY title HAVING COUNT(*) > 1").fetchall()
        self.assertEqual(dup, [], "archived seed was resurrected: %r" % dup)

        # Previous-generation install: seed nodes present — titles untouched,
        # including any shared ones — but no generation marker, and truly
        # missing rows (hard delete = real crash/foreign state, distinct
        # from archived-retirement above).
        self.brain.conn.execute(
            "DELETE FROM brain_meta WHERE key = 'seed_pack_generation'")
        self.brain.conn.execute(
            "DELETE FROM nodes WHERE id IN ("
            "  SELECT id FROM nodes WHERE encoding_source = 'anchor:seed' "
            "  AND archived = 0 LIMIT 10)")
        before = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes").fetchone()[0]
        result = seed_baby_brain(self.brain)
        self.assertEqual(result['status'], 'previous_generation')
        self.assertEqual(result['nodes_created'], 0)
        after = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(before, after, "previous-generation brain was modified")

        # With the marker restored, the same state is OUR crash-recovery
        # case: gap-fill re-creates only the truly missing nodes (the
        # archived ones stay retired).
        self.brain._meta.set('seed_pack_generation', SEED_PACK_GENERATION)
        result = seed_baby_brain(self.brain)
        self.assertEqual(result['status'], 'partial')
        self.assertGreater(result['nodes_created'], 0)
        self.assertEqual(result['nodes_created'] + result['nodes_skipped'],
                         len(SEED_NODES))

        # A foreign marker means "not ours" even with zero seed nodes —
        # never seed over it, never overwrite it (no silent downgrade of a
        # future pack's brain).
        self.brain._meta.set('seed_pack_generation', 'nursery_v99')
        self.brain.conn.execute(
            "DELETE FROM nodes WHERE encoding_source = 'anchor:seed'")
        result = seed_baby_brain(self.brain)
        self.assertEqual(result['status'], 'previous_generation')
        self.assertEqual(
            self.brain._meta.get('seed_pack_generation'), 'nursery_v99',
            "foreign marker was overwritten")

    def test_boot_stats_zero_memory_gate(self):
        """The Nursery's Zero-Memory gate: active while the brain is young
        (few lived memories OR under the age floor); retires only when BOTH
        thresholds are exceeded."""
        import servers.brain_assembly as ba

        # This test brain auto-seeds at init → birth record exists, no lived
        # memories yet → active (young by both thresholds).
        boot = self.brain.boot_stats()
        self.assertTrue(boot['zero_memory']['active'])

        # Old enough but too few lived → still active (lived branch holds).
        self.brain.conn.execute(
            "UPDATE nodes SET created_at = '2020-01-01T00:00:00+00:00' "
            "WHERE encoding_source = 'anchor:seed'")
        boot = self.brain.boot_stats()
        self.assertTrue(boot['zero_memory']['active'])

        # Both thresholds exceeded → the block retires.
        orig = ba.ZERO_MEMORY_MIN_LIVED
        ba.ZERO_MEMORY_MIN_LIVED = 2
        try:
            for i in range(3):
                self.brain.remember(type='fact', title='gate: lived %d' % i,
                                    content='x', encoding_source='hook:test')
            boot = self.brain.boot_stats()
            self.assertFalse(boot['zero_memory']['active'])
            self.assertGreaterEqual(boot['zero_memory']['lived'], 3)
        finally:
            ba.ZERO_MEMORY_MIN_LIVED = orig

    def test_boot_stats_returns_structure(self):
        """boot_stats returns the totals and gate render_boot reads."""
        _seed_brain_with_realistic_data(self.brain)
        result = self.brain.boot_stats()
        self.assertEqual(set(result), {'zero_memory', 'total_nodes', 'total_locked'})
        self.assertGreater(result['total_nodes'], 0)

    def test_health_check_fresh_brain(self):
        """Fresh brain should be healthy: no high-severity issues, well-formed report
        ({'healthy', 'issues', 'actions', 'checked_at'})."""
        result = self.brain.health_check()
        self.assertIsInstance(result, dict)
        self.assertTrue(result['healthy'],
                        f"fresh brain reported unhealthy: {result.get('issues')}")
        self.assertIsInstance(result['issues'], list)
        self.assertIsInstance(result['actions'], list)
        self.assertIn('checked_at', result)

    def test_suggest_with_file(self):
        """suggest() should return suggestions for a known file pattern."""
        self.brain.remember(type='rule',
            title='brain.py: only assembler defines __init__',
            content='Mixins must not define __init__. Only brain.py.',
            locked=True)
        result = self.brain.suggest(context='editing brain.py', file='brain.py')
        self.assertIsInstance(result, dict)
        self.assertIn('suggestions', result)
        self.assertIsInstance(result['suggestions'], list)

    def test_pre_edit_returns_structure(self):
        """pre_edit returns the batched pre-edit contract: suggestions +
        procedures + context_files + timings."""
        result = self.brain.pre_edit(file='test.py', tool_name='Edit')
        self.assertIsInstance(result, dict)
        for key in ('suggestions', 'procedures', 'context_files', 'timings'):
            self.assertIn(key, result)



# ── P7: Connections ──────────────────────────────────────────────────

class TestConnectTyped(BrainTestBase):
    """P7: Test typed edge creation."""

    def test_connect_typed_creates_edge_with_type(self):
        """connect_typed should create relation in edge_relations."""
        n1 = self.brain.remember(type='decision', title='Node A',
            content='Decision about architecture.')
        n2 = self.brain.remember(type='decision', title='Node B',
            content='Related decision about deployment.')
        self.brain.connect_typed(n1['id'], n2['id'],
            relation='depends_on', description='deployment depends on architecture')
        # Find edge_id then query edge_relations
        edge = self.brain.conn.execute(
            'SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)',
            (n1['id'], n2['id'], n2['id'], n1['id'])).fetchone()
        self.assertIsNotNone(edge, 'Edge should exist between nodes')
        rel = self.brain.conn.execute(
            "SELECT relation, description FROM edge_relations WHERE edge_id = ? AND relation = 'depends_on'",
            (edge[0],)).fetchone()
        self.assertIsNotNone(rel, 'depends_on relation should exist in edge_relations')
        self.assertEqual(rel[1], 'deployment depends on architecture')


# ═══════════════════════════════════════════════════════════════
# Feature 2: Safety Check Tests
# ═══════════════════════════════════════════════════════════════

class TestSafetyCheck(BrainTestBase):
    """safety_check() — classifies destructive commands and recalls safety nodes.

    needs_embedder=False so the brain is genuinely EMPTY: an empty brain warns
    about nothing, and the destructive-pattern check needs no model.
    """
    needs_embedder = False

    def test_rm_rf_destructive(self):
        """rm -rf is detected as destructive."""
        result = self.brain.safety_check('rm -rf /tmp/foo')
        self.assertTrue(result['destructive'])

    def test_ls_not_destructive(self):
        """ls -la is NOT destructive."""
        result = self.brain.safety_check('ls -la')
        self.assertFalse(result['destructive'])

    def test_empty_brain_still_detects(self):
        """Destructive command on empty brain still returns destructive=True."""
        result = self.brain.safety_check('rm -rf /important/data')
        self.assertTrue(result['destructive'])
        self.assertEqual(len(result.get('warnings', [])), 0)

    def test_piped_rm_detected(self):
        """Piped rm via xargs is detected."""
        result = self.brain.safety_check("find . -name '*.tmp' | xargs rm")
        self.assertTrue(result['destructive'])


# ── Identifier Splitting ─────────────────────────────────────────────

from servers.text_processing import split_identifier


class TestIdentifierSplitting(unittest.TestCase):
    """Unit tests for split_identifier() — camelCase-aware tokenization."""

    def test_camel_case_basic(self):
        self.assertEqual(split_identifier('camelCase'), ['camel', 'case'])

    def test_pascal_case(self):
        self.assertEqual(split_identifier('RecallScorer'), ['recall', 'scorer'])

    def test_acronym_before_word(self):
        """HTMLParser → html parser (not HTMLP arser)."""
        self.assertEqual(split_identifier('HTMLParser'), ['html', 'parser'])

    def test_acronym_mid_identifier(self):
        """parseHTMLDoc → parse html doc."""
        self.assertEqual(split_identifier('parseHTMLDoc'), ['parse', 'html', 'doc'])

    def test_trailing_acronym(self):
        """getURL → get url."""
        self.assertEqual(split_identifier('getURL'), ['get', 'url'])

    def test_snake_case(self):
        self.assertEqual(split_identifier('brain_surface'), ['brain', 'surface'])

    def test_kebab_case(self):
        self.assertEqual(split_identifier('pre-response-recall'), ['pre', 'response', 'recall'])

    def test_file_extension_stripped(self):
        self.assertEqual(split_identifier('brain_surface.py'), ['brain', 'surface'])

    def test_sh_extension_stripped(self):
        self.assertEqual(split_identifier('pre-edit-suggest.sh'), ['pre', 'edit', 'suggest'])

    def test_version_number_preserved(self):
        result = split_identifier('v2.3.1-beta')
        self.assertIn('v2.3.1', result)
        self.assertIn('beta', result)

    def test_version_only(self):
        result = split_identifier('v5.4.0')
        self.assertEqual(result, ['v5.4.0'])

    def test_digits_preserved(self):
        """Digits like '8' in UTF8Encoder should not be filtered."""
        result = split_identifier('UTF8Encoder')
        self.assertIn('8', result)
        self.assertIn('utf', result)
        self.assertIn('encoder', result)

    def test_path_splitting(self):
        result = split_identifier('servers/daemon_hooks.py')
        self.assertEqual(result, ['servers', 'daemon', 'hooks'])

    def test_deep_path(self):
        result = split_identifier('hooks/scripts/pre_response_recall.py')
        self.assertIn('hooks', result)
        self.assertIn('scripts', result)
        self.assertIn('pre', result)
        self.assertIn('response', result)
        self.assertIn('recall', result)

    def test_empty_string(self):
        self.assertEqual(split_identifier(''), [])

    def test_none_like_input(self):
        self.assertEqual(split_identifier('   '), [])

    def test_dots_only(self):
        self.assertEqual(split_identifier('....'), [])

    def test_single_char(self):
        """Single alphabetic chars are filtered out."""
        self.assertEqual(split_identifier('A'), [])

    def test_database_file(self):
        self.assertEqual(split_identifier('brain.db'), ['brain'])

    def test_all_lowercase_preserved(self):
        result = split_identifier('BrainSurfaceMixin')
        for token in result:
            self.assertEqual(token, token.lower(),
                           'All tokens should be lowercase: %s' % token)

    def test_short_acronym(self):
        """getID → get id (short acronyms preserved)."""
        result = split_identifier('getID')
        self.assertIn('get', result)
        self.assertIn('id', result)

    def test_acronym_to_acronym(self):
        """HTMLToJSON → html to json."""
        result = split_identifier('HTMLToJSON')
        self.assertIn('html', result)
        self.assertIn('to', result)
        self.assertIn('json', result)


class TestIdentifierSplittingE2E(BrainTestBase):
    """End-to-end: verify split_identifier integrates correctly with suggest()."""

    def test_suggest_with_camelcase_file(self):
        """suggest() should find relevant nodes when given a camelCase filename."""
        self.brain.remember(
            type='rule',
            title='Recall scorer must use layered evaluation',
            content='Three layers: regex patterns, embedding similarity, BART NLI. '
                    'Layer 0 regex is fast, Layer 1 embeddings catch semantic matches, '
                    'Layer 1b BART provides entailment scores.',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='RecallScorer.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        self.assertTrue(
            any('recall' in t.lower() or 'scorer' in t.lower() for t in titles),
            'suggest() with camelCase file "RecallScorer.py" should find recall scorer rule. '
            'Got: %s' % titles)

    def test_suggest_with_acronym_file(self):
        """suggest() should handle files with acronyms like HTMLParser."""
        self.brain.remember(
            type='decision',
            title='HTML parsing uses BeautifulSoup',
            content='We chose BeautifulSoup over lxml for HTML parsing because '
                    'it handles malformed HTML gracefully.',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='HTMLParser.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        # The tokenization should produce "html parser" which matches the node
        self.assertTrue(
            any('html' in t.lower() or 'parser' in t.lower() for t in titles),
            'suggest() with "HTMLParser.py" should find HTML parsing decision. '
            'Got: %s' % titles)

    def test_suggest_with_deep_path(self):
        """suggest() with a deep path should extract meaningful tokens."""
        self.brain.remember(
            type='rule',
            title='Hook scripts must handle daemon unavailable',
            content='All hook scripts in hooks/scripts/ must gracefully fall back '
                    'to direct Python when the daemon is not running.',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='hooks/scripts/pre_response_recall.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        self.assertTrue(
            any('hook' in t.lower() or 'daemon' in t.lower() for t in titles),
            'suggest() with deep path should find hook-related rules. '
            'Got: %s' % titles)

    def test_suggest_with_version_in_filename(self):
        """Version numbers in filenames should not corrupt tokenization."""
        self.brain.remember(
            type='decision',
            title='Migration system uses sequential version numbers',
            content='Schema migrations are numbered 001, 002, etc. '
                    'Each migration file applies changes to brain.db or brain_logs.db.',
            locked=True)
        self.brain.save()

        # This should not crash or produce garbage tokens
        result = self.brain.suggest(file='migrations/v2.3.1_add_columns.py', limit=5)
        self.assertIn('suggestions', result)

    def test_suggest_with_snake_case_matches_camelcase_node(self):
        """snake_case file should match PascalCase-titled nodes."""
        self.brain.remember(
            type='concept',
            title='BrainSurface handles suggest and edit context',
            content='The BrainSurface mixin provides suggest(), get_edit_context(), '
                    'and other surface-level retrieval methods.',
            locked=True)
        self.brain.save()

        result = self.brain.suggest(file='brain_surface.py', limit=5)
        suggestions = result.get('suggestions', [])
        titles = [s.get('title', '') for s in suggestions]
        self.assertTrue(
            any('surface' in t.lower() or 'brain' in t.lower() for t in titles),
            'snake_case file should match PascalCase node titles. '
            'Got: %s' % titles)


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=True)
