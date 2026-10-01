"""Tests for the `absorbed_into` survivor-redirect edge (Phase 1 of
TRACE-NODE-RESOLUTION).

On a merge-archive, archive_node writes a first-class `absorbed_into` edge
(source = absorbed/dead, target = survivor/live). The relation is multi-homed:
`correction_improvement` (correction_enrich walks it) + `survivor_lineage` (the
archival-exempt redirect role). The edge must:
  - be written only on a MERGE (survivor passed via extra), not a plain archive
  - land and stay live (archived=0) despite archive_node soft-archiving the
    node's other edges (the survivor_lineage exemption)
  - survive a chained absorb (A→B→C) and survive the Healer's
    archive_dangling_edges sweep — both exempt survivor_lineage relations
  - coexist with the `_sys_archived_survivor_id` audit breadcrumb (backfill
    source + resolve_live's current read path)

The exempt list is sourced from the aspect taxonomy at the call site
(brain.aspects.relations_in(['survivor_lineage'])), not hardcoded in the DAL.
BrainTestBase isolates the aspect registry to a per-test working copy seeded
from the repo seed, so survivor_lineage (and the exemption) is present.

The edge's source endpoint is archived, so get_connections_bulk (which filters
archived endpoints) won't surface it — we read the raw edge tables, which is
also how a future edge-based resolve_live must read it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.aspect_store import member_name as _member_name


def _rel_names(members):
    """Seed members may be {verb, orientation} objects since the seed was oriented."""
    return [_member_name(m, 'edge_relations') for m in members]


class TestAbsorbedIntoEdge(BrainTestBase):
    needs_embedder = False

    def _node(self, title, content='content', **kw):
        r = self.brain.remember(type='fact', title=title, content=content,
                                encoding_source='anchor', **kw)
        return r['id']

    def _absorbed_into(self, source):
        """Raw rows for absorbed_into edges out of `source` (archived endpoint
        is invisible to get_connections_bulk, so query the tables directly)."""
        return self.brain.conn.execute(
            "SELECT e.source_id, e.target_id, er.relation, er.archived, "
            "       er.encoding_source "
            "FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE er.relation = 'absorbed_into' AND e.source_id = ?",
            (source,)).fetchall()

    # ── aspect membership (the SHIPPED SEED) ──

    def test_absorbed_into_multi_homed_in_seed(self):
        """Phase 1 ships absorbed_into multi-homed in the SEED: correction_
        improvement (correction walk) AND survivor_lineage (archival-exempt
        redirect role). Updating an EXISTING brain's live working copy is a
        separate supervised production step; assert the shipped seed here."""
        import json
        from servers.scales.s2.aspect_contract import SEED_ASPECTS_JSON_PATH
        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
        self.assertIn('absorbed_into',
                      _rel_names(seed['correction_improvement']['edge_relations']))
        self.assertIn('survivor_lineage', seed)
        self.assertEqual(_rel_names(seed['survivor_lineage']['edge_relations']),
                         ['absorbed_into'])

    def test_survivor_lineage_is_required_aspect(self):
        from servers.aspects import REQUIRED_ASPECTS
        self.assertIn('survivor_lineage', REQUIRED_ASPECTS)
        # resolvable on the seed-backed registry (setUp pointed it at the seed)
        self.assertEqual(
            tuple(self.brain.aspects.relations_in(['survivor_lineage'])),
            ('absorbed_into',))

    # ── writer: merge writes the edge ──

    def test_absorb_writes_live_absorbed_into_edge(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        r = self.brain.absorb(survivor, absorbed)
        self.assertTrue(r['ok'], r)

        edges = self._absorbed_into(absorbed)
        self.assertEqual(len(edges), 1, edges)
        src, tgt, rel, archived, enc = edges[0]
        self.assertEqual(src, absorbed)      # source = absorbed/dead
        self.assertEqual(tgt, survivor)      # target = survivor/live
        self.assertEqual(archived, 0)        # LIVE despite node being archived
        self.assertTrue(enc)                 # attributable

    def test_sys_breadcrumb_still_written_alongside_edge(self):
        """_sys_archived_survivor_id is kept as the audit/backfill source even
        though the edge is now the first-class link."""
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        self.brain.absorb(survivor, absorbed)
        prov = self.brain.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE node_id = ? "
            "AND key = '_sys_archived_survivor_id'", (absorbed,)).fetchone()
        self.assertIsNotNone(prov)
        self.assertEqual(prov[0], survivor)
        self.assertEqual(len(self._absorbed_into(absorbed)), 1)

    # ── writer: plain archive does NOT write the edge ──

    def test_plain_archive_writes_no_edge(self):
        n = self._node('plain')
        r = self.brain.archive_node(n, archived_by='anchor', reason='just archive')
        self.assertTrue(r['ok'], r)
        self.assertEqual(self._absorbed_into(n), [])

    # ── chain: edge survives a later absorb of the survivor ──

    def test_chain_absorbed_into_edges_survive(self):
        """A→B→C: absorbing B into C must NOT re-archive the A→B absorbed_into
        edge (the step-3 exemption). Both hops stay live and traversable."""
        a = self._node('a')
        b = self._node('b')
        c = self._node('c')
        self.assertTrue(self.brain.absorb(b, a)['ok'])   # a absorbed into b
        self.assertTrue(self.brain.absorb(c, b)['ok'])   # b absorbed into c

        a_edges = self._absorbed_into(a)
        b_edges = self._absorbed_into(b)
        self.assertEqual(len(a_edges), 1, a_edges)
        self.assertEqual(a_edges[0][1], b)               # a -> b
        self.assertEqual(a_edges[0][3], 0)               # STILL live after b archived
        self.assertEqual(len(b_edges), 1, b_edges)
        self.assertEqual(b_edges[0][1], c)               # b -> c
        self.assertEqual(b_edges[0][3], 0)               # live

    def test_chain_resolves_end_to_end_via_resolve_live(self):
        """The metadata read-path (resolve_live, unchanged) still resolves the
        chain end-to-end — edge writing doesn't regress it."""
        a = self._node('a')
        b = self._node('b')
        c = self._node('c')
        self.brain.absorb(b, a)
        self.brain.absorb(c, b)
        out = self.brain._nodes.resolve_live([a])
        self.assertEqual(out['live'], [c])
        self.assertEqual(out['redirected'], {a: c})
        self.assertEqual(out['orphans'], [])

    # ── plain archive of other edges still archives them ──

    def test_other_edges_still_archived_on_merge(self):
        """The exemption is surgical: only absorbed_into survives the archive.
        An external edge on the absorbed node is still soft-archived."""
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        neighbor = self._node('neighbor')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on')
        self.brain.absorb(survivor, absorbed)
        # the depends_on edge_relation touching absorbed is archived
        row = self.brain.conn.execute(
            "SELECT er.archived FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "WHERE er.relation = 'depends_on' AND e.source_id = ?",
            (absorbed,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)

    # ── the Healer (archive_dangling_edges) must NOT scrub absorbed_into ──

    def test_healer_dangling_sweep_spares_absorbed_into(self):
        """archive_dangling_edges archives edges touching archived nodes — an
        absorbed_into edge has an archived source BY CONSTRUCTION. Without the
        survivor_lineage exemption the Healer would scrub it every cycle and
        sever the redirect. Prove both directions: exempt → survives;
        not-exempt → scrubbed (so the exemption is what saves it)."""
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        self.brain.absorb(survivor, absorbed)
        self.assertEqual(self._absorbed_into(absorbed)[0][3], 0)  # live pre-sweep

        # Healer's real call: exempt survivor_lineage → edge survives.
        self.brain._graph.archive_dangling_edges(
            archived_by='s2:healer',
            exempt_relations=self.brain.aspects.relations_in(['survivor_lineage']))
        self.assertEqual(self._absorbed_into(absorbed)[0][3], 0,
                         'Healer scrubbed the redirect despite the exemption')

        # Control: no exemption → the same sweep DOES scrub it.
        self.brain._graph.archive_dangling_edges(archived_by='s2:healer')
        self.assertEqual(self._absorbed_into(absorbed)[0][3], 1,
                         'without exemption the edge should be scrubbed')

    # ── archive_node atomicity (standalone path) ──

    def test_survivor_param_writes_pointer_and_edge(self):
        """The survivor_id PARAMETER is the one lineage door: it must write
        both the `_sys_archived_survivor_id` pointer (resolve_live's read
        path) and the absorbed_into edge."""
        survivor = self._node('survivor')
        victim = self._node('victim')
        r = self.brain.archive_node(victim, archived_by='test',
                                    reason='superseded', survivor_id=survivor)
        self.assertTrue(r['ok'], r)
        kv = self.brain.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE node_id=? "
            "AND key='_sys_archived_survivor_id'", (victim,)).fetchone()
        self.assertEqual(kv[0], survivor)
        self.assertEqual(len(self._absorbed_into(victim)), 1)
        resolved = self.brain._nodes.resolve_live([victim], on_orphan='mark')
        self.assertEqual(resolved['live'], [survivor])
        self.assertEqual(resolved['orphans'], [])

    def test_survivor_lookalike_extra_key_refused_loudly(self):
        """A survivor-looking key in `extra` is the misspelling that silently
        orphaned 9 handoff nodes (`superseded_by`, 2026-07-30). It must be
        DROPPED (extra is pure audit — no second lineage vocabulary) and
        warned about; no pointer, no edge."""
        survivor = self._node('survivor')
        victim = self._node('victim')
        r = self.brain.archive_node(victim, archived_by='test',
                                    reason='superseded',
                                    extra={'superseded_by': survivor,
                                           'note': 'kept-audit-key'})
        self.assertTrue(r['ok'], r)
        rows = dict(self.brain.conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id=? "
            "AND key LIKE '_sys_archived_%'", (victim,)).fetchall())
        self.assertNotIn('_sys_archived_superseded_by', rows)
        self.assertNotIn('_sys_archived_survivor_id', rows)
        self.assertEqual(rows.get('_sys_archived_note'), 'kept-audit-key')
        self.assertEqual(self._absorbed_into(victim), [])
        warned = self.brain._logs_dal.conn.execute(
            "SELECT COUNT(*) FROM debug_log "
            "WHERE source='archive_survivor_key_in_extra'").fetchone()[0]
        self.assertGreaterEqual(warned, 1)

    def test_standalone_archive_commits_atomically(self):
        """A standalone archive_node (not inside a batch) still commits node +
        edges + absorbed_into in one go via the in_batch envelope."""
        survivor = self._node('survivor')
        victim = self._node('victim')
        r = self.brain.archive_node(victim, archived_by='test',
                                    survivor_id=survivor)
        self.assertTrue(r['ok'], r)
        self.assertEqual(self.brain.conn.execute(
            'SELECT archived FROM nodes WHERE id=?', (victim,)).fetchone()[0], 1)
        edges = self._absorbed_into(victim)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][3], 0)  # absorbed_into live

    def test_standalone_archive_rolls_back_on_midstep_failure(self):
        """The envelope makes a standalone archive all-or-nothing: a failure in
        the edge step must roll back the archived=1 flag (no half-archived node
        committed, the regression the inline-commit reintroduced)."""
        victim = self._node('victim')
        orig = self.brain._graph.delete_node_edges

        def boom(*a, **k):
            raise RuntimeError('injected mid-archive failure')
        self.brain._graph.delete_node_edges = boom
        try:
            with self.assertRaises(RuntimeError):
                self.brain.archive_node(victim, archived_by='test')
        finally:
            self.brain._graph.delete_node_edges = orig
        # rolled back: node still live, no partial commit
        self.assertEqual(self.brain.conn.execute(
            'SELECT archived FROM nodes WHERE id=?', (victim,)).fetchone()[0], 0)

    # ── loud exemption helper ──

    def test_archive_exempt_relations_resolves_and_is_quiet(self):
        calls = []
        orig = self.brain._log_error
        self.brain._log_error = lambda *a, **k: calls.append(a)
        try:
            rels = self.brain.archive_exempt_relations()
        finally:
            self.brain._log_error = orig
        self.assertEqual(tuple(rels), ('absorbed_into',))
        self.assertEqual(calls, [], 'must not log when the aspect is present')

    def test_archive_exempt_relations_loud_when_empty(self):
        calls = []
        orig_log = self.brain._log_error
        orig_ri = self.brain.aspects.relations_in
        self.brain._log_error = lambda *a, **k: calls.append(a)
        self.brain.aspects.relations_in = lambda names: ()  # simulate missing aspect
        try:
            rels = self.brain.archive_exempt_relations()
        finally:
            self.brain.aspects.relations_in = orig_ri
            self.brain._log_error = orig_log
        self.assertEqual(tuple(rels), ())
        self.assertTrue(calls, 'empty survivor_lineage must log loudly')

    # ── DAL-level exemption (no aspect dependency) ──

    def test_dal_delete_node_edges_exempts_passed_relations(self):
        survivor = self._node('survivor')
        absorbed = self._node('absorbed')
        neighbor = self._node('neighbor')
        self.brain._graph.add_relation(absorbed, survivor, 'absorbed_into')
        self.brain._graph.add_relation(absorbed, neighbor, 'depends_on')
        # archive absorbed's edges, exempting absorbed_into explicitly
        self.brain._graph.delete_node_edges(
            [absorbed], archived_by='test', exempt_relations=['absorbed_into'])
        ai = self.brain.conn.execute(
            "SELECT er.archived FROM edges e JOIN edge_relations er "
            "ON er.edge_id = e.edge_id WHERE er.relation='absorbed_into' "
            "AND e.source_id=?", (absorbed,)).fetchone()
        dep = self.brain.conn.execute(
            "SELECT er.archived FROM edges e JOIN edge_relations er "
            "ON er.edge_id = e.edge_id WHERE er.relation='depends_on' "
            "AND e.source_id=?", (absorbed,)).fetchone()
        self.assertEqual(ai[0], 0)   # exempt → live
        self.assertEqual(dep[0], 1)  # not exempt → archived


class TestAspectSelfHeal(unittest.TestCase):
    """reconcile_working_copy self-heals a missing REQUIRED aspect into an
    EXISTING working copy — how survivor_lineage reaches an already-running
    brain with no manual migration. Must preserve existing/grown members."""

    def test_missing_required_aspect_healed_and_idempotent(self):
        import json
        import shutil
        import tempfile
        from servers.aspects import reconcile_working_copy
        from servers.aspect_store import SEED_ASPECTS_JSON_PATH

        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
        tmpdir = tempfile.mkdtemp()
        orig = os.environ.get('ASPECTS_JSON_PATH')
        try:
            wc = os.path.join(tmpdir, 'aspects_v1.json')
            # Simulate a pre-migration brain: working copy lacks survivor_lineage
            # but its correction_improvement carries an operator-grown member.
            stale = {k: dict(v) for k, v in seed.items() if k != 'survivor_lineage'}
            stale['correction_improvement']['edge_relations'] = (
                stale['correction_improvement']['edge_relations'] + ['operator_grown'])
            with open(wc, 'w') as f:
                json.dump(stale, f)

            os.environ['ASPECTS_JSON_PATH'] = wc
            self.assertTrue(reconcile_working_copy())   # heal happened
            with open(wc) as f:
                healed = json.load(f)
            self.assertIn('survivor_lineage', healed)     # missing required added
            self.assertEqual(_rel_names(healed['survivor_lineage']['edge_relations']),
                             ['absorbed_into'])
            # existing entry untouched — operator-grown member preserved
            self.assertIn('operator_grown',
                          healed['correction_improvement']['edge_relations'])
            # idempotent: nothing missing now → no-op
            self.assertFalse(reconcile_working_copy())
        finally:
            if orig is None:
                os.environ.pop('ASPECTS_JSON_PATH', None)
            else:
                os.environ['ASPECTS_JSON_PATH'] = orig
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_missing_member_of_existing_aspect_healed(self):
        """A seed MEMBER the working copy lacks is appended.

        Without this, a curated membership fix ships to fresh installs only and
        every existing brain keeps the defect while the seed-based contract
        tests stay green — how `absorbed_into` and then `supersedes` each sat
        outside the correction walk in production while the seed was correct.
        """
        import json
        import shutil
        import tempfile
        from servers.aspects import reconcile_working_copy
        from servers.aspect_store import SEED_ASPECTS_JSON_PATH

        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
        tmpdir = tempfile.mkdtemp()
        orig = os.environ.get('ASPECTS_JSON_PATH')
        try:
            wc = os.path.join(tmpdir, 'aspects_v1.json')
            # Working copy has every aspect, but its correction_improvement is
            # missing the replacement verbs the seed carries — and holds a
            # classifier-grown member of its own that must survive the heal.
            stale = {k: {kk: (list(vv) if isinstance(vv, list) else vv)
                         for kk, vv in v.items()}
                     for k, v in seed.items()}
            ci = stale['correction_improvement']['edge_relations']
            ci[:] = [m for m in ci if _member_name(m, 'edge_relations')
                     not in ('supersedes', 'superseded_by')]
            ci.append('classifier_grown')
            with open(wc, 'w') as f:
                json.dump(stale, f)

            os.environ['ASPECTS_JSON_PATH'] = wc
            logged = []
            self.assertTrue(reconcile_working_copy(log_fn=logged.append))
            with open(wc) as f:
                healed = json.load(f)
            ci_healed = healed['correction_improvement']['edge_relations']
            for verb in ('supersedes', 'superseded_by'):
                self.assertIn(verb, _rel_names(ci_healed))
            # additive: the classifier's own member survives
            self.assertIn('classifier_grown', ci_healed)
            # appended, not inserted — pin the WHOLE prior list as a prefix,
            # not just the first 8. The verbs removed above sit past index 8, so
            # a slice comparison passes even if the heal inserted at index 8.
            self.assertEqual(ci_healed[:len(ci)], ci)
            # the seed's members arrive as it ships them — oriented objects
            self.assertEqual(_rel_names(ci_healed[len(ci):]), ['supersedes', 'superseded_by'])
            # loud: the heal announced itself
            self.assertTrue(logged, 'member heal must call log_fn')
            self.assertIn('supersedes', logged[0])
            # idempotent
            self.assertFalse(reconcile_working_copy())
        finally:
            if orig is None:
                os.environ.pop('ASPECTS_JSON_PATH', None)
            else:
                os.environ['ASPECTS_JSON_PATH'] = orig
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
