"""Per-node `connect_to` in `remember_batch` and `brain_batch`.

Locks the design contract for the sibling-aware connect_to feature:

- Per-node `connect_to` lives on each node spec (remember_batch) or each
  `remember` op (brain_batch).
- Resolution order: same-batch siblings (case-insensitive exact match) first,
  then catalog via find_node_by_title(threshold=0.75).
- NEW wins: when both a sibling AND a catalog node share the title, the
  sibling created in this batch resolves. If you really mean the catalog
  node, use `revise` on its id, not duplicate-title `remember`.
- Sequencing-agnostic: declaration order of remember ops within a batch
  doesn't matter — sibling resolution happens after all nodes exist.
- Failure-resilient AND loud: an unresolved / invalid / self-referential
  connect_to entry is LOUDLY logged to `debug_log` (event_type='error',
  visible on the dashboard) and skipped; it does not fail the batch or
  other entries. Sources used:
    * `connect_to_unresolved` — title matched neither sibling nor catalog
    * `connect_to_bad_id`     — hex-shaped target missed / ambiguous at Pass 0
                                (terminal: never falls through to title match)
    * `connect_to_self`       — would create a self-edge
    * `connect_to_invalid`    — entry shape malformed (no title field, wrong type)
    * `connect_to_failed`     — connect_typed raised (FK violation, etc.)
- Top-level `connect_to` (batch-level argument on remember_batch) keeps its
  existing semantics — applies to all created nodes, excludes siblings.
"""

import unittest
from tests.brain_test_base import BrainTestBase
from servers.dal_graph import GraphDAL
from servers.daemon_dispatch import COMMAND_TABLE, dispatch_command


def _edges_from(brain, source_id):
    """Return list of {target_id, relation, description, weight} edges
    sourced FROM source_id (active relations only)."""
    rows = brain.conn.execute(
        '''SELECT e.edge_id, e.target_id, er.relation, er.description, er.weight
           FROM edges e
           JOIN edge_relations er ON er.edge_id = e.edge_id
           WHERE e.source_id = ? AND er.archived = 0''',
        (source_id,)).fetchall()
    return [{'target_id': r[1], 'relation': r[2],
             'description': r[3] or '', 'weight': r[4]} for r in rows]


def _dispatch(brain, cmd, args):
    """Call a dispatch command through `dispatch_command` — the same execution
    chokepoint MCP, the encoder closure and IsolatedBrain all route through."""
    if cmd not in COMMAND_TABLE:
        raise AssertionError("unknown command: %s" % cmd)
    return dispatch_command(brain, cmd, args, [])


def _recent_errors(brain, source_prefix=''):
    """Return error rows from debug_log — same source the dashboard reads.
    Filter by `source` prefix (e.g. 'connect_to_')."""
    rows = brain.logs_conn.execute(
        "SELECT source, metadata FROM debug_log "
        "WHERE event_type='error' AND source LIKE ? "
        "ORDER BY id DESC", (source_prefix + '%',)
    ).fetchall()
    return [{'source': r[0], 'metadata': r[1]} for r in rows]


# ────────────────────────────────────────────────────────────────────────
# remember_batch — per-node connect_to
# ────────────────────────────────────────────────────────────────────────


class TestRememberBatchIntraBatch(BrainTestBase):
    needs_embedder = True

    def test_basic_sibling_link(self):
        """B's per-node connect_to to A creates a typed edge B→A."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'TCP migration',
             'content': 'Replaced unix sockets with TCP for the daemon.'},
            {'type': 'fact', 'title': 'Stale socket files',
             'content': 'Unix sockets leave files behind after process death.',
             'connect_to': [{'title': 'TCP migration', 'relation': 'grounds',
                             'why': 'this fact is the technical reason TCP was chosen'}]},
        ])
        b_id = result['results'][1]['id']
        edges = [e for e in _edges_from(self.brain, b_id) if e['relation'] == 'grounds']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], result['results'][0]['id'])
        self.assertIn('technical reason', edges[0]['description'])

    def test_ordering_agnostic_forward(self):
        """A defined before B that connects to A — works."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Anchor node A',
             'content': 'First node.'},
            {'type': 'fact', 'title': 'Second node B',
             'content': 'Second.',
             'connect_to': [{'title': 'Anchor node A', 'relation': 'extends', 'why': 'B builds on A'}]},
        ])
        b_id = result['results'][1]['id']
        a_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, b_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], a_id)

    def test_ordering_agnostic_backward(self):
        """B with connect_to 'A' declared BEFORE A — must still link.

        This is the key sequencing-agnostic guarantee: encoder doesn't have
        to think about order. Sibling resolution happens AFTER all nodes
        in the batch are created.
        """
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'First-declared node B',
             'content': 'B comes first in the array.',
             'connect_to': [{'title': 'Late-declared node A', 'relation': 'extends',
                             'why': 'B references A even though A appears later in the batch'}]},
            {'type': 'principle', 'title': 'Late-declared node A',
             'content': 'A comes second.'},
        ])
        b_id = result['results'][0]['id']
        a_id = result['results'][1]['id']
        edges = [e for e in _edges_from(self.brain, b_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1, "backward-declared sibling should resolve")
        self.assertEqual(edges[0]['target_id'], a_id)

    def test_new_wins_over_catalog(self):
        """Catalog has node X; batch creates new X with same title.
        Another sibling's connect_to 'X' resolves to the NEW sibling, not catalog.
        """
        catalog = self.brain.remember(type='principle', title='Shared title',
                                      content='Pre-existing catalog node.')
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Shared title',
             'content': 'New batch sibling with same title.'},
            {'type': 'fact', 'title': 'Linker node',
             'content': 'Links to Shared title.',
             'connect_to': [{'title': 'Shared title', 'relation': 'extends',
                             'why': 'should resolve to sibling, not catalog'}]},
        ])
        sibling_id = result['results'][0]['id']
        linker_id = result['results'][1]['id']
        edges = [e for e in _edges_from(self.brain, linker_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], sibling_id,
                         "NEW wins: should link to sibling, not catalog")
        self.assertNotEqual(edges[0]['target_id'], catalog['id'])

    def test_catalog_fallback_when_no_sibling(self):
        """connect_to title only matches catalog (no sibling) — resolves to catalog."""
        catalog = self.brain.remember(type='principle', title='Catalog-only target',
                                      content='Only in catalog, no sibling with this title.')
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Linker node',
             'content': 'Links to a catalog-only target.',
             'connect_to': [{'title': 'Catalog-only target', 'relation': 'grounds',
                             'why': 'fallback path to catalog'}]},
        ])
        linker_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, linker_id) if e['relation'] == 'grounds']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], catalog['id'])

    def test_unmatched_title_skipped_not_failed(self):
        """connect_to title nobody matches — entry is skipped, batch still succeeds."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Lonely node',
             'content': 'Has a connect_to nobody can match.',
             'connect_to': [{'title': 'zzz_nonexistent_title_no_match_possible',
                             'relation': 'extends', 'why': 'should be dropped'}]},
        ])
        self.assertEqual(result['nodes_created'], 1)
        node_id = result['results'][0]['id']
        edges = _edges_from(self.brain, node_id)
        # No 'extends' edge may exist from the unresolved connect_to
        self.assertEqual([e for e in edges if e['relation'] == 'extends'], [])
        # Loud-by-default: dashboard sees the unresolved entry
        errors = _recent_errors(self.brain, 'connect_to_unresolved')
        self.assertGreaterEqual(len(errors), 1, "unresolved title must log loudly")

    def test_self_reference_skipped(self):
        """Node's connect_to to its own title — skipped (no self-edge)."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Self-targeting node',
             'content': 'Has connect_to to itself.',
             'connect_to': [{'title': 'Self-targeting node', 'relation': 'extends',
                             'why': 'should not create self-edge'}]},
        ])
        node_id = result['results'][0]['id']
        edges = _edges_from(self.brain, node_id)
        # No edge to self
        self.assertEqual([e for e in edges if e['target_id'] == node_id], [])
        # Loud-by-default: dashboard sees the self-reference attempt
        errors = _recent_errors(self.brain, 'connect_to_self')
        self.assertGreaterEqual(len(errors), 1, "self-reference must log loudly")

    def test_invalid_entry_shape_skipped(self):
        """connect_to entry without title — skipped, batch still succeeds, valid entries still process."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Valid sibling', 'content': 'First.'},
            {'type': 'fact', 'title': 'Mixed-validity linker',
             'content': 'Has one valid + one invalid connect_to.',
             'connect_to': [
                 {'why': 'no title field — invalid'},  # malformed
                 {'title': 'Valid sibling', 'relation': 'extends', 'why': 'this one works'},
             ]},
        ])
        self.assertEqual(result['nodes_created'], 2)
        linker_id = result['results'][1]['id']
        valid_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, linker_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], valid_id)
        # Loud-by-default: dashboard sees the malformed entry
        errors = _recent_errors(self.brain, 'connect_to_invalid')
        self.assertGreaterEqual(len(errors), 1, "malformed entry must log loudly")

    def test_non_string_title_returns_failed_not_raises(self):
        """A non-string `title` (e.g. an int) must NOT crash _apply_connect_to.

        Regression: `entry.get('title', '')` cleared the falsy guard for a
        non-empty non-string (int like 12345678, or a list), then the
        downstream .strip()/.lower()/regex calls raised AttributeError. That
        exception escaped _apply_connect_to — violating its "never raises"
        contract — and rolled back the entire brain_batch. The entry must
        instead surface as a {'created': [], 'failed': [...]} result.
        """
        src = self.brain.remember(type='fact', title='Non-string-title source',
                                  content='Source for the non-string title test.')
        # 12345678 is also 8 digits — would match the hex-id regex AS A STRING,
        # so the type guard must fire before the regex/.strip() ever runs.
        result = self.brain._apply_connect_to(src['id'], [
            {'title': 12345678, 'relation': 'extends', 'why': 'int title — invalid'},
        ])
        self.assertEqual(result['created'], [])
        self.assertEqual(len(result['failed']), 1)
        self.assertIn('must be a string', result['failed'][0]['reason'])
        # A list title is also non-string — same path, no crash.
        result2 = self.brain._apply_connect_to(src['id'], [
            {'title': ['not', 'a', 'string'], 'relation': 'extends', 'why': 'list title'},
        ])
        self.assertEqual(result2['created'], [])
        self.assertEqual(len(result2['failed']), 1)
        # Loud-by-default: dashboard sees the malformed entry
        errors = _recent_errors(self.brain, 'connect_to_invalid')
        self.assertGreaterEqual(len(errors), 1, "non-string title must log loudly")

    def test_relations_array_format(self):
        """Per-node connect_to with `relations: [...]` produces multiple typed edges."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'principle', 'title': 'Multi-rel target', 'content': 'Anchor.'},
            {'type': 'fact', 'title': 'Multi-rel source', 'content': 'Has 2 relations to the anchor.',
             'connect_to': [{'title': 'Multi-rel target',
                             'relations': [
                                 {'relation': 'grounds', 'why': 'first reason'},
                                 {'relation': 'extends', 'why': 'second reason'},
                             ]}]},
        ])
        src_id = result['results'][1]['id']
        tgt_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, src_id) if e['target_id'] == tgt_id]
        rels = sorted(e['relation'] for e in edges)
        self.assertEqual(rels, ['extends', 'grounds'])

    def test_top_level_connect_to_unchanged_excludes_siblings(self):
        """Top-level connect_to keeps existing behavior: applies to all created nodes,
        but siblings being created in this batch are excluded — only catalog targets."""
        catalog = self.brain.remember(type='principle', title='External anchor',
                                      content='Catalog target for batch top-level.')
        # Sibling shares a title with the top-level connect_to target
        result = self.brain.remember_batch(
            nodes=[
                {'type': 'principle', 'title': 'External anchor', 'content': 'Sibling with same title.'},
                {'type': 'fact', 'title': 'Other sibling', 'content': 'Plain.'},
            ],
            connect_to=[{'title': 'External anchor', 'relation': 'related',
                         'why': 'top-level anchors all batch members'}],
        )
        # Top-level resolves to catalog, NOT the sibling (back-compat).
        # Both siblings should have an edge to the CATALOG anchor.
        sibling_id = result['results'][0]['id']
        other_id = result['results'][1]['id']
        for nid in (sibling_id, other_id):
            edges = [e for e in _edges_from(self.brain, nid) if e['relation'] == 'related']
            tgts = [e['target_id'] for e in edges]
            self.assertIn(catalog['id'], tgts,
                          "top-level connect_to must resolve to catalog target")
            self.assertNotIn(sibling_id, [t for t in tgts if t != sibling_id],
                             "top-level must NOT link to sibling with same title")


# ────────────────────────────────────────────────────────────────────────
# brain_batch — per-op connect_to inside `remember` ops
# ────────────────────────────────────────────────────────────────────────


class TestBrainBatchIntraBatch(BrainTestBase):
    needs_embedder = True

    def test_basic_sibling_link(self):
        """remember ops in brain_batch with per-op connect_to link siblings."""
        r = _dispatch(self.brain, 'brain_batch', {
            'operations': [
                {'op': 'remember', 'type': 'principle', 'title': 'BB target',
                 'content': 'Anchor.'},
                {'op': 'remember', 'type': 'fact', 'title': 'BB source',
                 'content': 'Links to BB target.',
                 'connect_to': [{'title': 'BB target', 'relation': 'grounds',
                                 'why': 'sibling link via brain_batch'}]},
            ]
        })
        self.assertTrue(r['ok'], r)
        results = r['result']['results']
        tgt_id = results[0]['result']['id']
        src_id = results[1]['result']['id']
        edges = [e for e in _edges_from(self.brain, src_id) if e['relation'] == 'grounds']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], tgt_id)

    def test_ordering_agnostic_backward_in_brain_batch(self):
        """Source remember declared BEFORE target remember — must still link.

        Mirrors the remember_batch backward test for the brain_batch surface.
        """
        r = _dispatch(self.brain, 'brain_batch', {
            'operations': [
                {'op': 'remember', 'type': 'fact', 'title': 'BB early source',
                 'content': 'Declared first.',
                 'connect_to': [{'title': 'BB late target', 'relation': 'extends',
                                 'why': 'forward reference to a sibling not yet declared'}]},
                {'op': 'remember', 'type': 'principle', 'title': 'BB late target',
                 'content': 'Declared after the source.'},
            ]
        })
        self.assertTrue(r['ok'], r)
        results = r['result']['results']
        src_id = results[0]['result']['id']
        tgt_id = results[1]['result']['id']
        edges = [e for e in _edges_from(self.brain, src_id) if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1, "backward-declared sibling should resolve in brain_batch")
        self.assertEqual(edges[0]['target_id'], tgt_id)

    def test_mixed_ops_dont_break_connect_to(self):
        """remember + connect + revise mixed — per-op connect_to still resolves.

        Revise/connect ops interleaved with remembers must not interfere with
        the deferred sibling resolution.
        """
        # Pre-existing node to revise + connect to
        existing = self.brain.remember(type='fact', title='Existing node X',
                                       content='Pre-existing for mixed-op test.')
        r = _dispatch(self.brain, 'brain_batch', {
            'operations': [
                {'op': 'remember', 'type': 'principle', 'title': 'Mixed sibling',
                 'content': 'Created in mixed batch.'},
                {'op': 'revise', 'node_id': existing['id'],
                 'reason': 'mixed-op test', 'content': 'Revised content.'},
                {'op': 'remember', 'type': 'fact', 'title': 'Mixed source',
                 'content': 'Links to a sibling AND to existing.',
                 'connect_to': [
                     {'title': 'Mixed sibling', 'relation': 'extends',
                      'why': 'sibling link mid-mixed-batch'},
                     {'title': 'Existing node X', 'relation': 'grounds',
                      'why': 'catalog fallback in mixed-batch'},
                 ]},
            ]
        })
        self.assertTrue(r['ok'], r)
        results = r['result']['results']
        sibling_id = results[0]['result']['id']
        src_id = results[2]['result']['id']
        edges = _edges_from(self.brain, src_id)
        rel_to_target = {e['target_id']: e['relation'] for e in edges
                         if e['relation'] in ('extends', 'grounds')}
        self.assertEqual(rel_to_target.get(sibling_id), 'extends')
        self.assertEqual(rel_to_target.get(existing['id']), 'grounds')


def _edge_traces(brain):
    """edge_relation_revised trace events (parsed metadata) from the test
    brain's logs db — the directional edge substrate that, with node-lifecycle
    deltas, makes the graph reconstructable from traces alone."""
    import json
    rows = brain.logs_conn.execute(
        "SELECT metadata FROM trace_events "
        "WHERE event_type='delta' AND ref_type='edge_relation_revised'"
    ).fetchall()
    return [json.loads(r[0]) for r in rows if r[0]]


class TestBrainBatchAffectedAndEdgeTraces(BrainTestBase):
    """Phase 0 contract: dispatch write handlers return the authoritative
    node-lifecycle `affected` split, and EVERY edge (including connect_to)
    emits a directional edge_relation_revised trace. Together — nodes from
    `affected`, edges from edge events — the graph is reconstructable from the
    trace substrate alone. `connected` (a directionless two-sided-era vestige)
    is gone."""
    needs_embedder = True

    def test_brain_batch_returns_affected_node_lifecycle(self):
        existing = self.brain.remember(type='fact', title='Affected existing',
                                       content='Gets revised in the batch.')
        doomed = self.brain.remember(type='fact', title='Affected doomed',
                                     content='Gets archived in the batch.')
        r = _dispatch(self.brain, 'brain_batch', {'operations': [
            {'op': 'remember', 'type': 'principle', 'title': 'Affected new',
             'content': 'A freshly created node.'},
            {'op': 'revise', 'node_id': existing['id'], 'reason': 'test',
             'content': 'Revised body.'},
            {'op': 'archive', 'node_id': doomed['id'], 'reason': 'test'},
        ]})
        self.assertTrue(r['ok'], r)
        aff = r['affected']
        new_id = r['result']['results'][0]['result']['id']
        self.assertEqual(aff['created'], [new_id])
        self.assertIn(existing['id'], aff['revised'])
        self.assertIn(doomed['id'], aff['archived'])
        # `connected` is gone — edges are not node lifecycle.
        self.assertNotIn('connected', aff)

    def test_absorb_affected_survivor_revised_absorbed_archived(self):
        # Pins the `affected` attribution — survivor→revised, absorbed→archived
        # — which holds regardless of field overrides. (The content-override-
        # in-batch savepoint bug this test originally side-stepped was fixed in
        # 47dbc41; its own regression test lives in test_absorb.py.)
        surv = self.brain.remember(type='fact', title='Absorb survivor',
                                   content='Stays as the survivor.')
        absorbed = self.brain.remember(type='fact', title='Absorb folded',
                                       content='Folded into the survivor.')
        r = _dispatch(self.brain, 'brain_batch', {'operations': [
            {'op': 'absorb', 'survivor_id': surv['id'],
             'absorbed_id': absorbed['id'], 'reason': 'test merge'},
        ]})
        self.assertTrue(r['ok'], r)
        sub = r['result']['results'][0]
        self.assertTrue(sub.get('ok'), sub)
        aff = r['affected']
        self.assertIn(surv['id'], aff['revised'])
        self.assertIn(absorbed['id'], aff['archived'])

    def test_connect_to_emits_directional_edge_trace(self):
        r = _dispatch(self.brain, 'brain_batch', {'operations': [
            {'op': 'remember', 'type': 'principle', 'title': 'Edge-trace target',
             'content': 'Anchor node.'},
            {'op': 'remember', 'type': 'fact', 'title': 'Edge-trace source',
             'content': 'Links to the anchor.',
             'connect_to': [{'title': 'Edge-trace target', 'relation': 'grounds',
                             'why': 'connect_to must leave a directional edge trace'}]},
        ]})
        self.assertTrue(r['ok'], r)
        results = r['result']['results']
        tgt_id = results[0]['result']['id']
        src_id = results[1]['result']['id']
        match = [m for m in _edge_traces(self.brain)
                 if m.get('source_id') == src_id
                 and m.get('target_id') == tgt_id
                 and m.get('relation') == 'grounds']
        self.assertEqual(len(match), 1,
                         "connect_to edge must emit exactly one directional "
                         "edge_relation_revised trace (was the silent path)")

    def test_co_anchored_edge_emits_no_trace_and_stays_out_of_payload(self):
        """co_anchored is in the NOISE aspect (ruled 2026-08-04, emitter step
        6): the graph edge is still written, but it emits NO trace —
        soft/derived edges stay out of trace coverage — and co_anchored_made
        never reaches the agent-facing payload."""
        ref = 'aaaaaaaa'  # 8-char hex source-ref id
        a = _dispatch(self.brain, 'remember', {
            'type': 'fact', 'title': 'Anchor A', 'content': 'first',
            'source_refs': [ref]})
        b = _dispatch(self.brain, 'remember', {
            'type': 'fact', 'title': 'Anchor B', 'content': 'second',
            'source_refs': [ref]})
        a_id = a['result']['id']
        b_id = b['result']['id']
        # The graph edge exists (co_anchored still fires at dispatch)...
        row = self.brain.conn.execute(
            "SELECT 1 FROM edge_relations er JOIN edges e USING (edge_id) "
            "WHERE er.relation='co_anchored' AND er.archived=0 "
            "AND ((e.source_id=? AND e.target_id=?) "
            "  OR (e.source_id=? AND e.target_id=?))",
            (b_id, a_id, a_id, b_id)).fetchone()
        self.assertTrue(row, "co_anchored graph edge must still be written")
        # ...but no trace row records it.
        match = [m for m in _edge_traces(self.brain)
                 if m.get('relation') == 'co_anchored']
        self.assertEqual(match, [],
                         "co_anchored is noise-aspect: no edge trace")
        # And it must NOT leak co_anchored_made into the agent-facing payload.
        self.assertNotIn('co_anchored_made', b['result'])

    def test_batch_subop_traces_join_chain_and_chain_id_not_leaked(self):
        """A brain_batch sub-op's edge traces join the batch's chain_id (not a
        date-fallback chain), AND chain_id — a control field — must not leak
        onto the created node's metadata. Probes chain routing via a
        connect_to edge (co_anchored is noise-aspect and emits nothing as of
        step 6)."""
        self.brain.remember(type='fact', title='Chain target', content='pre')
        chain = 's1e-testchain-7'
        r = _dispatch(self.brain, 'brain_batch', {
            'chain_id': chain,
            'operations': [
                {'op': 'remember', 'type': 'fact', 'title': 'Chain B',
                 'content': 'b',
                 'connect_to': [{'title': 'Chain target',
                                 'relation': 'grounds',
                                 'why': 'chain routing probe edge'}]},
            ]})
        self.assertTrue(r['ok'], r)
        nid = r['result']['results'][0]['result']['id']
        # (a) chain_id must NOT have leaked into node metadata.
        kv = self.brain._meta_kv.get_all_bulk([nid]).get(nid, {})
        self.assertNotIn('chain_id', kv,
                         "chain_id is a control field — must not land on the node")
        # (b) the connect_to edge trace joined the batch chain (not date-fallback).
        rows = self.brain.logs_conn.execute(
            "SELECT chain_id FROM trace_events "
            "WHERE ref_type='edge_relation_revised' AND metadata LIKE ?",
            ('%' + nid + '%',)).fetchall()
        self.assertTrue(rows, "expected an edge trace for the connect_to edge")
        self.assertTrue(all(row[0] == chain for row in rows),
                        "batch sub-op edge traces must join the batch chain_id, got %s"
                        % [row[0] for row in rows])


class TestHexShapedTargets(BrainTestBase):
    """Pass 0 is TERMINAL for hex-shaped connect_to targets (Option D, s1e
    v34: ids are the expected form for catalog targets, so a hex string that
    misses at Pass 0 is a miscopy, not a title). Miss or ambiguous prefix →
    `connect_to_bad_id`, edge skipped, title matching never entered — a bad
    hex must not silently attach to a node whose TITLE quotes a hex id.
    One exception: a same-batch sibling whose exact title is hex-shaped
    still resolves as a sibling (Pass 1 semantics)."""
    needs_embedder = True

    def test_hex_id_resolves_by_pass0(self):
        """A real node id in the title slot resolves to that node."""
        target = self.brain.remember(type='fact', title='Pass-0 target node',
                                     content='Target for id-based connect_to.')
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Id-linker node',
             'content': 'Links by id.',
             'connect_to': [{'title': target['id'], 'relation': 'grounds',
                             'why': 'id-form catalog target'}]},
        ])
        linker_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, linker_id)
                 if e['relation'] == 'grounds']
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['target_id'], target['id'])

    def test_hex_miss_is_terminal_never_title_matched(self):
        """A hex target matching no id must NOT fall through to title
        matching — even when a catalog node's title IS that exact hex."""
        decoy = self.brain.remember(
            type='fact', title='feedbeef',
            content='Catalog node whose title is pure hex — the trap.')
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'Miscopy linker',
             'content': 'Carries a miscopied id.',
             'connect_to': [{'title': 'feedbeef', 'relation': 'extends',
                             'why': 'hex-shaped, matches no id'}]},
        ])
        self.assertEqual(result['nodes_created'], 1)
        linker_id = result['results'][0]['id']
        edges = [e for e in _edges_from(self.brain, linker_id)
                 if e['relation'] == 'extends']
        self.assertEqual(edges, [],
                         "hex miss must not title-match the decoy node")
        self.assertNotIn(decoy['id'],
                         [e['target_id'] for e in _edges_from(self.brain, linker_id)])
        errors = _recent_errors(self.brain, 'connect_to_bad_id')
        self.assertGreaterEqual(len(errors), 1,
                                "hex miss must log connect_to_bad_id")

    def test_hex_ambiguous_prefix_is_terminal(self):
        """Two ids sharing the query prefix → terminal connect_to_bad_id,
        no title fallback. (Production ids are 8-char so ambiguity needs
        longer injected ids, but the branch must hold.)"""
        for fake_id in ('abcdef120001', 'abcdef120002'):
            self.brain.conn.execute(
                "INSERT INTO nodes (id, type, title, content) "
                "VALUES (?, 'fact', ?, 'ambiguity fixture')",
                (fake_id, 'fixture ' + fake_id))
        self.brain.conn.commit()
        src = self.brain.remember(type='fact', title='Ambiguity source',
                                  content='Source for ambiguous prefix test.')
        result = self.brain._apply_connect_to(src['id'], [
            {'title': 'abcdef12', 'relation': 'extends',
             'why': 'ambiguous prefix'},
        ])
        self.assertEqual(result['created'], [])
        self.assertEqual(len(result['failed']), 1)
        self.assertIn('ambiguous', result['failed'][0]['reason'])
        errors = _recent_errors(self.brain, 'connect_to_bad_id')
        self.assertGreaterEqual(len(errors), 1)

    def test_hex_sibling_title_still_resolves(self):
        """A hex-shaped target that misses at Pass 0 but exactly matches a
        same-batch sibling's title resolves as a sibling — the one
        exception to the terminal decision."""
        result = self.brain.remember_batch(nodes=[
            {'type': 'fact', 'title': 'DEADBEEF01',
             'content': 'Sibling whose title happens to be pure hex.'},
            {'type': 'fact', 'title': 'Hex-sibling linker',
             'content': 'Links to the hex-titled sibling.',
             'connect_to': [{'title': 'deadbeef01', 'relation': 'extends',
                             'why': 'sibling exact title, case-insensitive'}]},
        ])
        sibling_id = result['results'][0]['id']
        linker_id = result['results'][1]['id']
        edges = [e for e in _edges_from(self.brain, linker_id)
                 if e['relation'] == 'extends']
        self.assertEqual(len(edges), 1, "hex sibling title must resolve")
        self.assertEqual(edges[0]['target_id'], sibling_id)


if __name__ == '__main__':
    unittest.main()
