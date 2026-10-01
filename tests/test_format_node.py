"""Tests for render_rich_node() — the standard node renderer for LLM consumers."""

import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.contract import render_rich_node, NODE_FORMAT_DEFAULTS


class TestFormatNode(BrainTestBase):
    needs_embedder = False

    def _make_node(self, **kwargs):
        """Create a node and return its full ID."""
        defaults = dict(type='rule', title='Test node', content='Test content')
        defaults.update(kwargs)
        result = self.brain.remember(**defaults)
        return result['id']

    def _add_edge(self, source_id, target_id, relation='related_to',
                  weight=0.8, description=''):
        """Insert an edge using the new multi-relation model."""
        from servers.dal_graph import GraphDAL
        dal = GraphDAL(self.brain.conn)
        dal.add_relation(source_id, target_id, relation, description, weight)

    def _render(self, node_id, config=None):
        """Fetch node + render: the two-step pattern replacing format_node()."""
        node = self.brain.get_node(node_id)
        if not node:
            return None
        return render_rich_node(node, config)

    def _stamp_relation(self, source_id, target_id, ts, column='created_at'):
        """Backdate (or clear) a relation's timestamp on the (source, target) pair."""
        self.brain.conn.execute(
            "UPDATE edge_relations SET %s = ? WHERE edge_id = "
            "(SELECT edge_id FROM edges WHERE source_id = ? AND target_id = ?)" % column,
            (ts, source_id, target_id))
        self.brain.conn.commit()

    def _edge_line(self, node_id, relation, config=None):
        """The rendered edge line carrying `relation` (absolute dates by default)."""
        out = self._render(node_id, config)
        return next(l for l in out.split('\n') if l.startswith('    [') and relation in l)

    # ── Basic rendering ──

    def test_full_node(self):
        """Full node with core fields renders header + content + keywords."""
        nid = self._make_node(
            type='decision', title='Use Postgres', content='We chose Postgres for reliability.',
            keywords='db postgres sql', confidence=0.9, locked=True)
        out = self._render(nid)
        self.assertIsNotNone(out)
        self.assertIn('[decision]', out)
        self.assertIn('"Use Postgres"', out)
        # Confidence display rooted out 2026-05-31: show_confidence defaults
        # off (the field is dormant — read by no ranking path). Default render
        # must NOT show it. Same strictness, opposite contract.
        self.assertNotIn('conf:', out)
        self.assertIn('locked', out)
        self.assertIn('We chose Postgres', out)
        # Keywords column dropped in schema v28; render block removed.
        # Asserting absence (same strictness, opposite contract).
        self.assertNotIn('Keywords:', out)

    def test_confidence_renders_when_enabled(self):
        """Confidence is hidden by default but still renders when a caller
        explicitly opts in via show_confidence=True (the dormant field's
        opt-in path stays covered after the 2026-05-31 default flip)."""
        nid = self._make_node(
            type='decision', title='Use Postgres', content='We chose Postgres.',
            confidence=0.9)
        node = self.brain.get_node(nid)
        out = render_rich_node(node, {'show_confidence': True})
        self.assertIn('conf:0.9', out)

    def test_nonexistent_node_returns_none(self):
        """Render returns None for an ID that doesn't exist."""
        out = self._render('nonexistent-id-12345678')
        self.assertIsNone(out)

    # ── Header format ──

    def test_header_includes_id_prefix(self):
        """Header shows first 8 chars of node ID."""
        nid = self._make_node(type='lesson', title='Header test')
        out = self._render(nid)
        self.assertIn('id:%s' % nid[:8], out)

    def test_header_unlocked_no_locked_flag(self):
        """Unlocked nodes don't show ', locked' in header."""
        nid = self._make_node(locked=False)
        out = self._render(nid)
        self.assertNotIn('locked', out)

    def test_header_encoding_source(self):
        """encoding_source appears in header when set."""
        nid = self._make_node(encoding_source='encoder:sonnet')
        out = self._render(nid)
        self.assertIn('src:encoder:sonnet', out)

    # ── Content truncation ──

    def test_content_limit_truncates(self):
        """content_limit config truncates long content."""
        long_content = 'A' * 500
        nid = self._make_node(content=long_content)
        out = self._render(nid, config={'content_limit': 50})
        # truncate_marked caps at <= limit chars INCLUDING the ellipsis
        # (s[:limit-1] + '…'), so a 500-char body renders as 49 A's + '…'.
        self.assertIn('A' * 49 + '…', out)
        self.assertNotIn('A' * 50, out)

    def test_no_content_limit_shows_full(self):
        """Default (content_limit=None) shows full content."""
        long_content = 'B' * 500
        nid = self._make_node(content=long_content)
        out = self._render(nid)
        self.assertIn('B' * 500, out)

    # ── Metadata ──

    def test_metadata_situation(self):
        """Situation from node_metadata_kv (canonical store) is rendered.

        v24+: situation lives in node_metadata_kv, not the removed
        node_embeddings table. Stronger than the old test — verifies
        both the canonical storage location AND the rendered output.
        """
        sit = 'When choosing a database for OLTP workloads'
        nid = self._make_node(situation=sit)
        # Verify storage location — situation must be in kv, not elsewhere
        kv_row = self.brain.conn.execute(
            "SELECT value FROM node_metadata_kv WHERE node_id=? AND key='situation'",
            (nid,)).fetchone()
        self.assertIsNotNone(kv_row, 'situation should be stored in node_metadata_kv')
        self.assertEqual(kv_row[0], sit, 'kv value must match the written situation exactly')
        # Verify render
        out = self._render(nid)
        self.assertIn('Situation: When choosing a database', out)
        self.assertIn(sit, out)  # full exact-match, stronger than substring

    def test_metadata_reasoning(self):
        """Reasoning from metadata_kv is rendered."""
        nid = self._make_node(reasoning='Postgres has better JSON support than MySQL')
        out = self._render(nid)
        self.assertIn('Reasoning:', out)
        self.assertIn('Postgres has better JSON support', out)

    def test_open_key_renders_under_the_key_a_write_sends(self):
        """A key the contract does not declare shows as itself, so a writer
        sends back exactly the name it read; a declared field keeps its
        label."""
        nid = self._make_node(reasoning='Postgres has better JSON support',
                              community_learning_arc='Fix drafted (id:0b0b0b0b)')
        out = self._render(nid)
        self.assertIn('  community_learning_arc: Fix drafted (id:0b0b0b0b)', out)
        self.assertNotIn('Community Learning Arc', out)
        self.assertIn('Reasoning: Postgres has better JSON support', out)

    def test_correction_edge_annotations(self):
        """A `corrects` edge warns the corrected node and is history for the
        corrector.

        correction_enrich walks correction_improvement-aspect edges
        (corrects, supersedes, reframes, ...). The corrected node's view
        carries '⚠ Corrected by'; the corrector's view shows the same edge as
        an ordinary Edges line, with no ⚠.
        """
        original_id = self._make_node(title='Use MySQL')
        correction_id = self._make_node(title='Use Postgres instead')
        # Edge: correction_id corrects original_id
        self.brain.connect_typed(
            source_id=correction_id, target_id=original_id,
            relation='corrects', weight=0.5,
            description='Postgres reliability beats MySQL for this workload',
            encoding_source='test:correction_edge_annotations')

        # Corrector's view: the edge, not a warning
        out_corrector = self._render(correction_id)
        self.assertNotIn('⚠', out_corrector)
        self.assertIn('this corrects "Use MySQL"', out_corrector)
        # Corrected node's view: 'Corrected by' (the `corrects` flavor),
        # directly under the node's own content — the first Content: under a
        # header is always the node's
        out_original = self._render(original_id)
        self.assertIn('⚠ Corrected by "Use Postgres instead"', out_original)
        lines = out_original.splitlines()
        warn = next(i for i, l in enumerate(lines) if '⚠ Corrected by' in l)
        body = next(i for i, l in enumerate(lines) if l.strip().startswith('Content:'))
        self.assertEqual(body, 1, lines)             # the node's own content first
        self.assertEqual(warn, body + 1, lines)      # then what corrected it

    def test_corrector_relation_shows_once(self):
        """A ⚠ line rendered with its verb (balanced/heavy) carries all its
        Edges line would, so the relation leaves Edges and frees its slot; a
        lean pointer lacks the verb, so its Edges line stays; the pair's other
        relations stay; `correction_dedup` False keeps the repeat; the node
        dict is not modified (one node renders under several configs)."""
        old = self._make_node(title='Use MySQL')
        new = self._make_node(title='Use Postgres instead')
        other = self._make_node(title='Unrelated neighbour')
        self.brain.connect_typed(source_id=new, target_id=old, relation='supersedes',
                                 weight=0.9, description='Postgres replaced it',
                                 encoding_source='test:corrector_once')
        self.brain.connect_typed(source_id=new, target_id=old, relation='extends',
                                 weight=0.5, description='and builds on it',
                                 encoding_source='test:corrector_once')
        self._add_edge(old, other, relation='grounds', weight=0.4,
                       description='a plain edge')
        node = self.brain.get_node(old)
        before = repr(node['connections'])

        balanced = render_rich_node(node, {'correction_render': 'balanced',
                                           'edge_limit': 1})
        self.assertIn('⚠ Replaced by "Use Postgres instead"', balanced)
        self.assertNotIn('supersedes this', balanced)
        # the freed slot goes to the next edge, not to the repeat
        self.assertIn('"Use Postgres instead" extends this — and builds on it', balanced)
        full = render_rich_node(node, {'correction_render': 'balanced'})
        self.assertIn('this grounds "Unrelated neighbour"', full)

        lean = render_rich_node(node, {'correction_render': 'lean'})
        self.assertIn('"Use Postgres instead" supersedes this', lean)
        kept = render_rich_node(node, {'correction_render': 'balanced',
                                       'correction_dedup': False})
        self.assertIn('"Use Postgres instead" supersedes this', kept)
        self.assertEqual(repr(node['connections']), before)

    def test_dedup_keeps_what_the_warning_cannot_show(self):
        """Only a one-verb corrector entry leaves Edges. A neutral entry names
        no actor, so the Edges arrow keeps the stored direction; an entry with
        several verbs shows their descriptions joined, which matches no stored
        one, so Edges keeps each whole for a reader repairing it."""
        from servers.contract import edges_outside_corrections

        def conn(nid, *verbs):
            return {'id': nid, 'direction': 'incoming', 'relation': verbs[0],
                    'relations': [{'relation': v, 'description': 'why ' + v}
                                  for v in verbs]}

        conns = [conn('one00000', 'supersedes'), conn('neut0000', 'zz_fix'),
                 conn('many0000', 'resolves', 'supersedes')]
        entries = [
            {'id': 'one00000', 'direction': 'corrector', 'relation': 'supersedes'},
            {'id': 'neut0000', 'direction': None, 'relation': 'zz_fix'},
            {'id': 'many0000', 'direction': 'corrector',
             'relation': 'resolves, supersedes'},
        ]
        kept = edges_outside_corrections(conns, entries, mode='balanced')
        self.assertEqual([c['id'] for c in kept], ['neut0000', 'many0000'])
        self.assertEqual([r['relation'] for r in kept[1]['relations']],
                         ['resolves', 'supersedes'])
        # a per-role mode that renders the corrector lean drops nothing
        self.assertEqual(edges_outside_corrections(conns, entries,
                                                   mode={'corrector': 'lean'}), conns)

    # ── Edges ──

    def test_edges_shown(self):
        """Edges appear with target title and relation."""
        nid = self._make_node(title='Source node')
        target_id = self._make_node(type='mechanism', title='Target node')
        self._add_edge(nid, target_id, relation='depends_on', weight=0.9,
                       description='runtime dependency')
        out = self._render(nid)
        self.assertIn('Edges:', out)
        self.assertIn('"Target node"', out)
        self.assertIn('depends_on', out)
        self.assertIn('runtime dependency', out)

    def test_edge_limit_config(self):
        """edge_limit config caps the number of edges shown."""
        nid = self._make_node(title='Hub node')
        for i in range(6):
            tid = self._make_node(title='Spoke %d' % i)
            self._add_edge(nid, tid, weight=0.5 + i * 0.01)
        # Default limit is 5 — should show 5 of 6
        # Edge lines rendered as '    [type id:XX date] ...' (4-space indent + bracket)
        out = self._render(nid)
        edge_lines = [l for l in out.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines), 5)
        # With limit 2
        out2 = self._render(nid, config={'edge_limit': 2})
        edge_lines2 = [l for l in out2.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines2), 2)

    def test_edge_line_age_is_the_relations_not_the_neighbors(self):
        """The age in an edge line's bracket is when THIS relation was written,
        never the neighbor node's age — which used to sit there unlabeled and
        read as the edge's."""
        nid = self._make_node(title='Owner')
        old_nbr = self._make_node(title='Ancient neighbor')
        self.brain.conn.execute("UPDATE nodes SET created_at = ? WHERE id = ?",
                                ('2020-01-01T00:00:00+00:00', old_nbr))
        self._add_edge(nid, old_nbr, relation='grounds', description='written today')
        edge_line = self._edge_line(nid, 'grounds')   # absolute dates
        self.assertNotIn('2020', edge_line)
        from servers.clock import iso_now
        self.assertIn(iso_now()[:10], edge_line)

    def test_get_node_excludes_noise_relations_for_every_reader(self):
        """The read exclusion lives in get_node (registry structural_exclusions
        = the noise aspect), so every reader of `connections` — Anchor, the
        recall surface, the encoder catalog, the healer, consolidation — is
        noise-free without a filter of its own. A pair that carries a noise
        relation AND a semantic one keeps the semantic line; a pair that is
        noise only disappears; the compat `relation` is never a noise verb."""
        self.assertIn('community_member', self.brain.aspects.structural_exclusions)
        nid = self._make_node(title='Member')
        comm = self._make_node(type='community', title='A community')
        peer = self._make_node(title='Peer')
        self._add_edge(comm, nid, relation='community_member', weight=0.9)
        self._add_edge(nid, peer, relation='co_anchored', weight=0.9,
                       description='shared episodic anchor')
        self._add_edge(nid, peer, relation='extends', weight=0.5,
                       description='the semantic claim')
        node = self.brain.get_node(nid)
        self.assertEqual([c['id'] for c in node['connections']], [peer])
        (conn,) = node['connections']
        self.assertEqual([r['relation'] for r in conn['relations']], ['extends'])
        self.assertEqual(conn['relation'], 'extends')
        out = render_rich_node(node)
        self.assertNotIn('community_member', out)
        self.assertNotIn('co_anchored', out)
        self.assertIn('this extends "Peer" — the semantic claim', out)

    def test_communities_ride_as_their_own_line_where_a_format_opts_in(self):
        """Community membership is a `communities` attachment on the canonical
        pull and renders as ONE `Communities:` line — never as edge lines
        (community_member is noise-excluded from connections). cfg
        `communities`: unset → nothing; 'title' → titles only (the encoders,
        the picker's full render); 'ref' → "title" (id) for the reader that
        follows ids with a pull (Anchor's get_nodes / recall)."""
        from servers.contract import (
            GET_NODES_DETAIL_FORMAT, GET_NODES_SCAN_FORMAT, node_format_for)
        from servers.scales.s1.encoder_view import S1_NODE_CONFIG
        from servers.scales.s1.surface_contract import HAIKU_FORMAT, HAIKU_FORMAT_LEAN
        from servers.scales.s2.consolidation_contract import CONSOLIDATION_NODE_FORMAT
        nid = self._make_node(title='Member')
        comm = self._make_node(type='community', title='A community')
        self._add_edge(comm, nid, relation='community_member', weight=0.9)
        node = self.brain.get_node(nid)
        self.assertEqual(node['communities'], [{'id': comm, 'title': 'A community'}])
        self.assertEqual(node['connections'], [])
        # the same member pulled in one batch WITH its community keeps its
        # placement — the lookup decides which endpoint is the community by
        # the node's type, not by which id the caller asked for
        batch = self.brain.get_node([nid, comm])
        self.assertEqual(batch[nid]['communities'], [{'id': comm, 'title': 'A community'}])
        self.assertEqual(batch[comm]['communities'], [])
        ref_line = '  Communities: "A community" (id:%s)' % comm[:8]
        title_line = '  Communities: "A community"'
        # Under the view policy the encoder renders NO community line since
        # 2026-09-16 — it is handed no id to link to, so the line was placement
        # trivia it could not act on (S2CE owns placement). Flag off, the
        # control arm keeps the title.
        from servers.scales.s1.encoder_view import CATALOG_TRIM_CONFIG
        self.assertNotIn('Communities:', render_rich_node(
            node, dict(S1_NODE_CONFIG, **CATALOG_TRIM_CONFIG)))
        for cfg in (None, CONSOLIDATION_NODE_FORMAT, HAIKU_FORMAT_LEAN):
            self.assertNotIn('Communities:', render_rich_node(node, cfg))
        for cfg in (S1_NODE_CONFIG, HAIKU_FORMAT):
            out = render_rich_node(node, cfg)
            self.assertIn(title_line, out)
            self.assertNotIn(comm[:8], out)             # no id to link to
            self.assertNotIn('community_member', out)
        for cfg in (GET_NODES_DETAIL_FORMAT, GET_NODES_SCAN_FORMAT,
                    node_format_for(1, rich=True)):
            out = render_rich_node(node, cfg)
            self.assertIn(ref_line, out)
            self.assertNotIn('community_member', out)
        # a node in no community renders no empty line
        lonely = self._make_node(title='Lonely')
        self.assertEqual(self.brain.get_node(lonely)['communities'], [])
        self.assertNotIn('Communities:', render_rich_node(
            self.brain.get_node(lonely), GET_NODES_DETAIL_FORMAT))

    def test_flat_weights_break_ties_by_relation_recency_and_the_cut_says_so(self):
        """Weights are flat in production (0.5/0.6 everywhere), so the top-N
        cut used to be a tie broken by SQL row order. get_node orders equal
        weights by relation_age, newest first, and the fetch views say when
        the cut dropped edges: 'Edges (5 of 7)'. (The encoder catalog says it
        too, through the view policy's cfg — pinned in test_encoder_view.)"""
        from servers.contract import GET_NODES_DETAIL_FORMAT, GET_NODES_SCAN_FORMAT
        hub = self._make_node(title='Hub')
        spokes = [self._make_node(title='Spoke %d' % i) for i in range(7)]
        for i, s in enumerate(spokes):
            self._add_edge(hub, s, relation='extends', weight=0.6,
                           description='claim %d' % i)
            # the relation's birth: spoke 0 oldest ... spoke 6 newest
            self._stamp_relation(hub, s, '2026-01-%02dT00:00:00+00:00' % (i + 1))
        node = self.brain.get_node(hub)
        self.assertEqual([c['id'] for c in node['connections']], spokes[::-1])
        out = render_rich_node(node, GET_NODES_SCAN_FORMAT)         # limit 5
        self.assertIn('  Edges (5 of 7):', out)
        self.assertIn('claim 6', out)
        self.assertNotIn('claim 0', out)          # the oldest are the ones cut
        # a limit that does not cut (8) keeps the bare header
        self.assertIn('  Edges:\n', render_rich_node(node, GET_NODES_DETAIL_FORMAT))
        # the recency that orders is the recency the line prints: a claim
        # REPAIRED today outranks its untouched siblings even though it was
        # born first
        self._stamp_relation(hub, spokes[0], '2026-09-01T00:00:00+00:00', column='updated_at')
        self.assertEqual(self.brain.get_node(hub)['connections'][0]['id'], spokes[0])
        # a higher weight still wins over recency
        heavy = self._make_node(title='Heavy')
        self._add_edge(hub, heavy, relation='grounds', weight=0.9, description='heavy')
        self._stamp_relation(hub, heavy, '2020-01-01T00:00:00+00:00')
        self.assertEqual(self.brain.get_node(hub)['connections'][0]['id'], heavy)
        # a pair's weight is the max over its SURVIVING relations: a heavy
        # noise relation on a weak pair must not lift it above semantic peers
        noisy = self._make_node(title='Noisy')
        self._add_edge(hub, noisy, relation='extends', weight=0.5, description='weak claim')
        self._add_edge(hub, noisy, relation='co_anchored', weight=0.95)
        conns = self.brain.get_node(hub)['connections']
        self.assertEqual(conns[-1]['id'], noisy)
        self.assertEqual(conns[-1]['weight'], 0.5)

    def test_edge_line_age_is_the_repair_when_the_claim_changed(self):
        """A description repaired in place used to keep its birth date on the
        line — the age lied after exactly the repair we want. edge_relations
        now carries updated_at, stamped only when the claim changes
        (description / weight via the upsert, the verb via rename); a no-op
        re-connect leaves it NULL and the line keeps the honest birth date."""
        from servers.clock import iso_now
        nid = self._make_node(title='Owner')
        nbr = self._make_node(title='Neighbor')
        # seed through the same door the assertions exercise, so the
        # re-connect below is a true no-op whatever weight it resolves
        self.brain.connect_typed(nid, nbr, relation='gaps_in',
                                 description='asserted 9.6.0', encoding_source='test')
        self._stamp_relation(nid, nbr, '2020-01-01T00:00:00+00:00')

        def _row():
            return self.brain.conn.execute(
                "SELECT er.relation, er.updated_at FROM edge_relations er JOIN edges e "
                "ON e.edge_id = er.edge_id WHERE e.source_id = ? AND e.target_id = ? "
                "AND er.archived = 0", (nid, nbr)).fetchone()

        # untouched: birth date on the line, no stamp
        self.assertIsNone(_row()[1])
        self.assertIn('2020', self._edge_line(nid, 'gaps_in'))
        # a re-connect that changes nothing is a true no-op
        self.brain.connect_typed(nid, nbr, relation='gaps_in',
                                 description='asserted 9.6.0', encoding_source='test')
        self.assertIsNone(_row()[1])
        # the repair stamps, and the line's age is the repair
        self.brain.connect_typed(nid, nbr, relation='gaps_in',
                                 description='moved to 9.7.2', encoding_source='test')
        self.assertTrue(_row()[1])
        line = self._edge_line(nid, 'gaps_in')
        self.assertNotIn('2020', line)
        self.assertIn(iso_now()[:10], line)
        self.assertIn('moved to 9.7.2', line)
        # renaming the verb is a claim change too
        self._stamp_relation(nid, nbr, None, column='updated_at')
        self.brain.revise_edge(nid, nbr, 'gaps_in', new_relation='closes',
                               encoding_source='test')
        relation, stamped = _row()
        self.assertEqual(relation, 'closes')
        self.assertTrue(stamped)

    def test_edge_lines_one_per_relation_descriptions_whole(self):
        """A pair carrying several relations renders one line per relation,
        each with its own description untruncated — a reader may copy it
        verbatim as a swap's `old`."""
        nid = self._make_node(title='Owner')
        nbr = self._make_node(title='Neighbor')
        long_desc = 'x' * 250 + ' the load-bearing tail'
        self._add_edge(nid, nbr, relation='extends', description=long_desc)
        self._add_edge(nid, nbr, relation='grounds', description='second claim')
        out = self._render(nid)
        edge_lines = [l for l in out.split('\n') if l.startswith('    [')]
        self.assertEqual(len(edge_lines), 2)
        self.assertTrue(any(l.endswith(' — ' + long_desc) for l in edge_lines), edge_lines)
        self.assertTrue(any(l.endswith(' — second claim') for l in edge_lines))
        self.assertTrue(all(' | ' not in l for l in edge_lines))
        # incoming direction still reads actor-first
        out_nbr = self._render(nbr)
        self.assertIn('"Owner" extends this — ' + long_desc, out_nbr)

    def test_edge_title_cuts_at_the_ref_limit_and_says_so(self):
        """The title on an edge line is a lure, not a summary: it cuts at
        TITLE_REF_LIMIT and ends in the mark, so a reader who wants the rest
        fetches the node instead of answering from a phrase. The description
        beside it stays whole — the two have different jobs."""
        from servers.contract import TITLE_REF_LIMIT
        from servers.loud_truncation import looks_truncated
        long_title = 'T' * (TITLE_REF_LIMIT + 60) + ' the verdict'
        nid = self._make_node(title='Owner')
        nbr = self._make_node(title=long_title)
        self._add_edge(nid, nbr, relation='extends', description='why it holds')
        line = [l for l in self._render(nid).split('\n')
                if l.startswith('    [')][0]
        rendered = line.split('"')[1]
        self.assertEqual(len(rendered), TITLE_REF_LIMIT)
        self.assertTrue(looks_truncated(rendered), rendered[-20:])
        self.assertTrue(line.endswith(' — why it holds'))   # description whole

    def test_short_edge_title_carries_no_mark(self):
        """The mark is a signal, not decoration — it only appears on a cut."""
        nid = self._make_node(title='Owner')
        nbr = self._make_node(title='Short neighbour')
        self._add_edge(nid, nbr, relation='extends', description='d')
        self.assertIn('"Short neighbour"', self._render(nid))

    def test_differential_project_mark_on_mismatch(self):
        """cfg['scope']: foreign project renders the ⚠ mark, the generic
        'Project:' KV line is suppressed."""
        nid = self._make_node(title='Foreign fact', project='acme-site')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertIn('⚠ From another project: acme-site', out)
        self.assertNotIn('Project: acme-site', out)

    def test_differential_project_silent_on_match(self):
        """Same-project node renders NO project line at all in differential
        mode — a same-project line on a one-project corpus is noise."""
        nid = self._make_node(title='Home fact', project='brain')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertNotIn('From another project', out)
        self.assertNotIn('Project:', out)

    def test_differential_project_neutral_on_unscoped_node(self):
        """A node with no project provenance is never marked foreign —
        unknown is neutral, matching the scope lane semantics."""
        nid = self._make_node(title='Unscoped fact')
        out = self._render(nid, {'scope': {'project': 'brain'}})
        self.assertNotIn('From another project', out)

    def test_differential_counterpart_mark_on_mismatch(self):
        """The counterpart dimension marks through the SAME central
        scope_marks path — adding a dimension re-threads nothing."""
        nid = self._make_node(title='Other-speaker fact', counterpart='Dana')
        out = self._render(nid, {'scope': {'project': 'brain',
                                           'counterpart': 'Ada'}})
        self.assertIn('⚠ Learned with another counterpart: Dana', out)
        self.assertNotIn('Counterpart: Dana', out)

    def test_differential_counterpart_silent_on_match(self):
        nid = self._make_node(title='Same-speaker fact', counterpart='Ada')
        out = self._render(nid, {'scope': {'counterpart': 'Ada'}})
        self.assertNotIn('another counterpart', out)
        self.assertNotIn('Counterpart:', out)

    def test_legacy_render_keeps_generic_project_line(self):
        """Callers that don't declare a scope keep the pre-existing generic
        KV render — no information loss for unwired consumers."""
        nid = self._make_node(title='Legacy view', project='acme-site',
                              counterpart='Ada')
        out = self._render(nid)
        self.assertIn('Project: acme-site', out)
        self.assertNotIn('From another project', out)
        # counterpart is differential-ONLY: its value is the install default
        # (identical on every node), so the generic KV line is pure noise
        # and stays suppressed even for undeclared callers.
        self.assertNotIn('Counterpart:', out)

class TestRenderDepthLadder(unittest.TestCase):
    """Every config that reaches render_rich_node draws its metadata depth
    from the shared ladder (contract.META_*). The bug this pins: the S1
    encoder catalog declared no `metadata_limit`, silently inherited 300,
    and handed the encoder a cut `reasoning` — which it then copied,
    ellipsis and all, into a swap's `old` that could never match (0682f808).
    An unnamed number nobody could grep is what made that invisible.
    """

    # The modules that own a config reaching render_rich_node. Membership is
    # checked by test_no_render_config_lives_outside_these_modules below, so
    # the list cannot silently miss a sixth home.
    CONFIG_MODULES = (
        'servers.contract',
        'servers.scales.s1.encoder_view',
        'servers.scales.s1.surface_contract',
        'servers.scales.s2.community_contract',
        'servers.scales.s2.consolidation_contract',
    )

    def _registry(self):
        """Every node-render config, DISCOVERED rather than listed — a
        hand-written roster passes green on the config it forgot, which is
        the same miss the ladder exists to close."""
        import importlib
        found = {}
        for mod_name in self.CONFIG_MODULES:
            mod = importlib.import_module(mod_name)
            for name, val in vars(mod).items():
                if name.startswith('_') or not isinstance(val, dict):
                    continue
                # `metadata_limit` is the discriminator, not `content_limit`:
                # the latter is overloaded — the candidate-pool dicts carry it
                # as a pipeline setting, not a render depth.
                if 'metadata_limit' not in val:
                    continue
                found['%s.%s' % (mod_name.rsplit('.', 1)[-1], name)] = val
        return found

    def test_discovery_actually_finds_the_configs(self):
        """A derived roster that finds nothing passes every test below it —
        and a loose floor lets a whole module drop out unnoticed, so every
        owning module must still contribute."""
        found = self._registry()
        for mod_name in self.CONFIG_MODULES:
            short = mod_name.rsplit('.', 1)[-1]
            with self.subTest(module=short):
                self.assertTrue(
                    any(k.startswith(short + '.') for k in found),
                    '%s contributed no render config' % short)
        self.assertIn('contract.GET_NODES_DETAIL_FORMAT', found)
        self.assertIn('encoder_view.S1_NODE_CONFIG', found)

    def test_no_render_config_lives_outside_these_modules(self):
        """The module list is the only hand-maintained part — this closes it.
        A new file declaring a render depth fails here, by name."""
        import os
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'servers')
        allowed = {m.split('.', 1)[1].replace('.', os.sep) + '.py'
                   for m in self.CONFIG_MODULES}
        strays = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith('.py'):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                if rel in allowed:
                    continue
                with open(os.path.join(dirpath, f),
                          encoding='utf-8') as fh:
                    body = fh.read()
                if "'metadata_limit':" in body or '"metadata_limit":' in body:
                    strays.append(rel)
        self.assertEqual(strays, [], 'render depth declared outside the '
                                     'owning modules — add the module to '
                                     'CONFIG_MODULES or move the config')

    def test_ladder_is_ordered(self):
        from servers.contract import (META_BRIEF, META_DEEP, META_NONE,
                                      META_WHOLE)
        self.assertLess(META_NONE, META_BRIEF)
        self.assertLess(META_BRIEF, META_DEEP)
        self.assertLess(META_DEEP, META_WHOLE)

    def test_every_config_declares_a_depth_from_the_ladder(self):
        """Declared, never inherited — and never a bare number. A config
        that omits the key falls through to render_rich_node's default,
        which is how a 300 nobody chose reached the encoder."""
        from servers.contract import (META_BRIEF, META_DEEP, META_NONE,
                                      META_WHOLE)
        from servers.scales.s2.community_contract import S2CE_DECISION_META
        allowed = {META_NONE, META_BRIEF, META_DEEP, META_WHOLE,
                   S2CE_DECISION_META}
        for name, cfg in self._registry().items():
            with self.subTest(config=name):
                self.assertIn('metadata_limit', cfg,
                              '%s inherits the default instead of declaring '
                              'a depth' % name)
                self.assertIn(cfg['metadata_limit'], allowed,
                              '%s uses a depth outside the ladder — name it '
                              'in contract.py or own it locally' % name)

    def test_the_expand_doors_are_the_deepest(self):
        """The swap refusal tells a reader to expand with get_nodes. That
        door must therefore be deeper than the views it rescues, and the
        rich drill deeper still — otherwise the advice returns the reader
        to the same cut it is trying to escape."""
        from servers.contract import (GET_NODES_DETAIL_FORMAT,
                                      GET_NODES_RICH_LIFT, META_DEEP,
                                      META_WHOLE)
        self.assertGreaterEqual(
            GET_NODES_DETAIL_FORMAT['metadata_limit'], META_DEEP)
        self.assertGreaterEqual(
            GET_NODES_RICH_LIFT['metadata_limit'], META_WHOLE)

    def test_the_encoder_catalog_matches_its_own_drill_in(self):
        """The S1 encoder authors swaps from the catalog and is told to
        expand with get_nodes. If the two disagree, the deeper door does
        not rescue what the shallower one cut."""
        from servers.contract import GET_NODES_DETAIL_FORMAT
        from servers.scales.s1.encoder_view import CATALOG_FULL_CONFIG
        self.assertEqual(CATALOG_FULL_CONFIG['metadata_limit'],
                         GET_NODES_DETAIL_FORMAT['metadata_limit'])


if __name__ == '__main__':
    unittest.main()
