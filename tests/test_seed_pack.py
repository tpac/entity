"""Seed-pack integrity — the baby-brain bootstrap must not drift.

Static checks on SEED_NODES / SEED_EDGES (no brain needed, fast). They guard the
two drifts found 2026-06-07: banned `related`/empty edge relations, and the
v28-dropped `keywords` field lingering on seed nodes. The seed has no other test
coverage, which is exactly why those drifted unnoticed — this is the floor.
"""
import json
import os
import unittest

from servers.seed_pack import COMMUNITY_SLUG, SEED_NODES, SEED_EDGES

# Relations the rest of the system forbids: they carry zero information and
# pollute the activation kernel (the encoder is explicitly told NEVER to emit
# them). A fresh brain must not bootstrap with them.
BANNED_RELATIONS = {'related', 'related_to', ''}

# Fields dropped from the schema that must not appear on seed nodes — they would
# route to node_metadata_kv as dead metadata on every fresh install.
DEAD_NODE_FIELDS = {'keywords'}


class TestSeedPackIntegrity(unittest.TestCase):
    def test_no_banned_edge_relations(self):
        """SEED_EDGES must use meaningful relations — never related/related_to/empty."""
        offenders = [(e.get('source'), e.get('target'), e.get('relation'))
                     for e in SEED_EDGES
                     if (e.get('relation') or '').strip() in BANNED_RELATIONS]
        self.assertEqual(offenders, [],
                         "SEED_EDGES use banned/empty relations: %r" % offenders)

    def test_every_edge_has_relation_and_description(self):
        """Every seed edge needs a relation AND a why (description) — the why is
        embedded for recall, so a typed-but-undescribed edge is half-blind."""
        for e in SEED_EDGES:
            pair = (e.get('source'), e.get('target'))
            self.assertTrue((e.get('relation') or '').strip(),
                            "edge %r missing relation" % (pair,))
            self.assertTrue((e.get('description') or '').strip(),
                            "edge %r missing description" % (pair,))

    def test_no_dead_fields_on_seed_nodes(self):
        """Seed nodes must not carry schema-dropped fields (e.g. v28 `keywords`)."""
        for n in SEED_NODES:
            dead = DEAD_NODE_FIELDS & set(n.keys())
            self.assertEqual(dead, set(),
                             "seed node %r carries dead field(s) %r" % (n.get('slug'), dead))


class TestNurseryPackContracts(unittest.TestCase):
    """The D-5 redesign's invariants (2026-08-30). These are the pack's own
    design principles made mechanical — a drift in any of them re-opens a
    disease the redesign specifically cured."""

    @classmethod
    def setUpClass(cls):
        aspects_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'servers', 'scales', 's2', 'aspects_v1.json')
        with open(aspects_path) as f:
            aspects = json.load(f)
        cls.aspect_node_types = set()
        cls.aspect_edge_relations = set()
        cls.unrouted_edge_relations = set()  # noise + generic_relation
        from servers.aspect_store import member_name
        for name, spec in aspects.items():
            if name.startswith('_'):
                continue
            cls.aspect_node_types.update(
                member_name(m, 'node_types') for m in spec.get('node_types', []))
            rels = [member_name(m, 'edge_relations') for m in spec.get('edge_relations', [])]
            cls.aspect_edge_relations.update(rels)
            if name in ('noise', 'generic_relation'):
                cls.unrouted_edge_relations.update(rels)
        cls.slugs = {n['slug'] for n in SEED_NODES}

    def test_pack_shape(self):
        """28 nodes: 4 locked safety core, 6 developmental, 3 exemplars,
        exactly one community node (the Seed)."""
        self.assertEqual(len(SEED_NODES), 28)
        self.assertEqual(sum(1 for n in SEED_NODES if n.get('locked')), 4)
        self.assertEqual(sum(1 for n in SEED_NODES if n.get('developmental')), 6)
        self.assertEqual(sum(1 for n in SEED_NODES if n.get('exemplar')), 3)
        self.assertEqual([n['slug'] for n in SEED_NODES if n['type'] == 'community'],
                         [COMMUNITY_SLUG])

    def test_node_types_are_aspect_registered(self):
        """Every seed type routes into an aspect family — the pack must not
        teach the encoder unrouted type labels (v2 review blocker #4)."""
        for n in SEED_NODES:
            self.assertIn(n['type'], self.aspect_node_types,
                          "seed %r has unregistered type %r" % (n['slug'], n['type']))

    def test_edge_relations_are_registered_and_routed(self):
        """Every SEED_EDGES relation is aspect-registered and NOT in the
        noise/generic_relation families (those are dropped from cohesion)."""
        for e in SEED_EDGES:
            rel = e['relation']
            self.assertIn(rel, self.aspect_edge_relations,
                          "edge relation %r is not in aspects_v1.json" % rel)
            self.assertNotIn(rel, self.unrouted_edge_relations,
                             "edge relation %r routes to noise/generic" % rel)

    def test_edge_endpoints_resolve(self):
        for e in SEED_EDGES:
            self.assertIn(e['source'], self.slugs)
            self.assertIn(e['target'], self.slugs)

    def test_names_only_in_tribute_sites(self):
        """'Tom' / 'Anchor' appear ONLY in the pack's true origin story
        (the tribute amendment to P4, ratified 2026-08-30) — and never in
        a locked node. No config placeholders anywhere: the pack is
        name-free by construction."""
        tribute_slugs = {'seed_community', 'silent_failure_lesson',
                         'decision_shape_exemplar'}
        for n in SEED_NODES:
            text = ' '.join(str(v) for v in n.values())
            self.assertNotIn('{name}', text, n['slug'])
            self.assertNotIn('{operator}', text, n['slug'])
            has_name = ('Tom' in text) or ('Anchor' in text)
            if n['slug'] in tribute_slugs:
                self.assertTrue(has_name,
                                "tribute site %r lost its names" % n['slug'])
            else:
                self.assertFalse(has_name,
                                 "instance name outside tribute sites: %r" % n['slug'])
            if n.get('locked'):
                self.assertFalse(has_name,
                                 "locked node %r must stay name-free" % n['slug'])

    def test_developmental_nodes_are_unlocked_rules_of_low_confidence(self):
        """Scaffolds are unlocked, sub-0.8 confidence (they're placeholders),
        and their content carries a self-transformation commitment — the
        verbs vary by design (six different closings), so any of the
        commitment family counts."""
        commitment_verbs = ('revise', 'rewrite', 'becomes', 'become', 'archive')
        for n in SEED_NODES:
            if not n.get('developmental'):
                continue
            self.assertFalse(n.get('locked'), n['slug'])
            self.assertLess(n['confidence'], 0.8, n['slug'])
            content = n['content'].lower()
            self.assertTrue(any(v in content for v in commitment_verbs),
                            "scaffold %r has no self-transformation commitment" % n['slug'])


if __name__ == '__main__':
    unittest.main()
