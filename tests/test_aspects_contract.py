"""Contract test — locks the equivalence between REQUIRED_ASPECTS and aspects_v1.json.

The contract: every name in REQUIRED_ASPECTS must appear as a key in
aspects_v1.json, and vice versa. Adding a name to one without the other is a
failure — the test catches drift before it reaches a fresh brain.

Also verifies the seed JSON parses cleanly via AspectRegistry.from_dict — so
the in-memory shape stays in sync with the seed shape.

Step 4 of 14 in the unified-aspects work.
"""

import json
import os
import unittest

from servers.aspects import REQUIRED_ASPECTS, AspectRegistry, Aspect


SEED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'servers', 'scales', 's2', 'aspects_v1.json'
)


def _load_seed():
    """Seed WITHOUT the '_'-prefixed documentation keys (_schema) — the
    aspect entries every curation test iterates. Use _load_seed_raw for the
    file as-is."""
    return {k: v for k, v in _load_seed_raw().items() if not k.startswith('_')}


def _load_seed_raw():
    with open(SEED_PATH) as f:
        return json.load(f)


def _names(spec, category):
    """A seed entry's member strings for `category` — members may be bare
    strings or {verb|type, <field>: value} objects since orientation landed."""
    from servers.aspect_store import member_name
    return [member_name(m, category) for m in (spec.get(category) or [])]


class TestSchemaDocEntry(unittest.TestCase):
    """The `_schema` entry is the file's own documentation (JSON has no
    comments; '_'-prefixed keys are the reserved convention, skipped by the
    registry loader, the validator, the write door, and the dashboard).
    It must document every key an aspect entry carries — a new field without
    an explainer fails here."""

    def test_schema_documents_every_entry_key(self):
        raw = _load_seed_raw()
        self.assertIn('_schema', raw)
        doc = raw['_schema']
        entry_keys = set()
        for name, spec in raw.items():
            if not name.startswith('_'):
                entry_keys |= set(spec.keys())
        undocumented = entry_keys - set(doc.keys())
        self.assertEqual(
            undocumented, set(),
            'aspect entry keys missing a _schema explainer: %s' % undocumented)

    def test_schema_is_skipped_by_the_registry(self):
        from servers.aspects import AspectRegistry
        registry = AspectRegistry.from_dict(None, _load_seed_raw())
        self.assertNotIn('_schema', registry.all())

    def test_schema_passes_the_write_gate(self):
        from servers.aspect_store import validate_taxonomy
        self.assertEqual(validate_taxonomy(_load_seed_raw()), [])


# Verbs that REPLACE prior knowledge. Every one must live in
# correction_improvement, because `brain.correction_enrich()` walks that aspect
# alone — a replacement verb missing from it means superseded nodes surface with
# no pointer to their successor. Shared with
# tests/test_surface_transitions.py::test_correction_enrich_walks_every_replacement_verb
# so the membership contract and the end-to-end behaviour cover the same list.
REPLACEMENT_VERBS = ('supersedes', 'superseded_by', 'absorbed_into',
                     'consolidated_into')


class TestSeedExists(unittest.TestCase):
    def test_seed_file_exists(self):
        self.assertTrue(os.path.isfile(SEED_PATH),
                        "aspects_v1.json must exist at %s" % SEED_PATH)

    def test_seed_parses_as_json(self):
        # raises if not valid JSON
        data = _load_seed()
        self.assertIsInstance(data, dict)


class TestContractEquivalence(unittest.TestCase):
    """REQUIRED_ASPECTS keys ≡ aspects_v1.json keys."""

    def setUp(self):
        self.seed = _load_seed()
        self.required = set(REQUIRED_ASPECTS)
        self.seeded = set(self.seed.keys())

    def test_no_required_aspect_missing_from_seed(self):
        missing = self.required - self.seeded
        self.assertEqual(
            missing, set(),
            "REQUIRED_ASPECTS contains names absent from aspects_v1.json: %s" % missing
        )

    def test_no_extra_aspect_in_seed(self):
        # Seed should be EXACTLY the required set in v1. Emergent aspects
        # are created by AspectIntegration after boot, not seeded.
        extras = self.seeded - self.required
        self.assertEqual(
            extras, set(),
            "aspects_v1.json contains names not in REQUIRED_ASPECTS: %s" % extras
        )
        # Consolidated from the former test_set_equivalence: with no missing and
        # no extra names, the sets are equal — assert it directly here.
        self.assertEqual(self.required, self.seeded)


class TestSeedShape(unittest.TestCase):
    """Each seed entry has the structural keys AspectRegistry.from_dict expects."""

    def setUp(self):
        self.seed = _load_seed()

    def test_each_entry_has_required_fields(self):
        required_fields = {'node_types', 'edge_relations', 'meaning',
                           'dimension', 'locked', 'metadata',
                           'accepts', 'routable', 'prompt_visible'}
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                missing = required_fields - set(spec.keys())
                self.assertEqual(
                    missing, set(),
                    "%s missing required fields: %s" % (name, missing)
                )

    def test_accepts_is_nonempty_and_valid(self):
        # This assertion is what makes a JSON-only aspect addition safe: an
        # entry cannot ship without declaring which categories it takes.
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                accepts = spec['accepts']
                self.assertTrue(accepts, "%s.accepts is empty" % name)
                self.assertTrue(
                    set(accepts) <= {'node_types', 'edge_relations'},
                    "%s.accepts has unknown categories: %s" % (name, accepts))

    def test_fact_flags_are_booleans(self):
        for name, spec in self.seed.items():
            for flag in ('routable', 'prompt_visible'):
                with self.subTest(aspect=name, flag=flag):
                    self.assertIsInstance(spec[flag], bool)

    def test_non_routable_set_is_the_system_owned_pair(self):
        # survivor_lineage is system-written (absorbed_into), settlement is
        # the closed consolidation-suppression list — neither is ever offered
        # to the classifier. Everything else routes. A new non-routable
        # aspect is a deliberate edit — update this pin consciously.
        non_routable = {name for name, spec in self.seed.items()
                        if not spec['routable']}
        self.assertEqual(non_routable, {'survivor_lineage', 'settlement'})

    def test_prompt_invisible_set_is_pinned(self):
        # The catch-alls + the system aspects stay out of encoder vocabulary
        # blocks (the deleted EDGE_ASPECT_PROMPT_SKIP contract, now declared).
        # settlement's verbs already appear via their semantic home aspects —
        # offering them again as a "settlement" block would double-list them.
        invisible = {name for name, spec in self.seed.items()
                     if not spec['prompt_visible']}
        self.assertEqual(
            invisible,
            {'generic_relation', 'noise', 'survivor_lineage', 'settlement'})

    def test_no_aspect_name_is_a_member_string(self):
        # An aspect NAME is a role, never a relation or type string. The
        # noise seed once listed `temporal_sequence`, `extension_refinement`
        # and `validation_evidence` as edge relations (bug 40e7125a) — a
        # mis-seed that let an encoder's confused write land as noise and
        # made 'not in noise' lie about the taxonomy's own vocabulary.
        names = set(self.seed)
        for name, spec in self.seed.items():
            for category in ('node_types', 'edge_relations'):
                leaked = names & set(_names(spec, category))
                self.assertEqual(
                    leaked, set(),
                    "aspect '%s'.%s lists aspect names as members: %s"
                    % (name, category, sorted(leaked)))

    def test_no_aspect_is_empty(self):
        # Required aspects must have at least one member somewhere — otherwise
        # they're a husk. Emergent aspects (when they appear later) may be
        # empty transiently but required ones are seeded with members.
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                has_members = bool(spec['node_types']) or bool(spec['edge_relations'])
                self.assertTrue(
                    has_members,
                    "Required aspect '%s' has no node_types AND no edge_relations — empty husk" % name
                )

    def test_all_seeded_aspects_are_locked(self):
        # Required aspects ship locked=True so the maintenance unit can't
        # remove or rename them
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                self.assertTrue(
                    spec['locked'],
                    "Required aspect '%s' must be locked=True in seed" % name
                )

    def test_meaning_is_substantive(self):
        # Meaning is embedded for recall — empty/short strings hurt retrieval.
        # Require at least 30 chars (sentence-ish).
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                self.assertGreaterEqual(
                    len(spec['meaning']), 30,
                    "%s.meaning is too short (<30 chars) — won't embed usefully" % name
                )

    def test_dimension_is_known(self):
        # v1 only has 'semantic'. Future dimensions emerge via AspectIntegration.
        for name, spec in self.seed.items():
            with self.subTest(aspect=name):
                self.assertEqual(
                    spec['dimension'], 'semantic',
                    "v1 seed only supports dimension='semantic'; got '%s' for %s" % (
                        spec['dimension'], name)
                )


class TestSeedLoadsViaRegistry(unittest.TestCase):
    """The seed parses cleanly through AspectRegistry.from_dict."""

    def setUp(self):
        self.seed = _load_seed()
        self.registry = AspectRegistry.from_dict(brain=None, data=self.seed)

    def test_all_aspects_present(self):
        self.assertEqual(set(self.registry.all().keys()), set(self.seed.keys()))

    def test_required_attributes_resolve(self):
        # Every name in REQUIRED_ASPECTS resolves via attribute access
        for name in REQUIRED_ASPECTS:
            with self.subTest(aspect=name):
                aspect = getattr(self.registry, name)
                self.assertIsInstance(aspect, Aspect)
                self.assertEqual(aspect.name, name)

    def test_required_member_lists_non_empty(self):
        # Round-trip: registry's view of each required aspect should match
        # the seed's view (members, meaning).
        for name in REQUIRED_ASPECTS:
            with self.subTest(aspect=name):
                aspect = self.registry.by_name(name)
                seed_entry = self.seed[name]
                self.assertEqual(
                    list(aspect.node_types), _names(seed_entry, 'node_types'),
                    "%s.node_types mismatch" % name
                )
                self.assertEqual(
                    list(aspect.edge_relations), _names(seed_entry, 'edge_relations'),
                    "%s.edge_relations mismatch" % name
                )
                self.assertEqual(aspect.meaning, seed_entry['meaning'])
                self.assertEqual(aspect.dimension, seed_entry['dimension'])
                self.assertEqual(aspect.locked, seed_entry['locked'])

    def test_correction_improvement_has_both_sides(self):
        # The natural unification case — correction_improvement carries BOTH
        # node_types (from old correction_supersession) and edge_relations
        # (from old correction_improvement).
        a = self.registry.correction_improvement
        self.assertGreater(len(a.node_types), 0,
                           "correction_improvement should have node_types after unification")
        self.assertGreater(len(a.edge_relations), 0,
                           "correction_improvement should have edge_relations after unification")
        # specific markers
        self.assertIn('correction', a.node_types)
        self.assertIn('corrects', a.edge_relations)

    def test_replacement_verbs_are_correction_class(self):
        """`supersedes`/`superseded_by` must be in correction_improvement.

        `brain.correction_enrich()` walks correction_improvement ALONE, so a
        replacement verb missing from it means every superseded node surfaces
        with no pointer to its successor — the exact bug the seed shipped until
        2026-07-24 (332 supersedes edges, none reaching recall). They stay
        multi-homed in hierarchical_structure too; this pins the correction
        half, which is the one recall depends on.
        """
        rels = self.registry.correction_improvement.edge_relations
        for verb in REPLACEMENT_VERBS:
            with self.subTest(verb=verb):
                self.assertIn(
                    verb, rels,
                    "%s is a replacement verb — correction_enrich() must see it"
                    % verb)
        # Still structurally homed as well — multi-membership, not a move.
        self.assertIn('supersedes',
                      self.registry.hierarchical_structure.edge_relations)

    def test_healer_display_labels_present(self):
        # All 8 healer-facing aspects should carry display_label metadata
        healer_aspects = (
            'correction_improvement', 'extension_refinement', 'explanation_causation',
            'dependency_flow', 'contradiction_conflict', 'validation_evidence',
            'hierarchical_structure', 'temporal_sequence',
        )
        for name in healer_aspects:
            with self.subTest(aspect=name):
                aspect = self.registry.by_name(name)
                self.assertIn('display_label', aspect.metadata,
                              "%s should have metadata.display_label for healer" % name)
                self.assertGreater(len(aspect.metadata['display_label']), 0)


class TestWisdomCuration(unittest.TestCase):
    """The wisdom aspect stays curated: generative types in, operational and
    tactical record-keeping out. (Moved from test_frame.py 2026-08-05 when
    the Frame stopped consuming the aspect — the curation contract outlives
    that consumer: encoder routing and scope policy still read it.)"""

    def setUp(self):
        self.aspects = AspectRegistry.from_dict(brain=None, data=_load_seed())

    def test_wisdom_includes_generative_types(self):
        members = self.aspects.wisdom.node_types
        for t in ('insight', 'lesson', 'principle', 'vision'):
            self.assertIn(t, members)

    def test_wisdom_excludes_operational_and_tactical(self):
        members = self.aspects.wisdom.node_types
        for t in ('rule', 'operator', 'decision', 'fact', 'bug', 'mechanism'):
            self.assertNotIn(t, members)


class TestMultiMembershipShape(unittest.TestCase):
    """Multi-membership is allowed (a string can appear in 2+ aspects).
    Reverse lookup remains deterministic — noise wins if it claims the string
    at all, otherwise the FIRST aspect to list it in JSON iteration order wins
    for `by_node_type`/`by_edge_relation`.

    Was previously TestNoMemberOverlap, which enforced single-membership.
    Single-membership was dropped 2026-05-08 — recall expressivity (one
    edge serving multiple aspect-filtered queries) beats reverse-lookup
    determinism. The deterministic-lookup property is now achieved by
    JSON-iteration-order convention rather than uniqueness.
    """

    def setUp(self):
        self.seed = _load_seed()

    def test_reverse_lookup_resolves_deterministically(self):
        # If a string is in N aspects, by_X should still resolve to ONE.
        # The contract: first aspect to claim it (JSON iteration order) wins.
        from servers.aspects import AspectRegistry
        registry = AspectRegistry.from_dict(brain=None, data=self.seed)
        for name, spec in self.seed.items():
            for t in _names(spec, 'node_types'):
                resolved = registry.by_node_type(t)
                self.assertIsNotNone(resolved, 'by_node_type(%s) returned None' % t)
            for r in _names(spec, 'edge_relations'):
                resolved = registry.by_edge_relation(r)
                self.assertIsNotNone(resolved, 'by_edge_relation(%s) returned None' % r)
        # Pin FIRST-claimant on real multi-homed members — this is what caught
        # the from_dict/_load disagreement (last-claimant reported
        # hierarchical_structure / survivor_lineage for these instead).
        self.assertEqual(registry.by_edge_relation('supersedes').name,
                         'correction_improvement')
        self.assertEqual(registry.by_edge_relation('absorbed_into').name,
                         'correction_improvement')
        # And per-string agreement between the two construction paths, whole
        # seed: from_dict must resolve every string exactly like a file-loaded
        # registry would (one _adopt body — this pins that it stays one).
        # Rule: noise wins if it claims the string at all, else first-claimant.
        noise_nodes = set(_names(self.seed['noise'], 'node_types'))
        noise_edges = set(_names(self.seed['noise'], 'edge_relations'))
        for t, primary in registry._reverse_node.items():
            expected = 'noise' if t in noise_nodes else next(
                n for n, s in self.seed.items() if t in _names(s, 'node_types'))
            self.assertEqual(primary, expected,
                             'primary-aspect rule violated for node type %s' % t)
        for r, primary in registry._reverse_edge.items():
            expected = 'noise' if r in noise_edges else next(
                n for n, s in self.seed.items() if r in _names(s, 'edge_relations'))
            self.assertEqual(primary, expected,
                             'primary-aspect rule violated for relation %s' % r)


class TestNoiseVetoes(unittest.TestCase):
    """Noise is a VETO, not an exclusive category.

    A string in noise is machinery whatever else claims it. This replaced the
    noise ∩ semantic = ∅ invariant, which forbade the dual state instead of
    resolving it — and in doing so made the additive heal refuse whole
    reconciles, freezing every divergent install on its old taxonomy.

    Each test here pins one derived view against the dual state, because the
    filters read membership (already veto-shaped) while community adjacency
    reads the PRIMARY aspect. Those had to be made to agree.
    """

    def setUp(self):
        import copy
        self.seed = _load_seed()
        # `corrects` is semantic (correction_improvement, a lineage aspect) —
        # dual-home it into noise, the exact state the old gate refused.
        self.dual = copy.deepcopy(self.seed)
        self.dual['noise']['edge_relations'].append('corrects')
        self.registry = AspectRegistry.from_dict(None, self.dual)

    def test_primary_aspect_is_noise_regardless_of_file_order(self):
        """Community typed adjacency skips by PRIMARY family, so if primary
        stayed first-claimant a dual-homed noise relation would be counted as
        cohesion whenever noise sorted later in the file."""
        self.assertEqual(self.registry.by_edge_relation('corrects').name, 'noise')
        self.assertEqual(self.registry.primary_edge_map()['corrects'], 'noise')

    def test_exclusion_sets_still_carry_it(self):
        self.assertIn('corrects', self.registry.structural_exclusions)
        self.assertIn('corrects', self.registry.traversal_exclusions)

    def test_vocabulary_block_stops_offering_it(self):
        from servers.aspects import render_edge_aspects_block
        block = render_edge_aspects_block(self.registry.all())
        self.assertNotIn('corrects', block)
        # Still a useful block — only the vetoed verb is gone.
        self.assertIn('Edge Aspects', block)

    def test_semantic_membership_is_preserved_not_erased(self):
        """The veto governs what a relation DOES, not what it MEANS — anything
        asking which aspects claim it must still see the semantic one."""
        self.assertIn('corrects', self.registry.by_name(
            'correction_improvement').edge_relations)

    def test_baseline_seed_has_no_dual_membership_today(self):
        """The rules above only bite on dual state; the shipped seed has none,
        so enabling the veto changed no live behaviour."""
        noise = set(_names(self.seed['noise'], 'edge_relations'))
        overlap = {r for n, s in self.seed.items() if n != 'noise'
                   for r in _names(s, 'edge_relations') if r in noise}
        self.assertEqual(overlap, set())


class TestStructuralValidity(unittest.TestCase):
    """The seed satisfies the FULL structural invariant set — one call.

    validate_taxonomy is the single home of the structural rules (entries are
    objects; member lists are unique non-empty strings; all required aspects
    present). The registry's write door refuses on it and the boot load reports
    on it, so pinning the seed against the same function means the shipped
    baseline can never fail the gate a write is held to. Replaces the
    hand-rolled lists-shape / well-formed-members tests, which each
    re-implemented one slice of it.

    Noise ∩ semantic = ∅ is deliberately NOT here any more: noise vetoes in
    the registry instead of being forbidden by the gate. See
    TestNoiseVetoes below for the rules that replaced it.

    Seed-only CURATION standards (locked, meaning length, display labels,
    no-extra-aspects) deliberately stay as separate tests above — the working
    copy legitimately grows emergent unlocked aspects, so those must never
    enter the write gate.
    """

    def setUp(self):
        self.seed = _load_seed()

    def test_seed_passes_the_write_gate(self):
        from servers.aspect_store import validate_taxonomy
        self.assertEqual(validate_taxonomy(self.seed), [])

    def test_dual_noise_and_semantic_membership_is_valid(self):
        """A string in noise AND a semantic aspect must PASS the gate.

        This inverts the 2026-07-24 rule. Forbidding it made the additive heal
        refuse whole reconciles whenever a seed member collided with a
        classifier-grown working copy, which left every divergent install stuck
        on its old taxonomy with no migration lane. Noise vetoes instead — the
        behaviour lives in the registry (TestNoiseVetoes), not in a gate that
        rejects the data.
        """
        from servers.aspect_store import validate_taxonomy
        import copy
        dual = copy.deepcopy(self.seed)
        dual['noise']['edge_relations'].append('corrects')
        self.assertEqual(validate_taxonomy(dual), [])

    def test_validator_catches_malformed_and_duplicate_members(self):
        from servers.aspect_store import validate_taxonomy
        import copy
        broken = copy.deepcopy(self.seed)
        broken['correction_improvement']['edge_relations'].append('')
        broken['correction_improvement']['edge_relations'].append('corrects')
        violations = validate_taxonomy(broken)
        self.assertTrue(any('malformed member' in v for v in violations), violations)
        self.assertTrue(any("duplicate member 'corrects'" in v for v in violations),
                        violations)

    def test_validator_catches_missing_required_aspect(self):
        from servers.aspect_store import validate_taxonomy
        broken = {k: v for k, v in self.seed.items() if k != 'survivor_lineage'}
        violations = validate_taxonomy(broken)
        self.assertTrue(any('survivor_lineage' in v for v in violations), violations)

    def test_validator_catches_malformed_fact_fields(self):
        # Invariant 5: accepts must be a non-empty unique subset of the two
        # categories; the three flags must be booleans — when present.
        # Absence is legitimate (emergent aspects, pre-heal working copies).
        from servers.aspect_store import validate_taxonomy
        import copy
        broken = copy.deepcopy(self.seed)
        broken['noise']['accepts'] = []
        broken['wisdom']['accepts'] = ['node_types', 'everything']
        broken['lesson_insight']['routable'] = 'yes'
        violations = validate_taxonomy(broken)
        self.assertTrue(any("'noise'.accepts" in v for v in violations), violations)
        self.assertTrue(any("'wisdom'.accepts" in v for v in violations), violations)
        self.assertTrue(any("'lesson_insight'.routable" in v for v in violations),
                        violations)

    def test_validator_allows_absent_fact_fields(self):
        # A pre-Step-4 working copy (no fact keys anywhere) must still pass
        # the structural gate — presence is seed curation, not structure.
        from servers.aspect_store import validate_taxonomy
        import copy
        legacy = copy.deepcopy(self.seed)
        for spec in legacy.values():
            for k in ('accepts', 'routable', 'prompt_visible'):
                spec.pop(k, None)
        self.assertEqual(validate_taxonomy(legacy), [])


class TestMemberFieldsSpec(unittest.TestCase):
    """MEMBER_FIELDS is the one declaration every field reader derives from
    (validator, registry facts, write door, heal, classifier prompt and
    validation, decoder scan). Pin its shape so a half-declared field cannot
    reach any of them."""

    def test_spec_shape(self):
        from servers.aspect_store import MEMBER_FIELDS, MEMBER_KEY, REQUIRED_ASPECTS
        self.assertEqual(set(MEMBER_KEY), {'node_types', 'edge_relations'})
        self.assertIn('aspects', MEMBER_FIELDS)
        self.assertEqual(MEMBER_FIELDS['aspects']['kind'], 'menu')
        for name, spec in MEMBER_FIELDS.items():
            self.assertIn(spec['kind'], ('menu', 'enum'), name)
            self.assertTrue(spec['applies_to'], name)
            self.assertTrue(set(spec['applies_to']) <= set(MEMBER_KEY), name)
            if 'within' in spec:
                self.assertTrue(spec['within'], name)
                self.assertTrue(set(spec['within']) <= set(REQUIRED_ASPECTS), name)
            self.assertIsInstance(spec.get('curated', False), bool, name)
            self.assertTrue(len(spec['meaning']) > 20, name)
            if spec['kind'] == 'enum':
                self.assertTrue(spec['values'], name)
                self.assertTrue(spec.get('rule'), name)
                for val, meaning in spec['values'].items():
                    self.assertTrue(len(meaning) > 20, '%s.%s' % (name, val))

    def test_edge_fields_and_their_scope(self):
        from servers.aspect_store import stored_fields, filled_fields, field_values
        self.assertEqual(stored_fields('edge_relations'), ('orientation', 'flavor'))
        self.assertEqual(stored_fields('edge_relations', 'correction_improvement'),
                         ('orientation', 'flavor'))
        self.assertEqual(stored_fields('edge_relations', 'settlement'), ('orientation',))
        self.assertEqual(stored_fields('node_types'), ())
        self.assertEqual(set(field_values('orientation')),
                         {'source_actor', 'target_actor', 'symmetric'})
        self.assertEqual(set(field_values('flavor')),
                         {'replaced', 'corrected', 'resolved', 'reframed', 'unchanged'})
        self.assertEqual(field_values('aspects'), ())
        # the classifier fills orientation; flavor is curated by hand
        self.assertEqual(filled_fields('edge_relations'), ('orientation',))

    def test_curated_fields_stay_out_of_the_classifier_view(self):
        from servers.aspect_store import (CLASSIFIER_FIELDS, MEMBER_FIELDS,
                                          new_member_fill)
        self.assertNotIn('flavor', CLASSIFIER_FIELDS)
        self.assertEqual(set(CLASSIFIER_FIELDS),
                         {n for n, s in MEMBER_FIELDS.items() if not s.get('curated')})
        self.assertEqual(new_member_fill('edge_relations'), ['aspects', 'orientation'])

    def test_member_helpers(self):
        from servers.aspect_store import member_name, member_facts
        obj = {'verb': 'corrects', 'orientation': 'source_actor'}
        self.assertEqual(member_name(obj, 'edge_relations'), 'corrects')
        self.assertEqual(member_facts(obj, 'edge_relations'),
                         {'orientation': 'source_actor'})
        self.assertEqual(member_name('corrects', 'edge_relations'), 'corrects')
        self.assertEqual(member_facts('corrects', 'edge_relations'), {})
        # wrong key for the category, empty string, non-member → None
        self.assertIsNone(member_name({'type': 'corrects'}, 'edge_relations'))
        self.assertIsNone(member_name('', 'edge_relations'))
        self.assertIsNone(member_name(3, 'edge_relations'))
        # an undeclared key is not a fact
        self.assertEqual(member_facts({'verb': 'x', 'sign': '+'}, 'edge_relations'), {})


class TestMemberObjectValidation(unittest.TestCase):
    """The structural gate on member objects: shape, closed values, verb-keyed
    uniqueness, and one value per multi-homed verb across its homes."""

    def setUp(self):
        import copy
        self.seed = copy.deepcopy(_load_seed())

    def _orient(self, aspect, verb, value):
        rels = self.seed[aspect]['edge_relations']
        for i, m in enumerate(rels):
            if m == verb or (isinstance(m, dict) and m.get('verb') == verb):
                rels[i] = {'verb': verb, 'orientation': value}
                return
        raise AssertionError('%s not in %s' % (verb, aspect))

    def test_mixed_shapes_pass(self):
        from servers.aspect_store import validate_taxonomy
        self._orient('correction_improvement', 'corrects', 'source_actor')
        self._orient('correction_improvement', 'corrected_by', 'target_actor')
        self.assertEqual(validate_taxonomy(self.seed), [])

    def test_duplicate_is_keyed_on_verb_across_shapes(self):
        from servers.aspect_store import validate_taxonomy
        self.seed['correction_improvement']['edge_relations'].append(
            {'verb': 'corrects', 'orientation': 'source_actor'})
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any("duplicate member 'corrects'" in v for v in violations),
                        violations)

    def test_undeclared_key_refused(self):
        from servers.aspect_store import validate_taxonomy
        self.seed['correction_improvement']['edge_relations'].append(
            {'verb': 'new_rel', 'orientation': 'source_actor', 'sign': '+'})
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any('undeclared keys' in v and "'sign'" in v
                            for v in violations), violations)

    def test_unknown_value_refused(self):
        from servers.aspect_store import validate_taxonomy
        self._orient('correction_improvement', 'corrects', 'forwards')
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any("'corrects'.orientation = 'forwards'" in v
                            for v in violations), violations)

    def test_object_without_its_key_is_malformed(self):
        from servers.aspect_store import validate_taxonomy
        self.seed['noise']['edge_relations'].append({'orientation': 'symmetric'})
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any('malformed member' in v for v in violations), violations)

    def test_multi_homed_verb_must_agree(self):
        # supersedes lives in correction_improvement, hierarchical_structure,
        # settlement — one orientation across all three or the gate refuses.
        from servers.aspect_store import validate_taxonomy
        self._orient('correction_improvement', 'supersedes', 'source_actor')
        self._orient('settlement', 'supersedes', 'target_actor')
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any("'supersedes'.orientation disagrees" in v
                            for v in violations), violations)
        # agreement passes, and a bare copy in the third home is fine
        self._orient('settlement', 'supersedes', 'source_actor')
        self.assertEqual(validate_taxonomy(self.seed), [])

    def _set(self, aspect, verb, **facts):
        rels = self.seed[aspect]['edge_relations']
        for i, m in enumerate(rels):
            if m == verb or (isinstance(m, dict) and m.get('verb') == verb):
                obj = m if isinstance(m, dict) else {'verb': verb}
                obj.update(facts)
                rels[i] = obj
                return
        raise AssertionError('%s not in %s' % (verb, aspect))

    def test_scoped_field_lives_only_in_its_aspects(self):
        # flavor is `within` correction_improvement: accepted there, an
        # undeclared key on the same verb's home in another aspect.
        from servers.aspect_store import validate_taxonomy
        self._set('correction_improvement', 'supersedes', flavor='replaced')
        self.assertEqual(validate_taxonomy(self.seed), [])
        self._set('settlement', 'supersedes', flavor='replaced')
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any("aspect 'settlement'" in v and 'undeclared keys' in v
                            and "'flavor'" in v for v in violations), violations)

    def test_scoped_field_value_is_closed(self):
        from servers.aspect_store import validate_taxonomy
        self._set('correction_improvement', 'corrects', flavor='fixed')
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any("'corrects'.flavor = 'fixed'" in v
                            for v in violations), violations)

    def test_node_type_object_carries_no_fields_today(self):
        from servers.aspect_store import validate_taxonomy
        self.seed['lesson_insight']['node_types'].append({'type': 'new_type'})
        self.assertEqual(validate_taxonomy(self.seed), [])
        self.seed['lesson_insight']['node_types'].append(
            {'type': 'other_type', 'orientation': 'symmetric'})
        violations = validate_taxonomy(self.seed)
        self.assertTrue(any('undeclared keys' in v for v in violations), violations)


class TestSeedOrientation(unittest.TestCase):
    """Seed curation for the first member field. Ruling: no verb enters the
    taxonomy without an orientation — the classifier fills it in any brain
    that writes the verb, and the seed ships it for every verb the operator's
    graph had when the seed was oriented (2026-09-22). The bare list below is
    the closed set of seed verbs with no edge to judge from at that time; a
    new bare verb in the seed fails here and must be oriented or listed."""

    BARE_ALLOWED = {
        'adds_evidence_for', 'crowns', 'meets', 'necessitated',
        'provides_root_cause_for', 'relevant_to', 'requires_followup',
        'restates', 'should_feed_into',
        # noise: machinery, no dimension by design (the fill lane skips it)
        'co_anchored', 'community_member', 'dream_observation', 'dreamed_from',
    }

    def setUp(self):
        from servers.aspect_store import member_facts, member_name
        self.facts = {}
        self.bare = set()
        for name, entry in _load_seed().items():
            for m in entry.get('edge_relations', []) or []:
                v = member_name(m, 'edge_relations')
                f = member_facts(m, 'edge_relations')
                if f.get('orientation'):
                    self.facts.setdefault(v, f['orientation'])
                else:
                    self.bare.add(v)
        self.bare -= set(self.facts)

    def test_every_by_verb_is_target_actor(self):
        wrong = {v: o for v, o in self.facts.items() if v.endswith('_by') and o != 'target_actor'}
        self.assertEqual(wrong, {})

    def test_no_unlisted_bare_verb(self):
        self.assertEqual(self.bare - self.BARE_ALLOWED, set(),
                         'a seed verb entered without orientation — orient it or list it')
        self.assertEqual(self.BARE_ALLOWED - self.bare - set(self.facts), set(),
                         'BARE_ALLOWED names a verb the seed no longer has')

    def test_operator_rulings_2026_09_22(self):
        # Rulings on the classifier's run: `during` stays symmetric; `produced`
        # is source_actor by plain meaning (its examples were backwards rows);
        # the container acts (part_of → target_actor); the earlier event acts.
        self.assertEqual(self.facts['during'], 'symmetric')
        self.assertEqual(self.facts['produced'], 'source_actor')
        self.assertEqual(self.facts['part_of'], 'target_actor')
        self.assertEqual(self.facts['precedes'], 'source_actor')
        self.assertEqual(self.facts['followed'], 'target_actor')
        for v in ('corrects', 'supersedes', 'resolves', 'addresses'):
            self.assertEqual(self.facts[v], 'source_actor', v)
        for v in ('similar_to', 'related_to', 'contradicts'):
            self.assertEqual(self.facts[v], 'symmetric', v)

    def test_operator_rulings_2026_09_23(self):
        # The occasion acts, as for caught_during / found_during.
        self.assertEqual(self.facts['diagnosed_during'], 'target_actor')


if __name__ == '__main__':
    unittest.main()
