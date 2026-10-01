"""Tests for the Aspect value object + AspectRegistry skeleton.

Covers Step 2 scope: data shapes, from_dict construction, attribute access,
reverse lookups, cross-aspect unions, discovery/enumeration, surface helpers.

Also covers the write door (TestWriteDoorGate): add_members merge/refusal,
heal refusal, and degraded-but-loud boot on a malformed working copy.
"""

import unittest

from servers.aspects import (
    Aspect, AspectRegistry, AspectContractError, REQUIRED_ASPECTS,
)


# Sample data — mirrors aspects_v1.json shape (Step 3 will write the real file)
SAMPLE_DATA = {
    'identity_bearing': {
        'node_types': ['principle', 'identity', 'rule'],
        'edge_relations': [],
        'meaning': 'Nodes that anchor identity',
        'dimension': 'semantic',
        'locked': True,
        'metadata': {},
    },
    'correction_improvement': {
        'node_types': ['correction', 'bug_lesson'],
        'edge_relations': ['corrects', 'supersedes'],
        'meaning': 'Knowledge that updates prior knowledge',
        'dimension': 'semantic',
        'locked': True,
        'metadata': {'display_label': 'corrects/improves'},
    },
    'noise': {
        'node_types': [],
        'edge_relations': ['co_accessed', 'emergent_bridge'],
        'meaning': 'Structural edges, not semantic claims',
        'dimension': 'semantic',
        'locked': True,
        'metadata': {},
    },
    'emergent_xyz': {
        'node_types': ['hypothesis_v2'],
        'edge_relations': [],
        'meaning': 'An emergent aspect for testing',
        'dimension': 'semantic',
        'locked': False,
        'metadata': {},
    },
}


class TestAspect(unittest.TestCase):
    """The Aspect value object — frozen dataclass."""

    def test_construction_and_defaults(self):
        a = Aspect(name='x')
        self.assertEqual(a.name, 'x')
        self.assertEqual(a.node_types, ())
        self.assertEqual(a.edge_relations, ())
        self.assertEqual(a.meaning, '')
        self.assertEqual(a.dimension, 'semantic')
        self.assertFalse(a.locked)
        self.assertEqual(a.metadata, {})

    def test_frozen_blocks_field_reassignment(self):
        a = Aspect(name='x')
        with self.assertRaises(Exception):
            a.name = 'y'

    def test_contains(self):
        a = Aspect(name='x', node_types=('p', 'q'), edge_relations=('r',))
        self.assertIn('p', a)
        self.assertIn('r', a)
        self.assertNotIn('z', a)

    def test_member_count(self):
        a = Aspect(name='x', node_types=('p', 'q'), edge_relations=('r',))
        self.assertEqual(a.member_count, 3)


class TestAspectRegistryFromDict(unittest.TestCase):
    """AspectRegistry constructed via from_dict (the test/seed path)."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_contains_all_seeded(self):
        self.assertEqual(set(self.registry.all().keys()), set(SAMPLE_DATA.keys()))

    def test_attribute_access_required(self):
        a = self.registry.identity_bearing
        self.assertEqual(a.name, 'identity_bearing')
        self.assertIn('principle', a.node_types)

    def test_attribute_access_emergent_works_too(self):
        # __getattr__ doesn't distinguish required vs emergent — both resolve
        # if present. by_name() is the safer surface for emergent.
        a = self.registry.emergent_xyz
        self.assertEqual(a.name, 'emergent_xyz')

    def test_attribute_access_missing_raises(self):
        with self.assertRaises(AspectContractError):
            _ = self.registry.no_such_aspect

    def test_attribute_access_dunder_skipped(self):
        # Dunder names should raise AttributeError (not AspectContractError),
        # so Python internals (pickling, repr, etc.) work normally.
        with self.assertRaises(AttributeError):
            _ = self.registry.__some_dunder__

    def test_by_name_returns_none_for_missing(self):
        self.assertIsNone(self.registry.by_name('no_such_aspect'))
        # And returns the aspect for present names
        self.assertEqual(self.registry.by_name('identity_bearing').name,
                         'identity_bearing')


class TestAspectRegistryReverseLookups(unittest.TestCase):
    """by_node_type + by_edge_relation."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_by_node_type(self):
        a = self.registry.by_node_type('principle')
        self.assertEqual(a.name, 'identity_bearing')
        a2 = self.registry.by_node_type('correction')
        self.assertEqual(a2.name, 'correction_improvement')

    def test_by_node_type_missing(self):
        self.assertIsNone(self.registry.by_node_type('not_a_type'))

    def test_by_edge_relation(self):
        a = self.registry.by_edge_relation('corrects')
        self.assertEqual(a.name, 'correction_improvement')
        a2 = self.registry.by_edge_relation('co_accessed')
        self.assertEqual(a2.name, 'noise')

    def test_by_edge_relation_missing(self):
        self.assertIsNone(self.registry.by_edge_relation('not_a_relation'))


class TestAspectRegistryUnions(unittest.TestCase):
    """types_in + relations_in cross-aspect unions."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_types_in_single(self):
        result = self.registry.types_in(['identity_bearing'])
        self.assertEqual(set(result), {'principle', 'identity', 'rule'})

    def test_types_in_union_dedupes(self):
        # Add an aspect that overlaps to test dedup
        data_with_overlap = dict(SAMPLE_DATA)
        data_with_overlap['overlap_aspect'] = {
            'node_types': ['principle', 'new_type'],
            'edge_relations': [],
            'meaning': 'overlap test', 'dimension': 'semantic',
            'locked': False, 'metadata': {},
        }
        reg = AspectRegistry.from_dict(brain=None, data=data_with_overlap)
        result = reg.types_in(['identity_bearing', 'overlap_aspect'])
        # principle appears once despite being in both
        self.assertEqual(result.count('principle'), 1)
        self.assertIn('new_type', result)

    def test_relations_in(self):
        result = self.registry.relations_in(['correction_improvement', 'noise'])
        self.assertEqual(set(result),
                         {'corrects', 'supersedes', 'co_accessed', 'emergent_bridge'})

    def test_unions_skip_unknown_names(self):
        # Unknown aspect names are silently skipped, not errors
        result = self.registry.types_in(['identity_bearing', 'no_such_aspect'])
        self.assertEqual(set(result), {'principle', 'identity', 'rule'})


class TestAspectRegistryDiscovery(unittest.TestCase):
    """all() enumeration."""

    def setUp(self):
        self.registry = AspectRegistry.from_dict(brain=None, data=SAMPLE_DATA)

    def test_all(self):
        result = self.registry.all()
        self.assertEqual(set(result.keys()), set(SAMPLE_DATA.keys()))

    def test_all_returns_fresh_dict(self):
        # Mutating the returned dict shouldn't affect the registry
        result = self.registry.all()
        result['injected'] = 'should_not_persist'
        result2 = self.registry.all()
        self.assertNotIn('injected', result2)


class TestRequiredAspectsConstant(unittest.TestCase):
    """REQUIRED_ASPECTS is the contract surface."""

    def test_count(self):
        # 5 node-facing + 12 edge-facing = 17
        self.assertEqual(len(REQUIRED_ASPECTS), 17)

    def test_no_duplicates(self):
        self.assertEqual(len(set(REQUIRED_ASPECTS)), len(REQUIRED_ASPECTS))

    def test_immutable(self):
        # tuple, not list — can't be mutated
        self.assertIsInstance(REQUIRED_ASPECTS, tuple)


class _StubBrain:
    """Log-capturing stand-in for the registry's brain dependency."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def _log_error(self, category, exc, detail=''):
        self.errors.append((category, str(exc), detail))

    def _log_warning(self, category, message, detail=''):
        self.warnings.append((category, message, detail))


class _WorkingCopyCase(unittest.TestCase):
    """A registry over a temp working copy of the seed, with a log-capturing
    brain. No tests of its own, so door classes can share it."""

    def setUp(self):
        import json
        import os
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self._orig_env = os.environ.get('ASPECTS_JSON_PATH')
        self.work = os.path.join(self.tmp, 'aspects_v1.json')
        os.environ['ASPECTS_JSON_PATH'] = self.work
        from servers.aspect_store import SEED_ASPECTS_JSON_PATH
        with open(SEED_ASPECTS_JSON_PATH) as f:
            self.seed = json.load(f)
        with open(self.work, 'w') as f:
            json.dump(self.seed, f)
        self.brain = _StubBrain()
        self.registry = AspectRegistry(self.brain)

    def tearDown(self):
        import os
        import shutil
        if self._orig_env is None:
            os.environ.pop('ASPECTS_JSON_PATH', None)
        else:
            os.environ['ASPECTS_JSON_PATH'] = self._orig_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _file_bytes(self):
        with open(self.work, 'rb') as f:
            return f.read()


class TestWriteDoorGate(_WorkingCopyCase):
    """The registry's write door: add_members merges + reloads, REFUSES a
    structurally invalid result (file untouched), and a heal whose result
    would break the invariants is refused. Boot with a malformed working
    copy stays alive but logs every violation."""

    def test_add_members_writes_and_registry_sees_it_immediately(self):
        n = self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'door_test_rel',
              'aspects': ['correction_improvement']}],
            source='test')
        self.assertEqual(n, 1)
        # in-memory view refreshed without any invalidation call
        self.assertIn('door_test_rel',
                      self.registry.correction_improvement.edge_relations)
        # and it's on disk
        import json
        with open(self.work) as f:
            data = json.load(f)
        self.assertIn('door_test_rel',
                      data['correction_improvement']['edge_relations'])

    def test_add_members_accepts_noise_overlap_and_noise_then_vetoes(self):
        """Noise vetoes rather than being exclusive, so the
        door no longer refuses a semantic verb being filed as machinery. The
        veto is what makes this safe: the string keeps its semantic membership
        but resolves to noise and lands in the exclusion sets."""
        self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'corrects',
              'aspects': ['noise']}],
            source='test')
        self.assertIn('corrects', self.registry.noise.edge_relations)
        self.assertIn('corrects', self.registry.by_name(
            'correction_improvement').edge_relations)
        self.assertEqual(self.registry.by_edge_relation('corrects').name, 'noise')
        self.assertIn('corrects', self.registry.structural_exclusions)

    def test_add_members_refuses_unknown_aspect_name(self):
        # A typo'd aspect name must fail loud, never setdefault-create a
        # meaning-less husk aspect. New aspects are a human edit to the JSON.
        before = self._file_bytes()
        with self.assertRaises(AspectContractError) as ctx:
            self.registry.add_members(
                [{'category': 'edge_relations', 'value': 'some_rel',
                  'aspects': ['correction_improvment']}],   # sic — typo
                source='test')
        self.assertIn('correction_improvment', str(ctx.exception))
        self.assertEqual(self._file_bytes(), before)
        self.assertNotIn('correction_improvment', self.registry.all())

    def test_heal_lands_when_seed_and_working_copy_disagree(self):
        """The fleet gap, as a regression test.

        A working copy where the classifier filed a string one way and the
        seed files it another used to make the heal refuse the WHOLE reconcile
        — every divergent install froze on its old taxonomy, silently, with no
        migration lane to reach it (aspects have no version counter). Noise
        vetoing instead of excluding is what lets the additive heal land.
        """
        import json
        import os
        import servers.aspects as aspects_mod
        # Working copy grows a classifier-judged noise member...
        with open(self.work) as f:
            cur = json.load(f)
        cur['noise']['edge_relations'].append('contested_rel')
        with open(self.work, 'w') as f:
            json.dump(cur, f)
        # ...and a (temp) seed later files the same string as semantic.
        bad_seed = {k: {kk: (list(vv) if isinstance(vv, list) else vv)
                        for kk, vv in v.items()} for k, v in self.seed.items()}
        bad_seed['correction_improvement']['edge_relations'].append('contested_rel')
        seed_path = os.path.join(self.tmp, 'seed.json')
        with open(seed_path, 'w') as f:
            json.dump(bad_seed, f)
        before = self._file_bytes()
        orig = aspects_mod.SEED_ASPECTS_JSON_PATH
        aspects_mod.SEED_ASPECTS_JSON_PATH = seed_path
        try:
            logged = []
            changed = aspects_mod.reconcile_working_copy(log_fn=logged.append)
        finally:
            aspects_mod.SEED_ASPECTS_JSON_PATH = orig
        self.assertTrue(changed)                        # heal landed
        self.assertNotEqual(self._file_bytes(), before)  # file advanced
        self.assertFalse([m for m in logged if 'heal REFUSED' in m], logged)
        # The seed's semantic claim arrived, the working copy's noise call
        # survived, and noise wins for anything that filters.
        with open(self.work) as f:
            healed = json.load(f)
        self.assertIn('contested_rel',
                      healed['correction_improvement']['edge_relations'])
        self.assertIn('contested_rel', healed['noise']['edge_relations'])
        reloaded = aspects_mod.AspectRegistry.from_dict(None, healed)
        self.assertEqual(
            reloaded.by_edge_relation('contested_rel').name, 'noise')

    def test_reconcile_heals_missing_fact_fields(self):
        # Job 4: a pre-Step-4 working copy (no accepts/routable/... anywhere)
        # receives the seed's per-aspect facts on reconcile. Without this the
        # registry — which loads the WORKING copy — would run every derived
        # consumer (classifier routing, prompt visibility) on defaults while
        # seed-based tests pass.
        import json
        import servers.aspects as aspects_mod
        from servers.aspects import ASPECT_FACT_KEYS
        with open(self.work) as f:
            cur = json.load(f)
        for spec in cur.values():
            for k in ASPECT_FACT_KEYS:
                spec.pop(k, None)
        # A deliberately divergent PRESENT value must survive the heal.
        cur['temporal_sequence']['prompt_visible'] = False
        with open(self.work, 'w') as f:
            json.dump(cur, f)
        logged = []
        changed = aspects_mod.reconcile_working_copy(log_fn=logged.append)
        self.assertTrue(changed)
        with open(self.work) as f:
            healed = json.load(f)
        for name, spec in healed.items():
            if name.startswith('_'):
                continue                     # _schema docs, not an aspect
            for k in ASPECT_FACT_KEYS:
                self.assertIn(k, spec, '%s.%s not healed' % (name, k))
        # facts match the seed...
        self.assertEqual(healed['survivor_lineage']['routable'], False)
        self.assertEqual(healed['dependency_flow']['prompt_visible'], True)
        # ...except the present value, which was preserved, not overwritten
        self.assertEqual(healed['temporal_sequence']['prompt_visible'], False)
        self.assertTrue(any('facts +=' in m for m in logged), logged)
        # idempotent: second run finds nothing to heal
        self.assertFalse(aspects_mod.reconcile_working_copy())

    def test_malformed_working_copy_boots_degraded_and_loud(self):
        import json
        with open(self.work) as f:
            cur = json.load(f)
        cur['junk_entry'] = 'not-an-object'
        with open(self.work, 'w') as f:
            json.dump(cur, f)
        brain = _StubBrain()
        registry = AspectRegistry(brain)               # must not raise
        # degraded: the malformed entry is skipped, the rest loads
        self.assertIn('correction_improvement', registry.all())
        self.assertNotIn('junk_entry', registry.all())
        # loud: the violation reached the error log
        self.assertTrue(
            any('junk_entry' in e[2] for e in brain.errors
                if e[0] == 'aspect_contract'),
            brain.errors)


class TestMemberFacts(unittest.TestCase):
    """Member objects load into the same string tuples every consumer reads,
    plus relation_facts / fact_of for the consumers that want a dimension."""

    def setUp(self):
        import copy
        data = copy.deepcopy(SAMPLE_DATA)
        data['correction_improvement']['edge_relations'] = [
            {'verb': 'corrects', 'orientation': 'source_actor'},
            'supersedes',
            {'verb': 'corrected_by', 'orientation': 'target_actor'},
        ]
        data['noise']['edge_relations'].append(
            {'verb': 'supersedes', 'orientation': 'source_actor'})
        self.registry = AspectRegistry.from_dict(None, data)

    def test_tuples_stay_strings(self):
        ci = self.registry.correction_improvement
        self.assertEqual(ci.edge_relations, ('corrects', 'supersedes', 'corrected_by'))
        self.assertIn('corrects', ci)
        self.assertEqual(self.registry.by_edge_relation('corrected_by').name,
                         'correction_improvement')
        self.assertIn('corrected_by', self.registry.relations_in(['correction_improvement']))

    def test_relation_facts_per_aspect_and_merged(self):
        ci = self.registry.correction_improvement
        self.assertEqual(ci.relation_facts, {
            'corrects': {'orientation': 'source_actor'},
            'corrected_by': {'orientation': 'target_actor'}})
        self.assertEqual(self.registry.fact_of('corrects', 'orientation'), 'source_actor')
        self.assertEqual(self.registry.fact_of('corrected_by', 'orientation'), 'target_actor')
        # bare in one home, object in another → the object's value is the fact
        self.assertEqual(self.registry.fact_of('supersedes', 'orientation'), 'source_actor')
        self.assertIsNone(self.registry.fact_of('co_accessed', 'orientation'))
        self.assertIsNone(self.registry.fact_of('never_seen', 'orientation'))
        self.assertEqual(set(self.registry.relation_facts()),
                         {'corrects', 'corrected_by', 'supersedes'})

    def test_actor_of_reads_orientation_against_the_claims_ends(self):
        import copy
        data = copy.deepcopy(SAMPLE_DATA)
        data['correction_improvement']['edge_relations'] = [
            {'verb': 'corrects', 'orientation': 'source_actor'},
            {'verb': 'corrected_by', 'orientation': 'target_actor'},
            {'verb': 'contradicts', 'orientation': 'symmetric'},
            'unoriented_yet',
        ]
        reg = AspectRegistry.from_dict(None, data)
        self.assertEqual(reg.actor_of('corrects', 'a', 'b'), 'a')
        self.assertEqual(reg.actor_of('corrected_by', 'a', 'b'), 'b')
        self.assertIsNone(reg.actor_of('contradicts', 'a', 'b'))
        self.assertIsNone(reg.actor_of('unoriented_yet', 'a', 'b'))
        self.assertIsNone(reg.actor_of('never_seen', 'a', 'b'))

    def test_relation_facts_returns_a_copy(self):
        facts = self.registry.relation_facts()
        facts['corrects']['orientation'] = 'symmetric'
        self.assertEqual(self.registry.fact_of('corrects', 'orientation'), 'source_actor')


class TestWriteDoorFields(TestWriteDoorGate):
    """add_members with MEMBER_FIELDS values: a new verb is written as an
    object; an existing verb is upgraded in EVERY home at once; a clashing
    present value is kept and logged. Runs on a pre-orientation working copy
    (bare strings) so the door has fields to write."""

    @staticmethod
    def _bare(data):
        import json
        out = json.loads(json.dumps(data))
        for name, entry in out.items():
            if name.startswith('_') or not isinstance(entry, dict):
                continue
            entry['edge_relations'] = [m['verb'] if isinstance(m, dict) else m
                                       for m in entry.get('edge_relations', []) or []]
        return out

    def setUp(self):
        import json
        super().setUp()
        self.seed = self._bare(self.seed)          # the tests' seed is bare too
        with open(self.work, 'w') as f:
            json.dump(self.seed, f)
        self.registry._load()
        assert self.registry.fact_of('corrects', 'orientation') is None

    def _file(self):
        import json
        with open(self.work) as f:
            return json.load(f)

    def test_new_verb_with_field_is_written_as_object(self):
        n = self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'door_oriented_rel',
              'aspects': ['correction_improvement'],
              'orientation': 'target_actor'}],
            source='test')
        self.assertEqual(n, 1)
        self.assertIn('door_oriented_rel',
                      self.registry.correction_improvement.edge_relations)
        self.assertEqual(self.registry.fact_of('door_oriented_rel', 'orientation'),
                         'target_actor')
        self.assertIn({'verb': 'door_oriented_rel', 'orientation': 'target_actor'},
                      self._file()['correction_improvement']['edge_relations'])

    def test_fill_upgrades_existing_verb_in_every_home(self):
        # supersedes is bare in three homes; the fill lists only one of them.
        homes = [n for n, a in self.registry.all().items()
                 if 'supersedes' in a.edge_relations]
        self.assertEqual(len(homes), 3, homes)
        n = self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'supersedes',
              'aspects': ['correction_improvement'],
              'orientation': 'source_actor'}],
            source='test')
        self.assertEqual(n, 3)                       # 0 memberships, 3 fills
        data = self._file()
        for h in homes:
            self.assertIn({'verb': 'supersedes', 'orientation': 'source_actor'},
                          data[h]['edge_relations'], h)
            self.assertNotIn('supersedes', data[h]['edge_relations'], h)
            self.assertIn('supersedes', self.registry.by_name(h).edge_relations)
        self.assertEqual(self.registry.fact_of('supersedes', 'orientation'), 'source_actor')
        # idempotent
        self.assertEqual(self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'supersedes',
              'aspects': ['correction_improvement'],
              'orientation': 'source_actor'}], source='test'), 0)

    def test_present_value_is_kept_and_clash_logged(self):
        self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'corrects',
              'aspects': ['correction_improvement'],
              'orientation': 'source_actor'}], source='test')
        before = self._file_bytes()
        n = self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'corrects',
              'aspects': ['correction_improvement'],
              'orientation': 'target_actor'}], source='test')
        self.assertEqual(n, 0)
        self.assertEqual(self._file_bytes(), before)
        self.assertEqual(self.registry.fact_of('corrects', 'orientation'), 'source_actor')
        self.assertTrue(any('field disagreement' in w[1] for w in self.brain.warnings),
                        self.brain.warnings)

    def test_unknown_field_value_is_refused_at_the_door(self):
        before = self._file_bytes()
        with self.assertRaises(AspectContractError):
            self.registry.add_members(
                [{'category': 'edge_relations', 'value': 'door_bad_rel',
                  'aspects': ['correction_improvement'],
                  'orientation': 'sideways'}], source='test')
        self.assertEqual(self._file_bytes(), before)

    def test_reconcile_fills_member_fields_from_seed(self):
        # Job 5: a seed member carries an orientation the working copy's bare
        # copies lack → filled into EVERY home at once, no duplicate, heal
        # not refused. A value the working copy already holds anywhere for
        # the member — agreeing with the seed or not — wins: that member is
        # left alone, so the heal never manufactures a cross-home clash.
        import json
        import os
        import servers.aspects as aspects_mod
        seed = json.loads(json.dumps(self.seed))
        rels = seed['correction_improvement']['edge_relations']
        rels[rels.index('corrected_by')] = {'verb': 'corrected_by',
                                            'orientation': 'target_actor'}
        rels[rels.index('supersedes')] = {'verb': 'supersedes',
                                          'orientation': 'source_actor'}
        rels[rels.index('resolves')] = {'verb': 'resolves',
                                        'orientation': 'source_actor'}
        # the working copy already values supersedes — divergently — in one
        # of its three homes
        cur = self._file()
        wrels = cur['settlement']['edge_relations']
        wrels[wrels.index('supersedes')] = {'verb': 'supersedes',
                                            'orientation': 'target_actor'}
        with open(self.work, 'w') as f:
            json.dump(cur, f)
        seed_path = os.path.join(self.tmp, 'seed.json')
        with open(seed_path, 'w') as f:
            json.dump(seed, f)
        orig = aspects_mod.SEED_ASPECTS_JSON_PATH
        aspects_mod.SEED_ASPECTS_JSON_PATH = seed_path
        try:
            logged = []
            changed = aspects_mod.reconcile_working_copy(log_fn=logged.append)
            self.assertTrue(changed, logged)
            self.assertFalse([m for m in logged if 'REFUSED' in m], logged)
            healed = self._file()
            names = lambda ms: [m['verb'] if isinstance(m, dict) else m for m in ms]
            # corrected_by: bare in two homes → object in both, still one member each
            for h in ('correction_improvement', 'settlement'):
                self.assertIn({'verb': 'corrected_by', 'orientation': 'target_actor'},
                              healed[h]['edge_relations'], h)
                self.assertEqual(names(healed[h]['edge_relations']).count('corrected_by'), 1, h)
            # resolves: bare in two homes → filled in both
            for h in ('correction_improvement', 'settlement'):
                self.assertIn({'verb': 'resolves', 'orientation': 'source_actor'},
                              healed[h]['edge_relations'], h)
            # supersedes: the working copy had a value → untouched everywhere
            self.assertIn('supersedes', healed['correction_improvement']['edge_relations'])
            self.assertIn('supersedes', healed['hierarchical_structure']['edge_relations'])
            self.assertIn({'verb': 'supersedes', 'orientation': 'target_actor'},
                          healed['settlement']['edge_relations'])
            self.assertTrue(any('corrected_by += orientation=target_actor' in m
                                for m in logged), logged)
            # the registry reads it, and a second run finds nothing
            reg = aspects_mod.AspectRegistry.from_dict(None, healed)
            self.assertEqual(reg.fact_of('corrected_by', 'orientation'), 'target_actor')
            self.assertEqual(reg.fact_of('supersedes', 'orientation'), 'target_actor')
            self.assertFalse(aspects_mod.reconcile_working_copy())
        finally:
            aspects_mod.SEED_ASPECTS_JSON_PATH = orig

    def test_reconcile_fills_a_scoped_field_only_in_its_homes(self):
        # flavor is `within` correction_improvement. The seed carries it on
        # supersedes' correction home and orientation on its settlement home:
        # the heal reads both seed homes, fills orientation everywhere and
        # flavor in the correction home only — anywhere else is the violation
        # that would refuse the whole heal.
        import json
        import os
        import servers.aspects as aspects_mod
        seed = json.loads(json.dumps(self.seed))
        rels = seed['correction_improvement']['edge_relations']
        rels[rels.index('supersedes')] = {'verb': 'supersedes', 'flavor': 'replaced'}
        srels = seed['settlement']['edge_relations']
        srels[srels.index('supersedes')] = {'verb': 'supersedes',
                                            'orientation': 'source_actor'}
        seed_path = os.path.join(self.tmp, 'seed.json')
        with open(seed_path, 'w') as f:
            json.dump(seed, f)
        orig = aspects_mod.SEED_ASPECTS_JSON_PATH
        aspects_mod.SEED_ASPECTS_JSON_PATH = seed_path
        try:
            logged = []
            self.assertTrue(aspects_mod.reconcile_working_copy(log_fn=logged.append), logged)
            self.assertFalse([m for m in logged if 'REFUSED' in m], logged)
            healed = self._file()
            self.assertIn({'verb': 'supersedes', 'orientation': 'source_actor',
                           'flavor': 'replaced'},
                          healed['correction_improvement']['edge_relations'])
            for h in ('hierarchical_structure', 'settlement'):
                self.assertIn({'verb': 'supersedes', 'orientation': 'source_actor'},
                              healed[h]['edge_relations'], h)
            # the announcement names only the homes each field reached
            segs = [s.split(': ', 1)[-1] for m in logged for s in m.split('; ')]
            self.assertIn('edge_relations supersedes += flavor=replaced,'
                          'orientation=source_actor in correction_improvement', segs)
            self.assertIn('edge_relations supersedes += orientation=source_actor '
                          'in hierarchical_structure,settlement', segs)
            reg = aspects_mod.AspectRegistry.from_dict(None, healed)
            self.assertEqual(reg.fact_of('supersedes', 'flavor'), 'replaced')
            self.assertFalse(aspects_mod.reconcile_working_copy())
        finally:
            aspects_mod.SEED_ASPECTS_JSON_PATH = orig

    def test_scoped_field_rides_only_to_its_homes(self):
        n = self.registry.add_members(
            [{'category': 'edge_relations', 'value': 'supersedes',
              'aspects': ['correction_improvement'],
              'orientation': 'source_actor', 'flavor': 'replaced'}],
            source='test')
        self.assertEqual(n, 4)                       # orientation ×3 + flavor ×1
        data = self._file()
        self.assertIn({'verb': 'supersedes', 'orientation': 'source_actor',
                       'flavor': 'replaced'},
                      data['correction_improvement']['edge_relations'])
        for h in ('hierarchical_structure', 'settlement'):
            self.assertIn({'verb': 'supersedes', 'orientation': 'source_actor'},
                          data[h]['edge_relations'], h)
        self.assertEqual(self.registry.fact_of('supersedes', 'flavor'), 'replaced')


class TestCurationDoor(_WorkingCopyCase):
    """clear_fact: the reviewed correction add_members cannot make (it keeps a
    present value by design) — clears the value from EVERY home of the verb so
    the fill lane re-derives it, logs the source with each home's prior value,
    and refuses what the gate refuses."""

    def _file(self):
        import json
        with open(self.work) as f:
            return json.load(f)

    def test_clears_value_in_every_home(self):
        # supersedes: source_actor in three seed homes; one made bare, so only
        # two homes hold a value to clear.
        import json
        cur = self._file()
        rels = cur['settlement']['edge_relations']
        i = next(i for i, m in enumerate(rels)
                 if isinstance(m, dict) and m['verb'] == 'supersedes')
        rels[i] = 'supersedes'
        with open(self.work, 'w') as f:
            json.dump(cur, f)
        self.registry._load()
        homes = self.registry.homes_of('supersedes', 'edge_relations')
        self.assertEqual(len(homes), 3, homes)

        prior = self.registry.clear_fact('supersedes', 'orientation',
                                         source='review:test')
        self.assertEqual(prior, {'correction_improvement': 'source_actor',
                                 'hierarchical_structure': 'source_actor'})
        data = self._file()
        # the correction home keeps its curated flavor — clearing one field
        # leaves the others; a home with nothing else left is bare again
        self.assertIn({'verb': 'supersedes', 'flavor': 'replaced'},
                      data['correction_improvement']['edge_relations'])
        for h in homes:
            if h != 'correction_improvement':
                self.assertIn('supersedes', data[h]['edge_relations'], h)
        self.assertEqual(homes, self.registry.homes_of('supersedes', 'edge_relations'))
        self.assertIsNone(self.registry.fact_of('supersedes', 'orientation'))
        self.assertEqual(self.registry.fact_of('supersedes', 'flavor'), 'replaced')
        logged = [w[2] for w in self.brain.warnings if w[1] == 'fact cleared by curation']
        self.assertEqual(len(logged), 1, self.brain.warnings)
        self.assertIn('clear_fact(review:test)', logged[0])
        self.assertIn('source_actor in correction_improvement', logged[0])

        # idempotent: nothing left to clear → no write, no log
        before = self._file_bytes()
        self.assertEqual(self.registry.clear_fact(
            'supersedes', 'orientation', source='review:test'), {})
        self.assertEqual(self._file_bytes(), before)

    def test_refusals_leave_the_file_untouched(self):
        cases = {
            'unknown verb': ('never_listed_rel', 'orientation'),
            'field not stored on the member': ('supersedes', 'aspects'),
        }
        before = self._file_bytes()
        for label, args in cases.items():
            with self.subTest(label):
                with self.assertRaises(AspectContractError):
                    self.registry.clear_fact(*args, source='review:test')
                self.assertEqual(self._file_bytes(), before)
        # a running S2 cycle would overwrite the clear with its next batch
        self.brain.s2_running = True
        with self.assertRaises(AspectContractError):
            self.registry.clear_fact('supersedes', 'orientation', source='review:test')
        self.assertEqual(self._file_bytes(), before)
        self.assertEqual(self.registry.fact_of('supersedes', 'orientation'),
                         'source_actor')

    def test_undeclared_key_is_kept_for_the_validator_not_dropped(self):
        # A hand-edited member with a key MEMBER_FIELDS does not declare: the
        # clear must not flatten it to a bare string (silently deleting the
        # key) — it stays, and the gate refuses the whole write, loudly.
        import json
        cur = self._file()
        rels = cur['correction_improvement']['edge_relations']
        i = next(i for i, m in enumerate(rels)
                 if isinstance(m, dict) and m['verb'] == 'supersedes')
        rels[i] = dict(rels[i], note='hand-added')
        with open(self.work, 'w') as f:
            json.dump(cur, f)
        before = self._file_bytes()
        with self.assertRaises(AspectContractError) as ctx:
            self.registry.clear_fact('supersedes', 'orientation', source='review:test')
        self.assertIn('undeclared keys', str(ctx.exception))
        self.assertEqual(self._file_bytes(), before)


if __name__ == '__main__':
    unittest.main()
