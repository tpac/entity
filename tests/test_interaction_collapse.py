"""Tests for servers/interaction_collapse.py — the one-time install collapse.

The collapse's success signal is forgeable: "no override pointers, every name
on its code default" is also what a brain that never held overrides reads like.
So these tests never *inherit* a before-state — they CONSTRUCT one, covering
every verdict, and assert against state they created.

That is not fussiness. Inheriting production's pointers makes the suite
single-use: the first real daemon run destroys the state the assertions read,
and the file goes red-or-vacuous forever with no fixture able to restore it.
Constructing also lets each verdict be exercised on this install, where
production happens to carry no diverged COMPARE name at all.

Run: ./dev pytest tests/test_interaction_collapse.py -v
"""
import json

import pytest

from servers.interaction_collapse import (ADOPT, AUDIT_KEY, BACKUP_TAG,
                                          COLLAPSE_POLICY,
                                          COLLAPSE_VERSION_KEY, COMPARE, PIN,
                                          RETIRE, SKIP, _collapse_overrides,
                                          collapse_seeded_overrides)
from servers.interaction_defaults import (INTERACTION_DEFAULTS,
                                          interaction_fingerprint)
from tests.isolated_brain import brain_env

# One name per verdict, chosen so every branch of the policy loop is exercised.
MATCHING_COMPARE = 's1e'           # deployed byte-identical to the default
DIVERGED_COMPARE = 's2_healer'     # a real local override — must survive
ADOPT_NAME = 's2_community'
PIN_NAME = 'trace_recording'
SKIP_NAME = 'recall_laf'
RETIRE_NAME = 'boot'               # no code default at all
UNKNOWN_NAME = 'zz_collapse_probe_unknown'

PROBE_KEY = '_collapse_probe'      # marks a config as deliberately diverged


def _reset(brain):
    """Clear every pointer and any prior collapse bookkeeping.

    Lets the file keep working on an install that has already collapsed —
    otherwise the copied `logs_meta` stamp short-circuits the entry point and
    the copied audit blocks the first-write-wins branch.
    """
    brain.logs_conn.execute('DELETE FROM interaction_active')
    brain.logs_conn.execute('DELETE FROM logs_meta WHERE key IN (?, ?)',
                            (COLLAPSE_VERSION_KEY, AUDIT_KEY))
    brain.logs_conn.commit()


def _deploy(brain, name, template='', config=None, set_by='anchor'):
    """Register a version and make it the active override. Returns version."""
    row = brain.register_interaction(
        name, template=template, parameters=json.dumps(config or {}),
        created_by=set_by)
    brain.set_interaction_active(name, row['version'], set_by=set_by)
    return row['version']


def _deploy_default(brain, name, set_by='anchor'):
    """Deploy an override byte-identical to the code default."""
    template, config = INTERACTION_DEFAULTS[name]
    return _deploy(brain, name, template, config, set_by)


def _deploy_diverged(brain, name, set_by='anchor'):
    """Deploy an override that genuinely differs from the code default."""
    template, config = INTERACTION_DEFAULTS[name]
    return _deploy(brain, name, template, dict(config, **{PROBE_KEY: 1}),
                   set_by)


def _build_before_state(brain):
    """One pointer per verdict. Returns {name: version}."""
    _reset(brain)
    deployed = {
        MATCHING_COMPARE: _deploy_default(brain, MATCHING_COMPARE),
        DIVERGED_COMPARE: _deploy_diverged(brain, DIVERGED_COMPARE),
        ADOPT_NAME: _deploy_diverged(brain, ADOPT_NAME),
        # PIN deployed MATCHING on purpose: then dropping it would move no
        # effective value, so the drift check stays silent and the PIN verdict
        # is the only thing protecting it. That makes the test load-bearing.
        PIN_NAME: _deploy_default(brain, PIN_NAME),
        SKIP_NAME: _deploy_diverged(brain, SKIP_NAME),
        RETIRE_NAME: _deploy(brain, RETIRE_NAME, '', {'retired_key': 1}),
        UNKNOWN_NAME: _deploy(brain, UNKNOWN_NAME, '', {'local': 1}),
    }
    return deployed


def _pointers(brain):
    return {i['name']: i['active_version']
            for i in brain.list_interactions()
            if i.get('active_version') is not None}


def _fp(brain, name):
    return brain.get_interaction_stamp(name)['fingerprint']


def _default_fp(name):
    return interaction_fingerprint(name, *INTERACTION_DEFAULTS[name])


class TestPolicyTableShape:
    """The table must not drift from the defaults registry. RETIRE means
    'this name has no code default at all', so the two are one fact stated
    twice — and a name that gains or loses a default must move buckets."""

    def test_retire_is_exactly_the_names_without_a_code_default(self):
        retired = {n for n, v in COLLAPSE_POLICY.items() if v == RETIRE}
        with_default = {n for n in COLLAPSE_POLICY
                        if n in INTERACTION_DEFAULTS}
        assert retired == set(COLLAPSE_POLICY) - with_default, (
            "RETIRE must be exactly the policy names with no code default; "
            "retired=%s without_default=%s"
            % (sorted(retired), sorted(set(COLLAPSE_POLICY) - with_default)))

    def test_every_registry_name_has_a_policy_entry(self):
        missing = set(INTERACTION_DEFAULTS) - set(COLLAPSE_POLICY)
        assert not missing, (
            "registry names with no collapse verdict (they would be left "
            "frozen as 'unknown' forever): %s" % sorted(missing))

    def test_verdicts_are_from_the_closed_set(self):
        allowed = {COMPARE, ADOPT, PIN, SKIP, RETIRE}
        assert set(COLLAPSE_POLICY.values()) <= allowed

    def test_no_runtime_reader_names_a_retired_interaction(self):
        """RETIRE is the one bucket the drift check cannot cover: those names
        have no code default, so they are absent from the effective-value
        snapshot and their pointers are dropped unverified. That is only safe
        while nothing reads them — which is the property this holds."""
        import os
        import re
        root = os.path.join(os.path.dirname(__file__), '..', 'servers')
        retired = sorted(n for n, v in COLLAPSE_POLICY.items() if v == RETIRE)
        pattern = re.compile(
            r"get_interaction(?:_prompt|_config|_stamp)?\(\s*['\"](%s)['\"]"
            % '|'.join(retired))
        offenders = []
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding='utf-8') as f:
                    for hit in pattern.findall(f.read()):
                        offenders.append('%s reads %s' % (fn, hit))
        assert not offenders, (
            "a RETIRE name gained a reader — its pointer is dropped with no "
            "effective-value check: %s" % offenders)

    def test_probe_names_carry_the_verdicts_these_tests_assume(self):
        """If a name is re-bucketed, the behavioural tests below would quietly
        start asserting the wrong branch."""
        for name, verdict in ((MATCHING_COMPARE, COMPARE),
                              (DIVERGED_COMPARE, COMPARE),
                              (ADOPT_NAME, ADOPT), (PIN_NAME, PIN),
                              (SKIP_NAME, SKIP), (RETIRE_NAME, RETIRE)):
            assert COLLAPSE_POLICY[name] == verdict, name
        assert UNKNOWN_NAME not in COLLAPSE_POLICY


class TestDaemonOnly:
    """Ruled: the collapse must never fire from Brain.__init__, or a frozen
    eval corpus collapses and then floats with future code edits."""

    def test_constructing_a_brain_does_not_collapse(self):
        """State-independent: the production copy arrives STAMPED (the
        collapse ran live 2026-08-23), so asserting stamp==0 on the copy
        reads inherited state, not Brain behavior — the single-use-fixture
        trap. Un-stamp the copy, reopen a Brain on it, and assert the stamp
        is STILL absent: construction must never run the collapse."""
        from servers.brain import Brain
        from servers.schema import read_schema_version
        with brain_env(load_env=False) as env:
            env.brain.logs_conn.execute(
                'DELETE FROM logs_meta WHERE key = ?', (COLLAPSE_VERSION_KEY,))
            env.brain.logs_conn.commit()
            env.brain.save()
            env.brain.close()

            env.brain = Brain(env.brain_db)
            # The stamp, not the pointer count: a successfully collapsed brain
            # still carries PIN+SKIP pointers, so "pointers exist" cannot tell
            # "never collapsed" from "collapsed".
            assert read_schema_version(env.brain.logs_conn, 'logs_meta',
                                       COLLAPSE_VERSION_KEY) == 0, \
                "Brain.__init__ ran the collapse — frozen corpora would float"

    def test_daemon_load_brain_calls_the_collapse(self):
        """The wiring itself, at BODY level — the file-scan below stays green
        if someone deletes the call and leaves the import, and the collapse
        then ships dead on every uncollapsed install."""
        import inspect
        from servers import daemon_server
        src = inspect.getsource(daemon_server.BrainDaemon._load_brain)
        assert 'collapse_seeded_overrides(self.brain)' in src

    def test_only_the_daemon_reaches_the_collapse(self):
        import os
        root = os.path.join(os.path.dirname(__file__), '..', 'servers')
        hits = set()
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding='utf-8') as f:
                    if 'interaction_collapse' in f.read():
                        hits.add(fn)
        assert hits == {'interaction_collapse.py', 'daemon_server.py'}, (
            "only the daemon may reach the collapse; found: %s" % sorted(hits))


class TestCollapseBehavior:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env(load_env=False) as env:
            self.brain = env.brain
            self.deployed = _build_before_state(env.brain)
            self.before_fp = {n: _fp(env.brain, n)
                              for n in INTERACTION_DEFAULTS}
            yield

    def test_before_state_is_genuinely_collapsible(self):
        """The guard that makes every other assertion here mean something.

        Not "some pointer exists" — a fully collapsed brain satisfies that.
        A COMPARE name must be deployed AND stamping override, which is
        exactly what the collapse is supposed to undo."""
        from servers.schema import read_meta_value, read_schema_version
        pointers = _pointers(self.brain)
        assert MATCHING_COMPARE in pointers
        assert self.brain.get_interaction_stamp(
            MATCHING_COMPARE)['source'] == 'override'
        assert read_schema_version(self.brain.logs_conn, 'logs_meta',
                                   COLLAPSE_VERSION_KEY) == 0
        assert read_meta_value(self.brain.logs_conn, 'logs_meta',
                               AUDIT_KEY) is None

    def test_collapse_never_mints_a_pointer(self):
        """The pointer SET may only shrink. A minted pointer to a
        byte-identical row moves no fingerprint — invisible to every
        per-name assertion here — while re-freezing that name against
        future default changes: the exact freeze the migration removes."""
        before = set(_pointers(self.brain))
        _collapse_overrides(self.brain)
        after = set(_pointers(self.brain))
        assert after <= before, \
            'the collapse minted pointers: %s' % sorted(after - before)

    def test_matching_compare_drops_its_pointer(self):
        _collapse_overrides(self.brain)
        assert MATCHING_COMPARE not in _pointers(self.brain)
        assert _fp(self.brain, MATCHING_COMPARE) == \
            self.before_fp[MATCHING_COMPARE]

    def test_diverged_compare_keeps_its_pointer(self):
        """A real local override must survive the collapse untouched."""
        _collapse_overrides(self.brain)
        assert _pointers(self.brain).get(DIVERGED_COMPARE) == \
            self.deployed[DIVERGED_COMPARE], \
            "a diverged override was dropped — local decisions must survive"
        assert PROBE_KEY in self.brain.get_interaction_config(DIVERGED_COMPARE)

    def test_adopt_drops_its_pointer_and_converges_to_the_default(self):
        """ADOPT is the only verdict that CHANGES what runs, so it needs a
        test that requires the change rather than merely permitting it."""
        assert _fp(self.brain, ADOPT_NAME) != _default_fp(ADOPT_NAME), \
            "fixture did not diverge the ADOPT name"
        _collapse_overrides(self.brain)
        assert ADOPT_NAME not in _pointers(self.brain), \
            "ADOPT must drop its pointer unconditionally"
        assert _fp(self.brain, ADOPT_NAME) == _default_fp(ADOPT_NAME), \
            "ADOPT must leave the name running the code default"
        assert PROBE_KEY not in self.brain.get_interaction_config(ADOPT_NAME)

    def test_retire_drops_its_pointer(self):
        _collapse_overrides(self.brain)
        assert RETIRE_NAME not in _pointers(self.brain)

    def test_pinned_and_skipped_pointers_survive(self):
        _collapse_overrides(self.brain)
        after = _pointers(self.brain)
        for name in (PIN_NAME, SKIP_NAME):
            assert after.get(name) == self.deployed[name], \
                "%s (%s) must never be touched" % (name,
                                                   COLLAPSE_POLICY[name])

    def test_unknown_names_are_left_alone(self):
        _collapse_overrides(self.brain)
        assert _pointers(self.brain).get(UNKNOWN_NAME) == \
            self.deployed[UNKNOWN_NAME], \
            "a name with no policy entry must be left as-is, not dropped"

    def test_effective_values_move_only_where_adopt_allows(self):
        _collapse_overrides(self.brain)
        for name in INTERACTION_DEFAULTS:
            if COLLAPSE_POLICY.get(name) == ADOPT:
                continue
            assert _fp(self.brain, name) == self.before_fp[name], \
                "%s effective value changed — the collapse must be " \
                "value-preserving outside ADOPT" % name

    def test_no_interaction_rows_are_deleted(self):
        before = {i['name']: i['total_versions']
                  for i in self.brain.list_interactions()}
        _collapse_overrides(self.brain)
        after = {i['name']: i['total_versions']
                 for i in self.brain.list_interactions()}
        assert after == before, \
            "the collapse is pointer-only — version rows must survive"


class TestDriftRefusal:
    """The loud check: a verdict that would change a name's effective value
    without ADOPT's licence must restore the pointers and raise, so the
    version stays unstamped and the boot retries instead of half-collapsing."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env(load_env=False) as env:
            self.brain = env.brain
            self.deployed = _build_before_state(env.brain)
            yield

    def test_drift_restores_pointers_and_raises(self, monkeypatch):
        before = _pointers(self.brain)
        before_set_by = {i['name']: i.get('active_set_by')
                         for i in self.brain.list_interactions()}
        # RETIRE drops unconditionally and is not licensed to change a value:
        # exactly the shape of a mis-bucketed name.
        monkeypatch.setitem(COLLAPSE_POLICY, DIVERGED_COMPARE, RETIRE)

        with pytest.raises(RuntimeError, match='effective value'):
            _collapse_overrides(self.brain)

        assert _pointers(self.brain) == before, \
            "a refused collapse must restore every pointer it dropped"
        restored = {i['name']: i.get('active_set_by')
                    for i in self.brain.list_interactions()}
        assert restored == before_set_by, \
            "restore must replay provenance, not just the version number"

    def test_drift_writes_the_loud_channel(self, monkeypatch):
        monkeypatch.setitem(COLLAPSE_POLICY, DIVERGED_COMPARE, RETIRE)
        with pytest.raises(RuntimeError):
            _collapse_overrides(self.brain)
        # `source` on a query_logs entry names the TABLE; the event's own
        # source is `origin`.
        logged = self.brain._logs_dal.query_logs(source='debug', level='error',
                                                 hours=1, limit=200)
        origins = {e.get('origin') for e in logged.get('entries', [])}
        assert 'interaction_collapse_drift' in origins, \
            "a refused collapse must leave an error row, not just a print"

    def test_refusal_leaves_the_version_unstamped(self, monkeypatch):
        from servers.schema import read_schema_version
        monkeypatch.setitem(COLLAPSE_POLICY, DIVERGED_COMPARE, RETIRE)
        collapse_seeded_overrides(self.brain)  # swallows, logs, does not raise
        assert read_schema_version(self.brain.logs_conn, 'logs_meta',
                                   COLLAPSE_VERSION_KEY) == 0, \
            "a failed collapse must not stamp, or it never retries"
        assert _pointers(self.brain) == self.deployed, \
            "the swallowed failure must still leave the pointers restored"


class TestCrashSafety:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env(load_env=False) as env:
            self.brain = env.brain
            self.deployed = _build_before_state(env.brain)
            self.before_fp = {n: _fp(env.brain, n)
                              for n in INTERACTION_DEFAULTS}
            yield

    def test_audit_is_committed_before_the_first_drop(self, monkeypatch):
        """The module's central crash-safety claim. If the audit were written
        after the loop, an abort mid-loop would leave dropped pointers with no
        replay record."""
        from servers.schema import read_meta_value

        def boom(name):
            raise RuntimeError('simulated crash mid-collapse')

        monkeypatch.setattr(self.brain, 'clear_interaction_override', boom)
        with pytest.raises(RuntimeError, match='simulated crash'):
            _collapse_overrides(self.brain)

        raw = read_meta_value(self.brain.logs_conn, 'logs_meta', AUDIT_KEY)
        assert raw is not None, "no audit record survived the abort"
        audit = json.loads(raw)
        assert {e['name'] for e in audit} == set(self.deployed), \
            "audit must cover every pointer that existed before the drops"

    def test_abort_mid_loop_restores_every_dropped_pointer(self, monkeypatch):
        real = self.brain.clear_interaction_override
        calls = {'n': 0}

        def flaky(name):
            calls['n'] += 1
            if calls['n'] > 1:
                raise RuntimeError('simulated crash mid-collapse')
            return real(name)

        monkeypatch.setattr(self.brain, 'clear_interaction_override', flaky)
        with pytest.raises(RuntimeError, match='simulated crash'):
            _collapse_overrides(self.brain)
        assert _pointers(self.brain) == self.deployed, \
            "an aborted collapse must put back what it already dropped"

    def test_retry_after_a_partial_collapse_still_sees_the_true_baseline(self):
        """A crash can leave pointers dropped. The retry must compare against
        what those names USED to resolve to — read from the audit — not against
        the half-collapsed state it woke up in."""
        from servers.schema import read_meta_value
        _collapse_overrides(self.brain)
        audit = json.loads(read_meta_value(self.brain.logs_conn, 'logs_meta',
                                           AUDIT_KEY))
        recorded = {e['name']: e.get('effective_fingerprint') for e in audit}
        # Every registry name in the audit carries the value it resolved to
        # BEFORE anything was dropped — that is what makes a retry sound.
        for name in (MATCHING_COMPARE, ADOPT_NAME, DIVERGED_COMPARE):
            assert recorded.get(name) == self.before_fp[name], \
                "%s audit fingerprint is not its pre-collapse value" % name

    def test_audit_is_never_overwritten(self):
        from servers.schema import read_meta_value, write_meta_value
        write_meta_value(self.brain.logs_conn, 'logs_meta', AUDIT_KEY,
                         '[]')
        self.brain.logs_conn.commit()
        _collapse_overrides(self.brain)
        assert read_meta_value(self.brain.logs_conn, 'logs_meta',
                               AUDIT_KEY) == '[]', \
            "a retry must not overwrite the pre-first-attempt audit record"

    def test_backup_and_audit_keys_are_version_scoped(self):
        from servers.interaction_collapse import COLLAPSE_VERSION
        assert AUDIT_KEY.endswith('_v%d' % COLLAPSE_VERSION)
        assert BACKUP_TAG.endswith('-v%d' % COLLAPSE_VERSION)


class TestAuditRecord:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env(load_env=False) as env:
            self.brain = env.brain
            self.deployed = _build_before_state(env.brain)
            yield

    def test_audit_covers_every_pointer_and_replays(self):
        from servers.schema import read_meta_value
        _collapse_overrides(self.brain)
        audit = json.loads(read_meta_value(self.brain.logs_conn, 'logs_meta',
                                           AUDIT_KEY))
        assert {e['name'] for e in audit} == set(self.deployed), \
            "audit must cover exactly the pointers that existed"
        for entry in audit:
            for field in ('name', 'version', 'set_by', 'set_at',
                          'effective_fingerprint', 'parameters', 'verdict'):
                assert field in entry, "audit entry missing %s" % field

        for entry in audit:
            self.brain.set_interaction_active(entry['name'], entry['version'],
                                              set_by=entry['set_by'])
        assert _pointers(self.brain) == self.deployed, \
            "audit record is not a faithful replay"


class TestOnceOnly:
    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env(load_env=False) as env:
            self.brain = env.brain
            self.deployed = _build_before_state(env.brain)
            yield

    def test_second_run_does_not_touch_a_newly_deployed_override(self):
        """The version gate, tested by giving a second run something to eat."""
        collapse_seeded_overrides(self.brain)
        assert MATCHING_COMPARE not in _pointers(self.brain)
        version = _deploy_default(self.brain, MATCHING_COMPARE)
        collapse_seeded_overrides(self.brain)
        assert _pointers(self.brain).get(MATCHING_COMPARE) == version, \
            "the collapse re-ran — the version gate is not holding"

    def test_stamp_is_written_on_success(self):
        from servers.schema import read_schema_version
        collapse_seeded_overrides(self.brain)
        assert read_schema_version(self.brain.logs_conn, 'logs_meta',
                                   COLLAPSE_VERSION_KEY) >= 1
