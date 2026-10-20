"""Tests for interactions as living boundaries.

Code owns every interaction's default (servers/interaction_defaults.py);
the interactions table holds only deployed OVERRIDES. The resolver
accessors overlay the active row onto the code default at read time.

Run: python3 -m pytest tests/test_interactions_runtime.py -v
"""
import json
import pytest
from tests.isolated_brain import brain_env


# ═══════════════════════════════════════════════════════
# Brain methods
# ═══════════════════════════════════════════════════════

class TestBrainInteractionMethods:
    """Verify the resolver accessors: get_interaction_config() /
    get_interaction_prompt() overlay the active DB override onto the code
    default from INTERACTION_DEFAULTS."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def test_get_config_overlays_partial_override(self):
        """A one-key override changes that key and ONLY that key — every
        unmentioned key still tracks the code default (decision e183d22c:
        overlay, not whole-value replace)."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        default = INTERACTION_DEFAULTS['s1e'][1]
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'sentinel-x'}))
        self.brain._interaction_dal.set_active('s1e', 1, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'sentinel-x'
        for key, value in default.items():
            if key != 'effort':
                assert config[key] == value

    def test_get_config_unknown_name_raises(self):
        with pytest.raises(KeyError):
            self.brain.get_interaction_config('nonexistent')

    def test_get_prompt_returns_override_text(self):
        self.brain._interaction_dal.register(
            'surface', template='You are a test judge. Select wisely.',
            parameters=json.dumps({}))
        self.brain._interaction_dal.set_active('surface', 1, set_by='test')
        prompt = self.brain.get_interaction_prompt('surface')
        assert prompt == 'You are a test judge. Select wisely.'

    def test_get_prompt_unknown_name_raises(self):
        with pytest.raises(KeyError):
            self.brain.get_interaction_prompt('nonexistent')

    def test_register_never_auto_activates(self):
        """Registration NEVER changes the runtime pointer — not v1, not v2.
        A write is not a deployment decision; the name runs on its code
        default until set_active deploys an override.
        """
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'v1-effort'}))
        # v1 is dormant — runtime still reads the code default
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == INTERACTION_DEFAULTS['s1e'][1]['effort'], \
            "v1 register must NOT auto-activate; runtime runs the code default"

        # Deploy v1 explicitly — runtime flips to the override
        self.brain._interaction_dal.set_active('s1e', 1, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v1-effort'

        # Register v2 — must NOT change what runtime reads
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'v2-effort'}),
            created_by='sleep:s3')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v1-effort', \
            "v2 register should NOT auto-activate; runtime should still see v1"

        # Now explicitly activate v2 — runtime flips
        self.brain._interaction_dal.set_active('s1e', 2, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v2-effort'

        # Rollback to v1 by re-activating
        self.brain._interaction_dal.set_active('s1e', 1, set_by='test')
        config = self.brain.get_interaction_config('s1e')
        assert config['effort'] == 'v1-effort'

    def test_set_active_rejects_unknown_version(self):
        """set_active raises when target version isn't registered."""
        self.brain._interaction_dal.register(
            'one_version', template='', parameters=json.dumps({'k': 1}))
        with pytest.raises(ValueError):
            self.brain._interaction_dal.set_active('one_version', 99, set_by='test')
        with pytest.raises(ValueError):
            self.brain._interaction_dal.set_active('never_registered', 1, set_by='test')


# ═══════════════════════════════════════════════════════
# Versioning and comparison
# ═══════════════════════════════════════════════════════

class TestInteractionVersioning:
    """Verify version tracking and lineage."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._interaction_dal
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.commit()
            yield

    def test_auto_increment_version(self):
        r1 = self.dal.register('test', template='v1', parameters='{}')
        r2 = self.dal.register('test', template='v2', parameters='{}')
        assert r1['version'] == 1
        assert r2['version'] == 2

    def test_old_versions_preserved(self):
        self.dal.register('test', template='v1 prompt', parameters='{}')
        self.dal.register('test', template='v2 prompt', parameters='{}')
        v1 = self.dal.get_version('test', 1)
        v2 = self.dal.get_version('test', 2)
        assert v1['template'] == 'v1 prompt'
        assert v2['template'] == 'v2 prompt'

    def test_created_by_tracked(self):
        self.dal.register('test', template='', parameters='{}', created_by='anchor')
        self.dal.register('test', template='', parameters='{}', created_by='sleep:s3')
        v1 = self.dal.get_version('test', 1)
        v2 = self.dal.get_version('test', 2)
        assert v1['created_by'] == 'anchor'
        assert v2['created_by'] == 'sleep:s3'

    def test_parent_version_linked(self):
        self.dal.register('test', template='', parameters='{}')
        self.dal.register('test', template='', parameters='{}')
        v2 = self.dal.get_version('test', 2)
        assert v2.get('parent_version') == 1


# ═══════════════════════════════════════════════════════
# Trace linkage
# ═══════════════════════════════════════════════════════

class TestInteractionTraceLinkage:
    """Verify traces reference interaction versions."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._interaction_dal
            self.trace_dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_trace_includes_interaction_id(self):
        """Trace events reference which interaction version produced them."""
        result = self.dal.register('judge', template='test', parameters='{}')
        interaction_id = result['id']

        self.trace_dal.append(
            chain_id='s1r-test-1', scale='s1', event_type='delta',
            ref_type='additionalContext', summary='test output',
            interaction_id=interaction_id, session_id='test')

        chain = self.trace_dal.get_chain('s1r-test-1')
        assert len(chain) == 1
        # interaction_id should be queryable
        row = self.brain.logs_conn.execute(
            'SELECT interaction_id FROM trace_events WHERE chain_id = ?',
            ('s1r-test-1',)).fetchone()
        assert row[0] == interaction_id

    def test_compare_versions_via_traces(self):
        """Two interaction versions produce traces that can be compared."""
        r1 = self.dal.register('judge', template='v1', parameters='{}')
        r2 = self.dal.register('judge', template='v2', parameters='{}')

        # v1 produced a recall
        self.trace_dal.append(
            chain_id='s1r-v1-1', scale='s1', event_type='delta',
            ref_type='additionalContext', summary='v1 output',
            interaction_id=r1['id'], session_id='test')

        # v2 produced a recall
        self.trace_dal.append(
            chain_id='s1r-v2-1', scale='s1', event_type='delta',
            ref_type='additionalContext', summary='v2 output',
            interaction_id=r2['id'], session_id='test')

        # Query traces by interaction version
        v1_traces = self.brain.logs_conn.execute(
            'SELECT summary FROM trace_events WHERE interaction_id = ?',
            (r1['id'],)).fetchall()
        v2_traces = self.brain.logs_conn.execute(
            'SELECT summary FROM trace_events WHERE interaction_id = ?',
            (r2['id'],)).fetchall()

        assert len(v1_traces) == 1
        assert len(v2_traces) == 1
        assert v1_traces[0][0] == 'v1 output'
        assert v2_traces[0][0] == 'v2 output'


class TestS1RunsRecordTheirInteraction:
    """An S1 run resolves each interaction ONCE: the prompt and config it
    sends and the stamp its traces record come from that one read, so an
    override in effect is what both show. The S2 twin is
    test_s2_batch_completion.TestRunsRecordTheirInteraction."""

    _LOOP = {'rounds': 1, 'actions': 0, 'write_actions': 0, 'action_details': [],
             'read_calls': [], 'final_text': 'done', 'elapsed_ms': 1,
             'input_tokens': 0, 'output_tokens': 0, 'cache_read_tokens': 0,
             'cache_creation_tokens': 0, 'model': 'claude-test'}
    _WINDOW = [{'user': {'id': 'aaaaaaaa', 'ref_type': 'user_message', 'chain_id': 's0-test-0',
                    'created_at': '2026-10-19T00:00:00+00:00', 'metadata': {},
                    'summary': 'hi', 'content': 'hi', 'judge_output': None},
           'with': [], 'actions': [],
           'assistant': {'id': 'bbbbbbbb', 'ref_type': 'assistant_message',
                         'chain_id': 's0-test-0', 'metadata': {},
                         'created_at': '2026-10-19T00:00:01+00:00',
                         'summary': 'hello', 'content': 'hello'}}]

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env(needs_embedder=False) as env:
            self.brain = env.brain
            yield

    def test_encoder_runs_on_and_records_the_resolved_k(self):
        from unittest import mock
        from servers.scales.s1 import encode
        from tests.interaction_override import interaction_override
        brain, sent, traces = self.brain, {}, []

        def loop(**kw):
            sent.update(kw)
            return dict(self._LOOP)

        def dispatch(cmd, args):
            if cmd == 'trace_append':
                traces.append(args)
            return {'ok': True}

        with interaction_override(brain, 's1e', template='S1E-ARM',
                                  parameters={'model': 'claude-arm', 'effort': 'low'}), \
                interaction_override(brain, 's1e_gist', template='GIST-ARM\n'):
            enc = brain.get_interaction_stamp('s1e')
            gist = brain.get_interaction_stamp('s1e_gist')
            with mock.patch.object(encode, '_window', return_value=self._WINDOW), \
                    mock.patch('servers.scales.runner.make_client', return_value=object()), \
                    mock.patch.object(encode, 'run_llm_loop', side_effect=loop), \
                    mock.patch.object(brain, '_resolve_interaction',
                                      wraps=brain._resolve_interaction) as spy:
                encode.run_encoding(brain, dispatch, 1, 'sess-k-stamp')

        assert sent['system_prompt'].startswith('S1E-ARM')
        assert 'GIST-ARM' in sent['user_content']
        assert (sent['model'], sent['effort']) == ('claude-arm', 'low')
        by_ref = {t['ref_type']: t for t in traces}
        run = by_ref['encoding_run']
        assert enc['source'] == gist['source'] == 'override'
        assert run['metadata']['interaction_fingerprint'] == enc['fingerprint']
        assert run['metadata']['gist_interaction_fingerprint'] == gist['fingerprint']
        assert run['interaction_id'] == enc['id']
        assert 'K#%s' % enc['fingerprint'] in by_ref['encoding_prompt']['summary']
        # One resolution per K: separate reads can pair one row's template
        # with another row's stamp.
        resolved = [c.args[0] for c in spy.call_args_list]
        assert (resolved.count('s1e'), resolved.count('s1e_gist')) == (1, 1)

    def test_surface_sends_and_stamps_the_resolved_k(self):
        from unittest import mock
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        from servers.scales.s1 import surface, surface_capture
        from tests.interaction_override import interaction_override
        brain, captured = self.brain, {}

        class _Stop(Exception):
            pass

        def begin(*args, **kw):
            captured.update(kw)
            raise _Stop()

        with interaction_override(brain, 'surface', template='SURFACE-ARM'):
            stamp = brain.get_interaction_stamp('surface')
            with mock.patch.object(surface_capture, 'begin', side_effect=begin), \
                    mock.patch.object(brain, '_resolve_interaction',
                                      wraps=brain._resolve_interaction) as spy:
                with pytest.raises(_Stop):
                    surface._call_surface(brain, [], 'a message', [], 'sess-k-stamp', {})

        assert stamp['source'] == 'override'
        assert captured['surface_instructions'] == 'SURFACE-ARM'
        assert captured['interaction_stamp'] == stamp
        assert captured['layout'] == INTERACTION_DEFAULTS['surface'][1]['layout']
        assert [c.args[0] for c in spy.call_args_list].count('surface') == 1


# ═══════════════════════════════════════════════════════
# Fallback behavior + resolver guards
# ═══════════════════════════════════════════════════════

class TestInteractionFallback:
    """No DB row is the normal state after the override collapse: the
    resolver returns the code default, silently."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def test_config_fallback_returns_code_default(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        assert (self.brain.get_interaction_config('surface')
                == INTERACTION_DEFAULTS['surface'][1])

    def test_prompt_fallback_returns_code_default(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        assert (self.brain.get_interaction_prompt('surface')
                == INTERACTION_DEFAULTS['surface'][0])

    def test_stamp_fallback_is_default_source(self):
        """No row → source 'default', version 0, id None — and the
        fingerprint hashes the RESOLVED default, not ''/{}."""
        from servers.interaction_defaults import (
            INTERACTION_DEFAULTS, interaction_fingerprint)
        stamp = self.brain.get_interaction_stamp('surface')
        template, config = INTERACTION_DEFAULTS['surface']
        assert stamp == {
            'fingerprint': interaction_fingerprint('surface', template, config),
            'source': 'default', 'version': 0, 'id': None}

    def test_unknown_name_raises(self):
        """A typo'd or unregistered name must raise, not run on an empty
        prompt (guard 2 — safe because tests/test_interaction_defaults.py
        keeps the registry complete)."""
        with pytest.raises(KeyError):
            self.brain.get_interaction_config('judge')
        with pytest.raises(KeyError):
            self.brain.get_interaction_prompt('judge')
        with pytest.raises(KeyError):
            self.brain.get_interaction_stamp('judge')


class TestResolverGuards:
    """The dangerous rows: unparseable JSON and invalid values must degrade
    LOUDLY to the code default — a typo'd override silently reverting a
    boundary is the failure this resolver exists to prevent."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def _deploy(self, name, version=1):
        self.brain._interaction_dal.set_active(name, version, set_by='test')

    def _resolve_errors(self):
        return self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM debug_log WHERE source = "
            "'interaction_resolve'").fetchone()[0]

    def test_unparseable_json_falls_back_loudly(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self.brain._interaction_dal.register(
            's1e', template='', parameters='{not valid json')
        self._deploy('s1e')
        config = self.brain.get_interaction_config('s1e')
        assert config == INTERACTION_DEFAULTS['s1e'][1]
        assert self._resolve_errors() >= 1

    def test_non_object_json_falls_back_loudly(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps([1, 2, 3]))
        self._deploy('s1e')
        config = self.brain.get_interaction_config('s1e')
        assert config == INTERACTION_DEFAULTS['s1e'][1]
        assert self._resolve_errors() >= 1

    def test_validator_violation_falls_back_loudly(self):
        """A scopes override with an invalid mode (written past the
        register_interaction door, e.g. by an older version) must not
        become the running policy."""
        from servers.scopes import SCOPES_CONFIG_V1
        self.brain._interaction_dal.register(
            'scopes', template='',
            parameters=json.dumps({'project': {'mode': 'bogus'}}))
        self._deploy('scopes')
        config = self.brain.get_interaction_config('scopes')
        assert config == SCOPES_CONFIG_V1
        assert self._resolve_errors() >= 1

    def test_register_door_refuses_validator_violations(self):
        """The write door REFUSES what the read seam degrades — the generic
        INTERACTION_VALIDATORS path that replaced the scopes special-case."""
        with pytest.raises(ValueError):
            self.brain.register_interaction(
                'scopes',
                parameters=json.dumps({'project': {'mode': 'bogus'}}))
        with pytest.raises(ValueError):
            self.brain.register_interaction('scopes', parameters='{not json')

    def test_stamp_override_fingerprints_resolved_value(self):
        """source='override' when a row shadows the default, and the
        fingerprint hashes the OVERLAID effective config — not the raw
        override fragment."""
        from servers.interaction_defaults import (
            INTERACTION_DEFAULTS, interaction_fingerprint)
        row = self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'high'}))
        self._deploy('s1e')
        stamp = self.brain.get_interaction_stamp('s1e')
        template, default = INTERACTION_DEFAULTS['s1e']
        assert stamp['source'] == 'override'
        assert stamp['version'] == row['version']
        assert stamp['id'] == row['id']
        assert stamp['fingerprint'] == interaction_fingerprint(
            's1e', template, {**default, 'effort': 'high'})

    def test_stamp_vacuous_row_is_default_source(self):
        """A row that contributes nothing (empty template, empty config)
        does not shadow the default — it stamps 'default' like no row."""
        self.brain._interaction_dal.register(
            's1e', template='', parameters='{}')
        self._deploy('s1e')
        stamp = self.brain.get_interaction_stamp('s1e')
        assert stamp['source'] == 'default'
        assert stamp['version'] == 0
        assert stamp['id'] is None

    def test_caller_overlay_stamps_the_config_that_runs(self):
        """A caller's overlay (evals) lays key-level over the resolved config.
        When it changes the effective value, the fingerprint hashes what runs
        and source says 'caller'; version and id still name the row it sat
        on. An overlay that changes nothing stamps as no overlay."""
        from servers.interaction_defaults import (
            INTERACTION_DEFAULTS, interaction_fingerprint)
        row = self.brain._interaction_dal.register(
            's1e', template='', parameters=json.dumps({'effort': 'high'}))
        self._deploy('s1e')
        template, default = INTERACTION_DEFAULTS['s1e']
        assert (self.brain.get_interaction_effective(
                    's1e', overlay={'effort': 'high'})
                == self.brain.get_interaction_effective('s1e'))
        effective = self.brain.get_interaction_effective(
            's1e', overlay={'effort': 'low'})
        assert effective['config'] == {**default, 'effort': 'low'}
        assert effective['stamp'] == {
            'fingerprint': interaction_fingerprint(
                's1e', template, {**default, 'effort': 'low'}),
            'source': 'caller', 'version': row['version'], 'id': row['id']}


# ═══════════════════════════════════════════════════════
# Pointer delete = clear (Step 5: nothing resurrects MAX)
# ═══════════════════════════════════════════════════════

class TestPointerDeleteIsClear:
    """'No pointer' means 'no override deployed' — nothing may resurrect
    MAX(version). The motivating landmine: trace_recording sits at active=1
    with a dormant v2 = DEBUG (all payload kinds on); the old get_active
    fallback turned a pointer delete into silent full-payload capture, and
    the old ensure_logs_schema backstop re-pinned MAX on the next boot."""

    def test_pointer_delete_reverts_to_code_default_and_survives_reopen(self):
        from servers.brain import Brain
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        with brain_env() as env:
            brain = env.brain
            # Construct the whole deployed state — register v1 and point at
            # it — then register a dormant version ABOVE it: the exact shape
            # where a MAX(version) fallback returns something nobody
            # deployed. Registering is not optional scene-setting; set_active
            # refuses a version that was never registered.
            brain.register_interaction(
                'surface', template='deployed v1', parameters='{}')
            brain._interaction_dal.set_active('surface', 1, set_by='test')
            brain.register_interaction(
                'surface', template='SENTINEL vNext — never deployed',
                parameters='{}')
            assert brain.get_interaction_stamp('surface')['source'] == 'override'

            brain.logs_conn.execute(
                "DELETE FROM interaction_active WHERE name = 'surface'")
            brain.logs_conn.commit()

            # DAL: no pointer -> None, never the dormant MAX version.
            assert brain.get_interaction('surface') is None
            # Resolver: falls through to the code default.
            assert (brain.get_interaction_prompt('surface')
                    == INTERACTION_DEFAULTS['surface'][0])
            assert brain.get_interaction_stamp('surface')['source'] == 'default'

            # A boot must not re-create the pointer (Brain.__init__ writes
            # nothing to the interactions store).
            brain.save()
            brain.close()
            env.brain = Brain(env.brain_db)
            assert env.brain.get_interaction('surface') is None
            count = env.brain.logs_conn.execute(
                "SELECT COUNT(*) FROM interaction_active "
                "WHERE name = 'surface'").fetchone()[0]
            assert count == 0


# ═══════════════════════════════════════════════════════
# Override resolution (the whole override model in one test)
# ═══════════════════════════════════════════════════════

class TestOverrideResolution:
    """An override survives the code default moving underneath it — the
    core guarantee of the override model: nothing is ever written between
    code and DB, so a deployed override outlives any number of default
    changes, and clearing it lands on the CURRENT default."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def test_override_survives_a_default_change(self, monkeypatch):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        override_prompt = 'OVERRIDE PROMPT — deployed before the default moved'
        self.brain.register_interaction(
            's1e', template=override_prompt,
            parameters=json.dumps({'effort': 'override-effort'}))
        self.brain.set_interaction_active('s1e', 1, set_by='test')
        assert self.brain.get_interaction_prompt('s1e') == override_prompt

        # The repo moves on: a merge changes the code default underneath.
        _old_template, old_config = INTERACTION_DEFAULTS['s1e']
        new_default = ('NEW SHIPPED DEFAULT — landed after the override. ' * 4,
                       {**old_config, 'effort': 'new-default-effort'})
        monkeypatch.setitem(INTERACTION_DEFAULTS, 's1e', new_default)

        # The override still wins.
        assert self.brain.get_interaction_prompt('s1e') == override_prompt
        assert (self.brain.get_interaction_config('s1e')['effort']
                == 'override-effort')
        # Unmentioned keys track the NEW default (overlay, not snapshot).
        for key, value in new_default[1].items():
            if key != 'effort':
                assert self.brain.get_interaction_config('s1e')[key] == value

        # Clearing lands on the current default, not the one from deploy time.
        self.brain.clear_interaction_override('s1e')
        assert self.brain.get_interaction_prompt('s1e') == new_default[0]
        assert (self.brain.get_interaction_config('s1e')['effort']
                == 'new-default-effort')


# ═══════════════════════════════════════════════════════
# Boot writes nothing
# ═══════════════════════════════════════════════════════

class TestBrainOpenWritesNothing:
    """Brain.__init__ writes nothing to the interactions store. Seeding is
    gone; the daemon-boot collapse is the single sanctioned boot-path write.
    Reopening a Brain on an existing DB must leave every name, version count
    and pointer byte-identical — catches any future write-on-boot."""

    @staticmethod
    def _snapshot(brain):
        return sorted(
            (i['name'], i['total_versions'], i.get('active_version'))
            for i in brain.list_interactions())

    def test_reopen_is_byte_identical(self):
        from servers.brain import Brain
        with brain_env() as env:
            brain = env.brain
            # A registered row + pointer so the comparison is non-vacuous
            # even on an empty copy.
            brain.register_interaction(
                's1e', template='local override', parameters='{}')
            versions = brain.list_interaction_versions('s1e')
            brain.set_interaction_active(
                's1e', versions[-1]['version'], set_by='test')
            before = self._snapshot(brain)
            assert before, 'empty interactions store — comparison is vacuous'

            brain.save()
            brain.close()
            env.brain = Brain(env.brain_db)
            assert self._snapshot(env.brain) == before, \
                'constructing a Brain wrote to the interactions store'

    def test_fresh_brain_has_an_empty_interactions_store(self, tmp_path):
        """The reopen check above runs on a production COPY whose rows mask a
        re-added create-only seed (it skips existing names, so the snapshot
        stays byte-identical). Only a genuinely fresh brain can REQUIRE
        boot-writes-nothing rather than permit it."""
        from servers.brain import Brain
        brain = Brain(db_path=str(tmp_path / 'brain.db'), skip_embedder=True)
        try:
            assert brain.list_interactions() == [], \
                'Brain.__init__ seeded interaction rows on a fresh brain'
        finally:
            brain.close()


# ═══════════════════════════════════════════════════════
# Reserved provenance (the collapse migration and audits read these)
# ═══════════════════════════════════════════════════════

class TestReservedProvenance:
    """The MCP door may not mint the reserved provenance values — the
    collapse migration and pointer audits read them to tell system-placed
    pointers from human deployment decisions."""

    NAME = 's1e'

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            # A real registered version, so set_active's refusal below can
            # only come from the provenance check.
            self.brain.register_interaction(
                self.NAME, template='x', parameters='{}', created_by='test')
            yield

    def test_set_active_refuses_reserved_set_by(self):
        from servers.dal_logs import SYSTEM_PROVENANCE
        from servers.dispatch_observability import _handle_set_interaction_active
        for reserved in SYSTEM_PROVENANCE:
            result = _handle_set_interaction_active(
                self.brain, {'name': self.NAME, 'version': 1,
                             'set_by': reserved}, None)
            assert not result['ok'], reserved
            assert 'reserved' in result['error']

    def test_register_refuses_reserved_created_by(self):
        from servers.dal_logs import SYSTEM_PROVENANCE
        from servers.dispatch_observability import _handle_register_interaction
        for reserved in SYSTEM_PROVENANCE:
            result = _handle_register_interaction(
                self.brain,
                {'name': self.NAME, 'template': 'x',
                 'created_by': reserved}, None)
            assert not result['ok'], reserved
            assert 'reserved' in result['error']

    def test_normal_provenance_still_works(self):
        from servers.dispatch_observability import _handle_register_interaction
        result = _handle_register_interaction(
            self.brain,
            {'name': self.NAME, 'template': 'x', 'created_by': 'anchor'},
            None)
        assert result['ok']


# ═══════════════════════════════════════════════════════
# The clear verb (Step 6: override is a two-way door)
# ═══════════════════════════════════════════════════════

class TestClearInteractionOverride:
    """clear_interaction_override is the inverse of set_interaction_active:
    without it an override is a one-way door and the model degrades back
    into the per-name freeze. Clearing must revert to the code default
    IMMEDIATELY — the TTL caches are what would make a clear run late."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            env.brain.logs_conn.execute('DELETE FROM interactions')
            env.brain.logs_conn.execute('DELETE FROM interaction_active')
            env.brain.logs_conn.commit()
            yield

    def _deploy(self, name, template='', config=None):
        self.brain._interaction_dal.register(
            name, template=template, parameters=json.dumps(config or {}))
        self.brain._interaction_dal.set_active(name, 1, set_by='test')

    def test_clear_reverts_to_code_default_and_reports_distinctly(self):
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self._deploy('s1e', config={'effort': 'sentinel-clear'})
        assert (self.brain.get_interaction_config('s1e')['effort']
                == 'sentinel-clear')

        result = self.brain.clear_interaction_override('s1e')
        assert result == {'name': 's1e', 'cleared': True}
        assert (self.brain.get_interaction_config('s1e')
                == INTERACTION_DEFAULTS['s1e'][1])
        assert self.brain.get_interaction_stamp('s1e')['source'] == 'default'
        # Version rows survive the clear — re-activation stays possible.
        assert self.brain.get_interaction('s1e', version=1) is not None

        # No pointer existed — reported distinctly from "cleared".
        assert self.brain.clear_interaction_override('s1e') == {
            'name': 's1e', 'cleared': False}

    def test_clear_reaches_trace_recording_cache_immediately(self):
        """The landmine workflow: deploy a debug-ish override, clear it —
        the payload recorder must see the revert NOW, not a TTL later."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        self._deploy('trace_recording', config={'round_payload': True})
        warmed = self.brain._trace_recording_config()
        assert warmed['round_payload'] is True

        self.brain.clear_interaction_override('trace_recording')
        assert (self.brain._trace_recording_config()
                == INTERACTION_DEFAULTS['trace_recording'][1])

    def test_set_active_and_clear_reach_recall_laf_cache_immediately(self):
        """recall_laf's engine TTL cache had NO invalidation hook — a flip
        or clear ran up to CONFIG_TTL_S late. Both verbs must drop it."""
        from servers.recall_laf import get_engine
        engine = get_engine(self.brain)
        engine.config(self.brain)
        assert engine._cfg is not None

        # Through the Brain verbs — cache invalidation is their policy,
        # not the DAL's.
        self.brain._interaction_dal.register(
            'recall_laf', template='', parameters='{}')
        self.brain.set_interaction_active('recall_laf', 1, set_by='test')
        assert engine._cfg is None, 'set_active must drop the laf cache'

        engine.config(self.brain)
        assert engine._cfg is not None
        self.brain.clear_interaction_override('recall_laf')
        assert engine._cfg is None, 'clear must drop the laf cache'
        # Behavioral half: the very next config() read resolves the current
        # value (the code default) — no TTL wait.
        assert (engine.config(self.brain)
                == self.brain.get_interaction_config('recall_laf'))

    def test_clear_refuses_unknown_names(self):
        """A typo'd clear must never report 'already on the default' while
        the real override keeps running — nothing deleted + no code default
        means refusal, not a benign no-op."""
        with pytest.raises(KeyError):
            self.brain.clear_interaction_override('trace_recoding')  # typo

    def test_clear_purges_the_recall_result_cache(self):
        """The recall RESULT cache keys on the query alone — a laf flip or
        clear must purge it or an identical query re-asked within the TTL
        returns the pre-flip result."""
        self.brain._recall_cache_put(('probe',), {'results': ['stale']})
        assert self.brain._recall_cache_get(('probe',)) is not None
        self.brain.clear_interaction_override('recall_laf')
        assert self.brain._recall_cache_get(('probe',)) is None
