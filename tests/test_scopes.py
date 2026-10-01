"""Scope policy + veil (servers/scopes.py) — modes, overrides, and the
isolation wall.

Architecture under test: enforcement is ONE precomputed hidden-set (the
veil) built from indexed KV queries, checked by membership at every ambient
surfacing funnel — never per-candidate policy evaluation. Layers:

  1. Unit — ScopePolicy resolution + validate_scopes_config (no brain).
  2. Veil semantics — build through the REAL write path (sessions with
     different derived projects writing via dispatch handlers, provenance
     via stamp_scope_provenance), then assert the hidden-set both
     directions. Pins stamp→store→veil as one loop.
  3. Funnel coverage — the leak paths the D review found: recall e2e,
     neighbor attachments, filter_nodes enumeration, boot text,
     fetch-tools fail-closed.
  4. Behavior-neutrality — the seeded default config produces an empty
     veil (the ship-safe claim, proven).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.scopes import (ScopePolicy, SCOPES_CONFIG_V1, DEFAULT_MODE,
                            build_veil, validate_scopes_config)


class _Log:
    def __init__(self):
        self.entries = []

    def __call__(self, kind, exc, ctx=''):
        self.entries.append((kind, str(exc), ctx))


class TestScopePolicyResolution:
    """Layer 1 — pure resolution + validation, no brain."""

    def test_empty_config_defaults_scoped(self):
        p = ScopePolicy({})
        assert p.mode('project') == 'scoped'
        assert p.mode('counterpart', 'Anyone') == 'scoped'
        assert not p.has_isolation

    def test_seed_config_is_behavior_neutral(self):
        assert not ScopePolicy(SCOPES_CONFIG_V1).has_isolation
        assert validate_scopes_config(SCOPES_CONFIG_V1) == []

    def test_dimension_mode_and_value_override(self):
        p = ScopePolicy({'project': {'mode': 'open',
                                     'overrides': {'client-x': 'isolated'}}})
        assert p.mode('project') == 'open'
        assert p.mode('project', 'client-x') == 'isolated'
        assert p.mode('project', 'CLIENT-X') == 'isolated'   # case-insensitive
        assert p.mode('project', 'other') == 'open'
        assert p.has_isolation

    def test_mode_strings_and_dim_keys_normalized(self):
        # 'Isolated ' / 'Project' must not silently mean LESS isolation
        # than configured — normalization happens before judgment.
        p = ScopePolicy({'Project': {'mode': 'Scoped',
                                     'overrides': {'client-x': ' Isolated '}}})
        assert p.mode('project', 'client-x') == 'isolated'
        assert p.has_isolation

    def test_invalid_mode_falls_back_loudly_never_isolated(self):
        log = _Log()
        p = ScopePolicy({'project': {'mode': 'sealed',
                                     'overrides': {'x': 'locked'}}}, log=log)
        assert p.mode('project') == DEFAULT_MODE
        assert p.mode('project', 'x') == DEFAULT_MODE
        assert not p.has_isolation
        assert len(log.entries) == 2

    def test_unknown_dimension_logged_and_ignored(self):
        log = _Log()
        p = ScopePolicy({'flavor': {'mode': 'isolated'}}, log=log)
        assert not p.has_isolation
        assert log.entries

    def test_counterpart_isolation_refused_until_session_resolvable(self):
        # counterpart's session value is still the install constant —
        # isolating a foreign value would hide those nodes from EVERY
        # session including their own (un-exitable), and isolating the
        # constant would black out the stamped corpus. Refused loudly,
        # degrades to scoped; lifts when the speaker arc's F4 makes the
        # dimension session-resolvable.
        log = _Log()
        p = ScopePolicy({'counterpart': {'mode': 'scoped',
                                         'overrides': {'dana': 'isolated'}}},
                        log=log)
        assert p.mode('counterpart', 'dana') == 'scoped'
        assert not p.has_isolation
        assert any('refused' in msg for _, msg, _ in log.entries)
        assert validate_scopes_config(
            {'counterpart': {'mode': 'isolated'}})  # non-empty violations

    def test_validate_usable_at_the_write_door(self):
        v = validate_scopes_config({'project': {'mode': 'isolatd'}})
        assert v and 'isolatd' in v[0]
        assert validate_scopes_config(None) == []
        assert validate_scopes_config('nope')


def _mint(brain, sid, project, title):
    """Create a node through the REAL write path: session env → dispatch
    handler → stamp_scope_provenance. Never hand-set provenance in these
    tests — the loop under test is stamp→store→veil."""
    from servers.dispatch_write import _handle_remember
    ctx = brain.get_or_create_session(sid)
    ctx.set_env(cwd='/tmp/%s' % (project or 'none'), project=project)
    r = _handle_remember(brain, {
        'type': 'fact', 'title': title, 'content': 'content of ' + title,
        '_caller_session': sid,
    }, [])
    assert r['ok'], r
    return r['result']['id']


def _set_scopes(brain, config):
    """The full operator discipline: register (never activates) + explicit
    activate to deploy the override."""
    import json
    r = brain.register_interaction(
        'scopes', template='', parameters=json.dumps(config),
        created_by='anchor')
    brain.set_interaction_active('scopes', r['version'])


ISOLATE_CLIENT_X = {'project': {'mode': 'scoped',
                                'overrides': {'client-x': 'isolated'}}}


class TestVeilSemantics(BrainTestBase):
    """Layer 2 — the hidden-set, built from real stamped provenance."""
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.brain_node = _mint(self.brain, 'sess-brain', 'brain', 'brain fact')
        self.client_node = _mint(self.brain, 'sess-client', 'client-x',
                                 'client fact')
        self.unscoped_node = _mint(self.brain, 'sess-none', '', 'unscoped fact')

    def test_outward_wall_hides_walled_project_everywhere(self):
        policy = ScopePolicy(ISOLATE_CLIENT_X)
        # Foreign session and sessionless viewer both hide the walled node.
        for scope in ({'project': 'brain'}, None):
            veil = build_veil(self.brain, policy, scope)
            assert self.client_node in veil
            assert self.brain_node not in veil
            assert self.unscoped_node not in veil

    def test_inside_the_wall_sees_own_plus_unscoped(self):
        policy = ScopePolicy(ISOLATE_CLIENT_X)
        veil = build_veil(self.brain, policy, {'project': 'client-x'})
        assert self.client_node not in veil
        assert self.brain_node in veil          # inward: foreign hides
        assert self.unscoped_node not in veil   # unknown stays neutral

    def test_case_insensitive_both_sides(self):
        policy = ScopePolicy({'project': {'mode': 'scoped',
                                          'overrides': {'CLIENT-X': 'isolated'}}})
        veil = build_veil(self.brain, policy, {'project': 'Brain'})
        assert self.client_node in veil
        assert self.brain_node not in veil

    def test_brain_scope_veil_caches_and_self_invalidates(self):
        _set_scopes(self.brain, ISOLATE_CLIENT_X)
        veil1 = self.brain.scope_veil('sess-brain')
        assert self.client_node in veil1
        assert self.brain.scope_veil('sess-brain') is veil1   # cache hit
        # A newly stamped walled node invalidates via change_key — no TTL.
        newer = _mint(self.brain, 'sess-client', 'client-x', 'newer client fact')
        veil2 = self.brain.scope_veil('sess-brain')
        assert newer in veil2
        # A config flip invalidates via the version probe.
        _set_scopes(self.brain, SCOPES_CONFIG_V1)
        assert self.brain.scope_veil('sess-brain') == frozenset()

    def test_default_config_builds_empty_veil(self):
        _set_scopes(self.brain, SCOPES_CONFIG_V1)
        assert self.brain.scope_veil('sess-brain') == frozenset()
        assert self.brain.scope_veil('') == frozenset()


class TestFunnelCoverage(BrainTestBase):
    """Layer 3 — the ambient surfacing funnels the D review found leaking.

    needs_embedder=True because the gating assertions are ABSENCE assertions:
    a walled node must not come back. On a brain that cannot recall, nothing
    comes back and they hold for the wrong reason. Measured when the fixture
    started refusing blind recalls — the dropped-count test passed against an
    empty result the wall had no part in producing.
    """
    needs_embedder = True

    def setUp(self):
        super().setUp()
        self.brain_node = _mint(self.brain, 'sess-brain', 'brain', 'brain fact')
        self.client_node = _mint(self.brain, 'sess-client', 'client-x',
                                 'walled client fact')
        _set_scopes(self.brain, ISOLATE_CLIENT_X)

    def test_canonicalize_results_drops_walled_connections(self):
        # A kept node edged to a walled node must not leak the walled title
        # through `connections` — the id would feed the out-of-candidate
        # admission path (leak escalation). canonicalize_results is the one
        # door every recall shape goes through, so scrubbing here covers
        # by-query, by-id and batch at once.
        self.brain.connect(self.brain_node, self.client_node,
                           relation='extends')
        results = [{'id': self.brain_node}]
        self.brain.canonicalize_results(results, session_id='sess-brain')
        conn_ids = {c['id'] for c in results[0]['connections']}
        assert self.client_node not in conn_ids
        # get_node applies no veil — the edge IS there at the data layer.
        # Guards the test itself against the edge silently not existing.
        # (A sessionless viewer is NOT the unveiled case: an `isolated`
        # value is walled from every scope, including no scope at all.)
        assert self.client_node in {
            c['id'] for c in self.brain.get_node(self.brain_node)['connections']}

    def test_boot_text_is_gated(self):
        # Boot is the most ambient surface there is: a walled node must not
        # print into a foreign session's boot block. Standing items is the
        # boot lane that lists nodes by type; point it at 'fact' so both
        # fixture nodes are candidates.
        from unittest.mock import patch
        from servers.brain_voice import BrainVoice
        with patch.dict(os.environ, {'BRAIN_BOOT_INJECT_TYPES': 'fact'}):
            text = BrainVoice(self.brain).render_boot(session_id='sess-brain')
        assert 'walled client fact' not in text
        assert 'brain fact' in text   # the lane rendered, so absence means gated

    def test_sessionless_canonicalize_ignores_the_last_booted_session(self):
        # A sessionless reader (the dashboard recall probe) gets the outward
        # veil even when the session that booted last sits inside the wall —
        # that session's inward veil would show the walled node.
        self.brain.connect(self.brain_node, self.client_node,
                           relation='extends')
        self.brain.reset_session_activity('sess-walled')
        self.brain.get_or_create_session('sess-walled').set_env(
            cwd='/tmp/client-x', project='client-x')
        results = [{'id': self.brain_node}]
        self.brain.canonicalize_results(results)
        conn_ids = {c['id'] for c in results[0]['connections']}
        assert self.client_node not in conn_ids

    def test_filter_nodes_enumeration_is_gated(self):
        out = self.brain.filter_nodes(field='type', include=['fact'],
                                      rich=False)
        ids = {n['id'] for n in out.get('nodes', [])}
        assert self.client_node not in ids
        assert self.brain_node in ids

    def test_fetch_tools_fails_closed(self, monkeypatch=None):
        # A veil failure withholds results (default-deny) — never passes
        # them ungated. Exercises the REAL execute_tool tail.
        from servers.scales.s1 import fetch_tools
        fake_results = [{'id': self.client_node, 'title': 'walled'}]
        orig = fetch_tools._TOOL_FN_MAP.get('recall_topical')
        fetch_tools._TOOL_FN_MAP['recall_topical'] = \
            lambda brain, **kw: list(fake_results)
        real_veil = type(self.brain).scope_veil

        def _boom(brain_self, session_id):
            raise RuntimeError('veil build failed')
        try:
            type(self.brain).scope_veil = _boom
            out = fetch_tools.execute_tool(
                self.brain, 'recall_topical', {'query': 'x'},
                session_id='sess-brain')
        finally:
            type(self.brain).scope_veil = real_veil
            if orig is not None:
                fetch_tools._TOOL_FN_MAP['recall_topical'] = orig
        assert out['results'] == []   # withheld, not leaked

        # And with a healthy veil, the walled id is filtered, others pass.
        fetch_tools._TOOL_FN_MAP['recall_topical'] = \
            lambda brain, **kw: [{'id': self.client_node},
                                 {'id': self.brain_node}, 'junk-non-dict']
        try:
            out = fetch_tools.execute_tool(
                self.brain, 'recall_topical', {'query': 'x'},
                session_id='sess-brain')
        finally:
            if orig is not None:
                fetch_tools._TOOL_FN_MAP['recall_topical'] = orig
        ids = [r.get('id') for r in out['results'] if isinstance(r, dict)]
        assert self.client_node not in ids
        assert self.brain_node in ids

    def test_recall_result_reports_dropped_count(self):
        # Observability: the main path's result envelope carries the count
        # (present only when non-zero — legacy shape unchanged otherwise).
        out = self.brain.recall(query='walled client fact',
                                session_id='sess-brain', limit=10)
        ids = [n['id'] for n in out.get('results', [])]
        assert self.client_node not in ids


class TestRecallWiringEndToEnd(BrainTestBase):
    """The actual brain.recall path, embedder on: a walled node that wins
    on cosine never comes back; from inside the wall the same recall finds
    it; dropped slots backfill (pre-limit gating)."""
    needs_embedder = True

    def test_recall_gates_isolated_project_both_directions(self):
        walled = _mint(self.brain, 'sess-client', 'client-x',
                       'zebra migration patterns in the savanna')
        _mint(self.brain, 'sess-brain', 'brain',
              'daemon restart procedure notes')
        _set_scopes(self.brain, ISOLATE_CLIENT_X)

        out = self.brain.recall(query='zebra migration savanna',
                                session_id='sess-brain', limit=10)
        ids = [n['id'] for n in out.get('results', [])]
        assert walled not in ids
        assert out.get('_scope_isolated_dropped', 0) >= 1

        out = self.brain.recall(query='zebra migration savanna',
                                session_id='sess-client', limit=10)
        assert walled in [n['id'] for n in out.get('results', [])]


class TestReviewFixes(BrainTestBase):
    """Pins for the veil review's findings — each was a constructed leak."""
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.brain_node = _mint(self.brain, 'sess-brain', 'brain', 'brain fact')
        self.client_node = _mint(self.brain, 'sess-client', 'client-x',
                                 'walled client fact')

    def test_register_door_refuses_invalid_scopes_config(self):
        import json
        with pytest.raises(ValueError):
            self.brain.register_interaction(
                'scopes', template='',
                parameters=json.dumps({'project': {'mode': 'isolatd'}}))
        with pytest.raises(ValueError):
            self.brain.register_interaction(
                'scopes', template='',
                parameters=json.dumps(
                    {'counterpart': {'mode': 'isolated'}}))

    def test_open_override_under_isolated_dimension_is_a_shared_lane(self):
        shared = _mint(self.brain, 'sess-shared', 'shared-infra',
                       'shared infra fact')
        policy = ScopePolicy({'project': {
            'mode': 'isolated', 'overrides': {'shared-infra': 'open'}}})
        # From outside: everything stamped hides EXCEPT the open lane.
        veil = build_veil(self.brain, policy, {'project': 'brain'})
        assert self.client_node in veil
        assert shared not in veil
        # From inside a walled project: the open lane stays visible too.
        veil = build_veil(self.brain, policy, {'project': 'client-x'})
        assert self.brain_node in veil
        assert shared not in veil
        assert self.client_node not in veil

    def test_find_node_by_title_is_veiled(self):
        _set_scopes(self.brain, ISOLATE_CLIENT_X)
        hit = self.brain.find_node_by_title('walled client fact',
                                            threshold=0.3, top_k=5,
                                            session_id='sess-brain')
        ids = {h['id'] for h in (hit if isinstance(hit, list) else
                                 ([hit] if hit else []))}
        assert self.client_node not in ids
        # Inside the wall it resolves.
        hit = self.brain.find_node_by_title('walled client fact',
                                            threshold=0.3, top_k=5,
                                            session_id='sess-client')
        ids = {h['id'] for h in (hit if isinstance(hit, list) else
                                 ([hit] if hit else []))}
        assert self.client_node in ids

    def test_filter_nodes_never_borrows_another_sessions_veil(self):
        # Sessionless call = outward-only veil (default-deny), NEVER the
        # ambient last-seen session's inward veil (the complement — a leak).
        _set_scopes(self.brain, ISOLATE_CLIENT_X)
        out = self.brain.filter_nodes(field='type', include=['fact'],
                                      rich=False)  # no session passed
        ids = {n['id'] for n in out.get('nodes', [])}
        assert self.client_node not in ids

    def test_scrub_node_drops_walled_corrections_by_full_id(self):
        from servers.scopes import scrub_node
        node = {
            'connections': [{'id': self.client_node, 'title': 'walled'},
                            {'id': self.brain_node, 'title': 'fine'}],
            # a walled community's title would leak through the
            # Communities line exactly like a walled neighbor's through Edges
            'communities': [{'id': self.client_node, 'title': 'walled community'},
                            {'id': self.brain_node, 'title': 'fine'}],
            '_corrections': [{'id': self.client_node[:8],
                              'node_id': self.client_node,
                              'content': 'THE WALLED FULL TEXT'}],
        }
        scrub_node(node, frozenset({self.client_node}))
        assert [c['id'] for c in node['connections']] == [self.brain_node]
        assert [c['id'] for c in node['communities']] == [self.brain_node]
        assert node['_corrections'] == []
