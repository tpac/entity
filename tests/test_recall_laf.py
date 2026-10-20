"""LAF v1 recall variant — engine units, DAL door, flag gating (§19 P1).

Component tests (no embedder): lane math, config overlay, id resolution, the
uncapped event_vector_rows DAL door (incl. incremental `since`).
Integration smoke (real embedder + a seeded test brain): flag-on recall
returns laf-scored results end-to-end; flag-off same brain is champion-shaped.
"""
import os
import sys
import unittest
from bisect import bisect_left

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase                      # noqa: E402
from servers.recall_laf import (                                     # noqa: E402
    DEFAULT_CONFIG, LafV1Engine, MAXSIM_VIEWS, _unit, _zscore, idf_scores,
)


class TestLafEngineUnits(BrainTestBase):
    needs_embedder = False

    # ── episodic resolution gauge ──────────────────────────────────────
    class _GaugeBrain:
        """Records what the engine reported, without a graph behind it."""

        def __init__(self):
            self.debug, self.warnings = [], []

        def log_debug(self, event_type, source, **kw):
            self.debug.append((event_type, kw))
            return {'logged': True}

        def _log_warning(self, source, message, context='', ctx=None):
            self.warnings.append((source, message, context))

    def _gauge(self, harvested, credited):
        """Report one resolution of `harvested` ids of which `credited` landed."""
        ids = ['%08x' % i for i in range(harvested)]
        rows = {n: i for i, n in enumerate(ids[:credited])}
        brain = self._GaugeBrain()
        LafV1Engine()._report_resolution(brain, set(ids), rows,
                                         set(ids[credited:]))
        return brain

    def test_resolution_gauge_reads_every_call(self):
        # the reading is a gauge: written at every call, whatever it says
        brain = self._gauge(harvested=40, credited=38)
        self.assertEqual(len(brain.debug), 1)
        event, kw = brain.debug[0]
        self.assertEqual(event, 'laf_resolution')
        self.assertEqual((kw['harvested'], kw['credited'], kw['dropped']),
                         (40, 38, 2))
        self.assertAlmostEqual(kw['share'], 0.05)

    def test_partial_loss_never_warns(self):
        # history outliving nodes is the background, not an incident — and a
        # partial rise has no band to break, by design
        for credited in (38, 20, 2):
            with self.subTest(credited=credited):
                self.assertEqual(self._gauge(40, credited).warnings, [])

    def test_total_loss_warns(self):
        # nothing resolved needs no threshold: cold matrix, or a resolver that
        # stopped resolving
        brain = self._gauge(harvested=25, credited=0)
        self.assertEqual(len(brain.warnings), 1)
        source, message, _ctx = brain.warnings[0]
        self.assertEqual(source, 'laf_resolution_loss')
        self.assertIn('all 25', message)

    def test_small_total_loss_warns_too(self):
        # no minimum-harvest gate: a small harvest resolving to nothing is the
        # same failure as a large one
        self.assertEqual([w[0] for w in self._gauge(3, 0).warnings],
                         ['laf_resolution_loss'])

    def test_empty_harvest_still_reads(self):
        # "the lane ran and found no roles" must not look like "never ran"
        brain = self._gauge(harvested=0, credited=0)
        self.assertEqual(len(brain.debug), 1)
        self.assertEqual(brain.debug[0][1]['harvested'], 0)
        self.assertEqual(brain.warnings, [])      # nothing to resolve, nothing broken

    def test_maxsim_views_from_contract(self):
        # The views derive from EMBEDDING_GROUPS — the live cohort, no dead names.
        self.assertIn('_primary', MAXSIM_VIEWS)
        self.assertIn('title', MAXSIM_VIEWS)
        self.assertNotIn('_situation', MAXSIM_VIEWS)  # situation is its own lane

    def test_unit_normalizes_and_rejects_zero(self):
        v = _unit(np.array([3.0, 4.0], dtype=np.float32).tobytes())
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        self.assertIsNone(_unit(np.zeros(2, dtype=np.float32).tobytes()))
        self.assertIsNone(_unit(None))

    def test_zscore_unit_variance_nan_safe_and_mask(self):
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        z = _zscore(x, 5)
        finite = z[np.isfinite(x)]
        self.assertAlmostEqual(float(finite.std()), 1.0, places=6)
        self.assertEqual(z[3], 0.0)          # NaN slot → neutral, not poison
        # mask form (the laf_metrics.zscore delegation shape): masked-out
        # entries are neutral AND excluded from the statistics
        mask = np.array([True, True, True, True, False])
        zm = _zscore(x, 5, mask=mask)
        self.assertEqual(zm[4], 0.0)
        self.assertAlmostEqual(float(zm[[0, 1, 2]].std()), 1.0, places=6)

    def test_config_defaults_kstore_overlay_and_gain_per_field(self):
        # one gain key per registry field — the P3 fit surface
        for field in ('maxsim', 'pick', 'enc', 'idf', 'sit'):
            self.assertIn('gain_' + field, DEFAULT_CONFIG)

        class FakeBrain:
            # Models the resolver contract: get_interaction_config returns
            # the code default with the override overlaid — total by
            # construction (the key-level merge moved into the accessor).
            def get_interaction_config(self, name):
                assert name == 'recall_laf'
                return {**DEFAULT_CONFIG, 'gain_pick': 0.9}

        cfg = LafV1Engine().config(FakeBrain())
        self.assertEqual(cfg['gain_pick'], 0.9)          # override applied
        self.assertEqual(cfg['gain_enc'], DEFAULT_CONFIG['gain_enc'])

        class BrokenBrain:
            def get_interaction_config(self, name):
                raise RuntimeError('K-store down')
        # fresh engine (config is TTL-cached per engine); broken K-store →
        # defaults, and the failure is logged loud when _log_error exists
        logged = []
        BrokenBrain._log_error = lambda self, *a: logged.append(a)
        self.assertEqual(LafV1Engine().config(BrokenBrain()), DEFAULT_CONFIG)
        self.assertEqual(logged[0][0], 'recall_laf_config')

    def test_idf_scores_rare_token_wins(self):
        # 3 titles: row 0 has the rare token, rows 1-2 share a flood token
        title_tok = {0: frozenset({'spread_activation', 'recall'}),
                     1: frozenset({'recall', 'notes'}),
                     2: frozenset({'recall'})}
        title_df = {'spread_activation': 1, 'recall': 3, 'notes': 1}
        vec = idf_scores('how does spread_activation recall work',
                         title_tok, title_df, 3)
        self.assertGreater(vec[0], vec[1])   # rare-token title dominates
        # 'recall' hits ALL titles → idf log((3+1)/(3+1)) = 0: flood terms are
        # flattened to exactly nothing (the idf2 design), so rows 1-2 score 0
        self.assertEqual(vec[1], 0.0)
        self.assertEqual(vec[2], 0.0)
        # stopwords/short tokens contribute nothing
        self.assertTrue(np.all(idf_scores('a of to', title_tok, title_df, 3) == 0.0))

    def test_resolve_exact_only(self):
        eng = LafV1Engine()
        eng._idx = {'abcdef12': 0}
        self.assertEqual(eng._resolve('abcdef12'), 0)
        self.assertIsNone(eng._resolve('abcdef'))   # prefix is a miss
        self.assertIsNone(eng._resolve('ffffffff'))

    def test_event_vector_rows_uncapped_and_incremental(self):
        dal = self.brain._trace_dal
        vec = np.ones(4, dtype=np.float32).tobytes()
        for i in range(7):
            ts = '2026-01-0%dT00:00:00+00:00' % (i + 1)
            tid = 'trace-%d' % i
            dal.conn.execute(
                "INSERT INTO trace_events (id, chain_id, scale, event_type,"
                " ref_type, ref_id, summary, metadata, session_id, created_at)"
                " VALUES (?, ?, 's0', 'observation', 'user_message',"
                " ?, 's', '{}', 'sess', ?)",
                (tid, 's0-deadbeef-%d' % i, 'r%d' % i, ts))
            dal.conn.execute(
                'INSERT INTO trace_embeddings (trace_id, vector, model)'
                ' VALUES (?, ?, ?)', (tid, vec, 'test'))
        dal.conn.commit()
        rows = dal.event_vector_rows(scale='s0', ref_types=['user_message'])
        self.assertEqual(len(rows), 7)                      # uncapped, all rows
        self.assertEqual(rows[0][2][:10], '2026-01-01')     # ASC order
        # incremental: since the 5th row's timestamp → only rows 6-7
        newer = dal.event_vector_rows(scale='s0', ref_types=['user_message'],
                                      since=rows[4][2])
        self.assertEqual(len(newer), 2)
        # ref_type filter honored
        self.assertEqual(dal.event_vector_rows(scale='s0',
                                               ref_types=['tool_result']), [])


class TestZNormVariants(BrainTestBase):
    """P3.0 normalizer variants (code-review 2026-07-16): properties that
    were previously pinned only by eval/laf/walker/p3_norm.py's sanity
    fixture, which never runs under pytest — a K-store flip activates this
    code in production, so the suite must own it."""
    needs_embedder = False

    def test_support_dense_identical_sparse_bounded_zeros_neutral(self):
        from servers.recall_laf import _zscore_support
        rng = np.random.default_rng(7)
        n = 2000
        dense = rng.normal(0.6, 0.05, n)             # cosine-like, no zeros
        assert np.allclose(_zscore_support(dense, n), _zscore(dense, n))
        sparse = np.zeros(n)
        sparse[rng.choice(n, 40, replace=False)] = rng.uniform(0.5, 0.9, 40)
        assert _zscore(sparse, n).max() > 4.0        # the inflation is real
        z_sup = _zscore_support(sparse, n)
        assert z_sup.max() < 4.0                     # ...and support fixes it
        assert np.all(z_sup[sparse == 0.0] == 0.0)   # zeros stay neutral

    def test_rank_bounded_unconditionally_and_tie_stable(self):
        from servers.recall_laf import _zscore_rank
        rng = np.random.default_rng(7)
        n = 2000
        sparse = np.zeros(n)                          # 98% zero-tie block —
        sparse[rng.choice(n, 40, replace=False)] = 1.0  # the z-of-ranks trap
        for x in (rng.normal(0.6, 0.05, n), sparse):
            zr = _zscore_rank(x, n)
            assert np.abs(zr).max() <= np.sqrt(3) + 1e-9

    def test_variant_dispatch_and_unknown_kind_raises(self):
        from servers.recall_laf import zscore_variant, Z_NORMS
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(zscore_variant(x, 4, kind='current'),
                           _zscore(x, 4))
        assert set(Z_NORMS) == {'current', 'support', 'rank'}
        with self.assertRaises(ValueError):
            zscore_variant(x, 4, kind='suport')

    def test_config_merge_validates_z_norm_loudly(self):
        # A typo'd K-store flip must degrade to 'current' with one loud log —
        # never raise per-query inside scores() (which the caller's fallback
        # would turn into fleet-wide champion mode).
        logged = []

        class TypoBrain:
            def get_interaction_config(self, name):
                return {'z_norm': 'suport'}

            def _log_error(self, tag, e, msg):
                logged.append((tag, str(e)))

        cfg = LafV1Engine().config(TypoBrain())
        assert cfg['z_norm'] == 'current'
        assert logged and logged[0][0] == 'recall_laf_config'

    def test_support_zero_sea_lanes_exclude_contract_zero_lanes(self):
        # proj's 0.0 is a REAL activation (cross-project inhibition); the
        # support rule would zero the whole lane (all-1.0 support, std<1e-9).
        from servers.recall_laf import (SUPPORT_ZERO_SEA_LANES,
                                        _zscore_support)
        assert SUPPORT_ZERO_SEA_LANES == {'pick', 'enc', 'idf'}
        assert 'proj' not in SUPPORT_ZERO_SEA_LANES
        proj_like = np.array([1.0, 1.0, 1.0, 0.0, 0.0, np.nan])
        assert np.all(_zscore_support(proj_like, 6) == 0.0)  # why the gate


class TestProjLane(BrainTestBase):
    """proj lane: session-project provenance as a gain-dialed activation field.

    Design rules pinned here (2026-07-03):
      - missing data is NEUTRAL (NaN, excluded from z-stats) — the sit-lane
        zero-fill lesson as a contract;
      - no session project → the whole lane is inert;
      - gain_proj defaults to 0.0 → wired-but-contributing-nothing until a
        measured gain is registered (the gate corpus is single-project and
        cannot see this lane yet).
    """

    def _engine_with_projects(self):
        """Fresh brain + 3 embedded nodes (alpha / beta / no project)."""
        a = self.brain.remember(
            type='decision', title='Alpha decision on caching strategy',
            content='We cache aggressively at the edge for the dashboard.',
            project='alpha')
        b = self.brain.remember(
            type='decision', title='Beta decision on caching strategy',
            content='We cache conservatively at the origin for the API.',
            project='beta')
        c = self.brain.remember(
            type='decision', title='General decision on caching strategy',
            content='Caching should always be measured before tuning.')
        # third project-carrying node: _zscore's small-sample guard needs >2
        # finite entries or the lane (correctly) stays silent
        self.brain.remember(
            type='decision', title='Gamma decision on caching strategy',
            content='Gamma project caches nothing and recomputes on demand.',
            project='gamma')
        from servers.recall_laf import get_engine
        return get_engine(self.brain), a['id'], b['id'], c['id']

    def test_project_field_semantics(self):
        # unit: same-project 1.0, cross-project 0.0, no-project NaN,
        # no-session-project → all NaN (inert)
        eng = LafV1Engine()
        eng._proj_rows = np.array([0, 1], dtype=np.int64)   # row 2 carries none
        eng._proj_vals = np.array(['alpha', 'beta'], dtype=object)
        vec = eng._project_field('alpha', 3)
        assert vec[0] == 1.0 and vec[1] == 0.0 and np.isnan(vec[2])
        assert np.isnan(eng._project_field('', 3)).all()
        assert np.isnan(eng._project_field(None, 3)).all()

    def test_sit_lane_preserves_nan_for_vectorless_nodes(self):
        # REGRESSION (the sit-lane zero-fill bug): a node with no _situation
        # vector must reach the fusion as NaN — a real 0.0 z-scores ~10σ below
        # the corpus mean and buries the node. Pins the lane contract stated
        # in _fields' docstring for the lane that originally violated it.
        import servers.embedder as embedder
        from servers.recall_laf import get_engine, _unit as _u
        node = self.brain.remember(
            type='lesson', title='Node without a situation field',
            content='This node deliberately has no situation.')
        eng = get_engine(self.brain)
        qv = _u(embedder.embed_query('anything at all'))
        with eng._lock:
            eng._refresh_matrices(self.brain, None)
            nk = eng._refresh_titles(self.brain)
            eng._refresh_projects(self.brain, node_key=nk)
            eng._refresh_traces(self.brain)
            fields = eng._fields(self.brain, 'anything at all', qv,
                                 eng.config(self.brain), eng._n)
        row = eng._idx[node['id']]
        assert np.isnan(fields['sit'][row]), \
            'missing situation vector must be NaN in the sit lane, never 0.0'

    def test_full_matrix_build_resets_proj_arrays(self):
        # REGRESSION: proj arrays are row-keyed; a full matrix rebuild
        # reindexes rows, so the arrays must be force-reset like the titles
        # cache — stale indices would label the WRONG nodes.
        eng, *_ = self._engine_with_projects()
        import servers.embedder as embedder
        from servers.recall_laf import _unit as _u
        qv = _u(embedder.embed_query('caching'))
        eng.scores(self.brain, 'caching', qv, session_project='alpha')
        assert len(eng._proj_rows)                  # populated by the refresh
        with eng._lock:
            eng._full_matrix_build(self.brain, None)
        assert len(eng._proj_rows) == 0 and eng._proj_key is None

    def test_default_gain_zero_is_bit_identical(self):
        import servers.embedder as embedder
        eng, *_ = self._engine_with_projects()
        qv = embedder.embed_query('caching strategy decision')
        base, _ = eng.scores(self.brain, 'caching strategy decision', qv)
        with_proj, _ = eng.scores(self.brain, 'caching strategy decision', qv,
                                  session_project='alpha')
        assert base == with_proj      # gain 0 → lane wired, zero contribution

    def test_nonzero_gain_boosts_same_inhibits_cross_neutral_missing(self):
        import time as _time
        import servers.embedder as embedder
        eng, aid, bid, cid = self._engine_with_projects()
        # Isolate the lane: all gains 0 except proj → score IS the proj z.
        eng._cfg = {**DEFAULT_CONFIG,
                    'gain_maxsim': 0.0, 'gain_pick': 0.0, 'gain_enc': 0.0,
                    'gain_idf': 0.0, 'gain_sit': 0.0, 'gain_proj': 1.0}
        eng._cfg_ts = _time.monotonic()
        qv = embedder.embed_query('caching strategy decision')
        s, _ = eng.scores(self.brain, 'caching strategy decision', qv,
                          session_project='alpha')
        # z over finite {1.0, 0.0}: alpha +1σ, beta −1σ, no-project excluded → 0
        assert s[aid] > s[cid] > s[bid]
        # inert without a session project: everyone z=0 → equal scores
        s0, _ = eng.scores(self.brain, 'caching strategy decision', qv)
        assert s0[aid] == s0[bid] == s0[cid]

    def test_proj_ordering_survives_support_z_flip(self):
        # REGRESSION (code-review 2026-07-16): under z_norm='support' the
        # scores() z-loop must route proj through plain z (lane gate) — the
        # support rule on a {1.0, 0.0, NaN} lane would drop the 0.0s, leave
        # an all-1.0 support with std<1e-9, and zero the lane, silently
        # killing cross-project inhibition.
        import time as _time
        import servers.embedder as embedder
        eng, aid, bid, cid = self._engine_with_projects()
        eng._cfg = {**DEFAULT_CONFIG, 'z_norm': 'support',
                    'gain_maxsim': 0.0, 'gain_pick': 0.0, 'gain_enc': 0.0,
                    'gain_idf': 0.0, 'gain_sit': 0.0, 'gain_proj': 1.0}
        eng._cfg_ts = _time.monotonic()
        qv = embedder.embed_query('caching strategy decision')
        s, _ = eng.scores(self.brain, 'caching strategy decision', qv,
                          session_project='alpha')
        assert s[aid] > s[cid] > s[bid], \
            'support-z flip zeroed the proj lane — the lane gate is broken'

    def test_project_rows_kv_wins_over_column(self):
        # migration-proof read: node_metadata_kv['project'] beats the legacy
        # column once the migration lands, with no engine change.
        _, aid, _, cid = self._engine_with_projects()
        self.brain._meta_kv.set_many(aid, {'project': 'kv-project'})
        rows = dict(self.brain._nodes.project_rows())
        assert rows[aid] == 'kv-project'   # kv overrides column 'alpha'
        assert cid not in rows             # no project anywhere → absent


class _LafEngineFixtures(BrainTestBase):
    """Shared substrate for the engine-level classes below: backdated nodes,
    hand-written traces, and an engine whose caches are built over them.

    Not collected (no `Test` prefix, no test methods) — the classes that
    inherit it own the assertions.
    """

    EARLY = '2026-01-01T00:00:00.000000+00:00'
    MID = '2026-02-01T00:00:00.000000+00:00'
    LATE = '2026-03-01T00:00:00.000000+00:00'
    FUTURE = '2030-01-01T00:00:00.000000+00:00'

    def _mk_node(self, title, content, created):
        node = self.brain.remember(type='lesson', title=title, content=content)
        self.brain._nodes.conn.execute(
            'UPDATE nodes SET created_at=? WHERE id=?', (created, node['id']))
        self.brain._nodes.conn.commit()
        return node['id']

    def _mk_trace(self, chain_id, created, vec, session='sess-asof',
                  scale='s0', ref_type='user_message', ref_id='r'):
        dal = self.brain._trace_dal
        tid = 'asof-%s-%s' % (chain_id, created[:7])
        dal.conn.execute(
            'INSERT INTO trace_events (id, chain_id, scale, event_type,'
            " ref_type, ref_id, summary, metadata, session_id, created_at)"
            " VALUES (?, ?, ?, 'observation', ?, ?, 's', '{}', ?, ?)",
            (tid, chain_id, scale, ref_type, ref_id, session, created))
        if vec is not None:
            dal.conn.execute(
                'INSERT INTO trace_embeddings (trace_id, vector, model)'
                ' VALUES (?, ?, ?)', (tid, vec.astype(np.float32).tobytes(),
                                      'test'))
        dal.conn.commit()
        return tid

    def _fresh_engine(self, qv):
        """New engine with caches built over the CURRENT (backdated) rows."""
        eng = LafV1Engine()
        eng.warm(self.brain)
        return eng


class TestAsOfTimeTravel(_LafEngineFixtures):
    """as_of read-side time travel — the §20.11 test contract.

    (a) as_of=None is the identical live path (inert by construction; the
        pre-existing classes above pin its behavior), (b) as_of=now ≡ None
        exactly, (c) a node/trace/role-row created after as_of contributes
        nothing, (d) the walker cross-check runs offline (eval/laf/).
    """

    def test_asof_now_equals_none_exactly(self):
        # contract (b): a far-future as_of masks nothing → bit-identical output
        import servers.embedder as embedder
        for i, created in enumerate((self.EARLY, self.MID, self.LATE)):
            self._mk_node('Caching decision variant %d spread_activation' % i,
                          'Content about caching strategy %d.' % i, created)
        qv = embedder.embed_query('caching strategy spread_activation')
        eng = self._fresh_engine(qv)
        live_map, live_tel = eng.scores(self.brain, 'caching strategy', qv)
        asof_map, asof_tel = eng.scores(self.brain, 'caching strategy', qv,
                                        as_of=self.FUTURE)
        self.assertEqual(live_map, asof_map)
        self.assertEqual(live_tel, asof_tel)

    def test_asof_excludes_future_nodes(self):
        # contract (c), node universe: created-after-as_of nodes are absent
        # from the result; z-stats run over the masked universe only
        import servers.embedder as embedder
        early = self._mk_node('Early caching decision',
                              'Cache aggressively at the edge.', self.EARLY)
        late = self._mk_node('Late caching decision',
                             'Cache conservatively at the origin.', self.LATE)
        qv = embedder.embed_query('caching decision')
        eng = self._fresh_engine(qv)
        live_map, _ = eng.scores(self.brain, 'caching decision', qv)
        self.assertIn(early, live_map)
        self.assertIn(late, live_map)
        asof_map, asof_tel = eng.scores(self.brain, 'caching decision', qv,
                                        as_of=self.MID)
        self.assertIn(early, asof_map)
        self.assertNotIn(late, asof_map)
        self.assertNotIn(late, asof_tel or {})
        # nothing existed before EARLY → empty result, loud-and-empty not weird
        self.assertEqual(eng.scores(self.brain, 'caching decision', qv,
                                    as_of='2020-01-01T00:00:00+00:00'),
                         ({}, None))

    def test_asof_rejects_non_iso(self):
        import servers.embedder as embedder
        self._mk_node('Any node title here', 'Body.', self.EARLY)
        qv = embedder.embed_query('any')
        eng = self._fresh_engine(qv)
        with self.assertRaises(ValueError):
            eng.scores(self.brain, 'any', qv, as_of='last tuesday')

    def test_asof_idf_df_time_travels(self):
        # contract (c), idf lane: the df denominator counts only titles that
        # existed at as_of — bisect_left (strictly-before), walker semantics
        self._mk_node('zetatoken early note', 'Body one.', self.EARLY)
        self._mk_node('zetatoken late note', 'Body two.', self.LATE)
        self._mk_node('Unrelated filler title', 'Body three.', self.EARLY)
        import servers.embedder as embedder
        qv = embedder.embed_query('zetatoken')
        eng = self._fresh_engine(qv)
        n = eng._n
        # at MID: one zetatoken title of two total early titles
        self.assertEqual(
            bisect_left(eng._token_created['zetatoken'], self.MID), 1)
        self.assertEqual(bisect_left(eng._title_created, self.MID), 2)
        # two-token query: idf_scores normalizes by the query's total idf
        # mass, so a single-token query is always 1.0 — the df shift only
        # shows in the RATIO between tokens
        q = 'zetatoken filler'
        vec_mid = eng._idf_asof(q, n, self.MID)
        vec_live = eng._idf_asof(q, n, self.FUTURE)
        row_early = next(i for i in range(n)
                         if 'zetatoken' in eng._title_tok.get(i, frozenset())
                         and 'early' in eng._title_tok.get(i, frozenset()))
        # at MID both tokens have df=1/n=2 → the zeta-only title gets 0.5;
        # live df(zeta)=2 vs df(filler)=1 over n=3 → a smaller share
        self.assertAlmostEqual(float(vec_mid[row_early]), 0.5, places=9)
        self.assertGreater(0.5, float(vec_live[row_early]))
        # far-future as-of ≡ the live formula fed the live df
        live = idf_scores(q, eng._title_tok, eng._title_df, n)
        np.testing.assert_array_equal(vec_live, live)

    def test_asof_masks_future_traces_and_roles(self):
        # contract (c), trace universe: a trace created after as_of seeds no
        # moment; a role row (surface pick) created after as_of joins nothing
        # even when its moment is visible
        import json
        import servers.embedder as embedder
        node = self._mk_node('Episodic target node about walruses',
                             'Walrus habits.', self.EARLY)
        qv = _unit(embedder.embed_query('walrus habits'))
        # moment 1: trace LATE (masked at MID), surface row LATE
        self._mk_trace('s0-deadbeef-3', self.LATE, qv)
        self._mk_trace('s1r-deadbeef-3', self.LATE, None, scale='s1',
                       ref_type='surface_selected',
                       ref_id=json.dumps([node[:8]]))
        # moment 2: trace EARLY (visible at MID), but its surface row was
        # written LATE — the roles_for_moments door must drop it
        self._mk_trace('s0-deadbeef-5', self.EARLY, qv)
        self._mk_trace('s1r-deadbeef-5', self.LATE, None, scale='s1',
                       ref_type='surface_selected',
                       ref_id=json.dumps([node[:8]]))
        eng = self._fresh_engine(qv)
        cfg = eng.config(self.brain)
        n = eng._n
        row = eng._idx[node]
        pick_live, _ = eng._episodic_vectors(self.brain, qv, cfg, n)
        self.assertGreater(pick_live[row], 0.0)
        _, trace_mask = eng._asof_masks(self.MID, n)
        pick_asof, enc_asof = eng._episodic_vectors(
            self.brain, qv, cfg, n, as_of=self.MID, trace_mask=trace_mask)
        self.assertEqual(float(pick_asof[row]), 0.0)
        self.assertEqual(float(enc_asof.sum()), 0.0)

    def test_asof_roles_deep_history_window(self):
        # Regression (2026-08-07 fetch-then-filter sweep): the stream pull is
        # newest-first LIMIT pull_limit; the old Python `<= as_of` post-filter
        # ran AFTER the limit had already kept the wrong (newest) end — a
        # session with more post-as_of rows than pull_limit gave empty role
        # sets silently. The bound now rides the SQL, positioning the window
        # AT as_of.
        import json
        from servers.recall_laf import roles_for_moments
        node = self._mk_node('Deep history role target', 'Body.', self.EARLY)
        # the moment's surface pick, before as_of
        self._mk_trace('s1r-deadbeef-5', self.EARLY, None, scale='s1',
                       ref_type='surface_selected',
                       ref_id=json.dumps([node[:8]]))
        # a tail of NEWER surface rows, more than pull_limit of them — the
        # old pull saw only these
        for k in range(6):
            self._mk_trace('s1r-deadbeef-%d' % (50 + k), self.LATE, None,
                           scale='s1', ref_type='surface_selected',
                           ref_id=json.dumps(['ffffffff']))
        calls = []
        orig = self.brain._log_error
        self.brain._log_error = lambda *a, **kw: calls.append(a[0])
        try:
            recs = roles_for_moments(
                self.brain, {('sess-asof', 'deadbeef', 5): 0.9},
                window_turns=1, pull_limit=4, as_of=self.MID)
        finally:
            self.brain._log_error = orig
        self.assertEqual(len(recs), 1)
        self.assertIn(node[:8], recs[0]['picked'])
        # one row before as_of < pull_limit → no truncation false-positive
        self.assertNotIn('laf_roles_pull_truncated', calls)

    def test_asof_roles_pull_truncation_is_loud(self):
        # A window that is still full AT as_of is a clipped coverage read —
        # under replay that means wrong numbers, so it must flag loudly
        # (limit+1 probe → laf_roles_pull_truncated), never pass as complete.
        import json
        from servers.recall_laf import roles_for_moments
        for k in range(4):
            self._mk_trace('s1r-deadbeef-%d' % k,
                           '2026-01-01T00:00:0%d.000000+00:00' % k, None,
                           scale='s1', ref_type='surface_selected',
                           ref_id=json.dumps(['aaaaaaa%d' % k]))
        calls = []
        orig = self.brain._log_error
        self.brain._log_error = lambda *a, **kw: calls.append(a[0])
        try:
            recs = roles_for_moments(
                self.brain, {('sess-asof', 'deadbeef', 2): 0.5},
                window_turns=1, pull_limit=2, as_of=self.MID)
        finally:
            self.brain._log_error = orig
        self.assertEqual(len(recs), 1)
        self.assertIn('laf_roles_pull_truncated', calls)


class TestSurvivorCredit(_LafEngineFixtures):
    """Absorbed role ids credit their live survivor's row, never the floor.

    Role ids come from traces, so they are history (docs/TRACE-NODE-RESOLUTION.md):
    S2 consolidation absorbs A into B and archives A, leaving A with no row in the
    live-only matrix. Before the survivor walk the lookup returned None and the
    moment's evidence was dropped silently — B never inherited the activation
    history its own content earned. Reuses the as_of fixtures (_mk_node/_mk_trace/
    _fresh_engine) — same substrate, different contract.
    """

    def _absorbed_moment(self, survivor_id=None):
        """Node A picked in a past moment, then archived (into `survivor_id`)."""
        import json
        import servers.embedder as embedder
        dead = self._mk_node('Absorbed node about walruses', 'Walrus habits.',
                             self.EARLY)
        qv = _unit(embedder.embed_query('walrus habits'))
        self._mk_trace('s0-deadbeef-7', self.EARLY, qv)
        self._mk_trace('s1r-deadbeef-7', self.EARLY, None, scale='s1',
                       ref_type='surface_selected', ref_id=json.dumps([dead[:8]]))
        self.brain.archive_node(dead, archived_by='test',
                                reason='absorbed', survivor_id=survivor_id)
        return dead, qv

    def test_absorbed_pick_credits_survivor_row(self):
        surv = self._mk_node('Surviving node about walruses',
                             'Walrus habits, consolidated.', self.EARLY)
        dead, qv = self._absorbed_moment(survivor_id=surv)
        eng = self._fresh_engine(qv)
        n = eng._n
        self.assertIsNone(eng._resolve(dead[:8]))       # no row: A is archived
        pick, _ = eng._episodic_vectors(self.brain, qv, eng.config(self.brain), n)
        self.assertGreater(pick[eng._idx[surv]], 0.0)   # …credited to B instead

    def test_retired_node_stays_dropped(self):
        # archived with NO survivor is a retirement, not an identity claim —
        # there is nothing live to credit, so the evidence stays on the floor
        dead, qv = self._absorbed_moment(survivor_id=None)
        eng = self._fresh_engine(qv)
        pick, enc = eng._episodic_vectors(self.brain, qv, eng.config(self.brain),
                                          eng._n)
        self.assertEqual(float(pick.sum()), 0.0)
        self.assertEqual(float(enc.sum()), 0.0)

    def test_gauge_fires_from_the_real_call_path(self):
        # The unit tests drive _report_resolution directly, so swapping the
        # rows/unresolvable arguments at the call site would invert credited
        # and dropped with every one of them still green. Drive the real path
        # with one harvested id, archived and unrecoverable: the only correct
        # reading is credited=0, dropped=1 — the inverted one is its mirror.
        _dead, qv = self._absorbed_moment(survivor_id=None)
        eng = self._fresh_engine(qv)
        seen = []
        self.brain.log_debug = lambda event, src, **kw: seen.append((event, kw))
        try:
            eng._episodic_vectors(self.brain, qv, eng.config(self.brain), eng._n)
        finally:
            del self.brain.log_debug
        readings = [kw for event, kw in seen if event == 'laf_resolution']
        self.assertEqual(len(readings), 1, 'the real path wrote no reading')
        self.assertEqual((readings[0]['harvested'], readings[0]['credited'],
                          readings[0]['dropped']), (1, 0, 1))

    def test_role_rows_walks_chain_and_skips_live(self):
        # A→B→C: the walk is transitive, and a LIVE id never enters it (an
        # all-live id set must cost zero resolve_live queries)
        from servers.recall_laf import role_rows
        c = self._mk_node('Chain terminal', 'C.', self.EARLY)
        b = self._mk_node('Chain middle', 'B.', self.EARLY)
        a = self._mk_node('Chain head', 'A.', self.EARLY)
        self.brain.archive_node(b, archived_by='test', survivor_id=c)
        self.brain.archive_node(a, archived_by='test', survivor_id=b)
        rows = {c: 0}
        calls = []
        orig = self.brain.resolve_live
        self.brain.resolve_live = lambda ids, **kw: (calls.append(sorted(ids))
                                                     or orig(ids, **kw))
        try:
            got, orphans = role_rows(self.brain, [a, c], rows.get)
        finally:
            self.brain.resolve_live = orig
        self.assertEqual(got, {a: 0, c: 0})            # A→B→C lands on C's row
        self.assertEqual(orphans, set())               # both landed
        self.assertEqual(calls, [[a]])                 # live C never walked


class TestLafIntegration(unittest.TestCase):
    """End-to-end: recall is laf-scored; a scorer failure returns empty + loud.

    Runs on a seeded test brain (never the live DB) with the real embedder.
    Unconditional — the substrate is built from the repo seed pack, so there
    is no machine where this smoke is unavailable and silently skipped.
    """

    @classmethod
    def setUpClass(cls):
        from tests.isolated_brain import brain_env
        # needs_embedder=True: the template carries its document vectors, but
        # LAF still needs the model to encode each query.
        cls._env = brain_env(needs_embedder=True)
        cls._env.__enter__()
        cls.brain = cls._env.brain
        cls._seed_episodes(cls.brain)

    @staticmethod
    def _seed_episodes(brain):
        """Give the episodic lane something to ingest.

        The seed pack is nodes and edges; a fresh brain has no conversation
        behind it, so `_tr_blocks` would be legitimately empty and the
        assertion on it would pass only on a brain that had lived. Write s0
        turns and let the PRODUCTION producer embed them — rendering and
        vectors then match what the lane reads in production, at the live
        model, rather than a hand-rolled blob of the wrong dimension.
        """
        from servers import embed_queue
        from servers.clock import iso_now
        turns = [
            'I asked how the brain decides what is worth keeping.',
            'We agreed a correction from the operator is the most valuable input.',
            'Verify before claiming — I check the code before saying what it does.',
            'The seed pack is small on purpose; a young graph is not a broken one.',
        ]
        dal = brain._trace_dal
        for i, text in enumerate(turns):
            dal.conn.execute(
                "INSERT INTO trace_events (id, chain_id, scale, event_type,"
                " ref_type, ref_id, summary, metadata, session_id, created_at)"
                " VALUES (?, ?, 's0', 'observation', 'user_message',"
                " ?, ?, '{}', 'laf-smoke', ?)",
                ('lafsmoke%d' % i, 's0-lafsmoke-%d' % i, 'r%d' % i, text,
                 iso_now()))
        dal.conn.commit()
        embed_queue._drain_trace_embeddings_once(brain)

    @classmethod
    def tearDownClass(cls):
        cls._env.__exit__(None, None, None)

    def _recall(self, tag='laf'):
        # distinct queries per test → the 5s result cache can't cross them
        return self.brain.recall(
            query='how I grow and keep what matters %s' % tag, limit=10)

    def test_scores_via_field(self):
        res = self._recall()
        self.assertEqual(res['_recall_mode'], 'laf_v1')
        self.assertTrue(res['results'], 'laf recall returned nothing')
        for r in res['results']:
            self.assertEqual(r.get('_discovery'), 'laf_v1')
            # the (0,1) range contract the champion's floors/boosts assume
            self.assertGreater(r['effective_activation'], 0.0)
            self.assertLessEqual(r['effective_activation'], 1.0)
        # per-candidate field telemetry rides the result (P2 walker feed)
        self.assertIn('_laf_fields', res)
        some_fields = next(iter(res['_laf_fields'].values()))
        self.assertEqual(set(some_fields),
                         {'maxsim', 'pick', 'enc', 'idf', 'sit', 'proj'})
        # engine cached on the brain, matrices + trace blocks resident
        eng = self.brain._laf_engine
        self.assertGreater(eng._n, 0)
        self.assertTrue(eng._tr_blocks)
        # laf results visible in the source breakdown telemetry
        self.assertGreater(
            res['_retrieval_stats']['source_breakdown'].get('laf_v1', 0), 0)


def _seed_vec_rows(brain, n_nodes, start=0):
    """n_nodes nodes, each with one identical '_primary' vector under model
    'test' — the matrix substrate for the engine-cache classes below."""
    blob = (np.arange(768, dtype=np.float32) + 1.0).tobytes()
    for i in range(start, start + n_nodes):
        nid = 'capnode%09d' % i
        brain.conn.execute(
            "INSERT OR IGNORE INTO nodes (id, type, title, content, created_at)"
            " VALUES (?, 'fact', ?, 'body', '2026-01-01T00:00:00+00:00')",
            (nid, 'Capacity growth node %d' % i))
        brain.conn.execute(
            'INSERT OR REPLACE INTO node_enrichments '
            '(node_id, vector_type, text, embedding, model, created_at) '
            # ISO-T literal, never datetime('now') — the space-separated
            # form sorts before 'T' and corrupts as-of lexicographic
            # masks (CLAUDE.md time-window rule).
            "VALUES (?, '_primary', '', ?, 'test', '2026-01-01T00:00:00+00:00')",
            (nid, blob))
    brain.conn.commit()
    if hasattr(brain._vec_dal, 'reload'):
        brain._vec_dal.reload()


class TestCapacityGrowth(BrainTestBase):
    """The incremental append that crosses _cap must not index the pre-growth
    matrix (Python resolves the subscript target BEFORE the index expression's
    side effects — the single-expression form raised IndexError at exactly
    row 64; found 2026-07-17, 20-item pooled build)."""

    needs_embedder = False

    def test_incremental_append_across_capacity_boundary(self):
        eng = LafV1Engine()
        _seed_vec_rows(self.brain, 3)
        with eng._lock:
            eng._refresh_matrices(self.brain, 'test')
        self.assertGreaterEqual(eng._cap, 64)
        # append enough new rows to cross the 64-row minimum capacity
        _seed_vec_rows(self.brain, 80, start=3)
        with eng._lock:
            eng._refresh_matrices(self.brain, 'test')   # raised IndexError pre-fix
        self.assertGreaterEqual(eng._n, 80)
        self.assertGreater(eng._cap, 64)


class TestWarm(BrainTestBase):
    """warm() builds every cache scores() reads — Brain.warm_up runs it at
    daemon boot so the first recall pays no full matrix build."""

    needs_embedder = False

    def test_warm_leaves_no_matrix_build_for_scores(self):
        _seed_vec_rows(self.brain, 3)
        eng = LafV1Engine()
        self.assertEqual(eng.warm(self.brain, 'test'), 3)
        builds = []
        full_build = eng._full_matrix_build
        eng._full_matrix_build = lambda *a, **k: (builds.append(1),
                                                  full_build(*a, **k))
        qv = (np.arange(768, dtype=np.float32) + 1.0).tolist()
        scores, _ = eng.scores(self.brain, 'capacity growth node', qv,
                               model='test')
        self.assertEqual(builds, [])
        self.assertEqual(len(scores), 3)


class TestWarmUp(unittest.TestCase):
    """Brain.warm_up's LAF step. The matrices are not keyed by model, so a
    warm under any model value other than recall's would be reused silently
    forever — this pins the two to the same value. The Anthropic step is
    stubbed: no network from tests."""

    @classmethod
    def setUpClass(cls):
        from tests.isolated_brain import brain_env
        cls._env = brain_env(needs_embedder=True)
        cls._env.__enter__()
        cls.brain = cls._env.brain
        cls.brain._ensure_anthropic_client = lambda: None
        cls.brain.warm_anthropic_connection = lambda: True

    @classmethod
    def tearDownClass(cls):
        cls._env.__exit__(None, None, None)

    def test_warms_under_the_model_recall_uses(self):
        from servers.recall_laf import get_engine
        eng = get_engine(self.brain)
        models = []
        refresh = eng._refresh_matrices
        eng._refresh_matrices = lambda brain, model: (models.append(model),
                                                      refresh(brain, model))
        timings = self.brain.warm_up()
        self.assertGreater(eng._n, 0)
        self.assertEqual(timings['laf_rows'], eng._n)
        self.brain.recall(query='what the brain builds at boot', limit=5)
        self.assertEqual(len(models), 2)            # the warm, then the recall
        self.assertIsNotNone(models[0])
        self.assertEqual(models[0], models[1])

    def test_embedder_down_skips_the_laf_step(self):
        from unittest import mock
        from servers import embedder
        with mock.patch.object(embedder, 'is_ready', return_value=False):
            timings = self.brain.warm_up()
        self.assertNotIn('laf_rows', timings)
        self.assertNotIn('laf_error', timings)
        self.assertIn('total_ms', timings)


class TestMomentStack(BrainTestBase):
    """§20.17/§20.18 moment slot lanes — dormancy, stack assembly, gain table.

    The dormancy invariant is the ship condition: with the default config
    (moment_K=0, moment_gains={}) the moment path is unreachable and scores()
    is bit-identical to the moment-less engine. An arm is a gain table, not
    code — A0f (fitted-K0) and the additive shape are pinned here as pure
    config values.
    """

    SESS = 'sess-moment'
    T = ['2026-05-01T00:00:%02d.000000+00:00' % i for i in range(20)]

    def _mk_row(self, tid, ref_type, content, created, with_vec=True,
                host_turn=None, chain='s0-momsess-1'):
        """One conversational trace row as the S0 write door stores it
        (event_type 'K', content in metadata) + its embedding row.
        `host_turn` stamps the host's name for the turn, as the prompt
        hook's write door does."""
        import json as _json
        import servers.embedder as embedder
        from servers.trace_contract import HOST_TURN_FIELD
        meta = {'content': content}
        if host_turn:
            meta[HOST_TURN_FIELD] = host_turn
        dal = self.brain._trace_dal
        dal.conn.execute(
            "INSERT INTO trace_events (id, chain_id, scale, event_type,"
            " ref_type, ref_id, summary, metadata, session_id, created_at)"
            " VALUES (?, ?, 's0', 'K', ?, 'r', ?, ?, ?, ?)",
            (tid, chain, ref_type, content[:80],
             _json.dumps(meta), self.SESS, created))
        if with_vec:
            blob = embedder.embed_query(content)     # packed float32 blob
            if not isinstance(blob, (bytes, bytearray)):
                blob = np.asarray(blob, dtype=np.float32).tobytes()
            dal.conn.execute(
                'INSERT INTO trace_embeddings (trace_id, vector, model)'
                ' VALUES (?, ?, ?)', (tid, blob, 'test'))
        dal.conn.commit()

    def _mk_turn(self, i, user_text, anchor_text=None, user_vec=True):
        self._mk_row('mtu%02d' % i, 'user_message', user_text,
                     self.T[2 * i], with_vec=user_vec)
        if anchor_text is not None:
            self._mk_row('mta%02d' % i, 'assistant_message', anchor_text,
                         self.T[2 * i + 1])

    def test_correspondent_rows_dropped_from_moment_stack(self):
        # A dial-on delivery row riding in a turn (it has no vector in the
        # pinned matrix) never becomes an entry and holds no slot of its
        # own — counted under its own ledger key, apart from the harness's.
        from unittest import mock
        from servers import trace_contract as tc
        self._mk_turn(0, 'real question', 'real answer')
        self._mk_row('mtd01', 'thalamus_delivery',
                     'delivered 1 thalamus item(s) at stop',
                     self.T[3], with_vec=False)
        self._mk_turn(2, 'second question', 'second answer')
        eng = self._engine()
        with mock.patch.object(
                tc, 'CONVERSATIONAL_REF_TYPES',
                tc.CONVERSATIONAL_REF_TYPES + ('thalamus_delivery',)):
            stack, ledger = eng._moment_stack(self.brain, self.SESS, 3)
        self.assertEqual(ledger.get('correspondent_dropped', 0), 1)
        self.assertEqual(ledger['machine_dropped'], 0)
        # Both real turns still pair up; the delivery holds no slot.
        self.assertEqual({(s, j) for s, j, _v, _t in stack},
                         {('o', 1), ('a', 1), ('o', 2), ('a', 2)})

    def _engine(self, moment_K=None, moment_gains=None):
        import time as _t
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(self.brain, None)
            nk = eng._refresh_titles(self.brain)
            eng._refresh_projects(self.brain, node_key=nk)
            eng._refresh_traces(self.brain)
        over = {}
        if moment_K is not None:
            over['moment_K'] = moment_K
        if moment_gains is not None:
            over['moment_gains'] = moment_gains
        eng._cfg = dict(DEFAULT_CONFIG, **over)
        eng._cfg_ts = _t.monotonic()
        return eng

    def test_defaults_dormant_and_bit_identical(self):
        # the shipped defaults are falsy — the moment path is unreachable
        self.assertEqual(DEFAULT_CONFIG['moment_K'], 0)
        self.assertEqual(DEFAULT_CONFIG['moment_gains'], {})
        import servers.embedder as embedder
        self.brain.remember(type='lesson', title='Cache strategy note',
                            content='Cache aggressively at the edge.')
        self._mk_turn(0, 'how should caching work', 'aggressively, per note')
        qv = embedder.embed_query('caching')
        eng = self._engine()
        with_sess = eng.scores(self.brain, 'caching', qv, session_id=self.SESS)
        without = eng.scores(self.brain, 'caching', qv)
        self.assertEqual(with_sess, without)
        self.assertIsNone(eng._last_moment_ledger)

    def test_trace_cache_carries_ids_and_ref_types(self):
        self._mk_turn(0, 'first question here', 'first answer here')
        rows = self.brain._trace_dal.event_vector_rows(
            scale='s0', ref_types=['user_message', 'assistant_message'])
        self.assertEqual({(r[4], r[5]) for r in rows},
                         {('mtu00', 'user_message'),
                          ('mta00', 'assistant_message')})
        eng = self._engine()
        self.assertIn('mtu00', eng._tr_ids)
        vec = eng._tr_vec(eng._tr_ids['mtu00'])
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_stack_assembly_live_edge_machine_and_asof(self):
        self._mk_turn(0, 'tell me about submarines', 'sonar arrays, mostly')
        self._mk_turn(1, '<task-notification> background task done',
                      '(watching)')
        self._mk_turn(2, 'and their propulsion?', 'nuclear or diesel')
        self._mk_turn(3, 'what about ballast tanks')       # in-flight prompt
        eng = self._engine()
        stack, ledger = eng._moment_stack(self.brain, self.SESS, 3)
        # trailing answer-less user turn = the live edge → dropped
        self.assertEqual(ledger['live_edge_dropped'], 1)
        self.assertEqual(ledger['machine_dropped'], 1)
        self.assertEqual(ledger['missing_vec'], 0)
        # j=1 ← turn 2 (both sides), j=2 ← machine turn (anchor side only),
        # j=3 ← turn 0 (both sides)
        self.assertEqual({(s, j) for s, j, _v, _t in stack},
                         {('o', 1), ('a', 1), ('a', 2), ('o', 3), ('a', 3)})
        texts = {(s, j): t for s, j, _v, t in stack}
        self.assertEqual(texts[('o', 1)], 'and their propulsion?')
        self.assertEqual(texts[('a', 2)], '(watching)')
        for _s, _j, v, _t in stack:
            self.assertIsNotNone(v)
        # as_of strictly before turn 2's user row → window is turns 0-1,
        # complete, no live-edge drop
        stack2, ledger2 = eng._moment_stack(self.brain, self.SESS, 3,
                                            as_of=self.T[4])
        self.assertEqual(ledger2['live_edge_dropped'], 0)
        self.assertEqual({(s, j) for s, j, _v, _t in stack2},
                         {('a', 1), ('o', 2), ('a', 2)})

    def test_message_typed_mid_turn_shares_its_openers_slot(self):
        # The host named both messages one turn: the steer rides in its
        # opener's slot as a second o entry. With no cue, the in-flight
        # turn, both its messages, is the live edge.
        self._mk_row('mtu00', 'user_message', 'tell me about submarines',
                     self.T[0], host_turn='p1')
        self._mk_row('mtu01', 'user_message', 'and their sonar too',
                     self.T[1], host_turn='p1')
        self._mk_row('mta01', 'assistant_message', 'sonar arrays, mostly',
                     self.T[2])
        self._mk_row('mtu03', 'user_message', 'and propulsion?',
                     self.T[3], host_turn='p2')
        self._mk_row('mta03', 'assistant_message', 'nuclear or diesel',
                     self.T[4])
        self._mk_row('mtu05', 'user_message', 'what about ballast',
                     self.T[5], host_turn='p3')
        self._mk_row('mtu06', 'user_message', 'and trim tanks',
                     self.T[6], host_turn='p3')
        eng = self._engine()
        stack, ledger = eng._moment_stack(self.brain, self.SESS, 3)
        self.assertEqual(ledger['live_edge_dropped'], 1)
        self.assertEqual(ledger['turns'], 2)
        self.assertEqual(sorted((s, j, t) for s, j, _v, t in stack), [
            ('a', 1, 'nuclear or diesel'), ('a', 2, 'sonar arrays, mostly'),
            ('o', 1, 'and propulsion?'), ('o', 2, 'and their sonar too'),
            ('o', 2, 'tell me about submarines')])

    def test_unstamped_operator_rows_keep_a_slot_each(self):
        self._mk_turn(0, 'delete the old folder')
        self._mk_turn(1, 'wait, list it first', 'listed, nothing deleted')
        self._mk_turn(2, 'good', 'done')
        eng = self._engine()
        stack, _ledger = eng._moment_stack(self.brain, self.SESS, 3)
        self.assertEqual({(s, j) for s, j, _v, _t in stack},
                         {('o', 1), ('a', 1), ('o', 2), ('a', 2), ('o', 3)})

    def test_a_slots_stronger_message_counts_even_when_it_came_first(self):
        # Two operator messages in one slot: the slot keeps the stronger per
        # node, so the submarine question lifts the submarine node although
        # an unrelated steer came after it in the same turn.
        import servers.embedder as embedder
        sub = self.brain.remember(
            type='lesson', title='Submarine sonar arrays',
            content='Attack submarines mount bow sonar arrays.')['id']
        ada = self.brain.remember(
            type='lesson', title='Garden tomato watering',
            content='Water tomato plants at the roots, mornings.')['id']
        self._mk_row('mtu00', 'user_message', 'tell me about attack submarines',
                     self.T[0], host_turn='p1')
        self._mk_row('mtu01', 'user_message', 'also keep it short',
                     self.T[1], host_turn='p1')
        self._mk_row('mta01', 'assistant_message', 'short version follows',
                     self.T[2])
        query = 'watering the garden tomatoes'
        qv = embedder.embed_query(query)
        eng = self._engine(moment_K=1, moment_gains={'maxsim_o1': 8.0})
        mom_map, _ = eng.scores(self.brain, query, qv, session_id=self.SESS)
        self.assertGreater(mom_map[sub], mom_map[ada])

    def test_mid_turn_cue_keeps_its_opener_at_o1(self):
        # A opens a turn, B is typed while it runs, C comes after the reply.
        # Live, the window ends just before the cue: at B, A is the newest
        # turn's message at o1; at C, A and B share o1 with the reply.
        # Replay (as_of, no cue) at B drops A's open turn, as it always did.
        self._mk_row('mtu00', 'user_message', 'tell me about submarines',
                     self.T[0], host_turn='p0')
        self._mk_row('mta00', 'assistant_message', 'sonar arrays, mostly',
                     self.T[1])
        self._mk_row('mtu02', 'user_message', 'what about ballast',
                     self.T[2], host_turn='p1')
        self._mk_row('mtu03', 'user_message', 'and trim tanks',
                     self.T[3], host_turn='p1')
        self._mk_row('mta03', 'assistant_message', 'both flood to dive',
                     self.T[4])
        self._mk_row('mtu05', 'user_message', 'and surfacing?',
                     self.T[5], host_turn='p2')
        eng = self._engine()

        def slots(stack):
            return sorted((s, j, t) for s, j, _v, t in stack)
        at_b, led_b = eng._moment_stack(self.brain, self.SESS, 3,
                                        cue_trace_id='mtu03')
        self.assertEqual(led_b['live_edge_dropped'], 0)
        self.assertEqual(slots(at_b), [
            ('a', 2, 'sonar arrays, mostly'), ('o', 1, 'what about ballast'),
            ('o', 2, 'tell me about submarines')])
        at_c, _led = eng._moment_stack(self.brain, self.SESS, 3,
                                       cue_trace_id='mtu05')
        self.assertEqual(slots(at_c), [
            ('a', 1, 'both flood to dive'), ('a', 2, 'sonar arrays, mostly'),
            ('o', 1, 'and trim tanks'), ('o', 1, 'what about ballast'),
            ('o', 2, 'tell me about submarines')])
        replay_b, led_rb = eng._moment_stack(self.brain, self.SESS, 3,
                                             as_of=self.T[3])
        self.assertEqual(led_rb['live_edge_dropped'], 1)
        self.assertEqual(slots(replay_b), [
            ('a', 1, 'sonar arrays, mostly'),
            ('o', 1, 'tell me about submarines')])

    def test_cue_from_another_speaker_keeps_the_operators_opener(self):
        # A subagent's report and a session's message arriving mid-turn fire
        # recall on their own text. The cut is keyed on the cue's row, so the
        # operator's opener stays at o1 — a rule that dropped the newest
        # operator message would drop it.
        self._mk_row('mtu00', 'user_message', 'tell me about submarines',
                     self.T[0], host_turn='p0')
        self._mk_row('mta00', 'assistant_message', 'sonar arrays, mostly',
                     self.T[1])
        self._mk_row('mtu02', 'user_message', 'what about ballast',
                     self.T[2], host_turn='p1')
        self._mk_row('mts03', 'subagent_message', 'report: ballast tanks',
                     self.T[3], with_vec=False)
        self._mk_row('mts04', 'self_message', 'from a sibling session',
                     self.T[4], with_vec=False)
        eng = self._engine()
        for cue, riders in (('mts03', 0), ('mts04', 1)):
            stack, ledger = eng._moment_stack(self.brain, self.SESS, 3,
                                              cue_trace_id=cue)
            texts = {(s, j): t for s, j, _v, t in stack}
            self.assertEqual(texts[('o', 1)], 'what about ballast', cue)
            self.assertNotIn(('a', 1), texts, cue)
            self.assertEqual(texts[('o', 2)], 'tell me about submarines', cue)
            # the report rides in A's turn once it is behind the cut
            self.assertEqual(ledger['correspondent_dropped'], riders, cue)

    def test_harness_opened_turn_keeps_its_slot_without_an_o_entry(self):
        # A harness notice on its own chain opens a turn: the slot and its
        # reply stay (W2), the notice is no operator message.
        self._mk_turn(0, 'tell me about submarines', 'sonar arrays, mostly')
        self._mk_row('mte02', 'env_message', 'background task finished',
                     self.T[2], with_vec=False, chain='s0-momsess-2')
        self._mk_row('mta03', 'assistant_message', 'noted the task result',
                     self.T[3], chain='s0-momsess-2')
        self._mk_row('mtu04', 'user_message', 'and propulsion?', self.T[4],
                     chain='s0-momsess-3')
        self._mk_row('mta05', 'assistant_message', 'nuclear or diesel',
                     self.T[5], chain='s0-momsess-3')
        eng = self._engine()
        stack, ledger = eng._moment_stack(self.brain, self.SESS, 3)
        self.assertEqual(ledger['machine_dropped'], 1)
        self.assertEqual(sorted((s, j, t) for s, j, _v, t in stack), [
            ('a', 1, 'nuclear or diesel'), ('a', 2, 'noted the task result'),
            ('a', 3, 'sonar arrays, mostly'), ('o', 1, 'and propulsion?'),
            ('o', 3, 'tell me about submarines')])

    def test_recall_threads_the_cue_to_the_stack(self):
        # brain.recall(cue_trace_id=) reaches _moment_stack, and the result
        # cache keys on it: the same text under another cue is recomputed.
        from servers.recall_laf import get_engine
        import time as _t
        self._mk_turn(0, 'tell me about submarines', 'sonar arrays, mostly')
        eng = get_engine(self.brain)
        eng._cfg = dict(DEFAULT_CONFIG, moment_K=2,
                        moment_gains={'maxsim_o1': 1.0})
        eng._cfg_ts = _t.monotonic() + 3600
        cues = []
        real = eng._moment_stack

        def spy(brain, session_id, K, as_of=None, cue_trace_id=None):
            cues.append(cue_trace_id)
            return real(brain, session_id, K, as_of=as_of,
                        cue_trace_id=cue_trace_id)
        eng._moment_stack = spy
        for cue in ('mtu00', 'mtu00', 'mta00'):
            self.brain.recall(query='submarine sonar', session_id=self.SESS,
                              cue_trace_id=cue, mark_accessed=False)
        self.assertEqual(cues, ['mtu00', 'mta00'])

    def test_asof_deep_history_positions_window_at_asof(self):
        # Regression (2026-08-07 fetch-then-filter sweep): get_conversation
        # pulls newest-first LIMIT 4K+8; the old Python `< as_of` post-filter
        # emptied any replay whose as_of sat further back than the window —
        # ledger reported rows: N, turns: 0 instead of the turns that plainly
        # existed. The bound now rides the SQL (older_than), so the window is
        # "the last turns AT as_of".
        for i in range(10):
            self._mk_turn(i, 'question %d about topic' % i, 'answer %d' % i)
        eng = self._engine()
        # K=1: 14 rows exist after T[6] — a newest-first pull with a
        # Python-side cut would hold not a single pre-as_of row. The reader
        # reads K+1 turns at as_of (one spare for the live-edge rule).
        stack, ledger = eng._moment_stack(self.brain, self.SESS, 1,
                                          as_of=self.T[6])
        self.assertEqual(ledger['rows'], 4)
        self.assertEqual(ledger['turns'], 1)
        texts = {(s, j): t for s, j, _v, t in stack}
        self.assertEqual(texts[('o', 1)], 'question 2 about topic')
        self.assertEqual(texts[('a', 1)], 'answer 2')

    def test_moment_table_shifts_scores_toward_history(self):
        import servers.embedder as embedder
        sub = self.brain.remember(
            type='lesson', title='Submarine sonar arrays',
            content='Attack submarines mount bow sonar arrays.')['id']
        ada = self.brain.remember(
            type='lesson', title='Garden tomato watering',
            content='Water tomato plants at the roots, mornings.')['id']
        self._mk_turn(0, 'tell me about attack submarines',
                      'they mount bow sonar arrays and dive deep')
        self._mk_turn(1, 'noted', 'anything else?')   # complete turn on top
        query = 'watering the garden tomatoes'
        qv = embedder.embed_query(query)
        bare = self._engine()
        bare_map, _ = bare.scores(self.brain, query, qv, session_id=self.SESS)
        self.assertGreater(bare_map[ada], bare_map[sub])
        # additive-shape table: fitted j≥1 terms only, base gains untouched
        eng = self._engine(moment_K=3, moment_gains={'maxsim_a2': 8.0})
        mom_map, mom_tel = eng.scores(self.brain, query, qv,
                                      session_id=self.SESS)
        self.assertGreater(mom_map[sub], mom_map[ada],
                           'a2 slot (submarine anchor) should lift the '
                           'submarine node past the tomato node')
        self.assertEqual(eng._last_moment_ledger['turns'], 2)
        self.assertTrue(any('maxsim_a2' in f for f in mom_tel.values()))

    def test_empty_history_collapses_to_fitted_k0(self):
        # A1 on an empty stack ≡ A0f bit-identical (§20.18 C2): slot keys
        # contribute nothing when no history exists — fresh session id
        import servers.embedder as embedder
        self.brain.remember(type='lesson', title='Cache strategy note',
                            content='Cache aggressively at the edge.')
        qv = embedder.embed_query('caching')
        o0_table = {'maxsim_o0': 0.7, 'sit_o0': 0.26, 'idf_o0': 0.24,
                    'pick': 0.0, 'enc': 0.0}
        full_table = dict(o0_table, maxsim_a1=0.95, sit_a2=0.21, idf_a3=0.13)
        a0f = self._engine(moment_K=8, moment_gains=o0_table)
        a1 = self._engine(moment_K=8, moment_gains=full_table)
        self.assertEqual(
            a1.scores(self.brain, 'caching', qv, session_id='sess-empty'),
            a0f.scores(self.brain, 'caching', qv, session_id='sess-empty'))

    def test_o0_override_equals_plain_gain_change(self):
        # {'maxsim_o0': g} through the table ≡ gain_maxsim=g without it —
        # pins the override resolution order (A0f mechanics, moment_K=0)
        import servers.embedder as embedder
        import time as _t
        self.brain.remember(type='lesson', title='Cache strategy note',
                            content='Cache aggressively at the edge.')
        self.brain.remember(type='lesson', title='Unrelated gardening note',
                            content='Prune roses in late winter.')
        qv = embedder.embed_query('caching')
        via_table = self._engine(moment_K=0,
                                 moment_gains={'maxsim_o0': 0.25})
        plain = self._engine()
        plain._cfg = dict(DEFAULT_CONFIG, gain_maxsim=0.25)
        plain._cfg_ts = _t.monotonic()
        self.assertEqual(
            via_table.scores(self.brain, 'caching', qv, session_id=self.SESS),
            plain.scores(self.brain, 'caching', qv, session_id=self.SESS))


if __name__ == '__main__':
    unittest.main()
