"""Dashboard consolidation-runs: trace-authoritative enrichment.

A consolidation run's product is read from the run's OWN delta record, and
only what the write path accepted counts:
  • survivors / archived originals — the ok-gated `revised`/`archived`
    buckets; a refused op shows nowhere;
  • links — EVERY accepted connect decision (KEEP/SKIP similar_to,
    SUPERSESSION supersedes, CONTRADICTION corrects, partition depends_on),
    carrying its relation, minus the ops `failed_ops` reports refused;
  • nodes are fetched by id — never reconstructed from a time window, which
    would credit other writers' edges in the same hour to the run.
"""
import json
import sqlite3

from dashboard.queries.s2_runs import (
    _deconstruct_consolidation_ops,
    _enrich_consolidation,
)


def _delta_meta(operations, **top):
    """Delta metadata carrying one brain_batch action with `operations`."""
    m = {"action_details": [{"tool": "brain_batch",
                             "input": {"operations": operations}}]}
    m.update(top)
    return m


class TestDeconstructConsolidationOps:
    def test_absorb_splits_into_synth_and_archived(self):
        synth, archived, links = _deconstruct_consolidation_ops(_delta_meta(
            [{"op": "absorb", "survivor_id": "surv1", "absorbed_id": "orig1"}],
            revised=["surv1"], archived=["orig1"]))
        assert synth == ["surv1"]      # the enriched survivor
        assert archived == ["orig1"]   # the folded-in original
        assert links == []

    def test_connect_similar_to_captured(self):
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "a", "target_id": "b",
             "relation": "similar_to", "description": "two locked dups"},
        ]))
        assert links == [{"source_id": "a", "target_id": "b",
                          "relation": "similar_to", "description": "two locked dups"}]

    def test_connect_supersedes_captured(self):
        # SUPERSESSION — was silently dropped before the fix (only similar_to
        # matched). It is a standalone connect(supersedes), no absorb.
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "new", "target_id": "old",
             "relation": "supersedes", "description": "Q2 caps supersede static"},
        ]))
        assert links == [{"source_id": "new", "target_id": "old",
                          "relation": "supersedes",
                          "description": "Q2 caps supersede static"}]

    def test_connect_corrects_captured(self):
        # CONTRADICTION — the correction substrate; also dropped before the fix.
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "right", "target_id": "wrong",
             "relation": "corrects", "description": "CDN, not DB"},
        ]))
        assert links[0]["relation"] == "corrects"

    def test_connect_depends_on_captured(self):
        _, _, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "connect", "source_id": "x", "target_id": "y",
             "relation": "depends_on"},
        ]))
        assert links and links[0]["relation"] == "depends_on"

    def test_refused_connect_is_not_a_link(self):
        # failed_ops indexes the refused op by its position in `operations`.
        meta = _delta_meta([
            {"op": "connect", "source_id": "a", "target_id": "b",
             "relation": "similar_to"},
            {"op": "connect", "source_id": "c", "target_id": "d",
             "relation": "supersedes"},
        ])
        meta["action_details"][0]["failed_ops"] = [
            {"index": 0, "op": "connect", "error": "refused"}]
        _, _, links = _deconstruct_consolidation_ops(meta)
        assert [l["source_id"] for l in links] == ["c"]

    def test_stringified_operations_are_read_as_dispatch_ran_them(self):
        # Dispatch decodes a JSON-encoded operations array and JSON-encoded
        # elements; the card reads the same ops, at the same indexes.
        ops = [{"op": "connect", "source_id": "a", "target_id": "b",
                "relation": "similar_to"},
               json.dumps({"op": "connect", "source_id": "c", "target_id": "d",
                           "relation": "supersedes"})]
        meta = {"action_details": [{"tool": "brain_batch",
                                    "input": {"operations": json.dumps(ops)},
                                    "failed_ops": [{"index": 0, "op": "connect"}]}]}
        _, _, links = _deconstruct_consolidation_ops(meta)
        assert [(l["source_id"], l["relation"]) for l in links] == [("c", "supersedes")]

    def test_errored_call_draws_no_links(self):
        # An ok=False call can carry no failed_ops (a refusal before dispatch
        # lists none); the runner's `error` on the action is enough.
        meta = _delta_meta([{"op": "connect", "source_id": "a",
                             "target_id": "b", "relation": "similar_to"}])
        meta["action_details"][0]["error"] = "operator-channel only"
        assert _deconstruct_consolidation_ops(meta) == ([], [], [])

    def test_refused_call_draws_no_links(self):
        # index None = the whole call answered ok=False.
        meta = _delta_meta([{"op": "connect", "source_id": "a",
                             "target_id": "b", "relation": "similar_to"}])
        meta["action_details"][0]["failed_ops"] = [
            {"index": None, "op": "brain_batch", "error": "refused"}]
        assert _deconstruct_consolidation_ops(meta) == ([], [], [])

    def test_mixed_batch(self):
        synth, archived, links = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "absorb", "survivor_id": "s1", "absorbed_id": "o1"},
            {"op": "absorb", "survivor_id": "s2", "absorbed_id": "o2"},
            {"op": "connect", "source_id": "k1", "target_id": "k2",
             "relation": "similar_to"},
            {"op": "connect", "source_id": "n", "target_id": "o",
             "relation": "supersedes"},
        ], revised=["s1", "s2"], archived=["o1", "o2"]))
        assert synth == ["s1", "s2"]
        assert archived == ["o1", "o2"]
        assert [l["relation"] for l in links] == ["similar_to", "supersedes"]

    def test_ok_gated_buckets_win_over_op_input(self):
        # The ok-gated top-level buckets are the truth (a failed absorb is
        # excluded from them); the request-side op input never names nodes.
        synth, archived, links = _deconstruct_consolidation_ops(_delta_meta(
            [{"op": "absorb", "survivor_id": "OP_S", "absorbed_id": "OP_A"},
             {"op": "connect", "source_id": "x", "target_id": "y",
              "relation": "similar_to"}],
            revised=["BUCKET_S"], archived=["BUCKET_A"]))
        assert synth == ["BUCKET_S"]      # bucket wins over op survivor
        assert archived == ["BUCKET_A"]   # bucket wins over op absorbed
        assert links[0]["source_id"] == "x"  # links always from ops (not bucketed as pairs)

    def test_refused_absorb_shows_nowhere(self):
        # An absorb the write path refused is in neither bucket: the card must
        # not show the requested survivor as enriched or its original as
        # folded in while it is still live.
        synth, archived, _ = _deconstruct_consolidation_ops(_delta_meta([
            {"op": "absorb", "survivor_id": "S", "absorbed_id": "A"},
        ]))
        assert synth == [] and archived == []

    def test_empty_meta(self):
        assert _deconstruct_consolidation_ops({}) == ([], [], [])


class TestEnrichConsolidationById:
    def _conn(self):
        c = sqlite3.connect(":memory:")
        c.execute(
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT, title TEXT, "
            "content TEXT, confidence REAL, encoding_source TEXT, "
            "created_at TEXT, archived INTEGER DEFAULT 0)")
        # Edge tables exist so a test can put another writer's edge in the
        # run's hour and show it is not credited to the run.
        c.execute("CREATE TABLE edges (edge_id INTEGER, source_id TEXT, "
                  "target_id TEXT, weight REAL, created_at TEXT)")
        c.execute("CREATE TABLE edge_relations (edge_id INTEGER, relation TEXT, "
                  "description TEXT, archived INTEGER DEFAULT 0, "
                  "claim_source_id TEXT, claim_target_id TEXT)")
        return c

    def _node(self, c, nid, archived=0, etype='principle', title=None):
        c.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                  (nid, etype, title or ('Node ' + nid), 'content-' + nid,
                   0.9, 's2:consolidation', '2026-06-14T10:00:00+00:00', archived))

    def test_fetches_recorded_nodes_by_id_no_window(self):
        c = self._conn()
        self._node(c, 'surv1', archived=0, title='Survivor title')
        self._node(c, 'orig1', archived=1, etype='fact', title='Folded original')
        c.commit()
        meta = _delta_meta([{"op": "absorb", "survivor_id": "surv1",
                             "absorbed_id": "orig1"}],
                           revised=["surv1"], archived=["orig1"])
        delta_row = ("s2-20260614-consolidation", "1 action",
                     json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, lambda *_: {})
        assert [n["id"] for n in out["synthesized"]] == ["surv1"]
        assert out["synthesized"][0]["title"] == "Survivor title"
        assert [n["id"] for n in out["archived"]] == ["orig1"]
        assert "kept" not in out and "evolved" not in out  # unified into links

    def test_supersedes_link_carries_relation(self):
        c = self._conn()
        self._node(c, 'new1', title='New state')
        self._node(c, 'old1', title='Prior state')
        c.commit()
        meta = _delta_meta([{"op": "connect", "source_id": "new1",
                             "target_id": "old1", "relation": "supersedes",
                             "description": "new supersedes old"}])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, lambda *_: {})
        assert out["links"] == [{"source": "New state", "target": "Prior state",
                                 "relation": "supersedes",
                                 "description": "new supersedes old"}]
        # supersedes-only run must NOT fall into the window (synthesized empty,
        # link present)
        assert out["synthesized"] == [] and out["archived"] == []

    def test_archived_survivor_shown_as_archived_not_live(self):
        # A survivor archived by a later run (DB archived=1) must not render as
        # a live SYNTHESIZED node — the DB flag is the liveness tiebreaker.
        c = self._conn()
        self._node(c, 'survX', archived=1)   # later archived
        self._node(c, 'origX', archived=1)
        c.commit()
        meta = _delta_meta([{"op": "absorb", "survivor_id": "survX",
                             "absorbed_id": "origX"}],
                           revised=["survX"], archived=["origX"])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, lambda *_: {})
        assert out["synthesized"] == []
        assert {n["id"] for n in out["archived"]} == {"survX", "origX"}

    def test_chain_merge_dedup(self):
        # A node that is a survivor in op1 and absorbed in op2 (chain merge):
        # shown once, in archived (it didn't survive), never duplicated.
        c = self._conn()
        self._node(c, 'A', archived=1)   # survived op1, absorbed in op2 → archived
        self._node(c, 'B', archived=1)
        self._node(c, 'C', archived=0)
        c.commit()
        meta = _delta_meta([
            {"op": "absorb", "survivor_id": "A", "absorbed_id": "B"},
            {"op": "absorb", "survivor_id": "C", "absorbed_id": "A"},
        ], revised=["A", "C"], archived=["B", "A"])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, lambda *_: {})
        assert [n["id"] for n in out["synthesized"]] == ["C"]
        assert {n["id"] for n in out["archived"]} == {"A", "B"}

    def test_missing_link_endpoint_falls_back_to_short_id(self):
        c = self._conn()
        self._node(c, 'present', title='Present node')
        c.commit()
        meta = _delta_meta([{"op": "connect", "source_id": "present",
                             "target_id": "deadbeef", "relation": "similar_to"}])
        delta_row = ("chain", "s", json.dumps(meta), "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, lambda *_: {})
        assert out["links"][0]["source"] == "Present node"
        assert out["links"][0]["target"] == "deadbeef"  # short-id fallback

    def test_corrupt_meta_is_logged_not_swallowed(self):
        # Loud-by-default: a delta with unparseable metadata must surface a
        # warning (stderr + Logs-tab ring) and still degrade, not vanish.
        from dashboard import log
        log.clear()
        c = self._conn()
        delta_row = ("chain-corrupt", "summary", "{not valid json",
                     "2026-06-14T10:00:00+00:00")
        out = _enrich_consolidation(c, delta_row, lambda *_: {})  # must NOT raise
        assert out["synthesized"] == [] and out["links"] == []
        msgs = [e["message"] for e in log.recent()]
        assert any("unparseable metadata" in m for m in msgs), msgs

    def test_run_with_no_writes_shows_nothing(self):
        # A failed or zero-write run shows empty, even when a consolidation
        # node and another writer's similar_to edge sit in the same hour.
        c = self._conn()
        c.execute("INSERT INTO nodes VALUES ('n1','fact','Neighbor',"
                  "'c',0.7,'s2:consolidation','2026-09-29T00:05:00+00:00',0)")
        c.execute("INSERT INTO nodes VALUES ('n2','fact','Other',"
                  "'c',0.7,'s1:encoder','2026-09-29T00:05:00+00:00',0)")
        c.execute("INSERT INTO edges VALUES (1,'n1','n2',1.0,"
                  "'2026-09-29T00:10:00+00:00')")
        c.execute("INSERT INTO edge_relations VALUES (1,'similar_to','x',0,'n1','n2')")
        c.commit()
        delta_row = ("chain", "0 actions", json.dumps({"action_details": []}),
                     "2026-09-29T00:04:56+00:00")
        out = _enrich_consolidation(c, delta_row, lambda *_: {})
        assert out["synthesized"] == [] and out["archived"] == []
        assert out["links"] == []


class TestSameChainRunsGetOwnOK:
    """One chain can hold several O/K/delta triples. Keying O/K by chain_id
    alone collapses them onto the chain's single (oldest) O/K pair, so every
    card shows identical cluster counts; each delta must snap to the O/K that
    immediately precede IT in time.
    """

    def _build_dbs(self, tmp_path, runs):
        """Write minimal brain.db + brain_logs.db. `runs` is a list of
        (offset_minutes_ago, k_summary) — each becomes an O→K→delta triple in
        the shared chain. Returns the chain_id used."""
        from datetime import datetime, timezone, timedelta

        chain = 's2-20260614-consolidation'
        now = datetime.now(timezone.utc)

        def iso(mins_ago):
            return (now - timedelta(minutes=mins_ago)).strftime(
                '%Y-%m-%dT%H:%M:%S+00:00')

        logs = sqlite3.connect(str(tmp_path / 'brain_logs.db'))
        logs.execute(
            "CREATE TABLE trace_events (id TEXT, chain_id TEXT, scale TEXT, "
            "event_type TEXT, ref_type TEXT, summary TEXT, metadata TEXT, "
            "created_at TEXT)")
        n = 0
        for mins_ago, k_summary in runs:
            # O slightly before K, K before delta — same order the decoder
            # writes them. Spread by seconds so timestamps are distinct.
            base = mins_ago
            # Larger mins-ago = further in the past. O oldest, delta newest —
            # the order the decoder/encoder write them within a run.
            for et, ref, summ, sub in [
                ('O', 'consolidation_candidates', 'scanned', 0.2),
                ('K', 'consolidation_proposals', k_summary, 0.1),
                ('delta', 'consolidation_run', '%d action' % n, 0.0),
            ]:
                meta = (json.dumps(_delta_meta(
                    [{"op": "connect", "source_id": "live%d" % n,
                      "target_id": "dead%d" % n, "relation": "similar_to"}]))
                    if et == 'delta' else '')
                logs.execute(
                    "INSERT INTO trace_events VALUES (?,?,?,?,?,?,?,?)",
                    ('id%d' % n, chain, 's2', et, ref, summ, meta,
                     iso(base + sub)))  # O furthest back, delta newest
                n += 1
        logs.commit()
        logs.close()

        brain = sqlite3.connect(str(tmp_path / 'brain.db'))
        brain.execute(
            "CREATE TABLE nodes (id TEXT PRIMARY KEY, type TEXT, title TEXT, "
            "content TEXT, confidence REAL, encoding_source TEXT, "
            "created_at TEXT, archived INTEGER DEFAULT 0)")
        brain.execute("CREATE TABLE edges (edge_id INTEGER, source_id TEXT, "
                      "target_id TEXT, weight REAL, created_at TEXT)")
        brain.execute("CREATE TABLE edge_relations (edge_id INTEGER, "
                      "relation TEXT, description TEXT, archived INTEGER DEFAULT 0, "
                      "claim_source_id TEXT, claim_target_id TEXT)")
        for i in range(len(runs)):
            brain.execute(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,0)",
                ('live%d' % (i * 3), 'principle', 'Live node %d' % i, 'c',
                 0.9, 's2:consolidation', iso(0)))
        brain.commit()
        brain.close()
        return chain

    def test_each_run_gets_its_own_k_summary(self, tmp_path, monkeypatch):
        monkeypatch.setenv('BRAIN_DB_DIR', str(tmp_path))
        # Newest run first in input is irrelevant; build two distinct runs.
        self._build_dbs(tmp_path, [
            (10, '196 clusters: 181 needs_judgment, 15 likely_consolidate'),
            (5,  '30 clusters: 29 needs_judgment, 1 likely_consolidate'),
        ])
        from dashboard.queries.s2_runs import query_consolidation_runs
        runs = query_consolidation_runs(hours=24)

        # Two delta cards, ordered newest-first.
        by_k = sorted(r['k_summary'] for r in runs)
        assert len(runs) == 2, runs
        # The bug: both would read the oldest K ("196 clusters ..."). The fix:
        # each delta snaps to its own run's K.
        assert by_k == [
            '196 clusters: 181 needs_judgment, 15 likely_consolidate',
            '30 clusters: 29 needs_judgment, 1 likely_consolidate',
        ], [r['k_summary'] for r in runs]
