"""Trace-tool rendering (2026-06-29) — mirrors the get_nodes convention.

query_traces / get_traces / get_trace render through the single trace renderer
(trace_contract.render_trace), never a raw json.dumps. The default bounds the
heavy `metadata` field (s2 rows reach ~140KB) to a gist; rich=true returns full
metadata. Bodies are trimmed only on multi-row pulls, and the trim says what it
dropped — get_trace names one row and renders it whole. recall_episodes shares
the same renderer; recall_batch reuses the recall formatter instead of
raw-dumping.

Pure-function tests — _format_result + render_trace take a value, no
brain/embedder.

Run: ./dev python3 -m pytest tests/test_trace_render.py -v
"""
import json
import unittest

from servers.brain_mcp import _format_result


def _trace_row(tid="ab120001", scale="s2", etype="K",
               ref_type="community_enrichment", big_meta=True):
    """Structural trace row: heavy metadata blob (BLOB_TAIL past the gist cap)
    + a small scalar field + a sentinel summary."""
    meta = {"cluster_size": 12}
    if big_meta:
        meta["prompt"] = "P" * 5000 + " BLOB_TAIL_SENTINEL"
    return {
        "id": tid, "session_id": "sess0001", "scale": scale,
        "event_type": etype, "ref_type": ref_type,
        "summary": "Enriched a cluster into a narrative SUMMARY_SENTINEL",
        "metadata": meta, "created_at": "2026-06-29T00:30:00+00:00",
    }


def _grouped_event(tid):
    """An event as TraceDAL.get_chains actually produces it: carries its own
    id, but NO scale/session_id (those are chain-level). The grouped render
    branch must propagate scale/session_id onto it — synthetic events with
    those fields baked in would mask that (the probe-fidelity trap)."""
    return {"id": tid, "event_type": "K", "ref_type": "community_enrichment",
            "ref_id": "", "summary": "narrative SUMMARY_SENTINEL",
            "metadata": {"cluster_size": 12}, "created_at": "2026-06-29T00:30:00+00:00"}


def _is_raw_json(s):
    try:
        json.loads(s)
        return True
    except Exception:
        return False


class TestTraceRender(unittest.TestCase):

    # ── query_traces ──────────────────────────────────────────────────
    def test_query_traces_compact_not_raw(self):
        """Default: rendered (not raw JSON), body shown, the 140KB-class blob
        elided to a gist, small fields kept, output bounded."""
        out = _format_result("query_traces", {"events": [_trace_row()]})
        self.assertFalse(_is_raw_json(out))
        self.assertIn("SUMMARY_SENTINEL", out)            # body shown
        self.assertIn("prompt=<", out)                    # blob elided to "<N chars>"
        self.assertNotIn("BLOB_TAIL_SENTINEL", out)       # blob NOT leaked
        self.assertIn("cluster_size=12", out)             # small field shown
        self.assertLess(len(out), 500)                    # bounded

    def test_query_traces_rich_full(self):
        """rich=true → full metadata present."""
        out = _format_result("query_traces", {"events": [_trace_row()]}, rich=True)
        self.assertFalse(_is_raw_json(out))
        self.assertIn("BLOB_TAIL_SENTINEL", out)

    def test_query_traces_bulk_summary_only(self):
        """> TRACE_BULK_MAX rows → summary-only (metadata dropped) to protect
        context; every body still rendered."""
        rows = [_trace_row("ab%06d" % i) for i in range(25)]
        out = _format_result("query_traces", {"events": rows})
        self.assertFalse(_is_raw_json(out))
        self.assertEqual(out.count("SUMMARY_SENTINEL"), 25)
        self.assertNotIn("prompt=", out)                  # no metadata gist at bulk
        self.assertNotIn("cluster_size=", out)

    def test_query_traces_grouped(self):
        """grouped {chains}: chain header carries chain_id/scale/session; each
        event renders DRILLABLE (trace:id present) with chain-level scale +
        session propagated — no empty (trace:) or [] brackets, even though the
        get_chains event shape lacks per-event scale/session_id."""
        chains = [{"chain_id": "s2-x-comm", "scale": "s2", "session_id": "sess0001",
                   "events": [_grouped_event("ev000001"), _grouped_event("ev000002")]}]
        out = _format_result("query_traces", {"chains": chains})
        self.assertFalse(_is_raw_json(out))
        self.assertIn("chain s2-x-comm", out)             # chain header
        self.assertEqual(out.count("SUMMARY_SENTINEL"), 2)
        self.assertIn("(trace:ev000001)", out)            # drillable — the fix
        self.assertIn("[sess0001]", out)                  # session propagated, not []
        self.assertNotIn("(trace:)", out)                 # no empty id
        self.assertNotIn("[]", out)                       # no empty session bracket
        self.assertIn("s2 K", out)                        # scale propagated

    # ── get_trace / get_traces ────────────────────────────────────────
    def test_get_trace_single(self):
        """Point lookup: metadata gisted by default, full under rich."""
        out = _format_result("get_trace", _trace_row())
        self.assertFalse(_is_raw_json(out))
        self.assertIn("SUMMARY_SENTINEL", out)
        self.assertNotIn("BLOB_TAIL_SENTINEL", out)
        self.assertIn("BLOB_TAIL_SENTINEL", _format_result("get_trace", _trace_row(), rich=True))

    def test_get_trace_body_never_trimmed(self):
        """A named row renders its body WHOLE at both detail levels — the
        280-char cap was protecting against a body that cannot get large
        (metadata.content is capped at write time)."""
        row = _trace_row(big_meta=False)
        row["metadata"] = {"content": "B" * 3000 + "BODY_TAIL_SENTINEL"}
        for rich in (False, True):
            out = _format_result("get_trace", row, rich=rich)
            self.assertIn("BODY_TAIL_SENTINEL", out, "rich=%s trimmed the body" % rich)
            self.assertNotIn("chars — get_trace", out)

    def test_session_less_trace_no_empty_bracket(self):
        """Session-less traces (S2 system runs — no session_id, no score) render
        with NO leading [] bracket; the header starts at the label."""
        row = _trace_row()
        row["session_id"] = ""
        out = _format_result("get_trace", row)
        self.assertNotIn("[]", out)
        self.assertTrue(out.startswith("community_enrichment ·"))

    def test_get_traces_envelope(self):
        """Batch envelope: every row renders."""
        out = _format_result("get_traces", {
            "traces": [_trace_row("aa000001"), _trace_row("aa000002")],
            "missing_ids": []})
        self.assertFalse(_is_raw_json(out))
        self.assertEqual(out.count("SUMMARY_SENTINEL"), 2)

    def test_get_traces_names_missing_ids(self):
        """An id that matched no row is named — a short list must never be the
        only sign that something wasn't found."""
        out = _format_result("get_traces", {
            "traces": [_trace_row("aa000001")], "missing_ids": ["deadbeef"]})
        self.assertIn("SUMMARY_SENTINEL", out)
        self.assertIn("deadbeef", out)

    def test_get_traces_all_missing_still_speaks(self):
        """No rows at all → still the named ids, never a raw empty dump."""
        out = _format_result("get_traces", {"traces": [], "missing_ids": ["deadbeef"]})
        self.assertFalse(_is_raw_json(out))
        self.assertIn("deadbeef", out)

    def test_multi_row_body_cut_names_the_loss(self):
        """A trimmed body says how much it dropped and where to get the rest —
        a bare '…' reads as the speaker trailing off."""
        row = _trace_row("aa000001", big_meta=False)
        row["metadata"] = {"content": "B" * 3000 + "BODY_TAIL_SENTINEL"}
        out = _format_result("get_traces", {"traces": [row], "missing_ids": []})
        self.assertNotIn("BODY_TAIL_SENTINEL", out)
        self.assertIn("chars — get_trace for the whole body", out)
        self.assertIn("+2", out)  # dropped-char count, not a bare ellipsis

    def test_trim_never_inflates_the_body(self):
        """A body barely over the cap renders WHOLE: the ~40-char notice would
        cost more than the trim saves, and body_limit is a cap, not a floor."""
        from servers.trace_contract import _trim_body
        for over in (1, 5, 20, 40):
            body = "z" * (280 + over)
            out = _trim_body(body, 280)
            self.assertLessEqual(len(out), len(body),
                                 "trimming inflated a body %d over the cap" % over)
            self.assertEqual(out, body)  # left intact, not chopped for no gain
        # Far past the cap, the trim earns its keep and fires.
        long_body = "z" * 2000
        self.assertLess(len(_trim_body(long_body, 280)), len(long_body))

    # ── recall_batch reuses the recall formatter ──────────────────────
    def test_recall_batch_renders_not_raw(self):
        result = [
            {"query": "q-alpha", "results": [
                {"id": "node0001", "type": "fact", "title": "Batch hit", "content": "body"}]},
            {"query": "q-empty", "results": []},
            {"query": "q-err", "results": [], "error": "boom"},
        ]
        out = _format_result("recall_batch", result)
        self.assertFalse(_is_raw_json(out))
        self.assertIn('▸ "q-alpha"', out)                 # per-query header
        self.assertIn("Batch hit", out)                   # routed through recall fmt
        self.assertIn("No results found.", out)           # empty query
        self.assertIn("error: boom", out)                 # error entry

    # ── recall_episodes still renders (folded onto render_trace) ──────
    def test_recall_episodes_still_renders(self):
        ep = {"id": "ff00ff00", "session_id": "aaaa1111", "scale": "s0",
              "event_type": "delta", "ref_type": "assistant_message",
              "summary": "", "created_at": "2026-06-29T00:31:00+00:00", "_score": 0.87,
              "metadata": {"content": "Hello from Anchor EPISODE_SENTINEL",
                           "agent_identity": "Anchor"}}
        out = _format_result("recall_episodes",
                             {"episodes": [ep], "ranked_by": "relevance"})
        self.assertFalse(_is_raw_json(out))
        self.assertIn("1 episode · ranked by relevance", out)
        self.assertIn("Anchor", out)                      # who-framing preserved
        self.assertIn("EPISODE_SENTINEL", out)            # body
        self.assertNotIn("s0 delta", out)                 # episode format hides scale chrome


if __name__ == "__main__":
    unittest.main()
