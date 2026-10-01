"""get_nodes_config override + result-shape handling on _format_result (2026-06-23).

Pins the fix for the S2 community encoder's 217K-token blowup.

Root cause (two layers):
  1. SHAPE: the dispatch handler `_handle_get_nodes` (dispatch_read.py) returns
     a LIST of rich nodes (+ {"id","error"} entries) — the shape every encoder
     and Anchor's MCP get_nodes actually receive. `_format_result` originally
     gated its render branch on `isinstance(result, dict)`, so on the real
     (list) path the branch never fired and output fell through to
     `json.dumps` — the raw `_corrections` firehose, at every batch size.
  2. CONFIG: even when rendered, a small (<=3) batch took a raw-JSON escape
     hatch. run_llm_loop now threads a caller-declared `get_nodes_config` so
     S2 encoders render bounded at every batch size.

Fix: _format_result handles BOTH list and dict shapes; with get_nodes_config
set it renders via render_rich_node at every batch size. The community encoder
passes S2CE_NODE_FORMAT (content 800, edges 5, corrections 'balanced').

The <=3 raw-JSON escape hatch is GONE entirely. get_node / get_nodes /
filter_nodes / the recall tool's results always render through
render_rich_node (representation is a render concern; brain.get_node stays the
always-full data layer), through the one selector contract.node_format_for:
DETAIL up to GET_NODES_DETAIL_MAX nodes (whole content, bounded
edges/corrections), SCAN above; the MCP `rich=true` opt-in lifts DETAIL to
every edge and the heavy correction K/V. get_nodes_config still overrides
all of it — encoders are never blocked.

Pure-function tests — _format_result + render_rich_node take a result value,
no brain/embedder.

Run: ./dev python3 -m pytest tests/test_get_nodes_config_override.py -v
"""
import json
import unittest

from servers.brain_mcp import _format_result
from servers.scales.s2.community_contract import S2CE_NODE_FORMAT


def _hub_node(node_id="hub00001"):
    """A rich node shaped like a hub member: long content + a heavy correction
    carrying full K/V. Sentinels mark the firehose-only tails."""
    return {
        "id": node_id,
        "title": "Hub member",
        "type": "decision",
        "content": "A" * 900 + " CONTENT_TAIL_SENTINEL",   # tail past 800
        "situation": "When the encoder inspects a hub member",
        "confidence": 0.9,
        "created_at": "2026-06-20T00:00:00+00:00",
        # Real edge shapes — exercise both render branches (multi-relation via
        # the `relations` list; single-relation via flat `relation`) and the
        # edge_limit bound. 7 edges > S2CE_NODE_FORMAT's edge_limit=5, so the
        # last two (OVERFLOW sentinels) must be dropped.
        "connections": [
            {"id": "nbr00001", "type": "decision", "title": "Neighbor one",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "weight": 0.6, "relations": [
                 {"relation": "depends_on", "description": "needs neighbor one", "weight": 0.6},
                 {"relation": "extends", "description": "also extends it", "weight": 0.4}]},
            {"id": "nbr00002", "type": "finding", "title": "Neighbor two",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "incoming",
             "relation": "implements", "description": "implements the thing", "weight": 0.5},
            {"id": "nbr00003", "type": "rule", "title": "Neighbor three",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d3"},
            {"id": "nbr00004", "type": "rule", "title": "Neighbor four",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d4"},
            {"id": "nbr00005", "type": "rule", "title": "Neighbor five",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d5"},
            {"id": "nbr00006", "type": "rule", "title": "OVERFLOW_EDGE_SENTINEL six",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d6"},
            {"id": "nbr00007", "type": "rule", "title": "OVERFLOW_EDGE_SENTINEL seven",
             "created_at": "2026-06-19T00:00:00+00:00", "direction": "outgoing",
             "relation": "informs", "description": "d7"},
        ],
        "_corrections": [{
            "id": "corr0001",
            "title": "A correction",
            "type": "correction",
            "direction": "incoming",
            "relation": "corrects",
            "edge_description": "supersedes the earlier assumption",
            "content": "C" * 300 + " CORR_TAIL_SENTINEL",
            "reasoning": "REASONING_SENTINEL why the correction landed",
            "my_raw_quote": "ANCHOR_QUOTE_SENTINEL",
            "their_raw_quote": "USER_QUOTE_SENTINEL",
        }],
    }


def _dispatch_list(n=1, with_error=False):
    """The shape _handle_get_nodes actually returns: a LIST of rich nodes,
    with {"id","error"} entries appended for unresolved ids."""
    out = [_hub_node("hub%05d" % i) for i in range(n)]
    if with_error:
        out.append({"id": "missing1", "error": "not found"})
    return out


def _is_raw_json(s):
    """Raw json.dumps output is parseable JSON; render_rich_node output starts
    with a `[type] "title"` header and is NOT valid JSON. (Can't discriminate
    on a leading '[' — rendered nodes lead with their bracketed type tag.)"""
    try:
        json.loads(s)
        return True
    except Exception:
        return False


class TestGetNodesConfigOverride(unittest.TestCase):

    # ── production (list) shape ──────────────────────────────────────

    def test_default_small_batch_bounded_not_raw(self):
        """A small pull, no config, rich=false → the DETAIL view, NOT the old
        raw-JSON firehose. Full content stays (content is the signal you
        fetched for); heavy correction K/V is dropped (balanced corrections)."""
        result = _dispatch_list(1)
        out = _format_result("get_nodes", result)            # config=None, rich=False
        self.assertFalse(_is_raw_json(out))                  # rendered, not raw
        self.assertIn("Hub member", out)
        self.assertIn("CONTENT_TAIL_SENTINEL", out)          # full content kept
        self.assertIn("A correction", out)                   # correction gist present
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)       # heavy K/V dropped
        self.assertNotIn("REASONING_SENTINEL", out)
        self.assertNotIn("USER_QUOTE_SENTINEL", out)

    def test_rich_small_batch_full_view(self):
        """rich=true → DETAIL lifted: full content + ALL edges + heavy
        correction K/V (the deliberate firehose), still rendered (not raw)."""
        result = _dispatch_list(1)
        out = _format_result("get_nodes", result, rich=True)
        self.assertFalse(_is_raw_json(out))
        self.assertIn("CONTENT_TAIL_SENTINEL", out)          # full content
        self.assertIn("ANCHOR_QUOTE_SENTINEL", out)          # heavy K/V present
        self.assertIn("REASONING_SENTINEL", out)
        self.assertIn("OVERFLOW_EDGE_SENTINEL", out)         # all edges (edge_limit None)
        # Full view is strictly larger than the bounded default.
        self.assertGreater(len(out), len(_format_result("get_nodes", result)))

    def test_get_node_single_renders(self):
        """get_node (single dict, not a list) takes the same render path —
        bounded by default, full under rich=true. Never a raw dump."""
        node = _hub_node("solo0001")
        out = _format_result("get_node", node)               # rich=False
        self.assertFalse(_is_raw_json(out))
        self.assertIn("Hub member", out)
        self.assertIn("CONTENT_TAIL_SENTINEL", out)          # full content
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)       # balanced default
        rich_out = _format_result("get_node", node, rich=True)
        self.assertIn("ANCHOR_QUOTE_SENTINEL", rich_out)     # heavy K/V under rich

    def test_filter_nodes_enriched_renders_bounded(self):
        """filter_nodes enriched result ({nodes, total_count}) renders bounded —
        never the raw dump that made a 50-node rich filter a firehose. The MCP
        render opt-in does NOT lift it (multi-node scan is bounded by design)."""
        result = {"nodes": [_hub_node("flt%05d" % i) for i in range(2)],
                  "total_count": 2}
        out = _format_result("filter_nodes", result, rich=True)   # rich ignored here
        self.assertFalse(_is_raw_json(out))
        self.assertIn("2 nodes (of 2 total)", out)
        self.assertEqual(out.count("Hub member"), 2)
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)       # bounded, no heavy K/V

    def test_filter_nodes_skinny_one_liners(self):
        """Skinny shape (no 'connections') → compact one-line-per-node, with
        the filtered field value surfaced for discovery."""
        result = {"nodes": [
            {"id": "skin0001", "title": "Skinny one", "type": "decision",
             "confidence": 0.9, "created_at": "2026-06-20T00:00:00+00:00",
             "encoding_source": "anchor"}], "total_count": 1}
        out = _format_result("filter_nodes", result)
        self.assertFalse(_is_raw_json(out))
        self.assertIn("Skinny one", out)
        self.assertIn("encoding_source=anchor", out)         # filtered field shown
        self.assertNotIn("Content:", out)                    # no rich body

    def test_filter_nodes_skinny_bounds_long_field(self):
        """A long-valued filtered field (e.g. field='content') is TRUNCATED in
        the skinny one-liner — the discovery scan stays bounded, not the
        firehose the raw-JSON path would have dumped."""
        result = {"nodes": [
            {"id": "skin0002", "title": "Long field node", "type": "fact",
             "confidence": 0.9, "created_at": "2026-06-20T00:00:00+00:00",
             "content": "X" * 5000 + " LONGFIELD_TAIL_SENTINEL"}],
            "total_count": 1}
        out = _format_result("filter_nodes", result)
        self.assertFalse(_is_raw_json(out))
        self.assertIn("Long field node", out)
        self.assertNotIn("LONGFIELD_TAIL_SENTINEL", out)     # tail truncated
        self.assertLess(len(out), 400)                       # bounded, not 5000+

    def test_default_large_batch_list_renders(self):
        """REGRESSION GUARD for the dead branch: a >10-node LIST with no config
        must render via the batch-size heuristic — NOT fall through to raw JSON.
        This is exactly what was broken (isinstance(result, dict) gate)."""
        result = _dispatch_list(12)
        out = _format_result("get_nodes", result)          # config=None
        self.assertFalse(_is_raw_json(out))                  # rendered, not raw
        self.assertEqual(out.count("Hub member"), 12)

    def test_config_override_bounds_small_batch_list(self):
        """With S2CE_NODE_FORMAT, a <=3-node LIST renders bounded — no raw JSON,
        content truncated to 800, corrections 'balanced' (no heavy K/V)."""
        result = _dispatch_list(1)
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)

        self.assertFalse(_is_raw_json(out))                  # rendered
        self.assertIn("Hub member", out)                     # title rendered
        self.assertNotIn("CONTENT_TAIL_SENTINEL", out)       # content capped at 800
        # 'balanced' corrections: title + id + edge "why" + short excerpt,
        # but NOT heavy K/V (reasoning / raw quotes) or full correction content.
        self.assertIn("A correction", out)
        self.assertIn("supersedes the earlier assumption", out)
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)
        self.assertNotIn("USER_QUOTE_SENTINEL", out)
        self.assertNotIn("REASONING_SENTINEL", out)
        self.assertNotIn("CORR_TAIL_SENTINEL", out)          # 300-char tail > 150 cap
        # Edge path exercised (the heaviest field) and bounded by edge_limit=5:
        self.assertIn("depends_on", out)                     # multi-relation edge rendered
        self.assertIn("implements", out)                     # single-relation edge rendered
        self.assertNotIn("OVERFLOW_EDGE_SENTINEL", out)      # edges 6-7 dropped at edge_limit=5
        # Bounded render is far smaller than the firehose.
        self.assertLess(len(out), len(_format_result("get_nodes", result)))

    def test_config_override_large_batch_list(self):
        """Override is batch-size agnostic on the list shape too."""
        result = _dispatch_list(12)
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)
        self.assertFalse(_is_raw_json(out))
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)       # balanced, not heavy
        self.assertEqual(out.count("Hub member"), 12)

    def test_recall_results_render_exactly_like_get_nodes(self):
        """The recall tool's results and a get_nodes of the same ids go through
        one selector — byte-identical node renders, at both view sizes — so a
        recall and a pull of one node never read differently."""
        for n in (2, 12):
            nodes = [_hub_node("same%04d" % i) for i in range(n)]
            via_get_nodes = _format_result("get_nodes", list(nodes))
            via_recall = _format_result("recall", {"results": list(nodes)})
            self.assertEqual(via_recall, via_get_nodes)
        # the two views are really two: DETAIL keeps the content whole, SCAN cuts
        self.assertIn("CONTENT_TAIL_SENTINEL", _format_result("get_nodes", _dispatch_list(1)))
        self.assertNotIn("CONTENT_TAIL_SENTINEL", _format_result("get_nodes", _dispatch_list(12)))

    def test_error_entries_render_as_miss_line(self):
        """{"id","error"} entries from unresolved ids must not render as
        nodes (they lack node fields) — but they must not vanish either:
        the real nodes render, misses land on a trailing ⚠ line (the
        id-resolution unification's loud-miss contract)."""
        result = _dispatch_list(2, with_error=True)
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)
        self.assertEqual(out.count("Hub member"), 2)         # both real nodes
        self.assertIn("not found", out)                      # miss reported
        self.assertIn("missing1", out)                       # ...with its id

    def test_all_miss_renders_miss_line_not_raw_json(self):
        """All ids unknown → the ⚠ line alone, same shape as the partial-miss
        case — never a raw JSON dump."""
        result = [{"id": "missing1", "error": "not found"},
                  {"id": "missing2", "error": "not found"}]
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)
        self.assertIn("not found", out)
        self.assertIn("missing1", out)
        self.assertIn("missing2", out)
        self.assertFalse(_is_raw_json(out))

    # ── dict shape (brain.get_node(ids) direct) still works ──────────

    def test_dict_shape_also_renders(self):
        """Defensive: the {node_id: rich_node} dict shape (brain.get_node)
        renders the same way under a config override."""
        result = {"hub00001": _hub_node("hub00001")}
        out = _format_result("get_nodes", result,
                             get_nodes_config=S2CE_NODE_FORMAT)
        self.assertFalse(_is_raw_json(out))                  # rendered, not raw
        self.assertIn("Hub member", out)
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)


class TestRunnerHonorsTheCallsRich(unittest.TestCase):
    """An encoder's get_nodes rich=true gets the full view, as it does through
    the MCP proxy — the runner used to render every encoder reply as if rich
    were false. A declared get_nodes_config still wins over it."""

    def _tool_reply(self, tool_input, get_nodes_config=None):
        """The get_nodes tool_result text the model reads in round 2."""
        from servers.scales.runner import run_llm_loop
        from tests.test_runner_refused_ops import _Block, _Client, _Msg
        rounds = []
        run_llm_loop(
            client=_Client([
                _Msg([_Block('tool_use', id='t0', name='get_nodes',
                             input=tool_input)], stop_reason='tool_use'),
                _Msg([_Block('text', text='done')])]),
            model='claude-test', max_tokens=256, max_rounds=3,
            system_prompt='S', user_content='U', tools=[],
            dispatch_fn=lambda name, args: {"ok": True,
                                            "result": _dispatch_list(1)},
            log_fn=lambda m: None, get_nodes_config=get_nodes_config,
            record_round_fn=lambda i, parts: rounds.append(parts))
        (block,) = rounds[1]["messages"][-1]["content"]
        return block["content"]

    def test_rich_true_gets_the_full_view(self):
        out = self._tool_reply({"node_ids": ["hub00000"], "rich": True})
        self.assertIn("ANCHOR_QUOTE_SENTINEL", out)          # heavy K/V
        self.assertIn("OVERFLOW_EDGE_SENTINEL", out)         # every edge
        self.assertEqual(out, _format_result("get_nodes", _dispatch_list(1),
                                             rich=True))

    def test_without_rich_the_default_view(self):
        out = self._tool_reply({"node_ids": ["hub00000"]})
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)
        self.assertEqual(out, _format_result("get_nodes", _dispatch_list(1)))

    def test_declared_config_wins_over_rich(self):
        out = self._tool_reply({"node_ids": ["hub00000"], "rich": True},
                               get_nodes_config=S2CE_NODE_FORMAT)
        self.assertNotIn("ANCHOR_QUOTE_SENTINEL", out)
        self.assertEqual(out, _format_result("get_nodes", _dispatch_list(1),
                                             get_nodes_config=S2CE_NODE_FORMAT))


if __name__ == "__main__":
    unittest.main()
