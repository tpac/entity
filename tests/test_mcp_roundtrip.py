"""
MCP Round-Trip Tests — verifies every MCP tool works end-to-end.

Each tool exposed via brain_mcp.py is tested through the daemon dispatch
layer against a real Brain instance. This catches:
  - MCP tool -> daemon command mapping failures
  - Parameter name mismatches between MCP schema and handler
  - Brain method errors not surfaced properly
  - Missing dispatch handlers for new tools

Run: python3 -m pytest tests/test_mcp_roundtrip.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_mcp import TOOLS, CRITICAL_TOOLS
from servers.daemon_dispatch import COMMAND_TABLE, dispatch_command
from servers.trace_contract import session_short


class TestMCPRoundTrip(BrainTestBase):
    """Test every MCP tool through daemon dispatch against a real brain."""

    def _dispatch(self, cmd, args):
        """Simulate daemon dispatch — route through dispatch_command (THE one
        execution path; handlers are never called directly, and since step 4
        mutation-trace emission lives at that chokepoint), then ENFORCE +
        unwrap the {"ok": True, "result": ...} envelope.

        Every table handler MUST return that envelope; the daemon sends the
        return verbatim, so a raw dict reaches the MCP client as a falsy `ok`
        with no `error` — surfacing as "Unknown daemon error" on a call that
        actually succeeded (the dispatch_self bug, c4f6386). The previous
        lenient `if "result" in raw: ... else return raw` passed un-enveloped
        returns straight through, which is exactly why this suite — though it
        exercises self_presence/send/etc. — never caught that bug. Enforcing
        the envelope here makes the violation fail for EVERY tool, in CI."""
        entry = COMMAND_TABLE.get(cmd)
        self.assertIsNotNone(entry, "No dispatch handler for: %s" % cmd)
        raw = dispatch_command(self.brain, cmd, args, [])
        self.assertIsInstance(raw, dict, "%s handler returned non-dict: %r" % (cmd, raw))
        self.assertIs(raw.get("ok"), True,
                      "%s handler must return the {'ok': True, 'result': ...} "
                      "envelope (raw return reads as 'Unknown daemon error'), got: %r"
                      % (cmd, raw))
        self.assertIn("result", raw, "%s handler envelope missing 'result'" % cmd)
        return raw["result"]

    # ── Core memory operations ──

    def test_recall(self):
        """recall returns results list."""
        result = self._dispatch("recall", {"query": "test", "limit": 3})
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)

    def test_remember(self):
        """remember stores a node and returns its ID."""
        result = self._dispatch("remember", {
            "type": "lesson", "title": "Test lesson",
            "content": "This is a test lesson for round-trip verification."
        })
        self.assertIn("id", result)
        self.assertTrue(len(result["id"]) > 0)
        # V5 enrichment loop is retired — the response must not carry its prompt.
        # Asserted here, on a real dispatched response, not on remember()'s source.
        self.assertNotIn("enrichment_prompt", result)

    def test_remember_then_recall(self):
        """remember -> recall: stored node should be findable."""
        r1 = self._dispatch("remember", {
            "type": "decision", "title": "Use Arctic v1.5 for embeddings",
            "content": "Chose Arctic v1.5 because it balances quality and speed."
        })
        node_id = r1["id"]

        r2 = self._dispatch("recall", {"query": "Arctic embedding model choice", "limit": 5})
        found_ids = [r["id"] for r in r2["results"]]
        self.assertIn(node_id, found_ids,
                      "Freshly stored node not found by recall — embedding may have failed")

    def test_remember_batch(self):
        """remember_batch stores multiple nodes in one call."""
        result = self._dispatch("remember_batch", {
            "nodes": [
                {"type": "concept", "title": "Batch node A", "content": "First batch node"},
                {"type": "concept", "title": "Batch node B", "content": "Second batch node"},
            ],
        })
        self.assertIn("nodes_created", result)
        self.assertEqual(result["nodes_created"], 2)

    def test_remember_batch_refuses_a_non_object_entry_loudly(self):
        """A string where a node spec belongs is a whole-call error naming the
        index — never a raised AttributeError, which aborts the encoder's whole
        run (id:23a29491)."""
        raw = dispatch_command(self.brain, "remember_batch", {"nodes": [
            {"type": "concept", "title": "Good entry", "content": "A well-formed node"},
            "not an object"]}, [])
        self.assertIs(raw.get("ok"), False)
        self.assertIn("nodes[1]", raw["error"])
        self.assertIn("got str", raw["error"])

    def test_remember_batch_unwraps_a_json_encoded_entry(self):
        """A string element that parses to a dict is the intended spec — the
        serialization quirk brain_batch already tolerates, one level down."""
        import json as _json
        result = self._dispatch("remember_batch", {"nodes": [
            _json.dumps({"type": "concept", "title": "Stringified entry", "content": "Arrived JSON-encoded"})]})
        self.assertEqual(result["nodes_created"], 1)

    def test_revise_batch_refuses_a_non_object_entry_loudly(self):
        raw = dispatch_command(self.brain, "revise_batch", {"revisions": ["not an object"]}, [])
        self.assertIs(raw.get("ok"), False)
        self.assertIn("revisions[0]", raw["error"])

    def test_connect_batch_reports_a_non_object_entry_per_item(self):
        result = self._dispatch("connect_batch", {"connections": ["not an object"]})
        self.assertEqual(result["edges_created"], 0)
        self.assertEqual(result["failures"], 1)
        self.assertIn("connections[0]", result["failure_details"][0]["reason"])

    def test_revise(self):
        """revise updates an existing node."""
        n = self._dispatch("remember", {
            "type": "lesson", "title": "Revise me", "content": "Original content"
        })
        result = self._dispatch("revise", {
            "node_id": n["id"], "content": "Revised content", "reason": "test"
        })
        self.assertIn("id", result)
        self.assertTrue(result.get("revised_at"))
        # Verify content changed
        row = self.brain.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (n["id"],)).fetchone()
        self.assertEqual(row[0], "Revised content")

    def test_revise_batch(self):
        """revise_batch updates multiple nodes in one call."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "Rev batch A", "content": "A"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "Rev batch B", "content": "B"})
        result = self._dispatch("revise_batch", {
            "revisions": [
                {"node_id": n1["id"], "content": "A revised", "reason": "test"},
                {"node_id": n2["id"], "content": "B revised", "reason": "test"},
            ]
        })
        self.assertIn("revised", result)
        self.assertEqual(result["revised"], 2)

    def test_set_node_lock(self):
        """set_node_lock two-phase confirm: token round-trip flips the flag."""
        n = self._dispatch("remember", {
            "type": "principle", "title": "Lock me", "content": "canonical"})
        phase1 = self._dispatch("set_node_lock", {
            "node_id": n["id"], "locked": True, "reason": "roundtrip test"})
        self.assertTrue(phase1["confirmation_required"])
        phase2 = self._dispatch("set_node_lock", {
            "node_id": n["id"], "locked": True, "reason": "roundtrip test",
            "confirm_token": phase1["confirm_token"]})
        self.assertTrue(phase2["changed"])
        row = self.brain.conn.execute(
            "SELECT locked FROM nodes WHERE id = ?", (n["id"],)).fetchone()
        self.assertEqual(row[0], 1)

    def test_connect(self):
        """connect creates an edge between two nodes."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "Node A", "content": "First node"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "Node B", "content": "Second node"})
        self._dispatch("connect", {
            "source_id": n1["id"], "target_id": n2["id"],
            "relation": "related_to", "weight": 0.8
        })
        edge = self.brain.conn.execute(
            "SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
            (n1["id"], n2["id"], n2["id"], n1["id"])).fetchone()
        self.assertIsNotNone(edge, "Edge should exist")
        rel = self.brain.conn.execute(
            "SELECT relation, weight FROM edge_relations WHERE edge_id = ? AND relation = 'related_to'",
            (edge[0],)).fetchone()
        self.assertIsNotNone(rel, "Edge relation should exist after connect")
        self.assertEqual(rel[0], "related_to")

    def test_connect_batch(self):
        """connect_batch creates multiple edges in one call."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "CB A", "content": "A"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "CB B", "content": "B"})
        n3 = self._dispatch("remember", {"type": "concept", "title": "CB C", "content": "C"})
        result = self._dispatch("connect_batch", {
            "connections": [
                {"source_id": n1["id"], "target_id": n2["id"], "relation": "related_to"},
                {"source_id": n2["id"], "target_id": n3["id"], "relation": "related_to"},
            ]
        })
        self.assertIn("edges_created", result)
        self.assertEqual(result["edges_created"], 2)

    def test_revise_edge(self):
        """revise_edge renames a relation + updates its description in place,
        identified by (source_id, target_id, relation). Create a node pair +
        edge, revise it through the dispatch path, assert the rename and the
        new description persisted on the edge_relations row."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "RE A", "content": "First node"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "RE B", "content": "Second node"})
        self._dispatch("connect", {
            "source_id": n1["id"], "target_id": n2["id"],
            "relation": "related_to", "weight": 0.5, "description": "original"
        })
        result = self._dispatch("revise_edge", {
            "source_id": n1["id"], "target_id": n2["id"], "relation": "related_to",
            "new_relation": "depends_on", "description": "revised description",
            "reason": "test"
        })
        self.assertTrue(result.get("ok") or result.get("edge_id"))
        self.assertEqual(result["relation"], "depends_on")
        # The new relation persisted; the old one is no longer active.
        edge_id = self.brain.conn.execute(
            "SELECT edge_id FROM edges WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
            (n1["id"], n2["id"], n2["id"], n1["id"])).fetchone()[0]
        rel = self.brain.conn.execute(
            "SELECT description FROM edge_relations "
            "WHERE edge_id = ? AND relation = 'depends_on' AND archived_at IS NULL",
            (edge_id,)).fetchone()
        self.assertIsNotNone(rel, "renamed relation 'depends_on' should be active after revise_edge")
        self.assertEqual(rel[0], "revised description")
        stale = self.brain.conn.execute(
            "SELECT 1 FROM edge_relations "
            "WHERE edge_id = ? AND relation = 'related_to' AND archived_at IS NULL",
            (edge_id,)).fetchone()
        self.assertIsNone(stale, "old relation 'related_to' should no longer be active")

    def test_brain_batch(self):
        """brain_batch runs mixed operations in one call."""
        result = self._dispatch("brain_batch", {
            "operations": [
                {"op": "remember", "type": "concept", "title": "Batch op node", "content": "Created via brain_batch"},
            ]
        })
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1)

    def test_recall_batch(self):
        """recall_batch runs multiple queries in one call."""
        result = self._dispatch("recall_batch", {
            "queries": ["test query one", "test query two"],
            "limit": 3
        })
        # Returns a list of per-query results
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    # ── Lookup operations ──

    def test_find_node_by_title(self):
        """find_node_by_title locates nodes by fuzzy title matching."""
        self._dispatch("remember", {
            "type": "decision", "title": "Use Arctic v1.5 for embeddings",
            "content": "Chose Arctic v1.5 because it balances quality and speed."
        })
        result = self._dispatch("find_node_by_title", {
            "title_query": "Arctic embedding model", "threshold": 0.5
        })
        self.assertIsNotNone(result)
        self.assertIn("id", result)
        self.assertGreater(result["similarity"], 0.5)

    def test_find_node_by_title_no_match(self):
        """find_node_by_title returns None when nothing matches."""
        result = self._dispatch("find_node_by_title", {
            "title_query": "completely unrelated xyzzy topic", "threshold": 0.9
        })
        self.assertIsNone(result)

    def test_get_node(self):
        """get_node returns full node data by ID."""
        n = self._dispatch("remember", {"type": "concept", "title": "Get me", "content": "Full content here"})
        result = self._dispatch("get_node", {"node_id": n["id"]})
        self.assertIn("id", result)
        self.assertEqual(result["title"], "Get me")

    def test_get_nodes(self):
        """get_nodes returns multiple nodes by ID."""
        n1 = self._dispatch("remember", {"type": "concept", "title": "Multi A", "content": "A"})
        n2 = self._dispatch("remember", {"type": "concept", "title": "Multi B", "content": "B"})
        result = self._dispatch("get_nodes", {"node_ids": [n1["id"], n2["id"]]})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_get_nodes_overflow_is_named(self):
        """Past GET_NODES_MAX_IDS the door serves the first page and NAMES the
        rest in an actions note — the cap used to be a silent slice, and the
        ids past it weren't even listed as not found."""
        from servers.brain_mcp import render_reply
        from servers.contract import GET_NODES_MAX_IDS, reply_notes
        n = self._dispatch("remember", {"type": "concept", "title": "Served one", "content": "S"})
        asked = [n["id"]] + ['%08x' % i for i in range(GET_NODES_MAX_IDS + 2)]
        reply = dispatch_command(self.brain, "get_nodes", {"node_ids": asked}, [])
        self.assertIs(reply.get("ok"), True)
        # The served page answered, and every id in it is accounted for.
        self.assertEqual(len(reply["result"]), GET_NODES_MAX_IDS)
        self.assertEqual(reply["result"][0]["title"], "Served one")
        (note,) = reply_notes(reply)["actions"]
        self.assertIn("call again", note)
        for dropped in asked[GET_NODES_MAX_IDS:]:
            self.assertIn(dropped, note)
        # The render both readers share shows the note after the nodes.
        self.assertIn("Actions:\n" + note, render_reply("get_nodes", reply))

    def test_get_nodes_duplicate_ids_dont_spend_the_cap(self):
        """Repeats are deduped BEFORE the cap: a repeat that spent a slot
        would drop a distinct id that fit, and flag a fully-served request."""
        from servers.contract import GET_NODES_MAX_IDS, reply_notes
        n = self._dispatch("remember", {"type": "concept", "title": "Dupe one", "content": "D"})
        uniq = [n["id"]] + ['%08x' % i for i in range(GET_NODES_MAX_IDS - 6)]
        asked = uniq[:10] + uniq[:10] + uniq[10:]
        self.assertGreater(len(asked), GET_NODES_MAX_IDS)   # raw list overflows
        reply = dispatch_command(self.brain, "get_nodes", {"node_ids": asked}, [])
        self.assertEqual(reply_notes(reply), {}, "flagged a fully-served request")
        # Every distinct id served once: the one real node, then the misses.
        self.assertEqual([r["id"] for r in reply["result"]], uniq)

    def test_get_nodes_refuses_a_non_list(self):
        """A stringified list would be fetched character by character — the
        door refuses it and names what it got."""
        reply = dispatch_command(self.brain, "get_nodes",
                                 {"node_ids": '["abcd1234"]'}, [])
        self.assertIs(reply.get("ok"), False)
        self.assertIn("got str", reply["error"])

    def test_get_nodes_non_string_entries_stay_per_id(self):
        """A non-string entry is looked up as its text and reported per id,
        like an unknown or empty one — it never fails the ids around it, and
        past the cap it is named in the note (which a raw None would break)."""
        from servers.contract import GET_NODES_MAX_IDS, reply_notes
        n = self._dispatch("remember", {"type": "concept", "title": "Kept", "content": "K"})
        result = self._dispatch("get_nodes", {"node_ids": [n["id"], None, 7, ""]})
        self.assertEqual(result[0]["title"], "Kept")
        self.assertEqual([r["id"] for r in result[1:]], ["None", "7", ""])
        self.assertTrue(all(r["error"] == "not found" for r in result[1:]))
        asked = ['%08x' % i for i in range(GET_NODES_MAX_IDS)] + [None]
        reply = dispatch_command(self.brain, "get_nodes", {"node_ids": asked}, [])
        self.assertIn("None", reply_notes(reply)["actions"][0])

    def test_filter_nodes(self):
        """filter_nodes returns {'nodes': [...], 'total_count': N} for a
        structural-field query (field='type', no include/exclude → all
        non-archived nodes). NOTE: the old comment here claimed 'returns
        distinct values' — the DAL never did that; it always returns the
        nodes+count shape. Asserting the real contract catches shape drift."""
        result = self._dispatch("filter_nodes", {"field": "type"})
        self.assertIsInstance(result, dict)
        self.assertIn("nodes", result)
        self.assertIsInstance(result["nodes"], list)
        self.assertIn("total_count", result)
        self.assertIsInstance(result["total_count"], int)

    # ── Introspection ──

    # NOTE: test_engineering_context removed 2026-04-26.
    # The engineering_context tool was removed 2026-04-13 (was a stub
    # returning {}).
    # Roundtrip coverage of a removed tool is meaningless.

    def test_eval(self):
        """eval executes arbitrary Python on brain object."""
        result = self._dispatch("eval", {"code": "brain.conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]"})
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_gate(self):
        """gate reads and flips the run switch in the user env file — under a
        temp XDG_CONFIG_HOME here, never the operator's ~/.config."""
        import tempfile
        from unittest import mock
        cfg = tempfile.mkdtemp()
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": cfg}):
            os.environ.pop("BRAIN_PAUSED", None)
            result = self._dispatch("gate", {"action": "status"})
            self.assertFalse(result["paused"])
            self.assertEqual(result["env_file"], os.path.join(cfg, "brain", "env"))
            self.assertTrue(self._dispatch("gate", {"action": "pause"})["paused"])
            self.assertFalse(self._dispatch("gate", {"action": "resume"})["paused"])

    # ── Trace/log queries ──

    def test_query_logs(self):
        """query_logs returns {'entries': [...], 'counts': {...}} (DAL contract)."""
        result = self._dispatch("query_logs", {"hours": 1})
        self.assertIsInstance(result, dict)
        self.assertIn("entries", result)
        self.assertIsInstance(result["entries"], list)
        self.assertIn("counts", result)
        self.assertIsInstance(result["counts"], dict)

    def test_query_traces(self):
        """query_traces default mode returns {'events': [...]} (flat recent)."""
        result = self._dispatch("query_traces", {"hours": 1, "limit": 5})
        self.assertIsInstance(result, dict)
        self.assertIn("events", result)
        self.assertIsInstance(result["events"], list)

    def test_get_trace(self):
        """get_trace returns the full row for a known trace_id, or error for missing."""
        # Append a trace and read it back
        tid = self.brain._trace_dal.append(
            chain_id='roundtrip-get-trace', scale='s0', event_type='K',
            ref_type='user_message', summary='roundtrip get_trace probe')
        result = self._dispatch("get_trace", {"trace_id": tid})
        self.assertIsInstance(result, dict)
        self.assertEqual(result['id'], tid)
        self.assertEqual(result['summary'], 'roundtrip get_trace probe')

    def test_get_traces(self):
        """get_traces returns the {traces, missing_ids} envelope; an id that
        matched nothing is NAMED, not silently skipped.
        v29: ids are 8-char hex strings (reviewer F2 — int input rejected
        loudly). The phantom id 'deadbeef' won't match any real row."""
        a = self.brain._trace_dal.append(
            chain_id='roundtrip-get-traces', scale='s0', event_type='K',
            ref_type='user_message', summary='get_traces probe A')
        b = self.brain._trace_dal.append(
            chain_id='roundtrip-get-traces', scale='s0', event_type='delta',
            ref_type='assistant_message', summary='get_traces probe B')
        result = self._dispatch("get_traces", {"trace_ids": [a, b, 'deadbeef']})
        self.assertIsInstance(result, dict)
        self.assertEqual({r['id'] for r in result['traces']}, {a, b})
        self.assertEqual(result['missing_ids'], ['deadbeef'])
        self.assertNotIn('truncated', result)

    def test_get_traces_overflow_is_flagged(self):
        """Past TRACE_BATCH_MAX_IDS the door serves a page and SAYS SO — the
        cap used to be a silent list slice."""
        from servers.trace_contract import TRACE_BATCH_MAX_IDS
        tid = self.brain._trace_dal.append(
            chain_id='roundtrip-overflow', scale='s0', event_type='K',
            ref_type='user_message', summary='overflow probe')
        asked = [tid] + ['%08x' % i for i in range(TRACE_BATCH_MAX_IDS + 3)]
        result = self._dispatch("get_traces", {"trace_ids": asked})
        self.assertIn('truncated', result)
        self.assertEqual(result['truncated']['limit'], TRACE_BATCH_MAX_IDS)
        self.assertEqual(result['truncated']['dropped_ids'],
                         asked[TRACE_BATCH_MAX_IDS:])
        self.assertIn('call again', result['truncated']['note'])
        # The note must NAME the dropped ids, not just count them: the banner
        # renders only `note`, and the caller can't re-derive the set (the cap
        # applies after dedupe, so slicing their own input at 50 is wrong).
        for dropped in result['truncated']['dropped_ids']:
            self.assertIn(dropped, result['truncated']['note'])
        # The served page still answered: the one real id is in it.
        self.assertEqual([r['id'] for r in result['traces']], [tid])

    def test_get_traces_duplicate_ids_dont_spend_the_cap(self):
        """Repeats are deduped BEFORE the cap. Expanding source_refs across
        nodes repeats shared anchors; a repeat that spends a slot would drop a
        unique id that fit (silently — missing_ids only covers what was looked
        up) and would flag truncation on a fully-served request."""
        from servers.trace_contract import TRACE_BATCH_MAX_IDS
        tid = self.brain._trace_dal.append(
            chain_id='roundtrip-dupes', scale='s0', event_type='K',
            ref_type='user_message', summary='dupe probe')
        # cap+5 entries, but only cap-5 distinct — every unique id must be served.
        uniq = [tid] + ['%08x' % i for i in range(TRACE_BATCH_MAX_IDS - 6)]
        asked = uniq[:10] + uniq[:10] + uniq[10:]
        self.assertGreater(len(asked), TRACE_BATCH_MAX_IDS)   # raw list overflows
        self.assertLess(len(set(asked)), TRACE_BATCH_MAX_IDS)  # unique set fits
        result = self._dispatch("get_traces", {"trace_ids": asked})
        self.assertNotIn('truncated', result, 'flagged a fully-served request')
        # Nothing unique was skipped: every id is either a row or named missing.
        accounted = {r['id'] for r in result['traces']} | set(result['missing_ids'])
        self.assertEqual(accounted, set(asked))
        self.assertEqual([r['id'] for r in result['traces']], [tid])


    def test_count_traces(self):
        """count_traces returns {event_type: int}. Append two known events and
        verify they're counted — stronger than the old isinstance-dict check."""
        for _ in range(2):
            self.brain._trace_dal.append(
                chain_id="roundtrip-count", scale="s1", event_type="O",
                ref_type="recall", summary="count probe")
        result = self._dispatch("count_traces", {"field": "event_type"})
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(v, int) for v in result.values()),
                        "count values must be ints")
        self.assertGreaterEqual(result.get("O", 0), 2,
                                "two appended 'O' events not reflected in counts")

    def test_recall_episodes(self):
        """recall_episodes: lexical `contains` over trace_events returns the
        full matching episode records (not nodes), scoped + attributed. The
        decoy without the token must not leak through the filter."""
        for et, rt, summ in (
                ('K', 'user_message', 'zorblax episode one'),
                ('delta', 'assistant_message', 'zorblax episode two'),
                ('K', 'user_message', 'totally unrelated decoy')):
            self.brain._trace_dal.append(
                chain_id='roundtrip-ep', scale='s0', event_type=et,
                ref_type=rt, summary=summ, session_id='roundtrip-ep-sess')

        result = self._dispatch("recall_episodes",
                                {"contains": "zorblax", "limit": 10})
        self.assertIn("episodes", result)
        self.assertEqual(result["ranked_by"], "time")
        summaries = [e["summary"] for e in result["episodes"]]
        self.assertEqual(len(summaries), 2,
                         "contains filter returned wrong count: %r" % summaries)
        self.assertTrue(all("zorblax" in s for s in summaries),
                        "contains leaked non-matching episodes: %r" % summaries)
        # Both matches present (order-insensitive — created_at ties are
        # microsecond-fragile), attribution + metadata carried as full records.
        self.assertEqual(set(summaries),
                         {"zorblax episode one", "zorblax episode two"})
        self.assertTrue(all(e["session_id"] == "roundtrip-ep-sess"
                            for e in result["episodes"]))
        self.assertIn("metadata", result["episodes"][0])

    def test_recall_episodes_session_scope_and_sort(self):
        """session_id scopes the pull; sort_order=asc returns oldest-first."""
        for summ in ('scoped alpha', 'scoped beta', 'scoped gamma'):
            self.brain._trace_dal.append(
                chain_id='roundtrip-ep2', scale='s0', event_type='K',
                ref_type='user_message', summary=summ,
                session_id='roundtrip-scope-sess')
        result = self._dispatch("recall_episodes", {
            "session_id": "roundtrip-scope-sess",
            "sort_order": "asc", "limit": 50})
        summaries = [e["summary"] for e in result["episodes"]]
        self.assertEqual(summaries[:3], ['scoped alpha', 'scoped beta', 'scoped gamma'],
                         "asc sort / session scope wrong: %r" % summaries)
        self.assertTrue(all(e["session_id"] == "roundtrip-scope-sess"
                            for e in result["episodes"]),
                        "session scope leaked other sessions")

    def test_recall_episodes_and_query_traces_filter_by_the_shown_short(self):
        """The session filters take a session's full id or the short trace
        headers show; both forms pull the same rows, in session_id and in
        session_ids alike."""
        from servers.trace_contract import session_short
        full = '0199aa11-2b3c-7d4e-8f50-61a2b3c4d5e6'
        other = '0199aa11-2b3c-7d4e-8f50-000000000001'
        for sess in (full, other):
            self.brain._trace_dal.append(
                chain_id='short-%s' % sess, scale='s0', event_type='K',
                ref_type='user_message', summary='shortfilter token',
                session_id=sess)
        for cmd, key in (("recall_episodes", "episodes"), ("query_traces", "events")):
            by_full = {e["id"] for e in self._dispatch(
                cmd, {"session_id": full, "limit": 50})[key]}
            by_short = {e["id"] for e in self._dispatch(
                cmd, {"session_id": session_short(full), "limit": 50})[key]}
            self.assertTrue(by_full, cmd)
            self.assertEqual(by_short, by_full, cmd)
            both = {e["session_id"] for e in self._dispatch(
                cmd, {"session_ids": [session_short(full), other], "limit": 50})[key]}
            self.assertEqual(both, {full, other}, cmd)

    def test_session_filter_short_resolves_only_at_the_door(self):
        """Internal callers pass full ids to brain.query_traces /
        recall_episodes and stay exact: a short reaching them directly
        matches nothing, because resolution lives at the model-facing door."""
        from servers.trace_contract import session_short
        full = '0199aa11-2b3c-7d4e-8f50-0000000000aa'
        self.brain._trace_dal.append(
            chain_id='exact-%s' % full, scale='s0', event_type='K',
            ref_type='user_message', summary='exact token', session_id=full)
        short = session_short(full)
        self.assertEqual(self.brain.query_traces(session_id=short)['events'], [])
        self.assertEqual(self.brain.recall_episodes(session_id=short)['episodes'], [])
        self.assertTrue(self._dispatch("query_traces", {"session_id": short})["events"])

    def test_recall_episodes_and_query_traces_ambiguous_short_is_loud(self):
        """Two recorded sessions sharing a short: the door refuses the short
        and names both full ids."""
        a = '44444444-4444-4aaa-8bbb-cccccccccccc'
        b = '55555555-5555-4aaa-8bbb-cccccccccccc'
        for sess in (a, b):
            self.brain._trace_dal.append(
                chain_id='amb-%s' % sess, scale='s0', event_type='K',
                ref_type='user_message', summary='amb token', session_id=sess)
        for cmd in ("recall_episodes", "query_traces"):
            raw = dispatch_command(self.brain, cmd, {"session_id": "cccccccccccc"}, [])
            self.assertIs(raw.get("ok"), False, cmd)
            self.assertIn(a, raw["error"])
            self.assertIn(b, raw["error"])

    def test_query_traces_chain_id_does_not_resolve_the_session_filter(self):
        """chain_id overrides every other filter, so a session filter beside it
        is neither resolved nor applied — an unknown short does not fail it."""
        self.brain._trace_dal.append(
            chain_id='chain-only', scale='s0', event_type='K',
            ref_type='user_message', summary='chain token',
            session_id='chain-only-sess')
        chain = self._dispatch("query_traces", {
            "chain_id": "chain-only", "session_id": "f00dfacecafe"})["chain"]
        self.assertEqual([e["summary"] for e in chain], ["chain token"])

    def test_recall_episodes_and_query_traces_unresolved_short_is_loud(self):
        """A short that names no recorded session is an error, never an empty
        result that reads as "nothing happened in that session"."""
        for cmd in ("recall_episodes", "query_traces"):
            for args in ({"session_id": "f00dfacecafe"},
                         {"session_ids": ["f00dfacecafe"]}):
                raw = dispatch_command(self.brain, cmd, dict(args), [])
                self.assertIs(raw.get("ok"), False, (cmd, args))
                self.assertIn(cmd, raw["error"])
                self.assertIn(
                    "no session in the trace history matches 'f00dfacecafe'",
                    raw["error"])

    def test_recall_episodes_and_query_traces_empty_session_ids_is_loud(self):
        """session_ids that names no session — [] or an empty entry — is an
        error: downstream an empty scope is no scope, so the pull would come
        back with every session's rows, reading as a scoped answer."""
        for sess in ('empty-scope-A', 'empty-scope-B'):
            self.brain._trace_dal.append(
                chain_id='empty-%s' % sess, scale='s0', event_type='K',
                ref_type='user_message', summary='empty scope token', session_id=sess)
        for cmd in ("recall_episodes", "query_traces"):
            for sids in ([], [""], ["", "empty-scope-A"]):
                raw = dispatch_command(self.brain, cmd, {"session_ids": sids}, [])
                self.assertIs(raw.get("ok"), False, (cmd, sids, raw))
                self.assertIn("must name at least one session", raw["error"])

    def test_query_traces_hours_bound_scoped_pull_says_so(self):
        """A scoped pull that stays hours-bound (ref_type, grouped) carries an
        information note naming the window — the rendered rows alone read as
        the whole session. A grouped pull with no scope is refused."""
        from servers.brain_mcp import render_reply
        self.brain._trace_dal.append(
            chain_id='hb-chain', scale='s0', event_type='K',
            ref_type='user_message', summary='hours bound token',
            session_id='hours-bound-sess')
        for args in ({"session_ids": ["hours-bound-sess"], "grouped": True},
                     {"session_id": "hours-bound-sess", "ref_type": "user_message"}):
            raw = dispatch_command(self.brain, "query_traces", dict(args, hours=24), [])
            self.assertIs(raw.get("ok"), True, raw)
            self.assertIn("last 24h", render_reply("query_traces", raw), args)
        raw = dispatch_command(self.brain, "query_traces", {"grouped": True}, [])
        self.assertIs(raw.get("ok"), False, raw)
        self.assertIn("needs session_id or session_ids", raw["error"])

    def test_recall_episodes_semantic_ranks_by_meaning(self):
        """recall_episodes with `query` ranks by cosine over the existing
        trace_embeddings, not by recency. Seed two embedded episodes (the
        semantically-far one appended LAST, so recency would mis-rank it
        first) and confirm meaning wins."""
        from servers import embedder
        pet = self.brain._trace_dal.append(
            chain_id='roundtrip-sem', scale='s0', event_type='K',
            ref_type='user_message',
            summary='cats and dogs make wonderful household pets',
            session_id='roundtrip-sem-sess')
        db = self.brain._trace_dal.append(
            chain_id='roundtrip-sem', scale='s0', event_type='K',
            ref_type='user_message',
            summary='sqlite index migration and schema version drift',
            session_id='roundtrip-sem-sess')
        for tid, text in ((pet, 'cats and dogs make wonderful household pets'),
                          (db, 'sqlite index migration and schema version drift')):
            self.brain._trace_dal.store_embeddings(
                [(tid, embedder.embed_batch([text], kind='document')[0], text)],
                model='test')

        result = self._dispatch("recall_episodes", {
            "query": "feline and canine companion animals",
            "session_id": "roundtrip-sem-sess", "limit": 2})
        self.assertEqual(result["ranked_by"], "relevance")
        self.assertGreaterEqual(len(result["episodes"]), 1)
        self.assertEqual(result["episodes"][0]["id"], pet,
                         "semantic rank should put the pets episode first, "
                         "not the database one")
        self.assertIn("_score", result["episodes"][0])

    def test_recall_episodes_older_than_not_clobbered_by_default_window(self):
        """older_than alone must NOT get a default 7-day younger floor forced
        on it (which would make the window empty). Regression: the default-
        window guard previously ignored older_than."""
        from servers.clock import iso_cutoff
        tid = self.brain._trace_dal.append(
            chain_id='roundtrip-old', scale='s0', event_type='K',
            ref_type='user_message', summary='ancient episode about migrations',
            session_id='roundtrip-old-sess')
        # Backdate to 10 days ago — older than both older_than='1d' and the 7d floor.
        self.brain._trace_dal.conn.execute(
            "UPDATE trace_events SET created_at = ? WHERE id = ?",
            (iso_cutoff(days=10), tid))
        result = self._dispatch("recall_episodes", {"older_than": "1d", "limit": 10})
        ids = [e["id"] for e in result["episodes"]]
        self.assertIn(tid, ids,
                      "older_than lost a 10-day-old episode — a default 7d "
                      "younger floor wrongly clobbered the older_than window")

    def test_resolve_time_bound_parsing(self):
        """Relative shorthand is case-insensitive; unparseable bounds raise
        instead of silently binding a non-timestamp into a lex comparison."""
        from servers.brain_traces import _resolve_time_bound
        self.assertEqual(_resolve_time_bound(''), '')
        self.assertRegex(_resolve_time_bound('3d'), r'^\d{4}-\d{2}-\d{2}')
        # Case-insensitive: uppercase units are parsed to a timestamp, not
        # passed through verbatim (the pre-fix silent-empty bug).
        self.assertRegex(_resolve_time_bound('2H'), r'^\d{4}-\d{2}-\d{2}')
        self.assertNotEqual(_resolve_time_bound('2H'), '2H')
        # Naive literals are normalized to tz-aware UTC to match storage
        # ('+00:00'); already-aware passes through; space form → ISO-T (avoids
        # the 'T' > ' ' lex hazard). Out-of-range / trailing-junk dates raise.
        self.assertEqual(_resolve_time_bound('2026-06-14T00:00:00'),
                         '2026-06-14T00:00:00+00:00')
        self.assertEqual(_resolve_time_bound('2026-06-14T00:00:00+00:00'),
                         '2026-06-14T00:00:00+00:00')
        self.assertEqual(_resolve_time_bound('2026-06-14 12:00:00'),
                         '2026-06-14T12:00:00+00:00')
        # An offset-bearing literal is CONVERTED to UTC, not passed through
        # wearing its own offset: as text '…12:00:00+03:00' sorts AFTER
        # '…09:30:00+00:00' while naming an earlier instant, so the window it
        # bounds silently shifted.
        self.assertEqual(_resolve_time_bound('2026-06-14T12:00:00+03:00'),
                         '2026-06-14T09:00:00+00:00')
        for junk in ('1mo', 'garbage', '7', '2026-13-99', '2026-06-14xyz'):
            with self.assertRaises(ValueError):
                _resolve_time_bound(junk)

    def test_recall_episodes_ref_type_dial_optin_firehose(self):
        """Default ref_type lens is PINNED to operator dialogue
        (OPERATOR_DIALOGUE_REF_TYPES — deliberately NOT the dial, Option A):
        tool_result, heartbeat, structural noise, AND brain/session deliveries
        stay out of the default even when a correspondent is enabled for
        encoding. Everything else is opt-in: a single ref_type opts into one
        lens; a list is the interleaved said+did firehose."""
        sess = 'roundtrip-tr-sess'
        msg = self.brain._trace_dal.append(
            chain_id='roundtrip-tr', scale='s0', event_type='K',
            ref_type='user_message', summary='discuss the dal refactor',
            session_id=sess)
        tool = self.brain._trace_dal.append(
            chain_id='roundtrip-tr', scale='s0', event_type='delta',
            ref_type='tool_result', summary='Edit: servers/dal.py filter_events',
            session_id=sess)
        beat = self.brain._trace_dal.append(
            chain_id='roundtrip-tr', scale='s0', event_type='K',
            ref_type='heartbeat', summary='watch re-arm', session_id=sess)
        dlv = self.brain._trace_dal.append(
            chain_id='roundtrip-tr', scale='s0', event_type='K',
            ref_type='thalamus_delivery', ref_id='stop',
            summary='delivered 1 thalamus item(s) at stop', session_id=sess)

        # Default = operator dialogue only: user_message in; tool_result,
        # heartbeat, and the delivery all out — even with the dial flipped
        # (the pin is what keeps "what we said" stable across a flip).
        from unittest import mock
        from servers import trace_contract as tc
        with mock.patch.dict(tc.S0_CONVERSATIONAL_INCOMING,
                             {'thalamus_delivery': True}):
            default = {e["id"] for e in self._dispatch(
                "recall_episodes", {"session_id": sess, "limit": 50})["episodes"]}
        self.assertIn(msg, default)
        self.assertNotIn(tool, default)
        self.assertNotIn(beat, default)
        self.assertNotIn(dlv, default,
                         "a delivery entered the default lens — the episodes "
                         "default must be pinned, not dial-derived (Option A)")
        # The delivery stays reachable by explicit opt-in.
        dlvs = {e["id"] for e in self._dispatch(
            "recall_episodes", {"session_id": sess,
                                "ref_type": "thalamus_delivery",
                                "limit": 50})["episodes"]}
        self.assertEqual(dlvs, {dlv})
        # Single opt-in lens.
        tools = {e["id"] for e in self._dispatch(
            "recall_episodes", {"session_id": sess, "ref_type": "tool_result",
                                "limit": 50})["episodes"]}
        self.assertEqual(tools, {tool})
        # List = interleaved said+did firehose.
        fire = {e["id"] for e in self._dispatch(
            "recall_episodes",
            {"session_id": sess,
             "ref_type": ["user_message", "tool_result", "heartbeat"],
             "limit": 50})["episodes"]}
        self.assertEqual(fire, {msg, tool, beat})

    def test_recall_episodes_render(self):
        """The MCP renderer turns episode records into attributed lines (not raw
        JSON): who · when · trace-handle · body. Locks the attribution coupling
        to metadata keys — a non-default identity must appear verbatim, so a key
        rename (which would fall back to 'Anchor') fails this test."""
        from servers.brain_mcp import _format_result
        rendered = _format_result("recall_episodes", {
            "ranked_by": "relevance",
            "episodes": [{
                "id": "abc12345", "ref_type": "assistant_message",
                "session_id": "c3a2e74c-x",
                "created_at": "2026-06-14T02:58:56+00:00",
                "metadata": {"agent_identity": "Anchor-7",
                             "content": "the cadence fix landed"},
                "_score": 0.5123}]})
        self.assertIn("Anchor-7", rendered)          # reads agent_identity
        self.assertIn("trace:abc12345", rendered)    # expansion handle
        self.assertIn("the cadence fix landed", rendered)
        self.assertIn("0.51", rendered)              # score rendered
        self.assertNotIn("{", rendered)              # not raw JSON

    def test_caller_session_stamped_never_as_session_id(self):
        """The MCP proxy stamps the calling session under the reserved
        `_caller_session` key — NEVER `session_id`. That keeps session_id a
        pure caller-supplied filter, so cross-session reads (recall_episodes,
        query_traces) default to all sessions. Identity stays available under
        its own name; an explicit caller-supplied session_id is preserved.

        (Adapted from the old recall_episodes-specific denylist test: the fix
        is now command-agnostic — no command is special-cased.)"""
        import os
        from servers.brain_mcp import _stamp_caller_session, CALLER_SESSION_KEY
        prev = os.environ.get("CLAUDE_CODE_SESSION_ID")
        os.environ["CLAUDE_CODE_SESSION_ID"] = "sess-current"
        try:
            stamped = _stamp_caller_session({})
            self.assertEqual(stamped.get(CALLER_SESSION_KEY), "sess-current",
                             "identity must be stamped under _caller_session")
            self.assertNotIn("session_id", stamped,
                             "session_id must never be auto-injected — it is a "
                             "pure caller-supplied filter now")
            # An explicit caller filter is preserved untouched, alongside the
            # ambient identity.
            explicit = _stamp_caller_session({"session_id": "x"})
            self.assertEqual(explicit["session_id"], "x")
            self.assertEqual(explicit.get(CALLER_SESSION_KEY), "sess-current")
        finally:
            if prev is None:
                os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
            else:
                os.environ["CLAUDE_CODE_SESSION_ID"] = prev

    def test_recall_episodes_no_scope_is_cross_session(self):
        """recall_episodes with the ambient `_caller_session` present but NO
        explicit session_id must span ALL sessions. This is the freshly-awoken
        session case: identity is known, but an unscoped recall reaches all of
        itself — the original bug auto-scoped it to the newborn session."""
        from servers.dispatch_common import CALLER_SESSION_KEY
        for sess in ('re-xsess-A', 're-xsess-B'):
            self.brain._trace_dal.append(
                chain_id='re-%s' % sess, scale='s0', event_type='O',
                ref_type='user_message', summary='rexsess shared token',
                session_id=sess)
        eps = self._dispatch("recall_episodes", {
            "contains": "rexsess shared token", "limit": 50,
            CALLER_SESSION_KEY: 're-xsess-A'})["episodes"]
        sessions = {e["session_id"] for e in eps}
        self.assertEqual(
            sessions, {'re-xsess-A', 're-xsess-B'},
            "_caller_session scoped recall_episodes to the calling session — "
            "the all-sessions default was overridden (the original bug)")

    def test_query_traces_no_scope_is_cross_session(self):
        """query_traces shares recall_episodes' contract: the ambient
        `_caller_session` must NOT scope the pull (the latent bug the stopgap
        denylist never covered). An omitted session_id spans all sessions."""
        from servers.dispatch_common import CALLER_SESSION_KEY
        for sess in ('qt-xsess-A', 'qt-xsess-B'):
            self.brain._trace_dal.append(
                chain_id='qt-%s' % sess, scale='s0', event_type='O',
                ref_type='user_message', summary='qtxsess token',
                session_id=sess)
        events = self._dispatch("query_traces", {
            "hours": 1, "limit": 500, CALLER_SESSION_KEY: 'qt-xsess-A'})["events"]
        sessions = {e["session_id"] for e in events}
        self.assertIn('qt-xsess-A', sessions)
        self.assertIn('qt-xsess-B', sessions,
                      "_caller_session scoped query_traces to the calling "
                      "session — the all-sessions default was overridden")

    def test_caller_session_reaches_attribution_handler(self):
        """An identity (attribution) command must reach the calling session via
        the ambient `_caller_session` key alone — no explicit session_id, which
        the write tool schemas don't surface. Proves identity survived the move
        off the overloaded session_id arg: the node_revised trace is attributed
        to the caller."""
        from servers.dispatch_common import CALLER_SESSION_KEY
        sess = "attrib-sess-9f1c"
        node_id = self._dispatch("remember", {
            "type": "lesson", "title": "Attribution probe",
            "content": "original content for the attribution probe."})["id"]
        # Revise carrying ONLY the ambient identity key — exactly what the MCP
        # proxy stamps for a tool whose schema doesn't expose session_id.
        self._dispatch("revise", {
            "node_id": node_id, "reason": "attribution test",
            "content": "revised content, forcing a delta and its trace.",
            CALLER_SESSION_KEY: sess})
        events = self._dispatch("query_traces", {
            "session_id": sess, "ref_type": "node_revised"})["events"]
        self.assertTrue(events,
                        "no node_revised trace attributed to the calling "
                        "session via _caller_session")
        self.assertTrue(all(e["session_id"] == sess for e in events),
                        "attribution handler did not stamp the caller session")

    def test_connect_attributed_to_stamp_not_written_session_id(self):
        """A session_id the model writes beside the stamp is not identity: the
        edge trace belongs to the stamped caller, and nothing lands under the
        written value."""
        from servers.dispatch_common import CALLER_SESSION_KEY
        sess, written = "connect-attrib-5b1e", "written-5b1e"
        a = self._dispatch("remember", {"type": "concept", "title": "Stamp attrib A",
                                        "content": "node A for stamped connect attribution."})["id"]
        b = self._dispatch("remember", {"type": "concept", "title": "Stamp attrib B",
                                        "content": "node B for stamped connect attribution."})["id"]
        self._dispatch("connect", {
            "source_id": a, "target_id": b, "relation": "grounds_stamp_probe",
            "description": "edge whose trace must follow the stamped caller.",
            CALLER_SESSION_KEY: sess, "session_id": written})
        edges = self.brain.query_traces(session_id=sess, hours=None,
                                        ref_type="edge_relation_revised")["events"]
        self.assertTrue(edges, "connect's edge trace was not attributed to the stamp")
        self.assertEqual(self.brain.query_traces(
            session_id=written, hours=None, ref_type="edge_relation_revised")["events"], [])

    def test_remember_emits_attributed_node_created_and_edge_traces(self):
        """Step 6 pins. (a) remember through dispatch emits a node_created
        trace — the row that closes the partial-run catalog gap; zero of
        these existed before the emitter went live. (b) remember-path
        connect_to edge traces carry the caller session — the pop-then-read
        bug (_pop_session_ctx mutates args, the legacy emit read
        them after) shipped session_id='' on EVERY such trace; the chokepoint
        captures identity pre-handler, killing the bug structurally."""
        from servers.dispatch_common import CALLER_SESSION_KEY
        sess = "remember-attrib-3c9d"
        tgt = self._dispatch("remember", {
            "type": "concept", "title": "Created-trace target",
            "content": "target for the attribution edge."})["id"]
        src = self._dispatch("remember", {
            "type": "concept", "title": "Created-trace source",
            "content": "source node whose traces must be attributed.",
            CALLER_SESSION_KEY: sess,
            "connect_to": [{"title": tgt, "relation": "grounds",
                            "why": "attribution probe edge for step 6"}],
        })["id"]
        created = self._dispatch("query_traces", {
            "session_id": sess, "ref_type": "node_created"})["events"]
        self.assertTrue(created,
                        "remember emitted no caller-attributed node_created")
        self.assertTrue(any(e["ref_id"] == src for e in created))
        edges = self._dispatch("query_traces", {
            "session_id": sess, "ref_type": "edge_relation_revised"})["events"]
        self.assertTrue(edges,
                        "remember's connect_to edge trace lost the caller "
                        "session — the pop-then-read class is back")

    def test_brain_batch_sub_ops_attributed_to_caller(self):
        """EVERY brain_batch sub-op (revise, connect — not just remember) must
        attribute its trace to the caller carried under the ambient
        _caller_session key. Guards the per-op-type attribution split: a sub-op
        that fails to propagate identity emits an unattributed (session_id='')
        trace, invisible to per-session forensics. The fix propagates identity
        under the reserved key into every op's op_args."""
        from servers.dispatch_common import CALLER_SESSION_KEY
        sess = "batch-attrib-7a2e"
        a = self._dispatch("remember", {"type": "concept", "title": "Batch attrib A",
                                        "content": "node A for batch attribution."})["id"]
        b = self._dispatch("remember", {"type": "concept", "title": "Batch attrib B",
                                        "content": "node B for batch attribution."})["id"]
        res = self._dispatch("brain_batch", {
            CALLER_SESSION_KEY: sess,
            "operations": [
                {"op": "revise", "node_id": a, "reason": "batch attrib revise",
                 "content": "revised A content forcing a delta and its trace."},
                {"op": "connect", "source_id": a, "target_id": b,
                 "relation": "relates_to_batch",
                 "description": "edge A->B created inside a batch for attribution coverage."},
            ]})
        self.assertEqual(res["failed"], 0, "batch ops failed: %r" % res)
        revised = self._dispatch("query_traces",
                                 {"session_id": sess, "ref_type": "node_revised"})["events"]
        edges = self._dispatch("query_traces",
                               {"session_id": sess, "ref_type": "edge_relation_revised"})["events"]
        self.assertTrue(revised, "batch revise op emitted no caller-attributed node_revised trace")
        self.assertTrue(edges, "batch connect op emitted no caller-attributed edge_relation_revised trace")
        self.assertTrue(all(e["session_id"] == sess for e in revised + edges),
                        "a brain_batch sub-op trace was not attributed to the caller session")

    def test_brain_batch_sub_ops_take_the_batch_caller_over_their_own(self):
        """An op that carries its own `_caller_session` or session_id still
        runs as the batch's caller — the proxy scrubs only the top level, so
        the batch assigns identity to every op rather than defaulting it."""
        from unittest import mock
        from servers import dispatch_write
        from servers.dispatch_common import CALLER_SESSION_KEY, caller_session
        seen = []

        def _record(brain, op_args, graph_changes):
            seen.append(caller_session(op_args))
            return {"ok": True, "result": {}}

        with mock.patch.object(dispatch_write, "_handle_revise", _record), \
                mock.patch.object(dispatch_write, "_handle_remember", _record), \
                mock.patch.object(dispatch_write, "_handle_connect", _record):
            self._dispatch("brain_batch", {
                CALLER_SESSION_KEY: "batch-caller-4d1a",
                "operations": [
                    {"op": "revise", "node_id": "n1", "reason": "r",
                     "content": "c", CALLER_SESSION_KEY: "op-stamp"},
                    {"op": "revise", "node_id": "n2", "reason": "r",
                     "content": "c", "session_id": "op-sid"},
                    {"op": "remember", "type": "concept", "title": "t",
                     "content": "c", CALLER_SESSION_KEY: "op-stamp"},
                    {"op": "connect", "source_id": "a", "target_id": "b",
                     "relation": "grounds", "description": "d",
                     CALLER_SESSION_KEY: "op-stamp"}]})
        self.assertEqual(seen, ["batch-caller-4d1a"] * 4)

    def test_check_unknown_keys_exempts_caller_session(self):
        """check_unknown_keys (run by the daemon BEFORE every handler) must not
        flag the proxy-stamped _caller_session against a handler's `accepts`
        set — else every accepts-guarded write logs dispatch_unknown_keys. The
        _dispatch helper bypasses this guard, so exercise it directly."""
        from servers.daemon_dispatch import COMMAND_TABLE
        from servers.dispatch_common import check_unknown_keys, CALLER_SESSION_KEY
        entry = COMMAND_TABLE["connect"]  # has an `accepts` frozenset
        self.assertIsNotNone(entry.accepts, "test assumes connect declares accepts")
        logged = []
        orig = self.brain._log_error
        self.brain._log_error = lambda *a, **k: logged.append((a, k))
        try:
            check_unknown_keys("connect", entry,
                               {"source_id": "x", "target_id": "y", CALLER_SESSION_KEY: "s"},
                               self.brain)
            self.assertEqual(logged, [], "_caller_session was wrongly flagged as an unknown key")
            # The exemption is not a blanket pass — a real unknown key still flags.
            check_unknown_keys("connect", entry,
                               {"source_id": "x", "bogus_key": 1}, self.brain)
            self.assertTrue(logged, "a genuinely unknown key was not flagged")
        finally:
            self.brain._log_error = orig

    def test_stamp_then_dispatch_reads_are_cross_session(self):
        """End-to-end composition: what the proxy ACTUALLY stamps
        (_stamp_caller_session) fed into the filter reads must still span all
        sessions. The handler-only cross-session tests inject _caller_session by
        hand; this guards the proxy half — a regression re-injecting session_id
        at daemon_send for reads would slip past those."""
        import os
        from servers.brain_mcp import _stamp_caller_session
        for sess in ('compose-A', 'compose-B'):
            self.brain._trace_dal.append(
                chain_id='compose-%s' % sess, scale='s0', event_type='O',
                ref_type='user_message', summary='compose token', session_id=sess)
        prev = os.environ.get("CLAUDE_CODE_SESSION_ID")
        os.environ["CLAUDE_CODE_SESSION_ID"] = "compose-A"
        try:
            args = _stamp_caller_session({"contains": "compose token", "limit": 50})
            self.assertNotIn("session_id", args, "proxy injected session_id into a read")
            eps = self._dispatch("recall_episodes", args)["episodes"]
            self.assertEqual({e["session_id"] for e in eps}, {"compose-A", "compose-B"},
                             "stamp+dispatch scoped recall_episodes to the caller session")
            qt = self._dispatch("query_traces",
                                _stamp_caller_session({"hours": 1, "limit": 500}))["events"]
            seen = {e["session_id"] for e in qt}
            self.assertIn("compose-A", seen)
            self.assertIn("compose-B", seen)
        finally:
            if prev is None:
                os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
            else:
                os.environ["CLAUDE_CODE_SESSION_ID"] = prev

    def test_stamp_caller_session_scrubs_forged_key(self):
        """The proxy is the SOLE writer of _caller_session: with the session env
        unset it must SCRUB an inbound _caller_session to an empty stamp rather
        than let the daemon honor it as identity (MCP payloads can carry
        arbitrary keys)."""
        import os
        from servers.brain_mcp import _stamp_caller_session, CALLER_SESSION_KEY
        prev = os.environ.get("CLAUDE_CODE_SESSION_ID")
        os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        try:
            scrubbed = _stamp_caller_session({CALLER_SESSION_KEY: "forged-session", "x": 1})
            self.assertEqual(scrubbed.get(CALLER_SESSION_KEY), "",
                             "a forged _caller_session was trusted when the env was unset")
            self.assertEqual(scrubbed.get("x"), 1, "scrub clobbered unrelated args")
        finally:
            if prev is not None:
                os.environ["CLAUDE_CODE_SESSION_ID"] = prev

    def test_unattributed_proxy_call_cannot_claim_a_written_session(self):
        """A proxy call the host gave no identity reaches the daemon with an
        EMPTY stamp, so a session_id the model writes is not read as identity:
        self_inbox(session_id=<sibling>) leaves the sibling's inbox alone. A
        direct daemon_client caller (the poller, hooks) writes no stamp and
        still names itself with session_id."""
        import os
        from servers.brain_mcp import _stamp_caller_session
        from servers.channels.self_channel import signal
        sibling = "51b11ae0-1111-2222-3333-444444444444"
        self._dispatch("self_send", {"to": sibling, "body": "for the sibling",
                                     "from_session": "roundtrip-sender"})
        prev = os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        try:
            args = _stamp_caller_session({"session_id": sibling})
        finally:
            if prev is not None:
                os.environ["CLAUDE_CODE_SESSION_ID"] = prev
        refused = dispatch_command(self.brain, "self_inbox", args, [])
        self.assertIs(refused.get("ok"), False, refused)
        self.assertEqual([m["body"] for m in signal.peek_inbox(self.brain, sibling)],
                         ["for the sibling"],
                         "an unattributed proxy call drained another session's inbox")
        # The sender side too: a written from_session neither sends as nor
        # reads the outbox of the session it names.
        prev = os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        try:
            send = _stamp_caller_session({"to": sibling, "body": "spoofed",
                                          "from_session": "roundtrip-sender"})
            outbox = _stamp_caller_session({"from_session": "roundtrip-sender"})
        finally:
            if prev is not None:
                os.environ["CLAUDE_CODE_SESSION_ID"] = prev
        for cmd, cmd_args in (("self_send", send), ("self_outbox", outbox)):
            raw = dispatch_command(self.brain, cmd, cmd_args, [])
            self.assertIs(raw.get("ok"), False, (cmd, raw))
        # A direct daemon_client caller (no stamp) still names itself.
        drained = self._dispatch("self_inbox", {"session_id": sibling})
        self.assertEqual([m["body"] for m in drained["messages"]], ["for the sibling"])

    def test_unattributed_live_remind_cannot_broadcast_as_a_written_session(self):
        """remind(for_whom='live') speaks as the filing session, so thalamus
        refuses it without one. An unattributed proxy call is anonymous: a
        session_id the model writes beside it must not become that sender."""
        import os
        from servers.brain_mcp import _stamp_caller_session
        sibling = "51b11ae0-2222-3333-4444-555555555555"
        prev = os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
        try:
            args = _stamp_caller_session({"what": "spoofed live broadcast",
                                          "for_whom": "live", "session_id": sibling})
        finally:
            if prev is not None:
                os.environ["CLAUDE_CODE_SESSION_ID"] = prev
        raw = dispatch_command(self.brain, "remind", args, [])
        self.assertIs(raw.get("ok"), False, raw)
        self.assertIn("requires a filing session", raw["error"])
        sent = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM self_inflight WHERE body = 'spoofed live broadcast'"
        ).fetchone()[0]
        self.assertEqual(sent, 0, "an unattributed call broadcast as the session it named")

    def test_recall_episodes_all_scales_not_collapsed(self):
        """scale='' (all scales) must NOT collapse to the s0 conversational
        whitelist — an s1/s2 trace must be reachable. Regression for the
        `(scale or 's0')` guard that treated '' like 's0'."""
        s1 = self.brain._trace_dal.append(
            chain_id='roundtrip-allscale', scale='s1', event_type='delta',
            ref_type='encoding_run', summary='zqx encode run',
            session_id='roundtrip-allscale-sess')
        result = self._dispatch("recall_episodes",
                                {"contains": "zqx", "scale": "", "limit": 50})
        ids = {e["id"] for e in result["episodes"]}
        self.assertIn(s1, ids,
                      "scale='' dropped an s1 trace — 'all scales' collapsed to "
                      "the s0 conversational lens")

    def test_recall_episodes_embed_unavailable_is_loud(self):
        """A semantic query that can't be embedded (embedder returns None) must
        log a breadcrumb and degrade to time — not silently answer by recency."""
        from unittest import mock
        from servers import embedder
        logged = []
        with mock.patch.object(embedder, "embed_query", return_value=None), \
             mock.patch.object(self.brain, "_log_error",
                               side_effect=lambda *a, **k: logged.append(a)):
            result = self.brain.recall_episodes(query="anything", limit=5)
        self.assertEqual(result["ranked_by"], "time")
        self.assertTrue(
            any("embed_unavailable" in str(a[0]) for a in logged),
            "embedder-unavailable degrade was silent — no _log_error fired")

    # ── Interactions ──

    def test_list_interactions(self):
        """list_interactions returns registered interactions. Register a probe
        and confirm it appears by name — stronger than isinstance-list."""
        self._dispatch("register_interaction", {
            "name": "roundtrip_list_probe", "template": "probe",
            "parameters": "{}", "created_by": "roundtrip_test"})
        result = self._dispatch("list_interactions", {})
        self.assertIsInstance(result, list)
        names = {i.get("name") for i in result}
        self.assertIn("roundtrip_list_probe", names)

    def test_get_interaction(self):
        """get_interaction returns the stored interaction row. Register a probe
        with a known template, then read it back. The old check
        (`result is None or isinstance(result, dict)`) passed even on the
        handler's {'ok': False, 'error': ...} envelope — it asserted nothing
        about correctness. get_active returns id/name/version/template/parameters."""
        self._dispatch("register_interaction", {
            "name": "roundtrip_get_probe", "template": "PROBE TEMPLATE BODY",
            "parameters": "{}", "created_by": "roundtrip_test"})
        # Registration never activates — deploy v1 so there is an active row.
        self._dispatch("set_interaction_active", {
            "name": "roundtrip_get_probe", "version": 1,
            "set_by": "roundtrip_test"})
        result = self._dispatch("get_interaction", {"name": "roundtrip_get_probe"})
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("name"), "roundtrip_get_probe")
        self.assertEqual(result.get("template"), "PROBE TEMPLATE BODY")

    def test_get_interaction_effective(self):
        """get_interaction_effective returns the RESOLVED value: the code
        default when no override is deployed, the override-overlaid value
        when one is, with the stamp's source flipping — and refuses a name
        with no code default (nothing to resolve)."""
        from servers.interaction_defaults import INTERACTION_DEFAULTS
        default_template, default_config = INTERACTION_DEFAULTS["s1e"]

        result = self._dispatch("get_interaction_effective", {"name": "s1e"})
        self.assertEqual(result.get("name"), "s1e")
        # Bounded by default: preview + length, no full body.
        self.assertNotIn("template", result)
        self.assertEqual(result.get("template_preview"), default_template[:400])
        self.assertEqual(result.get("template_length"), len(default_template))
        self.assertEqual(result.get("config"), default_config)
        self.assertEqual(result.get("stamp", {}).get("source"), "default")
        full = self._dispatch("get_interaction_effective",
                              {"name": "s1e", "include_template": True})
        self.assertEqual(full.get("template"), default_template)

        # Deploy a config-only override: the resolved config overlays the
        # one key, the template stays the default, the stamp flips.
        reg = self._dispatch("register_interaction", {
            "name": "s1e", "template": "",
            "parameters": '{"effort": "roundtrip-effective"}',
            "created_by": "roundtrip_test"})
        self._dispatch("set_interaction_active", {
            "name": "s1e", "version": reg["registered_version"],
            "set_by": "roundtrip_test"})
        eff = self._dispatch("get_interaction_effective",
                             {"name": "s1e", "include_template": True})
        self.assertEqual(eff["config"].get("effort"), "roundtrip-effective")
        self.assertEqual(eff["template"], default_template)
        self.assertEqual(eff["stamp"]["source"], "override")
        for key, value in default_config.items():
            if key != "effort":
                self.assertEqual(eff["config"].get(key), value)

        # Clear restores the pure default (leaves the fixture clean too).
        self._dispatch("clear_interaction_override", {"name": "s1e"})
        back = self._dispatch("get_interaction_effective", {"name": "s1e"})
        self.assertEqual(back["stamp"]["source"], "default")

        # A name with no code default has nothing to resolve — refused.
        raw = dispatch_command(self.brain, "get_interaction_effective",
                               {"name": "roundtrip_no_default"}, [])
        self.assertIs(raw.get("ok"), False)
        self.assertIn("unknown interaction", raw.get("error", ""))

    def test_clear_interaction_override(self):
        """clear_interaction_override deletes the pointer, reports 'cleared'
        vs 'no pointer existed' distinctly, and refuses typo'd names."""
        # A deployed non-registry name (the legacy-cleanup path): the delete
        # succeeds regardless of the code-default registry.
        self._dispatch("register_interaction", {
            "name": "roundtrip_clear_probe", "template": "OVERRIDE BODY",
            "parameters": "{}", "created_by": "roundtrip_test"})
        self._dispatch("set_interaction_active", {
            "name": "roundtrip_clear_probe", "version": 1,
            "set_by": "roundtrip_test"})
        result = self._dispatch("clear_interaction_override",
                                {"name": "roundtrip_clear_probe"})
        self.assertIs(result.get("cleared"), True)

        # A registry name with no pointer: distinct "already on the default".
        again = self._dispatch("clear_interaction_override", {"name": "s1e"})
        self.assertIs(again.get("cleared"), False)
        self.assertIn("code default", again.get("note", ""))

        # A typo'd name (nothing deleted, no code default): refused, never
        # a benign success.
        raw = dispatch_command(self.brain, "clear_interaction_override",
                               {"name": "trace_recoding"}, [])
        self.assertIs(raw.get("ok"), False)
        self.assertIn("unknown interaction", raw.get("error", ""))

    def test_register_interaction(self):
        """register_interaction adds a new interaction version."""
        result = self._dispatch("register_interaction", {
            "name": "test_roundtrip_interaction",
            "template": "Hello, {name}!",
            "parameters": '{"max_messages": 5}',
            "created_by": "roundtrip_test",
        })
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get('registered_version'), 1)
        # Registration NEVER activates — the door must report that honestly,
        # not tell the operator its write is live.
        self.assertIsNone(result.get('active_version'))
        self.assertIn('did NOT activate', result.get('note', ''))

    def test_set_interaction_active(self):
        """set_interaction_active flips the active version pointer."""
        # First register two versions so we have something to flip between
        self._dispatch("register_interaction", {
            "name": "test_active_flip",
            "template": "v1 content",
            "parameters": '{}',
            "created_by": "roundtrip_test",
        })
        self._dispatch("register_interaction", {
            "name": "test_active_flip",
            "template": "v2 content",
            "parameters": '{}',
            "created_by": "roundtrip_test",
        })
        # Now flip active to v2
        result = self._dispatch("set_interaction_active", {
            "name": "test_active_flip",
            "version": 2,
            "set_by": "roundtrip_test",
        })
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("version"), 2,
                         f"Expected version 2, got: {result}")

    def test_clear_errors(self):
        """clear_errors deletes hook_errors and reports the rowcount. Insert a
        hook error, clear, and confirm it's actually gone — the old check only
        asserted the return was a dict."""
        self.brain.logs_conn.execute(
            "INSERT INTO hook_errors (created_at, hook_name, level, error) "
            "VALUES (?, ?, ?, ?)",
            ("2026-05-29T00:00:00+00:00", "test_hook", "error", "probe error"))
        self.brain.logs_conn.commit()
        result = self._dispatch("clear_errors", {})
        self.assertIsInstance(result, dict)
        self.assertIn("hook_errors", result)
        self.assertGreaterEqual(result["hook_errors"], 1)
        remaining = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM hook_errors").fetchone()[0]
        self.assertEqual(remaining, 0, "hook_errors not actually cleared")

    # ── Coverage check ──

    # ── Self channel (presence — pull) ──

    def test_self_presence(self):
        """self_presence returns a roster of live sessions + a rendered line.

        Liveness/focus come from real-turn S0 traces (present_sessions reads
        traces, not session_state), so seed a user_message trace for the session."""
        self.brain._trace_dal.append(
            chain_id='s0-rtstrmA-0', scale='s0', event_type='K',
            ref_type='user_message', summary='roundtrip focus A',
            session_id='roundtrip-session-A')
        result = self._dispatch("self_presence", {"session_id": "roundtrip-self", "limit": 5})
        self.assertIn("sessions", result)
        self.assertIn("line", result)
        self.assertIsInstance(result["sessions"], list)
        ids = {s["session_id"] for s in result["sessions"]}
        self.assertIn("roundtrip-session-A", ids)

    def test_self_peek(self):
        """self_peek returns one session's full current focus."""
        self.brain.set_config('session_context_roundtrip-peek',
                              'peek focus line one\nline two')
        result = self._dispatch("self_peek", {"session_id": "roundtrip-peek"})
        self.assertTrue(result["found"])
        self.assertEqual(result["focus"], "peek focus line one\nline two")

    def test_self_send(self):
        """self_send places a directed message in the courier. A full session
        UUID is the canonical address — honored directly (resolve_to), no
        live-roster lookup needed."""
        recipient = "11111111-2222-3333-4444-555555555555"
        result = self._dispatch("self_send", {
            "to": recipient, "body": "tap on the shoulder",
            "from_session": "roundtrip-sender"})
        self.assertIn("id", result)
        self.assertEqual(result["address"], "self:" + recipient)

    def test_self_inbox(self):
        """self_inbox drains messages addressed to the caller, consume-once."""
        recipient = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        self._dispatch("self_send", {
            "to": recipient, "body": "you have mail",
            "from_session": "roundtrip-sender"})
        result = self._dispatch("self_inbox", {"session_id": recipient})
        self.assertIn("you have mail", [m["body"] for m in result["messages"]])
        again = self._dispatch("self_inbox", {"session_id": recipient})
        self.assertNotIn("you have mail", [m["body"] for m in again["messages"]])

    def test_self_outbox(self):
        """self_outbox returns delivery status of messages the caller SENT —
        the sender-side receipt view (who drained, and whether a directed
        target is still pending)."""
        sender = "deadbeef-0000-1111-2222-333344445555"
        recipient = "cafebabe-0000-1111-2222-333344445555"
        self._dispatch("self_send", {
            "to": recipient, "body": "did you get this?", "from_session": sender})
        result = self._dispatch("self_outbox", {"from_session": sender})
        self.assertIn("messages", result)
        directed = [m for m in result["messages"]
                    if m.get("target") == session_short(recipient)]
        self.assertTrue(directed, "outbox must show the directed message just sent")
        # nobody has drained it yet → still pending (silence read correctly)
        self.assertTrue(directed[0]["pending"])
        self.assertEqual(directed[0]["delivered_to"], [])

    # ── Thalamus (the brain speaking to its sessions) ──

    def test_remind(self):
        """remind files a Thalamus item — the producer verb. An ask
        (needs_answer) defaults to the every_session audience and queues for
        boot delivery; the item lands open with a window."""
        result = self._dispatch("remind", {
            "what": "roundtrip: decide the thing", "needs_answer": True})
        self.assertTrue(result["ok"])
        self.assertTrue(result["id"].startswith("th_"))
        self.assertEqual(result["route"], "queue")
        self.assertEqual(result["audience"], "every_session")
        self.assertTrue(result["expires_at"])

    def test_remind_for_whom_accepts_a_live_sessions_short(self):
        """A directed reminder names its session the way self_send names a
        target: full id or short, resolved against the live sessions. The item
        stores the full id; a short matching no live session is refused loudly,
        and the audience words pass through untouched."""
        from servers.trace_contract import session_short
        full = 'aaaaaaaa-1111-4222-8333-555555555555'
        self.brain.stamp_boot_liveness(full)
        filed = self._dispatch("remind", {
            "what": "roundtrip: for one session", "for_whom": session_short(full)})
        target = self.brain.logs_conn.execute(
            "SELECT target_session FROM thalamus_items WHERE id = ?",
            (filed["id"],)).fetchone()[0]
        self.assertEqual(target, full)
        raw = dispatch_command(self.brain, "remind", {
            "what": "roundtrip: nobody", "for_whom": "f00dfacecafe"}, [])
        self.assertIs(raw.get("ok"), False)
        self.assertIn("no live session matches 'f00dfacecafe'", raw["error"])
        self.assertEqual(self._dispatch("remind", {
            "what": "roundtrip: everyone", "for_whom": "all"})["audience"],
            "every_session")

    def test_remind_expires_sets_the_window(self):
        """`expires` reaches the door: it replaces the kind's default window,
        and one that ends before the item is due is refused loudly."""
        default = self._dispatch("remind", {
            "what": "roundtrip: default window", "when": "1d"})
        short = self._dispatch("remind", {
            "what": "roundtrip: short window", "when": "1d", "expires": "2d"})
        self.assertLess(short["expires_at"], default["expires_at"])
        raw = dispatch_command(self.brain, "remind", {
            "what": "roundtrip: ends before due", "when": "3d",
            "expires": "1d"}, [])
        self.assertIs(raw.get("ok"), False)
        self.assertIn("before it ever becomes due", raw.get("error", ""))

    def test_thalamus_list(self):
        """thalamus_list returns open items with delivery counts."""
        filed = self._dispatch("remind", {"what": "roundtrip: listed item"})
        result = self._dispatch("thalamus_list", {})
        self.assertEqual(result["count"], 1)
        item = result["items"][0]
        self.assertEqual(item["id"], filed["id"])
        self.assertEqual(item["body"], "roundtrip: listed item")
        self.assertEqual(item["deliveries"], 0)

    def test_thalamus_resolve(self):
        """thalamus_resolve closes an item; a second close is a loud error,
        and the answer payload rides the item."""
        filed = self._dispatch("remind", {
            "what": "roundtrip: answer me", "needs_answer": True})
        result = self._dispatch("thalamus_resolve", {
            "id": filed["id"], "answer": "the answer"})
        self.assertEqual(result["state"], "answered")
        raw = dispatch_command(self.brain, "thalamus_resolve",
                               {"id": filed["id"], "dismiss": True}, [])
        self.assertIs(raw.get("ok"), False,
                      "re-resolving a closed item must be a loud error")

    def test_all_mcp_tools_have_roundtrip_tests(self):
        """Every MCP tool should have a corresponding test above."""
        mcp_tool_names = {t["name"] for t in TOOLS}
        test_methods = {m for m in dir(self) if m.startswith("test_") and m != "test_all_mcp_tools_have_roundtrip_tests"}

        tested = set()
        for method in test_methods:
            tool_name = method.replace("test_", "").replace("_then_recall", "").replace("_no_match", "")
            if tool_name in mcp_tool_names:
                tested.add(tool_name)

        # Commands that can't be round-trip tested (they end the serve loop —
        # restart reloads the daemon in place, shutdown exits it)
        untestable = {'restart', 'shutdown'}
        untested = mcp_tool_names - tested - untestable
        self.assertEqual(untested, set(),
                         "MCP tools without round-trip tests: %s" % untested)

    def test_roundtrip_tests_assert_on_content(self):
        """Gate against smoke-test theater: every MCP tool's round-trip test
        must assert on actual keys/values, not merely that *something* came back.

        A test whose only assertion is `assertIsInstance(result, dict/list)` on
        the bare top-level result passes no matter what the handler returns — it
        can't catch a handler that drifts to the wrong shape. This AST check
        requires each tool to have at least one test method with a substantive
        assertion (assertEqual / assertIn / assertGreater / ... or an
        assertIsInstance on a SUBSCRIPT like result['key'], which checks a
        sub-field rather than the envelope).

        Pairs with test_all_mcp_tools_have_roundtrip_tests: that gate ensures
        coverage *exists*; this one ensures the coverage *bites*. Without it,
        the easy way to satisfy the coverage gate is an isinstance smoke test
        (bucket B of the 2026-05 redundancy hunt was exactly that backlog)."""
        import ast
        import inspect
        import textwrap

        SUBSTANTIVE = {
            'assertEqual', 'assertNotEqual', 'assertIn', 'assertNotIn',
            'assertGreater', 'assertGreaterEqual', 'assertLess', 'assertLessEqual',
            'assertTrue', 'assertFalse', 'assertIsNone', 'assertIsNotNone',
            'assertAlmostEqual', 'assertListEqual', 'assertDictEqual', 'assertRegex',
        }

        def _is_substantive(method_name):
            src = textwrap.dedent(inspect.getsource(getattr(type(self), method_name)))
            for node in ast.walk(ast.parse(src)):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)):
                    continue
                attr = node.func.attr
                if attr in SUBSTANTIVE:
                    return True
                # assertIsInstance(result['key'], ...) inspects a sub-field's
                # shape, not just the top-level envelope — that counts.
                if (attr == 'assertIsInstance' and node.args
                        and isinstance(node.args[0], ast.Subscript)):
                    return True
            return False

        mcp_tool_names = {t["name"] for t in TOOLS}
        untestable = {'restart', 'shutdown'}

        # Map each tool to the test methods exercising it (same name-stripping
        # as the coverage gate above), skipping the two meta-gates themselves.
        tool_methods = {}
        for method in dir(self):
            if not method.startswith("test_"):
                continue
            if method in ('test_all_mcp_tools_have_roundtrip_tests',
                          'test_roundtrip_tests_assert_on_content'):
                continue
            tool_name = method.replace("test_", "").replace("_then_recall", "").replace("_no_match", "")
            if tool_name in mcp_tool_names:
                tool_methods.setdefault(tool_name, []).append(method)

        theater = []
        for tool, methods in sorted(tool_methods.items()):
            if tool in untestable:
                continue
            if not any(_is_substantive(m) for m in methods):
                theater.append("%s (only: %s)" % (tool, ", ".join(sorted(methods))))

        self.assertEqual(theater, [],
                         "MCP tools whose round-trip tests assert only type, not "
                         "content (smoke-test theater — strengthen to assert "
                         "keys/values): %s" % theater)


class TestCriticalToolsAlwaysLoad(unittest.TestCase):
    """Tool-search deferral contract.

    Claude Code defers MCP tools behind ToolSearch when they'd exceed ~10% of
    the context window (the default). CRITICAL_TOOLS opt out via
    anthropic/alwaysLoad in their _meta so the brain's hot-path tools load
    eagerly for EVERY installer — the flag ships in tools/list, not in any
    user's settings.json. These gates lock that contract: the right tools are
    eager, the long tail still defers. Pure module inspection — no brain.
    """

    def test_critical_names_are_real_tools(self):
        """Every CRITICAL_TOOLS name must match an actual tool. A typo would
        silently defer a tool we meant eager. _stamp_always_load raises at
        import on mismatch; this locks the same invariant as a test."""
        names = {t["name"] for t in TOOLS}
        self.assertTrue(
            CRITICAL_TOOLS <= names,
            "CRITICAL_TOOLS contains non-existent tool(s): %s"
            % sorted(CRITICAL_TOOLS - names))

    def test_critical_tools_carry_always_load(self):
        """Each critical tool emits _meta['anthropic/alwaysLoad'] == True."""
        for t in TOOLS:
            if t["name"] in CRITICAL_TOOLS:
                self.assertEqual(
                    t.get("_meta", {}).get("anthropic/alwaysLoad"), True,
                    "%s should carry anthropic/alwaysLoad" % t["name"])

    def test_non_critical_tools_do_not_carry_always_load(self):
        """Non-critical tools must NOT be eager — otherwise the whole point
        (lean context, defer the long tail) collapses to load-everything."""
        for t in TOOLS:
            if t["name"] not in CRITICAL_TOOLS:
                self.assertNotIn(
                    "anthropic/alwaysLoad", t.get("_meta", {}),
                    "%s unexpectedly marked alwaysLoad" % t["name"])


class TestMissingEnvelopeIsLoud(unittest.TestCase):
    """brain_mcp must NAME a missing-envelope response, not bury it in the
    generic "Unknown daemon error". Locks the diagnostic added after the
    dispatch_self envelope bug (c4f6386): a handler returning a raw dict should
    produce an error that points at the envelope and shows the offending keys,
    so the next slip is a one-line diagnosis instead of a multi-turn hunt."""

    def test_raw_dict_response_yields_descriptive_error(self):
        from unittest import mock
        from servers import brain_mcp
        # daemon_send returns a raw payload dict (no ok/result/error) — exactly
        # what an un-enveloped handler produces over the wire.
        with mock.patch.object(brain_mcp, "daemon_send",
                               return_value={"sessions": [], "line": ""}):
            resp = brain_mcp.handle_tools_call(
                "req-1", {"name": "self_presence", "arguments": {}})
        payload = resp["result"]
        self.assertTrue(payload.get("isError"))
        text = payload["content"][0]["text"]
        self.assertIn("envelope", text)
        self.assertNotIn("Unknown daemon error", text)
        # the offending keys are surfaced for fast diagnosis
        self.assertIn("sessions", text)
        self.assertIn("line", text)


class TestToolCallRetryIsAtMostOnce(unittest.TestCase):
    """handle_tools_call resends a tool call only when it never ran: the
    connection was refused, the request was reset unread in a reloading
    daemon's listen queue, or the daemon declined it before running
    (NOT_RUN_FIELD, a busy write lock). A daemon that read the call may have
    committed it, so a timeout, an empty or garbled reply, or any other error
    the daemon returned is never resent: a resent `remember` lands twice.

    A stub daemon on a real socket records every request it reads; the proxy's
    wire (send_command) is pointed at it with a short timeout."""

    CLIENT_TIMEOUT = 0.3

    def setUp(self):
        import socket
        import threading
        from unittest import mock
        self.received = []     # cmd of every request the stub read
        self.replied = threading.Event()
        self.delay_first = 0.0
        self.reply = {"ok": True, "result": {"id": "abc12345"}}
        self.reply_first = None  # the stub's answer to its first request
        self.close_first = False  # read the first request, close unanswered
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()
        # A port with nothing bound to it: connect is refused.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        self.dead_port = probe.getsockname()[1]
        probe.close()
        self.refuse_first = 0  # tool calls to send to the dead port first
        for p in (mock.patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": "test-sess"}),
                  mock.patch("time.sleep")):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.listener.close()

    def _serve(self):
        import json
        import threading
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            with conn:
                cmd = json.loads(conn.makefile().readline())["cmd"]
                self.received.append(cmd)
                if self.close_first and len(self.received) == 1:
                    self.replied.set()
                    continue                   # closes with the request read
                if self.delay_first and len(self.received) == 1:
                    # The write commits, slowly. Not time.sleep: setUp
                    # mocks it to skip the proxy's backoff.
                    threading.Event().wait(self.delay_first)
                reply = self.reply
                if self.reply_first and len(self.received) == 1:
                    reply = self.reply_first
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except OSError:
                    pass                           # the client gave up
                self.replied.set()

    def _call(self, name="remember"):
        from unittest import mock
        from servers import brain_mcp
        from servers.daemon_client import send_command
        sent = []

        def wire(cmd, args=None, timeout=None, role=None):
            port = self.port
            if cmd == name:
                sent.append(cmd)
                if len(sent) <= self.refuse_first:
                    port = self.dead_port
            return send_command(cmd, args, timeout=self.CLIENT_TIMEOUT, port=port, role=role)

        with mock.patch.object(brain_mcp, "send_command", wire):
            resp = brain_mcp.handle_tools_call(
                "req-1", {"name": name, "arguments": {"title": "t", "content": "c"}})
        return resp["result"], sent

    def test_slow_daemon_write_is_sent_once(self):
        self.delay_first = self.CLIENT_TIMEOUT * 3
        out, sent = self._call()
        self.assertTrue(self.replied.wait(5), "stub never finished the write")
        self.assertEqual(sent, ["remember"])
        self.assertEqual(self.received.count("remember"), 1)
        self.assertTrue(out.get("isError"))
        self.assertIn("timeout", out["content"][0]["text"].lower())

    def test_empty_reply_after_the_daemon_read_the_call_is_not_resent(self):
        # A handler killed mid-run closes a connection it already read: the
        # write may have committed, so the proxy reports instead of resending.
        self.close_first = True
        out, sent = self._call()
        self.assertEqual(sent, ["remember"])
        self.assertEqual(self.received.count("remember"), 1)
        self.assertTrue(out.get("isError"))
        # The empty-reply path itself, not a timeout that would also send once.
        self.assertIn("empty response", out["content"][0]["text"])

    def test_daemon_error_naming_a_timeout_is_not_resent(self):
        # The daemon answered: the call ran. Its error text mentioning
        # "timeout" or "connection" says nothing about the wire.
        self.reply = {"ok": False, "error": "LLM timeout on the connection pool"}
        out, sent = self._call()
        self.assertEqual(sent, ["remember"])
        self.assertEqual(self.received.count("remember"), 1)
        self.assertTrue(out.get("isError"))

    def test_refused_connection_is_retried_until_the_daemon_is_back(self):
        # A restarting daemon refuses the first try; the retry reaches it.
        self.refuse_first = 1
        out, sent = self._call()
        self.assertEqual(sent, ["remember", "remember"])
        self.assertEqual(self.received.count("remember"), 1)
        self.assertNotIn("isError", out)

    def test_write_declined_for_a_busy_lock_is_retried(self):
        # The daemon answered without running the call: it marks the reply
        # NOT_RUN_FIELD, and the resend is the call's only run.
        from servers.contract import NOT_RUN_FIELD
        self.reply_first = {"ok": False, NOT_RUN_FIELD: True,
                            "error": "Write lock timeout — another operation is holding the lock"}
        out, sent = self._call()
        self.assertEqual(sent, ["remember", "remember"])
        self.assertNotIn("isError", out)

    def test_call_cut_off_by_a_reload_is_retried(self):
        # The daemon's listen queue held the request unread when the reload
        # closed the socket: nothing ran, so the retry is the only delivery.
        import socket
        import threading
        from unittest import mock
        from servers import brain_mcp
        from servers.daemon_client import send_command
        sent = []
        first = {}

        def wire(cmd, args=None, timeout=None, role=None):
            if cmd != "remember":
                return send_command(cmd, args, timeout=self.CLIENT_TIMEOUT, port=self.port, role=role)
            sent.append(cmd)
            if len(sent) > 1:
                return send_command(cmd, args, timeout=self.CLIENT_TIMEOUT, port=self.port, role=role)
            reloading = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reloading.bind(("127.0.0.1", 0))
            reloading.listen(1)
            threading.Timer(0.1, reloading.close).start()
            first.update(send_command(cmd, args, timeout=2.0, port=reloading.getsockname()[1]))
            return first

        with mock.patch.object(brain_mcp, "send_command", wire):
            out = brain_mcp.handle_tools_call(
                "req-1", {"name": "remember", "arguments": {"title": "t", "content": "c"}})["result"]
        self.assertEqual(first.get("transport"), "reset",
                         "the first try must hit the reload window, not a refused port")
        self.assertEqual(sent, ["remember", "remember"])
        self.assertEqual(self.received.count("remember"), 1)
        self.assertNotIn("isError", out)

    def test_refused_every_time_gives_up_after_three_tries(self):
        self.refuse_first = 99
        out, sent = self._call()
        self.assertEqual(sent, ["remember"] * 3)
        self.assertEqual(self.received.count("remember"), 0)
        self.assertTrue(out.get("isError"))


class TestAgentLimit(unittest.TestCase):
    """The shared dispatch-door limit clamp keeps every agent-facing read
    bounded. The underlying DAL read is unbounded (limit=None) for internal
    scans; this clamp sits at the dispatch door those never come through.
    One helper, parameterized per door (filter_nodes, recall_episodes).
    """

    def test_clamp(self):
        from servers.dispatch_common import _agent_limit
        # filter_nodes door (default 50, ceiling 200)
        self.assertEqual(_agent_limit(None, 50, 200), 50)     # absent → default page
        self.assertEqual(_agent_limit(10, 50, 200), 10)       # honest small page
        self.assertEqual(_agent_limit(0, 50, 200), 1)         # floor of 1
        self.assertEqual(_agent_limit(10_000, 50, 200), 200)  # never exceeds ceiling
        # recall_episodes door (default 10, ceiling 500) — same clamp, its values
        self.assertEqual(_agent_limit(None, 10, 500), 10)
        self.assertEqual(_agent_limit(9_999, 10, 500), 500)
        # A default larger than the ceiling is still clamped — the None branch
        # routes the default THROUGH the ceiling, not around it.
        self.assertEqual(_agent_limit(None, 999, 500), 500)


if __name__ == "__main__":
    unittest.main()
