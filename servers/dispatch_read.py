"""Daemon dispatch — read handlers (recall + node/graph reads).

Lock-free reads: recall, batch recall, get_node(s), title lookup, structured
filter, boot context.
"""

from .dispatch_common import caller_session, _agent_limit


def _handle_recall(brain, args, graph_changes):
    # By-ID recall: returns single enriched node (canonicalized inside
    # recall_node, so this door and the by-query one below agree)
    node_id = args.get("node_id")
    if node_id:
        result = brain.recall_node(
            node_id, session_id=caller_session(args))
        return {"ok": True, "result": result}

    sid = caller_session(args)  # identity: drives this session's fatigue/activity
    # By-query recall. brain.recall() owns its degradation (an embedder or
    # scorer failure returns empty with a loud mode), so there is nothing to
    # fall back to at this layer. If it raises, surface the failure loudly instead
    # of masking it as an empty/degraded success.
    #
    # mark_accessed defaults True (every real caller). A read-only observer —
    # the dashboard's recall probe — passes False so its looking never lands
    # in access_count / last_accessed / fatigue, which are the brain's own
    # record of what IT recalled.
    try:
        result = brain.recall(
            query=args.get("query", ""), filter=args.get("filter"),
            limit=args.get("limit", 8), session_id=sid,
            source=args.get("source") or 'mcp',
            mark_accessed=args.get("mark_accessed", True))
    except Exception as e:
        brain._log_error("recall_failed", e, "MCP by-query recall raised")
        return {"ok": False, "error": "recall failed: %s" % e}

    brain.canonicalize_results(result.get("results", []), session_id=sid)

    return {"ok": True, "result": result}


def _handle_recall_batch(brain, args, graph_changes):
    """Batch recall — N queries, each through the SAME door as single recall.

    Delegates to _handle_recall per query instead of re-running the
    recall + canonicalize sequence, so the batch cannot drift from single
    recall: one handler body, one place a failure gets logged
    (brain._log_error inside _handle_recall), one place mark_accessed /
    source are honored. The per-query shape stays compact — {query, results}
    — dropping the stats/gap chrome single recall renders, which is the
    intended batch compaction, not drift. A canonicalize failure propagates
    loud, exactly as it does for single recall — a batch is no reason to
    swallow an unreadable correction chain.
    """
    queries = args.get("queries", [])
    # Control fields shared by every query — everything except the batch's own
    # `queries` (and any by-id / single-query key a caller mis-bundled, so each
    # delegated call is an unambiguous by-query recall). Carries the
    # proxy-stamped caller session so identity resolves the same per call.
    # `limit` is injected with the batch default (5, vs single recall's 8) so
    # delegation honors the batch contract.
    base = {k: v for k, v in args.items()
            if k not in ("queries", "query", "node_id")}
    base["limit"] = args.get("limit", 5)
    results = []
    for q in queries[:10]:  # cap at 10 queries
        r = _handle_recall(brain, {**base, "query": q}, graph_changes)
        if r.get("ok"):
            results.append(
                {"query": q, "results": (r.get("result") or {}).get("results", [])})
        else:
            results.append(
                {"query": q, "results": [], "error": r.get("error", "recall failed")})
    return {"ok": True, "result": results}


def _handle_get_node(brain, args, graph_changes):
    node_id = args.get("node_id", "")
    if not node_id:
        return {"ok": False, "error": "node_id is required"}

    node = brain.get_node(node_id)
    if not node:
        return {"ok": False, "error": "Node not found: {}".format(node_id)}

    return {"ok": True, "result": node}


def _handle_get_nodes(brain, args, graph_changes):
    """Batch get_node — multiple node IDs in one call, same rich shape as get_node.

    Serves GET_NODES_MAX_IDS distinct ids per call and names the overflow.
    The result is a list, so the overflow rides the reply as an `actions`
    note (contract.reply_notes), which the MCP proxy and the encoder runner
    both render."""
    from servers.contract import (GET_NODES_MAX_IDS, add_reply_note,
                                  truncation_banner, truncation_payload_ids)
    asked = args.get("node_ids", [])
    # A non-list (a stringified list, say) would be read character by
    # character — refuse it. A non-string entry is looked up as its text, so it
    # is served or reported per id like any other, never failing the ids
    # around it.
    if not isinstance(asked, list):
        return {"ok": False,
                "error": "node_ids must be a list of node id strings, got %s"
                         % type(asked).__name__}
    # Dedupe BEFORE the cap, so a repeat never spends a slot a distinct id
    # needed; dict.fromkeys keeps caller order.
    requested = list(dict.fromkeys(
        n if isinstance(n, str) else str(n) for n in asked))
    node_ids = requested[:GET_NODES_MAX_IDS]
    real_ids = [nid for nid in node_ids if nid]
    rich_map = brain.get_node(real_ids) if real_ids else {}
    # Batch returns {requested_id: rich_dict}. Preserve request order, and
    # report each unknown or empty id instead of silently dropping it
    # (get_node's single-id door reports its miss; the batch door must match).
    results = [rich_map[nid] for nid in real_ids if nid in rich_map]
    results.extend({"id": nid or "", "error": "not found"}
                   for nid in node_ids if not (nid and nid in rich_map))
    reply = {"ok": True, "result": results}
    if len(requested) > GET_NODES_MAX_IDS:
        add_reply_note(reply, "actions", truncation_banner(
            truncation_payload_ids(requested, GET_NODES_MAX_IDS)))
    return reply


def _handle_find_node_by_title(brain, args, graph_changes):
    result = brain.find_node_by_title(
        title_query=args.get("title_query", ""),
        threshold=args.get("threshold", 0.75),
        top_k=args.get("top_k", 1),
        session_id=caller_session(args))
    return {"ok": True, "result": result}


def _handle_filter_nodes(brain, args, graph_changes):
    """Structured query: filter nodes by any structural field."""
    from servers.contract import NODE_QUERY_DEFAULT_LIMIT, NODE_QUERY_MAX_LIMIT
    result = brain.filter_nodes(
        field=args.get("field", ""),
        include=args.get("include"),
        exclude=args.get("exclude"),
        lt=args.get("lt"),
        gt=args.get("gt"),
        contains=args.get("contains"),
        prefix=args.get("prefix"),
        limit=_agent_limit(args.get("limit"),
                           NODE_QUERY_DEFAULT_LIMIT, NODE_QUERY_MAX_LIMIT),
        sort_by=args.get("sort_by", "created_at"),
        sort_order=args.get("sort_order", "desc"),
        rich=args.get("rich", True),
        session_id=caller_session(args))
    if "error" in result:
        return {"ok": False, "error": result["error"]}
    return {"ok": True, "result": result}


def _handle_context_boot(brain, args, graph_changes):
    text = brain.format_boot_context(session_id=args.get("session_id", ""))
    # Faithful boot capture: record exactly what we served this session so the
    # dashboard can show "what actually got to boot". Only on real
    # SessionStarts (session_id present); record_boot_render is best-effort and
    # never raises. The hook prints format_boot_context's string verbatim.
    sid = args.get("session_id", "")
    if sid:
        brain.record_boot_render(sid, text)
    return {"ok": True, "result": text}
