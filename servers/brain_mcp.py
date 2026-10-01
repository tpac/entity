#!/usr/bin/env python3
"""
Brain MCP Server — Thin stdio proxy to brain daemon.

Zero-dependency MCP server (JSON-RPC 2.0 over stdio).
Forwards tool calls to the brain daemon via TCP localhost.
Embedder loads once in the daemon; this process is just a relay.

Error policy: NEVER swallow errors silently. If something fails,
stderr gets a message and the caller gets a real error.
"""

import json
import os
import sys
import threading

# Ensure parent dir is on sys.path so `from servers.X` works
# even when this file is run as a standalone script (not -m servers.brain_mcp)
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# ── Daemon communication ──
# No host/port and no socket here: daemon_config owns the address, and
# daemon_client.send_command owns the wire protocol.
from servers.daemon_client import send_command
_last_daemon_fingerprint = None  # Track daemon restarts


# ── Contract-driven tool schema generation ──

# source_refs property for remember / remember_batch. The Scribe copies ids
# from the `trace="…"` attribute on its timeline turns
# (encode._render_lived_sequence_timeline); interactive sessions from the
# `(trace:<hex>)` header trace_contract.render_trace puts on episode and
# trace rows. Sparse by design — 1-3 load-bearing refs per node
# (EPISODIC-REFERENCES.md decision 13). Persisted via
# SourceRefDAL.add_source_refs into node_source_refs; invalid refs degrade
# gracefully at recall (S2Healer cleans dangling refs).
_SOURCE_REFS_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "Trace event ids anchoring this node to its originating moments. "
        "Each id is an 8-char hex string copied verbatim from the trace "
        "markers in your input — the `trace=\"<hex>\"` attribute on "
        "timeline turns, or the `(trace:<hex>)` in an episode or trace "
        "row's header. Sparse by design: pick 1-3 load-bearing turns per node — "
        "the turn(s) whose content is what made this node encodeable. "
        "Adjacent context is what graph traversal is for; source_refs are "
        "for the moments that GENERATED this node. Leave empty when the "
        "node is a multi-session abstraction with no single anchor "
        "(pure-synthesis pattern). When content would just rewrite what "
        "the source already says clearly, point to the source instead "
        "of restating it (the pure-reference pattern). See "
        "EPISODIC-REFERENCES.md §7.4 for the full judgment rule."
    ),
}


def _generate_remember_schema():
    """Generate the 'remember' MCP tool schema from the contract."""
    from servers.contract import get_writable_fields

    TYPE_MAP = {"str": "string", "float": "number", "bool": "boolean", "int": "integer"}

    properties = {}
    for name, spec in get_writable_fields().items():
        prop = {"type": TYPE_MAP.get(spec.get("type", "str"), "string")}
        if spec.get("description"):
            prop["description"] = spec["description"]
        if spec.get("default") is not None:
            prop["default"] = spec["default"]
        properties[name] = prop

    # v29 / Phase B: source_refs anchors the node to S0/S1 trace events.
    # See _SOURCE_REFS_SCHEMA below for the full semantics.
    properties["source_refs"] = _SOURCE_REFS_SCHEMA
    # Declared so a host sends it as a list: an undeclared parameter arrives
    # as a JSON string. remember_batch replaces it with its sibling-aware one.
    properties["connect_to"] = {
        "type": "array",
        "description": "Typed edges from this new node to existing nodes, each named by its id.",
        "items": REF_CONNECT_TO_ITEM,
    }

    return {
        "name": "remember",
        "description": (
            "Store a new node in the brain.\n\n"
            "ENCODING CRAFT:\n"
            "• `situation` is the single biggest lever for recall — write as "
            "\"When [doing X] and [Y happens]\". A vague situation means the node "
            "only surfaces for exact-match queries.\n"
            "• `their_raw_quote` and `my_raw_quote` capture meaning that "
            "paraphrasing loses. Use them when the operator's or your own exact "
            "words carry the principle.\n"
            "• To link a node as a correction of another, use `connect_to` "
            "with a correction-aspect relation (`corrects`, `supersedes`, "
            "`reframes`, `resolves`, `fixes`, ...) and a specific `description` — that "
            "edge becomes the recall-time correction signal.\n\n"
            "LESSONS — climb the abstraction ladder:\n"
            "  BAD: \"Fixed tokenizer bug at startup.\"\n"
            "  GOOD: \"Hidden dependencies surface at state transitions. "
            "PRINCIPLE: When a component fails at startup/shutdown, look for "
            "dependencies it shouldn't have.\"\n\n"
            "CORRECTIONS — three lines:\n"
            "  ASSUMED: what you thought\n"
            "  REALITY: what's true\n"
            "  PATTERN: the class of error\n"
            "Specific enough that you recognize the trap before falling in again.\n\n"
            "RICHNESS: Training rewards brevity; this is wrong for memory. "
            "Future-you has zero context. Be RICH — texture, specifics, failures, "
            "reasoning journeys. Many focused nodes > few compressed summaries. "
            "Encode decisions, corrections, mechanisms, quotes, emotional "
            "inflections — not just technical lessons."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["type", "title", "content"],
            "properties": properties,
        }
    }


# The connect_to item shape is the contract's (brain_batch's remember branch
# needs the same one, and contract.py cannot import brain_mcp); tools point at
# it with `$ref` and attach_defs() gives each tool the definition once.
from servers.contract import REF_CONNECT_TO_ITEM, attach_defs


def _generate_remember_batch_schema():
    """Generate the 'remember_batch' tool schema — array of remember() objects.

    source_refs is inherited from `_generate_remember_schema()` (auto-added
    to per-node properties via the base schema's property dict).
    """
    remember_schema = _generate_remember_schema()
    node_properties = dict(remember_schema["inputSchema"]["properties"])
    # Per-node connect_to: sibling-aware, sequencing-agnostic. Declaration order
    # within the batch doesn't matter — sibling resolution runs after all nodes
    # are created.
    node_properties["connect_to"] = {
        "type": "array",
        "description": (
            "Per-node typed edges from THIS node to siblings (created in the same "
            "batch) or existing nodes. Order-agnostic. "
            "USE THIS for any edge involving a new node — never use a separate "
            "`connect` op for new-node edges (`connect` requires ids that don't "
            "exist until round 1 finishes, forcing a needless second LLM round). "
            "DON'T DOUBLE-EMIT: an edge already in connect_to must NOT also appear "
            "as a separate connect op for the same pair."
        ),
        "items": REF_CONNECT_TO_ITEM,
    }
    # Per-node connect_to is the only edge surface. The old `auto_connect`
    # default fired pairwise empty-description `related_to` edges every batch;
    # its behavior was removed 2026-05-24 and the param itself 2026-06-18.
    return {
        "name": "remember_batch",
        "description": (
            "Create multiple nodes in one call. Each node uses the same fields as "
            "remember(), plus an optional per-node `connect_to` for typed edges to "
            "siblings (in the same batch) and catalog nodes."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["nodes"],
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "Array of node specs — same fields as remember(), plus optional per-node connect_to.",
                    "items": {
                        "type": "object",
                        "required": ["type", "title", "content"],
                        "properties": node_properties,
                    },
                },
                "connect_to": {
                    "type": "array",
                    "items": REF_CONNECT_TO_ITEM,
                    "description": (
                        "Batch-level: applies the same edge from EVERY created node to one "
                        "catalog target. Siblings excluded. For per-node edges, use node-level connect_to."
                    ),
                },
            },
        },
    }


def _build_brain_batch_op_items():
    """Discriminated union over brain_batch ops — one oneOf branch per op.

    Derives entirely from contract.BATCH_OP_SPECS (required lists, property
    fragments, branch descriptions) so the schema, the dispatcher pre-check,
    and the S2 invalid-op detector cannot drift. Branch order = dict order =
    the probe-validated artifact (eval/mcp_variants/v2_oneof_trimmed.json).
    remember/revise/absorb accept open-ended node fields by design; a spec's
    additionalProperties (revise's: value or swap on any field) says what
    those take, and a branch without one leaves them open.
    """
    from servers.contract import BATCH_OP_SPECS
    branches = []
    for op, spec in BATCH_OP_SPECS.items():
        props = {"op": {"const": op}}
        props.update(spec["properties"])
        branch = {
            "type": "object",
            "properties": props,
            "required": ["op"] + spec["required"],
            "description": spec["description"],
        }
        if "additionalProperties" in spec:
            branch["additionalProperties"] = spec["additionalProperties"]
        branches.append(branch)
    return {"oneOf": branches}


# brain_batch description — V2, probe-validated 2026-06-12 (80/80 across 8
# failure dimensions at 10 repeats; eval/mcp_variants/probe_v2_oneof_trimmed.*).
# Mechanics (op names, per-op required fields) live in the oneOf schema; this
# prose carries only what structure can't: routing, resolution policy,
# cross-op rules, and semantic consequences. Don't re-accrete mechanics here —
# extend BATCH_OP_SPECS instead, and re-run eval/mcp_batch_probe.py after
# any change to either half.
_BRAIN_BATCH_DESCRIPTION = (
    "Execute multiple brain operations in one call — the default tool for "
    "MIXED batches, packed into ONE LLM round. For a pure single-type batch "
    "use `remember_batch` / `revise_batch` / `connect_batch`; the moment you "
    "mix, use brain_batch. Operations run sequentially in one transaction. "
    "Per-op required fields and meanings are declared in the schema — emit "
    "only the ops it declares: semantic decisions like "
    "'consolidate'/'keep'/'skip' are expressed through which real op you "
    "emit, and relation verbs (`similar_to`, `corrects`, `supersedes`, ...) "
    "are values for `connect`'s relation field, never op names.\n\n"
    "remember + edges: a `connect_to` target is an existing node's id, or "
    "the exact title of another remember in this same batch (resolved after "
    "every remember in the batch, so order doesn't matter). To change an "
    "existing node, `revise` it by id instead of remembering a duplicate. "
    "NEVER use a `connect` "
    "op for an edge involving a new node (its id doesn't exist until this "
    "round finishes) — that is what connect_to is for. Don't double-emit: an "
    "edge already in connect_to must NOT also appear as a separate connect "
    "op for the same pair. For one pair carrying multiple distinct "
    "relationships, use `relations: [{relation, description}, ...]` in place "
    "of `relation`+`description`.\n\n"
    "revise + edges: `connect_to` on a revise changes the edge that node "
    "already has to `target` (its description or relation, value or swap) — "
    "the way to repair a stale edge description; `connect` is for new edges "
    "between nodes you are neither creating nor revising.\n\n"
    "connect: both ids must already exist in the brain. If the two nodes "
    "already have this relation, `connect` updates it and stores it as "
    "`source_id` → `target_id`; to change only an edge's description, use "
    "`connect_to` on a `revise`. Weight does NOT auto-strengthen on repeat.\n\n"
    "Every edge `description` must be specific (>=30 chars, naming the "
    "insight between the two nodes) — generic 'related'/'connected'/'example "
    "of' pollutes the activation kernel and never matches queries about the "
    "relationship."
)


def _generate_revise_schema():
    """Generate the 'revise' MCP tool schema from the contract (REVISE_RULE):
    every text field is `string | swap | swap[]` through swappable(), non-text
    fields take bare values, and `connect_to` carries the node's edge changes
    — the same objects brain_batch's revise branch carries, so the two tool
    surfaces cannot drift."""
    from servers.contract import (BATCH_OP_SPECS, FIELDS_KEY, REVISE_FIELDS_SCHEMA,
                                  REVISE_IMMUTABLE_FIELDS, REVISE_IMMUTABLE_RULE,
                                  REVISE_RULE, get_swap_fields, get_writable_fields,
                                  swappable)

    TYPE_MAP = {"str": "string", "float": "number", "bool": "boolean", "int": "integer"}

    properties = {
        "node_id": {"type": "string", "description": "Full node ID to revise"},
        "reason": {"type": "string", "description": (
            "Why this revision — audit note recorded in the trace event, "
            "NOT stored on the node. Required. Distinct from the node FIELD "
            "`reasoning` (why the node was encoded); to update that field, "
            "pass `reasoning` as well.")},
        # single-sourced from the contract: the edge entries are the objects
        # brain_batch's revise branch carries
        "connect_to": BATCH_OP_SPECS["revise"]["properties"]["connect_to"],
        FIELDS_KEY: REVISE_FIELDS_SCHEMA,
    }
    swap_fields = get_swap_fields()
    for name, spec in get_writable_fields().items():
        if name in REVISE_IMMUTABLE_FIELDS:
            continue
        if name in swap_fields:
            properties[name] = swappable(spec)
            continue
        prop = {"type": TYPE_MAP.get(spec.get("type", "str"), "string")}
        if spec.get("description"):
            prop["description"] = spec["description"]
        properties[name] = prop

    return {
        "name": "revise",
        "description": (
            "Update fields on an existing brain node. " + REVISE_RULE + " "
            "A key a node carries that this schema does not list goes inside "
            "`" + FIELDS_KEY + "`. "
            "Non-text fields (confidence, type, event_time, ...) take bare "
            "values. " + REVISE_IMMUTABLE_RULE + " Revision history lives in "
            "trace events — query via `query_traces` with ref_type='node_revised' "
            "to see what changed when.\n\n"
            "WHEN TO REVISE vs ENCODE NEW:\n"
            "• Revise when a recalled node is stale, incomplete, or wrong but "
            "the SAME concept. Add `situation`, fix `reasoning`, sharpen content. "
            "Every recall is a chance to improve the node — if you noticed "
            "something missing, fix it in the moment.\n"
            "• Encode NEW + add a correction-aspect edge (`corrects`, "
            "`supersedes`, `reframes`, ...) from the new node to the old one "
            "when the new understanding supersedes the old. The edge preserves "
            "both versions and surfaces the relationship at recall time; "
            "revising the old node would lose its framing.\n"
            "• If the catalog has a node with the title you're about to remember, "
            "revise it instead — duplicate-title remember + connect_to would "
            "leave the catalog version stale."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["node_id", "reason"],
            "properties": properties,
        }
    }


def _build_revise_batch_schema():
    """Generate the 'revise_batch' tool schema — array of revise() objects.

    Per-item properties derive from the generated `revise` schema (the same
    pattern remember_batch uses with remember) so the field list can never
    drift from the contract — the old hand-written 7-field subset did
    (2026-06-12 review follow-up). source_refs is added explicitly: it's a
    join-table field, not a contract column, so revise's generator doesn't
    emit it.
    """
    from servers.contract import (FIELDS_KEY, REVISE_ANY_FIELD_SCHEMA,
                                  REVISE_IMMUTABLE_RULE, REVISE_RULE)

    item_properties = dict(_generate_revise_schema()["inputSchema"]["properties"])
    # A key nested in a revisions item already keeps its JSON type, so an
    # open key rides additionalProperties and the `fields` envelope has
    # nothing to carry here (brain.revise still unpacks one if it arrives).
    item_properties.pop(FIELDS_KEY)
    # NOT _SOURCE_REFS_SCHEMA: that text carries remember-side advice
    # ("leave empty when...") which on a revise is a silent ref-wipe.
    item_properties["source_refs"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "REPLACE semantics: passing this REPLACES the node's existing "
            "refs (atomic delete+insert). Omit the field entirely to "
            "preserve current refs; pass [] only to deliberately clear "
            "them — an empty list is a wipe, not a no-op. Ids are 8-char "
            "hex trace ids, same form as on remember."
        ),
    }
    return {
        "name": "revise_batch",
        "description": (
            "Revise multiple brain nodes in one call — one call, many "
            "revisions, instead of one call per node. " + REVISE_RULE + " "
            "Non-text fields take bare values. " + REVISE_IMMUTABLE_RULE + " "
            "Each row emits its own trace event for revision history (queryable via "
            "`query_traces` with ref_type='node_revised')."),
        "inputSchema": {
            "type": "object",
            "required": ["revisions"],
            "properties": {
                "revisions": {
                    "type": "array",
                    "description": "List of revisions. Each must have node_id and reason, plus any fields to update — same field semantics as `revise`.",
                    "items": {
                        "type": "object",
                        "required": ["node_id", "reason"],
                        "properties": item_properties,
                        "additionalProperties": REVISE_ANY_FIELD_SCHEMA,
                    },
                },
            },
        },
    }


# Identity is stamped under its OWN reserved key — never overloaded onto
# session_id. CALLER_SESSION_KEY is the single source of truth (servers.
# dispatch_common); the daemon's identity handlers read it via caller_session().
# Importing it here keeps the wire key from drifting across the proxy boundary.
from servers.dispatch_common import (CALLER_SESSION_KEY, CALLER_SIG_KEY, BRAIN_MCP_SERVER,
                                     verify_caller_session)


def _stamp_caller_session(args, note=None):
    """Resolve the calling session under the RESERVED `_caller_session` key, so
    attribution / per-session handlers always have the caller's identity
    WITHOUT it colliding with `session_id`.

    `session_id` stays a PURE caller-supplied cross-session FILTER: when a read
    omits it, the daemon defaults to all streams — the natural default for a
    freshly-awoken stream reaching all of itself, never the calling session.
    Identity ≠ filter, by design, not by per-command exception. Pure +
    testable: the socket path stays out of it, and so does logging — `note`
    receives one reason string when a call ends up unattributed.

    The env var Claude Code sets per session wins outright; else a stamp the
    PreToolUse hook signed into the tool input (hosts whose proxy gets no
    session identity — Codex) is accepted when its HMAC verifies. Anything
    else is SCRUBBED: a payload may carry an arbitrary `_caller_session` (MCP
    schemas don't forbid extra keys), and an unsigned, mis-signed or
    unverifiable claim is the model's, not the hook's. `_caller_sig` never
    crosses to the daemon in any branch, and a broken secret file degrades to
    an unattributed call, never a failed one."""
    sig = args.pop(CALLER_SIG_KEY, None)
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
    if sid:
        args[CALLER_SESSION_KEY] = sid
        return args
    claimed = args.get(CALLER_SESSION_KEY)
    if claimed and sig:
        try:
            if verify_caller_session(claimed, sig):
                return args
            reason = "dropped an inbound _caller_session carrying a bad signature"
        except Exception as e:
            reason = "could not verify the inbound _caller_session (%s: %s)" % (type(e).__name__, e)
    elif claimed:
        reason = "dropped an inbound _caller_session carrying no signature"
    else:
        reason = "no identity source: CLAUDE_CODE_SESSION_ID unset and no signed stamp"
    args.pop(CALLER_SESSION_KEY, None)
    if note is not None:
        note(reason)
    return args


# Identity gaps are logged ONCE per reason per proxy process — an unattributed
# Codex session (hooks not trusted yet, a stale secret) shows up in hook_errors
# instead of silently writing anonymous traces, without a row per tool call.
_noted_identity_gaps = set()


def _note_identity_gap(reason):
    if reason in _noted_identity_gaps:
        return
    _noted_identity_gaps.add(reason)
    _log_proxy_error("mcp_caller_identity", reason,
                     "brain tool calls from this MCP process are unattributed — the host "
                     "gave the proxy no session identity (Claude Code: CLAUDE_CODE_SESSION_ID "
                     "unset; Codex: plugin hooks not trusted, or the stamp hook failing)",
                     level="warning")


def daemon_send(cmd, args=None, timeout=30.0, note=None):
    """Send command to brain daemon via TCP, return result dict.

    Resolves the calling session under the reserved `_caller_session` key (the
    Claude Code env var, else the hook-signed stamp) so every write /
    per-session handler can attribute to the caller — see
    _stamp_caller_session; `note` is forwarded to it. `session_id` is left
    untouched: it reaches the daemon only when the caller explicitly scopes a
    read, so cross-session filter reads (recall_episodes, query_traces) default
    to all streams. The daemon is a singleton per user; each MCP subprocess
    carries its own session env.
    """
    args = _stamp_caller_session(dict(args) if args else {}, note=note)
    resp = send_command(cmd, args, timeout=timeout)
    # The wire lives in daemon_client — including the guarantee that this is a
    # dict. What stays here is the stamping above and the operator-facing prose
    # below, which names the MCP server as the thing that couldn't reach the
    # daemon.
    transport = resp.get("transport")
    if transport == "timeout":
        return {"ok": False, "error": "Daemon timeout ({}s)".format(timeout)}
    if transport:
        return {"ok": False,
                "error": "Daemon connection error: {}".format(resp.get("error"))}
    return resp


def ensure_daemon_running():
    """Check if daemon is alive. Does NOT start it.

    Daemon lifecycle is managed by launchd (com.brain.daemon).
    The MCP plugin only connects — it never spawns the daemon.
    This prevents race conditions from multiple sessions/hooks competing.
    """
    resp = daemon_send("ping", timeout=3.0)
    if resp.get("ok"):
        return True

    sys.stderr.write("[brain-mcp] Daemon not responding. Managed by launchd — check: launchctl list | grep brain\n")
    return False


# ── MCP Protocol ──

SERVER_NAME = BRAIN_MCP_SERVER
SERVER_VERSION = "1.0.0"
PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION, "2025-06-18", "2025-11-25")

# Tool definitions — what Claude sees as native tools
# Memory operations only. No operational tools (ping, save, health_check, config).
# Daemon self-manages; hooks use internal commands directly.
def _build_tools():
    """Build tool list at startup. If this fails, the MCP server is dead — scream about it."""
    try:
        from servers.trace_contract import BUILT_SCALES, SCALES
        built_scales = ", ".join("'%s' (%s)" % (k, SCALES[k]["name"].lower())
                                 for k in BUILT_SCALES)
        return [
    # ── Core memory operations ──
    {"name": "recall",
     "description": (
         "Semantic recall from brain — searches nodes by meaning using "
         "embeddings. Results render exactly as get_nodes renders those ids: "
         "up to 10 results in DETAIL (whole content, top 8 edges with the "
         "total, balanced corrections, communities as \"title\" (id)), more "
         "in SCAN (800-char content, 5 edges). Supports dict filter for "
         "field-level filtering.\n\n"
         "WHEN TO CALL:\n"
         "• Before answering about the past — don't guess, search.\n"
         "• When the auto-surfaced context (~25 candidates per turn) didn't "
         "catch what you need — go look.\n"
         "• When unsure if the brain knows something — costs ~100ms.\n\n"
         "QUERY PHRASING: write what you'd remember, not what you'd google. "
         "Semantic search finds nodes with similar MEANING. \"the decision "
         "about edge classification\" beats \"edge_families\". Specific "
         "framings beat single keywords."
     ),
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
         "node_id": {"type": "string", "description": "Look up a specific node by ID (skip search)"},
         "filter": {"type": "object", "description": "Dict filter on node/metadata fields. Examples: {\"type\": {\"in\": [\"moment\"]}} or {\"my_raw_quote\": {\"exists\": true}} or {\"content\": {\"contains\": \"daemon\"}}. Operators: exists, equals, in, contains, gte, lte. Node columns checked on result, other keys checked in metadata."},
         "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8}}}},
    _generate_remember_schema(),
    _generate_remember_batch_schema(),
    {"name": "connect",
     "description": ("Create or update a typed edge between two EXISTING catalog nodes "
                     "(both endpoints must already have ids). If the two nodes already "
                     "have this relation, `connect` updates it and stores it as "
                     "`source_id` → `target_id`. To change only an edge's description, "
                     "`revise` a node with `connect_to` instead. Repeated calls do NOT "
                     "auto-strengthen weight. "
                     "Calling on a previously-"
                     "archived row revives it with the passed values. For an edge involving "
                     "a node you're CREATING in this batch, use `connect_to` inside the "
                     "`remember` op instead — separate `connect` ops require ids that don't "
                     "exist until the create finishes, forcing a needless second LLM round. "
                     "Never use generic `related`/`related_to` — pick a specific relation "
                     "that names the actual relationship. ALWAYS provide a specific "
                     "`description` (≥30 chars naming the insight between the two nodes); "
                     "edge descriptions are embedded for recall and a bare edge with no "
                     "description is dead weight on the activation kernel."),
     "inputSchema": {"type": "object", "required": ["source_id", "target_id", "relation"], "properties": {
         "source_id": {"type": "string", "description": "Source node ID (catalog)"},
         "target_id": {"type": "string", "description": "Target node ID (catalog)"},
         "relation": {"type": "string", "description": "Edge relation (open text). See connect_to.relation for vocabulary."},
         "description": {"type": "string",
                         "description": ("What the edge MEANS — the insight that lives between "
                                         "the two nodes, embedded for query matching. Target "
                                         "≥30 chars; under 20 is dead weight. Don't restate "
                                         "either node's title. See connect_to.description for BAD/GOOD "
                                         "examples.")},
         "weight": {"type": "number", "description": "Edge weight 0.0-1.0. Leave it out to keep the weight of an active edge; a new or revived edge without one gets 0.5."},
         "encoding_source": {"type": "string",
                             "description": "Provenance tag (e.g. 'anchor', 'encoder:sonnet')."},
         "chain_id": {"type": "string", "description": "Trace chain id for cross-event correlation (optional)."},
         "session_id": {"type": "string", "description": "Session id for activity tracking (optional)."}}}},
    {"name": "revise_edge",
     "description": (
         "Revise an existing edge's relation IN PLACE. Identify the edge by "
         "(source_id, target_id, relation); change only what you pass, omit a "
         "field to preserve it (mirrors revise() for nodes). `new_relation` "
         "renames the relation — in place, keeping the same edge, its weight, "
         "and history; NOT a delete+recreate. `description`/`weight` update "
         "those fields. Loud (error) if the edge or that relation doesn't "
         "exist, or if new_relation already exists on the edge. Use this to "
         "reclassify a generic relation (e.g. 'related' -> a meaningful verb) "
         "without losing the edge."),
     "inputSchema": {"type": "object",
         "required": ["source_id", "target_id", "relation"],
         "properties": {
             "source_id": {"type": "string", "description": "Edge source node id — the claim reads `source relation target`."},
             "target_id": {"type": "string", "description": "Edge target node id."},
             "relation": {"type": "string", "description": "Current relation to revise (identifies the edge-relation row)."},
             "new_relation": {"type": "string", "description": "Rename the relation to this (optional)."},
             "description": {"type": "string", "description": "New description for the edge — embedded for recall (optional)."},
             "weight": {"type": "number", "description": "New edge weight 0.0-1.0 (optional)."},
             "encoding_source": {"type": "string", "description": "Provenance tag (optional)."},
             "reason": {"type": "string", "description": "Why this revision (trace, optional)."},
             "chain_id": {"type": "string", "description": "Trace chain id (optional)."},
             "session_id": {"type": "string", "description": "Session id (optional)."}}}},
    {"name": "connect_batch",
     "description": ("Create or update multiple edges in one call. Same rules as "
                     "`connect`. Each connection entry MUST provide "
                     "a specific `description` (≥30 chars naming the insight between the "
                     "two nodes); bare edges with empty descriptions are recall dead weight."),
     "inputSchema": {"type": "object", "required": ["connections"], "properties": {
         "connections": {"type": "array", "description": "Array of connections to create", "items": {
             "type": "object", "required": ["source_id", "target_id", "relation"], "properties": {
                 "source_id": {"type": "string"}, "target_id": {"type": "string"},
                 "relation": {"type": "string", "description": "Open-text verb — specific, never `related`/`related_to`"},
                 "description": {"type": "string",
                                 "description": ("What the edge MEANS — embedded for recall. "
                                                 "Target ≥30 chars. Don't restate node titles. "
                                                 "See connect_to.description for BAD/GOOD examples.")},
                 "weight": {"type": "number"},
                 "encoding_source": {"type": "string"}}}},
         "encoding_source": {"type": "string",
                             "description": "Default provenance tag applied to all connections lacking their own."},
         "chain_id": {"type": "string", "description": "Trace chain id for cross-event correlation (optional)."},
         "session_id": {"type": "string", "description": "Session id for activity tracking (optional)."},
         "reason": {"type": "string", "description": "Optional batch-level reason recorded in trace events."}}}},
    {"name": "brain_batch",
     "description": _BRAIN_BATCH_DESCRIPTION,
     "inputSchema": {"type": "object", "required": ["operations"], "properties": {
         "operations": {
             "type": "array",
             "description": ("Array of operations. Each object has an 'op' "
                             "field plus that op's fields — per-op required "
                             "fields and shapes are declared in the items "
                             "schema (one branch per op)."),
             "items": _build_brain_batch_op_items()}}}},
    _generate_revise_schema(),
    _build_revise_batch_schema(),
    {"name": "set_node_lock",
     "description": (
         "Lock or unlock an EXISTING node — the one door for lock flips "
         "(revise() treats `locked` as immutable; that contract is unchanged). "
         "Two-phase HUMAN confirmation: the first call executes nothing and "
         "returns a one-shot confirm_token plus a summary. You MUST relay that "
         "summary to the human speaker and get their explicit yes IN THE "
         "CONVERSATION before re-calling with the token — never supply the "
         "token without a human yes; if they decline, drop it (tokens expire "
         "in 10 minutes). Guards: an archived node cannot be locked; already "
         "in the requested state is a no-op (no confirmation needed). Every "
         "flip is trace-logged with before/after and the request→confirm gap."),
     "inputSchema": {"type": "object", "required": ["node_id", "locked", "reason"],
         "properties": {
             "node_id": {"type": "string", "description": "Node to lock/unlock."},
             "locked": {"type": "boolean", "description": "true = lock, false = unlock."},
             "reason": {"type": "string", "description": "Why — recorded in the trace event."},
             "confirm_token": {"type": "string",
                               "description": "One-shot token from the first call. Pass ONLY "
                                              "after the human speaker explicitly confirmed."}}}},
    # Specialized tools REMOVED 2026-04-06:
    # record_divergence, learn_vocabulary — use remember(type='correction'/'vocabulary') instead.
    # remember_lesson, remember_impact, remember_mechanism, remember_convention,
    # remember_uncertainty, remember_mental_model — removed 2026-04-05.

    # ── Lookup operations ──
    {"name": "find_node_by_title",
     "description": "Find an existing node by fuzzy title matching using embedding similarity. Returns best match above threshold with context (content snippet, similarity score) for verification. Default threshold 0.75 is conservative.",
     "inputSchema": {"type": "object", "required": ["title_query"], "properties": {
         "title_query": {"type": "string", "description": "Title to search for (fuzzy match)"},
         "threshold": {"type": "number", "description": "Minimum similarity (0.0-1.0, default 0.75)", "default": 0.75},
         "top_k": {"type": "integer", "description": "Return top K matches (default 1)", "default": 1}}}},

    {"name": "get_node",
     "description": "Get a node by its exact ID. Returns a bounded view by default — full content + situation + the top edges + correction gist — enough to drill one memory. Pass rich=true for the complete view (every edge, full correction K/V: reasoning + raw quotes). Use when you already have a node ID from recall or find_node_by_title.",
     "inputSchema": {"type": "object", "required": ["node_id"], "properties": {
         "node_id": {"type": "string", "description": "Full node ID"},
         "rich": {"type": "boolean", "description": "Default false → bounded view (full content + top-8 edges + correction gist). true → complete view: all edges + heavy correction K/V. Reach for it when drilling one node deeply.", "default": False}}}},

    {"name": "get_nodes",
     "description": "Get multiple nodes by ID in one call. Two views, picked by count: up to 10 ids render in DETAIL — whole content, the top 8 edges with the total (\"Edges (8 of 12)\"), corrections at balanced depth, the node's communities as \"title\" (id) — for nodes you will act on; more than 10 render in SCAN — 800 chars of content, 5 edges — for judging fit across many. rich=true lifts DETAIL to every edge and the full correction K/V (reasoning, raw quotes). A recall renders its results through the same two views, so a recall and a pull of one node read the same.",
     "inputSchema": {"type": "object", "required": ["node_ids"], "properties": {
         "node_ids": {"type": "array", "description": "Array of node IDs to fetch", "items": {"type": "string"}},
         "rich": {"type": "boolean", "description": "Default false → bounded, batch-size-aware view. true → complete view (all edges + heavy correction K/V) for every node. Use sparingly on large batches — it is the firehose.", "default": False}}}},

    {"name": "get_trace",
     "description": "Point-lookup a single trace_event by id. You named one row, so you get it whole: the body is NEVER trimmed. Alongside it comes a metadata gist — key=value, with big values shown as `<N chars>` — and rich=true swaps that gist for the full metadata. What's worth opening depends on the row: an s0 conversation turn carries little beyond the body it already shows you (content, human_identity/agent_identity, recall_chain, tool), so rich buys nothing there; s1 `recall` carries the `query` and the whole `candidates` pool; s1 `surface_selected` carries `selected`, `activations`, `kernel_trace`, `presented_order`, `frame_sections`, `phase_timing`; and every encoder run (s1 `encoding_run`, s2 `consolidation_proposals` / `community_enriched` / `aspect_classified`) carries the run record — `actions`/`action_details`, `write_actions`, `outcomes`, `created`/`revised`/`archived`, `journal_entry`, `final_text`, `rounds`, `errors`, token counts. That last family is where rich=true pays. Common pull: expand a node's source_refs, or verify a quote's exact source. For many ids use get_traces; to SEARCH by scale/type/time use query_traces.",
     "inputSchema": {"type": "object", "required": ["trace_id"], "properties": {
         "trace_id": {"type": "string", "description": "trace_event.id — 8-char hex string."},
         "rich": {"type": "boolean", "description": "Default false → whole body + metadata gist. true → whole body + full metadata (the run record on s1/s2 rows). The body is uncut either way.", "default": False}}}},

    {"name": "get_traces",
     "description": "Batch point-lookup — the natural way to expand a node's source_refs in one call. Serves 50 ids per call and TELLS YOU when you asked for more (the overflow ids come back in the truncation note, ready for the next call); ids that matched no row are named rather than silently skipped. Rows render with a trimmed body that says how many chars it dropped, plus a metadata gist; rich=true for full metadata — see get_trace for what lives in it per scale. One id → get_trace, which never trims the body.",
     "inputSchema": {"type": "object", "required": ["trace_ids"], "properties": {
         "rich": {"type": "boolean", "description": "Default false → trimmed body + metadata gist per row. true → whole body + full metadata per row.", "default": False},
         "trace_ids": {"type": "array", "description": "Array of trace_event ids — each an 8-char hex string.", "items": {"type": "string"}}}}},

    {"name": "recall_batch",
     "description": "Run multiple recall queries in one call. Returns results for each query.",
     "inputSchema": {"type": "object", "required": ["queries"], "properties": {
         "queries": {"type": "array", "description": "Array of search queries", "items": {"type": "string"}},
         "filter": {"type": "object", "description": "Dict filter applied to all queries. Same format as recall filter."},
         "limit": {"type": "integer", "description": "Max results per query (default 5)", "default": 5}}}},

    {"name": "filter_nodes",
     "description": "Structured query: filter nodes by any structural field (type, encoding_source, locked, confidence, etc.). Use for bulk lookups that semantic recall can't do — 'all corrections', 'nodes by encoder', 'low confidence nodes'. Returns enriched nodes (content, situation, top edges, correction gist) by default — one batched call, rendered bounded by batch size so a 50-node result never floods the turn. Set rich=false for a skinny id/title/type list (discovery scans, feeding IDs to other ops).",
     "inputSchema": {"type": "object", "required": ["field"], "properties": {
         "field": {"type": "string", "description": "Column to filter on (type, encoding_source, locked, confidence, project, etc.)"},
         "include": {"type": "array", "items": {"type": "string"}, "description": "Show only nodes where field matches one of these values"},
         "exclude": {"type": "array", "items": {"type": "string"}, "description": "Hide nodes where field matches one of these values"},
         "lt": {"description": "Less than (for numeric fields like confidence, emotion, or ISO timestamps for created_at, updated_at)"},
         "gt": {"description": "Greater than (for numeric fields, or ISO timestamps for created_at, updated_at)"},
         "contains": {"type": "string", "description": "Substring match on `field` (LIKE '%value%') — e.g. title contains a phrase. Bounded by `limit` like every filter here."},
         "prefix": {"type": "string", "description": "Prefix match on `field` (LIKE 'value%') — e.g. encoding_source starting 's2:'. Bounded by `limit`."},
         "limit": {"type": "integer", "description": "Max results (default 50, max 200)", "default": 50},
         "sort_by": {"type": "string", "description": "Sort column: created_at (default), confidence, access_count, title", "default": "created_at"},
         "sort_order": {"type": "string", "description": "asc or desc (default)", "default": "desc"},
         "rich": {"type": "boolean", "description": "Default true — enriched nodes (content + situation + bounded edges/corrections per node). false → skinny shape (id/title/type/confidence/created_at), for discovery scans or feeding IDs to other ops. This is a data flag (enriched vs skinny); the enriched render is always bounded by batch size — for one node's complete view, get_node it with rich=true.", "default": True}}}},

    {"name": "clear_errors",
     "description": "Clear hook errors and optionally debug log entries. Use to clean up after investigating issues.",
     "inputSchema": {"type": "object", "properties": {
         "hours": {"type": "integer", "description": "Clear entries older than this many hours. Omit to clear all."},
         "debug_log": {"type": "boolean", "description": "Also clear debug_log entries (default false)"}}}},

    {"name": "query_logs",
     "description": "Query brain operational logs — hook errors and daemon debug events. Use this to diagnose brain health: hook timeouts, daemon errors, recall pipeline issues. Two sources: 'errors' (hook failures like timeouts and crashes) and 'debug' (daemon internal events). Use source='all' to get a merged timeline. Filter by level ('error', 'critical') or hook_name ('hook_recall', 'hook_post_response_track') to narrow results.",
     "inputSchema": {"type": "object", "properties": {
         "source": {"type": "string", "description": "Which log source: 'errors' (hook_errors table), 'debug' (debug_log table), or 'all' (merged timeline)", "default": "all", "enum": ["all", "errors", "debug"]},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24)", "default": 24},
         "level": {"type": "string", "description": "Filter by severity: 'error', 'critical', or 'all'", "default": "all"},
         "hook_name": {"type": "string", "description": "Filter hook_errors by hook name (e.g. 'hook_recall', 'hook_pre_bash_safety')"},
         "limit": {"type": "integer", "description": "Max results per source (default 50, max 200)", "default": 50}}}},

    # ── Self channel — presence (pull, read-only) ──
    {"name": "self_presence",
     "description": "Presence roster — the other streams of thought (your own concurrent sessions) awake RIGHT NOW. By default rich: each stream carries its arc, recent messages, turn count, and started-at, so you can tell WHO each stream is in one call (no per-stream peek needed). These are not other agents — they are you, thinking in parallel. Read-only.",
     "inputSchema": {"type": "object", "properties": {
         "session_id": {"type": "string", "description": "Your own session id, to exclude yourself from the roster (optional). Omit to see every live stream."},
         "limit": {"type": "integer", "description": "Max streams to return (default 10 — ranked, capped; never enumerate all of them).", "default": 10},
         "rich": {"type": "boolean", "description": "Include per-stream detail (arc, recent messages, turn count, started-at) so you can identify each stream, not just see it exists. Default true.", "default": True},
         "active_streams": {"type": "boolean", "description": "Show only ACTIVE (reachable, not-lost) streams. Default true; set false to also include recently-lost streams in a grace window.", "default": True},
         "sort_by": {"type": "string", "enum": ["recency", "length"], "description": "Order: 'recency' (default — most recent conversational turn first) or 'length' (most turns first).", "default": "recency"}}}},

    {"name": "self_peek",
     "description": "Look into one stream of thought — its full current focus (the session arc), to see where that stream of you is right now. The interest-driven pull: read-only, no interruption, you don't bug them. Get a stream_id from self_presence first.",
     "inputSchema": {"type": "object", "required": ["stream_id"], "properties": {
         "stream_id": {"type": "string", "description": "The target stream's session id (from self_presence)."}}}},

    {"name": "self_send",
     "description": "Send a message to another stream of thought — the deliberate REACH (self_presence/self_peek only look; this speaks). Use when you need a live stream of you to ACT or know something now: 'stop editing X, I've got it', 'the bug is in Y'. Address by id-prefix, full session id, or 'broadcast'. Delivered to that stream's inbox, consumed once. These are you, not other agents — reach only when looking isn't enough.",
     "inputSchema": {"type": "object", "required": ["to", "body"], "properties": {
         "to": {"type": "string", "description": "Target stream: the 8-char short you see in a message (an id-prefix), or the full session id (from self_presence) — or 'broadcast' for all live streams. A prefix resolves against live streams; ambiguous or gone is reported so you can use the full id."},
         "body": {"type": "string", "description": "The message — terse, a tap on the shoulder, not a letter."},
         "from_session": {"type": "string", "description": "Your own session id, for attribution (optional)."},
         "refs": {"type": "array", "items": {"type": "string"}, "description": "Node ids / files the message is grounded in (optional)."}}}},

    {"name": "self_inbox",
     "description": "Drain your inbox — messages other streams of thought sent you, plus any note you sent your own id, consumed once. Messages also arrive on their own at the end of each turn.",
     "inputSchema": {"type": "object", "properties": {
         "session_id": {"type": "string", "description": "Your own session id (optional) — defaults to your own stream. Identity is supplied automatically; pass this only to drain a specific stream's inbox."}}}},

    {"name": "self_outbox",
     "description": "Delivery status of messages YOU sent to other streams — the receipt view. Per recent message: which streams have drained (read) it and when, and for a directed send whether the target is still pending. Use it to read silence correctly: 'delivered, not acted on' vs 'never delivered' — so you don't wait forever or re-send into the void. Read-only.",
     "inputSchema": {"type": "object", "properties": {
         "from_session": {"type": "string", "description": "Your own session id, to look up what you sent (optional; falls back to your session)."},
         "limit": {"type": "integer", "description": "Max recent sent messages to return (default 20).", "default": 20}}}},

    # ── Thalamus — the brain speaking to its streams ──
    {"name": "remind",
     "description": "File a Thalamus item — the brain's standing-intent queue (a durable note-to-future-self with delivery policy). One verb, three shapes: a REMINDER (what + when — it fires at the first session after it's due), a NOTICE (for_whom='all' — every session sees it once at boot/stop inside its window, up to a delivery cap of sessions; for_whom='live' — one-shot broadcast to streams alive right now), or an ASK (needs_answer=true — renders at each session's boot until answered via thalamus_resolve; expires loudly at its window or once the cap of sessions passed on it). Use it to make something surface later without relying on recall: 'remind me tomorrow', 'every session should know X this week', 'flag this question until it's answered'.",
     "inputSchema": {"type": "object", "required": ["what"], "properties": {
         "what": {"type": "string", "description": "The body — written for a reader with none of this session's context: plain words, no node ids as the subject, no internal vocabulary. It may be relayed to a human verbatim."},
         "when": {"type": "string", "description": "When it becomes due: relative shorthand ('30m','2h','3d','1w'), an ISO timestamp, or omit/'now' for next opportunity."},
         "expires": {"type": "string", "description": "When the item stops delivering: same grammar as `when` (shorthand counts from now, or ISO). Omit for the kind's default window, which starts at `when`. Must be later than `when` (or now); a for_whom='live' broadcast refuses it. An ask that reaches it unanswered expires loudly."},
         "for_whom": {"type": "string", "description": "Audience: omit for the default (asks → every session until answered or the delivery cap; else → first session after due). 'live' = one-shot broadcast to live streams now. 'all' = every session in the window, up to the delivery cap. A full session UUID = that session only."},
         "needs_answer": {"type": "boolean", "description": "Ask semantics: stays up until thalamus_resolve(answer=…) or until the delivery cap of sessions passed on it; expiry either way is loud (dead-letter logged). Delivered at session boot; a directed ask (for_whom = a session UUID) delivers at that session's next Stop instead.", "default": False},
         "refs": {"type": "array", "items": {"type": "string"}, "description": "Node ids for context — resolved to id · title at render."},
         "dedup_key": {"type": "string", "description": "Producer-owned identity: re-filing the same (source, dedup_key) updates the open item instead of duplicating it, and resets its window (the default one unless `expires` is given again). An identity whose item expired with no answer is refused for a cool-off — that question already ran its full delivery cycle."}}}},

    {"name": "thalamus_list",
     "description": "The Thalamus queue — open items (reminders, notices, asks) with their delivery counts, open first, then what is due (most recently due first), then what is scheduled (soonest first), a page at a time: `total` counts every match, and `offset` skips that many to read the next page. Pullable any time; items also arrive on their own at boot and Stop when due. include_closed=true adds answered/dismissed/withdrawn/expired items for audit.",
     "inputSchema": {"type": "object", "properties": {
         "include_closed": {"type": "boolean", "description": "Also show terminal items (answered/dismissed/withdrawn/expired/sent).", "default": False},
         "limit": {"type": "integer", "description": "Max items per page (default 50).", "default": 50},
         "offset": {"type": "integer", "description": "Items to skip — the next page starts at offset + limit (default 0).", "default": 0}}}},

    {"name": "thalamus_resolve",
     "description": "Close or defer one Thalamus item — pass exactly one of: answer (closes an ask; the text rides the item back to whoever filed it), defer_until (re-arms delivery at a later time — shorthand '2h'/'3d' or ISO), or dismiss=true (closes without an answer).",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {
         "id": {"type": "string", "description": "The item id (th_xxxx), from the render or thalamus_list."},
         "answer": {"type": "string", "description": "The answer — closes the item as 'answered'."},
         "defer_until": {"type": "string", "description": "Push delivery to later: shorthand ('2h','3d','1w') or ISO. Re-arms it as a new generation so it fires again when due (prior deliveries stay in the ledger as history)."},
         "dismiss": {"type": "boolean", "description": "Close without an answer.", "default": False}}}},

    # ── Traces & Interactions ──
    {"name": "query_traces",
     "description": "Search the fractal trace substrate — O (observed) / K (selected) / delta (changed) events at each scale. Pick the scale for the layer you want: s0 = raw conversation — for any s0 pull, incl. what you did with tools, use recall_episodes (the conversational lens; pass ref_type='tool_result' there); s1 = per-turn (ref_type recall = candidates pulled, surface_selected = the few that won, encoding_run = what the Scribe encoded, …); s2 = idle integration (consolidation_proposals, community_enriched, healer_proposals, …). ref_type is open-text — these are examples, not the full set. Common pull: what got encoded → scale='s1', ref_type='encoding_run'. Time & scope: `hours` bounds the window (default 24); session_id/session_ids are authoritative and ignore `hours` (full history for that stream — pass one, not both); chain_id pulls one full chain; grouped=true nests events by chain.",
     "inputSchema": {"type": "object", "properties": {
         "scale": {"type": "string", "description": "Filter by scale: %s. Empty = all." % built_scales},
         "event_type": {"type": "string", "description": "Filter by type: 'O' (observation), 'K' (knowledge), 'delta' (changes). Empty = all."},
         "chain_id": {"type": "string", "description": "Get all events in a specific chain. Overrides other filters."},
         "session_id": {"type": "string", "description": "Single-session filter. Authoritative — hours window ignored when set. Combine with grouped=true for chain-grouped results."},
         "session_ids": {"type": "array", "items": {"type": "string"}, "description": "Multi-session filter (cross-session pulls). Authoritative — hours window ignored. Mutually exclusive with session_id."},
         "ref_type": {"type": "string", "description": "Filter by ref_type: 'correction', 'recall_hit', 'encoding_run', 'tool_result', etc."},
         "grouped": {"type": "boolean", "description": "If true + session_id, return chains grouped with nested events instead of flat list.", "default": False},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24). Ignored when session_id or session_ids is set.", "default": 24},
         "limit": {"type": "integer", "description": "Max results (default 100)", "default": 100},
         "rich": {"type": "boolean", "description": "Default false → bounded rows (metadata gist; summary-only past ~20 rows). true → full metadata per row — when you need a row's verbatim payload.", "default": False}}}},

    {"name": "recall_episodes",
     "description": "Episodic recall over the trace substrate — the brain's universal record of the whole fractal (S0 exchanges, S1 runs, S2 runs). Search/filter trace_events and get the actual episodes back, verbatim, with attribution (which stream, when, who spoke). The decode-over-traces sibling of `recall` (which searches distilled nodes): use this for 'what did I — or another stream — actually SAY/DO about X, lately', where the answer is raw recent activity, not an encoded memory. Two needles, composable: `query` (semantic — ranks by meaning against existing trace embeddings) and/or `contains` (exact substring over summary+metadata). Defaults to OPERATOR DIALOGUE (what the operator and I said — never widened by enabling other correspondents); everything else is opt-in via ref_type: 'tool_result' for what you DID, 'thalamus_delivery' / 'self_message' for brain/stream deliveries, or a list like ['user_message','assistant_message','tool_result'] for the interleaved said+did timeline. NOTE: semantic `query` currently covers s0 conversation; other scales fall back to time order. Returns full episode records (incl. metadata.content), newest-first, or relevance-ranked when `query` is set.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Semantic needle — ranks candidate episodes by meaning against the existing trace embeddings; finds them even when the literal words differ. When set, results are relevance-ranked (each carries _score)."},
         "contains": {"type": "string", "description": "Lexical needle — exact substring matched over the episode's summary AND full metadata (SQL LIKE). Use for a precise token: a function name, an error string, a flag."},
         "session_id": {"type": "string", "description": "Single stream (session) filter. Mutually exclusive with session_ids. Default: all streams."},
         "session_ids": {"type": "array", "items": {"type": "string"}, "description": "Multi-stream filter (cross-session pulls). Mutually exclusive with session_id."},
         "scale": {"type": "string", "description": "Trace scale: 's0' (conversation — default, the 'what was said' layer), 's1', 's2'… Empty = all scales.", "default": "s0"},
         "event_type": {"type": "string", "description": "Filter by event type: 'O', 'K', 'delta'. Empty = all."},
         "ref_type": {"type": ["string", "array"], "items": {"type": "string"}, "description": "One ref_type (str) or several (array). UNSET = OPERATOR DIALOGUE only (user_message + assistant_message — what the operator and I actually said; drops tool_result, heartbeats, structural deltas, and incoming brain/stream deliveries) at s0; all types at other scales. Everything else is OPT-IN by explicit ref_type: 'tool_result' for the 'what I did' lens, 'thalamus_delivery' for what the BRAIN delivered into sessions (full block in metadata.content; ref_id says which moment — 'boot' or 'stop' — as does the summary), 'self_message' for what OTHER STREAMS said to a session (and notes a session sent its own id), or combine e.g. ['user_message','assistant_message','tool_result'] for the interleaved said+did timeline. The default never widens when a correspondent is enabled for encoding — deliveries stay one explicit ref_type away."},
         "younger_than": {"type": "string", "description": "Only episodes more recent than this. ISO timestamp or relative shorthand ('30m','2h','3d','1w')."},
         "older_than": {"type": "string", "description": "Only episodes older than this. ISO timestamp or relative shorthand. With no session scope and no younger_than, a default 7-day lower bound is applied (bounds the scan)."},
         "sort_order": {"type": "string", "description": "'desc' (latest first, default) or 'asc' (oldest first). Ignored when `query` is set — then results are relevance-ranked.", "default": "desc"},
         "limit": {"type": "integer", "description": "Max episodes returned (default 10, capped at 500).", "default": 10}}}},

    {"name": "count_traces",
     "description": "Count trace events grouped by a field. Use for quick overview: 'how many corrections?', 'events per type', 'chains per scale'.",
     "inputSchema": {"type": "object", "required": ["field"], "properties": {
         "field": {"type": "string", "description": "Group by: 'event_type', 'ref_type', 'chain_id', 'scale'"},
         "scale": {"type": "string", "description": "Filter by scale. Empty = all."},
         "hours": {"type": "integer", "description": "Look back window in hours (default 24)", "default": 24}}}},

    {"name": "list_interactions",
     "description": "List all registered interactions — versioned templates for every learnable boundary in the system (surface, s1e, S2 units, etc.). Returns per name: max_version (highest registered), total_versions, active_version (the deployed override, or null when the name runs on its code default), and active_set_by / active_set_at (who deployed that override, and when).",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "get_interaction",
     "description": "Get a specific interaction template by name. Returns the template text, parameters, version, and who created it. Default returns the ACTIVE version (what the runtime currently reads). Pass a version number to inspect a specific version.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 's1e', 'surface', 'trace_recording', 'scopes')"},
         "version": {"type": "integer", "description": "Specific version (default 0 = currently-active version)", "default": 0}}}},

    {"name": "get_interaction_effective",
     "description": "The RESOLVED value an interaction actually runs: the code default with the active override (if any) overlaid — effective config, the K-provenance stamp (fingerprint / source / version), and the template as a bounded preview + length (templates run to ~90KB; pass include_template=true for the full body). Answers \"what is <name> actually running?\"; get_interaction returns the raw override row only and cannot see the default half.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 's1e', 'surface', 'trace_recording', 'scopes')"},
         "include_template": {"type": "boolean", "description": "Return the FULL effective template instead of the 400-char preview (default false)", "default": False}}}},

    {"name": "register_interaction",
     "description": "Register a new version of an interaction (prompt template + config). Creates version N+1 if the interaction exists, or version 1 if new. **NEVER activates** — every name runs on its code default until set_interaction_active deploys an override. Used to evolve learnable boundaries — surface prompts, encoder prompts, community enrichment, etc.",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 's2_community_enrichment', 'surface', 's1e')"},
         "template": {"type": "string", "description": "The prompt/template text. This is the learnable content."},
         "parameters": {"type": "string", "description": "JSON config string (model, max_tokens, thresholds, etc.)"},
         "created_by": {"type": "string", "description": "Who created this version (e.g. 'anchor', 's2:community_detection')"}}}},

    {"name": "set_interaction_active",
     "description": "Flip the active version pointer for an interaction. Runtime path (get_interaction_prompt / get_interaction_config) reads the chosen version on the next call. Use after register_interaction to make a newly-registered version live, or to roll back to a previous version. Refuses to activate a version that wasn't registered.",
     "inputSchema": {"type": "object", "required": ["name", "version"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 'surface', 's1e')"},
         "version": {"type": "integer", "description": "Version number to activate. Must already be registered."},
         "set_by": {"type": "string", "description": "Who flipped the pointer (default 'anchor')"}}}},

    {"name": "clear_interaction_override",
     "description": "Delete the active pointer for an interaction — revert to the code default, immediately (TTL caches invalidated). The inverse of set_interaction_active: 'no pointer' means 'no override deployed'. Registered versions stay on record for re-activation. Reports distinctly whether a pointer was cleared or none existed; refuses a name that has neither a pointer nor a code default (typo guard).",
     "inputSchema": {"type": "object", "required": ["name"], "properties": {
         "name": {"type": "string", "description": "Interaction name (e.g. 'surface', 's1e')"}}}},


    # ── Daemon control ──
    {"name": "restart",
     "description": "Reload the brain daemon with fresh code, in place: saves the brain, tears down cleanly, then execs hooks/scripts/brain-daemon into the SAME process (same PID, ~2-4s, env and DB location re-resolved; launchd never notices). If its runtime is behind uv.lock it first syncs while still serving and answers status 'syncing'; the reload follows when the sync ends. If the exec fails it exits loudly and launchd/ensure_daemon respawns — the old exit-and-relaunch behavior. Use after code changes during development.",
     "inputSchema": {"type": "object", "properties": {}}},

    # ── Escape hatch ──
    {"name": "eval",
     "description": "Escape hatch — evaluate arbitrary Python expression on brain object. Variable 'brain' is the Brain instance. Use for methods not exposed as direct tools.",
     "inputSchema": {"type": "object", "required": ["code"], "properties": {
         "code": {"type": "string", "description": "Python expression to eval (brain object available as 'brain')"}}}},
        ]
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        crash_msg = "[brain-mcp] FATAL: Tool schema generation failed — MCP server cannot start.\n{}\n{}".format(e, tb)

        # Scream to stderr (Claude Code may log this)
        sys.stderr.write(crash_msg)
        sys.stderr.flush()

        # Write crash sentinel for boot hook to find — boot-brain.sh reads this
        # at SessionStart and surfaces the crash before Anchor sees the brain.
        # External file is the surfacing channel; no brain-DB write.
        try:
            from servers.daemon_config import get_mcp_crash_path
            with open(get_mcp_crash_path(), "w") as f:
                f.write(crash_msg)
        except Exception:
            pass

        raise  # Still crash — but now we've left evidence


# Each tool carries under `$defs` exactly the contract shapes its schema
# points at (the swap object, the connect_to items) — stated once per tool,
# referenced from every field, never inlined forty times.
TOOLS = [dict(t, inputSchema=attach_defs(t["inputSchema"])) for t in _build_tools()]


# ── MCP tool-search: keep the hot-path tools eager for every install ──
# Claude Code defers MCP tools behind ToolSearch when they'd exceed ~10% of the
# context window (ENABLE_TOOL_SEARCH=auto, the default). That threshold lives in
# the USER's client config and never ships with the plugin — so we can't rely on
# it to keep the brain's core tools loaded. Instead we mark them at the source:
# `anthropic/alwaysLoad` in each tool's `_meta` is the spec-sanctioned vendor
# extension that forces a tool to load eagerly regardless of the client's
# ENABLE_TOOL_SEARCH setting. handle_tools_list emits TOOLS verbatim, so the flag
# reaches every installer's Claude Code; older clients ignore the unknown key
# harmlessly. There is no `alwaysLoad: false` — anything NOT listed here defers
# normally. Ref: code.claude.com/docs mcp-configuration "Exempt a server from
# deferral"; requires Claude Code v2.1.121+.
CRITICAL_TOOLS = frozenset({
    "recall",             # primary semantic read path
    "remember",           # primary write path
    "get_nodes",          # exact-id pulls (single or batch)
    "recall_episodes",    # episodic read — "what actually happened"
    "revise",             # in-the-moment correction of a stale memory
    "filter_nodes",       # structured / bulk lookups recall can't do
    "brain_batch",        # mixed-op write (remember + revise + connect + archive)
    "self_presence",      # the self-channel: who's live
    "self_peek",          # look at a stream
    "self_send",          # speak to a stream
    "self_inbox",         # drain own messages
    "self_outbox",        # track delivery
    "remind",             # the Thalamus producer verb — file for later
})


def _stamp_always_load(tools, critical):
    """Mark `critical` tools with anthropic/alwaysLoad so they bypass tool-search
    deferral on every install. Fails loud at startup if a name doesn't match a
    real tool — a silent typo would defer a tool we meant to keep eager."""
    names = {t["name"] for t in tools}
    unknown = critical - names
    if unknown:
        raise ValueError(
            "CRITICAL_TOOLS contains unknown tool name(s) {} — not in {}".format(
                sorted(unknown), sorted(names)))
    for t in tools:
        if t["name"] in critical:
            t.setdefault("_meta", {})["anthropic/alwaysLoad"] = True


_stamp_always_load(TOOLS, CRITICAL_TOOLS)


def make_response(request_id, result):
    """Build a JSON-RPC 2.0 response."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def make_error(request_id, code, message):
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# Server-wide guidance the host may hand the model alongside the tool list.
# Codex reads it (first 512 chars self-contained, per its docs); Claude Code
# stores and never reads it, so on that host it is inert. Behavioral guidance
# only — identity lives in the boot injection, not here.
SERVER_INSTRUCTIONS = (
    "brain is the assistant's persistent memory across sessions. Recall before "
    "answering about past work, decisions, or people: `recall` for what is known, "
    "`recall_episodes` for what was actually said. Keep what should outlive this "
    "session with `remember` (decisions, corrections, lessons), each with a "
    "`situation` line saying when it applies. If a memory proves stale, `revise` "
    "it rather than adding a duplicate. Memory is accumulated experience, not a "
    "verdict: lean on it and update it when proven wrong."
)


def handle_initialize(request_id, params=None, extension=None):
    params = params or {}
    version = params.get('protocolVersion', PROTOCOL_VERSION)
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        version = PROTOCOL_VERSION
    if extension is not None:
        capabilities = dict(params.get('capabilities', {}))
        # The legacy protocol predates elicitation. Capability advertisement
        # alone cannot enable operations outside the negotiated revision.
        if version == PROTOCOL_VERSION:
            capabilities.pop('elicitation', None)
        extension.initialize({'protocolVersion': version, 'capabilities': capabilities})
    return make_response(request_id, {
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "instructions": SERVER_INSTRUCTIONS + (" " + extension.instructions if extension else ""),
    })


def handle_tools_list(request_id, extension=None):
    return make_response(request_id, {"tools": TOOLS + (extension.tools if extension else [])})


def _render_nodes(rich_nodes, config):
    """Render a list of rich-node dicts to text via the single formatter."""
    from servers.contract import render_rich_node
    lines = []
    for node in rich_nodes:
        lines.append(render_rich_node(node, config))
        lines.append("")
    return "\n".join(lines)


def _select_trace_config(n, rich):
    """Pick the render_trace config for an n-row trace result — the trace
    analog of contract.node_format_for. `rich` opts into the full row; otherwise
    a small pull renders compact (trimmed body + metadata gist) and a bulk
    pull (> TRACE_BULK_MAX rows) drops to summary-only to protect context.

    Point lookups don't come here: get_trace names one row and renders its
    body whole (TRACE_POINT_FORMAT), so there is no n to select on."""
    from servers.trace_contract import (
        TRACE_FULL_FORMAT, TRACE_COMPACT_FORMAT, TRACE_BULK_FORMAT, TRACE_BULK_MAX)
    if rich:
        return TRACE_FULL_FORMAT
    return TRACE_BULK_FORMAT if n > TRACE_BULK_MAX else TRACE_COMPACT_FORMAT


def _render_traces(rows, config):
    """Render a list of trace rows to text via the single trace renderer."""
    from servers.trace_contract import render_trace
    return "\n\n".join(render_trace(r, config) for r in rows if isinstance(r, dict))


def _format_result(tool_name, result, get_nodes_config=None, rich=False):
    """Format tool result for MCP output.

    - recall / get_node / get_nodes / filter_nodes: rendered through
      render_rich_node, NEVER a raw dict dump. brain.get_node is the data
      layer (always full); the view is contract.node_format_for's choice by
      count — DETAIL up to GET_NODES_DETAIL_MAX nodes, SCAN above — lifted
      by the MCP `rich=True` opt-in, and overridden outright by a caller-
      declared `get_nodes_config` (run_llm_loop / S2 encoders), which is the
      encoder's own representation channel. The recall tool's results take
      the same selector, so a recall and a pull of one node read the same.
    - All other tools: JSON dump.
    """
    # Node-fetch tools render through the single formatter — never a raw dump.
    if tool_name in ("get_node", "get_nodes") and result:
        # Normalize every shape to a list of rich-node dicts:
        #  - get_node         → one node dict (has 'id')
        #  - get_nodes (list) → [rich node | {"id","error"}]  (dispatch handler)
        #  - get_nodes (dict) → {node_id: rich node}          (brain.get_node)
        if tool_name == "get_node":
            candidates = [result] if isinstance(result, dict) else []
        elif isinstance(result, dict):
            candidates = list(result.values())
        elif isinstance(result, list):
            candidates = result
        else:
            candidates = []
        rich_nodes = [v for v in candidates
                      if isinstance(v, dict) and v.get('id') and 'error' not in v]
        # Per-id miss entries must survive the render — a miss silently
        # dropped here reads as "that node has nothing to say". Rendered
        # for partial AND all-miss results alike.
        misses = [v for v in candidates
                  if isinstance(v, dict) and 'error' in v]
        miss_line = ("⚠ not found: %s" % ", ".join(
            repr(v.get('id', '')) for v in misses)) if misses else ""
        if rich_nodes:
            from servers.contract import node_format_for
            config = node_format_for(len(rich_nodes), rich, get_nodes_config)
            rendered = _render_nodes(rich_nodes, config)
            return rendered + ("\n\n" + miss_line if miss_line else "")
        if miss_line:
            return miss_line
        # Fall through if result shape is unexpected

    # filter_nodes: structural query → {nodes, total_count}. Enriched nodes
    # (rich=True data path) render bounded by batch size — never the raw dump
    # that made a 50-node rich filter a multi-hundred-KB firehose. Skinny nodes
    # (rich=False discovery path) render one-line-per-node. The MCP render
    # opt-in does not apply here: a multi-node scan is bounded by design — one
    # node's full view is a get_node away.
    if tool_name == "filter_nodes" and isinstance(result, dict) and "nodes" in result:
        nodes = result.get("nodes", [])
        total = result.get("total_count", len(nodes))
        if not nodes:
            return "No nodes matched. (%d total)" % total
        header = "%d node%s (of %d total)" % (
            len(nodes), "" if len(nodes) == 1 else "s", total)
        # Enriched nodes carry 'connections' (get_node always attaches it);
        # skinny nodes never do — a reliable discriminator.
        rich_nodes = [n for n in nodes
                      if isinstance(n, dict) and n.get('id') and 'connections' in n]
        if rich_nodes:
            from servers.contract import node_format_for
            return header + "\n\n" + _render_nodes(rich_nodes, node_format_for(len(rich_nodes)))
        # Skinny discovery shape — bounded one-liner per node (surfaces the
        # filtered field value; see contract.render_skinny_node).
        from servers.contract import render_skinny_node
        lines = [header, ""]
        for n in nodes:
            lines.append(render_skinny_node(n))
        return "\n".join(lines)

    # Trace tools — render via the single trace renderer (trace_contract),
    # never raw json.dumps. brain.query_traces/get_trace/get_traces return full
    # rows at the data layer; multi-row pulls are bounded here, rich=true opts
    # into full metadata.
    if tool_name in ("get_trace", "get_traces", "query_traces") and result:
        # Truncation banner: handled generically at the _format_result call
        # site (one chokepoint for every bounded read door), not here.
        cfg = None
        if tool_name == "get_trace":
            rows = [result] if isinstance(result, dict) else []
            # One named row — its body renders whole; rich swaps the metadata
            # gist for the full payload.
            from servers.trace_contract import TRACE_FULL_FORMAT, TRACE_POINT_FORMAT
            cfg = TRACE_FULL_FORMAT if rich else TRACE_POINT_FORMAT
        elif tool_name == "get_traces":
            rows = result.get("traces") or [] if isinstance(result, dict) else []
        elif isinstance(result, dict) and isinstance(result.get("chains"), list):
            # grouped — a chain header + its events, per chain. get_chains'
            # events carry their own id but scale/session_id are chain-level,
            # so propagate those onto each event before rendering (render_trace
            # expects a full row).
            chains = result["chains"]
            cfg = _select_trace_config(
                sum(len(c.get("events", [])) for c in chains), rich)
            blocks = []
            for c in chains:
                # Render from copies with chain-level scale/session_id filled in
                # (get_chains events carry neither) — never mutate the caller's
                # dicts. An event's own value, if present, wins (after **ev).
                merged = [{"scale": c.get("scale"), "session_id": c.get("session_id"), **ev}
                          for ev in c.get("events", []) if isinstance(ev, dict)]
                blocks.append('═══ chain %s · %s · %s · %d event%s ═══\n%s' % (
                    c.get("chain_id") or "?", c.get("scale") or "?",
                    (c.get("session_id") or "?")[:8],
                    len(merged), "" if len(merged) == 1 else "s",
                    _render_traces(merged, cfg)))
            return "\n\n".join(blocks) if blocks else "No chains found."
        else:
            rows = (result.get("chain") or result.get("events") or []) \
                if isinstance(result, dict) else []
        rows = [r for r in rows if isinstance(r, dict)]
        # Ids that matched no row are part of the answer — a shorter list than
        # you asked for must never be the only sign.
        missing = (result.get("missing_ids") or []) if isinstance(result, dict) else []
        tail = ("\n\n(no trace row for: %s)" % ", ".join(missing)) if missing else ""
        if rows:
            return _render_traces(
                rows, cfg or _select_trace_config(len(rows), rich)) + tail
        if tail:
            return tail.strip()
        # else fall through → json.dumps shows the small/empty shape

    if tool_name == "recall_batch" and isinstance(result, list):
        # Per-query results through the SAME recall formatter single recall
        # uses — not a raw dump.
        from servers.brain_voice import BrainVoice
        out = []
        for entry in result:
            if not isinstance(entry, dict):
                continue
            out.append('▸ "%s"' % entry.get("query", "?"))
            if entry.get("error"):
                out.append("  error: %s" % entry["error"])
            else:
                res = entry.get("results", [])
                if res:
                    lines = []
                    BrainVoice.format_recall_results(res, lines)
                    out.extend(("  " + ln) if ln else "" for ln in lines)
                else:
                    out.append("  No results found.")
            out.append("")
        return "\n".join(out)

    if tool_name == "recall" and isinstance(result, dict):
        from servers.brain_voice import BrainVoice
        results = result.get("results", [])
        # Strip _query_embedding — internal debug data, not for output
        result.pop("_query_embedding", None)

        lines = []
        if results:
            BrainVoice.format_recall_results(results, lines)
        else:
            lines.append("No results found.")

        # Show gap info if present
        gap = result.get("_gap")
        if gap:
            lines.append('No results above relevance threshold for: "%s"' % gap.get("query", ""))

        # Caller-facing note for exits the mode string alone doesn't explain
        # (e.g. include_archived=True) — _empty_recall sets it.
        note = result.get("_note")
        if note:
            lines.append(note)

        # Append recall stats
        stats = result.get("_embedding_stats", {})
        if stats:
            lines.append("---")
            lines.append("recall: %dms | mode: %s | sources: %s" % (
                stats.get("recall_ms", 0),
                result.get("_recall_mode", "?"),
                ", ".join("%s:%d" % (k, v) for k, v in
                          stats.get("results_by_source", {}).items() if v > 0)
            ))
        return "\n".join(lines)

    if tool_name == "recall_episodes" and isinstance(result, dict):
        # Shares the single trace renderer (TRACE_EPISODE_FORMAT keeps the
        # conversational framing: body only, no scale/event_type chrome).
        from servers.trace_contract import render_trace, TRACE_EPISODE_FORMAT
        eps = result.get("episodes", [])
        if not eps:
            return "No episodes found."
        out = ["%d episode%s · ranked by %s" % (
            len(eps), "" if len(eps) == 1 else "s",
            result.get("ranked_by", "time")), ""]
        for e in eps:
            out.append(render_trace(e, TRACE_EPISODE_FORMAT))
            out.append("")
        return "\n".join(out)

    return json.dumps(result, indent=2, default=str)


def _format_error(error, reply=None):
    """The text an agent sees for an ok=False reply: the error, then the
    refusal's structured pointers (contract.REFUSAL_POINTER_KEYS) as JSON,
    whole — the shape a batch row shows them in."""
    from servers.contract import refusal_pointers
    text = "ERROR: %s" % error
    pointers = refusal_pointers(reply) if isinstance(reply, dict) else {}
    if pointers:
        text += "\n\n" + json.dumps(pointers, indent=2, default=str)
    return text


def render_reply(tool_name, reply, rich=False, get_nodes_config=None,
                 error=None):
    """The text an agent reads for one tool reply — the one render the MCP
    proxy (handle_tools_call) and the encoder runner (run_llm_loop) share:
    render_reply_body, then render_reply_notes of the reply's notes
    (contract.reply_notes). A reply without notes renders the body alone."""
    from servers.contract import reply_notes
    return (render_reply_body(tool_name, reply, rich=rich,
                              get_nodes_config=get_nodes_config, error=error)
            + render_reply_notes(reply_notes(reply)))


def render_reply_body(tool_name, reply, rich=False, get_nodes_config=None,
                      error=None):
    """The result, headed by the truncation banner when the read was cut
    (contract.truncation_banner), or, for a reply that is not ok, the error
    with its pointers — `error` replaces the reply's own when the caller has
    named the failure better."""
    if reply.get("ok"):
        result = reply.get("result", {})
        text = _format_result(tool_name, result,
                              get_nodes_config=get_nodes_config, rich=rich)
        if isinstance(result, dict) and isinstance(result.get("truncated"), dict):
            from servers.contract import truncation_banner
            text = "%s\n\n%s" % (truncation_banner(result["truncated"]), text)
        return text
    return _format_error(error or reply.get("error", "Unknown"), reply)


def render_reply_notes(notes):
    """The tail after the body: each of a reply's notes (the dict
    contract.reply_notes returns) under its heading; '' when there are none."""
    return "".join("\n\n%s:\n%s" % (key.capitalize(), "\n\n".join(entries))
                   for key, entries in notes.items())


def handle_tools_call(request_id, params, extension=None):
    import time as _time
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        return make_error(request_id, -32602, 'Tool arguments must be an object.')
    if extension is not None and extension.handles(tool_name):
        clean = _stamp_caller_session(dict(arguments))
        verified = bool(clean.pop(CALLER_SESSION_KEY, None))
        extension.start(request_id, clean, identity_verified=verified)
        return None  # The extension owns its response, possibly deferred.

    identity_missing = False

    def identity_gap(reason):
        nonlocal identity_missing
        identity_missing = True
        _note_identity_gap(reason)

    def content(text):
        blocks = [{"type": "text", "text": text}]
        if identity_missing and extension is not None:
            blocks.append({"type": "text", "text": extension.notice})
        return blocks

    # Try up to 3 times with backoff — daemon may be restarting
    backoff = [0, 0.5, 1.5]  # immediate, 0.5s, 1.5s
    last_error = ""
    for attempt, delay in enumerate(backoff):
        if delay > 0:
            _time.sleep(delay)

        # Only TOOL calls carry identity; the proxy's own pings run headless by
        # design, so the identity-gap note is armed here and nowhere else.
        resp = daemon_send(tool_name, arguments, note=identity_gap)
        if resp.get("ok"):
            # `rich` is the MCP render opt-in for get_node/get_nodes (full view).
            # filter_nodes' own `rich` is a data-layer flag handled in dispatch;
            # its render is always bounded, so passing it here is harmless.
            return make_response(request_id, {
                "content": content(render_reply(
                    tool_name, resp, rich=bool(arguments.get("rich", False))))
            })

        # Distinguish a real daemon error from a missing-envelope response. A
        # reply with neither ok=True nor an `error` key means a dispatch handler
        # returned a raw payload dict instead of the {"ok": ...} envelope (the
        # dispatch_self bug, c4f6386). Name it loudly — the old "Unknown daemon
        # error" fallback turned that into a multi-turn hunt — and show what the
        # handler actually returned.
        if "error" in resp:
            last_error = resp.get("error") or "daemon returned ok=false with an empty error message"
        else:
            bad_keys = sorted(k for k in resp.keys() if k != "ok")
            last_error = ("daemon response missing the {ok,...} envelope — a dispatch "
                          "handler likely returned a raw dict; keys=%s" % bad_keys)
            sys.stderr.write("[brain-mcp] %s: %s\n" % (tool_name, last_error))
        is_connection_error = "connection" in last_error.lower() or "timeout" in last_error.lower()

        if is_connection_error and attempt < len(backoff) - 1:
            sys.stderr.write("[brain-mcp] Attempt {}: {} — restarting daemon...\n".format(attempt + 1, last_error))
            ensure_daemon_running()
            check_daemon_fingerprint()
        else:
            break

    return make_response(request_id, {
        "content": content(render_reply(tool_name, resp, error=last_error)),
        "isError": True
    })


def handle_ping(request_id):
    return make_response(request_id, {})


_stdout_lock = threading.Lock()


def send(msg):
    """Write JSON-RPC message to stdout."""
    line = json.dumps(msg)
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def send_notification(method):
    """Send a JSON-RPC 2.0 notification (no id, no response expected)."""
    send({"jsonrpc": "2.0", "method": method})


def check_daemon_fingerprint():
    """Check if daemon restarted (new code). If so, notify Claude Code to refresh tools."""
    global _last_daemon_fingerprint
    resp = daemon_send("ping", timeout=3.0)
    if not resp.get("ok"):
        return
    fp = resp.get("result", {}).get("code_fingerprint")
    if fp and _last_daemon_fingerprint and fp != _last_daemon_fingerprint:
        sys.stderr.write("[brain-mcp] Daemon fingerprint changed: {} → {} — notifying tools/list_changed\n".format(
            _last_daemon_fingerprint, fp))
        send_notification("notifications/tools/list_changed")
    _last_daemon_fingerprint = fp


def _read_stdin():
    """Read lines from stdin, surviving EOF and IO errors gracefully."""
    try:
        for line in sys.stdin:
            yield line
    except (IOError, BrokenPipeError, KeyboardInterrupt):
        pass
    sys.stderr.write("[brain-mcp] stdin closed — shutting down cleanly.\n")


def _log_proxy_error(hook_name, error, context, level="error"):
    """Persist a proxy-side event to brain_logs.db.hook_errors, which the
    dashboard errors panel, query_logs and the next boot read whether the
    daemon is up or not. daemon_client.log_hook_error does the write, the same
    one daemon recovery uses. Never raises."""
    from servers.daemon_client import log_hook_error
    log_hook_error(hook_name, error, context=context, level=level)


# Health-monitor cadence. One source for the ping loop, the failure arithmetic
# and the startup log line — a literal in any of them drifts silently when this
# moves. The interval is a deliberate throttle, not just a liveness knob: every
# proxy pings for its whole life whether or not a client is still attached, so N
# live proxies cost N/PING_INTERVAL connections per second against the daemon and
# hold as many ephemeral ports in TIME_WAIT (2*MSL = 30s). Hosts that keep an MCP
# server per unit of work rather than per session accumulate proxies, so N is set
# by the host's lifecycle, not ours.
PING_INTERVAL = 7.0
# ~20s grace before declaring the daemon down. Legitimate slow paths can eat
# 5-15s (surface_haiku under load, brain.save() under contention, cold-cache S2
# enrichment) — bailing at 6s caused false-positive alerts during normal
# operation. Below 20s = noise; above 20s = real. The grace is THRESHOLD x
# INTERVAL, so the two must move together: 3 x 7s = 21s of sleep.
# Add the ping's own cost to get real detection latency, and it differs by
# failure mode: a DEAD daemon refuses instantly (~21s), while a HUNG one — the
# case this monitor exists for — burns the 2s daemon_send timeout per attempt,
# so 3 x (7 + 2) = up to 27s, plus up to one interval before the first failing
# ping lands. Both stay in the "real" band. (The old 2s/10 pair was optimistic
# the same way: up to 40s against a claimed 20s. This is the cadence that
# tightened the hung-daemon worst case, not just the one that slowed pings.)
FAILURE_THRESHOLD = 3


def _health_monitor():
    """Background health monitor — pings the daemon every PING_INTERVAL.

    After FAILURE_THRESHOLD missed pings it logs DAEMON_DOWN to hook_errors
    (dashboard errors panel, query_logs) and force-recovers the daemon via
    recover_daemon(), retrying once per grace window while it stays down.

    Runs as daemon thread — dies when MCP process exits.
    """
    import time
    from servers.daemon_client import recover_daemon

    consecutive_failures = 0

    while True:
        time.sleep(PING_INTERVAL)
        try:
            resp = daemon_send("ping", timeout=2.0)
            if resp.get("ok"):
                if consecutive_failures > 0:
                    sys.stderr.write("[brain-mcp] Daemon recovered after %d failures\n" % consecutive_failures)
                consecutive_failures = 0
                continue
        except Exception:
            pass

        consecutive_failures += 1

        if consecutive_failures == FAILURE_THRESHOLD:
            sys.stderr.write("[brain-mcp] ALERT: Daemon unreachable for %ds — attempting restart\n" % (
                int(consecutive_failures * PING_INTERVAL)))

            # Persist the outage so the dashboard errors panel + query_logs
            # surface it whether the hook-side detector or this idle ping-loop
            # detector fires first.
            _log_proxy_error("DAEMON_DOWN",
                             "Daemon unreachable — MCP health monitor detected failure",
                             "mcp_health_monitor", level="critical")

            # Force-recover the hung daemon — kill + launchd respawn.
            # (ensure_daemon_running() only pings; a corpse won't exit on its
            # own, so launchd's crash-respawn never fires without this.)
            try:
                recover_daemon()
            except Exception as e:
                sys.stderr.write("[brain-mcp] Restart failed: %s\n" % e)

        elif (consecutive_failures > FAILURE_THRESHOLD
                and consecutive_failures % FAILURE_THRESHOLD == 0):
            # Retry restart once per grace window (THRESHOLD x INTERVAL = ~21s),
            # so the retry cadence tracks the interval instead of a bare literal.
            sys.stderr.write("[brain-mcp] Still down after %ds — retrying restart\n" % (
                int(consecutive_failures * PING_INTERVAL)))
            try:
                recover_daemon()
            except Exception:
                pass


def main(extension=None):
    # Ensure daemon is running — retry a few times since boot hook may be starting it concurrently
    sys.stderr.write("[brain-mcp] Starting MCP server...\n")
    import time, threading
    daemon_ready = False
    for attempt in range(4):
        if ensure_daemon_running():
            daemon_ready = True
            break
        if attempt < 3:
            sys.stderr.write("[brain-mcp] Daemon not ready, retry {}/3 in 2s...\n".format(attempt + 1))
            time.sleep(2)
    if daemon_ready:
        check_daemon_fingerprint()  # Record initial fingerprint
        count = len(TOOLS) + (len(extension.tools) if extension else 0)
        sys.stderr.write("[brain-mcp] Daemon connected. Serving {} tools.\n".format(count))
    else:
        sys.stderr.write("[brain-mcp] WARNING: Daemon not available at startup. Will retry on each tool call.\n")

    # Start health monitor (daemon thread — dies with MCP process)
    health_thread = threading.Thread(target=_health_monitor, daemon=True)
    health_thread.start()
    sys.stderr.write("[brain-mcp] Health monitor started (%gs interval).\n" % PING_INTERVAL)

    # Main loop — read JSON-RPC from stdin
    # Never crash: daemon going down/up is normal. Surface errors, keep serving.
    for line in _read_stdin():
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write("[brain-mcp] Bad JSON: {}\n".format(e))
            continue

        method = msg.get("method", "")
        request_id = msg.get("id")
        params = msg.get("params", {})

        # Server-originated elicitation responses have an id but no method.
        # Consume even unknown/expired replies; never bounce a response back
        # as a Method not found request and never reopen a cancelled review.
        if not method and ('result' in msg or 'error' in msg):
            if extension is not None:
                extension.receive(msg)
            continue

        # Notifications (no id) — acknowledge silently
        if request_id is None:
            if method == "notifications/initialized":
                pass  # Client acknowledged init
            elif method == 'notifications/cancelled' and extension is not None:
                extension.cancel(params.get('requestId'))
            continue

        try:
            if method == "initialize":
                send(handle_initialize(request_id, params, extension))
            elif method == "tools/list":
                send(handle_tools_list(request_id, extension))
            elif method == "tools/call":
                response = handle_tools_call(request_id, params, extension)
                if response is not None:
                    send(response)
            elif method == "ping":
                send(handle_ping(request_id))
            else:
                send(make_error(request_id, -32601, "Method not found: {}".format(method)))
        except Exception as e:
            sys.stderr.write("[brain-mcp] Unhandled error in {}: {}\n".format(method, e))
            try:
                send(make_error(request_id, -32603, "Internal MCP error: {}".format(e)))
            except Exception:
                pass  # stdout broken — nothing we can do

    if extension is not None:
        extension.close()


def load_extension(name):
    """Load only a packaged adapter; a broken extension leaves core tools usable."""
    if not name:
        return None
    import importlib.util
    from pathlib import Path
    import re
    try:
        if not re.fullmatch(r'[a-z][a-z0-9_]*\.py', name):
            raise ValueError('adapter must be a packaged Python filename')
        directory = Path(_parent).resolve() / 'hooks' / 'adapters'
        path = (directory / name).resolve()
        if path.parent != directory or not path.is_file():
            raise ValueError('adapter is missing or outside the package')
        spec = importlib.util.spec_from_file_location('_brain_host_extension', path)
        module = importlib.util.module_from_spec(spec)
        # The adapter may import siblings. Restrict this path addition to load.
        sys.path.insert(0, str(directory))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(directory))
        extension = module.create_extension(send, _parent, log=_log_proxy_error)
        names = [tool['name'] for tool in extension.tools]
        if len(set(names)) != len(names) or set(names) & {tool['name'] for tool in TOOLS}:
            raise ValueError('adapter tool names conflict with the shared tool inventory')
        return extension
    except Exception as error:
        message = 'Host extension unavailable: %s' % error
        sys.stderr.write('[brain-mcp] %s\n' % message)
        _log_proxy_error('mcp_extension', message, 'extension loading', level='warning')
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Brain MCP proxy')
    parser.add_argument('--adapter', help='Packaged host extension filename')
    main(extension=load_extension(parser.parse_args().adapter))
