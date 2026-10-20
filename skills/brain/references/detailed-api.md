# brain — Detailed API Reference

Tool signatures and field reference for the brain plugin. The MCP tool schemas
are the authoritative signatures — they are generated from `servers/contract.py`
at daemon startup and carry full per-parameter documentation. This file is the
map: what exists, how the pieces relate, and the semantics that don't fit in a
parameter description.

## Architecture

A launchd-managed daemon (`servers/daemon_server.py`) owns the databases and the
embedder; the MCP server (`servers/brain_mcp.py`) is a thin stdio proxy that
forwards tool calls to it over TCP localhost. Hooks observe the session
(SessionStart, Stop, ...) and talk to the same daemon.

Two databases:

- **`brain.db`** — the knowledge graph: nodes, edges, embeddings, metadata.
- **`brain_logs.db`** — the operational substrate: trace events, session state,
  interaction templates, errors.

## Tools

### Writing memory

| Tool | Purpose |
|---|---|
| `remember` | Create one node. Required: `type`, `title`, `content`. With `connect_to`, the reply's `connect_to_result` reports which edges formed and why any didn't. |
| `remember_batch` | Create many nodes in one call. Per-node `connect_to` draws typed edges to siblings (created in the same batch) or catalog nodes. |
| `revise` | Update fields on an existing node. Required: `node_id`, `reason` (audit note recorded in trace events, not stored on the node). Every text field takes its whole new value or `{old, new}` swaps (a list for several spots) that change only the stale span; `old` must match the stored text exactly once or the whole op is refused and nothing is written. A key a node carries that the tool does not list goes inside `fields: {key: value or swap}`, under its key. Unspecified fields are preserved. |
| `revise_batch` | Many revisions in one call. On revise, `source_refs` has REPLACE semantics — omit to preserve, `[]` wipes. |
| `connect` | Create/update a typed edge between two existing nodes. If the two nodes already have this relation, `connect` updates it and stores it as `source_id` → `target_id`; to change only an edge's description, `revise` a node with `connect_to`. Repeated calls do not auto-strengthen weight. |
| `connect_batch` | Many edges in one call, same upsert contract. |
| `revise_edge` | Rename/redescribe an existing edge relation in place — same edge, weight, and history. |
| `brain_batch` | Mixed operations in one transaction. Six ops: `remember`, `revise`, `connect`, `disconnect` (soft-archive one relation on an edge), `archive` (soft-archive a node; `survivor_id` records supersession lineage), `absorb` (merge one node into another — edges, refs, and metadata move over on their own; its content does not, so send `content` that also says what the absorbed node said). |
| `set_node_lock` | The one door for lock flips (`revise` treats `locked` as immutable). Two-phase: first call returns a one-shot `confirm_token` + summary; relay to the human, get an explicit yes, then re-call with the token. Tokens expire in 10 minutes. |

Edge rules that apply everywhere: relations are open-text verbs, but never
`related`/`related_to` — generic relations pollute the activation kernel and
match no query. Every edge `description` should be ≥30 chars naming the
insight between the two nodes; edge descriptions are embedded and searched at
recall. For an edge involving a node being created in the same call, use
`connect_to` (a node created in the same call is named by its exact title,
resolved after all creates; an existing node by its id) — never a separate
`connect` op, whose ids don't exist yet.

### Reading memory

| Tool | Purpose |
|---|---|
| `recall` | Semantic search over nodes by meaning. `query` (or `node_id` for direct lookup), optional dict `filter`, `limit` (default 8). Phrase queries as what you'd remember, not keywords. |
| `recall_batch` | Multiple recall queries in one call (default 5 results each). |
| `get_node` | Fetch one node by id. Bounded view by default (full content, top edges, correction gist); `rich=true` for everything. |
| `get_nodes` | Batch fetch. The count picks the view: a small batch comes back whole (full content, top edges with the total), a larger one as excerpts with fewer edges; the tool description gives the cutoffs. `rich=true` gives everything at any size. |
| `find_node_by_title` | Fuzzy title match via embedding similarity. `threshold` default 0.75, `top_k` default 1. |
| `filter_nodes` | Structured query on any structural field: `include`/`exclude` value lists, `lt`/`gt` ranges, `contains`/`prefix` LIKE matches. `limit` default 50, max 200. `rich=true` (default) returns enriched nodes; `rich=false` a skinny id/title/type list. |

The `filter` dict (recall/recall_batch) supports per-key operators: `exists`,
`equals`, `in`, `contains`, `gte`, `lte`. Node columns are checked on the
result row; other keys are checked in metadata. Example:
`{"type": {"in": ["decision"]}, "their_raw_quote": {"exists": true}}`.

Every canonical node pull walks correction edges and attaches `_corrections` —
renders show a ⚠ line directly under the node's content for each node that
corrected this one, naming what happened to its claim (`⚠ Replaced by`,
`⚠ Corrected by`, `⚠ Resolved by`, `⚠ Reframed by`; `⚠ Updated by` when the
verb has no curated flavor; `⚠ Correction:` when it names no side), so a stale
claim is never served without its correction. The line names the corrector
the way an edge line does, then the verb, with `description:` on the next line;
get_nodes and recall do not repeat that relation among the edges.
A node this one corrected is history, as is a relation that changed nothing in
the claim yet (`addresses`); both appear among its ordinary edges. Edges arrive
as `connections` with direction, relation, and description.

Bounded window reads that hit their row limit attach a `truncated` payload and
render a `⚠ TRUNCATED` banner naming the covered time slice — a partial result
never silently reads as a complete one. Ranked top-k reads (recall) are exempt:
there, truncation is the contract.

### Traces and episodes

Every scale of the system writes its activity to `trace_events`: `scale` `s0`
(the conversation itself), `s1` (per-turn recall/surface/encoding runs), `s2`
(idle integration units); `event_type` `O` (observed), `K` (knowledge selected),
`delta` (changed). Events group into chains (`chain_id`), and ids are 8-char
hex strings — the same ids `source_refs` on nodes point at.

| Tool | Purpose |
|---|---|
| `recall_episodes` | Episodic recall — the verbatim record of what was said and done, with attribution. `query` (semantic) and/or `contains` (exact substring), composable. Defaults to conversation messages at s0; `ref_type='tool_result'` for the "what I did" lens. `limit` default 10, cap 500. |
| `query_traces` | Search trace events by scale / event_type / ref_type / chain / session. `hours` default 24; a plain `session_id` or `session_ids` filter ignores the window (it still applies with `ref_type`, and with `grouped=true`, and the reply says so). A session filter takes the full id or the short trace headers show; `session_ids` takes several, never an empty list. `grouped=true` nests events by chain — it needs `session_id` or `session_ids` and is refused beside `ref_type` or `event_type`. |
| `get_trace` | Point-lookup by id — the body renders whole, never trimmed. `rich=true` swaps the metadata gist for the full payload (the run record on s1/s2 encoder rows; nothing extra on an s0 turn). |
| `get_traces` | Batch point-lookup — the natural way to expand a node's `source_refs`. Serves 50 ids per call and flags the overflow; unmatched ids are named. Bodies are trimmed with the dropped-char count; `rich=true` for full metadata. |
| `count_traces` | Group-by counts over trace events (`event_type`, `ref_type`, `chain_id`, `scale`). |

Rule of thumb: `recall` answers "what do I know", `recall_episodes` answers
"what actually happened" — memories can be stale about an event; the episodes
never are.

### Self channel (parallel sessions)

Concurrent sessions of the same agent are discoverable and reachable:

| Tool | Purpose |
|---|---|
| `self_presence` | Roster of live sessions, each with its arc and recent activity. |
| `self_peek` | Read one session's current focus. Read-only, no interruption. |
| `self_send` | Message a session's inbox (the short id you see, the full session id, or `broadcast`). Consumed once. |
| `self_inbox` | Drain messages other sessions sent you. |
| `self_outbox` | Delivery receipts for what you sent — distinguishes "delivered, not acted on" from "never delivered". |

### Thalamus (standing intents)

| Tool | Purpose |
|---|---|
| `remind` | File an item with delivery policy: a reminder (`what` + `when` — fires at the first session after due), a notice (`for_whom='all'` window broadcast, or `'live'` one-shot to live sessions), or an ask (`needs_answer=true` — renders at every session boot until answered, or until a cap of sessions passed on it, then expires loudly). `dedup_key` makes re-filing update instead of duplicate; `expires` sets when delivery ends (default: the kind's window from `when`). |
| `thalamus_list` | The open queue, due items first (most recently due first), then scheduled ones (soonest first); `include_closed=true` adds terminal items. A page at a time: `total` counts every match, `offset` reads the next page. |
| `thalamus_resolve` | Close or defer one item: exactly one of `answer`, `defer_until`, or `dismiss=true`. |

### Interactions (the K store)

Every learnable boundary (surface prompt, encoder prompt, S2 units, ...) is a
named interaction. The code default is authoritative; the database holds only
deployed overrides.

| Tool | Purpose |
|---|---|
| `list_interactions` | All names with version counts and the active override pointer (null = running on code default). |
| `get_interaction` | One registered version's raw template row. |
| `get_interaction_effective` | What actually runs: code default with any active override overlaid, plus provenance stamp. |
| `register_interaction` | Register version N+1. Never activates. |
| `set_interaction_active` | Deploy a registered version as the override. |
| `clear_interaction_override` | Delete the pointer — revert to the code default immediately. |

### Operational

| Tool | Purpose |
|---|---|
| `query_logs` | Operational logs: `source` = `errors` (hook failures), `debug` (daemon events), or `all` (merged timeline). Filter by `level` or `hook_name`. |
| `clear_errors` | Clear hook errors (and optionally debug log entries). |
| `restart` | Reload the daemon in place with fresh code (same PID, ~2-4s). |
| `eval` | Escape hatch — evaluate a Python expression with the `brain` instance in scope. |

## Node fields

Defined in `servers/contract.py` — the single source every layer (schema, DAL,
dispatch, recall, MCP schemas, encoder prompts) derives from.

**Agent-writable fields** (`remember` / `revise` / batch ops):

| Field | Type | Notes |
|---|---|---|
| `type` | str | Required. Open vocabulary — no constraint; use what fits (`decision`, `lesson`, `mechanism`, `correction`, `moment`, `open`, ...). |
| `title` | str | Required. Itself an embedded recall vector — specificity is findability. |
| `content` | str | Required. Rich — future readers have zero context. |
| `situation` | str | "When is this relevant?" — one sentence, its own embedding. The single biggest lever for recall. |
| `question` | str | The question this node answers, as the other side would ask it. Own embedding. |
| `reasoning` | str | What the claim rests on — how established, how strongly, what would change it. |
| `thought` | str | The agent's own read on the memory — rendered beside it at recall. |
| `their_raw_quote` / `my_raw_quote` | str | Verbatim words — meaning paraphrase loses. |
| `event_time` | str | When the thing HAPPENED (ISO 8601), distinct from `created_at` (when written). |
| `confidence` | float | 0.0-1.0; below 1.0 for hedged/contested/inferred claims. |
| `emotion` / `emotion_label` | float / str | Signed charge + felt register. |
| `evolution_status` | str | Claim lifecycle: `active`, `resolved`, `validated`, `confirmed`, `disproven`, `dismissed`. |
| `correction_pattern` | str | Behavioral pattern behind a correction. |
| `source_context` | str | Session/context when encoded. |
| `locked` | bool | Set at creation only; flips go through `set_node_lock`. |
| `source_refs` | array | 1-3 8-char hex trace ids anchoring the node to the moments that generated it (join table, not a column). |

**System-stamped fields** — never agent-authored; the write boundary derives
them and drops supplied values: `project` (repo provenance — where this was
learned), `counterpart` (who the session was with), `encoding_source` (who
wrote the node: `anchor`, `encoder:sonnet`, `s2:<unit>`, `hook:<event>`;
only `anchor` can lock).

**Immutable**: `id`, `created_at`, `updated_at`. Revision history lives in
trace events (`ref_type='node_revised'`), not on the node.

### Embedding views

Each node is searchable through multiple vectors, not one: `title`, the
title+content blend (`_primary`), `question`, `_situation`, a
situation+quotes blend, a reasoning blend, and `edge_context` (concatenated
edge descriptions — a node is findable through its relationships). Title and
question are the highest-weighted views. Per-field vectors additionally feed
the per-turn surface kernel. Registry: `EMBEDDING_GROUPS` in
`servers/pipeline_contract.py`.

## Edge model

Two tables: `edges` holds one physical row per node pair (`edge_id`,
`source_id`, `target_id`, aggregate `weight`); `edge_relations` holds one or
more semantic relations per edge, each with its own claim direction (the claim
reads `source relation target`), open-text `relation`, `description`,
`weight`, and stored embedding. Which end acts is the verb's orientation, not
the end's position.
Disconnect soft-archives a single relation; other relations on the pair
survive.

Relations and node types are grouped into semantic roles by the aspect
taxonomy (`servers/scales/s2/aspects_v1.json`, read via `brain.aspects`) —
e.g. the `correction_improvement` aspect (`corrects`, `supersedes`,
`reframes`, `resolves`, `fixes`, ...) is what makes an edge behave as a
correction at recall time, and `noise` relations (`community_member`, ...) are
excluded from standard edge renders. Adding an aspect is a human edit to the
JSON; encoders route strings into existing aspects only.

## Database schema

`brain.db`: `nodes`, `edges`, `edge_relations`,
`node_source_refs`, `node_metadata_kv` (promoted fields live here as K/V),
`node_enrichments` (all embedding vectors), `nodes_fts` (full-text title and
content index), `s2_rejections`, `brain_meta`.

Key `nodes` columns: `id`, `type`, `title`, `content`, `confidence`, `locked`,
`archived`, `emotion`, `emotion_label`, `evolution_status`,
`encoding_source`, `revised_at`, `last_accessed`, `created_at`, `updated_at`.
Everything else (situation, question, reasoning, quotes, project, ...) lives
in `node_metadata_kv`.

`brain_logs.db`: `trace_events` + `trace_embeddings` (the episodic substrate),
`session_state`, `interactions` + `interaction_active` (the K store),
`hook_errors`, `debug_log`, `self_inflight` + `self_delivered` (self channel),
`thalamus_items` + `thalamus_deliveries`, `boot_renders`, `logs_meta`.
