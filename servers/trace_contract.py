"""Trace Contract — single source of truth for the fractal trace system.

Defines: scales, event types, ref types, and validation.
All trace writers MUST validate against this contract.
All trace readers can rely on these guarantees.

Architecture: docs/ARCHITECTURE-FRACTAL.md
"""

from servers.loud_truncation import (cap_text_loud, cap_list_loud,
                                     compose_block_loud, one_line)


# ── SCALES ──
# The fractal hierarchy. Each scale observes the one below.

SCALES = {
    "s0": {
        "name": "Exchange",
        "description": "Raw partnership interaction — messages, tool calls, tool results",
        "triggers": "Every turn (Stop hook) + every tool call (PostToolUse hook)",
    },
    "s1": {
        "name": "Turn",
        "description": "Brain's first processing pass — surface, encode",
        "triggers": "UserPromptSubmit (surface) + Stop every 5th (encode)",
    },
    "s2": {
        "name": "Graph",
        "description": "Graph-wide operations on S1's accumulated output — communities, dedup, confidence, corrections",
        "triggers": "Idle hook (between sessions)",
    },
    "s3": {
        "name": "Reasoning",
        "description": "Cross-cluster patterns, abstract insights, resolved uncertainties",
        "triggers": "Periodic / scheduled",
        "status": "NOT BUILT",
    },
    "s4": {
        "name": "Growth",
        "description": "External knowledge and long-term evolution",
        "triggers": "Periodic / weekly",
        "status": "NOT BUILT",
    },
}

# Scales that write traces today — what an agent can filter by.
BUILT_SCALES = tuple(k for k, v in SCALES.items() if v.get("status") != "NOT BUILT")


# ── EVENT TYPES ──
# O/K/Δ — structurally identical at every scale.

EVENT_TYPES = {
    "O": "Observation — everything available at this moment",
    "K": "Knowledge — what was selected as relevant from O",
    "delta": "Changes — what was produced (the response, encoding, reorganization)",
}


# ── REF TYPES ──
# What ref_type values are valid per scale + event_type.
# ref_type tells you WHAT the event is about. ref_id points to it.

# The Thalamus delivery marker — declared here (not in thalamus_contract)
# because channels/delivery.py needs it at load and thalamus_contract imports
# delivery for the moment names; this file imports neither, so the constant
# is reachable from both without a cycle. thalamus_contract re-exports it.
REF_THALAMUS_DELIVERY = "thalamus_delivery"
# The filing-side marker — one row on the PRODUCER's run chain per accepted
# filing (ref_id = item id), the symmetry journal_note rows already have. An
# item's life is then joinable across scales: filed (s1 Δ, the encoder's
# chain) → delivered (s0 K thalamus_delivery, the session's chain) → answered
# (item state). Written by brain_traces.write_thalamus_filed for every
# journaling encoder — the S1 Scribe (directed, Stop) and the S2 units
# (broadcast, boot); never s0 (a filing is a run's act, not a turn's).
REF_THALAMUS_FILED = "thalamus_filed"

REF_TYPES = {
    # Scale 0: raw exchange
    ("s0", "K"):       ["user_message",
                         "self_message",      # incoming turn from a stream of thought (self↔self),
                                              # not the operator — same exchange, different correspondent
                         REF_THALAMUS_DELIVERY,  # Thalamus items rendered into this session (boot or
                                              # Stop — channels/delivery.py traces both legs); the
                                              # brain speaking to its streams
                         "heartbeat"],        # a /watch wakeup re-arm with no real input (no operator
                                              # prompt, empty inbox). Recorded for observability, but
                                              # NOT a conversational turn — see S0 TURN CLASSIFICATION.
    ("s0", "delta"):   ["assistant_message", "tool_result",
                         "node_revised", "edge_relation_revised",
                         # Mutation events from the emitter (servers/mutation_emitter.py).
                         # Registered at s0/s1/s2 — the same three scales the revise pair
                         # occupies — because scale is derived per row from the mutation's
                         # own encoding_source, so one command can emit at several scales.
                         "node_created", "node_archived",
                         "node_lock_changed",  # lock flip via set_node_lock (emitter)
                         "anchor_touched"],   # per-turn aggregate of what Anchor's own
                                              # MCP tools touched this turn (the keyset is
                                              # ANCHOR_TOUCHED_KEYS) — the S0 mirror of the
                                              # S1 encoding_run delta; feeds the encoder's
                                              # widened catalog (trace_links). See
                                              # build_anchor_touched_metadata.

    # Scale 1: turn integration
    # Surface path (chain prefix: s1r-): O=candidates, K=surfaced picks, delta=context sent to Anchor
    # Encode path (chain prefix: s1e-): O=prompt given, K=node catalog, delta=actions+reasoning
    ("s1", "O"):       ["recall",            # candidates with scores
                         "encoding_prompt",    # what the encoder was given
                         "scout_input"],       # retired with the scout muster; history rows read through it
    ("s1", "K"):       ["surface_selected",  # what the surfacer picked
                         "node_catalog",       # which nodes available to encoder
                         "scout_findings"],    # retired with the scout muster; history rows read through it
    ("s1", "delta"):   ["additionalContext",       # what reached Anchor
                         "encoding_run",            # what the encoder produced
                         "encoding_run_failed",     # LLM loop died — no writes; NOT read by
                                                    # trace_links (coverage joins on encoding_run
                                                    # only, so a failed run never claims turns)
                         "node_revised",            # field-level revise emitted by S1 encoder
                         "edge_relation_revised",   # connect upsert / archive emitted by S1 encoder
                         "node_created",            # node born (emitter) — closes the catalog gap
                         "node_archived",           # node archived (emitter)
                         "node_lock_changed",       # lock flip (emitter; scale derived per row)
                         "journal_note",            # S1 Scribe residue — one note (subject=ref_id) per row
                         REF_THALAMUS_FILED],       # S1 Scribe filed a Thalamus item (ref_id = item id)

    # Scale 2: graph integration
    # Fires during idle hook. Operates on S1's accumulated output (the graph).
    # Multiple integration units, each with own O/K/Δ.
    ("s2", "O"):       ["graph_structure",      # nodes + edges observed (community detection)
                         "graph_stats",          # node/edge counts, density
                         "s1_delta",             # S1 encoding/surfacing traces since last run
                         "consolidation_candidates",  # embedding scan + behavioral evidence
                         "heal_archive",         # decoder-level archive of broken artifacts (e.g. 0-member communities)
                         "correction_chains",    # brain-wide correction chain traversal
                         "healer_scan",          # S2 Healer: gaps + flags scanned
                         "aspect_scan"],         # S2 AspectIntegration: distinct types/relations vs aspects_v1.json
    ("s2", "K"):       ["community_proposals",  # S2CD proposals (placements, overlaps, splits, seeds)
                         "community_partition",  # algorithm output (communities + membership)
                         "community_diff",       # comparison with previous run
                         "consolidation_proposals",   # enriched clusters with pre-classification
                         "stale_nodes",          # nodes not accessed recently
                         "healer_proposals",     # S2 Healer: nodes to heal (fill missing fields)
                         "aspect_proposals"],    # S2 AspectIntegration: candidate strings + example records
    ("s2", "delta"):   ["community_enriched",   # S2CE enrichment results (accepted, rejected, placed)
                         "community_created",    # new community node
                         "community_updated",    # revised community node
                         "community_removed",    # stale community archived
                         "community_assignments",# membership edges updated
                         "recall_quality_signal",# recall diagnostic (false positive, redundancy, gap)
                         "consolidated",         # new node from smart merge
                         "evolved",              # evolution edge added
                         "kept_distinct",        # similar_to edge, no merge
                         "confidence_adjust",       # adjusted confidence scores
                         "healer_generated",        # S2 Healer: missing fields generated + stored
                         "aspect_classified",       # S2 AspectIntegration: candidates merged into aspects_v1.json
                         "node_revised",            # field-level revise emitted by S2 units (healer, consolidation)
                         "edge_relation_revised",   # connect upsert / archive emitted by S2 units
                         "node_created",            # node born (emitter)
                         "node_archived",           # node archived (emitter) — S2 units archive
                                                    # superseded nodes and dead communities
                         "node_lock_changed",       # lock flip (emitter; scale derived per row)
                         "journal_note",            # S2 unit residue (consolidation, community) — one note per row
                         REF_THALAMUS_FILED],       # S2 unit filed a Thalamus item (ref_id = item id)

    # Scale 3: reasoning integration
    # Operates on S2's output (clusters, trajectories, landscapes).
    ("s3", "O"):       ["cluster_patterns",     # S2 clusters across parameters
                         "correction_trajectories",  # how understanding evolved
                         "confidence_landscapes"],    # stable vs turbulent areas
    ("s3", "K"):       ["cross_cluster",        # nodes appearing across multiple clusters
                         "learning_curves"],     # correction trajectories over time
    ("s3", "delta"):   ["abstract_insight",     # cross-cluster pattern recognized
                         "resolved_question",    # uncertainty answered
                         "meta_optimization"],   # S2 prompt/config improvement

    # Scale 4: growth integration
    # Fires periodically (weekly). Sees full graph + external sources.
    ("s4", "O"):       ["uncertainty_nodes",   # brain's open questions
                         "external_research"],  # web search results, papers
    ("s4", "K"):       ["stale_decisions",     # decisions that may be outdated
                         "open_questions"],     # unresolved uncertainties
    ("s4", "delta"):   ["research_finding",    # new knowledge from outside
                         "decision_update",     # stale decision refreshed
                         "cross_project"],      # bridge between projects
}


# ── S0 TURN CLASSIFICATION ──
# Every stop produces one S0 turn, classified by its incoming-side (s0,"K")
# ref_type. NOT every turn is a conversation worth encoding. This is the single
# source of truth for "what's filtered and what's not" — consumers read these
# constants instead of re-deciding inline.
#
#   incoming ref_type   what it is                                conversational?
#   ──────────────────────────────────────────────────────────────────────────
#   user_message        real operator prompt (hook_recall ran;     YES
#                        last_user_activity reset this turn)
#   self_message        inbound msg from another stream            no  (planned)
#                        (anchor↔anchor)
#   thalamus_delivery   the brain's own due items rendered into    no  (planned)
#                        this session (channels/delivery.py)
#   heartbeat           /watch wakeup re-arm, no real input        no  (never)
#                        (no prompt + empty inbox)
#
# The dial governs TWO things, both timeline-side:
#   (a) the encoder's conversation window — get_session_turns selects
#       CONVERSATIONAL_REF_TYPES, derived below, so a True row's turns enter
#       the timeline as their own incoming side (the trace's metadata.content
#       carries the delivered block; channels/delivery.py stamps it);
#   (b) reaction classification — the stop right after a delivery-block
#       (post_response_common's delivery-continuation branch) records the
#       response as a real assistant_message iff any delivered ref_type is
#       True here; otherwise it stays a heartbeat. Flipping a row therefore
#       makes the incoming message AND the reaction appear together.
# The dial does NOT govern the S1 Scribe's CADENCE: turns_since_last_encode
# counts s0 user_message rows only (dal_logs.conversational_turns_since,
# hardcoded ref_type), so a flipped correspondent enters the conversation
# cleanly without ticking encode cadence. Cadence is also distinct from
# stop_counter, the per-stop SEQUENCE number that advances on EVERY stop
# (incl. heartbeats) so chain IDs stay unique. Non-conversational turns are
# still written to S0 (for observability) but never drive or feed encoding.
#
# anchor↔anchor and brain↔anchor encoding are PLANNED capabilities, switched
# OFF until the encoder prompt is taught the correspondent elements. The
# single dial per correspondent is below. heartbeat stays False forever.
S0_CONVERSATIONAL_INCOMING = {
    "user_message": True,
    "self_message": False,        # another stream of me — flip to include anchor↔anchor turns
    "thalamus_delivery": False,   # the brain's own items — flip to include brain↔anchor turns
    "heartbeat":    False,        # a wakeup re-arm is never a turn
}

# Flat ref_type set the encoder's conversation window (dal.get_session_turns)
# selects: the conversational incoming types + the assistant response side.
# DERIVED from S0_CONVERSATIONAL_INCOMING so there is exactly one dial. This
# whitelist binds at IMPORT; the classification half (arms_continuation,
# below) reads the dict live so tests can simulate a flip — in production
# both move together, because a flip is a source edit + daemon restart
# (deploy semantics, like every contract constant), never a runtime mutation.
CONVERSATIONAL_REF_TYPES = tuple(
    rt for rt, conv in S0_CONVERSATIONAL_INCOMING.items() if conv
) + ("assistant_message",)

def arms_continuation(traced_ref_types):
    """The dial's classification consumer, named: a Stop-blocking delivery of
    these (traced) incoming ref_types arms a reaction stamp iff any of them
    is a dial-on correspondent. Reads the dial dict LIVE by design — a test
    simulates a flip by patch.dict'ing one row; in production the dict only
    changes with a source edit + restart, so this and the import-frozen
    timeline whitelist move together at deploy time."""
    return any(S0_CONVERSATIONAL_INCOMING.get(rt, False)
               for rt in traced_ref_types)


# The window measures the WHOLE continuation turn — armed at the end of the
# blocked Stop, compared at the start of the next one — not the resume
# latency (the counter match already restricts to the very next Stop). So it
# must accommodate a continuation that runs a test suite or an agent chain,
# while still refusing an abandoned one: ESC fires no Stop, freezing counter
# and stamp, and a /watch wakeup HOURS later would otherwise claim the
# reaction. Wall-clock: hooks are off the grain axis.
DELIVERY_REACTION_WINDOW_MIN = 60


# ── S0 SESSION STAMP ──
# Per-session facts every S0 row carries, next to the identity stamp: which
# model produced the turn and which resolved host runtime it rides on.
# Unlike human_identity / agent_identity — a
# process-wide property stamped by TraceDAL from env — these vary PER SESSION
# and per turn (one daemon serves streams on different models; a stream can
# switch model mid-session), so they live on the SessionContext and are
# stamped by the S0 write door (brain_traces.stamp_s0_session) from the ctx
# the daemon resolved. Fed in by the UserPromptSubmit / Stop hooks
# (hook_common.turn_model / host_tells): Codex puts `model` on every hook
# payload; Claude Code exposes it only in the transcript's assistant entries.
# The session row mirrors the LATEST value so presence can say which model a
# stream is on right now; the per-turn truth is the S0 row.
S0_SESSION_STAMP_FIELDS = ('model', 'host')

# Set True on a user_message row the recall hook registered without running
# recall or surface (a short answer, a wake envelope — the client routes it
# register_only). The turn stays conversational; readers that pair turns with
# surface runs read this instead of re-deciding the client's rule.
REGISTER_ONLY_FIELD = 'register_only'


# ── HOST NORMALIZATION VOCABULARY (the normalizer's OUTPUT) ──
# What a row says about a tool call or a host once the host's own names have
# been translated. Host DIALECTS (which tool name means what on which harness,
# which env tell identifies it) live in servers/host_contract.py and are
# validated against THESE tuples; consumers past the boundary (encoder, presence,
# Frame) read only these and never a host's name. Output vocabulary on the
# output side, so the encoder never imports a host dialect to read a kind
# (docs/HOST-CONTRACT-DESIGN.md D1).
ACTION_KINDS = ('edit', 'shell', 'read', 'search', 'agent', 'mcp')
# Stamped beside the kind: did the host's tool map know the name?
KIND_STATUS = ('ok', 'unknown')
# How the harness was identified for a row: a host-specific tell fired, only a
# family tell (a CC-compatible plugin host, not necessarily Codex), tells of
# two hosts fired, none fired, or the row came from a client that sent a
# pre-resolved host name and no tells (legacy wire).
HOST_STATUS = ('strong', 'family', 'ambiguous', 'unknown', 'legacy')
# Per-envelope-tag policy values; `extract:<name>` names a registered pure
# extractor in host_contract.EXTRACTORS (the registry is closed — a name with
# no function is a contract violation, not a first-prompt KeyError).
ENVELOPE_POLICIES = ('drop', 'keep', 'marker')
ENVELOPE_EXTRACT_PREFIX = 'extract:'


# Operator dialogue — the two ref_types that ARE the operator↔Anchor
# exchange. Presence (focus / recency ranking / recent_msgs), the
# recall_episodes conversation default, the LAF trace matrix, and the
# dual-store trace chain are PINNED here, deliberately NOT dial-derived: a
# correspondent flipped on in the dial enters the encoder timeline (and its
# embed lockstep) WITHOUT changing what "the conversation" means to
# presence, episodes, or scoring. Flipped correspondents stay reachable
# there explicitly (recall_episodes ref_type='self_message' /
# 'thalamus_delivery' — the same opt-in convention as tool_result).
OPERATOR_DIALOGUE_REF_TYPES = ("user_message", "assistant_message")

# Presence LIVENESS — operator dialogue PLUS heartbeat. A watch listener living
# purely on heartbeats is the most reachable stream there is, so a heartbeat
# proves this session can be reached (B2, 2026-06-04).
#
# It does NOT prove work happened, so this set answers "can it be reached", never
# "has it gone quiet". A gate asking the second question wants the row set its own
# counter uses — see conversational_turns_since — not this one. Named here rather
# than assembled inline so a consumer inherits it visibly.
PRESENCE_LIVE_REF_TYPES = OPERATOR_DIALOGUE_REF_TYPES + ("heartbeat",)

# The "said + did" timeline: conversation plus tool activity. What the S1
# encoder's lived timeline reads. The embed queue deliberately does NOT
# follow this set (see embed_queue.EAGER_TRACE_REF_TYPES): a tool row is
# rendered in the timeline but carries no vector, because no recall lane
# ranks one.
SAID_AND_DID_REF_TYPES = CONVERSATIONAL_REF_TYPES + ("tool_result",)


# A wakeup ignite (e.g. a background-task notification) arrives as turn CONTENT,
# not a distinct ref_type: it runs recall, so it's recorded as a `user_message`
# (conversational) even though it's an ENVELOPE, not work. Presence focus skips
# any conversational turn whose summary starts with this marker. One constant so
# the skip is defined ONCE, not reproduced as a scattered SQL literal — the
# predicate below reads it too. Claude Code's envelope tag; it moves into the
# host contract's envelope table when the boundary classifies envelopes
# (docs/HOST-CONTRACT-DESIGN.md step 3).
WAKE_ENVELOPE_MARKER = "<task-notification>"


def is_machine_turn(op_text) -> bool:
    """Harness-injected machine turn — a background-task completion packaged
    as a prompt through UserPromptSubmit. NOT an operator turn: the operator
    side is dropped wherever turns feed scoring (the LAF moment stack, the
    walker's labeling — v6's 778-mislabel lesson), while Anchor's response to
    it is real and kept as history. ONE definition shared by production and
    eval (eval/laf/walker/extract.py imports this) so the filter can't drift;
    production's recall hook routes these register_only (node b2953766)."""
    return WAKE_ENVELOPE_MARKER in (op_text or '')


# ── CHAIN ID CONVENTIONS ──
# chain_id groups related O/K/Δ events.
#
# One chain per stop at S0. Everything between stop N-1 and stop N
# (messages, tool calls) belongs to the same S0 chain.
# S1 chains reference the S0 chain via parent_chain in metadata.

CHAIN_PREFIXES = {
    "s0":         "s0-{session_short}-{stop}",        # one chain per stop — messages + tools
    "s1_recall":  "s1r-{session_short}-{stop}",       # surface for this stop
    "s1_encode":  "s1e-{session_short}-{stop}",       # encoding run triggered at this stop
    "s2":         "s2-{datetime}-{operation}",          # datetime=YYYYMMDDHHMMSS (seconds, per-run — see s2/base.py chain_id), operation=community/consolidation/etc
    "s3":         "s3-{date}-{operation}",             # date=YYYYMMDD, operation=synthesis/meta/etc
    "s4":         "s4-{date}-{topic}",                 # date=YYYYMMDD, topic=what was researched
}


def scale_for_chain(chain_id):
    """The scale a chain id encodes, from its CHAIN_PREFIXES prefix
    ('s1e-…' → 's1'). A writer handed a run chain by its caller need not be
    handed the scale too — a second parameter for the same fact drifts (a
    chain-only call once defaulted the scale to '' and silently lost every
    row). Raises ValueError on a chain no prefix claims: an unknown chain is
    a producer bug, not a row."""
    for key, template in CHAIN_PREFIXES.items():
        if (chain_id or '').startswith(template.split('{', 1)[0]):
            return key.split('_', 1)[0]
    raise ValueError('chain_id %r matches no CHAIN_PREFIXES entry' % (chain_id,))


# ── DELTA METADATA SHAPE ──
# Agentic encoders (S1E, S2 community, S2 consolidation, S2 healer) all
# have the same structural shape: an LLM loop that processes inputs, runs
# N rounds, produces write actions, writes a journal entry, and may record
# rejection fingerprints. One schema, unit-specific vocab in `outcomes`.

DELTA_METADATA_SHAPE = {
    'actions':           int,     # total tool calls
    'write_actions':     int,     # successful writes to the graph
    'rounds':            int,     # LLM conversation rounds
    'inputs_processed':  int,     # clusters / proposals / nodes seen
    'outcomes':          dict,    # unit-specific vocab: {action_name: count}
    'rejection_skipped': int,     # fingerprints recorded this run
    'journal_entry':     str,     # THIS RUN's journal contribution (extracted)
    'action_details':    list,    # per-action records (truncated if huge)
    'final_text':        str,     # raw agent text, first 2KB
    'errors':            list,    # first 5 errors (incl. refused ops left unrepaired)
    # Every write op the run's calls refused (a batch answers ok=True with
    # ok=False op rows), in order, each marked `repaired` when a later action
    # in the run wrote the same node. Lets an eval tell "did not revise" from
    # "tried, was refused, fixed it" from "tried and lost the intent".
    'refused_ops':       list,
    # Op-attributed node-id lists — the structured Δ that S2 reads (community
    # detection, consolidation catalog-blindness). Authoritative split returned
    # by the dispatch write handlers (`affected`), aggregated here over
    # action_details. NODE lifecycle only — edges are NOT in this delta.
    #
    # Edges: every FIRST-CLASS typed edge — explicit connect / connect_batch /
    # revise_edge / disconnect / connect_to / co_anchored — is its own
    # directional `edge_relation_revised` event carrying
    # source_id/target_id/relation, so a flat directionless `connected` node-id
    # list (a two-sided-era vestige that couldn't represent a v22
    # single-direction edge) was removed. SOFT/derived edges (the noise
    # aspect's structural relations, e.g. co_anchored) are intentionally NOT
    # traced — they're recomputable and excluded from the graph views
    # (dashboard, S2 decisions). So "reconstruct the graph from traces" means
    # the first-class typed graph, not the soft layer.
    'created':           list,    # node ids created this run
    'revised':           list,    # node ids revised this run (incl. absorb survivors — content rewritten)
    'archived':          list,    # node ids archived this run (incl. absorb's folded-in originals)
    # AspectIntegration's structured Δ. Aspect mutates aspects_v1.json, not the
    # graph, so created/revised/archived don't apply — the real change record is
    # WHICH string routed to WHICH aspect(s): [{category, value, aspects}, ...].
    # First-class (validated + dashboard-known + capped) rather than smuggled
    # through **extras. Empty [] for every non-aspect delta.
    'classifications':   list,
    # Cost & provenance of producing this Δ. elapsed_ms + token counts let you
    # trend encoder latency/cost over time — and, paired with the K block,
    # compare cost across prompt versions — straight from traces. truncated
    # flags silent data loss (a max_tokens cut mid tool-call corrupts the
    # write). The K block records WHICH prompt+config produced this Δ:
    # interaction_fingerprint is the content-address of the EFFECTIVE value
    # (stable across installs; always stamped — the builder requires it),
    # interaction_source is where it came from ('default' = code, 'override' =
    # DB row, 'caller' = a caller's config laid over either, which no
    # registered version holds), interaction_version the override version (0 on
    # a default run — meaningful only next to source). interaction_id (the
    # override rowid, on the trace row itself) is install-local, display-only.
    'elapsed_ms':            int,
    'input_tokens':          int,
    'output_tokens':         int,
    'cache_read_tokens':     int,
    'cache_creation_tokens': int,
    'truncated':             int,
    # The LLM that produced this Δ (the unit's resolved config model, read off
    # the runner's result) — same key and meaning as the S0 session stamp: the
    # model behind the row. Makes the token counts priceable and a model A/B
    # attributable per run without decoding the K fingerprint. '' = unstamped.
    'model':                 str,
    'interaction_version':   int,
    'interaction_fingerprint': str,
    'interaction_source':      str,
}

DELTA_FINAL_TEXT_LIMIT = 2000
DELTA_ERROR_LIST_LIMIT = 5
DELTA_CLASSIFICATIONS_LIMIT = 200  # cap aspect's per-item Δ (cold-start runs can be large)
DELTA_REFUSED_OPS_LIMIT = 50       # refused_ops entries kept per run
DELTA_REFUSAL_TEXT_LIMIT = 600     # each refusal's error text, in refused_ops and errors


# ── AGENT-RUN TELEMETRY (the shared cost+loop field-set) ──
# The cost of producing ONE agent run — an encoder Δ OR a Surface selection.
# Every agent that drives an LLM loop spends the same currency: wall-clock,
# rounds, output truncation, and the four token counts. Defined ONCE here so
# the encoder delta (build_delta_metadata) and the Surface K trace build their
# cost block through the same builder and can never drift into two field-sets.
#
# Kept FLAT on purpose (not a nested sub-object): the dashboard cost lane, the
# loud telemetry guards, and DELTA_METADATA_SHAPE already read these as
# top-level keys, so flat = zero consumer migration. This unifies the
# DEFINITION (one builder) the way runner.USAGE_FIELDS unified the SDK attribute
# NAMES — two different concerns, each single-sourced. (USAGE_FIELDS stays in
# runner.py next to read_usage, the SDK mapper; this is the trace-payload set.)
RUN_TELEMETRY_FIELDS = (
    'elapsed_ms', 'rounds', 'truncated',
    'input_tokens', 'output_tokens',
    'cache_read_tokens', 'cache_creation_tokens',
    'model',   # the only str: which LLM the run called (see DELTA_METADATA_SHAPE)
)


# ── TRACE DETAIL MODES ──
# How extensive trace recording is — discrete MODES, not a per-field dial:
# either we're debugging (open everything up) or we're not (record the bare
# minimum, so the trace store never grows from routine operation). Consumers
# call trace_detail() and read caps off the active mode; nobody hardcodes a
# number. All values are CHARACTER caps.
#
# `tool_result_cap`: a single formatted tool result larger than this is
#   truncated before it enters the LLM conversation (the 1M-token 400 killer:
#   one brain_batch result hit ~6M chars, 2026-07-31). Forensics ride the
#   trace substrate (result_chars + result_head on the action record); full
#   capture is the payload recorder below (`record_payload` — `failed_run`
#   and the per-round `round_payload` kinds are wired; the truncated result
#   appears verbatim inside the round_payload msgs, so a separate
#   `tool_result` capture site remains future work).
# `failed_action_input_cap`: per-action `input` head salvaged onto an
#   encoding_run_failed trace — bounded even when the run died precisely
#   because something was enormous.
# `result_head_cap`: head of an oversized tool result preserved inline so
#   the trace shows WHAT the content was.
TRACE_MODES = {
    'normal': {
        'tool_result_cap': 200_000,
        'failed_action_input_cap': 2_000,
        'result_head_cap': 500,
    },
    'debug': {
        # tool_result_cap is deliberately mode-INVARIANT: it bounds what the
        # LLM SEES, so a debug value would make debug runs a different
        # conversation than the runs they reproduce. Capture is observation-
        # neutral — debug widens what's recorded, never what the model reads.
        'tool_result_cap': 200_000,
        'failed_action_input_cap': 50_000,
        'result_head_cap': 10_000,
    },
}


def trace_detail():
    """Caps for the ACTIVE trace mode. BRAIN_TRACE_MODE env selects
    ('normal' default; unknown values fall back to normal — never crash a
    write path over a config typo). The single door consumers read through,
    so switching modes never touches consumer code."""
    import os
    return TRACE_MODES.get(os.environ.get('BRAIN_TRACE_MODE', 'normal'),
                           TRACE_MODES['normal'])


# ── Payload recording contract (docs/TRACE-MODES-DESIGN.md) ─────────────────
# Fat payloads live in FILES under {db_dir}/payloads/{date}/{chain_id}/, never
# in trace rows. `kind` is the gate-config key verbatim AND the filename
# segment — a kind that needs two formats is two kinds, so the extension is
# fixed here. brain.record_payload / brain.read_payload (brain_traces.py) are
# the only writer/reader; sanctioned direct-file readers: the dashboard
# (daemon-down debuggability) and the eval harness (daemonless corpus brains).
PAYLOAD_KIND_EXT = {
    'prompt': 'md',           # assembled agent prompt, readable as a document
    'judge': 'json',          # surface/judge output (S1R chain)
    'round_payload': 'json',  # full per-round request payload (eval capture)
    'tool_result': 'txt',     # full untruncated tool result
    'failed_run': 'json',     # full msgs at RunLoopError time — the 2AM story
}

# Named shapes for the `trace_recording` K-store interaction — modes as config
# versions, not an env dial. Fresh brains seed NORMAL as v1 and DEBUG as v2,
# both dormant — the resolver serves NORMAL as the code default; "entering
# debug" = set_interaction_active('trace_recording', 2), one MCP call, no
# restart, and clear_interaction_override reverts it. Capture is observation-neutral: these gates change what's RECORDED,
# never what the model sees (that's tool_result_cap, mode-invariant above).
# Both shapes derive from PAYLOAD_KIND_EXT so a new kind can never be
# silently absent from one of them (the recorder additionally overlays the
# active config onto these defaults, so kinds added after a brain's config
# was seeded still resolve — see brain_traces._payload_kind_enabled).
# NOTE: `round_payload` is wired (run_llm_loop's record_round_fn →
# brain.round_recorder); `tool_result` has no call site yet — flipping debug
# records nothing for that kind until one is wired.
_NORMAL_ON_KINDS = ('prompt', 'judge', 'failed_run')
TRACE_RECORDING_NORMAL = {
    'kinds': {k: k in _NORMAL_ON_KINDS for k in PAYLOAD_KIND_EXT},
    'retention_days': 14,
}
TRACE_RECORDING_DEBUG = {
    'kinds': {k: True for k in PAYLOAD_KIND_EXT},
    'retention_days': 14,
}

# Cap on the error string inside a `failed_run` payload file (the payload
# SHAPE {'error', 'messages'} is owned by brain.record_failed_run — consumers
# never build it). Distinct from build_failed_run_metadata's 500-char trace-
# row cap: the file can afford more context.
FAILED_RUN_ERROR_CAP = 2_000


def build_round_payload(*, label, round_idx, seq, model, effort, system,
                        messages, tools):
    """The `round_payload` payload shape — one full per-round LLM request
    exactly as issued (system + messages + tool names). Owned here so the
    eval harness's body-parsing checks (ab_encode soft checks) and the
    recorder stay pinned to one dict; keys match the retired
    BRAIN_PROMPT_CAPTURE_DIR dump (`label, round, seq, model, effort,
    system, messages, tools`) so existing parsers port unchanged.
    `label` is the chain_id (the old arm__session__stop label died with the
    filename-keyed capture dir); `seq` is the file seq — for multi-batch
    encoders it carries the batch offset (round_recorder's seq_base), so a
    payload file self-identifies which batch produced it."""
    return {
        'label': label,
        'round': round_idx,
        'seq': seq,
        'model': model,
        'effort': effort,              # None = the model's API default
        'system': system,              # full text, not a length
        'messages': messages,          # full, every content block
        'tools': tools,                # names only
    }


def interaction_fields(stamp, prefix=''):
    """The K block for one interaction stamp — {prefix}interaction_version /
    _fingerprint / _source. The builders write the run's own stamp through it;
    a run that also depends on a second interaction (S1E's gist, community's
    decoder config) adds that one with a prefix, as extras. A missing stamp or
    fingerprint is a caller bug and fails here, named, rather than recording
    an unattributed run."""
    if not isinstance(stamp, dict) or not stamp.get('fingerprint'):
        raise ValueError('interaction stamp with a fingerprint required, got %r'
                         % (stamp,))
    return {
        prefix + 'interaction_version':     int(stamp.get('version') or 0),
        prefix + 'interaction_fingerprint': str(stamp['fingerprint']),
        prefix + 'interaction_source':      str(stamp.get('source') or ''),
    }


def build_failed_run_metadata(*, error, stop_counter, inputs_processed,
                              interaction, partial_actions=None,
                              payload_pointer=None):
    """Metadata payload for an `encoding_run_failed` delta — the failure-path
    sibling of build_delta_metadata. Owns the bounding of salvaged action
    records (RunLoopError.partial_actions) so consumers hand over raw actions
    and never touch the record structure. Bounded by the active trace mode:
    a failed-run trace row must stay small even when the run died precisely
    because something was enormous."""
    import json as _json
    d = trace_detail()
    in_cap = d['failed_action_input_cap']
    partial = []
    for a in (partial_actions or []):
        inp = a.get('input')
        partial.append({
            'tool': a.get('tool'),
            'ops': len((inp or {}).get('operations', []))
                   if isinstance(inp, dict) else 0,
            'input_head': _json.dumps(inp)[:in_cap] if inp is not None else '',
            'result_chars': a.get('result_chars'),
            'result_head': (a.get('result_head') or '')
                           if a.get('result_truncated') else '',
            'error': a.get('error'),
        })
    md = {'error': str(error)[:500], 'stop_counter': stop_counter,
          'inputs_processed': inputs_processed,
          'partial_actions': partial,
          # A failed run still names the K it was running.
          **interaction_fields(interaction)}
    if payload_pointer:
        # Relative pointer to the failed_run payload file (the full msgs at
        # failure time) — enrichment only; the bounded forensics above stay
        # self-sufficient when the file is pruned.
        md['payload_pointer'] = payload_pointer
    return md


def build_run_telemetry(*, elapsed_ms=0, rounds=0, truncated=0,
                        input_tokens=0, output_tokens=0,
                        cache_read_tokens=0, cache_creation_tokens=0,
                        model=''):
    """Build the shared agent-run cost block (a flat dict of RUN_TELEMETRY_FIELDS).

    Used by build_delta_metadata (encoders) and the Surface K-trace writer.
    Counts are int, default 0 — `truncated` is a count of rounds cut at
    max_tokens, `rounds` the number of LLM calls, the rest wall-clock + token
    spend. `model` is the LLM the run called ('' when unknown) — the runner
    returns it next to the usage so callers thread it like the token counts.
    Spread flat into the surrounding metadata dict; never nest it.
    """
    return {
        'elapsed_ms':            int(elapsed_ms or 0),
        'rounds':                int(rounds or 0),
        'truncated':             int(truncated or 0),
        'input_tokens':          int(input_tokens or 0),
        'output_tokens':         int(output_tokens or 0),
        'cache_read_tokens':     int(cache_read_tokens or 0),
        'cache_creation_tokens': int(cache_creation_tokens or 0),
        'model':                 str(model or ''),
    }


def build_delta_metadata(*,
                         actions=0, write_actions=0, rounds=0,
                         inputs_processed=0, outcomes=None,
                         rejection_skipped=0, journal_entry='',
                         action_details=None, read_calls=None,
                         final_text='',
                         errors=None,
                         created=None, revised=None, archived=None,
                         classifications=None,
                         elapsed_ms=0, input_tokens=0, output_tokens=0,
                         cache_read_tokens=0, cache_creation_tokens=0,
                         truncated=0, model='', interaction,
                         **extras):
    """Build a unified delta trace metadata dict.

    `interaction` is required: the K-provenance stamp of the prompt+config
    this run resolved (brain.get_interaction_stamp / get_interaction_effective
    ['stamp'] — {'fingerprint', 'source', 'version', 'id'}). Every agent's run
    delta records which K produced it; a new encoder can't forget it, because
    the call fails.

    All agentic encoders (S1E, S2 units) should call this to build the
    metadata payload for their `delta` trace event. Standardizes field
    names, applies truncation, and lets each unit pass additional keys
    via **extras (e.g. clusters_processed, batches).

    read_calls captures non-write tool invocations (recall_batch, get_nodes,
    etc.). Useful for observability — answering "what did the encoder ask
    for that the catalog didn't already give it?" without parsing logs.

    created/revised/archived default to an aggregation over action_details
    (each write action carries its own op-attributed split — the `affected`
    dict the dispatch handler returned, copied onto the action by the runner).
    Pass them explicitly only to override. This is the structured Δ S2 reads
    — `revised` includes absorb survivors and `archived` their folded-in
    originals, so a merge-only consolidation run is no longer invisible. Edges
    are out of scope here (see the shape comment) — they live in directional
    `edge_relation_revised` events.

    Returns a dict ready to pass as the metadata kwarg to a trace writer.
    """
    ad = list(action_details or [])

    def _agg(key, explicit):
        if explicit is not None:
            return list(explicit)
        out = []
        for a in ad:
            if isinstance(a, dict):
                out.extend(a.get(key) or [])
        return out

    def _cap(items, limit):
        # Loud-in-data truncation for a dict-list (can't append a string marker
        # like cap_list_loud): keep `limit`, append a sentinel naming the drop.
        items = list(items or [])
        if len(items) <= limit:
            return items
        return items[:limit] + [{'_truncated': len(items) - limit}]

    # A refused op is repaired when a LATER action in the run wrote the same
    # node — the retry that landed. Only later actions count: an earlier or
    # same-call write to that node is not a response to the refusal. An op
    # that names no node (a refused remember, a malformed call) cannot be
    # matched to a retry and stays unrepaired.
    written_after = [set() for _ in ad]       # node ids written by ad[i+1:]
    for i in range(len(ad) - 2, -1, -1):
        b = ad[i + 1]
        written_after[i] = written_after[i + 1] | (
            {str(x)[:8] for key in ('created', 'revised', 'archived')
             for x in (b.get(key) or [])} if isinstance(b, dict) else set())
    refused = []
    for i, a in enumerate(ad):
        if not isinstance(a, dict):
            continue
        for f in a.get('failed_ops') or []:
            nid = str(f.get('node_id') or '')
            refused.append({
                'tool': a.get('tool', ''), 'op': f.get('op') or '',
                'index': f.get('index'), 'node_id': nid,
                'error': cap_text_loud(str(f.get('error') or ''),
                                       DELTA_REFUSAL_TEXT_LIMIT),
                'repaired': bool(nid) and nid[:8] in written_after[i],
            })
    # From the kept entries only, so every error line has its refused_ops row.
    lost = ['%s %s refused, not repaired: %s' % (r['op'], r['node_id'][:8] or '(no node)',
                                                 r['error'])
            for r in refused[:DELTA_REFUSED_OPS_LIMIT] if not r['repaired']]

    metadata = {
        'actions':           int(actions or 0),
        'write_actions':     int(write_actions or 0),
        # 'rounds' is emitted by build_run_telemetry below (shared cost block).
        'inputs_processed':  int(inputs_processed or 0),
        'outcomes':          dict(outcomes or {}),
        'rejection_skipped': int(rejection_skipped or 0),
        'journal_entry':     cap_text_loud(journal_entry, DELTA_FINAL_TEXT_LIMIT),
        'action_details':    ad,
        'read_calls':        list(read_calls or []),
        'final_text':        cap_text_loud(final_text, DELTA_FINAL_TEXT_LIMIT),
        'errors':            cap_list_loud(list(errors or []) + lost,
                                           DELTA_ERROR_LIST_LIMIT),
        'refused_ops':       _cap(refused, DELTA_REFUSED_OPS_LIMIT),
        'created':           _agg('created', created),
        'revised':           _agg('revised', revised),
        'archived':          _agg('archived', archived),
        'classifications':   _cap(classifications, DELTA_CLASSIFICATIONS_LIMIT),
        # Shared cost+loop block (elapsed_ms/rounds/truncated + token counts) —
        # one builder so the encoder Δ and the Surface K trace can't drift.
        **build_run_telemetry(
            elapsed_ms=elapsed_ms, rounds=rounds, truncated=truncated,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens, model=model),
        **interaction_fields(interaction),
    }
    # Extras preserved for per-unit fields (can't collide with shared keys).
    for k, v in extras.items():
        if k not in metadata:
            metadata[k] = v
    return metadata


# ── ANCHOR-TOUCHED METADATA SHAPE ──
# The S0 mirror of the S1 encoding_run delta: one per-turn aggregate of the
# nodes Anchor's OWN MCP tools touched this turn. Deliberately reuses the encode
# delta's `created`/`revised`/`archived` keys so trace_links reads BOTH the S1
# encode delta and this S0 delta through ONE parser (no second parsing path) —
# plus read-side keys S1 has no analog for (`recalled` = deliberate get_node(s);
# `endo` = endo-surface ids, empty until endo lands). NODE ids only (full),
# never edges or the encoding_source field (technical, encoder-invisible).
# `recalled` = deliberate by-id reads, folded into the encoder catalog as full
# bodies. `looked_up` = SEARCH-tool results, rendered on the encoder's
# <provenance> line but never catalog-folded: a recall returns whole result
# pages, and folding those would flood the catalog the aging work just cut.
# WHICH tools land in which of those two keys is owned by
# daemon_server.BrainDaemon._LOOKUP_KEY at the dispatch chokepoint and is
# deliberately not re-listed here — no test reads this comment.
ANCHOR_TOUCHED_KEYS = ('created', 'revised', 'archived', 'recalled',
                       'looked_up', 'endo')
# Shape for validate_trace_metadata — every key an id-list. Derived from the
# constant (not a second hand-written list) so the keyset can't drift.
ANCHOR_TOUCHED_SHAPE = {k: list for k in ANCHOR_TOUCHED_KEYS}


def build_anchor_touched_metadata(**ids):
    """Build the anchor_touched delta metadata — a flat dict of id-lists, ONE per
    ANCHOR_TOUCHED_KEYS (the constant drives the keyset; pass any subset). Order-
    preserving dedup per key; never the raw encoding_source. Reuses the encode
    delta's `created`/`revised`/`archived` key names so one reader serves both.
    Unknown kwargs are ignored (the flush passes `**ctx.touched`, which is itself
    keyed by ANCHOR_TOUCHED_KEYS — so they always agree)."""
    return {k: list(dict.fromkeys(ids.get(k) or [])) for k in ANCHOR_TOUCHED_KEYS}


# ── TOOL RESULT METADATA SHAPE ──
# The S0 row the PostToolUse hook writes on every tool call (~2500/day). The
# hook is a bare socket send and records the host's RAW tool name under `tool`
# (post_tool_trace); everything host-neutral about the call — its kind, how
# the host was identified, which vocabulary did the translating — is stamped
# by the daemon at the S0 write door (docs/HOST-CONTRACT-DESIGN.md D9: the
# first consumer of a kind is the encoder, hours later, so classification runs
# daemon-side, restart-deployable, never in the hook). Every new tool row is
# stamped there, even from old clients or without a session id, so the full
# normalization shape is required at the DAL write chokepoint.
TOOL_RESULT_METADATA_SHAPE = {
    'tool': str,   # the host's raw tool name, verbatim (redacted input, never the caller stamp)
    'kind': str,
    'kind_status': str,
    'host_status': str,
    'tells': list,
    'vocab_version': int,
    'impl_identity': str,
    'tool_use_id': str,
    'turn_id': str,
    'payload_keys': list,
}
# Optional capture facts remain extras, not required normalization stamps:
# git_invocation=True records a positively identified literal call;
# capture_filter_incomplete=True announces an oversized unavailable command.
# Required on new writes: even old clients and sessionless tool rows pass the
# daemon's stamper. Historical rows are never normalized on read (D10).
TOOL_RESULT_NORMALIZATION_KEYS = (
    'kind',            # one of ACTION_KINDS, '' when unknown
    'kind_status',     # one of KIND_STATUS
    'host_status',     # one of HOST_STATUS
    'tells',           # env tells that fired in the hook process (names only)
    'vocab_version',   # host_contract.VOCAB_VERSION that produced `kind`
    'impl_identity',   # host_contract.contract_fingerprint() at stamp time
    'tool_use_id',     # host's per-call id (join key for reconciliation)
    'turn_id',         # host's turn_id when present; prompt_id stays a separate raw extra
    'payload_keys',    # top-level stdin keys the hook saw — names only, no values
)


def build_tool_result_metadata(*, tool, **normalization):
    """Build a tool_result row's metadata: the raw `tool` name plus any subset
    of TOOL_RESULT_NORMALIZATION_KEYS. Unknown kwargs raise — the keyset is the
    contract, and a misspelled stamp key must fail at the writer, not land as
    an invisible extra. Keys passed as None are omitted, so a row from before a
    value was learned carries no key at all (the stamp_s0_session posture)."""
    unknown = set(normalization) - set(TOOL_RESULT_NORMALIZATION_KEYS)
    if unknown:
        raise ValueError('tool_result metadata: unknown keys %s (allowed: %s)'
                         % (sorted(unknown), list(TOOL_RESULT_NORMALIZATION_KEYS)))
    md = {'tool': str(tool or '')}
    for k in TOOL_RESULT_NORMALIZATION_KEYS:
        v = normalization.get(k)
        if v is None:
            continue
        md[k] = list(v) if isinstance(v, (tuple, set, frozenset)) else v
    return md


# ── JOURNAL NOTE METADATA SHAPE ──
# A journal note is a Δ written as its OWN trace event (event_type='delta',
# ref_type='journal_note'): the residue of a run's integrate() — the why, the
# friction, the doubt, the surprise. SEPARATE from the run's objective ops-delta
# (encoding_run / consolidated / ...): the ops-delta records what the hands did,
# the note records what the mind did. Per run = 1 ops-delta + 0..N notes, ALL
# sharing the run's chain_id — which is per-run-unique at BOTH scales (S1
# `s1e-{session}-{stop}`, S2 `s2-{YYYYMMDDHHMMSS}-{unit}`), so the read groups
# runs by chain_id with no separate run_id field.
#
# The SUBJECT lives in the trace's ref_id, NOT here — a node id / cluster / tool /
# input the note is about. It's the load-bearing index (N notes on one ref_id =
# a hotspot) and the quality gate (can't name a subject → not a note). Notes are
# s1/s2 scale → never embedded (EAGER_TRACE_SCALES=('s0',)) → unreachable by
# recall()/recall_episodes(); the only door is the traces-module notes() query.

JOURNAL_NOTE_METADATA_SHAPE = {
    'note': str,    # the prose: the why / friction / doubt / surprise (required)
    'tag':  str,    # one open word for the KIND of thing (friction, doubt, ...); '' when absent
    'undelivered': str,  # an addressed line (tell/ask) the Thalamus door rejected,
                         # kept as residue: the door's reason. '' for a plain note.
                         # A field, not prose appended to `note` — the note stays
                         # the line the encoder wrote, and the reason cannot be
                         # eaten by the note cap.
}

JOURNAL_NOTE_LIMIT = 600   # a note is terse residue, not an essay — capped loud like other delta text
JOURNAL_TAG_LIMIT = 40     # 'one word' — cap drift loud rather than let a sentence become a grouping key


# ═══════════════════════════════════════════════════════
# THALAMUS_FILED metadata — a producer's filing, on the producer's run chain
# ═══════════════════════════════════════════════════════
# ref_id is the item id; the row is the filed→delivered→answered join's first
# link. Door vocabulary only — the traces layer knows nothing of the journal
# grammar that produced the filing (subject = dedup_key; ask/notice falls out
# of needs_answer). `body` is copied so the row outlives a swept item; a
# dedup re-file rewrites the item's body while earlier rows keep theirs
# (each row is what THAT run said). `route` is 'queue' or 'live': a live item
# is courier-delivered and never yields a thalamus_delivery row, so a
# filed→delivered join filters route='queue'. `filing` names what the door
# did: 'new' (inserted), 'refresh' (identical re-file, window only),
# 'rearm' (changed re-file, delivers again).
THALAMUS_FILED_METADATA_SHAPE = {
    'source':         str,   # the producer's encoding_source
    'body':           str,   # the item body, capped loud
    'target_session': str,   # '' for broadcast, the session UUID when directed
    'needs_answer':   bool,  # ask (True) vs notice/reminder (False)
    'dedup_key':      str,   # producer-owned identity, '' when none
    'route':          str,   # 'queue' | 'live'
    'filing':         str,   # 'new' | 'refresh' | 'rearm'
}
THALAMUS_FILED_BODY_LIMIT = 1500  # mirrors the delivery render's per-item body cap
THALAMUS_FILINGS = ('new', 'refresh', 'rearm')


def build_thalamus_filed_metadata(*, source, body, target_session='',
                                  needs_answer=False, dedup_key='',
                                  route='queue', filing='new'):
    """Build metadata for one thalamus_filed row. Raises ValueError on an
    unknown `filing` — the three values are the only states the door's dedup
    logic can produce, and a fourth would be a producer bug."""
    if filing not in THALAMUS_FILINGS:
        raise ValueError('thalamus_filed: filing=%r not in %s'
                         % (filing, THALAMUS_FILINGS))
    return {
        'source': source or '',
        'body': cap_text_loud(body or '', THALAMUS_FILED_BODY_LIMIT),
        'target_session': target_session or '',
        'needs_answer': bool(needs_answer),
        'dedup_key': dedup_key or '',
        'route': route or 'queue',
        'filing': filing,
    }


def build_journal_note_metadata(*, note, tag='', undelivered=''):
    """Build trace metadata for one journal note (ref_type='journal_note').

    The SUBJECT is the trace's ref_id, supplied by the writer — not here.
    `note` is the prose (required, non-empty — the same gate the parser
    applies); `tag` is one open word, '' when the encoder gave none. Both are
    capped loud via cap_text_loud, like every other delta text field. Raises
    ValueError on an empty note so the builder and parser AGREE on validity —
    the write path only feeds parser-validated notes and isolates per-note
    write errors, so this fires solely on direct misuse, never normal flow.
    """
    note = (note or '').strip()
    if not note:
        raise ValueError('journal note requires non-empty prose (got empty note)')
    return {
        'note': cap_text_loud(note, JOURNAL_NOTE_LIMIT),
        'tag':  cap_text_loud((tag or '').strip(), JOURNAL_TAG_LIMIT),
        'undelivered': cap_text_loud((undelivered or '').strip(),
                                     JOURNAL_NOTE_LIMIT),
    }


# ── JOURNAL REVIEW BLOCK + PARSER (single source for all journaling encoders) ──
# §7.1/§7.3: one shared instruction block (roles-free — a bar to clear, not
# buckets to fill); each encoder appends ONLY its own examples + subject
# vocabulary via render_journal_review_block(). The encoder emits one note per
# line as `tag · subject · note`; the write path calls parse_journal_notes() to
# split them into rows. Single source here so the five encoders can't re-diverge
# into the five reinventions this redesign is removing.

JOURNAL_NOTE_DELIMITER = '·'

# ── Journal lifecycle verbs (2026-07-28, audit finding #6) ──
# Read-time only — traces stay append-only. `resolved`/`retire` drops older
# same-subject notes from the continuity prefix; `open` pins the newest note
# per subject beyond the K-run window until resolved. Matching is normalized
# (casefold+strip) subject equality — the corpus showed paraphrase references
# never match. Encoders reference a prior note by echoing its rendered head,
# so the slot they fill is recovered at read time by `resolve_target` rather
# than demanded of the prompt (an instruction costs encoder attention; a
# tolerant reader costs nothing).
JOURNAL_RESOLVE_TAGS = ('resolved', 'retire')
JOURNAL_OPEN_TAGS = ('open', 'still-open')   # still-open: pre-existing wild alias
# Verbs whose payload is (tag, subject) — the trailing `why` is optional, so a
# two-field line is a complete lifecycle note rather than a malformed one.
JOURNAL_LIFECYCLE_TAGS = JOURNAL_RESOLVE_TAGS + JOURNAL_OPEN_TAGS
# ── Addressed verbs ──
# Notes written to the LIVE SESSION, not to the next run: `tell` (a notice)
# and `ask` (needs an answer). Same `tag · subject · note` line, same parser;
# the write door hands them back to a binding that has a source, which files
# each as a Thalamus item — directed to its session when it has one (the
# Scribe, delivered at Stop), broadcast when it has none (an S2 unit,
# delivered at boot). A binding without a source writes them as plain notes
# and warns — no reader exists for them there.
JOURNAL_TELL_TAG = 'tell'
JOURNAL_ASK_TAG = 'ask'
JOURNAL_ADDRESSED_TAGS = (JOURNAL_TELL_TAG, JOURNAL_ASK_TAG)
JOURNAL_RUN_SUBJECT = 'run'   # the subject a two-field addressed line gets —
                              # "the run itself", the instruction's third kind


def journal_key(value):
    """The comparison form of a journal tag or subject — stripped and
    casefolded. Every match against the JOURNAL_*_TAGS vocabulary and every
    subject-equality test (parser, read door, resolve targets, dedup and
    withdraw keys) goes through this one normalizer, so no two doors can
    disagree on what "the same subject" means."""
    return (value or '').strip().casefold()


def journal_anchor_refs(subject):
    """What an addressed line (tell/ask) anchors on: the node ids its subject
    EMBEDS ('d4ed89c5', '37c548a7_vs_21a88035') — once each, in order. A
    tool, an input or the run itself anchors nothing. Returns a list
    (possibly empty)."""
    from servers.contract import find_node_ids
    return list(dict.fromkeys(find_node_ids(journal_key(subject))))
JOURNAL_OPEN_PIN_CAP = 10        # max pinned subjects carried beyond the window
JOURNAL_OPEN_NUDGE_RUNS = 5      # open ×N at/past this → render the hand-it-up nudge

# Working continuity, shared by every modern encoder. Storage and hotspot
# history are unaffected. Limits bound context and per-request catch-up work.
JOURNAL_VIEW_MAX_CHARS = 8000
JOURNAL_VIEW_HISTORY_LIMIT = 200
JOURNAL_VIEW_PAGE_SIZE = 200
JOURNAL_VIEW_MAX_PAGES = 10
JOURNAL_VIEW_VERSION = 1

# Self-grounding by design (no `brain`/`trace`/`operator`/agent-verb/identity
# tokens): the block means the same dropped into any host prompt or standing
# alone, so a host-prompt edit can't silently shift the journal, and the block
# is testable in isolation. EAGER by intent: no value-filter
# gate — capture residue freely; dedupe/mine later. The earlier "two tests"
# (reconstruction/successor) were removed as over-correction against the OLD
# journal's restatement disease, not an evidenced need. Iterate from LIVE
# results, not synthetic probes (which can't reproduce the encoder's lived run).
_REVIEW_HEAD = (
    "A review — a short note to the next run of this work, about anything "
    "noticed here that won't be visible in the actions taken.\n"
    "The changes made are already recorded automatically; don't restate them. "
    "This note is only for what the actions don't capture — a doubt, a "
    "friction, a surprise, a pattern forming.\n\n"
    "`tag` — one word for the kind of thing (friction, doubt, surprise, "
    "dead-end — examples, not a list).\n"
    "`subject` — what the note is about: the specific thing touched (its id), "
    "a tool or input handed in, or the run itself.\n\n"
    "To clear a handled note, write `resolved %s <its exact subject> %s why` "
    "— one line per subject.\n"
    "Mark a persisting item once: `open %s subject %s note` — it stays "
    "visible until resolved; don't repeat existing items, they carry over "
    "automatically.\n\n"
) % ((JOURNAL_NOTE_DELIMITER,) * 4)

_REVIEW_TAIL = (
    "Put the notes under a `## Review` heading, inside a fenced code block — "
    "one note per line as `tag %s subject %s note`. A clean run is an empty "
    "fence — leave it empty rather than saying there's nothing to note.\n\n"
    "Time is precious — actions are already logged automatically; no need "
    "to rephrase. Stay sharp."
) % ((JOURNAL_NOTE_DELIMITER,) * 2)

# ── The addressed verbs, as the encoder reads them ──
# One text for every encoder (same instructions; delivery differs by
# audience — the door does the routing). Sits between the `open` line and
# the output-format close. The flag is the one switch for every journaling
# encoder at once: off, and the block is residue-only again — the exit if
# the measurement says noise.
JOURNAL_ADDRESSED_LIVE = True
JOURNAL_ADDRESSED_INSTRUCTION = (
    "Two notes go to the live work, not to your next run:\n"
    "`%(tell)s %(d)s subject %(d)s note` — the \"wait, one thing\" that "
    "surfaces while you encode and bears on what they're doing now.\n"
    "`%(ask)s %(d)s subject %(d)s note` — the \"what about…?\" only they can "
    "settle.\n"
    "Interrupt only when it touches the present work, would change it, and "
    "is worth the stop; otherwise it's a plain note.\n"
    "Plain words, for a reader with none of your context. One line per "
    "subject — sending the same subject again updates that message, no "
    "subject means the run "
    "itself, `resolved %(d)s subject %(d)s why` withdraws it. Next run, "
    "YOUR MESSAGES shows how each ended.\n\n"
) % {'tell': JOURNAL_TELL_TAG, 'ask': JOURNAL_ASK_TAG,
     'd': JOURNAL_NOTE_DELIMITER}

# THE review block every encoder reads — one text, assembled once from the
# flag. Readers compare prompts against this constant (or the render, which
# returns it); the head/tail halves are assembly detail.
JOURNAL_REVIEW_INSTRUCTION = (
    _REVIEW_HEAD + (JOURNAL_ADDRESSED_INSTRUCTION if JOURNAL_ADDRESSED_LIVE
                    else '') + _REVIEW_TAIL)


def is_addressed(tag):
    """True when a note's tag is one of the addressed verbs — a Thalamus
    item, not a journal row. The one predicate the write door, the eval
    harnesses and the binding share."""
    return journal_key(tag) in JOURNAL_ADDRESSED_TAGS


def render_journal_review_block():
    """The shared review block — self-contained (output structure + close folded
    in), identical for every encoder: JOURNAL_REVIEW_INSTRUCTION."""
    return JOURNAL_REVIEW_INSTRUCTION


# The arc — the SECOND closing act (§7.2: Encode → Arc → Review), a journal-
# mechanism component distinct from the review: the review is residue notes
# (traces, per-note rows); the arc is ONE line of session orientation
# (accumulated onto a running per-session digest that downstream readers rank
# and orient against). Never merged into the review — different shape,
# different reader. Self-grounding like the review block (no host-coupled
# tokens); placement is stated HERE, not in the closure, so the closure stays
# shared with encoders that never emit an arc. Per-encoder opt-in: injected
# only by encoders that write a session arc (S1 Scribe today).
JOURNAL_ARC_INSTRUCTION = (
    "The arc — ONE line: what progressed in this stretch of work, this run.\n"
    "It accumulates onto a running digest of the whole conversation, so write "
    "only the new movement — never a recap of what the digest already says.\n\n"
    "Put it under a `## Arc` heading, inside a fenced code block — a single "
    "line, on the same final reply as the review, just before it. If nothing "
    "meaningfully progressed, leave the fence empty.\n\n"
    "Example: `judge reliability crisis found — 85% timeout rate`"
)


def render_journal_arc_block():
    """The shared arc block — the write-side instruction for the session arc,
    identical for every encoder that opts in. Single-sourced here (never baked
    into a registered prompt) for the same reason as the review block: it
    iterates in one place and every opted-in encoder gets it live.
    """
    return JOURNAL_ARC_INSTRUCTION


def render_prompt_closure():
    """The run's CLOSURE — separate concern from the review block. Defines the
    terminal turn the way the runner does (a reply with no tool call IS the
    final one), places the review on it whether the encoder acted or not, and
    carries the `DONE` stop signal. Injected as the LAST block of the prompt,
    independent of the review block — so removing or relocating the review never
    drags the closure with it. References the `## Review` artifact by name; it
    does NOT define it (that's render_journal_review_block's job).

    The no-tool-call branch is the fix for the no-action batch: an all-reject /
    nothing-to-change reply terminates the loop on its first turn, and that turn
    must still carry the review (an empty fence on a clean run).
    """
    return (
        "## Finishing\n\n"
        "The run is done when a reply makes no tool call — that final reply is the "
        "only place the review goes. Two ways to get there, both ending the same:\n"
        "- After tool calls: the run closes on the first reply that makes no tool call — "
        "a read's results are followed by the write; the write's results by the final reply.\n"
        "- A reply with no tool call at all (nothing needed changing): that reply is "
        "already the final one.\n\n"
        'End the final reply with the `## Review`, then write "DONE".'
    )


JOURNAL_NOTES_HEADER = ('%s — already recorded by your recent runs; they stay '
                        'here without being rewritten. Add a note only for '
                        'something new this run:')


def render_journal_notes_prefix(notes, label='RECENT REVIEW NOTES', *,
                                annotate_tag=True):
    """Render journal_notes() output into a prompt prefix — the READ side of
    the journal (residue continuity). Shared single source so every encoder
    (S2 units now, S1E later) feeds continuity the same way.

    `notes` is the list of {tag, subject, note, ...} dicts journal_notes()
    returns (newest first, already bounded to the last K note-bearing runs).
    Returns '' when there are none, so a clean history adds nothing to the
    prompt — no "first run, no notes" filler. Each line mirrors the write
    format `tag · subject · note`.
    """
    if not notes:
        return ''
    lines = [JOURNAL_NOTES_HEADER % label]
    lines.extend(_render_journal_note(n, annotate_tag=annotate_tag) for n in notes)
    return '\n'.join(lines) + '\n\n'


def _render_journal_note(n, *, annotate_tag):
    """Render one complete note, including persistence and delivery feedback."""
    tag = (n.get('tag') or '').strip()
    line = _journal_line(tag, n.get('subject', ''), n.get('note', ''))
    # Open items render their persistence: the loader computed ×N (distinct
    # runs mentioning the subject) and pins the newest note beyond the
    # window. Past the threshold, the nudge appears ON the item, in the run
    # that should act — zero standing prompt cost.
    runs = n.get('open_runs') or 0
    if runs:
        since = (n.get('first_seen') or '')[5:10]
        persistence = '×%d%s' % (runs, (' since %s' % since) if since else '')
        if annotate_tag:
            line = '- %s %s · %s · %s' % (
                tag or 'open', persistence,
                n.get('subject', ''), n.get('note', ''))
        else:
            # Keep the copyable lifecycle tag intact. Persistence is
            # read-side context, never a new encoder-authored tag.
            line += '\n  Persistence: %s' % persistence
    if n.get('undelivered'):
        # The line was addressed to the people working and the door
        # refused it — the reason is what the encoder reads next run.
        line += ' — not delivered: %s' % n['undelivered']
    if runs >= JOURNAL_OPEN_NUDGE_RUNS:
        # A note that has persisted this long is a question for the live
        # work, not residue — hand it up through the addressed verb; the
        # door delivers it, expires it, and carries the
        # answer back (YOUR MESSAGES).
        line += (
            "\n  ⚠ long-lived — resolve it, or hand it up: "
            "`%(ask)s %(d)s %(s)s %(d)s <the question>`, then "
            "`resolved %(d)s %(s)s %(d)s handed up` (the pin clears; "
            "the item carries it from here)"
            % {'ask': JOURNAL_ASK_TAG, 'd': JOURNAL_NOTE_DELIMITER,
               's': n.get('subject', '')})
    return line


def render_journal_view(view):
    """Bound working continuity by whole rows, with explicit omissions.

    Lifecycle state precedes ordinary observations. The trace history and
    selected state stay intact when a row cannot fit in the prompt.
    Returns text and telemetry shared by every encoder binding.
    """
    notes = view.get('notes', [])
    ordered = sorted(notes, key=lambda n: journal_key(n['tag'])
                     not in JOURNAL_LIFECYCLE_TAGS)
    coverage = []
    if view.get('history_truncated'):
        coverage.append('Older journal history beyond the newest %d events '
                        'is outside this view.' % JOURNAL_VIEW_HISTORY_LIMIT)
    if view.get('changes_pending'):
        coverage.append('Journal updates remain unread; this view is incomplete.')
    if view.get('stale'):
        coverage.append('Latest journal read failed; showing last known continuity.')
    if view.get('selection_failed'):
        coverage.append('Initial journal selection unavailable; private notes '
                        'resume next run.')
    if view.get('render_failed'):
        coverage.append('Journal rendering failed; private notes omitted.')

    def notice(omitted):
        parts = list(coverage)
        if omitted:
            parts.append('%d journal entries omitted from context; retained '
                         'in trace history.' % omitted)
        return ('\n'.join(parts) + '\n\n') if parts else ''

    text, rendered = '', 0
    # Reserve the largest omission notice before selecting complete rows.
    reserve = len(notice(len(notes)))
    for note in ordered:
        row = _render_journal_note(note, annotate_tag=False)
        prefix = text.rstrip('\n') if text else JOURNAL_NOTES_HEADER % 'RECENT REVIEW NOTES'
        candidate = prefix + '\n' + row + '\n\n'
        if len(candidate) + reserve <= JOURNAL_VIEW_MAX_CHARS:
            text = candidate
            rendered += 1
    omitted = len(notes) - rendered
    text += notice(omitted)
    return text, {
        'journal_view_version': JOURNAL_VIEW_VERSION,
        'selected_rows': len(notes), 'rendered_rows': rendered,
        'omitted_rows': omitted, 'residue_chars': len(text),
        'history_truncated': bool(view.get('history_truncated')),
        'changes_pending': bool(view.get('changes_pending')),
        'stale': bool(view.get('stale')),
        'selection_failed': bool(view.get('selection_failed')),
        'render_failed': bool(view.get('render_failed')),
    }


# ── Producer view: what the encoder told or asked, and how it ended ──
# The READ side of the addressed verbs, after the residue notes. MINIMAL:
# outcomes only — open / answered: <text> / dismissed / expired — never
# delivery counts, moments or dates (the encoder's job is its perspective
# slice, not managing its mail; delivery state is the Thalamus's). The
# binding does the join and hands plain rows {tag, subject, note, fate,
# answer}; the fate tokens are thalamus_contract.FATE_*, phrased here.
PRODUCER_VIEW_MAX = 10           # rows — loud overflow, never a silent cut
PRODUCER_VIEW_BLOCK_MAX = 2500   # chars — the block's own budget, like every
                                 # other injected block
PRODUCER_VIEW_NOTE_LIMIT = 300   # an item body, or an answer, is one line here
PRODUCER_VIEW_SUBJECT_LIMIT = 80  # a subject is a key, not prose
PRODUCER_VIEW_LABEL = ('YOUR MESSAGES — what you told or asked, and how it '
                       'ended (not a to-do list):')


def _journal_line(tag, subject, note):
    """The one line grammar both journal renders share: `- tag · subject ·
    note` (tag omitted when empty) — the mirror of the write format."""
    head = ('%s · ' % tag) if tag else ''
    return '- %s%s · %s' % (head, subject, note)


def _producer_view_line(r):
    """One row: {tag, subject, note, fate, answer} — the binding built every
    key. Note and answer are flattened and capped so a long or multi-line
    answer cannot forge rows or blow the budget."""
    from servers.channels.thalamus.thalamus_contract import FATE_ANSWERED
    fate = r['fate']
    if fate == FATE_ANSWERED:
        fate = 'answered: %s' % cap_text_loud(one_line(r['answer']),
                                              PRODUCER_VIEW_NOTE_LIMIT)
    # Read-only feedback deliberately differs from the writable review
    # grammar. Appending fate to `ask · subject · body` made encoders copy
    # "— open" into the body and re-arm an otherwise unchanged message.
    return '- %s [%s]\n  Message: %s\n  Status: %s' % (
        r['tag'], cap_text_loud(one_line(r['subject']),
                               PRODUCER_VIEW_SUBJECT_LIMIT),
        cap_text_loud(one_line(r['note']), PRODUCER_VIEW_NOTE_LIMIT), fate)


def render_producer_view(rows):
    """Render rows (already ordered open-first, newest-settled first) into
    the prompt block; '' for none — a producer that never spoke sees no new
    block. Two loud caps: row count and block chars; the tail names what it
    dropped."""
    if not rows:
        return ''
    body, kept, _ = compose_block_loud(
        rows[:PRODUCER_VIEW_MAX], _producer_view_line, PRODUCER_VIEW_BLOCK_MAX,
        reserved=len(PRODUCER_VIEW_LABEL) + 1, sep='\n')
    out = PRODUCER_VIEW_LABEL + '\n' + body
    if len(rows) > kept:
        out += '\n(+%d older, not shown)' % (len(rows) - kept)
    return out + '\n\n'


def parse_journal_notes(text):
    """Parse an encoder's review section into notes.

    One note per line: `tag · subject · note`, split on '·' with maxsplit=2 so
    a '·' inside the prose is safe (it all stays in `note`).
      • 3 fields → (tag, subject, note)
      • 2 fields → (subject, note) with tag='' — tag is optional
      • no delimiter, or empty subject/note → MALFORMED
    Blank lines and markdown headers (`#`-prefixed, e.g. the `## Review`
    header) are skipped silently. Malformed lines are NOT silently dropped —
    they're returned so the caller logs loud (loud-by-default).

    Returns (notes, malformed): notes is a list of {'tag','subject','note'};
    malformed is a list of the raw offending lines.
    """
    notes, malformed = [], []
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] in '-*•':       # tolerate a leading markdown bullet (LLMs
            line = line[1:].lstrip()   # list-format their review); keep the tag clean
            if not line:
                continue
        if JOURNAL_NOTE_DELIMITER not in line:
            # No delimiter: a markdown header (e.g. the `## Review` title) is
            # structural — skip silently. Anything else is a malformed note,
            # surfaced loud (never silently dropped). A delimiter-bearing line
            # is ALWAYS a note candidate even if it starts with '#', so a
            # subject like a `#1234` issue id isn't eaten by the header skip.
            if line.startswith('#'):
                continue
            malformed.append(raw)
            continue
        parts = [p.strip() for p in line.split(JOURNAL_NOTE_DELIMITER, 2)]
        if len(parts) == 3:
            tag, subject, note = parts
        elif journal_key(parts[0]) in JOURNAL_LIFECYCLE_TAGS:
            # `resolved · subject` — a lifecycle verb carries its payload in
            # (tag, subject) and the trailing `why` is optional. Without this
            # branch the two-field default below reads the VERB as the subject,
            # so the lifecycle action is lost and the line looks well-formed.
            tag, subject, note = parts[0], parts[1], ''
        elif journal_key(parts[0]) in JOURNAL_ADDRESSED_TAGS:
            # `tell · message` — an addressed verb with no subject is about
            # the run itself. Without this branch the message becomes a
            # residue note whose subject is the word "tell", never delivered.
            tag, subject, note = parts[0], JOURNAL_RUN_SUBJECT, parts[1]
        else:  # delimiter present + maxsplit=2 → exactly 2 parts here
            tag, subject, note = '', parts[0], parts[1]
        if not subject or (not note
                           and journal_key(tag) not in JOURNAL_LIFECYCLE_TAGS):
            malformed.append(raw)
            continue
        notes.append({'tag': tag, 'subject': subject, 'note': note})
    return notes, malformed


def resolve_target(subject, note, known_subjects):
    """The subject a `resolved`/`retire` note actually retires.

    Encoders reference a prior note by echoing its rendered `tag · subject ·
    note` head, which lands the old TAG in the subject slot and the real
    subject at the head of `note` (maxsplit=2 keeps it intact there). Recover
    it when that leading segment names a subject that exists — `known_subjects`
    is the guard, so a target is never invented.

    Falls back to the subject slot, so well-formed resolves are untouched.
    Recovering also REPLACES the tag-shaped subject rather than adding to it:
    otherwise a word like `friction` enters the retire set and silently drops
    an unrelated note that happens to use it as a subject.
    """
    lead = journal_key((note or '').split(JOURNAL_NOTE_DELIMITER, 1)[0])
    return lead if lead and lead in known_subjects else subject


JOURNAL_REVIEW_MARKER = '## Review'   # the section heading the encoder emits; the
                                      # write path keys on it. Kept in sync with the
                                      # prompt structure (§7.2) — #8 wires the prompt.

JOURNAL_ARC_MARKER = '## Arc'         # the arc heading (§7.2: "Arc — ONE line: what
                                      # progressed this run"). Its write path
                                      # (write_session_arc) keys on it. A journal-
                                      # mechanism component, per-encoder opt-in —
                                      # S1 Scribe today; any S2 unit later.


def _fenced_section_scan(text, marker):
    """The ONE scanner for a journal section — finds the `marker` heading and
    its own fenced ``` block, returning `(section_start, section_end, content)`
    with the section span as [start, end) character offsets and `content` the
    fence body (language tag skipped, stripped). None when the section is
    absent or its fence is malformed. Both the extractors and the strip below
    ride this, so the section-ownership rules can't drift apart.
    """
    if not text:
        return None
    idx = text.find(marker)
    if idx == -1:
        return None
    after = text[idx + len(marker):]
    open_fence = after.find('```')
    if open_fence == -1:
        return None
    # The fence must belong to THIS section. If a new `## ` heading starts
    # before the opening fence, this section has no fence of its own and the
    # fence we found belongs to a LATER section — return None (drift) rather
    # than capturing the wrong section's content. Without this, a fenceless
    # `## Arc` reaches forward into the `## Review` fence (§7.2 orders Arc
    # before Review) and review notes get written as the session arc, silently.
    # Checking position (heading-before-fence) — not blunt truncation — leaves
    # legit fence content that itself contains a `## ` line intact.
    next_heading = after.find('\n## ')
    if next_heading != -1 and next_heading < open_fence:
        return None
    rest = after[open_fence + 3:]
    nl = rest.find('\n')           # skip an optional language tag on the fence line
    if nl != -1:
        rest = rest[nl + 1:]
    close_fence = rest.find('```')
    if close_fence == -1:
        return None
    content_end_in_after = (open_fence + 3
                            + (nl + 1 if nl != -1 else 0)
                            + close_fence + 3)
    return (idx, idx + len(marker) + content_end_in_after,
            rest[:close_fence].strip())


def _extract_fenced_block(text, marker):
    """Pull a fenced block out of an encoder's final text: find the `marker`
    section heading and return the content of its first fenced ``` block.

    Three-valued so writers can tell the cases apart and stay loud:
      • **None** — no `marker` section, or a marker with no parseable fence
        (missing open/close fence). The caller distinguishes "no section" from
        "format drift" by re-checking `marker in text`.
      • **''** — a fenced block that's empty (a legit clean run), distinct
        from drift.
      • **str** — the fence content.
    Extracting ONLY the fence (not the whole section) keeps surrounding prose
    from being mis-parsed as content.
    """
    scan = _fenced_section_scan(text, marker)
    return scan[2] if scan else None


def strip_journal_sections(text):
    """Remove the journal sections (`## Arc` / `## Review` heading + fenced
    block) from an encoder's final text, returning the payload remainder.

    This is harvest's envelope rule for single-shot agents: their response
    carries a JSON payload AND the journal fence in one text, and
    `extract_json`'s rfind-based scan would be corrupted by a `]`/`}` inside
    a fence that follows the payload — so the journal is stripped first.
    Sections that are absent or malformed (drift) are left untouched.
    """
    if not text:
        return text
    stripped_any = False
    for marker in (JOURNAL_ARC_MARKER, JOURNAL_REVIEW_MARKER):
        scan = _fenced_section_scan(text, marker)
        if scan:
            text = text[:scan[0]] + text[scan[1]:]
            stripped_any = True
    # Only tidy whitespace when a section was actually removed — a text with
    # no journal sections passes through byte-identical.
    return text.strip() if stripped_any else text


def extract_review_block(text):
    """The `## Review` fence — content ready for `parse_journal_notes`.
    Three-valued; see `_extract_fenced_block`."""
    return _extract_fenced_block(text, JOURNAL_REVIEW_MARKER)


def extract_arc_block(text):
    """The `## Arc` fence — the run's one-line arc delta, ready for
    `write_session_arc`. Three-valued; see `_extract_fenced_block`."""
    return _extract_fenced_block(text, JOURNAL_ARC_MARKER)


def salvage_review_fence(text):
    """Drift salvage for the write door: notes the encoder fenced WITHOUT the
    `## Review` heading. Observed on Haiku community runs — a perfectly formed
    notes fence loses its heading and the whole batch's residue was dropped on
    the strict marker match.

    Strict all-or-nothing gate, so a code/table fence can never be harvested:
    a fence qualifies only when `parse_journal_notes` accepts EVERY non-blank
    line (>=1 note, zero malformed). Well-formed journal sections are stripped
    first so an `## Arc` fence is never mistaken for notes. Multiple qualifying
    fences → the LAST one (the closure puts the review at the end of the final
    reply). Returns the fence content, or None when nothing qualifies — an
    empty heading-less fence does NOT qualify (indistinguishable from a stray
    code block, unlike a fenced `## Review` where empty means a clean run).
    """
    remainder = strip_journal_sections(text)
    if not remainder:
        return None
    salvaged = None
    pos = 0
    while True:
        open_fence = remainder.find('```', pos)
        if open_fence == -1:
            break
        rest = remainder[open_fence + 3:]
        nl = rest.find('\n')          # skip an optional language tag
        if nl == -1:
            break
        body = rest[nl + 1:]
        close_fence = body.find('```')
        if close_fence == -1:
            break
        pos = open_fence + 3 + nl + 1 + close_fence + 3
        content = body[:close_fence].strip()
        if not content:
            continue
        # A JSON payload fence is never notes: a single-line array/object
        # whose strings contain '·' parses cleanly as one "note" and would
        # qualify — harvesting a single-shot encoder's PAYLOAD as residue
        # (reachable since single-shot responses carry fenced JSON; loop
        # encoders' final text never did).
        if content[0] in '[{':
            import json
            try:
                json.loads(content)
                continue
            except ValueError:
                pass
        notes, malformed = parse_journal_notes(content)
        if notes and not malformed:
            salvaged = content
    return salvaged


# Per-encoder continuity window: how many of an encoder's most recent
# note-bearing runs the "where things stand" read pulls into the next run's
# prompt. Bounds the READ, never storage — notes are append-only and retained
# (§2.7); a 9th run simply doesn't read the 1st's note, which still exists for
# the operator + future miner. A contract constant, NOT interaction-tunable
# by design: continuity depth is a structural property of each encoder's
# cadence, not a knob S2 should self-tune. Keys are the encoder identity used by
# notes() (S1 chain prefix `s1e`; S2 unit NAME). Unlisted encoders use DEFAULT.
JOURNAL_CONTINUITY_RUNS = {
    's1e':                 5,   # S1 Scribe — every 5th Stop; a session spans several runs
    'consolidation':       3,   # S2 idle units run far apart; 3 is enough for cross-run escalation
    'community_detection': 3,
}
JOURNAL_CONTINUITY_RUNS_DEFAULT = 3


# Residue ref_types: encoder *notes*, not integration deltas. Consumers that
# read per-run integration deltas (S2 idle-gating `_last_run_timestamp`, the
# dashboard run-card queries) must EXCLUDE these — a journal_note shares the
# run's chain_id + event_type='delta', so an unfiltered `event_type='delta'`
# pull would otherwise scoop notes and miscount them as runs. Single source for
# the ops-delta-vs-residue partition; exclusion-style so it stays
# behavior-preserving (everything that isn't residue still counts) and
# forward-compatible (add a residue type here, every consumer excludes it).
# A Thalamus filing is residue by the same test: it shares the run's chain
# and event_type='delta' but is not the run's integration delta. Stored
# journal_invocation / journal_checkpoint rows (no longer written) sit on run
# chains the same way.
RESIDUE_REF_TYPES = ('journal_note', 'journal_invocation', 'journal_checkpoint',
                     REF_THALAMUS_FILED)


# Per-mutation ref_types written by the emitter (servers/mutation_emitter.py).
# Excluded by the same consumers, for the same reason as residue: they share the
# run's chain_id + event_type='delta', but they are per-WRITE rows, not the
# unit's per-RUN integration delta. An unfiltered pull would read a single node
# revise as "the run completed" and re-arm the S2 idle gate.
#
# Includes the pre-existing revise pair, not just the new types. Verified safe
# (2026-08-04): every S2 unit stamps its OWN delta ref_type on every exit path —
# `aspect_classified`, `community_enriched`, `consolidated`, `healer_generated` —
# including the early-out and failure branches, so excluding the pair cannot
# starve a unit's cold-start gate. 30 days of live traces confirm it.
# Bonus: this also fixes the dashboard's pre-existing mis-enrichment of revise
# rows as run cards.
EMITTER_REF_TYPES = (
    'node_created', 'node_archived',
    'node_revised', 'edge_relation_revised',
    'node_lock_changed',
)


# ── LLM-ENCODER TELEMETRY GUARD (loud at the write boundary) ──
# A delta produced by an agent that actually called an LLM MUST carry the
# cost/latency telemetry build_delta_metadata accepts (elapsed_ms + token
# counts). These are the ref_types of those deltas — one per LLM encoder.
# (Selection deltas, node/edge_revised, and bare early-out markers are NOT
# here: they have no LLM round to measure.)
LLM_ENCODER_DELTA_REF_TYPES = (
    'encoding_run',        # S1 Scribe
    'consolidated',        # S2 consolidation
    'community_enriched',  # S2 community
    'healer_generated',    # S2 healer
    'aspect_classified',   # S2 aspect integration
)


def check_delta_telemetry(ref_type, metadata):
    """Detect an LLM-encoder delta that ran the model AND did work, yet
    recorded output_tokens==0 — the silent telemetry-threading gap where an
    encoder built its delta without passing run_llm_loop's / the API response's
    token counts to build_delta_metadata (the 2026-06-24 fleet-wide S2 gap).

    Returns a one-line warning string for the caller to log via
    brain._log_error / _log_warning, or None when there's nothing to flag. Pure
    — no logger here (this and build_delta_metadata are contract functions; the
    WRITE boundary owns logging, per "loud at the write boundary"; TraceDAL, the
    other chokepoint, can't reach the errors table mid-append).

    Returns None (no flag), by design, for:
      • non-LLM-encoder ref_types (selection deltas, node/edge_revised, markers);
      • bare markers / no payload (metadata None or not a dict — the early-out
        "No clusters to process" traces);
      • no-work runs (actions==0). actions>0 — not rounds>0 alone — is the
        load-bearing guard. The model can't emit a tool call or a parsed JSON
        result without spending output tokens, so actions>0 with
        output_tokens==0 is an UNAMBIGUOUS wiring gap. Gating on actions>0 also
        excludes the all-LLM-calls-failed case (e.g. healer, whose `rounds`
        counts batches ATTEMPTED — a run where every call raised has rounds>0
        but actions==0; that's an LLM failure, already logged, not a telemetry
        gap, so it must not cry wolf here).
    """
    if ref_type not in LLM_ENCODER_DELTA_REF_TYPES:
        return None
    if not isinstance(metadata, dict):
        return None
    rounds = metadata.get('rounds') or 0
    actions = metadata.get('actions') or 0
    output_tokens = metadata.get('output_tokens') or 0
    if rounds > 0 and actions > 0 and output_tokens == 0:
        return ('%s delta ran %d round(s) with %d action(s) but recorded '
                'output_tokens=0 — LLM telemetry not threaded into '
                'build_delta_metadata' % (ref_type, rounds, actions))
    return None


# Surface (the S1 decoder) is the one LLM agent that is NOT a delta encoder: it
# spends tokens selecting from candidates and writes its cost into the K trace
# (ref_type 'surface_selected'), not a delta. Same silent-regression risk as the
# encoders had pre-2026-06-24 — this is its guard. Unlike the encoder case there
# is no `actions` gate: Haiku ALWAYS emits output (the selection JSON, even an
# empty {"selected":[]}), so rounds>0 with output_tokens==0 is an unambiguous
# wiring gap on its own.


def check_surface_telemetry(metadata):
    """Detect a Surface K trace that ran Haiku yet recorded output_tokens==0 —
    the surface-side analog of check_delta_telemetry (the cost telemetry was
    not threaded from read_usage into build_run_telemetry into the K trace).

    Returns a one-line warning string for the caller to log via
    brain._log_error, or None when there's nothing to flag. Pure — the write
    boundary owns logging (same contract as check_delta_telemetry).
    """
    if not isinstance(metadata, dict):
        return None
    rounds = metadata.get('rounds') or 0
    output_tokens = metadata.get('output_tokens') or 0
    if rounds > 0 and output_tokens == 0:
        return ('surface_selected K trace ran %d Haiku round(s) but recorded '
                'output_tokens=0 — surface cost telemetry not threaded into '
                'the K trace metadata' % rounds)
    return None


# ── SELECTION METADATA SHAPE ──
# Decode-style units (S1R) don't have LLM rounds or write actions — they
# select from candidates. Sibling shape keeps them typed correctly and
# gives the dashboard/S3 a second vocabulary to read. It is the payload of
# the `additionalContext` delta; the `surface_selected` K trace beside it
# carries its own undeclared metadata, built in surface._write_traces.

SELECTION_METADATA_SHAPE = {
    'candidates_considered': int,    # how many inputs scored
    'selected':              list,   # IDs/tags of picks
    'dropped':               list,   # IDs/tags of rejects
    'outcomes_per_candidate': dict,  # {candidate_id: 'selected'|'dropped'|...}
    'content':               str,    # the delta output (e.g. additionalContext), truncated
    # K block — same semantics as DELTA_METADATA_SHAPE's: which prompt+config
    # produced this selection (always stamped: the builder requires it).
    'interaction_fingerprint': str,
    'interaction_source':      str,
    'interaction_version':     int,
}

SELECTION_CONTENT_LIMIT = 4000


def build_selection_metadata(*,
                             candidates_considered=0, selected=None,
                             dropped=None, outcomes_per_candidate=None,
                             content='', interaction, **extras):
    """Build a unified selection-style trace metadata dict (S1R-like).
    `interaction` is required — the selector's K stamp, as in
    build_delta_metadata."""
    metadata = {
        'candidates_considered':  int(candidates_considered or 0),
        'selected':               list(selected or []),
        'dropped':                list(dropped or []),
        'outcomes_per_candidate': dict(outcomes_per_candidate or {}),
        'content':                (content or '')[:SELECTION_CONTENT_LIMIT],
        **interaction_fields(interaction),
    }
    for k, v in extras.items():
        if k not in metadata:
            metadata[k] = v
    return metadata


# ── REVISE METADATA SHAPE ──
# Field-level revise events (event_type='delta', ref_type='node_revised')
# carry per-field deltas + warnings instead of the LLM-loop shape. Used by
# every caller of revise() — direct MCP, S1 encoder, S2 units. Same shape
# whether the caller is dispatch, an encoder, or the operator via MCP.
#
# Warnings carry attempts that didn't land (immutable field passed,
# archive blocked on a locked node). The trace event is emitted
# even when deltas is empty as long as warnings is non-empty — so that
# audit history captures attempted-but-rejected operations, not just
# successful changes.

REVISE_METADATA_SHAPE = {
    'node_id':         str,    # which node was revised
    'reason':          str,    # human-readable reason (required at API)
    'encoding_source': str,    # who made the change (anchor, encoder:sonnet, s2:healer, ...)
    'deltas':          list,   # [{'field': str, 'old': any, 'new': any}, ...]
    'warnings':        list,   # ['immutable field skipped: id', 'archive blocked (locked): archived', ...]
}


def build_revise_metadata(*, node_id, reason, encoding_source='',
                          deltas=None, warnings=None):
    """Build trace metadata for a node revise event.

    Caller responsibility: collect (old, new) pairs for each field that
    actually changed, pass them as `deltas`. Pass `warnings` for fields
    that were rejected (immutable, locked-archive). The trace event is
    worth emitting whenever EITHER deltas or warnings is non-empty.

    Sole node-side producer: the mutation emitter (servers/mutation_emitter.py),
    fed by the `mutations.nodes.revised[]` manifest rows the revise handlers
    return. Returns a dict ready to pass as the metadata kwarg to a trace
    writer.
    """
    return {
        'node_id':         node_id,
        'reason':          reason or '',
        'encoding_source': encoding_source or '',
        'deltas':          list(deltas or []),
        'warnings':        list(warnings or []),
    }


# ── EDGE REVISE METADATA SHAPE (Stage 1B) ──
# Edge-level revise events (event_type='delta', ref_type='edge_relation_revised')
# carry the same delta+warnings shape as node revises but identified by
# (edge_id, relation) tuple. ref_id encoding: f"{edge_id}:{relation}".
#
# Single ref_type covers both create-via-upsert and update-via-upsert from
# `connect()` / `connect_to`, plus archive via polymorphic `archive` op. Empty
# `old` in a delta means the field was just created; populated `old` = update.
#
# source_id/target_id make the edge SELF-DESCRIBING: the directional pair is in
# the trace itself, so the graph's edges are reconstructable from the trace
# substrate alone — without joining the live edges table to invert edge_id.

EDGE_REVISE_METADATA_SHAPE = {
    'edge_id':         str,    # physical edge id (deterministic from source+target)
    'source_id':       str,    # the claim's source — it reads `source relation target`
    'target_id':       str,    # the claim's target (who acts is the verb's orientation)
    'relation':        str,    # which specific relation on that edge
    'reason':          str,    # human-readable reason (required at API)
    'encoding_source': str,    # who made the change
    'deltas':          list,   # [{'field': str, 'old': any, 'new': any}, ...]
    'warnings':        list,   # any skipped/blocked operations
}


def build_edge_revise_metadata(*, edge_id, relation, reason, encoding_source='',
                               source_id='', target_id='',
                               deltas=None, warnings=None):
    """Build trace metadata for an edge_relation revise event.

    Mirrors build_revise_metadata for nodes; same delta shape captures
    connect-upsert outcomes (empty `old` = create, populated `old` = update)
    and polymorphic archive (deltas show archived flag flipping).

    source_id/target_id carry the directional pair so the edge is
    reconstructable from the trace alone (edge_id is a one-way hash of the
    pair — not invertible without the live edges table).

    Sole producer: the mutation emitter, fed by the `mutations.edges[]`
    manifest rows the edge handlers return (connect, connect_batch,
    revise_edge, disconnect, and remember's connect_to path). co_anchored
    edges are noise-aspect and never traced (ruled 2026-08-04).
    """
    return {
        'edge_id':         edge_id,
        'source_id':       source_id or '',
        'target_id':       target_id or '',
        'relation':        relation,
        'reason':          reason or '',
        'encoding_source': encoding_source or '',
        'deltas':          list(deltas or []),
        'warnings':        list(warnings or []),
    }


# ── NODE LIFECYCLE METADATA SHAPES (the emitter's new ref_types) ──
# node_revised above covers "a node changed". These three cover the rest of a
# node's life: born, archived, erased. All are written by ONE producer
# (servers/mutation_emitter.py) via the builders below, so — as with the revise
# pair — every key can be required: the builder always fills it.
#
# `type` and `title` ride along on all three deliberately. They make a row
# self-describing, so a reader can say WHAT was archived or erased without
# joining the nodes table — which for a HARD DELETE is not optional, because the
# node is gone and the trace is the only surviving record of it.

NODE_CREATED_METADATA_SHAPE = {
    'node_id':         str,    # the new node
    'type':            str,    # node type at birth
    'title':           str,    # title at birth (nodes get retitled; this is the birth name)
    'encoding_source': str,    # who created it (anchor, encoder:sonnet, s2:consolidation, ...)
    'reason':          str,    # why, when the caller supplied one
}


def build_node_created_metadata(*, node_id, type='', title='',
                                encoding_source='', reason=''):
    """Build trace metadata for a node-creation event.

    The row that closes the partial-run catalog gap: a run that dies before its
    encoding_run delta still leaves one of these per node it created, so a
    successor can find them by id instead of re-creating duplicates.
    """
    return {
        'node_id':         node_id,
        'type':            type or '',
        'title':           title or '',
        'encoding_source': encoding_source or '',
        'reason':          reason or '',
    }


NODE_LOCK_CHANGED_METADATA_SHAPE = {
    'node_id':           str,    # the flipped node
    'type':              str,    # self-describing, same rationale as the lifecycle trio
    'title':             str,
    'locked':            bool,   # the NEW state
    'reason':            str,    # the operator-facing why
    'encoding_source':   str,    # who ran the command (anchor by channel)
    'confirm_latency_s': float,  # request→confirm gap of the two-phase door
}


def build_lock_change_metadata(*, node_id, type='', title='', locked=False,
                               reason='', encoding_source='',
                               confirm_latency_s=0.0):
    """Build trace metadata for a lock flip (set_node_lock's two-phase door).

    A lifecycle/permission event, deliberately NOT node_revised: the emitter's
    own rule is that a non-content change rendering as "Refined N memories" is
    a lie, and revise history readers must not see `locked` — the field
    revise() treats as immutable — appear as a revised field.
    """
    return {
        'node_id':           node_id,
        'type':              type or '',
        'title':             title or '',
        'locked':            bool(locked),
        'reason':            reason or '',
        'encoding_source':   encoding_source or '',
        'confirm_latency_s': float(confirm_latency_s or 0.0),
    }


NODE_ARCHIVED_METADATA_SHAPE = {
    'node_id':         str,    # the archived node
    'type':            str,
    'title':           str,
    'archived_by':     str,    # archive attribution (distinct from encoding_source)
    'encoding_source': str,    # who ran the command
    'reason':          str,
    'edge_relations':  list,   # [[edge_id, relation], ...] archived WITH the node
    'vectors_deleted': int,    # embedding rows dropped by the cascade
}


def build_node_archived_metadata(*, node_id, type='', title='', archived_by='',
                                 encoding_source='', reason='',
                                 edge_relations=None, vectors_deleted=0):
    """Build trace metadata for a node-archive event.

    Replaces `archive_node`'s inline off-chain `tool_result` trace: archives now
    land on the caller's real chain with real attribution.

    `edge_relations` carries only the (edge_id, relation) pairs the archive
    actually flipped — NOT every edge touching the node. The archive
    deliberately exempts some relations (`absorbed_into` redirects survive), and
    claiming those were archived would make the trace lie about the graph.
    """
    return {
        'node_id':         node_id,
        'type':            type or '',
        'title':           title or '',
        'archived_by':     archived_by or '',
        'encoding_source': encoding_source or '',
        'reason':          reason or '',
        'edge_relations':  [list(p) for p in (edge_relations or [])],
        'vectors_deleted': int(vectors_deleted or 0),
    }


# ── METADATA PAYLOAD VALIDATION (the chokepoint guard) ──
# PLACEMENT IS LOAD-BEARING: this block sits BELOW every *_METADATA_SHAPE it
# references, because the registry dict is built at import time. Moving it above
# a shape definition is a module-level NameError — and trace_contract is imported
# by dal_logs, so the daemon stops booting. (validate_trace_metadata reads the
# dict inside its body, at call time, so only the DICT has this constraint.)
# validate_trace_event() checks the (scale, event_type, ref_type) envelope.
# It historically said nothing about the metadata PAYLOAD — which is exactly
# how two writers emitted two different shapes for the same `encoding_run`
# ref_type, undetected, for weeks. This closes that hole: a ref_type with a
# declared schema must carry every required key with the right type.
# Keyed by ref_type (the unit of shape divergence). Covers every delta built by
# build_delta_metadata — the S1 Scribe plus the four S2 units — so a malformed
# payload on any of them is caught, not just encoding_run. (reclassify's
# `community_assignments` is excluded: it only ever writes a bare summary marker,
# with no build_delta_metadata payload to shape-check.)
METADATA_REQUIRED_BY_REF_TYPE = {
    'encoding_run':       DELTA_METADATA_SHAPE,  # S1 Scribe
    'consolidated':       DELTA_METADATA_SHAPE,  # S2 consolidation
    'community_enriched': DELTA_METADATA_SHAPE,  # S2 community
    'healer_generated':   DELTA_METADATA_SHAPE,  # S2 healer
    'aspect_classified':  DELTA_METADATA_SHAPE,  # S2 aspect integration
    'journal_note':       JOURNAL_NOTE_METADATA_SHAPE,  # encoder residue (one note per row)
    REF_THALAMUS_FILED:   THALAMUS_FILED_METADATA_SHAPE,  # a producer's filing, on its run chain
    'anchor_touched':     ANCHOR_TOUCHED_SHAPE,  # S0 per-turn Anchor action aggregate
    # The highest-volume S0 row. The daemon stamps every new tool row before
    # this chokepoint, including old clients and rows without a session id.
    'tool_result':        TOOL_RESULT_METADATA_SHAPE,
    # Node lifecycle, written only by servers/mutation_emitter.py. Enforced from
    # the start — these have exactly one producer and one builder each, so there
    # is no legacy shape to grandfather in.
    'node_created':       NODE_CREATED_METADATA_SHAPE,
    'node_archived':      NODE_ARCHIVED_METADATA_SHAPE,
    'node_lock_changed':  NODE_LOCK_CHANGED_METADATA_SHAPE,
    # The two mutation ref_types that predate the emitter. Their shapes were
    # DECLARED but never enforced — validation was dead for exactly the two
    # highest-volume mutation events in the system. Enforced as of 2026-08-04,
    # after verifying it cannot fire on existing traffic: the mutation
    # emitter is the sole producer for both (node paths at step 4, edge
    # paths at steps 5-6 — connect_to rides the manifest; co_anchored is
    # noise-aspect and never traced), building metadata via the builders
    # below, and each builder's minimal-args output satisfies its shape.
    # Validation runs at WRITE time only, so historic rows are unaffected.
    'node_revised':          REVISE_METADATA_SHAPE,
    'edge_relation_revised': EDGE_REVISE_METADATA_SHAPE,
    # The S1 Surface delta. surface._write_traces is its sole producer and
    # always builds it with build_selection_metadata, which coerces every key
    # to its declared type. Enforced because its readers (trace_links'
    # candidate-pool join, recall_laf's supervision negatives) read a row with
    # a missing key as contributing nothing, so drift would pass silently.
    'additionalContext':     SELECTION_METADATA_SHAPE,
}


def validate_trace_metadata(event_type, ref_type, metadata):
    """Validate a trace event's metadata payload against its ref_type schema.

    Returns (ok, error_message). Two ways to pass:
      • ref_types without a declared schema (permissive — we only lock shapes
        that have a builder);
      • a bare marker with NO metadata (None) — the delta ref_types double as
        early-out/error markers (`self.trace('delta','consolidated','No clusters
        to process')`), which legitimately carry no payload.
    A PRESENT payload, though, must match the schema. The contract HELPS (catches
    a malformed delta dict) without BLOCKING a no-op marker or dropping anything —
    the caller logs loud and writes the full payload regardless.
    """
    schema = METADATA_REQUIRED_BY_REF_TYPE.get(ref_type or '')
    if not schema:
        return True, ""
    if metadata is None:
        return True, ""   # bare marker — no payload to shape-check
    if not isinstance(metadata, dict):
        return False, "metadata for ref_type '%s' must be a dict or None, got %s" % (
            ref_type, type(metadata).__name__)
    missing = [k for k in schema if k not in metadata]
    if missing:
        return False, "metadata for ref_type '%s' missing required keys: %s" % (
            ref_type, missing)
    bad = [k for k in schema
           if not isinstance(metadata[k], schema[k])]
    if bad:
        return False, "metadata for ref_type '%s' wrong types on keys: %s" % (
            ref_type, bad)
    return True, ""




def validate_trace_event(scale, event_type, ref_type=""):
    """Validate a trace event against the contract.

    Returns (ok, error_message).
    """
    if scale not in SCALES:
        return False, "Unknown scale '%s'. Valid: %s" % (scale, ', '.join(SCALES.keys()))

    if event_type not in EVENT_TYPES:
        return False, "Unknown event_type '%s'. Valid: %s" % (event_type, ', '.join(EVENT_TYPES.keys()))

    if ref_type:
        key = (scale, event_type)
        if key in REF_TYPES and ref_type not in REF_TYPES[key]:
            return False, "Invalid ref_type '%s' for (%s, %s). Valid: %s" % (
                ref_type, scale, event_type, REF_TYPES[key])

    return True, ""


# ── TRACE RENDERING ──
# Mirrors contract.py's node-render layer. brain.query_traces / get_trace /
# get_traces return full rows (the data layer — S2 units read them
# programmatically); the MCP layer renders text via render_trace + these
# configs, never a raw json.dumps. recall_episodes shares this renderer.
#
# The heavy field is `metadata` — s2 K/delta rows reach ~140KB. Bounding it is
# the lever here, exactly as the edge tail was for get_nodes. `rich=true` opts
# into the full metadata.
#
# The BODY is not the lever: `metadata.content` is capped at write time
# (daemon_hooks), so a body is at most a few KB. A point lookup names one row
# and gets its body whole; only multi-row pulls trim it, and then they say so.

TRACE_BODY_CHARS = 280          # body cap for multi-row pulls
TRACE_BULK_BODY_CHARS = 200     # tighter body cap for bulk pulls (>TRACE_BULK_MAX rows)
TRACE_GIST_VALUE_CHARS = 80     # per-key value cap in gist metadata
TRACE_GIST_MAX_KEYS = 8         # keys shown in gist before "+N more"
TRACE_BULK_MAX = 20             # above this many rows, default drops to summary-only
TRACE_BATCH_MAX_IDS = 50        # get_traces: ids served per call; overflow is flagged

# Point lookup (get_trace): the whole body — you named this row — plus a
# metadata gist (key=value, big values elided to "<N chars>" so a 140KB blob
# can't leak). rich=true swaps the gist for full metadata.
TRACE_POINT_FORMAT = {'body_limit': None, 'metadata_mode': 'gist',
                      'show_scale': True}
# Several rows (get_traces, small query): trimmed body + metadata gist.
TRACE_COMPACT_FORMAT = {'body_limit': TRACE_BODY_CHARS, 'metadata_mode': 'gist',
                        'show_scale': True}
# Many rows (large query_traces/get_traces): summary only, no metadata.
TRACE_BULK_FORMAT = {'body_limit': TRACE_BULK_BODY_CHARS, 'metadata_mode': 'none', 'show_scale': True}
# rich=true opt-in: the complete row — full body + full metadata.
TRACE_FULL_FORMAT = {'body_limit': None, 'metadata_mode': 'full', 'show_scale': True}
# recall_episodes (conversational): matches its historic render — body only, no
# scale/event_type chrome (it's always s0 conversation).
TRACE_EPISODE_FORMAT = {'body_limit': TRACE_BODY_CHARS, 'metadata_mode': 'none',
                        'show_scale': False}


def _render_trace_metadata(meta, mode):
    """Render a trace row's metadata dict. 'gist' = key=value with big values
    elided to "<N chars>" (kills the blob, keeps the shape); 'full' = complete.
    'content' is rendered as the body, never repeated here."""
    items = [(k, v) for k, v in meta.items()
             if k != 'content' and v not in (None, '', [], {})]
    if not items:
        return []
    if mode == 'full':
        import json as _json
        out = ['  metadata:']
        for k, v in items:
            sval = v if isinstance(v, str) else _json.dumps(v, default=str)
            out.append('    %s: %s' % (k, sval))
        return out
    # gist
    bits = []
    for k, v in items[:TRACE_GIST_MAX_KEYS]:
        sval = v if isinstance(v, str) else str(v)
        bits.append('%s=<%d chars>' % (k, len(sval)) if len(sval) > TRACE_GIST_VALUE_CHARS
                    else '%s=%s' % (k, sval))
    line = '  ' + '  '.join(bits)
    if len(items) > TRACE_GIST_MAX_KEYS:
        line += '  +%d more' % (len(items) - TRACE_GIST_MAX_KEYS)
    return [line]


def _trim_body(body, limit):
    """Cap a rendered body, naming what was dropped.

    A bare '…' reads as the speaker trailing off — the reader can't tell a
    natural ending from a chopped one. The dropped-char count plus the escape
    hatch says which it is, mirroring how a truncated windowed read names its
    remedy rather than just its limit.

    The notice costs ~40 chars, so a body barely over the limit would render
    LONGER trimmed than whole. Trimming that costs more than it saves isn't
    trimming — those bodies pass through intact.
    """
    if not limit or len(body) <= limit:
        return body
    kept = body[:limit - 1].rstrip()
    notice = '… (+%d chars — get_trace for the whole body)' % (len(body) - len(kept))
    return body if len(kept) + len(notice) >= len(body) else kept + notice


def render_trace(row, config=None):
    """Render one trace_event row to text — the single trace renderer.

    The MCP trace tools (query_traces / get_traces / get_trace) and
    recall_episodes all route through here, mirroring how render_rich_node is
    the one node renderer. Body source: metadata['content'] (conversational
    episodes) falls back to summary (structural traces). `metadata` is bounded
    per config.metadata_mode.
    """
    cfg = {**TRACE_COMPACT_FORMAT, **(config or {})}
    meta = row.get('metadata')
    if not isinstance(meta, dict):
        meta = {}

    sid = (row.get('session_id') or '')[:8]
    score = row.get('_score')
    score_str = ' %.2f' % score if isinstance(score, (int, float)) else ''
    ref_type = row.get('ref_type') or ''
    if ref_type == 'assistant_message':
        # Stamped identity wins — an event records who was speaking when it
        # was written, so a renamed entity's old events keep their old name.
        # The fallback is this install's current name (D-12: config owns it),
        # never a literal; the human's fallback below is a generic role
        # because we cannot invent a person's name.
        from servers.daemon_config import get_agent_name
        label = meta.get('agent_identity') or get_agent_name()
    elif ref_type == 'user_message':
        label = meta.get('human_identity') or 'Operator'
    elif ref_type == 'tool_result':
        label = meta.get('tool') or 'tool_result'
    else:
        label = ref_type or '?'
    when = (row.get('created_at') or '')[:16].replace('T', ' ')

    # Middle segments: [scale event_type] then ref_type (unless it IS the label)
    mids = []
    if cfg.get('show_scale'):
        mids.append('%s %s' % (row.get('scale') or '?', row.get('event_type') or '?'))
    if ref_type and ref_type != label:
        mids.append(ref_type)
    mid_str = (' · ' + ' '.join(mids)) if mids else ''
    # The leading [sid score] bracket is omitted entirely when a trace has
    # neither (session-less S2 system traces, grouped events) — no empty "[]".
    inner = (sid + score_str).strip()
    bracket = '[%s] ' % inner if inner else ''
    tid = row.get('id') or ''
    tid_str = ' (trace:%s)' % tid if tid else ''
    header = '%s%s · %s%s%s' % (bracket, label, when, mid_str, tid_str)

    lines = [header]
    body = (meta.get('content') or row.get('summary') or '').strip()
    if body:
        body = _trim_body(body, cfg.get('body_limit'))
        lines.append('  ' + body.replace('\n', '\n  '))
    if cfg.get('metadata_mode', 'gist') != 'none':
        lines.extend(_render_trace_metadata(meta, cfg['metadata_mode']))
    return '\n'.join(lines)
