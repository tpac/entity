"""Daemon dispatch — observability + learnable-boundary handlers.

Traces, logs, outcomes (the nervous system) and interactions (the K store).
"""

import json

from .clock import iso_cutoff
from .contract import truncation_payload_ids
from .dispatch_common import _agent_limit
from .trace_contract import TRACE_BATCH_MAX_IDS


def _handle_trace_append(brain, args, graph_changes):
    """Append a trace event from any source. Validates against trace_contract.

    Cross-process clients (e.g. PostToolUse hook) typically JSON-encode
    metadata before putting it on the wire because the daemon JSON
    decoder only unwraps one level. Decode it back here so TraceDAL
    stores a clean dict (and identity stamping receives a dict, not a
    string).
    """
    raw_meta = args.get("metadata")
    if isinstance(raw_meta, str) and raw_meta:
        try:
            raw_meta = json.loads(raw_meta)
        except (ValueError, TypeError):
            raw_meta = {"raw": raw_meta}
    # Payload-shape validation lives at the single chokepoint every writer passes
    # — TraceDAL.append/append_batch (loud, never blocking). It used to ALSO run
    # here, which double-validated dispatched traces and split logging across two
    # sinks (errors table here, stderr in the DAL). One point now; the DAL warns
    # to stderr (it can't write the errors table mid-append without risking the
    # brain_batch commit). The json-decode below still normalizes wire metadata.
    # Every captured tool row is normalized here, including before the first
    # prompt or without a session id. Hooks send facts; the daemon owns kinds.
    scale = args.get("scale", "s0")
    session_id = args.get("session_id", "")
    if scale == 's0':
        from .brain_traces import stamp_s0_session, stamp_tool_result
        env = brain.session_env_for(session_id) if session_id else {}
        if args.get('ref_type') == 'tool_result':
            raw_meta = stamp_tool_result(raw_meta, env)
            raw_meta.pop('git_invocation', None)
            raw_meta.pop('capture_filter_incomplete', None)
            if raw_meta['kind'] == 'shell' and raw_meta['kind_status'] == 'ok':
                from .action_policy import load_action_policy, has_git_command
                policy = load_action_policy()
                command = args.get('tool_command')
                if args.get('tool_command_omitted'):
                    raw_meta['capture_filter_incomplete'] = True
                    brain._log_error('action_capture_filter_incomplete',
                                     ValueError('full command exceeded trace transport limit'),
                                     'retaining diagnostic; not classifying a cropped command',
                                     session_id=session_id)
                else:
                    if not isinstance(command, str):
                        command = str(args.get('summary') or '').partition(': ')[2]
                    if has_git_command(command):
                        if policy.exclude_git:
                            return {'ok': True, 'result': {'excluded': 'git_call'}}
                        # Preserve the observed fact if the user later enables
                        # exclusion: a stored short cue may omit the invocation.
                        raw_meta['git_invocation'] = True
            if raw_meta['kind_status'] == 'unknown':
                # _log_error dedups by the first 100 message chars. Put a hash
                # first so long raw names with a shared prefix stay distinct.
                import hashlib
                pair = json.dumps([raw_meta['host'], raw_meta['tool']])
                key = hashlib.sha256(pair.encode()).hexdigest()[:16]
                brain._log_error('tool_kind_unknown',
                                 ValueError('%s: %s' % (key, pair)),
                                 'unclassified tool at the S0 write door; session_id=%r' % session_id,
                                 session_id=session_id)
        elif session_id:
            raw_meta = stamp_s0_session(raw_meta, env)
    try:
        event_id = brain._trace_dal.append(
            chain_id=args.get("chain_id", ""),
            scale=scale,
            event_type=args.get("event_type", ""),
            ref_type=args.get("ref_type", ""),
            ref_id=args.get("ref_id", ""),
            summary=args.get("summary", ""),
            metadata=raw_meta,
            session_id=session_id,
            interaction_id=args.get("interaction_id"))
        return {"ok": True, "result": {"event_id": event_id}}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


def _handle_get_trace(brain, args, graph_changes):
    """Point-lookup a trace_event by id. v29: id is 8-char hex string.
    Rejects int input loudly — coercion was removed because random hex
    generation made it unsafe (collision with migrated-int range)."""
    trace_id = args.get("trace_id")
    if trace_id is None:
        return {"ok": False, "error": "trace_id is required"}
    if not isinstance(trace_id, str) or not trace_id.strip():
        return {"ok": False,
                "error": "trace_id must be an 8-char hex string (v29), got %r" % (trace_id,)}
    tid = trace_id.strip().lower()
    row = brain.get_trace(tid)
    if not row:
        return {"ok": False, "error": "Trace not found: %s" % tid}
    return {"ok": True, "result": row}


def _handle_get_traces(brain, args, graph_changes):
    """Batch trace_event lookup. v29: ids are 8-char hex strings.

    Serves TRACE_BATCH_MAX_IDS ids per call and SAYS SO when the request
    overflows — the cap used to be a silent list slice, which returned a
    partial set that read as complete. Ids that matched no row are named in
    `missing_ids` for the same reason. Rejects non-string ids loudly.
    """
    trace_ids = args.get("trace_ids", [])
    if not isinstance(trace_ids, list):
        return {"ok": False, "error": "trace_ids must be a list of 8-char hex strings"}
    bad = [t for t in trace_ids if not (isinstance(t, str) and t.strip())]
    if bad:
        return {"ok": False,
                "error": "trace_ids must be 8-char hex strings (v29); rejected: %r" % bad[:5]}
    # Dedupe BEFORE the cap. Expanding source_refs across nodes repeats shared
    # anchors, and a repeat that spends a slot costs a unique id that would
    # have fit — a loss invisible in the render, since missing_ids only covers
    # what we looked up. Capping the raw list also flags truncation on a
    # request that was fully served. dict.fromkeys keeps caller order.
    unique = list(dict.fromkeys(t.strip().lower() for t in trace_ids))
    served = unique[:TRACE_BATCH_MAX_IDS]
    rows = brain.get_traces(served) if served else []
    result = {"traces": rows,
              "missing_ids": sorted(set(served) - {r["id"] for r in rows})}
    if len(unique) > TRACE_BATCH_MAX_IDS:
        result["truncated"] = truncation_payload_ids(unique, TRACE_BATCH_MAX_IDS)
    return {"ok": True, "result": result}


def _handle_query_traces(brain, args, graph_changes):
    """Query trace events: O/K/Δ/outcome at every scale.

    session_id / session_ids handling: brain.query_traces (and ultimately
    TraceDAL.get_recent) refuses to combine the two. We forward what the
    caller sent verbatim and surface any ValueError as a structured error
    rather than crashing the daemon thread.
    """
    sids = args.get("session_ids")
    if sids is not None and not isinstance(sids, list):
        return {"ok": False, "error": "session_ids must be a list of strings"}
    try:
        result = brain.query_traces(
            scale=args.get("scale", ""),
            hours=args.get("hours", 24),
            event_type=args.get("event_type", ""),
            chain_id=args.get("chain_id", ""),
            session_id=args.get("session_id", ""),
            session_ids=sids,
            ref_type=args.get("ref_type", ""),
            grouped=args.get("grouped", False),
            limit=args.get("limit", 100))
        return {"ok": True, "result": result}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


def _handle_recall_episodes(brain, args, graph_changes):
    """Episodic pull over the traces layer — semantic (`query`) and/or lexical
    (`contains`) search over trace_events, returning full episode records.

    session_id / session_ids are mutually exclusive (filter_events raises);
    surface any ValueError as a structured error rather than crashing the
    daemon thread, same as query_traces.
    """
    from .brain_constants import EPISODE_DEFAULT_LIMIT, EPISODE_MAX_LIMIT
    sids = args.get("session_ids")
    if sids is not None and not isinstance(sids, list):
        return {"ok": False, "error": "session_ids must be a list of strings"}
    try:
        result = brain.recall_episodes(
            query=args.get("query"),
            contains=args.get("contains"),
            session_id=args.get("session_id"),
            session_ids=sids,
            scale=args.get("scale", "s0"),
            event_type=args.get("event_type"),
            ref_type=args.get("ref_type"),
            older_than=args.get("older_than"),
            younger_than=args.get("younger_than"),
            sort_order=args.get("sort_order", "desc"),
            limit=_agent_limit(args.get("limit"),
                               EPISODE_DEFAULT_LIMIT, EPISODE_MAX_LIMIT))
        return {"ok": True, "result": result}
    except ValueError as e:
        return {"ok": False, "error": str(e)}


def _handle_count_traces(brain, args, graph_changes):
    """Count trace events grouped by a field."""
    result = brain.count_traces(
        field=args.get("field", "event_type"),
        scale=args.get("scale", ""),
        hours=args.get("hours", 24))
    return {"ok": True, "result": result}


def _handle_query_logs(brain, args, graph_changes):
    """Query brain logs: hook errors and debug events."""
    result = brain.query_logs(
        source=args.get("source", "all"),
        hours=args.get("hours", 24),
        level=args.get("level", "all"),
        hook_name=args.get("hook_name", ""),
        limit=args.get("limit", 50))
    return {"ok": True, "result": result}


def _handle_clear_errors(brain, args, graph_changes):
    """Clear hook errors and optionally debug log entries."""
    hours = args.get("hours")  # None = clear all
    cleared = brain._logs_dal.clear_errors(
        cutoff_iso=iso_cutoff(hours=int(hours)) if hours else '',
        include_debug_log=bool(args.get("debug_log", False)))
    return {"ok": True, "result": cleared}


def _handle_list_interactions(brain, args, graph_changes):
    """List all registered interactions."""
    return {"ok": True, "result": brain.list_interactions()}


def _handle_get_interaction(brain, args, graph_changes):
    """Get a specific interaction by name. version=0 returns the ACTIVE row."""
    name = args.get("name", "")
    version = args.get("version", 0)
    result = brain.get_interaction(name=name, version=version)
    if not result:
        # Post-collapse, "no active row" is the NORMAL state of a healthy
        # name: no override deployed, the runtime reads the code default.
        # Saying "not found" for it reads as if the name itself is gone.
        # ok:False is kept for every branch — callers key on ok, and the
        # prose is display-only (daemon_client contract).
        from .interaction_defaults import INTERACTION_DEFAULTS
        known = name in INTERACTION_DEFAULTS or brain.list_interaction_versions(name)
        if version and known:
            return {"ok": False,
                    "error": "%s has no version %s on record" % (name, version)}
        if known:
            return {"ok": False, "error":
                    "%s has no override deployed — the runtime reads its code "
                    "default (servers/interaction_defaults.py). Registered "
                    "versions stay on record: pass version=N to inspect one, "
                    "or register_interaction + set_interaction_active to "
                    "deploy an override." % name}
        return {"ok": False, "error": "Interaction not found: %s" % name}
    return {"ok": True, "result": result}


def _handle_get_interaction_effective(brain, args, graph_changes):
    """The RESOLVED value a run of `name` actually uses: the code default
    with the active override (if any) overlaid — effective template,
    effective config, and the K-provenance stamp, from ONE resolution.
    The read-side complement of get_interaction, which returns the raw
    override ROW and cannot see the default half.

    Templates run to ~90KB (s1e), so the full body ships only on
    include_template=true; the default answer carries a bounded preview
    plus template_length — enough to identify the content without flooding
    an MCP render that has no truncation chokepoint of its own."""
    name = args.get("name", "")
    if not name:
        return {"ok": False, "error": "name is required"}
    try:
        result = brain.get_interaction_effective(name)
    except KeyError as e:
        return {"ok": False, "error": str(e).strip("'\"")}
    template = result.pop("template") or ""
    result["template_length"] = len(template)
    if args.get("include_template"):
        result["template"] = template
    else:
        result["template_preview"] = template[:400]
    return {"ok": True, "result": result}


def _handle_set_interaction_active(brain, args, graph_changes):
    """Flip the active version pointer for an interaction.

    After this call, the runtime path (brain.get_interaction_prompt /
    get_interaction_config) reads the chosen version. The version must
    already be registered. Use after register_interaction to make a
    newly-registered version live, OR to roll back to a previous version.
    """
    from .dal_logs import SYSTEM_PROVENANCE
    name = args.get("name", "")
    version = args.get("version")
    set_by = args.get("set_by", "anchor")
    if not name or version is None:
        return {"ok": False, "error": "name and version are required"}
    if set_by in SYSTEM_PROVENANCE:
        # Reserved: the collapse migration and pointer audits read these to
        # tell system-placed pointers from a human's deployment decisions.
        # Accepting one here would relabel this deployment decision as
        # system-placed residue.
        return {"ok": False,
                "error": "set_by '%s' is reserved for system provenance"
                         % set_by}
    try:
        version = int(version)
    except (TypeError, ValueError):
        return {"ok": False, "error": "version must be an integer"}
    try:
        result = brain.set_interaction_active(name, version, set_by)
        from .interaction_defaults import INTERACTION_DEFAULTS
        if name not in INTERACTION_DEFAULTS:
            # The write doors accept any name (tests and legacy cleanup need
            # that), but only registry names have a runtime reader — say so
            # HERE, at deploy time, or the typo is discovered never.
            result = dict(result or {})
            result["note"] = (
                "WARNING: %r has no code default in interaction_defaults.py "
                "— no runtime path resolves this name, so this pointer "
                "deploys nothing. Typo?" % name)
        return {"ok": True, "result": result}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def _handle_clear_interaction_override(brain, args, graph_changes):
    """Delete the active pointer for an interaction — revert to the code default.

    The inverse of set_interaction_active. After this call the runtime path
    (brain.get_interaction_prompt / get_interaction_config) resolves the code
    default, immediately — TTL caches are invalidated. Registered versions
    stay on record for re-activation. No set_by is written (a delete leaves
    no row), so the reserved-provenance check does not apply. A typo'd name
    (nothing deleted, no code default) is refused, never reported as
    "already on the default".
    """
    name = args.get("name", "")
    if not name:
        return {"ok": False, "error": "name is required"}
    try:
        result = brain.clear_interaction_override(name)
        result["note"] = (
            "Override cleared — runtime now resolves the code default."
            if result.get("cleared")
            else "No pointer existed — already running the code default.")
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def _handle_register_interaction(brain, args, graph_changes):
    """Register a new version of an interaction (prompt + config).

    Creates a new version if the interaction exists, or version 1 if new.

    NOTE: Registration NEVER activates (a write is not a deployment
    decision) — every name runs on its code default until
    set_interaction_active deploys an override.
    """
    from .dal_logs import SYSTEM_PROVENANCE
    name = args.get("name", "")
    template = args.get("template", "")
    parameters = args.get("parameters", "{}")
    created_by = args.get("created_by", "")

    if not name:
        return {"ok": False, "error": "name is required"}
    if created_by in SYSTEM_PROVENANCE:
        # Reserved — see _handle_set_interaction_active. A version wearing
        # system provenance would read as system residue to the audits.
        return {"ok": False,
                "error": "created_by '%s' is reserved for system provenance"
                         % created_by}

    try:
        result = brain.register_interaction(
            name=name,
            template=template,
            parameters=parameters,
            created_by=created_by)
        # Registration does NOT activate. Return the newly-registered version
        # plus the currently-active version so the caller knows whether it
        # took effect at runtime.
        active = brain.get_interaction(name)
        note = ("Registration created a new version but did NOT activate it. "
                "Call set_interaction_active to deploy it as the override.")
        from .interaction_defaults import INTERACTION_DEFAULTS
        if name not in INTERACTION_DEFAULTS:
            note += (" WARNING: %r has no code default in "
                     "interaction_defaults.py — no runtime path resolves "
                     "this name, so activating it deploys nothing. Typo?"
                     % name)
        return {"ok": True, "result": {
            "name": name,
            "registered_version": result.get("version"),
            "active_version": active.get("version") if active else None,
            "template_length": len(template),
            "note": note,
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
