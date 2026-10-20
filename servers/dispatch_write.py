"""Daemon dispatch — write handlers (mutations).

remember / revise / connect (each with its _batch form), revise_edge,
brain_batch and set_node_lock, plus their helpers:
source_refs validation and manifest row shaping. Every node/graph mutation
goes through here (other is_write commands — save, set_config, trace_append,
register_interaction, clear_errors, eval — live in sibling dispatch modules);
handlers return a `mutations` manifest and the dispatch chokepoint
emits the traces (servers/mutation_emitter.py) — no handler writes a trace
row itself.
"""

import json
import re
import sys

from .dispatch_common import _pop_session_ctx, caller_session, CALLER_SESSION_KEY, CALLER_SIG_KEY
from .scales.dispatch import stamp_scope_provenance
from .scales.s2.community_contract import (
    community_create_refusal, community_member_targets)


_HEX_TRACE_ID_RE = re.compile(r'^[0-9a-f]{8}$')


def _stamp_session_scope(brain, cmd, args, ctx=None):
    """Deterministic scope provenance at the MCP write boundary.

    Resolves the calling session's scope policy (brain.scope_policy_for —
    the single builder, dimension-generic) and applies
    stamp_scope_provenance: node-creating payloads get the values
    force-stamped, agent-supplied values elsewhere are dropped. Sessionless
    callers (the in-process encoder dispatch — already stamped at its own
    chokepoint — and direct handler calls) pass through untouched (None
    policy = no session authority).

    Returns warning strings for the handler to surface in its result.
    """
    sid = ctx.session_id if ctx is not None else caller_session(args)
    if cmd in ('revise', 'revise_batch'):
        # revise policy strips regardless of the VALUES — only the
        # session's existence matters, so skip the value resolution
        from .scales.dispatch import SCOPE_PROVENANCE_FIELDS
        scope = {f: '' for f in SCOPE_PROVENANCE_FIELDS} if sid else None
    else:
        scope = brain.scope_policy_for(sid)
    warnings = stamp_scope_provenance(cmd, args, scope)
    for w in warnings:
        # agent drift, not a failure (the fields are system_stamped and not
        # in the MCP schemas, but agents may still emit them from
        # habit/training) — warning severity keeps the error feed real
        brain._log_warning('scope_provenance_stamp', w, 'cmd=%s' % cmd)
    return warnings


def _validate_source_refs(refs, location: str):
    """v29 / Phase B Step 4 — validation for source_refs at the write boundary.

    Returns (ok, error|None). Sparseness + hex-format warnings logged
    separately (non-fatal).

    Rules:
      - Must be a list (or None — handled by caller as "no refs"; empty list
        is legitimate per the unified revise contract — explicit clear).
      - Each item must be a string. Int input rejected loudly (no coercion);
        v29 contract is hex strings end-to-end (reviewer F2).
      - Empty strings rejected.
      - Hex format mismatch (^[0-9a-f]{8}$): soft warn here; the DAL
        refuses non-hex refs at persist (logged as source_refs_persist*,
        node/field writes unaffected — the ref is the only thing dropped).
      - >5 refs logged as sparseness warning (decision 13 / §7.5 v22 prompt:
        1-3 typical; second-guess at 5-6; 10+ is "ref everything" anti-pattern).
        Non-fatal — write proceeds, log fires.
    """
    if refs is None:
        return True, None
    if not isinstance(refs, list):
        return False, "source_refs at %s must be a list, got %s" % (
            location, type(refs).__name__)
    for i, r in enumerate(refs):
        if not isinstance(r, str):
            return False, ("source_refs[%d] at %s must be an 8-char hex "
                           "string (v29), got %s (%r)") % (
                i, location, type(r).__name__, r)
        if not r.strip():
            return False, "source_refs[%d] at %s is empty/whitespace" % (i, location)
    return True, None


def _maybe_warn_source_refs_hex_format(brain, refs, location: str):
    """Soft warn when any source_refs entry doesn't match the v29 8-char hex
    shape. Non-fatal — refs persist; S2Healer's invalid_refs_dropped_total
    eventually cleans hallucinated ids but this surfaces them at the write
    boundary (reviewer F6). Most common cause: encoder copied a literal
    `<trace-...>` placeholder from an example instead of substituting real
    hex from the conversation timeline."""
    if not refs or not isinstance(refs, list):
        return
    malformed = [r for r in refs
                 if isinstance(r, str) and not _HEX_TRACE_ID_RE.match(r)]
    if malformed:
        try:
            brain._log_warning(
                'source_refs_hex_format',
                'source_refs at %s has %d entries that do not match the v29 '
                '8-char hex pattern (^[0-9a-f]{8}$). Examples: %r. Likely '
                'literal example-placeholders or pre-v29 integer ids. Refs '
                'persist; recall will degrade gracefully but S2Healer will '
                'archive them.' % (location, len(malformed), malformed[:3]),
                context='location=%s, malformed_count=%d' % (
                    location, len(malformed)))
        except Exception:
            pass  # logging failure must never block the write


def _maybe_warn_source_refs_sparseness(brain, refs, location: str):
    """Log a sparseness warning when source_refs exceeds 5 (decision 13 +
    §7.5 v22 prompt). 1-3 typical, second-guess at 5-6, 10+ is the
    "ref everything" anti-pattern. Non-fatal — the write proceeds.
    Surfaces systemic over-anchoring.

    v22 lowered the threshold from >10 to >5 to align with the prompt's
    sparseness teaching ("when you find yourself wanting to add a 5th or
    6th ref, ask: would that ref actually be the one that surfaces this
    memory next time, or is it just adjacent context?").

    Routes through brain._log_warning for rate-limiting + file-log mirror +
    canonical surface (reviewer F4)."""
    if not refs or not isinstance(refs, list):
        return
    if len(refs) > 5:
        try:
            brain._log_warning(
                'source_refs_sparseness',
                'source_refs at %s has %d entries (decision 13: 1-3 '
                'typical, second-guess at 5-6, 10+ is anti-pattern). '
                'Likely over-anchoring; recall will dilute rather than '
                'focus.' % (location, len(refs)),
                context='count=%d, location=%s' % (len(refs), location))
        except Exception:
            pass  # logging failure must never block the write


def _missing_reason_error(spec, prefix=''):
    """Build the missing-`reason` error for revise, disambiguating `reasoning`.

    `reason` (audit note, recorded in the node_revised trace event, never
    stored on the node) and `reasoning` (a PROMOTED node field — "why this
    was encoded", stored in node_metadata_kv) are near-identical names for
    different concepts, and agents confuse them (observed 2026-06-12: a
    revise op carrying `reasoning` where `reason` was meant). A blind alias
    would be wrong — `reasoning` is a legitimate field update on revise —
    so when it's present the error names both and tells the caller how to
    self-correct.
    """
    base = (prefix + "reason is required — the audit note explaining why "
            "this revision (recorded in the trace event, NOT stored on "
            "the node)")
    if spec.get('reasoning'):
        return (base + ". You passed `reasoning`, which is a node FIELD "
                "(why the node was encoded — updates node metadata), not "
                "the audit note. If you meant the audit note, rename it to "
                "`reason`; if you meant to update the node's reasoning "
                "field, keep `reasoning` and add `reason`.")
    return base


def _edge_row(result, relation, reason, encoding_source, source_id, target_id):
    """Shape one edges[] manifest row from an edge-op result (add_relation /
    revise_edge / remove_relation return shapes all carry edge_id + deltas).
    Keys are exactly build_edge_revise_metadata's kwargs — builder-shaped by
    construction. The emitter's _changed gate decides emission, so a no-op
    row (empty deltas+warnings) is safe to include."""
    return {
        'edge_id': result.get('edge_id') or '',
        'relation': result.get('relation') or relation,
        'reason': reason,
        'encoding_source': encoding_source or '',
        'source_id': source_id or '',
        'target_id': target_id or '',
        'deltas': result.get('deltas') or [],
        'warnings': list(result.get('warnings') or []),
    }


def _archived_row(arch, archived_by, edge_relations):
    """Shape one nodes.archived manifest row from an archive_node result.
    Keys are exactly build_node_archived_metadata's kwargs. Shared by
    _op_archive and _op_absorb (the absorbed node's row) so the shape can't
    drift between the two paths that archive.

    encoding_source = archived_by deliberately: it's the actor the graph
    carries (_sys_archived_by, the edge stamps), it's what routes the row's
    scale, and it keeps all of one absorb's rows on one scale instead of
    splitting the archived row from its siblings. The command runner is
    recoverable from session_id + chain."""
    return {
        'node_id': arch.get('node_id') or '',
        'type': arch.get('type') or '',
        'title': arch.get('title') or '',
        'archived_by': archived_by,
        'encoding_source': archived_by,
        'reason': arch.get('reason') or '',
        'edge_relations': edge_relations,
        'vectors_deleted': arch.get('vectors_deleted', 0),
    }


def _connect_to_rows(connect_to_result, encoding_source, reason='connect_to'):
    """edges[] manifest rows from a connect_to result's made-list ({src_id,
    target_id, relation, edge_id, deltas[, warnings]} entries) — a dict with
    `created`, or the list itself. The emitter's _changed gate drops idempotent
    re-connects (empty deltas), exactly as the legacy batch emit did."""
    made = (connect_to_result if isinstance(connect_to_result, list)
            else (connect_to_result or {}).get('created') or [])
    return [_edge_row(e, e.get('relation', ''), reason, encoding_source,
                      e.get('src_id', ''), e.get('target_id', ''))
            for e in made if isinstance(e, dict) and e.get('edge_id')]


def _revise_edge_rows(connect_to_result, encoding_source, reason):
    """edges[] manifest rows for connect_to on REVISE: created edges under the
    connect_to reason (as on remember), revised ones under the revise's own
    reason. Shared by the single and batch revise handlers."""
    ct = connect_to_result or {}
    return (_connect_to_rows(ct.get('created') or [], encoding_source)
            + _connect_to_rows(ct.get('revised') or [], encoding_source, reason))


def _affected(created=None, revised=None, archived=None):
    """Build the authoritative node-lifecycle split a write handler returns.

    Sits at the TOP LEVEL of a handler's return (sibling to `result`), so it
    reaches the trace substrate — the runner copies it onto each action and
    the daemon forwards it verbatim — without ever entering the agent-facing
    payload (only `result` is formatted back to the model). NODE lifecycle
    only; edges are directional `edge_relation_revised` events, not a flat
    list here. Replaces the runner's tool-name `_split_action_ids` heuristic
    with attribution from the layer that actually knows the op.
    """
    return {'created': list(created or []),
            'revised': list(revised or []),
            'archived': list(archived or [])}


def _resolve_archived_by(op_spec, top_encoding_source):
    """Attribution fallback for batch archive/absorb/disconnect ops:
    op-level encoding_source → op-level archived_by → batch encoding_source →
    'unknown'. One source so the three _op_* helpers can't drift."""
    return (op_spec.get('encoding_source') or op_spec.get('archived_by')
            or top_encoding_source or 'unknown')


def _handle_remember_command(brain, args, graph_changes):
    """The `remember` command. Its connect_to is applied inline by
    brain.remember, so the community-members check reads it here — the batch
    ops strip connect_to before reaching _handle_remember and check first."""
    refusal = community_create_refusal(args)
    if refusal:
        return {"ok": False, "error": refusal}
    return _handle_remember(brain, args, graph_changes)


def _handle_remember(brain, args, graph_changes):
    from .contract import unpack_fields, validate_field

    # `fields` unpacks before every check below, so each sees its keys as
    # top-level ones (brain.remember binds fields to named parameters).
    if isinstance(args, dict):
        args, err = unpack_fields(args)
        if err:
            return {"ok": False, "error": err}
    # session_id is a control field — strip BEFORE validation/passthrough so
    # it doesn't land in node_metadata_kv as silent metadata. Loads the
    # caller's ctx: the scope stamp below reads it, and remember() gets it.
    ctx, args = _pop_session_ctx(brain, args)
    # chain_id is likewise a control field (trace grouping, NOT a node field).
    # remember() routes unknown kwargs to node_metadata_kv, so an un-popped
    # chain_id would leak onto the node — pop it; the chokepoint captured it
    # for the emitter BEFORE this handler ran.
    if isinstance(args, dict):
        args.pop('chain_id', None)

    # Deterministic project provenance — session-derived, never agent-authored.
    scope_warnings = _stamp_session_scope(brain, 'remember', args, ctx=ctx)

    # v29 / Phase B Step 4 — validate source_refs shape at the write boundary.
    refs = args.get('source_refs')
    ok, err = _validate_source_refs(refs, 'remember')
    if not ok:
        return {"ok": False, "error": err}
    _maybe_warn_source_refs_sparseness(brain, refs, 'remember')
    _maybe_warn_source_refs_hex_format(brain, refs, 'remember')

    # Validate all provided fields against contract
    for field, value in args.items():
        ok, err = validate_field(field, value)
        if not ok:
            return {"ok": False, "error": err}

    # Mirror of revise's reason/reasoning trap: `reason` is revise's audit
    # param and a remember control field — _store_node_metadata drops it
    # silently. An agent passing it here almost always meant the node field
    # `reasoning`. Semantics unchanged (still dropped); surface the drop.
    reason_warning = None
    if args.get('reason') and not args.get('reasoning'):
        reason_warning = (
            "`reason` is not a node field and was dropped — it is the audit "
            "note for revise ops. Did you mean `reasoning` (why this was "
            "encoded — stored on the node)?")

    # Pass ALL fields through to remember() — contract fields go to nodes table,
    # promoted fields to metadata, everything else to node_metadata_kv as extra_fields.
    # Don't filter — remember() handles routing via **extra_fields kwargs.
    remember_args = {k: v for k, v in args.items() if v is not None}
    result = brain.remember(**remember_args, ctx=ctx)
    if reason_warning and isinstance(result, dict):
        result.setdefault('warnings', []).append(reason_warning)
    if scope_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(scope_warnings)
    full_id = result.get("id", "") if isinstance(result, dict) else ""
    node_id = full_id[:8] if full_id else "?"
    graph_changes.append(
        "REMEMBER: [%s] %s (%s...)" % (
            args.get("type", "?"), args.get("title", "")[:50], node_id))
    # Manifest for the emitter: the node's birth row + connect_to edge rows
    # (made-entries are src-tagged with edge_id/deltas via _apply_connect_to).
    # Session/chain attribution comes from the chokepoint's PRE-handler capture
    # — the pop-then-read bug (session_id='' on every remember-path edge
    # trace) dies here, structurally.
    enc_src = args.get('encoding_source', '')
    edge_rows = _connect_to_rows(
        result.get('connect_to_result') if isinstance(result, dict) else None,
        enc_src)
    # pop: co_anchored is an automatic internal edge — out of the payload, and
    # NOT traced: co_anchored is in the noise aspect (ruled 2026-08-04 —
    # soft/derived edges stay out of trace coverage).
    if isinstance(result, dict):
        result.pop('co_anchored_made', None)
    created_rows = []
    if full_id:
        created_rows.append({
            'node_id': full_id,
            'type': args.get('type', ''),
            'title': args.get('title', ''),
            'encoding_source': enc_src,
            'reason': '',
        })
    return {"ok": True, "result": result,
            "affected": _affected(created=[full_id] if full_id else None),
            "mutations": {"nodes": {"created": created_rows},
                          "edges": edge_rows}}


def _element_as_dict(element):
    """A batch element as the dict it must be, or None.

    Sonnet sometimes emits an element JSON-encoded as a string (seen in S2
    community batches and in S1E remember_batch calls); a string that parses
    to a dict is the intended spec. Anything else is a malformed element the
    handler must refuse loudly — a raised AttributeError here aborts the
    encoder's whole run (id:23a29491). Returns (dict or None, unwrapped).
    """
    if isinstance(element, dict):
        return element, False
    if isinstance(element, str):
        try:
            parsed = json.loads(element)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed, True
    return None, False


def _warn_string_element(brain, cmd, key, index):
    try:
        brain._log_warning(
            'batch_string_element',
            '%s.%s[%d] arrived JSON-encoded as a string — unwrapped to the intended spec' % (cmd, key, index),
            'lossless recovery; the caller emitted a stringified element')
    except Exception:
        pass


def _handle_remember_batch(brain, args, graph_changes):
    from .contract import unpack_fields, validate_field

    # Pop session_id BEFORE spec scrubbing so per-node specs don't inherit
    # control fields. Also strip from each spec defensively in case the
    # caller bundled it inside a node spec.
    ctx, args = _pop_session_ctx(brain, args)

    nodes = args.get("nodes", [])
    if not nodes:
        return {"ok": False, "error": "nodes array is required"}

    # Deterministic project provenance — session-derived, never agent-authored.
    scope_warnings = _stamp_session_scope(
        brain, 'remember_batch', args, ctx=ctx)

    # Inherit top-level encoding_source into each node (dispatch wrapper injects this)
    top_encoding_source = args.get("encoding_source")

    cleaned_nodes = []
    reason_warnings = []  # reason/reasoning confusion — see _handle_remember
    # The batch-level connect_to is written from every created node, so a
    # community_member target there is a member of each community created.
    batch_members = community_member_targets(
        {'type': 'community', 'connect_to': args.get('connect_to')})
    for i, spec in enumerate(nodes):
        spec, unwrapped = _element_as_dict(spec)
        if spec is None:
            return {"ok": False, "error": "nodes[%d] must be an object (a node spec), got %s — no nodes were written"
                    % (i, type(nodes[i]).__name__)}
        if unwrapped:
            _warn_string_element(brain, 'remember_batch', 'nodes', i)
        # `fields` unpacks before the checks below, as in _handle_remember
        # (the scope stamp already ran: a stamped key inside `fields` is
        # reserved and refused).
        spec, err = unpack_fields(spec)
        if err:
            return {"ok": False, "error": "nodes[%d]: %s — no nodes were "
                    "written" % (i, err)}
        # defensive: no identity key is a node field. _pop_session_ctx
        # already stripped the top-level args; this guards a spec that bundled
        # one per-node (so it can't cascade into node_metadata_kv).
        spec.pop('session_id', None)
        spec.pop(CALLER_SESSION_KEY, None)
        spec.pop(CALLER_SIG_KEY, None)
        if spec.get('reason') and not spec.get('reasoning'):
            reason_warnings.append(
                "nodes[%d]: `reason` is not a node field and was dropped — "
                "it is the audit note for revise ops. Did you mean "
                "`reasoning` (why this was encoded — stored on the node)?" % i)
        # v29 / Phase B Step 4 — per-node source_refs validation
        refs = spec.get('source_refs')
        ok, err = _validate_source_refs(refs, 'remember_batch.nodes[%d]' % i)
        if not ok:
            return {"ok": False, "error": err}
        _maybe_warn_source_refs_sparseness(
            brain, refs, 'remember_batch.nodes[%d]' % i)
        _maybe_warn_source_refs_hex_format(
            brain, refs, 'remember_batch.nodes[%d]' % i)
        for field, value in spec.items():
            ok, err = validate_field(field, value)
            if not ok:
                return {"ok": False, "error": "node[%d].%s: %s" % (i, field, err)}
        refusal = None if batch_members else community_create_refusal(spec)
        if refusal:
            return {"ok": False, "error": "nodes[%d]: %s" % (i, refusal)}
        # Pass ALL fields through — contract fields go to nodes table,
        # promoted fields to metadata, extras to node_metadata_kv
        cleaned = {k: v for k, v in spec.items() if v is not None}
        if top_encoding_source and 'encoding_source' not in cleaned:
            cleaned['encoding_source'] = top_encoding_source
        cleaned_nodes.append(cleaned)

    # remember_batch forwards only nodes/connect_to/ctx.
    result = brain.remember_batch(
        nodes=cleaned_nodes,
        connect_to=args.get("connect_to"),
        ctx=ctx)
    # ctx mutations persist via autosave (no per-call save).
    if reason_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(reason_warnings)
    if scope_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(scope_warnings)
    graph_changes.append("REMEMBER_BATCH: %d nodes" % result.get("nodes_created", 0))
    enc_src = args.get('encoding_source', '')
    # `connect_to_made` is the brain method's edge record — popped into the
    # manifest so it doesn't bloat the agent-facing payload. Attribution comes
    # from the chokepoint's pre-handler capture (kills the pop-then-read
    # session_id='' bug).
    made = result.pop('connect_to_made', None) if isinstance(result, dict) else None
    edge_rows = _connect_to_rows({'created': made or []}, enc_src)
    # co_anchored edges fire inside each per-node remember(); popped for
    # payload hygiene, NOT traced — noise aspect (ruled 2026-08-04).
    created_rows = []
    for node_r, spec in zip(result.get('results') or [], cleaned_nodes):
        if not isinstance(node_r, dict):
            continue
        node_r.pop('co_anchored_made', None)
        if node_r.get('id'):
            created_rows.append({
                'node_id': node_r['id'],
                'type': spec.get('type', ''),
                'title': spec.get('title', ''),
                # per-row source (spec inherited top when absent) so the
                # emitter routes each node's scale/chain by its own creator
                'encoding_source': spec.get('encoding_source', ''),
                'reason': '',
            })
    created_ids = [r.get('id') for r in (result.get('results') or [])
                   if isinstance(r, dict) and r.get('id')]
    return {"ok": True, "result": result,
            "affected": _affected(created=created_ids),
            "mutations": {"nodes": {"created": created_rows},
                          "edges": edge_rows}}


def _handle_revise(brain, args, graph_changes):
    """Update any field(s) on an existing node via revise()."""
    from .contract import refusal_pointers, unpack_fields, validate_field

    # `fields` unpacks before every check below, as in _handle_remember.
    args, err = unpack_fields(args, revising=True)
    if err:
        return {"ok": False, "error": err + " — nothing was written"}
    node_id = args.get("node_id", "")
    reason = args.get("reason", "")
    if not node_id:
        return {"ok": False, "error": "node_id is required"}
    if not reason:
        return {"ok": False, "error": _missing_reason_error(args)}

    # Project is birth provenance — a revise never moves it (migration does).
    scope_warnings = _stamp_session_scope(brain, 'revise', args)

    # Reserve known dispatch keys so they don't get treated as field updates.
    # The identity keys (the ambient session the proxy stamps, and the hook
    # signature it strips) are reserved too, so neither lands in `updates` as
    # a bogus node field.
    DISPATCH_KEYS = {"node_id", "reason", "encoding_source", "chain_id",
                     "session_id", CALLER_SESSION_KEY, CALLER_SIG_KEY}
    updates = {k: v for k, v in args.items() if k not in DISPATCH_KEYS}

    # v29 / Phase B Step 4 — source_refs validation on revise
    refs = updates.get('source_refs')
    ok, err = _validate_source_refs(refs, 'revise')
    if not ok:
        return {"ok": False, "error": err}
    _maybe_warn_source_refs_sparseness(brain, refs, 'revise')
    _maybe_warn_source_refs_hex_format(brain, refs, 'revise')

    for field, value in updates.items():
        ok, err = validate_field(field, value, revising=True)
        if not ok:
            return {"ok": False, "error": err}

    content = updates.pop("content", None)
    enc_src = args.get('encoding_source', '')
    result = brain.revise(node_id=node_id, content=content, reason=reason,
                          updates=updates, encoding_source=enc_src or None)
    if result.get('error'):
        return {"ok": False, "error": result['error'], **refusal_pointers(result)}

    # connect_to on revise: one directional edge_relation_revised row per
    # edge changed. Failures stay in the agent-facing connect_to_result.
    edge_rows = _revise_edge_rows(result.get('connect_to_result'), enc_src, reason)
    for e in edge_rows:
        graph_changes.append("%s: %s -[%s]-> %s" % (
            'CONNECT_TO' if e['reason'] == 'connect_to' else 'REVISE_EDGE',
            e['source_id'][:8], e['relation'], e['target_id'][:8]))

    # Surface verification failures as warnings
    if not result.get('verified', True):
        failures = result.get('verification_failures', [])
        graph_changes.append("VERIFY_FAIL: revise %s — fields not confirmed: %s" % (
            node_id[:12], ', '.join(failures)))
        # Log to brain error system so integrity producer can surface it
        try:
            brain._log_error('write_verification',
                Exception('Revise verification failed for %s: %s' % (node_id[:12], failures)),
                'Fields claimed updated but read-back shows mismatch')
        except Exception as e2:
            print('[daemon_dispatch] ERROR logging write_verification: %s' % e2, file=sys.stderr)

    graph_changes.append("REVISE: [%s] %s" % (
        result.get("type", "?"), result.get("title", "")[:50]))
    # Manifest row for the emitter (node_revised replaces the legacy
    # _sys_revision_history substrate). Row warnings are the revise's OWN
    # (attempted-but-rejected ops), captured before scope_warnings are
    # appended to the result — dispatch bookkeeping is not audit history and
    # must not flip the emit gate. A no-change revise (empty deltas+warnings)
    # stays in `affected` but the emitter stays silent.
    row = {
        "node_id": node_id,
        "reason": reason,
        "encoding_source": args.get('encoding_source', ''),
        "deltas": result.get('deltas', []),
        "warnings": list(result.get('warnings', [])),
    }
    if scope_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(scope_warnings)
    return {"ok": True, "result": result,
            "affected": _affected(revised=[node_id]),
            "mutations": {"nodes": {"revised": [row]}, "edges": edge_rows}}


def _handle_revise_batch(brain, args, graph_changes):
    """Revise multiple nodes in one call."""
    from .contract import unpack_fields, validate_field

    revisions = args.get("revisions", [])
    if not revisions:
        return {"ok": False, "error": "revisions array is required"}

    # Project is birth provenance — a revise never moves it (migration does).
    scope_warnings = _stamp_session_scope(brain, 'revise_batch', args)

    # Inherit encoding_source from dispatch wrapper
    top_encoding_source = args.get("encoding_source")

    # Validate each revision
    for i, spec in enumerate(revisions):
        spec, unwrapped = _element_as_dict(spec)
        if spec is None:
            return {"ok": False, "error": "revisions[%d] must be an object (a revision spec), got %s — nothing was revised"
                    % (i, type(revisions[i]).__name__)}
        if unwrapped:
            _warn_string_element(brain, 'revise_batch', 'revisions', i)
        # `fields` unpacks before the checks below, as in _handle_revise.
        spec, err = unpack_fields(spec, revising=True)
        if err:
            return {"ok": False, "error": "revisions[%d]: %s — nothing was "
                    "revised" % (i, err)}
        revisions[i] = spec
        if not spec.get("node_id"):
            return {"ok": False, "error": "revisions[%d]: node_id required" % i}
        if not spec.get("reason"):
            return {"ok": False,
                    "error": _missing_reason_error(spec, "revisions[%d]: " % i)}
        # v29 / Phase B Step 4 — per-revision source_refs validation
        refs = spec.get('source_refs')
        ok, err = _validate_source_refs(refs, 'revise_batch.revisions[%d]' % i)
        if not ok:
            return {"ok": False, "error": err}
        _maybe_warn_source_refs_sparseness(
            brain, refs, 'revise_batch.revisions[%d]' % i)
        _maybe_warn_source_refs_hex_format(
            brain, refs, 'revise_batch.revisions[%d]' % i)
        for field, value in spec.items():
            if field not in ("node_id", "reason"):
                ok, err = validate_field(field, value, revising=True)
                if not ok:
                    return {"ok": False, "error": "revisions[%d].%s: %s" % (i, field, err)}

    resolved = []
    for spec in revisions:
        r = dict(spec)
        if top_encoding_source and 'encoding_source' not in r:
            r['encoding_source'] = top_encoding_source
        resolved.append(r)

    result = brain.revise_batch(resolved)
    graph_changes.append("REVISE_BATCH: %d revised" % result.get("revised", 0))

    # One manifest row per revised node, each carrying its OWN
    # encoding_source — one batch can span scales and the emitter routes
    # per row. Warnings included so audit history captures
    # attempted-but-rejected ops.
    revised_ids = []
    manifest_rows = []
    edge_rows = []
    for row, spec in zip(result.get('results', []), resolved):
        if row.get('status') == 'revised':
            revised_ids.append(row['node_id'])
            enc_src = spec.get('encoding_source', '') or top_encoding_source or ''
            manifest_rows.append({
                "node_id": row['node_id'],
                "reason": spec.get('reason', ''),
                "encoding_source": enc_src,
                "deltas": row.get('deltas', []),
                "warnings": list(row.get('warnings', [])),
            })
            edge_rows.extend(_revise_edge_rows(
                row.get('connect_to_result'), enc_src, spec.get('reason', '')))

    if scope_warnings and isinstance(result, dict):
        result.setdefault('warnings', []).extend(scope_warnings)
    return {"ok": True, "result": result,
            "affected": _affected(revised=revised_ids),
            "mutations": {"nodes": {"revised": manifest_rows}, "edges": edge_rows}}


def _op_archive(brain, op_spec, top_encoding_source, graph_changes):
    """brain_batch `archive` op — soft-archive a node (guards, edges, vectors,
    audit all handled by archive_node). Returns archive_node's result plus the
    node-lifecycle `affected` split. Batch-only op (no standalone tool), hence
    a `_op_*` helper rather than a `_handle_*` handler."""
    node_id = op_spec.get("node_id")
    archived_by = _resolve_archived_by(op_spec, top_encoding_source)
    # survivor_id is validated HERE, before archive_node writes anything —
    # archive_node stores the _sys_archived_survivor_id pointer before any
    # existence check (trusted-caller contract), so a garbage survivor from
    # the agent-facing schema would otherwise yield ok=True with a dead
    # lineage pointer: resolve_live drops the node as a permanent orphan
    # (review 2026-08-07). Mirrors absorb's up-front survivor validation.
    survivor_id = op_spec.get('survivor_id') or None
    if survivor_id:
        if survivor_id == node_id:
            return {"ok": False, "node_id": node_id,
                    "error": "survivor_id must be a different node"}
        alive = brain.conn.execute(
            'SELECT 1 FROM nodes WHERE id = ? AND archived = 0',
            (survivor_id,)).fetchone()
        if not alive:
            return {"ok": False, "node_id": node_id,
                    "error": "survivor not found or archived: %s"
                             % str(op_spec.get('survivor_id'))[:24]}
    r = brain.archive_node(
        node_id, archived_by=archived_by, reason=op_spec.get('reason', ''),
        survivor_id=survivor_id)
    # Collections popped UNCONDITIONALLY — they exist for the manifest, and
    # **r splices everything left into the agent-visible results[]. The
    # scalar edges_deleted / vectors_deleted counts stay visible as before.
    edge_relations = r.pop('edge_relations', [])
    redirect = r.pop('absorbed_into_edge', None)
    if r.get('ok'):
        graph_changes.append("ARCHIVE: %s" % node_id[:8])
        r["affected"] = _affected(archived=[node_id])
        mutations = {"nodes": {"archived": [
            _archived_row(r, archived_by, edge_relations)]}}
        if redirect:
            # Supersession lineage: the absorbed_into redirect minted by
            # archive_node when survivor_id is passed — same edge-row shape
            # _op_absorb emits for its internal archive.
            mutations["edges"] = [_edge_row(
                redirect, 'absorbed_into',
                op_spec.get('reason', '') or 'archive supersession redirect',
                archived_by, redirect.get('source_id'),
                redirect.get('target_id'))]
        r["mutations"] = mutations
    return r


def _op_absorb(brain, op_spec, top_encoding_source, graph_changes):
    """brain_batch `absorb` op — merge: fold absorbed INTO survivor,
    then archive absorbed (transfer-by-default: source_refs, edges,
    access_count, KV — so a merge can't silently drop what the imperative
    revise+connect+archive path did). survivor may be locked; absorbed must
    not be. affected: survivor revised (content rewritten) + absorbed
    archived, so a merge-only run is no longer invisible to S2."""
    survivor_id = op_spec.get("survivor_id")
    absorbed_id = op_spec.get("absorbed_id")
    archived_by = _resolve_archived_by(op_spec, top_encoding_source)
    # Revise-op style: every non-control key is a survivor field override
    # (content, title, confidence, situation, ...), forwarded to absorb()'s
    # updates — same mental model as the revise op.
    _CONTROL = {'op', 'survivor_id', 'absorbed_id', 'prune_edges',
                'drop_fields', 'archived_by', 'encoding_source', 'reason',
                'session_id', 'chain_id'}
    field_updates = {k: v for k, v in op_spec.items() if k not in _CONTROL}
    r = brain.absorb(
        survivor_id, absorbed_id, updates=field_updates or None,
        archived_by=archived_by, reason=op_spec.get('reason', ''),
        prune_edges=op_spec.get('prune_edges'),
        drop_fields=op_spec.get('drop_fields'))
    # Popped UNCONDITIONALLY: on success they become the manifest; on a
    # savepoint-unwound merge the writes they describe rolled back, so they
    # must neither emit nor splice into the agent-visible results[].
    deltas = r.pop('deltas', [])
    migrated_edges = r.pop('migrated_edges', [])
    absorbed_archive = r.pop('absorbed_archive', None)
    if r.get('ok'):
        graph_changes.append("ABSORB: %s <- %s" % (
            survivor_id[:8], absorbed_id[:8]))
        r["affected"] = _affected(revised=[survivor_id], archived=[absorbed_id])
        # archived_by is what absorb actually stamped on the migrated edges
        # (brain_remember kw={'encoding_source': archived_by}) and it already
        # folds in op-level archived_by — a row carrying anything else could
        # contradict the graph and mis-route scale (review 2026-08-06).
        row_source = archived_by
        reason = (op_spec.get('reason', '')
                  or 'absorb %s into %s' % (absorbed_id[:8], survivor_id[:8]))
        # Survivor node_revised row + one edge row per migrated relation +
        # the absorbed node's archived row and its absorbed_into redirect
        # edge (both from archive_node's cascade results, via the report).
        # A merge with no field overrides has empty deltas; the emitter's
        # _changed gate keeps that row silent, matching the revise handlers.
        edge_rows = [_edge_row(e, e.get('relation', ''), reason, row_source,
                               e.get('source_id'), e.get('target_id'))
                     for e in migrated_edges]
        archived_rows = []
        if absorbed_archive:
            archived_rows.append(_archived_row(
                absorbed_archive, archived_by,
                absorbed_archive.get('edge_relations', [])))
            redirect = absorbed_archive.get('absorbed_into_edge')
            if redirect:
                edge_rows.append(_edge_row(
                    redirect, 'absorbed_into', reason, row_source,
                    redirect.get('source_id'), redirect.get('target_id')))
        r["mutations"] = {
            "nodes": {
                "revised": [{
                    'node_id': survivor_id,
                    'reason': reason,
                    'encoding_source': row_source,
                    'deltas': deltas,
                    'warnings': list(r.get('revise_warnings', [])),
                }],
                "archived": archived_rows,
            },
            "edges": edge_rows,
        }
    return r


def _op_disconnect(brain, op_spec, top_encoding_source, args, graph_changes):
    """brain_batch `disconnect` op — soft-archive ONE relation on an edge
    (other relations on the same edge survive; archived row kept for
    forensics/recovery). Edge-only — no node lifecycle, so no `affected`.
    The manifest row carries remove_relation's OBSERVED deltas: a disconnect
    of an already-archived or nonexistent relation has empty deltas and the
    emitter stays silent, where the legacy emit fabricated a 0→1 flip."""
    source_id = op_spec.get("source_id", "")
    target_id = op_spec.get("target_id", "")
    relation = op_spec.get("relation")
    archived_by = _resolve_archived_by(op_spec, top_encoding_source)
    # remove_relation gates its own commit on conn.in_batch (True here) →
    # deferred to the batch's single COMMIT. It returns edge_id itself — no
    # separate get_edge_id lookup.
    r = brain._graph.remove_relation(
        source_id, target_id, relation, archived_by=archived_by)
    if not r.get('edge_id'):
        # No edge between these ids at all — a miscopied id or stale pointer.
        # Loud, matching connect (add_relation raises) and revise (error dict);
        # an already-archived relation on an EXISTING edge stays a quiet no-op
        # (edge_id present, empty deltas — idempotent by design).
        return {"ok": False,
                "error": "disconnect: no edge between %r and %r"
                         % (source_id[:12], target_id[:12])}
    graph_changes.append("DISCONNECT: %s -[%s]-> %s" % (
        source_id[:8], relation, target_id[:8]))
    mutations = {"edges": [_edge_row(
        r, relation, op_spec.get('reason', '') or args.get('reason', ''),
        archived_by, source_id, target_id)]}
    return {"ok": True, "mutations": mutations}


def _handle_brain_batch(brain, args, graph_changes):
    """Execute multiple brain operations in one call.

    Accepts mixed operations: remember, revise, connect in any order.
    Each operation is validated and executed sequentially.
    Returns results for each operation.

    Args:
        operations: list of {op: "remember"|"revise"|"connect", ...fields}
    """
    from .contract import BATCH_OP_SPECS, unwrap_operations

    operations = args.get("operations", [])
    # Lossless unwrap of the JSON-string serialization quirk: LLM callers
    # occasionally emit the ops array JSON-encoded as a string. Iterating
    # that string would fan out one "must be a dict" error PER CHARACTER
    # (27k+ per-op errors and a 3.3MB tool result in one wasted encoder
    # round). The string holds the exact intended array, so recover it —
    # loudly, so the quirk's frequency stays visible — and collapse anything
    # unrecoverable into ONE whole-call error.
    if isinstance(operations, str):
        parsed = unwrap_operations(operations)
        if parsed is None:
            return {"ok": False, "error":
                    "operations must be a JSON array of op dicts; got a "
                    "string that does not parse to one. Emit the array as a "
                    "JSON value, not a JSON-encoded string. No ops were "
                    "executed."}
        try:
            brain._log_error(
                'brain_batch_string_ops',
                RuntimeError('operations arrived JSON-encoded as a string '
                             '(%d chars) — unwrapped to %d op(s)'
                             % (len(operations), len(parsed))),
                'lossless recovery, batch proceeded; caller emitted a '
                'stringified array')
        except Exception:
            pass
        operations = parsed
    if not isinstance(operations, list):
        return {"ok": False, "error": "operations must be an array, got %s"
                % type(operations).__name__}
    if not operations:
        return {"ok": False, "error":
                "operations array is required — if you have no operations "
                "to perform, do not call brain_batch at all; state your "
                "decision in plain text instead"}

    # Deterministic project provenance at the BATCH boundary. The remember/
    # revise sub-ops would be covered by their leaf handlers anyway, but ops
    # that don't delegate (absorb — whose field overrides flow into revise()
    # via brain.absorb(updates=...)) are only guarded here; without this an
    # agent-supplied `project` on an absorb op silently moves birth provenance.
    scope_warnings = _stamp_session_scope(brain, 'brain_batch', args)

    # Valid nested op names live in contract.VALID_BATCH_OPS (single source of
    # truth — see that constant). The dispatcher matches them via the if/elif
    # chain below; the final `else` is the invalid-op guard. Sonnet has been
    # observed inventing structural op names like `consolidate` / `keep` /
    # `evolve` / `skip` / `reject` — these are prompt-level
    # DECISIONS, not dispatch ops. They land in the "Unknown op" branch, get
    # logged loudly (brain_batch_invalid_op), and the S2 rejection_table now
    # detects them so a dropped op isn't mistaken for a clean SKIP.

    top_encoding_source = args.get("encoding_source")
    # The caller's identity, resolved once. Propagated to every sub-op under the
    # RESERVED key (never session_id) so attribution flows to the caller while
    # `session_id == filter` stays true at every layer — a sub-op can't be
    # accidentally scoped by the ambient identity. Assigned over whatever an op
    # carries ('' included), so an op's own `_caller_session` or session_id
    # never names another caller. Sub-handlers read it via caller_session();
    # remember additionally pops it through _pop_session_ctx.
    top_session_id = caller_session(args)
    # The batch's trace chain — injected into each sub-op so node_revised /
    # edge_relation_revised / co_anchored traces join the encoder's chain
    # instead of scattering to a date-fallback chain. Sub-handlers treat
    # chain_id as control (revise via DISPATCH_KEYS, remember pops it, connect
    # passes explicit kwargs) so it never leaks onto a node. NOTE: for manifest
    # traces the chain override is captured at the dispatch chokepoint from the
    # BATCH's args — a per-op chain_id is not honored (no producer sets one; if
    # that changes, carry it on the manifest row).
    top_chain_id = args.get('chain_id', '')
    results = []

    # Sibling-aware connect_to: defer per-op connect_to from `remember` ops
    # until ALL ops in this batch have run, so siblings declared in any order
    # can resolve. NEW wins on title collision (sibling beats catalog).
    sibling_map = {}  # lowercased title → new node_id
    deferred_connects = []  # [(src_node_id, connect_to_spec)]
    unwrapped_elements = 0  # string ops recovered via json.loads (logged once)

    # Node-lifecycle Δ aggregated across all sub-ops — the authoritative
    # `affected` the batch returns at top level (the runner reads it for the
    # encoding_run delta; the agent never sees it). Popped off each sub-result
    # so the per-op `results` array stays the agent-facing shape it always was.
    agg = {'created': [], 'revised': [], 'archived': []}

    def _accumulate(aff):
        if aff:
            agg['created'].extend(aff.get('created') or [])
            agg['revised'].extend(aff.get('revised') or [])
            agg['archived'].extend(aff.get('archived') or [])

    # Mutation manifest aggregated across sub-ops, same pop-off-each-sub-result
    # pattern as `affected` — the rows must NOT stay in the per-op `results`
    # (the agent-facing shape) and must return at TOP level, where
    # dispatch_command emits them AFTER this handler's single commit. That
    # post-commit hop is the point: the legacy inline emits fired mid-envelope
    # and could orphan a trace for a batch that then rolled back. revise and
    # the edge ops (connect, disconnect) contribute today; steps 6-7 add the
    # remaining slots.
    mutation_rows = {'nodes': {}, 'edges': []}

    def _accumulate_mutations(m):
        # Slot-agnostic on purpose: a sub-op returning a node slot this code
        # predates (archived/deleted land at steps 7-8) accumulates instead of
        # silently dropping — the emitter's MANIFEST_TRACE_MAP is the one
        # place that decides which slots exist.
        if m:
            for slot, rows in (m.get('nodes') or {}).items():
                mutation_rows['nodes'].setdefault(slot, []).extend(rows or [])
            mutation_rows['edges'].extend(m.get('edges') or [])

    # Wrap the whole batch in ONE SQLite transaction. Sub-handlers and DAL
    # writers gate their commits on brain.conn.in_batch (commit_unless_batched);
    # setting it True here makes every per-op commit a no-op so the outer
    # BEGIN IMMEDIATE / COMMIT below owns the single durability point.
    # Pre-change every op committed individually, so a batch of N ops hit
    # the WAL writer slot N times — bad for parallel-session contention,
    # no rollback semantic if op #37 broke. Per-op exceptions are still
    # caught by the inner try/except below and recorded in `results` —
    # that "best-effort" surface is preserved. Only when something
    # escapes the per-op handler entirely (a programmer bug, not a
    # caller-visible op failure) do we rollback the whole batch.
    brain.conn.in_batch = True
    transaction_started = False
    # brain_batch owns its own BEGIN IMMEDIATE / COMMIT envelope. If self.conn
    # is ALREADY mid-transaction here, an upstream op left a deferred auto-BEGIN
    # open without committing (self.conn uses Python's default deferred
    # isolation). Inheriting it would (a) make the explicit BEGIN IMMEDIATE
    # below throw "cannot start a transaction within a transaction" and (b)
    # silently fold that op's uncommitted writes into THIS batch's commit.
    # Flush the orphan cleanly and log loud so the upstream leak stays visible.
    if brain.conn.in_transaction:
        try:
            brain.conn.commit()
        except Exception:
            try:
                brain.conn.rollback()
            except Exception:
                pass
            # The leaked op may have been a cache-mutating revise/archive; its
            # eager cache drop survives this rollback, so resync (same reason
            # as the batch-exception rollback below).
            brain._resync_vector_cache('brain_batch entry-flush rollback')
        print('[brain_batch] WARN: flushed a leaked transaction at entry — an '
              'upstream write did not commit (self.conn was mid-transaction)',
              flush=True)
        try:
            brain._log_error(
                'brain_batch_stale_txn',
                RuntimeError('self.conn was mid-transaction at brain_batch entry'),
                'flushed leaked txn before BEGIN IMMEDIATE; upstream op that ran '
                'just before brain_batch did not commit')
        except Exception:
            pass
    try:
        brain.conn.execute('BEGIN IMMEDIATE')
        transaction_started = True

        for i, op_spec in enumerate(operations):
            # Same serialization quirk one level down: a string ELEMENT that
            # parses to a dict is the intended op. Unrecoverable elements keep
            # the per-op error — fan-out is bounded by the element count the
            # caller actually sent.
            op_spec, unwrapped = _element_as_dict(op_spec)
            if unwrapped:
                unwrapped_elements += 1
            if op_spec is None:
                results.append({"op": "?", "index": i, "ok": False,
                                "error": "operation must be a dict, got %s" % type(operations[i]).__name__})
                continue
            op = op_spec.get("op", "")

            # Per-op required-field pre-check, derived from the SAME contract
            # the MCP oneOf schema is built from (contract.BATCH_OP_SPECS) —
            # schema signal at generation time, this check at dispatch time,
            # one source. This is the ONLY missing-field gate for batch ops
            # (the old in-branch guards were removed as unreachable;
            # tests/test_brain_batch_op_contract.py pins enforcement per
            # field). Unknown op names fall through to the invalid-op guard
            # below. revise keeps its rich reason/reasoning disambiguation
            # (_missing_reason_error) instead of the generic message.
            _op_contract = BATCH_OP_SPECS.get(op)
            if _op_contract:
                missing = [f for f in _op_contract["required"]
                           if not op_spec.get(f)]
                if missing:
                    if op == "revise" and "reason" in missing:
                        # Keep the reason/reasoning disambiguation whenever
                        # reason is among the missing fields — not only when
                        # it's the sole one (review 2026-06-12 #2).
                        err = _missing_reason_error(op_spec)
                        others = [f for f in missing if f != "reason"]
                        if others:
                            err += " Also missing: %s." % ", ".join(others)
                    else:
                        err = ("%s op missing required field(s): %s — "
                               "per-op required fields are declared in the "
                               "brain_batch schema, one branch per op"
                               % (op, ", ".join(missing)))
                    results.append({"op": op, "index": i, "ok": False,
                                    "error": err})
                    continue

            try:
                if op == "remember":
                    refusal = community_create_refusal(op_spec)
                    if refusal:
                        results.append({"op": op, "index": i, "ok": False,
                                        "error": refusal})
                        continue
                    # Pop per-op connect_to BEFORE handler so it's not processed
                    # eagerly with an empty sibling_map — defer to after the loop.
                    ct_spec = op_spec.get("connect_to")
                    op_args = {k: v for k, v in op_spec.items()
                               if k not in ("op", "connect_to")}
                    if top_encoding_source and "encoding_source" not in op_args:
                        op_args["encoding_source"] = top_encoding_source
                    op_args[CALLER_SESSION_KEY] = top_session_id
                    if top_chain_id and "chain_id" not in op_args:
                        op_args["chain_id"] = top_chain_id
                    r = _handle_remember(brain, op_args, graph_changes)
                    _accumulate(r.pop("affected", None))
                    _accumulate_mutations(r.pop("mutations", None))
                    results.append({"op": "remember", "index": i, **r})
                    # Capture for sibling_map + deferred resolution
                    if r.get("ok"):
                        inner = r.get("result") or {}
                        new_id = inner.get("id")
                        if new_id:
                            title = (op_args.get("title") or "").lower()
                            if title:
                                sibling_map[title] = new_id
                            if ct_spec:
                                deferred_connects.append((new_id, ct_spec))

                elif op == "revise":
                    op_args = {k: v for k, v in op_spec.items() if k != "op"}
                    if top_encoding_source and "encoding_source" not in op_args:
                        op_args["encoding_source"] = top_encoding_source
                    op_args[CALLER_SESSION_KEY] = top_session_id
                    if top_chain_id and "chain_id" not in op_args:
                        op_args["chain_id"] = top_chain_id
                    r = _handle_revise(brain, op_args, graph_changes)
                    _accumulate(r.pop("affected", None))
                    _accumulate_mutations(r.pop("mutations", None))
                    results.append({"op": "revise", "index": i, **r})

                elif op == "connect":
                    op_args = {k: v for k, v in op_spec.items() if k != "op"}
                    if top_encoding_source and "encoding_source" not in op_args:
                        op_args["encoding_source"] = top_encoding_source
                    op_args[CALLER_SESSION_KEY] = top_session_id
                    if top_chain_id and "chain_id" not in op_args:
                        op_args["chain_id"] = top_chain_id
                    r = _handle_connect(brain, op_args, graph_changes)
                    _accumulate(r.pop("affected", None))
                    _accumulate_mutations(r.pop("mutations", None))
                    results.append({"op": "connect", "index": i, **r})

                elif op == "archive":
                    # node_id presence guaranteed by the BATCH_OP_SPECS pre-check.
                    r = _op_archive(brain, op_spec, top_encoding_source, graph_changes)
                    _accumulate(r.pop("affected", None))
                    _accumulate_mutations(r.pop("mutations", None))
                    results.append({"op": "archive", "index": i, **r})

                elif op == "absorb":
                    # id presence guaranteed by the BATCH_OP_SPECS pre-check.
                    r = _op_absorb(brain, op_spec, top_encoding_source, graph_changes)
                    _accumulate(r.pop("affected", None))
                    _accumulate_mutations(r.pop("mutations", None))
                    results.append({"op": "absorb", "index": i, **r})

                elif op == "disconnect":
                    # field presence guaranteed by the BATCH_OP_SPECS pre-check.
                    r = _op_disconnect(brain, op_spec, top_encoding_source,
                                       args, graph_changes)
                    _accumulate(r.pop("affected", None))
                    _accumulate_mutations(r.pop("mutations", None))
                    results.append({"op": "disconnect", "index": i, **r})

                else:
                    # Invalid op name — log loudly. Sonnet sometimes invents
                    # structural names (consolidate/keep/evolve/skip) that were
                    # never valid ops. Previously this returned ok=False and the
                    # caller moved on silently.
                    err_msg = ("Unknown op: %s (use remember, revise, connect, disconnect, archive, absorb)"
                               % op)
                    try:
                        brain._log_error(
                            'brain_batch_invalid_op',
                            ValueError(err_msg),
                            'op_spec=%s' % str(op_spec)[:300])
                    except Exception:
                        pass
                    results.append({"op": op, "index": i, "ok": False, "error": err_msg})
            except Exception as e:
                results.append({"op": op, "index": i, "ok": False, "error": str(e)[:200]})

        # Pass 2: deferred per-op connect_to resolution. Runs AFTER all ops so
        # siblings declared in any order resolve correctly. _apply_connect_to
        # logs all failures to debug_log; this is sequencing-agnostic and never
        # raises. Returns (edges_created, failures_logged) — both surfaced in
        # the batch result so a cycle with N requested connect_to and 0 edges
        # has a visible "connect_to_failures=N" reason.
        connect_to_edges = 0
        connect_to_failed = []  # [{title, reason}]
        connect_to_made = []    # [{src_id, target_id, relation, edge_id, deltas}]
        for src_id, ct_spec in deferred_connects:
            # The src node was created by a remember op in this batch, which
            # inherited top_encoding_source (or 'anchor' for direct MCP) — give
            # its connect_to edges the same provenance.
            r = brain._apply_connect_to(
                src_id, ct_spec, sibling_map=sibling_map,
                encoding_source=top_encoding_source or 'anchor',
                exclude_ids=set(sibling_map.values()))
            connect_to_edges += len(r['created'])
            connect_to_made.extend(r['created'])  # entries src-tagged by _apply_connect_to
            connect_to_failed.extend(r['failed'])
        if connect_to_edges:
            graph_changes.append("CONNECT_TO: %d edges" % connect_to_edges)
        if connect_to_failed:
            graph_changes.append("CONNECT_TO_FAILURES: %d" % len(connect_to_failed))

        if unwrapped_elements:
            # One aggregate log per batch, not one per element — same
            # visibility contract as the top-level unwrap above.
            try:
                brain._log_error(
                    'brain_batch_string_ops',
                    RuntimeError('%d op element(s) arrived JSON-encoded as '
                                 'strings — unwrapped losslessly'
                                 % unwrapped_elements),
                    'lossless recovery, batch proceeded; caller emitted '
                    'stringified op dicts')
            except Exception:
                pass

        # One commit for the whole batch — all per-op writes land here. Edge
        # traces are emitted AFTER this (post-finally), never before: traces
        # live in a different DB, so emitting pre-commit would orphan them if
        # the batch rolled back.
        brain.conn.commit()
    except Exception as e:
        # Per-op exceptions are caught above and recorded in `results` — reaching
        # this handler means something escaped the per-op guard (programmer bug
        # in a sub-handler, lost DB connection, etc.). Roll back so we don't
        # leave half a batch persisted.
        if transaction_started:
            try:
                brain.conn.rollback()
            except Exception as re:
                try:
                    brain._log_error(
                        'brain_batch_rollback_failed', re,
                        'rollback after batch exception failed')
                except Exception:
                    pass
            # revise/archive/absorb ops mutate the vector cache eagerly
            # mid-envelope; the rollback restored the DB but not the cache —
            # resync or live nodes stay invisible until restart.
            # _resync_vector_cache is already total (getattr-guarded, internal
            # try/except → _log_error); no second swallow layer here.
            brain._resync_vector_cache('brain_batch rollback')
        try:
            brain._log_error(
                'brain_batch_transaction_failed', e,
                'ops=%d ran_before_fail=%d' % (
                    len(operations), len(results)))
        except Exception:
            pass
        # Re-raise so the dispatcher reports the failure to the caller. The
        # daemon's outer dispatch wraps this in its own try/except and turns
        # it into {ok: False, error: ...}.
        raise
    finally:
        brain.conn.in_batch = False

    # POST-COMMIT: the batch is durable (any exception above re-raised past
    # this point, discarding mutation_rows with it). The deferred connect_to
    # edges join the manifest here and are emitted by the chokepoint after
    # this handler returns — still post-commit, now on the one emit path.
    # (co_anchored edges are noise-aspect: popped, never traced.)
    mutation_rows['edges'].extend(_connect_to_rows(
        {'created': connect_to_made}, top_encoding_source or ''))

    succeeded = sum(1 for r in results if r.get("ok"))
    _batch_result = {
        "total": len(operations),
        "succeeded": succeeded,
        "failed": len(operations) - succeeded,
        "connect_to_edges": connect_to_edges,
        "connect_to_failures": len(connect_to_failed),
        "connect_to_failed": connect_to_failed,
        "results": results,
    }
    if scope_warnings:
        _batch_result["warnings"] = scope_warnings
    return {"ok": True, "result": _batch_result,
            "affected": _affected(created=agg['created'], revised=agg['revised'],
                                  archived=agg['archived']),
            "mutations": mutation_rows}


def _handle_connect(brain, args, graph_changes):
    # Resolve the edge's CREATOR. The encoder pre-stamps args['encoding_source']
    # at its dispatch boundary (apply_encoder_attribution); a direct-MCP connect
    # arrives with none — and an unstamped write reaching this handler IS anchor
    # by definition (every other writer stamps upstream). So `or 'anchor'` mirrors
    # the node birth default (remember(): `encoding_source or 'anchor'`). This is
    # applied at CREATE only — add_relation preserves encoding_source on an
    # active-row update — so re-connecting an existing edge never relabels it.
    # No silent default: an unnamed relation used to become `related_to`,
    # the one verb every layer bans (0.2× lift, pollutes the activation
    # kernel). Missing relation fails loudly instead.
    relation = args.get("relation")
    if not relation:
        return {"ok": False,
                "error": "relation is required — a specific verb; nothing "
                         "defaults to related_to any more"}
    encoding_source = args.get("encoding_source") or 'anchor'
    src_id = args.get("source_id", "")
    tgt_id = args.get("target_id", "")
    result = brain.connect_typed(
        source_id=src_id,
        target_id=tgt_id,
        relation=relation,
        weight=args.get("weight"),
        description=args.get("description"),
        encoding_source=encoding_source)

    graph_changes.append(
        "CONNECT: %s -[%s]-> %s" % (
            args.get("source_id", "?")[:8],
            relation,
            args.get("target_id", "?")[:8]))
    # Manifest row for the emitter — its _changed gate keeps an idempotent
    # re-connect (no deltas, no warnings) silent, as the legacy emit did.
    mutations = None
    if result:
        mutations = {"edges": [_edge_row(
            result, relation, args.get('reason', ''),
            encoding_source, src_id, tgt_id)]}
    return {"ok": True, "result": {"connected": True}, "mutations": mutations}


def _handle_revise_edge(brain, args, graph_changes):
    """Revise an existing edge relation in place — rename and/or update
    description/weight. Identify by (source_id, target_id, relation); omitted
    fields preserve (mirrors revise()). Routes the rename to the in-place
    GraphDAL.rename_relation primitive, not a connect+disconnect pair."""
    source_id = args.get("source_id", "")
    target_id = args.get("target_id", "")
    relation = args.get("relation")
    if not (source_id and target_id and relation):
        return {"ok": False, "error": "source_id, target_id, relation are required"}

    result = brain.revise_edge(
        source_id=source_id, target_id=target_id, relation=relation,
        new_relation=args.get("new_relation"),
        description=args.get("description"),
        weight=args.get("weight"),
        encoding_source=args.get("encoding_source"),
        reason=args.get("reason", ""))
    if not result.get("ok"):
        return result

    # The pair in args names the claim in either order; the manifest carries
    # the claim's own ends, which revise_edge returns.
    graph_changes.append(
        "REVISE_EDGE: %s -[%s]-> %s" % (
            str(result["source_id"])[:8], result["relation"],
            str(result["target_id"])[:8]))
    return {"ok": True, "result": result,
            "mutations": {"edges": [_edge_row(
                result, result["relation"], args.get("reason", ""),
                args.get("encoding_source") or "",
                result["source_id"], result["target_id"])]}}


def _handle_connect_batch(brain, args, graph_changes):
    """Create multiple edges in one call."""
    connections = args.get("connections", [])
    if not connections:
        return {"ok": False, "error": "connections array is required"}

    top_encoding_source = args.get('encoding_source', '')
    edge_rows = []

    # Known per-connection keys — used to build a "did you mean" hint when a
    # caller sends an aliased key (e.g. from_id/to_id) instead of the canonical
    # source_id/target_id. We keep ONE canonical name (no aliasing) but make the
    # error self-correcting.
    KNOWN_CONN_KEYS = {"source_id", "target_id", "relation", "weight",
                       "description", "encoding_source", "reason"}

    created = 0
    updated = 0
    failure_details = []  # [{source_id, target_id, relation, reason}]
    for i, c in enumerate(connections):
        c, unwrapped = _element_as_dict(c)
        if c is None:
            failure_details.append({
                "source_id": "", "target_id": "", "relation": "",
                "reason": "connections[%d] must be an object, got %s" % (i, type(connections[i]).__name__)})
            continue
        if unwrapped:
            _warn_string_element(brain, 'connect_batch', 'connections', i)
        relation = c.get("relation", "")
        src_raw = c.get("source_id", "")
        tgt_raw = c.get("target_id", "")
        # Self-correcting guard: if an endpoint is missing AND the caller sent
        # keys we don't recognize, the reason names them and suggests the fix.
        if not src_raw or not tgt_raw:
            unknown = [k for k in c if k not in KNOWN_CONN_KEYS]
            hint = ""
            if unknown:
                hint = (" — unrecognized key(s) %s; edges use 'source_id' and "
                        "'target_id'" % unknown)
            failure_details.append({
                "source_id": str(src_raw)[:8], "target_id": str(tgt_raw)[:8],
                "relation": relation,
                "reason": "missing source_id or target_id%s" % hint})
            brain._log_error(
                'connect_batch_missing_endpoint',
                ValueError("missing source_id/target_id%s" % hint),
                'keys=%s' % sorted(c.keys()))
            continue
        # No silent default: an unnamed relation used to become `related_to`,
        # the one verb every layer bans. Skip the edge, say why.
        if not relation:
            failure_details.append({
                "source_id": str(src_raw)[:8], "target_id": str(tgt_raw)[:8],
                "relation": "",
                "reason": "relation is required — a specific verb; nothing "
                          "defaults to related_to any more"})
            brain._log_error(
                'connect_batch_missing_relation',
                ValueError("missing relation: %s -> %s"
                           % (str(src_raw)[:8], str(tgt_raw)[:8])),
                'edge skipped')
            continue
        try:
            # Resolve the edge's CREATOR (per-row → batch-level → 'anchor'),
            # mirroring _handle_connect. Applied at CREATE only — add_relation
            # preserves encoding_source on an active-row update — so a re-connect
            # never relabels an existing edge.
            encoding_source = (c.get("encoding_source")
                               or top_encoding_source or 'anchor')
            result = brain.connect_typed(
                source_id=src_raw,
                target_id=tgt_raw,
                relation=relation,
                weight=c.get("weight"),
                description=c.get("description"),
                encoding_source=encoding_source)
            # A re-connect of an existing edge is an update (or a no-op), not
            # a creation — the counts say which happened.
            if (result or {}).get('created'):
                created += 1
            elif (result or {}).get('updated'):
                updated += 1
            # One manifest row per connection, carrying the per-row resolved
            # encoding_source; the emitter's _changed gate drops no-op rows.
            if result:
                edge_rows.append(_edge_row(
                    result, relation,
                    c.get('reason', '') or args.get('reason', ''),
                    encoding_source, src_raw, tgt_raw))
        except Exception as e:
            # No silent drops: a failed edge in a batch must surface, not vanish
            # — and now with its REASON in the response, not just the dashboard.
            failure_details.append({
                "source_id": str(src_raw)[:8], "target_id": str(tgt_raw)[:8],
                "relation": relation, "reason": str(e)[:160]})
            brain._log_error(
                'connect_batch_edge_failed', e,
                '%s -[%s]-> %s' % (str(src_raw)[:8], relation, str(tgt_raw)[:8]))
    graph_changes.append("CONNECT_BATCH: %d created, %d updated" % (created, updated))
    if failure_details:
        graph_changes.append("CONNECT_BATCH_FAILURES: %d" % len(failure_details))
    return {"ok": True, "result": {
        "edges_created": created,
        "edges_updated": updated,
        "failures": len(failure_details),
        "failure_details": failure_details,
    }, "mutations": {"edges": edge_rows}}


def _handle_set_node_lock(brain, args, graph_changes):
    """Flip a node's locked flag via the two-phase confirm door
    (brain.set_node_lock). Operator-channel only — the encoder dispatch
    closure refuses this command; it is deliberately NOT a brain_batch op."""
    node_id = args.get("node_id", "")
    reason = args.get("reason", "")
    if not node_id:
        return {"ok": False, "error": "node_id is required"}
    if not isinstance(args.get("locked"), bool):
        # bool('false') is True — a string here silently inverts the request,
        # so the type check is a guard, not pedantry.
        return {"ok": False, "error": "locked must be a JSON boolean "
                                      "(true/false), got %r" % (args.get("locked"),)}
    if not reason:
        return {"ok": False, "error": "reason is required — why is this node "
                                      "being locked/unlocked?"}

    result = brain.set_node_lock(
        node_id=node_id, locked=args["locked"], reason=reason,
        confirm_token=args.get("confirm_token"),
        encoding_source=args.get("encoding_source") or 'anchor',
        session_id=caller_session(args))
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "set_node_lock failed")}
    if not result.get("changed"):
        # Phase 1 (confirmation_required) or no-op — nothing flipped, no trace.
        return {"ok": True, "result": result}

    graph_changes.append("LOCK: %s %s" % (
        "locked" if result.get("locked") else "unlocked", node_id[:12]))
    # node_lock_changed manifest row → the mutation emitter writes the loud
    # trace. A dedicated lifecycle ref_type, NOT node_revised: revise history
    # must never show `locked` — the field revise() treats as immutable — as
    # a revised field.
    row = {
        "node_id": node_id,
        "type": result.get("type", ""),
        "title": result.get("title", ""),
        "locked": result.get("locked", False),
        "reason": reason,
        "encoding_source": args.get("encoding_source", ""),
        "confirm_latency_s": result.get("confirm_latency_s", 0.0),
    }
    return {"ok": True, "result": result,
            "affected": _affected(revised=[node_id]),
            "mutations": {"nodes": {"lock_changed": [row]}}}
