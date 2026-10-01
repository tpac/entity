"""brain — Mutation-Trace Emitter

THE one writer of graph-mutation traces. Every node born, revised, archived or
erased, and every edge relation touched, becomes a trace row here and nowhere
else. Called from `daemon_dispatch.dispatch_command` — the single execution
chokepoint — after the handler returns.

Design record: docs/MUTATION-EMITTER-DESIGN.md. Work order:
docs/MUTATION-EMITTER-PLAN.md.

WHAT REPLACED WHAT
    Before, eleven hand-rolled `_emit_*` call sites in dispatch_write.py plus an
    inline trace in brain_remember.archive_node each decided their own timing,
    scale, chain and attribution. Ten of the twelve could emit while brain.conn
    was still inside the brain_batch envelope, producing a durable trace for a
    graph write that then rolled back. Creates emitted nothing at all.

THE UNIFICATION IS DATA, NOT CODE
    One table (`MANIFEST_TRACE_MAP`), one loop. Everything that used to vary
    per site is a column: ref_type, ref_id shape, metadata builder, emit
    predicate, chain fallback. Adding a sixth mutation kind must be one table
    row plus one builder in trace_contract — if it needs a branch in this file,
    the unification is fake. `tests/test_mutation_emitter.py` pins that.

WHAT THIS FILE GUARANTEES
    * POST-COMMIT, and it CHECKS rather than assumes (see the in_transaction
      gate). The stated asymmetry: a missing trace is recoverable from the graph;
      an orphaned trace lies about it. Prefer missing.
    * Per-row scale and chain, derived from each row's OWN encoding_source — one
      brain_batch can carry an s2:consolidation archive and an anchor revise, and
      they must not land on the same chain.
    * One append_batch per command, however many mutations it carried. Row
      VOLUME is unchanged; only the commit count collapses.
    * Never raises into the caller. A trace failure is loud (brain._log_error)
      and the write still succeeds — the graph is the truth.

FOR WHOEVER WIRES STEPS 4-7
    A manifest row must be BUILDER-SHAPED: its keys are the builder's kwargs for
    that slot, and slots differ (a revise row has deltas/warnings, a created row
    has type/title). Unknown keys are dropped and logged rather than fatal — but
    that logging is per command, so putting bookkeeping fields on rows (`op`,
    `index`, sub-op scaffolding) turns the error feed into noise. Strip them where
    the manifest is assembled, not here.
"""

from .clock import brain_today
from .trace_contract import (
    build_edge_revise_metadata,
    build_lock_change_metadata,
    build_node_archived_metadata,
    build_node_created_metadata,
    build_revise_metadata,
    validate_trace_event,
)


# Rows carrying real before/after data are only worth a trace when something
# actually changed — an idempotent re-connect (empty deltas, no warnings) stays
# silent, exactly as the hand-rolled emitters behaved.
def _changed(row):
    return bool(row.get('deltas') or row.get('warnings'))


def _always(row):
    return True


def _node_ref_id(row):
    return row.get('node_id') or ''


def _edge_ref_id(row):
    # Composite, and no reader splits it — pinned by the reader map in the
    # design record §6. edge_id alone is ambiguous: one physical edge carries
    # many relations, each with its own lifecycle.
    return '%s:%s' % (row.get('edge_id') or '', row.get('relation') or '')


def _changed_summary(field_fmt, noop):
    """Summary for deltas/warnings-carrying rows — the exact strings the
    hand-rolled emitters wrote ('revised N field(s): …' for nodes,
    'N field(s): …' for edges), so trace-stream readers see no drift."""
    def _summary(row):
        parts = []
        if row.get('deltas'):
            parts.append(field_fmt % (
                len(row['deltas']),
                ', '.join(d.get('field', '?') for d in row['deltas'])))
        if row.get('warnings'):
            parts.append('%d warning(s)' % len(row['warnings']))
        return '; '.join(parts) if parts else noop
    return _summary


def _lifecycle_summary(verb):
    """Summary for created/archived rows: the verb + what it hit."""
    def _summary(row):
        return '%s [%s] %s' % (verb, row.get('type') or '?',
                               (row.get('title') or '')[:50])
    return _summary


def _lock_summary(row):
    """Lock flips pick their verb from the row's new state."""
    return '%s [%s] %s' % ('locked' if row.get('locked') else 'unlocked',
                           row.get('type') or '?', (row.get('title') or '')[:50])


# The whole mapping, and the only place mutation kinds are enumerated.
#   path       where the rows live in the manifest
#   ref_type   the registered trace ref_type
#   builder    trace_contract builder — a manifest row is filtered to its kwargs
#              (_builder_kwargs); unknown keys are dropped and logged, not fatal
#   ref_id     how to address the mutated thing
#   emit_when  predicate; False means "nothing changed, stay silent"
#   summary    human line for trace-stream readers (metadata stays canonical)
MANIFEST_TRACE_MAP = (
    (('nodes', 'created'),  'node_created',          build_node_created_metadata,  _node_ref_id, _always,  _lifecycle_summary('created')),
    (('nodes', 'revised'),  'node_revised',          build_revise_metadata,        _node_ref_id, _changed, _changed_summary('revised %d field(s): %s', 'revise no-op')),
    (('nodes', 'archived'), 'node_archived',         build_node_archived_metadata, _node_ref_id, _always,  _lifecycle_summary('archived')),
    (('nodes', 'lock_changed'), 'node_lock_changed', build_lock_change_metadata,   _node_ref_id, _always,  _lock_summary),
    (('edges',),            'edge_relation_revised', build_edge_revise_metadata,   _edge_ref_id, _changed, _changed_summary('%d field(s): %s', 'edge revise no-op')),
)

# The two ref_types that predate the emitter keep their date-fallback chain
# suffix so existing readers stay bit-compatible: the dashboard classifies any
# chain ending `-revise` as kind `revise`, and `_stop_of` yields None on it.
# The new lifecycle types must NOT land there — a creation or an archive
# rendering as "Refined N memories" is a lie — so they fall back to `-mutation`.
_REVISE_FALLBACK_REF_TYPES = frozenset({'node_revised', 'edge_relation_revised'})


def _accepted_kwargs(builder):
    """The keyword names a builder accepts — computed once, at import."""
    import inspect
    return frozenset(inspect.signature(builder).parameters)


# builder -> accepted kwarg names, so a manifest row can be filtered rather than
# splatted blind. Each slot's rows have a DIFFERENT key set (a revise row has
# deltas/warnings, a created row has type/title), and a spurious key must not
# cost the whole command its traces — see _builder_kwargs.
_BUILDER_KWARGS = {rt: _accepted_kwargs(b)
                   for _p, rt, b, _r, _w, _s in MANIFEST_TRACE_MAP}


def _builder_kwargs(ref_type, row, row_source, drops):
    """Filter a manifest row down to its builder's kwargs.

    Deliberately NOT strict. Splatting the row directly means one unexpected key
    raises TypeError and the loud-wrap drops EVERY trace for that command — far
    too much blast radius for a spurious field, and steps 4-7 hand-write these
    rows across eight handlers. So: drop unknown keys, record them, and let the
    caller log them. Same severity choice `check_unknown_keys` makes for dispatch
    args — loud, never blocking. A trace missing an optional field is still true;
    no traces at all is not.
    """
    accepted = _BUILDER_KWARGS[ref_type]
    unknown = set(row) - accepted
    if unknown:
        drops.append('%s: %s' % (ref_type, sorted(unknown)))
    kwargs = {k: v for k, v in row.items() if k in accepted}
    kwargs['encoding_source'] = row_source
    return kwargs


def _rows_at(manifest, path):
    """Pull one slot's rows out of the manifest, tolerating absent slots."""
    cur = manifest
    for key in path:
        if not isinstance(cur, dict):
            return []
        cur = cur.get(key)
    return cur if isinstance(cur, list) else []


def _scale_and_chain(brain, encoding_source, chain_id_override, ref_type):
    """Resolve (scale, chain_id) for ONE row from that row's encoding_source.

    Scale inference is the rule the hand-rolled emitters used
    ('s2:*' -> s2, 'encoder:*' -> s1, else s0), now in one place. A
    caller-provided chain always wins — encoder runs pass theirs so a
    mutation joins the run that caused it instead of a date bucket.
    """
    src = encoding_source or ''
    if src.startswith('s2:'):
        scale = 's2'
    elif src.startswith('encoder:'):
        scale = 's1'
    else:
        scale = 's0'
    if chain_id_override:
        return scale, chain_id_override
    suffix = 'revise' if ref_type in _REVISE_FALLBACK_REF_TYPES else 'mutation'
    return scale, '%s-%s-%s' % (scale, brain_today(brain).strftime('%Y%m%d'), suffix)


def build_events(brain, manifest, *, session_id='', chain_id='',
                 encoding_source='', drops=None):
    """Turn one command's manifest into trace-event dicts. Pure; no I/O.

    Split out from the emit path so tests can assert the mapping without a DB,
    and so the row-completeness check happens before anything is written.

    Raises only on rows that cannot become a truthful trace — a non-dict row, an
    unusable ref_id, an unregistered ref_type. All-or-nothing per command in that
    case: no traces, never a partial set. Unknown row KEYS are not in that class;
    they are dropped and appended to `drops` for the caller to log.
    """
    if drops is None:
        drops = []
    events = []
    for path, ref_type, builder, ref_id_of, emit_when, summary_of in MANIFEST_TRACE_MAP:
        for row in _rows_at(manifest, path):
            if not isinstance(row, dict):
                raise TypeError('manifest %s row must be a dict, got %s'
                                % ('.'.join(path), type(row).__name__))
            if not emit_when(row):
                continue
            ref_id = ref_id_of(row)
            if not ref_id or ref_id.startswith(':') or ref_id.endswith(':'):
                raise ValueError('manifest %s row has an unusable ref_id %r; '
                                 'the mutated thing must be identifiable or the '
                                 'trace is unresolvable' % ('.'.join(path), ref_id))
            # Per-row: the row's own source decides scale AND chain. A single
            # brain_batch legitimately spans scales.
            row_source = row.get('encoding_source') or encoding_source or ''
            scale, row_chain = _scale_and_chain(brain, row_source, chain_id, ref_type)
            ok, err = validate_trace_event(scale, 'delta', ref_type)
            if not ok:
                raise ValueError('emitter would write an unregistered trace: %s' % err)
            metadata = builder(**_builder_kwargs(ref_type, row, row_source, drops))
            events.append({
                'chain_id': row_chain,
                'scale': scale,
                'event_type': 'delta',
                'ref_type': ref_type,
                'ref_id': ref_id,
                'summary': summary_of(row),
                'metadata': metadata,
                'session_id': session_id or '',
                'interaction_id': None,
            })
    return events


def emit_mutation_traces(brain, cmd, manifest, *, session_id='', chain_id='',
                         encoding_source=''):
    """Emit every trace for one dispatched command. Never raises.

    Call AFTER the handler returns and INSIDE the caller's write lock (see
    dispatch_command's docstring for why the lock matters).
    """
    try:
        if not manifest:
            return

        # THE TIMING GATE. The design's post-commit property is conditional, not
        # structural: handlers CAN return with brain.conn mid-transaction. The
        # proof is in-tree — brain_batch has an entry-flush guard and a
        # `brain_batch_stale_txn` error row precisely because upstream writes
        # leak open deferred transactions (MetadataKVDAL.set_many doesn't
        # commit; archive_dangling_edges had none until step 9). Emitting now
        # would publish a durable trace for a graph write that can still roll
        # back — the one thing this design refuses to do. So we check, and skip:
        # a missing trace is recoverable, an orphaned one is a lie.
        #
        # This makes the emitter a DETECTOR of the transaction-leak class, which
        # is worth more than the traces it declines to write.
        if brain.conn.in_transaction:
            brain._log_error(
                'mutation_trace_txn_open',
                RuntimeError('%s returned with an open transaction on brain.conn; '
                             'mutation traces skipped to avoid orphaning them' % cmd),
                'cmd=%s; an upstream write leaked a transaction — the traces are '
                'lost but the graph is intact' % cmd)
            return

        drops = []
        events = build_events(brain, manifest, session_id=session_id,
                              chain_id=chain_id, encoding_source=encoding_source,
                              drops=drops)
        if drops:
            # Loud, non-blocking: the rows still went out, but a manifest key the
            # builder doesn't accept means the producing handler and the contract
            # have drifted — usually a typo silently emptying a field.
            brain._log_error(
                'mutation_trace_unknown_keys',
                ValueError('cmd=%s dropped manifest keys: %s' % (cmd, '; '.join(drops))),
                'the trace was still written without them')
        if not events:
            return
        brain._trace_dal.append_batch(events)
    except Exception as e:
        # Loud-wrapped as a unit: a trace failure must never reach the caller,
        # because the graph write already succeeded and re-raising would report a
        # successful mutation as failed. The error row is the alarm.
        try:
            brain._log_error(
                'mutation_trace_emit', e,
                'cmd=%s; manifest=%s' % (cmd, str(manifest)[:400]))
        except Exception:
            import sys
            print('[mutation_emitter] emit AND error-logging both failed for %s: %s'
                  % (cmd, e), file=sys.stderr, flush=True)


def edge_flip_rows(conn, flipped, encoding_source, reason):
    """Shape bulk_archive_relations' [edge_id, relation] flips into edges[]
    manifest rows — for the sanctioned direct-emit callers (the idle decay
    prune, the Healer's dangling sweep) that never cross the dispatch
    chokepoint. One query enriches the rows with each claim's own ends so
    the edge is reconstructable from the trace alone; the archived 0→1
    delta is the observed flip the primitive guarantees."""
    if not flipped:
        return []
    pairs = sorted({(f[0], f[1]) for f in flipped})
    endpoints = {}
    # Only the flipped claims, looked up by (edge_id, relation). Chunked like
    # delete_node_edges — SQLite caps host parameters (32766 on the bundled
    # build) and the bulk historic-leak sweep this serves is exactly the case
    # that could exceed it; 500 pairs is 1,000 parameters.
    for i in range(0, len(pairs), 500):
        chunk = pairs[i:i + 500]
        endpoints.update({(r[0], r[1]): (r[2], r[3]) for r in conn.execute(
            'SELECT edge_id, relation, claim_source_id, claim_target_id '
            'FROM edge_relations WHERE (edge_id, relation) IN (VALUES %s)'
            % ','.join('(?, ?)' for _ in chunk),
            [v for pair in chunk for v in pair]).fetchall()})
    rows = []
    for eid, rel in flipped:
        src, tgt = endpoints.get((eid, rel), ('', ''))
        rows.append({
            'edge_id': eid, 'relation': rel, 'reason': reason,
            'encoding_source': encoding_source,
            'source_id': src, 'target_id': tgt,
            'deltas': [{'field': 'archived', 'old': 0, 'new': 1}],
            'warnings': [],
        })
    return rows


# EMITTER_REF_TYPES is deliberately NOT re-exported: trace_contract is its single
# source and consumers (the S2 idle gate, the dashboard mirror) import it there. A
# second import path for one constant only invites "which one do I import?".
__all__ = ['emit_mutation_traces', 'build_events', 'MANIFEST_TRACE_MAP',
           'edge_flip_rows']
