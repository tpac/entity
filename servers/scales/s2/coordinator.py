"""S2 Coordinator — runs all S2 integration units in order.

**Do not call this function directly — go through `Brain.run_s2()`.**

That method is the one door to S2 activation and it owns the single-flight
guard, so every caller in the process is serialized: the daemon's maintenance
poll (via `Brain.run_maintenance_if_due`, which owns the "is it time?" policy),
plus evals, benchmarks and IsolatedBrain. A guard held by ONE caller cannot
protect the others — a second caller could overlap it and run consolidation
twice in parallel (node:daaf63a9). The daemon's `_maintenance_poll_running`
flag is a submission throttle, NOT that guard. Calling this module function directly
bypasses the lock and re-opens the bug.

Each unit checks its own traces to decide whether it should fire.

Ordering matters:
1. AspectIntegration — classify new node types / edge relations into the required aspects
2. Community detection — detect structure
3. Healer — fill missing findability fields on nodes that lack them (uses community context)
4. Consolidation — merge convergent nodes; runs last so its cluster fingerprints
   are stamped after every other unit's writes (see the note in select_units)

Note: temporal interval extraction is NOT an S2 unit. It runs in
embed_queue alongside vector backfill — every remember/revise/add_relation
enqueues, the worker drains in batches. Conceptually the same kind of
enrichment as embeddings, in the same pipeline.
"""


def select_units(units=None):
    """The unit classes a cycle runs, in the fixed order.

    `units` is an optional list of unit NAMEs (e.g. ['aspect_integration']);
    the selection keeps the fixed order whatever order it is given in. None
    selects every unit. Anything else — an unknown name, an empty list, a bare
    string, a dict — raises ValueError: a filter that silently ran nothing, or
    ran something unintended, would read as a clean cycle.
    """
    from .aspect_integration import AspectIntegration
    from .consolidation import Consolidation
    from .community import CommunityDetection
    from .healer import Healer

    # AspectIntegration runs first and cannot re-fire the units after it: with
    # nothing to classify it is a true no-op (no proposals, no trace — the
    # decoder's empty-batch early-out, pinned by tests/test_aspect_decoder.py),
    # and no other unit reads aspect_scan traces; each gates on its own state.
    # Consolidation runs LAST: every earlier unit's writes
    # (healer field-fills, community placements, aspect classifications) land
    # BEFORE consolidation reads the graph and stamps its cluster
    # fingerprints — so nothing later in the same cycle bumps a member's
    # updated_at and re-arms a just-recorded fingerprint (a healer
    # question-fill after consolidation would invalidate every new-node
    # cluster's fresh fingerprint each cycle). Consolidation also benefits from healed
    # fields and current community membership in its cluster payloads.
    unit_classes = [
        AspectIntegration,
        CommunityDetection,
        Healer,
        Consolidation,
    ]
    if units is None:
        return unit_classes
    known = [cls.NAME for cls in unit_classes]
    if (not isinstance(units, (list, tuple)) or not units
            or not all(isinstance(u, str) and u in known for u in units)):
        raise ValueError('units must be a non-empty list of S2 unit names '
                         'from %s, got %r' % (known, units))
    return [cls for cls in unit_classes if cls.NAME in units]


def run_s2(brain, units=None, force=False):
    """Run all S2 integration units in order.

    Args:
        brain: Brain instance (direct DB access, inline execution).
        units: optional list of unit NAMEs to run instead of every unit —
            see select_units, which validates it.
        force: the cycle was requested, not scheduled — set on each unit as
            `forced` (IntegrationUnit), so a unit that batches runs what waits.

    Returns:
        dict of {unit_name: result_dict}
    """
    results = {}
    for unit in [cls(brain) for cls in select_units(units)]:
        unit.forced = force
        failure_key = 's2_%s_consecutive_failures' % unit.NAME
        failure = None
        try:
            result = unit.run()
            # Units may return partial progress with an error. Keep that
            # progress and use the same failure policy as an exception.
            if isinstance(result, dict) and result.get('error'):
                failure = RuntimeError(result['error'])
            # Reset failure counter on success. 'skipped' counts as neutral
            # (unit declined to run), don't reset on that.
            elif not (isinstance(result, dict) and result.get('skipped')):
                prior = int(brain.get_config(failure_key) or 0)
                if prior > 0:
                    brain.set_config(failure_key, '0')
                    print('[s2:%s] Recovered after %d consecutive failures'
                          % (unit.NAME, prior), flush=True)
        except Exception as e:
            failure = e
            result = {'error': str(e) or type(e).__name__}
        results[unit.NAME] = result
        if failure is not None:
            print('[s2:%s] ERROR: %s' % (unit.NAME, failure), flush=True)
            brain._log_error('s2_%s' % unit.NAME, failure, 'coordinator run')

            # Persistent-failure escalation — rate-limited brain._log_error
            # suppresses repeats, hiding the fact that a unit is broken every
            # run. Track a running counter in brain_meta; each new failure
            # fires a DIFFERENT error source (encoded with the count) so the
            # rate limiter can't collapse them in the errors view. Resets on
            # next successful run.
            try:
                count = int(brain.get_config(failure_key) or 0) + 1
                brain.set_config(failure_key, str(count))
                if count >= 3:
                    brain._log_error(
                        's2_%s_persistent_failure' % unit.NAME,
                        RuntimeError(
                            '%s has failed %d runs in a row: %s' % (
                                unit.NAME, count, str(failure)[:150])),
                        'unit is consistently broken — investigate immediately')
            except Exception as persist_err:
                # The whole point of this block is to make failures loud.
                # Swallowing the escalation is the very pattern we're fighting.
                brain._log_error(
                    's2_%s_persistent_failure_tracker_crashed' % unit.NAME,
                    persist_err,
                    'could not update failure counter for %s' % unit.NAME)

    return results
