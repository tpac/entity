"""The one-time install collapse: reclassify seeded interaction rows as
"no override deployed" so every name starts following its code default.

Create-only seeding gave each install a DB row for every interaction name, and
under the override model a row that merely *restates* the shipped default reads
as a deployment decision — freezing that name against every future code change.
This module drops those pointers once, per install, so the accessors resolve to
the code default the way the override model always claimed they did.

**Pointer-only. Zero row deletes.** `interactions` rows are the override history
AND the target of thousands of resolvable historical trace pointers, on a column
with no `REFERENCES` clause — deleting rows raises nothing and silently orphans
display data. Only `interaction_active` pointers are touched, so every version
stays re-activatable.

**Daemon-only.** `ensure_logs_schema` runs
inside `Brain.__init__`, so shipping this as a `LOGS_MIGRATIONS` step would fire
it in frozen eval corpora, `IsolatedBrain` copies and test brains. Effective
values are unchanged at that instant — that IS the predicate — but a collapsed
corpus then floats with future code edits, which is exactly what a frozen corpus
must never do. The daemon is a launchd singleton that runs this before the port
opens, so there is one writer and eval brains never reach it.

**The classifier is fingerprint equality, never `source`.** A row byte-identical
to the code default still stamps `source='override'`, because `source` reports
whether a row contributed, not whether it deviated.

**Why the audit record is the rollback path, not a transaction.** Pointer drops
go through the interactions DAL (its own write connection, committing per call
so the write lock is released) while the version stamp goes through
`brain.logs_conn` — two connections on one file. So there is no single envelope
to roll back. Instead the audit is committed BEFORE the first drop and never
overwritten, which is strictly stronger: it survives a hard crash mid-collapse,
where a rollback would leave no record of what was attempted.

**Rolling back by hand** is a pure replay of the audit —
`set_interaction_active(name, version, set_by)` per entry — AND deleting the
`interaction_collapse_version` row from `logs_meta`. Skipping that second half
puts the pointers back on an install that will never re-collapse.
"""
import json

from .interaction_defaults import INTERACTION_DEFAULTS, interaction_fingerprint

COLLAPSE_VERSION = 1
COLLAPSE_VERSION_KEY = 'interaction_collapse_version'
# Both keyed by version: a future pass must take its own backup and write its
# own audit, or it runs against a stale snapshot and leaves a record of the
# WRONG pass — and this record is the whole rollback story.
AUDIT_KEY = 'interaction_collapse_audit_v%d' % COLLAPSE_VERSION
BACKUP_TAG = 'pre-override-collapse-v%d' % COLLAPSE_VERSION

# Verdicts. COMPARE is the only conditional one; the rest are decisions.
COMPARE = 'compare'
ADOPT = 'adopt'
PIN = 'pin'
SKIP = 'skip'
RETIRE = 'retire'

# Per-name policy — the deployment decision, in a reviewable diff.
#
# COMPARE  run the predicate: a row whose effective value fingerprints equal to
#          the code default is not an override, so drop the pointer; a row that
#          differs is a real local decision and is kept.
# ADOPT    drop unconditionally. These have a code default but their row can
#          never converge with it, so COMPARE would guarantee permanent override
#          status for content nothing wants: `s2_community`'s row holds 8 keys,
#          only one of them (min_community_size) in the code dict, and nothing
#          reads the other seven.
# PIN      never touched. `trace_recording` is the one name where activating the
#          wrong version turns on full payload capture for every LLM round, and
#          it is the only name whose active version is deliberately not its
#          highest. The conservative verdict costs one inert pointer.
# SKIP     never touched. `recall_laf`'s row carries measured gain tuning the
#          code default does not, so COMPARE would keep it anyway — SKIP says
#          not to look, so a future predicate change cannot reclassify a real
#          measurement into a drop.
# RETIRE   drop unconditionally. These names have NO code default — they are
#          the `INTERACTION_DEFAULTS` names as "deliberately absent": retired
#          names whose every config key grepped reader-less, plus dead legacy
#          names. Their rows are inert history; the pointer is the only part
#          that still asserts anything.
#
# The RETIRE bucket is exactly the set of names without a code default, and
# every other bucket requires one. `test_interaction_collapse.py` holds that
# correspondence, so the table cannot drift from the registry.
COLLAPSE_POLICY = {
    's1e':                          COMPARE,
    's1e_gist':                     COMPARE,
    'surface':                      COMPARE,
    's2_aspects':                   COMPARE,
    's2_healer':                    COMPARE,
    's2_community_enrichment':      COMPARE,
    's2_consolidation_enrichment':  COMPARE,
    'scopes':                       COMPARE,
    'edge_context':                 COMPARE,

    's2_community':                 ADOPT,

    'trace_recording':              PIN,

    'recall_laf':                   SKIP,

    'boot':                         RETIRE,
    'pre_edit':                     RETIRE,
    'voice_surface':                RETIRE,
    'signal_assembler':             RETIRE,
    'encoding_agent':               RETIRE,
    's2_edge_families':             RETIRE,
    's2_node_families':             RETIRE,
    'recall_query_expansion':       RETIRE,
}

# Verdicts allowed to change a name's effective value. Only ADOPT: its whole
# definition is "the row and the default disagree and the default wins".
_MAY_CHANGE_EFFECTIVE = (ADOPT,)


def _effective_fingerprints(brain):
    """{name: fingerprint} of the resolved value, through the accessors.

    Registry names only — a RETIRE name has no code default, so the accessors
    raise for it by design and there is no effective value to preserve.
    """
    return {name: brain.get_interaction_stamp(name)['fingerprint']
            for name in INTERACTION_DEFAULTS}


def _audit_entries(brain, live, effective):
    """Replay record for every live pointer: enough to put it back verbatim,
    plus what it was RESOLVING TO before anything was dropped.

    `set_at` is not needed to replay (re-activating stamps a fresh one) but it
    is what tells an operator whether a pointer was a recent deliberate deploy
    or years-old seed residue — the judgment a rollback actually turns on.

    `effective_fingerprint` is the load-bearing one. It is the drift check's
    own currency, so a retry after a crash can rebuild the true pre-collapse
    baseline instead of measuring the half-collapsed state it woke up in. The
    raw row is recoverable from `version` + `parameters`; the value the row
    RESOLVED to is not, once the pointer is gone.
    """
    entries = []
    for info in live:
        name = info['name']
        version = info.get('active_version')
        if version is None:
            continue
        row = brain.get_interaction(name) or {}
        entries.append({
            'name': name,
            'version': version,
            'set_by': info.get('active_set_by'),
            'set_at': info.get('active_set_at'),
            'effective_fingerprint': effective.get(name),
            'parameters': row.get('parameters') or '',
            'verdict': COLLAPSE_POLICY.get(name, 'unknown'),
        })
    return entries


def _restore(brain, audit, dropped):
    """Replay the audit for every pointer this run dropped.

    Never raises: a restore is always the recovery path for a failure already
    in flight, and letting one un-replayable name abort the loop would both
    strand the rest and replace the original error with this one. Returns the
    names that could not be put back, for the caller to name in that error.
    """
    by_name = {e['name']: e for e in audit}
    failed = []
    for name in dropped:
        entry = by_name.get(name)
        if not entry:
            failed.append(name)
            continue
        try:
            brain.set_interaction_active(name, entry['version'],
                                         set_by=entry['set_by'])
        except Exception:
            failed.append(name)
    return failed


def _collapse_overrides(brain):
    """Drop the pointers the policy table calls non-overrides, then prove no
    effective value moved except where the table says it may."""
    from .db_backup import backup_before_destructive
    from .schema import read_meta_value, write_meta_value

    live = brain.list_interactions()
    pointered = [i for i in live if i.get('active_version') is not None]
    if not pointered:
        print('[collapse] no override pointers — nothing to collapse',
              flush=True)
        return

    # Refuse rather than proceed unbacked: this is the one code path that
    # rewrites a production DB, and an unstamped run simply retries next boot.
    # compress=False for the same reason the migration runner uses it — this
    # runs before the port answers pings and the health monitor force-restarts
    # an unresponsive daemon.
    if not backup_before_destructive(brain.logs_db_path, BACKUP_TAG,
                                     compress=False):
        raise RuntimeError(
            'pre-collapse backup of %s failed — refusing to drop pointers'
            % brain.logs_db_path)

    before = _effective_fingerprints(brain)
    entries = _audit_entries(brain, pointered, before)
    # First write wins: a retry must not overwrite the pre-first-attempt
    # record, or the original pointers become unrecoverable.
    stored = read_meta_value(brain.logs_conn, 'logs_meta', AUDIT_KEY)
    if stored is None:
        write_meta_value(brain.logs_conn, 'logs_meta', AUDIT_KEY,
                         json.dumps(entries))
        brain.logs_conn.commit()
        audit = entries
    else:
        audit = json.loads(stored)

    # A previous attempt may have dropped pointers and died before stamping.
    # `before` measured now would be that half-collapsed state, and comparing
    # it to itself makes the drift check pass vacuously for exactly the names
    # a crash left unverified. The audit holds what they truly resolved to.
    for entry in audit:
        recorded = entry.get('effective_fingerprint')
        if recorded and entry['name'] in before:
            before[entry['name']] = recorded

    dropped, kept, unknown = [], [], []
    try:
        for entry in entries:
            name, verdict = entry['name'], entry['verdict']
            if verdict in (PIN, SKIP):
                kept.append('%s(%s)' % (name, verdict))
                continue
            if verdict == 'unknown':
                # A name this build has never heard of. Leaving it is the
                # conservative read: it may be a deliberate local deployment.
                unknown.append(name)
                continue
            if verdict == COMPARE:
                default_template, default_config = INTERACTION_DEFAULTS[name]
                if before[name] != interaction_fingerprint(
                        name, default_template, default_config):
                    kept.append('%s(differs)' % name)
                    continue
            # Recorded BEFORE the call: the delete commits first and cache
            # invalidation follows it, so an invalidator that raises would
            # otherwise leave a dropped pointer outside the restore's reach.
            dropped.append(name)
            brain.clear_interaction_override(name)
    except Exception:
        _restore(brain, audit, dropped)
        raise

    after = _effective_fingerprints(brain)
    drifted = sorted(n for n in before
                     if before[n] != after[n]
                     and COLLAPSE_POLICY.get(n) not in _MAY_CHANGE_EFFECTIVE)
    if drifted:
        failed = _restore(brain, audit, dropped)
        detail = ', '.join('%s %s->%s' % (n, before[n], after[n])
                           for n in drifted)
        if failed:
            detail += ' | COULD NOT RESTORE: %s' % ', '.join(sorted(failed))
        error = RuntimeError(
            'collapse changed the effective value of %s — pointers restored '
            'from the audit record; nothing stamped, retrying next boot'
            % detail)
        brain._log_error('interaction_collapse_drift', error, detail)
        raise error

    by_verdict = {}
    for name in dropped:
        by_verdict.setdefault(COLLAPSE_POLICY.get(name, 'unknown'),
                              []).append(name)
    print('[collapse] dropped %d pointer(s): %s'
          % (len(dropped),
             '; '.join('%s=%s' % (v, ','.join(sorted(names)))
                       for v, names in sorted(by_verdict.items())) or '-'),
          flush=True)
    if kept:
        print('[collapse] kept as real overrides: %s'
              % ', '.join(sorted(kept)), flush=True)
    if unknown:
        print('[collapse] left alone (no policy entry): %s'
              % ', '.join(sorted(unknown)), flush=True)


def collapse_seeded_overrides(brain):
    """Version-gated entry point — runs once per install.

    Daemon-only by design (see the module docstring). The version stamp buys
    once-only semantics and nothing else: this is an install event, not a
    schema shape change, which is why it is not a `LOGS_MIGRATIONS` step.
    """
    from .schema import run_versioned_migrations
    try:
        run_versioned_migrations(
            brain.logs_conn, 'logs_meta', COLLAPSE_VERSION_KEY,
            COLLAPSE_VERSION,
            [(COLLAPSE_VERSION, lambda _conn: _collapse_overrides(brain))],
            label='override collapse')
        brain.logs_conn.commit()
    except Exception as e:
        # Never block boot: the install keeps running its current pointers and
        # the unstamped version retries on the next daemon start. Silence here
        # would rebuild the freeze this exists to remove, so both channels fire.
        try:
            brain.logs_conn.rollback()
        except Exception:
            pass
        print('[collapse] WARNING: skipped (%s)' % e, flush=True)
        try:
            brain._log_error('interaction_collapse_failed', e,
                             'collapse_version=%d' % COLLAPSE_VERSION)
        except Exception:
            pass
