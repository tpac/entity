"""The one door an eval or A/B arm uses to put a prompt or config in front of
a run — and to take it back.

Seven eval sites used to hand-roll register+activate, each having
independently rediscovered merge-vs-replace and read-back verification, and
none of them could revert: a forgotten override silently opts that name out
of code defaults for the life of that brain.

Two verbs:

    version = override_interaction(brain, 's1e', template=candidate)

    with interaction_override(brain, 'recall_laf', parameters=gains):
        ...                     # cleared on exit, including on exception

Both route through `brain.register_interaction` / `brain.set_interaction_active`
— never `brain._interaction_dal` — so the `scopes` validator runs at the
register door and every cache keyed on the name is invalidated on the flip.
Reaching past those doors is how an eval override used to read stale for a
whole TTL window, and it was latent only by engine-attach ordering luck.

Reads use the RESOLVER (`get_interaction_prompt` / `get_interaction_config`),
not the row. On a brain with no pointer the row is absent and the code default
is the truth — a `_interaction_dal.get_active` read there returns None, and
`template=None` "preserve what's active" silently becomes "override with an
empty template".

WHY IT CAN VERIFY WITHOUT COMPARING TEMPLATES
`get_interaction_stamp(name)['fingerprint']` is 12 hex over the resolved
(overlaid) template+config, so one comparison covers both halves. An arm
compares arms on that; it never string-compares a multi-KB prompt to decide
whether its own override took. A fingerprint that does not MOVE across the
override is the silent-arm-collapse signal: both arms would
measure the same K and the A/B would report a difference of zero as a result.

Comparing arms is a CROSS-arm assertion, and no single run can make it: each
arm only ever sees its own stamp, which is precisely the gap the old
version-int corpus address hid in. Read each arm's stamp from that arm's own
brain and assert the two differ. ⚠ Reading one means opening a per-item corpus
copy — never the live `brain.db` while the daemon is up: a second writer
connection is how an index gets corrupted (CLAUDE.md). Route a live read
through the daemon (`daemon_client.send_command`) instead.

Leak check: `./dev check-overrides`.
"""
import json
import sys
from contextlib import contextmanager


def _as_dict(parameters):
    """Accept a config as a dict or as the JSON string the DAL stores."""
    if parameters is None or isinstance(parameters, dict):
        return parameters
    if isinstance(parameters, str):
        if not parameters.strip():
            return {}
        parsed = json.loads(parameters)
        # The resolver refuses a non-object override config at READ time by
        # logging and falling back to the default — silently, from the caller's
        # side. Refusing it here instead means the eval fails where the mistake
        # is, rather than running an arm on the code default it thought it had
        # overridden.
        if not isinstance(parsed, dict):
            raise TypeError(
                'parameters JSON must be an object, got %s'
                % type(parsed).__name__)
        return parsed
    raise TypeError('parameters must be a dict, a JSON string, or None — got %r'
                    % type(parameters).__name__)


def override_interaction(brain, name, *, template=None, parameters=None,
                         merge=False, set_by=None):
    """Register + activate an override for `name` in THIS brain. Returns the version.

    `template=None` preserves the effective template, `parameters=None` the
    effective config, so either half can be overridden alone (a config-only
    interaction like `recall_laf` passes `template=''`).

    `merge=True` overlays the given keys onto the effective config instead of
    storing them alone — needed whenever the stored config also carries state
    you must not wipe (recall_laf's config is the carrier of fitted gains, so
    a wholesale replace would reset a corpus brain's gains to module defaults
    and let a base-parity check pass against a config that brain never ran).
    Side effect to know: the stored row then carries the FULL merged dict, so
    outside the self-reverting CM a merge=True override pins every config key
    against future code-default changes, not just the ones you set.

    Raises RuntimeError if the override did not reach the resolver. Prints a
    loud warning — not an error — when the fingerprint does not move: that is
    a legitimate no-op (an override byte-identical to the default) but it
    means an A/B built on it is comparing a K against itself.
    """
    set_by = set_by or 'eval-override-%s' % name
    before = brain.get_interaction_stamp(name)['fingerprint']

    if template is None:
        template = brain.get_interaction_prompt(name) or ''
    config = _as_dict(parameters)
    if config is None:
        config = dict(brain.get_interaction_config(name) or {})
    elif merge:
        merged = dict(brain.get_interaction_config(name) or {})
        merged.update(config)
        config = merged

    result = brain.register_interaction(
        name, template=template, parameters=json.dumps(config),
        created_by=set_by)
    version = result['version'] if isinstance(result, dict) else int(result)
    # register() never activates — the pointer flip is always explicit since
    # Step 6 dropped AUTO_V1. An eval that gates activation on version > 1
    # (the old auto-activate assumption) never activates its v1 override on a
    # freshly wiped corpus brain, and both arms then run the same prompt.
    brain.set_interaction_active(name, version, set_by=set_by)

    stamp = brain.get_interaction_stamp(name)
    if stamp['source'] != 'override' or stamp['version'] != version:
        raise RuntimeError(
            'override of %r did not reach the resolver: registered v%s, '
            'stamp says %s v%s' % (name, version, stamp['source'],
                                   stamp['version']))
    # An EMPTY template is the config-only idiom, not a template override: the
    # resolver takes the row's template only when non-empty, so it keeps
    # serving the code default. Asserting equality here would fail every
    # config-only override of a name that HAS a default template
    # (prompt templates run to ~1k chars), which is why this checks the
    # template only when one was actually set.
    if template:
        effective_template = brain.get_interaction_prompt(name) or ''
        if effective_template != template:
            raise RuntimeError(
                'override of %r took the pointer but not the template: '
                'resolver returns %d chars, %d were set'
                % (name, len(effective_template), len(template)))
    effective_config = brain.get_interaction_config(name) or {}
    drifted = {k: (v, effective_config.get(k)) for k, v in config.items()
               if effective_config.get(k) != v}
    if drifted:
        raise RuntimeError(
            'override of %r took the pointer but not the config: %s '
            '(set → effective)' % (name, drifted))
    if stamp['fingerprint'] == before:
        print('[override] WARN %s v%s is byte-identical to what was already '
              'effective (fingerprint %s unchanged) — an A/B across this '
              'override compares a K against itself'
              % (name, version, before), file=sys.stderr, flush=True)
    return version


@contextmanager
def interaction_override(brain, name, *, template=None, parameters=None,
                         merge=False, set_by=None):
    """`override_interaction` that reverts itself, exception or not.

    Yields the version. On exit the pointer is DELETED — the name lands on its
    code default, NOT on whatever override was active before. That is the
    behaviour an eval arm wants (a baseline arm must measure the default, and
    an `IsolatedBrain` inherits every production override it snapshot-copied,
    so "restore what was there" would restore the very thing being controlled
    for). It is also the one surprise here: if you need the prior override
    back, capture it and re-apply it yourself. The two-way door Step 6 shipped
    is what makes any of this writable.

    The exit clear is deliberately UNGUARDED, and that is safe for exactly one
    reason: `clear_interaction_override` raises KeyError only for a name that
    deleted nothing AND has no code default (the typo guard — a clear of
    'trace_recoding' must not report "already on the default" while the real
    override keeps running), and no such name can reach this line. Entry reads
    the resolver, which refuses an unregistered name before anything is
    written. So every name that gets here has a code default, and its clear
    returns quietly whether or not a pointer survived the body. A guard here
    would be unreachable, and a `try/except` around a clear in __exit__ is
    worse than unreachable — it is where a leak-proof block turns into an
    error-masking one, swallowing the failure being debugged.
    `test_interaction_override.py` pins the entry gate that holds this up.
    """
    version = override_interaction(brain, name, template=template,
                                   parameters=parameters, merge=merge,
                                   set_by=set_by)
    try:
        yield version
    finally:
        brain.clear_interaction_override(name)


# ─── Leak check ────────────────────────────────────────────────────────────

def stray_overrides(brain=None):
    """Every live override pointer — on the running daemon, or on `brain`.

    After the override collapse a production install carries only its PIN and
    SKIP baseline (`trace_recording`, `recall_laf` — see
    `servers/interaction_collapse.COLLAPSE_POLICY`), so any other pointer is
    either a deliberate deployment or a forgotten revert.

    Pass `brain` to inspect a CORPUS or IsolatedBrain copy instead of the
    live install — an eval baseline arm inherits every pointer the
    production copy carried, and a stray third pointer there silently turns
    "treatment vs default" into "treatment vs somebody's override". Checking
    the copy before the run is the only way to see that.

    With no `brain`, routed through the daemon rather than a second Brain:
    two writer connections against a live WAL database is how an index gets
    corrupted.
    """
    if brain is not None:
        return [e for e in brain.list_interactions()
                if e.get('active_version') is not None]
    from servers.daemon_client import send_command
    r = send_command('list_interactions', {})
    if not r.get('ok'):
        raise RuntimeError('list_interactions failed: %s' % (r,))
    return [e for e in (r.get('result') or [])
            if e.get('active_version') is not None]


def main():
    # Exit codes are a contract for scripted gates: 0 = clean, 1 = eval
    # leak found, 2 = COULD NOT CHECK (daemon unreachable). Letting the
    # transport error escape as a traceback would exit 1 — the same code
    # as a real leak — and a CI gate could never tell them apart.
    try:
        rows = stray_overrides()
    except RuntimeError as e:
        print('[check-overrides] could not check: %s' % e, file=sys.stderr)
        return 2
    if not rows:
        print('[check-overrides] no override pointers — every interaction is '
              'on its code default.')
        return 0
    def is_eval(entry):
        return str(entry.get('active_set_by') or '').startswith('eval')

    eval_tagged = [e for e in rows if is_eval(e)]
    print('[check-overrides] %d override pointer(s) live:' % len(rows))
    for e in rows:
        print('  %-28s v%-4s set_by=%s%s'
              % (e['name'], e['active_version'], e.get('active_set_by'),
                 ' ← EVAL LEAK' if is_eval(e) else ''))
    if eval_tagged:
        print('\n%d pointer(s) were set by an eval and never reverted. Each one '
              'opts that name out of code defaults until cleared:'
              % len(eval_tagged))
        for e in eval_tagged:
            print("  clear_interaction_override('%s')" % e['name'])
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
