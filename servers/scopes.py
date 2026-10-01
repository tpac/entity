"""Scope policy + veil — the operator's separation contract per dimension.

The user-facing control surface for the scope lane (scope shape = Stamp /
Session anchor / Lane / Differential exposure / Frame anchor). Config lives
in the interactions table (name: 'scopes', config-only — versioned, edited
via register_interaction + set_interaction_active), shaped as:

    {
      "project":     {"mode": "scoped", "overrides": {"client-x": "isolated"}},
      "counterpart": {"mode": "scoped", "overrides": {}}
    }

Three discrete modes per dimension (policies, not scalars):

    open     — no recall pressure from this dimension. Mismatch marks still
               render (information is never hidden), the Frame anchor stays.
    scoped   — DEFAULT. Soft separation: the LAF lane applies its fitted
               gain (package E; gain 0 until fitted, so scoped behaves like
               open until pressure is earned from real data).
    isolated — hard wall, both directions, enforced by the VEIL below (v1's
               only teeth). NO exemptions (operator ruling 2026-08-05).

THE VEIL — one precomputed hidden-set, not a filter at N call sites
-------------------------------------------------------------------
The enforcement object is "the set of node ids this session must not see",
materialized by two indexed KV queries per isolated dimension:

    outward — nodes stamped with an isolated value that is not the
              session's own: hidden EVERYWHERE else, including sessionless
              reads (default-deny).
    inward  — when the SESSION's own value is isolated: every node stamped
              with a foreign value on that dimension. Nodes with NO
              provenance stay visible — unknown is neutral even inside a
              wall (hiding the legacy unscoped corpus would empty the brain
              there).

Consumers (recall assembly pre-limit, edge attachments, spread
expansion, boot lanes, filter_nodes, fetch tools) do ONE set-membership
check. There is no per-candidate policy evaluation, no per-call KV fetch,
and therefore no fail-open I/O path at check time. Cache: keyed on
(active config version, MetadataDAL.change_key(), session scope signature)
— a new walled node or a config flip invalidates on the next read; a build
failure keeps the last good veil and logs CRITICAL (never silently
un-walls); a first-build failure raises — a loud dead recall beats a
silent leak.

Isolation governs what rises UNBIDDEN. An explicit get_node(id) /
recall_node(id) is a deliberate reach for a known id and stays open.

Dimensions whose session-side resolver is still an install constant
(counterpart, until the speaker arc's F4 makes it per-session) REFUSE
`isolated` — with a constant session value, isolating a foreign value
would hide those nodes from every session including their own
(un-exitable), and isolating the constant would black out the stamped
corpus everywhere. The refusal logs loud and degrades to `scoped`;
flipping SESSION_RESOLVABLE when F4 lands lifts it.
"""

from .contract import SCOPE_PROVENANCE_FIELDS

MODES = ('open', 'scoped', 'isolated')
DEFAULT_MODE = 'scoped'
_INTERACTION_NAME = 'scopes'

# Dimensions whose session-side value is genuinely per-session today.
# counterpart flips in when brain.counterpart_for stops returning the
# install constant (speaker arc F4).
SESSION_RESOLVABLE = ('project',)

# Code default (indexed by interaction_defaults) — explicit defaults
# for every dimension so the operator edits a real shape, not an empty dict.
SCOPES_CONFIG_V1 = {
    dim: {'mode': DEFAULT_MODE, 'overrides': {}}
    for dim in SCOPE_PROVENANCE_FIELDS
}


def _safe_log(log, kind, exc, ctx=''):
    if log:
        try:
            log(kind, exc, ctx)
        except Exception:
            pass


def validate_scopes_config(config):
    """Violation strings for a scopes config — the one definition of valid,
    callable at the write door (register path), by ScopePolicy, and by
    tests (the aspect_store.validate_taxonomy shape). Empty list = valid.
    Keys and modes are judged after normalization (strip/lower), so a
    violation here is a REAL shape problem, not a case slip."""
    violations = []
    if config is None:
        return violations
    if not isinstance(config, dict):
        return ['config must be a dict, got %s' % type(config).__name__]
    known = {d.lower() for d in SCOPE_PROVENANCE_FIELDS}
    for dim, raw in config.items():
        dkey = str(dim).strip().lower()
        if dkey not in known:
            violations.append("unknown dimension %r" % dim)
            continue
        if not isinstance(raw, dict):
            violations.append("%s: entry must be a dict" % dkey)
            continue
        mode = raw.get('mode')
        if mode is not None and str(mode).strip().lower() not in MODES:
            violations.append("%s.mode: %r not in %s" % (dkey, mode, MODES))
        for value, vmode in (raw.get('overrides') or {}).items():
            if not isinstance(value, str) or not value.strip():
                violations.append("%s.overrides: empty/non-string key" % dkey)
            if str(vmode).strip().lower() not in MODES:
                violations.append("%s.overrides[%s]: %r not in %s"
                                  % (dkey, value, vmode, MODES))
        if dkey not in SESSION_RESOLVABLE:
            wants_iso = (str(mode).strip().lower() == 'isolated' or any(
                str(m).strip().lower() == 'isolated'
                for m in (raw.get('overrides') or {}).values()))
            if wants_iso:
                violations.append(
                    "%s: 'isolated' refused — this dimension's session "
                    "value is still the install constant, so isolation "
                    "would be un-exitable (see module docstring); lifts "
                    "when the dimension becomes session-resolvable" % dkey)
    return violations


class ScopePolicy:
    """Parsed, normalized scope config. Violations degrade LOUDLY to the
    default — a typo must never crash recall, and must never silently mean
    'isolated'; refused-isolation (non-session-resolvable dims) degrades to
    'scoped' the same way."""

    def __init__(self, config, log=None):
        for v in validate_scopes_config(config):
            _safe_log(log, 'scopes_config_invalid', ValueError(v),
                      'degrading to %r' % DEFAULT_MODE)
        self._dims = {}
        norm = {str(k).strip().lower(): v for k, v in (config or {}).items()
                if isinstance(v, dict)}
        for dim in SCOPE_PROVENANCE_FIELDS:
            raw = norm.get(dim) or {}
            iso_ok = dim in SESSION_RESOLVABLE
            self._dims[dim] = {
                'mode': self._norm_mode(raw.get('mode'), iso_ok),
                'overrides': {
                    str(k).strip().lower(): self._norm_mode(v, iso_ok)
                    for k, v in (raw.get('overrides') or {}).items()
                    if isinstance(k, str) and k.strip()},
            }

    @staticmethod
    def _norm_mode(mode, isolation_allowed):
        m = str(mode).strip().lower() if mode is not None else DEFAULT_MODE
        if m not in MODES:
            return DEFAULT_MODE
        if m == 'isolated' and not isolation_allowed:
            return DEFAULT_MODE
        return m

    def mode(self, dim: str, value: str = '') -> str:
        """Effective mode for a dimension value: per-value override →
        dimension mode → default."""
        d = self._dims.get(dim)
        if d is None:
            return DEFAULT_MODE
        if value:
            override = d['overrides'].get(value.strip().lower())
            if override:
                return override
        return d['mode']

    def isolated_values(self, dim: str):
        """Lowercased values explicitly isolated on this dimension, plus
        whether the DIMENSION default is isolated (every value walled)."""
        d = self._dims.get(dim) or {'mode': DEFAULT_MODE, 'overrides': {}}
        values = {v for v, m in d['overrides'].items() if m == 'isolated'}
        return values, d['mode'] == 'isolated'

    @property
    def has_isolation(self) -> bool:
        return any(self.isolated_values(dim)[0] or self.isolated_values(dim)[1]
                   for dim in self._dims)


def build_veil(brain, policy: ScopePolicy, session_scope) -> frozenset:
    """Materialize the hidden-set for one session scope.

    Per-VALUE policy is resolved in Python over the dimension's DISTINCT
    stamped values (a handful per provenance key), then one indexed id
    lookup fetches the walled set — so per-value overrides are honored
    everywhere (an `open` override under an isolated dimension default IS
    the shared lane it declares, in both directions). Raises on DB failure
    (the caller owns keep-last-good semantics).

    Walled(value) for a session with own value S on dimension d:
      - value != S and mode(d, value) == 'isolated'   (outward wall)
      - value != S and mode(d, S) == 'isolated'
        and mode(d, value) != 'open'                  (inward wall — an
        explicit per-value `open` stays visible even inside a wall; that
        is what the override means)
    Unknown stays neutral: nodes without the key never match.
    """
    session_scope = session_scope or {}
    hidden = set()
    for dim in SCOPE_PROVENANCE_FIELDS:
        sess_val = (session_scope.get(dim) or '').strip().lower()
        values = {v.strip().lower()
                  for v in brain._meta_kv.distinct_values_for_key(dim)}
        values.discard(sess_val)
        values.discard('')
        inside_wall = bool(sess_val) and policy.mode(dim, sess_val) == 'isolated'
        walled = {
            v for v in values
            if policy.mode(dim, v) == 'isolated'
            or (inside_wall and policy.mode(dim, v) != 'open')
        }
        if walled:
            hidden.update(
                brain._meta_kv.node_ids_with_value_in(dim, sorted(walled)))
    return frozenset(hidden)


def scrub_node(node, veil) -> None:
    """Remove walled entries from a node dict's cross-node attachments, in
    place — connections and communities (id+title lines) and _corrections
    (which carry the corrector's FULL content, reasoning and raw quotes;
    matched on the full `node_id` field — the display `id` is an 8-char
    short form the veil can't match). The node itself was already admitted;
    this stops its EDGES from carrying walled payload across the wall."""
    if not veil or not isinstance(node, dict):
        return
    for key in ('connections', 'communities'):
        entries = node.get(key)
        if isinstance(entries, list):
            node[key] = [e for e in entries
                         if not isinstance(e, dict)
                         or e.get('id') not in veil]
    corrections = node.get('_corrections')
    if isinstance(corrections, list):
        node['_corrections'] = [
            c for c in corrections
            if not isinstance(c, dict) or c.get('node_id') not in veil]


def load_scope_policy(brain) -> ScopePolicy:
    """Construct the policy from the 'scopes' interaction config (empty /
    missing → all defaults, behavior-neutral). Callers cache — see
    Brain.scope_veil."""
    try:
        config = brain.get_interaction_config(_INTERACTION_NAME) or {}
    except Exception as e:
        _safe_log(brain._log_error, 'scopes_config_load', e,
                  'scopes config unreadable — using defaults')
        config = {}
    return ScopePolicy(config, log=brain._log_error)
