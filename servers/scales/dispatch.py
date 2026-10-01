"""Scale dispatch infrastructure — env loading + the write-command classification.

Shared by all scale agents (S1 Scribe, the S2 units, future scales). Both
run IN-PROCESS on the daemon's brain and write through
`scales/s2/base.py::_make_encoder_dispatch`, which calls COMMAND_TABLE handlers
directly under `brain.write_lock` (the same lock daemon_server.py uses for
client requests, so cross-writer serialization is guaranteed).

What lives here: provider-failure classification, `load_env` /
`resolve_api_key`, the canonical write-command CLASSIFICATION the attribution
chokepoint reads, and the scope-provenance stamp.
"""

import os
import re

from ..brain_constants import user_config_dir


# ── Provider failure classification (the LLM seam) ──
#
# The brain reacts to a REFUSED call — pause encoding on a dead key or an
# exhausted quota — and those reactions must not know WHICH provider
# refused. This maps one provider's exception onto a closed vocabulary; adding a
# second provider means adding a second mapper here, never a branch at the
# reaction site. Key resolution already lives in this module, so interpreting
# the refusal of the key we resolved belongs beside it.
#
# Status code first, exception class second, deliberately: HTTP semantics port
# across providers, `anthropic.AuthenticationError` does not. Nothing here
# imports an SDK — the classifier is provider-neutral by construction, and the
# test suite can exercise every branch without one.

LLM_AUTH_REJECTED = 'auth_rejected'      # the credential was refused (401/403)
LLM_QUOTA_EXHAUSTED = 'quota_exhausted'  # spend cap / credit balance reached
LLM_RATE_LIMITED = 'rate_limited'        # slow down, retry later (429)
LLM_INVALID_REQUEST = 'invalid_request'  # this payload is wrong (400) — e.g. too long
LLM_TRANSIENT = 'transient'              # connectivity or a provider-side 5xx
LLM_UNKNOWN = 'unknown'                  # not an LLM failure at all

# The connectivity family carries no status code, so it is matched by exception
# class NAME — the one place names beat types, since matching types would drag
# an SDK import into a module that deliberately has none.
_TRANSIENT_NAMES = frozenset((
    'APIConnectionError', 'APITimeoutError', 'InternalServerError',
    'ConnectError', 'ConnectTimeout', 'ReadTimeout', 'TimeoutException',
))

# A quota refusal is a 400 or 429 whose BODY says "you have spent enough" —
# indistinguishable from a malformed request or a rate limit by status alone.
_QUOTA_MARKERS = ('usage limit', 'credit balance', 'quota', 'billing')

_STATUS_RE = re.compile(r'error code:\s*(\d{3})', re.I)

_UNKNOWN_OUTCOME = {'kind': LLM_UNKNOWN, 'retry_after': 0, 'detail': ''}


def _status_of(exc) -> int:
    """HTTP status behind an exception, 0 when there isn't one. Checks the
    attribute, then the response object, then the rendered message — SDKs
    differ on which they populate, and a stringified error is sometimes all a
    wrapper preserved."""
    for probe in (getattr(exc, 'status_code', None),
                  getattr(getattr(exc, 'response', None), 'status_code', None)):
        if isinstance(probe, int):
            return probe
    m = _STATUS_RE.search(str(exc))
    return int(m.group(1)) if m else 0


def _retry_after_of(exc) -> int:
    """Seconds the provider asked us to wait, 0 when unstated."""
    headers = getattr(getattr(exc, 'response', None), 'headers', None)
    try:
        return int(float(headers.get('retry-after')))
    except (AttributeError, TypeError, ValueError):
        return 0


def _classify_one(exc) -> dict:
    text = str(exc)
    status = _status_of(exc)
    quota = any(m in text.lower() for m in _QUOTA_MARKERS)

    if status in (401, 403):
        return {'kind': LLM_AUTH_REJECTED, 'retry_after': 0,
                'detail': text[:200]}
    if quota and status in (400, 402, 429):
        return {'kind': LLM_QUOTA_EXHAUSTED,
                'retry_after': _retry_after_of(exc), 'detail': text[:200]}
    if status == 429:
        return {'kind': LLM_RATE_LIMITED,
                'retry_after': _retry_after_of(exc), 'detail': text[:200]}
    if status == 400:
        return {'kind': LLM_INVALID_REQUEST, 'retry_after': 0,
                'detail': text[:200]}
    if status == 408 or 500 <= status < 600 or type(exc).__name__ in _TRANSIENT_NAMES:
        return {'kind': LLM_TRANSIENT, 'retry_after': 0,
                'detail': text[:200]}
    return dict(_UNKNOWN_OUTCOME)


def classify_llm_failure(exc) -> dict:
    """Map a provider exception onto the brain's failure vocabulary.

    Returns {'kind', 'retry_after', 'detail'} — `kind` is one of the LLM_*
    constants above, `retry_after` the seconds the provider asked for (rate
    limits only), `detail` the provider's own words, which carry any reset
    date it named.

    Unwraps `__cause__`: RunLoopError wraps mid-run failures and carries no
    status of its own, so a round-2 401 would otherwise classify as unknown —
    the same unwrap `retry_on_transient_api_error` does, for the same reason.

    Anything that isn't a provider refusal classifies as `unknown`, so this is
    safe to call on every exception the brain logs, LLM-related or not.
    """
    if exc is None:
        return dict(_UNKNOWN_OUTCOME)
    for candidate in (exc, getattr(exc, '__cause__', None)):
        if candidate is None:
            continue
        outcome = _classify_one(candidate)
        if outcome['kind'] != LLM_UNKNOWN:
            return outcome
    return dict(_UNKNOWN_OUTCOME)


def load_env():
    """Load ANTHROPIC_API_KEY from the canonical config location.

    Source: ${XDG_CONFIG_HOME:-~/.config}/brain/env (dotenv format, mode 600).
    Matches the CLI-tool convention (gh, stripe, kubectl, ...).

    A key already present in os.environ (real shell env, daemon plist) wins
    and the file is not read.
    """
    if os.environ.get('ANTHROPIC_API_KEY'):
        return
    env_path = os.path.join(user_config_dir(), 'brain', 'env')
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if not os.environ.get(k):
                        os.environ[k] = v
    except Exception:
        # A profile-resolution failure must never crash an LLM call.
        return


def resolve_api_key() -> str:
    """The API key the daemon SHOULD be using RIGHT NOW.

    The env file is the canonical user-editable source — the dashboard's
    /setup form and boot-brain.sh's userConfig mirror both write it, and both
    promise "picked up automatically, no restart". load_env()'s no-override
    policy makes that promise false for key REPLACEMENT: once any key is in
    os.environ (first resolution, hook inheritance), a rewritten file is
    invisible (code review 2026-07-17, heal-truth finding). So: the file's
    current value wins when present; os.environ (real shell export, plist)
    is the fallback. Read every call — one stat + tiny read, only ever on
    LLM-availability checks, not the recall hot path.
    """
    env_path = os.path.join(user_config_dir(), 'brain', 'env')
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith('ANTHROPIC_API_KEY='):
                    v = line.split('=', 1)[1].strip()
                    if v:
                        return v
    except OSError:
        pass
    return os.environ.get('ANTHROPIC_API_KEY', '')


def key_fingerprint(key: str) -> str:
    """Short, non-reversible tag for a credential — enough to notice the key
    CHANGED, never enough to reconstruct it. The rejection latch has to tell
    "the operator re-enabled the same key" (wait for the clock) from "the
    operator pasted a new one" (resume now), and that state sits in memory and
    in log lines, where a raw key must never appear."""
    import hashlib
    return hashlib.sha256((key or '').encode()).hexdigest()[:12]


# The write commands a scale agent can issue — the canonical classification read
# by the attribution chokepoint (s2/base.apply_encoder_attribution) and the
# contract test (test_contract_sync Layer 5: must be a subset of COMMAND_TABLE).
WRITE_COMMANDS = {
    'remember', 'remember_batch', 'revise', 'revise_batch',
    'connect', 'connect_batch', 'brain_batch',
    'trace_append', 'set_config',
}

# The subset of WRITE_COMMANDS that create or attribute nodes/edges and so must
# carry the scale agent's encoding_source. Every command here reads a top-level
# `encoding_source` and cascades it (the batch handlers fan it out per-op).
# DERIVED from WRITE_COMMANDS (single source of truth) minus the explicitly
# non-attributed writes — so a newly-added attributed write is tagged
# automatically and can't silently slip through as 'anchor' (the gap that once
# mislabelled brain_batch encodes and hid them from the dashboard's encoder view).
NON_ATTRIBUTED_WRITES = {'trace_append', 'set_config'}
ATTRIBUTED_WRITE_COMMANDS = WRITE_COMMANDS - NON_ATTRIBUTED_WRITES

# Commands only the operator channel may issue — the shared S1/S2 encoder
# dispatch closure (scales/s2/base.py) refuses these by membership, and the
# owner method enforces anchor-only encoding_source at the write boundary.
# NOT derived from WRITE_COMMANDS: that set is an attribution classification,
# not a reachability gate — the closure passes any command it doesn't refuse.
OPERATOR_ONLY_COMMANDS = frozenset({'set_node_lock'})


# The scope-provenance dimension set has ONE source: contract.py derives it
# from PROMOTED_FIELDS' system_stamped flags. Re-exported here because the
# stamp machinery below and the S2 unit policies consume it at this layer.
from ..contract import SCOPE_PROVENANCE_FIELDS


def stamp_scope_provenance(cmd, cmd_args, scope):
    """Enforce deterministic scope provenance on an outgoing write's args.

    Scope fields on a node are PROVENANCE — the session context in which the
    node was learned: `project` (the repo, from SessionContext.project) and
    `counterpart` (who the session was with; today the install default).
    They are never agent-authored: node-creating payloads get the session's
    value force-stamped; agent-supplied values on any other write are
    dropped. Mutates cmd_args in place; returns a list of warning strings
    for values that were overridden or dropped (callers surface them).

    `scope` is {field: policy} over SCOPE_PROVENANCE_FIELDS; per-field
    policy value (None scope entirely → no-op):
      - None  → no-op for that field. The caller has no session authority
        (bare handler call without an ambient session); an upstream
        chokepoint may already have stamped, so args pass through untouched.
      - ''    → authoritative "none here": strip agent-supplied values
        (non-repo session, or an S2 unit — graph-scope work never invents
        provenance).
      - 'x'   → force-stamp onto node-creating payloads, strip elsewhere.
    """
    if not scope or not isinstance(cmd_args, dict):
        return []
    fields = [(f, scope[f]) for f in SCOPE_PROVENANCE_FIELDS
              if scope.get(f) is not None]
    if not fields:
        return []
    warnings = []

    def _force(node_dict, where):
        for field, value in fields:
            supplied = node_dict.get(field)
            if value:
                if supplied and supplied != value:
                    warnings.append(
                        "%s: %s is session-derived provenance — supplied "
                        "%r replaced with %r" % (where, field, supplied, value))
                node_dict[field] = value
            elif field in node_dict:
                # presence-based, not truthiness: an explicit '' is also
                # agent-authored and must not reach the node
                supplied = node_dict.pop(field)
                if supplied:
                    warnings.append(
                        "%s: %s is session-derived provenance and this "
                        "session has none — supplied %r dropped"
                        % (where, field, supplied))

    def _strip(d, where):
        # presence-based: `project: ''` through a revise would WIPE birth
        # provenance (validate_field accepts '', revise writes the column) —
        # pop the key whenever it appears, warn when it carried a value
        if not isinstance(d, dict):
            return
        for field, _ in fields:
            if field in d:
                supplied = d.pop(field)
                if supplied:
                    warnings.append(
                        "%s: %s is session-derived provenance — "
                        "agent-supplied %r dropped (set at node creation, "
                        "moved only by migration)" % (where, field, supplied))

    if cmd == 'remember':
        _force(cmd_args, 'remember')
    elif cmd == 'remember_batch':
        for i, spec in enumerate(cmd_args.get('nodes') or []):
            if isinstance(spec, dict):
                _force(spec, 'remember_batch.nodes[%d]' % i)
    elif cmd in ('revise', 'revise_batch'):
        if cmd == 'revise':
            _strip(cmd_args, 'revise')
        else:
            for i, spec in enumerate(cmd_args.get('revisions') or []):
                _strip(spec, 'revise_batch.revisions[%d]' % i)
    elif cmd == 'brain_batch':
        # force-vs-strip derives from the op contract (creates_node flag on
        # BATCH_OP_SPECS), not a local op enumeration — a new node-creating
        # op added via the documented contract path inherits the stamp.
        from ..contract import BATCH_OP_SPECS
        for i, op in enumerate(cmd_args.get('operations') or []):
            if not isinstance(op, dict):
                continue
            where = 'brain_batch.operations[%d]' % i
            spec = BATCH_OP_SPECS.get(op.get('op'), {})
            if spec.get('creates_node'):
                _force(op, where)
            else:
                _strip(op, where)
    return warnings
