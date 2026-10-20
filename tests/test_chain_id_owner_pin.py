"""Pin: keys derived from a session id are built only by their owners.

trace_contract.session_short owns the short form of a session id — the
session part of s0/s1r/s1e chain ids (SessionContext's generators and
CHAIN_PREFIXES format them), the session a log line names, and the short
every surface shows or addresses a stream by (presence, self-channel
messages, trace headers, the dashboard, reports); brain_constants.
session_marker_path owns per-session marker files. A copy built anywhere else
drifts from its owner without a sound — a hand-built payload glob matches
nothing, a log line's session_id[:8] lands in the error-dedup fingerprint,
where Codex's UUIDv7 ids collide, and a displayed head names a stream by a
form its own chains and logs never show.

Tuned to over-report: a flag on a legitimate line costs one ALLOWLIST entry
with its reason; a missed copy costs a silent bug. Checks:

  chain-literal    a string that builds an s0-/s1r-/s1e- chain (format
                   string, f-string, or 'prefix-' + …). Bare prefixes
                   ('s0-' in startswith) and docstrings pass.
  dash-key         a '-'-joined key ('%s-%s' % …, '-'.join([…]), f'{a}-{b}')
                   with a piece that names a session or a chain scale.
  session-slice    a [:N] / [0:N] / [-N:] cut of an expression that names a
                   session, or whose dict key or assignment target does.
  session-split    split/rsplit/partition(…)[k] on a session.
  session-truncate '%.Ns', f'{x:.N}' or '{:.N}'.format(…) on a session.
  short-keyword    session_short=… formatted from anything but the owner.
  marker-path      a 'brain-…' file name built from a session id.
  owner-copy       a session_short defined anywhere but its owner — the
                   other checks trust that name. Even inside an owner file,
                   only the owner's own definition is allowed (ALLOWLIST).
  id-strip         (JS) a hyphen strip of an id — how a short is derived;
                   only the dashboard's one copy, sessionShort, may.
  session-late     a brain log call whose session_short is not the first
                   placeholder within FINGERPRINT_ROOM characters: the
                   error-dedup fingerprint keeps the first 100 characters of
                   the message and of the context, so a session named later
                   merges two sessions' errors.

Message text is no exception: one session has one short, in a print() or a
raise as anywhere else. The owners are scanned too, exempt only from the
checks their own key-building trips.

Limits: a session is recognised by name (sid, session…, or a dict key /
assignment target naming one); an id held under target, id, ts or r[1]
passes unseen. Scope: servers, hooks, dashboard, scripts, eval — not
eval/fixtures (frozen records of past runs) and not tests (fixtures write
literal chains, and a test that restates a derivation fails loudly when it
changes).
"""
import ast
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Owner → the checks its own key-building trips (proved by the scanner);
# every other check still applies to it.
OWNERS = {"servers/trace_contract.py": {"chain-literal"},
          "servers/session_context.py": {"chain-literal"},
          "servers/brain_constants.py": {"marker-path", "dash-key"}}

SCAN_DIRS = ("servers", "dashboard", "hooks", "eval", "scripts")
SKIP_DIRS = ("__pycache__", "node_modules", ".venv", "venv", "data")
SKIP_PREFIXES = ("eval/fixtures/",)

# A chain prefix followed by a placeholder (format, glob, regex group or
# class) builds or matches chain ids; a fully literal id names one chain.
CHAIN_RE = re.compile(r"(?<![\w-])(s0|s1r|s1e)-[%{$*(\[]")
BARE_PREFIXES = {"s0-", "s1r-", "s1e-"}
SCALES = {"s0", "s1r", "s1e"}
# Identifier tokens that name a session: sid, hsid, sids, sid8, sess…
SESSION_TOKEN_RE = re.compile(r"^(h?sids?\d*|sess\w*)$")
SPLITS = {"split", "rsplit", "partition", "rpartition"}
PRECISION_RE = re.compile(r"%\.\d+s")
FORMAT_PRECISION_RE = re.compile(r"\{[^{}]*:[^{}]*\.\d+[^{}]*\}")
DASH_FORMAT_RE = re.compile(r"%[sdr]-%|\}-\{")
# The brain log calls whose message and context feed the dedup fingerprint
# (brain._log_event keeps message[:100] and context[:100]); the session must
# fit inside with room for session_short's 12 characters.
FINGERPRINT_CALLS = {"_log_error", "_log_warning", "loud", "log_error",
                     "log_warning"}
FINGERPRINT_ROOM = 88
PLACEHOLDER_RE = re.compile(
    r"%(?:\([^)]*\))?[-#0 +]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[a-zA-Z%]")

JS_CHAIN_RES = (re.compile(r"""['"`](s0|s1r|s1e)-[%{$*(\[]"""),
                re.compile(r"""['"`](s0|s1r|s1e)-['"`]\s*\+"""))
JS_SLICE_RE = re.compile(r"(sess|sid).*\.(slice|substring|substr)\(\s*-?\d", re.I)
# An id held under a bare name (id, key) is cut the same way.
JS_ID_CUT_RE = re.compile(
    r"\b(id|key|sid|sessionId|session_id)\b[^;]{0,24}?\.(slice|substring|substr)\(\s*-?\d")
JS_STRIP_RE = re.compile(r"replace\(/-/g,\s*''\)|replaceAll\(\s*['\"]-['\"]"
                         r"|split\(\s*['\"]-['\"]\s*\)\s*\.join\(")
SH_CHAIN_RE = re.compile(r"""(s0|s1r|s1e)-\$""")
SH_SLICE_RES = (re.compile(r"\$\{\w*(SESS|SID)\w*:\s*-?\d", re.I),
                re.compile(r"(SESS|SID).*\bcut\s+-c|\bcut\s+-c.*(SESS|SID)", re.I),
                re.compile(r"printf\s+['\"]%\.\d+s"),
                re.compile(r"\b(tail|head)\s+-c\s*-?\d"))

# (relpath, check, stripped source line) → why the line is legitimate.
ALLOWLIST = {
    ('servers/trace_contract.py', 'owner-copy',
     'def session_short(session_id):'):
        'the owner',
    ('servers/session_context.py', 'owner-copy',
     'def session_short(self) -> str:'):
        "SessionContext's property; delegates to the owner",
    ('dashboard/static/lib/sessions.js', 'id-strip',
     "return (id || '').replace(/-/g, '').slice(-12);"):
        "the dashboard's one copy of session_short (it cannot import servers); pinned by test_dashboard_disconnection",
    ('hooks/scripts/brain-env.sh', 'session-slice',
     '[ -n "${2:-}" ] && _name="$1-$(printf \'%s\' "$2" | tr -d \'-\' | tail -c 4)"'):
        'process tag: the last 4 of the session short (15-char process-name cap); pinned by test_deploy_contract',
    ('dashboard/static/lib/trace_friendly.js', 'session-slice',
     "return bar >= 0 ? { id: str.slice(0, bar), title: str.slice(bar + 1) } : { id: str, title: '' };"):
        "splits an 'id|title' string; no session",
    ('dashboard/static/lib/node_detail.js', 'session-slice',
     "el('span', null, 'id: ' + (n.id || '').substring(0, 8)),"):
        'node id, not a session',
    ('scripts/rehearse.sh', 'session-slice',
     'cp "$f" "$ART/bootstrap-$(echo "$f" | tr \'/\' \'_\' | tail -c 60)"'):
        'cuts a file path into an artifact name; no session',
    ('servers/recall_laf.py', 'short-keyword',
     "'chain_id': CHAIN_PREFIXES['s0'].format(session_short=short,"):
        're-assembles target chains from a short parsed out of stored chain ids; nodes_for_traces joins them by stop only',
    ('servers/dal_logs.py', 'session-slice',
     "'[' + ','.join(session_short(s) for s in session_ids[:3])"):
        'list slice: the first three session ids of a filter, each named by its short',
    ('servers/pipeline_contract.py', 'session-slice',
     'metadata[sid[:8]] = meta'):
        'node id named sid, not a session',
    ('servers/scales/s1/surface.py', 'session-slice',
     "sid = str(s.get('id') or '')[:8]"):
        'node id named sid/_sid, not a session (the context carries the full session id)',
    ('servers/scales/s1/surface.py', 'session-slice',
     "'— not rendered again' % _sid[:8],"):
        'node id named sid/_sid, not a session (the context carries the full session id)',
    ('dashboard/queries/recalls.py', 'chain-literal',
     "s0_by_s1 = {c: 's0-' + c[len('s1r-'):] for c in chain_ids"):
        'rewrites a stored s1r chain into its sibling s0 chain (same turn); derives nothing from a session id',
    ('eval/encoder_prompt_reassembly.py', 'chain-literal',
     "m = re.fullmatch(r's1e-([^-]+)-(\\d+)(?:\\.\\d+)?', chain)"):
        'parses payload dir names; the session part is [^-]+, no length assumed',
    ('eval/longmem/e1_cases.py', 'dash-key',
     "sid = 'i%s-%s-s%d' % (h[:7], qid, sess_idx)"):
        "eval session-id builder 'i<hash>-<qid>-s<n>'; sess_idx is an index, not an id",
    ('eval/longmem/build_corpus.py', 'dash-key',
     '"sid": "i%s-%s-s%d" % (h[:7], qid, sess_idx),'):
        "eval session-id builder 'i<hash>-<qid>-s<n>'; sess_idx is an index, not an id",
    ('eval/longmem/v_ladder.py', 'dash-key',
     'sid = "i%s-%s-s%d" % (h[:7], qid, sess_idx)'):
        "eval session-id builder 'i<hash>-<qid>-s<n>'; sess_idx is an index, not an id",
    ('eval/oracle_audit/recall_feature_table.py', 'session-slice',
     'sid = nid[:8]'):
        'node id named sid, not a session',
    ('eval/oracle_audit/ab_tool_use_audit.py', 'chain-literal',
     '"""SELECT chain_id, event_type, ref_type, ref_id, metadata, created_at, session_id'):
        "reader: SQL LIKE 's1r-%' over every session's recall chains",
    ('eval/agent_introspect/surface_review.py', 'session-slice',
     "sid_prefix = sid[:8] if isinstance(sid, str) else ''"):
        'node id named sid, not a session',
    ('eval/laf/walker/external_score.py', 'dash-key',
     "sid = 'i%s-%s-s%d' % (h[:7], qid, sess_idx)"):
        "eval session-id builder 'i<hash>-<qid>-s<n>'; sess_idx is an index, not an id",
    ('scripts/replay_surface.py', 'chain-literal',
     "db_dir, 'payloads', '*', 's1r-*', '*-judge*.json'))"):
        "reader: glob over every session's s1r payload dirs (and its --help text)",
    ('scripts/replay_surface.py', 'chain-literal',
     "help='Path to a judge payload '"):
        "reader: glob over every session's s1r payload dirs (and its --help text)",
}


def _scan_files():
    for base in SCAN_DIRS:
        root = os.path.join(REPO, base)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith((".py", ".js", ".sh")):
                    path = os.path.join(dirpath, fn)
                    rel = os.path.relpath(path, REPO)
                    if not rel.startswith(SKIP_PREFIXES):
                        yield path, rel


def _call_name(call):
    f = call.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")


def _int_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return True
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and _int_const(node.operand))


def _is_zero(node):
    return isinstance(node, ast.Constant) and node.value == 0


def _is_edge_slice(sl):
    """[:N], [0:N] or [-N:] — a prefix or suffix cut, the shape of a short id."""
    if not isinstance(sl, ast.Slice) or sl.step is not None:
        return False
    lower = None if sl.lower is not None and _is_zero(sl.lower) else sl.lower
    return ((lower is None and sl.upper is not None and _int_const(sl.upper))
            or (sl.upper is None and lower is not None and _int_const(lower)))


def _names_session(text):
    """True when an identifier or key names a session: split on _ and
    camelCase, any token like sid / session."""
    tokens = re.split(r"_|(?<=[a-z])(?=[A-Z])|[^A-Za-z0-9]", text or '')
    return any(SESSION_TOKEN_RE.match(t.lower()) for t in tokens if t)


def _mentions_session(node):
    """Does the expression itself name a session — sid, ctx.session_id,
    e['sid'], (session or ''), args.get('session_id'),
    session_id.replace('-', '')? Comprehensions and other containers are
    not descended into."""
    if isinstance(node, ast.Name):
        return _names_session(node.id)
    if isinstance(node, ast.Attribute):
        return _names_session(node.attr) or _mentions_session(node.value)
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and _names_session(node.value)
    if isinstance(node, ast.Subscript):
        return _mentions_session(node.value) or _mentions_session(node.slice)
    if isinstance(node, ast.BoolOp):
        return any(_mentions_session(v) for v in node.values)
    if isinstance(node, ast.IfExp):
        return _mentions_session(node.body) or _mentions_session(node.orelse)
    if isinstance(node, ast.Call):
        return (_names_session(_call_name(node))
                or (isinstance(node.func, ast.Attribute)
                    and _mentions_session(node.func.value))
                or any(_mentions_session(a) for a in node.args))
    return False


def _from_owner(node):
    """A piece the owner already derived: session_short(…) / x.session_short."""
    return ((isinstance(node, ast.Call) and _call_name(node) == "session_short")
            or (isinstance(node, ast.Attribute) and node.attr == "session_short"))


def _any_session_name(node):
    """Any identifier in the subtree naming a session — re.sub(…, session_id)
    included, the owner's own session_short excluded. String constants are
    left out: 'session=%s' is text."""
    if _from_owner(node):
        return False
    if isinstance(node, (ast.Name, ast.Attribute)) and _names_session(
            node.id if isinstance(node, ast.Name) else node.attr):
        return True
    return any(_any_session_name(c) for c in ast.iter_child_nodes(node))


def _target_names_session(node, parents):
    """The slice sits under a dict key or assignment target naming a session:
    {'session_short': (r[1] or '')[:8]}, session_x = …[:8]."""
    p = parents.get(node)
    while isinstance(p, (ast.IfExp, ast.BoolOp)):
        node, p = p, parents.get(p)
    if isinstance(p, ast.Dict):
        for k, v in zip(p.keys, p.values):
            if v is node and isinstance(k, ast.Constant):
                return _names_session(str(k.value))
    if isinstance(p, (ast.Assign, ast.AnnAssign)):
        targets = p.targets if isinstance(p, ast.Assign) else [p.target]
        return any(isinstance(t, ast.Name) and _names_session(t.id)
                   for t in targets)
    return False


def _fstring_text(node):
    return ''.join(v.value if isinstance(v, ast.Constant) else '{}'
                   for v in node.values)


def _add_operands(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _add_operands(node.left) + _add_operands(node.right)
    return [node]


def _brain_const(node):
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.startswith("brain-"))


def _marker_format(node):
    """A 'brain-…' file name built from a session id: % operands, f-string
    fields, .format() arguments, + concatenation, os.path.join pieces."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and _brain_const(node.left):
        return _any_session_name(node.right)
    if isinstance(node, ast.JoinedStr) and _fstring_text(node).startswith("brain-"):
        return any(_any_session_name(v) for v in node.values
                   if isinstance(v, ast.FormattedValue))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        ops = _add_operands(node)
        return (any(_brain_const(o) for o in ops)
                and any(_any_session_name(o) for o in ops))
    if isinstance(node, ast.Call):
        name = _call_name(node)
        if (name == "format" and isinstance(node.func, ast.Attribute)
                and _brain_const(node.func.value)):
            return any(_any_session_name(a) for a in
                       list(node.args) + [k.value for k in node.keywords])
        if name == "join" and any(_brain_const(a) for a in node.args):
            return any(_any_session_name(a) for a in node.args)
    return False


def _dash_piece(node):
    return (isinstance(node, ast.Constant) and node.value in SCALES) or _any_session_name(node)


def _dash_key(node):
    """A '-'-joined key with a piece naming a session or a chain scale."""
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and DASH_FORMAT_RE.search(str(node.left.value))):
        pieces = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        return any(_dash_piece(p) for p in pieces)
    if isinstance(node, ast.JoinedStr) and "}-{" in _fstring_text(node):
        return any(_dash_piece(v.value) for v in node.values
                   if isinstance(v, ast.FormattedValue))
    if (isinstance(node, ast.Call) and _call_name(node) == "join"
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Constant)
            and node.func.value.value == "-" and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))):
        return any(_dash_piece(p) for p in node.args[0].elts)
    return False


def _truncates_session(node):
    """'%.Ns' % sid, f'{sid:.N}', '{:.N}'.format(sid)."""
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and PRECISION_RE.search(str(node.left.value))):
        return _any_session_name(node.right)
    if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
        spec = _fstring_text(node.format_spec)
        return bool(re.search(r"\.\d", spec)) and _any_session_name(node.value)
    if (isinstance(node, ast.Call) and _call_name(node) == "format"
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Constant)
            and FORMAT_PRECISION_RE.search(str(node.func.value.value))):
        return any(_any_session_name(a) for a in
                   list(node.args) + [k.value for k in node.keywords])
    return False


def _split_index(node):
    """sid.split('-')[0]: an int index into a split of a session."""
    return (isinstance(node, ast.Subscript) and _int_const(node.slice)
            and isinstance(node.value, ast.Call)
            and _call_name(node.value) in SPLITS
            and isinstance(node.value.func, ast.Attribute)
            and _mentions_session(node.value.func.value))


def _in_fingerprint_call(node, parents):
    p = parents.get(node)
    while p is not None:
        if isinstance(p, ast.Call) and _call_name(p) in FINGERPRINT_CALLS:
            return True
        p = parents.get(p)
    return False


def _session_late(node):
    """A %-format or f-string naming the session (session_short) whose
    session is not the first placeholder, or starts past FINGERPRINT_ROOM."""
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)):
        pieces = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        owner = [i for i, piece in enumerate(pieces) if _from_owner(piece)]
        if not owner:
            return False
        holders = [m for m in PLACEHOLDER_RE.finditer(node.left.value)
                   if m.group() != "%%"]
        i = owner[0]
        return (i != 0 or not holders
                or holders[0].start() > FINGERPRINT_ROOM)
    if isinstance(node, ast.JoinedStr):
        fields = [v for v in node.values if isinstance(v, ast.FormattedValue)]
        if not any(_from_owner(v.value) for v in fields):
            return False
        lead = ''
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                return not _from_owner(v.value) or len(lead) > FINGERPRINT_ROOM
            lead += v.value
    return False


def _scan_python(text):
    """[(lineno, check)] for one module's source."""
    tree = ast.parse(text)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    hits = []
    for node in ast.walk(tree):
        if _marker_format(node):
            hits.append((node.lineno, "marker-path"))
        if _dash_key(node):
            hits.append((node.lineno, "dash-key"))
        if _truncates_session(node):
            hits.append((node.lineno, "session-truncate"))
        if _split_index(node):
            hits.append((node.lineno, "session-split"))
        if _session_late(node) and _in_fingerprint_call(node, parents):
            hits.append((node.lineno, "session-late"))
        if ((isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == "session_short")
                or (isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "session_short"
                    for t in node.targets))):
            hits.append((node.lineno, "owner-copy"))
        if isinstance(node, ast.JoinedStr):
            if CHAIN_RE.search(_fstring_text(node)):
                hits.append((node.lineno, "chain-literal"))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and not isinstance(parents.get(node), (ast.Expr, ast.JoinedStr))
              and CHAIN_RE.search(node.value)):
            hits.append((node.lineno, "chain-literal"))
        elif (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
              and isinstance(node.left, ast.Constant)
              and node.left.value in BARE_PREFIXES):
            hits.append((node.lineno, "chain-literal"))
        elif (isinstance(node, ast.Subscript) and _is_edge_slice(node.slice)
              and (_mentions_session(node.value)
                   or _target_names_session(node, parents))):
            hits.append((node.lineno, "session-slice"))
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "session_short" and not _from_owner(kw.value):
                    hits.append((node.lineno, "short-keyword"))
    return hits


def _scan_lines(text, rules, comment=("#",)):
    return [(i, check) for i, line in enumerate(text.splitlines(), 1)
            if not line.strip().startswith(comment)
            for check, regex in rules if regex.search(line)]


def _hits(rel, text):
    if rel.endswith(".py"):
        return _scan_python(text)
    if rel.endswith(".js"):
        return _scan_lines(text, [("chain-literal", r) for r in JS_CHAIN_RES]
                           + [("session-slice", JS_SLICE_RE),
                              ("session-slice", JS_ID_CUT_RE),
                              ("id-strip", JS_STRIP_RE)],
                           comment=("//", "/*", "*"))
    return _scan_lines(text, [("chain-literal", SH_CHAIN_RE)]
                       + [("session-slice", r) for r in SH_SLICE_RES])


def _offenders():
    found = []
    for path, rel in _scan_files():
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        lines = text.splitlines()
        for lineno, check in sorted(set(_hits(rel, text))):
            if check not in OWNERS.get(rel, ()):
                found.append((rel, check, lines[lineno - 1].strip()))
    return found


class TestSessionKeyOwnerPin(unittest.TestCase):

    def test_no_session_key_built_outside_its_owners(self):
        offenders = [o for o in _offenders() if o not in ALLOWLIST]
        self.assertFalse(
            offenders,
            "session-derived keys built outside their owners "
            "(trace_contract.session_short, SessionContext chain generators, "
            "brain_constants.session_marker_path) — route through the owner, "
            "or ALLOWLIST the line with its reason:\n"
            + "\n".join("%s [%s] %s" % o for o in offenders))

    def test_allowlist_entries_are_still_flagged(self):
        """An entry the scan no longer flags is stale — drop it, so the
        allowlist never silently covers a line that changed. An entry whose
        file is absent is not judged: the public tree ships no eval/ or
        scripts/, and the dev tree, where they exist, still judges them."""
        flagged = set(_offenders())
        stale = [e for e in ALLOWLIST
                 if e not in flagged and os.path.exists(os.path.join(REPO, e[0]))]
        self.assertFalse(stale, "stale ALLOWLIST entries: %r" % stale)

    def test_owners_still_build_their_keys(self):
        """An owner the scan no longer catches building its key means the
        key moved — update OWNERS so the pin guards the new home."""
        for rel, owned in OWNERS.items():
            with open(os.path.join(REPO, rel), encoding="utf-8") as f:
                checks = {c for _n, c in _hits(rel, f.read())}
            self.assertLessEqual(owned, checks,
                                 "%s no longer trips its owned checks %s"
                                 % (rel, sorted(owned - checks)))

    def test_catches_the_shapes_that_drifted(self):
        """The scan flags every shape a session-derived key has taken, or
        could take, outside its owner."""
        shapes = [
            "CHAIN_PREFIXES['s0'].format(session_short=session_id[:8], stop=0)",
            "enc_chain = 's1e-%s-%d' % (session_id[:8], counter)",
            "recall_ref = '%s-%s' % (session_id[:8], _current_stop)",
            "t = {'chain_id': 's0-%s-%d' % (short, ws)}",
            "sid8 = (session or '')[:8]",
            "shorts = {e['sid'][:8] for e in plan}",
            "journal.harvest(text, f's1e-{SID[:8]}-{wn}')",
            "p = os.path.join(d, 'brain-gate-%d-%s' % (os.getuid(), session_id[:8]))",
            "c = 's0-' + sid8 + '-' + str(stop)",
            "m = os.path.join(base, 'brain-hook-%s-%s-%s' % (moment, os.getuid(), re.sub(r'[^\\w-]', '_', session_id)))",
            "m = f'brain-fatigue-{session_id}.json'",
            "short = (session_id or '').replace('-', '')[-12:]",
            "key = session_id.replace('-', '')[:12]",
            "key = session_id[0:8]",
            "key = session_id.split('-')[0]",
            "k = hashlib.sha1(session_id.encode()).hexdigest()[:8]",
            "short = '%.8s' % session_id",
            "short = f'{session_id:.8}'",
            "short = '{:.8}'.format(session_id)",
            "p = 'brain-gate-{uid}-{sid}'.format(uid=os.getuid(), sid=session_id)",
            "p = os.path.join(d, 'brain-x-' + session_id)",
            "p = os.path.join(d, 'brain-x', session_id)",
            "c = '%s-%s-%d' % ('s0', sid, stop)",
            "c = '-'.join(['s0', short, str(stop)])",
            "c = f'{scale}-{sid}-{stop}'",
            "row = {'session_short': (r[1] or '')[:8]}",
            "brain._log_error('x', None, 'session=%s' % session_id[:8])",
            "def session_short(s):\n    return s[:8]",
            "session_short = lambda s: s[:8]",
            "brain._log_error('x', e, 'head: %s; session=%s' % (head, session_short(sid)))",
            "brain.loud('x', '%s session=%s' % (reader, session_short(sid)))",
            "brain._log_error('x', e, '" + "w" * 90 + " session=%s' % session_short(sid))",
            "brain._log_warning('x', f'{reader} session={session_short(sid)}')",
            "print('session=%s' % session_id[:8])",
            "raise ValueError('session %s' % sid[:8])",
            "print('[%s] %s' % (t['sid'][:8], label))",
            "sid = (row.get('session_id') or '')[:8]",
        ]
        missed = [s for s in shapes if not _scan_python(s + "\n")]
        self.assertFalse(missed, "scan misses: %r" % missed)
        quiet = ("print('session=%s' % session_short(session_id))\n"
                 "raise ValueError('session %s' % session_short(sid))\n"
                 "if chain.startswith('s1e-'): pass\n"
                 "p = os.path.join(transcripts, session_id + '.jsonl')\n"
                 "ref = '%s-%s' % (ctx.session_short, stop)\n"
                 "brain._log_error('x', e, 'session=%s stop=%d' % (session_short(sid), n))\n"
                 "brain._log_warning('x', f'session={session_short(sid)}: {n}')\n"
                 "def f():\n    '''s0-{short}-{stop} is the shape.'''\n")
        self.assertEqual(_scan_python(quiet), [])
        self.assertTrue(_scan_lines("x = (s.session_id || '').slice(0, 8);\n",
                                    [("session-slice", JS_SLICE_RE)]))
        self.assertTrue(_hits("x.sh", 'S=$(echo "$SESSION_ID" | cut -c1-8)\n'))
        self.assertTrue(_hits("x.js", "const s = id.slice(0, 8);\n"))
        self.assertTrue(_hits("x.js", "const s = (c.key || '').slice(-4);\n"))
        self.assertTrue(_hits("x.js", "const s = sid.replace(/-/g, '');\n"))
        self.assertTrue(_hits("x.js", "const s = id.replaceAll('-', '');\n"))
        self.assertTrue(_hits("x.js", "const s = id.split('-').join('');\n"))
        self.assertTrue(_hits("x.sh", 'T=$(printf %s "$2" | tail -c 4)\n'))


if __name__ == "__main__":
    unittest.main()
