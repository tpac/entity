"""Contract sync test for the trace system.

Scans all trace write sites in the codebase and verifies they only use
scale/event_type/ref_type combinations that exist in the trace contract.

Run after EVERY change to trace writers or trace_contract.py.

Run: python3 -m pytest tests/test_trace_contract_sync.py -v
"""
import ast
import os
import re
import pytest

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files that contain trace writes (production code only, not tests)
TRACE_WRITER_FILES = [
    'servers/daemon_hooks.py',
    'servers/channels/delivery.py',  # the last-mile leg traces each delivery (s0/K per source)
    'servers/brain.py',            # stamp_boot_liveness writes a boot heartbeat (s0/K/heartbeat)
    'servers/brain_traces.py',     # write_journal_notes batches journal_note rows
    'servers/mutation_emitter.py', # THE mutation-trace writer (node_created/archived/deleted)
    'servers/scales/s1/encode.py',
    'servers/scales/s1/surface.py',
    'servers/scales/s2/community.py',
    'servers/scales/s2/community_decoder.py',
    'servers/scales/s2/community_encoder.py',
    'servers/scales/s2/consolidation_decoder.py',
    'servers/scales/s2/consolidation_encoder.py',
    'servers/scales/s2/healer_decoder.py',
    'servers/scales/s2/healer_encoder.py',
    'servers/scales/s2/aspect_decoder.py',
    'servers/scales/s2/aspect_encoder.py',
    # reclassify.py retired to servers/scales/s2/archive/ in 3df4181 — its
    # community_assignments trace write went with it (no successor inherited it;
    # the ref_type lives on in trace_contract.py for historic events only).
    'hooks/scripts/post_tool_trace.py',
]


# ═══════════════════════════════════════════════════════
# Trace-write extractor (AST)
# ═══════════════════════════════════════════════════════
# The trace substrate has FOUR write doors, and a writer reaches one of them
# either directly or through a helper that binds part of the triple. The
# extractor resolves each shape below; anything it cannot resolve statically is
# invisible, which is what KNOWN_EXTRACTOR_BLIND ratchets.
#
#   door / shape                              scale comes from
#   ────────────────────────────────────────────────────────────────────────
#   x._trace_dal.append(scale=, event_type=,  the `scale` kwarg
#     ref_type=)
#   x._trace_dal.append_batch([{...}, ...])   each element's 'scale' key
#   _s0_trace(brain, ctx, event_type=,        the helper — it hardcodes 's0'
#     ref_type=)                              (brain_traces._s0_trace)
#   self.trace(event_type, ref_type, ...)     the enclosing class's SCALE
#     (S2Unit.trace)                          attribute, inherited if needed
#   dispatch*('trace_append', {...})          the dict's 'scale' key
#   {"cmd": "trace_append", "args": {...}}    the args dict's 'scale' key
#
# A regex can't do this: `append(chain_id=ctx.s0_chain(), scale='s0', ...)`
# defeats any `[^)]*` scan on the nested call's paren, which is exactly how
# brain.py and daemon_hooks.py went unseen.

# Helpers that bind the scale themselves, so their call sites never pass one.
SCALE_BINDING_HELPERS = {'_s0_trace': 's0'}

# Call names that dispatch a command by string: fn('trace_append', {payload}).
DISPATCH_CALL_NAMES = {'dispatch_fn', 'dispatch', '_daemon_tcp_send', 'send_command'}


def _str(node):
    """The literal str value of an AST node, or '' if it isn't a str literal."""
    return node.value if isinstance(node, ast.Constant) and isinstance(
        node.value, str) else ''


def _kwarg(call, name):
    """Literal str value of a keyword arg on a Call, or ''."""
    for kw in call.keywords:
        if kw.arg == name:
            return _str(kw.value)
    return ''


def _dict_get(node, key):
    """Literal str value for `key` in a dict literal (or a dict(...) call), or ''."""
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if _str(k) == key:
                return _str(v)
        return ''
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == 'dict':
        return _kwarg(node, key)
    return ''


def _dict_node(node):
    """True for a dict literal or a dict(...) call — the two payload shapes."""
    return isinstance(node, ast.Dict) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'dict')


def _attr_name(func):
    """Trailing attribute name of a call target ('append' for x.y.append)."""
    return func.attr if isinstance(func, ast.Attribute) else ''


def _is_trace_dal_call(func, method):
    """True for `<anything>._trace_dal.<method>` — the DAL door."""
    return (isinstance(func, ast.Attribute) and func.attr == method
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == '_trace_dal')


def _class_scale_index():
    """Map class name → SCALE literal across servers/, resolving inheritance.

    S2 units carry their scale as a class attribute (`SCALE = 's2'`) and call
    `self.trace(event_type, ref_type, ...)` without one — so resolving
    `self.trace` needs the class's SCALE, and subclasses like
    CommunityDetection(CommunityDecoder) inherit it from another FILE. Built
    once per session (module-level cache below).
    """
    import glob
    own, bases = {}, {}
    for path in glob.glob(os.path.join(ROOT, 'servers', '**', '*.py'),
                          recursive=True):
        try:
            tree = ast.parse(open(path).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases[node.name] = [b.id for b in node.bases
                                if isinstance(b, ast.Name)]
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == 'SCALE'
                        for t in stmt.targets):
                    if _str(stmt.value):
                        own[node.name] = _str(stmt.value)

    def resolve(name, seen=()):
        if name in own:
            return own[name]
        if name in seen:
            return ''
        for base in bases.get(name, []):
            got = resolve(base, seen + (name,))
            if got:
                return got
        return ''

    return {name: resolve(name) for name in bases}


_CLASS_SCALES = None


def _class_scale(name):
    global _CLASS_SCALES
    if _CLASS_SCALES is None:
        _CLASS_SCALES = _class_scale_index()
    return _CLASS_SCALES.get(name, '')


class _TraceWriteVisitor(ast.NodeVisitor):
    """Collect (scale, event_type, ref_type, filepath, lineno) per write site."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.triples = []
        self._class_stack = []

    def visit_ClassDef(self, node):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _add(self, scale, event_type, ref_type, node):
        # Only record a site the extractor fully resolved on the two fields the
        # contract keys on. A partial resolve is blindness, not a triple.
        if scale and event_type:
            self.triples.append((scale, event_type, ref_type or '',
                                 self.filepath, node.lineno))

    def _add_payload(self, payload, node):
        self._add(_dict_get(payload, 'scale'), _dict_get(payload, 'event_type'),
                  _dict_get(payload, 'ref_type'), node)

    def visit_Call(self, node):
        func = node.func

        # ── Door 1: x._trace_dal.append(scale=..., event_type=..., ref_type=...)
        if _is_trace_dal_call(func, 'append'):
            self._add(_kwarg(node, 'scale'), _kwarg(node, 'event_type'),
                      _kwarg(node, 'ref_type'), node)

        # ── Door 2: x._trace_dal.append_batch([{...}, dict(...), ...])
        elif _is_trace_dal_call(func, 'append_batch'):
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                for elt in node.args[0].elts:
                    if _dict_node(elt):
                        self._add_payload(elt, elt)

        # ── Door 3: a helper that binds the scale (e.g. _s0_trace → 's0')
        elif isinstance(func, ast.Name) and func.id in SCALE_BINDING_HELPERS:
            self._add(SCALE_BINDING_HELPERS[func.id],
                      _kwarg(node, 'event_type'), _kwarg(node, 'ref_type'), node)

        # ── Door 4: S2Unit.trace(event_type, ref_type, ...) — scale from the class
        elif _attr_name(func) == 'trace' and isinstance(func.value, ast.Name) \
                and func.value.id == 'self':
            args = node.args
            event_type = _str(args[0]) if len(args) > 0 else _kwarg(node, 'event_type')
            ref_type = _str(args[1]) if len(args) > 1 else _kwarg(node, 'ref_type')
            scale = _class_scale(self._class_stack[-1]) if self._class_stack else ''
            self._add(scale, event_type, ref_type, node)

        # ── Door 5: dispatch_fn('trace_append', {...})
        elif isinstance(func, ast.Name) and func.id in DISPATCH_CALL_NAMES \
                and len(node.args) >= 2 and _str(node.args[0]) == 'trace_append' \
                and _dict_node(node.args[1]):
            self._add_payload(node.args[1], node)
        elif _attr_name(func) in DISPATCH_CALL_NAMES \
                and len(node.args) >= 2 and _str(node.args[0]) == 'trace_append' \
                and _dict_node(node.args[1]):
            self._add_payload(node.args[1], node)

        self.generic_visit(node)


def _extract_trace_writes_from_file(filepath):
    """Extract (scale, event_type, ref_type, filepath, lineno) per trace write.

    AST-based: see the door table above for the shapes it resolves. Sites whose
    triple isn't statically knowable (a payload built in a loop, a scale read
    from a variable) yield nothing — a file where that is ALL the sites is
    extractor-blind and must be declared in KNOWN_EXTRACTOR_BLIND.
    """
    with open(os.path.join(ROOT, filepath)) as f:
        tree = ast.parse(f.read())

    # Door 6: a JSON command dict — {"cmd": "trace_append", "args": {...}} —
    # handed to json.dumps and written to the daemon socket (hook scripts,
    # which have no brain object to reach the DAL through). Matched on the
    # dict shape itself, wherever it appears.
    visitor = _TraceWriteVisitor(filepath)
    visitor.visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and _dict_get(node, 'cmd') == 'trace_append':
            for k, v in zip(node.keys, node.values):
                if _str(k) == 'args' and _dict_node(v):
                    visitor._add_payload(v, node)

    return visitor.triples


def _extract_chain_patterns(filepath):
    """Extract chain_id format patterns from a file."""
    full_path = os.path.join(ROOT, filepath)
    with open(full_path) as f:
        content = f.read()

    patterns = []
    # Match: chain_id = "s0-..." or chain_id="s1r-..." etc
    for match in re.finditer(
            r'chain_id\s*[=:]\s*["\']?(s[0-4][re]?)-',
            content):
        patterns.append((match.group(1), filepath, match.start()))

    # Match: "chain_id": "s0-..." in JSON dict
    for match in re.finditer(
            r'["\']chain_id["\']\s*:\s*["\'](s[0-4][re]?)-',
            content):
        patterns.append((match.group(1), filepath, match.start()))

    return patterns


# ═══════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════

class TestTraceContractSync:
    """Verify all trace writers in the codebase match the contract."""

    # Writer files the extractor cannot see, and WHY. A file listed here
    # contributes nothing to the three contract assertions below — its triples
    # are unchecked — so the entry is a debt marker, not an exemption to reach
    # for. The ratchet works both ways: a NEW blind file fails (the contract
    # silently stopped covering a writer), and a file that HEALS fails too
    # (delete the entry, the checks now cover it).
    #
    # Both entries build their payloads in a loop from runtime values, so no
    # (scale, event_type, ref_type) triple exists in the source to read.
    KNOWN_EXTRACTOR_BLIND = {
        # Validates itself instead: _emit_mutation_traces calls
        # validate_trace_event(scale, 'delta', ref_type) per row before writing.
        'servers/mutation_emitter.py',
    }
    # servers/brain_traces.py is visible through write_tool_moment's (s1, K)
    # call; its ref_type is the REF_TOOL_SHOWN name, which the extractor reads
    # as '' — validate_trace_event checks it at write time. Two doors are not
    # visible at all: write_journal_notes (ref_type
    # the literal 'journal_note', `scale` the caller's — s1 Scribe or an S2
    # unit — covered by the (s1|s2, delta, journal_note) registrations) and
    # write_thalamus_filed (scale DERIVED from the chain via scale_for_chain,
    # ref_type the REF_THALAMUS_FILED name — covered by the (s1|s2, delta)
    # registrations; an unregistered scale is caught at the write boundary and
    # pinned by test_thalamus.TestFiledTrace).

    def test_extractor_sees_every_writer_file(self):
        """Every declared writer file yields at least one triple — or is a
        declared blind spot. Guards the failure mode this whole file has: an
        assertion that iterates an empty list passes vacuously, so a writer
        going invisible looks exactly like a writer that's fine."""
        blind, healed = [], []
        for filepath in TRACE_WRITER_FILES:
            visible = bool(_extract_trace_writes_from_file(filepath))
            known = filepath in self.KNOWN_EXTRACTOR_BLIND
            if not visible and not known:
                blind.append(filepath)
            elif visible and known:
                healed.append(filepath)

        assert not blind, (
            "Trace writes in these files are invisible to the extractor, so "
            "their scale/event_type/ref_type triples are NOT contract-checked: "
            "%s\nEither teach _TraceWriteVisitor the call shape (see the door "
            "table), or declare it in KNOWN_EXTRACTOR_BLIND with the reason."
            % blind)
        assert not healed, (
            "These files are now visible to the extractor — remove them from "
            "KNOWN_EXTRACTOR_BLIND so their triples are checked: %s" % healed)

    def test_known_blind_files_are_writer_files(self):
        """KNOWN_EXTRACTOR_BLIND can only name files that are actually declared
        writers — otherwise an entry outlives the file it excused."""
        stale = self.KNOWN_EXTRACTOR_BLIND - set(TRACE_WRITER_FILES)
        assert not stale, (
            "KNOWN_EXTRACTOR_BLIND names non-writer files: %s" % sorted(stale))

    def setup_method(self):
        from servers.trace_contract import validate_trace_event, SCALES, CHAIN_PREFIXES
        self.validate = validate_trace_event
        self.SCALES = SCALES
        self.CHAIN_PREFIXES = CHAIN_PREFIXES

    def test_all_writer_files_exist(self):
        """All listed trace writer files exist."""
        for f in TRACE_WRITER_FILES:
            full = os.path.join(ROOT, f)
            assert os.path.exists(full), "Trace writer file missing: %s" % f

    def test_all_trace_writes_use_valid_scales(self):
        """Every trace write uses a scale defined in the contract."""
        for filepath in TRACE_WRITER_FILES:
            triples = _extract_trace_writes_from_file(filepath)
            for scale, event_type, ref_type, fpath, line in triples:
                assert scale in self.SCALES, \
                    "Invalid scale '%s' in %s:%d (event_type=%s, ref_type=%s)" % (
                        scale, fpath, line, event_type, ref_type)

    def test_all_trace_writes_use_valid_event_types(self):
        """Every trace write uses a valid event_type."""
        from servers.trace_contract import EVENT_TYPES
        for filepath in TRACE_WRITER_FILES:
            triples = _extract_trace_writes_from_file(filepath)
            for scale, event_type, ref_type, fpath, line in triples:
                assert event_type in EVENT_TYPES, \
                    "Invalid event_type '%s' in %s:%d (scale=%s, ref_type=%s)" % (
                        event_type, fpath, line, scale, ref_type)

    def test_all_trace_writes_use_valid_ref_types(self):
        """Every (scale, event_type, ref_type) triple passes contract validation."""
        for filepath in TRACE_WRITER_FILES:
            triples = _extract_trace_writes_from_file(filepath)
            for scale, event_type, ref_type, fpath, line in triples:
                ok, error = self.validate(scale, event_type, ref_type)
                assert ok, "Contract violation in %s:%d: %s (scale=%s, event_type=%s, ref_type=%s)" % (
                    fpath, line, error, scale, event_type, ref_type)

    def test_all_chain_ids_follow_conventions(self):
        """Chain ID prefixes match CHAIN_PREFIXES patterns."""
        valid_prefixes = set()
        for key, pattern in self.CHAIN_PREFIXES.items():
            # Extract the prefix before the first {
            prefix = pattern.split('{')[0]
            valid_prefixes.add(prefix)

        for filepath in TRACE_WRITER_FILES:
            patterns = _extract_chain_patterns(filepath)
            for prefix, fpath, pos in patterns:
                matched = any(prefix + '-' == vp or prefix == vp.rstrip('-')
                              for vp in valid_prefixes)
                if not matched:
                    # Check if prefix- matches any valid prefix pattern
                    matched = any((prefix + '-').startswith(vp) for vp in valid_prefixes)
                assert matched, \
                    "Chain prefix '%s' in %s doesn't match any CHAIN_PREFIXES: %s" % (
                        prefix, fpath, valid_prefixes)

    def test_contract_covers_s0_and_s1(self):
        """S0 and S1 have REF_TYPES defined for all event types in use."""
        from servers.trace_contract import REF_TYPES
        for scale in ('s0', 's1'):
            # At minimum, K and delta should have ref_types defined
            assert (scale, 'K') in REF_TYPES, \
                "Missing REF_TYPES for (%s, K)" % scale
            assert (scale, 'delta') in REF_TYPES, \
                "Missing REF_TYPES for (%s, delta)" % scale

    def test_no_trace_writes_outside_known_files(self):
        """No unexpected files contain trace writes (catch rogue writers)."""
        import glob
        all_py = glob.glob(os.path.join(ROOT, 'servers', '*.py'))
        all_py += glob.glob(os.path.join(ROOT, 'hooks', 'scripts', '*.py'))

        known_writers = set(os.path.join(ROOT, f) for f in TRACE_WRITER_FILES)
        # Also allow dal.py (the DAL itself) and the dispatch layer. The dispatch
        # handlers split out of daemon_dispatch 2026-05-28: revise/edge trace
        # emitters live in dispatch_write.py, the trace_append handler in
        # dispatch_observability.py.
        known_writers.add(os.path.join(ROOT, 'servers', 'dal.py'))
        known_writers.add(os.path.join(ROOT, 'servers', 'daemon_dispatch.py'))
        known_writers.add(os.path.join(ROOT, 'servers', 'dispatch_write.py'))
        known_writers.add(os.path.join(ROOT, 'servers', 'dispatch_observability.py'))

        for py_file in all_py:
            if py_file in known_writers:
                continue
            with open(py_file) as f:
                content = f.read()
            # Check for direct trace writes (not imports or comments).
            # append_batch is its own door — matching only 'append(' is how
            # mutation_emitter.py and brain_traces.py wrote traces for months
            # without ever being declared here.
            doors = ('trace_dal.append(', 'trace_dal.append_batch(',
                     "trace_append'")
            if any(d in content for d in doors):
                # Ignore if it's just a reference in a comment or string
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue
                    if any(d in stripped for d in doors):
                        assert False, \
                            "Unexpected trace write in %s line %d: %s\n" \
                            "Add to TRACE_WRITER_FILES in test_trace_contract_sync.py" % (
                                os.path.basename(py_file), i + 1, stripped[:80])

    def test_every_declared_payload_shape_is_enforced(self):
        """validate_trace_metadata reads only METADATA_REQUIRED_BY_REF_TYPE, so
        a *_SHAPE dict declared in trace_contract but never registered there
        looks like a payload contract and checks nothing."""
        import servers.trace_contract as tc
        declared = {name for name, value in vars(tc).items()
                    if name.endswith('_SHAPE') and isinstance(value, dict)}
        assert 'DELTA_METADATA_SHAPE' in declared  # the scan sees shapes at all
        enforced = {id(shape) for shape in tc.METADATA_REQUIRED_BY_REF_TYPE.values()}
        unenforced = sorted(n for n in declared if id(getattr(tc, n)) not in enforced)
        assert not unenforced, (
            "declared but never registered in METADATA_REQUIRED_BY_REF_TYPE "
            "(so never validated): %s" % unenforced)


# ═══════════════════════════════════════════════════════
# Revise contract (Stage 1A — added 2026-05-04)
# ═══════════════════════════════════════════════════════

class TestReviseContract:
    """Verify the node_revised event contract.

    Stage 1A: every revise() call emits a (delta, node_revised) trace event
    carrying field-level deltas + reason. Replaces the old _sys_revision_history
    KV blob as the canonical revision history substrate.
    """

    def test_node_revised_registered_for_all_revise_origins(self):
        """node_revised must be a valid ref_type for s0/s1/s2 delta events.

        Revisions originate at every scale: S0 (operator via MCP), S1 (encoder),
        S2 (healer, consolidation, future aspect_integration).
        """
        from servers.trace_contract import REF_TYPES
        for scale in ('s0', 's1', 's2'):
            ref_types = REF_TYPES[(scale, 'delta')]
            assert 'node_revised' in ref_types, (
                "node_revised missing from (%s, delta) ref_types: %s" % (
                    scale, ref_types))

    def test_validate_accepts_node_revised(self):
        """validate_trace_event accepts (scale, 'delta', 'node_revised') for s0/s1/s2."""
        from servers.trace_contract import validate_trace_event
        for scale in ('s0', 's1', 's2'):
            ok, err = validate_trace_event(scale, 'delta', 'node_revised')
            assert ok, "Validation failed for (%s, delta, node_revised): %s" % (
                scale, err)

    def test_revise_metadata_shape_exists(self):
        """REVISE_METADATA_SHAPE is exported with the expected keys."""
        from servers.trace_contract import REVISE_METADATA_SHAPE
        expected_keys = {'node_id', 'reason', 'encoding_source',
                         'deltas', 'warnings'}
        assert set(REVISE_METADATA_SHAPE.keys()) == expected_keys, (
            "REVISE_METADATA_SHAPE keys: %s, expected: %s" % (
                set(REVISE_METADATA_SHAPE.keys()), expected_keys))

    def test_build_revise_metadata_full_call(self):
        """build_revise_metadata preserves all passed values."""
        from servers.trace_contract import build_revise_metadata
        meta = build_revise_metadata(
            node_id='test123',
            reason='unit test',
            encoding_source='test:phase_a',
            deltas=[{'field': 'situation', 'old': 'before', 'new': 'after'}],
            warnings=['immutable field skipped: id'],
        )
        assert meta['node_id'] == 'test123'
        assert meta['reason'] == 'unit test'
        assert meta['encoding_source'] == 'test:phase_a'
        assert meta['deltas'] == [
            {'field': 'situation', 'old': 'before', 'new': 'after'}]
        assert meta['warnings'] == ['immutable field skipped: id']

    def test_build_revise_metadata_defaults(self):
        """build_revise_metadata fills sensible defaults for omitted args."""
        from servers.trace_contract import build_revise_metadata
        meta = build_revise_metadata(node_id='test', reason='r')
        assert meta['node_id'] == 'test'
        assert meta['reason'] == 'r'
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []

    def test_build_revise_metadata_handles_none(self):
        """build_revise_metadata coerces None values to safe defaults."""
        from servers.trace_contract import build_revise_metadata
        meta = build_revise_metadata(
            node_id='test', reason=None,
            encoding_source=None, deltas=None, warnings=None)
        assert meta['reason'] == ''
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []


# ═══════════════════════════════════════════════════════
# Edge revise contract (Stage 1B — added 2026-05-04)
# ═══════════════════════════════════════════════════════

class TestEdgeReviseContract:
    """Verify the edge_relation_revised event contract.

    Stage 1B: every connect upsert (create or field update) and every
    archive that targets an edge_relation emits one trace event with
    field-level deltas, mirroring node revise semantics. Single ref_type
    covers create-via-upsert AND update-via-upsert AND polymorphic archive.

    ref_id encoding: f"{edge_id}:{relation}".
    """

    def test_edge_relation_revised_registered_for_all_revise_origins(self):
        """edge_relation_revised must be a valid ref_type for s0/s1/s2 delta."""
        from servers.trace_contract import REF_TYPES
        for scale in ('s0', 's1', 's2'):
            ref_types = REF_TYPES[(scale, 'delta')]
            assert 'edge_relation_revised' in ref_types, (
                "edge_relation_revised missing from (%s, delta) ref_types: %s" % (
                    scale, ref_types))

    def test_validate_accepts_edge_relation_revised(self):
        """validate_trace_event accepts (scale, 'delta', 'edge_relation_revised')."""
        from servers.trace_contract import validate_trace_event
        for scale in ('s0', 's1', 's2'):
            ok, err = validate_trace_event(scale, 'delta', 'edge_relation_revised')
            assert ok, "Validation failed for (%s, delta, edge_relation_revised): %s" % (
                scale, err)

    def test_edge_revise_metadata_shape_exists(self):
        """EDGE_REVISE_METADATA_SHAPE is exported with the expected keys.

        source_id/target_id carry the directional pair so an edge is
        reconstructable from the trace alone (edge_id is a one-way hash)."""
        from servers.trace_contract import EDGE_REVISE_METADATA_SHAPE
        expected_keys = {'edge_id', 'source_id', 'target_id', 'relation',
                         'reason', 'encoding_source', 'deltas', 'warnings'}
        assert set(EDGE_REVISE_METADATA_SHAPE.keys()) == expected_keys, (
            "EDGE_REVISE_METADATA_SHAPE keys: %s, expected: %s" % (
                set(EDGE_REVISE_METADATA_SHAPE.keys()), expected_keys))

    def test_build_edge_revise_metadata_full_call(self):
        """build_edge_revise_metadata preserves all passed values."""
        from servers.trace_contract import build_edge_revise_metadata
        meta = build_edge_revise_metadata(
            edge_id='edg_a3f12c',
            relation='extends',
            reason='upsert via connect',
            encoding_source='encoder:sonnet',
            deltas=[{'field': 'description', 'old': 'old', 'new': 'new'}],
            warnings=[],
        )
        assert meta['edge_id'] == 'edg_a3f12c'
        assert meta['relation'] == 'extends'
        assert meta['reason'] == 'upsert via connect'
        assert meta['encoding_source'] == 'encoder:sonnet'
        assert meta['deltas'] == [
            {'field': 'description', 'old': 'old', 'new': 'new'}]
        assert meta['warnings'] == []

    def test_build_edge_revise_metadata_defaults(self):
        """build_edge_revise_metadata fills sensible defaults for omitted args."""
        from servers.trace_contract import build_edge_revise_metadata
        meta = build_edge_revise_metadata(
            edge_id='edg_a3f12c', relation='extends', reason='r')
        assert meta['edge_id'] == 'edg_a3f12c'
        assert meta['relation'] == 'extends'
        assert meta['reason'] == 'r'
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []

    def test_build_edge_revise_metadata_handles_none(self):
        """build_edge_revise_metadata coerces None values to safe defaults."""
        from servers.trace_contract import build_edge_revise_metadata
        meta = build_edge_revise_metadata(
            edge_id='edg_x', relation='r', reason=None,
            encoding_source=None, deltas=None, warnings=None)
        assert meta['reason'] == ''
        assert meta['encoding_source'] == ''
        assert meta['deltas'] == []
        assert meta['warnings'] == []

    def test_create_via_upsert_uses_same_ref_type(self):
        """Create-via-upsert uses edge_relation_revised with empty `old` in deltas.

        Single ref_type for create + update is the design call (Stage 1B Option A —
        mirror node pattern). Empty `old` in a delta semantically means 'created'.
        """
        from servers.trace_contract import build_edge_revise_metadata, validate_trace_event
        # Validates the contract supports it
        ok, _ = validate_trace_event('s2', 'delta', 'edge_relation_revised')
        assert ok
        # Builder accepts create-style deltas (old=None)
        meta = build_edge_revise_metadata(
            edge_id='edg_new', relation='extends', reason='created',
            encoding_source='encoder:sonnet',
            deltas=[
                {'field': 'description', 'old': None, 'new': 'new desc'},
                {'field': 'weight', 'old': None, 'new': 0.7},
            ])
        assert len(meta['deltas']) == 2
        assert all(d['old'] is None for d in meta['deltas'])


class TestS0TurnClassification:
    """The S0 turn-classification contract — single source of truth for which
    turns are conversation worth encoding (vs. recorded-only). See the
    'S0 TURN CLASSIFICATION' block in trace_contract.py."""

    def test_heartbeat_is_a_valid_s0_incoming_ref_type(self):
        from servers.trace_contract import validate_trace_event
        ok, _ = validate_trace_event('s0', 'K', 'heartbeat')
        assert ok, "heartbeat must be a valid (s0, K) ref_type"

    def test_the_dial_as_decided(self):
        # Locks the decisions: operator prompts encode; the prompt reader's
        # harness and subagent rows encode (each opens a turn holding my
        # reply); anchor↔anchor and brain↔anchor stay OFF until their
        # flip-day checklist clears; heartbeats never. Flipping a row is a
        # deliberate change.
        from servers.trace_contract import S0_CONVERSATIONAL_INCOMING
        assert S0_CONVERSATIONAL_INCOMING['user_message'] is True
        assert S0_CONVERSATIONAL_INCOMING['self_message'] is False
        assert S0_CONVERSATIONAL_INCOMING['thalamus_delivery'] is False
        assert S0_CONVERSATIONAL_INCOMING['env_message'] is True
        assert S0_CONVERSATIONAL_INCOMING['subagent_message'] is True
        assert S0_CONVERSATIONAL_INCOMING['heartbeat'] is False

    def test_every_dial_row_but_heartbeat_has_a_speaker(self):
        # One speaker per incoming ref_type the dial can turn on; a heartbeat
        # carries no message, so it has none. A correspondent added to the
        # dial without a speaker would render under nobody's name.
        from servers.trace_contract import (
            INCOMING_SPEAKERS, S0_CONVERSATIONAL_INCOMING)
        assert set(INCOMING_SPEAKERS) == set(S0_CONVERSATIONAL_INCOMING) - {'heartbeat'}
        assert INCOMING_SPEAKERS['user_message'] == 'operator'
        assert len(set(INCOMING_SPEAKERS.values())) == len(INCOMING_SPEAKERS)

    def test_prompt_pieces_are_written_only_as_dial_on_rows(self):
        from servers.trace_contract import (
            CADENCE_REF_TYPES, S0_CONVERSATIONAL_INCOMING, SPEAKER_REF_TYPES)
        assert all(S0_CONVERSATIONAL_INCOMING[rt] for rt in SPEAKER_REF_TYPES.values())
        assert 'session' not in SPEAKER_REF_TYPES and 'brain' not in SPEAKER_REF_TYPES
        assert all(S0_CONVERSATIONAL_INCOMING[rt] for rt in CADENCE_REF_TYPES)
        assert 'env_message' not in CADENCE_REF_TYPES   # a notice never brings an encode due

    def test_operator_dialogue_is_a_subset_of_conversational(self):
        # Option A's safety property: every pinned consumer (presence,
        # episodes default, LAF, dual-store) selects only ref_types the
        # encoder whitelist also carries — so a pinned scope can never see a
        # row the timeline excludes. Aliasing the two constants back together
        # ("dedupe the twin tuples") is the regression this guards.
        from servers.trace_contract import (
            CONVERSATIONAL_REF_TYPES, OPERATOR_DIALOGUE_REF_TYPES)
        assert set(OPERATOR_DIALOGUE_REF_TYPES) <= set(CONVERSATIONAL_REF_TYPES)
        assert OPERATOR_DIALOGUE_REF_TYPES == ('user_message', 'assistant_message')

    def test_conversational_ref_types_derived_from_one_dial(self):
        # CONVERSATIONAL_REF_TYPES must be DERIVED from S0_CONVERSATIONAL_INCOMING
        # (the single dial) + the assistant response side — never hardcoded.
        from servers.trace_contract import (
            CONVERSATIONAL_REF_TYPES, S0_CONVERSATIONAL_INCOMING)
        expected = tuple(
            rt for rt, conv in S0_CONVERSATIONAL_INCOMING.items() if conv
        ) + ('assistant_message',)
        assert CONVERSATIONAL_REF_TYPES == expected

    def test_whitelist_is_exactly_what_the_encoder_reads(self):
        # The get_session_turns whitelist is what the encoder and the embed
        # lockstep read: a ref_type entering it is a deliberate change. The
        # prompt reader's harness and subagent rows are in it because each
        # opens a turn that holds my reply.
        from servers.trace_contract import CONVERSATIONAL_REF_TYPES
        assert set(CONVERSATIONAL_REF_TYPES) == {
            'user_message', 'assistant_message', 'env_message', 'subagent_message'}


# ═══════════════════════════════════════════════════════
# Journal note contract (encoder residue — journal redesign Phase 2)
# ═══════════════════════════════════════════════════════

class TestJournalNoteContract:
    """The journal_note event contract: residue notes written as their own
    delta trace events, separate from the run's objective ops-delta. The
    subject lives in ref_id; metadata carries {note, tag}. Registered for
    s1 + s2 delta only — never s0 (notes are an encoder concern, and keeping
    them off s0 is part of the recall guard: s1/s2 traces aren't embedded)."""

    def test_thalamus_filed_is_encoder_delta_residue(self):
        """The filing-side marker rides a journaling encoder's run chain —
        the S1 Scribe's and the S2 units' — is residue (never counted as a
        run), and is NOT registered for s0: a filing is a run's act, not a
        turn's; an s0 chain must fail loudly at the write boundary."""
        from servers.trace_contract import (REF_TYPES, RESIDUE_REF_TYPES,
                                            REF_THALAMUS_FILED,
                                            validate_trace_event)
        for scale in ('s1', 's2'):
            assert REF_THALAMUS_FILED in REF_TYPES[(scale, 'delta')]
            assert validate_trace_event(scale, 'delta', REF_THALAMUS_FILED)[0]
        assert REF_THALAMUS_FILED in RESIDUE_REF_TYPES
        assert not validate_trace_event('s0', 'delta', REF_THALAMUS_FILED)[0]

    def test_thalamus_filed_metadata_is_a_registered_shape(self):
        """The payload is contract-owned and ENFORCED at the write boundary,
        like journal_note — a permissive (unregistered) ref_type is how two
        writers once emitted two shapes undetected."""
        from servers.trace_contract import (
            REF_THALAMUS_FILED, METADATA_REQUIRED_BY_REF_TYPE,
            THALAMUS_FILED_METADATA_SHAPE, build_thalamus_filed_metadata,
            validate_trace_metadata, THALAMUS_FILED_BODY_LIMIT)
        assert METADATA_REQUIRED_BY_REF_TYPE[REF_THALAMUS_FILED] is \
            THALAMUS_FILED_METADATA_SHAPE
        m = build_thalamus_filed_metadata(
            source='encoder:sonnet', body='x' * (THALAMUS_FILED_BODY_LIMIT + 50),
            target_session='abc', needs_answer=1, dedup_key='7e6decd2',
            route='queue', filing='rearm')
        assert set(m) == set(THALAMUS_FILED_METADATA_SHAPE)
        assert m['needs_answer'] is True
        assert len(m['body']) < THALAMUS_FILED_BODY_LIMIT + 50  # capped, loud
        assert validate_trace_metadata('delta', REF_THALAMUS_FILED, m)[0]
        ok, err = validate_trace_metadata('delta', REF_THALAMUS_FILED,
                                          {'source': 'x'})
        assert not ok and 'body' in err
        try:
            build_thalamus_filed_metadata(source='x', body='y', filing='bumped')
        except ValueError as e:
            assert 'filing' in str(e)
        else:
            raise AssertionError('unknown filing value must raise')

    def test_scale_for_chain_derives_from_chain_prefixes(self):
        """A writer handed a run chain is not also handed the scale — the
        prefix already says it; an unclaimed chain raises (producer bug)."""
        from servers.trace_contract import scale_for_chain, CHAIN_PREFIXES
        assert scale_for_chain('s1e-aaaaaaaa-3') == 's1'
        assert scale_for_chain('s1r-aaaaaaaa-3') == 's1'
        assert scale_for_chain('s0-aaaaaaaa-3') == 's0'
        assert scale_for_chain('s2-20260905120000-consolidation') == 's2'
        for key, template in CHAIN_PREFIXES.items():
            assert scale_for_chain(template.split('{', 1)[0] + 'x') == \
                key.split('_', 1)[0]
        for bad in ('', 'chain-1', 's9-abc'):
            try:
                scale_for_chain(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('%r must not resolve to a scale' % bad)

    def test_registered_for_s1_and_s2_delta(self):
        from servers.trace_contract import REF_TYPES
        for scale in ('s1', 's2'):
            assert 'journal_note' in REF_TYPES[(scale, 'delta')], (
                "journal_note missing from (%s, delta): %s" % (
                    scale, REF_TYPES[(scale, 'delta')]))

    def test_not_registered_for_s0(self):
        from servers.trace_contract import validate_trace_event
        ok, _ = validate_trace_event('s0', 'delta', 'journal_note')
        assert not ok

    def test_validate_accepts_journal_note(self):
        from servers.trace_contract import validate_trace_event
        for scale in ('s1', 's2'):
            ok, err = validate_trace_event(scale, 'delta', 'journal_note')
            assert ok, "Validation failed for (%s, delta, journal_note): %s" % (
                scale, err)

    def test_metadata_shape_keys(self):
        from servers.trace_contract import JOURNAL_NOTE_METADATA_SHAPE
        assert set(JOURNAL_NOTE_METADATA_SHAPE.keys()) == {'note', 'tag',
                                                           'undelivered'}

    def test_build_defaults_tag_empty(self):
        from servers.trace_contract import build_journal_note_metadata
        m = build_journal_note_metadata(note='merged a1/b2 but unsure')
        assert m == {'note': 'merged a1/b2 but unsure', 'tag': '', 'undelivered': ''}

    def test_build_strips_tag(self):
        from servers.trace_contract import build_journal_note_metadata
        assert build_journal_note_metadata(note='x', tag='  doubt ')['tag'] == 'doubt'

    def test_metadata_requires_note(self):
        from servers.trace_contract import validate_trace_metadata
        ok, err = validate_trace_metadata('delta', 'journal_note', {'tag': 'doubt'})
        assert not ok and 'note' in err

    def test_metadata_accepts_well_formed(self):
        from servers.trace_contract import (build_journal_note_metadata,
                                            validate_trace_metadata)
        m = build_journal_note_metadata(
            note='the temporal scout misread a number', tag='friction')
        ok, err = validate_trace_metadata('delta', 'journal_note', m)
        assert ok, err

    def test_build_rejects_empty_note(self):
        # Builder and parser must AGREE that an empty note is invalid.
        import pytest
        from servers.trace_contract import build_journal_note_metadata
        for bad in ('', '   ', None):
            with pytest.raises(ValueError):
                build_journal_note_metadata(note=bad)

    def test_build_caps_long_note_and_tag(self):
        # note/tag are capped loud like every other delta text field.
        from servers.trace_contract import build_journal_note_metadata
        m = build_journal_note_metadata(note='x' * 800, tag='y' * 200)
        assert len(m['note']) < 800   # capped
        assert len(m['tag']) < 200    # capped


class TestJournalNoteParser:
    """parse_journal_notes splits an encoder review section into rows;
    render_journal_review_block assembles the shared instruction + per-encoder
    examples. Single source for all journaling encoders (§7.1/§7.3)."""

    def test_three_field_line(self):
        from servers.trace_contract import parse_journal_notes
        notes, bad = parse_journal_notes('friction · temporal-scout · misread a number')
        assert bad == []
        assert notes == [{'tag': 'friction', 'subject': 'temporal-scout',
                          'note': 'misread a number'}]

    def test_two_field_line_tag_optional(self):
        from servers.trace_contract import parse_journal_notes
        notes, bad = parse_journal_notes('nodes a1/b2 · merged but unsure')
        assert bad == []
        assert notes == [{'tag': '', 'subject': 'nodes a1/b2',
                          'note': 'merged but unsure'}]

    def test_delimiter_in_note_preserved(self):
        # maxsplit=2 keeps any '·' inside the prose in the note field.
        from servers.trace_contract import parse_journal_notes
        notes, _ = parse_journal_notes('surprise · recall · old · beat fresh — odd')
        assert notes[0]['note'] == 'old · beat fresh — odd'

    def test_no_delimiter_is_malformed(self):
        from servers.trace_contract import parse_journal_notes
        notes, bad = parse_journal_notes('this line has no delimiter')
        assert notes == [] and len(bad) == 1

    def test_empty_subject_or_note_malformed(self):
        from servers.trace_contract import parse_journal_notes
        notes, bad = parse_journal_notes('friction ·  · ')
        assert notes == [] and len(bad) == 1

    def test_headers_and_blanks_skipped(self):
        from servers.trace_contract import parse_journal_notes
        notes, bad = parse_journal_notes('## Review\n\nfriction · scout · misfired\n')
        assert bad == [] and len(notes) == 1

    def test_render_block_is_the_one_instruction(self):
        """One text for every encoder: the render IS the constant readers
        compare prompts against (no per-encoder variants, no examples fence)."""
        from servers.trace_contract import (render_journal_review_block,
                                            JOURNAL_REVIEW_INSTRUCTION)
        block = render_journal_review_block()
        assert block == JOURNAL_REVIEW_INSTRUCTION
        assert block.startswith('A review — a short note to the next run')
        assert '```' not in block

    def test_hash_subject_not_dropped(self):
        # A delimiter-bearing line whose subject starts with '#' (e.g. an issue
        # id) must parse — not be eaten by the markdown-header skip.
        from servers.trace_contract import parse_journal_notes
        notes, bad = parse_journal_notes('#49019 · still unresolved after refresh')
        assert bad == []
        assert notes == [{'tag': '', 'subject': '#49019',
                          'note': 'still unresolved after refresh'}]

    def test_markdown_headers_skipped_not_malformed(self):
        # Header lines (no delimiter, '#'-prefixed) are structural — skipped
        # silently, never logged as malformed.
        from servers.trace_contract import parse_journal_notes
        notes, bad = parse_journal_notes('## Review\n### Notes')
        assert notes == [] and bad == []

    def test_leading_markdown_bullet_stripped(self):
        # LLMs list-format their review; a leading '-'/'*'/'•' must not become
        # part of the tag (the future miner's grouping key).
        from servers.trace_contract import parse_journal_notes
        for bullet in ('- ', '* ', '• '):
            notes, bad = parse_journal_notes(bullet + 'friction · nodeA · misread')
            assert bad == []
            assert notes == [{'tag': 'friction', 'subject': 'nodeA', 'note': 'misread'}]

    def test_extract_review_block_none_vs_empty(self):
        # None = no section / broken fence (drift); '' = empty fence (clean run);
        # str = content. The writer keys its loud-vs-quiet decision on this.
        from servers.trace_contract import extract_review_block
        assert extract_review_block('no marker here') is None
        assert extract_review_block('## Review\nbare line, no fence') is None
        assert extract_review_block('## Review\n```\nunclosed') is None
        assert extract_review_block('## Review\n```\n```\n') == ''
        assert extract_review_block('## Review\n```\nfriction · a · b\n```') == 'friction · a · b'

    def test_anchor_refs_are_the_node_ids_the_subject_embeds(self):
        """A subject that embeds node ids anchors the line on them, in
        order, once each; free text and a 16-hex hash anchor nothing."""
        from servers.trace_contract import journal_anchor_refs as anchor
        assert anchor('7e6decd2') == ['7e6decd2']
        assert anchor('d4ed89c5_journal_hold_timing') == ['d4ed89c5']
        assert anchor('37c548a7_vs_21a88035_overlap') == ['37c548a7', '21a88035']
        assert anchor('D4ED89C5 again d4ed89c5') == ['d4ed89c5']
        assert anchor('segment 6.a') == []
        assert anchor('[1]_merge_rejected') == []
        assert anchor('0123456789abcdef') == []

    def test_extract_arc_block_none_vs_empty(self):
        # The arc extractor shares the fence machinery: same three-valued
        # contract, keyed on `## Arc`.
        from servers.trace_contract import extract_arc_block
        assert extract_arc_block('no marker here') is None
        assert extract_arc_block('## Arc\nbare line, no fence') is None
        assert extract_arc_block('## Arc\n```\nunclosed') is None
        assert extract_arc_block('## Arc\n```\n```\n') == ''
        assert extract_arc_block('## Arc\n```\narc fix shipped\n```') == 'arc fix shipped'

    def test_arc_and_review_extract_independently(self):
        # A final reply carries BOTH sections (§7.2: Encode → Arc → Review);
        # each extractor pulls only its own fence.
        from servers.trace_contract import extract_arc_block, extract_review_block
        text = ('narrative\n\n## Arc\n```\narc write-path built\n```\n\n'
                '## Review\n```\ndoubt · arc-fence · one-liner may drift\n```\nDONE')
        assert extract_arc_block(text) == 'arc write-path built'
        assert extract_review_block(text) == 'doubt · arc-fence · one-liner may drift'

    def test_fenceless_arc_does_not_capture_review_fence(self):
        # Regression (code-review 2026-07-03): §7.2 orders Arc BEFORE Review.
        # A fenceless `## Arc` must NOT reach forward into the `## Review` fence
        # — else review notes get silently written as the session arc. A heading
        # before the fence = drift → None (→ write_session_arc 'no_arc_extracted').
        from servers.trace_contract import extract_arc_block, extract_review_block
        drift = ('## Arc\nI made progress but forgot to fence it\n\n'
                 '## Review\n```\ndoubt · x · y\n```\nDONE')
        assert extract_arc_block(drift) is None                # NOT 'doubt · x · y'
        assert extract_review_block(drift) == 'doubt · x · y'  # review still fine

    def test_arc_fence_content_with_hash_hash_line_survives(self):
        # The fix checks heading POSITION (before the fence), not blunt
        # truncation — so legit fence content containing a '## ' line is kept.
        from servers.trace_contract import extract_arc_block
        assert extract_arc_block('## Arc\n```\n## shipped the writer\n```\nDONE') \
            == '## shipped the writer'

    def test_render_arc_block_names_marker_and_placement(self):
        # The arc block must teach the `## Arc` marker its writer keys on, and
        # state its own placement (before the review) — the closure deliberately
        # doesn't mention the arc (shared with encoders that never emit one).
        from servers.trace_contract import (render_journal_arc_block,
                                            render_prompt_closure,
                                            JOURNAL_ARC_MARKER)
        block = render_journal_arc_block()
        assert JOURNAL_ARC_MARKER in block
        assert 'ONE line' in block
        assert JOURNAL_ARC_MARKER not in render_prompt_closure()
