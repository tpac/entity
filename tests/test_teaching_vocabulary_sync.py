"""The prompt, the gist, and the tool layer teach ONE vocabulary — the contract's.

Drift guardrail for changes to the batch-op contract (docs/REVISE-SHAPE-SPEC.md
§5). The tool layer outranks the prompt by position (E10: the field summary and
tool descriptions are injected last), so a name that lives in the contract but
is silent on a teaching surface — or retired from the contract but still
taught — makes the prompt lose without anyone noticing. Each assertion here
reads the contract as the source and checks the surfaces against it.

Retirement is tests/test_retired_fields.RETIRED_NODE_FIELDS' job: a name listed
there is exempt from the taught set here, and that scan then enforces its
absence on every surface this file checks presence on.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.contract import (BATCH_OP_SPECS, CONNECT_TO_ITEM_SCHEMA,  # noqa: E402
                              FOLD_RULE, REF_SWAP, REVISE_RULE,
                              SWAP_SCHEMA, generate_field_summary,
                              get_swap_fields, get_writable_fields)
from servers.interaction_defaults import INTERACTION_DEFAULTS  # noqa: E402
from servers.scales.s1.encode_contract import ENCODING_TOOLS  # noqa: E402
from servers.scales.s1.encoding_prompt import SYSTEM_PROMPT  # noqa: E402
from servers import brain_mcp  # noqa: E402
from tests.test_retired_fields import RETIRED_NODE_FIELDS  # noqa: E402

TOOLS = {t['name']: t for t in brain_mcp.TOOLS}
GIST = INTERACTION_DEFAULTS['s1e_gist'][0]

# Identifiers the gist may backtick beyond the contract's own names: the
# relation verbs it teaches on purpose (relations are open vocabulary) and the
# one type it names (types are open too; this one is load-bearing). Extend
# when the gist does — one allowlist, not three.
GIST_OPEN_VOCABULARY = {
    'resolves', 'partially_resolves', 'open',
    # the gist's own procedure: its four list labels and the status words
    # a target line carries (V3.6) — not fields, not ops, teaching on purpose
    'changes', 'targets', 'fetch', 'stale', 'clean', 'unread', 'resolved',
    # a node field the MCP schema adds beside the contract's writable set
    # (brain_mcp._SOURCE_REFS_SCHEMA) and the field summary already names;
    # it has no contract field spec of its own — that gap is a follow-on
    'source_refs',
}

# The prose surfaces that teach the revise vocabulary, evaluated once — a
# field is taught here when its name appears in the text.
SURFACES = [
    ('s1e prompt', SYSTEM_PROMPT),
    ('gist', GIST),
    ('field summary', generate_field_summary()),
]

# The tool surfaces: a field is taught here when its OWN property carries a
# description. A schema key alone is not teaching (the model gets no
# guidance), and a mention inside another property's prose is not either
# (the rule's sentence lists five field names — every one would pass).
TOOL_PROPS = [
    ('MCP revise', TOOLS['revise']['inputSchema']['properties']),
    ('MCP revise_batch',
     TOOLS['revise_batch']['inputSchema']['properties']['revisions']['items']['properties']),
    ('brain_batch revise spec', BATCH_OP_SPECS['revise']['properties']),
]


def _names(text, identifier):
    """Word-boundary match — `content` must not pass on `content_edits`."""
    return re.search(r'\b%s\b' % re.escape(identifier), text) is not None


def _fenced(text):
    out, cur, inside = [], [], False
    for line in text.split('\n'):
        if line.startswith('```'):
            if inside:
                out.append('\n'.join(cur))
                cur = []
            inside = not inside
            continue
        if inside:
            cur.append(line)
    return out


def _connect_to_arrays(block):
    """Text of every `connect_to: [ … ]` array in a fenced example."""
    arrays = []
    for m in re.finditer(r'connect_to:\s*\[', block):
        depth, i = 0, m.end() - 1
        while i < len(block):
            if block[i] == '[':
                depth += 1
            elif block[i] == ']':
                depth -= 1
                if depth == 0:
                    arrays.append(block[m.end():i])
                    break
            i += 1
    return arrays


def _connect_to_item_keys(block):
    """Keys of every connect_to item in a fenced example — both example
    grammars: the `connect_to: [ {…}, … ]` array (brain_batch /
    remember_batch) and the YAML list (`connect_to:` then `- key: …` lines,
    deeper-indented until a blank line or a dedent — the temporal example)."""
    keys = set()
    for arr in _connect_to_arrays(block):
        keys.update(re.findall(r'[{,]\s*([a-z_]+)\s*:', arr))
    lines = block.split('\n')
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)connect_to:\s*$', line)
        if not m:
            continue
        base = len(m.group(1))
        for nxt in lines[i + 1:]:
            if not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= base:
                break
            km = re.match(r'^\s*(?:-\s*)?([a-z_]+):\s', nxt)
            if km:
                keys.add(km.group(1))
    return keys


def _taught_revise_fields():
    """Every property the revise spec offers beyond its keys, minus retired
    names (the retirement scan enforces their absence)."""
    return [f for f in BATCH_OP_SPECS['revise']['properties']
            if f not in ('node_id', 'reason')
            and f not in RETIRED_NODE_FIELDS]


def test_every_example_op_is_a_batch_op():
    ops = set()
    for block in _fenced(SYSTEM_PROMPT):
        ops.update(re.findall(r'\bop:\s*"([a-z_]+)"', block))
    assert ops, 'no {op: "…"} examples found in the prompt — the regex or the prompt moved'
    unknown = ops - set(BATCH_OP_SPECS)
    assert not unknown, 'prompt examples use ops the contract has no spec for: %s' % sorted(unknown)


def test_every_connect_to_key_in_examples_is_in_the_item_schema():
    """Item keys, plus the swap keys a revise entry's `relation`/`why` may
    carry (REVISE_CONNECT_TO_ITEM_SCHEMA makes both swappable) — the sweep
    example's a45c88f1 entry swaps its why."""
    allowed = set(CONNECT_TO_ITEM_SCHEMA['properties'])
    allowed |= set(CONNECT_TO_ITEM_SCHEMA['properties']['relations']['items']['properties'])
    allowed |= set(SWAP_SCHEMA['properties'])
    seen = set()
    for block in _fenced(SYSTEM_PROMPT):
        seen |= _connect_to_item_keys(block)
    assert seen, 'no connect_to items found in the prompt examples'
    unknown = seen - allowed
    assert not unknown, 'connect_to items in the prompt use keys the schema lacks: %s' % sorted(unknown)


def test_every_revise_spec_field_is_taught_on_every_surface():
    """A field the contract offers on revise but a surface never names is the
    E11 defect (present, not taught) — and the position asymmetry means the
    silent surface wins."""
    fields = _taught_revise_fields()
    assert fields, 'revise spec has no teachable fields — did the contract move?'
    silent = [(name, f) for f in fields for name, text in SURFACES
              if not _names(text, f)]
    silent += [(name, f) for f in fields for name, props in TOOL_PROPS
               if not (props.get(f) or {}).get('description')]
    assert not silent, 'revise fields the contract offers but a surface never teaches: %s' % silent


def test_every_swap_field_is_value_or_swap_on_the_mcp_revise_surfaces():
    """REVISE_RULE says every text field takes a swap; the schema must say it
    per field, not per exemplar — a field the tool advertises as a plain
    string is a field the encoder will never swap. Non-text fields stay bare."""
    props_by_surface = (
        ('revise', TOOLS['revise']['inputSchema']['properties']),
        ('revise_batch', TOOLS['revise_batch']['inputSchema']['properties']
                         ['revisions']['items']['properties']),
    )
    swap_fields = get_swap_fields()
    for surface, props in props_by_surface:
        # the swap is referenced, so the tool must also carry its definition
        assert TOOLS[surface]['inputSchema']['$defs']['swap'] == SWAP_SCHEMA, surface
        for f in swap_fields:
            assert REF_SWAP in props.get(f, {}).get('anyOf', []), \
                '%s.%s is not value-or-swap' % (surface, f)
        for f, spec in get_writable_fields().items():
            if f in swap_fields:
                continue
            assert 'anyOf' not in props.get(f, {}), \
                '%s.%s is %s but advertised as swappable' % (surface, f, spec.get('type'))


def test_the_one_rule_is_quoted_verbatim_by_every_tool_surface():
    """REVISE_RULE is one string every tool surface quotes (spec §4 row 1) —
    a paraphrase on one surface is the drift this file exists to catch."""
    quoted = (
        ('field summary', generate_field_summary()),
        ('MCP revise', TOOLS['revise']['description']),
        ('MCP revise_batch', TOOLS['revise_batch']['description']),
        ('brain_batch revise spec', BATCH_OP_SPECS['revise']['description']),
    )
    missing = [name for name, text in quoted if REVISE_RULE not in text]
    assert not missing, 'surfaces that do not quote REVISE_RULE verbatim: %s' % missing


def test_gist_uses_only_contract_vocabulary():
    known = set(BATCH_OP_SPECS) | ENCODING_TOOLS | set(get_writable_fields())
    known |= {prop for spec in BATCH_OP_SPECS.values() for prop in spec['properties']}
    known |= set(CONNECT_TO_ITEM_SCHEMA['properties'])
    known |= set(SWAP_SCHEMA['properties']) | GIST_OPEN_VOCABULARY
    tokens = set(re.findall(r'`([a-z_]+)`', GIST))
    assert tokens, 'gist has no backticked identifiers — did the gist move?'
    unknown = tokens - known
    assert not unknown, 'gist names identifiers the contract does not have: %s' % sorted(unknown)


def test_encoder_tool_names_in_prompt_exist():
    names = set(re.findall(r'`([a-z_]+_batch|get_nodes)`', SYSTEM_PROMPT))
    assert names, 'prompt names no batch tools'
    unknown = names - set(TOOLS)
    assert not unknown, 'prompt names tools that do not exist: %s' % sorted(unknown)
    not_encoder = names - ENCODING_TOOLS
    assert not not_encoder, 'prompt names tools the encoder does not hold: %s' % sorted(not_encoder)


def _tool_prose(o, out):
    """Every description string a tool carries, plus each declared property
    name — the text a model reads, without the JSON punctuation around it."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'description' and isinstance(v, str):
                out.append(v)
            elif k == 'properties' and isinstance(v, dict):
                out.extend('`%s`' % name for name in v)
            _tool_prose(v, out)
    elif isinstance(o, list):
        for item in o:
            _tool_prose(item, out)
    return out


def test_tool_layer_never_teaches_the_legacy_edge_alias():
    """An edge's key is `description`; `why` is only its accepted alias
    (contract.connect_to_description). No tool text names the alias as a key:
    backticked, dotted, quoted, or inside a `{...}` key list. Bare English
    'why' is not matched."""
    legacy = re.compile(r"`why`|\.why\b|'why'|\{[^{}]*\bwhy\b[^{}]*\}")
    hits = {name: [m for text in _tool_prose(tool, []) for m in legacy.findall(text)]
            for name, tool in TOOLS.items()}
    hits = {name: found for name, found in hits.items() if found}
    assert not hits, 'tools that teach the legacy `why`: %s' % hits


def test_absorb_is_described_by_its_op_alone():
    """brain_batch's own description leaves absorb to the absorb op, which
    states contract.FOLD_RULE verbatim."""
    assert 'absorb' not in TOOLS['brain_batch']['description']
    assert FOLD_RULE in BATCH_OP_SPECS['absorb']['description']


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('  PASS  %s' % name)
            except AssertionError as e:
                fails += 1
                print('  FAIL  %s — %s' % (name, e))
    print('\n%s' % ('ALL PASS' if not fails else '%d FAILED' % fails))
    sys.exit(1 if fails else 0)


def test_encoder_summary_hides_fields_kept_on_the_mcp_surface():
    """encoder_summary=False is an audience split, not a retirement: the field
    stays writable and declared on the MCP remember schema (Anchor's hand) and
    is absent from the encoder's field summary — an advertised field with no
    worked carrier was written as a constant (confidence 0.7 on every node)."""
    from servers.contract import ALL_FIELDS, encoder_summary_fields
    hidden = {k for k, v in ALL_FIELDS.items() if v.get('encoder_summary') is False}
    assert hidden, 'no field is marked encoder_summary=False — did the flag move?'
    summary_names = {ln.split()[0] for ln in generate_field_summary().split('\n') if '  (' in ln}
    writable = get_writable_fields()
    remember_props = TOOLS['remember']['inputSchema']['properties']
    for f in hidden:
        assert f in writable, '%s hidden from the encoder must stay writable by hand' % f
        assert f in remember_props, '%s hidden from the encoder must stay on the MCP remember schema' % f
        assert f not in summary_names, '%s is marked encoder_summary=False but the field summary lists it' % f
    assert set(encoder_summary_fields()) == set(writable) - hidden
