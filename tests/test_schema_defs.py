"""Tool schemas reference shared shapes by `$ref` and carry them once under
`$defs` (contract.SCHEMA_DEFS / attach_defs). Every pointer must resolve inside
its own tool — a dangling `$ref` is a field the model cannot fill and nothing
at runtime would notice."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers import brain_mcp  # noqa: E402
from servers.contract import (SCHEMA_DEFS, SWAP_SCHEMA, attach_defs,  # noqa: E402
                              referenced_defs)


def _refs(node, out):
    if isinstance(node, dict):
        if isinstance(node.get('$ref'), str):
            out.append(node['$ref'])
        for v in node.values():
            _refs(v, out)
    elif isinstance(node, list):
        for v in node:
            _refs(v, out)
    return out


def test_every_ref_in_every_tool_resolves_within_that_tool():
    for tool in brain_mcp.TOOLS:
        schema = tool['inputSchema']
        refs = _refs(schema, [])
        defs = schema.get('$defs', {})
        for r in refs:
            assert r.startswith('#/$defs/'), (tool['name'], r)
            name = r[len('#/$defs/'):]
            assert name in defs, '%s points at %s but carries no such definition' % (tool['name'], r)
            assert defs[name] == SCHEMA_DEFS[name], (tool['name'], name)
        # and nothing is carried that is not referenced
        assert set(defs) == set(n[len('#/$defs/'):] for n in refs) | set(
            n for n in defs if any(r.endswith('/' + n) for r in _refs(list(defs.values()), []))), tool['name']


def test_tools_without_references_carry_no_defs():
    for tool in brain_mcp.TOOLS:
        if not _refs(tool['inputSchema'], []):
            assert '$defs' not in tool['inputSchema'], tool['name']


def test_encoder_write_tools_share_one_swap_definition():
    tools = {t['name']: t for t in brain_mcp.TOOLS}
    for name in ('revise', 'revise_batch', 'brain_batch'):
        assert tools[name]['inputSchema']['$defs']['swap'] == SWAP_SCHEMA, name


def test_transitive_reference_pulls_the_nested_definition():
    # revise_connect_to_item's relation/why are swappable → swap comes along
    names = referenced_defs({'items': {'$ref': '#/$defs/revise_connect_to_item'}})
    assert names == {'revise_connect_to_item', 'swap'}


def test_dangling_reference_is_a_build_error():
    with pytest.raises(ValueError):
        attach_defs({'properties': {'x': {'$ref': '#/$defs/not_a_shape'}}})
    with pytest.raises(ValueError):
        attach_defs({'properties': {'x': {'$ref': 'other.json#/foo'}}})


def test_attach_defs_is_identity_without_references():
    schema = {'type': 'object', 'properties': {'q': {'type': 'string'}}}
    assert attach_defs(schema) is schema
