"""Retired node fields must not be taught, advertised, or queried.

WHY THIS EXISTS: schema v28 dropped `nodes.keywords`, and every write
surface was cleaned — but nothing asserted ABSENCE. The encoder prompts
kept teaching the field, `additionalProperties` is deliberately open on
every write branch, and unknown keys route to node_metadata_kv. Result:
229 nodes silently accumulated a dead `keywords` KV row between July 1
and the audit, and the whole existing suite stayed green throughout.

Tests assert presence. This one asserts absence.

Note on scope: a general "unknown field" check is impossible by design —
open KV accepts any key (`emotional_context`, `event_time`, ...) and that
freedom is the point. So this pins the specific fields we RETIRED, which
is the case that actually regressed.
"""
import re
import unittest

from servers.contract import get_writable_fields, STRUCTURAL_FIELDS, PROMOTED_FIELDS
from servers.brain_recall import _NODE_COLUMNS

# Fields removed from the node schema. A field lands here when its column
# or KV slot is retired, and it must then be absent from every surface
# below. Keep the reason inline — it is the whole documentation.
RETIRED_NODE_FIELDS = {
    'keywords': 'nodes.keywords dropped in schema v28 (auto-extractor noise)',
    'content_edits': 'revise alias of `content: [swaps]`, retired once every '
                     'field took swaps',
    'source_turn_id': 'nodes.source_turn_id dropped in schema v36 (pointed at '
                      'the deleted message_stream; node_source_refs replaced it)',
    'encoding_version': 'nodes.encoding_version dropped in schema v36 (a '
                        'constant nothing read)',
    'emotion_source': 'nodes.emotion_source dropped in schema v36 (no reader)',
    'recency_score': 'nodes.recency_score dropped in schema v36 (session '
                     'activity replaced it)',
    'due_date': 'nodes.due_date dropped in schema v36 (no writer, no reader)',
    'resolved_at': 'nodes.resolved_at dropped in schema v36 (never set)',
}


def _seed_prompts():
    """The four encoder prompts' code defaults, as (name, text). Production
    runs these unless an interaction override is active (`./dev
    check-overrides` lists them), so scanning the defaults scans what the
    encoders see."""
    from servers.scales.s1.encoding_prompt import SYSTEM_PROMPT as s1e
    from servers.scales.s2.community_enrichment_prompt import SYSTEM_PROMPT as comm
    from servers.scales.s2.consolidation_enrichment_prompt import SYSTEM_PROMPT as cons
    from servers.scales.s2.healer_prompt import SYSTEM_PROMPT as heal
    return [
        ('s1e', s1e),
        ('s2_community_enrichment', comm),
        ('s2_consolidation_enrichment', cons),
        ('s2_healer', heal),
    ]


def _op_surfaces():
    """The teaching surfaces beyond the prompts: the encoder's payload gist,
    the contract field summary injected after the prompt, and every
    brain_batch op description. A retired op FIELD is retired here the same
    way a node field is — add it to RETIRED_NODE_FIELDS and every surface must
    fall silent."""
    from servers.contract import BATCH_OP_SPECS, generate_field_summary
    from servers.interaction_defaults import INTERACTION_DEFAULTS
    return ([('s1e gist', INTERACTION_DEFAULTS['s1e_gist'][0]),
             ('field summary', generate_field_summary())]
            + [('brain_batch %s description' % op, spec['description'])
               for op, spec in BATCH_OP_SPECS.items()])


class TestRetiredFieldsNotTaught(unittest.TestCase):
    """The encoder must not be TOLD to write a field the schema dropped."""

    def test_retired_field_absent_from_every_encoder_prompt(self):
        # Retired community keys too: schema v38 deleted their rows, so a
        # prompt that taught one would bring back a second home for a fact.
        # (Open KV keys, so the writable/registry/column checks below don't
        # apply to them.)
        from servers.scales.s2.community_contract import COMMUNITY_RETIRED_KEYS
        retired = list(RETIRED_NODE_FIELDS.items()) + [
            (key, 'retired community key (COMMUNITY_RETIRED_KEYS)')
            for key in sorted(COMMUNITY_RETIRED_KEYS)]
        for field, reason in retired:
            for name, text in _seed_prompts() + _op_surfaces():
                hits = [ln.strip() for ln in text.split('\n') if field in ln]
                self.assertEqual(
                    hits, [],
                    '%s prompt still references retired field %r (%s).\n'
                    'Offending lines:\n  %s'
                    % (name, field, reason, '\n  '.join(hits[:5])))


class TestRetiredFieldsNotAdvertised(unittest.TestCase):
    """The write schema is generated from the contract — a retired field
    left in the contract silently reappears in every MCP write tool."""

    def test_retired_field_not_writable(self):
        writable = get_writable_fields()
        for field, reason in RETIRED_NODE_FIELDS.items():
            self.assertNotIn(field, writable, '%s (%s)' % (field, reason))

    def test_retired_field_not_in_field_registries(self):
        for field, reason in RETIRED_NODE_FIELDS.items():
            self.assertNotIn(field, STRUCTURAL_FIELDS, '%s (%s)' % (field, reason))
            self.assertNotIn(field, PROMOTED_FIELDS, '%s (%s)' % (field, reason))


class TestRetiredFieldsNotQueryable(unittest.TestCase):
    """_NODE_COLUMNS drives dict-filter SQL. A dropped column left in it
    builds a query against a column that no longer exists."""

    def test_retired_field_not_in_node_columns(self):
        for field, reason in RETIRED_NODE_FIELDS.items():
            self.assertNotIn(
                field, _NODE_COLUMNS,
                '%s is in _NODE_COLUMNS but the column is gone (%s) — a dict '
                'filter on it would build SQL against a dropped column'
                % (field, reason))


class TestRetiredFieldsNotInMcpSchemas(unittest.TestCase):
    """The generated MCP tool schemas are a FOURTH surface that talks to a
    model without being code, prompts, or seed files.

    Node 75c661d5 recorded three audit surfaces for retiring a param (code,
    interactions-DB prompts, seed .py). A fourth bit us: `find_node_by_title`
    advertised "(content snippet, keywords)" as its return shape long after
    v28 stopped returning it — not an example, a PROMISE, in a description
    no other check reads.
    """

    # (tool, field) pairs where the word legitimately appears.
    #   recall  — query PHRASING advice ("semantic, not keyword")
    ALLOWED = {
        ('recall', 'keywords'),
    }

    @staticmethod
    def _declared_property_names(schema):
        names = set()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == 'properties' and isinstance(v, dict):
                        names.update(v.keys())
                    walk(v)
            elif isinstance(o, list):
                for item in o:
                    walk(item)

        walk(schema)
        return names

    def test_no_write_op_declares_a_retired_field(self):
        """A retired field declared as a parameter is unambiguous — the
        schema is inviting a model to send a field the write path drops."""
        from servers.brain_mcp import TOOLS
        for tool in TOOLS:
            declared = self._declared_property_names(tool.get('inputSchema', {}))
            for field in RETIRED_NODE_FIELDS:
                if (tool['name'], field) in self.ALLOWED:
                    continue
                self.assertNotIn(
                    field, declared,
                    'MCP tool %r declares retired field %r as a parameter'
                    % (tool['name'], field))

    def test_no_tool_description_mentions_a_retired_field(self):
        """Descriptions are prose a model reads as fact. A retired field
        named here is either a stale return-shape promise or an example
        teaching a dead write."""
        import json
        from servers.brain_mcp import TOOLS
        for tool in TOOLS:
            blob = json.dumps(tool)
            for field in RETIRED_NODE_FIELDS:
                if (tool['name'], field) in self.ALLOWED:
                    continue
                self.assertNotIn(
                    field, blob,
                    'MCP tool %r mentions retired field %r in its schema or '
                    'description. If the mention is legitimate (a different '
                    'concept sharing the name), add (%r, %r) to ALLOWED with '
                    'a reason.' % (tool['name'], field, tool['name'], field))


class TestV5EnrichmentLoopIsGone(unittest.TestCase):
    """The V5 manual loop (remember → enrichment_prompt → enrich → Q/A/B/K
    vectors) is retired. Its surfaces must stay absent: `question` is now a
    plain `remember` field derived from node_metadata_kv, and `anchor` /
    `bridge` / `keywords` have no producer."""

    def test_enrich_command_is_not_registered(self):
        from servers.daemon_dispatch import COMMAND_TABLE
        self.assertNotIn('enrich', COMMAND_TABLE)

    def test_store_enrichments_is_gone(self):
        from servers.brain_remember import BrainRememberMixin
        self.assertFalse(hasattr(BrainRememberMixin, 'store_enrichments'))
        self.assertFalse(hasattr(BrainRememberMixin, '_build_enrichment_prompt'))

    # The response-shape half of this contract is asserted against a real
    # dispatched remember() in test_mcp_roundtrip.TestMCPRoundTrip.test_remember
    # — source-text matching would pass while a helper re-added the key.


if __name__ == '__main__':
    unittest.main()
