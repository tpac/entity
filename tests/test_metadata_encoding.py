"""Tests for MetadataDAL value encoding (Step 5a).

set / set_many / bulk_set_key now JSON-encode list and dict values so they
round-trip cleanly through json.loads. String values pass through unchanged.
"""

import json
import os
import sqlite3
import tempfile
import unittest

from servers.dal_metadata import MetadataDAL, _encode_value
from servers.schema import ensure_schema


class TestEncodeValue(unittest.TestCase):
    """The _encode_value helper — pure function, no DB."""

    def test_none_returns_none(self):
        self.assertIsNone(_encode_value(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_encode_value(''))
        self.assertIsNone(_encode_value('   '))

    def test_string_returns_string(self):
        self.assertEqual(_encode_value('hello'), 'hello')

    def test_list_returns_json(self):
        self.assertEqual(_encode_value(['a', 'b']), '["a", "b"]')
        self.assertEqual(_encode_value([]), '[]')

    def test_dict_returns_json(self):
        self.assertEqual(_encode_value({'k': 'v'}), '{"k": "v"}')
        self.assertEqual(_encode_value({}), '{}')

    def test_tuple_normalizes_to_list_json(self):
        self.assertEqual(_encode_value(('a', 'b')), '["a", "b"]')

    def test_int_coerces_to_str(self):
        self.assertEqual(_encode_value(42), '42')

    def test_float_coerces_to_str(self):
        self.assertEqual(_encode_value(3.14), '3.14')

    def test_bool_coerces_to_str(self):
        self.assertEqual(_encode_value(True), 'True')
        self.assertEqual(_encode_value(False), 'False')


class TestMetadataDALEncoding(unittest.TestCase):
    """End-to-end: write list/dict via DAL, read back, verify encoding."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.conn = sqlite3.connect(os.path.join(self.tmp, 'test.db'))
        ensure_schema(self.conn)
        self.dal = MetadataDAL(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_set_list_stores_json(self):
        self.dal.set('node1', 'node_types', ['principle', 'rule'])
        stored = self.dal.get_field('node1', 'node_types')
        self.assertEqual(stored, '["principle", "rule"]')
        # And json.loads round-trips
        self.assertEqual(json.loads(stored), ['principle', 'rule'])

    def test_set_dict_stores_json(self):
        self.dal.set('node1', 'config', {'k': 'v', 'n': 1})
        stored = self.dal.get_field('node1', 'config')
        decoded = json.loads(stored)
        self.assertEqual(decoded, {'k': 'v', 'n': 1})

    def test_set_string_stores_as_is(self):
        self.dal.set('node1', 'reasoning', 'because reasons')
        self.assertEqual(self.dal.get_field('node1', 'reasoning'), 'because reasons')

    def test_set_none_skips(self):
        self.dal.set('node1', 'optional', None)
        self.assertIsNone(self.dal.get_field('node1', 'optional'))

    def test_set_empty_string_skips(self):
        self.dal.set('node1', 'optional', '   ')
        self.assertIsNone(self.dal.get_field('node1', 'optional'))

    def test_set_many_mixed_types(self):
        self.dal.set_many('node1', {
            'reasoning': 'because reasons',
            'node_types': ['p', 'r'],
            'metadata': {'display_label': 'foo/bar'},
            'count': 7,
            'empty': '',
            'none_val': None,
        })
        # Strings + numbers + JSON for list/dict
        self.assertEqual(self.dal.get_field('node1', 'reasoning'), 'because reasons')
        self.assertEqual(json.loads(self.dal.get_field('node1', 'node_types')), ['p', 'r'])
        self.assertEqual(json.loads(self.dal.get_field('node1', 'metadata')),
                         {'display_label': 'foo/bar'})
        self.assertEqual(self.dal.get_field('node1', 'count'), '7')
        # Empty + None skipped
        self.assertIsNone(self.dal.get_field('node1', 'empty'))
        self.assertIsNone(self.dal.get_field('node1', 'none_val'))

    def test_set_many_returns_correct_count(self):
        # 4 stored, 2 skipped (empty + None)
        count = self.dal.set_many('node1', {
            'a': 'x', 'b': ['y'], 'c': '', 'd': None, 'e': 'z', 'f': {'k': 1}
        })
        self.assertEqual(count, 4)

    def test_bulk_set_key_with_lists(self):
        self.dal.bulk_set_key('keywords', {
            'node1': ['kw1', 'kw2'],
            'node2': 'plain string',
            'node3': None,  # skipped
        })
        self.assertEqual(json.loads(self.dal.get_field('node1', 'keywords')),
                         ['kw1', 'kw2'])
        self.assertEqual(self.dal.get_field('node2', 'keywords'), 'plain string')
        self.assertIsNone(self.dal.get_field('node3', 'keywords'))


if __name__ == '__main__':
    unittest.main()
