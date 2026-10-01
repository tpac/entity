"""Unit tests for VectorCache — in-memory store, no SQLite.

Run: python3 -m pytest tests/test_vector_cache.py -v
"""
import pytest

from servers.vector_cache import VectorCache


def _blob(i: int) -> bytes:
    # Deterministic, distinguishable embedding bytes per test case.
    return (b'\x00' * (i * 4)) + b'\x01\x02\x03\x04'


class TestLoad:
    def test_load_replaces_contents(self):
        c = VectorCache()
        c.load([('n1', '_primary', _blob(1), 'hello', 'm1')])
        assert c.stats()['total_rows'] == 1
        # Second load replaces, doesn't append.
        c.load([('n2', '_primary', _blob(2), 'world', 'm1')])
        assert c.stats()['total_rows'] == 1
        assert c.get('n1', '_primary') is None
        assert c.get('n2', '_primary') is not None

    def test_load_skips_empty_rows(self):
        c = VectorCache()
        count = c.load([
            ('n1', '_primary', _blob(1), 't', 'm'),
            ('', '_primary', _blob(2), 't', 'm'),       # empty node_id
            ('n2', '', _blob(3), 't', 'm'),              # empty vector_type
            ('n3', '_primary', None, 't', 'm'),          # None embedding
        ])
        assert count == 1
        assert c.stats()['total_rows'] == 1


class TestAdd:
    def test_add_upserts(self):
        c = VectorCache()
        c.add('n1', '_primary', _blob(1), 'first', 'm1')
        assert c.get('n1', '_primary')['text'] == 'first'
        # Upsert: same key replaces.
        c.add('n1', '_primary', _blob(2), 'second', 'm1')
        assert c.get('n1', '_primary')['text'] == 'second'
        assert c.stats()['total_rows'] == 1

    def test_add_none_embedding_skipped(self):
        c = VectorCache()
        c.add('n1', '_primary', None, 't', 'm')
        assert c.stats()['total_rows'] == 0

    def test_add_batch(self):
        c = VectorCache()
        n = c.add_batch([
            ('n1', '_primary', _blob(1), 't', 'm'),
            ('n2', '_primary', _blob(2), 't', 'm'),
            ('n3', '_primary', None, 't', 'm'),  # skipped
        ])
        assert n == 2
        assert c.stats()['total_rows'] == 2


class TestDropNode:
    def test_drop_removes_all_vector_types(self):
        c = VectorCache()
        c.load([
            ('n1', '_primary', _blob(1), 't', 'm'),
            ('n1', '_situation', _blob(2), 't', 'm'),
            ('n1', 'title', _blob(3), 't', 'm'),
            ('n2', '_primary', _blob(4), 't', 'm'),
        ])
        n = c.drop_node('n1')
        assert n == 3
        assert c.get('n1', '_primary') is None
        assert c.get('n2', '_primary') is not None
        assert c.stats()['total_rows'] == 1

    def test_drop_nonexistent_no_error(self):
        c = VectorCache()
        assert c.drop_node('nope') == 0


class TestQuery:
    def _make(self):
        c = VectorCache()
        c.load([
            ('n1', '_primary',   _blob(1), 'a', 'm1'),
            ('n2', '_primary',   _blob(2), 'b', 'm1'),
            ('n3', '_primary',   _blob(3), 'c', 'm2'),   # different model
            ('n1', '_situation', _blob(4), 'd', 'm1'),
            ('n2', '_situation', _blob(5), 'e', 'm1'),
            ('n1', 'title',      _blob(6), 'f', 'm1'),
        ])
        return c

    def test_query_all(self):
        c = self._make()
        rows = c.query()
        assert len(rows) == 6
        # Shape matches VectorDAL.get_all_vectors.
        assert set(rows[0].keys()) == {'node_id', 'vector_type', 'embedding'}

    def test_query_by_vector_type(self):
        c = self._make()
        rows = c.query(vector_types=['_primary'])
        assert {r['vector_type'] for r in rows} == {'_primary'}
        assert len(rows) == 3

    def test_query_by_model(self):
        c = self._make()
        rows = c.query(model='m2')
        assert len(rows) == 1
        assert rows[0]['node_id'] == 'n3'

    def test_query_exclude_nodes(self):
        c = self._make()
        rows = c.query(exclude_node_ids={'n1'})
        assert all(r['node_id'] != 'n1' for r in rows)

    def test_query_primary_with_text(self):
        c = self._make()
        rows = c.query_primary_with_text()
        assert len(rows) == 3
        # Returns tuples of (node_id, embedding).
        assert all(isinstance(r, tuple) and len(r) == 2 for r in rows)


class TestStats:
    def test_stats_shape(self):
        c = VectorCache()
        c.load([
            ('n1', '_primary', _blob(1), 't', 'm'),
            ('n2', '_primary', _blob(1), 't', 'm'),
            ('n1', '_situation', _blob(1), 't', 'm'),
        ])
        s = c.stats()
        assert s['total_rows'] == 3
        assert s['total_nodes'] == 2
        assert s['by_vector_type'] == {'_primary': 2, '_situation': 1}
        assert s['embedding_bytes'] > 0
        assert 'version' in s

    def test_version_increments_on_mutation(self):
        c = VectorCache()
        v0 = c.stats()['version']
        c.add('n1', '_primary', _blob(1), 't', 'm')
        assert c.stats()['version'] == v0 + 1
        c.drop_node('n1')
        assert c.stats()['version'] == v0 + 2
        # Drop of nonexistent doesn't bump.
        c.drop_node('nope')
        assert c.stats()['version'] == v0 + 2


class TestThreadSafety:
    def test_concurrent_add_and_query(self):
        import threading
        c = VectorCache()
        errors = []

        def writer(base):
            try:
                for i in range(100):
                    c.add(f'n{base}_{i}', '_primary', _blob(i), 't', 'm')
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    c.query(vector_types=['_primary'])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        assert c.stats()['total_rows'] == 400
