"""Parity + write-through tests for CachedVectorDAL.

Guarantees:
  1. Parity: for every read method, CachedVectorDAL returns byte-identical
     results to VectorDAL on the same SQLite connection.
  2. Write-through: writes land in SQLite first; cache reflects them after.
  3. Deletion: delete_for_node(vector_types=None|[...]) removes DB rows AND the
     mirrored cache rows in one call — the single deletion surface (the old
     cache-only public drop_node was removed 2026-07-17).

Run: python3 -m pytest tests/test_dal_vector_cached.py -v
"""
import os
import sqlite3
import sys
import tempfile

import pytest

# Ensure repo is on path (test_vector_cache already imports directly).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from servers.dal import VectorDAL
from servers.dal_vector_cached import CachedVectorDAL
from servers.schema import ensure_schema

MODEL = 'test-embedder'  # the stamp every fixture vector carries


def _primary(dal, node_id):
    """The node's `_primary` embedding through the DAL's own read, or None."""
    return next((v['embedding'] for v in dal.get_for_node(node_id)
                 if v['vector_type'] == '_primary'), None)


# ═══════════════════════════════════════════════════════════════
# Fixture: a fresh SQLite DB with nodes + node_enrichments populated
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def brain_db():
    """Temp DB with schema + a small realistic dataset."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    conn = sqlite3.connect(path)
    ensure_schema(conn)

    # Seed nodes — 5 active, 1 archived, mixed types.
    now = '2026-04-19T12:00:00Z'
    nodes = [
        ('n1', 'decision',  'Recall uses nomic',         'content 1', 0, 0.8),
        ('n2', 'rule',      'Single-writer daemon',      'content 2', 0, 0.9),
        ('n3', 'insight',   'Cache thrashing pattern',   'content 3', 0, 0.7),
        ('n4', 'concept',   'S2 integration units',      'content 4', 0, 0.6),
        ('n5', 'open',      'Next-session checklist',    'content 5', 0, 0.5),
        ('n6', 'decision',  'Archived choice',           'content 6', 1, 0.8),
    ]
    for (nid, ntype, title, content, archived, conf) in nodes:
        conn.execute(
            'INSERT INTO nodes (id, type, title, content, archived, confidence, '
            'created_at, personal, personal_context, '
            'emotion, access_count) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0, 0)',
            (nid, ntype, title, content, archived, conf,
             now, 'ctx-' + nid))

    # Seed vectors — 5 primary (n1-n5 only; n6 archived has none), 2 situations.
    dal = VectorDAL(conn)
    for i, nid in enumerate(['n1', 'n2', 'n3', 'n4', 'n5']):
        blob = bytes([i + 1]) * 32
        dal.store(nid, '_primary', f'primary text {nid}', blob, MODEL)
    dal.store('n1', '_situation', 'situation for n1', b'\x10' * 32, MODEL)
    dal.store('n2', '_situation', 'situation for n2', b'\x20' * 32, MODEL)
    dal.store('n1', 'title', 'title text n1', b'\x30' * 32, MODEL)

    # v24+: situation text lives in node_metadata_kv, not node_enrichments.text.
    # Seed both so legacy-read tests and kv-read tests both have data.
    conn.execute(
        "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) "
        "VALUES ('n1', 'situation', 'situation for n1')")
    conn.execute(
        "INSERT OR REPLACE INTO node_metadata_kv (node_id, key, value) "
        "VALUES ('n2', 'situation', 'situation for n2')")

    # n6 archived also has a vector — tests exclude_archived filter.
    dal.store('n6', '_primary', 'archived primary', b'\xff' * 32, MODEL)

    conn.commit()
    yield conn
    conn.close()
    os.remove(path)


# ═══════════════════════════════════════════════════════════════
# Parity: every read method returns the same data on both DALs
# ═══════════════════════════════════════════════════════════════

def _normalize(rows):
    """Sort rows by (node_id, vector_type) for stable comparison."""
    return sorted(rows, key=lambda r: (r.get('node_id', ''),
                                        r.get('vector_type', '')))


class TestParity:
    def test_get_all_vectors_default(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        assert _normalize(inner.get_all_vectors()) == \
               _normalize(cached.get_all_vectors())

    def test_get_all_vectors_exclude_archived(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        i = _normalize(inner.get_all_vectors(exclude_archived=True))
        c = _normalize(cached.get_all_vectors(exclude_archived=True))
        assert i == c
        # Sanity: n6 (archived) excluded.
        assert all(r['node_id'] != 'n6' for r in c)

    def test_get_all_vectors_by_type(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        assert _normalize(inner.get_all_vectors(vector_types=['_primary'])) == \
               _normalize(cached.get_all_vectors(vector_types=['_primary']))

    def test_get_all_vectors_by_model(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        model = MODEL
        assert _normalize(inner.get_all_vectors(model=model)) == \
               _normalize(cached.get_all_vectors(model=model))
        # Wrong model → empty on both.
        assert cached.get_all_vectors(model='does-not-exist') == []
        assert inner.get_all_vectors(model='does-not-exist') == []

    def test_get_all_with_context(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        i = sorted(inner.get_all_with_context(), key=lambda r: r['node_id'])
        c = sorted(cached.get_all_with_context(), key=lambda r: r['node_id'])
        assert len(i) == len(c)
        # All contextual fields must match per-node.
        for ir, cr in zip(i, c):
            assert ir == cr

    def test_get_all_with_context_types_filter(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        i = sorted(inner.get_all_with_context(types=['decision']),
                   key=lambda r: r['node_id'])
        c = sorted(cached.get_all_with_context(types=['decision']),
                   key=lambda r: r['node_id'])
        assert i == c

    def test_primary_vector(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        assert _primary(inner, 'n1') == _primary(cached, 'n1')
        assert _primary(cached, 'does-not-exist') is None

    def test_situation_reads_from_metadata_kv(self, brain_db):
        """Contract: situation is first-class metadata, not a vector.

        As of v24, `situation` lives in node_metadata_kv alongside
        question/reasoning/quotes. The VectorDAL/CachedVectorDAL
        get_situation_text wrappers were retired — readers go through
        MetadataDAL.get_field like any other metadata field.
        """
        from servers.dal_metadata import MetadataDAL
        md = MetadataDAL(brain_db)
        # Seeded for n1/n2 in the fixture.
        assert md.get_field('n1', 'situation') == 'situation for n1'
        assert md.get_field('n2', 'situation') == 'situation for n2'
        assert md.get_field('n3', 'situation') is None  # no situation seeded
        assert md.get_field('does-not-exist', 'situation') is None
        # Belt-and-braces: old wrapper methods must stay removed. If
        # someone re-adds them, this test catches it so they update the
        # contract intentionally instead of sneaking in a ghost API.
        assert not hasattr(VectorDAL, 'get_situation_text'), (
            'VectorDAL.get_situation_text was removed in v24; use MetadataDAL')
        assert not hasattr(CachedVectorDAL, 'get_situation_text'), (
            'CachedVectorDAL.get_situation_text was removed in v24; use MetadataDAL')

    def test_get_for_node(self, brain_db):
        inner = VectorDAL(brain_db)
        cached = CachedVectorDAL(brain_db)
        i = sorted(inner.get_for_node('n1'), key=lambda r: r['vector_type'])
        c = sorted(cached.get_for_node('n1'), key=lambda r: r['vector_type'])
        assert i == c


# ═══════════════════════════════════════════════════════════════
# Write-through: SQL is truth, cache reflects after SQL commits
# ═══════════════════════════════════════════════════════════════

class TestWriteThrough:
    def test_store_reaches_db_and_cache(self, brain_db):
        cached = CachedVectorDAL(brain_db)
        cached.store('n2', 'high_meta', 'meta text', b'\x77' * 32, MODEL)
        # Cache sees it.
        assert cached.get_for_node('n2')  # not empty
        high = [r for r in cached.get_for_node('n2') if r['vector_type'] == 'high_meta']
        assert high and high[0]['embedding'] == b'\x77' * 32
        # Inner DAL (direct SQL) also sees it.
        inner = VectorDAL(brain_db)
        high_from_db = [r for r in inner.get_for_node('n2') if r['vector_type'] == 'high_meta']
        assert high_from_db and high_from_db[0]['embedding'] == b'\x77' * 32

    def test_store_batch_cache_matches_db(self, brain_db):
        cached = CachedVectorDAL(brain_db)
        rows = [
            ('n3', 'other_meta', 'other1', b'\xaa' * 32),
            ('n4', 'other_meta', 'other2', b'\xbb' * 32),
            ('n5', 'other_meta', None,      b'dropped'),  # None text ok
        ]
        n = cached.store_batch(rows, MODEL)
        assert n == 3
        # Cache reflects the batched writes exactly as the DB does (per-node
        # parity — count() was removed in the DAL Phase-A cleanup).
        inner = VectorDAL(brain_db)
        for nid in ('n3', 'n4', 'n5'):
            assert (sorted(cached.get_for_node(nid), key=lambda r: r['vector_type'])
                    == sorted(inner.get_for_node(nid), key=lambda r: r['vector_type']))

    def test_none_embedding_skipped_everywhere(self, brain_db):
        cached = CachedVectorDAL(brain_db)
        cached.store('nX', '_primary', 'text', None, MODEL)
        # None embedding is skipped — nX gets no NON-NULL vector in cache or DB
        # (count() measured `embedding IS NOT NULL`; it was removed in the DAL
        # Phase-A cleanup, so assert the same property via get_for_node).
        inner = VectorDAL(brain_db)
        assert [r for r in cached.get_for_node('nX') if r['embedding'] is not None] == []
        assert [r for r in inner.get_for_node('nX') if r['embedding'] is not None] == []

    def test_delete_for_node_drops_from_cache(self, brain_db):
        cached = CachedVectorDAL(brain_db)
        assert _primary(cached, 'n1') is not None
        n = cached.delete_for_node('n1')
        assert n > 0
        assert _primary(cached, 'n1') is None
        # Inner DAL confirms SQL delete.
        inner = VectorDAL(brain_db)
        assert _primary(inner, 'n1') is None

    def test_delete_for_node_full_removes_db_and_cache(self, brain_db):
        """delete_for_node (no types) is the archive/delete path — DB rows
        deleted AND cache rows gone, one call (the consolidated contract;
        the old cache-only public drop_node is deliberately no more)."""
        cached = CachedVectorDAL(brain_db)
        n = cached.delete_for_node('n2')
        assert n > 0
        assert _primary(cached, 'n2') is None
        inner = VectorDAL(brain_db)
        assert _primary(inner, 'n2') is None
        assert not hasattr(cached, 'drop_node'), \
            'cache-only public drop is gone — deletion is ONE surface'

    def test_delete_for_node_typed_mirrors_db_and_cache(self, brain_db):
        """Typed deletion (revise invalidation): ONLY the given types leave
        the DB and the cache; survivors stay in BOTH — the invariant whose
        violation was the 2026-07-17 healer-invisibility bug."""
        cached = CachedVectorDAL(brain_db)
        assert len(cached.get_for_node('n1')) == 3   # _primary/_situation/title
        n = cached.delete_for_node('n1', vector_types=['title'])
        assert n == 1
        inner = VectorDAL(brain_db)
        assert _primary(cached, 'n1') is not None      # survivor in cache
        assert _primary(inner, 'n1') is not None       # survivor in DB
        remaining = {v['vector_type'] for v in cached.get_for_node('n1')}
        assert remaining == {'_primary', '_situation'}
        # empty list = nothing to do, loudly zero
        assert cached.delete_for_node('n1', vector_types=[]) == 0


# ═══════════════════════════════════════════════════════════════
# Reload — for scripts that bypass the cache
# ═══════════════════════════════════════════════════════════════

class TestReload:
    def test_reload_picks_up_out_of_band_write(self, brain_db):
        cached = CachedVectorDAL(brain_db)
        # Simulate a direct SQL write — bypassing the cache.
        inner = VectorDAL(brain_db)
        inner.store('n4', 'question', 'direct-sql', b'\x99' * 32, MODEL)
        brain_db.commit()
        # Before reload, cache doesn't see it.
        assert not any(r['vector_type'] == 'question'
                       for r in cached.get_for_node('n4'))
        # After reload, cache is in sync.
        cached.reload()
        assert any(r['vector_type'] == 'question'
                   for r in cached.get_for_node('n4'))


# ═══════════════════════════════════════════════════════════════
# Cache stats — diagnostic surface for `brain diagnose`
# ═══════════════════════════════════════════════════════════════

class TestCacheStats:
    def test_cache_stats_shape(self, brain_db):
        cached = CachedVectorDAL(brain_db)
        s = cached.cache_stats()
        assert 'total_rows' in s
        assert 'total_nodes' in s
        assert 'by_vector_type' in s
        assert 'embedding_bytes' in s
        assert s['total_rows'] > 0


# ═══════════════════════════════════════════════════════════════
# Archive integration — confirms brain.archive_node() invalidates
# the cache so recall's in-memory matrix doesn't retain dead rows.
# ═══════════════════════════════════════════════════════════════

class TestArchiveInvalidatesCache:
    """Regression guard for the delete-must-mirror-the-cache wiring.

    The DB delete and the cache drop are ONE call (delete_for_node), so a node
    whose vectors leave the DB always leaves the cache too — the seam can no
    longer delete from the DB without signaling the cache (the drift that made
    archived/deleted nodes linger in cache-served recall).
    """

    def test_archive_seam_removes_vectors_from_db_and_cache(self, brain_db):
        """The archive code path's DAL seam is delete_for_node (consolidated
        2026-07-17 — archive routes through _deindex_node → delete_for_node):
        cache stats shrink by exactly the node's rows AND the DB rows go."""
        cached = CachedVectorDAL(brain_db)

        # n1 has 3 vectors seeded in the fixture: _primary, _situation, title.
        before = cached.cache_stats()['total_rows']
        n1_vecs_before = len(cached.get_for_node('n1'))
        assert n1_vecs_before == 3  # fixture sanity

        cached.delete_for_node('n1')

        after = cached.cache_stats()['total_rows']
        assert after == before - n1_vecs_before
        assert cached.get_for_node('n1') == []
        assert VectorDAL(brain_db).get_for_node('n1') == []

    def test_archive_node_integration(self, tmp_path):
        """Full Brain.archive_node path invalidates the cache.

        Uses a fresh temp brain (no production data) for speed and isolation.
        """
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from servers.brain import Brain

        db_path = str(tmp_path / 'brain.db')
        brain = Brain(db_path=db_path)
        try:
            # Create a node with a situation so at least one vector is stored.
            # Use remember() directly — it routes through the full write path.
            result = brain.remember(
                type='rule',
                title='Archive should drop cache',
                content='When a node is archived, its vectors must be evicted from CachedVectorDAL.',
                situation='Verifying archive_node cache invalidation',
                keywords='archive cache invalidation regression test')
            node_id = result.get('id') or result.get('node_id')
            assert node_id, f'remember() returned no id: {result}'

            # After remember + any embedding backfill, the cache has at least
            # the situation text row (embedding may be None, but VectorCache
            # only stores rows with non-None embeddings — so the cache may
            # have 0 rows until backfill runs). What we CAN assert: after
            # archive, the cache has no rows for this node regardless.
            brain.archive_node(node_id=node_id, archived_by='test',
                               reason='regression test')

            # Cache holds nothing for the archived node.
            assert brain._vec_dal.get_for_node(node_id) == []
        finally:
            brain.close()
