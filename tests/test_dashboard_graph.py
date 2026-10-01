"""The Live tab's graph payload: one relation per node pair — the one its line
shows — the live taxonomy's primary aspect per node type and relation, which
graph.js maps to a colour slot per aspect name, and the brain's semantic
depth axis, which graph.js spends on the Z axis.
"""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from dashboard.queries import graph
from dashboard.queries.aspects import primary_aspects
from servers.aspect_store import SEED_ASPECTS_JSON_PATH

ROOT = Path(__file__).resolve().parents[1]

_TAXONOMY = {
    'lesson_insight': {'node_types': ['lesson']},
    'dependency_flow': {'edge_relations': ['depends_on']},
    'explanation_causation': {'edge_relations': ['grounds', 'explains']},
    'noise': {'edge_relations': ['co_anchored']},
}


def _schema(tmp_path, vectors=False):
    """The tables query_graph3d reads; `vectors` adds node_enrichments."""
    conn = sqlite3.connect(tmp_path / 'brain.db')
    conn.execute("CREATE TABLE nodes (id TEXT, type TEXT, title TEXT, "
                 "access_count INT, created_at TEXT, last_accessed TEXT, "
                 "archived INT DEFAULT 0)")
    conn.execute("CREATE TABLE edges (edge_id TEXT)")
    conn.execute("CREATE TABLE edge_relations (edge_id TEXT, claim_source_id TEXT, "
                 "claim_target_id TEXT, relation TEXT, weight REAL, created_at TEXT, "
                 "archived INT DEFAULT 0)")
    if vectors:
        conn.execute("CREATE TABLE node_enrichments (id TEXT, node_id TEXT, vector_type TEXT, "
                     "text TEXT, embedding BLOB, model TEXT, created_at TEXT)")
    return conn


def _serve(tmp_path, monkeypatch, conn, taxonomy):
    """Close the brain, point the dashboard at it, and read the payload."""
    conn.commit()
    conn.close()
    (tmp_path / 'aspects_v1.json').write_text(json.dumps(taxonomy))
    monkeypatch.setenv('BRAIN_DB_DIR', str(tmp_path))
    monkeypatch.setenv('ASPECTS_JSON_PATH', str(tmp_path / 'aspects_v1.json'))
    return graph.query_graph3d()


def _brain(tmp_path, monkeypatch, relations):
    """relations: (src, tgt, relation, weight, created_at[, archived])."""
    conn = _schema(tmp_path)
    for nid in ('a', 'b', 'c'):
        conn.execute("INSERT INTO nodes (id, type, title) VALUES (?, 'lesson', ?)",
                     (nid, nid))
    for i, (src, tgt, rel, weight, created_at, *archived) in enumerate(relations):
        conn.execute("INSERT INTO edges VALUES (?)", ('e%d' % i,))
        conn.execute("INSERT INTO edge_relations VALUES (?,?,?,?,?,?,?)",
                     ('e%d' % i, src, tgt, rel, weight, created_at,
                      archived[0] if archived else 0))
    return _serve(tmp_path, monkeypatch, conn, _TAXONOMY)


def _kept(payload):
    return {(e['source'], e['target']): e['relation'] for e in payload['edges']}


def test_the_strongest_relation_of_a_pair_wins(tmp_path, monkeypatch):
    kept = _kept(_brain(tmp_path, monkeypatch, [
        ('a', 'b', 'explains', 0.5, '2026-01-01'),
        ('a', 'b', 'depends_on', 0.9, '2026-02-01'),
        ('a', 'b', 'grounds', 0.6, '2026-03-01'),
    ]))
    assert kept == {('a', 'b'): 'depends_on'}


def test_a_tie_keeps_the_earliest_stated(tmp_path, monkeypatch):
    kept = _kept(_brain(tmp_path, monkeypatch, [
        ('a', 'b', 'grounds', 0.6, '2026-03-01'),
        ('a', 'b', 'explains', 0.6, '2026-01-01'),
    ]))
    assert kept == {('a', 'b'): 'explains'}


def test_noise_links_are_not_sent_and_never_hide_a_real_one(tmp_path, monkeypatch):
    # Noise is bookkeeping the graph never draws or counts: a stronger noise
    # relation leaves the pair its real one, and a noise-only pair is not sent.
    kept = _kept(_brain(tmp_path, monkeypatch, [
        ('a', 'b', 'co_anchored', 0.9, '2026-01-01'),
        ('a', 'b', 'grounds', 0.5, '2026-02-01'),
        ('b', 'c', 'co_anchored', 0.5, '2026-01-01'),
        ('a', 'c', 'grounds', 0.9, '2026-01-01', 1),
    ]))
    assert kept == {('a', 'b'): 'grounds'}


def test_membership_arrives_on_the_node_not_as_links(tmp_path, monkeypatch):
    payload = _brain_with_topics(tmp_path, monkeypatch)
    assert not any(e['relation'] == 'community_member' for e in payload['edges'])
    members = {n['id']: n['community'] for n in payload['nodes'] if n['community']}
    assert members['self0'] == 'self' and len(members) == 12


def test_the_payload_carries_the_live_primary_aspects(tmp_path, monkeypatch):
    payload = _brain(tmp_path, monkeypatch, [('a', 'b', 'grounds', 0.5, '2026-01-01')])
    assert payload['aspects'] == {
        'node_types': {'lesson': 'lesson_insight'},
        'edge_relations': {'depends_on': 'dependency_flow',
                           'grounds': 'explanation_causation',
                           'explains': 'explanation_causation',
                           'co_anchored': 'noise'}}


def test_a_broken_taxonomy_degrades_to_default_colours(tmp_path, monkeypatch):
    _brain(tmp_path, monkeypatch, [])
    (tmp_path / 'aspects_v1.json').write_text('["not", "an", "object"]')
    payload = graph.query_graph3d()
    assert payload['aspects'] == {'node_types': {}, 'edge_relations': {}}
    assert len(payload['nodes']) == 3


def test_every_primary_aspect_has_a_colour_slot(monkeypatch):
    # An aspect that is some string's primary but has no slot draws that
    # string in the default colour — a silent recolour.
    src = (ROOT / 'dashboard' / 'static' / 'tabs' / 'graph.js').read_text()
    slots = dict(re.findall(r'(\w+):(\d+)',
                            re.search(r'const ASPECT_SLOT=\{(.*?)\};', src).group(1)))
    monkeypatch.setenv('ASPECTS_JSON_PATH', SEED_ASPECTS_JSON_PATH)
    maps = primary_aspects()
    primaries = set(maps['node_types'].values()) | set(maps['edge_relations'].values())
    assert primaries <= set(slots), primaries - set(slots)
    seed = json.loads(Path(SEED_ASPECTS_JSON_PATH).read_text())
    assert set(slots) <= {name for name in seed if not name.startswith('_')}
    hues = len(re.findall(r'\[\d+,\d+,\d+\]',
                          re.search(r'const HUE=\[(.*?)\];', src, re.S).group(1)))
    assert sorted(int(v) for v in slots.values()) == list(range(hues))


def _brain_with_topics(tmp_path, monkeypatch, vectors=True, topics=None):
    """Communities on distinct topics — by default identity memories along
    +e0, lessons along -e0, a third along e1 — each member a little off its
    topic."""
    conn = _schema(tmp_path, vectors=True)
    rng = np.random.default_rng(7)
    topics = topics or {'self': ('identity', 0, 1.0), 'craft': ('lesson', 0, -1.0),
                        'ops': ('lesson', 1, 1.0)}
    for c, (ntype, axis, sign) in topics.items():
        conn.execute("INSERT INTO nodes (id, type, title) VALUES (?, 'community', ?)", (c, c.upper()))
        for k in range(4):
            nid = '%s%d' % (c, k)
            conn.execute("INSERT INTO nodes (id, type, title) VALUES (?, ?, ?)", (nid, ntype, nid))
            eid = 'm-' + nid
            conn.execute("INSERT INTO edges VALUES (?)", (eid,))
            conn.execute("INSERT INTO edge_relations VALUES (?,?,?,?,?,?,?)",
                         (eid, c, nid, 'community_member', 1.0, '2026-01-01', 0))
            if vectors:
                v = rng.normal(0, 0.15, 8).astype(np.float32)
                v[axis] += sign
                conn.execute("INSERT INTO node_enrichments VALUES (?,?,?,?,?,?,?)",
                             (nid, nid, '_primary', '', v.tobytes(), 'm', '2026-01-01'))
    return _serve(tmp_path, monkeypatch, conn,
                  dict(_TAXONOMY, identity_bearing={'node_types': ['identity']},
                       noise={'edge_relations': ['co_anchored', 'community_member']}))


def test_depth_puts_identity_up_and_keeps_each_knot_at_its_topic(tmp_path, monkeypatch):
    payload = _brain_with_topics(tmp_path, monkeypatch)
    depth = {n['id']: n['depth'] for n in payload['nodes'] if 'depth' in n}
    selves = [depth['self%d' % k] for k in range(4)]
    crafts = [depth['craft%d' % k] for k in range(4)]
    # The main axis runs self ↔ craft, and identity is its positive side.
    assert min(selves) > 0 > max(crafts)
    assert all(-1 <= d <= 1 for d in depth.values())
    # A community's own node sits at its knot's topic.
    assert depth['self'] > 0 > depth['craft']
    assert payload['depth_axis']['up'][0] == 'SELF'
    assert payload['depth_axis']['down'][0] == 'CRAFT'
    assert set(payload['communities']) == {'self', 'craft', 'ops'}
    assert all(0 <= c['coherence'] <= 1 for c in payload['communities'].values())


def test_a_brain_without_vectors_keeps_a_flat_disc(tmp_path, monkeypatch):
    payload = _brain_with_topics(tmp_path, monkeypatch, vectors=False)
    assert not any('depth' in n for n in payload['nodes'])
    assert payload['communities'] == {}
    assert payload['depth_axis'] is None


def test_a_brain_without_the_enrichments_table_keeps_a_flat_disc(tmp_path, monkeypatch):
    payload = _brain(tmp_path, monkeypatch, [('a', 'b', 'grounds', 0.5, '2026-01-01')])
    assert not any('depth' in n for n in payload['nodes'])
    assert payload['depth_axis'] is None


def _add_vector(tmp_path, nid, vector, model):
    conn = sqlite3.connect(tmp_path / 'brain.db')
    conn.execute("INSERT INTO node_enrichments VALUES (?,?,?,?,?,?,?)",
                 (nid + model, nid, '_primary', '', np.asarray(vector, np.float32).tobytes(),
                  model, '2026-02-01'))
    conn.commit()
    conn.close()


def test_a_stray_vector_from_another_model_does_not_flatten_the_brain(tmp_path, monkeypatch):
    _brain_with_topics(tmp_path, monkeypatch)
    _add_vector(tmp_path, 'self0', np.ones(384), 'other-model')
    payload = graph.query_graph3d()
    assert sum('depth' in n for n in payload['nodes']) == 15


def test_a_brain_with_one_community_takes_its_axis_from_its_memories(tmp_path, monkeypatch):
    payload = _brain_with_topics(tmp_path, monkeypatch, topics={'seed': ('identity', 0, 1.0)})
    depth = [n['depth'] for n in payload['nodes'] if 'depth' in n]
    assert len(depth) == 5 and max(depth) > min(depth)
    # Poles name memories, and no coherence rank is invented from one community.
    assert set(payload['depth_axis']['up'] + payload['depth_axis']['down']) <= {
        'seed0', 'seed1', 'seed2', 'seed3'}
    assert payload['communities'] == {}


def test_a_new_vector_refreshes_the_cached_depth(tmp_path, monkeypatch):
    payload = _brain_with_topics(tmp_path, monkeypatch)
    before = {n['id']: n.get('depth') for n in payload['nodes']}
    # A re-embed rewrites the row and stamps it, as the brain's writer does.
    conn = sqlite3.connect(tmp_path / 'brain.db')
    conn.execute("UPDATE node_enrichments SET embedding = ?, created_at = '2026-02-01' "
                 "WHERE node_id = 'ops0'", (np.float32(-5 * np.eye(8)[1]).tobytes(),))
    conn.commit()
    conn.close()
    after = {n['id']: n.get('depth') for n in graph.query_graph3d()['nodes']}
    assert after != before
