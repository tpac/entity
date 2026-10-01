"""Root↔neighbor dedup in the activation render.

When a node is rendered as a top-level (root) memory in the inject, AND it's
also a neighbor-edge of another rendered root, its content appears twice — once
in full as a root, once as a redundant edge-line. Spread-activation makes this
systematic (a root's neighbors are exactly the nodes most likely to also be
roots). These tests lock the dedup: an edge-line pointing to an already-rendered
root is dropped; edges to non-rendered nodes are KEPT (conservative — nothing
goes invisible); root blocks themselves are untouched.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.scales.s1.surface_contract import (
    _render_seed,
    render_surface_inject,
)


def _conn(nid, rel='supports', title=None, type_='fact'):
    return {
        'id': nid, 'type': type_,
        'title': title or ('Node ' + nid[0].upper()),
        'relation': rel, 'direction': 'outgoing',
        'edge_description': 'edge toward ' + nid,
    }


def _node(nid, title, connections=None, type_='principle'):
    return {
        'id': nid, 'type': type_, 'title': title,
        'content': 'Content for ' + title + '.',
        'connections': connections or [], 'metadata_kv': {},
    }


class TestRenderDropsEdgeToSeenRoot:
    def _node_a(self):
        return _node('aaaaaaaa', 'Node A', connections=[
            _conn('bbbbbbbb', 'supports', 'Node B'),
            _conn('cccccccc', 'extends', 'Node C'),
        ])

    def test_without_dedup_both_edges_render(self):
        out = _render_seed(self._node_a(), 'arc', 2000,
                           query_vec=None, brain=None,
                           seen_root_ids=None)[0]
        assert 'bbbbbbbb' in out  # edge to B
        assert 'cccccccc' in out  # edge to C

    def test_edge_to_seen_root_is_dropped(self):
        # B is already a rendered root → A's edge-line to B is redundant.
        out = _render_seed(self._node_a(), 'arc', 2000,
                           query_vec=None, brain=None,
                           seen_root_ids={'bbbbbbbb'})[0]
        assert 'bbbbbbbb' not in out   # dropped — B shown in full elsewhere
        assert 'cccccccc' in out       # C is NOT a root → kept (conservative)

    def test_dedup_shrinks_the_render(self):
        a = self._node_a()
        big = _render_seed(a, 'arc', 2000, query_vec=None,
                           brain=None, seen_root_ids=None)[0]
        small = _render_seed(a, 'arc', 2000, query_vec=None,
                             brain=None, seen_root_ids={'bbbbbbbb'})[0]
        assert len(small) < len(big)


class TestInjectEndToEnd:
    def test_root_neighbor_dedup_in_full_inject(self):
        # B (higher activation) renders first as a root; A (lower) renders
        # second and lists B + C as edges. B's edge under A must be dropped.
        b = _node('bbbbbbbb', 'Node B', connections=[])
        a = _node('aaaaaaaa', 'Node A', connections=[
            _conn('bbbbbbbb', 'supports', 'Node B'),
            _conn('cccccccc', 'extends', 'Node C'),
        ])
        rich = {'bbbbbbbb': b, 'aaaaaaaa': a}
        acts = {'bbbbbbbb': 0.9, 'aaaaaaaa': 0.5}

        out = render_surface_inject(
            acts, {}, rich, selected_mode={'bbbbbbbb': 'arc', 'aaaaaaaa': 'arc'},
            query_vec=None, brain=None)['text']

        assert 'Node B' in out and 'Node A' in out   # both render as roots
        assert out.count('bbbbbbbb') == 1            # ONCE (root header), not also A's edge
        assert 'cccccccc' in out                     # A's edge to non-root C survives

    def test_no_seen_means_no_change_for_first_root(self):
        # The first (highest-activation) root renders against an empty seen
        # set — its edges are untouched. Guards against over-dedup.
        a = _node('aaaaaaaa', 'Node A', connections=[
            _conn('dddddddd', 'extends', 'Node D'),
        ])
        out = render_surface_inject(
            {'aaaaaaaa': 0.9}, {}, {'aaaaaaaa': a},
            selected_mode={'aaaaaaaa': 'arc'}, query_vec=None, brain=None)['text']
        assert 'dddddddd' in out  # D never rendered as root → edge kept
