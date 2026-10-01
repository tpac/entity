"""Encoder view policy (servers/scales/s1/encoder_view.py) — the flag-gated
feeding decisions: catalog aging, action filtering/stubbing, the provenance
verb split, and the encoded-turn trim reversal.

Two invariants under test, per policy surface:
  • Flag OFF (default) → byte-identical to the pre-policy render — the A/B
    control arm. Every render test in test_s1e_trace_links.py already runs
    with the default; here the OFF arm is asserted explicitly against the
    legacy markers.
  • Flag ON → the filter marks itself in place (a stubbed <actions> says
    trimmed, an aged entry announces its unshown edges with the get_nodes
    hint) — absence must never read as "nothing happened".
"""
import os
import sys
import pytest


@pytest.fixture(autouse=True)
def balanced_action_policy(monkeypatch):
    monkeypatch.setenv('BRAIN_ACTIONS_PROFILE', 'balanced')
    monkeypatch.setenv('BRAIN_ACTIONS_EXCLUDE_GIT', '0')
    monkeypatch.setenv('BRAIN_ACTIONS_MAX_LINES', '200')
    monkeypatch.setenv('BRAIN_ACTIONS_MAX_BYTES', '24000')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s1.encoder_view import (  # noqa: E402
    view_policy_enabled, aging_cutoff, catalog_view, action_mode, action_stub,
    actions_stub_line, CATALOG_FULL_ROUNDS,
    DROPPED_ACTION_TOOLS, STUBBED_ACTION_TOOLS,
    PROVENANCE_SPLIT,
)
from servers.scales.s1.encode import _render_lived_sequence_timeline  # noqa: E402
from servers.scales.s1.encode_contract import build_node_catalog  # noqa: E402

SHORT = 'abcd1234'

# The plugin-adapter tool prefix, built from the manifests — never the literal
# (test_deploy_contract's containment gate keeps the adapter shape out of
# source; .claude/settings.json is its one legitimate home). CC's shape is
# mcp__plugin_<plugin>_<server>__<tool>: the plugin name comes from plugin.json,
# the server name is the .mcp.json key — two namespaces (D-11), not one.
import json  # noqa: E402
with open(os.path.join(ROOT, '.claude-plugin', 'plugin.json')) as _f:
    _PLUGIN = json.load(_f)['name']
with open(os.path.join(ROOT, '.mcp.json')) as _f:
    _servers = list(json.load(_f))
assert _servers == ['brain'], f'.mcp.json servers changed: {_servers}'
_SERVER = _servers[0]
PLUGIN_TOOL = ('mcp__plugin_%s_%s__' % (_PLUGIN, _SERVER)) + '%s'


# ── the flag ──

def test_flag_on_by_default(monkeypatch):
    # Activated 2026-08-18 on an operator gate: unset means the policy; 0 is the
    # emergency off-switch and the A/B control arm.
    monkeypatch.delenv('BRAIN_S1E_VIEW_POLICY', raising=False)
    assert view_policy_enabled() is True
    monkeypatch.setenv('BRAIN_S1E_VIEW_POLICY', '1')
    assert view_policy_enabled() is True
    monkeypatch.setenv('BRAIN_S1E_VIEW_POLICY', '0')
    assert view_policy_enabled() is False


# ── pure policy: aging tiers ──

def test_aging_cutoff_is_nth_newest_run_stop():
    assert aging_cutoff([5, 10, 15]) == 10        # 2nd-newest of 3
    assert aging_cutoff([5, 10]) == 5             # exactly N runs
    assert aging_cutoff([5]) is None              # fewer than N → no aging
    assert aging_cutoff([]) is None
    assert aging_cutoff(None) is None
    assert aging_cutoff([10, 5, 10]) == 5         # unsorted + dup input


def test_catalog_view_orders_oldest_to_newest_unknown_last():
    ids = {'old1', 'new1', 'surf'}
    stops = {'old1': 3, 'new1': 15}
    ordered, aged = catalog_view(ids, stops, run_stops=[10, 15])
    assert ordered == ['old1', 'new1', 'surf']    # unknown stop sorts last
    assert aged == {'old1'}                       # 3 < cutoff(10)


def test_catalog_view_protected_ids_never_age():
    stops = {'old1': 3, 'old2': 4}
    ordered, aged = catalog_view({'old1', 'old2'}, stops, run_stops=[10, 15],
                                 protected={'old2'})   # surfaced this window
    assert aged == {'old1'}
    assert 'old2' not in aged


def test_catalog_view_no_aging_before_enough_runs():
    ordered, aged = catalog_view({'a', 'b'}, {'a': 1, 'b': 2}, run_stops=[5])
    assert aged == set()
    assert ordered == ['a', 'b']                  # still ordered by stop


def test_catalog_view_boundary_stop_stays_full():
    # An id last touched exactly AT the cutoff belongs to the newest N rounds.
    ordered, aged = catalog_view({'edge'}, {'edge': 10}, run_stops=[10, 15])
    assert aged == set()


# ── pure policy: action visibility ──

def test_action_mode_drop_stub_full():
    # writes + by-id reads: provenance carries everything → drop
    for t in ('remember_batch', 'revise', 'brain_batch', 'get_nodes'):
        assert action_mode(PLUGIN_TOOL % t) == 'drop'
    assert action_mode('mcp__brain__revise') == 'drop'   # user-scope registration
    # search tools: the query head survives as a stub (intent + empty results)
    for t in ('recall', 'recall_batch', 'find_node_by_title', 'filter_nodes'):
        assert action_mode(PLUGIN_TOOL % t) == 'stub'
    # edge ops stay visible — "connect has no provenance home, by contract"
    for t in ('connect', 'disconnect', 'revise_edge'):
        assert action_mode(PLUGIN_TOOL % t) == 'full'
    # non-brain tools stay visible, whatever their basename
    assert action_mode('Bash') == 'full'
    assert action_mode('mcp__slack__get_nodes') == 'full'
    # unknown defaults to visible
    assert action_mode(None) == 'full'
    assert action_mode('') == 'full'


def test_action_stub_keeps_query_head():
    s = action_stub(PLUGIN_TOOL % 'recall' + ': {"query": "wal-index '
                    'contention", "limit": 8, "filter": {"type": {"in": '
                    '["decision"]}}}')
    assert s.startswith('recall: {"query": "wal-index contention"')
    assert s.endswith('→ results in provenance')
    assert '…' in s and len(s) < 110                 # trimmed, marked
    # short args survive whole, no ellipsis
    s2 = action_stub(PLUGIN_TOOL % 'filter_nodes' + ': {"field": "type"}')
    assert s2 == 'filter_nodes: {"field": "type"} → results in provenance'
    assert isinstance(action_stub(None), str)        # defensive, never raises


def test_actions_stub_marks_itself():
    line = actions_stub_line(7)
    assert 'trimmed' in line and '7' in line


# ── timeline render, policy ON vs OFF (the encode.py wiring) ──
# Fixtures mirror test_s1e_trace_links: turns at stops 5-8, run@7 covers 5,6.

def _msg(ref_type, tid, stop, content, ts, tool=None):
    md = {'content': content} if tool is None else {'tool': tool}
    return {'id': tid, 'ref_type': ref_type,
            'chain_id': 's0-%s-%d' % (SHORT, stop),
            'summary': content[:120], 'created_at': ts, 'metadata': md}


def _encode_trace(stop, created, revised, tid=None):
    return {'id': tid or ('e%d' % stop),
            'chain_id': 's1e-%s-%d' % (SHORT, stop),
            'metadata': {'created': created, 'revised': revised}}


def _touched_trace(stop, **keys):
    md = {k: [] for k in ('created', 'revised', 'archived', 'recalled', 'endo')}
    md.update(keys)
    return {'id': 'at%d' % stop, 'chain_id': 's0-%s-%d' % (SHORT, stop),
            'metadata': md}


class _Brain:
    """Serves the doors the lived timeline pulls: recall_episodes,
    query_traces (gather), and naked title lookups."""
    def __init__(self, episodes, surface=(), encode=(), touched=(), titles=None):
        self._episodes = list(episodes)
        self._surface, self._encode, self._touched = \
            list(surface), list(encode), list(touched)
        self._nodes = self._Nodes(titles or {})

    class _Nodes:
        def __init__(self, titles):
            self._titles = titles

        def get_bulk(self, ids):
            return {i: {'title': self._titles[i]}
                    for i in ids if i in self._titles}

    def recall_episodes(self, **kw):
        rts = kw.get('ref_type') or []
        eps = [e for e in self._episodes if e.get('ref_type') in rts]
        return {'episodes': eps}

    def query_traces(self, **kw):
        rt = kw.get('ref_type')
        events = {'surface_selected': self._surface,
                  'encoding_run': self._encode,
                  'anchor_touched': self._touched}.get(rt, [])
        return {'events': events}


def _two_turn_eps(covered_body='covered turn text', tail_body='tail turn text'):
    return [
        _msg('user_message', 'u5', 5, covered_body, '2026-08-01T00:00:01'),
        _msg('assistant_message', 'a5', 5, 'reply five', '2026-08-01T00:00:02'),
        _msg('tool_result', 't5a', 5, 'Bash: ls -la', '2026-08-01T00:00:03',
             tool='Bash'),
        _msg('user_message', 'u6', 6, tail_body, '2026-08-01T00:00:04'),
        _msg('assistant_message', 'a6', 6, 'reply six', '2026-08-01T00:00:05'),
        _msg('tool_result', 't6a', 6, 'Edit: foo.py', '2026-08-01T00:00:06',
             tool='Edit'),
        _msg('tool_result', 't6b', 6,
             PLUGIN_TOOL % 'remember_batch' + ': {"nodes": [...]}',
             '2026-08-01T00:00:07', tool=PLUGIN_TOOL % 'remember_batch'),
        _msg('tool_result', 't6c', 6,
             PLUGIN_TOOL % 'recall' + ': {"query": "catalog aging design"}',
             '2026-08-01T00:00:08', tool=PLUGIN_TOOL % 'recall'),
    ]


_TWO_MSGS = [{'role': 'user'} for _ in range(2)]


def _render(view_policy, eps=None, now=None, **brain_kw):
    brain_kw.setdefault('encode', [_encode_trace(6, ['nodeCCCC3333'], [],
                                                 tid='run6')])
    brain = _Brain(eps if eps is not None else _two_turn_eps(), **brain_kw)
    return _render_lived_sequence_timeline(brain, 'sess', _TWO_MSGS,
                                           view_policy=view_policy, now=now)


def test_policy_off_is_legacy_render():
    # Control arm: no policy markers, actions listed verbatim, and covered
    # turns keep full text here too (encoded_turn_trim retired at activation).
    long_body = 'x' * 800
    out = _render(False, eps=_two_turn_eps(covered_body='covered ' + long_body))
    t5 = out.split('<turn n="1"')[1].split('</turn>')[0]
    assert t5.split('<other')[1].split('</other>')[0].count('x') == 800
    assert 'trimmed —' not in out                              # no stub
    assert 'remember_batch' in out                             # node op visible
    assert 'Bash: ls -la' in out


def test_policy_on_encoded_turn_full_text_and_actions_stub():
    long_body = 'x' * 800
    out = _render(True, eps=_two_turn_eps(covered_body='covered ' + long_body))
    t5 = out.split('<turn n="6"')[1].split('</turn>')[0]
    t6 = out.split('<turn n="7"')[1].split('</turn>')[0]
    # covered turn keeps FULL text …
    assert t5.split('<other')[1].split('</other>')[0].count('x') == 800
    # … and its actions collapse to the self-marking stub (element stays)
    assert '<actions>trimmed — 1 action(s)' in t5
    assert 'Bash: ls -la' not in t5
    # the unencoded tail keeps its actions
    assert 'Edit: foo.py' in t6


def test_tail_budget_wiring_reaches_the_renderer():
    # Transition: the TAIL turn gets the tail budget, older turns the
    # per-turn budget — pinned through _render_lived_sequence_timeline
    # itself, not just the condenser parameter (the four-angle review
    # mutation-tested `is_tail=False` and nothing failed).
    from servers.scales.s1.encoder_view import (
        ACTION_BUDGETS, ACTIONS_BUDGET_SOFT_EDGE)
    budget = ACTION_BUDGETS['balanced'][0]
    n = budget + ACTIONS_BUDGET_SOFT_EDGE + 3   # rolls on old turns only
    def eps_with_flood(flood_stop):
        # turn grouping is sequential — build episodes in lived order
        # timestamps must sit inside the owning turn — the renderer re-sorts
        # episodes chronologically before grouping
        base = '03' if flood_stop == 5 else '10'
        flood = [_msg('tool_result', 'f%d' % i, flood_stop,
                      'Bash: sweep %d servers/x.py' % i,
                      '2026-08-01T00:00:%s.%03d' % (base, i), tool='Bash')
                 for i in range(n)]
        eps = [_msg('user_message', 'u5', 5, 'first turn', '2026-08-01T00:00:01'),
               _msg('assistant_message', 'a5', 5, 'reply five', '2026-08-01T00:00:02')]
        if flood_stop == 5:
            eps += flood
        eps += [_msg('user_message', 'u6', 6, 'tail turn', '2026-08-01T00:00:04'),
                _msg('assistant_message', 'a6', 6, 'reply six', '2026-08-01T00:00:05')]
        if flood_stop == 6:
            eps += flood
        return eps
    # flood on the FIRST (older) turn → small budget → rollup appears
    out_old = _render(True, eps=eps_with_flood(5), encode=[])
    assert 'more actions, not shown' in out_old
    # same flood on the LAST (tail) turn → big budget → renders verbatim
    out_tail = _render(True, eps=eps_with_flood(6), encode=[])
    assert 'more actions, not shown' not in out_tail


def test_policy_on_drops_node_op_lines_on_unencoded_turns():
    out = _render(True)
    t6 = out.split('<turn n="7"')[1].split('</turn>')[0]
    assert 'remember_batch' not in t6          # provenance owns it now
    assert 'Edit: foo.py' in t6                # world-changing lines stay


def test_policy_on_search_lines_stub_to_query_head():
    # recall keeps its intent (the query) but points at provenance for results
    out = _render(True)
    t6 = out.split('<turn n="7"')[1].split('</turn>')[0]
    assert ('recall: {"query": "catalog aging design"} → results in provenance'
            in t6)
    assert PLUGIN_TOOL % 'recall' not in t6   # bare name, no prefix
    # control arm renders the raw line untouched
    off = _render(False)
    assert PLUGIN_TOOL % 'recall' + ': {"query"' in off
    assert '→ results in provenance' not in off


def test_policy_on_all_actions_hidden_drops_element():
    # A tail turn whose only actions are dropped node-ops: nothing to show —
    # unencoded turns carry no coverage claim, so the element may drop.
    eps = [e for e in _two_turn_eps() if e['id'] not in ('t6a', 't6c')]
    out = _render(True, eps=eps)
    t6 = out.split('<turn n="7"')[1].split('</turn>')[0]
    assert '<actions>' not in t6


def test_policy_on_provenance_verb_split():
    touched = [_touched_trace(6, created=['nodeAAAA1111'],
                              revised=['nodeBBBB2222'],
                              recalled=['nodeDDDD4444'],
                              looked_up=['nodeFFFF6666', 'nodeDDDD4444'],
                              archived=['nodeEEEE5555'])]
    titles = {'nodeAAAA1111': 'fresh insight', 'nodeDDDD4444': 'looked up',
              'nodeFFFF6666': 'search hit'}
    on = _render(True, touched=touched, titles=titles)
    t5 = on.split('<turn n="6"')[1].split('</turn>')[0]
    t6 = on.split('<turn n="7"')[1].split('</turn>')[0]
    # the canonical node_ref token under the policy — identical to the shape
    # the catalog header, the edge lines and the ⚠ blocks use
    # write verbs carry the id alone — the body is a catalog entry above
    assert 'created(me): (id:nodeAAAA)' in t6
    assert 'revised(me): (id:nodeBBBB)' in t6            # no title → id alone
    # recalled KEEPS its titles: only 43% of recalled ids have a catalog
    # entry, so a bare id there is a reference with no body anywhere
    # recalled(me) merges by-id reads with search results, deduped
    assert ('recalled(me): "looked up" (id:nodeDDDD) "search hit" (id:nodeFFFF)'
            in t6)
    assert t6.count('id:nodeDDDD') == 1
    assert 'archived(me): (id:nodeEEEE)' in t6
    # run attribution speaks turn coordinates (run6's post-increment stop =
    # 1-based already) — and it MATCHES the covered turn's displayed number
    assert 'encoded(me, turn 6): (id:nodeCCCC)' in t5      # write verb → id alone
    assert 'encoded(S1S)' not in on
    assert 'encoded(me):' not in on             # the merged label retires
    # control arm: merged label, no verbs, id-first «tag», no looked_up render
    off = _render(False, touched=touched, titles=titles)
    assert 'encoded(me): id:nodeAAAA «fresh insight»' in off
    assert 'encoded(S1S): id:nodeCCCC' in off
    assert 'created(me)' not in off and 'recalled(me)' not in off
    assert 'nodeFFFF' not in off


def test_policy_on_turn_age_and_now_stamp():
    from datetime import datetime, timezone
    from servers.scales.s1.encoder_view import timeline_now_attr
    now = datetime(2026, 8, 1, 3, 0, 0, tzinfo=timezone.utc)  # eps at ~00:00
    on = _render(True, now=now)
    # 2h59m floors to 2h — same floor semantics as the coarse 'Nd ago' scale;
    # n is the REAL turn number, 1-based (chain stop + 1) under the policy
    assert '<turn n="6" age="2h ago" encoded="true">' in on
    assert '<turn n="7" age="2h ago" encoded="false">' in on
    # no now → no age attr (degraded render, never a broken one)
    assert 'age=' not in _render(True, now=None)
    # control arm never carries ages
    assert 'age=' not in _render(False, now=now)
    # the <timeline now="…"> stamp (assembled in _build_user_content)
    assert timeline_now_attr(now) == ' now="2026-08-01 03:00 UTC"'
    assert timeline_now_attr(None) == ''


# ── catalog aging (build_node_catalog wiring) ──

class _Aspects:
    def relations_in(self, names):
        return ('corrects', 'supersedes', 'fixes', 'resolves', 'reframes') \
            if 'correction_improvement' in names else ()


class _CatalogBrain:
    """Stub for build_node_catalog: node bodies + the community-filter conn
    + the aspect registry the correction dedup reads."""
    aspects = _Aspects()

    def __init__(self, nodes):
        self._map = nodes

    def execute(self, sql, params=None):
        return []                                # no community nodes

    conn = property(lambda self: self)

    def get_node(self, ids):
        return {i: dict(self._map[i]) for i in ids if i in self._map}


def _node(nid, content_chars=1200, edges=2):
    return {
        'id': nid, 'type': 'finding', 'title': 'title of %s' % nid,
        # distinctive tail — the truncation tests key on it: a repeating body
        # would let content[-40:] match inside a truncated head (vacuous pass)
        'content': (('claim first. ' + 'body ' * 400)[:content_chars - 20]
                    + ' <<tail of %s>>' % nid),
        'situation': 'when testing catalog aging',
        'created_at': '2026-08-01T00:00:00',
        '_metadata': {'reasoning': 'long reasoning ' * 20},
        '_corrections': [{'id': 'corr1234', 'title': 'the correction',
                          'type': 'decision', 'direction': 'corrector',
                          'relation': 'corrects',
                          'edge_description': 'why it changed',
                          'relation_at': '2026-08-01T00:00:00',
                          'content': 'corrector body ' * 30}],
        'connections': [{'id': 'tgt%d' % i, 'title': 'target %d' % i,
                         'relation': 'grounds', 'direction': 'outgoing',
                         'description': 'edge description ' * 5,
                         'created_at': '2026-08-01T00:00:00'}
                        for i in range(edges)],
    }


def _catalog(view_policy, extra_ids):
    brain = _CatalogBrain({'oldnode1': _node('oldnode1'),
                           'newnode1': _node('newnode1')})
    return build_node_catalog([], brain, extra_ids=extra_ids,
                              view_policy=view_policy)


_EXTRA = {'encoded': {'oldnode1', 'newnode1'}, 'authored': set(),
          'recalled': set(), 'stops': {'oldnode1': 5, 'newnode1': 15},
          'run_stops': [5, 10, 15]}


def test_full_tier_trim_and_edge_shape():
    """The 2026-09-16 trim (CATALOG_TRIM_CONFIG), on the FULL tier only and
    only under the flag — the four dropped fields, the call-shaped edge line,
    the canonical id token, and the direction-split correction render."""
    on, _ = _catalog(True, _EXTRA)
    entry = on.split('title of newnode1')[1]           # newest round = full
    # the four fields the measurement retired
    for gone in ('Situation:', 'Question:', 'My Raw Quote:', 'Communities:'):
        assert gone not in entry
    # edge line: one line per field of the write that would repair it
    assert '→ grounds "target 0" (id:tgt0 · ' in entry
    assert '\n      description: edge description' in entry
    assert '[finding id:tgt0' not in entry            # the old filing-record form
    assert ' this grounds ' not in entry              # direction rides the arrow
    assert 'edges: → reads "this <relation> …"' in on  # …explained once, in the header
    # canonical id token: metadata beside the id, never comma-joined
    assert '(id:newnode1 · ' in entry
    # corrections: the corrector reads like its Edges line — same reference
    # token, the verb after a dash, description on its own line — then its body
    assert '⚠ Updated by "the correction" (id:corr1234 · decision · ' in entry
    assert ') — corrects\n     description: why it changed\n' in entry
    assert 'corrector body' in entry
    # flag OFF is the A/B control — none of the above
    off, _ = _catalog(False, _EXTRA)
    assert 'Situation: when testing catalog aging' in off
    assert '→ grounds' not in off and 'this grounds' in off
    assert '(id:newnode1, ' in off


def test_catalog_aging_trims_old_rounds_keeps_new_full():
    text, ids = _catalog(True, _EXTRA)
    assert ids == {'oldnode1', 'newnode1'}
    old_entry = text.split('title of oldnode1')[1].split('[encoded(me, turn 15)]')[0]
    new_entry = text.split('title of newnode1')[1]
    # aged: no edges, no reasoning, lean correction — but the body stays WHOLE
    # (the arm-D invariant: the tail C's truncation destroyed must survive)
    assert 'Edges:' not in old_entry
    assert 'Reasoning' not in old_entry
    assert 'corrector body' not in old_entry           # heavy body gone …
    assert '⚠ Updated by "the correction" (id:corr1234)\n' in old_entry   # … pointer stays
    # situation dropped from BOTH tiers (2026-09-16): the whole body is the
    # "does this already exist" anchor, so the one-line restatement is cost
    assert 'Situation: when testing catalog aging' not in old_entry
    assert _node('oldnode1')['content'][-40:] in old_entry   # tail intact
    # …but the withheld edges announce themselves in place, NAMING the ids so
    # the neighbours are one call away rather than two (2026-09-16)
    assert 'Edges (2 not shown): get_nodes([' in old_entry
    assert '"tgt0"' in old_entry and '"tgt1"' in old_entry
    # fresh round keeps the full render
    assert 'Edges:' in new_entry
    assert 'corrector body' in new_entry
    # no aged tag or catalog header (the render is self-describing);
    # provenance tags speak first-person TURN coordinates
    assert '[encoded(me, turn 5)]' in text
    assert '[body only]' not in text and '[aged]' not in text
    assert '[encoded] ' not in text                  # legacy tag retired here
    # flag off keeps the legacy tag vocabulary
    off, _ = _catalog(False, _EXTRA)
    assert '[encoded] ' in off and '(me, turn' not in off


def test_catalog_explicit_cap_still_truncates():
    # The eval harness reproduces the retired truncating arms by passing
    # aged_content_chars explicitly — the override must keep cutting even
    # though the shipped policy default is body-whole.
    brain = _CatalogBrain({'oldnode1': _node('oldnode1'),
                           'newnode1': _node('newnode1')})
    text, _ = build_node_catalog([], brain, extra_ids=_EXTRA,
                                 view_policy=True, aged_content_chars=400)
    old_entry = text.split('title of oldnode1')[1].split('[encoded(me, turn 15)]')[0]
    assert _node('oldnode1')['content'][-40:] not in old_entry
    content_line = [l for l in old_entry.split('\n') if 'Content:' in l][0]
    assert len(content_line) < 400 + 30


def test_catalog_aging_orders_oldest_first():
    text, _ = _catalog(True, _EXTRA)
    assert text.index('title of oldnode1') < text.index('title of newnode1')


def test_catalog_policy_off_renders_full_everywhere():
    text, ids = _catalog(False, _EXTRA)   # stops/run_stops present but inert
    assert 'not shown — get_nodes' not in text
    assert text.count('Edges:') == 2      # both nodes full depth


def _corrected_node(nid):
    # a full-depth node whose connection to its corrector carries the verb its
    # ⚠ entry shows (duplicated by the block), a correction-aspect verb the
    # entry does NOT carry (an `unchanged` one — it stays an edge), and a
    # plain one
    n = _node(nid)
    n['_corrections'][0]['relation'] = 'supersedes'   # as correction_enrich writes it
    n['connections'].append({
        'id': 'corr1234', 'title': 'the correction', 'direction': 'incoming',
        'relation': 'supersedes', 'description': 'dup of the warn block',
        'created_at': '2026-08-01T00:00:00',
        'relations': [{'relation': 'supersedes', 'description': 'dup of the warn block'},
                      {'relation': 'addresses', 'description': 'it also addresses it'},
                      {'relation': 'extends', 'description': 'also extends it'}]})
    return n


def test_catalog_edge_total_indicator():
    # 7 edges, limit 5 → the header says so under the policy; flag off keeps
    # the bare header (byte-identity)
    n = _node('bignode11', edges=7)
    extra = {'encoded': {'bignode11'}, 'authored': set(), 'recalled': set(),
             'stops': {'bignode11': 15}, 'run_stops': [10, 15]}
    on, _ = build_node_catalog([], _CatalogBrain({'bignode11': _node('bignode11', edges=7)}),
                               extra_ids=extra, view_policy=True)
    assert 'Edges (5 of 7):' in on
    off, _ = build_node_catalog([], _CatalogBrain({'bignode11': n}),
                                extra_ids=extra, view_policy=False)
    assert 'Edges (5 of 7):' not in off and 'Edges:' in off
    # within the limit → plain header on both arms
    small, _ = build_node_catalog([], _CatalogBrain({'bignode11': _node('bignode11', edges=3)}),
                                  extra_ids=extra, view_policy=True)
    assert 'Edges:' in small and ' of ' not in small.split('Edges')[1][:12]


def test_catalog_correction_edge_dedup():
    extra = {'encoded': {'dednode11'}, 'authored': set(), 'recalled': set(),
             'stops': {'dednode11': 15}, 'run_stops': [10, 15]}
    # flag off: the duplication renders (⚠ block AND the edge relation)
    off, _ = build_node_catalog([], _CatalogBrain({'dednode11': _corrected_node('dednode11')}),
                                extra_ids=extra, view_policy=False)
    assert '⚠ Updated by "the correction"' in off
    assert 'dup of the warn block' in off
    # policy on: the verb the ⚠ entry carries dedups out of Edges; a
    # correction verb it does not carry and the plain relation on the same
    # connection survive
    on, _ = build_node_catalog([], _CatalogBrain({'dednode11': _corrected_node('dednode11')}),
                               extra_ids=extra, view_policy=True)
    assert '⚠ Updated by "the correction"' in on     # privileged render stays
    assert 'dup of the warn block' not in on
    assert 'it also addresses it' in on
    assert 'also extends it' in on
    # aged: the ⚠ pointer is lean (no verb), so nothing leaves Edges — the
    # not-shown pointer counts and names the corrector with the other edges
    aged_extra = dict(extra, stops={'dednode11': 10}, run_stops=[10, 15, 20])
    aged, _ = build_node_catalog([], _CatalogBrain({'dednode11': _corrected_node('dednode11')}),
                                 extra_ids=aged_extra, view_policy=True)
    assert '⚠ Updated by "the correction" (id:corr1234)\n' in aged
    assert 'Edges (3 not shown): get_nodes(["tgt0", "tgt1", "corr1234"])' in aged


def test_relative_time_fine_and_now_injection():
    from datetime import datetime, timezone
    from servers.scales.s1.surface_contract import _relative_time
    now = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    assert _relative_time('2026-08-16T11:35:00+00:00', now=now, fine=True) == '25m ago'
    assert _relative_time('2026-08-16T09:00:00+00:00', now=now, fine=True) == '3h ago'
    assert _relative_time('2026-08-16T11:59:30+00:00', now=now, fine=True) == 'just now'
    # past a day, fine falls through to the coarse scale
    assert _relative_time('2026-08-14T09:00:00+00:00', now=now, fine=True) == '2d ago'
    # default (surface) vocabulary is untouched
    assert _relative_time('2026-08-16T09:00:00+00:00', now=now) == 'today'


def test_catalog_header_relative_time_and_session_ownership():
    from datetime import datetime, timezone
    brain = _CatalogBrain({'oldnode1': _node('oldnode1'),
                           'newnode1': _node('newnode1')})
    now = datetime(2026, 8, 1, 3, 0, 0, tzinfo=timezone.utc)  # nodes created 00:00
    text, _ = build_node_catalog([], brain, extra_ids=_EXTRA,
                                 view_policy=True, now=now)
    # relative fine header + ownership mark (both ids are session-encoded)
    assert '(id:newnode1 · 3h ago · this session)' in text
    # relative mode suppresses the duplicate absolute `Created:` line
    assert 'Created: 2026-08-01' not in text
    # flag off: absolute date, doubled render, no ownership mark — unchanged
    off, _ = build_node_catalog([], brain, extra_ids=_EXTRA, view_policy=False)
    assert '(id:newnode1, 2026-08-01)' in off
    assert 'Created: 2026-08-01' in off
    assert 'this session' not in off


def test_catalog_ownership_mark_excludes_read_only_ids():
    # a node only READ this session (recalled) gets no ownership mark
    brain = _CatalogBrain({'readnode1': _node('readnode1')})
    extra = {'encoded': set(), 'authored': set(), 'recalled': {'readnode1'},
             'stops': {'readnode1': 15}, 'run_stops': [5, 10, 15]}
    text, _ = build_node_catalog([], brain, extra_ids=extra, view_policy=True)
    assert 'readnode1' in text
    assert 'this session' not in text


def test_catalog_surfaced_ids_protected_from_aging():
    # oldnode1 is old by stop but surfaced for the CURRENT window → full body.
    judge = ['picked id:oldnode1 for this turn']
    brain = _CatalogBrain({'oldnode1': _node('oldnode1'),
                           'newnode1': _node('newnode1')})
    text, _ = build_node_catalog(judge, brain, extra_ids=_EXTRA,
                                 view_policy=True)
    assert 'not shown — get_nodes' not in text     # nothing left to age
    assert text.count('Edges:') == 2


def test_hidden_tools_and_split_stay_in_sync_with_substrate():
    # The split keys must exist in the link contract; the hidden set must never
    # cover the edge tools ("connect has no provenance home, by contract"), and
    # every hidden lookup tool must have an accumulator route (else the drop
    # loses information instead of moving it).
    from servers.scales.s1.trace_links import nodes_for_traces
    from servers.daemon_server import BrainDaemon
    link = nodes_for_traces([], [], [{'id': 'x', 'chain_id': 's0-a-1'}])['x']
    for _label, keys, _titles in PROVENANCE_SPLIT:
        for key in keys:
            assert key in link
    non_full = DROPPED_ACTION_TOOLS | STUBBED_ACTION_TOOLS
    assert not {'connect', 'disconnect', 'revise_edge'} & non_full
    assert not DROPPED_ACTION_TOOLS & STUBBED_ACTION_TOOLS   # modes disjoint
    write_ops = {'remember', 'remember_batch', 'revise', 'revise_batch',
                 'brain_batch'}                      # provenance via `affected`
    for tool in non_full - write_ops:                # lookups need _LOOKUP_KEY
        assert tool in BrainDaemon._LOOKUP_KEY, tool
