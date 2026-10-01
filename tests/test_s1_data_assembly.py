"""Tests for S1 encode data assembly functions.

Tests the pure data assembly layer that prepares input for the S1 encoder LLM:
- _gather_messages: reads S0 traces into the format expected by the encoder
- build_node_catalog: extracts and formats nodes from surface outputs

No LLM calls — these are all deterministic data transformations.

Run: python3 -m pytest tests/test_s1_data_assembly.py -v
"""
import pytest
from datetime import datetime, timezone
from tests.isolated_brain import SeedBrain


def _insert_node(conn, node_id, node_type='rule', title='Test node', content='Test content.'):
    """Insert a synthetic node under the caller's id.

    Most callers pass a plain 8-hex id; test_regex_handles_typed_prefix_ids
    deliberately passes a typed-prefix one, so the id shape is the caller's
    choice rather than a promise this helper makes.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO nodes (id, type, title, content, confidence, locked, archived, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 0.9, 0, 0, ?, ?)",
        (node_id, node_type, title, content, now, now))
    conn.commit()


class TestGatherMessages:
    """Tests for _gather_messages(brain, session_id) — trace-based message retrieval."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with SeedBrain() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            # No trace clearing: a SeedBrain's logs DB is freshly built and
            # holds zero trace_events, and every test here writes and reads
            # its own session_id. The old DELETE existed because IsolatedBrain
            # cloned production's traces.
            yield

    def test_empty_session_returns_empty(self):
        """No traces for session -> empty list."""
        from servers.scales.s1.encode import _gather_messages
        result = _gather_messages(self.brain, 'nonexistent-session-xyz')
        assert result == []

    def test_returns_turns_from_s0_traces(self):
        """S0 user+assistant traces -> list of turn dicts with correct shape."""
        from servers.scales.s1.encode import _gather_messages
        session_id = 'test-gather-001'

        # Write S0 traces: a user message and an assistant response
        chain = 's0-testgath-1'
        self.dal.append(
            chain_id=chain, scale='s0', event_type='K',
            ref_type='user_message', ref_id='',
            summary='How does recall work?',
            metadata={'content': 'How does recall work?', 'recall_chain': 's1r-testgath-1'},
            session_id=session_id)
        self.dal.append(
            chain_id=chain, scale='s0', event_type='delta',
            ref_type='assistant_message', ref_id='',
            summary='Recall uses embeddings...',
            metadata={'content': 'Recall uses embeddings to find similar nodes.'},
            session_id=session_id)
        self.brain.logs_conn.commit()

        result = _gather_messages(self.brain, session_id)
        assert len(result) == 2

        user_turn = result[0]
        assert user_turn['role'] == 'user'
        assert 'recall' in user_turn['content'].lower()
        assert user_turn['trace_id']  # the reference source_refs copy

        asst_turn = result[1]
        assert asst_turn['role'] == 'assistant'
        assert 'embeddings' in asst_turn['content'].lower()

    def test_respects_max_messages_limit(self):
        """When more turns exist than max_messages, only the most recent are returned."""
        from servers.scales.s1.encode import _gather_messages
        from servers.scales.s1.encode_contract import ENCODING_AGENT

        session_id = 'test-gather-limit'
        max_msgs = ENCODING_AGENT['max_messages']

        # Write more turns than the limit (each turn = user + assistant = 2 messages)
        num_turns = (max_msgs // 2) + 5
        for i in range(num_turns):
            chain = 's0-testlim-%d' % i
            self.dal.append(
                chain_id=chain, scale='s0', event_type='K',
                ref_type='user_message', ref_id='',
                summary='Question %d' % i,
                metadata={'content': 'Question number %d about the brain' % i},
                session_id=session_id)
            self.dal.append(
                chain_id=chain, scale='s0', event_type='delta',
                ref_type='assistant_message', ref_id='',
                summary='Answer %d' % i,
                metadata={'content': 'Answer to question %d' % i},
                session_id=session_id)
        self.brain.logs_conn.commit()

        result = _gather_messages(self.brain, session_id)
        assert len(result) <= max_msgs, (
            "Should respect max_messages=%d, got %d" % (max_msgs, len(result)))

    def test_content_preserves_whole_stored_episode(self):
        """Gathering must not discard the end of a stored episode."""
        from servers.scales.s1.encode import _gather_messages

        session_id = 'test-gather-trunc'
        long_content = 'x' * 7990 + 'TAIL_PROOF'

        chain = 's0-testtrn-1'
        self.dal.append(
            chain_id=chain, scale='s0', event_type='K',
            ref_type='user_message', ref_id='',
            summary='Long message',
            metadata={'content': long_content},
            session_id=session_id)
        self.brain.logs_conn.commit()

        result = _gather_messages(self.brain, session_id)
        assert len(result) >= 1
        assert result[0]['content'] == long_content


class TestBuildNodeCatalog:
    """Tests for build_node_catalog(surface_outputs, brain) — catalog from surface selections."""

    NODE_A = 'aa11bb22'
    NODE_B = 'cc33dd44'
    NODE_C = 'ee55ff66'
    NODE_TYPED = 'con_ab12'   # typed-prefix id, not plain hex

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with SeedBrain() as env:
            self.brain = env.brain
            self.conn = env.brain.conn
            # Insert synthetic nodes with hex IDs (matches the regex in build_node_catalog)
            _insert_node(self.conn, self.NODE_A, 'rule', 'Test catalog rule',
                             'This rule governs recall behavior.')
            _insert_node(self.conn, self.NODE_B, 'lesson', 'Test catalog lesson',
                             'Learned from encoding experiments.')
            _insert_node(self.conn, self.NODE_TYPED, 'concept', 'Typed prefix node',
                             'Carries a con_ prefix id rather than plain hex.')
            yield

    def test_empty_surface_outputs(self):
        """No surface outputs -> empty catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog
        text, ids = build_node_catalog([], self.brain)
        assert text == ''
        assert ids == set()

    def test_none_and_no_selection_skipped(self):
        """None and '(no selection)' entries are ignored."""
        from servers.scales.s1.encode_contract import build_node_catalog
        text, ids = build_node_catalog([None, '(no selection)', None], self.brain)
        assert text == ''
        assert ids == set()

    def test_extracts_node_ids_from_surface_output(self):
        """Extracts id:XXXXXXXX patterns from surface output strings."""
        from servers.scales.s1.encode_contract import build_node_catalog

        surface_output = '[rule] "Test catalog rule" (id:%s, conf:0.9)' % self.NODE_A
        text, ids = build_node_catalog([surface_output], self.brain)

        assert self.NODE_A in ids
        assert 'Node Catalog' in text

    def test_deduplicates_across_outputs(self):
        """Same node ID in multiple surface outputs appears only once in catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog

        so1 = '[rule] "Title A" (id:%s, conf:0.9)' % self.NODE_A
        so2 = '[lesson] "Title B" (id:%s, conf:0.8)' % self.NODE_A

        text, ids = build_node_catalog([so1, so2], self.brain)
        assert len(ids) == 1
        assert self.NODE_A in ids

    def test_multiple_distinct_ids(self):
        """Multiple different node IDs from different surface outputs are all extracted."""
        from servers.scales.s1.encode_contract import build_node_catalog

        so1 = '[rule] "First" (id:%s)' % self.NODE_A
        so2 = '[lesson] "Second" (id:%s)' % self.NODE_B

        text, ids = build_node_catalog([so1, so2], self.brain)
        assert self.NODE_A in ids
        assert self.NODE_B in ids
        assert len(ids) == 2

    def test_uses_format_node_for_rich_display(self):
        """Catalog entries use format_node, which includes type/title/content/edges."""
        from servers.scales.s1.encode_contract import build_node_catalog

        surface_output = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog([surface_output], self.brain)

        assert self.NODE_A in ids
        # format_node includes the node type in brackets and title
        assert '[rule]' in text
        assert 'Test catalog rule' in text

    def test_includes_correction_annotations(self):
        """Nodes with corrected_by edges get Updated by annotation in catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog

        # Create a correcting node and a `corrects` edge.
        # correction_enrich walks correction_improvement-aspect edges
        # (corrects, supersedes, reframes, ...), so we set up via a typed
        # edge rather than the legacy correction_of metadata field.
        _insert_node(self.conn, self.NODE_C, 'rule', 'Updated version of rule',
                         'New improved content.')
        # Edge: NODE_C corrects NODE_A
        self.brain.connect_typed(
            source_id=self.NODE_C, target_id=self.NODE_A,
            relation='corrects', weight=0.5,
            description='Test setup: NODE_C supersedes NODE_A',
            encoding_source='test:correction_annotation')

        judge_output = '[rule] "Test" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog([judge_output], self.brain)

        assert self.NODE_A in ids
        assert '⚠ Corrected by' in text              # the `corrects` flavor

    def test_nonexistent_id_excluded_from_formatted(self):
        """IDs extracted from surface output but not in DB are not in formatted_ids."""
        from servers.scales.s1.encode_contract import build_node_catalog

        fake_id = 'deadbeef'
        surface_output = '[rule] "Ghost" (id:%s)' % fake_id
        text, ids = build_node_catalog([surface_output], self.brain)

        # The ID is extracted by regex but format_node returns None -> not in formatted_ids
        assert fake_id not in ids

    def test_regex_handles_typed_prefix_ids(self):
        """Typed prefix IDs (con_xxxx) must be extracted by build_node_catalog."""
        from servers.scales.s1.encode_contract import build_node_catalog

        # NODE_TYPED is seeded by the fixture. This used to read an arbitrary
        # `LIMIT 1` row and skip when it came back plain hex, so whether the
        # regex was tested at all depended on unspecified row order.
        node_id = self.NODE_TYPED
        surface_output = '[rule] "Test" (id:%s, conf:0.9)' % node_id
        text, ids = build_node_catalog([surface_output], self.brain)

        assert node_id in ids, (
            "Regex should match typed-prefix ID %s" % node_id)

    # ── Piece 3: the widened catalog (surfaced ∪ encoded ∪ authored ∪ recalled) ──

    def test_extra_ids_fold_in_tagged(self):
        """authored/recalled/encoded ids fold into the catalog, each tagged."""
        from servers.scales.s1.encode_contract import build_node_catalog
        _insert_node(self.conn, self.NODE_C, 'note', 'Recalled node', 'looked up')
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A  # surfaced
        text, ids = build_node_catalog(
            [judge], self.brain,
            extra_ids={'authored': {self.NODE_B}, 'recalled': {self.NODE_C},
                       'encoded': set()})
        assert {self.NODE_A, self.NODE_B, self.NODE_C} <= ids   # union
        assert '[anchor-authored]' in text and '[anchor-recalled]' in text

    def test_priority_authored_beats_surfaced(self):
        """A node both surfaced AND authored gets the higher-signal authored tag."""
        from servers.scales.s1.encode_contract import build_node_catalog
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog(
            [judge], self.brain, extra_ids={'authored': {self.NODE_A}})
        assert self.NODE_A in ids
        assert '[anchor-authored]' in text          # tagged, not left untagged

    def test_flag_off_surfaced_only_unchanged(self):
        """No extra_ids → no tags, legacy header (the A/B control arm)."""
        from servers.scales.s1.encode_contract import build_node_catalog
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, _ = build_node_catalog([judge], self.brain)
        assert '[anchor-' not in text and '[encoded]' not in text
        assert 'surfaced this session' in text      # legacy header preserved

    def test_community_node_skipped_even_when_in_extra(self):
        """A community node folded via extra_ids is still skipped (S2CE owns it)."""
        from servers.scales.s1.encode_contract import build_node_catalog
        _insert_node(self.conn, self.NODE_C, 'community', 'A community', 'narrative')
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A
        text, ids = build_node_catalog(
            [judge], self.brain, extra_ids={'encoded': {self.NODE_C}})
        assert self.NODE_A in ids
        assert self.NODE_C not in ids               # community filtered out

    def test_noise_relations_excluded_on_every_arm(self):
        """Noise-aspect relations (community_member, co_anchored, ...) never
        reach the catalog — the exclusion is get_node's, so the encoder has no
        filter of its own and the control arm (extra_ids=None) is as clean as
        the lived one. Semantic relations survive."""
        from servers.scales.s1.encode_contract import build_node_catalog
        from servers.dal_graph import GraphDAL
        # sanity: the aspect registry classifies community_member as noise
        assert 'community_member' in self.brain.aspects.structural_exclusions
        gdal = GraphDAL(self.conn)
        gdal.add_relation(self.NODE_A, self.NODE_B, 'community_member',
                          description='structural placement edge')
        gdal.add_relation(self.NODE_A, self.NODE_B, 'extends',
                          description='rule extends the lesson semantically')
        judge = '[rule] "Test catalog rule" (id:%s)' % self.NODE_A

        for extra_ids in ({}, None):
            text, _ = build_node_catalog([judge], self.brain, extra_ids=extra_ids)
            assert 'community_member' not in text        # noise dropped
            assert 'extends' in text                     # semantic survives


class _StubRecallBrain:
    """Minimal brain for _associated_stub_ids: canned episodes + recorded
    recalls. No trace DAL — the test passes empty streams so _turn_links
    joins nothing (every turn reads unencoded, the probe's convention)."""

    def __init__(self, episodes, results_by_query):
        self._episodes = episodes
        self._results = results_by_query
        self.recall_calls = []

    def recall_episodes(self, **kw):
        return {'episodes': list(self._episodes)}

    def recall(self, query, **kw):
        self.recall_calls.append((query, kw))
        return {'results': list(self._results.get(query, []))}


def _ep(eid, ref_type, content, created_at):
    return {'id': eid, 'ref_type': ref_type, 'chain_id': 's0-testsess-1',
            'created_at': created_at, 'summary': content[:200],
            'metadata': {'content': content}}


_EMPTY_STREAMS = {'surface': [], 'encode': [], 'touched': []}
_DATED_MSGS = [
    {'role': 'user', 'content': 'alpha question'},
    {'role': 'user', 'content': 'beta question'},
]


class TestAssociatedStubIds:
    """_associated_stub_ids — the subconscious retrieval (turnmax over
    per-message production recall, catalog excluded)."""

    def _episodes(self):
        return [
            _ep('u1', 'user_message', 'alpha question', '2026-08-21T10:00:00+00:00'),
            _ep('a1', 'assistant_message', 'alpha answer', '2026-08-21T10:01:00+00:00'),
            _ep('u2', 'user_message', 'beta question', '2026-08-21T10:02:00+00:00'),
        ]

    def test_turnmax_exclusion_and_ordering(self):
        from servers.scales.s1.encode import _associated_stub_ids
        brain = _StubRecallBrain(self._episodes(), {
            'alpha question': [
                {'id': 'aaaa1111', 'type': 'finding', 'effective_activation': 0.9},
                {'id': 'cata0001', 'type': 'rule', 'effective_activation': 0.95},
            ],
            # same node again, lower score — turnmax keeps 0.9, never sums
            'alpha answer': [
                {'id': 'aaaa1111', 'type': 'finding', 'effective_activation': 0.5},
            ],
            'beta question': [
                {'id': 'bbbb2222', 'type': 'lesson', 'effective_activation': 0.7},
            ],
        })
        out = _associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                   {'cata0001'}, streams=_EMPTY_STREAMS)
        # catalog id filtered; rank order by best single match
        assert out == ['aaaa1111', 'bbbb2222']
        # both voices of both turns seeded
        assert [q for q, _kw in brain.recall_calls] == [
            'alpha question', 'alpha answer', 'beta question']
        for _q, kw in brain.recall_calls:
            # pure read
            assert kw['mark_accessed'] is False
            assert kw['session_id'] == 'testsess'
            # NO as_of, deliberately — every candidate value breaks a
            # verified case (operator-TZ skew live, empty universe in
            # replays, laf_v1 coupling); see the function docstring
            assert kw.get('as_of') is None

    def test_seed_cap_keeps_newest(self):
        from servers.scales.s1.encode import _associated_stub_ids
        from servers.scales.s1.encoder_view import ASSOCIATED_SEED_CAP
        # more distinct user turns than the cap — each its own seed text
        n = ASSOCIATED_SEED_CAP + 5
        eps = [_ep('u%d' % i, 'user_message', 'question number %d' % i,
                   '2026-08-21T10:%02d:00+00:00' % i) for i in range(n)]
        brain = _StubRecallBrain(eps, {})
        msgs = [{'role': 'user', 'content': '[Current date: 2026-08-21] q'}] * n
        _associated_stub_ids(brain, msgs, 'testsess', {'x'},
                             streams=_EMPTY_STREAMS)
        queries = [q for q, _kw in brain.recall_calls]
        assert len(queries) == ASSOCIATED_SEED_CAP
        # the NEWEST seeds survive the cap (tail turns are the working material)
        assert queries[-1] == 'question number %d' % (n - 1)
        assert queries[0] == 'question number %d' % (n - ASSOCIATED_SEED_CAP)

    def test_duplicate_seeds_recalled_once(self):
        from servers.scales.s1.encode import _associated_stub_ids
        eps = [_ep('u%d' % i, 'user_message', 'ok',
                   '2026-08-21T10:%02d:00+00:00' % i) for i in range(3)]
        brain = _StubRecallBrain(eps, {})
        msgs = [{'role': 'user', 'content': 'ok'}] * 3
        _associated_stub_ids(brain, msgs, 'testsess', {'x'},
                             streams=_EMPTY_STREAMS)
        assert [q for q, _kw in brain.recall_calls] == ['ok']

    def test_k_cap(self):
        from servers.scales.s1.encode import _associated_stub_ids
        from servers.scales.s1.encoder_view import ASSOCIATED_STUBS_K
        hits = [{'id': 'node%04d' % i, 'type': 'fact',
                 'effective_activation': 1.0 - i * 0.01} for i in range(9)]
        brain = _StubRecallBrain(self._episodes(), {'alpha question': hits})
        out = _associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                   {'unrelated'}, streams=_EMPTY_STREAMS)
        assert len(out) == ASSOCIATED_STUBS_K
        assert out == ['node%04d' % i for i in range(ASSOCIATED_STUBS_K)]

    def test_covered_turns_not_seeded(self, monkeypatch):
        from servers.scales.s1 import encode
        brain = _StubRecallBrain(self._episodes(), {
            'beta question': [
                {'id': 'bbbb2222', 'type': 'lesson', 'effective_activation': 0.7}],
        })
        # turn 1 (u1) already encoded by a prior run; turn 2 (u2) is not
        monkeypatch.setattr(encode, '_turn_links', lambda *a, **kw: (
            {'u1': {'surfaced': [], 'encoded': [], 'encoded_by': 'run-1'},
             'u2': {'surfaced': [], 'encoded': [], 'encoded_by': None}}, {}))
        out = encode._associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                          {'unrelated'}, streams=_EMPTY_STREAMS)
        assert [q for q, _kw in brain.recall_calls] == ['beta question']
        assert out == ['bbbb2222']

    def test_no_seeds_returns_empty_without_recall(self):
        from servers.scales.s1.encode import _associated_stub_ids
        brain = _StubRecallBrain([], {})
        out = _associated_stub_ids(brain, _DATED_MSGS, 'testsess',
                                   {'x'}, streams=_EMPTY_STREAMS)
        assert out == []
        assert brain.recall_calls == []


class TestAssociatedStubsRender:
    """build_node_catalog's associated_ids arm + the _build_catalog gate."""

    NODE_A = 'aa11bb22'
    NODE_B = 'cc33dd44'
    NODE_C = 'ee55ff66'

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with SeedBrain() as env:
            self.brain = env.brain
            self.conn = env.brain.conn
            _insert_node(self.conn, self.NODE_A, 'rule', 'Surfaced rule',
                             'This rule was surfaced.')
            _insert_node(self.conn, self.NODE_B, 'lesson', 'Associated lesson',
                             'Recall ranked this near the window.')
            _insert_node(self.conn, self.NODE_C, 'finding', 'Associated finding',
                             'Second subconscious entry.')
            yield

    def _judge(self):
        return '[rule] "Surfaced rule" (id:%s, conf:0.9)' % self.NODE_A

    def test_associated_render_last_tagged_and_in_ids(self):
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG, build_node_catalog
        text, ids = build_node_catalog(
            [self._judge()], self.brain, extra_ids={},
            associated_ids=[self.NODE_B, self.NODE_C])
        assert {self.NODE_A, self.NODE_B, self.NODE_C} <= ids
        assert ASSOCIATED_TAG in text
        # stubs are the catalog's LAST entries, rank order preserved
        assert text.index('Surfaced rule') < text.index('Associated lesson')
        assert text.index('Associated lesson') < text.index('Associated finding')
        assert text.splitlines()[0] == 'Node Catalog (3 nodes)'
        # tag sits on the stub entries only
        for line in text.splitlines():
            if line.startswith(ASSOCIATED_TAG):
                assert 'Associated' in line
            if 'Surfaced rule' in line:
                assert ASSOCIATED_TAG not in line

    def test_associated_none_and_empty_byte_identical(self):
        from servers.scales.s1.encode_contract import build_node_catalog
        base_text, base_ids = build_node_catalog(
            [self._judge()], self.brain, extra_ids={})
        for assoc in (None, []):
            text, ids = build_node_catalog(
                [self._judge()], self.brain, extra_ids={}, associated_ids=assoc)
            assert text == base_text
            assert ids == base_ids

    def test_associated_community_never_renders(self):
        # the render is the chokepoint for 'S1E never gets community bodies' —
        # it must hold for ANY caller of build_node_catalog, not just the
        # production retrieval (whose recall already type-excludes them)
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG, build_node_catalog
        comm_id = 'dd77ee88'
        _insert_node(self.conn, comm_id, 'community', 'A community',
                         'Community body that must never render.')
        text, ids = build_node_catalog(
            [self._judge()], self.brain, extra_ids={},
            associated_ids=[comm_id, self.NODE_B])
        assert comm_id not in ids
        assert 'A community' not in text
        assert self.NODE_B in ids
        # header counts only what can render (1 surfaced + 1 stub)
        assert text.splitlines()[0] == 'Node Catalog (2 nodes)'
        assert ASSOCIATED_TAG in text

    def test_associated_dupe_of_catalog_dropped(self):
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG, build_node_catalog
        text, ids = build_node_catalog(
            [self._judge()], self.brain, extra_ids={},
            associated_ids=[self.NODE_A])
        # already cataloged: renders once, untagged, not double-counted —
        # and with no stub actually rendering, the surfaced-only header stands
        assert text.count('Surfaced rule') == 1
        assert ASSOCIATED_TAG not in text
        assert text.splitlines()[0] == 'Node Catalog (1 nodes surfaced this session)'

    def test_build_catalog_gate_flag_off_no_retrieval(self, monkeypatch):
        from servers.scales.s1 import encode
        monkeypatch.delenv('BRAIN_S1E_ASSOCIATED_STUBS', raising=False)
        monkeypatch.setattr(encode, '_associated_stub_ids',
                            lambda *a, **kw: pytest.fail('retrieval ran with flag off'))
        messages = [{'role': 'user', 'content': 'hi', 'judge_output': self._judge()}]
        text, ids, _streams = encode._build_catalog(
            self.brain, messages, 'assoc-gate-off')
        assert self.NODE_A in ids

    def test_build_catalog_gate_flag_on_renders_stubs(self, monkeypatch):
        from servers.scales.s1 import encode
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG
        monkeypatch.setenv('BRAIN_S1E_ASSOCIATED_STUBS', '1')
        seen = {}

        def fake_retrieval(brain, messages, session_id, exclude_ids,
                           streams=None, turns=None):
            seen['exclude'] = set(exclude_ids)
            return [self.NODE_B]

        monkeypatch.setattr(encode, '_associated_stub_ids', fake_retrieval)
        messages = [{'role': 'user', 'content': 'hi', 'judge_output': self._judge()}]
        text, ids, _streams = encode._build_catalog(
            self.brain, messages, 'assoc-gate-on')
        # exclusion set is exactly what the catalog will show
        assert self.NODE_A in seen['exclude']
        assert ASSOCIATED_TAG in text
        assert self.NODE_B in ids

    def test_build_catalog_retrieval_failure_degrades_loud(self, monkeypatch):
        from servers.scales.s1 import encode
        monkeypatch.setenv('BRAIN_S1E_ASSOCIATED_STUBS', '1')

        def boom(*a, **kw):
            raise RuntimeError('recall exploded')

        monkeypatch.setattr(encode, '_associated_stub_ids', boom)
        messages = [{'role': 'user', 'content': 'hi', 'judge_output': self._judge()}]
        text, ids, _streams = encode._build_catalog(
            self.brain, messages, 'assoc-gate-err')
        # catalog still renders, stubs absent
        assert self.NODE_A in ids
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG
        assert ASSOCIATED_TAG not in text

    def _run_with_stubs_on(self, template, monkeypatch):
        """run_encoding with the stubs flag on and the LLM loop stubbed.
        The flag reads True exactly once — a second env read raises
        StopIteration, so the check and the render share one resolution.
        Returns (logged error sources, the user content sent)."""
        from unittest import mock
        from servers.scales.s1 import encode, encoder_view
        from tests.interaction_override import interaction_override
        monkeypatch.setattr(encode, '_associated_stub_ids',
                            lambda *a, **kw: [self.NODE_B])
        messages = [{'role': 'user', 'content': 'hi', 'trace_id': 'aaaaaaaa',
                     'judge_output': self._judge()},
                    {'role': 'assistant', 'content': 'hello', 'trace_id': 'bbbbbbbb'}]
        loop_result = {'rounds': 1, 'actions': 0, 'write_actions': 0,
                       'action_details': [], 'read_calls': [], 'final_text': 'done',
                       'elapsed_ms': 1, 'input_tokens': 0, 'output_tokens': 0,
                       'cache_read_tokens': 0, 'cache_creation_tokens': 0,
                       'model': 'claude-test'}
        sent = {}

        def loop(**kw):
            sent.update(kw)
            return dict(loop_result)

        with interaction_override(self.brain, 's1e', template=template), \
                mock.patch.object(encoder_view, 'associated_stubs_enabled',
                                  side_effect=[True]), \
                mock.patch.object(encode, '_gather_messages', return_value=messages), \
                mock.patch('servers.scales.runner.make_client', return_value=object()), \
                mock.patch.object(encode, 'run_llm_loop', side_effect=loop), \
                mock.patch.object(self.brain, '_log_error',
                                  wraps=self.brain._log_error) as spy:
            encode.run_encoding(self.brain, lambda cmd, args: {'ok': True}, 1,
                                'assoc-untaught')
        return [c.args[0] for c in spy.call_args_list], sent['user_content']

    def test_run_warns_when_the_prompt_does_not_teach_the_tag(self, monkeypatch):
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG
        logged, body = self._run_with_stubs_on('A prompt without the tag.', monkeypatch)
        assert logged.count('s1e_associated_stubs_untaught') == 1
        assert ASSOCIATED_TAG in body   # the one flag read reached the render

    def test_run_is_quiet_when_the_prompt_teaches_the_tag(self, monkeypatch):
        from servers.scales.s1.encode_contract import ASSOCIATED_TAG
        logged, body = self._run_with_stubs_on(
            'Entries tagged %s come from recall.' % ASSOCIATED_TAG, monkeypatch)
        assert 's1e_associated_stubs_untaught' not in logged
        assert ASSOCIATED_TAG in body
