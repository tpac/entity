"""Tests for NodeDAL."""

from tests.brain_test_base import BrainTestBase
from servers.dal import NodeDAL


class TestNodeDALGetTitle(BrainTestBase):
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.dal = NodeDAL(self.brain.conn)
        self.node = self.brain.remember(
            type='rule', title='Unique title for get_title',
            content='Content for get_title tests')
        self.node_id = self.node['id']

    def test_exact_match(self):
        result = self.dal.get_title(self.node_id)
        self.assertEqual(result, 'Unique title for get_title')

    def test_no_match_returns_none(self):
        result = self.dal.get_title('zzz_nonexistent')
        self.assertIsNone(result)

    def test_prefix_does_not_match(self):
        # Exact-id contract: a truncated id is a miss, never a prefix bind.
        result = self.dal.get_title(self.node_id[:6])
        self.assertIsNone(result)


class TestNodeDALFilterTextOps(BrainTestBase):
    """contains / prefix operators on filter_nodes — structural + kv, escaped.

    'Zqx' is a nonsense marker so distinctive tokens never collide with seed
    nodes the fresh brain ships with.
    """
    needs_embedder = False

    def setUp(self):
        super().setUp()
        self.dal = NodeDAL(self.brain.conn)
        self.a = self.brain.remember(
            type='zqxalpha', title='Zqx crossing guide',
            content='c', situation='When Zqx debugging the daemon')['id']
        self.b = self.brain.remember(
            type='zqxbeta', title='Zqx print pattern',
            content='c', situation='When printing')['id']
        self.c = self.brain.remember(
            type='zqxother', title='Giraffe notes',
            content='c', situation='never')['id']

    def _ids(self, res):
        return {n['id'] for n in res['nodes']}

    def test_contains_structural_title(self):
        res = self.dal.filter_nodes(field='title', contains='Zqx', limit=50)
        ids = self._ids(res)
        self.assertIn(self.a, ids)
        self.assertIn(self.b, ids)
        self.assertNotIn(self.c, ids)
        self.assertEqual(res['total_count'], 2)  # unclamped, filter-scoped

    def test_prefix_structural_type(self):
        res = self.dal.filter_nodes(field='type', prefix='zqxa', limit=50)
        self.assertEqual(self._ids(res), {self.a})

    def test_prefix_is_anchored_not_substring(self):
        # 'beta' is a suffix of 'zqxbeta' — prefix must NOT match mid-string.
        res = self.dal.filter_nodes(field='type', prefix='beta', limit=50)
        self.assertEqual(self._ids(res), set())

    def test_contains_escapes_underscore(self):
        lit = self.brain.remember(type='zqxesc', title='Zqx a_b literal',
                                  content='c')['id']
        wild = self.brain.remember(type='zqxesc', title='Zqx axb other',
                                   content='c')['id']
        res = self.dal.filter_nodes(field='title', contains='a_b', limit=50)
        ids = self._ids(res)
        self.assertIn(lit, ids)
        # '_' escaped → literal underscore, not a single-char wildcard.
        self.assertNotIn(wild, ids)

    def test_contains_kv_situation(self):
        res = self.dal.filter_nodes(field='situation',
                                    contains='Zqx debugging', limit=50)
        self.assertEqual(self._ids(res), {self.a})

    def test_prefix_kv_situation(self):
        res = self.dal.filter_nodes(field='situation',
                                    prefix='When Zqx', limit=50)
        self.assertEqual(self._ids(res), {self.a})

    def test_wrapper_threads_prefix_to_dal(self):
        # brain.filter_nodes (brain_recall wrapper) must thread the operator
        # through to the DAL — the wiring the MCP dispatch relies on.
        res = self.brain.filter_nodes(field='type', prefix='zqx',
                                      rich=False, limit=50)
        self.assertEqual(self._ids(res), {self.a, self.b, self.c})


class TestNodeDALFilterHonestLimit(BrainTestBase):
    """limit=None → all matches; numeric → honest page; no silent 200 ceiling.

    The read is unbounded so an internal id-set scan (every s2:* node) gets
    every row; the agent-facing cap lives at the dispatch door, not here.
    """
    needs_embedder = False

    def test_unbounded_past_the_old_200_ceiling(self):
        dal = NodeDAL(self.brain.conn)
        for i in range(205):
            self.brain.remember(type='zqxbig', title='Zqxbig %d' % i, content='c')

        allres = dal.filter_nodes(field='type', include=['zqxbig'], limit=None)
        self.assertEqual(len(allres['nodes']), 205)   # None → all, no clamp
        self.assertEqual(allres['total_count'], 205)

        big = dal.filter_nodes(field='type', include=['zqxbig'], limit=250)
        self.assertEqual(len(big['nodes']), 205)      # numeric > 200 honored

        page = dal.filter_nodes(field='type', include=['zqxbig'], limit=50)
        self.assertEqual(len(page['nodes']), 50)      # honest page
        self.assertEqual(page['total_count'], 205)    # exact, unclamped

    def test_wrapper_limit_none_returns_all(self):
        # The brain wrapper must survive limit=None (no *mult / *2 / >None math).
        for i in range(5):
            self.brain.remember(type='zqxnone', title='Zqxnone %d' % i, content='c')
        res = self.brain.filter_nodes(field='type', include=['zqxnone'],
                                      rich=False, limit=None)
        self.assertEqual(len(res['nodes']), 5)
        self.assertNotIn('truncated', res)            # asked for all, got all
