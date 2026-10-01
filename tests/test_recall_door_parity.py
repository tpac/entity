"""Guard: every recall door hands back the same canonical shape.

Recall has more than one door — MCP by-query, MCP by-id, MCP batch, and the
in-brain hook path — and for months they silently disagreed. `recall_node`
read the bare DB row, so `recall({node_id})` returned a node with NO
corrections, NO situation and NO connections: a superseded claim handed over
with its correction marker stripped off, while the by-query door on the same
node returned all three. `recall_batch` had the same hole.

The fix was one door — `Brain.canonicalize_results` — that every shape routes
through. These tests hold that line, because the failure mode is invisible
from any single door: each one looks fine on its own, and only a comparison
shows the drift.

Two ways the drift comes back, one test each:

1. A NEW door is added that calls brain.recall* directly and skips the
   canonical pull. Caught by `test_recall_producing_handlers_are_known` — a
   tripwire on the set of dispatch handlers that call recall directly, across
   every `servers/dispatch_*.py`. It fails when someone adds one, which is the
   moment to add it to the parity test below. A door that instead DELEGATES to
   `_handle_recall` (as `_handle_recall_batch` does) inherits the canonical
   pull through the single door — it stays out of this set and is held by its
   own behavioural parity case, not this tripwire.

2. `get_node` GROWS an attachment that CANONICAL_ATTACHMENT_KEYS doesn't
   list. Then the doors diverge again on the new field, silently, because the
   overlay only copies what the list names. Caught by
   `test_canonical_keys_cover_what_get_node_attaches`.
"""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from servers import dispatch_read
from servers.contract import CANONICAL_ATTACHMENT_KEYS
from servers.dispatch_common import CALLER_SESSION_KEY
from tests.brain_test_base import BrainTestBase

# Dispatch handlers that call brain.recall*/recall_node DIRECTLY — the one
# place the canonical pull is applied. _handle_recall_batch is deliberately
# NOT here: it delegates to _handle_recall per query rather than calling
# brain.recall itself, so it inherits canonicalization through the single door
# and is verified behaviourally by test_batch_door_matches_the_canonical_pull
# below. A NEW handler that calls brain.recall* directly must be added here
# (and given its own parity case) — that addition is the regression this set
# exists to make loud.
KNOWN_RECALL_DOORS = {'_handle_recall'}

# Node-returning recall entry points. `recall_episodes` is deliberately absent:
# it returns trace episodes, not nodes, so it has no canonical node shape to
# agree with.
RECALL_CALLS = ('recall', 'recall_node', 'recall_batch')

SESSION = 'sess-parity'


def _handlers_producing_recall_results():
    """Every dispatch handler whose body calls brain.recall / recall_node.

    Scans ALL servers/dispatch_*.py modules, not just dispatch_read — a door
    added in a sibling module is exactly the case this guard exists to catch,
    and scoping the scan to one file would let it through while the suite
    reported green.
    """
    found = set()
    for path in sorted((REPO / 'servers').glob('dispatch_*.py')):
        tree = ast.parse(path.read_text())
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)]:
            for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                if getattr(call.func, 'attr', None) in RECALL_CALLS:
                    found.add(fn.name)
    return found


def test_recall_producing_handlers_are_known():
    """A new recall door must be added to the parity test, deliberately."""
    found = _handlers_producing_recall_results()
    assert found == KNOWN_RECALL_DOORS, (
        'handlers calling brain.recall* directly changed: %s. A new direct '
        'door must route through Brain.canonicalize_results and get its own '
        '*_door_matches_the_canonical_pull case in TestRecallDoorParity '
        'below, then be added to KNOWN_RECALL_DOORS. (A handler that instead '
        'delegates to _handle_recall inherits the pull and stays out of this '
        'set.)'
        % sorted(found ^ KNOWN_RECALL_DOORS))


class TestRecallDoorParity(BrainTestBase):
    """Behavioural parity, per door, against the canonical pull."""

    def setUp(self):
        super().setUp()
        # A node worth comparing: it carries a situation, an incoming
        # correction and an edge, so all the canonical attachments are
        # non-empty and a dropped one is visible.
        self.claim = self.brain.remember(
            type='finding',
            title='Zarquon telemetry drifted during the eclipse window',
            content='The zarquon counter disagreed with the eclipse log.',
            situation='When reconciling zarquon telemetry against eclipse logs',
        )['id']
        self.fix = self.brain.remember(
            type='correction',
            title='Zarquon drift was the clock, not the counter',
            content='The counter was right; the clock had skewed.',
        )['id']
        self.brain.connect_typed(
            self.fix, self.claim, relation='corrects',
            description='the skew explains the whole disagreement')
        self.canonical = self.brain.get_node(self.claim)
        # Compare against what get_node ACTUALLY attaches, derived live —
        # deliberately not against CANONICAL_ATTACHMENT_KEYS. Deriving the
        # comparison from the same constant the overlay reads would give both
        # the same blind spot: drop a key from the tuple and the doors would
        # stop copying that field AND the test would stop checking it.
        naked = self.brain._nodes.get_naked_node(self.claim)
        self.attached = set(self.canonical) - set(naked)

    def _assert_matches_canonical(self, node, door):
        for key in sorted(self.attached):
            self.assertEqual(
                node.get(key), self.canonical.get(key),
                '%s door disagrees with get_node on %r — the recall doors '
                'have drifted apart' % (door, key))

    def test_by_id_door_matches_the_canonical_pull(self):
        out = dispatch_read._handle_recall(
            self.brain, {'node_id': self.claim,
                         CALLER_SESSION_KEY: SESSION}, None)
        self._assert_matches_canonical(out['result']['results'][0], 'by-id')

    def test_by_query_door_matches_the_canonical_pull(self):
        out = dispatch_read._handle_recall(
            self.brain, {'query': 'zarquon telemetry eclipse window',
                         CALLER_SESSION_KEY: SESSION}, None)
        node = self._find(out['result']['results'], 'by-query')
        self._assert_matches_canonical(node, 'by-query')

    def test_batch_door_matches_the_canonical_pull(self):
        out = dispatch_read._handle_recall_batch(
            self.brain, {'queries': ['zarquon telemetry eclipse window'],
                         CALLER_SESSION_KEY: SESSION}, None)
        node = self._find(out['result'][0]['results'], 'batch')
        self._assert_matches_canonical(node, 'batch')

    def _find(self, results, door):
        for r in results:
            if r.get('id') == self.claim:
                return r
        self.fail('%s door returned no result for the seeded node (%d '
                  'results) — recall itself is broken, or the seed no longer '
                  'matches the query' % (door, len(results)))

    def test_canonicalize_covers_every_result_not_a_slice(self):
        """No cap — every result, however deep the list.

        A cap here reads as a cost saving and acts as a correctness boundary:
        the renderer draws whatever it is handed, so results past the cap
        render as authoritative with their correction chain silently absent.
        Measured live before this was fixed: with an 8-result cap and
        limit=11, indices 8 and 9 came back with no connections, no situation
        and no _corrections — and index 8 was a node that HAS corrections.
        """
        ids = [self.brain.remember(
                   type='fact', title='Zarquon sample %d' % i,
                   content='sample body %d' % i)['id']
               for i in range(11)]
        results = [{'id': nid} for nid in ids]
        self.brain.canonicalize_results(results, session_id=SESSION)
        # get_node sets `connections` unconditionally on every node it finds,
        # so its absence means this result never went through the pull.
        bare = [i for i, r in enumerate(results) if 'connections' not in r]
        assert not bare, (
            'results at %s were returned without the canonical pull — a cap '
            'is silently rendering later results as authoritative with their '
            'corrections absent' % bare)

    def test_canonical_keys_cover_what_get_node_attaches(self):
        """The overlay list must not fall behind get_node's assembly.

        get_node attaches on top of the bare DB row; anything it adds that
        CANONICAL_ATTACHMENT_KEYS doesn't name would be copied by no door,
        reintroducing the divergence on a new field.
        """
        missing = self.attached - set(CANONICAL_ATTACHMENT_KEYS)
        assert not missing, (
            'get_node now attaches %s, which CANONICAL_ATTACHMENT_KEYS does '
            'not list — canonicalize_results will not copy it and the recall '
            'doors will disagree on it. Add it to the tuple in contract.py.'
            % sorted(missing))
