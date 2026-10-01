"""recall(mark_accessed=False) must not touch the brain's own access record.

Why this exists: the flag was added so a read-only observer (the dashboard's
recall probe) can run the REAL recall pipeline without its looking becoming
part of what the brain believes it recalled — access_count / last_accessed are
what the graph renders as recall heat.

The pins: a read-only recall marks nothing, and a default recall marks exactly
the nodes it returned — never nodes it scored but did not return.

Assertions COUNT CALLS to `_mark_accessed`, which is exactly what the flag
governs. The two tempting observation points are both races: `_mark_accessed`
does no DB I/O (it enqueues into recall_write_queue), and a background worker
drains that queue every EMBED_DRAIN_INTERVAL seconds — so asserting on the
`nodes` table OR on `queue_depth()` passes or fails depending on whether the
worker happened to fire mid-test. Measured: the queue-depth version failed its
positive case for exactly that reason. A flaky test on this contract is worse
than none.
"""

import pytest

from tests.isolated_brain import brain_env

# A query the nursery seed pack answers — every assertion below is guarded by
# "recall returned something", and a query recall could not answer would make
# these tests vacuous without failing.
QUERY = 'a correction from my operator is treasure'


@pytest.fixture(scope='module')
def brain():
    # needs_embedder=True: the seed template carries its document vectors,
    # but recall still needs the model to encode the QUERY.
    with brain_env(needs_embedder=True) as env:
        yield env.brain


@pytest.fixture
def marks(brain, monkeypatch):
    """Record every node id `_mark_accessed` is called with, and suppress the
    real enqueue so tests never feed the live write queue."""
    seen = []
    monkeypatch.setattr(type(brain), '_mark_accessed',
                        lambda self, node_id, session_id, ctx=None: seen.append(node_id))
    return seen


def test_read_only_recall_marks_nothing(brain, marks):
    """The guarantee: mark_accessed=False marks zero nodes."""
    result = brain.recall(query=QUERY, limit=8, session_id='probe-session',
                          source='dashboard', mark_accessed=False)
    # Guard against a vacuous pass — if recall found nothing there is nothing
    # it could have marked, and the assertion below would be meaningless.
    assert result.get('results'), 'recall returned no results — test proves nothing'
    assert marks == [], (
        'read-only recall marked %d node(s): %r' % (len(marks), marks[:5]))


def test_default_recall_still_marks(brain, marks):
    """The default path marks exactly what it returned — the exception must not
    become the rule (disabling marking everywhere would otherwise pass), and a
    node recall scored but did not return is not a recognition."""
    result = brain.recall(query=QUERY, limit=8, session_id='real-session',
                          source='mcp', mark_accessed=True)
    assert result.get('results'), 'recall returned no results — test proves nothing'
    assert sorted(marks) == sorted(n['id'] for n in result['results']), (
        'default recall marked %r, returned %r'
        % (marks, [n['id'] for n in result['results']]))

