"""What `brain_env()` promises every test that opens one.

`_assert_template_complete` is the definition of a complete fixture, and
`_seed_template` calls it at the build boundary. This file calls the SAME
function against an opened copy, which is the half a build-time check can
never see: a template can be perfect and the provisioning path — copy2,
`Brain()` open, schema migration, aspects pinning — still hand back something
else. It then adds the one property that only exists once a test is holding
the fixture: that recall over it returns anything at all.

The failure direction all of this exists for is silent and green. A template
that lost its vectors, a default that flipped back to cloning ~2 GB, a seed
pack that grew nodes the template never got — none of those turn the suite
red. They make it slower, or quieter, or both. That is the shape that put "the
seed pack is too small for cosine top-k to fill k" into `brain_env()`'s
docstring as standing guidance: the template had shipped with zero embeddings
for its whole life, every first `recall()` returned nothing, and the silence
was read as a property of the data rather than a defect in the fixture. Two
accidents compounding into a rule, and 22 tests kept on production clones to
work around it.

Env isolation is deliberately NOT asserted here — `conftest.py` guards every
scope and fails the session on any leak, which is stronger than one assertion
in one test.

Run: ./dev pytest tests/test_fixture_contract.py -v
"""
import pytest

from servers.brain_recall import RECALL_FAILURE_MODES
from servers.pipeline_contract import get_excluded_types
from tests.isolated_brain import _assert_template_complete, brain_env

# `_seed_template` caches on `vectorized` and `SeedBrain` picks per its own
# `needs_embedder`, so these are two different files built by two different
# paths, and the contract differs between them in both directions — the
# vectorised one must carry every vector, the plain one must carry none.
BOTH_TEMPLATES = [False, True]

# Self-retrieval canary: a node queried by its own title must come back. Same
# instrument and same numbers as the eval validity layer
# (eval/longmem/validity.py, id:b8f49736) — held equal on purpose so a miss
# here and a miss there mean the same thing, not because 3 and 10 are tuned
# for a 26-node graph.
CANARY_SAMPLE_NODES = 3
CANARY_TOP_K = 10


@pytest.mark.parametrize('needs_embedder', BOTH_TEMPLATES)
def test_the_fixture_brain_env_hands_back_is_complete(needs_embedder):
    """The provisioning path preserved what the build established.

    One assertion because there is one definition. `_seed_template` already
    ran this against the template before snapshotting it; running it again on
    the opened copy is what covers everything between those two points —
    `SeedBrain._provision`'s file copy, `Brain()`'s open-time migrations, and
    the aspects pinning. The failure message comes from the function, so it
    names the property rather than the assertion that noticed.
    """
    with brain_env(needs_embedder=needs_embedder, load_env=False) as env:
        _assert_template_complete(env.brain, vectorized=needs_embedder)


# One case per refusal in `_assert_template_complete`, in the order the
# function checks them: a case that trips an EARLIER refusal proves nothing
# about the one it was written for, and `match=` is what holds each case to
# its own branch — every pattern below appears in exactly one refusal.
#
# Damage is the SMALLEST that reaches the branch, which for the vector cases
# is one row and not the table: the vector question goes through
# `vector_coverage_sweep`, which REPAIRS before it reports, so damaging every
# row means re-embedding the whole pack before the refusal arrives. They
# target `_primary` — the LAF-visibility invariant, and the only type `stuck`
# probes — so the case lands whether the door repairs it or gives up on it.
#
# Each subquery selects the row it damages BY the property under test, not
# by position: `(SELECT id FROM nodes LIMIT 1)` happens to be a seed node
# today only because the template holds nothing else, and would silently
# become a no-op case the day that stops being true.
_ONE_SEED_NODE = ("(SELECT id FROM nodes WHERE encoding_source = 'anchor:seed'"
                  " ORDER BY id LIMIT 1)")
_ONE_PRIMARY_ROW = ("(SELECT id FROM node_enrichments "
                    "WHERE vector_type = '_primary' ORDER BY id LIMIT 1)")


@pytest.mark.parametrize('damage,pattern', [
    pytest.param(
        'DELETE FROM nodes WHERE id = %s' % _ONE_SEED_NODE,
        'seed nodes present', id='seed-node-deleted'),
    # Archived, not deleted: the seed-node count above carries no archived
    # filter on purpose (an archived seed is retired, not missing), so this
    # is the only damage that can reach the live-count refusal behind it.
    pytest.param(
        'UPDATE nodes SET archived = 1 WHERE id = %s' % _ONE_SEED_NODE,
        'different graph than the pack defines', id='seed-node-archived'),
    pytest.param(
        "DELETE FROM edge_relations WHERE relation = 'community_member'",
        'edge relations do not match', id='membership-edges-gone'),
    pytest.param(
        "UPDATE brain_meta SET value = 'some_other_pack' "
        "WHERE key = 'seed_pack_generation'",
        'seeded by pack', id='foreign-pack-marker'),
    pytest.param(
        "UPDATE brain_meta SET value = '1' "
        "WHERE key = 'brain_schema_version'",
        'migrations did not run', id='schema-behind-code'),
    pytest.param(
        'UPDATE node_enrichments SET embedding = NULL WHERE id = %s'
        % _ONE_PRIMARY_ROW,
        'repaired on inspection', id='vector-missing'),
    # A model swap is refused by the REPAIR door, not by the model-identity
    # backstop beside it: `vector_coverage_sweep` passes `model=` into
    # `find_missing`, which reads a stale-model row as missing, so the swap
    # comes back as a repair. That is the door's model-awareness
    # demonstrated, and it is why the fixture needs no cosine check.
    pytest.param(
        "UPDATE node_enrichments SET model = 'some-other-model' WHERE id = %s"
        % _ONE_PRIMARY_ROW,
        'repaired on inspection', id='vector-at-stale-model'),
])
def test_the_completeness_contract_refuses_a_damaged_fixture(damage, pattern):
    """The contract has teeth — each property it names can actually fail it.

    Without this the test above is a call to a function that could be empty:
    every property could be silently dropped and the suite would stay green,
    which is the failure mode this whole file exists to close. A test that
    PERMITS a change instead of REQUIRING it is zero coverage (id:8a8529e6),
    and that is measured by mutation, not by reading.

    The damage is applied to a disposable copy, one property per case, so a
    check that stopped working is named by WHICH case went green.

    One refusal has no case here and cannot get one: the model-identity
    backstop. With a live embedder the repair door reads a stale-model row as
    MISSING and repairs it first, so any swap written here is caught one
    branch earlier — which the two vector cases already demonstrate. The
    backstop only fires on drift the sweep cannot see (a stale model on a
    non-`_primary` type while the embedder is down, since `stuck` probes
    `_primary` alone). Faking that state would test the fake.
    """
    with brain_env(needs_embedder=True, load_env=False) as env:
        env.brain.conn.execute(damage)
        env.brain.conn.commit()
        with pytest.raises(RuntimeError, match=pattern):
            _assert_template_complete(env.brain, vectorized=True)


def test_the_contract_refuses_a_plain_template_carrying_vectors():
    """The other direction, and the one the parametrised cases cannot reach.

    Every case above runs on the vectorised template, so none of them can
    trip the refusal that guards the OPPOSITE promise: a template built for
    tests that never load the model must carry no vectors, because a vector
    nothing can read is ~7s per process bought for nothing. That refusal is
    a cost contract, and its failure direction is silent and green — it
    makes the suite slower, never red. Without this, it could be deleted
    and nothing would notice, which is the whole reason this file exists.

    One inserted row, on a fixture that never loads the model — so the case
    that guards ~7s costs milliseconds.
    """
    with brain_env(needs_embedder=False, load_env=False) as env:
        node_id = env.brain.conn.execute(
            'SELECT id FROM nodes LIMIT 1').fetchone()[0]
        env.brain.conn.execute(
            "INSERT INTO node_enrichments (id, node_id, vector_type, text, "
            "embedding) VALUES ('probe-vec', ?, '_primary', 'x', X'00')",
            (node_id,))
        env.brain.conn.commit()
        with pytest.raises(RuntimeError,
                           match='paying to embed what it cannot read'):
            _assert_template_complete(env.brain, vectorized=False)


def test_first_recall_on_a_fresh_fixture_returns_hits():
    """The bug that motivated all of this: a fixture nothing can recall from.

    Every other assertion here can pass on a template whose first `recall()`
    returns nothing, because the harness drains AFTER the call it wraps — so
    the first query runs against an empty vector store and the drain populates
    it just in time for the second. Measured before the fix: 0 hits in 6.56s,
    then 5 hits in 0.01s for the identical query.

    So the FIRST recall in this scope has to be the probe. Nothing above it
    may call `remember`, `revise` or `recall` — all three are drain-wrapped,
    and any of them would repair the very state under test. That also rules
    out `_assert_template_complete`, whose vector question REPAIRS; this test
    holds its own fixture for that reason, and a later refactor that shares
    one with the test above would make this assertion vacuous.

    `filter_nodes` is safe: a structural read that embeds nothing.

    The probe is self-retrieval — a node queried by its own title — sampled by
    title so the same nodes are probed on every run. Recall-excluded types are
    dropped through `get_excluded_types`, not by naming `community` here:
    sampling an S2-authored type fails a healthy brain by design.
    """
    with brain_env(needs_embedder=True, load_env=False) as env:
        sample = (env.brain.filter_nodes(
            field='type', exclude=sorted(get_excluded_types('recall')),
            rich=False, limit=CANARY_SAMPLE_NODES,
            sort_by='title', sort_order='asc') or {}).get('nodes') or []
        assert len(sample) == CANARY_SAMPLE_NODES, (
            'only %d recallable node(s) to probe — the canary would report '
            'healthy without having run' % len(sample))

        misses = []
        for i, node in enumerate(sample):
            got = env.brain.recall(node['title'], limit=CANARY_TOP_K,
                                   mark_accessed=False) or {}
            mode = got.get('_recall_mode', '') or ''
            assert mode not in RECALL_FAILURE_MODES, (
                'recall reported %r on probe #%d. On a fresh fixture the '
                'likeliest cause is an EMPTY vector store — the template was '
                'snapshotted before its embeddings were drained, so LAF has '
                'nothing to score against. Check the completeness test above '
                'before suspecting recall itself.' % (mode, i))
            hits = {r.get('id') for r in got.get('results') or []
                    if isinstance(r, dict)}
            if node['id'] not in hits:
                misses.append('#%d %s (%d result(s) for its own title)'
                              % (i, node['id'][:8], len(hits)))

        assert not misses, (
            'seed node(s) did not surface for their own title: %s. A miss on '
            'probe #0 means the template carries no vectors — it is the FIRST '
            'recall, the one that runs before the harness drains. A miss only '
            'on a later probe is a ranking problem, not an empty store.'
            % '; '.join(misses))
