"""The test brain's own guarantees — the fixture the rest of the suite trusts.

It backs ~2800 tests, so what it silently tolerates becomes what those tests
silently assume.

It also holds the wall: `test_no_test_module_can_reach_the_production_clone`
is what keeps the operator's brain out of the suite now that the clone still
exists for evals. That test is the ENFORCEMENT, not hygiene — see its
docstring before touching it.

Run: ./dev pytest tests/test_isolated_brain.py -v
"""
import ast
import functools
import glob
import os
import pathlib
import sys
import tempfile

import pytest

from tests.eval_optional import HAVE_EVAL
from tests.isolated_brain import SeedBrain, brain_env

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_DIR = TESTS_DIR.parent
EVAL_DIR = REPO_DIR / 'eval'
CLONE_FILE = EVAL_DIR / 'isolated_brain.py'
NO_EVAL_REASON = 'eval/ is not in this tree (D-8 keeps it out of the public export)'

# The clone module, by every name that reaches it. `eval.isolated_brain` is
# the qualified path; the bare `isolated_brain` is the one that actually bites
# — eval scripts prepend both `<repo>/eval` and `<repo>/tests` to sys.path, in
# that order, so a bare import silently resolves to the clone.
CLONE_MODULES = ('eval.isolated_brain', 'isolated_brain')
# Names that clone the operator's brain, by whatever route. `copy_brain_for_testing`
# was a second door — a snapshot of brain.db + brain_logs.db into /tmp, owned by
# brain_test_base rather than by the clone. It is deleted; it is listed here so
# rebuilding it under its old name fails loudly instead of quietly reopening the door.
CLONE_DISCOVERY = ('_default_production_dir', 'copy_brain_for_testing')
# What the textual scan hunts. The qualified module name is spelled once, here,
# rather than again beside the loop — the two halves of the scan must not drift.
TEXTUAL_NEEDLES = CLONE_MODULES[:1] + CLONE_DISCOVERY


def _imported_names(nodes):
    """(name, line, level) for each import among `nodes`. `from X import y`
    yields X.y as well, because y may be a submodule."""
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno, 0
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            yield base, node.lineno, node.level
            for alias in node.names:
                yield ('%s.%s' % (base, alias.name) if base else alias.name,
                       node.lineno, node.level)


def _run_at_import(tree):
    """The nodes that can run when a module is imported: all but function
    bodies. Class bodies and module-level if/try blocks do run; a
    `__main__` block is counted too, which errs toward flagging."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


@functools.lru_cache(maxsize=None)
def _import_time_imports(path):
    """What `path` imports when it is imported, parsed once per file. A file
    that does not parse cannot be imported either, so it loads nothing."""
    try:
        tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    except SyntaxError:
        return ()
    return tuple(_imported_names(_run_at_import(tree)))


def _eval_files(name, level, importer):
    """The eval/ files that importing `name` from `importer` executes.

    `eval.*` resolves from the repo root. A bare name resolves the way eval
    scripts make it resolve — they put their own directory and eval/ on
    sys.path, and some tests put eval/ there too — so `from
    encoder_prompt_reassembly import ...` is followed. Precedence is
    Python's: a regular package, then a module, then a namespace directory,
    which yields to a later path entry. A package's `__init__.py` runs
    before its submodule.
    """
    if level:
        bases = [importer.parents[level - 1]]
    elif name.split('.')[0] == 'eval':
        bases = [REPO_DIR]
    elif EVAL_DIR in importer.parents:
        bases = [importer.parent, EVAL_DIR]
    else:
        bases = [EVAL_DIR]
    parts = [p for p in name.split('.') if p]
    for base in bases:
        files, here = [], base
        for i, part in enumerate(parts):
            here = here / part
            if (here / '__init__.py').is_file():
                files.append(here / '__init__.py')
            elif i == len(parts) - 1 and here.with_suffix('.py').is_file():
                files.append(here.with_suffix('.py'))
            elif not here.is_dir():
                break
        else:
            if files:
                return [f for f in files if EVAL_DIR in f.parents]
    return []


def _clone_chain(path, seen):
    """The imports by which executing `path` loads the clone — [] when `path`
    is the clone — or None when it does not."""
    if path == CLONE_FILE:
        return []
    if path in seen:
        return None
    seen.add(path)
    for name, line, level in _import_time_imports(path):
        for target in _eval_files(name, level, path):
            rest = _clone_chain(target, seen)
            if rest is not None:
                return ['%s:%d' % (path.relative_to(REPO_DIR), line)] + rest
    return None


def _offenders(path):
    """How one module under tests/ reaches the clone, one entry per line."""
    source = path.read_text(encoding='utf-8', errors='replace')
    rel = path.relative_to(REPO_DIR)

    # Textual first: catches importlib, mock's string targets, and any
    # other dynamic route that never becomes an ast.Import node.
    found = ['%s names %s' % (rel, needle)
             for needle in TEXTUAL_NEEDLES if needle in source]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    reported = set()
    # level > 0 is a relative import, which cannot reach eval/ from here.
    for name, line, level in _imported_names(ast.walk(tree)):
        if level or line in reported:
            continue
        if name in CLONE_MODULES:
            reported.add(line)
            found.append('%s:%d imports %s' % (rel, line, name))
            continue
        for target in _eval_files(name, 0, path):
            chain = _clone_chain(target, set())
            if chain is not None:
                reported.add(line)
                hops = chain + [str(CLONE_FILE.relative_to(REPO_DIR))]
                found.append('%s:%d imports %s, which loads the clone: %s'
                             % (rel, line, name, ' -> '.join(hops)))
                break
    return found


def test_brain_env_routes_to_the_substrate_it_names():
    """brain_env() is the single door for every test brain, so which class it
    hands back is a contract, not an implementation detail.

    Without this, swapping the substrate sends every call site back to cloning
    ~2 GB each: the suite stays GREEN and merely becomes an hour slower, which
    is the one failure mode nothing else here would catch.

    This file owns WHICH substrate comes back. What that substrate must
    CONTAIN — current generation and schema, the whole seed graph, its
    embeddings at the live model, a first recall that returns hits — is
    tests/test_fixture_contract.py.
    """
    assert isinstance(brain_env(), SeedBrain)


def test_the_door_has_no_production_branch():
    """`production=True` must not be an argument the factory accepts.

    The clone is reachable only from eval/, so a test asking this door for one
    is asking for something that no longer exists here. A TypeError naming the
    unexpected keyword is the honest answer; silently handing back a seed
    brain would make the argument a lie.
    """
    with pytest.raises(TypeError, match='production'):
        brain_env(production=True)


def test_no_test_module_can_reach_the_production_clone():
    """No module under tests/ may import the production clone. THE WALL.

    The operator's standing ruling: nothing of their brain is to be copied or
    used for the test suite. The clone was kept — evals measuring recall over
    grown memory are its one legitimate use — so it MOVED to eval/ rather
    than being deleted. That trade was made with its cost
    stated: because the class still exists somewhere, the auto-discovery
    ladder that finds the operator's brain still exists too, and "tests
    cannot reach it" is therefore a TESTED property rather than a structural
    impossibility.

    This test is that property. It is the enforcement, not hygiene. Adding an
    exemption here does not weaken a lint — it removes the ruling, and the
    CLAUDE.md rule against widening an exclusion to make a change pass is
    pointed directly at this function.

    If a test seems to need the clone, it does not: the survey behind the
    ruling checked all 22 sites that used one and not one asserted on
    operator-specific content. The three that looked hardest — a recall whose
    query was operator content, a tool test wanting FTS5 hits, and the LAF
    integration smoke — all run on the seed substrate now. What a test
    actually needs is either a seeded graph (brain_env) or a synthetic
    production dir (see eval/test_isolated_brain.py, which exercises the
    whole clone path without copying anything real).

    The scan is static, so a lazy or in-function import in a test module
    cannot slip past it, and it covers the BARE module name as well as the
    qualified one: eval scripts prepend <repo>/eval before <repo>/tests on
    sys.path, so `import isolated_brain` resolves to the clone, not to this
    package's module.

    It is also transitive: an eval module a test imports is followed through
    what it imports at module level, so the clone cannot arrive one hop out,
    through a harness that imports it at the top. What source cannot settle
    is left to runtime — an import inside an eval function, which runs if a
    test calls it or the module calls it while importing, and an `importlib`
    call in an eval module. conftest records every database the suite opens
    outside a temp dir.
    """
    offenders = []
    for path in sorted(TESTS_DIR.rglob('*.py')):
        if path == pathlib.Path(__file__).resolve():
            continue  # the scanner necessarily names what it searches for
        offenders.extend(_offenders(path))

    assert not offenders, (
        'the test suite reached the production clone:\n  %s\n\n'
        'Do not add an exemption — that deletes the guarantee this test IS. '
        'Use brain_env() for a seeded graph, or a synthetic production dir '
        'the way eval/test_isolated_brain.py does.' % '\n  '.join(offenders))


def test_the_wall_follows_a_clone_import_through_eval_modules(tmp_path,
                                                              monkeypatch):
    """The wall must be able to fail, or it is not a wall.

    It asserts an empty list over today's tree, which stays just as green
    when the resolver silently finds nothing. So it runs here over a
    synthetic tree holding the routes it exists to catch — a test's eval
    import that loads the clone two hops out through a bare sibling name,
    and the same module imported bare, the way a test that puts eval/ on
    sys.path would — and the one it must NOT flag: a clone import inside an
    eval function.
    """
    module = sys.modules[__name__]
    eval_dir, tests_dir = tmp_path / 'eval', tmp_path / 'tests'
    eval_dir.mkdir()
    tests_dir.mkdir()
    (eval_dir / 'isolated_brain.py').write_text('')
    (eval_dir / 'harness.py').write_text('import helper\n')
    (eval_dir / 'helper.py').write_text(
        'from eval.isolated_brain import IsolatedBrain\n')
    (eval_dir / 'lazy.py').write_text(
        'def main():\n    from eval.isolated_brain import IsolatedBrain\n')
    probe = tests_dir / 'test_reach.py'
    probe.write_text('from eval import harness\nimport harness\n'
                     'from eval import lazy\n')
    monkeypatch.setattr(module, 'REPO_DIR', tmp_path)
    monkeypatch.setattr(module, 'EVAL_DIR', eval_dir)
    monkeypatch.setattr(module, 'CLONE_FILE', eval_dir / 'isolated_brain.py')

    chain = ('eval/harness.py:1 -> eval/helper.py:1 -> eval/isolated_brain.py')
    assert _offenders(probe) == [
        'tests/test_reach.py:1 imports eval.harness, which loads the clone: '
        + chain,
        'tests/test_reach.py:2 imports harness, which loads the clone: '
        + chain,
    ]


def test_the_runtime_wall_records_a_database_outside_the_temp_dir():
    """conftest's half of the wall must be installed, and must record.

    It is heard from only on a bad run, so losing it — a refactor that moves
    the install somewhere that never runs, or drops it — leaves every run
    green. The probe opens a read-only URI on a path that does not exist, so
    nothing is created or read, then takes its own entry back out.
    """
    import sqlite3

    report = getattr(sqlite3.connect, 'outside_tmp', None)
    assert report is not None, 'conftest did not install its sqlite3.connect guard'
    probe = os.path.join(os.sep, 'brain_wall_probe_%d' % os.getpid(),
                         'brain.db')
    before = list(report)
    try:
        with pytest.raises(sqlite3.OperationalError):
            sqlite3.connect('file:%s?mode=ro' % probe, uri=True)
        recorded = [e for e in report if e not in before]
    finally:
        report[:] = before
    assert len(recorded) == 1 and recorded[0].startswith(probe), recorded


@pytest.mark.skipif(not HAVE_EVAL, reason=NO_EVAL_REASON)
def test_collecting_eval_imports_only_the_clone_s_contract_test():
    """`./dev pytest tests/ eval/` must collect exactly one eval test file.

    pytest collects `test_*.py` AND `*_test.py`, and collecting a file means
    importing it, so an eval script named like a test runs its module level
    inside the suite. One did: it cloned the operator's brain at collection
    on every full run. Scripts take a `_probe` suffix instead.
    """
    collected = sorted(str(p.relative_to(REPO_DIR))
                       for pattern in ('test_*.py', '*_test.py')
                       for p in EVAL_DIR.rglob(pattern))
    assert collected == ['eval/test_isolated_brain.py'], collected


@pytest.mark.skipif(not HAVE_EVAL, reason=NO_EVAL_REASON)
def test_the_clone_is_still_where_evals_expect_it():
    """The wall must not become a deletion by accident.

    Skipped where `eval/` is not in the tree at all — D-8 keeps it out of the
    public export, and "the clone is missing" is the correct state there. The
    guardrail above is NOT skipped: it must hold in every tree, and it needs
    no eval/ to run.

    ~137 eval modules import `eval.isolated_brain`; none of them run in the
    default suite, so a move or rename that broke them would surface only
    when someone next ran an eval. Resolving the spec proves the module is
    there WITHOUT importing it — which would itself breach the wall above.
    """
    import importlib.util
    assert importlib.util.find_spec('eval.isolated_brain') is not None, (
        'eval.isolated_brain is gone — the evals that measure against the '
        "operator's grown memory import it by that path")


def test_a_dead_embedder_is_refused_at_open_and_leaves_nothing_behind():
    """`needs_embedder=True` plus a dead embedder must raise HERE.

    The embedder is a process global: one failed load poisons every later test
    in the run, and the tests that fail are never the test that broke it.

    The cleanup half is the subtle one and is why this test exists. The check
    cannot run before `__enter__` mutates the environment — it needs the Brain
    that the environment points at — and a raise inside `__enter__` means
    Python never calls `__exit__`. Without explicit teardown the three env
    vars would leak pointing at a temp dir that is then never removed, and
    conftest's env-leak guard would fail the session with a second, louder
    failure that buries the first.
    """
    from unittest import mock

    from servers import embedder
    from tests.isolated_brain import brain_env

    # Warm the template before mocking, so this test is about the OPEN gate
    # in any collection order. A cold process would build the template inside
    # the mock and fail in the BUILD gate instead — also correct, also loud,
    # but a different guarantee than the one under test here.
    with brain_env(needs_embedder=True, load_env=False):
        pass

    keys = ('BRAIN_DB_DIR', 'ASPECTS_JSON_PATH', 'BRAIN_TMP_DIR')
    before = {k: os.environ.get(k) for k in keys}
    # This process's own dirs only — a concurrent `pytest` in another worktree
    # creates and removes its own brain_test_* dirs in the same shared tmp root,
    # and either one inside this window fails the orphan assertion below for a
    # reason that has nothing to do with the refused open.
    pattern = os.path.join(tempfile.gettempdir(), 'brain_test_%d_*' % os.getpid())
    dirs_before = set(glob.glob(pattern))

    with mock.patch.object(embedder, 'is_ready', return_value=False):
        with pytest.raises(RuntimeError, match='embedder is not ready'):
            with brain_env(needs_embedder=True, load_env=False):
                pass

    assert {k: os.environ.get(k) for k in keys} == before, (
        'the refused open leaked environment pointing at a temp dir it also '
        'deleted — every later test in the process resolves the brain there')
    assert set(glob.glob(pattern)) == dirs_before, (
        'the refused open orphaned its temp directory')


def test_seeded_false_provisions_an_empty_graph():
    """`seeded=False` means no nursery pack, and `seeded` defaults to on.

    Both halves matter. The empty graph is what ~167 BrainTestBase classes get
    and what several of them assert against — a test whose subject is the
    empty-graph path reads differently the moment a 26-node prior appears. The
    default staying on is what keeps every other caller unchanged.
    """
    from servers.seed_pack import SEED_NODES
    from tests.isolated_brain import brain_env

    with brain_env(needs_embedder=False, seeded=False, load_env=False) as env:
        empty = env.brain.conn.execute(
            'SELECT COUNT(*) FROM nodes').fetchone()[0]
    with brain_env(needs_embedder=False, load_env=False) as env:
        default = env.brain.conn.execute(
            'SELECT COUNT(*) FROM nodes').fetchone()[0]

    assert empty == 0, 'seeded=False still handed out %d nodes' % empty
    assert default == len(SEED_NODES), (
        'the seeded default changed — every caller that never passes `seeded` '
        'just had its graph swapped under it')


def test_brain_env_refuses_seeded_false_with_an_embedder():
    """The one combination the factory cannot honour, refused rather than faked.

    servers/brain.py gates seeding on `if not skip_embedder`, so a brain that
    loads the model is seeded whether or not the factory asks. Returning a
    seeded brain for `seeded=False` would make the argument a lie, and the lie
    would be invisible — the caller would simply have a graph it asked not to
    have.
    """
    from tests.isolated_brain import brain_env
    with pytest.raises(TypeError, match='welded'):
        brain_env(seeded=False, needs_embedder=True)


def test_recall_without_an_embedder_refuses_instead_of_returning_empty():
    """An unequipped recall must not be readable as "nothing matched".

    Both are an empty list at the call site, and the caller sees `0 != 1` and
    goes looking at their query or their fixture data. Reading that silence as
    a property of the seed pack — rather than as a fixture that had never been
    embedded — is the misdiagnosis this whole layer exists to prevent.

    The escape hatch is asserted too: a test that genuinely wants the
    unequipped path must keep a way through, or this guard just relocates the
    problem.
    """
    from tests.isolated_brain import brain_env
    with brain_env(needs_embedder=False, load_env=False) as env:
        with pytest.raises(RuntimeError, match='NOT because nothing matched'):
            env.brain.recall(query='anything at all', limit=5)

        raw = env.brain.recall.__wrapped__(query='anything at all',
                                           limit=5) or {}
        assert raw.get('results') == [], (
            'the bypass should reach the real, unequipped recall')


def test_log_sweep_reports_only_this_brain_s_own_window():
    """The sweep's window is the context, not the file.

    A production clone opens on top of the operator's log history, so a sweep
    that read the table would report months of the operator's rows as though
    this run produced them. The watermark is taken after provisioning, so
    what the substrate wrote while opening — seed-pack included — is not
    this measurement's noise either.
    """
    from tests.isolated_brain import brain_env

    mine = {'recall_laf', 'laf_resolution_loss'}
    with brain_env(needs_embedder=False) as env:
        # Deliberately NOT `== []`: open-time work may legitimately log, and
        # an exact-empty assertion would fail on a correct implementation the
        # day it does. What must hold is that nothing of MINE is in there yet.
        assert not [r for r in env.log_sweep() if r['origin'] in mine]

        env.brain.log_debug('laf_resolution', 'recall_laf',
                            harvested=40, credited=38, dropped=2, share=0.05)
        env.brain.log_debug('laf_resolution', 'recall_laf',
                            harvested=9, credited=9, dropped=0, share=0.0)
        env.brain._log_warning('laf_resolution_loss', 'all 12 reached no live row')

        rows = {(r['origin'], r['level']): r for r in env.log_sweep()}

    gauge = rows[('recall_laf', 'laf_resolution')]
    assert gauge['count'] == 2, gauge
    assert gauge['samples'], 'a group with no sample message is unreadable'

    # Warnings are IN. The reader decides what invalidates a measurement, so
    # the sweep must not pre-filter to errors the way the longmem suspect
    # marking does — that filter is why warning-level signal was invisible.
    assert ('laf_resolution_loss', 'warning') in rows, sorted(rows)


def test_unestablished_window_is_refused_not_widened():
    """A failed watermark read must not silently become "since row 0".

    0 is a legitimate watermark meaning "the table was empty", so a failed
    read that returned it would widen the window to everything — on a clone,
    the operator's entire history reported as this run's output. The sweep
    refuses instead, which is the one outcome nobody can misread.
    """
    from tests.isolated_brain import brain_env

    with brain_env(needs_embedder=False) as env:
        env.brain.log_debug('probe', 'test_window', message='in the window')
        assert env.log_sweep(), 'precondition: a bounded window still sweeps'

        env.log_open_id = None          # what a failed watermark read leaves
        for call in (env.log_sweep, env.print_log_sweep):
            try:
                call()
            except LookupError:
                continue
            raise AssertionError(
                '%s swept an unbounded window instead of refusing'
                % call.__name__)


def test_test_substrates_do_not_sweep_on_exit():
    """The mechanism is shared; the printing is not. A test asserts on what it
    came to assert on, and the suite's stdout is not a measurement report."""
    from tests.isolated_brain import BrainEnv, SeedBrain

    assert BrainEnv.sweep_logs_on_exit is False
    assert SeedBrain.sweep_logs_on_exit is False


def test_sweep_samples_a_group_whose_origin_is_null():
    """debug_log.source is nullable, so a group key can be NULL.

    The per-group sample query matches with `IS`, not `=`, because `= NULL`
    is never true in SQL: with `=` the NULL group would still be counted and
    then show no messages at all. Nothing else pins that, and `=` is what a
    later cleanup would reach for.
    """
    import sqlite3
    from tests.isolated_brain import brain_env

    with brain_env(needs_embedder=False) as env:
        conn = sqlite3.connect(env.logs_db)
        try:
            conn.execute(
                "INSERT INTO debug_log (event_type, source, metadata, created_at) "
                "VALUES ('telemetry', NULL, '{\"message\":\"orphan row\"}', ?)",
                ('2026-09-22T00:00:00Z',))
            conn.commit()
        finally:
            conn.close()
        groups = [r for r in env.log_sweep() if r['level'] == 'telemetry']

    assert len(groups) == 1, groups
    assert groups[0]['origin'] == '', groups[0]
    assert groups[0]['samples'] == ['orphan row'], groups[0]


def test_sweep_shows_an_empty_message_as_empty():
    """An empty message is a finding, not a missing field.

    A truthiness chain would fall through it and print the whole metadata
    row, hiding that the producer emitted nothing.
    """
    from tests.isolated_brain import brain_env

    with brain_env(needs_embedder=False) as env:
        env.brain.log_debug('blank', 'test_blank', message='')
        groups = [r for r in env.log_sweep() if r['origin'] == 'test_blank']

    assert groups[0]['samples'] == [''], groups[0]
