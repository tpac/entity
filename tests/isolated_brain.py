"""The isolated brain environment the test suite runs on.

`BrainEnv` is the machinery — a brain in a temp directory that is cleaned up
on exit, with the environment pinned around it and never a write to live data.
A substrate fills that directory by overriding two hooks, `_provision` and
`_seed_aspects_file`.

`SeedBrain` is the only substrate reachable from here, and `brain_env()` is
the door that hands it out: a fresh brain born with the nursery seed pack,
~2 MB, stamped in milliseconds, identical on every machine, carrying none of
the operator's content.

    from tests.isolated_brain import brain_env

    with brain_env() as env:
        env.brain    # the seed pack's nodes and edges, vectors intact, no embedder
        env.dispatch("recall", {"query": "test", "limit": 5})

    # Or keep the temp dir for inspection:
    with brain_env(cleanup=False) as env:
        print("Inspect at:", env.db_dir)

The other substrate — a clone of the operator's production brain — lives in
`eval/isolated_brain.py` and is eval-only. Nothing under `tests/` may reach
it; `tests/test_isolated_brain.py` enforces that, and it is the whole
enforcement rather than hygiene — the operator's standing ruling is that
nothing of their brain is copied or used for the test suite. Widening that
test to make something pass is how the wall comes down.
"""
import atexit
import os
import shutil
import tempfile


_SERVERS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'servers')


class BrainEnv:
    """Context manager holding an isolated brain in a temp directory.

    Owns everything a substrate shares: the temp dir, the three environment
    pins and their restore, the aspects pin, the dead-embedder refusal, the
    synchronous drain wrapper and the blind-recall guard. A subclass supplies
    only WHERE the two databases come from.

    Subclasses override two hooks and nothing else:
        _provision()          fill self.db_dir with brain.db + brain_logs.db
        _seed_aspects_file()  put an aspects_v1.json beside them

    `brain_env()` is how a test gets one — it names the substrate so call
    sites do not have to.
    """

    # Whether to load the embedding model. A substrate whose content already
    # carries its vectors sets this False and opens in milliseconds; one that
    # must embed at open time, or whose tests encode a query, sets it True.
    needs_embedder = True

    # Whether to print this brain's own log window when the context closes.
    # A measurement substrate sets it True: an eval run is blind to anything
    # its brain complained about, and the complaint is usually the reason a
    # number looks the way it does. Test substrates leave it False — a test
    # asserts on what it came to assert on, and stdout is the suite's.
    sweep_logs_on_exit = False

    def __init__(self, cleanup=True, load_env=True):
        """
        Args:
            cleanup: delete temp dir on exit (False to keep for inspection)
            load_env: load .env file for API keys (needed for Haiku/Sonnet calls)
        """
        self.cleanup = cleanup
        self.load_env = load_env
        self.db_dir = None
        self.brain = None
        self.brain_db = None
        self.logs_db = None
        self.log_open_id = None      # None = window not established

    def _provision(self):
        """Put brain.db + brain_logs.db into self.db_dir. Subclasses only.

        Called from __enter__ BEFORE any environment mutation, so a substrate
        that cannot produce its databases fails without leaving pins behind.
        """
        raise NotImplementedError(
            '%s must override _provision to say where its databases come from'
            % type(self).__name__)

    def _seed_aspects_file(self):
        """Put an aspects_v1.json in self.db_dir. Subclasses only.

        Called only when this env owns the aspects pin — a caller that pinned
        ASPECTS_JSON_PATH itself keeps its own file.
        """
        raise NotImplementedError(
            '%s must override _seed_aspects_file to say which taxonomy it '
            'reads' % type(self).__name__)

    def __enter__(self):
        # Create temp directory. The PID is in the name because the tmp root is
        # shared by every process of this user: parallel streams run their own
        # `pytest` against sibling worktrees, and a test that reasons about the
        # brain_test_* dirs on disk (test_isolated_brain's orphan check) must be
        # able to see only its own. Same reason log_test_result keys its run id
        # by PID.
        self.db_dir = tempfile.mkdtemp(prefix='brain_test_%d_' % os.getpid())
        self.brain_db = os.path.join(self.db_dir, 'brain.db')
        self.logs_db = os.path.join(self.db_dir, 'brain_logs.db')

        # Fill db_dir with the two databases; the substrate picks the source.
        # The failure-prone step runs BEFORE any os.environ mutation: if it
        # raises, __enter__ propagates and Python never calls __exit__, so a
        # half-set environment would leak. Env assignments (below) can't raise.
        #
        # That same "no __exit__ after a failed __enter__" is why the cleanup
        # is explicit here: the temp dir already exists by this point, so a
        # provisioning failure would otherwise orphan it.
        try:
            self._provision()
        except BaseException:
            shutil.rmtree(self.db_dir, ignore_errors=True)
            raise

        # Load .env for API keys
        if self.load_env:
            _load_env()

        # Set environment so Brain resolves correctly. Both BRAIN_DB_DIR and
        # ASPECTS_JSON_PATH are saved here and restored in __exit__ so the
        # process env returns to its pre-context state — the temp dir is
        # rmtree'd on exit, so a leaked override would point at a deleted dir.
        self._orig_brain_db_dir = os.environ.get('BRAIN_DB_DIR')
        self._orig_aspects_json_path = os.environ.get('ASPECTS_JSON_PATH')
        self._orig_brain_tmp_dir = os.environ.get('BRAIN_TMP_DIR')

        # Aspect isolation. If the caller already pinned ASPECTS_JSON_PATH
        # (e.g. run_aspect_cycles_on_clone.py points it at its own work file),
        # honor it — they own the aspects location. Otherwise pin it into our
        # temp dir so the BRAIN_DB_DIR fallback in aspects_json_path() can't
        # leak heals into the live file, and let the substrate fill it.
        #
        # conftest's session-wide safety pin is NOT such a caller — it is a
        # default, and treating it as one would make this branch dead for every
        # test in the process and quietly hand them all one shared aspects file.
        # It publishes its value so the two can be told apart.
        #
        # The failure-prone copy runs BEFORE any os.environ mutation: if it
        # raises, __enter__ propagates and Python never calls __exit__, so a
        # half-set environment would leak. Env assignments (below) can't raise.
        pin_aspects = self._orig_aspects_json_path in (
            None, os.environ.get('BRAIN_TEST_ASPECTS_DEFAULT_PIN'))
        if pin_aspects:
            self._seed_aspects_file()

        os.environ['BRAIN_DB_DIR'] = self.db_dir
        if pin_aspects:
            os.environ['ASPECTS_JSON_PATH'] = os.path.join(
                self.db_dir, 'aspects_v1.json')
        # Route brain's ephemeral /tmp files into this isolated run's dir so
        # concurrent eval/test processes don't collide on fixed filenames and
        # the files are cleaned up with the temp dir (see daemon_config.brain_tmp_dir).
        os.environ['BRAIN_TMP_DIR'] = self.db_dir

        # Create isolated Brain instance
        from servers.brain import Brain
        self.brain = Brain(self.brain_db,
                           skip_embedder=not self.needs_embedder)

        # A caller that asked for the embedder and got a dead one must hear it
        # HERE, not twenty assertions later on absent vectors. The embedder is
        # a process global: one failed load poisons every later test in the
        # run, and the tests that fail are never the test that broke it.
        # BrainTestBase guards its own raw-Brain path the same way; this
        # covers everything that comes through brain_env(), including the
        # cached-template path where no build-time gate runs.
        #
        # Unlike the checks above, this one cannot run before the environment
        # is mutated — it needs the Brain that the environment points at. So
        # it undoes the context itself before raising: a raise inside
        # __enter__ means Python never calls __exit__, and the three env vars
        # set above would otherwise leak pointing at a temp dir that no longer
        # exists. __exit__ is exactly that teardown, so call it rather than
        # hand-rolling a second copy.
        from servers import embedder
        if self.needs_embedder and not embedder.is_ready():
            status = embedder.get_model_status()
            self.__exit__(None, None, None)
            raise RuntimeError(
                'this brain was opened with needs_embedder=True but the '
                'embedder is not ready — %s. Every vector or recall assertion '
                'against it would fail for this reason and not its own.'
                % status)

        # Auto-drain embeddings after writes — same rationale as in
        # BrainTestBase._patch_writes_to_auto_drain. The production embed
        # queue worker fires every 5s; tests run faster, so a remember()
        # → recall() in the same scope returns nothing. Wrap the write
        # APIs to drain synchronously, restoring the pre-deferral
        # contract that test code is written against.
        from servers import embed_queue, recall_write_queue

        # Only a brain that asked for the embedder may write vectors. The
        # embedder is a PROCESS GLOBAL, so without this an embedder-less brain
        # silently starts embedding as soon as any other test in the run loads
        # the model — its writes gain vectors in a batch run and lack them when
        # the file runs alone. That is a test reading differently depending on
        # what ran before it; measured as `find_missing('edge_context')`
        # returning 5 of 7 nodes in a batch and 7 of 7 alone.
        #
        # The access-mark queue stays unconditional: it needs no model, and a
        # test asserting on access_count after recall() depends on it draining.
        drains_vectors = self.needs_embedder

        def _check_blind_recall(result):
            """Refuse a recall this brain was never equipped to answer.

            A brain that declared needs_embedder=False has no query encoder and
            no document vectors, so recall returns an empty list. Empty is also
            what "nothing matched" looks like, and the two read identically at
            the call site — the caller sees `0 != 1` and goes looking at their
            query, their fixture data, their ranking. That confusion is the
            whole reason this fixture layer exists: the seed pack was called
            "too small for cosine top-k" for months when the real answer was
            that nothing had been embedded.

            The brain already distinguishes them — `_recall_mode` reports a
            member of RECALL_FAILURE_MODES rather than a result mode. Nobody
            reads a key they had no reason to suspect, so say it here instead.

            Only recalls the TEST makes directly are checked. Production code
            under servers/ recalls for its own reasons — `hook_recall` runs one
            to represent "an operator turn happened" — and an empty result is
            the degraded path it is built to tolerate. Raising there would turn
            a handled condition into a hard failure in tests that never
            mentioned recall, which is a different error from the one this
            guard exists to prevent.

            A test that genuinely wants the unequipped behaviour (the keyword
            net, or asserting on the degraded mode itself) can call
            `brain.recall.__wrapped__(...)` to bypass this.
            """
            from servers.brain_recall import RECALL_FAILURE_MODES
            mode = (result or {}).get('_recall_mode') or ''
            if mode not in RECALL_FAILURE_MODES:
                return
            raise RuntimeError(
                "recall on a brain opened with needs_embedder=False returned "
                "_recall_mode=%r, so the result is empty because the machinery "
                "is unavailable — NOT because nothing matched. Without the "
                "model there is no query encoder, and this brain's template "
                "carries no document vectors either. If the test expects hits, "
                "open it with brain_env(needs_embedder=True). If it genuinely "
                "wants the unequipped path, call brain.recall.__wrapped__()."
                % mode)

        def _wrap(name):
            orig = getattr(self.brain, name, None)
            if orig is None or getattr(orig, '_drain_wrapped', False):
                return
            brain = self.brain
            checks_recall = (name == 'recall') and not self.needs_embedder

            def w(*args, **kwargs):
                result = orig(*args, **kwargs)
                if checks_recall and not _called_from_servers():
                    _check_blind_recall(result)
                try:
                    if drains_vectors:
                        embed_queue._drain_once(brain)
                except Exception as e:
                    import sys
                    print('[isolated_brain] embed drain after %s '
                          'failed: %s' % (name, e), file=sys.stderr)
                try:
                    recall_write_queue.drain_once(brain)
                except Exception as e:
                    import sys
                    print('[isolated_brain] recall_write drain after %s '
                          'failed: %s' % (name, e), file=sys.stderr)
                return result
            w._drain_wrapped = True
            w.__wrapped__ = orig
            setattr(self.brain, name, w)

        # Wrap recall too (Phase 5, 2026-05-18): mark_accessed enqueue
        # happens on recall; tests asserting on access_count need
        # the bg_writer drain to fire synchronously.
        for name in ('remember', 'revise', 'recall'):
            _wrap(name)

        # Open end of the log window. A production clone carries the
        # operator's own rows, so 'what happened during this run' is only
        # separable by id — and ids, unlike a time window, do not shift when
        # retention prunes mid-run.
        self.log_open_id = self.brain.log_watermark()

        return self

    def log_sweep(self, samples=None):
        """Everything this brain logged since it opened, grouped by origin.

        Not filtered to errors. A measurement is invalidated as often by a
        warning or a gauge reading as by an exception, and which is which is
        a judgement the reader makes with the rows in front of them — so the
        sweep suppresses nothing and classifies nothing.

        Raises LookupError when the window was never established (the
        watermark read failed at open). Sweeping from 0 would report the
        whole table — on a clone, the operator's own months of history as
        this run's output — so an unbounded window is refused, not widened.
        """
        if not self.brain:
            return []
        if self.log_open_id is None:
            raise LookupError('log window was never established — the '
                              'watermark read failed when this brain opened')
        # samples=None defers to the DAL's own default rather than restating
        # it here: this module cannot import servers at module scope (the env
        # pins must be set first), and a second copy of a default is how two
        # callers get different answers to the same question.
        kw = {} if samples is None else {'samples': samples}
        return self.brain.log_activity_since(self.log_open_id, **kw)

    def print_log_sweep(self, samples=None):
        """The sweep as a block on stdout. Silence is stated, not implied —
        'nothing was logged' and 'nobody looked' must not read the same."""
        rows = self.log_sweep(samples=samples)
        total = sum(r['count'] for r in rows)
        print('\n[log-sweep] %d row(s) across %d origin(s) while this brain '
              'was open' % (total, len(rows)), flush=True)
        if not rows:
            print('[log-sweep]   (nothing logged)', flush=True)
        # 'origin' and 'level' are debug_log's source and event_type, named as
        # query_logs already names them. The two writers fill them differently
        # — _log_event puts the failure class in origin and the severity in
        # level; log_debug puts the producer in origin and the event name in
        # level — so both are printed whole, and neither is read as severity.
        for r in rows:
            print('[log-sweep]   %5d  %-26s %s' % (r['count'], r['origin'],
                                                   r['level']), flush=True)
            for m in r['samples']:
                print('[log-sweep]          %s' % m[:150], flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.brain:
            # save() before the sweep, close() after: a row written during
            # the final flush belongs to this run's window, and the sweep
            # still needs the brain's read door open to go and get it.
            try:
                self.brain.save()
            except Exception:
                pass
            if self.sweep_logs_on_exit:
                try:
                    self.print_log_sweep()
                except Exception as e:
                    print('[log-sweep] FAILED: %r' % e, flush=True)
            try:
                self.brain.close()
            except Exception:
                pass
        # Restore env overrides so they don't leak to later code (the temp dir
        # we pointed them at is about to be rmtree'd).
        if hasattr(self, '_orig_brain_db_dir'):
            if self._orig_brain_db_dir is None:
                os.environ.pop('BRAIN_DB_DIR', None)
            else:
                os.environ['BRAIN_DB_DIR'] = self._orig_brain_db_dir
        if hasattr(self, '_orig_aspects_json_path'):
            if self._orig_aspects_json_path is None:
                os.environ.pop('ASPECTS_JSON_PATH', None)
            else:
                os.environ['ASPECTS_JSON_PATH'] = self._orig_aspects_json_path
        if hasattr(self, '_orig_brain_tmp_dir'):
            if self._orig_brain_tmp_dir is None:
                os.environ.pop('BRAIN_TMP_DIR', None)
            else:
                os.environ['BRAIN_TMP_DIR'] = self._orig_brain_tmp_dir
        if self.cleanup and self.db_dir:
            shutil.rmtree(self.db_dir, ignore_errors=True)
        return False

    def dispatch(self, cmd, args=None):
        """Execute a daemon command against the isolated brain.

        Routes through `dispatch_command` — the same execution chokepoint the
        daemon and the encoder closure use, so eval/test runs exercise the real
        arg-contract validation and per-op loudness path. No TCP, no daemon.

        No write lock is taken: an isolated brain is single-threaded by
        construction, with no daemon, autosave or embed_queue to serialize
        against.
        """
        from servers.daemon_dispatch import dispatch_command
        return dispatch_command(self.brain, cmd, args or {}, [])

    def node_count(self):
        """Quick check: how many non-archived nodes."""
        row = self.brain.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()
        return row[0] if row else 0


# ── Seed brain: the default test substrate ───────────────────────────
# Built once per process, then stamped per test. The operator's brain is
# never involved.

# One model cache for the whole suite, and the reason it has to be said here
# rather than in servers/brain.py.
#
# Brain derives cache_dir from db_path (brain.py:1687) — durable next to
# brain.db, which is right for the daemon: its db sits still, and fastembed's
# own $TMPDIR default got purged and silently re-pulled the model at whatever
# `main` pointed to that day. A test brain's db is a fresh mkdtemp, so that
# same rule hands every test a cache_dir no other test shares, and
# load_model's singleton guard (`config == _config`) can never hold: each
# seeded test built a second ONNX session for the same pinned artifact —
# measured 0.169s x 244 tests. The payload itself was already cheap (the xet
# seed hardlinks the 130MB blob), so what this buys back is session build
# time, not disk.
#
# The template carries the pin in its brain_meta, so every copy inherits it
# and the whole suite agrees on one path. Deliberately NOT changed in
# servers/brain.py: eval's per-item brains want the opposite — a fresh
# per-item cache is their retry seam, so one poisoned seed heals on the next
# item instead of every item (eval/longmem/fresh_brain.py:93).
#
# Durable, not $TMPDIR: a purge mid-suite would re-download under the pin and
# fail every test loudly. Seeding it the first time is the existing
# stage-and-rename path, which is concurrency-safe across worktrees.
_SHARED_MODEL_CACHE = os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'),
    'brain-test-fastembed')

# Keyed on `vectorized` — the two templates answer different needs and cost
# very differently (~1.3s plain, ~7.4s drained), so a process builds only the
# one(s) its tests ask for.
_SEED_TEMPLATE_DIRS = {}


def _called_from_servers():
    """True when the immediate caller of a wrapped brain method is brain code.

    Distinguishes "the test asked for this recall" from "production code the
    test is exercising asked for it" — `hook_recall` runs one to represent an
    operator turn, and the test around it never mentions recall. One frame is
    the right depth: it is the actual caller, so a test calling recall()
    directly is always seen as a test.
    """
    import sys
    caller = sys._getframe(2).f_code.co_filename
    return os.path.abspath(caller).startswith(_SERVERS_ROOT)


def _assert_template_complete(brain, vectorized):
    """Refuse a seed brain that is not what `vectorized` promised.

    ONE definition of complete, called twice: here at the build boundary
    before the snapshot, and from tests/test_fixture_contract.py against an
    OPENED copy — where passing also proves the provisioning path preserved
    what the build established. A template can be perfect and the copy path
    still broken, which is the half a build-time check can never see.

    Two definitions is how the halves drift, and these two had already: the
    build counted seed nodes one way and the contract test another, on the
    day the test was written. So a new property goes HERE and both callers
    get it, rather than becoming a third scattered `if`.

    Raises RuntimeError naming what is missing and who produces it.

    NOT side-effect free when `vectorized`: the vector question goes through
    `brain.vector_coverage_sweep()` — the one model-aware repair-and-report
    door — which REPAIRS before it reports. That is correct at the build
    boundary and harmless on a disposable copy, but a caller that needs to
    assert on an UNREPAIRED brain must not share one with a caller that came
    through here.
    """
    import collections
    from servers import embedder
    from servers.schema import (BRAIN_VERSION, BRAIN_VERSION_KEY,
                                read_meta_value, read_schema_version)
    from servers.seed_pack import (COMMUNITY_SLUG, SEED_EDGES, SEED_NODES,
                                   SEED_PACK_GENERATION)

    def refuse(msg):
        raise RuntimeError('seed template is not complete — %s' % msg)

    # Does every copy open against the ONE shared model cache? Without the
    # pin, cache_dir falls back to db_path's directory — a fresh mkdtemp per
    # test — and load_model's singleton guard stops holding, so each seeded
    # test builds a second ONNX session for the same pinned artifact. The
    # only symptom is a suite ~43s slower, which nothing else would catch:
    # a perf property needs a gate exactly because a red test is the only
    # thing that makes slowness visible.
    pinned = brain.get_config('embedder_cache_dir', None)
    if pinned != _SHARED_MODEL_CACHE:
        refuse('embedder_cache_dir is %r, want %r — every test would seed and '
               'load its own copy of the model' % (pinned, _SHARED_MODEL_CACHE))

    # Was the pack fully seeded? Producer semantics, deliberately:
    # seed_baby_brain treats an ARCHIVED seed as PRESENT ("an archived seed is
    # retired, not missing") and its own gap-fill decision counts the same
    # way, so this carries no archived filter. A stricter count here would
    # refuse templates the loader considers finished.
    #
    # seed_pack failures are caught and only WARNed inside Brain(), so
    # construction succeeding is not evidence that seeding happened.
    seeded = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE encoding_source = 'anchor:seed'"
    ).fetchone()[0]
    if seeded != len(SEED_NODES):
        refuse(
            "%d of %d seed nodes present. At zero, seed_baby_brain did not run "
            "at all (servers/brain.py gates it on `not skip_embedder`); short "
            "of the full pack, it stopped partway. Refusing to hand tests the "
            "blind-priors graph the seed pack exists to prevent."
            % (seeded, len(SEED_NODES)))

    # Does it hand tests the whole LIVE graph? A different question from the
    # one above, so a different instrument: filter_nodes is the owner door for
    # live node counts and excludes archived rows. If a seed were ever
    # archived at build time the check above passes and this one fails —
    # which is the honest report: the pack is complete, the fixture is not.
    live = (brain.filter_nodes(field='created_at', rich=False,
                               limit=1) or {}).get('total_count')
    if live != len(SEED_NODES):
        refuse("%s live nodes against %d seed nodes — the fixture hands out a "
               "different graph than the pack defines"
               % (live, len(SEED_NODES)))

    # Edges PER RELATION, never as one total: a total would stay green while
    # the typed edges and the community membership traded places. Derived the
    # same way seed_baby_brain writes them — SEED_EDGES plus one generated
    # community_member per member.
    expected = collections.Counter(e['relation'] for e in SEED_EDGES)
    expected['community_member'] += sum(
        1 for n in SEED_NODES if n['slug'] != COMMUNITY_SLUG)
    actual = dict(brain.conn.execute(
        "SELECT relation, COUNT(*) FROM edge_relations GROUP BY relation"))
    if actual != dict(expected):
        refuse('edge relations do not match the seed pack — have %r, want %r'
               % (actual, dict(expected)))

    # The marker records WHICH pack seeded this brain — a fact titles cannot
    # carry, because packs share titles verbatim. Without it matching,
    # seed_baby_brain reads the brain as previous-generation and leaves it
    # untouched, so the template ships whatever it already held.
    marker = read_meta_value(brain.conn, 'brain_meta', 'seed_pack_generation')
    if marker != SEED_PACK_GENERATION:
        refuse('seeded by pack %r, code is at %r — seed_baby_brain leaves a '
               'foreign-marker brain untouched'
               % (marker, SEED_PACK_GENERATION))
    version = read_schema_version(brain.conn, 'brain_meta', BRAIN_VERSION_KEY)
    if version != BRAIN_VERSION:
        refuse('schema v%s against code v%s — migrations did not run, so every '
               'test in this process runs against the wrong shape'
               % (version, BRAIN_VERSION))

    node_vecs = brain.conn.execute(
        'SELECT COUNT(*) FROM node_enrichments WHERE embedding IS NOT NULL'
    ).fetchone()[0]
    edge_vecs = brain.conn.execute(
        'SELECT COUNT(*) FROM edge_relations WHERE embedding IS NOT NULL'
    ).fetchone()[0]

    if not vectorized:
        # The other direction, and it is a cost contract rather than a
        # correctness one: a brain that never loads the model has no query
        # encoder, so vectors it cannot read would be ~7s per process bought
        # for nothing. Undetected, that regression is silent and green — it
        # only makes the suite slower.
        if node_vecs or edge_vecs:
            refuse('built vectorized=False but carries %d node + %d edge '
                   'vectors — every test that never loads the model is paying '
                   'to embed what it cannot read'
                   % (node_vecs, edge_vecs))
        return

    # seed_baby_brain writes through remember(), which only ENQUEUES, so an
    # undrained brain has its nodes and zero embeddings. "The drain ran" is
    # not "the vectors are there": a dead embedder makes backfill_vectors
    # return {'error': ...} rather than raise, and _drain_once documents that
    # re-enqueued ids stay queued rather than being retried. So ask the
    # repair-and-report door, which on a just-drained template must find
    # NOTHING to do — anything it repaired, the drain left behind.
    sweep = brain.vector_coverage_sweep()
    if sweep.get('repaired') or sweep.get('remaining') or sweep.get('stuck'):
        refuse('%d vector(s) repaired on inspection (remaining=%s stuck=%s, '
               'by_type=%r). Two causes reach here: the drain never finished '
               '(embed_queue._drain_once is the producer — check '
               'embedder.get_model_status()), or the rows are at a stale '
               'model, which this door reads as missing and re-embeds.'
               % (sweep.get('repaired'), sweep.get('remaining'),
                  [n.get('id') for n in (sweep.get('stuck') or [])][:3],
                  sweep.get('by_type')))

    # Model-string identity, as the backstop it actually is. The door above is
    # already model-aware — it reads a stale-model row as MISSING — so with a
    # live embedder any swap is repaired and refused there first; measured, a
    # whole-table model swap comes back as 233 repaired, never as this branch.
    # What reaches here is drift the sweep cannot see: a stale model on a
    # non-`_primary` type while the embedder is down, since `stuck` probes
    # `_primary` alone. Narrow, one query, and the only assertion left if the
    # sweep's repair behaviour ever changes. NEVER a cosine comparison: batch
    # padding puts a healthy stored-vs-fresh re-embed at ~0.96-0.98, so any
    # threshold that catches real drift also fires on a correct template.
    models = {r[0] for r in brain.conn.execute(
        'SELECT DISTINCT model FROM node_enrichments '
        'WHERE embedding IS NOT NULL')}
    live_model = embedder.stats.get('model_name')
    if models != {live_model}:
        # repr per element, not sorted(): the column is nullable, and sorting
        # a set holding both None and a string raises from inside the very
        # message that exists to explain the mismatch.
        refuse('vectors were written by %s but the live model is %r — they '
               'score against geometry the query encoder does not share'
               % (sorted(map(repr, models)) or 'nothing', live_model))


def _seed_template(vectorized=True):
    """Build (once per process) a nursery-seeded brain; return its directory.

    Seeding needs the embedder — seed_baby_brain writes through
    brain.remember(), and servers/brain.py gates the call on
    `if not skip_embedder`. USING the result does not: every copy carries the
    seed nodes and their edges, and a `vectorized` copy carries their
    embeddings too, so a test opens it with skip_embedder=True and still sees
    a whole graph.

    `vectorized` (default True) drains the embed queue before snapshotting, so
    node_enrichments and edge_relations.embedding are populated and no copy
    has to derive them. It costs ~7.4s against ~1.3s, once per process. Pass
    False for a brain that never loads the model: without a query encoder
    stored vectors cannot be scored against, so it would pay for something it
    cannot read. SeedBrain picks per its own `needs_embedder`.

    The two files are checkpointed through the backup API so each is
    self-contained (committed WAL tail folded in) — a per-test copy is two
    copy2 calls with no -wal/-shm juggling.
    """
    cached = _SEED_TEMPLATE_DIRS.get(vectorized)
    if cached and os.path.isfile(os.path.join(cached, 'brain.db')):
        return cached

    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    build_dir = tempfile.mkdtemp(prefix='brain_seed_build_')
    tpl_dir = tempfile.mkdtemp(prefix='brain_seed_tpl_')

    # Brain() resolves all three at call time. Pin them at the build dir so
    # seeding can never read or heal the operator's live aspects file, and so
    # the build's scratch files land somewhere we delete.
    saved = {k: os.environ.get(k) for k in
             ('BRAIN_DB_DIR', 'ASPECTS_JSON_PATH', 'BRAIN_TMP_DIR')}
    os.environ['BRAIN_DB_DIR'] = build_dir
    os.environ['ASPECTS_JSON_PATH'] = os.path.join(build_dir, 'aspects_v1.json')
    os.environ['BRAIN_TMP_DIR'] = build_dir
    brain = None
    try:
        from servers.brain import Brain
        from servers.db_backends import current as db_backend
        brain = Brain(os.path.join(build_dir, 'brain.db'), skip_embedder=False)
        # Into the template's brain_meta, so every copy opens against the one
        # shared cache instead of seeding its own. See _SHARED_MODEL_CACHE.
        brain.set_embedder_config(cache_dir=_SHARED_MODEL_CACHE)
        if vectorized:
            # seed_baby_brain writes through remember(), which only ENQUEUES,
            # so an undrained brain has its nodes and zero embeddings. Drain
            # before the snapshot or every copy pays it again — and pays it
            # too late, because the harness drains AFTER the call it wraps,
            # leaving a test's first recall() to run against an empty store.
            from servers import embed_queue, recall_write_queue
            embed_queue._drain_once(brain)
            recall_write_queue.drain_once(brain)
        # One definition of complete, and this is the boundary where a failure
        # costs least: refusing here fails every test in the process loudly,
        # instead of handing out a fixture whose defects read as quiet
        # zero-hit recalls. The contract test calls the same function against
        # an opened copy — see _assert_template_complete.
        _assert_template_complete(brain, vectorized)
        brain.save()
        brain.close()
        brain = None
        for name in ('brain.db', 'brain_logs.db'):
            db_backend.snapshot_to(os.path.join(build_dir, name),
                                   os.path.join(tpl_dir, name))
    except BaseException:
        # tpl_dir is created before the build, so a failure here would orphan
        # it exactly as a failed __enter__ would orphan a db_dir.
        shutil.rmtree(tpl_dir, ignore_errors=True)
        raise
    finally:
        # Close on the raise paths too: the finally below deletes build_dir,
        # and an open connection would be left holding deleted files.
        if brain is not None:
            try:
                brain.close()
            except Exception:
                pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(build_dir, ignore_errors=True)

    # Nothing else deletes tpl_dir — it outlives every individual test by
    # design, so the process exit is the only correct cleanup point.
    atexit.register(shutil.rmtree, tpl_dir, True)
    _SEED_TEMPLATE_DIRS[vectorized] = tpl_dir
    return tpl_dir


class SeedBrain(BrainEnv):
    """A test brain born with the nursery seed pack — never the operator's.

    This is the substrate for tests. It is ~1000x smaller than a production
    clone (about 2 MB against 2 GB) and is identical on every machine and
    every run.

    Cost: the FIRST SeedBrain in a process builds the template — a real
    seeded Brain, ~5s, most of it the one-time embedder load the process
    shares with everything else. Every SeedBrain after it is a ~3ms file
    copy. A single-test run pays the build; a file or a suite amortizes it.

    It is also the safer substrate. A production clone carries the
    operator's `brain_meta` globals — `session_context` most of all — into
    tests that write their own; that leak once made the encoder decline to
    encode a whole eval corpus because the operator's session context was
    alien to the transcripts under test.

        with SeedBrain() as env:
            env.brain          # the seed pack's nodes and edges, vectors intact
            env.dispatch(...)  # the daemon's own execution chokepoint

    The seed graph's SHAPE is fixed — same nodes, same titles, same edges on
    every run — but its node IDs are minted when the template is built, so
    they are stable within one pytest process and different across processes.
    Select a seed node by title, never by a hardcoded id.
    """

    # The template is drained at build time, so its seed vectors are stored
    # and opening a copy needs no model. Pass needs_embedder=True for tests
    # that embed NEW text — or that call recall(), which needs the model to
    # encode the QUERY even when every document vector is already present.
    needs_embedder = False

    # Whether the brain is born with the nursery pack. False provisions
    # NOTHING — `Brain()` creates an empty database — which is what a test
    # wants when its subject is the empty-graph path, or when a 26-node prior
    # would change what its assertions mean. It is not a third template: an
    # unseeded brain has nothing to snapshot, so the cache stays at two.
    seeded = True

    def __init__(self, cleanup=True, load_env=True, needs_embedder=None,
                 seeded=None):
        # load_env's default is restated here only to keep the signature
        # readable; it must stay the base's. A substrate that silently
        # differed on .env would make an LLM-calling test fail far from the
        # cause, or degrade into asserting on empty output.
        super().__init__(cleanup=cleanup, load_env=load_env)
        if needs_embedder is not None:
            self.needs_embedder = needs_embedder
        if seeded is not None:
            self.seeded = seeded

    def _provision(self):
        if not self.seeded:
            # Nothing to copy. Brain() creates brain.db here, and seeds a
            # fresh logs schema beside it because brain_logs.db is absent —
            # the same empty pair a raw Brain(tmp) has always produced.
            return
        # Only a brain that loads the model can USE stored vectors — without
        # it there is no query encoder, so recall cannot score against them.
        # Asking for the drained template in that case would buy nothing and
        # cost ~6s of embedding per process.
        tpl = _seed_template(vectorized=self.needs_embedder)
        for name in ('brain.db', 'brain_logs.db'):
            shutil.copy2(os.path.join(tpl, name),
                         os.path.join(self.db_dir, name))

    def _seed_aspects_file(self):
        """No-op — a seed brain must not inherit the operator's aspects.

        Brain's AspectRegistry seeds the pinned path from the repo seed via
        reconcile_working_copy, which is the operator-agnostic source a seed
        brain should read.
        """
        return


def brain_env(*, needs_embedder=None, seeded=None, **kwargs):
    """The one door a test gets a brain through.

    Tests call this rather than naming a class, so the answer to "what is a
    test brain?" lives here instead of at every call site. Changing the
    substrate is then one line, not one edit per test file.

    It hands back a SeedBrain: a nursery-seeded brain stamped from a ~2 MB
    template that carries its own vectors. There is no second substrate to
    choose — the production clone lives in `eval/isolated_brain.py` and is
    unreachable from `tests/` by design (see the module docstring).

    A test that calls recall() and expects hits belongs HERE, with
    needs_embedder=True: the template carries document vectors and the model
    encodes the query. Pick a query the 26-node nursery pack can answer. This
    is ENFORCED rather than advised — recall on a needs_embedder=False brain
    raises instead of returning an empty list, because an empty list is also
    what "nothing matched" looks like and the two are indistinguishable at the
    call site. `brain.recall.__wrapped__()` is the deliberate way past it.

    `seeded=False` gives an EMPTY graph rather than the 26-node pack — for a
    test whose subject is the empty-graph path, or one whose assertions a seed
    prior would change. It provisions nothing at all, so it is cheaper than
    either template rather than a third one. It requires needs_embedder=False:
    servers/brain.py welds seeding to the embedder, so a brain that loads the
    model is seeded whether or not this factory asks for it, and the factory
    raises rather than accept an argument it cannot honour.

    Not named test_brain(): pytest collects module-level `test_*` callables,
    so importing it into a test module would make pytest try to run the
    factory itself as a test case.
    """
    if seeded is False and needs_embedder:
        # Not an arbitrary restriction — `servers/brain.py` gates seeding on
        # `if not skip_embedder`, so a brain that loads the model IS seeded by
        # construction and this factory cannot honour the combination. Raising
        # says so; returning a seeded brain would make the argument a lie.
        # Splitting the two is its own step, with its own blast radius.
        raise TypeError(
            "seeded=False needs needs_embedder=False — seeding is welded to "
            "the embedder in servers/brain.py (`if not skip_embedder: "
            "seed_baby_brain(self)`), so a brain that loads the model is "
            "seeded whether or not this factory asks for it.")
    return SeedBrain(needs_embedder=needs_embedder, seeded=seeded, **kwargs)


def _load_env():
    """Load .env for API keys."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
