"""Tests for SessionContext — per-request session identity.

SessionContext flows with every brain call. The brain doesn't own sessions.
Every hook, MCP call, and encoding run receives a SessionContext.

Run: python3 -m pytest tests/test_session_context.py -v
"""
import pytest
from tests.isolated_brain import brain_env


# ═══════════════════════════════════════════════════════
# SessionContext object
# ═══════════════════════════════════════════════════════

class TestSessionContextCreation:
    """Verify SessionContext can be created and carries identity."""

    def test_create_with_id(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc123')
        assert ctx.session_id == 'abc123'

    def test_missing_session_id_uses_fallback(self):
        """An empty session_id gets a generated fallback."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='')
        assert ctx.session_id  # not empty
        assert len(ctx.session_id) > 0

    def test_session_id_short(self):
        """Short form for chain IDs — the last 12 hyphen-free chars; for a
        UUID that is exactly its last group."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='01a122b6-9059-7ca1-bb1b-a5e6ff4ce0cf')
        assert ctx.session_short == 'a5e6ff4ce0cf'

    def test_uuidv7_same_minute_chains_differ(self):
        """Codex ids are UUIDv7: two chats opened in the same minute share
        their first 8 chars (the timestamp). Their chains must not."""
        from servers.session_context import SessionContext
        a = SessionContext('01a122b6-9059-7ca1-bb1b-a5e6ff4ce0cf', stop_counter=3)
        b = SessionContext('01a122b6-f3c2-7d10-8e4a-17c09b2d5e61', stop_counter=3)
        assert a.session_id[:8] == b.session_id[:8]
        for gen in ('s0_chain', 's1r_chain', 's1e_chain'):
            assert getattr(a, gen)() != getattr(b, gen)()

    def test_short_strips_hyphens_before_cutting(self):
        """Chain parsers split on '-', so the short never carries one — even
        when the cut would land inside a non-UUID id. Shorter ids come back
        whole."""
        from servers.session_context import SessionContext
        from servers.scales.s1.trace_links import _stop_of
        ctx = SessionContext('i1a2b3c-e47becba-s0', stop_counter=7)
        assert '-' not in ctx.session_short
        assert ctx.s0_chain().split('-') == ['s0', ctx.session_short, '7']
        assert _stop_of(ctx.s1e_chain()) == 7
        assert SessionContext('sess-1').session_short == 'sess1'

    def test_stop_counter(self):
        """SessionContext carries stop counter."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc', stop_counter=42)
        assert ctx.stop_counter == 42

    def test_stop_counter_default(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc')
        assert ctx.stop_counter == 0

    def test_increment_stop(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abc', stop_counter=5)
        ctx.increment_stop()
        assert ctx.stop_counter == 6


# ═══════════════════════════════════════════════════════
# SessionContext persistence
# ═══════════════════════════════════════════════════════

class TestSessionContextPersistence:
    """Verify SessionContext can be saved to and loaded from DB."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            yield

    def test_save_and_load(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='persist-test', stop_counter=10)
        ctx.save(self.brain._session_state)

        loaded = SessionContext.load(self.brain._session_state, 'persist-test')
        assert loaded is not None
        assert loaded.session_id == 'persist-test'
        assert loaded.stop_counter == 10

    def test_delivery_stamp_roundtrips(self):
        # last_delivery_stop mirrors last_recall_stop through save/load. What
        # this guarantees is the SAVE/LOAD contract only: session state
        # reaches disk via the session flush (not brain.save()), so a crash
        # inside the flush window between a delivery-block and its
        # continuation degrades the reaction to a heartbeat — a documented
        # limitation, not a corruption (the stamp is consumed-on-match, so a
        # stale value can never mis-classify a later stop either way).
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='dlv-persist', stop_counter=4)
        ctx.last_delivery_stop = 4
        ctx.last_delivery_armed_at = '2026-09-03T10:00:00+00:00'
        ctx.save(self.brain._session_state)
        loaded = SessionContext.load(self.brain._session_state, 'dlv-persist')
        assert loaded.last_delivery_stop == 4
        assert loaded.last_delivery_armed_at == '2026-09-03T10:00:00+00:00'

    def test_load_nonexistent_returns_none(self):
        from servers.session_context import SessionContext
        loaded = SessionContext.load(self.brain._session_state, 'nonexistent')
        assert loaded is None

    def test_load_corrupt_blob_raises_typed_signal(self):
        # A corrupt session_state blob is SIGNALED (SessionContextCorrupt), not
        # silently None — the brain-holding caller catches it and logs via the
        # canonical _log_error. Absent rows still return None (test above).
        from servers.session_context import SessionContext, SessionContextCorrupt
        from servers.dal_logs import SessionStateDAL
        SessionStateDAL(self.brain.logs_conn).set('corrupt-sess', '_session_context', 'not json{')
        with pytest.raises(SessionContextCorrupt):
            SessionContext.load(self.brain._session_state, 'corrupt-sess')
        # the brain caller catches the signal and degrades gracefully (no crash)
        assert self.brain.session_env_for('corrupt-sess') == {
            'cwd': '', 'branch': '', 'worktree': '', 'project': '',
            'model': '', 'host': ''}

    def test_save_updates_existing(self):
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='update-test', stop_counter=5)
        ctx.save(self.brain._session_state)

        ctx.stop_counter = 15
        ctx.save(self.brain._session_state)

        loaded = SessionContext.load(self.brain._session_state, 'update-test')
        assert loaded.stop_counter == 15

    def test_touched_persists_with_the_row(self):
        # A restart mid-turn keeps the touches made before it: the Stop flush
        # writes them as that turn's anchor_touched trace.
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='touched-persist')
        ctx.touched['created'].append('abcd1234')
        ctx.touched['looked_up'].extend(['e1f2a3b4', 'c5d6e7f8'])
        ctx.save(self.brain._session_state)
        loaded = SessionContext.load(self.brain._session_state, 'touched-persist')
        assert loaded.touched == ctx.touched

    def test_row_without_touched_loads_empty_contract_keys(self):
        from servers.dal_logs import SessionStateDAL
        from servers.session_context import SessionContext
        from servers.trace_contract import ANCHOR_TOUCHED_KEYS
        SessionStateDAL(self.brain.logs_conn).set(
            'pre-touched-row', '_session_context', '{"stop_counter": 3}')
        loaded = SessionContext.load(self.brain._session_state, 'pre-touched-row')
        assert loaded.stop_counter == 3
        assert loaded.touched == {k: [] for k in ANCHOR_TOUCHED_KEYS}

    def test_flush_writes_only_changed_sessions(self):
        # The periodic flush rewrites a session's row only when its state
        # changed since the last write — an idle session's row, updated_at
        # included, is left alone.
        from unittest.mock import patch
        from servers.session_context import SessionContext
        b = self.brain
        changed = b.get_or_create_session('flush-changed')
        b.get_or_create_session('flush-idle')
        b.save_session_contexts()                     # both differ from the seeded default row

        def stamp(sid):
            return b.logs_conn.execute(
                "SELECT updated_at FROM session_state "
                "WHERE session_id=? AND key='_session_context'", (sid,)).fetchone()[0]
        idle_before = stamp('flush-idle')
        changed.increment_stop()
        with patch.object(b._session_state, 'set', wraps=b._session_state.set) as m:
            assert b.save_session_contexts() == 1
            assert b.save_session_contexts() == 0    # nothing changed since
        assert [c.args[0] for c in m.call_args_list] == ['flush-changed']
        assert stamp('flush-idle') == idle_before
        assert SessionContext.load(b._session_state, 'flush-changed').stop_counter == 1

    def test_save_and_load_cwd_branch(self):
        # cwd/branch/worktree/project are session identity (fed from the boot
        # hook) — they must round-trip through the JSON blob like the other fields.
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='env-test')
        ctx.cwd = '/work/tree/x'
        ctx.branch = 'claude/x'
        ctx.worktree = 'emb-bench'
        ctx.project = 'brain'
        ctx.save(self.brain._session_state)
        loaded = SessionContext.load(self.brain._session_state, 'env-test')
        assert loaded.cwd == '/work/tree/x'
        assert loaded.branch == 'claude/x'
        assert loaded.worktree == 'emb-bench'
        assert loaded.project == 'brain'

    def test_set_env_mutator(self):
        # set_env is the single per-session env stamper. cwd/branch refresh only on
        # a non-empty value; worktree/project use a None sentinel so '' can CLEAR
        # (WorktreeRemove / non-repo) distinct from "leave unchanged".
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='setenv-test')
        ctx.set_env(cwd='/a', branch='b', worktree='wt', project='brain')
        assert (ctx.cwd, ctx.branch, ctx.worktree, ctx.project) == \
            ('/a', 'b', 'wt', 'brain')
        ctx.set_env(cwd='', branch='')               # empty → leave cwd/branch
        assert (ctx.cwd, ctx.branch) == ('/a', 'b')
        ctx.set_env(cwd='/c')                    # worktree/project None → unchanged
        assert ctx.worktree == 'wt'
        assert ctx.project == 'brain'
        ctx.set_env(worktree='', project='')           # '' → explicit clear
        assert ctx.worktree == ''
        assert ctx.project == ''

    def test_reset_session_activity_stamps_cwd(self):
        # boot feeds cwd → a NEW session's reset stamps cwd + derived branch.
        from servers.session_context import SessionContext
        is_resume = self.brain.reset_session_activity(session_id='env-sess', cwd='/work/tree/y')
        assert is_resume is False                    # never booted → new session
        after = SessionContext.load(self.brain._session_state, 'env-sess')
        assert after.cwd == '/work/tree/y'          # stamped from the boot feed
        assert after.branch                          # derived (real branch or 'unknown')

    def test_worktree_from_gitdir_marker(self):
        # Unit-test the git-dir parser against the loose-marker edge cases — a repo
        # whose OWN path contains a 'worktrees' segment. Anchored on
        # '.git/worktrees/' + last-occurrence split, so neither a main tree nor a
        # linked tree under such a path mis-resolves. (A bare '/worktrees/' marker
        # mis-parsed both — the bug this test pins.)
        from servers.session_env import worktree_from_gitdir as f
        assert f('.git') == ''                                             # main, relative
        assert f('/Users/t/brain/.git') == ''                             # main, absolute (subdir)
        assert f('/Users/t/brain/.git/worktrees/emb-bench') == 'emb-bench'  # linked
        assert f('/x/worktrees/repo/.git') == ''                          # main UNDER worktrees/
        assert f('/x/worktrees/repo/.git/worktrees/wt1') == 'wt1'          # linked UNDER worktrees/

    def test_project_from_common_dir(self):
        # Unit-test the common-dir → project parser. The common dir is the SAME
        # from the main tree and every linked worktree, so project is stable
        # across checkouts. Relative output ('.git' from the repo root) resolves
        # against cwd; submodule git-dirs resolve to the superproject; unknown
        # shapes → '' (never a crash, never a wrong slug).
        from servers.session_env import project_from_common_dir as f
        assert f('/Users/t/brain/.git', '/anywhere') == 'brain'           # absolute
        assert f('.git', '/Users/t/brain') == 'brain'                     # relative → cwd
        assert f('/Users/t/brain/.git/modules/sub', '/x') == 'brain'      # submodule
        assert f('/x/worktrees/repo/.git', '/y') == 'repo'                # repo under worktrees/
        assert f('', '/x') == ''                                          # no output
        assert f('gitdir-without-marker', '/x') == ''                     # unrecognized

    def test_detect_git_env_and_stamp_worktree_real(self):
        # Real verification of boot-time derivation: build a temp repo + linked
        # worktree, confirm detect_git_env returns (branch, worktree) — '' worktree
        # for the main tree, the NAME for the linked checkout, and (‘unknown’, None)
        # on git failure (None lets set_env KEEP a known worktree on a flaky
        # resume) — then that reset_session_activity stamps it onto the session.
        import shutil, subprocess, tempfile, os
        if not shutil.which('git'):
            pytest.skip('git not available')
        root = tempfile.mkdtemp()
        try:
            repo = os.path.join(root, 'repo')
            os.makedirs(repo)
            def git(*a):
                subprocess.run(['git', '-C', repo, *a], check=True,
                               capture_output=True, text=True)
            git('init', '-q')
            git('config', 'user.email', 't@t')
            git('config', 'user.name', 't')
            git('commit', '--allow-empty', '-m', 'init', '-q')
            wt = os.path.join(root, 'wt-emb-bench')
            git('worktree', 'add', '-q', '-b', 'feature', wt)

            assert self.brain.detect_git_env(repo)[1] == ''             # main tree → ''
            assert self.brain.detect_git_env(wt)[1] == 'wt-emb-bench'   # linked → name
            # project = main repo dir name — SAME from main tree and worktree
            assert self.brain.detect_git_env(repo)[2] == 'repo'
            assert self.brain.detect_git_env(wt)[2] == 'repo'
            assert self.brain.detect_git_env('/no/such/dir') == \
                ('unknown', None, None)                                 # failed → keep

            self.brain.reset_session_activity(session_id='wt-sess', cwd=wt)
            env = self.brain.session_env_for('wt-sess')
            assert env['worktree'] == 'wt-emb-bench'
            assert env['project'] == 'repo'
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_project_resolution_marker_and_basename(self):
        # The host-adapter resolution chain (session_env): marker file beats
        # git beats cwd basename; junk anchors resolve to '' (unscoped);
        # malformed markers are skipped, never fatal.
        import shutil, subprocess, tempfile, os
        from servers.session_env import detect_session_env, \
            project_from_cwd_basename
        if not shutil.which('git'):
            pytest.skip('git not available')
        root = tempfile.mkdtemp()
        try:
            # Non-repo folder → basename fallback (the Slack-session fix:
            # a real working folder must not stamp project='').
            plain = os.path.join(root, 'slack-watch')
            os.makedirs(plain)
            assert detect_session_env(plain) == ('unknown', '', 'slack-watch')

            # Marker beats basename; marker in a PARENT found by the walk.
            with open(os.path.join(plain, '.brain-project'), 'w') as f:
                f.write('slack\n')
            assert detect_session_env(plain)[2] == 'slack'
            child = os.path.join(plain, 'sub', 'dir')
            os.makedirs(child)
            assert detect_session_env(child)[2] == 'slack'

            # Marker beats git — rename-stability for repos.
            repo = os.path.join(root, 'renamed-checkout')
            os.makedirs(repo)
            subprocess.run(['git', '-C', repo, 'init', '-q'], check=True)
            subprocess.run(['git', '-C', repo, '-c', 'user.email=t@t',
                            '-c', 'user.name=t', 'commit', '--allow-empty',
                            '-m', 'init', '-q'], check=True)
            assert detect_session_env(repo)[2] == 'renamed-checkout'
            with open(os.path.join(repo, '.brain-project'), 'w') as f:
                f.write('brain\n')
            assert detect_session_env(repo)[2] == 'brain'

            # Malformed marker → STOPS the marker search (nearest explicit
            # intent must not lose to a farther ancestor), falls through to
            # git/basename.
            bad = os.path.join(root, 'bad-marker')
            os.makedirs(bad)
            with open(os.path.join(bad, '.brain-project'), 'w') as f:
                f.write('../evil path\n')
            assert detect_session_env(bad)[2] == 'bad-marker'
            # Ancestor-shadow: an invalid nearer marker must NOT be beaten
            # by a valid ancestor marker ('slack' at plain/) — the child
            # resolves by basename instead.
            bad_child = os.path.join(plain, 'bad-child')
            os.makedirs(bad_child)
            with open(os.path.join(bad_child, '.brain-project'), 'w') as f:
                f.write('two words\n')
            assert detect_session_env(bad_child)[2] == 'bad-child'

            # Junk anchors → None (KEEP what the session has — a re-boot
            # whose cwd lands on $HOME/tmp/Downloads says nothing about
            # intent and must not wipe known provenance). Named non-repo
            # folders assert their identity; anchors don't.
            downloads = os.path.join(root, 'Downloads')
            os.makedirs(downloads)
            assert detect_session_env(downloads)[2] is None
            assert project_from_cwd_basename(os.path.expanduser('~')) == ''
            assert project_from_cwd_basename('/tmp') == ''
            assert project_from_cwd_basename('/') == ''
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_resume_git_failure_preserves_worktree(self):
        # REGRESSION: a re-boot whose git probe fails must NOT wipe a worktree the
        # session already had. detect_git_env returns worktree=None on failure, and
        # set_env's None-sentinel leaves the field unchanged.
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='wt-resume')
        ctx.boot_time = self.brain.now()       # mark booted → next reset is a RESUME
        ctx.worktree = 'feature-x'
        ctx.save(self.brain._session_state)
        self.brain._session_contexts['wt-resume'] = ctx
        # cwd that git can't resolve → detect_git_env → (‘unknown’, None)
        self.brain.reset_session_activity(session_id='wt-resume', cwd='/no/such/dir')
        assert self.brain.session_env_for('wt-resume')['worktree'] == 'feature-x'

    def test_reset_session_activity_resume_preserves_state(self):
        # REGRESSION: a re-boot of an already-booted session is a RESUME — it must
        # CONTINUE accumulated state, not zero it. Pre-fix, every boot built a
        # fresh ctx, so under parallel sessions the global resume-guard misfired
        # and reset live sessions: stop_counter→0 (duplicate chain IDs) and fatigue
        # lost. Resume detection now lives in the session object
        # (ctx.boot_time). (The Scribe cadence itself is trace-derived now, so it's
        # immune regardless — see test_turns_since_last_encode_trace_pull.)
        from servers.session_context import SessionContext
        assert self.brain.reset_session_activity(session_id='resume-sess', cwd='/w/a') is False
        ctx = self.brain._session_contexts['resume-sess']
        ctx.stop_counter = 7
        ctx.fatigue['n1'] = 2
        # Re-boot the SAME session (resume) — different cwd to prove identity still
        # refreshes while accumulated state is preserved.
        assert self.brain.reset_session_activity(session_id='resume-sess', cwd='/w/b') is True
        after = SessionContext.load(self.brain._session_state, 'resume-sess')
        assert after.stop_counter == 7               # chain-id sequence preserved (no dup chains)
        assert after.fatigue == {'n1': 2}
        assert after.cwd == '/w/b'                    # identity refreshed on resume

    def test_resume_after_daemon_restart_loads_from_db(self):
        # A daemon restart empties the in-memory cache but the persisted row
        # survives. The next boot must still detect a resume (from the DB row) and
        # preserve state — the suspend-restart path that was resetting live
        # sessions in production.
        from servers.session_context import SessionContext
        self.brain.reset_session_activity(session_id='restart-sess', cwd='/w/a')
        ctx = self.brain._session_contexts['restart-sess']
        ctx.stop_counter = 9
        ctx.save(self.brain._session_state)               # persist before the "restart"
        self.brain._session_contexts.clear()         # simulate daemon restart
        assert self.brain.reset_session_activity(session_id='restart-sess', cwd='/w/a') is True
        after = SessionContext.load(self.brain._session_state, 'restart-sess')
        assert after.stop_counter == 9               # survived the restart + re-boot

    def test_turns_since_last_encode_trace_pull(self):
        # The S1 Scribe cadence is derived LIVE from traces — no stored counter.
        # Pins the definition: a turn == one s0 user_message; <task-notification>
        # ignitions don't count; only turns after the last SUCCESSFUL encode
        # (encoding_run delta) count, anchored at that run's START (its chain's
        # encoding_prompt). A failed run (prompt, no delta) must NOT reset the
        # cadence — failed runs stay due and get retried (2026-07-28).
        sid = 'cadence-sess'
        c = self.brain.logs_conn
        _n = [0]
        def ins(scale, etype, ref_type, ts, summary='hi', chain=None):
            _n[0] += 1
            c.execute(
                "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, "
                "ref_id, summary, metadata, session_id, interaction_id, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ('t%07d' % _n[0], chain or '%s-x-%d' % (scale, _n[0]), scale, etype,
                 ref_type, '', summary, None, sid, None, ts))
        # 3 turns, no encode yet → counts all 3
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:01+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:02+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:03+00:00')
        c.commit()
        assert self.brain.turns_since_last_encode(sid) == 3
        # SUCCESSFUL encode (prompt + run delta on one chain), then 2 real
        # turns + 1 task-notification ignition after it
        ins('s1', 'O', 'encoding_prompt', '2026-06-13T10:00:04+00:00', chain='s1e-cad-1')
        ins('s1', 'delta', 'encoding_run', '2026-06-13T10:00:04.900000+00:00', chain='s1e-cad-1')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:05+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:06+00:00')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:07+00:00', summary='<task-notification> wake')
        c.commit()
        assert self.brain.turns_since_last_encode(sid) == 2   # ignition excluded; only real turns since encode
        # FAILED encode (prompt, no run delta) — cadence does NOT reset; the
        # count keeps growing so the session stays due for a retry.
        ins('s1', 'O', 'encoding_prompt', '2026-06-13T10:00:08+00:00', chain='s1e-cad-2')
        ins('s0', 'K', 'user_message', '2026-06-13T10:00:09+00:00')
        c.commit()
        assert self.brain.turns_since_last_encode(sid) == 3   # 2 pre-fail + 1 post-fail

    def test_parallel_sessions_count_cadence_independently(self):
        # The cross-session fix: the Scribe cadence is per-session, derived live
        # from traces. With interleaved turns from two parallel streams, each
        # session must count ONLY its own turns since ITS OWN last encode — and
        # one session encoding must NOT reset the other's count. Pre-fix, a
        # global conversational_count was clobbered last-writer-wins, so a busy
        # stream could starve a quiet one (or fire it early). This pins that the
        # two streams never see each other's cadence.
        a, b = 'stream-A', 'stream-B'
        c = self.brain.logs_conn
        _n = [0]

        def ins(sid, scale, etype, ref_type, ts, summary='hi', chain=None):
            _n[0] += 1
            c.execute(
                "INSERT INTO trace_events (id, chain_id, scale, event_type, ref_type, "
                "ref_id, summary, metadata, session_id, interaction_id, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ('p%07d' % _n[0], chain or '%s-%s-%d' % (scale, sid[:6], _n[0]),
                 scale, etype, ref_type, '', summary, None, sid, None, ts))

        # Interleave: A, B, A, B, A → A has 3 turns, B has 2, in real time order.
        ins(a, 's0', 'K', 'user_message', '2026-06-13T10:00:01+00:00')
        ins(b, 's0', 'K', 'user_message', '2026-06-13T10:00:02+00:00')
        ins(a, 's0', 'K', 'user_message', '2026-06-13T10:00:03+00:00')
        ins(b, 's0', 'K', 'user_message', '2026-06-13T10:00:04+00:00')
        ins(a, 's0', 'K', 'user_message', '2026-06-13T10:00:05+00:00')
        c.commit()
        # Each sees only its own turns — A's 3 are invisible to B and vice versa.
        assert self.brain.turns_since_last_encode(a) == 3
        assert self.brain.turns_since_last_encode(b) == 2

        # A encodes SUCCESSFULLY (prompt + run delta on one chain). This must
        # reset ONLY A's cadence — B's count is untouched.
        ins(a, 's1', 'O', 'encoding_prompt', '2026-06-13T10:00:06+00:00', chain='s1e-A-1')
        ins(a, 's1', 'delta', 'encoding_run', '2026-06-13T10:00:06.900000+00:00', chain='s1e-A-1')
        c.commit()
        assert self.brain.turns_since_last_encode(a) == 0   # A's backlog cleared
        assert self.brain.turns_since_last_encode(b) == 2   # B unaffected by A's encode

        # More interleaved turns after A's encode: A +2, B +1.
        ins(a, 's0', 'K', 'user_message', '2026-06-13T10:00:07+00:00')
        ins(b, 's0', 'K', 'user_message', '2026-06-13T10:00:08+00:00')
        ins(a, 's0', 'K', 'user_message', '2026-06-13T10:00:09+00:00')
        c.commit()
        assert self.brain.turns_since_last_encode(a) == 2   # only post-encode turns
        assert self.brain.turns_since_last_encode(b) == 3   # all B turns; B never encoded

    def test_multiple_sessions_isolated(self):
        """Two sessions don't interfere with each other."""
        from servers.session_context import SessionContext
        ctx1 = SessionContext(session_id='sess-1', stop_counter=10)
        ctx2 = SessionContext(session_id='sess-2', stop_counter=20)
        ctx1.save(self.brain._session_state)
        ctx2.save(self.brain._session_state)

        loaded1 = SessionContext.load(self.brain._session_state, 'sess-1')
        loaded2 = SessionContext.load(self.brain._session_state, 'sess-2')
        assert loaded1.stop_counter == 10
        assert loaded2.stop_counter == 20


class TestScribeStarvationThreshold:
    """The Scribe-starvation alarm is a pure threshold decision (level trigger,
    rate-limited). The gate (daemon_hooks) logs a loud brain error when it trips —
    the monitor that would have caught the 20h encode-drought on hour one."""

    def test_below_threshold_not_starved(self):
        from servers.scales.s1.encode_contract import (
            scribe_is_starved, SCRIBE_STARVATION_TURNS, ENCODE_EVERY)
        assert SCRIBE_STARVATION_TURNS == 4 * ENCODE_EVERY
        assert not scribe_is_starved(0)
        assert not scribe_is_starved(ENCODE_EVERY)               # normal: gate fires, no alarm
        assert not scribe_is_starved(SCRIBE_STARVATION_TURNS - 1)

    def test_at_threshold_then_rate_limited(self):
        from servers.scales.s1.encode_contract import (
            scribe_is_starved, SCRIBE_STARVATION_TURNS, ENCODE_EVERY)
        assert scribe_is_starved(SCRIBE_STARVATION_TURNS)                  # first alert at threshold
        assert not scribe_is_starved(SCRIBE_STARVATION_TURNS + 1)          # rate-limited between cadences
        assert scribe_is_starved(SCRIBE_STARVATION_TURNS + ENCODE_EVERY)   # next alert one cadence later

    def test_cadence_of_one_still_rate_limits(self):
        # BRAIN_ENCODE_EVERY is operator-settable; at 1 an unfloored modulo is
        # true every turn, turning the rate limit into the flood it prevents.
        from unittest.mock import patch
        from servers.scales.s1 import encode_contract as ec
        with patch.object(ec, 'ENCODE_EVERY', 1):
            assert ec.scribe_is_starved(ec.SCRIBE_STARVATION_TURNS)
            assert not ec.scribe_is_starved(ec.SCRIBE_STARVATION_TURNS + 1)


# ═══════════════════════════════════════════════════════
# SessionContext in trace writes
# ═══════════════════════════════════════════════════════

class TestSessionContextInTraces:
    """Verify traces use SessionContext for chain IDs and session tagging."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            self.dal = env.brain._trace_dal
            env.brain.logs_conn.execute('DELETE FROM trace_events')
            env.brain.logs_conn.commit()
            yield

    def test_chain_id_from_context(self):
        """Chain IDs use session_short and stop from context."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='abcdef1234567890', stop_counter=42)

        s0_chain = ctx.s0_chain()
        assert s0_chain == 's0-ef1234567890-42'

        s1r_chain = ctx.s1r_chain()
        assert s1r_chain == 's1r-ef1234567890-42'

        s1e_chain = ctx.s1e_chain()
        assert s1e_chain == 's1e-ef1234567890-42'

    def test_trace_write_with_context(self):
        """Trace events use session_id from context."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='trace-ctx-test', stop_counter=5)

        self.dal.append(
            chain_id=ctx.s0_chain(), scale='s0', event_type='K',
            ref_type='user_message', summary='test',
            session_id=ctx.session_id)

        events = self.dal.get_chain(ctx.s0_chain())
        assert len(events) == 1
        chain = self.dal.get_chains(session_id='trace-ctx-test', scale='s0')
        assert len(chain) == 1

    def test_consistent_across_stops(self):
        """Same session, different stops — different chains, same session_id."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='consist-test', stop_counter=1)

        self.dal.append(chain_id=ctx.s0_chain(), scale='s0', event_type='K',
                        ref_type='user_message', summary='stop 1',
                        session_id=ctx.session_id)
        ctx.increment_stop()
        self.dal.append(chain_id=ctx.s0_chain(), scale='s0', event_type='K',
                        ref_type='user_message', summary='stop 2',
                        session_id=ctx.session_id)

        chains = self.dal.get_chains(session_id='consist-test', scale='s0')
        assert len(chains) == 2
        assert chains[0]['chain_id'] != chains[1]['chain_id']


# ═══════════════════════════════════════════════════════
# SessionContext replaces brain.session_id
# ═══════════════════════════════════════════════════════

class TestSessionContextReplacesSingleton:
    """Verify SessionContext works where brain.session_id used to."""

    @pytest.fixture(autouse=True)
    def setup_brain(self):
        with brain_env() as env:
            self.brain = env.brain
            yield

    def test_daemon_restart_preserves_session(self):
        """If daemon restarts, loading the saved context preserves session_id."""
        from servers.session_context import SessionContext
        ctx = SessionContext(session_id='stable-session', stop_counter=50)
        ctx.save(self.brain._session_state)

        # Simulate daemon restart — new Brain instance, same DB
        from servers.brain import Brain
        brain2 = Brain(self.brain.db_path)

        loaded = SessionContext.load(brain2._session_state, 'stable-session')
        assert loaded is not None
        assert loaded.session_id == 'stable-session'
        assert loaded.stop_counter == 50
        brain2.close()

    def test_parallel_sessions_no_conflict(self):
        """Two sessions writing traces simultaneously don't interfere."""
        from servers.session_context import SessionContext
        ctx_a = SessionContext(session_id='parallel-a', stop_counter=1)
        ctx_b = SessionContext(session_id='parallel-b', stop_counter=1)

        dal = self.brain._trace_dal
        self.brain.logs_conn.execute('DELETE FROM trace_events')
        self.brain.logs_conn.commit()

        dal.append(chain_id=ctx_a.s0_chain(), scale='s0', event_type='K',
                   ref_type='user_message', summary='from A',
                   session_id=ctx_a.session_id)
        dal.append(chain_id=ctx_b.s0_chain(), scale='s0', event_type='K',
                   ref_type='user_message', summary='from B',
                   session_id=ctx_b.session_id)

        chains_a = dal.get_chains(session_id='parallel-a', scale='s0')
        chains_b = dal.get_chains(session_id='parallel-b', scale='s0')
        assert len(chains_a) == 1
        assert len(chains_b) == 1
        assert chains_a[0]['events'][0]['summary'] == 'from A'
        assert chains_b[0]['events'][0]['summary'] == 'from B'

    def test_empty_session_id_raises(self):
        """No session id means no session state — never another session's."""
        with pytest.raises(ValueError):
            self.brain.get_or_create_session('')
        with pytest.raises(ValueError):
            self.brain.reset_session_activity('')
