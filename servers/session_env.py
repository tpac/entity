"""Session environment derivation — the HOST-ADAPTER layer.

Everything here inspects the host machine (git, filesystem) to derive the
per-session identity the brain RECEIVES: (branch, worktree, project). The
brain never derives these itself — `SessionContext.set_env()` takes the
values, three-state (None = detection failed, keep what we have).

Porting the brain to a different host (another agent runtime, a chat
product) means reimplementing THIS module's one entry point — not touching
Brain:

    detect_session_env(cwd) -> (branch, worktree, project)

Project resolution order (first hit wins):
  1. `.brain-project` marker file — one sanitized line, walked upward from
     cwd to the filesystem root. Explicit operator intent; beats git, so a
     renamed/moved folder keeps its identity. Never auto-written.
  2. git main-repo directory name — identical from the main tree and every
     linked worktree (same `--git-common-dir`).
  3. cwd basename — non-repo sessions: a named working folder asserts its
     identity. Denylisted anchors ($HOME, /, /tmp, Downloads/Desktop/
     Documents) say nothing about intent → None (keep the session's known
     value; a fresh session simply stays unscoped).

Failure semantics: "not a git repository" is a DEFINITIVE probe result and
falls through to the basename; any other git failure (missing dir, timeout)
is TRANSIENT and returns None fields so `set_env` keeps what the session
already had — a hiccup on resume must not wipe a known worktree/project
(and in a worktree the cwd basename is the WORKTREE name, not the project,
so guessing on transient failure would mislabel provenance).
"""

import os
import re

MARKER_FILENAME = '.brain-project'
# One line, no path tricks: alnum start, then alnum/dot/dash/underscore.
_MARKER_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
# Junk anchors that must not become a project identity.
_DENY_BASENAMES = {'downloads', 'desktop', 'documents'}


def _git(cwd, *args, log=None):
    """Run `git -C cwd <args>`; return (returncode, stdout, stderr).
    (None, '', '') on exception (no cwd, timeout, missing git). Single source
    for the daemon's env-NEUTRAL git shelling — timeout / error-logging /
    hardening live in one place."""
    if not cwd:
        return (None, '', '')
    try:
        import subprocess
        # LC_ALL=C pins git's message wording/locale — detect_session_env
        # classifies "not a git repository" from stderr text, and a
        # French/German daemon must not silently turn every non-repo
        # session into a keep-what-we-have no-op.
        env = dict(os.environ, LC_ALL='C')
        r = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True, text=True, timeout=5, env=env)
        return (r.returncode, r.stdout.strip(), r.stderr.strip())
    except Exception as e:
        if log:
            try:
                log('git_run', e, 'git ' + ' '.join(args))
            except Exception:
                pass
        return (None, '', '')


def worktree_from_gitdir(gitdir: str) -> str:
    """Linked-worktree name from a git-dir string, or '' for the main tree.

    A linked worktree's git-dir ends '<repo>/.git/worktrees/<name>'; the main
    tree's is plain '.git' (or an absolute '<repo>/.git' from a subdir).
    Anchored on '.git/worktrees/' (NOT a bare '/worktrees/') and split on the
    LAST occurrence, so a repo whose own path contains a 'worktrees' segment
    can't false-match — neither a main tree at '/x/worktrees/repo/.git' nor a
    linked tree under it."""
    marker = ".git/worktrees/"
    if marker not in gitdir:
        return ''
    return gitdir.split(marker)[-1].split("/", 1)[0].strip()


def project_from_common_dir(common: str, cwd: str) -> str:
    """Repo identity from `git rev-parse --git-common-dir`: the main repo's
    directory name, identical from the main tree and every linked worktree
    (both report the same common dir, e.g. '/Users/x/brain/.git' → 'brain').
    Relative output ('.git' from the repo root) is resolved against cwd.
    Split on the LAST '/.git' so submodule git-dirs ('<repo>/.git/modules/…')
    still resolve to the superproject. '' when the shape is unrecognized."""
    if not common:
        return ''
    if not os.path.isabs(common):
        common = os.path.abspath(os.path.join(cwd or '.', common))
    marker = '/.git'
    if marker not in common:
        return ''
    repo_root = common.rsplit(marker, 1)[0]
    return os.path.basename(repo_root)


def find_project_marker(cwd: str, log=None) -> str:
    """First valid `.brain-project` marker walking cwd upward to the root.
    '' when none. The walk stops at $HOME (checked, then not exceeded) so a
    stray marker above the home directory can't rebrand every session; paths
    outside $HOME walk to the filesystem root. An unreadable or malformed
    marker is skipped (logged), never fatal — a bad file must not break
    session boot."""
    if not cwd:
        return ''
    d = os.path.realpath(cwd)
    home = os.path.realpath(os.path.expanduser('~'))
    while True:
        path = os.path.join(d, MARKER_FILENAME)
        try:
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    line = f.readline(256).strip()
                if _MARKER_RE.match(line):
                    return line
                # A malformed marker STOPS the search: the nearest marker
                # file is the operator's explicit intent — valid or not, it
                # must not lose to a farther ancestor's marker. Invalid →
                # logged, and resolution falls through to git/basename.
                if log:
                    try:
                        log('project_marker_invalid',
                            ValueError(line[:80]), path)
                    except Exception:
                        pass
                return ''
        except Exception as e:
            if log:
                try:
                    log('project_marker_read', e, path)
                except Exception:
                    pass
        parent = os.path.dirname(d)
        if d == home or parent == d:
            return ''
        d = parent


def project_from_cwd_basename(cwd: str) -> str:
    """Non-repo fallback: the folder name, unless it's a junk anchor.
    '' (unscoped) for $HOME, /, tmp trees, and generic user folders."""
    if not cwd:
        return ''
    p = os.path.realpath(cwd)
    home = os.path.realpath(os.path.expanduser('~'))
    if p in ('/', home):
        return ''
    for tmp in ('/tmp', '/private/tmp', '/var/tmp'):
        if p == tmp or p.startswith(tmp + '/'):
            return ''
    base = os.path.basename(p).strip()
    if not base or base.lower() in _DENY_BASENAMES:
        return ''
    return base


def detect_session_env(cwd: str, log=None):
    """Branch + worktree + project from ONE git probe (one fork+exec per
    SessionStart) plus the marker/basename resolution. Returns
    (branch, worktree, project):

      - transient git failure → ('unknown', None, marker-or-None). None is
        the 'keep what we have' signal for set_env.
      - not a git repository  → ('unknown', '', marker-or-basename-or-None:
        a junk-anchor cwd keeps the known project, a named folder asserts)
      - main working tree     → (branch, '', marker-or-repo_name)
      - linked worktree       → (branch, name, marker-or-repo_name)

    `git rev-parse --abbrev-ref HEAD --git-dir --git-common-dir` prints
    branch / git-dir / common-dir on lines 1-3."""
    marker = find_project_marker(cwd, log=log)
    rc, out, err = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD",
                        "--git-dir", "--git-common-dir", log=log)
    if rc == 0:
        lines = out.splitlines()
        branch = (lines[0].strip() if lines else '') or 'unknown'
        gitdir = lines[1].strip() if len(lines) > 1 else ''
        common = lines[2].strip() if len(lines) > 2 else ''
        return (branch, worktree_from_gitdir(gitdir),
                marker or project_from_common_dir(common, cwd))
    if rc is not None and 'not a git repository' in err.lower():
        # Definitive: this cwd is simply not a repo — worktree clears.
        # Project: a real named folder asserts its identity (basename); a
        # junk anchor ($HOME, /tmp, Downloads) says NOTHING about intent, so
        # it keeps the session's known value (None) rather than wiping it —
        # a re-boot whose cwd lands on an anchor must not strip provenance
        # from the rest of the session's writes.
        basename = project_from_cwd_basename(cwd)
        return ('unknown', '', marker or basename or None)
    # Transient (missing dir, timeout, git absent): keep what we have —
    # except a marker, which is explicit and safe to assert regardless.
    return ('unknown', None, marker or None)
