"""Brain gate — may the brain's automatic mechanics run here?

ONE owner for the operator's run switches. Every hook asks it before doing
anything (hook_common.run_hook), on both hosts, so "pause the brain" and
"the brain stays out of this project" are decided in one place and never
re-derived per hook. A hook the gate refuses exits silently: no boot
context, no recall or surface, no S0 traces, no encode trigger, no session
row — the daemon never hears from that session. Explicit MCP tools keep
working (that is how the switches get flipped from inside a session).

Two switches, both lines of the one user-editable config file
(brain_constants.user_env_file — the API key's file; rulings id:b49a0082,
id:b626a547: user-facing knobs live where the user can edit them):

    BRAIN_PAUSED=1                       every session, every hook
    BRAIN_IGNORE_PROJECTS=brain,client-* sessions whose PROJECT matches

The ignore list is fnmatch patterns, comma- or colon-separated; a leading
`!` marks an include that beats any match (`*,!brain` = only brain runs).
It keys on the session's derived project (session_env.detect_session_env:
marker → git main-repo name → folder name), so every worktree of a repo
follows its repo, on either host — not on a path prefix. The project is
derived once per session and cached in a tmp marker; the switches are
re-read on every hook (a flip applies to the next hook of a live session).

The FILE is the only source — never the process env. brain-env.sh does
source the file (`set -a`) for the wrapped hooks and the daemon launcher, so
a BRAIN_PAUSED line is in those processes' env too, but it is the env of
when they STARTED: a daemon that booted paused would keep reporting paused
after `gate resume` removed the line, and a hook run as bare `python3`
(hooks.codex.json's PostToolUse) sees no shell at all. Reading the file every
time gives every process the same current answer.

Leaf: stdlib + the two leaf modules it composes (brain_constants owns the
file's grammar, reader and writer; session_env owns the project). Importable
by the system python3 that boot-brain.sh runs before the runtime is up, and
cheap enough for the per-tool-call hook. What a PROMPT holds is a different
axis (per prompt, not per session) and stays with host_contract.read_prompt
— the recall hook reads both.
"""
import fnmatch
from collections import namedtuple

from servers.brain_constants import (user_env_file, read_env_file_key,
                                     write_env_file_key, session_marker_path,
                                     dashboard_url)

PAUSED_KEY = 'BRAIN_PAUSED'
IGNORE_KEY = 'BRAIN_IGNORE_PROJECTS'
SWITCH_KEYS = (PAUSED_KEY, IGNORE_KEY)
# A pattern opening with this is an include that beats any match. The
# dashboard's Config tab mirrors the literal (tests/test_dashboard_disconnection
# pins it); the `gate` tool's description spells it out for the operator.
INCLUDE_PREFIX = '!'

_TRUE = ('1', 'true', 'yes', 'on')

Decision = namedtuple('Decision', ('run', 'reason'))

REASON_PAUSED = 'paused'
REASON_IGNORED = 'ignored_project'


def read_switch(key, env_path=None):
    """The switch's raw value from the env file, '' when unset."""
    return read_env_file_key(env_path or user_env_file(), key) or ''


def is_true(raw) -> bool:
    return (raw or '').strip().lower() in _TRUE


def parse_patterns(raw):
    """'a, b:c' → ('a', 'b', 'c'); empty entries dropped."""
    out = []
    for part in (raw or '').replace(':', ',').split(','):
        part = part.strip()
        if part:
            out.append(part)
    return tuple(out)


def project_ignored(project, patterns) -> bool:
    """fnmatch over the pattern list; an include (`!pat`) beats any match.
    An unscoped session (project '') matches `*` like anything else — `*`
    means everything."""
    project = project or ''
    includes = [p[1:] for p in patterns if p.startswith(INCLUDE_PREFIX) and len(p) > 1]
    excludes = [p for p in patterns if not p.startswith(INCLUDE_PREFIX)]
    if any(fnmatch.fnmatchcase(project, p) for p in includes):
        return False
    return any(fnmatch.fnmatchcase(project, p) for p in excludes)


def _switches(env_path=None):
    """Both switches, parsed, from one pass over the file: (paused, patterns)."""
    return (is_true(read_switch(PAUSED_KEY, env_path)),
            parse_patterns(read_switch(IGNORE_KEY, env_path)))


def _verdict(project, paused, patterns) -> Decision:
    if paused:
        return Decision(False, REASON_PAUSED)
    if project_ignored(project, patterns):
        return Decision(False, '%s:%s' % (REASON_IGNORED, project))
    return Decision(True, '')


def decide(project, env_path=None) -> Decision:
    """The verdict for a session whose derived project is `project`."""
    return _verdict(project, *_switches(env_path))


def verdicts(projects, env_path=None) -> dict:
    """{project: Decision} for many projects from ONE read of the switches —
    the gate status answers the dashboard's live roster this way instead of
    re-reading the file per project."""
    paused, patterns = _switches(env_path)
    return {p: _verdict(p, paused, patterns) for p in projects}


def status(env_path=None) -> dict:
    path = env_path or user_env_file()
    paused, patterns = _switches(path)
    return {'paused': paused, 'ignore_projects': list(patterns), 'env_file': path}


# ── the per-session project, cached ─────────────────────────────────────
# One git probe per session (18–70 ms measured), not one per tool call: the
# first hook derives it and leaves a marker the rest read
# (brain_constants.session_marker_path).


def session_project(cwd, session_id):
    """The derived project for this session: the cached marker when one
    exists, else detect_session_env(cwd) (cached when session_id is known
    and the probe was definitive — a transient git failure returns None and
    is not cached, so the next hook probes again). '' when neither cwd nor a
    marker says."""
    marker = session_marker_path('gate', session_id)
    if marker:
        try:
            with open(marker) as f:
                return f.read().strip()
        except OSError:
            pass
    if not cwd:
        return ''
    from servers.session_env import detect_session_env
    project = detect_session_env(cwd)[2]
    if project is None:
        return ''
    if marker:
        try:
            with open(marker, 'w') as f:
                f.write(project)
        except OSError:
            pass
    return project


def hook_allows(cwd, session_id, env_path=None) -> Decision:
    """The hook-side entry: derive (or recall) the project, then decide."""
    return decide(session_project(cwd, session_id), env_path)


def notice(decision, env_path=None) -> str:
    """The one line a refused BOOT prints — the only output a gated session
    ever gets. Without it a pattern set months ago silences a project and
    nobody learns why the brain went quiet; with it the assistant sees at
    session start that the brain is off here and says so once. Names the
    switch, the matching pattern and where to flip it. '' for an open gate."""
    if decision.run:
        return ''
    path = env_path or user_env_file()
    where = ("Change it on the dashboard's Config tab (%s) or with the gate tool."
             % dashboard_url())
    if decision.reason == REASON_PAUSED:
        return ("[brain] PAUSED — the brain records nothing in any session (BRAIN_PAUSED in %s). "
                "Tell the operator in one line at your first reply. %s" % (path, where))
    project = decision.reason.split(':', 1)[1]
    hit = next((p for p in parse_patterns(read_switch(IGNORE_KEY, path))
                if not p.startswith(INCLUDE_PREFIX) and fnmatch.fnmatchcase(project, p)), '?')
    return ("[brain] NOT recording this session — project %r matches BRAIN_IGNORE_PROJECTS "
            "pattern %r in %s: no memory, no recall, no traces here. Tell the operator in "
            "one line at your first reply. %s" % (project, hit, path, where))


# ── the writer — the MCP `gate` tool flips switches through it ──────────

def set_switch(key, value, env_path=None):
    """Set (or, with value '', remove) one gate switch in the env file."""
    if key not in SWITCH_KEYS:
        raise ValueError('not a gate switch: %r (one of %s)' % (key, SWITCH_KEYS))
    write_env_file_key(env_path or user_env_file(), key, value)


def set_paused(on, env_path=None):
    set_switch(PAUSED_KEY, '1' if on else '', env_path)


def add_ignore(pattern, env_path=None):
    pattern = (pattern or '').strip()
    if not pattern:
        raise ValueError('an ignore pattern is required')
    pats = list(parse_patterns(read_switch(IGNORE_KEY, env_path)))
    if pattern not in pats:
        pats.append(pattern)
    set_switch(IGNORE_KEY, ','.join(pats), env_path)


def remove_ignore(pattern, env_path=None):
    pattern = (pattern or '').strip()
    pats = [p for p in parse_patterns(read_switch(IGNORE_KEY, env_path)) if p != pattern]
    set_switch(IGNORE_KEY, ','.join(pats), env_path)
