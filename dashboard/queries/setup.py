"""First-run setup backend — API-key persistence for the /setup page.

The ONE file the dashboard ever writes: the user env file
(~/.config/brain/env). User config, never a DB — the passive-observer
invariant (CLAUDE.md) is about the brain's databases. Lives in queries/
per server.py's header contract (routing there, data access here).

The key value is a secret end-to-end: validated, written 0600 atomically,
never logged, never echoed back, never read into a response (presence only).
"""

import os


def brain_env_path() -> str:
    """Canonical user env file — same resolution as brain-env.sh/load_env."""
    xdg = os.environ.get('XDG_CONFIG_HOME') or os.path.join(
        os.path.expanduser('~'), '.config')
    return os.path.join(xdg, 'brain', 'env')


def api_key_present(env_path: str = None) -> bool:
    """True when the env file carries an ANTHROPIC_API_KEY line.

    Presence only — the value is never read into a response.
    """
    path = env_path or brain_env_path()
    try:
        with open(path) as f:
            return any(line.startswith('ANTHROPIC_API_KEY=') for line in f)
    except OSError:
        return False


def write_api_key(env_path: str, key: str):
    """Persist the API key to the dotenv file (mode 600, atomic replace).

    Returns (ok, message). The message never contains the key. Unlike the
    boot-hook mirror (which never overwrites), the setup form is explicit
    user intent — an existing ANTHROPIC_API_KEY line is REPLACED.
    """
    if not key.startswith('sk-') or len(key) < 10 or len(key) > 300:
        return False, "That doesn't look like an Anthropic API key (expected sk-...)."
    if any(c.isspace() for c in key):
        return False, "The key contains whitespace — check the paste."
    lines = []
    try:
        with open(env_path) as f:
            lines = [l.rstrip('\n') for l in f
                     if not l.startswith('ANTHROPIC_API_KEY=')]
    except OSError:
        pass
    lines.append('ANTHROPIC_API_KEY=%s' % key)
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    tmp = env_path + '.tmp.%d' % os.getpid()
    try:
        # O_EXCL: never follow/overwrite a pre-existing file or symlink at
        # the tmp path (predictable name). A leftover from a crashed writer
        # is removed first — same-pid reuse is the only way it exists.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        os.replace(tmp, env_path)
        os.chmod(env_path, 0o600)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False, "Could not write %s: %s" % (env_path, e.strerror or e)
    return True, ("Key saved. Your brain picks it up on the next message — "
                  "no restart needed.")
