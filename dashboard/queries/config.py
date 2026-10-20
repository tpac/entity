"""Configuration — which brain this dashboard watches, and what keeps the
brain from recording.

One composed read behind the Config tab and the cross-tab pause banner:
  * the daemon's ping, as system.daemon_record reads it — the brain it
    actually writes (`db_dir`), where its code runs;
  * the daemon's `gate status` — pause, ignore patterns, and a verdict for
    each project the caller names (the live sessions' projects). The rule
    comes from its owner over the socket; the dashboard copies no matcher;
  * the dashboard's own resolution of the same two paths (its brain dir, the
    user env file), so a divergence shows as the misconfiguration it is.

The two writes the tab and the /setup page offer — pause/resume and the API
key — go to the daemon over the one socket (`gate`, `set_api_key`); the
daemon is the user env file's writer. The dashboard reads that file for
presence only (the key's value is never read into a response).
"""

import os

from ..daemon_client import daemon_call, daemon_send
from ..db import _read_env_file_key, brain_db_path, user_env_file
from .system import daemon_record


def api_key_present(env_path: str = None) -> bool:
    """True when the env file carries an ANTHROPIC_API_KEY line.

    Presence only — the value is never read into a response. Read with the
    dashboard's shell-tolerant env-file reader (export prefix, quotes), the
    same one that resolves BRAIN_DB_DIR from this file.
    """
    return _read_env_file_key(env_path or user_env_file(), 'ANTHROPIC_API_KEY') is not None


def _same_dir(a: str, b: str) -> bool:
    return os.path.realpath(a) == os.path.realpath(b)


def query_config(projects=()):
    """The Config tab's one payload. `projects`: the live sessions' derived
    projects; the gate answers a verdict for each, plus for '' (a session
    whose project the brain never derived — `*` matches it too). `gate` is
    None when the daemon is down; `db_dir_matches` is None then too, since
    what the daemon writes is unknown."""
    dash_dir = os.path.dirname(brain_db_path())
    daemon = daemon_record()
    out = {
        'daemon': daemon,
        'dashboard': {'db_dir': dash_dir, 'env_file': user_env_file()},
        'db_dir_matches': None,
        'gate': None,
        'api_key_present': api_key_present(),
    }
    if not daemon['alive']:
        return out
    out['db_dir_matches'] = bool(daemon['db_dir']) and _same_dir(daemon['db_dir'], dash_dir)
    wanted = list(dict.fromkeys([''] + [str(p) for p in projects]))
    out['gate'] = daemon_send('gate', {'action': 'status', 'projects': wanted}, timeout=3)
    return out


def set_paused(on: bool) -> dict:
    """Flip the pause switch through the daemon's gate command; the reply
    envelope carries the resulting status or the daemon's refusal."""
    return daemon_call('gate', {'action': 'pause' if on else 'resume'}, timeout=5)


def set_api_key(key: str):
    """Hand the /setup page's key to the daemon, which validates it and
    writes the env file. Returns (ok, message); the message never carries
    the key — the daemon's refusal text does not echo it either."""
    resp = daemon_call('set_api_key', {'api_key': key}, timeout=5)
    if resp.get('ok'):
        return True, ("Key saved. Your brain picks it up on the next message — "
                      "no restart needed.")
    error = resp.get('error') or 'the daemon refused the key'
    if error == 'daemon unreachable':
        error = ("The brain daemon is not running — it writes the key. Open a "
                 "Claude session (its boot starts the daemon) and try again.")
    return False, error
