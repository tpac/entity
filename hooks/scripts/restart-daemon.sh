#!/bin/bash
# Restart the brain daemon — clears cache, re-execs with fresh code.
# Usage: hooks/scripts/restart-daemon.sh
#
# The wire protocol and the daemon address both have owners in Python
# (daemon_client.send_command, daemon_config); this script is a thin CLI over
# them, not a second implementation. Sourcing the resolver first gives us
# PLUGIN_ROOT and the venv Python.
source "$(dirname "$0")/resolve-brain-db.sh"

# `cd` to the tree we import from, and use its OWN interpreter:
#   * `python3 -c` puts the CWD at sys.path[0], AHEAD of PYTHONPATH — run from a
#     directory that happens to contain a `servers` package and we would import
#     THAT one. cd makes the intended tree the cwd, so the two agree.
#   * $BRAIN_PYTHON is the venv interpreter every other brain entry point uses;
#     bare `python3` is only the venv's when brain-env.sh got far enough to
#     prepend it to PATH.
# `cd ""` returns 0 and stays put in bash, zsh and dash — an emptiness test is
# the only thing that actually catches an unresolved PLUGIN_ROOT here.
[ -n "$PLUGIN_ROOT" ] || { echo "PLUGIN_ROOT unresolved — cannot locate servers/" >&2; exit 1; }
cd "$PLUGIN_ROOT" || { echo "Cannot enter plugin root: $PLUGIN_ROOT" >&2; exit 1; }

"${BRAIN_PYTHON:-python3}" -c "
import sys
from servers.daemon_client import send_command
from servers.daemon_config import get_daemon_addr

resp = send_command('restart', timeout=3.0)
if resp.get('ok') and resp.get('result', {}).get('status') == 'syncing':
    print('Restart sent. The daemon is syncing its runtime to uv.lock while it serves; it re-execs when the sync ends.')
elif resp.get('ok'):
    print('Restart sent. Daemon will re-exec in ~4s.')
elif resp.get('transport'):
    # A wire failure — the daemon never answered. Report the port, the part
    # an operator can act on; the host is always loopback (DAEMON_HOST).
    print('Cannot reach daemon on port %s: %s' % (get_daemon_addr()[1], resp.get('error')))
    sys.exit(1)
else:
    print('Restart failed: %s' % resp.get('error', '?'))
    sys.exit(1)
"
