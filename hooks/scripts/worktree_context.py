"""WorktreeCreate — records the per-session worktree/branch in brain.
Thin client: sends hook_worktree_context to daemon.

NEVER print to stdout here. Claude Code consumes a WorktreeCreate hook's STDOUT as
the new worktree path ("redirect any other output to stderr so it doesn't interfere
with the path"), so any printed line becomes a path it tries to chdir into. The
daemon records to the session object; this client emits nothing.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, run_hook

hook_input = get_hook_input()

def main():
    if daemon_available():
        daemon_call_raw("hook_worktree_context", hook_input, timeout=5.0)
    else:
        daemon_unavailable_error("worktree_context")

run_hook("worktree_context", main)
