"""WorktreeRemove — clears the per-session worktree identity in brain.
Thin client: sends hook_worktree_cleanup to daemon.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, run_hook

hook_input = get_hook_input()

def main():
    if daemon_available():
        daemon_call_raw("hook_worktree_cleanup", hook_input, timeout=5.0)
    else:
        daemon_unavailable_error("worktree_cleanup")

run_hook("worktree_cleanup", main)
