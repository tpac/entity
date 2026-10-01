"""StopFailure — logs API failures to brain for pattern detection.
Thin client: sends hook_stop_failure_log to daemon.
"""
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import get_hook_input, daemon_available, daemon_call_raw, daemon_unavailable_error, run_hook

hook_input = get_hook_input()

def main():
    if daemon_available():
        daemon_call_raw("hook_stop_failure_log", hook_input, timeout=5.0)
    else:
        daemon_unavailable_error("stop_failure_log")

run_hook("stop_failure_log", main)
