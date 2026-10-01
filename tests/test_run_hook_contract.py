"""Contract: the run_hook error-boundary standard.

Every event-hook script routes its body through run_hook(name, fn[, on_error]) —
a single error boundary that logs to hook_errors and never lets a hook crash
the host. This test LOCKS that standard so a new hook can't silently swallow
exceptions (the regression that took a ~15-file sweep to fix). It also guards
the two adjacent invariants: no hand-rolled top-level try/except in an event
hook, and no reintroduction of the deleted log_hook_output no-op.

Pure file/AST inspection — no brain, no daemon.
"""
import os
import ast
import glob
import unittest

_HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks", "scripts")

# Scripts in hooks/scripts/ that are NOT event hooks — shared infra
# (hook_common defines run_hook) and standalone utilities. The run_hook
# contract does not apply to these.
_NON_EVENT = {
    "hook_common.py",
    # Monitor-launched long-running poller for /watch-live — an infinite
    # peek loop, not a Claude Code event hook. It has its own resilience
    # model (transient errors → stderr, loop continues); run_hook is a
    # once-and-return error boundary that doesn't fit a never-returning poller.
    "self_inbox_poller.py",
}


def _event_hook_paths():
    return [p for p in sorted(glob.glob(os.path.join(_HOOKS_DIR, "*.py")))
            if os.path.basename(p) not in _NON_EVENT]


class TestRunHookContract(unittest.TestCase):
    def test_event_hooks_discovered(self):
        """Guard against a path typo silently making the other tests vacuous."""
        self.assertGreaterEqual(
            len(_event_hook_paths()), 10,
            "expected the event-hook scripts to be discovered in %s" % _HOOKS_DIR)

    def test_every_event_hook_routes_through_run_hook(self):
        """Each event hook must call run_hook(...) — the error boundary. A new
        hook that hand-rolls its own try/except (or none) fails here."""
        for p in _event_hook_paths():
            tree = ast.parse(open(p).read(), filename=p)
            calls = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "run_hook"
                for n in ast.walk(tree))
            self.assertTrue(
                calls,
                "%s must route through run_hook (the error-boundary standard), "
                "not a hand-rolled try/except" % os.path.basename(p))

    def test_no_top_level_try_in_event_hooks(self):
        """The hook's error handling belongs in run_hook, not a module-level
        try/except. Nested try (inside a function or an if — e.g. best-effort
        transcript parsing) is fine; only a top-level try is the anti-pattern."""
        for p in _event_hook_paths():
            tree = ast.parse(open(p).read(), filename=p)
            self.assertFalse(
                any(isinstance(n, ast.Try) for n in tree.body),
                "%s has a module-level try/except — wrap the body in def main() "
                "and use run_hook instead" % os.path.basename(p))

    def test_log_hook_output_stays_deleted(self):
        """The deleted no-op must stay dead — re-adding it reopens the silent-
        failure trap (it looked like it logged but wrote nothing)."""
        hc = open(os.path.join(_HOOKS_DIR, "hook_common.py")).read()
        self.assertNotIn("def log_hook_output", hc,
                         "log_hook_output no-op was reintroduced in hook_common.py")
        for p in glob.glob(os.path.join(_HOOKS_DIR, "*.py")):
            self.assertNotIn(
                "log_hook_output(", open(p).read(),
                "%s calls the deleted log_hook_output no-op" % os.path.basename(p))


if __name__ == "__main__":
    unittest.main()
