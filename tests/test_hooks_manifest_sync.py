"""Contract: the two hook manifests are one set of scripts, projected per host.

hooks/hooks.json is the Claude Code manifest; hooks/hooks.codex.json is the
Codex projection of it — same commands, only the event set and per-host handler
fields differ. Two files that share seven identical handlers drift silently, so
this locks BOTH directions: every Codex handler is backed by a CC handler (same
command, a subset of its tool names) unless it is a declared Codex-only handler
(CODEX_ONLY_HANDLERS, each naming the Claude Code mechanism that replaces it),
and every CC handler on a shared event is projected into the Codex file (minus
the tool names that only exist on Claude Code). Plus the two Codex-specific
constraints — SessionEnd within Codex's 3 s cap, and the injecting hooks
lifting Codex's ~2,500-token spill limit — and timeout parity everywhere else.
Pure file inspection.

The host contract (servers/host_contract.py) DECLARES each host's registered
events, tool names and matcher aliases; the manifests stay the source of truth
(nothing parses them at runtime). TestContractManifestParity holds the two in
step both ways (design D4) — the engine event lists are read from the contract.
"""
import json
import os
import re
import sys
import unittest

_HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from servers.host_contract import HOST_CONTRACT  # noqa: E402

# Events the Codex hook engine dispatches (learn.chatgpt.com/docs/hooks) — the
# contract owns the list, dated by the host version it was read against.
CODEX_EVENTS = set(HOST_CONTRACT["codex"]["engine_events"])
# Claude Code events with no Codex counterpart — must not appear in the projection.
CC_ONLY_EVENTS = set(HOST_CONTRACT["claude-code"]["events"]) - CODEX_EVENTS
# Claude Code local-tool names that never fire under Codex: its file tool is
# apply_patch (matched via the Edit|Write aliases), sub-agents are spawn_agent
# (matched via the Agent alias), and search/fetch are hosted tools that skip the
# hook path entirely.
CC_ONLY_TOOL_NAMES = ("Read", "Glob", "Grep", "WebSearch", "WebFetch", "NotebookEdit")

# Handlers that exist ONLY in the Codex projection, keyed by script name, each
# with the Claude Code mechanism that makes it unnecessary there. An entry here
# is a statement, and it is checked: the script must be registered in
# hooks.codex.json and absent from hooks.json, or the entry is stale.
CODEX_ONLY_HANDLERS = {
    "stamp-caller-session.sh":
        "Claude Code hands the MCP proxy CLAUDE_CODE_SESSION_ID; Codex passes stdio "
        "MCP servers no thread id, so a PreToolUse hook signs session_id into the "
        "brain tool input (decision fa0f5f5a keeps Claude Code on the env var)",
}

# Codex caps SessionEnd (and Interrupt) handlers at 3 seconds.
CODEX_SESSION_END_MAX_TIMEOUT = 3
# Events whose additionalContext Codex spills to disk above ~2,500 tokens unless
# the handler sets additionalContextLimit (0 = pass everything through).
INJECTING_EVENTS = ("SessionStart", "UserPromptSubmit")


def _load(name):
    with open(os.path.join(_HOOKS_DIR, name)) as f:
        doc = json.load(f)
    assert isinstance(doc.get("hooks"), dict), "%s must have the {\"hooks\": {...}} wrapper" % name
    return doc["hooks"]


def _handlers(events, event):
    """(matcher, command, timeout) triples registered for `event`."""
    return {
        (group.get("matcher", "") or "", handler["command"], handler.get("timeout"))
        for group in events.get(event, [])
        for handler in group["hooks"]
    }


def _codex_only(command):
    return any(name in command for name in CODEX_ONLY_HANDLERS)


def _alternatives(matcher):
    """The tool names a matcher selects; None means every tool ("" or "*")."""
    if matcher in ("", "*"):
        return None
    return set(matcher.split("|"))


def _covers(wide, narrow):
    """True when matcher `wide` selects every tool matcher `narrow` selects."""
    w, n = _alternatives(wide), _alternatives(narrow)
    if w is None:
        return True
    if n is None:
        return False
    return n <= w


def _projected(cc_matcher):
    """What a CC matcher should look like on Codex: the same tools minus the
    Claude-Code-only names. None = universal; empty set = nothing to project."""
    alts = _alternatives(cc_matcher)
    if alts is None:
        return None
    return alts - set(CC_ONLY_TOOL_NAMES)


class TestHooksManifestSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cc = _load("hooks.json")
        cls.cx = _load("hooks.codex.json")
        cls.shared_events = set(cls.cc) - CC_ONLY_EVENTS

    def test_codex_events_are_supported(self):
        unknown = set(self.cx) - CODEX_EVENTS
        self.assertFalse(unknown, "hooks.codex.json registers events Codex has no engine for: %s" % sorted(unknown))

    def test_cc_only_events_absent(self):
        leaked = set(self.cx) & CC_ONLY_EVENTS
        self.assertFalse(leaked, "Claude-Code-only events in the Codex projection: %s" % sorted(leaked))

    def test_every_shared_event_is_projected(self):
        missing = self.shared_events - set(self.cx)
        self.assertFalse(missing, "events in hooks.json with no Codex projection: %s" % sorted(missing))

    def test_codex_handlers_are_backed_by_cc_handlers(self):
        # Codex ⊆ CC: a Codex handler must exist in hooks.json with the same
        # command and a matcher that selects at least the same tools.
        for event in self.cx:
            cc = _handlers(self.cc, event)
            for matcher, command, _timeout in _handlers(self.cx, event):
                if _codex_only(command):
                    continue
                self.assertTrue(
                    any(c == command and _covers(m, matcher) for m, c, _t in cc),
                    "%s handler %r under matcher %r has no backing handler in hooks.json — "
                    "the Codex manifest is a projection (same command, a subset of the tool "
                    "names); a deliberately Codex-only handler goes in CODEX_ONLY_HANDLERS"
                    % (event, command, matcher))

    def test_codex_only_handlers_are_registered_and_absent_from_cc(self):
        # The allowlist stays armed: a name nothing registers guards nothing,
        # and a name hooks.json also carries is not Codex-only any more.
        cx_commands = {c for event in self.cx for _m, c, _t in _handlers(self.cx, event)}
        cc_commands = {c for event in self.cc for _m, c, _t in _handlers(self.cc, event)}
        for name, why in CODEX_ONLY_HANDLERS.items():
            self.assertTrue(any(name in c for c in cx_commands),
                            "CODEX_ONLY_HANDLERS names %r but hooks.codex.json does not register it" % name)
            self.assertFalse(any(name in c for c in cc_commands),
                             "%r is registered in hooks.json too — drop it from CODEX_ONLY_HANDLERS "
                             "(the entry claims: %s)" % (name, why))
            # Timeout parity is skipped for it, so pin the one thing parity gave.
            for event in self.cx:
                for _m, c, t in _handlers(self.cx, event):
                    if name in c:
                        self.assertIsNotNone(t, "%s must carry an explicit timeout" % name)

    def test_cc_handlers_are_projected_to_codex(self):
        # CC ⊆ Codex (minus CC-only tools): every handler hooks.json registers on
        # a shared event must reach the Codex file, or the Codex host silently
        # loses it the day someone adds a handler on the Claude Code side.
        for event in self.shared_events:
            cx = _handlers(self.cx, event)
            for matcher, command, _timeout in _handlers(self.cc, event):
                wanted = _projected(matcher)
                if wanted is not None and not wanted:
                    continue  # only Claude-Code-only tools: nothing to project
                narrow = "" if wanted is None else "|".join(sorted(wanted))
                self.assertTrue(
                    any(c == command and _covers(m, narrow) for m, c, _t in cx),
                    "%s handler %r (matcher %r) is not projected into hooks.codex.json — "
                    "add it there (Codex tools: %s)" % (event, command, matcher, narrow or "all"))

    def test_no_cc_only_tool_names_in_codex_matchers(self):
        for event in ("PreToolUse", "PostToolUse"):
            for matcher, _c, _t in _handlers(self.cx, event):
                for name in CC_ONLY_TOOL_NAMES:
                    self.assertNotIn(
                        name, matcher.split("|"),
                        "%s matcher %r names %s, which never fires under Codex" % (event, matcher, name))

    def test_timeouts_match_except_session_end(self):
        # The handler tuning is one value per script; only SessionEnd differs,
        # because Codex caps it (checked below).
        for event in self.cx:
            if event == "SessionEnd":
                continue
            cc = {c: t for _m, c, t in _handlers(self.cc, event)}
            for _matcher, command, timeout in _handlers(self.cx, event):
                if _codex_only(command):
                    continue
                self.assertEqual(timeout, cc.get(command),
                                 "%s handler %r: Codex timeout %r != Claude Code timeout %r"
                                 % (event, command, timeout, cc.get(command)))

    def test_session_end_within_codex_cap(self):
        for _m, _c, timeout in _handlers(self.cx, "SessionEnd"):
            self.assertLessEqual(
                timeout if timeout is not None else 600, CODEX_SESSION_END_MAX_TIMEOUT,
                "Codex kills SessionEnd handlers after %d s" % CODEX_SESSION_END_MAX_TIMEOUT)

    def test_injecting_hooks_lift_spill_limit(self):
        for event in INJECTING_EVENTS:
            for group in self.cx.get(event, []):
                for handler in group["hooks"]:
                    self.assertIn(
                        "additionalContextLimit", handler,
                        "%s handler must set additionalContextLimit — boot and recall "
                        "injections sit at Codex's default spill threshold" % event)


_PATTERN_MATCHER_RE = re.compile(r"[.*+?\[\]()^$\\]")


def _tool_matchers(events, event):
    """The literal tool names each PreToolUse/PostToolUse matcher selects;
    regex matchers (mcp__.*) are returned as ('pattern', matcher)."""
    out = []
    for group in events.get(event, []):
        matcher = group.get("matcher", "") or ""
        if matcher in ("", "*"):
            continue
        if _PATTERN_MATCHER_RE.search(matcher):
            out.append(("pattern", matcher))
        else:
            out.extend(("name", n) for n in matcher.split("|"))
    return out


class TestContractManifestParity(unittest.TestCase):
    """D4: the contract's `events` is DECLARED and held equal to the host's
    manifest both ways; the manifest's tool matchers are held to the contract's
    tool map (directly or through a declared alias), and every declared tool is
    captured by some PostToolUse matcher — a tool the contract classifies but
    no hook ever fires for would be a silent coverage hole."""

    def _manifest(self, host):
        return _load(os.path.basename(HOST_CONTRACT[host]["manifest"]))

    def test_declared_events_equal_registered_events(self):
        for host, entry in HOST_CONTRACT.items():
            self.assertEqual(set(entry["events"]), set(self._manifest(host)),
                             "%s: contract events != %s events" % (host, entry["manifest"]))

    def test_registered_events_within_engine_events(self):
        for host, entry in HOST_CONTRACT.items():
            unknown = set(self._manifest(host)) - set(entry["engine_events"])
            self.assertFalse(unknown, "%s registers events its engine does not dispatch: %s"
                             % (host, sorted(unknown)))

    def test_matcher_tool_names_are_declared(self):
        for host, entry in HOST_CONTRACT.items():
            declared = set(entry["tools"]) | set(entry["matcher_aliases"])
            for event in ("PreToolUse", "PostToolUse"):
                for kind, value in _tool_matchers(self._manifest(host), event):
                    if kind == "name":
                        self.assertIn(value, declared,
                                      "%s %s matcher names %r, which the contract neither "
                                      "declares as a tool nor as a matcher alias" % (host, event, value))

    def test_every_declared_tool_is_captured_by_a_post_tool_matcher(self):
        for host, entry in HOST_CONTRACT.items():
            matched = {v for k, v in _tool_matchers(self._manifest(host), "PostToolUse") if k == "name"}
            patterns = {v for k, v in _tool_matchers(self._manifest(host), "PostToolUse") if k == "pattern"}
            via_alias = {target for alias, target in entry["matcher_aliases"].items() if alias in matched}
            uncaptured = set(entry["tools"]) - matched - via_alias
            self.assertFalse(uncaptured, "%s declares tools no PostToolUse matcher captures: %s"
                             % (host, sorted(uncaptured)))
            if any(pat.startswith("^mcp__") for pat, _k in entry["tool_patterns"]):
                self.assertTrue(any("mcp__" in p for p in patterns),
                                "%s classifies mcp__ tools but no PostToolUse matcher captures them" % host)


_SCRIPTS_DIR = os.path.join(_HOOKS_DIR, "scripts")
_HANDLER_SCRIPT = re.compile(r'hooks/scripts/([\w.-]+)"')
_SHIM_EXEC_PY = re.compile(r'\$\(dirname "\$0"\)/(\w+\.py)"')


def _reads_hook_input(py_name):
    with open(os.path.join(_SCRIPTS_DIR, py_name)) as f:
        return "get_hook_input()" in f.read()


class TestShimsDeliverPayload(unittest.TestCase):
    """A hook's stdin payload reaches its Python client only through the
    HOOK_INPUT env var the bash shim exports; hook_common.get_hook_input reads
    it. A client run without that export parses {} — no session_id on Codex,
    where no env var backfills it."""

    def test_every_payload_reader_gets_hook_input(self):
        for name in ("hooks.json", "hooks.codex.json"):
            for event, groups in _load(name).items():
                for group in groups:
                    for handler in group["hooks"]:
                        script = _HANDLER_SCRIPT.search(handler["command"]).group(1)
                        if script.endswith(".py"):
                            self.assertFalse(
                                _reads_hook_input(script),
                                "%s %s runs %s directly, but it reads HOOK_INPUT, "
                                "which only a shim exports" % (name, event, script))
                            continue
                        with open(os.path.join(_SCRIPTS_DIR, script)) as f:
                            shim = f.read()
                        clients = _SHIM_EXEC_PY.findall(shim)
                        self.assertTrue(clients, "%s execs no recognizable client" % script)
                        for client in clients:
                            if _reads_hook_input(client):
                                self.assertIn(
                                    "export HOOK_INPUT=", shim,
                                    "%s %s: %s runs %s, which reads HOOK_INPUT, "
                                    "but never exports it" % (name, event, script, client))


if __name__ == "__main__":
    unittest.main()
