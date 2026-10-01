#!/bin/bash
# brain — the ONE definition of "is the isolated runtime ready?".
#
# Sourced by ensure-runtime.sh (fast path + loser-wait recheck),
# boot-brain.sh (cold-install detection) and mcp-launch.sh (cold wait).
# When the runtime layout changes (e.g. the deferred $CLAUDE_PLUGIN_DATA
# relocation), THIS is the only place to touch — a split-brain
# cold-detection between the launchers re-creates the concurrent-bootstrap
# race.
#
# Two states, both answered here:
#   installed — a venv that finished at least one sync (the sentinel exists).
#               It can serve, even while a re-sync is due.
#   ready     — installed AND synced for the current uv.lock + .python-version:
#               the sentinel holds their fingerprint. A lock change (an ordinary
#               diff through main) re-syncs on the next bootstrap, so an install
#               never stays frozen at the library versions of the day it was built.
#
# The ready check runs on every hook, so its steady state is builtins only:
# while neither input was written after the sentinel, the sentinel stands. A
# rewrite (git checkout, merge) costs one content comparison, and an equal
# result touches the sentinel so the comparison doesn't repeat. A deployer that
# restores old mtimes while extracting (unzip) must touch uv.lock after it
# (redeploy.sh does); an empty sentinel (written before the lock existed) is
# never ready.
#
# Usage: brain_runtime_ready "$PLUGIN_ROOT" && echo warm

brain_runtime_fingerprint() {
    { cat "$1/uv.lock" "$1/.python-version" 2>/dev/null || true; } | cksum
}

brain_runtime_installed() {
    [ -f "$1/.runtime-ready" ] && [ -x "$1/venv/bin/python" ]
}

brain_runtime_ready() {
    local _fp=
    [ -f "$1/uv.lock" ] && brain_runtime_installed "$1" || return 1
    read -r _fp < "$1/.runtime-ready" || true
    [ -n "$_fp" ] || return 1
    if [ "$1/uv.lock" -nt "$1/.runtime-ready" ] ||
       [ "$1/.python-version" -nt "$1/.runtime-ready" ]; then
        [ "$_fp" = "$(brain_runtime_fingerprint "$1")" ] || return 1
        touch "$1/.runtime-ready" 2>/dev/null || true
    fi
    return 0
}
