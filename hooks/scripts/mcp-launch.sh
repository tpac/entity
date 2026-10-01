#!/bin/bash
# brain — MCP server launcher
#
# Entry point from .mcp.json. Ensures the isolated Python runtime exists,
# then execs brain_mcp.py under the venv's Python. Claude Code speaks
# stdio MCP to this process, so stdout is reserved for MCP protocol —
# bootstrap output goes to stderr only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MCP_PLUGIN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/runtime-state.sh"

# ── Cold install: NEVER bootstrap inline — wait on the sentinel ────────────
# Claude Code gives this spawn 30s and never retries a failed connection. A
# fresh bootstrap can exceed that on slow networks, and racing it against the
# SessionStart hook's bootstrap SIGKILLed uv on the first laptop install
# (2026-07-17). So: kick the (mkdir-locked, so concurrency-safe) bootstrap
# detached in case nothing else has, then poll the sentinel with a deadline
# under the host startup window. The default stays under CC's 30s; a host
# manifest with a longer startup window can set BRAIN_MCP_BOOTSTRAP_WAIT_S.
# Ready in time → connect on this same connection. Otherwise the detached
# bootstrap continues and a later connection takes the ~8ms fast path.
if ! brain_runtime_installed "$_MCP_PLUGIN_ROOT"; then
    _mcp_wait_s="${BRAIN_MCP_BOOTSTRAP_WAIT_S:-25}"
    if ! [[ "$_mcp_wait_s" =~ ^[1-9][0-9]{0,3}$ ]]; then
        echo "[brain-mcp] BRAIN_MCP_BOOTSTRAP_WAIT_S must be a positive integer below 10000" >&2
        exit 1
    fi
    # Subshell + nohup: detach from this launcher's process group — when the
    # deadline below exits 1 and the host tears the MCP spawn down, a
    # group-directed signal must not kill the bootstrap mid-install.
    ( nohup "$SCRIPT_DIR/ensure-runtime.sh" \
        >> "$_MCP_PLUGIN_ROOT/.bootstrap.log" 2>&1 & )
    echo "[brain-mcp] cold install — waiting for runtime bootstrap (max ${_mcp_wait_s}s)..." >&2
    _mcp_deadline=$(( $(date +%s) + _mcp_wait_s ))
    while ! brain_runtime_ready "$_MCP_PLUGIN_ROOT"; do
        if [ "$(date +%s)" -ge "$_mcp_deadline" ]; then
            echo "[brain-mcp] runtime still bootstrapping — brain tools will be available next session (progress: $_MCP_PLUGIN_ROOT/.bootstrap.log)" >&2
            exit 1
        fi
        sleep 1
    done
    echo "[brain-mcp] runtime ready — connecting" >&2
elif ! brain_runtime_ready "$_MCP_PLUGIN_ROOT"; then
    # Installed but synced to an older uv.lock: re-sync detached and serve on
    # the libraries already in place, so this connection opens now.
    ( nohup "$SCRIPT_DIR/ensure-runtime.sh" \
        >> "$_MCP_PLUGIN_ROOT/.bootstrap.log" 2>&1 & )
    export BRAIN_RUNTIME_SERVE_STALE=1
fi

# Runtime present (fast path ~8ms) — wire env and exec the server.
source "$SCRIPT_DIR/brain-env.sh" 1>&2

# Exec MCP server under the venv Python. stdout stays clean for MCP protocol.
# One proxy per session (stdio transport), so the process carries the
# session's short id: `Entity-mcp-3207` pairs with that session's
# `Entity-in-3207` listener in Activity Monitor (brain_python_as).
exec "$(brain_python_as Entity-mcp "${CLAUDE_CODE_SESSION_ID:-}")" "$PLUGIN_DIR/servers/brain_mcp.py" "$@"
