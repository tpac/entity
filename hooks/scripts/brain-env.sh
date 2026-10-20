#!/bin/sh
# brain — shared environment setup, sourced by every hook .sh
# POSIX sh — sourced (never executed), so it runs in the consumer's shell:
# bash hooks, zsh, dash (/bin/sh on Linux). Keep bashism-free
# (TestResolverChainPortability).
#
# After sourcing:
#   $PLUGIN_DIR    resolves to plugin root
#   $BRAIN_PYTHON  points to the venv's python (the ONLY python hooks use)
#   $BRAIN_PYTHON_DAEMON / _DASH / _HOOK
#                  the same interpreter under a role name (see brain_python_as)
#   $PATH          has $PLUGIN_DIR/venv/bin prepended so `python3` resolves there too
#
# First invocation triggers ensure-runtime.sh (blocks ~60-90s on fresh install).
# Subsequent invocations are instant — just PATH + env var wiring.
# BRAIN_MCP_BOOTSTRAP_WAIT_S is consumed by mcp-launch.sh before sourcing us:
# the host manifest may extend its cold-start wait (default 25s; Codex 300s).
# Keep the host startup timeout larger to leave time for MCP initialization.

# Resolve plugin dir from whichever .sh sourced us.
# ${BASH_SOURCE:-$0}, NOT ${BASH_SOURCE[0]}: the subscripted form resolves to
# the CWD under zsh and is a fatal "Bad substitution" under dash, so a sourcer
# in either shell silently loaded the wrong tree — or died. Bare $BASH_SOURCE
# is element 0 in bash, and $0 is the correct fallback everywhere else.
# resolve-brain-db.sh carries the same idiom for the same reason.
_BRAIN_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"
export PLUGIN_DIR="$(cd "$_BRAIN_ENV_DIR/../.." && pwd)"

# API-key + user-config resolution — owned by api-key-env.sh, shared with
# boot-brain.sh (which runs before this file is reachable).
# Readability-guarded: `.` on a missing file is a special-builtin failure that
# `|| true` cannot rescue — it exits the shell outright under dash and under
# `set -e` (brain-daemon sets it), before any resolution can happen. A damaged
# install must degrade, not take the daemon down.
if [ -r "$_BRAIN_ENV_DIR/api-key-env.sh" ]; then
    . "$_BRAIN_ENV_DIR/api-key-env.sh"
    # Sourced and used in the same branch: an undefined function is a 127 that
    # `set -e` (brain-daemon sets it) turns into a dead daemon, and a flag
    # recording "the source worked" is just this branch, spelled twice.
    brain_source_user_env
    brain_api_key_from_plugin_option
else
    echo "[brain-env] WARN: api-key-env.sh missing or unreadable (damaged install)" >&2
    echo "[brain-env] — user config and API key will NOT be loaded this run" >&2
fi

# Source the canonical user config (~/.config/brain/env) so secrets and
# identity tokens (ANTHROPIC_API_KEY, BRAIN_OPERATOR_NAME, BRAIN_AGENT_NAME, ...)
# propagate into both the hook scripts and the launchd-spawned daemon
# launcher. Unconditional here (unlike boot-brain.sh's key-only read): every
# variable in the file is wanted. A value in the file wins over the process
# env for everything downstream of this line — so a BRAIN_DB_DIR knob line
# re-points even a plist-baked daemon env; the resolver's ladder then
# re-confirms the same choice.
# Daemon rendezvous port — set EARLY, before the runtime-bootstrap guard below,
# since it depends only on the uid (not the venv). The ONE shell source of the
# per-user port; shell scripts + hook Python read $BRAIN_DAEMON_PORT (the formula
# survives only as a resilience fallback). An explicit value (shell / the user
# env above) wins. Python inside servers/ uses daemon_config.DAEMON_PORT.
export BRAIN_DAEMON_PORT="${BRAIN_DAEMON_PORT:-$((47200 + $(id -u) % 100))}"

# BRAIN_INSTANCE — multi-entity keying (eval entities, sandbox installs).
# Never set here: production runs unkeyed. An entity launcher sets it together
# with its OWN BRAIN_DAEMON_PORT and BRAIN_DB_DIR; daemon_config then suffixes
# every /tmp rendezvous path and the launchd label, refuses production's port
# value, and the daemon's bind-time DB lock rejects a second writer on any one
# brain.db. install-daemon-service.sh no-ops under it (entities never launchd),
# and resolve-brain-db.sh takes its absolute BRAIN_DB_DIR as the location
# without walking the ladder or writing resolved.env (entities never adopt a
# brain).

# Ensure runtime is installed (idempotent, fast-path on sentinel)
if ! "$_BRAIN_ENV_DIR/ensure-runtime.sh"; then
    echo "[brain-env] FATAL: runtime bootstrap failed — brain disabled" >&2
    # Don't `exit` — we're sourced. Let the calling hook handle it.
    return 1 2>/dev/null || exit 1
fi

# Wire the venv as the authoritative Python
export BRAIN_PYTHON="$PLUGIN_DIR/venv/bin/python"
# Idempotent prepend: a chained environment (in-place daemon reloads re-source
# this file in the same process's env) must not grow PATH by one entry per
# generation.
case ":$PATH:" in
    *":$PLUGIN_DIR/venv/bin:"*) ;;
    *) export PATH="$PLUGIN_DIR/venv/bin:$PATH" ;;
esac

# Ensure nothing in the shell environment overrides venv resolution
unset PYTHONHOME

# ── Process names ─────────────────────────────────────────────────────────
# Activity Monitor, top and pgrep show the kernel's process name: the final
# filename the exec resolved to. Through venv/bin/python (a symlink into the
# standalone interpreter) every brain process read `python3.11`, so a 25 GB
# daemon and a stray test run were indistinguishable. A HARD LINK to the
# interpreter binary is the same file under a different final name: it keeps
# the venv (pyvenv.cfg is found from the venv/bin symlink) and the dylib load
# path (@executable_path/../lib, relative to the link's own directory) while
# the kernel reports the role. Names stay within 15 chars — the Linux comm
# limit (macOS shows 16). Per-session roles carry the last 4 characters of
# the session's short (trace_contract.session_short, the hyphen-free tail
# self_presence shows) — all of it would not fit; tests/test_deploy_contract.py
# pins the tag to the owner:
#
#   brain_python_as Entity-mcp "$CLAUDE_CODE_SESSION_ID"  → venv/bin/Entity-mcp-3207
#
# Degrades to $BRAIN_PYTHON (still runs, just named python3.11) whenever the
# link cannot be made — a naming failure must never take a launcher down.
brain_python_as() {
    _role="$1"
    _name="$1"
    [ -n "${2:-}" ] && _name="$1-$(printf '%s' "$2" | tr -d '-' | tail -c 4)"
    if [ ${#_name} -gt 15 ]; then
        echo "[brain-env] WARN: process name '$_name' exceeds 15 chars — running as python" >&2
        echo "$BRAIN_PYTHON"; return 0
    fi
    if [ ! -x "$BRAIN_PYTHON" ]; then
        echo "$BRAIN_PYTHON"; return 0
    fi
    _real="$(readlink -f "$BRAIN_PYTHON" 2>/dev/null)" || _real=""
    _bindir="${_real%/*}"
    _venv_bin="${BRAIN_PYTHON%/*}"
    if [ -n "$_real" ] && [ ! -e "$_bindir/$_name" ]; then
        ln "$_real" "$_bindir/$_name" 2>/dev/null || true
    fi
    if [ -e "$_bindir/$_name" ] && [ ! -e "$_venv_bin/$_name" ]; then
        ln -s "$_bindir/$_name" "$_venv_bin/$_name" 2>/dev/null || true
    fi
    if [ ! -x "$_venv_bin/$_name" ]; then
        echo "[brain-env] WARN: could not name process '$_name' — running as python" >&2
        echo "$BRAIN_PYTHON"; return 0
    fi
    # Per-session names leave one link per session behind. Sweep this role's
    # siblings whose session has no live process (pgrep -x matches the kernel
    # name). Skipped without pgrep: a sweep that cannot see live processes
    # would unlink them.
    if [ -n "${2:-}" ] && command -v pgrep >/dev/null 2>&1; then
        # find, not a glob: a zsh consumer aborts on an unmatched glob.
        for _l in $(find "$_venv_bin" -maxdepth 1 -type l -name "$_role-*" 2>/dev/null || true); do
            _n="${_l##*/}"
            [ "$_n" = "$_name" ] && continue
            pgrep -x "$_n" >/dev/null 2>&1 && continue
            rm -f "$_l" "$_bindir/$_n"
        done
    fi
    echo "$_venv_bin/$_name"
}
export BRAIN_PYTHON_DAEMON="$(brain_python_as Entity-daemon)"
export BRAIN_PYTHON_DASH="$(brain_python_as Entity-dash)"
export BRAIN_PYTHON_HOOK="$(brain_python_as Entity-hook)"

# S1 Scribe associated stubs — ON renders the encoder's subconscious: K≈5
# nodes production recall ranks nearest the window's unencoded messages that
# the catalog doesn't already show, as the catalog's LAST entries tagged
# [associated] (encode._associated_stub_ids → build_node_catalog). Default
# OFF until an eval gates it (input changes ship flag-gated — the
# view-policy flag is the precedent). Gating harness caveat: the current
# encoder_prompt_ab/reassembly evals rebuild catalogs directly and CANNOT
# see this flag — the stubs arm needs retrieval wiring (rides the co-review
# ship package); only paths through run_encoding (longmem replay, live)
# exercise it end-to-end. Flip-on also wants the s1e prompt version that
# teaches [associated] active first (a loud s1e_associated_stubs_untaught
# error fires otherwise). Read by the DAEMON's S1 Scribe —
# takes effect at daemon restart. Enable: set to "1" and restart.
export BRAIN_S1E_ASSOCIATED_STUBS="0"

# S1 Scribe operating-guide preamble — ON swaps the encoder's one-line
# preamble for the lists-first sentence (the encoder's first reply carries
# its fetch / changes / targets / new lists as text, then the tool call in the
# same reply). Pairs with the guide s1e + s1e_gist candidates
# (eval/candidate_prompts/s1e_guide_v1_*, s1e_gist_guide_v1_*); alone it asks
# for lists the prompt never defines. Default OFF — an input change ships
# flag-gated so the eval can A/B it. Read by the DAEMON's S1 Scribe at encode
# time (encode._build_user_content). Enable: set to "1" and restart.
export BRAIN_S1E_LISTS_PREAMBLE="0"
