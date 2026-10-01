#!/bin/sh
# brain — the ONE definition of "where does ANTHROPIC_API_KEY come from?".
# POSIX sh — sourced (never executed), so it runs in the consumer's shell:
# bash hooks, zsh, dash (/bin/sh on Linux). Keep bashism-free
# (TestResolverChainPortability).
#
# Sourced by brain-env.sh (every hook) and boot-brain.sh (which runs BEFORE
# brain-env is reachable — the bootstrap ordering that forced the copy in the
# first place). One home for the env-file source and the dual-casing
# CLAUDE_PLUGIN_OPTION fallback: a casing fix that lands in only one copy
# re-creates the 2026-07-15 failure where a user filled in the plugin's key
# field and the daemon still ran keyless.
#
# Four functions, because the callers legitimately differ on WHEN they read:
#   brain_user_env_file               — the canonical config path
#   brain_source_user_env             — source it (set -a: every var exported)
#   brain_api_key_from_plugin_option  — userConfig fallback, both casings
#   brain_append_user_env             — append a line to it, safely
#
# POLICY stays with the callers: brain-env.sh sources the file unconditionally
# because it needs every variable in it, not just the key; boot-brain.sh reads
# it only when the key is missing and mirrors a userConfig-resolved key back
# into the file for the launchd-spawned daemon.
#
# Written with `if` blocks and no bare `&&` statements: brain-daemon runs under
# `set -e` and sources this transitively, where a failing test as a standalone
# command would abort the whole resolver.

# The canonical user config file. One definition — every consumer that names
# this path (resolvers, installers, the boot notices) is naming the same file.
# Assigned at source time and read as a VARIABLE on the hook hot path: every
# hook sources this, and wrapping a pure parameter expansion in $(...) is a
# fork per call for nothing. The function stays for callers that want the
# accessor form.
BRAIN_USER_ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/brain/env"

brain_user_env_file() {
    printf '%s' "$BRAIN_USER_ENV_FILE"
}

# Source the user config so secrets and identity tokens (ANTHROPIC_API_KEY,
# BRAIN_OPERATOR_NAME, BRAIN_AGENT_NAME, ...) propagate to everything
# downstream. NOTE: plain sourcing OVERWRITES an already-set shell value — a
# variable in this file always wins over the process env from here on.
brain_source_user_env() {
    local _brain_env_f _brain_errexit
    _brain_env_f="${1:-$BRAIN_USER_ENV_FILE}"
    if [ -f "$_brain_env_f" ]; then
        # The file is hand-edited (2026-08-12: the operator's own), so a
        # failing command in it is a matter of time. Under the daemon
        # launcher's `set -e`, errexit stays ACTIVE inside a sourced file on
        # macOS bash 3.2 even when the `.` call is guarded with `|| true`
        # (verified 2026-08-13) — one bad line would crash-loop the daemon on
        # launchd's throttle while hooks (no set -e) keep working. Disable
        # errexit around the source; restore it only if the caller had it on.
        case $- in
            *e*) _brain_errexit=1 ;;
            *)   _brain_errexit="" ;;
        esac
        set +e
        set -a
        . "$_brain_env_f"
        set +a
        if [ -n "$_brain_errexit" ]; then
            set -e
        fi
    fi
    return 0
}

# Append a line to the user config file, safely. Hand-edited files whose last
# line has NO trailing newline exist in production (2026-08-12: the operator's
# own), and a bare `>>` glues the new line onto the old one —
# `BRAIN_AGENT_NAME=xxxANTHROPIC_API_KEY=sk-...`, silently corrupting both.
# Leading \n costs one blank line in the normal case and is never wrong.
brain_append_user_env() {
    local _brain_env_f
    _brain_env_f="$BRAIN_USER_ENV_FILE"
    mkdir -p "$(dirname "$_brain_env_f")" 2>/dev/null
    printf '\n%s\n' "$1" >> "$_brain_env_f" || return 1
    # chmod failure is a FAILURE, not a detail to swallow: the caller announces
    # "mode 600" to the operator, and announcing that over a world-readable file
    # holding an API key is worse than saying nothing. Let chmod's own stderr
    # through and report it.
    chmod 600 "$_brain_env_f" || return 1
    return 0
}

# Additive userConfig fallback: if the env file / shell didn't supply the key,
# take it from the plugin-config value CC injects as CLAUDE_PLUGIN_OPTION_<KEY>
# (per plugins-reference). Env file / shell still win. Both casings checked —
# the doc doesn't pin <KEY>'s case and a wrong name would be a silent no-op.
brain_api_key_from_plugin_option() {
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        if [ -n "${CLAUDE_PLUGIN_OPTION_API_KEY:-}" ]; then
            export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_API_KEY"
        elif [ -n "${CLAUDE_PLUGIN_OPTION_api_key:-}" ]; then
            export ANTHROPIC_API_KEY="$CLAUDE_PLUGIN_OPTION_api_key"
        fi
    fi
    return 0
}
