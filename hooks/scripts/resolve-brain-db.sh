#!/bin/sh
# Shared brain DB resolution — sourced by all hook scripts.
# Boot sets BRAIN_DB_DIR once; other hooks reuse it or re-resolve.
#
# POSIX sh (the shebang is the lint dialect — this file is only ever SOURCED,
# so it runs in the consumer's shell): bash hooks, zsh skills/terminals, and
# dash (/bin/sh on Linux, D-3). Keep the whole chain (this file, brain-env.sh,
# api-key-env.sh) bashism-free — TestResolverChainPortability holds the line.
# Consumers must source from a SIBLING script: under dash $0 is the caller's
# path (there is no BASH_SOURCE), so self-location works only when the caller
# lives in this directory — which every hook does.
#
# Usage: source "$(dirname "$0")/resolve-brain-db.sh"
# After sourcing: BRAIN_DB_DIR, BRAIN_SERVER_DIR, PLUGIN_ROOT, BRAIN_PYTHON are set.
# If no brain.db found, BRAIN_DB_DIR is empty — caller should exit 0.

# ── A hook machine with no route to the daemon: nothing to resolve ─────────
# Cowork runs plugin command hooks in a cloud Linux VM that cannot reach the
# operator's Mac (brain node bc9e9e30): no runtime is worth building there
# and no brain belongs there. The marker is host_contract's `remote_hooks`
# (hook_common.REMOTE_HOOK_MARKERS is the Python copy;
# tests/test_remote_hook_machine.py runs every manifest command against the
# contract's markers). macOS is never such a machine. Here, before brain-env.sh
# and its runtime bootstrap, so every hook ends before the ladder too. Not in
# brain-env.sh: ./dev sources that file directly, and a developer on such a
# machine needs a runtime and no daemon. `exit`, not `return` — a deliberate
# break from this chain's "we're sourced, let the caller decide" rule, and only
# off macOS: no hook can do anything here, and boot-brain.sh says so once, in
# its own pass. The env compare is a builtin, so a local host pays no fork;
# uname runs only when the marker matches.
case "${CLAUDE_CODE_ENTRYPOINT:-}" in
  remote_cowork)
    if [ "$(uname -s)" != Darwin ]; then
      echo "[resolve-brain-db] no route to the brain daemon from this machine (CLAUDE_CODE_ENTRYPOINT=$CLAUDE_CODE_ENTRYPOINT) — nothing to resolve here" >&2
      exit 0
    fi
    ;;
esac

# Ensure the isolated runtime is installed and BRAIN_PYTHON / PATH point at it.
# First hook invocation on a fresh install blocks ~15s downloading uv + Python
# + deps. Every subsequent invocation is ~8ms (sentinel fast path).
# ${BASH_SOURCE:-$0}: BASH_SOURCE is unset under POSIX sh — fall back to $0
# instead of silently resolving to cwd (observed: "brain-env.sh doesn't exist").
# NO [0] subscript: bare $BASH_SOURCE is element 0 in bash, while the array
# subscript is a Bad substitution under dash (Linux /bin/sh) that kills the
# fallback entirely.
_RESOLVE_DIR="$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)"

# brain-env.sh sources api-key-env.sh (guarded) and locates it with the same
# ${BASH_SOURCE:-$0} idiom this file uses, so the config-path accessor is in
# scope for the knob rung below. No second source here: two sourcing sites for
# one helper is the duplication this step removes, not a safety net.
# `.`, not `source`: `source` is a bashism — under dash it fails 127 on every
# source (fatal under set -e; a silent permanent damaged-install fallback
# without). Readability-guarded because `.` on a missing file is a
# special-builtin failure `|| true` cannot rescue — dash and set -e both exit
# outright. The guard keeps a damaged install degrading to the literal-path
# fallback below, mirroring brain-env.sh's own guard around api-key-env.sh.
if [ -r "$_RESOLVE_DIR/brain-env.sh" ]; then
  . "$_RESOLVE_DIR/brain-env.sh"
else
  echo "[resolve-brain-db] WARN: brain-env.sh missing or unreadable (damaged install)" >&2
fi

# Always resolve from script location — never use CLAUDE_PLUGIN_ROOT cache.
# The cache can be stale (old code, old socket protocol). Working dir is truth.
PLUGIN_ROOT="$(cd "$_RESOLVE_DIR/../.." && pwd)"
BRAIN_SERVER_DIR="$PLUGIN_ROOT/servers"

# Persist the resolution for non-hook shells. CLAUDE_PLUGIN_DATA and
# CLAUDE_PLUGIN_OPTION_* exist ONLY in hook executions — a skill's Bash shell,
# a user terminal, or launchd sees none of them, so without this file they
# cannot find a brain that lives at the $CLAUDE_PLUGIN_DATA default. Hooks
# persist what they resolve; everyone else reads it (step 4b below).
# Generated file — user config stays untouched in ../brain/env.
# BRAIN_RESOLVE_NO_PERSIST: consumers of the resolution that are not an
# authority on it (brain-daemon — launchd hands it a baked snapshot) set
# this so a relaunch can't clobber a fresher hook-persisted record with
# stale baked state.
_brain_persist_state() {
  [ -z "${BRAIN_RESOLVE_NO_PERSIST:-}" ] || return 0
  [ -n "$BRAIN_DB_DIR" ] && [ -f "$BRAIN_DB_DIR/brain.db" ] || return 0
  local _dir _state _content _existing
  _dir="${XDG_CONFIG_HOME:-$HOME/.config}/brain"
  _state="$_dir/resolved.env"
  # Values are QUOTED — the file is `.`-sourced by consumers (skills,
  # terminals), and an unquoted path with a space breaks sourcing silently.
  _content="# generated by resolve-brain-db.sh — do not edit (user config lives in ./env)
BRAIN_DB_DIR='$BRAIN_DB_DIR'
PLUGIN_ROOT='$PLUGIN_ROOT'"
  # $(cat ...), not the bash-only $(<...): under dash that expands to empty,
  # so the comparison always mismatched and every source rewrote the file.
  # One cat fork, only when a prior record exists.
  _existing=""
  [ -f "$_state" ] && _existing="$(cat "$_state")"
  if [ "$_content" != "$_existing" ]; then
    # || true: sourced under the daemon launcher's set -e — persisting the
    # record is best-effort and must never abort the resolver.
    mkdir -p "$_dir" 2>/dev/null || true
    # tmp + mv: atomic replace — concurrent hooks/skills may be sourcing it.
    printf '%s\n' "$_content" > "$_state.tmp.$$" 2>/dev/null \
      && mv -f "$_state.tmp.$$" "$_state" 2>/dev/null \
      || rm -f "$_state.tmp.$$" 2>/dev/null
  fi
}

# ── The two durable adoption channels, read in ONE place each ──────────────
# Both are read here AND by the launchd installers' identity guard (which may
# only let a durable channel re-point an installed service's brain). They were
# read in one place because a second copy of either drifts: the stdout discard
# below is the kind of fix that lands in one copy and not the other.

# The ~/.config/brain/env BRAIN_DB_DIR knob, read RAW: no `~` expansion and no
# absolute-only filter, because that is rung 1b's POLICY (below) and not what
# the installers' identity comparison wants. Sourced in a subshell — the file
# is shell grammar, so one format, one parser — with the file's own stdout
# discarded, since a user `echo` in it must not pollute the value. Empty when
# the file or the knob is absent.
#
# ONE reader for both shell-grammar config files (the user knob and the
# generated resolved.env): same grammar, same subshell reset, same stdout
# discard, so a fix to any of those lands once.
_brain_db_dir_from() {
  [ -f "$1" ] || return 0
  (BRAIN_DB_DIR=''; { . "$1"; } >/dev/null 2>&1; printf '%s' "$BRAIN_DB_DIR")
}

# Reading the knob must depend on NOTHING but $HOME. api-key-env.sh owns the
# config path, but if it could not be sourced (a damaged install) we fall back
# to the literal expression rather than failing: a brain the user named must
# not become unreachable because a sibling file went missing. brain-env.sh
# already WARNed about the damaged install, so this stays silent.
brain_config_knob_db_dir() {
  if command -v brain_user_env_file >/dev/null 2>&1; then
    _brain_db_dir_from "$BRAIN_USER_ENV_FILE"
  else
    _brain_db_dir_from "${XDG_CONFIG_HOME:-$HOME/.config}/brain/env"
  fi
}

# The brain dir an installed launchd service definition bakes in (its
# BRAIN_DB_DIR), read RAW from the plist file given. Compared by the
# installers' identity guard; read by the adoption net below as a record of
# where the brain lived — 9.6.0 wrote no resolved.env, so for a brain it
# parked the service definition is the only record.
brain_plist_db_dir() {
  [ -f "$1" ] || return 0
  sed -n '/<key>BRAIN_DB_DIR<\/key>/{n;s|.*<string>\(.*\)</string>.*|\1|p;}' "$1" | head -1
}

# The brain path a user set in the plugin's settings. Claude Code injects it as
# CLAUDE_PLUGIN_OPTION_BRAIN_PATH; the casing is unpinned by the plugins
# reference, so both are checked — the same rule api-key-env.sh applies to the
# key, and a one-sided fix here is the same silent no-op.
brain_config_option_db_dir() {
  printf '%s' "${CLAUDE_PLUGIN_OPTION_BRAIN_PATH:-${CLAUDE_PLUGIN_OPTION_brain_path:-}}"
}

# userConfig: the "I already have a brain, here's where" install answer.
# Shell/env BRAIN_DB_DIR still wins (fill only when unset), mirroring the
# api_key rule. NOTE: reaches the daemon on the hook/Popen path; a
# launchd-spawned daemon won't see CLAUDE_PLUGIN_OPTION_*. An entity never
# takes it: that setting names the operator's brain (see the entity rule below).
if [ -z "${BRAIN_DB_DIR:-}" ] && [ -z "${BRAIN_INSTANCE:-}" ]; then
  # Expanded inline, not through the accessor: this runs on every hook and a
  # $(...) around a pure parameter expansion is a fork for nothing. The
  # accessor exists for launchd-install.sh, which needs the function form.
  BRAIN_DB_DIR="${CLAUDE_PLUGIN_OPTION_BRAIN_PATH:-${CLAUDE_PLUGIN_OPTION_brain_path:-}}"
  if [ -n "$BRAIN_DB_DIR" ]; then
    # Honor the chosen location even if it doesn't exist yet — create it so the
    # resolution chain adopts it instead of silently falling through to the
    # default brain. A typo then surfaces as an empty brain at the named path
    # (visible), not a hidden default elsewhere. || true: set -e daemon path —
    # an uncreatable hint degrades to the resolution chain, never aborts.
    mkdir -p "$BRAIN_DB_DIR" 2>/dev/null || true
    export BRAIN_DB_DIR
  fi
fi

# The service-owned data home (D-13): host-independent, never renames. New
# brains are born here, relocation targets it, and boot's notices print it.
# Exported so consumers (boot, relocate-brain.sh) reuse it instead of
# re-deriving the literal — three copies of this expression already drifted
# apart once.
BRAIN_XDG_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/brain"
export BRAIN_XDG_DIR

# A resolved brain living under a host's plugin-data root is parked on a
# trapdoor: `claude plugin uninstall` deletes that whole tree by default
# (--keep-data is opt-in; a plugin update is safe). Matched on the PATH,
# not the rung that resolved it — a knob, env hint, or resolved.env pointing
# into the same dir is parked all the same. The roots ARE the adoption net's
# scan roots (the parent of $CLAUDE_PLUGIN_DATA — so renamed SIBLING plugin
# dirs match — plus the default plugins-data root), never a bare unanchored
# glob: a dev checkout that merely contains "plugins/data" must not warn.
# BRAIN_PARKED_ACK (user knob, ~/.config/brain/env) is the informed-stay
# opt-out. Sets BRAIN_HOST_PARKED for boot to surface relocation; never
# moves anything itself.
_brain_check_parked() {
  BRAIN_HOST_PARKED=""
  if [ -z "${BRAIN_PARKED_ACK:-}" ]; then
    # ${...%/} / ${...%/*}: string builtins only — this runs on every hook.
    # The sentinel is a path no real dir can live under (/dev/null is a
    # character device), so an unset CPD can never match an absolute path.
    _cpd="${CLAUDE_PLUGIN_DATA:-}"; _cpd="${_cpd%/}"
    _cpd_root="${_cpd%/*}"
    case "$BRAIN_DB_DIR" in
      "$HOME/.claude/plugins/data"/*|\
      "${_cpd:-/dev/null/no-cpd}"|\
      "${_cpd_root:-/dev/null/no-cpd}"/*)
        BRAIN_HOST_PARKED="$BRAIN_DB_DIR" ;;
    esac
  fi
  export BRAIN_HOST_PARKED
}

# Clear any inherited candidate first — a stale value from a parent process
# must not survive past a successful resolution.
BRAIN_ADOPTION_CANDIDATE=""
BRAIN_MISSING_BRAIN=""

# An entity (BRAIN_INSTANCE: an eval arm, a smoke sandbox, the test suite)
# never adopts a brain, as it never touches launchd (launchd-install.sh). The
# ladder below exists to FIND a user's brain, and an entity that walks it lands
# on the operator's. An entity's brain is its own absolute BRAIN_DB_DIR,
# created if missing (its birthplace). Without one — or with a relative one,
# which would resolve against each consumer's cwd — it has none, and
# daemon_config refuses to start it. It never writes resolved.env: that record
# belongs to the operator's brain.
if [ -n "${BRAIN_INSTANCE:-}" ]; then
  # || true: set -e daemon path — an uncreatable dir must not abort the resolver.
  case "${BRAIN_DB_DIR:-}" in
    /*) [ -d "$BRAIN_DB_DIR" ] || mkdir -p "$BRAIN_DB_DIR" 2>/dev/null || true ;;
    *)  BRAIN_DB_DIR="" ;;
  esac
  export BRAIN_DB_DIR BRAIN_SERVER_DIR PLUGIN_ROOT
  _brain_check_parked
  return 0 2>/dev/null || true
fi

# If BRAIN_DB_DIR already set and valid (e.g. from boot), skip resolution.
if [ -n "$BRAIN_DB_DIR" ] && [ -f "$BRAIN_DB_DIR/brain.db" ]; then
  export BRAIN_DB_DIR BRAIN_SERVER_DIR PLUGIN_ROOT
  _brain_check_parked
  _brain_persist_state
  return 0 2>/dev/null || true
fi

# Full resolution chain (runs at boot or if env not set).
#
# Priority is: explicit-choice > existing-brain-found > create-fresh.
# Auto-create only happens at the XDG service location (D-13), never at a
# legacy path (existing users who never migrated must opt in explicitly) — and
# never while a candidate brain sits unreachable at an old default (the
# adoption net below).
DB_DIR=""

# 1. Explicit override — adopted outright only when a brain actually lives
#    there. A dir WITHOUT brain.db is demoted to a hint of last resort (used
#    below only if no rung finds a real brain): a daemon relaunched off a
#    stale plist-baked path must follow the ladder to the brain that moved
#    (resolved.env, step 4b), not birth a shadow brain at the abandoned dir —
#    while a genuinely fresh install (baked dir exists, daemon creates
#    brain.db on first load) still lands on the hint.
_ENV_HINT=""
if [ -n "$BRAIN_DB_DIR" ] && [ -d "$BRAIN_DB_DIR" ]; then
  if [ -f "$BRAIN_DB_DIR/brain.db" ]; then
    DB_DIR="$BRAIN_DB_DIR"
  else
    _ENV_HINT="$BRAIN_DB_DIR"
  fi
fi

# 1b. The user config knob (~/.config/brain/env, BRAIN_DB_DIR=...) — the
#     durable adoption channel the boot notice points users at, and the same
#     rung daemon_config.resolve_db_dir reads second. The READ is
#     brain_config_knob_db_dir above (shared with the installers' identity
#     guard); the POLICY is here. A knob dir WITH brain.db is adopted
#     outright; WITHOUT one it is the user's explicit choice of birthplace —
#     held as a hint that beats a stale plist-baked hint at the last-resort
#     rung (mkdir happens only there, so reading the knob has no side
#     effects). A `~/` prefix is expanded (the Python reader expands it too —
#     two readers, one grammar); any other non-absolute value is ignored,
#     since it would resolve against each consumer's cwd and split the system.
_CFG_HINT=""
if [ -z "$DB_DIR" ]; then
  _cfg_db_dir="$(brain_config_knob_db_dir)"
  case "$_cfg_db_dir" in
    "~/"*) _cfg_db_dir="$HOME/${_cfg_db_dir#"~/"}" ;;
  esac
  case "$_cfg_db_dir" in
    /*)
      if [ -f "$_cfg_db_dir/brain.db" ]; then
        DB_DIR="$_cfg_db_dir"
      else
        _CFG_HINT="$_cfg_db_dir"
      fi
      ;;
    *) _cfg_db_dir="" ;;
  esac
fi

# 2b. The XDG service location — where new brains are born (D-13). A brain
#     already living there is the native case and beats every legacy rung.
if [ -z "$DB_DIR" ] && [ -f "$BRAIN_XDG_DIR/brain.db" ]; then
  DB_DIR="$BRAIN_XDG_DIR"
fi

# 3. Claude Code plugin data location — ADOPTION ONLY (pre-D-13 installs
#    were born here; $CLAUDE_PLUGIN_DATA is keyed to the plugin name, so a
#    rename orphans it — see the adoption net below).
if [ -z "$DB_DIR" ] && [ -n "${CLAUDE_PLUGIN_DATA:-}" ] && [ -f "$CLAUDE_PLUGIN_DATA/brain/brain.db" ]; then
  DB_DIR="$CLAUDE_PLUGIN_DATA/brain"
fi

# 4. Legacy local path (~/AgentsContext/brain/) — adoption only, for
#    pre-CLAUDE_PLUGIN_DATA installs. New installs land at $BRAIN_XDG_DIR.
if [ -z "$DB_DIR" ] && [ -f "$HOME/AgentsContext/brain/brain.db" ]; then
  DB_DIR="$HOME/AgentsContext/brain"
fi

# 4b. Persisted state from a prior hook run (written by _brain_persist_state).
#     The path a non-hook shell (skill Bash, terminal, launchd) lands on when
#     the brain lives at the $CLAUDE_PLUGIN_DATA default it cannot see.
#     Read by sourcing in a subshell — the SAME mechanism every other consumer
#     of this file uses (skills `.`-source it), so one format, one parser.
_state_db_dir=""
if [ -z "$DB_DIR" ]; then
  _state_env="${XDG_CONFIG_HOME:-$HOME/.config}/brain/resolved.env"
  if [ -f "$_state_env" ]; then
    _state_db_dir="$(_brain_db_dir_from "$_state_env")"
    [ -n "$_state_db_dir" ] && [ -f "$_state_db_dir/brain.db" ] && DB_DIR="$_state_db_dir"
  fi
fi

# 4c. The demoted explicit hints: no rung found an existing brain, so an
#     explicitly-named empty dir is where a new one belongs — beats
#     auto-creating at the standard location. The config knob (a durable
#     user choice) beats the env hint (possibly a stale plist-baked path).
if [ -z "$DB_DIR" ] && [ -n "$_CFG_HINT" ]; then
  # the hint becomes the birthplace only here — mkdir at selection time, so
  # a typo'd knob surfaces as a visible empty brain at the named path.
  # || true: this file is sourced under the daemon launcher's set -e, and a
  # failed create (read-only parent) must not abort the whole resolver.
  mkdir -p "$_CFG_HINT" 2>/dev/null || true
  DB_DIR="$_CFG_HINT"
fi
if [ -z "$DB_DIR" ] && [ -n "$_ENV_HINT" ]; then
  DB_DIR="$_ENV_HINT"
fi

# 6. First-run create at the XDG service location — guarded by the adoption
#    net: if a brain sits unreachable at an old host-owned default (plugin
#    renamed/reinstalled, so $CLAUDE_PLUGIN_DATA points elsewhere), creating
#    fresh here would be the silent-amnesia failure D-13 exists to kill.
#    Refuse instead: leave BRAIN_DB_DIR empty (every consumer's existing
#    no-brain path) and name the candidate so boot can surface adoption.
#    The net never fires past an explicit choice — a knob/hint/option dir
#    was already taken above. Never moves or deletes anything.
#    A record of a brain that is no longer there refuses the same way: the
#    resolved.env a hook wrote, or the brain dir the installed daemon service
#    bakes in, names where this machine's brain lived, so a brain existed, and
#    a fresh one would hide its loss (an uninstall without --keep-data deletes
#    a parked brain while its daemon may still hold it open).
#    BRAIN_MISSING_BRAIN names that dir for boot to surface.
if [ -z "$DB_DIR" ]; then
  _scan_cpd_root=""
  [ -n "${CLAUDE_PLUGIN_DATA:-}" ] && _scan_cpd_root="$(dirname "$CLAUDE_PLUGIN_DATA")"
  # find, not a glob: a no-match glob is a fatal error in a SOURCED file
  # under zsh nomatch / bash failglob — it would abort the whole resolver.
  for _scan_root in "$_scan_cpd_root" "$HOME/.claude/plugins/data"; do
    [ -n "$_scan_root" ] && [ -d "$_scan_root" ] || continue
    _cand="$(find "$_scan_root" -mindepth 3 -maxdepth 3 \
               -path '*/brain/brain.db' 2>/dev/null | head -n 1)"
    if [ -n "$_cand" ]; then
      BRAIN_ADOPTION_CANDIDATE="$(dirname "$_cand")"
      break
    fi
  done
  # The service label is daemon_launch.LAUNCHD_LABEL; an entity returned above.
  # Only a dir that is GONE counts: a first install renders the service over
  # the standard dir it just created, a moment before the daemon writes
  # brain.db there. resolved.env needs no such filter — it is written only
  # once a brain.db exists.
  _svc_db_dir="$(brain_plist_db_dir "$HOME/Library/LaunchAgents/com.brain.daemon.plist")"
  [ -z "$_svc_db_dir" ] || [ ! -d "$_svc_db_dir" ] || _svc_db_dir=""
  if [ -z "$BRAIN_ADOPTION_CANDIDATE" ] && [ -n "$_state_db_dir$_svc_db_dir" ]; then
    BRAIN_MISSING_BRAIN="${_state_db_dir:-$_svc_db_dir}"
  elif [ -z "$BRAIN_ADOPTION_CANDIDATE" ]; then
    DB_DIR="$BRAIN_XDG_DIR"
    # a failed create (read-only ~/.local, ...) must surface as the boot
    # hook's "auto-create failed" guidance, not a path that doesn't exist
    mkdir -p "$DB_DIR" 2>/dev/null || DB_DIR=""
  fi
fi

BRAIN_DB_DIR="$DB_DIR"
export BRAIN_DB_DIR BRAIN_SERVER_DIR PLUGIN_ROOT BRAIN_ADOPTION_CANDIDATE BRAIN_MISSING_BRAIN
_brain_check_parked
_brain_persist_state
