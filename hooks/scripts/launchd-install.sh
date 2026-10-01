#!/bin/bash
# brain — the ONE definition of "how a LaunchAgent gets materialized, loaded,
# and reloaded on drift". Sourced by install-daemon-service.sh and
# ensure-dashboard.sh; the label is an ARGUMENT (D-11: the service layer never
# renames, and a helper that bakes one label is a helper for one service).
#
#
# MECHANISM here, POLICY in the callers — they legitimately differ:
#   install-daemon-service.sh  install-only; verifies with launchctl afterwards
#   ensure-dashboard.sh        also probes liveness, kickstarts a loaded-but-down
#                              service, and verifies by waiting for :PORT
#
# Requires resolve-brain-db.sh to have been sourced first: the identity guard
# reads the two durable adoption channels through that file's
# brain_config_knob_db_dir / brain_config_option_db_dir. If it was not, both
# read empty and the guard keeps the INSTALLED brain dir — the safe direction
# (never hijack a service onto another brain), never a silent re-point.

# Computed once at source time, not per call: `$(id -u)` is a fork + exec, and
# install-daemon-service.sh runs SYNCHRONOUSLY on the SessionStart budget.
_BRAIN_LAUNCHD_DOMAIN="gui/$(id -u)"

brain_launchd_domain() {
    printf '%s' "$_BRAIN_LAUNCHD_DOMAIN"
}

brain_launchd_target() {
    printf '%s/Library/LaunchAgents/%s.plist' "$HOME" "$1"
}

# True iff launchd has this label loaded in the user's GUI domain.
brain_launchd_managed() {
    launchctl print "$_BRAIN_LAUNCHD_DOMAIN/$1" >/dev/null 2>&1
}

# Entities (BRAIN_INSTANCE set — eval brains, install smoke runs) never use
# launchd: they run daemon_server as a child process on their own port. The
# loaded service is uid-wide while the plist path is $HOME-relative, so an
# entity under a scratch $HOME reads "no plist at my path" as drift and would
# bootout + re-bootstrap PRODUCTION's job onto its own tree — persistent
# across reboots. Every caller checks this first; the mutators below refuse
# regardless, so a caller that forgets cannot do the damage.
brain_launchd_entity() {
    [ -n "${BRAIN_INSTANCE:-}" ]
}
_brain_launchd_refuse_entity() {
    # $1 what  $2 log prefix — true (and loud) when the caller must stop
    brain_launchd_entity || return 1
    echo "${2:-[launchd-install]} refusing to $1 under BRAIN_INSTANCE=$BRAIN_INSTANCE — entities never touch launchd" >&2
    return 0
}

# True iff tree $1 is a strictly newer release of the same plugin than tree $2.
# Both hosts install a release into <...>/<marketplace>/<plugin>/<version>/
# (~/.claude/plugins/cache, $CODEX_HOME/plugins/cache), so the rule reads only
# that shape and names neither host: same <plugin> dir name, both basenames
# dotted numerics (release.sh ships X.Y.Z), $1's version greater. A tree of
# any other shape — a repo checkout, a directory marketplace — carries no
# version and is never superseded by one: that is the dev pin. Versions order
# totally, so two hosts' sessions agree on the owner and never ping-pong, and
# an older caller never takes the service back (the older code would open a
# brain the newer one already migrated). Builtins only — the SessionStart path.
# The one definition: servers/daemon_launch.py release_supersedes asks this
# function for a plain-process daemon rather than keeping its own copy.
_BRAIN_LAUNCHD_VERSION_RE='^[0-9]+(\.[0-9]+)*$'
_brain_launchd_supersedes() {
    local _new="${1##*/}" _old="${2##*/}" _p1="${1%/*}" _p2="${2%/*}" _i=0 _a _b
    [ "${_p1##*/}" = "${_p2##*/}" ] || return 1
    [[ $_new =~ $_BRAIN_LAUNCHD_VERSION_RE && $_old =~ $_BRAIN_LAUNCHD_VERSION_RE ]] || return 1
    local IFS=.
    local -a _nv=($_new) _ov=($_old)
    while [ "$_i" -lt "${#_nv[@]}" ] || [ "$_i" -lt "${#_ov[@]}" ]; do
        _a=$((10#${_nv[_i]:-0})); _b=$((10#${_ov[_i]:-0}))
        [ "$_a" -gt "$_b" ] && return 0
        [ "$_a" -lt "$_b" ] && return 1
        _i=$((_i + 1))
    done
    return 1
}

# Materialize the template for THIS machine into $6 (real paths, no tokens).
# Rendering is deterministic, so the caller can diff the result against the
# installed copy to detect drift without touching launchd.
#
# Identity preservation — an already-installed plist names the tree that OWNS
# the service and the brain it serves; re-materialization converges the
# TEMPLATE (shape, env set, launchd timing) but must not silently re-point
# either:
#   PLUGIN_DIR    hooks run from the installed plugin copy while a dev
#                 machine's daemon is launchd-pinned to the repo — rendering
#                 with the caller's own tree would flip service ownership every
#                 boot (and ping-pong back on repo-side runs), and Claude Code
#                 and Codex, installed side by side, share this one service.
#                 Keep the installed tree while its launcher still exists,
#                 unless the caller's tree is a newer release of the same
#                 plugin (_brain_launchd_supersedes); a vanished tree (an
#                 uninstall, or a Codex update — it deletes the old version
#                 dir) falls back to the caller's.
#   BRAIN_DB_DIR  only the durable adoption channels (the ~/.config/brain/env
#                 knob or the userConfig brain path) may re-point an installed
#                 service's brain. An ephemeral shell BRAIN_DB_DIR (eval runs,
#                 isolated copies) must never hijack the shared service onto a
#                 temp brain.
# The installed-tree extraction is launcher-name-agnostic (anchored on
# /hooks/scripts/ — only ProgramArguments carries that path shape) so a plist
# written BEFORE a launcher rename still yields its tree; validity is then
# checked against the launcher the template ships, because that is what the
# re-materialized plist will exec.
brain_launchd_render() {
    # $1 label  $2 template  $3 launcher  $4 plugin_dir  $5 db_dir  $6 out  $7 log prefix
    local _label="$1" _template="$2" _launcher="$3"
    local _plugin_dir="$4" _db_dir="$5" _out="$6" _prefix="$7"
    local _target _installed_dir _installed_db _knob _opt
    _target="$(brain_launchd_target "$_label")"

    if [ -f "$_target" ]; then
        _installed_dir="$(sed -n 's|.*<string>\(.*\)/hooks/scripts/[^<]*</string>.*|\1|p' "$_target" | head -1)"
        if [ -n "$_installed_dir" ] && [ -x "$_installed_dir/hooks/scripts/$_launcher" ] \
           && ! _brain_launchd_supersedes "$_plugin_dir" "$_installed_dir"; then
            _plugin_dir="$_installed_dir"
        fi
        _installed_db="$(brain_plist_db_dir "$_target")"
        if [ -n "$_installed_db" ] && [ "$_installed_db" != "$_db_dir" ]; then
            _knob="$(brain_config_knob_db_dir 2>/dev/null)"
            _opt="$(brain_config_option_db_dir 2>/dev/null)"
            # The XDG service dir counts as a durable target alongside the
            # two config channels: it is the D-13 service-owned default, so
            # a resolver that landed there did so via a real brain.db — the
            # relocation flow deliberately needs no pointer, and without
            # this arm the guard would pin both plists to the vacated path
            # forever (cmp then blesses the divergence every boot).
            if [ "$_db_dir" != "$_knob" ] && [ "$_db_dir" != "$_opt" ] \
               && [ "$_db_dir" != "${XDG_DATA_HOME:-$HOME/.local/share}/brain" ]; then
                _db_dir="$_installed_db"
            fi
        fi
    fi

    # A `|` in either path makes sed exit non-zero having written a PARTIAL
    # file, and an `&` makes it succeed while re-emitting the token literally.
    # Either way the result is not an installable plist, and an unchecked
    # failure is indistinguishable from drift — the caller would boot out a
    # working service to install the broken file. So: check both, and say which
    # path shape did it, here where that knowledge belongs rather than in two
    # callers' error strings.
    if ! sed -e "s|__PLUGIN_DIR__|$_plugin_dir|g" \
             -e "s|__BRAIN_DB_DIR__|$_db_dir|g" \
             "$_template" > "$_out"; then
        echo "$_prefix FATAL: could not render $_label.plist (a '|' in a path?)" >&2
        return 1
    fi
    # `case` on a builtin read, not `grep`: one fewer fork on the SessionStart
    # path, and it dodges grep's exit-2-on-unreadable-file, which a `! grep -q`
    # would have inverted into "render fine".
    case "$(<"$_out")" in
        *__PLUGIN_DIR__*|*__BRAIN_DB_DIR__*)
            echo "$_prefix FATAL: $_label.plist still holds unsubstituted tokens (an '&' in a path?)" >&2
            return 1 ;;
    esac
    return 0
}

# Replace an ALREADY-LOADED service with the rendered plist. `kickstart` is not
# enough: launchd keeps the EnvironmentVariables it read at bootstrap time, so
# the plist must be re-bootstrapped for a template change or a brain-location
# adoption to reach the running service.
#
# Record who stopped the running daemon and on what evidence: the same
# hook_errors row the Python clients leave (daemon_client.record_forced_restart).
# daemon.log shows only the SIGTERM. The evidence is the caller's own words, so
# nothing here reads the live PID file or probes the port. Needs the brain's
# runtime; on a boot before it exists the stop goes unrecorded, and says so.
# Never fails the caller.
brain_record_forced_restart() {
    # $1 action  $2 log prefix (the caller's name)  $3 evidence
    if [ ! -x "${BRAIN_PYTHON:-}" ]; then
        echo "$2 WARN: no runtime yet — not recording: $1" >&2
        return 0
    fi
    local _root
    _root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    PYTHONPATH="$_root" "$BRAIN_PYTHON" -c '
import sys
from servers.daemon_client import record_forced_restart
record_forced_restart(sys.argv[1], evidence=sys.argv[3], caller=sys.argv[2])
' "$1" "$2 pid $$" "$3" || echo "$2 WARN: could not record: $1" >&2
    return 0
}

# Unload a service and VERIFY it actually unloaded. Bootout alone is
# fire-and-forget; a service that survives it keeps its bootstrap-time env
# and its open files. Returns 1 when the service is still loaded — the
# caller decides how loud to be. Also the door for out-of-band consumers
# (relocate-brain.sh) that must stop a service without copying this ritual.
#
# `launchctl bootout` returns at once, and the service stays loaded until its
# process exits, which the daemon may take up to its shutdown backstop to do.
# launchd SIGKILLs it at the end of its SIGTERM→SIGKILL window
# (daemon_config.LAUNCHD_SIGKILL_GRACE_S), so waiting past that window makes
# the unload certain; a test pins the relation. This is longer than the
# SessionStart budget: boot-brain.sh's call never unloads (install-daemon-
# service.sh --boot hands drift to a background run).
_BRAIN_LAUNCHD_UNLOAD_WAIT_S=25
brain_launchd_unload() {
    # $1 label  $2 log prefix
    local _label="$1" _prefix="$2" _stops_daemon="" _i=0
    _brain_launchd_refuse_entity "unload $_label" "$_prefix" && return 1
    case "$_label" in
        com.brain.daemon*) brain_launchd_managed "$_label" && _stops_daemon=1 ;;
    esac
    launchctl bootout "$_BRAIN_LAUNCHD_DOMAIN/$_label" >/dev/null 2>&1 || true
    while [ "$_i" -lt "$_BRAIN_LAUNCHD_UNLOAD_WAIT_S" ] && brain_launchd_managed "$_label"; do
        sleep 1
        _i=$((_i + 1))
    done
    if brain_launchd_managed "$_label"; then
        echo "$_prefix WARN: bootout did not unload $_label" >&2
        return 1
    fi
    [ -n "$_stops_daemon" ] && brain_record_forced_restart "launchctl bootout $_label" \
        "$_prefix" "the service was loaded; bootout unloaded it"
    return 0
}

# One install at a time per label. Every session runs install-daemon-service.sh,
# and a cold session also runs it detached after the runtime bootstrap, so two
# runs can find the same drift together — typically the first sessions after a
# plugin update. Unserialized, both boot the daemon out, and a later bootout can
# land on the service the earlier run just bootstrapped, stopping the new daemon
# seconds into its start.
#
# mkdir is the atomic test-and-set (macOS ships no flock(1)), as in
# ensure-runtime.sh. Only brain_launchd_install takes it, and callers reach
# that only to mutate, so the no-drift path every session takes stays
# lock-free. A waiter gives up after _BRAIN_LAUNCHD_LOCK_WAIT_S rather than
# spend the SessionStart budget (15s); install-daemon-service.sh's background
# run, off that budget, raises it to the stale ceiling. A lock older than
# _BRAIN_LAUNCHD_LOCK_STALE_S belongs to a run that died: a live holder's
# unload waits at most _BRAIN_LAUNCHD_UNLOAD_WAIT_S, then it records the stop
# and bootstraps. Both limits are pinned by TestLaunchdInstallLockBudget.
# Keyed by $HOME like the plist it guards, but kept out of LaunchAgents:
# bootstrapping that directory fails on any entry that is not a plist,
# dot-named ones included.
_BRAIN_LAUNCHD_LOCK_WAIT_S=10
_BRAIN_LAUNCHD_LOCK_STALE_S=120

_brain_launchd_lock_dir() {
    printf '%s/Library/Caches/%s.install.lock' "$HOME" "$1"
}

_brain_launchd_lock() {
    # $1 label  $2 log prefix — 0 held, 1 not had within the wait
    local _dir _i=0 _m
    _dir="$(_brain_launchd_lock_dir "$1")"
    mkdir -p "${_dir%/*}"
    while ! mkdir "$_dir" 2>/dev/null; do
        # /usr/bin/stat by path: a GNU stat earlier in PATH reads -f as
        # file-system mode, and its text in $(( )) would abort the script.
        _m="$(/usr/bin/stat -f %m "$_dir" 2>/dev/null)"
        case "$_m" in *[!0-9]*) _m="" ;; esac
        if [ -n "$_m" ] && [ $(( $(date +%s) - _m )) -gt "$_BRAIN_LAUNCHD_LOCK_STALE_S" ]; then
            # Rename first, so two waiters that judged the same lock stale
            # cannot both remove it and win. A waiter slow enough to rename a
            # successor's fresh lock instead is the narrow window
            # ensure-runtime.sh's lock accepts too.
            mv "$_dir" "$_dir.stale.$$" 2>/dev/null && rm -rf "$_dir.stale.$$"
            continue
        fi
        if [ "$_i" -ge "$_BRAIN_LAUNCHD_LOCK_WAIT_S" ]; then
            echo "$2 WARN: could not take the $1 install lock in ${_BRAIN_LAUNCHD_LOCK_WAIT_S}s ($_dir) — leaving launchd to the run that holds it" >&2
            return 1
        fi
        sleep 1
        _i=$((_i + 1))
    done
    echo "$$" > "$_dir/owner"
    return 0
}

# The maintenance lock — daemon_config.get_maintenance_path owns the path; a
# test pins this spelling to it. Whoever holds it has stopped the daemon to
# work on the brain's files (relocate-brain.sh copies them) and restores
# services itself when done. A bootstrap meanwhile would start a daemon on
# those files: RunAtLoad starts it, and the daemon does not check the lock.
# BRAIN_MAINTENANCE_LOCK is for sandboxed tests only, as in relocate-brain.sh.
brain_maintenance_lock() {
    printf '%s' "${BRAIN_MAINTENANCE_LOCK:-/tmp/brain-maintenance-${_BRAIN_LAUNCHD_DOMAIN#gui/}.lock}"
}
_brain_launchd_maintenance_held() {
    # $1 label  $2 log prefix — true (and loud) when the caller must not load
    local _lock
    _lock="$(brain_maintenance_lock)"
    [ -e "$_lock" ] || return 1
    echo "$2 maintenance lock $_lock is held — not loading $1; the holder restores services when it finishes" >&2
    return 0
}

# Releases only a lock this run still owns: a run slower than the stale ceiling
# may have lost it to a successor.
_brain_launchd_unlock() {
    # $1 label
    local _dir _owner=""
    _dir="$(_brain_launchd_lock_dir "$1")"
    { read -r _owner < "$_dir/owner"; } 2>/dev/null
    [ "$_owner" = "$$" ] && rm -rf "$_dir"
    return 0
}

# Install the plist for THIS machine under the label's lock: first install, or
# replace a loaded service. The render runs again under the lock because the
# caller's lock-free render may predate a concurrent install. A run that waited
# finds that install and leaves launchd alone, where its own earlier render
# (from another tree, perhaps) would replace it.
#
# Replacing unloads FIRST and VERIFIES it unloaded before touching the file — a
# still-loaded service keeps its bootstrap-time env, so overwriting the plist
# under it would leave "file current, launchd stale", where every future diff
# blesses the divergence and it never heals.
#
# Returns 1 when the render fails, the lock is not had, the maintenance lock is
# held, or the service would not unload (installed plist untouched, drift
# retries next run); the caller decides how loud to be and verifies otherwise.
brain_launchd_install() {
    # $1-$7 as brain_launchd_render: label template launcher plugin_dir db_dir out prefix
    local _label="$1" _out="$6" _prefix="$7" _rc=0
    _brain_launchd_refuse_entity "install $_label" "$_prefix" && return 1
    _brain_launchd_maintenance_held "$_label" "$_prefix" && return 1
    _brain_launchd_lock "$_label" "$_prefix" || return 1
    if ! brain_launchd_render "$@"; then
        _rc=1
    elif ! brain_launchd_managed "$_label"; then
        echo "$_prefix first run — installing launchd service $_label" >&2
        _brain_launchd_load "$_label" "$_out" "$_prefix" || _rc=1
    elif cmp -s "$_out" "$(brain_launchd_target "$_label")"; then
        echo "$_prefix $_label is already installed and loaded — nothing to do" >&2
    else
        echo "$_prefix installed plist drifted from template — re-materializing" >&2
        if brain_launchd_unload "$_label" "$_prefix"; then
            _brain_launchd_load "$_label" "$_out" "$_prefix" || _rc=1
        else
            echo "$_prefix WARN: keeping installed plist (drift retries next run)" >&2
            _rc=1
        fi
    fi
    _brain_launchd_unlock "$_label"
    return "$_rc"
}

# Install the rendered plist and hand it to launchd — the tail both paths share.
# `load -w` is the fallback for contexts where `bootstrap` is unavailable; both
# are swallowed, so callers that care must VERIFY afterwards rather than assume.
# Retried because a just-booted-out label can take a moment to free up; a first
# install does not need the patience, but one loading routine that is
# occasionally patient beats two that drift apart.
# The maintenance check repeats here, at the bootstrap itself: a relocation can
# start while this run waits for the install lock or for its unload.
_brain_launchd_load() {
    # $1 label  $2 rendered  $3 log prefix — 1 only when the maintenance lock refuses
    local _label="$1" _rendered="$2" _target _bootstrapped=""
    _brain_launchd_maintenance_held "$_label" "$3" && return 1
    _target="$(brain_launchd_target "$_label")"

    mkdir -p "$(dirname "$_target")"
    cp "$_rendered" "$_target"
    for _ in 1 2 3; do
        launchctl bootstrap "$_BRAIN_LAUNCHD_DOMAIN" "$_target" >/dev/null 2>&1 && { _bootstrapped=1; break; }
        sleep 1
    done
    [ -n "$_bootstrapped" ] || launchctl load -w "$_target" >/dev/null 2>&1 || true
}
