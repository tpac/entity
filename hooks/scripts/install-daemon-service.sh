#!/bin/bash
# brain — ensure the daemon's launchd service is INSTALLED (macOS). Idempotent.
#
# Division of labor: the daemon's LIVENESS is owned by ensure_daemon() (Python)
# + launchd KeepAlive + the ping nets. This script does the ONE thing they can't
# — provision the LaunchAgent on a fresh install, so launchd manages the daemon
# from the first boot instead of it coming up as ensure_daemon's detached
# no-launchd fallback (no KeepAlive, no boot persistence). It is a mirror of
# ensure-dashboard.sh, narrowed to install-only. See docs/DAEMON-LIFECYCLE-ARCH-PLAN.md
# Step 7.
#
# Called from boot-brain.sh BEFORE ensure_daemon(), so once installed,
# ensure_daemon sees launchd managing the daemon (manages() → True) and routes
# (re)starts through `launchctl kickstart -k` rather than direct-spawning a rival.
#
#   managed, installed plist current      → no-op (every non-fresh boot)
#   managed, installed plist DRIFTED      → re-materialize + re-bootstrap (the
#                                           installed plist is a frozen snapshot;
#                                           a template change or a brain-location
#                                           adoption must reach it — D-13 step 4c).
#                                           Under --boot, in a background run once
#                                           the session has booted (below)
#   not managed, macOS                    → materialize the plist template for THIS
#                                           machine + launchctl bootstrap it
#                                           (RunAtLoad starts it; ensure_daemon
#                                           converges on it next in boot)
#   another run installing it             → wait for it, then find its work done
#                                           (one install at a time per label)
#   maintenance lock held                 → leave launchd alone; the holder
#                                           (relocate-brain.sh) restores services
#   non-macOS                             → no-op (no launchd; ensure_daemon's
#                                           detached spawn is the only off-macOS path)
#
# NOTE: no `set -u` — resolve-brain-db.sh (sourced below) is not nounset-safe.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --boot            boot-brain.sh's call, inside the SessionStart hook's budget.
#                   Exits 3 when it hands a drift reinstall to a background run.
# --after-pid PID   the background run --boot hands a drift reinstall to
BOOT="" AFTER_PID=""
case "${1:-}" in
  --boot) BOOT=1 ;;
  --after-pid) AFTER_PID="${2:-}" ;;
esac

# launchd-install.sh owns the plist ritual (render + identity preservation +
# bootstrap/reload) shared with ensure-dashboard.sh, AND the entity predicate:
# entities never use launchd, and the reason lives with the predicate.
# Sourced first so the check costs nothing beyond the domain lookup the
# ritual needs anyway.
source "$SCRIPT_DIR/launchd-install.sh" \
  || { echo "[install-daemon-service] FATAL: launchd-install.sh missing or unreadable (damaged install)" >&2; exit 1; }
brain_launchd_entity && exit 0
# Off the SessionStart budget, a background run waits for the install lock as
# long as a live holder can keep it, not the boot's few seconds.
[ -n "$AFTER_PID" ] && _BRAIN_LAUNCHD_LOCK_WAIT_S=$_BRAIN_LAUNCHD_LOCK_STALE_S

LABEL="com.brain.daemon"
TEMPLATE="$SCRIPT_DIR/$LABEL.plist"

# No launchd off macOS — ensure_daemon's detached spawn owns that path.
[ "$(uname -s)" = "Darwin" ] || exit 0

# The session that handed this reinstall over boots against the daemon it
# found; the reinstall restarts that daemon, so let the boot finish first.
# Capped, because a finished process's PID can be reused.
if [ -n "$AFTER_PID" ]; then
  _waited=0
  while [ "$_waited" -lt 60 ] && kill -0 "$AFTER_PID" 2>/dev/null; do
    sleep 1
    _waited=$((_waited + 1))
  done
fi

# Resolve BRAIN_DB_DIR the same way every launch path does, so the installed
# plist points at the right brain. boot-brain.sh already resolved + exported it
# before calling us; this re-source fast-paths (runtime sentinel is set by now,
# so no ensure-runtime recursion) and just confirms the value.
# stdout suppressed (bootstrap noise); stderr KEPT — its sibling
# ensure-dashboard.sh documents why: a resolution failure must say WHY, not
# surface below as a bare "BRAIN_DB_DIR unresolved". This runs detached on the
# cold-install chain, where .bootstrap.log is the only channel there is.
source "$SCRIPT_DIR/resolve-brain-db.sh" >/dev/null || true
# The policy below — install-only, then verify — is this script's own.
TARGET="$(brain_launchd_target "$LABEL")"

[ -f "$TEMPLATE" ] || { echo "[install-daemon-service] FATAL: plist template missing: $TEMPLATE" >&2; exit 1; }
[ -n "${BRAIN_DB_DIR:-}" ] || { echo "[install-daemon-service] FATAL: BRAIN_DB_DIR unresolved — not installing" >&2; exit 1; }

# Render to a temp file FIRST — rendering is deterministic, so comparing it
# against the installed copy detects drift (template evolved, or the brain
# legitimately moved) without touching launchd on the no-drift common case.
# LAUNCHER is what the CURRENT template's ProgramArguments points at.
RENDERED="$(mktemp "${TMPDIR:-/tmp}/$LABEL.plist.XXXXXX")"
trap 'rm -f "$RENDERED"' EXIT
brain_launchd_render "$LABEL" "$TEMPLATE" "brain-daemon" \
                     "$PLUGIN_DIR" "$BRAIN_DB_DIR" "$RENDERED" \
                     "[install-daemon-service]" \
  || { echo "[install-daemon-service] leaving the installed service untouched" >&2; exit 1; }

# Every non-fresh boot ends here, without the install lock. The file is read
# before the load state: a concurrent install copies the plist only after its
# bootout, so in this order its half-done state never reads as current.
if cmp -s "$RENDERED" "$TARGET" 2>/dev/null && brain_launchd_managed "$LABEL"; then
  exit 0
fi
# Drift on a loaded service, inside the SessionStart hook. The reinstall waits
# for the running daemon to exit, which can outlast the hook's budget, and a
# hook killed between bootout and bootstrap would leave the service down. A
# background run does it once this session has booted, and exit 3 tells
# boot-brain.sh to boot on the running daemon: an ensure_daemon restart now
# would relaunch it on the plist not yet replaced (onto the old brain, after a
# move). A first install stays here: nothing is running to stop.
if [ -n "$BOOT" ] && brain_launchd_managed "$LABEL"; then
  _brain_launchd_maintenance_held "$LABEL" "[install-daemon-service]" && exit 1
  ( nohup bash "$SCRIPT_DIR/install-daemon-service.sh" --after-pid "$PPID" \
      >> "$PLUGIN_DIR/.bootstrap.log" 2>&1 & )
  echo "[install-daemon-service] installed plist drifted — reinstalling in the background once this session has booted (log: $PLUGIN_DIR/.bootstrap.log)" >&2
  exit 3
fi
# Drift or a first run. The installed plist is a frozen snapshot launchd read at
# bootstrap time, so a template change or a brain-location adoption needs a
# re-bootstrap. brain_launchd_install decides again under the label's lock,
# because a concurrent run may already have done this.
brain_launchd_install "$LABEL" "$TEMPLATE" "brain-daemon" \
                      "$PLUGIN_DIR" "$BRAIN_DB_DIR" "$RENDERED" \
                      "[install-daemon-service]" || exit 1
# VERIFY instead of assuming: bootstrap/load failures were swallowed above (the
# fallback chain needs that), so re-probe launchd for the truth. A false
# "installed" here would mask exactly the state this script exists to prevent —
# the daemon silently running detached (no KeepAlive) forever.
if brain_launchd_managed "$LABEL"; then
  echo "[install-daemon-service] installed $TARGET (launchd now owns the daemon)" >&2
else
  echo "[install-daemon-service] WARN: wrote $TARGET but launchctl bootstrap failed —" >&2
  echo "[install-daemon-service] daemon will run DETACHED via ensure_daemon (no KeepAlive," >&2
  echo "[install-daemon-service] no boot persistence). Retry: launchctl bootstrap $(brain_launchd_domain) $TARGET" >&2
  exit 1
fi
