#!/bin/bash
# brain — ensure the dashboard singleton is running, installing its launchd
# service on FIRST use. Idempotent. Called by the /dashboard skill (and safe to
# run directly).
#
#   already up         → no-op
#   down, not installed (macOS, first run) → materialize the plist template for
#                        THIS machine + launchctl bootstrap it. launchd's
#                        RunAtLoad + KeepAlive keep it up across reboots/crashes
#                        from then on — "install once, then it just works".
#   down, installed    → launchctl kickstart it
#   non-macOS          → detached fallback (per-session, no boot persistence;
#                        systemd deferred)
#
# NOTE: no `set -u` — resolve-brain-db.sh (sourced below) is not nounset-safe.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# launchd-install.sh owns the plist ritual shared with
# install-daemon-service.sh, AND the entity predicate — why entities never get
# a managed service lives there. An entity gets no dashboard from this script;
# its port and brain are its own to serve.
source "$SCRIPT_DIR/launchd-install.sh" \
  || { echo "[ensure-dashboard] FATAL: launchd-install.sh missing or unreadable (damaged install)" >&2; exit 1; }
if brain_launchd_entity; then
  echo "[ensure-dashboard] BRAIN_INSTANCE=$BRAIN_INSTANCE — entities get no managed dashboard"
  exit 0
fi

LABEL="com.brain.dashboard"
TEMPLATE="$SCRIPT_DIR/$LABEL.plist"

# Resolve BRAIN_DB_DIR (+ DASHBOARD_PORT if the user set it) the same way every
# launch path does — so the installed plist points at the right brain.
# stdout suppressed (bootstrap noise); stderr kept — a resolution failure must
# say WHY, not surface later as a bare "BRAIN_DB_DIR unresolved".
source "$SCRIPT_DIR/resolve-brain-db.sh" >/dev/null || true
# The liveness policy below is this script's own.
PORT="${DASHBOARD_PORT:-47303}"

_up() { [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/" 2>/dev/null)" = "200" ]; }

# Detect installed-plist drift BEFORE the up fast-path: an "up" dashboard whose
# plist points at a moved brain is serving the WRONG data — exactly the state
# this check exists to fix (D-13 step 4c). Render the template deterministically
# to a temp file and compare; no drift costs one sed + cmp.
_DRIFTED=""
RENDERED=""
if [ "$(uname -s)" = "Darwin" ] && [ -f "$TEMPLATE" ] && [ -n "${BRAIN_DB_DIR:-}" ]; then
  TARGET="$(brain_launchd_target "$LABEL")"
  RENDERED="$(mktemp "${TMPDIR:-/tmp}/$LABEL.plist.XXXXXX")"
  trap 'rm -f "$RENDERED"' EXIT
  brain_launchd_render "$LABEL" "$TEMPLATE" "brain-dashboard" \
                       "$PLUGIN_DIR" "$BRAIN_DB_DIR" "$RENDERED" \
                       "[ensure-dashboard]" \
    || { echo "[ensure-dashboard] leaving the installed service untouched" >&2; exit 1; }
  if brain_launchd_managed "$LABEL" \
     && ! cmp -s "$RENDERED" "$TARGET" 2>/dev/null; then
    _DRIFTED=1
  fi
fi

if [ -z "$_DRIFTED" ] && _up; then
  echo "[ensure-dashboard] already up on :$PORT"
  exit 0
fi

case "$(uname -s)" in
  Darwin)
    if [ -n "$_DRIFTED" ] || ! brain_launchd_managed "$LABEL"; then
      # Drift or a first run. On drift, re-materialize + re-bootstrap: launchd
      # keeps the plist's EnvironmentVariables as read at bootstrap time —
      # kickstart alone would restart the dashboard with the STALE env.
      # $RENDERED is non-empty iff the drift block above ran (Darwin, template
      # present, BRAIN_DB_DIR set) and the render succeeded — it exits on
      # failure. Inside this Darwin branch the only way it is empty is a
      # missing template or an unresolved brain dir, so those two say why.
      [ -f "$TEMPLATE" ] || { echo "[ensure-dashboard] FATAL: plist template missing: $TEMPLATE" >&2; exit 1; }
      [ -n "${BRAIN_DB_DIR:-}" ] || { echo "[ensure-dashboard] FATAL: BRAIN_DB_DIR unresolved" >&2; exit 1; }
      brain_launchd_install "$LABEL" "$TEMPLATE" "brain-dashboard" \
                            "$PLUGIN_DIR" "$BRAIN_DB_DIR" "$RENDERED" \
                            "[ensure-dashboard]" || exit 1
    else
      echo "[ensure-dashboard] service loaded but down — kickstarting"
      launchctl kickstart -k "$(brain_launchd_domain)/$LABEL" >/dev/null 2>&1 || true
    fi
    ;;
  *)
    echo "[ensure-dashboard] $(uname -s): no launchd — starting detached (no boot persistence)"
    nohup "$PLUGIN_DIR/hooks/scripts/brain-dashboard" >/dev/null 2>&1 &
    ;;
esac

# Wait for it to answer.
for _ in $(seq 1 15); do
  _up && { echo "[ensure-dashboard] up on :$PORT"; exit 0; }
  sleep 1
done
echo "[ensure-dashboard] WARN: not up on :$PORT after 15s — see ${BRAIN_DB_DIR:-?}/dashboard.log" >&2
exit 1
