#!/bin/bash
# Move an old `brain` plugin install (brain@<marketplace>, 9.x) onto
# `entity@anchor` and keep the brain. MIGRATING.md is the page a person reads;
# this file is the procedure.
#
#   bash migrate-from-brain.sh            1. inventory — changes nothing
#   bash migrate-from-brain.sh --apply    2. stop, back up, swap the plugins
#       then open ONE Claude Code session, let it finish booting, quit it
#   bash migrate-from-brain.sh --finish   3. move the brain home, verify, print the way back
#
#   --brain DIR    name the live brain when the inventory finds more than one
#   ENTITY_SOURCE  the marketplace --apply adds (default: tpac/entity)
#
# --apply and --finish refuse while any Claude Code or Codex session, the
# Claude desktop app or the ChatGPT app (it runs Codex) is open: they stop the
# brain's services and move files a running session would respawn or read.
# Every step checks before it acts and says what state it leaves behind if it
# stops. Nothing is ever deleted.
#
# Hazards, each verified against Claude Code 2.1.281 and the old trees:
#  1. `claude plugin marketplace remove` uninstalls the plugins still installed
#     from that marketplace and DELETES their data dirs, which is where 9.6.0
#     parks the brain ($CLAUDE_PLUGIN_DATA/brain). So every install is
#     uninstalled with --keep-data first, brain.db is asserted, and only an
#     emptied marketplace is removed.
#  2. A backup beside the brain dies with the folder an uninstall deletes, so
#     the backup (SQLite online backup, quick_check, node count N) goes to
#     ~/brain-migration-<timestamp>/.
#  3. Old and new share the launchd label com.brain.daemon and the daemon port
#     (47200 + uid % 100), an uninstall leaves the old code in the plugin
#     cache, and the service installer keeps an installed plist on a tree whose
#     launcher still exists. So the old daemon and dashboard are stopped, their
#     plists move into the backup, and --finish checks the running daemon.
#  4. An old daemon relaunched on its own resolves its brain itself: a plugin
#     data dir, ~/AgentsContext/brain, or a fresh one it seeded. So every
#     brain.db is inventoried, the one the running daemon holds open is named,
#     each one is backed up, and --apply stops when the live one is ambiguous.
#  5. relocate-brain.sh only exists from 9.7.2, so --finish runs the NEW
#     plugin's copy.
#  6. Install ids vary (brain@brain, brain@<upload marketplace>, project and
#     local scopes): every brain@* in installed_plugins.json is handled.
#
# The way back is printed at the end of --apply and --finish: a verified copy
# of the brain as it was sits in the backup, and the original stays where it
# was (renamed beside it after a relocation), upgraded by entity's first
# start. The old plugin itself cannot be reinstalled; its catalog was renamed.
#
# REHEARSAL (maintainer, from the dev repo). The install smoke harness cannot
# run old trees on a working account: they ignore BRAIN_INSTANCE, and 9.6.0
# ignores BRAIN_DAEMON_PORT, so their boot hook takes over that account's
# daemon. Rehearse as a separate macOS user (its own uid, so its own port,
# launchd domain and /tmp files), kept logged in (launchd's gui domain only
# exists then), driven over ssh, with its own copy of the claude binary.
#   1. scripts/build-migration-catalogs.sh OUT; then --serve OUT PORT. Claude
#      Code rejects file:// and git:// marketplaces and clones shallow, so the
#      catalogs are served as smart-HTTP git: CAT=http://127.0.0.1:PORT.
#   2. As the rehearsal user, per case (main = the old catalog's branch,
#      moved with `git -C OUT/brain.git update-ref refs/heads/main <tag>`):
#        parked 9.6.0    main=v9.6.0; `claude plugin marketplace add
#                        $CAT/brain.git`; `claude plugin install brain@brain`;
#                        `claude -p hi` until ~/.claude/plugins/data/brain-brain/brain exists
#        parked 9.7.2,   the same, then main=v9.7.2, `marketplace update brain`,
#        dark, daemon    `plugin update brain@brain`, `claude -p hi`; then
#        still running   main=renamed and `marketplace update brain`
#        XDG 9.7.x       main=v9.7.2 from the start (born at ~/.local/share/brain)
#        legacy 9.7.2    the same, then with every service down the brain moved
#                        to ~/AgentsContext/brain for 9.7.2 to adopt
#      then this script, dry run and --apply with ENTITY_SOURCE=$CAT/entity.git;
#      `claude -p hi` until the first-run install finishes; --finish.
#   Sessions need no login: SessionStart hooks run before the auth check. The
#   dashboard port is machine-wide, so the rehearsal user never gets its own.
#   Reset between cases: that user's ~/.claude, ~/.config/brain,
#   ~/.local/share/brain, ~/brain-migration-*, LaunchAgents/com.brain.*.
set -u
set -o pipefail

MODE=inventory
BRAIN_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply)  MODE=apply ;;
    --finish) MODE=finish ;;
    --brain)  BRAIN_ARG="${2:?--brain needs a directory}"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# ── where things are: the paths every old and new version uses ──────────────
UID_N="$(id -u)"
CC_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
PLUGINS="$CC_DIR/plugins"
REGISTRY="$PLUGINS/installed_plugins.json"
MARKETS="$PLUGINS/known_marketplaces.json"
CONF="${XDG_CONFIG_HOME:-$HOME/.config}/brain"
ENVF="$CONF/env"
RESOLVED="$CONF/resolved.env"
XDG_BRAIN="${XDG_DATA_HOME:-$HOME/.local/share}/brain"
LEGACY="$HOME/AgentsContext/brain"
LA="$HOME/Library/LaunchAgents"
LABELS="com.brain.daemon com.brain.dashboard"
# The daemon's rendezvous, the same in every version old and new. Owners:
# daemon_config._resolve_daemon_port / get_pid_path / get_maintenance_path and
# daemon_launch.LAUNCHD_LABEL; this script cannot import them (it runs before
# entity exists), so tests/test_migrate_from_brain.py pins the copies to them.
PORT="${BRAIN_DAEMON_PORT:-$( [ -f "$ENVF" ] && (BRAIN_DAEMON_PORT=''; . "$ENVF" >/dev/null 2>&1; printf '%s' "$BRAIN_DAEMON_PORT"))}"
case "$PORT" in ''|*[!0-9]*) PORT=$((47200 + UID_N % 100)) ;; esac   # the owner falls back the same way
# Overridable for sandboxed tests only, as in relocate-brain.sh.
LOCK="${BRAIN_MAINTENANCE_LOCK:-/tmp/brain-maintenance-$UID_N.lock}"
PIDFILE="${BRAIN_DAEMON_PIDFILE:-/tmp/brain-daemon-$UID_N.pid}"
ENTITY_SOURCE="${ENTITY_SOURCE:-tpac/entity}"
IS_MAC=0; [ "$(uname -s)" = Darwin ] && IS_MAC=1
# Field separator for install records. Not a tab: tab is IFS whitespace, so
# `read` collapses an empty projectPath and shifts every later field.
SEP=$'\x1f'

say()     { printf '%s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }
item()    { printf '   %s\n' "$*"; }
STOP_NOTE=""   # the state a stop at the current step leaves behind
stop() {
  printf '\nSTOPPED: %s\n' "$*"
  [ -z "$STOP_NOTE" ] || printf '%s\n' "$STOP_NOTE"
  exit 1
}

[ -z "${BRAIN_INSTANCE:-}" ] \
  || stop "BRAIN_INSTANCE=$BRAIN_INSTANCE is set: run this from a plain terminal, not an entity's shell"
command -v python3 >/dev/null 2>&1 || stop "python3 not found (the old plugin needed it too)"

# JSON, plists and SQLite, one small program; each command documents itself.
read -r -d '' PYLIB <<'PY' || true
import json, os, pathlib, re, sqlite3, sys, time
cmd, a = sys.argv[1], sys.argv[2:]

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

if cmd == 'home':          # this account's home, whatever HOME says
    import pwd
    print(pwd.getpwuid(os.getuid()).pw_dir)
elif cmd == 'installs':    # id, scope, projectPath, installPath, name its tree declares
    for pid, entries in sorted((load(a[0]).get('plugins') or {}).items()):
        for e in entries:
            ip = e.get('installPath') or ''
            name = load(os.path.join(ip, '.claude-plugin', 'plugin.json')).get('name', '') if ip else ''
            print('\x1f'.join([pid, e.get('scope') or 'user', e.get('projectPath') or '', ip, name]))
elif cmd == 'marketplace': # where a marketplace was added from; empty when absent
    s = (load(a[0]).get(a[1]) or {}).get('source') or {}
    print(s.get('repo') or s.get('url') or s.get('path') or '')
elif cmd == 'plist':       # a LaunchAgent's program, then its baked BRAIN_DB_DIR
    import plistlib
    try:
        with open(a[0], 'rb') as f:
            p = plistlib.load(f)
    except Exception:
        p = {}
    print((p.get('ProgramArguments') or [''])[0])
    print((p.get('EnvironmentVariables') or {}).get('BRAIN_DB_DIR', ''))
elif cmd == 'stat':        # bytes of the brain's databases, and when they last changed
    files = [os.path.join(a[0], n) for n in
             ('brain.db', 'brain.db-wal', 'brain_logs.db', 'brain_logs.db-wal')]
    files = [f for f in files if os.path.exists(f)]
    print('%d\t%s' % (sum(os.path.getsize(f) for f in files),
                      time.strftime('%Y-%m-%d %H:%M', time.localtime(
                          max(os.path.getmtime(f) for f in files)))))
elif cmd == 'backup':      # SQLite online backup SRC -> DST; prints its check and node rows
    # With the writers stopped the copy must also hold every table's rows
    # exactly; a live source (the after/ snapshot) is only quick_checked.
    src, dst, writers = a
    try:
        s = sqlite3.connect(pathlib.Path(src).as_uri() + '?mode=ro', uri=True)
        d = sqlite3.connect(dst)
        s.backup(d)
    except sqlite3.OperationalError:
        # A writer killed mid-WAL leaves a recovery a read-only connection
        # cannot run. With every writer stopped, an ordinary connection runs
        # it, exactly as the next daemon start would.
        if writers != 'stopped':
            raise
        if os.path.exists(dst):
            os.remove(dst)
        s = sqlite3.connect(src)
        d = sqlite3.connect(dst)
        s.backup(d)
    d.execute('PRAGMA journal_mode=DELETE')
    check = d.execute('PRAGMA quick_check').fetchone()[0]
    if check == 'ok' and writers == 'stopped':
        for (t,) in d.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall():
            n_src = s.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0]
            n_dst = d.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0]
            if n_src != n_dst:
                check = 'table %s holds %d rows in the source but %d in the copy' % (t, n_src, n_dst)
                break
    s.close()
    has = d.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='nodes'").fetchone()
    rows = d.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] if has else '-'
    d.close()
    print('%s\t%s' % (check, rows))
elif cmd == 'learned':     # nodes a brain made after its birth (every version marks its
    # starting pack anchor:seed); -1 when the schema carries no encoding_source
    c = sqlite3.connect(pathlib.Path(a[0]).as_uri() + '?mode=ro', uri=True)
    if 'encoding_source' not in [r[1] for r in c.execute('PRAGMA table_info(nodes)')]:
        print(-1)
    else:
        print(c.execute("SELECT COUNT(*) FROM nodes WHERE COALESCE(encoding_source, '') != 'anchor:seed'").fetchone()[0])
elif cmd == 'knob':        # make the env file's BRAIN_DB_DIR name DIR; older lines commented
    path, value = a
    try:
        with open(path) as f:
            lines = f.read().splitlines(True)
    except FileNotFoundError:
        lines = []
    out = ['# replaced by migrate-from-brain.sh: ' + ln
           if re.match(r'\s*(export\s+)?BRAIN_DB_DIR=', ln) else ln for ln in lines]
    if out and not out[-1].endswith('\n'):
        out[-1] += '\n'
    out.append("BRAIN_DB_DIR='%s'\n" % value.replace("'", "'\\''"))
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + '.tmp-migrate'
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
        f.writelines(out)
    os.replace(tmp, path)
PY
_py() { python3 -c "$PYLIB" "$@"; }

# launchd domains and the daemon's /tmp files belong to the ACCOUNT, not to
# HOME: under a borrowed HOME every mutation here would hit the account's real
# services. Inspecting is harmless; changing is refused.
REAL_HOME="$(_py home)"
SANDBOXED=0; [ "$(cd "$HOME" 2>/dev/null && pwd -P)" = "$(cd "$REAL_HOME" 2>/dev/null && pwd -P)" ] || SANDBOXED=1

# The CLI, or the copy the desktop app bundles when none is on PATH: under
# claude-code/<version>/, directly or one folder deeper (<version>/<id>/).
CLAUDE="$(command -v claude 2>/dev/null || true)"
if [ -z "$CLAUDE" ]; then
  _desk="$HOME/Library/Application Support/Claude/claude-code"
  CLAUDE="$(ls -d "$_desk/"*/claude.app/Contents/MacOS/claude "$_desk/"*/*/claude.app/Contents/MacOS/claude 2>/dev/null | sort -V | tail -1)"
fi

_in()     { local x="$1" e; shift; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }
_n()      { [ -n "$1" ] || return 0; if [ -d "$1" ]; then (cd "$1" && pwd -P); else printf '%s\n' "${1%/}"; fi; }
_mb()     { awk -v b="$1" 'BEGIN { printf "%.0f MB", b / 1048576 }'; }
_slug()   { local s="${1#https://github.com/}"; printf '%s' "${s%.git}"; }
_sq()     { printf %s "$1" | sed "s/'/'\\\\''/g"; }   # a path inside '…' in a command the user pastes
_installs() { _py installs "$REGISTRY"; }
# A shell-grammar config file, read the way the resolver reads it: sourced in
# a subshell with its own output discarded.
_envkey() { [ -f "$1" ] || return 0; (BRAIN_DB_DIR=''; . "$1" >/dev/null 2>&1; printf '%s' "$BRAIN_DB_DIR"); }
_class() {
  case "$1" in
    "$(_n "$PLUGINS")/data/"*) echo parked ;;         # an uninstall without --keep-data deletes it
    "$(_n "$PLUGINS")/"*)      echo inside-plugin ;;  # any uninstall deletes it
    "$(_n "$XDG_BRAIN")")      echo standard ;;
    "$(_n "$LEGACY")")         echo legacy ;;
    *)                         echo custom ;;
  esac
}
# The classes --finish moves to the standard location: a parked brain, which an
# uninstall can delete, and a legacy one, so every brain ends in one place (D-13).
# A custom location is the user's choice and stays.
_moves() { case "$1" in parked|legacy) return 0 ;; esac; return 1; }
_loaded() { [ "$IS_MAC" = 1 ] && [ "$SANDBOXED" = 0 ] && launchctl print "gui/$UID_N/$1" >/dev/null 2>&1; }
_listener_pid() {   # the process serving the daemon port, when it is this account's
  local p
  p="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)"
  [ -n "$p" ] && [ "$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ')" = "$UID_N" ] && printf '%s\n' "$p"
}
_pid_brain() { lsof -nP -p "$1" -Fn 2>/dev/null | sed -n 's|^n\(/.*\)/brain\.db$|\1|p' | head -1; }
_sessions_running() {  # hosts that can start a brain daemon: Claude Code (CLI, the
  # desktop app's bundled copy, npm) and Codex (CLI, the ChatGPT app's bundled copy, npm)
  ps -axo uid=,comm= | awk -v u="$UID_N" '$1 == u {
    sub(/^ *[0-9]+ +/, ""); n = $0; sub(/.*\//, "", n); n = tolower(n)
    if (n == "claude" || n == "codex") print "   " $0 }'
  # an npm install runs as node, with the package path in its arguments
  ps -axo uid=,command= | awk -v u="$UID_N" '$1 == u && /@anthropic-ai\/claude-code\/|@openai\/codex\// {
    sub(/^ *[0-9]+ +/, ""); print "   " substr($0, 1, 120) }'
}

# ── inventory ────────────────────────────────────────────────────────────────
collect_installs() {
  UNINST=() OLD_MKTS=() OLD_TREES=() MKT_REMOVE=() MKT_KEEP=() LEFT_ALONE=()
  ALL_INSTALLS="$(_installs)"
  local id scope proj ipath tname m other
  while IFS="$SEP" read -r id scope proj ipath tname; do
    case "$id" in brain@*) ;; *) continue ;; esac
    # a dev install that kept the old id but carries the new tree is not ours
    if [ -n "$tname" ] && [ "$tname" != brain ]; then
      LEFT_ALONE+=("$id ($scope): its tree declares '$tname', not the old plugin")
      continue
    fi
    UNINST+=("$id$SEP$scope$SEP$proj")
    [ -z "$ipath" ] || OLD_TREES+=("$ipath")
    m="${id#brain@}"
    _in "$m" ${OLD_MKTS[@]+"${OLD_MKTS[@]}"} || OLD_MKTS+=("$m")
  done <<< "$ALL_INSTALLS"
  # entity@brain, the renamed catalog's offer under the old marketplace, is
  # the old catalog's even when brain@brain was uninstalled by hand
  if printf '%s\n' "$ALL_INSTALLS" | awk -F"$SEP" '$1 == "entity@brain" { f = 1 } END { exit !f }'; then
    _in brain ${OLD_MKTS[@]+"${OLD_MKTS[@]}"} || OLD_MKTS+=(brain)
  fi
  # entity@<old marketplace> is what the renamed catalog offers a refreshed
  # install; it goes too. Anything else keeps its marketplace, because
  # removing a marketplace deletes the data of what is still installed from it.
  for m in ${OLD_MKTS[@]+"${OLD_MKTS[@]}"}; do
    other=0
    while IFS="$SEP" read -r id scope proj ipath tname; do
      [ "${id#*@}" = "$m" ] || continue
      case "${id%@*}" in
        brain)  _in "$id$SEP$scope$SEP$proj" ${UNINST[@]+"${UNINST[@]}"} || other=1 ;;
        entity) [ "$m" != anchor ] && UNINST+=("$id$SEP$scope$SEP$proj") ;;
        *)      other=1 ;;
      esac
    done <<< "$ALL_INSTALLS"
    if [ "$other" = 0 ]; then MKT_REMOVE+=("$m"); else MKT_KEEP+=("$m"); fi
  done
  ENTITY_ROOT="$(printf '%s\n' "$ALL_INSTALLS" | awk -F"$SEP" '$1 == "entity@anchor" && $2 == "user" { print $4; exit }')"
  ANCHOR_SRC="$(_py marketplace "$MARKETS" anchor)"
}

collect_brains() {
  DAEMON_PID="$(_listener_pid)" DAEMON_CMD="" DAEMON_DB=""
  if [ -n "$DAEMON_PID" ]; then
    DAEMON_CMD="$(ps -o command= -p "$DAEMON_PID" 2>/dev/null)"
    DAEMON_DB="$(_pid_brain "$DAEMON_PID")"
  fi
  PLIST_PROG="" PLIST_DB=""
  [ -f "$LA/com.brain.daemon.plist" ] \
    && { IFS= read -r PLIST_PROG; IFS= read -r PLIST_DB; } < <(_py plist "$LA/com.brain.daemon.plist")
  RESOLVED_DB="$(_envkey "$RESOLVED")"
  KNOB_DB="$(_envkey "$ENVF")"
  case "$KNOB_DB" in "~/"*) KNOB_DB="$HOME/${KNOB_DB#"~/"}" ;; esac
  BRAIN_DIRS=()
  local d f
  while IFS= read -r d; do [ -n "$d" ] && BRAIN_DIRS+=("$d"); done < <(
    {
      for d in "$DAEMON_DB" "$PLIST_DB" "$RESOLVED_DB" "$KNOB_DB" "$XDG_BRAIN" "$LEGACY" "$BRAIN_ARG"; do
        [ -n "$d" ] && [ -f "$d/brain.db" ] && _n "$d"
      done
      [ -d "$PLUGINS" ] && find "$PLUGINS" -maxdepth 7 -type f -name brain.db 2>/dev/null \
        | while IFS= read -r f; do _n "$(dirname "$f")"; done
    } | sort -u)
}

_evidence() {   # what says a brain was in use here: the old install, services, records
  local e id scope proj
  for e in ${UNINST[@]+"${UNINST[@]}"}; do
    IFS="$SEP" read -r id scope proj <<< "$e"
    printf 'the old plugin is installed: %s (%s)\n' "$id" "$scope"
  done
  [ -z "$(_py marketplace "$MARKETS" brain)" ] || printf 'the old marketplace "brain" is still declared\n'
  [ -z "$DAEMON_PID" ] || printf 'a brain service is running (pid %s)%s\n' "$DAEMON_PID" "${DAEMON_DB:+, holding $DAEMON_DB/brain.db}"
  [ -z "$PLIST_PROG" ] || printf 'the brain service definition runs %s%s\n' "$PLIST_PROG" "${PLIST_DB:+ for $PLIST_DB}"
  [ -z "$RESOLVED_DB" ] || printf '%s names %s\n' "$RESOLVED" "$RESOLVED_DB"
  [ -z "$KNOB_DB" ] || printf '%s names %s\n' "$ENVF" "$KNOB_DB"
  return 0
}

_knob_needed() {   # the env knob must be (re)written to name the chosen brain
  # The standard location needs no knob, unless a knob names another brain:
  # entity reads the knob first, so it would follow that one instead.
  [ -n "$PRIMARY" ] && [ "$(_n "$KNOB_DB")" != "$PRIMARY" ] \
    && { [ "$PRIMARY" != "$(_n "$XDG_BRAIN")" ] || [ -n "$KNOB_DB" ]; }
}

_tags() {   # which records name this brain
  local d="$1" t=""
  [ -n "$DAEMON_DB" ]   && [ "$d" = "$(_n "$DAEMON_DB")" ]   && t="$t running-daemon"
  [ -n "$PLIST_DB" ]    && [ "$d" = "$(_n "$PLIST_DB")" ]    && t="$t service"
  [ -n "$RESOLVED_DB" ] && [ "$d" = "$(_n "$RESOLVED_DB")" ] && t="$t resolved.env"
  [ -n "$KNOB_DB" ]     && [ "$d" = "$(_n "$KNOB_DB")" ]     && t="$t env-knob"
  printf '%s' "${t# }"
}

choose_primary() {   # the live brain: named by --brain, or by every record that names one
  PRIMARY="" AMBIG=""
  if [ -n "$BRAIN_ARG" ]; then
    [ -f "$BRAIN_ARG/brain.db" ] || stop "--brain $BRAIN_ARG holds no brain.db"
    PRIMARY="$(_n "$BRAIN_ARG")"
    return
  fi
  local named="" d
  for d in "$DAEMON_DB" "$PLIST_DB" "$RESOLVED_DB" "$KNOB_DB"; do
    [ -n "$d" ] && [ -f "$d/brain.db" ] && named="$named$(_n "$d")"$'\n'
  done
  named="$(printf '%s' "$named" | awk 'NF' | sort -u)"
  case "$(printf '%s' "$named" | awk 'NF' | wc -l | tr -d ' ')" in
    1) PRIMARY="$named" ;;
    0) if [ "${#BRAIN_DIRS[@]}" -eq 1 ]; then PRIMARY="${BRAIN_DIRS[0]}"
       elif [ "${#BRAIN_DIRS[@]}" -gt 1 ]; then AMBIG="no record names a brain and ${#BRAIN_DIRS[@]} exist"
       fi ;;
    *) AMBIG="the running daemon, its service, resolved.env and the env knob name different brains" ;;
  esac
}

inventory() {
  REFUSE=""
  _refuse() { REFUSE="$REFUSE   - $*"$'\n'; }

  section "Claude Code"
  { [ -n "$CLAUDE" ] && [ -x "$CLAUDE" ]; } || stop "no claude command: not on PATH, and no desktop-app copy"
  item "$CLAUDE ($("$CLAUDE" --version 2>/dev/null | head -1))"
  "$CLAUDE" plugin uninstall --help 2>/dev/null | grep -q -- '--keep-data' \
    || stop "this Claude Code has no 'plugin uninstall --keep-data': update Claude Code first, because without it an uninstall deletes a parked brain"
  item "plugin uninstall --keep-data: supported"

  collect_installs
  collect_brains
  choose_primary

  section "Old plugin installs"
  local e id scope proj m src
  if [ "${#UNINST[@]}" -eq 0 ]; then item "none"; fi
  for e in ${UNINST[@]+"${UNINST[@]}"}; do
    IFS="$SEP" read -r id scope proj <<< "$e"
    item "$id  scope=$scope${proj:+  project=$proj (its .claude/settings loses the entry)}"
  done
  for e in ${LEFT_ALONE[@]+"${LEFT_ALONE[@]}"}; do item "left alone: $e"; done
  for m in ${MKT_REMOVE[@]+"${MKT_REMOVE[@]}"}; do
    item "marketplace $m ($(_py marketplace "$MARKETS" "$m")): removed once its plugins are uninstalled"
  done
  for m in ${MKT_KEEP[@]+"${MKT_KEEP[@]}"}; do
    item "marketplace $m: kept, other plugins are installed from it"
  done

  section "New plugin"
  if [ -n "$ANCHOR_SRC" ]; then
    item "marketplace anchor: $ANCHOR_SRC"
    [ "$(_slug "$ANCHOR_SRC")" = "$(_slug "$ENTITY_SOURCE")" ] \
      || _refuse "a marketplace named anchor already comes from $ANCHOR_SRC, not $ENTITY_SOURCE"
  else
    item "marketplace anchor: not added yet (--apply adds $ENTITY_SOURCE)"
    src="$ENTITY_SOURCE"; case "$src" in *://*) ;; *) src="https://github.com/$src.git" ;; esac
    GIT_TERMINAL_PROMPT=0 git ls-remote "$src" HEAD >/dev/null 2>&1 \
      || _refuse "$src is not reachable, and --apply would stop between removing the old plugin and installing the new one"
  fi
  item "entity@anchor: ${ENTITY_ROOT:-not installed}"

  section "Brain services"
  if [ -n "$DAEMON_PID" ]; then
    item "daemon on port $PORT: pid $DAEMON_PID  $DAEMON_CMD"
  else
    item "daemon on port $PORT: not running"
  fi
  if [ "$SANDBOXED" = 1 ]; then
    item "launchd not inspected: HOME is not this account's home ($REAL_HOME)"
  elif [ "$IS_MAC" = 1 ]; then
    local l
    for l in $LABELS; do
      if _loaded "$l"; then item "$l: loaded"; else item "$l: not loaded"; fi
    done
    [ -z "$PLIST_PROG" ] || item "com.brain.daemon plist runs $PLIST_PROG"
  fi

  section "Brains found"
  local d cls st mark bytes=0 evidence learned
  evidence="$(_evidence)"
  if [ "${#BRAIN_DIRS[@]}" -eq 0 ]; then
    if [ -n "$evidence" ]; then item "none, although a brain was in use here (see Result)"
    else item "none (nothing to preserve; the new plugin starts a fresh brain)"; fi
  fi
  # The daemon's brain.db gone from disk while the daemon still holds it open:
  # the only copy may be that process's open files (macOS gives no way to read
  # them from outside it), and a restart ends it.
  if [ -n "$DAEMON_DB" ] && [ ! -f "$DAEMON_DB/brain.db" ]; then
    item "the running daemon (pid $DAEMON_PID) holds $DAEMON_DB/brain.db, which is no longer on disk"
    _refuse "the brain service still running (pid $DAEMON_PID) holds a brain whose folder is gone ($DAEMON_DB): your memories may exist only inside that process. Do not restart, log out, quit it or install anything; send this whole output to whoever pointed you here"
  fi
  for d in ${BRAIN_DIRS[@]+"${BRAIN_DIRS[@]}"}; do
    cls="$(_class "$d")"; st="$(_py stat "$d")"; bytes=$((bytes + ${st%%$'\t'*}))
    mark=" "; [ "$d" = "$PRIMARY" ] && mark="*"
    learned="$(_py learned "$d/brain.db" 2>/dev/null)"
    printf ' %s %s\n     %s, %s, last written %s%s%s\n' "$mark" "$d" "$cls" "$(_mb "${st%%$'\t'*}")" \
      "${st#*$'\t'}" "$( [ -n "$learned" ] && [ "$learned" != -1 ] && printf ', %s nodes learned beyond its starting pack' "$learned")" \
      "$( t="$(_tags "$d")"; [ -z "$t" ] || printf ', named by: %s' "$t")"
  done
  if [ -n "$PRIMARY" ]; then
    item "live brain (*): $PRIMARY"
    case "$(_class "$PRIMARY")" in
      inside-plugin) _refuse "the live brain sits inside a plugin's code folder, which any uninstall deletes: ask the maintainer" ;;
    esac
    if [ -f "$XDG_BRAIN/brain.db" ] && [ "$(_n "$XDG_BRAIN")" != "$PRIMARY" ]; then
      _refuse "a second brain sits at the standard location $XDG_BRAIN: decide which is current, move the other aside, rerun"
    fi
    # Whoever used the old plugin has a brain that learned something; one
    # holding only its starting pack is a fresh birth — possibly seeded over
    # a lost brain by a plugin that could not find it. Naming it confirms it.
    if [ -z "$BRAIN_ARG" ] && [ "$(_py learned "$PRIMARY/brain.db" 2>/dev/null)" = 0 ]; then
      _refuse "the brain found ($PRIMARY) holds only its starting nodes: it looks newly created, not like a brain that was used. If yours is elsewhere, find it first. If this is the one, rerun with --brain '$(_sq "$PRIMARY")'"
    fi
  elif [ -n "$AMBIG" ]; then
    _refuse "which brain is live is ambiguous ($AMBIG): rerun with --brain <dir>"
  fi

  section "Also"
  local key
  # [[ ]], not case: bash 3.2 ends a $( ) at a case pattern's `)`
  key="$( [ -f "$ENVF" ] && (ANTHROPIC_API_KEY=''; . "$ENVF" >/dev/null 2>&1; [[ "$ANTHROPIC_API_KEY" == sk-* ]] && echo yes))"
  if [ "$key" = yes ]; then item "API key in $ENVF: yes"
  else item "API key in $ENVF: no. After the move, paste it in the entity plugin's settings or the dashboard's /setup page."
  fi
  local avail_kb
  avail_kb="$(df -Pk "$HOME" | awk 'NR == 2 { print $4 }')"
  # the backups, plus for a brain --finish moves relocate-brain.sh's staging
  # copy of its whole folder, plus the after/ snapshot of brain.db
  local need_kb=$((bytes / 1024))
  if [ -n "$PRIMARY" ]; then
    _moves "$(_class "$PRIMARY")" && need_kb=$((need_kb + $(du -sk "$PRIMARY" | awk '{ print $1 }')))
    need_kb=$((need_kb + $(du -sk "$PRIMARY/brain.db" | awk '{ print $1 }')))
  fi
  need_kb=$((need_kb * 12 / 10))
  item "free space: $(_mb $((avail_kb * 1024))) (the migration needs about $(_mb $((need_kb * 1024))))"
  [ "$need_kb" -lt "$avail_kb" ] || _refuse "not enough free space in $HOME for the backup and the move"
  local tree
  for tree in ${OLD_TREES[@]+"${OLD_TREES[@]}"}; do
    grep -qF "$tree/" "$CC_DIR/settings.json" 2>/dev/null \
      && item "$CC_DIR/settings.json runs something from the old plugin's folder (a statusLine?): point it at the entity install after --finish"
  done
  [ ! -e "$LOCK" ] || _refuse "the maintenance lock $LOCK exists: another brain operation is running, or one crashed (remove the file if none is running)"
  [ "$SANDBOXED" = 0 ] || _refuse "HOME is not this account's home: launchd and the daemon's files belong to the account, so changes would hit its real services"
  # The dry run may run inside a session (MIGRATING.md has Claude run it); only
  # the steps that change things need every session closed.
  [ "$MODE" = inventory ] || [ -z "$(_sessions_running)" ] \
    || _refuse "a Claude Code or Codex session is running: quit every one, the Claude desktop app and the ChatGPT app (it runs Codex)"
  if [ "${#UNINST[@]}" -eq 0 ] && [ -n "$ENTITY_ROOT" ]; then NOTHING_TO_APPLY=1; else NOTHING_TO_APPLY=0; fi
  # Whoever ran the old plugin has a brain. Its install, services or records
  # with no brain found means the brain went missing, never "nothing to
  # migrate": installing now would start an empty brain over the loss.
  # No old install left to remove is still a migration when a brain exists (the
  # old plugin was uninstalled by hand; its daemon may linger). It is not one on
  # a machine whose only brain@ ids carry the new tree, or with nothing at all.
  if [ "${#BRAIN_DIRS[@]}" -eq 0 ] && [ -n "$evidence" ]; then
    _refuse "a brain was in use on this machine, but none was found. What says so:
$(printf '%s\n' "$evidence" | sed 's/^/       /')
     Install nothing and delete nothing: send this whole output to whoever pointed you here.
     (Never used the old plugin? Then uninstall it and install entity as its README says.)"
  elif [ "${#UNINST[@]}" -eq 0 ] && [ "$NOTHING_TO_APPLY" = 0 ]; then
    if [ "${#LEFT_ALONE[@]}" -gt 0 ]; then
      _refuse "the only brain@ installs here already run the new plugin's tree (a development install): nothing to migrate"
    elif [ "${#BRAIN_DIRS[@]}" -eq 0 ]; then
      _refuse "no old install and no brain found: nothing to migrate (install entity as its README says)"
    fi
  fi
}

plan() {
  section "What --apply will do"
  local i=1 e id scope proj m
  item "$i. maintenance lock on; stop the daemon and dashboard"; i=$((i + 1))
  item "$i. back up every brain found (SQLite online backup, quick_check, node count) to ~/brain-migration-<time>/"; i=$((i + 1))
  item "$i. move the old service definitions (LaunchAgents/com.brain.*.plist) into that backup"; i=$((i + 1))
  if _knob_needed; then
    item "$i. write BRAIN_DB_DIR='$PRIMARY' into $ENVF so the new plugin finds it"; i=$((i + 1))
  fi
  for e in ${UNINST[@]+"${UNINST[@]}"}; do
    IFS="$SEP" read -r id scope proj <<< "$e"
    item "$i. claude plugin uninstall $id --keep-data --scope $scope${proj:+   (from $proj)}"; i=$((i + 1))
  done
  for m in ${MKT_REMOVE[@]+"${MKT_REMOVE[@]}"}; do
    item "$i. check the brain is still there, then: claude plugin marketplace remove $m"; i=$((i + 1))
  done
  [ -n "$ANCHOR_SRC" ] || { item "$i. claude plugin marketplace add $ENTITY_SOURCE"; i=$((i + 1)); }
  [ -n "$ENTITY_ROOT" ] || item "$i. claude plugin install entity@anchor"
}

# ── stopping the services ────────────────────────────────────────────────────
_is_ours() { case "$(ps -o command= -p "$1" 2>/dev/null)" in *daemon_server*|*brain_dashboard_standalone*) return 0 ;; esac; return 1; }
_kill_wait() {   # SIGTERM, up to 20s for a clean flush, then SIGKILL
  local pid="$1" t=0
  kill "$pid" 2>/dev/null || return 0
  while kill -0 "$pid" 2>/dev/null && [ "$t" -lt 20 ]; do sleep 1; t=$((t + 1)); done
  kill -0 "$pid" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; sleep 1; }
  item "stopped pid $pid"
}
stop_services() {
  local l t pid
  if [ "$IS_MAC" = 1 ]; then
    for l in $LABELS; do
      _loaded "$l" || continue
      launchctl bootout "gui/$UID_N/$l" >/dev/null 2>&1 || true
      t=0; while _loaded "$l" && [ "$t" -lt 20 ]; do sleep 1; t=$((t + 1)); done
      _loaded "$l" && stop "launchd still has $l loaded after bootout"
      item "unloaded $l"
    done
  fi
  # daemons launchd never managed (the direct-spawn fallback), and one still flushing
  for pid in $(cat "$PIDFILE" 2>/dev/null) $(_listener_pid); do
    kill -0 "$pid" 2>/dev/null && _is_ours "$pid" && _kill_wait "$pid"
  done
  pid="$(_listener_pid)"
  [ -z "$pid" ] || stop "something that is not a brain daemon listens on port $PORT: $(ps -o command= -p "$pid")"
  # whatever still holds a brain's database open: ours are stopped, others named
  local d holders
  for d in ${BRAIN_DIRS[@]+"${BRAIN_DIRS[@]}"}; do
    for pid in $(lsof -t -- "$d/brain.db" "$d/brain_logs.db" 2>/dev/null | sort -u); do
      _is_ours "$pid" && _kill_wait "$pid"
    done
    holders="$(lsof -t -- "$d/brain.db" "$d/brain_logs.db" 2>/dev/null | sort -u)"
    [ -z "$holders" ] || stop "these processes still hold $d open — quit them and rerun:
$(for pid in $holders; do ps -o pid=,command= -p "$pid"; done)"
  done
  item "no process holds a brain open"
}

_assert_brains() {   # every brain that must survive an uninstall still has its brain.db
  local d
  for d in ${BRAIN_DIRS[@]+"${BRAIN_DIRS[@]}"}; do
    [ "$(_class "$d")" = inside-plugin ] && continue
    [ -f "$d/brain.db" ] || stop "brain.db vanished from $d. Restore it from $BK/brains before anything else."
  done
}

LOCK_OURS=0
cleanup() { [ "$LOCK_OURS" = 1 ] && rm -f "$LOCK"; }
trap cleanup EXIT
trap 'stop "interrupted"' INT TERM

_state() { local kv; for kv in "$@"; do printf '%s=%q\n' "${kv%%=*}" "${kv#*=}"; done >> "$BK/state.env"; }

way_back() {
  section "The way back (nothing was deleted)"
  if [ -n "${BRAIN:-}" ]; then
    item "the brain exactly as it was before the migration: ${COPY:-?}"
    item "  (a verified copy of its databases and the files beside them, ${N:-?} nodes)"
    # entity's first start upgrades the brain it finds in place, so after
    # --finish the original is no longer as it was; its upgrade backup rides along
    if [ -n "${SPARE:-}" ]; then
      item "the brain before the move: $SPARE (the relocation renamed it beside its old path;"
      item "  entity had already opened and upgraded it)"
    elif [ "$MODE" = finish ]; then
      item "the brain itself, where it was: $BRAIN (entity has opened and upgraded it)"
    else
      item "the brain itself, still untouched: $BRAIN (entity upgrades it at its first start)"
    fi
    item "to run entity on the original: quit Claude Code, then"
    item "    printf \"\\nBRAIN_DB_DIR='%s'\\n\" '$(_sq "${SPARE:-$BRAIN}")' >> '$(_sq "$ENVF")'"
    item "  and start a new session (the service follows that setting)."
    item "to use the copy instead: copy everything in ${COPY:-?} except SOURCE into an empty folder and name that folder the same way."
  fi
  item "the old service definitions are kept in $BK/LaunchAgents, for inspection only."
  item "the old plugin cannot be reinstalled: its catalog was renamed. Bring this output to whoever sent you here."
}

# ── modes ────────────────────────────────────────────────────────────────────
dry_run() {
  inventory
  plan
  section "Result"
  if [ "$NOTHING_TO_APPLY" = 1 ]; then
    say "   Already applied: no old install left and entity@anchor is installed. Next: bash $0 --finish"
  elif [ -n "$REFUSE" ]; then
    say "   NOT READY. --apply would refuse:"; printf '%s' "$REFUSE"; exit 1
  else
    say "   READY. Nothing was changed. Quit every Claude Code and Codex session, the Claude desktop app"
    say "   and the ChatGPT app (it runs Codex),"
    say "   then run, from a plain terminal:  bash $0 --apply${BRAIN_ARG:+ --brain '$(_sq "$BRAIN_ARG")'}"
  fi
}

apply() {
  inventory
  [ "$NOTHING_TO_APPLY" = 0 ] || { say; say "Nothing to apply: no old install left and entity@anchor is installed. Next: bash $0 --finish"; exit 0; }
  [ -z "$REFUSE" ] || stop "--apply refuses:"$'\n'"$REFUSE"

  BK="$HOME/brain-migration-$(date +%Y%m%dT%H%M%S)"
  mkdir -m 700 "$BK" || stop "cannot create $BK"
  exec > >(tee -a "$BK/migrate.log") 2>&1
  local primary_cls=""; [ -z "$PRIMARY" ] || primary_cls="$(_class "$PRIMARY")"
  _state STAGE=started BRAIN="$PRIMARY" CLASS="$primary_cls" MKTS_REMOVED="${MKT_REMOVE[*]:-}"

  section "1/7 maintenance lock"
  : > "$LOCK" || stop "cannot write $LOCK"
  LOCK_OURS=1
  item "on: $LOCK (no daemon starts while it exists)"
  STOP_NOTE="Nothing was changed. The brain's services come back at your next login (or your next session, if the old plugin still loads). Rerun --apply when ready."

  section "2/7 stop the brain's services"
  stop_services

  section "3/7 back up to $BK"
  mkdir -p "$BK/brains" "$BK/config" "$BK/LaunchAgents"
  local i=0 d db r nodes f N=0 COPY=""
  for d in ${BRAIN_DIRS[@]+"${BRAIN_DIRS[@]}"}; do
    i=$((i + 1)); nodes="-"
    mkdir -p "$BK/brains/$i"
    for db in brain.db brain_logs.db; do
      [ -f "$d/$db" ] || continue
      r="$(_py backup "$d/$db" "$BK/brains/$i/$db" stopped)" || stop "backing up $d/$db failed (see above)"
      [ "${r%%$'\t'*}" = ok ] || stop "verifying the copy of $d/$db: ${r%%$'\t'*}"
      [ "$db" = brain.db ] && nodes="${r#*$'\t'}"
    done
    # state kept beside the databases, not in them: the aspect taxonomy and
    # its proposals among it
    for f in "$d"/*.json; do
      [ -f "$f" ] && { cp -p "$f" "$BK/brains/$i/" || stop "could not copy $f"; }
    done
    printf 'source=%s\nclass=%s\nnodes=%s\n' "$d" "$(_class "$d")" "$nodes" > "$BK/brains/$i/SOURCE"
    item "$d -> brains/$i  ($nodes nodes; quick_check ok, every table's rows match)"
    [ "$d" = "$PRIMARY" ] && { N="$nodes"; COPY="$BK/brains/$i"; }
  done
  for f in "$ENVF" "$RESOLVED" "$REGISTRY" "$MARKETS" "$CC_DIR/settings.json"; do
    [ -f "$f" ] && cp -p "$f" "$BK/config/$(basename "$(dirname "$f")")-$(basename "$f")"
  done
  item "config files -> config/"
  _state N="$N" COPY="$COPY"

  section "4/7 set the old services aside"
  local l
  for l in $LABELS; do
    [ -f "$LA/$l.plist" ] || continue
    mv "$LA/$l.plist" "$BK/LaunchAgents/" || stop "could not move $LA/$l.plist"
    item "$l.plist -> LaunchAgents/"
  done
  STOP_NOTE="Services are stopped and their definitions are in $BK/LaunchAgents; nothing is uninstalled. Rerun --apply to continue. The brain is untouched at ${PRIMARY:-?}."

  section "5/7 point the new plugin at the brain"
  if _knob_needed; then
    _py knob "$ENVF" "$PRIMARY" || stop "could not write $ENVF"
    item "BRAIN_DB_DIR='$PRIMARY' in $ENVF (previous copy: config/brain-env)"
  else
    item "nothing to write: the new plugin finds the brain on its own"
  fi

  section "6/7 remove the old plugin, keeping its data"
  STOP_NOTE="The old plugin may be partly or fully uninstalled (data kept). Rerun --apply to finish; the brain is at ${PRIMARY:-?} and copied under $BK/brains."
  local e id scope proj out m left
  for e in ${UNINST[@]+"${UNINST[@]}"}; do
    IFS="$SEP" read -r id scope proj <<< "$e"
    if [ "$scope" = user ]; then
      out="$("$CLAUDE" plugin uninstall "$id" --keep-data --scope user 2>&1)"
    else
      [ -d "$proj" ] || stop "$id is installed for $proj ($scope scope), which no longer exists. Recreate the folder or ask the maintainer."
      out="$(cd "$proj" && "$CLAUDE" plugin uninstall "$id" --keep-data --scope "$scope" 2>&1)"
    fi
    [ $? -eq 0 ] || stop "uninstall $id ($scope) failed: $out"
    _installs | awk -F"$SEP" -v i="$id" -v s="$scope" -v p="$proj" '$1 == i && $2 == s && $3 == p { f = 1 } END { exit !f }' \
      && stop "uninstall $id ($scope) reported success but the install is still registered"
    _assert_brains
    item "uninstalled $id ($scope${proj:+, $proj}); data kept"
  done
  for m in ${MKT_REMOVE[@]+"${MKT_REMOVE[@]}"}; do
    left="$(_installs | awk -F"$SEP" -v m="$m" '{ n = split($1, a, "@"); if (a[n] == m) print $1 " (" $2 ")" }')"
    [ -z "$left" ] || stop "still installed from $m: $left. Not removing it: that would delete their data."
    _assert_brains
    out="$("$CLAUDE" plugin marketplace remove "$m" 2>&1)" || stop "marketplace remove $m failed: $out"
    _assert_brains
    item "removed marketplace $m; every brain still in place"
  done
  for m in ${MKT_KEEP[@]+"${MKT_KEEP[@]}"}; do item "kept marketplace $m (other plugins come from it)"; done

  section "7/7 install entity"
  if [ -z "$ANCHOR_SRC" ]; then
    "$CLAUDE" plugin marketplace add "$ENTITY_SOURCE" || stop "marketplace add $ENTITY_SOURCE failed"
  fi
  [ -n "$(_py marketplace "$MARKETS" anchor)" ] || stop "marketplace add $ENTITY_SOURCE registered no marketplace named anchor"
  if [ -z "$ENTITY_ROOT" ]; then
    "$CLAUDE" plugin install entity@anchor --scope user || stop "install entity@anchor failed"
    collect_installs
  fi
  [ -n "$ENTITY_ROOT" ] || stop "entity@anchor is not registered after the install"
  item "entity@anchor at $ENTITY_ROOT"
  _state ENTITY_ROOT="$ENTITY_ROOT" STAGE=applied
  cleanup; LOCK_OURS=0; STOP_NOTE=""
  item "maintenance lock off"

  section "Next"
  say "   1. Open ONE Claude Code session (any folder). Its first start builds entity's runtime"
  say "      in the background: a couple of minutes. Wait until it says the install finished,"
  say "      or ask it anything once and wait for the reply."
  say "   2. Quit Claude Code completely, then run:  bash $0 --finish"
  BRAIN="$PRIMARY"
  way_back
}

finish() {
  [ "$SANDBOXED" = 0 ] || stop "HOME is not this account's home ($REAL_HOME): --finish would act on the account's real services"
  BK="$(ls -d "$HOME"/brain-migration-* 2>/dev/null | sort | tail -1)"
  { [ -n "$BK" ] && [ -f "$BK/state.env" ]; } || stop "no ~/brain-migration-* from --apply found: run --apply first"
  STAGE="" BRAIN="" CLASS="" N=0 COPY="" SPARE="" FINAL="" MKTS_REMOVED=""
  . "$BK/state.env"
  case "$STAGE" in applied|finished) ;; *) stop "the last --apply ($BK) stopped before it finished: run --apply again" ;; esac
  exec > >(tee -a "$BK/migrate.log") 2>&1
  local running; running="$(_sessions_running)"
  [ -z "$running" ] || stop "a Claude Code or Codex session is running: quit every one, the Claude desktop app and the ChatGPT app (it runs Codex) first:"$'\n'"$running"
  FAILS=0
  _pass() { item "PASS  $*"; }
  _fail() { item "FAIL  $*"; FAILS=$((FAILS + 1)); }

  section "1/5 the new plugin"
  ALL_INSTALLS="$(_installs)"
  ENTITY_ROOT="$(printf '%s\n' "$ALL_INSTALLS" | awk -F"$SEP" '$1 == "entity@anchor" && $2 == "user" { print $4; exit }')"
  [ -n "$ENTITY_ROOT" ] && [ -d "$ENTITY_ROOT" ] || stop "entity@anchor is not installed: run --apply again"
  . "$ENTITY_ROOT/hooks/scripts/runtime-state.sh" || stop "$ENTITY_ROOT looks damaged (no runtime-state.sh)"
  if ! brain_runtime_ready "$ENTITY_ROOT"; then
    if [ -f "$ENTITY_ROOT/.bootstrap-failed" ]; then
      tail -20 "$ENTITY_ROOT/.bootstrap.log" 2>/dev/null
      stop "entity's first-run install failed (log above). Open a new session to retry it, then rerun --finish."
    fi
    if [ ! -f "$ENTITY_ROOT/.bootstrap.log" ]; then
      say "   entity has not started yet: open one Claude Code session, let it boot, quit it,"
      say "   then rerun:  bash $0 --finish"
      exit 2
    fi
    say "   entity's first-run install is still running (progress: $ENTITY_ROOT/.bootstrap.log)."
    say "   Wait for it, then rerun:  bash $0 --finish"
    exit 2
  fi
  item "runtime ready: $ENTITY_ROOT"
  # .runtime-ready lands BEFORE the first-run chain installs the services, and a
  # daemon launchd starts does not honour the maintenance lock: moving the brain
  # while install-daemon-service.sh / ensure-dashboard.sh still run could start
  # a daemon on the old path mid-copy. Wait for the chain to finish first.
  local t=0
  while ps -axo uid=,command= | awk -v u="$UID_N" -v r="$ENTITY_ROOT/hooks/scripts/" \
          '$1 == u && index($0, r) && /(ensure-runtime|install-daemon-service|ensure-dashboard)\.sh/ { f = 1 } END { exit !f }'; do
    [ "$t" -ge 180 ] && stop "entity's first-run setup is still running after 180s; wait for it, then rerun --finish"
    sleep 2; t=$((t + 2))
  done

  section "2/5 the brain's home"
  FINAL="$BRAIN"
  if _moves "$CLASS"; then
    # the move needs every session closed, and the check above was a while ago
    running="$(_sessions_running)"
    [ -z "$running" ] || stop "a session opened since --finish started: quit every one, then rerun --finish:"$'\n'"$running"
    if [ -f "$BRAIN/brain.db" ]; then
      env -u BRAIN_DB_DIR bash "$ENTITY_ROOT/hooks/scripts/relocate-brain.sh" "$BRAIN" \
        || stop "the relocation stopped (its output is above); relocation never deletes, so the brain is still at $BRAIN or at $XDG_BRAIN"
    elif [ ! -f "$XDG_BRAIN/brain.db" ]; then
      stop "the brain is neither at $BRAIN nor at $XDG_BRAIN: restore it from $COPY (see the way back)"
    fi
    FINAL="$(_n "$XDG_BRAIN")"
    SPARE="$(ls -d "$BRAIN".relocated-* 2>/dev/null | sort | tail -1)"
    _state FINAL="$FINAL" SPARE="$SPARE"
    item "brain at $FINAL; the original kept as ${SPARE:-?}"
  elif [ -n "$BRAIN" ]; then
    item "stays at $BRAIN ($CLASS location)"
  else
    FINAL="$(_n "$XDG_BRAIN")"
    item "no brain existed before the migration: entity's own brain is at $FINAL"
  fi

  section "3/5 the running daemon"
  local pid="" cmd db
  t=0
  while [ -z "$pid" ] && [ "$t" -lt 120 ]; do
    pid="$(_listener_pid)"
    [ -n "$pid" ] && break
    [ "$t" = 15 ] && _loaded com.brain.daemon && launchctl kickstart "gui/$UID_N/com.brain.daemon" >/dev/null 2>&1
    sleep 1; t=$((t + 1))
  done
  local root_n; root_n="$(_n "$ENTITY_ROOT")"
  if [ -z "$pid" ]; then
    _fail "no daemon on port $PORT after 120s"
  else
    cmd="$(ps -o command= -p "$pid" 2>/dev/null)"
    case "$cmd" in
      *"$ENTITY_ROOT/"*|*"$root_n/"*) _pass "daemon pid $pid runs entity's code" ;;
      *) _fail "daemon pid $pid runs something else: $cmd" ;;
    esac
    t=0; db=""
    while [ -z "$db" ] && [ "$t" -lt 60 ]; do db="$(_pid_brain "$pid")"; [ -n "$db" ] || { sleep 1; t=$((t + 1)); }; done
    if [ "$(_n "$db")" = "$FINAL" ]; then _pass "it serves $FINAL"; else _fail "it serves '${db:-nothing}', expected $FINAL"; fi
  fi

  section "4/5 services and records"
  local prog pdb l
  if [ "$IS_MAC" = 1 ]; then
    for l in $LABELS; do
      if [ ! -f "$LA/$l.plist" ]; then
        if [ "$l" = com.brain.daemon ]; then _fail "no $l.plist"; else item "      $l: not installed (the /dashboard skill installs it)"; fi
        continue
      fi
      { IFS= read -r prog; IFS= read -r pdb; } < <(_py plist "$LA/$l.plist")
      case "$prog" in
        "$ENTITY_ROOT"/*|"$root_n"/*) _pass "$l runs from entity" ;;
        *) _fail "$l runs $prog" ;;
      esac
      if [ "$(_n "$pdb")" = "$FINAL" ]; then _pass "$l names $FINAL"; else _fail "$l names '$pdb', expected $FINAL"; fi
    done
  fi
  local rdb rroot
  rdb="$(_envkey "$RESOLVED")"
  rroot="$( [ -f "$RESOLVED" ] && (PLUGIN_ROOT=''; . "$RESOLVED" >/dev/null 2>&1; printf '%s' "$PLUGIN_ROOT"))"
  if [ "$(_n "$rdb")" = "$FINAL" ] && [ "$(_n "$rroot")" = "$root_n" ]; then
    _pass "resolved.env records $FINAL and entity"
  else
    _fail "resolved.env records brain '$rdb' and plugin '$rroot'"
  fi
  local old m
  old="$(printf '%s\n' "$ALL_INSTALLS" | awk -F"$SEP" '($1 ~ /^brain@/ && ($5 == "" || $5 == "brain")) || $1 == "entity@brain" { print $1 " (" $2 ")" }')"
  if [ -z "$old" ]; then _pass "no brain@ install left"; else _fail "still installed: $old"; fi
  for m in $MKTS_REMOVED; do
    if [ -z "$(_py marketplace "$MARKETS" "$m")" ]; then _pass "marketplace $m gone"; else _fail "marketplace $m is still declared"; fi
  done

  # exit 0 pass, 1 fail, 2 this Claude Code does not report the fields
  local lr=2
  if [ -n "$CLAUDE" ]; then
    "$CLAUDE" plugin list --json 2>/dev/null | python3 -c '
import json, sys
try:
    p = [x for x in json.load(sys.stdin) if x.get("id") == "entity@anchor"]
except ValueError:
    sys.exit(2)
if not p or not p[0].get("enabled", True):
    sys.exit(1)
sys.exit(2 if "mcpServers" not in p[0] else 0 if "brain" in (p[0]["mcpServers"] or {}) else 1)'
    lr=$?
  fi
  case "$lr" in
    0) _pass "Claude Code lists entity@anchor enabled, with its brain tools server" ;;
    1) _fail "Claude Code does not list entity@anchor enabled with its brain tools server" ;;
    *) item "      could not ask Claude Code which servers entity@anchor declares (older CLI, or no claude binary)" ;;
  esac
  if (cd "$ENTITY_ROOT" && PYTHONPATH="$ENTITY_ROOT" "$ENTITY_ROOT/venv/bin/python" -c 'import servers.brain_mcp') >/dev/null 2>&1; then
    _pass "the brain tools server imports"
  else
    _fail "the brain tools server does not import: $ENTITY_ROOT/servers/brain_mcp.py"
  fi

  section "5/5 nothing lost"
  local r n
  mkdir -p "$BK/after"
  if r="$(_py backup "$FINAL/brain.db" "$BK/after/brain.db" live)"; then
    n="${r#*$'\t'}"
    if [ "${r%%$'\t'*}" != ok ]; then
      _fail "quick_check on a fresh copy of $FINAL/brain.db reported: ${r%%$'\t'*}"
    elif [ "$n" != - ] && [ "$n" -ge "${N:-0}" ] 2>/dev/null; then
      _pass "$n nodes now, $N before the migration (a few more is the brain still learning)"
    else
      _fail "$n nodes now, $N before the migration"
    fi
  else
    _fail "could not read $FINAL/brain.db"
  fi

  section "Result"
  if [ "$FAILS" = 0 ]; then
    _state STAGE=finished
    say "   DONE. Last check, by hand: open a session and ask it what it remembers about"
    say "   you. It must answer from memory; if it has no memory tools, send this output."
    say "   From here on every release is an ordinary \`claude plugin update entity@anchor\`."
  else
    say "   $FAILS check(s) FAILED. Nothing was deleted; see the way back below."
  fi
  way_back
  [ "$FAILS" = 0 ]
}

case "$MODE" in
  inventory) dry_run ;;
  apply)     apply ;;
  finish)    finish ;;
esac
