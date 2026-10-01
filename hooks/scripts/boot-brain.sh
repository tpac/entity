#!/bin/bash
# brain — SessionStart hook: boots brain, prints the boot context (stance, identity, Frame).
# Output: full brain state for Claude's context (injected via SessionStart stdout)
#
# Brain DB resolution: see resolve-brain-db.sh — explicit choice (env/config
# knob) > existing brain (XDG service dir, plugin-data, legacy) > fresh create
# at ${XDG_DATA_HOME:-~/.local/share}/brain (D-13). If a candidate brain sits
# at an unreachable old default, the resolver refuses to create and this hook
# surfaces adoption instead — silent data loss is worse than a blocked boot.


# Save stdin early — inline python commands below would consume it.
# Extract session_id from hook input and pass as env var.
HOOK_STDIN=$(cat)
BRAIN_HOOK_SESSION_ID=$(echo "$HOOK_STDIN" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('session_id',''))" 2>/dev/null)
export BRAIN_HOOK_SESSION_ID
# cwd from hook input — fed to the daemon as session identity (the daemon never
# introspects the Claude env itself). See SessionContext.cwd / session_env_for.
BRAIN_HOOK_CWD=$(echo "$HOOK_STDIN" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('cwd',''))" 2>/dev/null)
export BRAIN_HOOK_CWD
# source distinguishes a real session start from resume/compact — SessionStart
# fires for all of them (empty matcher), and per-session notices must not
# re-inject on every compaction of a long session.
BRAIN_HOOK_SOURCE=$(echo "$HOOK_STDIN" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('source',''))" 2>/dev/null)

# ── Resolve ANTHROPIC_API_KEY from canonical config location ──
# Mechanism owned by api-key-env.sh (shared with brain-env.sh, which cannot be
# sourced this early — it triggers the runtime bootstrap). Single source:
# ${XDG_CONFIG_HOME:-$HOME/.config}/brain/env (mode 600, dotenv format), the
# CLI-tool convention (gh, stripe, kubectl, ...). Env-var override: a key
# already in ANTHROPIC_API_KEY (shell export) wins, so the file is read only
# when the key is still missing. The daemon's dispatch.load_env mirrors the
# env-file/shell resolution but NOT the userConfig fallback — it inherits
# ANTHROPIC_API_KEY from this hook's env on the direct-spawn path. A
# launchd-spawned daemon sees neither CLAUDE_PLUGIN_OPTION_* nor this export,
# so a userConfig-only key needs the env file there (hence the mirror below).
source "$(dirname "$0")/api-key-env.sh"
BRAIN_ENV_FILE="$(brain_user_env_file)"
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  brain_source_user_env "$BRAIN_ENV_FILE"
fi

# Inside this branch the key was missing, so anything the fallback finds came
# from userConfig — the only channel the launchd-spawned daemon cannot see.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  brain_api_key_from_plugin_option
  # MIRROR a userConfig-resolved key to the env file (mode 600). Required, not
  # optional: CLAUDE_PLUGIN_OPTION_* exists only in hook executions, and this
  # export dies with the hook — the launchd-spawned daemon (a separate process
  # tree; installed by install-daemon-service.sh on every fresh macOS install)
  # resolves the key ONLY via dispatch.load_env = env file + shell. Without
  # the mirror, a user who fills the plugin's key field still runs a keyless
  # daemon: llm_unavailable in the dashboard while boot looks fine (first
  # laptop install, 2026-07-15). Never overwrites an existing key line.
  if [[ "${ANTHROPIC_API_KEY:-}" == sk-* ]] \
     && ! grep -q '^ANTHROPIC_API_KEY=' "$BRAIN_ENV_FILE" 2>/dev/null; then
    brain_append_user_env "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" \
      && echo "[brain-boot] plugin-config key mirrored to $BRAIN_ENV_FILE (mode 600) for the background daemon" >&2
  fi
fi

# Missing key is NOT a boot failure — the daemon boots keyless by design
# (brain.llm_available gates surface/encode/S2/warms; everything local runs
# without it: runtime bootstrap, embedder, brain.db, traces, direct recall).
# Keep booting and present the key as the one remaining setup step — a
# designed onboarding stage, not an error.
BRAIN_KEYLESS_BOOT=0
if [ -z "${ANTHROPIC_API_KEY:-}" ] || [[ "${ANTHROPIC_API_KEY}" != sk-* ]]; then
  BRAIN_KEYLESS_BOOT=1   # read below: keyless warm boots bring the dashboard up
fi

# Paths embedded in commands the USER will paste: single-quote them with
# internal quotes escaped, so a path carrying spaces/$/backticks can neither
# break the command nor be expanded by the user's shell. Shared by every
# notice below that prints a path-bearing command.
_sq() { printf %s "$1" | sed "s/'/'\\\\''/g"; }

# Emitted at the two points a keyless boot actually proceeds (cold install,
# and warm boot after DB resolution succeeds) — NOT at detection: a boot the
# adoption net blocks must not first promise "memory works from this session
# on" and then refuse to start a brain.
_keyless_notice() {
  cat <<EOF
🧠 Entity — setup in progress

Your brain is initializing: local runtime, embedding model, memory database,
and the brain daemon are being set up automatically — no action needed for
those. Memory storage, history traces, and direct recall work from this
session on.

One step remains to complete setup — learning (writing new memories) and
automatic memory surfacing use the Anthropic API and need your key:

  → Open http://localhost:${DASHBOARD_PORT:-47303}/setup and paste it there.
    (That's your brain's local dashboard — it also shows what Entity
    remembers, recalls and learns. The page runs on this machine, and the
    key you paste is saved to your local config.
    On a brand-new install the page comes alive when the first-run setup
    finishes, ~1-2 minutes.)

  Alternatives: the Entity plugin's settings in Claude Code — fill the
  "Anthropic API key" field, then start a new session (it's applied at
  session start). Or the env file directly:
     mkdir -p "${XDG_CONFIG_HOME:-\$HOME/.config}/brain"
     printf 'ANTHROPIC_API_KEY=sk-ant-...\n' > "${XDG_CONFIG_HOME:-\$HOME/.config}/brain/env"
     chmod 600 "${XDG_CONFIG_HOME:-\$HOME/.config}/brain/env"

Get a key at https://console.anthropic.com/settings/keys — the dashboard
and env-file paths take effect on your next message, no restart needed.
EOF
}
if [ "$BRAIN_KEYLESS_BOOT" = "1" ]; then
  cat >&2 <<EOF
[brain-boot] ANTHROPIC_API_KEY not set — booting in local-only mode (no encode/surface).
[brain-boot] Key location: $BRAIN_ENV_FILE — or the /setup page on the dashboard.
EOF
fi

# ── Cold install: own the bootstrap DETACHED, answer instantly ─────────────
# resolve-brain-db.sh → brain-env.sh → ensure-runtime.sh blocks 60-90s on a
# truly fresh install — this hook has a 15s timeout, so inline bootstrap
# means CC kills us and the identity injection + key notice are DROPPED
# (first laptop install, 2026-07-17). Instead: launch ensure-runtime as a
# detached child (survives our death; its mkdir-lock makes concurrent
# launches from the MCP spawn safe), print a static no-Python notice, and
# exit. The key resolution + mirror above already ran (pure shell). The MCP
# launcher waits on the sentinel and connects in-session when the bootstrap
# is quick; otherwise the next session lands on the ~8ms fast path.
_BOOT_PLUGIN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$(dirname "$0")/runtime-state.sh"
if ! brain_runtime_installed "$_BOOT_PLUGIN_ROOT"; then
  # Chain the full first-run provisioning after the bootstrap, in order:
  # runtime → daemon launchd service (RunAtLoad starts the daemon — without
  # this the first session had no daemon until MCP's health monitor kicked
  # in) → dashboard (the setup URL printed above must come alive without
  # the user invoking /dashboard — it IS the presented key-entry path).
  # Positional-args bash -c: metachar-proof for any install path (quotes,
  # $, backticks — not just spaces). Subshell + nohup: detach from the
  # hook's job/process group so CC's teardown can't take the chain down.
  _BOOT_SD="$(dirname "$0")"
  ( nohup bash -c '"$1" && "$2" && "$3"' _ \
      "$_BOOT_SD/ensure-runtime.sh" \
      "$_BOOT_SD/install-daemon-service.sh" \
      "$_BOOT_SD/ensure-dashboard.sh" \
      >> "$_BOOT_PLUGIN_ROOT/.bootstrap.log" 2>&1 & )
  [ "$BRAIN_KEYLESS_BOOT" = "1" ] && _keyless_notice
  cat <<EOF

🧠 Entity — first-run install in progress

Entity is building its local runtime in the background (isolated Python +
embedding model — a couple of minutes on typical networks; progress:
$_BOOT_PLUGIN_ROOT/.bootstrap.log). Nothing to do. Memory tools usually
appear from the NEXT session on (this session only on a very fast
connection) — everything else about this session works normally.
EOF
  echo "[brain-boot] cold install — bootstrap detached, hook exiting fast" >&2
  exit 0
fi

# Installed but synced to an older uv.lock: re-sync detached (the same
# mkdir-locked bootstrap) and boot on the libraries already in place — this
# session keeps its identity and memory; the next one runs on the new set.
if ! brain_runtime_ready "$_BOOT_PLUGIN_ROOT"; then
  ( nohup "$(dirname "$0")/ensure-runtime.sh" \
      >> "$_BOOT_PLUGIN_ROOT/.bootstrap.log" 2>&1 & )
  export BRAIN_RUNTIME_SERVE_STALE=1
  echo "[brain-boot] runtime re-sync to uv.lock started in the background" >&2
fi

# Keyless warm boot: make sure the dashboard (and with it the /setup page the
# notices point to) is actually up. The curl probe skips the fork + full env
# resolution when it already answers (the common case after the first boot);
# output goes to the bootstrap log — a dashboard that fails to come up must
# not die silently while the boot notice points users at its URL.
if [ "$BRAIN_KEYLESS_BOOT" = "1" ]; then
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 1 "http://127.0.0.1:${DASHBOARD_PORT:-47303}/" 2>/dev/null)" != "200" ]; then
    ( nohup "$(dirname "$0")/ensure-dashboard.sh" \
        >> "$(cd "$(dirname "$0")/../.." && pwd)/.bootstrap.log" 2>&1 & )
  fi
fi

source "$(dirname "$0")/resolve-brain-db.sh"

# The old `brain` plugin's installs still registered with Claude Code: brain@*
# carrying the old tree, or the entity@brain its renamed catalog offers. The
# same rule as migrate-from-brain.sh's collect_installs, which cannot import
# anything (it runs before entity exists). Where the old plugin is installed,
# the brain belongs to the migration, not to a fresh start here.
_old_plugin_installs() {
  python3 - "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/plugins/installed_plugins.json" 2>/dev/null <<'PY' || true
import json, os, sys
try:
    plugins = json.load(open(sys.argv[1])).get('plugins') or {}
except (OSError, ValueError):
    sys.exit(0)
for pid, entries in sorted(plugins.items()):
    name, _, market = pid.partition('@')
    for e in entries:
        try:
            with open(os.path.join(e.get('installPath') or '', '.claude-plugin', 'plugin.json')) as f:
                tree = json.load(f).get('name', '')
        except (OSError, ValueError):
            tree = ''
        if (name == 'brain' and tree in ('', 'brain')) or (name == 'entity' and market == 'brain'):
            print(pid)
            break
PY
}
# The public install carries the migration script and its page beside the
# plugin; the packaged zip (uploads, Codex) carries neither, only the page's name.
if [ -f "$PLUGIN_ROOT/migrate-from-brain.sh" ]; then
  _MIGRATE_HOW="    bash '$(_sq "$PLUGIN_ROOT/migrate-from-brain.sh")'
(the steps after it: $PLUGIN_ROOT/MIGRATING.md)"
else
  _MIGRATE_HOW="    MIGRATING.md in Entity's repository: its step 1"
fi

# No DB resolved. Two distinct cases:
#  (a) the adoption net fired — an existing brain sits at an old host-owned
#      default the ladder can no longer reach (plugin renamed/reinstalled).
#      Refusing to create IS the feature: surface adoption, repeat every
#      session until the user sets the knob. Never create/move/delete here.
#  (b) genuinely nothing found and auto-create failed — guide the user.
# The notice is written to be answerable without any lookup: candidate path,
# exact commands, both options inline.
if [ -z "$BRAIN_DB_DIR" ]; then
  _CFG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/brain"
  # the resolver (sourced above) exports the service dir — one owner, no
  # re-derived copy of the literal here
  _XDG_FRESH="$BRAIN_XDG_DIR"
  # printf with a leading \n (not echo >>): a knob file whose last line lacks
  # a trailing newline must not have the new line glued onto it.
  _CFG_Q="'$(_sq "$_CFG_FILE")'"
  _OLD_INSTALLS="$(_old_plugin_installs | tr '\n' ' ')"
  if [ -n "${BRAIN_MISSING_BRAIN:-}" ]; then
    cat <<EOF

🧠 Entity — your brain is missing

This machine's brain lived at:

    $BRAIN_MISSING_BRAIN

and it is no longer there. Nothing was created: a new, empty brain would
hide the loss.
${_OLD_INSTALLS:+
The old brain plugin is still installed here: $_OLD_INSTALLS
}
If the old \`brain\` plugin was uninstalled, or its folder deleted, its
background service may still be running and holding your memories. Do not
restart or log out until this is sorted. The migration's first step only
reads, and says what it finds:

$_MIGRATE_HOW

Send its whole output to whoever pointed you to Entity.

If you moved the brain yourself, point Entity at it:
       mkdir -p $_CFG_Q
       printf "\nBRAIN_DB_DIR='%s'\n" '/path/to/the/brain/folder' >> $_CFG_Q/env

To start over with a new, empty brain on purpose:
       mkdir -p $_CFG_Q
       printf "\nBRAIN_DB_DIR='%s'\n" '$(_sq "$_XDG_FRESH")' >> $_CFG_Q/env

Claude: present this to the user now, with the exact commands above. Do not
run them or create/move/delete anything yourself unless the user explicitly
asks. This notice repeats every session until the brain is found or
BRAIN_DB_DIR is set.
EOF
    echo "[brain-boot] missing brain: a record names $BRAIN_MISSING_BRAIN, nothing there — refused silent create" >&2
    exit 0
  fi
  if [ -n "${BRAIN_ADOPTION_CANDIDATE:-}" ] && [ -n "$_OLD_INSTALLS" ]; then
    cat <<EOF

🧠 Entity — the old brain plugin is still installed

Your brain (memories intact) is at:

    $BRAIN_ADOPTION_CANDIDATE

and the old plugin that made it is still installed: $_OLD_INSTALLS
Entity does not start a second brain beside it. Nothing was created, moved,
or deleted. Move over with the migration: it backs the brain up, removes the
old plugin while keeping its data, and keeps your brain. Its first step only
reads:

$_MIGRATE_HOW

then follow what it prints.

Claude: present this to the user now. Do not run the migration yourself
beyond its first, read-only step unless the user explicitly asks. This
notice repeats every session until the migration is done.
EOF
    echo "[brain-boot] adoption net: candidate brain at $BRAIN_ADOPTION_CANDIDATE, old plugin installed ($_OLD_INSTALLS) — pointed at the migration" >&2
    exit 0
  fi
  if [ -n "${BRAIN_ADOPTION_CANDIDATE:-}" ]; then
    _CAND_Q="'$(_sq "$BRAIN_ADOPTION_CANDIDATE")'"
    cat <<EOF

🧠 Entity — your existing brain needs one pointer

An existing brain (memories intact) was found at:

    $BRAIN_ADOPTION_CANDIDATE

but it lives at an old install location this session's hooks can no longer
reach automatically (this happens after the plugin is renamed or
reinstalled). Nothing was created, moved, or deleted — this session runs
without memory rather than silently starting a new, empty brain.

Pick one, then start a new session:

  1. MOVE IT to the standard location (recommended — plugin renames,
     updates, and uninstalls never touch it). One command does it safely:
     stops the daemon under the maintenance lock, verifies the copy, keeps
     the original as a renamed spare, and restores services:
       bash '$(_sq "$(dirname "$0")")/relocate-brain.sh' $_CAND_Q
     No pointer needed afterwards — it is found there automatically.

  2. KEEP IT WHERE IT IS — point Entity at it:
       mkdir -p $_CFG_Q
       printf "\nBRAIN_DB_DIR='%s'\n" $_CAND_Q >> $_CFG_Q/env
     (or set the brain path field in the Entity plugin's settings to that
      same folder)
     ⚠ That folder is deleted by a default \`claude plugin uninstall\`
       (only --keep-data spares it). A relocation notice will repeat each
       session; silence it with BRAIN_PARKED_ACK=1 in $_CFG_Q/env.

  3. START FRESH — leave the old brain untouched on disk and begin a new
     one at the standard location:
       mkdir -p $_CFG_Q
       printf "\nBRAIN_DB_DIR='%s'\n" '$(_sq "$_XDG_FRESH")' >> $_CFG_Q/env

Claude: present these three options to the user now, with the exact commands
above — everything needed is in this notice, no lookups required. Do not
run the commands or create/move/delete anything yourself unless the user
explicitly chooses an option and asks you to apply it. This notice repeats
every session until the brain is moved or BRAIN_DB_DIR is set.
EOF
    echo "[brain-boot] adoption net: candidate brain at $BRAIN_ADOPTION_CANDIDATE — refused silent create" >&2
    exit 0
  fi
  cat <<EOF

brain: No brain.db found and auto-create failed.

Two options:

  1. CONNECT TO EXISTING BRAIN — point Entity at your brain folder:
       mkdir -p $_CFG_Q
       printf "\nBRAIN_DB_DIR='%s'\n" '/path/to/your/brain/folder' >> $_CFG_Q/env
     The folder should contain (or will contain) brain.db.

  2. START FRESH — create a new brain at the standard location:
       mkdir -p '$(_sq "$_XDG_FRESH")'
     Then restart this session. The brain will initialize automatically.

Searched locations:
  - \$BRAIN_DB_DIR env var (not set)
  - $_CFG_FILE/env BRAIN_DB_DIR= (not set)
  - $_XDG_FRESH/ (standard — not found, create failed)
  - /sessions/*/mnt/AgentsContext/brain/ (Cowork — not found)
  - \$CLAUDE_PLUGIN_DATA/brain/ ($([ -n "$CLAUDE_PLUGIN_DATA" ] && echo "not found at $CLAUDE_PLUGIN_DATA/brain" || echo "\$CLAUDE_PLUGIN_DATA not set"))
  - \$HOME/AgentsContext/brain/ (not found, legacy)

EOF
  exit 0
fi

# Keyless warm boot proceeds with a resolved brain — now the setup notice is
# true (local memory works this session; only encode/surface need the key).
[ "$BRAIN_KEYLESS_BOOT" = "1" ] && _keyless_notice

# A resolved brain parked under a host plugin-data root (BRAIN_HOST_PARKED,
# set by the resolver on path shape): memory works this session, but a
# default `claude plugin uninstall` deletes that whole tree. The relocation
# itself lives in relocate-brain.sh — one owner for the lock/stop/verify
# sequence; this notice only points at it. Gated on source=startup so a
# resume or compaction doesn't re-inject it mid-session; repeats at each new
# session until the brain is moved or BRAIN_PARKED_ACK acknowledges staying.
_parked_notice() {
  _P_ID="$(basename "$(dirname "$BRAIN_HOST_PARKED")")"
  cat <<EOF

🧠 Entity — your brain lives in a folder \`claude plugin uninstall\` deletes

Memory works normally this session, but the brain's files sit at:

    $BRAIN_HOST_PARKED

A plugin UPDATE is safe there; a default UNINSTALL deletes that whole folder,
memories included (only \`claude plugin uninstall <plugin> --keep-data\` spares
it — the plugin whose data folder is \`$_P_ID\`; \`claude plugin list\` shows
which). Recommended: move the brain to the standard location, which no plugin
operation ever touches. One command does it safely — stops the daemon under
the maintenance lock, verifies the copy, keeps the original as a renamed
spare, and restores services:

    bash '$(_sq "$(dirname "$0")")/relocate-brain.sh'

Then start a new session. Prefer to stay put? Silence this notice with
BRAIN_PARKED_ACK=1 in $BRAIN_ENV_FILE

Claude: tell the user. Run the script only if they explicitly ask. This
notice repeats at each new session until the brain is moved or acknowledged.
EOF
}
[ -n "${BRAIN_HOST_PARKED:-}" ] && [ "${BRAIN_HOOK_SOURCE:-startup}" = "startup" ] && _parked_notice

# A brain resolved while the old plugin is still installed: both plugins now
# run on one brain. Memory works; the migration is what ends the overlap.
if [ "${BRAIN_HOOK_SOURCE:-startup}" = "startup" ]; then
  _OLD_INSTALLS="$(_old_plugin_installs | tr '\n' ' ')"
  [ -z "$_OLD_INSTALLS" ] || cat <<EOF

🧠 Entity — the old brain plugin is still installed beside Entity

Memory works this session, but the old plugin ($_OLD_INSTALLS) still runs
on the same brain. Finish the move so only one plugin runs it. The migration's
first step only reads:

$_MIGRATE_HOW

Claude: tell the user. Run only that first step, and only if they ask.
EOF
fi

# ── Provision the launchd service on fresh installs (macOS, idempotent) ──
# Install the LaunchAgent BEFORE ensure_daemon() so launchd owns the daemon from
# the first boot (KeepAlive + RunAtLoad). Without this, a fresh macOS install has
# no plist, so ensure_daemon's manages() is False and it direct-spawns a DETACHED
# daemon — no KeepAlive, no boot persistence. No-op on every non-fresh boot and
# off macOS. Non-fatal: on failure ensure_daemon's detached fallback still brings
# the daemon up (boot never has set -e). --boot: a drifted plist on a loaded
# service is reinstalled by a background run after this boot, since waiting out
# the running daemon's shutdown can outlast this hook's timeout; exit 3 says so.
bash "$(dirname "$0")/install-daemon-service.sh" --boot
_SERVICE_RC=$?

# ── Dashboard plist drift (managed installs only) ──
# ensure-dashboard.sh owns first-install (keyless boot / the /dashboard skill);
# this call ONLY reconciles an ALREADY-managed dashboard whose installed plist
# drifted from the template — without it, a drifted-but-up dashboard passes
# every curl probe and keeps serving stale config until /dashboard is invoked.
# Backgrounded: no boot-latency cost; up + no drift exits in one sed + cmp.
if [ "$(uname -s)" = "Darwin" ] && launchctl print "gui/$(id -u)/com.brain.dashboard" >/dev/null 2>&1; then
  ( nohup "$(dirname "$0")/ensure-dashboard.sh" \
      >> "$(cd "$(dirname "$0")/../.." && pwd)/.bootstrap.log" 2>&1 & )
fi

# ── Start daemon via ensure_daemon() — fcntl-locked singleton ──
# No inline spawning. ensure_daemon() handles: lock, ping, spawn, code-change restart.
# Subshell cd: `python3 -c` puts the cwd at sys.path[0], AHEAD of PYTHONPATH —
# run from a user project with its own top-level servers/ package and we would
# import THAT one (restart-daemon.sh pins the same hazard). PLUGIN_ROOT is set
# by resolve-brain-db.sh above.
# Skipped when the service reinstall was handed to a background run (exit 3):
# the plist is replaced only after this boot, so a restart now would relaunch
# the daemon on the old plist — onto the old brain after a move, where
# ensure_daemon's db-dir check never converges — and the background run
# restarts it again anyway. This boot uses the daemon that is running.
if [ "$_SERVICE_RC" = 3 ]; then
  echo "[brain-boot] daemon service reinstall runs after this boot — booting on the running daemon" >&2
else
  (cd "$PLUGIN_ROOT" && PYTHONPATH="$PLUGIN_ROOT" python3 -c "
from servers.daemon_client import ensure_daemon; import os, sys
db = os.path.join(os.environ.get('BRAIN_DB_DIR', ''), 'brain.db')
sys.stderr.write('[brain-boot] Daemon %s\n' % ('ready' if ensure_daemon(db) else 'FAILED'))
")
fi

# ── Verify MCP server can start ──
# Claude Code starts the MCP server as a separate process. If it crashes (e.g. import error),
# Claude silently gets no brain tools. Catch that here and scream.
(cd "$PLUGIN_ROOT" && PYTHONPATH="$PLUGIN_ROOT" python3 -c "
import sys
try:
    from servers.brain_mcp import TOOLS
    sys.stderr.write('[brain-boot] MCP server OK — %d tools available\n' % len(TOOLS))
except Exception as e:
    # Write to stdout so it appears in session context
    print('⚠️  MCP SERVER BROKEN — Entity will have NO direct brain tools')
    print('   Error: %s' % e)
    print('   The brain hooks still work, but recall/remember/connect MCP tools are dead.')
    print('   Fix the import error in servers/brain_mcp.py and restart the session.')
    sys.exit(1)
") 2>&1
MCP_EXIT=$?
if [ $MCP_EXIT -ne 0 ]; then
  # Also check for crash sentinel from a previous failed startup — the path
  # daemon_config.get_mcp_crash_path() gives the MCP server.
  MCP_CRASH_FILE="/tmp/brain-mcp-crash-$(id -u)${BRAIN_INSTANCE:+-$BRAIN_INSTANCE}.txt"
  if [ -f "$MCP_CRASH_FILE" ]; then
    echo ""
    echo "Last MCP crash details:"
    head -5 "$MCP_CRASH_FILE"
  fi
fi

# Run boot
exec "${BRAIN_PYTHON_HOOK:-python3}" "$(dirname "$0")/boot_brain.py"
