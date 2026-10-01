#!/bin/bash
# brain — runtime bootstrap
#
# Installs an isolated Python 3.11+ runtime inside the plugin dir so the
# brain doesn't depend on the user's system Python version. Idempotent:
# builds a venv for uv.lock, writes the lock's fingerprint to
# $PLUGIN_DIR/.runtime-ready, and skips while that fingerprint still matches.
# A changed lock builds again on the next run (predicate: runtime-state.sh).
#
# Layout after a successful run:
#   $PLUGIN_DIR/bin/uv                  # uv binary (platform-correct)
#   $PLUGIN_DIR/py/<version>/...        # standalone Python, the pinned patch
#   $PLUGIN_DIR/venvs/<build>/...       # one venv per sync, exactly the uv.lock set
#   $PLUGIN_DIR/venv -> venvs/<build>   # the one serving
#   $PLUGIN_DIR/.runtime-ready          # fingerprint of the synced uv.lock
#
# A plugin update's shape: each sync builds a complete venv beside the one
# serving, and `venv` is repointed in one rename only when the build is done.
# So venv/bin/python never disappears (an in-place sync across a Python change
# deleted the whole venv first), and the build before this one stays for the
# processes started from it; older builds are pruned.
#
# Hard-fail on any error. Silent degradation back to system Python is what
# this script exists to escape from. The one fallback is to the install's own
# previous sync: when a RE-sync fails (offline, index outage, proxy), the venv
# that already served keeps serving, untouched, loudly, and the next bootstrap
# retries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/runtime-state.sh"

SENTINEL="$PLUGIN_DIR/.runtime-ready"
UV_BIN="$PLUGIN_DIR/bin/uv"
PY_DIR="$PLUGIN_DIR/py"
VENV_DIR="$PLUGIN_DIR/venv"
VENV_PY="$VENV_DIR/bin/python"
LOCK_FILE="$PLUGIN_DIR/uv.lock"

# Builtin read, not $(cat): this line runs before the fast path on every hook.
PY_VERSION="${BRAIN_PY_VERSION:-}"
[ -n "$PY_VERSION" ] || { read -r PY_VERSION < "$PLUGIN_DIR/.python-version"; } 2>/dev/null || true
# Pinned for reproducibility, and moved together with uv.lock: an older uv
# cannot read a lock a newer one wrote, so an install whose bin/uv differs
# re-fetches it before syncing (step 1).
UV_VERSION="${BRAIN_UV_VERSION:-0.12.18}"

# Corporate networks TLS-intercept HTTPS with a proxy CA that lives in the OS
# keychain but not in uv's bundled Rust cert store — uv's downloads (Python,
# PyPI deps) then die with `invalid peer certificate: UnknownIssuer` (work
# laptop install, 2026-07-30). Native TLS makes uv use the OS trust store,
# which also contains the standard public roots — a no-op on clean networks.
export UV_SYSTEM_CERTS=1

# The interpreter comes only from $PY_DIR: when `uv sync` builds a venv on a
# .python-version patch the install lacks, it downloads that patch there
# itself. only-managed stops uv taking whatever python3.11 is on PATH instead
# (measured 2026-09-23: ~/.local/bin/python3.11); automatic overrides a user's
# uv.toml `python-downloads = "never"`, which would otherwise leave an install
# on its old patch. An automatic download links nothing into ~/.local/bin,
# unlike `uv python install`.
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_DOWNLOADS=automatic

# Fast path — already bootstrapped (predicate: runtime-state.sh).
if brain_runtime_ready "$PLUGIN_DIR"; then
    exit 0
fi

# A launcher that must never block (boot-brain.sh, mcp-launch.sh) starts the
# re-sync detached and sets BRAIN_RUNTIME_SERVE_STALE=1 for its own sourcing:
# an installed runtime serves as-is while that runs.
if [ "${BRAIN_RUNTIME_SERVE_STALE:-}" = 1 ] && brain_runtime_installed "$PLUGIN_DIR"; then
    exit 0
fi

# ── Concurrency guard ──────────────────────────────────────────
# brain-env.sh has ~13 sourcers (MCP spawn, every hook, launchers, plists);
# on a clean install they ALL hit the cold path simultaneously. Unserialized,
# two racers extract uv over each other: overwriting a RUNNING (ad-hoc
# signed) binary in place invalidates its code signature and macOS SIGKILLs
# it — observed as `uv python install ... Killed: 9` on the first laptop
# install (2026-07-17), which took the whole MCP connection down with it.
# mkdir is the atomic test-and-set (macOS ships no flock(1)): exactly one
# winner runs the bootstrap; losers wait for the sentinel, then fast-path.
LOCK_DIR="$PLUGIN_DIR/.runtime-bootstrap.lock"
FAIL_SENTINEL="$PLUGIN_DIR/.bootstrap-failed"
# Stale ceiling must comfortably exceed a WORST-CASE legitimate bootstrap
# (uv + Python + ~200MB deps + ~150MB model on a slow link) — a premature
# steal re-creates the concurrent-uv SIGKILL race the lock exists to kill
# (code review 2026-07-17). Wait ceiling sits above it so the fatal branch
# is reachable only past the steal opportunity.
LOCK_STALE_S=1800    # 30 min: a lock older than this is a dead winner
WAIT_MAX_S=2100      # loser ceiling (callers' own timeouts kill far sooner)
FAIL_COOLDOWN_S=300  # a recent deterministic failure fails waiters fast

# mtime in epoch seconds, or nothing. GNU's spelling goes first: BSD stat
# rejects -c with no output, while GNU stat reads `-f` as file-system mode and
# prints a block before failing — which the fallback would have appended to,
# and whose text in $(( )) aborts the script. Digits only, or nothing.
_mtime() {
    local _m
    _m=$(stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null)
    case "$_m" in ''|*[!0-9]*) return 1 ;; esac
    echo "$_m"
}
_lock_age() {
    # seconds since the lock dir was created; 0 if unreadable (treat as fresh)
    local _m
    _m=$(_mtime "$LOCK_DIR") || { echo 0; return; }
    echo $(( $(date +%s) - _m ))
}
_file_age() {
    local _m
    _m=$(_mtime "$1") || { echo 999999; return; }
    echo $(( $(date +%s) - _m ))
}

_deadline=$(( $(date +%s) + WAIT_MAX_S ))
while :; do
    # Re-check the fast path each round — the winner may have finished.
    if brain_runtime_ready "$PLUGIN_DIR"; then
        exit 0
    fi
    # Recent deterministic failure (offline, unsupported platform): fail
    # fast instead of a thundering herd of expensive serial retries. Past
    # the cooldown, retries are allowed again (transient network heals).
    if [ -f "$FAIL_SENTINEL" ] && [ "$(_file_age "$FAIL_SENTINEL")" -lt "$FAIL_COOLDOWN_S" ]; then
        # A failed re-sync of a runtime that already served: keep serving it.
        brain_runtime_installed "$PLUGIN_DIR" && exit 0
        echo "[brain-boot] FATAL: a bootstrap attempt failed $(_file_age "$FAIL_SENTINEL")s ago ($(cat "$FAIL_SENTINEL" 2>/dev/null)) — retry after cooldown" >&2
        exit 1
    fi
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        echo "$$" > "$LOCK_DIR/owner" 2>/dev/null || true
        break  # we are the winner — run the bootstrap below
    fi
    if [ "$(_lock_age)" -gt "$LOCK_STALE_S" ]; then
        # Atomic steal: rename first — only ONE stealer's mv succeeds, so
        # two waiters can never both demolish the lock and double-win (the
        # rm-then-mkdir TOCTOU from the first cut of this code).
        if mv "$LOCK_DIR" "$LOCK_DIR.stale.$$" 2>/dev/null; then
            echo "[brain-boot] stale bootstrap lock (winner died?) — stole it" >&2
            rm -rf "$LOCK_DIR.stale.$$" 2>/dev/null || true
        fi
        continue
    fi
    if [ "$(date +%s)" -ge "$_deadline" ]; then
        echo "[brain-boot] FATAL: waited ${WAIT_MAX_S}s for a concurrent bootstrap that never finished" >&2
        exit 1
    fi
    sleep 1
done
# Winner cleanup — on ANY exit: release the lock IF still ours (the owner
# token stops a slow winner's trap from demolishing a successor's lock
# after a steal), and stamp the failure sentinel on non-zero exit so
# waiters fail fast instead of serially re-running the whole bootstrap.
# Success clears any stale failure stamp. A failed RE-sync of a runtime that
# already served exits 0 instead: the previous sync keeps serving (the sentinel
# is untouched, so the next bootstrap after the cooldown retries).
_had_runtime=0
brain_runtime_installed "$PLUGIN_DIR" && _had_runtime=1
_BUILD_DIR=""   # this run's venv build; removed unless `venv` points at it
_bootstrap_cleanup() {
    local _code=$1
    if [ "$(cat "$LOCK_DIR/owner" 2>/dev/null)" = "$$" ]; then
        rm -rf "$LOCK_DIR" 2>/dev/null || true
    fi
    if [ -n "$_BUILD_DIR" ] && [ "$(readlink "$VENV_DIR" 2>/dev/null)" != "venvs/${_BUILD_DIR##*/}" ]; then
        rm -rf "$_BUILD_DIR" 2>/dev/null || true
    fi
    if [ "$_code" -ne 0 ]; then
        echo "exit=$_code pid=$$ $(date '+%Y-%m-%dT%H:%M:%S')" > "$FAIL_SENTINEL" 2>/dev/null || true
        if [ "$_had_runtime" = 1 ]; then
            echo "[brain-boot] WARN: runtime re-sync to uv.lock failed — keeping the previously synced libraries; retrying after ${FAIL_COOLDOWN_S}s" >&2
            exit 0
        fi
    else
        rm -f "$FAIL_SENTINEL" 2>/dev/null || true
    fi
}
trap '_bootstrap_cleanup $?' EXIT

echo "[brain-boot] runtime setup — isolated Python $PY_VERSION, deps synced to uv.lock" >&2
echo "[brain-boot]   plugin dir: $PLUGIN_DIR" >&2

# ── 1. uv binary ───────────────────────────────────────────────
_uv_installed="$("$UV_BIN" --version 2>/dev/null | awk '{print $2}')" || true
if [ "$_uv_installed" != "$UV_VERSION" ]; then
    echo "[brain-boot]   [1/3] installing uv $UV_VERSION..." >&2
    mkdir -p "$PLUGIN_DIR/bin"

    # Detect platform triple for uv release artifact
    OS="$(uname -s)"
    ARCH="$(uname -m)"
    case "$OS-$ARCH" in
        Darwin-arm64)         UV_TRIPLE="aarch64-apple-darwin" ;;
        Darwin-x86_64)        UV_TRIPLE="x86_64-apple-darwin" ;;
        Linux-x86_64)         UV_TRIPLE="x86_64-unknown-linux-gnu" ;;
        Linux-aarch64|Linux-arm64) UV_TRIPLE="aarch64-unknown-linux-gnu" ;;
        *)
            echo "[brain-boot] FATAL: unsupported platform $OS-$ARCH" >&2
            exit 1
            ;;
    esac

    UV_URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_TRIPLE.tar.gz"
    TMP_TAR="$PLUGIN_DIR/bin/uv.tar.gz"

    if ! curl -LsSf --fail --max-time 120 -o "$TMP_TAR" "$UV_URL"; then
        echo "[brain-boot] FATAL: uv download failed from $UV_URL" >&2
        exit 1
    fi

    # Archive contains uv-<triple>/uv and uv-<triple>/uvx. Extract to a temp
    # dir and rename into place: mv is atomic and never invalidates a running
    # binary's code signature, unlike extracting over bin/uv in place (macOS
    # SIGKILLs a running ad-hoc-signed executable whose file is overwritten).
    # Defense in depth on top of the bootstrap lock above.
    TMP_EXTRACT="$PLUGIN_DIR/bin/.uv-extract.$$"
    mkdir -p "$TMP_EXTRACT"
    if ! tar -xzf "$TMP_TAR" -C "$TMP_EXTRACT" --strip-components=1 "uv-$UV_TRIPLE/uv"; then
        echo "[brain-boot] FATAL: uv extraction failed" >&2
        rm -rf "$TMP_EXTRACT"
        exit 1
    fi
    chmod +x "$TMP_EXTRACT/uv"
    mv -f "$TMP_EXTRACT/uv" "$UV_BIN"
    rm -rf "$TMP_EXTRACT" "$TMP_TAR"

    if [ ! -x "$UV_BIN" ]; then
        echo "[brain-boot] FATAL: uv extracted but not executable at $UV_BIN" >&2
        exit 1
    fi
fi

# ── 2. Build this lock's venv beside the one serving ───────────
if [ ! -f "$LOCK_FILE" ]; then
    echo "[brain-boot] FATAL: $LOCK_FILE missing from plugin" >&2
    exit 1
fi

# `uv sync` into a fresh directory creates the venv on the pinned Python and
# installs exactly the lock (from uv's cache where it can, cloned on APFS).
# Source checkouts (a .git entry — packages never carry one) also get the dev
# group: the test suite and py-spy.
_BUILD_DIR="$PLUGIN_DIR/venvs/$(date +%Y%m%dT%H%M%S)-$$"
_BUILD_PY="$_BUILD_DIR/bin/python"
mkdir -p "$PLUGIN_DIR/venvs"
_sync_groups=(--no-dev)
[ -e "$PLUGIN_DIR/.git" ] && _sync_groups=()
echo "[brain-boot]   [2/3] building the venv for uv.lock on Python $PY_VERSION (~200MB of deps)..." >&2
if ! UV_PROJECT_ENVIRONMENT="$_BUILD_DIR" UV_PYTHON_INSTALL_DIR="$PY_DIR" \
     "$UV_BIN" sync --project "$PLUGIN_DIR" --frozen --no-install-project \
     ${_sync_groups[@]+"${_sync_groups[@]}"} >&2; then
    echo "[brain-boot] FATAL: dep sync failed" >&2
    exit 1
fi

# ── 3. Pre-fetch embedding model ───────────────────────────────
# Pull the embedding model into fastembed's cache now, during this (already
# blocking) first-run bootstrap, so the daemon's first load_model() is cache-fast
# instead of blocking the first SessionStart on a ~150MB download. Non-fatal: if
# this fails (offline, etc.), the daemon downloads it on first recall as before.
# The model comes from the embedder pin (pyproject.toml, read by
# servers/embedder.py), so it never drifts from what the daemon loads. Run from
# the plugin dir so `servers` resolves to this tree; the install path reaches
# python only as the cwd, never interpolated into the code string.
echo "[brain-boot]   [3/3] pre-fetching the pinned embedding model..." >&2
if ! (cd "$PLUGIN_DIR" && "$_BUILD_PY" -c "import truststore; truststore.inject_into_ssl(); from fastembed import TextEmbedding; from servers.embedder import EMBEDDER_PIN; TextEmbedding(model_name=EMBEDDER_PIN['model_name'])") >&2; then
    echo "[brain-boot] WARN: model pre-fetch failed — daemon will download it on first recall (slower); continuing" >&2
fi

# ── Swap: `venv` now names the new build ───────────────────────
# One rename(2) of a relative symlink — os.replace, because mv follows a
# symlink to a directory and moves the new link INSIDE it. An install from
# before this layout has a real venv/ directory; it moves into venvs/ first,
# in the same process, so venv/ is absent for microseconds, once.
_prev="$(readlink "$VENV_DIR" 2>/dev/null)" || _prev=""
[ -d "$VENV_DIR" ] && [ ! -L "$VENV_DIR" ] && _prev="venvs/legacy-$$"
if ! "$_BUILD_PY" -c '
import os, sys
venv, link, target, legacy = sys.argv[1:]
os.symlink(target, link)
if os.path.isdir(venv) and not os.path.islink(venv):
    os.rename(venv, legacy)
os.replace(link, venv)
' "$VENV_DIR" "$PLUGIN_DIR/.venv-swap.$$" "venvs/${_BUILD_DIR##*/}" "$PLUGIN_DIR/$_prev"; then
    rm -f "$PLUGIN_DIR/.venv-swap.$$"
    echo "[brain-boot] FATAL: could not point venv at the new build" >&2
    exit 1
fi
brain_runtime_fingerprint "$PLUGIN_DIR" > "$SENTINEL"

# Keep the build that served until now (processes started from it) and this
# one; older builds go.
for _b in "$PLUGIN_DIR"/venvs/*; do
    [ -e "$_b" ] || continue
    case "venvs/${_b##*/}" in "venvs/${_BUILD_DIR##*/}"|"$_prev") continue ;; esac
    rm -rf "$_b"
done
echo "[brain-boot] runtime ready — $("$VENV_PY" -c 'import sys; print(sys.version.split()[0])')" >&2
