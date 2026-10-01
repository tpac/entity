#!/bin/bash
# brain — Claude Code status line script.
# Reads brain daemon status from /tmp/brain-status-{uid}.json
# and formats a persistent one-line display at bottom of terminal.
#
# Configure in ~/.claude/settings.json:
#   "statusLine": {
#     "type": "command",
#     "command": "/path/to/brain/hooks/scripts/brain-statusline.sh"
#   }

# Read Claude Code input (model, context window, etc.)
INPUT=$(cat)
MODEL=$(echo "$INPUT" | jq -r '.model.display_name // "?"' 2>/dev/null)
PCT=$(echo "$INPUT" | jq -r '.context_window.used_percentage // 0' 2>/dev/null | cut -d. -f1)

# Read brain status file. Instance-keyed like daemon_config.get_status_path():
# in an entity session (BRAIN_INSTANCE set) show the entity's daemon, not
# production's.
UID_NUM=$(id -u)
STATUS_FILE="/tmp/brain-status-${UID_NUM}${BRAIN_INSTANCE:+-$BRAIN_INSTANCE}.json"

if [ ! -f "$STATUS_FILE" ]; then
    printf '%b' "[\033[33m$MODEL\033[0m] \033[31m🧠 Brain offline\033[0m | ctx ${PCT}%%"
    exit 0
fi

# Parse brain status
NODES=$(jq -r '.nodes // 0' "$STATUS_FILE" 2>/dev/null)
LOCKED=$(jq -r '.locked // 0' "$STATUS_FILE" 2>/dev/null)
TENSIONS=$(jq -r '.tensions // 0' "$STATUS_FILE" 2>/dev/null)
EMB_READY=$(jq -r '.embedder_ready // false' "$STATUS_FILE" 2>/dev/null)
LAST_ENCODE=$(jq -r '.last_encode_at // ""' "$STATUS_FILE" 2>/dev/null)
UPDATED=$(jq -r '.updated_at // ""' "$STATUS_FILE" 2>/dev/null)

# Calculate time since last encode
if [ -n "$LAST_ENCODE" ] && [ "$LAST_ENCODE" != "null" ]; then
    NOW=$(date +%s 2>/dev/null)
    # Try to parse the timestamp (macOS date -j vs GNU date -d)
    ENCODE_TS=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$LAST_ENCODE" +%s 2>/dev/null || \
                date -d "$LAST_ENCODE" +%s 2>/dev/null || echo "0")
    if [ "$ENCODE_TS" -gt 0 ] 2>/dev/null; then
        DIFF=$(( (NOW - ENCODE_TS) / 60 ))
        if [ "$DIFF" -lt 60 ]; then
            ENCODE_AGE="${DIFF}m"
        else
            ENCODE_AGE="$(( DIFF / 60 ))h"
        fi
    else
        ENCODE_AGE="?"
    fi
else
    ENCODE_AGE="never"
fi

# Embedder indicator
if [ "$EMB_READY" = "true" ]; then
    EMB_ICON="\033[32m✓\033[0m"
else
    EMB_ICON="\033[31m✗\033[0m"
fi

# Tension indicator
if [ "$TENSIONS" -gt 0 ] 2>/dev/null; then
    TENSION_STR=" | ⚡${TENSIONS}"
else
    TENSION_STR=""
fi

# Format: [Model] 🧠 894 nodes | 🔒 692 | ✓emb | ⚡3 | enc 5m | ctx 42%
printf '%b' "[\033[33m$MODEL\033[0m] \033[36m🧠 ${NODES}\033[0m | 🔒${LOCKED} | ${EMB_ICON}emb${TENSION_STR} | enc ${ENCODE_AGE} | ctx ${PCT}%%"
