#!/bin/bash
# brain — WorktreeRemove hook: clears worktree context from brain.
# Output: none
source "$(dirname "$0")/resolve-brain-db.sh"
[ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ] && exit 0
export HOOK_INPUT=$(cat)
exec "${BRAIN_PYTHON_HOOK:-python3}" "$(dirname "$0")/worktree_cleanup.py"
