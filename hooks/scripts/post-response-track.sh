#!/bin/bash
# brain — Stop hook: records the turn (S0 traces) and delivers pending self-messages.
# Output: {"decision":"block","reason":...} only when a self-message must be
# delivered (the host continues the turn with it); otherwise nothing.
#
# Pure shim: env setup + exec. ALL policy (daemon liveness, unconfigured-install
# BRAIN OFFLINE, recovery) lives in the python client / hook_common — the
# daemon owns everything behind it. Do not add gates here.
source "$(dirname "$0")/resolve-brain-db.sh"
export HOOK_INPUT=$(cat)
exec "${BRAIN_PYTHON_HOOK:-python3}" "$(dirname "$0")/post_response_track.py"
