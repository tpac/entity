#!/usr/bin/env python3
"""PostToolUse trace capture — Scale 0 tool results.

Thin client: reads stdin JSON, sends trace event to daemon via TCP.
Attaches to the current stop's S0 chain (read from tmp file written by recall hook).
"""
import sys
import json
import socket
import os

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import run_hook, tool_target_file, strip_caller_stamp, host_tells


PATCH_TEXT_CAP = 16_384


def _raw_metadata(data, tool_name, tool_input):
    """Content-free join/probe facts plus a bounded patch; never classify tools."""
    metadata = {"tool": tool_name, "tells": host_tells(), "payload_keys": sorted(data)}
    for key in ('tool_use_id', 'turn_id', 'prompt_id'):
        if key in data:
            metadata[key] = data[key]
    if tool_name == 'apply_patch':
        patch = tool_input.get('command')
        if isinstance(patch, str):
            metadata['patch'] = patch[:PATCH_TEXT_CAP]
            if len(patch) > PATCH_TEXT_CAP:
                metadata['patch_truncated_chars'] = len(patch) - PATCH_TEXT_CAP
    return metadata


def _build_summary(tool_name, tool_input):
    """Build a human-readable summary of the tool call."""
    if tool_name in ("Edit", "Write"):
        file_path = tool_input.get("file_path", "")
        old = (tool_input.get("old_string", "") or "")[:80]
        new = (tool_input.get("new_string", "") or tool_input.get("content", "") or "")[:80]
        s = "%s: %s" % (tool_name, file_path)
        if old:
            s += "\n  old: %s\n  new: %s" % (old, new)
        return s
    elif tool_name == "Bash":
        return "Bash: %s" % (tool_input.get("command", "") or "")[:200]
    elif tool_name == "Read":
        return "Read: %s" % tool_input.get("file_path", "")
    elif tool_name == "Glob":
        return "Glob: %s" % tool_input.get("pattern", "")
    elif tool_name == "Grep":
        return "Grep: %s in %s" % (tool_input.get("pattern", ""), tool_input.get("path", "."))
    elif tool_name == "apply_patch":
        return "apply_patch: %s" % tool_target_file(tool_input)
    elif tool_name == "Agent":
        return "Agent: %s" % (tool_input.get("description", "") or "")[:150]
    elif tool_name == "WebSearch":
        return "WebSearch: %s" % (tool_input.get("query", "") or "")[:150]
    elif tool_name == "WebFetch":
        return "WebFetch: %s" % (tool_input.get("url", "") or "")[:150]
    else:
        return "%s: %s" % (tool_name, json.dumps(tool_input)[:150])


def _read_current_stop(session_id):
    """Read current stop counter from tmp file written by recall hook."""
    try:
        # BRAIN_TMP_DIR env protocol — must match the daemon-side WRITER
        # (servers.daemon_config.brain_tmp_dir(); default /tmp). This hook is a
        # separate process, so it honors the same env var to locate the file;
        # otherwise a relocated tmp root would silently break stop-chain linkage.
        path = os.path.join(os.environ.get('BRAIN_TMP_DIR', '/tmp'),
                            "brain-%s-current-stop.txt" % session_id)
        if os.path.exists(path):
            with open(path) as f:
                return f.read().strip()
    except Exception:
        pass
    return "0"


def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            return
        data = json.loads(raw)
    except Exception:
        return

    tool_name = data.get("tool_name", "")
    # A brain tool's input arrives REWRITTEN with the caller-identity pair; a
    # recorded signature is a replayable identity, so it never enters a trace.
    tool_input = strip_caller_stamp(data.get("tool_input", {}))
    session_id = data.get("session_id", "")

    summary = _build_summary(tool_name, tool_input)
    stop = _read_current_stop(session_id)

    # Attach to current stop's S0 chain
    chain_id = "s0-%s-%s" % (session_id[:8], stop)

    packet = {
        "cmd": "trace_append",
        "args": {
            "chain_id": chain_id,
            "scale": "s0",
            "event_type": "delta",
            "ref_type": "tool_result",
            "summary": summary[:500],
            "metadata": json.dumps(_raw_metadata(data, tool_name, tool_input)),
            "session_id": session_id,
        }
    }
    # A wire-only argument fact: the daemon classifies the tool and decides
    # capture policy. The 200-character display cue is insufficient to see a
    # Git invocation late in a mixed call. Never retain this as trace metadata.
    command = tool_input.get('command')
    if isinstance(command, str):
        packet['args']['tool_command'] = command
    msg = json.dumps(packet)
    if len(msg.encode('utf-8')) > 900_000:
        # Stay below the daemon's 1 MiB transport ceiling. Don't pretend the
        # cropped display summary is the full command: retain a loud diagnostic.
        packet['args'].pop('tool_command', None)
        packet['args']['tool_command_omitted'] = True
        msg = json.dumps(packet)

    # DELIBERATELY hand-rolled, NOT routed through daemon_client.send_command
    # like every other client. The reason is COST, on the hottest path in the
    # system: importing servers.daemon_client measures ~44ms, of which ~22ms is
    # daemon_config md5-walking every servers/*.py to build the code
    # fingerprint at import time. This hook fires on EVERY Bash/Edit/Read/Grep/
    # MCP call — roughly 2500/day, so routing it through the owner would burn
    # 100+ seconds a day to send one fire-and-forget line.
    # servers/daemon_config.py records the same decoupling for the port read.
    # If that import-time fingerprint ever goes lazy (DISTRIBUTION-ARCH-PLAN
    # step 9), this exception stops being worth its keep. Until then: keep this
    # socket in step with send_command by hand; tests/test_deploy_contract.py
    # names it a declared owner so the divergence is visible, not accidental,
    # and tests/test_daemon_instance.py pins its host to DAEMON_HOST.
    port = int(os.environ.get("BRAIN_DAEMON_PORT") or (47200 + os.getuid() % 100))  # env (brain-env.sh) is the live source; formula is the fallback
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("127.0.0.1", port))
        sock.sendall((msg + "\n").encode())
        sock.recv(4096)
        sock.close()
    except Exception:
        pass


run_hook("post_tool_trace", main)
