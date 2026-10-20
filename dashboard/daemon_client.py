"""TCP client for the brain daemon.

Dashboard is a passive observer; this module is the only place that opens a
socket to the daemon. If the daemon is unreachable, callers fall back to
direct read-only SQLite queries via `dashboard.db`.
"""

import json
import os
import socket

DAEMON_HOST = "127.0.0.1"  # = servers.daemon_config.DAEMON_HOST; tests/test_dashboard_disconnection.py pins it
DAEMON_PORT = int(os.environ.get("BRAIN_DAEMON_PORT") or (47200 + os.getuid() % 100))  # env (brain-env.sh) is the live source; formula is the fallback


def daemon_call(cmd: str, args=None, timeout: float = 10) -> dict:
    """Send a command to the daemon and return its whole reply envelope,
    {"ok": True, "result": …} or {"ok": False, "error": …}; a daemon that
    cannot be reached answers {"ok": False, "error": "daemon unreachable"}.
    For the dashboard's operator-facing writes (pause, the API key), whose
    pages show the daemon's own refusal text.

    This is the ONE sanctioned copy of the wire protocol
    (servers.daemon_client.send_command is the owner): the dashboard must keep
    running with servers/ absent or broken, so it imports nothing from it. The
    framing is kept identical to the owner's — read until the newline the
    daemon terminates every reply with, rather than re-parsing the whole
    buffer as JSON after every chunk.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        payload = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        s.sendall(payload.encode("utf-8"))
        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        s.close()
        resp = json.loads(b"".join(chunks).decode("utf-8").strip())
        if isinstance(resp, dict):
            return resp
    except Exception:
        pass
    return {"ok": False, "error": "daemon unreachable"}


def daemon_send(cmd: str, args=None, timeout: float = 10):
    """Send a command to the daemon, return result or None.

    Returns None on any failure (connection refused, timeout, daemon-side
    error). Callers must handle None — the dashboard never crashes because
    the daemon is down. The read-side shape over daemon_call.
    """
    resp = daemon_call(cmd, args, timeout)
    return resp.get("result") if resp.get("ok") else None


def daemon_alive() -> bool:
    """Quick check if daemon is responding."""
    return daemon_send("ping", timeout=3) is not None
