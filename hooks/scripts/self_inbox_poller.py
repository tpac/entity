#!/usr/bin/env python3
"""Self-channel live poller — the event source for /watch (launched by brain-watch).

Polls THIS session's self-inbox via the daemon (read-only peek, never consumes)
and prints one stdout line per NEW message. Run under Claude Code's Monitor tool:
each printed line becomes a notification that starts a turn in the watching
session within ~the poll interval.

Design notes:
- PEEK, not consume. The Stop hook still owns the real consume-once drain
  (signal.drain_inbox). We only detect arrivals. We never write self_delivered.
- High-water `seen` set: peek keeps returning a message until the ignited turn's
  Stop hook consumes it, so we announce each id exactly once.
- Prime on first poll: existing pending mail is marked seen WITHOUT emitting —
  it's delivered by the normal Stop-hook drain. We only announce messages that
  arrive AFTER the listener starts (those are the "events").
- Resilient: a transient daemon hiccup logs to stderr (not an event) and the
  loop continues. stdout is reserved exclusively for message events.
- Adaptive cadence: poll fast (5s) so a live exchange feels real-time, but back
  off (x2 each idle poll, up to 60s) when the channel is quiet so watching a
  silent inbox costs almost nothing. The moment a message lands, snap back to 5s
  — an active back-and-forth stays responsive. A daemon-down/error counts as
  idle (keep backing off, don't hammer a dead socket).

Usage: self_inbox_poller.py [session_id] [fast_seconds=5] [slow_seconds=60] [backoff=2.0]
       session_id falls back to $CLAUDE_CODE_SESSION_ID when omitted.
"""
import os
import sys
import time

# Python puts THIS script's dir (hooks/scripts/) on sys.path[0], not the repo
# root — so `import servers` needs the root added explicitly, regardless of cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def main():
    # session_id: explicit argv[1] wins; else fall back to CLAUDE_CODE_SESSION_ID
    # (the full UUID Claude Code exports per command). Belt-and-suspenders for the
    # brain-watch launcher's own default — keeps the poller usable when invoked
    # directly with no arg from inside a session.
    arg = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    sid = arg or os.environ.get("CLAUDE_CODE_SESSION_ID", "").strip()
    if not sid:
        sys.stderr.write("self_inbox_poller: session_id required "
                         "(pass as arg or set CLAUDE_CODE_SESSION_ID)\n")
        return 2
    fast = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0     # responsive floor
    slow = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0    # idle ceiling
    backoff = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0  # idle multiplier

    from servers import daemon_client

    # Startup banner → stderr (stdout is reserved for the message events the
    # Monitor tool turns into notifications). Surfaces WHICH inbox armed so an
    # operator can confirm the right stream without the poller faking an event.
    sys.stderr.write("self_inbox_poller: watching inbox for %s "
                     "(poll %gs→%gs, x%g backoff)\n" % (sid, fast, slow, backoff))
    sys.stderr.flush()

    seen = set()
    primed = False
    interval = fast
    while True:
        resp = daemon_client.send_command(
            'self_inbox_peek', {'session_id': sid}, timeout=5.0)
        new_found = False
        if resp.get('ok'):
            msgs = (resp.get('result') or {}).get('messages', []) or []
            if not primed:
                seen = {m['id'] for m in msgs}
                primed = True
            else:
                for m in msgs:
                    mid = m['id']
                    if mid in seen:
                        continue
                    seen.add(mid)
                    new_found = True
                    body = (m.get('body') or '').replace('\n', ' / ')
                    if m.get('own'):
                        print("⚡ your own note: %s" % body, flush=True)
                    else:
                        frm = m.get('from') or '????'
                        print("⚡ from %s: %s" % (frm, body), flush=True)
        else:
            sys.stderr.write(
                "self_inbox_poller: peek failed: %s\n" % resp.get('error', '?'))
        # Snap to `fast` the moment a message lands (stay responsive through an
        # active exchange); else back off toward `slow` so a quiet channel costs
        # almost nothing. Errors/daemon-down count as idle.
        interval = fast if new_found else min(slow, interval * backoff)
        time.sleep(interval)


if __name__ == '__main__':
    sys.exit(main())
