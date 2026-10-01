"""Memory watchdog — periodic RSS sampling.

Why this exists: the daemon process has leaked twice now (2026-04-26 grew
to 4.6 GB after ~4h uptime). When that happens we want diagnostics already
in place — not a scramble to bolt-on profiling against a process that's
already swelling.

Scope, deliberately narrow: RSS + thread-count sampling and nothing else.
A *watchdog* must be cheap. If turning it on slows the thing it observes,
it stops being a watchdog and starts being the bug: an in-process
allocation profiler (tracemalloc) makes every recall take minutes, because
the recall hot path is allocation-heavy. For allocation-level profiling,
run a one-shot diagnostic script or attach `py-spy`/`lldb` to the live
daemon.

Config keys (read at thread start; toggling at runtime requires a
daemon restart):

  memory_watchdog.enabled            bool   default False
  memory_watchdog.interval_seconds   int    default 60     (RSS sampling)

The watchdog is a `daemon=True` background thread — it never blocks
shutdown. All errors are caught + logged via brain._log_error so a
sampling failure can never crash the daemon.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional


def _rss_bytes() -> int:
    """Resident set size in bytes for the current process.

    Reads from /proc on Linux, ps on macOS. Returns 0 on failure rather
    than raising — a sampling failure must never affect the daemon.
    """
    # Linux fast path
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    kb = int(line.split()[1])
                    return kb * 1024
    except FileNotFoundError:
        pass  # macOS / non-Linux
    except Exception:
        pass
    # macOS / fallback: ps
    try:
        import subprocess
        out = subprocess.check_output(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            timeout=2.0,
        ).decode().strip()
        return int(out) * 1024
    except Exception:
        return 0


def _human(n_bytes: int) -> str:
    if n_bytes < 1024:
        return '%dB' % n_bytes
    if n_bytes < 1024 * 1024:
        return '%.1fKB' % (n_bytes / 1024)
    if n_bytes < 1024 ** 3:
        return '%.1fMB' % (n_bytes / 1024 / 1024)
    return '%.2fGB' % (n_bytes / 1024 ** 3)


def _signed_human(delta_bytes: int) -> str:
    sign = '+' if delta_bytes >= 0 else '-'
    return sign + _human(abs(delta_bytes))


class MemoryWatchdog:
    """Background memory sampler. Spawn one per daemon at most."""

    def __init__(self, brain, log_fn=None):
        self.brain = brain
        self.log_fn = log_fn or (lambda msg: print('[mem] %s' % msg, flush=True))
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._last_rss = 0

        # Snapshot config at start — runtime toggle requires restart.
        self.interval = max(10, int(brain.get_config(
            'memory_watchdog.interval_seconds', 60)))

    @classmethod
    def maybe_start(cls, brain, log_fn=None) -> Optional['MemoryWatchdog']:
        """Start the watchdog iff `memory_watchdog.enabled` is True.

        Returns the running watchdog or None if disabled. Idempotent:
        second call from same daemon returns None (the existing
        watchdog stays). Safe to call from daemon startup unconditionally.
        """
        if not brain.get_config('memory_watchdog.enabled', False):
            return None
        wd = cls(brain, log_fn=log_fn)
        wd.start()
        return wd

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='memory-watchdog')
        self._thread.start()
        self._last_rss = _rss_bytes()
        self.log_fn('started — interval=%ds, baseline RSS=%s' % (
            self.interval, _human(self._last_rss)))

    def stop(self):
        """Stop the watchdog."""
        self.running = False

    def _loop(self):
        while self.running:
            try:
                time.sleep(self.interval)
                self._sample_rss()
            except Exception as e:
                # Sampling must never crash the daemon. Log and continue.
                try:
                    self.brain._log_error(
                        'memory_watchdog_loop', e,
                        'sampling failed; loop continues')
                except Exception:
                    pass

    def _sample_rss(self):
        rss = _rss_bytes()
        if rss == 0:
            return  # couldn't read, skip
        delta = rss - self._last_rss
        self._last_rss = rss
        threads = threading.active_count()
        # Always log absolute; flag growth above 50MB since last sample.
        flag = ' ⚠ growth' if delta > 50 * 1024 * 1024 else ''
        self.log_fn('rss=%s (%s vs prev) threads=%d%s' % (
            _human(rss), _signed_human(delta), threads, flag))
