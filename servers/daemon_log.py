"""
brain — Daemon Log

The daemon's own log file (daemon_config.get_daemon_log_path): every line
carries a dated stamp, and the file's history is bounded. launchd points the
daemon's stdout and stderr at it (StandardOutPath/StandardErrorPath); the
no-launchd spawn appends them to the same path.

- stamp_std_streams() — installed once by daemon_server.main(), so _log,
  module print()s and tracebacks are all stamped at one layer.
- rotate_if_due() — run by BrainDaemon._rotate_daemon_log on the
  db-maintenance thread.
"""

import os
import sys
import threading
from datetime import datetime, timezone

from .clock import iso_now


class StampedStream:
    """Prefixes every line written through it with an iso_now() stamp — the
    format brain.log and the traces use, so the three line up without zone
    conversion. Writes straight to the fd (faulthandler, native libraries)
    bypass it. Reentrant lock: a signal handler that logs runs on the main
    thread, possibly in the middle of a write."""

    def __init__(self, stream):
        self._stream = stream
        self._lock = threading.RLock()
        self._at_line_start = True

    def write(self, text):
        with self._lock:
            stamp = None
            out = []
            for line in text.splitlines(keepends=True):
                if self._at_line_start:
                    stamp = stamp or "[%s] " % iso_now()
                    out.append(stamp)
                out.append(line)
                self._at_line_start = line.endswith("\n")
            self._stream.write("".join(out))
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def stamp_std_streams():
    """Wrap sys.stdout and sys.stderr — the daemon log — in StampedStream."""
    sys.stdout = StampedStream(sys.stdout)
    sys.stderr = StampedStream(sys.stderr)


def rotate_if_due(path, keep_days, fds=(1, 2), now=None):
    """Rotate the daemon log into path + '.1' once its first line is
    keep_days / 2 old, so the oldest line on disk is at most keep_days old.

    launchd holds the log open on fds 1 and 2, so a rename alone would leave
    the daemon writing into the backup: the fresh file is dup2'd onto them,
    which also carries writes that bypass sys.stderr. A line written before the
    dup2 lands in the backup, after it in the fresh file — none is lost. The
    fresh file is opened and headed before either rename, so a failure there
    leaves the log as it was. Replacing the old backup is the prune. A first
    line with no stamp is the pre-stamp log and counts as due. Leaves the log
    alone when the fds point elsewhere (a daemon run from a terminal or a
    test)."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return {'rotated': False, 'reason': 'no log file'}
    for fd in fds:
        fst = os.fstat(fd)
        if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
            return {'rotated': False, 'reason': 'fd %d is not this log' % fd}
    now = now or datetime.now(timezone.utc)
    with open(path, errors='replace') as f:
        first = f.readline()
    started = None
    if first.startswith('[') and ']' in first:
        try:
            started = datetime.fromisoformat(first[1:first.index(']')])
        except ValueError:
            pass
    if started is not None and started.tzinfo is not None:
        age_days = (now - started).total_seconds() / 86400
        if age_days < keep_days / 2:
            return {'rotated': False, 'age_days': round(age_days, 1)}
    fresh = path + '.new'
    new_fd = os.open(fresh, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o644)
    try:
        os.write(new_fd, ("[%s] [brain-daemon] daemon.log rotated (keep %d days); "
                          "earlier lines are in daemon.log.1\n"
                          % (iso_now(), keep_days)).encode())
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        os.replace(path, path + '.1')
        os.replace(fresh, path)
        for fd in fds:
            os.dup2(new_fd, fd)
    finally:
        os.close(new_fd)
    return {'rotated': True, 'previous_bytes': st.st_size,
            'previous_started': started.isoformat() if started else None}
