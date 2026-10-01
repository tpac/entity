"""Database backup policy — the ONE owner of every backup the brain takes.

Backend-agnostic. The consistent-snapshot mechanism (online backup API +
gzip) lives in the active db_backend (`snapshot_to` / `backup_snapshot`);
this module owns the backend-independent policy: destination naming,
listing/parsing the backup set, which snapshots to keep (GFS retention),
and the two pre-destructive-operation entry points every destructive
caller routes through:

- `backup_before_destructive(db_path, tag)` — a named, durable
  `{db}.{tag}.bak.gz` taken immediately before a specific rewrite
  (schema migration, table rebuild, destructive script). Idempotent per
  tag so a retried operation doesn't overwrite the pre-first-attempt
  state.
- `ensure_backup_fresh(db_path)` — the freshness gate for automatic
  destructive maintenance (idle sweeps): guarantees a destructive op
  never runs more than `max_age_s` away from a restorable rolling
  snapshot, snapshotting first when the newest one is stale.

The AUTOMATIC rolling backups (`backup_database`, daily scheduler) write
`{basename}.{timestamp}.gz` into the backup dir; retention only ever
touches files matching that scheme, so tagged `.bak.gz` files and
hand-made `.bak*` files are never listed, kept, or pruned by it.

Retention is GFS:
- `keep_daily`   — newest snapshot of each of the most recent N days
- `keep_weekly`  — newest snapshot of each of the most recent N ISO weeks
- `keep_monthly` — newest snapshot of each of the most recent N months
A snapshot survives if any tier selects it (union). At daily cadence this
holds ~keep_daily + keep_weekly + keep_monthly files at steady state.

Tagged `.bak[.gz]` files get TTL retention instead of GFS: a
pre-destructive backup is a verification net, not an archive — its value
decays with time since the operation it guarded, not with snapshot
density (an unreaped one measured 888MB). `reap_by_ttl` is the generic
age-gated primitive (other retention policies can compose it);
`backup_database`'s daily rotation feeds it the tagged set beside the DB.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

# Filename scheme: `{db_basename}.{YYYYMMDDTHHMMSSZ}.gz`
# e.g. `brain.db.20260625T151500Z.gz`. The timestamp is UTC, sortable,
# and round-trips back to a datetime for GFS bucketing.
_TS_FMT = '%Y%m%dT%H%M%SZ'
_TS_RE = re.compile(r'\.(\d{8}T\d{6}Z)\.gz$')

_DEFAULT_KEEP = {'daily': 7, 'weekly': 4, 'monthly': 3}

# Rolling-snapshot cadence. Owned here (backup policy); the maintenance
# scheduler imports it rather than carrying its own copy, so the freshness
# gate below can never drift out of proportion with the real cadence.
BACKUP_INTERVAL_S = 24 * 60 * 60

# Freshness gate ceiling: 1.5× the snapshot cadence, so the gate is a
# no-op whenever the scheduler is healthy and only snapshots inline when the
# daemon has been down past a full cycle (fresh install, multi-day outage).
_DEFAULT_MAX_AGE_S = int(1.5 * BACKUP_INTERVAL_S)

# Tagged pre-destructive backups (`{db}.{tag}.bak[.gz]`) live this long
# after their mtime, then the daily rotation reaps them — long enough that
# the operation the backup guards has been verified in real use. A retried
# operation past the TTL takes a fresh backup (backup_before_destructive's
# idempotent reuse protects attempts, not archaeology).
TAGGED_BAK_TTL_DAYS = 14

# Matched against the REMAINDER after `{db_basename}.` — a non-empty tag
# segment then `.bak[.gz]`. Anchoring on the remainder (not searching the
# whole name) is what keeps a hand-made `cp brain.db brain.db.bak` out of
# the set: its remainder is bare `bak`, no tag segment. Also can't match
# the DB itself, -wal/-shm siblings, or rolling `.{ts}.gz` snapshots.
_TAGGED_BAK_RE = re.compile(r'^[A-Za-z0-9_.-]+\.bak(\.gz)?$')


def default_backup_dir(db_path: str) -> str:
    """The rolling-backup directory for a DB: `backups/` beside the file.
    One definition so the scheduler wiring and the freshness gate can never
    look in two different places."""
    return os.path.join(os.path.dirname(db_path), 'backups')


def _dest_path(db_path: str, backup_dir: str, ts: datetime) -> str:
    base = os.path.basename(db_path)
    return os.path.join(backup_dir, '%s.%s.gz' % (base, ts.strftime(_TS_FMT)))


def list_backups(db_path: str, backup_dir: str) -> List[Tuple[datetime, str]]:
    """All auto-backups for this DB, newest first: [(timestamp, path)].

    Matches only `{basename}.{ts}.gz` — hand-made `.bak*` files don't have
    a parseable timestamp segment and are skipped, so they're invisible to
    retention."""
    base = os.path.basename(db_path)
    prefix = base + '.'
    out: List[Tuple[datetime, str]] = []
    try:
        names = os.listdir(backup_dir)
    except OSError:
        return out
    for name in names:
        if not name.startswith(prefix):
            continue
        m = _TS_RE.search(name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), _TS_FMT).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append((ts, os.path.join(backup_dir, name)))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def select_retained(timestamps: List[datetime],
                    keep_daily: int, keep_weekly: int,
                    keep_monthly: int) -> Set[datetime]:
    """GFS retained set. For each tier, keep the newest snapshot of each
    of the most recent N periods. A snapshot is retained if any tier
    selects it."""
    retained: Set[datetime] = set()
    ordered = sorted(timestamps, reverse=True)   # sort once, reuse per tier

    def pick(key_fn, keep_n: int) -> None:
        if keep_n <= 0:
            return
        newest_per_period: Dict = {}
        for ts in ordered:
            k = key_fn(ts)
            if k not in newest_per_period:   # newest first → first wins
                newest_per_period[k] = ts
        for k in sorted(newest_per_period, reverse=True)[:keep_n]:
            retained.add(newest_per_period[k])

    pick(lambda ts: ts.date(), keep_daily)
    pick(lambda ts: ts.isocalendar()[:2], keep_weekly)   # (iso_year, iso_week)
    pick(lambda ts: (ts.year, ts.month), keep_monthly)
    return retained


def prune(db_path: str, backup_dir: str,
          keep_daily: int, keep_weekly: int, keep_monthly: int) -> Dict:
    """Delete auto-backups not selected by the GFS retained set."""
    backups = list_backups(db_path, backup_dir)
    if not backups:
        return {'kept': 0, 'deleted': 0}
    retained = select_retained([ts for ts, _ in backups],
                               keep_daily, keep_weekly, keep_monthly)
    # Never delete the newest snapshot, whatever the policy — guards against
    # an all-zero keep config silently wiping the just-created backup.
    retained.add(backups[0][0])   # backups is newest-first
    deleted = 0
    for ts, path in backups:
        if ts not in retained:
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                pass
    return {'kept': len(retained), 'deleted': deleted}


def list_tagged_backups(db_path: str) -> List[Tuple[float, str]]:
    """Tagged pre-destructive backups beside the DB, newest first:
    [(mtime_epoch, path)] for every `{db_basename}.{tag}.bak[.gz]` in the
    DB's own directory. The rolling backups/ subdir is not scanned — its
    files belong to GFS retention, not TTL."""
    d = os.path.dirname(db_path) or '.'
    prefix = os.path.basename(db_path) + '.'
    out: List[Tuple[float, str]] = []
    try:
        names = os.listdir(d)
    except OSError as e:
        # Loud: an unreadable DB directory must not read as "no tagged
        # backups" — the caller would silently skip retention forever.
        print('[brain] list_tagged_backups FAILED for %s: %s' % (d, e),
              flush=True)
        return out
    for name in names:
        if not (name.startswith(prefix)
                and _TAGGED_BAK_RE.match(name[len(prefix):])):
            continue
        path = os.path.join(d, name)
        try:
            out.append((os.path.getmtime(path), path))
        except OSError as e:
            print('[brain] list_tagged_backups: cannot stat %s: %s'
                  % (path, e), flush=True)
            continue
    out.sort(reverse=True)
    return out


def reap_by_ttl(files: List[str], ttl_days: float,
                now: Optional[datetime] = None) -> Dict:
    """Generic age-gated reaper: delete the given files whose mtime is older
    than `ttl_days`. The TTL-retention primitive other policies compose —
    deletion is age-gated only; the CALLER owns the listing and therefore
    the blast radius. Returns {'reaped': [basenames], 'kept': n}."""
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - ttl_days * 86400
    reaped: List[str] = []
    failed: List[str] = []
    kept = 0
    for path in files:
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                reaped.append(os.path.basename(path))
            else:
                kept += 1
        except OSError as e:
            # Loud: a reap that cannot delete must not report clean success —
            # the disk-bloat pathology this exists to fix would silently
            # return (module convention: failures print, e.g. 'Backup FAILED').
            failed.append(os.path.basename(path))
            print('[brain] Reap FAILED for %s: %s' % (path, e), flush=True)
    return {'reaped': reaped, 'kept': kept, 'failed': failed}


def seconds_since_last_backup(db_path: str, backup_dir: str,
                              now: Optional[datetime] = None) -> float:
    """Age of the newest auto-backup, in seconds. `inf` if none exist.

    Lets the scheduler seed its clock from the filesystem so a daemon
    restart doesn't re-snapshot a DB it backed up minutes ago — no extra
    persisted state needed."""
    backups = list_backups(db_path, backup_dir)
    if not backups:
        return float('inf')
    if now is None:
        now = datetime.now(timezone.utc)
    return max(0.0, (now - backups[0][0]).total_seconds())


def backup_database(db_path: str, backup_dir: str, *,
                    keep_daily: int = _DEFAULT_KEEP['daily'],
                    keep_weekly: int = _DEFAULT_KEEP['weekly'],
                    keep_monthly: int = _DEFAULT_KEEP['monthly'],
                    now: Optional[datetime] = None) -> Dict:
    """Snapshot `db_path` into `backup_dir` (compressed), then prune to the
    GFS retained set. Returns snapshot diagnostics + kept/deleted counts."""
    if now is None:
        now = datetime.now(timezone.utc)
    os.makedirs(backup_dir, exist_ok=True)
    dest = _dest_path(db_path, backup_dir, now)

    from . import db_backends
    result = dict(db_backends.current.backup_snapshot(db_path, dest))

    result['dest'] = os.path.basename(dest)
    result.update(prune(db_path, backup_dir, keep_daily, keep_weekly, keep_monthly))
    # TTL pass over tagged pre-destructive backups beside the DB — rides the
    # same daily rotation so no second scheduler entry exists to drift.
    # Operator-ruled 2026-08-30 with the consumers known: the recovery
    # scripts that globbed this set served the closed spring incident
    # class; absorbed_into pointers + the canonical-pull redirect made
    # consolidation traceable without bak archaeology.
    tagged = reap_by_ttl([p for _, p in list_tagged_backups(db_path)],
                         TAGGED_BAK_TTL_DAYS, now=now)
    if tagged['reaped']:
        result['tagged_reaped'] = tagged['reaped']
        print('[brain] Reaped %d tagged backup(s) past %dd TTL: %s'
              % (len(tagged['reaped']), TAGGED_BAK_TTL_DAYS,
                 ', '.join(tagged['reaped'])), flush=True)
    return result


def backup_before_destructive(db_path: str, tag: str,
                              compress: bool = True) -> Optional[str]:
    """Named pre-destructive backup beside the DB: `{db_path}.{tag}.bak.gz`,
    or `.bak` (raw) with `compress=False`.

    The one entry point for "this code is about to rewrite the DB": schema
    migrations, table rebuilds, destructive scripts. Consistent even while
    the daemon is live and independent of any caller connection's
    transaction state (online backup API on its own read-only connection).

    `compress=False` is for time-budgeted callers: the daemon-boot
    migration path runs before the daemon answers pings, and the MCP
    health monitor force-restarts an unresponsive daemon after ~20s — a
    raw snapshot of the production brain.db measures ~1s where the
    gzipped one measures ~13s, so boot backups skip compression.

    Idempotent per tag: an existing backup is returned untouched, so a
    retried migration cannot overwrite the pre-first-attempt state. Reuse
    is announced with the file's age — for one-shot script tags a
    months-old leftover is a trap, and the operator must see it. Two
    unserialized processes racing the same tag cannot interleave: each
    builds a uniquely-named temp and the atomic rename wins whole.
    The guarantee holds only while the tag file exists: an explicit
    `reap_by_ttl` that removes it mid-operation makes a later retry
    snapshot the CURRENT (possibly half-mutated) state — reap only tags
    whose operation is verified complete. Nothing reaps automatically.

    Returns the backup path, or None on failure (printed loudly; the caller
    decides whether the operation may proceed without it).
    """
    dest = '%s.%s.bak%s' % (db_path, tag, '.gz' if compress else '')
    if os.path.exists(dest):
        age_days = (datetime.now(timezone.utc).timestamp()
                    - os.path.getmtime(dest)) / 86400.0
        print('[brain] Reusing existing backup (%.1fd old): %s'
              % (age_days, dest), flush=True)
        return dest
    try:
        from . import db_backends
        if compress:
            db_backends.current.backup_snapshot(db_path, dest, compresslevel=1)
        else:
            tmp = '%s.tmp.%s' % (dest, os.urandom(4).hex())
            try:
                db_backends.current.snapshot_to(db_path, tmp)
                os.replace(tmp, dest)
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        print('[brain] Backup created: %s' % dest, flush=True)
        return dest
    except Exception as e:
        print('[brain] Backup FAILED for %s (%s): %s' % (db_path, tag, e),
              flush=True)
        return None


def ensure_backup_fresh(db_path: str, backup_dir: Optional[str] = None,
                        max_age_s: float = _DEFAULT_MAX_AGE_S) -> bool:
    """Freshness gate for automatic destructive maintenance.

    True means: a rolling snapshot newer than `max_age_s` exists (usually
    the scheduler's — this returns without doing anything), or one was just
    taken. False means no restorable snapshot could be guaranteed — the
    caller must skip its destructive work this cycle rather than run it
    with no net.
    """
    if backup_dir is None:
        backup_dir = default_backup_dir(db_path)
    if seconds_since_last_backup(db_path, backup_dir) <= max_age_s:
        return True
    try:
        backup_database(db_path, backup_dir)
        return True
    except Exception as e:
        print('[brain] Freshness-gate backup FAILED for %s: %s'
              % (db_path, e), flush=True)
        return False


def materialize_backup(backup_path: str, work_dir: Optional[str] = None) -> str:
    """A directly-openable .db path for a backup of any shape this module
    writes. Plain `.db`/`.bak` files are returned as-is; `.gz` snapshots are
    decompressed into `work_dir` — defaulting to the system temp dir, NEVER
    beside the source: a full-size sibling per rolling snapshot would fill
    the backups dir with files no retention rule tracks. The name carries a
    hash of the source path so an existing decompression is reused within
    and across runs, and the temp dir's normal purging owns cleanup."""
    if not backup_path.endswith('.gz'):
        return backup_path
    import gzip
    import hashlib
    import shutil
    import tempfile
    base = '%s.%s.materialized' % (
        os.path.basename(backup_path)[:-3],
        hashlib.md5(os.path.abspath(backup_path).encode()).hexdigest()[:8])
    out = os.path.join(work_dir or tempfile.gettempdir(), base)
    if os.path.exists(out):
        return out
    tmp = out + '.part'
    with gzip.open(backup_path, 'rb') as f_in, open(tmp, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    os.replace(tmp, out)
    return out
