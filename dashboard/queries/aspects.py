"""Aspect taxonomy — reads the aspects + member counts from the live brain.

`aspects_v1.json` is the single source of truth (servers/scales/s2/aspects_v1.json
in the repo as a seed; the live working copy lives at
$BRAIN_DB_DIR/aspects_v1.json). Each aspect declares `node_types` and
`edge_relations` it claims; this query counts how many active nodes/edges
fall into each aspect against brain.db.

We deliberately re-read the JSON each call rather than caching — it's tiny
(~5 KB) and the encoder rewrites it when AspectIntegration runs. Caching
would mean stale counts on the dashboard.
"""

import json
import os
import sys

from ..db import _brain_dir, brain_db_path, ro_connect
from ..log import warn


def _aspects_json_path():
    """Resolve the same file servers/scales/s2/aspect_contract.py uses.

    Disconnection contract forbids importing from servers.*, so we replicate
    the path-resolution rules here:
      - $ASPECTS_JSON_PATH env override wins (matches the brain runtime)
      - otherwise aspects_v1.json in the resolved brain dir (db._brain_dir,
        the D-13 resolution chain)
    """
    explicit = os.environ.get('ASPECTS_JSON_PATH')
    if explicit:
        return explicit
    return os.path.join(_brain_dir(), 'aspects_v1.json')


def _repo_seed_path():
    """Fallback when the user-dir copy doesn't exist yet (fresh brain).

    Architectural note (deliberate exception to the disconnection contract):
    this walks `../../servers/scales/s2/aspects_v1.json` — a path-level
    reach into the brain repo that `test_dashboard_disconnection` cannot
    catch because it scans imports, not filesystem paths.

    Why it's allowed:
      * The dashboard reads ONLY — it never writes the file.
      * The standing principle: "Dashboard inspects existing data — never
        changes core behavior to serve display." The aspect
        taxonomy IS existing data: the brain already reads this exact JSON
        as its config. The dashboard reading the same JSON is consuming
        the SAME single source of truth, not creating a parallel funnel.
      * The alternative — adding a `get_aspects` daemon TCP command — WOULD
        be a core change, exactly the pattern that principle forbids.

    Disposition:
      * On a healthy brain the user-dir copy exists and this path is
        never read (the live config is the single source of truth).
      * On a fresh brain (no user-dir copy yet) this seed path is the
        single source of truth — the same file the brain itself bootstraps
        from. Reading it here mirrors that bootstrap.
      * If the path ever becomes wrong (brain moves the seed), the symptom
        is "(no aspects loaded)" in the UI — degrades gracefully.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(
        here, '..', '..', 'servers', 'scales', 's2', 'aspects_v1.json'))


# The key a member object carries its string under — mirrors
# servers/aspect_store.py MEMBER_KEY (a member is a bare string or an object
# holding the string plus its MEMBER_FIELDS facts).
_MEMBER_KEY = {'node_types': 'type', 'edge_relations': 'verb'}


def _load_taxonomy():
    """The live aspects_v1.json as a dict (the repo seed on a fresh brain);
    None when neither loads."""
    path = _aspects_json_path()
    if not os.path.exists(path):
        path = _repo_seed_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        warn('queries.aspects', 'aspects load %s failed' % path, exc=e)
        return None
    if not isinstance(data, dict):
        warn('queries.aspects', 'aspects %s is not a JSON object' % path)
        return None
    return data


def _aspect_entries(data):
    """(name, entry) per aspect — '_'-prefixed keys are in-file documentation
    (_schema), not aspects."""
    return [(name, entry) for name, entry in data.items()
            if not name.startswith('_') and isinstance(entry, dict)]


def _members(entry, category):
    """The member strings an aspect lists under `category`; a malformed list
    or member is skipped, as the brain's loader skips it."""
    members = entry.get(category) or []
    if not isinstance(members, list):
        return []
    names = [m.get(_MEMBER_KEY[category]) if isinstance(m, dict) else m
             for m in members]
    return [n for n in names if isinstance(n, str) and n]


def primary_aspects():
    """{'node_types': {type: aspect}, 'edge_relations': {verb: aspect}} — the
    primary aspect of each string as the brain reads it (AspectRegistry._adopt):
    the first aspect in file order to list it, except that noise takes every
    string it lists. Empty maps when the taxonomy doesn't load. Pinned against
    the registry by tests/test_dashboard_disconnection.py."""
    data = _load_taxonomy() or {}
    out = {'node_types': {}, 'edge_relations': {}}
    for category, primary in out.items():
        for name, entry in _aspect_entries(data):
            for member in _members(entry, category):
                if name == 'noise':
                    primary[member] = name
                else:
                    primary.setdefault(member, name)
    return out


def query_aspects():
    """Return [{name, meaning, dimension, locked, node_types: [{name, count}],
    edge_relations: [{name, count}], totals: {nodes, edges}}].

    On any load failure (missing file, parse error) returns []. The caller
    renders a "(no aspects loaded)" placeholder rather than crashing.
    """
    data = _load_taxonomy()
    if data is None:
        return []

    # Count members from the live brain DB. One round-trip per query function
    # call — counts on ~5K nodes / ~25K edges is sub-ms in SQLite.
    with ro_connect(brain_db_path()) as conn:
        if conn is None:
            type_counts = {}
            relation_counts = {}
        else:
            try:
                type_counts = dict(conn.execute(
                    "SELECT type, COUNT(*) FROM nodes WHERE archived = 0 GROUP BY type"
                ).fetchall())
                relation_counts = dict(conn.execute(
                    "SELECT relation, COUNT(*) FROM edge_relations WHERE archived = 0 GROUP BY relation"
                ).fetchall())
            except Exception as e:
                print('[dashboard] query_aspects counts failed: %s' % e, file=sys.stderr)
                type_counts = {}
                relation_counts = {}

    out = []
    for name, payload in _aspect_entries(data):
        nt = [{'name': t, 'count': int(type_counts.get(t, 0))}
              for t in _members(payload, 'node_types')]
        er = [{'name': r, 'count': int(relation_counts.get(r, 0))}
              for r in _members(payload, 'edge_relations')]
        # Sort members by count desc so the high-volume ones lead.
        nt.sort(key=lambda x: -x['count'])
        er.sort(key=lambda x: -x['count'])
        out.append({
            'name': name,
            'meaning': (payload.get('meaning') or '')[:600],
            'dimension': payload.get('dimension', ''),
            'locked': bool(payload.get('locked')),
            'node_types': nt,
            'edge_relations': er,
            'totals': {
                'nodes': sum(t['count'] for t in nt),
                'edges': sum(r['count'] for r in er),
            },
        })
    # Sort aspects by total node count desc — the ones doing the most work
    # rise to the top.
    out.sort(key=lambda a: -a['totals']['nodes'])
    return out
