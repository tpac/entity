"""Shared trace-field constants and metadata extractors used by multiple
query modules.

Leading underscore by convention: these are package-internal helpers, not
public API. Nothing in `dashboard.server` should import from here directly —
go through the public `queries.X` modules that wrap them with their own
shaping.
"""

import json


# The S0 session stamp — mirrors trace_contract.S0_SESSION_STAMP_FIELDS (the
# dashboard may not import servers.*; tests/test_s0_session_stamp.py holds the
# two literals in step). Which model produced the turn and which host runtime
# the session rides on; per-turn on S0 rows, latest on the session row.
S0_SESSION_STAMP_FIELDS = ('model', 'host')

# `ref_id` of a surface_selected K row that picked nothing (the surfacer
# writes json.dumps of its selected ids).
EMPTY_SELECTION_REF_ID = '[]'

# Opens a harness wake envelope (a background-task completion delivered as a
# user_message) — mirrors trace_contract.WAKE_ENVELOPE_MARKER, pinned by
# tests/test_dashboard_disconnection.py.
WAKE_ENVELOPE_MARKER = '<task-notification>'

# Metadata key, True on a user_message row the recall hook registered without
# recall or surface — mirrors trace_contract.REGISTER_ONLY_FIELD, pinned by
# tests/test_dashboard_disconnection.py.
REGISTER_ONLY_FIELD = 'register_only'


def extract_meta_fields(metadata_raw, *keys) -> tuple:
    """Pull `keys` out of a trace's metadata JSON as a tuple of strings, in
    order. '' for every key on any failure (no metadata, invalid JSON, key
    absent) — the dashboard treats absent as "unknown", never as an error."""
    empty = tuple('' for _ in keys)
    if not metadata_raw:
        return empty
    try:
        d = json.loads(metadata_raw)
    except (ValueError, TypeError):
        return empty
    if not isinstance(d, dict):
        return empty
    return tuple((d.get(k, '') or '') for k in keys)
