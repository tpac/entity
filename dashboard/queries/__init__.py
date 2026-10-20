"""Dashboard query modules — one module per data source.

The dashboard reads-only; its writes (pause/resume, the API key, self-send)
are daemon commands over the socket. Each module here owns one slice of the
brain: aspects, config, encoding, errors, explorer, graph, insights_scanner,
journals, recalls, s2_runs, self_channel, sessions, stats, system, thalamus,
traces; _meta holds the shared metadata extractors. Server-side route handlers in `dashboard.server` import
from this package and never touch SQL directly.

The dashboard never imports servers.* (tests/test_dashboard_disconnection.py),
so this package owns its read-only SQL.
"""
