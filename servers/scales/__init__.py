"""Scales — the GRAIN axis of the fractal. Owner of the placement rule below.

Belongs here: a unit running an `integrate(O, K) → Δ` loop (`s1/`, `s2/`), and
the shared machinery that serves that work (`dispatch.py`, `journal.py`,
`runner.py`). Those three run no loop themselves, and `dispatch.py` is also
reached by the write boundary and `brain.py` — living here does not mean
serving only scales.

Does NOT belong here: a package whose DOMAIN clock is real-elapsed — delivery
windows, TTLs, roster staleness. (Perf timers and `created_at` stamps are
bookkeeping, not domain time; they are fine here and both contracts already
exempt them.) Such a package is a correspondent addressing a live stream, not
a grain: it goes in `servers/channels/`, the way `self_channel` and `thalamus`
do.

That boundary is load-bearing, not taxonomic. Both clock contracts
(`tests/test_clock_contract_sync.py`, `tests/test_time_window_contract.py`)
scan this tree by PREFIX, so every future grain is guarded without anyone
remembering to list it — and a prefix only stays honest while nothing here
needs an exemption. When those two packages did sit here, the pair of scans
would have flagged fifteen lines. Moving the package is the fix; growing an
exemption list is not.

Shape and rationale: docs/ARCHITECTURE-FRACTAL.md
"""
