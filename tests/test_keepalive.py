"""Anthropic-connection keepalive: gating, warm primitive, and the tick.

The keepalive re-warms the surface Haiku connection during idle so the first
recall after a quiet period doesn't pay a cold-TLS tax (idle inflates
surface_haiku ~6s->~10s). Units under test:

  - BrainDaemon._keepalive_due       — pure gating decision (no clock, no I/O)
  - BrainDaemon._keepalive_tick      — one loop decision (gating + backoff)
  - Brain.warm_anthropic_connection  — the free models.retrieve warm. Builds the
                                        client if absent (self-heal), raises on
                                        error (callers own the policy), and is
                                        idempotent under concurrency (non-blocking
                                        lock: never two warms at once).

Exercised without a real Brain/daemon (no DB, socket, or embedder) by calling
the static / duck-typed methods directly.
"""
import threading

from servers.daemon_server import BrainDaemon
from servers.brain import Brain


# ── gating ──────────────────────────────────────────────────────────────────

def test_keepalive_due_fires_when_idle_and_not_recently_pinged():
    due = BrainDaemon._keepalive_due
    assert due(300, 300, 300) is True        # idle a full interval, ping due
    assert due(600, 600, 300) is True        # long idle, ping overdue
    assert due(300.0, 300.0, 300.0) is True  # exact boundary fires


def test_keepalive_due_skips_active_and_recently_pinged():
    due = BrainDaemon._keepalive_due
    assert due(120, 9999, 300) is False  # active session (recent recall) -> skip
    assert due(9999, 120, 300) is False  # idle but pinged <1 interval ago -> skip
    assert due(0, 0, 300) is False       # just-now activity -> skip


# ── warm primitive ───────────────────────────────────────────────────────────

class _StubModels:
    def __init__(self):
        self.calls = []

    def retrieve(self, model):
        self.calls.append(model)
        return {"id": model}


class _StubClient:
    def __init__(self):
        self.models = _StubModels()


class _StubBrain:
    """Duck-typed stand-in: only the attributes warm_anthropic_connection touches.

    `_ensure_anthropic_client` returns the preset client (no construction) — the
    real construction path is covered separately by the self-heal test below."""
    def __init__(self, client):
        self.anthropic_client = client
        self._anthropic_warm_lock = threading.Lock()

    def _ensure_anthropic_client(self):
        return self.anthropic_client


def test_warm_calls_models_retrieve_once():
    client = _StubClient()
    brain = _StubBrain(client)
    assert Brain.warm_anthropic_connection(brain) is True
    assert client.models.calls == ['claude-haiku-4-5']  # one free retrieve, right model


def test_warm_self_heals_missing_client():
    """If the client was never built (boot-warmup failure left it None), warm
    builds it via _ensure_anthropic_client rather than no-op'ing."""
    built = []

    class _HealBrain:
        def __init__(self):
            self.anthropic_client = None
            self._anthropic_warm_lock = threading.Lock()

        def _ensure_anthropic_client(self):
            if self.anthropic_client is None:
                self.anthropic_client = _StubClient()
                built.append(1)
            return self.anthropic_client

    b = _HealBrain()
    assert Brain.warm_anthropic_connection(b) is True
    assert built == [1]                      # constructed the missing client
    assert b.anthropic_client.models.calls   # then warmed it


def test_warm_raises_on_api_error():
    import pytest

    class _Boom:
        @property
        def models(self):
            raise RuntimeError("network down")

    brain = _StubBrain(_Boom())
    # The primitive does NOT swallow: warm_up() and the keepalive loop each wrap
    # it and apply their own failure policy. Daemon-crash safety lives in those
    # callers' try/except, not in this primitive.
    with pytest.raises(RuntimeError):
        Brain.warm_anthropic_connection(brain)


def test_warm_releases_lock_after_error():
    """A raised error must still release the lock (finally) so the next tick can
    warm again rather than skipping forever."""
    import pytest

    class _Boom:
        @property
        def models(self):
            raise RuntimeError("network down")

    brain = _StubBrain(_Boom())
    with pytest.raises(RuntimeError):
        Brain.warm_anthropic_connection(brain)
    # Lock is free again -> a subsequent call proceeds (and raises again).
    assert brain._anthropic_warm_lock.acquire(blocking=False) is True
    brain._anthropic_warm_lock.release()


def test_warm_skips_when_already_in_flight():
    """If a warm is already running (lock held), a concurrent caller skips
    rather than firing a redundant models.retrieve — the idempotence guard."""
    client = _StubClient()
    brain = _StubBrain(client)
    brain._anthropic_warm_lock.acquire()  # simulate a warm in progress
    try:
        assert Brain.warm_anthropic_connection(brain) is False
        assert client.models.calls == []  # did NOT double-warm
    finally:
        brain._anthropic_warm_lock.release()


# ── keepalive tick (the per-loop decision, extracted so it's testable without
#    driving the thread) ──────────────────────────────────────────────────────

class _StubDaemonBrain:
    """Minimal brain for tick tests: get_config + warm + _log_error."""
    def __init__(self, *, enabled=True, interval=300, warm=None,
                 llm_available=True):
        from servers.activity_state import ActivityState
        self._cfg = {'surface_keepalive.enabled': enabled,
                     'surface_keepalive.interval_seconds': interval}
        self.warm_calls = 0
        self._warm = warm  # optional callable to simulate a raising warm
        self.logged = []
        # Keyed by default — the tick gates the warm on llm_available
        # (keyless onboarding); keyless ticks are tested explicitly.
        self.llm_available = llm_available
        self.unavailable_notes = []
        # Keepalive reads the user-activity clock from brain.activity now.
        self.activity = ActivityState()

    def note_llm_unavailable(self, where):
        self.unavailable_notes.append(where)

    def get_config(self, key, default):
        return self._cfg.get(key, default)

    def warm_anthropic_connection(self):
        self.warm_calls += 1
        if self._warm is not None:
            return self._warm()
        return True

    def _log_error(self, *a):
        self.logged.append(a)


class _StubDaemon:
    """Duck-typed daemon exposing only what _keepalive_tick reads."""
    _keepalive_due = staticmethod(BrainDaemon._keepalive_due)

    def __init__(self, brain, last_user_activity=0.0):
        self.brain = brain
        # Keepalive reads brain.activity.last_user_activity (single source).
        self.brain.activity.last_user_activity = last_user_activity

    def tick(self, now, last_ping):
        return BrainDaemon._keepalive_tick(self, now, last_ping)


def test_tick_warms_when_idle_and_advances_last_ping():
    brain = _StubDaemonBrain()
    d = _StubDaemon(brain, last_user_activity=0.0)   # huge idle
    new_lp = d.tick(now=1000.0, last_ping=0.0)        # since_last_ping=1000 >= 300
    assert brain.warm_calls == 1
    assert new_lp == 1000.0  # last_ping advanced


def test_tick_keyless_skips_warm_notes_once_and_keeps_cadence():
    """Keyless onboarding: the warm can only 401 — the tick must skip it,
    note the state (once, via note_llm_unavailable), and still advance
    last_ping so the check stays on the interval cadence."""
    brain = _StubDaemonBrain(llm_available=False)
    d = _StubDaemon(brain, last_user_activity=0.0)
    new_lp = d.tick(now=1000.0, last_ping=0.0)
    assert brain.warm_calls == 0                      # no doomed API attempt
    assert brain.unavailable_notes == ['keepalive warm']
    assert brain.logged == []                         # no error-table spam
    assert new_lp == 1000.0                           # cadence preserved


def test_tick_backs_off_after_raising_warm():
    """A raising warm must STILL advance last_ping, so the next tick within the
    interval does not re-fire every cadence."""
    def boom():
        raise RuntimeError("api down")

    brain = _StubDaemonBrain(warm=boom)
    d = _StubDaemon(brain, last_user_activity=0.0)
    new_lp = d.tick(now=1000.0, last_ping=0.0)
    assert brain.warm_calls == 1
    assert new_lp == 1000.0   # advanced despite the raise -> backs off
    assert brain.logged       # error logged, not propagated
    # A tick one cadence (30s) later is NOT due -> no second warm.
    d.tick(now=1030.0, last_ping=new_lp)
    assert brain.warm_calls == 1


def test_tick_disabled_does_not_warm():
    brain = _StubDaemonBrain(enabled=False)
    d = _StubDaemon(brain, last_user_activity=0.0)
    assert d.tick(now=1000.0, last_ping=0.0) == 0.0  # last_ping unchanged
    assert brain.warm_calls == 0


def test_tick_bad_interval_falls_back_to_default_not_disabled():
    """A non-numeric interval must not raise or disable the loop — it falls back
    to the 300s default and still warms when idle."""
    brain = _StubDaemonBrain(interval="5min")
    d = _StubDaemon(brain, last_user_activity=0.0)
    new_lp = d.tick(now=1000.0, last_ping=0.0)
    assert brain.warm_calls == 1   # default 300 applied; idle 1000 >= 300
    assert new_lp == 1000.0
    assert not brain.logged        # handled internally, no tick error


def test_tick_not_due_when_recently_pinged():
    brain = _StubDaemonBrain()
    d = _StubDaemon(brain, last_user_activity=0.0)
    new_lp = d.tick(now=1000.0, last_ping=900.0)  # since_last_ping=100 < 300
    assert brain.warm_calls == 0
    assert new_lp == 900.0


def test_tick_non_positive_interval_disables():
    brain = _StubDaemonBrain(interval=0)
    d = _StubDaemon(brain, last_user_activity=0.0)
    assert d.tick(now=1000.0, last_ping=0.0) == 0.0
    assert brain.warm_calls == 0
