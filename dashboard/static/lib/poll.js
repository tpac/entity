// ===========================================================================
// lib/poll.js — single polling scheduler.
// ---------------------------------------------------------------------------
// Pattern: one 1Hz tick walks a registry. Each entry says "fire fn() every
// `interval` ms when the activeWhen() predicate is true and document is
// visible". On fire, the function gets awaited; concurrent firings of the
// same key are coalesced via a single in-flight Promise per key.
//
// Inspired by SWR / TanStack Query (`refreshInterval` + dedupe +
// visibility gating). No framework — just a registry + setInterval(1000).
//
// One setInterval at 1000ms; entries self-gate; document.hidden = nothing
// fires; tab switch fires the new tab immediately.
//
// Usage:
//
//   import { poll } from './lib/poll.js';
//
//   poll.register({
//     key: 'recalls',           // unique id (dedupe in-flight)
//     interval: 2000,           // min ms between fires
//     activeWhen: () => true,   // gating predicate (e.g. tab visible)
//     fetcher: async () => {
//       const body = await api.recalls({ limit: 20 });  // raw response body
//       bus.publish('recalls:batch', body);
//     },
//   });
//
//   poll.unregister('recalls')  // stop firing
//   poll.fireNow('recalls')     // ad-hoc immediate fire (used on tab switch)
// ===========================================================================

const _entries = new Map();   // key → { interval, activeWhen, fetcher, lastFiredAt, inflight }
let _ticking = false;

const TICK_MS = 1000;

function _shouldFire(entry, nowMs) {
  if (document.hidden) return false;
  if (entry.inflight) return false;
  if (entry.activeWhen && !entry.activeWhen()) return false;
  return (nowMs - entry.lastFiredAt) >= entry.interval;
}

async function _fire(key, entry) {
  entry.lastFiredAt = Date.now();
  entry.inflight = (async () => {
    try {
      await entry.fetcher();
    } catch (e) {
      console.error('[poll] fetcher "' + key + '" threw:', e);
    } finally {
      entry.inflight = null;
    }
  })();
  return entry.inflight;
}

function _tick() {
  const now = Date.now();
  for (const [key, entry] of _entries) {
    if (_shouldFire(entry, now)) {
      _fire(key, entry);
    }
  }
}

function _startTicking() {
  if (_ticking) return;
  _ticking = true;
  setInterval(_tick, TICK_MS);
  // Re-fire visible entries when the tab regains focus — covers the case
  // where document.hidden suppressed firing for minutes and the user wants
  // fresh data the moment they look at the window.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) _tick();
  });
}

export const poll = {
  register({ key, interval, activeWhen, fetcher }) {
    if (!key || !fetcher) throw new Error('poll.register: key + fetcher required');
    _entries.set(key, {
      interval: interval || 5000,
      activeWhen: activeWhen || (() => true),
      fetcher,
      lastFiredAt: 0,
      inflight: null,
    });
    _startTicking();
    // Fire immediately if conditions allow — first-load shouldn't wait
    // up to `interval` ms before the panel populates.
    const entry = _entries.get(key);
    if (_shouldFire(entry, Date.now())) _fire(key, entry);
  },

  unregister(key) {
    _entries.delete(key);
  },

  // Trigger a key's fetcher right now, regardless of interval. Used on
  // tab-switch ("show me fresh data") and user-driven refresh buttons.
  fireNow(key) {
    const entry = _entries.get(key);
    if (entry && !entry.inflight) return _fire(key, entry);
  },

  // Debug helper — inspect current schedule from devtools.
  _debug_entries() {
    const out = {};
    for (const [key, e] of _entries) {
      out[key] = {
        interval: e.interval,
        lastFiredAt: e.lastFiredAt,
        msSince: Date.now() - e.lastFiredAt,
        inflight: !!e.inflight,
        active: e.activeWhen(),
      };
    }
    return out;
  },
};

export default poll;
