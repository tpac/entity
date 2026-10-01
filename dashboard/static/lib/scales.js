// ===========================================================================
// lib/scales.js — single source of truth for scale color codes.
// ---------------------------------------------------------------------------
// The fractal trace system tags every event with a scale (s0/s1/s2/s3/s4).
// Renderers that color-code scales read this one map:
//
//   traces.js      — per-chain accent stripe
//   node_detail.js — per-trace-ref color in the "Encoded from N traces"
//                    section
// ===========================================================================

export const SCALE_COLORS = {
  s0: '#888',
  s1: '#7eb8ff',
  s2: '#ffaa33',
  s3: '#33ff88',
  s4: '#ff66aa',
};
