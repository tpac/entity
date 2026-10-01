// ===========================================================================
// lib/journal.js — the one journal renderer, for every encoder.
// ---------------------------------------------------------------------------
// A journal note is an encoder's residue: what it noticed that its ACTIONS
// don't record — a doubt, a friction, a surprise, a pattern forming. All five
// encoders (S1 Scribe + the four S2 units) write the same shape, so all five
// read the same way here.
//
// Three exports, one visual language:
//   journalRows(notes)     — the note rows, for inline use inside a run card
//   journalChip(notes, …)  — the 📓 header button: count + open-item warning
//   openJournalPeek(…)     — the popover: read the residue WITHOUT expanding
//                            the run
//
// The peek exists because the two are different questions. Expanding a run
// asks "what did this run DO"; the journal asks "what did it THINK". You
// often want the second without paying for the first — an encode card's body
// is dozens of node/edge rows deep.
// ===========================================================================

import { el, relativeTime } from '/static/lib/dom.js';

// Tag → accent. Tags are OPEN text by contract (the encoder invents the word),
// so this is a hint map, not an enum: anything unlisted gets the neutral
// treatment rather than being coerced into a bucket it doesn't belong in.
const TAG_ACCENT = {
  open:      'journal-tag--open',
  friction:  'journal-tag--friction',
  doubt:     'journal-tag--doubt',
  surprise:  'journal-tag--surprise',
  failure:   'journal-tag--failure',
  'dead-end': 'journal-tag--failure',
  reject:    'journal-tag--failure',
  resolved:  'journal-tag--resolved',
  retire:    'journal-tag--resolved',
};

// Matches JOURNAL_OPEN_NUDGE_RUNS in servers/trace_contract.py — an item
// re-flagged this many runs without resolution is one the encoder can't
// close on its own. Mirror-and-pin, like the dashboard's other server
// constants (queries/s2_runs.py does the same for ref-type families).
export const OPEN_NUDGE_RUNS = 5;

/** Is this note a long-lived open item — the kind that needs an operator? */
export function isStanding(note) {
  return (note?.open_runs || 0) >= OPEN_NUDGE_RUNS;
}

/** Unit → short display label. Unknown units render their raw name so a new
 *  encoder shows up unlabelled-but-visible rather than silently as "other". */
const UNIT_LABEL = {
  s1e:                 'scribe',
  consolidation:       'consolidation',
  community_detection: 'community',
  healer:              'healer',
  aspect_integration:  'aspects',
};
export function unitLabel(unit) { return UNIT_LABEL[unit] || unit || 'other'; }

// ── One note row ───────────────────────────────────────────────────────────
// `showUnit` adds the encoder label — on by default in the cross-encoder
// Journals view, off inside a run card where the encoder is already the card's
// identity. `onSubject` makes the subject clickable (the Journals view routes
// it to node detail when the subject parses as a node id).
export function journalRow(note, { showUnit = false, onSubject = null } = {}) {
  const tag = note.tag || '';
  const standing = isStanding(note);
  // Lifecycle suffix: "×4 since 08-17". Rendered as its own pill rather than
  // glued to the tag — it's the note's AGE, and the whole reason the reader
  // splits it off the grouping key.
  const life = note.open_runs
    ? '×' + note.open_runs + (note.since ? ' since ' + note.since : '')
    : '';
  return el('div', {
    class: ['journal-row', standing && 'journal-row--standing'].filter(Boolean),
    title: note.created_at ? relativeTime(note.created_at) : '',
  },
    el('div', { class: 'journal-row-head' },
      tag ? el('span', { class: ['journal-tag', TAG_ACCENT[tag.toLowerCase()]].filter(Boolean) }, tag) : null,
      life ? el('span', { class: 'journal-life', title: 'runs this stayed open' }, life) : null,
      showUnit ? el('span', { class: 'journal-unit' }, unitLabel(note.unit)) : null,
      note.subject ? el('span', {
        class: ['journal-subject', onSubject && 'journal-subject--clickable'].filter(Boolean),
        title: note.subject,
        onclick: onSubject ? (e) => { e.stopPropagation(); onSubject(note); } : null,
      }, note.subject) : null,
      standing ? el('span', { class: 'journal-standing-flag', title:
        'open ' + note.open_runs + ' runs — the encoder cannot close this alone' }, 'needs you') : null,
    ),
    el('div', { class: 'journal-note' }, note.note || ''),
  );
}

/** The note rows for one run — [] when the run left no residue, so callers
 *  can `...journalRows(run.journal_notes)` without an emptiness guard. */
export function journalRows(notes, opts = {}) {
  return (notes || []).map(n => journalRow(n, opts));
}

// ── The header chip ────────────────────────────────────────────────────────
/** 📓 button for a run card's header: opens the peek, never the card body.
 *  Returns null when the run has no notes — a clean run shows no chip, which
 *  is itself the signal (the encoder had nothing to flag). */
export function journalChip(notes, { title = 'Journal' } = {}) {
  const list = notes || [];
  if (!list.length) return null;
  const standing = list.some(isStanding);
  const chip = el('button', {
    class: ['journal-chip', standing && 'journal-chip--standing'].filter(Boolean),
    title: standing
      ? 'Read the journal — includes a long-lived open item'
      : 'Read the journal without expanding the run',
  }, '📓', el('span', { class: 'journal-chip-count' }, String(list.length)));
  // stopPropagation: the header click toggles the card body, and the whole
  // point of this chip is to read residue WITHOUT that.
  chip.addEventListener('click', (e) => {
    e.stopPropagation();
    openJournalPeek(list, title);
  });
  return chip;
}

// ── The peek popover ───────────────────────────────────────────────────────
// One instance, reused. Mounted lazily on first open and kept in the DOM
// afterwards — the alternative (build/teardown per open) re-registers the
// dismiss listeners every time, which is the leak pattern the insights
// panel's one-time wiring exists to avoid.
let _peek = null;

function _ensurePeek() {
  if (_peek) return _peek;
  const body = el('div', { class: 'journal-peek-body' });
  const title = el('div', { class: 'journal-peek-title' });
  const close = el('button', { class: 'journal-peek-close', title: 'Close (Esc)' }, '×');
  const panel = el('div', { class: 'journal-peek' },
    el('div', { class: 'journal-peek-head' }, title, close),
    body,
  );
  const scrim = el('div', { class: 'journal-peek-scrim' }, panel);
  close.addEventListener('click', () => closeJournalPeek());
  // Click the scrim (but not the panel) to dismiss.
  scrim.addEventListener('click', (e) => { if (e.target === scrim) closeJournalPeek(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && scrim.classList.contains('open')) closeJournalPeek();
  });
  document.body.appendChild(scrim);
  _peek = { scrim, panel, title, body };
  return _peek;
}

export function openJournalPeek(notes, title = 'Journal', opts = {}) {
  const p = _ensurePeek();
  const list = notes || [];
  const standing = list.filter(isStanding).length;
  // replaceChildren does NOT filter nulls the way el() does — a null child
  // lands as the literal text "null". Build, then filter.
  p.title.replaceChildren(...[
    el('span', { class: 'journal-peek-name' }, title),
    el('span', { class: 'journal-peek-count' },
      list.length + (list.length === 1 ? ' note' : ' notes')),
    standing ? el('span', { class: 'journal-standing-flag' },
      standing + ' need' + (standing === 1 ? 's' : '') + ' you') : null,
  ].filter(Boolean));
  p.body.replaceChildren(
    ...(list.length
      ? journalRows(list, opts)
      : [el('div', { class: 'feed-empty' }, 'No residue — a clean run.')]),
  );
  p.scrim.classList.add('open');
}

export function closeJournalPeek() {
  if (_peek) _peek.scrim.classList.remove('open');
}

// ── Inline block for a run card body ───────────────────────────────────────
/** The journal section as it appears INSIDE an expanded run card: a labelled
 *  block above the run's actions, so the operator reads the encoder's mind
 *  before its hands. Returns null for a clean run. */
export function journalBlock(notes, opts = {}) {
  const list = notes || [];
  if (!list.length) return null;
  return el('div', { class: 'journal-block' },
    el('div', { class: 'journal-block-label' }, 'Journal'),
    ...journalRows(list, opts),
  );
}

export default { journalRow, journalRows, journalChip, journalBlock,
                 openJournalPeek, closeJournalPeek, isStanding, unitLabel,
                 OPEN_NUDGE_RUNS };
