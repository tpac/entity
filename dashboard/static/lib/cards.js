// ===========================================================================
// lib/cards.js — the shared card primitives.
// ---------------------------------------------------------------------------
// Every activity card in the dashboard is the same envelope: a header of chips
// with an optional lazy "Show Prompt" body, over a collapsible body of rows.
// Shared by lib/activity.js and tabs/s2.js so neither tab owns the other's
// primitives.
//
// Exports:
//   wirePromptToggle(btn, body, lazyLoad)  — Show/Hide Prompt with one fetch
//   promptSection()                        — the {button, body} pair, built
//   subRow({...})                          — a CREATED/REVISED/… action row
//   tierRow(letter, refType, summary, …)   — an O / K / Δ row
//   edgeRow(edge)                          — a "source —relation→ target" row
// ===========================================================================

import { el } from '/static/lib/dom.js';
import { loadNodeDetail } from '/static/lib/node_detail.js';

/** Wire a "Show Prompt" / "Hide Prompt" button to its collapsible body.
 * `lazyLoad` is an async function called the FIRST time the body is expanded;
 * its return value populates the <pre>. Prompts are hundreds of KB, which is
 * why nothing ships them inline with a polled list. */
export function wirePromptToggle(button, body, lazyLoad) {
  button.addEventListener('click', async (e) => {
    // The header click toggles the card; a prompt button must not also do that.
    e.stopPropagation();
    const showing = body.style.display !== 'none';
    if (showing) {
      body.style.display = 'none';
      button.textContent = button.dataset.showLabel || 'Show Prompt';
      return;
    }
    body.style.display = 'block';
    button.textContent = button.dataset.hideLabel || 'Hide Prompt';
    if (lazyLoad) {
      const pre = body.querySelector('pre');
      if (pre && pre.textContent === 'Loading...') {
        try { pre.textContent = await lazyLoad(); }
        catch (_) { pre.textContent = '(failed to load prompt)'; }
      }
    }
  });
}

/** Build a wired prompt section: `{ button, body }`, already connected.
 * `label` names what prompt it is — with recognition and remembering on the
 * same card, "Show Prompt" alone stops being enough. */
export function promptSection(lazyLoad, { label = 'Prompt' } = {}) {
  const body = el('div', { class: 'card-prompt-body', style: { display: 'none' } },
    el('pre', { class: 'enc-prompt-pre' }, 'Loading...'));
  const button = el('button', {
    class: 'hook-details-btn',
    dataset: { showLabel: label, hideLabel: 'Hide ' + label.toLowerCase() },
  }, label);
  wirePromptToggle(button, body, lazyLoad);
  return { button, body };
}

/** An action row: a kind pill, the node's type badge, its title, and a
 * content excerpt. Clicking opens node detail when `nodeId` is given. */
export function subRow({ kindClass, kindLabel, typeName, title, content,
                         contentDim, nodeId }) {
  const kindEl = el('span', { class: ['enc-kind', kindClass].filter(Boolean) }, kindLabel);
  const typeEl = typeName ? el('span', { class: 'type-badge type--' + typeName }, typeName) : null;
  return el('div', {
    class: ['enc-entry', 'enc-sub-row', nodeId && 'enc-sub-row--clickable', kindClass].filter(Boolean),
    dataset: kindClass ? { kind: kindClass } : null,
    onclick: nodeId ? (e) => { e.stopPropagation(); loadNodeDetail(nodeId); } : null,
  },
    kindEl,
    ' ',
    typeEl,
    typeEl ? ' ' : null,
    el('span', { class: 'enc-title' }, title || ''),
    content ? el('div', {
      class: ['enc-meta-line', contentDim && 'enc-meta-line--dim'].filter(Boolean),
    }, (content || '').substring(0, contentDim ? 250 : 400)) : null,
  );
}

/** An O / K / Δ tier row. `accentClass` styles the label for the standard
 * tiers; `accentStyle` is the escape hatch for a unit's own color (data, not
 * design). Returns null for an empty summary so callers can spread freely. */
export function tierRow(letter, refType, summary, { accentClass, accentStyle } = {}) {
  if (!summary) return null;
  const labelText = refType ? letter + ' ' + refType + ':' : letter + ':';
  return el('div', { class: 'enc-tier-row' },
    el('strong', { class: accentClass || null, style: accentStyle || null }, labelText),
    ' ',
    summary,
  );
}

/** A "source —relation→ target" edge row. */
export function edgeRow(e) {
  return el('div', {
    class: 'enc-entry enc-sub-row connected',
    dataset: { kind: 'connected' },
  },
    el('span', { class: 'enc-kind connected' }, 'CONNECTED'),
    ' ',
    e.source_title || '',
    ' ',
    el('span', { class: 'enc-edge-arrow' }, '—' + (e.relation || '') + '→'),
    ' ',
    e.target_title || '',
  );
}

export default { wirePromptToggle, promptSection, subRow, tierRow, edgeRow };
