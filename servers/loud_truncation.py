"""Loud truncation — the standing truncation contract.

Every truncation point in the brain must be LOUD: never a silent slice, always a
marker naming what was dropped. These are the shared primitives. Callers pass a
domain-specific marker (e.g. one pointing at where the full text still lives) —
the mechanism (rstrip head + dropped-count marker) is the same everywhere.

Used wherever the brain shapes text for a bounded destination — trace stores,
delta metadata, and the channel blocks injected into a session. Imports
nothing from servers — it sits below every domain that caps, so none of them
owns another's primitive.
"""

import re


def cap_text_loud(s, limit, marker="…[+%d chars truncated]"):
    """Truncate `s` to `limit` chars LOUDLY: rstrip the head and append `marker`.
    A marker with `%d` receives the dropped char count; one without (a bare
    '…') only says the text goes on. Returns `s` unchanged when it fits. The
    dropped tail is genuinely lost at call sites that keep only a head, so the
    marker is the only record it happened."""
    s = s or ''
    if len(s) <= limit:
        return s
    tail = marker % (len(s) - limit) if '%d' in marker else marker
    return s[:limit].rstrip() + " " + tail


def cap_text_at_boundary(s, limit, marker="…[+%d chars truncated]"):
    """cap_text_loud that lands on a sentence boundary — the last sentence end
    (failing that, the last word break) past 60% of `limit` — so the kept head
    reads as complete text a reader can act on. A marker with `%d` names how
    much is behind it; one without (a bare '…') only says the text goes on.
    Returns `s` unchanged when it fits."""
    s = s or ''
    if limit <= 0 or len(s) <= limit:
        return s
    head = s[:limit]
    floor = int(limit * 0.6)
    cut = -1
    for mark in ('. ', '.\n', '! ', '? ', '\n'):
        pos = head.rfind(mark)
        if pos >= floor:
            cut = max(cut, pos + 1)
    if cut < 0:
        pos = head.rfind(' ')
        cut = pos if pos >= floor else limit
    kept = s[:cut].rstrip()
    tail = marker % (len(s) - len(kept)) if '%d' in marker else marker
    return kept + " " + tail


_TRUNCATION_TAIL_RE = re.compile(r'\u2026(?:\s*[\[(]\+\d+[^\])]*[\])])?\s*$')


def looks_truncated(s):
    """True when `s` ends in a truncation marker — the contract above read
    backwards. Covers both families: the bare '\u2026' cut and the loud
    '\u2026[+N chars truncated]' / '\u2026 (+N chars: get_nodes)' tails.

    The one consumer today is the swap write boundary: text an agent copied
    out of a CUT render can never match the stored value, and the remedy is
    the full new value rather than a re-read of the same door.
    """
    return bool(_TRUNCATION_TAIL_RE.search(s or ''))


def one_line(s):
    """Collapse all whitespace (incl. newlines) to single spaces — the shape
    for text that must occupy ONE line of a rendered block (a row in a
    prompt table, a summary). Pairs with cap_text_loud."""
    return ' '.join(str(s or '').split())


def cap_list_loud(items, limit, marker="…[+%d more truncated]"):
    """Keep the first `limit` items LOUDLY: append a marker element naming how
    many were dropped, vs a silent slice. Stays a list (shape-valid)."""
    items = list(items or [])
    if len(items) <= limit:
        return items
    return items[:limit] + [marker % (len(items) - limit)]


def compose_block_loud(items, render, cap, reserved=0, sep="\n\n"):
    """Fit rendered items under `cap` chars LOUDLY: render in order, stop at
    the first item that would overflow (always keeping one, so a single
    oversize item still shows), and report how many were dropped so the
    caller can name them at the tail. `reserved` is what the caller's head
    already spent from the budget. Items past the cut are never rendered.
    `sep` joins the kept items — a blank line for multi-line items (the
    default), a newline for one-line rows. Returns (body, kept, dropped).
    The caller owns head and tail wording: how a dropped item comes back
    differs per domain (a thalamus item stays due for a later pull; a courier
    message stays in the inbox for self_inbox or the next Stop)."""
    parts, used, dropped = [], reserved, 0
    for i, item in enumerate(items):
        rendered = render(item).strip()
        if parts and used + len(rendered) + len(sep) > cap:
            dropped = len(items) - i
            break
        parts.append(rendered)
        used += len(rendered) + len(sep)
    return sep.join(parts), len(parts), dropped
