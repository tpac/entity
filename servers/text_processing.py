"""
brain — Text Processing Utilities

- split_identifier(): Convert programming identifiers to search tokens.
  Consumed by brain_assembly.py — filename hints become search queries.
"""

import re
from typing import List


# ── Identifier Splitting ─────────────────────────────────────────────

# Known code file extensions to strip
_EXT_RE = re.compile(
    r'\.(?:py|js|ts|tsx|jsx|sh|md|txt|json|yaml|yml|toml|cfg|db|sql|'
    r'html|css|scss|less|xml|csv|ini|env|lock|log|bak|tmp|gz|zip)$',
    re.IGNORECASE
)

# Version patterns to protect: v5.3.1, 2.0.1, V2, etc.
_VERSION_RE = re.compile(r'\bv?\d+(?:\.\d+)+\b', re.IGNORECASE)

# Core token pattern — order matters (earlier alternatives tried first):
#   1. [A-Z]+(?=[A-Z][a-z])  — acronym run before TitleCase: "HTML" in "HTMLParser"
#   2. [A-Z][a-z]+            — TitleCase word: "Parser", "Auth"
#   3. [A-Z]+                 — trailing/standalone acronym: "HTML" at end
#   4. [a-z]+                 — lowercase word: "parse", "get"
#   5. \d+                    — number run: "2" in "sha256"
_TOKEN_RE = re.compile(
    r'[A-Z]+(?=[A-Z][a-z])'   # acronym before Title word
    r'|[A-Z][a-z]+'            # Title word
    r'|[A-Z]+'                 # trailing/standalone acronym
    r'|[a-z]+'                 # lowercase word
    r'|\d+'                    # numbers
)


def split_identifier(name: str) -> List[str]:
    """Split a programming identifier into lowercase tokens for search.

    Handles camelCase, PascalCase, snake_case, kebab-case, file paths,
    file extensions, version numbers, and acronyms.

    Examples:
        >>> split_identifier('parseHTMLDoc')
        ['parse', 'html', 'doc']
        >>> split_identifier('getURLParser')
        ['get', 'url', 'parser']
        >>> split_identifier('pre_response_recall.py')
        ['pre', 'response', 'recall']
        >>> split_identifier('v2.3.1-beta')
        ['v2.3.1', 'beta']
        >>> split_identifier('servers/daemon_hooks.py')
        ['servers', 'daemon', 'hooks']
    """
    if not name or not name.strip():
        return []

    # Strip file extension
    name = _EXT_RE.sub('', name)

    # Protect version numbers by replacing with placeholders
    versions = {}
    def _protect_version(m):
        key = '\x00V%d\x00' % len(versions)
        versions[key] = m.group(0)
        return key
    name = _VERSION_RE.sub(_protect_version, name)

    # Split on separators: underscores, hyphens, dots, slashes
    parts = re.split(r'[_\-./\\]+', name)

    # Tokenize each part using the camelCase-aware pattern
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in versions:
            tokens.append(versions[part])
        else:
            found = _TOKEN_RE.findall(part)
            tokens.extend(found)

    # Lowercase and filter single-char alphabetic noise (keep digits)
    result = []
    for t in tokens:
        if t in versions.values():
            result.append(t.lower())
        elif t.isdigit():
            result.append(t)  # Always keep digits (e.g., "8" in "UTF8")
        elif len(t) > 1:
            result.append(t.lower())

    return result
