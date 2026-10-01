"""Contract tests for servers/interaction_defaults.py — the name→(template,
config) index the override resolver serves.

The completeness test is what makes a raising resolver safe to ship (plan
step 4): every literal interaction name the runtime passes to the accessors
must have a registry entry, or a default-run on that name resolves to
nothing.
"""
import os
import re

from servers.interaction_defaults import INTERACTION_DEFAULTS

SERVERS_DIR = os.path.join(os.path.dirname(__file__), '..', 'servers')

# LLM-prompt-backed names carry a real template; config-only names carry ''.
PROMPT_BACKED = {
    's1e', 's1e_gist', 'surface',
    's2_community_enrichment', 's2_consolidation_enrichment', 's2_healer',
    's2_aspects',
}


class TestRegistryShape:
    def test_every_entry_is_template_config_pair(self):
        for name, (template, config) in INTERACTION_DEFAULTS.items():
            assert isinstance(template, str), name
            assert isinstance(config, dict), name
            assert len(config) > 0, "%s config is empty" % name

    def test_prompt_backed_names_carry_real_templates(self):
        for name in PROMPT_BACKED:
            template, _ = INTERACTION_DEFAULTS[name]
            assert len(template) > 100, \
                "%s template suspiciously short (%d chars)" % (name, len(template))

    def test_config_only_names_carry_empty_templates(self):
        for name, (template, _) in INTERACTION_DEFAULTS.items():
            if name not in PROMPT_BACKED:
                assert template == '', \
                    "%s is config-only but carries a template" % name


class TestDefaultFilesOwnTheirContent:
    """The inverse of the old seed-role check: no default file may claim the
    DB is authoritative or carry sync provenance. Editing a *_prompt.py IS
    the deployment now — a docstring telling editors otherwise re-teaches
    the model this migration deleted."""

    FORBIDDEN = ('seed', 'authoritative', 'last sync')

    def _prompt_files(self):
        found = []
        for root, _dirs, files in os.walk(SERVERS_DIR):
            for fn in files:
                if fn.endswith('_prompt.py'):
                    found.append(os.path.join(root, fn))
        return found

    def test_no_default_file_claims_db_ownership(self):
        import ast
        files = self._prompt_files()
        assert len(files) >= len(PROMPT_BACKED), \
            'prompt-file walk found %d files for %d prompt-backed names' % (
                len(files), len(PROMPT_BACKED))
        offenders = {}
        for path in files:
            with open(path, encoding='utf-8') as f:
                doc = (ast.get_docstring(ast.parse(f.read())) or '').lower()
            hits = [w for w in self.FORBIDDEN if w in doc]
            if hits:
                offenders[os.path.basename(path)] = hits
        assert not offenders, (
            'default files still carrying pre-override-model docstrings '
            '(code owns the default; the DB holds only overrides): %s'
            % offenders)


class TestSurfaceDefaultPairsTemplateWithLayout:
    """Template + layout flip atomically — that's why layout rides in the
    interaction config. The renderer builds ONE layout, so the default must
    name that one and the template must teach the grammar it emits.
    Behavior-based: renders a candidate and reads the result, no whitelist."""

    def test_surface_template_and_layout_flip_together(self):
        from servers.scales.s1.surface_contract import (
            build_surface_prompt, SURFACE_LAYOUT)
        template, config = INTERACTION_DEFAULTS['surface']
        # Subscript, never .get(fallback): a fallback here would name a
        # layout no renderer builds, which is the defect this guards.
        assert config['layout'] == SURFACE_LAYOUT, \
            'the surface default names layout %r, but the renderer builds ' \
            '%r' % (config['layout'], SURFACE_LAYOUT)
        cand = {'id': 'a' * 32, 'title': 'Default check', 'type': 'fact',
                'content': 'body', 'score': 0.9,
                'created_at': '2026-07-01T00:00:00+00:00'}
        prompt, _ = build_surface_prompt([cand], 'a message')
        assert '<candidate id="aaaaaaaa"' in prompt, \
            'default layout did not reach the XML renderer'
        assert '<candidate' in template, \
            'the shipped template never teaches the <candidate> grammar ' \
            'the renderer emits'


    def test_surface_template_pick_cap_matches_config(self):
        """The pick cap reaches Haiku only as prose ("Up to N");
        SURFACE['max_selected'] is read by the eval bench's over-cap check
        and the inject budget arithmetic. Nothing enforces the cap after
        Haiku answers, so the prose and the constant must be the same number."""
        from servers.scales.s1.surface_contract import SURFACE
        template, _config = INTERACTION_DEFAULTS['surface']
        assert 'Up to %d ' % SURFACE['max_selected'] in template, \
            'surface prompt prose names a different pick cap than ' \
            "SURFACE['max_selected']=%d" % SURFACE['max_selected']


class TestRegistryCompleteness:
    """Every name the runtime passes to the resolver must be a registry key.
    A name outside the registry has no code default — under the override
    model a default-run on it resolves to nothing (the resolver raises, and
    the unit ships green because a test tier cannot fail for a registry
    entry nobody wrote).

    Three consumer shapes, each collected its own way — a literal-only scan
    misses names that reach the resolver through variables
    (`_call_llm('s2_healer', ...)` → `base.py`'s
    `get_interaction_effective(interaction_name)`):
      1. direct accessor literals,
      2. `_call_llm('<name>', ...)` literals (the S2 unit pattern),
      3. name REGISTRIES that feed the resolver (scopes)."""

    _CALL_RE = re.compile(
        r"get_interaction(?:_prompt|_config|_stamp|_effective)?\(\s*['\"]([a-z0-9_]+)['\"]")
    _CALL_LLM_RE = re.compile(r"_call_llm\(\s*['\"]([a-z0-9_]+)['\"]")

    def _literal_names_in_servers(self):
        names = set()
        for root, _dirs, files in os.walk(SERVERS_DIR):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(root, fn)
                with open(path, encoding='utf-8') as f:
                    text = f.read()
                names.update(self._CALL_RE.findall(text))
                names.update(self._CALL_LLM_RE.findall(text))
        return names

    def test_every_runtime_literal_is_a_registry_key(self):
        literals = self._literal_names_in_servers()
        assert literals, "grep found no accessor literals — regex broken?"
        # The scan must see the S2-unit shape, or a regex edit quietly
        # shrinks coverage back to the literal-only blind spot.
        assert 's2_healer' in literals and 's2_aspects' in literals, \
            "_call_llm literal collection went blind"
        missing = literals - set(INTERACTION_DEFAULTS)
        assert not missing, \
            "accessor literals with no code default: %s" % sorted(missing)

    def test_registry_fed_names_are_registry_keys(self):
        """Consumers whose interaction name arrives through a registry, not
        a literal: ScopePolicy loads scopes._INTERACTION_NAME. A registry
        name without a code default must fail HERE, not in production via
        the errors table."""
        from servers.scopes import _INTERACTION_NAME as SCOPES_NAME
        assert SCOPES_NAME in INTERACTION_DEFAULTS


class TestDefaultsPassTheirOwnValidator:
    """A code default never passes through `register_interaction`'s door, and
    the resolver only validates a config the DB actually contributed — so no
    runtime check ever judges the value that ships to everyone. Once the
    collapse drops the override pointers, that value IS the running config.
    Committing the default is its only write boundary, which makes this test
    the door for it."""

    def test_every_validated_name_has_a_valid_default(self):
        from servers.interaction_defaults import INTERACTION_VALIDATORS
        assert 'scopes' in INTERACTION_VALIDATORS, (
            "scopes lost its validator — this loop would go vacuous and the "
            "one config that governs isolation would ship unchecked")
        for name, validate in INTERACTION_VALIDATORS.items():
            _template, config = INTERACTION_DEFAULTS[name]
            violations = validate(config)
            assert not violations, (
                "%s ships a code default its own validator rejects: %s"
                % (name, violations))

    def test_validators_name_registry_entries(self):
        from servers.interaction_defaults import INTERACTION_VALIDATORS
        unknown = set(INTERACTION_VALIDATORS) - set(INTERACTION_DEFAULTS)
        assert not unknown, \
            "validators for names with no code default: %s" % sorted(unknown)


class TestFingerprint:
    def test_fingerprint_is_stable_and_canonical(self):
        from servers.interaction_defaults import interaction_fingerprint
        a = interaction_fingerprint('n', 't', {'b': 1, 'a': 2})
        b = interaction_fingerprint('n', 't', {'a': 2, 'b': 1})
        assert a == b, "canonicalization must be key-order independent"
        assert len(a) == 12 and int(a, 16) >= 0
        assert interaction_fingerprint('n', 't', {'a': 1}) != a
