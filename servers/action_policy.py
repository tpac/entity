"""Shared action policy. Plain settings and shell facts; no host/brain imports.

The launcher loads ~/.config/brain/env. Callers resolve one immutable snapshot
per capture request or encoder window. Git recognition is conservative syntax
inspection, never execution: computed executables and script internals remain
unknown. Only a literal invocation authorizes excluding a whole call.
"""
from dataclasses import dataclass
import os
import re
import shlex
import sys


@dataclass(frozen=True)
class ActionPolicy:
    profile: str = 'thin'
    exclude_git: bool = True
    max_lines: int = 40
    max_bytes: int = 6000


def load_action_policy(environ=None):
    env = os.environ if environ is None else environ
    defaults = ActionPolicy()

    def setting(name, default, parse):
        raw = env.get('BRAIN_ACTIONS_' + name, '').strip()
        if not raw:
            return default
        try:
            return parse(raw)
        except (ValueError, KeyError):
            print('[actions] invalid BRAIN_ACTIONS_%s=%r; using %r' %
                  (name, raw, default), file=sys.stderr)
            return default

    def bounded(raw, low, high):
        value = int(raw)
        if not low <= value <= high:
            raise ValueError(raw)
        return value

    def profile(raw):
        if raw not in ('thin', 'balanced', 'full'):
            raise ValueError(raw)
        return raw

    return ActionPolicy(
        setting('PROFILE', defaults.profile, profile),
        setting('EXCLUDE_GIT', defaults.exclude_git, lambda s: {
            '1': True, 'true': True, '0': False, 'false': False}[s.lower()]),
        setting('MAX_LINES', defaults.max_lines, lambda s: bounded(s, 4, 200)),
        setting('MAX_BYTES', defaults.max_bytes, lambda s: bounded(s, 512, 24000)))


_ASSIGNMENT = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*=')
_SEPARATORS = frozenset({'&&', '||', ';', '&', '|', '|&', '(', ')', '\n'})
_PREFIXES = frozenset({'if', 'then', 'elif', 'else', 'while', 'until', 'do', '!', '{', '}'})
_WRAPPERS = frozenset({'env', 'command', 'exec', 'sudo', 'nohup', 'nice',
                       'timeout', 'time', 'caffeinate', 'dev'})
_OPTION_ARGS = {
    'env': {'-u', '--unset', '-C', '--chdir'},
    'sudo': {'-u', '--user', '-g', '--group', '-C', '--close-from',
             '-h', '--host', '-p', '--prompt', '-r', '--role', '-t', '--type'},
    'nice': {'-n', '--adjustment'},
    'timeout': {'-k', '--kill-after', '-s', '--signal'},
    'caffeinate': {'-t', '-w'},
    'time': {'-o', '--output', '-f', '--format'},
    'exec': {'-a'},
}
_SHELLS = frozenset({'sh', 'bash', 'dash', 'zsh', 'ksh'})
_OPERATORS = re.compile(r'<<<|<<|>>|>&|<&|<>|&>|&&|\|\||\|&|[;&|()<>\n]')


def _words(text, start=0):
    """Scan one logical shell header; return raw words and the next offset.

    Quotes, escapes and comments retain their shell boundaries. Reading from
    an offset avoids repeatedly lexing the prefix of a multiline script.
    Incomplete historical cues return only their complete token prefix.
    """
    words, word, quote, i = [], [], '', start
    while i < len(text):
        char = text[i]
        if char == "\\" and quote != "'":
            if i + 1 == len(text):
                return words, len(text)
            if text[i + 1] != "\n":
                word.append(text[i:i + 2])
            i += 2
            continue
        if quote:
            word.append(char)
            if char == quote:
                quote = ''
        elif char in "\"'`":
            quote = char
            word.append(char)
        elif char == '#' and not word:
            end = text.find('\n', i)
            return words, len(text) if end < 0 else end + 1
        elif char in ' \t\r' or char in ';&|()<>\n':
            if word:
                words.append(''.join(word))
                word = []
            if char == '\n':
                return words, i + 1
            if char in ';&|()<>':
                operator = _OPERATORS.match(text, i).group()
                words.append(operator)
                i += len(operator)
                continue
        else:
            word.append(char)
        i += 1
    if word and not quote:
        words.append(''.join(word))
    return words, i


def _literal(word):
    # Quote removal is safe for literal words. Expanded executables stay unknown.
    if '$' in word or '`' in word:
        return word
    try:
        parts = shlex.split(word, comments=False)
        return parts[0] if len(parts) == 1 else word
    except ValueError:
        return word


def _shell_body(word):
    if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
        return word[1:-1]
    return word


def _substitutions(text):
    """Literal shell bodies in $() and backticks, including double quotes.

    This only finds bounded source spans. Recursive recognition uses the same
    parser; single-quoted text and escaped dollar/backtick characters are data.
    """
    i, quote = 0, ''
    while i < len(text):
        c = text[i]
        if c == '\\' and quote != "'":
            i += 2
            continue
        if c == "'" and quote != '"':
            quote = '' if quote else "'"
        elif c == '"' and quote != "'":
            quote = '' if quote else '"'
        elif quote != "'" and (text.startswith('$(', i) or c == '`'):
            if text.startswith('$((', i):
                i += 3
                continue
            start = i + (2 if c == '$' else 1)
            end, depth, inner_quote = start, 1, ''
            while end < len(text):
                ch = text[end]
                if ch == '\\' and inner_quote != "'":
                    end += 2
                    continue
                if ch in "\"'" and (not inner_quote or inner_quote == ch):
                    inner_quote = '' if inner_quote else ch
                elif not inner_quote:
                    if c == '`' and ch == '`':
                        break
                    if c == '$':
                        depth += (ch == '(') - (ch == ')')
                        if depth == 0:
                            break
                end += 1
            if end < len(text):
                yield text[start:end]
                i = end
        i += 1


def _command_chunks(command):
    """Yield command words, skipping heredoc data without rescanning prefixes.

    A quoted heredoc is entirely data; only substitutions execute in an
    unquoted one. Incomplete historical summaries may still provide a definite
    leading Git invocation, so the lexer retains complete token prefixes.
    """
    offset = 0
    while offset < len(command):
        words, offset = _words(command, offset)
        yield words
        for i, word in enumerate(words[:-1]):
            if word != '<<':
                continue
            delimiter = words[i + 1]
            strip_tabs = delimiter.startswith('-')
            if strip_tabs:
                delimiter = delimiter[1:]
            quoted = any(c in delimiter for c in "\\\"'")
            try:
                literal = shlex.split(delimiter, comments=False)[0]
            except (ValueError, IndexError):
                literal = delimiter
            body_start = offset
            while offset < len(command):
                end = command.find('\n', offset)
                end = len(command) if end < 0 else end + 1
                candidate = command[offset:end].rstrip('\r\n')
                if strip_tabs:
                    candidate = candidate.lstrip('\t')
                if candidate == literal:
                    if not quoted:
                        for body in _substitutions(command[body_start:offset]):
                            yield from _command_chunks(body)
                    offset = end
                    break
                offset = end


def _segment_has_git(words, depth):
    i = 0
    while i < len(words):
        word = _literal(words[i])
        if _ASSIGNMENT.match(words[i]) or words[i] in _PREFIXES:
            i += 1
            continue
        # Leading redirects do not choose the executable.
        if words[i] in ('>', '>>', '<', '<<', '<<<', '<>', '>&', '<&'):
            i += 2
            continue
        if word.isdigit() and i + 1 < len(words) and words[i + 1].startswith(('>', '<')):
            i += 1
            continue
        base = word.rsplit('/', 1)[-1]
        if base == 'git':
            return True
        if base in _SHELLS:
            for j in range(i + 1, len(words) - 1):
                flag = _literal(words[j])
                if not flag.startswith('-') or flag == '--':
                    break  # a script filename ends the shell's option list
                if flag == '--noexec' or (not flag.startswith('--') and 'n' in flag):
                    return False  # syntax checking does not run the body
                if flag.startswith('-') and not flag.startswith('--') and 'c' in flag:
                    return has_git_command(_shell_body(words[j + 1]), depth + 1)
            return False
        if base not in _WRAPPERS:
            return False
        i += 1
        while i < len(words):
            option = _literal(words[i])
            if _ASSIGNMENT.match(words[i]):
                i += 1
            elif option == '--':
                i += 1
                break
            elif option.startswith('-'):
                flags = option[1:] if not option.startswith('--') else ''
                if base == 'command' and any(c in flags for c in 'vV'):
                    return False  # resolves a name without executing it
                if option in ('--help', '--version'):
                    return False
                if base == 'sudo' and (option in ('--list', '--validate', '--remove-timestamp', '--reset-timestamp')
                                       or any(c in flags for c in 'lvVK')):
                    return False
                if base == 'env' and option in ('-S', '--split-string') and i + 1 < len(words):
                    return has_git_command(_shell_body(words[i + 1]), depth + 1)
                i += 2 if option in _OPTION_ARGS.get(base, ()) else 1
            else:
                break
        if base == 'timeout':
            i += 1  # duration
    return False


def has_git_command(command, depth=0):
    """True only for an observed literal Git invocation, anywhere in the call.

    Handles command lists, pipelines, common wrappers, shell -c, substitutions
    and heredocs. Does not interpret aliases, computed executable names or
    Python/etc. program bodies. Recognition never executes supplied code.
    """
    if not isinstance(command, str) or depth > 8:
        return False
    for words in _command_chunks(command):
        # Function bodies are deferred code, not observed invocations. Once a
        # declaration appears, retain the call rather than attempt function
        # execution/name resolution (the same boundary as aliases/scripts).
        if any((word == 'function' and (i == 0 or words[i - 1] in _SEPARATORS
                                        or words[i - 1] in _PREFIXES)) or (
                re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', word)
                and words[i + 1:i + 3] == ['(', ')'])
               for i, word in enumerate(words)):
            return False
        # Tokens have shell comments removed, while quote boundaries survive.
        if any(has_git_command(body, depth + 1)
               for body in _substitutions(' '.join(words))):
            return True
        segment = []
        for word in words + ['\n']:
            if word in _SEPARATORS or (word and set(word) == {'\n'}):
                if _segment_has_git(segment, depth):
                    return True
                segment = []
            else:
                segment.append(word)
    return False
