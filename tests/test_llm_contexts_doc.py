"""``docs/llm_contexts.md`` is the reference for every LLM call, so the
settings, timeouts and caps it names have to be the ones the code has.

The page is prose, and prose drifts: a key is renamed and the page keeps
the old name, a default changes and the page keeps the old number. Nothing
at runtime notices, because nothing reads the page. These tests read it the
way a person does and hold it to the code:

* every key in the page's key list must be a setting;
* an identifier written as a setting must be a setting;
* an identifier that is neither a setting nor anything the code defines is
  invented;
* a number written next to a setting must be that setting's default;
* a number written next to a constant must be that constant's value;
* a place in the code is named by a symbol, never by a line number.

All but the last compare against the code at test time, so a default that is
changed on purpose fails here until the page follows, and nothing in this
file has to be edited when it does.

The checkers take the page's text as an argument so that they can be run on
a page written to be wrong: a check that has never been seen to fail proves
nothing about the next typo.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import Counter
from pathlib import Path

import pytest

from jarvis.config import get_default_config

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "llm_contexts.md"
SRC = ROOT / "src" / "jarvis"

# A fenced block is a diagram or a sample, not a claim about a setting.
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SPAN = re.compile(r"`([^`\n]+)`")

# `cfg.<name>` names a settings attribute wherever it appears, even inside
# a longer span such as ``get_auxiliary_backend(cfg, cfg.intent_judge_model)``.
_CFG_ATTRIBUTE = re.compile(r"\bcfg\.([A-Za-z_][A-Za-z0-9_]*)")

# A lower-case snake_case word that is not a call, not an attribute of
# something else and not part of a path or a file name.
_BARE_WORD = re.compile(
    r"(?<![A-Za-z0-9_./\-])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![A-Za-z0-9_(/\-]|\.[A-Za-z])"
)

# The same word, wherever it sits in a span: also a call (``get_llm_backend(``)
# or an attribute (``DialogueMemory.record_tool_turn``). The look-around only
# keeps out paths and file names.
_SNAKE_WORD = re.compile(
    r"(?<![A-Za-z0-9_/\-])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![A-Za-z0-9_/\-]|\.[A-Za-z])"
)

# The page's list of what can be set, from its heading to the next one.
_KEY_LIST = re.compile(
    r"^## Config keys[ \t]*\n(.*?)(?=^## |\Z)", re.DOTALL | re.MULTILINE
)

# Words the key list may carry that are an option of a backend, not a setting.
_NOT_SETTINGS = frozenset({"keep_alive"})

# `` `llm_chat_timeout_sec` (180s) ``, `` `x_timeout_sec` (default 8 s, ...) ``,
# `` `agentic_max_turns` (8) ``, `` `transcript_buffer_duration_sec=120` ``.
_SETTING_NUMBER = re.compile(
    r"`([a-z][a-z0-9_]*)`\s*\((?:default\s+)?(\d+(?:\.\d+)?)\s?s?(?=[,;)])"
    r"|`([a-z][a-z0-9_]*)=(\d+(?:\.\d+)?)`"
)

# `` `_DIGEST_MAX_CHARS` (500) ``, `` `SHUTDOWN_DIARY_TIMEOUT_SEC` (45s) ``.
_CONSTANT_NUMBER = re.compile(r"`(_?[A-Z][A-Z0-9_]+)`\s*\((\d+)\s?s?\)")


# `engine.py:68`, `(~line 60)`, `lines 48 & 136`: a position, not a name.
_LINE_NUMBER = re.compile(r"\.py:\d+|~\s?lines?\s+\d+|\blines?\s+\d+")

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _prose(text: str) -> str:
    return _FENCE.sub("", text)


def _setting_suffixes(defaults: dict) -> set[str]:
    """The last word of every setting name that at least two settings end in.

    ``model``, ``enabled``, ``sec``... A snake_case word in the page that ends
    like that is read as a claim about a setting. Derived from the defaults
    rather than listed, so a new family of settings is covered the day it
    exists. A one-off ending is left out on purpose: it would flag ordinary
    identifiers (``tool_calls`` ends like ``tool_search_max_calls``).
    """
    endings = Counter(k.rsplit("_", 1)[1] for k in defaults if "_" in k)
    return {word for word, n in endings.items() if n >= 2}


def _cited_setting_names(text: str, defaults: dict) -> dict[str, str]:
    """Every identifier the page presents as a setting, with where it is."""
    suffixes = _setting_suffixes(defaults)
    cited: dict[str, str] = {}
    for span in _SPAN.findall(_prose(text)):
        for name in _CFG_ATTRIBUTE.findall(span):
            cited.setdefault(name, span)
        without_attributes = _CFG_ATTRIBUTE.sub(" ", span)
        for word in _BARE_WORD.findall(without_attributes):
            if word.rsplit("_", 1)[1] in suffixes:
                cited.setdefault(word, span)
    return cited


def unknown_setting_like_names(text: str, defaults: dict) -> dict[str, str]:
    """Every name the page cites as a setting that no setting has."""
    return {
        name: span
        for name, span in sorted(_cited_setting_names(text, defaults).items())
        if name not in defaults
    }


def unknown_in_key_list(text: str, defaults: dict) -> dict[str, str]:
    """Every word in the page's key list that is not a setting.

    The list says what can be set, so each snake_case word in it is held to
    the settings themselves, with no shared ending to shelter behind: the
    suffix rule alone lets ``intent_judge_mdl`` through, because no second
    setting ends in ``mdl``. A call (``warm_up_chat_model()``) is a function,
    not a key.
    """
    section = _KEY_LIST.search(_prose(text))
    if section is None:
        return {}
    unknown: dict[str, str] = {}
    for span in _SPAN.findall(section.group(1)):
        names = _CFG_ATTRIBUTE.findall(span)
        names += _BARE_WORD.findall(_CFG_ATTRIBUTE.sub(" ", span))
        for name in names:
            if name not in defaults and name not in _NOT_SETTINGS:
                unknown.setdefault(name, span)
    return dict(sorted(unknown.items()))


def invented_identifiers(
    text: str, defaults: dict, known: frozenset[str]
) -> dict[str, str]:
    """Every snake_case word in a span that is neither a setting nor in the code.

    Catches a typo anywhere on the page, a function or a variable included,
    which the two checks above cannot see: a word that appears nowhere under
    ``src/jarvis`` was never the name of anything.
    """
    unknown: dict[str, str] = {}
    for span in _SPAN.findall(_prose(text)):
        for word in _SNAKE_WORD.findall(span):
            if word not in defaults and word not in known:
                unknown.setdefault(word, span)
    return dict(sorted(unknown.items()))


@functools.cache
def _parsed_sources() -> tuple[ast.Module, ...]:
    """Every module under ``src/jarvis``, parsed once for the whole file."""
    trees = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            trees.append(ast.parse(path.read_text(encoding="utf-8")))
        except (SyntaxError, UnicodeDecodeError):
            continue
    return tuple(trees)


@functools.cache
def _source_identifiers() -> frozenset[str]:
    """Every name the code gives or uses, and every word its strings carry.

    The words in strings are what name a database table, a dict key handed to
    a server or a tool: ``conversation_summaries``, ``keep_alive``. A
    docstring is prose about the code rather than part of it, so it is left
    out; otherwise a misspelling copied from one would vouch for itself.
    """
    found: set[str] = set()
    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for tree in _parsed_sources():
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, scopes)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.add(node.id)
            elif isinstance(node, ast.Attribute):
                found.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, ast.arg):
                found.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                found.add(node.arg)
            elif isinstance(node, ast.alias):
                found.add((node.asname or node.name).rsplit(".", 1)[-1])
            elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and id(node) not in docstrings):
                found.update(_WORD.findall(node.value))
    return frozenset(found)


def _module_level_numbers() -> dict[str, set[float]]:
    """Every module-level ``NAME = <number>`` under ``src/jarvis``."""
    found: dict[str, set[float]] = {}
    for tree in _parsed_sources():
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not (isinstance(value, ast.Constant)
                    and isinstance(value.value, (int, float))
                    and not isinstance(value.value, bool)):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    found.setdefault(target.id, set()).add(float(value.value))
    return found


# Every misspelling below is one a reader would take for a real key, and none
# of them shares an ending with a second setting, which is exactly how the
# suffix rule alone lets a typo through.
_RENAMED_SETTINGS = [
    "intent_judge_mdl",
    "planner_active",
    "reply_language",
    "tool_selection_stratgy",
    "auto_redact_before_remote",
    "llm_api_token_env",
    "tool_search_budget",
    "llm_digest_timeout",
]


@pytest.mark.unit
@pytest.mark.parametrize("renamed", _RENAMED_SETTINGS)
def test_a_renamed_setting_in_the_key_list_is_caught(renamed):
    """The key list is the page's claim about what can be set, word for word."""
    page = (
        "## Config keys\n\n"
        f"- Flags: `memory_digest_enabled`, `{renamed}`, `llm_chat_model`\n"
    )

    assert renamed in unknown_in_key_list(page, get_default_config())


@pytest.mark.unit
@pytest.mark.parametrize("renamed", _RENAMED_SETTINGS)
def test_a_renamed_setting_cited_in_passing_is_caught(renamed):
    """Outside the key list, a setting is named in the middle of a sentence.

    Judged against a code that knows none of the misspellings, so that the
    check is exercised whatever else happens to be defined under ``src``.
    """
    page = f"## 7. Tool Router\n\n- **Trigger**: gated by `{renamed}`.\n"

    found = invented_identifiers(page, get_default_config(), frozenset())

    assert renamed in found


@pytest.mark.unit
def test_a_misspelt_function_on_the_page_is_caught():
    page = (
        "## 1. Main loop\n\n"
        "- **File**: `run_reply_enginee()` and `get_auxilary_backend(cfg)`.\n"
    )

    found = invented_identifiers(page, get_default_config(), _source_identifiers())

    assert set(found) == {"run_reply_enginee", "get_auxilary_backend"}


@pytest.mark.unit
def test_the_checkers_leave_real_names_alone():
    defaults = get_default_config()
    page = (
        "## Config keys\n\n"
        "- Models: `llm_chat_model`, `intent_judge_model`\n"
        "- Timeouts: `llm_chat_timeout_sec` (180s)\n"
        "- Runtime: Ollama `keep_alive`, `warm_up_chat_model()`\n\n"
        "## 3. Extractor\n\n"
        "`get_auxiliary_backend(cfg, cfg.intent_judge_model)`, "
        "`DialogueMemory.record_tool_turn`, the `conversation_summaries` table.\n"
    )

    assert unknown_in_key_list(page, defaults) == {}
    assert unknown_setting_like_names(page, defaults) == {}
    assert invented_identifiers(page, defaults, _source_identifiers()) == {}


@pytest.mark.unit
def test_the_page_has_a_key_list():
    """The strict check reads the list by its heading, so it must find it."""
    section = _KEY_LIST.search(_prose(DOC.read_text(encoding="utf-8")))

    assert section is not None, "no '## Config keys' section: the check is blind"
    assert "`llm_chat_model`" in section.group(1)


@pytest.mark.unit
def test_every_key_in_the_key_list_is_a_setting():
    """The key list names what can be set; nothing in it may be made up."""
    unknown = unknown_in_key_list(
        DOC.read_text(encoding="utf-8"), get_default_config()
    )

    assert not unknown, (
        "the '## Config keys' list of docs/llm_contexts.md names keys that are "
        "not settings:\n"
        + "\n".join(f"  {name}   in `{span}`" for name, span in unknown.items())
    )


@pytest.mark.unit
def test_every_setting_the_page_names_exists():
    """A key the page cites must be one the settings define."""
    defaults = get_default_config()
    text = DOC.read_text(encoding="utf-8")
    cited = _cited_setting_names(text, defaults)

    assert cited, "no setting was found in the page: the extraction is broken"

    missing = unknown_setting_like_names(text, defaults)
    assert not missing, (
        "docs/llm_contexts.md names settings that do not exist. Either the key "
        "was renamed or removed, or the identifier is not a setting and should "
        "not read like one:\n"
        + "\n".join(f"  {name}   in `{span}`" for name, span in missing.items())
    )


@pytest.mark.unit
def test_no_name_on_the_page_is_invented():
    """A name that is not a setting and appears nowhere in the code is a typo."""
    invented = invented_identifiers(
        DOC.read_text(encoding="utf-8"), get_default_config(), _source_identifiers()
    )

    assert not invented, (
        "docs/llm_contexts.md names things that are neither a setting nor "
        "defined or used under src/jarvis:\n"
        + "\n".join(f"  {name}   in `{span}`" for name, span in invented.items())
    )


@pytest.mark.unit
def test_every_default_the_page_states_is_the_default():
    """``llm_chat_timeout_sec`` (180s) must say what the settings say."""
    defaults = get_default_config()
    checked = 0
    wrong: list[str] = []
    for match in _SETTING_NUMBER.finditer(_prose(DOC.read_text(encoding="utf-8"))):
        name = match.group(1) or match.group(3)
        cited = float(match.group(2) or match.group(4))
        actual = defaults.get(name)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            continue
        checked += 1
        if float(actual) != cited:
            wrong.append(f"  {name}: page says {cited:g}, settings say {actual:g}")

    assert checked, "no default was found in the page: the extraction is broken"
    assert not wrong, (
        "docs/llm_contexts.md states defaults the settings do not have:\n"
        + "\n".join(wrong)
    )


@pytest.mark.unit
def test_every_cap_the_page_states_is_the_constant():
    """``_DIGEST_MAX_CHARS`` (500) must say what the module says."""
    constants = _module_level_numbers()
    checked = 0
    problems: list[str] = []
    for match in _CONSTANT_NUMBER.finditer(_prose(DOC.read_text(encoding="utf-8"))):
        name, cited = match.group(1), float(match.group(2))
        values = constants.get(name)
        if values is None:
            problems.append(f"  {name}: no module under src/jarvis defines it")
            continue
        checked += 1
        if cited not in values:
            shown = ", ".join(f"{v:g}" for v in sorted(values))
            problems.append(f"  {name}: page says {cited:g}, code says {shown}")

    assert checked, "no cap was found in the page: the extraction is broken"
    assert not problems, (
        "docs/llm_contexts.md states caps the code does not have:\n"
        + "\n".join(problems)
    )


@pytest.mark.unit
def test_the_page_points_at_symbols_not_line_numbers():
    """A line number is wrong by the next commit; a function name is not.

    The page names where each context lives by function or constant, so a
    refactor that moves code leaves it true.
    """
    text = DOC.read_text(encoding="utf-8")
    found = [
        f"  line {number}: {line.strip()[:110]}"
        for number, line in enumerate(text.splitlines(), start=1)
        if _LINE_NUMBER.search(_FENCE.sub("", line))
    ]
    assert not found, (
        "docs/llm_contexts.md refers to source by line number; name the "
        "function or constant instead:\n" + "\n".join(found)
    )
