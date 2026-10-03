"""``docs/llm_contexts.md`` is the reference for every LLM call, so the
settings, timeouts and caps it names have to be the ones the code has.

The page is prose, and prose drifts: a key is renamed and the page keeps
the old name, a default changes and the page keeps the old number. Nothing
at runtime notices, because nothing reads the page. These tests read it the
way a person does and hold it to the code:

* an identifier written as a setting must be a setting;
* a number written next to a setting must be that setting's default;
* a number written next to a constant must be that constant's value;
* a place in the code is named by a symbol, never by a line number.

The first three compare against the code at test time, so a default that is
changed on purpose fails here until the page follows, and nothing in this
file has to be edited when it does.
"""

from __future__ import annotations

import ast
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


def _prose() -> str:
    return _FENCE.sub("", DOC.read_text(encoding="utf-8"))


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


def _cited_setting_names(defaults: dict) -> dict[str, str]:
    """Every identifier the page presents as a setting, with where it is."""
    suffixes = _setting_suffixes(defaults)
    cited: dict[str, str] = {}
    for span in _SPAN.findall(_prose()):
        for name in _CFG_ATTRIBUTE.findall(span):
            cited.setdefault(name, span)
        without_attributes = _CFG_ATTRIBUTE.sub(" ", span)
        for word in _BARE_WORD.findall(without_attributes):
            if word.rsplit("_", 1)[1] in suffixes:
                cited.setdefault(word, span)
    return cited


def _module_level_numbers() -> dict[str, set[float]]:
    """Every module-level ``NAME = <number>`` under ``src/jarvis``."""
    found: dict[str, set[float]] = {}
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
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


@pytest.mark.unit
def test_every_setting_the_page_names_exists():
    """A key the page cites must be one the settings define.

    This is the check a phantom key fails: a function parameter written as
    if it were a key, which no config file could ever set.
    """
    defaults = get_default_config()
    cited = _cited_setting_names(defaults)

    assert cited, "no setting was found in the page: the extraction is broken"

    missing = {
        name: span for name, span in sorted(cited.items()) if name not in defaults
    }
    assert not missing, (
        "docs/llm_contexts.md names settings that do not exist. Either the key "
        "was renamed or removed, or the identifier is not a setting and should "
        "not read like one:\n"
        + "\n".join(f"  {name}   in `{span}`" for name, span in missing.items())
    )


@pytest.mark.unit
def test_every_default_the_page_states_is_the_default():
    """``llm_chat_timeout_sec`` (180s) must say what the settings say."""
    defaults = get_default_config()
    checked = 0
    wrong: list[str] = []
    for match in _SETTING_NUMBER.finditer(_prose()):
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
    for match in _CONSTANT_NUMBER.finditer(_prose()):
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
