"""Nothing the repository says about LLM providers outruns the factory.

A registry row, a spec, a docstring and a config comment each told the reader
that Anthropic-compatible servers are supported. The factory only builds
Ollama and OpenAI-compatible backends, so someone who set up the third one
would have found out at request time, in a log line.

The rule is about the claim, not the word: an Anthropic model used as an
example is fine. What must not appear is Anthropic named as a *backend* the
project can talk to, until a module that implements it exists. The day one
does, this test steps aside by itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_PACKAGE = REPO_ROOT / "src" / "jarvis" / "llm"

# Phrases that present Anthropic as a wire shape or a provider the code speaks.
_BACKEND_CLAIM = re.compile(
    r"anthropic[- ]compatible|anthropic backend|anthropic sse|/v1/messages",
    re.IGNORECASE,
)

_TEXT_SUFFIXES = {".md", ".py", ".json", ".yml", ".yaml", ".txt"}


def _files_that_describe_the_project():
    """Instruction files, docs, examples, and everything under src/."""
    for name in ("CLAUDE.md", "AGENTS.md", "README.md"):
        yield REPO_ROOT / name
    for folder in ("docs", "examples", "src"):
        for path in (REPO_ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in _TEXT_SUFFIXES:
                yield path


def _a_backend_for_it_exists() -> bool:
    return any(LLM_PACKAGE.glob("anthropic*.py"))


def test_the_scan_reads_the_files_it_is_meant_to():
    """Guard the guard: an empty scan would pass for the wrong reason."""
    scanned = list(_files_that_describe_the_project())

    assert REPO_ROOT / "CLAUDE.md" in scanned
    assert LLM_PACKAGE / "factory.py" in scanned


def test_naming_an_unbuilt_provider_in_config_gives_ollama():
    """The provider set the factory serves is the set the docs may claim.

    Selecting a provider that has no backend must not reach a made-up one: it
    lands on the Ollama default, which is what the docs can honestly describe.
    """
    from types import SimpleNamespace

    from jarvis.llm import OllamaBackend, OpenAICompatibleBackend
    from jarvis.llm.factory import clear_backend_cache, get_llm_backend

    def backend_for(provider):
        clear_backend_cache()
        return get_llm_backend(SimpleNamespace(
            llm_provider=provider,
            ollama_base_url="http://127.0.0.1:11434",
            llm_base_url="http://127.0.0.1:1234/v1",
        ))

    assert isinstance(backend_for("ollama"), OllamaBackend)
    assert isinstance(backend_for("openai_compatible"), OpenAICompatibleBackend)
    assert isinstance(backend_for("anthropic"), OllamaBackend)


@pytest.mark.skipif(
    _a_backend_for_it_exists(),
    reason="a backend module for it exists, so the claim is true",
)
def test_no_file_presents_anthropic_as_a_backend():
    offenders = []
    for path in _files_that_describe_the_project():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if _BACKEND_CLAIM.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{number}")

    assert not offenders, (
        "These lines name an Anthropic backend the factory cannot build: "
        + ", ".join(offenders)
        + ". Remove the claim, or add the backend.")
