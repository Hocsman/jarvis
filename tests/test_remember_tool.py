"""Behavioural tests for the remember tool.

The tool is the only way an explicit "remember that..." or a correction
reaches the core, so these tests are about what ends up in the user's
files and what the assistant is told came of it.
"""

import pytest

from src.jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore
from src.jarvis.tools.builtin.remember import RememberTool


class _Cfg:
    def __init__(self, db_path):
        self.db_path = str(db_path)


class _Context:
    """Minimal stand-in for ToolContext with the fields the tool reads."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.db = None
        self.system_prompt = ""
        self.original_prompt = ""
        self.redacted_text = ""
        self.max_retries = 1
        self.printed: list[str] = []
        self.language = None

    def user_print(self, message):
        self.printed.append(message)


@pytest.fixture
def tool():
    return RememberTool()


@pytest.fixture
def context(tmp_path):
    return _Context(_Cfg(tmp_path / "jarvis.db"))


@pytest.fixture
def core(tmp_path):
    return MemoryCore(tmp_path / "yuba")


# ── Recording ─────────────────────────────────────────────────────────


def test_a_fact_lands_in_the_profile(tool, context, core):
    result = tool.run({"text": "Il s'appelle Hocine."}, context)

    assert result.success is True
    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il s'appelle Hocine."]


def test_a_rule_lands_in_the_rules(tool, context, core):
    result = tool.run({"text": "Répondre en français.", "kind": "rule"}, context)

    assert result.success is True
    assert [e.text for e in core.active(SECTION_RULES)] == ["Répondre en français."]
    assert core.active(SECTION_PROFILE) == []


def test_the_core_is_written_beside_the_database(tool, context, tmp_path):
    tool.run({"text": "Il s'appelle Hocine."}, context)

    assert (tmp_path / "yuba" / "profil.md").exists()


def test_the_assistant_is_told_what_was_recorded(tool, context):
    result = tool.run({"text": "Il s'appelle Hocine."}, context)

    assert "Il s'appelle Hocine." in result.reply_text


def test_the_user_sees_that_something_was_remembered(tool, context):
    tool.run({"text": "Il s'appelle Hocine."}, context)

    assert any("Il s'appelle Hocine." in line for line in context.printed)


def test_remembering_a_known_fact_says_so_without_failing(tool, context, core):
    tool.run({"text": "Il s'appelle Hocine."}, context)
    result = tool.run({"text": "Il s'appelle Hocine."}, context)

    assert result.success is True
    assert len(core.active(SECTION_PROFILE)) == 1
    assert "déjà" in result.reply_text.lower() or "already" in result.reply_text.lower()


def test_a_known_fact_is_echoed_back_as_it_is_stored(tool, context):
    tool.run({"text": "Il s'appelle Hocine."}, context)

    result = tool.run({"text": "il s'appelle hocine."}, context)

    assert "Il s'appelle Hocine." in result.reply_text
    assert any("Il s'appelle Hocine." in line for line in context.printed)


# ── Correcting ────────────────────────────────────────────────────────


def test_a_correction_replaces_what_was_believed(tool, context, core):
    tool.run({"text": "Il vit à Paris."}, context)

    tool.run({"text": "Il vit à Lyon.", "replaces": "Il vit à Paris."}, context)

    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il vit à Lyon."]


def test_a_correction_leaves_the_old_belief_visible_in_the_file(tool, context, core):
    tool.run({"text": "Il vit à Paris."}, context)
    tool.run({"text": "Il vit à Lyon.", "replaces": "Il vit à Paris."}, context)

    content = core.path_for(SECTION_PROFILE).read_text(encoding="utf-8")
    assert "Il vit à Paris." in content

    retired = [e for e in core.entries(SECTION_PROFILE) if e.retired]
    assert [e.text for e in retired] == ["Il vit à Paris."]


def test_a_correction_is_attributed_as_one(tool, context, core):
    tool.run({"text": "Il vit à Paris."}, context)
    tool.run({"text": "Il vit à Lyon.", "replaces": "Il vit à Paris."}, context)

    assert core.active(SECTION_PROFILE)[0].source == "corrigé"


def test_a_correction_finds_the_old_belief_in_the_other_section(tool, context, core):
    tool.run({"text": "Toujours répondre en anglais.", "kind": "rule"}, context)

    tool.run(
        {
            "text": "Toujours répondre en français.",
            "kind": "rule",
            "replaces": "Toujours répondre en anglais.",
        },
        context,
    )

    assert [e.text for e in core.active(SECTION_RULES)] == [
        "Toujours répondre en français.",
    ]


def test_a_correction_of_something_unknown_still_records_the_new_fact(
    tool, context, core,
):
    result = tool.run(
        {"text": "Il vit à Lyon.", "replaces": "Il vit sur Mars."}, context,
    )

    assert result.success is True
    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il vit à Lyon."]


# ── Refusals and failures ─────────────────────────────────────────────


def test_nothing_to_remember_is_a_failure_not_an_empty_entry(tool, context, core):
    result = tool.run({"text": "   "}, context)

    assert result.success is False
    assert core.active(SECTION_PROFILE) == []


def test_missing_arguments_are_a_failure(tool, context, core):
    result = tool.run(None, context)

    assert result.success is False
    assert core.active(SECTION_PROFILE) == []


def test_an_unknown_kind_is_treated_as_a_fact(tool, context, core):
    tool.run({"text": "Il s'appelle Hocine.", "kind": "banana"}, context)

    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il s'appelle Hocine."]


def test_a_write_failure_is_reported_rather_than_claimed_as_success(
    tool, context, monkeypatch,
):
    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)

    result = tool.run({"text": "Il s'appelle Hocine."}, context)

    assert result.success is False


# ── Reaching the next turn ────────────────────────────────────────────


def test_a_write_announces_itself_so_caches_can_drop(tool, context):
    from src.jarvis.memory import core as core_module

    seen = []

    def _listener(*, action, section):
        seen.append((action, section))

    core_module.register_core_mutation_listener(_listener)
    try:
        tool.run({"text": "Il s'appelle Hocine."}, context)
    finally:
        core_module.unregister_core_mutation_listener(_listener)

    assert ("remember", SECTION_PROFILE) in seen


# ── Refusing to store what redaction already removed ──────────────────


def test_a_redacted_value_is_not_stored(tool, context, core):
    """The model only ever sees redacted text, so "remember my email"
    reaches the tool as a placeholder. Writing that keeps nothing useful
    and leaves the user believing their email was saved."""
    result = tool.run({"text": "Son email est [REDACTED_EMAIL]."}, context)

    assert result.success is False
    assert core.active(SECTION_PROFILE) == []


def test_the_refusal_tells_the_model_what_to_say(tool, context):
    result = tool.run({"text": "Son email est [REDACTED_EMAIL]."}, context)

    assert "[REDACTED_EMAIL]" not in (result.reply_text or "")
    assert "sensitive" in (result.reply_text or "").lower()


def test_a_bare_placeholder_is_refused_too(tool, context, core):
    result = tool.run({"text": "Son mot de passe est password=[REDACTED]."}, context)

    assert result.success is False
    assert core.active(SECTION_PROFILE) == []


def test_a_rule_carrying_a_placeholder_is_refused(tool, context, core):
    result = tool.run(
        {"text": "Se connecter avec [REDACTED_TOKEN].", "kind": "rule"}, context,
    )

    assert result.success is False
    assert core.active(SECTION_RULES) == []


def test_ordinary_text_is_unaffected(tool, context, core):
    result = tool.run({"text": "Il vit à Lyon."}, context)

    assert result.success is True
    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il vit à Lyon."]
