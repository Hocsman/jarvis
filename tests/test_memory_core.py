"""Behavioural tests for the core — the assistant's readable memory files.

The core is the one part of the memory system the user opens and edits by
hand, so these tests are written against what someone reading the files
would observe: entries appear, corrections leave a visible trail, hand
edits survive a rewrite, and nothing the user wrote is ever dropped.
"""

import pytest

from src.jarvis.memory.core import (
    SECTION_PROFILE,
    SECTION_RULES,
    MemoryCore,
    build_core_profile,
    format_warm_profile_block,
)


@pytest.fixture
def core(tmp_path):
    """A core rooted in an empty temporary directory."""
    return MemoryCore(tmp_path / "yuba")


# ── Reading an empty core ─────────────────────────────────────────────


def test_empty_core_reads_as_no_entries(core):
    assert core.entries(SECTION_PROFILE) == []
    assert core.entries(SECTION_RULES) == []


def test_empty_core_creates_no_files_until_written(core, tmp_path):
    core.entries(SECTION_PROFILE)
    assert not (tmp_path / "yuba").exists()


def test_empty_core_yields_an_empty_profile_block(core):
    profile = build_core_profile(core)
    assert profile == {"user": "", "directives": ""}


# ── Remembering ───────────────────────────────────────────────────────


def test_remembering_a_fact_makes_it_readable_back(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    texts = [e.text for e in core.entries(SECTION_PROFILE)]
    assert texts == ["Il s'appelle Hocine."]


def test_a_remembered_fact_carries_its_date_and_source(core):
    core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-25")

    entry = core.entries(SECTION_PROFILE)[0]
    assert entry.date == "2026-07-25"
    assert entry.source == "dit"
    assert entry.retired is False


def test_a_correction_is_attributed_as_a_correction(core):
    core.remember(
        SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-25", corrected=True,
    )

    assert core.entries(SECTION_PROFILE)[0].source == "corrigé"


def test_profile_and_rules_are_separate_files(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")
    core.remember(SECTION_RULES, "Répondre en français.", on_date="2026-07-25")

    assert [e.text for e in core.entries(SECTION_PROFILE)] == ["Il s'appelle Hocine."]
    assert [e.text for e in core.entries(SECTION_RULES)] == ["Répondre en français."]


def test_the_written_file_explains_its_own_format(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    content = core.path_for(SECTION_PROFILE).read_text(encoding="utf-8")
    # A user opening the file cold should find a heading and guidance,
    # not a bare list of bullets.
    assert content.lstrip().startswith("#")
    assert "<!--" in content


def test_remembering_the_same_fact_twice_changes_nothing(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")
    before = core.path_for(SECTION_PROFILE).read_text(encoding="utf-8")

    result = core.remember(
        SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-26",
    )

    assert result.already_known is True
    assert core.path_for(SECTION_PROFILE).read_text(encoding="utf-8") == before


def test_duplicate_detection_ignores_case_and_surrounding_whitespace(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    result = core.remember(
        SECTION_PROFILE, "  il s'appelle hocine.  ", on_date="2026-07-26",
    )

    assert result.already_known is True
    assert len(core.entries(SECTION_PROFILE)) == 1


def test_a_retired_fact_can_be_remembered_again(core):
    core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
    core.retire(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-25", reason="corrigé")

    result = core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-26")

    assert result.already_known is False
    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il vit à Paris."]


def test_empty_text_is_refused(core):
    with pytest.raises(ValueError):
        core.remember(SECTION_PROFILE, "   ", on_date="2026-07-25")


def test_newlines_in_an_entry_do_not_break_the_line_format(core):
    core.remember(
        SECTION_PROFILE, "Il vit à Lyon.\nIl y travaille.", on_date="2026-07-25",
    )

    entries = core.entries(SECTION_PROFILE)
    assert len(entries) == 1
    assert "\n" not in entries[0].raw
    assert "Il y travaille." in entries[0].text


# ── Correcting ────────────────────────────────────────────────────────


def test_retiring_an_entry_removes_it_from_what_the_assistant_believes(core):
    core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-18")

    core.retire(
        SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-25", reason="corrigé",
    )

    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il s'appelle Hocine."]


def test_a_retired_entry_stays_visible_in_the_file(core):
    core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
    core.retire(
        SECTION_PROFILE,
        "Il vit à Paris.",
        on_date="2026-07-25",
        reason="corrigé par l'utilisateur",
    )

    content = core.path_for(SECTION_PROFILE).read_text(encoding="utf-8")
    assert "Il vit à Paris." in content
    assert "2026-07-25" in content
    assert "corrigé par l'utilisateur" in content


def test_a_retired_entry_is_still_readable_as_history(core):
    core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
    core.retire(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-25", reason="corrigé")

    entry = core.entries(SECTION_PROFILE)[0]
    assert entry.retired is True
    assert entry.retired_on == "2026-07-25"
    assert entry.text == "Il vit à Paris."


def test_retiring_something_absent_reports_it_rather_than_writing(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")
    before = core.path_for(SECTION_PROFILE).read_text(encoding="utf-8")

    assert core.retire(
        SECTION_PROFILE, "Il vit à Mars.", on_date="2026-07-26", reason="corrigé",
    ) is False
    assert core.path_for(SECTION_PROFILE).read_text(encoding="utf-8") == before


def test_retiring_an_already_retired_entry_does_not_double_strike_it(core):
    core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
    core.retire(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-25", reason="corrigé")

    assert core.retire(
        SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-26", reason="encore",
    ) is False
    content = core.path_for(SECTION_PROFILE).read_text(encoding="utf-8")
    assert content.count("~~~~") == 0
    assert len(core.entries(SECTION_PROFILE)) == 1


# ── The user's own edits ──────────────────────────────────────────────


def test_a_hand_written_line_is_believed(core):
    path = core.path_for(SECTION_PROFILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Profil\n\n- Il déteste les réunions le lundi.\n", encoding="utf-8",
    )

    assert [e.text for e in core.active(SECTION_PROFILE)] == [
        "Il déteste les réunions le lundi.",
    ]


def test_a_hand_written_line_survives_a_later_write(core):
    path = core.path_for(SECTION_PROFILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Mon profil à moi\n\nUne note en prose.\n\n"
        "- Il déteste les réunions le lundi.\n",
        encoding="utf-8",
    )

    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    content = path.read_text(encoding="utf-8")
    assert "# Mon profil à moi" in content
    assert "Une note en prose." in content
    assert "Il déteste les réunions le lundi." in content
    assert "Il s'appelle Hocine." in content


def test_an_unreadable_core_yields_no_entries_rather_than_raising(core, monkeypatch):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    def _boom(*args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr("pathlib.Path.read_text", _boom)

    assert core.entries(SECTION_PROFILE) == []


def test_a_failed_write_is_reported_rather_than_silently_swallowed(core, monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)

    with pytest.raises(OSError):
        core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")


def test_a_core_that_cannot_be_read_is_never_overwritten(core, monkeypatch):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    def _boom(*args, **kwargs):
        raise OSError("read error")

    monkeypatch.setattr("pathlib.Path.read_text", _boom)

    with pytest.raises(OSError):
        core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-25")

    monkeypatch.undo()
    assert [e.text for e in core.active(SECTION_PROFILE)] == ["Il s'appelle Hocine."]


def test_a_failed_write_leaves_no_debris_behind(core, monkeypatch, tmp_path):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    import os as _os

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "replace", _boom)

    with pytest.raises(OSError):
        core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-25")

    assert sorted(p.name for p in (tmp_path / "yuba").iterdir()) == ["profil.md"]


# ── The block that reaches the prompt ─────────────────────────────────


def test_the_profile_block_carries_facts_and_rules_separately(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")
    core.remember(SECTION_RULES, "Répondre en français.", on_date="2026-07-25")

    profile = build_core_profile(core)

    assert "Il s'appelle Hocine." in profile["user"]
    assert "Il s'appelle Hocine." not in profile["directives"]
    assert "Répondre en français." in profile["directives"]


def test_retired_entries_never_reach_the_prompt(core):
    core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
    core.retire(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-25", reason="corrigé")

    assert "Paris" not in build_core_profile(core)["user"]


def test_the_prompt_sees_the_fact_not_its_bookkeeping(core):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    user_block = build_core_profile(core)["user"]
    assert "Il s'appelle Hocine." in user_block
    assert "2026-07-25" not in user_block
    assert "dit" not in user_block


def test_the_newest_facts_survive_the_character_budget(core):
    for day in range(1, 10):
        core.remember(
            SECTION_PROFILE, f"Fait numéro {day}.", on_date=f"2026-07-0{day}",
        )

    user_block = build_core_profile(core, user_max_chars=40)["user"]

    assert "Fait numéro 9." in user_block
    assert "Fait numéro 1." not in user_block
    assert len(user_block) <= 40


def test_a_written_core_survives_a_reopen(core, tmp_path):
    core.remember(SECTION_PROFILE, "Il s'appelle Hocine.", on_date="2026-07-25")

    reopened = MemoryCore(tmp_path / "yuba")

    assert [e.text for e in reopened.active(SECTION_PROFILE)] == ["Il s'appelle Hocine."]


# ── How the block is worded for the model ─────────────────────────────


def test_an_empty_core_adds_nothing_to_the_prompt():
    assert format_warm_profile_block({"user": "", "directives": ""}) == ""


def test_whitespace_only_sections_add_nothing_to_the_prompt():
    assert format_warm_profile_block({"user": "   \n", "directives": "\t"}) == ""


def test_facts_are_labelled_as_things_the_user_already_shared():
    out = format_warm_profile_block({"user": "Name is Hocine.", "directives": ""})

    assert "INFORMATION THE USER HAS SHARED" in out
    assert "STANDING INSTRUCTIONS" not in out
    assert "Hocine" in out


def test_rules_are_labelled_as_standing_instructions():
    out = format_warm_profile_block({"user": "", "directives": "Reply briefly."})

    assert "STANDING INSTRUCTIONS" in out
    assert "INFORMATION THE USER HAS SHARED" not in out
    assert "briefly" in out


def test_who_the_user_is_comes_before_what_they_asked_for():
    out = format_warm_profile_block(
        {"user": "Name is Hocine.", "directives": "Reply briefly."}
    )

    assert out.index("INFORMATION THE USER") < out.index("STANDING INSTRUCTIONS")
