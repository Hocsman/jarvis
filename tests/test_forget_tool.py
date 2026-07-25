"""Behavioural tests for the forget tool.

Forgetting is the one core operation where being helpful and being wrong
are close together: the user asks for something to go away, and the tool
has to find which line they meant among entries it can only match on
words. Retiring the wrong belief is worse than retiring none, because
the user is told it worked and walks away believing their memory is
clean.

So the shape these tests pin is: match confidently or not at all, never
guess between candidates, and never erase.
"""

import pytest

from src.jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore
from src.jarvis.tools.builtin.forget import ForgetTool


class _Cfg:
    def __init__(self, db_path):
        self.db_path = str(db_path)


class _Context:
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
    return ForgetTool()


@pytest.fixture
def context(tmp_path):
    return _Context(_Cfg(tmp_path / "jarvis.db"))


@pytest.fixture
def core(tmp_path):
    return MemoryCore(tmp_path / "yuba")


@pytest.fixture
def populated(core):
    core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-20")
    core.remember(SECTION_PROFILE, "Il est végétarien.", on_date="2026-07-21")
    core.remember(SECTION_RULES, "Toujours répondre en français.", on_date="2026-07-22")
    return core


# ── Forgetting what the user meant ────────────────────────────────────


def test_an_entry_quoted_exactly_is_forgotten(tool, context, populated):
    result = tool.run({"target": "Il vit à Lyon."}, context)

    assert result.success is True
    assert [e.text for e in populated.active(SECTION_PROFILE)] == ["Il est végétarien."]


def test_a_paraphrase_that_covers_the_entry_is_enough(tool, context, populated):
    """The model does not have to quote punctuation for punctuation, but
    what it passes has to account for the entry, not merely brush against
    it."""
    result = tool.run({"target": "il vit à lyon"}, context)

    assert result.success is True
    assert "Il vit à Lyon." not in [e.text for e in populated.active(SECTION_PROFILE)]


def test_a_bare_topic_word_drops_nothing(tool, context, populated):
    """"Lyon" appears in one entry, so a permissive matcher would drop it.
    One shared word is not evidence the user meant that line, and being
    told a drop worked is what stops them checking."""
    result = tool.run({"target": "Lyon"}, context)

    assert result.success is False
    assert len(populated.active(SECTION_PROFILE)) == 2


def test_a_sweeping_word_does_not_drop_an_entry_that_merely_contains_it(
    tool, context, core,
):
    core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-20")
    core.remember(SECTION_PROFILE, "Il mange de tout.", on_date="2026-07-21")

    result = tool.run({"target": "tout"}, context)

    assert result.success is False
    assert len(core.active(SECTION_PROFILE)) == 2


def test_shared_function_words_do_not_carry_a_match(tool, context, core):
    """"son adresse" reads as the postal one to a human. A matcher scoring
    on shared words alone lands on the email entry instead and drops it,
    which is both the wrong line and a more sensitive one."""
    core.remember(SECTION_PROFILE, "Il vit au 8 rue des Capucins à Lyon.")
    core.remember(SECTION_PROFILE, "Son adresse mail est hocsman92@gmail.com.")

    result = tool.run({"target": "son adresse"}, context)

    assert result.success is False
    assert len(core.active(SECTION_PROFILE)) == 2


def test_a_request_that_matches_nothing_names_what_is_held(tool, context, populated):
    """A dead end wastes the turn. Naming the entries lets the model call
    again with wording that will land."""
    result = tool.run({"target": "Lyon"}, context)

    assert "Il vit à Lyon." in result.reply_text
    assert "Il est végétarien." in result.reply_text


def test_a_rule_can_be_dropped(tool, context, populated):
    result = tool.run({"target": "Toujours répondre en français."}, context)

    assert result.success is True
    assert populated.active(SECTION_RULES) == []
    assert len(populated.active(SECTION_PROFILE)) == 2


def test_the_forgotten_line_stays_visible_in_the_file(tool, context, populated):
    tool.run({"target": "Il vit à Lyon."}, context)

    content = populated.path_for(SECTION_PROFILE).read_text(encoding="utf-8")
    assert "Il vit à Lyon." in content
    retired = [e for e in populated.entries(SECTION_PROFILE) if e.retired]
    assert [e.text for e in retired] == ["Il vit à Lyon."]


def test_the_assistant_is_told_what_was_dropped(tool, context, populated):
    result = tool.run({"target": "Il vit à Lyon."}, context)

    assert "Il vit à Lyon." in result.reply_text


def test_the_user_sees_what_was_dropped(tool, context, populated):
    tool.run({"target": "Il vit à Lyon."}, context)

    assert any("Il vit à Lyon." in line for line in context.printed)


# ── Refusing to guess ─────────────────────────────────────────────────


def test_an_ambiguous_request_drops_nothing(tool, context, core):
    core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-20")
    core.remember(SECTION_PROFILE, "Sa sœur vit à Lyon aussi.", on_date="2026-07-21")

    result = tool.run({"target": "Lyon"}, context)

    assert len(core.active(SECTION_PROFILE)) == 2


def test_an_ambiguous_request_reports_the_candidates(tool, context, core):
    core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-20")
    core.remember(SECTION_PROFILE, "Sa sœur vit à Lyon aussi.", on_date="2026-07-21")

    result = tool.run({"target": "Lyon"}, context)

    assert "Il vit à Lyon." in result.reply_text
    assert "Sa sœur vit à Lyon aussi." in result.reply_text


def test_a_request_matching_nothing_drops_nothing_and_says_so(tool, context, populated):
    result = tool.run({"target": "son abonnement à la salle de sport"}, context)

    assert len(populated.active(SECTION_PROFILE)) == 2
    assert len(populated.active(SECTION_RULES)) == 1
    assert result.success is False


def test_forgetting_the_same_thing_twice_reports_nothing_left_to_do(
    tool, context, populated,
):
    tool.run({"target": "Il vit à Lyon."}, context)

    result = tool.run({"target": "Il vit à Lyon."}, context)

    assert result.success is False
    retired = [e for e in populated.entries(SECTION_PROFILE) if e.retired]
    assert len(retired) == 1


def test_an_empty_target_is_refused(tool, context, populated):
    result = tool.run({"target": "   "}, context)

    assert result.success is False
    assert len(populated.active(SECTION_PROFILE)) == 2


def test_missing_arguments_are_refused(tool, context, populated):
    result = tool.run(None, context)

    assert result.success is False
    assert len(populated.active(SECTION_PROFILE)) == 2


def test_a_sweeping_request_does_not_empty_the_core(tool, context, populated):
    """"Forget everything" has no representation here on purpose: the tool
    drops one entry at a time. Wiping a memory the user spent months
    building must go through their own hands, on a file they can see."""
    for sweeping in ("tout", "everything", "all of it"):
        tool.run({"target": sweeping}, context)

    assert len(populated.active(SECTION_PROFILE)) == 2
    assert len(populated.active(SECTION_RULES)) == 1


# ── Failure is reported, never claimed as success ─────────────────────


def test_a_write_failure_is_reported(tool, context, populated, monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", _boom)

    result = tool.run({"target": "Il vit à Lyon."}, context)

    assert result.success is False


def test_an_empty_core_is_not_an_error_the_user_has_to_decode(tool, context, core):
    result = tool.run({"target": "Il vit à Lyon."}, context)

    assert result.success is False
    assert result.reply_text


# ── Reaching the next turn ────────────────────────────────────────────


def test_a_drop_announces_itself_so_caches_can_rebuild(tool, context, populated):
    from src.jarvis.memory import core as core_module

    seen = []

    def _listener(*, action, section):
        seen.append((action, section))

    core_module.register_core_mutation_listener(_listener)
    try:
        tool.run({"target": "Il vit à Lyon."}, context)
    finally:
        core_module.unregister_core_mutation_listener(_listener)

    assert ("retire", SECTION_PROFILE) in seen
