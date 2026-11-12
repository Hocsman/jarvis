"""She keeps the thread of a long conversation.

A conversation is the run of messages with no pause of
``dialogue_memory_timeout`` between two of them. Its latest part reaches the
model word for word, within ``dialogue_context_max_chars``; its earlier part
reaches it as the diary's summary of the days it spans. Two hours of talk
with a message every minute are one conversation: what was said at the
start is not forgotten because five minutes have passed since.

See ``src/jarvis/reply/reply.spec.md``, Recent Dialogue Context.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory, earlier_in_conversation
from src.jarvis.reply.prompts.model_variants import dialogue_context_budget


class _Clock:
    """The wall clock, moved by hand."""

    def __init__(self, at: datetime):
        self.t = at.timestamp()

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    c = _Clock(datetime(2026, 10, 31, 22, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(time, "time", c)
    return c


def _talk(dm, clock, exchanges, *, every=60.0, size=80, first=0):
    """``exchanges`` question/answer pairs, one every ``every`` seconds."""
    for i in range(first, first + exchanges):
        dm.add_message("user", f"question {i} " + "q" * size)
        clock.advance(5)
        dm.add_message("assistant", f"answer {i} " + "a" * size)
        clock.advance(every - 5)


def _texts(messages):
    return [m["content"] for m in messages]


def _saved(dm):
    _, snapshot = dm.get_pending_chunks_with_snapshot()
    dm.mark_saved_up_to(snapshot)


# ── The conversation lasts as long as nobody pauses ─────────────────────


@pytest.mark.unit
class TestTheConversation:

    def test_a_conversation_that_never_pauses_is_kept_whole(self, clock):
        """Forty minutes, one exchange a minute: the first question is
        still there at the end, though it is far older than the timeout."""
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 40)

        messages = dm.get_recent_messages()

        assert len(messages) == 80
        assert messages[0]["content"].startswith("question 0 ")
        assert messages[-1]["content"].startswith("answer 39 ")

    def test_a_pause_as_long_as_the_timeout_ends_it(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 3)
        clock.advance(301)
        _talk(dm, clock, 2, first=10)

        messages = dm.get_recent_messages()

        assert [t.split(" ")[1] for t in _texts(messages)] == ["10", "10", "11", "11"]
        assert dm.earlier_part_start() is None

    def test_after_a_pause_there_is_no_conversation(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0)
        _talk(dm, clock, 2, every=10)
        clock.advance(300)

        assert not dm.has_recent_messages()
        assert dm.get_recent_messages() == []
        assert dm.earlier_part_start() is None

    def test_a_diary_save_keeps_the_conversation_still_going(self, clock):
        """The diary is written every few minutes while they talk; writing
        it must not take the start of the conversation away from her."""
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 30)
        _saved(dm)
        _talk(dm, clock, 1, first=30)

        messages = dm.get_recent_messages()

        assert messages[0]["content"].startswith("question 0 ")
        assert len(messages) == 62

    def test_a_diary_save_forgets_conversations_that_have_ended(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0)
        dm.add_message("user", "an old conversation")
        clock.advance(600)
        dm.add_message("user", "the one going on")
        _saved(dm)

        assert not dm.rewind_before_user_message(1, "an old conversation")
        assert _texts(dm.get_recent_messages()) == ["the one going on"]

    def test_an_unsaved_message_of_an_ended_conversation_waits_for_the_diary(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0)
        dm.add_message("user", "said before the pause")
        clock.advance(600)
        dm.add_message("user", "said after it")
        dm.mark_saved_up_to(0.0)

        assert any("said before the pause" in c for c in dm.get_pending_chunks())


# ── Its latest part, word for word ──────────────────────────────────────


@pytest.mark.unit
class TestTheWordForWordPart:

    def test_the_latest_messages_fill_the_budget(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)

        messages = dm.get_recent_messages()

        assert sum(len(t) for t in _texts(messages)) <= 1_000
        assert messages[-1]["content"].startswith("answer 29 ")
        # The latest ones, in order, none skipped.
        numbers = [int(t.split(" ")[1]) for t in _texts(messages)]
        assert numbers == sorted(numbers)
        assert numbers[-1] - numbers[0] + 1 == len(set(numbers))

    def test_the_part_handed_over_opens_on_the_user(self, clock):
        """A cut that would start on her answer starts on the next
        question instead: the model is never handed an answer without the
        words it answered."""
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        dm.add_message("user", "u" * 100)
        dm.add_message("assistant", "a" * 600)
        dm.add_message("user", "the latest question")
        dm.add_message("assistant", "a" * 300)
        _saved(dm)

        messages = dm.get_recent_messages()

        assert messages[0] == {"role": "user", "content": "the latest question"}

    def test_a_conversation_that_opens_on_her_words_keeps_them_when_it_fits(self, clock):
        """A reminder said aloud, or a delegated job's report, can open a
        conversation. Nothing is cut, so nothing is dropped."""
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        dm.add_message("assistant", "Rappel : sortir le plat du four.")
        dm.add_message("user", "décale-le de dix minutes")

        assert _texts(dm.get_recent_messages()) == [
            "Rappel : sortir le plat du four.",
            "décale-le de dix minutes",
        ]
        assert dm.earlier_part_start() is None

    def test_the_last_exchange_survives_a_budget_too_small_for_it(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=10)
        dm.add_message("user", "an earlier question")
        dm.add_message("assistant", "an earlier answer")
        dm.add_message("user", "a question longer than the budget")
        dm.add_message("assistant", "and an even longer answer to it")

        assert _texts(dm.get_recent_messages()) == [
            "a question longer than the budget",
            "and an even longer answer to it",
        ]

    def test_what_the_diary_has_not_read_is_not_cut(self, clock):
        """Cut before the diary has read it, a message would be in neither
        the words she reads nor the summary."""
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 20)
        _saved(dm)
        _talk(dm, clock, 8, first=20)

        texts = _texts(dm.get_recent_messages())

        assert sum(len(t) for t in texts) > 1_000
        assert texts[0].startswith("question 20 ")

    def test_what_the_diary_has_not_read_stops_at_twice_the_budget(self, clock):
        """A diary that keeps failing does not let the words she reads grow
        without end."""
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)

        texts = _texts(dm.get_recent_messages())

        assert sum(len(t) for t in texts) <= 2 * 1_000
        assert texts[-1].startswith("answer 29 ")

    def test_tool_turns_still_join_the_part_handed_over(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 10)
        dm.add_message("user", "look it up")
        dm.record_tool_turn([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "webSearch", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "the result"},
        ])
        dm.add_message("assistant", "here it is")

        turns = dm.get_recent_turns_with_tools()

        assert turns[0]["content"].startswith("question 0 ")
        assert [m.get("content") for m in turns[-3:]] == ["", "the result", "here it is"]


# ── Its earlier part ────────────────────────────────────────────────────


@pytest.mark.unit
class TestTheEarlierPart:

    def test_a_conversation_that_fits_has_no_earlier_part(self, clock):
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 20)

        assert dm.earlier_part_start() is None

    def test_a_conversation_that_does_not_fit_says_when_it_began(self, clock):
        began = clock()
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)

        assert dm.earlier_part_start() == pytest.approx(began, abs=1e-3)

    def test_it_still_knows_when_it_began_after_a_diary_save(self, clock):
        began = clock()
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)
        _saved(dm)

        assert dm.earlier_part_start() == pytest.approx(began, abs=1e-3)

    def test_when_the_days_do_not_all_fit_the_latest_is_kept(self, db, monkeypatch):
        """Across midnight the summaries of both days may not fit; the
        one nearest the words she reads is the one kept."""
        clock = _Clock(datetime(2026, 10, 31, 23, 40, tzinfo=timezone.utc))
        monkeypatch.setattr(time, "time", clock)
        db.upsert_conversation_summary(
            "2026-10-31", " ".join(["Avant minuit, ils ont parlé."] * 14), source_app="voice")
        db.upsert_conversation_summary(
            "2026-11-01", " ".join(["Après minuit, ils ont parlé."] * 14), source_app="voice")
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 40)

        earlier = earlier_in_conversation(db, dm)

        assert len(earlier) <= 500
        assert "Après minuit, ils ont parlé." in earlier

    def test_a_summary_written_without_spaces_is_cut_at_a_sentence(self, clock, db):
        """Chinese and Japanese put no space after a full stop; the cut
        still falls at the end of a sentence."""
        db.upsert_conversation_summary(
            "2026-10-31", "彼は友人の話をした。" * 80, source_app="voice")
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)

        earlier = earlier_in_conversation(db, dm)

        assert 0 < len(earlier) <= 500
        assert earlier.endswith("話をした。")


# ── The diary reads every message ───────────────────────────────────────


def _flush(db, dm, summaries):
    """One diary pass, the summariser answering from ``summaries`` (a
    ``(summary, topics)`` pair per call, ``(None, None)`` for a failure);
    returns the chunks each call was handed."""
    from src.jarvis.memory import conversation

    handed = []

    def _summarise(chunks, previous, cfg, **_):
        handed.append(list(chunks))
        return summaries.pop(0) if summaries else ("Résumé.", "sujets")

    cfg = SimpleNamespace(embedding_model="", llm_chat_model="test-large")
    with patch.object(conversation, "generate_conversation_summary", side_effect=_summarise), \
         patch.object(conversation, "_learn_from_the_day", return_value=None):
        conversation.update_diary_from_dialogue_memory(db, dm, cfg, force=True)
    return handed


@pytest.mark.unit
class TestTheDiaryReadsEveryMessage:

    def test_a_pass_hands_every_waiting_message_to_the_summariser(self, clock, db):
        """Eighteen exchanges waiting: the first question reaches the
        summariser, not only the latest ones."""
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 18, every=40)

        handed = _flush(db, dm, [])

        seen = [c for call in handed for c in call]
        assert any(c.startswith("User: question 0 ") for c in seen)
        assert len(seen) == 36
        assert not dm.has_pending_chunks()

    def test_what_the_summariser_never_read_stays_waiting(self, clock, db):
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 18, every=40)

        handed = _flush(db, dm, [("Résumé.", "sujets"), (None, None)])

        read = {c for c in handed[0]}
        waiting = dm.get_pending_chunks()
        assert waiting
        assert not read & set(waiting)
        assert any(c.startswith("User: question 17 ") for c in waiting)


# ── The budget ──────────────────────────────────────────────────────────


def _settings(**kw):
    """Settings with the chat model on a provider off home."""
    values = dict(
        dialogue_context_max_chars=0,
        llm_provider="openai_compatible",
        llm_base_url="https://api.example.com/v1",
        ollama_base_url="http://127.0.0.1:11434",
        llm_chat_model="vendor/large-model",
        ollama_chat_model="",
    )
    values.update(kw)
    return SimpleNamespace(**values)


@pytest.mark.unit
class TestTheBudget:

    def test_a_configured_budget_is_the_one_used(self):
        assert dialogue_context_budget(_settings(dialogue_context_max_chars=5_000)) == 5_000

    def test_left_at_zero_a_small_model_reads_less_than_a_large_one(self):
        small = _settings(llm_chat_model="vendor/qwen2.5:3b")
        large = _settings(llm_chat_model="vendor/large-model")

        assert 0 < dialogue_context_budget(small) < dialogue_context_budget(large)

    def test_a_model_run_at_home_reads_what_its_window_holds(self):
        """Ollama runs every chat model in the same window, whatever its
        size: a large local model reads no more than a small one."""
        local_large = _settings(llm_provider="ollama", llm_chat_model="gpt-oss:20b")
        local_small = _settings(llm_provider="ollama", llm_chat_model="gemma4:e2b")
        remote_large = _settings()

        assert dialogue_context_budget(local_large) == dialogue_context_budget(local_small)
        assert dialogue_context_budget(local_large) < dialogue_context_budget(remote_large)

    def test_the_settings_default_follows_the_model(self):
        from src.jarvis.config import get_default_config

        default = get_default_config()["dialogue_context_max_chars"]
        small = _settings(dialogue_context_max_chars=default,
                          llm_chat_model="vendor/qwen2.5:3b")
        large = _settings(dialogue_context_max_chars=default)

        assert dialogue_context_budget(small) != dialogue_context_budget(large)

    @pytest.mark.parametrize("written", [-1, "beaucoup"])
    def test_a_budget_that_is_not_a_size_is_left_to_the_model(
        self, written, tmp_path, monkeypatch,
    ):
        import json

        from src.jarvis.config import load_settings

        path = tmp_path / "config.json"
        path.write_text(json.dumps({"dialogue_context_max_chars": written}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))

        assert load_settings().dialogue_context_max_chars == 0


# ── What the model is handed ────────────────────────────────────────────


def _cfg(tmp_path):
    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = "test-large"
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 8
    cfg.memory_enrichment_source = "diary"
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.tool_search_max_calls = 3
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_tools_timeout_sec = 8.0
    return cfg


@pytest.fixture
def db():
    from src.jarvis.memory.db import Database

    d = Database(":memory:", sqlite_vss_path=None)
    yield d
    d.close()


def _handed_to_the_model(db, cfg, dm, text="et du coup, c'était qui ?", **kw):
    from src.jarvis.reply.engine import run_reply_engine

    seen = {}

    def _capture(**call):
        seen.setdefault("messages", list(call.get("messages") or []))
        return {"message": {"content": "ok"}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="ok"):
        run_reply_engine(db=db, cfg=cfg, tts=None, text=text, dialogue_memory=dm, **kw)
    return seen["messages"]


@pytest.mark.unit
class TestWhatTheModelIsHanded:

    def test_the_earlier_part_arrives_as_the_diary_summary(self, clock, db, tmp_path):
        db.upsert_conversation_summary(
            "2026-10-31", "Nadia a changé d'agent, a-t-il dit au début.", source_app="voice")
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)
        _saved(dm)

        messages = _handed_to_the_model(db, _cfg(tmp_path), dm)

        system = messages[0]["content"]
        assert "Nadia a changé d'agent" in system
        assert "earlier in this conversation" in system.lower()
        prior = messages[1:-1]
        assert prior[0]["role"] == "user"
        assert sum(len(m["content"]) for m in prior) <= 1_000

    def test_a_conversation_that_fits_is_handed_over_without_a_summary(self, clock, db, tmp_path):
        db.upsert_conversation_summary(
            "2026-10-31", "Un matin sans rapport avec ce qui suit.", source_app="voice")
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=100_000)
        _talk(dm, clock, 5)

        messages = _handed_to_the_model(db, _cfg(tmp_path), dm)

        assert "Un matin sans rapport" not in messages[0]["content"]
        assert messages[1]["content"].startswith("question 0 ")

    def test_a_conversation_begun_yesterday_brings_both_days(self, db, tmp_path, monkeypatch):
        clock = _Clock(datetime(2026, 10, 31, 23, 40, tzinfo=timezone.utc))
        monkeypatch.setattr(time, "time", clock)
        db.upsert_conversation_summary("2026-10-30", "AVANT-HIER.", source_app="voice")
        db.upsert_conversation_summary("2026-10-31", "HIER SOIR.", source_app="voice")
        db.upsert_conversation_summary("2026-11-01", "APRÈS MINUIT.", source_app="voice")
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 40)
        assert datetime.fromtimestamp(clock(), timezone.utc).date().isoformat() == "2026-11-01"

        system = _handed_to_the_model(db, _cfg(tmp_path), dm)[0]["content"]

        assert "HIER SOIR." in system
        assert "APRÈS MINUIT." in system
        assert system.index("HIER SOIR.") < system.index("APRÈS MINUIT.")
        assert "AVANT-HIER." not in system

    def test_the_summary_is_cut_at_a_sentence_within_half_the_budget(self, clock, db, tmp_path):
        sentences = " ".join(f"Phrase {i:03d} du résumé." for i in range(200))
        db.upsert_conversation_summary("2026-10-31", sentences, source_app="voice")
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)

        system = _handed_to_the_model(db, _cfg(tmp_path), dm)[0]["content"]

        start = system.index("Phrase 000 du résumé.")
        summary = system[start:].split("\n")[0]
        assert len(summary) <= 500
        assert summary.endswith("du résumé.")
        assert "Phrase 199" not in system

    def test_a_routine_turn_is_never_handed_the_day(self, clock, db, tmp_path):
        from src.jarvis.routines.scope import RoutineScope

        db.upsert_conversation_summary(
            "2026-10-31", "Ce qu'il a raconté ce soir.", source_app="voice")
        # A conversation the day's summary would continue, were this not a
        # routine.
        dm = DialogueMemory(inactivity_timeout=300.0, context_max_chars=1_000)
        _talk(dm, clock, 30)
        _saved(dm)
        assert dm.earlier_part_start() is not None

        for memoire in (False, True):
            messages = _handed_to_the_model(
                db, _cfg(tmp_path), dm, text="résume les mails", origin="routine",
                scope=RoutineScope(nom="matin", outils=["webSearch"], memoire=memoire),
            )

            assert "Ce qu'il a raconté ce soir." not in messages[0]["content"]
