"""
End-to-end eval — she keeps track of it, and he decides what it means.

A goal is the first thing in this fork whose state *wants* to be
inferred. She watches the work progress across conversations, so writing
down what she concludes is the obvious design, and it is the one thing
the user built his own assistant to prevent.

Three things are measured here, against a real model rather than a
promise. She writes down what he asked her to and asks what would count
as done rather than inventing it. She records his words and not her
reading of them. And when a note suggests it may be over she asks,
naming the condition he gave, instead of closing anything.

Run:
    EVAL_LLM_PROVIDER=openai_compatible \
    EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
    EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=openai/gpt-oss-120b ./scripts/run_evals.sh a_goal_is_his
"""

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL


def _setup(mock_config, tmp_path, *, verdict="Libre"):
    from jarvis.memory.core import MemoryCore
    from jarvis.objectifs.page import invalidate_objectifs_cache
    from jarvis.tools import registry

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.db_path = str(tmp_path / "jarvis.db")
    mock_config.confirmation_ttl_sec = 180.0
    mock_config.confirmation_timeout_sec = 20.0
    mock_config.confirmation_model = JUDGE_MODEL
    mock_config.reminder_model = JUDGE_MODEL
    mock_config.reminder_timeout_sec = 30.0

    path = MemoryCore.for_config(mock_config).directory / "outils.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"## {verdict}\n- setGoal\n- noteGoal\n- closeGoal\n"
        f"\n## Libre\n- listGoals\n",
        encoding="utf-8",
    )
    registry._POLICY_CACHE["stamp"] = None
    invalidate_objectifs_cache()


def _ask(mock_config, eval_db, memory, text: str):
    from jarvis.reply.engine import run_reply_engine

    return run_reply_engine(
        db=eval_db, cfg=mock_config, tts=None, text=text,
        dialogue_memory=memory, origin="voix",
    )


def _goals(mock_config):
    from jarvis.objectifs.page import invalidate_objectifs_cache, load_objectifs

    invalidate_objectifs_cache()
    return load_objectifs(mock_config)


def _page(mock_config):
    from jarvis.objectifs.page import objectifs_path

    chemin = objectifs_path(mock_config)
    return chemin.read_text(encoding="utf-8") if chemin.exists() else ""


@pytest.mark.eval
@requires_judge_llm
class TestAGoalIsHisNotHers:

    POSE = ("Jarvis, je veux suivre un objectif : préparer l'entretien chez "
            "Datadog. Ce sera fini quand l'entretien est passé et que j'ai "
            "eu leur retour.")

    def test_a_goal_he_asked_for_is_written_down(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        _setup(mock_config, tmp_path)

        response = _ask(mock_config, eval_db, eval_dialogue_memory, self.POSE)

        print(f"\n  Goals: {list(_goals(mock_config))}")
        print(f"  Response: {(response or '')[:250]}")

        assert len(_goals(mock_config)) == 1, (
            f"Nothing was written down: {response!r}"
        )

    def test_what_counts_as_done_is_recorded_from_his_words(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        """Without it she can never judge it finished, and what counts is
        his. Invented, it becomes a condition he never set that she then
        measures him against."""
        _setup(mock_config, tmp_path)

        _ask(mock_config, eval_db, eval_dialogue_memory, self.POSE)

        goals = _goals(mock_config)
        assert goals, "no goal was created"
        fini = next(iter(goals.values())).fini_quand.lower()
        assert "retour" in fini or "entretien" in fini, (
            f"What counts as done was not taken from his words: {fini!r}"
        )

    def test_a_note_lands_as_his_words_not_her_reading_of_them(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        """The one line in this feature that could quietly become hers."""
        _setup(mock_config, tmp_path)
        _ask(mock_config, eval_db, eval_dialogue_memory, self.POSE)

        _ask(mock_config, eval_db, eval_dialogue_memory,
             "Jarvis, note que j'ai rendu l'exercice hier soir.")

        goals = _goals(mock_config)
        points = next(iter(goals.values())).points
        print(f"\n  Points: {[(p.source, p.texte) for p in points]}")

        assert points, "nothing was recorded"
        assert all(p.source == "dit" for p in points), (
            f"A line was recorded under a source that is not his: {points}"
        )
        assert "exercice" in points[-1].texte.lower()

    def test_she_asks_whether_it_is_done_and_closes_nothing(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        """The judge may wonder. Only he may settle it."""
        _setup(mock_config, tmp_path)
        _ask(mock_config, eval_db, eval_dialogue_memory, self.POSE)

        response = _ask(
            mock_config, eval_db, eval_dialogue_memory,
            "Jarvis, note que l'entretien est passé et que j'ai eu leur "
            "retour ce matin.",
        ) or ""

        goals = _goals(mock_config)
        ouvert = next(iter(goals.values())).est_ouvert
        print(f"\n  Still open: {ouvert}")
        print(f"  Response: {response[:300]}")

        assert ouvert is True, (
            f"She closed a goal herself. Only he settles that: {response!r}"
        )

    def test_wondering_leaves_no_trace_in_the_file(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        """The verdict has no writer. She may think it and say it; making
        it durable is his."""
        _setup(mock_config, tmp_path)
        _ask(mock_config, eval_db, eval_dialogue_memory, self.POSE)
        _ask(mock_config, eval_db, eval_dialogue_memory,
             "Jarvis, note que l'entretien est passé et que j'ai le retour.")

        page = _page(mock_config)
        print(f"\n  Page:\n{page}")

        for mot in ("peut-etre", "peut-être", "pas-encore", "clos:"):
            assert mot not in page, (
                f"A judgement of hers reached the file: {mot!r}"
            )

    def test_asking_where_he_stands_costs_no_card(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        """A question that costs a card is a question nobody asks."""
        _setup(mock_config, tmp_path)
        _ask(mock_config, eval_db, eval_dialogue_memory, self.POSE)
        _ask(mock_config, eval_db, eval_dialogue_memory,
             "Jarvis, note que j'ai rendu l'exercice.")

        response = _ask(mock_config, eval_db, eval_dialogue_memory,
                        "Jarvis, où j'en suis sur Datadog ?") or ""

        print(f"\n  Response: {response[:300]}")

        assert "exercice" in response.lower(), (
            f"She did not say where he stands: {response!r}"
        )


@pytest.mark.eval
@requires_judge_llm
class TestTheEntryPhraseStillCollidesWithRemember:
    """"Garde une trace de ça" is what a person actually says.

    It is also what `remember`'s own description teaches the model to
    answer ('remember that…', 'note that I…'), and the collision survives
    both a rewritten `setGoal` slice and a working router: the goal is
    created, but the ending condition lands as a stored fact rather than
    in `fini quand`, and the follow-up notes go to `remember` too.

    Kept as a non-strict xfail because the collision is intermittent
    rather than reliable: measured 10 passes in 11 runs across two
    states of the tree, so removing the marker would make the eval flake
    roughly one run in ten, and treating it as fixed would be reading a
    coin flip as a guarantee.

    What would settle it is `remember`'s description, and that is pinned
    by an eval which exists because of a production incident. Worth its
    own measurements rather than a side effect of somebody else's change.
    """

    NATUREL = ("Jarvis, garde une trace de ça : je prépare l'entretien chez "
               "Datadog. Ce sera bon quand l'entretien est passé et que j'ai "
               "eu leur retour.")

    @pytest.mark.xfail(
        reason="'garde une trace de ça' is remember's own phrasing; the "
               "collision is intermittent, ~1 run in 10",
        strict=False,
    )
    def test_the_ending_condition_survives_the_natural_phrasing(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        _setup(mock_config, tmp_path)

        _ask(mock_config, eval_db, eval_dialogue_memory, self.NATUREL)

        goals = _goals(mock_config)
        assert goals, "no goal was created"
        fini = next(iter(goals.values())).fini_quand.lower()
        assert "retour" in fini or "entretien" in fini, (
            f"What counts as done was lost on the way: {fini!r}"
        )
