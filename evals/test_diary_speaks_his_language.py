"""
The diary is his to read, so it is written in the language he spoke.

The summariser prompt is long, detailed, entirely in English, and says
nothing at all about which language to write the summary in. A model
handed English instructions writes English, and measured on a real
French-speaking machine that is what happens: two of the ten most recent
rows are in French, and one of the English ones reads

    "The user conversed in French with the assistant, who responded in
    French."

Three things read these rows and none of them is a machine that does not
care. He can open the diary himself in the memory viewer. The graph
extractor turns them into stored knowledge that comes back in a reply.
And the learning step reads them to propose lines for `profil.md`, a
French file he hand-edits, so a summariser writing English hands him
English sentences to accept about himself.

The rule has to be stated without naming a language, because the point is
not French. A conversation is summarised in the language it happened in,
whichever that is, and a summary that translates loses the words he
actually used.
"""

from dataclasses import dataclass
from typing import Tuple

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL, MockConfig


# The summariser's own English idiom. Its presence in a summary of a
# conversation held in another language is the defect itself: these are
# the two nouns every rule in the prompt is written about, so they are
# what leaks when the model answers in the prompt's language rather than
# the conversation's.
_ENGLISH_TELLS = ("the user", "the assistant", "user asked", "user said")


@dataclass
class LanguageCase:
    langue: str
    chunks: Tuple[str, ...]
    # Function words that a summary in this language can hardly avoid.
    attendus: Tuple[str, ...]


CASES = [
    pytest.param(
        LanguageCase(
            langue="français",
            chunks=(
                "Utilisateur : Yuba, je pars à Lyon vendredi pour l'entretien "
                "chez Datadog.",
                "Assistant : Bonne chance. Tu veux que je note quelque chose ?",
                "Utilisateur : Oui, je préfère prendre le train plutôt que "
                "l'avion, c'est plus reposant.",
            ),
            attendus=("utilisateur", "il ", "le ", "à "),
        ),
        id="français",
    ),
    pytest.param(
        LanguageCase(
            langue="español",
            chunks=(
                "Usuario: Yuba, el viernes voy a Lyon para una entrevista.",
                "Asistente: Buena suerte.",
                "Usuario: Prefiero el tren al avión, es más tranquilo.",
            ),
            attendus=("usuario", "el ", "de "),
        ),
        id="español",
    ),
    pytest.param(
        LanguageCase(
            langue="türkçe",
            chunks=(
                "Kullanıcı: Yuba, cuma günü bir görüşme için Lyon'a gidiyorum.",
                "Asistan: Bol şans.",
                "Kullanıcı: Uçak yerine treni tercih ederim, daha sakin.",
            ),
            attendus=("kullanıcı", "bir "),
        ),
        id="türkçe",
    ),
]


def _summarise(chunks) -> str:
    from jarvis.memory.conversation import generate_conversation_summary

    cfg = MockConfig()
    cfg.llm_chat_model = JUDGE_MODEL
    summary, _topics = generate_conversation_summary(
        recent_chunks=list(chunks), previous_summary=None, cfg=cfg,
        timeout_sec=60.0,
    )
    return (summary or "").lower()


@pytest.mark.eval
@requires_judge_llm
class TestTheDiaryKeepsTheLanguageItWasSpokenIn:

    @pytest.mark.parametrize("case", CASES)
    def test_the_summary_is_in_the_language_of_the_conversation(self, case):
        resume = _summarise(case.chunks)

        print(f"\n  [{case.langue}] {resume[:220]}")

        assert resume, "the summariser returned nothing"

        anglais = [t for t in _ENGLISH_TELLS if t in resume]
        assert not anglais, (
            f"The summary of a {case.langue} conversation carries {anglais!r}, "
            f"the prompt's own English idiom. He opens this file himself, and "
            f"the learning step proposes lines from it into a file he wrote in "
            f"his own language. Summary: {resume[:300]}"
        )
        assert any(w in resume for w in case.attendus), (
            f"The summary carries none of {case.attendus!r}, so it is not in "
            f"{case.langue} either. Summary: {resume[:300]}"
        )

    def test_a_conversation_in_english_stays_in_english(self):
        """The rule is 'keep the language', not 'translate to the user's'.
        A summariser that started translating English into French would
        be the same defect facing the other way."""
        resume = _summarise((
            "User: Yuba, I'm going to Lyon on Friday for an interview.",
            "Assistant: Good luck.",
            "User: I prefer the train over flying, it is calmer.",
        ))

        print(f"\n  [english] {resume[:220]}")
        assert any(t in resume for t in _ENGLISH_TELLS + ("lyon",))
        assert "utilisateur" not in resume
