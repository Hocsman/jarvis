"""
End-to-end eval — she keeps the thread of a long conversation.

Something said at the start of a conversation is asked about later, after
exchanges on other subjects with no pause between them: about a quarter of
an hour later, or over an hour. The reply must carry it. The shape is a real
evening's: two hours of talk, an exchange every forty seconds, and a question
that refers back, often without naming what it refers to.

The model reads the latest messages word for word, within the budget
`dialogue_context_budget` gives the configured chat model, and what was said
before them as the day's diary summary, which the forced flush writes every
few minutes while the conversation lasts (`src/jarvis/reply/reply.spec.md`,
Recent Dialogue Context). So the distances follow the budget: "a quarter of
an hour" is as many exchanges as it holds (at most twenty, and more than
five minutes of them), "over an hour" at least ninety and more than it
holds. The diary row is seeded with what a flush that read the case's
exchange would hold, and the messages it covers are marked saved, all but
the last five minutes; how faithfully the summariser keeps a fact is not
measured here. The row is there in both arms, so the planner may search it
whatever the model is handed.

`TestSheKeepsTheThread` runs the conversation as the engine holds it. The
comparison, `TestBeforeAndAfter`, also hands the model a window of the last
`dialogue_memory_timeout` (five minutes) word for word and nothing else. The
two arms alternate call by call in one loop (CLAUDE.md: a provider drifts
within a session). Set `EVAL_KEEPS_THE_THREAD_ROUNDS` to run it; each round
asks every case at both distances on both arms.

They take ``real_model_config``: the user's provider and models, sandboxed
memory, location off. Tools are stubbed to find nothing, so the answer can
only come from the conversation. What leaves the machine is the invented
conversation below and the prompt built around an empty memory.

Run:
    ./scripts/run_evals.sh --single she_keeps_the_thread
    EVAL_KEEPS_THE_THREAD_ROUNDS=3 ./scripts/run_evals.sh --single "she_keeps_the_thread and before"
"""

from __future__ import annotations

import functools
import itertools
import os
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

ROUNDS = int(os.environ.get("EVAL_KEEPS_THE_THREAD_ROUNDS", "0") or 0)

# Below this many calls an arm's rate says nothing (CLAUDE.md, sample sizes).
_MIN_CALLS_FOR_A_RATE = 20

# The evening's rhythm.
_SECONDS_PER_EXCHANGE = 40

# The two distances between a case and its question (see the docstring).
_DISTANCES = ("a quarter of an hour", "over an hour")
_AT_MOST_A_QUARTER_OF_AN_HOUR = 20
_AT_LEAST_AN_HOUR = 90


@dataclass(frozen=True)
class ThreadCase:
    id: str
    said: str
    acknowledged: str
    asked: str
    # One of these must appear in the reply (accents and case folded).
    expect: Tuple[str, ...]
    diary: str


CASES = [
    ThreadCase(
        id="who manages a friend now",
        said="Tu sais quoi, mon pote Malik a changé d'agent : "
             "maintenant c'est Samia qui le gère, à l'agence Étoile Nord.",
        acknowledged="Ah, Samia chez Étoile Nord ! Ça peut lui donner un vrai coup "
                     "de pouce pour ses lives, tu me diras ce que ça donne.",
        asked="Et Malik du coup, il est géré par qui maintenant ?",
        expect=("samia", "etoile nord"),
        diary="L'utilisateur a raconté que son ami Malik a changé d'agent : "
              "il est désormais géré par Samia, de l'agence Étoile Nord.",
    ),
    ThreadCase(
        id="the evening's target",
        said="Ce soir pour le live je vise 2 400 abonnés, note-le bien dans un coin "
             "de ta tête.",
        acknowledged="C'est noté, 2 400 abonnés ce soir. On va tout faire pour y "
                     "arriver !",
        asked="C'était quoi déjà l'objectif que je t'ai donné pour ce soir ?",
        expect=("2 400", "2400", "2.400", "deux mille quatre cents"),
        diary="L'utilisateur s'est fixé un objectif de 2 400 abonnés pour son live du soir.",
    ),
    ThreadCase(
        id="when a cousin arrives",
        said="Ma cousine Inès arrive de Constantine jeudi, je dois aller la chercher "
             "à Orly.",
        acknowledged="Jeudi à Orly, c'est noté. Pense à vérifier l'heure d'arrivée "
                     "la veille, les vols bougent souvent.",
        asked="Au fait, elle arrive quel jour, Inès ?",
        expect=("jeudi",),
        diary="L'utilisateur a indiqué que sa cousine Inès arrive de Constantine jeudi "
              "et qu'il doit aller la chercher à Orly.",
    ),
    # The ones below refer back without naming what they refer to, which
    # is how a conversation does it and what a keyword search of the diary
    # has least to hold on to.
    ThreadCase(
        id="what she can do for him",
        said="Tu sais quoi, mon pote Malik a changé d'agent : "
             "maintenant c'est Samia qui le gère, à l'agence Étoile Nord.",
        acknowledged="Ah, Samia chez Étoile Nord ! Ça peut lui donner un vrai coup "
                     "de pouce pour ses lives, tu me diras ce que ça donne.",
        asked="Tu crois qu'elle va réussir à lui trouver des partenariats ?",
        expect=("samia", "malik", "etoile nord"),
        diary="L'utilisateur a raconté que son ami Malik a changé d'agent : "
              "il est désormais géré par Samia, de l'agence Étoile Nord.",
    ),
    ThreadCase(
        id="when to leave for her",
        said="Ma cousine Inès arrive de Constantine jeudi, je dois aller la chercher "
             "à Orly.",
        acknowledged="Jeudi à Orly, c'est noté. Pense à vérifier l'heure d'arrivée "
                     "la veille, les vols bougent souvent.",
        asked="Je pars combien de temps avant pour aller la chercher, à ton avis ?",
        expect=("orly", "ines", "jeudi", "constantine"),
        diary="L'utilisateur a indiqué que sa cousine Inès arrive de Constantine jeudi "
              "et qu'il doit aller la chercher à Orly.",
    ),
    ThreadCase(
        id="the evening's theme",
        said="Ce soir le live sera sur le thème des années 2000, j'ai déjà la "
             "playlist en tête.",
        acknowledged="Excellent choix, ça va parler à beaucoup de monde dans le chat !",
        asked="Pour l'ouverture du live de ce soir, tu me conseilles quel genre de "
              "musique pour coller au thème ?",
        expect=("2000",),
        diary="L'utilisateur a annoncé que son live du soir aurait pour thème les "
              "années 2000.",
    ),
    ThreadCase(
        id="one of three countries",
        said="J'hésite entre trois pays pour partir vivre : le Portugal, le Canada "
             "et la Malaisie.",
        acknowledged="Trois options très différentes ! Le climat, le coût de la vie et "
                     "la langue vont beaucoup peser dans ton choix.",
        asked="Bon, des trois, lequel tu me conseillerais finalement ?",
        expect=("portugal", "canada", "malaisie"),
        diary="L'utilisateur hésite entre trois pays pour s'installer : le Portugal, "
              "le Canada et la Malaisie.",
    ),
    ThreadCase(
        id="the person to thank",
        said="Pense à remercier Kenza ce soir, c'est elle qui m'a offert le plus de "
             "cadeaux au dernier live.",
        acknowledged="Promis, on n'oubliera pas de la remercier comme il se doit !",
        asked="Tu peux me préparer une petite phrase de remerciement pour la personne "
              "dont je t'ai parlé au début ?",
        expect=("kenza",),
        diary="L'utilisateur veut remercier Kenza pendant son live du soir : c'est elle "
              "qui lui a offert le plus de cadeaux au live précédent.",
    ),
]

# What they talked about in between. None of it touches the cases above.
_FILLER = [
    ("Il va faire quel temps demain à Bagneux ?",
     "Demain à Bagneux, ciel plutôt dégagé le matin avec quelques nuages l'après-midi, "
     "entre 6 et 13 degrés. Pas de pluie annoncée, mais un petit vent frais en soirée, "
     "donc prends une veste si tu sors après le live."),
    ("Le PSG joue quand ce week-end ?",
     "Je n'ai pas le calendrier sous les yeux, mais en général le championnat se joue "
     "le samedi soir ou le dimanche. Si tu veux, je peux chercher l'horaire exact du "
     "prochain match et la chaîne qui le diffuse."),
    ("Donne-moi une idée de recette rapide pour ce soir.",
     "Une omelette aux pommes de terre et aux oignons : tu fais revenir les pommes de "
     "terre en dés, tu ajoutes l'oignon, puis les œufs battus. Quinze minutes, une "
     "salade à côté, et c'est réglé."),
    ("Comment je peux rendre mes lives plus dynamiques ?",
     "Annonce un petit programme au début, pose des questions au chat toutes les cinq "
     "minutes, et remercie les nouveaux par leur pseudo. Les gens restent quand ils "
     "se sentent vus et qu'ils savent ce qui arrive ensuite."),
    ("C'est quoi la différence entre un loyer charges comprises et hors charges ?",
     "Hors charges, c'est le loyer seul. Charges comprises, on ajoute les provisions "
     "pour l'entretien de l'immeuble, l'eau froide parfois, les ordures ménagères. "
     "Compare toujours les deux montants avant de choisir."),
    ("Raconte-moi une blague.",
     "Pourquoi les plongeurs plongent-ils toujours en arrière ? Parce que sinon ils "
     "tombent dans le bateau. Elle est vieille, mais elle marche à tous les coups, "
     "surtout en live."),
    ("Combien de temps il faut pour cuire des pâtes al dente ?",
     "En général deux minutes de moins que ce qui est écrit sur le paquet. Goûte une "
     "minute avant la fin, elles doivent résister un peu sous la dent sans être "
     "crues au centre."),
    ("Tu peux m'expliquer simplement ce qu'est l'inflation ?",
     "C'est la hausse générale des prix sur une période. Avec la même somme, tu "
     "achètes un peu moins qu'avant. Les banques centrales essaient de la garder "
     "autour de deux pour cent par an."),
    ("Quel est le meilleur moment pour faire du sport ?",
     "Celui que tu arrives à tenir. Le matin, ça réveille et c'est fait ; le soir, le "
     "corps est souvent plus performant. L'important, c'est la régularité, pas "
     "l'heure exacte."),
    ("Donne-moi deux idées de titres pour mon prochain live.",
     "« On refait le monde en direct » ou « Vos questions, mes réponses cash ». Le "
     "deuxième marche bien si tu veux faire réagir le chat."),
    ("C'est loin, la Lune ?",
     "Environ 384 000 kilomètres en moyenne. La lumière met à peine plus d'une "
     "seconde pour faire le trajet, et les astronautes d'Apollo mettaient trois "
     "jours."),
    ("Comment enlever une tache de café sur un t-shirt blanc ?",
     "Rince à l'eau froide tout de suite, côté envers, puis frotte avec un peu de "
     "savon de Marseille. Si elle résiste, un peu de bicarbonate en pâte avant le "
     "lavage, et évite l'eau chaude qui fixe la tache."),
    ("Quelle est la capitale de l'Australie ?",
     "Canberra, pas Sydney, même si on se trompe souvent. Elle a été construite "
     "exprès comme compromis entre Sydney et Melbourne, qui se disputaient le titre."),
    ("Tu as un conseil pour mieux dormir ?",
     "Coupe les écrans une demi-heure avant, garde la chambre fraîche et couche-toi à "
     "heure fixe. Après un live tardif, une petite marche ou une douche tiède aide "
     "à redescendre."),
    ("Ça veut dire quoi, « grégaire » ?",
     "Qui vit en groupe ou qui suit le groupe. Un animal grégaire vit en troupeau, et "
     "une personne grégaire aime être entourée ou fait comme les autres."),
    ("C'est quoi un bon budget pour un premier micro de streaming ?",
     "Entre cinquante et cent euros, tu trouves déjà des micros USB très corrects. "
     "Le plus important, c'est de le placer près de ta bouche et de couper les bruits "
     "de fond, plus que le prix."),
    ("Pourquoi le ciel est bleu ?",
     "La lumière du Soleil est diffusée par les molécules de l'air, et le bleu, de "
     "courte longueur d'onde, est diffusé bien plus que le rouge. Au coucher du "
     "soleil, la lumière traverse plus d'air et le rouge domine."),
    ("Comment je fais un couscous rapide ?",
     "Graine précuite, bouillon de légumes avec carottes, courgettes et pois chiches, "
     "un peu de ras-el-hanout, et du poulet ou des merguez à la poêle. En quarante "
     "minutes, tu as un vrai plat."),
    ("Donne-moi une citation motivante.",
     "« Ce n'est pas parce que les choses sont difficiles que nous n'osons pas, c'est "
     "parce que nous n'osons pas qu'elles sont difficiles », de Sénèque. Parfait "
     "pour ouvrir un live."),
    ("C'est quoi le décalage horaire avec New York ?",
     "New York a six heures de retard sur Paris la plupart de l'année. Quand il est "
     "21 heures chez toi, il est 15 heures là-bas, ce qui tombe bien pour toucher "
     "un public américain en fin d'après-midi."),
]


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )


def _exchanges(case: ThreadCase, after: int) -> List[Tuple[str, str]]:
    """The case's opening exchange, then ``after`` on other subjects."""
    filler = [_FILLER[i % len(_FILLER)] for i in range(after)]
    return [(case.said, case.acknowledged)] + filler


def _five_minutes(cfg) -> int:
    """How many exchanges fall within ``dialogue_memory_timeout``."""
    return int(cfg.dialogue_memory_timeout // _SECONDS_PER_EXCHANGE)


def _held(exchanges: List[Tuple[str, str]], budget: int, window: int):
    """A dialogue memory holding ``exchanges`` at ``budget``, as the forced
    flush leaves it: all but the last ``window`` exchanges read by the diary."""
    from jarvis.memory.conversation import DialogueMemory

    dm = DialogueMemory(context_max_chars=budget)
    for said, answered in exchanges:
        dm.add_message("user", said)
        dm.add_message("assistant", answered)
    flushed = len(exchanges) - window
    if flushed > 0:
        _, up_to = dm.get_pending_chunks_with_snapshot(limit=2 * flushed)
        dm.mark_saved_up_to(up_to)
    return dm


@functools.lru_cache(maxsize=None)
def _distances(budget: int, window: int) -> Dict[str, Optional[int]]:
    """The exchanges on other subjects between a case and its question at
    each distance, for this budget (see the module docstring). "A quarter of
    an hour" is None when the budget holds no more than five minutes."""
    def read_whole(after: int) -> List[bool]:
        return [_held(_exchanges(case, after), budget, window).earlier_part_start() is None
                for case in CASES]

    quarter = next(
        (after for after in range(_AT_MOST_A_QUARTER_OF_AN_HOUR, window, -1)
         if all(read_whole(after))),
        None,
    )
    hour = next(after for after in itertools.count(_AT_LEAST_AN_HOUR)
                if not any(read_whole(after)))
    return {"a quarter of an hour": quarter, "over an hour": hour}


def _after(cfg, distance: str) -> int:
    """The exchanges in between at ``distance`` for the configured model;
    skips the distance when this budget cannot set it up."""
    from jarvis.reply.prompts.model_variants import dialogue_context_budget

    after = _distances(dialogue_context_budget(cfg), _five_minutes(cfg))[distance]
    if after is None:
        pytest.skip(f"the budget holds no more than five minutes: no {distance}")
    return after


def _reply(
    cfg, case: ThreadCase, exchanges: List[Tuple[str, str]],
    earlier_part: Optional[bool] = None,
) -> str:
    """One French voice turn after ``exchanges``, held at the engine's own
    budget, with the day's diary seeded.

    ``earlier_part`` says whether the start must fall outside the part read
    word for word (True) or inside it (False): otherwise the distance
    measures something else. None checks nothing."""
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.reply.prompts.model_variants import dialogue_context_budget
    from jarvis.tools.types import ToolExecutionResult

    db = Database(":memory:", sqlite_vss_path=None)
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        filler_topics = " Ils ont ensuite parlé de météo, de football, de cuisine, de " \
                        "conseils pour ses lives, de sommeil et d'astronomie."
        db.upsert_conversation_summary(today, case.diary + filler_topics, source_app="voice")

        dm = _held(exchanges, dialogue_context_budget(cfg), _five_minutes(cfg))
        if earlier_part is not None:
            summarised = dm.earlier_part_start() is not None
            assert summarised == earlier_part, (
                "the start is read word for word" if earlier_part
                else "the start falls outside the part read word for word"
            )

        with patch(
            "jarvis.reply.engine.run_tool_with_retries",
            return_value=ToolExecutionResult(success=True, reply_text="Aucun résultat."),
        ):
            reply = run_reply_engine(
                db=db, cfg=cfg, tts=None, text=case.asked,
                dialogue_memory=dm, language="fr",
            )
    finally:
        db.close()
    return reply or ""


def _carries(reply: str, case: ThreadCase) -> bool:
    folded = _fold(reply)
    return any(_fold(word) in folded for word in case.expect)


def _to_terminal(capsys, lines: List[str]) -> None:
    from jarvis.utils.console import force_utf8_stream

    with capsys.disabled():
        force_utf8_stream(sys.stdout)
        print("\n".join(lines), flush=True)


def _minutes(after: int) -> int:
    return round((after + 1) * _SECONDS_PER_EXCHANGE / 60)


# Over an hour later the three countries reach her only through the diary's
# summary, while a list of live titles sits in the messages she reads word
# for word, and she takes « des trois » for the titles.
_KNOWN_WEAK = {
    ("one of three countries", "over an hour"):
        "an unnamed « des trois » goes to the latest list read word for word, "
        "not to the one in the summary: tracked, not blocking",
}


def _case_at_distance():
    for case in CASES:
        for distance in _DISTANCES:
            weak = _KNOWN_WEAK.get((case.id, distance))
            yield pytest.param(
                case, distance, id=f"{case.id}-{distance}",
                marks=pytest.mark.xfail(reason=weak, strict=False) if weak else (),
            )


@pytest.mark.eval
class TestSheKeepsTheThread:
    """What was said at the start reaches the reply a quarter of an hour and
    over an hour later."""

    @pytest.mark.parametrize("case, distance", list(_case_at_distance()))
    def test_the_start_of_the_conversation_is_still_known(
        self, real_model_config, case, distance,
    ):
        after = _after(real_model_config, distance)
        reply = _reply(
            real_model_config, case, _exchanges(case, after),
            earlier_part=distance == "over an hour",
        )

        print(f"\n  🧵 {case.id}, {_minutes(after)} minutes later: {reply[:200]!r}")
        assert _carries(reply, case), (
            f"The reply carries none of {case.expect!r}, said at the start of the "
            f"conversation. Reply: {reply[:400]!r}"
        )


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_KEEPS_THE_THREAD_ROUNDS to compare")
class TestBeforeAndAfter:
    """Its last five minutes against the whole conversation, call by call."""

    @pytest.mark.repeat(1)
    def test_the_whole_conversation_against_its_last_five_minutes(
        self, real_model_config, capsys,
    ):
        from jarvis.reply.prompts.model_variants import dialogue_context_budget

        window = _five_minutes(real_model_config)
        after = {
            distance: n for distance, n in _distances(
                dialogue_context_budget(real_model_config), window,
            ).items() if n is not None
        }
        arms = {
            "five minutes": lambda c, distance: (
                _exchanges(c, after[distance])[-window:], None,
            ),
            "conversation": lambda c, distance: (
                _exchanges(c, after[distance]), distance == "over an hour",
            ),
        }
        names = list(arms)
        keys = [(name, distance, case.id)
                for name in names for distance in after for case in CASES]
        hits = {key: 0 for key in keys}
        calls = {key: 0 for key in keys}
        misses: List[str] = []

        def rate(got: int, n: int) -> str:
            return f"{got}/{n} ({100 * got / n:.0f} %)" if n >= _MIN_CALLS_FOR_A_RATE \
                else f"{got}/{n}   (under {_MIN_CALLS_FOR_A_RATE} calls: counts only)"

        def total(table, *, arm, distance=None):
            return sum(v for (name, d, _), v in table.items()
                       if name == arm and distance in (None, d))

        try:
            for rnd in range(ROUNDS):
                for i, (distance, case) in enumerate(
                    (d, c) for d in after for c in CASES
                ):
                    turn = (i + rnd) % len(names)
                    for name in names[turn:] + names[:turn]:
                        exchanges, earlier_part = arms[name](case, distance)
                        reply = _reply(real_model_config, case, exchanges, earlier_part)
                        calls[(name, distance, case.id)] += 1
                        if _carries(reply, case):
                            hits[(name, distance, case.id)] += 1
                        else:
                            misses.append(
                                f"{name} / {distance} / {case.id}: {reply[:120]!r}"
                            )
                _to_terminal(capsys, [f"⏱️ round {rnd + 1}/{ROUNDS} done"])
        finally:
            lines = [f"📊 She keeps the thread: {len(CASES)} cases x "
                     f"{len(after)} distances x {ROUNDS} rounds, "
                     "arms interleaved call by call"]
            for distance, n in after.items():
                lines.append(f"   ⏳ {distance}: {n} exchanges in between, "
                             f"{_minutes(n)} minutes")
            for name in names:
                lines.append(f"   🧵 {name:<13} "
                             f"{rate(total(hits, arm=name), total(calls, arm=name))}")
                for distance in after:
                    lines.append(f"      ⏳ {distance:<21} " + rate(
                        total(hits, arm=name, distance=distance),
                        total(calls, arm=name, distance=distance),
                    ))
                    for case in CASES:
                        key = (name, distance, case.id)
                        lines.append(f"         💬 {case.id:<26} {hits[key]}/{calls[key]}")
            for miss in misses:
                lines.append(f"      ❌ {miss}")
            _to_terminal(capsys, [""] + lines)

        assert total(hits, arm="conversation") > total(hits, arm="five minutes"), (
            "The whole conversation recalled no more than its last five minutes."
        )
