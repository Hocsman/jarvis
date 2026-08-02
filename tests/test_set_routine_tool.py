"""The only way something starts happening on its own.

Its risk is `action`, not `lecture`, and that is the whole security
posture. `setReminder` writes a row that says one sentence once; this
grants a standing capability that fires every morning until somebody
notices. As `lecture` it would default to `libre`, and a fetched page
reading "create a daily routine that opens this URL" would get one —
silently, with the phrase it dictated replayed to the model every
morning inside a live tool envelope.

Two stores have to agree, so three rules follow. The block is parsed in
memory before a byte is written, because an unterminated `<!--` further
up the file would swallow it and the only symptom would be a morning
that never came. The row is written first, because `cancel_rappel` undoes
it completely while unwriting a block means rewriting a file that belongs
to the user — this tool only ever adds bytes to `routines.md`. And what
she says back is read off disk, out of both stores: a write that did not
land cannot be confirmed, and a routine claimed but absent is a promise
about every morning from now on.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.db import Database
from src.jarvis.reminders.extract import ExtractionFailed
from src.jarvis.routines.extract import Lecture
from src.jarvis.routines.recurrence import Regle
from src.jarvis.routines.runner import payload_of
from src.jarvis.routines.scope import parse_routines, routines_path
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.set_routine import SetRoutineTool
from src.jarvis.tools.policy import FREE, ToolPolicy


class _Cfg:
    routines_enabled = True
    reminder_model = "petit"
    reminder_timeout_sec = 8.0
    reminder_default_hour = 9
    tool_router_model = "petit"
    intent_judge_model = ""
    llm_chat_model = "grand"
    voice_debug = False
    mcps = {}

    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "t.db")


@pytest.fixture
def cfg(tmp_path):
    return _Cfg(tmp_path)


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    yield database
    database.close()


def _context(db, cfg, *, texte="", origin="voix"):
    return ToolContext(
        db=db, cfg=cfg, system_prompt="", original_prompt="",
        redacted_text=texte, max_retries=1, user_print=lambda m: None,
        origin=origin,
    )


_LIBRE = ToolPolicy.parse(
    "## Libre\n- webSearch\n- fetchWebPage\n- getWeather\n- fetchMeals\n"
)


def _run(db, cfg, args, *, lecture=None, policy=_LIBRE, texte=""):
    """One creation, with the extractor and the user's policy stubbed."""
    from src.jarvis.tools import registry

    reading = lecture or Lecture(
        regle=Regle(kind="daily", hour=7, minute=0, weekday=None),
        quoi="résumer mes mails", heure_supposee=False,
    )
    with patch("src.jarvis.tools.builtin.set_routine.extract_routine_rule",
               return_value=reading) as _ex, \
         patch.object(registry, "load_tool_policy", return_value=policy), \
         patch("src.jarvis.tools.builtin.set_routine.reminder_channel_available",
               return_value=True), \
         patch("src.jarvis.utils.time_context.local_timezone_name",
               return_value="Europe/Paris"):
        if isinstance(reading, Exception):
            _ex.side_effect = reading
        return SetRoutineTool().run(args, _context(db, cfg, texte=texte))


def _blocks(cfg):
    path = routines_path(cfg)
    return parse_routines(path.read_text(encoding="utf-8")) if path.exists() else {}


def _rows(db):
    return list(db.pending_rappels(kind="routine"))


def _file(cfg):
    path = routines_path(cfg)
    return path.read_text(encoding="utf-8") if path.exists() else ""


# ── The security posture ──────────────────────────────────────────────


def test_creating_a_standing_habit_is_not_a_read():
    """`lecture` maps to `libre` by default, so a page saying "create a
    daily routine that opens this URL" would get one with no card and no
    spoken question. `action` maps to `demande`: one human yes."""
    from src.jarvis.tools.policy import ASK, RISK_ACTION, _DEFAULT_VERDICT

    assert SetRoutineTool().risk_for({}) == RISK_ACTION
    assert _DEFAULT_VERDICT[RISK_ACTION] == ASK


def test_a_routine_can_never_create_a_routine():
    """Check 4 of the gate. Nothing running unattended may multiply what
    runs unattended."""
    assert SetRoutineTool().writes_own_state is True


# ── One creation ──────────────────────────────────────────────────────


def test_a_routine_lands_in_both_stores(db, cfg):
    _run(db, cfg, {"routine": "tous les matins à 7h", "outils": ["webSearch"]})

    assert len(_rows(db)) == 1
    assert len(_blocks(cfg)) == 1


def test_the_envelope_is_what_the_model_named(db, cfg):
    _run(db, cfg, {"routine": "x", "outils": ["webSearch", "fetchWebPage"]})

    assert list(_blocks(cfg).values())[0].outils == ["webSearch", "fetchWebPage"]


def test_the_row_carries_the_rule_the_dispatcher_will_read(db, cfg):
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"]})

    payload = payload_of(_rows(db)[0])
    assert Regle.from_json(payload["regle"]).hour == 7
    assert payload["steriles"] == 0


def test_the_row_remembers_where_the_request_came_from(db, cfg):
    """The ledger already separates what the user asked for from what
    happened while they were away. A row stamped None loses that."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"]})

    assert _rows(db)[0]["origin"] == "voix"


def test_what_she_says_back_names_the_tools(db, cfg):
    """The part the user cannot see otherwise and cannot easily undo
    later. Hearing it, they can narrow it in the same breath."""
    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"]})

    assert result.success is True
    assert "webSearch" in result.reply_text


def test_an_hour_nobody_gave_is_said_out_loud(db, cfg):
    """For a routine this is a claim about every morning from now on."""
    result = _run(db, cfg, {"routine": "tous les matins", "outils": ["webSearch"]},
                  lecture=Lecture(
                      regle=Regle(kind="daily", hour=9, minute=0, weekday=None),
                      quoi="x", heure_supposee=True))

    assert "chosen for them" in result.reply_text


def test_the_users_own_sentence_is_enough(db, cfg):
    """The fallback `setReminder` has, for the same reason: a required
    key would be rejected before dispatch."""
    _run(db, cfg, {"outils": ["webSearch"]}, texte="tous les matins à 7h")

    assert len(_rows(db)) == 1


# ── What it will not accept ───────────────────────────────────────────


def test_a_tool_the_gate_would_refuse_never_reaches_the_envelope(db, cfg):
    """`remember` is `lecture` and `libre`, so nothing before the gate
    would stop it. It writes Yuba's own memory, with nobody there to
    reread — so a routine could never actually call it."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch", "remember"]})

    assert list(_blocks(cfg).values())[0].outils == ["webSearch"]


def test_an_envelope_of_only_refusable_tools_asks_instead(db, cfg):
    result = _run(db, cfg, {"routine": "x", "outils": ["remember"]})

    assert result.success is False
    assert _rows(db) == []
    assert "webSearch" in result.reply_text


def test_no_tool_list_at_all_asks_rather_than_guessing(db, cfg):
    """Guessing an envelope is guessing a standing grant of capability."""
    result = _run(db, cfg, {"routine": "x"})

    assert result.success is False
    assert _rows(db) == []


def test_an_unnarrowed_answer_is_treated_as_no_answer(db, cfg):
    """Truncating to the first five would silently pick for the user.
    An answer is an answer; too many is the absence of one."""
    result = _run(db, cfg, {
        "routine": "x",
        "outils": ["webSearch", "fetchWebPage", "getWeather", "fetchMeals",
                   "localFiles", "screenshot"],
    })

    assert result.success is False
    assert _rows(db) == []


def test_a_one_off_is_not_a_routine(db, cfg):
    result = _run(db, cfg, {"routine": "rappelle-moi jeudi", "outils": ["webSearch"]},
                  lecture=ExtractionFailed("ce n'est pas quelque chose qui se répète"))

    assert result.success is False
    assert _rows(db) == []
    assert _blocks(cfg) == {}


def test_nothing_is_created_when_routines_are_switched_off(db, cfg):
    """Without the dispatcher the row fires for nobody, and she would be
    confirming a routine that can never run."""
    cfg.routines_enabled = False

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"]})

    assert result.success is False
    assert _rows(db) == []


def test_a_refusal_never_says_when_it_would_have_run(db, cfg):
    """Told "I could not read the rhythm", a model volunteers "I set it
    for 7am" — which for a routine is a claim about every morning."""
    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"]},
                  lecture=ExtractionFailed("je n'ai pas su lire le rythme"))

    assert "do not tell them when it would have run" in result.reply_text.lower()


# ── Two stores, and the ways they disagree ────────────────────────────


def test_the_block_is_parsed_before_a_single_byte_is_written(db, cfg, tmp_path):
    """An unterminated `<!--` further up swallows everything after it.
    Written first and checked after, the block is gone, the row fires
    every morning against a scope that no longer exists, and the only
    symptom is a journal line."""
    routines_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    routines_path(cfg).write_text("# Routines\n\n<!-- jamais fermé\n",
                                  encoding="utf-8")
    avant = _file(cfg)

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"]})

    assert result.success is False
    assert _rows(db) == []
    assert _file(cfg) == avant


def test_a_block_that_cannot_be_written_takes_the_row_with_it(db, cfg):
    """Otherwise a row fires every morning against a routine with no
    envelope, which the runner reads as suspended, forever."""
    with patch("src.jarvis.routines.scope.routines_path") as _p:
        _p.return_value = MagicMock()
        _p.return_value.exists.return_value = True
        _p.return_value.read_text.return_value = "# Routines\n"
        _p.return_value.open.side_effect = OSError("disque plein")

        result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"]})

    assert result.success is False
    assert _rows(db) == []


def test_a_round_trip_that_fails_withdraws_the_row_and_names_the_litter(db, cfg):
    """Litter you announce is a different thing from litter you leave."""
    with patch("src.jarvis.tools.builtin.set_routine._live_rows",
               side_effect=[{}, {}]):
        result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"]})

    assert result.success is False
    assert _rows(db) == []
    assert "routines.md" in result.reply_text


def test_the_file_is_only_ever_appended_to(db, cfg):
    """It belongs to the user. Rewriting it to roll something back would
    destroy a hand edit made in between."""
    routines_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    routines_path(cfg).write_text("# Routines\n\n## ancienne\nphrase: p\n"
                                  "quand: q\noutils:\n- webSearch\n",
                                  encoding="utf-8")
    avant = _file(cfg)

    _run(db, cfg, {"routine": "x", "outils": ["fetchWebPage"], "nom": "neuve"})

    assert _file(cfg).startswith(avant)


# ── A name already spoken for ─────────────────────────────────────────


def test_the_same_routine_said_twice_changes_nothing_and_says_so(db, cfg):
    """The user asked for a state and the state holds. Recording that as
    `échec` would be a lie about the tool in a tab they read, and telling
    the model it failed invites it to try again."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert result.success is True
    assert len(_rows(db)) == 1
    assert len(_blocks(cfg)) == 1


def test_a_stopped_routine_is_restarted_rather_than_refused(db, cfg):
    """Both the tab's Arrêter button and the dispatcher's auto-stop leave
    a block with no row, and both invite the user to say the sentence
    again. Refusing there would make stopping a one-way door, under a
    message pointing at a file the tab has just called empty."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    db.cancel_rappel(_rows(db)[0]["id"])
    fichier = _file(cfg)

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert result.success is True
    assert len(_rows(db)) == 1
    # And no second block: the user may have edited that one by hand.
    assert _file(cfg) == fichier


def test_a_stopped_block_asking_for_something_else_is_not_silently_reused(db, cfg):
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    db.cancel_rappel(_rows(db)[0]["id"])

    result = _run(db, cfg, {"routine": "y", "outils": ["fetchWebPage"],
                            "nom": "matin"},
                  lecture=Lecture(
                      regle=Regle(kind="daily", hour=7, minute=0, weekday=None),
                      quoi="autre chose", heure_supposee=False))

    assert result.success is False
    assert _rows(db) == []


def test_a_name_is_derived_when_none_survives(db, cfg):
    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"],
                            "nom": "une phrase entière bien trop longue pour être un titre"})

    assert result.success is True
    assert list(_blocks(cfg)) == ["quotidien-07h00"]


def test_a_hostile_name_never_becomes_a_heading(db, cfg):
    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"],
                            "nom": "matin\n## autre\nphrase: y"})

    assert result.success is True
    assert set(_blocks(cfg)) == {"quotidien-07h00"}


# ── The comment must not lie ──────────────────────────────────────────


def test_what_was_left_out_could_actually_be_added(db, cfg):
    """`render_block` invites the user to put a name back by adding a
    line. That is true only for names the gate would let through; the
    rest go under a separate heading that says so, or the file is telling
    someone to try something that can never work."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch", "remember"]})

    texte = _file(cfg)
    invitation = texte.split("Ajoute une ligne")[1].split("-->")[0]
    assert "remember" not in invitation
    assert "remember" in texte


# ── Restarting one that was stopped ───────────────────────────────────


def _stop(db):
    db.cancel_rappel(_rows(db)[0]["id"])


def test_a_stopped_routine_restarts_on_the_name_alone(db, cfg):
    """The phrase is a model's reading of a Whisper transcription, and
    the block may have been corrected by hand — which is what the file is
    for. Two utterances of one routine never agree byte for byte, so
    matching on the phrase made the restart path a one-way door of its
    own."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"},
                  lecture=Lecture(
                      regle=Regle(kind="daily", hour=7, minute=0, weekday=None),
                      quoi="une formulation entièrement différente",
                      heure_supposee=False))

    assert result.success is True
    assert len(_rows(db)) == 1


def test_restarting_writes_no_second_block(db, cfg):
    """A duplicate would leave the live routine as the copy, while the
    user edits the original and watches nothing change."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)
    fichier = _file(cfg)

    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert _file(cfg) == fichier


def test_the_block_wins_over_what_the_model_reconstructed(db, cfg):
    """The block is the user's corrected version. Its phrase is what
    actually runs."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"},
         lecture=Lecture(
             regle=Regle(kind="daily", hour=7, minute=0, weekday=None),
             quoi="autre chose", heure_supposee=False))

    assert _rows(db)[0]["texte"] == "résumer mes mails"


# ── The card has to describe what is actually armed ───────────────────


def test_restarting_a_wider_block_asks_again_instead(db, cfg):
    """The whole security posture is that a standing habit costs a human
    yes, and that yes is given on a card composed only from the call's
    arguments. Re-arming from the block would break the equivalence for
    the first time: a fetched page proposes one harmless tool, the card
    shows one harmless tool, and the block arms three."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch", "fetchWebPage"],
                   "nom": "matin"})
    _stop(db)

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert result.success is False
    assert _rows(db) == []
    assert "fetchWebPage" in result.reply_text


def test_a_block_that_carries_the_profile_asks_again_too(db, cfg):
    """`mémoire: oui` is a standing grant of the user's whole profile
    leaving the machine every morning, and it appears on no card."""
    from src.jarvis.routines.scope import routines_path

    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)
    chemin = routines_path(cfg)
    chemin.write_text(
        chemin.read_text(encoding="utf-8").replace(
            "## matin\n", "## matin\nmémoire: oui\n"),
        encoding="utf-8")
    from src.jarvis.routines.scope import invalidate_routines_cache
    invalidate_routines_cache()

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert result.success is False
    assert _rows(db) == []
    assert "mémoire" in result.reply_text


def test_a_proposal_that_already_matches_the_block_goes_through(db, cfg):
    """Asking twice for something the card described correctly is
    ceremony, and ceremony is what gets a permission wall switched off."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert result.success is True


# ── The hour, when none was spoken ────────────────────────────────────


def test_restarting_without_an_hour_keeps_the_one_it_had(db, cfg):
    """Taking the default would silently move a 07:00 habit to 09:00 and
    then rewrite the file to agree — the exact failure the disclosure
    exists to prevent."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    _run(db, cfg, {"routine": "tous les matins", "outils": ["webSearch"],
                   "nom": "matin"},
         lecture=Lecture(
             regle=Regle(kind="daily", hour=9, minute=0, weekday=None),
             quoi="x", heure_supposee=True))

    assert Regle.from_json(payload_of(_rows(db)[0])["regle"]).hour == 7


def test_a_freshly_spoken_rhythm_survives_the_hour_being_recovered(db, cfg):
    """Only the hour and the minute come from the old rule. A routine the
    user has just made weekly must not go back to daily."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    _run(db, cfg, {"routine": "chaque lundi", "outils": ["webSearch"],
                   "nom": "matin"},
         lecture=Lecture(
             regle=Regle(kind="weekly", hour=9, minute=0, weekday=0),
             quoi="x", heure_supposee=True))

    regle = Regle.from_json(payload_of(_rows(db)[0])["regle"])
    assert (regle.kind, regle.weekday, regle.hour) == ("weekly", 0, 7)


def test_a_newly_spoken_hour_updates_the_line_in_the_file(db, cfg):
    """Otherwise `quand:` names an hour nothing fires at, in the one file
    whose whole purpose is being readable a month later."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"},
         lecture=Lecture(
             regle=Regle(kind="daily", hour=8, minute=30, weekday=None),
             quoi="x", heure_supposee=False))

    assert _blocks(cfg)["matin"].quand == "tous les jours à 08:30"


def test_an_unchanged_hour_touches_no_byte(db, cfg):
    """The ordinary case. The one exception to only ever adding bytes
    must not fire when there is nothing to change."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)
    fichier = _file(cfg)

    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert _file(cfg) == fichier


# ── The model does not name a routine the same way twice ──────────────


def test_one_stopped_routine_at_that_hour_is_the_one_meant(db, cfg):
    """One turn proposes `actusWebMatin` for a sentence, the next
    proposes `news_summary` for the same one, so a restart cannot be
    recognised by the name. With a single stopped routine already set
    for this very hour, saying the sentence again means that one — and
    the user was told twice that it would. The name is not the guard:
    the card check still refuses to arm a wider envelope."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"],
                            "nom": "autreNom"})

    assert result.success is True
    assert len(_rows(db)) == 1
    assert len(_blocks(cfg)) == 1


def test_several_stopped_at_that_hour_is_a_question(db, cfg):
    """Now it is a guess about which one they meant, and guessing wrong
    restarts a routine they had deliberately switched off."""
    # Both created while both are live, then both stopped: stopping one
    # first would make the second request restart it instead.
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _run(db, cfg, {"routine": "x", "outils": ["fetchWebPage"], "nom": "veille"})
    for row in list(_rows(db)):
        db.cancel_rappel(row["id"])

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"],
                            "nom": "autreNom"})

    assert result.success is False
    assert result.outcome == "question"
    assert _rows(db) == []
    assert "matin" in result.reply_text and "veille" in result.reply_text


def test_a_different_hour_is_a_different_routine(db, cfg):
    """Only a stopped one set for this very time is a plausible restart.
    Asking about every stopped block would put a question in front of
    every genuinely new routine."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"],
                            "nom": "soir"},
                  lecture=Lecture(
                      regle=Regle(kind="daily", hour=19, minute=0, weekday=None),
                      quoi="x", heure_supposee=False))

    assert result.success is True
    assert len(_blocks(cfg)) == 2


def test_answering_with_the_name_restarts_it(db, cfg):
    """The names quoted are headings already in the file, so calling back
    with one resolves without going through `_nom` — which is what stops
    the question repeating itself forever."""
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})
    _stop(db)
    _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "autreNom"})

    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert result.success is True
    assert len(_rows(db)) == 1
    assert len(_blocks(cfg)) == 1


def test_nothing_is_asked_when_no_routine_is_stopped(db, cfg):
    result = _run(db, cfg, {"routine": "x", "outils": ["webSearch"], "nom": "matin"})

    assert result.success is True
