"""Two whole turns, through the real engine.

Everything else in this feature is tested a piece at a time. This is the
one that would catch the pieces being wired together wrongly: a real
`run_reply_engine` call, a real gate, a real policy file, a real store,
and a tool whose side effect is observable — so "it did not run" is a
fact about the world rather than about a mock's call count.

Turn one must end on the question the code wrote, without going back to
the chat model: past the short-circuit a long result is rewritten by a
small model and the loop then tells the model to keep calling tools.
Turn two must run the pinned call once, and only on a yes.
"""

from __future__ import annotations

import json
import time
from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.tools.base import Tool
from src.jarvis.tools.confirmation import DENIED, GRANTED, SPOKEN_TEMPLATE, UNCLEAR
from src.jarvis.tools.policy import ToolPolicy
from src.jarvis.tools.types import ToolExecutionResult


def _cfg(tmp_path):
    cfg = Mock()
    cfg.llm_chat_model = "test-large"
    cfg.ollama_chat_model = "test-large"
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.tool_router_timeout_sec = 8.0
    cfg.llm_embedding_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    cfg.confirmation_ttl_sec = 180.0
    cfg.confirmation_timeout_sec = 8.0
    cfg.confirmation_model = ""
    cfg.db_path = str(tmp_path / "jarvis.db")
    return cfg


class _Sabotage(Tool):
    """Its side effect is a list anyone can look at."""

    def __init__(self):
        self.fired = []

    def risk_for(self, args):
        return "action"

    @property
    def name(self):
        return "sabotage"

    @property
    def description(self):
        return "fait quelque chose"

    @property
    def inputSchema(self):
        return {"type": "object", "properties": {"cible": {"type": "string"}}}

    def run(self, args, context):
        self.fired.append(args)
        return ToolExecutionResult(success=True, reply_text="c'est fait")


def _tool_call_response(name, args):
    return {"message": {"content": "", "tool_calls": [
        {"function": {"name": name, "arguments": args}},
    ]}}


@pytest.fixture
def scene(tmp_path):
    """One tool, one policy that asks about it, one shared memory."""
    from src.jarvis.tools import registry

    tool = _Sabotage()
    dm = DialogueMemory()
    cfg = _cfg(tmp_path)
    policy = ToolPolicy.parse("## Demande\n- sabotage\n")

    catalogue = {**registry.BUILTIN_TOOLS, "sabotage": tool}
    # The engine binds its own name at import time, so patching the
    # registry's dict alone leaves the allow-list built from the old one
    # and the tool never reaches the gate.
    with patch.object(registry, "BUILTIN_TOOLS", catalogue), \
         patch("src.jarvis.reply.engine.BUILTIN_TOOLS", catalogue), \
         patch.object(registry, "load_tool_policy", return_value=policy), \
         patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}):
        yield tool, dm, cfg


def _turn(cfg, dm, text, *, chat_responses, extract="", origin="voix", db=None):
    with patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value=extract):
        chat.side_effect = chat_responses
        reply = run_reply_engine(
            db=db if db is not None else Mock(), cfg=cfg, tts=None, text=text,
            dialogue_memory=dm, origin=origin,
        )
        return reply, chat


# ── Turn one: she asks ────────────────────────────────────────────────


def test_a_turn_that_needs_permission_ends_by_asking(scene):
    tool, dm, cfg = scene

    reply, _ = _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    assert reply == SPOKEN_TEMPLATE.format(tool="sabotage")
    assert tool.fired == []


def test_she_does_not_go_back_to_the_model_after_asking(scene):
    """One chat call, for the tool call itself. A second would be the
    model getting a chance to rewrite the question."""
    tool, dm, cfg = scene

    _, chat = _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    assert chat.call_count == 1


def test_the_question_is_waiting_afterwards(scene):
    tool, dm, cfg = scene

    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    waiting = dm.peek_pending()
    assert waiting is not None
    assert waiting.tool == "sabotage"
    assert waiting.args == {"cible": "/a"}


# ── Turn two: the answer ──────────────────────────────────────────────


def test_a_spoken_yes_runs_the_pinned_call_once(scene):
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="Voilà, c'est fait.", chat_responses=[
            {"message": {"content": "Voilà, c'est fait."}},
        ])

    assert tool.fired == [{"cible": "/a"}]


def test_a_spoken_no_runs_nothing_and_says_so(scene):
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=DENIED):
        reply, chat = _turn(cfg, dm, "surtout pas", chat_responses=[])

    assert tool.fired == []
    assert reply
    # And it cost no model call: acknowledging a refusal does not need one,
    # and going back to the model would let it re-propose what was refused.
    assert chat.call_count == 0


def test_the_question_is_gone_either_way(scene):
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=DENIED):
        _turn(cfg, dm, "surtout pas", chat_responses=[])

    assert dm.peek_pending() is None


def test_a_yes_two_turns_later_does_not_run_it(scene):
    """The window is the very next turn. A grant that survives an
    intervening exchange answers a question the user has stopped thinking
    about."""
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    _turn(cfg, dm, "quelle heure est-il", extract="Il est midi.",
          chat_responses=[{"message": {"content": "Il est midi."}}])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="ok",
              chat_responses=[{"message": {"content": "ok"}}])

    assert tool.fired == []


def test_the_yes_runs_what_was_shown_not_what_the_model_says_next(scene):
    """She asked about /a and the user said yes to /a. On the next turn
    the model proposes /b. The grant runs /a — the call it was given for
    — and does not stretch to cover /b."""
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="ok", chat_responses=[
            _tool_call_response("sabotage", {"cible": "/b"}),
            {"message": {"content": "ok"}},
        ])

    assert tool.fired == [{"cible": "/a"}]


def test_the_model_cannot_slip_a_second_call_past_the_grant(scene):
    """A resume turn does one thing: run the approved call and say what
    happened. The model gets no tools at all, which is stronger than
    asking about whatever it proposes — one approval, one action, and
    nothing left over to answer."""
    tool, dm, cfg = scene
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", extract="ok", chat_responses=[
            _tool_call_response("sabotage", {"cible": "/b"}),
            {"message": {"content": "ok"}},
        ])

    assert tool.fired == [{"cible": "/a"}]
    assert dm.peek_pending() is None


# ── A gap: the conversation ends before the answer comes ──────────────


def _ledger(db, request_id):
    """The outcomes recorded for one question, oldest first."""
    return [r["outcome"] for r in reversed(db.recent_actions(200))
            if r.get("request_id") == request_id]


def _after_a_silence(dm):
    """What was said falls out of the conversation window, as it does when
    he comes back to her minutes later: the next turn starts a new
    conversation."""
    dm.RECENT_WINDOW_SEC = 0.01
    time.sleep(0.05)
    assert not dm.has_recent_messages()


def _settled_events(capsys):
    from src.jarvis import daemon

    events = []
    for line in capsys.readouterr().out.splitlines():
        if line.startswith(daemon.CHAT_IPC_PREFIX):
            event = json.loads(line[len(daemon.CHAT_IPC_PREFIX):])
            if event.get("type") == "confirm_settled":
                events.append(event["data"])
    return events


@pytest.fixture
def cards():
    """What the surfaces showing a card are told is over, through the
    callback a bundled desktop wires."""
    from src.jarvis import daemon

    settled = []
    daemon.set_confirmation_callbacks(
        on_confirm_settled=lambda rid, outcome: settled.append((rid, outcome))
    )
    yield settled
    daemon.set_confirmation_callbacks()


def test_a_question_the_conversation_outlived_is_closed_as_expired(scene, tmp_path):
    """A question raised before the gap does not survive it, and it has
    to end where the Activity tab can see it end. Dropped without a row,
    it reads as a question she is still waiting on until the next
    start-up sweeps it."""
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))
    _turn(cfg, dm, "lance sabotage sur /a", db=db, chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])
    asked = dm.peek_pending()

    _after_a_silence(dm)
    _turn(cfg, dm, "quelle heure est-il", db=db, extract="Il est midi.",
          chat_responses=[{"message": {"content": "Il est midi."}}])

    assert dm.peek_pending() is None
    assert _ledger(db, asked.request_id) == ["demandé", "expiré"]
    assert tool.fired == []


def test_a_yes_after_the_gap_runs_nothing(scene, tmp_path):
    """The reason the question goes. Without the gap this exact yes runs
    the call: same origin, the very next turn, well inside the deadline.
    It is the gap alone that makes it an answer to nothing."""
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))
    _turn(cfg, dm, "lance sabotage sur /a", db=db, chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])
    asked = dm.peek_pending()

    _after_a_silence(dm)
    with patch("src.jarvis.reply.engine.read_approval", return_value=GRANTED):
        _turn(cfg, dm, "vas-y", db=db, extract="ok",
              chat_responses=[{"message": {"content": "ok"}}])

    assert tool.fired == []
    assert dm.peek_pending() is None
    assert _ledger(db, asked.request_id) == ["demandé", "expiré"]


def test_the_card_of_a_question_the_conversation_outlived_is_withdrawn(scene, cards, capsys):
    """The card is still on screen and its question no longer exists: a
    click on it would answer nothing. Both surfaces are told it lapsed,
    the bus a subprocess desktop reads and the callback a bundled one
    wires."""
    tool, dm, cfg = scene
    # Listening from the first turn, which opens a conversation with nothing
    # waiting: that one must take nothing down.
    _turn(cfg, dm, "lance sabotage sur /a", chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])
    asked = dm.peek_pending()
    _after_a_silence(dm)
    _turn(cfg, dm, "quelle heure est-il", extract="Il est midi.",
          chat_responses=[{"message": {"content": "Il est midi."}}])

    assert cards == [(asked.request_id, "expiré")]
    assert _settled_events(capsys) == [
        {"request_id": asked.request_id, "outcome": "expiré"},
    ]


def test_inside_the_conversation_the_next_turn_leaves_the_question_open(scene, tmp_path, cards):
    """The control: the gap closes it, not any next turn. Something typed
    while a spoken question waits is not an answer to it, and the card
    stays live for a click."""
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))
    _turn(cfg, dm, "lance sabotage sur /a", db=db, chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])
    asked = dm.peek_pending()

    _turn(cfg, dm, "quelle heure est-il", db=db, origin="chat",
          extract="Il est midi.",
          chat_responses=[{"message": {"content": "Il est midi."}}])

    assert dm.peek_pending() is asked
    assert _ledger(db, asked.request_id) == ["demandé"]
    assert cards == []


def test_a_turn_with_nothing_waiting_ends_nothing(scene, tmp_path, cards):
    """Every turn that is not a click passes where an answer is read. With
    no question waiting it writes no ending and takes no card down."""
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))

    _turn(cfg, dm, "quelle heure est-il", db=db, extract="Il est midi.",
          chat_responses=[{"message": {"content": "Il est midi."}}])

    assert db.recent_actions(50) == []
    assert cards == []


# ── A spoken answer: how its question ends ────────────────────────────


def _ask(cfg, dm, db=None):
    _turn(cfg, dm, "lance sabotage sur /a", db=db, chat_responses=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])
    return dm.peek_pending()


# What he says for each reading of it, and what the model says back when
# the turn reaches it. A no ends the turn before any model is called.
_SPOKEN = {
    GRANTED: ("vas-y", "Voilà, c'est fait."),
    DENIED: ("surtout pas", None),
    UNCLEAR: ("euh attends, quelle heure est-il", "Il est midi."),
}


def _answer(cfg, dm, verdict, *, db=None, then=None):
    words, said = _SPOKEN[verdict]
    responses = list(then or []) + ([{"message": {"content": said}}] if said else [])
    with patch("src.jarvis.reply.engine.read_approval", return_value=verdict):
        return _turn(cfg, dm, words, db=db, extract=said or "",
                     chat_responses=responses)


@pytest.mark.parametrize("verdict, ending", [
    pytest.param(UNCLEAR, "illisible", id="unreadable"),
    pytest.param(DENIED, "décliné", id="no"),
    pytest.param(GRANTED, "ok", id="yes"),
])
def test_a_spoken_answer_ends_its_episode(scene, tmp_path, verdict, ending):
    """The question is claimed before the judge reads the answer, so
    nothing else can close it in this session. An answer she could not
    read left open is swept at the next start-up as `expiré`, a word for
    silence: he did answer."""
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))
    asked = _ask(cfg, dm, db)

    _answer(cfg, dm, verdict, db=db)

    assert dm.peek_pending() is None
    assert _ledger(db, asked.request_id) == ["demandé", ending]


@pytest.mark.parametrize("verdict", [
    pytest.param(UNCLEAR, id="unreadable"),
    pytest.param(DENIED, id="no"),
])
def test_the_row_that_ends_it_keeps_the_request_not_the_answer(scene, tmp_path, verdict):
    """The Activity tab shows the ending row with its query, to answer
    "did I ask for this?". That is the request she asked about. What he
    said back is often about something else entirely, and keeping it would
    file an unrelated sentence under a tool for ninety days."""
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))
    asked = _ask(cfg, dm, db)

    _answer(cfg, dm, verdict, db=db)

    ending = next(r for r in db.recent_actions(200)
                  if r.get("request_id") == asked.request_id
                  and r["outcome"] != "demandé")
    assert ending["query"] == asked.query_redacted
    assert _SPOKEN[verdict][0] not in ending["query"]


def test_after_an_answer_she_could_not_read_the_turn_is_answered(scene):
    """Not an answer she can act on, it is still something he said, and
    it gets the reply any query would."""
    tool, dm, cfg = scene
    _ask(cfg, dm)

    reply, _ = _answer(cfg, dm, UNCLEAR)

    assert reply == _SPOKEN[UNCLEAR][1]
    assert tool.fired == []


@pytest.mark.parametrize("verdict, outcome", [
    pytest.param(UNCLEAR, "illisible", id="unreadable"),
    pytest.param(DENIED, "décliné", id="no"),
    pytest.param(GRANTED, "accordé", id="yes"),
])
def test_a_spoken_answer_takes_its_card_down(scene, cards, capsys, verdict, outcome):
    """The card keeps no time of its own: left up, it waits for a click on
    a question that no longer exists. Both surfaces are told, once, and
    told how it ended, as they are after a click."""
    tool, dm, cfg = scene
    asked = _ask(cfg, dm)

    _answer(cfg, dm, verdict)

    assert cards == [(asked.request_id, outcome)]
    assert _settled_events(capsys) == [
        {"request_id": asked.request_id, "outcome": outcome},
    ]


def test_asked_again_after_an_unreadable_answer_is_a_new_episode(scene, tmp_path, cards):
    """The model may propose the same call again on the turn that could
    not be read. The first question is over by then, so this one is a new
    question with its own id and its own card, not the old one reopened."""
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))
    asked = _ask(cfg, dm, db)

    _answer(cfg, dm, UNCLEAR, db=db, then=[
        _tool_call_response("sabotage", {"cible": "/a"}),
    ])

    again = dm.peek_pending()
    assert again is not None and again.request_id != asked.request_id
    assert _ledger(db, asked.request_id) == ["demandé", "illisible"]
    assert _ledger(db, again.request_id) == ["demandé"]
    assert cards == [(asked.request_id, "illisible")]
    assert tool.fired == []


def test_a_click_takes_its_card_down_once(scene, tmp_path, cards):
    """The control for the spoken door. A click is announced by the daemon
    as it claims the question; the turn that then runs the call must not
    announce it again."""
    from src.jarvis import daemon
    from src.jarvis.memory.db import Database

    tool, dm, cfg = scene
    db = Database(str(tmp_path / "ledger.db"))
    asked = _ask(cfg, dm, db)

    saved = (daemon._global_dialogue_memory, daemon._global_db,
             daemon._global_cfg, daemon._global_stop_requested)
    resume = daemon._resume_after_confirmation
    daemon._global_dialogue_memory, daemon._global_db = dm, db
    daemon._global_cfg, daemon._global_stop_requested = cfg, False
    try:
        # The click, with its worker thread held back so the call runs
        # here, where the test can wait for it.
        with patch.object(daemon, "_resume_after_confirmation"):
            assert daemon.resolve_confirmation(asked.request_id, True) == "ok"
        with patch("src.jarvis.reply.engine.chat_with_messages") as chat, \
             patch("src.jarvis.reply.engine.extract_text_from_response",
                   return_value="Voilà, c'est fait."):
            chat.side_effect = [{"message": {"content": "Voilà, c'est fait."}}]
            resume(asked)
    finally:
        (daemon._global_dialogue_memory, daemon._global_db,
         daemon._global_cfg, daemon._global_stop_requested) = saved
        if daemon._chat_query_lock.locked():
            daemon._chat_query_lock.release()

    assert tool.fired == [{"cible": "/a"}]
    assert cards == [(asked.request_id, "accordé")]
