"""What an adversarial reading of this feature turned up.

Every case here is a defect that shipped and was found by attacking the
code rather than by writing it. They share a shape: the refusal half was
sound — no route was found by which a tool ran without a deliberate
decision — and the failures were all in the other half, turning a
correct yes into a result, or telling the user what became of their
answer.

That asymmetry is worth remembering. A system that discards correct human
input is trusted less than one that occasionally errs, and this channel
was discarding several kinds.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from rapidfuzz import fuzz

from src.jarvis.listening.echo_detection import EchoDetector
from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.tools.base import Tool
from src.jarvis.tools.confirmation import (
    CHANNEL_GESTE,
    CHANNEL_PAROLE,
    LIKELY_ANSWERS,
    channel_for_call,
    describe_action,
)
from src.jarvis.tools.policy import RISK_ACTION, RISK_DESTRUCTIVE, ToolPolicy
from src.jarvis.tools.types import ToolExecutionResult


# ── A tool name is third-party text ───────────────────────────────────


@pytest.mark.parametrize("name", [
    "okta__list_users", "bookmarks__search", "webhookSend",
    "notebook__append", "getToken", "slack__lookup_user",
])
def test_a_tool_name_cannot_make_her_delete_your_answer(name):
    """Any name containing "ok" pushes a spoken "ok" to 100 against her
    own question, and the listener drops it as echo before the judge sees
    it. The user answers, nothing happens, nothing says why."""
    spoken = describe_action(name, {}, RISK_ACTION).spoken
    worst = max(fuzz.partial_ratio(a, spoken.lower()) for a in LIKELY_ANSWERS)

    assert worst < EchoDetector.PURE_ECHO_THRESHOLD


def test_an_ordinary_name_is_still_spoken():
    """The fallback must not swallow every name — hearing which tool she
    means is most of the value of asking aloud."""
    assert "localFiles" in describe_action("localFiles", {}, RISK_ACTION).spoken


def test_a_newline_in_a_tool_name_cannot_forge_a_card_heading():
    """The card's first line is the heading. A name carrying a newline
    would write a second one saying whatever it liked."""
    d = describe_action("evil\nlocalFiles · lecture", {}, RISK_DESTRUCTIVE)

    assert d.shown.count("\n") == 1


def test_an_unusual_tool_name_is_flagged():
    d = describe_action("evil\nname", {}, RISK_DESTRUCTIVE)

    assert d.hazards


def test_an_unusual_tool_name_is_not_read_aloud():
    d = describe_action("evil\nname", {}, RISK_ACTION)

    assert "evil" not in d.spoken


# ── Invisible characters, the modern set ──────────────────────────────


@pytest.mark.parametrize("ch,label", [
    ("⁦", "LRI"), ("⁧", "RLI"), ("⁨", "FSI"), ("⁩", "PDI"),
    ("‮", "RLO"), ("​", "zero-width space"), ("­", "soft hyphen"),
    (" ", "line separator"), (" ", "paragraph separator"),
    ("\U000e0041", "tag letter"),
])
def test_every_hiding_character_is_flagged_and_made_visible(ch, label):
    """The isolates are the half now recommended over the overrides, and
    an enumerated range list had missed exactly them. Categories, not
    ranges: a range list is always a Unicode revision behind."""
    args = {"path": f"/a{ch}b"}
    d = describe_action("localFiles", args, RISK_DESTRUCTIVE)

    assert d.hazards, f"{label} not flagged"
    assert ch not in d.shown, f"{label} shown raw"


@pytest.mark.parametrize("ch", ["⁧", "\U000e0041", " "])
def test_making_it_visible_keeps_the_card_parseable(ch):
    """Above the BMP that means a surrogate pair — a bare five-digit
    `\\uXXXXX` is not valid JSON, and the card's copy has to keep parsing
    back to exactly what would run."""
    args = {"path": f"/a{ch}b"}
    d = describe_action("localFiles", args, RISK_DESTRUCTIVE)

    assert json.loads(d.shown.split("\n", 1)[1]) == args


def test_a_tab_is_not_treated_as_a_hiding_character():
    """Real whitespace is already shown as a JSON escape. Flagging it
    would make the hazard strip fire on ordinary arguments."""
    d = describe_action("localFiles", {"path": "/a\tb"}, RISK_DESTRUCTIVE)

    assert not d.hazards


# ── An unreadable action is not offered the voice door ────────────────


class _Builtin(Tool):
    def risk_for(self, args):
        return RISK_ACTION

    @property
    def name(self):
        return "fake"

    @property
    def description(self):
        return ""

    @property
    def inputSchema(self):
        return {"type": "object", "properties": {}}

    def run(self, args, context):
        raise NotImplementedError


def test_an_action_she_cannot_read_honestly_needs_a_click():
    """An approval given to a sentence that conceals its own meaning is
    not consent."""
    channel = channel_for_call(
        RISK_ACTION, _Builtin(), "fake", {"path": "/a‮b"},
    )

    assert channel == CHANNEL_GESTE


def test_an_ordinary_action_still_takes_a_spoken_answer():
    assert channel_for_call(
        RISK_ACTION, _Builtin(), "fake", {"path": "/a"},
    ) == CHANNEL_PAROLE


# ── The policy file cannot break every tool call ──────────────────────


def test_a_policy_file_that_is_not_utf8_yields_the_defaults(tmp_path):
    """`read_text` raises UnicodeDecodeError — a ValueError, not an
    OSError — which a user reaches by reopening their own generated
    policy in a Windows editor and saving it as ANSI. Escaping would
    break every tool call on every turn until the file was repaired."""
    from src.jarvis.tools import registry

    class _Cfg:
        db_path = str(tmp_path / "jarvis.db")

    path = tmp_path / "yuba" / "outils.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("## Libre\n- webSearch  caf\xe9\n".encode("cp1252"))
    registry._POLICY_CACHE["stamp"] = None

    policy = registry.load_tool_policy(_Cfg())

    assert isinstance(policy, ToolPolicy)
    assert policy.verdict("localFiles", RISK_DESTRUCTIVE) == "demande"


# ── A collision does not advise weakening the policy ──────────────────


class _Cfg:
    mcps = {}
    voice_debug = False
    db_path = "/tmp/does-not-exist/jarvis.db"
    confirmation_ttl_sec = 180.0


def _run(name, tool, args, confirmation, db):
    from src.jarvis.tools import registry

    with patch.object(registry, "BUILTIN_TOOLS",
                      {**registry.BUILTIN_TOOLS, name: tool}), \
         patch.object(registry, "load_tool_policy", return_value=ToolPolicy.empty()):
        return registry.run_tool_with_retries(
            db=db, cfg=_Cfg(), tool_name=name, tool_args=args,
            system_prompt="", original_prompt="", redacted_text="fais-le",
            max_retries=1, origin="voix", confirmation=confirmation,
        )


@pytest.fixture
def collided():
    """One question waiting, and a second, different one arriving."""
    from src.jarvis.tools.confirmation import Confirmation

    dm = DialogueMemory()
    dm.begin_turn()
    db = MagicMock()
    conf = Confirmation(store=dm, publish=lambda a: None, ttl_sec=180.0)

    _run("fake", _Builtin(), {"cible": "/a"}, conf, db)
    db.reset_mock()
    second = _run("fake", _Builtin(), {"cible": "/b"}, conf, db)
    return dm, db, second


def test_a_collision_does_not_tell_the_user_to_open_up_their_policy(collided):
    """`## Libre` is the advice for "nothing can ask". Here something is
    asking — the user is mid-answer — and advising them to weaken the
    policy to get past their own question is the worst possible reading
    of the situation."""
    _, _, second = collided

    assert "outils.md" not in (second.reply_text or "")
    assert "Libre" not in (second.reply_text or "")


def test_a_collision_names_what_is_still_waiting(collided):
    _, _, second = collided

    assert "fake" in (second.reply_text or "")


def test_a_collision_leaves_a_ledger_row(collided):
    """Silently dropping it makes the ledger claim she never tried."""
    _, db, _ = collided

    assert db.record_action.called


def test_a_collision_does_not_displace_the_waiting_question(collided):
    dm, _, _ = collided

    assert dm.peek_pending().args == {"cible": "/a"}


# ── Turns are counted per channel ─────────────────────────────────────


def test_a_turn_on_the_other_channel_does_not_expire_a_spoken_yes():
    """One global counter would let a chat turn — or the resume worker's
    own turn — advance the number past a voice question nobody had a
    chance to answer, disqualifying a perfectly timely yes."""
    from src.jarvis.tools.confirmation import PendingAction

    dm = DialogueMemory()
    dm.begin_turn("voix")
    dm.raise_pending(PendingAction.create(
        tool="fake", args={}, risk=RISK_ACTION, channel=CHANNEL_PAROLE,
        origin="voix", query_redacted="", raised_at_turn=dm.current_turn("voix"),
        ttl_sec=180.0,
    ))

    dm.begin_turn("chat")  # someone typed something meanwhile
    seq = dm.begin_turn("voix")

    assert dm.take_pending_for_utterance("voix", seq) is not None


# ── A re-ask belongs to whoever just asked ────────────────────────────


def test_a_re_ask_from_the_other_surface_can_be_answered_there():
    """Recognised as the same call by its digest, which carries only the
    tool and arguments — so without moving the origin, the surface that
    just asked could not accept the answer it is about to get."""
    from src.jarvis.tools.confirmation import PendingAction

    dm = DialogueMemory()
    dm.begin_turn("voix")
    dm.raise_pending(PendingAction.create(
        tool="fake", args={"x": 1}, risk=RISK_ACTION, channel=CHANNEL_PAROLE,
        origin="voix", query_redacted="", raised_at_turn=dm.current_turn("voix"),
        ttl_sec=180.0,
    ))

    seq = dm.begin_turn("chat")
    dm.raise_pending(PendingAction.create(
        tool="fake", args={"x": 1}, risk=RISK_ACTION, channel=CHANNEL_PAROLE,
        origin="chat", query_redacted="", raised_at_turn=seq, ttl_sec=180.0,
    ))

    assert dm.take_pending_for_utterance("chat", seq + 1) is not None


# ── An unanswered call is not offered as carryover ────────────────────


def test_a_question_leaves_no_orphan_tool_call_behind():
    """The assistant message carrying the tool_calls array is appended
    before the gate runs. Left in place when nothing ran, it becomes
    carryover: a call with no result, offered to the next turn as if it
    had happened."""
    from src.jarvis.reply.engine import _end_turn_with_question

    messages = [
        {"role": "user", "content": "fais-le"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "fake", "arguments": {}}},
        ]},
    ]
    result = ToolExecutionResult(success=False, reply_text=None,
                                 pending_id="cf_x")

    _end_turn_with_question(
        result, None, "fais-le", tts=None, cfg=MagicMock(voice_debug=False),
        record_carryover=lambda: None, messages=messages,
    )

    assert not any(m.get("tool_calls") for m in messages)
