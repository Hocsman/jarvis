"""Yuba never asks Ollama to keep a model for less time than the server would.

A request that names a residency (`keep_alive`) sets it for the model it
loads or uses, in place of the server's own default, and a request that
names none leaves a loaded model's residency as it was. Yuba names 30
minutes on its start-up warm-ups and on each intent-judge call. On a server
set to keep models longer (`OLLAMA_KEEP_ALIVE`, 24 hours on the machine this
was found on), that cut the residency short: the router's model left memory
half an hour after the start-up warm-up, and the first exchange after a
longer pause loaded it again inside the router's deadline.

So when the server's residency, read from `OLLAMA_KEEP_ALIVE` in Yuba's own
environment, is longer, it is the one asked for. Low-power mode keeps its
short residency: there the user asked Yuba for less.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from jarvis.listening.intent_judge import (
    DEFAULT_OLLAMA_KEEP_ALIVE,
    LOW_POWER_OLLAMA_KEEP_ALIVE,
    IntentJudge,
    IntentJudgeConfig,
)
from jarvis.listening.transcript_buffer import TranscriptSegment


def _warm_up_asks(low_power_mode: bool = False) -> str:
    """The residency the judge's start-up warm-up asks the backend for."""
    cfg = SimpleNamespace(llm_provider="ollama", llm_base_url="", low_power_mode=low_power_mode)
    judge = IntentJudge(IntentJudgeConfig(model="gemma4:e2b", cfg=cfg))
    backend = MagicMock()
    backend.warm_up.return_value = True
    with patch("jarvis.listening.intent_judge.get_auxiliary_backend", return_value=backend):
        judge.warm_up()
    return backend.warm_up.call_args.kwargs["keep_alive"]


def _judge_call_asks(low_power_mode: bool = False) -> str:
    """The residency an intent-judge call asks the backend for."""
    cfg = SimpleNamespace(low_power_mode=low_power_mode)
    judge = IntentJudge(IntentJudgeConfig(cfg=cfg))
    backend = MagicMock()
    backend.chat.return_value = {"message": {"content": (
        '{"directed": true, "query": "time", "stop": false, '
        '"confidence": "high", "reasoning": "ok"}'
    )}}
    with patch("jarvis.listening.intent_judge.get_auxiliary_backend", return_value=backend):
        judge.judge([TranscriptSegment("jarvis time", 1000.0, 1001.0)])
    return backend.chat.call_args.kwargs["extra_options"]["keep_alive"]


@pytest.mark.unit
@pytest.mark.parametrize("server, asked", [
    ("24h", "24h"),
    ("24h0m0s", "24h0m0s"),
    ("1h30m", "1h30m"),
    ("2.5h", "2.5h"),
    ("-1m", "-1m"),
    # Whole seconds, which the server reads and the API does not: sent with
    # their unit.
    ("86400", "86400s"),
    ("-1", "-1s"),
])
def test_a_longer_residency_the_server_is_set_to_is_the_one_asked_for(monkeypatch, server, asked):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", server)

    assert _warm_up_asks() == asked
    assert _judge_call_asks() == asked


@pytest.mark.unit
@pytest.mark.parametrize("server", ["5m", "1800", "0", "", "for ever", "10x", "h"])
def test_a_shorter_or_unreadable_server_residency_leaves_yubas_own(monkeypatch, server):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", server)

    assert _warm_up_asks() == DEFAULT_OLLAMA_KEEP_ALIVE
    assert _judge_call_asks() == DEFAULT_OLLAMA_KEEP_ALIVE


@pytest.mark.unit
def test_with_no_server_residency_set_yubas_own_is_asked_for(monkeypatch):
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)

    assert _warm_up_asks() == DEFAULT_OLLAMA_KEEP_ALIVE


@pytest.mark.unit
def test_low_power_mode_keeps_its_short_residency_whatever_the_server_says(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "24h")

    assert _warm_up_asks(low_power_mode=True) == LOW_POWER_OLLAMA_KEEP_ALIVE
    assert _judge_call_asks(low_power_mode=True) == LOW_POWER_OLLAMA_KEEP_ALIVE
