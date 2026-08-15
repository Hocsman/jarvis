"""Behaviour tests for preparation-phase progress reporting.

Several seconds pass between submitting a query and the first generated
token (tool routing, memory lookup, tool execution). The engine announces
each phase through ``on_stage``; the daemon forwards it as a ``stage`` IPC
event so a UI can show what is happening instead of a frozen screen.

Two properties are load-bearing:

* Stage ids are **neutral identifiers**, never display strings — the
  assistant is not tied to one language, so wording belongs to the
  presenting layer (see ``_STAGE_VIEW`` in the dashboard bridge).
* Reporting is advisory: a failing consumer must never cost a reply.
"""

from __future__ import annotations

import json
import time

from src.jarvis import daemon


def _install_memory():
    from src.jarvis.memory.conversation import DialogueMemory
    daemon._global_dialogue_memory = DialogueMemory(inactivity_timeout=300, max_interactions=20)
    daemon._global_cfg = object()
    daemon._global_db = object()
    daemon._global_stop_requested = False


def _reset():
    daemon._global_dialogue_memory = None
    daemon._global_cfg = None
    daemon._global_db = None
    daemon._global_stop_requested = False


def _events(capsys, timeout=5.0):
    deadline = time.time() + timeout
    collected = []
    while time.time() < deadline:
        for ln in capsys.readouterr().out.splitlines():
            if ln.startswith(daemon.CHAT_IPC_PREFIX):
                try:
                    collected.append(json.loads(ln[len(daemon.CHAT_IPC_PREFIX):]))
                except json.JSONDecodeError:
                    pass
        if any(e.get("type") == "complete" for e in collected):
            return collected
        time.sleep(0.02)
    return collected


class TestStageIPC:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_stages_are_emitted_before_the_reply(self, monkeypatch, capsys):
        _install_memory()

        def fake_engine(*a, on_stage=None, on_token=None, **k):
            if on_stage:
                on_stage("routing", None)
                on_stage("tool", "getWeather")
                on_stage("generating", None)
            if on_token:
                on_token("Il fait beau")
            return "Il fait beau"

        monkeypatch.setattr("src.jarvis.reply.engine.run_reply_engine", fake_engine)
        daemon.submit_text_query("météo ?", use_ipc=True)
        events = _events(capsys)

        stages = [e["data"] for e in events if e["type"] == "stage"]
        assert [s["stage"] for s in stages] == ["routing", "tool", "generating"]
        # The tool stage carries which tool is running.
        assert stages[1]["detail"] == "getWeather"

        kinds = [e["type"] for e in events]
        assert kinds.index("stage") < kinds.index("complete")

    def test_stage_ids_are_identifiers_not_display_text(self, monkeypatch, capsys):
        # Guards the multilingual constraint: the engine must not ship wording.
        _install_memory()

        def fake_engine(*a, on_stage=None, **k):
            if on_stage:
                on_stage("routing", None)
            return "ok"

        monkeypatch.setattr("src.jarvis.reply.engine.run_reply_engine", fake_engine)
        daemon.submit_text_query("salut", use_ipc=True)
        stages = [e["data"]["stage"] for e in _events(capsys) if e["type"] == "stage"]
        assert stages == ["routing"]
        for s in stages:
            assert s.isidentifier(), f"{s!r} looks like prose, not an id"

    def test_engine_without_on_stage_still_replies(self, monkeypatch, capsys):
        _install_memory()
        monkeypatch.setattr(
            "src.jarvis.reply.engine.run_reply_engine",
            lambda db, cfg, tts, text, dialogue_memory, language=None, origin=None: "ok",
        )
        daemon.submit_text_query("salut", use_ipc=True)
        complete = [e for e in _events(capsys) if e["type"] == "complete"]
        assert complete and complete[-1]["data"] == "ok"


class TestStageLabels:
    """The dashboard owns the wording — one label per known stage id."""

    def test_every_stage_id_has_a_label(self):
        from desktop_app.dashboard.bridge import _STAGE_VIEW, stage_label

        for stage_id in ("routing", "memory", "planning", "tool", "generating"):
            assert stage_id in _STAGE_VIEW, f"no label for {stage_id}"
            assert stage_label(stage_id, None)

    def test_tool_label_includes_the_tool_name(self):
        from desktop_app.dashboard.bridge import stage_label

        assert "getWeather" in stage_label("tool", "getWeather")

    def test_unknown_stage_falls_back_without_crashing(self):
        from desktop_app.dashboard.bridge import stage_label

        assert stage_label("something_new", None)


class TestBridgeStageSignal:
    def test_stage_is_emitted_to_the_page(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        import sys
        if QApplication.instance() is None:
            QApplication(sys.argv)
        from desktop_app.dashboard.bridge import DashboardBridge, dispatch_chat_ipc_line
        from jarvis.daemon import CHAT_IPC_PREFIX

        bridge = DashboardBridge()
        seen = []
        bridge.stageChanged.connect(seen.append)
        line = CHAT_IPC_PREFIX + json.dumps(
            {"type": "stage", "data": {"stage": "routing", "detail": None}}
        )
        assert dispatch_chat_ipc_line(line, bridge) is True
        assert seen and "outil" in seen[0].lower()
