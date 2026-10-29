"""The daemon starts the talking face only when asked and when Kyutai can feed it
(`src/jarvis/output/face/face.spec.md`, "Off unless asked for", "Configuration")."""

from types import SimpleNamespace

import pytest

from jarvis import daemon
from jarvis.config import load_settings
from jarvis.output.face import client


class FakeRenderer:
    made = []

    def __init__(self, portrait, framing="head"):
        self.portrait, self.framing = portrait, framing
        self.started = False
        self.delay = 0.75
        FakeRenderer.made.append(self)

    def start(self):
        self.started = True

    def ready(self):
        return True

    def watched(self):
        return True


@pytest.fixture
def renderers(monkeypatch):
    FakeRenderer.made = []
    monkeypatch.setattr(client, "FaceRenderer", FakeRenderer)
    return FakeRenderer.made


def _cfg(**overrides):
    values = dict(face_enabled=True, face_portrait_path="portrait.png", face_framing="face", tts_engine="kyutai")
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("overrides", [
    {"face_enabled": False},
    {"face_portrait_path": ""},
    {"tts_engine": "piper"},
])
def test_no_face_is_started_unless_asked_for_with_a_portrait_and_kyutai(renderers, overrides):
    assert daemon.start_face(_cfg(**overrides)) is None
    assert renderers == []


def test_a_face_asked_for_is_started_with_its_portrait_and_framing(renderers):
    face = daemon.start_face(_cfg())

    assert face is renderers[0] and face.started
    assert (face.portrait, face.framing) == ("portrait.png", "face")


def test_the_face_reaches_the_voice_the_daemon_builds(renderers, tmp_path, monkeypatch):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tts_engine": "kyutai", "face_enabled": True,
                                "face_portrait_path": "portrait.png"}), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    cfg = load_settings()
    face = daemon.start_face(cfg)

    tts = daemon.build_tts_engine(cfg, face=face)

    assert tts.voice_delay() == face.delay


def test_the_face_settings_default_to_off_and_an_unknown_framing_to_the_head(tmp_path, monkeypatch):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"face_framing": "shoulders"}), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))

    cfg = load_settings()

    assert cfg.face_enabled is False
    assert cfg.face_portrait_path == ""
    assert cfg.face_framing == "head"
