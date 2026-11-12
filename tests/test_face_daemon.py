"""The daemon starts the talking face only when asked and when Kyutai can feed it
(`src/jarvis/output/face/face.spec.md`, "Off unless asked for", "Configuration")."""

import json
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


@pytest.fixture
def portrait(tmp_path):
    path = tmp_path / "portrait.png"
    path.write_bytes(b"a picture")
    return str(path)


def _cfg(portrait, **overrides):
    values = dict(face_enabled=True, face_portrait_path=portrait, face_framing="face",
                  tts_engine="kyutai", tts_enabled=True)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("overrides", [
    {"face_enabled": False},
    {"face_portrait_path": ""},
    {"tts_engine": "piper"},
    {"tts_enabled": False},
])
def test_no_face_is_started_unless_asked_for_with_a_portrait_and_the_kyutai_voice_on(renderers, portrait, overrides):
    assert daemon.start_face(_cfg(portrait, **overrides)) is None
    assert renderers == []


def test_a_portrait_that_is_not_there_starts_nothing_and_says_why_without_the_path(renderers, tmp_path, capsys):
    missing = str(tmp_path / "nowhere" / "portrait.png")

    assert daemon.start_face(_cfg(missing)) is None

    assert renderers == []
    out = capsys.readouterr().out
    assert "portrait" in out and "nowhere" not in out


def test_a_face_asked_for_is_started_with_its_portrait_and_framing(renderers, portrait):
    face = daemon.start_face(_cfg(portrait))

    assert face is renderers[0] and face.started
    assert (face.portrait, face.framing) == (portrait, "face")


def test_the_face_reaches_the_voice_the_daemon_builds(renderers, portrait, tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"tts_engine": "kyutai", "face_enabled": True,
                                "face_portrait_path": portrait}), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    cfg = load_settings()
    face = daemon.start_face(cfg)

    tts = daemon.build_tts_engine(cfg, face=face)

    assert tts.voice_delay() == face.delay


def test_the_face_settings_default_to_off_and_an_unknown_framing_to_the_head(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"face_framing": "shoulders"}), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))

    cfg = load_settings()

    assert cfg.face_enabled is False
    assert cfg.face_portrait_path == ""
    assert cfg.face_framing == "head"
