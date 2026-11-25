"""Whether his turn is over, read from his voice (`jarvis.listening.turn_end`).

The end-of-turn model is Smart Turn v3.2, an ONNX classifier that reads the
last eight seconds of his voice and says how likely it is that he has
finished. These tests hold what the listener relies on: the model file is
fetched from one pinned revision and never loaded unless it matches its
published digest, the audio is prepared the way the model was trained on,
and a detector that cannot give a verdict says so with None instead of
raising.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from jarvis.listening import turn_end


def _tone(seconds: float, seed: int = 0) -> np.ndarray:
    """A voiced-looking signal: a few harmonics that wander, and a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * turn_end.SAMPLE_RATE)) / turn_end.SAMPLE_RATE
    pitch = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / turn_end.SAMPLE_RATE
    voice = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.1 * voice + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


# ── The audio the model reads ─────────────────────────────────────────


class TestWhatTheModelReads:
    def test_a_short_turn_is_preceded_by_silence_up_to_the_window(self):
        sound = _tone(2.5)
        window = turn_end.last_window(sound)
        assert window.size == turn_end.WINDOW_SAMPLES
        assert np.array_equal(window[-sound.size:], sound)
        assert not window[:-sound.size].any()

    def test_a_long_turn_keeps_only_its_last_seconds(self):
        sound = _tone(11.0)
        window = turn_end.last_window(sound)
        assert np.array_equal(window, sound[-turn_end.WINDOW_SAMPLES:])

    @pytest.mark.parametrize("seconds", [0.4, 2.5, 8.0, 11.0])
    def test_features_match_the_extractor_the_model_was_trained_with(self, seconds):
        """The model was trained on `transformers`' WhisperFeatureExtractor,
        fed the window zero-padded at the start and normalised. Ours must
        give the same features without importing transformers."""
        transformers = pytest.importorskip("transformers")
        extractor = transformers.WhisperFeatureExtractor(chunk_length=turn_end.WINDOW_SECONDS)
        sound = _tone(seconds, seed=int(seconds * 10))
        reference = extractor(
            turn_end.last_window(sound), sampling_rate=turn_end.SAMPLE_RATE,
            return_tensors="np", padding="max_length", max_length=turn_end.WINDOW_SAMPLES,
            truncation=True, do_normalize=True,
        ).input_features[0]

        ours = turn_end.log_mel(sound)

        assert ours.dtype == np.float32
        assert ours.shape == reference.shape
        np.testing.assert_allclose(ours, reference, atol=1e-3)


# ── The model file ────────────────────────────────────────────────────


def _publish(monkeypatch, content: bytes) -> None:
    """Pretend *content* is the published file."""
    monkeypatch.setattr(turn_end, "SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(turn_end, "SIZE", len(content))


class _Hub:
    """A download that writes *content* where huggingface_hub would, and
    remembers what it was asked."""

    def __init__(self, content: bytes = b"model", fails: bool = False):
        self.content, self.fails, self.calls = content, fails, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fails:
            raise OSError("no network")
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content)
        return str(path)


class TestTheModelFile:
    def test_it_is_fetched_from_the_pinned_revision_without_a_token(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")
        hub = _Hub(b"model")

        path = turn_end.fetch(tmp_path, download=hub)

        assert path.read_bytes() == b"model"
        assert path.parent == tmp_path
        (call,) = hub.calls
        assert call["repo_id"] == turn_end.REPO
        assert call["filename"] == turn_end.FILE
        assert call["revision"] == turn_end.REVISION
        assert call["token"] is False

    def test_a_file_already_in_place_is_not_fetched_again(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")
        turn_end.fetch(tmp_path, download=_Hub(b"model"))
        hub = _Hub(b"model")

        turn_end.fetch(tmp_path, download=hub)

        assert hub.calls == []

    def test_a_file_that_does_not_match_its_digest_is_deleted_never_kept(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")

        with pytest.raises(RuntimeError, match="digest"):
            turn_end.fetch(tmp_path, download=_Hub(b"tampered"))

        assert not (tmp_path / turn_end.FILE).exists()
        assert not any(p.is_file() for p in tmp_path.rglob("*"))

    def test_a_failed_download_says_so(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")

        with pytest.raises(RuntimeError, match="download"):
            turn_end.fetch(tmp_path, download=_Hub(fails=True))

    def test_a_file_in_place_that_does_not_match_is_fetched_again(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")
        (tmp_path / turn_end.FILE).write_bytes(b"tampered")
        hub = _Hub(b"model")

        path = turn_end.fetch(tmp_path, download=hub)

        assert len(hub.calls) == 1
        assert path.read_bytes() == b"model"

    def test_a_file_in_place_that_does_not_match_is_never_kept(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")
        (tmp_path / turn_end.FILE).write_bytes(b"tampered")

        with pytest.raises(RuntimeError, match="download"):
            turn_end.fetch(tmp_path, download=_Hub(fails=True))

        assert not (tmp_path / turn_end.FILE).exists()

    def test_the_download_is_announced_each_time_the_file_is_fetched(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")
        announced = []

        turn_end.fetch(tmp_path, download=_Hub(b"model"), on_download=lambda: announced.append(1))
        turn_end.fetch(tmp_path, download=_Hub(b"model"), on_download=lambda: announced.append(2))
        (tmp_path / turn_end.FILE).write_bytes(b"tampered")
        turn_end.fetch(tmp_path, download=_Hub(b"model"), on_download=lambda: announced.append(3))

        assert announced == [1, 3], "a file in place was announced, or a fetch was not"

    def test_in_the_suite_the_model_is_kept_in_the_sandbox(self, tmp_path_factory, monkeypatch):
        """Running the suite is not permission to write into the user's
        models: a detector built without a folder fetches into the run's
        sandbox, and a listener builds it that way."""
        _publish(monkeypatch, b"model")
        hub = _Hub(b"model")

        turn_end.TurnEndDetector(download=hub, open_session=lambda path: _Session()).load()

        (call,) = hub.calls
        sandbox = tmp_path_factory.getbasetemp().resolve()
        assert Path(call["local_dir"]).resolve().is_relative_to(sandbox), call["local_dir"]


# ── The detector ──────────────────────────────────────────────────────


class _Session:
    """An ONNX session that answers *probability* and keeps its inputs."""

    def __init__(self, probability=0.8, fails=False):
        self.probability, self.fails, self.inputs = probability, fails, []

    def run(self, _outputs, feeds):
        self.inputs.append(feeds)
        if self.fails:
            raise RuntimeError("onnx failed")
        return [np.array([[self.probability]], dtype=np.float32)]


def _detector(tmp_path, monkeypatch, session=None, hub=None):
    _publish(monkeypatch, b"model")
    session = session or _Session()
    return turn_end.TurnEndDetector(
        root=tmp_path, download=hub or _Hub(b"model"),
        open_session=lambda path: session,
    ), session


class TestTheDetector:
    def test_no_verdict_before_it_is_loaded(self, tmp_path, monkeypatch):
        detector, session = _detector(tmp_path, monkeypatch)

        assert detector.probability(_tone(2.0)) is None
        assert session.inputs == []

    def test_once_loaded_it_gives_the_models_probability(self, tmp_path, monkeypatch):
        detector, session = _detector(tmp_path, monkeypatch, _Session(0.83))

        assert detector.load() is True
        probability = detector.probability(_tone(2.0))

        assert probability == pytest.approx(0.83)
        (feeds,) = session.inputs
        (features,) = feeds.values()
        assert features.shape == (1, 80, 100 * turn_end.WINDOW_SECONDS)

    def test_a_model_that_cannot_be_fetched_leaves_it_without_a_verdict(self, tmp_path, monkeypatch):
        detector, _session = _detector(tmp_path, monkeypatch, hub=_Hub(fails=True))

        assert detector.load() is False
        assert detector.status == "unavailable"
        assert "download" in detector.why
        assert detector.probability(_tone(2.0)) is None

    def test_a_model_that_will_not_open_leaves_it_without_a_verdict(self, tmp_path, monkeypatch):
        _publish(monkeypatch, b"model")

        def broken(_path):
            raise RuntimeError("not an onnx file")

        detector = turn_end.TurnEndDetector(root=tmp_path, download=_Hub(b"model"),
                                            open_session=broken)
        assert detector.load() is False
        assert detector.probability(_tone(2.0)) is None

    def test_a_failing_model_gives_no_verdict_and_never_raises(self, tmp_path, monkeypatch):
        detector, _session = _detector(tmp_path, monkeypatch, _Session(fails=True))
        detector.load()

        assert detector.probability(_tone(2.0)) is None

    @pytest.mark.parametrize("answer", [float("nan"), float("inf")])
    def test_an_answer_that_is_not_a_probability_is_no_verdict(self, tmp_path, monkeypatch, answer):
        detector, _session = _detector(tmp_path, monkeypatch, _Session(answer))
        detector.load()

        assert detector.probability(_tone(2.0)) is None

    def test_it_says_when_it_has_settled(self, tmp_path, monkeypatch):
        """Whoever waits for the model hears once whether it is ready or
        not, after loading has ended."""
        _publish(monkeypatch, b"model")
        settled = []
        ready = turn_end.TurnEndDetector(root=tmp_path / "a", download=_Hub(b"model"),
                                         open_session=lambda path: _Session(),
                                         on_settled=settled.append)
        broken = turn_end.TurnEndDetector(root=tmp_path / "b", download=_Hub(fails=True),
                                          on_settled=settled.append)

        ready.load()
        broken.load()

        assert [d.status for d in settled] == ["ready", "unavailable"]

    def test_nothing_to_hear_gives_no_verdict(self, tmp_path, monkeypatch):
        detector, session = _detector(tmp_path, monkeypatch)
        detector.load()

        assert detector.probability(np.zeros(0, dtype=np.float32)) is None
        assert detector.probability(None) is None
        assert session.inputs == []

    def test_loading_on_its_own_thread_ends_ready(self, tmp_path, monkeypatch):
        detector, _session = _detector(tmp_path, monkeypatch)

        detector.start().join(timeout=5)

        assert detector.status == "ready"


    def test_without_onnxruntime_it_has_no_verdict(self, tmp_path, monkeypatch):
        """The suite refuses onnxruntime (`native_guard.py`), as a machine
        without it would: the model file is there, and still no verdict."""
        _publish(monkeypatch, b"model")
        detector = turn_end.TurnEndDetector(root=tmp_path, download=_Hub(b"model"))

        assert detector.load() is False
        assert detector.status == "unavailable"
        assert detector.probability(_tone(2.0)) is None


# ── The settings ──────────────────────────────────────────────────────


class TestTheSettings:
    def test_his_values_reach_the_settings(self, tmp_path, monkeypatch):
        from test_reglages_qui_ne_font_rien import _charge

        cfg = _charge(tmp_path, monkeypatch, {"turn_detector_enabled": False,
                                              "turn_unfinished_wait_seconds": 6.5})

        assert cfg.turn_detector_enabled is False
        assert cfg.turn_unfinished_wait_seconds == 6.5

    def test_off_written_as_text_is_off(self, tmp_path, monkeypatch):
        from test_reglages_qui_ne_font_rien import _charge

        cfg = _charge(tmp_path, monkeypatch, {"turn_detector_enabled": "false"})

        assert cfg.turn_detector_enabled is False

    def test_a_negative_wait_is_no_wait(self, tmp_path, monkeypatch):
        from test_reglages_qui_ne_font_rien import _charge

        cfg = _charge(tmp_path, monkeypatch, {"turn_unfinished_wait_seconds": -2})

        assert cfg.turn_unfinished_wait_seconds == 0.0
