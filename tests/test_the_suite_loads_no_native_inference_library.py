"""A unit test loads no native inference library.

A test that reaches for an inference runtime measures the machine it runs on:
the library's DLLs, the C++ runtime they link against, whatever else the
process has already mapped into its address space. A fault while that code
loads is not an exception a test can fail on. It ends the whole pytest
process, with no test to blame beyond the one that happened to be running.

Nothing in a unit test needs one. The voice is a mock, the model is a stub
file, and what is under test is the code around them. So the guard in
``native_guard.py`` (armed for every test from ``conftest.py``) puts a
stand-in where the real ``piper`` package would load ``onnxruntime``, and makes
``onnxruntime`` itself unimportable. These tests hold it to that, and check a
test can still patch over the stand-in the way the piper tests do.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import textwrap
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"


def test_the_guard_is_armed_for_every_test_in_this_suite(request):
    assert "_no_native_inference" in request.fixturenames


class TestTheInferenceRuntime:
    def test_it_cannot_be_imported(self):
        with pytest.raises(ImportError):
            import onnxruntime  # noqa: F401

    def test_it_looks_uninstalled_to_code_that_checks_before_importing(self):
        assert importlib.util.find_spec("onnxruntime") is None

    def test_its_native_module_cannot_be_imported_either(self):
        with pytest.raises(ImportError):
            importlib.import_module("onnxruntime.capi.onnxruntime_pybind11_state")


class TestThePiperATestGets:
    def test_it_offers_the_synthesis_settings_the_engine_passes(self):
        from piper.config import SynthesisConfig

        settings = SynthesisConfig(
            speaker_id=2, length_scale=1.1, noise_scale=0.5, noise_w_scale=0.7
        )

        assert (settings.speaker_id, settings.length_scale) == (2, 1.1)
        assert (settings.noise_scale, settings.noise_w_scale) == (0.5, 0.7)

    def test_it_offers_a_voice_class_that_loads_nothing(self):
        from piper.voice import PiperVoice

        with pytest.raises(RuntimeError, match="does not load a Piper voice"):
            PiperVoice.load("a/voice.onnx", "a/voice.onnx.json")

    def test_it_has_no_submodule_beyond_the_two_it_offers(self):
        with pytest.raises(ImportError):
            importlib.import_module("piper.phonemize_espeak")

    def test_a_test_can_patch_a_voice_over_it_and_it_comes_back_afterwards(self, monkeypatch):
        stand_in = sys.modules["piper.voice"]
        own = types.SimpleNamespace(PiperVoice=object())

        with monkeypatch.context() as inside:
            inside.setitem(sys.modules, "piper.voice", own)
            from piper.voice import PiperVoice

            assert PiperVoice is own.PiperVoice

        assert sys.modules["piper.voice"] is stand_in

    def test_a_test_can_make_the_package_look_uninstalled(self):
        with patch.dict("sys.modules", {"piper": None, "piper.voice": None}):
            with pytest.raises(ImportError):
                from piper.voice import PiperVoice  # noqa: F401


def _installed_piper_config_source():
    """The text of the real ``piper/config.py``, found without importing it
    (importing the package is what loads the inference runtime)."""
    spec = importlib.machinery.PathFinder.find_spec("piper")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(next(iter(spec.submodule_search_locations))) / "config.py"
    return path.read_text(encoding="utf-8") if path.is_file() else None


def test_the_stand_in_settings_claim_only_fields_the_real_package_has():
    """The stand-in must not accept a setting the real class would refuse."""
    import ast

    source = _installed_piper_config_source()
    if source is None:
        pytest.skip("piper-tts is not installed, so there is nothing to compare with")
    real = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "SynthesisConfig"
    )
    real_fields = {
        item.target.id for item in real.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }

    from piper.config import SynthesisConfig

    claimed = {field.name for field in dataclasses.fields(SynthesisConfig)}
    assert claimed <= real_fields, sorted(claimed - real_fields)


INNER_CONFTEST = """
from native_guard import _no_native_inference  # noqa: F401
"""

# The scenario the guard exists for: the engine speaks through a mock voice, and
# on the way to the speakers it imports the piper settings class. Nothing of the
# real package is patched, so whatever that import resolves to is the guard's.
INNER_TESTS = """
import sys
import types
from unittest.mock import MagicMock

import numpy as np

from jarvis.output.tts import PiperTTS


def _loaded(name):
    return sys.modules.get(name) is not None


def test_speaking_through_a_mock_voice_loads_no_inference_runtime(monkeypatch):
    opened = []
    finished = []

    class _Stream:
        active = False

        def __init__(self, samplerate, **kwargs):
            opened.append(samplerate)

        def start(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        OutputStream=_Stream, CallbackAbort=Exception, CallbackStop=Exception))
    monkeypatch.setattr(PiperTTS, "_notify_speaking_state", lambda self, speaking: None)

    engine = PiperTTS(enabled=True)
    engine._voice = MagicMock()
    engine._sample_rate = 16000
    engine._voice.synthesize.return_value = [
        types.SimpleNamespace(audio_int16_array=np.zeros(1600, dtype=np.int16))
    ]
    engine._initialized = True
    engine._completion_callback = lambda: finished.append(True)

    engine._speak_once("Hello test")

    assert opened == [16000], "the utterance never reached the speakers"
    assert finished == [True]
    assert not _loaded("onnxruntime")
    assert not _loaded("onnxruntime.capi.onnxruntime_pybind11_state")
"""


def _run_inner_session(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(textwrap.dedent(INNER_CONFTEST), encoding="utf-8")
    (tmp_path / "test_inner.py").write_text(textwrap.dedent(INNER_TESTS), encoding="utf-8")
    report = tmp_path / "report.xml"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(TESTS_DIR), str(SRC_DIR)]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         f"--junitxml={report}", "test_inner.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180,
    )
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        failures = [child for child in case if child.tag in ("error", "failure")]
        outcomes[case.get("name")] = "\n".join(
            (failure.get("message") or "") + (failure.text or "") for failure in failures
        )
    return outcomes


class TestTheGuardEndToEnd:
    @pytest.fixture(scope="class")
    def inner(self, tmp_path_factory):
        return _run_inner_session(tmp_path_factory.mktemp("inner_session"))

    def test_speaking_through_a_mock_voice_loads_no_inference_runtime(self, inner):
        failure = inner["test_speaking_through_a_mock_voice_loads_no_inference_runtime"]
        assert failure == "", failure
