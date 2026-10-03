"""The suite's guard against a test that loads a native inference library.

``tests/conftest.py`` imports the autouse fixture below, so it wraps every test
in the suite. For the duration of the test:

* ``onnxruntime`` cannot be imported. ``import onnxruntime`` raises the
  ``ImportError`` a machine without the package gives, and
  ``importlib.util.find_spec("onnxruntime")`` answers ``None``, so code that
  checks before it imports takes its not-installed path;
* ``piper`` is a stand-in: a package holding ``piper.config.SynthesisConfig``
  and ``piper.voice.PiperVoice`` and nothing else. The real package imports
  ``onnxruntime`` when its ``__init__`` runs, so even ``from piper.config
  import SynthesisConfig`` loads the inference runtime, in a test that has no
  use for one.

A test that needs a voice to speak patches its own ``piper.voice`` over the
stand-in, as the Piper tests do, and the stand-in is back when it ends. The
stand-in's ``PiperVoice.load`` refuses, so a test that forgot to patch one
fails the way a model that cannot be read does, and never loads a model.

``tests/test_the_suite_loads_no_native_inference_library.py`` holds the guard
to all of this.
"""

import dataclasses
import importlib.machinery
import sys
import types
from typing import Optional

import pytest

# Packages whose import is refused outright. Each one loads native inference
# code, so a unit test supplies a fake for whatever sits above it.
REFUSED_RUNTIMES = ("onnxruntime",)

VOICE_REFUSAL = (
    "the test suite does not load a Piper voice: a test that needs one "
    "patches its own piper.voice over the stand-in"
)


@dataclasses.dataclass
class SynthesisConfig:
    """The settings the engine hands ``PiperVoice.synthesize``.

    A subset of the fields of piper's own class, by name; a test holds the two
    to that, so the stand-in never accepts a setting the real one would refuse.
    """

    speaker_id: Optional[int] = None
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w_scale: Optional[float] = None


class PiperVoice:
    """A voice that cannot be loaded."""

    @staticmethod
    def load(model_path, config_path=None, **options):
        raise RuntimeError(VOICE_REFUSAL)


def _module(name: str, is_package: bool = False, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=is_package)
    if is_package:
        # An empty search path: `import piper.anything_else` is a plain
        # ModuleNotFoundError instead of a lookup on disk.
        module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def stand_in_modules() -> dict:
    """The modules that take ``piper``'s place, keyed by import name."""
    config = _module("piper.config", SynthesisConfig=SynthesisConfig)
    voice = _module("piper.voice", PiperVoice=PiperVoice)
    package = _module(
        "piper",
        is_package=True,
        config=config,
        voice=voice,
        SynthesisConfig=SynthesisConfig,
        PiperVoice=PiperVoice,
    )
    return {"piper": package, "piper.config": config, "piper.voice": voice}


@pytest.fixture(autouse=True)
def _no_native_inference(monkeypatch):
    for name, module in stand_in_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in REFUSED_RUNTIMES:
        monkeypatch.setitem(sys.modules, name, None)
