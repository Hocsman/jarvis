"""`examples/config.json` is what the generator writes from today's defaults.

The file is written by `scripts/generate_config_examples.py` from
`export_example_config()`, and nothing ran the generator: the committed copy
carried about half of the defaults, a key that no longer exists, and old
values for the ones it did have. A reader copying from it was copying a
configuration the app has not had for months.

The test below fails the moment the file and the defaults part ways, and says
how to put them back. It pins the mechanism (the file equals the generator's
output) rather than any particular key or value.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "examples" / "config.json"
GENERATOR = REPO_ROOT / "scripts" / "generate_config_examples.py"

PLATFORMS = ("win32", "darwin", "linux")


def _generator():
    spec = importlib.util.spec_from_file_location("generate_config_examples", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example_under(monkeypatch, platform: str):
    from jarvis.config import export_example_config

    monkeypatch.setattr(sys, "platform", platform)
    return export_example_config(include_db_path=False)


def test_the_committed_example_is_what_the_generator_writes():
    generator = _generator()
    text = EXAMPLE.read_text(encoding="utf-8")
    committed = json.loads(text)
    generated = generator.build_example_config()

    missing = sorted(set(generated) - set(committed))
    stale = sorted(set(committed) - set(generated))
    changed = sorted(
        key for key in set(committed) & set(generated)
        if committed[key] != generated[key]
    )

    assert not (missing or stale or changed), (
        "examples/config.json is out of step with the defaults. "
        f"Missing: {missing}. Not a setting any more: {stale}. "
        f"Different value: {changed}. "
        "Regenerate it with: python scripts/generate_config_examples.py")
    # Same order and layout too, so a regeneration leaves a diff of only what
    # changed (line endings are the checkout's business, not the file's).
    assert text.splitlines() == generator.render_example_config(generated).splitlines(), (
        "examples/config.json holds the right settings in a different order or "
        "layout. Regenerate it with: python scripts/generate_config_examples.py")


def test_the_example_is_the_same_whichever_machine_writes_it(monkeypatch, tmp_path):
    """CI writes on Linux and Windows, a developer on any of three platforms.

    A default that depends on the host would make the committed file equal to
    the generator's output on one of them and not the others, so the guard
    above would fail for everyone else. The example leaves such defaults out:
    a reader copying it keeps their own platform's value.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "somebody"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "somebody"))

    written = {platform: _example_under(monkeypatch, platform) for platform in PLATFORMS}

    reference = written[PLATFORMS[0]]
    for platform in PLATFORMS[1:]:
        assert written[platform] == reference, (
            f"the example written on {platform} differs from the one written "
            f"on {PLATFORMS[0]}: a default depends on the host and must be "
            "left out of export_example_config().")


def test_the_example_leaves_out_only_what_depends_on_the_host(monkeypatch):
    """Dropping a default is for the host-dependent ones, never for a missing one."""
    from jarvis.config import get_default_config

    defaults = {}
    for platform in PLATFORMS:
        monkeypatch.setattr(sys, "platform", platform)
        defaults[platform] = get_default_config()
    host_dependent = {
        key for key in defaults[PLATFORMS[0]]
        if len({repr(defaults[platform][key]) for platform in PLATFORMS}) > 1
    }

    committed = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    assert set(defaults[PLATFORMS[0]]) - set(committed) == host_dependent
