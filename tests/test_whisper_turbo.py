"""Whether the installed faster-whisper can load large-v3-turbo.

The setup wizard asks on its UI thread, so the answer comes from the
package's installed version without importing it (importing faster-whisper
loads ctranslate2). A missing or unreadable install reads as unsupported.
"""

import sys
from unittest.mock import patch

import pytest

from jarvis.listening.whisper_turbo import is_faster_whisper_turbo_supported


@pytest.mark.parametrize(
    ("version", "supported"),
    [("1.0.3", False), ("1.1.0", True), ("1.1.1", True), ("2.0.0", True)],
)
def test_turbo_follows_the_installed_version(installed_faster_whisper, version, supported):
    installed_faster_whisper(version)

    assert is_faster_whisper_turbo_supported() is supported


def test_a_missing_install_reads_as_unsupported(installed_faster_whisper):
    installed_faster_whisper(None)

    assert is_faster_whisper_turbo_supported() is False


def test_an_unreadable_version_reads_as_unsupported(installed_faster_whisper):
    installed_faster_whisper("not-a-version")

    assert is_faster_whisper_turbo_supported() is False


def test_asking_does_not_import_faster_whisper(installed_faster_whisper):
    installed_faster_whisper("1.1.1")

    # A ``None`` entry makes any ``import`` of the package fail.
    with patch.dict(sys.modules, {"faster_whisper": None, "ctranslate2": None}):
        assert is_faster_whisper_turbo_supported() is True
