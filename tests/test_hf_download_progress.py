"""Behavioural tests for download_snapshot_with_progress.

No network: huggingface_hub.snapshot_download and HfApi are faked, and the
blob growth of a real download is simulated by writing into a fake cache.
"""

import re
import threading
import time
from types import SimpleNamespace

import pytest


PATTERNS = ["config.json", "model.bin", "tokenizer.json"]
REPO = "Systran/faster-whisper-medium.en"


def _make_fake_snapshot(blobs_dir, steps_mb, step_delay=0.1):
    """A fake snapshot_download that grows a blob file like a real download."""
    def fake(repo_id, allow_patterns=None, **kwargs):
        target = blobs_dir / "abc123.incomplete"
        target.parent.mkdir(parents=True, exist_ok=True)
        for mb in steps_mb:
            target.write_bytes(b"\0" * (mb * 1024 * 1024))
            time.sleep(step_delay)
        return str(blobs_dir.parent)
    return fake


def _fake_model_info(total_mb, include_non_matching=False):
    siblings = [
        SimpleNamespace(rfilename="model.bin", size=total_mb * 1024 * 1024),
        SimpleNamespace(rfilename="config.json", size=1024),
    ]
    if include_non_matching:
        siblings.append(SimpleNamespace(rfilename="README.md", size=5 * 1024 * 1024))
    return SimpleNamespace(siblings=siblings)


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Point the helper at a fake HF cache with a controllable snapshot."""
    import huggingface_hub
    import huggingface_hub.constants

    cache = tmp_path / "hub"
    blobs = cache / "models--Systran--faster-whisper-medium.en" / "blobs"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))

    class FakeHfApi:
        def __init__(self):
            self.info = _fake_model_info(1530)

        def model_info(self, repo_id, files_metadata=False):
            return self.info

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return SimpleNamespace(cache=cache, blobs=blobs, monkeypatch=monkeypatch, hf=huggingface_hub)


def _progress_lines(capsys):
    out = capsys.readouterr()
    return [line for line in (out.out + out.err).splitlines() if "Downloading" in line or "downloading" in line]


@pytest.mark.unit
def test_progress_lines_show_done_total_and_rate(monkeypatch, fake_hub, capsys):
    from jarvis.utils.hf_download import download_snapshot_with_progress

    fake_hub.monkeypatch.setattr(
        fake_hub.hf, "snapshot_download",
        _make_fake_snapshot(fake_hub.blobs, steps_mb=[400, 900, 1530], step_delay=0.2),
    )

    download_snapshot_with_progress(REPO, PATTERNS, "Whisper 'medium.en'",
                                    poll_interval_sec=0.05)

    lines = _progress_lines(capsys)
    assert lines, "expected at least one progress line during the download"
    assert any(re.search(r"\d+/1530 MB · [\d.]+ MB/s", line) for line in lines), lines
    assert any("Whisper 'medium.en'" in line for line in lines)


@pytest.mark.unit
def test_progress_without_total_size_falls_back_to_so_far(monkeypatch, fake_hub, capsys):
    from jarvis.utils.hf_download import download_snapshot_with_progress

    class FailingApi:
        def model_info(self, repo_id, files_metadata=False):
            raise ConnectionError("offline")

    fake_hub.monkeypatch.setattr(fake_hub.hf, "HfApi", FailingApi)
    fake_hub.monkeypatch.setattr(
        fake_hub.hf, "snapshot_download",
        _make_fake_snapshot(fake_hub.blobs, steps_mb=[400, 900], step_delay=0.2),
    )

    download_snapshot_with_progress(REPO, PATTERNS, "Whisper 'medium.en'",
                                    poll_interval_sec=0.05)

    lines = _progress_lines(capsys)
    assert lines, "expected progress lines even without a known total"
    assert any(re.search(r"\d+ MB so far · [\d.]+ MB/s", line) for line in lines), lines


@pytest.mark.unit
def test_warm_cache_prints_no_progress(monkeypatch, fake_hub, capsys):
    from jarvis.utils.hf_download import download_snapshot_with_progress

    def instant_snapshot(repo_id, allow_patterns=None, **kwargs):
        return str(fake_hub.blobs.parent)

    fake_hub.monkeypatch.setattr(fake_hub.hf, "snapshot_download", instant_snapshot)

    download_snapshot_with_progress(REPO, PATTERNS, "Whisper 'medium.en'",
                                    poll_interval_sec=0.05)

    assert _progress_lines(capsys) == []


@pytest.mark.unit
def test_snapshot_exception_propagates_to_caller(monkeypatch, fake_hub):
    from jarvis.utils.hf_download import download_snapshot_with_progress

    def boom(repo_id, allow_patterns=None, **kwargs):
        raise RuntimeError("429 too many requests")

    fake_hub.monkeypatch.setattr(fake_hub.hf, "snapshot_download", boom)

    with pytest.raises(RuntimeError, match="429"):
        download_snapshot_with_progress(REPO, PATTERNS, "Whisper 'medium.en'",
                                        poll_interval_sec=0.05)


@pytest.mark.unit
def test_stalled_download_reports_it_is_still_alive(monkeypatch, fake_hub, capsys):
    from jarvis.utils.hf_download import download_snapshot_with_progress

    def stalled_snapshot(repo_id, allow_patterns=None, **kwargs):
        fake_hub.blobs.mkdir(parents=True, exist_ok=True)
        (fake_hub.blobs / "abc.incomplete").write_bytes(b"\0" * (10 * 1024 * 1024))
        time.sleep(1.0)  # no growth after the initial write
        return str(fake_hub.blobs.parent)

    fake_hub.monkeypatch.setattr(fake_hub.hf, "snapshot_download", stalled_snapshot)

    download_snapshot_with_progress(REPO, PATTERNS, "Whisper 'medium.en'",
                                    poll_interval_sec=0.05, stall_after_sec=0.2)

    lines = _progress_lines(capsys)
    assert any("still downloading" in line.lower() for line in lines), lines


@pytest.mark.unit
def test_total_size_ignores_files_outside_allow_patterns(monkeypatch, fake_hub, capsys):
    from jarvis.utils.hf_download import download_snapshot_with_progress

    fake_hub.monkeypatch.setattr(
        fake_hub.hf, "snapshot_download",
        _make_fake_snapshot(fake_hub.blobs, steps_mb=[1530], step_delay=0.3),
    )
    # model_info carries a 5 MB README.md which does not match the patterns
    import huggingface_hub

    class PatchedApi:
        def model_info(self, repo_id, files_metadata=False):
            return _fake_model_info(1530, include_non_matching=True)

    fake_hub.monkeypatch.setattr(huggingface_hub, "HfApi", PatchedApi)

    download_snapshot_with_progress(REPO, PATTERNS, "Whisper 'medium.en'",
                                    poll_interval_sec=0.05)

    lines = _progress_lines(capsys)
    assert lines
    # Total must be 1530 MB (model.bin) + ~0 MB (config.json), NOT +5 MB of README
    assert all("/1535" not in line for line in lines)
    assert any("/1530 MB" in line for line in lines)


@pytest.mark.unit
def test_returns_snapshot_path(monkeypatch, fake_hub):
    from jarvis.utils.hf_download import download_snapshot_with_progress

    def instant_snapshot(repo_id, allow_patterns=None, **kwargs):
        return "/snap/shot/path"

    fake_hub.monkeypatch.setattr(fake_hub.hf, "snapshot_download", instant_snapshot)

    result = download_snapshot_with_progress(REPO, PATTERNS, "Whisper 'medium.en'",
                                             poll_interval_sec=0.05)
    assert result == "/snap/shot/path"


@pytest.mark.unit
def test_progress_lines_survive_a_console_that_cannot_print_unicode(monkeypatch, fake_hub):
    """Windows consoles without the UTF-8 forcing raise UnicodeEncodeError on
    the emoji; the download must degrade to ASCII lines, never crash."""
    import jarvis.utils.hf_download as hf_download

    printed = []

    def cp1252_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if any(ord(c) > 127 for c in text):
            raise UnicodeEncodeError("charmap", text, 0, 1, "character maps to <undefined>")
        printed.append(text)

    monkeypatch.setattr(hf_download, "print", cp1252_print, raising=False)
    fake_hub.monkeypatch.setattr(
        fake_hub.hf, "snapshot_download",
        _make_fake_snapshot(fake_hub.blobs, steps_mb=[400, 900], step_delay=0.2),
    )

    result = hf_download.download_snapshot_with_progress(
        REPO, PATTERNS, "Whisper 'medium.en'", poll_interval_sec=0.05,
    )

    assert result  # no exception escaped
    assert any("Downloading" in line and "MB/s" in line for line in printed), printed


@pytest.mark.unit
def test_symlink_privilege_error_retries_with_copies(monkeypatch, fake_hub, capsys):
    """WinError 1314 (no symlink privilege) must not kill a completed download:
    force huggingface_hub's copy fallback and retry once. Upstream issue #636."""
    import jarvis.utils.hf_download as hf_download
    import huggingface_hub.file_download as fd

    # The fix writes into huggingface_hub's process-global probe cache;
    # snapshot and restore it so the rest of the session is unaffected.
    probe_cache = dict(fd._are_symlinks_supported_in_dir)
    monkeypatch.setattr(fd, "_are_symlinks_supported_in_dir", probe_cache)

    attempts = []

    def flaky_snapshot(repo_id, allow_patterns=None, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            err = OSError("A required privilege is not held by the client")
            err.winerror = 1314
            raise err
        return "/snap/shot/path"

    fake_hub.monkeypatch.setattr(fake_hub.hf, "snapshot_download", flaky_snapshot)

    result = hf_download.download_snapshot_with_progress(
        REPO, PATTERNS, "Whisper 'medium.en'", poll_interval_sec=0.05,
    )

    assert result == "/snap/shot/path"
    assert len(attempts) == 2
    # The probe cache is forced to "unsupported" so the retry copies files
    assert fd._are_symlinks_supported_in_dir
    assert all(v is False for v in fd._are_symlinks_supported_in_dir.values())


@pytest.mark.unit
def test_other_oserrors_do_not_retry(monkeypatch, fake_hub):
    import jarvis.utils.hf_download as hf_download

    def disk_full(repo_id, allow_patterns=None, **kwargs):
        err = OSError("No space left on device")
        err.winerror = 112
        raise err

    fake_hub.monkeypatch.setattr(fake_hub.hf, "snapshot_download", disk_full)

    with pytest.raises(OSError, match="No space"):
        hf_download.download_snapshot_with_progress(
            REPO, PATTERNS, "Whisper 'medium.en'", poll_interval_sec=0.05,
        )
