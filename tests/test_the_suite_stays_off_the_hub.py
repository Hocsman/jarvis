"""Running the suite is not permission to download models.

The listener fetches its Whisper snapshot before it builds the model, and
most listener tests stub the model without stubbing the fetch. Left alone,
that fetch asks the Hugging Face Hub for several gigabytes and stores them in
the developer's own model cache, on every run of the pre-push hook. The guard
in ``conftest.py`` stands between every test and the Hub; these tests hold it
to that.

The tripwire refuses every name lookup and connection to a host other than
this machine while it is armed, and records every open, listing and write
that lands in a Hub repository folder (``models--<org>--<name>``). The session
sandbox is the only place such a folder may be touched.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.test_voice_listener import _create_mock_config


_NETWORK_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.connect"}
_FILE_EVENTS = {
    "open", "os.listdir", "os.scandir", "os.mkdir", "os.rename", "os.remove",
    "os.rmdir", "os.symlink", "os.link", "os.utime", "shutil.rmtree",
    "shutil.copyfile", "shutil.move",
}
_LOOPBACK = {None, "", b"", "localhost", b"localhost", "::1", b"::1"}

_armed: list = []
_hook_installed = False


class _Tripwire:
    def __init__(self):
        self.network: list = []
        self.repo_folders: list = []


def _is_loopback(host) -> bool:
    if host in _LOOPBACK:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return isinstance(host, str) and host.startswith("127.")


def _audit(event, args):
    if not _armed:
        return
    wire = _armed[0]
    if event in _NETWORK_EVENTS:
        host = args[1] if event == "socket.connect" else args[0]
        if isinstance(host, tuple):
            host = host[0] if host else None
        if not _is_loopback(host):
            wire.network.append((event, host))
            raise ConnectionRefusedError(f"the network is closed to this test ({host!r})")
    elif event in _FILE_EVENTS:
        for arg in args:
            if isinstance(arg, (str, bytes, os.PathLike)):
                path = os.fsdecode(arg)
                if "models--" in path:
                    wire.repo_folders.append((event, path))


@pytest.fixture
def tripwire():
    global _hook_installed
    if not _hook_installed:
        # An audit hook cannot be removed, so it is installed once and stays
        # inert while nothing is armed.
        sys.addaudithook(_audit)
        _hook_installed = True
    wire = _Tripwire()
    _armed.append(wire)
    try:
        yield wire
    finally:
        _armed.clear()


def _outside(paths, sandbox: Path):
    root = sandbox.resolve()
    return [(event, p) for event, p in paths if not Path(p).resolve().is_relative_to(root)]


def test_loading_the_listener_whisper_model_stays_off_the_hub(tripwire, tmp_path_factory):
    """The model load most listener tests perform, with only WhisperModel
    stubbed, reaches neither the Hub nor any model cache outside the sandbox."""
    with patch("jarvis.listening.listener.sys") as mock_sys:
        mock_sys.platform = "linux"
        with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True):
            with patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False):
                with patch("jarvis.listening.listener.WhisperModel", return_value=MagicMock()):
                    with patch("jarvis.listening.listener.sd") as mock_sd:
                        mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
                        mock_sd.InputStream.side_effect = Exception("Stop test here")

                        from jarvis.listening.listener import VoiceListener
                        listener = VoiceListener(
                            MagicMock(), _create_mock_config(whisper_model="small"),
                            MagicMock(), MagicMock(),
                        )
                        listener.run()

    assert listener.model is not None, "the load under test did not run"
    assert tripwire.network == [], f"the load reached for the network: {tripwire.network}"
    assert _outside(tripwire.repo_folders, tmp_path_factory.getbasetemp()) == [], (
        "the load touched a model cache outside the sandbox"
    )


@pytest.fixture(scope="module")
def _hub_session_built_before_the_test():
    """huggingface_hub keeps one HTTP session per thread and decides its
    offline mode when it builds it, so a session left over from earlier in
    the run is the case an offline switch can miss."""
    pytest.importorskip("huggingface_hub.utils").get_session()


@pytest.mark.unit
def test_a_hub_request_is_refused_before_it_leaves_the_machine(
    _hub_session_built_before_the_test, tripwire,
):
    """Any code path that asks the Hub directly, past the listener's own
    fetch, is refused offline, even on a thread that talked to the Hub
    before the test began."""
    import huggingface_hub

    with pytest.raises(Exception):
        huggingface_hub.HfApi().model_info("jarvis-tests/never-published")

    assert tripwire.network == [], f"the request reached for the network: {tripwire.network}"


@pytest.mark.unit
def test_the_hub_cache_a_test_sees_is_the_sandbox(tmp_path_factory):
    """In this process and in any process a test spawns."""
    pytest.importorskip("huggingface_hub")
    import huggingface_hub.constants

    sandbox = tmp_path_factory.getbasetemp().resolve()
    in_process = Path(huggingface_hub.constants.HF_HUB_CACHE).resolve()
    assert in_process.is_relative_to(sandbox), f"this process caches models in {in_process}"
    inherited = os.environ.get("HF_HUB_CACHE")
    assert inherited and Path(inherited).resolve().is_relative_to(sandbox), (
        f"a spawned process would cache models in {inherited or 'its default cache'}"
    )
