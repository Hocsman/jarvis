"""A microphone that is missing, at start-up or later, is waited for.

PortAudio reads its device list when it initialises and only then, so a
microphone connected after start-up stays invisible until PortAudio is
initialised again, and initialising again closes every stream the process
holds. These tests run the real listener, dictation engine, speech engines,
thinking tune and refresh helper against a stand-in for sounddevice with a
machine behind it: microphones are plugged and unplugged on the machine, and
the stand-in lists what was connected when it last initialised, as the real
library does.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import thread_guard
import jarvis.dictation.dictation_engine as dictation_module
import jarvis.listening.listener as listener_module
import jarvis.utils.audio_lock as audio_lock
from jarvis.output import tts as tts_module
from jarvis.output.tune_player import TunePlayer
from jarvis.state import JarvisState, get_jarvis_state
from jarvis.utils.audio_stream import (
    DeviceListUnavailable,
    InputDeviceNotFound,
    open_input_stream,
    refresh_input_devices,
    resolve_input_device,
)

pytestmark = pytest.mark.unit

DESK_MIC = "Microphone (Orbital Desk Mic)"
OTHER_MIC = "Microphone (Lantern Webcam)"
SPEAKERS = "Speakers (Harbour Monitor)"


def _input(name: str) -> dict:
    return {"name": name, "max_input_channels": 2, "default_samplerate": 48000.0}


def _output(name: str) -> dict:
    return {"name": name, "max_input_channels": 0, "default_samplerate": 48000.0}


# The name of a capture stream's delivery thread.
CAPTURE_THREAD = "test-portaudio-callback"


class _Capture:
    """A capture stream on one device. While it runs and its device is
    connected, a thread of its own hands the callback a block at the
    machine's sound level."""

    def __init__(self, machine: "Machine", device: str, callback, blocksize: int, channels: int):
        self._machine = machine
        self.device = device
        self._callback = callback
        self._blocksize = blocksize
        self._channels = channels
        self.closed = False
        self._running = threading.Event()
        self._thread = None

    @property
    def active(self) -> bool:
        if self.closed or not self._running.is_set():
            return False
        if self._machine.fail_when_asked:
            self._machine.fail_when_asked -= 1
            raise RuntimeError("Error querying stream state [PaErrorCode -9988]")
        return self._machine.connected(self.device) or not self._machine.stops_streams_on_removal

    def start(self) -> None:
        if self.closed:
            raise RuntimeError("Error starting stream: stream is closed")
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._deliver, daemon=True, name=CAPTURE_THREAD)
        self._thread.start()

    def _deliver(self) -> None:
        while self._running.is_set():
            if self._machine.connected(self.device) and self._callback is not None:
                block = np.full((self._blocksize, self._channels), self._machine.level, dtype=np.float32)
                self._callback(block, self._blocksize, None, None)
                self._machine.delivered += 1
            time.sleep(0.01)

    def stop(self) -> None:
        self._halt()

    def abort(self) -> None:
        self._halt()

    def close(self) -> None:
        self._halt()
        self.closed = True

    def _halt(self) -> None:
        self._running.clear()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)


class Machine:
    """sounddevice, with the machine it runs on behind it.

    ``plug`` and ``unplug`` change what is connected; ``query_devices`` answers
    with what was connected when PortAudio last initialised. A stream on a
    device that is unplugged stops receiving audio, and with
    ``stops_streams_on_removal`` it also stops being active, as some host
    APIs report it. ``initialisations`` counts the readings of the device
    list, ``delivered`` the blocks handed to capture callbacks, and
    ``level`` is how loud those blocks are.
    """

    def __init__(self, devices, *, stops_streams_on_removal: bool = False):
        self._connected = [dict(d) for d in devices]
        self._listed = [dict(d) for d in devices]
        self._initialized = 1
        self.stops_streams_on_removal = stops_streams_on_removal
        self.streams: list[_Capture] = []
        self.terminated_under_open_streams = 0
        self.default = SimpleNamespace(device=(None, None))
        self.initialisations = 0
        self.initialised_at: list[float] = []
        self.initialise_failures = 0
        self.delivered = 0
        self.level = 0.0
        self.fail_when_asked = 0

    # The machine.
    def plug(self, device: dict) -> None:
        self._connected.append(dict(device))

    def unplug(self, name: str) -> None:
        self._connected = [d for d in self._connected if d["name"] != name]

    def connected(self, name: str) -> bool:
        return any(d["name"] == name for d in self._connected)

    def opened(self) -> list[str]:
        return [s.device for s in self.streams]

    # sounddevice.
    def _terminate(self) -> None:
        if any(not s.closed for s in self.streams):
            self.terminated_under_open_streams += 1
        self._initialized -= 1

    def _initialize(self) -> None:
        if self.initialise_failures:
            self.initialise_failures -= 1
            raise RuntimeError("Error initializing PortAudio: Unanticipated host error [PaErrorCode -9999]")
        self._initialized += 1
        self.initialisations += 1
        self.initialised_at.append(time.monotonic())
        self._listed = [dict(d) for d in self._connected]

    def query_devices(self, device=None, kind=None):
        if self._initialized <= 0:
            raise RuntimeError("PortAudio not initialized [PaErrorCode -10000]")
        if device is not None:
            return dict(self._listed[device])
        if kind == "input":
            return next(dict(d) for d in self._listed if d["max_input_channels"] > 0)
        return [dict(d) for d in self._listed]

    def InputStream(self, samplerate, channels, dtype, blocksize, callback=None, device=None, **kwargs):
        if device is None:
            entry = next((d for d in self._listed if d["max_input_channels"] > 0), None)
        else:
            entry = self._listed[device]
        if entry is None or not self.connected(entry["name"]):
            raise OSError("Error opening InputStream: Device unavailable [PaErrorCode -9985]")
        stream = _Capture(self, entry["name"], callback, blocksize, channels)
        self.streams.append(stream)
        return stream


def _names(devices) -> list[str]:
    return [d["name"] for d in devices]


def _until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _rereads(machine: Machine, more: int) -> bool:
    """The device list is read afresh ``more`` more times."""
    target = machine.initialisations + more
    return _until(lambda: machine.initialisations >= target)


class _Console:
    """What the listener printed, read as it goes."""

    def __init__(self, capsys):
        self._capsys = capsys
        self._seen = ""

    @property
    def text(self) -> str:
        self._seen += self._capsys.readouterr().out
        return self._seen

    def shows(self, words: str, timeout: float = 5.0) -> bool:
        return _until(lambda: words in self.text, timeout)


# ── The device list is re-read only while no stream is open ─────────────


def test_a_microphone_connected_since_start_up_appears_in_the_list_read_afresh():
    machine = Machine([_input(OTHER_MIC)])
    machine.plug(_input(DESK_MIC))
    assert DESK_MIC not in _names(machine.query_devices())

    devices = refresh_input_devices(machine)

    assert DESK_MIC in _names(devices)
    assert DESK_MIC in _names(machine.query_devices())


def test_the_list_is_not_read_afresh_while_a_stream_is_open():
    machine = Machine([_input(OTHER_MIC)])
    stream, _, _ = open_input_stream(48000, 20, {}, sd_backend=machine)
    machine.plug(_input(DESK_MIC))

    assert refresh_input_devices(machine) is None
    assert DESK_MIC not in _names(machine.query_devices())

    with audio_lock.portaudio_lock:
        stream.close()
    assert DESK_MIC in _names(refresh_input_devices(machine))
    assert machine.terminated_under_open_streams == 0


def test_an_open_made_without_the_lock_holds_the_list_back_from_its_start():
    # The Windows permission probe opens its stream without the lock.
    entered, gate = threading.Event(), threading.Event()
    opened = []

    class SlowMachine(Machine):
        def InputStream(self, **kwargs):
            entered.set()
            gate.wait(5)
            return super().InputStream(**kwargs)

    machine = SlowMachine([_input(OTHER_MIC)])
    probe = threading.Thread(target=lambda: opened.append(
        open_input_stream(16000, 100, {}, serialise=False, sd_backend=machine)[0]))
    probe.start()
    try:
        assert entered.wait(5)
        assert refresh_input_devices(machine) is None, "re-read while a stream was being opened"
    finally:
        gate.set()
        probe.join(5)
    assert refresh_input_devices(machine) is None, "re-read under the probe's open stream"

    with audio_lock.portaudio_lock:
        opened[0].close()
    assert refresh_input_devices(machine) is not None
    assert machine.terminated_under_open_streams == 0


def test_a_stream_whose_state_cannot_be_read_counts_as_open():
    machine = Machine([_input(OTHER_MIC)])
    audio_lock.track_stream(object())

    assert refresh_input_devices(machine) is None
    assert machine._initialized == 1, "PortAudio was terminated all the same"


def test_portaudio_that_will_not_initialise_at_once_is_tried_once_more():
    machine = Machine([_input(OTHER_MIC)])
    machine.plug(_input(DESK_MIC))
    machine.initialise_failures = 1

    devices = refresh_input_devices(machine)

    assert DESK_MIC in _names(devices)
    assert DESK_MIC in _names(machine.query_devices()), "PortAudio was left down"


def test_portaudio_that_will_not_initialise_again_is_not_taken_for_a_held_back_reading():
    machine = Machine([_input(OTHER_MIC)])
    machine.plug(_input(DESK_MIC))
    machine.initialise_failures = 2

    with pytest.raises(DeviceListUnavailable):
        refresh_input_devices(machine)

    # PortAudio is down until a later reading brings it back.
    assert DESK_MIC in _names(refresh_input_devices(machine))


class _Speakers:
    """A sounddevice for the speech engines and the tune: each output stream
    plays until it is released, aborted or closed."""

    def __init__(self):
        self.streams = []
        self.release = threading.Event()

    def module(self):
        speakers = self

        class _Output:
            def __init__(self, **kwargs):
                self.closed = False
                self._aborted = False
                self.latency = 0.0
                speakers.streams.append(self)

            @property
            def active(self):
                return not (self.closed or self._aborted or speakers.release.is_set())

            def start(self):
                pass

            def abort(self):
                self._aborted = True

            def stop(self):
                self._aborted = True

            def close(self):
                self.closed = True

        class _Abort(Exception):
            pass

        class _Stop(Exception):
            pass

        return SimpleNamespace(OutputStream=_Output, CallbackAbort=_Abort, CallbackStop=_Stop)

    def playing(self) -> bool:
        return any(not s.closed for s in self.streams)


@pytest.mark.parametrize("engine", ["piper", "kokoro", "kyutai"])
def test_the_list_is_not_read_afresh_while_she_speaks(monkeypatch, tmp_path, engine):
    from test_a_stop_word_stops_the_whole_reply import _kokoro_rig, _kyutai_rig, _piper_rig

    rig = {"piper": lambda: _piper_rig(monkeypatch, tmp_path),
           "kokoro": lambda: _kokoro_rig(monkeypatch),
           "kyutai": lambda: _kyutai_rig(monkeypatch)}[engine]()
    rig.identify_audio = False
    speakers = _Speakers()
    monkeypatch.setitem(sys.modules, "sounddevice", speakers.module())
    machine = Machine([_input(OTHER_MIC)])
    rig.start_worker()
    try:
        rig.engine.speak("Une phrase à dire.")
        assert _until(speakers.playing), "she never started speaking"
        assert refresh_input_devices(machine) is None

        speakers.release.set()
        assert _until(lambda: not speakers.playing())
        assert refresh_input_devices(machine) is not None
    finally:
        speakers.release.set()
        rig.close()
    assert machine.terminated_under_open_streams == 0


class _WatchedLock:
    """``portaudio_lock`` as a caller takes it, saying when one is about to."""

    def __init__(self, lock):
        self._lock = lock
        self.wanted = threading.Event()

    def __enter__(self):
        self.wanted.set()
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


def test_the_settings_window_waits_for_a_reading_in_progress(monkeypatch):
    # In the bundled app the listener shares the window's process: the
    # window never reads the list while PortAudio is being initialised.
    from desktop_app.settings_window import get_input_devices

    initialising, finish = threading.Event(), threading.Event()

    class SlowMachine(Machine):
        def _initialize(self):
            initialising.set()
            finish.wait(5)
            super()._initialize()

    machine = SlowMachine([_input(OTHER_MIC)])
    machine.plug(_input(DESK_MIC))
    monkeypatch.setitem(sys.modules, "sounddevice", machine)
    watched = _WatchedLock(audio_lock.portaudio_lock)
    listed = []
    reading = threading.Thread(target=refresh_input_devices, args=(machine,))
    window = threading.Thread(target=lambda: listed.append(get_input_devices()))
    reading.start()
    try:
        assert initialising.wait(5)
        monkeypatch.setattr(audio_lock, "portaudio_lock", watched)
        window.start()
        assert watched.wanted.wait(5), "the window read the list without the lock"
        assert listed == [], "the window read the list while PortAudio was initialising"
    finally:
        finish.set()
        reading.join(5)
        if window.ident is not None:
            window.join(5)
    assert DESK_MIC in [value for value, _ in listed[0]]


def test_the_list_is_not_read_afresh_while_the_tune_plays(monkeypatch):
    speakers = _Speakers()
    monkeypatch.setitem(sys.modules, "sounddevice", speakers.module())
    machine = Machine([_input(OTHER_MIC)])
    tune = TunePlayer(enabled=True)
    tune.start_tune()
    try:
        assert _until(speakers.playing), "the tune never started"
        assert refresh_input_devices(machine) is None
    finally:
        tune.stop_tune()
    assert not speakers.playing()
    assert refresh_input_devices(machine) is not None


# ── An index keeps the microphone it named ──────────────────────────────


def test_an_index_keeps_the_microphone_it_named_after_the_list_is_read_afresh():
    machine = Machine([_output(SPEAKERS), _input(DESK_MIC), _input(OTHER_MIC)])
    assert resolve_input_device("1", machine.query_devices) == 1

    machine.unplug(DESK_MIC)
    refresh_input_devices(machine)
    assert _names(machine.query_devices())[1] == OTHER_MIC
    with pytest.raises(InputDeviceNotFound):
        resolve_input_device("1", machine.query_devices)

    machine.plug(_input(DESK_MIC))
    refresh_input_devices(machine)
    index = resolve_input_device("1", machine.query_devices)
    assert machine.query_devices()[index]["name"] == DESK_MIC


@pytest.mark.parametrize("configured", ["0", "5"])
def test_an_index_that_names_no_input_at_start_up_never_names_one(configured):
    # Device 0 is output-only and 5 names no device: neither is a microphone.
    machine = Machine([_output(SPEAKERS), _input(OTHER_MIC)])
    machine.plug(_input(DESK_MIC))
    machine.plug(_input("Microphone (Spare Headset)"))
    machine.plug(_input("Microphone (Second Spare)"))
    machine.plug(_input("Microphone (Third Spare)"))

    with pytest.raises(InputDeviceNotFound):
        resolve_input_device(configured, machine.query_devices)
    refresh_input_devices(machine)
    with pytest.raises(InputDeviceNotFound):
        resolve_input_device(configured, machine.query_devices)


# ── The listener waits for its microphone ───────────────────────────────


def _config(voice_device, **settings):
    cfg = MagicMock()
    cfg.voice_device = voice_device
    cfg.wake_word = "jarvis"
    cfg.whisper_model = "small"
    cfg.whisper_device = "cpu"
    cfg.whisper_compute_type = "int8"
    cfg.whisper_backend = "faster-whisper"
    cfg.whisper_vad = False
    cfg.low_power_mode = True
    cfg.sample_rate = 16000
    cfg.vad_enabled = False
    cfg.vad_frame_ms = 20
    cfg.echo_tolerance = 0.3
    cfg.echo_energy_threshold = 2.0
    cfg.hot_window_enabled = True
    cfg.hot_window_seconds = 3.0
    cfg.voice_collect_seconds = 2.0
    cfg.voice_max_collect_seconds = 60.0
    cfg.voice_debug = False
    cfg.tune_enabled = False
    cfg.llm_chat_model = ""
    for name, value in settings.items():
        setattr(cfg, name, value)
    return cfg


@contextlib.contextmanager
def _listening(monkeypatch, machine: Machine, voice_device, platform: str = "linux",
               heard: list | None = None, **settings):
    """The real listener's run(), on its own thread, until the block ends.
    What Whisper is asked to transcribe goes to ``heard``."""
    def transcribe(audio, *args, **kwargs):
        if heard is not None:
            heard.append(np.asarray(audio).copy())
        return iter(()), None

    whisper = SimpleNamespace(model=SimpleNamespace(device="cpu"), transcribe=transcribe)
    monkeypatch.setattr(listener_module, "sd", machine)
    monkeypatch.setattr(listener_module, "sys", SimpleNamespace(platform=platform, executable=sys.executable))
    monkeypatch.setattr(listener_module, "FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(listener_module, "MLX_WHISPER_AVAILABLE", False)
    monkeypatch.setattr(listener_module, "WhisperModel", lambda *args, **kwargs: whisper)
    monkeypatch.setattr(listener_module, "_predownload_whisper_snapshot", lambda name: None)
    monkeypatch.setattr(listener_module, "MICROPHONE_RESCAN_SECONDS", 0.05)
    monkeypatch.setattr(listener_module, "CAPTURE_CHECK_SECONDS", 0.05)
    monkeypatch.setattr(listener_module, "CAPTURE_LOST_SECONDS", 1.5)

    before = thread_guard.snapshot()
    voice = listener_module.VoiceListener(MagicMock(), _config(voice_device, **settings), None, MagicMock())
    runner = threading.Thread(target=voice.run, daemon=True, name="test-voice-listener")
    runner.start()
    try:
        yield voice
    finally:
        voice.stop()
        voice.state_manager.stop()
        thread_guard.wait_for_workers(before)
    assert machine.terminated_under_open_streams == 0


def _listening_now(machine: Machine) -> bool:
    """Audio from an open stream is reaching the listener."""
    seen = machine.delivered
    return any(not s.closed for s in machine.streams) and _until(
        lambda: machine.delivered > seen, timeout=1.0)


def test_a_missing_microphone_never_records_another_one(monkeypatch, capsys):
    machine = Machine([_output(SPEAKERS), _input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Microphone not connected")
        assert _rereads(machine, 3)

    assert machine.opened() == []
    assert console.text.count("Microphone not connected") == 1


@pytest.mark.parametrize("configured", ["0", "5"])
def test_an_index_that_names_no_input_is_refused_rather_than_waited_for(monkeypatch, capsys, configured):
    # Device 0 is output-only and 5 names no device: neither is a
    # microphone, and no microphone connected later is one either.
    machine = Machine([_output(SPEAKERS), _input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, configured):
        assert console.shows("Selected microphone not found. Choose an available input in Settings.")
        assert not console.shows("Microphone not connected", timeout=0.3)

    assert machine.opened() == []
    assert machine.initialisations == 0


def test_the_waiting_line_does_not_name_the_microphone(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Microphone not connected")

    waiting = [line for line in console.text.splitlines() if "not connected" in line]
    assert waiting and all("Orbital" not in line for line in waiting)


def test_a_microphone_missing_at_start_up_is_listened_to_once_connected(monkeypatch, capsys):
    machine = Machine([_output(SPEAKERS), _input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Microphone not connected")
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone connected: listening")
        assert _listening_now(machine)

    assert machine.opened() == [DESK_MIC]
    assert console.text.count("Microphone connected") == 1


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("refuses", ["open", "start"])
def test_a_microphone_that_will_not_work_at_once_is_tried_again(monkeypatch, capsys, refuses, platform):
    # A device just plugged in can be listed before it is ready. On Windows
    # the permission probe meets the refusals first.
    refusals = [OSError("Unanticipated host error [PaErrorCode -9999]") for _ in range(3)]

    class Sluggish(Machine):
        def InputStream(self, **kwargs):
            if refuses == "open" and refusals:
                raise refusals.pop()
            stream = super().InputStream(**kwargs)
            if refuses == "start" and refusals:
                failure = refusals.pop()

                def start():
                    raise failure
                stream.start = start
            return stream

    machine = Sluggish([_input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk", platform=platform):
        assert console.shows("Microphone not connected")
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone connected: listening")
        assert _listening_now(machine)

    assert set(machine.opened()) == {DESK_MIC}
    assert refusals == []
    assert console.text.count("-9999") == 1, "the refusal was not said once"
    assert console.text.count("Using audio device") == 1
    assert "BLOCKED BY WINDOWS" not in console.text


def test_a_microphone_gone_before_its_first_open_is_waited_for(monkeypatch, capsys):
    # Listed when the process started, unplugged while the models loaded.
    machine = Machine([_input(DESK_MIC)])
    machine.unplug(DESK_MIC)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Microphone lost")
        assert _rereads(machine, 2)
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone back: listening")
        assert _listening_now(machine)

    assert machine.opened() == [DESK_MIC]
    assert console.text.count("Listening!") == 1


def test_a_microphone_gone_before_its_permission_probe_is_waited_for_on_windows(monkeypatch, capsys):
    machine = Machine([_input(DESK_MIC)])
    machine.unplug(DESK_MIC)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk", platform="win32"):
        assert console.shows("Microphone lost")
        assert _rereads(machine, 2)
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone back: listening")
        assert _listening_now(machine)

    assert "Microphone permission OK" in console.text
    assert "BLOCKED BY WINDOWS" not in console.text


def test_a_probe_that_fails_from_start_up_on_is_said_to_fail_once(monkeypatch, capsys):
    # The failure that starts the wait counts as said for that wait.
    refusals = [OSError("Device unavailable [PaErrorCode -9985]") for _ in range(2)]

    class NotReady(Machine):
        def InputStream(self, **kwargs):
            if refusals:
                raise refusals.pop()
            return super().InputStream(**kwargs)

    machine = NotReady([_input(DESK_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk", platform="win32"):
        assert console.shows("Microphone back: listening")
        assert _listening_now(machine)

    assert refusals == []
    assert console.text.count("permission check failed") == 1
    assert "Microphone permission OK" in console.text
    assert "BLOCKED BY WINDOWS" not in console.text


def test_a_stop_ends_the_wait(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)
    before = thread_guard.snapshot()

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Microphone not connected")
        voice.stop()
        assert thread_guard.new_survivors(before, grace=2.0) == [], "the wait outlived the stop"
    assert machine.opened() == []


def test_the_list_is_not_read_afresh_under_another_open_stream(monkeypatch, capsys):
    # Her voice, the tune or a dictation holds a stream: PortAudio is left
    # alone until it is closed, and only then does she find the microphone.
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)
    other = _Speakers().module().OutputStream()
    audio_lock.track_stream(other)
    attempts = []
    read_afresh = listener_module.refresh_input_devices
    monkeypatch.setattr(listener_module, "refresh_input_devices",
                        lambda sd: attempts.append(sd) or read_afresh(sd))

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Microphone not connected")
        machine.plug(_input(DESK_MIC))
        tried = len(attempts)
        assert _until(lambda: len(attempts) >= tried + 3)
        assert machine.opened() == [], "the list was read afresh under an open stream"
        assert machine.initialisations == 0

        other.close()
        assert console.shows("Microphone connected: listening")
        assert _listening_now(machine)

    assert machine.opened() == [DESK_MIC]


def test_a_reading_held_back_for_long_is_said_once(monkeypatch, capsys):
    monkeypatch.setattr(listener_module, "HELD_BACK_NOTICE_SECONDS", 0.3)
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)
    other = _Speakers().module().OutputStream()
    audio_lock.track_stream(other)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("waits for other audio to finish")
        machine.plug(_input(DESK_MIC))
        other.close()
        assert console.shows("Microphone connected: listening")

    notices = [line for line in console.text.splitlines() if "other audio to finish" in line]
    assert len(notices) == 1
    assert "Orbital" not in notices[0] and "Desk" not in notices[0]


def test_portaudio_that_will_not_initialise_again_is_tried_on_the_next_reading(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    machine.plug(_input(DESK_MIC))
    machine.initialise_failures = 2
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Microphone connected: listening")
        assert _listening_now(machine)

    assert machine.opened() == [DESK_MIC]


@pytest.mark.parametrize("stops_streams_on_removal", [False, True],
                         ids=["audio-stops", "stream-stops"])
def test_a_microphone_lost_while_running_is_listened_to_again_once_back(
        monkeypatch, capsys, stops_streams_on_removal):
    machine = Machine([_input(OTHER_MIC), _input(DESK_MIC)],
                      stops_streams_on_removal=stops_streams_on_removal)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Listening!")
        assert _listening_now(machine)

        machine.unplug(DESK_MIC)
        assert console.shows("Microphone lost")
        assert _until(lambda: all(s.closed for s in machine.streams)), "the dead stream was kept open"
        assert _rereads(machine, 3)
        assert machine.opened() == [DESK_MIC], "another input was opened in its place"

        machine.plug(_input(DESK_MIC))
        assert console.shows("Microphone back: listening")
        assert _listening_now(machine)

    assert machine.opened() == [DESK_MIC, DESK_MIC]
    assert console.text.count("Microphone lost") == 1
    assert console.text.count("Microphone back") == 1
    assert console.text.count("Listening!") == 1


def test_an_index_never_moves_to_the_microphone_that_takes_its_place(monkeypatch, capsys):
    machine = Machine([_output(SPEAKERS), _input(DESK_MIC), _input(OTHER_MIC)],
                      stops_streams_on_removal=True)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "1"):
        assert console.shows("Listening!")
        assert _listening_now(machine)

        # Unplugged, the desk microphone leaves its place to the webcam.
        machine.unplug(DESK_MIC)
        assert console.shows("Microphone lost")
        assert _rereads(machine, 3)
        assert machine.opened() == [DESK_MIC], "the webcam was opened in its place"
        assert "Microphone back" not in console.text

        machine.plug(_input(DESK_MIC))
        assert console.shows("Microphone back: listening")
        assert _listening_now(machine)

    assert machine.opened() == [DESK_MIC, DESK_MIC]


def test_a_stream_that_fails_when_asked_is_reopened(monkeypatch, capsys):
    machine = Machine([_input(DESK_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Listening!")
        assert _listening_now(machine)

        machine.fail_when_asked = 1
        assert console.shows("Microphone lost")
        assert console.shows("Microphone back: listening")
        assert _listening_now(machine)

    assert machine.opened() == [DESK_MIC, DESK_MIC]
    assert machine.streams[0].closed


def test_the_stream_is_not_taken_for_dead_while_he_dictates(monkeypatch, capsys):
    # Dictation pauses the listener while it records from the same device.
    machine = Machine([_input(DESK_MIC)], stops_streams_on_removal=True)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Listening!")
        assert _listening_now(machine)

        voice._dictation_active = True
        machine.unplug(DESK_MIC)
        assert not console.shows("Microphone lost", timeout=0.5)

        machine.plug(_input(DESK_MIC))
        voice._dictation_active = False
        assert _listening_now(machine)
        assert not console.shows("Microphone lost", timeout=0.3)

    assert machine.opened() == [DESK_MIC]


def test_a_loss_drops_the_words_it_cut_off(monkeypatch, capsys):
    # What the dead stream had delivered (a half-heard utterance, blocks
    # still queued) is not joined to what he says once it is back.
    machine = Machine([_input(DESK_MIC)], stops_streams_on_removal=True)
    console = _Console(capsys)
    heard: list = []
    cut_off, said_after = 0.3, 0.6

    with _listening(monkeypatch, machine, "Orbital Desk", heard=heard,
                    voice_min_energy=0.05, endpoint_silence_ms=100, max_utterance_ms=12000,
                    vad_pre_roll_ms=20, whisper_min_audio_duration=0.0,
                    tts_max_utterance_ms=3000):
        assert console.shows("Listening!")
        assert _listening_now(machine)
        warm_up = len(heard)

        machine.level = cut_off
        delivered = machine.delivered
        assert _until(lambda: machine.delivered >= delivered + 10)
        machine.unplug(DESK_MIC)
        assert console.shows("Microphone lost")

        machine.level = said_after
        machine.plug(_input(DESK_MIC))
        assert console.shows("Microphone back: listening")
        delivered = machine.delivered
        assert _until(lambda: machine.delivered >= delivered + 10)
        machine.level = 0.0
        assert _until(lambda: len(heard) > warm_up), "his words after the loss were never transcribed"

    utterances = heard[warm_up:]
    assert not any(np.isclose(utterance, cut_off).any() for utterance in utterances)
    assert any(np.isclose(utterance, said_after).any() for utterance in utterances)


def test_a_lost_microphone_is_reported_once_while_it_stays_away(monkeypatch, capsys):
    machine = Machine([_input(DESK_MIC)], stops_streams_on_removal=True)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk"):
        assert console.shows("Listening!")
        assert _listening_now(machine)
        machine.unplug(DESK_MIC)
        assert console.shows("Microphone lost")
        assert _rereads(machine, 3)

    assert console.text.count("Microphone lost") == 1
    assert "Microphone back" not in console.text


def test_a_microphone_that_appears_is_checked_for_permission_first_on_windows(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk", platform="win32"):
        assert console.shows("Microphone not connected")
        assert "Checking microphone permission" not in console.text
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone connected: listening")
        assert _listening_now(machine)

    assert "Microphone permission OK" in console.text
    # The probe's short stream, then capture, both on the microphone named.
    assert machine.opened() == [DESK_MIC, DESK_MIC]


def test_the_permission_probe_waits_for_her_voice_to_end(monkeypatch, capsys):
    # The probe opens without the lock, so it never runs while another
    # stream is open in the process.
    console = _Console(capsys)
    other = _Speakers().module().OutputStream()
    probing_under = []

    class Watched(Machine):
        def _initialize(self):
            super()._initialize()
            if self.connected(DESK_MIC) and not other.closed:
                # She starts speaking right after the reading that finds it.
                audio_lock.track_stream(other)

        def InputStream(self, **kwargs):
            probing_under.append(other.closed)
            return super().InputStream(**kwargs)

    machine = Watched([_input(OTHER_MIC)])

    with _listening(monkeypatch, machine, "Orbital Desk", platform="win32"):
        assert console.shows("Microphone not connected")
        machine.plug(_input(DESK_MIC))
        assert _rereads(machine, 1)
        assert not console.shows("Checking microphone permission", timeout=0.3)

        other.close()
        assert console.shows("Microphone connected: listening")

    assert probing_under and all(probing_under), "the probe opened while she spoke"


def test_no_hot_window_opens_while_the_microphone_is_waited_for(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Microphone not connected")
        get_jarvis_state().set_state(JarvisState.SPEAKING)

        # Her speech ends: what the speech engine's completion callback calls.
        voice.activate_hot_window()

        assert not console.shows("Listening for follow-up", timeout=0.8)
        assert get_jarvis_state().state == JarvisState.IDLE


def test_the_list_is_read_afresh_less_often_after_a_minute_unless_dictation_misses(monkeypatch, capsys):
    monkeypatch.setattr(listener_module, "MICROPHONE_RESCAN_FAST_FOR_SECONDS", 0.4)
    monkeypatch.setattr(listener_module, "MICROPHONE_RESCAN_SLOW_SECONDS", 1.0)
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)

    def gaps():
        times = machine.initialised_at
        return [later - earlier for earlier, later in zip(times, times[1:])]

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Microphone not connected")
        assert _until(lambda: sum(gap >= 0.95 for gap in gaps()) >= 2, timeout=8)
        backed_off = gaps()
        first_slow = next(i for i, gap in enumerate(backed_off) if gap >= 0.95)
        assert first_slow > 0, "the first readings were not frequent"
        assert all(gap >= 0.95 for gap in backed_off[first_slow:]), "the back-off did not hold"

        # A dictation press that misses the microphone.
        readings = machine.initialisations
        voice.microphone_missed()
        assert _until(lambda: machine.initialisations >= readings + 3, timeout=2)
        assert all(gap < 0.95 for gap in gaps()[readings:readings + 2])


# ── Dictation looks again ───────────────────────────────────────────────


def _press(engine) -> None:
    before = thread_guard.snapshot()
    engine._start_recording()
    _wait_for_the_press(before)
    engine._stop_recording(discard=True)
    thread_guard.wait_for_workers(before)


def _wait_for_the_press(before, timeout: float = 5.0) -> None:
    """Let the worker a press starts finish, or fail naming what is left.

    The press's microphone delivers on its own thread from its start until
    the release closes it, so that thread is not waited for here; the
    release's `thread_guard.wait_for_workers` then checks it has ended.
    """
    deadline = time.monotonic() + timeout
    while True:
        workers = [
            thread for thread in threading.enumerate()
            if thread not in before
            and thread.name != CAPTURE_THREAD
            and thread is not threading.current_thread()
        ]
        if not workers:
            return
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{len(workers)} worker thread(s) still running after {timeout}s: "
                + ", ".join(thread_guard.describe(thread) for thread in workers),
                pytrace=False,
            )
        time.sleep(0.01)


def _dictation(machine: Machine, voice_device: str, **callbacks):
    return dictation_module.DictationEngine(
        whisper_model_ref=lambda: object(),
        whisper_backend_ref=lambda: "faster-whisper",
        mlx_repo_ref=lambda: None,
        voice_device=voice_device,
        **callbacks,
    )


def test_dictation_finds_a_microphone_connected_since_start_up(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    monkeypatch.setattr(dictation_module, "sd", machine)
    missed = []
    engine = _dictation(machine, "Orbital Desk", on_microphone_missing=lambda: missed.append(1))
    with patch.object(dictation_module, "_play_beep"):
        _press(engine)
        assert machine.opened() == []
        assert "microphone not found" in capsys.readouterr().out.lower()
        assert len(missed) == 1, "the listener was not told the press missed"

        machine.plug(_input(DESK_MIC))
        _press(engine)

    assert machine.opened() == [DESK_MIC]
    assert all(s.closed for s in machine.streams)
    assert machine.terminated_under_open_streams == 0
    assert len(missed) == 1


def test_dictation_keeps_the_list_it_has_while_a_stream_is_open(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    monkeypatch.setattr(dictation_module, "sd", machine)
    engine = _dictation(machine, "Orbital Desk")
    other = _Speakers().module().OutputStream()
    audio_lock.track_stream(other)
    machine.plug(_input(DESK_MIC))
    with patch.object(dictation_module, "_play_beep"):
        _press(engine)
        assert machine.opened() == []
        assert "microphone not found" in capsys.readouterr().out.lower()
        assert machine.initialisations == 0, "PortAudio was read afresh under an open stream"

        other.close()
        _press(engine)

    assert machine.opened() == [DESK_MIC]
    assert machine.terminated_under_open_streams == 0


def test_dictation_keeps_an_index_on_the_microphone_it_named(monkeypatch, capsys):
    machine = Machine([_output(SPEAKERS), _input(DESK_MIC), _input(OTHER_MIC)])
    monkeypatch.setattr(dictation_module, "sd", machine)
    engine = _dictation(machine, "1")
    with patch.object(dictation_module, "_play_beep"):
        _press(engine)
        assert machine.opened() == [DESK_MIC]

        # Unplugged, and the list read afresh: the webcam is now device 1.
        machine.unplug(DESK_MIC)
        refresh_input_devices(machine)
        _press(engine)

    assert machine.opened() == [DESK_MIC], "dictation recorded from the webcam"
    assert "microphone not found" in capsys.readouterr().out.lower()
