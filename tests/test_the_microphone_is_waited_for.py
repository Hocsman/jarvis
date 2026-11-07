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
from jarvis.utils.audio_stream import open_input_stream, refresh_input_devices

pytestmark = pytest.mark.unit

DESK_MIC = "Microphone (Orbital Desk Mic)"
OTHER_MIC = "Microphone (Lantern Webcam)"
SPEAKERS = "Speakers (Harbour Monitor)"


def _input(name: str) -> dict:
    return {"name": name, "max_input_channels": 2, "default_samplerate": 48000.0}


def _output(name: str) -> dict:
    return {"name": name, "max_input_channels": 0, "default_samplerate": 48000.0}


class _Capture:
    """A capture stream on one device. While it runs and its device is
    connected, a thread of its own hands the callback a silent block."""

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
        return self._machine.connected(self.device) or not self._machine.stops_streams_on_removal

    def start(self) -> None:
        if self.closed:
            raise RuntimeError("Error starting stream: stream is closed")
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._deliver, daemon=True, name="test-portaudio-callback")
        self._thread.start()

    def _deliver(self) -> None:
        block = np.zeros((self._blocksize, self._channels), dtype=np.float32)
        while self._running.is_set():
            if self._machine.connected(self.device) and self._callback is not None:
                self._callback(block, self._blocksize, None, None)
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
    APIs report it.
    """

    def __init__(self, devices, *, stops_streams_on_removal: bool = False):
        self._connected = [dict(d) for d in devices]
        self._listed = [dict(d) for d in devices]
        self._initialized = 1
        self.stops_streams_on_removal = stops_streams_on_removal
        self.streams: list[_Capture] = []
        self.terminated_under_open_streams = 0
        self.default = SimpleNamespace(device=(None, None))

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
        self._initialized += 1
        self._listed = [dict(d) for d in self._connected]

    def query_devices(self, device=None, kind=None):
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


@pytest.fixture(autouse=True)
def no_stream_open(monkeypatch):
    """This test's own count of open streams: the suite's other tests leave
    stand-in streams behind in the process-wide one."""
    monkeypatch.setattr(audio_lock, "_streams", [])
    monkeypatch.setattr(audio_lock, "_unguarded_opens", 0)


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
    listed = []
    reading = threading.Thread(target=refresh_input_devices, args=(machine,))
    window = threading.Thread(target=lambda: listed.append(get_input_devices()))
    reading.start()
    try:
        assert initialising.wait(5)
        window.start()
        time.sleep(0.2)
        assert listed == [], "the window read the list while PortAudio was initialising"
    finally:
        finish.set()
        reading.join(5)
        if window.is_alive() or window.ident is not None:
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


# ── The listener waits for its microphone ───────────────────────────────


def _config(voice_device):
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
    cfg.hot_window_seconds = 3.0
    cfg.voice_collect_seconds = 2.0
    cfg.voice_max_collect_seconds = 60.0
    cfg.voice_debug = False
    cfg.tune_enabled = False
    cfg.llm_chat_model = ""
    return cfg


@contextlib.contextmanager
def _listening(monkeypatch, machine: Machine, voice_device, platform: str = "linux"):
    """The real listener's run(), on its own thread, until the block ends."""
    whisper = SimpleNamespace(
        model=SimpleNamespace(device="cpu"),
        transcribe=lambda *args, **kwargs: (iter(()), None),
    )
    monkeypatch.setattr(listener_module, "sd", machine)
    monkeypatch.setattr(listener_module, "sys", SimpleNamespace(platform=platform, executable=sys.executable))
    monkeypatch.setattr(listener_module, "FASTER_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(listener_module, "MLX_WHISPER_AVAILABLE", False)
    monkeypatch.setattr(listener_module, "WhisperModel", lambda *args, **kwargs: whisper)
    monkeypatch.setattr(listener_module, "_predownload_whisper_snapshot", lambda name: None)
    monkeypatch.setattr(listener_module, "MICROPHONE_RESCAN_SECONDS", 0.05)
    monkeypatch.setattr(listener_module, "CAPTURE_CHECK_SECONDS", 0.05)
    monkeypatch.setattr(listener_module, "CAPTURE_LOST_SECONDS", 1.5)

    voice = listener_module.VoiceListener(MagicMock(), _config(voice_device), None, MagicMock())
    runner = threading.Thread(target=voice.run, daemon=True, name="test-voice-listener")
    runner.start()
    try:
        yield voice
    finally:
        voice.stop()
        runner.join(timeout=5)
        if voice._replies is not None:
            voice._replies.join(timeout=5)
        voice.state_manager.stop()
        assert not runner.is_alive(), "the listener did not stop"
    assert machine.terminated_under_open_streams == 0


def _listening_now(voice, machine: Machine) -> bool:
    """Audio from an open stream is reaching the listener."""
    seen = voice._callback_count
    return any(not s.closed for s in machine.streams) and _until(
        lambda: voice._callback_count > seen, timeout=1.0)


@pytest.mark.parametrize("configured", ["Orbital Desk", "0", "5"])
def test_a_missing_microphone_never_records_another_one(monkeypatch, capsys, configured):
    # Device 0 is output-only and 5 names no device: neither is a microphone.
    machine = Machine([_output(SPEAKERS), _input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, configured):
        assert console.shows("Microphone not connected")
        time.sleep(0.3)  # several re-reads of the device list

    assert machine.opened() == []
    assert console.text.count("Microphone not connected") == 1


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

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Microphone not connected")
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone connected: listening")
        assert _listening_now(voice, machine)

    assert machine.opened() == [DESK_MIC]
    assert console.text.count("Microphone connected") == 1


@pytest.mark.parametrize("refuses", ["open", "start"])
def test_a_microphone_that_will_not_work_at_once_is_tried_again(monkeypatch, capsys, refuses):
    # A device just plugged in can be listed before it is ready.
    error = OSError("Unanticipated host error [PaErrorCode -9999]")
    refusals = [error]

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

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Microphone not connected")
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone connected: listening")
        assert _listening_now(voice, machine)

    assert set(machine.opened()) == {DESK_MIC}
    assert refusals == []


def test_a_stop_ends_the_wait(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Microphone not connected")
        stopped_at = time.monotonic()
        voice.stop()
        assert _until(lambda: voice._replies is None or not voice._replies.is_alive(), 2)
    assert time.monotonic() - stopped_at < 2.0
    assert machine.opened() == []


def test_the_list_is_not_read_afresh_under_another_open_stream(monkeypatch, capsys):
    # Her voice, the tune or a dictation holds a stream: PortAudio is left
    # alone until it is closed, and only then does she find the microphone.
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)
    other = _Speakers().module().OutputStream()
    audio_lock.track_stream(other)

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Microphone not connected")
        machine.plug(_input(DESK_MIC))
        time.sleep(0.3)
        assert machine.opened() == [], "the list was read afresh under an open stream"

        other.close()
        assert console.shows("Microphone connected: listening")
        assert _listening_now(voice, machine)

    assert machine.opened() == [DESK_MIC]


@pytest.mark.parametrize("stops_streams_on_removal", [False, True],
                         ids=["audio-stops", "stream-stops"])
def test_a_microphone_lost_while_running_is_listened_to_again_once_back(
        monkeypatch, capsys, stops_streams_on_removal):
    machine = Machine([_input(OTHER_MIC), _input(DESK_MIC)],
                      stops_streams_on_removal=stops_streams_on_removal)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Listening!")
        assert _listening_now(voice, machine)

        machine.unplug(DESK_MIC)
        assert console.shows("Microphone lost")
        assert _until(lambda: all(s.closed for s in machine.streams)), "the dead stream was kept open"
        time.sleep(0.3)
        assert machine.opened() == [DESK_MIC], "another input was opened in its place"

        machine.plug(_input(DESK_MIC))
        assert console.shows("Microphone back: listening")
        assert _listening_now(voice, machine)

    assert machine.opened() == [DESK_MIC, DESK_MIC]
    assert console.text.count("Microphone lost") == 1
    assert console.text.count("Microphone back") == 1
    assert console.text.count("Listening!") == 1


def test_a_short_stall_does_not_reopen_the_stream(monkeypatch, capsys):
    machine = Machine([_input(DESK_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Listening!")
        assert _listening_now(voice, machine)
        machine.unplug(DESK_MIC)
        time.sleep(0.5)  # well under the threshold
        machine.plug(_input(DESK_MIC))
        assert _listening_now(voice, machine)
        time.sleep(1.5)

    assert machine.opened() == [DESK_MIC]
    assert "Microphone lost" not in console.text


def test_a_lost_microphone_is_reported_once_while_it_stays_away(monkeypatch, capsys):
    machine = Machine([_input(DESK_MIC)], stops_streams_on_removal=True)
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk") as voice:
        assert console.shows("Listening!")
        assert _listening_now(voice, machine)
        machine.unplug(DESK_MIC)
        assert console.shows("Microphone lost")
        time.sleep(0.5)

    assert console.text.count("Microphone lost") == 1
    assert "Microphone back" not in console.text


def test_a_microphone_that_appears_is_checked_for_permission_first_on_windows(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    console = _Console(capsys)

    with _listening(monkeypatch, machine, "Orbital Desk", platform="win32") as voice:
        assert console.shows("Microphone not connected")
        assert "Checking microphone permission" not in console.text
        machine.plug(_input(DESK_MIC))

        assert console.shows("Microphone connected: listening")
        assert _listening_now(voice, machine)

    assert "Microphone permission OK" in console.text
    # The probe's short stream, then capture, both on the microphone named.
    assert machine.opened() == [DESK_MIC, DESK_MIC]


# ── Dictation looks again ───────────────────────────────────────────────


def _press(engine) -> None:
    before = thread_guard.snapshot()
    engine._start_recording()
    thread_guard.wait_for_workers(before)
    engine._stop_recording(discard=True)
    thread_guard.wait_for_workers(before)


def test_dictation_finds_a_microphone_connected_since_start_up(monkeypatch, capsys):
    machine = Machine([_input(OTHER_MIC)])
    monkeypatch.setattr(dictation_module, "sd", machine)
    engine = dictation_module.DictationEngine(
        whisper_model_ref=lambda: object(),
        whisper_backend_ref=lambda: "faster-whisper",
        mlx_repo_ref=lambda: None,
        voice_device="Orbital Desk",
    )
    with patch.object(dictation_module, "_play_beep"):
        _press(engine)
        assert machine.opened() == []
        assert "microphone not found" in capsys.readouterr().out.lower()

        machine.plug(_input(DESK_MIC))
        _press(engine)

    assert machine.opened() == [DESK_MIC]
    assert all(s.closed for s in machine.streams)
    assert machine.terminated_under_open_streams == 0
