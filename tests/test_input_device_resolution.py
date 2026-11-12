"""`voice_device` names the same microphone for every part of Jarvis that records.

The listener and hold-to-dictate both read it through one resolver, so the
rule below is the whole contract: blank, ``default`` or ``system`` is the
system default input, anything ``int()`` reads is the PortAudio index of an
input-capable device, and any other text picks the first input-capable device
whose name contains it, ignoring case. Anything else is refused rather than
recorded from another input.
"""

import pytest

from jarvis.utils.audio_stream import InputDeviceNotFound, resolve_input_device

pytestmark = pytest.mark.unit


# A desk microphone listed once per Windows host API, next to an output device.
DESK = [
    {'name': 'Microsoft Sound Mapper - Input', 'max_input_channels': 2},
    {'name': 'Microphone (MSI GV60)', 'max_input_channels': 2},
    {'name': 'Speakers (USB2.0 Device)', 'max_input_channels': 0},
    {'name': 'Microphone (MSI GV60)', 'max_input_channels': 2},
    {'name': 'Microphone (MSI GV60)', 'max_input_channels': 2},
]


def _no_lookup():
    pytest.fail('the device list is not needed for the system default')


@pytest.mark.parametrize('value', [None, '', '   ', 'default', 'System', ' DEFAULT '])
def test_blank_or_default_means_the_system_input(value):
    assert resolve_input_device(value, _no_lookup) is None


@pytest.mark.parametrize('value, index', [('3', 3), (' 4 ', 4), ('0', 0), ('+1', 1)])
def test_a_number_is_a_device_index(value, index):
    assert resolve_input_device(value, lambda: DESK) == index


@pytest.mark.parametrize('value', ['2', '5', '-1'])
def test_an_index_that_names_no_input_is_refused(value):
    # Device 2 is output-only; 5 and -1 name no device at all.
    with pytest.raises(InputDeviceNotFound):
        resolve_input_device(value, lambda: DESK)


@pytest.mark.parametrize('value', ['MSI GV60', 'msi gv60', '  Gv60 '])
def test_a_name_picks_the_first_input_that_contains_it(value):
    assert resolve_input_device(value, lambda: DESK) == 1


def test_a_name_skips_output_only_devices():
    devices = [
        {'name': 'Headset Earphone (Jabra)', 'max_input_channels': 0},
        {'name': 'Headset Microphone (Jabra)', 'max_input_channels': 1},
    ]
    assert resolve_input_device('Jabra', lambda: devices) == 1


@pytest.mark.parametrize('value', ['Disconnected Headset', 'Speakers'])
def test_a_name_that_matches_no_input_is_refused(value):
    with pytest.raises(InputDeviceNotFound):
        resolve_input_device(value, lambda: DESK)


def test_the_refusal_does_not_repeat_the_name():
    # A device name can carry a person's, and an exception's text ends up in
    # logs and tracebacks.
    with pytest.raises(InputDeviceNotFound) as refused:
        resolve_input_device("Alice's AirPods", lambda: DESK)
    assert 'alice' not in str(refused.value).lower()
