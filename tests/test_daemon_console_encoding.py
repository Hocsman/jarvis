"""Behaviour tests for the daemon's Windows console encoding setup.

The daemon forces UTF-8 on the standard streams so emoji survive the Windows
console. That setup runs at import time, and the suite imports the daemon under
two module identities (``jarvis.daemon`` and ``src.jarvis.daemon``, see the note
at the top of ``tests/conftest.py``), so it runs more than once per process.

Running it twice has to leave the streams usable. Swapping in a fresh wrapper
does not: the previous wrapper loses its last reference, its finaliser closes
the buffer both wrappers share, and every later write raises ``ValueError``.
That takes the whole process down with it, pytest's own output capture
included, so the failure shows up as a collection error far from its cause.
"""

from __future__ import annotations

import gc
import io
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jarvis import daemon

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.unit
def test_console_setup_run_twice_leaves_the_stream_writable():
    """Repeating the setup keeps the same stream, and its buffer, alive."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding='ascii', errors='strict')

    daemon._force_utf8_stream(stream)
    daemon._force_utf8_stream(stream)
    gc.collect()

    assert not buffer.closed
    stream.write("📝")
    stream.flush()
    assert "📝".encode('utf-8') in buffer.getvalue()


@pytest.mark.unit
def test_console_setup_upgrades_the_encoding_to_utf8():
    """The point of the setup: emoji stop raising on a narrow encoding."""
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding='ascii', errors='strict')

    daemon._force_utf8_stream(stream)

    assert stream.encoding.lower().replace('-', '') == 'utf8'


@pytest.mark.unit
def test_console_setup_tolerates_a_stream_it_cannot_reconfigure():
    """A custom writer without ``reconfigure`` is left alone, not crashed on."""

    class PlainWriter:
        def write(self, text):
            return len(text)

    daemon._force_utf8_stream(PlainWriter())
    daemon._force_utf8_stream(None)


@pytest.mark.integration
def test_importing_the_daemon_under_both_module_paths_keeps_stdout_open():
    """The end-to-end shape of the bug, on the import paths the suite uses.

    ``sys.platform`` is forced so the Windows-only branch is exercised on any
    host: the defect is invisible elsewhere, and a test that only runs on
    Windows would guard nothing in CI.
    """
    script = textwrap.dedent(
        f"""
        import gc, io, sys
        sys.platform = 'win32'
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(ROOT / 'src')!r})

        buffer = io.BytesIO()
        sys.stdout = io.TextIOWrapper(buffer, encoding='utf-8')

        import jarvis.daemon
        import src.jarvis.daemon
        gc.collect()

        sys.__stderr__.write('CLOSED' if buffer.closed else 'OPEN')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.stderr.strip().endswith('OPEN'), (
        f"stdout was closed by the second import\n{result.stderr}"
    )
