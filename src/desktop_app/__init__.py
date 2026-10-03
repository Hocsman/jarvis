"""
Jarvis Desktop App - System Tray Application

A cross-platform system tray app for controlling the Jarvis voice assistant.
Supports Windows, Ubuntu (Linux), and macOS.
"""

from __future__ import annotations
import sys
import os

# Fix OpenBLAS threading crash in bundled apps
# Must be set before numpy is imported (via faster-whisper, etc.)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Re-export main for entry point
from desktop_app.app import main

__all__ = ['main']
