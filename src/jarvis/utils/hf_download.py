"""Visible progress for Hugging Face snapshot downloads.

faster-whisper silences huggingface_hub's progress bars (``disabled_tqdm``)
and the remaining bars auto-disable when stderr is not a TTY, which is always
the case when the desktop app pipes the daemon's output. The result: a 1.5 GB
model download with zero feedback, which users read as a crash.

This helper runs ``snapshot_download`` on a worker thread and watches the
repo's ``blobs`` directory grow, printing a throttled progress line. Resume
on failure is huggingface_hub's own ``.incomplete`` + ``Range`` mechanism, so
nothing here re-implements it.
"""

import fnmatch
import threading
import time
from pathlib import Path
from typing import List, Optional

import huggingface_hub
import huggingface_hub.constants

from ..debug import debug_log

_MB = 1024 * 1024


def _emit(line: str) -> None:
    """Print a progress line, degrading to ASCII on consoles that cannot
    encode the emoji (the daemon forces UTF-8, but this helper must never
    crash a download on a misconfigured console)."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def _blobs_dir(repo_id: str) -> Path:
    return (Path(huggingface_hub.constants.HF_HUB_CACHE)
            / f"models--{repo_id.replace('/', '--')}" / "blobs")


def _blobs_size_bytes(repo_id: str) -> int:
    blobs = _blobs_dir(repo_id)
    if not blobs.is_dir():
        return 0
    total = 0
    for entry in blobs.iterdir():
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            pass
    return total


def _expected_total_bytes(repo_id: str, allow_patterns: Optional[List[str]]) -> Optional[int]:
    """Best-effort total size of the files snapshot_download will fetch."""
    try:
        info = huggingface_hub.HfApi().model_info(repo_id, files_metadata=True)
        total = 0
        for sibling in info.siblings or []:
            if allow_patterns is None or any(
                fnmatch.fnmatch(sibling.rfilename, pat) for pat in allow_patterns
            ):
                total += sibling.size or 0
        return total or None
    except Exception as e:
        debug_log(f"model_info unavailable for {repo_id}, progress without total: {e}", "voice")
        return None


def download_snapshot_with_progress(
    repo_id: str,
    allow_patterns: Optional[List[str]],
    description: str,
    *,
    poll_interval_sec: float = 5.0,
    stall_after_sec: float = 30.0,
) -> str:
    """Download an HF snapshot with throttled progress lines on stdout.

    Prints nothing when the cache is already warm (no blob bytes move).
    Returns the snapshot path; re-raises whatever snapshot_download raised.
    """
    total_bytes = _expected_total_bytes(repo_id, allow_patterns)

    result: dict = {}

    def _snapshot() -> str:
        try:
            return huggingface_hub.snapshot_download(
                repo_id, allow_patterns=allow_patterns,
            )
        except OSError as e:
            # WinError 1314 (privilege not held) can escape huggingface_hub's
            # symlink support probe on Windows and kills a fully downloaded
            # snapshot at the pointer-creation step. Force the copy fallback
            # and retry once; any other OSError propagates untouched.
            if getattr(e, "winerror", None) != 1314:
                raise
            import huggingface_hub.file_download as fd
            repo_cache = Path(huggingface_hub.constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
            fd._are_symlinks_supported_in_dir[str(repo_cache.expanduser().resolve())] = False
            for key in list(fd._are_symlinks_supported_in_dir):
                fd._are_symlinks_supported_in_dir[key] = False
            _emit("  ⚠️  Windows denied symlink creation; storing plain file copies instead")
            debug_log(f"symlink creation failed (WinError 1314), retrying {repo_id} with copies", "voice")
            return huggingface_hub.snapshot_download(
                repo_id, allow_patterns=allow_patterns,
            )

    def _run() -> None:
        try:
            result["path"] = _snapshot()
        except BaseException as e:  # re-raised on the caller thread below
            result["error"] = e

    worker = threading.Thread(target=_run, daemon=True, name="hf-snapshot-download")
    debug_log(f"Downloading {description} from {repo_id} into the HF cache", "voice")
    worker.start()

    last_bytes = _blobs_size_bytes(repo_id)
    last_moved_at = time.monotonic()
    last_print_at = last_moved_at
    stall_announced = False

    while worker.is_alive():
        worker.join(timeout=poll_interval_sec)
        now = time.monotonic()
        current = _blobs_size_bytes(repo_id)
        if current > last_bytes:
            rate_mb_s = (current - last_bytes) / _MB / max(now - last_print_at, 0.001)
            done_mb = current / _MB
            if total_bytes:
                total_mb = total_bytes / _MB
                _emit(f"  ⬇️ Downloading {description} · {min(done_mb, total_mb):.0f}/{total_mb:.0f} MB "
                      f"· {rate_mb_s:.1f} MB/s")
            else:
                _emit(f"  ⬇️ Downloading {description} · {done_mb:.0f} MB so far "
                      f"· {rate_mb_s:.1f} MB/s")
            last_bytes = current
            last_moved_at = now
            last_print_at = now
            stall_announced = False
        elif not stall_announced and now - last_moved_at >= stall_after_sec and current > 0:
            _emit(f"  ⬇️ Still downloading {description} (no data for "
                  f"{int(now - last_moved_at)}s)...")
            stall_announced = True

    worker.join()
    if "error" in result:
        raise result["error"]
    debug_log(f"{description} snapshot ready: {result['path']}", "voice")
    return result["path"]
