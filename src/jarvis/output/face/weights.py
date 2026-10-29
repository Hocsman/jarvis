"""Ditto's model files: which ones the renderer needs, where they live, and
fetching the ones that are missing.

The files come from one pinned revision of the Hugging Face repository. Each is
downloaded into a staging folder, checked against its SHA-256 there, and only
then moved into place: a file at its final path has always matched, and one
that does not match is deleted, never loaded. The request sends no token.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

from ...debug import debug_log

REPO = "digital-avatar/ditto-talkinghead"
REVISION = "e4a2f60328ee7c32af585ac4b3cce299e4c8e254"

# (path in the repository, SHA-256, size in bytes)
FILES = (
    ("ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "8e4c9ecb8dbb5aa4d22cb3f9b2c6ceabc62499b04af5fe074826d5b1044c4146", 31002),
    ("ditto_pytorch/models/appearance_extractor.pth", "5279bb8654293dbdf327030b397f107237dd9212fb11dd75b83dfb635211ceb5", 3387959),
    ("ditto_pytorch/models/decoder.pth", "4780afc7909a9f84e24c01d73b31a555ef651521a1fe3b2429bd04534d992aee", 221813590),
    ("ditto_pytorch/models/lmdm_v0.4_hubert.pth", "40ef5aa80b76e9fc45a0f606cdd507bbeb09d3d078b016ac61fc47e369be8c94", 191247715),
    ("ditto_pytorch/models/motion_extractor.pth", "251e6a94ad667a1d0c69526d292677165110ef7f0cf0f6d199f0e414e8aa0ca5", 112545506),
    ("ditto_pytorch/models/stitch_network.pth", "3652d5a3f95099141a56986aaddec92fadf0a73c87a20fac9a2c07c32b28b611", 2393098),
    ("ditto_pytorch/models/warp_network.pth", "2f61a6f265fe344f14132364859a78bdbbc2068577170693da57fb96d636e282", 182180086),
    ("ditto_pytorch/aux_models/det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91", 16923827),
    ("ditto_pytorch/aux_models/2d106det.onnx", "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf", 5030888),
    ("ditto_pytorch/aux_models/landmark203.onnx", "31d22a5041326c31f19b78886939a634a5aedcaa5ab8b9b951a1167595d147db", 114666491),
    ("ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx", "04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697", 1460740880),
)

CONFIG = "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
DATA_ROOT = "ditto_pytorch"
WARP = "ditto_pytorch/models/warp_network.pth"
DECODER = "ditto_pytorch/models/decoder.pth"
_STAGING = ".staging"


def digest(name: str) -> str:
    """The published SHA-256 of one of the files."""
    return next(sha for file, sha, _size in FILES if file == name)


def models_dir() -> Path:
    return Path("~/.local/share/jarvis/models/ditto").expanduser()


def missing(root: Path) -> list:
    """The files absent from ``root`` or of the wrong size."""
    return [name for name, _sha, size in FILES
            if not (root / name).is_file() or (root / name).stat().st_size != size]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root: Path, download: Optional[Callable[..., str]] = None) -> None:
    """Download every missing file into ``root`` and check it.

    Raises ``RuntimeError`` naming the file when a download fails or a file does
    not match its digest. ``download`` defaults to ``huggingface_hub``'s, which
    writes under a temporary name and renames once complete.
    """
    if download is None:
        from huggingface_hub import hf_hub_download as download
    staging = root / _STAGING
    for name in missing(root):
        debug_log(f"face: fetching {name}", "face")
        try:
            path = Path(download(repo_id=REPO, filename=name, revision=REVISION, local_dir=str(staging),
                                 token=False))
        except Exception as e:
            raise RuntimeError(f"could not download {name}: {e}") from e
        if _sha256(path) != digest(name):
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{name} does not match its published digest")
        final = root / name
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, final)
