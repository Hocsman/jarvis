"""The portrait the face is drawn from (`face.spec.md`, "The model", "Framing").

It is read into memory and nowhere else. Ditto animates a square around the
face that is wider than the face, so the picture is first widened on every
side by stretching its own edges and blurring them: the square then never runs
off the picture, and what it finds beyond the edge looks like more of the
out-of-focus background.
"""

from pathlib import Path

import numpy as np

MAX_SIDE = 1920        # a larger picture is scaled down first
WIDEN = 0.25           # added on each side, as a share of the longer side
BLUR = 0.03            # the blur of the added edges, as a share of the longer side
RAMP = 0.02            # how far into the picture the blur fades in, likewise


class UnreadablePortrait(ValueError):
    """The file is missing or is not an image."""


def load(path: str) -> np.ndarray:
    """The portrait as RGB, widened; raises UnreadablePortrait."""
    import cv2

    try:
        data = np.fromfile(str(Path(path).expanduser()), np.uint8)   # cv2.imread cannot open every Windows path
    except OSError as e:
        raise UnreadablePortrait(f"the portrait cannot be read: {e}") from e
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if bgr is None:
        raise UnreadablePortrait("the portrait is not an image")
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(widen(bgr), cv2.COLOR_BGR2RGB)


def widen(img: np.ndarray) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    side = max(h, w)
    pad = round(WIDEN * side)
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    soft = cv2.GaussianBlur(big, (0, 0), BLUR * side)
    ramp = max(1.0, RAMP * side)
    y, x = np.arange(h + 2 * pad)[:, None], np.arange(w + 2 * pad)[None, :]
    inside = np.minimum(np.minimum(x - pad, w + pad - 1 - x), np.minimum(y - pad, h + pad - 1 - y))
    keep = np.clip((inside + 1) / ramp, 0.0, 1.0)[..., None]
    return (big * keep + soft * (1.0 - keep)).astype(np.uint8)
