# Modified for Jarvis from antgroup/ditto-talkinghead (Apache-2.0, commit c3e47ee):
# the eyes are measured from Ditto's 203-point landmarks instead of MediaPipe's
# 478-point mesh, with the same formulas. Two-dimensional landmarks carry no
# depth, so the gaze's depth component is zero; on a portrait seen from the
# front the measures match MediaPipe's to within a few hundredths.
import numpy as np

# The image-left eye is the ring 0..23 (0 the outer corner, 12 the inner one, the
# top running 1..11 and the bottom 13..23 back), the image-right eye the ring
# 24..47 (24 inner, 36 outer), and the irises are 197 and 198. Each pair is a
# point of the lower lid and the point of the upper lid above it.
_LEFT = {"corners": (0, 12), "lids": ((18, 6), (21, 3), (15, 9)), "iris": 197}
_RIGHT = {"corners": (36, 24), "lids": ((42, 30), (45, 27), (39, 33)), "iris": 198}


def _measure(p, eye):
    outer, inner = p[eye["corners"][0]], p[eye["corners"][1]]
    width = np.linalg.norm(outer - inner)
    opening = sum(np.linalg.norm(p[a] - p[b]) for a, b in eye["lids"]) / (width + 1e-8)
    centre = (outer + inner) / 2
    ball = p[eye["iris"]]
    dist = np.linalg.norm(ball - centre)
    eye_dir = (outer - inner) / (width + 1e-8)
    move = ((ball - centre) / (dist + 1e-8) - eye_dir) * dist
    return opening, np.append(move, 0.0)


def eye_info_from_lmk203(lmk203, size):
    """Openness of both eyes [1, 2] and gaze [1, 6], from 203 landmarks given in
    the pixels of a square crop of side ``size``."""
    p = np.asarray(lmk203, np.float64) / size
    l_open, l_move = _measure(p, _LEFT)
    r_open, r_move = _measure(p, _RIGHT)
    eye_open = np.array([[l_open, r_open]], np.float32)
    eye_ball = np.stack([l_move, r_move], -1).reshape(1, 6).astype(np.float32)
    return eye_open, eye_ball
