"""Static geometry for the orb: icosphere mesh + particle halo.

Pure-numpy. No GL imports here so the module is testable headlessly.
The orb_widget consumes the returned arrays and uploads them as VBOs.

Two builders
------------
- ``build_icosphere(subdivisions: int) -> Mesh``: a unit sphere from
  icosahedron midpoint subdivision. Three subdivisions is the spec
  default (~320 triangles, ~162 vertices). One additional subdivision
  ~quadruples the triangle count.
- ``build_particles(count: int, seed: int = 0) -> Particles``: a halo
  of points distributed across orbits with random radius/inclination/
  phase. Positions are *parametric* (rendered per-frame by the vertex
  shader from the orbit parameters); the buffer contains the orbit
  spec, not the live position.

Both builders return frozen dataclasses with the numpy buffers in the
exact dtype / layout the shaders expect (float32, contiguous, packed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ── Icosphere ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mesh:
    """An indexed triangle mesh on the unit sphere.

    ``positions`` and ``normals`` are identical for the unit sphere
    (the position vector from the centre *is* the normal). They are
    kept as two separate buffers so the vertex shader does not need
    to renormalise per-vertex; the displacement step in the vertex
    shader mutates positions while normals stay as authored.
    """
    positions: np.ndarray  # (V, 3) float32, unit length
    normals: np.ndarray    # (V, 3) float32, == positions on the unit sphere
    indices: np.ndarray    # (T*3,) uint32, GL_TRIANGLES order

    @property
    def vertex_count(self) -> int:
        return self.positions.shape[0]

    @property
    def triangle_count(self) -> int:
        return self.indices.shape[0] // 3


def _icosahedron_seed() -> Tuple[np.ndarray, np.ndarray]:
    """Return the 12 vertices and 20 triangles of a unit icosahedron.

    Vertices are constructed from the golden-ratio rectangles and
    normalised to the unit sphere.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = np.array(
        [
            (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
            (0, -1,  phi), (0,  1,  phi), (0, -1, -phi), (0,  1, -phi),
            ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
        ],
        dtype=np.float32,
    )
    # Normalise to the unit sphere.
    lengths = np.linalg.norm(raw, axis=1, keepdims=True)
    vertices = (raw / lengths).astype(np.float32)

    faces = np.array(
        [
            ( 0, 11,  5), ( 0,  5,  1), ( 0,  1,  7), ( 0,  7, 10), ( 0, 10, 11),
            ( 1,  5,  9), ( 5, 11,  4), (11, 10,  2), (10,  7,  6), ( 7,  1,  8),
            ( 3,  9,  4), ( 3,  4,  2), ( 3,  2,  6), ( 3,  6,  8), ( 3,  8,  9),
            ( 4,  9,  5), ( 2,  4, 11), ( 6,  2, 10), ( 8,  6,  7), ( 9,  8,  1),
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def build_icosphere(subdivisions: int = 2) -> Mesh:
    """Subdivide an icosahedron ``subdivisions`` times and return a
    unit-sphere mesh ready for GL upload.

    Each step splits every triangle into 4 via midpoints projected
    back onto the sphere. Cost: O(20 * 4^n) triangles:
    - n=2: 320 triangles, 162 vertices  (spec default, matches the
      "~320 tris" target in the orb spec)
    - n=3: 1280 triangles, 642 vertices (smoother at the cost of GPU
      bandwidth; visually marginal at 320x320 px)
    - n=4: 5120 triangles, 2562 vertices

    The widget may bump this to 3 for displacement-heavy states; the
    default keeps the silhouette honest to the spec.
    """
    if subdivisions < 0:
        raise ValueError(f"subdivisions must be >= 0, got {subdivisions}")

    verts_list, faces = _icosahedron_seed()
    verts: list[np.ndarray] = [v for v in verts_list]
    # Cache midpoint per edge so adjacent triangles share the new vertex.
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint_index(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        mid = (verts[a] + verts[b]) * 0.5
        mid = mid / np.linalg.norm(mid)
        verts.append(mid.astype(np.float32))
        idx = len(verts) - 1
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = np.empty((faces.shape[0] * 4, 3), dtype=np.uint32)
        for i, (a, b, c) in enumerate(faces):
            ab = midpoint_index(int(a), int(b))
            bc = midpoint_index(int(b), int(c))
            ca = midpoint_index(int(c), int(a))
            base = i * 4
            new_faces[base + 0] = (a, ab, ca)
            new_faces[base + 1] = (b, bc, ab)
            new_faces[base + 2] = (c, ca, bc)
            new_faces[base + 3] = (ab, bc, ca)
        faces = new_faces

    positions = np.ascontiguousarray(np.stack(verts), dtype=np.float32)
    # Unit sphere: normal == position. Cheaper than recomputing per draw.
    normals = positions.copy()
    indices = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)
    return Mesh(positions=positions, normals=normals, indices=indices)


# ── Particle halo ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Particles:
    """A halo of orbit-parameterised points.

    Each row of ``orbits`` describes one particle as
    ``(radius, inclination, longitude_phase, angular_speed, size)``.
    The vertex shader expands these into a live position per frame
    from a single time uniform, so the CPU only uploads the spec
    once at startup. Animation is GPU-side and reacts to high-band
    audio energy via a uniform.
    """
    orbits: np.ndarray  # (N, 5) float32: r, inclination, phase, speed, size

    @property
    def count(self) -> int:
        return self.orbits.shape[0]


def build_particles(count: int = 256, seed: int = 0) -> Particles:
    """Create a deterministic halo of ``count`` particles around the orb.

    Deterministic by default (``seed=0``) so the spawn layout is
    identical across launches. Pass a non-zero seed for variation.

    Distribution:
    - radius: uniform in [1.25, 1.75] so the halo lives just outside
      the orb's unit-sphere surface (mesh + displacement room).
    - inclination: uniform in [-pi/2, pi/2] (full latitude range,
      slight pole bias falls out of uniform-cos in shader).
    - longitude phase: uniform in [0, 2*pi].
    - angular speed: uniform in [0.15, 0.45] rad/s (slow drift).
    - size: uniform in [0.005, 0.020] (NDC-ish at 320x320).
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    rng = np.random.default_rng(seed)
    orbits = np.empty((count, 5), dtype=np.float32)
    orbits[:, 0] = rng.uniform(1.25, 1.75, count)        # radius
    orbits[:, 1] = rng.uniform(-math.pi / 2, math.pi / 2, count)  # inclination
    orbits[:, 2] = rng.uniform(0.0, 2.0 * math.pi, count)         # phase
    orbits[:, 3] = rng.uniform(0.15, 0.45, count)        # angular speed
    orbits[:, 4] = rng.uniform(0.005, 0.020, count)      # particle size
    return Particles(orbits=np.ascontiguousarray(orbits, dtype=np.float32))
