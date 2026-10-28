"""Pool geometry: shoebox with a linearly tilted floor.

Implements RESEARCH_PLAN.md section 4.2 rows 3 (geometry), 9 (cutoff) and 13
(positions), and the pool presets of section 6.3 / Appendix A.

Coordinates
-----------
x in [0, L] along the pool length, y in [0, W] across it, z the depth,
**positive down** from the free surface (z = 0).  The floor is the plane

    z = H(x) = H0 + (H1 - H0) x / L,

i.e. it tilts along x only.  The pool is the convex polyhedron
{0 <= x <= L, 0 <= y <= W, 0 <= z <= H(x)}, written as six half-spaces
n . p <= d with outward unit normals n (see :meth:`Pool.planes`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FT = 0.3048
SURFACE_NAMES = ("x0", "xL", "y0", "yW", "surface", "floor")
"""Order of the per-surface columns used throughout uwsim.ism."""


def cutoff_frequency(c: float, H: float) -> float:
    """Lowest mode cutoff of an isovelocity waveguide with a pressure-release
    top and a rigid bottom: f_c = c / (4 H) (plan row 9)."""
    return c / (4.0 * H)


def mirror_point(p: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Mirror point(s) ``p`` (..., 3) across the plane n . p = d (|n| = 1)."""
    p = np.asarray(p, dtype=float)
    return p - 2.0 * (p @ n - d)[..., None] * n


@dataclass(frozen=True)
class Pool:
    """Shoebox pool, L x W, floor depth H0 at x = 0 and H1 at x = L [m]."""

    L: float
    W: float
    H0: float
    H1: float | None = None
    name: str = field(default="pool", compare=False)

    def __post_init__(self) -> None:
        if self.H1 is None:
            object.__setattr__(self, "H1", self.H0)
        for k in ("L", "W", "H0", "H1"):
            if not getattr(self, k) > 0:
                raise ValueError(f"Pool.{k} must be > 0")

    # ---- depth -------------------------------------------------------------
    @property
    def slope(self) -> float:
        """dH/dx of the floor."""
        return (self.H1 - self.H0) / self.L

    @property
    def is_flat(self) -> bool:
        return self.H0 == self.H1

    @property
    def mean_depth(self) -> float:
        """Pool-average depth (H0 + H1) / 2."""
        return 0.5 * (self.H0 + self.H1)

    @property
    def volume(self) -> float:
        return self.L * self.W * self.mean_depth

    def depth_at(self, x: np.ndarray | float) -> np.ndarray:
        """Floor depth H(x); x is clipped to [0, L]."""
        return self.H0 + self.slope * np.clip(np.asarray(x, dtype=float), 0.0, self.L)

    def mean_depth_along(self, src: np.ndarray, rx: np.ndarray) -> float:
        """Mean floor depth under the straight src--rx path.

        The floor is linear in x, so the path average equals the depth at the
        midpoint x.  This is the depth used for the higher-order (>= 3) image
        approximation in :mod:`uwse.uwsim.ism`.
        """
        return float(self.depth_at(0.5 * (np.asarray(src)[0] + np.asarray(rx)[0])))

    # ---- planes and containment -------------------------------------------
    def floor_plane(self) -> tuple[np.ndarray, float]:
        """Outward unit normal and offset of the floor: n . p = d."""
        n = np.array([-self.slope, 0.0, 1.0])
        norm = np.linalg.norm(n)
        return n / norm, self.H0 / norm

    def planes(self) -> dict[str, tuple[np.ndarray, float]]:
        """All six boundaries as {name: (outward unit normal, d)}; inside is n . p <= d."""
        return {
            "x0": (np.array([-1.0, 0.0, 0.0]), 0.0),
            "xL": (np.array([1.0, 0.0, 0.0]), float(self.L)),
            "y0": (np.array([0.0, -1.0, 0.0]), 0.0),
            "yW": (np.array([0.0, 1.0, 0.0]), float(self.W)),
            "surface": (np.array([0.0, 0.0, -1.0]), 0.0),
            "floor": self.floor_plane(),
        }

    def contains(self, p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """True where point(s) p (..., 3) lie in the closed water volume (within tol [m])."""
        p = np.asarray(p, dtype=float)
        ok = np.ones(p.shape[:-1], dtype=bool)
        for n, d in self.planes().values():
            ok &= p @ n <= d + tol
        return ok

    def check_position(self, p: np.ndarray, label: str = "point", margin: float = 0.0) -> np.ndarray:
        """Return p as a float array, or raise ValueError if it is not strictly
        inside the water with at least ``margin`` [m] to every boundary."""
        p = np.asarray(p, dtype=float)
        if p.shape != (3,):
            raise ValueError(f"{label} must be (x, y, z), got shape {p.shape}")
        for name, (n, d) in self.planes().items():
            if not p @ n < d - margin:
                raise ValueError(
                    f"{label} {p.tolist()} is not inside the water of {self.name!r} "
                    f"(violates the {name} boundary with margin {margin} m)"
                )
        return p

    # ---- acoustics --------------------------------------------------------
    def cutoff_frequency(self, c: float = 1500.0, depth: float | None = None) -> float:
        """Waveguide cutoff c / (4 H) at ``depth`` (default: the mean depth)."""
        return cutoff_frequency(c, self.mean_depth if depth is None else depth)

    def cutoff_range(self, c: float = 1500.0) -> tuple[float, float]:
        """(deep-end, shallow-end) cutoffs; the slope spreads the cutoff (plan row 3)."""
        return cutoff_frequency(c, max(self.H0, self.H1)), cutoff_frequency(c, min(self.H0, self.H1))


def avery_competition_pool() -> Pool:
    """Avery Aquatic Center competition pool, 37 x 20 m, 3.35--4.27 m deep.

    Which end is deep is not documented in the plan; H0 = 3.35 m at x = 0 is an
    assumption to be replaced by the measured depth profile (section 6.3).
    """
    return Pool(L=37.0, W=20.0, H0=3.35, H1=4.27, name="avery_competition")


def maas_dive_well() -> Pool:
    """Maas dive well, 25 x 17 m, flat 18 ft (5.486 m) deep."""
    return Pool(L=25.0, W=17.0, H0=18 * FT, H1=18 * FT, name="maas_dive_well")
