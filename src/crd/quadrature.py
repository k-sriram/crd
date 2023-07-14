from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import voigt_profile

from crd.typing import Array, Float, UFloat


@dataclass(slots=True)
class Quadrature:
    """A quadrature is a set of points and weights used to approximate an integral."""

    points: Array
    weights: Array

    def norm(self) -> Float:
        return Float(np.sum(self.weights))

    def normalize(self, value: UFloat = Float(1.0)) -> Quadrature:
        self.weights *= value / self.norm()
        return self

    def integrate(self, f: Callable[[Array | UFloat], Array | UFloat]) -> Float:
        return np.sum(self.weights * Float(f(self.points)))

    def integrate_points(self, f: Array) -> Float:
        if f.dtype != Float:
            f = f.astype(Float)
        if len(f) != len(self):
            raise ValueError("f must have the same length as the quadrature")
        return np.sum(self.weights * f)

    @property
    def n(self) -> int:
        return len(self)

    def __getitem__(self, index: int) -> Float:
        return self.points[index]

    def __len__(self) -> int:
        return len(self.points)


def integrate_quadrature(
    f: Callable[[Array | UFloat], Array | UFloat] | Array, q: Quadrature
) -> Float:
    if isinstance(f, np.ndarray):
        return q.integrate_points(f)
    elif isinstance(f, Sequence):
        return q.integrate_points(np.array(f))
    else:
        return q.integrate(f)


def gauss_legendre(
    n: int, limits: tuple[UFloat, UFloat] = (Float(-1.0), Float(1.0))
) -> Quadrature:
    """Gauss-Legendre quadrature."""
    a = Float(limits[0])
    b = Float(limits[1])
    x, w = np.polynomial.legendre.leggauss(n)
    x = x * (b - a) / 2 + (a + b) / 2
    w = w * (b - a) / 2
    return Quadrature(x, w)


def double_gauss_legendre(n: int, limit: UFloat = 1.0) -> Quadrature:
    """Double Gauss-Legendre quadrature. Gives a quadrature between
    (-limit, limit), with each half of the domain ((-limit, 0) & (0, limit))
    being a gauss legendre quadrature.
    """
    glquad1 = gauss_legendre(n, (-limit, Float(0.0)))
    glquad2 = gauss_legendre(n, (Float(0.0), limit))
    x1, w1 = glquad1.points, glquad1.weights
    x2, w2 = glquad2.points, glquad2.weights
    x = np.concatenate((x1, x2))
    w = np.concatenate((w1, w2)) / Float(2.0)
    return Quadrature(x, w)


def trapezoid(a: UFloat, b: UFloat, n: int) -> Quadrature:
    """Uses trapezoid rule to integrate. Creates a uniform set of n points
    between a and b.
    """
    a, b = Float(a), Float(b)
    x = np.linspace(a, b, n, dtype=Float)
    return trapezoid_points(x)


def trapezoid_points(x: Array) -> Quadrature:
    """Trapezoid quadrature. The array x defines a custom basis."""
    w = np.diff(x, append=Float(0.0)) + np.diff(x, prepend=Float(0.0))
    w /= Float(2.0)
    return Quadrature(x, w)


def log_uniform(log_min: int, log_max: int, points_per_decade: int) -> Quadrature:
    """Trapezoid quadrature where the points are spaced uniformly on the
    log-scale."""
    n = int((log_max - log_min) * points_per_decade + 1)
    x = np.logspace(log_min, log_max, n, dtype=Float)
    return trapezoid_points(x)


def double_log_uniform(
    log_min: int, log_max: int, points_per_decade: int
) -> Quadrature:
    """Trapezoid quadrature where the first half is uniform on log scale
    and in the second half the spacings are mirrored as in the first half.
    Goes from 10^log_min_tau to 2 * 10^log_max_tau.
    """
    q_half = log_uniform(log_min, log_max, points_per_decade)
    x = q_half.points
    x = np.concatenate((x, 2 * x[-1] - x[-2::-1]))
    return trapezoid_points(x)


def cosines(n: int) -> Quadrature:
    """Gives Double Gauss Legendre quadrature between (-1, 1).
    This is suitable for angle-cosines in a plane-parallel atmosphere.
    """
    return double_gauss_legendre(n, limit=1.0)


def freqs(
    phi_f: Callable[[Float], UFloat],
    x_max: UFloat = Float(6.0),
    *,
    n: int | None = None,
    n_per_unit: int = 4,
    half: bool = False,
) -> tuple[Quadrature, Array]:
    x_max = Float(x_max)
    if half:
        if n is None:
            n = int(x_max * n_per_unit + 1)
        quad = trapezoid(Float(0.0), x_max, n)
    else:
        if n is None:
            n = int(2 * x_max * n_per_unit + 1)
        quad = trapezoid(-x_max, x_max, n)
    phi = _freqs_points(phi_f, quad)
    return quad, phi


def freqs_voigt(
    sigma: float = 1 / np.sqrt(2),
    gamma: float = 0.0,
    x_max: UFloat = Float(6.0),
    *,
    n: int | None = None,
    n_per_unit: int = 4,
    half: bool = False,
) -> tuple[Quadrature, Array]:
    return freqs(
        lambda x: voigt_profile(float(x), sigma, gamma),
        x_max,
        n=n,
        n_per_unit=n_per_unit,
        half=half,
    )


def _freqs_points(phi_f: Callable[[Float], UFloat], quad_mut: Quadrature) -> Array:
    phi = np.array([Float(phi_f(x)) for x in quad_mut.points])
    quad_mut.weights /= quad_mut.integrate_points(phi)
    return phi
