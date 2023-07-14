from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.special import voigt_profile

Array = npt.NDArray[np.float64]


@dataclass(slots=True)
class Quadrature:
    """A quadrature is a set of points and weights used to approximate an integral."""

    points: Array
    weights: Array

    def norm(self) -> float:
        return np.sum(self.weights)  # type: ignore

    def normalize(self, value: float = 1.0) -> Quadrature:
        self.weights *= value / self.norm()
        return self

    def integrate(self, f: Callable[[Array | float], Array | float]) -> float:
        return np.sum(self.weights * f(self.points))

    def integrate_points(self, f: Array) -> float:
        if len(f) != len(self):
            raise ValueError("f must have the same length as the quadrature")
        return np.sum(self.weights * f)

    @property
    def n(self) -> int:
        return len(self)

    def __getitem__(self, index: int) -> float:
        return self.points[index]

    def __len__(self) -> int:
        return len(self.points)


def integrate_quadrature(
    f: Callable[[Array | float], Array | float] | Array, q: Quadrature
) -> float:
    if isinstance(f, np.ndarray):
        return q.integrate_points(f)
    elif isinstance(f, Sequence):
        return q.integrate_points(np.array(f))
    else:
        return q.integrate(f)


def gauss_legendre(n: int, limits: tuple[float, float] = (-1.0, 1.0)) -> Quadrature:
    """Gauss-Legendre quadrature."""
    a, b = limits
    x, w = np.polynomial.legendre.leggauss(n)
    x = x * (b - a) / 2 + (a + b) / 2
    w = w * (b - a) / 2
    return Quadrature(x, w)


def double_gauss_legendre(n: int, limit: float = 1.0) -> Quadrature:
    """Double Gauss-Legendre quadrature. Gives a quadrature between
    (-limit, limit), with each half of the domain ((-limit, 0) & (0, limit))
    being a gauss legendre quadrature.
    """
    glquad1 = gauss_legendre(n, (-limit, 0.0))
    glquad2 = gauss_legendre(n, (0.0, limit))
    x1, w1 = glquad1.points, glquad1.weights
    x2, w2 = glquad2.points, glquad2.weights
    x = np.concatenate((x1, x2))
    w = np.concatenate((w1, w2)) / 2
    return Quadrature(x, w)


def trapezoid(a: float, b: float, n: int) -> Quadrature:
    """Uses trapezoid rule to integrate. Creates a uniform set of n points
    between a and b.
    """
    x = np.linspace(a, b, n)
    return trapezoid_points(x)


def trapezoid_points(x: Array) -> Quadrature:
    """Trapezoid quadrature. The array x defines a custom basis."""
    a, b = x[0], x[-1]
    n = len(x)
    w = np.ones(n)
    w[0] = 0.5
    w[-1] = 0.5
    w *= (b - a) / (n - 1)
    return Quadrature(x, w)


def log_uniform(log_min: int, log_max: int, points_per_decade: int) -> Quadrature:
    """Trapezoid quadrature where the points are spaced uniformly on the
    log-scale."""
    n = int((log_max - log_min) * points_per_decade + 1)
    x = np.logspace(log_min, log_max, n)
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
    phi_f: Callable[[float], float],
    x_max: float = 6.0,
    *,
    n: int | None = None,
    n_per_unit: int = 4,
    half: bool = False,
) -> tuple[Quadrature, Array]:
    if half:
        if n is None:
            n = int(x_max * n_per_unit + 1)
        quad = trapezoid(0.0, x_max, n)
    else:
        if n is None:
            n = int(2 * x_max * n_per_unit + 1)
        quad = trapezoid(-x_max, x_max, n)
    phi = _freqs_points(phi_f, quad)
    return quad, phi


def freqs_voigt(
    sigma: float = 1 / np.sqrt(2),
    gamma: float = 0.0,
    x_max: float = 6.0,
    *,
    n: int | None = None,
    n_per_unit: int = 4,
    half: bool = False,
) -> tuple[Quadrature, Array]:
    return freqs(
        lambda x: voigt_profile(x, sigma, gamma),
        x_max,
        n=n,
        n_per_unit=n_per_unit,
        half=half,
    )


def _freqs_points(phi_f: Callable[[float], float], quad_mut: Quadrature) -> Array:
    phi = np.array([phi_f(x) for x in quad_mut.points])
    quad_mut.weights /= quad_mut.integrate_points(phi)
    return phi
