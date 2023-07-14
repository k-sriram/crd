from typing import Literal
import warnings

import numpy as np

from crd import formalsol
from crd.typing import ArrayND, Float, UFloat

ONE = Float(1.0)


def ali(
    grid: formalsol.Grid,
    epsilon: UFloat,
    s0: ArrayND | Literal["zero", "one"] = "one",
    *,
    tol: UFloat = Float(1e-8),
    max_iter: int = 1000,
    fs: formalsol.FormalSolver | None = None,
) -> list[ArrayND]:
    epsilon, tol = Float(epsilon), Float(tol)

    if fs is None:
        fs = grid.solver()
    if s0 == "zero":
        s = np.zeros_like(grid.b, dtype=Float)
    elif s0 == "one":
        s = np.ones_like(grid.b, dtype=Float)
    else:
        raise ValueError("Ivalid s0")

    lstar = fs.lstar
    dinv = np.reciprocal(ONE - (ONE - epsilon) * lstar)
    epsb = epsilon * grid.b

    shist = [s]
    for _ in range(max_iter):
        if np.isnan(shist[-1]).any():
            raise ValueError("NaN in s")

        s = shist[-1]
        sx = (grid.r + grid.phi[None, :] * s[:, None]) / (grid.r + grid.phi[None, :])
        Ix = fs.solve(sx, grid.bc)
        _, J_bar = grid.calc_J_bar(Ix)
        sfs = (ONE - epsilon) * J_bar + epsb
        ds = dinv * (sfs - shist[-1])
        shist.append(shist[-1] + ds)

        if max_rel_error(shist, -1) < tol:
            break
    else:
        warnings.warn("Warning: maximum number of iterations reached.")

    return shist


def max_rel_error(shist, i):
    return np.max(np.abs(shist[i] - shist[i - 1]) / np.abs(shist[i - 1]))
