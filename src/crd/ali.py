from typing import Annotated, Literal
import warnings

import numpy as np
import numpy.typing as npt

from crd import formalsol

Array = npt.NDArray[np.float64]
ArrayND = Annotated[Array, ("ND",)]
ArrayNA = Annotated[Array, ("NA",)]
ArrayNF = Annotated[Array, ("NF",)]
ArrayNDxND = Annotated[Array, ("ND", "ND")]
ArrayNDxNA = Annotated[Array, ("ND", "NA")]
ArrayNDxNF = Annotated[Array, ("ND", "NF")]
ArrayNAxNF = Annotated[Array, ("NA", "NF")]
ArrayNDxNAxNF = Annotated[Array, ("ND", "NA", "NF")]


def ali(
    grid: formalsol.Grid,
    epsilon: float,
    s0: ArrayND | Literal["zero", "one"] = "one",
    *,
    tol: float = 1e-8,
    max_iter: int = 1000,
    fs: formalsol.FormalSolver | None = None,
) -> list[ArrayND]:
    if fs is None:
        fs = grid.solver()

    if s0 == "zero":
        s = np.zeros_like(grid.b)
    elif s0 == "one":
        s = np.ones_like(grid.b)
    else:
        raise ValueError("Ivalid s0")

    lstar = fs.lstar
    dinv = np.reciprocal(1 - (1 - epsilon) * lstar)
    epsb = epsilon * grid.b

    shist = [s]
    for _ in range(max_iter):
        if np.isnan(shist[-1]).any():
            raise ValueError("NaN in s")
        Ix = fs.solve(shist[-1], grid.bc)
        _, J_bar = grid.calc_J_bar(Ix)
        sfs = (1 - epsilon) * J_bar + epsb
        ds = dinv * (sfs - shist[-1])
        shist.append(shist[-1] + ds)

        if max_rel_error(shist, -1) < tol:
            break
    else:
        warnings.warn("Warning: maximum number of iterations reached.")

    return shist


def max_rel_error(shist, i):
    return np.max(np.abs(shist[i] - shist[i - 1]) / np.abs(shist[i - 1]))
