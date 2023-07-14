from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

from crd import quadrature
from crd.quadrature import Quadrature

Array = npt.NDArray[np.float64]
ArrayND = Annotated[Array, ("ND",)]
ArrayNA = Annotated[Array, ("NA",)]
ArrayNF = Annotated[Array, ("NF",)]
ArrayNDxND = Annotated[Array, ("ND", "ND")]
ArrayNDxNA = Annotated[Array, ("ND", "NA")]
ArrayNDxNF = Annotated[Array, ("ND", "NF")]
ArrayNAxNF = Annotated[Array, ("NA", "NF")]
ArrayNDxNAxNF = Annotated[Array, ("ND", "NA", "NF")]
ArrayMask = npt.NDArray[np.bool_]
ArrayMaskNA = Annotated[ArrayMask, ("NA",)]

MAX_TAYLOR_SERIES = 20


class FormalSolver:
    """Formal solver for 1D plane-parallel radiative transfer.
    A formal solution is a solution to the radiative transfer equation where the
    source function is given and the intensity is to be calculated.

    This class precomputes the coefficients of the formal solution which are
    independent of the source function. It also provides lstar, which is the
    diagonal of the Lambda operator.

    The formulas used are based on Olson & Kunasz (1987: Journal of Quantitative
    Spectroscopy and Radiative Transfer, 38, 325).
    """

    def __init__(
        self,
        tau: Quadrature,
        mu: Quadrature,
        x: Quadrature,
        phi: ArrayNF,
        # If it is possible to provide dtau more precisely than from a simple
        # difference of tau.points, then this can be provided here.
        dtau: ArrayND | None = None,
    ):
        """Initialize the formal solver.

        Parameters
        ----------
        tau : Quadrature
            The optical depth grid.

        mu : Quadrature
            The angular grid.

        x : Quadrature
            The frequency grid.

        phi : Array(shape=(NF,))
            The normalized line profile.

        dtau : Array(shape=(ND - 1,)), optional
            The optical depth grid spacing. If it is possible to provide dtau
            more precisely than from a simple difference of tau.points, then
            this can be provided here. If not provided, it is calculated.
        """
        self.tau = tau
        self.mu = mu
        self.x = x
        self.phi = phi

        od = mu.points > 0  # outgoing directions
        id = mu.points < 0  # incoming directions

        nd = len(tau)
        na = len(mu)
        nf = len(x)

        if dtau is None:
            dtau = np.diff(tau.points, append=np.nan)

        # dtau(d) -> dtau_mn(d,m,n) = dtau(d) * phi(n) / |mu(m)|
        dtau_mn = _expand_dtau(dtau, mu.points, phi)
        dtau_mn_minus = np.roll(dtau_mn, 1, axis=0)

        assert dtau_mn.shape == (nd, na, nf), f"{dtau_mn.shape=} must be {(nd, na, nf)}"

        alpha_f, beta_f, gamma_f = propagation_coefficients_formula(
            od, id, nd, na, nf, dtau_mn, dtau_mn_minus
        )

        alpha_ts, beta_ts, gamma_ts = propagation_coefficients_taylor_series(
            od, id, nd, na, nf, dtau_mn, dtau_mn_minus
        )

        alpha = np.zeros((nd, na, nf))
        beta = np.zeros((nd, na, nf))
        gamma = np.zeros((nd, na, nf))

        small_dt = dtau_mn < 0.1
        alpha[small_dt] = alpha_ts[small_dt]
        alpha[~small_dt] = alpha_f[~small_dt]
        beta[small_dt] = beta_ts[small_dt]
        beta[~small_dt] = beta_f[~small_dt]
        gamma[small_dt] = gamma_ts[small_dt]
        gamma[~small_dt] = gamma_f[~small_dt]

        # Calculating lstar

        lstar_mn = np.zeros((nd, na, nf))
        lstar_mn[:-1, od, :] += beta[:-1, od, :]
        lstar_mn[1:, id, :] += beta[1:, id, :]
        # The following two lines are very small and are not present in Hubeny+Mihalas
        # They are only there in the Olson+Kunasz paper.
        # lstar_mn[:-2, od, :] += alpha[1:-1, od, :] * np.exp(-dtau_mn[:-2, od, :])
        # lstar_mn[2:, id, :] += gamma[1:-1, id, :] * np.exp(-dtau_mn[1:-1, id, :])

        lstar_n = np.sum(lstar_mn * mu.weights[None, :, None], axis=1)
        lstar = np.sum(lstar_n * phi[None, :] * x.weights[None, :], axis=1)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.outdirs = od
        self.indirs = id
        self.expdt = np.exp(-dtau_mn)
        self.lstar = lstar

    def solve(
        self, s: ArrayND | ArrayNDxNF | ArrayNDxNAxNF, bc: ArrayNAxNF
    ) -> ArrayNDxNAxNF:
        """Solve the formal solution for the given source function and boundary
        conditions.

        Parameters
        ----------
        s : Array(shape=(ND, NA, NF)) | Array(shape=(ND, NF)) | Array(shape=(ND,))
            The source function. The code allows us to provide a source function
            as a function of depth, frequency, and angle. However, the angle or
            both frequency and angle dependence can be omitted and provided as
            a 1D or 2D array instead.

        bc : Array(shape=(NA, NF))
            The boundary condition. Positive mu is for incoming intensity
            at bottom and negative mu is for incoming intensity at top.

        Returns
        -------
        Ix : Array(shape=(ND, NA, NF))
            The intensity.
        """
        od = self.outdirs
        id = self.indirs

        nd = len(self.tau)
        na = len(self.mu)
        nf = len(self.x)

        s = _conform_shape(s, nd, na, nf)

        Ix = np.zeros((nd, na, nf))

        # For mu > 0. Set boundary condition at bottom and solve upwards.
        Ix[-1, od, :] = bc[od, :]
        for d in range(nd - 2, 0, -1):
            Ix[d, od, :] = Ix[d + 1, od, :] * self.expdt[d, od, :] + (
                self.alpha[d, od, :] * s[d - 1, od, :]
                + self.beta[d, od, :] * s[d, od, :]
                + self.gamma[d, od, :] * s[d + 1, od, :]
            )
        # Linear interpolation for last step.
        Ix[0, od, :] = Ix[1, od, :] * self.expdt[0, od, :] + (
            self.beta[0, od, :] * s[0, od, :] + self.gamma[0, od, :] * s[1, od, :]
        )

        # For mu < 0. Set boundary condition at top and solve downwards.
        Ix[0, id, :] = bc[id, :]
        for d in range(1, nd - 1, 1):
            Ix[d, id, :] = Ix[d - 1, id, :] * self.expdt[d - 1, id, :] + (
                self.alpha[d, id, :] * s[d - 1, id, :]
                + self.beta[d, id, :] * s[d, id, :]
                + self.gamma[d, id, :] * s[d + 1, id, :]
            )
        # Linear interpolation for last step.
        Ix[-1, id, :] = Ix[-2, id, :] * self.expdt[-2, id, :] + (
            self.alpha[-1, id, :] * s[-2, id, :] + self.beta[-1, id, :] * s[-1, id, :]
        )

        return Ix


@dataclass(frozen=True)
class Grid:
    """A convenience class for storing the grids used in the formal solver."""

    tau: Quadrature
    mu: Quadrature
    x: Quadrature
    phi: ArrayNF
    bc: ArrayNAxNF
    b: ArrayND
    dtau: ArrayND

    @classmethod
    def new(
        cls,
        log_min_tau: int = -4,
        log_max_tau: int = 8,
        points_per_decade=5,
        depth_mirrored: bool = False,
        n_angles: int = 3,
        gamma: float = 0.0,
        x_max: float = 6.0,
        freqs_per_unit: int = 4,
        half_freqs: bool = False,
        bc_type: Literal["zero", "semi-inf"] = "zero",
        b: Literal["one"] = "one",
    ) -> Grid:
        if depth_mirrored:
            tau = quadrature.double_log_uniform(
                log_min_tau, log_max_tau, points_per_decade
            )
            dtau_half = np.diff(tau.points[: len(tau) // 2 + 1])
            dtau: ArrayND = np.concatenate(
                (dtau_half, dtau_half[-1::-1], np.array([np.nan]))
            )
        else:
            tau = quadrature.log_uniform(log_min_tau, log_max_tau, points_per_decade)
            dtau = np.diff(tau.points, append=np.nan)

        mu = quadrature.cosines(n_angles)

        x, phi = quadrature.freqs_voigt(
            gamma=gamma, x_max=x_max, n_per_unit=freqs_per_unit, half=half_freqs
        )

        bc = np.zeros((len(mu), len(x)))
        if bc_type == "semi-inf":
            bc[mu.points > 0.0, :] = 1.0

        if b == "one":
            b_arr = np.ones(len(tau))

        return cls(tau, mu, x, phi, bc, b_arr, dtau)

    def solver(self) -> FormalSolver:
        return FormalSolver(self.tau, self.mu, self.x, self.phi, self.dtau)

    def calc_J_bar(self, Ix: ArrayNDxNAxNF) -> tuple[ArrayNDxNF, ArrayND]:
        Jx = np.sum(Ix * self.mu.weights[None, :, None], axis=1)
        J_bar = np.sum(Jx * self.phi[None, :] * self.x.weights[None, :], axis=1)
        return Jx, J_bar


def _conform_shape(
    s: ArrayND | ArrayNDxNF | ArrayNDxNAxNF, nd: int, na: int, nf: int
) -> ArrayNDxNAxNF:
    if s.shape == (nd,):
        s = np.repeat(s, nf).reshape((nd, nf))

    if s.shape == (nd, nf):
        s = np.tile(s[:, None, :], (1, na, 1)).reshape((nd, na, nf))

    if s.shape != (nd, na, nf):
        raise ValueError(f"{s.shape=} must be {(nd, na, nf)}")
    return s


def _expand_dtau(dtau: ArrayND, mu: ArrayNA, phi: ArrayNF) -> ArrayNDxNAxNF:
    return (
        dtau[:, None, None]
        * np.reciprocal(np.abs(mu[None, :, None]))
        * phi[None, None, :]
    )
    # return np.einsum("d,m,n->dmn", dtau, np.reciprocal(np.abs(mu.points)), phi)


def propagation_coefficients_formula(
    od: ArrayMaskNA,
    id: ArrayMaskNA,
    nd: int,
    na: int,
    nf: int,
    dtau_mn: ArrayNDxNAxNF,
    dtau_mn_minus: ArrayNDxNAxNF,
) -> tuple[ArrayNDxNAxNF, ArrayNDxNAxNF, ArrayNDxNAxNF]:
    e0 = -np.expm1(-dtau_mn_minus)
    e1 = dtau_mn_minus - e0
    e2 = np.square(dtau_mn_minus) - 2 * e1

    e0_plus = e0[1:, :, :]
    e1_plus = e1[1:, :, :]
    e2_plus = e2[1:, :, :]

    alpha = np.zeros((nd, na, nf))
    alpha[1:-1, od, :] = (
        e2_plus[1:, od, :] - dtau_mn[1:-1, od, :] * e1_plus[1:, od, :]
    ) / (
        dtau_mn_minus[1:-1, od, :] * (dtau_mn[1:-1, od, :] + dtau_mn_minus[1:-1, od, :])
    )
    alpha[-1, od, :] = np.nan
    alpha[0, od, :] = 0.0
    alpha[1:-1, id, :] = e0[1:-1, id, :] + (
        e2[1:-1, id, :]
        - (dtau_mn[1:-1, id, :] + 2 * dtau_mn_minus[1:-1, id, :]) * e1[1:-1, id, :]
    ) / (
        dtau_mn_minus[1:-1, id, :] * (dtau_mn[1:-1, id, :] + dtau_mn_minus[1:-1, id, :])
    )
    alpha[0, id, :] = np.nan
    alpha[-1, id, :] = e0[-1, id, :] - e1[-1, id, :] / dtau_mn_minus[-1, id, :]

    beta = np.zeros((nd, na, nf))
    beta[1:-1, od, :] = (
        (dtau_mn[1:-1, od, :] + dtau_mn_minus[1:-1, od, :]) * e1_plus[1:, od, :]
        - e2_plus[1:, od, :]
    ) / (dtau_mn_minus[1:-1, od, :] * dtau_mn[1:-1, od, :])
    beta[-1, od, :] = np.nan
    beta[0, od, :] = e1_plus[0, od, :] / dtau_mn[0, od, :]
    beta[1:-1, id, :] = (
        (dtau_mn[1:-1, id, :] + dtau_mn_minus[1:-1, id, :]) * e1[1:-1, id, :]
        - e2[1:-1, id, :]
    ) / (dtau_mn[1:-1, id, :] * dtau_mn_minus[1:-1, id, :])
    beta[0, id, :] = np.nan
    beta[-1, id, :] = e1[-1, id, :] / dtau_mn_minus[-1, id, :]

    gamma = np.zeros((nd, na, nf))
    gamma[1:-1, od, :] = e0_plus[1:, od, :] + (
        e2_plus[1:, od, :]
        - (dtau_mn_minus[1:-1, od, :] + 2 * dtau_mn[1:-1, od, :]) * e1_plus[1:, od, :]
    ) / (dtau_mn[1:-1, od, :] * (dtau_mn[1:-1, od, :] + dtau_mn_minus[1:-1, od, :]))
    gamma[-1, od, :] = np.nan
    gamma[0, od, :] = e0_plus[0, od, :] - e1_plus[0, od, :] / dtau_mn[0, od, :]
    gamma[1:-1, id, :] = (
        e2[1:-1, id, :] - dtau_mn_minus[1:-1, id, :] * e1[1:-1, id, :]
    ) / (dtau_mn[1:-1, id, :] * (dtau_mn[1:-1, id, :] + dtau_mn_minus[1:-1, id, :]))
    gamma[0, id, :] = np.nan
    gamma[-1, id, :] = 0.0

    return alpha, beta, gamma


def propagation_coefficients_taylor_series(
    od: ArrayMaskNA,
    id: ArrayMaskNA,
    nd: int,
    na: int,
    nf: int,
    dtau_mn: ArrayNDxNAxNF,
    dtau_mn_minus: ArrayNDxNAxNF,
) -> tuple[ArrayNDxNAxNF, ArrayNDxNAxNF, ArrayNDxNAxNF]:
    alpha = np.zeros((nd, na, nf))
    alpha[1:-1, od, :] = prop_coeff_ts_d(
        dtau_mn[1:-1, od, :], dtau_mn_minus[1:-1, od, :]
    )
    alpha[-1, od, :] = np.nan
    alpha[0, od, :] = prop_coeff_ts_lin_d(dtau_mn[0, od, :])
    alpha[1:-1, id, :] = prop_coeff_ts_u(
        dtau_mn_minus[1:-1, id, :], dtau_mn[1:-1, id, :]
    )
    alpha[0, id, :] = np.nan
    alpha[-1, id, :] = prop_coeff_ts_lin_u(dtau_mn_minus[-1, id, :])

    beta = np.zeros((nd, na, nf))
    beta[1:-1, od, :] = prop_coeff_ts_0(
        dtau_mn[1:-1, od, :], dtau_mn_minus[1:-1, od, :]
    )
    beta[-1, od, :] = np.nan
    beta[0, od, :] = prop_coeff_ts_lin_0(dtau_mn[0, od, :])
    beta[1:-1, id, :] = prop_coeff_ts_0(
        dtau_mn_minus[1:-1, id, :], dtau_mn[1:-1, id, :]
    )
    beta[0, id, :] = np.nan
    beta[-1, id, :] = prop_coeff_ts_lin_0(dtau_mn_minus[-1, id, :])

    gamma = np.zeros((nd, na, nf))
    gamma[1:-1, od, :] = prop_coeff_ts_u(
        dtau_mn[1:-1, od, :], dtau_mn_minus[1:-1, od, :]
    )
    gamma[-1, od, :] = np.nan
    gamma[0, od, :] = prop_coeff_ts_lin_u(dtau_mn[0, od, :])
    gamma[1:-1, id, :] = prop_coeff_ts_d(
        dtau_mn_minus[1:-1, id, :], dtau_mn[1:-1, id, :]
    )
    gamma[0, id, :] = np.nan
    gamma[-1, id, :] = prop_coeff_ts_lin_d(dtau_mn_minus[-1, id, :])

    return alpha, beta, gamma


def prop_coeff_ts_d(dtu: Array, dtd: Array) -> Array:
    psid = np.zeros_like(dtu)
    for n in range(3, MAX_TAYLOR_SERIES):
        psid += np.power(-dtu, n) * (n - 2) / factorial(n)
    psid /= dtd * (dtu + dtd)

    return psid


def prop_coeff_ts_0(dtu: Array, dtd: Array) -> Array:
    psi0 = np.zeros_like(dtu)
    for n in range(3, MAX_TAYLOR_SERIES):
        psi0 += np.power(-dtu, n) * (2 - n) / factorial(n)
    psi0 /= dtd
    for n in range(2, MAX_TAYLOR_SERIES):
        psi0 += np.power(-dtu, n) / factorial(n)
    psi0 /= dtu

    return psi0


def prop_coeff_ts_u(dtu: Array, dtd: Array) -> Array:
    psiu = np.zeros_like(dtu)
    for n in range(2, MAX_TAYLOR_SERIES):
        psiu += np.power(-dtu, n) * (n - 1) / factorial(n)
    psiu *= dtd
    for n in range(3, MAX_TAYLOR_SERIES):
        psiu += np.power(-dtu, n) * (-2 + 3 * n - n**2) / factorial(n)
    psiu /= dtu * (dtd + dtu)

    return psiu


def prop_coeff_ts_lin_d(dt: Array) -> Array:
    psid = np.zeros_like(dt)
    return psid


def prop_coeff_ts_lin_0(dt: Array) -> Array:
    psi0 = np.zeros_like(dt)
    for n in range(2, MAX_TAYLOR_SERIES):
        psi0 += np.power(-dt, n) / factorial(n)
    psi0 /= dt
    return psi0


def prop_coeff_ts_lin_u(dt: Array) -> Array:
    psiu = np.zeros_like(dt)
    for n in range(2, MAX_TAYLOR_SERIES):
        psiu += np.power(-dt, n) * (n - 1) / factorial(n)
    psiu /= dt
    return psiu


def factorial(n: int) -> float:
    if n == 0:
        return 1
    return float(np.prod(np.arange(1, n + 1)).item())
