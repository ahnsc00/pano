#!/usr/bin/env python3
"""Paper-faithful CRTM core from Xue et al. (NeurIPS 2023 Algorithm 1).

The implementation follows the supplement equations for Confidence Region with
Truncated Mean (CRTM): uniform reward zeroing, ONS estimator update, and
V-metric projection onto the Euclidean parameter ball.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


@dataclass(frozen=True)
class CRTMSelection:
    """Diagnostic payload for one CRTM arm-selection step."""

    action: int
    optimistic_values: np.ndarray
    predictions: np.ndarray
    widths: np.ndarray


@dataclass(frozen=True)
class CRTMUpdate:
    """Diagnostic payload for one CRTM update step."""

    round: int
    reward: float
    truncated_reward: float
    was_truncated: bool
    leverage: float
    gradient: np.ndarray
    theta_unprojected: np.ndarray


def _log_positive(value: float, name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return float(np.log(value))


def theory_gamma(
    dimension: int,
    delta: float,
    epsilon: float,
    moment_bound_u: float,
    kappa: float,
    parameter_radius_S: float,
    mean_bound_U: float,
    horizon: int,
    lambda_: float | None = None,
) -> float:
    """Exact CRTM confidence radius gamma from Algorithm 1 / Theorem 1.

    Source equation:
    224*u^(2/(1+eps))*ln(4T/delta)^(2eps/(1+eps))*T^((1-eps)/(1+eps))
    * (4d/kappa)*ln(1+kappa*T/(2*lambda*d))
    + 2*lambda*S^2
    + (48*U^2*d/kappa)*ln(1+kappa*T/(2*lambda*d)).
    """

    d, T = int(dimension), int(horizon)
    if d <= 0:
        raise ValueError("dimension must be positive")
    if T <= 0:
        raise ValueError("horizon must be positive")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if not (0.0 < float(epsilon) <= 1.0):
        raise ValueError("epsilon must be in (0, 1]")
    u = float(moment_bound_u)
    k = float(kappa)
    S = float(parameter_radius_S)
    U = float(mean_bound_U)
    if u <= 0.0 or k <= 0.0 or S < 0.0 or U < 0.0:
        raise ValueError("moment_bound_u,kappa must be positive; radii/bounds non-negative")
    lam = float(max(1.0, k / 2.0) if lambda_ is None else lambda_)
    if lam <= 0.0:
        raise ValueError("lambda_ must be positive")

    log_conf = _log_positive(4.0 * T / float(delta), "4T/delta")
    log_det = _log_positive(1.0 + k * T / (2.0 * lam * d), "1+kT/(2lambda d)")
    heavy_tail = (
        224.0
        * u ** (2.0 / (1.0 + epsilon))
        * log_conf ** (2.0 * epsilon / (1.0 + epsilon))
        * T ** ((1.0 - epsilon) / (1.0 + epsilon))
        * (4.0 * d / k)
        * log_det
    )
    init_radius = 2.0 * lam * S * S
    mean_radius = (48.0 * U * U * d / k) * log_det
    return float(heavy_tail + init_radius + mean_radius)


def experimental_gamma(
    dimension: int,
    delta: float,
    epsilon: float,
    horizon: int,
    lambda_: float,
    confidence_scale: float = 1.0,
) -> float:
    """Xue-style experimental confidence radius.

    Uses c*d*log(4T/delta)^(2eps/(1+eps))*log(T/(d*lambda)+1)
    * T^((1-eps)/(1+eps)), where c is ``confidence_scale``.
    """

    d, T = int(dimension), int(horizon)
    if d <= 0 or T <= 0:
        raise ValueError("dimension and horizon must be positive")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if not (0.0 < float(epsilon) <= 1.0):
        raise ValueError("epsilon must be in (0, 1]")
    lam = float(lambda_)
    c = float(confidence_scale)
    if lam <= 0.0 or c < 0.0:
        raise ValueError("lambda_ must be positive and confidence_scale non-negative")
    return float(
        c
        * d
        * _log_positive(4.0 * T / float(delta), "4T/delta")
        ** (2.0 * epsilon / (1.0 + epsilon))
        * _log_positive(T / (d * lam) + 1.0, "T/(d lambda)+1")
        * T ** ((1.0 - epsilon) / (1.0 + epsilon))
    )


def truncation_threshold(
    dimension: int,
    delta: float,
    epsilon: float,
    moment_bound_u: float,
    kappa: float,
    horizon: int,
    lambda_: float | None = None,
) -> float:
    """Exact Algorithm 1 uniform CRTM truncation criterion Γ."""

    d, T = int(dimension), int(horizon)
    if d <= 0 or T <= 0:
        raise ValueError("dimension and horizon must be positive")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if not (0.0 < float(epsilon) <= 1.0):
        raise ValueError("epsilon must be in (0, 1]")
    u = float(moment_bound_u)
    k = float(kappa)
    if u <= 0.0 or k <= 0.0:
        raise ValueError("moment_bound_u and kappa must be positive")
    lam = float(max(1.0, k / 2.0) if lambda_ is None else lambda_)
    if lam <= 0.0:
        raise ValueError("lambda_ must be positive")
    return float(
        2.0
        * (u / _log_positive(4.0 * T / float(delta), "4T/delta"))
        ** (1.0 / (1.0 + epsilon))
        * (d * _log_positive(1.0 + k * T / (2.0 * lam * d), "1+kT/(2lambda d)") / k)
        ** 0.5
        * T ** ((1.0 - epsilon) / (2.0 * (1.0 + epsilon)))
    )


def project_l2_ball_v_metric(point: ArrayLike, V: np.ndarray, radius: float) -> np.ndarray:
    """Project ``point`` onto {theta: ||theta||_2 <= radius} in V norm.

    Solves argmin_{||theta||_2<=S} 0.5*(theta-point)^T V (theta-point).
    For points outside the ball, the KKT solution is found by bisection in the
    Lagrange multiplier after diagonalizing V.
    """

    z = np.asarray(point, dtype=float)
    V = np.asarray(V, dtype=float)
    S = float(radius)
    if z.ndim != 1:
        raise ValueError("point must be a vector")
    if V.shape != (z.size, z.size):
        raise ValueError("V shape must match point dimension")
    if S < 0.0:
        raise ValueError("radius must be non-negative")
    if S == 0.0:
        return np.zeros_like(z)
    if np.linalg.norm(z) <= S + 1e-14:
        return z.copy()

    V_sym = 0.5 * (V + V.T)
    evals, Q = np.linalg.eigh(V_sym)
    if np.min(evals) <= 0.0:
        raise ValueError("V must be positive definite")
    a = Q.T @ z

    def norm_at(rho: float) -> float:
        factors = evals / (evals + rho)
        return float(np.linalg.norm(factors * a))

    lo, hi = 0.0, 1.0
    while norm_at(hi) > S:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if norm_at(mid) > S:
            lo = mid
        else:
            hi = mid
    factors = evals / (evals + hi)
    projected = Q @ (factors * a)
    return np.asarray(projected, dtype=float)


class XueCRTM:
    """CRTM policy using Xue et al. Algorithm 1 equations.

    Parameters mirror the paper inputs. ``link`` is the GLM mean function μ.
    Selection uses x^T theta_hat + sqrt(gamma)*||x||_{V^{-1}}.
    Updates only consume the chosen context/reward pair.
    """

    def __init__(
        self,
        dimension: int,
        link: Callable[[np.ndarray | float], np.ndarray | float],
        delta: float,
        epsilon: float,
        moment_bound_u: float,
        kappa: float,
        parameter_radius_S: float,
        mean_bound_U: float,
        horizon: int,
        confidence_mode: str = "theory",
        confidence_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        self.link = link
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        self.moment_bound_u = float(moment_bound_u)
        self.kappa = float(kappa)
        self.parameter_radius_S = float(parameter_radius_S)
        self.mean_bound_U = float(mean_bound_U)
        self.horizon = int(horizon)
        self.confidence_mode = str(confidence_mode)
        self.confidence_scale = float(confidence_scale)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.lambda_ = max(1.0, self.kappa / 2.0)
        self.V = self.lambda_ * np.eye(self.dimension, dtype=float)
        self.V_inv = np.linalg.inv(self.V)
        self.theta = np.zeros(self.dimension, dtype=float)
        self.round = 1
        self.truncated_count = 0
        self.last_selection: CRTMSelection | None = None
        self.last_update: CRTMUpdate | None = None

        self.truncation_threshold = truncation_threshold(
            self.dimension,
            self.delta,
            self.epsilon,
            self.moment_bound_u,
            self.kappa,
            self.horizon,
            self.lambda_,
        )
        if self.confidence_mode == "theory":
            self.gamma = theory_gamma(
                self.dimension,
                self.delta,
                self.epsilon,
                self.moment_bound_u,
                self.kappa,
                self.parameter_radius_S,
                self.mean_bound_U,
                self.horizon,
                self.lambda_,
            )
        elif self.confidence_mode == "experimental":
            self.gamma = experimental_gamma(
                self.dimension,
                self.delta,
                self.epsilon,
                self.horizon,
                self.lambda_,
                self.confidence_scale,
            )
        else:
            raise ValueError("confidence_mode must be 'theory' or 'experimental'")

    def select(self, contexts: np.ndarray) -> int:
        """Return the CRTM optimistic arm index for a context matrix."""

        X = np.asarray(contexts, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError(f"contexts must have shape (n,{self.dimension})")
        predictions = X @ self.theta
        quad = np.einsum("ij,jk,ik->i", X, self.V_inv, X)
        widths = np.sqrt(self.gamma) * np.sqrt(np.maximum(quad, 0.0))
        optimistic = predictions + widths
        best = np.flatnonzero(optimistic == np.max(optimistic))
        action = int(self.rng.choice(best))
        self.last_selection = CRTMSelection(
            action=action,
            optimistic_values=optimistic.copy(),
            predictions=predictions.copy(),
            widths=widths.copy(),
        )
        return action

    def update(self, context: ArrayLike, reward: float) -> CRTMUpdate:
        """Apply Algorithm 1 update using the chosen context and observed reward."""

        x = np.asarray(context, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"context must have shape ({self.dimension},)")
        y = float(reward)
        leverage = float(np.sqrt(max(float(x @ self.V_inv @ x), 0.0)))
        keep = leverage * abs(y) <= self.truncation_threshold
        y_tilde = y if keep else 0.0
        if not keep:
            self.truncated_count += 1

        mu = float(np.asarray(self.link(float(x @ self.theta))))
        gradient = (-y_tilde + mu) * x

        self.V = self.V + (self.kappa / 2.0) * np.outer(x, x)
        self.V = 0.5 * (self.V + self.V.T)
        self.V_inv = np.linalg.inv(self.V)
        self.V_inv = 0.5 * (self.V_inv + self.V_inv.T)

        theta_unprojected = self.theta - self.V_inv @ gradient
        self.theta = project_l2_ball_v_metric(
            theta_unprojected, self.V, self.parameter_radius_S
        )
        self.round += 1
        self.last_update = CRTMUpdate(
            round=self.round - 1,
            reward=y,
            truncated_reward=float(y_tilde),
            was_truncated=not keep,
            leverage=leverage,
            gradient=gradient.copy(),
            theta_unprojected=theta_unprojected.copy(),
        )
        return self.last_update
