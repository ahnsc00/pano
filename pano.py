import math

import numpy as np


def _project_ball(point: np.ndarray, matrix: np.ndarray, radius: float) -> np.ndarray:
    """행렬이 정하는 거리에서 유클리드 공에 투영한다."""
    point = np.asarray(point, dtype=float)
    if float(np.linalg.norm(point)) <= radius:
        return point.copy()

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, basis = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    coordinates = basis.T @ point

    def norm_at(multiplier: float) -> float:
        weights = eigenvalues / (eigenvalues + multiplier)
        return float(np.sqrt(np.sum((weights * coordinates) ** 2)))

    low, high = 0.0, 1.0
    while norm_at(high) > radius:
        high *= 2.0
    for _ in range(90):
        middle = 0.5 * (low + high)
        if norm_at(middle) > radius:
            low = middle
        else:
            high = middle
    weights = eigenvalues / (eigenvalues + high)
    return basis @ (weights * coordinates)


def _online_schedules(dimension, radius, score_low, score_high,
                      theta_bound, delta, ridge_scale, beta_scale):
    b = -score_high
    diameter = score_high - score_low
    gamma = 4.0 * diameter * diameter / (b * b) + 2.0 * (math.e - 1.0) * max(
        3.0 + 4.0 * diameter * diameter / (b * b),
        2.0 + 2.0 * diameter / b,
    )

    def steps():
        ridge_max = 1.0 / (b * b)
        initial_ridge = None
        start = 0
        while True:
            time = np.arange(start, start + dimension + 1, dtype=float)
            effective_time = np.maximum(time, 1.0)
            tail = (4.0 / b) * np.log(2.0 * effective_time * effective_time / delta)
            gradient_bound = 1.0 / b + tail
            regularizer = (
                np.square(gradient_bound)
                * np.log(2.0 * dimension * effective_time * effective_time / delta)
                + effective_time * np.exp(-b * tail)
                * (np.square(tail) + 4.0 * tail / b + 5.0 / (b * b))
            )
            ridge_max = np.maximum(ridge_max, np.max(2.0 * regularizer))
            scaled_ridge = ridge_max * ridge_scale
            ridge = float(scaled_ridge)
            if initial_ridge is None:
                initial_ridge = ridge
            remainder = 2.0 * diameter * diameter * np.square(tail) * np.log(
                2.0 * effective_time * effective_time / delta
            )
            for index in range(dimension):
                t = start + index
                bound = float(gradient_bound[index])
                correction = float(remainder[index])
                online_bound = (
                    4.0 * ridge * theta_bound * theta_bound / gamma
                    + gamma / 4.0 * dimension * math.log1p(
                        max(float(t), 1.0) * bound * bound
                        / (dimension * max(initial_ridge, 1e-300))
                    )
                    + correction / gamma
                )
                confidence = math.sqrt(
                    2.0 * gamma * online_bound
                    + 8.0 * gamma * (b + diameter) / b * math.log(1.0 / delta)
                    + correction
                )
                beta_base = np.float64(confidence + math.sqrt(ridge) * radius)
                yield scaled_ridge, beta_base * beta_scale
            start += dimension

    return gamma / 2.0, steps()


class PANO:
    """로그보상으로 추정치를 갱신하고 score-UCB로 후보를 선택한다."""

    def __init__(self, theta_init, radius: float, score_low: float, score_high: float,
                 *, delta: float = 0.01, ridge_scale: float = 1e-8,
                 beta_scale: float = 0.01, seed: int = 0):
        center = np.asarray(theta_init, dtype=float)
        if center.ndim != 1 or center.size == 0 or not np.isfinite(center).all():
            raise ValueError("theta_init must be a nonempty finite vector")
        parameters = (radius, score_low, score_high, delta, ridge_scale, beta_scale)
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("parameters must be finite")
        if radius <= 0 or ridge_scale <= 0 or beta_scale <= 0:
            raise ValueError("radius, ridge_scale and beta_scale must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0, 1)")
        if not score_low < score_high < -1.0:
            raise ValueError("score bounds must satisfy score_low < score_high < -1")

        self.theta_init = center.copy()
        self.radius = float(radius)
        self.score_low = float(score_low)
        self.score_high = float(score_high)
        self.dim = len(center)
        theta_bound = float(np.linalg.norm(center) + self.radius)
        self.eta, self._schedule = _online_schedules(
            self.dim, self.radius, self.score_low, self.score_high,
            theta_bound, delta, ridge_scale, beta_scale,
        )
        self.rng = np.random.default_rng(100_000 + int(seed))
        self.theta_tilde = center.copy()
        self.theta_hat = center.copy()
        self.H_bar = np.zeros((self.dim, self.dim), dtype=float)
        self.S = np.zeros(self.dim, dtype=float)
        self.updates = 0
        self.last_widths = None
        self.last_values = None
        self._refresh_information()

    def _refresh_information(self):
        ridge, beta = next(self._schedule)
        self.epsilon, self._current_beta = float(ridge), float(beta)
        self.H = self.H_bar + self.epsilon * np.eye(self.dim)
        self.H = 0.5 * (self.H + self.H.T)
        self.H_inv = np.linalg.inv(self.H)
        self.H_inv = 0.5 * (self.H_inv + self.H_inv.T)

    def _check_contexts(self, contexts, *, candidates: bool):
        values = np.asarray(contexts, dtype=float)
        if candidates:
            valid_shape = values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == self.dim
        else:
            valid_shape = values.shape == (self.dim,)
        if not valid_shape or not np.isfinite(values).all():
            raise ValueError("context shape or values are invalid")
        norms = np.linalg.norm(values, axis=1) if candidates else np.linalg.norm(values)
        if np.any(norms > 1.0 + 1e-8):
            raise ValueError("context norm must not exceed one")
        return values

    def select(self, contexts) -> int:
        """후보 행렬의 각 행을 평가하고 선택한 행의 인덱스를 반환한다."""
        candidates = self._check_contexts(contexts, candidates=True)
        scores = candidates @ self.theta_hat
        if float(np.max(scores)) > self.score_high + 1e-8 or float(np.max(scores)) >= -1.0:
            raise ValueError("candidate scores left the finite-mean domain")
        solved = candidates @ self.H_inv.T
        squared_width = np.maximum(np.sum(candidates * solved, axis=1), 0.0)
        widths = self._current_beta * np.sqrt(np.maximum(squared_width, 1e-14))
        values = scores + widths
        self.last_widths = widths.copy()
        self.last_values = np.asarray(values, dtype=float).copy()
        best = np.flatnonzero(values == np.max(values))
        return int(self.rng.choice(best))

    def update(self, context, log_reward: float):
        """선택한 context와 log(보상/최소 보상)으로 한 차례 갱신한다."""
        x = self._check_contexts(context, candidates=False)
        if not math.isfinite(log_reward) or log_reward < 0:
            raise ValueError("log_reward must be finite and nonnegative")
        previous = self.theta_tilde.copy()
        score = float(x @ previous)
        if score >= -1.0:
            raise ValueError("observation context left the finite-mean domain")
        gradient = (-1.0 / score - float(log_reward)) * x

        self.S = self.S + gradient * float(gradient @ previous)
        self.H_bar = self.H_bar + np.outer(gradient, gradient)
        self.H_bar = 0.5 * (self.H_bar + self.H_bar.T)
        self.updates += 1
        self._refresh_information()

        rhs = self.S + self.epsilon * self.theta_init
        self.theta_hat = self.theta_init + _project_ball(
            self.H_inv @ rhs - self.theta_init, self.H, self.radius
        )
        self.theta_tilde = self.theta_init + _project_ball(
            previous - self.eta * (self.H_inv @ gradient) - self.theta_init,
            self.H, self.radius,
        )
