"""
ekf.py
======
Pairwise relative-state EKF, matching the paper's Background Sec. "Relative-
State Estimation": state x = [rx, ry, vx, vy] (position/velocity of
neighbor j relative to drone i), constant-velocity dynamics driven by a
relative-acceleration control input u = a_j - a_i, and a scalar range
measurement model h(x) = ||[rx, ry]||.

Every update also records the innovation, its variance, and the resulting
NIS -- exactly the "diagnostic byproduct" the paper's Layer 1 (Sec. 6.1)
later turns into an active gate.
"""
from __future__ import annotations
import numpy as np


class RelativeRangeEKF:
    def __init__(self, initial_p: np.ndarray, initial_v: np.ndarray,
                 process_noise_q: float = 0.15, range_noise_std: float = 0.08,
                 init_pos_var: float = 0.05, init_vel_var: float = 0.05):
        self.x = np.array([initial_p[0], initial_p[1], initial_v[0], initial_v[1]], dtype=float)
        self.P = np.diag([init_pos_var, init_pos_var, init_vel_var, init_vel_var])
        self.q = process_noise_q
        self.R = range_noise_std ** 2
        self.min_range = 1e-3
        # last-update diagnostics (Sec. 2.3 / Sec. 6.1)
        self.last_innovation = 0.0
        self.last_innovation_var = 1.0
        self.last_nis = 0.0

    def predict(self, u_rel_accel: np.ndarray, dt: float) -> None:
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        G = np.array([[0.5 * dt ** 2, 0],
                      [0, 0.5 * dt ** 2],
                      [dt, 0],
                      [0, dt]])
        q = self.q
        Q = q * np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2],
        ])
        self.x = F @ self.x + G @ u_rel_accel
        self.P = F @ self.P @ F.T + Q

    def update(self, z_range: float) -> None:
        """Range-only EKF update. Records innovation/NIS as a byproduct."""
        r = np.array([self.x[0], self.x[1]])
        r_norm = np.linalg.norm(r)
        if r_norm < self.min_range:
            return  # singular Jacobian, skip (Sec. 2.3)
        h = r_norm
        H = np.array([r[0] / r_norm, r[1] / r_norm, 0.0, 0.0])
        y = z_range - h
        S = float(H @ self.P @ H.T + self.R)
        K = (self.P @ H.T) / S
        self.x = self.x + K * y
        self.P = (np.eye(4) - np.outer(K, H)) @ self.P

        self.last_innovation = y
        self.last_innovation_var = S
        self.last_nis = (y * y) / S

    @property
    def rel_pos(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def rel_vel(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def standardized_innovation(self) -> float:
        """nu_t / sqrt(S_t), the signed quantity Layer 1's CUSUM accumulates."""
        return self.last_innovation / np.sqrt(max(self.last_innovation_var, 1e-9))
