"""
model.py
========
Ground-truth swarm dynamics, formation guidance, and the collision-safety
filter, mirroring the pipeline in the paper's System Overview
(Sec. 5.2, Fig. "pipeline"):

    range measurement -> pairwise EKF -> formation guidance
        -> collision-safety filter -> commanded acceleration

Every drone's *controller* only ever sees its own EKF's estimated relative
states (never ground truth). Ground-truth positions/velocities are kept
separately here purely so the simulation can score outcomes (true minimum
separation, collisions) -- exactly the asymmetry the paper's threat model
relies on: the swarm's shared picture is constructed entirely from ranging,
with no external ground truth to check it against.
"""
from __future__ import annotations
import numpy as np


class SwarmState:
    """Ground-truth kinematic state of all N drones (double integrator, 2D)."""

    def __init__(self, positions: np.ndarray, velocities: np.ndarray | None = None):
        self.n = positions.shape[0]
        self.pos = positions.astype(float).copy()          # (n, 2)
        self.vel = np.zeros_like(self.pos) if velocities is None else velocities.copy()

    def true_relative(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """True (p, v) of drone j relative to drone i."""
        return self.pos[j] - self.pos[i], self.vel[j] - self.vel[i]

    def min_pairwise_distance(self) -> float:
        d = np.inf
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = min(d, np.linalg.norm(self.pos[j] - self.pos[i]))
        return d

    def step(self, accel: np.ndarray, dt: float) -> None:
        """
        Integrate one control cycle under commanded accel (n,2).

        No artificial drag/damping here: earlier versions of this
        simulation added light drag (-k*v) to stop the swarm's centroid
        from drifting unboundedly, since pure peer-to-peer relative
        formation control has no absolute anchor. That is no longer
        needed now that guidance targets a leader moving along a real
        route (see simulate.py) -- an actual anchor, not a numerical
        patch -- and drag would actively fight against legitimately
        matching the leader's nonzero velocity.
        """
        self.vel += accel * dt
        self.pos += self.vel * dt


class FormationGuidance:
    """
    Simple virtual spring-damper formation controller. Each drone tries to
    hold its *nominal* relative offset to every other drone, using only its
    own pairwise EKF's estimated relative position/velocity for that
    neighbor -- never ground truth (Sec. 5.2).
    """

    def __init__(self, nominal_offsets: dict[tuple[int, int], np.ndarray],
                 kp: float = 2.0, kd: float = 2.5, accel_limit: float = 4.0):
        self.nominal_offsets = nominal_offsets  # (i, j) -> desired p_j_rel_i
        self.kp = kp
        self.kd = kd
        self.accel_limit = accel_limit

    def desired_accel(self, drone_i: int, neighbors: list[int],
                       estimates: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Sum spring-damper contributions from every neighbor's *estimated*
        relative position/velocity toward the nominal formation offset.
        `estimates[j] = (p_hat, v_hat)` = j relative to drone_i, from the
        pairwise EKF drone_i maintains for neighbor j.
        """
        a = np.zeros(2)
        for j in neighbors:
            p_hat, v_hat = estimates[j]
            p_des = self.nominal_offsets[(drone_i, j)]
            a += self.kp * (p_hat - p_des) + self.kd * v_hat
        norm = np.linalg.norm(a)
        if norm > self.accel_limit:
            a = a * (self.accel_limit / norm)
        return a


class CollisionSafetyFilter:
    """
    Pairwise control-barrier-function safety filter, matching the paper's
    barrier constraint p^T a_rel >= c (Sec. 5.2, Fig. "pipeline").

    For each neighbor j, using drone_i's own EKF estimate (p_hat, v_hat) of
    j relative to i, define h = ||p_hat||^2 - d_min^2 and require
        d/dt[ d/dt h ] + 2*alpha*(d/dt h) + alpha^2*h >= 0
    which reduces to a halfspace constraint p_hat^T a_rel >= c on the
    *relative* acceleration a_rel = a_j - a_i. Since a drone only controls
    its own acceleration, we conservatively require p_hat^T(-a_i) >= c/2,
    i.e. treat half the necessary correction as this drone's share -- a
    simplification appropriate for a demonstration filter, not a claim of
    optimality.
    """

    def __init__(self, d_min: float = 1.0, alpha: float = 2.0):
        self.d_min = d_min
        self.alpha = alpha

    def filter(self, a_desired: np.ndarray, neighbors: list[int],
               estimates: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        a = a_desired.copy()
        for j in neighbors:
            p_hat, v_hat = estimates[j]
            dist2 = float(p_hat @ p_hat)
            h = dist2 - self.d_min ** 2
            hdot = 2.0 * float(p_hat @ v_hat)
            # required: p^T a_rel >= c  (a_rel = a_j - a_i; assume a_j ~ 0 contribution here)
            c = -(self.alpha ** 2) * h - 2 * self.alpha * hdot - 2.0 * float(v_hat @ v_hat)
            c_share = c / 2.0
            lhs = float(p_hat @ (-a))
            if lhs < c_share:
                # minimal correction along p_hat; floor the denominator well above
                # zero so a near-zero separation cannot produce an arbitrarily
                # large single-cycle correction (a numerical stability choice
                # for this demonstration filter, not part of the paper's design).
                pn2 = max(dist2, 0.25 * self.d_min ** 2)
                correction = ((c_share - lhs) / pn2) * p_hat
                corr_norm = np.linalg.norm(correction)
                if corr_norm > 4.0:
                    correction = correction * (4.0 / corr_norm)
                a = a - correction
        norm = np.linalg.norm(a)
        if norm > 5.0:
            a = a * (5.0 / norm)
        return a
