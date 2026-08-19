"""
detection.py
============
Implements the paper's three detection layers (Sec. 6) plus the local
per-neighbor evidence combination (Sec. 6.4).

Simplification, stated explicitly: this simulation is fully connected (every
drone "overhears" every ranging exchange every cycle), consistent with the
connectivity assumption in Sec. 3 (System and Trust Assumptions). Under full
connectivity, every drone's local Layer-2 range matrix is identical, so we
compute Layer 2's leave-one-out attribution once per cycle rather than once
per drone -- the paper's design remains per-drone/local in general (a
sparser mesh would give each drone its own partial matrix), this is purely
a simulation-scale simplification, noted here rather than left implicit.

Layer 2 also uses plain (unweighted) classical MDS rather than the paper's
IRLS-robust variant (Sec. 6.2) -- with a single attacker and n as small as
used here, the leave-one-out signal is already clean without the extra
robust-reweighting machinery; the paper's full method would be needed at
larger swarm sizes or with multiple simultaneous falsified links.
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------
# Layer 1: per-link statistical gate (Sec. 6.1)
# ----------------------------------------------------------------------
class PerLinkGate:
    def __init__(self, chi2_thresh: float = 10.83, k: float = 0.3, h: float = 5.0):
        self.chi2_thresh = chi2_thresh
        self.k = k
        self.h = h
        self.C_plus = 0.0
        self.C_minus = 0.0
        self.instant_flag = False
        self.sequential_flag = False
        self.direction: str | None = None

    def update(self, nis: float, standardized_innovation: float) -> None:
        self.instant_flag = nis > self.chi2_thresh
        self.C_plus = max(0.0, self.C_plus + standardized_innovation - self.k)
        self.C_minus = max(0.0, self.C_minus - standardized_innovation - self.k)
        self.sequential_flag = (self.C_plus > self.h) or (self.C_minus > self.h)
        if self.C_plus > self.h:
            self.direction = "enlarge"
        elif self.C_minus > self.h:
            self.direction = "reduce"


# ----------------------------------------------------------------------
# Layer 2: distributed geometric consistency (Sec. 6.2)
# ----------------------------------------------------------------------
def classical_mds_2d(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:2]
    top_vals = np.clip(eigvals[order], 0, None)
    L = np.diag(np.sqrt(top_vals))
    return eigvecs[:, order] @ L


def _stress(D: np.ndarray, P: np.ndarray) -> float:
    n = P.shape[0]
    total = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            dhat = np.linalg.norm(P[a] - P[b])
            r = D[a, b] - dhat
            total += r * r
    return total


def leave_one_out_scores(D: np.ndarray) -> np.ndarray:
    """Delta_m per Algorithm 2 (Sec. 6.2): stress reduction from removing node m."""
    n = D.shape[0]
    P_full = classical_mds_2d(D)
    stress_full = _stress(D, P_full)
    deltas = np.zeros(n)
    for m in range(n):
        idx = [k for k in range(n) if k != m]
        D_sub = D[np.ix_(idx, idx)]
        if D_sub.shape[0] < 4:
            continue  # need >=4 nodes for a meaningful embedding (Sec. 6.2)
        P_sub = classical_mds_2d(D_sub)
        stress_sub = _stress(D_sub, P_sub)
        deltas[m] = stress_full - stress_sub
    return deltas


# ----------------------------------------------------------------------
# Layer 3: dual-computation cross-check (Sec. 6.3)
# ----------------------------------------------------------------------
def layer3_check(reported: float, recomputed: float, delta: float = 0.05) -> bool:
    """|d_A - d_B| > delta => flag the responder as the attacker for this exchange."""
    return abs(reported - recomputed) > delta


# ----------------------------------------------------------------------
# Sec. 6.4: local per-neighbor evidence combination, no broadcast/voting
# ----------------------------------------------------------------------
class EvidenceTracker:
    """
    Maintains, for a single observing drone, a running evidence score
    E_i(j) per neighbor j, combining Layer 1/2/3 signals exactly per the
    weighted sum in Sec. 6.4 (lambda_3 >> lambda_2 >> lambda_1).
    """

    def __init__(self, n_drones: int, lambda1: float = 0.4, lambda2: float = 0.05,
                 lambda3: float = 6.0, theta: float = 3.0, sustain_cycles: int = 2):
        self.n = n_drones
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.theta = theta
        self.sustain_cycles = sustain_cycles
        self.E = np.zeros(n_drones)
        self._above_streak = np.zeros(n_drones, dtype=int)
        self.declared_compromised: set[int] = set()

    def accumulate(self, j: int, layer1_alarm: bool, delta_j: float, layer3_flag: bool) -> None:
        if j in self.declared_compromised:
            return  # already declared; no need to keep piling up evidence
        self.E[j] += (self.lambda1 * float(layer1_alarm)
                      + self.lambda2 * max(delta_j, 0.0)
                      + self.lambda3 * float(layer3_flag))
        if self.E[j] > self.theta:
            self._above_streak[j] += 1
        else:
            self._above_streak[j] = 0
        if self._above_streak[j] >= self.sustain_cycles:
            self.declared_compromised.add(j)
