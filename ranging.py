"""
ranging.py
==========
Simulates one full-mesh ranging cycle (Sec. "TDMA Mesh Schedule": every
pair ranges once per cycle) and the attacker's range-falsification
mechanism (Sec. 3, "Mechanism of Range Falsification").

Attacker model implemented here (documented explicitly, since the paper's
Layer 3 defense depends on exactly *how* the lie is introduced):
the compromised drone's raw DS-TWR timestamp components are computed
honestly by its own firmware; the single point of tampering is the final
reported distance scalar it broadcasts, overwritten just before
transmission -- mirroring the original (unmodified) implementation's
`n_dist_cm[idx]` assignment. Consequently:
  - The falsified value is what every OTHER drone's pairwise EKF and Layer-2
    matrix observe for that link.
  - The attacker's own raw components remain honest, so an honest
    initiator's independent recomputation (Layer 3, Sec. 6.3) reproduces the
    TRUE distance and will mismatch the attacker's tampered broadcast.
This is one faithful, natural realization of "numeric fabrication"
(Sec. 3, Sec. 6.3) -- not the only conceivable one, but the one that keeps
the attacker's local raw timestamps genuinely honest.
"""
from __future__ import annotations
import numpy as np


class AttackConfig:
    def __init__(self, attacker: int, victims: list[int], direction: str,
                 magnitude_m: float, onset_cycle: int,
                 step_every: int = 10, step_size_m: float = 0.05):
        """
        Staircase bias profile: the reported distance jumps by `step_size_m`
        every `step_every` cycles until `magnitude_m` is reached, then holds.
        A *smooth* linear ramp was tried first and found to be perfectly
        absorbed by a constant-velocity EKF's velocity state (a linear range
        ramp is indistinguishable from the neighbor legitimately moving away
        at constant speed) -- verified empirically, not assumed. The
        staircase keeps each individual jump small enough to (usually) stay
        under the instantaneous chi-square gate while still giving Layer 1's
        CUSUM accumulator (Sec. 6.1) a persistent, consistently-signed
        residual to build evidence from.
        """
        assert direction in ("enlarge", "reduce")
        self.attacker = attacker
        self.victims = set(victims)
        self.direction = direction
        self.magnitude_m = magnitude_m
        self.onset_cycle = onset_cycle
        self.step_every = max(step_every, 1)
        self.step_size_m = step_size_m

    def bias_at(self, cycle_idx: int) -> float:
        """Signed bias (m) added to the honest distance at this cycle."""
        if cycle_idx < self.onset_cycle:
            return 0.0
        sign = 1.0 if self.direction == "enlarge" else -1.0
        n_steps = (cycle_idx - self.onset_cycle) // self.step_every
        magnitude = min(self.magnitude_m, n_steps * self.step_size_m)
        return sign * magnitude

    def affects(self, i: int, j: int) -> bool:
        """True if this exchange has the attacker reporting to a victim."""
        if self.attacker == i:
            return j in self.victims
        if self.attacker == j:
            return i in self.victims
        return False


class RangingCycle:
    """One full-mesh ranging cycle's results, for a given swarm ground truth."""

    def __init__(self, swarm, sigma_range: float, rng: np.random.Generator,
                 attack: AttackConfig | None, cycle_idx: int,
                 quantization_std: float = 0.01):
        n = swarm.n
        self.true_dist: dict[tuple[int, int], float] = {}
        self.reported_dist: dict[tuple[int, int], float] = {}
        # Layer-3 independent recomputation, only meaningful/used for pairs
        # where the attacker is the responder (Sec. 6.3).
        self.layer3_recompute: dict[tuple[int, int], float] = {}
        self.attacked_pairs: set[tuple[int, int]] = set()

        for i in range(n):
            for j in range(i + 1, n):
                d_true = float(np.linalg.norm(swarm.pos[j] - swarm.pos[i]))
                self.true_dist[(i, j)] = d_true

                noisy = d_true + rng.normal(0.0, sigma_range)
                reported = noisy
                if attack is not None and attack.affects(i, j):
                    bias = attack.bias_at(cycle_idx)
                    if bias != 0.0:
                        reported = noisy + bias
                        self.attacked_pairs.add((i, j))
                self.reported_dist[(i, j)] = reported

                # Honest raw-component recomputation an initiator could do:
                # it recomputes from the SAME underlying exchange (same raw
                # timestamps, same noise realization), not a fresh independent
                # measurement -- so absent an attack it agrees with `noisy`
                # up to only quantization-scale residual (Sec. 6.3), not the
                # full ranging-noise std used for `noisy` itself.
                if attack is not None and (i == attack.attacker or j == attack.attacker):
                    self.layer3_recompute[(i, j)] = noisy + rng.normal(0.0, quantization_std)

    def get(self, i: int, j: int) -> float:
        return self.reported_dist[(i, j)] if i < j else self.reported_dist[(j, i)]

    def get_layer3(self, i: int, j: int) -> float | None:
        key = (i, j) if i < j else (j, i)
        return self.layer3_recompute.get(key)
