# Swarm Range-Falsification Simulation

A Python simulation of the system architecture from the paper
*"Distributed Detection and Attribution of Range-Falsification Attacks in
GNSS-Denied UWB Drone Swarms"* — built to answer two questions concretely:

1. **Without any detection/mitigation, what does a range-falsification
   attack actually do to the swarm?** (Sec. 3 threat model, Sec. 5.2 pipeline)
2. **How does the swarm's proposed detection architecture catch and
   attribute the attack?** (Sec. 6, all three layers + the Sec. 6.4
   local evidence combination)

## Running it

```
pip install -r requirements.txt
python3 simulate.py
```

Results are deterministic: `RNG_SEED = 7` in `simulate.py` reproduces exactly
the numbers in the table below. Produces `outputs/part1_*.png`,
`outputs/part2_*.png`, and `outputs/summary.txt`.

## What's simulated

- **6 drones** in a well-spread 2D formation (mirrors the paper's Fig. 1
  teaser: D1..D6), ranging against every other drone once per cycle
  (Sec. "TDMA Mesh Schedule").
- **D2 is the attacker, D5 is the victim** — the same falsified link shown
  in the paper's teaser figure.
- Every drone runs the **real pipeline** from Sec. 5.2: range measurement
  → pairwise EKF (Sec. 2.3) → formation guidance (spring-damper toward a
  nominal formation) → collision-safety filter (CBF-style, `p^T a_rel ≥ c`).
- **No mitigation is applied to a flagged link** — this is a deliberate,
  direct reflection of the paper's explicit scope (Sec. 3): the paper's
  contribution stops at detection and attribution, not response. Part 1
  therefore shows what an *undefended* swarm actually suffers.
- Detection (Sec. 6) runs in parallel, observing the same ranging stream.
  It does not alter the physics — this lets Part 1 (physical outcome) and
  Part 2 (detection outcome) be compared on the same timeline.

## The attack model, explicitly

The compromised drone's raw DS-TWR timestamp components are computed
honestly; the single point of tampering is the **final reported distance
scalar**, overwritten just before broadcast — mirroring the original
(unmodified) firmware's `n_dist_cm[idx]` assignment. This is one faithful
realization of "numeric fabrication" (Sec. 3, Sec. 6.3), chosen because it
keeps the attacker's raw timestamps genuinely honest, which is exactly what
makes Layer 3's cross-check unambiguous when it fires.

The bias is applied as a **staircase** (small discrete jumps, not a smooth
ramp). This was not the first thing tried — a smooth linear ramp was tested
first and found, by direct inspection of the EKF's internal state, to be
**perfectly absorbed by the constant-velocity filter's velocity state**: a
linearly-growing range is mathematically indistinguishable from the
neighbor honestly moving away at constant speed. That's a genuine
limitation of innovation-based detection against a sophisticated attacker,
not a simulation bug, and it's why the staircase profile is used instead —
consistent with the paper's own framing of the gradual attack as one that
"remains within the swarm's ordinary noise floor on any single exchange."

## Key results (seed=7, default parameters)

| Event | Time |
|---|---|
| Attack onset | t = 9.0s |
| **Collision threshold crossed (undefended)** | **t = 19.35s** |
| D5 (victim, has direct Layer-3 access) declares D2 compromised | **t = 10.80s** |
| D6 declares D2 compromised (Layer 1 + Layer 2 only) | t = 34.20s |
| D1, D3, D4 declare D2 compromised (Layer 1 + Layer 2 only) | t = 36.75s |

The headline result: **the victim detects and correctly attributes the
attack ~8.5 seconds before physical collision occurs.** Drones without
direct Layer-3 access (i.e., that never initiated an exchange against the
attacker themselves) take substantially longer, relying on Layer 1's CUSUM
and Layer 2's geometric corroboration alone — a direct, visible
illustration of why the paper layers multiple mechanisms rather than
relying on any one of them.

## Known simplifications (stated plainly, not hidden)

- **Full connectivity assumed**: every drone overhears every broadcast, so
  Layer 2's matrix is centralized (computed once/cycle) rather than
  genuinely per-drone/partial. The paper's design is local-by-construction;
  this only matters differently at lower connectivity (Sec. 3 assumption).
- **Unweighted (non-robust) MDS** for Layer 2, not the paper's IRLS-robust
  variant (Sec. 6.2) — visible in `part2_layer2.png` as some honest nodes'
  Δ scores drifting upward too (the "leverage effect" the paper's robust
  fitting is specifically designed to fix).
- **CUSUM direction labels (C⁺/C⁻) aren't reliably meaningful in this
  specific demo** — the staircase attack's discrete jumps interact with the
  EKF's velocity state in a way that can make the "wrong" accumulator
  dominate. That *an* anomaly is flagged is reliable; which accumulator
  fires isn't a trustworthy direction indicator here.
- A light aerodynamic-drag term (`model.py`, `SwarmState.step`) was added
  to prevent the whole formation's centroid from drifting unboundedly — a
  known property of purely relative-only formation control with no
  absolute anchor, unrelated to the attack itself.
- Part 1 plots are truncated shortly after collision, since continuing to
  integrate the simplified controller well past a real collision produces
  unbounded drift that isn't part of the phenomenon being demonstrated.
  Part 2 (detection) plots use the full simulated range.

## File overview

- `model.py` — ground-truth dynamics, formation guidance, collision-safety filter
- `ekf.py` — pairwise relative-state EKF (Sec. 2.3)
- `ranging.py` — full-mesh ranging + the attack model
- `detection.py` — Layers 1/2/3 + Sec. 6.4 local evidence combination
- `simulate.py` — orchestration, plotting, `summary.txt`
