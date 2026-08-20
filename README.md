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
python3 animate.py     # optional: animated visualization (see below)
```

Results are deterministic: `RNG_SEED = 7` in `simulate.py` reproduces exactly
the numbers in the table below. Produces `outputs/part1_*.png`,
`outputs/part2_*.png`, and `outputs/summary.txt`.

`animate.py` re-runs the same deterministic scenario and renders it as an
animation (`outputs/swarm_animation.mp4` and `.gif`): drones flying in
formation, the falsified link highlighted in red once the attack starts, a
live green checkmark next to each drone the moment it independently
declares D2 compromised, and a synchronized safety-margin panel showing the
true minimum separation and collision threshold. The MP4 requires ffmpeg on
your PATH; the GIF only needs Pillow (already in `requirements.txt`) and is
produced either way.

![Swarm range-falsification attack: detection and attribution](outputs/swarm_animation.gif)

`compare_n.py` runs the same scenario at N=6 and N=15 side by side (see
"Scaling to more drones" below) into `outputs/n06/` and `outputs/n15/`.

## Scaling to more drones

The formation generalizes to any N via `generate_formation()` (a
sunflower/Fibonacci disk layout — even 2D spread, no accidental
collinearity, for any N). The attacker and its victim (nearest neighbor)
are chosen automatically.

**A real, non-obvious finding from doing this**: with a *fixed* attack
rate, N=15 shows dramatically *less* physical danger than N=6 — no
collision at all, versus a clean collision at N=6. This is not a bug. Each
drone's formation guidance (Sec. 5.2) sums a spring correction from every
other drone; the one falsified link is 1-in-5 of that sum at N=6 but only
1-in-14 at N=15, so a larger, more connected swarm is structurally more
robust to a single liar purely as a byproduct of full-mesh formation
control — nothing to do with detection. `compare_n.py` re-tunes the
attack's ramp rate per N (found by direct sweep, not guessed) to restore
comparable danger at both sizes for an apples-to-apples comparison:

| N | attack step size | Collision? | Time to collision | Victim's detection time |
|---|---|---|---|---|
| 6 | 0.5 m / 10 cycles | Yes | t = 55.80s | t = 10.65s |
| 15 | 1.2 m / 10 cycles | Yes | t = 35.25s | t = 10.65s |

The detection architecture itself required **no changes at all** to scale
from 6 to 15 drones — same code, same layers, same local evidence rule.
Only the physical attack parameters needed re-tuning, and only because the
*control* architecture's robustness scales with connectivity, which is a
separate, interesting result in its own right.

**Also worth knowing**: attack severity does not increase monotonically
with the ramp rate (`step_size_m`) — e.g. at N=6, `step_size=0.7` and `0.9`
produced *no* collision while `0.5` did. This reflects genuine nonlinear
dynamics in the collision-safety filter's reactive correction (a large
enough sudden error can trigger a strong-enough corrective response to
overshoot back to safety), not a bug. Don't assume "bigger attack = more
danger" without checking; `compare_n.py`'s sweep methodology (documented in
its module docstring) is the reliable way to find a dangerous setting for
a new configuration.

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

## Key results (seed=7, default parameters: N=15)

| Event | Time |
|---|---|
| Attack onset | t = 9.0s |
| **Collision threshold crossed (undefended)** | **t = 35.25s** |
| D7 (victim, has direct Layer-3 access) declares D2 compromised | **t = 10.65s** |
| Fastest indirect observer (D8, Layer 1 + Layer 2 only) | t = 12.90s |
| Remaining 12 non-victim drones (Layer 1 + Layer 2 only) | t = 13.05-13.35s |

The headline result: **the victim detects and correctly attributes the
attack ~24.6 seconds before physical collision occurs.** Drones without
direct Layer-3 access (i.e., that never initiated an exchange against the
attacker themselves) take a bit longer, relying on Layer 1's CUSUM
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
- `animate.py` — animated visualization of the same scenario (MP4 + GIF)
