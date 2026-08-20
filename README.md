# Swarm Range-Falsification Simulation

A Python simulation of the system architecture from the paper
*"Distributed Detection and Attribution of Range-Falsification Attacks in
GNSS-Denied UWB Drone Swarms"*.

**Current focus: detection and attribution only** (Sec. 6) — does the
swarm detect that a range-falsification attack is happening, and does it
correctly identify the compromised drone? Physical consequences (whether a
falsified range leads to a collision) were an earlier focus of this
simulation and are retained for context (Part 1 plots), but are no longer
the point; no physical-danger tuning is maintained going forward. The
attacker may falsify its reported range to more than one victim
simultaneously, and victims are chosen at random (reproducibly) rather
than fixed by hand.

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
formation, the falsified links highlighted in red once the attack starts
(one per victim), a green ring around each drone's dot the moment it
independently declares D2 compromised (a single aggregate counter is used
instead of per-drone text, since N individual labels overlap illegibly at
higher N), and a synchronized safety-margin panel. The MP4 requires ffmpeg
on your PATH; the GIF only needs Pillow (already in `requirements.txt`)
and is produced either way.

![Swarm range-falsification attack: detection and attribution](outputs/swarm_animation.gif)

**Configuring the attack** (`simulate.py`): `N_VICTIMS` sets how many
drones the attacker simultaneously falsifies its reported range to (victims
are chosen at random from a seeded generator, reproducible but not fixed
by hand). `part2_detection_count.png` plots directly the thing this is
built to answer: how many of the N-1 other drones have, by each point in
time, independently detected and correctly identified the attacker.
`summary.txt` also states this as a single explicit count
("X / N-1 other drones detected and identified the attacker").

`compare_n.py` runs an earlier, collision-focused version of this scenario
at N=6 and N=15 side by side (see "Scaling to more drones" below). It
predates the shift to detection-only and still uses single, nearest-
neighbor victim selection rather than the random multi-victim default
described above — kept because the N-scaling finding it demonstrates
(below) is still valid and worth having, not as the current default
scenario.

## Scaling to more drones

The formation generalizes to any N via `generate_formation()` (a
sunflower/Fibonacci disk layout — even 2D spread, no accidental
collinearity, for any N).

**A real, non-obvious finding from doing this** (from `compare_n.py`,
back when physical danger was still the focus): with a *fixed* attack
rate, N=15 showed dramatically *less* physical danger than N=6 — no
collision at all, versus a clean collision at N=6. This was not a bug. Each
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

**The result that still matters now**: the detection architecture itself
required **no changes at all** to scale from 6 to 15 drones — same code,
same layers, same local evidence rule. Only the physical attack parameters
needed re-tuning, and only because the
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

- **N=15 drones** by default (scales to any N — see "Scaling to more
  drones" below) in a well-spread 2D formation, ranging against every
  other drone once per cycle (Sec. "TDMA Mesh Schedule").
- **A leader that does not count in the simulation at all**: never eligible
  as attacker or victim, invisible to every detection layer, never in the
  ranging matrix. It moves on a real, scripted route (constant velocity —
  zero acceleration by construction) that the swarm follows via direct PD
  control on true position/velocity error. This gives the swarm a genuine,
  legitimate anchor to fly in formation around, rather than the noise-
  driven whole-swarm drift an earlier version of this simulation had with
  no absolute reference at all.
- **D2 is the attacker**; it falsifies its reported range to `N_VICTIMS`
  (default 3) other drones simultaneously, chosen at random from a seeded
  generator rather than fixed by hand.
- **Motion is driven entirely by leader-following** — peer-to-peer range
  estimates are used only for detection, never to drive physical movement.
  This makes the swarm's physical motion provably immune to the attack by
  construction (the attacker is never the leader), and sidesteps a couple
  of real closed-loop stability issues found and documented in git history
  (a peer-spring/leader-anchor interaction, and a textbook range-only
  bearing ambiguity) rather than papering over them.
- Each drone can only overhear a **subset** of other drones' ranging
  exchanges, based on physical proximity — see "Partial connectivity"
  below. Its own direct measurements (to every other drone, including the
  attacker) are always available regardless.
- **No mitigation is applied to a flagged link** — this is a deliberate,
  direct reflection of the paper's explicit scope (Sec. 3): the paper's
  contribution stops at detection and attribution, not response.
- Detection (Sec. 6) runs in parallel, observing the same ranging stream.
  It does not alter the physics. Physical outcome (Part 1) is retained for
  context but is no longer the point — detection and attribution (Part 2),
  in particular how many of the N-1 other drones detect and correctly
  identify the attacker, is.

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

## Key results (seed=7, default parameters: N=15, 3 random victims, partial connectivity)

| Event | Time |
|---|---|
| Attack onset | t = 9.0s |
| The 3 direct victims (D10, D11, D13) declare D2 compromised | **t = 10.65s** |
| 6 more drones with Layer 2 corroboration declare D2 compromised | t = 12.45s – 14.55s |
| **Final detection count** | **9 / 14 other drones** (5 never detect) |

The headline result: **the drones the attacker directly targeted detect
and correctly identify it almost immediately** (t=10.65s) via Layer 3's
unambiguous cross-check — this never depends on connectivity, since it's
each victim's own direct measurement. Drones with enough nearby structure
to corroborate via Layer 2 catch up within another few seconds. But
**5 of the 14 other drones never detect the attack at all**: they were
never directly lied to (not victims) and are geometrically too far from
the attacker/victims to get any Layer 2 evidence about it — a genuine,
visible limitation of partial connectivity, not a bug to be tuned away.

## Partial connectivity: each drone hears only a subset of other links

Ranging is still full-mesh every cycle (every drone directly ranges with
every other drone, per the TDMA schedule) -- so **Layer 1 and Layer 3 are
completely unaffected**: they only ever use a drone's own direct
measurement, which always exists regardless of connectivity. What partial
connectivity limits is **Layer 2**: overhearing *other* pairs' broadcasts,
which its geometric consistency check needs several of.

A drone `k` can use another pair `(i,j)`'s broadcast distance only if at
least one of `i, j` is within `HEARING_RANGE_M` of `k` (12m by default,
near the formation's median pairwise distance). This guarantees `k`'s
resulting local matrix is always fully self-consistent (no partial-row
gaps to work around), while still being genuinely partial: if the
attacker isn't within `k`'s local neighborhood at all, `k` simply gets no
Layer 2 evidence about it that cycle, and relies on Layer 1/3 alone
(which may also never fire, if `k` was never one of the falsified links).

**This produces a materially different, more realistic result** than
full connectivity: at the default settings, only **9 of 14** other drones
ever detect and identify the attacker -- the remaining 5 are geometrically
distant from both the attacker and its victims, get no Layer 2
corroboration, and (since they were never lied to directly) get nothing
from Layer 1 either. `part2_detection_count.png` shows this as a
staircase that climbs and then plateaus below "all 14," rather than
reaching it -- a direct, visible illustration of a genuine limitation of
partial connectivity, not swept under the rug.

## Known simplifications (stated plainly, not hidden)

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
- Part 1's trajectory/separation plots are truncated shortly after a
  collision *if one occurs* (continuing to integrate the simplified
  controller well past a real collision produces unbounded drift that
  isn't part of the phenomenon being demonstrated); with no collision, as
  in the current default scenario, no truncation happens. Part 2
  (detection) plots always use the full simulated range regardless.

## File overview

- `model.py` — ground-truth dynamics, formation guidance, collision-safety filter
- `ekf.py` — pairwise relative-state EKF (Sec. 2.3)
- `ranging.py` — full-mesh ranging + the attack model
- `detection.py` — Layers 1/2/3 + Sec. 6.4 local evidence combination
- `simulate.py` — orchestration, plotting, `summary.txt`
- `animate.py` — animated visualization of the same scenario (MP4 + GIF)
- `compare_n.py` — legacy N=6 vs N=15 collision-focused comparison (see
  "Scaling to more drones"); not part of the current default scenario
