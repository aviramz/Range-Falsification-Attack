"""
simulate.py
===========
Orchestrates the full simulation described to the user:

  1. n drones organized in a fixed 2D formation (a well-spread sunflower/
     Fibonacci disk layout, scalable to any N -- see generate_formation()).
     The attacker is fixed; its victims are chosen at random (reproducibly,
     via a seeded generator) and it falsifies its reported range to all of
     them simultaneously (see N_VICTIMS below).
  2. Ranging + EKF + formation guidance + collision-safety filter run
     every cycle exactly as in the paper's pipeline (Sec. 5.2). No
     mitigation is applied to a flagged link -- consistent with the
     paper's explicit scope (Sec. 3): detection and attribution only,
     response is out of scope. Physical danger (Part 1) is retained for
     context but is no longer the focus of this simulation -- detection
     and attribution (Part 2) is.
  3. In parallel (not causally affecting the physics), every layer of
     the detection architecture (Sec. 6) observes the same ranging
     stream and produces its diagnostics -- Part 2, including how many
     of the N-1 other drones have independently detected and identified
     the attacker.

Outputs (written to ./outputs/):
  part1_trajectories.png     - ground-truth flight paths
  part1_min_separation.png   - true distance on every pair vs. time
  part2_layer1.png           - NIS and two-sided CUSUM on a representative victim's link
  part2_layer2.png           - Layer-2 leave-one-out attribution score per node
  part2_evidence.png         - combined per-neighbor evidence E_i(j) (Sec. 6.4)
  part2_detection_count.png  - number of drones that have identified the attacker, over time
  summary.txt                - key numbers, including final detection count
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SwarmState
from ekf import RelativeRangeEKF
from ranging import AttackConfig, RangingCycle
from detection import PerLinkGate, leave_one_out_scores, layer3_check, EvidenceTracker

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
RNG_SEED = 7
N = 15
DT = 0.15
CYCLES = 420
SIGMA_RANGE = 0.05          # UWB range measurement noise, std (m)
PROCESS_NOISE_Q = 0.02      # trusts its own constant-velocity model somewhat
                             # more than a fully responsive tracker would;
                             # enough for CUSUM to accumulate a persistent
                             # signal from the staircase attack (Sec. 6.1)
                             # while still reacting realistically for control
D_MIN = 1.2                 # collision radius (m)
KP, KD, A_LIMIT = 2.0, 2.5, 3.0
SAFETY_ALPHA = 2.5

ATTACKER = 1   # index into the generated formation (see generate_formation below)
ATTACK_DIRECTION = "enlarge"
ATTACK_MAGNITUDE_M = 3.0    # a real, binding ceiling now (not the old
                             # "non-binding" placeholder from when large
                             # sustained bias was needed to force a
                             # collision): detection (Sec. 6) triggers
                             # easily from a modest, bounded lie, and an
                             # unbounded 40+ m bias was causing the
                             # directly-affected drones to fly off in
                             # unrealistic excursions -- unnecessary now
                             # that physical danger is not the point
ATTACK_ONSET_CYCLE = 60
ATTACK_STEP_EVERY = 10      # staircase: +ATTACK_STEP_SIZE_M every this many cycles
ATTACK_STEP_SIZE_M = 1.2
N_VICTIMS = 3                # attacker falsifies its reported range to this
                             # many other drones simultaneously, each an
                             # independent falsified link

# ---- Partial connectivity (Sec. 3 / Sec. 6.2) ----
# Ranging itself is still full-mesh every cycle (every drone directly
# ranges with every other drone, per the TDMA schedule, Sec. "TDMA Mesh
# Schedule") -- so Layer 1 and Layer 3 are UNAFFECTED by this: they only
# ever use a drone's own direct measurement to the attacker, which always
# exists. What partial connectivity limits is OVERHEARING other pairs'
# broadcasts for Layer 2's geometric consistency check (Sec. 6.2), which
# needs several OTHER drones' mutual distances, not just its own. A drone
# far from the rest of a sparse cluster may simply not have enough nearby
# structure to run a meaningful Layer 2 check on a distant node, even
# though it still has a perfectly good direct (Layer 1/3) measurement to
# it -- this models that limitation honestly rather than assuming global
# broadcast visibility (see detection.py's "full connectivity" note,
# which this replaces with an actual partial model).
HEARING_RANGE_M = 12.0  # a drone can only use ANOTHER pair (i,j)'s
                          # broadcast distance for Layer 2 if at least one
                          # of i, j is within this range of it -- chosen
                          # near the formation's median pairwise distance
                          # (~12m for the default N=15 layout), giving
                          # genuine partial (not all-or-nothing) coverage

# ---- Leader (does not participate in the simulation at all -- Sec. n/a) ----
# The leader is not one of the N drones: it never ranges as part of the
# N(N-1) mesh, is never a possible attacker or victim, and is invisible to
# every detection layer. It exists purely to give the swarm a real,
# deliberate reference to follow -- each drone ranges to it separately
# (noisy, like any other range) and additionally corrects toward its
# nominal offset from the leader. This also fixes an unrelated cosmetic
# issue: without any absolute anchor, the swarm's centroid does a slow
# random walk driven by measurement noise (a known property of purely
# relative formation control, see "Known simplifications" in the README).
# A leader moving on a real route gives every drone a true, deliberate
# common reference, replacing that noise-driven drift with smooth,
# purposeful motion.
#
# The leader's route uses the SAME "manual motion" mechanism as the
# original source simulator (simulator/scenarios/models.py's
# RadialAcceleration, applied in runtime.py's _manual_control_command):
# rather than a closed-form parametric path, the leader's velocity is
# recomputed every cycle as the tangent direction around LEADER_CENTER
# from its CURRENT true position, with speed capped at
# sqrt(LEADER_MAX_RADIAL_ACCEL * radius) -- a physically-grounded limit
# on how tight a turn is sustainable, taken directly from the source. See
# radial_leader_velocity() below, a direct port of that logic.
#
# Each drone's formation offset is defined in the LEADER'S BODY FRAME
# (relative to its current heading) rather than as a fixed world-frame
# vector, so the whole formation rotates rigidly with the leader as it
# turns -- exactly like real formation flight banking through a turn
# together, rather than a fixed-orientation shape being dragged along an
# arc. See leader_nominal_offset_body and the target_pos/vel computation
# in run() below.
LEADER_CENTER = np.array([0.0, -40.0])     # center of the leader's circular route
LEADER_RADIUS0 = 40.0                       # initial distance from center (m)
LEADER_PHI0 = np.pi / 2                     # starting angle on the circle
LEADER_SPEED = 1.8                          # requested tangential speed (m/s),
                                              # matching the previous straight-
                                              # line speed
LEADER_MAX_RADIAL_ACCEL = 4.0               # m/s^2 -- caps speed via
                                              # sqrt(a*r), exactly the safety
                                              # cap from the source's
                                              # RadialAcceleration
LEADER_TURN_SIGN = 1.0                      # +1 = counter-clockwise

LEADER_KP, LEADER_KD = 2.5, 6.0  # KD >= 2*sqrt(KP) (~3.16 here) for
                                   # critical/over-damping; the previous
                                   # KP=KD=2.5 was underdamped, and at
                                   # DT=0.15s that discretization could
                                   # grow into a visible oscillation for
                                   # some drones' specific geometry even
                                   # with no attack involved at all
LEADER_ACCEL_LIMIT = 4.0  # m/s^2, hard saturation -- see comment at
                            # leader_accel_i below


def radial_leader_velocity(pos: np.ndarray) -> np.ndarray:
    """
    Direct port of the source simulator's RadialAcceleration manual-motion
    rule (simulator/scenarios/runtime.py, _manual_control_command): the
    tangent direction around LEADER_CENTER from the CURRENT position `pos`,
    at a speed capped by sqrt(LEADER_MAX_RADIAL_ACCEL * radius) -- the
    source's physically-grounded limit on sustainable turn tightness.
    """
    radial = pos - LEADER_CENTER
    tangent = np.array([-radial[1], radial[0]])  # cross((0,0,1), radial) in 2D
    radius = np.linalg.norm(tangent)
    max_speed = np.sqrt(LEADER_MAX_RADIAL_ACCEL * radius)
    speed = min(LEADER_SPEED, max_speed)
    direction = tangent / radius
    if LEADER_TURN_SIGN < 0:
        direction = -direction
    return direction * speed


def generate_formation(n: int, spacing: float = 3.0) -> np.ndarray:
    """
    Well-spread 2D formation for any N, via a sunflower/Fibonacci disk
    pattern: guarantees genuine 2D geometry (no accidental collinearity)
    and even spread, unlike an arbitrary hand-placed layout that only
    works for one specific N. `spacing` targets the approximate
    nearest-neighbor distance (m), matching the scale used when this was
    a fixed 6-drone layout.
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radius_scale = spacing * np.sqrt(n / np.pi)
    pts = np.zeros((n, 2))
    for k in range(n):
        r = radius_scale * np.sqrt((k + 0.5) / n)
        theta = k * golden_angle
        pts[k] = [r * np.cos(theta), r * np.sin(theta)]
    return pts


INITIAL_POS = generate_formation(N, spacing=6.0)

# Victims are chosen at random from a dedicated, seeded generator (kept
# separate from the ranging-noise generator inside run(), so victim
# selection is reproducible independent of anything else). Not restricted
# to nearest neighbors -- a real attacker has no reason to prefer them,
# and physical danger is no longer the point of this simulation (see
# README): the question here is purely whether the swarm detects and
# identifies a compromised drone, regardless of which links it targets.
_victim_rng = np.random.default_rng(RNG_SEED)
VICTIMS = sorted(_victim_rng.choice(
    [i for i in range(N) if i != ATTACKER], size=N_VICTIMS, replace=False
).tolist())
VICTIM = VICTIMS[0]  # representative victim, used for single-link illustrative plots


def run():
    rng = np.random.default_rng(RNG_SEED)
    swarm = SwarmState(INITIAL_POS)

    # NOTE: peer-to-peer FormationGuidance and CollisionSafetyFilter
    # (Sec. 5.2's original formation spring and collision-avoidance net)
    # are intentionally NOT used to drive motion -- see the comment at the
    # leader-only guidance block below. Motion depends solely on each
    # drone's own leader-tracking EKF.

    # Each drone maintains one relative-state EKF per neighbor (Sec. 2.3 / 5.2).
    ekfs: dict[tuple[int, int], RelativeRangeEKF] = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                ekfs[(i, j)] = RelativeRangeEKF(
                    initial_p=INITIAL_POS[j] - INITIAL_POS[i],
                    initial_v=np.zeros(2),
                    process_noise_q=PROCESS_NOISE_Q, range_noise_std=SIGMA_RANGE,
                )

    # --- Leader tracking: does not count in the simulation at all (never
    # touched by Layer 1/2/3 or evidence accumulation, never appears in the
    # N(N-1) ranging matrix, never eligible as attacker or victim).
    #
    # IMPORTANT: this does NOT reuse the range-only RelativeRangeEKF used
    # for peer tracking. That was tried first and found, by direct
    # inspection, to suffer a textbook range-only bearing ambiguity: with
    # only a scalar distance (no bearing) and no bounds on how close the
    # true relative position could pass to zero, the filter would
    # occasionally lock onto the wrong direction once distance shrank
    # toward zero and grew again on the "other side" -- a well-known
    # failure mode of scalar-range tracking, not fixable by gain tuning
    # (verified: gain sweeps and output saturation did not fix it, only
    # bounded the resulting blowup). Since the leader explicitly does not
    # count in the simulation at all, there is no reason to route this
    # auxiliary, non-adversarial control channel through the same noisy,
    # ambiguity-prone estimator used for the (attackable, detection-
    # relevant) peer links. Direct PD control on true position/velocity
    # error to a scripted target is unconditionally stable and has no
    # such ambiguity, and does not touch anything detection depends on.
    #
    # Offsets are stored as plain WORLD-FRAME vectors at t=0 (the initial
    # true offset from the leader) -- NOT pre-rotated into any "body
    # frame". Each cycle, this initial offset is rotated by the
    # INCREMENTAL heading change since t=0 (identity at t=0 by
    # construction, so the initial condition is exactly correct); this is
    # what makes the formation rotate rigidly with the leader through a
    # turn rather than being dragged along at a fixed absolute
    # orientation. Target velocity is obtained by finite-differencing the
    # rotated target position cycle-to-cycle rather than a closed-form
    # derivative, since the leader's own velocity (radial_leader_velocity)
    # is itself a feedback rule recomputed from its current position, not
    # a closed-form function of time -- finite-differencing works
    # correctly regardless of the underlying leader motion law.
    leader_pos = LEADER_CENTER + LEADER_RADIUS0 * np.array(
        [np.cos(LEADER_PHI0), np.sin(LEADER_PHI0)])
    leader_vel0 = radial_leader_velocity(leader_pos)
    heading0 = np.arctan2(leader_vel0[1], leader_vel0[0])
    leader_nominal_offset_world0 = {
        i: INITIAL_POS[i] - leader_pos for i in range(N)
    }
    prev_target_pos = {i: INITIAL_POS[i].copy() for i in range(N)}

    attack = AttackConfig(ATTACKER, VICTIMS, ATTACK_DIRECTION, ATTACK_MAGNITUDE_M,
                           ATTACK_ONSET_CYCLE, ATTACK_STEP_EVERY, ATTACK_STEP_SIZE_M)

    # ---- Detection state (Sec. 6) ----
    layer1_gates: dict[tuple[int, int], PerLinkGate] = {k: PerLinkGate() for k in ekfs}
    evidence = {i: EvidenceTracker(N) for i in range(N)}  # each drone's local view

    # ---- History for plotting ----
    hist_pos = np.zeros((CYCLES, N, 2))
    hist_leader_pos = np.zeros((CYCLES, 2))
    hist_true_dist = np.zeros((CYCLES, N, N))
    hist_nis_link = np.zeros(CYCLES)          # representative victim's EKF-for-attacker
    hist_cplus = np.zeros(CYCLES)
    hist_cminus = np.zeros(CYCLES)
    hist_delta = np.zeros((CYCLES, N))        # Layer-2 leave-one-out score per node
    hist_evidence_attacker = np.zeros((CYCLES, N))  # E_i(attacker) for every observer i
    hist_layer3_flag = np.zeros(CYCLES, dtype=bool)  # representative victim's link only
    hist_n_detected = np.zeros(CYCLES, dtype=int)    # how many observers have detected, by this cycle
    detection_cycle_per_observer = {i: None for i in range(N) if i != ATTACKER}
    true_min_sep_series = np.zeros(CYCLES)
    collision_cycle = None

    for c in range(CYCLES):
        cyc = RangingCycle(swarm, SIGMA_RANGE, rng, attack, c)
        # Leader state for this cycle: velocity is a feedback rule on the
        # CURRENT position (radial_leader_velocity), matching the source's
        # RadialAcceleration manual-motion mechanism -- not a closed-form
        # function of time.
        leader_true_pos = leader_pos
        leader_true_vel = radial_leader_velocity(leader_true_pos)

        # --- EKF predict+update for every ordered pair (i observes j) ---
        # accelerations realized last step are shared (only RANGE is attacked,
        # per the paper's threat model, Sec. 3) -- filled in after control below.
        estimates: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {i: {} for i in range(N)}
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                ek = ekfs[(i, j)]
                u = getattr(swarm, "_last_accel", np.zeros((N, 2)))[j] - \
                    getattr(swarm, "_last_accel", np.zeros((N, 2)))[i]
                ek.predict(u, DT)
                # a rational attacker does not fool its own controller (Sec. 3):
                # it uses the TRUE distance for its own EKFs.
                z = cyc.true_dist[(min(i, j), max(i, j))] + rng.normal(0, SIGMA_RANGE) \
                    if i == ATTACKER else cyc.get(i, j)
                ek.update(z)
                estimates[i][j] = (ek.rel_pos, ek.rel_vel)

                # ---- Layer 1 (Sec. 6.1) ----
                gate = layer1_gates[(i, j)]
                gate.update(ek.last_nis, ek.standardized_innovation)

        # --- Leader-following: direct PD control on TRUE position/velocity
        # error to a target that ROTATES with the leader's current heading
        # (see module-level comment above) -- not a fixed world-frame
        # offset. Target velocity comes from finite-differencing the
        # rotated target position cycle-to-cycle, which is correct
        # regardless of the underlying leader motion law. No estimator in
        # this loop at all -- see the module-level comment above for why.
        # Detection (Sec. 6) is entirely unaffected: it still runs on the
        # full peer EKFs/ranging exactly as before, independent of what
        # drives physical motion.
        heading_t = np.arctan2(leader_true_vel[1], leader_true_vel[0])
        delta_theta = heading_t - heading0
        ct, st = np.cos(delta_theta), np.sin(delta_theta)
        Rt = np.array([[ct, -st], [st, ct]])

        accel = np.zeros((N, 2))
        for i in range(N):
            target_pos = leader_true_pos + Rt @ leader_nominal_offset_world0[i]
            target_vel = (target_pos - prev_target_pos[i]) / DT if c > 0 else leader_true_vel
            prev_target_pos[i] = target_pos.copy()
            pos_error = target_pos - swarm.pos[i]
            vel_error = target_vel - swarm.vel[i]
            a = LEADER_KP * pos_error + LEADER_KD * vel_error
            a_norm = np.linalg.norm(a)
            if a_norm > LEADER_ACCEL_LIMIT:
                a = a * (LEADER_ACCEL_LIMIT / a_norm)
            accel[i] = a
        swarm._last_accel = accel
        swarm.step(accel, DT)

        # Advance leader state for next cycle (after this cycle's targets
        # were computed from its current position, matching how the
        # source's manual-motion velocity command is applied per-step).
        leader_pos = leader_true_pos + leader_true_vel * DT

        # --- Layer 2 (Sec. 6.2): PER-DRONE local matrix, not a shared
        # global one -- each drone k only uses entries it could actually
        # overhear (see HEARING_RANGE_M comment above). For any i,j both
        # within k's hearing range (or equal to k, since k always has its
        # own direct measurements), the exchange is audible to k because
        # at least one endpoint is within range of k by construction; this
        # guarantees the restricted submatrix is fully observed without
        # needing a separate max-clique search. deltas_per_observer[k] is
        # None if k doesn't have enough locally-audible structure (<4
        # nodes) for a meaningful embedding (Sec. 6.2's redundancy
        # requirement), or if the attacker isn't in k's audible set at
        # all -- in either case k simply gets no Layer 2 evidence about
        # the attacker this cycle, falling back on Layer 1/3 alone.
        deltas_per_observer: dict[int, float] = {}
        for k in range(N):
            audible = [k] + [j for j in range(N) if j != k
                              and cyc.true_dist[(min(k, j), max(k, j))] < HEARING_RANGE_M]
            if ATTACKER not in audible or len(audible) < 4:
                deltas_per_observer[k] = 0.0
                continue
            idx = {node: local for local, node in enumerate(audible)}
            D_k = np.zeros((len(audible), len(audible)))
            for a in range(len(audible)):
                for b in range(a + 1, len(audible)):
                    d = cyc.get(audible[a], audible[b])
                    D_k[a, b] = D_k[b, a] = d
            local_deltas = leave_one_out_scores(D_k)
            deltas_per_observer[k] = float(local_deltas[idx[ATTACKER]])

        # --- Layer 3 (Sec. 6.3): checked independently for EVERY drone that
        # ranged against the attacker this cycle, not just one hardcoded pair
        # -- generalizes cleanly to any number of simultaneously falsified
        # links, since cyc.get_layer3 is populated for every pair the
        # attacker is a party to (see ranging.py), and naturally reports no
        # mismatch on pairs that were not actually falsified this cycle. ---
        l3_flag_per_observer: dict[int, bool] = {}
        for i in range(N):
            if i == ATTACKER:
                continue
            recompute = cyc.get_layer3(min(i, ATTACKER), max(i, ATTACKER))
            if recompute is not None:
                reported = cyc.get(i, ATTACKER)
                l3_flag_per_observer[i] = layer3_check(reported, recompute)
            else:
                l3_flag_per_observer[i] = False

        # Also save the representative observer's (VICTIM's) full per-node
        # vector, for the Layer 2 illustrative plot -- NaN for any node
        # outside VICTIM's own audible set, since under partial
        # connectivity there is no longer one shared global vector every
        # observer sees identically (each observer has its own local view;
        # VICTIM's is just the one plotted as an example).
        victim_audible = [VICTIM] + [j for j in range(N) if j != VICTIM
                          and cyc.true_dist[(min(VICTIM, j), max(VICTIM, j))] < HEARING_RANGE_M]
        victim_delta_vec = np.full(N, np.nan)
        if len(victim_audible) >= 4:
            v_idx = {node: local for local, node in enumerate(victim_audible)}
            D_v = np.zeros((len(victim_audible), len(victim_audible)))
            for a in range(len(victim_audible)):
                for b in range(a + 1, len(victim_audible)):
                    d = cyc.get(victim_audible[a], victim_audible[b])
                    D_v[a, b] = D_v[b, a] = d
            v_deltas = leave_one_out_scores(D_v)
            for node, local in v_idx.items():
                victim_delta_vec[node] = v_deltas[local]

        # --- Sec. 6.4: combine into each observer's local evidence about the attacker ---
        n_detected_this_cycle = 0
        for i in range(N):
            if i == ATTACKER:
                continue
            gate = layer1_gates[(i, ATTACKER)]
            evidence[i].accumulate(
                ATTACKER,
                layer1_alarm=gate.sequential_flag,
                delta_j=deltas_per_observer[i],
                layer3_flag=l3_flag_per_observer[i],
            )
            hist_evidence_attacker[c, i] = evidence[i].E[ATTACKER]
            if detection_cycle_per_observer[i] is None and ATTACKER in evidence[i].declared_compromised:
                detection_cycle_per_observer[i] = c
            if ATTACKER in evidence[i].declared_compromised:
                n_detected_this_cycle += 1
        hist_n_detected[c] = n_detected_this_cycle

        # --- bookkeeping ---
        hist_pos[c] = swarm.pos
        hist_leader_pos[c] = leader_true_pos
        for i in range(N):
            for j in range(N):
                hist_true_dist[c, i, j] = np.linalg.norm(swarm.pos[j] - swarm.pos[i]) if i != j else 0.0
        g = layer1_gates[(VICTIM, ATTACKER)]
        hist_nis_link[c] = ekfs[(VICTIM, ATTACKER)].last_nis
        hist_cplus[c] = g.C_plus
        hist_cminus[c] = g.C_minus
        hist_delta[c] = victim_delta_vec
        hist_layer3_flag[c] = l3_flag_per_observer.get(VICTIM, False)
        min_sep = swarm.min_pairwise_distance()
        true_min_sep_series[c] = min_sep
        if collision_cycle is None and min_sep < D_MIN:
            collision_cycle = c

    return dict(
        hist_pos=hist_pos, hist_leader_pos=hist_leader_pos,
        hist_true_dist=hist_true_dist, hist_nis_link=hist_nis_link,
        hist_cplus=hist_cplus, hist_cminus=hist_cminus, hist_delta=hist_delta,
        hist_evidence_attacker=hist_evidence_attacker, hist_layer3_flag=hist_layer3_flag,
        hist_n_detected=hist_n_detected,
        detection_cycle_per_observer=detection_cycle_per_observer,
        true_min_sep_series=true_min_sep_series, collision_cycle=collision_cycle,
        attack=attack,
    )


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def make_plots(res: dict) -> None:
    t_full = np.arange(CYCLES) * DT
    onset_t = ATTACK_ONSET_CYCLE * DT

    # Truncate plotting shortly after a true collision: physically the
    # swarm has already "stopped functioning" at that point (Sec. 1), and
    # continuing to integrate the simplified formation controller well
    # past that point produces unbounded drift from having no absolute
    # position anchor -- a known limitation of purely relative-only
    # formation control, not part of the phenomenon being demonstrated
    # here. Detection (Part 2) is unaffected: it resolves at t=9.6-19.5s,
    # well before this cutoff.
    if res["collision_cycle"] is not None:
        cutoff = min(CYCLES, res["collision_cycle"] + 60)
    else:
        cutoff = CYCLES
    t = t_full[:cutoff]

    # ---- Part 1a: trajectories ----
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = [f"D{k+1}" for k in range(N)]
    other_labeled = False
    victim_labeled = False
    for k in range(N):
        traj = res["hist_pos"][:cutoff, k, :]
        if k == ATTACKER:
            color, lbl = "#d62728", f"{labels[k]} (attacker)"
        elif k in VICTIMS:
            color = "#1f77b4"
            lbl = ("victims: " + ", ".join(labels[v] for v in VICTIMS)) if not victim_labeled else None
            victim_labeled = True
        else:
            color = "#7f7f7f"
            lbl = "other drones" if not other_labeled else None
            other_labeled = True
        lw = 1.8 if k in (ATTACKER, *VICTIMS) else 0.8
        alpha = 1.0 if k in (ATTACKER, *VICTIMS) else 0.6
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, alpha=alpha, label=lbl)
        ax.scatter(*traj[0], marker="o", color=color, s=30, zorder=5)
        ax.scatter(*traj[-1], marker="s", color=color, s=30, zorder=5)
    leader_traj = res["hist_leader_pos"][:cutoff]
    ax.plot(leader_traj[:, 0], leader_traj[:, 1], color="black", lw=1.5, ls="--", zorder=6, label="leader")
    ax.scatter(*leader_traj[0], marker="*", color="black", s=120, zorder=7)
    ax.scatter(*leader_traj[-1], marker="*", color="black", s=120, zorder=7)
    ax.set_title("Part 1: ground-truth trajectories under an undefended range-falsification attack\n"
                 "(circles = start, squares/star = end)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part1_trajectories.png"), dpi=150)
    plt.close(fig)

    # ---- Part 1b: minimum pairwise separation vs time ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, res["true_min_sep_series"][:cutoff], color="#333333", lw=1.5, label="true min. pairwise separation")
    ax.axhline(D_MIN, color="#d62728", ls="--", lw=1.2, label=f"collision radius ({D_MIN} m)")
    ax.axvline(onset_t, color="#999999", ls=":", lw=1.2, label="attack onset")
    if res["collision_cycle"] is not None:
        ax.axvline(res["collision_cycle"] * DT, color="#d62728", ls="-", lw=1.0, alpha=0.6,
                   label="collision threshold crossed")
    ax.set_title("Part 1: swarm safety margin with NO detection/mitigation active")
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance (m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part1_min_separation.png"), dpi=150)
    plt.close(fig)

    # ---- Part 2a: Layer 1 (NIS + CUSUM) on the attacked link ----
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_full, res["hist_nis_link"], color="#333333", lw=1.0)
    axes[0].axhline(10.83, color="#d62728", ls="--", lw=1.0, label=r"$\chi^2_1$ threshold (p=0.001)")
    axes[0].axvline(onset_t, color="#999999", ls=":", lw=1.0)
    axes[0].set_ylabel("NIS")
    axes[0].set_title(f"Part 2, Layer 1: instantaneous gate on link "
                      f"(D{VICTIM+1} observes D{ATTACKER+1})")
    axes[0].legend(fontsize=8)
    axes[1].plot(t_full, res["hist_cplus"], color="#d62728", lw=1.3, label=r"$C^+$ accumulator")
    axes[1].plot(t_full, res["hist_cminus"], color="#1f77b4", lw=1.3, label=r"$C^-$ accumulator")
    axes[1].axhline(5.0, color="#333333", ls="--", lw=1.0, label="alarm threshold h")
    axes[1].axvline(onset_t, color="#999999", ls=":", lw=1.0, label="attack onset")
    axes[1].set_ylabel("CUSUM statistic"); axes[1].set_xlabel("time (s)")
    axes[1].set_title("Two-sided CUSUM (Sec. 6.1): either accumulator crossing h flags a\n"
                      "sustained anomaly the instantaneous gate above mostly misses")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part2_layer1.png"), dpi=150)
    plt.close(fig)

    # ---- Part 2b: Layer 2 leave-one-out attribution ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    other_labeled = False
    for k in range(N):
        if k == ATTACKER:
            ax.plot(t_full, res["hist_delta"][:, k], color="#d62728", lw=2.2,
                    label=f"{labels[k]} (attacker)", zorder=5)
        else:
            ax.plot(t_full, res["hist_delta"][:, k], color="#7f7f7f", lw=0.8, alpha=0.7,
                    label="other drones" if not other_labeled else None)
            other_labeled = True
    ax.axvline(onset_t, color="#999999", ls=":", lw=1.0, label="attack onset")
    ax.set_title(r"Part 2, Layer 2: $D{}$'s local leave-one-out attribution $\Delta_m$ (Algorithm 2)"
                 .format(VICTIM + 1))
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$\Delta_m$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part2_layer2.png"), dpi=150)
    plt.close(fig)

    # ---- Part 2c: combined evidence E_i(attacker) per observer ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    other_labeled = False
    victim_labeled = False
    for i in range(N):
        if i == ATTACKER:
            continue
        if i in VICTIMS:
            ax.plot(t_full, res["hist_evidence_attacker"][:, i], color="#1f77b4", lw=2.0,
                    label="victims' view (Layer 3 available)" if not victim_labeled else None, zorder=5)
            victim_labeled = True
        else:
            ax.plot(t_full, res["hist_evidence_attacker"][:, i], color="#7f7f7f", lw=0.8, alpha=0.7,
                    label="other drones' view" if not other_labeled else None)
            other_labeled = True
    ax.axhline(3.0, color="#333333", ls="--", lw=1.0, label=r"declare-compromised threshold $\Theta$")
    ax.axvline(onset_t, color="#999999", ls=":", lw=1.0, label="attack onset")
    dc = res["detection_cycle_per_observer"].get(VICTIM)
    if dc is not None:
        ax.axvline(dc * DT, color="#2ca02c", ls="-", lw=1.3,
                  label=f"D{VICTIM+1} (a victim) declares D{ATTACKER+1} compromised (t={dc*DT:.2f}s)")
    ax.set_title(r"Part 2, Sec. 6.4: each drone's local evidence score $E_i(\mathrm{attacker})$")
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$E_i(j{=}\mathrm{attacker})$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part2_evidence.png"), dpi=150)
    plt.close(fig)

    # ---- Part 2d: how many drones have detected AND identified the attacker ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_others = N - 1
    ax.plot(t_full, res["hist_n_detected"], color="#2ca02c", lw=2.0, drawstyle="steps-post")
    ax.axhline(n_others, color="#999999", ls="--", lw=1.0, label=f"all {n_others} other drones")
    ax.axvline(onset_t, color="#999999", ls=":", lw=1.0, label="attack onset")
    ax.set_ylim(-0.5, n_others + 1)
    ax.set_title(f"Part 2: number of drones that have identified D{ATTACKER+1} as compromised")
    ax.set_xlabel("time (s)"); ax.set_ylabel("drones that have declared the attacker compromised")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part2_detection_count.png"), dpi=150)
    plt.close(fig)


def write_summary(res: dict) -> None:
    lines = []
    lines.append("=== Baseline outcome (no mitigation, Sec. 3 scope; physical danger not the focus) ===")
    if res["collision_cycle"] is not None:
        lines.append(f"Collision threshold ({D_MIN} m) crossed at cycle {res['collision_cycle']} "
                      f"(t = {res['collision_cycle']*DT:.2f}s)")
    else:
        lines.append(f"Collision threshold ({D_MIN} m) never crossed in {CYCLES} cycles "
                      f"(min separation reached: {res['true_min_sep_series'].min():.2f} m)")
    lines.append("")
    lines.append("=== Detection outcome (Sec. 6) ===")
    n_others = N - 1
    n_ever_detected = sum(1 for c in res["detection_cycle_per_observer"].values() if c is not None)
    lines.append(f"{n_ever_detected} / {n_others} other drones detected and identified "
                 f"D{ATTACKER+1} as the compromised drone by the end of the simulation.")
    lines.append("")
    for i, c in sorted(res["detection_cycle_per_observer"].items(), key=lambda kv: (kv[1] is None, kv[1])):
        label = f"D{i+1}"
        tag = " (victim)" if i in VICTIMS else ""
        if c is None:
            lines.append(f"  {label}{tag}: never declared D{ATTACKER+1} compromised in {CYCLES} cycles")
        else:
            lines.append(f"  {label}{tag}: declared D{ATTACKER+1} compromised at cycle {c} (t = {c*DT:.2f}s)")
    lines.append("")
    victim_labels = ", ".join(f"D{v+1}" for v in VICTIMS)
    lines.append(f"Attack: D{ATTACKER+1} -> {{{victim_labels}}} ({len(VICTIMS)} simultaneously "
                 f"falsified links), {ATTACK_DIRECTION} by {ATTACK_MAGNITUDE_M} m, "
                 f"onset cycle {ATTACK_ONSET_CYCLE}, staircase +{ATTACK_STEP_SIZE_M} m every "
                 f"{ATTACK_STEP_EVERY} cycles ({ATTACK_STEP_EVERY*DT:.1f}s per step)")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    result = run()
    make_plots(result)
    write_summary(result)
    print(f"\nPlots written to {OUT}")
