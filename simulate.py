"""
simulate.py
===========
Orchestrates the full simulation described to the user:

  1. n drones organized in a fixed 2D formation (mirrors the paper's
     teaser figure, Fig. 1: drones D1..D6, attacker D2 falsifying its
     range to D5).
  2. Ranging + EKF + formation guidance + collision-safety filter run
     every cycle exactly as in the paper's pipeline (Sec. 5.2). No
     mitigation is applied to a flagged link -- consistent with the
     paper's explicit scope (Sec. 3): detection and attribution only,
     response is out of scope. This lets Part 1 show what an
     undefended swarm actually suffers.
  3. In parallel (not causally affecting the physics), every layer of
     the detection architecture (Sec. 6) observes the same ranging
     stream and produces its diagnostics -- Part 2.

Outputs (written to ./outputs/):
  part1_trajectories.png   - ground-truth flight paths
  part1_min_separation.png - true distance on every pair vs. time,
                             collision threshold, detection-time marker
  part2_layer1.png         - NIS and two-sided CUSUM on the attacked link
  part2_layer2.png         - Layer-2 leave-one-out attribution score per node
  part2_evidence.png       - combined per-neighbor evidence E_i(j) (Sec. 6.4)
  summary.txt              - key numbers (time-to-collision, time-to-detect, ...)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SwarmState, FormationGuidance, CollisionSafetyFilter
from ekf import RelativeRangeEKF
from ranging import AttackConfig, RangingCycle
from detection import PerLinkGate, leave_one_out_scores, layer3_check, EvidenceTracker

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
RNG_SEED = 7
N = 6
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

ATTACKER = 1   # "D2" in the paper's 1-indexed teaser figure
VICTIM = 4     # "D5" -- same falsified link shown in Fig. 1
ATTACK_DIRECTION = "enlarge"
ATTACK_MAGNITUDE_M = 2.5
ATTACK_ONSET_CYCLE = 60
ATTACK_STEP_EVERY = 10      # staircase: +ATTACK_STEP_SIZE_M every this many cycles
ATTACK_STEP_SIZE_M = 0.05   # individually small -- stays under the single-sample
                             # gate almost always; CUSUM accumulates the trend

# well-spread initial formation, echoing the teaser figure's layout
INITIAL_POS = np.array([
    [0.0, 3.0],   # D1
    [3.2, 4.0],   # D2  <- attacker
    [6.4, 2.6],   # D3
    [1.0, 0.0],   # D4
    [4.2, -0.6],  # D5  <- victim
    [7.4, 1.0],   # D6
])


def run():
    rng = np.random.default_rng(RNG_SEED)
    swarm = SwarmState(INITIAL_POS)

    nominal_offsets = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                nominal_offsets[(i, j)] = INITIAL_POS[j] - INITIAL_POS[i]

    guidance = FormationGuidance(nominal_offsets, kp=KP, kd=KD, accel_limit=A_LIMIT)
    safety = CollisionSafetyFilter(d_min=D_MIN, alpha=SAFETY_ALPHA)

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

    attack = AttackConfig(ATTACKER, [VICTIM], ATTACK_DIRECTION, ATTACK_MAGNITUDE_M,
                           ATTACK_ONSET_CYCLE, ATTACK_STEP_EVERY, ATTACK_STEP_SIZE_M)

    # ---- Detection state (Sec. 6) ----
    layer1_gates: dict[tuple[int, int], PerLinkGate] = {k: PerLinkGate() for k in ekfs}
    evidence = {i: EvidenceTracker(N) for i in range(N)}  # each drone's local view

    # ---- History for plotting ----
    hist_pos = np.zeros((CYCLES, N, 2))
    hist_true_dist = np.zeros((CYCLES, N, N))
    hist_nis_link = np.zeros(CYCLES)          # victim's EKF-for-attacker
    hist_cplus = np.zeros(CYCLES)
    hist_cminus = np.zeros(CYCLES)
    hist_delta = np.zeros((CYCLES, N))        # Layer-2 leave-one-out score per node
    hist_evidence_attacker = np.zeros((CYCLES, N))  # E_i(attacker) for every observer i
    hist_layer3_flag = np.zeros(CYCLES, dtype=bool)
    detection_cycle_per_observer = {i: None for i in range(N) if i != ATTACKER}
    true_min_sep_series = np.zeros(CYCLES)
    collision_cycle = None

    for c in range(CYCLES):
        cyc = RangingCycle(swarm, SIGMA_RANGE, rng, attack, c)

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

        # --- Formation guidance + collision-safety filter (Sec. 5.2) ---
        accel = np.zeros((N, 2))
        neighbors_all = [k for k in range(N)]
        for i in range(N):
            neighbors = [j for j in neighbors_all if j != i]
            a_des = guidance.desired_accel(i, neighbors, estimates[i])
            a_safe = safety.filter(a_des, neighbors, estimates[i])
            accel[i] = a_safe
        swarm._last_accel = accel
        swarm.step(accel, DT)

        # --- Layer 2 (Sec. 6.2): global matrix under full-connectivity (see detection.py) ---
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                D[i, j] = D[j, i] = cyc.get(i, j)
        deltas = leave_one_out_scores(D)

        # --- Layer 3 (Sec. 6.3): checked on the attacked pair whenever attacker responds ---
        l3_flag = False
        recompute = cyc.get_layer3(min(ATTACKER, VICTIM), max(ATTACKER, VICTIM))
        if recompute is not None:
            reported = cyc.get(ATTACKER, VICTIM)
            l3_flag = layer3_check(reported, recompute)

        # --- Sec. 6.4: combine into each observer's local evidence about the attacker ---
        for i in range(N):
            if i == ATTACKER:
                continue
            gate = layer1_gates[(i, ATTACKER)]
            evidence[i].accumulate(
                ATTACKER,
                layer1_alarm=gate.sequential_flag,
                delta_j=deltas[ATTACKER],
                layer3_flag=l3_flag if i == VICTIM else False,
            )
            hist_evidence_attacker[c, i] = evidence[i].E[ATTACKER]
            if detection_cycle_per_observer[i] is None and ATTACKER in evidence[i].declared_compromised:
                detection_cycle_per_observer[i] = c

        # --- bookkeeping ---
        hist_pos[c] = swarm.pos
        for i in range(N):
            for j in range(N):
                hist_true_dist[c, i, j] = np.linalg.norm(swarm.pos[j] - swarm.pos[i]) if i != j else 0.0
        g = layer1_gates[(VICTIM, ATTACKER)]
        hist_nis_link[c] = ekfs[(VICTIM, ATTACKER)].last_nis
        hist_cplus[c] = g.C_plus
        hist_cminus[c] = g.C_minus
        hist_delta[c] = deltas
        hist_layer3_flag[c] = l3_flag
        min_sep = swarm.min_pairwise_distance()
        true_min_sep_series[c] = min_sep
        if collision_cycle is None and min_sep < D_MIN:
            collision_cycle = c

    return dict(
        hist_pos=hist_pos, hist_true_dist=hist_true_dist, hist_nis_link=hist_nis_link,
        hist_cplus=hist_cplus, hist_cminus=hist_cminus, hist_delta=hist_delta,
        hist_evidence_attacker=hist_evidence_attacker, hist_layer3_flag=hist_layer3_flag,
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
    for k in range(N):
        traj = res["hist_pos"][:cutoff, k, :]
        color = "#d62728" if k == ATTACKER else ("#1f77b4" if k == VICTIM else "#7f7f7f")
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.6,
                label=f"{labels[k]}" + (" (attacker)" if k == ATTACKER else " (victim)" if k == VICTIM else ""))
        ax.scatter(*traj[0], marker="o", color=color, s=40, zorder=5)
        ax.scatter(*traj[-1], marker="s", color=color, s=40, zorder=5)
    ax.set_title("Part 1: ground-truth trajectories under an undefended range-falsification attack\n"
                 "(circles = start, squares = end)")
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
    axes[0].set_ylabel("NIS"); axes[0].set_title("Part 2, Layer 1: instantaneous gate on link (D5 observes D2)")
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
    for k in range(N):
        color = "#d62728" if k == ATTACKER else "#7f7f7f"
        lw = 2.0 if k == ATTACKER else 1.0
        ax.plot(t_full, res["hist_delta"][:, k], color=color, lw=lw,
                label=labels[k] + (" (attacker)" if k == ATTACKER else ""))
    ax.axvline(onset_t, color="#999999", ls=":", lw=1.0, label="attack onset")
    ax.set_title(r"Part 2, Layer 2: leave-one-out attribution score $\Delta_m$ (Algorithm 2)")
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$\Delta_m$")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part2_layer2.png"), dpi=150)
    plt.close(fig)

    # ---- Part 2c: combined evidence E_i(attacker) per observer ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(N):
        if i == ATTACKER:
            continue
        color = "#1f77b4" if i == VICTIM else "#7f7f7f"
        lw = 2.0 if i == VICTIM else 0.9
        ax.plot(t_full, res["hist_evidence_attacker"][:, i], color=color, lw=lw,
                label=labels[i] + ("'s view (initiator, Layer 3 available)" if i == VICTIM else "'s view"))
    ax.axhline(3.0, color="#333333", ls="--", lw=1.0, label=r"declare-compromised threshold $\Theta$")
    ax.axvline(onset_t, color="#999999", ls=":", lw=1.0, label="attack onset")
    dc = res["detection_cycle_per_observer"].get(VICTIM)
    if dc is not None:
        ax.axvline(dc * DT, color="#2ca02c", ls="-", lw=1.3, label=f"D5 declares D2 compromised (t={dc*DT:.2f}s)")
    ax.set_title(r"Part 2, Sec. 6.4: each drone's local evidence score $E_i(\mathrm{attacker})$")
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$E_i(j{=}\mathrm{attacker})$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "part2_evidence.png"), dpi=150)
    plt.close(fig)


def write_summary(res: dict) -> None:
    lines = []
    lines.append("=== Baseline outcome (no mitigation, Sec. 3 scope) ===")
    if res["collision_cycle"] is not None:
        lines.append(f"Collision threshold ({D_MIN} m) crossed at cycle {res['collision_cycle']} "
                      f"(t = {res['collision_cycle']*DT:.2f}s)")
    else:
        lines.append(f"Collision threshold ({D_MIN} m) never crossed in {CYCLES} cycles "
                      f"(min separation reached: {res['true_min_sep_series'].min():.2f} m)")
    lines.append("")
    lines.append("=== Detection outcome (Sec. 6) ===")
    for i, c in res["detection_cycle_per_observer"].items():
        label = f"D{i+1}"
        if c is None:
            lines.append(f"  {label}: never declared D{ATTACKER+1} compromised in {CYCLES} cycles")
        else:
            lines.append(f"  {label}: declared D{ATTACKER+1} compromised at cycle {c} (t = {c*DT:.2f}s)")
    lines.append("")
    lines.append(f"Attack: D{ATTACKER+1} -> D{VICTIM+1}, {ATTACK_DIRECTION} by {ATTACK_MAGNITUDE_M} m, "
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
